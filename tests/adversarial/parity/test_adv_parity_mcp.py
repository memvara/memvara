"""A model reads the same text from the MCP server, however the server is reached.

One session of tool calls runs through three surfaces, each with a store of its own:

* the in-process server: `MemvaraMCPServer.handle_line`, built from the environment the
  way `memvara.server.cli.main` builds it;
* the stdio server in local mode, a real child process over a real pipe
  (`harness.stdio.McpProcess`);
* the stdio server in cloud mode, pointed at `FakeV1` served on 127.0.0.1.

`compare.normalise_text` labels ids by the text they name, replaces the confirm token,
and replaces the instants each run took from its clock. After that the in-process server
and the local stdio server must write exactly the same text for every step, because
nothing but the transport separates them.

**Cloud mode differs from them only where the code documents that it does.** Each such
difference has a test of its own, which names the documentation and checks that the
difference is real:

* `memory_stats` has a `storage:` line only for a local store (`memvara/server/mcp.py`,
  `_storage_fact`);
* `memory_standing` ends with a "more not shown" line only for a local store, because
  `GET /v1/standing` reports no total (`memvara/server/tools.py`, `_standing`);
* `memory_recall` refuses `budget` and `valid_at` in cloud mode
  (`memvara/server/memory_api.py`, `MemoryAPI.recall`); memvara/memvara#298 tracks
  giving the hosted recall a time axis.

**One difference is a known bug, and it is pinned where the session meets it.** A write
receipt read through the hosted client drops four lists the local receipt reports, so in
cloud mode `memory_remember` leaves out the notes about a value added beside live ones, a
weaker value kept beside a stronger one, a value closed at the instant it began, a fact
re-filed under a memory type the caller asserted, and `procedural` refused for a subject
other than the user (memvara/memvara#334, registered as B52). The session makes the
writes that produce those notes, and their comparison in cloud mode is a strict expected
failure that raises `known_bugs.Reproduced` only when the missing notes are the whole
difference; any other line that differs still fails the run.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import pathlib
import re
from datetime import timedelta
from typing import Any, Callable, Iterator, Mapping, Sequence

import pytest

from harness import known_bugs
from harness.env import child_env
from harness.fakes.fake_v1 import FakeV1
from harness.stdio import McpProcess, ToolResult, kill_all
from memvara.server.config import ServerConfig, build_memvara
from memvara.server.mcp import MemvaraMCPServer
from memvara.types import utcnow

from .compare import ADDED, assert_same, normalise_text, text_labels, timed

SURFACES = ("in-process", "stdio local", "stdio cloud")
QUESTION = "where does the user live"
#: A day while the user lived in Berlin.
IN_BERLIN = "2024-01-31T00:00:00Z"
MISSING = "cl_00000000000000000000"
#: When the expiring fact is erased: 30 days after this module is imported, the same
#: instant on every surface.
EXPIRES = ((utcnow() + timedelta(days=30)).replace(microsecond=0)
           .isoformat().replace("+00:00", "Z"))


class InProcessServer:
    """`MemvaraMCPServer`, built from `env` the way `cli.main` builds it, and driven one
    JSON-RPC line at a time through `handle_line`. It answers the same four calls a test
    makes on `McpProcess`."""

    def __init__(self, env: Mapping[str, str]) -> None:
        config = ServerConfig.from_env(env)
        self.server = MemvaraMCPServer(
            build_memvara(config), read_only=config.read_only, anchored=config.anchored,
            features_off=config.features_off, **config.scope_kwargs)
        self._next_id = 0

    def request(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        self._next_id += 1
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": self._next_id, "method": method}
        if params is not None:
            message["params"] = dict(params)
        reply = self.server.handle_line(json.dumps(message))
        assert reply is not None, f"no reply to {method}"
        return json.loads(reply)["result"]

    def initialize(self) -> Any:
        result = self.request("initialize", {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "memvara-adversarial-suite", "version": "0"}})
        assert self.server.handle_line(
            '{"jsonrpc": "2.0", "method": "notifications/initialized"}') is None
        return result

    def list_tools(self) -> list[Any]:
        return list(self.request("tools/list")["tools"])

    def call(self, name: str, /, **arguments: Any) -> ToolResult:
        result = self.request("tools/call", {"name": name, "arguments": arguments})
        text = "".join(str(block.get("text", "")) for block in result.get("content", []))
        return ToolResult(text=text, is_error=bool(result.get("isError")), raw=result)

    def kill(self) -> None:
        self.server.close()


@dataclasses.dataclass(frozen=True)
class Call:
    """One tool call of the session.

    `arguments` is the call's arguments, or a function that builds them from the text
    every earlier step returned, by step name, so a step can use an id an earlier reply
    named.
    """

    name: str
    tool: str
    arguments: Mapping[str, Any] | Callable[[dict[str, str]], Mapping[str, Any]]


#: What a step uses when the earlier reply it reads names no such id or token: ones no
#: store issued. The step then still runs, and the comparison of the reply that lacked
#: the id, and of the step that needed it, each show what differs.
NO_CLAIM = "cl_ffffffffffffffffffff"
NO_TURN = "ep_ffffffffffffffffffff"
NO_TOKEN = "no-token-in-the-preview"


def _claim(got: dict[str, str], step: str) -> str:
    """The id of the claim a write step added, from its receipt's `+ [<id>]` line."""
    found = ADDED.search(got.get(step, ""))
    return found.group(1) if found else NO_CLAIM


def _turn(got: dict[str, str]) -> str:
    """The id of the turn `memory_add` stored, from its receipt's `turn id(s):` line."""
    found = re.search(r"turn id\(s\): (ep_[0-9a-f]{20})", got.get("add", ""))
    return found.group(1) if found else NO_TURN


def _token(got: dict[str, str], preview: str = "forget_matching.preview") -> str:
    """The confirm token the reply to an earlier preview step ends with."""
    found = re.search(r"^confirm: (\S+)$", got.get(preview, ""), re.MULTILINE)
    return found.group(1) if found else NO_TOKEN


def _fact(predicate: str, value: str, **more: Any) -> dict[str, Any]:
    return {"subject": "user", "predicate": predicate, "object": value, **more}


SESSION: tuple[Call, ...] = (
    Call("remember", "memory_remember",
         _fact("lives_in", "Berlin", true_since="2024-01-01T00:00:00Z")),
    Call("remember.replacing", "memory_remember",
         _fact("lives_in", "Lisbon", true_since="2025-01-01T00:00:00Z")),
    Call("remember.rule", "memory_remember",
         _fact("prefers", "short answers", memory_type="procedural")),
    Call("remember.second_rule", "memory_remember",
         _fact("prefers", "tabs over spaces", memory_type="procedural")),
    Call("remember.finished", "memory_remember",
         _fact("located_now", "Porto", true_since="2025-06-01T00:00:00Z",
               true_until="2025-09-01T00:00:00Z", until_reason="the trip ended")),
    Call("remember.expiring", "memory_remember",
         _fact("goal", "run a marathon", expires_at=EXPIRES,
               expire_reason="a goal for this season")),
    Call("remember.employer", "memory_remember",
         _fact("works_at", "Acme", true_since="2025-01-01T00:00:00Z")),
    Call("remember.taste", "memory_remember", _fact("likes", "jazz")),
    # Four pairs of writes whose second write gets a note of its own: a value added beside
    # a live one in a slot with no cardinality, a weaker value stored beside a stronger
    # one, a value closed at the instant it began, and a fact re-filed under another
    # memory type. See `MISSING_NOTES`.
    Call("remember.tagged", "memory_remember", _fact("tagged_with", "gardening")),
    Call("remember.beside", "memory_remember", _fact("tagged_with", "chess")),
    Call("remember.timezone", "memory_remember",
         _fact("timezone", "Europe/Lisbon", confidence=1.0)),
    Call("remember.disputed", "memory_remember",
         _fact("timezone", "Europe/Berlin", confidence=0.1)),
    Call("remember.title", "memory_remember",
         _fact("job_title", "engineer", true_since="2025-06-01T00:00:00Z")),
    Call("remember.same_start", "memory_remember",
         _fact("job_title", "manager", true_since="2025-06-01T00:00:00Z")),
    Call("remember.refiled", "memory_remember",
         _fact("likes", "jazz", memory_type="episodic")),
    # Procedural is for the user, so the store files this one as semantic and says so.
    Call("remember.not_procedural", "memory_remember",
         {"subject": "ci-server", "predicate": "prefers", "object": "fast builds",
          "memory_type": "procedural"}),
    Call("add", "memory_add", {"text": "My name is Ada."}),
    Call("remember.cited", "memory_remember",
         lambda got: _fact("speaks", "Portuguese", sources=[_turn(got)])),
    Call("search", "memory_search", {"query": QUESTION}),
    Call("search.past", "memory_search", {"query": QUESTION, "valid_at": IN_BERLIN}),
    Call("recall", "memory_recall", {"query": QUESTION}),
    Call("recall.budget", "memory_recall", {"query": QUESTION, "budget": 12}),
    Call("recall.past", "memory_recall", {"query": QUESTION, "valid_at": IN_BERLIN}),
    Call("history", "memory_history", {"subject": "user", "predicate": "lives_in"}),
    Call("why", "memory_why",
         lambda got: {"claim_id": _claim(got, "remember.replacing")}),
    Call("why.cited", "memory_why", lambda got: {"claim_id": _claim(got, "remember.cited")}),
    Call("why.missing", "memory_why", {"claim_id": MISSING}),
    Call("stats", "memory_stats", {}),
    Call("standing", "memory_standing", {}),
    Call("standing.one", "memory_standing", {"k": 1}),
    Call("profile", "memory_profile", {"query": QUESTION}),
    Call("forget_matching.preview", "memory_forget_matching", {"query": "jazz", "k": 1}),
    Call("forget_matching.other_closure", "memory_end_matching",
         lambda got: {"confirm": _token(got), "k": 1}),
    Call("forget_matching.confirm", "memory_forget_matching",
         lambda got: {"confirm": _token(got), "k": 1}),
    Call("forget_matching.replayed", "memory_forget_matching",
         lambda got: {"confirm": _token(got), "k": 1}),
    Call("end_matching.preview", "memory_end_matching", {"query": "marathon", "k": 1}),
    Call("end_matching.confirm", "memory_end_matching",
         lambda got: {"confirm": _token(got, "end_matching.preview"), "k": 1}),
    Call("forget.claim", "memory_forget",
         lambda got: {"claim_id": _claim(got, "remember.replacing"),
                      "reason": "it was Porto"}),
    Call("forget.missing", "memory_forget", {"claim_id": MISSING}),
    Call("end.slot", "memory_end",
         {"subject": "user", "predicate": "works_at", "at": "2025-10-01T00:00:00Z"}),
    Call("end.claim", "memory_end", lambda got: {"claim_id": _claim(got, "remember.cited")}),
    Call("forget.slot", "memory_forget", {"subject": "user", "predicate": "prefers"}),
    Call("document.add", "memory_add_document",
         {"content": "A runbook. Restart the service.", "custom_id": "docs/runbook",
          "title": "Runbook", "metadata": {"team": "ops"}}),
    Call("document.get", "memory_get_document", {"id": "docs/runbook"}),
    Call("document.get.missing", "memory_get_document", {"id": "docs/none"}),
    Call("document.list", "memory_list_documents", {}),
    Call("document.delete", "memory_delete_document", {"id": "docs/runbook"}),
    Call("document.delete.again", "memory_delete_document", {"id": "docs/runbook"}),
    Call("stats.after", "memory_stats", {}),
)


@dataclasses.dataclass(frozen=True)
class Played:
    """What every surface answered, played once per module."""

    #: Each surface's answers to `initialize` and `tools/list`, as parsed JSON.
    handshake: dict[str, dict[str, Any]]
    #: Each surface's error flag and normalised text for every call, by step name.
    replies: dict[str, dict[str, tuple[bool, str]]]


def converse(server: Any) -> tuple[dict[str, Any], dict[str, tuple[bool, str]]]:
    """Run the session on `server`: the handshake a client opens with, then every call.

    Returns the answers to `initialize` and `tools/list`, and every call's error flag and
    text by step name.
    """
    handshake = {"initialize": server.initialize(), "tools/list": server.list_tools()}
    replies: dict[str, tuple[bool, str]] = {}
    got: dict[str, str] = {}
    for call in SESSION:
        arguments = call.arguments(got) if callable(call.arguments) else call.arguments
        result = server.call(call.tool, **arguments)
        got[call.name] = result.text
        replies[call.name] = (result.is_error, result.text)
    return handshake, replies


@contextlib.contextmanager
def serving(root: pathlib.Path,
            surfaces: Sequence[str] = SURFACES) -> Iterator[dict[str, Any]]:
    """The named surfaces, by name, each over a store of its own under `root`, and all
    stopped when the block ends.

    The in-process server and the local stdio server open a store file each. The cloud
    server talks to a `FakeV1` of its own on 127.0.0.1, and every server is stopped
    before that fake closes, so nothing the fake sends meets a closed server.
    """
    home = root / "home"
    home.mkdir()
    started: list[Any] = []
    with FakeV1() as fake:
        try:
            for surface in surfaces:
                if surface == "in-process":
                    server: Any = InProcessServer(child_env(home, {
                        "MEMVARA_DB": str(root / "in-process.db"), "MEMVARA_USER": "alice"}))
                elif surface == "stdio local":
                    server = McpProcess(root / "stdio-local.db", home=home, user="alice")
                else:
                    # MEMVARA_DB is set by McpProcess and ignored in cloud mode, which
                    # makes no file.
                    server = McpProcess(root / "unused.db", home=home, user="alice",
                                        env={"MEMVARA_MODE": "cloud",
                                             "MEMVARA_API_KEY": fake.api_key,
                                             "MEMVARA_SERVER_URL": fake.serve()})
                started.append(server)
            yield dict(zip(surfaces, started))
        finally:
            kill_all(started)


@pytest.fixture(scope="module")
def played(tmp_path_factory: pytest.TempPathFactory) -> Played:
    """Every surface's answers to the session, each surface over a store of its own."""
    def play_all() -> dict[str, tuple[dict[str, Any], dict[str, tuple[bool, str]]]]:
        with serving(tmp_path_factory.mktemp("parity-mcp")) as servers:
            return {surface: converse(server) for surface, server in servers.items()}

    raw, run = timed(play_all)
    replies: dict[str, dict[str, tuple[bool, str]]] = {}
    for surface, (_handshake, texts) in raw.items():
        names = text_labels([text for _error, text in texts.values()])
        replies[surface] = {step: (error, normalise_text(text, names, run=run))
                            for step, (error, text) in texts.items()}
    return Played(handshake={surface: shake for surface, (shake, _texts) in raw.items()},
                  replies=replies)


class _Forgetful:
    """A server whose every reply names no id, as a surface that dropped them would."""

    def initialize(self) -> dict[str, Any]:
        return {}

    def list_tools(self) -> list[Any]:
        return []

    def call(self, name: str, /, **arguments: Any) -> ToolResult:
        return ToolResult(text=f"{name} answered", is_error=False, raw={})


def test_a_reply_that_lacks_an_id_does_not_stop_the_session() -> None:
    """A step that needs an id from an earlier reply still runs, with an id no store
    holds, so its own comparison shows what differs. Otherwise one missing id would stop
    the session and fail every test, without saying which surface or step it was."""
    _handshake, replies = converse(_Forgetful())
    assert list(replies) == [call.name for call in SESSION]


@pytest.mark.parametrize("surface", SURFACES[1:])
@pytest.mark.parametrize("answer", ["initialize", "tools/list"])
def test_every_surface_answers_the_handshake_as_the_in_process_server_does(
        played: Played, answer: str, surface: str) -> None:
    assert_same(played.handshake["in-process"][answer], played.handshake[surface][answer],
                f"{answer} through {surface}")


# -- what cloud mode documents it writes differently -----------------------------------


@dataclasses.dataclass(frozen=True)
class LocalLine:
    """A line the code documents as written only when the server has a local store."""

    #: The steps whose reply carries the line.
    steps: tuple[str, ...]
    #: How the line begins.
    start: str
    #: Where the code says so.
    documented: str


LOCAL_LINES = (
    LocalLine(("stats", "stats.after"), "storage: ",
              "memvara/server/mcp.py, _storage_fact: None for anything without a local "
              "SQLite store, because a hosted deployment's disks are its operator's to "
              "encrypt"),
    LocalLine(("standing.one",), "(1 more not shown",
              "memvara/server/tools.py, _standing: GET /v1/standing caps at k and reports "
              "no total, so against a hosted deployment the hint cannot fire"),
)

#: The steps whose cloud reply a documented rule of its own describes, each checked by its
#: own test below rather than by the step-by-step comparison.
CLOUD_BY_OWN_TEST = frozenset({"recall.budget", "recall.past"})


def without_local_lines(step: str, text: str) -> str:
    """`text` with every line `LOCAL_LINES` documents for `step` removed."""
    starts = tuple(line.start for line in LOCAL_LINES if step in line.steps)
    if not starts:
        return text
    return "\n".join(row for row in text.split("\n") if not row.startswith(starts))


# -- the known difference: memvara/memvara#334 --------------------------------------------

#: The notes `memory_remember` writes from the four receipt lists a hosted receipt leaves
#: empty (memvara/memvara#334, registered as B52), so cloud mode leaves them out.
MISSING_NOTES = re.compile(
    r"^note: \d+ (?:value\(s\) landed in a slot that already had live values"
    r"|value\(s\) were stored without replacing what was already there"
    r"|value\(s\) were closed at the instant they began"
    r"|already-known fact\(s\) were re-filed|fact\(s\) arrived as procedural)")

#: The steps whose local reply carries one of those notes.
NOTE_STEPS = frozenset({"remember.beside", "remember.disputed", "remember.same_start",
                        "remember.refiled", "remember.not_procedural"})


def reply_is_known_334(local: list[str], cloud: list[str]) -> bool:
    """Whether cloud mode's reply differs from the local one only by #334: one or more
    lines `MISSING_NOTES` matches are missing, and every other line is the same."""
    kept = [row for row in local if not MISSING_NOTES.match(row)]
    return len(kept) < len(local) and cloud == kept


def test_the_pin_for_334_absorbs_only_its_own_symptom() -> None:
    """A strict expected failure absorbs whatever its test reports as the known bug. So a
    reply that differs in anything besides the missing notes must fail as a new bug."""
    note = "note: 1 value(s) were closed at the instant they began, so they are now ..."
    local = ["added 1, ended 1, retired 0", "+ [<id>] user job title manager", note]
    assert reply_is_known_334(local, local[:2])
    assert not reply_is_known_334(local, local)
    assert not reply_is_known_334(local, local[:1])
    assert not reply_is_known_334(local, [*local[:2], "note: something else"])
    assert not reply_is_known_334(local[:2], local[:2])


def _compared() -> Iterator[Any]:
    for call in SESSION:
        for surface in SURFACES[1:]:
            if surface == "stdio cloud" and call.name in CLOUD_BY_OWN_TEST:
                continue
            known = surface == "stdio cloud" and call.name in NOTE_STEPS
            yield pytest.param(call.name, surface, id=f"{call.name}-{surface}",
                               marks=[known_bugs.xfail("B52")] if known else [])


# Every tool the session calls, whose reply this test compares on all three surfaces, and
# the three variables a cloud-mode server reads to reach the fake: if any were ignored,
# the cloud replies would differ. `test_the_session_calls_every_tool_it_claims_to_cover`
# keeps the tool list honest.
@pytest.mark.covers(
    "tool:memory_remember", "tool:memory_add", "tool:memory_search", "tool:memory_recall",
    "tool:memory_history", "tool:memory_why", "tool:memory_stats", "tool:memory_standing",
    "tool:memory_profile", "tool:memory_forget_matching", "tool:memory_end_matching",
    "tool:memory_forget", "tool:memory_end", "tool:memory_add_document",
    "tool:memory_get_document", "tool:memory_list_documents", "tool:memory_delete_document",
    "env:MEMVARA_MODE", "env:MEMVARA_API_KEY", "env:MEMVARA_SERVER_URL")
@pytest.mark.parametrize(("step", "surface"), list(_compared()))
def test_every_surface_writes_what_the_in_process_server_writes(
        played: Played, step: str, surface: str) -> None:
    error, text = played.replies["in-process"][step]
    if surface == "stdio cloud":
        text = without_local_lines(step, text)
    actual_error, actual_text = played.replies[surface][step]
    local, cloud = text.split("\n"), actual_text.split("\n")
    if (surface == "stdio cloud" and step in NOTE_STEPS and error == actual_error
            and reply_is_known_334(local, cloud)):
        missing = [row[:60] for row in local if MISSING_NOTES.match(row)]
        raise known_bugs.Reproduced(f"B52: cloud mode's {step} leaves out {missing}")
    assert_same({"error": error, "lines": local}, {"error": actual_error, "lines": cloud},
                f"{step} through {surface}")


def test_the_session_calls_every_tool_it_claims_to_cover() -> None:
    """The checklist reads the covers mark above from this file's source, without
    running anything, so the mark could outlive a change to `SESSION`. The tools it
    names must be exactly the tools the session calls."""
    marks = getattr(test_every_surface_writes_what_the_in_process_server_writes,
                    "pytestmark")
    declared = {item.split(":", 1)[1] for mark in marks if mark.name == "covers"
                for item in mark.args if item.startswith("tool:")}
    assert declared == {call.tool for call in SESSION}


@pytest.mark.parametrize("line", LOCAL_LINES, ids=lambda line: line.start.strip(" ("))
def test_a_line_documented_as_local_is_written_locally_and_not_in_cloud_mode(
        played: Played, line: LocalLine) -> None:
    for step in line.steps:
        for surface in SURFACES:
            rows = played.replies[surface][step][1].split("\n")
            found = [row for row in rows if row.startswith(line.start)]
            if surface == "stdio cloud":
                assert found == [], f"{step} in cloud mode: {line.documented}"
            else:
                assert len(found) == 1, f"{step} through {surface}: {rows}"


def test_cloud_mode_refuses_a_dated_recall(played: Played) -> None:
    """`memvara/server/memory_api.py`, `MemoryAPI.recall`: `valid_at` is refused against a
    hosted deployment, because `POST /v1/recall` has no time axis and a dated read that
    silently answered with the present would be a wrong prompt. memvara/memvara#298
    tracks giving it one; when that lands, this test fails, and the dated recall joins
    the step-by-step comparison."""
    for surface in ("in-process", "stdio local"):
        error, text = played.replies[surface]["recall.past"]
        assert not error and "as things were on 31 January 2024" in text, (surface, text)
    error, text = played.replies["stdio cloud"]["recall.past"]
    assert error
    assert text.startswith("memory_recall failed: ValueError: recall(valid_at=...) is not "
                           "available against a hosted deployment"), text


def test_cloud_mode_refuses_a_recall_budget(played: Played) -> None:
    """`memvara/server/memory_api.py`, `MemoryAPI.recall`: `ScopedRemoteMemvara.recall`
    raises for any budget other than None, because `POST /v1/recall` renders the block
    on the server and takes no budget."""
    for surface in ("in-process", "stdio local"):
        error, text = played.replies[surface]["recall.budget"]
        assert not error and "did not fit" in text, (surface, text)
    error, text = played.replies["stdio cloud"]["recall.budget"]
    assert error
    assert text.startswith("memory_recall failed: ValueError: recall(budget=...) is not "
                           "available against a hosted deployment"), text
