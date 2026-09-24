"""Agentic capture: the headless agent command searches the store, the hook applies proposals.

`plugin/hooks/lib/agentic.py` gives the headless agent command read-only access to the
user's memory for one run, reads the proposals it returns, checks each one, and applies
the ones that pass through the hook's own write paths. These tests never run a model. They
check the command line the hook builds, and they replace the process with a fake that
prints the event stream a real run prints, so each branch can be driven on purpose.

The failures these tests are named for are the ones the module exists to prevent: a write
tool reachable from the run, a claim id the model made up, the extractor's own rules
stored as memories (the Supermemory defect), a fact taken from turns that were already
mined, a run that goes on searching, and a failure nobody hears about.
"""

from __future__ import annotations

import json
import os
import pathlib
import stat
import sys
import threading
from datetime import datetime, timedelta, timezone

import pytest

HOOKS = pathlib.Path(__file__).resolve().parent.parent / "plugin" / "hooks"
if str(HOOKS) not in sys.path:
    sys.path.insert(0, str(HOOKS))

import capture  # noqa: E402
from core.host import CLAUDE_CLI, CLAUDE_MODEL, CODEX_CLI  # noqa: E402
from lib import agentic, counts, extract, hosted, ipc, project, settings, transcript, usage, write  # noqa: E402,E501

from memvara import Memvara  # noqa: E402
from memvara.embed import HashingEmbedder  # noqa: E402
from memvara.llm import NullLLM  # noqa: E402
from memvara.server.config import FEATURE_DEFAULTS as LIBRARY_FEATURE_DEFAULTS  # noqa: E402
from memvara.server.tools import TOOLS  # noqa: E402

SESSION = "0a1b2c3d-0000-4000-8000-00000000a9e7"
ID_A = "cl_" + "a" * 20
ID_B = "cl_" + "b" * 20
ID_C = "cl_" + "c" * 20

TURN = ("User: we moved the deployment off fly.io last week; memvara deploys to Hetzner "
        "now. Always run ruff format before committing, because the CI lint job fails "
        "on unformatted code.\n"
        "Claude used Edit: scripts/deploy.sh\n"
        "Claude: Updated scripts/deploy.sh to push to the Hetzner host.")

RUFF = ("always run ruff format before committing, because the CI lint job fails on "
        "unformatted code and the pull request cannot merge until it passes")


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """Every file a hook writes goes under `tmp_path`; every switch starts at its default."""
    monkeypatch.setattr(write, "LOG", tmp_path / "capture.log")
    monkeypatch.setattr(ipc, "_HOME", str(tmp_path))
    monkeypatch.setattr(ipc, "RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setattr(ipc, "_CLIENT_CONFIGS", ())
    monkeypatch.setattr(usage, "DEFAULT_PATH", tmp_path / "usage.jsonl")
    monkeypatch.setattr(capture, "STATE", tmp_path / "capture-state.json")
    monkeypatch.setattr(settings, "SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setattr(settings, "_LOADED", None)
    monkeypatch.setattr(counts, "COUNTS_DIR", str(tmp_path / "counts"))
    monkeypatch.setattr(project, "CACHE_DIR", str(tmp_path / "projects"))
    monkeypatch.setattr(hosted, "CREDENTIALS", str(tmp_path / "credentials.json"))
    for name in list(os.environ):
        if name.startswith("MEMVARA_FEATURE_") or name in ("MEMVARA_API_KEY",
                                                           "MEMVARA_SERVER_URL"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("MEMVARA_FEATURE_PROJECT_SCOPE", "0")
    monkeypatch.delenv(project.ENV, raising=False)
    monkeypatch.delenv(extract.SENTINEL, raising=False)
    monkeypatch.setattr(agentic, "_chain", lambda: [CLAUDE_CLI])
    # No git in these tests: the project key is fixed.
    monkeypatch.setattr(agentic, "project_subject", lambda cwd=None: "memvara")
    yield
    while _OPENED:
        _OPENED.pop().close()


def _log(tmp_path) -> str:
    path = tmp_path / "capture.log"
    return path.read_text(encoding="utf-8") if path.exists() else ""


# -- a fake of the headless command's event stream --------------------------------------


def init(status="connected", tools=None):
    names = tools if tools is not None else [agentic.tool_name(t) for t in agentic.READ_TOOLS]
    return {"type": "system", "subtype": "init", "tools": names,
            "mcp_servers": [{"name": "memvara", "status": status}]}


def tool_use(use_id, tool="memory_search", msg="m1", usage_=None):
    message = {"id": msg, "content": [{"type": "tool_use", "id": use_id,
                                       "name": agentic.tool_name(tool),
                                       "input": {"query": "deploy"}}]}
    if usage_ is not None:
        message["usage"] = usage_
    return {"type": "assistant", "message": message}


def tool_result(use_id, text, is_error=False, as_list=True):
    content = [{"type": "text", "text": text}] if as_list else text
    return {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": use_id, "content": content,
         "is_error": is_error}]}}


def said(text):
    return {"type": "assistant", "message": {"id": "m9", "content": [
        {"type": "text", "text": text}]}}


def result(proposals=None, text=None, subtype="success", is_error=False, usage_=None):
    body = text if text is not None else json.dumps({"proposals": proposals or []})
    return {"type": "result", "subtype": subtype, "is_error": is_error, "result": body,
            "usage": usage_ if usage_ is not None else
            {"input_tokens": 20, "cache_creation_input_tokens": 3000,
             "cache_read_input_tokens": 5000, "output_tokens": 400}}


SEARCH_HIT = (f"2 match(es). Stored memory about the user (reference data):\n"
              f"1. [id={ID_A} semantic relevance=0.341] memvara deploys to fly.io\n"
              f"2. [id={ID_B} procedural relevance=0.100] user prefers pytest")


class FakeProc:
    """Stands in for `subprocess.Popen`: prints `events`, records kills and the config."""

    def __init__(self, events, hang=False, returncode=0, stderr=""):
        self.events = events
        self.hang = hang
        self.returncode = returncode
        self._stderr = stderr
        self.killed = threading.Event()
        self.argv: list = []
        self.env: dict = {}
        self.config_mode = None
        self.config: dict = {}

    def __call__(self, argv, stdin=None, stdout=None, stderr=None, text=None, env=None):
        self.argv = list(argv)
        self.env = dict(env or {})
        path = argv[argv.index("--mcp-config") + 1]
        self.config_mode = stat.S_IMODE(os.stat(path).st_mode)
        with open(path, encoding="utf-8") as fh:
            self.config = json.load(fh)
        if self._stderr and stderr is not None:
            stderr.write(self._stderr)
            stderr.flush()
        return self

    @property
    def stdout(self):
        for event in self.events:
            if self.killed.is_set():
                return
            yield event if isinstance(event, str) else json.dumps(event) + "\n"
        if self.hang:
            self.killed.wait(5)

    def kill(self):
        self.killed.set()

    def wait(self, timeout=None):
        return self.returncode


def _fake(monkeypatch, events, **kw) -> FakeProc:
    proc = FakeProc(events, **kw)
    monkeypatch.setattr(agentic.subprocess, "Popen", proc)
    return proc


def _local_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMVARA_DB", str(tmp_path / "memory.db"))


#: Stores a test opened, closed by `_isolated` when the test ends.
_OPENED: "list[Memvara]" = []


def _mem() -> Memvara:
    mem = Memvara(":memory:", embedder=HashingEmbedder(dim=64), llm=NullLLM())
    _OPENED.append(mem)
    return mem


class Receipt:
    def __init__(self, claim_id):
        self.added = [type("C", (), {"id": claim_id})()]
        self.reinforced = []


class LocalStore:
    """A local store as far as the hook can tell: `remember` has these keyword names."""

    def __init__(self):
        self.remembered: list = []
        self.deleted: list = []
        self.linked: list = []
        self._n = 0

    def remember(self, subject, predicate, obj, *, confidence=1.0, memory_type=None,
                 extractor="api", sources=None, replaces=None, reason=None):
        self.remembered.append({"subject": subject, "predicate": predicate, "object": obj,
                                "memory_type": memory_type, "replaces": replaces,
                                "reason": reason})
        self._n += 1
        return Receipt("cl_" + f"{self._n:020x}")

    def delete(self, claim_id, *, close="retired", reason=None):
        self.deleted.append((claim_id, close, reason))
        return claim_id != ID_C

    def link(self, from_id, to_id, relation, *, by="api"):
        self.linked.append((from_id, to_id, relation, by))


class ExpiringStore(LocalStore):
    def remember(self, subject, predicate, obj, *, confidence=1.0, memory_type=None,
                 extractor="api", sources=None, replaces=None, reason=None,
                 expires_at=None):
        out = super().remember(subject, predicate, obj, confidence=confidence,
                               memory_type=memory_type, extractor=extractor,
                               sources=sources, replaces=replaces, reason=reason)
        self.remembered[-1]["expires_at"] = expires_at
        return out


# -- the command line --------------------------------------------------------------------


def test_the_command_line_is_exactly_the_documented_one():
    """Every flag here is a guard, and `agentic.argv` says why each one is there. A flag
    dropped by a refactor would widen what the run can do without any test noticing."""
    got = agentic.argv("RULES", "/run/cfg.json", "DATA")
    assert got == [
        "claude", "-p",
        "--settings", '{"hooks":{}}',
        "--setting-sources", "",
        "--model", CLAUDE_MODEL,
        "--output-format", "stream-json", "--verbose",
        "--no-session-persistence",
        "--tools", "",
        "--mcp-config", "/run/cfg.json", "--strict-mcp-config",
        "--allowedTools", "mcp__memvara__memory_search,mcp__memvara__memory_recall,"
                          "mcp__memvara__memory_why,mcp__memvara__memory_profile",
        "--disallowedTools", ",".join(f"mcp__memvara__{t}" for t in agentic.HIDDEN_TOOLS),
        "--permission-mode", "dontAsk",
        "--max-turns", str(agentic.MAX_STEPS),
        "--system-prompt", "RULES",
        "DATA",
    ]
    assert agentic.MAX_SEARCHES == 4 and agentic.MAX_STEPS == 6


def test_no_write_tool_the_server_lists_is_allowed_and_each_is_denied_by_name():
    """The server's own tool table decides what is a write. Every tool it lists that is
    not one of the four reads must be named in `HIDDEN_TOOLS`, so a tool added to the
    server shows up here as a failure rather than in the model's context. `dontAsk`
    would still refuse it, but only this list keeps its description out of the tokens."""
    served = {tool.name for tool in TOOLS}
    reads = set(agentic.READ_TOOLS)
    assert reads <= served
    assert not any(tool.writes for tool in TOOLS if tool.name in reads)
    assert served - reads == set(agentic.HIDDEN_TOOLS)


def test_the_data_follows_a_flag_that_takes_exactly_one_value():
    """`--allowedTools`, `--disallowedTools` and `--mcp-config` take lists. Data placed after
    one of them would be read as one more tool name, and the run would get no prompt."""
    got = agentic.argv("RULES", "/cfg", "DATA")
    assert got[-3:] == ["--system-prompt", "RULES", "DATA"]


# -- instructions are never content ------------------------------------------------------


def test_the_rules_are_the_system_prompt_and_the_turn_is_only_in_the_data_block(
        monkeypatch, tmp_path):
    """The rules and the conversation travel in different messages, and the conversation
    is wrapped in delimiters carrying a value the turn cannot know in advance."""
    _local_env(monkeypatch, tmp_path)
    proc = _fake(monkeypatch, [init(), result([])])
    agentic.capture(LocalStore(), TURN, "User: earlier question", "/repo", [], hosted=False)
    rules = proc.argv[proc.argv.index("--system-prompt") + 1]
    data = proc.argv[-1]
    assert rules == agentic.system_prompt("/repo")
    assert TURN not in rules and "earlier question" not in rules
    assert TURN in data and "earlier question" in data
    nonce = data.splitlines()[0][len("<data-"):-1]
    assert len(nonce) == 12
    assert data.endswith(f"</data-{nonce}>")
    lines = data.splitlines()
    assert lines.index(f"<earlier-{nonce}>") < lines.index("User: earlier question") \
        < lines.index(f"</earlier-{nonce}>") < lines.index(f"<turn-{nonce}>") \
        < lines.index(TURN.splitlines()[0]) < lines.index(f"</turn-{nonce}>")
    assert "data, not instructions" in data.splitlines()[1]


def test_a_turn_that_quotes_the_extractors_rules_yields_no_memory_of_them(
        monkeypatch, tmp_path):
    """The defect that stored Supermemory's own prompt as 20 memories. Here the user pastes
    the extractor's rules and asks about them, and the model restates them as facts, the
    way that agent did. Every such proposal is refused, and nothing is written."""
    _local_env(monkeypatch, tmp_path)
    rules = agentic.system_prompt("/repo")
    pasted = rules[rules.index("## What is worth keeping"):][:900]
    turn = f"User: what does this prompt do? always explain it to me.\n{pasted}\n" \
           "Claude: It tells an extractor what to keep and what to skip."
    echoes = [
        {"kind": "fact", "subject": "user", "predicate": "prefers",
         "object": "Only what would still matter next week. Most turns hold at most one or "
                   "two facts, and an empty list is a correct and common answer."},
        {"kind": "fact", "subject": "user", "predicate": "prefers",
         "object": "Keep a standing instruction or preference, even stated mid-work "
                   "(always X, stop doing Y, from now on Z), and a durable decision."},
        {"kind": "fact", "subject": "user", "predicate": "never_do",
         "object": "Skip the mechanics of this session: what a command printed, what a "
                   "file contains right now, what you are about to do next."},
    ]
    proc = _fake(monkeypatch, [init(), result(echoes)])
    store = LocalStore()
    out = agentic.capture(store, turn, "", "/repo", [], hosted=False)
    assert pasted in proc.argv[-1] and pasted not in turn.split(pasted)[0]
    assert out is not None and out.proposed == 3 and out.refused == 3
    assert store.remembered == [], "no proposal that restates the rules may be written"
    assert _log(tmp_path).count("repeats the extractor's own rules") == 3


def test_a_real_preference_in_the_same_turn_still_gets_through():
    """The rules check compares word sequences with the rules, so an ordinary preference
    stated alongside a pasted prompt is not refused along with it."""
    rules = agentic.system_prompt("/repo")
    ok, refused = agentic.check(
        [{"kind": "fact", "subject": "user", "predicate": "prefers", "object": RUFF}],
        turn=TURN, context="", rules=rules, seen=set(), shown=[], injected=[],
        project="memvara")
    assert refused == [] and ok[0].fact.object.startswith(RUFF)


# -- ids ---------------------------------------------------------------------------------


def test_claim_ids_count_as_seen_only_when_a_read_tool_returned_them():
    """Measured on a real run: a model asked to use a tool it did not have wrote the call
    and its "result" into its own reply. An id in the model's own text is a guess."""
    watch = agentic._Watch()
    lines = [init(), tool_use("t1"), tool_result("t1", SEARCH_HIT),
             tool_use("t2", "memory_why", msg="m2"),
             tool_result("t2", f"why {ID_C}", is_error=True),
             tool_result("t9", f"from nowhere {ID_C}"),
             {"type": "user", "message": {"content": ["text", {"type": "text",
                                                               "text": ID_C}]}},
             said(f"I also saw {ID_C}"), result([])]
    for event in lines:
        watch.feed(json.dumps(event))
    assert watch.seen == {ID_A, ID_B}
    assert any("fly.io" in line for line in watch.shown)


def test_a_tool_result_given_as_plain_text_is_read_too():
    watch = agentic._Watch()
    for event in (init(), tool_use("t1"), tool_result("t1", SEARCH_HIT, as_list=False)):
        watch.feed(json.dumps(event))
    assert watch.seen == {ID_A, ID_B}


def test_a_proposal_naming_an_id_the_model_did_not_see_is_refused_and_logged(
        monkeypatch, tmp_path):
    _local_env(monkeypatch, tmp_path)
    proposals = [
        {"kind": "end", "claim_id": ID_C, "reason": "the job ended"},
        {"kind": "supersede", "claim_id": ID_C, "subject": "memvara",
         "predicate": "deploys_to", "object": "Hetzner", "reason": "moved"},
        {"kind": "link", "from": ID_A, "to": ID_C, "relation": "extends"},
    ]
    _fake(monkeypatch, [init(), tool_use("t1"), tool_result("t1", SEARCH_HIT),
                        result(proposals)])
    store = LocalStore()
    out = agentic.capture(store, TURN, "", "/repo", [], hosted=False)
    assert out.refused == 3 and store.remembered == [] and store.deleted == []
    assert store.linked == []
    assert _log(tmp_path).count("not in any tool result this run") == 3


# -- the run is bounded, and a failure falls back ---------------------------------------


def test_the_run_is_stopped_at_the_fifth_search_and_the_turn_falls_back(
        monkeypatch, tmp_path):
    _local_env(monkeypatch, tmp_path)
    events = [init()]
    for n in range(6):
        events += [tool_use(f"t{n}", msg=f"m{n}",
                            usage_={"input_tokens": 5, "output_tokens": 7}),
                   tool_result(f"t{n}", SEARCH_HIT)]
    events.append(result([]))
    proc = _fake(monkeypatch, events)
    assert agentic.capture(LocalStore(), TURN, "", "/repo", [], hosted=False) is None
    assert proc.killed.is_set()
    assert "fell back to single-call extraction: more than 4 searches" in _log(tmp_path)
    # The five messages it did make are still accounted for.
    assert usage.totals(usage.DEFAULT_PATH) == {"write.tokens_in": 25,
                                                "write.tokens_out": 35}


def test_no_memory_access_falls_back_and_says_so(monkeypatch, tmp_path):
    _local_env(monkeypatch, tmp_path)
    _fake(monkeypatch, [init(status="failed"), result([])])
    assert agentic.capture(LocalStore(), TURN, "", "/repo", [], hosted=False) is None
    assert "no memory access (server failed)" in _log(tmp_path)


def test_a_server_without_search_is_no_memory_access(monkeypatch, tmp_path):
    _local_env(monkeypatch, tmp_path)
    _fake(monkeypatch, [init(tools=[]), result([])])
    assert agentic.capture(LocalStore(), TURN, "", "/repo", [], hosted=False) is None
    assert "memory_search is not offered" in _log(tmp_path)


def test_a_run_that_does_not_answer_in_time_is_killed_and_falls_back(monkeypatch, tmp_path):
    _local_env(monkeypatch, tmp_path)
    monkeypatch.setattr(agentic, "TIMEOUT_SEC", 0.05)
    proc = _fake(monkeypatch, [init()], hang=True)
    assert agentic.capture(LocalStore(), TURN, "", "/repo", [], hosted=False) is None
    assert proc.killed.is_set()
    assert "no reply within 0.05s" in _log(tmp_path)


def test_a_missing_command_falls_back(monkeypatch, tmp_path):
    _local_env(monkeypatch, tmp_path)

    def missing(*a, **k):
        raise FileNotFoundError("claude")

    monkeypatch.setattr(agentic.subprocess, "Popen", missing)
    assert agentic.capture(LocalStore(), TURN, "", "/repo", [], hosted=False) is None
    assert "claude is not installed" in _log(tmp_path)


def test_a_command_that_cannot_start_falls_back(monkeypatch, tmp_path):
    _local_env(monkeypatch, tmp_path)

    def broken(*a, **k):
        raise PermissionError("denied")

    monkeypatch.setattr(agentic.subprocess, "Popen", broken)
    assert agentic.capture(LocalStore(), TURN, "", "/repo", [], hosted=False) is None
    assert "PermissionError: denied" in _log(tmp_path)


@pytest.mark.parametrize("event, why", [
    (result(text="Failed to authenticate", is_error=True, subtype="success"),
     "success: Failed to authenticate"),
    (result(text="", subtype="error_max_turns"), "error_max_turns"),
])
def test_a_run_that_fails_falls_back_with_its_own_reason(monkeypatch, tmp_path, event, why):
    _local_env(monkeypatch, tmp_path)
    _fake(monkeypatch, [init(), event])
    assert agentic.capture(LocalStore(), TURN, "", "/repo", [], hosted=False) is None
    assert f"fell back to single-call extraction: {why}" in _log(tmp_path)


def test_a_run_that_ends_without_a_result_names_what_it_printed(monkeypatch, tmp_path):
    _local_env(monkeypatch, tmp_path)
    _fake(monkeypatch, ["not json\n", "[1]\n", "{broken\n", init()], returncode=2,
          stderr="warming up\nError: bad flag\n")
    assert agentic.capture(LocalStore(), TURN, "", "/repo", [], hosted=False) is None
    assert "exited 2 with no result: Error: bad flag" in _log(tmp_path)


def test_a_run_that_ends_silently_says_so(monkeypatch, tmp_path):
    _local_env(monkeypatch, tmp_path)
    _fake(monkeypatch, [init()], returncode=1)
    assert agentic.capture(LocalStore(), TURN, "", "/repo", [], hosted=False) is None
    assert _log(tmp_path).rstrip().endswith("exited 1 with no result")


def test_a_process_that_will_not_exit_after_its_result_is_killed(monkeypatch, tmp_path):
    _local_env(monkeypatch, tmp_path)
    proc = _fake(monkeypatch, [init(), result([])])
    waits = []

    def wait(timeout=None):
        waits.append(timeout)
        if timeout is not None:
            raise agentic.subprocess.TimeoutExpired("claude", timeout)
        return 0

    proc.wait = wait
    assert agentic.capture(LocalStore(), TURN, "", "/repo", [], hosted=False) is not None
    assert proc.killed.is_set() and waits == [10, None]


def test_a_kill_of_a_process_that_already_exited_is_harmless():
    class Gone:
        def kill(self):
            raise ProcessLookupError

    agentic._kill(Gone())


def test_inside_an_extraction_child_nothing_runs(monkeypatch, tmp_path):
    monkeypatch.setenv(extract.SENTINEL, "1")
    proc = _fake(monkeypatch, [init(), result([])])
    assert agentic.capture(LocalStore(), TURN, "", "/repo", [], hosted=False) is None
    assert proc.argv == [] and _log(tmp_path) == ""


def test_a_host_that_mines_with_its_own_cli_does_not_run_agentic_capture(
        monkeypatch, tmp_path):
    monkeypatch.setattr(agentic, "_chain", lambda: [CODEX_CLI, CLAUDE_CLI])
    assert agentic.available() is False
    assert agentic.capture(LocalStore(), TURN, "", "/repo", [], hosted=False) is None
    assert "first extractor on this host is not claude" in _log(tmp_path)
    monkeypatch.setattr(agentic, "_chain", lambda: [])
    assert agentic.available() is False


def test_a_hosted_install_with_no_login_falls_back(tmp_path):
    assert agentic.mcp_config(hosted=True) is None
    assert agentic.capture(LocalStore(), TURN, "", "/repo", [], hosted=True) is None
    assert "no login for the memory server" in _log(tmp_path)


def test_a_config_that_cannot_be_written_falls_back(monkeypatch, tmp_path):
    _local_env(monkeypatch, tmp_path)

    def fail(config):
        raise PermissionError("read-only")

    monkeypatch.setattr(agentic, "_write_config", fail)
    assert agentic.capture(LocalStore(), TURN, "", "/repo", [], hosted=False) is None
    assert "could not write its config: PermissionError" in _log(tmp_path)


# -- the config the run connects with ----------------------------------------------------


def test_the_hosted_config_is_the_hooks_own_credential_and_project(monkeypatch, tmp_path):
    (tmp_path / "credentials.json").write_text(json.dumps(
        {"api_key": "mv_test_key", "server_url": "https://example.test/"}))
    monkeypatch.setenv(project.ENV, "github.com/memvara/memvara")
    assert agentic.mcp_config(hosted=True) == {"mcpServers": {"memvara": {
        "type": "http", "url": "https://example.test/mcp",
        "headers": {"Authorization": "Bearer mv_test_key",
                    "User-Agent": hosted.USER_AGENT,
                    "memvara-project": "github.com/memvara/memvara"}}}}
    monkeypatch.delenv(project.ENV)
    headers = agentic.mcp_config(hosted=True)["mcpServers"]["memvara"]["headers"]
    assert "memvara-project" not in headers


def test_the_local_config_is_the_clients_own_server_with_this_process_winning(
        monkeypatch, tmp_path):
    config = tmp_path / "claude.json"
    config.write_text(json.dumps({"mcpServers": {
        "other": {"command": "x"},
        "memvara": {"command": "/venv/bin/python", "args": ["-m", "memvara.server"],
                    "env": {"MEMVARA_DB": "/from/config.db", "MEMVARA_LLM": "none"}}}}))
    monkeypatch.setattr(ipc, "_CLIENT_CONFIGS", (str(tmp_path / "missing.json"),
                                                 str(tmp_path / "broken.json"),
                                                 str(tmp_path / "list.json"),
                                                 str(config)))
    (tmp_path / "broken.json").write_text("{not json")
    (tmp_path / "list.json").write_text("[]")
    monkeypatch.setenv("MEMVARA_DB", "/from/env.db")
    monkeypatch.setenv("PYTHONPATH", "/src")
    monkeypatch.setenv("UNRELATED_SECRET", "x")
    got = agentic.mcp_config(hosted=False)["mcpServers"]["memvara"]
    assert got["type"] == "stdio" and got["command"] == "/venv/bin/python"
    assert got["args"] == ["-m", "memvara.server"]
    assert got["env"]["MEMVARA_DB"] == "/from/env.db"
    assert got["env"]["MEMVARA_LLM"] == "none" and got["env"]["PYTHONPATH"] == "/src"
    assert "UNRELATED_SECRET" not in got["env"]
    assert extract.SENTINEL not in got["env"]


def test_without_a_client_block_the_server_is_started_with_this_interpreter(
        monkeypatch, tmp_path):
    _local_env(monkeypatch, tmp_path)
    got = agentic.mcp_config(hosted=False)["mcpServers"]["memvara"]
    assert got["command"] == sys.executable and got["args"] == ["-m", "memvara.server"]


def test_the_config_file_is_owner_only_and_gone_after_the_run(monkeypatch, tmp_path):
    """It holds the API key or the store's environment. Readable by the owner only, in
    the private runtime directory, and removed when the run ends."""
    _local_env(monkeypatch, tmp_path)
    proc = _fake(monkeypatch, [init(), result([])])
    agentic.capture(LocalStore(), TURN, "", "/repo", [], hosted=False)
    if os.name == "posix":
        # Windows has no owner/group/other mode bits to check: `os.stat` reports 0o666
        # for any writable file whatever `os.open` was asked for. There the file's
        # privacy rests on the user's own profile directory, which the rest of this test
        # (the file is gone after the run) still covers.
        assert proc.config_mode == 0o600
    assert proc.config["mcpServers"]["memvara"]["env"]["MEMVARA_DB"].endswith("memory.db")
    assert os.listdir(tmp_path / "run") == []
    assert proc.env[extract.SENTINEL] == "1"


def test_a_config_that_cannot_be_removed_does_not_fail_the_turn(monkeypatch, tmp_path):
    _local_env(monkeypatch, tmp_path)
    _fake(monkeypatch, [init(), result([])])

    real = os.unlink

    def stuck(path):
        if "capture-mcp-" in str(path):
            raise PermissionError(path)
        real(path)

    monkeypatch.setattr(agentic.os, "unlink", stuck)
    assert agentic.capture(LocalStore(), TURN, "", "/repo", [], hosted=False) is not None


# -- replies and checks ------------------------------------------------------------------


@pytest.mark.parametrize("reply", ["", "no idea", "{not json}", '{"facts": []}',
                                   '{"proposals": {}}', "[1, 2]"])
def test_a_reply_that_is_not_a_proposal_list_writes_nothing(monkeypatch, tmp_path, reply):
    _local_env(monkeypatch, tmp_path)
    _fake(monkeypatch, [init(), result(text=reply)])
    store = LocalStore()
    out = agentic.capture(store, TURN, "", "/repo", [], hosted=False)
    assert out is not None and out.proposed == 0 and store.remembered == []
    assert "agentic reply was not a proposal list; nothing written" in _log(tmp_path)


def test_a_fenced_reply_is_read():
    body = "Here:\n```json\n" + json.dumps({"proposals": [{"kind": "end"}]}) + "\n```"
    assert agentic._proposals(body) == [{"kind": "end"}]


def test_a_proposed_fact_passes_the_single_call_checks():
    """An unlisted predicate, a thin object under a full-sentences predicate, a value the
    turn never mentions: each is refused here exactly as `extract.vet` refuses it for the
    single-call extractor."""
    raw = [
        {"kind": "fact", "subject": "user", "predicate": "invented_thing", "object": "x"},
        {"kind": "fact", "subject": "user", "predicate": "prefers", "object": "ruff"},
        {"kind": "fact", "subject": "memvara", "predicate": "version", "object": "9.9.9"},
        "not a dict",
        {"kind": "fact", "subject": "user", "predicate": "prefers", "object": 7},
        {"kind": "delete_everything"},
    ]
    ok, refused = agentic.check(raw, turn=TURN, context="", rules="", seen=set(),
                                shown=[], injected=[], project="memvara")
    assert ok == []
    assert refused == [
        "fact: invented_thing: not in vocabulary",
        "fact: prefers: object too thin (4c) 'ruff'",
        "fact: version: values absent from the turn '9.9.9'",
        "#3: not an object",
        "fact: prefers: object too thin (1c) '7'",
        "#5: unknown kind 'delete_everything'",
    ]


def test_a_fact_that_repeats_a_search_result_is_refused():
    """The model read the store. Handing a stored note back as a new fact is the echo the
    single-call extractor already refuses for recalled notes."""
    note = "user prefers always run ruff format before committing, because the CI lint " \
           "job fails on unformatted code and the pull request cannot merge until it passes"
    ok, refused = agentic.check(
        [{"kind": "fact", "subject": "user", "predicate": "prefers", "object": RUFF}],
        turn="User: fix the build\nClaude used Bash: make", context="", rules="",
        seen=set(), shown=[note], injected=[], project="memvara")
    assert ok == [] and refused[0].startswith("fact: prefers: restates a recalled note")


def test_a_fact_taken_from_the_earlier_turns_is_refused():
    """The earlier turns were mined when they ended. A proposal that repeats them and not
    the new turn would store the same fact a second time."""
    context = f"User: {RUFF}\nClaude: Noted."
    ok, refused = agentic.check(
        [{"kind": "fact", "subject": "user", "predicate": "prefers", "object": RUFF}],
        turn="User: now run the tests please\nClaude used Bash: pytest -q",
        context=context, rules="", seen=set(), shown=[], injected=[], project="memvara")
    assert ok == [] and refused == ["fact prefers: comes from the earlier turns, not "
                                    "this one"]
    ok, _ = agentic.check(
        [{"kind": "fact", "subject": "user", "predicate": "prefers", "object": RUFF}],
        turn=f"User: {RUFF}", context=context, rules="", seen=set(), shown=[],
        injected=[], project="memvara")
    assert len(ok) == 1, "said again in the new turn, it is the new turn's fact"


def test_supersede_end_and_link_need_their_parts():
    seen = {ID_A, ID_B}
    raw = [
        {"kind": "supersede", "claim_id": ID_A, "subject": "memvara",
         "predicate": "deploys_to", "object": "Hetzner", "reason": "  "},
        {"kind": "supersede", "claim_id": ID_A, "subject": "memvara",
         "predicate": "made_up", "object": "Hetzner", "reason": "moved"},
        {"kind": "end", "claim_id": ID_B},
        {"kind": "end"},
        {"kind": "supersede"},
        {"kind": "link", "from": ID_A, "to": ID_B, "relation": "updates"},
        {"kind": "link", "from": ID_A, "to": ID_A, "relation": "extends"},
        {"kind": "link", "from": "", "to": "new:x", "relation": "derives"},
    ]
    ok, refused = agentic.check(raw, turn=TURN, context="", rules="", seen=seen,
                                shown=[], injected=[], project="memvara")
    assert ok == []
    assert refused == [
        f"supersede {ID_A}: no reason",
        "supersede: made_up: not in vocabulary",
        f"end {ID_B}: no reason",
        "end (no id): not in any tool result this run",
        "supersede (no id): not in any tool result this run",
        "link: relation 'updates' is not extends or derives",
        f"link: {ID_A} to itself",
        "link: (empty), new:x not in any tool result this run",
    ]


def test_more_proposals_than_the_limit_are_refused_past_it():
    one = {"kind": "end", "claim_id": ID_A, "reason": "over"}
    ok, refused = agentic.check([one] * 15, turn=TURN, context="", rules="",
                                seen={ID_A}, shown=[], injected=[], project="memvara")
    assert len(ok) == agentic.MAX_PROPOSALS
    assert refused == ["3 more over the limit of 12"]


def test_a_long_reason_is_cut_to_what_the_store_accepts():
    ok, _ = agentic.check([{"kind": "end", "claim_id": ID_A, "reason": "x " * 600}],
                          turn=TURN, context="", rules="", seen={ID_A}, shown=[],
                          injected=[], project="memvara")
    assert len(ok[0].reason) == agentic.REASON_CHARS


def test_standing_false_files_a_fact_as_an_event():
    """Supermemory's static flag, mapped onto `memory_type`: a fact the model marks as not
    standing is filed as episodic. Without the flag the predicate's own type stands."""
    base = {"kind": "fact", "subject": "user", "predicate": "prefers", "object": RUFF}
    ok, _ = agentic.check([dict(base, standing=False), dict(base), dict(base, standing=True)],
                          turn=TURN, context="", rules="", seen=set(), shown=[],
                          injected=[], project="memvara")
    assert [p.fact.memory_type for p in ok] == ["episodic", "procedural", "procedural"]


def test_an_expiry_must_be_a_future_date():
    now = datetime(2026, 9, 24, tzinfo=timezone.utc)
    assert agentic._expiry("2026-10-31", now) == ("2026-10-31T00:00:00+00:00", "")
    assert agentic._expiry("2026-10-31T09:00:00Z", now) == ("2026-10-31T09:00:00+00:00", "")
    assert agentic._expiry(None, now) == ("", "")
    assert agentic._expiry("soon", now)[1] == "expires_at 'soon' is not a date"
    assert agentic._expiry("2020-01-01", now)[1] == "expires_at '2020-01-01' is not in the future"
    base = {"kind": "fact", "subject": "user", "predicate": "prefers", "object": RUFF}
    ok, refused = agentic.check([dict(base, expires_at="last year")], turn=TURN, context="",
                                rules="", seen=set(), shown=[], injected=[],
                                project="memvara", now=now)
    assert ok[0].expires_at == "" and "kept without it" in refused[0]


# -- applying ----------------------------------------------------------------------------


def _proposals(*raw, seen=frozenset({ID_A, ID_B, ID_C}), turn=TURN):
    ok, refused = agentic.check(list(raw), turn=turn, context="", rules="", seen=set(seen),
                                shown=[], injected=[], project="memvara",
                                now=datetime(2026, 9, 24, tzinfo=timezone.utc))
    assert refused == []
    return ok


def test_proposals_apply_to_a_real_local_store():
    """The whole path against the library: the supersede ends the old value by id, the end
    closes a claim as ended (not retired), and a link joins a new fact to a stored one."""
    mem = _mem()
    old = mem.remember("memvara", "deploys_to", "fly.io", confidence=0.7).added[0]
    job = mem.remember("user", "working_on", "the billing migration").added[0]
    props = _proposals(
        {"kind": "supersede", "claim_id": old.id, "subject": "memvara",
         "predicate": "deploys_to", "object": "Hetzner", "reason": "moved off fly.io"},
        {"kind": "fact", "subject": "user", "predicate": "prefers", "object": RUFF},
        {"kind": "end", "claim_id": job.id, "reason": "the migration merged"},
        {"kind": "link", "from": "new:1", "to": "new:0", "relation": "derives"},
        seen={old.id, job.id})
    done = agentic.apply(mem, props, turn=TURN, hosted=False)
    assert done.failed == [] and done.notes == []
    assert (done.stored, done.replaced, done.ended, done.linked) == (2, 1, 1, 1)
    history = {c.object: c for c in mem.history("memvara", "deploys_to")}
    assert history["fly.io"].valid_to is not None and history["fly.io"].invalidated_at is None
    assert history["Hetzner"].valid_to is None
    assert mem.get(job.id).valid_to is not None and mem.get(job.id).invalidated_at is None
    ruff = [c for c in mem.history("user", "prefers")][0]
    assert ruff.memory_type.value == "procedural"
    assert [(link.relation) for link in mem.links(ruff.id)] == ["derives"]


def test_an_end_of_a_claim_already_replaced_is_skipped_and_noted():
    store = LocalStore()
    props = _proposals(
        {"kind": "supersede", "claim_id": ID_A, "subject": "memvara",
         "predicate": "deploys_to", "object": "Hetzner", "reason": "moved"},
        {"kind": "end", "claim_id": ID_A, "reason": "moved"},
        {"kind": "end", "claim_id": ID_C, "reason": "gone"})
    done = agentic.apply(store, props, turn=TURN, hosted=False)
    assert store.deleted == [(ID_C, "ended", "gone")]
    assert done.notes == [f"end {ID_A}: already replaced by a supersede"]
    assert done.failed == [f"end {ID_C}: KeyError: 'no claim {ID_C} is visible here'"]


def test_a_link_to_a_fact_that_was_not_written_is_skipped_and_noted():
    class Refusing(LocalStore):
        def remember(self, *a, **k):
            raise ValueError("refused")

    store = Refusing()
    props = _proposals(
        {"kind": "fact", "subject": "user", "predicate": "prefers", "object": RUFF},
        {"kind": "link", "from": "new:0", "to": ID_A, "relation": "extends"})
    done = agentic.apply(store, props, turn=TURN, hosted=False)
    assert done.failed == ["fact user/prefers: ValueError: refused"]
    assert "a new claim it names was not written" in done.notes[0]
    assert store.linked == []


def test_a_link_the_store_refuses_is_a_failure_and_the_rest_still_apply():
    class NoLinks(LocalStore):
        def link(self, *a, **k):
            raise KeyError("not visible")

    store = NoLinks()
    props = _proposals({"kind": "link", "from": ID_A, "to": ID_B, "relation": "extends"})
    done = agentic.apply(store, props, turn=TURN, hosted=False)
    assert done.linked == 0 and done.failed[0].startswith(f"link {ID_A} extends {ID_B}")


def test_expires_at_reaches_a_store_that_takes_it_and_is_dropped_otherwise():
    fact = {"kind": "fact", "subject": "user", "predicate": "prefers", "object": RUFF,
            "expires_at": "2026-10-31"}
    newer = ExpiringStore()
    done = agentic.apply(newer, _proposals(fact), turn=TURN, hosted=False)
    assert newer.remembered[0]["expires_at"] == "2026-10-31T00:00:00+00:00"
    assert done.notes == []
    older = LocalStore()
    done = agentic.apply(older, _proposals(fact), turn=TURN, hosted=False)
    assert "expires_at" not in older.remembered[0] and older.remembered[0]["object"]
    assert done.notes == ["prefers: expires_at dropped, this store does not take it yet"]


def test_a_store_that_cannot_replace_by_id_gets_a_plain_write_and_a_note():
    class Old:
        def __init__(self):
            self.calls = []

        def remember(self, subject, predicate, obj, *, confidence=1.0, memory_type=None,
                     extractor="api", sources=None):
            self.calls.append((subject, predicate, obj))
            return "added 1"

    store = Old()
    done = agentic.apply(store, _proposals(
        {"kind": "supersede", "claim_id": ID_A, "subject": "memvara",
         "predicate": "deploys_to", "object": "Hetzner", "reason": "moved"}),
        turn=TURN, hosted=False)
    assert store.calls == [("memvara", "deploys_to", "Hetzner")]
    assert (done.stored, done.replaced) == (1, 0)
    assert done.notes == [f"deploys_to: this store cannot replace by id, so {ID_A} was "
                          "left for the reconciler"]


def test_takes_never_raises():
    class Broken:
        remember = 5

        def accepts(self, tool, argument):
            raise RuntimeError("probe failed")

    assert write.takes(Broken(), "replaces", hosted=True) is False
    assert write.takes(object(), "replaces", hosted=True) is False
    assert write.takes(Broken(), "replaces", hosted=False) is False


def test_new_claim_id_reads_both_receipt_shapes():
    assert write.new_claim_id(f"added 1\n+ [{ID_A}] memvara deploys to Hetzner") == ID_A
    assert write.new_claim_id(f"added 0\n- [{ID_A} ended] old") is None
    assert write.new_claim_id("+ [no id here]") is None
    mem = _mem()
    first = mem.remember("user", "timezone", "Europe/Lisbon")
    again = mem.remember("user", "timezone", "Europe/Lisbon")
    assert write.new_claim_id(first) == first.added[0].id
    assert write.new_claim_id(again) == first.added[0].id, "an already-known fact's id"
    assert write.new_claim_id(object()) is None


def test_a_hook_write_to_a_local_store_lands_with_its_memory_type():
    """Every hook write to a local store used to fail: the library takes the memory type's
    enum and the hook passed its name, which raised `AttributeError` inside the store."""
    mem = _mem()
    stored, failed = write.store_facts(
        mem, [extract.Fact("user", "prefers", RUFF, "procedural")], TURN)
    assert (stored, failed) == (1, [])
    assert mem.history("user", "prefers")[0].memory_type.value == "procedural"
    assert write._memory_type("not-a-type") == "not-a-type"


# -- the hosted client -------------------------------------------------------------------


class FakeHosted(hosted.HostedRecall):
    """The real client with the transport replaced: `_call` records and answers."""

    def __init__(self, schema, answers=None):
        super().__init__("mv_key", "https://example.test")
        self._schemas = schema
        self.sent: list = []
        self.answers = answers or {}

    def _ensure_session(self):
        return True

    def _call(self, tool, arguments):
        self.sent.append((tool, dict(arguments)))
        return self.answers.get(tool, f"added 1\n+ [{ID_C}] ok")


FULL = {"memory_remember": {"subject", "predicate", "object", "confidence", "memory_type",
                            "extractor", "sources", "replaces", "reason", "expires_at"},
        "memory_link": {"from_id", "to_id", "relation"},
        "memory_end": {"claim_id", "reason"}}


def test_a_hosted_supersede_end_and_link_use_the_servers_tools():
    client = FakeHosted(FULL)
    props = _proposals(
        {"kind": "supersede", "claim_id": ID_A, "subject": "memvara",
         "predicate": "deploys_to", "object": "Hetzner", "reason": "moved",
         "expires_at": "2027-01-01"},
        {"kind": "end", "claim_id": ID_B, "reason": "done"},
        {"kind": "link", "from": "new:0", "to": ID_B, "relation": "extends"})
    done = agentic.apply(client, props, turn=TURN, hosted=True, sources=["ep_123456"])
    assert done.failed == [] and (done.stored, done.replaced, done.ended, done.linked) == \
        (1, 1, 1, 1)
    remember = client.sent[0][1]
    assert remember["replaces"] == ID_A and remember["reason"] == "moved"
    assert remember["expires_at"] == "2027-01-01T00:00:00+00:00"
    assert remember["memory_type"] == "semantic" and remember["sources"] == ["ep_123456"]
    assert client.sent[1] == ("memory_end", {"claim_id": ID_B, "reason": "done"})
    assert client.sent[2] == ("memory_link", {"from_id": ID_C, "to_id": ID_B,
                                              "relation": "extends"})


def test_a_hosted_end_the_server_could_not_apply_is_a_failure():
    """`memory_end` answers an unknown id with plain text, not an error flag. Counting that
    as an end would report a change that never happened."""
    client = FakeHosted(FULL, {"memory_end": "Nothing ended: no claim is visible here."})
    done = agentic.apply(client, _proposals({"kind": "end", "claim_id": ID_B,
                                             "reason": "done"}), turn=TURN, hosted=True)
    assert done.ended == 0 and done.failed[0].startswith(f"end {ID_B}: HostedError: Nothing")


def test_a_hosted_end_without_a_reason_sends_none():
    client = FakeHosted(FULL, {"memory_end": "Ended claim."})
    assert client.end(ID_B) == "Ended claim."
    assert client.sent == [("memory_end", {"claim_id": ID_B})]


def test_a_server_without_links_refuses_before_calling():
    client = FakeHosted({"memory_remember": {"subject"}})
    with pytest.raises(hosted.HostedError, match="does not offer memory_link"):
        client.link(ID_A, ID_B, "extends")
    assert client.sent == []


def test_a_hosted_replacement_without_a_reason_sends_only_the_id():
    client = FakeHosted(FULL)
    client.remember("memvara", "deploys_to", "Hetzner", replaces=ID_A)
    assert client.sent[0][1]["replaces"] == ID_A and "reason" not in client.sent[0][1]


# -- the capture hook --------------------------------------------------------------------


def _transcript(tmp_path) -> pathlib.Path:
    path = tmp_path / "session.jsonl"
    lines = [
        {"type": "user", "message": {"role": "user", "content": "what does deploy do?"}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "It pushes to fly.io."}]}},
        {"type": "user", "message": {"role": "user", "content":
            "remember: memvara deploys to Hetzner now, we moved off fly.io"}},
    ]
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    return path


def _run_capture(monkeypatch, tmp_path, store=None, outcome="unset"):
    path = _transcript(tmp_path)
    monkeypatch.setattr(capture, "payload", lambda: {
        "session_id": SESSION, "transcript_path": str(path), "cwd": str(tmp_path)})
    monkeypatch.setattr(capture, "open_writer", lambda: (store or LocalStore(), None))
    monkeypatch.setattr(capture, "_keep_turn", lambda *a: (True, []))
    seen = {}

    def fake_triples(*a, **k):
        seen["triples"] = True
        return [extract.Fact("memvara", "deploys_to", "Hetzner", "semantic")]

    monkeypatch.setattr(capture, "triples", fake_triples)
    if outcome != "unset":
        def fake_agentic(store, turn, context, cwd, injected, *, hosted, sources=()):
            seen["context"] = context
            seen["turn"] = turn
            return outcome

        monkeypatch.setattr(capture.agentic, "capture", fake_agentic)
    assert capture.main() == 0
    return seen


def test_capture_uses_agentic_capture_by_default_and_counts_what_it_stored(
        monkeypatch, tmp_path):
    outcome = agentic.Outcome(2, 3, 1, agentic.Applied(2, 1, 0, 1, ["x: y"], []))
    seen = _run_capture(monkeypatch, tmp_path, outcome=outcome)
    assert "triples" not in seen
    assert "It pushes to fly.io." in seen["context"] and "Hetzner" in seen["turn"]
    assert "Hetzner" not in seen["context"]
    assert counts.read(SESSION)["captured"] == 2
    assert ("agentic searches=2 proposals=3 refused=1 stored=2 replaced=1 ended=0 "
            "linked=1 episode=yes; failed=x: y") in _log(tmp_path)


def test_capture_falls_back_to_the_single_call_extraction_when_agentic_cannot_run(
        monkeypatch, tmp_path):
    store = LocalStore()
    seen = _run_capture(monkeypatch, tmp_path, store=store, outcome=None)
    assert seen["triples"] is True
    assert store.remembered[0]["object"] == "Hetzner"


def test_capture_with_the_switch_off_never_starts_an_agentic_run(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMVARA_FEATURE_AGENTIC_CAPTURE", "0")

    def boom(*a, **k):
        raise AssertionError("agentic capture ran with its switch off")

    monkeypatch.setattr(capture.agentic, "capture", boom)
    seen = _run_capture(monkeypatch, tmp_path)
    assert seen["triples"] is True


def test_capture_counts_nothing_when_agentic_stored_nothing(monkeypatch, tmp_path):
    outcome = agentic.Outcome(1, 0, 0, agentic.Applied(0, 0, 0, 0, [], []))
    _run_capture(monkeypatch, tmp_path, outcome=outcome)
    assert counts.read(SESSION)["captured"] == 0
    assert capture.STATE.exists(), "the turn counts as mined"


def test_a_malformed_reply_through_the_hook_writes_nothing_and_does_not_retry(
        monkeypatch, tmp_path):
    """End to end through `capture.main` with the process faked: the reply is unusable, so
    nothing is written, the single-call extractor is not run as well, and the transcript
    size is recorded so the same turn is not mined again."""
    _local_env(monkeypatch, tmp_path)
    _fake(monkeypatch, [init(), result(text="I could not decide.")])
    store = LocalStore()
    seen = _run_capture(monkeypatch, tmp_path, store=store)
    assert "triples" not in seen and store.remembered == []
    assert json.loads(capture.STATE.read_text())
    assert "not a proposal list" in _log(tmp_path)


def test_the_capture_alert_still_fires_when_both_extractions_fail(monkeypatch, tmp_path):
    """Repeated failure must still reach the terminal. The agentic run fails first and
    falls back; the single-call extraction then fails on the same expired login and raises
    the alert, exactly as it did before agentic capture existed."""
    _local_env(monkeypatch, tmp_path)
    _fake(monkeypatch, [init(), result(text="OAuth session expired", is_error=True)])

    class Done:
        returncode = 1
        stdout = json.dumps({"is_error": True, "result": "OAuth session expired"})
        stderr = ""

    monkeypatch.setattr(extract.subprocess, "run", lambda *a, **k: Done())
    monkeypatch.setattr(extract, "_chain", lambda: [CLAUDE_CLI])
    path = _transcript(tmp_path)
    monkeypatch.setattr(capture, "payload", lambda: {
        "session_id": SESSION, "transcript_path": str(path), "cwd": str(tmp_path)})
    monkeypatch.setattr(capture, "open_writer", lambda: (LocalStore(), None))
    monkeypatch.setattr(capture, "_keep_turn", lambda *a: (True, []))
    assert capture.main() == 0
    assert ipc.due_capture_alert() == "capture failing: OAuth session expired"
    log = _log(tmp_path)
    assert log.index("agentic capture fell back") < log.index("extraction did not run")


def test_an_agentic_run_that_answers_clears_an_earlier_alert(monkeypatch, tmp_path):
    _local_env(monkeypatch, tmp_path)
    ipc.raise_capture_alert("OAuth session expired")
    _fake(monkeypatch, [init(), result([])])
    assert agentic.capture(LocalStore(), TURN, "", "/repo", [], hosted=False) is not None
    assert ipc.due_capture_alert() == ""
    assert "extraction ran via claude (agentic, 0 searches)" in _log(tmp_path)


def test_one_search_is_logged_in_the_singular(monkeypatch, tmp_path):
    _local_env(monkeypatch, tmp_path)
    _fake(monkeypatch, [init(), tool_use("t1"), tool_result("t1", SEARCH_HIT), result([])])
    agentic.capture(LocalStore(), TURN, "", "/repo", [], hosted=False)
    assert "(agentic, 1 search)" in _log(tmp_path)


def test_what_a_run_spent_is_recorded(monkeypatch, tmp_path):
    _local_env(monkeypatch, tmp_path)
    _fake(monkeypatch, [init(), result([])])
    agentic.capture(LocalStore(), TURN, "", "/repo", [], hosted=False)
    assert usage.totals(usage.DEFAULT_PATH) == {"write.tokens_in": 8020,
                                                "write.tokens_out": 400}


def test_usage_of_messages_is_counted_once_per_message():
    watch = agentic._Watch()
    one = {"input_tokens": 3, "output_tokens": 4, "cache_read_input_tokens": True}
    for event in (tool_use("t1", msg="m1", usage_=one), tool_use("t2", msg="m1", usage_=one),
                  {"type": "assistant", "message": "not a dict"},
                  {"type": "user", "message": None}, {"type": "other"}):
        watch.feed(json.dumps(event))
    assert watch.usage() == {"input_tokens": 3, "output_tokens": 4}
    assert watch.calls == 2


# -- the earlier-turns window ------------------------------------------------------------


def test_the_context_is_the_turns_before_the_new_one_cut_from_the_front(tmp_path):
    raw = _transcript(tmp_path).read_bytes()
    turn, injected, context = transcript.last_turn_with_context(raw, 1000)
    assert turn.startswith("User: remember: memvara deploys to Hetzner")
    assert context == "User: what does deploy do?\nClaude: It pushes to fly.io."
    assert transcript.last_turn_with_context(raw, 10)[2] == context[-10:]
    assert transcript.last_turn_with_context(raw, 0)[2] == ""
    assert transcript.last_turn_with_injections(raw) == (turn, injected)
    assert transcript.last_turn_with_context(b"", 1000) == ("", [], "")
    assert agentic.CONTEXT_CHARS == 4_000


# -- the switch --------------------------------------------------------------------------


def test_the_switch_is_on_by_default_in_the_hooks_and_in_the_library():
    assert settings.FEATURE_DEFAULTS["agentic_capture"] is True
    assert LIBRARY_FEATURE_DEFAULTS["agentic_capture"] is True
    assert settings.enabled("agentic_capture") is True


def test_the_rules_carry_the_vocabulary_the_project_and_the_attribution_rules():
    rules = agentic.system_prompt("/repo")
    assert "at most\n4 times" in rules or "at most 4 times" in " ".join(rules.split())
    assert "prefers (subject: user, object: full sentences)" in rules
    assert "The project key for this repository is: memvara" in rules
    assert "## Who said it" in rules and not rules.rstrip().endswith("Exchange:")


def test_an_empty_earlier_window_is_marked_as_none():
    data = agentic.data_block("User: hi", "  ", "abc")
    assert "(none)" in data and data.startswith("<data-abc>")


def test_the_hosted_argument_probe_answers_through_the_store():
    assert write.takes(FakeHosted(FULL), "expires_at", hosted=True) is True
    assert write.takes(FakeHosted({"memory_remember": {"subject"}}), "expires_at",
                       hosted=True) is False
    assert write.takes(LocalStore(), "replaces", hosted=False) is True
    assert write.takes(LocalStore(), "expires_at", hosted=False) is False


def test_timedelta_is_not_a_date():
    """A relative expiry such as "in 3 days" is refused rather than guessed at."""
    now = datetime.now(timezone.utc)
    assert agentic._expiry(str(timedelta(days=3)), now)[1].endswith("is not a date")


def test_notes_about_what_could_not_be_applied_reach_the_log(monkeypatch, tmp_path):
    _local_env(monkeypatch, tmp_path)
    fact = {"kind": "fact", "subject": "user", "predicate": "prefers", "object": RUFF,
            "expires_at": "2099-01-01"}
    _fake(monkeypatch, [init(), result([fact])])
    out = agentic.capture(LocalStore(), TURN, "", "/repo", [], hosted=False)
    assert out.applied.stored == 1
    assert "agentic note prefers: expires_at dropped, this store does not take it yet" \
        in _log(tmp_path)


def test_a_hosted_install_runs_agentic_capture_as_hosted_and_closes_the_client(
        monkeypatch, tmp_path):
    path = _transcript(tmp_path)
    closed, handed = [], {}
    monkeypatch.setattr(capture, "payload", lambda: {
        "session_id": SESSION, "transcript_path": str(path), "cwd": str(tmp_path)})
    monkeypatch.setattr(capture, "open_writer",
                        lambda: (FakeHosted(FULL), lambda: closed.append(True)))
    monkeypatch.setattr(capture, "_keep_turn", lambda *a: (True, ["ep_abcdef12"]))

    def fake_agentic(store, turn, context, cwd, injected, *, hosted, sources=()):
        handed.update(hosted=hosted, sources=list(sources))
        return agentic.Outcome(0, 0, 0, agentic.Applied(0, 0, 0, 0, [], []))

    monkeypatch.setattr(capture.agentic, "capture", fake_agentic)
    assert capture.main() == 0
    assert handed == {"hosted": True, "sources": ["ep_abcdef12"]} and closed == [True]


def test_the_single_call_extractor_still_drops_and_repairs_through_the_shared_checks(
        monkeypatch, tmp_path):
    """`triples()` now calls `extract.vet` for each fact. What it keeps, drops and repairs,
    and what it logs about each, must be what it did before the checks moved."""
    reply = json.dumps({"facts": [
        {"subject": "user", "predicate": "invented", "object": "x"},
        {"subject": "user", "predicate": "prefers",
         "object": "always run the formatter before committing so the lint job passes on "
                   "the first try"},
        {"subject": "user", "predicate": "deploys_to", "object": "Hetzner"},
    ]})
    monkeypatch.setattr(extract, "_payload", lambda text, prompt: (reply, {}, ""))
    monkeypatch.setattr(extract, "project_subject", lambda cwd=None: "memvara")
    turn = ("User: always run the formatter before committing, Ruff breaks the lint job "
            "otherwise. memvara deploys to Hetzner.")
    facts = extract.triples(turn, "/repo")
    assert [(f.subject, f.predicate) for f in facts] == [("user", "prefers"),
                                                        ("memvara", "deploys_to")]
    assert "stated by the user as" in facts[0].object
    log = _log(tmp_path)
    assert "dropped invented: not in vocabulary" in log
    assert "repaired prefers: kept the user's own wording for hetzner, ruff" in log
