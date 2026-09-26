"""What the hook conformance tests share: the hosts and what each one reads and sends, a
store that holds one known memory, the turn a capture mines, and a way to make many hook
runs side by side."""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import json
import os
import pathlib
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterator, Mapping, Sequence, TypeVar

import pytest

from harness import known_bugs, stores
from harness.fakes.cli import NO_FAKES, FakeClis, HangingClis
from harness.fakes.hosted_mcp import FakeHostedMcp
from harness.hooks import HookResult, HookRunner, host_record, process_alive, short_dir

#: The hosts the plugin has a record for, in plugin/hooks/hosts.
HOSTS = ("claude", "codex", "copilot", "cursor", "opencode")

#: The user every store here belongs to.
USER = "tester"

#: The one memory the stores here hold, as recall renders it, and a prompt that the
#: hashing embedder matches with it above the recall hook's score floor.
MEMORY = "user lives in Lisbon"
PROMPT = "user lives in Lisbon"

#: A prompt that matches nothing in such a store.
UNRELATED = "which database does the billing service use"

#: Words from the header of the standing preferences block, which session start injects
#: and recall refreshes (plugin/hooks/session_start.py and recall.py, STANDING_HEADER).
STANDING_WORDS = "how this user wants work done"


def status_line(words: str) -> str:
    """The status line the hooks show a person for `words`, on the one host that shows
    one."""
    return f"⋈ Memvara · {words}"


#: One turn of a conversation that states a fact, for a capture to mine. "remember" is
#: one of the words that make capture mine a turn however short it is.
USER_TURN = "Please remember that I live in Lisbon now."
ASSISTANT_TURN = "Noted: you live in Lisbon."

#: What a fake extractor answers for that turn. The single-call extraction reads a list
#: of facts (plugin/hooks/lib/extract.py). Agentic capture, which runs first on Claude
#: Code, reads a list of proposals (plugin/hooks/lib/agentic.py).
FACTS_REPLY = json.dumps({"facts": [
    {"subject": "user", "predicate": "lives_in", "object": "Lisbon"}]})
PROPOSALS_REPLY = json.dumps({"proposals": [
    {"kind": "fact", "subject": "user", "predicate": "lives_in", "object": "Lisbon"}]})

#: The event each host fires for each hook, as plugin/hooks/hosts/<id>.py records it.
#: Cursor fires none for recall.
EVENTS: Mapping[str, Mapping[str, str]] = {
    "claude": {"session_start": "SessionStart", "recall": "UserPromptSubmit",
               "capture": "Stop", "approve": "PreToolUse"},
    "codex": {"session_start": "SessionStart", "recall": "UserPromptSubmit",
              "capture": "Stop", "approve": "PreToolUse"},
    "copilot": {"session_start": "SessionStart", "recall": "UserPromptSubmit",
                "capture": "Stop", "approve": "PreToolUse"},
    "cursor": {"session_start": "sessionStart", "capture": "sessionEnd",
               "approve": "preToolUse"},
    "opencode": {"session_start": "chat.message", "recall": "chat.message",
                 "capture": "session.idle", "approve": "permission.ask"},
}

#: Where each host reads a hook's answer, as its host record says the client was measured
#: to read it: whether the reply is nested in `hookSpecificOutput`, the key of the status
#: line a person sees (None when the host shows none), the key of the text put in front
#: of the model, and the keys of an approval's verdict and its reason. OpenCode's client
#: is plugin/hooks/js/shim.mjs, which reads `additionalContext` and `status`.
SHAPES: Mapping[str, tuple[bool, str | None, str, str, str]] = {
    "claude": (True, "systemMessage", "additionalContext",
               "permissionDecision", "permissionDecisionReason"),
    "codex": (True, None, "additionalContext",
              "permissionDecision", "permissionDecisionReason"),
    "copilot": (False, None, "additionalContext",
                "permissionDecision", "permissionDecisionReason"),
    "cursor": (False, None, "additional_context", "permission", "reason"),
    "opencode": (False, None, "additionalContext", "status", "reason"),
}

#: The words a client matches exactly in a reply: every event name, and the verdict.
EXACT_WORDS = frozenset({"allow"} | {event for events in EVENTS.values()
                                     for event in events.values()})

#: The hosts whose capture run.py hands to a child in a new session, so that the client's
#: turn is never held (hosts/<id>.py, `detach_capture`).
DETACHES = frozenset({"codex", "copilot", "cursor"})

#: The limit, in seconds, each host gives each hook: the hook contract the design names
#: (session start 20, recall 10, approve 5, capture 180), as each host record declares it
#: in `timeouts`. Capture's limit is 120 seconds on the hosts other than Claude Code,
#: where only the single-call extraction runs.
LIMITS: Mapping[str, Mapping[str, int]] = {
    "claude": {"session_start": 20, "recall": 10, "capture": 180, "approve": 5},
    "codex": {"session_start": 20, "recall": 10, "capture": 120, "approve": 5},
    "copilot": {"session_start": 20, "recall": 10, "capture": 120, "approve": 5},
    "cursor": {"session_start": 20, "capture": 120, "approve": 5},
    "opencode": {"session_start": 20, "recall": 10, "capture": 120, "approve": 5},
}

#: The names each host gives a tool of the memvara server when it reaches the approve
#: hook. Claude Code and Codex prefix `mcp__<server>__` (hosts/claude.py, hosts/codex.py),
#: and a server a plugin installed is named `plugin_memvara_memvara` (approve.py).
#: Copilot joins with a hyphen, which was measured (hosts/copilot.py). Nobody has
#: measured Cursor's or OpenCode's names, so these use the first separator each record
#: lists.
TOOL_NAMES: Mapping[str, tuple[str, ...]] = {
    "claude": ("mcp__memvara__{tool}", "mcp__plugin_memvara_memvara__{tool}"),
    "codex": ("mcp__memvara__{tool}",),
    "copilot": ("memvara-{tool}",),
    "cursor": ("memvara__{tool}",),
    "opencode": ("memvara__{tool}",),
}

#: Keys each host sends beside the ones the hooks read, as its host record lists them.
#: Sending them shows the hooks ignore what they do not read. Cursor sends no `cwd`: it
#: sends `workspace_roots`, a list, and it is the only host that sends the user's email
#: address. OpenCode's payload is framed by plugin/hooks/js/opencode.mjs, which adds
#: nothing.
EXTRA_KEYS: Mapping[str, Mapping[str, Any]] = {
    "claude": {"permission_mode": "default"},
    "codex": {"model": "gpt-5-codex", "permission_mode": "default", "source": "startup",
              "turn_id": "turn-1"},
    "copilot": {},
    "cursor": {"conversation_id": "conversation-1", "cursor_version": "2026.08.25",
               "generation_id": "generation-1", "is_background_agent": False,
               "model": "auto", "user_email": "tester@example.com"},
    "opencode": {},
}

#: The four hooks, in the order they run in a session (plugin/hooks/core/host.py).
HOOKS = ("session_start", "recall", "capture", "approve")

#: Invalid UTF-8 inside an otherwise ordinary payload.
_INVALID_UTF8 = (b'{"session_id": "s", "prompt": "user lives in \xff\xfe Lisbon", '
                 b'"tool_name": "mcp__memvara__memory_search\xc3"}')

#: What the hostile-payload tests send to every hook, each with the variables its hook
#: process gets. Invalid UTF-8 is sent twice, because a child's stdin decodes it one of
#: two ways: strictly, so the read fails, or with surrogateescape, which Python uses in
#: its UTF-8 mode and under the C locale, and which turns each bad byte into a lone
#: surrogate. `PYTHONIOENCODING` picks one, so the result does not depend on the locale
#: of the machine running the tests.
HOSTILE: Mapping[str, tuple[bytes, Mapping[str, str]]] = {
    "no fields at all": (b"{}", {}),
    "empty stdin": (b"", {}),
    "text that is not JSON": (b"this is not json {", {}),
    "a JSON list": (b'[{"session_id": "s", "prompt": "user lives in Lisbon"}]', {}),
    "invalid UTF-8, read strictly": (_INVALID_UTF8, {"PYTHONIOENCODING": "utf-8:strict"}),
    "invalid UTF-8, read with surrogateescape": (
        _INVALID_UTF8, {"PYTHONIOENCODING": "utf-8:surrogateescape"}),
    "a prompt that is a lone surrogate": (json.dumps({
        "session_id": "s", "cwd": "\ud800", "prompt": "\ud800", "tool_name": "\ud800",
        "transcript_path": "\ud800"}).encode(), {}),
    "fields of the wrong type": (json.dumps({
        "session_id": 7, "cwd": {"a": 1}, "prompt": ["user lives in Lisbon"],
        "tool_name": None, "transcript_path": 3.5, "stop_hook_active": "yes",
        "workspace_roots": "not a list"}).encode(), {}),
    "unknown fields": (json.dumps({
        "session_id": "s", "prompt": "user lives in Lisbon", "hook_event_name": "NoSuchEvent",
        "unknown": {"deep": [1, {"x": None}]}, "": ""}).encode(), {}),
    "an 8 MB payload": (b'{"session_id": "s", "prompt": "user lives in Lisbon", "padding": "'
                        + b"x" * 8_000_000 + b'"}', {}),
    "nesting 100,000 levels deep": (b'{"prompt": ' + b"[" * 100_000 + b"]" * 100_000 + b"}",
                                    {}),
}

#: JSON values that are not objects, which the nightly tier adds.
MORE_HOSTILE: Mapping[str, tuple[bytes, Mapping[str, str]]] = {
    "a JSON string": (b'"user lives in Lisbon"', {}),
    "JSON null": (b"null", {}),
    "a JSON number": (b"42", {}),
}

#: The payloads that are not a JSON object once they are read. A hook must answer each
#: one exactly as it answers `{}`: plugin/hooks/core/envelope.py, `read_event`, says that
#: anything unreadable becomes an empty event.
UNREADABLE = ("empty stdin", "text that is not JSON", "a JSON list",
              "invalid UTF-8, read strictly")

#: The line run.py writes when a hook's body raised and its last guard kept the exit code
#: at 0 (plugin/hooks/run.py, `main`). A line about handing capture to a child is not one.
_CRASH = re.compile(r"^failed hook=\S+ host=\S+ (?!detach)")

#: One hook run, waiting to be made by `run_all`.
Job = Callable[[], HookResult]
Jobs = dict[tuple[str, ...], Job]

K = TypeVar("K")


def tool_name(host: str, tool: str) -> str:
    """The first name `host` gives the memvara tool `tool` (TOOL_NAMES)."""
    return TOOL_NAMES[host][0].format(tool=tool)


#: The stdin keys a host sends as a list, of which the hooks read the first item
#: (plugin/hooks/core/envelope.py, `read_event`): Cursor's `workspace_roots`.
_LIST_KEYS = frozenset({"workspace_roots"})


def host_payload(host: str, hook: str, *, session: str, cwd: pathlib.Path,
                 **fields: Any) -> dict[str, Any]:
    """The stdin `host` sends for `hook`.

    The working directory goes under the first key the host record lists for it, which is
    the key the host itself sends: `workspace_roots`, a list, on Cursor, and `cwd`
    elsewhere. `fields` are keys every host spells alike: prompt, tool_name, tool_input
    and transcript_path.
    """
    body: dict[str, Any] = {"hook_event_name": EVENTS[host][hook], "session_id": session}
    key = host_record(host).fields["cwd"][0]
    body[key] = [str(cwd)] if key in _LIST_KEYS else str(cwd)
    body.update(EXTRA_KEYS[host])
    body.update(fields)
    return body


def host_json(host: str, hook: str, **options: Any) -> str:
    """`host_payload`, as the JSON text the host writes to the hook's stdin."""
    return json.dumps(host_payload(host, hook, **options))


def write_transcript(host: str, path: pathlib.Path,
                     turns: Sequence[tuple[str, str]]) -> pathlib.Path:
    """Write `turns`, each a (user, assistant) pair, as the transcript `host` keeps.

    The format is the one the host record names (`transcript.format`). JSONL keeps the
    speaker under the record's `role_key`, which is `type` on Claude Code and OpenCode and
    `role` on Cursor. A Codex rollout is a list of `response_item` entries, and Copilot's
    is an event log. plugin/hooks/lib/transcript.py reads each one.
    """
    entries: list[dict[str, Any]] = []
    for user, assistant in turns:
        entries += _entries(host, user, assistant)
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries),
                    encoding="utf-8")
    return path


def _entries(host: str, user: str, assistant: str) -> list[dict[str, Any]]:
    """One turn, in the transcript format the record of `host` names."""
    spec = host_record(host).transcript
    if spec.format == "codex-rollout":
        return [_codex_message("user", "input_text", user),
                _codex_message("assistant", "output_text", assistant)]
    if spec.format == "copilot-events":
        return [{"type": "user.message", "data": {"content": user}},
                {"type": "assistant.message", "data": {"content": assistant}}]
    if spec.format != "jsonl":
        raise ValueError(f"write_transcript cannot write the {spec.format!r} transcript "
                         f"that {host} keeps")
    return [{spec.role_key: "user",
             "message": {"content": [{"type": "text", "text": user}]}},
            {spec.role_key: "assistant",
             "message": {"content": [{"type": "text", "text": assistant}]}}]


def _codex_message(role: str, kind: str, text: str) -> dict[str, Any]:
    return {"type": "response_item", "payload": {
        "type": "message", "role": role, "content": [{"type": kind, "text": text}]}}


def script_clis(directory: pathlib.Path, host: str, runs: int = 1) -> FakeClis:
    """Fake `claude` and `codex` that mine USER_TURN on `host`, `runs` times each.

    On Claude Code, agentic capture runs first and reads proposals. On Codex, the codex
    CLI answers. Elsewhere the host's own CLI is not on PATH, so the chain falls back to
    `claude` for the single-call extraction (plugin/hooks/lib/extract.py, `_chain`).
    """
    clis = FakeClis(directory)
    clis.script("claude", *[PROPOSALS_REPLY if host == "claude" else FACTS_REPLY] * runs)
    clis.script("codex", *[FACTS_REPLY] * runs)
    return clis


def check_capture_frees_the_turn(make: Callable[..., HookRunner], work: pathlib.Path,
                                 host: str) -> None:
    """On a host that hands capture to a child, the hook returns at once however long the
    extraction takes: here the extractor never answers, and the hook returns within 3
    seconds with the child still running. The runner kills that child, and the stub it
    started, when it closes.

    "Still running" is read with `process_alive`, because a child that has ended but that
    nothing has reaped yet still answers signal 0."""
    if sys.platform == "win32":
        pytest.skip(NO_FAKES)
    env = store_env(make_store(work / "capture.db", memory=False))
    runner = make(host, env=env, stubs=HangingClis(work / "hanging"))
    transcript = write_transcript(host, work / "t.jsonl", [(USER_TURN, ASSISTANT_TURN)])
    result = runner.run("capture", session="s", transcript_path=str(transcript),
                        wait_detached=False)
    assert (result.exit_code, result.stdout) == (0, "")
    assert result.detached_pid is not None
    assert result.elapsed < 3, result.elapsed
    assert process_alive(result.detached_pid), (
        f"the child {result.detached_pid} the capture was handed to had already ended")


def make_store(path: pathlib.Path, *, memory: bool = True) -> pathlib.Path:
    """A store file at `path`, holding MEMORY for USER unless `memory` is False."""
    with stores.file(path) as mem:
        if memory:
            mem.scope(user=USER).remember("user", "lives_in", "Lisbon")
    return path


def store_env(path: pathlib.Path) -> dict[str, str]:
    """The variables that name the store at `path`, which belongs to USER, to the hooks."""
    return {"MEMVARA_DB": str(path), "MEMVARA_USER": USER}


@contextlib.contextmanager
def runner_factory(work: pathlib.Path) -> Iterator[Callable[..., HookRunner]]:
    """Make HookRunners that share the working directory `work`, each with a short home
    of its own unless it is given one. On exit, close every runner, which stops any daemon
    it allowed and any capture child it left running, and remove the homes."""
    homes: list[pathlib.Path] = []
    made: list[HookRunner] = []

    def make(host: str, *, home: pathlib.Path | None = None,
             **options: Any) -> HookRunner:
        if home is None:
            home = short_dir("home")
            homes.append(home)
        runner = HookRunner(host, home=home, cwd=work, **options)
        made.append(runner)
        return runner

    try:
        yield make
    finally:
        for runner in made:
            runner.close()
        for home in homes:
            shutil.rmtree(home, ignore_errors=True)


def job(runner: HookRunner, hook: str, **options: Any) -> Job:
    """One run of `hook` through `runner`, to be made later by `run_all`."""
    return functools.partial(runner.run, hook, **options)


def run_all(jobs: Mapping[K, Job]) -> dict[K, HookResult | Exception]:
    """Make every run in `jobs` side by side, and return each one's result, or the
    exception it raised, under its key.

    A hook run spends most of its time starting Python and importing, so a module's runs
    made together take a fraction of the time they take one by one. Each run has a home
    of its own, so none can see another's logs or state.
    """
    def attempt(run: Job) -> HookResult | Exception:
        try:
            return run()
        except Exception as exc:  # noqa: BLE001 - the test that reads it raises it
            return exc

    with ThreadPoolExecutor(max_workers=max(2, min(8, os.cpu_count() or 2))) as pool:
        results = list(pool.map(attempt, jobs.values()))
    return dict(zip(jobs, results))


def result_of(outcome: HookResult | Exception) -> HookResult:
    """The result of a run `run_all` made, raising what the run raised instead."""
    if isinstance(outcome, Exception):
        raise outcome
    return outcome


@dataclasses.dataclass
class Runs:
    """A module's runs, by key, and the stores its captures wrote to, by host."""

    results: Mapping[Any, HookResult | Exception]
    stores: Mapping[str, pathlib.Path] = dataclasses.field(default_factory=dict)

    def __getitem__(self, key: Any) -> HookResult:
        """The result of the run under `key`, raising what that run raised instead."""
        return result_of(self.results[key])


def _body(host: str, reply: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """The part of a reply that holds the context and the verdict on `host`."""
    if reply is None:
        return {}
    if SHAPES[host][0]:
        inner = reply.get("hookSpecificOutput")
        return inner if isinstance(inner, dict) else {}
    return reply


def context_of(host: str, reply: Mapping[str, Any] | None) -> str:
    """The text a reply puts in front of the model on `host`, or "" when it puts none."""
    return str(_body(host, reply).get(SHAPES[host][2], ""))


def decision_of(host: str, reply: Mapping[str, Any] | None) -> str | None:
    """The verdict an approve reply gives on `host`, or None when it gives none."""
    verdict = _body(host, reply).get(SHAPES[host][3])
    return verdict if isinstance(verdict, str) else None


def status_of(host: str, reply: Mapping[str, Any] | None) -> str | None:
    """The status line a reply shows the person on `host`, or None when it shows none."""
    key = SHAPES[host][1]
    if key is None or reply is None:
        return None
    status = reply.get(key)
    return status if isinstance(status, str) else None


def crashes(result: HookResult) -> list[str]:
    """The lines in which run.py says the hook's body raised."""
    return [line for line in result.log("hooks") if _CRASH.match(line)]


def observed(result: HookResult) -> tuple[str, tuple[tuple[str, tuple[str, ...]], ...]]:
    """What a person or a log reader can see of one hook run: its reply, and the lines it
    added to each log, without their timestamps."""
    return json.dumps(result.reply, sort_keys=True), tuple(sorted(result.logs.items()))


# -- the four outcomes -------------------------------------------------------------------

#: The ways session start and recall can end. For session start, "nothing matches" is a
#: store that holds nothing.
OUTCOMES = ("not configured", "store cannot open", "store unreachable", "nothing matches",
            "memories injected")

#: The hooks that read the store.
READING_HOOKS = ("session_start", "recall")


def recall_hosts(hosts: Sequence[str]) -> list[str]:
    """The hosts among `hosts` that fire an event for recall."""
    return [host for host in hosts if "recall" in EVENTS[host]]


def outcome_cases(hosts: Sequence[str], outcomes: Sequence[str]) -> list[tuple[str, str, str]]:
    """Every (host, hook, outcome) to check: each outcome, for each reading hook each host
    fires."""
    return [(host, hook, outcome) for host in hosts for hook in READING_HOOKS
            if hook in EVENTS[host] for outcome in outcomes]


def outcome_matrix(base: pathlib.Path, hosts: Sequence[str]) -> Runs:
    """Run session start and recall on each host once for every outcome, keyed by
    (host, hook, outcome), each run with a home of its own.

    A store that cannot open is a file that is not a SQLite database. A store that cannot
    be reached is the hosted endpoint, `FakeHostedMcp`, refusing every handshake with a
    503; the hooks use it because no local store is named and the variables name a key
    and the endpoint.
    """
    work = base / "work"
    work.mkdir()
    full = make_store(base / "memory.db")
    empty = make_store(base / "empty.db", memory=False)
    broken = base / "broken.db"
    broken.write_bytes(b"this file is not a SQLite database " * 64)
    jobs: Jobs = {}
    with FakeHostedMcp() as fake, runner_factory(work) as make:
        fake.fail("initialize", 503)
        url = fake.serve()
        for host, hook, outcome in outcome_cases(hosts, OUTCOMES):
            env, prompt = {
                "not configured": ({}, PROMPT),
                "store cannot open": (store_env(broken), PROMPT),
                "store unreachable": ({"MEMVARA_API_KEY": fake.api_key,
                                       "MEMVARA_SERVER_URL": url}, PROMPT),
                "nothing matches": (store_env(empty if hook == "session_start" else full),
                                    UNRELATED),
                "memories injected": (store_env(full), PROMPT),
            }[outcome]
            fields = {"prompt": prompt} if hook == "recall" else {}
            jobs[host, hook, outcome] = job(
                make(host, env=env), hook,
                stdin=host_json(host, hook, session="outcomes", cwd=work, **fields))
        results = run_all(jobs)
    return Runs(results)


def check_told_apart(runs: Runs, host: str, hook: str, one: str, other: str) -> None:
    """A person or a log reader can tell outcome `one` from outcome `other`."""
    first, second = runs[host, hook, one], runs[host, hook, other]
    assert observed(first) != observed(second), (
        f"{hook} on {host} looks the same for {one!r} and {other!r}: {observed(first)}")


def check_recall_says_it_could_not_ask(runs: Runs, host: str) -> None:
    """Recall that could not ask the store says so in its log. On a host that shows no
    status line, the log is the only account there is."""
    result = runs[host, "recall", "store unreachable"]
    assert "failed reason=unknown" in result.log("recall"), result.logs


def silent_hosts(hosts: Sequence[str]) -> list[str]:
    """The hosts among `hosts` that show a person no status line."""
    return [host for host in hosts if SHAPES[host][1] is None]


def _not_configured_reply(host: str) -> dict[str, str] | None:
    """What a reading hook prints on `host` when nothing is configured."""
    status = SHAPES[host][1]
    return None if status is None else {status: status_line("not configured")}


def pin_cannot_open(runs: Runs, host: str, hook: str) -> None:
    """B55: a configured local store that cannot open reads exactly as no store at all:
    the reply for "not configured", no line in any log."""
    broken, missing = runs[host, hook, "store cannot open"], runs[host, hook, "not configured"]
    if (broken.reply == _not_configured_reply(host) and broken.logs == {}
            and observed(broken) == observed(missing)):
        raise known_bugs.Reproduced(
            f"B55: {hook} on {host} reports a store that cannot open as not configured")
    check_told_apart(runs, host, hook, "store cannot open", "not configured")


def pin_nothing_matches(runs: Runs, host: str) -> None:
    """B56: on a host that shows no status line, recall prints nothing and logs nothing,
    both when nothing matches and when nothing is configured."""
    nothing = runs[host, "recall", "nothing matches"]
    missing = runs[host, "recall", "not configured"]
    if (nothing.reply, nothing.logs, missing.reply, missing.logs) == (None, {}, None, {}):
        raise known_bugs.Reproduced(
            f"B56: recall on {host} prints and logs nothing both when nothing matches and "
            f"when nothing is configured")
    check_told_apart(runs, host, "recall", "nothing matches", "not configured")


def pin_unreachable(runs: Runs, host: str) -> None:
    """B57: session start says "nothing stored yet" of a hosted store it could not reach,
    where the host shows a status line, and elsewhere looks exactly as it does when
    nothing is configured."""
    unreachable = runs[host, "session_start", "store unreachable"]
    missing = runs[host, "session_start", "not configured"]
    if SHAPES[host][1] is not None:
        wrong = status_of(host, unreachable.reply) == status_line("nothing stored yet")
    else:
        wrong = (unreachable.reply, unreachable.logs, missing.reply, missing.logs) == (
            None, {}, None, {})
    if wrong:
        raise known_bugs.Reproduced(
            f"B57: session start on {host} reports an unreachable store as "
            f"{'empty' if SHAPES[host][1] else 'no store at all'}")
    assert status_of(host, unreachable.reply) != status_line("nothing stored yet")
    check_told_apart(runs, host, "session_start", "store unreachable", "not configured")


# -- hostile payloads --------------------------------------------------------------------

@dataclasses.dataclass
class Hostile:
    """A hostile-payload matrix: its runs, keyed by (host, hook, payload), and the fake
    agent CLIs its captures were given, or None where the fakes cannot run."""

    runs: Runs
    clis: FakeClis | None


def hostile_cases(hosts: Sequence[str],
                  payloads: Sequence[str] | Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """Every (host, hook, payload) to check: each payload, on every hook each host fires."""
    return [(host, hook, name) for host in hosts for hook in HOOKS if hook in EVENTS[host]
            for name in payloads]


def hostile_matrix(base: pathlib.Path, hosts: Sequence[str],
                   payloads: Mapping[str, tuple[bytes, Mapping[str, str]]]) -> Hostile:
    """Send every payload, and `{}`, to every hook each host fires, against a store that
    holds MEMORY. Captures get fake agent CLIs with nothing scripted, because no payload
    here names a transcript, so none may start an extraction."""
    payloads = {"no fields at all": (b"{}", {}), **payloads}
    work = base / "work"
    work.mkdir()
    env = store_env(make_store(base / "memory.db"))
    clis = None if sys.platform == "win32" else FakeClis(base / "clis")
    jobs: Jobs = {}
    with runner_factory(work) as make:
        for host, hook, name in hostile_cases(hosts, payloads):
            if hook == "capture" and clis is None:
                continue
            data, extra = payloads[name]
            runner = make(host, env={**env, **extra},
                          stubs=clis if hook == "capture" else None)
            jobs[host, hook, name] = job(runner, hook, stdin=data)
        results = run_all(jobs)
    return Hostile(Runs(results), clis)


def check_left_alone(hostile: Hostile, host: str, hook: str, payload: str) -> None:
    """The hook exited 0 and wrote nothing to stderr. HookRunner has already checked that
    it printed nothing or one JSON object, within its host's limit."""
    if hook == "capture" and hostile.clis is None:
        pytest.skip(NO_FAKES)
    result = hostile.runs[host, hook, payload]
    assert result.exit_code == 0, result
    assert result.stderr == "", result.stderr


def check_read_as_empty(hostile: Hostile, host: str, hook: str, payload: str) -> None:
    """The hook answered exactly as it answers `{}`, and its body did not crash."""
    if hook == "capture" and hostile.clis is None:
        pytest.skip(NO_FAKES)
    result = hostile.runs[host, hook, payload]
    assert result.reply == hostile.runs[host, hook, "no fields at all"].reply
    assert crashes(result) == []


def check_no_extraction(hostile: Hostile) -> None:
    """No capture in the matrix started an agent CLI."""
    if hostile.clis is None:
        pytest.skip(NO_FAKES)
    assert hostile.clis.calls("claude") == []
    assert hostile.clis.calls("codex") == []


#: The hostile payload that B64 is about.
DEEP = "nesting 100,000 levels deep"


def pin_deep_nesting(hostile: Hostile, host: str, hook: str) -> None:
    """B64: json.loads raises RecursionError on the payload, which neither
    plugin/hooks/lib/ipc.py, `payload`, nor core/envelope.py, `read_event`, catches, so the
    hook's body raises and only run.py's last guard keeps the exit code at 0."""
    if hook == "capture" and hostile.clis is None:
        pytest.skip(NO_FAKES)
    crashed = [line for line in crashes(hostile.runs[host, hook, DEEP])
               if "RecursionError" in line]
    if crashed:
        raise known_bugs.Reproduced(f"B64: {hook} on {host}: {crashed[0]}")
    check_read_as_empty(hostile, host, hook, DEEP)
