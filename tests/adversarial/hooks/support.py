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
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterator, Mapping, Sequence, TypeVar

from harness import stores
from harness.fakes.cli import FakeClis
from harness.hooks import HookResult, HookRunner

#: The hosts the plugin has a record for, in plugin/hooks/hosts.
HOSTS = ("claude", "codex", "copilot", "cursor", "opencode")

#: The user every store here belongs to.
USER = "tester"

#: The one memory the stores here hold, as recall renders it, and a prompt that the
#: hashing embedder matches with it above the recall hook's score floor.
MEMORY = "user lives in Lisbon"
PROMPT = "user lives in Lisbon"

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

#: One hook run, waiting to be made by `run_all`.
Job = Callable[[], HookResult]
Jobs = dict[tuple[str, str], Job]

K = TypeVar("K")


def tool_name(host: str, tool: str) -> str:
    """The first name `host` gives the memvara tool `tool` (TOOL_NAMES)."""
    return TOOL_NAMES[host][0].format(tool=tool)


def host_payload(host: str, hook: str, *, session: str, cwd: pathlib.Path,
                 **fields: Any) -> dict[str, Any]:
    """The stdin `host` sends for `hook`. `fields` are keys every host spells alike:
    prompt, tool_name, tool_input and transcript_path."""
    body: dict[str, Any] = {"hook_event_name": EVENTS[host][hook], "session_id": session}
    if host == "cursor":
        body["workspace_roots"] = [str(cwd)]
    else:
        body["cwd"] = str(cwd)
    body.update(EXTRA_KEYS[host])
    body.update(fields)
    return body


def host_json(host: str, hook: str, **options: Any) -> str:
    """`host_payload`, as the JSON text the host writes to the hook's stdin."""
    return json.dumps(host_payload(host, hook, **options))


def write_transcript(host: str, path: pathlib.Path,
                     turns: Sequence[tuple[str, str]]) -> pathlib.Path:
    """Write `turns`, each a (user, assistant) pair, as the transcript `host` keeps.

    Claude Code and OpenCode write JSONL with the speaker under `type`, and Cursor the
    same with it under `role`. Codex writes a rollout of `response_item` entries, and
    Copilot an event log. plugin/hooks/lib/transcript.py reads each one.
    """
    entries: list[dict[str, Any]] = []
    for user, assistant in turns:
        entries += _entries(host, user, assistant)
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries),
                    encoding="utf-8")
    return path


def _entries(host: str, user: str, assistant: str) -> list[dict[str, Any]]:
    if host == "codex":
        return [_codex_message("user", "input_text", user),
                _codex_message("assistant", "output_text", assistant)]
    if host == "copilot":
        return [{"type": "user.message", "data": {"content": user}},
                {"type": "assistant.message", "data": {"content": assistant}}]
    speaker = "role" if host == "cursor" else "type"
    return [{speaker: "user", "message": {"content": [{"type": "text", "text": user}]}},
            {speaker: "assistant",
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


def make_store(path: pathlib.Path, *, memory: bool = True) -> pathlib.Path:
    """A store file at `path`, holding MEMORY for USER unless `memory` is False."""
    with stores.file(path) as mem:
        if memory:
            mem.scope(user=USER).remember("user", "lives_in", "Lisbon")
    return path


def short_dir(prefix: str) -> pathlib.Path:
    """A new private directory with a short path.

    The recall daemon's socket lives under the hooks' home, and macOS refuses a unix
    socket path longer than 104 bytes. A pytest temporary directory under a long TMPDIR
    can pass that length before the hooks add their part, so homes come from a short base
    instead.
    """
    base = tempfile.gettempdir()
    if len(base) > 40 and os.path.isdir("/tmp"):
        base = "/tmp"
    return pathlib.Path(tempfile.mkdtemp(prefix=f"mv-{prefix}-", dir=base))


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


def status_of(host: str, reply: Mapping[str, Any] | None) -> str | None:
    """The status line a reply shows the person on `host`, or None when it shows none."""
    key = SHAPES[host][1]
    if key is None or reply is None:
        return None
    status = reply.get(key)
    return status if isinstance(status, str) else None
