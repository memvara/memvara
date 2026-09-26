"""Fake `claude` and `codex` executables for the plugin's capture hook.

The capture hook mines a turn by starting a headless agent CLI that the user is already
signed in to (`plugin/hooks/lib/extract.py` and `lib/agentic.py`). It finds the CLI on
`PATH`, passes the prompt as the last argument, and reads what the CLI prints:

* `claude -p ... --output-format json <prompt>` prints one JSON object, and the hook reads
  its `result`, `usage` and `is_error` (`CLAUDE_CLI` in `core/host.py`).
* `claude -p ... --output-format stream-json --verbose ... <data>`, the agentic run,
  prints one JSON event per line: an `init` event that names the connected MCP servers and
  the tools, then the run, then a `result` event (`_Watch` in `lib/agentic.py`).
* `codex exec --skip-git-repo-check --json <prompt>` prints one JSON event per line, and
  the hook reads the text of the `item.completed` event whose item is an `agent_message`,
  and the `usage` of the `turn.completed` event (`CODEX_CLI` in `core/host.py`).

`FakeClis(directory)` writes a `claude` and a `codex` into `directory`. Put `path()` in a
child process's environment as `PATH`, and a hook in that process starts the fakes instead
of the real CLIs. Each run takes the next reply from its script, prints it in the format
its arguments ask for, and appends its arguments and its stdin to a log that `calls()`
reads. Scripting again starts over from the first new reply, and the log keeps every run.
A run that finds no reply left says so on stderr and exits with status 3, so a hook that
starts a CLI once more than the test expected fails loudly.

A run reads its stdin to the end before it answers, unless stdin is a terminal. So a
caller that starts a fake with stdin left open, and never closes it, holds the run until
its own timeout.

`HangingClis(directory)` writes a `claude` and a `codex` that never answer, for a test of
what the capture hook does when its extractor hangs. It has the same `path()`.

POSIX only. The executables are shell scripts, and on Windows a program that another
starts without a shell is found on `PATH` only as an `.exe`. A test that needs them skips
there with the reason `NO_FAKES`, which has a rule in tests/harness/skips.py.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import shlex
import stat
import sys
from typing import Any, Mapping

#: The CLIs the capture hook can start.
NAMES = ("claude", "codex")

#: The token counts a reply reports when the test names none.
USAGE: Mapping[str, int] = {"input_tokens": 10, "output_tokens": 5}

#: Exit status of a run that found no scripted reply left.
EXHAUSTED = 3

#: Why a test that needs these executables skips on Windows. The rule for it in
#: tests/harness/skips.py matches these words exactly.
NO_FAKES = "the fake agent CLIs are POSIX shell scripts"


@dataclasses.dataclass(frozen=True)
class CliReply:
    """What one run of a fake CLI prints."""

    #: What the model answered. The fake wraps it in the format the arguments ask for.
    text: str = ""
    #: When set, printed exactly as given instead of any envelope, for output the hook
    #: cannot parse.
    stdout: str | None = None
    stderr: str = ""
    exit_code: int = 0
    #: For `claude`, the envelope's `is_error` flag. For `codex`, a `turn.failed` event is
    #: printed instead of the reply.
    is_error: bool = False
    usage: Mapping[str, int] = dataclasses.field(default_factory=lambda: dict(USAGE))
    #: `claude --output-format stream-json` only: the events to print instead of the
    #: default `init`, `assistant` and `result`, for an agentic run that calls tools.
    events: tuple[Mapping[str, Any], ...] | None = None


@dataclasses.dataclass(frozen=True)
class CliCall:
    """One run of a fake CLI: its arguments without the program name, and its stdin."""

    argv: list[str]
    stdin: str


class _Executables:
    """Executables written into one directory, for a child process to find first on
    `PATH`. They are shell scripts, so they are refused on Windows."""

    def __init__(self, directory: pathlib.Path) -> None:
        if sys.platform == "win32":
            raise NotImplementedError(
                f"{NO_FAKES}, and on Windows a program started without a shell is found on "
                "PATH only as an .exe")
        #: The directory to put first on `PATH`.
        self.bin = pathlib.Path(directory)
        self.bin.mkdir(parents=True, exist_ok=True)

    def path(self, rest: str | None = None) -> str:
        """A `PATH` value with these executables first. `rest` follows them, and defaults
        to this process's own `PATH`, so a child still finds everything else it runs."""
        rest = os.environ.get("PATH", "") if rest is None else rest
        return os.pathsep.join(part for part in (str(self.bin), rest) if part)


class HangingClis(_Executables):
    """A `claude` and a `codex` that never answer: each sleeps until it is killed, or for
    two minutes."""

    def __init__(self, directory: pathlib.Path) -> None:
        super().__init__(directory)
        for name in NAMES:
            script = self.bin / name
            script.write_text("#!/bin/sh\nexec sleep 120\n", encoding="utf-8")
            script.chmod(0o755)


class FakeClis(_Executables):
    """A `claude` and a `codex` in one directory, each with its own script and log."""

    def __init__(self, directory: pathlib.Path) -> None:
        super().__init__(directory)
        runner = self.bin / "_fake_cli.py"
        runner.write_text(_RUNNER, encoding="utf-8")
        for name in NAMES:
            script = self.bin / name
            # `-I` runs the runner isolated from the child's PYTHON* variables and user
            # site, so what the hook's environment holds cannot change what it does.
            script.write_text(f"#!/bin/sh\nexec {shlex.quote(sys.executable)} -I "
                              f"{shlex.quote(str(runner))} {name} \"$@\"\n",
                              encoding="utf-8")
            script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            self.script(name)

    def script(self, name: str, *replies: CliReply | str) -> None:
        """Replace what `name` answers with `replies`, one per run, in order, starting
        with the next run. A plain string is a reply with that text. With no replies,
        every run fails as exhausted. The call log keeps the runs made before."""
        self._check(name)
        queue = [dataclasses.asdict(r if isinstance(r, CliReply) else CliReply(text=r))
                 for r in replies]
        (self.bin / f"{name}.replies.json").write_text(json.dumps(queue), encoding="utf-8")
        # The position of the next reply, which each run reads and advances.
        (self.bin / f"{name}.next").write_text("0", encoding="utf-8")

    def calls(self, name: str) -> list[CliCall]:
        """Every run of `name` so far, in order."""
        self._check(name)
        log = self.bin / f"{name}.calls.jsonl"
        if not log.exists():
            return []
        return [CliCall(**json.loads(line))
                for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _check(self, name: str) -> None:
        if name not in NAMES:
            raise ValueError(f"there is no fake {name!r}; the fakes are {', '.join(NAMES)}")


#: The program both executables run. Standard library only, so it runs under any Python.
_RUNNER = r'''"""A fake agent CLI for the memvara test suite, written by tests/harness/fakes/cli.py."""
import fcntl
import json
import os
import sys

name, argv = sys.argv[1], sys.argv[2:]
here = os.path.dirname(os.path.abspath(__file__))
stdin = "" if sys.stdin is None or sys.stdin.isatty() else sys.stdin.read()

# The position of the next reply is read and advanced, and the call appended to the log,
# under one lock, so two runs at once each get their own reply. `script()` resets the
# position when it replaces the replies.
with open(os.path.join(here, name + ".calls.jsonl"), "a", encoding="utf-8") as log:
    fcntl.flock(log, fcntl.LOCK_EX)
    position = os.path.join(here, name + ".next")
    with open(position, encoding="utf-8") as fh:
        number = int(fh.read())
    with open(position, "w", encoding="utf-8") as fh:
        fh.write(str(number + 1))
    log.write(json.dumps({"argv": argv, "stdin": stdin}) + "\n")
    log.flush()

with open(os.path.join(here, name + ".replies.json"), encoding="utf-8") as script:
    replies = json.load(script)
if number >= len(replies):
    sys.stderr.write(f"fake {name}: no scripted reply left for run {number + 1}\n")
    sys.exit(3)
reply = replies[number]


def emit(event):
    sys.stdout.write(json.dumps(event) + "\n")


def option(flag):
    return argv[argv.index(flag) + 1] if flag in argv[:-1] else None


if reply["stdout"] is not None:
    sys.stdout.write(reply["stdout"])
elif name == "codex":
    emit({"type": "thread.started", "thread_id": "fake-thread"})
    emit({"type": "turn.started"})
    if reply["is_error"]:
        emit({"type": "turn.failed", "error": {"message": reply["text"]}})
    else:
        emit({"type": "item.completed",
              "item": {"id": "item_0", "type": "agent_message", "text": reply["text"]}})
        emit({"type": "turn.completed", "usage": reply["usage"]})
elif option("--output-format") == "stream-json":
    events = reply["events"]
    if events is None:
        events = [
            {"type": "system", "subtype": "init", "session_id": "fake-session",
             "mcp_servers": [{"name": "memvara", "status": "connected"}],
             "tools": ["mcp__memvara__memory_search", "mcp__memvara__memory_recall",
                       "mcp__memvara__memory_why", "mcp__memvara__memory_profile"]},
            {"type": "assistant",
             "message": {"id": "msg_fake", "role": "assistant", "usage": reply["usage"],
                         "content": [{"type": "text", "text": reply["text"]}]}},
            {"type": "result",
             "subtype": "error_during_execution" if reply["is_error"] else "success",
             "is_error": reply["is_error"], "result": reply["text"],
             "usage": reply["usage"]},
        ]
    for event in events:
        emit(event)
else:
    emit({"type": "result",
          "subtype": "error_during_execution" if reply["is_error"] else "success",
          "is_error": reply["is_error"], "result": reply["text"], "usage": reply["usage"],
          "session_id": "fake-session"})
sys.stdout.flush()
sys.stderr.write(reply["stderr"])
sys.exit(reply["exit_code"])
'''

__all__ = ["CliCall", "CliReply", "EXHAUSTED", "FakeClis", "HangingClis", "NAMES",
           "NO_FAKES", "USAGE"]
