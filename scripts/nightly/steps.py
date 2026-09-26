"""Steps: the parts of a night, each run to its own time cap.

A step's code gets the instant its cap ends and returns an `Outcome`. The commands it runs
go through `run_command`, which stops a command at the cap together with every process it
started, so one stuck step cannot eat the night or leave a server running into the
morning. `run_steps` runs the steps in order and reports each one, including the steps
whose code has not landed yet, so nothing is left out of a report without a reason.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

PASSED = "passed"
FAILED = "failed"
TIMED_OUT = "timed out"
NOT_BUILT = "not built yet"
NOT_RUN = "not run"
ERROR = "error"


@dataclass(frozen=True)
class Outcome:
    """What a step's code reports: a status and one line for the report."""

    status: str
    summary: str = ""


@dataclass(frozen=True)
class Step:
    """One part of a night."""

    name: str
    #: Seconds the step may take.
    cap: float
    #: The step's code, called as run(context, deadline), where the deadline is the
    #: `time.monotonic()` instant the cap ends. None when the code has not landed.
    run: Callable[[Any, float], Outcome] | None = None
    #: A path in the tested checkout whose arrival means the step can be wired in.
    waits_for: str = ""
    #: Why the step cannot run yet, for the report.
    not_built: str = ""
    #: When this step does not pass, the steps after it cannot run.
    essential: bool = False


@dataclass(frozen=True)
class StepResult:
    name: str
    status: str
    seconds: float
    cap: float
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CommandResult:
    #: The command's exit code, or None when it was stopped at the cap.
    returncode: int | None
    timed_out: bool
    seconds: float


def run_command(argv: Sequence[str], *, cwd: Path | str, env: dict[str, str], log: Path,
                deadline: float, clock: Callable[[], float] = time.monotonic
                ) -> CommandResult:
    """Run one command until it exits or `deadline` passes, with its output appended to
    `log` and its standard input closed.

    The command starts a process group of its own. At the deadline the whole group is
    stopped, not only the command, and on POSIX whatever is left of the group is stopped
    when the command exits too, so a test server or daemon the command leaked cannot
    outlive its step.
    """
    start = clock()
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as out:
        out.write(f"$ {' '.join(argv)}\n".encode())
        out.flush()
        if deadline - start <= 0:
            out.write(b"not started: the step's cap had already been reached\n")
            return CommandResult(None, True, 0.0)
        options: dict[str, Any] = {}
        if sys.platform == "win32":
            options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            options["start_new_session"] = True
        proc = subprocess.Popen(list(argv), cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                stdout=out, stderr=subprocess.STDOUT, **options)
        try:
            returncode: int | None = proc.wait(timeout=deadline - start)
            timed_out = False
        except subprocess.TimeoutExpired:
            kill_tree(proc)
            returncode, timed_out = None, True
            out.write(b"\nstopped: the step's cap was reached\n")
        except BaseException:
            kill_tree(proc)
            raise
        if sys.platform != "win32":
            _kill_group(proc.pid)
    return CommandResult(returncode, timed_out, clock() - start)


def kill_tree(proc: subprocess.Popen[bytes]) -> None:
    """Stop `proc` and every process it started, and wait for `proc` to end."""
    if sys.platform == "win32":
        # taskkill finds the children by their parent's id, so it runs before the parent
        # is gone; proc.kill() is the fallback for the parent alone.
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if proc.poll() is None:
            proc.kill()
    else:
        _kill_group(proc.pid)
    proc.wait()


def _kill_group(group: int) -> None:
    """Stop every process left in the POSIX process group `group`, if any."""
    if sys.platform != "win32":
        try:
            os.killpg(group, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def run_steps(steps: Sequence[Step], context: Any, *, worktree: Path | None = None,
              clock: Callable[[], float] = time.monotonic) -> list[StepResult]:
    """Run `steps` in order and report each one.

    A step with no code is "not built yet", with its reason. After an essential step that
    did not pass, every later step with code is "not run", naming that step. A step that
    raises is an "error", and the night goes on. A step that returns after its cap ended
    is "timed out", because work in this process cannot be stopped at the cap.
    """
    results: list[StepResult] = []
    blocked_by: str | None = None
    for step in steps:
        if step.run is None:
            results.append(StepResult(step.name, NOT_BUILT, 0.0, step.cap,
                                      _not_built(step, worktree)))
            continue
        if blocked_by is not None:
            results.append(StepResult(
                step.name, NOT_RUN, 0.0, step.cap,
                f"Not run, because the {blocked_by} step did not pass."))
            continue
        start = clock()
        deadline = start + step.cap
        try:
            outcome = step.run(context, deadline)
        except Exception as exc:  # noqa: BLE001 - one step's bug must not end the night
            traceback.print_exc()
            outcome = Outcome(ERROR, _last_line(exc))
        end = clock()
        status, summary = outcome.status, outcome.summary
        if status in (PASSED, FAILED) and end > deadline:
            status = TIMED_OUT
            summary = (f"{summary} The step returned {end - deadline:.0f} seconds after "
                       "its cap.").strip()
        results.append(StepResult(step.name, status, end - start, step.cap, summary))
        if step.essential and status != PASSED:
            blocked_by = step.name
    return results


def _not_built(step: Step, worktree: Path | None) -> str:
    if step.waits_for and worktree is not None and (worktree / step.waits_for).exists():
        return (f"{step.waits_for} has landed, but the nightly run does not start it yet: "
                "add its command to STEPS in scripts/nightly/run.py.")
    return step.not_built


def _last_line(exc: BaseException) -> str:
    lines = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: {lines[-1]}" if lines else type(exc).__name__
