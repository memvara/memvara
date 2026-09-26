"""The nightly run's steps: run in order, each stopped at its own time cap.

A night is a list of steps. Each step runs child processes, and a step that runs past its
cap is stopped together with every process it started, so one stuck step cannot eat the
night or leave servers running into the morning. A step whose code has not landed is
reported as "not built yet" with its reason rather than skipped in silence, and after an
essential step fails, the steps that depend on it are reported as "not run".

The fake steps here are small Python child processes, started with the suite's child
environment.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any, Callable

import pytest

from harness import env as harness_env
from harness.env import REPO, child_env

if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))
from nightly import night, steps  # noqa: E402 - scripts/ is not on the path until above


@pytest.fixture
def child(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """The environment every fake step's child process runs with."""
    return child_env(tmp_path_factory.mktemp("home"))


def _python_step(name: str, code: str, env: dict[str, str], logs: pathlib.Path, *,
                 cap: float = 30.0, essential: bool = False,
                 args: tuple[str, ...] = ()) -> steps.Step:
    """A step that runs `code` in a child Python and passes when it exits 0."""
    def run(context: Any, deadline: float) -> steps.Outcome:
        result = steps.run_command([sys.executable, "-c", code, *args], cwd=logs, env=env,
                                   log=logs / f"{name}.log", deadline=deadline)
        if result.timed_out:
            return steps.Outcome(steps.TIMED_OUT, "stopped at its cap")
        status = steps.PASSED if result.returncode == 0 else steps.FAILED
        return steps.Outcome(status, f"exited {result.returncode}")
    return steps.Step(name, cap, run, essential=essential)


def _append(name: str) -> str:
    """Code that appends two lines to the file named by its first argument."""
    return ("import sys\n"
            "with open(sys.argv[1], 'a') as out:\n"
            f"    out.write('{name} start\\n')\n"
            f"    out.write('{name} end\\n')\n")


def test_steps_run_one_after_another_in_the_order_given(
        tmp_path: pathlib.Path, child: dict[str, str]) -> None:
    """A step that started before the one ahead of it ended would share the machine and
    the store with it, and its time would include the other step's."""
    order = tmp_path / "order.txt"
    table = [_python_step(name, _append(name), child, tmp_path, args=(str(order),))
             for name in ("preflight", "regressions", "soak")]
    results = steps.run_steps(table, context=None)
    assert [(result.name, result.status) for result in results] == [
        ("preflight", "passed"), ("regressions", "passed"), ("soak", "passed")]
    assert order.read_text().splitlines() == [
        "preflight start", "preflight end", "regressions start", "regressions end",
        "soak start", "soak end"]


def _alive(pid: int) -> bool:
    """Whether a process is still running, without sending it anything that stops it."""
    if sys.platform == "win32":
        import ctypes  # noqa: PLC0415 - Windows only

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    stat = pathlib.Path(f"/proc/{pid}/stat")
    try:
        if stat.read_text().rsplit(")", 1)[1].split()[0] == "Z":
            return False  # a zombie has exited and waits only to be reaped
    except (OSError, IndexError):
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _poll(condition: Callable[[], bool], seconds: float) -> bool:
    """Whether `condition` becomes true within `seconds`, checking every 50 ms."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.05)
    return condition()


#: A step that starts a long-lived grandchild, records its process id, and then hangs.
HANGS_WITH_A_GRANDCHILD = (
    "import pathlib, subprocess, sys, time\n"
    "grandchild = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
    "pathlib.Path(sys.argv[1]).write_text(str(grandchild.pid))\n"
    "print('grandchild started', flush=True)\n"
    "time.sleep(300)\n")


def test_a_step_past_its_cap_is_stopped_with_every_process_it_started(
        tmp_path: pathlib.Path, child: dict[str, str]) -> None:
    """A stuck step must not eat the night: it is stopped at its cap, its output so far is
    kept, and the next step still runs. Everything it started goes too; a server or a
    daemon left behind would hold the store open and run into the morning."""
    pid_file = tmp_path / "grandchild.pid"
    table = [_python_step("stuck", HANGS_WITH_A_GRANDCHILD, child, tmp_path, cap=4.0,
                          args=(str(pid_file),)),
             _python_step("next", "pass", child, tmp_path)]
    started = time.monotonic()
    results = steps.run_steps(table, context=None)
    assert time.monotonic() - started < 60
    assert [(result.name, result.status) for result in results] == [
        ("stuck", "timed out"), ("next", "passed")]
    assert results[0].seconds < 60
    assert "grandchild started" in (tmp_path / "stuck.log").read_text()
    grandchild = int(pid_file.read_text())
    assert _poll(lambda: not _alive(grandchild), 10), (
        f"process {grandchild}, started by a step that was stopped at its cap, is still "
        "running")


def test_a_step_that_is_not_built_says_why(tmp_path: pathlib.Path) -> None:
    """A step that cannot run yet must be visible in the report, with the reason, and the
    report must notice when the code the step waits for has landed but nobody has wired
    it into the run."""
    step = steps.Step("soak", 1200.0, waits_for="bench/soak.py",
                      not_built="The soak has not landed on main.")
    first = steps.run_steps([step], context=None, worktree=tmp_path)
    assert (first[0].status, first[0].summary) == (
        "not built yet", "The soak has not landed on main.")
    (tmp_path / "bench").mkdir()
    (tmp_path / "bench" / "soak.py").write_text("")
    second = steps.run_steps([step], context=None, worktree=tmp_path)
    assert second[0].status == "not built yet"
    assert "bench/soak.py has landed" in second[0].summary
    assert "scripts/nightly/run.py" in second[0].summary


def test_the_worktree_a_step_waits_in_can_be_created_by_an_earlier_step(
        tmp_path: pathlib.Path) -> None:
    """Preflight creates the worktree, so the check for a step's file must look at the
    worktree when that step is reached, not when the night began."""
    created: dict[str, pathlib.Path] = {}

    def preflight(context: Any, deadline: float) -> steps.Outcome:
        (tmp_path / "bench").mkdir()
        (tmp_path / "bench" / "soak.py").write_text("")
        created["worktree"] = tmp_path
        return steps.Outcome(steps.PASSED)

    results = steps.run_steps(
        [steps.Step("preflight", 60.0, preflight),
         steps.Step("soak", 60.0, waits_for="bench/soak.py", not_built="not landed")],
        context=None, worktree=lambda: created.get("worktree"))
    assert "bench/soak.py has landed" in results[1].summary


def test_after_an_essential_step_fails_the_steps_after_it_are_not_run(
        tmp_path: pathlib.Path, child: dict[str, str]) -> None:
    """With no worktree or no virtual environment, every later step would fail for the
    same reason and bury it, so they are reported as not run, naming the step that
    failed. A step that is not built stays "not built yet", which is still true."""
    table = [_python_step("preflight", "raise SystemExit(3)", child, tmp_path,
                          essential=True),
             _python_step("regressions", "pass", child, tmp_path),
             steps.Step("agents", 60.0, not_built="The agent layer has not landed.")]
    results = steps.run_steps(table, context=None)
    assert [(result.name, result.status) for result in results] == [
        ("preflight", "failed"), ("regressions", "not run"), ("agents", "not built yet")]
    assert "preflight" in results[1].summary
    assert results[2].summary == "The agent layer has not landed."


def test_a_step_that_raises_is_an_error_and_the_night_goes_on(
        tmp_path: pathlib.Path, child: dict[str, str]) -> None:
    """A bug in one step's code must cost that step, not the night's report."""
    def broken(context: Any, deadline: float) -> steps.Outcome:
        raise RuntimeError("could not read the results\nthe file was empty")

    table = [steps.Step("performance", 60.0, broken),
             _python_step("soak", "pass", child, tmp_path)]
    results = steps.run_steps(table, context=None)
    assert [(result.name, result.status) for result in results] == [
        ("performance", "error"), ("soak", "passed")]
    assert results[0].summary == "RuntimeError: the file was empty"


def test_a_step_that_returns_after_its_deadline_is_timed_out() -> None:
    """Work a step does in the runner's own process cannot be killed at the cap, so a step
    that returns late is reported as timed out even though it says it passed."""
    ticks = [100.0, 175.0]

    def clock() -> float:
        return ticks.pop(0) if len(ticks) > 1 else ticks[0]

    def slow(context: Any, deadline: float) -> steps.Outcome:
        assert deadline == 160.0
        return steps.Outcome(steps.PASSED, "done")

    results = steps.run_steps([steps.Step("hosted", 60.0, slow)], context=None, clock=clock)
    assert (results[0].status, results[0].seconds) == ("timed out", 75.0)


def test_the_steps_environment_drops_what_the_harness_drops_and_sets_no_memvara_variable(
        tmp_path: pathlib.Path) -> None:
    """The run is started from an agent session and from a shell that may hold real keys.
    None of that may reach the suite, and the suite must run as it does in CI, with no
    MEMVARA_ variable set, since several tests read the defaults."""
    base = {prefix + "SOMETHING": "secret" for prefix in harness_env._DROPPED}
    base.update({"KEEP_ME": "1", "PATH": os.pathsep.join(["/usr/bin", "/bin"])})
    home, tmp, worktree, bin_dir = (tmp_path / name for name in ("home", "tmp", "wt", "bin"))
    result = night.step_env(base, worktree=worktree, home=home, tmp=tmp, bin_dir=bin_dir)
    assert not [key for key in result if key.startswith(harness_env._DROPPED)]
    assert result["KEEP_ME"] == "1"
    assert (result["HOME"], result["USERPROFILE"]) == (str(home), str(home))
    assert result["TMPDIR"] == result["TMP"] == result["TEMP"] == str(tmp)
    assert result["PYTHONPATH"] == str(worktree)
    assert result["PATH"].split(os.pathsep) == [str(bin_dir), "/usr/bin", "/bin"]


def test_a_history_line_torn_by_a_killed_run_is_skipped_and_reported(
        tmp_path: pathlib.Path) -> None:
    """A run killed while it appended to the history leaves half a line. Every later night
    reads the history, so that line must cost itself only, and be reported."""
    path = tmp_path / "history.jsonl"
    night.append_jsonl(path, {"date": "2026-09-26"})
    night.append_jsonl(path, {"date": "2026-09-27"})
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"date": "2026-09-2')
    records, bad = night.read_jsonl(path)
    assert records == [{"date": "2026-09-26"}, {"date": "2026-09-27"}]
    assert bad == [3]
    assert night.read_jsonl(tmp_path / "missing.jsonl") == ([], [])


def test_the_heartbeat_is_readable_json_with_its_start_and_finish(
        tmp_path: pathlib.Path) -> None:
    path = tmp_path / "heartbeat.json"
    night.write_heartbeat(path, night="2026-09-27", started_at="2026-09-27T01:30:00+02:00")
    assert json.loads(path.read_text()) == {
        "night": "2026-09-27", "started_at": "2026-09-27T01:30:00+02:00",
        "finished_at": None, "status": "running"}
    night.write_heartbeat(path, night="2026-09-27", started_at="2026-09-27T01:30:00+02:00",
                          finished_at="2026-09-27T02:41:00+02:00", status="finished")
    assert night.read_json(path)["finished_at"] == "2026-09-27T02:41:00+02:00"


def _git(*args: str, cwd: pathlib.Path, env: dict[str, str]) -> str:
    return subprocess.run(["git", "-c", "user.name=nightly test",
                           "-c", "user.email=nightly@example.invalid", *args],
                          cwd=cwd, env=env, check=True, capture_output=True,
                          text=True).stdout.strip()


def test_a_run_started_in_a_worktree_writes_to_the_main_checkout(
        tmp_path: pathlib.Path, child: dict[str, str]) -> None:
    """The reports belong to the main checkout's local/ folder whichever worktree the run
    was started from, or they would be scattered and deleted with the worktrees."""
    main = tmp_path / "main"
    main.mkdir()
    _git("init", "-q", cwd=main, env=child)
    _git("commit", "-q", "--allow-empty", "-m", "first", cwd=main, env=child)
    _git("worktree", "add", "-q", "--detach", str(tmp_path / "other"), cwd=main, env=child)
    assert night.main_checkout(tmp_path / "other") == main.resolve()
    assert night.main_checkout(main) == main.resolve()
