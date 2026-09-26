"""The hooks give up in time.

Each host kills a hook that runs past its limit (`support.LIMITS`): session start 20
seconds, recall 10, approve 5, and capture 180 on Claude Code and 120 elsewhere. A hook
killed there prints nothing at all, so each hook has to keep its own work inside the
limit. These tests check that without waiting a real limit out: they read the budgets the
hooks declare, and they shrink a limit inside the hook process (`HookRunner(patches=...)`)
and put an agent CLI that never answers behind it. The nightly tier waits recall's real
7.5-second budget out.

Approve reads no store and starts no program, so it has no slow backend to put behind
it. Every approve run in this suite finishes inside its 5 seconds, because HookRunner
raises HookTimeout when a hook runs past its host's limit.
"""

from __future__ import annotations

import importlib
import os
import pathlib
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace
from typing import Callable, cast

import pytest

from harness.env import child_env
from harness.fakes.cli import NO_FAKES, HangingClis
from harness.hooks import HOOKS_DIR, HookResult, HookRunner, host_record, process_alive

from . import support

Make = Callable[..., HookRunner]


def _hook_module(name: str) -> ModuleType:
    """A module of plugin/hooks, imported to read the limits it declares."""
    if str(HOOKS_DIR) not in sys.path:
        sys.path.insert(0, str(HOOKS_DIR))
    return importlib.import_module(name)


@pytest.mark.parametrize("host", support.HOSTS)
def test_every_hook_has_the_limit_the_contract_names(host: str) -> None:
    assert dict(host_record(host).timeouts) == support.LIMITS[host]


def test_recall_stops_its_optional_work_early_enough_to_wait_for_a_rewrite() -> None:
    """Recall starts no optional work, a standing refresh or a wider second read, after
    7.5 seconds, which leaves 2.5 of its 10 for the read it must make. A query rewrite
    may take 5 seconds, so it is started only while that much budget is left."""
    recall, fast = _hook_module("recall"), _hook_module("lib.fast")
    assert recall.OVERALL_BUDGET_SEC == 7.5
    assert fast.REWRITE_WAIT_SEC < recall.OVERALL_BUDGET_SEC < support.LIMITS["claude"]["recall"]


def test_recall_skips_its_optional_work_once_its_budget_is_spent_and_still_answers(
        hooks: Make, store_env: dict[str, str]) -> None:
    runner = hooks("claude", env=store_env, patches={"recall.OVERALL_BUDGET_SEC": 0.0})
    found = runner.run("recall", session="one", prompt=support.PROMPT)
    assert support.MEMORY in support.context_of("claude", found.reply)
    assert "skipped=standing refresh, budget exhausted" in found.log("recall")
    nothing = runner.run("recall", session="two", prompt=support.UNRELATED)
    assert support.status_of("claude", nothing.reply) == (
        "⋈ Memvara · no matching memories")
    assert "skipped=episode widen, budget exhausted" in nothing.log("recall")


@pytest.mark.parametrize("host", support.HOSTS)
def test_the_capture_limit_covers_every_extraction_a_capture_can_run(host: str) -> None:
    """On Claude Code, capture first runs the agentic extraction, then, if that fails,
    the single call. lib/agentic.py, `_run`, waits up to 10 more seconds for a run it
    has killed. Every other host runs the single call alone."""
    agentic, extract = _hook_module("lib.agentic"), _hook_module("lib.extract")
    worst = extract.TIMEOUT_SEC + (agentic.TIMEOUT_SEC + 10 if host == "claude" else 0)
    assert worst <= support.LIMITS[host]["capture"]


def test_capture_gives_up_on_an_extractor_that_never_answers_and_says_so(
        hooks: Make, tmp_path: pathlib.Path) -> None:
    """With both extraction limits shrunk to a second, a `claude` that never answers
    costs capture about two seconds. It logs both timeouts, and the next prompt's status
    line carries the capture alert, which is how a person learns capture is failing."""
    if sys.platform == "win32":
        pytest.skip(NO_FAKES)
    env = support.store_env(support.make_store(tmp_path / "capture.db", memory=False))
    runner = hooks("claude", env=env, stubs=HangingClis(tmp_path / "hanging"),
                   patches={"lib.agentic.TIMEOUT_SEC": 1.0, "lib.extract.TIMEOUT_SEC": 1.0})
    transcript = support.write_transcript("claude", tmp_path / "t.jsonl",
                                          [(support.USER_TURN, support.ASSISTANT_TURN)])
    result = runner.run("capture", session="s", transcript_path=str(transcript), timeout=8)
    assert result.exit_code == 0
    log = result.log("capture")
    assert "agentic capture fell back to single-call extraction: no reply within 1.0s" in log
    assert "extraction did not run via claude: no reply within 1.0s" in log
    after = hooks("claude", home=runner.home, env=env).run("recall", session="s",
                                                             prompt=support.PROMPT)
    status = support.status_of("claude", after.reply)
    assert status is not None and status.endswith("capture failing: no reply within 1.0s")


def test_capture_on_codex_frees_the_turn_while_its_extractor_hangs(
        hooks: Make, tmp_path: pathlib.Path) -> None:
    support.check_capture_frees_the_turn(hooks, tmp_path, "codex")


def test_the_check_that_capture_frees_the_turn_fails_when_the_child_has_ended(
        tmp_path: pathlib.Path) -> None:
    """support.check_capture_frees_the_turn must see that the child it was handed still
    runs. A child that has ended, but that nothing has reaped yet, still answers signal 0,
    so here the capture hands the turn to such a child, and the check must fail."""
    if sys.platform == "win32":
        pytest.skip(NO_FAKES)
    child = subprocess.Popen([sys.executable, "-c", "pass"], env=child_env(tmp_path))
    try:
        deadline = time.monotonic() + 10
        while process_alive(child.pid):
            assert time.monotonic() < deadline, "the child did not end"
            time.sleep(0.02)
        os.kill(child.pid, 0)  # it has ended, and it still answers signal 0
        ended = HookResult(exit_code=0, stdout="", stderr="", reply=None, elapsed=0.1,
                           detached_pid=child.pid)
        runner = cast(HookRunner, SimpleNamespace(run=lambda hook, **options: ended))
        with pytest.raises(AssertionError, match="had already ended"):
            support.check_capture_frees_the_turn(lambda host, **options: runner, tmp_path,
                                                 "codex")
    finally:
        child.wait()
