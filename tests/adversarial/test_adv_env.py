"""The environment the adversarial suite gives every child process."""

from __future__ import annotations

import pathlib

import pytest

import memvara
from harness.env import REAL_HOME, REPO, child_env


def test_the_suite_imports_the_checkout_it_lives_in() -> None:
    """A worktree can import another copy of memvara through a stale editable install.

    Every result in this suite would then describe code that is not under test, so this
    fails first and says how to fix it.
    """
    imported = pathlib.Path(memvara.__file__).resolve()
    assert REPO in imported.parents, (
        f"memvara was imported from {imported}, not from this checkout ({REPO}). Run "
        "with PYTHONPATH set to the checkout, or from a virtual environment that has this "
        "checkout installed in editable mode. docs/claude/testing.md explains both.")


def test_a_child_never_gets_the_real_home_directory() -> None:
    with pytest.raises(ValueError, match="real home directory"):
        child_env(REAL_HOME)


def test_a_child_does_not_inherit_memvara_or_model_variables(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMVARA_API_KEY", "mv_must_not_reach_a_child")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-must-not-reach-a-child")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://example.invalid")
    monkeypatch.setenv("CLAUDECODE", "1")
    env = child_env(tmp_path)
    for name in ("MEMVARA_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_BASE_URL", "CLAUDECODE"):
        assert name not in env


def test_a_child_runs_this_checkout_offline_in_the_home_it_was_given(
        tmp_path: pathlib.Path) -> None:
    env = child_env(tmp_path, {"MEMVARA_USER": "tester"})
    assert env["HOME"] == env["USERPROFILE"] == str(tmp_path.resolve())
    assert env["PYTHONPATH"] == str(REPO)
    assert env["MEMVARA_EMBEDDER"] == "hashing"
    assert env["MEMVARA_FEATURE_ENCRYPTION"] == "0"
    assert env["MEMVARA_FEATURE_PROJECT_SCOPE"] == "0"
    assert env["MEMVARA_DAEMON"] == "1"
    assert env["MEMVARA_USER"] == "tester"
