"""The skip ledger: a skip whose reason no rule explains fails the run."""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

from harness.env import REPO, child_env
from harness.skips import RULES, explained, reason_of

_PLUGIN = '''
from harness.skips import SkipLedger


def pytest_configure(config):
    config.pluginmanager.register(SkipLedger(), "skip-ledger")
'''


def _run_one_skip(tmp_path: pathlib.Path, reason: str) -> subprocess.CompletedProcess[str]:
    """Run pytest in a child process on one test that skips with `reason`."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "conftest.py").write_text(_PLUGIN, encoding="utf-8")
    (project / "test_one.py").write_text(
        "import pytest\n\n\ndef test_skips():\n"
        f"    pytest.skip({reason!r})\n", encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir()
    env = child_env(home, {"PYTHONPATH": str(REPO / "tests")})
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                          cwd=project, env=env, capture_output=True, text=True, timeout=120)


def test_a_listed_reason_is_explained() -> None:
    assert explained("git is not installed")
    assert explained("could not import 'pypdf': No module named 'pypdf'")


def test_an_unlisted_reason_is_not_explained() -> None:
    assert not explained("flaky on Tuesdays")


def test_the_reason_is_read_without_the_prefix_pytest_adds() -> None:
    assert reason_of(("tests/test_x.py", 3, "Skipped: git is not installed")) == (
        "git is not installed")


def test_every_rule_says_why_its_skip_is_legitimate() -> None:
    for rule in RULES:
        assert rule.why.strip(), rule.pattern


def test_an_unexplained_skip_fails_the_run(tmp_path: pathlib.Path) -> None:
    done = _run_one_skip(tmp_path, "flaky on Tuesdays")
    assert done.returncode == 1, done.stdout + done.stderr
    assert "skips with no rule in tests/harness/skips.py" in done.stdout
    assert "flaky on Tuesdays" in done.stdout


def test_an_explained_skip_leaves_the_run_green(tmp_path: pathlib.Path) -> None:
    done = _run_one_skip(tmp_path, "git is not installed")
    assert done.returncode == 0, done.stdout + done.stderr


def test_a_windows_only_reason_is_not_explained_elsewhere() -> None:
    reason = "no POSIX permission bits to check"
    assert explained(reason, platform="win32", version=(3, 13))
    assert not explained(reason, platform="linux", version=(3, 13))


def test_a_version_bound_reason_is_explained_only_on_that_version() -> None:
    assert explained("tomllib arrived in 3.11", platform="linux", version=(3, 10))
    assert not explained("tomllib arrived in 3.11", platform="linux", version=(3, 12))
    assert explained("3.10 only", platform="linux", version=(3, 12))
    assert not explained("3.10 only", platform="linux", version=(3, 10))


def test_the_ledger_is_registered_for_every_run(request: pytest.FixtureRequest) -> None:
    assert request.config.pluginmanager.get_plugin("memvara-skip-ledger") is not None
