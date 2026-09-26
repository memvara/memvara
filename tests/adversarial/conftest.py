"""Fixtures shared by the adversarial suite."""

from __future__ import annotations

import pathlib
import sys
from typing import Any, Callable, Iterator

import pytest

from harness.fakes.cli import NO_FAKES, FakeClis
from harness.hooks import HookRunner
from harness.stdio import McpProcess, kill_all


def pytest_configure(config: pytest.Config) -> None:
    """Register the covers mark, which tests/harness/checklist.py reads from each test's
    source. docs/claude/testing.md explains it."""
    config.addinivalue_line(
        "markers",
        "covers(*items): the checklist items this test covers, such as "
        "'tool:memory_recall' or 'inv:I3'. See docs/claude/testing.md.")


@pytest.fixture
def mcp(tmp_path: pathlib.Path,
        tmp_path_factory: pytest.TempPathFactory) -> Iterator[Callable[..., McpProcess]]:
    """Start real memvara MCP servers. Each one opens a new store file in tmp_path,
    unless the test passes `db` to make two servers share one.

    Every server this fixture started is killed when the test ends, before pytest deletes
    the temporary directory. On Windows, a file that a live process holds open cannot be
    deleted.
    """
    home = tmp_path_factory.mktemp("mcp-home")
    started: list[McpProcess] = []

    def start(db: pathlib.Path | None = None, **options: Any) -> McpProcess:
        server = McpProcess(db or tmp_path / f"memory-{len(started) + 1}.db", home=home,
                            **options)
        started.append(server)
        return server

    yield start
    kill_all(started)


@pytest.fixture
def hook_runner(tmp_path: pathlib.Path,
                tmp_path_factory: pytest.TempPathFactory) -> Iterator[Callable[..., HookRunner]]:
    """Build HookRunners that share one scratch home and one working directory.

    Every runner is closed when the test ends, which stops any recall daemon it allowed
    and any capture child it did not wait for; each would otherwise keep running.
    """
    home = tmp_path_factory.mktemp("hook-home")
    work = tmp_path / "work"
    work.mkdir()
    made: list[HookRunner] = []

    def make(host: str, **options: Any) -> HookRunner:
        runner = HookRunner(host, home=home, cwd=work, **options)
        made.append(runner)
        return runner

    yield make
    for runner in made:
        runner.close()


@pytest.fixture
def clis(tmp_path: pathlib.Path) -> FakeClis:
    """Fake `claude` and `codex` executables with nothing scripted yet. They are shell
    scripts, so a test that asks for them skips on Windows."""
    if sys.platform == "win32":
        pytest.skip(NO_FAKES)
    return FakeClis(tmp_path / "clis")
