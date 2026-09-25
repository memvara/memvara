"""Fixtures shared by the adversarial suite."""

from __future__ import annotations

import pathlib
from typing import Any, Callable, Iterator

import pytest

from harness.hooks import HookRunner
from harness.stdio import McpProcess, kill_all


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
                tmp_path_factory: pytest.TempPathFactory) -> Callable[..., HookRunner]:
    """Build HookRunners that share one scratch home and one working directory."""
    home = tmp_path_factory.mktemp("hook-home")
    work = tmp_path / "work"
    work.mkdir()

    def make(host: str, **options: Any) -> HookRunner:
        return HookRunner(host, home=home, cwd=work, **options)

    return make
