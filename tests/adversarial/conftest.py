"""Fixtures shared by the adversarial suite."""

from __future__ import annotations

import pathlib
from typing import Any, Callable, Iterator

import pytest

from harness.stdio import McpProcess


@pytest.fixture
def mcp(tmp_path: pathlib.Path,
        tmp_path_factory: pytest.TempPathFactory) -> Iterator[Callable[..., McpProcess]]:
    """Start real memvara MCP servers. By default each one opens memory.db in tmp_path.

    Every server this fixture started is killed when the test ends, before pytest deletes
    the temporary directory. On Windows, a file that a live process holds open cannot be
    deleted.
    """
    home = tmp_path_factory.mktemp("mcp-home")
    started: list[McpProcess] = []

    def start(db: pathlib.Path | None = None, **options: Any) -> McpProcess:
        server = McpProcess(db or tmp_path / "memory.db", home=home, **options)
        started.append(server)
        return server

    yield start
    for server in started:
        server.kill()
