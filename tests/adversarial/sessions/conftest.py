"""Fixtures for the scripted sessions and the tool-surface tests."""

from __future__ import annotations

import pathlib
from typing import Callable

import pytest

from harness.stdio import McpProcess

from . import switches


@pytest.fixture(scope="session")
def surface_template(tmp_path_factory: pytest.TempPathFactory) -> switches.Template:
    """The seeded store every real server of the tool-surface tests starts from. It is
    built once per run and copied for each server, so no run sees another's writes."""
    return switches.Template.build(tmp_path_factory.mktemp("surface-template"))


@pytest.fixture
def check_real_server(mcp: Callable[..., McpProcess], surface_template: switches.Template,
                      tmp_path: pathlib.Path) -> Callable[[switches.Combination], None]:
    """The one check the fast, nightly and weekly real-server tests share: start a real
    server with a combination, on a copy of the template store, and fail with every
    problem `switches.over_the_pipe` finds."""
    def check(combination: switches.Combination) -> None:
        problems = switches.over_the_pipe(mcp, combination, surface_template, tmp_path)
        assert problems == [], "\n".join(problems)

    return check
