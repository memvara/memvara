"""The weekly tier's real servers: every one of the 2,048 combinations
(`switches.every_combination`). The whole file took 7 minutes 54 seconds on a laptop that
was running other tests at the same time."""

from __future__ import annotations

import pathlib
from typing import Callable

import pytest

from harness.stdio import McpProcess

from .. import switches


@pytest.mark.parametrize("combination", switches.every_combination(),
                         ids=lambda c: c.label)
def test_every_combination_on_a_real_server(
        combination: switches.Combination, mcp: Callable[..., McpProcess],
        surface_template: switches.Template, tmp_path: pathlib.Path) -> None:
    assert switches.over_the_pipe(mcp, combination, surface_template, tmp_path) == []
