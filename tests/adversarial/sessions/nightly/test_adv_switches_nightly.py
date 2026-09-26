"""The nightly tier's real servers: the fold-over of the fast tier's array, the same 12
runs with every setting reversed (`switches.nightly_runs`). The nightly run also collects
the fast tier's runs, and with them every combination of every three settings is started
exactly three times."""

from __future__ import annotations

import pathlib
from typing import Callable

import pytest

from harness.stdio import McpProcess

from .. import switches


@pytest.mark.parametrize("combination", switches.nightly_runs(), ids=lambda c: c.label)
def test_a_real_server_serves_what_the_oracle_predicts_nightly(
        combination: switches.Combination, mcp: Callable[..., McpProcess],
        surface_template: switches.Template, tmp_path: pathlib.Path) -> None:
    assert switches.over_the_pipe(mcp, combination, surface_template, tmp_path) == []
