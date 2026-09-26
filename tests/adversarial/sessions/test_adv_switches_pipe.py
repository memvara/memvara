"""Real servers, started with the switch combinations of the fast tier's array, each
checked against the oracle in switches.py and made to run every tool it lists.

The array is the 12-run Plackett-Burman design (`switches.fast_runs`). The nightly and
weekly tiers start larger arrays with the same check, in nightly/ and weekly/.
"""

from __future__ import annotations

import pathlib
from typing import Callable

import pytest

from harness.stdio import McpProcess

from . import switches


@pytest.mark.parametrize("combination", switches.fast_runs(), ids=lambda c: c.label)
def test_a_real_server_serves_what_the_oracle_predicts(
        combination: switches.Combination, mcp: Callable[..., McpProcess],
        surface_template: switches.Template, tmp_path: pathlib.Path) -> None:
    assert switches.over_the_pipe(mcp, combination, surface_template, tmp_path) == []
