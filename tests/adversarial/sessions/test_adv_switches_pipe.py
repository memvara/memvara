"""Real servers, started with the switch combinations of the fast tier's array, each
checked against the oracle in switches.py and made to run every tool it lists.

The array is the 12-run Plackett-Burman design (`switches.fast_runs`). The nightly and
weekly tiers start larger arrays with the same check, in nightly/ and weekly/.
"""

from __future__ import annotations

import pathlib
from typing import Any, Callable

import pytest

from harness.stdio import McpProcess

from . import switches


@pytest.mark.parametrize("combination", switches.fast_runs(), ids=lambda c: c.label)
def test_a_real_server_serves_what_the_oracle_predicts(
        combination: switches.Combination,
        check_real_server: Callable[[switches.Combination], None]) -> None:
    check_real_server(combination)


def test_the_check_reports_a_server_that_ignores_read_only_mode(
        mcp: Callable[..., McpProcess], surface_template: switches.Template,
        tmp_path: pathlib.Path) -> None:
    """The server is started writable while the check expects read-only mode. It lists the
    write tools, runs them when they are called, and changes the store, and the check must
    report all three."""
    def writable(db: pathlib.Path, **options: Any) -> McpProcess:
        return mcp(db, **{**options, "env": {**options["env"], "MEMVARA_READ_ONLY": "0"}})

    problems = switches.over_the_pipe(writable, switches.Combination.of("read_only"),
                                      surface_template, tmp_path)
    assert problems[0].startswith("lists ")
    assert any("should be refused naming 'this memory server is read-only'" in problem
               for problem in problems)
    assert problems[-1].startswith("a read-only server changed the store")
