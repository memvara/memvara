"""The nightly tier's tool-surface tests.

Real servers run the fold-over of the fast tier's array: the same 12 runs with every
setting reversed (`switches.nightly_runs`). The nightly run also collects the fast tier's
runs, and with them every combination of every three settings is started exactly three
times. In this process, a server for each of the 2,048 combinations runs every tool it
lists.
"""

from __future__ import annotations

import pathlib
from typing import Callable

import pytest

from .. import switches


@pytest.mark.parametrize("combination", switches.nightly_runs(), ids=lambda c: c.label)
def test_a_real_server_serves_what_the_oracle_predicts_nightly(
        combination: switches.Combination,
        check_real_server: Callable[[switches.Combination], None]) -> None:
    check_real_server(combination)


def test_every_combination_runs_every_listed_tool_in_process(tmp_path: pathlib.Path) -> None:
    """The fast tier runs the minimal calls in process for its 12 rows only. Here every
    combination does, so a handler that breaks under a combination the arrays never start
    fails tonight instead of at the weekly run."""
    runs = switches.every_combination()
    failures = switches.in_process_failures(runs, switches.base_env(tmp_path), calls=True)
    assert not failures, switches.summary(failures, len(runs))
