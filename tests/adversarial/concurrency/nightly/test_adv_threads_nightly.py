"""The thread check of `test_adv_threads.py`, with sixteen threads of 500 operations
each, so that far more interleavings are tried than a pull request can wait for."""

from __future__ import annotations

import pathlib

from ..test_adv_threads import run_threads


def test_sixteen_threads_on_one_handle_leave_a_consistent_store(
        tmp_path: pathlib.Path) -> None:
    run_threads(tmp_path / "s.db", threads=16, operations=500, seed=20260926)
