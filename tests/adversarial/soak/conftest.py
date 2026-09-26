"""Put `bench/` on the import path, so these tests can import `soak` and `perf_budget`, and
say where the nightly and weekly runs keep their records.

`bench/` is a folder of scripts rather than a package, and its scripts import each other
by bare name (`import evalkit`), as they do when run as `python bench/soak.py`.
`tests/test_bench_eval.py` handles the same folder the same way.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[3] / "bench"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))


@pytest.fixture
def records_dir() -> Path:
    """The folder the long runs write their records to and read their history from.

    `$NIGHTLY_RECORDS_DIR` when it is set, and otherwise `local/nightly/records` in this
    checkout, which git ignores. The nightly run starts each night in a clean worktree, so
    it points the variable at a folder outside the worktree; otherwise every night would
    start with no history and the regression rules could never apply.
    """
    configured = os.environ.get("NIGHTLY_RECORDS_DIR")
    folder = Path(configured) if configured else BENCH.parent / "local" / "nightly" / "records"
    folder.mkdir(parents=True, exist_ok=True)
    return folder
