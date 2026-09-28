"""Put `bench/` on the import path, so these tests can import `soak` and `perf_budget`, and
say where the nightly and weekly runs keep their records.

`bench/` is a folder of scripts rather than a package, and its scripts import each other
by bare name (`import evalkit`), as they do when run as `python bench/soak.py`.
`tests/test_bench_eval.py` handles the same folder the same way.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Callable, NamedTuple

import pytest

from harness.env import NIGHTLY_RECORDS

BENCH = Path(__file__).resolve().parents[3] / "bench"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import perf_budget  # noqa: E402 - bench/ is on the path from the lines above
import soak  # noqa: E402


class LongSoak(NamedTuple):
    """What a long soak run through `soak.main` left: its exit code, what it printed, and
    the names of the detectors that failed."""

    code: int
    printed: str
    failing: frozenset[str]


@pytest.fixture
def records_dir() -> Path:
    """The folder the long runs write their records to and read their history from.

    `$NIGHTLY_RECORDS_DIR` when it is set, and otherwise `local/nightly/records` in this
    checkout, which git ignores. The nightly run starts each night in a clean worktree, so
    it points the variable at a folder outside the worktree; otherwise every night would
    start with no history and the regression rules could never apply.
    """
    configured = os.environ.get("NIGHTLY_RECORDS_DIR")
    folder = Path(configured) if configured else BENCH.parent.joinpath(*NIGHTLY_RECORDS)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


@pytest.fixture
def long_soak(records_dir: Path,
              capsys: pytest.CaptureFixture[str]) -> Callable[[int], LongSoak]:
    """Run a seed-0 soak of the given length the way the nightly run does.

    `soak.main` writes the record to the records folder before anything is asserted, so a
    failing run still leaves its evidence, and it judges store growth against the earlier
    records in the same folder.
    """

    def run(turns: int) -> LongSoak:
        folder = records_dir / "soak"
        out = folder / f"soak-{turns}-0-{perf_budget.record_stamp()}.json"
        code = soak.main(["--turns", str(turns), "--seed", "0", "--history", str(folder),
                          "--out", str(out)])
        findings = json.loads(out.read_text(encoding="utf-8"))["findings"]
        return LongSoak(code, capsys.readouterr().out,
                        frozenset(f["detector"] for f in findings if f["status"] == "fail"))

    return run
