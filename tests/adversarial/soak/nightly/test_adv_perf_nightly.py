"""The nightly timing run: every series at 1,000, 10,000 and 100,000 claims, under the
design's budget rule.

The record is written before anything is asserted. A run on battery or under load is
skipped with its reasons, because the design marks it invalid rather than failed; the skip
ledger explains that reason. A valid run fails on a breached hard ceiling, a regression
confirmed by a re-measure, or a library p95 over a committed budget.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

import perf_budget


def test_the_timing_budgets_hold(records_dir: Path,
                                 capsys: pytest.CaptureFixture[str]) -> None:
    folder = records_dir / "perf"
    folder.mkdir(exist_ok=True)
    out = folder / f"perf-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    code = perf_budget.main(["--out", str(out), "--history", str(folder)])
    printed = capsys.readouterr().out
    if code == perf_budget.EXIT_INVALID:
        pytest.skip(perf_budget.skip_reason(json.loads(out.read_text(encoding="utf-8"))))
    assert code == 0, printed
