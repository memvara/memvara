"""The nightly soak: 10,000 seeded turns, with every silent-failure detector watching.

The record is written to the records folder before anything is asserted, so a night that
fails still leaves its evidence, and the next night's store growth is judged against it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

import soak

TURNS = 10_000


def test_ten_thousand_turns_trip_no_detector(records_dir: Path,
                                            capsys: pytest.CaptureFixture[str]) -> None:
    folder = records_dir / "soak"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    code = soak.main(["--turns", str(TURNS), "--seed", "0", "--history", str(folder),
                      "--out", str(folder / f"soak-{TURNS}-0-{stamp}.json")])
    printed = capsys.readouterr().out
    assert code == 0, printed
