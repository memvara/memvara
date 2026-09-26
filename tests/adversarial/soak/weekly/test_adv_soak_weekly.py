"""The weekly soak: 100,000 seeded turns, with every silent-failure detector watching.

The same run as the nightly soak at ten times the turns, over the same 21 simulated days,
so its slots, merges and store are ten times as busy. The record is written before
anything is asserted.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

import soak

TURNS = 100_000


def test_a_hundred_thousand_turns_trip_no_detector(records_dir: Path,
                                                   capsys: pytest.CaptureFixture[str]) -> None:
    folder = records_dir / "soak"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    code = soak.main(["--turns", str(TURNS), "--seed", "0", "--history", str(folder),
                      "--out", str(folder / f"soak-{TURNS}-0-{stamp}.json")])
    printed = capsys.readouterr().out
    assert code == 0, printed
