"""The weekly soak: 100,000 seeded turns, with every silent-failure detector watching.

The same run as the nightly soak at ten times the turns, over the same 21 simulated days,
so its slots, merges and store are ten times as busy. The record is written before
anything is asserted.

It is pinned to B51 (#333): over 100,000 turns the most-restated facts reach the salience
cap and outrank the facts the probes ask about, so the salience detector fails. That
failure, and only that one, counts as the known bug. A failure of any other detector still
fails the run, and when B51 is fixed the run passes and the strict marker says so.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

import soak
from harness import known_bugs

TURNS = 100_000


@known_bugs.xfail("B51")
def test_a_hundred_thousand_turns_trip_no_detector(records_dir: Path,
                                                   capsys: pytest.CaptureFixture[str]) -> None:
    folder = records_dir / "soak"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = folder / f"soak-{TURNS}-0-{stamp}.json"
    code = soak.main(["--turns", str(TURNS), "--seed", "0", "--history", str(folder),
                      "--out", str(out)])
    printed = capsys.readouterr().out
    findings = json.loads(out.read_text(encoding="utf-8"))["findings"]
    failing = {finding["detector"] for finding in findings if finding["status"] == "fail"}
    if failing == {"salience over relevance"}:
        raise known_bugs.Reproduced("B51: the salience detector failed, and no other")
    assert code == 0, printed
