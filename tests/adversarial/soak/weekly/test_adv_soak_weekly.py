"""The weekly soak: 100,000 seeded turns, with every silent-failure detector watching.

The same run as the nightly soak at ten times the turns, over the same 21 simulated days,
so its slots, merges and store are ten times as busy. The record is written before
anything is asserted.

It carried two known bugs, and both are fixed. Until B50 (#332) was fixed, the user
repeated an earlier sentence about their city or employer word for word after that fact
had changed, `add()` dropped the turn, and the current-facts detector failed. The salience
detector failed here until B51 (#333) was fixed.
"""

from __future__ import annotations

from typing import Any, Callable

from harness import known_bugs

TURNS = 100_000
CURRENT_FACTS = "current facts match what was last said"


def test_a_hundred_thousand_turns_trip_no_detector(long_soak: Callable[[int], Any]) -> None:
    run = long_soak(TURNS)
    if run.failing == {CURRENT_FACTS}:
        raise known_bugs.Reproduced(
            "B50: the store's current facts differ from what the workload last said; no "
            "other detector failed")
    assert run.code == 0, run.printed
