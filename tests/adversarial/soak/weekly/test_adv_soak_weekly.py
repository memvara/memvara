"""The weekly soak: 100,000 seeded turns, with every silent-failure detector watching.

The same run as the nightly soak at ten times the turns, over the same 21 simulated days,
so its slots, merges and store are ten times as busy. The record is written before
anything is asserted.

It carries two known bugs, and is registered under the first:

* B50 (#332): the user repeats an earlier sentence about their city or employer word for
  word after that fact has changed, `add()` drops the turn, and the current-facts detector
  fails.
* B51 (#333): the most-restated facts reach the salience cap and outrank the facts the
  probes ask about, and the salience detector fails.

Those two detectors failing, and no others, count as the known bugs. A failure of any
other detector still fails the run. When either bug is fixed, the set of failing detectors
changes, the run fails, and the fix updates this pin.
"""

from __future__ import annotations

from typing import Any, Callable

from harness import known_bugs

TURNS = 100_000
CURRENT_FACTS = "current facts match what was last said"
SALIENCE = "salience over relevance"


@known_bugs.xfail("B50")
def test_a_hundred_thousand_turns_trip_no_detector(long_soak: Callable[[int], Any]) -> None:
    run = long_soak(TURNS)
    if run.failing == {CURRENT_FACTS, SALIENCE}:
        raise known_bugs.Reproduced(
            "B50 and B51: the store's current facts differ from what the workload last "
            "said, and a fact at the salience cap outranked the fact a probe asked about; "
            "no other detector failed")
    assert run.code == 0, run.printed
