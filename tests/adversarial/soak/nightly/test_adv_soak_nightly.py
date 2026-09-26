"""The nightly soak: 10,000 seeded turns, with every silent-failure detector watching.

The record is written to the records folder before anything is asserted, so a night that
fails still leaves its evidence, and the next night's store growth is judged against it.

It is pinned to B50 (#332). In these 10,000 turns the user repeats an earlier sentence
about their city or employer word for word 310 times after that fact has changed, and
`add()` drops each of those turns, so the store's current city or employer ends up
different from the one the user last named. The current-facts detector failing, and no
other, counts as the known bug. A failure of any other detector still fails the run, and
when B50 is fixed the run passes and the strict marker says so.
"""

from __future__ import annotations

from typing import Any, Callable

from harness import known_bugs

TURNS = 10_000
CURRENT_FACTS = "current facts match what was last said"


@known_bugs.xfail("B50")
def test_ten_thousand_turns_trip_no_detector(long_soak: Callable[[int], Any]) -> None:
    run = long_soak(TURNS)
    if run.failing == {CURRENT_FACTS}:
        raise known_bugs.Reproduced("B50: the store's current facts differ from what the "
                                    "workload last said, and no other detector failed")
    assert run.code == 0, run.printed
