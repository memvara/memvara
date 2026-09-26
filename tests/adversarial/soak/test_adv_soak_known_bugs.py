"""The bugs the soak found, each pinned by a strict expected failure that cites its issue.

Each test states the behaviour the fix must produce. It raises known_bugs.Reproduced only
when it has seen that bug's own symptom, and the bug's marker accepts nothing else, so a
different failure in the same test fails loudly instead of passing for the known bug.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from harness import known_bugs, stores
from memvara.select import PLAIN_READ
from memvara.types import MAX_SALIENCE

T0 = datetime(2026, 3, 1, tzinfo=timezone.utc)
DAY = timedelta(days=1)


# -- B50: a turn repeated word for word after its value changed is dropped ----------------

@known_bugs.xfail("B50")
def test_moving_back_in_the_same_words_makes_the_old_city_current_again() -> None:
    """A user who moves back and says so in the words they used before has moved back
    (#332). Tier 0 reads the repeat as a restatement of the first turn's claim, which has
    since ended, so nothing is reinforced and nothing is extracted."""
    mem = stores.memory(user="u1")
    first = mem.add("I live in Berlin.", ts=T0)
    mem.add("I moved to Paris.", ts=T0 + DAY)
    again = mem.add("I live in Berlin.", ts=T0 + 2 * DAY)
    live = [claim.object for claim in mem.get_all() if claim.predicate == "lives_in"]
    if (live == ["Paris"] and again.skipped == 1 and not again.added
            and again.episode_ids == first.episode_ids):
        raise known_bugs.Reproduced("B50: the repeated turn was skipped and Paris stayed "
                                    "current")
    assert live == ["Berlin"]


@known_bugs.xfail("B50")
def test_liking_again_in_the_same_words_after_taking_it_back_is_kept() -> None:
    """A like taken back and then stated again in the same words is held again (#332)."""
    mem = stores.memory(user="u1")
    first = mem.add("I like vasnu.", ts=T0)
    mem.add("I no longer like vasnu.", ts=T0 + DAY)
    again = mem.add("I like vasnu.", ts=T0 + 2 * DAY)
    live = [claim.object for claim in mem.get_all() if claim.predicate == "likes"]
    if (live == [] and again.skipped == 1 and not again.added
            and again.episode_ids == first.episode_ids):
        raise known_bugs.Reproduced("B50: the repeated like was skipped and nothing is live")
    assert live == ["vasnu"]


# -- B51: a fact at the salience cap outranks the fact a query asks about ----------------

@known_bugs.xfail("B51")
def test_a_fact_at_the_salience_cap_does_not_outrank_the_fact_asked_about() -> None:
    """Ranking is evidence first, with freshness and salience as the tiebreak
    (`scoring.normalized_score`). A fact restated 200 times over three weeks reaches the
    salience cap, and its quality factor of 1.27 then outranks a claim with more evidence
    for the question (#333). The soak saw this on 20% of its probes over 100,000 turns."""
    start = datetime.now(timezone.utc) - timedelta(days=21)
    mem = stores.memory(user="u1")
    mem.remember("Galphitor", "lives_in", "Viquinix", valid_from=start)
    mem.remember("Galphitor", "likes", "orixtor", valid_from=start)
    for n in range(200):
        mem.remember("Galphitor", "likes", "orixtor",
                     valid_from=start + timedelta(days=21) * (n + 1) / 201)
        if n % 9 == 0:
            mem.consolidate()
    first, second = mem.search("tell me where Galphitor lives", k=3, **PLAIN_READ)[:2]
    if (first.claim.text == "Galphitor likes orixtor"
            and first.explain.salience >= MAX_SALIENCE
            and second.claim.text == "Galphitor lives in Viquinix"
            and (second.explain.vector_score or 0.0) > (first.explain.vector_score or 0.0)):
        raise known_bugs.Reproduced("B51: the fact at the salience cap ranked above the "
                                    "fact the query asks about")
    assert first.claim.text == "Galphitor lives in Viquinix"
