"""Recency decay and the final rescoring combinator.

The values here are pinned rather than merely ordered, because the interesting bugs in
decay are quiet ones: measuring age from `recorded_at` instead of `valid_from`, or
reading a half-life in the wrong unit, both still produce a curve that decreases.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from engram.retrieve import final_score, recency_factor
from engram.schema import PredicateRegistry
from engram.types import Claim, utcnow

NOW = datetime(2026, 8, 8, 12, 0, 0, tzinfo=timezone.utc)

# The retriever's defaults, restated so a change there shows up as a failure here.
W_RECENCY, W_CONFIDENCE, W_SALIENCE = 0.25, 0.15, 0.10


@pytest.fixture
def registry() -> PredicateRegistry:
    return PredicateRegistry()


def claim(predicate: str, *, age_days: float = 0.0, **kw) -> Claim:
    return Claim(
        subject="user",
        predicate=predicate,
        object="something",
        valid_from=NOW - timedelta(days=age_days),
        **kw,
    )


# --- recency_factor ---------------------------------------------------------


def test_fresh_claim_is_undecayed(registry: PredicateRegistry) -> None:
    assert recency_factor(claim("working_on"), registry, NOW) == pytest.approx(1.0)


def test_one_half_life_halves_the_factor(registry: PredicateRegistry) -> None:
    # working_on is FAST -> 7-day half-life; works_at is SLOW -> 730 days.
    assert recency_factor(claim("working_on", age_days=7), registry, NOW) == pytest.approx(0.5)
    assert recency_factor(claim("works_at", age_days=730), registry, NOW) == pytest.approx(0.5)


def test_fast_predicate_is_nearly_gone_after_a_month(registry: PredicateRegistry) -> None:
    """'What I'm working on today' must not still be ranking a month later."""
    factor = recency_factor(claim("working_on", age_days=30), registry, NOW)

    assert factor == pytest.approx(0.5 ** (30 / 7))
    assert factor < 0.06


def test_static_predicate_is_essentially_undecayed_after_a_decade(
    registry: PredicateRegistry,
) -> None:
    """A birthplace does not become less true. This is the case pure recency ranking
    gets wrong, and the reason decay is keyed to the predicate rather than the memory."""
    factor = recency_factor(claim("born_in", age_days=3650), registry, NOW)

    assert factor == pytest.approx(0.5 ** (3650 / 36500))
    assert factor > 0.93


def test_volatility_orders_the_decay_at_equal_age(registry: PredicateRegistry) -> None:
    age = 365.0
    static = recency_factor(claim("born_in", age_days=age), registry, NOW)
    slow = recency_factor(claim("works_at", age_days=age), registry, NOW)
    fast = recency_factor(claim("mood", age_days=age), registry, NOW)

    assert static > slow > fast
    assert fast < 1e-15  # a year-old mood is not evidence of anything


def test_aliases_resolve_to_the_canonical_half_life(registry: PredicateRegistry) -> None:
    """`current_task` is an alias of `working_on`; it must decay as FAST, not as the
    SLOW default for unknown predicates."""
    aliased = recency_factor(claim("current_task", age_days=7), registry, NOW)

    assert aliased == pytest.approx(0.5)


def test_unknown_predicate_falls_back_to_the_slow_default(registry: PredicateRegistry) -> None:
    assert recency_factor(claim("wibbles_at", age_days=730), registry, NOW) == pytest.approx(0.5)


def test_age_is_measured_from_valid_from_not_recorded_at(registry: PredicateRegistry) -> None:
    """Backfilling an old fact today must not make it look fresh.

    Both claims were written to the store at the same instant; only one was *true*
    recently. Reading `recorded_at` here would score them identically.
    """
    backfilled = Claim(
        subject="user", predicate="working_on", object="the 2019 migration",
        valid_from=NOW - timedelta(days=60), recorded_at=NOW,
    )
    current = Claim(
        subject="user", predicate="working_on", object="the auth refactor",
        valid_from=NOW, recorded_at=NOW,
    )

    assert recency_factor(backfilled, registry, NOW) < 0.01
    assert recency_factor(current, registry, NOW) == pytest.approx(1.0)


def test_future_validity_is_clamped_rather_than_amplified(registry: PredicateRegistry) -> None:
    """A negative age would send the exponent positive and hand a not-yet-true fact a
    score above every current one."""
    future = recency_factor(claim("working_on", age_days=-90), registry, NOW)

    assert future == pytest.approx(1.0)
    assert future <= 1.0


def test_factor_stays_in_unit_range_across_extreme_ages(registry: PredicateRegistry) -> None:
    for age in (0.0, 1.0, 7.0, 365.0, 3650.0, 200_000.0):
        for predicate in ("born_in", "works_at", "mood"):
            factor = recency_factor(claim(predicate, age_days=age), registry, NOW)
            assert 0.0 <= factor <= 1.0


def test_decay_is_monotonically_decreasing_in_age(registry: PredicateRegistry) -> None:
    ages = [0, 1, 3, 7, 14, 30, 90, 365]
    factors = [recency_factor(claim("working_on", age_days=a), registry, NOW) for a in ages]

    assert factors == sorted(factors, reverse=True)
    assert len(set(factors)) == len(factors)


def test_naive_datetimes_are_treated_as_utc(registry: PredicateRegistry) -> None:
    """Hand-built datetimes at API edges are routinely naive; a TypeError raised from
    inside ranking is a far worse outcome than assuming the store's convention."""
    naive_claim = Claim(
        subject="user", predicate="working_on", object="x",
        valid_from=datetime(2026, 8, 1, 12, 0, 0),  # no tzinfo
    )

    aware = recency_factor(naive_claim, registry, NOW)
    both_naive = recency_factor(naive_claim, registry, NOW.replace(tzinfo=None))

    assert aware == pytest.approx(0.5 ** (7 / 7))
    assert both_naive == pytest.approx(aware)


def test_nonpositive_half_life_returns_zero_instead_of_dividing_by_zero(
    registry: PredicateRegistry,
) -> None:
    """A learned predicate could in principle carry a broken half-life. Discovering it
    via ZeroDivisionError in the middle of a search is not acceptable."""

    class BrokenRegistry:
        def half_life_days(self, predicate: str) -> float:
            return 0.0

    assert recency_factor(claim("whatever", age_days=5), BrokenRegistry(), NOW) == 0.0


def test_now_defaults_are_not_read_from_the_clock(registry: PredicateRegistry) -> None:
    """`now` is a parameter so a time-travel query decays relative to the instant being
    asked about. Scoring 2024's facts as two years stale would defeat `as_of`."""
    then = datetime(2024, 1, 1, tzinfo=timezone.utc)
    c = Claim(subject="user", predicate="working_on", object="x", valid_from=then)

    assert recency_factor(c, registry, then) == pytest.approx(1.0)
    assert recency_factor(c, registry, utcnow()) < 1e-20


# --- final_score ------------------------------------------------------------


def score(fusion: float, recency: float, confidence: float = 1.0, salience: float = 1.0) -> float:
    return final_score(
        fusion,
        recency=recency,
        confidence=confidence,
        salience=salience,
        w_recency=W_RECENCY,
        w_confidence=W_CONFIDENCE,
        w_salience=W_SALIENCE,
    )


def test_perfect_signals_apply_the_full_multiplier() -> None:
    # boost = 1 + 0.25 + 0.15 + 0.10 = 1.5
    assert score(0.02, 1.0) == pytest.approx(0.03)


def test_each_factor_contributes_its_own_weight() -> None:
    assert score(1.0, 0.0, confidence=0.0, salience=0.0) == pytest.approx(1.0)
    assert score(1.0, 1.0, confidence=0.0, salience=0.0) == pytest.approx(1.25)
    assert score(1.0, 0.0, confidence=1.0, salience=0.0) == pytest.approx(1.15)
    assert score(1.0, 0.0, confidence=0.0, salience=1.0) == pytest.approx(1.10)


def test_zero_weights_reduce_to_pure_fusion() -> None:
    """Turning a signal off must remove it exactly, not re-baseline the others."""
    out = final_score(
        0.0164, recency=0.01, confidence=0.3, salience=9.0,
        w_recency=0.0, w_confidence=0.0, w_salience=0.0,
    )

    assert out == pytest.approx(0.0164)


def test_score_is_monotone_in_every_factor() -> None:
    base = score(0.02, 0.5, confidence=0.5, salience=0.5)

    assert score(0.02, 0.9, confidence=0.5, salience=0.5) > base
    assert score(0.02, 0.5, confidence=0.9, salience=0.5) > base
    assert score(0.02, 0.5, confidence=0.5, salience=0.9) > base
    assert score(0.03, 0.5, confidence=0.5, salience=0.5) > base


def test_a_fresh_claim_overtakes_a_stale_one_ranked_above_it() -> None:
    """The headline behaviour: a stale FAST fact ranked #1 by relevance loses to a
    fresh one ranked #2. Vector-only retrieval has no mechanism to do this."""
    stale = score(1.0 / 61.0, 0.5 ** (30 / 7))  # 30-day-old `working_on`, rank 0
    fresh = score(1.0 / 62.0, 1.0)              # today's `working_on`,   rank 1

    assert fresh > stale


def test_recency_cannot_drag_an_irrelevant_claim_out_of_the_tail() -> None:
    """The counterweight to the test above. Quality reorders neighbours; it must not
    let freshness beat relevance outright, or search collapses into a recency feed."""
    fresh_deep = score(1.0 / 121.0, 1.0)   # perfectly fresh but rank 60
    stale_top = score(1.0 / 61.0, 0.0)     # fully decayed but rank 0

    assert stale_top > fresh_deep


def test_multiplicative_combination_preserves_order_under_equal_quality() -> None:
    """With identical quality signals the ranking must be exactly the RRF ranking.

    An additive combinator fails this: adding ~1.0-scale factors to ~0.016-scale fusion
    scores flattens them into a tie and hands the ordering to floating-point noise.
    """
    fusions = [1.0 / (60 + r + 1) for r in range(10)]
    scored = [score(f, 0.4, confidence=0.8, salience=1.2) for f in fusions]

    assert scored == sorted(scored, reverse=True)
    assert len(set(scored)) == len(scored)


def test_salience_above_one_is_not_clamped() -> None:
    """Repeated observation is earned headroom, not an error to be normalized away."""
    once = score(0.02, 1.0, salience=1.0)
    reinforced = score(0.02, 1.0, salience=3.0)

    assert reinforced > once
    assert reinforced == pytest.approx(0.02 * (1 + 0.25 + 0.15 + 0.10 * 3.0))


def test_zero_fusion_stays_zero() -> None:
    """An item no retriever ranked has no relevance for quality to amplify."""
    assert score(0.0, 1.0, confidence=1.0, salience=1.0) == 0.0
