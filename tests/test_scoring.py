"""Recency decay and the final rescoring combinator.

The values here are pinned rather than merely ordered, because the interesting bugs in
decay are quiet ones: measuring age from `recorded_at` instead of `valid_from`, or
reading a half-life in the wrong unit, both still produce a curve that decreases.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from engram.retrieve import (
    calibrate_min_score,
    final_score,
    lexical_relevance,
    normalized_score,
    quality_boost,
    recency_factor,
    relevance,
    vector_relevance,
)
from engram.retrieve.scoring import LEXICAL_HALF_SATURATION
from engram.schema import PredicateRegistry
from engram.types import Claim, Explanation, Result, utcnow

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


def test_final_score_is_fusion_times_the_quality_boost() -> None:
    """The two are one function split in half, so they cannot drift apart."""
    boost = quality_boost(recency=0.3, confidence=0.8, salience=1.4,
                          w_recency=W_RECENCY, w_confidence=W_CONFIDENCE,
                          w_salience=W_SALIENCE)

    assert boost == pytest.approx(1.0 + 0.25 * 0.3 + 0.15 * 0.8 + 0.10 * 1.4)
    assert score(0.02, 0.3, confidence=0.8, salience=1.4) == pytest.approx(0.02 * boost)


def test_the_boost_span_is_what_the_docstring_claims() -> None:
    """The arithmetic the old docstring got wrong, pinned so it cannot rot again.

    A 1.2x span at equal confidence and salience, 1.5x against a claim with every
    signal at zero, and 1.66x once salience has been reinforced to 2.6 - against an
    RRF gap of 1.016x between adjacent ranks, which is the whole reason quality is no
    longer multiplied into a fused rank.
    """
    def boost(recency: float, confidence: float = 1.0, salience: float = 1.0) -> float:
        return quality_boost(recency=recency, confidence=confidence, salience=salience,
                             w_recency=W_RECENCY, w_confidence=W_CONFIDENCE,
                             w_salience=W_SALIENCE)

    assert boost(1.0) / boost(0.0) == pytest.approx(1.2)
    assert boost(1.0) / boost(0.0, confidence=0.0, salience=0.0) == pytest.approx(1.5)
    assert boost(1.0, salience=2.6) / boost(0.0, 0.0, 0.0) == pytest.approx(1.66)
    assert (1.0 / 61.0) / (1.0 / 62.0) == pytest.approx(1.016, abs=0.001)


# --- absolute per-retriever relevance ---------------------------------------


def test_lexical_relevance_is_bounded_and_monotone() -> None:
    values = [lexical_relevance(b, terms=1) for b in (0.0, 0.5, 1.5, 3.0, 12.0, 1e6)]

    assert values == sorted(values)
    assert all(0.0 <= v < 1.0 for v in values)
    assert values[0] == 0.0
    assert values[-1] > 0.99  # saturates rather than running away with a rare term


def test_lexical_relevance_half_saturates_at_the_documented_point() -> None:
    assert lexical_relevance(LEXICAL_HALF_SATURATION, terms=1) == pytest.approx(0.5)
    assert lexical_relevance(2 * LEXICAL_HALF_SATURATION, terms=2) == pytest.approx(0.5)


def test_lexical_relevance_is_per_term_so_queries_are_comparable() -> None:
    """BM25 sums over query terms, so a long question outscores a short one against
    the same claim. Without dividing, one threshold cannot serve both."""
    two_terms_both_matched = lexical_relevance(6.0, terms=2)
    six_terms_one_matched = lexical_relevance(6.0, terms=6)

    assert two_terms_both_matched > six_terms_one_matched
    assert lexical_relevance(3.0, terms=1) == lexical_relevance(9.0, terms=3)


def test_lexical_relevance_treats_no_terms_and_no_score_as_no_evidence() -> None:
    """`terms=0` is the abstaining leg; a non-positive BM25 is a row that matched
    nothing. Neither may divide, and neither is evidence."""
    assert lexical_relevance(5.0, terms=0) == 0.0
    assert lexical_relevance(0.0, terms=3) == 0.0
    assert lexical_relevance(-2.0, terms=3) == 0.0


def test_vector_relevance_clamps_to_the_unit_interval() -> None:
    """A vector pointing away from the query is no evidence, not negative evidence -
    left unclamped it would subtract from what the other retriever found."""
    assert vector_relevance(-1.0) == 0.0
    assert vector_relevance(-0.001) == 0.0
    assert vector_relevance(0.42) == pytest.approx(0.42)
    assert vector_relevance(1.0) == 1.0
    assert vector_relevance(1.0000001) == 1.0  # float noise in a dot product


# --- blending the legs ------------------------------------------------------


def blend(vector: float | None, lexical: float | None, wv: float = 1.0,
          wl: float = 1.0) -> float:
    return relevance(vector=vector, lexical=lexical, w_vector=wv, w_lexical=wl)


def test_an_abstaining_leg_is_dropped_not_counted_as_zero() -> None:
    """The distinction the whole normalization rests on.

    `None` means the leg never ran - a CJK query the embedder cannot see, or an exact
    identifier with no content terms. Averaging in a vote it never cast would halve
    every result of those queries and make them indistinguishable from a claim that a
    working retriever actively failed to find.
    """
    assert blend(None, 0.6) == pytest.approx(0.6)
    assert blend(0.6, None) == pytest.approx(0.6)
    assert blend(0.6, 0.0) == pytest.approx(0.3)
    assert blend(None, None) == 0.0


def test_a_leg_that_ran_and_missed_costs_more_than_one_that_abstained() -> None:
    """Corroboration is priced as disagreement, not as a bonus.

    Two legs agreeing at 0.5 land in the same place as one leg alone at 0.5 - the
    honest reading, since neither found more than half-strength evidence. What moves
    the number is a leg that was in a position to corroborate and did not.
    """
    assert blend(0.5, 0.5) == pytest.approx(blend(None, 0.5))
    assert blend(0.5, 0.5) > blend(0.5, 0.2) > blend(0.5, 0.0)
    assert blend(0.9, 0.1) == pytest.approx(0.5)


def test_weights_scale_the_blend_and_zero_removes_a_leg_entirely() -> None:
    assert blend(0.8, 0.4, wv=3.0, wl=1.0) == pytest.approx((3 * 0.8 + 0.4) / 4)
    assert blend(0.8, 0.4, wv=0.0) == pytest.approx(0.4)
    assert blend(0.8, 0.4, wl=0.0) == pytest.approx(0.8)
    assert blend(0.8, 0.4, wv=0.0, wl=0.0) == 0.0


def test_blend_stays_in_the_unit_interval() -> None:
    for v in (0.0, 0.25, 1.0):
        for lx in (0.0, 0.25, 1.0):
            assert 0.0 <= blend(v, lx) <= 1.0


# --- normalized_score -------------------------------------------------------


def norm(relevance_value: float, recency: float = 1.0, confidence: float = 1.0,
         salience: float = 1.0) -> float:
    return normalized_score(relevance_value, recency=recency, confidence=confidence,
                            salience=salience, w_recency=W_RECENCY,
                            w_confidence=W_CONFIDENCE, w_salience=W_SALIENCE)


def test_a_perfect_claim_scores_its_evidence_exactly() -> None:
    """Quality at its maximum is the identity, so the number a caller thresholds on is
    the retrievers' own conviction and nothing else."""
    assert norm(0.75) == pytest.approx(0.75)
    assert norm(1.0) == pytest.approx(1.0)
    assert norm(0.0) == 0.0


def test_quality_can_only_pull_a_result_down() -> None:
    """The inversion of the shipped arithmetic. Freshness reorders neighbours; it can
    never lift a claim above the evidence that surfaced it."""
    evidence = 0.6
    worst = norm(evidence, recency=0.0, confidence=0.0, salience=0.0)

    assert worst == pytest.approx(evidence / 1.5)
    for recency in (0.0, 0.3, 1.0):
        for confidence in (0.0, 0.5, 1.0):
            got = norm(evidence, recency=recency, confidence=confidence, salience=0.5)
            assert worst <= got <= evidence


def test_scores_stay_in_the_unit_interval_even_for_a_reinforced_claim() -> None:
    """Salience is deliberately uncapped upstream, so the clamp is what keeps the
    contract - `Result.score` is in [0, 1] or callers cannot threshold on it."""
    assert norm(1.0, salience=50.0) == 1.0
    assert norm(0.99, salience=50.0) == 1.0
    assert 0.0 <= norm(0.4, salience=50.0) <= 1.0


def test_zero_weights_reduce_to_pure_evidence() -> None:
    plain = normalized_score(0.42, recency=0.01, confidence=0.3, salience=9.0,
                             w_recency=0.0, w_confidence=0.0, w_salience=0.0)

    assert plain == pytest.approx(0.42)


def test_normalization_separates_an_answerable_query_from_an_unanswerable_one() -> None:
    """The failure that motivated contract B, reduced to arithmetic.

    Both of these are rank 0 in both retrievers, so RRF scores them identically no
    matter what `rrf_k` is set to - which is why the shipped score gave "what is my
    mother's maiden name?" a *higher* number than the best answerable query. The raw
    retriever signals are not identical at all, and the normalized score reads those.
    """
    fusion = 2.0 / 61.0  # rank 0 in both legs, the best RRF can offer
    answerable = norm(blend(vector_relevance(0.60), lexical_relevance(3.6, terms=2)))
    unanswerable = norm(blend(vector_relevance(0.13), lexical_relevance(0.0, terms=3)))

    # The raw score is a function of rank and quality alone - there is no argument
    # through which the evidence above could reach it.
    assert score(fusion, 1.0) == pytest.approx(fusion * 1.5)
    assert answerable > 0.5 > unanswerable
    assert answerable > 4 * unanswerable


# --- calibrating a floor ----------------------------------------------------


def fake_search(scores: dict[str, float]):
    """A search function with pinned scores, so the calibrator is tested on arithmetic
    rather than on whatever the embedder happens to do today."""
    claim = Claim(subject="user", predicate="lives_in", object="Lisbon")

    def search(query: str) -> list[Result]:
        score = scores[query]
        return [] if score is None else [
            Result(claim=claim, score=score, explain=Explanation())]

    return search


def test_a_clean_split_puts_the_floor_between_the_two_classes() -> None:
    report = calibrate_min_score(
        fake_search({"a": 0.50, "b": 0.40, "x": 0.20, "y": 0.10}),
        answerable=["a", "b"], unanswerable=["x", "y"])

    assert report.separable
    assert report.floor == pytest.approx(0.30)  # midway between 0.20 and 0.40
    assert (report.kept, report.answerable) == (2, 2)
    assert (report.silenced, report.unanswerable) == (2, 2)


def test_margin_slides_the_floor_between_noise_and_the_weakest_answer() -> None:
    """0.0 sits on the noise ceiling and keeps everything; 1.0 sits on the weakest
    correct answer. Which error is worse is the caller's judgement, not the library's."""
    probes = {"answerable": ["a", "b"], "unanswerable": ["x", "y"]}
    search = fake_search({"a": 0.50, "b": 0.40, "x": 0.20, "y": 0.10})

    assert calibrate_min_score(search, margin=0.0, **probes).floor == pytest.approx(0.20)
    assert calibrate_min_score(search, margin=1.0, **probes).floor == pytest.approx(0.40)


def test_overlapping_classes_are_reported_rather_than_papered_over() -> None:
    """No threshold can separate these, and saying so is the useful output - the fix is
    better retrieval, not a better number."""
    report = calibrate_min_score(
        fake_search({"a": 0.50, "b": 0.20, "x": 0.30, "y": 0.10}),
        answerable=["a", "b"], unanswerable=["x", "y"])

    assert not report.separable
    assert report.kept + report.silenced == 3  # the best any single cut can do
    assert "OVERLAPPING" in str(report)


def test_an_overlap_tie_prefers_the_lower_floor() -> None:
    """Two cuts score equally, so the one that answers more questions wins: a silently
    withheld memory is the error the caller cannot see."""
    report = calibrate_min_score(
        fake_search({"a": 0.40, "b": 0.20, "x": 0.30, "y": 0.10}),
        answerable=["a", "b"], unanswerable=["x", "y"])

    assert report.floor == pytest.approx(0.20)
    assert report.kept == 2


def test_a_probe_that_returns_nothing_counts_as_zero() -> None:
    report = calibrate_min_score(
        fake_search({"a": 0.50, "x": None}), answerable=["a"], unanswerable=["x"])

    assert report.separable and report.floor == pytest.approx(0.25)


@pytest.mark.parametrize("probes", [
    {"answerable": [], "unanswerable": ["x"]},
    {"answerable": ["a"], "unanswerable": []},
    {"answerable": [], "unanswerable": []},
])
def test_calibration_refuses_to_run_on_one_class(probes: dict) -> None:
    with pytest.raises(ValueError, match="both answerable and unanswerable"):
        calibrate_min_score(fake_search({"a": 0.5, "x": 0.1}), **probes)
