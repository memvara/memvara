"""The statistics and the budget rules in bench/perf_budget.py, on numbers worked out by hand.

The design fixes the budget decision rule before any timing is measured, so these tests
pin the rule itself: the hard ceilings, the 1-2-5 rounding of a budget, and the three
conditions a regression needs. None of them measures anything, so none can be slowed down
by a busy machine.
"""

from __future__ import annotations

import pytest

import perf_budget as pb


# --- percentiles and the bootstrap interval ---------------------------------------------


ONE_TO_HUNDRED = [float(v) for v in range(1, 101)]


def test_percentiles_are_taken_by_nearest_rank() -> None:
    # Nearest rank, as bench/evalkit.py's `percentile` takes it: the value at index
    # round(q * (n - 1)) of the sorted samples. For 1..100 that index is 50, 89, 94, 98.
    summary = pb.summarize(ONE_TO_HUNDRED, resamples=200)
    assert (summary["n"], summary["p50"], summary["p90"], summary["p95"], summary["p99"],
            summary["max"]) == (100, 51.0, 90.0, 95.0, 99.0, 100.0)


def test_each_bootstrap_interval_contains_its_percentile() -> None:
    summary = pb.summarize(ONE_TO_HUNDRED, resamples=500)
    for stat in ("p50", "p90", "p95", "p99"):
        low, high = summary["interval"][stat]
        assert low <= summary[stat] <= high, stat
        assert low < high, f"{stat}: 100 distinct samples give an interval of some width"


def test_the_bootstrap_gives_the_same_interval_for_the_same_seed() -> None:
    first = pb.bootstrap_interval(ONE_TO_HUNDRED, 0.95, resamples=300, seed=7)
    second = pb.bootstrap_interval(ONE_TO_HUNDRED, 0.95, resamples=300, seed=7)
    assert first == second


def test_a_constant_sample_has_an_interval_of_zero_width() -> None:
    assert pb.bootstrap_interval([4.0] * 30, 0.95, resamples=300) == (4.0, 4.0)


def test_an_empty_sample_is_refused_rather_than_summarized_as_zero() -> None:
    with pytest.raises(ValueError, match="no samples"):
        pb.summarize([])


# --- rounding a budget up to a 1-2-5 step ------------------------------------------------


@pytest.mark.parametrize("value, step", [
    (0.7, 1.0), (1.0, 1.0), (1.01, 2.0), (2.0, 2.0), (3.0, 5.0), (5.0, 5.0),
    (7.5, 10.0), (13.0, 20.0), (150.0, 200.0), (0.012, 0.02),
    # A step except for floating-point noise stays on the step.
    (1.5 * 4 / 3, 2.0), (2.01, 5.0),
])
def test_a_budget_rounds_up_to_the_next_1_2_5_step(value: float, step: float) -> None:
    assert pb.round_up_125(value) == pytest.approx(step, rel=1e-12)


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_a_budget_that_is_not_positive_is_refused(value: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        pb.round_up_125(value)


def test_the_median_absolute_deviation_is_unscaled() -> None:
    # Median 2; the deviations 1, 1, 0, 0, 2, 4, 7 have median 1.
    assert pb.mad([1, 1, 2, 2, 4, 6, 9]) == 1


# --- the regression rule ------------------------------------------------------------------


class Remeasure:
    """Stands in for measuring a series again, and counts how often it was asked."""

    def __init__(self, value: float) -> None:
        self.value = value
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        return self.value


def test_fewer_than_seven_nights_is_no_history_and_nothing_is_remeasured() -> None:
    again = Remeasure(50.0)
    verdict = pb.judge(50.0, [10.0] * 6, again, floor=pb.LATENCY_FLOOR_MS)
    assert verdict.outcome == "no history" and again.calls == 0


def test_an_increase_below_the_ratio_is_not_a_regression() -> None:
    again = Remeasure(11.9)
    verdict = pb.judge(11.9, [10.0] * 7, again, floor=pb.LATENCY_FLOOR_MS)
    assert (verdict.outcome, verdict.median, again.calls) == ("ok", 10.0, 0)


def test_an_increase_inside_three_deviations_is_not_a_regression() -> None:
    # Median 100 and deviation 10, so 125 passes the ratio (above 120) but its increase of
    # 25 is not above 3 x 10.
    again = Remeasure(125.0)
    history = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0]
    verdict = pb.judge(125.0, history, again, floor=pb.LATENCY_FLOOR_MS)
    assert (verdict.outcome, verdict.mad, again.calls) == ("ok", 10.0, 0)


def test_an_increase_below_the_two_millisecond_floor_is_not_a_regression() -> None:
    again = Remeasure(2.5)
    verdict = pb.judge(2.5, [1.0] * 7, again, floor=pb.LATENCY_FLOOR_MS)
    assert (verdict.outcome, again.calls) == ("ok", 0)


def test_an_increase_the_remeasure_does_not_reproduce_is_not_a_regression() -> None:
    again = Remeasure(10.0)
    verdict = pb.judge(13.0, [10.0] * 7, again, floor=pb.LATENCY_FLOOR_MS)
    assert (verdict.outcome, verdict.remeasured, again.calls) == ("not reproduced", 10.0, 1)


def test_an_increase_that_meets_all_three_conditions_is_a_regression() -> None:
    again = Remeasure(13.0)
    verdict = pb.judge(13.0, [10.0] * 7, again, floor=pb.LATENCY_FLOOR_MS)
    assert (verdict.outcome, verdict.remeasured, again.calls) == ("regression", 13.0, 1)


def test_only_the_last_seven_nights_are_the_rolling_median() -> None:
    # Over all fourteen nights the median is 55, and 13 would be nowhere near it.
    again = Remeasure(13.0)
    verdict = pb.judge(13.0, [100.0] * 7 + [10.0] * 7, again, floor=pb.LATENCY_FLOOR_MS)
    assert (verdict.outcome, verdict.median) == ("regression", 10.0)


# --- library budgets ----------------------------------------------------------------------


def nights(values: list[float], series: str = "search@1000/warm") -> list[dict[str, float]]:
    return [{series: value} for value in values]


def test_budgets_are_refused_before_fourteen_valid_nights() -> None:
    with pytest.raises(ValueError, match="14"):
        pb.derive_budgets(nights([8.0] * 13))


def test_a_budget_is_one_and_a_half_times_the_median_p95_rounded_up() -> None:
    # The median of these fourteen is 8, and 1.5 x 8 = 12 rounds up to 20.
    p95s = [6.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 9.0, 9.0, 10.0, 11.0, 12.0]
    assert pb.derive_budgets(nights(p95s)) == {"search@1000/warm": 20.0}


def test_a_series_missing_from_any_of_the_nights_gets_no_budget() -> None:
    measured = nights([3.0] * 14)
    for night in measured[1:]:
        night["recall@1000/warm"] = 4.0
    assert pb.derive_budgets(measured) == {"search@1000/warm": 5.0}


# --- the hard ceilings from the hook contract --------------------------------------------


def hook_series(p95: float = 100.0, maximum: float = 200.0, timeouts: int = 0) -> dict:
    return {"p95": p95, "max": maximum, "timeouts": timeouts}


def breaches(series: dict[str, dict]) -> set[tuple[str, str]]:
    return {(entry["series"], entry["stat"]) for entry in pb.check_ceilings(series)
            if entry["breached"]}


RECALL_COLD = pb.series_key("hook.recall", 1000, "cold")
START_WARM = pb.series_key("hook.session_start", 1000, "warm")


def test_the_recall_hook_may_reach_its_p95_ceiling_but_not_pass_it() -> None:
    assert breaches({RECALL_COLD: hook_series(p95=7_500.0)}) == set()
    assert breaches({RECALL_COLD: hook_series(p95=7_501.0)}) == {(RECALL_COLD, "p95")}


def test_the_recall_hook_may_never_take_longer_than_ten_seconds() -> None:
    assert breaches({RECALL_COLD: hook_series(maximum=10_001.0)}) == {(RECALL_COLD, "max")}


def test_a_recall_hook_that_timed_out_breaches_the_maximum_whatever_it_recorded() -> None:
    assert breaches({RECALL_COLD: hook_series(timeouts=1)}) == {(RECALL_COLD, "max")}


def test_session_start_may_not_pass_its_p95_ceiling() -> None:
    assert breaches({START_WARM: hook_series(p95=20_001.0)}) == {(START_WARM, "p95")}
    assert breaches({START_WARM: hook_series(p95=20_000.0, maximum=60_000.0)}) == set()


def test_every_hook_series_is_checked_and_no_library_series_is() -> None:
    series = {RECALL_COLD: hook_series(), START_WARM: hook_series(),
              pb.series_key("search", 1000, "cold"): hook_series(p95=1e9, maximum=1e9)}
    checked = {(entry["series"], entry["stat"]) for entry in pb.check_ceilings(series)}
    assert checked == {(RECALL_COLD, "p95"), (RECALL_COLD, "max"), (START_WARM, "p95")}
