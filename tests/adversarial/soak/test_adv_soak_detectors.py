"""Each soak detector on hand-built observations: its boundary, and what it says with no evidence.

A detector that reads "nothing recorded" as "nothing wrong" passes forever once the
telemetry it reads is disconnected, which is the silent failure these detectors exist to
catch. So every detector fails, or for the tracked ones says so, when its evidence is
missing. The fault tests in `test_adv_soak_faults.py` show the same detectors firing on
real runs.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

import soak
from memvara.telemetry import (
    CONSOLIDATE_MERGED,
    REDACT_INSPECTED,
    WRITE_RETRACTION,
    WRITE_TURNS,
)


def recorder(*, merged: int | None = 3, retired: int = 4, noop: int = 0,
             inspected: int | None = 100, turns: int = 100) -> soak.SoakRecorder:
    """A recorder holding what a healthy run of the named sizes would have counted."""
    rec = soak.SoakRecorder()
    if merged is not None:
        rec.counter(CONSOLIDATE_MERGED, merged)
    if retired:
        rec.counter(WRITE_RETRACTION, retired, outcome="retired")
    if noop:
        rec.counter(WRITE_RETRACTION, noop, outcome="noop")
    if inspected is not None:
        rec.counter(REDACT_INSPECTED, inspected, field="episode", script="latin")
    rec.counter(WRITE_TURNS, turns)
    return rec


def healthy(**changes: Any) -> soak.Observations:
    """Observations of a healthy run, with `changes` applied."""
    observed = soak.Observations(
        config=soak.SoakConfig(200), recorder=recorder(),
        predicates=set(soak.VOCABULARY), crowded=[], panel_correlations=[0.5, 0.4, -0.1],
        probes=100, probe_hits=100, gate={"latin": [50, 50], "han": [5, 5]},
        planted=[(0, True), (1, True)], unplanted_changed=0, store_bytes=None,
        elapsed_s=1.0)
    return dataclasses.replace(observed, **changes)


def test_every_detector_is_quiet_on_healthy_observations() -> None:
    findings = soak.judge_soak(healthy())
    assert [f.detector for f in findings] == [
        "predicate explosion", "recency refresh", "flip-flop row growth",
        "salience over relevance", "script bias in the gate",
        "retraction that retires nothing", "redaction drift", "store growth"]
    assert {f.status for f in findings} <= {"ok", "tracked"}


# --- predicate explosion ------------------------------------------------------------------


def test_the_vocabulary_itself_is_not_an_explosion() -> None:
    assert soak.predicate_explosion(healthy()).status == "ok"


def test_one_predicate_beyond_a_vocabulary_of_six_is_an_explosion() -> None:
    finding = soak.predicate_explosion(healthy(predicates=set(soak.VOCABULARY) | {"resides_in"}))
    assert (finding.status, finding.value, finding.flagged) == ("fail", 7.0, ("resides_in",))


def test_a_store_with_no_claims_did_not_measure_its_predicates() -> None:
    finding = soak.predicate_explosion(healthy(predicates=set()))
    assert finding.status == "fail" and "not measured" in finding.detail


# --- recency refresh ----------------------------------------------------------------------


@pytest.mark.parametrize("values, status", [
    ([0.0], "fail"), ([0.01], "ok"), ([-0.5, 0.0, 0.2], "fail"), ([-0.5, 0.1, 0.2], "ok"),
])
def test_recency_fails_at_a_median_correlation_of_zero_or_less(
        values: list[float], status: str) -> None:
    assert soak.recency_refresh(healthy(panel_correlations=values)).status == status


def test_no_panel_correlation_means_recency_was_not_measured() -> None:
    finding = soak.recency_refresh(healthy(panel_correlations=[]))
    assert finding.status == "fail" and "not measured" in finding.detail


# --- flip-flop row growth -----------------------------------------------------------------


def test_a_single_valued_slot_with_two_live_claims_is_row_growth() -> None:
    finding = soak.flip_flop(healthy(crowded=[("Talvor", "lives_in", 2)]))
    assert (finding.status, finding.value, finding.flagged) == (
        "fail", 2.0, ("Talvor lives_in",))


def test_consolidation_that_never_merges_is_row_growth() -> None:
    finding = soak.flip_flop(healthy(recorder=recorder(merged=0)))
    assert finding.status == "fail" and "merged nothing" in finding.detail


def test_a_run_with_no_consolidation_pass_did_not_measure_row_growth() -> None:
    finding = soak.flip_flop(healthy(recorder=recorder(merged=None)))
    assert finding.status == "fail" and "not measured" in finding.detail


# --- salience over relevance --------------------------------------------------------------


@pytest.mark.parametrize("hits, status", [(95, "ok"), (94, "fail")])
def test_the_relevant_claim_must_rank_first_in_95_of_100_probes(hits: int, status: str) -> None:
    assert soak.salience_over_relevance(healthy(probe_hits=hits)).status == status


def test_no_probe_means_salience_was_not_measured() -> None:
    finding = soak.salience_over_relevance(healthy(probes=0, probe_hits=0))
    assert finding.status == "fail" and "not measured" in finding.detail


# --- script bias in the gate (tracked) ----------------------------------------------------


def test_a_script_far_below_the_latin_rate_is_named_but_does_not_fail() -> None:
    finding = soak.script_bias(healthy(gate={"latin": [10, 10], "han": [10, 7],
                                             "kana": [10, 9]}))
    assert (finding.status, finding.flagged) == ("tracked", ("han",))


def test_no_latin_fact_at_the_gate_means_script_bias_was_not_measured() -> None:
    finding = soak.script_bias(healthy(gate={"han": [5, 5]}))
    assert finding.status == "tracked" and "not measured" in finding.detail


# --- a retraction that retires nothing ----------------------------------------------------


def test_one_retraction_that_closes_nothing_fails() -> None:
    finding = soak.retraction_noop(healthy(recorder=recorder(noop=1)))
    assert (finding.status, finding.value) == ("fail", 1.0)


def test_no_retraction_counted_means_retractions_were_not_measured() -> None:
    finding = soak.retraction_noop(healthy(recorder=recorder(retired=0)))
    assert finding.status == "fail" and "not measured" in finding.detail


# --- redaction drift ----------------------------------------------------------------------


def test_a_day_at_99_percent_is_no_drift_and_a_day_at_98_is() -> None:
    ninety_nine = [(0, True)] * 99 + [(0, False)]
    ninety_eight = [(1, True)] * 49 + [(1, False)]
    assert soak.redaction_drift(healthy(planted=ninety_nine)).status == "ok"
    finding = soak.redaction_drift(healthy(planted=ninety_nine + ninety_eight))
    assert (finding.status, finding.value) == ("fail", 0.98)
    assert "day 2" in finding.detail


def test_nothing_planted_means_redaction_was_not_measured() -> None:
    finding = soak.redaction_drift(healthy(planted=[]))
    assert finding.status == "fail" and "not measured" in finding.detail


def test_turns_written_with_nothing_offered_to_the_redactor_is_a_lost_policy() -> None:
    finding = soak.redaction_drift(healthy(recorder=recorder(inspected=None)))
    assert finding.status == "fail" and "not running" in finding.detail


# --- store growth -------------------------------------------------------------------------


def test_a_store_in_memory_has_no_growth_to_measure() -> None:
    finding = soak.store_growth(healthy(), [], lambda: 0.0)
    assert finding.status == "tracked" and "in memory" in finding.detail


def test_growth_without_seven_earlier_soaks_is_tracked() -> None:
    finding = soak.store_growth(healthy(store_bytes=200_000), [1_000.0] * 6, lambda: 0.0)
    assert (finding.status, finding.value) == ("tracked", 1_000.0)


def test_growth_the_relative_rule_confirms_fails_and_growth_it_does_not_passes() -> None:
    grown = healthy(store_bytes=300_000)          # 1,500 bytes a turn against 1,000
    assert soak.store_growth(grown, [1_000.0] * 7, lambda: 1_500.0).status == "fail"
    assert soak.store_growth(grown, [1_000.0] * 7, lambda: 1_000.0).status == "ok"
    assert soak.store_growth(healthy(store_bytes=200_000), [1_000.0] * 7,
                             lambda: 1_000.0).status == "ok"


def test_history_without_a_way_to_measure_again_is_refused() -> None:
    with pytest.raises(ValueError, match="measure"):
        soak.judge_soak(healthy(store_bytes=200_000), history=[1_000.0] * 7)
