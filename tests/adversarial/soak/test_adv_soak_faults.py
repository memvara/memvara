"""Each soak detector fires on a fault injected into a real run, and none fires on a healthy one.

This is the design's proof for the soak: "each detector fires on an injected fault". Every
run here is a 200-turn soak through the real library, in memory. A fault is one of three
kinds, whichever is closest to how the failure would really arrive: a patch to memvara's
own code (reinforcement, alias folding, the merge threshold, the gate, the ranking), a
configuration a deployment could really have (a registry, a missing redactor), or a
change to the workload's data (a misspelt retraction, personal data in a new format).
Each test reads the finding's value or detail as well as its status, so that it fails if
the detector fires for some other reason.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import pytest

import soak
from memvara.retrieve.hybrid import HybridRetriever
from memvara.schema import BUILTIN_PREDICATES, Cardinality, PredicateRegistry, _slugify
from memvara.write.reconcile import Reconciler

TURNS = 200


def findings_of(**kwargs: Any) -> dict[str, soak.Finding]:
    observed = soak.run(soak.SoakConfig(TURNS), None, **kwargs)
    return {finding.detector: finding for finding in soak.judge_soak(observed)}


@pytest.fixture(scope="module")
def healthy() -> dict[str, soak.Finding]:
    return findings_of()


def test_a_healthy_soak_trips_no_detector(healthy: dict[str, soak.Finding]) -> None:
    failing = {name: finding.detail for name, finding in healthy.items()
               if finding.status == "fail"}
    assert failing == {}
    assert healthy["script bias in the gate"].flagged == ()


def test_aliases_that_stop_folding_are_a_predicate_explosion(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PredicateRegistry, "normalize", lambda self, raw: _slugify(raw))
    finding = findings_of()["predicate explosion"]
    aliases = {spelling for spellings in soak.ALIASES.values() for spelling in spellings}
    assert finding.status == "fail"
    assert finding.flagged and set(finding.flagged) <= aliases


def test_reinforcement_that_refreshes_nothing_is_caught_by_recency(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # The bug memvara/telemetry.py describes: a restatement is counted, but neither its
    # observation time nor its storage strength moves, so a fact restated every day ranks
    # as if it were last heard when it was first said.
    def stale(self: Reconciler, claim: Any, sources: Any, observed_at: Any = None) -> Any:
        claim.observation_count += 1
        claim.sources = list(dict.fromkeys([*claim.sources, *sources]))
        self.store.put_claim(claim)
        return self.store.get_claim(claim.id) or claim

    monkeypatch.setattr(Reconciler, "reinforce", stale)
    finding = findings_of()["recency refresh"]
    assert finding.status == "fail" and finding.value is not None and finding.value <= 0


def test_single_valued_predicates_declared_as_many_are_row_growth() -> None:
    many = {"lives_in", "works_at", "working_on"}
    specs = tuple(dataclasses.replace(spec, cardinality=Cardinality.MANY)
                  if spec.name in many else spec for spec in BUILTIN_PREDICATES)
    finding = findings_of(options={"registry": PredicateRegistry(specs)})[
        "flip-flop row growth"]
    assert finding.status == "fail" and finding.value is not None and finding.value > 1
    assert {slot.split()[-1] for slot in finding.flagged} <= many


def test_a_merge_that_never_fires_is_row_growth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("memvara.consolidate.merge.calibration_of",
                        lambda embedder: SimpleNamespace(merge=1.01))
    finding = findings_of()["flip-flop row growth"]
    assert finding.status == "fail" and "merged nothing" in finding.detail


def test_a_ranking_that_puts_salience_before_relevance_is_caught_by_the_probes(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # Salience decides the order and relevance only breaks ties. A weight fault is the
    # gentler form of the same failure, but 200 turns restate too little for it to show:
    # with read_w_salience at 30, every probe still ranked right. At 10,000 turns a weight
    # of 1.0 is enough, measured at 427 of 1,361 probes ranked right.
    original = HybridRetriever._rank

    def salience_first(self: HybridRetriever, results: list[Any], k: int) -> list[Any]:
        ranked = original(self, results, len(results))
        ranked.sort(key=lambda result: -result.claim.salience)
        return ranked[:k]

    monkeypatch.setattr(HybridRetriever, "_rank", salience_first)
    finding = findings_of()["salience over relevance"]
    assert finding.status == "fail"
    assert finding.value is not None and finding.value < 0.95


def test_a_gate_that_drops_unspaced_scripts_is_tracked_by_name(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("memvara.write.gate._MIN_UNSPACED_CHARS", 10**6)
    findings = findings_of()
    finding = findings["script bias in the gate"]
    assert finding.status == "tracked" and set(finding.flagged) == {"han", "kana"}
    assert [f.detector for f in findings.values() if f.status == "fail"] == []


def test_a_retraction_of_a_misspelt_like_retires_nothing(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(soak.Workload, "retraction_text",
                        lambda self, item: f"I no longer like x{item}.")
    finding = findings_of()["retraction that retires nothing"]
    assert finding.status == "fail" and finding.value is not None and finding.value >= 1


def test_personal_data_in_a_format_the_rules_miss_is_redaction_drift(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # Unpunctuated phone numbers are one of the misses PatternRedactor's docstring lists,
    # which is exactly the drift the redaction series exists to show.
    original = soak.Workload.pii_text

    def drifting(self: soak.Workload, index: int) -> str:
        if index < TURNS // 2:
            return original(self, index)
        return f"You can reach me on 555555{index:04d}."

    monkeypatch.setattr(soak.Workload, "pii_text", drifting)
    finding = findings_of()["redaction drift"]
    assert finding.status == "fail"
    assert finding.value is not None and finding.value < 0.99


def test_a_deployment_that_dropped_its_redactor_is_caught() -> None:
    finding = findings_of(options={"redactor": None})["redaction drift"]
    assert finding.status == "fail" and "not running" in finding.detail
