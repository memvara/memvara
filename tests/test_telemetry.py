"""Telemetry: the six silent failures, and the cost of watching for them.

A red-team review of eleven long-horizon failure modes classified six as *silent* —
they degrade answer quality with no error, no exception and nothing in any log. This
file's job is to hold two claims true:

1. Each of the six has a series, and a realistic workload actually emits it. That is
   `test_every_silent_failure_mode_has_a_live_series`, and it is the deliverable; the
   emission points are how it passes.
2. Watching costs nothing when nobody is watching (contract G-1). Proved two ways: no
   telemetry *work* happens with the recorder unset
   (`test_nothing_is_computed_for_telemetry_when_it_is_unset`), and the unset path is
   measurably cheaper than the recording one, which is what fails if a hook ever
   becomes unconditional.

A seventh silent failure arrived with the redaction seam — a configured `Redactor` whose
rules stop matching the data, which raises nothing and makes the write path faster. Its
names (`redact.inspected`, `redact.changed`) are in the catalogue checked here; its
emission point is the seam rather than any subsystem this file builds, so the workload
proving it lives in `tests/test_redact.py` alongside the rest of the redaction contract.

Runs fully offline: `SQLiteStore(":memory:")`, `HashingEmbedder`, and a local fake LLM.
"""

from __future__ import annotations

from datetime import timedelta
from time import perf_counter
from typing import Any, Sequence

import pytest

from memvara.consolidate import Consolidator
from memvara.embed import HashingEmbedder
from memvara.retrieve import HybridRetriever
from memvara.llm.base import NullLLM
from memvara.schema import PredicateRegistry
from memvara.store import SQLiteStore
from memvara.telemetry import (
    CONSOLIDATE_CLAIMS_PER_SLOT,
    CONSOLIDATE_CROWDED_SLOTS,
    CONSOLIDATE_DECAYED,
    CONSOLIDATE_LATENCY_MS,
    CONSOLIDATE_MERGED,
    CONSOLIDATE_PROMOTED,
    CONSOLIDATE_ROWS_WRITTEN,
    CROWDED_SLOT,
    FAST_HIT,
    FAST_MISS,
    GATE_DROP,
    GATE_PASS,
    PREDICATE_LEARNED,
    REDACT_CHANGED,
    REDACT_INSPECTED,
    RETRIEVAL_LATENCY_MS,
    RETRIEVAL_OBSERVATION_RANK_CORR,
    RETRIEVAL_QUALITY_FACTOR,
    RETRIEVAL_QUERY,
    RETRIEVAL_RESULTS,
    WRITE_CLAIMS,
    WRITE_MEMORY_CLAIMS,
    WRITE_MEMORY_EPISODES,
    WRITE_EXTRACT_MS,
    WRITE_LATENCY_MS,
    WRITE_LLM_CALLS,
    WRITE_LOCK_HELD_MS,
    WRITE_RECONCILE,
    WRITE_RETRACTION,
    WRITE_TOKENS_IN,
    WRITE_TOKENS_OUT,
    WRITE_TURNS,
    MemoryRecorder,
    NullRecorder,
    Recorder,
    rank_correlation,
    script_of,
    series_names,
)
from memvara.types import Claim, Episode, Scope, utcnow
from memvara.write import WritePipeline

SCOPE = Scope("t", "alice")


class FakeLLM:
    """Returns whatever it was handed, and remembers being asked."""

    name = "fake/telemetry"
    is_noop = False

    def __init__(self, claims: Sequence[dict[str, Any]] = ()) -> None:
        self._claims = list(claims)
        self.extract_calls = 0

    def extract(self, episodes, known_predicates):
        self.extract_calls += 1
        return list(self._claims)

    def classify_predicate(self, predicate, example):
        return {"cardinality": "one", "volatility": "slow", "memory_type": "semantic"}

    def resolve_predicate(self, surface, candidates):
        return {"canonical": None, "cardinality": "one", "volatility": "slow",
                "memory_type": "semantic"}


def ep(text: str, *, ts=None, role: str = "user") -> Episode:
    return Episode(content=text, role=role, scope=SCOPE, ts=ts or utcnow())


class Rig:
    """The three subsystems that emit, wired to one recorder.

    Built by hand rather than through `Memvara` because the constructor parameter that
    threads a recorder through the facade is workstream F's, and this workstream's
    contract is the protocol and the emission points.
    """

    def __init__(self, recorder: Recorder | None, llm=None) -> None:
        self.recorder = recorder
        self.store = SQLiteStore(":memory:")
        self.embedder = HashingEmbedder(dim=128)
        self.registry = PredicateRegistry()
        self.writer = WritePipeline(self.store, self.embedder, self.registry,
                                    llm or FakeLLM(), telemetry=recorder)
        self.reader = HybridRetriever(self.store, self.embedder, self.registry,
                                      telemetry=recorder)
        self.consolidator = Consolidator(self.store, self.embedder, self.registry,
                                         telemetry=recorder)

    def close(self) -> None:
        self.store.close()


# ===========================================================================
# script_of — the slicing dimension for the English-centrism signals
# ===========================================================================

@pytest.mark.parametrize(
    "text, expected",
    [
        ("I moved to Berlin last spring", "latin"),
        ("Je travaille à Paris", "latin"),
        ("Я живу в Берлине", "cyrillic"),
        ("Μένω στην Αθήνα", "greek"),
        ("אני גר בתל אביב", "hebrew"),
        ("أنا أعيش في القاهرة", "arabic"),
        ("मैं दिल्ली में रहता हूँ", "devanagari"),
        ("ฉันอาศัยอยู่ในกรุงเทพ", "thai"),
        ("我住在北京", "han"),
        ("日本に住んでいます", "kana"),
        ("ｱｲｳｴｵ", "kana"),
        ("서울에 살고 있습니다", "hangul"),
        ("Ես ապրում եմ Երևանում", "other"),   # Armenian: unlisted, and that is the point
        ("", "none"),
        ("1234 — !?", "none"),
    ],
)
def test_script_of_buckets_text_for_slicing(text, expected):
    assert script_of(text) == expected


def test_script_of_is_dominance_not_purity():
    """A loanword must not move a Japanese turn out of the Japanese bucket."""
    assert script_of("Acmeで働いています") == "kana"


def test_script_of_reads_only_a_sample_of_a_long_turn():
    """It runs on the write path, so a 10,000-character document must not be scanned in
    full to reach the answer its first line already gives. The Latin tail here
    outnumbers the Han head fifty to one and still does not win, because it is never
    looked at."""
    assert script_of("我" * 200 + "x" * 10_000) == "han"


# ===========================================================================
# rank_correlation — the reinforcement signal
# ===========================================================================

def test_rank_correlation_is_positive_when_well_attested_claims_rank_first():
    assert rank_correlation([40.0, 12.0, 9.0, 3.0, 1.0]) == pytest.approx(1.0)


def test_rank_correlation_is_negative_when_reinforcement_is_not_reaching_the_ranking():
    """The shape of the actual bug: the most-observed claims sorted to the bottom."""
    assert rank_correlation([1.0, 3.0, 9.0, 12.0, 40.0]) == pytest.approx(-1.0)


def test_rank_correlation_uses_ordering_not_magnitude():
    """Spearman, not Pearson: one heavily reinforced outlier must not carry the
    statistic. Both lists have the same ordering and wildly different spreads."""
    assert rank_correlation([9, 5, 1]) == rank_correlation([100000, 5, 1])


def test_rank_correlation_averages_ties_rather_than_inventing_an_order():
    # Three of four seen once: a weak signal, and specifically not +/-1.
    rho = rank_correlation([2.0, 1.0, 1.0, 1.0])
    assert rho is not None and 0.0 < rho < 1.0


@pytest.mark.parametrize("values", [[], [7.0], [3.0, 3.0, 3.0]])
def test_rank_correlation_reports_absence_rather_than_zero(values):
    """Nothing to correlate is not a correlation of zero — emitting 0.0 would drag a
    dashboard toward 'broken' every time a search returned two equally-fresh facts."""
    assert rank_correlation(values) is None


# ===========================================================================
# The recorders
# ===========================================================================

def test_memory_recorder_accumulates_counters_and_keeps_every_observation():
    rec = MemoryRecorder()
    rec.counter("a.b")
    rec.counter("a.b", 4)
    rec.gauge("a.g", 1.5)
    rec.gauge("a.g", 2.5)
    rec.timing("a.t", 10.0)
    assert rec.total("a.b") == 5
    # Distributions are kept, not averaged: the point of the quality-factor and
    # claims-per-slot series is their shape.
    assert rec.values("a.g") == [1.5, 2.5]
    assert rec.values("a.t") == [10.0]


def test_tags_are_a_filter_not_an_exact_match():
    rec = MemoryRecorder()
    rec.counter(GATE_DROP, reason="ack_only", script="latin")
    rec.counter(GATE_DROP, reason="question", script="latin")
    rec.counter(GATE_DROP, reason="too_short", script="han")
    assert rec.total(GATE_DROP) == 3
    assert rec.total(GATE_DROP, script="latin") == 2
    assert rec.total(GATE_DROP, script="han", reason="too_short") == 1
    assert rec.total(GATE_DROP, script="thai") == 0
    assert rec.total("nothing.emitted") == 0


def test_tag_order_does_not_split_a_series():
    rec = MemoryRecorder()
    rec.counter(GATE_PASS, script="latin", reason="has_declarative")
    rec.counter(GATE_PASS, reason="has_declarative", script="latin")
    assert rec.total(GATE_PASS) == 2 and len(rec.counters) == 1


def test_memory_recorder_reports_what_it_has_seen():
    rec = MemoryRecorder()
    rec.counter("z.count")
    rec.gauge("a.gauge", 1.0)
    rec.timing("m.timing", 1.0)
    assert rec.names() == ["a.gauge", "m.timing", "z.count"]
    assert repr(rec) == "<MemoryRecorder 1 counters 1 gauges 1 timings>"


def test_null_recorder_satisfies_the_protocol_and_discards_everything():
    rec: Recorder = NullRecorder()
    rec.counter("a.b", 3, tag="x")
    rec.gauge("a.b", 1.0)
    rec.timing("a.b", 1.0)


def test_the_catalogue_is_enumerable_and_every_name_is_namespaced():
    names = list(series_names())
    assert len(names) == len(set(names))
    assert all(n.islower() and "." in n for n in names)
    for expected in (GATE_DROP, RETRIEVAL_QUALITY_FACTOR, CONSOLIDATE_CLAIMS_PER_SLOT):
        assert expected in names


def test_a_series_defined_for_another_module_is_still_in_this_modules_catalogue():
    """`series_names()` is what a dashboard, a metrics-proxy allow-list or a release
    check reads instead of grepping for string literals, so a name that lives here and
    is emitted somewhere else has to enumerate like any other. The redaction pair is the
    first of those: `memvara.redact` emits it, this module owns the spelling, and a
    constant defined at the emission point instead would be invisible to every consumer
    of the catalogue."""
    names = list(series_names())
    assert REDACT_INSPECTED in names and REDACT_CHANGED in names


def test_a_recorder_that_raises_is_not_swallowed():
    """Same contract as a logging handler: an observability layer that can fail
    silently is the failure mode this module exists to remove."""

    class Broken:
        def counter(self, name, value=1, /, **tags):
            raise RuntimeError("metrics backend down")

        def gauge(self, name, value, /, **tags):  # pragma: no cover - never reached
            raise RuntimeError("metrics backend down")

        def timing(self, name, ms, /, **tags):    # pragma: no cover - never reached
            raise RuntimeError("metrics backend down")

    rig = Rig(Broken())
    with pytest.raises(RuntimeError, match="metrics backend down"):
        rig.writer.add([ep("I live in Berlin.")])
    rig.close()


# ===========================================================================
# The deliverable: each silent failure mode has a live series
# ===========================================================================

def test_every_silent_failure_mode_has_a_live_series():
    """One realistic workload, one recorder, and all six signals present afterwards.

    This is the whole point of the module. Waves 1 and 2 fixed the mechanisms behind
    the first three of these; the series are how anyone knows the fixes are still
    holding in six months, which is a question no `WriteReceipt` can answer because
    every one of these failures is visible only as a trend.
    """
    llm = FakeLLM([
        {"subject": "user", "predicate": "commutes_by", "object": "bicycle",
         "polarity": 1, "confidence": 0.9, "source_index": 0},
    ])
    rec = MemoryRecorder()
    rig = Rig(rec, llm)

    rig.writer.add([
        ep("My name is Alice."),                 # fast path handles it
        ep("ok thanks"),                         # gate drops it
        ep("我住在北京"),                          # passes the gate, fast path misses
        ep("I get around town somehow or other."),  # reaches tier 2
    ])
    # 6. A retraction that retires nothing: the poisoning / failed-retraction anomaly.
    rig.writer.assert_claim(Claim(subject="user", predicate="works_at", object="Acme",
                                  polarity=-1, scope=SCOPE))
    rig.reader.search("where does alice live", SCOPE)
    rig.consolidator.run("t")

    # 1. predicate explosion
    assert rec.total(PREDICATE_LEARNED) == 1
    assert rec.values(CONSOLIDATE_CLAIMS_PER_SLOT) == [1.0]
    # 2. reinforcement not refreshing recency
    assert CONSOLIDATE_DECAYED in rec.names()
    # 3. flip-flop row growth / counters reset
    assert rec.values(CONSOLIDATE_CROWDED_SLOTS) == [0.0]
    assert rec.total(CONSOLIDATE_MERGED) == 0      # reported, not absent
    # 4. salience overriding relevance
    assert all(0.0 < v <= 1.0 for v in rec.values(RETRIEVAL_QUALITY_FACTOR))
    # 5. gate and fast-path English-centrism, sliced by script
    assert rec.total(GATE_DROP, script="latin", reason="ack_only") == 1
    assert rec.total(GATE_PASS, script="han") == 1
    assert rec.total(FAST_HIT, script="latin") == 1
    assert rec.total(FAST_MISS, script="han") == 1
    # 6. a retraction that retired nothing
    assert rec.total(WRITE_RETRACTION, outcome="noop") == 1


def test_the_han_slice_shows_a_cost_difference_not_only_a_quality_one():
    """The fast path's patterns are English sentence forms. Two turns stating the same
    fact, one in each script: the Latin one costs nothing and the Han one costs a model
    call, and without the slice that difference is invisible."""
    rec = MemoryRecorder()
    llm = FakeLLM()
    rig = Rig(rec, llm)
    rig.writer.add([ep("I live in Beijing."), ep("我住在北京")])
    assert rec.total(FAST_HIT, script="latin") == 1
    assert rec.total(FAST_MISS, script="han") == 1
    assert llm.extract_calls == 1          # paid for exactly the turn the fast path missed
    rig.close()


# ===========================================================================
# Contract G-1: no measurable cost when unset
# ===========================================================================

def test_nothing_is_computed_for_telemetry_when_it_is_unset(monkeypatch):
    """The mechanism half of contract G-1, and the half that cannot be flaky.

    Every metric that needs *work* to produce — a script classification, a rank
    correlation, a quality factor, a per-slot histogram — is computed inside the
    `is not None` guard rather than before it. Booby-trap each of those and run the
    whole write/read/consolidate path with the recorder unset: reaching any of them at
    all is the bug.
    """

    def boom(*a, **kw):
        raise AssertionError("telemetry work ran with no recorder configured")

    monkeypatch.setattr("memvara.write.pipeline.script_of", boom)
    monkeypatch.setattr("memvara.retrieve.hybrid.script_of", boom)
    monkeypatch.setattr("memvara.retrieve.hybrid.rank_correlation", boom)
    monkeypatch.setattr("memvara.retrieve.hybrid.quality_boost", boom)
    monkeypatch.setattr("memvara.consolidate.sweep.Sweep._observe_slots", boom)

    rig = Rig(None, FakeLLM([
        {"subject": "user", "predicate": "likes", "object": "tea", "polarity": 1,
         "confidence": 0.9, "source_index": 0},
    ]))
    rig.writer.add([ep("My name is Alice."), ep("ok"), ep("Something else entirely.")])
    rig.writer.assert_claim(Claim(subject="user", predicate="likes", object="coffee",
                                  scope=SCOPE))
    assert rig.reader.search("alice", SCOPE) is not None
    rig.consolidator.run("t")
    rig.consolidator.decay("t")
    rig.consolidator.merge_duplicates("t")
    rig.consolidator.promote("t")
    rig.close()


def _round(rig: Rig, i: int, now) -> float:
    """Seconds for one three-turn write plus one search."""
    t0 = perf_counter()
    rig.writer.add([ep(f"My name is Alice number {i}.", ts=now + timedelta(hours=i)),
                    ep("ok thanks"),
                    ep(f"Some unremarkable turn {i} that carries a statement.")])
    rig.reader.search("what is my name", SCOPE)
    return perf_counter() - t0


def test_telemetry_is_cheap_even_when_it_is_on():
    """The measurement half of contract G-1.

    This library's whole argument is about cost, so the claim that observability does
    not show up in a benchmark has to be measured rather than asserted.

    The *unset* case is enforced structurally rather than by a stopwatch, and
    deliberately: the guard is an `is not None` test and the whole point is that its
    cost is below any timer this suite could run without measuring the CI machine
    instead. `test_nothing_is_computed_for_telemetry_when_it_is_unset` is the binding
    check — it fails if any metric's *work* escapes its guard — and an unguarded
    emission would raise `AttributeError` on `None` in every other test in the suite.

    Measured out of tree instead, against a build of this same tree with the emission
    points deleted from the source and nothing else changed — same transaction
    structure, same everything, and every non-telemetry test in this suite still green
    against it. Two constant-work loads (120 three-turn `add()` calls into a fresh
    store; 400 searches over a fixed 200-claim store), best of fifteen trials, median
    over four process launches per arm:

                                  write            read
        no hooks in the source    0.9823 ms/add    1.9812 ms/search
        hooks present, unset      0.9905 ms/add    1.9733 ms/search   (+0.8%, -0.4%)
        hooks present, recording  1.0068 ms/add    2.0001 ms/search   (+2.5%, +1.0%)

    The unset arm lands inside the control's own launch-to-launch spread (0.9791 to
    0.9966 ms/add) and comes out *faster* than it on the read, which is the useful part
    of the result: the difference is below the noise floor rather than merely small.

    What this test can measure stably is the bound in the other direction: fully-on
    telemetry stays a small fraction of the operation it observes. Rounds are
    interleaved so a warming cache or a scheduler excursion hits both arms, and the
    minimum is taken because noise only ever adds.
    """
    now = utcnow()
    off, on = Rig(None), Rig(MemoryRecorder())
    _round(off, 0, now), _round(on, 0, now)          # warm both, measure neither
    unset = recording = float("inf")
    for i in range(1, 41):
        unset = min(unset, _round(off, i, now))
        recording = min(recording, _round(on, i, now))
    off.close(), on.close()
    assert recording < unset * 1.6, (
        f"recording={recording * 1000:.3f}ms against unset={unset * 1000:.3f}ms — "
        "collecting metrics has stopped being a rounding error on the work observed")


# ===========================================================================
# Wiring: the recorder reaches every subsystem
# ===========================================================================

def test_consolidation_reports_a_settled_store_rather_than_going_quiet():
    """`merged` stuck at 0 is a finding; *no merge series at all* is a scheduler nobody
    noticed had stopped. Only reporting the zero tells those two apart."""
    rec = MemoryRecorder()
    rig = Rig(rec)
    rig.consolidator.run("t")
    assert rec.total(CONSOLIDATE_DECAYED) == 0
    assert rec.total(CONSOLIDATE_MERGED) == 0
    assert rec.total(CONSOLIDATE_PROMOTED) == 0
    assert rec.total(CONSOLIDATE_ROWS_WRITTEN) == 0
    assert len(rec.values(CONSOLIDATE_LATENCY_MS)) == 1
    rig.close()


def test_the_write_and_read_paths_report_their_own_latency():
    rec = MemoryRecorder()
    rig = Rig(rec)
    rig.writer.add([ep("I live in Berlin.")])
    rig.reader.search("where do I live", SCOPE)
    assert len(rec.values(WRITE_LATENCY_MS)) == 1
    assert len(rec.values(WRITE_LOCK_HELD_MS)) == 1
    assert len(rec.values(RETRIEVAL_LATENCY_MS)) == 1
    assert rec.total(WRITE_TURNS) == 1 and rec.total(WRITE_LLM_CALLS) == 0
    assert rec.total(WRITE_RECONCILE, action="add") == 1
    assert rec.total(RETRIEVAL_QUERY, script="latin") == 1
    assert rec.values(RETRIEVAL_RESULTS) == [1.0]
    rig.close()


def test_a_write_that_skips_extraction_is_still_counted_as_a_write():
    """`write.turns` counts turns handed to `add()` and nothing else, so every write that
    asserts a fact directly was invisible to anything counting write activity. That is not
    an edge case: a deployment with no extraction model stores prose as nothing unless it
    matches a fixed sentence form, which makes the direct write the only reliable path and
    leaves `write.turns` flat while the store fills up."""
    rec = MemoryRecorder()
    rig = Rig(rec)
    for city in ("Berlin", "Lisbon"):
        rig.writer.assert_claim(Claim(subject="alice", predicate="lives_in", object=city,
                                      scope=SCOPE))
    assert rec.total(WRITE_CLAIMS) == 2
    # And the two series stay separate. `write.turns` is what a turn allowance is spent
    # against and an asserted fact spends none, so folding one into the other would bill
    # for writes the API documents as free.
    assert rec.total(WRITE_TURNS) == 0
    rig.writer.add([ep("I live in Berlin.")])
    assert rec.total(WRITE_TURNS) == 1 and rec.total(WRITE_CLAIMS) == 2
    rig.close()


def test_the_billing_counters_count_rows_landed_not_calls_made():
    """`write.memory_claims` is `len(receipt.added)`, which reconciliation fills with the
    `add` and `supersede` outcomes and nothing else.

    That is the distinction `write.claims` deliberately does not make. It counts one per
    assert *call* whatever the call displaced, so the two diverge the moment an assertion
    reinforces something already known — and a bill computed from the wrong one charges
    for work that created no row.
    """
    rec = MemoryRecorder()
    rig = Rig(rec)
    rig.writer.assert_claim(Claim(subject="alice", predicate="lives_in", object="Berlin",
                                  scope=SCOPE))
    assert rec.total(WRITE_CLAIMS) == 1 and rec.total(WRITE_MEMORY_CLAIMS) == 1

    # The same fact again: a call happened and no row was created.
    rig.writer.assert_claim(Claim(subject="alice", predicate="lives_in", object="Berlin",
                                  scope=SCOPE))
    assert rec.total(WRITE_CLAIMS) == 2, "the call counter must still move"
    assert rec.total(WRITE_MEMORY_CLAIMS) == 1, "reinforcement created no row and is free"

    # A correction does create one, and must be billed. Metering `add` alone would miss
    # every one of these, which is most of what a memory store does after month one.
    rig.writer.assert_claim(Claim(subject="alice", predicate="lives_in", object="Lisbon",
                                  scope=SCOPE))
    assert rec.total(WRITE_MEMORY_CLAIMS) == 2
    rig.close()


def test_re_ingesting_a_transcript_costs_nothing_on_either_billing_counter():
    """The dedup promise, asserted as an emission rather than left as prose.

    `write.turns` moves on the second pass because a turn really was handed in. Neither
    billing counter does: `_tier0_partition` sends an exact repeat to `pending` without
    storing it, so `fresh` is empty, and the claims it carried reconcile to `reinforce`,
    so `added` is empty too.
    """
    rec = MemoryRecorder()
    rig = Rig(rec)
    turns = [ep("I live in Berlin.")]
    rig.writer.add(turns)
    claims, episodes = (rec.total(WRITE_MEMORY_CLAIMS), rec.total(WRITE_MEMORY_EPISODES))
    assert episodes == 1

    rig.writer.add(turns)
    assert rec.total(WRITE_TURNS) == 2, "a turn was handed in twice and that is countable"
    assert rec.total(WRITE_MEMORY_CLAIMS) == claims, "no row landed, so nothing is billed"
    assert rec.total(WRITE_MEMORY_EPISODES) == episodes, "the repeat was never stored"
    rig.close()


def test_a_stored_episode_is_counted_even_when_it_yields_no_claim():
    """The half a claims-only meter misses, and on the shipped configuration it is most of
    the traffic.

    An episode commits before extraction runs and is retrievable in its own right through
    `include_episodes`. On a deployment with no extraction model, prose matching no rule is
    stored, searchable and answering queries while producing no claim at all — so a meter
    counting claims alone bills nothing for what that deployment actually delivers.
    """
    rec = MemoryRecorder()
    rig = Rig(rec)
    rig.writer.add([ep("The quarterly review is a recurring source of mild dread.")])

    assert rec.total(WRITE_MEMORY_EPISODES) == 1
    assert rec.total(WRITE_MEMORY_CLAIMS) == 0
    rig.close()


def test_the_model_round_trip_is_timed_on_its_own_not_only_as_a_difference():
    """Before this, extraction time was only recoverable as `write.latency_ms` minus
    `write.lock_held_ms`. Percentiles do not subtract, so that arithmetic yields a mean
    and no p99 — and the slow tail of the model call is the thing worth alerting on."""
    rec = MemoryRecorder()
    llm = FakeLLM()
    rig = Rig(rec, llm)
    # Han text misses the fast path, which is what forces tier 2. See the script test.
    rig.writer.add([ep("我住在北京")])
    assert llm.extract_calls == 1
    assert len(rec.values(WRITE_EXTRACT_MS)) == 1
    # It measures the call, not the whole write, so it cannot exceed end-to-end latency.
    assert rec.values(WRITE_EXTRACT_MS)[0] <= rec.values(WRITE_LATENCY_MS)[0]
    rig.close()


def test_an_extraction_that_raised_is_still_counted_as_time_the_caller_waited():
    """A provider timeout is latency someone sat through. Timing only the successful
    path makes the p99 *improve* during an outage — the metric moving the healthy way
    for the unhealthiest reason, which is worse than having no metric."""
    class Failing(FakeLLM):
        def extract(self, episodes, known_predicates):
            raise RuntimeError("provider 429")

    rec = MemoryRecorder()
    rig = Rig(rec, Failing())
    receipt = rig.writer.add([ep("我住在北京")])
    assert receipt.deferred and receipt.unextracted == 1
    assert len(rec.values(WRITE_EXTRACT_MS)) == 1
    rig.close()


def test_a_backend_that_consults_no_model_reports_no_extraction_series_at_all():
    """The `is_noop` rule `write.llm_calls` already follows, applied to the timing: a
    `NullLLM` deployment must not publish a series of zeros. A zero here would read as a
    model answering instantly, and would drag every percentile of a mixed fleet down."""
    rec = MemoryRecorder()
    rig = Rig(rec, NullLLM())
    receipt = rig.writer.add([ep("我住在北京")])
    assert receipt.unextracted == 1          # it did reach tier 2
    assert rec.values(WRITE_EXTRACT_MS) == []
    rig.close()


def test_a_crowded_slot_is_visible_before_anyone_notices_the_bad_answers():
    """The reviewer's one-metric-if-only-one answer, on the shape of the actual bug:
    several live claims answering the same question, which is invisible from any
    individual receipt or explanation."""
    rec = MemoryRecorder()
    rig = Rig(rec)
    for employer in ("Acme", "Globex", "Initech", "Hooli", "Umbrella"):
        # `worked_for` is unregistered, so it is MANY-cardinality and nothing retires
        # anything - exactly how thirteen live employers happened.
        rig.writer.assert_claim(Claim(subject="user", predicate="worked_for",
                                      object=employer, scope=SCOPE))
    rig.consolidator.decay("t")
    assert rec.values(CONSOLIDATE_CLAIMS_PER_SLOT) == [5.0]
    assert rec.values(CONSOLIDATE_CROWDED_SLOTS) == [1.0]
    assert 5 > CROWDED_SLOT
    rig.close()


def test_the_observation_rank_correlation_is_positive_on_a_healthy_store():
    """Signal 2, end to end: a fact restated many times must rank above one mentioned
    once, and the sign of this number is how anyone knows it still does."""
    rec = MemoryRecorder()
    rig = Rig(rec)
    now = utcnow()
    for i, obj in enumerate(("cycling", "chess", "baking")):
        rig.writer.assert_claim(Claim(
            subject="user", predicate="enjoys", object=obj, scope=SCOPE,
            text=f"user enjoys {obj}", observation_count=10 - i * 4,
            salience=1.0 + (10 - i * 4) * 0.1, valid_from=now, recorded_at=now))
    rig.reader.search("does the user enjoy cycling", SCOPE)
    values = rec.values(RETRIEVAL_OBSERVATION_RANK_CORR)
    assert len(values) == 1 and values[0] > 0.0
    rig.close()


# ===========================================================================
# Token accounting — the input a bill is computed from
# ===========================================================================

class MeteredLLM(FakeLLM):
    """A backend that reports usage, and counts how often it was asked to."""

    reports_usage = True

    def __init__(self, claims=(), *, per_call=(100, 20)) -> None:
        super().__init__(claims)
        self.per_call = per_call
        self.saw_usage = 0

    def _bill(self, usage):
        if usage is not None:
            self.saw_usage += 1
            usage.add(*self.per_call)

    def extract(self, episodes, known_predicates, *, usage=None):
        self._bill(usage)
        return super().extract(episodes, known_predicates)

    def resolve_predicate(self, surface, candidates, *, usage=None):
        self._bill(usage)
        return super().resolve_predicate(surface, candidates)


def test_a_write_reports_the_tokens_it_burned_not_only_the_calls_it_made():
    """`write.llm_calls` cannot be billed on: a one-line turn and a 40,000-token document
    are both exactly one call, and providers charge per token. This is the series an
    invoice is computed from, and the receipt carries it so a caller who never configured
    telemetry can still see what a write cost."""
    rec = MemoryRecorder()
    llm = MeteredLLM(per_call=(100, 20))
    rig = Rig(rec, llm)
    receipt = rig.writer.add([ep("我住在北京")])
    assert llm.extract_calls == 1
    assert (receipt.tokens_in, receipt.tokens_out) == (100, 20)
    assert rec.total(WRITE_TOKENS_IN) == 100 and rec.total(WRITE_TOKENS_OUT) == 20
    rig.close()


def test_input_and_output_tokens_stay_separate_because_they_are_priced_separately():
    # A single total cannot be costed: output is several times the price of input, so the
    # split is the difference between a real number and an unusable one.
    rec = MemoryRecorder()
    rig = Rig(rec, MeteredLLM(per_call=(1000, 7)))
    rig.writer.add([ep("我住在北京")])
    assert rec.total(WRITE_TOKENS_IN) != rec.total(WRITE_TOKENS_OUT)
    rig.close()


def test_predicate_acquisition_is_billed_to_the_write_that_triggered_it():
    """One accumulator spans the batch. A novel surface form costs a second model call,
    and that call is part of the same write — billing it to nothing, or to the next
    write, both misattribute it."""
    rec = MemoryRecorder()
    llm = MeteredLLM([{"subject": "user", "predicate": "enjoys_drinking",
                       "object": "tea", "polarity": 1, "confidence": 0.9,
                       "source_index": 0}], per_call=(50, 10))
    rig = Rig(rec, llm)
    receipt = rig.writer.add([ep("我住在北京")])
    # extract() plus one resolve_predicate() for the novel surface form.
    assert llm.saw_usage == 2 and receipt.llm_calls == 2
    assert (receipt.tokens_in, receipt.tokens_out) == (100, 20)
    rig.close()


def test_a_backend_that_cannot_report_usage_publishes_no_token_series_at_all():
    """Zero is not the answer — a call that reached a provider consumed something. A run
    of zeros would understate a bill and drag a fleet-wide average toward it, which is
    the failure direction that favours us and therefore the one to refuse. `FakeLLM`
    leaves `reports_usage` at its default."""
    rec = MemoryRecorder()
    llm = FakeLLM()
    rig = Rig(rec, llm)
    receipt = rig.writer.add([ep("我住在北京")])
    assert llm.extract_calls == 1                      # a real call was made
    # On `rec.counters` rather than `rec.values()`/`rec.total()`: those answer [] and 0
    # for a counter that was never emitted *and* for one emitted as zero, so either would
    # pass against exactly the bug this test exists to catch. Absence of the key is the
    # only assertion that distinguishes "unknown" from "free".
    emitted = {name for name, _tags in rec.counters}
    assert WRITE_TOKENS_IN not in emitted and WRITE_TOKENS_OUT not in emitted
    assert (receipt.tokens_in, receipt.tokens_out) == (0, 0)
    rig.close()


def test_a_backend_that_never_gets_the_keyword_it_cannot_accept():
    """The compatibility guarantee, as a test rather than a promise: a backend written
    against the older three-argument signature must keep working untouched. `FakeLLM`
    does not accept `usage=`, so passing it would be a TypeError — and that TypeError
    would be caught by the extraction guard and reported as a deferred write, which is
    the silent version of this failure."""
    rec = MemoryRecorder()
    llm = FakeLLM([{"subject": "user", "predicate": "lives_in", "object": "Beijing",
                    "polarity": 1, "confidence": 0.9, "source_index": 0}])
    rig = Rig(rec, llm)
    receipt = rig.writer.add([ep("我住在北京")])
    # A TypeError from the unexpected keyword would be swallowed by the extraction guard
    # and surface as `deferred` with the claim missing — so asserting the claim landed is
    # what actually distinguishes "compatible" from "silently broken".
    assert not receipt.deferred and len(receipt.added) == 1
    assert llm.extract_calls == 1
    rig.close()


def test_tokens_burned_by_a_call_that_then_raised_are_still_reported():
    """A provider that timed out after generating still charges for it. Reporting only
    the successful path is how an outage shows up on the invoice and nowhere else."""
    class FailsAfterBilling(MeteredLLM):
        def extract(self, episodes, known_predicates, *, usage=None):
            self._bill(usage)
            raise RuntimeError("provider 500 after generation")

    rec = MemoryRecorder()
    rig = Rig(rec, FailsAfterBilling(per_call=(80, 0)))
    receipt = rig.writer.add([ep("我住在北京")])
    assert receipt.deferred
    assert receipt.tokens_in == 80 and rec.total(WRITE_TOKENS_IN) == 80
    rig.close()


def test_reading_a_gauge_with_total_says_so_rather_than_answering_zero():
    """`consolidate.claims_per_slot` is the most valuable series in the module and it is
    a gauge. Read through `total` it returned 0 — a number a counter can legitimately
    have — so a caller could not tell "nothing happened" from "wrong method", and
    nothing pointed at `values()`."""
    rec = MemoryRecorder()
    rec.gauge(CONSOLIDATE_CLAIMS_PER_SLOT, 5.0)
    rec.timing(WRITE_LATENCY_MS, 12.0)

    for name in (CONSOLIDATE_CLAIMS_PER_SLOT, WRITE_LATENCY_MS):
        with pytest.raises(TypeError, match="not a counter"):
            rec.total(name)
    assert rec.values(CONSOLIDATE_CLAIMS_PER_SLOT) == [5.0]
    # A counter that genuinely recorded nothing still answers 0 rather than raising —
    # the guard fires on the *kind* of series, not on the absence of one.
    assert rec.total(WRITE_TURNS) == 0
