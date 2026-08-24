"""WritePipeline: the cost ladder, measured.

Every test here asserts on `receipt.llm_calls`. That is not incidental — the entire
reason this subsystem exists is that mem0 spends one LLM call per write, and a test that
only checks the claims came out would pass just as happily on a design that called a
model on every turn. The fake LLM below therefore counts calls and records batch sizes,
and refuses to be used without those numbers being checked.

Runs fully offline: SQLiteStore(":memory:"), HashingEmbedder, and the local fake.
"""

from __future__ import annotations

import threading
import time
import warnings
from contextlib import contextmanager
from datetime import timedelta
from time import perf_counter
from typing import Any, Sequence

import pytest

import pathlib

from memvara import Memvara
from memvara.embed import HashingEmbedder
from memvara.llm.base import NullLLM
from memvara.retrieve import HybridRetriever
from memvara.schema import Cardinality, PredicateRegistry
from memvara.store import SQLiteStore
from memvara.telemetry import (
    FAST_HIT,
    FAST_MISS,
    GATE_DROP,
    GATE_PASS,
    PREDICATE_ACCUMULATED,
    PREDICATE_ALIAS,
    PREDICATE_CAPPED,
    PREDICATE_LEARNED,
    WRITE_EMBEDDING_REJECTED,
    WRITE_MEMORY_CLAIMS,
    WRITE_MEMORY_EPISODES,
    WRITE_LATENCY_MS,
    WRITE_LOCK_HELD_MS,
    WRITE_RECONCILE,
    WRITE_RETRACTION,
    MemoryRecorder,
)
from memvara.types import Claim, Derivation, Episode, Scope, utcnow
from memvara.write import SalienceGate, WritePipeline

ROOT = pathlib.Path(__file__).resolve().parent.parent


class CountingLLM:
    """Local fake. Counts every call, because the call count is the thing under test."""

    name = "fake/counting"

    def __init__(self, claims: Sequence[dict[str, Any]] | None = None,
                 responder=None,
                 classification: dict[str, str] | None = None) -> None:
        self._claims = list(claims or [])
        self._responder = responder
        self._classification = classification or {
            "cardinality": "many", "volatility": "slow", "memory_type": "semantic",
        }
        self.extract_calls = 0
        self.classify_calls = 0
        self.classified: list[str] = []
        self.batch_sizes: list[int] = []
        self.seen_known_predicates: list[Sequence[str]] = []

    def extract(self, episodes, known_predicates):
        self.extract_calls += 1
        self.batch_sizes.append(len(episodes))
        self.seen_known_predicates.append(list(known_predicates))
        if self._responder is not None:
            return self._responder(episodes)
        return list(self._claims)

    def classify_predicate(self, predicate, example):
        self.classify_calls += 1
        self.classified.append(predicate)
        return dict(self._classification)

    @property
    def total_calls(self) -> int:
        return self.extract_calls + self.classify_calls


class ResolvingLLM(CountingLLM):
    """A backend that implements contract D, so acquisition can merge rather than
    classify. Kept separate from `CountingLLM` because the pipeline has to keep working
    with backends that predate `resolve_predicate`, and that fallback is worth testing
    against a fake that genuinely lacks the method."""

    name = "fake/resolving"

    def __init__(self, *args, canonical: str | None = None, **kw) -> None:
        super().__init__(*args, **kw)
        self._canonical = canonical
        self.resolve_calls = 0
        self.offered: list[Sequence[str]] = []

    def resolve_predicate(self, surface, candidates):
        self.resolve_calls += 1
        self.offered.append(list(candidates))
        return {"canonical": self._canonical, **self._classification}

    @property
    def total_calls(self) -> int:
        return self.extract_calls + self.classify_calls + self.resolve_calls


def build(llm=None, **kw):
    store = SQLiteStore(":memory:")
    registry = PredicateRegistry()
    pipe = WritePipeline(store, HashingEmbedder(), registry, llm or CountingLLM(), **kw)
    return pipe, store, registry


SCOPE = Scope("acme", "alice")


def ep(content: str, role: str = "user", **kw) -> Episode:
    kw.setdefault("scope", SCOPE)
    return Episode(content=content, role=role, **kw)


def live(store, tenant: str = "acme"):
    # `is_live` rather than `iter_claims`'s own filter. That one reads the belief axis
    # alone — which is exactly what `include_invalidated` names — and a superseded claim
    # now keeps its belief axis open, because it was true and we were never wrong about
    # it. Without this the helper counted every city the user had ever lived in as live.
    return sorted((c.predicate, c.object)
                  for c in store.iter_claims(tenant) if c.is_live())


# --- the headline number -----------------------------------------------------

def test_twenty_turns_three_facts_costs_exactly_one_call():
    """20 turns, 3 of which carry facts: one batched extract call, 17 skipped."""
    facts = [
        "My daughter Maya started third grade at Oakwood Elementary this fall.",
        "We adopted a rescue greyhound last month.",
        "The team standardized on Rust for the new billing service.",
    ]
    chitchat = [
        ep("ok"), ep("thanks"), ep("sounds good"), ep("got it"), ep("perfect"),
        ep("great, thanks"),
        ep("What time is it?"), ep("Can you help me with something?"),
        ep("Where should we start?"), ep("How does that work?"),
        ep("Is that right?"), ep("Why did that happen?"),
        ep("Here is a summary of the plan.", role="assistant"),
        ep("Let me look that up for you.", role="assistant"),
        ep("I have updated the document.", role="assistant"),
        ep("Anything else I can do?", role="assistant"),
        ep("Happy to help.", role="assistant"),
    ]
    assert len(chitchat) == 17

    def responder(episodes):
        return [{"subject": "user", "predicate": p, "object": o, "polarity": 1,
                 "memory_type": "semantic", "confidence": 0.8, "source_index": i}
                for i, (p, o) in enumerate([("likes", "oakwood elementary"),
                                            ("owns_pet", "greyhound"),
                                            ("goal", "learn rust")])]

    llm = CountingLLM(responder=responder)
    pipe, store, _ = build(llm)

    episodes = [ep(f) for f in facts] + chitchat
    receipt = pipe.add(episodes)

    assert receipt.skipped == 17
    assert llm.extract_calls == 1
    assert llm.classify_calls == 0
    assert receipt.llm_calls == 1
    # One call for the whole batch, not one per surviving turn.
    assert llm.batch_sizes == [3]
    assert len(receipt.added) == 3
    assert len(receipt.episode_ids) == 20
    assert receipt.latency_ms >= 0.0
    assert receipt.deferred is False
    store.close()


def test_pure_chitchat_never_reaches_the_model():
    llm = CountingLLM()
    pipe, store, _ = build(llm)
    receipt = pipe.add([ep("ok"), ep("thanks!"), ep("sure thing"), ep("no worries"),
                        ep("Is that done?"), ep("Sounds good to me")])
    assert receipt.llm_calls == 0
    assert llm.total_calls == 0
    assert receipt.skipped == 6
    assert receipt.added == []
    store.close()


# --- tier 1: the fast path handles the common facts for free -----------------

def test_fast_path_facts_cost_nothing():
    llm = CountingLLM()
    pipe, store, _ = build(llm)
    receipt = pipe.add([ep("My name is Goldy."), ep("I live in Berlin."),
                        ep("I work at Acme.")])

    assert receipt.llm_calls == 0
    assert llm.total_calls == 0
    assert len(receipt.added) == 3
    assert all(c.derivation is Derivation.FAST_PATH for c in receipt.added)
    assert live(store) == [("lives_in", "Berlin"), ("name", "Goldy"),
                           ("works_at", "Acme")]
    store.close()


def test_moving_city_supersedes_with_no_llm_call():
    llm = CountingLLM()
    pipe, store, _ = build(llm)

    first = pipe.add([ep("I live in Berlin.")])
    second = pipe.add([ep("I moved to Lisbon.")])

    assert (first.llm_calls, second.llm_calls) == (0, 0)
    assert llm.total_calls == 0
    assert live(store) == [("lives_in", "Lisbon")]

    berlin = first.added[0]
    stored = store.get_claim(berlin.id)
    # Ended, not retired. The extraction path only ever reports what someone said about
    # the world, so it is never in a position to mark a stored fact as an error — and
    # a model deciding that on its own is the thing this write path exists not to do.
    assert stored.state == "ended"
    assert stored.valid_to is not None and stored.invalidated_at is None
    assert stored.invalidated_by == second.added[0].id
    assert [c.id for c in second.invalidated] == [berlin.id]
    store.close()


def test_superseded_claim_is_still_visible_as_of_the_past():
    pipe, store, _ = build()
    t0 = utcnow() - timedelta(days=10)
    first = pipe.add([ep("I live in Berlin.", ts=t0)])
    before = utcnow()
    pipe.add([ep("I moved to Lisbon.")])

    berlin = store.get_claim(first.added[0].id)
    assert berlin.is_live(before) is True     # what we believed then
    assert berlin.is_live() is False          # what we believe now
    store.close()


def test_multi_valued_preferences_both_stay_live():
    llm = CountingLLM()
    pipe, store, _ = build(llm)
    pipe.add([ep("I like coffee.")])
    receipt = pipe.add([ep("I like tea.")])

    assert receipt.llm_calls == 0
    assert llm.total_calls == 0
    assert receipt.invalidated == []
    # "likes" is multi-valued, so a second answer is not a contradiction.
    assert live(store) == [("likes", "coffee"), ("likes", "tea")]
    store.close()


def test_retraction_leaves_no_live_negative_claim():
    llm = CountingLLM()
    pipe, store, _ = build(llm)
    added = pipe.add([ep("I work at Acme.")])
    receipt = pipe.add([ep("I no longer work at Acme.")])

    assert receipt.llm_calls == 0
    assert llm.total_calls == 0
    assert [c.id for c in receipt.invalidated] == [added.added[0].id]
    assert receipt.added == []
    assert live(store) == []
    # "I no longer work at Acme" ends the employment; it does not say we were wrong to
    # have recorded it. So the job is over on the world clock and still believed, which
    # is what keeps "where did they work last year" answerable after they leave.
    ended = store.get_claim(added.added[0].id)
    assert ended.state == "ended" and ended.invalidated_at is None
    store.close()


# --- tier 0: repeats are free ------------------------------------------------

def test_re_adding_identical_content_reinforces_without_a_call():
    llm = CountingLLM()
    pipe, store, _ = build(llm)

    first = pipe.add([ep("I live in Berlin.")])
    claim_id = first.added[0].id
    before = store.stats()["claims"]

    receipt = pipe.add([ep("I live in Berlin.")])

    assert receipt.llm_calls == 0
    assert llm.total_calls == 0
    assert receipt.added == []
    assert store.stats()["claims"] == before          # no new claim
    assert store.get_claim(claim_id).observation_count == 2
    assert [c.id for c in receipt.reinforced] == [claim_id]
    assert receipt.skipped == 1
    store.close()


def test_repeated_adds_keep_incrementing_the_same_claim():
    pipe, store, _ = build()
    first = pipe.add([ep("I work at Acme.")])
    for _ in range(4):
        pipe.add([ep("I work at Acme.")])
    stored = store.get_claim(first.added[0].id)
    assert stored.observation_count == 5
    assert store.stats()["claims"] == 1
    store.close()


def test_duplicate_episode_is_not_stored_twice():
    pipe, store, _ = build()
    pipe.add([ep("I live in Berlin.")])
    pipe.add([ep("I live in Berlin.")])
    assert store.stats()["episodes"] == 1
    store.close()


def test_near_duplicate_restatement_reinforces_instead_of_adding():
    llm = CountingLLM()
    pipe, store, _ = build(llm)
    first = pipe.add([ep("I live in Berlin.")])
    claim = first.added[0]

    # Byte-different from the original turn, but identical to the claim's own rendering,
    # so it lands above the cosine threshold and is recognised as a restatement.
    receipt = pipe.add([ep(claim.text)])

    assert receipt.llm_calls == 0
    assert llm.total_calls == 0
    assert receipt.added == []
    assert [c.id for c in receipt.reinforced] == [claim.id]
    assert store.get_claim(claim.id).observation_count == 2
    store.close()


def test_near_dup_threshold_of_one_disables_the_shortcut():
    # Guard against the threshold being ignored: at 1.0 only an exact vector match counts.
    pipe, store, _ = build(near_dup_threshold=1.01)
    first = pipe.add([ep("I live in Berlin.")])
    receipt = pipe.add([ep(first.added[0].text)])
    assert receipt.reinforced == [] or receipt.added != []
    store.close()


# --- tier 0: finding what a repeated turn already produced --------------------
#
# The repeat branch has to answer "which claims came from this turn?". It used to do that
# by scanning every claim in the tenant, which was affordable only while repeats were
# rare — and the redaction seam is precisely what stops them being rare, since two turns
# differing only inside a redacted span are one turn once the redactor has run.

class StoreWithoutReverseIndex:
    """A third-party `Store` written before `claims_citing` existed.

    Delegates everything else to a real store, so the only difference under test is the
    one missing method. `__getattr__` runs only for attributes not found normally, which
    is what makes the delegation total.
    """

    def __init__(self, inner) -> None:
        self._inner = inner

    def __getattr__(self, name: str):
        if name == "claims_citing":
            raise AttributeError(name)
        return getattr(self._inner, name)


def test_an_exact_repeat_finds_its_claims_through_the_reverse_index():
    """The lookup this branch is built on. Asserted by making the scan it replaced
    impossible: `iter_claims` raises, so a reinforcement can only have come from
    `claims_citing`."""
    pipe, store, _ = build()
    first = pipe.add([ep("I live in Berlin.")])
    claim_id = first.added[0].id
    store.iter_claims = lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError("the tenant was scanned"))

    receipt = pipe.add([ep("I live in Berlin.")])

    assert [c.id for c in receipt.reinforced] == [claim_id]
    assert store.get_claim(claim_id).observation_count == 2
    store.close()


def test_a_fact_restated_all_year_keeps_every_turn_that_supports_it():
    """The one property the answer to unbounded `sources` had to preserve.

    Provenance is cumulative and nothing caps it: a fact the user restates in new words
    every day for a year accumulates 365 source ids, each of which is rewritten into the
    claim's row on every observation and maintained as an edge. Capping the list is the
    cheap fix and it is the wrong one — `why()` silently not naming a turn it was derived
    from, in the library whose pitch is that provenance always resolves, is worse than a
    slow write. So the write was made cheap instead (222.4 us at 365 sources against
    95.6, by finding the edge delta from the array already in hand rather than by reading
    the edges back), and the list still holds everything.

    A hundred restatements rather than 365, because the property is "none are dropped"
    and the number only changes the runtime.
    """
    pipe, store, _ = build()
    first = pipe.add([ep("I live in Berlin.")])
    claim_id = first.added[0].id

    for day in range(100):
        # Byte-different every time, so each is a genuinely new turn rather than the
        # exact-repeat branch, and each one adds a source.
        pipe.add([ep(f"Just to say again on day {day}: I live in Berlin.")])

    stored = store.get_claim(claim_id)
    assert len(stored.sources) == 101, "a source went missing"
    assert len(set(stored.sources)) == 101, "or was counted twice"
    assert {r[0] for r in store._db.execute(
        "SELECT episode_id FROM claim_sources WHERE claim_id=?", (claim_id,))} == set(
        stored.sources), "the index and the array disagree about what supports this"
    assert all(store.claims_citing("acme", s) for s in stored.sources), \
        "every turn resolves backwards too"
    store.close()


def test_a_store_without_the_reverse_index_still_reinforces():
    """`claims_citing` is new to the protocol, and a `Store` someone else wrote must not
    start raising on the write path because of it. The fallback scan is slower and that
    is the whole of the difference."""
    inner = SQLiteStore(":memory:")
    pipe = WritePipeline(StoreWithoutReverseIndex(inner), HashingEmbedder(),
                         PredicateRegistry(), CountingLLM())

    first = pipe.add([ep("I live in Berlin.")])
    receipt = pipe.add([ep("I live in Berlin.")])

    assert [c.id for c in receipt.reinforced] == [first.added[0].id]
    assert inner.get_claim(first.added[0].id).observation_count == 2
    inner.close()


def test_a_repeat_does_not_reinforce_a_claim_that_has_been_retired():
    """`claims_citing` answers a provenance question and so includes retired claims,
    which the scan it replaced did not. Reinforcement raises a claim's storage strength
    so retrieval ranks it higher, and a claim nothing believes any more has no ranking to
    raise — so the filter belongs here, on the caller, not in the store."""
    pipe, store, _ = build()
    first = pipe.add([ep("I live in Berlin.")])
    claim = first.added[0]
    store.invalidate(claim.id, utcnow(), "cl_superseded")

    receipt = pipe.add([ep("I live in Berlin.")])

    assert receipt.reinforced == []
    assert receipt.skipped == 1, "the turn is still recognised as a repeat"
    assert store.get_claim(claim.id).observation_count == 1
    store.close()


# --- tier 2: schema acquisition is paid for once -----------------------------

def test_novel_predicate_is_classified_exactly_once():
    def responder(episodes):
        return [{"subject": "user", "predicate": "collects", "object": obj,
                 "polarity": 1, "memory_type": "semantic", "confidence": 0.9,
                 "source_index": i}
                for i, obj in enumerate(["vinyl records", "vintage stamps"])][:len(episodes)]

    llm = CountingLLM(responder=responder)
    pipe, store, registry = build(llm)

    receipt = pipe.add([
        ep("Been hunting down rare pressings at the market again."),
        ep("Picked up a whole album of pre-war postage at the fair."),
    ])

    # Two claims, one novel predicate, one classification. The Nth occurrence is free
    # because the 1st paid for the schema.
    assert llm.classify_calls == 1
    assert llm.classified == ["collects"]
    assert receipt.llm_calls == 2          # one extract + one classify
    assert registry.known("collects")
    store.close()


def test_a_learned_predicate_is_never_classified_again():
    def responder(episodes):
        return [{"subject": "user", "predicate": "collects", "object": "vinyl",
                 "polarity": 1, "memory_type": "semantic", "confidence": 0.9,
                 "source_index": 0}]

    llm = CountingLLM(responder=responder)
    pipe, store, _ = build(llm)

    for text in ["Been hunting rare pressings again.",
                 "Found another crate at the flea market.",
                 "Picked up two more at the fair today."]:
        pipe.add([ep(text)])

    assert llm.extract_calls == 3          # extraction still runs per batch
    assert llm.classify_calls == 1         # schema acquisition does not
    store.close()


def test_unknown_predicate_defaults_to_many_and_retires_nothing():
    objs = iter(["vinyl", "stamps"])

    def responder(episodes):
        return [{"subject": "user", "predicate": "collects", "object": next(objs),
                 "polarity": 1, "memory_type": "semantic", "confidence": 0.9,
                 "source_index": 0}]

    llm = CountingLLM(responder=responder)   # classifies as "many"
    pipe, store, _ = build(llm)
    pipe.add([ep("Been hunting rare vinyl pressings again.")])
    receipt = pipe.add([ep("Found a crate of pre-war postage stamps today.")])

    assert receipt.invalidated == []
    assert live(store) == [("collects", "stamps"), ("collects", "vinyl")]
    store.close()


def test_a_predicate_classified_as_one_does_supersede():
    objs = iter(["ergonomic split", "low profile"])

    def responder(episodes):
        return [{"subject": "user", "predicate": "keyboard_layout", "object": next(objs),
                 "polarity": 1, "memory_type": "semantic", "confidence": 0.9,
                 "source_index": 0}]

    llm = CountingLLM(responder=responder,
                      classification={"cardinality": "one", "volatility": "slow",
                                      "memory_type": "semantic"})
    pipe, store, registry = build(llm)
    pipe.add([ep("Switched over to a split board this week.")])
    receipt = pipe.add([ep("Went with the low profile switches in the end.")])

    # Retiring is only allowed because the LLM classified it ONE and it was registered.
    assert registry.spec("keyboard_layout").cardinality is Cardinality.ONE
    assert len(receipt.invalidated) == 1
    assert live(store) == [("keyboard_layout", "low profile")]
    store.close()


def test_fast_path_turns_never_reach_the_llm():
    llm = CountingLLM()
    pipe, store, _ = build(llm)
    pipe.add([ep("I live in Berlin."), ep("The quarterly review is next Tuesday.")])
    # Only the turn the fast path could not handle is batched.
    assert llm.batch_sizes == [1]
    store.close()


def test_known_predicates_are_offered_to_the_extractor():
    llm = CountingLLM()
    pipe, store, _ = build(llm)
    pipe.add([ep("The quarterly review is next Tuesday.")])
    known = llm.seen_known_predicates[0]
    # Reusing an existing predicate is how contradictions stay detectable, so the model
    # has to be told what already exists.
    assert "lives_in" in known and "works_at" in known
    store.close()


def test_a_novel_surface_form_costs_one_resolution_call_and_then_folds():
    """`resolve_predicate` replaced `classify_predicate` as the acquisition call: the
    same one-off spend, but bought a merge instead of a fourth slot for one question."""
    objs = iter(["Acme", "Globex"])

    def responder(episodes):
        return [{"subject": "user", "predicate": "paycheck_source", "object": next(objs),
                 "polarity": 1, "memory_type": "semantic", "confidence": 0.9,
                 "source_index": 0}]

    llm = ResolvingLLM(responder=responder, canonical="works_at")
    pipe, store, registry = build(llm)
    pipe.add([ep("Payroll switched over to Acme this quarter.")])
    receipt = pipe.add([ep("Payroll moved to Globex after the acquisition closed.")])

    assert llm.resolve_calls == 1          # once for the surface form, ever
    assert llm.classify_calls == 0
    assert receipt.llm_calls == 1          # the second write pays for extraction only
    assert registry.normalize("paycheck_source") == "works_at"
    # And because it is the same slot now, the second employer retires the first.
    assert live(store) == [("works_at", "Globex")]
    store.close()


def test_a_deterministically_foldable_form_never_reaches_the_model():
    llm = ResolvingLLM(claims=[
        {"subject": "user", "predicate": "employer_name", "object": "Acme", "polarity": 1,
         "memory_type": "semantic", "confidence": 0.9, "source_index": 0},
    ], canonical="works_at")
    pipe, store, _ = build(llm)
    receipt = pipe.add([ep("The payroll record now names Acme after the review.")])

    assert llm.resolve_calls == 0, "morphology must run before anything is billed"
    assert receipt.llm_calls == 1
    assert live(store) == [("works_at", "Acme")]
    store.close()


def test_a_no_op_backend_is_not_billed_and_reports_the_loss():
    from memvara.llm import NullLLM

    pipe, store, _ = build(NullLLM())
    receipt = pipe.add([ep("The quarterly review is next Tuesday."),
                        ep("The offsite moved to the Lisbon office.")])
    assert receipt.llm_calls == 0, "a call that never left the process is not a cost"
    assert receipt.unextracted == 2
    store.close()


def test_unextracted_counts_only_the_turns_that_yielded_nothing():
    llm = CountingLLM(claims=[
        {"subject": "user", "predicate": "likes", "object": "quarterly reviews",
         "polarity": 1, "memory_type": "semantic", "confidence": 0.9, "source_index": 0},
    ])
    pipe, store, _ = build(llm)
    receipt = pipe.add([ep("The quarterly review is next Tuesday."),
                        ep("The offsite moved to the Lisbon office.")])
    assert (receipt.llm_calls, receipt.unextracted) == (1, 1)
    store.close()


def test_a_failed_extraction_reports_every_turn_as_unextracted():
    class Failing(CountingLLM):
        def extract(self, episodes, known_predicates):
            raise RuntimeError("429 rate limited")

    pipe, store, _ = build(Failing())
    receipt = pipe.add([ep("The quarterly review is next Tuesday."),
                        ep("The offsite moved to the Lisbon office.")])
    assert receipt.deferred and receipt.unextracted == 2
    store.close()


def test_a_deferred_batch_can_be_read_again_once_the_provider_recovers():
    """The failure `receipt.deferred` names, and what could not be done about it before.

    A 429 keeps the episodes and loses the extraction. Nothing retried them, so the facts
    in that batch were gone while the text sat on disk. This is the retry.
    """
    class Flaky(CountingLLM):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.fail = True

        def extract(self, episodes, known_predicates):
            if self.fail:
                raise RuntimeError("429 rate limited")
            return super().extract(episodes, known_predicates)

    llm = Flaky(claims=[
        {"subject": "user", "predicate": "works_at", "object": "Globex", "polarity": 1,
         "memory_type": "semantic", "confidence": 0.9, "source_index": 0},
    ])
    store = SQLiteStore(":memory:")
    mem = Memvara(store=store, embedder=HashingEmbedder(), llm=llm, user="alice")
    first = mem.add("Payroll moved over to Globex after the acquisition closed.")
    assert first.deferred and first.added == []

    llm.fail = False
    receipt = mem.reextract()
    assert [(c.predicate, c.object) for c in receipt.added] == [("works_at", "Globex")]
    assert mem.pending_extraction() == [], "the turn is no longer pending"
    mem.close()


def test_the_work_list_drops_what_the_gate_would_drop():
    """`add()` stores episodes before gating them, so chitchat has no claims either.

    Without this the work list would hold every "thanks" in the store forever, and a
    scheduled pass would pay a model call to rediscover that on every sweep. The gate is
    deterministic and free, so it runs in the query.
    """
    store = SQLiteStore(":memory:")
    mem = Memvara(store=store, embedder=HashingEmbedder(), llm=NullLLM(), user="alice")
    mem.add("The deployment failed because of a race condition in the scheduler.")
    mem.add("thanks")
    contents = [e.content for e in mem.pending_extraction()]
    assert contents == ["The deployment failed because of a race condition in the scheduler."]
    mem.close()


def test_a_turn_the_model_already_declined_is_excluded_by_the_caller():
    """The population nothing on the episode can record, and the one `exclude` is for.

    A turn extraction read and produced nothing from — or produced only claims
    `reject_ungrounded` refused — ends with no claims citing it, which from the store is
    indistinguishable from never having been read. Measured against a local 4B: such a
    turn came back pending on the next sweep and cost another ~250s of CPU to be rejected
    again. `reextract` reports every turn it read, and the scheduler hands them back.
    """
    llm = CountingLLM(claims=[])          # reads the turn, finds nothing
    store = SQLiteStore(":memory:")
    mem = Memvara(store=store, embedder=HashingEmbedder(), llm=llm, user="alice")
    mem.add("The deployment failed because of a race condition in the scheduler.")

    first = mem.reextract()
    assert first.llm_calls == 1 and first.added == []
    assert len(first.episode_ids) == 1, "it must report what it read, or nothing can"

    # Without the feedback it is pending again — this is the loop, asserted rather than
    # described, so a regression that drops `exclude` fails here.
    assert len(mem.pending_extraction()) == 1
    assert mem.pending_extraction(exclude=first.episode_ids) == []

    second = mem.reextract(exclude=first.episode_ids)
    assert second.llm_calls == 0, "a converged sweep spends nothing"
    mem.close()


def test_re_reading_a_turn_that_already_has_claims_is_refused_not_reinforced():
    """Re-reading stored text is not new evidence, but the reconciler cannot tell.

    An identical claim arriving twice reconciles to `reinforce` and bumps salience, so a
    sweep that ran over the same episode twice would quietly promote what it had already
    extracted — a ranking change nobody asked for and nothing reports. Skipping is the
    idempotency guarantee, and the count is how a scheduler notices it is re-selecting.
    """
    llm = CountingLLM(claims=[
        {"subject": "user", "predicate": "works_at", "object": "Globex", "polarity": 1,
         "memory_type": "semantic", "confidence": 0.9, "source_index": 0},
    ])
    store = SQLiteStore(":memory:")
    mem = Memvara(store=store, embedder=HashingEmbedder(), llm=llm, user="alice")
    mem.add("Payroll moved over to Globex after the acquisition closed.")
    [claim] = mem.get_all()
    before, calls = claim.salience_base, llm.extract_calls

    # Explicit ids, which is the only way to reach an already-extracted turn: the work
    # list excludes it by construction.
    receipt = mem.reextract([claim.sources[0]])
    assert receipt.already_extracted == 1
    assert receipt.added == [] and llm.extract_calls == calls, "no model call, no write"
    assert mem.get_all()[0].salience_base == before, "and no silent promotion"
    mem.close()


def test_explicit_ids_still_go_through_the_gate():
    """The work list filters chitchat, so only a caller naming ids can smuggle it in.

    That is the path a retry takes — `receipt.episode_ids` from a failed batch, handed
    straight back — and a batch that failed at tier 2 contains whatever was in it,
    including turns the gate had already dropped. Without the gate here those reach the
    model, which on a CPU-bound local extractor is minutes of work to rediscover that
    "thanks" carries no fact.
    """
    llm = CountingLLM(claims=[])
    store = SQLiteStore(":memory:")
    mem = Memvara(store=store, embedder=HashingEmbedder(), llm=llm, user="alice")
    mem.add("thanks")
    [chitchat] = list(store.iter_episodes("default"))

    receipt = mem.reextract([chitchat.id])
    assert llm.extract_calls == 0, "the gate must run before the model, not after"
    assert receipt.skipped == 1
    mem.close()


def test_reextract_reports_the_same_series_a_write_does():
    """Except `write.memory_episodes`, which this path must not emit at all.

    A zero there would be indistinguishable from a write that stored no episode and was
    charged for it, which is the confusion that series exists to end. The rest are the
    ordinary write series: a scheduled pass has to show up in the same dashboards.
    """
    rec = MemoryRecorder()
    store = SQLiteStore(":memory:")
    # Stored with no extractor, which is the state this whole feature is for.
    mem = Memvara(store=store, embedder=HashingEmbedder(), llm=NullLLM(), user="alice",
                  telemetry=rec)
    mem.add("Payroll moved over to Globex after the acquisition closed.")

    mem.writer.llm = CountingLLM(claims=[
        {"subject": "user", "predicate": "works_at", "object": "Globex", "polarity": 1,
         "memory_type": "semantic", "confidence": 0.9, "source_index": 0},
    ])
    before_episodes = rec.total(WRITE_MEMORY_EPISODES)
    receipt = mem.reextract()

    assert len(receipt.added) == 1
    assert rec.total(WRITE_MEMORY_EPISODES) == before_episodes, (
        "reextract stores no episode, so it must not move that counter")
    assert rec.total(WRITE_MEMORY_CLAIMS) >= 1 and receipt.latency_ms >= 0.0
    mem.close()


def test_reextract_runs_the_redactor_over_what_it_extracts():
    """"Every claim in the store passed the redactor exactly once" has to hold on every
    write path, or it is an argument about transitivity rather than a property."""
    from memvara.redact import PatternRedactor

    store = SQLiteStore(":memory:")
    mem = Memvara(store=store, embedder=HashingEmbedder(), llm=NullLLM(), user="alice",
                  redactor=PatternRedactor(), write_reject_ungrounded=False)
    mem.add("There is a number written down somewhere in my profile notes.")

    mem.writer.llm = CountingLLM(claims=[
        {"subject": "user", "predicate": "contact_at", "object": "call 555-123-4567",
         "polarity": 1, "memory_type": "semantic", "confidence": 0.9,
         "source_index": 0},
    ])
    receipt = mem.reextract()
    assert receipt.added, "the claim must land, redacted rather than dropped"
    assert "555-123-4567" not in receipt.added[0].object
    mem.close()


def test_the_scoped_and_async_views_reach_both_new_methods():
    """`ScopedMemvara` and the async facades are the objects a server actually holds."""
    import asyncio

    from memvara.aio import AsyncMemvara

    llm = CountingLLM(claims=[
        {"subject": "user", "predicate": "works_at", "object": "Globex", "polarity": 1,
         "memory_type": "semantic", "confidence": 0.9, "source_index": 0},
    ])
    store = SQLiteStore(":memory:")
    mem = Memvara(store=store, embedder=HashingEmbedder(), llm=CountingLLM(claims=[]),
                  user="alice")
    mem.add("Payroll moved over to Globex after the acquisition closed.")

    scoped = mem.scope(user="alice")
    assert len(scoped.pending_extraction()) == 1
    mem.writer.llm = llm
    assert scoped.reextract(limit=5).added, "the scoped view must reach the write"

    async def via_async():
        amem = AsyncMemvara(mem)
        pending = await amem.pending_extraction()
        receipt = await amem.reextract(exclude=[e.id for e in pending])
        ascoped = amem.scope(user="alice")
        await ascoped.pending_extraction()
        await ascoped.reextract(exclude=[e.id for e in pending])
        return receipt

    assert asyncio.run(via_async()).llm_calls == 0, "everything was already excluded"
    mem.close()


def test_limit_stops_the_scan_rather_than_trimming_afterwards():
    """The bound a scheduled pass runs behind, and the reason it is a bound at all.

    On CPU-bound extraction a turn costs minutes, so a sweep has to take a fixed bite and
    stop. Short-circuiting the scan is also what keeps the per-episode `claims_citing`
    affordable: it is the N+1 named in `pending_extraction`, and `limit` is what stops it
    walking the whole store to build a list the caller will discard.
    """
    store = SQLiteStore(":memory:")
    mem = Memvara(store=store, embedder=HashingEmbedder(), llm=NullLLM(), user="alice")
    for n in range(5):
        mem.add(f"The deployment failed because of race condition number {n}.")

    assert len(mem.pending_extraction()) == 5
    assert len(mem.pending_extraction(limit=2)) == 2
    mem.close()


def test_reextract_never_stores_an_episode():
    """Tier 0 is skipped: these turns are already stored and already embedded.

    Running it would find each episode as an exact repeat of itself, which is why the
    `write.memory_episodes` counter this path deliberately does not emit would be a lie.
    """
    source = (ROOT / "memvara" / "write" / "pipeline.py").read_text(encoding="utf-8")
    body = source[source.index("def reextract"):source.index("def _transaction")]
    assert "add_episode" not in body
    assert "_tier0" not in body
    assert "WRITE_MEMORY_EPISODES" not in body


def test_the_learned_cap_folds_instead_of_growing_the_schema():
    """The backstop: past the cap a novel form attaches to its nearest neighbour rather
    than claiming a slot. Unbounded schema growth is what breaks contradiction
    detection, so bounding it matters more than getting every fold right."""
    predicates = iter(["previous_employer", "prior_employer", "old_employer"])

    def responder(episodes):
        return [{"subject": "user", "predicate": next(predicates), "object": "Acme",
                 "polarity": 1, "memory_type": "semantic", "confidence": 0.9,
                 "source_index": 0}]

    llm = ResolvingLLM(responder=responder, canonical=None)
    store = SQLiteStore(":memory:")
    registry = PredicateRegistry(max_learned=1)
    pipe = WritePipeline(store, HashingEmbedder(), registry, llm)
    for text in ["Payroll switched over to the new provider this quarter.",
                 "Payroll moved again after the acquisition closed.",
                 "Payroll changed hands a third time in the autumn."]:
        pipe.add([ep(text)])

    assert len([s for s in registry.all_specs() if s.learned]) == 1
    # And the ones past the cap folded rather than being asked about at all.
    assert llm.resolve_calls == 1
    assert registry.normalize("prior_employer") == "works_at"
    store.close()


# --- trust boundary on model output ------------------------------------------

@pytest.mark.parametrize(
    "bad",
    [
        {"subject": "user", "predicate": "likes", "object": "tea", "source_index": 99},
        {"subject": "user", "predicate": "likes", "object": "tea", "source_index": -1},
        {"subject": "user", "predicate": "likes", "object": "tea"},
        {"subject": "user", "predicate": "likes", "object": "tea", "source_index": "0"},
        {"subject": "user", "predicate": "", "object": "tea", "source_index": 0},
        {"subject": "user", "predicate": "likes", "object": "", "source_index": 0},
    ],
)
def test_malformed_model_output_is_dropped_not_repaired(bad):
    llm = CountingLLM(claims=[bad])
    pipe, store, _ = build(llm)
    receipt = pipe.add([ep("The quarterly review is next Tuesday.")])
    # A claim with no usable provenance is exactly what this library exists not to store.
    assert receipt.added == []
    store.close()


# -- reject_ungrounded ---------------------------------------------------------
#
# Off by default -- see WritePipeline.__init__ -- so every test above this section
# keeps working with fabricated-but-structurally-valid objects like "tea" against
# unrelated filler text. These turn the option on explicitly.

def test_a_claim_sharing_no_vocabulary_with_its_source_is_dropped():
    """The exact shape the option exists for: a well-formed claim, invented whole.

    "tea" shares nothing with a sentence about a quarterly review. Structurally this
    claim is fine -- real predicate, real object, valid source_index -- which is why
    `test_malformed_model_output_is_dropped_not_repaired` above does not catch it: that
    test is about form, not truth.
    """
    llm = CountingLLM(claims=[
        {"subject": "user", "predicate": "likes", "object": "tea", "polarity": 1,
         "memory_type": "semantic", "confidence": 0.9, "source_index": 0},
    ])
    pipe, store, _ = build(llm, reject_ungrounded=True)
    receipt = pipe.add([ep("The quarterly review is next Tuesday.")])
    assert receipt.added == []
    assert receipt.ungrounded == 1
    # The rejected claim was the only thing proposed for this turn, so it also reads as
    # unextracted -- `out` never received it either way. One rejection, two honest
    # numbers describing the same event from two angles, not a double-count.
    assert receipt.unextracted == 1
    store.close()


def test_a_grounded_claim_survives_the_same_option():
    """The option is a filter, not a tax on every write that turns it on."""
    llm = CountingLLM(claims=[
        {"subject": "user", "predicate": "works_at", "object": "Acme Robotics",
         "polarity": 1, "memory_type": "semantic", "confidence": 0.9, "source_index": 0},
    ])
    pipe, store, _ = build(llm, reject_ungrounded=True)
    receipt = pipe.add([ep("I just started at Acme Robotics as a machinist.")])
    assert len(receipt.added) == 1
    assert receipt.ungrounded == 0
    store.close()


def test_a_compound_identifier_is_matched_as_a_substring_not_an_exact_token():
    """The check that motivated substring matching over exact-token matching.

    Real content here is full of paths and hyphenated identifiers. An object of
    "expense-tracker" must be found inside a source token like
    "expense-tracker-bb03f971", or every claim naming a project by its short form would
    be rejected for citing the same project by its long one.
    """
    llm = CountingLLM(claims=[
        {"subject": "user", "predicate": "working_on", "object": "expense-tracker",
         "polarity": 1, "memory_type": "semantic", "confidence": 0.9, "source_index": 0},
    ])
    pipe, store, _ = build(llm, reject_ungrounded=True)
    receipt = pipe.add([ep(
        "[MIGRATION_CHUNK source=/Users/x/.grok/memory/expense-tracker-bb03f971/MEMORY.md]"
    )])
    assert len(receipt.added) == 1
    assert receipt.ungrounded == 0
    store.close()


def test_an_object_with_nothing_to_check_is_not_rejected_on_that_alone():
    """An object that tokenizes to zero content words -- all stopwords, or too short --
    means no signal either way, and no signal must not read as a bad signal.

    Rejecting on an empty check would be worse than not checking: it turns "this object
    could not be parsed into words" into "this object is fabricated", which it is not
    evidence of. `"it"` is deliberately chosen: `_content_words("it")` is `[]` (a
    stopword), so this exercises the early return in `_wholly_ungrounded` directly
    rather than relying on coincidental word overlap.
    """
    llm = CountingLLM(claims=[
        {"subject": "user", "predicate": "mood", "object": "it", "polarity": 1,
         "memory_type": "semantic", "confidence": 0.9, "source_index": 0},
    ])
    pipe, store, _ = build(llm, reject_ungrounded=True)
    receipt = pipe.add([ep("Completely unrelated conversation about something else.")])
    assert len(receipt.added) == 1
    assert receipt.ungrounded == 0
    store.close()


class KeyedEmbedder:
    """Deterministic stand-in for a semantic embedder, keyed on marker substrings.

    A text containing a marker embeds as that marker's unit vector; anything else gets
    a vector orthogonal to all of them. That is the whole of what the rescue relies on
    a real semantic embedder for -- paraphrases land near their sources, inventions do
    not -- reduced to something offline and exact. `HashingEmbedder` cannot play this
    role in a test *because* the rescue correctly no-ops on it: character n-grams score
    ~0.0-0.11 on zero-overlap pairs, well under the floor.

    Markers are only seen inside the first `HORIZON` characters of each text, because
    the real property being modelled includes the failure: MiniLM-class embedders
    truncate around 512 tokens, and content past the truncation point contributes
    nothing to the vector. Without this, the chunking test passes against a rescue
    that embeds the whole episode in one piece -- which is exactly the defect chunking
    exists to prevent, invisible to a fake that reads to the end.
    """

    dim = 4
    HORIZON = 1200

    def __init__(self, markers: dict[str, int], fail_on_first: str | None = None) -> None:
        self._markers = markers
        self._fail_on_first = fail_on_first

    def encode(self, texts):
        import numpy as np

        if self._fail_on_first is not None and texts and texts[0] == self._fail_on_first:
            raise RuntimeError("embedder fell over mid-rescue")
        out = np.zeros((len(texts), self.dim))
        for row, text in enumerate(texts):
            axis = 3
            for marker, index in self._markers.items():
                # Case-insensitive, as the real thing is by nature: "Short answers"
                # and "short answers" are the same meaning to a semantic embedder.
                if marker in text[:self.HORIZON].lower():
                    axis = index
                    break
            out[row, axis] = 1.0
        return out


def test_auto_rescues_a_paraphrase_the_lexical_check_would_reject():
    """The reason "auto" exists, and the reason it can default on.

    "keep replies brief" shares not one content word with "I prefer short answers" --
    the strict lexical check rejects it, and it is a true fact. Under "auto" the
    embedder gets a veto first: a genuine paraphrase embeds near its source (measured
    0.45-0.58 on MiniLM for pairs like this, against <=0.33 for every wholesale
    invention), so it is kept. The fake maps both phrasings onto one axis, which is the
    property being borrowed from the real embedder, made exact.
    """
    llm = CountingLLM(claims=[
        {"subject": "user", "predicate": "prefers", "object": "keep replies brief",
         "polarity": 1, "memory_type": "procedural", "confidence": 0.9,
         "source_index": 0},
    ])
    store = SQLiteStore(":memory:")
    embedder = KeyedEmbedder({"brief": 0, "short answers": 0})
    pipe = WritePipeline(store, embedder, PredicateRegistry(), llm)
    # Not "I prefer short answers": that is a deterministic fast-path form, and a turn
    # the fast path handles never reaches the model or this rescue at all.
    receipt = pipe.add([ep("Short answers only please, and no padding.")])
    assert llm.extract_calls == 1, "the turn must actually reach tier 2"
    assert len(receipt.added) == 1
    assert receipt.ungrounded == 0
    store.close()


def test_auto_still_rejects_what_the_embedder_cannot_connect_either():
    """A fabrication fails both checks: no shared vocabulary, orthogonal embedding."""
    llm = CountingLLM(claims=[
        {"subject": "user", "predicate": "works_at", "object": "Acme", "polarity": 1,
         "memory_type": "semantic", "confidence": 1.0, "source_index": 0},
    ])
    store = SQLiteStore(":memory:")
    # "billing" pins the source to its own axis; the object matches no marker and
    # lands on the default axis, orthogonal to it -- which is what a wholesale
    # invention looks like through a real semantic embedder.
    pipe = WritePipeline(store, KeyedEmbedder({"billing": 1}), PredicateRegistry(), llm)
    receipt = pipe.add([ep("We migrated the billing job to run nightly.")])
    assert llm.extract_calls == 1
    assert receipt.added == []
    assert receipt.ungrounded == 1
    assert "ungrounded=1" in str(receipt)
    store.close()


def test_strict_mode_takes_no_second_opinion():
    """`True` is the lexical check alone, even with a semantic embedder configured.

    The mode exists for a caller who has measured their extractor and wants the
    hard line; a rescue that ran anyway would make True and "auto" the same mode.
    """
    llm = CountingLLM(claims=[
        {"subject": "user", "predicate": "prefers", "object": "keep replies brief",
         "polarity": 1, "memory_type": "procedural", "confidence": 0.9,
         "source_index": 0},
    ])
    store = SQLiteStore(":memory:")
    embedder = KeyedEmbedder({"brief": 0, "short answers": 0})
    pipe = WritePipeline(store, embedder, PredicateRegistry(), llm,
                         reject_ungrounded=True)
    receipt = pipe.add([ep("Short answers only please, and no padding.")])
    assert llm.extract_calls == 1
    assert receipt.added == []
    assert receipt.ungrounded == 1
    store.close()


def test_the_rescue_reads_a_long_turn_in_chunks():
    """A fact grounded late in a 7,000-character turn must not be invisible.

    MiniLM-class embedders truncate around 512 tokens, so a single whole-episode
    embedding is blind to everything past the truncation point. The rescue compares
    against chunks and keeps the best score; the marker here sits past the first chunk
    boundary, so a whole-episode comparison through this fake would have missed it.
    """
    llm = CountingLLM(claims=[
        {"subject": "user", "predicate": "likes", "object": "felines", "polarity": 1,
         "memory_type": "semantic", "confidence": 0.9, "source_index": 0},
    ])
    store = SQLiteStore(":memory:")
    # The fake keys the *first* marker found; filler text carries none, so only the
    # chunk containing "cats" lands on the object's axis.
    embedder = KeyedEmbedder({"felines": 0, "cats": 0})
    pipe = WritePipeline(store, embedder, PredicateRegistry(), llm)
    long_turn = ("The deploy pipeline rebuilt every image twice today. " * 30
                 + "Completely unrelatedly: cats are wonderful companions.")
    assert len(long_turn) > 1200, "the marker must sit past the first chunk"
    receipt = pipe.add([ep(long_turn)])
    assert len(receipt.added) == 1
    assert receipt.ungrounded == 0
    store.close()


def test_a_broken_embedder_fails_the_rescue_open_and_says_so_once():
    """The rescue's failure must cost precision, never a claim.

    An embedder that raises mid-rescue leaves "auto" unable to obtain the second
    opinion it promises -- and the lexical trigger alone is not grounds for rejection
    under that mode, so the claim is kept. Warned once per pipeline, same rule as the
    embedding-rejection warnings beside it: silence here would leave the mode quietly
    running lexical-only with nothing saying so.
    """
    llm = CountingLLM(claims=[
        {"subject": "user", "predicate": "works_at", "object": "Initech", "polarity": 1,
         "memory_type": "semantic", "confidence": 0.9, "source_index": 0},
    ])
    store = SQLiteStore(":memory:")
    # Fails only when handed the rescue's batch (which leads with the object); the
    # tier-0 near-duplicate embedding of episode content is left working.
    embedder = KeyedEmbedder({}, fail_on_first="Initech")
    pipe = WritePipeline(store, embedder, PredicateRegistry(), llm)
    with pytest.warns(RuntimeWarning, match="grounding rescue"):
        receipt = pipe.add([ep("The billing job moved to a nightly schedule.")])
    assert len(receipt.added) == 1
    assert receipt.ungrounded == 0
    # Once per pipeline: a second trip through the same failure stays quiet.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pipe.add([ep("The nightly schedule was confirmed by the operator.")])
    store.close()


def test_a_fabricated_claim_cannot_retire_a_true_fact_by_default():
    """The stake, stated as behaviour: why this defaults on rather than off.

    `works_at` is a builtin ONE-cardinality predicate, so a fabricated employer does
    not sit harmlessly beside the real one -- it supersedes it, and the true fact is
    *ended*. Storing fabrication is the destructive direction; rejection only ever
    costs a claim from one turn. Both halves asserted: the default blocks it, and
    turning the option off demonstrates exactly the damage it would have done.
    """
    def scenario(**kw):
        llm = CountingLLM(claims=[
            {"subject": "user", "predicate": "works_at", "object": "Acme",
             "polarity": 1, "memory_type": "semantic", "confidence": 1.0,
             "source_index": 0},
        ])
        pipe, store, _ = build(llm, **kw)
        pipe.add([ep("I work at Globex.")])  # fast path, grounded, true
        pipe.add([ep("We migrated the billing job to run nightly.")])
        outcome = live(store)
        store.close()
        return outcome

    assert scenario() == [("works_at", "Globex")], "the default keeps the true employer"
    assert scenario(reject_ungrounded=False) == [("works_at", "Acme")], (
        "with the filter off, the fabrication supersedes and ends the true fact -- "
        "this line is the measurement of what the default prevents")


def test_a_cross_script_claim_is_not_rejected_by_a_check_that_cannot_read_it():
    """`lives_in: Beijing` from 我住在北京 is grounded; the tokenizer just cannot see it.

    The lexical check is built from a Latin-script word regex, so a source it cannot
    tokenize at all would trip on every claim extracted from it -- every fact a CJK,
    Cyrillic or Arabic turn yields, rejected not because it is ungrounded but because
    the check is out of its depth. It abstains instead, the same rule the empty-object
    early return applies, mirrored onto the source. Found by an existing telemetry
    test, not foresight: the keyword-compatibility test extracts exactly this claim
    from exactly this sentence, and the default flip broke it.
    """
    llm = CountingLLM(claims=[
        {"subject": "user", "predicate": "lives_in", "object": "Beijing", "polarity": 1,
         "memory_type": "semantic", "confidence": 0.9, "source_index": 0},
    ])
    pipe, store, _ = build(llm)
    receipt = pipe.add([ep("我住在北京，工作也在这里。")])
    assert len(receipt.added) == 1
    assert receipt.ungrounded == 0
    store.close()


def test_a_mode_nobody_defined_is_a_construction_error():
    with pytest.raises(TypeError, match="is not a mode"):
        build(reject_ungrounded="yes")


def test_the_default_mode_is_auto():
    pipe, store, _ = build()
    assert pipe.reject_ungrounded == "auto"
    store.close()


def test_the_option_is_reachable_through_memvara_s_tuning_prefix():
    """`Memvara(write_reject_ungrounded=True, ...)` -- no plumbing needed beyond the
    constructor parameter, because `_split_tuning` validates against the real signature.
    """
    from memvara import Memvara
    from memvara.llm.base import NullLLM

    mem = Memvara(":memory:", user="alice", llm=NullLLM(),
                  embedder=HashingEmbedder(), write_reject_ungrounded=True)
    assert mem.writer.reject_ungrounded is True
    mem.close()


def test_confidence_is_clamped():
    llm = CountingLLM(claims=[
        {"subject": "user", "predicate": "likes", "object": "quarterly reviews",
         "polarity": 1, "memory_type": "semantic", "confidence": 7.5, "source_index": 0},
    ])
    pipe, store, _ = build(llm)
    receipt = pipe.add([ep("The quarterly review is next Tuesday.")])
    assert receipt.added[0].confidence == 1.0
    store.close()


def test_source_index_maps_back_to_the_right_episode():
    llm = CountingLLM(claims=[
        {"subject": "user", "predicate": "likes", "object": "the Lisbon office",
         "polarity": 1, "memory_type": "semantic", "confidence": 0.9, "source_index": 1},
    ])
    pipe, store, _ = build(llm)
    a = ep("The quarterly review is next Tuesday.")
    b = ep("The offsite got moved to the Lisbon office.")
    receipt = pipe.add([a, b])
    assert receipt.added[0].sources == [b.id]
    store.close()


# --- edge cases --------------------------------------------------------------

def test_empty_input():
    llm = CountingLLM()
    pipe, store, _ = build(llm)
    receipt = pipe.add([])
    assert (receipt.llm_calls, receipt.added, receipt.skipped) == (0, [], 0)
    assert llm.total_calls == 0
    store.close()


def test_empty_and_whitespace_turns():
    llm = CountingLLM()
    pipe, store, _ = build(llm)
    receipt = pipe.add([ep(""), ep("   "), ep("\n\t  \n")])
    assert receipt.llm_calls == 0
    assert llm.total_calls == 0
    assert receipt.added == []
    assert receipt.skipped == 3
    store.close()


def test_unicode_content_round_trips():
    pipe, store, _ = build()
    receipt = pipe.add([ep("I live in München."), ep("I like 日本茶.")])
    assert receipt.llm_calls == 0
    assert live(store) == [("likes", "日本茶"), ("lives_in", "München")]
    store.close()


def test_unicode_turn_that_needs_the_llm_still_batches():
    llm = CountingLLM()
    pipe, store, _ = build(llm)
    receipt = pipe.add([ep("私は東京に住んでいます。"), ep("Ich arbeite bei Siemens.")])
    assert llm.batch_sizes == [2]
    assert receipt.llm_calls == 1
    store.close()


def test_50kb_turn_is_handled():
    llm = CountingLLM()
    pipe, store, _ = build(llm)
    big = ("I reviewed the deployment logs and everything looked fine. " * 850
           + " I live in Berlin.")
    assert len(big) > 50_000
    receipt = pipe.add([ep(big)])
    # The fast path found the fact, so the 50KB never had to be sent anywhere.
    assert receipt.llm_calls == 0
    assert live(store) == [("lives_in", "Berlin")]
    store.close()


def test_a_turn_can_carry_several_facts():
    pipe, store, _ = build()
    receipt = pipe.add([ep("My name is Goldy and I live in Berlin, I work at Acme.")])
    assert receipt.llm_calls == 0
    assert len(receipt.added) == 3
    store.close()


def test_two_users_in_one_tenant_stay_separate():
    pipe, store, _ = build()
    alice = Episode(content="I live in Berlin.", scope=Scope("acme", "alice"))
    bob = Episode(content="I live in Lisbon.", scope=Scope("acme", "bob"))
    receipt = pipe.add([alice, bob])
    assert receipt.invalidated == []
    assert live(store) == [("lives_in", "Berlin"), ("lives_in", "Lisbon")]
    store.close()


def test_conflicting_facts_inside_one_batch_resolve_deterministically():
    pipe, store, _ = build()
    receipt = pipe.add([ep("I live in Berlin."), ep("I moved to Lisbon.")])
    # Last writer in input order wins, and the loser is invalidated rather than dropped.
    assert live(store) == [("lives_in", "Lisbon")]
    assert len(receipt.invalidated) == 1
    store.close()


# --- embeddings and provenance ----------------------------------------------

def test_every_stored_claim_gets_an_embedding():
    pipe, store, _ = build()
    pipe.add([ep("I live in Berlin."), ep("I like coffee.")])
    stats = store.stats()
    assert stats["embeddings"] >= stats["live_claims"] > 0
    store.close()


def test_claims_are_traceable_back_to_their_episode():
    pipe, store, _ = build()
    e = ep("I live in Berlin.")
    receipt = pipe.add([e])
    claim = receipt.added[0]
    assert claim.sources == [e.id]
    assert store.get_episode(e.id).content == "I live in Berlin."
    store.close()


# --- assert_claim ------------------------------------------------------------

def test_assert_claim_never_calls_the_model():
    llm = CountingLLM()
    pipe, store, _ = build(llm)
    receipt = pipe.assert_claim(
        Claim(subject="user", predicate="lives_in", object="Berlin", scope=SCOPE))
    assert receipt.llm_calls == 0
    assert llm.total_calls == 0
    assert len(receipt.added) == 1
    assert receipt.added[0].derivation is Derivation.USER
    store.close()


def test_assert_claim_supersedes_an_extracted_claim():
    pipe, store, _ = build()
    added = pipe.add([ep("I live in Berlin.")])
    receipt = pipe.assert_claim(
        Claim(subject="user", predicate="lives_in", object="Lisbon", scope=SCOPE))
    assert [c.id for c in receipt.invalidated] == [added.added[0].id]
    assert live(store) == [("lives_in", "Lisbon")]
    store.close()


def test_assert_claim_preserves_an_explicit_derivation():
    pipe, store, _ = build()
    claim = Claim(subject="user", predicate="lives_in", object="Berlin", scope=SCOPE,
                  derivation=Derivation.CONSOLIDATION, extractor="consolidator/v1")
    receipt = pipe.assert_claim(claim)
    assert receipt.added[0].derivation is Derivation.CONSOLIDATION
    store.close()


def test_assert_claim_normalizes_an_alias_predicate():
    pipe, store, _ = build()
    pipe.add([ep("I live in Berlin.")])
    receipt = pipe.assert_claim(
        Claim(subject="user", predicate="resides_in", object="Lisbon", scope=SCOPE))
    # Without normalization "resides_in" is a different slot and the contradiction is
    # invisible - the exact failure mode of a free-text memory store.
    assert receipt.added[0].predicate == "lives_in"
    assert len(receipt.invalidated) == 1
    store.close()


# --- determinism -------------------------------------------------------------

def test_the_whole_pipeline_is_deterministic():
    turns = ["My name is Goldy.", "ok thanks", "I live in Berlin.",
             "What's the weather?", "I like coffee.", "I moved to Lisbon.",
             "I like tea.", "I work at Acme.", "I no longer work at Acme."]

    def run():
        pipe, store, _ = build()
        totals = []
        for t in turns:
            r = pipe.add([Episode(content=t, scope=SCOPE)])
            totals.append((r.llm_calls, len(r.added), len(r.invalidated),
                           len(r.reinforced), r.skipped))
        result = (totals, live(store))
        store.close()
        return result

    first, second = run(), run()
    assert first == second
    assert sum(t[0] for t in first[0]) == 0        # not one model call in the whole run
    assert first[1] == [("likes", "coffee"), ("likes", "tea"), ("lives_in", "Lisbon"),
                        ("name", "Goldy")]


# =============================================================================
# Where the transaction starts
#
# `add()` used to run every tier inside one `store.batch()`, and `SQLiteStore.batch`
# holds a process-wide RLock for the block. Tier 2 calls `llm.extract()`, so one slow
# extraction stalled every read and every write for every tenant in the process for the
# length of a provider round trip. Measured with a 1.0 s fake extraction and a reader
# thread searching a 200-claim store throughout: 1 completed search in the whole window
# at p50 1,006 ms, against 516 completed searches at p50 1.91 ms and p95 2.23 ms after
# the hoist. The write takes 1.01 s either way — it stopped being everyone else's
# problem, it did not get faster.
# =============================================================================

class _BatchWatcher:
    """A Store proxy that reports whether a transaction is open right now."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.depth = 0
        self.opened = 0

    @contextmanager
    def batch(self):
        self.depth += 1
        self.opened += 1
        try:
            with self._inner.batch():
                yield self
        finally:
            self.depth -= 1

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_the_extraction_call_does_not_happen_inside_the_transaction():
    """The whole point of the hoist, asserted structurally rather than by a stopwatch.

    A model round trip inside the store's transaction is a write outage for every other
    tenant in the process, so "was a transaction open when the model was called" is the
    property, and it is a boolean rather than a latency."""
    watcher = _BatchWatcher(SQLiteStore(":memory:"))
    seen: list[int] = []

    class Observing(CountingLLM):
        def extract(self, episodes, known_predicates):
            seen.append(watcher.depth)
            return super().extract(episodes, known_predicates)

    pipe = WritePipeline(watcher, HashingEmbedder(), PredicateRegistry(), Observing())
    pipe.add([ep("The offsite moved to the Lisbon office.")])
    assert seen == [0], "llm.extract() ran with a transaction open"
    watcher.close()


def test_the_near_duplicate_encode_does_not_happen_inside_the_transaction():
    """Same argument, one tier earlier: `Embedder.encode` is a local hash for the
    shipped embedder and a network round trip for a hosted one, and which of those was
    configured must not decide how long the store's lock is held."""
    watcher = _BatchWatcher(SQLiteStore(":memory:"))
    seen: list[int] = []

    class Observing(HashingEmbedder):
        def encode(self, texts):
            seen.append(watcher.depth)
            return super().encode(texts)

    pipe = WritePipeline(watcher, Observing(), PredicateRegistry(), CountingLLM())
    pipe.add([ep("I live in Berlin.")])
    # First encode is the near-duplicate check and must be outside; the last is the
    # claim vectors, which are written inside the transaction by design.
    assert seen[0] == 0
    watcher.close()


def test_a_reader_is_not_blocked_by_a_slow_extraction():
    """The behaviour the structural test stands in for, with a real second thread.

    The margin is three orders of magnitude - a 0.4 s extraction against reads that take
    single-digit milliseconds - so the bound below is loose enough to survive a busy CI
    box and would still fail outright on the old single-transaction shape, where the
    reader completed one search in the entire window."""
    store = SQLiteStore(":memory:")
    embedder = HashingEmbedder()
    registry = PredicateRegistry()
    pipe = WritePipeline(store, embedder, registry, CountingLLM())
    pipe.add([ep("I live in Berlin.")])

    class Slow(CountingLLM):
        def extract(self, episodes, known_predicates):
            time.sleep(0.4)
            return super().extract(episodes, known_predicates)

    pipe.llm = Slow()
    reader = HybridRetriever(store, embedder, registry)
    latencies: list[float] = []
    stop = threading.Event()

    def read_loop():
        while not stop.is_set():
            t0 = perf_counter()
            reader.search("where do I live", Scope("acme", "alice"))
            latencies.append((perf_counter() - t0) * 1000.0)

    thread = threading.Thread(target=read_loop, daemon=True)
    thread.start()
    try:
        pipe.add([ep("The offsite moved to the Lisbon office.")])
    finally:
        stop.set()
        thread.join()

    assert len(latencies) > 10, (
        f"only {len(latencies)} reads completed during a 0.4 s extraction; the write "
        "path is holding the store lock across the model call")
    assert max(latencies) < 200.0
    store.close()


def test_episodes_are_durable_before_claims_are_written():
    """The trade the hoist accepts, stated as a test rather than as a comment.

    A crash between the episode commit and the claim write leaves an episode with no
    claims. That is the recoverable direction: episodes are the source of truth, and a
    retry converges on the same rows because the content hash matches."""
    pipe, store, _ = build()
    episode = ep("My name is Goldy.")

    def explode(*a, **kw):
        raise RuntimeError("crash between the two writes")

    pipe.reconciler.apply = explode
    with pytest.raises(RuntimeError):
        pipe.add([episode])
    assert store.get_episode(episode.id) is not None, "the raw turn was lost"
    assert list(store.iter_claims("acme")) == []
    store.close()


def test_a_retry_after_that_crash_converges_instead_of_duplicating():
    """The other half of the trade, stated exactly rather than optimistically.

    The content hash makes the retry converge on the *turn*: it is stored once, not
    twice, and the caller gets back the id it already has. It does not re-extract - an
    exact repeat is precisely the case tier 0 exists to charge nothing for - so the
    orphaned turn keeps no claims. That is survivable only because wave 2 made
    unextracted turns retrievable in their own right, which is asserted here so the
    dependency is visible if it ever goes away."""
    pipe, store, _ = build()
    text = "My name is Goldy."
    broken = ep(text)

    original = pipe.reconciler.apply
    pipe.reconciler.apply = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        pipe.add([broken])

    pipe.reconciler.apply = original
    receipt = pipe.add([ep(text)])
    assert receipt.episode_ids == [broken.id]
    assert receipt.skipped == 1
    assert store.stats()["episodes"] == 1
    assert live(store) == []
    # Still findable, as a turn rather than as a fact.
    hits = store.lexical_search("Goldy", [SCOPE], 5)
    assert [eid for eid, _ in
            store.lexical_search_episodes("Goldy", [SCOPE], 5)] == [broken.id]
    assert hits == []
    store.close()


def test_a_line_repeated_inside_one_batch_is_still_stored_once():
    """Hash-identical turns in one batch used to be caught by the lookup seeing an
    insert made earlier in the same transaction. Every lookup now happens before every
    insert, so the batch has to remember its own turns."""
    pipe, store, _ = build()
    first, second = ep("My name is Goldy."), ep("My name is Goldy.")
    receipt = pipe.add([first, second, ep("I live in Berlin.")])
    assert receipt.episode_ids == [first.id, first.id, receipt.episode_ids[2]]
    assert receipt.skipped == 1
    assert store.stats()["episodes"] == 2
    store.close()


# =============================================================================
# Telemetry emission points
# =============================================================================

def test_the_gate_and_the_fast_path_are_sliced_by_script():
    """Both tier-1 stages are English by construction - a filler vocabulary and a set of
    English sentence patterns - and both fail quietly on text they were not built for.
    Without the slice, "the write path is cheap" is an unqualified claim that may only
    hold for Latin-script users."""
    rec = MemoryRecorder()
    pipe, store, _ = build(telemetry=rec)
    pipe.add([ep("My name is Goldy."), ep("ok thanks"), ep("我住在北京"),
              ep("Anything else?")])
    assert rec.total(GATE_PASS, script="latin") == 1
    assert rec.total(GATE_PASS, script="han") == 1
    assert rec.total(GATE_DROP, reason="ack_only", script="latin") == 1
    assert rec.total(GATE_DROP, reason="question", script="latin") == 1
    assert rec.total(FAST_HIT, script="latin") == 1
    assert rec.total(FAST_MISS, script="han") == 1
    store.close()


def test_novel_predicate_registrations_are_counted():
    """The rate that ran away in the simulation that produced 41 predicates for six
    concepts. It should fall toward zero as a tenant's vocabulary settles; rising
    steadily is predicate explosion, which has no other symptom until `recall()` starts
    answering with four employers at once."""
    rec = MemoryRecorder()
    llm = ResolvingLLM(claims=[
        {"subject": "user", "predicate": "commutes_by", "object": "bicycle",
         "polarity": 1, "memory_type": "semantic", "confidence": 0.9,
         "source_index": 0},
    ], canonical=None)
    pipe, store, _ = build(llm, telemetry=rec)
    pipe.add([ep("I get around town somehow or other.")])
    assert rec.total(PREDICATE_LEARNED) == 1
    assert rec.total(PREDICATE_ALIAS) == 0
    store.close()


def test_a_folded_surface_form_is_counted_as_an_alias_not_a_registration():
    """A form the deterministic pre-pass cannot guess at, which the model then merges.
    The two counters have to move separately: a healthy tenant's alias rate keeps
    climbing while its registration rate goes to zero, and one counter cannot say
    that."""
    rec = MemoryRecorder()
    llm = ResolvingLLM(claims=[
        {"subject": "user", "predicate": "paycheck_source", "object": "Acme",
         "polarity": 1, "memory_type": "semantic", "confidence": 0.9,
         "source_index": 0},
    ], canonical="works_at")
    pipe, store, _ = build(llm, telemetry=rec)
    pipe.add([ep("Payroll switched over to the new provider this quarter.")])
    assert rec.total(PREDICATE_ALIAS) == 1
    assert rec.total(PREDICATE_LEARNED) == 0
    store.close()


def test_the_registry_cap_firing_is_counted_and_says_whether_it_folded():
    """Any of this at all means the ceiling is load-bearing rather than a backstop, and
    `folded=no` means a surface form is live and unregistered - multi-valued, retiring
    nothing, which is where duplicate slots come from."""
    rec = MemoryRecorder()
    predicates = iter(["prior_employer", "zzz_qqq_unrelated"])

    def responder(episodes):
        return [{"subject": "user", "predicate": next(predicates), "object": "Acme",
                 "polarity": 1, "memory_type": "semantic", "confidence": 0.9,
                 "source_index": 0}]

    store = SQLiteStore(":memory:")
    registry = PredicateRegistry(max_learned=0)
    pipe = WritePipeline(store, HashingEmbedder(), registry,
                         ResolvingLLM(responder=responder), telemetry=rec)
    pipe.add([ep("Payroll switched over to the new provider this quarter.")])
    pipe.add([ep("Payroll moved again after the acquisition closed.")])
    assert rec.total(PREDICATE_CAPPED, folded="yes") == 1
    assert rec.total(PREDICATE_CAPPED, folded="no") == 1
    store.close()


def slot_claim(predicate: str, obj: str, subject: str = "user") -> Claim:
    """One caller-asserted triple, the shape `remember()` builds and hands to the writer."""
    return Claim(subject=subject, predicate=predicate, object=obj, scope=SCOPE,
                 derivation=Derivation.USER, extractor="api")


def test_a_value_piling_up_under_an_undeclared_predicate_is_counted():
    """`predicate.accumulated`: one per write that adds a second live answer to a slot
    whose predicate has no spec.

    An unregistered predicate is MANY, MANY retires nothing, and `write.reconcile` reports
    `add` — which is exactly what a correct *first* write reports too, so no existing
    series moves when a slot quietly acquires a second simultaneous answer. This one does,
    and the receipt carries the names the counter deliberately does not tag with."""
    rec = MemoryRecorder()
    pipe, store, _ = build(telemetry=rec)
    pipe.assert_claim(slot_claim("status", "not installed", "quota_gate"))
    second = pipe.assert_claim(slot_claim("status", "installed", "quota_gate"))

    assert rec.total(PREDICATE_ACCUMULATED) == 1
    assert [(a.subject, a.predicate, a.existing) for a in second.accumulated] == [
        ("quota_gate", "status", 1)]
    # Per triggering write, not per slot: the third value counts again, so the rate says
    # how much of this a deployment is doing and not how many slots ever did it once.
    third = pipe.assert_claim(slot_claim("status", "rolled back", "quota_gate"))
    assert rec.total(PREDICATE_ACCUMULATED) == 2
    assert third.accumulated[0].existing == 2
    store.close()


def test_the_counter_stays_at_zero_for_writes_that_are_working_as_designed():
    """A first write to an empty slot, a declared multi-valued predicate accumulating on
    purpose, and a declared single-valued predicate superseding. None of the three is the
    defect, and a counter that fires on them is one nobody can alert on."""
    rec = MemoryRecorder()
    pipe, store, _ = build(telemetry=rec)
    pipe.assert_claim(slot_claim("status", "installed", "quota_gate"))
    pipe.assert_claim(slot_claim("likes", "coffee"))
    pipe.assert_claim(slot_claim("likes", "tea"))
    pipe.assert_claim(slot_claim("lives_in", "Berlin"))
    superseded = pipe.assert_claim(slot_claim("lives_in", "Lisbon"))

    assert len(superseded.closed) == 1
    assert rec.total(PREDICATE_ACCUMULATED) == 0
    # …and the series is genuinely wired here, so the zero above is an absence of the
    # event rather than an absence of the emission point.
    pipe.assert_claim(slot_claim("status", "rolled back", "quota_gate"))
    assert rec.total(PREDICATE_ACCUMULATED) == 1
    store.close()


def test_the_receipt_reports_it_with_no_telemetry_configured():
    """`telemetry=None` is the default and the fast path. The receipt is the account of
    what a write did and cannot be contingent on whether anyone wired a metrics backend —
    the same rule `unextracted` follows."""
    pipe, store, _ = build()
    pipe.assert_claim(slot_claim("status", "not installed", "quota_gate"))
    receipt = pipe.assert_claim(slot_claim("status", "installed", "quota_gate"))
    assert pipe.telemetry is None
    assert [a.predicate for a in receipt.accumulated] == ["status"]
    assert "accumulated=1" in str(receipt)
    store.close()


def test_extraction_that_acquires_a_spec_first_reports_nothing():
    """The population that sees this is the whole point of the trigger. On the `add()`
    path with a model configured, acquisition registers the predicate *before* its claim
    reaches the reconciler, so the slot is governed by a real spec by the time anything
    lands in it and nothing is reported. What is left is exactly where cardinality can
    never be learned: `remember()`, and a deployment with no extraction model."""
    rec = MemoryRecorder()
    objs = iter(["not installed", "installed"])
    llm = ResolvingLLM(classification={"cardinality": "one", "volatility": "fast",
                                       "memory_type": "semantic"},
                       responder=lambda episodes: [
                           {"subject": "quota_gate", "predicate": "status",
                            "object": next(objs), "polarity": 1, "source_index": 0}])
    pipe, store, registry = build(llm, telemetry=rec)
    pipe.add([ep("The quota gate is not installed yet.")])
    second = pipe.add([ep("The quota gate was installed this morning.")])

    assert registry.known("status") and registry.spec("status").functional
    assert second.accumulated == [] and rec.total(PREDICATE_ACCUMULATED) == 0
    assert len(second.closed) == 1        # it superseded, which is the whole point
    store.close()


def test_a_retraction_that_retires_nothing_is_recorded_as_an_anomaly():
    """`forget()` returns an ordinary receipt whether it cleared the slot or matched no
    claim at all. Two things produce the empty case and both matter: a user failing to
    take back something poisoned into their memory, and a retraction whose object does
    not match what is on record."""
    rec = MemoryRecorder()
    pipe, store, _ = build(telemetry=rec)
    pipe.add([ep("I work at Acme.")])

    pipe.assert_claim(Claim(subject="user", predicate="works_at", object="Globex",
                            polarity=-1, scope=SCOPE))
    assert rec.total(WRITE_RETRACTION, outcome="noop") == 1

    pipe.assert_claim(Claim(subject="user", predicate="works_at", object="Acme",
                            polarity=-1, scope=SCOPE))
    assert rec.total(WRITE_RETRACTION, outcome="retired") == 1
    store.close()


def test_reconciliation_outcomes_are_counted_by_action():
    rec = MemoryRecorder()
    pipe, store, _ = build(telemetry=rec)
    pipe.add([ep("I live in Berlin.")])
    pipe.add([ep("I live in Lisbon.")])
    assert rec.total(WRITE_RECONCILE, action="add") == 1
    assert rec.total(WRITE_RECONCILE, action="supersede") == 1
    store.close()


def test_a_rejected_embedding_is_counted_every_time_not_only_warned_once():
    """Warn-once is right for a human reading stderr and useless for anyone asking six
    months later how big the hole is."""
    rec = MemoryRecorder()
    pipe, store, _ = build(telemetry=rec)
    pipe.add([ep("I live in Berlin.")])
    pipe.embedder = HashingEmbedder(dim=8)          # wrong dimension for this index
    with pytest.warns(RuntimeWarning):
        pipe.add([ep("My name is Goldy.")])
    pipe.add([ep("I work at Acme.")])                # warns no more; still counted
    assert rec.total(WRITE_EMBEDDING_REJECTED) == 2
    store.close()


def test_the_write_reports_how_long_it_held_the_lock_separately_from_its_own_latency():
    """The gap between the two is the work the rest of the process was not blocked by,
    and before the hoist it was zero."""
    rec = MemoryRecorder()

    class Slow(CountingLLM):
        def extract(self, episodes, known_predicates):
            time.sleep(0.05)
            return super().extract(episodes, known_predicates)

    pipe, store, _ = build(Slow(), telemetry=rec)
    pipe.add([ep("The offsite moved to the Lisbon office.")])
    held = rec.values(WRITE_LOCK_HELD_MS)[0]
    total = rec.values(WRITE_LATENCY_MS)[0]
    assert total >= 50.0, "the fake extraction did not run"
    assert held < total / 2.0, (
        f"held the transaction for {held:.1f}ms of a {total:.1f}ms write")
    store.close()


def test_assert_claim_reports_latency_without_inventing_a_lock_window():
    rec = MemoryRecorder()
    pipe, store, _ = build(telemetry=rec)
    pipe.assert_claim(Claim(subject="user", predicate="likes", object="tea",
                            scope=SCOPE))
    assert len(rec.values(WRITE_LATENCY_MS)) == 1
    # This path opens no transaction of its own, so there is no window to report.
    assert rec.values(WRITE_LOCK_HELD_MS) == []
    store.close()


def test_an_empty_batch_emits_nothing():
    rec = MemoryRecorder()
    pipe, store, _ = build(telemetry=rec)
    assert pipe.add([]).episode_ids == []
    assert rec.names() == []
    store.close()


def test_evidence_roles_reaches_the_gate_from_the_pipeline_constructor():
    """`write_evidence_roles=` on `Memvara` works because this parameter is keyword-only
    here — the facade forwards `write_*` by reading this signature. Pinned so the
    passthrough cannot be broken by making it positional."""
    pipe, store, _ = build(evidence_roles=None)
    assert pipe.gate.evidence_roles is None
    assert pipe.gate.carries_fact(ep("I just started at Acme", role="melanie"))[0]

    default, _, _ = build()
    assert default.gate.evidence_roles == SalienceGate.DEFAULT_EVIDENCE_ROLES
    store.close()


def test_a_redactor_does_not_relabel_a_non_latin_turn_as_latin():
    """The gate/fast-path script slices exist to measure how English-centric tier 1 is.
    Redaction runs first, and a replacement token is Latin — "[redacted:phone]" is
    thirteen Latin letters — so on a short Han turn it outvoted the real script and the
    slice reported `latin`. The signal was wrong precisely for the deployments careful
    enough to turn redaction on."""
    from memvara.redact import PatternRedactor
    from memvara.telemetry import MemoryRecorder, script_of

    turn = "我住在北京，电话 555-123-4567"
    assert script_of(turn) == "han", "fixture no longer exercises the case"

    rec = MemoryRecorder()
    pipe, store, _ = build(telemetry=rec, redactor=PatternRedactor())
    pipe.add([ep(turn)])

    assert rec.total("gate.pass", script="han") == 1
    assert rec.total("gate.pass", script="latin") == 0
    assert rec.total("fast.miss", script="han") == 1
    # And the redaction still happened — this is not the fix working by not redacting.
    assert rec.total("redact.changed") >= 1
    store.close()


def test_the_script_is_read_from_the_stored_turn_when_no_redactor_rewrote_it():
    """The pre-redaction map is only populated when a redactor is configured, so the
    unredacted path must still classify. Cheap to get wrong by making the map the only
    source."""
    from memvara.telemetry import MemoryRecorder

    rec = MemoryRecorder()
    pipe, store, _ = build(telemetry=rec)
    pipe.add([ep("我住在北京")])
    assert rec.total("gate.pass", script="han") == 1
    store.close()
