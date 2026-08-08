"""WritePipeline: the cost ladder, measured.

Every test here asserts on `receipt.llm_calls`. That is not incidental — the entire
reason this subsystem exists is that mem0 spends one LLM call per write, and a test that
only checks the claims came out would pass just as happily on a design that called a
model on every turn. The fake LLM below therefore counts calls and records batch sizes,
and refuses to be used without those numbers being checked.

Runs fully offline: SQLiteStore(":memory:"), HashingEmbedder, and the local fake.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Sequence

import pytest

from engram.embed import HashingEmbedder
from engram.schema import Cardinality, PredicateRegistry
from engram.store import SQLiteStore
from engram.types import Claim, Derivation, Episode, Scope, utcnow
from engram.write import WritePipeline


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
    return sorted((c.predicate, c.object) for c in store.iter_claims(tenant))


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
                for i, (p, o) in enumerate([("owns_pet", "greyhound"),
                                            ("goal", "learn rust"),
                                            ("likes", "oakwood elementary")])]

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
    assert stored.invalidated_at is not None
    assert stored.valid_to is not None
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
    assert store.get_claim(added.added[0].id).invalidated_at is not None
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
    pipe.add([ep("Been hunting rare pressings again.")])
    receipt = pipe.add([ep("Found a crate of pre-war postage today.")])

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
    receipt = pipe.add([ep("Went with the flatter switches in the end.")])

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


def test_confidence_is_clamped():
    llm = CountingLLM(claims=[
        {"subject": "user", "predicate": "likes", "object": "tea", "polarity": 1,
         "memory_type": "semantic", "confidence": 7.5, "source_index": 0},
    ])
    pipe, store, _ = build(llm)
    receipt = pipe.add([ep("The quarterly review is next Tuesday.")])
    assert receipt.added[0].confidence == 1.0
    store.close()


def test_source_index_maps_back_to_the_right_episode():
    llm = CountingLLM(claims=[
        {"subject": "user", "predicate": "likes", "object": "tea", "polarity": 1,
         "memory_type": "semantic", "confidence": 0.9, "source_index": 1},
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
