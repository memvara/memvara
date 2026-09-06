"""The pollution guard, held to the measurement it was built from.

`tests/fixtures/phi4_spike/` is what a small model actually produced over three real turns
under six configurations — 255 claims. The scorer below is the spike's own, moved here so
the guard's effect is a number the suite holds rather than one somebody remembers: how
many wrong-predicate claims it removes, how many duplicates, and — the one that decides
whether it is a guard or a loss — how many keyed facts it costs. That last number is zero
in every configuration, and this file is where that stops being a claim.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from memvara.embed import HashingEmbedder
from memvara.schema import PredicateRegistry
from memvara.store.sqlite import SQLiteStore
from memvara.types import Claim, Episode, Scope
from memvara.write import pollution
from memvara.write.pipeline import WritePipeline

FIXTURE = Path(__file__).parent / "fixtures" / "phi4_spike"
KEYS = json.loads((FIXTURE / "keys.json").read_text())
CONFIGS = ("12", "24", "32", "none", "A", "B")
EPISODES = ("gate", "billing", "retrieval")


def _snake(raw) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(raw or "").lower()).strip("_")


def classify(claims, episode: str):
    """The spike's scorer, verbatim in logic: hit / wrong-predicate / duplicate / unkeyed,
    and the set of keyed facts found."""
    facts = KEYS["facts"][episode]
    generic = set(KEYS["generic_predicates"])
    seen, buckets, found = set(), {"hit": 0, "wrong_predicate": 0, "duplicate": 0,
                                    "unkeyed": 0}, set()
    labels: list[str] = []
    for c in claims:
        subj = str(c.get("subject", "")).lower()
        pred = _snake(c.get("predicate", ""))
        obj = str(c.get("object", "")).lower()
        if (pred, obj) in seen:
            buckets["duplicate"] += 1
            labels.append("duplicate")
            continue
        seen.add((pred, obj))
        matched = None
        for fact in facts:
            if re.search(fact["object_re"], obj, re.I):
                matched = fact["fact"]
                if pred in fact["predicates"] or (
                        pred in generic and re.search(fact["subject_re"], subj, re.I)):
                    buckets["hit"] += 1
                    found.add(fact["fact"])
                    matched = "__hit__"
                break
        if matched is None:
            buckets["unkeyed"] += 1
            labels.append("unkeyed")
        elif matched != "__hit__":
            buckets["wrong_predicate"] += 1
            labels.append("wrong_predicate")
        else:
            labels.append("hit")
    return buckets, found, labels


def _load(config: str, episode: str) -> list[dict]:
    return json.loads((FIXTURE / "claims" / f"{config}-{episode}.json").read_text())


# -- the measurement, as a test ------------------------------------------------------------


def test_the_fixture_is_what_was_measured():
    """255 claims: 68 hits, 46 wrong-predicate, 32 duplicates, 109 unkeyed, 60 of 90 keyed
    facts found. If this moves, the fixture moved, and every number below is suspect."""
    totals = {"hit": 0, "wrong_predicate": 0, "duplicate": 0, "unkeyed": 0}
    found = 0
    for config in CONFIGS:
        for episode in EPISODES:
            buckets, facts, _ = classify(_load(config, episode), episode)
            for k in totals:
                totals[k] += buckets[k]
            found += len(facts)
    assert totals == {"hit": 68, "wrong_predicate": 46, "duplicate": 32, "unkeyed": 109}
    assert found == 60


def test_the_guard_removes_26_of_46_wrong_predicate_claims_and_every_duplicate_at_no_fact_cost():
    """The table in `write/pollution.py`, held. Wrong-predicate 46 → 20, duplicates 32 → 0,
    and keyed facts found 60/90 before and 60/90 after — in **every** configuration, not
    only in total, because a rule that trades one arm's recall for another's would hide in
    a sum."""
    registry = PredicateRegistry()
    wrong_before = wrong_after = dupes_before = dupes_after = 0
    for config in CONFIGS:
        found_before = found_after = 0
        for episode in EPISODES:
            claims = _load(config, episode)
            before, facts_before, _ = classify(claims, episode)
            kept, refused, _ = pollution.guard(claims, registry)
            after, facts_after, _ = classify(kept, episode)
            assert len(kept) + refused == len(claims)
            wrong_before += before["wrong_predicate"]; wrong_after += after["wrong_predicate"]
            dupes_before += before["duplicate"]; dupes_after += after["duplicate"]
            found_before += len(facts_before); found_after += len(facts_after)
        assert found_after == found_before, f"config {config} lost a keyed fact"
    assert (wrong_before, wrong_after) == (46, 20)
    assert (dupes_before, dupes_after) == (32, 0)


def test_the_only_hits_the_guard_removes_are_second_hits_on_a_fact_already_found():
    """Three "hits" go: all `gate / status / Port 61434`, each beside an identical claim
    under `port`. Deduplication, not loss — the previous test's per-config fact count is
    the proof, and this one names the three so a change in them is visible."""
    registry = PredicateRegistry()
    removed_hits = []
    for config in CONFIGS:
        for episode in EPISODES:
            claims = _load(config, episode)
            _, _, labels = classify(claims, episode)
            kept, _, _ = pollution.guard(claims, registry)
            # The guard returns the original dicts (or a copy with only `confidence`
            # changed), so identity by (subject, predicate, object) is exact here.
            kept_keys = {(k.get("subject"), k.get("predicate"), k.get("object")) for k in kept}
            for c, label in zip(claims, labels):
                key = (c.get("subject"), c.get("predicate"), c.get("object"))
                if label == "hit" and key not in kept_keys:
                    removed_hits.append((config, _snake(c["predicate"]), c["object"]))
    assert sorted(removed_hits) == [("24", "status", "Port 61434"),
                                    ("32", "status", "Port 61434"),
                                    ("none", "status", "Port 61434")]


def test_arm_a_the_production_candidate_keeps_four_of_its_six_polluted_claims_out():
    """The configuration a deployment would run. Two escapes are named in the module
    docstring. `live_worker_version` is novel and R4 lands it at 0.4; `goal` is a MANY slot
    that ends nothing whatever arrives, so it is stored as proposed and threatens nothing."""
    registry = PredicateRegistry()
    escapes = []
    for episode in EPISODES:
        claims = _load("A", episode)
        kept, _, _ = pollution.guard(claims, registry)
        _, _, labels = classify(kept, episode)
        for c, label in zip(kept, labels):
            if label == "wrong_predicate":
                escapes.append((_snake(c["predicate"]), c["object"], c["confidence"]))
    assert sorted(escapes) == [
        ("goal", "refuse", 1.0),
        ("live_worker_version", "9000 memories a month", 0.4),
    ]


# -- the rules, one at a time --------------------------------------------------------------


def _item(subject, predicate, obj, confidence=0.9, index=0):
    return {"subject": subject, "predicate": predicate, "object": obj, "polarity": 1,
            "memory_type": "semantic", "confidence": confidence, "source_index": index}


def test_r1_one_value_under_several_unknown_predicates_keeps_the_first():
    registry = PredicateRegistry()
    raw = [_item("gate", "endpoint", "Port 61434"), _item("gate", "build_status", "Port 61434"),
           _item("gate", "port", "Port 61434")]
    kept, refused, _ = pollution.guard(raw, registry)
    assert [c["predicate"] for c in kept] == ["endpoint"] and refused == 2


def test_r1_keeps_every_known_predicate_and_drops_the_unknown_ones_beside_them():
    """The measured failure is unknown predicates piling onto a found value. Two known
    slots for one value in one turn — born in Lisbon, still living there — are both facts."""
    registry = PredicateRegistry()
    raw = [_item("user", "born_in", "Lisbon"), _item("user", "lives_in", "Lisbon"),
           _item("user", "city_of_record", "Lisbon")]
    kept, refused, _ = pollution.guard(raw, registry)
    assert [c["predicate"] for c in kept] == ["born_in", "lives_in"] and refused == 1
    # `employer_name` folds onto `works_at`, so both are known and both stay; the
    # reconciler sees one slot twice and reinforces rather than storing two values.
    raw = [_item("user", "employer_name", "Acme"), _item("user", "works_at", "Acme")]
    kept, refused, _ = pollution.guard(raw, registry)
    assert refused == 0 and len(kept) == 2


def test_r1_groups_within_a_turn_not_across_the_batch():
    """`add()` extracts a whole batch in one call. Two turns agreeing on a value is
    evidence, not the failure; one turn wearing the same value on three predicates is."""
    registry = PredicateRegistry()
    raw = [_item("user", "likes", "Lisbon", index=0), _item("user", "likes", "Lisbon", index=1)]
    kept, refused, _ = pollution.guard(raw, registry)
    assert refused == 0 and len(kept) == 2
    raw = [_item("user", "likes", "Lisbon", index=0), _item("user", "visited", "Lisbon", index=0)]
    kept, refused, _ = pollution.guard(raw, registry)
    assert refused == 1 and kept[0]["predicate"] == "likes"


def test_r1_does_not_touch_the_same_value_on_different_subjects():
    registry = PredicateRegistry()
    raw = [_item("alice", "lives_in", "Lisbon"), _item("bob", "lives_in", "Lisbon")]
    kept, refused, _ = pollution.guard(raw, registry)
    assert refused == 0 and len(kept) == 2


def test_a_speaker_predicate_on_a_named_third_party_is_a_fact_not_pollution():
    """There is deliberately no subject rule. "My wife Alice lives in Porto" is
    `alice / lives_in / Porto`, its own slot, which cannot end the speaker's; and a
    two-person conversation — the shape `read_route_roles=False` exists for — yields
    nothing else. The first draft refused all of these and was measured to catch nothing
    R3 did not."""
    registry = PredicateRegistry()
    kept, refused, _ = pollution.guard([_item("alice", "lives_in", "Porto"),
                                        _item("caroline", "works_at", "the clinic"),
                                        _item("vector index", "type", "hnsw")], registry)
    assert refused == 0 and len(kept) == 3
    # A blank subject means the speaker, as `_claim_from_dict` reads it.
    kept, refused, _ = pollution.guard([_item("", "lives_in", "Lisbon")], registry)
    assert refused == 0


def test_r3_refuses_a_place_predicate_whose_object_is_not_a_place():
    registry = PredicateRegistry()
    for obj in ("Port 61434", "http://x", "console.aurora-notes.dev", "Berlin 2"):
        _, refused, _ = pollution.guard([_item("user", "lives_in", obj)], registry)
        assert refused == 1, obj
    # On any subject: this is the fixture's own `gate / lives_in / Port 61434`.
    _, refused, _ = pollution.guard([_item("gate", "lives_in", "Port 61434")], registry)
    assert refused == 1
    for obj in ("Lisbon", "the Porto office", "New York"):
        _, refused, _ = pollution.guard([_item("user", "lives_in", obj)], registry)
        assert refused == 0, obj
    # Narrow on purpose: a digit under a ONE-slot non-place predicate is R4's, a discount
    # rather than a refusal — and a Roman numeral is not a digit.
    kept, refused, _ = pollution.guard([_item("user", "job_title", "Engineer 2")], registry)
    assert refused == 0 and kept[0]["confidence"] == 0.4
    kept, refused, _ = pollution.guard([_item("user", "job_title", "Engineer II")], registry)
    assert refused == 0 and kept[0]["confidence"] == 0.9


def test_r4_discounts_a_novel_predicate_and_a_risky_functional_slot_and_nothing_else():
    registry = PredicateRegistry()
    kept, refused, discounted = pollution.guard([
        _item("user", "live_worker_version", "9000 memories a month"),   # novel
        _item("user", "works_at", "Port 61434"),                          # ONE slot, digit
        _item("user", "works_at", "Globex"),                              # clean
        _item("user", "likes", "Lisbon 2024", confidence=0.3),            # MANY: untouched
    ], registry)
    assert refused == 0 and discounted == 2
    assert [c["confidence"] for c in kept] == [0.4, 0.4, 0.9, 0.3]
    # The two ONE-slot builtins whose values always carry digits are left alone: a real
    # UTC offset at 0.4 could never supersede the stale one it corrects.
    kept, refused, discounted = pollution.guard([
        _item("user", "timezone", "UTC+5:30"), _item("user", "born_on", "1990-05-01"),
    ], registry)
    assert (refused, discounted) == (0, 0) and [c["confidence"] for c in kept] == [0.9, 0.9]


def test_r4_reads_an_unparseable_confidence_as_the_shaping_layer_s_default():
    """`_claim_from_dict` reads a confidence it cannot parse as 0.7; the discount has to
    agree with it, or a novel predicate arriving with `"confidence": "high"` would be
    stored at 0.7 undiscounted while every well-formed one lands at 0.4."""
    registry = PredicateRegistry()
    kept, refused, discounted = pollution.guard(
        [_item("user", "live_worker_version", "9000 memories a month", confidence="high")],
        registry)
    assert (refused, discounted) == (0, 1) and kept[0]["confidence"] == 0.4


def test_the_guard_passes_malformed_items_through_for_the_shaping_layer_to_drop():
    registry = PredicateRegistry()
    raw = [{"subject": "user", "predicate": "", "object": "x", "source_index": 0},
           {"subject": "user", "predicate": "likes", "object": "", "source_index": 0}]
    kept, refused, discounted = pollution.guard(raw, registry)
    assert (kept, refused, discounted) == (raw, 0, 0)


# -- through the pipeline: the destructive direction, stopped ------------------------------


class Proposes:
    is_noop = False
    reports_usage = False
    name = "fake/model"

    def __init__(self, *claims):
        self.claims = list(claims)

    def extract(self, episodes, known_predicates):
        return [dict(c) for c in self.claims]


def _pipe(llm, **kw):
    store = SQLiteStore(":memory:")
    pipe = WritePipeline(store, HashingEmbedder(dim=32), PredicateRegistry(), llm, **kw)
    return pipe, store


def _live(store, predicate):
    return sorted((c.subject, c.object) for c in store.iter_claims("acme")
                  if c.predicate == predicate and c.is_live())


def _ep(text):
    return Episode(content=text, role="user", scope=Scope("acme", "alice"))


def test_a_polluted_claim_cannot_end_the_true_fact_in_a_one_cardinality_slot():
    """`gate / lives_in / Port 61434`, the fixture's own example, against a store that
    already knows the user lives in Lisbon. Lisbon is still current afterwards, the
    refusal is counted, and the turn that yielded only pollution reads as unextracted."""
    pipe, store = _pipe(Proposes(_item("gate", "lives_in", "Port 61434")))
    pipe.assert_claim(Claim(subject="user", predicate="lives_in", object="Lisbon",
                            scope=Scope("acme", "alice")))
    receipt = pipe.add([_ep("The gate starts its own Postgres on port 61434.")])
    assert receipt.polluted == 1 and receipt.added == []
    assert receipt.unextracted == 1
    assert _live(store, "lives_in") == [("user", "Lisbon")]
    store.close()


def test_a_discounted_claim_is_stored_beside_the_incumbent_not_over_it():
    """R4 through the reconciler's half rule: `user / works_at / Port 61434` arrives at
    0.9 and is stored at 0.4, so Globex at 1.0 is not ended."""
    pipe, store = _pipe(Proposes(_item("user", "works_at", "Port 61434")))
    pipe.assert_claim(Claim(subject="user", predicate="works_at", object="Globex",
                            scope=Scope("acme", "alice"), confidence=1.0))
    receipt = pipe.add([_ep("The gate starts its own Postgres on port 61434.")])
    assert receipt.polluted == 0 and len(receipt.added) == 1
    assert receipt.added[0].confidence == 0.4
    assert ("user", "Globex") in _live(store, "works_at")
    store.close()


def test_the_guard_can_be_turned_off_and_then_the_receipt_says_nothing():
    pipe, store = _pipe(Proposes(_item("gate", "lives_in", "Port 61434")),
                        reject_polluted=False)
    receipt = pipe.add([_ep("The gate starts its own Postgres on port 61434.")])
    assert receipt.polluted == 0 and len(receipt.added) == 1
    store.close()


def test_a_refused_duplicate_s_novel_predicate_is_never_acquired():
    """R1 runs before acquisition on purpose: an invented predicate beside `works_at` for
    the same value must not cost a model call to register the pollution's spelling."""
    class Counting(Proposes):
        resolve_calls = 0

        def resolve_predicate(self, surface, candidates):
            self.resolve_calls += 1
            return {"predicate": surface, "is_new": True}

        def classify_predicate(self, *a, **k):
            self.resolve_calls += 1
            return {"cardinality": "many", "volatility": "slow", "memory_type": "semantic"}

    llm = Counting(_item("user", "payroll_company_field", "Acme"),
                   _item("user", "works_at", "Acme"))
    pipe, store = _pipe(llm)
    receipt = pipe.add([_ep("The payroll record now names Acme after the review.")])
    assert receipt.polluted == 1
    assert llm.resolve_calls == 0
    assert _live(store, "works_at") == [("user", "Acme")]
    store.close()


def test_memvara_exposes_the_option_under_the_write_prefix():
    from memvara import Memvara
    mem = Memvara(":memory:", llm=Proposes(_item("gate", "lives_in", "Port 61434")),
                  embedder=HashingEmbedder(dim=32))
    assert mem.writer.reject_polluted is True
    mem.close()
    mem = Memvara(":memory:", llm=Proposes(), embedder=HashingEmbedder(dim=32),
                  write_reject_polluted=False)
    assert mem.writer.reject_polluted is False
    mem.close()
