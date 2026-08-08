"""Duplicate merging and episodic -> semantic promotion."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from engram.consolidate import Consolidator
from engram.embed.base import HashingEmbedder
from engram.schema import PredicateRegistry
from engram.store.sqlite import SQLiteStore
from engram.types import Claim, Derivation, MemoryType, Scope, utcnow

NOW = utcnow()
SCOPE = Scope(tenant="acme", user="u1")


@pytest.fixture
def consolidator():
    store = SQLiteStore(":memory:")
    yield Consolidator(store, HashingEmbedder(dim=256), PredicateRegistry())
    store.close()


def add(store, claim_id: str, obj: str, *, predicate: str = "works_at",
        obs: int = 1, age_days: float = 0.0, sources=(), **kw) -> Claim:
    ts = NOW - timedelta(days=age_days)
    claim = Claim(
        id=claim_id,
        subject="user",
        predicate=predicate,
        object=obj,
        scope=SCOPE,
        observation_count=obs,
        sources=list(sources),
        valid_from=ts,
        recorded_at=ts,
        **kw,
    )
    store.put_claim(claim)
    return claim


def live_ids(store) -> set[str]:
    return {c.id for c in store.iter_claims("acme", include_invalidated=False)}


# -- survivor selection -----------------------------------------------------


def test_survivor_is_the_best_attested_claim(consolidator):
    store = consolidator.store
    add(store, "cl_thin", "Acme", obs=1)
    add(store, "cl_thick", "acme", obs=7)

    assert consolidator.merge_duplicates() == 1
    assert live_ids(store) == {"cl_thick"}


@pytest.mark.parametrize("order", [[0, 1, 2], [2, 1, 0], [1, 0, 2]])
def test_equal_evidence_breaks_on_earliest_recorded_at(order):
    store = SQLiteStore(":memory:")
    c = Consolidator(store, HashingEmbedder(dim=256), PredicateRegistry())
    rows = [
        ("cl_b_old", 10.0),   # believed longest -> wins despite the unlucky id
        ("cl_a_new", 1.0),
        ("cl_z_new", 1.0),
    ]
    for i in order:
        claim_id, age = rows[i]
        add(store, claim_id, "acme", obs=3, age_days=age)

    assert c.merge_duplicates() == 2
    assert live_ids(store) == {"cl_b_old"}
    store.close()


@pytest.mark.parametrize("order", [["cl_aaa", "cl_zzz"], ["cl_zzz", "cl_aaa"]])
def test_a_dead_heat_is_settled_by_id(order):
    """Same count, same instant: without a third key the winner would be arbitrary."""
    store = SQLiteStore(":memory:")
    c = Consolidator(store, HashingEmbedder(dim=256), PredicateRegistry())
    for claim_id in order:
        add(store, claim_id, "acme", obs=3, age_days=4.0)

    assert c.merge_duplicates() == 1
    assert live_ids(store) == {"cl_aaa"}
    store.close()


def test_survivor_is_identical_across_repeated_independent_runs():
    """Same data, different insertion order, same answer - twenty times over."""
    survivors = set()
    for seed in range(20):
        store = SQLiteStore(":memory:")
        c = Consolidator(store, HashingEmbedder(dim=256), PredicateRegistry())
        rows = [("cl_x", 4, 2.0), ("cl_y", 4, 2.0), ("cl_w", 4, 9.0), ("cl_v", 2, 30.0)]
        rng = np.random.default_rng(seed)
        for i in rng.permutation(len(rows)):
            claim_id, obs, age = rows[int(i)]
            add(store, claim_id, "acme", obs=obs, age_days=age)

        c.merge_duplicates()
        survivors.add(frozenset(live_ids(store)))
        store.close()

    assert survivors == {frozenset({"cl_w"})}


# -- what the merge preserves -----------------------------------------------


def test_sources_and_observation_counts_are_folded_into_the_survivor(consolidator):
    store = consolidator.store
    add(store, "cl_win", "acme", obs=5, sources=["ep_1", "ep_2"])
    add(store, "cl_a", "Acme", obs=3, sources=["ep_2", "ep_3"])
    add(store, "cl_b", "ACME", obs=2, sources=["ep_4"])

    assert consolidator.merge_duplicates() == 2

    survivor = store.get_claim("cl_win")
    assert survivor.observation_count == 10
    # `ep_2` was already known; provenance is a set, not a bag.
    assert survivor.sources == ["ep_1", "ep_2", "ep_3", "ep_4"]


def test_losers_are_invalidated_not_deleted_and_point_at_the_survivor(consolidator):
    store = consolidator.store
    add(store, "cl_win", "acme", obs=5)
    add(store, "cl_lose", "Acme", obs=1)

    consolidator.merge_duplicates()

    stored = store.get_claim("cl_lose")
    assert stored is not None, "merged-away claims must never be hard-deleted"
    assert stored.invalidated_by == "cl_win"
    assert stored.invalidated_at is not None
    # The duplicate was never *false*, only redundant, so valid time is untouched and a
    # time-travel query from before the merge still sees the world as it was.
    assert stored.valid_to is None
    assert stored.is_live(as_of=stored.invalidated_at - timedelta(microseconds=1))
    assert not stored.is_live()
    assert store.stats()["claims"] == 2


def test_merge_does_not_cross_fact_keys(consolidator):
    """Different questions are never duplicates, however similar the text."""
    store = consolidator.store
    add(store, "cl_job", "acme", predicate="works_at")
    add(store, "cl_home", "acme", predicate="lives_in")

    assert consolidator.merge_duplicates() == 0
    assert live_ids(store) == {"cl_job", "cl_home"}


def test_distinct_values_in_the_same_slot_are_left_alone(consolidator):
    store = consolidator.store
    add(store, "cl_coffee", "coffee", predicate="likes")
    add(store, "cl_tea", "green tea", predicate="likes")

    assert consolidator.merge_duplicates() == 0
    assert live_ids(store) == {"cl_coffee", "cl_tea"}


def test_threshold_is_respected(consolidator):
    store = consolidator.store
    add(store, "cl_coffee", "coffee", predicate="likes", obs=2)
    add(store, "cl_tea", "green tea", predicate="likes")

    # A threshold low enough to call anything a duplicate should merge them; the point is
    # that the cutoff is what decides, not some hidden heuristic.
    assert consolidator.merge_duplicates(threshold=0.0) == 1
    assert live_ids(store) == {"cl_coffee"}


def test_two_separate_clusters_in_one_slot_each_keep_a_survivor(consolidator):
    store = consolidator.store
    add(store, "cl_coffee_a", "coffee", predicate="likes", obs=5)
    add(store, "cl_coffee_b", "Coffee", predicate="likes", obs=1)
    add(store, "cl_tea_a", "green tea", predicate="likes", obs=4)
    add(store, "cl_tea_b", "Green Tea", predicate="likes", obs=2)

    assert consolidator.merge_duplicates() == 2
    assert live_ids(store) == {"cl_coffee_a", "cl_tea_a"}
    assert store.get_claim("cl_coffee_a").observation_count == 6
    assert store.get_claim("cl_tea_a").observation_count == 6


def test_already_invalidated_claims_are_not_merge_candidates(consolidator):
    store = consolidator.store
    add(store, "cl_win", "acme", obs=5)
    add(store, "cl_dead", "Acme", obs=9)
    store.invalidate("cl_dead", at=NOW - timedelta(days=1), by="cl_win")

    assert consolidator.merge_duplicates() == 0
    assert store.get_claim("cl_dead").invalidated_by == "cl_win"


def test_merge_is_scoped_to_one_tenant(consolidator):
    store = consolidator.store
    add(store, "cl_mine", "acme", obs=2)
    other = Claim(id="cl_theirs", subject="user", predicate="works_at", object="acme",
                  scope=Scope(tenant="other"))
    store.put_claim(other)

    assert consolidator.merge_duplicates(tenant="acme") == 0
    assert live_ids(store) == {"cl_mine"}
    assert store.get_claim("cl_theirs").invalidated_at is None


# -- degenerate inputs ------------------------------------------------------


def test_empty_store(consolidator):
    assert consolidator.merge_duplicates() == 0
    assert consolidator.promote() == 0


def test_single_claim_store(consolidator):
    add(consolidator.store, "cl_only", "acme")
    assert consolidator.merge_duplicates() == 0
    assert live_ids(consolidator.store) == {"cl_only"}


def test_claim_with_no_embedding_does_not_crash_or_merge(consolidator):
    """A claim that encodes to a zero vector has no direction, so it matches nothing."""
    store = consolidator.store
    add(store, "cl_real", "acme", obs=3)
    add(store, "cl_blank", "acme", obs=1, text=" ")

    assert np.linalg.norm(consolidator.embedder.encode([" "])[0]) == 0.0
    assert consolidator.merge_duplicates() == 0
    assert live_ids(store) == {"cl_real", "cl_blank"}


def test_merge_works_without_any_stored_embeddings(consolidator):
    """`set_embedding` is never called here - merging must not depend on it."""
    store = consolidator.store
    add(store, "cl_win", "acme", obs=4)
    add(store, "cl_lose", "Acme", obs=1)

    assert store.stats()["embeddings"] == 0
    assert consolidator.merge_duplicates() == 1


def test_unicode_objects_merge_on_meaning_not_bytes(consolidator):
    store = consolidator.store
    add(store, "cl_a", "Lisbon", predicate="lives_in", obs=3)
    add(store, "cl_b", "lisbon", predicate="lives_in", obs=1)
    add(store, "cl_c", "東京", predicate="lives_in", obs=1)

    assert consolidator.merge_duplicates() == 1
    assert live_ids(store) == {"cl_a", "cl_c"}


# -- idempotency ------------------------------------------------------------


def test_merging_twice_does_not_re_merge_or_double_count(consolidator):
    store = consolidator.store
    add(store, "cl_win", "acme", obs=5, sources=["ep_1"])
    add(store, "cl_lose", "Acme", obs=3, sources=["ep_2"])

    assert consolidator.merge_duplicates() == 1
    after = store.get_claim("cl_win")
    assert after.observation_count == 8

    assert consolidator.merge_duplicates() == 0
    assert store.get_claim("cl_win").observation_count == 8
    assert store.get_claim("cl_win").sources == ["ep_1", "ep_2"]
    assert store.stats()["live_claims"] == 1


# -- promotion --------------------------------------------------------------


def test_promote_at_and_above_min_observations(consolidator):
    store = consolidator.store
    add(store, "cl_at", "ship v2", predicate="goal", obs=3,
        memory_type=MemoryType.EPISODIC)
    add(store, "cl_above", "ship v3", predicate="goal", obs=9,
        memory_type=MemoryType.EPISODIC)

    assert consolidator.promote(min_observations=3) == 2
    for claim_id in ("cl_at", "cl_above"):
        stored = store.get_claim(claim_id)
        assert stored.memory_type is MemoryType.SEMANTIC
        assert stored.derivation is Derivation.CONSOLIDATION
        assert stored.meta["promoted_from"] == "episodic"


def test_promote_never_below_min_observations(consolidator):
    store = consolidator.store
    add(store, "cl_under", "ship v2", predicate="goal", obs=2,
        memory_type=MemoryType.EPISODIC)

    assert consolidator.promote(min_observations=3) == 0
    stored = store.get_claim("cl_under")
    assert stored.memory_type is MemoryType.EPISODIC
    assert stored.derivation is Derivation.LLM_EXTRACT


def test_promote_honours_a_custom_threshold(consolidator):
    store = consolidator.store
    add(store, "cl_five", "ship v2", predicate="goal", obs=5,
        memory_type=MemoryType.EPISODIC)

    assert consolidator.promote(min_observations=6) == 0
    assert consolidator.promote(min_observations=5) == 1


def test_promote_ignores_non_episodic_claims(consolidator):
    store = consolidator.store
    add(store, "cl_sem", "acme", obs=50)
    add(store, "cl_proc", "pytest", predicate="prefers_tool", obs=50,
        memory_type=MemoryType.PROCEDURAL)

    assert consolidator.promote(min_observations=3) == 0
    assert store.get_claim("cl_proc").memory_type is MemoryType.PROCEDURAL


def test_promoting_twice_is_a_no_op(consolidator):
    store = consolidator.store
    add(store, "cl_goal", "ship v2", predicate="goal", obs=4,
        memory_type=MemoryType.EPISODIC)

    assert consolidator.promote() == 1
    assert consolidator.promote() == 0
    assert store.get_claim("cl_goal").memory_type is MemoryType.SEMANTIC


def test_merge_folds_counts_so_a_split_pattern_can_promote(consolidator):
    """Two rows at 2 observations each are one pattern, not two events."""
    store = consolidator.store
    add(store, "cl_goal_a", "ship v2", predicate="goal", obs=2,
        memory_type=MemoryType.EPISODIC)
    add(store, "cl_goal_b", "Ship V2", predicate="goal", obs=2,
        memory_type=MemoryType.EPISODIC)

    assert consolidator.promote(min_observations=3) == 0

    counts = consolidator.run()
    assert counts["merged"] == 1 and counts["promoted"] == 1
    assert store.get_claim("cl_goal_a").memory_type is MemoryType.SEMANTIC
    assert store.get_claim("cl_goal_a").observation_count == 4
