"""SQLite store: persistence, the indexed conflict lookup, hybrid search primitives,
and the bitemporal SQL that makes time travel work."""

import gc
import os
import pathlib
import re
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from memvara.embed import HashingEmbedder
from memvara.filters import SearchFilter
from memvara.store import (STATES, SQLStore, SQLiteStore, StoreInUseError,
                           live_predicate, state_predicate, stored_state_predicate,
                           unexpired_predicate)
from memvara.store import sqlite as sqlite_store
from memvara.store.base import Store
from memvara.store.sqlite import _WALKABLE as _WALKABLE_SQL
from memvara.store.sqlite import SCHEMA_VERSION, _fts_query
from memvara.types import Claim, Derivation, Episode, MemoryType, Scope

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
TMID = datetime(2024, 6, 1, tzinfo=timezone.utc)
T1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
T2 = datetime(2026, 1, 1, tzinfo=timezone.utc)

SCOPE = Scope("acme", "alice")


@pytest.fixture()
def store() -> SQLiteStore:
    s = SQLiteStore(":memory:")
    yield s
    s.close()


@pytest.fixture()
def emb() -> HashingEmbedder:
    return HashingEmbedder(dim=64)


def claim(**kw) -> Claim:
    base = dict(subject="user", predicate="lives_in", object="Berlin", scope=SCOPE,
                recorded_at=T0, valid_from=T0)
    base.update(kw)
    return Claim(**base)


def put(store, emb=None, **kw) -> Claim:
    c = claim(**kw)
    store.put_claim(c)
    if emb is not None:
        store.set_embedding(c.id, emb.encode([c.text])[0])
    return c


def turn(store, emb=None, content="hello", scope=SCOPE, **kw) -> Episode:
    """Store an episode, embedding it unless `emb` is None."""
    ep = Episode(content=content, scope=scope, **kw)
    store.add_episode(ep)
    if emb is not None:
        store.set_episode_embedding(ep.id, emb.encode([ep.content])[0])
    return ep


# --- Episodes ---------------------------------------------------------------

def test_episode_round_trips_with_every_field(store):
    ep = Episode(content="I live in Berlin", scope=SCOPE, role="user", ts=T0,
                 meta={"turn": 4, "nested": {"a": [1, 2]}})
    store.add_episode(ep)
    got = store.get_episode(ep.id)
    assert got.content == ep.content
    assert got.scope == SCOPE
    assert got.role == "user"
    assert got.ts == T0
    assert got.meta == {"turn": 4, "nested": {"a": [1, 2]}}


def test_missing_episode_returns_none(store):
    assert store.get_episode("ep_nonexistent") is None


def test_episode_lookup_by_hash_enables_write_dedupe(store):
    ep = Episode(content="hello", scope=SCOPE)
    store.add_episode(ep)
    assert store.find_episode_by_hash("acme", ep.hash).id == ep.id
    assert store.find_episode_by_hash("acme", "nope") is None


def test_episode_hash_lookup_is_tenant_scoped(store):
    ep = Episode(content="hello", scope=SCOPE)
    store.add_episode(ep)
    assert store.find_episode_by_hash("other_tenant", ep.hash) is None


# --- Claims -----------------------------------------------------------------

def test_claim_round_trips_with_every_field(store):
    c = claim(object="Lisbon", polarity=-1, memory_type=MemoryType.PROCEDURAL,
              valid_to=T2, invalidated_at=T1, invalidated_by="cl_other",
              confidence=0.42, salience=2.5, observation_count=7,
              sources=["ep_a", "ep_b"], derivation=Derivation.CONSOLIDATION,
              extractor="test/v1", meta={"k": "v"})
    store.put_claim(c)
    g = store.get_claim(c.id)
    for f in ("subject", "predicate", "object", "text", "polarity", "memory_type",
              "valid_from", "valid_to", "recorded_at", "invalidated_at", "invalidated_by",
              "confidence", "salience", "observation_count", "sources", "derivation",
              "extractor", "meta"):
        assert getattr(g, f) == getattr(c, f), f
    assert g.scope == SCOPE


def test_missing_claim_returns_none(store):
    assert store.get_claim("cl_nope") is None


def test_put_claim_is_idempotent_upsert(store):
    c = put(store)
    c.object = "Lisbon"
    c.text = "user lives in Lisbon"
    store.put_claim(c)
    assert store.stats()["claims"] == 1
    assert store.get_claim(c.id).object == "Lisbon"


# --- The indexed conflict lookup -------------------------------------------

def test_competing_claims_finds_all_values_in_one_slot(store):
    a = put(store, object="Berlin")
    b = put(store, object="Lisbon")
    put(store, predicate="works_at", object="Acme")
    ids = {c.id for c in store.competing_claims("acme", a.fact_key)}
    assert ids == {a.id, b.id}


def test_competing_claims_excludes_invalidated(store):
    a = put(store, object="Berlin")
    b = put(store, object="Lisbon")
    store.invalidate(a.id, T1, b.id)
    assert [c.id for c in store.competing_claims("acme", a.fact_key)] == [b.id]


def test_competing_claims_does_not_cross_users(store):
    """The conflict lookup must never surface another person's fact."""
    a = put(store, scope=Scope("acme", "alice"))
    b = put(store, scope=Scope("acme", "bob"))
    assert [c.id for c in store.competing_claims("acme", a.fact_key)] == [a.id]
    assert [c.id for c in store.competing_claims("acme", b.fact_key)] == [b.id]


def test_competing_claims_respects_the_belief_clock(store):
    """The conflict lookup is a liveness query like any other: a claim retired before
    `known_at` is not competing for the slot at `known_at`."""
    a = put(store, object="Berlin")
    store.invalidate(a.id, T1, "cl_x")
    assert [c.id for c in store.competing_claims(
        "acme", a.fact_key, valid_at=TMID, known_at=TMID)] == [a.id]
    assert store.competing_claims("acme", a.fact_key, valid_at=T2, known_at=T2) == []


def test_unended_claims_returns_the_values_in_force_and_those_stored_to_begin_later(store):
    """What `Memvara.forget` closes, selected by the store: every claim in the slot that
    is believed at `known_at` and has not ended by `valid_at`. A value stored to begin
    later is among them. An ended, a retired and an expired value are not, and neither is
    another slot's. They come back oldest first, as `slot_history` returns them."""
    live = put(store, object="Berlin")
    later = put(store, object="Paris", valid_from=T2, recorded_at=T1)
    ended = put(store, object="Porto")
    store.set_valid_to(ended.id, TMID)
    retired = put(store, object="Rome")
    store.invalidate(retired.id, TMID, live.id)
    put(store, object="Oslo", expires_at=TMID)
    put(store, predicate="works_at", object="Acme")

    assert [c.id for c in store.unended_claims(
        "acme", live.fact_key, valid_at=T1, known_at=T1)] == [live.id, later.id]
    # Each clock on its own. In the world at T0 the ended value had not ended yet, so it
    # counts. As believed at TMID the retired value was already retired, and `later` had
    # not been recorded, so neither counts.
    assert {c.id for c in store.unended_claims(
        "acme", live.fact_key, valid_at=T0, known_at=TMID)} == {live.id, ended.id}


def test_count_competing_answers_exactly_what_competing_claims_would(store):
    """One number, one query, and it must not become a second definition of "live".

    The write path asks this on every write to an undeclared predicate, so a count that
    drifted from `competing_claims` would report a slot as crowded that reconciliation
    reads as empty, or the reverse — and nothing downstream could tell. Both are built
    from `_live_clause` for that reason; this asserts they agree across every axis the
    clause has: an ordinary slot, a retirement, an ending, another user's slot of the
    same name, and both clocks read in the past.
    """
    a = put(store, object="Berlin")
    put(store, object="Lisbon")
    put(store, predicate="works_at", object="Acme")
    other = put(store, scope=Scope("acme", "bob"))
    ended = put(store, object="Porto")
    store.set_valid_to(ended.id, T1)
    retired = put(store, object="Rome")
    store.invalidate(retired.id, T1, a.id)

    for tenant, key, axes in (
            ("acme", a.fact_key, {}),
            ("acme", other.fact_key, {}),
            ("acme", a.fact_key, {"valid_at": TMID, "known_at": TMID}),
            ("acme", a.fact_key, {"valid_at": T2, "known_at": T2}),
            ("acme", "no_such_slot", {}),
    ):
        assert (store.count_competing(tenant, key, **axes)
                == len(store.competing_claims(tenant, key, **axes))), (key, axes)
    # …and the numbers are not all trivially equal, or the agreement above proves nothing.
    assert store.count_competing("acme", a.fact_key) == 2
    assert store.count_competing("acme", other.fact_key) == 1
    assert store.count_competing("acme", "no_such_slot") == 0


def test_find_by_value_matches_exact_assertions_only(store):
    a = put(store, object="Berlin")
    put(store, object="Lisbon")
    assert [c.id for c in store.find_by_value("acme", a.value_key)] == [a.id]


# --- Invalidation and reinforcement ----------------------------------------

def test_invalidate_preserves_the_row_for_audit(store):
    a = put(store)
    store.invalidate(a.id, T1, "cl_new")
    g = store.get_claim(a.id)
    assert g is not None, "invalidation must never delete"
    assert g.invalidated_at == T1
    assert g.invalidated_by == "cl_new"


def test_set_valid_to_marks_end_of_world_validity(store):
    a = put(store)
    store.set_valid_to(a.id, T2)
    assert store.get_claim(a.id).valid_to == T2
    store.set_valid_to(a.id, None)
    assert store.get_claim(a.id).valid_to is None


def test_reinforce_merges_sources_without_duplicating(store):
    a = put(store, sources=["ep_1"])
    store.reinforce(a.id, salience=1.5, observation_count=2, sources=["ep_1", "ep_2"])
    g = store.get_claim(a.id)
    assert g.salience == 1.5
    assert g.observation_count == 2
    assert g.sources == ["ep_1", "ep_2"]


def test_reinforce_on_missing_claim_is_a_noop(store):
    store.reinforce("cl_nope", 1.0, 1, ["ep"])


# --- Lexical (BM25) search --------------------------------------------------

def test_lexical_search_finds_exact_tokens(store):
    c = put(store, object="Berlin")
    hits = store.lexical_search("Berlin", [SCOPE], limit=10)
    assert [h[0] for h in hits] == [c.id]


def test_lexical_search_is_case_insensitive(store):
    c = put(store, object="Berlin")
    assert store.lexical_search("berlin", [SCOPE], limit=10)[0][0] == c.id


def test_lexical_scores_are_ascending_better(store):
    put(store, object="Berlin")
    put(store, object="Berlin Berlin Berlin", predicate="likes")
    hits = store.lexical_search("berlin", [SCOPE], limit=10)
    assert len(hits) == 2
    assert hits[0][1] >= hits[1][1], "results must be ordered best-first"


def test_lexical_search_reflects_updated_text(store):
    c = put(store, object="Berlin")
    c.object, c.text = "Lisbon", "user lives in Lisbon"
    store.put_claim(c)
    assert store.lexical_search("berlin", [SCOPE], limit=10) == []
    assert store.lexical_search("lisbon", [SCOPE], limit=10)[0][0] == c.id


@pytest.mark.parametrize(
    "q",
    ["", "   ", "*", '"', '"""', "a AND (b", "NEAR/", "^", "()", "AND OR NOT",
     "-", "a*b", "x" * 10_000, "日本語", "'; DROP TABLE claims; --"],
)
def test_adversarial_queries_never_raise(store, q):
    """FTS5 treats much of this as query syntax, so an unescaped user string is both a
    crash and an injection surface."""
    assert isinstance(store.lexical_search(q, [SCOPE], limit=5), list)
    assert store.stats()["claims"] == 0 or True


def test_injection_attempt_does_not_drop_the_table(store):
    put(store)
    store.lexical_search("'; DROP TABLE claims; --", [SCOPE], limit=5)
    assert store.stats()["claims"] == 1


def test_lexical_search_respects_scope(store):
    put(store, scope=Scope("acme", "alice"))
    assert store.lexical_search("berlin", [Scope("acme", "bob")], limit=10) == []


def test_lexical_search_respects_the_time_axes(store):
    a = put(store, object="Berlin")
    store.invalidate(a.id, T1, "cl_x")
    assert store.lexical_search(
        "berlin", [SCOPE], limit=10, valid_at=TMID, known_at=TMID)[0][0] == a.id
    assert store.lexical_search(
        "berlin", [SCOPE], limit=10, valid_at=T2, known_at=T2) == []


def test_include_invalidated_reveals_retired_claims(store):
    a = put(store, object="Berlin")
    store.invalidate(a.id, T1, "cl_x")
    assert store.lexical_search("berlin", [SCOPE], limit=10) == []
    assert store.lexical_search(
        "berlin", [SCOPE], limit=10, include_invalidated=True
    )[0][0] == a.id


def _churn(store) -> None:
    """Every write that rewrites, frees or reuses a rowid, on both indexed tables."""
    claims = [put(store, predicate=f"p{i}", object=f"kayak trip {'river ' * (i % 4)}{i}")
              for i in range(12)]
    claims[3].object, claims[3].text = "canoe", "user p3 canoe"
    store.put_claim(claims[3])                        # text rewritten in place
    store.put_claim(claims[4])                        # unchanged, index left alone
    store.reinforce(claims[5].id, salience=1.5, observation_count=2, sources=["ep_1"])
    store.erase_claim(claims[7].id)                   # a gap in the middle
    store.erase_claim(claims[-1].id)                  # frees the highest rowid,
    put(store, predicate="late", object="kayak trip river late")   # which this reuses
    turns = [turn(store, content=f"kayak trip {'river ' * (i % 4)}{i}") for i in range(12)]
    store.add_episode(turns[4])                       # added again
    turns[5].content = "canoe trip"
    store.add_episode(turns[5])                       # edited
    store.erase_episode(turns[7].id)
    store.erase_episode(turns[-1].id)
    turn(store, content="kayak trip river late")


def test_a_lexical_hit_never_carries_another_rows_score(store):
    """Both lexical legs join the text index to its table on rowid rather than on the id
    the index row stores, because reading that id back out of the index cost more than
    the rest of the query. That is only correct while each index row sits at its own
    row's rowid. If the two ever parted, a claim would come back scored on another
    claim's text, and nothing downstream could tell. So after every write that moves or
    frees a rowid, each leg is checked against the index's own record of which row each
    entry indexes: the same rows, with the same scores."""
    _churn(store)
    query = "kayak river canoe"
    for search, fts, column in ((store.lexical_search, "claims_fts", "claim_id"),
                                (store.lexical_search_episodes, "episodes_fts",
                                 "episode_id")):
        truth = {r[0]: -r[1] for r in store._db.execute(
            f"SELECT {column}, bm25({fts}) FROM {fts} WHERE {fts} MATCH ?",
            (_fts_query(query),))}
        assert len(truth) > 10
        assert dict(search(query, [SCOPE], limit=100)) == pytest.approx(truth)


def _misfiled(path: str) -> dict[str, tuple[int, int]]:
    """Per indexed table: index rows not at the rowid of the row they name, and rows
    with no index row at their rowid."""
    db = sqlite3.connect(path)
    try:
        counts = {}
        for table, fts, column in (("claims", "claims_fts", "claim_id"),
                                   ("episodes", "episodes_fts", "episode_id")):
            astray = db.execute(
                f"SELECT COUNT(*) FROM {fts} f LEFT JOIN {table} t "
                f"ON t.rowid = f.rowid WHERE t.id IS NOT f.{column}").fetchone()[0]
            unindexed = db.execute(
                f"SELECT COUNT(*) FROM {table} t LEFT JOIN {fts} f "
                f"ON f.rowid = t.rowid WHERE f.rowid IS NULL").fetchone()[0]
            counts[table] = (astray, unindexed)
        return counts
    finally:
        db.close()


def test_every_index_row_sits_at_its_own_rows_rowid_in_every_copy(tmp_path):
    """The invariant the join above rests on, checked directly: each index row is at the
    rowid of the row it names, and every row has one. Then again after a `VACUUM`, which
    SQLite documents as free to renumber the rowids of a table without an INTEGER
    PRIMARY KEY, and in a `VACUUM INTO` copy. Both keep them for a table that has an
    index, as these two do, and this test is what says so if that ever changes. Erasure
    finds a row's index entry by the same rowid, so it would break too."""
    path, copy = str(tmp_path / "s.db"), str(tmp_path / "copy.db")
    with SQLiteStore(path) as s:
        _churn(s)
    aligned = {"claims": (0, 0), "episodes": (0, 0)}
    assert _misfiled(path) == aligned
    db = sqlite3.connect(path)
    try:
        db.execute("VACUUM")
        db.execute("VACUUM INTO ?", (copy,))
    finally:
        db.close()
    assert _misfiled(path) == aligned, "after VACUUM"
    assert _misfiled(copy) == aligned, "in a VACUUM INTO copy"


# --- Vector search ----------------------------------------------------------

def test_vector_search_ranks_by_cosine(store, emb):
    berlin = put(store, emb, object="Berlin")
    put(store, emb, predicate="likes", object="scuba diving")
    q = emb.encode(["user lives in Berlin"])[0]
    hits = store.vector_search(q, [SCOPE], limit=5)
    assert hits[0][0] == berlin.id
    assert hits[0][1] > 0.9


def test_vector_search_on_empty_store_returns_empty(store, emb):
    assert store.vector_search(emb.encode(["anything"])[0], [SCOPE], limit=5) == []


def test_vector_search_skips_claims_without_embeddings(store, emb):
    put(store)  # no embedding written
    assert store.vector_search(emb.encode(["berlin"])[0], [SCOPE], limit=5) == []


def test_vector_search_respects_scope(store, emb):
    put(store, emb, scope=Scope("acme", "alice"))
    q = emb.encode(["user lives in Berlin"])[0]
    assert store.vector_search(q, [Scope("acme", "bob")], limit=5) == []


def test_vector_search_respects_the_time_axes(store, emb):
    a = put(store, emb, object="Berlin")
    store.invalidate(a.id, T1, "cl_x")
    q = emb.encode(["user lives in Berlin"])[0]
    assert store.vector_search(
        q, [SCOPE], limit=5, valid_at=TMID, known_at=TMID)[0][0] == a.id
    assert store.vector_search(q, [SCOPE], limit=5, valid_at=T2, known_at=T2) == []


def test_vector_search_limit_is_honored(store, emb):
    for i in range(20):
        put(store, emb, object=f"City{i}", predicate=f"pred_{i}")
    q = emb.encode(["City3"])[0]
    assert len(store.vector_search(q, [SCOPE], limit=5)) == 5


def test_updating_an_embedding_replaces_rather_than_appends(store, emb):
    c = put(store, emb, object="Berlin")
    store.set_embedding(c.id, emb.encode(["totally different text"])[0])
    assert store.stats()["embeddings"] == 1


def test_mismatched_embedding_dim_is_rejected_before_the_write(store, emb):
    c = put(store, emb)
    with pytest.raises(ValueError, match="dim"):
        store.set_embedding(c.id, np.ones(999, dtype=np.float32))
    assert store.stats()["embeddings"] == 1, "rejected vector must not be persisted"


# --- Episode retrieval ------------------------------------------------------
#
# Episodes used to be write-only: stored, counted in the receipt, and reachable only
# through why() on a claim that happened to be extracted from them. Everything below
# is the property that fixes — with the scope isolation asserted in all three
# directions, exactly as it is for claims, because raw turn text is the more sensitive
# of the two payloads.

def test_episode_lexical_search_finds_a_turn_no_claim_was_extracted_from(store):
    ep = turn(store, content="We decided at the offsite to sunset the Kafka pipeline "
                             "because the ordering guarantees never held.")
    assert store.stats()["claims"] == 0
    hits = store.lexical_search_episodes("kafka pipeline", [SCOPE], limit=10)
    assert [h[0] for h in hits] == [ep.id]


def test_episode_lexical_search_is_case_insensitive(store):
    ep = turn(store, content="Kafka ordering guarantees")
    assert store.lexical_search_episodes("KAFKA", [SCOPE], limit=10)[0][0] == ep.id


def test_episode_lexical_scores_are_ascending_better(store):
    turn(store, content="kafka")
    turn(store, content="kafka kafka kafka")
    hits = store.lexical_search_episodes("kafka", [SCOPE], limit=10)
    assert len(hits) == 2
    assert hits[0][1] >= hits[1][1]


@pytest.mark.parametrize(
    "q", ["", "   ", "*", '"', "a AND (b", "NEAR/", "()", "-", "x" * 10_000,
          "日本語", "'; DROP TABLE episodes; --"],
)
def test_adversarial_episode_queries_never_raise(store, q):
    turn(store)
    assert isinstance(store.lexical_search_episodes(q, [SCOPE], limit=5), list)
    assert store.stats()["episodes"] == 1


def test_readding_a_turn_does_not_orphan_its_index_entry(store):
    """INSERT OR REPLACE assigns a new rowid, and the FTS row is keyed on the old one —
    the text would stay searchable under a rowid nothing points at, and purge would
    never find it."""
    ep = Episode(content="kafka ordering", scope=SCOPE)
    store.add_episode(ep)
    store.add_episode(ep)
    hits = store.lexical_search_episodes("kafka", [SCOPE], limit=10)
    assert [h[0] for h in hits] == [ep.id], "one entry, still resolvable"


def test_episode_lexical_search_reflects_edited_text(store):
    ep = Episode(content="kafka ordering", scope=SCOPE)
    store.add_episode(ep)
    ep.content = "kinesis ordering"
    store.add_episode(ep)
    assert store.lexical_search_episodes("kafka", [SCOPE], limit=10) == []
    assert store.lexical_search_episodes("kinesis", [SCOPE], limit=10)[0][0] == ep.id


def test_episode_vector_search_ranks_by_cosine(store, emb):
    kafka = turn(store, emb, content="the kafka pipeline is being sunset")
    turn(store, emb, content="lunch is at one o'clock")
    hits = store.vector_search_episodes(
        emb.encode(["the kafka pipeline is being sunset"])[0], [SCOPE], limit=5)
    assert hits[0][0] == kafka.id
    assert hits[0][1] > 0.9


def test_episode_vector_search_skips_turns_without_vectors(store, emb):
    turn(store)  # never embedded
    assert store.vector_search_episodes(emb.encode(["hello"])[0], [SCOPE], limit=5) == []


def test_episode_and_claim_vectors_never_share_a_row(store, emb):
    """One matrix, one slot space. Two allocators each computing 'one past my own
    maximum' would hand row 0 to both, and each would read back the other's vector."""
    c = put(store, emb, object="Berlin")
    ep = turn(store, emb, content="a completely unrelated sentence about otters")

    claim_slots = {r[0] for r in store._db.execute("SELECT slot FROM embeddings")}
    ep_slots = {r[0] for r in store._db.execute("SELECT slot FROM episode_embeddings")}
    assert claim_slots.isdisjoint(ep_slots)

    assert np.allclose(store.get_embedding(c.id), store._vec.get(c.id))
    assert np.allclose(store.get_episode_embedding(ep.id), store._vec.get(ep.id))


def test_a_turn_that_was_never_embedded_has_no_vector(store):
    assert store.get_episode_embedding(turn(store).id) is None


def test_mismatched_episode_embedding_dim_is_rejected_before_the_write(store, emb):
    put(store, emb)
    ep = turn(store)
    with pytest.raises(ValueError, match="dim"):
        store.set_episode_embedding(ep.id, np.ones(999, dtype=np.float32))
    assert store.stats()["embeddings"] == 1, "rejected vector must not be persisted"


def test_get_episodes_bulk_fetches_and_ignores_unknown_ids(store):
    a, b = turn(store, content="one"), turn(store, content="two")
    got = store.get_episodes([a.id, b.id, "ep_nope", a.id])
    assert set(got) == {a.id, b.id}
    assert got[a.id].content == "one"
    assert store.get_episodes([]) == {}


def test_get_episodes_chunks_past_the_sql_parameter_limit(store):
    ids = [turn(store, content=f"turn {i}").id for i in range(950)]
    assert len(store.get_episodes(ids)) == 950


def test_iter_episodes_filters_by_tenant(store):
    turn(store, scope=Scope("acme", "alice"))
    turn(store, scope=Scope("other", "bob"), content="elsewhere")
    assert len(list(store.iter_episodes(tenant="acme"))) == 1
    assert len(list(store.iter_episodes())) == 2


# --- Episode scope isolation, all three directions --------------------------

SIBLING_SESSION = Scope("acme", "alice", "bot", "s2")
SIBLING_AGENT = Scope("acme", "alice", "other_bot")
OTHER_TENANT = Scope("globex", "alice")
MINE = Scope("acme", "alice", "bot", "s1")


@pytest.fixture()
def neighbours(store, emb) -> dict[str, Episode]:
    """The same sentence stored at four scopes. Identical text on either side is what
    would slip through if the filter were on content rather than on scope."""
    text = "the kafka pipeline is being sunset"
    return {
        "mine": turn(store, emb, content=text, scope=MINE),
        "sibling_session": turn(store, emb, content=text, scope=SIBLING_SESSION),
        "sibling_agent": turn(store, emb, content=text, scope=SIBLING_AGENT),
        "other_tenant": turn(store, emb, content=text, scope=OTHER_TENANT),
    }


@pytest.mark.parametrize("neighbour", ["sibling_session", "sibling_agent", "other_tenant"])
def test_episode_search_never_reaches_sideways(store, emb, neighbours, neighbour):
    q = emb.encode(["kafka pipeline"])[0]
    for found in (
        set(store.episode_candidate_ids([MINE])),
        {h[0] for h in store.lexical_search_episodes("kafka", [MINE], limit=10)},
        {h[0] for h in store.vector_search_episodes(q, [MINE], limit=10)},
    ):
        assert found == {neighbours["mine"].id}
        assert neighbours[neighbour].id not in found


def test_episode_search_fails_closed_on_an_empty_scope_list(store, emb):
    """Same rule as claims: no scope resolved is a caller bug, and matching everything
    would hand back every tenant's transcript."""
    turn(store, emb, content="kafka")
    assert store.episode_candidate_ids([]) == []
    assert store.lexical_search_episodes("kafka", [], limit=10) == []
    assert store.vector_search_episodes(emb.encode(["kafka"])[0], [], limit=10) == []


def test_episode_search_inherits_upward_but_never_descends(store, emb):
    broad = turn(store, emb, content="kafka at user scope", scope=Scope("acme", "alice"))
    narrow = turn(store, emb, content="kafka in this session", scope=MINE)
    from_session = set(store.episode_candidate_ids(MINE.ancestors()))
    assert from_session == {broad.id, narrow.id}
    assert set(store.episode_candidate_ids([Scope("acme", "alice")])) == {broad.id}


def test_episode_search_respects_the_time_axes(store, emb):
    """A turn that had not happened yet is not something we could have recalled."""
    old = turn(store, emb, content="kafka is fine", ts=T0)
    new = turn(store, emb, content="kafka is being sunset", ts=T1)
    q = emb.encode(["kafka"])[0]
    assert set(store.episode_candidate_ids(
        [SCOPE], valid_at=TMID, known_at=TMID)) == {old.id}
    assert [h[0] for h in store.lexical_search_episodes(
        "kafka", [SCOPE], limit=10, valid_at=TMID, known_at=TMID)] == [old.id]
    assert [h[0] for h in store.vector_search_episodes(
        q, [SCOPE], limit=10, valid_at=TMID, known_at=TMID)] == [old.id]
    later = store.episode_candidate_ids([SCOPE], valid_at=T2, known_at=T2)
    assert len(later) == 2 and new.id in later


def test_an_episode_is_bounded_by_the_earlier_of_the_two_clocks(store):
    """A turn's single `ts` is both of its clocks at once — it happened and we knew of
    it at the same instant — so either axis alone can hide it. Without the `min` in
    `_happened_clause`, `valid_at=T0, known_at=T2` would return a turn that had not
    happened yet, which is a search result from the future."""
    turn(store, content="kafka is fine", ts=T1)
    assert store.episode_candidate_ids([SCOPE], valid_at=T0, known_at=T2) == []
    assert store.episode_candidate_ids([SCOPE], valid_at=T2, known_at=T0) == []
    assert len(store.episode_candidate_ids([SCOPE], valid_at=T2, known_at=T2)) == 1


def test_a_future_turn_is_invisible_to_a_present_query(store):
    turn(store, content="kafka next year", ts=datetime(2099, 1, 1, tzinfo=timezone.utc))
    assert store.lexical_search_episodes("kafka", [SCOPE], limit=10) == []


def test_the_turn_candidate_list_is_one_covering_index_range_per_scope(store):
    """Every turn a scope can see is the vector leg's candidate list. `ep_cover` holds
    every column that query reads, so no turn's row is read: 244 ms to 102 ms for the
    189,520 LongMemEval-S turns in one scope. Asking once per scope, rather than through
    the `OR` the capped reads use, also drops the temporary set SQLite keeps to return a
    row two terms of an `OR` both match only once: 102 ms to 84 ms."""
    turn(store)
    statements: list[str] = []
    store._db.set_trace_callback(statements.append)
    try:
        store.episode_candidate_ids(MINE.ancestors())
    finally:
        store._db.set_trace_callback(None)
    (sql,) = [s for s in statements if s.startswith("SELECT id FROM episodes")]
    # The trace fills in the bound values where SQLite can expand them. A marker left
    # over is bound to NULL, which SQLite plans as the same `IS` lookup.
    plan = [r[3] for r in store._db.execute("EXPLAIN QUERY PLAN " + sql,
                                            [None] * sql.count("?"))]
    reads = [step for step in plan if step.startswith(("SCAN", "SEARCH"))]
    assert not any("MULTI-INDEX OR" in step for step in plan), plan
    assert len(reads) == len(MINE.ancestors()), plan
    assert all("COVERING INDEX ep_cover" in step for step in reads), plan


def test_a_store_written_before_the_covering_index_gains_it_when_opened(tmp_path):
    """`ep_cover` is created with the other late indexes on every open, so a store an
    older build wrote gets it the first time this one opens it (1.7 s and 10 MB for
    190,000 turns, once), not only on a schema migration it may never have. `ep_scope`
    stays, because the older build creates it again on every open."""
    path = str(tmp_path / "s.db")
    with SQLiteStore(path) as s:
        turn(s)
        s._db.execute("DROP INDEX ep_cover")
        s._db.commit()
    with SQLiteStore(path) as s:
        names = {r[0] for r in s._db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'")}
    assert {"ep_cover", "ep_scope"} <= names


# --- Scoped episode listing --------------------------------------------------
#
# `iter_episodes` was the only listing, so a caller wanting one scope's turns walked the
# tenant and filtered in Python — which an adapter exposing a session transcript does on
# every read. These pin the ordering, the cap, and the isolation, in that order of how
# easy they are to get quietly wrong.

def test_scope_episodes_returns_only_the_scopes_asked_for(store):
    """Not the caller's siblings and not their neighbours. Raw turn text is the more
    sensitive of the two things this store holds, so a listing that leaked sideways
    would hand one session's transcript to another."""
    mine = turn(store, content="mine", scope=MINE)
    turn(store, content="sibling session", scope=SIBLING_SESSION)
    turn(store, content="sibling agent", scope=SIBLING_AGENT)
    turn(store, content="other tenant", scope=OTHER_TENANT)

    assert [e.id for e in store.scope_episodes([MINE])] == [mine.id]


def test_scope_episodes_does_not_descend_into_narrower_scopes(store):
    """The hierarchy widens upward and this listing does not walk it in either
    direction: it matches what it is given. A caller wanting the inherited view passes
    `ancestors()`, which is the same thing retrieval passes, rather than hoping this
    method guesses which direction it meant."""
    user_level = turn(store, content="written by a background job", scope=Scope("acme", "alice"))
    session = turn(store, content="said in the session", scope=MINE)

    assert [e.id for e in store.scope_episodes([Scope("acme", "alice")])] == [user_level.id]
    assert {e.id for e in store.scope_episodes(MINE.ancestors())} == {user_level.id, session.id}


def test_scope_episodes_fails_closed_on_an_empty_scope_list(store):
    """The same fail-closed as `candidate_ids`, and it matters more here: an empty scope
    list means no scope was resolved, and the generous reading of that hands back every
    turn in the tenant."""
    turn(store, content="hello")
    assert store.scope_episodes([]) == []


def test_scope_episodes_orders_by_ts_and_can_take_the_newest_end(store):
    """`newest_first` flips which end `limit` takes from, and that is the whole point of
    it: a caller filling a context window wants the last N turns, and an ascending order
    with a cap hands back the first N and drops everything recent."""
    first = turn(store, content="first", ts=T0)
    middle = turn(store, content="middle", ts=TMID)
    last = turn(store, content="last", ts=T1)

    assert [e.id for e in store.scope_episodes([SCOPE])] == [first.id, middle.id, last.id]
    assert [e.id for e in store.scope_episodes([SCOPE], newest_first=True)] \
        == [last.id, middle.id, first.id]
    assert [e.id for e in store.scope_episodes([SCOPE], limit=2)] == [first.id, middle.id]
    assert [e.id for e in store.scope_episodes([SCOPE], limit=2, newest_first=True)] \
        == [last.id, middle.id]


def test_scope_episodes_breaks_ts_ties_on_insertion_order(store):
    """Callers doing this in Python got insertion order free from a stable sort. Left to
    the query planner, two turns the clock could not separate come back in whatever order
    the pages happen to be in — and differently on different files."""
    ids = [turn(store, content=f"turn {i}", ts=T0).id for i in range(5)]
    assert [e.id for e in store.scope_episodes([SCOPE])] == ids
    assert [e.id for e in store.scope_episodes([SCOPE], newest_first=True)] == ids[::-1]
    assert [e.id for e in store.scope_episodes([SCOPE], limit=2, newest_first=True)] \
        == ids[:-3:-1], "and the cap takes the newest of the tied turns, not the first"


def test_scope_episodes_reads_a_negative_limit_as_none_rather_than_all(store):
    """SQLite reads a negative LIMIT as *no* limit, so a cap that came out of an
    arithmetic slip would return the whole scope instead of nothing. Fail closed, like
    the scope clause above it."""
    turn(store, content="hello")
    assert store.scope_episodes([SCOPE], limit=-1) == []
    assert store.scope_episodes([SCOPE], limit=0) == []


# --- Scope resolution -------------------------------------------------------

def test_candidate_ids_matches_scopes_exactly(store):
    a = put(store, scope=Scope("acme", "alice"))
    b = put(store, scope=Scope("acme", "alice", "bot", "s1"), predicate="likes")
    assert set(store.candidate_ids([Scope("acme", "alice")])) == {a.id}
    assert set(store.candidate_ids([Scope("acme", "alice"), Scope("acme", "alice", "bot", "s1")])) == {a.id, b.id}


def test_the_candidate_lists_return_a_row_exactly_when_its_scope_is_asked_for(store):
    """The candidate lists build one `SELECT` per scope instead of going through
    `_scope_clause`, so they are checked against what that clause means, over every
    shape a scope can take: each level below the tenant set or unset, in two tenants,
    with a claim and a turn at each, asked from each shape's own ancestors. An unset
    level is NULL in the table, so this is also where `=` written for `IS` would show,
    as rows that never come back."""
    shapes = [Scope(t, u, a, s, project=p)
              for t in ("acme", "globex") for u in (None, "alice")
              for p in (None, "gh/o/a") for a in (None, "bot") for s in (None, "s1")]
    home = {}
    for shape in shapes:
        home[put(store, scope=shape).id] = shape
        home[turn(store, scope=shape).id] = shape
    for shape in shapes:
        asked = shape.ancestors()
        got = set(store.candidate_ids(asked)) | set(store.episode_candidate_ids(asked))
        assert got == {i for i, at in home.items() if at in asked}, shape


def test_a_scope_listed_twice_returns_its_rows_once(store):
    """One `SELECT` per scope is joined by `UNION ALL`, where the scopes used to be
    `OR`ed together. An `OR` returns a row once however many of its terms match it; two
    identical branches of a `UNION ALL` return it twice, and the vector leg would rank
    it twice. So a repeated scope is dropped before the SQL is built, including one
    that is equal to another without being the same object."""
    c = put(store)
    ep = turn(store)
    assert store.candidate_ids([SCOPE, Scope("acme", "alice")]) == [c.id]
    assert store.episode_candidate_ids([SCOPE, Scope("acme", "alice")]) == [ep.id]


def test_reads_run_on_two_threads_only_where_each_thread_has_its_own_connection(tmp_path):
    """Another thread's read sees committed rows through its own connection. Inside
    `batch()` this thread's rows are not committed yet, and a database with no file has
    one connection, which a second thread would wait on for the whole batch."""
    file = SQLiteStore(str(tmp_path / "p.db"))
    assert file._parallel_reads()
    with file.batch():
        assert not file._parallel_reads()
    assert file._parallel_reads()
    assert not SQLiteStore(":memory:")._parallel_reads()
    file.close()


# --- The lexical leg over turns, ranked by the text index first ------------------------

def _full_lexical(store, monkeypatch, *args, **kw):
    """What `lexical_search_episodes` returned before it ranked the text index first: the
    full query, which joins every match to its turn."""
    with monkeypatch.context() as m:
        m.setattr(store, "_episode_text_first", lambda *a, **k: None)
        return store.lexical_search_episodes(*args, **kw)


def _recording(store, monkeypatch):
    """A list that every answer `_episode_text_first` gives is appended to, None for a
    search it left to the full query."""
    answers = []
    real = store._episode_text_first

    def record(*args, **kw):
        answers.append(real(*args, **kw))
        return answers[-1]

    monkeypatch.setattr(store, "_episode_text_first", record)
    return answers


_WORDS = ("kafka", "pipeline", "lunch", "berlin", "deploy", "ordering", "otter", "sunset")


def test_the_text_first_lexical_leg_returns_what_the_full_query_returns(store, monkeypatch):
    """The text-first form stands in for the full query only where it can prove the
    answer, so it is checked against the full query: the same turns, in the same order,
    with the same scores, for every shape of scope asked from its own ancestors, at
    instants before, among and after the turns, and at every limit, none included. Every
    third turn repeats one text, so many scores tie and the tie-break has to agree too.

    The sweep means something only if it reaches all three outcomes: a ranking that held
    every match, a ranking cut short that still proved its answer, and one that could not.
    A ranking that held every match must never fail to answer."""
    shapes = [Scope("acme", u, a, s) for u in (None, "alice") for a in (None, "bot")
              for s in (None, "s1")]
    for i in range(300):
        words = " ".join(_WORDS[(i + j) % len(_WORDS)] for j in range(1 + i % 5))
        turn(store, scope=shapes[i % len(shapes)],
             content="kafka pipeline" if i % 3 == 0 else words,
             ts=datetime(2024, 1 + i % 12, 1 + i % 28, tzinfo=timezone.utc))
    queries = ("kafka", "kafka pipeline", "lunch berlin", "deploy ordering", "zebra")
    matches = {q: store._db.execute(
        "SELECT count(*) FROM episodes_fts WHERE episodes_fts MATCH ?",
        (_fts_query(q),)).fetchone()[0] for q in queries}
    instants = [{}, {"valid_at": T0, "known_at": T0}, {"valid_at": TMID, "known_at": TMID},
                {"valid_at": T1, "known_at": TMID}, {"valid_at": TMID, "known_at": T1}]
    answers = _recording(store, monkeypatch)
    outcomes = set()
    for shape in shapes:
        asked = shape.ancestors()
        for at in instants:
            for q in queries:
                for limit in (1, 5, 30, 1000, 0, -1):
                    store._text_first_skips.clear()
                    before = len(answers)
                    got = store.lexical_search_episodes(q, asked, limit, **at)
                    want = _full_lexical(store, monkeypatch, q, asked, limit, **at)
                    assert got == want, (shape, at, q, limit)
                    if len(answers) > before:
                        top = max(limit * sqlite_store._TEXT_FIRST_WIDEN,
                                  sqlite_store._TEXT_FIRST_FLOOR)
                        outcomes.add((matches[q] >= top, answers[-1] is not None))
    assert outcomes == {(False, True), (True, True), (True, False)}


def test_a_tie_at_the_edge_of_the_ranked_rows_sends_the_leg_to_the_full_query(
        store, monkeypatch):
    """Turns with one text in one scope score alike, so the ranking is cut inside a tie
    and nothing scores better than its worst row. Which of the tied turns made the cut
    was SQLite's choice rather than the tie-break's, so the leg cannot prove its answer
    from them and asks the full query, which orders a tie by `hash`, then `id`."""
    eps = [turn(store, content="kafka pipeline") for _ in range(300)]
    answers = _recording(store, monkeypatch)
    got = store.lexical_search_episodes("kafka", [SCOPE], 10)
    assert answers == [None]
    assert [h[0] for h in got] == sorted(ep.id for ep in eps)[:10]


def test_the_edge_is_the_worst_ranked_match_whoever_it_belongs_to(store, monkeypatch):
    """The ranked rows prove whatever scores better than the worst of them, and that row
    need not be one the search may see. This scope's five turns are the best five matches
    and another user's turns fill the rest of the ranking, so all five are proved, the
    fifth included, although it is the worst of this scope's."""
    mine = [turn(store, content="kafka" + " word" * i).id for i in range(5)]
    for i in range(150):
        turn(store, content="kafka" + " word" * (5 + i), scope=Scope("acme", "bob"))
    answers = _recording(store, monkeypatch)
    hits = store.lexical_search_episodes("kafka", [SCOPE], 5)
    assert [h[0] for h in hits] == mine
    assert answers == [hits]


def test_a_search_matching_nothing_it_may_see_is_answered_by_the_ranking(
        store, monkeypatch):
    """Every match was ranked and none is in the scopes asked, so the answer is proved
    empty: no full query, and no backing off. A word no turn contains is the same case."""
    turn(store, content="kafka", scope=Scope("acme", "bob"))
    answers = _recording(store, monkeypatch)
    assert store.lexical_search_episodes("kafka", [SCOPE], 5) == []
    assert store.lexical_search_episodes("zebra", [SCOPE], 5) == []
    assert answers == [[], []]
    assert store._text_first_skips == {}


def test_a_scope_holding_few_of_the_matches_backs_off_the_text_first_form(
        store, monkeypatch):
    """Another user's short turns take every ranked row, so the ranking holds none of
    this scope's turns and proves nothing. That search and the next `_TEXT_FIRST_BACKOFF`
    go to the full query, and the one after tries the text-first form again."""
    for _ in range(200):
        turn(store, content="kafka", scope=Scope("acme", "bob"))
    wanted = [turn(store, content="kafka pipeline" + " word" * i).id for i in range(3)]
    answers = _recording(store, monkeypatch)
    left = []
    for _ in range(sqlite_store._TEXT_FIRST_BACKOFF + 2):
        assert [h[0] for h in store.lexical_search_episodes("kafka", [SCOPE], 5)] == wanted
        left.append(store._text_first_skips.get(((SCOPE,), False), 0))
    backoff = sqlite_store._TEXT_FIRST_BACKOFF
    assert left == list(range(backoff, 0, -1)) + [0, backoff]
    assert answers == [None] * (backoff + 2)


def test_a_miss_reading_the_past_leaves_reads_of_the_present_alone(store, monkeypatch):
    """Few turns had happened by an early instant, so a read pinned there often misses.
    Its misses back off only reads pinned to an instant: a read of the present, which
    usually proves its answer, is not sent to the full query by them."""
    eps = [turn(store, content="kafka" + " word" * i, ts=T0 if i == 149 else TMID)
           for i in range(150)]
    answers = _recording(store, monkeypatch)
    past = store.lexical_search_episodes("kafka", [SCOPE], 5, valid_at=T0, known_at=T0)
    assert [h[0] for h in past] == [eps[149].id]
    now = store.lexical_search_episodes("kafka", [SCOPE], 5)
    assert [h[0] for h in now] == [ep.id for ep in eps[:5]]
    assert answers == [None, now]
    assert store._text_first_skips == {((SCOPE,), True): sqlite_store._TEXT_FIRST_BACKOFF}


def test_the_backoff_forgets_every_list_once_it_holds_too_many(store, monkeypatch):
    """A host serving many scopes must not grow the counts without bound. A miss that
    finds `_TEXT_FIRST_SKIPS_KEPT` lists already counted forgets them and keeps its own."""
    monkeypatch.setattr(sqlite_store, "_TEXT_FIRST_SKIPS_KEPT", 3)
    for _ in range(200):
        turn(store, content="kafka", scope=Scope("acme", "bob"))
    for user in ("u1", "u2", "u3", "u4"):
        turn(store, content="kafka pipeline", scope=Scope("acme", user))
        store.lexical_search_episodes("kafka", [Scope("acme", user)], 5)
    assert list(store._text_first_skips) == [((Scope("acme", "u4"),), False)]


def test_the_text_first_leg_looks_up_only_the_ranked_turns(store):
    """The form is worth having only in its join order: rank in the text index, then
    read each ranked turn by rowid. Left to choose, SQLite walked every turn of the scope
    in the covering index and looked each one up in the text index, which is slower than
    the full query. `CROSS JOIN` pins the order, and a new store with no statistics is
    where the planner prefers the other one."""
    sc, sp = store._scope_clause([SCOPE], alias="e")
    hp, hpp = store._happened_clause(None, None, alias="e")
    plan = [r["detail"] for r in store._db.execute(
        "EXPLAIN QUERY PLAN " + sqlite_store._text_first_sql(sc, hp),
        [_fts_query("kafka"), 100] + sp + hpp)]
    # A substring, not the whole step: before 3.36 SQLite wrote the same step as
    # "SEARCH TABLE episodes AS e USING ...", and this package supports 3.35.
    assert any("INTEGER PRIMARY KEY (rowid=?)" in step for step in plan), plan
    assert not any("ep_cover" in step or "ep_scope" in step for step in plan), plan


# --- The vector leg over turns, from each scope's cached list ------------------------

def _sql_turn_search(store, qvec, scopes, limit, **at):
    """What `vector_search_episodes` ranked before each scope's turns were cached: the
    SQL candidate list, handed to the index."""
    allowed = store.episode_candidate_ids(scopes, **at)
    return store._vec.search(qvec, allowed, limit) if allowed else []


def test_the_cached_turn_search_returns_what_the_sql_candidate_list_returns(store):
    """The cache replaces a SQL candidate list, so it is checked against that list: the
    same turns, in the same order, with the same scores, for every shape of scope asked
    from its own ancestors, at instants before, among and after the turns, with the two
    axes apart as well as together, and at limits from one to everything. Half the
    turns share a vector with others, and many share a `ts`, because a tie is where a
    candidate order other than SQL's would put a different turn inside the limit."""
    rng = np.random.default_rng(7)
    shapes = [Scope("acme", u, a, s) for u in (None, "alice") for a in (None, "bot")
              for s in (None, "s1")]
    shared = rng.standard_normal((6, 64)).astype(np.float32)
    for i in range(240):
        ep = turn(store, scope=shapes[i % len(shapes)],
                  ts=datetime(2024, 1 + i % 12, 1 + i % 28, tzinfo=timezone.utc))
        store.set_episode_embedding(
            ep.id, shared[i % 6] if i % 2 else rng.standard_normal(64).astype(np.float32))
    instants = [{}, {"valid_at": T0, "known_at": T0}, {"valid_at": TMID, "known_at": TMID},
                {"valid_at": T1, "known_at": TMID}, {"valid_at": TMID, "known_at": T1}]
    for shape in shapes:
        asked = shape.ancestors()
        for at in instants:
            for limit in (1, 5, 17, 1000):
                q = shared[limit % 6] if limit % 2 else rng.standard_normal(64)
                got = store.vector_search_episodes(q, asked, limit, **at)
                assert got == _sql_turn_search(store, q, asked, limit, **at), (shape, at)
    assert store._turns, "the searches above were answered from the cache"


def test_a_scope_listed_twice_ranks_its_turns_once_from_the_cache(store):
    """The cached search drops a repeated scope as `_scoped_union` does. Two copies of
    one scope's list would put each of its turns in the ranking twice."""
    ep = turn(store)
    store.set_episode_embedding(ep.id, onehot(1))
    twice = [SCOPE, Scope("acme", "alice")]
    assert [h[0] for h in store.vector_search_episodes(onehot(1), twice, 5)] == [ep.id]
    assert store._turns, "the search was answered from the cache"


def test_a_turn_written_after_a_search_is_found_by_the_next_one(store):
    """Every commit empties the cache, so a turn written between two searches is a
    candidate for the second."""
    first = turn(store)
    store.set_episode_embedding(first.id, onehot(1))
    assert [h[0] for h in store.vector_search_episodes(onehot(2), [SCOPE], 5)] == [first.id]
    later = turn(store)
    store.set_episode_embedding(later.id, onehot(2))
    assert store.vector_search_episodes(onehot(2), [SCOPE], 1)[0][0] == later.id


def test_an_erased_turn_is_not_returned_with_the_vector_that_took_its_row(store):
    """Erasing a turn frees its row, and the next vector written takes it. A cache
    that still listed the erased turn at that row would return it, scored by a vector
    written for another scope."""
    gone = turn(store)
    store.set_episode_embedding(gone.id, onehot(1))
    row = store._vec._row[gone.id]
    assert store.vector_search_episodes(onehot(1), [SCOPE], 5)[0][0] == gone.id
    store.erase_episode(gone.id)
    elsewhere = turn(store, scope=Scope("acme", "bob"))
    store.set_episode_embedding(elsewhere.id, onehot(1))
    assert store._vec._row[elsewhere.id] == row
    assert store.vector_search_episodes(onehot(1), [SCOPE], 5) == []


def test_a_warm_cache_sees_what_another_worker_writes_and_erases(tmp_path):
    """Another process's commit reaches this one only as a new `data_version`, which
    the vector leg reads before every search and which empties the cache."""
    path = str(tmp_path / "c.db")
    a, b = SQLiteStore(path), SQLiteStore(path)
    first = turn(a)
    a.set_episode_embedding(first.id, onehot(1))
    assert a.vector_search_episodes(onehot(1), [SCOPE], 5)[0][0] == first.id
    ep = turn(b)
    b.set_episode_embedding(ep.id, onehot(5))
    assert a.vector_search_episodes(onehot(5), [SCOPE], 1)[0][0] == ep.id
    b.erase_episode(first.id)
    assert [h[0] for h in a.vector_search_episodes(onehot(1), [SCOPE], 5)] == [ep.id]
    a.close()
    b.close()


def _from_a_new_thread(read):
    """What `read()` returns when called from a thread that has never read the store."""
    out = []
    t = threading.Thread(target=lambda: out.append(read()))
    t.start()
    t.join()
    return out[0]


def test_a_thread_that_has_never_read_keeps_the_lists_it_finds(tmp_path):
    """Nothing was committed, so there is nothing to rebuild. When each thread asked its
    own connection whether anything had changed, a thread's first read emptied every
    list, and a host that searched from a new thread each time rebuilt them every time."""
    store = SQLiteStore(str(tmp_path / "t.db"))
    ep = turn(store)
    store.set_episode_embedding(ep.id, onehot(1))
    store.vector_search_episodes(onehot(1), [SCOPE], 5)
    built = store._turns[("acme", "alice", None, None, None)]
    hits = _from_a_new_thread(lambda: store.vector_search_episodes(onehot(1), [SCOPE], 5))
    assert [h[0] for h in hits] == [ep.id]
    assert store._turns[("acme", "alice", None, None, None)] is built
    store.close()


def test_a_commit_empties_the_lists_once_whichever_threads_search_after_it(tmp_path):
    """The first search after another connection's commit, on any thread, rebuilds the
    list with the new turn in it, and every later search, on that thread or another,
    reads that list rather than rebuilding it again."""
    path = str(tmp_path / "t.db")
    a, b = SQLiteStore(path), SQLiteStore(path)
    first = turn(a)
    a.set_episode_embedding(first.id, onehot(1))
    a.vector_search_episodes(onehot(1), [SCOPE], 5)
    ep = turn(b)
    b.set_episode_embedding(ep.id, onehot(2))
    assert _from_a_new_thread(
        lambda: a.vector_search_episodes(onehot(2), [SCOPE], 1))[0][0] == ep.id
    rebuilt = a._turns[("acme", "alice", None, None, None)]
    assert _from_a_new_thread(
        lambda: a.vector_search_episodes(onehot(2), [SCOPE], 1))[0][0] == ep.id
    assert a.vector_search_episodes(onehot(2), [SCOPE], 1)[0][0] == ep.id
    assert a._turns[("acme", "alice", None, None, None)] is rebuilt
    a.close()
    b.close()


def test_a_commit_just_after_the_map_is_refreshed_is_seen_by_the_next_search(
        tmp_path, monkeypatch):
    """The watch is read before the map is refreshed. Read after it, the watch could
    report a commit the map has not folded in yet: the list built next would leave out
    that commit's turns, which have no row in the map, and no later look would find
    anything left to rebuild for."""
    path = str(tmp_path / "t.db")
    a, b = SQLiteStore(path), SQLiteStore(path)
    first = turn(a)
    a.set_episode_embedding(first.id, onehot(1))
    a.vector_search_episodes(onehot(1), [SCOPE], 5)
    refresh, late = a._ensure_index, []

    def then_another_worker_commits():
        refresh()
        late.append(turn(b))
        b.set_episode_embedding(late[0].id, onehot(2))

    monkeypatch.setattr(a, "_ensure_index", then_another_worker_commits)
    a.vector_search_episodes(onehot(2), [SCOPE], 5)
    monkeypatch.undo()
    assert a.vector_search_episodes(onehot(2), [SCOPE], 1)[0][0] == late[0].id
    a.close()
    b.close()


def test_a_search_on_a_closed_store_fails_without_opening_a_connection(tmp_path):
    """The connection that watches for commits is opened by the first search that reads
    the lists. After `close()` that search raises, and leaves no connection open."""
    store = SQLiteStore(str(tmp_path / "t.db"))
    ep = turn(store)
    store.set_episode_embedding(ep.id, onehot(1))
    store.close()
    with pytest.raises(sqlite3.ProgrammingError):
        store.vector_search_episodes(onehot(1), [SCOPE], 5)
    assert store._watch is None and store._readers == []


def test_a_search_inside_a_batch_sees_its_own_turns_and_the_cache_keeps_none(store):
    """Inside `batch()` a thread reads its own uncommitted rows. A cache shared with
    other threads must never hold them, and after a rollback nothing may return a turn
    that was never written."""
    first = turn(store)
    store.set_episode_embedding(first.id, onehot(1))
    store.vector_search_episodes(onehot(1), [SCOPE], 5)
    with pytest.raises(RuntimeError):
        with store.batch():
            ep = turn(store)
            store.set_episode_embedding(ep.id, onehot(2))
            assert store.vector_search_episodes(onehot(2), [SCOPE], 1)[0][0] == ep.id
            raise RuntimeError("roll back")
    assert [h[0] for h in store.vector_search_episodes(onehot(2), [SCOPE], 5)] == [first.id]


def test_a_row_moved_after_the_cache_was_read_sends_the_search_to_sql(store):
    """A write in another thread can change the index between a search reading the
    cache and ranking, before its commit empties the cache. Here the index is changed
    directly, as an erasure and a re-embedding change it before they commit: each
    returned turn must still hold the row it was ranked by, and the matrix must still
    reach every row, or the search asks SQL instead."""
    a_ = turn(store)
    store.set_episode_embedding(a_.id, onehot(1))
    b_ = turn(store)
    store.set_episode_embedding(b_.id, onehot(2))
    store.vector_search_episodes(onehot(1), [SCOPE], 5)
    store._vec.forget(a_.id)
    assert [h[0] for h in store.vector_search_episodes(onehot(1), [SCOPE], 5)] == [b_.id]
    store._vec.reset()
    store._vec.put(b_.id, 0, onehot(2))
    assert [h[0] for h in store.vector_search_episodes(onehot(2), [SCOPE], 5)] == [b_.id]


def test_a_turn_list_built_across_a_commit_is_not_kept(store, monkeypatch):
    """`_scope_turns` reads SQL and the map outside every lock, so a commit can land
    between its read and its insert. The list is right for the search that built it,
    which began before the commit, and wrong for any later one."""
    ep = turn(store)
    store.set_episode_embedding(ep.id, onehot(1))
    key = ("acme", "alice", None, None, None)
    real = store._read

    @contextmanager
    def committed_meanwhile():
        with real() as conn:
            yield conn
        store._changed()

    monkeypatch.setattr(store, "_read", committed_meanwhile)
    assert store._scope_turns(key).ids == [ep.id]
    assert key not in store._turns


def test_the_turn_lists_stay_inside_their_row_budget(store, monkeypatch):
    """The least recently used scope goes first once the lists hold more turns than
    the budget, and one scope is kept whatever its size."""
    monkeypatch.setattr(sqlite_store, "_SCOPE_TURNS_ROWS", 5)
    scopes = [Scope("acme", u) for u in ("a", "b", "c")]
    for i, s in enumerate(scopes):
        for _ in range(3 + 2 * (i == 2)):
            ep = turn(store, scope=s)
            store.set_episode_embedding(ep.id, onehot(i))
    for s in scopes[:2]:
        store.vector_search_episodes(onehot(0), [s], 5)
    assert list(store._turns) == [("acme", "b", None, None, None)]
    store.vector_search_episodes(onehot(0), [scopes[2]], 5)
    assert list(store._turns) == [("acme", "c", None, None, None)]
    assert store._turns_held == 5


def test_no_scopes_matches_nothing_rather_than_everything(store):
    """Fail closed. An empty scope list means no scope was resolved — a caller bug —
    and matching everything would return every tenant's rows to whoever asked."""
    put(store, scope=Scope("acme", "alice"))
    put(store, scope=Scope("other", "bob"))
    assert store.candidate_ids([]) == []
    assert store.lexical_search("berlin", [], limit=10) == []


def test_no_scopes_fails_closed_for_vector_search_too(store, emb):
    c = put(store, emb, scope=Scope("acme", "alice"))
    assert store.get_embedding(c.id) is not None
    assert store.vector_search(emb.encode(["user lives in Berlin"])[0], [], limit=10) == []


def test_include_invalidated_still_cannot_see_the_future(store):
    """Auditing past belief must not leak knowledge acquired later — that is the one
    way a bitemporal query can actively lie."""
    old = put(store, object="Berlin", recorded_at=T0, valid_from=T0)
    store.invalidate(old.id, T1, "cl_b")
    later = put(store, object="Lisbon", recorded_at=T1, valid_from=T1)

    audit = set(store.candidate_ids(
        [SCOPE], valid_at=TMID, known_at=TMID, include_invalidated=True))
    assert audit == {old.id}, "a claim recorded after known_at is not past belief"
    assert later.id not in audit


def test_include_invalidated_reveals_expired_and_retracted_claims(store):
    a = put(store, object="Berlin", recorded_at=T0, valid_from=T0)
    store.invalidate(a.id, T1, None)
    store.set_valid_to(a.id, T1)
    assert store.candidate_ids([SCOPE]) == []
    assert set(store.candidate_ids([SCOPE], include_invalidated=True)) == {a.id}


def test_competing_claims_moves_its_two_axes_independently(store):
    """Off the diagonal, where a transposed pair is observable.

    Every other test of this method — in either repo — passes one instant to both axes,
    because every production caller does: reconciliation and `forget` all ask about one
    moment. So `_live_clause(valid, valid)` here was invisible to four suites at once,
    and the Postgres backend could have collapsed the pair with nothing going red. The
    code was correct on both; the coverage was absent, which is the harder thing to
    notice.

    It matters because this method's own docstring names the audit use — "what did we
    think in August held the salary slot in June" — and that question is off-diagonal by
    construction.

    Two claims in one slot: one scheduled (recorded early, valid later) and one
    backfilled (recorded later, valid early). Each is reachable from exactly one
    off-diagonal reading, and neither diagonal reading separates them.
    """
    scheduled = put(store, object="Lisbon", recorded_at=T0, valid_from=TMID)
    backfilled = put(store, object="Rome", recorded_at=TMID, valid_from=T0)
    assert scheduled.fact_key == backfilled.fact_key, "not one slot; the test is void"

    def slot(**kw):
        return sorted(c.object for c in
                      store.competing_claims("acme", scheduled.fact_key, **kw))

    assert slot(valid_at=TMID, known_at=T0) == ["Lisbon"]
    assert slot(valid_at=T0, known_at=TMID) == ["Rome"]
    assert slot(valid_at=T0, known_at=T0) == []
    assert slot(valid_at=TMID, known_at=TMID) == ["Lisbon", "Rome"]


def test_include_invalidated_lifts_the_whole_valid_interval_so_valid_at_is_inert(store):
    """The exact semantics of the flag, pinned because a backend got them wrong and the
    suite stayed green.

    `include_invalidated` drops `valid_from <= ?` as well as the two end-of-life
    constraints, so it lifts the **whole** valid-time interval and `valid_at` stops
    filtering anything. That is surprising — the flag's name and its old docstring both
    suggested the narrower reading, keeping the `valid_from` floor — and the Postgres
    backend implemented the narrow one first. Mutating it either way left every test in
    this file passing, which is how a documented cross-backend guarantee had nothing
    behind it.

    The claim below is *not yet valid* at the instant queried. Under the shipped
    behaviour it comes back anyway; under the narrow reading it does not. Nothing else
    here distinguishes them.
    """
    future = put(store, object="Lisbon", recorded_at=T0, valid_from=T2)

    assert store.candidate_ids([SCOPE], valid_at=T0, known_at=T0) == [], \
        "not valid yet, so the ordinary predicate excludes it"
    assert set(store.candidate_ids([SCOPE], valid_at=T0, known_at=T0,
                                   include_invalidated=True)) == {future.id}, \
        "include_invalidated lifts the valid-time floor too, so valid_at cannot exclude it"
    # The belief clock is *not* lifted, which is what keeps the flag from leaking the
    # future — see the test above. Both halves are load-bearing and only together do they
    # describe the flag.
    assert store.candidate_ids([SCOPE], known_at=datetime(2023, 1, 1, tzinfo=timezone.utc),
                               include_invalidated=True) == []


def test_candidate_ids_applies_bitemporal_filter(store):
    a = put(store, object="Berlin", recorded_at=T0, valid_from=T0)
    store.invalidate(a.id, T1, "cl_b")
    b = put(store, object="Lisbon", recorded_at=T1, valid_from=T1)
    assert set(store.candidate_ids([SCOPE], valid_at=TMID, known_at=TMID)) == {a.id}
    assert set(store.candidate_ids([SCOPE], valid_at=T2, known_at=T2)) == {b.id}
    assert set(store.candidate_ids([SCOPE], include_invalidated=True)) == {a.id, b.id}


def test_the_two_axes_move_independently_in_candidate_ids(store):
    """The store-level statement of the question `as_of` could not ask. `a` was true
    from T0 and we retired it at T1; `b` corrects the same slot but asserts a *valid*
    interval that also starts at T0. Asked with one instant the correction is either
    unknown (TMID) or the world has moved on (T2); asked with the axes apart it is the
    answer, which is what a late-arriving fact is for."""
    a = put(store, object="Berlin", recorded_at=T0, valid_from=T0)
    store.invalidate(a.id, T1, "cl_b")
    store.set_valid_to(a.id, T1)
    b = put(store, object="Lisbon", recorded_at=T1, valid_from=T0, valid_to=T1)

    assert set(store.candidate_ids([SCOPE], valid_at=TMID, known_at=TMID)) == {a.id}
    assert set(store.candidate_ids([SCOPE], valid_at=TMID, known_at=T2)) == {b.id}
    assert store.candidate_ids([SCOPE], valid_at=T2, known_at=T2) == []


# --- The vector leg over claims, from each scope's cached list -----------------------

NOW = datetime(2025, 6, 1, tzinfo=timezone.utc)
AHEAD = [NOW + timedelta(hours=h) for h in range(1, 6)]
ALICE = ("acme", "alice", None, None, None)


class _Clock:
    """The wall clock the store reads for the present, set by the test."""

    def __init__(self, monkeypatch, at):
        self.at = at
        monkeypatch.setattr(sqlite_store, "utcnow", lambda: self.at)


def _sql_claim_search(store, qvec, scopes, limit, at, **kw):
    """What `vector_search` ranks for a read of the present at `at` without the cache:
    the SQL candidate list with both instants pinned there, handed to the index. The
    expiry clause reads the wall clock, so that must be at `at` too."""
    allowed = store.candidate_ids(scopes, valid_at=at, known_at=at, **kw)
    store._ensure_index()
    return store._vec.search(qvec, allowed, limit) if allowed else []


def test_the_cached_claim_search_returns_what_the_sql_candidate_list_returns(
        store, monkeypatch):
    """The cache replaces a SQL candidate list, so it is checked against that list: the
    same claims, in the same order, with the same scores, for every shape of scope asked
    from its own ancestors, in every set of states, at limits from one to everything.
    Half the claims share a vector with others, because a tie is where a candidate order
    other than SQL's would put a different claim inside the limit.

    The claims are live, ended, retired and expired, and some are due to start, end, be
    retired, expire or become known in the hours after the first read. The clock then
    stops half an hour past each of those instants in turn, and goes back to the start,
    so every list is checked after the instant its `until` named, and against a clock
    earlier than the one it was built at."""
    clock = _Clock(monkeypatch, NOW)
    rng = np.random.default_rng(11)
    shapes = [Scope("acme", u, a, s) for u in (None, "alice") for a in (None, "bot")
              for s in (None, "s1")]
    shared = rng.standard_normal((6, 64)).astype(np.float32)
    # Each claim is due to change at one instant at most, so each instant can be found
    # only through the column that holds it.
    kinds = [({}, None), ({"valid_to": TMID}, None), ({"expires_at": TMID}, None),
             ({}, TMID), ({"valid_from": AHEAD[0]}, None), ({"valid_to": AHEAD[1]}, None),
             ({"expires_at": AHEAD[2]}, None), ({"recorded_at": AHEAD[3]}, None),
             ({}, AHEAD[4])]
    for i in range(280):
        fields, retired = kinds[i % len(kinds)]
        c = put(store, object=f"city {i}", scope=shapes[i % len(shapes)], **fields)
        if retired is not None:
            store.invalidate(c.id, retired, None)
        if i % 13 == 0:
            continue  # never embedded, so never a candidate on either path
        store.set_embedding(
            c.id, shared[i % 6] if i % 2 else rng.standard_normal(64).astype(np.float32))
    state_sets = [None, ["live"], ["retired"], ["ended"], ["live", "ended"],
                  ["live", "retired"], ["ended", "retired"], list(STATES)]
    for at in (NOW, *(due + timedelta(minutes=30) for due in AHEAD), NOW):
        clock.at = at
        for shape in shapes:
            asked = shape.ancestors()
            for states in state_sets:
                for limit in (1, 5, 17, 1000):
                    q = shared[limit % 6] if limit % 2 else rng.standard_normal(64)
                    got = store.vector_search(q, asked, limit, states=states)
                    want = _sql_claim_search(store, q, asked, limit, at, states=states)
                    assert got == want, (at, shape, states, limit)
    assert store._claims, "the searches above were answered from the cache"


def test_a_warm_claim_search_asks_sqlite_nothing_about_claims(store):
    """The point of the cache: once a scope's list is built, a read of the present reads
    no claim until something is written or a claim is due to change state."""
    c = put(store, object="Berlin")
    store.set_embedding(c.id, onehot(1))
    assert store.vector_search(onehot(1), [SCOPE], 5)[0][0] == c.id
    statements: list[str] = []
    store._db.set_trace_callback(statements.append)
    try:
        assert store.vector_search(onehot(1), [SCOPE], 5)[0][0] == c.id
    finally:
        store._db.set_trace_callback(None)
    assert not any("claims" in s for s in statements), statements


def test_the_cached_list_changes_when_the_clock_reaches_a_claims_next_instant(
        store, monkeypatch):
    """A claim's state can change with no write at all, when the clock reaches its
    start, its end, its retirement or its expiry. A list is kept only until the first
    such instant among its tenant's claims, and the entry built after it replaces the
    one before."""
    clock = _Clock(monkeypatch, NOW)
    starts = put(store, object="Lisbon", valid_from=AHEAD[0])
    expires = put(store, object="Berlin", expires_at=AHEAD[1])
    for c in (starts, expires):
        store.set_embedding(c.id, onehot(1))
    key = (ALICE, ("live",), True)

    assert [h[0] for h in store.vector_search(onehot(1), [SCOPE], 5)] == [expires.id]
    assert store._claims[key].until == AHEAD[0].timestamp()
    clock.at = AHEAD[0]
    assert ({h[0] for h in store.vector_search(onehot(1), [SCOPE], 5)}
            == {starts.id, expires.id})
    assert store._claims[key].until == AHEAD[1].timestamp()
    clock.at = AHEAD[1]
    assert [h[0] for h in store.vector_search(onehot(1), [SCOPE], 5)] == [starts.id]
    assert store._claims[key].until == float("inf")
    assert list(store._claims) == [key] and store._claims_held == 1


def test_a_claim_due_in_another_scope_of_the_tenant_brings_the_list_forward(
        store, monkeypatch):
    """`until` is found per tenant rather than per scope, so another user's claim can
    end this user's list early. That costs a rebuild and never a wrong answer."""
    clock = _Clock(monkeypatch, NOW)
    mine = put(store, object="Berlin")
    store.set_embedding(mine.id, onehot(1))
    put(store, object="Lisbon", scope=Scope("acme", "bob"), valid_from=AHEAD[0])
    assert [h[0] for h in store.vector_search(onehot(1), [SCOPE], 5)] == [mine.id]
    assert store._claims[(ALICE, ("live",), True)].until == AHEAD[0].timestamp()
    clock.at = AHEAD[0]
    assert [h[0] for h in store.vector_search(onehot(1), [SCOPE], 5)] == [mine.id]


def test_every_column_a_state_compares_with_the_clock_is_one_the_cache_watches():
    """A cached list is kept until the first instant a claim of the tenant changes state,
    found from the columns `_LAST_CHANGE` and `_NEXT_CHANGE` read. A state that came to
    compare another column with the instant would change as the clock moves, and the
    cache would keep a list past the change. So every column any state predicate, and the
    expiry clause, compares with the instant must be one those two read."""
    compared: set[str] = set()
    for states in (["live"], ["retired"], ["ended"], ["live", "ended"],
                   ["live", "retired"], ["ended", "retired"], list(STATES)):
        sql, _ = state_predicate("?", states=states)
        compared |= set(re.findall(r"(\w+) (?:<=|>=|<|>) \?", sql))
    compared |= set(re.findall(r"(\w+) (?:<=|>=|<|>) \?", unexpired_predicate("?")))
    assert compared == {"recorded_at", "invalidated_at", "valid_from", "valid_to",
                        "expires_at"}
    for expression in (sqlite_store._LAST_CHANGE, sqlite_store._NEXT_CHANGE):
        assert compared <= set(re.findall(r"\w+", expression)), expression


def test_the_next_change_is_found_through_its_index(store):
    """`until` costs one seek only while SQLite reads `cl_last_change` for it, which it
    does only for the exact expression the index holds; and the list half of the
    statement must read the claims as `candidate_ids` does, so they come back in the
    same order."""
    c = put(store)
    store.set_embedding(c.id, onehot(1))
    statements: list[str] = []
    store._db.set_trace_callback(statements.append)
    try:
        store.vector_search(onehot(1), [SCOPE], 5)
    finally:
        store._db.set_trace_callback(None)
    (sql,) = [s for s in statements if "UNION ALL SELECT min(" in s]
    plan = [r["detail"] for r in store._db.execute("EXPLAIN QUERY PLAN " + sql,
                                                   [None] * sql.count("?"))]
    # The expression as a range, not only the tenant: an expression that differs from
    # the index's still uses the index for `tenant`, and then reads every claim of it.
    assert any("cl_last_change (tenant=? AND <expr>>?)" in step for step in plan), plan
    assert any("cl_scope" in step for step in plan), plan


def test_a_claim_written_or_retired_after_a_search_is_seen_by_the_next_one(store):
    """Every commit empties the lists, so the next search sees what it wrote."""
    first = put(store, object="Berlin")
    store.set_embedding(first.id, onehot(1))
    assert [h[0] for h in store.vector_search(onehot(2), [SCOPE], 5)] == [first.id]
    later = put(store, object="Lisbon")
    store.set_embedding(later.id, onehot(2))
    assert store.vector_search(onehot(2), [SCOPE], 1)[0][0] == later.id
    store.invalidate(later.id, T1, None)
    assert [h[0] for h in store.vector_search(onehot(2), [SCOPE], 5)] == [first.id]


def test_a_warm_claim_cache_sees_what_another_worker_writes_and_retires(tmp_path):
    """Another process's commit reaches this one only as a new `data_version`, which
    the cached search reads before every search and which empties the lists."""
    path = str(tmp_path / "c.db")
    a, b = SQLiteStore(path), SQLiteStore(path)
    first = put(a, object="Berlin")
    a.set_embedding(first.id, onehot(1))
    assert a.vector_search(onehot(1), [SCOPE], 5)[0][0] == first.id
    c = put(b, object="Lisbon")
    b.set_embedding(c.id, onehot(5))
    assert a.vector_search(onehot(5), [SCOPE], 1)[0][0] == c.id
    b.invalidate(first.id, T1, None)
    assert [h[0] for h in a.vector_search(onehot(1), [SCOPE], 5)] == [c.id]
    a.close()
    b.close()


def test_a_claim_search_inside_a_batch_sees_its_own_claims_and_the_cache_keeps_none(
        store):
    """Inside `batch()` a thread reads its own uncommitted rows, which a list shared
    with other threads must never hold; after a rollback nothing may return a claim that
    was never written."""
    first = put(store, object="Berlin")
    store.set_embedding(first.id, onehot(1))
    store.vector_search(onehot(1), [SCOPE], 5)
    with pytest.raises(RuntimeError):
        with store.batch():
            c = put(store, object="Lisbon")
            store.set_embedding(c.id, onehot(2))
            assert store.vector_search(onehot(2), [SCOPE], 1)[0][0] == c.id
            raise RuntimeError("roll back")
    assert [h[0] for h in store.vector_search(onehot(2), [SCOPE], 5)] == [first.id]


def test_a_pinned_or_filtered_claim_search_asks_sql(store, monkeypatch):
    """The lists answer a read of the present with no filter. A read pinned to an
    instant would need a list per instant, and a filter can name any metadata field."""
    c = put(store, object="Berlin", valid_from=T1)
    store.set_embedding(c.id, onehot(1))
    monkeypatch.setattr(store, "_cached_claim_search",
                        lambda *a, **k: pytest.fail("answered from the cache"))
    assert store.vector_search(onehot(1), [SCOPE], 5, valid_at=TMID, known_at=TMID) == []
    assert store.vector_search(onehot(1), [SCOPE], 5, known_at=T2)[0][0] == c.id
    assert store.vector_search(onehot(1), [SCOPE], 5,
                               where=SearchFilter(meta=(), filepath_prefix="docs/")) == []


def test_turning_expiry_hiding_off_is_not_answered_from_a_list_built_with_it_on(store):
    """`hide_expired` decides which claims a list holds, so it is part of the key."""
    gone = put(store, object="Berlin", expires_at=T1)
    store.set_embedding(gone.id, onehot(1))
    assert store.vector_search(onehot(1), [SCOPE], 5) == []
    store.hide_expired = False
    assert [h[0] for h in store.vector_search(onehot(1), [SCOPE], 5)] == [gone.id]


def test_a_scope_listed_twice_ranks_its_claims_once_from_the_cache(store):
    """The cached search drops a repeated scope as `_scoped_union` does. Two copies of
    one scope's list would put each of its claims in the ranking twice."""
    c = put(store)
    store.set_embedding(c.id, onehot(1))
    twice = [SCOPE, Scope("acme", "alice")]
    assert [h[0] for h in store.vector_search(onehot(1), twice, 5)] == [c.id]
    assert store._claims, "the search was answered from the cache"


def test_a_claim_row_moved_after_the_cache_was_read_sends_the_search_to_sql(store):
    """As for turns: a returned claim must still hold the row it was ranked by, or the
    search asks SQL instead."""
    a_ = put(store, object="Berlin")
    store.set_embedding(a_.id, onehot(1))
    b_ = put(store, object="Lisbon")
    store.set_embedding(b_.id, onehot(2))
    store.vector_search(onehot(1), [SCOPE], 5)
    store._vec.forget(a_.id)
    assert [h[0] for h in store.vector_search(onehot(1), [SCOPE], 5)] == [b_.id]


def test_a_claim_list_built_across_a_commit_is_not_kept(store, monkeypatch):
    """`_scope_claims` reads SQL and the map outside every lock, so a commit can land
    between its read and its insert. The list is right for the search that built it,
    which began before the commit, and wrong for any later one."""
    c = put(store)
    store.set_embedding(c.id, onehot(1))
    real = store._read

    @contextmanager
    def committed_meanwhile():
        with real() as conn:
            yield conn
        store._changed()

    monkeypatch.setattr(store, "_read", committed_meanwhile)
    now = datetime.now(timezone.utc)
    assert store._scope_claims(SCOPE, ALICE, ("live",), now).ids == [c.id]
    assert not store._claims


def test_the_claim_lists_stay_inside_their_row_budget(store, monkeypatch):
    """The least recently used scope goes first once the lists hold more claims than
    the budget, and one scope is kept whatever its size."""
    monkeypatch.setattr(sqlite_store, "_SCOPE_CLAIMS_ROWS", 5)
    scopes = [Scope("acme", u) for u in ("a", "b", "c")]
    for i, s in enumerate(scopes):
        for j in range(3 + 2 * (i == 2)):
            c = put(store, object=f"city {i} {j}", scope=s)
            store.set_embedding(c.id, onehot(i))
    for s in scopes[:2]:
        store.vector_search(onehot(0), [s], 5)
    assert [k[0] for k in store._claims] == [("acme", "b", None, None, None)]
    store.vector_search(onehot(0), [scopes[2]], 5)
    assert [k[0] for k in store._claims] == [("acme", "c", None, None, None)]
    assert store._claims_held == 5


# --- Maintenance ------------------------------------------------------------

def test_iter_claims_filters_by_tenant_and_liveness(store):
    a = put(store, scope=Scope("acme", "alice"))
    put(store, scope=Scope("other", "bob"))
    store.invalidate(a.id, T1, None)
    assert [c.id for c in store.iter_claims(tenant="acme")] == []
    assert [c.id for c in store.iter_claims(tenant="acme", include_invalidated=True)] == [a.id]
    assert len(list(store.iter_claims())) == 1


def test_purge_reports_what_it_actually_erased(store, emb):
    put(store, emb, object="Berlin")
    put(store, object="Lisbon", predicate="likes")  # never embedded
    store.add_episode(Episode(content="hello", scope=SCOPE))
    # `entities` is in the receipt because it is part of the erasure: the entity row
    # holds a subject's and object's first-seen spelling verbatim. These claims were
    # written through the store directly, so no entity was ever resolved for them.
    assert store.purge(SCOPE) == {"claims": 2, "episodes": 1, "embeddings": 1,
                                  "entities": 0, "documents": 0, "document_chunks": 0}


def test_purge_erases_a_large_scope_in_one_pass(store, emb):
    """Set-based rather than a statement pair per claim: erasing a user is one request,
    and the counts must not depend on how many claims that turned out to be."""
    with store.batch():
        for i in range(1500):
            put(store, emb, predicate=f"p{i}")
    counts = store.purge(SCOPE)
    assert counts["claims"] == 1500
    assert counts["embeddings"] == 1500
    assert store.stats() == {"episodes": 0, "claims": 0, "live_claims": 0,
                             "ended_claims": 0, "invalidated": 0, "embeddings": 0}
    assert store.lexical_search("berlin", [SCOPE], limit=10) == []


def test_purge_erases_the_episode_indexes_not_only_the_rows(store, emb):
    """An FTS row surviving the episode it describes is not a stale cache entry — it is
    the purged text still being searchable, which is the compliance failure the whole
    call exists to avoid."""
    ep = turn(store, emb, content="the kafka pipeline is being sunset")
    counts = store.purge(SCOPE)

    assert counts == {"claims": 0, "episodes": 1, "embeddings": 1, "entities": 0,
                      "documents": 0, "document_chunks": 0}
    assert store.lexical_search_episodes("kafka", [SCOPE], limit=10) == []
    assert store.get_episode_embedding(ep.id) is None
    assert store._db.execute("SELECT COUNT(*) FROM episodes_fts").fetchone()[0] == 0
    assert store._vec.get(ep.id) is None, "the matrix row must be blanked too"


def test_purge_hands_episode_rows_back_to_the_free_list(store, emb):
    turn(store, emb, content="one")
    turn(store, emb, content="two")
    store.purge(SCOPE)
    assert [r[0] for r in
            store._db.execute("SELECT slot FROM vec_free ORDER BY slot")] == [0, 1]
    reused = turn(store, emb, content="three")
    assert store._db.execute(
        "SELECT slot FROM episode_embeddings WHERE episode_id=?", (reused.id,)
    ).fetchone()[0] in (0, 1)


def test_purge_leaves_a_sibling_scopes_turns_alone(store, emb):
    mine = turn(store, emb, content="kafka", scope=MINE)
    theirs = turn(store, emb, content="kafka", scope=SIBLING_SESSION)
    store.purge(MINE)
    assert store.get_episode(mine.id) is None
    assert store.get_episode(theirs.id) is not None
    assert [h[0] for h in store.lexical_search_episodes(
        "kafka", [SIBLING_SESSION], limit=10)] == [theirs.id]


def test_stats_counts_live_and_invalidated_separately(store, emb):
    a = put(store, emb, object="Berlin")
    put(store, emb, object="Lisbon")
    store.invalidate(a.id, T1, None)
    s = store.stats()
    assert s == {"episodes": 0, "claims": 2, "live_claims": 1, "ended_claims": 0,
                 "invalidated": 1, "embeddings": 2}


def test_connectivity_reports_a_star_as_a_star(store):
    """Facts extracted from one person's sentences all share a subject, and their objects
    are leaves. That is what a personal knowledge graph is, and it has no paths in it.

    The number matters because it is the one that decides whether `read_w_graph > 0` can
    pay for itself, and "is this a graph" cannot: every claim is an edge already.
    """
    put(store, predicate="uses", object="pytest")
    put(store, predicate="lives_in", object="Delhi")
    assert store.connectivity() == {"live_claims": 2, "joinable_claims": 0}


def test_connectivity_counts_a_claim_that_leads_to_another_claim(store):
    put(store, predicate="uses", object="pytest")
    put(store, subject="pytest", predicate="configured_in", object="pyproject.toml")
    # Only the first leads anywhere: `pyproject.toml` is nobody's subject.
    assert store.connectivity() == {"live_claims": 2, "joinable_claims": 1}


def test_connectivity_does_not_walk_through_a_retired_claim(store):
    """A path through a claim we no longer believe is not a path, which is `adjacent`'s
    rule and has to be this counter's too, or the rate promises hops the walk refuses.
    """
    put(store, predicate="uses", object="pytest")
    bridge = put(store, subject="pytest", predicate="configured_in",
                 object="pyproject.toml")
    assert store.connectivity()["joinable_claims"] == 1
    store.invalidate(bridge.id, T1, None)
    assert store.connectivity() == {"live_claims": 1, "joinable_claims": 0}


def test_connectivity_ignores_self_loops_and_empty_ends(store):
    """A retraction stores `''` for the object it retracts. Counting empty ends would
    join every retraction to every claim about the retracting subject, which is the same
    giant-hub failure `adjacent` names — and a claim whose object folds onto its own
    subject leads back to where the walk already stands.
    """
    put(store, predicate="knows", object="user")            # self-loop
    put(store, predicate="retracted", object="")            # empty end
    assert store.connectivity() == {"live_claims": 2, "joinable_claims": 0}


def test_connectivity_does_not_count_a_negation_as_a_link(store):
    """"Alice does not work at Acme" is adjacency and is not a link — `_edges` drops it,
    so the counter must too. A rate built from edges the traverser refuses to follow
    promises hops that will not happen, which is worse than reporting no rate at all.
    """
    put(store, predicate="uses", object="pytest", polarity=-1)
    put(store, subject="pytest", predicate="configured_in", object="pyproject.toml")
    assert store.connectivity() == {"live_claims": 2, "joinable_claims": 0}

    # And the far end has to be walkable too, not merely present.
    put(store, predicate="uses", object="tox")
    put(store, subject="tox", predicate="configured_in", object="tox.ini", polarity=-1)
    assert store.connectivity()["joinable_claims"] == 0


def test_connectivity_is_scoped_to_one_tenant(store):
    """Two tenants' claims must not join to each other, for `stats()`'s reason: the rate
    would disclose a neighbour's shape, and it would be wrong about this one's.
    """
    put(store, predicate="uses", object="pytest")
    put(store, subject="pytest", predicate="configured_in", object="pyproject.toml",
        scope=Scope(tenant="other"))
    assert store.connectivity("acme") == {"live_claims": 1, "joinable_claims": 0}
    assert store.connectivity() == {"live_claims": 2, "joinable_claims": 1}


def test_live_claims_is_the_liveness_predicate_and_not_the_invalidated_column(store, emb):
    """`live_claims` used to be `invalidated_at IS NULL`, which was the same number only
    while superseding closed both clocks. It closes valid time alone now, so the cheap
    test counts every superseded version of every slot as live — a store holding one
    address that has changed four times would report four live claims, and
    `repr(Memvara)` would show `claims=5/5` for a store with one current fact in it.

    The three totals therefore do not sum, and that is the model rather than a rounding
    error: a claim that has *ended* is neither live nor invalidated. Any backend
    implementing `Store` has to count it the same way, or the same store reports a
    different size depending on where its rows live.
    """
    put(store, emb, object="Berlin", valid_to=T1)      # ended: over, still believed
    put(store, emb, object="Lisbon", predicate="p2")   # live
    gone = put(store, emb, object="Rome", predicate="p3")
    store.invalidate(gone.id, T1, None)                # retired: no longer believed

    s = store.stats()
    assert (s["claims"], s["live_claims"], s["invalidated"]) == (3, 1, 1)
    assert s["live_claims"] + s["invalidated"] < s["claims"], "three states, not two"


def test_ended_claims_does_not_double_count_a_claim_that_ended_then_was_retired(
        store, emb):
    """The reason `ended_claims` had to be a key rather than a subtraction.

    `claims - live_claims - invalidated` looks like it should give the ended population,
    and it under-counts, because the two sets it subtracts overlap: a claim whose world
    interval closed and which was *later* withdrawn is already inside `invalidated`, so
    the subtraction removes it twice and reports one ended claim where there are two.

    This is the row that proves it — ended in January, retired a year later — and it must
    land in `invalidated` and nowhere else. Its `Claim.state` is `retired`, because
    withdrawing a record supersedes the fact that it also finished: we are no longer
    asserting anything about it, including when it stopped.
    """
    both = put(store, emb, object="Berlin", valid_to=T1)   # ended...
    store.invalidate(both.id, T2, None)                    # ...and later retired
    put(store, emb, object="Rome", predicate="p2", valid_to=T1)   # ended, still believed
    put(store, emb, object="Lisbon", predicate="p3")              # live

    s = store.stats()
    assert store.get_claim(both.id).state == "retired"
    assert (s["claims"], s["live_claims"], s["ended_claims"], s["invalidated"]) \
        == (3, 1, 1, 1)
    # The populations are disjoint, which is the property the subtraction assumed and
    # did not have.
    assert s["live_claims"] + s["ended_claims"] + s["invalidated"] == s["claims"]
    # And the arithmetic it replaces gets the wrong answer on exactly this store.
    assert s["claims"] - s["live_claims"] - s["invalidated"] == 1 != s["ended_claims"] + 1


def test_ended_claims_is_taken_from_the_same_predicate_as_live_claims(store, emb):
    """Not `valid_to IS NOT NULL`. The cheap column test is what put the wrong liveness
    check into three files, and it is wrong here in the same direction — it counts the
    retired-after-ending row, which `invalidated` already has.

    Counted against the exported predicate on the same rows, with no store instance
    involved, because that is the form another backend's counter can actually use.
    """
    both = put(store, emb, object="Berlin", valid_to=T1)
    store.invalidate(both.id, T2, None)
    put(store, emb, object="Rome", predicate="p2", valid_to=T1)

    clock = "CAST(strftime('%s','now') AS REAL)"
    ended, _ = state_predicate(clock, states=["ended"])
    assert "?" not in ended, "a raw-connection sampler has nothing to bind"

    raw = store._db.execute(f"SELECT COUNT(*) FROM claims WHERE {ended}").fetchone()[0]
    cheap = store._db.execute(
        "SELECT COUNT(*) FROM claims WHERE valid_to IS NOT NULL").fetchone()[0]

    assert raw == store.stats()["ended_claims"] == 1
    assert cheap == 2, "which is the count that double-counts, and why it is not used"


# --- Durability and concurrency --------------------------------------------

def test_store_persists_across_reopen(tmp_path, emb):
    path = str(tmp_path / "m.db")
    s1 = SQLiteStore(path)
    c = put(s1, emb, object="Berlin")
    s1.close()

    s2 = SQLiteStore(path)
    assert s2.get_claim(c.id).object == "Berlin"
    assert s2.stats()["embeddings"] == 1
    hits = s2.vector_search(emb.encode(["user lives in Berlin"])[0], [SCOPE], limit=5)
    assert hits[0][0] == c.id, "vector index must rebuild from disk"
    s2.close()


def test_episode_indexes_survive_a_reopen(tmp_path, emb):
    path = str(tmp_path / "m.db")
    with SQLiteStore(path) as s1:
        ep = turn(s1, emb, content="the kafka pipeline is being sunset")

    with SQLiteStore(path) as s2:
        assert [h[0] for h in s2.lexical_search_episodes(
            "kafka", [SCOPE], limit=5)] == [ep.id]
        hits = s2.vector_search_episodes(
            emb.encode(["the kafka pipeline is being sunset"])[0], [SCOPE], limit=5)
        assert hits[0][0] == ep.id, "the episode's matrix row must come back too"


def test_a_deleted_matrix_file_is_rebuilt_for_episodes_as_well(tmp_path, emb):
    """The matrix is derived data, so losing it must be recoverable — for both kinds of
    vector, since they share the file."""
    import os

    path = str(tmp_path / "m.db")
    with SQLiteStore(path) as s1:
        c = put(s1, emb, object="Berlin")
        ep = turn(s1, emb, content="the kafka pipeline is being sunset")
    os.remove(path + ".vecs")

    with SQLiteStore(path) as s2:
        assert s2.vector_search(
            emb.encode(["user lives in Berlin"])[0], [SCOPE], limit=1)[0][0] == c.id
        assert s2.vector_search_episodes(
            emb.encode(["kafka pipeline"])[0], [SCOPE], limit=1)[0][0] == ep.id


def _writing(path: pathlib.Path) -> sqlite3.Connection:
    """Another connection holding the write lock on a new file that is still in
    rollback-journal mode, as a process part-way through creating the store holds it."""
    other = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    other.execute("CREATE TABLE t (x)")
    other.execute("BEGIN IMMEDIATE")
    return other


def test_opening_a_new_store_waits_for_another_connections_write_lock(tmp_path):
    """Opening a new store switches the file to WAL mode, and while another connection
    holds the write lock, SQLite refuses that switch at once instead of waiting. Before
    the fix for #281 the open then failed within a millisecond with "database is locked".
    It now tries again for as long as any write waits, and opens the store once the other
    connection lets go."""
    path = tmp_path / "c.db"
    other = _writing(path)
    started = time.monotonic()
    release = threading.Timer(0.3, lambda: other.execute("COMMIT"))
    release.start()
    try:
        with SQLiteStore(str(path)) as store:
            took = time.monotonic() - started
            assert store._db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        release.join()
        other.close()
    assert took >= 0.3, f"the store opened after {took:.3f} s, while the lock was held"


def test_opening_a_new_store_gives_up_when_the_lock_outlasts_the_busy_timeout(
        tmp_path, monkeypatch):
    """The open waits as long as any write waits, and no longer, then raises the error
    SQLite gave. The busy timeout is shortened here so the test does not wait five
    seconds."""
    monkeypatch.setattr(sqlite_store, "_BUSY_TIMEOUT", 0.2)
    path = tmp_path / "c.db"
    other = _writing(path)
    started = time.monotonic()
    try:
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            SQLiteStore(str(path))
        took = time.monotonic() - started
    finally:
        other.close()
    assert 0.2 <= took < 3.0, f"the open gave up after {took:.3f} s"


def _creating_elsewhere(path: pathlib.Path) -> sqlite3.Connection:
    """Another store part-way through creating or upgrading the file at `path`: it holds
    the write lock on `<db>.lock` that `SQLiteStore._creating` takes. It lets go with a
    rollback, as `_creating` does; a commit would wait for every shared lock."""
    other = sqlite3.connect(str(path) + ".lock", isolation_level=None,
                            check_same_thread=False)
    other.execute("BEGIN IMMEDIATE")
    return other


def test_a_store_opening_while_another_creates_the_file_waits_for_it(tmp_path):
    """Two stores that ran the schema and the migrations at once on one new file got in
    each other's way. With the switch to WAL mode retried, one could still fail inside a
    migration with "vtable constructor failed" while the other was still creating tables
    and indexes. So a store waits for another store's schema step to finish, and then
    finds the file finished (#281)."""
    path = tmp_path / "c.db"
    other = _creating_elsewhere(path)
    started = time.monotonic()
    release = threading.Timer(0.3, lambda: other.execute("ROLLBACK"))
    release.start()
    try:
        SQLiteStore(str(path)).close()
        took = time.monotonic() - started
    finally:
        release.join()
        other.close()
    assert took >= 0.3, f"the schema step ran after {took:.3f} s, beside the other one"


def test_a_store_gives_up_on_another_that_does_not_finish_creating_the_file(
        tmp_path, monkeypatch):
    monkeypatch.setattr(sqlite_store, "_PRESENCE_WAIT", 0.05)
    path = tmp_path / "c.db"
    other = _creating_elsewhere(path)
    try:
        with pytest.raises(StoreInUseError,
                           match="being created or upgraded by another store"):
            SQLiteStore(str(path))
    finally:
        other.close()


def test_a_store_that_is_open_does_not_hold_up_the_next_open(tmp_path, monkeypatch):
    """A store holds the write lock only while its schema step runs, and afterwards only
    the shared lock every open store holds, so opening never waits for a store that is
    merely open. With the wait this short, a lock held for as long as the store is open
    would make the third open fail."""
    monkeypatch.setattr(sqlite_store, "_PRESENCE_WAIT", 0.5)
    path = str(tmp_path / "c.db")
    with SQLiteStore(path), SQLiteStore(path):
        SQLiteStore(path).close()


def test_an_established_store_opens_without_the_creation_lock(tmp_path, monkeypatch):
    """A file this version has finished with needs no schema step, so its open does not
    take the creation lock, and established stores never queue behind one another. Here
    another store holds that lock throughout, and the wait is too short to outlast it, so
    an open that asked for the lock would fail. The writer connection still gets the two
    settings that belong to a connection rather than to the file: SQLite's defaults are
    secure_delete 0 and synchronous 2 (FULL), and the store sets 1 and 1 (NORMAL)."""
    monkeypatch.setattr(sqlite_store, "_PRESENCE_WAIT", 0.05)
    path = tmp_path / "c.db"
    SQLiteStore(str(path)).close()
    other = _creating_elsewhere(path)
    try:
        with SQLiteStore(str(path)) as store:
            assert store._db.execute("PRAGMA secure_delete").fetchone()[0] == 1
            assert store._db.execute("PRAGMA synchronous").fetchone()[0] == 1
    finally:
        other.close()


@pytest.mark.parametrize("why", ["behind", "rollback journal", "missing late index"])
def test_a_store_that_needs_its_schema_step_waits_for_the_creation_lock(tmp_path, why):
    """Only a file this version has finished with skips the schema step. One an older
    version wrote, one a tool switched out of WAL mode, and one that lacks an index
    `_LATE_INDEXES` adds without a version bump all take the step under the lock, so each
    waits for another store's step to finish, and each is finished afterwards."""
    path = tmp_path / "c.db"
    SQLiteStore(str(path)).close()
    raw = sqlite3.connect(path)
    if why == "behind":
        raw.execute(f"PRAGMA user_version = {SCHEMA_VERSION - 1}")
    elif why == "rollback journal":
        raw.execute("PRAGMA journal_mode=DELETE")
    else:
        raw.execute("DROP INDEX cl_last_change")
    raw.commit()
    raw.close()
    other = _creating_elsewhere(path)
    started = time.monotonic()
    release = threading.Timer(0.3, lambda: other.execute("ROLLBACK"))
    release.start()
    try:
        with SQLiteStore(str(path)) as store:
            took = time.monotonic() - started
            version = store._db.execute("PRAGMA user_version").fetchone()[0]
            mode = store._db.execute("PRAGMA journal_mode").fetchone()[0]
            indexes = {r[0] for r in store._db.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'")}
    finally:
        release.join()
        other.close()
    assert took >= 0.3, f"the open took {took:.3f} s, so it did not wait for the lock"
    assert (version, mode) == (SCHEMA_VERSION, "wal")
    assert "cl_last_change" in indexes


def test_the_fast_path_checks_every_index_the_late_indexes_create():
    """An open skips the schema step only when every late index exists, so the list it
    checks must be every statement in `_LATE_INDEXES`. A statement of another kind would be
    skipped on every established store."""
    body = "\n".join(line for line in sqlite_store._LATE_INDEXES.splitlines()
                     if not line.lstrip().startswith("--"))
    statements = [s.strip() for s in body.split(";") if s.strip()]
    names = [re.match(r"CREATE (?:UNIQUE )?INDEX IF NOT EXISTS (\w+)\s+ON\s", s)
             for s in statements]
    assert all(names), [s for s, n in zip(statements, names) if not n]
    assert {n.group(1) for n in names if n} == sqlite_store._LATE_INDEX_NAMES


def _behind(path: pathlib.Path) -> None:
    """A store at `path` that the next open must upgrade, so it takes the creation lock."""
    SQLiteStore(str(path)).close()
    raw = sqlite3.connect(path)
    raw.execute(f"PRAGMA user_version = {SCHEMA_VERSION - 1}")
    raw.commit()
    raw.close()


def test_the_creation_lock_adds_no_file_beside_the_store(tmp_path, monkeypatch):
    """Nothing is written through the creation lock's connection, so it needs no rollback
    journal. With SQLite's default journal, `BEGIN IMMEDIATE` on the empty lock file
    created `<db>.lock-journal`, so the schema step needed permission to add a file to the
    store's directory, which an open never needed before."""
    seen: list[list[str]] = []
    run_schema = SQLiteStore._run_schema

    def listing_first(self, *args, **kwargs):
        seen.append(sorted(p.name for p in tmp_path.iterdir()))
        return run_schema(self, *args, **kwargs)

    monkeypatch.setattr(SQLiteStore, "_run_schema", listing_first)
    SQLiteStore(str(tmp_path / "c.db")).close()
    assert seen and not any("c.db.lock-journal" in names for names in seen), seen


@pytest.mark.skipif(os.name != "posix", reason="Windows file modes do not express this")
def test_a_store_opens_in_a_directory_it_may_not_add_files_to(tmp_path):
    """The account may write the database and the lock file but may not add a file to the
    directory. The database needs its `-wal` and `-shm` files, which exist while another
    connection has it open, as a running server does. Such a store opened before the
    creation lock existed, and a store that needs its schema step opens so again."""
    path = tmp_path / "c.db"
    _behind(path)
    holder = sqlite3.connect(path)
    holder.execute("SELECT count(*) FROM sqlite_master").fetchall()
    os.chmod(tmp_path, 0o555)
    try:
        with SQLiteStore(str(path)) as store:
            version = store._db.execute("PRAGMA user_version").fetchone()[0]
    finally:
        os.chmod(tmp_path, 0o755)
        holder.close()
    assert version == SCHEMA_VERSION


@pytest.mark.skipif(os.name != "posix", reason="Windows file modes do not express this")
def test_an_upgrade_refuses_a_lock_file_it_may_not_write_and_says_so(tmp_path):
    """SQLite opens a file this process may not write read-only, and `BEGIN IMMEDIATE` on
    a read-only connection takes only a shared lock, without an error, so the creation
    lock would keep nobody out. An open that needs the schema step refuses instead, and
    names the file and what it needs. An established store needs only to read the lock
    file, as it always did, so it still opens."""
    path = tmp_path / "c.db"
    lock = tmp_path / "c.db.lock"
    _behind(path)
    os.chmod(lock, 0o444)
    try:
        if os.access(lock, os.W_OK):
            pytest.skip("this user may write a read-only file")
        with pytest.raises(PermissionError, match=r"c\.db\.lock") as refused:
            SQLiteStore(str(path))
        os.chmod(lock, 0o644)
        SQLiteStore(str(path)).close()            # upgrades the store
        os.chmod(lock, 0o444)
        with SQLiteStore(str(path)) as store:     # an established store: no lock taken
            assert store._db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    finally:
        os.chmod(lock, 0o644)
    message = str(refused.value)
    assert "may not write" in message and "delete it while nothing has the store open" \
        in message, message


# --- Cross-process coherence for episode vectors ----------------------------
#
# Claim vectors get this from `PRAGMA data_version` plus a monotonic `seq`: a reader
# notices that some other connection committed and folds in only the rows past its
# watermark. Episode vectors share the matrix, so they *should* inherit it — but
# "should" is the wrong basis for a property whose failure is silent. A worker that
# never sees another's turns on the vector leg looks completely healthy, because BM25
# still finds them and fusion merely ranks them worse.


def onehot(i: int, dim: int = 64) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[i % dim] = 1.0
    return v


def test_a_turn_embedded_by_another_worker_becomes_visible(tmp_path):
    path = str(tmp_path / "c.db")
    a, b = SQLiteStore(path), SQLiteStore(path)
    a.vector_search_episodes(onehot(0), [SCOPE], limit=1)  # nothing written yet

    ep = Episode(content="written by b", scope=SCOPE)
    b.add_episode(ep)
    b.set_episode_embedding(ep.id, onehot(5))

    hits = a.vector_search_episodes(onehot(5), [SCOPE], limit=1)
    assert [h[0] for h in hits] == [ep.id]
    assert hits[0][1] == pytest.approx(1.0)
    a.close()
    b.close()


def test_a_search_between_another_workers_turn_and_its_vector_finds_the_vector(tmp_path):
    """`add_episode` and `set_episode_embedding` commit separately, so a search can land
    between them: it has a candidate to rank and the store holds no vector yet. The index
    takes its width from the first vector, and it was marked loaded without one. Every
    refresh after that mapped the new rows with no matrix to score them in, so the vector
    leg returned nothing until this process wrote a vector itself, while BM25 kept
    answering and nothing said so."""
    path = str(tmp_path / "c.db")
    a, b = SQLiteStore(path), SQLiteStore(path)
    ep = Episode(content="written by b", scope=SCOPE)
    b.add_episode(ep)
    assert a.vector_search_episodes(onehot(5), [SCOPE], limit=5) == []

    b.set_episode_embedding(ep.id, onehot(5))

    hits = a.vector_search_episodes(onehot(5), [SCOPE], limit=5)
    assert [h[0] for h in hits] == [ep.id]
    assert hits[0][1] == pytest.approx(1.0)
    a.close()
    b.close()


def test_a_turn_re_embedded_by_another_worker_is_the_one_that_is_searched(tmp_path):
    """Re-embedding keeps the row, so the update arrives through the shared mapping
    with no re-read at all."""
    path = str(tmp_path / "c.db")
    a, b = SQLiteStore(path), SQLiteStore(path)
    ep = turn(a)
    a.set_episode_embedding(ep.id, onehot(1))
    assert a.vector_search_episodes(onehot(1), [SCOPE], limit=1)[0][1] == pytest.approx(1.0)

    b.set_episode_embedding(ep.id, onehot(6))

    assert a.vector_search_episodes(onehot(1), [SCOPE], limit=1)[0][1] == pytest.approx(0.0)
    assert a.vector_search_episodes(onehot(6), [SCOPE], limit=1)[0][1] == pytest.approx(1.0)
    a.close()
    b.close()


def test_the_two_watermarks_advance_independently(tmp_path):
    """The specific hazard of putting two tables behind one index. `seq` counts per
    table, so both start at 1 — and a single shared watermark would read the claim row
    at sequence 1, move past it, and then skip the *episode* at sequence 1 entirely.
    That turn would be invisible to this worker's vector leg forever, silently, because
    BM25 still finds it.
    """
    path = str(tmp_path / "c.db")
    a, b = SQLiteStore(path), SQLiteStore(path)
    a.vector_search(onehot(0), [SCOPE], limit=1)

    turns = []
    for i in range(3):
        ep = Episode(content=f"turn {i}", scope=SCOPE)
        b.add_episode(ep)
        b.set_episode_embedding(ep.id, onehot(10 + i))
        turns.append(ep)
    c = claim(predicate="written_late")
    b.put_claim(c)
    b.set_embedding(c.id, onehot(20))

    assert a._seq == {"embeddings": -1, "episode_embeddings": -1}, "A is still cold"
    assert [h[0] for h in a.vector_search(onehot(20), [SCOPE], limit=1)] == [c.id]
    # Every turn, not just the last: the ones a collapsed watermark would drop are the
    # low-numbered ones, and they are the ones that look fine in a spot check.
    for i, ep in enumerate(turns):
        assert [h[0] for h in a.vector_search_episodes(
            onehot(10 + i), [SCOPE], limit=1)] == [ep.id], f"turn {i} went missing"
    assert a._seq == {"embeddings": 1, "episode_embeddings": 3}
    a.close()
    b.close()


def test_an_unchanged_store_re_reads_neither_table(tmp_path, monkeypatch):
    """Two tables means two queries per refresh, so the generation check has to
    short-circuit before either of them — it runs on every read."""
    path = str(tmp_path / "c.db")
    with SQLiteStore(path) as s:
        ep = turn(s)
        s.set_episode_embedding(ep.id, onehot(1))
        s.vector_search_episodes(onehot(1), [SCOPE], limit=1)

        reads = []
        real = SQLiteStore._read_map
        monkeypatch.setattr(SQLiteStore, "_read_map",
                            lambda self: (reads.append(1), real(self))[1])
        for _ in range(5):
            s.vector_search_episodes(onehot(1), [SCOPE], limit=1)
            s.vector_search(onehot(1), [SCOPE], limit=1)
        assert reads == [], "no other connection committed; there is nothing to re-read"


def test_a_turn_indexed_by_another_worker_is_findable_by_text_immediately(tmp_path):
    """The FTS half needs no coherence machinery at all — it is a SQLite table, read
    inside the query — but that is worth pinning rather than assuming, since it is the
    leg a reader falls back on when the vector one is stale."""
    path = str(tmp_path / "c.db")
    a, b = SQLiteStore(path), SQLiteStore(path)
    assert a.lexical_search_episodes("kafka", [SCOPE], limit=5) == []

    turn(b, content="the kafka pipeline is being sunset")

    assert len(a.lexical_search_episodes("kafka", [SCOPE], limit=5)) == 1
    a.close()
    b.close()


# `clear_embeddings()` truncates the vector file, so it needs the store to itself.

_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Another process: open the store, search, wait for a line, search again.
_MAPPER = """
import sys
import numpy as np
from memvara.store import SQLiteStore
from memvara.types import Scope
q = np.zeros(16, dtype=np.float32)
q[3] = 1.0
store = SQLiteStore(sys.argv[1])
print(len(store.vector_search_episodes(q, [Scope("acme", "alice")], 5)), flush=True)
sys.stdin.readline()
print(len(store.vector_search_episodes(q, [Scope("acme", "alice")], 5)), flush=True)
store.close()
"""


def test_clearing_vectors_another_process_maps_is_refused_rather_than_fatal_to_it(
        tmp_path):
    """A process that still mapped the vector file died with SIGBUS on its next vector
    search after another process cleared the vectors: exit code -7, no exception, and
    nothing any caller could catch. It runs in a subprocess, because a SIGBUS in this
    one would end the whole suite."""
    path = str(tmp_path / "shared.db")
    with SQLiteStore(path) as writer:
        for i in range(300):
            writer.set_episode_embedding(turn(writer, content=f"turn {i}").id,
                                         onehot(i, 16))
    mapper = subprocess.Popen([sys.executable, "-c", _MAPPER, path], cwd=_ROOT,
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True)
    refused = ""
    try:
        assert mapper.stdout is not None and mapper.stdout.readline().strip() == "5"
        with SQLiteStore(path) as other:
            try:
                other.clear_embeddings()
            except StoreInUseError as exc:
                refused = str(exc)
                assert len(other.vector_search_episodes(onehot(3, 16), [SCOPE], 5)) == 5
        out, err = mapper.communicate("go\n", timeout=120)
    finally:
        mapper.kill()
    assert mapper.returncode == 0, f"the other process died ({mapper.returncode}): {err}"
    assert out.strip() == "5"
    assert "open in another process" in refused and "Nothing was changed" in refused
    with SQLiteStore(path) as alone:
        assert alone.clear_embeddings() == 300


def test_clearing_vectors_another_store_in_this_process_maps_is_refused(tmp_path,
                                                                        monkeypatch):
    """Two stores in one process map the file separately, so each counts as another.
    Nothing changes, the refusal leaves no lock behind that would keep a third store
    from opening, and once the other store closes, the clear goes through."""
    monkeypatch.setattr(sqlite_store, "_PRESENCE_WAIT", 0.05)
    path = str(tmp_path / "c.db")
    a, b = SQLiteStore(path), SQLiteStore(path)
    ep = turn(a)
    a.set_episode_embedding(ep.id, onehot(1))
    assert a.vector_search_episodes(onehot(1), [SCOPE], 1)[0][0] == ep.id
    with pytest.raises(StoreInUseError, match="another SQLiteStore in this one"):
        b.clear_embeddings()
    SQLiteStore(path).close()
    assert a.vector_search_episodes(onehot(1), [SCOPE], 1)[0][0] == ep.id
    assert b.vector_search_episodes(onehot(1), [SCOPE], 1)[0][0] == ep.id
    a.close()
    assert b.clear_embeddings() == 1
    assert b.vector_search_episodes(onehot(1), [SCOPE], 1) == []
    b.close()


def test_two_clears_at_once_are_both_refused_and_leave_the_file_alone(tmp_path):
    """A clear lets go of its own shared lock to ask for the exclusive one. Two clears
    that let go at once each found the other gone, and the one that got the lock
    truncated the file under the one refused, whose next search died with SIGBUS. So a
    clear asks while holding the database's write lock, and the second waits for that
    with its shared lock still held. `b` is stopped just after it lets go, where a
    scheduler could stop it."""
    path = str(tmp_path / "c.db")
    a, b = SQLiteStore(path), SQLiteStore(path)
    b.set_episode_embedding(turn(b).id, onehot(1))
    assert b.vector_search_episodes(onehot(1), [SCOPE], 1)
    vecs = pathlib.Path(path + ".vecs")
    size = vecs.stat().st_size
    let_go, go_on = threading.Event(), threading.Event()
    presence = b._presence

    class Stopped:
        def execute(self, sql, *args):
            cursor = presence.execute(sql, *args)
            if sql == "COMMIT":
                let_go.set()
                go_on.wait(10)
            return cursor

    outcome = {}

    def clear(name, store):
        try:
            outcome[name] = store.clear_embeddings()
        except StoreInUseError:
            outcome[name] = "refused"

    first = threading.Thread(target=clear, args=("b", b))
    second = threading.Thread(target=clear, args=("a", a))
    b._presence = Stopped()
    try:
        first.start()
        assert let_go.wait(10)
        second.start()
        second.join(0.5)
        assert "a" not in outcome and vecs.stat().st_size == size
    finally:
        go_on.set()
        first.join(10)
        second.join(10)
        b._presence = presence
    assert outcome == {"a": "refused", "b": "refused"}
    assert vecs.stat().st_size == size
    assert b.vector_search_episodes(onehot(1), [SCOPE], 1)
    a.close()
    b.close()


def test_a_store_nothing_refers_to_does_not_stop_a_clear(tmp_path):
    """A store that was never closed holds its lock until Python frees it, and a store
    sits in a reference cycle, so that waits for the cycle collector. A refused clear
    collects once and asks again. Automatic collection is off here, so the clear's own
    collection is the only one that can free the store."""
    path = str(tmp_path / "c.db")
    gc.disable()
    try:
        forgotten = SQLiteStore(path)
        forgotten.set_episode_embedding(turn(forgotten).id, onehot(1))
        del forgotten
        with SQLiteStore(path) as store:
            assert store.clear_embeddings() == 1
    finally:
        gc.enable()


def test_a_store_that_failed_to_open_does_not_count_as_open(tmp_path, monkeypatch):
    """`failed` keeps the traceback, and with it the store that failed to open, the way
    an interactive session keeps the last one, so no collection can free its lock."""
    path = str(tmp_path / "c.db")
    with SQLiteStore(path) as store:
        store.set_episode_embedding(turn(store).id, onehot(1))

    def fail(self):
        raise OSError("the vector file cannot be read")

    monkeypatch.setattr(SQLiteStore, "_attach_vectors", fail)
    with pytest.raises(OSError, match="cannot be read") as failed:
        SQLiteStore(path)
    monkeypatch.undo()
    with SQLiteStore(path) as store:
        assert store.clear_embeddings() == 1
    del failed  # alive until here


def test_a_clear_inside_a_batch_keeps_the_store_to_itself_until_the_batch_ends(
        tmp_path, monkeypatch):
    """A store that opened between the clear and the commit would map vectors the batch
    is about to delete, so it waits, and gives up after `_PRESENCE_WAIT` seconds."""
    monkeypatch.setattr(sqlite_store, "_PRESENCE_WAIT", 0.05)
    path = str(tmp_path / "c.db")
    with SQLiteStore(path) as store:
        store.set_episode_embedding(turn(store).id, onehot(1))
        with store.batch():
            assert store.clear_embeddings() == 1
            assert store.clear_embeddings() == 0  # already its own
            with pytest.raises(StoreInUseError, match="having its vectors cleared"):
                SQLiteStore(path)
        SQLiteStore(path).close()


def test_a_lock_file_that_is_not_a_database_is_named(tmp_path):
    (tmp_path / "c.db.lock").write_bytes(b"not a database " * 100)
    with pytest.raises(RuntimeError, match=r"c\.db\.lock cannot be used"):
        SQLiteStore(str(tmp_path / "c.db"))


@pytest.mark.parametrize("error", [sqlite3.InterfaceError, KeyboardInterrupt])
def test_a_lock_that_fails_to_be_taken_leaves_no_connection_behind(tmp_path, monkeypatch,
                                                                   error):
    """The connection to `<db>.lock` is closed however taking the lock fails, not only when
    SQLite refuses the lock. Left open, it would keep its shared lock until Python freed it,
    and a clear would count a store that never opened. `failed` keeps the traceback, and
    with it every frame of the failed open, alive until the check has run."""
    path = str(tmp_path / "c.db")
    share = SQLiteStore._share

    def share_then_fail(conn):
        share(conn)
        raise error("interrupted")

    monkeypatch.setattr(SQLiteStore, "_share", staticmethod(share_then_fail))
    with pytest.raises(error) as failed:
        SQLiteStore(path)
    monkeypatch.undo()
    alone = sqlite3.connect(path + ".lock", isolation_level=None, timeout=0)
    try:
        alone.execute("BEGIN EXCLUSIVE")    # refused while any other connection has a lock
        alone.execute("ROLLBACK")
    finally:
        alone.close()
    del failed


def write_v2(path: str) -> str:
    """A store as the previous release left it: episodes, and no index over them.

    Built with this code and then cut back, rather than spelled out: v2's DDL for every
    other table is byte-identical to today's, and the only thing that distinguishes the
    file is the absence of the two tables added here plus the stamp.
    """
    with SQLiteStore(path) as s:
        ep = turn(s, content="the kafka pipeline is being sunset")
    raw = sqlite3.connect(path)
    raw.execute("DROP TABLE episodes_fts")
    raw.execute("DROP TABLE episode_embeddings")
    raw.execute("PRAGMA user_version = 2")
    raw.commit()
    raw.close()
    return ep.id


def test_a_v2_file_gets_its_existing_turns_indexed(tmp_path):
    """`CREATE VIRTUAL TABLE IF NOT EXISTS` makes the table and leaves it empty, which
    is indistinguishable from a store whose turns genuinely match nothing — so every
    pre-v3 episode would stay exactly as unfindable as before the upgrade."""
    path = str(tmp_path / "v2.db")
    ep_id = write_v2(path)
    with SQLiteStore(path) as s:
        assert int(s._db.execute("PRAGMA user_version").fetchone()[0]) == SCHEMA_VERSION
        assert [h[0] for h in s.lexical_search_episodes(
            "kafka", [SCOPE], limit=5)] == [ep_id]


def test_the_backfill_does_not_run_twice(tmp_path):
    """It is stamped, and idempotent even if it were not: a doubled FTS row would
    double-count the term and quietly reorder BM25."""
    path = str(tmp_path / "v2.db")
    write_v2(path)
    for _ in range(2):
        with SQLiteStore(path) as s:
            assert len(s.lexical_search_episodes("kafka", [SCOPE], limit=5)) == 1


def test_a_migrated_store_still_takes_new_turns(tmp_path, emb):
    path = str(tmp_path / "v2.db")
    old = write_v2(path)
    with SQLiteStore(path) as s:
        fresh = turn(s, emb, content="kafka replacement is kinesis")
        assert {h[0] for h in s.lexical_search_episodes("kafka", [SCOPE], limit=5)} \
            == {old, fresh.id}
        assert s.vector_search_episodes(
            emb.encode(["kinesis"])[0], [SCOPE], limit=1)[0][0] == fresh.id


def test_concurrent_writers_do_not_corrupt_or_lose_rows(tmp_path, emb):
    store = SQLiteStore(str(tmp_path / "c.db"))
    errors: list[BaseException] = []

    def worker(n: int) -> None:
        try:
            for i in range(25):
                c = claim(predicate=f"p_{n}_{i}", object=f"v{i}")
                store.put_claim(c)
                store.set_embedding(c.id, emb.encode([c.text])[0])
        except BaseException as e:  # noqa: BLE001 - surfaced via assert below
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert store.stats()["claims"] == 200
    assert store.stats()["embeddings"] == 200
    store.close()


def test_claims_and_turns_written_at_once_never_collide_on_a_row(tmp_path, emb):
    """The cross-table allocator under contention. If it were per-table, a claim and a
    turn written concurrently would be handed the same row and each would read back the
    other's vector — silently, since both searches would still return something."""
    store = SQLiteStore(str(tmp_path / "mixed.db"))
    errors: list[BaseException] = []

    def worker(n: int) -> None:
        try:
            for i in range(20):
                put(store, emb, predicate=f"p_{n}_{i}", object=f"v{i}")
                turn(store, emb, content=f"turn {n} {i}")
        except BaseException as e:  # noqa: BLE001 - surfaced via assert below
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    slots = [r[0] for r in store._db.execute(
        "SELECT slot FROM embeddings UNION ALL SELECT slot FROM episode_embeddings")]
    assert len(slots) == 160
    assert len(set(slots)) == 160, "one row per vector, across both tables"
    store.close()


# --- Per-claim erasure ------------------------------------------------------
#
# `purge` erases a scope and `invalidate` retires without deleting anything. Between the
# two sat an erasure request naming one memory, whose only available answer was to retire
# it and report success — with the text, its source turn and its embedding all still on
# disk. These pin that the gap is closed and closed completely.

def test_erase_claim_reports_what_it_erased_in_the_same_shape_purge_does(store):
    """Per-table counts, not a flag. `purge` has always evidenced itself this way and
    this is the path an erasure request naming one memory actually takes, so it was the
    weaker witness of the two.

    The two facts the boolean carried are unchanged and live in `claims`: erasing twice
    is not two erasures, and an id that never existed erases nothing. `claims: 0` rather
    than an absent key or an empty dict, so a caller totalling an erasure campaign never
    special-cases the id that was not there."""
    c = put(store)
    nothing = {"claims": 0, "episodes": 0, "embeddings": 0, "entities": 0}

    assert store.erase_claim(c.id) == {"claims": 1, "episodes": 0, "embeddings": 0,
                                       "entities": 0}
    assert store.erase_claim(c.id) == nothing, "erasing twice is not two erasures"
    assert store.erase_claim("cl_never_existed") == nothing
    purged = set(store.purge(Scope("acme", "nobody")))
    assert set(nothing) <= purged, \
        "the same four keys as purge, or the two paths evidence themselves differently"
    # The two purge adds count documents, which erasing a claim never removes.
    assert purged - set(nothing) == {"documents", "document_chunks"}


def test_erase_claim_takes_the_row_the_index_and_the_vector(store, emb):
    c = put(store, emb, object="Berlin")
    survivor = put(store, emb, object="Lisbon", predicate="visited")

    assert store.erase_claim(c.id) == {"claims": 1, "episodes": 0, "embeddings": 1,
                                       "entities": 0}

    assert store.get_claim(c.id) is None
    assert store.lexical_search("berlin", [SCOPE], limit=10) == []
    assert store.get_embedding(c.id) is None
    assert store._vec.get(c.id) is None, "the matrix row must be blanked, not just unmapped"
    assert store.stats() == {"episodes": 0, "claims": 1, "live_claims": 1,
                             "ended_claims": 0, "invalidated": 0, "embeddings": 1}
    assert [h[0] for h in store.lexical_search("lisbon", [SCOPE], limit=10)] \
        == [survivor.id], "the neighbouring claim is untouched"


def test_erase_takes_the_fts_row_by_rowid_before_the_claim(store):
    """The one ordering in the method that cannot be recovered from. `claim_id` is
    UNINDEXED, so the entry is reachable only through the claim's rowid — delete the
    claim first and the text stays in the index forever: matchable, unhydratable, and
    removable only by rebuilding the whole thing."""
    c = put(store, text="the kafka pipeline is being sunset")
    store.erase_claim(c.id)
    assert store._db.execute("SELECT COUNT(*) FROM claims_fts").fetchone()[0] == 0


def test_an_erased_claims_row_is_handed_back_to_the_free_list(store, emb):
    a = put(store, emb, object="Berlin")
    put(store, emb, object="Lisbon", predicate="visited")
    store.erase_claim(a.id)

    assert [r[0] for r in store._db.execute("SELECT slot FROM vec_free")] == [0]
    reused = put(store, emb, object="Porto", predicate="dreams_of")
    assert store._db.execute(
        "SELECT slot FROM embeddings WHERE claim_id=?", (reused.id,)).fetchone()[0] == 0


def test_an_erased_claim_is_gone_from_history_not_marked_retired(store):
    """Erasure is not a louder retirement: `history()` and `as_of` are exactly what it
    has to defeat, since both are built to keep returning what `delete()` hides."""
    c = put(store, object="Berlin")
    later = put(store, object="Lisbon", recorded_at=T1, valid_from=T1)
    store.erase_claim(c.id)
    assert [h.id for h in store.slot_history("acme", later.fact_key)] == [later.id]


def test_erase_leaves_the_source_turn_alone_by_default(store, emb):
    """A turn is not the claim's private property: it can be the origin of several, and
    can hold a great deal the extractor never turned into one. Deleting it as a side
    effect erases data the caller did not name."""
    ep = turn(store, emb, content="I moved to Berlin last spring, before the merger")
    c = put(store, emb, sources=[ep.id])
    store.erase_claim(c.id)
    assert store.get_episode(ep.id) is not None
    assert len(store.lexical_search_episodes("merger", [SCOPE], limit=5)) == 1


def test_erase_with_sources_takes_the_turn_its_index_and_its_vector(store, emb):
    """The other half, and the one a note or an imported memory needs: there the claim
    *is* its source text, so an erasure that left the episode behind would leave the
    whole memory readable and searchable."""
    ep = turn(store, emb, content="the kafka pipeline is being sunset")
    c = put(store, emb, object="the kafka pipeline is being sunset",
            predicate="note", sources=[ep.id])

    # Two vectors, because the claim's and the turn's both went — `embeddings` is the
    # total across both tables here exactly as it is in `purge`.
    assert store.erase_claim(c.id, sources=True) == {
        "claims": 1, "episodes": 1, "embeddings": 2, "entities": 0}

    assert store.get_episode(ep.id) is None
    assert store.lexical_search_episodes("kafka", [SCOPE], limit=5) == []
    assert store.get_episode_embedding(ep.id) is None
    assert store._vec.get(ep.id) is None
    assert store.stats()["embeddings"] == 0


def test_erase_with_sources_keeps_a_turn_another_claim_still_cites(store, emb):
    """The check that makes `sources=True` safe to offer at all: shared provenance is
    the normal case for anything extracted, and a dangling `why()` is the failure this
    library is least allowed to have."""
    ep = turn(store, emb, content="I moved to Berlin, and I work at Acme")
    a = put(store, emb, object="Berlin", sources=[ep.id])
    b = put(store, emb, predicate="works_at", object="Acme", sources=[ep.id])

    store.erase_claim(a.id, sources=True)
    assert store.get_episode(ep.id) is not None

    store.erase_claim(b.id, sources=True)
    assert store.get_episode(ep.id) is None, "the last citer takes it with them"


def test_erase_rolls_back_with_its_batch(store, emb):
    """The matrix is a mapped file and takes no part in SQLite's transaction, so an
    erasure inside an abandoned batch has to put the vector back or the claim survives
    with nothing to find it by."""
    c = put(store, emb, object="Berlin")
    with pytest.raises(RuntimeError):
        with store.batch():
            store.erase_claim(c.id)
            raise RuntimeError("abandoned")
    assert store.get_claim(c.id) is not None
    assert store._vec.get(c.id) is not None
    assert [h[0] for h in store.vector_search(
        emb.encode(["user lives in Berlin"])[0], [SCOPE], limit=1)] == [c.id]


# --- Reverse provenance ------------------------------------------------------
#
# `claim_sources` is derived data: the authority is the `sources` array on the claim, and
# this table mirrors its edges so "which claims came from this turn?" is a lookup rather
# than a scan of the tenant. Two properties have to hold, and the second is the one with
# teeth. It must be *complete*, or the write path stops reinforcing and `erase(
# sources=True)` deletes turns other claims still cite. And it must not outlive what it
# describes, because a row that survives an erasure reads to `_orphan` as a live citer —
# the same shape as the entities bug wave 4 found, one table over.

def provenance(store) -> set[tuple[str, str]]:
    """The edges actually stored."""
    return {(r[0], r[1]) for r in store._db.execute(
        "SELECT episode_id, claim_id FROM claim_sources")}


def implied(store) -> set[tuple[str, str]]:
    """The edges the surviving claims say there should be."""
    return {(source, c.id)
            for c in store.iter_claims(include_invalidated=True) for source in c.sources}


def test_claims_citing_returns_what_was_extracted_from_one_turn(store):
    ep_a, ep_b = turn(store, content="a"), turn(store, content="b")
    here = put(store, sources=[ep_a.id])
    both = put(store, predicate="works_at", object="Acme", sources=[ep_a.id, ep_b.id])
    put(store, predicate="likes", object="rain", sources=[ep_b.id])

    assert {c.id for c in store.claims_citing("acme", ep_a.id)} == {here.id, both.id}
    assert store.claims_citing("acme", "ep_never_stored") == []


def test_claims_citing_is_tenant_scoped(store):
    """Provenance ids are opaque and a caller could hold one from anywhere. Answering
    across tenants would let a claim id from one customer be confirmed by another."""
    ep = turn(store, content="shared id")
    mine = put(store, sources=[ep.id])
    put(store, scope=Scope("globex", "alice"), sources=[ep.id])
    assert [c.id for c in store.claims_citing("acme", ep.id)] == [mine.id]


def test_claims_citing_includes_a_retired_claim(store):
    """No liveness filter, deliberately: a claim that was superseded last week was still
    extracted from that turn, and `why()` has to keep answering for it. Callers wanting
    only live claims say so — the write path does."""
    ep = turn(store, content="I live in Berlin")
    c = put(store, sources=[ep.id])
    store.invalidate(c.id, T1, "cl_later")
    assert [x.id for x in store.claims_citing("acme", ep.id)] == [c.id]


def test_claims_citing_sees_provenance_added_by_reinforce(store):
    """`reinforce` is the one write of `sources` that does not go through `put_claim`,
    so it is the one place the index can silently stop being told — on the path that
    exists to record repeated observations, which is exactly what this index serves."""
    old, fresh = turn(store, content="first"), turn(store, content="again")
    c = put(store, sources=[old.id])
    store.reinforce(c.id, salience=1.5, observation_count=2, sources=[fresh.id])

    assert [x.id for x in store.claims_citing("acme", fresh.id)] == [c.id]
    assert [x.id for x in store.claims_citing("acme", old.id)] == [c.id]
    assert provenance(store) == implied(store)


def test_a_re_put_that_drops_a_source_drops_its_edge(store):
    """The reason the sync cannot be insert-only. An edge left behind after a claim
    stops citing a turn reads to `_orphan` as a live citer, so the turn survives an
    erasure that should have taken it — under-erasing, silently."""
    kept, dropped = turn(store, content="kept"), turn(store, content="dropped")
    c = put(store, sources=[kept.id, dropped.id])
    c.sources = [kept.id]
    store.put_claim(c)

    assert store.claims_citing("acme", dropped.id) == []
    assert [x.id for x in store.claims_citing("acme", kept.id)] == [c.id]
    assert provenance(store) == implied(store)


def test_a_claim_citing_one_turn_twice_is_stored_once(store):
    """`sources` is whatever the caller passed, and the edge table has a primary key
    over the pair. A caller repeating an id must not take the write down."""
    ep = turn(store, content="once")
    c = put(store, sources=[ep.id, ep.id])
    assert [x.id for x in store.claims_citing("acme", ep.id)] == [c.id]
    assert provenance(store) == {(ep.id, c.id)}


def test_re_putting_an_unchanged_claim_leaves_its_edges_alone(store):
    """The shape every retirement and every reinforcement takes: the claim is rewritten
    with the sources it already had. Nothing to add and nothing to drop."""
    ep = turn(store, content="once")
    c = put(store, sources=[ep.id])
    store.put_claim(c)
    assert provenance(store) == {(ep.id, c.id)}


def edge_statements(store, fn) -> list[str]:
    """Every statement `fn()` runs against `claim_sources`.

    Through SQLite's own trace callback rather than by wrapping `execute`, which is a
    read-only attribute on a `Connection` — and this sees what actually reached the
    engine, including each row of an `executemany`.
    """
    seen: list[str] = []
    store._db.set_trace_callback(
        lambda sql: seen.append(sql) if "claim_sources" in sql else None)
    try:
        fn()
    finally:
        store._db.set_trace_callback(None)
    return seen


def test_re_putting_an_unchanged_claim_does_not_read_its_edges_back(store):
    """Not merely "writes nothing" — *touches* nothing. Finding the difference by
    reading `claim_sources` costs one indexed row per source on every write, whether or
    not anything moved, and provenance is cumulative and uncapped. That read, not the
    JSON column it was blamed on, is what made a re-put grow with the claim's history:
    222.4 us at 365 sources against 95.6 once the prior array is used instead. The claim
    is rewritten unchanged by every retirement, every decay pass and every sweep."""
    eps = [turn(store, content=f"turn {i}").id for i in range(40)]
    c = put(store, sources=eps)
    assert edge_statements(store, lambda: store.put_claim(c)) == []


def test_a_reinforcement_from_a_turn_already_cited_touches_no_edges(store):
    """An exactly repeated turn is the reverse index's own traffic — the write path
    looks up what that turn produced and bumps it. The claim already cites it, so the
    merge is a no-op, and paying to rediscover that on the hottest path this table has
    is the cost the index was supposed to remove."""
    ep = turn(store, content="I live in Berlin")
    c = put(store, sources=[ep.id])
    ran = edge_statements(
        store, lambda: store.reinforce(c.id, 1.5, 2, sources=[ep.id]))
    assert ran == []
    assert store.get_claim(c.id).sources == [ep.id]
    assert store.get_claim(c.id).salience == 1.5, "the bump still lands"


def test_a_reinforcement_from_a_new_turn_writes_only_the_new_edge(store):
    """The other half: a genuinely new source has to reach the index, and only it. A
    delete-and-reinsert here rewrites the whole history on every observation."""
    old = [turn(store, content=f"turn {i}").id for i in range(5)]
    fresh = turn(store, content="said again, differently")
    c = put(store, sources=old)
    ran = edge_statements(
        store, lambda: store.reinforce(c.id, 1.5, 2, sources=[fresh.id]))

    assert [s.split()[0] for s in ran] == ["INSERT"], "no delete, no read-back"
    assert store.get_claim(c.id).sources == [*old, fresh.id], "appended, order kept"
    assert provenance(store) == implied(store)


def fts_statements(store, fn) -> list[str]:
    """Every statement `fn()` runs against `claims_fts` *itself*.

    The same instrument as `edge_statements` and for the same class of question: not
    "did the index end up right" but "was it touched at all". Only the trace callback
    can answer the second, because a delete followed by an insert of identical text
    leaves the table exactly as it found it.

    Unlike `claim_sources`, `claims_fts` is a virtual table, and SQLite traces the
    statements FTS5 runs against its own shadow tables alongside ours. Those arrive
    commented out with a leading `--`, and dropping them is what leaves this reporting
    the two statements `put_claim` writes rather than the nine FTS5 expands them into.
    """
    seen: list[str] = []
    store._db.set_trace_callback(
        lambda sql: seen.append(sql)
        if "claims_fts" in sql and not sql.lstrip().startswith("--") else None)
    try:
        fn()
    finally:
        store._db.set_trace_callback(None)
    return seen


def test_a_new_claim_writes_its_fts_row(store):
    """The first write has nothing to compare against and must always index. Guarding
    the rewrite on a text comparison makes the no-prior-row case the one that breaks
    silently: the claim would be stored, readable, and permanently unsearchable."""
    ran = fts_statements(store, lambda: put(store, object="Berlin"))
    assert [x.split()[0] for x in ran] == ["DELETE", "INSERT"]
    assert store.lexical_search("Berlin", [SCOPE], limit=10), "indexed and findable"


def test_re_putting_a_claim_with_unchanged_text_does_not_rewrite_its_fts_row(store):
    """The hot path, and the reason this matters beyond the wasted work. Every
    reinforcement re-puts a claim whose text has not moved, and the rewrite it performs
    reproduces byte for byte what is already there.

    Under FTS5's `secure-delete` -- which `_migrate_to_v7` turns on for `claims_fts`, so
    it is on for every store this library has written since -- a delete rewrites the
    doclist inside existing segment pages rather than appending a marker. Enough of
    those inside one uncommitted transaction and FTS5 raises SQLITE_CORRUPT_VTAB, which
    surfaces as `database disk image is malformed` from a write that is changing
    nothing. Measured against a store that reproduces it: 19,420 of 19,420 rewrites in
    the first transaction were of unchanged text, and skipping them carried the run 5.5x
    past the write that otherwise fails.
    """
    c = put(store, object="Berlin", text="I live in Berlin")
    assert fts_statements(store, lambda: store.put_claim(c)) == []
    assert store.lexical_search("Berlin", [SCOPE], limit=10)[0][0] == c.id


def test_the_skip_holds_inside_an_open_batch(store):
    """The shape the fault actually takes: thousands of writes inside one uncommitted
    transaction, none of them visible to any other connection yet.

    `prior` is read through the writer connection, which is the only one that can see
    that transaction's own rows. Routing it through `_read()`'s snapshot connection
    instead would compare against stale committed text, and every case above would still
    pass -- outside a batch the two connections agree."""
    with store.batch():
        c = put(store, object="Berlin")
        assert fts_statements(store, lambda: store.put_claim(c)) == []

        moved = claim(id=c.id, object="Lisbon")
        ran = fts_statements(store, lambda: store.put_claim(moved))
        assert [x.split()[0] for x in ran] == ["DELETE", "INSERT"]

    assert store.lexical_search("Lisbon", [SCOPE], limit=10)[0][0] == c.id
    assert store.lexical_search("Berlin", [SCOPE], limit=10) == []


def test_changing_a_claims_text_still_replaces_what_search_finds(store):
    """The guard the skip needs, and the case the measurement never exercised: every
    write in that window was a reinforcement of identical text, so nothing there would
    have caught a skip that fired too widely.

    It also pins *which* text the comparison reads. `_CLAIM_UPSERT` sets every column
    including `text`, so a comparison made after it has run finds the stored value equal
    to the incoming one every time, skips unconditionally, and leaves the index
    answering with text the claim no longer carries -- with no error anywhere.
    """
    c = put(store, object="Berlin", text="I live in Berlin")
    moved = claim(id=c.id, object="Berlin", text="I live in Lisbon")
    ran = fts_statements(store, lambda: store.put_claim(moved))

    assert [x.split()[0] for x in ran] == ["DELETE", "INSERT"], "changed text rewrites"
    assert store.lexical_search("Lisbon", [SCOPE], limit=10)[0][0] == c.id
    assert store.lexical_search("Berlin", [SCOPE], limit=10) == [], "old text is gone"


def test_the_comparison_holds_for_text_the_claim_rendered_itself(store):
    """Most claims never carry an explicit `text`: `Claim.__post_init__` renders one
    from the triple when none is given, so the value being compared is derived rather
    than supplied. Both halves have to keep working for it -- an unchanged triple
    re-renders identical text and must skip, and a moved triple renders different text
    and must reach the index."""
    c = put(store, object="Berlin")
    assert c.text == "user lives in Berlin", "rendered, not supplied"
    assert fts_statements(store, lambda: store.put_claim(c)) == []

    moved = claim(id=c.id, object="Lisbon")
    ran = fts_statements(store, lambda: store.put_claim(moved))
    assert [x.split()[0] for x in ran] == ["DELETE", "INSERT"]
    assert store.lexical_search("Lisbon", [SCOPE], limit=10)[0][0] == c.id
    assert store.lexical_search("Berlin", [SCOPE], limit=10) == []


def test_erasing_a_claim_takes_its_provenance_with_it(store, emb):
    """The fourth-table trap. Wave 4 found `entities` surviving a purge that reported
    per-table counts as evidence; this is the same mistake one table over, and it does
    more than leave a row behind — the erased claim goes on voting to keep its own
    source turn alive, so `sources=True` quietly stops erasing anything."""
    ep = turn(store, emb, content="the kafka pipeline is being sunset")
    c = put(store, emb, sources=[ep.id])

    assert store.erase_claim(c.id, sources=True)["episodes"] == 1

    assert provenance(store) == set() == implied(store)
    assert store.get_episode(ep.id) is None, "the last citer must take the turn with it"


def test_purge_takes_the_provenance_of_every_claim_it_erases(store, emb):
    """Set-based deletion is the path that forgets a side table, because nothing walks
    the rows one at a time to remind you they exist."""
    ep = turn(store, emb, content="I live in Berlin")
    put(store, emb, sources=[ep.id])
    put(store, emb, predicate="works_at", object="Acme", sources=[ep.id])
    survivor_turn = turn(store, emb, content="elsewhere", scope=Scope("globex", "bob"))
    survivor = put(store, emb, scope=Scope("globex", "bob"), sources=[survivor_turn.id])

    store.purge(Scope("acme"))

    assert provenance(store) == {(survivor_turn.id, survivor.id)} == implied(store)


def test_purging_one_scope_leaves_a_neighbours_citation_intact(store, emb):
    """Provenance is allowed to dangle — a purge can take a turn a claim in another
    scope still cites, and wave 3 settled that erasure is not the moment to discover
    that by raising. What must not happen is the *claim's* edge disappearing with it,
    because then the surviving claim's `sources` and the index disagree."""
    ep = turn(store, emb, content="shared", scope=Scope("acme", "alice"))
    other = put(store, emb, scope=Scope("acme", "bob"), sources=[ep.id])

    store.purge(Scope("acme", "alice"))

    assert store.get_episode(ep.id) is None
    assert [c.id for c in store.claims_citing("acme", ep.id)] == [other.id]
    assert provenance(store) == implied(store)


def write_v4(path: str) -> tuple[str, str]:
    """A store as the previous release left it: claims with `sources`, no edge table.

    Built with this code and then cut back, like `write_v2` — every other table's DDL is
    byte-identical to today's, so the absence of `claim_sources` plus the stamp is the
    whole difference.
    """
    with SQLiteStore(path) as s:
        ep = turn(s, content="the kafka pipeline is being sunset")
        c = put(s, sources=[ep.id, ep.id])
        put(s, predicate="likes", object="rain")      # no sources: nothing to backfill
    raw = sqlite3.connect(path)
    raw.execute("DROP TABLE claim_sources")
    raw.execute("PRAGMA user_version = 4")
    raw.commit()
    raw.close()
    return ep.id, c.id


def test_a_v4_file_gets_its_provenance_backfilled_and_is_restamped(tmp_path):
    """`CREATE TABLE IF NOT EXISTS` makes the table and leaves it empty, and an empty
    reverse index is indistinguishable from a store in which nothing cites anything. So
    every pre-v5 claim would stop being reinforced when its turn came round again, and
    every turn behind one would look like an orphan to `erase(sources=True)` — a wrong
    answer in the deleting direction."""
    path = str(tmp_path / "v4.db")
    ep_id, claim_id = write_v4(path)
    with SQLiteStore(path) as s:
        assert int(s._db.execute("PRAGMA user_version").fetchone()[0]) == SCHEMA_VERSION
        assert [c.id for c in s.claims_citing("acme", ep_id)] == [claim_id]
        assert provenance(s) == implied(s)


def test_the_v5_backfill_is_idempotent(tmp_path):
    """It is stamped, and idempotent anyway: a file that starts below 4 walks the whole
    ladder, so this runs again on an upgrade that has already done it."""
    path = str(tmp_path / "v4.db")
    ep_id, claim_id = write_v4(path)
    for _ in range(2):
        with SQLiteStore(path) as s:
            assert [c.id for c in s.claims_citing("acme", ep_id)] == [claim_id]


def test_a_migrated_store_erases_a_backfilled_turn_correctly(tmp_path):
    """The backfill is only worth having if erasure believes it. Before it, the one
    claim citing this turn was invisible to `_orphan`, which would have called the turn
    an orphan and deleted it out from under a claim that still pointed at it."""
    path = str(tmp_path / "v4.db")
    ep_id, claim_id = write_v4(path)
    with SQLiteStore(path) as s:
        extra = put(s, predicate="note", object="n", sources=[ep_id])
        s.erase_claim(extra.id, sources=True)
        assert s.get_episode(ep_id) is not None, "the backfilled claim still cites it"
        s.erase_claim(claim_id, sources=True)
        assert s.get_episode(ep_id) is None, "and the last citer takes it"


def test_the_backfill_pages_rather_than_loading_every_claim(tmp_path):
    """A store large enough to need the index is large enough that reading every claim's
    `sources` into one list is what stops it opening. Asserted as a ratio, like the
    `iter_episodes` walk: at small sizes the fixed cost of one page dominates and any
    byte threshold would be measuring the page."""
    import tracemalloc

    def peak_opening(rows: int) -> int:
        path = str(tmp_path / f"v4_{rows}.db")
        with SQLiteStore(path) as s, s.batch():
            for i in range(rows):
                s.put_claim(claim(predicate=f"p{i}", sources=[f"ep_{i}_{j}"
                                                             for j in range(8)]))
        raw = sqlite3.connect(path)
        raw.execute("DROP TABLE claim_sources")
        raw.execute("PRAGMA user_version = 4")
        raw.commit()
        raw.close()

        tracemalloc.start()
        with SQLiteStore(path) as s:
            _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert len(provenance_of(path)) == rows * 8, "the backfill lost edges"
        return peak

    small = peak_opening(1000)
    large = peak_opening(4000)
    assert large < small * 2, (
        f"peak grew {large / small:.1f}x for 4x the claims — that is materialisation")


def provenance_of(path: str) -> set[tuple[str, str]]:
    with SQLiteStore(path) as s:
        return provenance(s)


# --- Resolved entities ------------------------------------------------------

def test_entities_round_trip_with_their_aliases(store):
    store.put_entity("en_acme", "Acme Corp", ["Acme", "ACME Corporation"], "acme")
    assert store.all_entities("acme") == [
        ("en_acme", "Acme Corp", ("Acme", "ACME Corporation"))]


def test_putting_an_entity_twice_updates_rather_than_duplicates(store):
    store.put_entity("en_acme", "Acme", [], "acme")
    store.put_entity("en_acme", "Acme Corp", ["Acme"], "acme")
    assert store.all_entities("acme") == [("en_acme", "Acme Corp", ("Acme",))]


def test_entities_are_ordered_by_id_not_by_insertion(store):
    """Two processes rebuilding the same resolver must agree on which alias wins when
    two entities claim one, and that ordering has to come out of the data rather than
    out of SQLite's page layout."""
    for name in ("en_c", "en_a", "en_b"):
        store.put_entity(name, name.upper(), [], "acme")
    assert [e[0] for e in store.all_entities("acme")] == ["en_a", "en_b", "en_c"]


def test_one_tenants_entity_resolution_is_not_anothers(store):
    """The whole reason this is scoped: deciding that "Acme" and "Acme Corp" name one
    company is a judgement about one customer's data."""
    store.put_entity("en_acme", "Acme Corp", ["Acme"], "acme")
    assert store.all_entities("other") == []
    assert store.all_entities() == [], "and the default tenant is just another tenant"


def test_entities_survive_a_reopen(tmp_path):
    """Entity ids are baked into the `fact_key`s already on disk, so a mapping that
    evaporated on restart would not merely forget a synonym — it would address a
    different slot and stop seeing the contradiction between two spellings."""
    path = str(tmp_path / "e.db")
    with SQLiteStore(path) as s:
        s.put_entity("en_acme", "Acme Corp", ["Acme"], "acme")
    with SQLiteStore(path) as s2:
        assert s2.all_entities("acme") == [("en_acme", "Acme Corp", ("Acme",))]


def write_v3(path: str) -> None:
    """A store as the previous release left it: no entities table, stamped at 3."""
    with SQLiteStore(path) as s:
        turn(s, content="the kafka pipeline is being sunset")
    raw = sqlite3.connect(path)
    raw.execute("DROP TABLE entities")
    raw.execute("PRAGMA user_version = 3")
    raw.commit()
    raw.close()


def test_a_v3_file_gains_entity_storage_and_is_restamped(tmp_path):
    path = str(tmp_path / "v3.db")
    write_v3(path)
    with SQLiteStore(path) as s:
        assert int(s._db.execute("PRAGMA user_version").fetchone()[0]) == SCHEMA_VERSION
        s.put_entity("en_acme", "Acme Corp", ["Acme"], "acme")
        assert s.all_entities("acme") == [("en_acme", "Acme Corp", ("Acme",))]
        assert len(s.lexical_search_episodes("kafka", [SCOPE], limit=5)) == 1, \
            "and the v3 backfill is idempotent under a second upgrade"


# --- Read concurrency -------------------------------------------------------
#
# WAL lets readers run against a snapshot while a writer holds the write lock, but only
# across separate connections — a connection is where SQLite keeps transaction state.
# Behind one connection and one mutex, which is what this was, a reader waited out the
# whole consolidation sweep.

def test_a_reader_is_not_blocked_by_a_writers_open_transaction(tmp_path, emb):
    path = str(tmp_path / "c.db")
    store = SQLiteStore(path)
    kept = put(store, emb, object="Berlin")

    started, reads_done = threading.Event(), threading.Event()
    seen: list = []

    def reader() -> None:
        started.wait(5)
        seen.append(store.get_claim(kept.id))
        seen.append(store.lexical_search("berlin", [SCOPE], limit=5))
        seen.append(store.stats()["claims"])
        reads_done.set()

    th = threading.Thread(target=reader)
    th.start()
    with store.batch():
        put(store, emb, predicate="uncommitted", object="Lisbon")
        started.set()
        # The assertion is that this returns at all: before, it blocked until the batch
        # committed, which on a real sweep is the whole sweep.
        assert reads_done.wait(5), "a read waited on an open write transaction"
    th.join()

    assert seen[0].id == kept.id
    assert [h[0] for h in seen[1]] == [kept.id]
    assert seen[2] == 1, "and it saw the snapshot, not the half-written transaction"
    store.close()


def test_a_thread_inside_a_batch_reads_its_own_uncommitted_writes(tmp_path, emb):
    """The one bug the snapshot connection could plausibly introduce, and the reason a
    writing thread never uses one: the reconciler asks `competing_claims` mid-write, and
    a snapshot predating the claim written two statements ago reports the slot empty and
    lets the contradiction through."""
    store = SQLiteStore(str(tmp_path / "c.db"))
    with store.batch():
        c = put(store, emb, object="Berlin")
        assert [x.id for x in store.competing_claims("acme", c.fact_key)] == [c.id]
        assert store.get_claim(c.id) is not None
        assert store.stats()["claims"] == 1
    store.close()


def test_an_in_memory_store_shares_its_one_connection(store, emb):
    """`:memory:` is scoped to its connection: a second one would be a second, empty
    database rather than a second view of this one."""
    c = put(store, emb, object="Berlin")
    assert store._reader() is None
    assert store.get_claim(c.id).id == c.id


def test_reader_connections_are_closed_with_the_store(tmp_path, emb):
    """An open reader holds a WAL read mark, which pins the log at the oldest live
    snapshot and stops checkpointing — a forgotten one shows up as a `-wal` file that
    grows without bound rather than as an error."""
    store = SQLiteStore(str(tmp_path / "c.db"))
    c = put(store, emb, object="Berlin")
    store.get_claim(c.id)
    (reader,) = store._readers

    store.close()
    assert store._readers == []
    with pytest.raises(sqlite3.ProgrammingError):
        reader.execute("SELECT 1")
    with pytest.raises(sqlite3.ProgrammingError):
        store.get_claim(c.id)


def test_a_thread_that_arrives_after_close_does_not_get_a_working_reader(tmp_path):
    """Racing a `close()`. The caller must get sqlite's "closed database" error rather
    than a working read against a store that is supposed to be shut."""
    store = SQLiteStore(str(tmp_path / "c.db"))
    store.close()
    assert store._reader() is None
    assert store._readers == []


def test_erase_with_sources_tolerates_a_turn_that_is_already_gone(store, emb):
    """Provenance can dangle: a scope-wide `purge` takes the turn and leaves a claim in
    another scope citing it, and an import can arrive with source ids for turns nobody
    kept. Erasure is the wrong moment to discover that by raising."""
    c = put(store, emb, sources=["ep_never_stored"])
    counts = store.erase_claim(c.id, sources=True)
    assert (counts["claims"], counts["episodes"]) == (1, 0), "no turn, so none erased"
    assert store.get_claim(c.id) is None


def test_readers_and_writers_interleave_without_deadlocking(tmp_path, emb):
    """Three locks now — the write lock, the index lock and the reader registry — and a
    reader takes them in the opposite direction to a writer unless the ordering is
    respected. A deadlock here is a hung process, not a failed assertion, so the join
    has a timeout and the timeout is the test."""
    store = SQLiteStore(str(tmp_path / "c.db"))
    errors: list[BaseException] = []
    stop = threading.Event()

    def writer(n: int) -> None:
        try:
            for i in range(40):
                with store.batch():
                    c = put(store, emb, predicate=f"p_{n}_{i}", object=f"v{i}")
                    store.set_embedding(c.id, emb.encode([c.text])[0])
        except BaseException as e:  # noqa: BLE001 - surfaced via assert below
            errors.append(e)
        finally:
            stop.set()

    def reader() -> None:
        try:
            while not stop.is_set():
                store.candidate_ids([SCOPE])
                store.lexical_search("v1", [SCOPE], limit=5)
                store.vector_search(emb.encode(["v1"])[0], [SCOPE], limit=5)
                store.stats()
        except BaseException as e:  # noqa: BLE001 - surfaced via assert below
            errors.append(e)

    threads = ([threading.Thread(target=writer, args=(n,)) for n in range(3)]
               + [threading.Thread(target=reader) for _ in range(3)])
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
        assert not t.is_alive(), "a thread never came back: lock ordering"

    assert not errors, errors
    assert store.stats()["claims"] == 120
    store.close()


def test_iter_episodes_peak_memory_does_not_scale_with_the_table(tmp_path):
    """`reembed()` exists to walk a store too large to have been embedded correctly the
    first time. A generator that ran `fetchall()` first was a generator in shape only:
    it held every row before yielding one, which turns the recovery path into the thing
    that runs the process out of memory.

    Asserted as a *ratio* rather than an absolute ceiling, because at small sizes the
    fixed cost of one page dominates and any byte threshold would be measuring the page
    rather than the property. Four times the rows through a paging walk costs roughly
    the same peak; through `fetchall()` it costs four times as much.
    """
    import tracemalloc

    def peak_walking(rows: int) -> int:
        store = SQLiteStore(str(tmp_path / f"m{rows}.db"))
        with store.batch():
            for i in range(rows):
                store.add_episode(Episode(content=f"turn {i} " + "x" * 300, scope=SCOPE))
        tracemalloc.start()
        seen = 0
        for _ in store.iter_episodes(SCOPE.tenant):
            seen += 1
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        store.close()
        assert seen == rows, "paging lost rows"
        return peak

    small = peak_walking(2000)
    large = peak_walking(8000)
    assert large < small * 2, (
        f"peak grew {large / small:.1f}x for 4x the rows — that is materialisation, "
        "not paging")


def test_iter_episodes_makes_progress_past_rows_another_tenant_owns(tmp_path):
    """The gap case. A page window can land entirely on rows the filter rejects, and
    an implementation that advanced only by the last row it *yielded* would spin there
    forever."""
    store = SQLiteStore(str(tmp_path / "m.db"))
    with store.batch():
        for i in range(2500):
            store.add_episode(Episode(content=f"other {i}", scope=Scope("other")))
        store.add_episode(Episode(content="ours", scope=SCOPE))

    got = [e.content for e in store.iter_episodes(SCOPE.tenant)]

    assert got == ["ours"]
    store.close()


@pytest.mark.parametrize("table, page", [
    ("episodes", "SELECT rowid AS _rid, * FROM episodes WHERE rowid > ? AND rowid <= ? "
                 "AND +tenant=? ORDER BY rowid LIMIT ?"),
    ("claims", "SELECT rowid AS _rid, * FROM claims WHERE rowid > ? AND rowid <= ? "
               "AND +invalidated_at IS NULL AND +tenant=? ORDER BY rowid LIMIT ?"),
])
def test_a_paged_walk_never_sorts_a_page_back_into_rowid_order(store, table, page):
    """The plus signs in `_iter_rows` are load-bearing and look like typos, so this is
    what stops someone tidying them away.

    Written plainly, `tenant=?` is an indexable term and SQLite prefers a secondary
    index for it — `ep_hash` over episodes, `cl_live` over claims — which leaves the
    rows out of rowid order and forces `USE TEMP B-TREE FOR ORDER BY`. Once per page. So
    the walk sorted the whole matching set again for every page of it, in the method
    whose only purpose is to make a full pass affordable: measured through
    `iter_episodes` over one tenant, 26.7 / 169.1 / 626.6 ms at 5k / 20k / 50k turns,
    against 17.4 / 69.2 / 173.3 with the plus. Note the shape, not just the size — 23x
    for 10x the rows is the quadratic, and it is what `reembed()` walks.
    """
    plan = [r[3] for r in store._db.execute(
        "EXPLAIN QUERY PLAN " + page, [0, 10, "acme", 1000])]

    assert not any("TEMP B-TREE" in step for step in plan), plan
    assert any("INTEGER PRIMARY KEY" in step for step in plan), plan


def test_iter_claims_filters_retired_rows_in_sql_not_in_the_caller(store):
    """`include_invalidated=False` has to be part of the walk. Paging over every row and
    dropping the retired ones afterwards reads the whole table to yield a fraction of
    it, and on a long-lived store most claims are retired — which is the case this
    argument exists for."""
    live = put(store, object="Berlin")
    dead = put(store, predicate="works_at", object="Acme")
    store.invalidate(dead.id, T1, live.id)

    assert [c.id for c in store.iter_claims("acme")] == [live.id]
    assert {c.id for c in store.iter_claims("acme", include_invalidated=True)} == {
        live.id, dead.id}


def test_iter_claims_peak_memory_does_not_scale_with_the_table(tmp_path):
    """The same materialisation `iter_episodes` was fixed for, on the table that has the
    bigger rows. Three callers are built around this streaming and one of them shows what
    it cost: `embed.fingerprint` reads at most 32 claims to learn the store's vector
    width, and a `fetchall()` walk built every `Claim` in the store to hand it those 32 —
    3.06 ms against 64.83 over 20,000 claims, and unbounded memory against one page.

    A ratio, like the episode walk, and for the same reason: at these sizes an absolute
    threshold measures the page rather than the property.
    """
    import tracemalloc

    def peak_probing(rows: int) -> int:
        store = SQLiteStore(str(tmp_path / f"c{rows}.db"))
        with store.batch():
            for i in range(rows):
                store.put_claim(claim(predicate=f"p{i}", text="x" * 300))
        tracemalloc.start()
        seen = 0
        for _ in store.iter_claims(SCOPE.tenant):
            seen += 1
            if seen >= 32:
                break
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        store.close()
        return peak

    small = peak_probing(2000)
    large = peak_probing(8000)
    assert large < small * 2, (
        f"peak grew {large / small:.1f}x for 4x the rows to read the same 32 — that is "
        "materialisation, not paging")


def test_an_old_sqlite_is_refused_at_construction_not_at_the_first_vector_write(monkeypatch):
    """`_vector_upsert` ends in `RETURNING`, which is SQLite 3.35 (March 2021), and every
    `set_embedding` runs it. `requires-python = ">=3.10"` admits interpreters linked
    against 3.31 — so without this check the package installs cleanly, opens a store,
    serves reads, and dies on the first write that touches a vector. Failing at
    construction turns a confusing runtime error into a sentence that names the fix.
    """
    import memvara.store.sqlite as mod

    monkeypatch.setattr(mod.sqlite3, "sqlite_version_info", (3, 31, 1))
    monkeypatch.setattr(mod.sqlite3, "sqlite_version", "3.31.1")
    with pytest.raises(RuntimeError, match=r"3\.35") as exc:
        SQLiteStore(":memory:")
    # The fix is a newer interpreter build, not a newer memvara — say so, because the
    # obvious reading of a version error is "upgrade the library".
    assert "3.31.1" in str(exc.value)
    assert "not a newer memvara" in str(exc.value)


def test_stats_for_one_tenant_does_not_disclose_another_tenants_vector_count(store, emb):
    """`embeddings` used to be exempt from tenant scoping, described as a property of
    the store rather than of a tenant. True of the file on disk, false of the number a
    tenant is handed: two tenants with one claim each were both told `embeddings: 2`, so
    a hosted store leaked its neighbours' write volume through a stats call."""
    for tenant in ("acme", "globex"):
        c = claim(scope=Scope(tenant))
        store.put_claim(claim=c)
        store.set_embedding(c.id, emb.encode([c.text])[0])

    assert store.stats("acme")["embeddings"] == 1
    assert store.stats("globex")["embeddings"] == 1
    # Unfiltered still reports the whole matrix, which is what sizing the store asks.
    assert store.stats()["embeddings"] == 2


def test_a_vector_for_a_claim_that_does_not_exist_is_not_persisted(store, emb):
    """It would be unreachable — every search joins back to `claims` — while still being
    counted by `stats()`, holding a matrix slot forever, and surviving `purge()`, which
    also deletes by joining to the owning table. Pure leak, and reachable:
    `WritePipeline._write_embeddings` catches a dimension error and carries on."""
    store.set_embedding("cl_never_written", emb.encode(["ghost"])[0])
    assert store.stats()["embeddings"] == 0
    assert store.get_embedding("cl_never_written") is None

    # The same guard for episodes, which have their own table and their own join.
    store.set_episode_embedding("ep_never_written", emb.encode(["ghost"])[0])
    assert store.stats()["embeddings"] == 0


def test_every_store_method_is_callable_by_the_protocols_parameter_names():
    """A `Store` is a Protocol, so an implementation diverging on a *parameter name*
    still satisfies `isinstance` and still breaks any caller using a keyword. Found for
    real: `put_claim(self, c)` against the protocol's `put_claim(self, claim)`, so
    `store.put_claim(claim=…)` raised `TypeError` against the reference implementation.
    """
    import inspect

    from memvara.store.base import Store

    def shape(f):
        return [(n, p.kind, p.default is not inspect.Parameter.empty)
                for n, p in inspect.signature(f).parameters.items()]

    diverged = [
        name for name in dir(Store)
        if not name.startswith("_") and callable(getattr(Store, name))
        and shape(getattr(Store, name)) != shape(getattr(SQLiteStore, name))
    ]
    assert diverged == []


def test_a_timestamp_at_the_far_edge_does_not_poison_the_scope_it_lands_in(store, emb):
    """A single accepted write used to break every later read of its scope, permanently.

    `datetime.max.timestamp()` is 253402300800.0, and float64 has no precision left at
    that magnitude — so `datetime(9999,12,31,23,59,59,999999)` rounds *up* onto it and
    `datetime.fromtimestamp` then raises `year 10000 is out of range`. The write returns
    success; `get_all()` and `search()` over that claim raise from then on, and nothing
    points at the row that caused it. Any caller able to write could do it in one call.
    """
    put(store, emb, object="Berlin")
    put(store, emb, object="Acme", predicate="works_at",
        valid_to=datetime(9999, 12, 31, 23, 59, 59, 999999, tzinfo=timezone.utc))

    assert len(list(store.iter_claims("acme"))) == 2
    assert store.lexical_search("berlin", [SCOPE], limit=10)
    assert store.stats("acme")["claims"] == 2


@pytest.mark.parametrize("moment", [
    datetime(1, 1, 1, tzinfo=timezone.utc),
    datetime(1970, 1, 1, tzinfo=timezone.utc),
    datetime(2024, 3, 5, 12, 0, 0, 123456, tzinfo=timezone.utc),
    datetime(9999, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
])
def test_the_clamp_moves_only_the_value_that_could_not_round_trip(moment):
    """The fix must not quietly shift ordinary timestamps — a store that rounds every
    write is worse than one that breaks on a value nobody sends.

    Stated as a property rather than as equality, because *which* values cannot round
    trip is a property of the platform's C library: POSIX reaches year 1 and year 9999,
    Windows rejects everything before 1970 and after 3001. Asserting equality for all
    four passed on POSIX and failed on Windows for two of them — while the clamp was
    doing exactly its job. What must hold everywhere is that a value the platform can
    represent is untouched, and one it cannot is moved to the nearest that it can and is
    still readable."""
    from memvara.store.sqlite import _MAX_TS, _MIN_TS, _dt, _ts

    stored = _ts(moment)
    assert stored is not None
    assert _dt(stored) is not None, "whatever was stored must read back"
    if _MIN_TS <= moment.timestamp() <= _MAX_TS:
        assert _dt(stored) == moment, "a representable moment must not be shifted"
    else:
        assert stored in (_MIN_TS, _MAX_TS), "clamped to a bound, not to something else"


def test_a_turn_no_claim_came_from_can_be_erased(store, emb):
    """`erase_claim(sources=True)` reaches a turn only *through* a claim, so a turn the
    extractor found nothing in — an acknowledgement, a greeting, a script tier 1 does not
    handle — was unreachable by any per-claim erasure and accumulated forever. `purge` is
    far too blunt for a retention rule over raw transcripts."""
    ep = turn(store, emb, content="ok thanks")

    assert store.erase_episode(ep.id) is True
    assert store.get_episode(ep.id) is None
    assert store.erase_episode(ep.id) is False, "erasing twice must not claim success"


def test_erasing_a_cited_turn_is_refused_unless_asked(store, emb):
    """Refusing by default keeps `why()` from resolving to nothing, which is the one
    thing this library promises always works. `cited=True` is the escape hatch a
    transcript-retention obligation needs, and the dangling provenance is then a
    deliberate consequence rather than an accident."""
    ep = turn(store, emb, content="I live in Berlin")
    put(store, emb, sources=[ep.id])

    assert store.erase_episode(ep.id) is False
    assert store.get_episode(ep.id) is not None

    assert store.erase_episode(ep.id, cited=True) is True
    assert store.get_episode(ep.id) is None


# --- portability: what the platform's C library will actually invert ------------------


def test_the_timestamp_ceiling_is_probed_rather_than_assumed(monkeypatch):
    """`_MAX_TS` exists so an accepted write cannot break every later read of its scope,
    and it was hard-coded to the POSIX bound. Windows' CRT refuses anything past roughly
    year 3000 with `OSError: [Errno 22]`, so the same defect was alive there at a lower
    ceiling — CI found it the first time it ran on Windows.

    Simulated here rather than skipped, because the platform this suite usually runs on
    is exactly the one that cannot exercise the branch."""
    import datetime as dtmod
    from memvara.store import sqlite as mod

    real = dtmod.datetime
    ceiling = real(3000, 12, 31, tzinfo=dtmod.timezone.utc).timestamp()

    class NarrowCRT(real):
        @classmethod
        def fromtimestamp(cls, ts, tz=None):
            if ts > ceiling:
                raise OSError(22, "Invalid argument")
            return real.fromtimestamp(ts, tz)

    monkeypatch.setattr(mod, "datetime", NarrowCRT)
    probed = mod._max_roundtrip_ts()

    assert probed <= ceiling, "a bound the platform cannot invert is not a bound"
    # And it is the *largest* such second, not merely a safe one — a clamp that gives up
    # decades of range would silently rewrite ordinary far-future dates.
    assert real.fromtimestamp(probed, dtmod.timezone.utc).year == 3000


def test_the_posix_ceiling_keeps_its_sub_second_headroom():
    """The probe must not cost precision where nothing was wrong. `datetime.max`'s own
    timestamp is unusable — float64 has no precision left at 2.5e11, so it rounds *up*
    onto year 10000, which is the original defect — so the answer is one ulp below it,
    and an implementation that seeded a search from `int()` of that would land a whole
    second lower. It did, before this test."""
    import sys
    from datetime import datetime, timezone
    from memvara.store.sqlite import _MAX_TS

    if sys.platform == "win32":            # its CRT stops at 3001; nothing to preserve
        pytest.skip("no sub-second headroom exists at this platform's ceiling")
    assert datetime.fromtimestamp(_MAX_TS, timezone.utc).year == 9999
    assert _MAX_TS != int(_MAX_TS), "the fractional headroom was rounded away"


def test_positional_file_io_does_not_depend_on_a_posix_only_call(tmp_path):
    """`os.pread`/`os.pwrite` do not exist on Windows at all, so every store with a
    `.vecs` sidecar raised AttributeError there — 95 of the 99 failures in the first CI
    run. One code path rather than a `hasattr` fallback, because a fallback only one
    platform exercises is a fallback nobody tests."""
    import os
    from memvara.store.sqlite import _read_at, _write_at

    path = tmp_path / "m.bin"
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        _write_at(fd, b"HEADER", 0)
        _write_at(fd, b"\0", 63)          # extends by writing the last byte
        assert _read_at(fd, 6, 0) == b"HEADER"
        assert os.fstat(fd).st_size == 64
        # Offsets are absolute: a second read does not continue from the first.
        assert _read_at(fd, 6, 0) == b"HEADER"
    finally:
        os.close(fd)


def test_a_date_before_the_epoch_is_clamped_rather_than_left_unreadable():
    """The lower half of the same defect, and the one that took a real test down.

    Windows' CRT rejects negative timestamps, so every date before 1970 was a write the
    store accepted and could not read back — `test_decay.py` dates a claim 600 years ago
    from an ordinary half-life check and the whole scope became unreadable. POSIX reaches
    year 1, so there was nothing there to find until CI ran somewhere else.

    Simulated for the same reason the ceiling is: this suite usually runs on the platform
    that cannot reach the branch."""
    import datetime as dtmod
    from memvara.store import sqlite as mod

    real = dtmod.datetime

    class NoNegatives(real):
        @classmethod
        def fromtimestamp(cls, ts, tz=None):
            if ts < 0:
                raise OSError(22, "Invalid argument")
            return real.fromtimestamp(ts, tz)

    original = mod.datetime
    try:
        mod.datetime = NoNegatives
        probed = mod._min_roundtrip_ts()
    finally:
        mod.datetime = original

    assert probed == 0.0, "the epoch is the floor when negatives are refused"
    assert real.fromtimestamp(probed, dtmod.timezone.utc).year == 1970

    # A floor that is negative but not the epoch — a CRT reaching 1900 but not year 1.
    # Refusing every negative and refusing only some are different searches, and the
    # second is the one that converges from above.
    cutoff = real(1900, 1, 1, tzinfo=dtmod.timezone.utc).timestamp()

    class Reaches1900(real):
        @classmethod
        def fromtimestamp(cls, ts, tz=None):
            if ts < cutoff:
                raise OSError(22, "Invalid argument")
            return real.fromtimestamp(ts, tz)

    try:
        mod.datetime = Reaches1900
        probed = mod._min_roundtrip_ts()
    finally:
        mod.datetime = original

    # Asserted as a property, not as `== cutoff`: the simulation sits on top of the real
    # CRT, and on a platform whose own floor is *above* 1900 the probe correctly stops
    # there instead. What must hold on every platform is that the answer is not below
    # what was asked for and that it reads back.
    assert probed >= cutoff
    assert real.fromtimestamp(probed, dtmod.timezone.utc) is not None


# --- the protocol two backends have to agree on -----------------------------


@pytest.mark.parametrize("name", ["_state_clause", "_live_clause", "_happened_clause"])
def test_the_clause_builders_match_the_signature_the_protocol_declares(name):
    """`SQLStore` exists so SQLite and Postgres write the same predicate in two
    repositories without drifting. A declaration nothing checks is a comment, and the
    way this drifts is silent: a backend that renamed an argument still runs, still
    returns SQL, and answers a different question.
    """
    import inspect

    declared = inspect.signature(getattr(SQLStore, name))
    assert inspect.signature(getattr(SQLiteStore, name)) == declared
    # Positional-or-keyword throughout: `_live_clause` is called by position inside the
    # store, so a backend that made the axes keyword-only would still satisfy a
    # name-only comparison and then fail at the first call site.
    kinds = [p.kind for p in declared.parameters.values()]
    assert all(k is inspect.Parameter.POSITIONAL_OR_KEYWORD for k in kinds)


@pytest.mark.parametrize("name", [
    "competing_claims", "unended_claims", "adjacent", "candidate_ids", "lexical_search",
    "vector_search", "episode_candidate_ids", "lexical_search_episodes",
    "vector_search_episodes",
])
def test_every_time_travelling_protocol_method_takes_both_axes_and_no_as_of(name):
    """`as_of` survives on the public facade and nowhere below it. A store method that
    kept it would be reinterpreting a caller's single instant as one of two clocks —
    the exact silent-wrong-answer this split exists to remove — and the store is where a
    third-party backend copies its signatures from."""
    import inspect

    for owner in (Store, SQLiteStore):
        params = inspect.signature(getattr(owner, name)).parameters
        assert "as_of" not in params, owner
        for axis in ("valid_at", "known_at"):
            assert params[axis].kind is inspect.Parameter.KEYWORD_ONLY, (owner, axis)
            assert params[axis].default is None


@pytest.mark.parametrize("alias", ["", "c"])
@pytest.mark.parametrize("include_invalidated", [False, True])
def test_the_store_takes_its_liveness_sql_from_the_exported_predicate(
        store, alias, include_invalidated):
    """The predicate has to have exactly one home, and this is what makes the store use
    it rather than keep a private copy that happens to agree today.

    The bind order is asserted here because it is the half `live_predicate` cannot
    enforce: it emits four identical markers, and a backend that bound the belief instant
    onto the world columns would be wrong in a way no `as_of` query can reveal, since
    that call passes the two axes equal and every transposition answers the same.

    Stated in the backend's own paramstyle and units rather than in SQLite's, because
    this file is another implementation's acceptance suite: what has to hold there is
    that its clause *is* the exported predicate carrying its own marker.
    """
    sql, params = store._live_clause(TMID, T1, include_invalidated, alias)

    marker = "?" if "?" in sql else "%s"
    exported = live_predicate(marker, include_invalidated=include_invalidated, alias=alias)
    if getattr(store, "hide_expired", False):
        # A store that keeps `expires_at` ANDs the exported expiry predicate onto the
        # exported state predicate, and binds the wall clock for it last. Both halves come
        # from `memvara.store.base`; neither is a private copy.
        assert sql == f"({exported} AND {unexpired_predicate(marker, alias=alias)})"
    else:
        assert sql == exported
    assert sql.count(marker) == len(params)
    if not include_invalidated:
        # Called as `_live_clause(valid_at=TMID, known_at=T1)`, and T1 is the later of
        # the two — so this reads "belief instant on the belief columns" whether the
        # backend binds epoch floats or timestamps.
        known, valid = params[:2], params[2:]
        assert known[0] == known[1] and valid[0] == valid[1]
        assert known[0] > valid[0], "known, known, valid, valid — and not transposed"


def test_the_liveness_predicate_builds_without_a_store_and_counts_what_stats_counts(
        store, emb):
    """The predicate's third copy was a billing gauge in another repository, sampling a
    tenant through a caller-supplied raw connection — no store instance to borrow a
    clause builder from, so it wrote `invalidated_at IS NULL` by hand and counted every
    superseded version of every slot as live. A store whose one address had changed four
    times billed five, and the step was unfalsifiable from inside the data because
    nothing else in the series moved with it.

    So the exported form has to work with no instance and no bind parameters: a server
    clock substituted straight into the SQL is the shape a sampler on a raw connection
    can actually use. Counted here against `stats()` on the same rows, because the two
    agreeing is the whole point of there being one predicate.
    """
    put(store, emb, object="Berlin", valid_to=T1)      # ended: over, still believed
    put(store, emb, object="Lisbon", predicate="p2")   # live
    gone = put(store, emb, object="Rome", predicate="p3")
    store.invalidate(gone.id, T1, None)                # retired: no longer believed

    # SQLite's clock in the units this schema stores. `now()` is the Postgres spelling;
    # what matters is that it is an expression rather than a marker.
    clock = "CAST(strftime('%s','now') AS REAL)"
    predicate = live_predicate(clock)
    assert "?" not in predicate, "a raw-connection sampler has nothing to bind"

    raw = store._db.execute(f"SELECT COUNT(*) FROM claims WHERE {predicate}").fetchone()[0]

    assert raw == store.stats()["live_claims"] == 1


# --- the three states, as SQL -----------------------------------------------
#
# `live_predicate` is one boolean over three states, so it can name {live} and
# {live, ended, retired} and nothing else. These pin the general form it is now the
# alias of: every subset expressible, the belief floor standing under all of them, and
# the bind order that a diagonal query can never reveal to be wrong.

SUBSETS = [("live",), ("ended",), ("retired",), ("live", "ended"), ("live", "retired"),
           ("ended", "retired"), ("live", "ended", "retired")]


@pytest.mark.parametrize("subset", SUBSETS)
def test_the_state_predicate_names_one_bind_axis_per_marker(store, subset):
    """The half `live_predicate` could only document in prose, and the half with a silent
    failure mode: four identical markers, and a backend that bound the belief instant
    onto the world columns is wrong in a way no `as_of` call can show, because that call
    passes the axes equal and every transposition answers the same.

    So the predicate now carries its own bind order and the store reads it. Checked for
    every subset, since the marker *count* varies with the population asked for — two for
    a pure belief-time question, four for liveness — and a backend that assumed four
    would bind past the end of a shorter one.
    """
    sql, axes = state_predicate("?", states=subset)
    assert sql.count("?") == len(axes)
    assert set(axes) <= {"known", "valid"}
    # Belief markers first, always. That is what makes "known, known, valid, valid"
    # generalise to seven populations rather than being a fact about one of them.
    assert axes == tuple(sorted(axes, key=["known", "valid"].index))

    clause, params = store._state_clause(TMID, T1, subset)
    if getattr(store, "hide_expired", False):
        # The exported expiry predicate is ANDed on, with the wall clock bound last, after
        # every marker the state predicate names.
        assert clause == f"({sql} AND {unexpired_predicate('?')})"
        assert len(params) == len(axes) + 1
        assert params[-1] > T1.timestamp(), "the expiry marker reads the wall clock"
    else:
        assert clause == sql and len(params) == len(axes)
    # T1 is the later instant and is the *belief* one, so this reads "belief instant on
    # the belief markers" without depending on how the backend stores a timestamp.
    for axis, bound in zip(axes, params):
        assert bound == (T1 if axis == "known" else TMID).timestamp(), axis


@pytest.mark.parametrize("subset", SUBSETS)
def test_every_state_subset_keeps_the_belief_floor_bound_to_the_whole_clause(
        store, subset):
    """`AND` binds tighter than `OR`. A subset wanting `retired` beside a world-time state
    is a disjunction, so `floor AND retired OR in_force` parses as
    `(floor AND retired) OR in_force` — the floor gone from half the predicate, and gone
    in the direction that returns rows we had not yet heard of.

    Asserted against the database rather than against the SQL text, because the text
    looks entirely reasonable either way. One claim, recorded after the instant being
    asked about: no population contains it, whatever it is otherwise.
    """
    future = put(store, object="Rome", predicate="p_future", recorded_at=T2,
                 valid_from=T0, valid_to=T1)
    store.invalidate(future.id, T2, None)

    seen = store.candidate_ids([SCOPE], valid_at=T2, known_at=T1, states=subset)
    assert future.id not in seen, subset


@pytest.mark.parametrize("subset", SUBSETS)
def test_the_state_filter_selects_exactly_the_claims_that_report_that_state(
        store, subset):
    """`states=` and `Claim.state` are one vocabulary written in two modules — a SQL
    predicate here and a pair of `is None` tests there. They have to pick the same rows
    or the store can be asked for a population nothing reports being in."""
    put(store, object="Lisbon", predicate="p_live")
    put(store, object="Berlin", predicate="p_ended", valid_to=T1)
    retired = put(store, object="Rome", predicate="p_retired")
    store.invalidate(retired.id, T1, None)

    got = store.get_claims(store.candidate_ids([SCOPE], states=subset))
    assert {c.state for c in got.values()} == set(subset), subset


def test_live_predicate_is_now_the_two_state_alias_and_says_the_same_thing(store):
    """Unchanged, and checked as an identity rather than by eye: the old spelling is the
    new one with a fixed pair of arguments, so it cannot start disagreeing while both
    exist."""
    for flag, subset in ((False, ("live",)), (True, STATES)):
        assert live_predicate("?", include_invalidated=flag) \
            == state_predicate("?", states=subset)[0]
        assert live_predicate("%s", include_invalidated=flag, alias="c") \
            == state_predicate("%s", states=subset, alias="c")[0]


def test_iter_claims_filters_the_stored_state_and_keeps_its_own_default(store):
    """The maintenance walk is not a read at an instant: it pages over rows and has no
    clock, so it filters what `Claim.state` reports.

    Its unflagged view is `("live", "ended")` and must stay so. `reembed()` walks this,
    and narrowing the default to `("live",)` would silently stop re-encoding every
    superseded version in the store — a walk that returns a fraction of what it did,
    from a call whose whole purpose is to visit everything we still believe.
    """
    live = put(store, object="Lisbon", predicate="p_live")
    ended = put(store, object="Berlin", predicate="p_ended", valid_to=T1)
    retired = put(store, object="Rome", predicate="p_retired")
    store.invalidate(retired.id, T1, None)

    def walked(**kw):
        return sorted(c.id for c in store.iter_claims("acme", **kw))

    assert walked(states=["retired"]) == [retired.id]
    assert walked(states=["ended"]) == [ended.id]
    assert walked(states=["live"]) == [live.id]
    assert walked() == walked(include_invalidated=False) == sorted([live.id, ended.id])
    assert walked(include_invalidated=True) == walked(states=STATES) \
        == sorted([live.id, ended.id, retired.id])


@pytest.mark.parametrize("subset", SUBSETS)
def test_the_stored_state_clause_keeps_the_plus_the_paged_walk_depends_on(subset):
    """`_iter_rows` writes every filter as `+column`, which is not decoration: the unary
    plus is what stops the planner taking `cl_live` and re-sorting each page back into
    rowid order, turning a full walk quadratic. A filter handed to it without the plus
    still returns the right rows and makes `reembed()` unrunnable at scale."""
    clause = stored_state_predicate(subset, prefix="+")
    if len(subset) == len(STATES):
        assert clause == "", "the complete set is no filter, not a tautology"
        return
    columns = [w for w in clause.replace("(", " ").split()
               if w.endswith("invalidated_at") or w.endswith("valid_to")]
    assert columns and all(c.startswith("+") for c in columns), clause


def test_a_multi_state_walk_does_not_leak_another_tenants_rows(store):
    """The precedence trap, on the walk instead of the read. A multi-state filter is a
    disjunction and `_iter_rows` `AND`s the tenant on beside it, so an unparenthesised
    clause parses as `stateA OR (stateB AND tenant=?)` — and the first disjunct then
    matches every tenant. Cross-tenant, from a filter that looks like a narrowing."""
    mine = put(store, object="Lisbon", predicate="p_retired")
    store.invalidate(mine.id, T1, None)
    # Live, and in the *other* tenant: it matches the first disjunct, which is the one an
    # unparenthesised clause leaves with no tenant term attached to it.
    put(store, object="Berlin", predicate="p_live", scope=Scope("globex", "bob"))

    walked = list(store.iter_claims("acme", states=["live", "retired"]))
    assert [c.id for c in walked] == [mine.id]


def test_the_stored_state_walk_reports_a_future_closure_as_already_closed(store):
    """Where the two predicates in this family genuinely differ, stated rather than left
    to be discovered. A claim retired *next* October is `retired` to `Claim.state` and
    still believed to a read at now — because one is a property of the row and the other
    is a question about a moment. The walk follows the row."""
    later = put(store, object="Rome", predicate="p_later")
    store.invalidate(later.id, T2, None)          # T2 is after T0/T1, the read instants

    assert [c.id for c in store.iter_claims("acme", states=["retired"])] == [later.id]
    assert store.candidate_ids([SCOPE], valid_at=T1, known_at=T1,
                               states=["retired"]) == []


def test_the_sqlite_store_still_satisfies_the_runtime_store_protocol(store):
    """`SQLStore` is deliberately a *second* protocol: the clause builders are SQL
    generation, and `Store` promises Qdrant and LanceDB can implement it. Folding them
    into `Store` would have changed what `isinstance` accepts, which is why they are not
    there."""
    assert isinstance(store, Store)
    assert not hasattr(Store, "_live_clause")


# --- ties, and the invariant that two stores holding the same data agree ------


def _sardine_store(order):
    """Eight claims that differ only in subject, so a query on the object ties them all."""
    from memvara import Memvara, NullLLM
    from memvara.embed import HashingEmbedder

    mem = Memvara(llm=NullLLM(), embedder=HashingEmbedder(dim=64), user="u")
    for i in order:
        mem.remember(f"Person{i}", "likes", "sardines")
    return mem


def test_bm25_ties_are_broken_on_content_so_ingest_order_cannot_decide_the_page():
    """`ORDER BY s ASC` with no tiebreak made the answer a property of insertion order.

    BM25 ties are not exotic: eight claims differing only in subject score identically
    for a query on the object. With nothing after `s`, SQLite returned them in rowid
    order — and because the `LIMIT` is inside the same statement (design invariant 7),
    that decided *which rows came back at all*, not merely how they were arranged.

    Two stores holding the same eight facts, filled in opposite orders, returned
    disjoint top-3s. Every other tie-break in this codebase is keyed on `value_key` for
    exactly this reason; this one was missed.
    """
    ascending = _sardine_store(range(8))
    descending = _sardine_store(reversed(range(8)))
    try:
        def top3(mem):
            return tuple(mem.store.get_claim(cid).subject for cid, _ in
                         mem.store.lexical_search("sardines", [mem.default_scope], 3))

        assert top3(ascending) == top3(descending), (
            "same data, different ingest order, different answer"
        )
    finally:
        ascending.close()
        descending.close()


def test_equidistant_turns_come_back_in_the_same_order_on_every_file():
    """`episodes_near` broke ties on `id`, and its docstring claimed that was enough.

    An episode id is minted at ingest, not derived from the turn, so `id` is stable
    within one file and means nothing across two. The docstring said the opposite in as
    many words, which is the kind of wrong that stops anybody checking.
    """
    import datetime as dt

    from memvara import Memvara, NullLLM
    from memvara.embed import HashingEmbedder

    at = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)

    def near(order):
        mem = Memvara(llm=NullLLM(), embedder=HashingEmbedder(dim=64), user="u")
        try:
            for word in order:
                mem.add(word, role="user", ts=at)       # identical ts => a real tie
            return tuple(mem.store.get_episode(eid).content for eid, _ in
                         mem.store.episodes_near(at, [mem.default_scope], 3))
        finally:
            mem.close()

    assert near(["alpha", "bravo", "charlie"]) == near(["charlie", "bravo", "alpha"])


def test_omittable_names_every_member_a_backend_may_actually_leave_out():
    """`Store` is `@runtime_checkable`, and `isinstance` on a Protocol is all-or-nothing.

    So it cannot answer "can this store walk a graph" — it asks whether all 44 members
    are present, and a backend that implements everything a memory needs and skips the
    six optional ones is `False`. The capability check in this codebase is therefore
    `getattr` per member, and `OMITTABLE` is the list those call sites are drawn from.
    Asserted here because it is documentation: nothing at runtime reads it, so nothing
    else would notice it going stale.
    """
    from memvara.store.base import OMITTABLE, Store

    # `vars(Store)`, not `Store.__protocol_attrs__`: that attribute arrived in CPython
    # 3.12 and this package supports 3.10. The class dict of a Protocol is its declared
    # members, on every version, and it agrees with `__protocol_attrs__` exactly where
    # both exist.
    members = {name for name in vars(Store) if not name.startswith("_")}

    assert set(OMITTABLE) <= members, (
        "OMITTABLE names something that is not on the protocol at all"
    )
    assert "get_claims" in OMITTABLE, (
        "get_claims was added after the first third-party backends existed and "
        "tests/test_edges.py pins that those keep working — optional by compatibility "
        "rather than by design, which is still optional"
    )
    # Both shipped stores implement everything, optional members included — which is why
    # the isinstance trap is invisible in this repository and waiting for the first
    # third-party backend.
    from memvara.store import SQLiteStore
    from memvara.store.remote import RemoteStore
    for cls in (SQLiteStore, RemoteStore):
        assert not [m for m in members if not hasattr(cls, m)]


# --- the predicate graph declaration ---------------------------------------------------


_V9_PREDICATES = """
CREATE TABLE predicates (
    tenant TEXT NOT NULL, name TEXT NOT NULL, cardinality TEXT NOT NULL,
    volatility TEXT NOT NULL, memory_type TEXT NOT NULL,
    aliases TEXT NOT NULL DEFAULT '[]', supersedes TEXT NOT NULL DEFAULT '[]',
    learned INTEGER NOT NULL DEFAULT 1, PRIMARY KEY (tenant, name));
INSERT INTO predicates VALUES ('t','works_at','one','slow','semantic','[]','[]',1);
PRAGMA user_version = 9;
"""


def test_a_graph_declaration_survives_a_restart(tmp_path):
    """The reason these columns exist at all.

    `put_spec` persists whatever spec it is handed, and a *declared* predicate reaches it
    whenever an alias is learned for one. Rehydration only protects a declared spec from a
    persisted *learned* one, so a declared spec written back without its graph fields
    would be reloaded as non-traversable and would overwrite the declaration. The store
    would stop walking edges it walked yesterday, with no error and nothing in the file
    saying why. Asserted on the whole spec rather than field by field, so a seventh field
    added later cannot be forgotten here.
    """
    from memvara.schema import Cardinality, PredicateSpec, Volatility

    path = str(tmp_path / "specs.db")
    spec = PredicateSpec("depends_on", Cardinality.MANY, Volatility.SLOW,
                         subject_type=("project",), object_type=("software",), graph=True,
                         inverse="depended_on_by", inverse_cardinality=Cardinality.MANY,
                         traversal_cost=0.5)
    first = SQLiteStore(path)
    first.put_spec(spec, "t")
    first.close()

    second = SQLiteStore(path)
    try:
        assert [s for s in second.all_specs("t") if s.name == "depends_on"] == [spec]
    finally:
        second.close()


def test_a_spec_with_no_inverse_round_trips_as_none(tmp_path):
    """The nullable half. Stored as SQL NULL rather than an empty string, because an
    empty inverse and no inverse would otherwise be two spellings of one state."""
    from memvara.schema import PredicateSpec

    path = str(tmp_path / "plain.db")
    spec = PredicateSpec("version", object_type=("value",), learned=True)
    first = SQLiteStore(path)
    first.put_spec(spec, "t")
    first.close()

    second = SQLiteStore(path)
    try:
        back, = [s for s in second.all_specs("t") if s.name == "version"]
        assert back == spec
        assert back.inverse is None and back.inverse_cardinality is None
        assert back.objects_are_entities is False
    finally:
        second.close()


def test_a_version_9_file_gains_the_columns_and_keeps_its_rows(tmp_path):
    """Nothing is backfilled, and that is not an omission: these columns are declared by a
    vocabulary rather than derived from anything the row already holds, so no function of
    the existing columns could fill them. The defaults say "takes values, walks nowhere",
    which is what an undeclared predicate means and the safe reading for a row whose pack
    is no longer loaded."""
    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.executescript(_V9_PREDICATES)
    conn.commit()
    conn.close()

    store = SQLiteStore(path)
    try:
        columns = {r["name"] for r in store._db.execute("PRAGMA table_info(predicates)")}
        assert {"subject_type", "object_type", "graph", "inverse",
                "inverse_cardinality", "traversal_cost"} <= columns
        assert int(store._db.execute("PRAGMA user_version").fetchone()[0]) == SCHEMA_VERSION

        survivor, = [s for s in store.all_specs("t") if s.name == "works_at"]
        assert survivor.learned is True
        assert survivor.graph is False
        assert survivor.objects_are_entities is False
        assert survivor.traversal_cost == 1.0
    finally:
        store.close()


def test_the_migration_is_a_no_op_on_a_fresh_file(tmp_path):
    """Shape-driven like every migration here, so a brand-new database that already has
    the columns from the schema passes through untouched. Running it twice is the test:
    a second ALTER TABLE for a column that exists would raise."""
    path = str(tmp_path / "fresh.db")
    store = SQLiteStore(path)
    try:
        store._migrate_to_v10()
        store._migrate_to_v10()
        columns = {r["name"] for r in store._db.execute("PRAGMA table_info(predicates)")}
        assert "traversal_cost" in columns
    finally:
        store.close()


# --- object kind, and the graph edges that depend on it --------------------------------


def _v10_claims_ddl() -> str:
    """The claims table as version 10 shaped it: today's, minus `object_kind`.

    Derived from `SCHEMA` rather than pasted, so this cannot drift into testing a table
    no version of memvara ever wrote. `ALTER TABLE ... DROP COLUMN` is not an option: it
    re-parses the stored DDL, and dropping the last column leaves the comment above it
    dangling, which SQLite rejects as incomplete input.
    """
    from memvara.store import sqlite as sq

    body = sq.SCHEMA.split("CREATE TABLE IF NOT EXISTS claims (", 1)[1]
    body = body.split("\n);", 1)[0]
    kept = [ln for ln in body.splitlines()
            if "object_kind" not in ln and "Version 11" not in ln
            and not ln.strip().startswith("-- classification rule")]
    # Drop the rest of the version-11 comment block and the now-trailing comma.
    kept = [ln for ln in kept if not ln.strip().startswith("--")
            or "Version 9" in ln or "answer for them" in ln]
    text = "\n".join(kept).rstrip().rstrip(",")
    return f"CREATE TABLE claims ({text}\n);"


def test_a_version_10_file_gains_object_kind_and_keeps_its_claims(tmp_path):
    """The upgrade must not switch off a graph that was walking yesterday.

    Nothing backfills the column, and nothing could: the kind is read from the predicate's
    declared object_type, and which vocabulary a deployment loads is environment rather
    than data, so a backfill would make two machines disagree about one file. The claim
    written before the rule therefore keeps `object_kind IS NULL`, and `_WALKABLE` admits
    it.
    """
    path = str(tmp_path / "v10.db")
    conn = sqlite3.connect(path)
    conn.executescript(_v10_claims_ddl())
    conn.execute("PRAGMA user_version = 10")
    conn.commit()
    conn.close()

    store = SQLiteStore(path)
    try:
        columns = {r["name"] for r in store._db.execute("PRAGMA table_info(claims)")}
        assert "object_kind" in columns
        assert int(store._db.execute("PRAGMA user_version").fetchone()[0]) == SCHEMA_VERSION
    finally:
        store.close()


def test_the_object_kind_migration_is_a_no_op_on_a_fresh_file(tmp_path):
    """Running it twice is the test: a second ALTER for a column that exists would raise."""
    store = SQLiteStore(str(tmp_path / "fresh.db"))
    try:
        store._migrate_to_v11()
        store._migrate_to_v11()
        columns = {r["name"] for r in store._db.execute("PRAGMA table_info(claims)")}
        assert "object_kind" in columns
    finally:
        store.close()


def test_a_value_object_is_not_a_graph_edge_but_an_unclassified_one_still_is(store):
    """`_WALKABLE` and `GraphTraverser._edges` have to agree, and this pins the SQL half.

    Three claims that are otherwise identical in shape: one classified as an entity, one
    as a value, one written before the rule existed. Only the value is refused.
    """
    from memvara.types import ObjectKind

    def put(cid, subj, obj, kind):
        c = claim(id=cid, subject=subj, predicate="depends_on", object=obj)
        c.object_kind = kind
        store.put_claim(c)

    put("cl_ent", "alpha", "beta", ObjectKind.ENTITY)
    put("cl_val", "gamma", "delta", ObjectKind.VALUE)
    put("cl_old", "epsilon", "zeta", None)

    walkable = {
        r["id"] for r in store._db.execute(
            "SELECT id FROM claims WHERE " + _WALKABLE_SQL.format(a="claims"))
    }
    assert "cl_ent" in walkable
    assert "cl_old" in walkable, "a claim written before the rule keeps its edges"
    assert "cl_val" not in walkable


def test_a_version_11_file_gains_project_on_claims_and_episodes(tmp_path):
    """Episodes are scoped too, and were missed on the first pass.

    Without the column on `episodes`, a turn recorded in one repository would be readable
    from every other, so the claims half of this migration would be enforced while the
    evidence behind those claims leaked across projects.
    """
    path = str(tmp_path / "v11.db")
    store = SQLiteStore(path)
    try:
        for table in ("claims", "episodes"):
            store._db.execute(f"ALTER TABLE {table} RENAME COLUMN project TO gone")
        for col in ("subject_type", "object_type"):
            store._db.execute(f"ALTER TABLE claims RENAME COLUMN {col} TO gone_{col}")
        store._migrate_to_v12()
        for table in ("claims", "episodes"):
            columns = {r["name"] for r in store._db.execute(f"PRAGMA table_info({table})")}
            assert "project" in columns, table
        columns = {r["name"] for r in store._db.execute("PRAGMA table_info(claims)")}
        assert {"subject_type", "object_type"} <= columns
    finally:
        store.close()


def test_the_v12_migration_re_folds_a_typed_key_and_both_hashes_that_read_it(store):
    """The half of version 12 that rewrites rows rather than adding columns.

    An entity identity now keeps its namespace, so a claim written before this version
    holds `company apple` where the same text now folds to `company:apple`. Three derived
    values read that key, and leaving any of them stale is a store that looks like it
    works:

    * `subject_key` is what traversal joins on, so a stale one is an entity that has
      quietly split in two.
    * `fact_key` decides what contradicts what, so a stale one stops contradicting
      anything.
    * `value_key` decides what counts as the same assertion, so a stale one turns the
      next re-statement of a known fact into a rival value instead of a reinforcement.

    The claim is written through the normal path and then forced back to the old shape,
    rather than being hand-built, so what the migration repairs is the shape this
    repository actually used to produce.
    """
    claim = put(store, subject="company:Apple Inc.", predicate="founded_in",
                object="fruit:apple")
    store._db.execute(
        "UPDATE claims SET subject_key = 'company apple', object_key = 'fruit apple', "
        "fact_key = 'stale-fact', value_key = 'stale-value', "
        "subject_type = '', object_type = ''"
    )

    store._migrate_to_v12()

    row = store._db.execute(
        "SELECT subject_key, object_key, fact_key, value_key, subject_type, object_type "
        "FROM claims WHERE id = ?", (claim.id,)).fetchone()
    fresh = store.get_claim(claim.id)
    assert fresh is not None
    assert row["subject_key"] == fresh.subject_key == "company:apple"
    assert row["object_key"] == fresh.object_key == "fruit:apple"
    assert row["fact_key"] == fresh.fact_key
    assert row["value_key"] == fresh.value_key
    assert (row["subject_type"], row["object_type"]) == ("company", "fruit")


def test_the_v12_migration_leaves_the_same_store_when_it_runs_twice(store):
    """Idempotent, like every migration here, and this one has to be checked.

    It re-derives keys from text and then hashes the keys, so a second pass reads its own
    output. That is the shape where a migration that is not idempotent corrupts rather
    than merely wasting time.
    """
    put(store, subject="company:Apple Inc.", predicate="founded_in", object="fruit:apple")
    store._migrate_to_v12()
    once = store._db.execute(
        "SELECT subject_key, object_key, fact_key, value_key, subject_type, object_type "
        "FROM claims").fetchall()
    store._migrate_to_v12()
    twice = store._db.execute(
        "SELECT subject_key, object_key, fact_key, value_key, subject_type, object_type "
        "FROM claims").fetchall()
    assert [tuple(r) for r in once] == [tuple(r) for r in twice]


def test_the_sql_rehash_agrees_with_the_python_slot_key():
    """The property the v12 migration lives or dies on.

    It recomputes every `fact_key` in SQL rather than loading each claim, so if the two
    derivations disagreed by a byte, every migrated claim would address a slot nothing
    else computes — and would silently stop contradicting anything.
    """
    from memvara.store.sqlite import _fact_key_of
    from memvara.types import OWNER_SEP, Scope, fact_key_for

    for project in ("gh/o/cloud", None):
        scope = Scope("t", "alice", project=project)
        packed = "postgresql" + OWNER_SEP + "version"
        assert fact_key_for(scope, "postgresql", "version") == _fact_key_of(
            "t", "alice", project, packed), project


def test_the_sql_rehash_agrees_with_the_python_value_key():
    """The same property for `value_key`, which the same migration also recomputes.

    Checked separately from `fact_key` because it hashes a different list of parts, and a
    transposition inside that list would leave both derivations wrong in the same way
    only if both were written from one place — which they are not.
    """
    from memvara.store.sqlite import _value_key_of
    from memvara.types import OWNER_SEP, Claim, Scope

    claim = Claim(scope=Scope("t", "alice"), subject="company:Apple Inc.",
                  predicate="founded_in", object="fruit:apple")
    packed = OWNER_SEP.join((claim.subject_key, claim.predicate, claim.object_key,
                             str(claim.polarity)))
    assert claim.value_key == _value_key_of("t", "alice", packed)


def test_enumeration_and_id_reads_agree_about_the_project(store):
    """A read that lists claims filters by project, exactly as a read by id does.

    These two paths answer the same question through different code. `sees()` authorizes
    an id-addressed read by comparing `Scope.key()`, which counted the project from the
    moment the field existed. The SQL every enumerating read shares did not, so a handle
    opened on one repository listed another repository's claims while `get()` on the very
    same id refused them — the permissive answer being the one that returns rows.

    The global claim in the middle is the other half of the rule and is why this cannot be
    tested by asserting that a foreign project contributes nothing: a claim written with
    no project must stay visible from inside every project, because visibility widens
    upward.
    """
    for project, obj in (("gh/o/a", "17"), ("gh/o/b", "16")):
        put(store, scope=Scope("t", "alice", project=project),
            subject="postgresql", predicate="version", object=obj)
    put(store, scope=Scope("t", "alice"), subject="user", predicate="prefers",
        object="dark mode")

    here = Scope("t", "alice", project="gh/o/a")
    listed = [store.get_claim(i) for i in store.candidate_ids(here.ancestors())]
    assert sorted((c.subject, c.object) for c in listed) == [
        ("postgresql", "17"), ("user", "dark mode")]

    there = Scope("t", "alice", project="gh/o/b")
    foreign = [store.get_claim(i) for i in store.candidate_ids(there.ancestors())]
    sixteen = [c for c in foreign if c.object == "16"]
    assert len(sixteen) == 1
    assert not here.sees(sixteen[0].scope)



def test_occupied_slots_names_the_slots_that_hold_a_live_claim(store):
    """The batched form of `count_competing() > 0`, which is what read-side shadowing
    asks for every candidate slot at once."""
    from memvara.types import Claim, Scope, close_out, utcnow
    scope = Scope("t", "alice")
    live = Claim(subject="user", predicate="editor", object="vim", scope=scope)
    gone = Claim(subject="user", predicate="shell", object="zsh", scope=scope)
    other = Claim(subject="user", predicate="pager", object="less", scope=Scope("u", "bob"))
    for claim in (live, gone, other):
        store.put_claim(claim)
    close_out(gone, utcnow(), None, "retired")
    store.put_claim(gone)
    keys = [live.fact_key, gone.fact_key, other.fact_key, "absent"]
    assert store.occupied_slots("t", keys) == {live.fact_key}
    assert store.occupied_slots("t", []) == set()
    many = [f"k{n}" for n in range(1500)] + [live.fact_key]
    assert store.occupied_slots("t", many) == {live.fact_key}, "chunked past SQLite's limit"
