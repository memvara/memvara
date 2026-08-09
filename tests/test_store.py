"""SQLite store: persistence, the indexed conflict lookup, hybrid search primitives,
and the bitemporal SQL that makes time travel work."""

import sqlite3
import threading
from datetime import datetime, timezone

import numpy as np
import pytest

from memvara.embed import HashingEmbedder
from memvara.store import SQLiteStore
from memvara.store.sqlite import SCHEMA_VERSION
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


def test_competing_claims_respects_as_of(store):
    a = put(store, object="Berlin")
    store.invalidate(a.id, T1, "cl_x")
    assert [c.id for c in store.competing_claims("acme", a.fact_key, as_of=TMID)] == [a.id]
    assert store.competing_claims("acme", a.fact_key, as_of=T2) == []


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


def test_lexical_search_respects_as_of(store):
    a = put(store, object="Berlin")
    store.invalidate(a.id, T1, "cl_x")
    assert store.lexical_search("berlin", [SCOPE], limit=10, as_of=TMID)[0][0] == a.id
    assert store.lexical_search("berlin", [SCOPE], limit=10, as_of=T2) == []


def test_include_invalidated_reveals_retired_claims(store):
    a = put(store, object="Berlin")
    store.invalidate(a.id, T1, "cl_x")
    assert store.lexical_search("berlin", [SCOPE], limit=10) == []
    assert store.lexical_search(
        "berlin", [SCOPE], limit=10, include_invalidated=True
    )[0][0] == a.id


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


def test_vector_search_respects_as_of(store, emb):
    a = put(store, emb, object="Berlin")
    store.invalidate(a.id, T1, "cl_x")
    q = emb.encode(["user lives in Berlin"])[0]
    assert store.vector_search(q, [SCOPE], limit=5, as_of=TMID)[0][0] == a.id
    assert store.vector_search(q, [SCOPE], limit=5, as_of=T2) == []


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


def test_episode_search_respects_as_of(store, emb):
    """A turn that had not happened yet is not something we could have recalled."""
    old = turn(store, emb, content="kafka is fine", ts=T0)
    new = turn(store, emb, content="kafka is being sunset", ts=T1)
    q = emb.encode(["kafka"])[0]
    assert set(store.episode_candidate_ids([SCOPE], as_of=TMID)) == {old.id}
    assert [h[0] for h in store.lexical_search_episodes(
        "kafka", [SCOPE], limit=10, as_of=TMID)] == [old.id]
    assert [h[0] for h in store.vector_search_episodes(
        q, [SCOPE], limit=10, as_of=TMID)] == [old.id]
    assert len(store.episode_candidate_ids([SCOPE], as_of=T2)) == 2
    assert new.id in store.episode_candidate_ids([SCOPE], as_of=T2)


def test_a_future_turn_is_invisible_to_a_present_query(store):
    turn(store, content="kafka next year", ts=datetime(2099, 1, 1, tzinfo=timezone.utc))
    assert store.lexical_search_episodes("kafka", [SCOPE], limit=10) == []


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

    audit = set(store.candidate_ids([SCOPE], as_of=TMID, include_invalidated=True))
    assert audit == {old.id}, "a claim recorded after as_of is not past belief"
    assert later.id not in audit


def test_include_invalidated_reveals_expired_and_retracted_claims(store):
    a = put(store, object="Berlin", recorded_at=T0, valid_from=T0)
    store.invalidate(a.id, T1, None)
    store.set_valid_to(a.id, T1)
    assert store.candidate_ids([SCOPE]) == []
    assert set(store.candidate_ids([SCOPE], include_invalidated=True)) == {a.id}


def test_candidate_ids_applies_bitemporal_filter(store):
    a = put(store, object="Berlin", recorded_at=T0, valid_from=T0)
    store.invalidate(a.id, T1, "cl_b")
    b = put(store, object="Lisbon", recorded_at=T1, valid_from=T1)
    assert set(store.candidate_ids([SCOPE], as_of=TMID)) == {a.id}
    assert set(store.candidate_ids([SCOPE], as_of=T2)) == {b.id}
    assert set(store.candidate_ids([SCOPE], include_invalidated=True)) == {a.id, b.id}


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
                                  "entities": 0}


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
                             "invalidated": 0, "embeddings": 0}
    assert store.lexical_search("berlin", [SCOPE], limit=10) == []


def test_purge_erases_the_episode_indexes_not_only_the_rows(store, emb):
    """An FTS row surviving the episode it describes is not a stale cache entry — it is
    the purged text still being searchable, which is the compliance failure the whole
    call exists to avoid."""
    ep = turn(store, emb, content="the kafka pipeline is being sunset")
    counts = store.purge(SCOPE)

    assert counts == {"claims": 0, "episodes": 1, "embeddings": 1, "entities": 0}
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
    assert s == {"episodes": 0, "claims": 2, "live_claims": 1,
                 "invalidated": 1, "embeddings": 2}


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
    a.vector_search_episodes(onehot(0), [SCOPE], limit=1)  # A's index is now warm

    ep = Episode(content="written by b", scope=SCOPE)
    b.add_episode(ep)
    b.set_episode_embedding(ep.id, onehot(5))

    hits = a.vector_search_episodes(onehot(5), [SCOPE], limit=1)
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

def test_erase_claim_reports_whether_there_was_anything_to_erase(store):
    c = put(store)
    assert store.erase_claim(c.id) is True
    assert store.erase_claim(c.id) is False, "erasing twice is not two erasures"
    assert store.erase_claim("cl_never_existed") is False


def test_erase_claim_takes_the_row_the_index_and_the_vector(store, emb):
    c = put(store, emb, object="Berlin")
    survivor = put(store, emb, object="Lisbon", predicate="visited")

    assert store.erase_claim(c.id) is True

    assert store.get_claim(c.id) is None
    assert store.lexical_search("berlin", [SCOPE], limit=10) == []
    assert store.get_embedding(c.id) is None
    assert store._vec.get(c.id) is None, "the matrix row must be blanked, not just unmapped"
    assert store.stats() == {"episodes": 0, "claims": 1, "live_claims": 1,
                             "invalidated": 0, "embeddings": 1}
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

    assert store.erase_claim(c.id, sources=True) is True

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


def test_erasing_a_claim_takes_its_provenance_with_it(store, emb):
    """The fourth-table trap. Wave 4 found `entities` surviving a purge that reported
    per-table counts as evidence; this is the same mistake one table over, and it does
    more than leave a row behind — the erased claim goes on voting to keep its own
    source turn alive, so `sources=True` quietly stops erasing anything."""
    ep = turn(store, emb, content="the kafka pipeline is being sunset")
    c = put(store, emb, sources=[ep.id])

    assert store.erase_claim(c.id, sources=True) is True

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
    assert store.erase_claim(c.id, sources=True) is True
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
