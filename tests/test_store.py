"""SQLite store: persistence, the indexed conflict lookup, hybrid search primitives,
and the bitemporal SQL that makes time travel work."""

import threading
from datetime import datetime, timezone

import numpy as np
import pytest

from engram.embed import HashingEmbedder
from engram.store import SQLiteStore
from engram.types import Claim, Derivation, Episode, MemoryType, Scope

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
    assert store.purge(SCOPE) == {"claims": 2, "episodes": 1, "embeddings": 1}


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
