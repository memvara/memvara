"""Edge cases, error paths, and adversarial input.

Every test here targets a branch the behavioral suites don't reach: dimension
mismatches, transaction rollback, malformed model output, capacity growth, cache
eviction, and input designed to break a parser. These are the paths that only run when
something has already gone wrong, which is exactly when they must not make it worse.
"""

import sqlite3
import threading
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from engram import (
    CachedEmbedder,
    Claim,
    Engram,
    Episode,
    HashingEmbedder,
    MemoryType,
    NullLLM,
    PredicateRegistry,
    Scope,
    SQLiteStore,
)
from engram.schema import Cardinality, PredicateSpec, Volatility
from engram.store.sqlite import _VecIndex
from engram.types import Explanation, utcnow

TZ = timezone.utc
PAST = datetime(2020, 1, 1, tzinfo=TZ)
FUTURE = datetime(2090, 1, 1, tzinfo=TZ)
SCOPE = Scope("acme", "alice")


def claim(**kw) -> Claim:
    base = dict(subject="user", predicate="lives_in", object="Berlin", scope=SCOPE)
    base.update(kw)
    return Claim(**base)


# =============================================================================
# Bitemporal edge cases
# =============================================================================

def test_a_fact_not_yet_true_is_not_live():
    """Valid-time start in the future: recorded, believed, but not yet in force."""
    c = claim(recorded_at=PAST, valid_from=FUTURE)
    assert not c.is_live(utcnow())
    assert c.is_live(FUTURE + timedelta(days=1))


def test_scheduled_future_fact_is_invisible_to_present_queries():
    with Engram(embedder=HashingEmbedder(dim=64), user="alice") as mem:
        mem.remember("user", "lives_in", "Mars", valid_from=FUTURE)
        assert mem.get_all() == []
        assert mem.search("Mars") == []


def test_explanation_summary_includes_rerank_when_present():
    e = Explanation(vector_rank=1, vector_score=0.5, rerank_score=0.87, final_score=0.9)
    assert "rerank=0.870" in e.summary()


def test_explanation_summary_omits_retrievers_that_did_not_fire():
    e = Explanation(lexical_rank=0, lexical_score=3.0, final_score=0.5)
    s = e.summary()
    assert "bm25" in s and "vector" not in s


# =============================================================================
# Predicate registry
# =============================================================================

def test_superseded_by_exposes_cross_predicate_links():
    reg = PredicateRegistry(specs=(
        PredicateSpec("unemployed", Cardinality.ONE, Volatility.SLOW,
                      supersedes=("works_at",)),
    ))
    assert reg.superseded_by("unemployed") == ("works_at",)
    assert reg.superseded_by("lives_in") == ()


def test_cross_predicate_supersession_actually_retires_the_other_slot():
    """The `supersedes` path had a latent bug: it rebuilt the fact key by hand and so
    looked up a key nothing was stored under."""
    reg = PredicateRegistry(specs=(
        PredicateSpec("works_at", Cardinality.ONE, Volatility.SLOW),
        PredicateSpec("unemployed", Cardinality.ONE, Volatility.SLOW,
                      supersedes=("works_at",)),
    ))
    with Engram(embedder=HashingEmbedder(dim=64), registry=reg, user="alice") as mem:
        mem.remember("user", "works_at", "Acme")
        mem.remember("user", "unemployed", "true")
        live = {(c.predicate, c.object) for c in mem.get_all()}
        assert ("works_at", "Acme") not in live, "asserting unemployment must retire the job"
        assert ("unemployed", "true") in live


def test_cross_predicate_supersession_does_not_cross_users():
    reg = PredicateRegistry(specs=(
        PredicateSpec("works_at", Cardinality.ONE, Volatility.SLOW),
        PredicateSpec("unemployed", Cardinality.ONE, Volatility.SLOW,
                      supersedes=("works_at",)),
    ))
    with Engram(embedder=HashingEmbedder(dim=64), registry=reg) as mem:
        mem.remember("user", "works_at", "Acme", user="alice")
        mem.remember("user", "unemployed", "true", user="bob")
        assert [c.object for c in mem.get_all(user="alice")] == ["Acme"]


# =============================================================================
# Embedders
# =============================================================================

def test_cached_embedder_evicts_when_full():
    inner = HashingEmbedder(dim=16)
    cache = CachedEmbedder(inner, max_items=4)
    for i in range(10):
        cache.encode([f"text {i}"])
    assert len(cache._cache) <= 4


def test_cached_embedder_reports_hits_and_misses():
    cache = CachedEmbedder(HashingEmbedder(dim=16), max_items=100)
    cache.encode(["a", "b"])
    cache.encode(["a", "b", "c"])
    assert cache.misses == 3 and cache.hits == 2


def test_cached_embedder_returns_identical_vectors_on_hit():
    cache = CachedEmbedder(HashingEmbedder(dim=16))
    first = cache.encode(["stable text"])
    assert np.array_equal(first, cache.encode(["stable text"]))


def test_hashing_embedder_is_deterministic_across_instances():
    a = HashingEmbedder(dim=32).encode(["hello world"])
    b = HashingEmbedder(dim=32).encode(["hello world"])
    assert np.array_equal(a, b)


def test_hashing_embedder_returns_unit_vectors():
    v = HashingEmbedder(dim=32).encode(["some text here"])[0]
    assert np.isclose(np.linalg.norm(v), 1.0)


def test_hashing_embedder_handles_empty_and_unicode():
    out = HashingEmbedder(dim=32).encode(["", "   ", "🙂🙂", "日本語テキスト"])
    assert out.shape == (4, 32)
    assert np.isclose(np.linalg.norm(out[0]), 0.0), "empty text has no direction"


def test_embedder_batch_shape_matches_input_length():
    assert HashingEmbedder(dim=8).encode([f"t{i}" for i in range(37)]).shape == (37, 8)


# =============================================================================
# NullLLM
# =============================================================================

def test_null_llm_classifies_conservatively():
    """Unknown predicates must default to multi-valued: keeping two facts degrades
    ranking, dropping a true one destroys information."""
    spec = NullLLM().classify_predicate("some_new_predicate", "example")
    assert spec["cardinality"] == "many"
    assert spec["volatility"] == "slow"
    assert spec["memory_type"] == "semantic"


def test_null_llm_extracts_nothing():
    assert NullLLM().extract([Episode(content="I live in Berlin")], []) == []


# =============================================================================
# Vector index internals
# =============================================================================

def test_vec_index_rejects_mismatched_dimension_on_add():
    idx = _VecIndex()
    idx.add("a", np.ones(8, dtype=np.float32))
    with pytest.raises(ValueError, match="dim"):
        idx.add("b", np.ones(9, dtype=np.float32))


def test_vec_index_rejects_mismatched_query_dimension():
    idx = _VecIndex()
    idx.add("a", np.ones(8, dtype=np.float32))
    with pytest.raises(ValueError, match="query dim"):
        idx.search(np.ones(3, dtype=np.float32), ["a"], 5)


def test_vec_index_grows_past_initial_capacity():
    """Initial capacity is 256; growth must preserve every prior vector."""
    idx = _VecIndex()
    for i in range(700):
        v = np.zeros(4, dtype=np.float32)
        v[i % 4] = float(i + 1)
        idx.add(f"c{i}", v)
    assert len(idx) == 700
    probe = np.array([1, 0, 0, 0], dtype=np.float32)
    hits = idx.search(probe, [f"c{i}" for i in range(700)], 3)
    assert len(hits) == 3
    assert all(h[0].startswith("c") for h in hits)


def test_vec_index_search_with_no_known_ids_returns_empty():
    idx = _VecIndex()
    idx.add("a", np.ones(4, dtype=np.float32))
    assert idx.search(np.ones(4, dtype=np.float32), ["unknown", "also_unknown"], 5) == []


def test_vec_index_search_on_empty_index_returns_empty():
    assert _VecIndex().search(np.ones(4, dtype=np.float32), ["a"], 5) == []


def test_vec_index_get_returns_none_for_unknown():
    assert _VecIndex().get("nope") is None


def test_vec_index_readd_overwrites_rather_than_duplicating():
    idx = _VecIndex()
    idx.add("a", np.array([1, 0], dtype=np.float32))
    idx.add("a", np.array([0, 1], dtype=np.float32))
    assert len(idx) == 1
    assert np.allclose(idx.get("a"), [0, 1])


# =============================================================================
# Store: embeddings, transactions, corruption
# =============================================================================

def test_get_embedding_returns_normalized_vector_from_cache_and_disk(tmp_path):
    path = str(tmp_path / "e.db")
    emb = HashingEmbedder(dim=32)
    s1 = SQLiteStore(path)
    c = claim()
    s1.put_claim(c)
    s1.set_embedding(c.id, emb.encode([c.text])[0] * 7.0)  # deliberately non-unit
    cached = s1.get_embedding(c.id)
    s1.close()

    s2 = SQLiteStore(path)
    from_disk = s2.get_embedding(c.id)
    s2.close()

    assert np.isclose(np.linalg.norm(cached), 1.0)
    assert np.isclose(np.linalg.norm(from_disk), 1.0)
    assert np.allclose(cached, from_disk), "cache and disk must agree"


def test_get_embedding_returns_none_when_never_embedded():
    with SQLiteStore(":memory:") as s:
        c = claim()
        s.put_claim(c)
        assert s.get_embedding(c.id) is None


def test_store_closes_itself_as_a_context_manager(tmp_path):
    path = str(tmp_path / "cm.db")
    with SQLiteStore(path) as s:
        s.put_claim(claim())
    with SQLiteStore(path) as s2:
        assert s2.stats()["claims"] == 1


def test_batch_rolls_back_claims_on_exception():
    s = SQLiteStore(":memory:")
    c = claim()
    with pytest.raises(RuntimeError):
        with s.batch():
            s.put_claim(c)
            raise RuntimeError("boom")
    assert s.get_claim(c.id) is None, "a failed batch must not leave a partial write"
    s.close()


def test_batch_commits_once_on_success():
    s = SQLiteStore(":memory:")
    with s.batch():
        for i in range(20):
            s.put_claim(claim(predicate=f"p{i}"))
    assert s.stats()["claims"] == 20
    s.close()


def test_batches_nest_without_committing_early():
    s = SQLiteStore(":memory:")
    with s.batch():
        s.put_claim(claim(predicate="outer"))
        with s.batch():
            s.put_claim(claim(predicate="inner"))
        assert s._batch_depth == 1, "inner exit must not end the outer transaction"
    assert s.stats()["claims"] == 2
    s.close()


def test_reopening_a_store_with_mixed_embedding_dims_fails_loudly(tmp_path):
    """Should be impossible via the API, but a hand-edited or migrated file must give a
    diagnosis rather than a cryptic numpy error."""
    path = str(tmp_path / "m.db")
    s = SQLiteStore(path)
    c = claim()
    s.put_claim(c)
    s.set_embedding(c.id, np.ones(8, dtype=np.float32))
    s.close()

    raw = sqlite3.connect(path)
    raw.execute("INSERT INTO embeddings (claim_id, dim, vec) VALUES (?,?,?)",
                ("cl_rogue", 3, np.ones(3, dtype=np.float32).tobytes()))
    raw.commit()
    raw.close()

    with pytest.raises(ValueError, match="mixed dimensions"):
        SQLiteStore(path)


def test_get_claims_bulk_handles_empty_unknown_and_duplicate_ids():
    s = SQLiteStore(":memory:")
    a, b = claim(predicate="p1"), claim(predicate="p2")
    s.put_claim(a)
    s.put_claim(b)
    assert s.get_claims([]) == {}
    assert s.get_claims(["nope"]) == {}
    got = s.get_claims([a.id, b.id, a.id, "nope"])
    assert set(got) == {a.id, b.id}
    s.close()


def test_get_claims_bulk_chunks_past_the_sql_parameter_limit():
    """The IN clause must chunk; SQLite rejects more than ~999 bound parameters."""
    s = SQLiteStore(":memory:")
    ids = []
    with s.batch():
        for i in range(2500):
            c = claim(predicate=f"p{i}")
            s.put_claim(c)
            ids.append(c.id)
    assert len(s.get_claims(ids)) == 2500
    s.close()


def test_upsert_preserves_rowid_so_the_text_index_stays_attached():
    """INSERT OR REPLACE would assign a new rowid and orphan the FTS entry, leaving
    stale text searchable forever."""
    s = SQLiteStore(":memory:")
    c = claim(object="Berlin")
    s.put_claim(c)
    for i in range(5):
        c.object = f"City{i}"
        c.text = f"user lives in City{i}"
        s.put_claim(c)
    assert s.lexical_search("berlin", [SCOPE], limit=5) == []
    assert s.lexical_search("city4", [SCOPE], limit=5)[0][0] == c.id
    assert s.stats()["claims"] == 1
    s.close()


# =============================================================================
# Retrieval robustness
# =============================================================================

def test_search_survives_a_claim_deleted_mid_query():
    """candidate_ids and hydration are separate reads; a row can vanish between them."""
    class VanishingStore(SQLiteStore):
        def get_claims(self, claim_ids):
            return {}

    mem = Engram(store=VanishingStore(":memory:"), embedder=HashingEmbedder(dim=64),
                 user="alice")
    mem.remember("user", "lives_in", "Berlin")
    assert mem.search("berlin") == []
    mem.close()


def test_search_falls_back_when_store_lacks_bulk_fetch():
    """Third-party Stores predating `get_claims` must keep working."""
    class OldStore(SQLiteStore):
        get_claims = None

    mem = Engram(store=OldStore(":memory:"), embedder=HashingEmbedder(dim=64),
                 user="alice")
    mem.remember("user", "lives_in", "Berlin")
    assert [r.claim.object for r in mem.search("berlin")] == ["Berlin"]
    mem.close()


@pytest.mark.parametrize("k", [0, -1, -100])
def test_non_positive_k_returns_nothing(k):
    with Engram(embedder=HashingEmbedder(dim=64), user="alice") as mem:
        mem.remember("user", "lives_in", "Berlin")
        assert mem.search("berlin", k=k) == []


def test_search_k_larger_than_corpus_is_harmless():
    with Engram(embedder=HashingEmbedder(dim=64), user="alice") as mem:
        mem.remember("user", "lives_in", "Berlin")
        assert len(mem.search("berlin", k=1000)) == 1


# =============================================================================
# Facade input handling
# =============================================================================

def test_add_accepts_prebuilt_episodes():
    with Engram(embedder=HashingEmbedder(dim=64), user="alice") as mem:
        ep = Episode(content="I live in Berlin", scope=Scope("default", "alice"))
        receipt = mem.add([ep])
        assert receipt.episode_ids == [ep.id]


def test_add_stringifies_non_string_message_content():
    with Engram(embedder=HashingEmbedder(dim=64), user="alice") as mem:
        receipt = mem.add([{"role": "user", "content": 12345}])
        ep = mem.store.get_episode(receipt.episode_ids[0])
        assert ep.content == "12345"


def test_add_accepts_a_single_mapping():
    with Engram(embedder=HashingEmbedder(dim=64), user="alice") as mem:
        assert len(mem.add({"role": "user", "content": "I live in Berlin"}).episode_ids) == 1


def test_forget_and_history_normalize_predicate_aliases():
    with Engram(embedder=HashingEmbedder(dim=64), user="alice") as mem:
        mem.remember("user", "lives_in", "Berlin")
        assert len(mem.history("user", "resides_in")) == 1
        assert len(mem.forget("user", "based_in")) == 1


# =============================================================================
# Malformed model output (the trust boundary)
# =============================================================================

class BadLLM:
    """Returns output that violates every field contract at once."""

    name = "bad"

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def extract(self, episodes, known_predicates):
        self.calls += 1
        return self.payload

    def classify_predicate(self, predicate, example):
        return {"cardinality": "many", "volatility": "slow", "memory_type": "semantic"}


@pytest.mark.parametrize("payload", [
    [{"subject": "user", "predicate": "lives_in", "object": "Berlin",
      "polarity": "banana", "memory_type": "semantic", "confidence": 0.9,
      "source_index": 0}],
    [{"subject": "user", "predicate": "lives_in", "object": "Berlin",
      "polarity": 1, "memory_type": "not_a_type", "confidence": 0.9,
      "source_index": 0}],
    [{"subject": "user", "predicate": "lives_in", "object": "Berlin",
      "polarity": 1, "memory_type": "semantic", "confidence": "high",
      "source_index": 0}],
    [{"subject": "user", "predicate": "lives_in", "object": "Berlin",
      "polarity": 1, "memory_type": "semantic", "confidence": None,
      "source_index": 0}],
])
def test_malformed_model_fields_are_coerced_not_crashed(payload):
    llm = BadLLM(payload)
    with Engram(embedder=HashingEmbedder(dim=64), llm=llm, user="alice") as mem:
        mem.add("Something the fast path will not touch, spoken at length here.")
        for c in mem.get_all():
            assert c.polarity in (1, -1)
            assert 0.0 <= c.confidence <= 1.0
            assert isinstance(c.memory_type, MemoryType)


@pytest.mark.parametrize("payload", [
    [],
    [{}],
    [{"subject": "user"}],
    [{"subject": "user", "predicate": "lives_in", "object": "X", "source_index": 99}],
    [{"subject": "user", "predicate": "lives_in", "object": "X", "source_index": -5}],
    [{"subject": "", "predicate": "", "object": "", "source_index": 0}],
])
def test_unusable_model_output_is_dropped_silently(payload):
    llm = BadLLM(payload)
    with Engram(embedder=HashingEmbedder(dim=64), llm=llm, user="alice") as mem:
        mem.add("Something the fast path will not touch, spoken at length here.")
        assert all(c.subject and c.predicate for c in mem.get_all())


def test_write_survives_an_embedder_that_changes_dimension_midway():
    """A misconfiguration must surface at retrieval, not destroy an otherwise valid
    write in progress."""
    class ShiftyEmbedder:
        dim = 32

        def __init__(self):
            self.calls = 0

        def encode(self, texts):
            self.calls += 1
            d = 32 if self.calls < 2 else 16
            return np.ones((len(texts), d), dtype=np.float32)

    mem = Engram(embedder=ShiftyEmbedder(), user="alice")
    mem.remember("user", "lives_in", "Berlin")
    with pytest.warns(RuntimeWarning, match="embedding rejected"):
        mem.remember("user", "works_at", "Acme")
    assert len(mem.get_all()) == 2, "a derived-index failure must not lose the claim"
    mem.close()


def test_embedding_rejection_warns_only_once():
    class BadEmbedder:
        dim = 4

        def encode(self, texts):
            return np.ones((len(texts), 4), dtype=np.float32)

    store = SQLiteStore(":memory:")
    store.put_claim(claim(predicate="seed"))
    store.set_embedding(claim(predicate="seed").id, np.ones(9, dtype=np.float32))
    mem = Engram(store=store, embedder=BadEmbedder(), user="alice")
    with pytest.warns(RuntimeWarning):
        mem.remember("user", "lives_in", "Berlin")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mem.remember("user", "works_at", "Acme")
    assert not [w for w in caught if "embedding rejected" in str(w.message)]
    mem.close()


# =============================================================================
# Fast extractor precision
# =============================================================================

@pytest.mark.parametrize("turn", [
    "I live in the place we discussed at length yesterday evening obviously.",
    "I live in a really lovely spot that I will describe to you in detail later on.",
    "I work at the company whose name I keep forgetting but you know the one.",
])
def test_overlong_or_vague_objects_are_left_to_the_llm(turn):
    with Engram(embedder=HashingEmbedder(dim=64), user="alice") as mem:
        mem.add(turn)
        assert mem.get_all() == [], "precision over recall: emit nothing rather than junk"


@pytest.mark.parametrize("turn,expected", [
    ("My name is Alice Tan.", ("name", "Alice Tan")),
    ("I live in Lisbon.", ("lives_in", "Lisbon")),
    ("I work at Acme Corp.", ("works_at", "Acme Corp")),
])
def test_high_confidence_forms_are_extracted_without_an_llm(turn, expected):
    llm = BadLLM([])
    with Engram(embedder=HashingEmbedder(dim=64), llm=llm, user="alice") as mem:
        mem.add(turn)
        assert (expected[0], expected[1]) in {(c.predicate, c.object) for c in mem.get_all()}
        assert llm.calls == 0


def test_questions_are_never_treated_as_assertions():
    with Engram(embedder=HashingEmbedder(dim=64), user="alice") as mem:
        mem.add("Do I live in Berlin? I wonder where I live these days.")
        assert mem.get_all() == []


def test_hypotheticals_are_not_stored_as_fact():
    with Engram(embedder=HashingEmbedder(dim=64), user="alice") as mem:
        mem.add("If I lived in Berlin I would be happier.")
        assert [c.object for c in mem.get_all()] == []


# =============================================================================
# Retraction edge cases
# =============================================================================

def test_retraction_without_a_named_value_clears_the_whole_slot():
    reg = PredicateRegistry()
    with Engram(embedder=HashingEmbedder(dim=64), registry=reg, user="alice") as mem:
        mem.remember("user", "works_at", "Acme")
        mem.remember("user", "works_at", "", polarity=-1)
        assert [c.object for c in mem.get_all()] == []
        assert len(mem.history("user", "works_at")) >= 1


def test_retracting_something_never_believed_is_harmless():
    with Engram(embedder=HashingEmbedder(dim=64), user="alice") as mem:
        mem.remember("user", "works_at", "Globex", polarity=-1)
        assert mem.get_all() == []


def test_repeated_retraction_does_not_accumulate_tombstones():
    with Engram(embedder=HashingEmbedder(dim=64), user="alice") as mem:
        mem.remember("user", "works_at", "Acme")
        for _ in range(3):
            mem.remember("user", "works_at", "Acme", polarity=-1)
        assert mem.get_all() == []
        assert mem.stats()["claims"] <= 3


# =============================================================================
# Consolidation edge cases
# =============================================================================

def test_consolidate_on_an_empty_store_is_a_noop():
    with Engram(embedder=HashingEmbedder(dim=64), user="alice") as mem:
        assert mem.consolidate() == {"decayed": 0, "merged": 0, "promoted": 0}


def test_consolidate_skips_slots_holding_a_single_claim():
    with Engram(embedder=HashingEmbedder(dim=64), user="alice") as mem:
        mem.remember("user", "lives_in", "Berlin")
        assert mem.consolidate()["merged"] == 0


def test_consolidate_tolerates_claims_with_no_embedding():
    store = SQLiteStore(":memory:")
    mem = Engram(store=store, embedder=HashingEmbedder(dim=64), user="alice")
    for i in range(3):
        store.put_claim(claim(predicate="likes", object=f"thing{i}"))
    assert isinstance(mem.consolidate(tenant="acme"), dict)
    mem.close()


# =============================================================================
# Concurrency and volume
# =============================================================================

def test_concurrent_adds_from_many_threads_lose_nothing(tmp_path):
    mem = Engram(str(tmp_path / "c.db"), embedder=HashingEmbedder(dim=64))
    errors: list[BaseException] = []

    def worker(n: int) -> None:
        try:
            for i in range(20):
                mem.remember("user", f"pred_{n}_{i}", f"v{i}", user=f"u{n}")
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert mem.stats()["claims"] == 160
    mem.close()


def test_concurrent_reads_during_writes_stay_consistent(tmp_path):
    mem = Engram(str(tmp_path / "rw.db"), embedder=HashingEmbedder(dim=64), user="alice")
    mem.remember("user", "lives_in", "Berlin")
    errors: list[BaseException] = []
    stop = threading.Event()

    def reader() -> None:
        try:
            while not stop.is_set():
                mem.search("berlin", k=5)
                mem.get_all()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    t = threading.Thread(target=reader)
    t.start()
    try:
        for i in range(150):
            mem.remember("user", f"p{i}", f"v{i}")
    finally:
        stop.set()
        t.join()
    assert not errors, errors
    mem.close()


def test_a_large_transcript_ingests_in_one_batch():
    llm = BadLLM([])
    with Engram(embedder=HashingEmbedder(dim=64), llm=llm, user="alice") as mem:
        turns = [f"Message number {i} with some filler content." for i in range(500)]
        receipt = mem.add(turns)
        assert len(receipt.episode_ids) == 500
        assert llm.calls <= 1, "a transcript must batch into at most one extraction call"


# =============================================================================
# Fuzz
# =============================================================================

FUZZ = [
    "", " ", "\n\t\r", "\x00", "\\", "%", "_", "'", '"', "`", ";--", "/*", "*/",
    "\\x00\\x01", "🙂" * 50, "日本語" * 30, "a" * 5000, "‮", "﻿",
    "SELECT * FROM claims", "'; DROP TABLE claims;--", "{{7*7}}", "${jndi:ldap://x}",
    "../../etc/passwd", "NaN", "Infinity", "-0", "1e400", "<script>alert(1)</script>",
    "\U0001F600\U0001F1FA\U0001F1F8", "ЁЂЃЄ", "\N{COMBINING ACUTE ACCENT}" * 20,
]


@pytest.mark.parametrize("payload", FUZZ)
def test_fuzz_input_never_raises_through_the_public_api(payload):
    with Engram(embedder=HashingEmbedder(dim=64), user="alice") as mem:
        mem.add(payload)
        mem.search(payload)
        mem.recall(payload)
        mem.remember("user", "likes", payload)
        mem.history("user", payload)
        mem.forget("user", payload)
        mem.consolidate()
        assert isinstance(mem.stats(), dict)


@pytest.mark.parametrize("payload", FUZZ)
def test_fuzz_payloads_survive_a_persistence_round_trip(tmp_path, payload):
    path = str(tmp_path / "f.db")
    with Engram(path, embedder=HashingEmbedder(dim=64), user="alice") as m1:
        m1.remember("user", "likes", payload)
    with Engram(path, embedder=HashingEmbedder(dim=64), user="alice") as m2:
        stored = [c.object for c in m2.get_all()]
        # Objects are whitespace-normalized on the way in — a value of "  " carries no
        # information and storing it would create an unanswerable slot. Everything else
        # must survive byte for byte, including control characters and astral-plane
        # codepoints, which is where a naive TEXT round trip goes wrong.
        expected = [payload.strip()] if payload.strip() else []
        assert stored == expected


def test_random_transcripts_never_corrupt_the_store(tmp_path):
    import random
    rng = random.Random(1234)
    vocab = ["I", "live", "in", "Berlin", "work", "at", "Acme", "and", "?", ".",
             "no", "not", "my", "name", "is", "🙂", "日本", "'", '"', ";"]
    mem = Engram(str(tmp_path / "r.db"), embedder=HashingEmbedder(dim=64), user="alice")
    for _ in range(300):
        turn = " ".join(rng.choice(vocab) for _ in range(rng.randint(0, 14)))
        mem.add(turn)
    mem.consolidate()
    stats = mem.stats()
    assert stats["claims"] == stats["live_claims"] + stats["invalidated"]
    for c in mem.get_all():
        assert c.subject and c.predicate
        assert c.invalidated_at is None
    mem.close()
