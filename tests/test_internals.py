"""Defensive branches and optional-dependency paths.

These lines only execute when a dependency is installed that this environment does not
have, or when a lower layer misbehaves in a way the layer above is built to absorb.
Both are exactly the code that must not be shipped unexercised: the first is invisible
in CI, the second only runs during an incident.
"""

import sys
import types

import numpy as np
import pytest

import memvara
from memvara import Claim, Memvara, Episode, HashingEmbedder, MemoryType, Scope, SQLiteStore
from memvara.write.pipeline import _coerce

SCOPE = Scope("acme", "alice")


def claim(**kw) -> Claim:
    base = dict(subject="user", predicate="lives_in", object="Berlin", scope=SCOPE)
    base.update(kw)
    return Claim(**base)


class CountingLLM:
    name = "counting"

    def __init__(self, payload):
        self.payload = payload
        self.extract_calls = 0
        self.classify_calls = 0
        self.classify_args: list[tuple[str, str]] = []

    def extract(self, episodes, known_predicates):
        self.extract_calls += 1
        return self.payload

    def classify_predicate(self, predicate, example):
        self.classify_calls += 1
        self.classify_args.append((predicate, example))
        return {"cardinality": "many", "volatility": "slow", "memory_type": "semantic"}


# =============================================================================
# Lazy optional imports
# =============================================================================

def test_anthropic_llm_is_reachable_lazily_from_the_package_root():
    """`import memvara` must not require the anthropic SDK, but `memvara.AnthropicLLM`
    must still resolve for people who installed it."""
    cls = memvara.AnthropicLLM
    assert cls.__name__ == "AnthropicLLM"


def test_unknown_package_attribute_still_raises_attribute_error():
    with pytest.raises(AttributeError, match="has no attribute"):
        memvara.definitely_not_a_real_symbol


def test_anthropic_backend_builds_a_default_client_when_the_sdk_is_present(monkeypatch):
    """Covers the default-construction path without installing the real SDK."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")  # now checked at construction
    fake = types.ModuleType("anthropic")
    built = []

    class FakeClient:
        pass

    def factory(*a, **kw):
        built.append(True)
        return FakeClient()

    fake.Anthropic = factory
    monkeypatch.setitem(sys.modules, "anthropic", fake)

    from memvara.llm.anthropic import AnthropicLLM

    llm = AnthropicLLM()
    assert built, "constructing without an explicit client must build the default one"
    assert "claude-opus-5" in llm.name


def test_anthropic_backend_explains_itself_when_the_sdk_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    from memvara.llm.anthropic import AnthropicLLM

    with pytest.raises(ImportError, match="client=|install|anthropic"):
        AnthropicLLM()


def test_local_embedder_wraps_sentence_transformers(monkeypatch):
    """Covers the optional local-embedding backend without downloading a model."""
    fake = types.ModuleType("sentence_transformers")

    class FakeST:
        def __init__(self, name):
            self.name = name

        def get_sentence_embedding_dimension(self):
            return 8

        def encode(self, texts, normalize_embeddings=True):
            return np.ones((len(texts), 8), dtype=np.float32)

    fake.SentenceTransformer = FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

    from memvara.embed.local import LocalEmbedder

    emb = LocalEmbedder()
    assert emb.dim == 8
    assert emb.encode(["a", "b"]).shape == (2, 8)


def test_default_embedder_falls_back_when_local_backend_is_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    from memvara.embed import default_embedder

    emb = default_embedder(dim=64)
    assert emb.encode(["hello"]).shape == (1, 64)


# =============================================================================
# Write path: absorbing a misbehaving store
# =============================================================================

def test_near_duplicate_check_survives_an_index_dimension_mismatch():
    """The near-dup lookup is an optimization. If the vector index was built by another
    embedder it must decline to help, not abort a valid write."""
    mem = Memvara(embedder=HashingEmbedder(dim=32), user="alice")
    mem.add("I live in Berlin.")
    assert len(mem.get_all()) == 1

    mem.writer.embedder = HashingEmbedder(dim=16)  # now disagrees with the index
    with pytest.warns(RuntimeWarning):
        mem.add("My name is Alice Tan.")
    assert len(mem.get_all()) == 2, "the write must land despite the broken optimization"
    mem.close()


def test_near_duplicate_hit_pointing_at_a_vanished_claim_is_ignored():
    """vector_search and get_claim are separate reads; the row can disappear between."""
    class GhostStore(SQLiteStore):
        def get_claim(self, claim_id):
            return None

    # Threshold 0 makes any vector hit count, so the near-duplicate branch is guaranteed
    # to run rather than depending on how similar two phrasings happen to embed.
    mem = Memvara(store=GhostStore(":memory:"), embedder=HashingEmbedder(dim=32),
                 user="alice", write_near_dup_threshold=0.0)
    mem.add("I live in Berlin.")
    # Not byte-identical, so episode-hash dedupe does not short-circuit it and the
    # near-duplicate vector check actually runs.
    mem.add("My name is Alice Tan!")
    assert mem.stats()["claims"] >= 1, "a dangling near-dup hit must not lose the write"
    mem.close()


def test_consolidation_skips_claims_that_are_not_yet_in_force():
    """A row with no invalidation is still not necessarily live — a fact scheduled to
    become true later must not be merged or decayed as though it already were."""
    from datetime import datetime, timezone

    store = SQLiteStore(":memory:")
    mem = Memvara(store=store, embedder=HashingEmbedder(dim=32), user="alice")
    future = datetime(2090, 1, 1, tzinfo=timezone.utc)
    for obj in ("alpha", "omega"):
        c = claim(predicate="likes", object=obj, valid_from=future)
        store.put_claim(c)
        store.set_embedding(c.id, mem.embedder.encode([c.text])[0])
    assert mem.consolidator.merge_duplicates("acme") == 0
    mem.close()


def test_retrieval_reapplies_the_transaction_floor_even_if_the_store_forgets():
    """Defense in depth: the store enforces `recorded_at <= as_of`, but a third-party
    Store might not, and future knowledge must never leak into a historical answer."""
    from datetime import datetime, timezone

    T = datetime(2024, 1, 1, tzinfo=timezone.utc)

    class LeakyStore(SQLiteStore):
        def lexical_search(self, query, scopes, limit, *, valid_at=None,
                           known_at=None, states=None, include_invalidated=None):
            return super().lexical_search(query, scopes, limit,
                                          include_invalidated=True)

        def vector_search(self, qvec, scopes, limit, *, valid_at=None,
                          known_at=None, states=None, include_invalidated=None):
            return super().vector_search(qvec, scopes, limit,
                                         include_invalidated=True)

    mem = Memvara(store=LeakyStore(":memory:"), embedder=HashingEmbedder(dim=32),
                 user="alice")
    mem.remember("user", "lives_in", "Lisbon",
                 recorded_at=datetime(2025, 1, 1, tzinfo=timezone.utc))
    assert mem.search("lives", as_of=T, include_invalidated=True) == []
    mem.close()


# =============================================================================
# Model output coercion
# =============================================================================

@pytest.mark.parametrize("raw,expected", [
    ("semantic", MemoryType.SEMANTIC),
    ("EPISODIC", MemoryType.EPISODIC),
    ("  procedural  ", MemoryType.PROCEDURAL),
    ("nonsense", MemoryType.SEMANTIC),
    (None, MemoryType.SEMANTIC),
    (42, MemoryType.SEMANTIC),
    ([], MemoryType.SEMANTIC),
])
def test_coerce_maps_model_strings_and_falls_back_on_junk(raw, expected):
    assert _coerce(MemoryType, raw, MemoryType.SEMANTIC) is expected


def test_the_learned_schema_outranks_a_per_claim_guess():
    """Once a predicate has been classified, the registry is authoritative. Letting each
    extraction re-decide would leave claims of the same predicate disagreeing about what
    kind of memory they are, which is what the schema exists to prevent."""
    payload = [{"subject": "user", "predicate": "collects_stamps", "object": "at length",
                "polarity": 1, "memory_type": "episodic", "confidence": 0.9,
                "source_index": 0}]
    llm = CountingLLM(payload)  # classifies every novel predicate as "semantic"
    with Memvara(embedder=HashingEmbedder(dim=32), llm=llm, user="alice") as mem:
        mem.add("Some sentence the rules will not touch, spoken at length here.")
        assert {c.memory_type for c in mem.get_all()} == {MemoryType.SEMANTIC}


class FailingClassifier(CountingLLM):
    """Schema acquisition fails outright — a rate limit, a timeout, a 500."""

    def classify_predicate(self, predicate, example):
        self.classify_calls += 1
        self.classify_args.append((predicate, example))
        raise RuntimeError("rate limited")


@pytest.mark.parametrize("raw,expected", [
    ("episodic", MemoryType.EPISODIC),
    ("procedural", MemoryType.PROCEDURAL),
    ("!!!", MemoryType.SEMANTIC),
    (None, MemoryType.SEMANTIC),
])
def test_a_failed_classification_falls_back_to_the_per_claim_memory_type(raw, expected):
    payload = [{"subject": "user", "predicate": "collects_stamps", "object": "at length",
                "polarity": 1, "memory_type": raw, "confidence": 0.9,
                "source_index": 0}]
    llm = FailingClassifier(payload)
    with Memvara(embedder=HashingEmbedder(dim=32), llm=llm, user="alice") as mem:
        mem.add("Some sentence the rules will not touch, spoken at length here.")
        stored = mem.get_all()
        assert stored, "a failed classification must not cost the caller the facts"
        assert {c.memory_type for c in stored} == {expected}


def test_a_failed_classification_is_not_retried_in_a_loop():
    payload = [{"subject": "user", "predicate": "collects_stamps", "object": "at length",
                "polarity": 1, "memory_type": "semantic", "confidence": 0.9,
                "source_index": 0}]
    llm = FailingClassifier(payload)
    with Memvara(embedder=HashingEmbedder(dim=32), llm=llm, user="alice") as mem:
        mem.add("Some sentence the rules will not touch, spoken at length here.")
        mem.add("A different sentence the rules will also not touch, at length.")
        assert llm.classify_calls == 1, "a failing predicate must not bill us repeatedly"


def test_an_unhelpful_classification_still_registers_safe_defaults():
    """Nonsense answers are coerced rather than rejected, so the predicate becomes
    known and — crucially — multi-valued, which never retires a true fact."""
    class Unhelpful(CountingLLM):
        def classify_predicate(self, predicate, example):
            self.classify_calls += 1
            return {"cardinality": "???", "volatility": "???", "memory_type": "???"}

    payload = [{"subject": "user", "predicate": "collects_stamps", "object": "a",
                "polarity": 1, "memory_type": "semantic", "confidence": 0.9,
                "source_index": 0}]
    llm = Unhelpful(payload)
    with Memvara(embedder=HashingEmbedder(dim=32), llm=llm, user="alice") as mem:
        mem.add("Some sentence the rules will not touch, spoken at length here.")
        assert mem.registry.known("collects_stamps")
        assert not mem.registry.functional("collects_stamps")


def test_a_novel_predicate_is_classified_exactly_once_across_many_claims():
    payload = [
        {"subject": "user", "predicate": "collects_stamps", "object": "penny black",
         "polarity": 1, "memory_type": "semantic", "confidence": 0.9, "source_index": 0},
        {"subject": "user", "predicate": "collects_stamps", "object": "blue mauritius",
         "polarity": 1, "memory_type": "semantic", "confidence": 0.9, "source_index": 0},
    ]
    llm = CountingLLM(payload)
    with Memvara(embedder=HashingEmbedder(dim=32), llm=llm, user="alice") as mem:
        mem.add("Some sentence the rules will not touch, spoken at length here.")
        mem.add("A different sentence the rules will also not touch, at length.")
        assert llm.classify_calls == 1, "schema acquisition is paid for once, ever"


def test_classification_example_falls_back_to_the_object_when_the_index_is_unusable():
    """`source_index` is the model's own pointer back into the batch. When it is out of
    range there is no source turn to quote, so the example must degrade rather than
    index out of bounds."""
    payload = [{"subject": "user", "predicate": "collects_stamps",
                "object": "penny black", "polarity": 1, "memory_type": "semantic",
                "confidence": 0.9, "source_index": 999}]
    llm = CountingLLM(payload)
    with Memvara(embedder=HashingEmbedder(dim=32), llm=llm, user="alice") as mem:
        mem.add("Some sentence the rules will not touch, spoken at length here.")
    if llm.classify_args:
        assert llm.classify_args[0][1] == "penny black"


# =============================================================================
# Reconciler guards
# =============================================================================

@pytest.mark.parametrize("subject,predicate", [("", "lives_in"), ("user", ""), ("", "")])
def test_a_claim_missing_subject_or_predicate_is_not_stored(subject, predicate):
    store = SQLiteStore(":memory:")
    mem = Memvara(store=store, embedder=HashingEmbedder(dim=32), user="alice")
    mem.writer.assert_claim(claim(subject=subject, predicate=predicate))
    assert store.stats()["claims"] == 0
    mem.close()


def test_a_positive_claim_with_no_value_is_not_stored():
    store = SQLiteStore(":memory:")
    mem = Memvara(store=store, embedder=HashingEmbedder(dim=32), user="alice")
    mem.remember("user", "likes", "   ")
    assert store.stats()["claims"] == 0
    mem.close()


# =============================================================================
# Fast extractor rejection paths
# =============================================================================

def test_an_object_longer_than_the_cap_is_left_to_the_llm():
    with Memvara(embedder=HashingEmbedder(dim=32), user="alice") as mem:
        mem.add("I live in " + "Averylongplacename" * 12 + ".")
        assert mem.get_all() == []


def test_an_object_with_no_alphanumeric_content_is_rejected():
    with Memvara(embedder=HashingEmbedder(dim=32), user="alice") as mem:
        mem.add("My name is ---.")
        assert mem.get_all() == []


@pytest.mark.parametrize("turn", [
    "I like coffee and tea.",
    "I like coffee or tea.",
    "I work at Acme but not really.",
])
def test_coordinated_objects_are_handed_to_the_llm_whole(turn):
    with Memvara(embedder=HashingEmbedder(dim=32), user="alice") as mem:
        mem.add(turn)
        assert mem.get_all() == []


def test_stacked_filler_is_stripped_down_to_the_value():
    with Memvara(embedder=HashingEmbedder(dim=32), user="alice") as mem:
        mem.add("I live in Berlin now.")
        assert [c.object for c in mem.get_all()] == ["Berlin"]


# =============================================================================
# Consolidation: dissimilar claims in one slot
# =============================================================================

def test_dissimilar_claims_sharing_a_slot_are_not_merged():
    """Same (subject, predicate) is not enough — merging requires the texts to actually
    be near-identical, or two genuinely different values would collapse into one."""
    store = SQLiteStore(":memory:")
    mem = Memvara(store=store, embedder=HashingEmbedder(dim=64), user="alice")
    for obj in ("completely different alpha", "utterly unrelated omega"):
        c = claim(predicate="likes", object=obj)
        store.put_claim(c)
        store.set_embedding(c.id, mem.embedder.encode([c.text])[0])
    assert mem.consolidator.merge_duplicates("acme") == 0
    assert len(list(store.iter_claims("acme"))) == 2
    mem.close()


def test_near_identical_claims_sharing_a_slot_do_merge():
    store = SQLiteStore(":memory:")
    mem = Memvara(store=store, embedder=HashingEmbedder(dim=64), user="alice")
    ids = []
    for _ in range(2):
        c = claim(predicate="likes", object="espresso")
        c.text = "user likes espresso"
        store.put_claim(c)
        store.set_embedding(c.id, mem.embedder.encode([c.text])[0])
        ids.append(c.id)
    assert mem.consolidator.merge_duplicates("acme") == 1
    survivors = [c for c in store.iter_claims("acme")]
    assert len(survivors) == 1
    assert survivors[0].observation_count == 2
    mem.close()


# =============================================================================
# Remaining conditional branches
# =============================================================================

def test_vector_index_tolerates_a_zero_norm_query():
    """A query that embeds to nothing has no direction. It must score everything at
    zero rather than divide by zero."""
    from memvara.store.sqlite import _VecIndex

    idx = _VecIndex()
    idx.add("a", np.array([1.0, 0.0], dtype=np.float32))
    hits = idx.search(np.zeros(2, dtype=np.float32), ["a"], 5)
    assert hits == [("a", 0.0)]


def test_an_exception_inside_a_nested_batch_rolls_back_the_whole_outer_transaction():
    """Only the outermost batch owns the rollback; an inner one must not half-commit."""
    store = SQLiteStore(":memory:")
    with pytest.raises(RuntimeError):
        with store.batch():
            store.put_claim(claim(predicate="outer"))
            with store.batch():
                store.put_claim(claim(predicate="inner"))
                raise RuntimeError("boom")
    assert store.stats()["claims"] == 0
    store.close()


def test_iter_claims_with_no_filters_returns_every_row():
    store = SQLiteStore(":memory:")
    live = claim(predicate="live_one")
    dead = claim(predicate="dead_one")
    store.put_claim(live)
    store.put_claim(dead)
    store.invalidate(dead.id, live.recorded_at, None)
    everything = list(store.iter_claims(tenant=None, include_invalidated=True))
    assert {c.id for c in everything} == {live.id, dead.id}
    store.close()


def test_an_already_expired_claim_is_not_retired_a_second_time():
    """A fact that stopped being true in 2020 is not a competing answer today, so
    superseding it must not re-date its end. Valid time only ever moves forward once."""
    from datetime import datetime, timezone

    store = SQLiteStore(":memory:")
    mem = Memvara(store=store, embedder=HashingEmbedder(dim=32), user="alice")
    old_end = datetime(2020, 6, 1, tzinfo=timezone.utc)

    first = claim(scope=Scope("default", "alice"), object="Berlin")
    store.put_claim(first)
    store.set_valid_to(first.id, old_end)

    mem.remember("user", "lives_in", "Lisbon")
    assert store.get_claim(first.id).valid_to == old_end
    mem.close()


def test_response_parsing_skips_non_text_blocks_before_the_answer():
    """Real responses interleave thinking and tool blocks ahead of the text one."""
    from memvara.llm.anthropic import _first_text

    class Block:
        def __init__(self, type_, text=None):
            self.type = type_
            self.text = text

    class Response:
        def __init__(self, content):
            self.content = content

    assert _first_text(Response([{"type": "thinking"}, {"type": "text", "text": "hi"}])) == "hi"
    assert _first_text(Response([Block("thinking"), Block("text", "yo")])) == "yo"
    assert _first_text(Response([])) == ""
    assert _first_text(Response(None)) == ""


def test_connectivity_is_empty_rather_than_zero_for_a_store_that_cannot_measure_it():
    """`{}` and `{"live_claims": 0, "joinable_claims": 0}` mean different things.

    The second is a measured star, which is a real finding an operator should act on by
    changing what the write path stores. The first is a backend that never looked. A
    caller that read a missing key as zero would report the finding without the
    measurement, and `memory_stats` prints no join-rate line at all in that case.
    """
    class NoConnectivity(SQLiteStore):
        connectivity = None            # type: ignore[assignment]

    mem = Memvara(store=NoConnectivity(":memory:"), embedder=HashingEmbedder(dim=32),
                 user="alice")
    assert mem.connectivity() == {}
    mem.close()


def test_connectivity_defaults_to_this_instances_tenant(monkeypatch):
    """Same rule as `stats()`: a shared store must not leak another tenant's shape, and
    the join rate is a shape.
    """
    mem = Memvara(":memory:", embedder=HashingEmbedder(dim=32), tenant="acme",
                 user="alice")
    asked: list = []
    real = mem.store.connectivity
    monkeypatch.setattr(mem.store, "connectivity",
                        lambda t=None: (asked.append(t), real(t))[1])
    mem.connectivity()
    mem.connectivity(tenant="other")
    assert asked == ["acme", "other"]
    mem.close()


def test_stats_falls_back_for_a_store_without_tenant_scoping():
    """A third-party Store predating the tenant argument must keep working."""
    class OldStore(SQLiteStore):
        def stats(self):  # no tenant parameter
            return {"episodes": 0, "claims": 99, "live_claims": 99,
                    "invalidated": 0, "embeddings": 0}

    mem = Memvara(store=OldStore(":memory:"), embedder=HashingEmbedder(dim=32),
                 user="alice")
    assert mem.stats()["claims"] == 99
    mem.close()
