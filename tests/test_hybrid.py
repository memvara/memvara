"""End-to-end hybrid retrieval: scope isolation, time travel, and the two failure
modes a vector-only store cannot fix.

Everything runs offline: `SQLiteStore(":memory:")` plus `HashingEmbedder`, claims
constructed directly and inserted through the store. Time is always passed in
explicitly - nothing here sleeps or patches a clock.
"""

from __future__ import annotations

import re
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from memvara.embed import HashingEmbedder
from memvara.retrieve import (
    CLAIM,
    EPISODE,
    STOPWORDS,
    EpisodeResult,
    HybridRetriever,
    analyze,
    calibrate_min_score,
    kind_of,
    lexical_relevance,
    relevance,
    tokenize,
    vector_relevance,
)
from memvara.retrieve import hybrid as hybrid_mod
from memvara.retrieve.hybrid import UnjoinedStoreWarning
from memvara.schema import PredicateRegistry
from memvara.store import SQLiteStore
from memvara.telemetry import (
    RETRIEVAL_LATENCY_MS,
    RETRIEVAL_OBSERVATION_RANK_CORR,
    RETRIEVAL_QUALITY_FACTOR,
    RETRIEVAL_QUERY,
    RETRIEVAL_RESULTS,
    MemoryRecorder,
)
from memvara.types import Claim, Episode, MemoryType, Scope

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
T1 = datetime(2025, 6, 1, tzinfo=timezone.utc)
T2 = datetime(2026, 8, 1, tzinfo=timezone.utc)


@pytest.fixture
def store() -> SQLiteStore:
    return SQLiteStore(":memory:")


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=512)


@pytest.fixture
def retriever(store: SQLiteStore, embedder: HashingEmbedder) -> HybridRetriever:
    return HybridRetriever(store, embedder, PredicateRegistry())


def add(
    store: SQLiteStore,
    embedder: HashingEmbedder | None,
    text: str,
    scope: Scope,
    *,
    subject: str = "user",
    predicate: str = "reported",
    **kw,
) -> Claim:
    """Insert a claim, embedding it unless `embedder` is None."""
    claim = Claim(subject=subject, predicate=predicate, object=text, text=text,
                  scope=scope, **kw)
    store.put_claim(claim)
    if embedder is not None:
        store.set_embedding(claim.id, embedder.encode([text])[0])
    return claim


def ids(results) -> list[str]:
    return [r.claim.id for r in results]


# ===========================================================================
# Scope isolation
# ===========================================================================

SESSION = Scope("acme", "alice", "agent_a", "sess_1")


@pytest.fixture
def scoped(store: SQLiteStore, embedder: HashingEmbedder) -> dict[str, Claim]:
    """One claim at every scope that matters, all matching the same query."""
    return {
        "tenant": add(store, embedder, "policy: deploys require two approvals",
                      Scope("acme")),
        "user": add(store, embedder, "policy: deploys happen on fridays",
                    Scope("acme", "alice")),
        "agent": add(store, embedder, "policy: deploys are logged to the audit trail",
                     Scope("acme", "alice", "agent_a")),
        "session": add(store, embedder, "policy: deploys are staged before release",
                       SESSION),
        # --- everything below must be unreachable from SESSION ---
        "sibling_session": add(store, embedder, "policy: deploys skip staging entirely",
                               Scope("acme", "alice", "agent_a", "sess_2")),
        "sibling_agent": add(store, embedder, "policy: deploys need a second reviewer",
                             Scope("acme", "alice", "agent_b")),
        "other_user": add(store, embedder, "policy: deploys happen on mondays",
                          Scope("acme", "bob")),
        "other_tenant": add(store, embedder, "policy: deploys require two approvals",
                            Scope("globex")),
    }


def test_session_query_inherits_every_ancestor_scope(
    retriever: HybridRetriever, scoped: dict[str, Claim]
) -> None:
    """The point of hierarchical scoping: a question asked in today's session can still
    be answered by what the user said months ago at a broader scope."""
    found = set(ids(retriever.search("deploy policy", SESSION, k=10)))

    for level in ("session", "agent", "user", "tenant"):
        assert scoped[level].id in found, f"{level}-scoped claim was not inherited"


def test_sibling_session_is_invisible(
    retriever: HybridRetriever, scoped: dict[str, Claim]
) -> None:
    """`ancestors()` walks strictly upward, so scope can widen but never step sideways."""
    found = set(ids(retriever.search("deploy policy", SESSION, k=10)))

    assert scoped["sibling_session"].id not in found
    assert scoped["sibling_agent"].id not in found


def test_other_users_memory_is_invisible(
    retriever: HybridRetriever, scoped: dict[str, Claim]
) -> None:
    found = set(ids(retriever.search("deploy policy", SESSION, k=10)))

    assert scoped["other_user"].id not in found


def test_cross_tenant_leakage_is_impossible(
    retriever: HybridRetriever, scoped: dict[str, Claim]
) -> None:
    """The worst bug this system could have, so it is asserted from both directions and
    with byte-identical text on either side - if the filter were being done on content
    rather than on scope, identical text is what would slip through."""
    from_acme = set(ids(retriever.search("deploy policy", SESSION, k=10)))
    from_globex = set(ids(retriever.search("deploy policy", Scope("globex"), k=10)))

    assert scoped["other_tenant"].id not in from_acme
    assert scoped["other_tenant"].id in from_globex
    assert from_acme.isdisjoint(from_globex)

    # Not one claim from the acme tenant is reachable from globex, at any scope depth.
    acme_ids = {c.id for name, c in scoped.items() if name != "other_tenant"}
    assert acme_ids.isdisjoint(from_globex)


def test_an_unknown_tenant_sees_nothing(
    retriever: HybridRetriever, scoped: dict[str, Claim]
) -> None:
    assert retriever.search("deploy policy", Scope("initech"), k=10) == []


def test_scope_widens_upward_only_and_never_descends(
    retriever: HybridRetriever, scoped: dict[str, Claim]
) -> None:
    """A tenant-scoped query sees tenant-scoped claims, not every user beneath it.

    This is `Scope.ancestors()` semantics as specified, and it is the least-privilege
    direction: no query can be constructed that reaches into a narrower context than
    the one it was asked in.
    """
    found = set(ids(retriever.search("deploy policy", Scope("acme"), k=10)))

    assert found == {scoped["tenant"].id}


def test_user_scope_sees_tenant_but_not_agent_or_session(
    retriever: HybridRetriever, scoped: dict[str, Claim]
) -> None:
    found = set(ids(retriever.search("deploy policy", Scope("acme", "alice"), k=10)))

    assert found == {scoped["user"].id, scoped["tenant"].id}


# ===========================================================================
# Time travel
# ===========================================================================

WORK_SCOPE = Scope("acme", "alice")


@pytest.fixture
def employment(store: SQLiteStore, embedder: HashingEmbedder) -> dict[str, Claim]:
    """A superseded fact and its replacement, on both time axes."""
    old = add(store, embedder, "the user works at Initech", WORK_SCOPE,
              predicate="works_at", valid_from=T0, recorded_at=T0)
    new = add(store, embedder, "the user works at Acme", WORK_SCOPE,
              predicate="works_at", valid_from=T1, recorded_at=T1)
    # Exactly what `Reconciler` does on supersession: retire, never delete.
    old.invalidated_at, old.invalidated_by, old.valid_to = T1, new.id, T1
    store.put_claim(old)
    return {"old": old, "new": new}


def test_current_search_returns_only_the_live_fact(
    retriever: HybridRetriever, employment: dict[str, Claim]
) -> None:
    found = ids(retriever.search("where does the user work", WORK_SCOPE, k=10))

    assert found == [employment["new"].id]


def test_as_of_returns_the_belief_of_that_moment(
    retriever: HybridRetriever, employment: dict[str, Claim]
) -> None:
    """The headline feature. In 2024 we believed Initech, and asking about 2024 must
    still say Initech - a claim being invalidated *later* cannot retroactively remove
    it from what we thought at the time."""
    found = ids(retriever.search("where does the user work", WORK_SCOPE,
                                 k=10, as_of=T0 + timedelta(days=30)))

    assert found == [employment["old"].id]


def test_as_of_excludes_claims_recorded_after_that_moment(
    retriever: HybridRetriever, employment: dict[str, Claim]
) -> None:
    """The other direction, and the one that is easy to get wrong: a historical answer
    must not contain knowledge from the future."""
    results = retriever.search("where does the user work", WORK_SCOPE,
                               k=10, as_of=T0 + timedelta(days=30))

    assert employment["new"].id not in ids(results)


def test_as_of_after_the_switch_returns_the_new_fact(
    retriever: HybridRetriever, employment: dict[str, Claim]
) -> None:
    found = ids(retriever.search("where does the user work", WORK_SCOPE,
                                 k=10, as_of=T2))

    assert found == [employment["new"].id]


def test_as_of_before_anything_was_known_returns_nothing(
    retriever: HybridRetriever, employment: dict[str, Claim]
) -> None:
    assert retriever.search("where does the user work", WORK_SCOPE, k=10,
                            as_of=T0 - timedelta(days=1)) == []


def test_include_invalidated_surfaces_retired_claims(
    retriever: HybridRetriever, employment: dict[str, Claim]
) -> None:
    found = set(ids(retriever.search("where does the user work", WORK_SCOPE,
                                     k=10, include_invalidated=True)))

    assert found == {employment["old"].id, employment["new"].id}


def test_include_invalidated_still_honours_the_transaction_floor(
    retriever: HybridRetriever, employment: dict[str, Claim]
) -> None:
    """`include_invalidated` lifts the liveness filter; it must not lift time itself.

    "What did we believe in 2024, including what we later retracted" cannot include a
    fact first recorded in 2025. Leaking it is the one way a bitemporal query can
    actually lie, so the retriever re-applies the floor rather than trusting the flag
    to be orthogonal on its own.
    """
    results = retriever.search("where does the user work", WORK_SCOPE, k=10,
                               as_of=T0 + timedelta(days=30), include_invalidated=True)

    assert ids(results) == [employment["old"].id]


def test_recency_is_measured_at_the_asked_instant_not_at_wall_clock(
    store: SQLiteStore, embedder: HashingEmbedder, retriever: HybridRetriever
) -> None:
    """A time-travel query must score that era's facts as fresh. Decaying them against
    today's clock would bury every historical answer under its own age - the answer
    would still be correct and would still rank last, which is the worst combination."""
    claim = add(store, embedder, "the user is working on the payments migration",
                WORK_SCOPE, predicate="working_on", valid_from=T0, recorded_at=T0)

    at_the_time = retriever.search("working on migration", WORK_SCOPE, k=5, as_of=T0)
    years_later = retriever.search("working on migration", WORK_SCOPE, k=5, as_of=T2)

    assert ids(at_the_time) == ids(years_later) == [claim.id]
    assert at_the_time[0].explain.recency == pytest.approx(1.0)
    assert years_later[0].explain.recency < 1e-30  # ~135 half-lives later


# ===========================================================================
# Recency ranking
# ===========================================================================


def test_stale_fast_claim_ranks_below_a_fresh_one(
    store: SQLiteStore, embedder: HashingEmbedder, retriever: HybridRetriever
) -> None:
    """`working_on` is FAST (7-day half-life). What someone was doing a month ago is
    not what they are doing now, and cosine similarity has no way to know that."""
    now = datetime(2026, 8, 8, tzinfo=timezone.utc)
    stale = add(store, embedder, "the user is working on the auth refactor phase one",
                WORK_SCOPE, predicate="working_on",
                valid_from=now - timedelta(days=30), recorded_at=now - timedelta(days=30))
    fresh = add(store, embedder, "the user is working on the auth refactor phase two",
                WORK_SCOPE, predicate="working_on", valid_from=now, recorded_at=now)

    results = retriever.search("auth refactor", WORK_SCOPE, k=5, as_of=now)

    assert ids(results)[0] == fresh.id
    assert stale.id in ids(results)  # decayed out of first place, not out of memory
    by_id = {r.claim.id: r for r in results}
    assert by_id[stale.id].explain.recency < 0.06
    assert by_id[fresh.id].explain.recency == pytest.approx(1.0)


def test_static_claim_survives_a_decade_undecayed(
    store: SQLiteStore, embedder: HashingEmbedder, retriever: HybridRetriever
) -> None:
    """The counterweight. A birthplace from ten years ago must rank exactly as it would
    have on day one; a recency-only ranker would have thrown it away long ago."""
    now = datetime(2026, 8, 8, tzinfo=timezone.utc)
    ancient = add(store, embedder, "the user was born in Osaka", WORK_SCOPE,
                  predicate="born_in", valid_from=now - timedelta(days=3650),
                  recorded_at=now - timedelta(days=3650))
    add(store, embedder, "the user is working on being born again in a new city",
        WORK_SCOPE, predicate="working_on", valid_from=now - timedelta(days=3650),
        recorded_at=now - timedelta(days=3650))

    results = retriever.search("born in Osaka", WORK_SCOPE, k=5, as_of=now)
    by_id = {r.claim.id: r for r in results}

    assert ids(results)[0] == ancient.id
    assert by_id[ancient.id].explain.recency > 0.93
    # Same age, same scope, same query - only the predicate's volatility differs.
    others = [r for r in results if r.claim.id != ancient.id]
    assert all(r.explain.recency < 1e-100 for r in others)


# ===========================================================================
# The case that justifies hybrid retrieval
# ===========================================================================


class BlurryEmbedder:
    """An embedder that drops opaque identifiers, the way a real one effectively does.

    `HashingEmbedder` cannot demonstrate this failure: it is a character-n-gram hash,
    so a rare token like `ERR_7734` is its *strongest* signal. A production sentence
    embedder behaves the opposite way - a subword tokenizer shreds the identifier into
    fragments that carry no trained meaning, and the resulting sentence vector is
    dominated by the surrounding prose. This models that end state exactly: two claims
    differing only in their error code become indistinguishable.

    Deterministic, offline, and no model download - it is `HashingEmbedder` with the
    identifiers removed first.
    """

    def __init__(self, dim: int = 256) -> None:
        self._inner = HashingEmbedder(dim=dim)
        self.dim = dim

    def encode(self, texts):
        return self._inner.encode([re.sub(r"\S*\d\S*", " ", t) for t in texts])


def test_exact_rare_token_is_found_by_bm25_when_the_embedding_cannot_see_it(
    store: SQLiteStore,
) -> None:
    """The whole argument for hybrid retrieval, in one query.

    Two claims are identical apart from their error code. The embedder maps them to
    the *same point* - not merely close, identical - so a vector-only top-k is a coin
    flip between a real answer and a wrong one. BM25 separates them instantly, because
    a five-digit code appearing in one document out of four has enormous IDF.
    """
    embedder = BlurryEmbedder()
    retriever = HybridRetriever(store, embedder, PredicateRegistry())
    scope = Scope("acme", "alice")

    target = add(store, embedder,
                 "the checkout service returned error code ERR_7734 after the tls upgrade",
                 scope)
    decoy = add(store, embedder,
                "the checkout service returned error code ERR_9912 after the tls upgrade",
                scope)
    add(store, embedder, "the checkout service was migrated last quarter", scope)
    add(store, embedder, "tls certificates rotate every ninety days", scope)

    # 1. The embedder is provably blind here: identical vectors, so cosine ranking
    #    between these two claims carries exactly zero information.
    vectors = embedder.encode([target.text, decoy.text])
    assert np.array_equal(vectors[0], vectors[1])

    prose_query = embedder.encode(["checkout error code ERR_7734"])[0]
    vector_scores = dict(store.vector_search(prose_query, scope.ancestors(), 10))
    assert vector_scores[target.id] == vector_scores[decoy.id]

    # 2. BM25 has no such trouble.
    lexical = store.lexical_search("ERR_7734", scope.ancestors(), 10)
    assert lexical[0][0] == target.id
    assert lexical[0][1] > lexical[1][1]

    # 3. Hybrid therefore answers correctly, and the explanation says why: the vector
    #    leg contributed nothing at all to this result.
    results = retriever.search("ERR_7734", scope, k=4)
    assert ids(results)[0] == target.id
    assert results[0].explain.vector_rank is None
    assert results[0].explain.lexical_rank == 0


def test_result_found_only_by_the_vector_leg_reports_no_lexical_rank(
    store: SQLiteStore, embedder: HashingEmbedder, retriever: HybridRetriever
) -> None:
    """The mirror image, with the real embedder: "lisboa" shares no stemmed token with
    "lisbon", so BM25 never sees it, but character n-grams do. A `None` rank is the
    finding - this claim surfaced on one retriever's evidence alone."""
    scope = Scope("acme", "alice")
    near_miss = add(store, embedder, "alice relocated to lisboa in the spring", scope)

    results = retriever.search("lisbon", scope, k=5)
    by_id = {r.claim.id: r for r in results}

    assert near_miss.id in by_id
    assert store.lexical_search("lisbon", scope.ancestors(), 10) == []
    assert by_id[near_miss.id].explain.vector_rank == 0
    assert by_id[near_miss.id].explain.lexical_rank is None
    assert by_id[near_miss.id].explain.lexical_score is None


def test_claim_with_no_embedding_is_still_retrievable(
    store: SQLiteStore, embedder: HashingEmbedder, retriever: HybridRetriever
) -> None:
    """A claim the write path never got round to embedding is invisible to a
    vector-only store. BM25 does not care, so hybrid still finds it."""
    scope = Scope("acme", "alice")
    unembedded = add(store, None, "alice is allergic to shellfish", scope,
                     predicate="allergic_to")

    assert store.vector_search(embedder.encode(["shellfish"])[0], [scope], 10) == []

    results = retriever.search("shellfish allergy", scope, k=5)

    assert ids(results) == [unembedded.id]
    assert results[0].explain.vector_rank is None
    assert results[0].explain.lexical_rank == 0


def test_cjk_query_is_answered_by_the_lexical_leg_alone(
    store: SQLiteStore, embedder: HashingEmbedder, retriever: HybridRetriever
) -> None:
    """`HashingEmbedder`'s word regex is ASCII-only, so a Japanese query embeds to a
    zero vector. FTS5's unicode61 tokenizer has no such gap, and the hybrid degrades to
    BM25 instead of returning a ranking that is really just index order."""
    scope = Scope("acme", "alice")
    jp = add(store, embedder, "ユーザー は 東京 に 住んでいる", scope, predicate="lives_in")
    add(store, embedder, "the user lives in tokyo", scope, predicate="lives_in")

    assert float(np.linalg.norm(embedder.encode(["東京"])[0])) == 0.0

    results = retriever.search("東京", scope, k=5)

    assert ids(results) == [jp.id]
    assert results[0].explain.vector_rank is None
    assert results[0].explain.lexical_rank == 0


# ===========================================================================
# Explanations
# ===========================================================================


def test_every_result_carries_a_populated_explanation(
    store: SQLiteStore, embedder: HashingEmbedder, retriever: HybridRetriever
) -> None:
    scope = Scope("acme", "alice")
    for text in ("alice lives in Lisbon", "alice works at Acme", "alice prefers pytest",
                 "alice dislikes early meetings"):
        add(store, embedder, text, scope)

    results = retriever.search("what does alice prefer in Lisbon", scope, k=4)

    assert results
    for r in results:
        e = r.explain
        assert e is not None
        assert e.fusion_score > 0.0
        assert e.final_score == pytest.approx(r.score)
        assert 0.0 <= e.recency <= 1.0
        assert e.confidence == r.claim.confidence
        assert e.salience == r.claim.salience
        # Surfaced by at least one retriever, or it would not be here at all.
        assert e.vector_rank is not None or e.lexical_rank is not None
        # A rank always travels with its raw score, and vice versa.
        assert (e.vector_rank is None) == (e.vector_score is None)
        assert (e.lexical_rank is None) == (e.lexical_score is None)
        assert e.rerank_score is None  # no cross-encoder in this tier
        assert isinstance(e.summary(), str) and e.summary()


def test_explanation_ranks_match_the_underlying_retrievers(
    store: SQLiteStore, embedder: HashingEmbedder, retriever: HybridRetriever
) -> None:
    """The explanation must be the real thing, not a plausible reconstruction.

    The lexical leg is asked about the *analyzed* query - "in" never reaches the store
    - so that is what the expectation is built from. Comparing against the raw string
    would be reconstructing a query the retriever never issued.
    """
    scope = Scope("acme", "alice")
    for text in ("alice lives in Lisbon", "alice moved to Lisbon last year",
                 "alice visits Lisbon often", "bob lives in Berlin"):
        add(store, embedder, text, scope)

    results = retriever.search("lives in Lisbon", scope, k=10)

    expected_vector = {cid: (i, s) for i, (cid, s) in
                       enumerate(store.vector_search(
                           embedder.encode(["lives in Lisbon"])[0], scope.ancestors(), 50))}
    expected_lexical = {cid: (i, s) for i, (cid, s) in
                        enumerate(store.lexical_search(analyze("lives in Lisbon").text,
                                                       scope.ancestors(), 50))}
    for r in results:
        v = expected_vector.get(r.claim.id)
        lx = expected_lexical.get(r.claim.id)
        assert r.explain.vector_rank == (None if v is None else v[0])
        assert r.explain.lexical_rank == (None if lx is None else lx[0])
        if v is not None:
            assert r.explain.vector_score == pytest.approx(v[1])
        if lx is not None:
            assert r.explain.lexical_score == pytest.approx(lx[1])


def test_agreement_between_retrievers_beats_a_single_strong_hit(
    store: SQLiteStore, embedder: HashingEmbedder, retriever: HybridRetriever
) -> None:
    scope = Scope("acme", "alice")
    for text in ("alice lives in Lisbon", "alice relocated to lisboa in the spring"):
        add(store, embedder, text, scope)

    results = retriever.search("lives in Lisbon", scope, k=5)
    top = results[0].explain

    assert top.vector_rank is not None and top.lexical_rank is not None
    assert top.fusion_score > results[-1].explain.fusion_score


# ===========================================================================
# Filters, limits, determinism
# ===========================================================================


def test_memory_types_filter_restricts_the_result_set(
    store: SQLiteStore, embedder: HashingEmbedder, retriever: HybridRetriever
) -> None:
    scope = Scope("acme", "alice")
    semantic = add(store, embedder, "alice works at Acme", scope,
                   memory_type=MemoryType.SEMANTIC)
    episodic = add(store, embedder, "alice shipped the release on tuesday", scope,
                   memory_type=MemoryType.EPISODIC)
    procedural = add(store, embedder, "alice wants tests run before every release", scope,
                     memory_type=MemoryType.PROCEDURAL)

    query = "alice release work"
    assert set(ids(retriever.search(query, scope, k=10))) == {
        semantic.id, episodic.id, procedural.id}
    assert ids(retriever.search(query, scope, k=10,
                                memory_types=[MemoryType.EPISODIC])) == [episodic.id]
    assert set(ids(retriever.search(query, scope, k=10, memory_types=[
        MemoryType.SEMANTIC, MemoryType.PROCEDURAL]))) == {semantic.id, procedural.id}
    assert retriever.search(query, scope, k=10, memory_types=[]) == []


def test_k_bounds_the_result_count(
    store: SQLiteStore, embedder: HashingEmbedder, retriever: HybridRetriever
) -> None:
    scope = Scope("acme", "alice")
    for i in range(25):
        add(store, embedder, f"alice noted deployment detail number {i}", scope)

    for k in (1, 3, 10, 25, 100):
        assert len(retriever.search("deployment detail", scope, k=k)) == min(k, 25)


def test_non_positive_k_returns_empty(
    store: SQLiteStore, embedder: HashingEmbedder, retriever: HybridRetriever
) -> None:
    scope = Scope("acme", "alice")
    add(store, embedder, "alice lives in Lisbon", scope)

    assert retriever.search("Lisbon", scope, k=0) == []
    assert retriever.search("Lisbon", scope, k=-5) == []


def test_top_k_is_a_prefix_of_a_wider_search(
    store: SQLiteStore, embedder: HashingEmbedder, retriever: HybridRetriever
) -> None:
    """Widening k must not reshuffle what was already returned.

    Not automatic: k also sets the candidate pool, so this asserts the pool is wide
    enough that the head of the ranking has stabilised.
    """
    scope = Scope("acme", "alice")
    for i in range(20):
        add(store, embedder, f"alice noted deployment detail number {i}", scope)

    narrow = ids(retriever.search("deployment detail", scope, k=5))
    wide = ids(retriever.search("deployment detail", scope, k=20))

    assert wide[:5] == narrow


def test_ranking_is_deterministic_across_repeated_searches(
    store: SQLiteStore, embedder: HashingEmbedder, retriever: HybridRetriever
) -> None:
    """Unstable ranking makes a retrieval regression impossible to bisect.

    `as_of` is pinned so the comparison covers scores as well as order. Without it the
    scores drift in the last few digits between calls, which is correct - decay is
    continuous and the wall clock really has advanced - but it is not what this test is
    about. The order guarantee under a live clock is asserted separately below.
    """
    now = datetime(2026, 8, 8, tzinfo=timezone.utc)
    scope = Scope("acme", "alice")
    for i in range(30):
        add(store, embedder, f"alice noted deployment detail number {i}", scope,
            valid_from=now - timedelta(days=1), recorded_at=now - timedelta(days=1))

    runs = [retriever.search("deployment detail", scope, k=10, as_of=now) for _ in range(5)]
    baseline = [(r.claim.id, r.score) for r in runs[0]]

    for run in runs[1:]:
        assert [(r.claim.id, r.score) for r in run] == baseline


def test_ranking_order_is_stable_under_a_live_clock(
    store: SQLiteStore, embedder: HashingEmbedder, retriever: HybridRetriever
) -> None:
    """The guarantee callers actually rely on: the same question asked twice in a row
    returns the same claims in the same order, even though decay is recomputed."""
    scope = Scope("acme", "alice")
    for i in range(30):
        add(store, embedder, f"alice noted deployment detail number {i}", scope)

    runs = [ids(retriever.search("deployment detail", scope, k=10)) for _ in range(5)]

    assert all(run == runs[0] for run in runs[1:])


def test_identical_claims_are_ordered_stably_by_id(
    store: SQLiteStore, embedder: HashingEmbedder, retriever: HybridRetriever
) -> None:
    """Byte-identical claims tie on every signal, so ordering falls through to the id
    tiebreak rather than to whatever order the store happened to enumerate."""
    scope = Scope("acme", "alice")
    made = [add(store, embedder, "alice lives in Lisbon", scope) for _ in range(6)]

    results = retriever.search("Lisbon", scope, k=6)
    scores = [r.score for r in results]

    assert len(results) == 6
    assert scores == sorted(scores, reverse=True)
    # Wherever the fused scores tie, the ids must be ascending within that group.
    for a, b in zip(results, results[1:]):
        if a.score == b.score:
            assert a.claim.id < b.claim.id
    assert {r.claim.id for r in results} == {c.id for c in made}


def test_results_are_sorted_by_descending_final_score(
    store: SQLiteStore, embedder: HashingEmbedder, retriever: HybridRetriever
) -> None:
    scope = Scope("acme", "alice")
    for i in range(15):
        add(store, embedder, f"alice noted deployment detail number {i}", scope,
            confidence=0.5 + i / 40.0, salience=0.4 + i / 30.0)

    results = retriever.search("deployment detail", scope, k=15)
    scores = [r.score for r in results]

    assert scores == sorted(scores, reverse=True)
    assert all(r.score == r.explain.final_score for r in results)


def test_tuning_weights_change_the_ranking(
    store: SQLiteStore, embedder: HashingEmbedder
) -> None:
    """The knobs are real. Zeroing every quality weight must leave the score equal to
    the retrievers' own evidence, with nothing rescaling it."""
    scope = Scope("acme", "alice")
    add(store, embedder, "alice lives in Lisbon", scope, confidence=0.4, salience=0.2)

    plain = HybridRetriever(store, embedder, PredicateRegistry(),
                            w_recency=0.0, w_confidence=0.0, w_salience=0.0)
    weighted = HybridRetriever(store, embedder, PredicateRegistry())
    result = plain.search("Lisbon", scope, k=1)[0]
    quality_scaled = weighted.search("Lisbon", scope, k=1)[0]

    evidence = relevance(
        vector=vector_relevance(result.explain.vector_score),
        lexical=lexical_relevance(result.explain.lexical_score,
                                  len(analyze("Lisbon").terms)),
        w_vector=1.0, w_lexical=1.0,
    )
    assert result.score == pytest.approx(evidence)
    # A low-confidence, low-salience claim is penalised once the weights are live.
    assert quality_scaled.score < result.score


def test_lexical_only_retriever_ignores_the_vector_leg(
    store: SQLiteStore, embedder: HashingEmbedder
) -> None:
    scope = Scope("acme", "alice")
    add(store, embedder, "alice relocated to lisboa in the spring", scope)

    lexical_only = HybridRetriever(store, embedder, PredicateRegistry(), w_vector=0.0)

    # Reachable only through the vector leg, which is now switched off.
    assert lexical_only.search("lisbon", scope, k=5) == []


# ===========================================================================
# Stopword-aware lexical retrieval
# ===========================================================================


def test_the_analyzer_tokenizes_exactly_as_the_store_does() -> None:
    """The two must agree about what a term is. If they disagree, the per-term BM25
    normalization divides by a count the store never used, and every score built on it
    is quietly wrong."""
    from memvara.store.sqlite import _fts_query

    for query in ("ERR_7734 in prod", "what's my mother's maiden name?",
                  "PLAT-2291", "東京 に 住んでいる", "a b cd", "!!! ???"):
        expected = [t.strip('"') for t in _fts_query(query).split(" OR ") if t]
        assert tokenize(query) == expected, query


def test_an_unspaced_script_becomes_one_token_and_a_substring_of_it_finds_nothing() -> None:
    """The limitation `tokenize`'s docstring describes, asserted rather than only stated.

    CJK survives tokenization — it is alphanumeric, so nothing discards it — and that is
    the sentence the docstring used to stop at. What it does not do is *segment*: with no
    spaces to split on, a contiguous run comes out as one token holding the entire phrase,
    so a search for a word inside it matches no term in the index.

    Pinned here because the old wording read as though the case were handled, and a claim
    about behaviour that no test exercises is one that can quietly stop being true — or,
    as here, can never have been true in the way a reader took it.
    """
    phrase = "我住在里斯本"
    assert tokenize(phrase) == [phrase], "the whole run, indexed under itself"

    # The substring a user would actually search for is a term the index does not hold.
    assert tokenize("里斯本") == ["里斯本"]
    assert "里斯本" not in tokenize(phrase)

    # Latin is unaffected, which is why this is easy to miss. Stopwords are a later
    # stage, so "in" is still a token here and "I" is gone only for being one character.
    assert tokenize("I live in Lisbon") == ["live", "in", "lisbon"]


def test_only_closed_class_words_are_stopwords() -> None:
    """The list is defensible only because membership of these classes is fixed by the
    grammar. Anything that could be the content of a memory has to stay out - "never"
    is a whole predicate, and "name", "live" and "work" are half the corpus."""
    for content in ("name", "names", "never", "live", "lives", "work", "know",
                    "like", "call", "called", "own", "owns", "prefer", "allergic"):
        assert content not in STOPWORDS, content

    for function_word in ("the", "of", "is", "do", "what", "my", "any", "about"):
        assert function_word in STOPWORDS, function_word


def test_single_characters_are_dropped_and_case_is_folded() -> None:
    assert tokenize("A b Cd EF") == ["cd", "ef"]
    assert analyze("Lisbon LISBON lisbon").terms == ("lisbon",)


@pytest.fixture
def personal(store: SQLiteStore, embedder: HashingEmbedder) -> dict[str, Claim]:
    """A small personal store, of the shape that makes stopwords dangerous.

    Note what is *not* here: the word "do" appears in exactly one claim. At this scale
    a stopword is the rarest token in the corpus, so BM25 hands it the largest IDF -
    the pathology gets worse as the store gets smaller, not better.
    """
    scope = Scope("acme", "alice")
    return {
        "chore": add(store, embedder, "alice asked the agent to do the weekly rollup",
                     scope, predicate="reported"),
        "city": add(store, embedder, "alice lives in Lisbon", scope, predicate="lives_in"),
        "employer": add(store, embedder, "alice works at Acme Robotics", scope,
                        predicate="works_at"),
        "allergy": add(store, embedder, "alice is allergic to shellfish", scope,
                       predicate="allergic_to"),
    }


def test_a_query_of_pure_stopwords_makes_the_lexical_leg_abstain(
    store: SQLiteStore, retriever: HybridRetriever, personal: dict[str, Claim]
) -> None:
    """The measured bug: `"what do you know about me?"` is six full-weight terms with
    no content in them, and the one claim containing "do" came back as the confident
    #1 answer to three unrelated questions.

    The lexical leg must abstain outright rather than rank on stopword IDF, exactly as
    the vector leg abstains on a zero-norm query. A fabricated ranking is worse than no
    ranking, because fusion reads positions and cannot tell the two apart.
    """
    scope = Scope("acme", "alice")
    # The store, asked the raw question, is happy to answer it - the guard cannot live
    # there, and this is what it is guarding against.
    raw = store.lexical_search("what do you know about me?", scope.ancestors(), 10)
    assert raw and raw[0][0] == personal["chore"].id

    results = retriever.search("what do you know about me?", scope, k=5)

    assert all(r.explain.lexical_rank is None for r in results)
    # The vector leg still has an opinion - cosine is defined for every pair - but with
    # no lexical corroboration behind it, everything it found scores below the weakest
    # answer this store gives to a question it can actually answer.
    answerable = retriever.search("what is alice allergic to?", scope, k=5)
    assert max(r.score for r in results) < answerable[0].score


def test_content_terms_survive_and_stopwords_are_dropped(
    store: SQLiteStore, retriever: HybridRetriever, personal: dict[str, Claim]
) -> None:
    """Reduction, not abstention, when the query does carry content."""
    scope = Scope("acme", "alice")

    assert analyze("where does she live?").terms == ("live",)
    results = retriever.search("where does she live?", scope, k=5)

    assert ids(results)[0] == personal["city"].id
    assert results[0].explain.lexical_rank == 0


def test_stopwords_no_longer_drag_an_unrelated_claim_to_first_place(
    store: SQLiteStore, retriever: HybridRetriever, personal: dict[str, Claim]
) -> None:
    """Three unrelated questions, none of which is about the weekly rollup."""
    scope = Scope("acme", "alice")

    for query, expected in (
        ("what do you do about the shellfish?", personal["allergy"]),
        ("where is it that they do work?", personal["employer"]),
        ("do you know what city she is in? Lisbon?", personal["city"]),
    ):
        assert ids(retriever.search(query, scope, k=5))[0] == expected.id, query


def test_repeated_terms_do_not_inflate_the_query(
    store: SQLiteStore, embedder: HashingEmbedder, retriever: HybridRetriever
) -> None:
    """A pasted document repeats its words; BM25 would weight them once per repeat and
    the per-term normalization would divide by a count that means nothing."""
    scope = Scope("acme", "alice")
    add(store, embedder, "alice lives in Lisbon", scope)

    assert analyze("lisbon lisbon lisbon lisbon").terms == ("lisbon",)
    once = retriever.search("lisbon", scope, k=1)[0]
    repeated = retriever.search("lisbon lisbon lisbon lisbon", scope, k=1)[0]

    assert repeated.explain.lexical_score == pytest.approx(once.explain.lexical_score)


def test_a_cjk_query_still_reaches_the_lexical_leg(
    store: SQLiteStore, embedder: HashingEmbedder, retriever: HybridRetriever
) -> None:
    """The stopword list is English, and the tokenizer is unicode-aware, so a script
    the analyzer knows nothing about must pass through untouched rather than being
    filtered into silence."""
    scope = Scope("acme", "alice")
    jp = add(store, embedder, "ユーザー は 東京 に 住んでいる", scope, predicate="lives_in")

    assert analyze("東京").terms == ("東京",)
    assert ids(retriever.search("東京", scope, k=5)) == [jp.id]


# ===========================================================================
# Normalized scores, and being able to say "nothing relevant"
# ===========================================================================


def test_every_score_is_a_normalized_relevance_in_the_unit_interval(
    store: SQLiteStore, embedder: HashingEmbedder, retriever: HybridRetriever,
    personal: dict[str, Claim]
) -> None:
    scope = Scope("acme", "alice")

    for query in ("shellfish", "where does alice live", "acme robotics", "rollup"):
        for r in retriever.search(query, scope, k=5):
            assert 0.0 <= r.score <= 1.0
            assert r.score == pytest.approx(r.explain.final_score)


def test_the_raw_fusion_product_is_preserved_for_debugging(
    store: SQLiteStore, retriever: HybridRetriever, personal: dict[str, Claim]
) -> None:
    """Contract B: the normalized number is what callers threshold on, the raw one is
    what you read when a ranking changes and you need to know which half moved."""
    scope = Scope("acme", "alice")

    top = retriever.search("shellfish allergy", scope, k=5)[0]

    assert top.explain.raw_score == pytest.approx(
        top.explain.fusion_score * (1 + 0.25 * top.explain.recency
                                    + 0.15 * top.claim.confidence
                                    + 0.10 * top.claim.salience))
    # The raw product is the one that cannot be thresholded - two orders of magnitude
    # below the normalized score, and capped by `rrf_k` rather than by relevance.
    assert top.explain.raw_score < 0.05 < top.score


def test_an_unanswerable_question_scores_below_an_answerable_one(
    store: SQLiteStore, retriever: HybridRetriever, personal: dict[str, Claim]
) -> None:
    """The headline: 6 of 6 unanswerable queries used to come back confident, and
    `"what is my mother's maiden name?"` scored *identically* to the best answerable
    query. The absolute evidence for it is near zero and now the score says so.

    Asserted as an ordering, not against a constant. The gap is real and stable; where
    it sits on the number line is a property of this corpus's size, which is why the
    library ships a calibrator instead of a floor.
    """
    scope = Scope("acme", "alice")

    for question in ("what is the capital of France?",
                     "what is my mother's maiden name?",
                     "what did the CFO say on the earnings call"):
        unanswerable = retriever.search(question, scope, k=5)
        answerable = retriever.search("what is alice allergic to?", scope, k=5)

        assert unanswerable, "still returned, with an honest score attached"
        assert unanswerable[0].score < answerable[0].score, question


def test_a_calibrated_floor_silences_the_unanswerable_and_keeps_the_rest(
    store: SQLiteStore, retriever: HybridRetriever, personal: dict[str, Claim]
) -> None:
    """End to end: measure the floor from probes, then use it as `min_score`."""
    scope = Scope("acme", "alice")

    def search(query: str):
        return retriever.search(query, scope, k=5)

    report = calibrate_min_score(
        search,
        answerable=["what is alice allergic to?", "where does alice live?",
                    "where does alice work?"],
        unanswerable=["what is the capital of France?",
                      "what is my mother's maiden name?",
                      "what do you know about me?"],
    )

    assert report.separable
    assert report.kept == report.answerable == 3
    assert report.silenced == report.unanswerable == 3
    assert "clean" in str(report)
    assert retriever.search("what is the capital of France?", scope, k=5,
                            min_score=report.floor) == []
    assert retriever.search("what is alice allergic to?", scope, k=5,
                            min_score=report.floor)


def test_the_calibrated_floor_moves_with_corpus_size(
    store: SQLiteStore, embedder: HashingEmbedder, retriever: HybridRetriever,
    personal: dict[str, Claim]
) -> None:
    """Why there is no `RELEVANCE_FLOOR` constant any more.

    The same claims, the same questions, the same embedder - only unrelated filler is
    added, and the floor that separates answerable from unanswerable moves. A constant
    calibrated at one size silences correct answers at the other.
    """
    scope = Scope("acme", "alice")
    probes = {
        "answerable": ["what is alice allergic to?", "where does alice live?"],
        "unanswerable": ["what is the capital of France?",
                         "what did the CFO say on the earnings call"],
    }

    small = calibrate_min_score(lambda q: retriever.search(q, scope, k=5), **probes)
    for i in range(120):
        add(store, embedder, f"the sprint board was retagged as batch {i} on friday",
            scope, predicate="reported")
    large = calibrate_min_score(lambda q: retriever.search(q, scope, k=5), **probes)

    assert large.floor > small.floor * 1.2, (small, large)


def test_min_score_filters_on_the_normalized_value(
    store: SQLiteStore, retriever: HybridRetriever, personal: dict[str, Claim]
) -> None:
    scope = Scope("acme", "alice")

    everything = retriever.search("alice lisbon shellfish", scope, k=10)
    assert len(everything) > 1

    for floor in (0.0, 0.1, 0.3, 0.5):
        kept = retriever.search("alice lisbon shellfish", scope, k=10, min_score=floor)
        assert ids(kept) == [r.claim.id for r in everything if r.score >= floor]

    assert retriever.search("alice lisbon shellfish", scope, k=10, min_score=1.01) == []


def test_the_floor_is_applied_before_k_not_after(
    store: SQLiteStore, retriever: HybridRetriever, personal: dict[str, Claim]
) -> None:
    """Otherwise a floor that rejects the top result would return k-1 things and look
    like a bug in `k`."""
    scope = Scope("acme", "alice")

    unfiltered = retriever.search("alice", scope, k=2)
    floor = min(r.score for r in unfiltered) + 1e-9
    filtered = retriever.search("alice", scope, k=2, min_score=floor)

    assert len(filtered) < len(unfiltered)
    assert all(r.score >= floor for r in filtered)


# ===========================================================================
# Filter starvation
# ===========================================================================


def test_a_narrow_filter_is_retried_against_a_wider_candidate_pool(
    store: SQLiteStore, embedder: HashingEmbedder, retriever: HybridRetriever
) -> None:
    """`memory_types` is applied after fusion truncated the pool, so the filter can
    starve: matches exist, but every slot in the pool was spent on claims the filter
    then rejected. Measured as returning 0 of 3 with a pool of 10.
    """
    scope = Scope("acme", "alice")
    for i in range(60):
        add(store, embedder, f"alice noted release detail number {i}", scope,
            memory_type=MemoryType.SEMANTIC)
    procedural = [
        add(store, embedder, "alice wants the release checklist followed", scope,
            memory_type=MemoryType.PROCEDURAL),
        add(store, embedder, "alice wants release notes written first", scope,
            memory_type=MemoryType.PROCEDURAL),
        add(store, embedder, "alice wants release tags signed", scope,
            memory_type=MemoryType.PROCEDURAL),
    ]

    starved = HybridRetriever(store, embedder, PredicateRegistry(),
                              filter_retry_multiplier=1, candidate_floor=0)
    query = "release detail"

    # Without the retry the pool of 10 is spent entirely on semantic claims, and a
    # filter with three live matches behind it returns nothing at all. The floor is
    # off so the pool *is* 10: at the shipped 50 it would hold every claim here.
    assert starved.search(query, scope, k=2, memory_types=[MemoryType.PROCEDURAL]) == []

    found = retriever.search(query, scope, k=2, memory_types=[MemoryType.PROCEDURAL])
    assert len(found) == 2
    assert {r.claim.id for r in found} <= {c.id for c in procedural}
    assert len(retriever.search(query, scope, k=3,
                                memory_types=[MemoryType.PROCEDURAL])) == 3


def test_the_retry_does_not_fire_when_the_pool_was_never_full(
    store: SQLiteStore, embedder: HashingEmbedder
) -> None:
    """A short result set from a leg that returned fewer hits than it was allowed has
    nothing more to give, and re-asking is pure cost."""
    scope = Scope("acme", "alice")
    add(store, embedder, "alice lives in Lisbon", scope, memory_type=MemoryType.SEMANTIC)

    calls: list[int] = []

    class CountingStore(SQLiteStore):
        def lexical_search(self, query, scopes, limit, *, valid_at=None,
                           known_at=None, states=None, include_invalidated=None):
            calls.append(limit)
            return super().lexical_search(
                query, scopes, limit, valid_at=valid_at, known_at=known_at,
                states=states, include_invalidated=include_invalidated)

    counting = CountingStore(":memory:")
    add(counting, embedder, "alice lives in Lisbon", scope)
    retriever = HybridRetriever(counting, embedder, PredicateRegistry())

    assert retriever.search("lisbon", scope, k=5,
                            memory_types=[MemoryType.PROCEDURAL]) == []
    assert calls == [50], "one pass only: the pool was never truncated"
    counting.close()


# ===========================================================================
# The candidate floor
# ===========================================================================


class OrderedLegs(SQLiteStore):
    """A real store whose vector leg returns a fixed ordering cut at `limit`.

    The test needs a claim at an exact position in a leg's own ranking, which no
    embedder gives deterministically. Everything else — the rows, hydration, the
    belief filters, the lexical leg's abstention — is `SQLiteStore` as shipped.
    """

    def __init__(self) -> None:
        super().__init__(":memory:")
        self.order: list[tuple[str, float]] = []
        self.limits: list[int] = []

    def vector_search(self, qvec, scopes, limit, **kw):
        self.limits.append(limit)
        return self.order[:limit]

    def lexical_search(self, query, scopes, limit, **kw):
        return []


def test_a_claim_past_k_times_the_multiplier_still_reaches_fusion_at_a_small_k() -> None:
    """memvara/memvara#155: the window was a pure multiple of `k`, so a caller asking
    for four results searched 20 deep and lost the best-scoring claim entirely.

    Twenty-one stale, low-quality claims outrank the answer on cosine alone, so the
    vector leg puts it 22nd. Rescoring would put it first — its evidence is a little
    lower and its quality is a lot higher — but rescoring only sees what the leg
    returned. At the shipped floor the leg returns 50 and the answer wins; with the
    floor off the leg returns 20 and the answer is not ranked low, it is absent.
    """
    scope = Scope("acme", "alice")
    store = OrderedLegs()
    fillers = [add(store, None, f"release note number {i}", scope,
                   confidence=0.0, salience=0.0) for i in range(21)]
    gold = add(store, None, "the version number cannot be trusted", scope)
    store.order = ([(c.id, 0.70 - i * 0.004) for i, c in enumerate(fillers)]
                   + [(gold.id, 0.60)])

    def retriever(**kw) -> HybridRetriever:
        return HybridRetriever(store, HashingEmbedder(dim=64), PredicateRegistry(),
                               w_lexical=0.0, w_recency=0.0, **kw)

    floored = retriever().search("version", scope, k=4)
    unfloored = retriever(candidate_floor=0).search("version", scope, k=4)

    assert ids(floored)[0] == gold.id
    assert gold.id not in ids(unfloored)
    assert store.limits == [50, 20]


@pytest.mark.parametrize("k, window", [(1, 50), (4, 50), (10, 50), (11, 55), (20, 100)])
def test_the_window_is_the_floor_until_the_multiple_passes_it(k: int, window: int) -> None:
    """`max(k * 5, 50)`: the floor decides at small `k` and costs nothing at large."""
    store = OrderedLegs()
    HybridRetriever(store, HashingEmbedder(dim=64), PredicateRegistry()).search(
        "anything", Scope("acme", "alice"), k=k)
    assert store.limits == [window]


# ===========================================================================
# Per-slot diversity
# ===========================================================================


@pytest.fixture
def cluster(store: SQLiteStore, embedder: HashingEmbedder) -> dict[str, list[Claim]]:
    """One slot holding five near-identical claims, plus four other topics."""
    scope = Scope("acme", "alice")
    dupes = [
        add(store, embedder, f"alice rated the standup format four out of five{tail}",
            scope, predicate="rated")
        for tail in ("", " today", " last week", " again", " as always")
    ]
    others = [
        add(store, embedder, "alice wants the standup moved to eleven", scope,
            predicate="prefers"),
        add(store, embedder, "alice finds the standup format too rigid", scope,
            predicate="dislikes"),
        add(store, embedder, "the standup rating survey closes on friday", scope,
            predicate="reported"),
        add(store, embedder, "alice skipped standup twice this format cycle", scope,
            predicate="attended"),
    ]
    return {"dupes": dupes, "others": others}


def test_one_slot_cannot_take_the_whole_result_set(
    retriever: HybridRetriever, cluster: dict[str, list[Claim]]
) -> None:
    """Measured: a cluster of near-identical claims took 5 of 8 prompt slots. Every one
    of them is the same answer, so four of those slots bought nothing."""
    scope = Scope("acme", "alice")

    results = retriever.search("standup format rating", scope, k=6)
    dupe_ids = {c.id for c in cluster["dupes"]}

    assert sum(1 for r in results if r.claim.id in dupe_ids) == 2
    assert len({r.claim.fact_key for r in results}) == 5


def test_diversity_demotes_rather_than_drops(
    retriever: HybridRetriever, cluster: dict[str, list[Claim]]
) -> None:
    """`k` must keep meaning "at most k results". Dropping the overflow would make the
    result count depend on how the corpus happened to cluster, and would hide the
    cluster from anyone auditing it."""
    scope = Scope("acme", "alice")

    wide = retriever.search("standup format rating", scope, k=9)
    dupe_ids = {c.id for c in cluster["dupes"]}

    assert len(wide) == 9
    assert dupe_ids <= set(ids(wide)), "every duplicate is still reachable"
    # Two keep their earned place in the head; the other three sit at the very back,
    # behind the topics they were crowding out.
    positions = [i for i, r in enumerate(wide) if r.claim.id in dupe_ids]
    assert len(positions) == 5
    assert positions[1] < 6
    assert positions[2:] == [6, 7, 8]


def test_the_slot_cap_is_tunable_and_switchable(
    store: SQLiteStore, embedder: HashingEmbedder, cluster: dict[str, list[Claim]]
) -> None:
    scope = Scope("acme", "alice")
    dupe_ids = {c.id for c in cluster["dupes"]}

    def dupes_in_head(max_per_slot: int) -> int:
        r = HybridRetriever(store, embedder, PredicateRegistry(),
                            max_per_slot=max_per_slot)
        return sum(1 for x in r.search("standup format rating", scope, k=5)
                   if x.claim.id in dupe_ids)

    assert dupes_in_head(1) == 1
    assert dupes_in_head(2) == 2
    # 0 disables the cap, and the cluster immediately takes almost the whole set back.
    assert dupes_in_head(0) == 4


def test_diversity_does_not_reorder_within_a_slot(
    retriever: HybridRetriever, cluster: dict[str, list[Claim]]
) -> None:
    """Demotion preserves relative order, so the best duplicate is still the one that
    represents the slot and the audit trail reads the same way."""
    scope = Scope("acme", "alice")
    dupe_ids = {c.id for c in cluster["dupes"]}

    capped = [r for r in retriever.search("standup format rating", scope, k=9)
              if r.claim.id in dupe_ids]
    uncapped = [r for r in HybridRetriever(
        retriever.store, retriever.embedder, PredicateRegistry(), max_per_slot=0
    ).search("standup format rating", scope, k=9) if r.claim.id in dupe_ids]

    assert ids(capped) == ids(uncapped)


# ===========================================================================
# Degenerate and adversarial input
# ===========================================================================


ADVERSARIAL = [
    pytest.param("", id="empty"),
    pytest.param("   \t\n  ", id="whitespace"),
    pytest.param("!!! ??? ...", id="punctuation"),
    pytest.param("*", id="bare_star"),
    pytest.param("a AND (b", id="unbalanced_fts_operator"),
    pytest.param('he said "hello', id="unterminated_quote"),
    pytest.param("NEAR(alice bob)", id="fts_near_operator"),
    pytest.param("text:alice", id="fts_column_filter"),
    pytest.param("^alice", id="fts_initial_token"),
    pytest.param("-alice -bob", id="fts_negation"),
    pytest.param("alice OR OR OR", id="dangling_or"),
    pytest.param("(((((", id="open_parens"),
    pytest.param("🙂🙂🙂", id="emoji_only"),
    pytest.param("Ünïcödé sûr naïve", id="accented"),
    pytest.param("東京 に 住んでいる", id="cjk"),
    pytest.param("x" * 10_000, id="ten_thousand_chars"),
    pytest.param(" ".join(f"tok{i}" for i in range(1500)), id="fifteen_hundred_terms"),
    pytest.param("quetzalcoatl xylophone bandersnatch", id="matches_nothing"),
]


@pytest.mark.parametrize("query", ADVERSARIAL)
def test_adversarial_queries_return_cleanly(
    store: SQLiteStore, embedder: HashingEmbedder, retriever: HybridRetriever, query: str
) -> None:
    """These are the strings that crash a naive `MATCH ?`. None may raise, and none may
    return a half-built result: an empty list is a fine answer, an exception is not."""
    scope = Scope("acme", "alice")
    for text in ("alice lives in Lisbon", "alice works at Acme", "bob prefers pytest"):
        add(store, embedder, text, scope)

    results = retriever.search(query, scope, k=5)

    assert isinstance(results, list)
    assert len(results) <= 5
    for r in results:
        assert r.explain is not None
        assert r.score == pytest.approx(r.explain.final_score)
        assert r.explain.vector_rank is not None or r.explain.lexical_rank is not None


@pytest.mark.parametrize("query", ["", "   ", "*", "!!!", "🙂"])
def test_signal_free_queries_return_nothing_rather_than_arbitrary_order(
    store: SQLiteStore, embedder: HashingEmbedder, retriever: HybridRetriever, query: str
) -> None:
    """A query with no usable content embeds to a zero vector, which gives every
    candidate an identical cosine of 0.0. Returning that "ranking" would dress index
    order up as relevance, so the vector leg abstains and, with BM25 also empty, the
    honest answer is nothing at all."""
    scope = Scope("acme", "alice")
    for text in ("alice lives in Lisbon", "alice works at Acme"):
        add(store, embedder, text, scope)

    assert retriever.search(query, scope, k=5) == []


def test_query_matching_nothing_returns_the_low_relevance_tail(
    store: SQLiteStore, embedder: HashingEmbedder, retriever: HybridRetriever
) -> None:
    """A word matching no claim is not the same as an empty result, and pretending
    otherwise would be the wrong contract.

    BM25 correctly finds nothing. The vector leg cannot: cosine similarity is defined
    for every pair, so it always produces a ranking of whatever is in scope. So the
    honest behaviour is a result set with no lexical support and near-zero vector
    scores. Callers wanting "nothing relevant" must threshold on the explanation -
    which is precisely why the raw per-retriever scores are exposed on it.
    """
    scope = Scope("acme", "alice")
    for text in ("alice lives in Lisbon", "alice works at Acme"):
        add(store, embedder, text, scope)

    assert store.lexical_search("quetzalcoatl", scope.ancestors(), 10) == []

    results = retriever.search("quetzalcoatl", scope, k=5)

    assert len(results) == 2
    assert all(r.explain.lexical_rank is None for r in results)
    assert all(abs(r.explain.vector_score) < 0.1 for r in results)

    # An empty scope, by contrast, genuinely has nothing to rank.
    assert retriever.search("quetzalcoatl", Scope("acme", "carol"), k=5) == []


def test_empty_store_returns_empty(retriever: HybridRetriever) -> None:
    assert retriever.search("anything at all", Scope("acme", "alice"), k=10) == []
    assert retriever.search("", Scope("acme", "alice"), k=10) == []


def test_store_with_claims_but_no_embeddings_at_all(
    store: SQLiteStore, embedder: HashingEmbedder, retriever: HybridRetriever
) -> None:
    """Nothing has ever been embedded, so the vector index is empty rather than merely
    sparse - a different code path in the store."""
    scope = Scope("acme", "alice")
    for text in ("alice lives in Lisbon", "alice works at Acme"):
        add(store, None, text, scope)

    results = retriever.search("where does alice live", scope, k=5)

    assert len(results) == 2
    assert all(r.explain.vector_rank is None for r in results)
    assert all(r.explain.lexical_rank is not None for r in results)


def test_unicode_text_survives_the_round_trip(
    store: SQLiteStore, embedder: HashingEmbedder, retriever: HybridRetriever
) -> None:
    scope = Scope("acme", "alice")
    claim = add(store, embedder, "Renée lives in Ōsaka and works at Ærø", scope)

    results = retriever.search("Ōsaka", scope, k=5)

    assert ids(results) == [claim.id]
    assert results[0].text == "Renée lives in Ōsaka and works at Ærø"


def test_very_long_query_does_not_blow_up_the_fts_expression(
    store: SQLiteStore, embedder: HashingEmbedder, retriever: HybridRetriever
) -> None:
    """A 10k-character query becomes a very wide OR expression inside FTS5."""
    scope = Scope("acme", "alice")
    claim = add(store, embedder, "alice lives in Lisbon", scope)
    query = "lisbon " + ("padding words that mean nothing " * 400)

    results = retriever.search(query, scope, k=5)

    assert len(query) > 10_000
    assert claim.id in ids(results)


def test_naive_as_of_datetime_is_accepted(
    retriever: HybridRetriever, employment: dict[str, Claim]
) -> None:
    """Hand-built datetimes at API edges are routinely naive."""
    aware = ids(retriever.search("where does the user work", WORK_SCOPE, k=5,
                                 as_of=T0 + timedelta(days=30)))
    naive = ids(retriever.search("where does the user work", WORK_SCOPE, k=5,
                                 as_of=(T0 + timedelta(days=30)).replace(tzinfo=None)))

    assert naive == aware == [employment["old"].id]


# ===========================================================================
# Episodes: the turns no claim was ever extracted from
# ===========================================================================
#
# `WriteReceipt.skipped` used to mean "we stored this and will never find it again".
# These tests are the property that it no longer does, plus the two ways widening
# retrieval could make it worse: raw turn text outranking a curated claim, and a
# transcript's worth of turns crowding out the facts.


EP_SCOPE = Scope("acme", "alice", "agent_a", "sess_1")


def turn(store: SQLiteStore, embedder: HashingEmbedder | None, content: str,
         scope: Scope, **kw) -> Episode:
    ep = Episode(content=content, scope=scope, **kw)
    store.add_episode(ep)
    if embedder is not None:
        store.set_episode_embedding(ep.id, embedder.encode([content])[0])
    return ep


KAFKA = ("We decided at the offsite to sunset the Kafka pipeline because the "
         "ordering guarantees never held.")


def test_a_turn_that_produced_no_claim_is_findable(
    store: SQLiteStore, embedder: HashingEmbedder, retriever: HybridRetriever
) -> None:
    """The whole point. This exact turn is a decision plus its reasoning, which no
    extractor turns into a triple, and it used to be reachable only through `why()` on
    a claim that was never created."""
    ep = turn(store, embedder, KAFKA, EP_SCOPE)

    assert retriever.search("kafka pipeline decision", EP_SCOPE, k=5) == []

    found = retriever.search("kafka pipeline decision", EP_SCOPE, k=5,
                             include_episodes=True)
    assert [r.episode.id for r in found] == [ep.id]
    assert found[0].text == KAFKA


def test_an_episode_result_is_not_a_claim_result(
    store: SQLiteStore, embedder: HashingEmbedder, retriever: HybridRetriever
) -> None:
    """The discriminator. A caller that renders the two identically has turned
    unverified conversation into asserted fact."""
    turn(store, embedder, KAFKA, EP_SCOPE)
    add(store, embedder, "the kafka pipeline is deprecated", EP_SCOPE)

    found = retriever.search("kafka pipeline", EP_SCOPE, k=5, include_episodes=True)
    episodes = [r for r in found if isinstance(r, EpisodeResult)]
    claims = [r for r in found if not isinstance(r, EpisodeResult)]

    assert len(episodes) == 1 and len(claims) == 1
    assert kind_of(episodes[0]) == EPISODE == "episode"
    assert kind_of(claims[0]) == CLAIM == "claim"
    assert not hasattr(episodes[0], "claim"), "an episode has no claim to offer"
    assert repr(episodes[0]).startswith("<EpisodeResult ")
    assert "kafka" in repr(episodes[0]).lower()


def test_an_episode_result_repr_survives_a_turn_with_no_retriever_support(
    store: SQLiteStore, embedder: HashingEmbedder
) -> None:
    """`_short` truncates and the leg list can be empty; both belong in the repr rather
    than in a 1,400-character dataclass dump."""
    ep = Episode(content="x " * 200, scope=EP_SCOPE)
    text = repr(EpisodeResult(episode=ep, score=0.0))
    assert text.endswith(f"no-retriever {ep.id}>")
    assert "…" in text and len(text) < 200


def test_episodes_are_off_by_default_so_existing_prompts_do_not_change(
    store: SQLiteStore, embedder: HashingEmbedder, retriever: HybridRetriever
) -> None:
    turn(store, embedder, KAFKA, EP_SCOPE)
    assert retriever.search("kafka", EP_SCOPE, k=5) == []
    assert retriever.search("kafka", EP_SCOPE, k=5, include_episodes=True) != []


def test_a_claim_outranks_the_turn_it_was_extracted_from(
    store: SQLiteStore, embedder: HashingEmbedder, retriever: HybridRetriever
) -> None:
    """The obvious failure mode: the turn contains the query's words verbatim while the
    claim is a normalized triple that may share none of them, so on raw evidence the
    turn wins. The weight is what stops it."""
    text = "alice lives in Lisbon"
    turn(store, embedder, text, EP_SCOPE)
    claim = add(store, embedder, text, EP_SCOPE)

    found = retriever.search(text, EP_SCOPE, k=5, include_episodes=True)

    assert not isinstance(found[0], EpisodeResult)
    assert found[0].claim.id == claim.id
    assert isinstance(found[1], EpisodeResult)
    assert found[1].score == pytest.approx(found[0].score * 0.5, rel=0.15)


def test_the_weight_is_a_discount_not_an_exclusion(
    store: SQLiteStore, embedder: HashingEmbedder
) -> None:
    """A turn still surfaces above a claim that is merely a worse match — the point is
    ordering under equal evidence, not banishment."""
    turn(store, embedder, KAFKA, EP_SCOPE)
    add(store, embedder, "alice prefers oat milk", EP_SCOPE)
    retriever = HybridRetriever(store, embedder, PredicateRegistry())

    found = retriever.search("kafka ordering guarantees", EP_SCOPE, k=5,
                             include_episodes=True)
    assert isinstance(found[0], EpisodeResult)


def test_the_episode_tail_is_capped(
    store: SQLiteStore, embedder: HashingEmbedder
) -> None:
    """A transcript has far more turns than facts. Uncapped, one well-worded
    conversation takes every slot."""
    for i in range(12):
        turn(store, embedder, f"we talked about the kafka pipeline, part {i}", EP_SCOPE)
    retriever = HybridRetriever(store, embedder, PredicateRegistry(), max_episodes=3)

    found = retriever.search("kafka pipeline", EP_SCOPE, k=10, include_episodes=True)
    assert len(found) == 3

    wider = HybridRetriever(store, embedder, PredicateRegistry(), max_episodes=7)
    assert len(wider.search("kafka pipeline", EP_SCOPE, k=10,
                            include_episodes=True)) == 7


def test_the_cap_never_costs_a_claim_its_place(
    store: SQLiteStore, embedder: HashingEmbedder
) -> None:
    """`k` is the total, and the episodes go in the space claims did not fill."""
    claims = [add(store, embedder, f"alice rated kafka {i} out of five", EP_SCOPE,
                  predicate=f"rated_{i}") for i in range(4)]
    for i in range(6):
        turn(store, embedder, f"kafka came up again, part {i}", EP_SCOPE)
    retriever = HybridRetriever(store, embedder, PredicateRegistry(), max_episodes=3)

    found = retriever.search("kafka", EP_SCOPE, k=5, include_episodes=True)
    assert len(found) == 5
    kept = {r.claim.id for r in found if not isinstance(r, EpisodeResult)}
    assert len(kept) >= 2 and kept <= {c.id for c in claims}


def test_the_diversity_demotion_survives_the_merge(
    store: SQLiteStore, embedder: HashingEmbedder
) -> None:
    """`_rank` hands back a list that is deliberately *not* in score order, so merging
    by a plain re-sort would silently undo the per-slot cap."""
    dupes = [add(store, embedder, f"alice rated the standup format four of five{tail}",
                 EP_SCOPE, predicate="rated")
             for tail in ("", " today", " last week", " again")]
    other = add(store, embedder, "alice rated the retro format three of five",
                EP_SCOPE, predicate="also_rated")
    retriever = HybridRetriever(store, embedder, PredicateRegistry(), max_per_slot=2)

    with_eps = retriever.search("alice rated the format", EP_SCOPE, k=5,
                                include_episodes=True)
    without = retriever.search("alice rated the format", EP_SCOPE, k=5)

    assert [r.claim.id for r in with_eps] == [r.claim.id for r in without]
    assert other.id in {r.claim.id for r in with_eps[:3]}
    assert {c.id for c in dupes} & {r.claim.id for r in with_eps}


def test_a_memory_type_filter_suppresses_the_episode_leg(
    store: SQLiteStore, embedder: HashingEmbedder, retriever: HybridRetriever
) -> None:
    """A turn has no memory type. Returning one anyway would mean `memory_types=` had
    silently stopped being a filter."""
    turn(store, embedder, KAFKA, EP_SCOPE)
    claim = add(store, embedder, "the kafka pipeline is deprecated", EP_SCOPE,
                memory_type=MemoryType.PROCEDURAL)

    found = retriever.search("kafka pipeline", EP_SCOPE, k=5, include_episodes=True,
                             memory_types=[MemoryType.PROCEDURAL])
    assert [r.claim.id for r in found] == [claim.id]


def test_episode_search_abstains_on_a_query_with_no_signal(
    store: SQLiteStore, embedder: HashingEmbedder, retriever: HybridRetriever
) -> None:
    """Both guards apply to the episode legs too, and the stopword one matters more
    here: turns are long and conversational, so BM25 on `"do"` ranks every turn in the
    store — with enormous IDF, because at personal-memory scale stopwords are rare."""
    turn(store, embedder, KAFKA, EP_SCOPE)

    assert retriever.search("🙂", EP_SCOPE, k=5, include_episodes=True) == [], \
        "no alphanumeric content: both legs abstain"

    # Content-free but not signal-free. Cosine is defined for every pair, so the vector
    # leg still produces a ranking - the honest answer is the same low-relevance tail a
    # claim gets, not a confident BM25 hit on "do".
    stopwords = retriever.search("what do you know about me?", EP_SCOPE, k=5,
                                 include_episodes=True)
    assert [r.explain.lexical_rank for r in stopwords] == [None]
    assert stopwords[0].score < 0.01


def test_min_score_applies_to_the_discounted_episode_score(
    store: SQLiteStore, embedder: HashingEmbedder, retriever: HybridRetriever
) -> None:
    """The floor is on what the caller is handed, so it has to be applied after the
    weight rather than before it."""
    turn(store, embedder, KAFKA, EP_SCOPE)
    found = retriever.search("kafka pipeline", EP_SCOPE, k=5, include_episodes=True)
    got = found[0].score

    assert retriever.search("kafka pipeline", EP_SCOPE, k=5, include_episodes=True,
                            min_score=got * 0.5) != []
    assert retriever.search("kafka pipeline", EP_SCOPE, k=5, include_episodes=True,
                            min_score=got * 1.5) == []


def test_episode_time_travel_cannot_return_a_turn_from_the_future(
    store: SQLiteStore, embedder: HashingEmbedder, retriever: HybridRetriever
) -> None:
    old = turn(store, embedder, "kafka is holding up fine", EP_SCOPE, ts=T0)
    turn(store, embedder, KAFKA, EP_SCOPE, ts=T2)

    found = retriever.search("kafka", EP_SCOPE, k=5, include_episodes=True, as_of=T1)
    assert [r.episode.id for r in found] == [old.id]


def test_an_episode_explanation_reports_both_legs_and_neutral_quality(
    store: SQLiteStore, embedder: HashingEmbedder, retriever: HybridRetriever
) -> None:
    """The quality fields sit at 1.0 and mean "not applicable": decay is keyed to a
    predicate, confidence is an extractor's self-report, salience is earned by
    re-observation, and a turn has none of the three."""
    turn(store, embedder, KAFKA, EP_SCOPE)
    r = retriever.search("kafka pipeline", EP_SCOPE, k=5, include_episodes=True)[0]

    assert r.explain.vector_rank == 0 and r.explain.lexical_rank == 0
    assert r.explain.vector_score > 0.0 and r.explain.lexical_score > 0.0
    assert (r.explain.recency, r.explain.confidence, r.explain.salience) == (1.0,) * 3
    assert r.explain.raw_score == r.explain.fusion_score
    assert r.score == pytest.approx(r.explain.final_score)


def test_a_turn_matched_by_only_one_leg_still_surfaces(
    store: SQLiteStore, embedder: HashingEmbedder, retriever: HybridRetriever
) -> None:
    """Turns written before `reembed()` have text but no vector, which must degrade to
    a lexical-only answer rather than to nothing."""
    ep = turn(store, None, KAFKA, EP_SCOPE)

    r = retriever.search("kafka pipeline", EP_SCOPE, k=5, include_episodes=True)[0]
    assert r.episode.id == ep.id
    assert r.explain.vector_rank is None
    assert r.explain.lexical_rank == 0


def test_episode_hydration_falls_back_for_a_store_without_bulk_fetch(
    store: SQLiteStore, embedder: HashingEmbedder
) -> None:
    """`get_episodes` is an optimization on the protocol, not a requirement of it."""
    class OneAtATime(SQLiteStore):
        get_episodes = None

    thin = OneAtATime(":memory:")
    ep = turn(thin, embedder, KAFKA, EP_SCOPE)
    retriever = HybridRetriever(thin, embedder, PredicateRegistry())

    found = retriever.search("kafka pipeline", EP_SCOPE, k=5, include_episodes=True)
    assert [r.episode.id for r in found] == [ep.id]
    thin.close()


def test_a_turn_purged_mid_query_is_dropped_rather_than_crashing(
    store: SQLiteStore, embedder: HashingEmbedder
) -> None:
    """A row that vanishes between ranking and hydration is a race, not a ranking
    error."""
    class Vanishing(SQLiteStore):
        def get_episodes(self, episode_ids):
            return {}

    vanishing = Vanishing(":memory:")
    turn(vanishing, embedder, KAFKA, EP_SCOPE)
    retriever = HybridRetriever(vanishing, embedder, PredicateRegistry())

    assert retriever.search("kafka", EP_SCOPE, k=5, include_episodes=True) == []
    vanishing.close()


def test_a_store_without_episode_search_degrades_instead_of_raising(
    embedder: HashingEmbedder
) -> None:
    """A third-party `Store` written against the old protocol. Losing the vector leg
    should narrow the answer, not remove it; losing both should return nothing rather
    than an AttributeError halfway through a query."""
    class LexicalOnly(SQLiteStore):
        vector_search_episodes = None

    class NoEpisodeSearch(SQLiteStore):
        vector_search_episodes = None
        lexical_search_episodes = None

    lexical = LexicalOnly(":memory:")
    ep = turn(lexical, embedder, KAFKA, EP_SCOPE)
    found = HybridRetriever(lexical, embedder, PredicateRegistry()).search(
        "kafka pipeline", EP_SCOPE, k=5, include_episodes=True)
    assert [r.episode.id for r in found] == [ep.id]
    assert found[0].explain.vector_rank is None
    lexical.close()

    blind = NoEpisodeSearch(":memory:")
    turn(blind, embedder, KAFKA, EP_SCOPE)
    assert HybridRetriever(blind, embedder, PredicateRegistry()).search(
        "kafka", EP_SCOPE, k=5, include_episodes=True) == []
    blind.close()


# --- Episode scope isolation, all three directions --------------------------

@pytest.fixture
def scoped_turns(store: SQLiteStore, embedder: HashingEmbedder) -> dict[str, Episode]:
    """Byte-identical text at every scope. If the filter were on content rather than on
    scope, identical text is what would slip through."""
    return {
        "session": turn(store, embedder, KAFKA, EP_SCOPE),
        "user": turn(store, embedder, KAFKA, Scope("acme", "alice")),
        "sibling_session": turn(store, embedder, KAFKA,
                                Scope("acme", "alice", "agent_a", "sess_2")),
        "sibling_agent": turn(store, embedder, KAFKA,
                              Scope("acme", "alice", "agent_b")),
        "other_user": turn(store, embedder, KAFKA, Scope("acme", "bob")),
        "other_tenant": turn(store, embedder, KAFKA, Scope("globex")),
    }


def test_a_session_inherits_its_users_turns(
    retriever: HybridRetriever, scoped_turns: dict[str, Episode]
) -> None:
    found = {r.episode.id for r in
             retriever.search("kafka", EP_SCOPE, k=10, include_episodes=True)}
    assert scoped_turns["user"].id in found
    assert scoped_turns["session"].id in found


@pytest.mark.parametrize(
    "neighbour", ["sibling_session", "sibling_agent", "other_user", "other_tenant"])
def test_no_query_reaches_sideways_for_turns_either(
    retriever: HybridRetriever, scoped_turns: dict[str, Episode], neighbour: str
) -> None:
    found = {r.episode.id for r in
             retriever.search("kafka", EP_SCOPE, k=10, include_episodes=True)}
    assert scoped_turns[neighbour].id not in found


def test_cross_tenant_turn_leakage_is_impossible(
    retriever: HybridRetriever, scoped_turns: dict[str, Episode]
) -> None:
    """The worst bug this system could have, asserted from both directions — and raw
    transcript is the more sensitive of the two payloads, not the less."""
    from_acme = {r.episode.id for r in
                 retriever.search("kafka", EP_SCOPE, k=10, include_episodes=True)}
    from_globex = {r.episode.id for r in
                   retriever.search("kafka", Scope("globex"), k=10,
                                    include_episodes=True)}

    assert from_globex == {scoped_turns["other_tenant"].id}
    assert from_acme.isdisjoint(from_globex)
    assert retriever.search("kafka", Scope("initech"), k=10,
                            include_episodes=True) == []


def test_a_zero_budget_turns_the_episode_leg_off_entirely(
    store: SQLiteStore, embedder: HashingEmbedder
) -> None:
    """The escape hatch for a caller who wants the flag wired but the tail suppressed —
    and the proof that the budget, not the flag, is what bounds the cost to the prompt."""
    turn(store, embedder, KAFKA, EP_SCOPE)
    off = HybridRetriever(store, embedder, PredicateRegistry(), max_episodes=0)
    assert off.search("kafka pipeline", EP_SCOPE, k=5, include_episodes=True) == []


def test_a_weak_turn_never_costs_a_claim_its_slot(
    store: SQLiteStore, embedder: HashingEmbedder
) -> None:
    """`k` stays the total, so the guarantee has to come from the merge: an episode
    takes a slot only from a claim it outscores after the discount."""
    claims = [add(store, embedder, f"alice rated kafka {i} of five", EP_SCOPE,
                  predicate=f"rated_{i}") for i in range(5)]
    turn(store, embedder, "we had lunch and discussed nothing of consequence", EP_SCOPE)
    retriever = HybridRetriever(store, embedder, PredicateRegistry(), max_per_slot=99)

    found = retriever.search("alice rated kafka", EP_SCOPE, k=5, include_episodes=True)
    assert {r.claim.id for r in found} == {c.id for c in claims}


# ===========================================================================
# Telemetry: the aggregate view an Explanation cannot give
# ===========================================================================
#
# `Explanation` already answers "why did this one result surface?" precisely. What it
# cannot answer is whether the ranking is drifting, because both of the silent
# retrieval failures are properties of a *distribution* across many searches: quality
# signals quietly outranking evidence, and reinforcement failing to reach the ranking
# at all.


def test_a_search_reports_volume_shape_and_latency(
    store: SQLiteStore, embedder: HashingEmbedder
) -> None:
    rec = MemoryRecorder()
    scope = Scope("acme", "alice")
    add(store, embedder, "alice works at acme", scope, predicate="works_at")
    reader = HybridRetriever(store, embedder, PredicateRegistry(), telemetry=rec)

    reader.search("where does alice work", scope)
    assert rec.total(RETRIEVAL_QUERY, script="latin") == 1
    assert rec.values(RETRIEVAL_RESULTS) == [1.0]
    assert len(rec.values(RETRIEVAL_LATENCY_MS)) == 1


def test_query_volume_is_sliced_by_script_so_it_pairs_with_the_gate(
    store: SQLiteStore, embedder: HashingEmbedder
) -> None:
    """A script with query volume and no `gate.pass` is a population whose writes are
    being dropped, and whose reads therefore find nothing. Neither half says that
    alone."""
    rec = MemoryRecorder()
    scope = Scope("acme", "alice")
    reader = HybridRetriever(store, embedder, PredicateRegistry(), telemetry=rec)
    reader.search("where does alice live", scope)
    reader.search("我住在哪里", scope)
    assert rec.total(RETRIEVAL_QUERY, script="latin") == 1
    assert rec.total(RETRIEVAL_QUERY, script="han") == 1


def test_an_empty_result_set_is_reported_rather_than_omitted(
    store: SQLiteStore, embedder: HashingEmbedder
) -> None:
    """Zero is the interesting value: it is `min_score` working, or the corpus not
    answering, and a series that simply goes quiet cannot show either."""
    rec = MemoryRecorder()
    scope = Scope("acme", "alice")
    add(store, embedder, "alice works at acme", scope, predicate="works_at")
    reader = HybridRetriever(store, embedder, PredicateRegistry(), telemetry=rec)
    reader.search("where does alice work", scope, min_score=0.99)
    assert rec.values(RETRIEVAL_RESULTS) == [0.0]
    assert rec.values(RETRIEVAL_QUALITY_FACTOR) == []


def test_the_quality_factor_is_bounded_by_the_normalization_it_reports_on(
    store: SQLiteStore, embedder: HashingEmbedder
) -> None:
    """Salience overriding relevance was the failure. The design intent is that quality
    can only pull a result *down* from its evidence, by at most `1/span` — and the one
    way past 1.0 is a salience reinforced beyond 1.0, which `quality_boost` deliberately
    does not clamp. So above 1.0 is the alarm, and the series has to be able to report
    it: the claim below is pinned at the top of the range reported from a production
    store, and its factor must come back greater than one rather than clipped to it."""
    rec = MemoryRecorder()
    scope = Scope("acme", "alice")
    now = datetime.now(timezone.utc)
    add(store, embedder, "alice works at acme", scope, predicate="works_at",
        valid_from=now, salience=2.6)
    add(store, embedder, "alice worked at initech", scope, predicate="worked_at",
        valid_from=now - timedelta(days=4000), salience=0.05, confidence=0.2)
    reader = HybridRetriever(store, embedder, PredicateRegistry(), telemetry=rec)
    reader.search("where does alice work", scope, k=5)

    span = 1.0 + reader.w_recency + reader.w_confidence + reader.w_salience
    factors = sorted(rec.values(RETRIEVAL_QUALITY_FACTOR))
    assert len(factors) == 2
    assert all(f >= 1.0 / span for f in factors)
    assert factors[0] < 1.0 < factors[1], (
        "the over-reinforced claim's factor was clipped, which is the one value worth "
        "seeing")
    # Fresh, confident and heavily reinforced against stale, unconfident and faded:
    # the spread is what makes the distribution worth plotting.
    assert factors[1] - factors[0] > 0.1


def test_an_episode_is_left_out_of_the_claim_distributions(
    store: SQLiteStore, embedder: HashingEmbedder
) -> None:
    """An episode's quality fields sit at their neutral 1.0 meaning "not applicable",
    so counting one would report a perfect quality factor for something quality never
    scored, and an observation count for something nothing observed."""
    rec = MemoryRecorder()
    turn(store, embedder, KAFKA, EP_SCOPE)
    add(store, embedder, "alice owns the kafka pipeline", EP_SCOPE, predicate="owns")
    reader = HybridRetriever(store, embedder, PredicateRegistry(), telemetry=rec)

    found = reader.search("kafka pipeline", EP_SCOPE, k=5, include_episodes=True)
    assert any(isinstance(r, EpisodeResult) for r in found)
    assert rec.values(RETRIEVAL_RESULTS) == [float(len(found))]
    assert len(rec.values(RETRIEVAL_QUALITY_FACTOR)) == 1


def test_the_observation_rank_correlation_is_emitted_with_the_sign_that_matters(
    store: SQLiteStore, embedder: HashingEmbedder
) -> None:
    """Positive is correct: a fact restated many times should rank above one mentioned
    once. It went negative when reinforcement was written onto the decayed `salience`
    rather than the storage base and the nightly pass then erased it — no exception, no
    log line, and no way to see it except as a trend across searches."""
    rec = MemoryRecorder()
    scope = Scope("acme", "alice")
    now = datetime.now(timezone.utc)
    add(store, embedder, "alice commutes to the office by bicycle", scope,
        predicate="commutes_by", valid_from=now, observation_count=12, salience=2.0)
    add(store, embedder, "alice mentioned a scooter", scope,
        predicate="mentioned", valid_from=now, observation_count=1)
    reader = HybridRetriever(store, embedder, PredicateRegistry(), telemetry=rec)

    reader.search("how does alice commute by bicycle", scope, k=5)
    values = rec.values(RETRIEVAL_OBSERVATION_RANK_CORR)
    assert len(values) == 1 and values[0] > 0.0


def test_no_correlation_is_reported_when_there_is_nothing_to_correlate(
    store: SQLiteStore, embedder: HashingEmbedder
) -> None:
    """Every claim seen once is an absence of evidence, not a correlation of zero, and
    reporting 0.0 would drag the average toward "broken" on every young store."""
    rec = MemoryRecorder()
    scope = Scope("acme", "alice")
    add(store, embedder, "alice works at acme", scope, predicate="works_at")
    add(store, embedder, "alice lives in berlin", scope, predicate="lives_in")
    reader = HybridRetriever(store, embedder, PredicateRegistry(), telemetry=rec)
    reader.search("alice", scope, k=5)
    assert rec.values(RETRIEVAL_OBSERVATION_RANK_CORR) == []


def test_a_degenerate_search_emits_nothing_because_it_ran_nothing(
    store: SQLiteStore, embedder: HashingEmbedder
) -> None:
    rec = MemoryRecorder()
    reader = HybridRetriever(store, embedder, PredicateRegistry(), telemetry=rec)
    assert reader.search("anything", Scope("acme", "alice"), k=0) == []
    assert rec.names() == []


def test_ties_break_on_content_so_two_ingests_of_one_corpus_rank_alike():
    """The module docstring promises identical inputs give an identical ordering. It was
    false across ingests: ties broke on `claim.id`, which is a fresh `uuid4` every time,
    so ordering was stable within a store and a coin flip between two stores holding the
    same data. That is precisely the comparison a benchmark, a regression test and a
    bisect all make — measured on LOCOMO, repeated runs disagreed by up to 0.07 points
    with nothing else changed.
    """
    from memvara import Memvara, HashingEmbedder, NullLLM

    def ingest_and_rank():
        mem = Memvara(embedder=HashingEmbedder(dim=64), llm=NullLLM(), user="alice")
        for i in range(20):
            # One scoring text across every row, so BM25 and cosine tie exactly and the
            # tiebreak is the only thing deciding the order.
            mem.remember("user", f"pred_{i}", f"value_{i}", text="alpha beta gamma")
        out = [r.claim.value_key for r in mem.search("alpha beta", k=20)]
        mem.close()
        return out

    orderings = {tuple(ingest_and_rank()) for _ in range(5)}
    assert len(orderings) == 1, "five ingests of one corpus produced different rankings"


def test_the_tiebreak_key_is_not_derived_from_the_row_id():
    """Stated structurally as well, because the test above only fails when a tie happens
    to occur — and a future scoring change could hide the defect by making exact ties
    rare rather than by fixing them."""
    import inspect

    from memvara.retrieve.hybrid import HybridRetriever

    for method in (HybridRetriever._rank, HybridRetriever._episodes):
        src = inspect.getsource(method)
        sort_lines = [l for l in src.splitlines() if ".sort(key=" in l]
        assert sort_lines, f"{method.__name__} no longer sorts; update this test"
        for line in sort_lines:
            assert "value_key" in line or "hash" in line, (
                f"{method.__name__} breaks ties without a content-derived key: {line.strip()}")


# --- the store-level graph gate ---------------------------------------------------

def _joined_store(tmp_path, star: bool):
    """A store that chains, or one that does not, and a retriever with the leg on."""
    from memvara import Memvara
    from memvara.llm import NullLLM
    mem = Memvara(":memory:", llm=NullLLM(), embedder=HashingEmbedder(dim=64))
    mem.remember("user", "uses", "pytest")
    if star:
        mem.remember("user", "lives_in", "Delhi")
    else:
        mem.remember("pytest", "configured_in", "pyproject.toml")
    return mem


RELATIONAL = "who is the manager of the person who uses pytest"


def test_a_store_where_nothing_chains_does_not_run_the_graph_leg(tmp_path):
    """A walk needs somewhere to go. Where no claim's object is another claim's subject
    there is nowhere, and the leg degenerates into returning other facts about the hub —
    ranked by a path score that is near-uniform when every path is one hop. Fusion reads
    positions, so that is a fabricated ranking, and on LongMemEval it cost 1.6 points.
    """
    mem = _joined_store(tmp_path, star=True)
    r = HybridRetriever(mem.store, mem.embedder, mem.registry, w_graph=1.0,
                        traverser=mem.traverser)
    with pytest.warns(UnjoinedStoreWarning, match="nothing in this store chains"):
        r.search(RELATIONAL, mem.default_scope, k=5)
    assert r._joins["default"] == (0, False)
    mem.close()


def test_a_store_that_chains_is_left_alone(tmp_path):
    mem = _joined_store(tmp_path, star=False)
    r = HybridRetriever(mem.store, mem.embedder, mem.registry, w_graph=1.0,
                        traverser=mem.traverser)
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # any warning here is a failure
        r.search(RELATIONAL, mem.default_scope, k=5)
    assert r._joins["default"] == (0, True)
    mem.close()


def test_the_gate_warns_once_per_retriever_and_not_once_per_search(tmp_path):
    """Same reason its parent class does: a store's shape does not change per query, and
    a warning per search buries the finding under itself.

    Counted by category rather than by asserting the whole caught list. `simplefilter
    ("always")` catches everything raised inside the block, including `ResourceWarning`
    from objects an earlier test left for the collector — which is a property of when the
    garbage collector runs, not of this retriever. On Windows it does so reliably: seven of
    them, on every run, failing a test about a gate that had behaved correctly. The
    assertion below says what the name says.
    """
    mem = _joined_store(tmp_path, star=True)
    r = HybridRetriever(mem.store, mem.embedder, mem.registry, w_graph=1.0,
                        traverser=mem.traverser)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            r.search(RELATIONAL, mem.default_scope, k=5)
    gate = [w for w in caught if issubclass(w.category, UnjoinedStoreWarning)]
    assert len(gate) == 1, "three searches, one warning"
    mem.close()


def test_the_reading_is_cached_and_retaken_on_a_counter_not_a_clock(tmp_path):
    """Deterministic by construction: the same sequence of searches re-measures at the
    same points on every run, which a wall-clock TTL could not promise."""
    mem = _joined_store(tmp_path, star=False)
    r = HybridRetriever(mem.store, mem.embedder, mem.registry, w_graph=1.0,
                        traverser=mem.traverser)
    calls = []
    real = mem.store.connectivity
    mem.store.connectivity = lambda t=None: (calls.append(t), real(t))[1]  # type: ignore
    for _ in range(hybrid_mod.GATE_RECHECK_EVERY + 2):
        r.search(RELATIONAL, mem.default_scope, k=5)
    assert len(calls) == 2, "measured once, then once more when the counter came round"
    mem.close()


def test_a_backend_that_cannot_measure_is_not_read_as_a_store_with_no_joins(tmp_path):
    """`{}` means it did not look. Reading that as zero would switch a working graph leg
    off on every third-party store at once, on a measurement nobody took.
    """
    mem = _joined_store(tmp_path, star=True)
    r = HybridRetriever(mem.store, mem.embedder, mem.registry, w_graph=1.0,
                        traverser=mem.traverser)
    mem.store.connectivity = lambda t=None: {}          # type: ignore
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r.search(RELATIONAL, mem.default_scope, k=5)
    assert r._joins["default"] == (0, True)
    mem.close()


def test_a_store_without_connectivity_at_all_keeps_its_graph_leg(tmp_path):
    mem = _joined_store(tmp_path, star=True)
    r = HybridRetriever(mem.store, mem.embedder, mem.registry, w_graph=1.0,
                        traverser=mem.traverser)
    mem.store.connectivity = None                       # type: ignore[assignment]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r.search(RELATIONAL, mem.default_scope, k=5)
    assert r._joins["default"] == (0, True)
    mem.close()


def test_the_shipped_default_never_asks_the_store_about_joins(tmp_path):
    """`w_graph=0.0` is the shipped configuration and must pay nothing for a gate on a
    leg it is not running."""
    mem = _joined_store(tmp_path, star=True)
    # `w_graph` defaults to 0, and a real traverser is supplied so the only reason the
    # store is not consulted is the gate declining to ask.
    r = HybridRetriever(mem.store, mem.embedder, mem.registry,
                        traverser=mem.traverser)
    calls = []
    mem.store.connectivity = lambda t=None: calls.append(t) or {}   # type: ignore
    r.search(RELATIONAL, mem.default_scope, k=5)
    assert calls == []
    assert r._joins == {}
    mem.close()


def test_each_tenant_is_measured_on_its_own(tmp_path):
    """One tenant's shape says nothing about another's, and a shared verdict would leak
    the fact that a neighbour's store is a star."""
    from memvara import Memvara
    from memvara.llm import NullLLM
    mem = Memvara(":memory:", llm=NullLLM(), embedder=HashingEmbedder(dim=64))
    mem.remember("user", "uses", "pytest", tenant="star")
    mem.remember("user", "lives_in", "Delhi", tenant="star")
    mem.remember("user", "uses", "pytest", tenant="web")
    mem.remember("pytest", "configured_in", "pyproject.toml", tenant="web")
    r = HybridRetriever(mem.store, mem.embedder, mem.registry, w_graph=1.0,
                        traverser=mem.traverser)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r.search(RELATIONAL, Scope(tenant="star"), k=5)
        r.search(RELATIONAL, Scope(tenant="web"), k=5)
    assert r._joins == {"star": (0, False), "web": (0, True)}
    mem.close()


def test_a_store_that_chains_but_seeds_nothing_still_runs_no_walk(tmp_path):
    """`graph_seeds=0` is the documented spelling for "seed nothing", and it has to be
    reachable *past* the store-level gate.

    The gate opens here — this store chains — so the leg is entered and gives up on the
    seed list instead, which is a different early return with a different meaning. Pinned
    because the gate took over the path that used to cover it: before it, a star store
    reached the seed check and returned empty from there, and that read as the same line
    being exercised when the two exits say quite different things. One is "there is
    nowhere to walk", the other is "there is nowhere to start".
    """
    mem = _joined_store(tmp_path, star=False)
    # `intent_weighting=False` so the classifier cannot close the leg before the seed
    # check is reached; this test is about the seed check and nothing else.
    r = HybridRetriever(mem.store, mem.embedder, mem.registry, w_graph=1.0,
                        traverser=mem.traverser, graph_seeds=0,
                        intent_weighting=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # the gate must not fire on this store
        rows = r.search(RELATIONAL, mem.default_scope, k=5)
    assert r._joins["default"] == (0, True)
    assert all(x.explain.graph_rank is None for x in rows), (
        "no seeds means no walk, so nothing may carry a graph rank"
    )
    mem.close()


# --- the read path's clock -----------------------------------------------------------

def test_two_searches_at_a_pinned_instant_score_identically(tmp_path):
    """The read path decays from the moment it is asked, so two identical searches
    seconds apart score every claim differently — measured on 2WikiMultihopQA at 3,000 of
    3,000 questions, in the low-order digits, which is enough to flip a near-tie at the
    `k` boundary and move a published figure with no code change behind it.

    `now` is the parameter `Consolidator.run()` already has, for the same reason: the
    write path had this defect and it was fixed there first.
    """
    from datetime import datetime, timezone
    mem = _joined_store(tmp_path, star=False)
    r = HybridRetriever(mem.store, mem.embedder, mem.registry, w_graph=1.0,
                        traverser=mem.traverser)
    at = datetime(2030, 6, 1, tzinfo=timezone.utc)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = r.search(RELATIONAL, mem.default_scope, k=5, now=at)
        b = r.search(RELATIONAL, mem.default_scope, k=5, now=at)
    assert [(x.claim.id, x.score) for x in a] == [(y.claim.id, y.score) for y in b]
    mem.close()


def test_known_at_still_wins_over_a_pinned_now(tmp_path):
    """`now` replaces the clock *read*, never the belief axis. A time-travel query keeps
    decaying from the instant it asked about."""
    from datetime import datetime, timezone
    mem = _joined_store(tmp_path, star=False)
    r = HybridRetriever(mem.store, mem.embedder, mem.registry)
    known = datetime(2030, 1, 1, tzinfo=timezone.utc)
    far = datetime(2040, 1, 1, tzinfo=timezone.utc)
    both = r.search("pytest", mem.default_scope, k=5, known_at=known, now=far)
    just_known = r.search("pytest", mem.default_scope, k=5, known_at=known)
    assert [(x.claim.id, x.score) for x in both] == \
           [(y.claim.id, y.score) for y in just_known], (
        "a pinned now must not shift a query that named its own instant"
    )
    mem.close()
