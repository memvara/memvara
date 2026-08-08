"""End-to-end hybrid retrieval: scope isolation, time travel, and the two failure
modes a vector-only store cannot fix.

Everything runs offline: `SQLiteStore(":memory:")` plus `HashingEmbedder`, claims
constructed directly and inserted through the store. Time is always passed in
explicitly - nothing here sleeps or patches a clock.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from engram.embed import HashingEmbedder
from engram.retrieve import (
    STOPWORDS,
    HybridRetriever,
    analyze,
    calibrate_min_score,
    lexical_relevance,
    relevance,
    tokenize,
    vector_relevance,
)
from engram.schema import PredicateRegistry
from engram.store import SQLiteStore
from engram.types import Claim, MemoryType, Scope

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
    from engram.store.sqlite import _fts_query

    for query in ("ERR_7734 in prod", "what's my mother's maiden name?",
                  "PLAT-2291", "東京 に 住んでいる", "a b cd", "!!! ???"):
        expected = [t.strip('"') for t in _fts_query(query).split(" OR ") if t]
        assert tokenize(query) == expected, query


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
                              filter_retry_multiplier=1)
    query = "release detail"

    # Without the retry the pool of 10 is spent entirely on semantic claims, and a
    # filter with three live matches behind it returns nothing at all.
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
        def lexical_search(self, query, scopes, limit, as_of=None,
                           include_invalidated=False):
            calls.append(limit)
            return super().lexical_search(query, scopes, limit, as_of,
                                          include_invalidated)

    counting = CountingStore(":memory:")
    add(counting, embedder, "alice lives in Lisbon", scope)
    retriever = HybridRetriever(counting, embedder, PredicateRegistry())

    assert retriever.search("lisbon", scope, k=5,
                            memory_types=[MemoryType.PROCEDURAL]) == []
    assert calls == [25], "one pass only: the pool was never truncated"
    counting.close()


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
