"""Hybrid retrieval: BM25 + vectors, fused, rescored, and explained.

What this replaces: mem0-style retrieval is a single vector top-k. Two failures fall
out of that, and both are routine rather than exotic.

1. **Exact tokens.** Embeddings are trained to map surface forms onto meaning, which
   is exactly the wrong behaviour for `ERR_7734`, `v2.14.1`, a ticket id or an unusual
   surname. A subword tokenizer shreds them and cosine similarity puts the claim that
   literally contains the token below a dozen claims that merely talk about errors.
   BM25 has the opposite bias - a rare term carries enormous IDF - so the two
   retrievers fail on disjoint inputs, which is the precondition for fusion to help.
2. **Time.** Cosine similarity has no opinion about whether a fact is current. A 2023
   employer that was superseded in 2026 scores identically to the one that replaced
   it. Rescoring by predicate-keyed decay fixes the ordering without deleting history.

Everything here is deterministic. No LLM sits on the read path, and identical inputs
produce an identical ordering, ties included - unstable ranking makes retrieval
regressions impossible to bisect.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

import numpy as np

from ..embed.base import Embedder
from ..schema import PredicateRegistry
from ..store.base import Store
from ..types import Claim, Explanation, MemoryType, Result, Scope, utcnow
from .fusion import reciprocal_rank_fusion
from .scoring import final_score, recency_factor

# Retriever names. Shared between the fusion weights and the `Explanation` fields so
# the two cannot drift apart under a rename.
VECTOR = "vector"
LEXICAL = "lexical"


def _as_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


class HybridRetriever:
    """Scope-aware, time-travelling hybrid search over the claim store."""

    def __init__(
        self,
        store: Store,
        embedder: Embedder,
        registry: PredicateRegistry,
        *,
        w_vector: float = 1.0,
        w_lexical: float = 1.0,
        rrf_k: int = 60,
        w_recency: float = 0.25,
        w_confidence: float = 0.15,
        w_salience: float = 0.10,
        candidate_multiplier: int = 5,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.registry = registry
        self.w_vector = w_vector
        self.w_lexical = w_lexical
        self.rrf_k = rrf_k
        self.w_recency = w_recency
        self.w_confidence = w_confidence
        self.w_salience = w_salience
        self.candidate_multiplier = candidate_multiplier

    def search(
        self,
        query: str,
        scope: Scope,
        *,
        k: int = 10,
        as_of: datetime | None = None,
        include_invalidated: bool = False,
        memory_types: Sequence[MemoryType] | None = None,
    ) -> list[Result]:
        """Return the top `k` claims for `query`, each with a populated `Explanation`.

        `as_of` is transaction-time travel: the result is what we believed at that
        instant, including claims we have since retracted. `include_invalidated`
        additionally lifts the liveness filter, surfacing claims that were already
        dead at `as_of` - useful for auditing, wrong for answering a question.
        """
        if k <= 0:
            return []

        # A narrow scope inherits everything above it, so a session-level question can
        # still answer from what the user told us months ago in another session. The
        # sibling direction never opens: `ancestors()` walks strictly upward, and the
        # store matches each scope tuple exactly, so no query can reach sideways into
        # a peer session, a peer user, or - the one that would actually matter -
        # another tenant.
        scopes = scope.ancestors()

        # Decay is measured at the instant being asked about, not at wall-clock now.
        now = _as_utc(as_of) if as_of is not None else utcnow()

        # Over-fetch per retriever: fusion can only rank what it was given, and a claim
        # that BM25 puts first is worthless if the vector list was cut before it and
        # the final k is small.
        limit = max(k * self.candidate_multiplier, k)

        vector_hits = self._vector_search(query, scopes, limit, as_of, include_invalidated)
        lexical_hits = list(
            self.store.lexical_search(query, scopes, limit, as_of, include_invalidated)
        )

        fused = reciprocal_rank_fusion(
            {VECTOR: vector_hits, LEXICAL: lexical_hits},
            k=self.rrf_k,
            weights={VECTOR: self.w_vector, LEXICAL: self.w_lexical},
        )
        if not fused:
            return []

        vector_pos = _positions(vector_hits)
        lexical_pos = _positions(lexical_hits)
        wanted = set(memory_types) if memory_types is not None else None

        # These filters run after fusion, so a rejected candidate still consumed a slot
        # in the pool the retrievers returned. That costs some recall when a filter is
        # narrow, and the alternative - pushing memory_type into both retrievers - would
        # widen the `Store` protocol for a case that is rare in practice. Raise
        # `candidate_multiplier` if you filter hard.
        results: list[Result] = []
        for claim_id, fusion in fused.items():
            claim = self.store.get_claim(claim_id)
            if claim is None:
                continue  # raced with a delete; a missing row is not a ranking error
            if wanted is not None and claim.memory_type not in wanted:
                continue
            if not self._believed_by(claim, as_of):
                continue

            results.append(self._explain(claim, fusion, vector_pos, lexical_pos, now))

        # Sort on the claim id as a secondary key. Fusion ties are common - two claims
        # each ranked first by one retriever score identically by construction - and
        # dict order alone would let the answer depend on insertion history.
        results.sort(key=lambda r: (-r.score, r.claim.id))
        return results[:k]

    # -- internals -----------------------------------------------------------

    def _vector_search(
        self,
        query: str,
        scopes: Sequence[Scope],
        limit: int,
        as_of: datetime | None,
        include_invalidated: bool,
    ) -> list[tuple[str, float]]:
        """Vector leg, skipped when the query embeds to nothing.

        A zero vector gives every candidate a cosine of exactly 0.0, so the store's
        "ranking" degenerates to whatever order the index happened to enumerate. Fusion
        would then read those positions as evidence. Two real inputs hit this: queries
        with no alphanumeric content at all (`"*"`, pure punctuation, whitespace), and
        - with the offline `HashingEmbedder`, whose word regex is ASCII-only - any
        purely CJK query. Returning nothing lets BM25 answer alone, which for the CJK
        case it does correctly, instead of burying it under fabricated ranks.
        """
        qvec = np.asarray(self.embedder.encode([query])[0], dtype=np.float32)
        if float(np.linalg.norm(qvec)) <= 0.0:
            return []
        return list(self.store.vector_search(qvec, scopes, limit, as_of, include_invalidated))

    @staticmethod
    def _believed_by(claim: Claim, as_of: datetime | None) -> bool:
        """Transaction-time floor, enforced here rather than left to the store.

        `include_invalidated=True` drops the store's entire liveness predicate, which
        also drops `recorded_at <= as_of` and lets claims recorded *after* the asked
        instant leak into a historical answer. "What did we believe in March,
        including what we later retracted" must never include something we had not yet
        heard in March - that is knowledge from the future, and it is the one way a
        bitemporal query can lie. Re-applying the floor costs a comparison per
        candidate and makes the two flags orthogonal, which is what callers assume.
        """
        if as_of is None:
            return True
        return _as_utc(claim.recorded_at) <= _as_utc(as_of)

    def _explain(
        self,
        claim: Claim,
        fusion: float,
        vector_pos: dict[str, tuple[int, float]],
        lexical_pos: dict[str, tuple[int, float]],
        now: datetime,
    ) -> Result:
        v = vector_pos.get(claim.id)
        lx = lexical_pos.get(claim.id)
        recency = recency_factor(claim, self.registry, now)
        score = final_score(
            fusion,
            recency=recency,
            confidence=claim.confidence,
            salience=claim.salience,
            w_recency=self.w_recency,
            w_confidence=self.w_confidence,
            w_salience=self.w_salience,
        )
        explain = Explanation(
            # `None` here is a finding, not a gap: it says this claim surfaced on one
            # retriever's evidence alone, which is exactly the signal you want when
            # debugging why something did or did not come back.
            vector_rank=None if v is None else v[0],
            vector_score=None if v is None else v[1],
            lexical_rank=None if lx is None else lx[0],
            lexical_score=None if lx is None else lx[1],
            fusion_score=fusion,
            recency=recency,
            confidence=claim.confidence,
            salience=claim.salience,
            # No cross-encoder in this tier. Left None so a future reranker's absence
            # is distinguishable from a reranker that scored zero.
            rerank_score=None,
            final_score=score,
        )
        return Result(claim=claim, score=score, explain=explain)


def _positions(hits: Sequence[tuple[str, float]]) -> dict[str, tuple[int, float]]:
    """Map item id -> (0-based rank, raw score), keeping the best rank on repeats."""
    out: dict[str, tuple[int, float]] = {}
    for rank, (item_id, score) in enumerate(hits):
        out.setdefault(item_id, (rank, score))
    return out
