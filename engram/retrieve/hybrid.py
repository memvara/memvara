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
3. **Absence.** Top-k always returns k things. Asked "what is the capital of France?"
   a memory store will hand back the user's city with the same confidence it hands
   back their name, because rank 0 is rank 0 whether or not anything in the corpus
   answers the question. Scoring on absolute retriever evidence rather than on fused
   rank (see `scoring.normalized_score`) makes "nothing here is relevant" a number a
   caller can act on, and `min_score` is how they act on it.
4. **Everything that was never a claim.** Extraction keeps facts and discards wording,
   and most of a real transcript is not a fact: a decision and the reason behind it, a
   constraint stated conditionally, an argument that was settled. Those turns were
   stored and then unreachable, because the only indexes were over claims -
   `WriteReceipt.skipped`, the number the write path is proudest of, meant "we kept
   this and will never find it again". `include_episodes=True` adds a second, weaker
   pair of legs over the raw turns; see `EpisodeResult` for why weaker.

Everything here is deterministic. No LLM sits on the read path, and identical inputs
produce an identical ordering, ties included - unstable ranking makes retrieval
regressions impossible to bisect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import ClassVar, Sequence

import numpy as np

from ..embed.base import Embedder
from ..schema import PredicateRegistry
from ..store.base import Store
from ..types import (
    CLAIM,
    EPISODE,
    Claim,
    Episode,
    Explanation,
    MemoryType,
    Result,
    Scope,
    utcnow,
)
from .analyze import analyze
from .fusion import reciprocal_rank_fusion
from .scoring import (
    final_score,
    lexical_relevance,
    normalized_score,
    recency_factor,
    relevance,
    vector_relevance,
)

# Retriever names. Shared between the fusion weights and the `Explanation` fields so
# the two cannot drift apart under a rename.
VECTOR = "vector"
LEXICAL = "lexical"


# There is deliberately no default relevance floor here. One was measured and shipped
# (0.25, calibrated on a 36-claim corpus) and it is wrong at both ends: the window
# between the weakest correct answer and the best wrong one moves with corpus size, and
# the windows at 5 claims and at 1,000 do not intersect. See `calibrate.py` for the
# measurement and for `calibrate_min_score`, which derives the number from a
# deployment's own probes instead of guessing it here.


def _as_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


@dataclass(slots=True)
class EpisodeResult:
    """A raw conversation turn that matched, and how well.

    Deliberately *not* a `Result`. The two look alike — a score, an `Explanation`, a
    `.text` — and they are not interchangeable: a `Claim` has been extracted,
    normalized, reconciled against what else is believed, and retired if something
    superseded it, while an episode is a verbatim thing someone said once. Rendering
    the second as though it were the first is how "I'm thinking of moving to Lisbon"
    becomes a stored fact about where the user lives, so the type system is where the
    distinction belongs and `isinstance` is the discriminator. `kind` carries the same
    answer for callers that serialize.

    The quality fields on `explain` sit at their neutral 1.0 and mean "not applicable"
    rather than "perfect". Recency decay is keyed to a predicate's half-life and an
    episode has no predicate; confidence is an extractor's self-report and nothing
    extracted this; salience is earned by re-observation, which is a claim's
    mechanism. `score` is therefore retriever evidence alone, times `w_episode`.
    """

    episode: Episode
    score: float
    explain: Explanation = field(default_factory=Explanation)

    kind: ClassVar[str] = EPISODE

    @property
    def text(self) -> str:
        return self.episode.content

    def __repr__(self) -> str:
        legs = []
        if self.explain.vector_rank is not None:
            legs.append(f"vector#{self.explain.vector_rank}")
        if self.explain.lexical_rank is not None:
            legs.append(f"bm25#{self.explain.lexical_rank}")
        return (f"<EpisodeResult {self.score:.4f} {_short(self.text)!r} "
                f"{'+'.join(legs) or 'no-retriever'} {self.episode.id}>")


#: Anything `search()` can return.
Retrieved = Result | EpisodeResult


def kind_of(item: Retrieved) -> str:
    """`"claim"` or `"episode"`, for callers that would rather branch on a string.

    `isinstance(item, EpisodeResult)` says the same thing and is the better spelling in
    Python; this exists because the distinction has to survive serialization into a
    prompt, a JSON payload or an MCP tool result, where the class does not.
    """
    return item.kind


def _short(text: str, limit: int = 48) -> str:
    """One-line, length-capped rendering, for reprs over arbitrary turn text."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


@dataclass(frozen=True, slots=True)
class _Legs:
    """One search's retriever output, in the shape scoring needs it.

    `*_active` is the abstention flag, and it is not the same question as "did this
    leg return this claim". A leg that never ran must be dropped from the relevance
    average; a leg that ran and did not rank this claim contributes a real zero.
    """

    vector: dict[str, tuple[int, float]]
    lexical: dict[str, tuple[int, float]]
    vector_active: bool
    lexical_active: bool
    lexical_terms: int


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
        max_per_slot: int = 2,
        filter_retry_multiplier: int = 10,
        w_episode: float = 0.5,
        max_episodes: int = 3,
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
        self.max_per_slot = max_per_slot
        self.filter_retry_multiplier = filter_retry_multiplier
        # An episode has to be *twice* as convincing as a claim to outrank it. Raw turn
        # text beating a curated fact is the obvious way this feature makes retrieval
        # worse, and it is easy to hit: an episode contains the query's words verbatim,
        # while the claim extracted from it is a normalized triple that may share none
        # of them. So the episode leg is discounted rather than trusted, and capped as
        # well - a transcript has far more turns than facts, and an uncapped tail lets
        # a single well-worded conversation crowd out everything the store knows.
        self.w_episode = w_episode
        self.max_episodes = max_episodes

    def search(
        self,
        query: str,
        scope: Scope,
        *,
        k: int = 10,
        as_of: datetime | None = None,
        include_invalidated: bool = False,
        memory_types: Sequence[MemoryType] | None = None,
        min_score: float = 0.0,
        include_episodes: bool = False,
    ) -> list[Retrieved]:
        """Return the top `k` results for `query`, each with a populated `Explanation`.

        `as_of` is transaction-time travel: the result is what we believed at that
        instant, including claims we have since retracted. `include_invalidated`
        additionally lifts the liveness filter, surfacing claims that were already
        dead at `as_of` - useful for auditing, wrong for answering a question.

        `min_score` drops results below a normalized relevance (see
        `scoring.normalized_score`); at the default 0.0 nothing is dropped. The right
        value is a property of the store rather than of this library - it moves with
        corpus size and with the embedder - so measure it with
        `calibrate.calibrate_min_score` rather than picking one.

        `include_episodes` widens the search to the raw turns behind the claims, and
        returns `EpisodeResult` for those - so the caller can always tell a fact from
        something someone said. It is opt-in rather than the default because the two
        are not substitutes: existing callers ask this method for facts, and quietly
        starting to answer with conversation would change what lands in every prompt
        built on it. `memory_types` is a claim-only filter and, when given, suppresses
        the episode leg entirely rather than pretending a turn has a memory type.
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
        wanted = set(memory_types) if memory_types is not None else None

        results, saturated = self._gather(
            query, scopes, limit, as_of, include_invalidated, wanted, now, min_score)

        # Filter starvation. `memory_types` is applied after fusion truncated the pool,
        # so a rejected candidate has already consumed a slot and a narrow filter can
        # come back empty while matches sit just past the cut. Pushing the filter into
        # both retrievers would widen the `Store` protocol for a case that is rare;
        # noticing that the pool was full and re-asking once is not. The retry is
        # bounded and happens only when the shortfall could actually be an artefact.
        if wanted is not None and saturated and len(results) < k:
            results, _ = self._gather(
                query, scopes, limit * self.filter_retry_multiplier, as_of,
                include_invalidated, wanted, now, min_score)

        ranked: list[Retrieved] = list(self._rank(results, k))
        if not include_episodes or wanted is not None:
            return ranked
        return self._interleave(
            ranked, self._episodes(query, scopes, limit, as_of, min_score), k)

    # -- internals -----------------------------------------------------------

    def _gather(
        self,
        query: str,
        scopes: Sequence[Scope],
        limit: int,
        as_of: datetime | None,
        include_invalidated: bool,
        wanted: set[MemoryType] | None,
        now: datetime,
        min_score: float,
    ) -> tuple[list[Result], bool]:
        """Run both legs at `limit` and return the surviving results, unsorted.

        The second element reports whether either leg came back full, i.e. whether
        there is any reason to believe candidates were cut off. It is the only honest
        trigger for a retry: a short result set from a leg that returned fewer than
        `limit` hits has nothing more to give, and re-asking would be pure cost.
        """
        vector_hits = self._vector_search(query, scopes, limit, as_of, include_invalidated)
        lexical_hits, lexical_terms = self._lexical_search(
            query, scopes, limit, as_of, include_invalidated)

        fused = reciprocal_rank_fusion(
            {VECTOR: vector_hits, LEXICAL: lexical_hits},
            k=self.rrf_k,
            weights={VECTOR: self.w_vector, LEXICAL: self.w_lexical},
        )
        saturated = len(vector_hits) >= limit or len(lexical_hits) >= limit
        if not fused:
            return [], saturated

        legs = _Legs(
            vector=_positions(vector_hits),
            lexical=_positions(lexical_hits),
            # A leg that returned nothing is indistinguishable from one that never ran,
            # and both must be dropped from the relevance average rather than counted
            # as a zero vote - otherwise every result of a lexical-only query is halved.
            vector_active=bool(vector_hits),
            lexical_active=bool(lexical_hits),
            lexical_terms=lexical_terms,
        )

        # Hydrate every fused candidate in one round trip. Fetching them individually
        # makes a search cost O(candidates) queries — the classic N+1 — so retrieval
        # would scale with how many results it considered rather than with the query.
        # `get_claims` is optional on the Store protocol; fall back for third-party ones.
        bulk = getattr(self.store, "get_claims", None)
        claims = (bulk(list(fused))
                  if bulk is not None
                  else {cid: c for cid in fused
                        if (c := self.store.get_claim(cid)) is not None})

        results: list[Result] = []
        for claim_id, fusion in fused.items():
            claim = claims.get(claim_id)
            if claim is None:
                continue  # raced with a delete; a missing row is not a ranking error
            if wanted is not None and claim.memory_type not in wanted:
                continue
            if not self._believed_by(claim, as_of):
                continue

            result = self._explain(claim, fusion, legs, now)
            if result.score < min_score:
                continue
            results.append(result)
        return results, saturated

    def _episodes(
        self,
        query: str,
        scopes: Sequence[Scope],
        limit: int,
        as_of: datetime | None,
        min_score: float,
    ) -> list[EpisodeResult]:
        """The same two legs over raw turns, discounted and capped.

        Structurally a copy of `_gather` minus everything episodes do not have. There
        is no liveness filter because nothing retires a turn, no `memory_types` because
        a turn has no type, and no quality rescoring because recency decay, confidence
        and salience are all properties of an extracted claim. What is left is the part
        that actually finds things: BM25 over the words that were used, and cosine over
        what they meant.
        """
        vector_hits = self._episode_vector_search(query, scopes, limit, as_of)
        lexical_hits, terms = self._episode_lexical_search(query, scopes, limit, as_of)

        fused = reciprocal_rank_fusion(
            {VECTOR: vector_hits, LEXICAL: lexical_hits},
            k=self.rrf_k,
            weights={VECTOR: self.w_vector, LEXICAL: self.w_lexical},
        )
        if not fused:
            return []

        vector_pos = _positions(vector_hits)
        lexical_pos = _positions(lexical_hits)
        episodes = self._hydrate_episodes(list(fused))

        out: list[EpisodeResult] = []
        for episode_id, fusion in fused.items():
            episode = episodes.get(episode_id)
            if episode is None:
                continue  # raced with a purge; a missing row is not a ranking error
            v = vector_pos.get(episode_id)
            lx = lexical_pos.get(episode_id)
            evidence = relevance(
                vector=(vector_relevance(0.0 if v is None else v[1])
                        if vector_hits else None),
                lexical=(lexical_relevance(0.0 if lx is None else lx[1], terms)
                         if lexical_hits else None),
                w_vector=self.w_vector,
                w_lexical=self.w_lexical,
            )
            score = evidence * self.w_episode
            if score < min_score:
                continue
            out.append(EpisodeResult(
                episode=episode,
                score=score,
                explain=Explanation(
                    vector_rank=None if v is None else v[0],
                    vector_score=None if v is None else v[1],
                    lexical_rank=None if lx is None else lx[0],
                    lexical_score=None if lx is None else lx[1],
                    fusion_score=fusion,
                    # No quality multiplier to divide back out, so the raw score is the
                    # fusion term itself - which keeps `raw_score` meaning the same
                    # thing it means for a claim: what fusion produced, before scoring.
                    raw_score=fusion,
                    final_score=score,
                ),
            ))
        out.sort(key=lambda r: (-r.score, r.episode.id))
        return out[:self.max_episodes]

    def _hydrate_episodes(self, ids: Sequence[str]) -> dict[str, Episode]:
        """One round trip if the store offers it, N if it is a third-party one."""
        bulk = getattr(self.store, "get_episodes", None)
        if bulk is not None:
            return bulk(ids)
        return {eid: e for eid in ids if (e := self.store.get_episode(eid)) is not None}

    def _episode_vector_search(
        self, query: str, scopes: Sequence[Scope], limit: int, as_of: datetime | None,
    ) -> list[tuple[str, float]]:
        """Vector leg over turns. Abstains on a zero-norm query, as the claim leg does.

        Absent from a third-party `Store`, this leg simply does not run. Degrading to
        the lexical half is right: it is the stronger of the two for verbatim recall
        anyway, and refusing to search at all would be a worse answer than a narrower
        one.
        """
        search = getattr(self.store, "vector_search_episodes", None)
        if search is None:
            return []
        qvec = np.asarray(self.embedder.encode([query])[0], dtype=np.float32)
        if float(np.linalg.norm(qvec)) <= 0.0:
            return []
        return list(search(qvec, scopes, limit, as_of))

    def _episode_lexical_search(
        self, query: str, scopes: Sequence[Scope], limit: int, as_of: datetime | None,
    ) -> tuple[list[tuple[str, float]], int]:
        """BM25 over turns, reduced to content terms exactly as the claim leg is.

        The stopword guard matters more here, not less: turns are long and
        conversational, so a query that survives as `"do"` and `"about"` matches
        essentially every episode in the store.
        """
        search = getattr(self.store, "lexical_search_episodes", None)
        if search is None:
            return [], 0
        reduced = analyze(query)
        if reduced.abstains:
            return [], 0
        return list(search(reduced.text, scopes, limit, as_of)), len(reduced.terms)

    @staticmethod
    def _interleave(claims: list[Retrieved], episodes: list[EpisodeResult],
                    k: int) -> list[Retrieved]:
        """Merge the episode tail into the claim ranking without disturbing it.

        A plain re-sort would undo the diversity pass in `_rank`, which deliberately
        demotes rather than drops and therefore hands back a list that is *not* in
        score order. So the claim order is taken as authoritative and each episode is
        placed at the first point where it beats the next claim. Ties go to the claim,
        which is the same tiebreak the weight expresses: equal evidence, prefer the
        thing that was extracted and reconciled.
        """
        out: list[Retrieved] = []
        pending = list(episodes)
        for r in claims:
            while pending and pending[0].score > r.score:
                out.append(pending.pop(0))
            out.append(r)
        out.extend(pending)
        return out[:k]

    def _rank(self, results: list[Result], k: int) -> list[Result]:
        """Order by score, then spread the head across fact slots, then cut to `k`.

        Sorting takes the claim id as a secondary key. Ties are common - byte-identical
        claims agree on every signal by construction - and dict order alone would let
        the answer depend on insertion history.

        The diversity pass demotes rather than drops. A cluster of near-identical
        claims in one slot was measured taking 5 of 8 prompt slots, which is a wasted
        prompt; but capping by deletion would make `k` mean something different
        depending on how the corpus happened to cluster, and would silently hide the
        cluster from anyone auditing it. Demotion costs nothing when there is nothing
        else to show and everything to gain when there is.

        Diversity is measured on `fact_key` - owner, subject, predicate - and
        deliberately not on embedding distance. Greedy MMR over the shipped
        `HashingEmbedder` measurably *reduced* topical coverage: 6.30 distinct
        predicates per result set at lambda=0.7 against 6.78 with no diversity pass at
        all, while still leaving 5 of 8 slots to the duplicate cluster. Those vectors
        are lexical, so two claims about one subject in different words look far apart
        and two claims about different subjects in similar words look close - MMR
        diversifies the wrong axis. Slot identity is what it was trying to approximate,
        and the store already knows it exactly: capping on it gives 7.04 with the
        ranking otherwise untouched.
        """
        results.sort(key=lambda r: (-r.score, r.claim.id))
        if self.max_per_slot <= 0:
            return results[:k]

        head: list[Result] = []
        overflow: list[Result] = []
        used: dict[str, int] = {}
        for r in results:
            slot = r.claim.fact_key
            seen = used.get(slot, 0)
            if seen < self.max_per_slot:
                used[slot] = seen + 1
                head.append(r)
            else:
                overflow.append(r)
        return (head + overflow)[:k]

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

    def _lexical_search(
        self,
        query: str,
        scopes: Sequence[Scope],
        limit: int,
        as_of: datetime | None,
        include_invalidated: bool,
    ) -> tuple[list[tuple[str, float]], int]:
        """Lexical leg, reduced to content terms and skipped when none survive.

        The store ORs every alphanumeric token it is handed, and at personal-memory
        scale the stopwords are the rare tokens - so `"what do you know about me?"`
        ranks on the IDF of "do". Sending only the content terms is the same guard the
        vector leg applies to a zero-norm query, expressed in the only place the
        retriever controls: what it asks for. See `analyze`.

        Returns the hits and the number of terms they were scored over, which is what
        makes BM25 comparable across queries of different lengths.
        """
        reduced = analyze(query)
        if reduced.abstains:
            return [], 0
        hits = self.store.lexical_search(
            reduced.text, scopes, limit, as_of, include_invalidated)
        return list(hits), len(reduced.terms)

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

    def _explain(self, claim: Claim, fusion: float, legs: _Legs, now: datetime) -> Result:
        v = legs.vector.get(claim.id)
        lx = legs.lexical.get(claim.id)
        recency = recency_factor(claim, self.registry, now)
        quality = {
            "recency": recency,
            "confidence": claim.confidence,
            "salience": claim.salience,
            "w_recency": self.w_recency,
            "w_confidence": self.w_confidence,
            "w_salience": self.w_salience,
        }
        # A leg that ran scores an unlisted claim 0.0; a leg that abstained scores it
        # `None`, which drops it from the average instead of voting against the claim.
        evidence = relevance(
            vector=(vector_relevance(0.0 if v is None else v[1])
                    if legs.vector_active else None),
            lexical=(lexical_relevance(0.0 if lx is None else lx[1], legs.lexical_terms)
                     if legs.lexical_active else None),
            w_vector=self.w_vector,
            w_lexical=self.w_lexical,
        )
        score = normalized_score(evidence, **quality)
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
            raw_score=final_score(fusion, **quality),
            final_score=score,
        )
        return Result(claim=claim, score=score, explain=explain)


def _positions(hits: Sequence[tuple[str, float]]) -> dict[str, tuple[int, float]]:
    """Map item id -> (0-based rank, raw score), keeping the best rank on repeats."""
    out: dict[str, tuple[int, float]] = {}
    for rank, (item_id, score) in enumerate(hits):
        out.setdefault(item_id, (rank, score))
    return out
