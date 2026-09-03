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
5. **Everything that is two rows and a join.** "Where is my manager's employer based" is
   two claims with nothing joining them, and neither lookup leg can find the second: the
   claim holding the answer shares no vocabulary with the question. The graph leg walks
   out of the entities the first two legs just named, which is Zep's φ_bfs — see
   `retrieve/spread.py` for why the seeds come from the answer rather than the query, and
   `retrieve/intent.py` for what keeps every other query from paying for it. It ships at
   `w_graph=0.0`; the measurement behind that default is in `docs/BENCHMARKS.md`.

Everything here is deterministic. No LLM sits on the read path, and identical inputs
produce an identical ordering, ties included - unstable ranking makes retrieval
regressions impossible to bisect. "Identical inputs" means the *content*: ties break on
a content hash rather than on a row id, because ids are minted per ingest and an
ordering that only holds within one store is not reproducibility, it is luck.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import (
    TYPE_CHECKING, Any, Callable, ClassVar, Collection, Iterable, Literal, NamedTuple,
    Sequence, overload,
)

import numpy as np

from ..embed.base import Embedder
from ..rerank import Reranker, rerank
from ..schema import PredicateRegistry
from ..store.base import Store, bulk_claims, resolve_states
from ..telemetry import (
    RETRIEVAL_LATENCY_MS,
    RETRIEVAL_OBSERVATION_RANK_CORR,
    RETRIEVAL_QUALITY_FACTOR,
    RETRIEVAL_QUERY,
    RETRIEVAL_RESULTS,
    Recorder,
    rank_correlation,
    script_of,
)
from ..types import (
    CLAIM,
    EPISODE,
    Claim,
    Episode,
    Explanation,
    MemoryType,
    Result,
    Scope,
    owner_key,
    time_axes,
    utcnow,
)
from .anchor import PATH, SUBJECT, anchor_of, query_tokens
from .analyze import analyze
from .fusion import reciprocal_rank_fusion
from .intent import Intent, classify
from .compose import names_derived
from .intent import is_comparison, is_relational, observed_refs
from .intent import weights as intent_weights
from .scoring import (
    final_score,
    lexical_relevance,
    normalized_score,
    quality_boost,
    recency_factor,
    relevance,
    vector_relevance,
)
from .spread import rank_paths, seed_keys
from .temporal import TEMPORAL, anchor_for, rank as rank_by_time
from .traverse import GraphTraverser

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from ..entities import EntityRegistry

# Retriever names. Shared between the fusion weights and the `Explanation` fields so
# the two cannot drift apart under a rename.
VECTOR = "vector"
LEXICAL = "lexical"
GRAPH = "graph"


# There is deliberately no default relevance floor here. One was measured and shipped
# (0.25, calibrated on a 36-claim corpus) and it is wrong at both ends: the window
# between the weakest correct answer and the best wrong one moves with corpus size, and
# the windows at 5 claims and at 1,000 do not intersect. See `calibrate.py` for the
# measurement and for `calibrate_min_score`, which derives the number from a
# deployment's own probes instead of guessing it here.


def _as_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


class DegradedRetrievalWarning(UserWarning):
    """A configured retrieval leg could not run against this store.

    Raised once per `HybridRetriever`, not once per query: a store that cannot traverse
    cannot traverse for the whole process, and a warning per search would bury the
    finding under itself.

    Degrading is right — two legs are a worse answer than three and a far better one than
    an exception out of `search()` — but degrading *silently* is how a deployment runs for
    a month believing it has multi-hop retrieval. `RemoteStore.adjacent` is the case this
    exists for: it is present on the object, which is what a `getattr` guard checks, and
    raises `NotImplementedError` when called.
    """


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
        if self.explain.temporal_rank is not None:
            # Without this a turn found only by proximity reprs as `no-retriever`, which
            # is the one reading that is wrong: a leg found it, and which one is the point.
            legs.append(f"time#{self.explain.temporal_rank}")
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


class _Weights(NamedTuple):
    """The leg weights one search actually ran with, and what decided them.

    Resolved once, at the top of `search`, and carried down — rather than read off
    `self` at each of the five places a weight is needed. The two spellings look
    identical until intent weighting is on, at which point reading `self.w_graph` inside
    `_explain` while fusion used the gated value produces a relevance average that
    divides by a leg the fusion never ran. That is a silent scoring error and nothing
    downstream can see it.
    """

    vector: float
    lexical: float
    graph: float
    temporal: float
    #: `None` when `intent_weighting` is off, which is a different statement from any of
    #: the four intents: it says nothing classified this query, so nothing scaled it.
    intent: Intent | None


@dataclass(frozen=True, slots=True)
class _Legs:
    """One search's retriever output, in the shape scoring needs it.

    `*_active` is the abstention flag, and it is not the same question as "did this
    leg return this claim". A leg that never ran must be dropped from the relevance
    average; a leg that ran and did not rank this claim contributes a real zero.
    """

    vector: dict[str, tuple[int, float]]
    lexical: dict[str, tuple[int, float]]
    graph: dict[str, tuple[int, float]]
    vector_active: bool
    lexical_active: bool
    graph_active: bool
    lexical_terms: int


class UnjoinedStoreWarning(DegradedRetrievalWarning):
    """The graph leg is configured, and this store holds nothing for it to walk.

    A sibling of its parent rather than the same warning with different text, because the
    two say different things and are fixed in different places. `DegradedRetrievalWarning`
    means the *backend* cannot traverse — permanent for the process, fixed by changing
    store. This means the *data* has no chains in it: every live claim's object is a leaf,
    so a walk can only re-return the neighbourhood the lookup legs already ranked. It is
    fixed in the write path, by storing facts whose subject is not the one hub everything
    hangs off, and it stops being true the moment one lands.

    Subclassed so an existing `filterwarnings` on the parent still catches it, and named
    separately so nobody reads "your backend is wrong" off a store that is merely young.

    Measured on LongMemEval, whose 78 claims share a single subject: running the leg there
    costs 1.6 points of single-session-user R@12 and gains nothing anywhere, because
    fusion reads positions and a leg with no paths still votes.
    """


#: Searches a tenant's connectivity reading is reused for before it is taken again.
#:
#: Not a clock, because a retriever that behaved differently at 3am would be untestable
#: and this repository pins `now=` everywhere for that reason. A counter is deterministic:
#: the same sequence of searches re-measures at the same points on every run.
#:
#: The staleness this admits only ever runs the leg *less*. A store gains joins and stays
#: gated for up to this many searches, which degrades to `w_graph=0.0` — the shipped
#: default. It cannot go the other way: claims do not un-join except by retirement, which
#: the liveness predicate already excludes. So the error is bounded, one-directional, and
#: lands on the configuration that ships.
#:
#: 256 against a measurement that costs 0.4 ms at 1,000 live claims and 2.6 ms at 10,000,
#: so the amortised cost is under 10 microseconds a search on a store far larger than most.
GATE_RECHECK_EVERY = 256


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
        max_per_source: int = 0,
        claims_as_index: bool = False,
        w_temporal: float = 0.0,
        w_graph: float = 0.0,
        derived_terms: "Collection[str]" = (),
        graph_seeds: int = 5,
        graph_depth: int = 2,
        traverser: "GraphTraverser | None" = None,
        intent_weighting: bool = True,
        reranker: "Reranker | None" = None,
        rerank_top_n: int = 20,
        telemetry: Recorder | None = None,
        entities: "EntityRegistry | None" = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.registry = registry
        #: Per-tenant `(searches since measured, does anything here chain?)`. Only ever
        #: written when `w_graph > 0`, so the shipped default pays nothing for it.
        self._joins: dict[str, tuple[int, bool]] = {}
        self._warned_unjoined = False
        #: Aggregate metrics sink, or `None` (the default and the fast path). See
        #: `memvara.telemetry`: every emission is guarded, and the two numbers that cost
        #: something to produce - the rank correlation and the quality factors - are
        #: computed inside the guard.
        self.telemetry = telemetry
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
        # How many turns one conversation may hold in the episode head, or 0 to rank
        # episodes on score alone. **It ships at 0**, because switching it on raised
        # the metric it targets and lowered the one that matters.
        #
        # It targets a real effect. Turns are ranked independently but are not
        # independent evidence: the turns scoring highest for a question are usually
        # neighbours in the one conversation that discussed it, so a question whose
        # answer spans two conversations is answered from one of them repeatedly. At
        # `max_episodes=5` on LongMemEval-S, one turn per conversation reached every
        # conversation the answer needed 84.2% of the time against 39.8% for five by
        # score - the evidence really was being retrieved and spent on duplicates.
        #
        # Judged accuracy went the other way: 41.8% against 61.0% for `max_episodes=8`
        # ranked on score alone. Reaching a conversation is not the same as carrying
        # the sentence that answers the question, and one turn from each of five
        # conversations loses the surrounding turns that make any of them usable.
        # Across four arms accuracy tracks *gold turns retrieved* - 2.2 turns for this
        # setting at 41.8%, 6.1 at 61.0%, 10.0 at 70.5% - and not conversations
        # reached. Raising it to 2 or 3 alongside a larger `max_episodes` is the
        # version of this idea that has not been measured yet.
        self.max_per_source = max_per_source
        #: Spend a matched claim on finding turns rather than on a slot of its own.
        #:
        #: A claim and the turn it came from currently compete for one slot, and across
        #: four LongMemEval arms the claim loses that competition: retrieval spending
        #: 68-72% of its slots on claims scored *below* retrieval spending 53%, while
        #: accuracy tracked how many source turns reached the prompt (2.2 turns at 41.8%,
        #: 6.1 at 61.0%, 10.0 at 70.5%). A turns-only arm scored 46.2%, no better than
        #: leaving the claims in — so claims are not obviously worth their slots, and the
        #: remaining question is whether they are worth anything as a *route* to turns.
        #:
        #: With this on, a ranked claim is replaced by the episodes its `sources` names,
        #: each inheriting the claim's score so the ranking is preserved rather than
        #: recomputed, merged with the episode leg and deduplicated. The claim does not
        #: appear: it was the index entry. This is the shape Supermemory publishes —
        #: search the memories, inject the chunk behind each hit.
        #:
        #: **It ships off**, and is inert unless `include_episodes` is also set, since
        #: without episodes there would be nothing left to return.
        self.claims_as_index = claims_as_index
        #: Weight of the **episode** temporal leg, and the switch that runs it at all.
        #: At 0.0 no time-ranked candidates are produced and `Explanation.temporal_rank`
        #: stays `None`.
        #:
        #: **It ships at 0.0**, for the reason in `docs/BENCHMARKS.md`. It is an episode
        #: leg only: a claim already carries a predicate-keyed half-life, which is a
        #: better time signal than raw proximity because only the predicate knows whether
        #: a fact from 2019 is stale. `include_episodes=False` therefore makes this leg
        #: inert whatever its weight, which is the shipped default of `search()`.
        self.w_temporal = w_temporal
        #: Weight of the graph leg in fusion and in the relevance average, and the switch
        #: that turns the leg on at all: at 0.0 no walk runs, nothing is fused from it,
        #: and `Explanation.graph_rank` stays `None` on every result.
        #:
        #: **It ships at 0.0.** See the measured table in `docs/BENCHMARKS.md`: the leg
        #: is a large win on the questions it was built for and a small loss on the ones
        #: it was not, and a default that trades the second for the first is a default
        #: nobody chose. `intent.py` is what makes it affordable — it turns the leg on for
        #: the query classes that gained and leaves it off for the rest — so the shipped
        #: switch is `intent_weighting`, and this is the raw knob under it.
        self.w_graph = w_graph
        #: Relation terms that name a *chain* rather than a stored predicate —
        #: "grandfather" over `father`+`father`. Acquired once from a model by
        #: `retrieve/compose.acquire()` and passed in; nothing here calls one, because
        #: `intent.py` promises to be model-free and `hybrid.py` promises reproducible
        #: retrieval, and a search that could block on an API call breaks both.
        #: Empty is the shipped default and the behaviour of every release before this.
        self.derived_terms = frozenset(derived_terms)
        #: How many entity keys the walk starts from, taken off the head of the fused
        #: vector+lexical list. Keys rather than claims: the key list is what reaches
        #: `Store.adjacent` and the frontier width is what a hop costs. See
        #: `spread.seed_keys`.
        self.graph_seeds = graph_seeds
        #: How many hops out the walk goes. Two, because that is where the measured gap
        #: between traversal and search-then-search opens (`bench/multihop.py`) and
        #: because `GraphTraverser` scores a path multiplicatively — a third hop is
        #: damped to at most 0.56 before edge quality is counted at all, so it rarely
        #: survives fusion against a direct hit and always costs a frontier expansion.
        self.graph_depth = graph_depth
        #: The traversal engine, or `None` — in which case the leg cannot run whatever
        #: `w_graph` says. `Memvara` builds one and hands it over; a `HybridRetriever`
        #: constructed directly against a third-party `Store` gets the two-leg search it
        #: had before, rather than an import-time dependency on a `Store` method that is
        #: optional in the protocol.
        self.traverser = traverser
        #: Whether the configured leg weights are scaled per query shape. On by default,
        #: and at the shipped weights it changes nothing at all: every multiplier in
        #: `intent.MULTIPLIERS` is 1.0 except the graph column, and `w_graph` ships at
        #: 0.0, so zero times a gate is zero either way. It becomes load-bearing the
        #: moment a deployment turns the graph leg on, which is the configuration it was
        #: measured for. `False` runs every query at the configured weights and leaves
        #: `Explanation.intent` unset, which is how a ranking difference is attributed to
        #: this stage rather than argued about.
        self.intent_weighting = intent_weighting
        # Set the first time a walk raises `NotImplementedError`, so a store that cannot
        # traverse is asked once rather than once per query. See
        # `DegradedRetrievalWarning`.
        self._graph_unsupported = False
        #: A reranking pass over the head of the fused list, or `None` — the default,
        #: and an absence rather than a no-op: with nothing configured the stage does not
        #: run, imports nothing and costs nothing, which is what keeps the shipped
        #: configuration offline. See `memvara.rerank`. `rerank_top_n` bounds what it
        #: costs when it is configured: the retriever cuts that deep instead of at `k`,
        #: reranks, and then cuts to `k` — so candidates that fusion put just past the
        #: caller's `k` can be promoted into it, which is the whole point of the stage.
        self.reranker = reranker
        self.rerank_top_n = rerank_top_n
        #: The owner's learned entity aliases, or `None`. Read by the anchoring pass so
        #: a question saying "Big Blue" names a claim filed under `ibm`; without one a
        #: key is its own only spelling, which is what an unmerged store has anyway.
        #: `Memvara` hands over the writer's live registry, so an alias learned this
        #: process anchors the next search without a round trip through the store.
        self.entities = entities

    # Three signatures for one method, because `include_episodes` decides what comes
    # back and the caller almost always knows which at the point of the call. Without
    # this, every caller of the common form is handed `Result | EpisodeResult` and has
    # to narrow a union that cannot occur. See `Memvara.search` for the full argument;
    # the engine carries the same overloads so a wrapper written against it inherits
    # them rather than re-deriving the union.
    @overload
    def search(
        self, query: str, scope: Scope, *, k: int = ...,
        as_of: datetime | None = ..., valid_at: datetime | None = ...,
        known_at: datetime | None = ..., states: Collection[str] | None = ...,
        include_invalidated: bool | None = ...,
        memory_types: Sequence[MemoryType] | None = ..., min_score: float = ...,
        anchored: bool = ..., include_episodes: Literal[False] = ...,
        now: datetime | None = ...,
    ) -> list[Result]: ...

    @overload
    def search(
        self, query: str, scope: Scope, *, k: int = ...,
        as_of: datetime | None = ..., valid_at: datetime | None = ...,
        known_at: datetime | None = ..., states: Collection[str] | None = ...,
        include_invalidated: bool | None = ...,
        memory_types: Sequence[MemoryType] | None = ..., min_score: float = ...,
        anchored: bool = ..., include_episodes: Literal[True],
        now: datetime | None = ...,
    ) -> list[Retrieved]: ...

    @overload
    def search(
        self, query: str, scope: Scope, *, k: int = ...,
        as_of: datetime | None = ..., valid_at: datetime | None = ...,
        known_at: datetime | None = ..., states: Collection[str] | None = ...,
        include_invalidated: bool | None = ...,
        memory_types: Sequence[MemoryType] | None = ..., min_score: float = ...,
        anchored: bool = ..., include_episodes: bool, now: datetime | None = ...,
    ) -> list[Retrieved]: ...

    def search(
        self,
        query: str,
        scope: Scope,
        *,
        k: int = 10,
        as_of: datetime | None = None,
        valid_at: datetime | None = None,
        known_at: datetime | None = None,
        states: Collection[str] | None = None,
        include_invalidated: bool | None = None,
        memory_types: Sequence[MemoryType] | None = None,
        min_score: float = 0.0,
        anchored: bool = False,
        include_episodes: bool = False,
        now: datetime | None = None,
    ) -> list[Any]:
        """Return the top `k` results for `query`, each with a populated `Explanation`.

        Two independent time axes, each defaulting to now. `known_at` is belief-time
        travel: the result is what we believed at that instant, including claims we
        have since retracted. `valid_at` is world-time travel: what was in force then,
        judged with everything we know today — which is how a correction learned in
        August about June becomes reachable at all. `as_of` sets both and is exact
        sugar for `valid_at=known_at=T`; passing it alongside either raises. See
        `memvara.types.time_axes`.

        `states` names the population, as any non-empty subset of
        `("live", "ended", "retired")`. `include_invalidated` is its two-valued alias -
        `False` is `["live"]`, `True` is all three - kept working and not deprecated;
        passing both raises. The flag is what a caller reached for to audit, and it
        cannot say "only the records we stopped believing", which is what an audit is.

        Asking for all three lifts the end-of-life filters, surfacing claims that were
        already dead - useful for auditing, wrong for answering a question. The belief
        floor stands under every subset, so knowledge from after `known_at` never leaks
        in; see `store.state_predicate` for exactly what each subset does and does not
        lift.

        `min_score` drops results below a normalized relevance (see
        `scoring.normalized_score`); at the default 0.0 nothing is dropped. The right
        value is a property of the store rather than of this library - it moves with
        corpus size and with the embedder - so measure it with
        `calibrate.calibrate_min_score` rather than picking one.

        `anchored` is the other way to say no, and it needs no number. Every result
        carries `Explanation.anchor` — which end of the claim the query named, or
        `"path"` for a claim the graph leg reached from one that was, or `None` for a
        claim the query names at neither end. `anchored=True` drops that last kind before
        the cut, so a question about an entity the store has never heard of comes back
        empty instead of answered from the nearest row about somebody else. It is a
        filter on *claims*: an episode has no subject to name, so the episode leg is
        unaffected by it. See `retrieve/anchor.py`.

        `include_episodes` widens the search to the raw turns behind the claims, and
        returns `EpisodeResult` for those - so the caller can always tell a fact from
        something someone said. It is opt-in rather than the default because the two
        are not substitutes: existing callers ask this method for facts, and quietly
        starting to answer with conversation would change what lands in every prompt
        built on it. `memory_types` is a claim-only filter and, when given, suppresses
        the episode leg entirely rather than pretending a turn has a memory type.

        The graph leg does not appear in this signature and is a constructor argument
        (`w_graph`) rather than a per-call one, deliberately. It changes which claims are
        *candidates*, not which are returned, so a caller flipping it per query would get
        two different rankings for one store with nothing in the result to say which they
        had asked for — `Explanation.graph_rank` and `Explanation.intent` are how a result
        says what ran.

        The declared return type follows that flag rather than covering both cases:
        `list[Result]` unless episodes were asked for. The annotation on the
        implementation is `list[Any]` only because an overloaded implementation cannot
        name a return type narrower than every variant's; the overloads above are what
        a caller sees.
        """
        valid_at, known_at = time_axes(as_of, valid_at, known_at)
        # Resolved once, here, and carried as a tuple from this line down. The alias is
        # a facade spelling; below it there is one parameter, so no inner call can pass
        # both and no inner call can disagree about what the flag meant.
        wanted_states = resolve_states(states, include_invalidated)
        if k <= 0:
            return []
        rec = self.telemetry
        t0 = perf_counter() if rec is not None else 0.0

        # A narrow scope inherits everything above it, so a session-level question can
        # still answer from what the user told us months ago in another session. The
        # sibling direction never opens: `ancestors()` walks strictly upward, and the
        # store matches each scope tuple exactly, so no query can reach sideways into
        # a peer session, a peer user, or - the one that would actually matter -
        # another tenant.
        scopes = scope.ancestors()

        # Decay is measured at the instant being asked about, not at wall-clock now —
        # and that instant is `known_at`, the belief clock, because recency answers
        # "how long ago did we last hear this, from where the question stands". Asking
        # what we now believe about June must not score an August restatement as two
        # months stale; asking what we believed on 1 August must score it from there.
        # `now` replaces the *clock read*, never either axis: a caller who named
        # `known_at` still decays from `known_at`. What it fixes is that two identical
        # searches seconds apart score every claim slightly differently, because the
        # fallback is the wall clock and `recency_factor` reads it. Measured on 2Wiki:
        # 3,000 of 3,000 questions differ between two back-to-back passes, in the
        # low-order digits only — enough, on a near-tie at the `k` boundary, to change
        # which rows land inside the cut. `Consolidator.run(now=)` is the same parameter
        # for the same reason on the write path.
        now = _as_utc(known_at) if known_at is not None else (
            _as_utc(now) if now is not None else utcnow())

        # Over-fetch per retriever: fusion can only rank what it was given, and a claim
        # that BM25 puts first is worthless if the vector list was cut before it and
        # the final k is small.
        # How deep the pipeline ranks before the caller's `k` is applied. Identical to
        # `k` unless a reranker is configured, in which case the stage needs candidates
        # below the cut to have anything to promote: reranking the same `k` items the
        # caller was going to get can only reorder them, which changes what is read
        # first and cannot change what is present at all.
        depth = k if self.reranker is None else max(k, self.rerank_top_n)

        limit = max(depth * self.candidate_multiplier, depth)
        wanted = set(memory_types) if memory_types is not None else None

        weights = self._weights(query, timed=valid_at is not None or known_at is not None)

        results, saturated = self._gather(
            query, scope, limit, valid_at, known_at, wanted_states, wanted, now,
            min_score, anchored, weights)

        # Filter starvation. `memory_types` is applied after fusion truncated the pool,
        # so a rejected candidate has already consumed a slot and a narrow filter can
        # come back empty while matches sit just past the cut. Pushing the filter into
        # both retrievers would widen the `Store` protocol for a case that is rare;
        # noticing that the pool was full and re-asking once is not. The retry is
        # bounded and happens only when the shortfall could actually be an artefact.
        #
        # `anchored` is a filter of the same shape and gets the same retry: an anchored
        # claim with little vocabulary in common with the question sits past the first
        # cut exactly as a filtered memory type does. `min_score` deliberately does
        # not — deeper candidates have less evidence, not more.
        if (wanted is not None or anchored) and saturated and len(results) < depth:
            results, _ = self._gather(
                query, scope, limit * self.filter_retry_multiplier, valid_at, known_at,
                wanted_states, wanted, now, min_score, anchored, weights)

        ranked: list[Retrieved] = list(self._rank(results, depth))
        if include_episodes and wanted is None:
            episodes = self._episodes(query, scopes, limit, valid_at, known_at,
                                      min_score, weights, now)
            if self.claims_as_index:
                ranked = self._sourced(ranked, episodes, depth)
            else:
                ranked = self._interleave(ranked, episodes, depth)
        if self.reranker is not None:
            # Last, deliberately. Everything above it — fusion, the recency half-lives,
            # the per-slot diversity cap, the episode discount — is the ranking this
            # library is arguing for, and the reranker is a second opinion on its head,
            # not a replacement for it. Running before the diversity pass would let a
            # model that likes one phrasing refill the slots the pass exists to spread.
            ranked = rerank(self.reranker, query, ranked, top_n=self.rerank_top_n)[:k]
        if rec is not None:
            self._observe(rec, query, ranked, (perf_counter() - t0) * 1000.0)
        return ranked

    # -- internals -----------------------------------------------------------

    def _weights(self, query: str, *, timed: bool) -> _Weights:
        """The configured weights, gated and scaled by what kind of question this is.

        One classification per search, not one per leg and not one per candidate: it is
        a property of the query, and running it twice would let two halves of one search
        disagree about what was being asked.

        `timed` outranks the marker vocabulary, and it is not a heuristic: it says the
        caller passed `valid_at` or `known_at`, which is a temporal intent stated
        outright. `classify` reads words, and the words are frequently the wrong place to
        look — "what was going on around then" carries `then`, which is a discourse
        connective at least as often as a time reference, while the instant the caller
        already resolved is sitting in the argument list. Deferring to the marker list
        there would gate the time leg off on the one call that named an instant.
        """
        if not self.intent_weighting:
            return _Weights(self.w_vector, self.w_lexical, self.w_graph,
                            self.w_temporal, None)
        shape = classify(query, self.registry)
        intent = Intent.TEMPORAL if timed else shape
        vector, lexical, graph, temporal = intent_weights(
            intent, vector=self.w_vector, lexical=self.w_lexical, graph=self.w_graph,
            temporal=self.w_temporal)
        if intent is Intent.TEMPORAL and not is_comparison(query) and (
                shape is Intent.RELATIONAL or is_relational(query, self.registry)):
            # A question can be about a chain *and* about an instant, and the enum can
            # only hold one of them. Naming an instant used to answer both: `timed`
            # overrode the classifier outright, so `Intent.TEMPORAL`'s multipliers
            # applied, and that row switches the graph leg off.
            #
            # The override earns its place for the temporal leg — a caller who resolved
            # an instant has said more about time than any word could — but it was never
            # meant to say anything about chains, and it silently said the strongest
            # possible thing. "Where was Alice's employer based in 2019" is exactly the
            # query this library exists for, and it was the shape that lost the walk.
            #
            # The same collision happens inside `classify` when the instant is named in
            # words rather than as an argument: "who *currently* leads the team that
            # owns the checkout service" is temporal first, so the chain it also names
            # was discarded and the walk switched off on the query class it exists for.
            # `is_relational` is the second reading the one label could not carry.
            # A comparison frame stays out, as everywhere else the walk is opened:
            # "whose grandfather was born earlier, A or B" is two lookups, and `whose`
            # would otherwise open the walk on the family where it costs the most.
            #
            # So the temporal row still decides three legs and the graph leg keeps the
            # weight the query shape asked for. `Explanation.intent` still reports
            # `temporal`, which is the honest primary reading; `graph_rank` on the rows
            # says the walk ran.
            graph = self.w_graph
        return _Weights(vector, lexical, graph, temporal, intent)

    def _observe(self, rec: Recorder, query: str, results: Sequence[Retrieved],
                 elapsed_ms: float) -> None:
        """Emit the aggregate view of one search.

        The per-call view is already `Explanation`, and this deliberately does not
        duplicate it: what is emitted here are the two *distributions* an explanation
        cannot show, because each of them is a property of many searches rather than of
        one.

        The recorder arrives as an argument rather than being re-read from
        `self.telemetry`, which is the same object. "Never reached with telemetry
        unset" was true and was written only in this docstring, so every emission below
        read as a call on `Recorder | None`; taking the recorder as a parameter puts
        the caller's guard in the signature, where it holds for a reader and for a type
        checker alike.
        """
        # Sliced by script for the same reason the gate is: query volume from a script
        # with no corresponding `gate.pass` is a population whose writes are being
        # dropped and whose reads therefore find nothing.
        rec.counter(RETRIEVAL_QUERY, script=script_of(query))
        rec.gauge(RETRIEVAL_RESULTS, float(len(results)))
        rec.timing(RETRIEVAL_LATENCY_MS, elapsed_ms)

        span = 1.0 + self.w_recency + self.w_confidence + self.w_salience
        counts: list[float] = []
        for r in results:
            if not isinstance(r, Result):
                # An episode's quality fields sit at their neutral 1.0 meaning "not
                # applicable" (see `EpisodeResult`), so including them would report a
                # perfect quality factor for something quality never scored, and an
                # `observation_count` for something nothing observed.
                continue
            counts.append(float(r.claim.observation_count))
            # Unclamped on purpose. Quality is *supposed* to be able only to pull a
            # result down from its evidence, and the single way past 1.0 is a salience
            # reinforced beyond 1.0 - so a value above 1.0 is the direct evidence that
            # freshness and salience are promoting rather than demoting, which is the
            # failure this series exists to catch. Clamping would hide it.
            rec.gauge(RETRIEVAL_QUALITY_FACTOR, quality_boost(
                recency=r.explain.recency,
                confidence=r.explain.confidence,
                salience=r.explain.salience,
                w_recency=self.w_recency,
                w_confidence=self.w_confidence,
                w_salience=self.w_salience,
            ) / span)

        correlation = rank_correlation(counts)
        if correlation is not None:
            # Positive is correct: a fact restated many times should rank above one
            # mentioned once. This went negative when reinforcement was written onto the
            # decayed `salience` rather than the storage base and the nightly pass then
            # erased it — a failure with no exception and no log line anywhere in it,
            # and one that only a trend across searches can show.
            rec.gauge(RETRIEVAL_OBSERVATION_RANK_CORR, correlation)

    def _gather(
        self,
        query: str,
        scope: Scope,
        limit: int,
        valid_at: datetime | None,
        known_at: datetime | None,
        states: Sequence[str],
        wanted: set[MemoryType] | None,
        now: datetime,
        min_score: float,
        anchored: bool,
        weights: _Weights,
    ) -> tuple[list[Result], bool]:
        """Run the legs at `limit` and return the surviving results, unsorted.

        The second element reports whether either *lookup* leg came back full, i.e.
        whether there is any reason to believe candidates were cut off. It is the only
        honest trigger for a retry: a short result set from a leg that returned fewer than
        `limit` hits has nothing more to give, and re-asking would be pure cost. The graph
        leg is deliberately not consulted for it — it is bounded by its own beam and depth
        rather than by `limit`, so a full return from it says nothing about whether a wider
        lookup would find more.

        Two legs run against the query; the third runs against the *answer to it*. See
        `_graph_search`.
        """
        scopes = scope.ancestors()
        vector_hits = self._vector_search(
            query, scopes, limit, valid_at, known_at, states)
        lexical_hits, lexical_terms = self._lexical_search(
            query, scopes, limit, valid_at, known_at, states)

        fused = reciprocal_rank_fusion(
            {VECTOR: vector_hits, LEXICAL: lexical_hits},
            k=self.rrf_k,
            weights={VECTOR: weights.vector, LEXICAL: weights.lexical},
        )
        saturated = len(vector_hits) >= limit or len(lexical_hits) >= limit
        if not fused:
            # Nothing to seed a walk from either. A graph leg that ran on an empty
            # candidate set would have to pick its entities out of the query text, which
            # is the second extractor this design exists to avoid.
            return [], saturated

        # Hydrate every fused candidate in one round trip. Fetching them individually
        # makes a search cost O(candidates) queries — the classic N+1 — so retrieval
        # would scale with how many results it considered rather than with the query.
        #
        # Through `bulk_claims`, which is also what `Memvara.get_all` and `produced` use.
        # The fallback for a store predating `get_claims` used to live here and only
        # here, so those two raised on the very stores this one supported.
        claims = bulk_claims(self.store, list(fused))

        # A second chance for the graph leg, decided by the rows rather than by a word
        # list. `classify` counts the predicates a question names, but it can only count
        # the ones the registry *declared* — and a predicate written through `remember()`
        # is never declared, so on a store whose vocabulary arrived that way the count is
        # always one or zero and every chain question reads as a lookup. Measured on
        # 2WikiMultihopQA, whose 34 relations are all of that kind: the gate captured 0.9
        # points of a 42-point gain.
        #
        # The candidates just hydrated are the store's vocabulary, observed instead of
        # declared, and narrowed to what this query already surfaced. If the question
        # names two of *those* predicates, it is a chain and the walk should run. A
        # comparison question names one — "who was born first" reaches two claims sharing
        # `date_of_birth` — so this stays off where the gate was right to be off.
        #
        # Only ever widens: it runs when intent weighting closed the leg on a store that
        # configured it open, and never narrows what the classifier allowed.
        if weights.graph <= 0.0 < self.w_graph and not is_comparison(query) and (
                names_derived(query, self.derived_terms)
                or len(observed_refs(query, {c.predicate for c in claims.values()},
                                     self.registry.normalize)) > 1):
            # A derived term counts on its own, where a predicate has to be one of two.
            # "Maternal grandfather" *is* the chain — it names no stored predicate at all,
            # which is why counting predicates could never see it. Measured on 2Wiki's
            # `inference` family: 52.6% -> 86.4% answer, 49.0% -> 83.8% chain.
            weights = weights._replace(graph=self.w_graph)

        # And the last word belongs to the store, because no reading of the *query* can
        # answer this one. A walk needs somewhere to go, and on a store where no claim's
        # object is another claim's subject there is nowhere: the leg degenerates into
        # returning other facts about whatever hub the seeds hang off, ranked by a path
        # score that is near-uniform when every path is one hop. Fusion reads positions,
        # so that is a fabricated ranking — the failure `MIN_PROXIMITY` prevents for the
        # temporal leg, which this leg had no equivalent of.
        #
        # After the intent weighting rather than inside it, and that is the whole design.
        # The second chance above can only *widen* — its guard is `weights.graph <= 0.0`
        # — so it cannot undo a walk `classify` opened. Measured: with that hook returning
        # False on all 802 of LongMemEval's gate calls, the run still lost 1.6 points of
        # single-session-user R@12, because `classify` had already opened the leg. A gate
        # that closes what the classifier opened has to sit here.
        if weights.graph > 0.0 and not self._store_has_joins(scope.tenant):
            weights = weights._replace(graph=0.0)

        # Which candidates the question actually names, decided before the walk so the
        # walk can say which of its paths started from one. Read off the rows the lookup
        # legs returned, exactly as the seeds are: no extractor runs over the query.
        #
        # Only the *named* end of each anchored claim is an origin. Its other end is the
        # value, and a walk out of the value reaches the rows that merely share it: from
        # `Project Atlas/deploy_region=eu-west-1`, every other project in `eu-west-1`, one
        # hop away and scored 1.0, each on the very predicate asked. Those are not
        # derivations from the question; they are derivations from its answer. Reached
        # the long way round — out of the named entity and back through the value — they
        # are derivations, two hops out, and are labelled and ranked as such.
        tokens = query_tokens(query)
        spellings = self._spellings(scope)
        anchors = {cid: anchor_of(claim, tokens, spellings)
                   for cid, claim in claims.items()}
        anchored_keys = frozenset(
            claims[cid].subject_key if end == SUBJECT else claims[cid].object_key
            for cid, end in anchors.items() if end is not None)

        graph_hits: list[tuple[str, float]] = []
        walked: dict[str, Claim] = {}
        derived: frozenset[str] = frozenset()
        if anchored and not anchored_keys:
            # Nothing the question is about is in hand, so no path could start from it
            # and nothing the walk found could survive the filter. Skipping is the
            # difference between a question about a stranger costing two legs and
            # costing three, twice — the retry below would walk again at ten times the
            # width.
            pass
        else:
            # `now` handed down rather than re-read. Before this the traverser called
            # `utcnow()` of its own, so one search decayed its quality multiplier and
            # its edge strengths from two instants microseconds apart — coherent within
            # each leg and not between them.
            graph_hits, walked, derived = self._graph_search(
                claims, fused, scope, limit, valid_at, known_at, states, weights.graph,
                now, anchored_keys)
        if graph_hits:
            # Re-fused rather than merged, because RRF reads positions and the positions
            # in the two-leg fusion are not the positions in the three-leg one. Doing it
            # twice is the cost of seeding the third leg from the first two.
            fused = reciprocal_rank_fusion(
                {VECTOR: vector_hits, LEXICAL: lexical_hits, GRAPH: graph_hits},
                k=self.rrf_k,
                weights={VECTOR: weights.vector, LEXICAL: weights.lexical,
                         GRAPH: weights.graph},
            )
            # No second round trip for what the walk found: a `Path` carries the claims
            # it is made of. The lookup legs' rows win a collision, which is a formality —
            # both came from the same store within one call — but keeps one object per id.
            claims = {**walked, **claims}

        legs = _Legs(
            vector=_positions(vector_hits),
            lexical=_positions(lexical_hits),
            graph=_positions(graph_hits),
            # A leg that returned nothing is indistinguishable from one that never ran,
            # and both must be dropped from the relevance average rather than counted
            # as a zero vote - otherwise every result of a lexical-only query is halved.
            vector_active=bool(vector_hits),
            lexical_active=bool(lexical_hits),
            graph_active=bool(graph_hits),
            lexical_terms=lexical_terms,
        )

        results: list[Result] = []
        for claim_id, fusion in fused.items():
            claim = claims.get(claim_id)
            if claim is None:
                continue  # raced with a delete; a missing row is not a ranking error
            if wanted is not None and claim.memory_type not in wanted:
                continue
            if not self._believed_by(claim, known_at):
                continue

            # A claim only the walk found was not in `anchors`; it can still be named
            # outright, and if it is not, the path it arrived on is the tie.
            anchor = (anchors[claim_id] if claim_id in anchors
                      else anchor_of(claim, tokens, spellings))
            if anchor is None and claim_id in derived:
                anchor = PATH
            if anchored and anchor is None:
                continue
            result = self._explain(claim, fusion, legs, now, weights, anchor)
            if result.score < min_score:
                continue
            results.append(result)
        return results, saturated

    def _spellings(self, scope: Scope) -> "Callable[[str], Iterable[str]]":
        """How a stored key may be spelled in this reader's question.

        Resolved under `owner_key(scope)` and nothing wider, for the reason
        `Memvara._probe_entities` gives: an alias learned at tenant level must not
        redefine a user's own entity underneath them.
        """
        entities = self.entities
        if entities is None:
            return lambda key: (key,)
        owner = owner_key(scope)
        return lambda key: entities.spellings(owner, key)

    def _store_has_joins(self, tenant: str) -> bool:
        """Does anything in this tenant lead to anything else in it?

        `True` also when the question cannot be answered, and that is the important half.
        A backend without `connectivity` returns `{}`, which means *it did not look* —
        reading that as "no joins" would switch a working graph leg off on every
        third-party store at once, on the strength of a measurement nobody took. The
        rule is `Memvara.connectivity()`'s: `{}` is not zero.

        Cached per tenant and re-measured every `GATE_RECHECK_EVERY` searches, because
        the reading is a few milliseconds and a search is a few milliseconds. See that
        constant for why the staleness is safe and why it counts searches and not seconds.

        Warns once per retriever, and only when the store actually answered. A caller who
        set `w_graph` and gets nothing is owed the reason, and the reason here is about
        their data rather than their backend — which is why `UnjoinedStoreWarning` is its
        own name.
        """
        seen, joined = self._joins.get(tenant, (GATE_RECHECK_EVERY, True))
        if seen < GATE_RECHECK_EVERY:
            self._joins[tenant] = (seen + 1, joined)
            return joined

        measure = getattr(self.store, "connectivity", None)
        if measure is None:
            self._joins[tenant] = (0, True)
            return True
        counts = measure(tenant)
        if not counts:
            # Present but unable to answer -- a hosted facade too old to report the
            # counts does this. Same verdict as absent, for the same reason.
            self._joins[tenant] = (0, True)
            return True

        joined = counts["joinable_claims"] > 0
        self._joins[tenant] = (0, joined)
        if not joined and not self._warned_unjoined:
            self._warned_unjoined = True
            warnings.warn(
                f"graph retrieval is configured (w_graph={self.w_graph}) and nothing in "
                f"this store chains: none of its {counts['live_claims']} live claim(s) "
                f"have an object that is another claim's subject, so a walk has nowhere "
                f"to go and the leg is not running. This is a property of what has been "
                f"written, not of the backend -- memory_stats reports it as a join rate. "
                f"It resolves on its own once the store holds a fact whose subject is "
                f"not the one everything else hangs off.",
                UnjoinedStoreWarning,
                stacklevel=2,
            )
        return joined

    def _graph_search(
        self,
        claims: dict[str, Claim],
        fused: dict[str, float],
        scope: Scope,
        limit: int,
        valid_at: datetime | None,
        known_at: datetime | None,
        states: Sequence[str],
        w_graph: float,
        now: datetime,
        anchored: frozenset[str] = frozenset(),
    ) -> tuple[list[tuple[str, float]], dict[str, Claim], frozenset[str]]:
        """The third leg: a bounded walk out of the entities the first two just named.

        Returns the ranked `(claim_id, path score)` list and the claims behind it, which
        the paths already carry — so a leg that reaches thirty rows the lookups missed
        costs zero extra store round trips beyond the hops themselves — and the ids of
        every claim on a path that *started from* one of `anchored`, the entity keys the
        question named. Those are the derivations: a claim the question does not mention
        that the store nevertheless ties to one it does, which is the reading
        `Explanation.anchor` reports as `"path"`. A path out of any other seed — the
        value end of a named claim, or the lookup legs' best guess on a question about
        nothing the store holds — proves nothing about the question and marks nothing.

        Every early return here is a *degradation*, and each is a different fact:

        * `w_graph <= 0` or no traverser — the leg is switched off, which is the shipped
          default. Nothing is warned, because nothing is wrong.
        * `NotImplementedError` — the store has `adjacent` and it does not work. That is
          `RemoteStore`, and it is the case a `getattr` guard cannot see, so it is caught
          rather than guarded, warned once, and remembered for the life of this retriever.
        * no seeds — every candidate's ends folded to nothing, which is possible only for
          a candidate set made entirely of retractions.
        * `states` does not include `live` — see below. Nothing is warned; the leg has
          nothing admissible to contribute rather than something it failed to fetch.

        `known_at`/`valid_at` are passed through unchanged, so the walk is evaluated at
        the same pair `search()` was asked about and pins it once before its first hop
        (`GraphTraverser._pin`). An axis left unset is filled by the walk's own clock read
        — the same treatment the store gives the two lookup legs, and the reason a
        three-hop chain here cannot be assembled out of two different afternoons.

        **`states` gates the leg rather than filtering its output.** `Store.adjacent` walks
        the live edges at the pinned instant and takes no `states` argument — a graph of
        retracted edges is not a graph, since the whole point of a retraction is that the
        connection was never there. So every row this leg can produce belongs to the live
        population, and a search asking only for `ended` or `retired` must not receive
        them. It was receiving them: `search(states=["retired"])` on a store where one
        retracted claim had a live neighbour returned that neighbour, ranked *above* the
        retired row the caller actually asked for, because the seeds come from the lookup
        legs and the retired row was a perfectly good seed. An audit query answered with
        live facts is the failure mode that matters here, and it is silent.

        Gating rather than post-filtering, because a post-filter would have to test
        `claim.state`, which is the claim's state **now** — and at a historical `known_at`
        the lookup legs correctly return rows that were live then and are retired today.
        Filtering those out would fix this leg by breaking time travel in the other two.
        """
        if w_graph <= 0.0 or self.traverser is None or self._graph_unsupported:
            return [], {}, frozenset()
        if "live" not in states:
            return [], {}, frozenset()
        # Driven from `fused`, which is the authority on scores, rather than from what
        # the hydration returned. `get_claims` is on the Store protocol and a third-party
        # one that returns an id nobody asked for would otherwise take retrieval down
        # with a `KeyError` — a store being loose with its return value should cost the
        # graph leg a seed, not cost the caller their search.
        seeds = seed_keys([(claims[cid], score) for cid, score in fused.items()
                           if cid in claims], self.graph_seeds)
        if not seeds:
            return [], {}, frozenset()
        try:
            paths = self.traverser.spread(
                seeds, scope, depth=self.graph_depth, k=limit,
                valid_at=valid_at, known_at=known_at, now=now)
        except NotImplementedError as exc:
            self._graph_unsupported = True
            warnings.warn(
                f"graph retrieval is configured (w_graph={w_graph}) and this store "
                f"cannot traverse, so search is running two legs instead of three: {exc}",
                DegradedRetrievalWarning,
                stacklevel=2,
            )
            return [], {}, frozenset()
        derived = frozenset(c.id for p in paths if p.nodes[0] in anchored
                            for c in p.claims)
        return rank_paths(paths), {c.id: c for p in paths for c in p.claims}, derived

    def _episodes(
        self,
        query: str,
        scopes: Sequence[Scope],
        limit: int,
        valid_at: datetime | None,
        known_at: datetime | None,
        min_score: float,
        weights: _Weights,
        now: datetime,
    ) -> list[EpisodeResult]:
        """The legs over raw turns, discounted and capped.

        Structurally a copy of `_gather` minus everything episodes do not have. There
        is no liveness filter because nothing retires a turn, no `memory_types` because
        a turn has no type, and no quality rescoring because recency decay, confidence
        and salience are all properties of an extracted claim. What is left is the part
        that actually finds things: BM25 over the words that were used, cosine over what
        they meant, and — at `w_temporal > 0` — proximity to the instant being asked
        about, which is the only leg here that reads no text at all. See
        `retrieve/temporal.py` for why that one lives on this side and not on the claim
        side.
        """
        vector_hits = self._episode_vector_search(
            query, scopes, limit, valid_at, known_at)
        lexical_hits, terms = self._episode_lexical_search(
            query, scopes, limit, valid_at, known_at)
        anchor = anchor_for(valid_at, known_at, now)
        time_hits = self._episode_time_search(
            scopes, limit, valid_at, known_at, weights.temporal, anchor)

        fused = reciprocal_rank_fusion(
            {VECTOR: vector_hits, LEXICAL: lexical_hits, TEMPORAL: time_hits},
            k=self.rrf_k,
            weights={VECTOR: weights.vector, LEXICAL: weights.lexical,
                     TEMPORAL: weights.temporal},
        )
        if not fused:
            return []

        vector_pos = _positions(vector_hits)
        lexical_pos = _positions(lexical_hits)
        time_pos = _positions(time_hits)
        episodes = self._hydrate_episodes(list(fused))

        out: list[EpisodeResult] = []
        for episode_id, fusion in fused.items():
            episode = episodes.get(episode_id)
            if episode is None:
                continue  # raced with a purge; a missing row is not a ranking error
            v = vector_pos.get(episode_id)
            lx = lexical_pos.get(episode_id)
            tm = time_pos.get(episode_id)
            evidence = relevance(
                vector=(vector_relevance(0.0 if v is None else v[1])
                        if vector_hits else None),
                lexical=(lexical_relevance(0.0 if lx is None else lx[1], terms)
                         if lexical_hits else None),
                # A proximity is already an absolute [0, 1] closeness, like a cosine, so
                # it goes in unmapped. It rides the `graph` slot of the average because
                # the two never run together: one is claims, one is episodes.
                graph=(0.0 if tm is None else tm[1]) if time_hits else None,
                w_vector=weights.vector,
                w_lexical=weights.lexical,
                w_graph=weights.temporal,
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
                    temporal_rank=None if tm is None else tm[0],
                    temporal_score=None if tm is None else tm[1],
                    fusion_score=fusion,
                    # No quality multiplier to divide back out, so the raw score is the
                    # fusion term itself - which keeps `raw_score` meaning the same
                    # thing it means for a claim: what fusion produced, before scoring.
                    raw_score=fusion,
                    final_score=score,
                ),
            ))
        # Content hash, not `id`. See `_rank_claims` for why — the same argument, and
        # `Episode.hash` is already the content digest tier 0 dedupes on.
        out.sort(key=lambda r: (-r.score, r.episode.hash, r.episode.id))
        return self._spread_episodes(out)[:self.max_episodes]

    def _spread_episodes(self, results: "list[EpisodeResult]") -> "list[EpisodeResult]":
        """Spread the episode head across source conversations, before the cut to `k`.

        The claim side of this is `_rank`, and this is the same trade in the same
        shape: demote rather than drop, on an exact key rather than on embedding
        distance. Both choices were measured, and `_rank`'s docstring records why
        greedy MMR lost to a key the store already knows.

        The key here is `Episode.ts`. One `add()` call is one conversation, and
        `_to_episodes` stamps the call's timestamp across the whole batch unless a turn
        carries its own - so equal timestamps mean "arrived together", which is the
        property being diversified over. On the 266-question LongMemEval-S run every
        conversation retrieval reached carried exactly one timestamp across its
        retrieved turns, 1324 of 1324. The converse is not quite exact and cannot be:
        two conversations that genuinely start at the same instant share a key and are
        treated as one, which happened for 2 of 1322 timestamps there. That direction
        costs a slot rather than correctness - the merged pair keeps its best turn.

        This ships disabled: see `max_per_source` in `__init__` for the measurement
        that argues against it as a default.

        Where turns *do* carry their own timestamps, every episode is its own source,
        the head takes them all in score order and the pass is an identity. It is an
        identity for a single-source corpus too, for the same reason `_rank` is: with
        one key, `head` is the first item and `overflow` is the remainder already in
        score order, so `head + overflow` is the input. The pass can only reorder a
        genuine cluster, which is the only case it was built for.
        """
        if self.max_per_source <= 0:
            return results

        head: "list[EpisodeResult]" = []
        overflow: "list[EpisodeResult]" = []
        used: dict[str, int] = {}
        for r in results:
            source = r.episode.ts.isoformat()
            seen = used.get(source, 0)
            if seen < self.max_per_source:
                used[source] = seen + 1
                head.append(r)
            else:
                overflow.append(r)
        return head + overflow

    def _hydrate_episodes(self, ids: Sequence[str]) -> dict[str, Episode]:
        """One round trip if the store offers it, N if it is a third-party one."""
        bulk = getattr(self.store, "get_episodes", None)
        if bulk is not None:
            return bulk(ids)
        return {eid: e for eid in ids if (e := self.store.get_episode(eid)) is not None}

    def _episode_time_search(
        self, scopes: Sequence[Scope], limit: int, valid_at: datetime | None,
        known_at: datetime | None, w_temporal: float, anchor: datetime,
    ) -> list[tuple[str, float]]:
        """Turns nearest the asked instant, or nothing. Degrades like the other legs.

        `episodes_near` is optional on the `Store` protocol, so a third-party store
        simply does not run this leg — the same treatment `vector_search_episodes`
        already gets, and right for the same reason: a narrower answer beats refusing to
        search. Unlike the graph leg there is no raising implementation to catch, because
        `RemoteStore` cannot serve any episode search at all and is refused a whole
        server before it gets here.
        """
        if w_temporal <= 0.0:
            return []
        near = getattr(self.store, "episodes_near", None)
        if near is None:
            return []
        return rank_by_time(
            near(anchor, scopes, limit, valid_at=valid_at, known_at=known_at), anchor)

    def _episode_vector_search(
        self, query: str, scopes: Sequence[Scope], limit: int,
        valid_at: datetime | None, known_at: datetime | None,
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
        return list(search(qvec, scopes, limit, valid_at=valid_at, known_at=known_at))

    def _episode_lexical_search(
        self, query: str, scopes: Sequence[Scope], limit: int,
        valid_at: datetime | None, known_at: datetime | None,
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
        hits = search(reduced.text, scopes, limit, valid_at=valid_at, known_at=known_at)
        return list(hits), len(reduced.terms)

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

    def _sourced(self, claims: "Sequence[Retrieved]",
                 episodes: "Sequence[EpisodeResult]", k: int) -> "list[Retrieved]":
        """Replace each ranked claim with the turns it was extracted from.

        The claim keeps its place in the ranking and gives up its slot: every episode it
        cites inherits its score, so the order the legs produced is preserved rather than
        recomputed. Recomputing would mean scoring each sourced turn against the query
        again — a second pass over text the episode leg has already ranked — which is the
        cost this is trying to avoid.

        **Inheriting the score is an approximation and worth naming.** A claim's score
        measures how well *the claim* matched, and it is being used to order a turn. The
        justification is that the claim was extracted from that turn, so a query the claim
        answers is a query the turn is about; the failure mode is a long turn that
        produced one narrowly-matching claim and now ranks on it.

        Deduplication is by episode id and the episode leg wins ties, because it reached
        the turn on the turn's own text. Several claims commonly cite one turn, so the
        result can be *shorter* than either input — that is the honest outcome, not a bug,
        and it is why the arm that measures this reports distinct turns retrieved
        alongside accuracy.

        A `sources` entry naming a turn that no longer exists contributes nothing.
        `erase` removes a turn and leaves the claims that cited it, so a dangling id is a
        normal state of the store and not a corruption to raise on.
        """
        seen = {e.episode.id for e in episodes}
        wanted: dict[str, float] = {}
        for r in claims:
            # `Retrieved` is a union and only the claim arm has provenance. `_rank`
            # returns claims alone today, so this never fires — but the annotation is
            # the contract, and an episode arriving here should pass through as a turn
            # rather than raise on a missing attribute.
            if isinstance(r, EpisodeResult):
                continue
            for episode_id in r.claim.sources or ():
                if episode_id in seen:
                    continue
                # First claim to cite a turn sets its score: claims arrive in rank order,
                # so this is the best-scoring claim that points at it.
                wanted.setdefault(episode_id, r.score)

        hydrated = self._hydrate_episodes(list(wanted)) if wanted else {}
        # Episodes throughout: every claim gave up its slot, so what comes back is turns.
        out: list[EpisodeResult] = list(episodes)
        for episode_id, score in wanted.items():
            episode = hydrated.get(episode_id)
            if episode is None:
                continue
            # An explanation, not the default empty one. "Retrieval that cannot explain
            # itself is impossible to debug" is this library's stated reason for having
            # the field at all, and a turn that arrived through a claim is exactly the
            # case a reader will be puzzled by: it may share no vocabulary with the query,
            # because the claim extracted from it did the matching. `raw_score` and
            # `final_score` carry the inherited value so a ranking diff still works, and
            # every leg stays `None` because this turn won no leg — it was cited.
            out.append(EpisodeResult(
                episode=episode, score=score,
                explain=Explanation(raw_score=score, final_score=score),
            ))
        # Content hash before id, for the reason `_episodes` states: an episode id is
        # minted at ingest, so breaking ties on it makes the order a property of which
        # ingest ran rather than of the data.
        out.sort(key=lambda r: (-r.score, r.episode.hash, r.episode.id))
        # Widened at the boundary rather than typed loosely throughout: `list` is
        # invariant, so a `list[EpisodeResult]` is not a `list[Retrieved]` however
        # obviously every element is one.
        widened: list[Retrieved] = [*out[:k]]
        return widened

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
        # `value_key` before `id`, and the difference is the whole promise in this
        # module's docstring. A claim id is `uuid4`, minted fresh at ingest — so breaking
        # ties on it gives an ordering that is stable *within* a store and a coin flip
        # *across* two ingests of identical data. That is exactly the comparison a
        # benchmark, a regression test and a `git bisect` all make, and it was measured:
        # repeated LOCOMO runs disagreed by up to 0.07 points with nothing else changed.
        # `value_key` is derived from the claim's content, so identical data ranks
        # identically everywhere. `id` stays as the final key so the order is still total
        # when two rows share a value.
        results.sort(key=lambda r: (-r.score, r.claim.value_key, r.claim.id))
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
        valid_at: datetime | None,
        known_at: datetime | None,
        states: Sequence[str],
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
        return list(self.store.vector_search(
            qvec, scopes, limit, valid_at=valid_at, known_at=known_at, states=states))

    def _lexical_search(
        self,
        query: str,
        scopes: Sequence[Scope],
        limit: int,
        valid_at: datetime | None,
        known_at: datetime | None,
        states: Sequence[str],
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
            reduced.text, scopes, limit, valid_at=valid_at, known_at=known_at,
            states=states)
        return list(hits), len(reduced.terms)

    @staticmethod
    def _believed_by(claim: Claim, known_at: datetime | None) -> bool:
        """Belief-time floor, enforced here rather than left to the store.

        A `states` set covering all three drops most of the store's liveness predicate,
        and a third-party store may drop the rest too - letting claims recorded *after*
        the asked instant leak into a historical answer. "What did we believe in March,
        including what we later retracted" must never include something we had not yet
        heard in March - that is knowledge from the future, and it is the one way a
        bitemporal query can lie. Re-applying the floor costs a comparison per
        candidate and makes the two flags orthogonal, which is what callers assume.

        Only the belief axis is re-checked. The valid-time floor is deliberately *not*
        mirrored here: the complete state set lifts the whole valid-time interval by
        design (see `store.state_predicate`), so re-imposing half of it in Python would
        make the filter mean something different depending on which layer answered.
        """
        if known_at is None:
            return True
        return _as_utc(claim.recorded_at) <= _as_utc(known_at)

    def _explain(self, claim: Claim, fusion: float, legs: _Legs, now: datetime,
                 weights: _Weights, anchor: str | None = None) -> Result:
        v = legs.vector.get(claim.id)
        lx = legs.lexical.get(claim.id)
        g = legs.graph.get(claim.id)
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
            # A path score is already an absolute [0, 1] relevance — `_extend` composes
            # factors that are each at most 1.0 — so unlike BM25 it needs no map onto the
            # unit interval and unlike cosine it needs no clamp.
            graph=(0.0 if g is None else g[1]) if legs.graph_active else None,
            w_vector=weights.vector,
            w_lexical=weights.lexical,
            w_graph=weights.graph,
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
            graph_rank=None if g is None else g[0],
            graph_score=None if g is None else g[1],
            fusion_score=fusion,
            recency=recency,
            confidence=claim.confidence,
            salience=claim.salience,
            # No cross-encoder in this tier. Left None so a future reranker's absence
            # is distinguishable from a reranker that scored zero.
            rerank_score=None,
            raw_score=final_score(fusion, **quality),
            final_score=score,
            intent=None if weights.intent is None else weights.intent.value,
            anchor=anchor,
        )
        return Result(claim=claim, score=score, explain=explain)


def _positions(hits: Sequence[tuple[str, float]]) -> dict[str, tuple[int, float]]:
    """Map item id -> (0-based rank, raw score), keeping the best rank on repeats."""
    out: dict[str, tuple[int, float]] = {}
    for rank, (item_id, score) in enumerate(hits):
        out.setdefault(item_id, (rank, score))
    return out
