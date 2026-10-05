"""Hybrid retrieval: BM25 and vectors, fused, rescored and explained.

A single vector top-k, which is what mem0-style retrieval does, fails in five routine
ways. Each leg or stage here exists to fix one of them.

1. **Exact tokens.** An embedding maps surface forms onto meaning, so a claim that
   literally contains `ERR_7734`, `v2.14.1` or an unusual surname can rank below claims
   that merely talk about errors. BM25 has the opposite bias, because a rare term carries
   a large IDF. The two legs fail on different inputs, which is why fusing them helps.
2. **Time.** Cosine similarity does not know whether a fact is current. Rescoring by a
   decay keyed to the predicate puts the current value first without deleting history.
3. **Absence.** Top-k always returns k things, even when nothing in the store answers
   the question. Scores are absolute retriever evidence rather than fused rank (see
   `scoring.normalized_score`), so "nothing here is relevant" is a number, and a caller
   acts on it with `min_score`.
4. **Text that was never a claim.** Extraction keeps facts and drops the rest of a turn,
   such as a decision's reason or a conditional constraint. `include_episodes=True` adds
   a second, weaker pair of legs over the raw turns; see `EpisodeResult` for why weaker.
5. **Answers that need two rows.** "Where is my manager's employer based" needs two
   claims, and the second shares no words with the question. The graph leg walks out of
   the entities the first two legs found. `retrieve/spread.py` explains why the seeds
   come from the answer rather than the query, and `retrieve/intent.py` decides which
   queries pay for the walk. It ships at `w_graph=0.0`; see `docs/BENCHMARKS.md`.

Retrieval is deterministic by default. No model runs on the read path unless one of the
two model stages below is configured, and identical inputs give an identical ordering,
ties included, so that a ranking regression can be bisected. Ties break on a content hash
rather than on a row id, because ids are minted per ingest and would make the order differ
between two stores holding the same data.

The two model stages each record what they did on the result:

- `search(query_rewrite=True)` with a `rewriter` configured makes one model call before
  retrieval, asking for other phrasings of the query and the date range it names. Each
  phrasing is retrieved by the deterministic pipeline and the lists are fused, so the
  model chooses what is searched for and never how anything is scored. `Memvara` turns
  this on by default when its `llm=` can chat. `SearchResults.rewrite` reports it.
- `search(ranked=True)` with a `read_selector` configured makes one model call per read,
  on the customer's own key, naming which of the reranked turns bear on the question. By
  default it sees the turns of the role the question asks about (`intent.routed_role`);
  `route_roles=False` gives it the whole reranked window. A plain read is unchanged.
  `SearchResults.selection` reports the outcome. See `memvara.select` and
  `HybridRetriever.search` for the order a ranked read takes.
"""

from __future__ import annotations

import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from functools import partial
from time import perf_counter
from typing import (
    TYPE_CHECKING, Any, Callable, ClassVar, Collection, Iterable, Literal, NamedTuple,
    Sequence, TypeVar, overload,
)

import numpy as np

from ..embed.base import Embedder, encode_queries
from ..filters import SearchFilter
from ..llm.base import Usage
from ..rerank import Reranker, rerank
from ..schema import PredicateRegistry
from ..select.base import (
    Candidate, Rewrite, Selection, Selector, SelectorBusy, SelectorRefused,
)
from ..select.stages import MAX_QUERIES, run_stage
from ..store.base import Store, bulk_claims, resolve_states
from ..telemetry import (
    RETRIEVAL_LATENCY_MS,
    RETRIEVAL_MODEL_FALLBACK,
    RETRIEVAL_MODEL_QUERY,
    RETRIEVAL_MODEL_REFUSED,
    RETRIEVAL_OBSERVATION_RANK_CORR,
    RETRIEVAL_QUALITY_FACTOR,
    RETRIEVAL_QUERY,
    RETRIEVAL_RESULTS,
    RETRIEVAL_SELECT_MS,
    RETRIEVAL_TOKENS_IN,
    RETRIEVAL_TOKENS_OUT,
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
    SearchResults,
    TimeWindow,
    owner_key,
    time_axes,
    time_window,
    utcnow,
)
from .anchor import PATH, SUBJECT, anchor_of, query_tokens
from .analyze import analyze
from .fusion import reciprocal_rank_fusion
from .intent import Intent, classify, routed_role
from .compose import names_derived
from .intent import is_comparison, is_relational, observed_refs
from .intent import weights as intent_weights
from .shadow import shadowed
from .scoring import (
    final_score,
    lexical_relevance,
    normalized_score,
    ranking_quality,
    recency_factor,
    relevance,
    vector_relevance,
)
from .spread import rank_paths, seed_keys
from .temporal import TEMPORAL, anchor_for, rank as rank_by_time
from .traverse import GraphTraverser

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from ..entities import EntityRegistry
    from ..select.stages import QueryRewriter

# Retriever names. Shared between the fusion weights and the `Explanation` fields so
# the two cannot drift apart under a rename.
VECTOR = "vector"
LEXICAL = "lexical"
GRAPH = "graph"


# There is deliberately no default relevance floor. The right floor moves with corpus
# size; `calibrate.py` has the measurement and `calibrate_min_score`, which derives one
# from a deployment's own probes.


def _as_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


class DegradedRetrievalWarning(UserWarning):
    """A configured retrieval leg could not run against this store.

    Raised once per `HybridRetriever`, not once per query, because a store that cannot
    traverse cannot do so for the whole process. Search still runs on the remaining legs;
    the warning exists so that the loss is not silent. The case it covers is a store whose
    `adjacent` method exists, so a `getattr` check passes, but raises
    `NotImplementedError` when called.
    """


@dataclass(slots=True)
class EpisodeResult:
    """A raw conversation turn that matched, and how well.

    This is deliberately not a `Result`. A `Claim` has been extracted, normalized,
    reconciled and retired when superseded; an episode is something someone said once.
    Treating one as the other is how "I'm thinking of moving to Lisbon" becomes a fact
    about where the user lives, so callers tell them apart with `isinstance`, or with
    `kind` after serialization.

    The quality fields on `explain` sit at 1.0, meaning "not applicable". Recency,
    confidence and salience are properties of an extracted claim, and an episode has none.
    `score` is therefore retriever evidence alone, times `w_episode`.
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
            # Otherwise a turn found only by proximity would show as `no-retriever`.
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

    Resolved once, at the top of `search`, and passed down. Code below must read weights
    from here and never from `self`: with intent weighting on, the two differ, and mixing
    them makes the relevance average divide by a leg that fusion never ran.
    """

    vector: float
    lexical: float
    graph: float
    temporal: float
    #: `None` when `intent_weighting` is off: nothing classified the query.
    intent: Intent | None


@dataclass(frozen=True, slots=True)
class _Legs:
    """One search's retriever output, in the shape scoring needs it.

    `*_active` says whether the leg returned anything. A leg that returned nothing is
    dropped from the relevance average; a leg that ran but did not rank a given claim
    gives that claim a real zero.
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

    `DegradedRetrievalWarning` means the backend cannot traverse. This warning means the
    data has no chains: no live claim's object is another claim's subject, so a walk
    could only return what the lookup legs already found. It is fixed by writing such
    facts, and stops applying as soon as one exists. It subclasses the parent so that an
    existing `filterwarnings` on the parent still catches it. Running the leg on such a
    store costs ranking quality; see docs/BENCHMARKS.md, "The graph leg, and what it
    costs on the corpora above".
    """


#: How many searches reuse a tenant's connectivity reading before it is measured again.
#:
#: A counter rather than a clock, so the same sequence of searches re-measures at the
#: same points on every run. Staleness can only switch the graph leg off, never on: a
#: store that gains joins stays gated for at most this many searches, which is the
#: shipped `w_graph=0.0` behaviour. The cost is in docs/INTERNALS.md, under the graph
#: leg's connectivity gate.
GATE_RECHECK_EVERY = 256

#: Threads a retriever keeps for running one leg beside another; see
#: `HybridRetriever._beside`. When every thread is busy, a pass runs its legs one after
#: another on the calling thread.
_LEG_THREADS = MAX_QUERIES

_T = TypeVar("_T")


def known_memory_types(
        memory_types: Sequence[MemoryType | str] | None) -> list[MemoryType] | None:
    """`memory_types` as `MemoryType` members, refusing a name that is not one.

    Without the refusal, a misspelled name would match no claim and the search would
    return nothing, which looks the same as a store with nothing relevant. The error
    message uses the same words as `remember()` (`core._as_memory_type`).

        >>> known_memory_types(["semantic"])
        [<MemoryType.SEMANTIC: 'semantic'>]
        >>> known_memory_types(["procedurel"])
        Traceback (most recent call last):
        ...
        ValueError: memory_type must be one of episodic, semantic, procedural, not 'procedurel'
    """
    if memory_types is None:
        return None
    known = []
    for value in memory_types:
        try:
            known.append(MemoryType(value))
        except ValueError:
            raise ValueError(
                "memory_type must be one of "
                + ", ".join(t.value for t in MemoryType) + f", not {value!r}") from None
    return known

#: Ranked-read outcomes that skip the final reranker pass. `applied` already reranked
#: its turns inside the ranked stage, and `disabled` is the operator's load-shedding
#: switch, which must spend nothing on the cross-encoder. Every other outcome serves the
#: plain read, including its reranker pass.
_NO_RERANK = frozenset({"applied", "disabled"})


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
        episode_score_floor: float = 0.0,
        w_temporal: float = 0.0,
        w_graph: float = 0.0,
        derived_terms: "Collection[str]" = (),
        graph_seeds: int = 5,
        graph_depth: int = 2,
        traverser: "GraphTraverser | None" = None,
        intent_weighting: bool = True,
        reranker: "Reranker | None" = None,
        rerank_top_n: int = 20,
        rerank_ranked_only: bool = False,
        selector: "Selector | None" = None,
        route_roles: bool = True,
        rewriter: "QueryRewriter | None" = None,
        rewrite_enabled: bool = True,
        telemetry: Recorder | None = None,
        entities: "EntityRegistry | None" = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.registry = registry
        #: Per tenant: (searches since the last connectivity reading, does anything in the
        #: tenant chain). Written only when `w_graph > 0`. See `_store_has_joins`.
        self._joins: dict[str, tuple[int, bool]] = {}
        self._warned_unjoined = False
        #: Metrics sink, or `None` (the default). Every emission is guarded, and the
        #: values that cost something to compute are computed inside the guard.
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
        # Episode scores are multiplied by `w_episode` (0.5), so a raw turn must be twice
        # as convincing as a claim to outrank it. A turn often contains the query's words
        # verbatim while the claim extracted from it shares none, so without the discount
        # turns would push out curated facts. `max_episodes` caps how many turns a search
        # returns, so one long conversation cannot crowd out everything else.
        self.w_episode = w_episode
        self.max_episodes = max_episodes
        #: Keep an episode only while its score is at least this fraction of the best
        #: episode's score. At 0.0, the default, `max_episodes` turns are kept regardless.
        #: A nonzero floor makes `max_episodes` a ceiling rather than a fixed count, so a
        #: question that needs more evidence gets more turns and one that needs less gets
        #: fewer. The measurement behind 0.55 is in the CHANGELOG entry that added the
        #: option. The floor is relative, unlike `min_score`, so a query whose results all
        #: score low still keeps its best turns.
        #:
        #: Values outside [0, 1] raise rather than being clamped. Above 1.0 every query
        #: would silently return exactly one episode (a caller passing 55 for 55% would hit
        #: this), and below 0.0 would silently disable the floor.
        if not 0.0 <= episode_score_floor <= 1.0:
            raise ValueError(
                f"episode_score_floor must be between 0.0 and 1.0, got "
                f"{episode_score_floor!r}. It is a fraction of the best result's score, "
                f"not a percentage: 0.55 keeps results scoring at least 55% of the top "
                f"one, and 0.0 disables the floor."
            )
        self.episode_score_floor = episode_score_floor
        #: Weight of the episode temporal leg, which ranks turns by closeness to the
        #: instant asked about. At 0.0, the shipped default (see docs/BENCHMARKS.md), the
        #: leg does not run and `Explanation.temporal_rank` stays `None`. It applies to
        #: episodes only, because a claim's predicate-keyed half-life is a better time
        #: signal than raw proximity, so with `include_episodes=False` it has no effect.
        self.w_temporal = w_temporal
        #: Weight of the graph leg in fusion and in the relevance average. At 0.0, the
        #: shipped default, no walk runs and `Explanation.graph_rank` stays `None`. The
        #: leg helps multi-hop questions and slightly hurts others (see
        #: docs/BENCHMARKS.md); `intent_weighting` is what turns it on only for the query
        #: shapes that gain.
        self.w_graph = w_graph
        #: Relation terms that name a chain of stored predicates rather than one
        #: predicate, such as "grandfather" for `father` then `father`. They are acquired
        #: once from a model by `retrieve/compose.acquire()` and passed in, so that search
        #: itself never calls a model. Empty by default.
        self.derived_terms = frozenset(derived_terms)
        #: How many entity keys the walk starts from, taken from the head of the fused
        #: vector and lexical list. The frontier width is what each hop costs. See
        #: `spread.seed_keys`.
        self.graph_seeds = graph_seeds
        #: How many hops the walk goes. Two, because `GraphTraverser` multiplies path
        #: scores, so a third hop is damped to at most 0.56 before edge quality and rarely
        #: survives fusion, while still costing a frontier expansion. See
        #: `bench/multihop.py`.
        self.graph_depth = graph_depth
        #: The traversal engine, or `None`, in which case the graph leg never runs.
        #: `Memvara` builds one. A `HybridRetriever` built directly against a third-party
        #: `Store` without one runs the two lookup legs only, because `Store.adjacent` is
        #: optional in the protocol.
        self.traverser = traverser
        #: Whether leg weights are scaled per query shape (`intent.MULTIPLIERS`). At the
        #: shipped weights this changes nothing, because only the graph column differs from
        #: 1.0 and `w_graph` ships at 0.0. `False` runs every query at the configured
        #: weights and leaves `Explanation.intent` unset.
        self.intent_weighting = intent_weighting
        # Set the first time a walk raises `NotImplementedError`, so a store that cannot
        # traverse is tried once rather than on every query.
        self._graph_unsupported = False
        #: Optional reranker over the head of the fused list. With `None`, the default,
        #: the stage does not run and imports nothing, which keeps the shipped
        #: configuration offline. See `memvara.rerank`. When configured, the retriever
        #: ranks `rerank_top_n` deep instead of `k`, reranks, and then cuts to `k`, so
        #: candidates just past `k` can be promoted into the results.
        self.reranker = reranker
        self.rerank_top_n = rerank_top_n
        #: When `True`, a plain read (`ranked=False`) skips the reranker entirely and is
        #: gathered at depth `k`, while a ranked read still reranks. The default `False`
        #: reranks every read. The hosted service sets `True`, because it configures a
        #: cross-encoder on every retriever for ranked reads and plain reads should not pay
        #: for it.
        self.rerank_ranked_only = rerank_ranked_only
        #: Model stage that names which reranked turns bear on the question, or `None`
        #: (the default). With `None`, `ranked=True` is served unranked with outcome
        #: `unconfigured`. See `memvara.select` and `search`'s `ranked` argument.
        self.selector = selector
        #: Whether a ranked read gives the selector only one role's turns (the user's,
        #: unless the question asks what the assistant said; see `intent.routed_role`).
        #: `True`, the default, assumes the `assistant` role holds a model's long and
        #: rarely relevant turns. In a store where both roles are people, routing drops
        #: one person's turns, so set `read_route_roles=False` there. The CHANGELOG entry
        #: that added the option has the measurement.
        self.route_roles = route_roles
        #: Model stage that suggests other phrasings of the query and the date range it
        #: means, or `None`. `Memvara` builds one when its `llm=` can chat. With `None`,
        #: `search(query_rewrite=True)` reports `unconfigured` and makes no call. See
        #: `memvara.select.stages`.
        self.rewriter = rewriter
        #: The `query_rewrite` switch. When `False`, a read that asks for a rewrite
        #: reports `disabled` and makes no call, whatever `rewriter` holds.
        self.rewrite_enabled = rewrite_enabled
        #: The owner's learned entity aliases, or `None`. Anchoring reads it so that a
        #: question saying "Big Blue" matches a claim filed under `ibm`. `Memvara` passes
        #: the writer's live registry, so an alias learned in this process is used by the
        #: next search.
        self.entities = entities
        #: Runs a rewritten read's alternative phrasings beside the original query. Created
        #: on first use and kept, because each thread holds its own SQLite read connection.
        self._phrasings: ThreadPoolExecutor | None = None
        self._phrasings_lock = threading.Lock()
        #: Runs a search's vector leg beside its lexical leg; see `_beside`. Created on
        #: first use and kept. `_legs_free` counts idle threads, so a search that finds
        #: none runs the leg itself instead of waiting.
        self._legs: ThreadPoolExecutor | None = None
        self._legs_lock = threading.Lock()
        self._legs_free = threading.Semaphore(_LEG_THREADS)
        #: Query vectors for the pass running on this thread, so the claim and episode
        #: legs embed a query only once. Cleared at the end of each pass.
        self._pass = threading.local()

    # Overloads, because `include_episodes` decides the return type: callers of the
    # common form get `list[Result]` instead of a union they would have to narrow. See
    # `Memvara.search`.
    @overload
    def search(
        self, query: str, scope: Scope, *, k: int = ...,
        as_of: datetime | None = ..., valid_at: datetime | None = ...,
        known_at: datetime | None = ..., states: Collection[str] | None = ...,
        valid_during: Sequence[datetime] | None = ...,
        include_invalidated: bool | None = ...,
        memory_types: Sequence[MemoryType] | None = ..., min_score: float = ...,
        anchored: bool = ..., include_episodes: Literal[False] = ...,
        now: datetime | None = ..., ranked: bool = ...,
        query_rewrite: bool = ...,
        where: SearchFilter | None = ...,
    ) -> list[Result]: ...

    @overload
    def search(
        self, query: str, scope: Scope, *, k: int = ...,
        as_of: datetime | None = ..., valid_at: datetime | None = ...,
        known_at: datetime | None = ..., states: Collection[str] | None = ...,
        valid_during: Sequence[datetime] | None = ...,
        include_invalidated: bool | None = ...,
        memory_types: Sequence[MemoryType] | None = ..., min_score: float = ...,
        anchored: bool = ..., include_episodes: Literal[True],
        now: datetime | None = ..., ranked: bool = ...,
        query_rewrite: bool = ...,
        where: SearchFilter | None = ...,
    ) -> list[Retrieved]: ...

    @overload
    def search(
        self, query: str, scope: Scope, *, k: int = ...,
        as_of: datetime | None = ..., valid_at: datetime | None = ...,
        known_at: datetime | None = ..., states: Collection[str] | None = ...,
        valid_during: Sequence[datetime] | None = ...,
        include_invalidated: bool | None = ...,
        memory_types: Sequence[MemoryType] | None = ..., min_score: float = ...,
        anchored: bool = ..., include_episodes: bool, now: datetime | None = ...,
        ranked: bool = ..., query_rewrite: bool = ...,
        where: SearchFilter | None = ...,
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
        valid_during: Sequence[datetime] | None = None,
        include_invalidated: bool | None = None,
        memory_types: Sequence[MemoryType] | None = None,
        min_score: float = 0.0,
        anchored: bool = False,
        include_episodes: bool = False,
        now: datetime | None = None,
        ranked: bool = False,
        query_rewrite: bool = False,
        where: SearchFilter | None = None,
    ) -> list[Any]:
        """Return the top `k` results for `query`, each with a populated `Explanation`.

        There are two time axes, each defaulting to now. `known_at` is belief time: the
        result is what we believed at that instant, including claims retracted since.
        `valid_at` is world time: what was true then, judged with everything known today,
        so a correction learned in August about June is found. `as_of` sets both and
        cannot be combined with either. See `memvara.types.time_axes`.

        `valid_during=(start, end)` asks about a period of world time: a claim is
        returned if it was true at any moment from `start` to `end`, both included. The
        claim searches use the whole period; the turns and the graph leg use its end as
        `valid_at`. It cannot be combined with `valid_at` or `as_of`; `known_at` still
        applies. See `memvara.types.time_window`.

        `states` is any non-empty subset of `("live", "ended", "retired")`.
        `include_invalidated` is a supported two-valued alias: `False` means `["live"]`
        and `True` means all three. Passing both raises. Asking for all three lifts the
        end-of-life filters, which suits auditing and not answering. Claims recorded
        after `known_at` are never returned whatever the subset; see
        `store.state_predicate`.

        `min_score` drops results below a normalized relevance (see
        `scoring.normalized_score`). The default 0.0 drops nothing. The right value
        depends on corpus size and embedder, so measure it with
        `calibrate.calibrate_min_score`.

        `anchored=True` drops claims the query names at neither end. Every result carries
        `Explanation.anchor`: which end of the claim the query named, `"path"` for a claim
        the graph leg reached from a named one, or `None`. With it, a question about an
        entity the store has never heard of returns nothing instead of the nearest row
        about someone else. It does not affect episodes, which have no subject. See
        `retrieve/anchor.py`.

        `include_episodes=True` also searches the raw turns and returns them as
        `EpisodeResult`, so a caller can tell a fact from something someone said. It is
        off by default so that existing callers keep getting facts only. Passing
        `memory_types`, a claim-only filter, disables the episode leg.

        The graph leg is configured on the constructor (`w_graph`), not per call, so one
        store does not rank the same query two ways. `Explanation.graph_rank` and
        `Explanation.intent` show what ran.

        The return type follows `include_episodes`: `list[Result]` unless episodes were
        asked for. The implementation is annotated `list[Any]` only because an
        overloaded implementation must cover every variant.

        `ranked=True` sends the reranked turns to the configured `read_selector`, which
        names the ones that bear on `query`. Those are returned first, whole, with
        `Explanation.selected` and `.span` set; see `memvara.select`. It requires
        `include_episodes=True` and no `memory_types`, and raises `ValueError` otherwise.
        The result is always a `SearchResults` whose `.selection` is `None` on a plain read
        and, on a ranked one, has outcome `applied`, `fallback`, `unconfigured` (no
        selector), `disabled` (operator switch) or `key_rejected` (provider rejected the
        key). Every outcome except `applied` returns the plain order.

        `query_rewrite=True` asks the configured `rewriter` (`memvara.select.stages`), in
        one model call before retrieval, for up to three other phrasings of `query` and the
        date range it refers to. The query and each phrasing are searched with the same
        arguments and the lists are fused by reciprocal rank, so a row found by several
        phrasings rises. A row keeps the `score` and `Explanation` from the first list that
        found it, the original query's first. On a ranked read only the original query goes
        to the selector, and the turns it kept stay in front. The date range becomes this
        read's `valid_during`, from the start of its first day to the end of its last day
        or now, whichever is earlier. The turns and the graph leg use its end as
        `valid_at` when that end is in the past. A caller's own `valid_at`, `as_of` or
        `valid_during` always wins, and a range starting after today is ignored.
        `.rewrite` reports the outcome, with the same five values as `ranked`; every
        outcome except `applied` serves the plain read. It defaults to `False` here and to `True` on `Memvara.search`. A plain
        read with `k <= 0` returns nothing, makes no call and reports no rewrite.

        `where` is the caller's metadata and file-path filter (`memvara.filters`), or
        `None`. It is passed to every store method that limits rows, so a filtered search
        with `k=5` returns five matches whenever five exist. The graph leg does not run on
        a filtered search; see `_graph_search`.
        """
        memory_types = known_memory_types(memory_types)
        if ranked and (not include_episodes or memory_types is not None):
            raise ValueError(
                "ranked=True needs turns to rank: it requires include_episodes=True and "
                "no memory_types (a type filter skips the episode leg entirely, so a "
                "ranked call would hand the selector nothing)."
            )
        # Resolved once and passed down, so no inner call can disagree about the instant
        # or the states. Done before the rewrite, so a call that will raise costs nothing.
        window = time_window(valid_during, as_of=as_of, valid_at=valid_at)
        valid_at, known_at = time_axes(as_of, valid_at, known_at)
        if window is not None:
            # Everything but the claim searches reads the window's end as its instant.
            valid_at = window[1]
        wanted_states = resolve_states(states, include_invalidated)
        once = partial(self._search_once, scope=scope, k=k, known_at=known_at,
                       wanted_states=wanted_states, memory_types=memory_types,
                       min_score=min_score, anchored=anchored,
                       include_episodes=include_episodes, where=where)
        # A read that returns nothing by construction is not worth a model call.
        if not query_rewrite or (k <= 0 and not ranked):
            return once(query, valid_at=valid_at, window=window, now=now, ranked=ranked)
        rec = self.telemetry
        # Started before the rewrite, because the caller waited through it too.
        t0 = perf_counter() if rec is not None else 0.0
        # One instant for the whole read: the model's "today", the decay clock, and the
        # cut-off a date range must end before.
        asked = _as_utc(now) if now is not None else utcnow()
        rewrite = self._rewrite(query, asked)
        alternatives: tuple[str, ...] = ()
        if rewrite.outcome == "applied":
            alternatives = rewrite.queries
            # `valid_at` is already `as_of` and a window's end folded in, so `None`
            # means the caller named no time at all.
            if (rewrite.date_from is not None and rewrite.date_to is not None
                    and valid_at is None):
                start = datetime(rewrite.date_from.year, rewrite.date_from.month,
                                 rewrite.date_from.day, tzinfo=timezone.utc)
                end = datetime(rewrite.date_to.year, rewrite.date_to.month,
                               rewrite.date_to.day, 23, 59, 59, tzinfo=timezone.utc)
                if start <= asked:
                    # The claim searches use the whole range, up to now at the
                    # latest; the rest use its end, if that is in the past.
                    window = (start, min(end, asked))
                    rewrite = replace(rewrite, valid_during=window)
                if end < asked:
                    valid_at = end
                    rewrite = replace(rewrite, valid_at=end)
        if not alternatives:
            main = once(query, valid_at=valid_at, window=window, now=asked, ranked=ranked,
                        observe=False)
            hits: list[Retrieved] = list(main)
        else:
            # Every phrasing in one call: an embedder behind a network pays a round trip
            # per call, not per text.
            vectors = encode_queries(self.embedder, [query, *alternatives])
            pending = [
                self._phrasing_pool().submit(
                    once, q, valid_at=valid_at, window=window, now=asked, ranked=False,
                    observe=False, rerank_final=False, qvec=vec)
                for q, vec in zip(alternatives, vectors[1:])]
            main = once(query, valid_at=valid_at, window=window, now=asked, ranked=ranked,
                        observe=False, rerank_final=False, qvec=vectors[0])
            kept, fused = self._fuse(main, [f.result() for f in pending])
            # The reranker runs once, on the fused list, rather than once per phrasing,
            # and is skipped for the outcomes in `_NO_RERANK`.
            skip = main.selection is not None and main.selection.outcome in _NO_RERANK
            reranker = (None if skip
                        else None if self.rerank_ranked_only else self.reranker)
            if reranker is not None:
                fused = rerank(reranker, query, fused, top_n=self.rerank_top_n)
            hits = [*kept, *fused[:k]]
        if rec is not None:
            self._observe(rec, query, hits, (perf_counter() - t0) * 1000.0)
        return SearchResults(hits, selection=main.selection, rewrite=rewrite)

    def _rewrite(self, query: str, asked: datetime) -> Rewrite:
        """The rewrite outcome for `query`, counted. See `stages.run_stage`."""
        return run_stage("rewrite", self.rewriter, self.rewrite_enabled, Rewrite,
                         lambda stage, usage: stage.rewrite(query, today=asked.date(),
                                                            usage=usage),
                         self.telemetry)

    def _phrasing_pool(self) -> ThreadPoolExecutor:
        with self._phrasings_lock:
            if self._phrasings is None:
                self._phrasings = ThreadPoolExecutor(
                    max_workers=MAX_QUERIES, thread_name_prefix="memvara-phrasing")
            return self._phrasings

    def _beside(self, query: str, leg: Callable[[], _T]) -> Callable[[], _T]:
        """Start the vector leg `leg` on another thread and return what collects its
        result.

        Each leg spends most of its time in SQLite or in a matrix product, both of which
        release the GIL, so running the two together costs the longer of the two rather
        than their sum. The leg runs on the calling thread instead when the store does not
        offer `_parallel_reads` (a private `SQLiteStore` method, read with a guarded
        `getattr`) or when every pool thread is busy, so a saturated pool never makes a
        search wait.

        `query` is embedded first, on the calling thread, and the vectors travel with the
        leg, so the embedder is only ever called from the thread that called `search()`.
        """
        can = getattr(self.store, "_parallel_reads", None)
        if can is not None and can():
            vector = self._query_vector(query)
            if self._legs_free.acquire(blocking=False):
                passing = getattr(self._pass, "vectors", None)
                carried = passing if passing is not None else {query: vector}

                def run() -> _T:
                    self._pass.vectors = carried
                    try:
                        return leg()
                    finally:
                        self._pass.vectors = None
                        self._legs_free.release()

                with self._legs_lock:
                    if self._legs is None:
                        self._legs = ThreadPoolExecutor(
                            max_workers=_LEG_THREADS, thread_name_prefix="memvara-leg")
                    return self._legs.submit(run).result
        done = leg()
        return lambda: done

    @staticmethod
    def _fuse(main: SearchResults, others: Sequence[SearchResults],
              ) -> tuple[list[Retrieved], list[Retrieved]]:
        """The original query's rows and each alternative's, fused by rank.

        Returns `(kept, fused)`: the turns a ranked read kept, and every other row in
        fused order, not yet cut to `k`, so that a reranker can still promote from below
        the cut.

        Turns a ranked read kept stay first and are not fused, so an alternative phrasing
        cannot push a turn the model chose below one it never saw. Everything else is
        ranked by reciprocal-rank fusion. Ties go to the row seen first (main list, then
        each alternative in order), which is deterministic across stores, unlike the id
        tiebreak inside `reciprocal_rank_fusion`.
        """
        def key(r: Retrieved) -> str:
            if isinstance(r, EpisodeResult):
                return f"{EPISODE}:{r.episode.id}"
            return f"{CLAIM}:{r.claim.id}"

        kept: list[Retrieved] = [
            r for r in main if isinstance(r, EpisodeResult) and r.explain.selected is True]
        taken = {key(r) for r in kept}
        first: dict[str, tuple[int, Retrieved]] = {}
        rankings: dict[str, list[tuple[str, float]]] = {}
        for i, rows in enumerate([main, *others]):
            ranking = []
            for r in rows:
                name = key(r)
                if name in taken:
                    continue
                ranking.append((name, r.score))
                first.setdefault(name, (len(first), r))
            rankings[str(i)] = ranking
        fused = reciprocal_rank_fusion(rankings)
        order = sorted(fused, key=lambda name: (-fused[name], first[name][0]))
        return kept, [first[name][1] for name in order]

    def _search_once(
        self, query: str, *, scope: Scope, k: int, valid_at: datetime | None,
        known_at: datetime | None, wanted_states: tuple[str, ...],
        window: TimeWindow | None = None,
        memory_types: Sequence[MemoryType] | None, min_score: float, anchored: bool,
        include_episodes: bool, now: datetime | None, ranked: bool,
        observe: bool = True, rerank_final: bool = True, qvec: Any = None,
        where: SearchFilter | None = None,
    ) -> SearchResults:
        """One retrieval of one query: everything `search` does except the rewrite.

        The time axes, the window and the states arrive resolved; with a window,
        `valid_at` is its end. A rewritten read passes `observe=False` and
        `rerank_final=False`, because it records telemetry and reranks once over the fused
        list. `qvec` is the query's vector when the caller already has it.
        """
        self._pass.vectors = {} if qvec is None else {
            query: np.asarray(qvec, dtype=np.float32)}
        try:
            return self._retrieve(query, scope=scope, k=k, valid_at=valid_at,
                                  known_at=known_at, wanted_states=wanted_states,
                                  window=window,
                                  memory_types=memory_types, min_score=min_score,
                                  anchored=anchored, include_episodes=include_episodes,
                                  now=now, ranked=ranked, observe=observe,
                                  rerank_final=rerank_final, where=where)
        finally:
            self._pass.vectors = None

    def _query_vector(self, query: str) -> np.ndarray:
        """`query` embedded, once per pass: both vector legs call this."""
        cache = getattr(self._pass, "vectors", None)
        if cache is not None and query in cache:
            return cache[query]
        vec = np.asarray(encode_queries(self.embedder, [query])[0], dtype=np.float32)
        if cache is not None:
            cache[query] = vec
        return vec

    def _retrieve(
        self, query: str, *, scope: Scope, k: int, valid_at: datetime | None,
        known_at: datetime | None, wanted_states: tuple[str, ...],
        window: TimeWindow | None = None,
        memory_types: Sequence[MemoryType] | None, min_score: float, anchored: bool,
        include_episodes: bool, now: datetime | None, ranked: bool, observe: bool,
        rerank_final: bool, where: SearchFilter | None = None,
    ) -> SearchResults:
        # Only a plain read may return early. A ranked read must always report an
        # outcome in `.selection`, even with `k <= 0`.
        if k <= 0 and not ranked:
            return SearchResults()
        rec = self.telemetry
        t0 = perf_counter() if rec is not None else 0.0

        # A scope also sees every scope above it, but never a sibling:
        # `ancestors()` walks strictly upward and the store matches each scope exactly,
        # so no query reaches another session, user or tenant.
        scopes = scope.ancestors()

        # Decay is measured from `known_at`, the belief clock, because recency means
        # "how long ago had we last heard this, as of the question". `now` only replaces
        # the wall-clock read when `known_at` is unset, so two identical searches score
        # identically. See docs/BENCHMARKS.md, "LOCOMO and LongMemEval".
        now = _as_utc(known_at) if known_at is not None else (
            _as_utc(now) if now is not None else utcnow())

        # `selector_ranked` is a ranked call that can actually run. A ranked call with no
        # selector is served exactly like a plain read, so the reranker gate reads
        # `selector_ranked` rather than `ranked`.
        selector_ranked = ranked and self.selector is not None
        reranker_active = (None if (self.rerank_ranked_only and not selector_ranked)
                           else self.reranker)

        # `depth` is how deep the pipeline ranks before cutting to `k`. It is `k` unless a
        # reranker or a selector will run, since those need candidates below the cut to
        # promote. `limit` over-fetches from each leg, because fusion can only rank what
        # every leg returned.
        depth = (k if (not selector_ranked and reranker_active is None)
                else max(k, self.rerank_top_n))

        limit = max(depth * self.candidate_multiplier, depth)
        wanted = set(memory_types) if memory_types is not None else None

        # A window is a dated read even when it reaches now, which leaves `valid_at` unset.
        weights = self._weights(query, timed=(valid_at is not None or known_at is not None
                                              or window is not None))

        results, saturated = self._gather(
            query, scope, limit, valid_at, known_at, wanted_states, wanted, now,
            min_score, anchored, weights, where, window)

        # `memory_types` and `anchored` filter after the legs have been cut to `limit`,
        # so a narrow filter can come back short while matches sit just past the cut.
        # When a leg came back full, retry once wider. `min_score` gets no retry, because
        # deeper candidates have less evidence, not more.
        if (wanted is not None or anchored) and saturated and len(results) < depth:
            results, _ = self._gather(
                query, scope, limit * self.filter_retry_multiplier, valid_at, known_at,
                wanted_states, wanted, now, min_score, anchored, weights, where, window)

        claims: list[Retrieved] = list(self._rank(results, depth))
        selection: Selection | None = None
        hits: list[Retrieved]

        if include_episodes and wanted is None:
            if selector_ranked:
                assert self.selector is not None  # narrows for mypy; see `selector_ranked`
                # On a ranked call the episode cap is `rerank_top_n` rather than
                # `max_episodes`; this pool is what the selector chooses from.
                episodes = self._episodes(query, scopes, limit, valid_at, known_at,
                                          min_score, weights, now, where,
                                          cap=self.rerank_top_n)
                selection, kept_turns, tail = self._run_ranked_stage(
                    rec, self.selector, query, episodes, now)
                if selection.outcome != "applied":
                    # Every outcome but `applied` serves the plain read (INTERNALS,
                    # invariant 1). The pool above is wider than a plain read's, so the
                    # plain read runs again rather than reusing it. It is timed from `t0`
                    # because the caller waited through the failed stage too, and calls
                    # `_retrieve` so the cached query vector is reused.
                    plain = list(self._retrieve(
                        query, scope=scope, k=k, valid_at=valid_at, known_at=known_at,
                        wanted_states=wanted_states, memory_types=memory_types,
                        min_score=min_score, anchored=anchored,
                        include_episodes=include_episodes, now=now, ranked=False,
                        observe=False,
                        rerank_final=rerank_final and selection.outcome not in _NO_RERANK,
                        where=where, window=window))
                    if rec is not None and observe:
                        self._observe(rec, query, plain, (perf_counter() - t0) * 1000.0)
                    return SearchResults(plain, selection=selection)
                merged = self._interleave(claims, tail, depth)[:k]
                hits = [*kept_turns, *merged]
            else:
                if ranked:
                    # `unconfigured` runs exactly the plain read below.
                    if rec is not None:
                        rec.counter(RETRIEVAL_MODEL_REFUSED, reason="unconfigured")
                    selection = Selection(outcome="unconfigured", candidates=0)
                episodes = self._episodes(query, scopes, limit, valid_at, known_at,
                                          min_score, weights, now, where,
                                          cap=self.max_episodes)
                hits = self._interleave(claims, episodes, depth)
        else:
            hits = claims

        if reranker_active is not None and not selector_ranked and rerank_final:
            # Last, so the reranker reorders the head of the finished ranking and cannot
            # undo the per-slot diversity pass. A real ranked call skips it, because
            # `_run_ranked_stage` already reranked the turns.
            hits = rerank(reranker_active, query, hits, top_n=self.rerank_top_n)[:k]
        if rec is not None and observe:
            self._observe(rec, query, hits, (perf_counter() - t0) * 1000.0)
        return SearchResults(hits, selection=selection)

    # -- internals -----------------------------------------------------------

    def _weights(self, query: str, *, timed: bool) -> _Weights:
        """The configured weights, gated and scaled by what kind of question this is.

        Called once per search, so every part of one search agrees on the intent.

        `timed` means the caller passed an instant or a window, which states a temporal
        intent outright, so it overrides what `classify` reads from the words.
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
            # A question can be about both a chain and an instant, such as "where was
            # Alice's employer based in 2019". The TEMPORAL row would switch the graph
            # leg off, so the graph weight is restored here while the temporal row still
            # sets the other three legs. Comparisons ("whose grandfather was born
            # earlier, A or B") stay excluded, as everywhere the walk is opened.
            # `Explanation.intent` still reports `temporal`.
            graph = self.w_graph
        return _Weights(vector, lexical, graph, temporal, intent)

    def _observe(self, rec: Recorder, query: str, results: Sequence[Retrieved],
                 elapsed_ms: float) -> None:
        """Emit the aggregate view of one search.

        `Explanation` already covers a single call. This emits what only many searches
        can show. The recorder is passed in, rather than read from `self.telemetry`, so
        that the caller's `None` check is visible in the signature.
        """
        # Sliced by script, as the write gate is, so reads from a script whose writes
        # are being dropped show up.
        rec.counter(RETRIEVAL_QUERY, script=script_of(query))
        rec.gauge(RETRIEVAL_RESULTS, float(len(results)))
        rec.timing(RETRIEVAL_LATENCY_MS, elapsed_ms)

        counts: list[float] = []
        for r in results:
            if not isinstance(r, Result):
                # An episode has no quality fields or observation count to report.
                continue
            counts.append(float(r.claim.observation_count))
            # The same factor the ranking used. A value above 1.0 means quality is
            # promoting a result past its evidence.
            rec.gauge(RETRIEVAL_QUALITY_FACTOR, ranking_quality(
                recency=r.explain.recency,
                confidence=r.explain.confidence,
                salience=r.explain.salience,
                w_recency=self.w_recency,
                w_confidence=self.w_confidence,
                w_salience=self.w_salience,
            ))

        correlation = rank_correlation(counts)
        if correlation is not None:
            # Should be positive: a fact restated many times should rank above one
            # mentioned once. A negative trend means reinforcement is being lost.
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
        where: SearchFilter | None = None,
        window: TimeWindow | None = None,
    ) -> tuple[list[Result], bool]:
        """Run the legs at `limit` and return the surviving results, unsorted.

        The second element is `True` when either lookup leg returned `limit` hits, which
        is the only case where a wider retry could find more. The graph leg is not
        counted, because its size is set by its own beam and depth, not by `limit`.

        The two lookup legs search for the query; the graph leg walks out of what they
        found. See `_graph_search`.
        """
        scopes = scope.ancestors()
        vector = self._beside(query, partial(
            self._vector_search, query, scopes, limit, valid_at, known_at, states, where,
            window))
        lexical_hits, lexical_terms = self._lexical_search(
            query, scopes, limit, valid_at, known_at, states, where, window)
        vector_hits = vector()

        fused = reciprocal_rank_fusion(
            {VECTOR: vector_hits, LEXICAL: lexical_hits},
            k=self.rrf_k,
            weights={VECTOR: weights.vector, LEXICAL: weights.lexical},
        )
        saturated = len(vector_hits) >= limit or len(lexical_hits) >= limit
        if not fused:
            # Nothing to seed a walk from either.
            return [], saturated

        # Load every fused candidate in one round trip, not one query per candidate.
        claims = bulk_claims(self.store, list(fused))

        # A second chance for the graph leg. `classify` only counts predicates the
        # registry declared, and a predicate written through `remember()` never is. So
        # also count the predicates of the candidates just loaded: if the question names
        # two of them, or a derived term such as "grandfather", it is a chain and the
        # walk runs. Comparisons stay excluded. This can only open the leg, never close
        # it. See docs/BENCHMARKS.md, "The graph leg on public data".
        if weights.graph <= 0.0 < self.w_graph and not is_comparison(query) and (
                names_derived(query, self.derived_terms)
                or len(observed_refs(query, {c.predicate for c in claims.values()},
                                     self.registry.normalize)) > 1):
            weights = weights._replace(graph=self.w_graph)

        # The store has the last word. If no claim's object is another claim's subject,
        # a walk only returns other facts about the same hub with near-equal scores, and
        # fusion would read those positions as evidence. This check sits after intent
        # weighting because it must be able to close a leg `classify` opened. See
        # docs/INTERNALS.md, "The store-level graph gate".
        if weights.graph > 0.0 and not self._store_has_joins(scope.tenant):
            weights = weights._replace(graph=0.0)

        # Which candidates the question names, decided before the walk so it can mark
        # paths that start from one. Only the named end of a claim counts as an origin:
        # a walk out of the value end (`deploy_region=eu-west-1`) would reach every other
        # row sharing that value, which relates to the answer, not to the question.
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
            # Nothing named is in hand, so nothing the walk found could pass the filter.
            pass
        else:
            # `now` is passed down so the walk decays from the same instant as the
            # lookup legs.
            graph_hits, walked, derived = self._graph_search(
                claims, fused, scope, limit, valid_at, known_at, states,
                0.0 if where is not None else weights.graph, now, anchored_keys)
        if graph_hits:
            # Fused again from scratch, because positions in a two-leg fusion differ
            # from positions in a three-leg one.
            fused = reciprocal_rank_fusion(
                {VECTOR: vector_hits, LEXICAL: lexical_hits, GRAPH: graph_hits},
                k=self.rrf_k,
                weights={VECTOR: weights.vector, LEXICAL: weights.lexical,
                         GRAPH: weights.graph},
            )
            # A `Path` carries its claims, so they need no second load. The lookup legs'
            # objects win a collision, keeping one object per id.
            claims = {**walked, **claims}

        legs = _Legs(
            vector=_positions(vector_hits),
            lexical=_positions(lexical_hits),
            graph=_positions(graph_hits),
            # A leg that returned nothing is dropped from the relevance average rather
            # than counted as a zero, or a lexical-only query would halve every score.
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
        # A present-tense read in a project leaves out a user-wide value that the
        # project's own value shadows (`retrieve/shadow.py`). Done last, so only claims
        # every other filter kept cost a lookup. A read with a window is not present
        # tense, even one that reaches now.
        if valid_at is None and known_at is None and window is None and results:
            hidden = shadowed(self.store, self.registry, (r.claim for r in results), scope)
            results = [r for r in results if r.claim.id not in hidden]
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

        Also `True` when the store cannot answer. A backend without `connectivity`, or
        one that returns `{}`, has not measured anything, and treating that as "no joins"
        would switch off a working graph leg on every third-party store.

        Cached per tenant and re-measured every `GATE_RECHECK_EVERY` searches. Warns
        `UnjoinedStoreWarning` once per retriever, only when the store answered "no".
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
            # Present but unable to answer: treated as absent.
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

        Returns three things: the ranked `(claim_id, path score)` list; the claims behind
        it, which the paths already carry, so no extra load is needed; and the ids of every
        claim on a path that started from one of `anchored`, the entity keys the question
        named. `Explanation.anchor` reports those as `"path"`.

        The leg returns nothing when:

        * `w_graph <= 0` or there is no traverser. This is the shipped default.
        * the store's `adjacent` raises `NotImplementedError`. A `getattr` check cannot
          detect this, so it is caught, warned once, and remembered for this retriever.
        * there are no seeds.
        * `states` does not include `live`.
        * the caller passed a metadata or file-path filter. `_gather` then passes a zero
          weight, because `Store.adjacent` takes no filter and a walk would reach rows the
          filter excludes. The lookup legs still serve the filtered search.

        `known_at` and `valid_at` are passed through unchanged, so the walk is evaluated
        at the same instants as the lookup legs (`GraphTraverser._pin`).

        `states` switches the whole leg off rather than filtering its output.
        `Store.adjacent` walks only live edges, so everything this leg returns is live, and
        a search for only `ended` or `retired` claims must not receive live rows. A
        post-filter on `claim.state` would not work, because that is the state now, and at
        a historical `known_at` the lookup legs correctly return rows that were live then.
        """
        if w_graph <= 0.0 or self.traverser is None or self._graph_unsupported:
            return [], {}, frozenset()
        if "live" not in states:
            return [], {}, frozenset()
        # Driven from `fused`, skipping ids the store did not return, so a third-party
        # store that returns the wrong ids costs a seed rather than a `KeyError`.
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
        where: SearchFilter | None = None,
        *,
        cap: int | None = None,
    ) -> list[EpisodeResult]:
        """The legs over raw turns, discounted and capped.

        Like `_gather`, without what turns do not have: no liveness filter, no memory
        types and no quality rescoring. The legs are BM25, cosine and, when
        `w_temporal > 0`, closeness to the instant asked about (see
        `retrieve/temporal.py`).

        `cap` overrides `max_episodes`. A ranked call passes `rerank_top_n`, so the
        selector has that many turns to choose from.
        """
        vector = self._beside(query, partial(
            self._episode_vector_search, query, scopes, limit, valid_at, known_at, where))
        lexical_hits, terms = self._episode_lexical_search(
            query, scopes, limit, valid_at, known_at, where)
        anchor = anchor_for(valid_at, known_at, now)
        time_hits = self._episode_time_search(
            scopes, limit, valid_at, known_at, weights.temporal, anchor, where)
        vector_hits = vector()

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
                # Closeness is already in [0, 1], so it goes in unmapped. It uses the
                # `graph` slot because the graph leg never runs on episodes.
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
                    # As for a claim, `raw_score` is what fusion produced.
                    raw_score=fusion,
                    final_score=score,
                ),
            ))
        # Ties break on the content hash rather than the id; see `_rank`.
        out.sort(key=lambda r: (-r.score, r.episode.hash, r.episode.id))
        return self._above_floor(out)[:(self.max_episodes if cap is None else cap)]

    @staticmethod
    def _select_ms(rec: "Recorder | None", t0: float) -> None:
        """`RETRIEVAL_SELECT_MS`, on every call the ranked stage actually makes.

        Every branch emits it, including failures, because a provider timeout is latency
        the caller waited through.
        """
        if rec is not None:
            rec.timing(RETRIEVAL_SELECT_MS, (perf_counter() - t0) * 1000.0)

    def _run_ranked_stage(
        self, rec: "Recorder | None", selector: Selector, query: str,
        episodes: "list[EpisodeResult]", now: datetime,
    ) -> "tuple[Selection, list[EpisodeResult], list[EpisodeResult]]":
        """Admit the read, rerank the turns, and ask the model which turns to keep.

        `episodes` arrives gathered at `rerank_top_n`, in the episode leg's score order.
        Returns `(selection, kept_turns, tail)`. `kept_turns` has `explain.selected` and
        `.span` set and is empty unless the outcome is `applied`. `tail` is the reranked
        turns minus those kept, or `episodes` unchanged for `disabled`, where the reranker
        never ran. The caller uses `tail` only on `applied`.

        Admission covers the reranker call as well as the model call, because the cap
        bounds how long a ranked read holds a thread, and that includes the cross-encoder.
        `SelectorBusy` is counted here and then propagates; nothing else about the read is
        counted.
        """
        try:
            with selector.admit():
                turn_order = episodes
                if self.reranker is not None:
                    turn_order = rerank(self.reranker, query, list(episodes),
                                        top_n=self.rerank_top_n)
                # The selector sees one role's turns, so long assistant turns do not take
                # the window's slots from the user turns that hold the answer. If no turn
                # has that role, it gets the other role's turns. `route_roles=False`
                # skips this.
                routed = ([e for e in turn_order if e.episode.role == routed_role(query)]
                          if self.route_roles else [])
                scope = (routed or turn_order)[:selector.top_n]
                candidates = [
                    Candidate(id=e.episode.id, when=e.episode.ts, text=e.episode.content)
                    for e in scope]
                t0 = perf_counter()
                usage = Usage()
                try:
                    chosen = selector.select(query, candidates, asked_on=now, usage=usage)
                except SelectorRefused as exc:
                    # The provider answered 401 or 403.
                    self._select_ms(rec, t0)
                    if rec is not None:
                        rec.counter(RETRIEVAL_MODEL_REFUSED, reason="key_rejected")
                    return (Selection(outcome="key_rejected", status=exc.status,
                                      candidates=len(candidates)),
                            [], turn_order)
                except TimeoutError:
                    self._select_ms(rec, t0)
                    if rec is not None:
                        rec.counter(RETRIEVAL_MODEL_FALLBACK, reason="timeout")
                    return (Selection(outcome="fallback", reason="timeout",
                                      candidates=len(candidates)),
                            [], turn_order)
                except ValueError:
                    self._select_ms(rec, t0)
                    if rec is not None:
                        rec.counter(RETRIEVAL_MODEL_FALLBACK, reason="malformed")
                    return (Selection(outcome="fallback", reason="malformed",
                                      candidates=len(candidates)),
                            [], turn_order)
                except Exception as exc:                      # noqa: BLE001 - deliberate
                    self._select_ms(rec, t0)
                    status = getattr(exc, "status_code", None)
                    reason = "provider" if status is not None else "error"
                    if rec is not None:
                        if status is not None:
                            rec.counter(RETRIEVAL_MODEL_FALLBACK, reason=reason,
                                       status=str(status))
                        else:
                            rec.counter(RETRIEVAL_MODEL_FALLBACK, reason=reason)
                    return (Selection(outcome="fallback", reason=reason, status=status,
                                      candidates=len(candidates)),
                            [], turn_order)

                # With no candidates `select()` makes no model call, so neither the
                # query counter nor the latency is emitted, though the outcome is still
                # `applied`.
                if candidates:
                    self._select_ms(rec, t0)
                    if rec is not None:
                        rec.counter(RETRIEVAL_MODEL_QUERY)
                        if usage.reported > 0:
                            rec.counter(RETRIEVAL_TOKENS_IN, usage.input_tokens)
                            rec.counter(RETRIEVAL_TOKENS_OUT, usage.output_tokens)
                spans = {s.id: s.span for s in chosen}
                kept: list[EpisodeResult] = []
                for e in scope:
                    if e.episode.id in spans:
                        e.explain.selected = True
                        e.explain.span = spans[e.episode.id]
                        kept.append(e)
                    else:
                        e.explain.selected = False
                tail = [e for e in turn_order if e.episode.id not in spans]
                return (Selection(outcome="applied", candidates=len(candidates),
                                  kept=len(kept)),
                        kept, tail)
        except SelectorBusy:
            if rec is not None:
                rec.counter(RETRIEVAL_MODEL_REFUSED, reason="inflight")
            raise
        except SelectorRefused:
            # Raised by `admit()` before the reranker ran: the operator disabled the
            # stage. `key_rejected` comes from `select()` and is caught above.
            if rec is not None:
                rec.counter(RETRIEVAL_MODEL_REFUSED, reason="disabled")
            return Selection(outcome="disabled", candidates=0), [], episodes

    def _above_floor(self, results: "list[EpisodeResult]") -> "list[EpisodeResult]":
        """Cut the tail where the score falls off, or hand back everything.

        The list arrives in descending score order. The best match always survives,
        whatever the floor, so a non-empty list never becomes empty.
        """
        if self.episode_score_floor <= 0.0 or not results:
            return results
        cutoff = results[0].score * self.episode_score_floor
        kept = [r for r in results if r.score >= cutoff]
        return kept or results[:1]

    def _hydrate_episodes(self, ids: Sequence[str]) -> dict[str, Episode]:
        """One round trip if the store offers it, N if it is a third-party one."""
        bulk = getattr(self.store, "get_episodes", None)
        if bulk is not None:
            return bulk(ids)
        return {eid: e for eid in ids if (e := self.store.get_episode(eid)) is not None}

    def _episode_time_search(
        self, scopes: Sequence[Scope], limit: int, valid_at: datetime | None,
        known_at: datetime | None, w_temporal: float, anchor: datetime,
        where: SearchFilter | None = None,
    ) -> list[tuple[str, float]]:
        """Turns nearest the asked instant, or nothing. Degrades like the other legs.

        `episodes_near` is optional on the `Store` protocol. A store without it does not
        run this leg, as with `vector_search_episodes`.
        """
        if w_temporal <= 0.0:
            return []
        near = getattr(self.store, "episodes_near", None)
        if near is None:
            return []
        return rank_by_time(
            near(anchor, scopes, limit, valid_at=valid_at, known_at=known_at,
                 **_narrowed(where)), anchor)

    def _episode_vector_search(
        self, query: str, scopes: Sequence[Scope], limit: int,
        valid_at: datetime | None, known_at: datetime | None,
        where: SearchFilter | None = None,
    ) -> list[tuple[str, float]]:
        """Vector leg over turns. Abstains on a zero-norm query, as the claim leg does.

        A store without `vector_search_episodes` does not run this leg, leaving the
        lexical leg, which is the stronger one for verbatim recall.
        """
        search = getattr(self.store, "vector_search_episodes", None)
        if search is None:
            return []
        qvec = self._query_vector(query)
        if float(np.linalg.norm(qvec)) <= 0.0:
            return []
        return list(search(qvec, scopes, limit, valid_at=valid_at, known_at=known_at,
                           **_narrowed(where)))

    def _episode_lexical_search(
        self, query: str, scopes: Sequence[Scope], limit: int,
        valid_at: datetime | None, known_at: datetime | None,
        where: SearchFilter | None = None,
    ) -> tuple[list[tuple[str, float]], int]:
        """BM25 over turns, reduced to content terms exactly as the claim leg is.

        Dropping stopwords matters more here than for claims: turns are long, so a query
        reduced to `"do"` and `"about"` would match almost every turn.
        """
        search = getattr(self.store, "lexical_search_episodes", None)
        if search is None:
            return [], 0
        reduced = analyze(query)
        if reduced.abstains:
            return [], 0
        hits = search(reduced.text, scopes, limit, valid_at=valid_at, known_at=known_at,
                      **_narrowed(where))
        return list(hits), len(reduced.terms)

    @staticmethod
    def _interleave(claims: list[Retrieved], episodes: list[EpisodeResult],
                    k: int) -> list[Retrieved]:
        """Merge the episode tail into the claim ranking without disturbing it.

        The claim list is not in pure score order after `_rank`'s diversity pass, so a
        re-sort would undo it. Instead each episode is placed before the first claim it
        outscores. Ties go to the claim.
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

        Ties break on `value_key`, which is derived from the claim's content, so identical
        data ranks identically in every store. `id`, a fresh `uuid4` per ingest, is the
        final key only so the order is total. See docs/BENCHMARKS.md, "LOCOMO and
        LongMemEval".

        The diversity pass allows at most `max_per_slot` claims per `fact_key` (owner,
        subject, predicate) in the head and demotes the rest rather than dropping them,
        so `k` keeps its meaning and nothing is hidden from an audit. Demoted claims go
        after the other matching results but before any result that scores 0, so the list
        is in score order except that a slot's claims beyond the cap come after other
        matching results, even lower-scoring ones. Slot identity is used rather than
        embedding distance (MMR); see docs/BENCHMARKS.md, "Diversity by fact slot rather
        than by MMR".
        """
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
        matching = next((i for i, r in enumerate(head) if r.score <= 0), len(head))
        return (head[:matching] + overflow + head[matching:])[:k]

    def _vector_search(
        self,
        query: str,
        scopes: Sequence[Scope],
        limit: int,
        valid_at: datetime | None,
        known_at: datetime | None,
        states: Sequence[str],
        where: SearchFilter | None = None,
        window: TimeWindow | None = None,
    ) -> list[tuple[str, float]]:
        """Vector leg, skipped when the query embeds to nothing.

        A zero vector gives every candidate a cosine of 0.0, so the order would be
        arbitrary and fusion would read it as evidence. Queries with no alphanumeric
        content embed to zero, and so does any purely CJK query under the ASCII-only
        `HashingEmbedder`. Returning nothing lets BM25 answer alone.
        """
        qvec = self._query_vector(query)
        if float(np.linalg.norm(qvec)) <= 0.0:
            return []
        return list(self.store.vector_search(
            qvec, scopes, limit, known_at=known_at, states=states,
            **_world(valid_at, window), **_narrowed(where)))

    def _lexical_search(
        self,
        query: str,
        scopes: Sequence[Scope],
        limit: int,
        valid_at: datetime | None,
        known_at: datetime | None,
        states: Sequence[str],
        where: SearchFilter | None = None,
        window: TimeWindow | None = None,
    ) -> tuple[list[tuple[str, float]], int]:
        """Lexical leg, reduced to content terms and skipped when none survive.

        The store ORs every token it is given, and in a small personal store stopwords
        can be rare enough to dominate the IDF, so only content terms are sent. See
        `analyze`. Returns the hits and the number of terms they were scored over, which
        makes BM25 scores comparable across query lengths.
        """
        reduced = analyze(query)
        if reduced.abstains:
            return [], 0
        hits = self.store.lexical_search(
            reduced.text, scopes, limit, known_at=known_at, states=states,
            **_world(valid_at, window), **_narrowed(where))
        return list(hits), len(reduced.terms)

    @staticmethod
    def _believed_by(claim: Claim, known_at: datetime | None) -> bool:
        """Belief-time floor, enforced here rather than left to the store.

        Asking for all three `states` lifts most of the store's liveness filter, and a
        third-party store may lift the rest, so a claim recorded after `known_at` could
        otherwise appear in a historical answer. Only the belief axis is re-checked: the
        full state set lifts the valid-time interval by design (see
        `store.state_predicate`).
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
            # A path score is already in [0, 1], so it needs no mapping or clamp.
            graph=(0.0 if g is None else g[1]) if legs.graph_active else None,
            w_vector=weights.vector,
            w_lexical=weights.lexical,
            w_graph=weights.graph,
        )
        score = normalized_score(evidence, **quality)
        explain = Explanation(
            # `None` means the leg did not rank this claim.
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
            # Set by `rerank` if a reranker runs; `None` means none scored this claim.
            rerank_score=None,
            raw_score=final_score(fusion, **quality),
            final_score=score,
            intent=None if weights.intent is None else weights.intent.value,
            anchor=anchor,
        )
        return Result(claim=claim, score=score, explain=explain)


def _narrowed(where: SearchFilter | None) -> dict[str, Any]:
    """`where=` as keyword arguments for a store call, or none at all.

    Passed only when the caller filtered, so a store that does not accept `where` still
    serves unfiltered reads, and a filtered read against it raises a `TypeError` rather
    than returning rows the filter would have excluded.
    """
    return {} if where is None else {"where": where}


def _world(valid_at: datetime | None, window: TimeWindow | None) -> dict[str, Any]:
    """The world clock as keyword arguments for a claim search: the window, or the instant.

    The window is passed only on a windowed read, for the reason `_narrowed` gives.
    """
    return {"valid_at": valid_at} if window is None else {"valid_during": window}


def _positions(hits: Sequence[tuple[str, float]]) -> dict[str, tuple[int, float]]:
    """Map item id -> (0-based rank, raw score), keeping the best rank on repeats."""
    out: dict[str, tuple[int, float]] = {}
    for rank, (item_id, score) in enumerate(hits):
        out.setdefault(item_id, (rank, score))
    return out
