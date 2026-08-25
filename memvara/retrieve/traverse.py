"""Multi-hop traversal: the claim store read as the graph it already is.

A `Claim` is a `(subject, predicate, object)` triple and entity resolution folds every
spelling of a name onto one identity — so the store has been a labelled directed graph
since `entities.py` landed, and nothing could query it transitively. "Where does Alice
work" was one indexed lookup; "who does Alice's manager report to" was not expressible at
all, at any cost, because the two facts that answer it are two rows with no join between
them.

What this module adds is the join, and four properties that decide whether the answer is
worth anything.

**One instant *per axis* for the whole traversal.** Every edge on a returned path is
evaluated at the same `(valid_at, known_at)` pair, pinned once before the first hop and
passed unchanged to every store call after it. Without that pin a three-hop walk asks the
clock three times and can return a path whose first edge was believed in the morning and
whose last was believed in the afternoon — a connection that was never simultaneously
true, reported as a fact. That is the worst thing this feature could do, it is invisible
in any result that does not carry its timestamps, and it is what a bitemporal store is
uniquely able to refuse. Two axes do not weaken it; they make it a pair that moves
together. It is also why an unset axis does not mean "let each hop use its own now": see
`_pin`.

**A negation is not a link.** "Alice does not work at Acme" is a claim about Alice and
Acme, and `Store.adjacent` returns it, because at that layer it genuinely is adjacency.
It is dropped here, before it can become an edge, so the guarantee holds for every store
implementation rather than for the ones that remembered.

**Scope is checked on every hop, on the reading rule.** `Store.adjacent` is tenant-scoped
and nothing finer, exactly like `competing_claims`; authorization is this module's, with
`Scope.sees` — the same predicate `get()`, `why()` and `produced()` use. A claim the
caller could not have read directly never becomes an edge and its far end never enters
the frontier, so a path can only ever be composed of facts the caller could already have
enumerated with `get_all()`. Traversal joins what is readable; it does not widen what is
readable. See `_visible` for the one place that is delicate.

**Bounded and deterministic.** A depth cap, a beam on the frontier, a per-hop cap on
edges, a cycle check, and a total order with no `uuid4` deciding anything that matters.
Identical inputs give an identical answer in every store that holds the same data — the
same promise `hybrid.py` makes and for the same reason: an ordering that only holds
within one ingest is not reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, NamedTuple, Sequence

from ..schema import PredicateRegistry
from ..store.base import Store
from ..types import Claim, Scope, as_utc, default_entity, time_axes, utcnow
from .scoring import recency_factor

#: What one more hop costs, as a multiplier on the path's score.
#:
#: The one fitted constant here, and it does exactly one thing: it sets the exchange rate
#: between length and edge quality. Without it a path of three perfectly-confident,
#: perfectly-fresh edges scores the same as a single such edge, which is wrong — an
#: inference chained through three facts is a weaker answer than one fact, even when
#: every link in it is certain, because the *composition* is what nobody asserted.
#:
#: At 0.75, a two-hop path has to be a third better on edge quality than a one-hop path
#: to outrank it, and a three-hop path has to be 78% better. Worked against the case the
#: number was chosen on: three 0.6-confidence edges score 0.6³ x 0.75² = 0.12, while one
#: 0.99 edge scores 0.99. Unlike `LEXICAL_HALF_SATURATION` this is *not* merely a
#: calibration knob — it is monotone within a length and reorders across lengths, which
#: is its whole job — so it is exposed as `GraphTraverser(damping=...)` rather than
#: buried.
HOP_DAMPING = 0.75


@dataclass(frozen=True, slots=True)
class Edge:
    """One claim, walked in one direction.

    The direction is carried rather than inferred because a claim read backwards is a
    different statement about the world: `Acme founded_by Bob` traversed from Bob is
    still "Acme was founded by Bob" and must not be rendered, or reasoned about, as
    though Bob had founded something called `founded_by`. `backward` is what keeps the
    triple's own direction recoverable after the walk has flattened it into a chain.
    """

    claim: Claim
    #: True when this edge was walked object -> subject.
    backward: bool
    #: This edge's own contribution to the path score, in [0, 1]. See
    #: `GraphTraverser._strength`.
    strength: float

    @property
    def source_key(self) -> str:
        """Folded identity of the end the walk arrived from."""
        return self.claim.object_key if self.backward else self.claim.subject_key

    @property
    def target_key(self) -> str:
        """Folded identity of the end the walk arrives at."""
        return self.claim.subject_key if self.backward else self.claim.object_key

    @property
    def source(self) -> str:
        """The surface form of `source_key` — what was actually written."""
        return self.claim.object if self.backward else self.claim.subject

    @property
    def target(self) -> str:
        return self.claim.subject if self.backward else self.claim.object

    @property
    def predicate(self) -> str:
        return self.claim.predicate

    def render(self, escape: "Callable[[str], str] | None" = None) -> str:
        """The arrow, always pointing subject -> object however it was walked.

        `escape` is applied to the predicate, which is stored text like any other. It is
        a hook rather than a fixed rule because who is reading decides what is dangerous:
        a REPL wants the arrow legible, and a surface that renders into a model's context
        needs stored text unable to spell one. See `Path.render`.
        """
        predicate = self.predicate if escape is None else escape(self.predicate)
        return (f"<-{predicate}- " if self.backward
                else f"-{predicate}-> ")

    def __repr__(self) -> str:
        return (f"<Edge {self.source} {self.render().strip()} {self.target} "
                f"{self.strength:.3f} {self.claim.id}>")


@dataclass(frozen=True, slots=True)
class Path:
    """A chain of claims connecting two entities, with the score of the whole chain.

    Returned instead of the bare claims because a path the caller cannot inspect is an
    answer they cannot check: "Alice and Carol are connected, 0.42" is not a fact, it is
    a number. `edges` carries the claims and the direction each was walked, so every hop
    can be taken to `why()`; `nodes` carries the entities the chain passes through.

    `nodes` are *folded identities* (`entity_key`), which is what makes them comparable —
    `Acme`, `Acme Corp` and `acme, inc.` are one node. `labels` is the same sequence as
    the spellings that were actually stored, which is what a human reads.
    """

    #: Folded entity identities, `len(edges) + 1` of them, seed first.
    nodes: tuple[str, ...]
    edges: tuple[Edge, ...]
    #: In [0, 1]. See `GraphTraverser._extend` for how it composes.
    score: float

    @property
    def hops(self) -> int:
        return len(self.edges)

    @property
    def claims(self) -> tuple[Claim, ...]:
        """The underlying claims, in walk order. Every one of them is live at the
        clock pair the traversal was evaluated at, and every one is individually readable
        at the scope that asked."""
        return tuple(e.claim for e in self.edges)

    @property
    def labels(self) -> tuple[str, ...]:
        """The nodes as they were spelled in the claims that named them."""
        if not self.edges:
            # Unreachable from the public API, which never returns a zero-hop path — but
            # `nodes` is the honest answer for one rather than an IndexError.
            return self.nodes
        return (self.edges[0].source,) + tuple(e.target for e in self.edges)

    @property
    def signature(self) -> tuple[str, ...]:
        """Content identity of this path: the seed plus each edge's `value_key`.

        What de-duplication and tie-breaking run on, and deliberately not the claim ids.
        Two rows asserting the same thing — one written at user scope, one at tenant
        scope, both visible to the same reader — are one edge, and returning the chain
        twice is noise. `value_key` is derived from content, so two stores holding the
        same data tie-break identically; ids are `uuid4` and would make the order a
        property of which ingest happened to run.
        """
        return (self.nodes[0],) + tuple(e.claim.value_key for e in self.edges)

    @property
    def undirected(self) -> tuple[str, ...]:
        """The same identity with the direction taken out: this path or its mirror.

        `signature` starts at `nodes[0]`, so a walk that reaches Acme from Alice and one
        that reaches Alice from Acme sign differently while being the same single stored
        claim read from opposite ends. `neighborhood` never sees the pair, because its
        seeds are one entity's aliases and arrival at a seed is blocked; `spread` seeds
        *distinct* entities, and `seed_keys` emits both ends of every top-ranked claim, so
        for the head of the fused list the pair is guaranteed rather than possible.

        Read as the lexicographically smaller of the two readings, so which one survives
        is a property of the content and not of the order the seeds happened to arrive in.

        >>> a = Claim(subject="Alice", predicate="works_at", object="Acme")
        >>> fwd = Path(nodes=("alice", "acme"), edges=(Edge(a, False, 1.0),), score=1.0)
        >>> rev = Path(nodes=("acme", "alice"), edges=(Edge(a, True, 1.0),), score=1.0)
        >>> fwd.signature == rev.signature
        False
        >>> fwd.undirected == rev.undirected
        True
        """
        keys = tuple(e.claim.value_key for e in self.edges)
        return min((self.nodes[0],) + keys, (self.nodes[-1],) + keys[::-1])

    def render(self, escape: "Callable[[str], str] | None" = None) -> str:
        """One line: `Alice -works_at-> Acme -founded_by-> Bob`.

        **The arrows are this renderer's grammar and the labels are stored text**, so a
        caller rendering into a model's context has to be able to neutralise one without
        losing the other. `escape` is applied to every label and every predicate, never to
        the arrows, and defaults to `None` because a REPL and a test want the plain form.

        Without it, a single claim whose object is `Acme -owned_by-> The_CIA` renders as a
        two-hop chain and the row still says `1 hop`. Nothing downstream can tell the
        forged hop from a walked one, which is why the hook is here rather than a
        post-hoc scrub at the call site: only this method knows which spans are ours.

        >>> a = Claim(subject="Alice", predicate="works_at", object="Acme -owned_by-> X")
        >>> path = Path(nodes=("alice", "acme"), edges=(Edge(a, False, 1.0),), score=1.0)
        >>> path.render()
        'Alice -works_at-> Acme -owned_by-> X'
        >>> path.render(escape=lambda s: s.replace("-", "\u2011"))
        'Alice -works_at-> Acme ‑owned_by‑> X'
        """
        labels = self.labels if escape is None else tuple(escape(x) for x in self.labels)
        out = [labels[0]]
        for edge, label in zip(self.edges, labels[1:]):
            out.append(edge.render(escape) + label)
        return " ".join(out)

    def __repr__(self) -> str:
        return f"<Path {self.score:.4f} {self.hops}h {self.render()}>"


class _Pin(NamedTuple):
    """The clock pair a whole traversal is evaluated at. See `GraphTraverser._pin`.

    Named rather than a bare tuple because the two ends are not interchangeable and
    `pin[1]` at a call site says nothing about which clock it is — and the one that
    feeds `recency_factor` has to be the belief clock specifically.
    """

    valid_at: datetime
    known_at: datetime


def _identities(surface: str, learned: Sequence[str]) -> tuple[str, ...]:
    """The folded keys one end of a question stands for, deterministic fold first.

    The fold is prepended here rather than trusted from `learned`, and that is the line
    that makes alias resolution safe: a caller can *add* identities to a probe and can
    never replace the one this module has always used, so a merge widens what a walk
    reaches and can never trade one half of an entity for the other. A caller that passes
    nothing gets the single key and the behaviour it had before any of this existed.

    Falsy keys are dropped, so an unfoldable end still refuses to become the empty node
    that every retraction in the tenant would hang off.

    >>> _identities("Acme, Inc.", ())
    ('acme',)
    >>> _identities("Big Blue", ("big blue", "ibm"))
    ('big blue', 'ibm')
    >>> _identities("...", ())          # unfoldable, and still not the empty node
    ('...',)
    >>> _identities("", ())
    ()
    """
    return tuple(dict.fromkeys(k for k in (default_entity(surface), *learned) if k))


def _order(path: Path) -> tuple:
    """The total order paths are ranked and truncated by.

    Score first, then content, then ids — the same three-tier shape `HybridRetriever`
    sorts on, for the same reason. The id tier only ever decides between two paths that
    are equal on score *and* assert the same values, so nothing observable depends on a
    `uuid4`; it is there to make the order total rather than merely deterministic.
    """
    return (-path.score, path.signature, tuple(e.claim.id for e in path.edges))


class GraphTraverser:
    """Bounded, scope-checked, single-clock-pair traversal over the claim graph.

    Constructed by `Memvara`; usable on its own against any `Store` that implements
    `adjacent`. Holds no state between calls, so one instance serves every scope.
    """

    def __init__(
        self,
        store: Store,
        registry: PredicateRegistry,
        *,
        damping: float = HOP_DAMPING,
        beam: int = 64,
        max_per_relation: int = 2,
        edge_limit: int = 1000,
    ) -> None:
        self.store = store
        self.registry = registry
        #: Per-hop score multiplier. See `HOP_DAMPING`. Values above 1.0 would make a
        #: longer path able to outscore its own prefix, which every consumer of `score`
        #: assumes cannot happen.
        self.damping = damping
        #: How many partial paths survive each level. The frontier cap, and the only
        #: thing standing between a hub node and a combinatorial explosion: an entity
        #: with 200 edges reached at depth 3 is 8,000,000 paths without it. Pruning is by
        #: the same total order the results are ranked by, so it drops the worst
        #: candidates and drops them reproducibly.
        self.beam = beam
        #: How many paths may leave one node by one relation in the same direction
        #: before the rest are demoted. See `_diversify`; 0 turns the pass off.
        self.max_per_relation = max_per_relation
        #: Claims fetched per hop, before scope filtering. See `_visible`.
        self.edge_limit = edge_limit

    # -- public ---------------------------------------------------------------

    def neighborhood(
        self,
        entity: str,
        scope: Scope,
        *,
        depth: int = 2,
        k: int = 10,
        min_hops: int = 1,
        predicates: Sequence[str] | None = None,
        as_of: datetime | None = None,
        valid_at: datetime | None = None,
        known_at: datetime | None = None,
        min_score: float = 0.0,
        entity_keys: Sequence[str] = (),
    ) -> list[Path]:
        """What is around `entity`: every path of `min_hops`..`depth` hops out of it.

        Both directions by default, because "who works at Acme" and "where does Alice
        work" are the same edge and a caller asking what surrounds an entity wants both.

        `min_hops` exists because of a number rather than a hunch. Score is
        non-increasing along a path, so *every* one-hop path outranks *every* two-hop one
        at equal edge quality — and on a graph where the interesting entities have a
        handful of relations each, a `k` of 5 or 12 is spent entirely on the immediate
        neighbours and the two-hop answer never appears. Measured on `bench/multihop.py`
        over questions whose answer is exactly two hops away:

            k                 5      12      25
            as is          5.3%   69.7%  100.0%
            min_hops=2    41.0%  100.0%  100.0%

        Raising `k` past the size of the one-hop frontier works too and is what a caller
        reaches for first; this is the version that does not require knowing that size.

        `entity_keys` are further folded identities that name the same entity as
        `entity` — what an `EntityRegistry.probe_keys` lookup returns for the reader's
        owner, which is how a probe spelled with a learned alias ("Big Blue") reaches the
        claims filed under the canonical key (`ibm`). This module holds no registry and
        resolves nothing itself; identity is owner-scoped and only the caller knows whose
        question this is. See `_identities` for why passing them can only ever widen the
        walk.
        """
        return self._walk(entity, None, scope, depth=depth, k=k, min_hops=min_hops,
                          predicates=predicates, as_of=as_of, valid_at=valid_at,
                          known_at=known_at, min_score=min_score,
                          source_keys=entity_keys, target_keys=())

    def between(
        self,
        source: str,
        target: str,
        scope: Scope,
        *,
        depth: int = 3,
        k: int = 3,
        predicates: Sequence[str] | None = None,
        as_of: datetime | None = None,
        valid_at: datetime | None = None,
        known_at: datetime | None = None,
        min_score: float = 0.0,
        source_keys: Sequence[str] = (),
        target_keys: Sequence[str] = (),
    ) -> list[Path]:
        """How `source` and `target` are connected: the best paths of at most `depth` hops.

        A path that reaches the target stops there and is not expanded further, so every
        result *ends* at the target rather than merely passing through it. Shorter paths
        therefore surface first for free, which is also the ranking: score is
        non-increasing along a path, so a two-hop route can only outrank a one-hop route
        when the one-hop route's own edge is weak.

        Returns `[]` when nothing connects them within `depth` — which is a real answer
        and, given the bounds, an answer about this search rather than about the store:
        a path can also be missed because the beam pruned its prefix. Widen `beam` before
        concluding two entities are unrelated.

        `source_keys` and `target_keys` widen either end onto the other spellings of its
        entity, exactly as `neighborhood`'s `entity_keys` does. When the two ends resolve
        to the same entity the answer is `[]` — "how is IBM connected to Big Blue" is a
        question about one thing, and the loops through everything it touches are not an
        answer to it.
        """
        return self._walk(source, target, scope, depth=depth, k=k, min_hops=1,
                          predicates=predicates, as_of=as_of, valid_at=valid_at,
                          known_at=known_at, min_score=min_score,
                          source_keys=source_keys, target_keys=target_keys)

    def spread(
        self,
        seed_keys: Sequence[str],
        scope: Scope,
        *,
        depth: int = 2,
        k: int = 10,
        min_hops: int = 1,
        predicates: Sequence[str] | None = None,
        as_of: datetime | None = None,
        valid_at: datetime | None = None,
        known_at: datetime | None = None,
        min_score: float = 0.0,
        now: datetime | None = None,
    ) -> list[Path]:
        """`neighborhood`, from several already-folded keys at once and no surface form.

        The entry point retrieval uses. `neighborhood` exists for a question spelled by a
        person — "what is around Acme" — so it takes a surface form, folds it, and widens
        the fold with whatever aliases the owner has learned. A retrieval seed is not
        spelled by anybody: it is `Claim.subject_key`, read off a row that the write path
        already folded and already resolved through the same registry. Handing that back
        to `_identities` would fold an entity key a second time, which is at best a no-op
        and at worst re-derives a different key from the canonical one.

        So this takes keys and trusts them, and the trust is bounded: falsy keys are
        dropped, which is the one check `_identities` was doing that still applies here —
        `object=""` is how a retraction clears a slot, and admitting the empty key as a
        node would fuse every retraction in the tenant into one hub.

        Several seeds in one walk rather than one walk per seed, for the reason
        `_identities` gives: the beam, the diversification and the per-level cap are
        properties of *one answer*, and running the walk n times and concatenating would
        multiply every bound by n.

        Scope, the clock pin and the negation rule are exactly `neighborhood`'s. Nothing a
        walk started this way can reach is anything the caller could not have read
        directly; see `_visible`.
        """
        pin = self._pin(*time_axes(as_of, valid_at, known_at), now=now)
        seeds = tuple(dict.fromkeys(key for key in seed_keys if key))
        if not seeds:
            return []
        return self._search(seeds, frozenset(), scope, depth=depth, k=k,
                            min_hops=min_hops, predicates=predicates, pin=pin,
                            min_score=min_score, blocked=frozenset())

    # -- internals ------------------------------------------------------------

    @staticmethod
    def _pin(valid_at: datetime | None, known_at: datetime | None,
             now: datetime | None = None) -> _Pin:
        """The one pair of instants the whole traversal is evaluated at.

        This is the load-bearing line of the module. An unset axis cannot be forwarded
        to the store, because `Store.adjacent` with a missing instant reads the clock
        *per call* — so a three-hop walk would evaluate its three hops at three
        different instants and could return a chain that was never simultaneously
        believed. A claim retired between hop one and hop two would still have been
        walked; a claim recorded between them would appear as though it had been there
        all along.

        Pinning `utcnow()` here makes both impossible, and makes them impossible in the
        direction that is honest: everything recorded after the traversal began is
        excluded, and everything retired after it began is still walked, which is exactly
        "the world as it stood when you asked".

        Splitting `as_of` into two axes did not weaken this, it widened what has to be
        pinned: the invariant is now that the *pair* is fixed before the first hop, not
        merely that each axis is. One clock read fills both defaults, so a walk with
        neither axis given is still evaluated at a single coherent moment rather than at
        two instants microseconds apart — which is what makes "what is around Alice
        right now" mean the same thing on hop three as on hop one, on both clocks.

        `now` chooses *which* moment that is, and changes nothing else. It replaces the
        clock read and not either axis, so time travel keeps its meaning: a caller who
        named `known_at` still gets `known_at`. Without it two identical walks seconds
        apart score their edges from two instants, and `_strength` decays from the belief
        clock — so the same walk over the same store returns the same paths with
        different weights, which is reproducible only if nobody looks closely. See
        `Consolidator.run(now=)`, which is this same parameter for the same reason on the
        write path.
        """
        moment = as_utc(now) if now is not None else utcnow()
        return _Pin(as_utc(valid_at) if valid_at is not None else moment,
                    as_utc(known_at) if known_at is not None else moment)

    def _predicates(self, predicates: Sequence[str] | None) -> list[str] | None:
        """Caller-supplied predicate names, folded onto the registry's canonical ones.

        Without this a filter for `works_at` silently misses claims stored under
        `employed_by_company`, which is the very collapse `schema.py` exists to perform —
        and it would miss them without saying so, as an empty result.
        """
        if predicates is None:
            return None
        return list(dict.fromkeys(self.registry.normalize(p) for p in predicates))

    def _walk(
        self,
        source: str,
        target: str | None,
        scope: Scope,
        *,
        depth: int,
        k: int,
        min_hops: int,
        predicates: Sequence[str] | None,
        as_of: datetime | None,
        valid_at: datetime | None,
        known_at: datetime | None,
        min_score: float,
        source_keys: Sequence[str] = (),
        target_keys: Sequence[str] = (),
    ) -> list[Path]:
        """Breadth-first, beam-pruned, one clock pair throughout.

        Both public calls land here.

        `min_score` prunes partial paths as they are built, and that is exact rather than
        approximate: `_extend` multiplies by two factors that are both at most 1.0, so a
        path's score can never rise. A prefix already below the floor cannot produce a
        suffix above it, so pruning early discards exactly what a floor applied at the
        end would have discarded, and does it before those prefixes cost a hop.

        An end that resolves to several identities is walked from all of them at once
        rather than once per identity, which is what keeps the beam, the diversification
        and the per-level `[:k]` cap meaning what they say: they are properties of one
        answer to one question, and running the walk twice and concatenating would
        double whatever they bound.
        """
        pin = self._pin(*time_axes(as_of, valid_at, known_at))
        seeds = _identities(source, source_keys)
        if not seeds:
            return []
        goals: frozenset[str] = frozenset()
        if target is not None:
            goals = frozenset(_identities(target, target_keys))
            if not goals or goals & frozenset(seeds):
                # An entity is not connected *to itself* by a path, and a target that
                # folds to nothing is not an entity at all. Both would otherwise return
                # every cycle through the seed. Overlap rather than equality because two
                # names of one merged entity are one endpoint, however they were spelled.
                return []
        return self._search(seeds, goals, scope, depth=depth, k=k, min_hops=min_hops,
                            predicates=predicates, pin=pin, min_score=min_score,
                            # Every seed here is a name for the *same* entity, so a hop
                            # onto any of them is a self-loop wearing another spelling.
                            blocked=frozenset(seeds))

    def _search(
        self,
        seeds: Sequence[str],
        goals: frozenset[str],
        scope: Scope,
        *,
        depth: int,
        k: int,
        min_hops: int,
        predicates: Sequence[str] | None,
        pin: _Pin,
        min_score: float,
        blocked: frozenset[str],
    ) -> list[Path]:
        """The walk itself, from resolved keys and a resolved clock pair.

        Split out from `_walk` for `spread`, which arrives holding canonical keys already
        and must not re-fold them — see there. Everything about the walk lives here, so
        the two entry points cannot drift into two different traversals: `_walk` decides
        what the seeds and goals *are*, this decides what happens to them.

        `blocked` is the one thing the two entry points genuinely disagree about, and it
        is not a knob. `_walk`'s seeds are several spellings of **one** entity, so
        arriving at any of them is a self-loop and every one-hop path would otherwise come
        back twice — once direct and once via the entity's other name. `spread`'s seeds
        are several **different** entities, so blocking them would delete the answer:
        every edge between two seeds is exactly the join the leg exists to make, and with
        five seeds off the head of a fused list most edges have both ends in the set. That
        was not a hypothetical — it returned zero paths on the first store it was pointed
        at, and zero paths from a leg that is allowed to abstain looks like a corpus with
        no structure in it.
        """
        if depth <= 0 or k <= 0:
            return []
        preds = self._predicates(predicates)
        if preds == []:
            return []       # "these predicates and no others", of which there are none

        # One partial path per identity, all at depth zero, all scoring 1.0. They are the
        # same node as far as the walk is concerned; only the store still keys them apart.
        frontier = [Path(nodes=(s,), edges=(), score=1.0) for s in seeds]
        found: dict[tuple[str, ...], Path] = {}
        for hops in range(1, depth + 1):
            edges = self._edges(scope, [p.nodes[-1] for p in frontier], preds, pin)
            grown: dict[tuple[str, ...], Path] = {}
            for path in frontier:
                for edge in edges.get(path.nodes[-1], ()):
                    if edge.target_key in path.nodes or edge.target_key in blocked:
                        # Cycle: a path may not revisit an entity. `blocked` extends that
                        # to a *merged* entity — an edge from `big blue` to `ibm` is a
                        # self-loop wearing two names, and `_edges` already drops the
                        # single-key kind. Without it every one-hop path came back twice,
                        # once direct and once via the entity's other name. It is empty
                        # for `spread`, whose seeds are different entities rather than
                        # one entity's aliases; see `_search`.
                        continue
                    nxt = self._extend(path, edge)
                    if nxt.score < min_score:
                        continue
                    rival = grown.get(nxt.signature)
                    if rival is None or _order(nxt) < _order(rival):
                        grown[nxt.signature] = nxt
            if not grown:
                break
            ranked = self._diversify(sorted(grown.values(), key=_order))
            if not goals:
                # `[:k]` and not the whole level: a level can be `beam` x fan-out wide,
                # and without a cap here a hub keeps every candidate it ever generated
                # alive to the final sort — the one collection in this walk that nothing
                # else bounds. What it keeps is the head of the same diversified order the
                # final pass ranks by, so the two agree about what a good level looks like.
                #
                # Short paths are still *expanded* when `min_hops` excludes them — they
                # are the only way to reach the deeper ones — they are simply not
                # collected as answers.
                #
                # One per *undirected* identity, and the dedup happens before `[:k]`
                # rather than after: a mirrored pair is one stored claim, and letting
                # both through spends two of the caller's `k` on it. With `spread`'s
                # seeds — both ends of each top-ranked claim — that is not an edge case,
                # it is what the head of the list looks like every time.
                #
                # Only for collection. `ranked` itself keeps both directions, because
                # `frontier` is taken from it and the two readings extend to genuinely
                # different places: `alice→acme` grows towards Acme's neighbours and
                # `acme→alice` towards Alice's. Deduping before the frontier would make
                # this cheaper by making the walk reach less.
                distinct: list[Path] = []
                seen: set[tuple[str, ...]] = set()
                for path in ranked:
                    if path.undirected not in seen:
                        seen.add(path.undirected)
                        distinct.append(path)
                for path in distinct[:k] if hops >= min_hops else ():
                    found.setdefault(path.signature, path)
            else:
                # Arrival is terminal. A path that has reached the target and keeps
                # walking is answering a different question, and the cycle check would
                # stop it returning anyway.
                arrived = [p for p in ranked if p.nodes[-1] in goals]
                for path in arrived[:k]:
                    found.setdefault(path.signature, path)
                ranked = [p for p in ranked if p.nodes[-1] not in goals]
            frontier = ranked[:self.beam]
            if not frontier:
                break
        # Diversified twice, and both are needed. The per-level pass above decides which
        # paths are *collected* when a level is wider than `k`; this one decides the
        # order they come back in. Applying it only to the levels looked like it worked
        # and did not — this final sort put the ranking straight back the way it was, so
        # the answer moved only when a level happened to truncate.
        return self._diversify(sorted(found.values(), key=_order))[:k]

    def _diversify(self, ranked: list[Path]) -> list[Path]:
        """Spread the head of a level across relations. Demotes, never drops.

        The failure this fixes was visible the first time `bench/multihop.py` printed an
        example. Asked what is two hops from a person, every path scored identically —
        equal confidence, equal recency, equal length — so the tie-break ran on a content
        hash, and the top of the list was six ways of saying "a colleague":

            Ada -works_at-> Ahmed Systems <-works_at- Ada Kovac
            Ada -works_at-> Ahmed Systems <-works_at- Mira Costa
            Ada -works_at-> Ahmed Systems <-works_at- Gita Ibarra

        while `-headquartered_in-> Tallinn`, the one path that answered the question, sat
        outside `k`. Ties are not the exception here the way they are in search: a graph
        of hand-asserted facts has confidence 1.0 everywhere, so *most* paths of one
        length tie, and whichever relation a hub happens to have most of takes the whole
        answer. Two-hop questions on `bench/multihop.py`, with `min_hops=2` and nothing
        else changed:

            k              5      12      25
            off         27.3%   56.7%   78.3%
            capped at 2 41.0%  100.0%  100.0%

        Keyed on (node departed from, relation, direction) — deliberately the same shape
        as `HybridRetriever._rank`'s cap on `fact_key`, and demoting rather than dropping
        for the same reason: capping by deletion would make `k` mean something different
        depending on how the graph happened to cluster, and would hide the cluster from
        anyone auditing it.
        """
        if self.max_per_relation <= 0:
            return ranked
        head: list[Path] = []
        overflow: list[Path] = []
        used: dict[tuple[str, str, bool], int] = {}
        for path in ranked:
            last = path.edges[-1]
            slot = (path.nodes[-2], last.predicate, last.backward)
            seen = used.get(slot, 0)
            if seen < self.max_per_relation:
                used[slot] = seen + 1
                head.append(path)
            else:
                overflow.append(path)
        return head + overflow

    def _extend(self, path: Path, edge: Edge) -> Path:
        """One more hop, and the whole of the scoring rule.

        `score = Π strengthᵢ × damping^(hops-1)`. Multiplicative, so uncertainty
        compounds the way it does in any chain of inferences and a weak link anywhere
        caps the whole path; and the first hop is undamped, so a single perfect edge
        scores 1.0 and the number stays comparable with `Result.score`, which is also a
        [0, 1] relevance.

        Both factors are at most 1.0, which gives the invariant everything else rests on:
        **a path can never outscore its own prefix.** That is what makes `min_score`
        prunable mid-walk, and what makes "shorter unless the short route is worse" fall
        out of the ranking instead of being imposed on it.
        """
        damped = self.damping if path.edges else 1.0
        return Path(
            nodes=path.nodes + (edge.target_key,),
            edges=path.edges + (edge,),
            score=path.score * edge.strength * damped,
        )

    def _strength(self, claim: Claim, at: datetime) -> float:
        """One edge's contribution: how sure we are, times how current it is.

        **Confidence** is the extractor's own report on the edge, which is the direct
        answer to "does this link exist", clamped because nothing enforces its range and
        a value above 1.0 would let a longer path beat its own prefix.

        **Recency** is `recency_factor`, the predicate-keyed decay retrieval already
        ranks by. It belongs here for the reason it belongs there and for one more:
        liveness is a hard filter, so a `works_at` from 2019 that nobody ever superseded
        is *live* and is still the weakest possible evidence about where someone works
        today, while a `born_in` from 2019 is as good as it will ever be. Only a
        predicate-keyed half-life can tell those apart, and traversal needs it more than
        search does, because search shows the user the stale fact and lets them judge it
        while a traversal buries it in the middle of a chain.

        **Salience is deliberately excluded**, and it is the one signal that looks like
        it belongs. It is unbounded above 1.0 by design — reinforcement earns headroom up
        to `MAX_SALIENCE` — so multiplying it along a path would let three well-rehearsed
        hops outscore one, breaking the prefix invariant outright. Clamping it to 1.0
        would fix that and introduce a worse problem: consolidation derives salience by
        decaying it on the *same* predicate half-life `recency_factor` uses, so the two
        clamped signals are one signal counted twice, and a path through three
        slow-moving facts would be squared out of the ranking by a coincidence of
        implementation rather than by anything about the world.
        """
        confidence = min(1.0, max(0.0, claim.confidence))
        return confidence * recency_factor(claim, self.registry, at)

    def _edges(self, scope: Scope, keys: Sequence[str], predicates: list[str] | None,
               pin: _Pin) -> dict[str, list[Edge]]:
        """Every walkable edge leaving the current frontier, indexed by the node it leaves.

        Three claims are dropped here rather than deeper, and each drop is a semantic:

        * **Negative polarity.** "Alice does not work at Acme" is adjacency and is not a
          link. Traversing it would report the absence of a relationship as the presence
          of one, which is the plainest way this feature could produce a confident lie.
        * **An empty endpoint.** `object=""` is how a retraction says "clear the slot",
          so the empty key is a real stored value and not a missing one. Admitting it as
          a node would fuse every retraction in the tenant into a single hub through
          which everything is connected to everything.
        * **A self-loop.** Dropped as an optimization only: its far end is the node the
          walk is standing on, so the cycle check would discard it one step later.
        """
        wanted = {k for k in keys if k}
        out: dict[str, list[Edge]] = {}
        for claim in self._visible(scope, sorted(wanted), predicates, pin):
            if claim.polarity <= 0:
                continue
            subject, obj = claim.subject_key, claim.object_key
            if not subject or not obj or subject == obj:
                continue
            # Decay from the *belief* clock, the same choice `HybridRetriever.search`
            # makes: recency answers "how long ago did we last hear this, from where
            # the question stands", and the question stands at `known_at`.
            strength = self._strength(claim, pin.known_at)
            for key, backward in ((subject, False), (obj, True)):
                if key in wanted:
                    out.setdefault(key, []).append(Edge(claim, backward, strength))
        return out

    def _visible(self, scope: Scope, keys: Sequence[str], predicates: list[str] | None,
                 pin: _Pin) -> list[Claim]:
        """The frontier's claims, reduced to the ones this caller may actually read.

        `Scope.sees`, the reading rule — a handle sees its own scope and every broader
        one, never a deeper or a sideways one. Applied *before* a claim becomes an edge
        and therefore before its far end can enter the frontier, which is what keeps
        traversal from becoming the door that reads across users: every node on every
        returned path was named by a claim the caller could have fetched with
        `get_all()`, so the reachable set is unchanged and only the questions that can be
        asked about it are new.

        **The scope goes to the store, and the check is repeated here.** This used to
        filter a page the store had already truncated, and re-ask wider when the cap was
        hit and something was dropped — the shape `HybridRetriever._gather` uses for
        filter starvation. That was unsound rather than merely slow: on a shared tenant
        the page fills with another user's claims and the caller's own edges are cut
        before this filter can keep them. Measured with one user holding 20 readable
        claims about a hub, 15,000 competing claims returned 19 and 40,000 returned 8,
        with nothing in the result to say it was partial. Widening only moved the
        threshold, which is why the retry is gone rather than tuned: a filter and a limit
        cannot be split across two layers and still give a correct top-k.

        The Python-side `sees` check stays, and is not redundant. `scopes` is part of a
        published protocol that third-party stores implement, so this is the line that
        makes the guarantee ours rather than theirs — a store that ignores the argument
        returns extra rows and they are dropped here, slowly but correctly.

        What remains is a genuinely enormous hub of claims the caller *can* read, which
        `edge_limit` still truncates. That is truncation rather than leakage, and it is
        deterministic — see `SQLiteStore.adjacent` on why the ordering is content-derived.
        """
        fetch = getattr(self.store, "adjacent", None)
        if fetch is None:
            raise NotImplementedError(
                f"{type(self.store).__name__} does not implement adjacent(); multi-hop "
                "traversal needs an index over folded subjects and objects, and a scan "
                "of the tenant per hop is not a substitute"
            )
        rows = fetch(scope.tenant, keys, predicates=predicates,
                     valid_at=pin.valid_at, known_at=pin.known_at,
                     scopes=scope.ancestors(), limit=self.edge_limit)
        return [c for c in rows if scope.sees(c.scope)]
