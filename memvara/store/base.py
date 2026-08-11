"""Storage protocol.

Everything the engine needs from persistence, and nothing more. The default
implementation is SQLite so the library works with no infrastructure, but the surface
is deliberately narrow enough that pgvector, Qdrant or LanceDB slot in behind it.

Note the shape of `competing_claims`: it is a keyed lookup, not a similarity search.
That signature is the whole reason contradiction detection can be exact rather than
best-effort.

**Two time parameters, not one.** Every read that used to take `as_of` now takes
`valid_at` (the world clock) and `known_at` (the belief clock), each defaulting to
`None` for "now". They are independent because the interesting bitemporal question is
the one where they differ — "what do we *now* believe was true in June" is
`valid_at=June, known_at=None`, and no single instant expresses it. `as_of` survives
only on the public facade, where it is exact sugar for `valid_at=known_at=T`; nothing
below the facade sees it. See `memvara.types.time_axes`.
"""

from __future__ import annotations

from datetime import datetime
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Iterable, Protocol, Sequence, runtime_checkable

import numpy as np

from ..types import Claim, Episode, Scope

if TYPE_CHECKING:
    # Only for annotations: a `Store` implementation should not have to import
    # the schema module to satisfy the protocol.
    from ..schema import PredicateSpec


@runtime_checkable
class Store(Protocol):
    # --- episodes ---------------------------------------------------------
    def add_episode(self, ep: Episode) -> None: ...
    def get_episode(self, episode_id: str) -> Episode | None: ...
    def find_episode_by_hash(self, tenant: str, ep_hash: str) -> Episode | None: ...

    def get_episodes(self, episode_ids: Sequence[str]) -> dict[str, Episode]:
        """Bulk fetch, so hydrating a result set is one query rather than one per hit."""
        ...

    def iter_episodes(self, tenant: str | None = None) -> Iterable[Episode]:
        """Every stored turn, optionally for one tenant. What re-embedding walks."""
        ...

    def scope_episodes(self, scopes: Sequence[Scope], *, limit: int | None = None,
                       newest_first: bool = False) -> list[Episode]:
        """Turns visible at these scopes, in `ts` order, newest end optional.

        Here because `iter_episodes` was the only listing, so a caller wanting one
        scope's turns filtered the whole tenant in Python — which an adapter exposing a
        session transcript does on every read, paying for the tenant's size to answer a
        question about one session.

        Matches the scopes given and does not descend into narrower ones; pass
        `scope.ancestors()` for the widening view retrieval uses. Fails closed on an
        empty sequence, exactly as `candidate_ids` does. `newest_first` flips the order
        and with it the end `limit` takes from, so "the last N turns" is expressible.
        """
        ...

    # --- claims -----------------------------------------------------------
    def put_claim(self, claim: Claim) -> None: ...
    def get_claim(self, claim_id: str) -> Claim | None: ...

    def get_claims(self, claim_ids: Sequence[str]) -> dict[str, Claim]:
        """Bulk fetch. Avoids the N+1 that otherwise makes retrieval scale with
        result count rather than with the query."""
        ...

    def batch(self) -> AbstractContextManager["Store"]:
        """Context manager deferring commits to one transaction for bulk work."""
        ...

    def competing_claims(self, tenant: str, fact_key: str, *,
                         valid_at: datetime | None = None,
                         known_at: datetime | None = None) -> list[Claim]:
        """Live claims occupying the same (subject, predicate) slot. Exact, indexed.

        The write path passes the same instant to both axes, because a write happens at
        one moment in both clocks — but the parameters stay separate so that this
        predicate is literally the one `candidate_ids` applies, rather than a second
        definition of liveness that could drift from it.
        """
        ...

    def find_by_value(self, tenant: str, value_key: str) -> list[Claim]: ...

    def claims_citing(self, tenant: str, episode_id: str) -> list[Claim]:
        """Every claim whose `sources` names this turn — provenance, backwards.

        In the protocol because the write path needs it on its hot path, not merely for
        auditing: an exactly repeated turn is reinforced rather than re-extracted, and
        finding what to reinforce without this is a scan of the tenant. Redaction makes
        exact repeats common, since two turns differing only inside a redacted span are
        one turn once the redactor has run, so the scan's cost rose with the store and
        the total went quadratic.

        No liveness filter: a retired claim was still extracted from that turn. Callers
        that want only live claims say so.
        """
        ...

    def slot_history(self, tenant: str, fact_key: str) -> list[Claim]:
        """Every claim ever recorded in one slot, oldest first — the audit trail."""
        ...

    def adjacent(self, tenant: str, keys: Sequence[str], *,
                 outgoing: bool = True, incoming: bool = True,
                 predicates: Sequence[str] | None = None,
                 valid_at: datetime | None = None,
                 known_at: datetime | None = None,
                 scopes: Sequence[Scope] | None = None,
                 limit: int = 1000) -> list[Claim]:
        """Claims whose folded subject (outgoing) or folded object (incoming) is in `keys`.

        The primitive multi-hop traversal is built from, and the one lookup no existing
        index could answer. `fact_key` and `value_key` both hash the predicate, so
        neither can be asked "which claims touch entity X" in *either* direction — and
        `subject`/`object` are the raw text somebody typed, which is four spellings of
        one employer. `keys` are therefore *folded* identities (`Claim.subject_key` /
        `Claim.object_key`, i.e. `memvara.entities.entity_key` plus any write-time
        stamp), not surface forms.

        `valid_at`/`known_at` apply the ordinary liveness predicate — retired and
        expired claims excluded on the axis that ends them. A traversal passes the same
        *pair* to every call it makes, because a path stitched from edges believed at
        different times is a connection that never simultaneously held, and that is as
        true of a pair of clocks as of one. Implementations must therefore treat both as
        exact instants and must not substitute their own clock per call; the caller pins
        the pair before the first hop precisely so they cannot.

        **`scopes` must be applied inside `limit`, not after it.** It is the same list
        `candidate_ids` takes — normally `Scope.ancestors()` — and `None` means "no scope
        filter", which is only correct when the caller is going to filter every row
        itself. It exists because doing that turned out to be unsound rather than merely
        slow: on a tenant two people share, another user's claims about the same entity
        fill the page, and the caller's own edges are cut before they can be filtered in.
        Measured on SQLite, one user holding 20 readable claims about a hub: at 15,000
        competing claims the answer came back 19, and at 40,000 it came back **8**. The
        result carries nothing to say it is partial, and its size is a function of a
        *different* user's write volume — so the answer is both wrong and faintly
        informative about them. Re-asking wider only moves the threshold.

        The rule it violates is general: a filter and a limit cannot be split across two
        layers and still yield a correct top-k. Whatever narrows the rows has to run
        where the truncation runs. An empty `scopes` list therefore fails closed and
        matches nothing, exactly as `candidate_ids` does — an unresolved scope is a
        caller bug, and matching everything would be the worst possible response to it.

        `limit` still truncates, so a genuinely enormous hub is still cut. Order that
        truncation deterministically and by something content-derived rather than by
        insertion order, so which claims survive is a function of the data and not of who
        wrote first, and two stores holding the same rows cut the same ones.

        **An empty key names no entity, and neither end of an empty key is adjacency.**
        A retraction stores `''` for the object it retracts, so implementations must
        ignore empty strings in `keys` *and* never match a row on an empty end —
        otherwise `adjacent(t, [""])` returns every retraction in the tenant as one giant
        hub. This is stated here rather than left to storage because the two backends
        represent it differently (SQLite `''`, Postgres `NULL`) and that difference must
        not be observable through this method.

        `predicates` matches the stored `predicate` column exactly; normalizing through
        the registry is the caller's job, and a short list is assumed. Polarity is *not*
        filtered: a negation is genuinely a claim about those entities, and whether it
        may be walked as a link between them is a traversal question, answered once in
        `GraphTraverser` so that no store implementation can get it wrong.
        """
        ...

    def invalidate(self, claim_id: str, at: datetime, by: str | None) -> None: ...

    def set_valid_to(self, claim_id: str, valid_to: datetime | None) -> None:
        """Close (or reopen) a claim's valid-time interval.

        Part of the protocol because `Memvara.forget` calls it directly: both time axes
        must move together or an `as_of` query lands between them and sees an
        inconsistent world.
        """
        ...

    def reinforce(self, claim_id: str, salience: float, observation_count: int,
                  sources: Sequence[str]) -> None:
        """Record that we saw an already-known fact again."""
        ...

    # --- retrieval --------------------------------------------------------
    def set_embedding(self, claim_id: str, vec: np.ndarray) -> None: ...

    def get_embedding(self, claim_id: str) -> np.ndarray | None:
        """Read a stored vector back, so background work never re-embeds text that has
        already been embedded."""
        ...

    def set_episode_embedding(self, episode_id: str, vec: np.ndarray) -> None:
        """Index a raw turn for semantic recall.

        Its own method rather than an overload of `set_embedding`, because claims and
        episodes are separate objects with separate lifecycles and inferring which one
        an id names is exactly the guess that puts a turn in the claims index.
        """
        ...

    def get_episode_embedding(self, episode_id: str) -> np.ndarray | None:
        """A turn's vector, or None if it was never indexed. Also the cheap probe for
        "has this already been embedded?"."""
        ...

    def clear_embeddings(self) -> int:
        """Drop every stored vector and release the dimension they fixed; return how
        many went.

        Part of the protocol because re-embedding cannot be expressed without it: an
        index fixes its width on the first vector it sees, so migrating to a new model
        must empty the store of vectors before writing the first new one. Claims and
        episodes are untouched — this drops derived data, not memory.
        """
        ...

    # The three time-travelling primitives. `valid_at`, `known_at` and
    # `include_invalidated` are keyword-only, deliberately: they replaced a positional
    # `as_of`, and a call site that still passes an instant third would otherwise be
    # silently reinterpreted as `valid_at` — a wrong answer with no error, which is the
    # single failure mode this whole change exists to remove.

    def candidate_ids(self, scopes: Sequence[Scope], *,
                      valid_at: datetime | None = None,
                      known_at: datetime | None = None,
                      include_invalidated: bool = False) -> list[str]: ...

    def lexical_search(self, query: str, scopes: Sequence[Scope], limit: int, *,
                       valid_at: datetime | None = None,
                       known_at: datetime | None = None,
                       include_invalidated: bool = False) -> list[tuple[str, float]]: ...

    def vector_search(self, qvec: np.ndarray, scopes: Sequence[Scope], limit: int, *,
                      valid_at: datetime | None = None,
                      known_at: datetime | None = None,
                      include_invalidated: bool = False) -> list[tuple[str, float]]: ...

    # --- episode retrieval ------------------------------------------------
    #
    # The same three primitives over raw turns. They take no `include_invalidated`:
    # episodes are not bitemporal — nothing retires or supersedes them — so there is no
    # end-of-life to lift.
    #
    # They still take both axes, and the reason is that a turn's single `ts` is *both* of
    # them at once: the turn happened at `ts` and we knew of it at `ts`, so applying the
    # claim rule to an episode gives `ts <= valid_at AND ts <= known_at`, i.e. the
    # earlier of the two bounds. That falls out of the model rather than being imposed on
    # it, and it degenerates to today's behaviour whenever the two agree.
    #
    # Scope filtering is *not* relaxed. It is the same question for raw text as for a
    # derived belief, and raw text is the more sensitive of the two: an unfiltered
    # episode search would hand one session's transcript to a sibling session and one
    # tenant's to another.

    def episode_candidate_ids(self, scopes: Sequence[Scope], *,
                              valid_at: datetime | None = None,
                              known_at: datetime | None = None) -> list[str]: ...

    def lexical_search_episodes(self, query: str, scopes: Sequence[Scope], limit: int, *,
                                valid_at: datetime | None = None,
                                known_at: datetime | None = None
                                ) -> list[tuple[str, float]]: ...

    def vector_search_episodes(self, qvec: np.ndarray, scopes: Sequence[Scope],
                               limit: int, *, valid_at: datetime | None = None,
                               known_at: datetime | None = None
                               ) -> list[tuple[str, float]]: ...

    def purge(self, scope: Scope) -> dict[str, int]:
        """Irreversibly erase a scope: claims, episodes, every vector, both text indexes.

        The one place deletion is correct. Retirement cannot satisfy a legal erasure
        request, because the text stays readable — and an index entry outliving the row
        it describes leaves the purged text searchable, which is the same failure with
        an extra step.
        """
        ...

    def erase_episode(self, episode_id: str, *, cited: bool = False) -> bool:
        """Irreversibly erase one turn — row, text index, vector. Returns whether it
        existed.

        The gap this closes: `erase_claim(sources=True)` reaches a turn only *through* a
        claim, so a turn the extractor found nothing in — an acknowledgement, a greeting,
        anything in a script tier 1 does not handle — is unreachable by any per-claim
        erasure and accumulates forever. `purge` takes a whole scope, which is far too
        blunt for a retention rule over raw transcripts.

        `cited=False` refuses a turn a surviving claim still cites, because erasing it
        leaves `why()` pointing at nothing — the one thing this library promises always
        resolves. `cited=True` erases anyway and is what a retention obligation over
        transcripts needs; the dangling provenance is then a deliberate, recorded
        consequence rather than an accident.
        """
        ...

    def erase_claim(self, claim_id: str, *, sources: bool = False) -> bool:
        """Irreversibly erase one claim — row, text index, vector. Returns whether it
        existed.

        In the protocol rather than left to `purge` because the gap between the two was
        an erasure request naming a single memory, which retirement cannot satisfy (the
        text stays readable) and a scope-wide purge over-answers. An implementation that
        cannot really erase must raise rather than retire: a caller told "deleted" who
        still has the text on disk is the worst outcome this interface can produce.

        `sources=True` also erases the source turns no surviving claim still cites —
        correct for a memory that *is* its source text, wrong for a fact extracted from
        a conversation turn that holds much else besides.
        """
        ...

    # --- learned schema ---------------------------------------------------
    def put_spec(self, spec: "PredicateSpec", tenant: str = "default") -> None:
        """Persist a learned predicate specification. Must survive restart: cardinality
        is what makes a contradiction detectable.

        Scoped to a tenant, because cardinality and volatility are what decide whether a
        claim retires another and how fast it decays — decisions one tenant must not be
        able to make on another's behalf. The default is the tenant `Memvara` uses when
        the caller names none, so a single-tenant caller need not think about it.
        """
        ...

    def all_specs(self, tenant: str = "default") -> list["PredicateSpec"]:
        """Every persisted predicate specification for one tenant."""
        ...

    # --- resolved entities ------------------------------------------------
    def put_entity(self, entity_id: str, canonical: str, aliases: Sequence[str],
                   tenant: str = "default") -> None:
        """Persist "these spellings name one thing".

        Must survive restart, and for a harder reason than the predicate schema: entity
        ids are baked into the `fact_key`s already on disk, so a process that re-derived
        the mapping and disagreed by one id would address a different slot and stop
        seeing the contradiction between two spellings of one subject.

        Tenant-scoped, because one tenant deciding "Acme" and "Acme Corp" are one entity
        must not decide it for another.
        """
        ...

    def all_entities(self, tenant: str = "default") -> list[tuple[str, str, tuple[str, ...]]]:
        """Every resolved entity for one tenant, as (id, canonical, aliases)."""
        ...

    # --- maintenance ------------------------------------------------------
    def iter_claims(self, tenant: str | None = None,
                    include_invalidated: bool = False) -> Iterable[Claim]: ...

    def stats(self, tenant: str | None = None) -> dict[str, int]:
        """Row counts, optionally for one tenant. `embeddings` counts what the store
        holds, not what any one process has mapped."""
        ...

    def close(self) -> None: ...


class SQLStore(Protocol):
    """The two clause builders every SQL-backed `Store` shares. Not part of `Store`.

    Deliberately a second protocol. `Store` is what a third-party backend implements,
    and the docstring at the top of this file promises Qdrant and LanceDB can slot in
    behind it — neither of which generates SQL, so requiring a clause builder of them
    would make `Store` describe an implementation rather than a capability, and would
    change what `isinstance(x, Store)` accepts.

    What it *is* for: SQLite and Postgres write the same predicate twice, in two
    repositories, and the two must agree clause for clause or the same question answers
    differently depending on where the rows live. Naming the shape once, here, is what
    makes that checkable. `memvara.types.Claim.is_live` is the third copy — the Python
    one — and it is held to the same wording.

    Not `@runtime_checkable`: these are private-by-name members, and an `isinstance`
    check that succeeds on a name beginning with an underscore is not a contract anyone
    should be relying on at run time.
    """

    def _live_clause(self, valid_at: datetime | None, known_at: datetime | None,
                     include_invalidated: bool, alias: str = "") -> tuple[str, list]:
        """SQL and bind parameters for "believed at `known_at`, in force at `valid_at`".

        Four constraints, two per axis, and each axis reads only its own clock:

        ==========================  =========  ==================================
        `recorded_at <= known_at`   belief     we had heard it
        `invalidated_at > known_at` belief     we had not yet retracted it
        `valid_from <= valid_at`    world      it had started being true
        `valid_to > valid_at`       world      it had not yet stopped
        ==========================  =========  ==================================

        `None` on either axis means that axis reads the current clock. Substitute it
        with **one** read serving both defaults, not one per axis: two reads put the
        clocks microseconds apart, which is unobservable until the day it is not and
        impossible for a caller to reproduce when it happens.

        `include_invalidated` lifts end-of-life, and lifts more than the name suggests.
        Stated in full here because it must be identical in every backend: the flag
        drops the retirement clause **and the whole valid-time interval**, floor
        included, leaving `recorded_at <= known_at` alone. Two consequences follow, and
        both are intended — under the flag `valid_at` has no effect at all, and the
        belief floor still holds, because returning something first heard in July when
        asked what we believed in March is the one way a bitemporal read can lie.

        `alias` prefixes every column with `"<alias>."` for use in a join, and is empty
        for a single-table statement. The return is `(sql, params)`, with one bind
        parameter per `?` in order — four when the flag is off, one when it is on.
        """
        ...

    def _happened_clause(self, valid_at: datetime | None, known_at: datetime | None,
                         alias: str = "") -> tuple[str, list]:
        """The same, for episodes: one column, `ts`, bounded by the earlier of the two.

        A turn has no separate record time — it happened and we knew it at the same
        instant — so its `valid_from` and its `recorded_at` are both `ts`, and the four
        clauses above collapse to `ts <= valid_at AND ts <= known_at`. Nothing retires a
        turn, so there is no end-of-life half and no `include_invalidated`.
        """
        ...
