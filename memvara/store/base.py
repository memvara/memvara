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
from typing import (TYPE_CHECKING, Collection, Iterable, Literal, Protocol, Sequence,
                    runtime_checkable)

import numpy as np

from ..types import Claim, Episode, Scope

if TYPE_CHECKING:
    # Only for annotations: a `Store` implementation should not have to import
    # the schema module to satisfy the protocol.
    from ..schema import PredicateSpec


# --- the three states, and the SQL that selects them ----------------------------

#: One of the three things a claim can be, in the order `Claim.state` decides them.
#: `live` is neither clock closed, `ended` is valid time closed (the world changed) and
#: `retired` is transaction time closed (the record was wrong). This is the vocabulary
#: `states=` accepts everywhere it appears, and it must stay word-for-word what
#: `Claim.state` returns — a filter naming a state the model does not have, or missing
#: one it does, is a population nobody can ask for.
ClaimState = Literal["live", "ended", "retired"]

#: Every legal state, as one tuple. Doubles as the canonical *order*: a resolved
#: `states=` is always sorted into this order, so one requested population always
#: compiles to one string rather than to however many orderings the caller could have
#: written it in.
STATES: tuple[ClaimState, ...] = ("live", "ended", "retired")

#: What `states=` means when nothing is passed, on the read path — and equally what
#: `include_invalidated=False` means there. See `resolve_states`.
LIVE_ONLY: tuple[ClaimState, ...] = ("live",)

#: The same, for the maintenance walk `iter_claims`, whose unflagged view has always
#: been "every row we still believe" rather than "every row in force right now". See
#: `Store.iter_claims`.
BELIEVED: tuple[ClaimState, ...] = ("live", "ended")


def resolve_states(states: Collection[str] | None = None,
                   include_invalidated: bool | None = None,
                   *, default: Collection[str] = LIVE_ONLY) -> tuple[ClaimState, ...]:
    """Turn a caller's `states=` / `include_invalidated=` into one canonical tuple.

    The single place the alias is applied, so no surface can invent its own reading of
    the older flag. `include_invalidated` is **not deprecated** and emits no warning:
    the core is tagged v0.1.0, `pyproject.toml` sets
    ``filterwarnings = ["error::DeprecationWarning"]``, and a warning here would turn
    every existing call site in the suite red for a spelling that still works perfectly.
    It is an alias, permanently, and nothing more.

    `default` is what the method returns when neither is given, and equally what
    `include_invalidated=False` means on it — they are the same view by definition, which
    is why one parameter names both and they cannot drift apart. It is `("live",)` on the
    read path and `("live", "ended")` on `iter_claims`, whose unflagged view has always
    been the belief axis alone.

    Passing both raises rather than picking one. There is no reading of
    `states=["retired"], include_invalidated=False` in which one of the two is not being
    ignored, and an ambiguous call is a bug at the call site rather than something to
    resolve on the caller's behalf.

    >>> resolve_states()
    ('live',)
    >>> resolve_states(include_invalidated=True)
    ('live', 'ended', 'retired')
    >>> resolve_states(["retired", "live"])          # canonical order, not call order
    ('live', 'retired')
    >>> resolve_states(["retired"], include_invalidated=True)   # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ValueError: pass states= or include_invalidated=, not both...
    """
    if states is not None and include_invalidated is not None:
        raise ValueError(
            "pass states= or include_invalidated=, not both. include_invalidated is an "
            f"alias — False means states={tuple(default)!r} and True means "
            f"states={STATES!r} — so the two spellings disagree about which population "
            "is wanted, and picking either would answer a question you did not ask."
        )
    if include_invalidated is not None:
        return _canonical(STATES if include_invalidated else default)
    return _canonical(default if states is None else states)


def _canonical(states: Collection[str]) -> tuple[ClaimState, ...]:
    """Validate a requested population and put it in `STATES` order."""
    wanted = set(states)
    unknown = sorted((s for s in wanted if s not in STATES), key=repr)
    if unknown:
        raise ValueError(
            f"{unknown[0]!r} is not a claim state. Use any non-empty subset of "
            f"{STATES!r}: 'live' is neither clock closed, 'ended' is a fact that stopped "
            "being true, 'retired' is a record we stopped believing."
        )
    if not wanted:
        raise ValueError(
            f"states= must name at least one of {STATES!r}. An empty set is a filter "
            "that can only return nothing, and a read that silently returns nothing is "
            "the failure this parameter exists to remove."
        )
    return tuple(s for s in STATES if s in wanted)


def _either(*parts: str) -> str:
    """`a OR b`, parenthesised inside *and* out so precedence is never load-bearing.

    The outer parentheses are the ones that matter. `AND` binds tighter than `OR`, so a
    bare disjunction dropped into a conjunction re-associates silently:
    `floor AND retired OR in_force` is `(floor AND retired) OR in_force`, which lets a
    row through on the world clause alone — the belief floor gone, and gone in the
    direction that answers "what did we believe in March" with something first heard in
    July. Nothing fails; the answer is merely from the future.
    """
    return (parts[0] if len(parts) == 1
            else "(" + " OR ".join(f"({p})" for p in parts) + ")")


def state_predicate(at: str = "?", *, states: Collection[str] | None = None,
                    alias: str = "") -> tuple[str, tuple[str, ...]]:
    """SQL for "in one of `states` at `at`", plus the axis each bind marker reads.

    The general form of `live_predicate`, which is this called with one state. `at` is
    the **SQL expression** for the instant and is substituted at every marker, exactly as
    there: a bind marker (`"?"`, `"%s"`) or a server clock (`"now()"`).

    The second element is the half `live_predicate` could only document. It names the
    clock behind each marker in order — `("known", "known", "valid", "valid")` for the
    live-only case, which is the bind order that function's docstring states — so a
    backend binds by *reading* it rather than by knowing it, and the one error a diagonal
    query cannot reveal (belief instant bound onto the world columns, invisible whenever
    `valid_at == known_at`) stops being expressible. Every combination keeps the belief
    markers ahead of the world markers, so the discipline generalises rather than
    changing shape per subset.

    **The three states do not tile the store, and asking for all three is the audit
    view.** A claim recorded but not yet in force at `valid_at` — a fact scheduled to
    start next month — is named by none of them, because `Claim.state` is absolute while
    this predicate is as-of. So the complete set does not compile to the union of its
    parts: it compiles to the belief floor alone, which readmits that row and leaves
    `valid_at` with nothing to constrain. That is exactly, and deliberately, the
    semantics `include_invalidated=True` has always had.

    >>> sql, axes = state_predicate("now()")
    >>> print(sql.replace(" AND ", "\\n AND "))
    (recorded_at <= now()
     AND (invalidated_at IS NULL OR invalidated_at > now())
     AND valid_from <= now()
     AND (valid_to IS NULL OR valid_to > now()))
    >>> axes
    ('known', 'known', 'valid', 'valid')
    >>> state_predicate("?", states=["retired"])
    ('(recorded_at <= ? AND invalidated_at IS NOT NULL AND invalidated_at <= ?)', ('known', 'known'))
    >>> state_predicate("%s", states=STATES, alias="c")
    ('(c.recorded_at <= %s)', ('known',))
    """
    a = f"{alias}." if alias else ""
    wanted = resolve_states(states)
    floor = f"{a}recorded_at <= {at}"
    if len(wanted) == len(STATES):
        return f"({floor})", ("known",)

    # The world half, per requested subset of the two world-time states. `live ∪ ended`
    # collapses to the valid-time *floor*: a closure never lands before the interval it
    # closes (see `types.close_out`), so `valid_to <= V` already implies `valid_from <= V`
    # and the two intervals between them cover everything that had started by `V`.
    world, world_axes = {
        ("live",): (f"{a}valid_from <= {at} "
                    f"AND ({a}valid_to IS NULL OR {a}valid_to > {at})",
                    ("valid", "valid")),
        ("ended",): (f"{a}valid_to IS NOT NULL AND {a}valid_to <= {at}", ("valid",)),
        ("live", "ended"): (f"{a}valid_from <= {at}", ("valid",)),
        (): ("", ()),
    }[tuple(s for s in wanted if s != "retired")]
    axes = ("known", "known") + world_axes

    if "retired" not in wanted:
        not_retired = f"({a}invalidated_at IS NULL OR {a}invalidated_at > {at})"
        return f"({floor} AND {not_retired} AND {world})", axes
    # With `retired` wanted, the "still believed" guard on the other half is redundant:
    # the two are exact complements, so `retired OR (believed AND in_force)` is just
    # `retired OR in_force`. The retired disjunct is written first so its belief marker
    # stays ahead of the world markers.
    retired = f"{a}invalidated_at IS NOT NULL AND {a}invalidated_at <= {at}"
    return f"({floor} AND {_either(*filter(None, (retired, world)))})", axes


def stored_state_predicate(states: Collection[str] | None = None, *,
                           prefix: str = "") -> str:
    """`Claim.state` as SQL: the two closure columns, and no instant at all.

    The member of this family for a walk that has no clock to read. `iter_claims` pages
    over rows rather than answering a question about a moment, and its filter has always
    been the *stored* state — which is what `Claim.state` reports, and which differs from
    `state_predicate` exactly where a timestamp is in the future: a claim retired next
    October is `retired` here and still believed there.

    Returns `""` for the complete set, meaning "no filter" — the caller drops it rather
    than emitting a tautology into a paged scan.

    `prefix` is placed before every column name. `iter_claims` passes `"+"`, which is
    load-bearing rather than decorative: see `SQLiteStore._iter_rows` for why an
    index-usable term there makes a full walk quadratic.

    >>> stored_state_predicate(["retired"])
    'invalidated_at IS NOT NULL'
    >>> stored_state_predicate(["live"], prefix="+")
    '+invalidated_at IS NULL AND +valid_to IS NULL'
    >>> stored_state_predicate(STATES)
    ''
    """
    wanted = resolve_states(states)
    if len(wanted) == len(STATES):
        return ""
    return _either(*(
        {
            "live": f"{prefix}invalidated_at IS NULL AND {prefix}valid_to IS NULL",
            "ended": f"{prefix}invalidated_at IS NULL AND {prefix}valid_to IS NOT NULL",
            "retired": f"{prefix}invalidated_at IS NOT NULL",
        }[s]
        for s in wanted
    ))


def live_predicate(at: str = "?", *, include_invalidated: bool = False,
                   alias: str = "") -> str:
    """SQL for "believed and in force at `at`", built without a store instance.

    The one exported spelling of the four-column liveness test. Three surfaces outside
    the store had each written it by hand and all three had written the *cheap* version,
    `invalidated_at IS NULL`. That selected the same rows only while superseding closed
    both clocks. It closes valid time alone now, so the column test counts every
    superseded version of every slot as live: a store whose one address has changed four
    times reports five. Nothing fails when that happens — one of the three copies was a
    billing gauge, where the step is unfalsifiable from inside the data because no other
    series moves with it.

    `at` is the **SQL expression** for the instant and is substituted at every axis: a
    bind marker (`"?"` for SQLite, `"%s"` for psycopg) or a server clock (`"now()"`).
    One expression rather than one per axis, deliberately. Every caller that cannot
    reach a store instance is counting *now*, and two markers would let one bind the
    pair transposed — belief instant onto the world columns — which is the single error
    a diagonal query cannot reveal, since `valid_at == known_at` answers the same either
    way.

    With a bind marker, the four markers take their values in the order **known, known,
    valid, valid**. That order is the half this function cannot enforce, so a backend
    should bind it in `_live_clause` and nowhere else — `SQLiteStore._live_clause` is
    the only place in this repository that does.

    `include_invalidated` lifts end-of-life *and the whole valid-time interval* with it,
    leaving one marker for the belief floor — see `SQLStore._live_clause` for why that
    floor is the clause which never lifts. `alias` prefixes every column, for a join.

    **This is now the two-state alias of `state_predicate`**, which takes the states
    themselves and can therefore express the one population this spelling cannot: the
    records we stopped believing, alone, which is what a correction audit reads. Both
    remain supported and neither is deprecated; this one drops the axis list, so a
    caller that needs to bind markers should prefer the general form.

    >>> print(live_predicate("now()").replace(" AND ", "\\n AND "))
    (recorded_at <= now()
     AND (invalidated_at IS NULL OR invalidated_at > now())
     AND valid_from <= now()
     AND (valid_to IS NULL OR valid_to > now()))
    >>> live_predicate("%s", include_invalidated=True, alias="c")
    '(c.recorded_at <= %s)'
    """
    return state_predicate(
        at, states=resolve_states(include_invalidated=include_invalidated), alias=alias,
    )[0]


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

    def count_competing(self, tenant: str, fact_key: str, *,
                        valid_at: datetime | None = None,
                        known_at: datetime | None = None) -> int:
        """`len(competing_claims(...))`, without building the claims to count them.

        Same slot, same two axes, same liveness predicate — only the answer is smaller.
        It exists because one caller wants the size of a slot and never its contents:
        `Reconciler._accumulation`, which reports a value landing beside values already
        live under a predicate whose cardinality nobody declared. That report is only
        ever needed for multi-valued slots, so the number it needs is exactly the one
        that grows without bound, and hydrating a claim per occupant to count them made
        the write cost rise with the pathology it exists to describe: measured at 20µs
        for one occupant, 1.8ms for 200 and 29ms for 3,000, against 4.7µs, 28µs and
        434µs for the count.

        Optional, and reached through `getattr` by its one caller, which falls back to
        counting `competing_claims` — a `Store` written before this method keeps working
        and merely pays what it used to. Kept out of `competing_claims` rather than
        offered as a `count_only=` flag on it, because a method whose return type depends
        on an argument is worse to type and worse to read than two methods.
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

    # The two targeted column writes. **No engine path calls either one**, and that is a
    # decision rather than an oversight: every write that ends a claim now goes through
    # `types.close_out` plus `put_claim`, because closing a claim means moving exactly one
    # clock and neither of these can express that on its own. They are kept because they
    # are the only single-statement writes in the protocol — a whole-row upsert is a
    # different cost and a different concurrency story on a server backend — and because
    # `set_valid_to` can do one thing no write path can. Both are exercised by the store
    # suites in this repository and in the Postgres backend, which is what keeps a
    # third-party implementation of them honest.

    def invalidate(self, claim_id: str, at: datetime, by: str | None) -> None:
        """Stop believing a claim at `at`, recording `by` as what displaced it.

        **Not the way to record a supersession.** It writes `invalidated_at` and
        `invalidated_by` in one statement, and `invalidated_at` means *the record was
        wrong* — so using it to note that a newer value arrived marks a claim that was
        true, and is still believed, as an error. That was `Reconciler._retire`'s bug and
        it is this signature's shape: the pointer and the belief stamp cannot be
        separated here, which is precisely why the reconciler stopped using it. Closing
        valid time instead is `types.close_out(claim, at, by, "ended")` followed by
        `put_claim`.

        What it *is* for: retiring a row in one statement, when the caller really does
        mean "we no longer believe this" — a repair tool, a retention job, a backend's
        own maintenance. `by=None` is the ordinary case there, since nothing replaced it.
        """
        ...

    def set_valid_to(self, claim_id: str, valid_to: datetime | None) -> None:
        """Close, move, or **reopen** a claim's valid-time interval.

        The reopen is why this survives having no engine caller: `valid_to=None` clears
        an end, and `close_out` cannot express that in either direction — it only ever
        moves an end *earlier*, never clears one, because a write path that could reopen
        an interval could silently un-end a fact. Undoing a mistaken end is therefore an
        explicit, targeted act, which is the right shape for it.

        Writes one column and no pointer, so unlike `invalidate` it cannot conflate the
        two axes. It also cannot record *what* replaced the claim; a supersession needs
        `invalidated_by` as well, and `put_claim` is the only way to write both without
        also writing `invalidated_at`.
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

    # The three time-travelling primitives. `valid_at`, `known_at`, `states` and
    # `include_invalidated` are keyword-only, deliberately: they replaced a positional
    # `as_of`, and a call site that still passes an instant third would otherwise be
    # silently reinterpreted as `valid_at` — a wrong answer with no error, which is the
    # single failure mode this whole change exists to remove.
    #
    # **`states` names the population, `include_invalidated` is its two-valued alias.**
    # The flag is one boolean over three states, so it can say "live" and "all of them"
    # and nothing else — and the population it cannot name is the one a correction audit
    # reads, the records we stopped believing. Client-side filtering is not the way out:
    # these are paginated, so filtering after `limit` under-returns silently.
    #
    # `None` on `states` means the method's own default (`("live",)` here), `None` on the
    # flag means "not passed", and passing both raises. Nothing is deprecated — see
    # `resolve_states`, which is the one place either is interpreted.

    def candidate_ids(self, scopes: Sequence[Scope], *,
                      valid_at: datetime | None = None,
                      known_at: datetime | None = None,
                      states: Collection[str] | None = None,
                      include_invalidated: bool | None = None) -> list[str]: ...

    def lexical_search(self, query: str, scopes: Sequence[Scope], limit: int, *,
                       valid_at: datetime | None = None,
                       known_at: datetime | None = None,
                       states: Collection[str] | None = None,
                       include_invalidated: bool | None = None
                       ) -> list[tuple[str, float]]: ...

    def vector_search(self, qvec: np.ndarray, scopes: Sequence[Scope], limit: int, *,
                      valid_at: datetime | None = None,
                      known_at: datetime | None = None,
                      states: Collection[str] | None = None,
                      include_invalidated: bool | None = None
                      ) -> list[tuple[str, float]]: ...

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

    def erase_claim(self, claim_id: str, *, sources: bool = False) -> dict[str, int]:
        """Irreversibly erase one claim — row, text index, vector. Returns per-table
        counts, the same four keys `purge` returns: `claims`, `episodes`, `embeddings`,
        `entities`.

        **The same shape as `purge` because it is the same kind of answer.** Both are
        erasure paths and both are asked to evidence what they erased; this one used to
        report a bare `bool`, so of the two the per-claim path — the one an individual
        erasure request actually names — gave the weaker evidence. `claims` is 0 or 1 and
        is the "did anything happen" the boolean used to carry, so an unknown id erases
        nothing and erasing twice erases once, in counts rather than in flags.

        In the protocol rather than left to `purge` because the gap between the two was
        an erasure request naming a single memory, which retirement cannot satisfy (the
        text stays readable) and a scope-wide purge over-answers. An implementation that
        cannot really erase must raise rather than retire: a caller told "deleted" who
        still has the text on disk is the worst outcome this interface can produce.

        `sources=True` also erases the source turns no surviving claim still cites —
        correct for a memory that *is* its source text, wrong for a fact extracted from
        a conversation turn that holds much else besides. Those turns are what `episodes`
        counts, so it is 0 without the flag.
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
                    include_invalidated: bool | None = None, *,
                    states: Collection[str] | None = None) -> Iterable[Claim]:
        """Every claim, optionally for one tenant, optionally in named states.

        **This one filters the *stored* state, not the state at an instant** — see
        `stored_state_predicate`. It is a walk over rows rather than a question about a
        moment, so it has no clock to read and reports what `Claim.state` reports.

        Its unflagged view is consequently `("live", "ended")`, not `("live",)`: the
        default has always been "every row we still believe", and it must stay that way.
        `reembed()` walks this, and a default that dropped ended claims would silently
        stop re-encoding every superseded version in the store — which is the same walk
        the whole method exists to make affordable. `include_invalidated=True` is
        unchanged and still means all three.

        `include_invalidated` stays positional here because it always was.
        """
        ...

    def stats(self, tenant: str | None = None) -> dict[str, int]:
        """Row counts, optionally for one tenant. `embeddings` counts what the store
        holds, not what any one process has mapped.

        **`live_claims` and `ended_claims` are the full state predicates —
        `state_predicate()` in this module — and not column tests.** `live_claims` and
        `invalidated_at IS NULL` counted the same rows only while superseding closed both
        clocks; since it closes valid time alone, the column test reports every version a
        slot ever held as live.

        `ended_claims` counts state `ended`: the world moved on and we still believe the
        record. It is emphatically **not** `valid_to IS NOT NULL`, because a claim that
        ended and was *later* retired is already inside `invalidated`, and counting it
        twice is precisely how a caller trying to derive this number by subtraction
        (`claims - live_claims - invalidated`) gets a wrong one. That subtraction is why
        the key exists: the largest non-live population was the only one not reported,
        and it was not derivable either.

        A consequence a backend must reproduce rather than round off: **the counts do not
        sum.** A claim recorded but not yet in force — scheduled to start next month — is
        in none of the three populations, `claims` is the only total that covers
        everything, and a store implementation that "corrects" the arithmetic has
        reintroduced the conflation.
        """
        ...

    def close(self) -> None: ...


class SQLStore(Protocol):
    """The three clause builders every SQL-backed `Store` shares. Not part of `Store`.

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

    def _state_clause(self, valid_at: datetime | None, known_at: datetime | None,
                      states: Collection[str] | None = None,
                      alias: str = "") -> tuple[str, list]:
        """SQL and bind parameters for "in one of `states` at (`valid_at`, `known_at`)".

        The parameterised form of `state_predicate`, and the method every read filter in
        a SQL backend should route through. The SQL and the *axis list* both come from
        there, so binding is a comprehension over the axes rather than a remembered
        order — which is what makes the one silent error impossible to write. See
        `state_predicate` for what each subset means, including why the complete set is
        the belief floor alone rather than the union of its parts.

        `_live_clause` below is this with the two-valued alias applied.
        """
        ...

    def _live_clause(self, valid_at: datetime | None, known_at: datetime | None,
                     include_invalidated: bool, alias: str = "") -> tuple[str, list]:
        """SQL and bind parameters for "believed at `known_at`, in force at `valid_at`".

        The parameterised form of `live_predicate` — the SQL should come from there
        rather than be written out again, so that a backend and a counter that cannot
        reach a store instance cannot drift apart. What this adds is the binding, and
        the binding is the half with a silent failure mode: see `live_predicate`.

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
