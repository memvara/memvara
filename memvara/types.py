"""Core data model.

The central departure from mem0 is that a memory is not an opaque string. It is a
`Claim`: a structured, bitemporal assertion with provenance. That structure is what
makes deterministic contradiction resolution and time-travel queries possible; you
cannot do either over free text without asking an LLM every time.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar, Literal, cast

from .entities import OWNER_SEP, entity_key


#: How coarse a resolved temporal boundary is. See `Claim.temporal_precision`.
Precision = Literal["instant", "day", "week", "month", "season", "year"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def as_utc(dt: datetime) -> datetime:
    """Treat a naive datetime as UTC rather than rejecting it.

    Callers build these by hand at API edges and in tests, and the store round-trips
    them through epoch floats, so both kinds arrive. A TypeError raised from inside
    ranking or decay is a far worse outcome than assuming the convention every
    persisted timestamp already follows.
    """
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def time_axes(as_of: datetime | None, valid_at: datetime | None,
              known_at: datetime | None) -> tuple[datetime | None, datetime | None]:
    """Resolve the three time keywords every read method takes into the two axes.

    Bitemporal data answers four questions, and one instant can only ask for one of
    them:

    ======================================  ===========================
    what do we now believe is true now      neither keyword
    what do we now believe was true in June ``valid_at=June``
    what did we believe on 1 August         ``known_at=August``
    what did we believe in June, about June ``as_of=June``
    ======================================  ===========================

    `valid_at` is the world clock and `known_at` the belief clock; each defaults to
    `None`, which every consumer reads as "now". `as_of` moves both together and is
    therefore *exact* sugar for the fourth row — kept because it is the published
    spelling, and correct because that row is a real question, just not the only one.

    The row that motivated the split is the second. A correction learned in August
    about June is invisible to `as_of=June`, because that call rewinds the belief clock
    past the correction — so the late-arriving fact, which is the entire reason to keep
    two axes, could not be asked about at all.

    Mixing `as_of` with either axis raises. There is no reading of
    `as_of=June, known_at=August` that is not one of the two being ignored, and
    silently picking either would answer a question the caller did not ask, with
    nothing in the result to say so.

    >>> time_axes(None, None, None)
    (None, None)
    >>> june = datetime(2026, 6, 1, tzinfo=timezone.utc)
    >>> time_axes(june, None, None) == (june, june)
    True
    >>> time_axes(june, None, june)      # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ValueError: as_of cannot be combined with known_at...
    """
    if as_of is None:
        return valid_at, known_at
    if valid_at is None and known_at is None:
        return as_of, as_of
    clash = " and ".join(
        name for name, value in (("valid_at", valid_at), ("known_at", known_at))
        if value is not None
    )
    raise ValueError(
        f"as_of cannot be combined with {clash}. as_of moves both clocks to one "
        "instant and is exactly valid_at=known_at=as_of, so the two spellings "
        "disagree about the question being asked; drop as_of to move the clocks apart."
    )


# --- memory dynamics -----------------------------------------------------------
# Two `meta` keys, named here rather than in the subsystems that write them, because
# three of them read these: the write path (reinforcement), consolidation (decay and
# merge) and retrieval (recency). They live in `meta` because it is already persisted
# as JSON, so the model below needed no store migration to land.

#: Undecayed *storage* strength. See `Claim.salience_base`.
SALIENCE_BASE = "salience_base"
#: Epoch seconds of the last independent re-observation. See `Claim.last_observed`.
LAST_OBSERVED = "last_observed_at"

# --- entity identity ------------------------------------------------------------
# Two more `meta` keys, holding the entity a claim's subject and object resolved to when
# it was written. `Reconciler` stamps them; `fact_key` and `value_key` read them.
#
# They are *frozen at write time* on purpose. Without a stamp these keys would be a pure
# function of the current alias table, so learning in month six that "Big Blue" is IBM
# would retroactively restructure month one: claims that had coexisted would start
# retiring each other and `slot_history()` would return a differently-shaped past.
# Nothing would be deleted, but "append-only *and* stable" would quietly become
# "append-only". With the stamp, a claim keeps the identity it was written with, and
# applying a late alias to existing rows is an explicit, dated, dry-run-first operation —
# see `memvara.write.reconcile.backfill_entities`.

#: The subject a first-person statement is filed under. Written by the deterministic
#: matcher (`write/fast.py`) and by a model extraction that named no subject
#: (`write/pipeline.py`), and read by `retrieve/anchor.py`, which has to know that "where
#: do I live" is a question about this row. One spelling, so the three cannot drift.
SELF_SUBJECT = "user"

#: Resolved entity of `Claim.subject`. See `memvara/entities.py`.
SUBJECT_ENTITY = "subject_entity"
#: Resolved entity of `Claim.object`.
OBJECT_ENTITY = "object_entity"
#: Timestamped record of every backfill that changed a claim's place in history.
ENTITY_REKEY = "entity_rekey"

# --- closure witness ------------------------------------------------------------
#: Timestamped record of every closure applied to a claim: which clock stopped, when, and
#: what displaced it. Written by `close_out`, which is the single place any claim ends.
#:
#: **The columns remain authoritative; this is a witness, not a second source of truth.**
#: Every query reads the columns — `state_predicate` never consults `meta`, and nothing in
#: the read path does. What this adds is evidence that survives the columns being wrong.
#:
#: That is not a hypothetical failure. Before commit `0c88a92`, `delete()` overwrote
#: `invalidated_by` with `None`, so a claim that had been superseded and was later deleted
#: lost the only record of what replaced it. Rows damaged that way are now permanently
#: unclassifiable: both clocks are closed and nothing distinguishes a supersession whose
#: pointer was erased from an ordinary retraction. The cloud's closure backfill has to
#: refuse that entire population, and says so. A witness written at the moment of closure
#: is what stops the set growing — a corrupted column can then *disagree* with the record,
#: and the disagreement is the finding rather than the dead end.
#:
#: A list, appended to, because a claim can end and later be retired: the world moved on,
#: and afterwards we decided we had been wrong to record it at all. A scalar would keep
#: only the second and lose exactly the fact that is hardest to recover. Same shape as
#: `ENTITY_REKEY` for the same reason.
CLOSURE = "closure"

#: The `meta` keys above, as one set: everything in `Claim.meta` that the engine owns
#: rather than the caller. Two surfaces need it and they need it for opposite reasons —
#: `Memvara.remember` **rejects** them on the way in, because `salience_base` reaching
#: the store is a permanent ranking override no documented argument can produce, and
#: `compat.mem0` **filters** them on the way out, so a compatibility layer never hands
#: internal bookkeeping back as user data. Defined here, beside the five constants, so
#: adding a sixth cannot leave one of those two surfaces behind.
RESERVED_META = frozenset({
    SALIENCE_BASE, LAST_OBSERVED, SUBJECT_ENTITY, OBJECT_ENTITY, ENTITY_REKEY, CLOSURE,
})

#: Decimal places kept on a stored salience. Salience is a ranking weight, not an
#: accounting figure, and quantizing kills the sub-nanosecond drift between two
#: scheduler ticks that would otherwise make an idempotent pass look like a change.
SALIENCE_PRECISION = 6

#: Ceiling on salience, however it was earned. Reinforcement and merge both raise it and
#: both have to stop at the same place: two ceilings that drift apart is how one path
#: quietly becomes the way to outrank everything else forever.
MAX_SALIENCE = 5.0

#: `.kind` discriminators for the two things a search can return. A retrieved fact and
#: a retrieved conversation turn are different types on purpose (see `Result` and
#: `HybridRetriever.EpisodeResult`); these are how the distinction survives being
#: serialized, where `isinstance` cannot follow it.
CLAIM = "claim"
EPISODE = "episode"

# --- closing a claim out --------------------------------------------------------

#: Which clock stops when a write ends a claim's life, named by the state it leaves
#: behind (see `Claim.state`). The two are not interchangeable and the difference is the
#: whole bitemporal model:
#:
#: ``"ended"``    valid time closes. **The world changed.** The claim was true and is not
#:                any more, and we still believe every word of it — which is what keeps it
#:                answering `valid_at=<back then>`.
#: ``"retired"``  transaction time closes. **The record was wrong.** We have stopped
#:                believing it, so it answers nothing at any world-time from here on; the
#:                row still says what it always said, which is what an audit reads.
#:
#: Each closure moves exactly one axis, and neither ever moves both. Closing valid time
#: on a mistaken row would assert a world event nobody witnessed; closing transaction
#: time on a superseded row calls a true record an error. Supersession — the only thing
#: the reconciler is ever told about — is always the first kind, so ``"ended"`` is the
#: default everywhere this appears.
Closure = Literal["ended", "retired"]

#: Every legal `Closure`, for validation and for error messages that can list the options.
CLOSURES: tuple[Closure, ...] = ("ended", "retired")


def closure(value: str) -> Closure:
    """Validate a caller-supplied `close=`, or raise with both readings spelled out.

    A typo would otherwise be silently indistinguishable from the default, and the two
    values mean opposite things about whether a stored fact was ever true — which is the
    one mistake in this library that cannot be found by reading the data afterwards.

    >>> closure("retired")
    'retired'
    >>> closure("deleted")                       # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ValueError: close='deleted' is not a closure...
    """
    if value in CLOSURES:
        return cast(Closure, value)
    raise ValueError(
        f"close={value!r} is not a closure. Use close='ended' when the world changed — "
        "the claim was true and stopped being true, and we still believe it — or "
        "close='retired' when the record was wrong and we no longer believe it at all."
    )


def close_out(claim: "Claim", at: datetime, by: str | None, close: Closure) -> None:
    """Stamp one end-of-life onto a claim in memory. The caller persists it.

    The single implementation of the rule above, shared by every path that ends a claim:
    the reconciler's supersessions and retractions, `Memvara.forget`, `Memvara.delete`,
    `Memvara.supersede` and the mem0 importer. Six callers wrote this out for themselves
    before the axes were separated, and every one of them wrote it the same wrong way —
    which is the argument for there being one of it.

    `at` is the instant on whichever axis `close` names: when the world moved for
    `"ended"`, when belief stopped for `"retired"`.

    `by` is the claim that displaced this one, and is written under either closure,
    because "this is what displaced me" is true whichever clock stopped. **`None` means
    "nothing replaced this", not "forget what did"** — `forget` and `delete` pass it
    because they name no successor, and writing it through would erase a pointer set
    earlier. That is not hypothetical: deleting an already-superseded claim cut the link
    between a value and the one that replaced it, so `why(successor).superseded` came
    back empty. In a store whose premise is that nothing vanishes without a trace, done
    by the operation documented as the reversible one.

    The object is stamped, not just the database. `forget()` used to hand back claims
    read *before* its update ran, so every claim the call had just closed out reported
    itself live to anyone who logged or rendered the return value.
    """
    if by is not None:
        claim.invalidated_by = by
    if close == "retired":
        claim.invalidated_at = as_utc(at)
        _witness(claim, claim.invalidated_at, by, close)
        return
    # Never before the claim's own start: a closure backdated past the fact it closes
    # collapses the interval to zero length rather than inverting it. An interval that
    # ends before it begins is not a shorter fact, it is a row no `as_of` window can
    # return consistently.
    edge = max(as_utc(at), as_utc(claim.valid_from))
    landed = claim.valid_to
    if landed is None or as_utc(landed) > edge:
        claim.valid_to = landed = edge
    _witness(claim, as_utc(landed), by, close)


def _witness(claim: "Claim", at: datetime, by: str | None, close: Closure) -> None:
    """Append the closure to `meta[CLOSURE]`. See that constant for why this exists.

    `at` is the instant that actually **landed on the axis**, not the one requested — the
    valid-time clamp above can move it, and a witness that disagreed with the column it
    describes would be worse than none. `by` is recorded even though `invalidated_by`
    holds it, because that column is precisely the one a later `delete()` was found
    overwriting; a pointer in two places survives one of them being erased.

    Re-closing an already-ended claim at a later instant appends an entry whose `at` is
    the *existing* `valid_to`, because that is still where the claim ends. The row says
    a closure was applied and where the axis stands, which is what a reader needs.
    """
    claim.meta.setdefault(CLOSURE, []).append(
        {"at": at.timestamp(), "close": close, "by": by})


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


def _short(text: str, limit: int = 48) -> str:
    """One-line, length-capped rendering of arbitrary stored text.

    Everything in this module can hold user-supplied text of any length containing
    newlines, so a `__repr__` that interpolates it raw is exactly as unreadable as the
    dataclass repr it replaces — worse, it can span the terminal.
    """
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def content_hash(*parts: str) -> str:
    h = hashlib.blake2b(digest_size=16)
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def owner_key(scope: "Scope") -> str:
    """Whose facts these are: tenant plus user, and deliberately nothing else.

    Extraction yields a generic subject ("user"), so the scope's owner is what actually
    distinguishes people — without it, two users in one tenant collide and Bob's "lives
    in Lisbon" silently retires Alice's "lives in Berlin".

    Agent and session are excluded on purpose: a durable fact about a person is the same
    fact no matter which agent or session observed it, so learning "I moved to Lisbon" in
    a fresh session must still retire the old city.
    """
    return f"{scope.tenant}{OWNER_SEP}{scope.user or ''}"


def default_entity(surface: str) -> str:
    """The identity a surface form has when nothing has been learned about it.

    A pure function of the text, which is what keeps `Memvara.history("user",
    "works_at")` working: that call builds a probe `Claim` with no meta and no registry
    in reach, and it still lands on the same key as the stored claims, because they were
    resolved by this same fold.

    An unfoldable surface ("...", an emoji, a bare separator) keeps its raw text rather
    than collapsing to the empty string, which would make every such value one value.
    """
    return entity_key(surface) or surface


def resolved_entity(meta: dict[str, Any], meta_key: str, surface: str) -> str:
    """Identity of one end of a claim: its stamp if it has one, else the fold.

    Only an *alias* produces a stamp — see `Reconciler._stamp`. Everything the
    deterministic fold resolves is unstamped, so the common claim carries no entity
    bookkeeping in its `meta` at all, and a probe built without one agrees with it.
    """
    stamped = meta.get(meta_key)
    if isinstance(stamped, str) and stamped:
        return stamped
    return default_entity(surface)


def fact_key_for(scope: "Scope", subject: str, predicate: str) -> str:
    """Slot identity for an arbitrary (scope, subject, predicate).

    Anything that needs a fact key for a predicate other than a claim's own — notably
    cross-predicate supersession, where asserting `unemployed` must retire `works_at` —
    must go through here. Recomputing the hash by hand is how the two silently drift
    apart and a lookup starts matching nothing.

    The subject is folded to its entity identity on the way in, so no caller can derive
    a key from a raw surface form by forgetting to. `entity_key` is idempotent, so
    passing an already-resolved identity (as `Claim.fact_key` does) is safe.
    """
    return content_hash(owner_key(scope), entity_key(subject) or subject, predicate)


class MemoryType(str, Enum):
    """Different kinds of memory decay and retrieve differently.

    EPISODIC   - something that happened at a point in time ("shipped v2 on Tuesday")
    SEMANTIC   - a durable fact about the world ("works at Acme")
    PROCEDURAL - how the user wants things done ("always use pytest, never unittest")
    """

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class Derivation(str, Enum):
    """How a claim came to exist. Kept for provenance and for eval slicing."""

    USER = "user"                  # asserted directly through the API
    FAST_PATH = "fast_path"        # deterministic extraction, no LLM
    LLM_EXTRACT = "llm_extract"    # structured extraction by an LLM
    CONSOLIDATION = "consolidation"  # derived by merging or promoting other claims


@dataclass(frozen=True, slots=True)
class Scope:
    """Hierarchical addressing: tenant > user > agent > session.

    Visibility widens *upward only*. A search at session scope also sees that agent's,
    that user's, and the tenant's memory (see `ancestors`), but a search at user scope
    does not descend into individual sessions, and nothing ever reaches sideways into a
    sibling session, a sibling agent, or another user.

    That direction is the useful one: a session should answer from what the user said
    months ago, while its own scratch state stays out of everyone else's results.
    mem0's flat user_id/agent_id/run_id triple cannot express the distinction.
    """

    tenant: str = "default"
    user: str | None = None
    agent: str | None = None
    session: str | None = None

    def key(self) -> str:
        return "/".join([self.tenant, self.user or "*", self.agent or "*", self.session or "*"])

    def ancestors(self) -> list["Scope"]:
        """This scope plus every broader scope it inherits from, narrowest first."""
        out = [self]
        if self.session is not None:
            out.append(replace(self, session=None))
        if self.agent is not None:
            out.append(replace(self, agent=None, session=None))
        if self.user is not None:
            out.append(Scope(tenant=self.tenant))
        # De-duplicate while preserving order.
        seen: set[str] = set()
        uniq = []
        for s in out:
            if s.key() not in seen:
                seen.add(s.key())
                uniq.append(s)
        return uniq

    def __repr__(self) -> str:
        return f"<Scope {self.key()}>"

    def sees(self, other: "Scope") -> bool:
        """True if a reader at this scope may read a claim written at `other`.

        The enumeration rule, stated once so it can be shared. `get_all`, `count` and
        `search` already implement it in SQL via `ancestors()`: a handle sees its own
        scope and every broader one, and never a deeper one. This is that same predicate
        for the id-addressed reads, which used to authorize with `contains` — the
        opposite direction — so `get()` and `why()` answered for claims that `get_all()`
        on the identical handle would not return. Two answers to one question, and the
        permissive one was reachable: with agents isolated by `agent=`, a handle scoped
        to a session could read a sibling agent's claim by id, and ids are not secret —
        receipts, `invalidated_by` pointers, results and logs all leak them.

        `contains` is still right for `forget` and `history`, which are slot operations
        where a broad caller reaching downward is the documented intent.
        """
        mine = {s.key() for s in self.ancestors()}
        return other.key() in mine

    def contains(self, other: "Scope") -> bool:
        """True if `other` is at or beneath this scope.

        The downward-reaching predicate: an unset field is a wildcard. Use `sees` for
        read authorization — see there for why the two must not be confused.
        """
        if self.tenant != other.tenant:
            return False
        for mine, theirs in (
            (self.user, other.user),
            (self.agent, other.agent),
            (self.session, other.session),
        ):
            if mine is not None and mine != theirs:
                return False
        return True


@dataclass(slots=True)
class Episode:
    """Raw source material. Claims point back at these, so every memory is traceable.

    mem0 keeps a history table of memory mutations but does not durably retain the
    source text a memory was derived from, which makes "why do you believe this?"
    unanswerable after the fact.
    """

    content: str
    scope: Scope = field(default_factory=Scope)
    role: str = "user"
    ts: datetime = field(default_factory=utcnow)
    id: str = field(default_factory=lambda: _new_id("ep"))
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Alias of `content`.

        `Claim`, `Result` and `Retrieved` all expose `.text`; this exposed only
        `.content`. Those four are exactly what a caller holds while writing an audit
        query — `why()` returns `Provenance.episodes`, and the obvious
        `[e.text for e in prov.episodes]` was an `AttributeError` on the one member of
        the set that spelled it differently. `content` stays the field; this is the
        name the other three taught the reader to expect.
        """
        return self.content

    @property
    def hash(self) -> str:
        return content_hash(self.scope.key(), self.role, self.content)

    def __repr__(self) -> str:
        return (f"<Episode {self.id} {self.scope.key()} {self.role} "
                f"{self.ts:%Y-%m-%d %H:%M}Z {_short(self.content)!r}>")


@dataclass(slots=True)
class Claim:
    """A bitemporal assertion.

    Two independent time axes, which is the thing almost every agent memory layer
    gets wrong by collapsing them into one `updated_at` column:

      valid time       (valid_from / valid_to)   - when the fact was true in the world
      transaction time (recorded_at / invalidated_at) - when *we* believed it

    Keeping them separate is what lets you ask both "where does she live now?" and
    "on March 1st, where did we think she lived?" - and lets a late-arriving fact
    correct the past without rewriting history.
    """

    subject: str
    predicate: str
    object: str

    scope: Scope = field(default_factory=Scope)
    text: str = ""                       # natural-language rendering, used for embedding
    polarity: int = 1                    # +1 asserts, -1 negates ("no longer works at X")
    memory_type: MemoryType = MemoryType.SEMANTIC

    # --- valid time: when this was true in the world ---
    #: The **earliest resolved temporal boundary at which this claim is asserted to
    #: hold** — not simply "the event time", because an event's occurrence and a state's
    #: onset are different things and this field holds whichever the source stated
    #: earliest. `recorded_at` is the separate question of when we were told.
    valid_from: datetime = field(default_factory=utcnow)
    valid_to: datetime | None = None
    #: How coarse `valid_from` is, or `None` when it was not resolved from a temporal
    #: expression — the `ep.ts` fallback, and what every claim written before this field
    #: existed carries.
    #:
    #: Resolving "last month" to the 1st invents a day nobody said. Without this, a
    #: reader sees `valid_from = 2026-08-01` and can only take it as an exact onset.
    #: It stays `None` in storage so the distinction between a fallback timestamp and a
    #: resolved expression survives, and so no existing claim needs migrating.
    #:
    #: **For ordering, `None` is an exact instant**, which makes `write.reconcile`'s rule
    #: reduce to the plain scalar comparison it replaced whenever nothing was resolved.
    #: That is the whole of it: there is no separate treatment of a fallback boundary
    #: meeting a stated one, because the write path does not produce that pairing on a
    #: predicate where it could matter — a boundary is resolved only for predicates that
    #: accumulate, never for the ones that supersede.
    temporal_precision: Precision | None = None

    # --- transaction time: when we believed it ---
    recorded_at: datetime = field(default_factory=utcnow)
    #: When we stopped believing this record. **Not a liveness flag**, however much
    #: `claim.invalidated_at is None` reads like one.
    #:
    #: It was one, once. Superseding used to close both clocks, so "we never retracted
    #: it" and "it is in force" selected the same claims. Superseding now closes valid
    #: time alone, so a superseded claim has `valid_to` set and this still `None` — it is
    #: `ended`: neither live nor retired. The old idiom counts every such claim as live,
    #: always erring in the same direction (too many), and nothing raises, because the
    #: expression stays perfectly valid Python that used to be right.
    #:
    #: Ask the question you actually have:
    #:
    #: ``claim.is_live()``                   is it in force? (both clocks, four columns)
    #: ``memvara.store.live_predicate()``    the same test, as SQL, for a query or a dashboard
    #: ``claim.invalidated_at is not None``  did we stop *believing* it? — still exactly right
    #: ``claim.invalidated_by is not None``  was it ever displaced, by either closure?
    #:
    #: See the "Live is no longer `invalidated_at IS NULL`" entry in `CHANGELOG.md` and
    #: `docs/UPGRADING.md` for the grep list.
    invalidated_at: datetime | None = None
    invalidated_by: str | None = None    # claim id that superseded this one

    # --- quality signals, used in ranking ---
    confidence: float = 1.0              # how sure the extractor was
    #: *Retrieval* strength: how available this claim is right now. Decays with time
    #: and is restored by re-observation. Derived — see `salience_base`, which is the
    #: half of the pair that re-observation actually raises.
    salience: float = 1.0
    observation_count: int = 1           # how many times we've independently seen this

    # --- quantity: at most one per claim ---
    #: The measured value this observation carries, with `unit`. A claim holds **at most
    #: one** quantity: "I ran 5 km in 30 minutes" is two independent measurements and is
    #: two claims, never one with a compound object.
    #:
    #: Generic, not tied to any predicate pack — `user | goal | save for camera` with
    #: `amount=1000, unit="usd"` is intended and valid.
    #:
    #: **Neither field takes part in identity.** They describe the particular
    #: observation, so `fact_key` and `value_key` ignore them; putting a quantity in the
    #: identity would make 70kg and 71kg two facts that never supersede each other.
    amount: float | None = None
    #: Canonical, singular, lowercase — `minute`, `kilometer`, `usd`. Folded by
    #: `write.when.normalize_unit`, which converts nothing: 120 minutes never becomes 2
    #: hours, and an unrecognised unit is kept rather than guessed at.
    unit: str | None = None

    # --- provenance ---
    sources: list[str] = field(default_factory=list)   # Episode ids
    derivation: Derivation = Derivation.LLM_EXTRACT
    extractor: str = ""                  # model id / rule version that produced it

    id: str = field(default_factory=lambda: _new_id("cl"))
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text:
            self.text = self.render()

    def render(self) -> str:
        """Human/embedding-facing rendering of the triple."""
        pred = self.predicate.replace("_", " ")
        neg = "no longer " if self.polarity < 0 else ""
        return f"{self.subject} {neg}{pred} {self.object}".strip()

    @property
    def _owner(self) -> str:
        """Whose fact this is. See `owner_key`.

        Extraction yields a generic subject ("user"), so the scope's owner is what
        actually distinguishes people - without it, two users in one tenant collide and
        Bob's "lives in Lisbon" silently retires Alice's "lives in Berlin".

        Agent and session are deliberately excluded: a durable fact about a person is the
        same fact no matter which agent or session observed it, so learning "I moved to
        Lisbon" in a new session must still retire the old city.
        """
        return owner_key(self.scope)

    # --- trace strength ------------------------------------------------------
    # Bjork & Bjork's new theory of disuse, which the two properties below implement:
    # a memory has *storage* strength (how well learned, never decreases) and
    # *retrieval* strength (how available now, decays), and re-observation raises the
    # first while restoring the second. Collapsing them into one number is what made
    # reinforcement and decay fight over `salience`, with decay always winning.

    @property
    def salience_base(self) -> float:
        """*Storage* strength: salience with the decay curve divided back out.

        This is the number reinforcement raises and the number decay reads. It never
        falls on its own, so a bump survives every later pass — whereas a bump written
        straight onto `salience` is erased the moment the pass recomputes the curve.

        Falls back to `salience` for a claim nothing has decayed yet, which is exactly
        right: an undecayed claim's retrieval strength *is* its storage strength.
        """
        stored = self.meta.get(SALIENCE_BASE)
        return float(stored) if isinstance(stored, (int, float)) else self.salience

    @property
    def last_observed(self) -> datetime | None:
        """When this fact was last independently re-observed, or None if never."""
        stored = self.meta.get(LAST_OBSERVED)
        if not isinstance(stored, (int, float)):
            return None
        return datetime.fromtimestamp(float(stored), tz=timezone.utc)

    @property
    def trace_from(self) -> datetime:
        """The instant staleness is measured from: the later of `valid_from` and the
        last re-observation.

        Age has to be able to *reset*, or repetition cannot beat forgetting. Keyed off
        `valid_from` alone, a fact the user restated this morning is scored as stale as
        one nobody has mentioned since it was first recorded — measured at a recency
        factor of 1.35e-04 for something observed 91 times.

        `valid_from` remains the floor, so nothing here weakens the promise that
        back-dating a fact we learned late does not hand it artificial freshness: a
        claim that was never re-observed has no observation timestamp to move it.
        """
        seen = self.last_observed
        began = as_utc(self.valid_from)
        return began if seen is None or seen < began else seen

    def record_observation(self, at: datetime, base: float) -> None:
        """Note a re-observation: pin storage strength and stamp the instant.

        The two move together or not at all. A base raised without the timestamp still
        decays from the original `valid_from`, so the gain evaporates on the next pass;
        a timestamp moved without the base makes repetition free. Either half alone
        reproduces one of the two bugs this pair exists to fix.

        The stamp only ever moves *forward*, because "last observed" is a maximum by
        definition. Replays do not arrive in order — an importer reconstructing a year
        of history from someone's existing store hands us whatever order its table
        happens to be in — and letting an old observation overwrite a newer one would
        make the recency signal a function of that order.
        """
        self.meta[SALIENCE_BASE] = round(base, SALIENCE_PRECISION)
        seen = self.last_observed
        moment = as_utc(at)
        if seen is None or seen < moment:
            self.meta[LAST_OBSERVED] = moment.timestamp()

    # --- identity ------------------------------------------------------------
    # Both keys hash *entity identities*, not the strings the user typed. Hashing the
    # raw strings made "Acme", "Acme Corp", "acme inc" and "ACME" four different
    # employers, so a single-valued predicate reported three job changes that never
    # happened — each one retiring the last, each one explained in full by `why()`.
    # `subject`/`object` still hold what was actually said; only identity changed.

    @property
    def subject_key(self) -> str:
        """Entity this claim is about. See `memvara/entities.py`."""
        return resolved_entity(self.meta, SUBJECT_ENTITY, self.subject)

    @property
    def object_key(self) -> str:
        """Entity this claim asserts as the value."""
        return resolved_entity(self.meta, OBJECT_ENTITY, self.object)

    @property
    def fact_key(self) -> str:
        """Identity of the *slot* this claim occupies.

        Two claims sharing a fact_key are competing answers to the same question.
        For single-valued predicates that is exactly the contradiction condition.
        """
        return fact_key_for(self.scope, self.subject_key, self.predicate)

    @property
    def value_key(self) -> str:
        """Identity of this exact assertion, for exact-duplicate detection.

        Keyed on the object's *entity*, so a respelling is a re-observation rather than
        a new value — which is the difference between reinforcing an employer and
        inventing a job change.
        """
        return content_hash(
            self._owner, self.subject_key, self.predicate, self.object_key,
            str(self.polarity),
        )

    @property
    def state(self) -> str:
        """`"live"`, `"ended"` or `"retired"` — the two axes reduced to one word.

        `retired` is transaction time: we stopped believing it. `ended` is valid time: we
        still believe it, and it stopped being true. `live` is neither. The distinction is
        the whole bitemporal model, and collapsing it to a boolean is how a surface ends
        up reporting "deleted" for a fact that merely finished.

        A property rather than a helper because three surfaces had each derived it
        independently — `server.tools._state`, this class's `__repr__`, and the REST
        renderer — and three copies of a rule this small is three chances to disagree
        about what a retired claim is called.

        Deliberately **not** relative to an `as_of`: a claim retired last week is
        `retired` even in a March view, and it is the caller who pairs that with the
        `as_of` it asked for. A state that silently means "as of your query" cannot
        express "believed then, retired since", which is the thing worth showing.
        """
        if self.invalidated_at is not None:
            return "retired"
        if self.valid_to is not None:
            return "ended"
        return "live"

    def __repr__(self) -> str:
        # State, not raw timestamps: "is this claim still believed?" is the question
        # anyone reading a list of claims at a REPL is actually asking, and the two
        # timestamp pairs that answer it are four fields of eighteen.
        state = self.state
        neg = "not " if self.polarity < 0 else ""
        return (
            f"<Claim {self.id} {self.scope.key()} {self.subject} "
            f"{neg}{self.predicate}={_short(self.object)!r} "
            f"{self.memory_type.value} conf={self.confidence:.2f} "
            f"sal={self.salience:.2f} {state}>"
        )

    def is_live(self, as_of: datetime | None = None, *,
                valid_at: datetime | None = None,
                known_at: datetime | None = None) -> bool:
        """Was this claim believed at `known_at`, and in force at `valid_at`?

        The Python mirror of the store's liveness predicate, and the two must agree
        clause for clause — `SQLiteStore._live_clause` is the SQL of exactly this.

        The two axes move independently. `known_at` is the belief clock: had we
        recorded it, and not yet retracted it. `valid_at` is the world clock: had it
        started being true, and not yet stopped. `as_of` sets both, which is why it can
        only ask about a past belief *about* that same past — see `time_axes`.

        >>> june = datetime(2026, 6, 15, tzinfo=timezone.utc)
        >>> august = datetime(2026, 8, 1, tzinfo=timezone.utc)
        >>> learned_late = Claim(subject="user", predicate="lived_in", object="Rome",
        ...                      valid_from=june, recorded_at=august)
        >>> learned_late.is_live(as_of=june)             # we had not heard it yet
        False
        >>> learned_late.is_live(valid_at=june, known_at=august)
        True
        """
        valid_at, known_at = time_axes(as_of, valid_at, known_at)
        # One clock read serves both defaults. Reading it twice would put the axes
        # microseconds apart on the commonest call of all — bare `is_live()` — which is
        # a difference nothing should be able to observe.
        now = utcnow()
        v = valid_at if valid_at is not None else now
        k = known_at if known_at is not None else now
        if self.recorded_at > k:
            return False                                  # we didn't know it yet
        if self.invalidated_at is not None and self.invalidated_at <= k:
            return False                                  # we'd already retracted it
        if self.valid_from > v:
            return False                                  # not true yet
        if self.valid_to is not None and self.valid_to <= v:
            return False                                  # no longer true
        return True


@dataclass(slots=True)
class Provenance:
    """Why a claim exists, resolved to source text. Returned by `Memvara.why()`."""

    claim: Claim
    episodes: list[Episode]
    derivation: Derivation
    extractor: str
    superseded: list["Claim"] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"<Provenance {self.claim.id} {_short(self.claim.text)!r} "
            f"via {self.extractor or '?'} ({self.derivation.value}) "
            f"sources={len(self.episodes)} superseded={len(self.superseded)}>"
        )


@dataclass(slots=True)
class Explanation:
    """Why a claim surfaced in a particular search. Attached to every result.

    Retrieval that cannot explain itself is impossible to debug, and silent recall
    failures are the single hardest class of agent bug to track down.
    """

    vector_rank: int | None = None
    vector_score: float | None = None
    lexical_rank: int | None = None
    lexical_score: float | None = None
    fusion_score: float = 0.0
    recency: float = 1.0
    confidence: float = 1.0
    salience: float = 1.0
    rerank_score: float | None = None
    #: The pre-normalization product of fusion and the quality multipliers. It is kept
    #: because it is what the retriever actually computes and what a ranking change
    #: should be diffed on; it is *not* comparable across queries, which is precisely
    #: why `Result.score` is the normalized value instead. The retriever owns how the
    #: two relate (see `memvara/retrieve/scoring.py`).
    raw_score: float = 0.0
    final_score: float = 0.0        # == Result.score, i.e. normalized into [0, 1]

    # --- fields added after 0.2.0, appended rather than slotted in beside their kin ---
    #
    # `graph_*` and `temporal_*` belong next to `vector_*` and `lexical_*` and are not
    # there, because this is a dataclass and its field order is an API. Inserted in the
    # readable place they shifted `fusion_score` and everything after it four positions
    # right, so `Explanation(0, 0.9, 1, 0.8, 0.5)` — a perfectly ordinary call written
    # against 0.2.x — put the fusion score into `graph_rank` and left `fusion_score` at
    # its default. No exception: an `int | None` field silently holding 0.5, and a
    # ranking explanation quietly reporting the wrong number about itself.
    #
    # Pickle is a separate question and appending does not answer it: `slots=True` makes
    # `__getstate__` a *name*-keyed dict, so an older pickle restores its own fields
    # correctly and simply leaves these five unset — reading one then raises
    # `AttributeError` rather than returning the default. That is true wherever they sit.
    #: Position and path score in the graph leg — the multi-hop walk seeded from the head
    #: of the other two (see `memvara/retrieve/spread.py`). `None` means this claim was
    #: not on any path the walk returned, which includes the ordinary case of the walk
    #: not having run: it is gated by query intent, it needs a `Store` with `adjacent`,
    #: and it is off entirely at `w_graph=0`. A number here says the claim was reached
    #: *through* something else, which is the one thing the other two legs cannot report.
    graph_rank: int | None = None
    graph_score: float | None = None
    #: Position and closeness in the temporal leg — the episode search that ranks on
    #: *when* and reads no text (see `memvara/retrieve/temporal.py`). Only ever set on an
    #: `EpisodeResult`: a claim's time signal is the predicate-keyed decay in `recency`,
    #: which knows what raw proximity cannot, that a `born_in` from 2019 is as current as
    #: it will ever be. `None` means the leg did not rank this turn, which includes the
    #: ordinary case of it not having run.
    temporal_rank: int | None = None
    temporal_score: float | None = None
    #: The query shape retrieval routed this search as, or `None` when intent weighting
    #: was off. It is on the explanation rather than only in a log because it is the
    #: answer to the question a surprising result set actually raises: not "why is this
    #: row here" but "why was the leg that would have found the other one switched off".
    #: See `memvara/retrieve/intent.py`.
    intent: str | None = None
    #: What tied this result to the question. `"subject"` or `"object"` when the query
    #: names that end of the claim, `"path"` when the graph leg reached it by walking out
    #: of a claim that was named, and `None` when nothing did — the result surfaced on
    #: vocabulary alone, which on a question the store cannot answer is what the best
    #: available row looks like. `search(anchored=True)` returns only the first three.
    #: See `memvara/retrieve/anchor.py`.
    anchor: str | None = None

    def summary(self) -> str:
        bits = []
        if self.vector_rank is not None:
            bits.append(f"vector#{self.vector_rank}({self.vector_score:.3f})")
        if self.lexical_rank is not None:
            bits.append(f"bm25#{self.lexical_rank}({self.lexical_score:.2f})")
        if self.graph_rank is not None:
            bits.append(f"graph#{self.graph_rank}({self.graph_score:.3f})")
        if self.temporal_rank is not None:
            bits.append(f"time#{self.temporal_rank}({self.temporal_score:.3f})")
        bits.append(f"recency={self.recency:.2f}")
        bits.append(f"conf={self.confidence:.2f}")
        bits.append(f"sal={self.salience:.2f}")
        if self.rerank_score is not None:
            bits.append(f"rerank={self.rerank_score:.3f}")
        if self.raw_score:
            # Shown only once a retriever populates it, so the line stays readable for
            # anything that scores without a normalization step.
            bits.append(f"raw={self.raw_score:.4f}")
        if self.intent is not None:
            bits.append(f"intent={self.intent}")
        return " ".join(bits) + f" -> {self.final_score:.4f}"

    def __repr__(self) -> str:
        return f"<Explanation {self.summary()}>"


@dataclass(frozen=True, slots=True)
class ErasureProof:
    """Whether one claim is actually gone, checked against the disk rather than inferred.

    `Memvara.erase()` used to report success from a return code: the store said it had
    deleted a row, and that was the whole of the evidence. That proves the code took the
    branch it thought it took, which is the same statement the return value already made.
    A proof has to be able to *disagree* with the delete, so this is built from a physical
    re-query — `Store.residue`, four `SELECT COUNT(*)`s over the tables a claim's content
    can survive in.

    **`proven` is false whenever it could not be established**, never merely when
    something survived. A store with no `residue` method yields `proven=False` with a
    reason naming it, and `erase()` refuses rather than reporting a success it cannot
    support. Unproven and proven-gone are different answers and only one of them is an
    erasure certificate; rendering the first as the second is the failure this type
    exists to remove.

    `residue` is the per-table count, and it is carried even when it is all zeroes,
    because "checked these four tables and found nothing" is the evidence. An **empty**
    residue is the opposite: it means nothing was counted, and it can never accompany
    `proven=True` — `all(n == 0 for n in {})` is vacuously true, which is exactly the trap
    this distinction exists to keep out of `erase()`. `record` is
    the `erasures` row, or `None` — a store that keeps no audit trail can still prove the
    rows are gone, and cannot prove that anything recorded their going.

    >>> ErasureProof(claim_id="c1", proven=True, residue={"claims": 0}).surviving
    {}
    >>> gone_wrong = ErasureProof(claim_id="c1", proven=False,
    ...                           residue={"claims": 0, "claims_fts": 1},
    ...                           reason="rows survived the delete")
    >>> gone_wrong.surviving
    {'claims_fts': 1}
    """

    claim_id: str
    #: True only when a physical re-query ran *and* every table came back empty.
    proven: bool
    #: Per-table surviving row counts, from `Store.residue`. Empty when nothing could be
    #: counted, which is a different thing from every count being zero.
    residue: dict[str, int] = field(default_factory=dict)
    #: Why `proven` is false, in one sentence. Empty when it is true.
    reason: str = ""
    #: The audit row this erasure wrote, if the store keeps one. See
    #: `SQLiteStore.erasure_record`; `None` also means "this store has no such table",
    #: so it is never on its own evidence that nothing happened.
    record: dict[str, Any] | None = None

    @property
    def surviving(self) -> dict[str, int]:
        """The tables that still hold something. Empty is the answer you want."""
        return {table: n for table, n in self.residue.items() if n}

    def __repr__(self) -> str:
        if self.proven:
            return f"<ErasureProof {self.claim_id} gone {sorted(self.residue)}>"
        return f"<ErasureProof {self.claim_id} UNPROVEN {self.reason!r}>"


@dataclass(slots=True)
class Result:
    """A retrieved claim with its score and the reason it was retrieved.

    `score` is a **normalized relevance in [0, 1]**, so it can be thresholded and
    compared across queries — `min_score=0.3` means the same thing tomorrow as today,
    and the integrations that expect a 0-1 relevance (mem0, CrewAI, LlamaIndex) get
    what they expect. The retriever's raw internal value lives on
    `explain.raw_score`; it is unbounded-ish and query-dependent, and thresholding on
    it is the bug this split exists to prevent.
    """

    claim: Claim
    score: float
    explain: Explanation

    #: Discriminator for callers that serialize a mixed result list. `isinstance` is
    #: the in-process answer and stays the authoritative one - a `Claim` has been
    #: extracted, reconciled and possibly retired, an `Episode` is a verbatim thing
    #: someone said once, and the two are not interchangeable however alike their
    #: fields look. This carries the same answer across a wire, where types do not go.
    kind: ClassVar[str] = CLAIM

    @property
    def text(self) -> str:
        return self.claim.text

    def __repr__(self) -> str:
        legs = []
        if self.explain.vector_rank is not None:
            legs.append(f"vector#{self.explain.vector_rank}")
        if self.explain.lexical_rank is not None:
            legs.append(f"bm25#{self.explain.lexical_rank}")
        if self.explain.graph_rank is not None:
            # Without this line a claim reached only by traversal reprs as
            # `no-retriever`, which is the one reading that is actually wrong: a
            # retriever found it, and which one is the whole point of the leg.
            legs.append(f"graph#{self.explain.graph_rank}")
        return (f"<Result {self.score:.4f} {_short(self.text)!r} "
                f"{'+'.join(legs) or 'no-retriever'} {self.claim.id}>")


@dataclass(frozen=True, slots=True)
class RecallResult:
    """What `recall(with_ids=True)` returns: the prompt block, and what is in it.

    `recall()` renders claims into text and threw the identities away, which made the
    prompt-shaped surface the one surface an agent could not cite from — it could read
    "user lives in Lisbon" back to someone and had nothing to name if asked which stored
    record that came from. `search()` returns `Result` objects and has always been
    citable; the block did not, so anything built on "what did the model actually lean
    on" had to abandon the surface it was built for or re-run retrieval and hope the
    second answer matched the first.

    **`claim_ids` is in render order and 1:1 with the claim notes**, so note *n* of the
    block under `Memvara.RECALL_HEADER` is `claim_ids[n - 1]`. Nothing else in the block
    is covered, and both omissions are deliberate: an episode is a verbatim turn rather
    than a claim and has no claim id to give, and a past value under
    `RECALL_HISTORY_HEADER` is a fact's *former* value, so citing it as the source of a
    present-tense answer would be a worse error than not citing at all.

    This carries no read that `recall()` did not already perform. Its signature is
    explicit precisely so `as_of`, `states` and `include_invalidated` cannot be forwarded
    into a live prompt — see `Memvara.recall` — and handing back the ids of claims the
    call has already rendered forwards nothing new: the text was the disclosure, and the
    id is the handle on text the caller is holding.

    >>> block = RecallResult(text="Known:\\n- user lives in Lisbon", claim_ids=("cl-1",))
    >>> block.text.splitlines()[1]
    '- user lives in Lisbon'
    >>> block
    <RecallResult 1 cited, 29 chars>
    """

    #: Exactly what `recall()` returns without `with_ids`, byte for byte.
    text: str
    #: The claims rendered as notes, in the order they appear.
    claim_ids: tuple[str, ...] = ()
    #: How many further notes `budget=` cut. `0` whenever no budget was given, and the
    #: machine-readable twin of the line the block ends with — a caller that had to
    #: parse that prose to learn its answer was bounded would be reading a sentence
    #: written for a model.
    dropped: int = 0

    def __repr__(self) -> str:
        # Not the dataclass repr: `text` is a whole system prompt, and printing one at a
        # REPL buries the two numbers a caller is actually checking.
        cut = f", {self.dropped} dropped" if self.dropped else ""
        return f"<RecallResult {len(self.claim_ids)} cited, {len(self.text)} chars{cut}>"


@dataclass(frozen=True, slots=True)
class Delta:
    """What `since()` returns: what arrived and what left while the caller was away.

    Two populations rather than one list, because "changed" is not a single event here.
    A claim in `added` is one this scope believes now and did not believe then. A claim
    in `gone` is the reverse — retired since, or finished in world time since — and it is
    the half a store without a belief clock cannot report at all, because by the time you
    ask, there is nothing left in the current view to notice the absence of.

    A supersession appears in **both**: the value we stopped holding in `gone`, the one
    that replaced it in `added`. That is not double-counting, it is the correction stated
    in the only way that carries the fact it corrected.

    `since` is the instant asked about, carried back so a caller logging or paging the
    result cannot separate the answer from the question it answers.

    >>> from datetime import datetime, timezone
    >>> Delta(since=datetime(2026, 8, 1, tzinfo=timezone.utc))
    <Delta since 2026-08-01T00:00:00+00:00 +0 -0>
    """

    #: The instant the caller asked about, resolved to UTC.
    since: datetime
    #: Believed now, not believed then. Newest first.
    added: tuple["Claim", ...] = ()
    #: Believed then, not believed now. Newest first.
    gone: tuple["Claim", ...] = ()

    def __repr__(self) -> str:
        return (f"<Delta since {self.since.isoformat()} "
                f"+{len(self.added)} -{len(self.gone)}>")


@dataclass(frozen=True, slots=True)
class Reading:
    """One fact slot, read three ways at one instant. The unit `Answer` is made of.

    Three populations, and the whole value of the type is that they can differ:

    * `now` — in force at this moment.
    * `then` — what we believe **today** was true at the instant asked about. A
      correction that arrived last week is in this answer, because it is about the world
      and we now know better.
    * `stated` — what this store **would have answered** at that instant. A correction
      that arrived last week is *not* in this answer, because it had not arrived.

    `then` and `stated` disagreeing is the finding, not an inconsistency: it means the
    record changed under a decision somebody already made. An agent that acted on
    2026-03-15 acted on `stated`, and is being audited against `then`.

    **`stated` is reconstructed, and `get_all(as_of=T)` does not agree with it.** That is
    deliberate and it is the one thing to understand about this type. A row's `valid_to`
    is written *in place* by the write that displaces it, so a row read on its own cannot
    say when its own ending came to be believed — `get_all(as_of=T)` therefore applies an
    ending that had not been recorded at `T`. `Reading` has the supersession chain in
    front of it, so it can date the ending at the successor's `recorded_at`, which is the
    instant the pointer was written. `Memvara._displaced_by` is the same rule, and
    `why()` has used it since a July view started reporting an August replacement.

    What it cannot recover: an ending whose successor has since been erased. The pointer
    survives the erasure and its target does not, so the closure is dated at this row's
    own `recorded_at` — the earliest instant it could have been known, which makes the
    claim stop answering sooner rather than later. Under-reporting a past answer is the
    safe direction in a store somebody is auditing.

    >>> Reading("user", "lives_in")
    <Reading user lives_in now=0 then=0 stated=0>
    """

    subject: str
    predicate: str
    #: In force now. Oldest first, as `history()` returns them.
    now: tuple["Claim", ...] = ()
    #: What we believe today was true at the instant asked about.
    then: tuple["Claim", ...] = ()
    #: What this store would have answered at that instant.
    stated: tuple["Claim", ...] = ()
    #: Every version of this slot, oldest first — including the ones no query returns.
    timeline: tuple["Claim", ...] = ()
    #: Whether this slot's predicate is declared to hold one value. Needed because a slot
    #: that *should* hold one and holds two is a different statement from a slot that
    #: holds many by design, and a renderer joining values with commas says the same thing
    #: for both. A bool rather than the `Cardinality` it comes from: `schema` imports this
    #: module, so the enum cannot travel back the other way. False for an undeclared
    #: predicate, which is the safe reading — undeclared means multi-valued everywhere
    #: else in the write path.
    single_valued: bool = False

    @property
    def diverged(self) -> bool:
        """Whether the record changed under the instant asked about.

        Compared on claim id rather than on value, because a value re-asserted as a new
        claim is a different record of the same string and the audit question is which
        record answered.
        """
        return {c.id for c in self.then} != {c.id for c in self.stated}

    @property
    def moved(self) -> bool:
        """Whether the world moved between the instant asked about and now."""
        return {c.id for c in self.then} != {c.id for c in self.now}

    def __repr__(self) -> str:
        return (f"<Reading {self.subject} {self.predicate} now={len(self.now)} "
                f"then={len(self.then)} stated={len(self.stated)}>")


@dataclass(frozen=True, slots=True)
class Answer:
    """What `ask()` returns: a question, the instant it was about, and the readings.

    `text` is the composed narrative and is the reason this type exists — `recall()`
    already returns ranked notes, and a ranked note cannot say *"that is what we believe
    today, and it is not what we would have told you then"*. Every sentence in it is
    rendered from a stored column; nothing here consults a model, and nothing here is
    inferred.

    >>> from datetime import datetime, timezone
    >>> Answer("where do they live?", datetime(2026, 3, 15, tzinfo=timezone.utc))
    <Answer 'where do they live?' at 2026-03-15T00:00:00+00:00, 0 slot(s)>
    """

    question: str
    #: The world instant the question was about, resolved to UTC. `ask()` defaults it to
    #: now, and carries it back so a caller logging the result cannot separate the answer
    #: from the question it answers — the same reason `Delta` carries `since`.
    at: datetime
    #: One per fact slot, best match first.
    readings: tuple[Reading, ...] = ()
    #: The narrative, rendered. Empty only when nothing matched.
    text: str = ""

    @property
    def diverged(self) -> tuple[Reading, ...]:
        """Readings where the record changed under the instant asked about."""
        return tuple(r for r in self.readings if r.diverged)

    def __repr__(self) -> str:
        return (f"<Answer {_short(self.question)!r} at {self.at.isoformat()}, "
                f"{len(self.readings)} slot(s)>")


@dataclass(frozen=True, slots=True)
class Accumulation:
    """One value that landed beside values already live in the same slot.

    Recorded when the write path *accumulated* where it might have been meant to
    replace: the predicate has no spec in the registry, so it is `Cardinality.MANY` by
    default, so the slot already holding `existing` live values simply gained another.
    Nothing is wrong with the row that was written — it is the absence of a decision
    about the predicate that this reports.

    Only the count is carried, not the claims. The occupants are already reachable by
    `get_all(subject=..., predicate=...)` and holding them here would make a receipt for
    a one-triple write grow with the size of the slot it landed in.

    >>> Accumulation("quota_gate", "status", 1)
    <Accumulation quota_gate status +1 beside 1>
    """

    subject: str
    predicate: str
    #: Live values in the slot **before** this write. One or more, always: a first write
    #: to an empty slot is an ordinary write and produces no `Accumulation` at all.
    existing: int

    def __repr__(self) -> str:
        return (f"<Accumulation {self.subject} {self.predicate} "
                f"+1 beside {self.existing}>")


@dataclass(slots=True)
class Retype:
    """A claim that was already known, re-filed under a different `memory_type`.

    An identical triple is the same fact, so re-asserting one reinforces the record
    rather than forking it — and until this existed, the `memory_type` the caller sent
    was dropped on that path. A claim filed wrongly could not be moved: writing it again
    with the corrected type reported `already-known 1`, left the type alone, and raised
    the confidence, so the attempt to fix it made the wrong filing more strongly believed.

    The type is not decoration. `memory_standing` returns `procedural` and nothing else,
    and clients inject that set at the top of every session, so a project fact misfiled as
    `procedural` is carried on every turn of every conversation until it is corrected.

    **Only an asserted type moves a claim.** `remember(memory_type=...)` is a caller
    saying what this is; `remember()` without one takes the predicate's default and says
    nothing, and extraction never reaches here at all. That asymmetry is the whole safety
    property: an agent writing the same triple back without an opinion cannot silently
    undo a correction somebody made deliberately.

    What does *not* change is `derivation`. The content still came from wherever it came
    from — only the filing moved, and rewriting the provenance of a fact because its
    drawer was wrong would lose the more important of the two. `consolidate.promote_pass`
    does change it, correctly, because there consolidation authored the reclassification
    rather than re-filing someone else's.

    >>> Retype("cl_1a2b", "agent-memory", "rejected", MemoryType.PROCEDURAL,
    ...        MemoryType.SEMANTIC)
    <Retype agent-memory rejected: procedural -> semantic>
    """

    claim_id: str
    subject: str
    predicate: str
    #: What it was filed as until this write.
    was: "MemoryType"
    #: What the caller asserted, and what it is filed as now.
    now: "MemoryType"

    def __repr__(self) -> str:
        return (f"<Retype {self.subject} {self.predicate}: "
                f"{self.was.value} -> {self.now.value}>")


@dataclass(slots=True)
class Dispute:
    """A value that did not displace the one already in its slot, because it is worth less.

    The write path resolves a contradiction by predicate cardinality and, until this
    existed, by nothing else — so a 0.10-confidence guess replaced a 1.00-confidence
    statement and stamped the displaced claim `ended`, which asserts that *the world
    changed*. It did not; a machine guessed. See `memvara.write.reconcile.AUTHORITY_SHARE`
    for the rule and for why confidence is the axis it reads.

    The candidate is stored either way. Nothing here refuses a write: the slot simply
    holds both values, the more confident one ranks above the other, and this says so.
    Which is the recoverable direction — keeping two competing facts degrades ranking,
    ending a true one destroys information.

    Unlike `Accumulation`, this carries the values rather than a count. A caller reading
    it has one decision to make about one pair of claims, and "which value stayed" is the
    whole of what they need to make it.

    >>> Dispute("cl_1a2b", "user", "lives_in", "London", 1.0, "Paris", 0.1)
    <Dispute user lives_in: 'London' 1.00 kept, 'Paris' 0.10 stored beside it>
    """

    #: The incumbent that stayed live. Addressable, because acting on this means
    #: deciding between two claims and the caller needs to be able to name one of them.
    claim_id: str
    subject: str
    predicate: str
    incumbent: str
    incumbent_confidence: float
    candidate: str
    candidate_confidence: float

    def __repr__(self) -> str:
        return (f"<Dispute {self.subject} {self.predicate}: "
                f"{self.incumbent!r} {self.incumbent_confidence:.2f} kept, "
                f"{self.candidate!r} {self.candidate_confidence:.2f} stored beside it>")


@dataclass(slots=True)
class Collapse:
    """A claim closed at or before its own start, so it is now true at no instant.

    `close_out` clamps a closure to the claim's `valid_from` rather than letting the
    interval invert, and that clamp is right: a fact that ends before it begins is not a
    shorter fact, it is a row no `as_of` window can return consistently. What the clamp
    cannot do is make the row answer anything. `valid_from == valid_to` is an empty
    interval — `valid_at=T` returns it at no `T`, `is_live` is false at every instant,
    and the receipt above still reads `added 1`.

    It arises whenever a value is superseded by one that begins at the same instant: any
    same-day correction, and every import that stamps dates rather than timestamps.

    Reported rather than prevented, and the alternative is worth naming so nobody
    re-opens it as an oversight. Nudging the edge forward by a tick would give the
    displaced claim an interval — one that nothing witnessed and nobody asserted, in a
    store whose whole argument is that its intervals come from evidence. The honest
    answer is that this claim has no interval, said out loud at the write that did it.
    `memory_remember` refuses the same shape when both ends arrive in one call, where the
    caller can simply restate it; here the row already exists and refusing would leave no
    way to close it at all.

    >>> Collapse("cl_1a2b", "user", "city", "Delhi", datetime(2026, 1, 10, tzinfo=timezone.utc))
    <Collapse user city 'Delhi' true at no instant, both ends 2026-01-10T00:00:00+00:00>
    """

    claim_id: str
    subject: str
    predicate: str
    object: str
    #: Where the interval now both begins and ends.
    at: datetime

    def __repr__(self) -> str:
        return (f"<Collapse {self.subject} {self.predicate} {self.object!r} "
                f"true at no instant, both ends {self.at.isoformat()}>")


@dataclass(slots=True)
class WriteReceipt:
    """What `add()` returns. Explicit about what the write path actually did.

    `llm_calls` is here on purpose: it is the number the whole write-path design is
    trying to drive to zero, so it should be visible in normal use, not buried in a
    metrics backend.
    """

    episode_ids: list[str] = field(default_factory=list)
    added: list[Claim] = field(default_factory=list)
    #: Claims this write closed out, as they read *after* the write — on **whichever**
    #: clock `close=` stopped. That is why the name is `closed` and not `ended`: under
    #: the default `close="ended"` these are claims whose *valid* time was closed (still
    #: believed, no longer in force), and under `close="retired"` they are claims we have
    #: stopped believing. One write passes one `close=`, so a given list is homogeneous
    #: in practice; the field as a whole is not, and a caller that renders it as one word
    #: will be wrong half the time. `ended` and `retired` below split it, and
    #: `Claim.state` says which axis moved on an individual claim.
    closed: list[Claim] = field(default_factory=list)
    reinforced: list[Claim] = field(default_factory=list)  # already known, salience bumped
    skipped: int = 0                                       # turns that carried no durable fact
    #: Turns that got all the way to the extraction tier and yielded nothing. Distinct
    #: from `skipped`, which is the write path working as designed (an acknowledgement
    #: carries no fact). This one is the honest count of *lost* content: with no model
    #: configured it is where a conversation's facts go, and without it the default
    #: configuration reports a clean, successful, empty write.
    unextracted: int = 0
    #: Claims the extractor proposed and this write refused to store because nothing
    #: tied them to the turn they cite as their source -- `WritePipeline`'s
    #: `reject_ungrounded`, which defaults to `"auto"`: no shared vocabulary at all, and
    #: no embedding similarity either, so the claim reads as invented rather than
    #: paraphrased. Zero here means either nothing tripped or the option was turned off
    #: -- the receipt cannot tell a reader which, so absence of this number is not
    #: evidence of absence of fabrication. A claim counted here also counts toward
    #: `unextracted` if it was the only thing proposed for its turn, since `out` never
    #: received it either way -- one rejection, two honest numbers, not a double count.
    ungrounded: int = 0
    #: Turns `reextract()` was handed that already had claims citing them, and so did not
    #: read again. Always 0 from `add()`, which never sees an episode twice.
    #:
    #: It is a skip rather than a no-op worth hiding: re-reading a stored turn is not new
    #: evidence, but an identical claim arriving twice reconciles to `reinforce` and bumps
    #: salience, so a sweep that ran over the same episode twice would quietly promote
    #: what it had already extracted. This is the count of the times that was avoided,
    #: and a scheduler seeing it climb is selecting episodes it has already done.
    already_extracted: int = 0
    #: Model calls actually made. Must stay 0 for a backend that advertises itself as a
    #: no-op (`llm.is_noop`): billing for a call that never left the process makes the
    #: one number this design exists to minimize into a lie.
    llm_calls: int = 0
    #: Tokens this write consumed, across every call it made. Here for the reason
    #: `llm_calls` is here — it is what the caller is actually charged for, and a cost a
    #: caller can only discover by configuring a metrics backend is a cost most callers
    #: never discover. `llm_calls` is the number the design drives toward zero; these are
    #: the number the invoice is computed from, and they do not move together.
    #:
    #: Both stay 0 when the configured backend does not report usage, which is
    #: indistinguishable from a write that made no calls — deliberately, because the
    #: honest per-write answer in that case is "unknown" and inventing an estimate is
    #: worse. `write.tokens_in`/`out` simply go unpublished; see `telemetry.py`.
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    deferred: bool = False                                 # extraction queued, not yet run
    #: Values this write added *beside* live values already in the same slot, because the
    #: predicate has no spec and an unspecified predicate is multi-valued. Empty on
    #: virtually every write, which is what makes a non-empty one worth reading.
    #:
    #: This is the one outcome the receipt could not previously distinguish. `added 1,
    #: ended 0` is what a successful supersession-that-did-not-supersede looks like, and
    #: it is also what an ordinary first write looks like, so `remember("quota_gate",
    #: "status", "installed")` over `"not installed"` reported exactly what a correct
    #: replacement reports while leaving both values answering `recall()`. Nothing else in
    #: the system says otherwise — `status` is not in the schema, acquisition never runs
    #: on this path, and MANY retires nothing by design.
    #:
    #: It is a report, not a verdict. A genuinely multi-valued predicate nobody has
    #: declared (`tagged_with`, `attended`) fills this in on every write after the first
    #: and is behaving correctly; see `memvara.write.reconcile` for why the write path
    #: cannot tell the two apart and deliberately does not try.
    accumulated: list[Accumulation] = field(default_factory=list)
    #: Values this write stored *without* displacing what was already in their slot,
    #: because the incumbent is worth more than twice as much — see
    #: `memvara.write.reconcile.AUTHORITY_SHARE`. Empty on every write between claims of
    #: comparable confidence, which is every write the shipped extraction paths produce.
    #:
    #: The outcome it names used to be indistinguishable from a correct replacement, and
    #: it was the wrong one: a low-confidence guess ended a high-confidence statement,
    #: and `ended` says the world changed. What actually happened is that two sources
    #: disagree, so both values are live and this says which one stayed.
    disputed: list[Dispute] = field(default_factory=list)
    #: Claims this write closed at or before the instant they began, leaving them true at
    #: no instant. See `Collapse`: it is what a supersession by a value starting at the
    #: same instant does, and `closed 1` looks exactly like an ordinary supersession
    #: without it.
    collapsed: list[Collapse] = field(default_factory=list)
    #: Claims this write re-filed under a different `memory_type`. See `Retype`. Only an
    #: asserted type moves one, and the move is otherwise invisible: the receipt reads
    #: `already-known 1` either way, which is what let a correction look like a no-op
    #: while it raised the confidence of the filing it was trying to fix.
    retyped: list[Retype] = field(default_factory=list)

    # --- the two halves of `closed` -------------------------------------------
    # Derived rather than stored, so they cannot disagree with the claims themselves.
    # `Claim.state` is the single authority on which clock stopped, and it is a pure
    # function of the two timestamps the write just stamped onto these objects.

    @property
    def ended(self) -> list[Claim]:
        """The claims this write closed on the **valid** axis: the world changed.

        Still believed, so they keep answering `valid_at=<back then>`. This is what a
        supersession produces, and what `close="ended"` means everywhere it appears.

        >>> moved = Claim(subject="user", predicate="lives_in", object="Berlin",
        ...               valid_to=utcnow())
        >>> receipt = WriteReceipt(closed=[moved])
        >>> len(receipt.ended), len(receipt.retired)
        (1, 0)
        """
        return [c for c in self.closed if c.state == "ended"]

    @property
    def retired(self) -> list[Claim]:
        """The claims this write closed on the **transaction** axis: the record was wrong.

        We have stopped believing them, so they answer nothing at any world-time from
        here on. This is what `close="retired"`, `forget()` and `delete()` produce.

        >>> misheard = Claim(subject="user", predicate="lives_in", object="Berlin",
        ...                  invalidated_at=utcnow())
        >>> [c.state for c in WriteReceipt(closed=[misheard]).retired]
        ['retired']
        """
        return [c for c in self.closed if c.state == "retired"]

    @property
    def invalidated(self) -> list[Claim]:
        """Deliberate alias of `closed`. Prefer `closed`, `ended` or `retired`.

        The old name for the field, from before the two axes were separated, when a
        closure really did set `invalidated_at` every time. It is kept — same list
        object, not a copy — because it is on the published API and this is not worth an
        `AttributeError` at somebody's call site.

        **No `DeprecationWarning` on purpose.** `pyproject.toml` sets
        `filterwarnings = ["error::DeprecationWarning"]`, so warning here would not warn
        anyone, it would raise on every existing caller including this package's own
        write path. Remove the alias at `1.0.0`, where the protocols are already allowed
        to change; until then the cost of keeping it is this docstring.

        >>> receipt = WriteReceipt()
        >>> receipt.invalidated is receipt.closed
        True
        """
        return self.closed

    def __str__(self) -> str:
        # `unextracted`, `ungrounded`, `accumulated`, `disputed` and `collapsed` appear
        # only when non-zero, so they read as events rather than as noise on the writes
        # that lost, rejected, piled up, disputed and emptied nothing.
        lost = f" unextracted={self.unextracted}" if self.unextracted else ""
        refused = f" ungrounded={self.ungrounded}" if self.ungrounded else ""
        piled = f" accumulated={len(self.accumulated)}" if self.accumulated else ""
        split = f" disputed={len(self.disputed)}" if self.disputed else ""
        empty = f" collapsed={len(self.collapsed)}" if self.collapsed else ""
        return (
            f"<WriteReceipt +{len(self.added)} ~{len(self.reinforced)} "
            f"-{len(self.closed)} skip={self.skipped}{lost}{refused}{piled}{split}{empty} "
            f"llm={self.llm_calls} "
            f"{self.latency_ms:.1f}ms{' deferred' if self.deferred else ''}>"
        )

    # The dataclass repr dumps every field of every nested Claim, which is unreadable at
    # a REPL and buries the one number the write path exists to minimize.
    __repr__ = __str__
