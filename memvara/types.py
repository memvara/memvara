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

#: Resolved entity of `Claim.subject`. See `memvara/entities.py`.
SUBJECT_ENTITY = "subject_entity"
#: Resolved entity of `Claim.object`.
OBJECT_ENTITY = "object_entity"
#: Timestamped record of every backfill that changed a claim's place in history.
ENTITY_REKEY = "entity_rekey"

#: The `meta` keys above, as one set: everything in `Claim.meta` that the engine owns
#: rather than the caller. Two surfaces need it and they need it for opposite reasons —
#: `Memvara.remember` **rejects** them on the way in, because `salience_base` reaching
#: the store is a permanent ranking override no documented argument can produce, and
#: `compat.mem0` **filters** them on the way out, so a compatibility layer never hands
#: internal bookkeeping back as user data. Defined here, beside the five constants, so
#: adding a sixth cannot leave one of those two surfaces behind.
RESERVED_META = frozenset({
    SALIENCE_BASE, LAST_OBSERVED, SUBJECT_ENTITY, OBJECT_ENTITY, ENTITY_REKEY,
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
        return
    # Never before the claim's own start: a closure backdated past the fact it closes
    # collapses the interval to zero length rather than inverting it. An interval that
    # ends before it begins is not a shorter fact, it is a row no `as_of` window can
    # return consistently.
    edge = max(as_utc(at), as_utc(claim.valid_from))
    if claim.valid_to is None or as_utc(claim.valid_to) > edge:
        claim.valid_to = edge


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
    valid_from: datetime = field(default_factory=utcnow)
    valid_to: datetime | None = None

    # --- transaction time: when we believed it ---
    recorded_at: datetime = field(default_factory=utcnow)
    invalidated_at: datetime | None = None
    invalidated_by: str | None = None    # claim id that superseded this one

    # --- quality signals, used in ranking ---
    confidence: float = 1.0              # how sure the extractor was
    #: *Retrieval* strength: how available this claim is right now. Decays with time
    #: and is restored by re-observation. Derived — see `salience_base`, which is the
    #: half of the pair that re-observation actually raises.
    salience: float = 1.0
    observation_count: int = 1           # how many times we've independently seen this

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

    def summary(self) -> str:
        bits = []
        if self.vector_rank is not None:
            bits.append(f"vector#{self.vector_rank}({self.vector_score:.3f})")
        if self.lexical_rank is not None:
            bits.append(f"bm25#{self.lexical_rank}({self.lexical_score:.2f})")
        bits.append(f"recency={self.recency:.2f}")
        bits.append(f"conf={self.confidence:.2f}")
        bits.append(f"sal={self.salience:.2f}")
        if self.rerank_score is not None:
            bits.append(f"rerank={self.rerank_score:.3f}")
        if self.raw_score:
            # Shown only once a retriever populates it, so the line stays readable for
            # anything that scores without a normalization step.
            bits.append(f"raw={self.raw_score:.4f}")
        return " ".join(bits) + f" -> {self.final_score:.4f}"

    def __repr__(self) -> str:
        return f"<Explanation {self.summary()}>"


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
        return (f"<Result {self.score:.4f} {_short(self.text)!r} "
                f"{'+'.join(legs) or 'no-retriever'} {self.claim.id}>")


@dataclass(slots=True)
class WriteReceipt:
    """What `add()` returns. Explicit about what the write path actually did.

    `llm_calls` is here on purpose: it is the number the whole write-path design is
    trying to drive to zero, so it should be visible in normal use, not buried in a
    metrics backend.
    """

    episode_ids: list[str] = field(default_factory=list)
    added: list[Claim] = field(default_factory=list)
    #: Claims this write closed out, as they read *after* the write. The name predates
    #: the two axes and is kept because it is on the published API: read it as "closed
    #: out", not as "`invalidated_at` was set". Under the default `close="ended"` these
    #: are claims whose *valid* time was closed — still believed, no longer in force —
    #: and `Claim.state` on each is the field that says which axis actually moved.
    invalidated: list[Claim] = field(default_factory=list)
    reinforced: list[Claim] = field(default_factory=list)  # already known, salience bumped
    skipped: int = 0                                       # turns that carried no durable fact
    #: Turns that got all the way to the extraction tier and yielded nothing. Distinct
    #: from `skipped`, which is the write path working as designed (an acknowledgement
    #: carries no fact). This one is the honest count of *lost* content: with no model
    #: configured it is where a conversation's facts go, and without it the default
    #: configuration reports a clean, successful, empty write.
    unextracted: int = 0
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

    def __str__(self) -> str:
        # `unextracted` appears only when it is non-zero, so it reads as an event rather
        # than as noise on the writes that lost nothing.
        lost = f" unextracted={self.unextracted}" if self.unextracted else ""
        return (
            f"<WriteReceipt +{len(self.added)} ~{len(self.reinforced)} "
            f"-{len(self.invalidated)} skip={self.skipped}{lost} "
            f"llm={self.llm_calls} "
            f"{self.latency_ms:.1f}ms{' deferred' if self.deferred else ''}>"
        )

    # The dataclass repr dumps every field of every nested Claim, which is unreadable at
    # a REPL and buries the one number the write path exists to minimize.
    __repr__ = __str__
