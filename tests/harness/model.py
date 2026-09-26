"""A pure-Python model of memvara's claim store over both clocks, for one scope per user.

The state machine in `tests/adversarial/model/` applies each operation to the model and
to a real store, then compares every read. The model is deliberately small: a dictionary
of rows and the rules copied from the code, each with the place it was copied from.

**What it models:**
- the two clocks and the three states: `memvara/store/base.py::state_predicate`;
- expiry, where reads drop a row whose `expires_at` has passed on the wall clock;
- `remember`, `forget`, `delete`, `erase` and `erase_expired`, from `Reconciler.apply`,
  `types.close_out` and `Memvara`'s methods of the same names, with `remember`'s
  `valid_to` and its refusal of an end at or before the start;
- `stats()`, `count()`, `history()` and `get()`.

**What it leaves out,** so a reader does not take its silence for a pass:
- projects, agents and sessions, which the second stage adds together with the #266 fix;
- `supersede`, `remember(replaces=...)`, links, documents and episodes;
- two handles on one file, reopening, and encrypted stores;
- ranking: `search` is checked only for returning visible rows, at most `k` of them;
- graph traversal, `ask()`'s reconstructed readings, and recall rendering;
- temporal precision: every `valid_from` here is an exact instant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Collection

from memvara.schema import PredicateRegistry
# A candidate closes a row only if its confidence is at least this share of the row's.
from memvara.write.reconcile import AUTHORITY_SHARE

STATES: tuple[str, ...] = ("live", "ended", "retired")


@dataclass
class Row:
    """One claim as the model holds it. `obj` is compared exactly: the pools the state
    machine draws from contain no two values that memvara's entity fold would merge."""

    id: str
    user: str
    subject: str
    predicate: str
    obj: str
    polarity: int
    confidence: float
    valid_from: datetime
    recorded_at: datetime
    valid_to: datetime | None = None
    invalidated_at: datetime | None = None
    expires_at: datetime | None = None
    observations: int = 1

    @property
    def state(self) -> str:
        """`Claim.state`: absolute, read from the two closure columns alone."""
        if self.invalidated_at is not None:
            return "retired"
        if self.valid_to is not None:
            return "ended"
        return "live"

    @property
    def slot(self) -> tuple[str, str, str]:
        return (self.user, self.subject, self.predicate)

    @property
    def value(self) -> tuple[str, str, str, str, int]:
        """What `value_key` compares, for one scope per user."""
        return (self.user, self.subject, self.predicate, self.obj, self.polarity)

    def expired(self, now: datetime) -> bool:
        return self.expires_at is not None and self.expires_at <= now


def in_states(row: Row, states: Collection[str], valid_at: datetime,
              known_at: datetime) -> bool:
    """`state_predicate` for one row: the belief floor, then the requested states.

    The belief floor is `recorded_at <= known_at` and never lifts. Asking for all three
    states is the floor alone, so a row recorded but not yet in force, which is in none of
    the three, is readmitted there.
    """
    wanted = set(states)
    if row.recorded_at > known_at:
        return False
    if wanted == set(STATES):
        return True
    world_states = frozenset(wanted - {"retired"})
    if world_states == {"live"}:
        world = row.valid_from <= valid_at and (row.valid_to is None or row.valid_to > valid_at)
    elif world_states == {"ended"}:
        world = row.valid_to is not None and row.valid_to <= valid_at
    elif world_states == {"live", "ended"}:
        world = row.valid_from <= valid_at
    else:
        world = False
    if "retired" not in wanted:
        believed = row.invalidated_at is None or row.invalidated_at > known_at
        return believed and world
    retired = row.invalidated_at is not None and row.invalidated_at <= known_at
    return retired or world


class ReferenceStore:
    """The rows one store should hold, and what each read should return."""

    def __init__(self) -> None:
        self.rows: dict[str, Row] = {}
        #: Ids erased so far. An erased row is gone from `rows` and from every read.
        self.erased: set[str] = set()
        #: The real claim id behind each handle, filled in by `drive.Pair`. Ties between
        #: rows recorded at the same instant break on it, as the reconciler's do.
        self.real_ids: dict[str, str] = {}
        self._count = 0

    def add(self, row: Row) -> None:
        self.rows[row.id] = row

    def visible(self, user: str, *, valid_at: datetime | None = None,
                known_at: datetime | None = None,
                states: Collection[str] = ("live",), now: datetime) -> set[str]:
        """`get_all`'s ids for one user. An instant that is not given is `now`, as the
        store reads the clock for an axis it was not given."""
        v = now if valid_at is None else valid_at
        k = now if known_at is None else known_at
        return {r.id for r in self.rows.values()
                if r.user == user and not r.expired(now) and in_states(r, states, v, k)}

    def stats(self, now: datetime) -> dict[str, int]:
        """`stats()` for the whole tenant. `claims` and `invalidated` read the raw
        columns; the live and ended counts use the state predicate, expiry included."""
        rows = list(self.rows.values())
        return {
            "claims": len(rows),
            "live_claims": sum(not r.expired(now) and in_states(r, ("live",), now, now)
                               for r in rows),
            "ended_claims": sum(not r.expired(now) and in_states(r, ("ended",), now, now)
                                for r in rows),
            "invalidated": sum(r.invalidated_at is not None for r in rows),
        }

    def history(self, user: str, subject: str, predicate: str, *,
                valid_at: datetime | None = None, known_at: datetime | None = None,
                now: datetime) -> list[str]:
        """`history()`: every unexpired row in the slot, in `(recorded_at, id)` order.
        Rows recorded at the same instant are ordered by their real claim id, as the
        store orders them, so the model breaks the tie on `real_ids` where it has one.

        The two clocks filter as `memvara/core.py::_in_timeline` does, which is not the
        state predicate: an axis that is not given filters nothing, and a retired row
        stays in the record."""
        slot = [r for r in self.rows.values()
                if r.slot == (user, subject, predicate) and not r.expired(now)
                and (known_at is None or r.recorded_at <= known_at)
                and (valid_at is None or (r.valid_from <= valid_at
                                          and (r.valid_to is None or r.valid_to > valid_at)))]
        return [r.id for r in sorted(slot, key=self._tie)]

    # -- writes ---------------------------------------------------------------------------
    #
    # Each write takes `t`, the wall clock read just before the real call. Every stamp
    # already in the model is earlier than `t`, and every instant in `clock.INSTANTS` is
    # too, so a decision that compares with the reconciler's own clock gives the same
    # answer with `t`. A stamp the store takes from the clock is left as a `Stamp` for the
    # pair to check and copy.

    def new_handle(self) -> str:
        self._count += 1
        return f"r{self._count}"

    def live_at(self, row: Row, t: datetime) -> bool:
        """`Claim.is_live(t)` on an unexpired row: what `competing_claims` and the
        duplicate lookup call live."""
        return not row.expired(t) and in_states(row, ("live",), t, t)

    def _tie(self, row: Row) -> tuple[datetime, str]:
        """`Reconciler._canonical_of`: the earliest recording wins, and the claim id
        breaks a tie."""
        return (row.recorded_at, self.real_ids.get(row.id, row.id))

    def remember(self, op: Remember, t: datetime) -> Expect:
        """`Reconciler.apply` for one candidate, in one scope per user, after
        `Memvara.remember`'s refusal of an end at or before the start."""
        began = op.valid_from or op.recorded_at or t
        if op.valid_to is not None and op.valid_to <= began:
            return Expect(refused=True)
        slot = (op.user, SUBJECT, op.predicate)
        value = (op.user, SUBJECT, op.predicate, op.obj, op.polarity)
        clock_start = op.valid_from is None and op.recorded_at is None
        valid_from = op.valid_from if op.valid_from is not None else op.recorded_at
        decide_from = t if valid_from is None else valid_from
        live = [r for r in self.rows.values() if r.slot == slot and self.live_at(r, t)]

        if op.polarity > 0:
            same = [r for r in self.rows.values() if r.value == value and self.live_at(r, t)]
            if (same and op.expires_at is None
                    and all(r.valid_from > decide_from for r in same)):
                # The same value from before every live row of it begins. A repeat that
                # names an expiry stays a repeat, below.
                end, covering = self._earlier_period(value, same, decide_from, op.valid_to, t)
                if covering is not None:
                    covering.observations += 1
                    return Expect(reinforced=[covering.id])
                e = Expect(added=True)
                row = self._new_row(op, valid_from, clock_start, t, e)
                row.valid_to = end
                return e
            if same:
                keep = min(same, key=self._tie)
                keep.observations += 1
                if op.expires_at is not None:
                    keep.expires_at = op.expires_at
                return Expect(reinforced=[keep.id])
            return self._add(op, slot, live, decide_from, valid_from, clock_start, t)

        matches = [r for r in live if r.obj == op.obj]
        if not matches:
            prior = [r for r in self.rows.values() if r.value == value]
            if prior:
                keep = min(prior, key=self._tie)
                keep.observations += 1
                return Expect(reinforced=[keep.id], reinforced_reported=False)
            if live:
                return Expect()
        return self._tombstone(op, matches, decide_from, valid_from, clock_start, t)

    def _new_row(self, op: Remember, valid_from: datetime | None, clock_start: bool,
                 t: datetime, e: Expect) -> Row:
        handle = self.new_handle()
        row = Row(id=handle, user=op.user, subject=SUBJECT, predicate=op.predicate,
                  obj=op.obj, polarity=op.polarity, confidence=op.confidence,
                  valid_from=valid_from if valid_from is not None else t,
                  recorded_at=op.recorded_at if op.recorded_at is not None else t,
                  valid_to=op.valid_to, expires_at=op.expires_at)
        if op.recorded_at is None:
            e.stamps.append(Stamp(handle, "recorded_at"))
        if clock_start:
            e.stamps.append(Stamp(handle, "valid_from"))
        self.add(row)
        e.new = handle
        return row

    def _earlier_period(self, value: tuple[str, str, str, str, int], live: list[Row],
                        start: datetime, valid_to: datetime | None,
                        t: datetime) -> tuple[datetime, Row | None]:
        """`Reconciler._earlier_period`: where a restatement that begins at `start`, before
        every live row of its value, ends, and the row that already holds that period.

        The period ends where the value's rows begin: the earliest live row, or an earlier
        row that runs up to it without a gap. A `valid_to` the caller gave that is earlier
        still wins. Only rows believed at `t` and not expired count."""
        held = [r for r in self.rows.values()
                if r.value == value and not r.expired(t) and r.recorded_at <= t
                and (r.invalidated_at is None or r.invalidated_at > t)]
        end = min(r.valid_from for r in live)
        while True:
            reaching = [r.valid_from for r in held
                        if start < r.valid_from < end and r.valid_to is not None
                        and r.valid_to >= end]
            if not reaching:
                break
            end = min(reaching)
        if valid_to is not None and valid_to < end:
            end = valid_to
        covering = [r for r in held
                    if r.valid_from <= start and r.valid_to is not None and r.valid_to >= end]
        return end, (min(covering, key=self._tie) if covering else None)

    def _close(self, row: Row, e: Expect, by: str, boundary: datetime | None,
               close: str, clock_boundary: bool) -> None:
        """`types.close_out` through `Reconciler._retire`: an ending at the successor's
        start, never before the row's own; or a retirement at the reconciler's clock."""
        e.closed.append(row.id)
        if close == "retired":
            if row.invalidated_at is None:
                e.stamps.append(Stamp(row.id, "invalidated_at"))
            return
        if clock_boundary:
            # The successor's start came from the clock, so it is later than this row's
            # start and the clamp cannot apply.
            e.stamps.append(Stamp(row.id, "valid_to", "valid_from_of", by))
            return
        assert boundary is not None
        edge = max(boundary, row.valid_from)
        if row.valid_to is None or row.valid_to > edge:
            row.valid_to = edge
            if edge == row.valid_from:
                e.collapsed.append(row.id)

    def _add(self, op: Remember, slot: tuple[str, str, str], live: list[Row],
             decide_from: datetime, valid_from: datetime | None, clock_start: bool,
             t: datetime) -> Expect:
        e = Expect(added=True)
        victims = ([r for r in live if r.polarity > 0 and r.obj != op.obj]
                   if op.predicate in FUNCTIONAL else [])
        newer = [r for r in victims if r.valid_from > decide_from]
        older = sorted((r for r in victims if not r.valid_from > decide_from), key=self._tie)
        if not older and op.predicate not in DECLARED:
            existing = len([r for r in live if r.polarity > 0])
            e.accumulated = existing or None
        keep = [v for v in older if op.confidence >= AUTHORITY_SHARE * v.confidence]
        e.disputed = [v.id for v in older if v not in keep]
        row = self._new_row(op, valid_from, clock_start, t, e)
        if newer:
            # History: a later value is on record, so this one ends where that begins, or
            # earlier if the caller gave an earlier end.
            boundary = min(r.valid_from for r in newer)
            if row.valid_to is None or row.valid_to > boundary:
                row.valid_to = boundary
        for v in keep:
            self._close(v, e, row.id, valid_from, op.close, clock_start)
        return e

    def _tombstone(self, op: Remember, matches: list[Row], decide_from: datetime,
                   valid_from: datetime | None, clock_start: bool, t: datetime) -> Expect:
        """A retraction: a row closed on both clocks at the reconciler's clock, then the
        matching live rows ended at the retraction's own start.

        The world clock closes at the clock or at the row's own start, whichever is later.
        That is the rule every closure follows (`types.not_before_start`), so a retraction
        dated in the future leaves a row whose interval is empty rather than inverted
        (#275)."""
        e = Expect()
        row = self._new_row(op, valid_from, clock_start, t, e)
        row.valid_to = None
        if valid_from is not None and valid_from > t:
            row.valid_to = valid_from
        else:
            e.stamps.append(Stamp(row.id, "valid_to"))
        row.invalidated_at = t
        e.stamps.append(Stamp(row.id, "invalidated_at"))
        for v in sorted(matches, key=self._tie):
            self._close(v, e, row.id, valid_from, op.close, clock_start)
        return e

    def forget(self, op: Forget, t: datetime) -> Expect:
        """`Memvara.forget`: retire, at the clock, every row in the slot that the store
        believes and that has not ended. That is the live rows and the rows stored to
        begin later; an ended row is left as it is."""
        e = Expect()
        slot = (op.user, SUBJECT, op.predicate)
        closed = [r for r in self.rows.values()
                  if r.slot == slot and not r.expired(t) and r.recorded_at <= t
                  and (r.invalidated_at is None or r.invalidated_at > t)
                  and (r.valid_to is None or r.valid_to > t)]
        for r in closed:
            e.stamps.append(Stamp(r.id, "invalidated_at"))
        e.closed = [r.id for r in closed]
        e.returned = sorted(e.closed)
        return e

    def delete(self, op: Delete, t: datetime) -> Expect:
        """`Memvara.delete`: close one row the caller can read, through `close_out`: a
        retirement at the clock, or an ending at the clock that never falls before the
        row's own start."""
        e = Expect(returned=False)
        row = self.rows.get(op.handle)
        if row is None or row.user != op.user or row.expired(t):
            return e
        e.returned = True
        if op.close == "retired":
            e.stamps.append(Stamp(row.id, "invalidated_at"))
            return e
        if row.valid_from > t:
            # The edge is the row's own start, later than the clock: a collapse.
            if row.valid_to is None or row.valid_to > row.valid_from:
                row.valid_to = row.valid_from
        elif row.valid_to is None or row.valid_to > t:
            e.stamps.append(Stamp(row.id, "valid_to"))
        return e

    def erase(self, op: Erase) -> Expect:
        """`Memvara.erase`: remove the row, expired or not, if the caller can read it."""
        row = self.rows.get(op.handle)
        if row is None or row.user != op.user:
            return Expect(returned=False)
        del self.rows[op.handle]
        self.erased.add(op.handle)
        return Expect(returned=True)

    def lapse(self, op: Lapse) -> Expect:
        row = self.rows.get(op.handle)
        if row is not None:
            row.expires_at = op.at
        return Expect()

    def erase_expired(self, now: datetime) -> Expect:
        """`Memvara.erase_expired`: every row whose expiry has passed, in every tenant."""
        gone = sorted(h for h, r in self.rows.items() if r.expired(now))
        for h in gone:
            del self.rows[h]
            self.erased.add(h)
        return Expect(returned=gone)


# -- operations ---------------------------------------------------------------------------
#
# One dataclass per operation, so a failing run prints as a list a person can read and
# `drive.replay` can run again. A row is named by its handle, `r1`, `r2`, ..., in the
# order rows were created, because a claim id is a random uuid and a replayed program
# needs names that are the same every time.

#: The two users. One scope per user in this stage.
USERS: tuple[str, ...] = ("u1", "u2")
#: The one subject every operation is about.
SUBJECT = "user"
#: The value pools. `lives_in` is single-valued; `likes` is declared many-valued;
#: `collects` is declared by nobody, so it is many-valued by default and reports an
#: accumulation. No two values in a pool fold to the same entity.
POOLS: dict[str, tuple[str, ...]] = {
    "lives_in": ("Berlin", "Paris", "Rome"),
    "likes": ("tea", "coffee"),
    "collects": ("stamps", "vinyl"),
}
_REGISTRY = PredicateRegistry()
#: The pool predicates that hold one value at a time, and those memvara's registry
#: declares at all, read from the registry so the model cannot drift from it.
#: `test_the_pools_hold_one_predicate_of_each_kind` checks each kind is still present.
FUNCTIONAL = frozenset(p for p in POOLS if _REGISTRY.spec(p).functional)
DECLARED = frozenset(p for p in POOLS if _REGISTRY.known(p))


@dataclass(frozen=True)
class Remember:
    """`remember()`. `polarity=-1` is a retraction. `expires_at` is written as given.
    `valid_to` is the end the caller gives, and one at or before the start is refused."""

    user: str
    predicate: str
    obj: str
    polarity: int = 1
    valid_from: datetime | None = None
    recorded_at: datetime | None = None
    confidence: float = 1.0
    close: str = "ended"
    expires_at: datetime | None = None
    valid_to: datetime | None = None


@dataclass(frozen=True)
class Forget:
    """`forget(subject, predicate)`: retire every row in the slot that is believed and has
    not ended, including a row stored to begin later."""

    user: str
    predicate: str


@dataclass(frozen=True)
class Delete:
    """`delete(claim_id, close=...)` on the row named by `handle`."""

    user: str
    handle: str
    close: str = "retired"


@dataclass(frozen=True)
class Erase:
    """`erase(claim_id)` on the row named by `handle`."""

    user: str
    handle: str


@dataclass(frozen=True)
class Lapse:
    """The row's expiry passes. Its `expires_at` is moved to `at` through the store, as
    `tests/test_expiry.py` does, because `remember()` refuses an expiry in the past."""

    handle: str
    at: datetime


@dataclass(frozen=True)
class EraseExpired:
    """`erase_expired()`."""


Op = Remember | Forget | Delete | Erase | Lapse | EraseExpired


# -- predictions --------------------------------------------------------------------------

@dataclass(frozen=True)
class Stamp:
    """A field the store fills from the wall clock, which the model cannot know ahead.

    `source="wall"` means the value must lie inside the operation's window, and is then
    copied from the store. `source="valid_from_of"` means the value must equal the named
    row's `valid_from` after that row's own stamps are copied: an ending dated at a
    successor whose start was itself taken from the clock.
    """

    handle: str
    name: str
    source: str = "wall"
    other: str = ""


@dataclass
class Expect:
    """What one operation should do, beyond the rows the model changed in place."""

    #: The handle of the row this operation adds, fact or tombstone.
    new: str | None = None
    #: Whether `receipt.added` holds the new row. A retraction's tombstone is stored but
    #: is not an added fact.
    added: bool = False
    #: Rows reinforced. A replayed retraction reinforces its tombstone while the receipt
    #: reports nothing, so `reinforced_reported` says whether the receipt shows it.
    reinforced: list[str] = field(default_factory=list)
    reinforced_reported: bool = True
    closed: list[str] = field(default_factory=list)
    disputed: list[str] = field(default_factory=list)
    collapsed: list[str] = field(default_factory=list)
    accumulated: int | None = None
    #: Stamps to take from the store once the operation has run.
    stamps: list[Stamp] = field(default_factory=list)
    #: What the method returns: forget's handles, delete's and erase's bool, and
    #: erase_expired's handles.
    returned: object = None
    #: The call raises `ValueError` and writes nothing.
    refused: bool = False
