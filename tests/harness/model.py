"""A pure-Python model of memvara's claim store over both clocks and the scopes of a user.

The state machine in `tests/adversarial/model/` applies each operation to the model and
to a real store, then compares every read. The model is deliberately small: a dictionary
of rows and the rules copied from the code, each with the place it was copied from.

**What it models:**
- the two clocks and the three states: `memvara/store/base.py::state_predicate`;
- expiry, where reads drop a row whose `expires_at` has passed on the wall clock;
- `remember`, `forget`, `delete`, `erase` and `erase_expired`, from `Reconciler.apply`,
  `types.close_out` and `Memvara`'s methods of the same names, with `remember`'s
  `valid_to` and its refusal of an end at or before the start, and `forget` and `delete`
  with either closure;
- `stats()`, `count()`, `history()` and `get()`;
- the scopes below a user that `LEVELS` names: sessions, an agent, a project, and a session
  inside that project. A new value ends only values at exactly its own scope, and a
  present-tense read takes a single-valued fact from the narrowest level of its chain that
  holds one (#266). A predicate declared global is filed with the project cleared, and a
  reader's chain includes the project-less form of its own session (#273). The chain is
  `Scope.ancestors`, the filing rule is `PredicateRegistry.slot_scope`, and the read rule
  is `memvara/retrieve/shadow.py`.

**What it leaves out,** so a reader does not take its silence for a pass:
- the tenant's own level, and the '*' and '' values a store from before 0.17 can hold;
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


#: A scope below a user, as (project, agent, session). `None` is a level not bound.
Level = tuple[str | None, str | None, str | None]
#: The one repository the model's scopes name.
PROJECT = "github.com/o/a"
#: The scopes a handle can be bound to below its user, by the name a program uses. Two
#: sibling sessions, an agent with and without a session, and a project with and without
#: a session: enough for each rule of #266 and #273 to have a writer and a reader on both
#: sides of it.
LEVELS: dict[str, Level] = {
    "": (None, None, None),
    "s1": (None, None, "s1"),
    "s2": (None, None, "s2"),
    "a1": (None, "a1", None),
    "a1/s1": (None, "a1", "s1"),
    "P": (PROJECT, None, None),
    "P/s1": (PROJECT, None, "s1"),
}


def chain(level: Level) -> list[Level]:
    """`Scope.ancestors` below the tenant: the reader's own scope, then its agent inside
    its project, the project, the project-less forms of its own session and agent, and the
    user, leaving out every level the reader is not bound to."""
    project, agent, session = level
    out = [level]
    if session is not None:
        out.append((project, agent, None))
    if agent is not None:
        out.append((project, None, None))
    if project is not None:
        if session is not None:
            out.append((None, agent, session))
        if agent is not None:
            out.append((None, agent, None))
        out.append((None, None, None))
    return list(dict.fromkeys(out))


def contains(slot: Level, level: Level) -> bool:
    """`Scope.contains` for one user: a level the slot leaves unbound matches any value."""
    return all(mine is None or mine == theirs for mine, theirs in zip(slot, level))


@dataclass
class Row:
    """One claim as the model holds it. `obj` is compared exactly: the pools the state
    machine draws from contain no two values that memvara's entity fold would merge.
    `level` is where the row is stored below its user."""

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
    level: Level = (None, None, None)

    @property
    def state(self) -> str:
        """`Claim.state`: absolute, read from the two closure columns alone."""
        if self.invalidated_at is not None:
            return "retired"
        if self.valid_to is not None:
            return "ended"
        return "live"

    @property
    def slot(self) -> tuple[str, str, str, str | None]:
        """What `fact_key` compares: the owner, the subject, the predicate and the project.
        It leaves out the agent and the session."""
        return (self.user, self.subject, self.predicate, self.level[0])

    @property
    def value(self) -> tuple[str, str, str, str, int]:
        """What `value_key` compares. It leaves out the project, the agent and the
        session."""
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


def _overlap(a_from: datetime, a_to: datetime | None, b_from: datetime,
             b_to: datetime | None) -> bool:
    """`reconcile._overlaps`: two half-open intervals share an instant. One of no length
    shares none."""
    if (a_to is not None and a_to <= a_from) or (b_to is not None and b_to <= b_from):
        return False
    return (a_to is None or a_to > b_from) and (b_to is None or b_to > a_from)


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
                states: Collection[str] = ("live",), now: datetime,
                level: str = "", shadowed: bool = True) -> set[str]:
        """`get_all`'s ids for one user bound to `level`. An instant that is not given is
        `now`, as the store reads the clock for an axis it was not given.

        A read given neither instant is in the present tense, and it leaves out what
        `shadowed_ids` names, unless `shadowed` is false, which is what `count()` needs."""
        v = now if valid_at is None else valid_at
        k = now if known_at is None else known_at
        reads = chain(LEVELS[level])
        found = {r.id for r in self.rows.values()
                 if r.user == user and r.level in reads and not r.expired(now)
                 and in_states(r, states, v, k)}
        if shadowed and valid_at is None and known_at is None:
            found -= self.shadowed_ids(level, found, now)
        return found

    def shadowed_ids(self, level: str, ids: Collection[str], now: datetime) -> set[str]:
        """`retrieve.shadow.shadowed`: the rows among `ids` that a present-tense read bound
        to `level` leaves out. A row is left out when it is live by its own columns, its
        predicate holds one value, and a level before its own in the reader's chain holds
        a row live now for the same user, subject and predicate. A level with a project
        cannot hold a row of a global predicate, so it is not asked about one."""
        reads = chain(LEVELS[level])
        hidden = set()
        for rid in ids:
            row = self.rows[rid]
            if row.state != "live" or row.predicate not in FUNCTIONAL:
                continue
            narrower = [lv for lv in reads[:reads.index(row.level)]
                        if row.predicate not in GLOBAL or lv[0] is None]
            if any(o.level in narrower and o.user == row.user and o.subject == row.subject
                   and o.predicate == row.predicate and self.live_at(o, now)
                   for o in self.rows.values()):
                hidden.add(rid)
        return hidden

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
                now: datetime, level: str = "") -> list[str]:
        """`history()`: every unexpired row in the slot, in `(recorded_at, id)` order.
        Rows recorded at the same instant are ordered by their real claim id, as the
        store orders them, so the model breaks the tie on `real_ids` where it has one.

        The slot is the one `level` files `predicate` at, and a row counts when that
        filing contains the row's level, so a broad reader reaches down into its sessions
        and never sideways. The two clocks filter as `memvara/core.py::_in_timeline` does,
        which is not the state predicate: an axis that is not given filters nothing, and a
        retired row stays in the record."""
        filed = filed_at(predicate, LEVELS[level])
        slot = [r for r in self.rows.values()
                if r.slot == (user, subject, predicate, filed[0])
                and contains(filed, r.level) and not r.expired(now)
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
        """`Reconciler.apply` for one candidate, after `Memvara.remember`'s refusal of an
        end at or before the start. The candidate is filed at `filed_at`, and every rule
        below that reads another row asks where that row is stored."""
        began = op.valid_from or op.recorded_at or t
        if op.valid_to is not None and op.valid_to <= began:
            return Expect(refused=True)
        level = filed_at(op.predicate, LEVELS[op.level])
        reads = chain(level)
        slot = (op.user, SUBJECT, op.predicate, level[0])
        value = (op.user, SUBJECT, op.predicate, op.obj, op.polarity)
        clock_start = op.valid_from is None and op.recorded_at is None
        valid_from = op.valid_from if op.valid_from is not None else op.recorded_at
        decide_from = t if valid_from is None else valid_from
        live = [r for r in self.rows.values() if r.slot == slot and self.live_at(r, t)]

        if op.polarity > 0:
            on_record = [r for r in self.rows.values()
                         if r.value == value and self.live_at(r, t)]
            # A repeat reinforces only a row its writer can see, and a repeat that names an
            # expiry only a row at exactly its own scope.
            same = [r for r in on_record if r.level in reads
                    and (op.expires_at is None or r.level == level)]
            if (same and op.expires_at is None
                    and all(r.valid_from > decide_from for r in same)):
                # The same value from before every live row of it begins. A repeat that
                # names an expiry stays a repeat, below.
                end, covering = self._earlier_period(value, same, decide_from, op.valid_to,
                                                     t, reads)
                if covering is not None:
                    covering.observations += 1
                    return Expect(reinforced=[covering.id])
                e = Expect(added=True)
                row = self._new_row(op, level, valid_from, clock_start, t, e)
                row.valid_to = end
                return e
            if same:
                keep = min(same, key=self._tie)
                keep.observations += 1
                if op.expires_at is not None:
                    keep.expires_at = op.expires_at
                return Expect(reinforced=[keep.id])
            # The same value is on record where this repeat may not reinforce it, so the
            # row written below is not a second answer beside it.
            separate = bool(on_record)
            return self._add(op, level, slot, live, decide_from, valid_from, clock_start,
                             t, separate)

        # `Reconciler._retract`'s slot: the rows the writer can see, without a row whose
        # retirement is recorded, which no later closure changes. A retraction that names
        # no value matches every one of them.
        closable = [r for r in live if r.invalidated_at is None and r.level in reads]
        matches = [r for r in closable if not op.obj or r.obj == op.obj]
        # Only a match the retraction would change: one already ended at or before the
        # retraction's start, by the same retraction dated in the future and sent before,
        # is left alone (#349).
        changed = [r for r in matches if op.close == "retired" or r.valid_to is None
                   or r.valid_to > max(decide_from, r.valid_from)]
        # The authority rule a new value faces (#307).
        closing = [r for r in changed if op.confidence >= AUTHORITY_SHARE * r.confidence]
        disputed = [r.id for r in changed if r not in closing]
        if not closing:
            # A tombstone whose expiry has passed does not count as the retraction on
            # record, just as `live_at` above leaves an expired row out of a repeated fact,
            # and nor does one the writer cannot see.
            prior = [r for r in self.rows.values()
                     if r.value == value and not r.expired(t) and r.level in reads]
            if prior:
                keep = min(prior, key=self._tie)
                keep.observations += 1
                return Expect(reinforced=[keep.id], reinforced_reported=False,
                              disputed=disputed)
            if closable and not matches:
                return Expect()
        return self._tombstone(op, level, closing, disputed, valid_from, clock_start, t)

    def _new_row(self, op: Remember, level: Level, valid_from: datetime | None,
                 clock_start: bool, t: datetime, e: Expect) -> Row:
        handle = self.new_handle()
        row = Row(id=handle, user=op.user, subject=SUBJECT, predicate=op.predicate,
                  obj=op.obj, polarity=op.polarity, confidence=op.confidence,
                  valid_from=valid_from if valid_from is not None else t,
                  recorded_at=op.recorded_at if op.recorded_at is not None else t,
                  valid_to=op.valid_to, expires_at=op.expires_at, level=level)
        if op.recorded_at is None:
            e.stamps.append(Stamp(handle, "recorded_at"))
        if clock_start:
            # Given neither instant, a write takes both from one reading under the lock,
            # unless it was given a `valid_to`, which was checked against the call.
            e.stamps.append(Stamp(handle, "valid_from", "recorded_at_of", handle)
                            if op.valid_to is None else Stamp(handle, "valid_from"))
        self.add(row)
        e.new = handle
        return row

    def _earlier_period(self, value: tuple[str, str, str, str, int], live: list[Row],
                        start: datetime, valid_to: datetime | None,
                        t: datetime, reads: list[Level]) -> tuple[datetime, Row | None]:
        """`Reconciler._earlier_period`: where a restatement that begins at `start`, before
        every live row of its value, ends, and the row that already holds that period.

        The period ends where the value's rows begin: the earliest live row, or an earlier
        row that runs up to it without a gap. A `valid_to` the caller gave that is earlier
        still wins. Only rows believed at `t`, not expired, and at a level the writer
        reads count."""
        held = [r for r in self.rows.values()
                if r.value == value and not r.expired(t) and r.recorded_at <= t
                and r.level in reads
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
               close: str, clock_boundary: bool, recorded_default: bool) -> None:
        """`types.close_out` through `Reconciler._retire`: an ending at the successor's
        start, never before the row's own; or a retirement at the reconciler's clock,
        which is the successor's `recorded_at` when the write was given none."""
        e.closed.append(row.id)
        if close == "retired":
            if row.invalidated_at is None:
                e.stamps.append(Stamp(row.id, "invalidated_at", "recorded_at_of", by)
                                if recorded_default else Stamp(row.id, "invalidated_at"))
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

    def _add(self, op: Remember, level: Level, slot: tuple[str, str, str, str | None],
             live: list[Row], decide_from: datetime, valid_from: datetime | None,
             clock_start: bool, t: datetime, separate: bool) -> Expect:
        e = Expect(added=True)
        # Every value at exactly the writer's scope recorded by `t` that is neither retired
        # nor ended by then, which includes one written to begin later, and only one true
        # at some instant the new row is true (`Reconciler._at_scope`). A row whose
        # retirement is recorded competes with nothing, even one still believed at `t`: no
        # later closure changes it (`Reconciler._occupants`).
        victims = ([r for r in self.rows.values()
                    if r.slot == slot and r.level == level
                    and r.polarity > 0 and r.obj != op.obj
                    and not r.expired(t) and r.recorded_at <= t
                    and r.invalidated_at is None
                    and (r.valid_to is None or r.valid_to > t)
                    and _overlap(r.valid_from, r.valid_to, decide_from, op.valid_to)]
                   if op.predicate in FUNCTIONAL else [])
        newer = [r for r in victims if r.valid_from > decide_from]
        older = sorted((r for r in victims if not r.valid_from > decide_from), key=self._tie)
        if not older and not separate and op.predicate not in DECLARED:
            # `count_competing` counts the whole slot, every session and agent included.
            existing = len([r for r in live if r.polarity > 0])
            e.accumulated = existing or None
        # History: a later value is on record, so this one ends where that begins, or
        # earlier if the caller gave an earlier end.
        end = op.valid_to
        if newer:
            boundary = min(r.valid_from for r in newer)
            if end is None or end > boundary:
                end = boundary
        if end is not None:
            # `Reconciler.apply`, step 3: a row that is already over is a repeat when a
            # believed row of its value already holds its whole period (#351).
            value = (op.user, SUBJECT, op.predicate, op.obj, op.polarity)
            # Only a row the writer can see counts, and for a repeat that names an expiry,
            # only one at exactly its own scope (`Reconciler._held`).
            covering = [r for r in self.rows.values()
                        if r.value == value and not r.expired(t) and r.recorded_at <= t
                        and (r.invalidated_at is None or r.invalidated_at > t)
                        and r.level in chain(level)
                        and (op.expires_at is None or r.level == level)
                        and r.valid_from <= decide_from
                        and r.valid_to is not None and r.valid_to >= end]
            if covering:
                repeat = min(covering, key=self._tie)
                repeat.observations += 1
                if op.expires_at is not None:
                    repeat.expires_at = op.expires_at
                return Expect(reinforced=[repeat.id])
        keep = [v for v in older if op.confidence >= AUTHORITY_SHARE * v.confidence]
        e.disputed = [v.id for v in older if v not in keep]
        row = self._new_row(op, level, valid_from, clock_start, t, e)
        row.valid_to = end
        for v in keep:
            self._close(v, e, row.id, valid_from, op.close, clock_start,
                        op.recorded_at is None)
        return e

    def _tombstone(self, op: Remember, level: Level, matches: list[Row],
                   disputed: list[str], valid_from: datetime | None,
                   clock_start: bool, t: datetime) -> Expect:
        """A retraction: a row closed on both clocks at the instant it is recorded, then
        the matching live rows ended at the retraction's own start.

        That instant is the reconciler's clock, or the `recorded_at` the write was given,
        so a backdated tombstone is believed at no instant (#317). The world clock closes
        there or at the row's own start, whichever is later. That is the rule every
        closure follows (`types.not_before_start`), so a retraction dated in the future
        leaves a row whose interval is empty rather than inverted (#275)."""
        e = Expect(disputed=disputed)
        row = self._new_row(op, level, valid_from, clock_start, t, e)
        if op.recorded_at is not None:
            row.invalidated_at = op.recorded_at
            row.valid_to = max(op.recorded_at, row.valid_from)
        else:
            e.stamps.append(Stamp(row.id, "invalidated_at", "recorded_at_of", row.id))
            if valid_from is not None and valid_from > t:
                row.valid_to = valid_from
            else:
                e.stamps.append(Stamp(row.id, "valid_to", "recorded_at_of", row.id))
        for v in sorted(matches, key=self._tie):
            self._close(v, e, row.id, valid_from, op.close, clock_start,
                        op.recorded_at is None)
        return e

    def forget(self, op: Forget, t: datetime) -> Expect:
        """`Memvara.forget`: close, at the clock, every row in the slot that the store
        believes and that has not ended. That is the live rows and the rows stored to
        begin later; an ended row is left as it is, and so is a row whose retirement is
        already recorded, even one dated to take effect later, which no closure changes
        (`Reconciler._victims`).

        Each row is closed through `close_out`: a retirement at the clock, or an ending at
        the clock that never falls before the row's own start. So under `close="ended"` a
        row stored to begin later ends at its start and is true at no instant."""
        e = Expect()
        filed = filed_at(op.predicate, LEVELS[op.level])
        slot = (op.user, SUBJECT, op.predicate, filed[0])
        # A broad caller reaches down into its sessions, and never sideways.
        closed = [r for r in self.rows.values()
                  if r.slot == slot and contains(filed, r.level)
                  and not r.expired(t) and r.recorded_at <= t
                  and r.invalidated_at is None
                  and (r.valid_to is None or r.valid_to > t)]
        for r in closed:
            if op.close == "retired":
                e.stamps.append(Stamp(r.id, "invalidated_at"))
            elif r.valid_from > t:
                # The edge is the row's own start, later than the clock.
                if r.valid_to is None or r.valid_to > r.valid_from:
                    r.valid_to = r.valid_from
            else:
                # The row has begun and ends after the clock or never, so it now ends at
                # the clock: no instant the model holds lies between `t` and the clock.
                e.stamps.append(Stamp(r.id, "valid_to"))
        e.closed = [r.id for r in closed]
        e.returned = sorted(e.closed)
        return e

    def delete(self, op: Delete, t: datetime) -> Expect:
        """`Memvara.delete`: close one row the caller can read, through `close_out`: a
        retirement at the clock, or an ending at the clock that never falls before the
        row's own start. A row already retired is left as it is, whichever closure is
        asked for, and the call still returns True."""
        e = Expect(returned=False)
        row = self.rows.get(op.handle)
        if row is None or not self.readable(row, op.user, op.level) or row.expired(t):
            return e
        e.returned = True
        if row.invalidated_at is not None:
            return e
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
        if row is None or not self.readable(row, op.user, op.level):
            return Expect(returned=False)
        del self.rows[op.handle]
        self.erased.add(op.handle)
        return Expect(returned=True)

    @staticmethod
    def readable(row: Row, user: str, level: str) -> bool:
        """`Scope.sees`: the row belongs to `user` and is stored at a level of the chain
        of a reader bound to `level`."""
        return row.user == user and row.level in chain(LEVELS[level])

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

#: The two users.
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
#: The pool predicates declared global, which are filed with the project cleared.
GLOBAL = frozenset(p for p in POOLS if not _REGISTRY.spec(p).project_scoped)


def filed_at(predicate: str, level: Level) -> Level:
    """`PredicateRegistry.slot_scope`: a global predicate is filed with the project
    cleared, and the agent and the session kept."""
    return (None, *level[1:]) if predicate in GLOBAL else level


@dataclass(frozen=True)
class Remember:
    """`remember()`. `polarity=-1` is a retraction, and one whose `obj` is empty names no
    value, so it retracts every value in the slot. `expires_at` is written as given.
    `valid_to` is the end the caller gives, and one at or before the start is refused.
    `level` names the scope in `LEVELS` the writer is bound to, as on every operation
    below that has one."""

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
    level: str = ""


@dataclass(frozen=True)
class Forget:
    """`forget(subject, predicate, close=...)`: close every row in the slot that is
    believed and has not ended, including a row stored to begin later. `close` is
    `"retired"`, the default, or `"ended"`."""

    user: str
    predicate: str
    close: str = "retired"
    level: str = ""


@dataclass(frozen=True)
class Delete:
    """`delete(claim_id, close=...)` on the row named by `handle`."""

    user: str
    handle: str
    close: str = "retired"
    level: str = ""


@dataclass(frozen=True)
class Erase:
    """`erase(claim_id)` on the row named by `handle`."""

    user: str
    handle: str
    level: str = ""


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
    successor whose start was itself taken from the clock. `source="recorded_at_of"` is
    the same for the named row's `recorded_at`: a retirement at the instant a write that
    was given no `recorded_at` recorded its own row, which is the one instant that write
    takes from the clock under the write lock, and the start of a row written with
    neither `valid_from` nor `recorded_at`, which takes that same instant.
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
