"""Run the same operations on a real store and on the reference model, and compare them.

`Pair.apply(op)` runs one operation on each, then `check()` compares every row, every
receipt field the model predicts, and every read. A stamp the store takes from the wall
clock is checked against the window the operation ran in and then copied into the model,
so the model predicts which rows change and how, never the exact microsecond.

Some checks read only the store, so they still hold if the model and the store were ever
wrong in the same way:
- no read returns a row recorded after the read's `known_at` (the belief floor, I5);
- a positive write that stores a row live now can be read back at once (I6);
- `get` returns nothing for an erased row (I7);
- no read by one user returns another user's row (I8);
- a positive write under a predicate nobody declared closes no row (I11).

`replay(ops)` runs a program on a fresh pair and raises `ModelDivergence` at the first
operation where the two disagree. `format_program(ops)` prints a program in the form
`replay` accepts, which is how a failure found at random becomes a fixed test.
"""

from __future__ import annotations

import itertools
import pathlib
from datetime import datetime, timedelta
from types import TracebackType
from typing import Any, Sequence

from memvara import Memvara, NullLLM
from memvara.embed import HashingEmbedder
from memvara.types import Claim, utcnow

from harness.clock import FAR_FUTURE, INSTANTS, within
from harness.model import (DECLARED, POOLS, STATES, SUBJECT, USERS, Delete, Erase,
                           EraseExpired, Expect, Forget, Lapse, Op, ReferenceStore,
                           Remember, Row)

#: Every non-empty set of states a read can ask for.
STATE_SETS: list[tuple[str, ...]] = [
    combo for n in (1, 2, 3) for combo in itertools.combinations(STATES, n)]
#: The instant pairs every check reads at, as `(valid_at, known_at)`, beside the present.
#: `None` on one axis reads the clock there. The pairs cover both orders of the two clocks,
#: each clock alone, and a valid time after every scheduled value begins. Every pair of
#: `INSTANTS` would be 36 pairs and several times the run time for the same branches.
READ_AT: list[tuple[datetime | None, datetime | None]] = [
    (INSTANTS[0], INSTANTS[0]), (INSTANTS[2], INSTANTS[4]), (INSTANTS[4], INSTANTS[2]),
    (INSTANTS[5], None), (None, INSTANTS[3]), (FAR_FUTURE + timedelta(days=1), None)]
#: For each row in the store, its two closing stamps: `valid_to` and `invalidated_at`.
Clocks = dict[str, tuple[datetime | None, datetime | None]]
#: The fields compared on every row.
FIELDS = ("valid_from", "valid_to", "recorded_at", "invalidated_at", "expires_at")


class ModelDivergence(AssertionError):
    """The store and the model disagree after the operation at `index`."""

    def __init__(self, index: int, op: Op, detail: str) -> None:
        super().__init__(f"after operation {index}, {op!r}: {detail}")
        self.index = index
        self.op = op
        self.detail = detail


class Pair:
    """A real store and the reference model, fed the same operations. The store is in
    memory, or in the SQLite file at `path` when one is given."""

    def __init__(self, path: pathlib.Path | None = None) -> None:
        self.mem = Memvara(None if path is None else str(path),
                           embedder=HashingEmbedder(dim=64), llm=NullLLM())
        self.model = ReferenceStore()
        self.ops: list[Op] = []
        #: Erased rows keep their real id here, so a later operation can still name them.
        self.handles: dict[str, str] = {}

    def __enter__(self) -> "Pair":
        return self

    def __exit__(self, kind: type[BaseException] | None, value: BaseException | None,
                 tb: TracebackType | None) -> None:
        self.close()

    def close(self) -> None:
        self.mem.close()

    def real(self, handle: str) -> str:
        return self.model.real_ids[handle]

    def handle_of(self, real_id: str) -> str:
        return self.handles.get(real_id, f"<unknown {real_id}>")

    def lapsed_instant(self) -> datetime:
        """One second ago: an expiry that has already passed on the wall clock."""
        return utcnow() - timedelta(seconds=1)

    # -- one operation --------------------------------------------------------------------

    def apply(self, op: Op) -> Expect:
        """Run `op` on both sides and compare. Raises `ModelDivergence` on a difference."""
        index = len(self.ops)
        self.ops.append(op)
        clocks = self._clocks()
        before = utcnow()
        expect, result = self._run(op, before)
        after = utcnow()
        clocks_after = self._clocks()
        try:
            self._bind(expect, set(clocks), set(clocks_after))
            self._stamp(expect, before, after)
            self._check_undeclared(op, clocks, clocks_after)
            self._compare_result(op, expect, result)
            self._check_read_back(op, expect)
            self.check(set(clocks_after))
        except AssertionError as exc:
            if isinstance(exc, ModelDivergence):
                raise
            raise ModelDivergence(index, op, str(exc)) from exc
        return expect

    def _run(self, op: Op, t: datetime) -> tuple[Expect, Any]:
        m, mem = self.model, self.mem
        if isinstance(op, Remember):
            expect = m.remember(op, t)
            try:
                result: Any = mem.remember(
                    SUBJECT, op.predicate, op.obj, user=op.user, polarity=op.polarity,
                    valid_from=op.valid_from, recorded_at=op.recorded_at,
                    confidence=op.confidence, close=op.close, expires_at=op.expires_at,
                    valid_to=op.valid_to)
            except ValueError as exc:
                result = exc
        elif isinstance(op, Forget):
            expect = m.forget(op, t)
            result = mem.forget(SUBJECT, op.predicate, user=op.user)
        elif isinstance(op, Delete):
            expect = m.delete(op, t)
            result = mem.delete(self.real(op.handle), user=op.user, close=op.close)
        elif isinstance(op, Erase):
            real_id = self.real(op.handle)
            expect = m.erase(op)
            result = mem.erase(real_id, user=op.user)
        elif isinstance(op, Lapse):
            expect = m.lapse(op)
            claim = mem.store.get_claim(self.real(op.handle))
            if claim is not None:
                claim.expires_at = op.at
                mem.store.put_claim(claim)
            result = None
        else:
            assert isinstance(op, EraseExpired)
            expect = m.erase_expired(t)
            result = mem.erase_expired()
        return expect, result

    def _row_ids(self) -> set[str]:
        """Every row in the store: all three states, expired rows included."""
        return set(self._clocks())

    def _clocks(self) -> Clocks:
        return {c.id: (c.valid_to, c.invalidated_at)
                for c in self.mem.store.iter_claims(None, True)}

    def _bind(self, expect: Expect, before_ids: set[str], after_ids: set[str]) -> None:
        added = after_ids - before_ids
        if expect.new is None:
            assert not added, f"the store added rows the model did not: {sorted(added)}"
            return
        assert len(added) == 1, f"expected one new row, the store added {len(added)}"
        real_id = added.pop()
        self.model.real_ids[expect.new] = real_id
        self.handles[real_id] = expect.new

    def _stamp(self, expect: Expect, before: datetime, after: datetime) -> None:
        # Stamps taken from the clock first; then those equal to another row's start,
        # which may itself have come from the clock.
        for stamp in sorted(expect.stamps, key=lambda s: s.source != "wall"):
            row = self.model.rows[stamp.handle]
            claim = self.mem.store.get_claim(self.real(stamp.handle))
            assert claim is not None, f"{stamp.handle} is missing from the store"
            got = getattr(claim, stamp.name)
            if stamp.source == "wall":
                assert got is not None and within(got, before, after), (
                    f"{stamp.handle}.{stamp.name} is {got}, outside the operation's "
                    f"window {before} .. {after}")
            else:
                want = self.model.rows[stamp.other].valid_from
                assert got == want, (
                    f"{stamp.handle}.{stamp.name} is {got}, not {stamp.other}'s start {want}")
            setattr(row, stamp.name, got)
        new = expect.new
        if new is not None and self.model.rows[new].polarity < 0:
            row = self.model.rows[new]
            assert row.valid_to == row.invalidated_at, (
                f"the tombstone {new} closed its clocks at two instants")

    def _check_undeclared(self, op: Op, clocks: Clocks, clocks_after: Clocks) -> None:
        """I11: a positive write under a predicate nobody declared closes no row."""
        if not isinstance(op, Remember) or op.polarity < 0 or op.predicate in DECLARED:
            return
        moved = sorted(self.handle_of(i) for i, stamps in clocks.items()
                       if clocks_after.get(i) != stamps)
        assert not moved, (
            f"a write under {op.predicate}, which nobody declared, closed {moved}")

    def _check_read_back(self, op: Op, expect: Expect) -> None:
        """I6: a positive write that stores a row live now can be read back at once."""
        if not isinstance(op, Remember) or op.polarity < 0 or expect.new is None:
            return
        claim = self.mem.store.get_claim(self.real(expect.new))
        assert claim is not None, f"{expect.new} is missing from the store"
        now = utcnow()
        live = (claim.invalidated_at is None and claim.recorded_at <= now
                and claim.valid_from <= now
                and (claim.valid_to is None or claim.valid_to > now)
                and (claim.expires_at is None or claim.expires_at > now))
        if live:
            assert claim.id in {c.id for c in self.mem.get_all(user=op.user)}, (
                f"{op.user} cannot read back {expect.new}, a live row it has just written")

    def _compare_result(self, op: Op, expect: Expect, result: Any) -> None:
        if expect.refused:
            assert isinstance(result, ValueError), f"{op!r} was not refused: {result!r}"
            return
        if isinstance(result, ValueError):
            raise AssertionError(f"the store refused {op!r}: {result}")
        if isinstance(op, Remember):
            self._compare_receipt(expect, result)
        elif isinstance(op, Forget):
            got = sorted(self.handle_of(c.id) for c in result)
            assert got == expect.returned, f"forget closed {got}, not {expect.returned}"
        elif isinstance(op, (Delete, Erase)):
            assert result is expect.returned, f"returned {result}, not {expect.returned}"
        elif isinstance(op, EraseExpired):
            got = sorted(self.handle_of(e.claim_id) for e in result)
            assert got == expect.returned, f"erase_expired erased {got}, not {expect.returned}"

    def _compare_receipt(self, e: Expect, receipt: Any) -> None:
        h = self.handle_of
        added = [h(c.id) for c in receipt.added]
        want_added = [e.new] if e.added and e.new is not None else []
        assert added == want_added, f"receipt.added is {added}, not {want_added}"
        reinforced = [h(c.id) for c in receipt.reinforced]
        want = e.reinforced if e.reinforced_reported else []
        assert reinforced == want, f"receipt.reinforced is {reinforced}, not {want}"
        closed = sorted(h(c.id) for c in receipt.closed)
        assert closed == sorted(e.closed), f"receipt.closed is {closed}, not {sorted(e.closed)}"
        disputed = sorted(h(d.claim_id) for d in receipt.disputed)
        assert disputed == sorted(e.disputed), f"disputed {disputed}, not {sorted(e.disputed)}"
        collapsed = sorted(h(c.claim_id) for c in receipt.collapsed)
        assert collapsed == sorted(e.collapsed), (
            f"collapsed {collapsed}, not {sorted(e.collapsed)}")
        accumulated = [a.existing for a in receipt.accumulated]
        want_acc = [e.accumulated] if e.accumulated else []
        assert accumulated == want_acc, f"accumulated {accumulated}, not {want_acc}"
        assert receipt.llm_calls == 0, f"an offline write made {receipt.llm_calls} model calls"

    # -- the whole store ------------------------------------------------------------------

    def check(self, ids: set[str] | None = None) -> None:
        """Compare every row and every read. Raises `AssertionError` on a difference.
        `ids` is the store's rows when the caller has just read them."""
        self._check_rows(self._row_ids() if ids is None else ids)
        now = utcnow()
        for user in USERS:
            self._check_reads(user, now)
        stats = self.mem.stats()
        got = {k: stats[k] for k in ("claims", "live_claims", "ended_claims", "invalidated")}
        want = self.model.stats(now)
        assert got == want, f"stats() is {got}, not {want}"

    def _check_rows(self, ids: set[str]) -> None:
        want_ids = {self.real(h) for h in self.model.rows}
        assert ids == want_ids, (
            f"rows in the store {sorted(self.handle_of(i) for i in ids - want_ids)} "
            f"are not in the model; rows in the model "
            f"{sorted(self.handle_of(i) for i in want_ids - ids)} are not in the store")
        for handle, row in self.model.rows.items():
            claim = self.mem.store.get_claim(self.real(handle))
            assert claim is not None
            self._compare_row(handle, row, claim)
        for handle in self.model.erased:
            real_id = self.real(handle)
            assert self.mem.store.get_claim(real_id) is None
            assert self.mem.store.erasure_record(real_id) is not None, (
                f"{handle} was erased with no audit record")
            for user in USERS:
                assert self.mem.get(real_id, user=user) is None, (
                    f"{user}: get() still returns the erased row {handle}")

    @staticmethod
    def _compare_row(handle: str, row: Row, claim: Claim) -> None:
        for name in FIELDS:
            got, want = getattr(claim, name), getattr(row, name)
            assert got == want, f"{handle}.{name} is {got}, the model has {want}"
        assert claim.observation_count == row.observations, (
            f"{handle} was observed {claim.observation_count} times, "
            f"the model has {row.observations}")
        assert (claim.polarity, claim.object, claim.confidence) == (
            row.polarity, row.obj, row.confidence), f"{handle}'s value differs"
        if claim.valid_to is not None:
            assert claim.valid_to >= claim.valid_from, (
                f"{handle} ends at {claim.valid_to}, before it starts at {claim.valid_from}")

    def _check_search(self, user: str, value: str, valid_at: datetime | None,
                      known_at: datetime | None, now: datetime) -> None:
        """I12: `search(k=3)` returns at most three rows, none twice, all visible."""
        found = [r.claim for r in self.mem.search(value, k=3, user=user, valid_at=valid_at,
                                                  known_at=known_at)]
        hits = self._read(user, found, now if known_at is None else known_at)
        ids = [c.id for c in found]
        twice = sorted({self.handle_of(i) for i in ids if ids.count(i) > 1})
        assert not twice, f"{user}: search({value!r}) returned {', '.join(twice)} more than once"
        assert len(found) <= 3, f"{user}: search({value!r}, k=3) returned {len(found)} rows"
        visible = self.model.visible(user, valid_at=valid_at, known_at=known_at, now=now)
        assert hits <= visible, (
            f"{user}: search({value!r}, valid_at={valid_at}, known_at={known_at}) returned "
            f"{sorted(hits - visible)}, which are not live in that view")

    def _read(self, user: str, claims: Sequence[Claim], known_at: datetime) -> set[str]:
        """The handles of the rows a read by `user` returned, after two checks that do
        not use the model: every row belongs to `user` (I8), and none was recorded after
        the read's `known_at` (I5)."""
        for c in claims:
            assert c.scope.user == user, (
                f"{user}: a read returned {self.handle_of(c.id)}, which belongs to "
                f"{c.scope.user}")
            assert c.recorded_at <= known_at, (
                f"{user}: a read returned {self.handle_of(c.id)}, recorded at "
                f"{c.recorded_at}, after the read's known_at {known_at}")
        return {self.handle_of(c.id) for c in claims}

    def _check_reads(self, user: str, now: datetime) -> None:
        mem, model = self.mem, self.model
        live = model.visible(user, now=now)
        got = self._read(user, mem.get_all(user=user), now)
        assert got == live, f"{user}: get_all() is {sorted(got)}, the model has {sorted(live)}"
        assert mem.count(user=user) == len(live)
        values = [v for pool in POOLS.values() for v in pool]
        for value in values:
            self._check_search(user, value, None, None, now)
        for i, (valid_at, known_at) in enumerate([(None, None), *READ_AT]):
            floor = now if known_at is None else known_at
            if i:
                # One search per pair, a different value each time: `search` is the
                # costliest read, and the present already searches for every value.
                self._check_search(user, values[i % len(values)], valid_at, known_at, now)
            for states in STATE_SETS:
                got = self._read(user, mem.get_all(user=user, valid_at=valid_at,
                                                   known_at=known_at, states=states), floor)
                want = model.visible(user, valid_at=valid_at, known_at=known_at,
                                     states=states, now=now)
                assert got == want, (
                    f"{user}: get_all(valid_at={valid_at}, known_at={known_at}, "
                    f"states={states}) is {sorted(got)}, the model has {sorted(want)}")
            for predicate in POOLS:
                got_history = [self.handle_of(c.id) for c in mem.history(
                    SUBJECT, predicate, user=user, valid_at=valid_at, known_at=known_at)]
                want_history = model.history(user, SUBJECT, predicate, valid_at=valid_at,
                                             known_at=known_at, now=now)
                assert got_history == want_history, (
                    f"{user}: history({predicate}, valid_at={valid_at}, "
                    f"known_at={known_at}) is {got_history}, the model has {want_history}")
        at = INSTANTS[2]
        got = self._read(user, mem.get_all(user=user, as_of=at), at)
        want = model.visible(user, valid_at=at, known_at=at, now=now)
        assert got == want, f"{user}: get_all(as_of={at}) is {sorted(got)}, not {sorted(want)}"
        for handle, row in model.rows.items():
            got_one = mem.get(self.real(handle), user=user)
            readable = row.user == user and not row.expired(now)
            assert (got_one is not None) == readable, (
                f"{user}: get({handle}) returned {got_one is not None}, expected {readable}")


def replay(ops: Sequence[Op], path: pathlib.Path | None = None) -> None:
    """Run a program on a fresh pair, in memory or in a store file at `path`. Raises
    `ModelDivergence` at the first difference."""
    with Pair(path) as pair:
        for op in ops:
            pair.apply(op)


def format_program(ops: Sequence[Op]) -> str:
    """The program as Python that `replay` accepts, one operation per line."""
    lines = ",\n    ".join(repr(op) for op in ops)
    return f"[\n    {lines},\n]" if ops else "[]"
