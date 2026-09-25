"""Random operation sequences, each run on a real store and on the reference model.

Every step runs through `drive.Pair.apply`, which compares every row, every predicted
receipt field and every read with the model, and makes the checks that read only the
store (invariants I4 to I13 in the plan). The machine adds the checks that need the
history of a run rather than one step, and one check at the end of each run:
- no step closes both clocks of a row that already existed (I1);
- a closed clock never moves later, and an ending is never cleared (I2);
- a row disappears from the store only through `erase` or `erase_expired` (I3);
- when the run ends, the store file passes `check_store_integrity` (I14). Each run keeps
  its store in a file of its own for this reason, and the file is deleted afterwards.

The value pools are small, so two operations often touch the same slot or value, and a
second `remember` rule writes only `lives_in` at two instants, so that two values often
begin at the same moment.

Operations that would trigger a known, open bug are skipped, so the machine keeps looking
for new bugs rather than rediscovering old ones. The test prints how many steps it ran and
how many it skipped for each bug (`pytest -s` shows the line), and it fails if skips grow
past a tenth of all steps. `STEERED` must name only bugs still registered in `known_bugs`,
so a fixed bug's detour cannot outlive it.

When a step diverges, the failure carries the whole program so far, in the form
`drive.replay` accepts: paste it into a test to reproduce the failure without Hypothesis.

The number of runs and steps comes from the tier's Hypothesis profile, so
`--tier nightly` runs this far longer than the fast tier does.
"""

from __future__ import annotations

import pathlib
import shutil
import sqlite3
import tempfile
from collections import Counter
from datetime import datetime

import pytest
from hypothesis import HealthCheck, Phase, event, note, settings
from hypothesis import strategies as st
from hypothesis.stateful import (RuleBasedStateMachine, invariant, precondition, rule,
                                 run_state_machine_as_test)

from harness import model as reference
from harness.clock import FAR_FUTURE, INSTANTS
from harness.drive import ModelDivergence, Pair, format_program
from harness.invariants import check_store_integrity
from harness.known_bugs import KNOWN_BUGS
from harness.model import (POOLS, USERS, Delete, Erase, EraseExpired, Expect, Forget, Lapse,
                           Op, Remember)

VALID_FROM = st.sampled_from([None, None, *INSTANTS, FAR_FUTURE])
VALID_TO = st.sampled_from([None, None, None, INSTANTS[2], INSTANTS[4], FAR_FUTURE])
RECORDED_AT = st.sampled_from([None, None, None, *INSTANTS])
CONFIDENCE = st.sampled_from([1.0, 1.0, 0.5, 0.4])
EXPIRES_AT = st.sampled_from([None, None, None, FAR_FUTURE])

#: The open bugs this machine steers around, by their ids in `known_bugs`. None today.
STEERED: tuple[str, ...] = ()
#: Steps and skips over every run of this process, for the check after a test's runs.
TOTALS: Counter[str] = Counter()


class MemoryMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.path = pathlib.Path(tempfile.mkdtemp(prefix="memvara-machine-")) / "store.db"
        self.pair = Pair(self.path)
        #: For each row that has closed a clock, the closing stamps last seen.
        self.closed: dict[str, tuple[datetime | None, datetime | None]] = {}

    def teardown(self) -> None:
        try:
            self.pair.close()
            problems = check_store_integrity(self.path)
        finally:
            shutil.rmtree(self.path.parent, ignore_errors=True)
        assert not problems, (
            "the store file is damaged when the run ends:\n" + "\n".join(problems))

    def avoid(self, bug: str) -> None:
        assert bug in STEERED
        TOTALS["avoided"] += 1
        TOTALS[f"avoided {bug}"] += 1
        event(f"avoided {bug}")

    # -- rules ----------------------------------------------------------------------------

    @rule(user=st.sampled_from(USERS), predicate=st.sampled_from(sorted(POOLS)),
          data=st.data(), polarity=st.sampled_from([1, 1, 1, -1]),
          valid_from=VALID_FROM, recorded_at=RECORDED_AT, confidence=CONFIDENCE,
          close=st.sampled_from(["ended", "ended", "retired"]), expires_at=EXPIRES_AT,
          valid_to=VALID_TO)
    def remember(self, user: str, predicate: str, data: st.DataObject, polarity: int,
                 valid_from: datetime | None, recorded_at: datetime | None,
                 confidence: float, close: str, expires_at: datetime | None,
                 valid_to: datetime | None) -> None:
        obj = data.draw(st.sampled_from(POOLS[predicate]))
        self._apply(Remember(user, predicate, obj, polarity, valid_from, recorded_at,
                             confidence, close, expires_at, valid_to))

    @rule(user=st.sampled_from(USERS), obj=st.sampled_from(POOLS["lives_in"]),
          valid_from=st.sampled_from([INSTANTS[1], INSTANTS[3]]))
    def remember_a_home(self, user: str, obj: str, valid_from: datetime) -> None:
        """Values of the one single-valued predicate, at two instants only, so that two
        often begin at the same moment and one collapses the other."""
        self._apply(Remember(user, "lives_in", obj, valid_from=valid_from))

    @rule(user=st.sampled_from(USERS), predicate=st.sampled_from(sorted(POOLS)))
    def forget(self, user: str, predicate: str) -> None:
        self._apply(Forget(user, predicate))

    @precondition(lambda self: bool(self.pair.model.real_ids))
    @rule(user=st.sampled_from(USERS), data=st.data(),
          close=st.sampled_from(["retired", "ended"]))
    def delete(self, user: str, data: st.DataObject, close: str) -> None:
        handle = data.draw(st.sampled_from(sorted(self.pair.model.real_ids)))
        row = self.pair.model.rows.get(handle)
        if close == "retired" and row is not None and row.invalidated_at is not None:
            return      # the machine retires a row at most once
        self._apply(Delete(user, handle, close))

    @precondition(lambda self: bool(self.pair.model.real_ids))
    @rule(user=st.sampled_from(USERS), data=st.data())
    def erase(self, user: str, data: st.DataObject) -> None:
        handle = data.draw(st.sampled_from(sorted(self.pair.model.real_ids)))
        self._apply(Erase(user, handle))

    @precondition(lambda self: any(r.expires_at == FAR_FUTURE
                                   for r in self.pair.model.rows.values()))
    @rule(data=st.data())
    def lapse(self, data: st.DataObject) -> None:
        handles = sorted(h for h, r in self.pair.model.rows.items()
                         if r.expires_at == FAR_FUTURE)
        self._apply(Lapse(data.draw(st.sampled_from(handles)), self.pair.lapsed_instant()))

    @rule()
    def erase_expired(self) -> None:
        self._apply(EraseExpired())

    # -- checks across steps ---------------------------------------------------------------

    def _apply(self, op: Op) -> None:
        TOTALS["steps"] += 1
        before = {h: (r.valid_to, r.invalidated_at) for h, r in self.pair.model.rows.items()}
        stored = self.pair._row_ids()
        try:
            self.pair.apply(op)
        except ModelDivergence:
            note(f"drive.replay({format_program(self.pair.ops)})")
            raise
        gone = stored - self.pair._row_ids()
        if gone:
            assert isinstance(op, (Erase, EraseExpired)), (
                f"{op!r} removed {sorted(self.pair.handle_of(i) for i in gone)} from the store")
        for handle in set(before) & set(self.pair.model.rows):
            row = self.pair.model.rows[handle]
            was_to, was_at = before[handle]
            closed_both = (was_to is None and row.valid_to is not None
                           and was_at is None and row.invalidated_at is not None)
            assert not closed_both, f"{op!r} closed both clocks of {handle}"

    @invariant()
    def a_closed_clock_never_moves_later(self) -> None:
        for handle, row in self.pair.model.rows.items():
            seen_to, seen_at = self.closed.get(handle, (None, None))
            if seen_at is not None:
                assert row.invalidated_at == seen_at, (
                    f"{handle}'s retirement moved from {seen_at} to {row.invalidated_at}")
            if seen_to is not None:
                assert row.valid_to is not None, f"{handle}'s ending at {seen_to} was cleared"
                assert row.valid_to <= seen_to, (
                    f"{handle}'s ending moved later, from {seen_to} to {row.valid_to}")
            if row.valid_to is not None or row.invalidated_at is not None:
                self.closed[handle] = (row.valid_to, row.invalidated_at)


#: A step checks every read, so the nightly profile's 3,000 runs of 100 steps would take
#: well over an hour. This machine stops at 300 runs; `docs/claude/testing.md` quotes how
#: long the nightly tier takes, measured on a laptop with nothing else running.
MAX_RUNS = 300
#: The largest share of steps the machine may skip for known bugs before a run fails:
#: more than this, and the runs are testing less than they appear to.
MAX_AVOIDED_SHARE = 0.10


def machine_settings(max_examples: int) -> settings:
    profile = settings()  # the tier's profile, which the suite's conftest loads
    return settings(profile, deadline=None,
                    max_examples=min(profile.max_examples, max_examples),
                    suppress_health_check=[HealthCheck.too_slow,
                                           HealthCheck.filter_too_much])


def test_random_operations_keep_the_store_and_the_model_in_step() -> None:
    TOTALS.clear()
    run_state_machine_as_test(MemoryMachine, settings=machine_settings(MAX_RUNS))
    skipped = ", ".join(f"{k.split()[1]} {v}" for k, v in sorted(TOTALS.items())
                        if k.startswith("avoided "))
    print(f"\nthe machine ran {TOTALS['steps']} steps; skipped: {skipped or 'none'}")
    assert TOTALS["avoided"] <= MAX_AVOIDED_SHARE * TOTALS["steps"], (
        f"{TOTALS['avoided']} of {TOTALS['steps']} steps were skipped for known bugs")


def test_the_bugs_the_machine_steers_around_are_still_open() -> None:
    """A fix removes its bug from the registry; this fails until the machine stops
    steering around it too."""
    assert set(STEERED) <= set(KNOWN_BUGS)


def test_a_failing_run_carries_a_program_replay_accepts(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A model that forgets nothing diverges at the first `forget`; the failure must
    carry the program, so the run can be replayed without Hypothesis."""
    monkeypatch.setattr(reference.ReferenceStore, "forget",
                        lambda self, op, t: Expect(returned=[]))
    # The first failure is enough here, and shrinking it through every read of every
    # step would take minutes.
    quick = settings(machine_settings(20), phases=[Phase.generate])
    with pytest.raises(ModelDivergence) as failed:
        run_state_machine_as_test(MemoryMachine, settings=quick)
    notes = "\n".join(getattr(failed.value, "__notes__", []))
    assert "drive.replay([" in notes and "Forget(" in notes, notes


def test_the_machine_checks_its_store_file_when_a_run_ends() -> None:
    machine = MemoryMachine()
    machine._apply(Remember("u1", "likes", "tea"))
    conn = sqlite3.connect(machine.path)
    conn.execute("DELETE FROM embeddings")
    conn.commit()
    conn.close()
    with pytest.raises(AssertionError, match="has no embedding row"):
        machine.teardown()
    assert not machine.path.parent.exists()
