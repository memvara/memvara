"""Two writers on one store, put in a fixed order by holding one of them inside memvara.

The first writer is a held child process. It stops before its first claim write: a
`remember()` after it has looked up the slot, a `delete()` after it has read the claim it
retires. There it holds the database's write lock. The test then starts a second write
through its own handle, on a thread, and lets the child carry on. The second write cannot
finish until the child's transaction ends, so the two writes always happen in the same
order and each test gives the same answer on every run. Each test checks that the second
write waited, and then checks what the store holds.

The child writes only names the store already knows. The first time a process meets a
name, `remember()` writes an entity row before it looks up the slot, and that write takes
the lock early whatever else the write does. With the names known in advance, the slot
lookup is the first thing the child's write does, which is the case these tests are for.
"""

from __future__ import annotations

import json
import pathlib
import threading
from collections.abc import Callable
from typing import Any

import pytest

from harness import stores
from harness.clock import INSTANTS
from harness.crash import Child
from harness.invariants import check_store_integrity

USER = "u1"
#: How long the test gives the second write while the child holds the lock. The write
#: cannot finish in that time, so this only has to be long enough for a write that did not
#: wait to have finished. It stays well below SQLite's five-second busy timeout, which the
#: waiting write must not reach.
HELD_SECONDS = 0.5


class Background:
    """One call, run on a thread of the test, keeping its result or its exception."""

    def __init__(self, call: Callable[[], Any]) -> None:
        self._call = call
        self._result: Any = None
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def _run(self) -> None:
        try:
            self._result = self._call()
        except BaseException as exc:  # noqa: BLE001 - raised again by `outcome`
            self._error = exc

    def running_after(self, seconds: float) -> bool:
        """Whether the call is still running `seconds` from now."""
        self._thread.join(timeout=seconds)
        return self._thread.is_alive()

    def outcome(self) -> Any:
        """The call's return value, once it has finished. Its exception, if it raised."""
        self._thread.join(timeout=30)
        assert not self._thread.is_alive(), "the second write never finished"
        if self._error is not None:
            raise self._error
        return self._result


def held(db: pathlib.Path, action: list[Any]) -> dict[str, Any]:
    """A program for a child that stops before its first claim write and waits there."""
    return {"db": str(db), "user": USER, "setup": [], "point": "before-claim",
            "hold": True, "action": action}


def lives_in(obj: str, valid_from: str | None = None) -> list[Any]:
    arguments: dict[str, Any] = {"predicate": "lives_in", "object": obj}
    if valid_from is not None:
        arguments["valid_from"] = valid_from
    return ["remember", arguments]


def delete(claim_id: str) -> list[Any]:
    return ["delete", {"id": claim_id}]


@pytest.mark.covers("inv:MM9")
def test_a_second_writer_waits_for_the_first_and_then_ends_its_value(
        tmp_path: pathlib.Path, home: pathlib.Path) -> None:
    """The child looks up the `lives_in` slot, finds it empty, and stops before it writes
    Berlin. The test then writes Paris, a later value, through its own handle. That write
    must wait for the child's transaction, then find Berlin and end it where Paris begins.
    If both writes had found the slot empty, both values would stay live."""
    db = tmp_path / "s.db"
    with stores.file(db) as mem:
        for place in ("Berlin", "Paris"):
            mem.remember("user", "likes", place, user=USER)
        with Child(held(db, lives_in("Berlin", INSTANTS[1].isoformat())),
                   home=home) as child:
            child.wait_for("POINT before-claim")
            second = Background(lambda: mem.remember(
                "user", "lives_in", "Paris", user=USER, valid_from=INSTANTS[3]))
            waited = second.running_after(HELD_SECONDS)
            child.release()
            child.wait_for("DONE")
            assert child.finish() == 0
            paris = second.outcome().added[0]
        rows = {c.object: c for c in mem.store.iter_claims(None, True)
                if c.predicate == "lives_in"}
        live = sorted(c.object for c in mem.get_all(user=USER) if c.predicate == "lives_in")
    assert live == ["Paris"], f"the live lives_in values are {live}"
    assert rows["Berlin"].valid_to == paris.valid_from
    assert waited, "the second write finished while the first writer held the slot"


def test_a_second_writer_of_the_same_value_reinforces_the_first(
        tmp_path: pathlib.Path, home: pathlib.Path) -> None:
    """Both writers state Berlin. The second must wait for the first, find its claim, and
    reinforce it, so the store holds one claim observed twice rather than two copies of
    the same value."""
    db = tmp_path / "s.db"
    with stores.file(db) as mem:
        mem.remember("user", "likes", "Berlin", user=USER)
        with Child(held(db, lives_in("Berlin")), home=home) as child:
            child.wait_for("POINT before-claim")
            second = Background(lambda: mem.remember("user", "lives_in", "Berlin",
                                                     user=USER))
            waited = second.running_after(HELD_SECONDS)
            child.release()
            first = json.loads(child.wait_for("DONE"))["ids"]
            assert child.finish() == 0
            receipt = second.outcome()
        rows = [c for c in mem.store.iter_claims(None, True) if c.predicate == "lives_in"]
    assert [c.id for c in rows] == first, (
        f"lives_in holds {len(rows)} claims where it should hold only the first writer's")
    assert not receipt.added and [c.id for c in receipt.reinforced] == first
    assert rows[0].observation_count == 2
    assert waited, "the second write finished while the first writer held the slot"


def test_a_new_value_waits_for_a_held_delete_and_every_ending_it_reports_is_kept(
        tmp_path: pathlib.Path, home: pathlib.Path) -> None:
    """The child's `delete` reads Berlin and stops before it writes Berlin back retired.
    The test then moves the user to Paris. That write must wait for the delete, and every
    ending its receipt reports must still be in the store afterwards. A delete that wrote
    back the copy it read before Paris arrived would undo the ending Paris reported."""
    db = tmp_path / "s.db"
    with stores.file(db) as mem:
        mem.remember("user", "likes", "Paris", user=USER)
        berlin = mem.remember("user", "lives_in", "Berlin", user=USER,
                              valid_from=INSTANTS[0]).added[0].id
        with Child(held(db, delete(berlin)), home=home) as child:
            child.wait_for("POINT before-claim")
            second = Background(lambda: mem.remember(
                "user", "lives_in", "Paris", user=USER, valid_from=INSTANTS[2]))
            waited = second.running_after(HELD_SECONDS)
            child.release()
            assert json.loads(child.wait_for("DONE"))["ids"] == ["yes"]
            assert child.finish() == 0
            receipt = second.outcome()
        stored = {c.id: mem.store.get_claim(c.id) for c in receipt.closed}
        kept = mem.store.get_claim(berlin)
    for closed in receipt.closed:
        assert stored[closed.id].valid_to == closed.valid_to, (
            f"Paris ended {closed.object} at {closed.valid_to}, and the store holds an "
            f"end of {stored[closed.id].valid_to}")
    assert kept.invalidated_at is not None, "the delete did not retire Berlin"
    assert waited, "Paris was written while the delete was held"


def test_a_delete_keeps_the_ending_a_concurrent_supersession_gave_the_claim(
        tmp_path: pathlib.Path, home: pathlib.Path) -> None:
    """The child moves the user to Paris: it finds Berlin in the slot and stops before it
    writes Paris and ends Berlin. The test then deletes Berlin. The delete must wait for
    the child, read Berlin as the child left it, ended where Paris begins, and retire it
    without undoing that ending."""
    db = tmp_path / "s.db"
    with stores.file(db) as mem:
        mem.remember("user", "likes", "Paris", user=USER)
        berlin = mem.remember("user", "lives_in", "Berlin", user=USER,
                              valid_from=INSTANTS[0]).added[0].id
        with Child(held(db, lives_in("Paris", INSTANTS[2].isoformat())),
                   home=home) as child:
            child.wait_for("POINT before-claim")
            deleting = Background(lambda: mem.delete(berlin, user=USER))
            waited = deleting.running_after(HELD_SECONDS)
            child.release()
            (paris,) = json.loads(child.wait_for("DONE"))["ids"]
            assert child.finish() == 0
            assert deleting.outcome() is True
        kept = mem.store.get_claim(berlin)
    assert (kept.valid_to, kept.invalidated_by) == (INSTANTS[2], paris), (
        f"Berlin's ending was lost: it ends at {kept.valid_to}, and "
        f"{kept.invalidated_by} replaced it")
    assert kept.invalidated_at is not None, "the delete's retirement was lost"
    assert waited, "the delete wrote while the supersession was held"


def test_a_claim_erased_while_a_delete_was_held_stays_erased(
        tmp_path: pathlib.Path, home: pathlib.Path) -> None:
    """The child's `delete` reads a claim and stops before it writes the claim back
    retired. The test then erases the claim. The erasure must wait for the delete and then
    remove the retired claim, and nothing may write the claim back afterwards: an erasure
    that reported success must leave no row, no text index entry and no vector."""
    db = tmp_path / "s.db"
    with stores.file(db) as mem:
        tea = mem.remember("user", "likes", "tea", user=USER).added[0].id
        with Child(held(db, delete(tea)), home=home) as child:
            child.wait_for("POINT before-claim")
            erasing = Background(lambda: mem.erase(tea, user=USER))
            waited = erasing.running_after(HELD_SECONDS)
            child.release()
            assert json.loads(child.wait_for("DONE"))["ids"] == ["yes"]
            assert child.finish() == 0
            erased = erasing.outcome()
        back = mem.store.get_claim(tea)
        record = mem.store.erasure_record(tea)
    assert back is None, "the erased claim is back in the store"
    assert erased is True and record is not None
    assert check_store_integrity(db) == []
    assert waited, "the erasure ran while the delete was held"
