"""import_mem0 dates each event of mem0's history at the time mem0 recorded it (#365).

mem0 writes an UPDATE or DELETE row with its memory's creation time in `created_at` and
the time of the event itself in `updated_at` (`Memory._update_memory` and
`_delete_memory` in mem0/memory/main.py). Each row's id is a random uuid4. The nightly
framework tests write such a file with mem0's own `SQLiteManager`; these tests write the
same rows by hand, so they run in the fast tier without mem0 installed.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from memvara import Memvara
from memvara.compat import import_mem0, read_history_db
from memvara.embed import HashingEmbedder
from memvara.llm import NullLLM

MARCH_1 = datetime(2024, 3, 1, tzinfo=timezone.utc)
MARCH_2 = datetime(2024, 3, 2, tzinfo=timezone.utc)
JUNE = datetime(2024, 6, 1, tzinfo=timezone.utc)
JULY = datetime(2024, 7, 1, tzinfo=timezone.utc)

COLUMNS = ("id", "memory_id", "old_memory", "new_memory", "event", "created_at",
           "updated_at", "is_deleted", "actor_id", "role")


@pytest.fixture()
def mem():
    m = Memvara(embedder=HashingEmbedder(dim=128), llm=NullLLM(), user="alice")
    yield m
    m.close()


def _history(tmp_path, *, add_first: bool) -> str:
    """Memory m1 is added on March 1 and updated on June 1, and memory m2 is added on
    March 2 and deleted on July 1, written the way mem0 writes them. The ids make each
    ADD row sort before, or after, the row that changes it, as mem0's random ids can."""
    add, change = ("0", "f") if add_first else ("f", "0")
    rows = [
        (add + "1", "m1", None, "Alice lives in Berlin", "ADD", MARCH_1, MARCH_1, 0),
        (change + "1", "m1", "Alice lives in Berlin", "Alice lives in Lisbon", "UPDATE",
         MARCH_1, JUNE, 0),
        (add + "2", "m2", None, "Alice likes tea", "ADD", MARCH_2, MARCH_2, 0),
        (change + "2", "m2", "Alice likes tea", None, "DELETE", MARCH_2, JULY, 1),
    ]
    path = tmp_path / "history.db"
    con = sqlite3.connect(path)
    with con:
        con.execute(f"CREATE TABLE history ({', '.join(c + ' TEXT' for c in COLUMNS)})")
        con.executemany(f"INSERT INTO history VALUES ({','.join('?' * len(COLUMNS))})",
                        [(i, m, old, new, event, created.isoformat(), updated.isoformat(),
                          deleted, "alice", "user")
                         for i, m, old, new, event, created, updated, deleted in rows])
    con.close()
    return str(path)


@pytest.mark.parametrize("add_first", [True, False], ids=["add rows first", "add rows last"])
def test_an_update_and_a_delete_are_dated_when_mem0_recorded_them(mem, tmp_path, add_first):
    """The value an UPDATE replaces ends on the day of the update, and a deleted memory
    stops being believed on the day of the delete, not on the day each was created."""
    import_mem0(mem, history_db=_history(tmp_path, add_first=add_first))
    claims = {c.object: c for c in mem.get_all(states=("live", "ended", "retired"))}
    assert claims["Alice lives in Berlin"].valid_to == JUNE
    assert claims["Alice likes tea"].invalidated_at == JULY


@pytest.mark.parametrize("add_first", [True, False], ids=["add rows first", "add rows last"])
def test_each_event_is_replayed_after_the_add_it_changes(mem, tmp_path, add_first):
    """Only the updated value is live afterwards, whatever order the random ids give."""
    import_mem0(mem, history_db=_history(tmp_path, add_first=add_first))
    assert sorted(c.object for c in mem.get_all()) == ["Alice lives in Lisbon"]


def test_the_log_is_read_in_the_order_its_events_happened(tmp_path):
    rows = read_history_db(_history(tmp_path, add_first=False))
    assert [(r.memory_id, r.event) for r in rows] == [
        ("m1", "ADD"), ("m2", "ADD"), ("m1", "UPDATE"), ("m2", "DELETE")]
    assert [r.at for r in rows] == [MARCH_1, MARCH_2, JUNE, JULY]
    assert rows[2].updated_at == JUNE


def test_an_add_sorts_before_an_event_recorded_at_the_same_instant(tmp_path):
    """A log whose update was recorded in the same second as its add must still replay
    the add first, even when the update's id sorts first."""
    path = tmp_path / "history.db"
    con = sqlite3.connect(path)
    with con:
        con.execute(f"CREATE TABLE history ({', '.join(c + ' TEXT' for c in COLUMNS)})")
        con.executemany(f"INSERT INTO history VALUES ({','.join('?' * len(COLUMNS))})", [
            ("f", "m1", None, "Alice lives in Berlin", "ADD", MARCH_1.isoformat(), None,
             0, None, None),
            ("0", "m1", "Alice lives in Berlin", "Alice lives in Lisbon", "UPDATE",
             MARCH_1.isoformat(), MARCH_1.isoformat(), 0, None, None)])
    con.close()
    assert [r.event for r in read_history_db(str(path))] == ["ADD", "UPDATE"]


def test_a_row_with_no_updated_at_is_dated_by_created_at(tmp_path):
    """Older mem0 schemas have no updated_at column, and a row may leave it empty."""
    path = tmp_path / "history.db"
    columns = ("id", "memory_id", "new_memory", "event", "created_at")
    con = sqlite3.connect(path)
    with con:
        con.execute(f"CREATE TABLE history ({', '.join(c + ' TEXT' for c in columns)})")
        con.execute("INSERT INTO history VALUES (?,?,?,?,?)",
                    ("h1", "m1", "Alice lives in Berlin", "UPDATE", JUNE.isoformat()))
    con.close()
    [row] = read_history_db(str(path))
    assert (row.updated_at, row.at) == (None, JUNE)


def test_an_updated_at_of_zero_is_a_time_not_a_missing_value(tmp_path):
    """`_parse_ts` reads a number as seconds since the epoch, so 0 is 1970-01-01. A
    column declared without a type hands the number back as an int, and a check for a
    missing value that tested truthiness dropped it and dated the row by created_at."""
    path = tmp_path / "history.db"
    con = sqlite3.connect(path)
    with con:
        con.execute("CREATE TABLE history (id, memory_id, new_memory, event, created_at, "
                    "updated_at)")
        con.execute("INSERT INTO history VALUES (?,?,?,?,?,?)",
                    ("h1", "m1", "Alice lives in Berlin", "UPDATE", JUNE.isoformat(), 0))
    con.close()
    [row] = read_history_db(str(path))
    assert row.at == datetime(1970, 1, 1, tzinfo=timezone.utc)
