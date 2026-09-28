"""A child is killed at each of the six fast points, and the store it leaves behind must
recover as the spec defines: intact, holding every acknowledged write, with the
interrupted operation either whole or absent, and ready for the next write.

The nightly tier kills at the other four points (`nightly/test_adv_kill_points_nightly.py`).
"""

from __future__ import annotations

import json
import pathlib
import sqlite3
from typing import Any

import pytest

from harness import stores
from harness.crash import Child, acked_claims, after_crash, kill_at
from harness.invariants import check_store_integrity

USER = "u1"
#: Two writes every test acknowledges before the one it interrupts.
SETUP = [["remember", {"predicate": "likes", "object": "green tea"}],
         ["remember", {"predicate": "likes", "object": "black coffee"}]]


def program(db: pathlib.Path, **fields: Any) -> dict[str, Any]:
    return {"db": str(db), "user": USER, "setup": SETUP, "point": None, "action": None,
            "hold": False, **fields}



def acked_live(child: Child) -> dict[str, str]:
    """The setup writes the child acknowledged, by id, with the text that finds each."""
    return acked_claims(child, SETUP)


def rows_per_table(raw: sqlite3.Connection) -> dict[str, int]:
    """The number of rows in every table of the file. A full-text table is counted by
    its content, and its shadow tables, which FTS5 fills with its own bookkeeping as
    soon as the table is created, are left out."""
    tables = list(raw.execute("SELECT name, sql FROM sqlite_master WHERE type = 'table'"))
    virtual = [n for n, sql in tables if (sql or "").startswith("CREATE VIRTUAL TABLE")]
    names = [n for n, _ in tables
             if not any(n.startswith(f"{v}_") for v in virtual)]
    return {n: raw.execute(f'SELECT count(*) FROM "{n}"').fetchone()[0] for n in names}


def objects(mem: Any, predicate: str) -> list[str]:
    return sorted(c.object for c in mem.store.iter_claims(None, True)
                  if c.predicate == predicate)


def test_a_kill_after_the_episode_leaves_no_episode(
        tmp_path: pathlib.Path, home: pathlib.Path) -> None:
    db = tmp_path / "s.db"
    child = kill_at(program(db, point="after-episode",
                            action=["add", {"text": "I live in Berlin"}]), home)
    mem = after_crash(db, USER, acked_live(child))
    try:
        assert objects(mem, "lives_in") == []
        raw = sqlite3.connect(db)
        try:
            episodes = raw.execute("SELECT count(*) FROM episodes").fetchone()[0]
            indexed = raw.execute("SELECT count(*) FROM episodes_fts").fetchone()[0]
        finally:
            raw.close()
        assert (episodes, indexed) == (0, 0)
    finally:
        mem.close()


@pytest.mark.parametrize("point", ["after-claim", "after-vector"])
def test_a_kill_inside_remember_leaves_no_part_of_the_claim(
        point: str, tmp_path: pathlib.Path, home: pathlib.Path) -> None:
    db = tmp_path / "s.db"
    child = kill_at(program(db, point=point, action=[
        "remember", {"predicate": "lives_in", "object": "Berlin"}]), home)
    mem = after_crash(db, USER, acked_live(child))
    try:
        assert objects(mem, "lives_in") == []
    finally:
        mem.close()


def test_a_kill_inside_erase_leaves_the_claim_whole_and_unrecorded(
        tmp_path: pathlib.Path, home: pathlib.Path) -> None:
    db = tmp_path / "s.db"
    child = kill_at(program(db, point="erase-before-delete",
                            action=["erase", {"id": {"ref": 1}}]), home)
    # Both setup claims must still be live: the erase of the second was never committed,
    # so `after_crash` also checks the second has no erasure record.
    after_crash(db, USER, acked_live(child)).close()


def test_a_kill_after_the_commit_keeps_the_claim(
        tmp_path: pathlib.Path, home: pathlib.Path) -> None:
    db = tmp_path / "s.db"
    spec = program(db, point="after-commit",
                   action=["remember", {"predicate": "lives_in", "object": "Berlin"}])
    with Child(spec, home=home) as child:
        done = child.wait_for("DONE")
        child.wait_for("POINT after-commit")
        child.kill()
    berlin = json.loads(done)["ids"][0]
    after_crash(db, USER, {**acked_live(child), berlin: "Berlin"}).close()


def test_a_kill_inside_a_batch_loses_the_whole_batch_and_nothing_before_it(
        tmp_path: pathlib.Path, home: pathlib.Path) -> None:
    db = tmp_path / "s.db"
    batch = [["remember", {"predicate": "collects", "object": name}]
             for name in ("stamps", "vinyl", "coins")]
    child = kill_at(program(db, point="inside-batch", action=["batch", {"ops": batch}]), home)
    mem = after_crash(db, USER, acked_live(child))
    try:
        assert objects(mem, "collects") == []
    finally:
        mem.close()


def test_a_kill_between_two_migrations_leaves_the_old_version(
        tmp_path: pathlib.Path, home: pathlib.Path) -> None:
    db = tmp_path / "s.db"
    with stores.file(db) as mem:
        episode = mem.add("the kafka pipeline is being sunset", user=USER).episode_ids[0]
    # Cut the store back to version 2, the way tests/test_store.py's write_v2 does.
    raw = sqlite3.connect(db)
    raw.execute("DROP TABLE episodes_fts")
    raw.execute("DROP TABLE episode_embeddings")
    raw.execute("PRAGMA user_version = 2")
    raw.commit()
    before = rows_per_table(raw)
    raw.close()
    spec = {"db": str(db), "user": USER, "setup": [], "point": "between-migrations",
            "action": ["open", {}], "hold": False}
    with Child(spec, home=home) as child:
        child.wait_for("POINT between-migrations")
        child.kill()
    raw = sqlite3.connect(db)
    try:
        assert raw.execute("PRAGMA user_version").fetchone()[0] == 2
        # All or nothing for the data. Every open runs the schema's `CREATE TABLE IF NOT
        # EXISTS` statements before it migrates, and those commit at once, so tables the
        # upgrade adds may exist; they must be empty.
        after = rows_per_table(raw)
        assert {t: n for t, n in after.items() if t in before} == before
        assert {t: n for t, n in after.items() if t not in before and n} == {}
    finally:
        raw.close()
    with stores.file(db) as mem:
        assert mem.store.get_episode(episode) is not None
    raw = sqlite3.connect(db)
    try:
        found = raw.execute("SELECT count(*) FROM episodes_fts WHERE episodes_fts MATCH "
                            "'kafka'").fetchone()[0]
    finally:
        raw.close()
    assert found == 1, "the finished upgrade did not index the version-2 episode"
    assert check_store_integrity(db) == []


def test_a_kill_while_a_new_store_is_created_leaves_a_file_the_next_open_finishes(
        tmp_path: pathlib.Path, home: pathlib.Path) -> None:
    """A new store runs every migration on its first open, so the same point interrupts
    its creation. The half-made file must not stop the next open."""
    db = tmp_path / "s.db"
    spec = {"db": str(db), "user": USER, "setup": [], "point": "between-migrations",
            "action": ["open", {}], "hold": False}
    with Child(spec, home=home) as child:
        child.wait_for("POINT between-migrations")
        child.kill()
    after_crash(db, USER, {}).close()


def test_after_a_kill_inside_remember_new_writes_find_only_their_own_claims(
        tmp_path: pathlib.Path, home: pathlib.Path) -> None:
    """A kill after the vector is written can leave vector bytes in a slot no row names.
    Whichever write takes that slot next must overwrite them, and no search may return
    a claim that does not exist."""
    db = tmp_path / "s.db"
    child = kill_at(program(db, point="after-vector", action=[
        "remember", {"predicate": "lives_in", "object": "Berlin"}]), home)
    mem = after_crash(db, USER, acked_live(child))
    try:
        cities = ["Paris", "Rome", "Lisbon", "Oslo", "Vienna", "Prague", "Dublin",
                  "Madrid", "Athens", "Warsaw", "Riga", "Tallinn", "Vilnius", "Sofia",
                  "Zagreb", "Bern", "Brussels", "Amsterdam", "Helsinki", "Copenhagen"]
        ids = {}
        for city in cities:
            ids[city] = mem.remember("user", "visited", city, user=USER).added[0].id
        existing = {c.id for c in mem.store.iter_claims(None, True)}
        for city, claim_id in ids.items():
            hits = [r.claim.id for r in mem.search(city, k=5, user=USER)]
            assert hits and hits[0] == claim_id, f"search({city!r}) returned {hits}"
            assert set(hits) <= existing, f"search({city!r}) returned a missing claim"
    finally:
        mem.close()
