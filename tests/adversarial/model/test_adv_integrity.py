"""`check_store_integrity` finds the damage a crash or a bad write can leave in a store."""

from __future__ import annotations

import pathlib
import sqlite3

import pytest

from harness import stores
from harness.invariants import check_store_integrity


@pytest.fixture()
def db(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "memory.db"
    with stores.file(path, user="u") as m:
        m.add("I live in Berlin.")
        m.remember("user", "likes", "tea")
        kept = m.remember("user", "likes", "coffee").added[0]
        m.erase(m.remember("user", "likes", "cake").added[0].id)
        assert kept.id
    return path


def _sql(path: pathlib.Path, statement: str, *args: object) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(statement, args)


def _a_claim(path: pathlib.Path) -> str:
    with sqlite3.connect(path) as conn:
        return str(conn.execute("SELECT id FROM claims LIMIT 1").fetchone()[0])


def test_a_healthy_store_has_no_problems(db: pathlib.Path) -> None:
    assert check_store_integrity(db) == []


def test_a_claim_missing_from_the_text_index_is_named(db: pathlib.Path) -> None:
    claim = _a_claim(db)
    _sql(db, "DELETE FROM claims_fts WHERE claim_id = ?", claim)
    assert check_store_integrity(db) == [f"claim {claim} has 0 text-index rows, not 1"]


def test_a_claim_with_no_vector_is_named(db: pathlib.Path) -> None:
    claim = _a_claim(db)
    _sql(db, "DELETE FROM embeddings WHERE claim_id = ?", claim)
    assert check_store_integrity(db) == [f"claim {claim} has no embedding row"]


def test_a_provenance_edge_that_outlived_its_claim_is_named(db: pathlib.Path) -> None:
    with sqlite3.connect(db) as conn:
        episode = str(conn.execute("SELECT id FROM episodes LIMIT 1").fetchone()[0])
    _sql(db, "INSERT INTO claim_sources (episode_id, claim_id) VALUES (?, 'cl_gone')",
         episode)
    assert check_store_integrity(db) == ["a source edge names claim cl_gone, which is gone"]


def test_an_erased_claim_that_still_exists_is_named(db: pathlib.Path) -> None:
    claim = _a_claim(db)
    _sql(db, "INSERT INTO erasures (claim_id, tenant, scope, erased_at, sources) "
             "VALUES (?, 'default', 'default/u/*/*/*', 0, 0)", claim)
    assert check_store_integrity(db) == [f"claim {claim} has an erasure record and still exists"]


def test_a_vector_slot_both_used_and_free_is_named(db: pathlib.Path) -> None:
    with sqlite3.connect(db) as conn:
        slot = conn.execute("SELECT slot FROM embeddings WHERE slot IS NOT NULL LIMIT 1"
                            ).fetchone()[0]
        conn.execute("INSERT INTO vec_free (slot) VALUES (?)", (slot,))
    assert check_store_integrity(db) == [f"vector slot {slot} is in use and on the free list"]


def test_a_source_edge_to_a_missing_episode_is_reported(db: pathlib.Path) -> None:
    _sql(db, "INSERT INTO claim_sources (episode_id, claim_id) VALUES (?, ?)",
         "ep_gone", _a_claim(db))
    assert any("names episode ep_gone, which is gone" in p
               for p in check_store_integrity(db))
