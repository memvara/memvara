"""An open that is going to be refused refuses before it upgrades an older store file.

`SQLiteStore` upgrades a file an older version wrote as it opens it, and commits the
upgrade. `Memvara` then checks that its embedder can read the stored vectors. When that
check refused the open, the file had already been upgraded, and the release that wrote it
could no longer open it (#300). `SQLiteStore(before_upgrade=...)` is called with the
width of the stored vectors before anything is written, and `Memvara` refuses from there.
"""

from __future__ import annotations

import pathlib
import sqlite3

import pytest

from memvara import EmbedderMismatchError, Memvara, NullLLM
from memvara.embed import HashingEmbedder
from memvara.store.sqlite import SCHEMA_VERSION, SQLiteStore


def older_store(tmp_path: pathlib.Path, dim: int = 64) -> pathlib.Path:
    """A store holding one claim's vector of width `dim`, stamped one version older."""
    db = tmp_path / "old.db"
    with Memvara(str(db), embedder=HashingEmbedder(dim=dim), llm=NullLLM()) as mem:
        mem.remember("user", "lives_in", "Lisbon")
    stamp(db, SCHEMA_VERSION - 1)
    return db


def stamp(db: pathlib.Path, version: int) -> None:
    conn = sqlite3.connect(db)
    conn.execute(f"PRAGMA user_version = {version}")
    conn.commit()
    conn.close()


def version_of(db: pathlib.Path) -> int:
    conn = sqlite3.connect(db)
    try:
        return int(conn.execute("PRAGMA user_version").fetchone()[0])
    finally:
        conn.close()


class Refused(Exception):
    pass


def test_the_hook_sees_the_stored_width_before_the_upgrade_writes(tmp_path):
    db = older_store(tmp_path, dim=64)
    seen: list[int | None] = []

    def hook(width: int | None) -> None:
        seen.append(width)
        raise Refused

    with pytest.raises(Refused):
        SQLiteStore(str(db), before_upgrade=hook)
    assert seen == [64]
    assert version_of(db) == SCHEMA_VERSION - 1


def test_the_hook_is_not_called_for_a_current_or_a_new_file(tmp_path):
    db = older_store(tmp_path)
    SQLiteStore(str(db)).close()  # upgrades it
    calls: list[int | None] = []
    SQLiteStore(str(db), before_upgrade=calls.append).close()
    SQLiteStore(str(tmp_path / "new.db"), before_upgrade=calls.append).close()
    assert calls == []


def test_an_older_store_with_no_vectors_passes_no_width(tmp_path):
    db = tmp_path / "empty.db"
    SQLiteStore(str(db)).close()
    stamp(db, SCHEMA_VERSION - 1)
    seen: list[int | None] = []
    SQLiteStore(str(db), before_upgrade=seen.append).close()
    assert seen == [None]
    assert version_of(db) == SCHEMA_VERSION


def test_a_given_embedder_of_another_width_is_refused_before_the_upgrade(tmp_path):
    db = older_store(tmp_path, dim=64)
    with pytest.raises(EmbedderMismatchError, match="64-dimensional vectors"):
        Memvara(str(db), embedder=HashingEmbedder(dim=128), llm=NullLLM())
    assert version_of(db) == SCHEMA_VERSION - 1


def test_the_default_embedder_of_another_width_is_refused_before_the_upgrade(
        tmp_path, monkeypatch):
    db = older_store(tmp_path, dim=64)
    monkeypatch.setattr("memvara.core.default_embedder",
                        lambda model=None: HashingEmbedder(dim=128))
    with pytest.raises(EmbedderMismatchError, match="64-dimensional vectors"):
        Memvara(str(db), llm=NullLLM())
    assert version_of(db) == SCHEMA_VERSION - 1


def test_the_default_embedder_chosen_before_the_upgrade_is_the_one_used(
        tmp_path, monkeypatch):
    db = older_store(tmp_path, dim=64)
    made: list[HashingEmbedder] = []

    def default(model=None):
        made.append(HashingEmbedder(dim=64))
        return made[-1]

    monkeypatch.setattr("memvara.core.default_embedder", default)
    with Memvara(str(db), llm=NullLLM()) as mem:
        assert mem.embedder is made[0] and len(made) == 1
    assert version_of(db) == SCHEMA_VERSION


def test_reembed_still_upgrades_and_migrates_an_older_store(tmp_path):
    db = older_store(tmp_path, dim=64)
    with Memvara(str(db), embedder=HashingEmbedder(dim=128), llm=NullLLM(),
                 reembed=True) as mem:
        assert [r.claim.object for r in mem.search("where do I live", k=1)] == ["Lisbon"]
    assert version_of(db) == SCHEMA_VERSION
