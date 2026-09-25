"""The four kill points too slow or too rare for every pull request: the vector file
growing, the embedder record being written, `encrypt_store` between its two renames, and
`add_document` between storing the chunks and extracting claims.

The recovery each must show is in the table in
`docs/superpowers/plans/2026-09-25-adversarial-crashes.md`.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any

import pytest

from memvara import Memvara, NullLLM
from memvara.embed import HashingEmbedder

from harness.crash import Child, after_crash

USER = "u1"
SETUP = [["remember", {"predicate": "likes", "object": "green tea"}],
         ["remember", {"predicate": "likes", "object": "black coffee"}]]
KEY = bytes(range(32))


@pytest.fixture()
def home(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "home"
    path.mkdir()
    return path


def kill_at(point: str, spec: dict[str, Any], home: pathlib.Path,
            env: dict[str, str] | None = None) -> Child:
    with Child({"user": USER, "hold": False, "point": point, **spec}, home=home, env=env,
               timeout=120) as child:
        child.wait_for(f"POINT {point}")
        child.kill()
    return child


def acked(child: Child, setup: list[list[Any]]) -> dict[str, str]:
    assert len(child.acked) == len(setup), f"the child acknowledged {len(child.acked)}"
    return {a["ids"][0]: op[1]["object"] for a, op in zip(child.acked, setup)}


def test_a_kill_while_the_vector_file_grows_keeps_every_acknowledged_claim(
        tmp_path: pathlib.Path, home: pathlib.Path) -> None:
    """The file holds 256 vectors before it first grows, so the 257th write grows it."""
    db = tmp_path / "s.db"
    setup = [["remember", {"predicate": "visited", "object": f"q{i:03d}x"}]
             for i in range(256)]
    child = kill_at("vecs-growth", {"db": str(db), "setup": setup, "action": [
        "remember", {"predicate": "visited", "object": "q256x"}]}, home)
    mem = after_crash(db, USER, acked(child, setup))
    try:
        assert "q256x" not in {c.object for c in mem.store.iter_claims(None, True)}
    finally:
        mem.close()


def test_a_kill_while_the_embedder_record_is_written_heals_on_the_next_open(
        tmp_path: pathlib.Path, home: pathlib.Path) -> None:
    """`write_fingerprint` is not atomic, so the kill leaves a torn record. A store with
    no vectors rewrites it on the next open, and the record then catches an embedder
    change again."""
    db = tmp_path / "s.db"
    kill_at("fingerprint-write", {"db": str(db), "setup": [], "action": ["open", {}]}, home)
    record = pathlib.Path(str(db) + ".embedder.json")
    with pytest.raises(json.JSONDecodeError):
        json.loads(record.read_text())
    after_crash(db, USER, {}).close()
    assert json.loads(record.read_text())["embedder"] == "hashing:512:3-5"
    with pytest.warns(Warning, match="unrelated vector spaces"):
        Memvara(str(db), embedder=HashingEmbedder(dim=512, ngram=(2, 4)),
                llm=NullLLM()).close()


def test_a_kill_between_encrypt_stores_two_renames_leaves_a_store_that_opens(
        tmp_path: pathlib.Path, home: pathlib.Path) -> None:
    """The database is already the encrypted copy and the vector file is still the old
    one. The next open with the key rebuilds the vectors, every claim is found, and a
    second `encrypt_store` reports the store as encrypted despite the files the kill
    left behind."""
    pytest.importorskip("sqlcipher3", reason="needs the encrypt extra (sqlcipher3)")
    from memvara.store.encryption import encrypt_store

    db = tmp_path / "s.db"
    env = {"MEMVARA_DB_KEY": KEY.hex()}
    child = kill_at("encrypt-between-renames", {"db": str(db), "setup": SETUP,
                                                "action": ["encrypt", {}]}, home, env)
    after_crash(db, USER, acked(child, SETUP), key=KEY).close()
    os.environ["MEMVARA_DB_KEY"] = KEY.hex()
    try:
        assert encrypt_store(str(db)).already is True
    finally:
        del os.environ["MEMVARA_DB_KEY"]


def test_a_kill_before_a_documents_claims_leaves_it_visibly_unfinished(
        tmp_path: pathlib.Path, home: pathlib.Path) -> None:
    """The document and its chunks were committed before the kill; its claims and chunk
    vectors were not. Whatever the next open does, a reader must be able to tell that
    the document is unfinished."""
    db = tmp_path / "s.db"
    text = "\n\n".join(f"Section {i}. The team moved the build to Rust in year {2020 + i}."
                       for i in range(8))
    child = kill_at("document-before-claims", {
        "db": str(db), "setup": SETUP,
        "action": ["add_document", {"content": text, "title": "notes"}]}, home)
    mem = after_crash(db, USER, acked(child, SETUP))
    try:
        [doc] = mem.list_documents(user=USER).items
        assert doc.status not in ("done", "stored"), (
            f"the document reads as {doc.status!r} although its claims were never "
            f"extracted")
        assert doc.chunks > 0
    finally:
        mem.close()
