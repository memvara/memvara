"""The four kill points too slow or too rare for every pull request: the vector file
growing, the embedder record being written, `encrypt_store` between its two renames, and
`add_document` between storing the chunks and extracting claims.

The recovery each must show is in the table in
`docs/superpowers/plans/2026-09-25-adversarial-crashes.md`.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from memvara import Memvara, NullLLM
from memvara.embed import HashingEmbedder

from harness.crash import acked_claims, after_crash, kill_at

USER = "u1"
SETUP = [["remember", {"predicate": "likes", "object": "green tea"}],
         ["remember", {"predicate": "likes", "object": "black coffee"}]]
KEY = bytes(range(32))



def killed(point: str, spec: dict[str, Any], home: pathlib.Path,
           env: dict[str, str] | None = None) -> Any:
    """`crash.kill_at` with this file's user, and a timeout long enough for 256 writes."""
    return kill_at({"user": USER, "hold": False, "point": point, **spec}, home, env=env,
                   timeout=120)


def test_a_kill_while_the_vector_file_grows_keeps_every_acknowledged_claim(
        tmp_path: pathlib.Path, home: pathlib.Path) -> None:
    """The file holds 256 vectors before it first grows, so the 257th write grows it."""
    db = tmp_path / "s.db"
    setup = [["remember", {"predicate": "visited", "object": f"q{i:03d}x"}]
             for i in range(256)]
    child = killed("vecs-growth", {"db": str(db), "setup": setup, "action": [
        "remember", {"predicate": "visited", "object": "q256x"}]}, home)
    mem = after_crash(db, USER, acked_claims(child, setup))
    try:
        assert "q256x" not in {c.object for c in mem.store.iter_claims(None, True)}
    finally:
        mem.close()


def test_a_kill_while_the_embedder_record_is_written_leaves_no_torn_record(
        tmp_path: pathlib.Path, home: pathlib.Path) -> None:
    """`write_fingerprint` writes a temporary file and renames it over the record, so the
    kill tears only the temporary file and the new store is left with no record at all.
    The next open writes it, and the record then catches an embedder change again."""
    db = tmp_path / "s.db"
    killed("fingerprint-write", {"db": str(db), "setup": [], "action": ["open", {}]}, home)
    record = pathlib.Path(str(db) + ".embedder.json")
    assert not record.exists(), f"the kill left a record: {record.read_text()!r}"
    [torn] = tmp_path.glob("s.db.embedder.json.*.tmp")
    with pytest.raises(json.JSONDecodeError):
        json.loads(torn.read_text())
    after_crash(db, USER, {}).close()
    assert json.loads(record.read_text())["embedder"] == "hashing:512:3-5"
    with pytest.warns(Warning, match="unrelated vector spaces"):
        Memvara(str(db), embedder=HashingEmbedder(dim=512, ngram=(2, 4)),
                llm=NullLLM()).close()


def test_a_kill_between_encrypt_stores_two_renames_leaves_a_store_that_opens(
        tmp_path: pathlib.Path, home: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The database is already the encrypted copy and the vector file is still the old
    one. The next open with the key rebuilds the vectors, every claim is found, and a
    second `encrypt_store` reports the store as encrypted despite the files the kill
    left behind."""
    pytest.importorskip("sqlcipher3", reason="needs the encrypt extra (sqlcipher3)")
    from memvara.store.encryption import encrypt_store

    db = tmp_path / "s.db"
    env = {"MEMVARA_DB_KEY": KEY.hex()}
    child = killed("encrypt-between-renames", {"db": str(db), "setup": SETUP,
                                                "action": ["encrypt", {}]}, home, env)
    after_crash(db, USER, acked_claims(child, SETUP), key=KEY).close()
    monkeypatch.setenv("MEMVARA_DB_KEY", KEY.hex())
    assert encrypt_store(str(db)).already is True


def test_a_kill_before_a_documents_claims_leaves_it_visibly_unfinished(
        tmp_path: pathlib.Path, home: pathlib.Path) -> None:
    """The document and its chunks were committed before the kill; its claims and chunk
    vectors were not. Whatever the next open does, a reader must be able to tell that
    the document is unfinished."""
    db = tmp_path / "s.db"
    text = "\n\n".join(f"Section {i}. The team moved the build to Rust in year {2020 + i}."
                       for i in range(8))
    child = killed("document-before-claims", {
        "db": str(db), "setup": SETUP,
        "action": ["add_document", {"content": text, "title": "notes"}]}, home)
    mem = after_crash(db, USER, acked_claims(child, SETUP))
    try:
        [doc] = mem.list_documents(user=USER).items
        assert doc.status not in ("done", "stored"), (
            f"the document reads as {doc.status!r} although its claims were never "
            f"extracted")
        assert doc.chunks > 0
        # Nothing was extracted, so the only claims are the two setup writes and the
        # one `after_crash` made.
        predicates = {c.predicate for c in mem.store.iter_claims(None, True)}
        assert predicates == {"likes"}, f"a claim came from the document: {predicates}"
    finally:
        mem.close()
