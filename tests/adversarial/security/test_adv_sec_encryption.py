"""Encryption at rest: what tampering is caught, and the limits that are documented.

An encrypted store is two encrypted files — a SQLCipher database and a vector sidecar
whose every row is sealed with AES-256-GCM, bound to its row number and its owner's id.
The guarantee under attack: a record swapped between rows, cut short, or written under
another key must not load silently, and a wrong key must not open the store. The limits
`docs/LIMITATIONS.md` records — the plaintext `<db>.embedder.json`, and the replay of an
older record for the same row and owner — are documented behaviour, asserted here as such
rather than reported as bugs. See `memvara/store/encryption.py`.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import stat

import numpy as np
import pytest

from harness import stores
from memvara.embed import HashingEmbedder
from memvara.store.encryption import (EncryptionError, EncryptionWarning, VectorSealer,
                                      key_file)

needs_extra = pytest.mark.skipif(
    importlib.util.find_spec("sqlcipher3") is None,
    reason="needs the encrypt extra: pip install 'memvara[encrypt]'")

KEY = "b" * 64
OTHER_KEY = "c" * 64
SENTINEL = "zzsentinelvalue"


def _encrypted(path: pathlib.Path, key: str = KEY):
    return stores.file(path, encryption=True, key_env={"MEMVARA_DB_KEY": key})


def _seed(path: pathlib.Path) -> None:
    with _encrypted(path) as mem:
        mem.remember("user", "lives_in", SENTINEL, user="alice")
        mem.remember("user", "works_at", "Acme", user="alice")


@needs_extra
def test_a_wrong_key_does_not_open_the_store_and_never_prints_the_key(
        tmp_path: pathlib.Path) -> None:
    """A store opened with the wrong key raises, and the message names where the key came
    from — never the key itself."""
    db = tmp_path / "e.db"
    _seed(db)
    with pytest.raises(EncryptionError) as caught:
        with _encrypted(db, OTHER_KEY):
            pass
    assert OTHER_KEY not in str(caught.value)


@needs_extra
def test_a_tampered_vector_record_is_detected_not_served(tmp_path: pathlib.Path) -> None:
    """Flipping a byte inside a record whose owner still exists fails authentication on
    reopen, so a search cannot be served a vector the key did not write."""
    db = tmp_path / "e.db"
    _seed(db)
    vecs = pathlib.Path(str(db) + ".vecs")
    raw = bytearray(vecs.read_bytes())
    raw[80] ^= 0xFF  # a byte inside the first record, past the 64-byte header
    vecs.write_bytes(raw)
    with pytest.raises(EncryptionError):
        with _encrypted(db) as mem:
            mem.search(SENTINEL, user="alice")


@needs_extra
def test_a_truncated_vector_file_is_detected_not_served(tmp_path: pathlib.Path) -> None:
    """A vector file cut short loses records past the cut; the store reports them missing
    rather than serving a wrong or empty vector silently."""
    db = tmp_path / "e.db"
    _seed(db)
    vecs = pathlib.Path(str(db) + ".vecs")
    raw = vecs.read_bytes()
    vecs.write_bytes(raw[:80])  # header plus a fragment of the first record
    with pytest.raises(EncryptionError):
        with _encrypted(db) as mem:
            mem.search(SENTINEL, user="alice")


@needs_extra
def test_neither_the_text_nor_its_vector_is_on_disk_in_the_clear(
        tmp_path: pathlib.Path) -> None:
    """The stored value's text is absent from both files, and its embedding is absent from
    the vector file — a plaintext vector would confirm a guess about the text it came
    from."""
    db = tmp_path / "e.db"
    _seed(db)
    db_bytes = db.read_bytes()
    vec_bytes = pathlib.Path(str(db) + ".vecs").read_bytes()
    assert SENTINEL.encode() not in db_bytes
    assert SENTINEL.encode() not in vec_bytes
    embedding = HashingEmbedder(dim=512).encode([f"user lives in {SENTINEL}"])[0]
    assert np.asarray(embedding, dtype=np.float32).tobytes() not in vec_bytes


@needs_extra
def test_the_embedder_record_is_plaintext_by_documented_design(
        tmp_path: pathlib.Path) -> None:
    """Documented in docs/LIMITATIONS.md: `<db>.embedder.json` is not encrypted. It names
    the embedding model and its width and nothing stored, so this is a limit, not a leak."""
    db = tmp_path / "e.db"
    _seed(db)
    record = json.loads(pathlib.Path(str(db) + ".embedder.json").read_text())
    assert record["dim"] == 512
    assert SENTINEL not in json.dumps(record)


def test_an_older_record_for_the_same_row_and_owner_still_authenticates() -> None:
    """Documented in docs/LIMITATIONS.md and the VectorSealer docstring: the seal binds a
    record to its row number and owner, not to its freshness, so an older record for the
    same row and owner authenticates — the one tampering case that passes. It still cannot
    make a search use another row's or owner's vector, which the last two checks pin.

    Uses VectorSealer directly, so no SQLCipher install is needed for this documented
    limit.
    """
    sealer = VectorSealer(bytes.fromhex(KEY))
    sealer.salt = b"\x00" * VectorSealer.SALT
    sealer.bind(4)
    older = sealer.seal(0, "cl_x", np.zeros(4, dtype=np.float32).tobytes())
    newer = sealer.seal(0, "cl_x", np.ones(4, dtype=np.float32).tobytes())
    # Both authenticate for (row 0, owner cl_x): freshness is not bound.
    assert sealer.open(0, "cl_x", older) == np.zeros(4, dtype=np.float32).tobytes()
    assert sealer.open(0, "cl_x", newer) == np.ones(4, dtype=np.float32).tobytes()
    # But the guarantee holds: the same record for another row or owner does not open.
    assert sealer.open(1, "cl_x", older) is None
    assert sealer.open(0, "cl_y", older) is None


@pytest.mark.skipif(os.name != "posix", reason="no POSIX permission bits to check")
@needs_extra
def test_a_key_file_other_users_can_read_is_named_in_a_warning(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A key kept in `~/.memvara/db.key` that other users can read draws an
    `EncryptionWarning` naming its mode, so a store whose key sits world-readable beside it
    says so rather than staying quiet."""
    home = tmp_path / "home"
    (home / ".memvara").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    path = key_file()
    path.write_text(KEY + "\n", encoding="utf-8")
    path.chmod(0o644)
    db = tmp_path / "e.db"
    with pytest.warns(EncryptionWarning, match="644"):
        # No key in the environment, so the store reads the loose key file.
        with stores.file(db, encryption=True):
            pass
    assert stat.S_IMODE(path.stat().st_mode) == 0o644
