"""Encryption at rest for a local store: the key, the two encrypted files, and conversion.

The failures these tests exist to prevent, most expensive first:

1. **A store that claims to be encrypted and is not.** The database with its text in the
   clear, or a vector file of plaintext floats beside an encrypted database. The second is
   the one that was the reason this feature was once declined: a plaintext embedding lets
   anyone with the file confirm a guess about the text. The tests read the bytes on disk.
2. **A damaged vector file that loads.** A record moved to another row, a file cut short,
   a record edited, a header swapped: each must fail with an error naming the file, not
   load zeros or someone else's vector and quietly weaken every search.
3. **A key that leaks.** Into an error message, a warning, a `repr` or CLI output. The
   one command allowed to print it is `memvara encrypt --export-key`.
4. **A conversion that damages the original when it fails.** `memvara encrypt` works on a
   copy and renames it over the original only after checking it; every failure before
   that point must leave the original byte for byte as it was.
5. **Silently falling back to plaintext.** A new store with encryption asked for and the
   extra missing is refused, and an existing unencrypted store is opened with a warning
   and a `memory_stats` line rather than refused.

No test touches the developer's keychain or `~/.memvara`: `tests/conftest.py` replaces
the keychain lookup and points `HOME` at a temporary directory for every test.
"""

from __future__ import annotations

import importlib.util
import io
import os
import re
import sqlite3
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pytest

import memvara.store.encryption as enc
from memvara import HashingEmbedder, Memvara, NullLLM
from memvara.cli import main as cli_main
from memvara.retrieve import EpisodeResult
from memvara.server.config import ConfigError, ServerConfig, build_memvara
from memvara.server.mcp import MemvaraMCPServer, _storage_fact
from memvara.store import SQLiteStore, StoreInUseError
from memvara.store.encryption import (EncryptionError, EncryptionUnavailable,
                                      EncryptionWarning, StoreKey, VectorSealer,
                                      encrypt_store, export_key, file_kind, key_file,
                                      resolve_key)
from memvara.types import Claim, Scope

from conftest import REAL_READ_KEYCHAIN

needs_extra = pytest.mark.skipif(
    importlib.util.find_spec("sqlcipher3") is None
    or importlib.util.find_spec("cryptography") is None,
    reason="needs the encrypt extra: pip install 'memvara[encrypt]'")

SCOPE = Scope("acme", "alice")
KEY = bytes(range(32))
OTHER = bytes(range(1, 33))
DIM = 16
#: 12-byte nonce, the row's float32 values, 16-byte tag.
RECORD = 12 + DIM * 4 + 16
HEADER = 64
SECRET = "Zanzibar-7731"


def onehot(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i % DIM] = 1.0
    return v


def embed(store: SQLiteStore, i: int, obj: str | None = None) -> Claim:
    c = Claim(subject="user", predicate=f"p{i}", object=obj or f"value {i}", scope=SCOPE)
    store.put_claim(c)
    store.set_embedding(c.id, onehot(i))
    return c


def hits(store: SQLiteStore, i: int) -> list[str]:
    return [cid for cid, score in store.vector_search(onehot(i), [SCOPE], 3) if score > 0.99]


def all_bytes(path: str) -> bytes:
    """Every file the store has beside it, concatenated: the database, the log, vectors."""
    out = b""
    for name in (path, path + "-wal", path + "-shm", path + ".vecs"):
        if os.path.exists(name):
            out += Path(name).read_bytes()
    return out


@pytest.fixture
def keyed(monkeypatch):
    """The key in MEMVARA_DB_KEY, which is where a server usually gets it."""
    monkeypatch.setenv("MEMVARA_DB_KEY", KEY.hex())
    return KEY


def fake_keyring(monkeypatch, *, value=None, raises=None):
    """A `keyring` module that answers from memory, and the real lookup that reads it."""
    calls = []

    def get_password(service, username):
        calls.append((service, username))
        if raises is not None:
            raise raises
        return value

    monkeypatch.setitem(sys.modules, "keyring",
                        types.SimpleNamespace(get_password=get_password))
    monkeypatch.setattr(enc, "_read_keychain", REAL_READ_KEYCHAIN)
    return calls


# --- the key -----------------------------------------------------------------


def test_the_keychain_is_asked_first_under_the_documented_names(monkeypatch, tmp_path):
    """A key in the keychain wins over the environment and the file, in that order."""
    calls = fake_keyring(monkeypatch, value=KEY.hex())
    monkeypatch.setenv("MEMVARA_DB_KEY", OTHER.hex())
    found = resolve_key(create=False)
    assert found.key == KEY and found.source == "keychain"
    assert calls == [("memvara", "db-key")]
    assert "OS keychain" in found.describe()


def test_the_environment_is_used_when_the_keychain_has_nothing(monkeypatch):
    fake_keyring(monkeypatch, value=None)
    monkeypatch.setenv("MEMVARA_DB_KEY", f"  {KEY.hex()}\n")
    found = resolve_key(create=False)
    assert (found.key, found.source) == (KEY, "environment")
    assert found.describe() == "the MEMVARA_DB_KEY environment variable"


def test_an_env_mapping_is_read_instead_of_the_process_environment():
    found = resolve_key(create=False, env={"MEMVARA_DB_KEY": KEY.hex()})
    assert found.key == KEY


def test_a_keychain_that_cannot_be_read_is_skipped_and_named_when_no_key_is_found(
        monkeypatch):
    """A Linux server has no Secret Service. That must not stop the environment variable
    from working, and when nothing works the error must say the keychain was unreadable,
    because that is the likeliest reason a key that is in there was not found."""
    fake_keyring(monkeypatch, raises=RuntimeError("no secret service on this bus"))
    monkeypatch.setenv("MEMVARA_DB_KEY", KEY.hex())
    assert resolve_key(create=False).source == "environment"
    monkeypatch.delenv("MEMVARA_DB_KEY")
    with pytest.raises(EncryptionError) as caught:
        resolve_key(create=False)
    message = str(caught.value)
    assert "could not be read (RuntimeError: no secret service on this bus)" in message
    assert "MEMVARA_DB_KEY" in message and str(key_file()) in message
    assert "cannot be read" in message


def test_a_missing_keyring_package_is_a_skipped_source(monkeypatch):
    monkeypatch.setattr(enc, "_read_keychain", REAL_READ_KEYCHAIN)
    monkeypatch.setitem(sys.modules, "keyring", None)
    with pytest.raises(EncryptionError, match="keyring package is not installed"):
        resolve_key(create=False)


def test_a_generated_key_is_written_once_with_mode_0600_and_a_backup_warning():
    """Generated only when asked to create, written where the docs say, never overwritten,
    and loud about the fact that losing it loses the store."""
    with pytest.warns(EncryptionWarning, match="memvara encrypt --export-key"):
        made = resolve_key(create=True)
    assert made.source == "generated" and len(made.key) == 32
    assert key_file().read_text().strip() == made.key.hex()
    if os.name == "posix":
        assert key_file().stat().st_mode & 0o777 == 0o600
        assert key_file().parent.stat().st_mode & 0o777 == 0o700
    assert made.key.hex() not in made.describe()
    with pytest.warns(EncryptionWarning, match="same home directory"):
        again = resolve_key(create=True)
    assert (again.key, again.source) == (made.key, "file")
    assert "generated" not in again.describe()


def test_two_processes_generating_at_once_end_up_with_one_key(monkeypatch):
    """The loser of the O_EXCL race reads the winner's key rather than replacing it."""
    key_file().parent.mkdir(parents=True)
    key_file().write_text(KEY.hex())
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    found = resolve_key(create=True)
    assert (found.key, found.source) == (KEY, "file")


def test_generating_leaves_no_temporary_file_and_never_replaces_a_key(monkeypatch):
    """The key is written to a temporary file and linked into place, so no other process
    can ever see the key file half written, and a key that is already there stays."""
    with pytest.warns(EncryptionWarning):
        made = resolve_key(create=True)
    assert sorted(p.name for p in key_file().parent.iterdir()) == ["db.key"]
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    again = resolve_key(create=True)
    assert again.key == made.key
    assert sorted(p.name for p in key_file().parent.iterdir()) == ["db.key"]


def test_a_key_file_still_being_written_is_waited_for(monkeypatch):
    """A key file that exists but is still empty belongs to a process that is writing it.
    It is read again, briefly, instead of being reported as a malformed key."""
    key_file().parent.mkdir(parents=True)
    key_file().write_text("")
    naps: list[float] = []

    def nap(seconds: float) -> None:
        naps.append(seconds)
        if len(naps) == 3:
            key_file().write_text(KEY.hex() + "\n")

    monkeypatch.setattr(enc.time, "sleep", nap)
    with pytest.warns(EncryptionWarning):
        found = resolve_key(create=False)
    assert found.key == KEY and len(naps) == 3


def test_a_key_file_that_stays_empty_is_an_error_after_the_wait(monkeypatch):
    key_file().parent.mkdir(parents=True)
    key_file().write_text("")
    naps: list[float] = []
    monkeypatch.setattr(enc.time, "sleep", naps.append)
    with pytest.raises(EncryptionError, match="0 characters"):
        resolve_key(create=False)
    # Compared as a list, not as a sum: before Python 3.12, `sum()` of twenty 0.05s is
    # 1.0000000000000002, and a bound on the total failed on 3.10 and 3.11 only.
    assert naps == [enc._KEY_FILE_PAUSE] * enc._KEY_FILE_READS
    assert enc._KEY_FILE_PAUSE * enc._KEY_FILE_READS == pytest.approx(1.0)


def test_the_loser_of_a_generation_race_waits_for_the_winners_key(monkeypatch):
    """Two processes generating at once: the loser finds the winner's file, possibly
    before the winner has finished writing it, and waits for the key rather than failing.
    """
    key_file().parent.mkdir(parents=True)
    key_file().write_text("")
    monkeypatch.setattr(Path, "is_file", lambda self: False)

    def nap(seconds: float) -> None:
        key_file().write_text(KEY.hex())

    monkeypatch.setattr(enc.time, "sleep", nap)
    assert resolve_key(create=True).key == KEY
    assert sorted(p.name for p in key_file().parent.iterdir()) == ["db.key"]


def test_generating_where_hard_links_are_not_supported_still_writes_the_key(monkeypatch):
    """Some file systems (FAT, some network shares) refuse a hard link. The key is then
    created in place with O_EXCL, which still never overwrites a key."""
    def refuse(src, dst):
        raise OSError("hard links are not supported here")

    monkeypatch.setattr(enc.os, "link", refuse)
    with pytest.warns(EncryptionWarning):
        made = resolve_key(create=True)
    assert key_file().read_text().strip() == made.key.hex()
    assert sorted(p.name for p in key_file().parent.iterdir()) == ["db.key"]
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    assert resolve_key(create=True).key == made.key


@pytest.mark.skipif(os.name != "posix", reason="Windows file modes do not express this")
def test_a_key_file_others_can_read_is_named_in_the_warning():
    """As ssh does for a private key, a key file readable by the group or by everyone is
    reported with its mode and the command that fixes it."""
    key_file().parent.mkdir(parents=True)
    key_file().write_text(KEY.hex())
    key_file().chmod(0o644)
    with pytest.warns(EncryptionWarning) as caught:
        resolve_key(create=False)
    message = str(caught[0].message)
    assert "mode is 644" in message and f"chmod 600 {key_file()}" in message
    key_file().chmod(0o600)
    with pytest.warns(EncryptionWarning) as caught:
        resolve_key(create=False)
    assert "chmod" not in str(caught[0].message)


def test_windows_modes_are_not_read_as_permissions(monkeypatch):
    """On Windows the mode bits do not say who can read a file, so nothing is claimed."""
    key_file().parent.mkdir(parents=True)
    key_file().write_text(KEY.hex())
    key_file().chmod(0o644)
    monkeypatch.setattr(enc, "_POSIX_MODES", False)
    with pytest.warns(EncryptionWarning) as caught:
        resolve_key(create=False)
    assert "chmod" not in str(caught[0].message)


def test_no_key_anywhere_is_an_error_when_not_creating():
    with pytest.raises(EncryptionError, match="No store key was found"):
        resolve_key(create=False)
    assert not key_file().exists()


@pytest.mark.parametrize("bad", ["abc", "zz" * 32, KEY.hex() + "00"])
def test_a_malformed_key_is_refused_without_repeating_it(monkeypatch, bad):
    monkeypatch.setenv("MEMVARA_DB_KEY", bad)
    with pytest.raises(EncryptionError) as caught:
        resolve_key(create=False)
    assert bad not in str(caught.value)
    assert "64 hexadecimal characters" in str(caught.value)


def test_a_malformed_keychain_entry_and_key_file_are_named_as_the_source(monkeypatch):
    fake_keyring(monkeypatch, value="nope")
    with pytest.raises(EncryptionError, match="The OS keychain entry memvara/db-key"):
        resolve_key(create=False)
    fake_keyring(monkeypatch, value=None)
    key_file().parent.mkdir(parents=True)
    key_file().write_text("nope")
    with pytest.raises(EncryptionError, match="The key file"):
        resolve_key(create=False)


def test_the_key_is_not_in_the_repr_of_the_object_that_carries_it():
    found = StoreKey(KEY, "environment")
    assert KEY.hex() not in repr(found) and repr(KEY) not in repr(found)


def test_export_key_never_generates_one():
    with pytest.raises(EncryptionError):
        export_key()
    assert not key_file().exists()


# --- the file on disk --------------------------------------------------------


def test_file_kind_reads_the_first_bytes(tmp_path):
    assert file_kind(str(tmp_path / "missing.db")) == "new"
    (tmp_path / "empty.db").write_bytes(b"")
    assert file_kind(str(tmp_path / "empty.db")) == "new"
    sqlite3.connect(tmp_path / "plain.db").execute("CREATE TABLE t (x)").connection.close()
    assert file_kind(str(tmp_path / "plain.db")) == "plain"
    (tmp_path / "noise.db").write_bytes(os.urandom(64))
    assert file_kind(str(tmp_path / "noise.db")) == "other"
    (tmp_path / "a-directory").mkdir()
    assert file_kind(str(tmp_path / "a-directory")) == "unreadable"


def test_a_path_that_cannot_be_read_fails_the_way_it_always_did(tmp_path):
    """SQLite's own error, not an OSError from peeking at the header first."""
    (tmp_path / "a-directory").mkdir()
    with pytest.raises(sqlite3.OperationalError):
        SQLiteStore(str(tmp_path / "a-directory"), encryption=True)


@needs_extra
def test_neither_the_text_nor_the_vectors_are_on_disk_in_the_clear(tmp_path, keyed):
    """The property the feature exists for, checked on the bytes, with a control.

    The control is the same writes into an unencrypted store, which must contain both the
    text and the raw vector bytes; without it this test would pass on a search that simply
    looked in the wrong place.
    """
    for encrypted in (False, True):
        path = str(tmp_path / f"{encrypted}.db")
        with SQLiteStore(path, encryption=encrypted) as store:
            c = embed(store, 3, obj=SECRET)
            raw = store.get_embedding(c.id).tobytes()
            assert store.encrypted is encrypted
            during = all_bytes(path)
        after = all_bytes(path)
        for blob in (during, after):
            assert (SECRET.encode() in blob) is not encrypted
            assert (raw in blob) is not encrypted
    assert Path(tmp_path / "True.db").read_bytes()[:16] != b"SQLite format 3\x00"
    assert Path(tmp_path / "True.db.vecs").read_bytes()[:8] == VectorSealer.MAGIC


@needs_extra
def test_an_encrypted_store_reopens_and_its_vectors_still_answer(tmp_path, keyed):
    path = str(tmp_path / "m.db")
    with SQLiteStore(path, encryption=True) as store:
        ids = [embed(store, i).id for i in range(5)]
        assert hits(store, 2) == [ids[2]]
    with SQLiteStore(path) as store:          # encryption=False: the file decides
        assert store.encrypted and store.key_source.startswith("the MEMVARA_DB_KEY")
        assert [hits(store, i) for i in range(5)] == [[c] for c in ids]
        assert store.lexical_search("value", [SCOPE], 10)


@needs_extra
def test_a_vector_file_that_ends_in_a_ctrl_z_byte_reopens_intact(tmp_path):
    """Windows' C runtime deletes a final 0x1A byte from a file opened for reading and
    writing in text mode, and `os.open` opens in text mode unless it is told otherwise.

    The last byte of the encrypted vector file is the last byte of an authentication
    tag, so it is 0x1A once in every 256 writes. When it was, reopening the store cut
    the file by one byte and the store refused to open with "the record ends before
    it". CI saw that as an intermittent failure; for a user it is a store that will not
    open. Re-embedding the last claim seals it again under a new nonce, so this repeats
    until the file ends in 0x1A and then checks that a reopen keeps the byte. It passes
    on every platform, and fails on Windows if the file is opened in text mode.
    """
    path = str(tmp_path / "m.db")
    with SQLiteStore(path, key=KEY) as store:
        ids = [embed(store, i).id for i in range(3)]
        size = os.path.getsize(path + ".vecs")
        for _ in range(5000):
            if Path(path + ".vecs").read_bytes()[-1] == 0x1A:
                break
            store.set_embedding(ids[-1], onehot(2))
        # A re-embedded claim keeps its row, so the file did not grow while looking.
        assert os.path.getsize(path + ".vecs") == size
        assert Path(path + ".vecs").read_bytes()[-1] == 0x1A
    with SQLiteStore(path, key=KEY) as store:
        assert os.path.getsize(path + ".vecs") == size
        assert [hits(store, i) for i in range(3)] == [[c] for c in ids]


@needs_extra
def test_memvara_end_to_end_with_encryption(tmp_path, keyed):
    path = str(tmp_path / "m.db")
    with Memvara(path, embedder=HashingEmbedder(dim=64), llm=NullLLM(), user="u",
                 encryption=True) as mem:
        mem.remember("user", "lives_in", SECRET)
    assert SECRET.encode() not in all_bytes(path)
    with Memvara(path, embedder=HashingEmbedder(dim=64), llm=NullLLM(), user="u") as mem:
        assert mem.store.encrypted
        assert SECRET in mem.search(SECRET, k=1)[0].claim.object


@needs_extra
@pytest.mark.parametrize("json_functions", [True, False])
def test_a_filtered_search_runs_against_an_encrypted_store(tmp_path, keyed,
                                                           json_functions):
    """Metadata and file-path filters run inside SQL, through SQLite's JSON functions or,
    where SQLite has none, through `mv_meta_match`, a function registered on each
    connection. An encrypted store's connections, the writer's and each reading thread's,
    come from SQLCipher, and a function missing from one of them would fail every
    filtered read with "no such function"."""
    path = str(tmp_path / "m.db")
    with Memvara(path, embedder=HashingEmbedder(dim=64), llm=NullLLM(), user="u",
                 encryption=True) as mem:
        mem.remember("user", "prefers", "tabs", team="web")
        mem.remember("user", "prefers", "spaces", team="infra")
        mem.add_document("The refund window is thirty days.", filepath="policies/r.md")
        mem.add_document("The refund window is ninety days.", filepath="drafts/r.md")
    with Memvara(path, embedder=HashingEmbedder(dim=64), llm=NullLLM(),
                 user="u") as mem:
        assert mem.store.encrypted
        mem.store._json_functions = json_functions
        assert mem.store._reader() is not None
        found = mem.search("prefers", filters={"team": "web"})
        assert [r.claim.object for r in found] == ["tabs"]
        turns = mem.search("refund window", include_episodes=True,
                           filepath_prefix="policies/")
        texts = [r.episode.content for r in turns if isinstance(r, EpisodeResult)]
        assert texts and all("thirty" in t for t in texts)


@needs_extra
def test_the_wrong_key_is_an_error_that_names_where_the_key_came_from(tmp_path, keyed,
                                                                     monkeypatch):
    path = str(tmp_path / "m.db")
    with SQLiteStore(path, encryption=True) as store:
        embed(store, 1)
    monkeypatch.setenv("MEMVARA_DB_KEY", OTHER.hex())
    with pytest.raises(EncryptionError) as caught:
        SQLiteStore(path)
    message = str(caught.value)
    assert "MEMVARA_DB_KEY environment variable" in message
    assert "cannot be read" in message
    assert KEY.hex() not in message and OTHER.hex() not in message
    with pytest.raises(EncryptionError, match="the key passed to SQLiteStore"):
        SQLiteStore(path, key=OTHER)


@needs_extra
def test_an_encrypted_store_with_no_key_anywhere_says_where_it_looked(tmp_path, keyed,
                                                                      monkeypatch):
    path = str(tmp_path / "m.db")
    SQLiteStore(path, encryption=True).close()
    monkeypatch.delenv("MEMVARA_DB_KEY")
    with pytest.raises(EncryptionError, match="No store key was found"):
        SQLiteStore(path)
    assert not key_file().exists(), "opening must never generate a key"


def test_a_key_of_the_wrong_length_is_refused(tmp_path):
    with pytest.raises(ValueError, match="32 bytes"):
        SQLiteStore(str(tmp_path / "m.db"), key=b"short")


def test_an_in_memory_store_is_never_encrypted_and_never_asks_for_a_key(monkeypatch):
    monkeypatch.setattr(enc, "resolve_key", lambda **kw: pytest.fail("asked for a key"))
    with SQLiteStore(":memory:", encryption=True) as store:
        assert store.encrypted is False and store.key_source is None


def test_an_unencrypted_store_opens_with_a_warning_when_encryption_is_asked_for(tmp_path):
    """Refusing would lock somebody out of the memory they already have."""
    path = str(tmp_path / "m.db")
    with SQLiteStore(path) as store:
        embed(store, 1)
    with pytest.warns(EncryptionWarning, match=re.escape(f"memvara encrypt {path}")):
        store = SQLiteStore(path, encryption=True)
    with store:
        assert store.encrypted is False and hits(store, 1)


def test_a_new_encrypted_store_without_the_extra_is_refused_naming_it(tmp_path,
                                                                      monkeypatch):
    monkeypatch.setitem(sys.modules, "sqlcipher3", None)
    monkeypatch.setitem(sys.modules, "sqlcipher3.dbapi2", None)
    with pytest.raises(EncryptionUnavailable, match=r"memvara\[encrypt\]"):
        SQLiteStore(str(tmp_path / "m.db"), encryption=True)
    assert not os.path.exists(tmp_path / "m.db")
    # The same store without encryption asked for is created as it always was.
    with SQLiteStore(str(tmp_path / "m.db")) as store:
        assert store.encrypted is False


def test_an_encrypted_file_without_the_extra_is_refused_naming_it(tmp_path, monkeypatch):
    (tmp_path / "m.db").write_bytes(os.urandom(4096))
    monkeypatch.setitem(sys.modules, "sqlcipher3", None)
    monkeypatch.setitem(sys.modules, "sqlcipher3.dbapi2", None)
    with pytest.raises(EncryptionUnavailable, match=r"memvara\[encrypt\]"):
        SQLiteStore(str(tmp_path / "m.db"))


def test_memvara_refuses_encryption_it_could_not_apply():
    with pytest.raises(TypeError, match="cannot be combined with store="):
        Memvara(store=SQLiteStore(), encryption=True, embedder=HashingEmbedder(dim=8))
    with pytest.raises(TypeError, match="cannot be combined with store="):
        Memvara(store=SQLiteStore(), key_env={}, embedder=HashingEmbedder(dim=8))
    with pytest.raises(TypeError, match="encryption"):
        Memvara(api_key="k", encryption=True)
    with pytest.raises(TypeError, match="key_env"):
        Memvara(api_key="k", key_env={"MEMVARA_DB_KEY": KEY.hex()})


# --- the encrypted vector file ------------------------------------------------


def _written(tmp_path, n: int = 4) -> tuple[str, list[str]]:
    path = str(tmp_path / "m.db")
    with SQLiteStore(path, key=KEY) as store:
        ids = [embed(store, i).id for i in range(n)]
    return path, ids


def _record(path: str, slot: int) -> bytes:
    with open(path + ".vecs", "rb") as fh:
        fh.seek(HEADER + slot * RECORD)
        return fh.read(RECORD)


def _put_record(path: str, slot: int, data: bytes) -> None:
    with open(path + ".vecs", "r+b") as fh:
        fh.seek(HEADER + slot * RECORD)
        fh.write(data)


@needs_extra
def test_a_record_moved_to_another_row_fails_to_load(tmp_path):
    """Each record is bound to its row and its owner, so swapping two is caught."""
    path, _ = _written(tmp_path)
    a, b = _record(path, 0), _record(path, 1)
    _put_record(path, 0, b)
    _put_record(path, 1, a)
    with SQLiteStore(path, key=KEY) as store:
        with pytest.raises(EncryptionError) as caught:
            hits(store, 0)
    message = str(caught.value)
    assert "fails authentication" in message and path + ".vecs" in message
    assert "deleting the vector file is safe" in message
    assert KEY.hex() not in message


@needs_extra
def test_a_vector_file_cut_short_fails_to_load(tmp_path):
    path, _ = _written(tmp_path)
    with open(path + ".vecs", "r+b") as fh:
        fh.truncate(HEADER + 2 * RECORD + 5)
    with SQLiteStore(path, key=KEY) as store:
        with pytest.raises(EncryptionError, match="ends before it"):
            hits(store, 0)


@needs_extra
def test_an_edited_record_fails_to_load(tmp_path):
    path, _ = _written(tmp_path)
    record = bytearray(_record(path, 2))
    record[20] ^= 0x01
    _put_record(path, 2, bytes(record))
    with SQLiteStore(path, key=KEY) as store:
        with pytest.raises(EncryptionError, match="row 2"):
            hits(store, 2)


@needs_extra
def test_a_record_from_a_file_written_with_another_key_fails_to_load(tmp_path):
    """Same store, same rows, same header, records sealed under a different key: the
    header is the one this store writes, so it is not rebuilt, and every record fails."""
    path, ids = _written(tmp_path)
    with SQLiteStore(path, key=KEY) as store:
        salt = store._vec._sealer.salt
        slots = dict(store._db.execute("SELECT claim_id, slot FROM embeddings").fetchall())
    forger = VectorSealer(OTHER)
    forger.salt = salt
    forger.bind(DIM)
    for i, cid in enumerate(ids):
        _put_record(path, slots[cid], forger.seal(slots[cid], cid, onehot(i).tobytes()))
    with SQLiteStore(path, key=KEY) as store:
        with pytest.raises(EncryptionError, match="another key"):
            hits(store, 0)


@needs_extra
def test_a_vector_file_from_another_store_is_rebuilt_not_trusted(tmp_path):
    """Another store's file carries another database's salt in its header, so this store
    does not recognise it as its own and rebuilds it from the database."""
    path, ids = _written(tmp_path)
    other = str(tmp_path / "other.db")
    with SQLiteStore(other, key=KEY) as store:
        for i in range(4):
            embed(store, i)
    Path(path + ".vecs").write_bytes(Path(other + ".vecs").read_bytes())
    with SQLiteStore(path, key=KEY) as store:
        assert [hits(store, i) for i in range(4)] == [[c] for c in ids]


@needs_extra
def test_two_processes_rewriting_a_missing_vector_file_agree_on_its_key(tmp_path):
    """The vector file's key is derived from a salt both processes read from the database,
    so two processes that each find the header missing write the same header.

    The interleaving that broke it: process A opens the store and rebuilds the file;
    process B, which also found the header missing a moment earlier, truncates the file
    and writes its own header; A then seals a new vector. With a salt chosen at random by
    each process, A's new record is under a key B's header does not describe, and the
    next open refuses the store. The third open here must succeed and find every vector.
    """
    path, ids = _written(tmp_path)
    os.remove(path + ".vecs")
    a = SQLiteStore(path, key=KEY)               # rebuilds the file, header included
    with open(path + ".vecs", "r+b") as fh:      # B found it missing too and starts over
        fh.truncate(0)
    b = SQLiteStore(path, key=KEY)
    late = embed(a, 9).id
    a.close()
    b.close()
    with SQLiteStore(path, key=KEY) as third:
        assert [hits(third, i) for i in range(4)] == [[c] for c in ids]
        assert hits(third, 9) == [late]


@needs_extra
def test_two_stores_opening_a_missing_vector_file_at_once_leave_it_readable(tmp_path):
    """The same race with real threads rather than a scripted interleaving."""
    import threading

    path, ids = _written(tmp_path, n=8)
    os.remove(path + ".vecs")
    opened: list[SQLiteStore] = []
    start = threading.Barrier(2)

    def open_one() -> None:
        start.wait()
        opened.append(SQLiteStore(path, key=KEY))

    threads = [threading.Thread(target=open_one) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    extra = [embed(store, 10 + n).id for n, store in enumerate(opened)]
    for store in opened:
        store.close()
    with SQLiteStore(path, key=KEY) as third:
        assert [hits(third, i) for i in range(8)] == [[c] for c in ids]
        assert [hits(third, 10 + n) for n in range(2)] == [[c] for c in extra]


@needs_extra
def test_a_record_that_fails_once_and_then_opens_is_a_torn_read_not_damage(tmp_path):
    """Another process may be rewriting a row at the moment it is read. One failure is
    followed by one more read, and a record that opens then is used."""
    path, ids = _written(tmp_path)
    with SQLiteStore(path, key=KEY) as store:
        real = store._vec.load
        seen: list[str] = []

        def flaky(item_id, slot):
            seen.append(item_id)
            return "bad" if seen.count(item_id) == 1 and item_id == ids[1] else \
                real(item_id, slot)

        store._vec.load = flaky
        assert hits(store, 1) == [ids[1]]
        assert seen.count(ids[1]) == 2


@needs_extra
def test_a_blank_record_is_read_from_the_database_instead(tmp_path):
    """A record another process has blanked while erasing its owner, before that erasure
    commits, is not damage. The vector comes from the database, which is the authority
    and is encrypted and authenticated itself."""
    path, ids = _written(tmp_path)
    _put_record(path, 1, bytes(RECORD))
    with SQLiteStore(path, key=KEY) as store:
        assert hits(store, 1) == [ids[1]]


@needs_extra
def test_an_owner_erased_or_moved_while_loading_takes_the_databases_answer(tmp_path):
    """The two races the reload exists for, made to happen in the middle of a load."""
    path, ids = _written(tmp_path)
    with SQLiteStore(path, key=KEY) as store, SQLiteStore(path, key=KEY) as other:
        real = store._vec.load

        def racing(item_id, slot):
            if item_id == ids[0]:
                # Another process erases the claim after this load read its slot.
                other.erase_claim(ids[0])
            if item_id == ids[1]:
                # Another process moves the vector to a new row, and its old record is
                # blanked on the way.
                other._db.execute("UPDATE embeddings SET slot = 9 WHERE claim_id = ?",
                                  (ids[1],))
                other._db.commit()
                _put_record(path, 1, bytes(RECORD))
            return real(item_id, slot)

        store._vec.load = racing
        assert hits(store, 0) == []
        assert store._vec._row[ids[1]] == 9
        assert hits(store, 1) == [ids[1]]


@needs_extra
def test_a_damaged_header_or_a_missing_file_is_rebuilt_from_the_database(tmp_path):
    """Neither is data loss: the database holds every vector. The rewritten header is the
    same one every process opening this database writes, because its salt is the
    database's own."""
    path, ids = _written(tmp_path)
    before = Path(path + ".vecs").read_bytes()[:32]
    with open(path + ".vecs", "r+b") as fh:
        fh.write(b"XXXXXXXX")
    with SQLiteStore(path, key=KEY) as store:
        assert hits(store, 3) == [ids[3]]
        salt = store._vec._sealer.salt
    head = Path(path + ".vecs").read_bytes()[:32]
    assert head == before and head[:8] == VectorSealer.MAGIC and head[16:32] == salt
    os.remove(path + ".vecs")
    with SQLiteStore(path, key=KEY) as store:
        assert hits(store, 2) == [ids[2]]


@needs_extra
def test_a_plaintext_vector_file_beside_an_encrypted_database_is_replaced(tmp_path):
    path, ids = _written(tmp_path)
    plain = str(tmp_path / "plain.db")
    with SQLiteStore(plain) as store:
        for i in range(4):
            embed(store, i)
    Path(path + ".vecs").write_bytes(Path(plain + ".vecs").read_bytes())
    with SQLiteStore(path, key=KEY) as store:
        assert hits(store, 0) == [ids[0]]
    assert Path(path + ".vecs").read_bytes()[:8] == VectorSealer.MAGIC


@needs_extra
def test_erasing_a_claim_blanks_its_record_in_the_file(tmp_path):
    """Encrypted is not erased: whoever holds the key could still decrypt a record left
    behind, so erasure zeroes it, as it zeroes a plaintext row."""
    path, ids = _written(tmp_path)
    assert _record(path, 2).strip(b"\0")
    with SQLiteStore(path, key=KEY) as store:
        hits(store, 2)                                   # load the index first
        store.erase_claim(ids[2])
    assert _record(path, 2) == bytes(RECORD)
    with SQLiteStore(path, key=KEY) as store:
        assert hits(store, 2) == [] and hits(store, 1) == [ids[1]]


@needs_extra
def test_another_process_write_is_seen_after_a_commit(tmp_path):
    """No shared mapping, so a vector another process wrote must be read in explicitly;
    otherwise it is silently missing from this process's vector search."""
    path, ids = _written(tmp_path, n=2)
    with SQLiteStore(path, key=KEY) as reader, SQLiteStore(path, key=KEY) as writer:
        assert hits(reader, 0) == [ids[0]]
        late = embed(writer, 5).id
        assert hits(reader, 5) == [late]


@needs_extra
def test_clearing_vectors_another_encrypted_store_holds_is_refused(tmp_path):
    """An encrypted store keeps its matrix on the heap, so truncating the file cannot
    crash another store the way it crashes one that maps it. The other store went on
    finding every cleared vector at its old score, even after the store was re-embedded
    at another width. So the refusal is the same."""
    path, ids = _written(tmp_path, n=2)
    with SQLiteStore(path, key=KEY) as reader, SQLiteStore(path, key=KEY) as clearer:
        assert hits(reader, 0) == [ids[0]]
        with pytest.raises(StoreInUseError, match="Nothing was changed"):
            clearer.clear_embeddings()
        assert hits(reader, 0) == [ids[0]] and hits(clearer, 1) == [ids[1]]
    with SQLiteStore(path, key=KEY) as alone:
        assert alone.clear_embeddings() == 2


@needs_extra
def test_a_rolled_back_batch_puts_the_encrypted_file_back(tmp_path):
    path, ids = _written(tmp_path, n=2)
    with SQLiteStore(path, key=KEY) as store:
        hits(store, 0)
        before = _record(path, 0)
        with pytest.raises(RuntimeError):
            with store.batch():
                store.set_embedding(ids[0], onehot(7))
                raise RuntimeError("abort")
        assert hits(store, 0) == [ids[0]]
        assert _record(path, 0) != before              # rewritten, fresh nonce
    with SQLiteStore(path, key=KEY) as store:
        assert hits(store, 0) == [ids[0]] and hits(store, 7) == []


@needs_extra
def test_clearing_embeddings_drops_the_encrypted_file_and_starts_a_new_one(tmp_path):
    path, ids = _written(tmp_path)
    with SQLiteStore(path, key=KEY) as store:
        assert store.clear_embeddings() == 4
        assert os.path.getsize(path + ".vecs") == 0
        store.set_embedding(ids[0], onehot(1))
        assert hits(store, 1) == [ids[0]]
    with SQLiteStore(path, key=KEY) as store:
        assert hits(store, 1) == [ids[0]]


@needs_extra
def test_the_sealer_accepts_only_its_own_header():
    sealer = VectorSealer(KEY)
    sealer.salt = b"s" * 16
    head = sealer.header(8)
    assert sealer.matches(head, 8)
    assert not sealer.matches(head, 16)
    assert not sealer.matches(head[:20], 8)
    assert not sealer.matches(b"NOTMAGIC" + head[8:], 8)
    other = VectorSealer(KEY)
    other.salt = b"t" * 16
    assert not other.matches(head, 8)


# --- converting an existing store ---------------------------------------------


def _plain_store(tmp_path, n: int = 3) -> tuple[str, list[str]]:
    path = str(tmp_path / "m.db")
    with SQLiteStore(path) as store:
        ids = [embed(store, i, obj=f"{SECRET} {i}").id for i in range(n)]
    return path, ids


@needs_extra
def test_encrypting_a_store_in_place_keeps_every_row_and_vector(tmp_path, keyed):
    path, ids = _plain_store(tmp_path)
    result = encrypt_store(path)
    assert (result.already, result.claims, result.vectors) == (False, 3, 3)
    assert "MEMVARA_DB_KEY" in result.key_source
    assert SECRET.encode() not in all_bytes(path)
    assert Path(path + ".vecs").read_bytes()[:8] == VectorSealer.MAGIC
    assert not [p for p in os.listdir(tmp_path) if "encrypting" in p]
    with SQLiteStore(path) as store:
        assert store.encrypted
        assert [hits(store, i) for i in range(3)] == [[c] for c in ids]
        assert len(store.lexical_search(SECRET, [SCOPE], 10)) == 3


@needs_extra
def test_encrypting_generates_a_key_when_there_is_none(tmp_path):
    path, _ = _plain_store(tmp_path)
    with pytest.warns(EncryptionWarning, match="Generated a new store key"):
        result = encrypt_store(path)
    assert "generated just now" in result.key_source
    with pytest.warns(EncryptionWarning):
        with SQLiteStore(path) as store:
            assert store.encrypted


@needs_extra
def test_encrypting_a_store_with_no_vectors_removes_a_stale_vector_file(tmp_path, keyed):
    path = str(tmp_path / "m.db")
    with SQLiteStore(path) as store:
        store.put_claim(Claim(subject="user", predicate="p", object="o", scope=SCOPE))
    Path(path + ".vecs").write_bytes(b"stale plaintext")
    assert encrypt_store(path).vectors == 0
    assert not os.path.exists(path + ".vecs")


@needs_extra
def test_encrypting_an_encrypted_store_changes_nothing(tmp_path, keyed):
    path, _ = _written(tmp_path)
    before = Path(path).read_bytes()
    assert encrypt_store(path).already is True
    assert Path(path).read_bytes() == before


@needs_extra
def test_encrypting_a_file_the_key_does_not_open_is_refused(tmp_path, keyed):
    path, _ = _written(tmp_path)
    before = Path(path).read_bytes()
    with pytest.raises(EncryptionError, match="Nothing was changed"):
        encrypt_store(path, key=StoreKey(OTHER, "environment"))
    assert Path(path).read_bytes() == before


@needs_extra
def test_encrypting_nothing_or_a_foreign_database_is_refused(tmp_path, keyed):
    with pytest.raises(EncryptionError, match="does not exist"):
        encrypt_store(str(tmp_path / "missing.db"))
    other = tmp_path / "other.db"
    sqlite3.connect(other).execute("CREATE TABLE t (x)").connection.close()
    before = other.read_bytes()
    with pytest.raises(EncryptionError, match="not a memvara store"):
        encrypt_store(str(other))
    assert other.read_bytes() == before


@needs_extra
def test_encrypting_a_store_another_process_has_open_is_refused(tmp_path, keyed):
    path, _ = _plain_store(tmp_path)
    holder = sqlite3.connect(path)
    holder.execute("PRAGMA journal_mode=WAL")
    holder.execute("SELECT COUNT(*) FROM claims").fetchone()
    try:
        before = Path(path).read_bytes()
        with pytest.raises(EncryptionError, match="open in another process"):
            encrypt_store(path)
        assert Path(path).read_bytes() == before
    finally:
        holder.close()


@needs_extra
def test_a_store_changed_during_the_conversion_is_left_alone(tmp_path, keyed,
                                                             monkeypatch):
    """A write that lands in the original after the export would be lost at the rename,
    so a changed original stops the conversion."""
    path, _ = _plain_store(tmp_path)
    real = enc._counts

    def touching(conn, tables):
        if not isinstance(conn, sqlite3.Connection):
            with open(path, "ab") as fh:
                fh.write(b"\0")
        return real(conn, tables)

    monkeypatch.setattr(enc, "_counts", touching)
    before_names = sorted(os.listdir(tmp_path))
    with pytest.raises(EncryptionError, match="opened or changed by another process"):
        encrypt_store(path)
    assert Path(path).read_bytes()[:16] == b"SQLite format 3\x00"
    assert sorted(os.listdir(tmp_path)) == before_names


@needs_extra
def test_a_copy_that_does_not_match_the_original_is_not_renamed(tmp_path, keyed,
                                                                monkeypatch):
    path, _ = _plain_store(tmp_path)
    real = enc._counts
    monkeypatch.setattr(enc, "_counts", lambda conn, tables: {
        **real(conn, tables), **({} if isinstance(conn, sqlite3.Connection)
                                 else {"claims": -1})})
    before = Path(path).read_bytes()
    with pytest.raises(EncryptionError, match="does not hold the same rows"):
        encrypt_store(path)
    assert Path(path).read_bytes() == before
    assert not [p for p in os.listdir(tmp_path) if "encrypting" in p]


@needs_extra
def test_a_copy_whose_vectors_do_not_all_read_back_is_not_renamed(tmp_path, keyed,
                                                                 monkeypatch):
    path, _ = _plain_store(tmp_path)
    real = SQLiteStore._ensure_index

    def losing_one(self):
        real(self)
        self._vec._row.popitem()

    monkeypatch.setattr(SQLiteStore, "_ensure_index", losing_one)
    before = Path(path).read_bytes()
    with pytest.raises(EncryptionError, match="read back 2 of 3 vectors"):
        encrypt_store(path)
    assert Path(path).read_bytes() == before


@needs_extra
def test_a_failure_part_way_leaves_the_original_byte_for_byte(tmp_path, keyed,
                                                              monkeypatch):
    path, _ = _plain_store(tmp_path)
    before = {p: Path(tmp_path / p).read_bytes() for p in os.listdir(tmp_path)}

    def crash(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(enc.os, "replace", crash)
    with pytest.raises(OSError, match="disk full"):
        encrypt_store(path)
    after = {p: Path(tmp_path / p).read_bytes() for p in os.listdir(tmp_path)}
    assert after == before


def test_encrypting_without_the_extra_names_it(tmp_path, monkeypatch):
    path, _ = _plain_store(tmp_path)
    monkeypatch.setitem(sys.modules, "sqlcipher3", None)
    monkeypatch.setitem(sys.modules, "sqlcipher3.dbapi2", None)
    with pytest.raises(EncryptionUnavailable, match=r"memvara\[encrypt\]"):
        encrypt_store(path)


# --- memvara encrypt -----------------------------------------------------------


def run(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = cli_main(["encrypt", *argv], stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


@needs_extra
def test_the_command_encrypts_and_never_prints_the_key(tmp_path, keyed):
    path, _ = _plain_store(tmp_path)
    code, out, err = run(path)
    assert code == 0, err
    assert "Encrypted" in out and "3 vector(s)" in out and "--export-key" in out
    assert KEY.hex() not in out + err
    code, out, _ = run(path)
    assert code == 0 and "already encrypted" in out


def test_export_key_prints_the_key_alone_on_stdout(keyed):
    code, out, err = run("--export-key")
    assert code == 0
    assert out == KEY.hex() + "\n"
    assert "MEMVARA_DB_KEY" in err and KEY.hex() not in err


def test_export_key_with_no_key_fails_without_making_one():
    code, out, err = run("--export-key")
    assert code == 1 and out == "" and "No store key was found" in err
    assert not key_file().exists()


@pytest.mark.parametrize("argv", [(), ("a.db", "b.db"), ("--force",)])
def test_the_command_takes_one_path_or_export_key(argv):
    code, _, err = run(*argv)
    assert code == 2 and "give one store path, or --export-key" in err


def test_the_command_has_help_and_is_listed_in_the_usage():
    code, out, _ = run("--help")
    assert code == 0 and "memvara encrypt --export-key" in out
    listing = io.StringIO()
    assert cli_main(["--help"], stdout=listing) == 0
    assert "memvara encrypt" in listing.getvalue()


def test_the_command_without_the_extra_names_it(tmp_path, monkeypatch):
    path, _ = _plain_store(tmp_path)
    monkeypatch.setitem(sys.modules, "sqlcipher3", None)
    monkeypatch.setitem(sys.modules, "sqlcipher3.dbapi2", None)
    code, _, err = run(path)
    assert code == 2 and "memvara[encrypt]" in err


def test_the_command_reports_a_failure_as_a_sentence(tmp_path):
    code, _, err = run(str(tmp_path / "missing.db"))
    assert code == 1 and "does not exist" in err


# --- the MCP server's switch ---------------------------------------------------


def _server_env(tmp_path, **extra: str) -> dict[str, str]:
    return {"MEMVARA_DB": str(tmp_path / "m.db"), "MEMVARA_EMBEDDER": "hashing:64",
            **extra}


def _stats(memory) -> str:
    server = MemvaraMCPServer(memory)
    try:
        return server._ctx.storage or ""
    finally:
        server.close()


@needs_extra
def test_the_switch_is_on_by_default_and_a_new_store_is_encrypted(tmp_path, keyed):
    memory = build_memvara(ServerConfig.from_env(_server_env(tmp_path)))
    assert memory.store.encrypted
    assert _stats(memory).startswith("encrypted on disk")


def test_the_switch_off_creates_an_unencrypted_store_and_says_so(tmp_path):
    memory = build_memvara(ServerConfig.from_env(
        _server_env(tmp_path, MEMVARA_FEATURE_ENCRYPTION="0")))
    assert not memory.store.encrypted
    assert _stats(memory).startswith("NOT encrypted on disk")


def test_the_switch_on_without_the_extra_refuses_a_new_store(tmp_path, monkeypatch):
    """Refused, not created unencrypted: the store file is made once, and the warning
    that could say so would go to a stderr nobody reads under stdio."""
    monkeypatch.setitem(sys.modules, "sqlcipher3", None)
    monkeypatch.setitem(sys.modules, "sqlcipher3.dbapi2", None)
    with pytest.raises(ConfigError) as caught:
        build_memvara(ServerConfig.from_env(_server_env(tmp_path)))
    assert "memvara[encrypt]" in str(caught.value)
    assert "MEMVARA_FEATURE_ENCRYPTION=0" in str(caught.value)
    assert not os.path.exists(tmp_path / "m.db")


def test_the_switch_on_opens_an_existing_unencrypted_store_and_reports_it(tmp_path):
    SQLiteStore(str(tmp_path / "m.db")).close()
    with pytest.warns(EncryptionWarning, match="memvara encrypt"):
        memory = build_memvara(ServerConfig.from_env(_server_env(tmp_path)))
    assert "memvara encrypt" in _stats(memory)


@needs_extra
def test_a_store_the_server_cannot_find_the_key_for_is_a_config_error(tmp_path, keyed,
                                                                      monkeypatch):
    SQLiteStore(str(tmp_path / "m.db"), encryption=True).close()
    monkeypatch.delenv("MEMVARA_DB_KEY")
    with pytest.raises(ConfigError, match="No store key was found"):
        build_memvara(ServerConfig.from_env(_server_env(tmp_path)))


@needs_extra
def test_the_server_reads_the_key_from_the_environment_it_was_given(tmp_path):
    """`ServerConfig.from_env(env)` reads MEMVARA_DB_KEY from `env`, not from the process
    environment, and the key never reaches the config's repr."""
    config = ServerConfig.from_env(_server_env(tmp_path, MEMVARA_DB_KEY=KEY.hex()))
    assert KEY.hex() not in repr(config)
    memory = build_memvara(config)
    assert memory.store.encrypted
    assert memory.store.key_source == "the MEMVARA_DB_KEY environment variable"
    memory.close()
    with SQLiteStore(str(tmp_path / "m.db"), key=KEY) as store:
        assert store.encrypted


def test_a_malformed_key_in_the_server_environment_is_refused_at_startup(tmp_path):
    with pytest.raises(ConfigError) as caught:
        ServerConfig.from_env(_server_env(tmp_path, MEMVARA_DB_KEY="not-a-key-at-all"))
    assert "MEMVARA_DB_KEY" in str(caught.value)
    assert "not-a-key-at-all" not in str(caught.value)


@needs_extra
def test_the_hooks_open_a_store_whose_key_is_only_in_the_server_block(tmp_path,
                                                                      monkeypatch):
    """The hooks read the MCP client's server block into a dict and never into
    `os.environ`. A key that lives only in that block must still open the store, or every
    prompt silently gets no memory while the server itself works."""
    hooks = Path(__file__).resolve().parent.parent / "plugin" / "hooks"
    monkeypatch.syspath_prepend(str(hooks))
    from lib import ipc
    from lib import open as opener

    block = _server_env(tmp_path, MEMVARA_DB_KEY=KEY.hex())
    SQLiteStore(block["MEMVARA_DB"], key=KEY).close()
    monkeypatch.setattr(ipc, "server_env", lambda: dict(block))
    for name in ("MEMVARA_DB", "MEMVARA_DB_KEY", "MEMVARA_EMBEDDER", "MEMVARA_MODE"):
        monkeypatch.delenv(name, raising=False)
    memory = opener.open_store()
    assert memory is not None, "the hooks could not open the store"
    try:
        assert memory.store.encrypted
        assert memory.store.key_source == "the MEMVARA_DB_KEY environment variable"
    finally:
        memory.close()
    assert not key_file().exists(), "the key came from the block, not a generated file"


def test_memory_stats_says_an_in_memory_store_never_reaches_disk():
    memory = build_memvara(ServerConfig.from_env(
        {"MEMVARA_DB": ":memory:", "MEMVARA_EMBEDDER": "hashing:64"}))
    assert _stats(memory) == "in memory only; nothing is written to disk"


def test_memory_stats_has_no_storage_line_for_a_store_it_cannot_describe():
    assert _storage_fact(types.SimpleNamespace(store=object())) is None
    assert _storage_fact(object()) is None


def test_the_storage_line_reaches_the_memory_stats_output(tmp_path):
    memory = build_memvara(ServerConfig.from_env(
        _server_env(tmp_path, MEMVARA_FEATURE_ENCRYPTION="0")))
    server = MemvaraMCPServer(memory)
    try:
        text = server._tools["memory_stats"].handler(server._ctx, {})
    finally:
        server.close()
    assert "storage: NOT encrypted on disk" in text


def test_no_warning_escapes_an_unencrypted_store_that_never_asked(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", EncryptionWarning)
        SQLiteStore(str(tmp_path / "m.db")).close()
        SQLiteStore(str(tmp_path / "m.db")).close()
