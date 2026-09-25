"""Encryption at rest for a local SQLite store: the key, the vector file, and conversion.

An encrypted store is two encrypted files. The database is encrypted by SQLCipher, which
encrypts and authenticates every page, so the tables, the full-text index, the write-ahead
log and the vectors the database holds are all ciphertext on disk. The vector file beside
it (`<db>.vecs`) is a copy of those vectors that makes opening fast, and in an encrypted
store each of its rows is encrypted with AES-256-GCM (`VectorSealer`). Encrypting only the
database would not be enough, because a plaintext vector lets anyone holding the file
confirm a guess about the text it came from: the embedding of the right guess matches it
exactly.

Everything here is optional. `sqlcipher3`, `cryptography` and `keyring` come from the
`encrypt` extra and are imported only inside the functions that need them, so a plain
`pip install memvara` never loads them (invariant 5 in `docs/INTERNALS.md`).

**The key.** One 32-byte key, written as 64 hexadecimal characters, is looked up in this
order: the OS keychain (the macOS Keychain or the Linux Secret Service, under service
`memvara` and account `db-key`), then the `MEMVARA_DB_KEY` environment variable, then the
file `~/.memvara/db.key`. When none of them has a key and a new encrypted store is being
created, a key is generated into that file with mode 0600, and a warning says so. Losing
the key makes every store it encrypted unreadable. There is no recovery.

**Nothing in this module prints a key**, except `export_key`, whose caller asked for it.
Error messages name where a key came from, never its value.
"""

from __future__ import annotations

import os
import secrets
import sqlite3
import struct
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "EncryptionError", "EncryptionUnavailable", "EncryptionWarning", "EncryptResult",
    "KEY_ENV", "KEYRING_SERVICE", "KEYRING_USERNAME", "StoreKey", "VectorSealer",
    "encrypt_store", "export_key", "file_kind", "key_file", "parse_key",
    "require_sqlcipher", "resolve_key",
]

#: Where the key is looked up in the OS keychain. Documented, because a user who wants the
#: key there puts it there themselves: `python3 -m keyring set memvara db-key`.
KEYRING_SERVICE = "memvara"
KEYRING_USERNAME = "db-key"

#: The environment variable that holds the key when the keychain does not.
KEY_ENV = "MEMVARA_DB_KEY"

#: The first 16 bytes of every unencrypted SQLite database. SQLCipher encrypts the whole
#: first page, header included, so an encrypted store never starts with them.
_SQLITE_MAGIC = b"SQLite format 3\x00"

_INSTALL = "pip install 'memvara[encrypt]'"


class EncryptionError(RuntimeError):
    """An encrypted store could not be opened or converted.

    Raised when no key can be found, when a key is malformed, when the key does not open
    the store, and when the encrypted vector file fails authentication. The message says
    where the key came from and never what it is.
    """


class EncryptionUnavailable(ImportError):
    """Encryption was needed and the `encrypt` extra is not installed.

    An `ImportError`, like every other missing extra in this package, and the message
    names the extra, so the fix is one `pip install` away.
    """


class EncryptionWarning(UserWarning):
    """Something about encryption the person running the store should know.

    Raised as a warning, not an error, for three cases: a store that is not encrypted
    although encryption was asked for, a key read from a file rather than from the OS
    keychain, and a key that was just generated and must be backed up.
    """


def key_file() -> Path:
    """Where a generated key is written, computed when called so `HOME` decides it."""
    return Path.home() / ".memvara" / "db.key"


@dataclass(frozen=True)
class StoreKey:
    """A 32-byte store key and where it was found.

    `repr=False` on the key, so the key cannot reach a log through a debug print of this
    object.
    """

    key: bytes = field(repr=False)
    #: One of "keychain", "environment", "file" or "generated".
    source: str
    #: The file the key was read from or written to, for the two sources that have one.
    path: str | None = None

    def describe(self) -> str:
        """Where the key came from, in words, for an error message or a CLI line."""
        if self.source == "keychain":
            return (f"the OS keychain (service {KEYRING_SERVICE!r}, account "
                    f"{KEYRING_USERNAME!r})")
        if self.source == "environment":
            return f"the {KEY_ENV} environment variable"
        if self.source == "generated":
            return f"the key file {self.path}, generated just now"
        return f"the key file {self.path}"


def parse_key(text: str, where: str) -> bytes:
    """A key from its 64-character hexadecimal form. The error never repeats the text."""
    value = text.strip()
    if len(value) != 64:
        raise EncryptionError(
            f"{where} does not hold a valid store key. A key is 64 hexadecimal "
            f"characters (32 bytes); this one has {len(value)} characters.")
    try:
        return bytes.fromhex(value)
    except ValueError:
        raise EncryptionError(
            f"{where} does not hold a valid store key. A key is 64 hexadecimal "
            "characters (32 bytes), and this one has characters outside 0-9 and a-f."
        ) from None


def _read_keychain() -> tuple[str | None, str | None]:
    """The key text stored in the OS keychain, and why not when there is none.

    Returns `(text, None)` when the keychain holds a key, `(None, None)` when it holds
    none, and `(None, reason)` when it could not be read at all. A keychain that cannot be
    read is not an error here: a Linux server usually has no Secret Service, and the next
    source is the environment variable. The reason is kept so that a store that cannot
    find its key can say that the keychain was not available, which is the likeliest
    explanation when the key really is in there.
    """
    try:
        import keyring
    except ImportError:
        return None, f"the keyring package is not installed ({_INSTALL})"
    try:
        return keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME), None
    except Exception as exc:  # noqa: BLE001
        # Every backend raises its own types: `keyring.errors.KeyringError` and its
        # subclasses, but also D-Bus errors from the Secret Service and OS errors from
        # the macOS Security framework. None of them carries the secret, and each means
        # the same thing here: this source is unavailable, so try the next one.
        return None, f"the OS keychain could not be read ({type(exc).__name__}: {exc})"


def resolve_key(*, create: bool, env: Mapping[str, str] | None = None) -> StoreKey:
    """Find the store key: the OS keychain, then `MEMVARA_DB_KEY`, then the key file.

    `create=True` generates a key into the key file when none of the three has one. Only
    the creation of a new encrypted store passes it. Opening an existing store passes
    `create=False`, because a key generated then could not open the store and would only
    hide the real problem, which is that the store's key is somewhere this process cannot
    see.

    A key read from the key file, or generated into it, comes with an `EncryptionWarning`.
    The file sits in the same home directory as the store, so anyone who can copy one can
    usually copy the other; the keychain or an environment variable filled from a secret
    manager keeps them apart.
    """
    environ = os.environ if env is None else env
    text, unavailable = _read_keychain()
    if text:
        return StoreKey(parse_key(text, "The OS keychain entry memvara/db-key"), "keychain")
    raw = (environ.get(KEY_ENV) or "").strip()
    if raw:
        return StoreKey(parse_key(raw, KEY_ENV), "environment")
    path = key_file()
    if path.is_file():
        found = StoreKey(_read_key_file(path), "file", str(path))
        warnings.warn(EncryptionWarning(
            f"The store key was read from {path}. That file is in the same home "
            "directory as your stores, so a copy of the directory is a copy of the key. "
            "Putting the key in the OS keychain (python3 -m keyring set memvara db-key) "
            f"or in {KEY_ENV} from a secret manager keeps the two apart."
            + _loose_mode(path)), stacklevel=3)
        return found
    if not create:
        looked = f"the OS keychain ({unavailable})" if unavailable else "the OS keychain"
        raise EncryptionError(
            f"No store key was found. memvara looked in {looked}, in the {KEY_ENV} "
            f"environment variable and in {path}. Put the key this store was encrypted "
            "with in one of those places. Without that key the store cannot be read.")
    return _generate(path)


#: How long a reader waits for a key file another process is still writing: 20 reads,
#: 50 ms apart, one second in all. Writing 65 bytes takes far less; the wait exists for
#: the one case below where the file can be seen before it is complete.
_KEY_FILE_READS = 20
_KEY_FILE_PAUSE = 0.05


def _read_key_file(path: Path) -> bytes:
    """The key in `path`, waiting briefly while it is shorter than a key.

    `_generate` links a complete file into place, so a key file is never seen half
    written when it made the file. On a file system without hard links it falls back to
    creating the file in place, and another process can then open it between its
    creation and its write. A file that is still short after the wait is reported as the
    malformed key it is.
    """
    text = path.read_text(encoding="utf-8")
    for _ in range(_KEY_FILE_READS):
        if len(text.strip()) >= 64:
            break
        time.sleep(_KEY_FILE_PAUSE)
        text = path.read_text(encoding="utf-8")
    return parse_key(text, f"The key file {path}")


#: Whether file mode bits say who can read a file. False on Windows; see `_loose_mode`.
_POSIX_MODES = os.name == "posix"


def _loose_mode(path: Path) -> str:
    """A sentence about a key file other users can read, or nothing.

    POSIX only, as ssh does for a private key. On Windows the mode bits Python reports do
    not describe who can read a file (that is an access control list), so there is
    nothing true to say from them, and this says nothing.
    """
    if not _POSIX_MODES:
        return ""
    mode = path.stat().st_mode & 0o777
    if not mode & 0o077:
        return ""
    return (f" The file's mode is {mode:o}, so other users on this machine may be able "
            f"to read the key; run `chmod 600 {path}`.")


def _generate(path: Path) -> StoreKey:
    """Write a new random key to `path`, mode 0600, and warn that it must be backed up.

    The key is written to a temporary file in the same directory and then hard-linked to
    `path`. A link fails when `path` exists, so a key already there is never replaced,
    and it makes the whole file appear at once, so no other process can read it half
    written. Two processes creating their first encrypted stores at the same moment both
    get here; the one whose link fails reads the key the other wrote, and both stores
    share it. Where the file system has no hard links, the key is created in place with
    `O_EXCL`, which still never replaces a key, and a reader waits for it
    (`_read_key_file`).
    """
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    fd, staged = tempfile.mkstemp(prefix=".db.key-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(key.hex() + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(staged, path)
        except FileExistsError:
            return StoreKey(_read_key_file(path), "file", str(path))
        except OSError:
            try:
                made = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                return StoreKey(_read_key_file(path), "file", str(path))
            with os.fdopen(made, "w", encoding="utf-8") as fh:
                fh.write(key.hex() + "\n")
    finally:
        os.remove(staged)
    warnings.warn(EncryptionWarning(
        f"Generated a new store key in {path}. Back it up now: run "
        "`memvara encrypt --export-key` and keep the output somewhere other than this "
        "machine. If this key is lost, every store it encrypts is unreadable, and there "
        "is no way to recover it."), stacklevel=4)
    return StoreKey(key, "generated", str(path))


def export_key(env: Mapping[str, str] | None = None) -> StoreKey:
    """The key an existing encrypted store would be opened with, for a backup.

    Never generates one: exporting a key that encrypts nothing would give a false sense
    that something was backed up.
    """
    return resolve_key(create=False, env=env)


def require_sqlcipher() -> Any:
    """The SQLCipher DB-API module, or `EncryptionUnavailable` naming the extra."""
    try:
        # `type: ignore` because `sqlcipher3` ships no type information, and a checker
        # run without the extra cannot find it at all. The module is handled as `Any`.
        import sqlcipher3.dbapi2 as sqlcipher  # type: ignore[import-untyped,import-not-found,unused-ignore]
    except ImportError:
        raise EncryptionUnavailable(
            "An encrypted store needs SQLCipher, which comes with the encrypt extra: "
            f"{_INSTALL}.") from None
    return sqlcipher


def file_kind(path: str) -> str:
    """What is on disk at `path`, read from its first bytes.

    "new" is a missing or empty file, which SQLite treats as an empty database. "plain" is
    an unencrypted SQLite database. "other" is anything else, which for a memvara store
    means an encrypted one; whether it really is one is only known once a key opens it.
    "unreadable" is a path this process cannot read, such as a directory or a file without
    read permission; the store then connects as it always did and lets SQLite say what is
    wrong, rather than raising a different error from here.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(len(_SQLITE_MAGIC))
    except FileNotFoundError:
        return "new"
    except OSError:
        return "unreadable"
    if not head:
        return "new"
    return "plain" if head == _SQLITE_MAGIC else "other"


class VectorSealer:
    """AES-256-GCM for the rows of an encrypted store's vector file.

    The file is a 64-byte header followed by one fixed-size record per matrix row. A
    record is a 12-byte random nonce, the row's float32 values encrypted, and a 16-byte
    authentication tag. A record of all zero bytes is an empty row: one never written, or
    one erased.

    **What a record is bound to.** The associated data of every record is the file header
    (magic, format, width and salt), the row number and the id of the claim or episode
    the row belongs to. So a record copied to another row, a record left over from the
    row's previous owner, a record from another store, and a header edited to a different
    width or salt all fail authentication. A file cut short loses the records past the
    cut, and the store reports them missing. None of these loads silently.

    **The key.** The row key is derived from the store key with HKDF-SHA256, under the
    database's own 16-byte salt (`PRAGMA cipher_salt`, the first 16 bytes of the
    encrypted database file), and the header repeats that salt. The salt is not secret.
    Taking it from the database is what makes every process that opens the store agree
    on the vector file's key: two processes that find the header missing at the same
    moment both write the same header, where a salt each picked at random would leave
    one process sealing records under a key the header no longer describes. The salt is
    fixed for the life of the database file, and a converted store is a new file with a
    new salt, so its old vector file is recognised as foreign and rebuilt. HKDF with its
    own `info` string keeps this key apart from the keys SQLCipher derives, and random
    96-bit nonces stay inside the bound AES-GCM needs, about 2^32 records per key, for
    as long as any store will realistically be written.

    **What it does not do.** A record can be replaced by an older record for the same row
    and the same owner, and that passes authentication: a search would then use that
    item's older vector. Preventing it would need a counter stored outside the file.
    """

    MAGIC = b"MEMVASEA"
    FORMAT = 1
    NONCE = 12
    TAG = 16
    SALT = 16
    _INFO = b"memvara vector file v1"

    def __init__(self, key: bytes) -> None:
        self._master = key
        self._aead: Any = None
        self._bound = b""
        #: The database's salt. `SQLiteStore` sets it once the database file exists,
        #: because a new database has no salt until its first page is written.
        self.salt = b""

    def header(self, dim: int) -> bytes:
        """The header's meaningful 32 bytes: magic, format, width and the salt."""
        return self.MAGIC + struct.pack("<II", self.FORMAT, dim) + self.salt

    def matches(self, head: bytes, dim: int) -> bool:
        """Whether `head` is the header this store writes for vectors of width `dim`."""
        return head[:16 + self.SALT] == self.header(dim)

    def bind(self, dim: int) -> None:
        """Derive the row key, and the associated data every record carries."""
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF

        derived = HKDF(algorithm=hashes.SHA256(), length=32, salt=self.salt,
                       info=self._INFO).derive(self._master)
        self._aead = AESGCM(derived)
        self._bound = self.header(dim)

    def record_size(self, dim: int) -> int:
        return self.NONCE + dim * 4 + self.TAG

    def _associated(self, slot: int, item_id: str) -> bytes:
        return self._bound + struct.pack("<Q", slot) + item_id.encode("utf-8")

    def seal(self, slot: int, item_id: str, data: bytes) -> bytes:
        nonce = os.urandom(self.NONCE)
        return nonce + self._aead.encrypt(nonce, data, self._associated(slot, item_id))

    def open(self, slot: int, item_id: str, record: bytes) -> bytes | None:
        """The row's float32 bytes, or None if the record fails authentication."""
        from cryptography.exceptions import InvalidTag

        try:
            return bytes(self._aead.decrypt(record[:self.NONCE], record[self.NONCE:],
                                            self._associated(slot, item_id)))
        except InvalidTag:
            return None


@dataclass(frozen=True)
class EncryptResult:
    """What `encrypt_store` did."""

    path: str
    #: True when the store was already encrypted and nothing was changed.
    already: bool
    claims: int = 0
    episodes: int = 0
    vectors: int = 0
    #: Where the key came from, in words (`StoreKey.describe`).
    key_source: str = ""


#: The tables whose row counts must match between the original and the encrypted copy.
#: Fixed rather than read from the file, because a migration run on the copy may fill a
#: derived table the original left empty, and a count that differs for that reason is not
#: a failed conversion.
_COUNTED = ("claims", "episodes", "embeddings", "episode_embeddings", "documents")


def _counts(conn: Any, tables: set[str]) -> dict[str, int]:
    return {t: int(conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
            for t in _COUNTED if t in tables}


def _in_use(path: str) -> bool:
    """Whether another connection has the database open.

    SQLite removes the `-wal` and `-shm` files when the last connection closes. This
    runs after this module has opened and closed its own connection, so if either file
    is still there, somebody else is holding the store.
    """
    return os.path.exists(path + "-wal") or os.path.exists(path + "-shm")


def _remove(*paths: str) -> None:
    for p in paths:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass


def encrypt_store(path: str, *, key: StoreKey | None = None) -> EncryptResult:
    """Convert an unencrypted store to an encrypted one, in place, vectors included.

    The work happens on a copy. The database is exported into a new encrypted file in the
    same directory with `sqlcipher_export`, the copy is opened as a store (which writes
    its encrypted vector file from the vectors in the database), the copy is reopened and
    every vector is read back and authenticated, and the row counts are compared with the
    original. Only then is the copy renamed over the original, database first and vector
    file second. A failure at any step before the rename deletes the copy and leaves the
    original exactly as it was. A failure between the two renames leaves an encrypted
    database beside an old vector file, which the next open detects and rebuilds from
    the database.

    Refuses while another process has the store open, because a write that lands in the
    original after the export would be lost at the rename. Stop the MCP server first.
    """
    from .sqlite import SQLiteStore, _lock_path, _vec_path

    kind = file_kind(path)
    if kind == "new":
        raise EncryptionError(f"{path} does not exist or is empty; there is nothing to "
                              "encrypt.")
    sqlcipher = require_sqlcipher()
    if kind == "other":
        found = key or resolve_key(create=False)
        try:
            with SQLiteStore(path, key=found.key):
                pass
        except EncryptionError:
            raise EncryptionError(
                f"{path} is not an unencrypted SQLite database, and {found.describe()} "
                "does not open it as an encrypted store. Nothing was changed.") from None
        return EncryptResult(path, already=True, key_source=found.describe())

    # Open and close once before anything else. The close checkpoints the write-ahead
    # log into the main file and deletes it, which makes the file the export reads
    # complete, and leaves `_in_use` a meaningful question.
    plain = sqlite3.connect(path)
    try:
        tables = {r[0] for r in plain.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        version = int(plain.execute("PRAGMA user_version").fetchone()[0])
        before = _counts(plain, tables)
    finally:
        plain.close()
    if "claims" not in tables:
        raise EncryptionError(f"{path} is a SQLite database but not a memvara store (it "
                              "has no claims table). Nothing was changed.")
    if _in_use(path):
        raise EncryptionError(
            f"{path} is open in another process. Stop every process using this store "
            "(the MCP server, a script, a notebook) and run this again. Nothing was "
            "changed.")
    found = key or resolve_key(create=True)
    stamp = os.stat(path)

    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".encrypting-",
                               dir=os.path.dirname(os.path.abspath(path)))
    os.close(fd)
    leftovers = (tmp, tmp + "-wal", tmp + "-shm", tmp + "-journal", _vec_path(tmp) or "",
                 _lock_path(tmp) or "")
    try:
        conn = sqlcipher.connect(path)
        try:
            # The raw-key form, `x'<hex>'`, skips SQLCipher's password stretching: the
            # key is already 32 random bytes. It is bound as a parameter so it never
            # appears in the text of a statement.
            conn.execute("ATTACH DATABASE ? AS encrypted KEY ?",
                         (tmp, f"x'{found.key.hex()}'"))
            conn.execute("SELECT sqlcipher_export('encrypted')")
            # `sqlcipher_export` copies every table and index but not the header field
            # the schema version lives in, and a store stamped 0 would be migrated from
            # scratch on its next open.
            conn.execute(f"PRAGMA encrypted.user_version = {version}")
            conn.execute("DETACH DATABASE encrypted")
        finally:
            conn.close()
        # The first open writes the encrypted vector file from the database's vectors.
        with SQLiteStore(tmp, key=found.key):
            pass
        # The second reads every record back through authentication, so a vector file
        # that would not open is found here rather than by the first search after the
        # rename.
        with SQLiteStore(tmp, key=found.key) as copy:
            copy._ensure_index()
            loaded = len(copy._vec._row)
            after = _counts(copy._db, tables)
        if after != before:
            raise EncryptionError(
                f"The encrypted copy of {path} does not hold the same rows as the "
                f"original ({after} against {before}). Nothing was changed.")
        vectors = before.get("embeddings", 0) + before.get("episode_embeddings", 0)
        if loaded != vectors:
            raise EncryptionError(
                f"The encrypted copy of {path} read back {loaded} of {vectors} vectors. "
                "Nothing was changed.")
        now = os.stat(path)
        if _in_use(path) or (now.st_size, now.st_mtime_ns) != (stamp.st_size,
                                                               stamp.st_mtime_ns):
            raise EncryptionError(
                f"{path} was opened or changed by another process while it was being "
                "encrypted. Stop every process using this store and run this again. "
                "Nothing was changed.")
        os.replace(tmp, path)
    except BaseException:
        _remove(*leftovers)
        raise
    # SQLite removes these when the copy's last connection closes. If a platform left
    # one behind, it would never be read again, because a write-ahead log is found by
    # the database's name and the database now has another one. The copy's lock file
    # is found the same way, and the store keeps its own beside the original.
    _remove(tmp + "-wal", tmp + "-shm", _lock_path(tmp) or "")
    # The database is encrypted from here on. If this rename fails, the store still
    # opens: the old vector file's header is not one an encrypted store writes, so the
    # next open rebuilds it from the database.
    target = _vec_path(path)
    source = _vec_path(tmp)
    assert target is not None and source is not None
    if os.path.exists(source):
        os.replace(source, target)
    else:
        _remove(target)
    return EncryptResult(path, already=False, claims=before.get("claims", 0),
                         episodes=before.get("episodes", 0), vectors=vectors,
                         key_source=found.describe())
