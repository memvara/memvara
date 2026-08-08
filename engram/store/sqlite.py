"""Default store: a single SQLite file, plus a memory-mapped vector matrix beside it.

Design notes:

* Time is persisted as epoch floats, not ISO strings. Both time axes get compared and
  range-scanned constantly, and float comparison is unambiguous across timezones and
  formats.
* Lexical search is FTS5/BM25 joined against the claims table, so scope and liveness
  filtering happen inside the query rather than by over-fetching and filtering after.
* Vector search is a numpy matmul over a rowid-addressed float32 matrix, restricted to
  the candidate rows for the scope. Exact, not approximate: at the scale a local memory
  store actually operates at, ANN buys nothing and costs recall.
* Episodes get both of those indexes too, on the same terms. They used to get neither,
  which meant a stored turn was reachable only through `why()` on a claim that happened
  to be extracted from it — so every turn the extractor declined (a decision, a reason,
  a constraint stated conditionally: most of a real transcript) was retained and then
  permanently unfindable. Their vectors share the matrix with the claims': one mapped
  file, one slot space, one dimension, because the same embedder writes both and a
  second matrix would double the coherence bookkeeping to no end.
* That matrix lives in a mapped file (`<db>.vecs`), not on the heap. SQLite owns the
  vectors; the file is a derived, rebuildable view of them addressed by an integer
  slot. Opening therefore costs a header read instead of deserializing every blob, the
  pages are shared between processes rather than copied per worker, and growth is a
  file extension instead of an `np.vstack` that transiently holds four times the old
  matrix.

The whole thing runs in WAL mode, with **one connection per writer and one per reading
thread**. That second half is what makes the first half true. WAL lets any number of
readers run against a consistent snapshot while a writer holds the write lock — but only
across *different connections*, because a connection is where SQLite's transaction state
lives. Behind a single connection and a single mutex, which is what this was, a reader
waited out the whole consolidation sweep. Measured, one reader thread against a
20k-claim pass:

    reads completed   1,470  ->  13,728      p95   3.44 ms  ->  0.31 ms
    mean               1.48 ms  ->  0.20 ms  p99  40.4  ms  ->  2.08 ms

An idle read is unchanged at 13 us and a sweep with nobody reading is unchanged at
1.81 s, so none of that was bought from the write path; the sweep does take ~25% longer
*while* a reader is running, because the reader is now doing nine times as much work
next to it instead of waiting. The cost is one file handle per reading thread.

Reads taken that way see the snapshot as of their own statement, so a reader observes a
sweep's windows as they commit rather than waiting for the end of one. The exception is
a thread inside `batch()`, which must see its own uncommitted rows — `competing_claims`
during a write is what makes contradiction detection exact — so it stays on the writer's
connection. See `_read`.
"""

from __future__ import annotations

import json
import os
import sqlite3
import struct
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Iterable, Iterator, Sequence

import numpy as np

from ..types import Claim, Derivation, Episode, MemoryType, Scope

if TYPE_CHECKING:  # pragma: no cover
    from ..schema import PredicateSpec

# Bump when the schema changes in a way that needs a migration. `CREATE TABLE IF NOT
# EXISTS` silently does nothing on an existing file, so without a stamped version an
# upgrade that adds a column deploys green, passes health checks, and then fails every
# write with "no such column" — forever, until someone rolls back.
#
# 2: predicate specs became tenant-scoped, and embeddings gained `slot` (their row in
#    the mapped matrix) and `seq` (a monotonic write counter another process reads to
#    find what it has not seen).
# 3: episodes became retrievable — an FTS index over their text and a vector table
#    sharing the claims' matrix. The DDL below is `IF NOT EXISTS`, so it does appear on
#    an old file; what does not appear is the *content* of an index for episodes that
#    were written before it existed, and backfilling that is what the stamp gates.
# 4: resolved entities became durable (`entities`). Nothing to backfill — no earlier
#    version wrote an entity anywhere — so here the DDL genuinely is the whole
#    migration. The stamp still earns its place: it is what tells version 5 whether the
#    table it is looking at was built by 4 or invented on the spot by its own
#    `IF NOT EXISTS`, which is the distinction every backfill after this one rests on.
SCHEMA_VERSION = 4

# Kept separate because the v1 -> v2 migration has to recreate this table: SQLite
# cannot add a column to an existing primary key, and (tenant, name) is now the key.
_PREDICATES_DDL = """
CREATE TABLE IF NOT EXISTS predicates (
    tenant      TEXT NOT NULL,
    name        TEXT NOT NULL,
    cardinality TEXT NOT NULL,
    volatility  TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    aliases     TEXT NOT NULL DEFAULT '[]',
    supersedes  TEXT NOT NULL DEFAULT '[]',
    learned     INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (tenant, name)
);
"""

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS episodes (
    id       TEXT PRIMARY KEY,
    tenant   TEXT NOT NULL,
    usr      TEXT,
    agent    TEXT,
    session  TEXT,
    role     TEXT NOT NULL,
    content  TEXT NOT NULL,
    ts       REAL NOT NULL,
    hash     TEXT NOT NULL,
    meta     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS ep_hash  ON episodes(tenant, hash);
-- `ts` last, so the same index serves both the scope filter and the as-of range scan
-- that bounds an episode search to turns that had already happened.
CREATE INDEX IF NOT EXISTS ep_scope ON episodes(tenant, usr, agent, session, ts);

CREATE TABLE IF NOT EXISTS claims (
    id             TEXT PRIMARY KEY,
    tenant         TEXT NOT NULL,
    usr            TEXT,
    agent          TEXT,
    session        TEXT,
    subject        TEXT NOT NULL,
    predicate      TEXT NOT NULL,
    object         TEXT NOT NULL,
    text           TEXT NOT NULL,
    polarity       INTEGER NOT NULL DEFAULT 1,
    memory_type    TEXT NOT NULL,
    valid_from     REAL NOT NULL,
    valid_to       REAL,
    recorded_at    REAL NOT NULL,
    invalidated_at REAL,
    invalidated_by TEXT,
    confidence     REAL NOT NULL DEFAULT 1.0,
    salience       REAL NOT NULL DEFAULT 1.0,
    obs_count      INTEGER NOT NULL DEFAULT 1,
    sources        TEXT NOT NULL DEFAULT '[]',
    derivation     TEXT NOT NULL,
    extractor      TEXT NOT NULL DEFAULT '',
    meta           TEXT NOT NULL DEFAULT '{}',
    fact_key       TEXT NOT NULL,
    value_key      TEXT NOT NULL
);
-- The index that makes contradiction detection O(1) instead of a similarity search.
CREATE INDEX IF NOT EXISTS cl_fact  ON claims(tenant, fact_key, invalidated_at);
CREATE INDEX IF NOT EXISTS cl_value ON claims(tenant, value_key);
CREATE INDEX IF NOT EXISTS cl_scope ON claims(tenant, usr, agent, session);
CREATE INDEX IF NOT EXISTS cl_live  ON claims(tenant, invalidated_at, recorded_at);

CREATE TABLE IF NOT EXISTS embeddings (
    claim_id TEXT PRIMARY KEY,
    dim      INTEGER NOT NULL,
    slot     INTEGER,
    seq      INTEGER NOT NULL DEFAULT 0,
    vec      BLOB NOT NULL
);

-- Episode vectors. A separate table because the two are separate objects with separate
-- lifecycles, but addressing *one* matrix: `slot` is drawn from the same allocator and
-- freed into the same `vec_free`, so a row belongs to exactly one of the two tables and
-- the mapped file needs neither a second header nor a second dimension.
CREATE TABLE IF NOT EXISTS episode_embeddings (
    episode_id TEXT PRIMARY KEY,
    dim        INTEGER NOT NULL,
    slot       INTEGER,
    seq        INTEGER NOT NULL DEFAULT 0,
    vec        BLOB NOT NULL
);

-- Slots freed by purge(), so a store that churns reuses rows instead of growing its
-- matrix forever. Popped inside the same write transaction that consumes the slot,
-- which is what keeps two processes from handing the same row to two claims.
CREATE TABLE IF NOT EXISTS vec_free (slot INTEGER PRIMARY KEY);

CREATE VIRTUAL TABLE IF NOT EXISTS claims_fts
    USING fts5(claim_id UNINDEXED, text, tokenize='porter unicode61');

-- The episode text index. Its absence was the larger half of "stored and never found
-- again": a turn the extractor declined still holds the reason, the constraint and the
-- decision, and BM25 over raw turns is the only thing that can return those verbatim.
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts
    USING fts5(episode_id UNINDEXED, content, tokenize='porter unicode61');

-- Resolved entities: which surface strings name the same thing. Durable and
-- tenant-scoped for exactly the reason the predicate table is — one tenant deciding
-- that "Acme" and "Acme Corp" are one entity must not decide it for another — and with
-- the same failure mode if it were not persisted: a fresh process would re-derive the
-- mapping, disagree with the ids already baked into stored `fact_key`s, and silently
-- stop recognising a contradiction between two spellings of one subject.
CREATE TABLE IF NOT EXISTS entities (
    tenant    TEXT NOT NULL,
    id        TEXT NOT NULL,
    canonical TEXT NOT NULL,
    aliases   TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (tenant, id)
);

-- Learned predicate schema. This has to be durable: cardinality is what makes a
-- contradiction detectable, so a registry that evaporates on restart means a fresh
-- process treats every learned predicate as multi-valued until it re-pays the
-- classification — and silently stops retiring superseded facts in the meantime.
""" + _PREDICATES_DDL

# Created after the migration, not with the tables: on a v1 file the columns these
# cover do not exist until `_migrate` has added them, and `executescript` would abort.
#
# Every one of them exists to keep a hot path off the blob pages. A row here is ~3 KB,
# so any query that touches the table proper reads a page per row; the same query
# answered from a covering index reads a few hundred pages for the whole table.
_LATE_INDEXES = """
CREATE UNIQUE INDEX IF NOT EXISTS emb_slot ON embeddings(slot);
CREATE INDEX IF NOT EXISTS emb_seq  ON embeddings(seq, slot, claim_id);
CREATE INDEX IF NOT EXISTS emb_dim  ON embeddings(dim, slot);

CREATE UNIQUE INDEX IF NOT EXISTS epemb_slot ON episode_embeddings(slot);
CREATE INDEX IF NOT EXISTS epemb_seq ON episode_embeddings(seq, slot, episode_id);
CREATE INDEX IF NOT EXISTS epemb_dim ON episode_embeddings(dim, slot);
"""

_CLAIM_FIELDS = (
    "id", "tenant", "usr", "agent", "session", "subject", "predicate", "object", "text",
    "polarity", "memory_type", "valid_from", "valid_to", "recorded_at", "invalidated_at",
    "invalidated_by", "confidence", "salience", "obs_count", "sources", "derivation",
    "extractor", "meta", "fact_key", "value_key",
)
_CLAIM_COLS = ", ".join(_CLAIM_FIELDS)
_CLAIM_VALUES = ", ".join("?" * len(_CLAIM_FIELDS))
# Upsert rather than INSERT OR REPLACE. REPLACE deletes the row and re-inserts it, which
# assigns a *new* rowid — and the FTS row is keyed on that rowid, so every update would
# orphan its index entry. ON CONFLICT ... DO UPDATE preserves the rowid.
_CLAIM_UPSERT = (
    f"INSERT INTO claims ({_CLAIM_COLS}) VALUES ({_CLAIM_VALUES}) "
    "ON CONFLICT(id) DO UPDATE SET "
    + ", ".join(f"{f}=excluded.{f}" for f in _CLAIM_FIELDS if f != "id")
)
_EPISODE_FIELDS = (
    "id", "tenant", "usr", "agent", "session", "role", "content", "ts", "hash", "meta",
)
# Upsert, not INSERT OR REPLACE, for the same reason `put_claim` uses one: REPLACE
# deletes and re-inserts, which assigns a new rowid, and the FTS row is keyed on the old
# one. Re-adding a turn would leave its text in the index under a rowid nothing points
# at — searchable, unhydratable, and undeletable by purge.
_EPISODE_UPSERT = (
    f"INSERT INTO episodes ({', '.join(_EPISODE_FIELDS)}) "
    f"VALUES ({', '.join('?' * len(_EPISODE_FIELDS))}) "
    "ON CONFLICT(id) DO UPDATE SET "
    + ", ".join(f"{f}=excluded.{f}" for f in _EPISODE_FIELDS if f != "id")
)

# SQLite's parameter limit is 999 on older builds; chunk bulk lookups below it.
_MAX_SQL_PARAMS = 900

_VEC_TABLE_NAMES = ("embeddings", "episode_embeddings")

# The next row of the matrix to hand out. Both vector tables address one matrix, so the
# allocator has to see both: computing "one past my own maximum" per table would give
# the first episode and the first claim the same row, and each would read back the
# other's vector.
_NEXT_SLOT = (
    "COALESCE((SELECT MIN(slot) FROM vec_free), (SELECT MAX(top) + 1 FROM ("
    + " UNION ALL ".join(f"SELECT COALESCE(MAX(slot), -1) AS top FROM {n}"
                         for n in _VEC_TABLE_NAMES)
    + ")))"
)


def _vector_upsert(table: str, key: str) -> str:
    """One statement, so slot allocation happens inside the write transaction and cannot
    race another process. The slot subquery is evaluated even when the row already
    exists, which is harmless: nothing consumes it, and the conflict branch deliberately
    leaves `slot` alone so a re-embedded item keeps its row.
    """
    return (
        f"INSERT INTO {table} ({key}, dim, slot, seq, vec) "
        f"VALUES (?, ?, {_NEXT_SLOT}, "
        f"(SELECT COALESCE(MAX(seq), 0) + 1 FROM {table}), ?) "
        f"ON CONFLICT({key}) DO UPDATE SET "
        "dim = excluded.dim, seq = excluded.seq, vec = excluded.vec "
        "RETURNING slot"
    )


@dataclass(frozen=True, slots=True)
class _VecTable:
    """Where one kind of vector is persisted, and under what key.

    Claims and episodes are separate objects with separate lifecycles, so they get
    separate tables — but every operation over them is identical, and writing it twice
    is how the two silently diverge on the one thing that must not: which of them owns
    a given row of the shared matrix.
    """

    name: str
    key: str
    upsert: str


#: Rows per page in `_iter_rows`. Large enough that a full scan is not dominated by
#: round trips, small enough that the page is not the thing that runs the process out of
#: memory — which is the failure the paging exists to prevent.
_ITER_PAGE = 1000

_CLAIM_VECS = _VecTable("embeddings", "claim_id",
                        _vector_upsert("embeddings", "claim_id"))
_EPISODE_VECS = _VecTable("episode_embeddings", "episode_id",
                          _vector_upsert("episode_embeddings", "episode_id"))
_VEC_TABLES = (_CLAIM_VECS, _EPISODE_VECS)

# Everything the index needs to know about what is on disk, in one query per open:
# how many vectors, whether they agree on a width, the highest row in use, and how many
# rows have no address yet. `SUM(n)`, not `COUNT(*)` — the outer query counts the two
# subquery rows, not the vectors they summarize.
_VEC_CENSUS = (
    "SELECT SUM(n) AS n, MIN(lo) AS lo, MAX(hi) AS hi, MAX(top) AS top, "
    "SUM(loose) AS loose FROM ("
    + " UNION ALL ".join(
        f"SELECT COUNT(*) AS n, MIN(dim) AS lo, MAX(dim) AS hi, MAX(slot) AS top, "
        f"COUNT(*) - COUNT(slot) AS loose FROM {n}" for n in _VEC_TABLE_NAMES)
    + ")"
)

_VEC_MAGIC = b"ENGRMVEC"
_VEC_FORMAT = 1
# 64 bytes of header keeps row 0 cache-line aligned, which matters because every query
# runs a BLAS product straight off these pages.
_VEC_HEADER = 64


def _ts(dt: datetime | None) -> float | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _dt(v: float | None) -> datetime | None:
    if v is None:
        return None
    return datetime.fromtimestamp(v, tz=timezone.utc)


def _unit(vec: np.ndarray) -> np.ndarray:
    """A flat float32 copy scaled to unit length; a zero vector stays zero.

    Normalizing in exactly one place is what lets the persisted bytes and the matrix
    row hold the same thing, so a cosine never depends on which one it came from.
    """
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0.0 else v


def _vec_path(db_path: str) -> str | None:
    """Where the matrix lives for a given database, or None when there is nowhere.

    An in-memory database has no file to sit beside, and nothing outlives the process
    anyway, so it keeps the matrix on the heap.
    """
    return None if db_path in (":memory:", "") else db_path + ".vecs"


def _fts_query(raw: str) -> str:
    """Turn arbitrary user text into a safe FTS5 MATCH expression.

    FTS5's query language treats plenty of punctuation as syntax, so an unescaped user
    query is both a crash and an injection surface. Reduce to bare alphanumeric tokens
    and OR them; ranking, not filtering, is what BM25 is here for.
    """
    toks = []
    cur = []
    for ch in raw.lower():
        if ch.isalnum():
            cur.append(ch)
        elif cur:
            toks.append("".join(cur))
            cur = []
    if cur:
        toks.append("".join(cur))
    toks = [t for t in toks if len(t) > 1]
    if not toks:
        return ""
    return " OR ".join(f'"{t}"' for t in toks)


class _VecIndex:
    """Exact cosine index over a slot-addressed float32 matrix.

    The matrix is a memory-mapped file whenever there is a path to map. Three
    properties follow, and they are the reason for the whole arrangement:

    * Opening maps a file instead of materializing one. A store larger than RAM can
      still be opened to be repaired.
    * The pages are shared, so N worker processes on a host hold one copy of the matrix
      between them rather than one each, and a vector written by one is visible to the
      others without a re-read.
    * Growth extends the file rather than copying it. Doubling a 100k x 768 matrix with
      `np.vstack` transiently holds four times the old matrix and stalls whichever user
      write happens to cross the boundary — measured at 562 ms here.

    Slots are assigned by whoever owns durability. The store allocates them inside its
    write transaction (so two processes cannot claim one row) and calls `put`/`map`;
    a bare index, with no store behind it, allocates its own via `add`.

    The keys are opaque strings, and the index does not care what kind of thing they
    name. Claims and episodes share it: their ids are disjoint by construction, they
    are written by the same embedder and therefore have one width, and a second matrix
    would mean a second file, a second header and a second staleness question for no
    benefit at all.
    """

    _INITIAL_ROWS = 256
    # Rows gathered per pass when the candidate set is sparse. 4096 x 768 float32 is a
    # 12 MB buffer that gets reused; fancy-indexing the candidate set instead allocated
    # and discarded 307 MB per query at 100k rows.
    _CHUNK = 4096
    # Above this share of the matrix, one contiguous product over everything and a
    # gather of the scores beats gathering rows first — the gather is the expensive
    # half, and it is the half that scales with `dim`.
    _DENSE_SHARE = 0.35

    def __init__(self, dim: int | None = None, path: str | None = None,
                 count: "Callable[[], int] | None" = None) -> None:
        self.dim = dim
        self.path = path
        # How many vectors exist, asked of whoever owns durability. Without it an index
        # that has not yet needed its map would report itself empty, and "empty" is a
        # different claim from "not looked at".
        self._count = count
        self._row: dict[str, int] = {}
        self._free: list[int] = []
        self._mat: np.ndarray | None = None
        self._fh = None
        self._rows = 0          # capacity
        self._high = 0          # one past the highest slot ever mapped
        # Its own lock, always taken inside the store's: growth swaps the mapping out
        # from under anything reading it.
        self._lock = threading.RLock()

    # -- backing store -------------------------------------------------------

    def attach(self, dim: int, rows: int) -> bool:
        """Map at least `rows` rows of `dim` floats, creating the file if absent.

        Returns False when the mapping came up empty and the caller has to repopulate
        it from durable storage: no file, a file written by a different embedder, or
        one truncated by something outside this process. The matrix is derived data,
        so rebuilding it is always available as the answer.
        """
        with self._lock:
            self.dim = dim
            if self.path is None:
                self._ensure_rows(max(rows, self._INITIAL_ROWS))
                return False
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
            self._fh = os.fdopen(fd, "r+b")
            size = os.fstat(fd).st_size
            head = os.pread(fd, _VEC_HEADER, 0) if size >= _VEC_HEADER else b""
            usable = (head[:8] == _VEC_MAGIC
                      and struct.unpack_from("<II", head, 8) == (_VEC_FORMAT, dim))
            if not usable:
                os.ftruncate(fd, 0)
                os.pwrite(fd, _VEC_MAGIC + struct.pack("<II", _VEC_FORMAT, dim), 0)
                self._rows = 0
            else:
                self._rows = (size - _VEC_HEADER) // (dim * 4)
            self._ensure_rows(max(rows, self._INITIAL_ROWS))
            return usable

    def _ensure_rows(self, need: int) -> None:
        if self._mat is not None and need <= self._rows:
            return
        assert self.dim is not None
        if self.path is None:
            target = max(need, self._rows * 2, self._INITIAL_ROWS)
            grown = np.zeros((target, self.dim), dtype=np.float32)
            if self._mat is not None:
                grown[:self._rows] = self._mat
            self._mat = grown
            self._rows = target
            return
        # Extend only when a row is actually missing. Sizing from what the file already
        # holds instead would double it on *every* open, since arriving here with no
        # mapping is the normal way a store starts — and a sparse file that doubles
        # forever eventually exceeds what can be mapped, which bricks the store this
        # was meant to keep openable.
        if need > self._rows:
            target = max(need, self._rows * 2, self._INITIAL_ROWS)
            fd = self._fh.fileno()
            size = _VEC_HEADER + target * self.dim * 4
            if size > os.fstat(fd).st_size:
                # A write past EOF extends the file; `ftruncate` would also shorten it,
                # and two processes growing at once would race to cut each other's rows.
                os.pwrite(fd, b"\0", size - 1)
        self._remap()

    def _remap(self) -> None:
        # Drop the old mapping first: holding both doubles the address space, and the
        # point of the exercise is that growth costs no memory.
        self._mat = None
        size = os.fstat(self._fh.fileno()).st_size
        self._rows = (size - _VEC_HEADER) // (self.dim * 4)
        self._mat = np.memmap(self._fh, dtype=np.float32, mode="r+",
                              offset=_VEC_HEADER, shape=(self._rows, self.dim))

    # -- mutation ------------------------------------------------------------

    def put(self, item_id: str, slot: int, vec: np.ndarray) -> None:
        """Write a unit vector into `slot` and point `item_id` at it."""
        with self._lock:
            if self.dim is None:
                self.attach(int(np.asarray(vec).reshape(-1).shape[0]), slot + 1)
            self._ensure_rows(slot + 1)
            self._mat[slot] = vec
            self._row[item_id] = slot
            self._high = max(self._high, slot + 1)

    def map(self, item_id: str, slot: int) -> None:
        """Point `item_id` at a row that already holds its vector.

        This is the whole cost of learning about another process's writes: the bytes
        arrived through the shared mapping, only the name-to-row map is process-local.
        """
        with self._lock:
            if self.dim is not None:
                self._ensure_rows(slot + 1)
            self._row[item_id] = slot
            self._high = max(self._high, slot + 1)

    def add(self, item_id: str, vec: np.ndarray) -> None:
        """Store a vector, allocating its slot. For an index with no store behind it."""
        v = _unit(vec)
        with self._lock:
            if self.dim is None:
                self.attach(int(v.shape[0]), self._INITIAL_ROWS)
            if v.shape[0] != self.dim:
                raise ValueError(
                    f"embedding dim {v.shape[0]} != index dim {self.dim}; "
                    "the store was built with a different embedder"
                )
            slot = self._row.get(item_id)
            if slot is None:
                slot = self._free.pop() if self._free else self._high
            self.put(item_id, slot, v)

    def forget(self, item_id: str) -> int | None:
        """Unmap an item and blank its row, returning the slot it held.

        Zeroing matters for erasure: purged text stays reconstructible from
        its embedding, and the file outlives the process. The slot is handed back to
        the caller rather than reused here, because when a store is present the free
        list has to be durable and shared.
        """
        with self._lock:
            slot = self._row.pop(item_id, None)
            if slot is not None and self._mat is not None and slot < self._rows:
                self._mat[slot] = 0.0
            return slot

    def reset(self) -> None:
        """Forget every vector and release the width they shared.

        Re-embedding is the caller. The index fixes its dimension on the first vector
        it sees and rejects every later one of a different width, so a migration to a
        new model cannot begin until that width is released. The file goes with it:
        "drop every vector" that leaves the bytes behind is not what it says.
        """
        with self._lock:
            self.dim = None
            self._row.clear()
            self._free.clear()
            self._rows = 0
            self._high = 0
            # Unmap before truncating. A mapping that outlives the pages behind it
            # faults on the next read rather than reporting a short file.
            self._mat = None
            if self._fh is not None:
                os.ftruncate(self._fh.fileno(), 0)
                self._fh.close()
                self._fh = None

    def remove(self, item_id: str) -> bool:
        """Drop a vector and keep its slot for reuse. Required for erasure — without
        it, purged text stays reconstructible from the embedding.

        Search resolves through the name-to-row map, so an unmapped id is unreachable
        even before the row is blanked.
        """
        slot = self.forget(item_id)
        if slot is None:
            return False
        self._free.append(slot)
        return True

    # -- query ---------------------------------------------------------------

    def _scores(self, rows: np.ndarray, qq: np.ndarray) -> np.ndarray:
        """Cosines for `rows` against `qq`, without materializing those rows.

        `self._mat[rows] @ qq` is the obvious spelling and the expensive one: advanced
        indexing copies, so a broad query at 100k x 768 allocated and threw away
        307 MB, and two thirds of the query was not the product.
        """
        if rows.shape[0] >= self._high * self._DENSE_SHARE:
            # Most of the matrix is in play: one pass over contiguous memory allocates
            # one float per row instead of one row per candidate.
            return (self._mat[:self._high] @ qq)[rows]
        out = np.empty(rows.shape[0], dtype=np.float32)
        buf = np.empty((min(self._CHUNK, rows.shape[0]), self.dim), dtype=np.float32)
        for i in range(0, rows.shape[0], self._CHUNK):
            part = rows[i:i + self._CHUNK]
            block = buf[:part.shape[0]]
            np.take(self._mat, part, axis=0, out=block)
            np.dot(block, qq, out=out[i:i + part.shape[0]])
        return out

    def search(self, q: np.ndarray, allowed: Sequence[str],
               limit: int) -> list[tuple[str, float]]:
        with self._lock:
            if self._mat is None or not self._row or limit <= 0:
                return []
            cids: list[str] = []
            rows: list[int] = []
            for c in allowed:
                r = self._row.get(c)
                if r is not None:
                    cids.append(c)
                    rows.append(r)
            if not rows:
                return []
            qq = _unit(q)
            if qq.shape[0] != self.dim:
                raise ValueError(f"query dim {qq.shape[0]} != index dim {self.dim}")
            scores = self._scores(np.asarray(rows, dtype=np.int64), qq)
        k = min(limit, scores.shape[0])
        part = (np.argpartition(-scores, k - 1)[:k]
                if k < scores.shape[0] else np.arange(scores.shape[0]))
        order = part[np.argsort(-scores[part])]
        return [(cids[int(i)], float(scores[int(i)])) for i in order]

    def get(self, item_id: str) -> np.ndarray | None:
        """The stored unit vector, or None if this item was never embedded."""
        with self._lock:
            row = self._row.get(item_id)
            if row is None or self._mat is None:
                return None
            return np.array(self._mat[row])

    def close(self) -> None:
        with self._lock:
            if self._mat is not None and self.path is not None:
                self._mat.flush()
            self._mat = None
            if self._fh is not None:
                self._fh.close()
                self._fh = None

    def __len__(self) -> int:
        # Live entries, not the high-water mark: removed rows are still allocated. With
        # a store behind it, the store's count — the name-to-row map is a cache built
        # on demand, and its size is a fact about this process, not about the data.
        return len(self._row) if self._count is None else self._count()


class SQLiteStore:
    """Reference `Store` implementation. Single file, no server, no Docker."""

    def __init__(self, path: str = ":memory:") -> None:
        self.path = path
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        # Per-thread: the snapshot connection this thread reads through, how deep it is
        # inside `batch()`, and the last `data_version` it saw. Each of those three is a
        # property of a *connection* rather than of the store, so keeping them store-wide
        # is how one reader's bookkeeping corrupts another's. See `_read`.
        self._local = threading.local()
        # Store-wide, because closing is: every connection handed out, so `close()` can
        # take them all back.
        self._readers: list[sqlite3.Connection] = []
        # Its own lock, and the innermost one: opening a connection must not queue behind
        # a consolidation sweep, which is precisely the wait this exists to remove.
        self._readers_lock = threading.Lock()
        self._closed = False
        # Rows of the matrix this transaction touched, as (table, id), so a rollback can
        # put the index back in step with the database.
        self._touched: set[tuple[_VecTable, str]] = set()
        self._cleared = False
        self._vec = _VecIndex(path=_vec_path(path), count=self._count_vectors)
        self._index_loaded = False
        # One write watermark per vector table: `_read_map` folds in everything written
        # past it, and the two tables count independently.
        self._seq = {t.name: -1 for t in _VEC_TABLES}
        # Guards the name-to-row map's bookkeeping. Always taken *inside* `_lock` when
        # both are wanted, never the other way round: a reader holds only this one, so
        # that ordering is what keeps a refreshing reader and a committing writer from
        # deadlocking on each other.
        self._index_lock = threading.RLock()
        self._data_version = -1
        with self._lock:
            self._db.executescript(SCHEMA)
            self._migrate()
            self._db.executescript(_LATE_INDEXES)
            self._db.commit()
            self._attach_vectors()

    # -- connections ---------------------------------------------------------

    @property
    def _batch_depth(self) -> int:
        """How deep the *calling thread* is inside `batch()`.

        Per thread rather than per store because it now routes reads as well as gating
        commits: a thread mid-transaction must read its own uncommitted rows, and no
        other connection can see them. It is the same number the commit gate always
        used — `batch()` holds the write lock for its whole body, so while a batch is
        open no other thread can be writing.
        """
        return getattr(self._local, "depth", 0)

    @_batch_depth.setter
    def _batch_depth(self, value: int) -> None:
        self._local.depth = value

    @property
    def _data_version(self) -> int:
        """The `PRAGMA data_version` this *thread* last saw.

        Per thread for the same reason the connection is: the pragma counts commits made
        by *other* connections, so its value means nothing across two of them. Compared
        against a number another thread's connection produced, the check either fires
        forever or never fires again — and "never again" is the silent one, because BM25
        still finds the claim and fusion merely ranks it worse.

        The name-to-row map it guards stays shared, and that is not an inconsistency:
        whichever thread notices a commit first folds the new rows in for everybody, and
        a thread arriving later re-reads a tail that is already empty.
        """
        return getattr(self._local, "version", -1)

    @_data_version.setter
    def _data_version(self, value: int) -> None:
        self._local.version = value

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        """A connection to run one read-only statement against.

        The thread's own connection when there is one, and then no lock is taken at all:
        that is the entire point, and it is what lets a `search()` complete while a
        consolidation sweep holds the write lock. Otherwise the writer's connection,
        under the write lock, exactly as every read here used to work.

        Two cases fall back, and both are correctness rather than caution:

        * **Inside `batch()`.** A reader on another connection sees the last commit, not
          the open transaction — so the reconciler asking `competing_claims` mid-write
          would miss the claim written two statements ago and let a contradiction
          through. That is the one bug this whole mechanism could plausibly introduce,
          so the thread that opened the transaction never leaves it.
        * **A database with no file.** `:memory:` (and `""`, a private temporary file)
          is scoped to its connection: a second one is a second, empty database, not a
          second view of this one.
        """
        conn = self._reader()
        if conn is None:
            with self._lock:
                yield self._db
        else:
            yield conn

    def _reader(self) -> sqlite3.Connection | None:
        """This thread's snapshot connection, opening it on first use, or None."""
        if self._batch_depth or self.path in (":memory:", ""):
            return None
        conn = getattr(self._local, "db", None)
        if conn is None:
            conn = sqlite3.connect(self.path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            # `_readers_lock`, emphatically not `_lock`: a thread taking its very first
            # read while a sweep holds the write lock would otherwise wait out the sweep
            # to open the connection that exists so it does not have to.
            with self._readers_lock:
                if self._closed:
                    # Racing a `close()`. Hand back the writer's connection instead, so
                    # the caller gets sqlite's "closed database" error rather than a
                    # working read against a store that is supposed to be shut.
                    conn.close()
                    return None
                # Tracked so `close()` can close them: a connection left open holds a
                # WAL read mark, which stops checkpointing and grows the -wal file
                # without bound.
                self._readers.append(conn)
            self._local.db = conn
        return conn

    def _migrate(self) -> None:
        """Stamp or upgrade the schema version.

        Refuses to open a file written by a newer Engram rather than corrupting it: a
        rollback that silently half-works is worse than one that refuses to start.
        """
        found = int(self._db.execute("PRAGMA user_version").fetchone()[0])
        if found > SCHEMA_VERSION:
            raise RuntimeError(
                f"{self.path}: schema version {found} was written by a newer Engram "
                f"(this build understands {SCHEMA_VERSION}). Upgrade rather than "
                "downgrade — opening it here could write rows the newer build cannot read."
            )
        if found < SCHEMA_VERSION:
            self._migrate_to_v2()
            self._migrate_to_v3()
            self._db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def _migrate_to_v2(self) -> None:
        """Tenant-scope the predicate table; make embeddings addressable.

        Driven by the shape of the file rather than by the version stamp, because a
        brand-new database also arrives here with version 0 and already has the v2
        tables from `SCHEMA`. Everything below is a no-op in that case.
        """
        pred = {r["name"] for r in self._db.execute("PRAGMA table_info(predicates)")}
        if "tenant" not in pred:
            # The table was global, so one tenant's classification silently set another
            # tenant's contradiction behaviour and decay half-life. Existing rows
            # predate multi-tenancy and belong to the tenant `Engram` uses when the
            # caller names none.
            self._db.execute("ALTER TABLE predicates RENAME TO predicates_v1")
            self._db.executescript(_PREDICATES_DDL)
            self._db.execute(
                "INSERT INTO predicates "
                "(tenant, name, cardinality, volatility, memory_type, aliases, "
                " supersedes, learned) "
                "SELECT 'default', name, cardinality, volatility, memory_type, aliases, "
                "supersedes, learned FROM predicates_v1"
            )
            self._db.execute("DROP TABLE predicates_v1")
        emb = {r["name"] for r in self._db.execute("PRAGMA table_info(embeddings)")}
        if "slot" not in emb:
            self._db.execute("ALTER TABLE embeddings ADD COLUMN slot INTEGER")
            self._db.execute(
                "ALTER TABLE embeddings ADD COLUMN seq INTEGER NOT NULL DEFAULT 0")

    def _migrate_to_v3(self) -> None:
        """Index the episodes an older file already holds.

        The DDL is not the migration here. `CREATE VIRTUAL TABLE IF NOT EXISTS` does
        create `episodes_fts` on an old file — and leaves it empty, which is
        indistinguishable from a store whose turns genuinely match nothing. Every
        pre-v3 episode would therefore stay exactly as unfindable as it was before the
        upgrade, silently, which is the bug this version exists to fix rather than to
        re-ship. Vectors need no equivalent pass: none were ever written, and
        `Engram.reembed()` is what supplies them.

        Shape-driven and idempotent, like `_migrate_to_v2`: a brand-new database also
        arrives here at version 0, with no episodes, and does nothing.
        """
        self._db.execute(
            "INSERT INTO episodes_fts (rowid, episode_id, content) "
            "SELECT rowid, id, content FROM episodes "
            "WHERE rowid NOT IN (SELECT rowid FROM episodes_fts)"
        )

    # -- vector index --------------------------------------------------------

    def _attach_vectors(self) -> None:
        """Point the index at its backing file. Deliberately reads no vectors.

        Everything here answers from a covering index, so opening a 100k-row store
        costs a few index pages rather than deserializing 300 MB of blobs into a heap
        matrix — which is what made a large store take seconds to open and, past a
        point, impossible to open at all.
        """
        # Every caller already holds the write lock; `_index_lock` on top of it, in that
        # order, is what keeps this from rewriting the watermarks under a reader that is
        # halfway through folding in another process's writes.
        with self._index_lock:
            row = self._db.execute(_VEC_CENSUS).fetchone()
            if not row["n"]:
                return  # dimension stays unknown until something is written
            if row["lo"] != row["hi"]:
                raise ValueError(
                    f"{self.path}: embeddings have mixed dimensions ({row['lo']} and "
                    f"{row['hi']}). This store was written with more than one embedder; "
                    "re-embed it with a single one."
                )
            top, base = row["top"], None
            if row["loose"]:
                base, top = self._assign_slots()
            if not self._vec.attach(int(row["lo"]), int(top) + 1):
                self._rebuild_matrix()
            elif base is not None:
                # The mapping was usable but says nothing about rows that had no address
                # a moment ago, and a row of zeros is not the vector that was written.
                for t in _VEC_TABLES:
                    self._fill(t, "WHERE slot >= ?", (base,))

    def _assign_slots(self) -> tuple[int, int]:
        """Give a row to every vector that has none; return the first and the last.

        Two things arrive here: a v1 file, where the column did not exist, and a row
        written by something other than this class. Either way the vector is invisible
        to search until it is addressable. Both tables are numbered from one counter,
        because they share one matrix.
        """
        base = int(self._db.execute(
            "SELECT MAX(top) + 1 FROM ("
            + " UNION ALL ".join(f"SELECT COALESCE(MAX(slot), -1) AS top FROM {n}"
                                 for n in _VEC_TABLE_NAMES)
            + ")").fetchone()[0])
        nxt = base
        for t in _VEC_TABLES:
            loose = self._db.execute(
                f"SELECT {t.key} AS k FROM {t.name} WHERE slot IS NULL ORDER BY rowid"
            ).fetchall()
            self._db.executemany(
                f"UPDATE {t.name} SET slot = ?, seq = ? WHERE {t.key} = ?",
                [(nxt + i, nxt + i + 1, r["k"]) for i, r in enumerate(loose)],
            )
            nxt += len(loose)
        self._db.commit()
        return base, nxt - 1

    def _rebuild_matrix(self) -> None:
        """Refill the whole matrix from the vectors SQLite holds.

        Only runs when the mapped file is missing, stale or written by a different
        embedder. The database is the authority; the file is a view of it, which is
        what makes deleting the file a recoverable mistake rather than data loss.
        """
        for t in _VEC_TABLES:
            self._fill(t)
            self._seq[t.name] = self._max_seq(t)
        self._index_loaded = True
        self._data_version = self._version()

    def _fill(self, t: _VecTable, where: str = "", params: tuple = ()) -> None:
        """Copy vectors out of SQLite into their rows.

        The only place blobs are read in bulk, and the only place a blob that disagrees
        with its recorded width can surface — numpy would otherwise raise a broadcast
        error naming neither the store nor the cause.
        """
        sql = f"SELECT {t.key} AS k, slot, vec FROM {t.name} {where} ORDER BY slot"
        for r in self._db.execute(sql, params):
            try:
                self._vec.put(r["k"], r["slot"],
                              np.frombuffer(r["vec"], dtype=np.float32))
            except ValueError as e:
                raise ValueError(
                    f"{self.path}: embeddings have mixed dimensions ({e}). This store "
                    "was written with more than one embedder; re-embed it with a "
                    "single one."
                ) from e

    def _count_vectors(self, t: _VecTable | None = None) -> int:
        """How many vectors the store holds, in one table or across the matrix."""
        tables = (t,) if t is not None else _VEC_TABLES
        with self._read() as conn:
            return sum(int(conn.execute(f"SELECT COUNT(*) FROM {x.name}").fetchone()[0])
                       for x in tables)

    def _version(self) -> int:
        with self._read() as conn:
            return int(conn.execute("PRAGMA data_version").fetchone()[0])

    def _max_seq(self, t: _VecTable) -> int:
        return int(self._db.execute(
            f"SELECT COALESCE(MAX(seq), -1) FROM {t.name}").fetchone()[0])

    def _read_map(self) -> None:
        """Fold every vector written since the watermarks into the name-to-row map.

        Serves both the first use and every refresh after it. The ORDER BY is what
        makes SQLite answer from `emb_seq`, which carries all three columns — without
        it the scan walks the table itself and reads a page per 3 KB blob.
        """
        with self._read() as conn:
            cur = conn.cursor()
            cur.row_factory = None  # 100k Row objects cost more than the query does
            for t in _VEC_TABLES:
                cur.execute(f"SELECT {t.key}, slot, seq FROM {t.name} WHERE seq > ? "
                            "ORDER BY seq", (self._seq[t.name],))
                for item_id, slot, seq in cur.fetchall():
                    self._vec.map(item_id, slot)
                    self._seq[t.name] = seq

    def _ensure_index(self) -> None:
        """Make the map usable and current before a read resolves through it.

        Loading it lazily keeps a process that only writes (or only reads claims) from
        paying for it at all. Refreshing it is what stops a claim written by another
        worker from being permanently invisible to this one's vector leg — invisible
        and silent, because BM25 still finds it and fusion merely ranks it worse.
        `PRAGMA data_version` moves only when a *different* connection commits, so the
        common case costs one pragma.

        The first load is the only part that takes the write lock, and it has to: with
        no vectors mapped yet, `_ensure_dim` may have to *assign* slots to rows that
        have none, which is a write. Every refresh afterwards runs on the reading
        thread's own connection, so a sweep in progress does not hold it up.
        """
        if not self._index_loaded:
            # No second check inside the lock. Two threads arriving together both load,
            # and the loser's pass costs one empty tail query — cheaper than a branch
            # that only ever runs under a race and can therefore never be tested.
            with self._lock, self._index_lock:
                self._ensure_dim()
                self._read_map()
                self._index_loaded = True
                self._data_version = self._version()
            return
        # `_read()` outside `_index_lock`, never the other way round. On a store with no
        # snapshot connection `_read()` takes the *write* lock, and the branch above takes
        # those same two in that order — reversed here, a refreshing reader and a thread
        # loading the index cold deadlock on each other. Nothing is read through the
        # context manager directly; it is entered for the ordering.
        with self._read(), self._index_lock:
            version = self._version()
            if version != self._data_version:
                self._data_version = version
                self._read_map()

    def _ensure_dim(self) -> None:
        """Learn the store's dimension if another process wrote the first vector."""
        if self._vec.dim is None:
            self._attach_vectors()

    def _reconcile_index(self) -> None:
        """Put the matrix back in step with the database after a rollback.

        The matrix is a mapped file and takes no part in SQLite's transaction, so
        without this a rolled-back batch leaves vectors nothing owns: they still match
        queries, `stats()` still counts them, and purge's erasure guarantee no longer
        holds. The database is the authority, so every row the batch touched is simply
        re-read from it.
        """
        if self._cleared:
            # A rolled-back `clear_embeddings` is the one case with nothing to repair
            # row by row: the vectors all came back and the matrix holds none of them.
            self._cleared = False
            self._vec.reset()
            self._attach_vectors()
            return
        for t, item_id in self._touched:
            r = self._db.execute(
                f"SELECT slot, vec FROM {t.name} WHERE {t.key}=?", (item_id,)
            ).fetchone()
            if r is None:
                self._vec.forget(item_id)
            else:
                self._vec.put(item_id, r["slot"],
                              np.frombuffer(r["vec"], dtype=np.float32))

    # -- transaction batching ------------------------------------------------

    def _maybe_commit(self) -> None:
        if self._batch_depth == 0:
            self._db.commit()

    @contextmanager
    def batch(self) -> Iterator["SQLiteStore"]:
        """Defer commits until the block exits, then commit once.

        Per-statement commits are the right default for a memory store, but bulk paths
        (ingesting a transcript, a consolidation sweep) pay a commit per claim for no
        benefit, since the whole sweep is one logical operation. Reentrant, so nesting
        is harmless.

        Durability caveat, stated precisely because it is easy to assume otherwise: this
        store runs `synchronous=NORMAL` in WAL mode, so a commit survives a *process*
        crash but not a machine or power loss until the next checkpoint. Set
        `synchronous=FULL` if acknowledged writes must survive power loss; it costs an
        fsync per commit. The mapped matrix inherits exactly that guarantee — its dirty
        pages live in the same page cache the database's do — and is rebuilt from the
        vectors SQLite holds if it is ever found stale.
        """
        with self._lock:
            self._batch_depth += 1
            try:
                yield self
            except BaseException:
                if self._batch_depth == 1:
                    self._db.rollback()
                    self._reconcile_index()
                raise
            finally:
                self._batch_depth -= 1
                if self._batch_depth == 0:
                    self._touched.clear()
                    self._cleared = False
                    self._maybe_commit()

    def _mark(self, t: _VecTable, item_id: str) -> None:
        if self._batch_depth:
            self._touched.add((t, item_id))

    # -- scope / liveness SQL ------------------------------------------------

    @staticmethod
    def _scope_clause(scopes: Sequence[Scope], alias: str = "") -> tuple[str, list]:
        a = f"{alias}." if alias else ""
        if not scopes:
            # Fail closed. An empty scope list means "no scope was resolved", which is a
            # caller bug — and matching everything would hand back every tenant's rows.
            # `Scope.ancestors()` never returns empty, so this is unreachable from the
            # public API today; it exists because `candidate_ids`, `lexical_search` and
            # `vector_search` are part of the published Store protocol, and a server that
            # computes scopes from a filter can hand us [].
            return "1=0", []
        parts, params = [], []
        for s in scopes:
            parts.append(f"({a}tenant IS ? AND {a}usr IS ? AND {a}agent IS ? AND {a}session IS ?)")
            params += [s.tenant, s.user, s.agent, s.session]
        return "(" + " OR ".join(parts) + ")", params

    @staticmethod
    def _live_clause(as_of: datetime | None, include_invalidated: bool,
                     alias: str = "") -> tuple[str, list]:
        """Both time axes must agree for a claim to count as believed-and-in-force.

        `include_invalidated` lifts the two *end-of-life* constraints so retracted and
        expired claims become visible for auditing. It deliberately does NOT lift the
        transaction-time floor: a claim recorded after `as_of` was not knowledge we had
        at `as_of`, and letting it through would answer "what did we believe in March?"
        with something we first heard in July. That is the one way a bitemporal query
        can actively lie, so the floor holds under every flag combination.
        """
        a = f"{alias}." if alias else ""
        t = _ts(as_of) if as_of is not None else datetime.now(timezone.utc).timestamp()
        if include_invalidated:
            return f"({a}recorded_at <= ?)", [t]
        clause = (
            f"({a}recorded_at <= ? "
            f"AND ({a}invalidated_at IS NULL OR {a}invalidated_at > ?) "
            f"AND {a}valid_from <= ? "
            f"AND ({a}valid_to IS NULL OR {a}valid_to > ?))"
        )
        return clause, [t, t, t, t]

    @staticmethod
    def _happened_clause(as_of: datetime | None, alias: str = "") -> tuple[str, list]:
        """An episode's one time axis: had this turn happened yet?

        Episodes are not bitemporal and have no liveness. Nothing retires them, nothing
        supersedes them, and `invalidated_at`/`valid_to` have no analogue — a turn that
        was said was said. What survives from `_live_clause` is only its floor, and for
        the same reason: returning a turn from July when asked what we knew in March is
        the one way a time-travel query can lie, and it lies about raw source text
        rather than about a derived belief.
        """
        a = f"{alias}." if alias else ""
        t = _ts(as_of) if as_of is not None else datetime.now(timezone.utc).timestamp()
        return f"({a}ts <= ?)", [t]

    # -- episodes ------------------------------------------------------------

    def add_episode(self, ep: Episode) -> None:
        with self._lock:
            self._db.execute(
                _EPISODE_UPSERT,
                (ep.id, ep.scope.tenant, ep.scope.user, ep.scope.agent, ep.scope.session,
                 ep.role, ep.content, _ts(ep.ts), ep.hash, json.dumps(ep.meta)),
            )
            # Mirror the episode's rowid into the FTS table, exactly as `put_claim`
            # does and for the same reason: `episode_id` is UNINDEXED, so deleting on
            # it scans the whole index, and the upsert above exists precisely so the
            # rowid survives a re-add.
            rowid = self._db.execute(
                "SELECT rowid FROM episodes WHERE id=?", (ep.id,)).fetchone()["rowid"]
            self._db.execute("DELETE FROM episodes_fts WHERE rowid=?", (rowid,))
            self._db.execute(
                "INSERT INTO episodes_fts (rowid, episode_id, content) VALUES (?,?,?)",
                (rowid, ep.id, ep.content),
            )
            self._maybe_commit()

    def _row_to_episode(self, r: sqlite3.Row) -> Episode:
        return Episode(
            id=r["id"],
            scope=Scope(r["tenant"], r["usr"], r["agent"], r["session"]),
            role=r["role"],
            content=r["content"],
            ts=_dt(r["ts"]),  # type: ignore[arg-type]
            meta=json.loads(r["meta"]),
        )

    def get_episode(self, episode_id: str) -> Episode | None:
        with self._read() as conn:
            r = conn.execute("SELECT * FROM episodes WHERE id=?", (episode_id,)).fetchone()
        return self._row_to_episode(r) if r else None

    def find_episode_by_hash(self, tenant: str, ep_hash: str) -> Episode | None:
        with self._read() as conn:
            r = conn.execute(
                "SELECT * FROM episodes WHERE tenant=? AND hash=? LIMIT 1", (tenant, ep_hash)
            ).fetchone()
        return self._row_to_episode(r) if r else None

    def get_episodes(self, episode_ids: Sequence[str]) -> dict[str, Episode]:
        """Bulk fetch, for the same reason `get_claims` exists: hydrating a result set
        one row at a time makes a search cost O(results) queries."""
        out: dict[str, Episode] = {}
        if not episode_ids:
            return out
        ids = list(dict.fromkeys(episode_ids))
        with self._read() as conn:
            for i in range(0, len(ids), _MAX_SQL_PARAMS):
                chunk = ids[i:i + _MAX_SQL_PARAMS]
                q = f"SELECT * FROM episodes WHERE id IN ({','.join('?' * len(chunk))})"
                for r in conn.execute(q, chunk):
                    out[r["id"]] = self._row_to_episode(r)
        return out

    def iter_episodes(self, tenant: str | None = None) -> Iterable[Episode]:
        """Every stored turn, optionally for one tenant. What `reembed()` walks."""
        for row in self._iter_rows("episodes", tenant):
            yield self._row_to_episode(row)

    def _iter_rows(self, table: str, tenant: str | None,
                   extra: Sequence[str] = ()) -> Iterator[sqlite3.Row]:
        """Walk a table in rowid order, a page at a time.

        A generator that ran `fetchall()` first was a generator in shape only: it
        materialised the whole table before yielding anything, which is exactly what
        `reembed()` must not do — that call exists to walk a store too large to have been
        embedded correctly the first time, and loading every episode into a list to
        re-encode them one at a time turns a slow operation into an unrunnable one.

        Keyset pagination on `rowid` rather than `LIMIT/OFFSET`, because OFFSET re-walks
        the rows it skips and makes a full scan quadratic. The ceiling is read once at the
        start so a write landing mid-walk cannot extend it indefinitely; rows deleted
        during the walk simply do not appear, which is the same guarantee the old
        `fetchall()` gave.

        The connection is taken and released per page, never held across a `yield`: the
        caller decides when to resume, and a generator abandoned half-way would otherwise
        keep a reader checked out until it was garbage collected.
        """
        conds = list(extra)
        params: list = []
        if tenant is not None:
            conds.append("tenant=?")
            params.append(tenant)
        where = (" AND " + " AND ".join(conds)) if conds else ""

        with self._read() as conn:
            row = conn.execute(f"SELECT MAX(rowid) AS m FROM {table}").fetchone()
        ceiling = row["m"] if row is not None and row["m"] is not None else 0

        after = 0
        while after < ceiling:
            with self._read() as conn:
                page = conn.execute(
                    # `rowid` explicitly: it is implicit in the table but not in `*`,
                    # and the walk needs it to know where the next page starts.
                    f"SELECT rowid AS _rid, * FROM {table} "
                    f"WHERE rowid > ? AND rowid <= ?{where} ORDER BY rowid LIMIT ?",
                    [after, ceiling, *params, _ITER_PAGE],
                ).fetchall()
            if not page:
                # Every remaining row was filtered out rather than absent, so stepping
                # past the page window is what makes progress. Without this the walk
                # spins forever on a tenant whose rows sit past a gap.
                after += _ITER_PAGE
                continue
            for r in page:
                yield r
            after = page[-1]["_rid"]

    # -- claims --------------------------------------------------------------

    def put_claim(self, c: Claim) -> None:
        with self._lock:
            self._db.execute(
                _CLAIM_UPSERT,
                (
                    c.id, c.scope.tenant, c.scope.user, c.scope.agent, c.scope.session,
                    c.subject, c.predicate, c.object, c.text, c.polarity,
                    c.memory_type.value, _ts(c.valid_from), _ts(c.valid_to),
                    _ts(c.recorded_at), _ts(c.invalidated_at), c.invalidated_by,
                    c.confidence, c.salience, c.observation_count,
                    json.dumps(c.sources), c.derivation.value, c.extractor,
                    json.dumps(c.meta), c.fact_key, c.value_key,
                ),
            )
            # Mirror the claim's rowid into the FTS table so the index entry can be
            # replaced by rowid. `claim_id` is UNINDEXED — deleting on it costs a full
            # scan of the FTS index, which turns N writes over N rows into O(n^2) and
            # was, measurably, the single slowest thing in the system.
            row = self._db.execute("SELECT rowid FROM claims WHERE id=?", (c.id,)).fetchone()
            rowid = row["rowid"]
            self._db.execute("DELETE FROM claims_fts WHERE rowid=?", (rowid,))
            self._db.execute(
                "INSERT INTO claims_fts (rowid, claim_id, text) VALUES (?,?,?)",
                (rowid, c.id, c.text),
            )
            self._maybe_commit()

    @staticmethod
    def _row_to_claim(r: sqlite3.Row) -> Claim:
        return Claim(
            id=r["id"],
            scope=Scope(r["tenant"], r["usr"], r["agent"], r["session"]),
            subject=r["subject"], predicate=r["predicate"], object=r["object"],
            text=r["text"], polarity=r["polarity"],
            memory_type=MemoryType(r["memory_type"]),
            valid_from=_dt(r["valid_from"]),  # type: ignore[arg-type]
            valid_to=_dt(r["valid_to"]),
            recorded_at=_dt(r["recorded_at"]),  # type: ignore[arg-type]
            invalidated_at=_dt(r["invalidated_at"]),
            invalidated_by=r["invalidated_by"],
            confidence=r["confidence"], salience=r["salience"],
            observation_count=r["obs_count"],
            sources=json.loads(r["sources"]),
            derivation=Derivation(r["derivation"]),
            extractor=r["extractor"],
            meta=json.loads(r["meta"]),
        )

    def get_claim(self, claim_id: str) -> Claim | None:
        with self._read() as conn:
            r = conn.execute("SELECT * FROM claims WHERE id=?", (claim_id,)).fetchone()
        return self._row_to_claim(r) if r else None

    def get_claims(self, claim_ids: Sequence[str]) -> dict[str, Claim]:
        """Fetch many claims in one round trip.

        Retrieval hydrates every fused candidate and `get_all` hydrates a whole scope;
        doing that one `get_claim` at a time is a classic N+1 that makes both scale with
        the number of results rather than with the query. Returns a mapping so callers
        keep their own ordering — the DB's row order is not the ranking.
        """
        out: dict[str, Claim] = {}
        if not claim_ids:
            return out
        ids = list(dict.fromkeys(claim_ids))
        with self._read() as conn:
            for i in range(0, len(ids), _MAX_SQL_PARAMS):
                chunk = ids[i:i + _MAX_SQL_PARAMS]
                q = f"SELECT * FROM claims WHERE id IN ({','.join('?' * len(chunk))})"
                for r in conn.execute(q, chunk):
                    out[r["id"]] = self._row_to_claim(r)
        return out

    def competing_claims(self, tenant: str, fact_key: str,
                         as_of: datetime | None = None) -> list[Claim]:
        """Every live claim in the same (subject, predicate) slot.

        One indexed lookup. No embeddings, no top-k, so a contradiction cannot hide
        below a similarity cutoff the way it can in a vector-search-based updater.
        """
        live, lp = self._live_clause(as_of, include_invalidated=False)
        # `_read` keeps a thread inside `batch()` on the writer's connection, which this
        # method depends on absolutely: the reconciler asks it mid-transaction, and a
        # snapshot that predates the claim written two statements ago would report the
        # slot empty and let the contradiction through.
        with self._read() as conn:
            rows = conn.execute(
                f"SELECT * FROM claims WHERE tenant=? AND fact_key=? AND {live}",
                [tenant, fact_key] + lp,
            ).fetchall()
        return [self._row_to_claim(r) for r in rows]

    def _erase_row(self, table: str, fts: str, t: _VecTable, item_id: str) -> bool:
        """Erase one row and everything derived from its text. Caller holds the lock.

        The order is the whole content of this method, and it is not recoverable if it
        is wrong. The FTS entry is keyed on the row's *rowid* — `claim_id` is UNINDEXED —
        so once the row is gone there is no way to find the index entry that describes
        it, and the erased text stays in the index: matchable by search, unhydratable,
        and undeletable except by rebuilding the whole index. So: FTS first, by rowid,
        then the vector (which leaks the text back under inversion), then the row.
        """
        row = self._db.execute(
            f"SELECT rowid FROM {table} WHERE id=?", (item_id,)).fetchone()
        if row is None:
            return False
        self._db.execute(f"DELETE FROM {fts} WHERE rowid=?", (row["rowid"],))
        self._db.execute(
            f"INSERT OR IGNORE INTO vec_free (slot) SELECT slot FROM {t.name} "
            f"WHERE {t.key}=? AND slot IS NOT NULL", (item_id,))
        self._db.execute(f"DELETE FROM {t.name} WHERE {t.key}=?", (item_id,))
        # Zeroes the matrix row as well as unmapping it: the file outlives the process,
        # and a vector left behind is the text left behind.
        self._vec.forget(item_id)
        self._mark(t, item_id)
        self._db.execute(f"DELETE FROM {table} WHERE id=?", (item_id,))
        return True

    def _orphan(self, episode_id: str) -> bool:
        """Whether no surviving claim still cites this turn as a source.

        `sources` is stored as a JSON array, so the needle is what `json.dumps` would
        have written for this one id — quotes and escaping included — which makes the
        substring match exact without needing the JSON1 extension, and that matters
        because it is not compiled into every SQLite build this library runs on.

        The LIKE metacharacters in the needle are escaped rather than trusted. Ids
        generated here are hex and could not collide, but `sources` accepts whatever the
        caller passed, and an unescaped `%` matches other claims' citations — which
        would leave an orphaned turn *un*erased. Under-erasing is the failure direction
        that matters here.

        A table scan, deliberately: there is no reverse provenance index, an erasure
        request is rare, and being right about it is not negotiable.
        """
        needle = json.dumps(episode_id)
        for char in ("\\", "%", "_"):
            needle = needle.replace(char, "\\" + char)
        hit = self._db.execute(
            "SELECT 1 FROM claims WHERE sources LIKE ? ESCAPE '\\' LIMIT 1",
            (f"%{needle}%",)).fetchone()
        return hit is None

    def erase_claim(self, claim_id: str, *, sources: bool = False) -> bool:
        """Irreversibly erase one claim. Returns whether it existed.

        The per-claim counterpart to `purge`, and the reason it has to exist: `purge`
        erases a whole scope, `invalidate` retires without deleting anything, and an
        erasure request naming one memory had no honest answer between the two. The
        claim row goes, its FTS entry goes, and its vector is zeroed and its slot
        returned to `vec_free` for reuse.

        **The source turn does not go by default, and that is a decision, not an
        oversight.** An `Episode` is a whole conversation turn: it can be the origin of
        several claims and can hold a great deal the extractor never turned into one, so
        erasing it as a side effect of erasing one derived claim deletes data the caller
        did not name. `sources=True` erases the turns behind this claim that no
        surviving claim still cites — which is exactly right for a memory that *is* its
        source text (an imported note, a verbatim `add(infer=False)`) and wrong for a
        fact extracted from a conversation. The caller knows which it has; this cannot.

        Nothing here is undoable and nothing is audited: `history()` will show a gap
        where the claim was, because that is what erasure means.
        """
        with self._lock:
            row = self._db.execute(
                "SELECT tenant, sources FROM claims WHERE id=?", (claim_id,)).fetchone()
            tenant = row["tenant"] if row is not None else None
            cited = json.loads(row["sources"]) if row is not None and sources else []
            if not self._erase_row("claims", "claims_fts", _CLAIM_VECS, claim_id):
                return False
            # Orphan-checked after the claim is gone, so it cannot count itself a citer.
            for episode_id in cited:
                if self._orphan(episode_id):
                    self._erase_row("episodes", "episodes_fts", _EPISODE_VECS, episode_id)
            # Same reason as `purge`: the entity row holds the subject's and object's
            # first-seen spelling verbatim, so leaving it behind erases the claim and
            # keeps the text. Reference-counted, because one entity is usually shared.
            self._gc_entities(tenant)
            self._maybe_commit()
        return True

    def purge(self, scope: Scope) -> dict[str, int]:
        """Irreversibly erase everything at `scope` and beneath it.

        This is the deliberate exception to "nothing is ever deleted". Retirement is the
        right default — it is what makes the audit trail and `as_of` work — but a GDPR
        Article 17 or CCPA erasure request is a legal obligation that retirement does not
        satisfy, because the text remains readable. Purging a user therefore also takes
        their agents and sessions.

        Everything derived from the text goes too: the claims, the source episodes,
        every vector (which leaks content under inversion), and both FTS indexes (which
        store the tokens directly). Returns per-table counts so the caller can evidence
        the erasure.

        Episodes are erased on the same terms as claims and always were — what is new
        is that they now have indexes, and an FTS row surviving the row it describes is
        not a stale cache entry, it is the purged text still being searchable.
        """
        conds = ["tenant = ?"]
        params: list = [scope.tenant]
        for col, val in (("usr", scope.user), ("agent", scope.agent),
                         ("session", scope.session)):
            if val is not None:
                conds.append(f"{col} = ?")
                params.append(val)
        where = " AND ".join(conds)

        with self._lock:
            # Set-based, not a statement pair per row: erasing a user with 50k claims
            # is one request, and 100k round trips through the SQL layer made it look
            # like the store had hung.
            gone = 0
            for t, table, params2 in ((_CLAIM_VECS, "claims", params),
                                      (_EPISODE_VECS, "episodes", params)):
                doomed = f"SELECT id FROM {table} WHERE {where}"
                rows = self._db.execute(
                    f"SELECT {t.key} AS k FROM {t.name} WHERE {t.key} IN ({doomed})",
                    params2,
                ).fetchall()
                self._db.execute(
                    f"INSERT OR IGNORE INTO vec_free (slot) SELECT slot FROM {t.name} "
                    f"WHERE {t.key} IN ({doomed}) AND slot IS NOT NULL", params2)
                self._db.execute(
                    f"DELETE FROM {t.name} WHERE {t.key} IN ({doomed})", params2)
                for r in rows:
                    self._vec.forget(r["k"])
                    self._mark(t, r["k"])
                gone += len(rows)
            # FTS entries are keyed on their row's rowid, so they must go before the
            # rows do — afterwards the rowids are gone and the text is orphaned but
            # still searchable.
            for fts, table in (("claims_fts", "claims"), ("episodes_fts", "episodes")):
                self._db.execute(
                    f"DELETE FROM {fts} WHERE rowid IN "
                    f"(SELECT rowid FROM {table} WHERE {where})", params)
            claims = self._db.execute(f"DELETE FROM claims WHERE {where}", params).rowcount
            episodes = self._db.execute(
                f"DELETE FROM episodes WHERE {where}", params
            ).rowcount
            entities = self._gc_entities(scope.tenant)
            # `_maybe_commit`, not `commit`: an unconditional commit here would end an
            # enclosing `batch()` early and silently void its rollback guarantee.
            self._maybe_commit()
        return {"claims": claims, "episodes": episodes, "embeddings": gone,
                "entities": entities}

    def _gc_entities(self, tenant: str) -> int:
        """Drop entity rows in `tenant` that no surviving claim refers to.

        Called after an erasure, and it is not housekeeping — it is part of the erasure.
        `entities.canonical` holds the **first spelling we ever saw** of every subject and
        object, so a store that erased claims, episodes, vectors and both FTS indexes
        still had "14 Rue de la Paix, Paris" and "Grüner & Sohn Bestattungen GmbH" sitting
        in a live row, while `purge()` returned per-table counts as evidence of the
        erasure and `stats()` reported zero. That is the worst shape a privacy bug can
        take: the caller is told the data is gone. It survives `VACUUM` and
        `secure_delete`, because it is a live row rather than freelist residue.

        Reference counting rather than a prefix match on the owner, because the two
        disagree exactly where it matters. Entity ids are *owner*-scoped (tenant + user)
        while a purge may be narrower — `purge(session=…)` deleting every entity its owner
        holds would take rows the user's surviving sessions still use. Counting references
        is right in all three cases at once: after a tenant purge nothing survives so
        everything goes, after a user purge only other owners' rows are referenced, and
        after a session purge the shared rows stay and the session-only ones go.

        The reference is derived from the surviving claim rather than read from its
        `meta`: `subject_entity`/`object_entity` are only stamped when resolution has
        something to record, so a claim whose surface form was already canonical carries
        no key at all and would look like a claim referring to nothing.
        """
        from ..entities import entity_id, entity_key
        from ..types import owner_key

        rows = self._db.execute("SELECT id FROM entities WHERE tenant=?",
                                (tenant,)).fetchall()
        if not rows:
            return 0

        live: set[str] = set()
        for c in self._db.execute(
            "SELECT usr, agent, session, subject, object FROM claims WHERE tenant=?",
            (tenant,),
        ):
            owner = owner_key(Scope(tenant, c["usr"], c["agent"], c["session"]))
            for surface in (c["subject"], c["object"]):
                if surface:
                    live.add(entity_id(owner, entity_key(surface)))

        doomed = [r["id"] for r in rows if r["id"] not in live]
        if doomed:
            self._db.executemany(
                "DELETE FROM entities WHERE tenant=? AND id=?",
                [(tenant, eid) for eid in doomed],
            )
        return len(doomed)

    # -- learned schema ------------------------------------------------------

    def put_spec(self, spec: "PredicateSpec", tenant: str = "default") -> None:
        """Persist a predicate specification, usually one just learned from a model.

        Scoped to a tenant because the table used to be global, which meant one
        tenant's classification silently set another's contradiction behaviour and
        decay half-life. The default matches the tenant `Engram` uses when the caller
        names none, so a single-tenant caller lands where its claims already are.
        """
        with self._lock:
            self._db.execute(
                "INSERT INTO predicates "
                "(tenant, name, cardinality, volatility, memory_type, aliases, "
                " supersedes, learned) "
                "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(tenant, name) DO UPDATE SET "
                "cardinality=excluded.cardinality, volatility=excluded.volatility, "
                "memory_type=excluded.memory_type, aliases=excluded.aliases, "
                "supersedes=excluded.supersedes, learned=excluded.learned",
                (tenant, spec.name, spec.cardinality.value, spec.volatility.value,
                 spec.memory_type.value, json.dumps(list(spec.aliases)),
                 json.dumps(list(spec.supersedes)), int(spec.learned)),
            )
            self._maybe_commit()

    def all_specs(self, tenant: str = "default") -> list["PredicateSpec"]:
        from ..schema import Cardinality, PredicateSpec, Volatility

        with self._read() as conn:
            rows = conn.execute(
                "SELECT * FROM predicates WHERE tenant=?", (tenant,)).fetchall()
        return [
            PredicateSpec(
                name=r["name"],
                cardinality=Cardinality(r["cardinality"]),
                volatility=Volatility(r["volatility"]),
                memory_type=MemoryType(r["memory_type"]),
                aliases=tuple(json.loads(r["aliases"])),
                supersedes=tuple(json.loads(r["supersedes"])),
                learned=bool(r["learned"]),
            )
            for r in rows
        ]

    # -- resolved entities ---------------------------------------------------

    def put_entity(self, entity_id: str, canonical: str, aliases: Sequence[str],
                   tenant: str = "default") -> None:
        """Persist "these spellings are one thing", for one tenant.

        Durable for the same reason the predicate schema is, and with a sharper failure
        if it were not: entity ids are baked into the `fact_key`s already on disk, so a
        process that re-derived the mapping and disagreed by one id would not merely
        forget a synonym — it would address a different slot, stop seeing the
        contradiction between two spellings of one subject, and leave both live.

        Tenant-scoped because deciding that "Acme" and "Acme Corp" are one entity is a
        judgement about one customer's data, and the default matches the tenant `Engram`
        uses when the caller names none.
        """
        with self._lock:
            self._db.execute(
                "INSERT INTO entities (tenant, id, canonical, aliases) VALUES (?,?,?,?) "
                "ON CONFLICT(tenant, id) DO UPDATE SET "
                "canonical=excluded.canonical, aliases=excluded.aliases",
                (tenant, entity_id, canonical, json.dumps(list(aliases))),
            )
            self._maybe_commit()

    def all_entities(self, tenant: str = "default") -> list[tuple[str, str, tuple[str, ...]]]:
        """Every resolved entity for one tenant, as (id, canonical, aliases).

        Ordered by id so two processes rebuilding the same resolver agree on which
        alias wins when two entities claim one — an ordering that comes out of the data
        rather than out of SQLite's page layout.
        """
        with self._read() as conn:
            rows = conn.execute(
                "SELECT id, canonical, aliases FROM entities WHERE tenant=? ORDER BY id",
                (tenant,)).fetchall()
        return [(r["id"], r["canonical"], tuple(json.loads(r["aliases"]))) for r in rows]

    def slot_history(self, tenant: str, fact_key: str) -> list[Claim]:
        """Every claim ever recorded in one (subject, predicate) slot, oldest first.

        This is the audit trail for a single fact: what we believed, when we believed it,
        and what replaced it. Free here only because invalidation never deletes.
        """
        with self._read() as conn:
            rows = conn.execute(
                "SELECT * FROM claims WHERE tenant=? AND fact_key=? "
                "ORDER BY recorded_at ASC, id ASC",
                (tenant, fact_key),
            ).fetchall()
        return [self._row_to_claim(r) for r in rows]

    def find_by_value(self, tenant: str, value_key: str) -> list[Claim]:
        with self._read() as conn:
            rows = conn.execute(
                "SELECT * FROM claims WHERE tenant=? AND value_key=?", (tenant, value_key)
            ).fetchall()
        return [self._row_to_claim(r) for r in rows]

    def invalidate(self, claim_id: str, at: datetime, by: str | None) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE claims SET invalidated_at=?, invalidated_by=? WHERE id=?",
                (_ts(at), by, claim_id),
            )
            self._maybe_commit()

    def set_valid_to(self, claim_id: str, valid_to: datetime | None) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE claims SET valid_to=? WHERE id=?", (_ts(valid_to), claim_id)
            )
            self._maybe_commit()

    def reinforce(self, claim_id: str, salience: float, observation_count: int,
                  sources: Sequence[str]) -> None:
        with self._lock:
            r = self._db.execute("SELECT sources FROM claims WHERE id=?", (claim_id,)).fetchone()
            if r is None:
                return
            merged = list(dict.fromkeys(json.loads(r["sources"]) + list(sources)))
            self._db.execute(
                "UPDATE claims SET salience=?, obs_count=?, sources=? WHERE id=?",
                (salience, observation_count, json.dumps(merged), claim_id),
            )
            self._maybe_commit()

    # -- retrieval -----------------------------------------------------------

    def _set_vector(self, t: _VecTable, item_id: str, vec: np.ndarray) -> None:
        # Normalize once, here, so the persisted bytes and the matrix row hold the same
        # thing. Otherwise every cosine computed against a vector silently depends on
        # which of the two it was read from.
        v = _unit(vec)
        with self._lock:
            # Validate against the store's dimension *before* touching the DB. Writing
            # first would leave a row the index rejects, and the store would then fail
            # to reopen.
            self._ensure_dim()
            if self._vec.dim is not None and v.shape[0] != self._vec.dim:
                raise ValueError(
                    f"embedding dim {v.shape[0]} != store dim {self._vec.dim}; "
                    "this store was built with a different embedder"
                )
            # SQLite assigns the slot, inside the write transaction, because it is the
            # only party both processes agree with. Two workers computing "next free
            # row" independently would hand one row to two items and each would read
            # back the other's vector.
            slot = int(self._db.execute(
                t.upsert, (item_id, int(v.shape[0]), v.tobytes())
            ).fetchone()[0])
            self._db.execute("DELETE FROM vec_free WHERE slot=?", (slot,))
            self._vec.put(item_id, slot, v)
            self._mark(t, item_id)
            self._maybe_commit()

    def set_embedding(self, claim_id: str, vec: np.ndarray) -> None:
        self._set_vector(_CLAIM_VECS, claim_id, vec)

    def set_episode_embedding(self, episode_id: str, vec: np.ndarray) -> None:
        """Give a stored turn a vector, so semantic recall can reach it.

        Separate from `set_embedding` rather than overloaded on the id, because the two
        address different tables and silently guessing from a prefix is how an episode
        ends up in the claims index. Same matrix, same width, same slot space.
        """
        self._set_vector(_EPISODE_VECS, episode_id, vec)

    def clear_embeddings(self) -> int:
        """Drop every stored vector and release the dimension they fixed.

        The one thing re-embedding needs that `set_embedding` cannot express. The index
        binds its width to the first vector it sees and rejects every later one of a
        different width, so migrating to a new model has to empty the store of vectors
        *before* writing the first new one — otherwise the migration fails on that
        first write, having already replaced nothing, and the error names a dimension
        mismatch rather than the fact that it was never going to work.

        Claims, episodes and history are untouched: this drops the derived vectors, not
        the memory. Returns how many went, so the caller can report the migration.

        Both tables, unconditionally. Leaving episode vectors behind would keep the old
        model's width alive in the shared matrix, and the first re-embedded claim would
        be rejected by the very dimension check this call exists to release.
        """
        with self._lock:
            dropped = self._count_vectors()
            for t in _VEC_TABLES:
                self._db.execute(f"DELETE FROM {t.name}")
            self._db.execute("DELETE FROM vec_free")
            self._vec.reset()
            self._index_loaded = False
            self._seq = {t.name: -1 for t in _VEC_TABLES}
            self._data_version = -1
            # Row-by-row repair cannot undo this, so a rollback has to rebuild instead.
            self._cleared = bool(self._batch_depth)
            self._maybe_commit()
        return dropped

    def _get_vector(self, t: _VecTable, item_id: str) -> np.ndarray | None:
        with self._read() as conn:
            r = conn.execute(
                f"SELECT vec FROM {t.name} WHERE {t.key}=?", (item_id,)
            ).fetchone()
        return np.frombuffer(r["vec"], dtype=np.float32).copy() if r else None

    def get_embedding(self, claim_id: str) -> np.ndarray | None:
        """Read back a stored vector.

        Without this, anything needing claim similarity outside of search has to
        re-encode the text — which turns a background consolidation sweep into an
        embedding call per claim, i.e. a network round trip per claim against a hosted
        embedder. The vectors are already on disk; read them.

        Answered from SQLite rather than from the matrix so there is exactly one
        source: a cached read and a cold read cannot disagree if there is nothing to
        disagree with.
        """
        return self._get_vector(_CLAIM_VECS, claim_id)

    def get_episode_embedding(self, episode_id: str) -> np.ndarray | None:
        """The vector for one turn, or None if it has never been embedded.

        Also the cheap "is this turn already indexed?" probe, which is what keeps
        re-ingesting a transcript from re-encoding every turn in it.
        """
        return self._get_vector(_EPISODE_VECS, episode_id)

    def candidate_ids(self, scopes: Sequence[Scope], as_of: datetime | None = None,
                      include_invalidated: bool = False) -> list[str]:
        sc, sp = self._scope_clause(scopes)
        lv, lp = self._live_clause(as_of, include_invalidated)
        with self._read() as conn:
            cur = conn.cursor()
            # A whole-tenant scope returns every claim id; building a `Row` object for
            # each of them costs more than the query.
            cur.row_factory = None
            cur.execute(f"SELECT id FROM claims WHERE {sc} AND {lv}", sp + lp)
            return [r[0] for r in cur.fetchall()]

    def episode_candidate_ids(self, scopes: Sequence[Scope],
                              as_of: datetime | None = None) -> list[str]:
        """Every turn visible at these scopes. The episode half of `candidate_ids`.

        No `include_invalidated`: episodes have no end-of-life to lift. Scope, though,
        is filtered exactly as claims are — the same `_scope_clause`, the same
        fail-closed on an empty list — because "which turns can this caller see" is
        precisely the same question for raw text as for a derived belief, and raw text
        is the more sensitive of the two.
        """
        sc, sp = self._scope_clause(scopes)
        hp, hpp = self._happened_clause(as_of)
        with self._read() as conn:
            cur = conn.cursor()
            cur.row_factory = None
            cur.execute(f"SELECT id FROM episodes WHERE {sc} AND {hp}", sp + hpp)
            return [r[0] for r in cur.fetchall()]

    def lexical_search(self, query: str, scopes: Sequence[Scope], limit: int,
                       as_of: datetime | None = None,
                       include_invalidated: bool = False) -> list[tuple[str, float]]:
        m = _fts_query(query)
        if not m:
            return []
        sc, sp = self._scope_clause(scopes, alias="c")
        lv, lp = self._live_clause(as_of, include_invalidated, alias="c")
        sql = (
            "SELECT f.claim_id AS cid, bm25(claims_fts) AS s "
            "FROM claims_fts f JOIN claims c ON c.id = f.claim_id "
            f"WHERE claims_fts MATCH ? AND {sc} AND {lv} "
            "ORDER BY s ASC LIMIT ?"
        )
        with self._read() as conn:
            rows = conn.execute(sql, [m] + sp + lp + [limit]).fetchall()
        # bm25() is negative-is-better; flip it so callers see a normal ascending score.
        return [(r["cid"], -float(r["s"])) for r in rows]

    def lexical_search_episodes(self, query: str, scopes: Sequence[Scope], limit: int,
                                as_of: datetime | None = None) -> list[tuple[str, float]]:
        """BM25 over raw turn text, scope-filtered inside the query.

        The leg that matters most for episodes: a decision, a reason or a constraint is
        usually recalled by a word that was actually in it, and the claim extracted from
        the same turn — if any was — kept the fact and threw the wording away.
        """
        m = _fts_query(query)
        if not m:
            return []
        sc, sp = self._scope_clause(scopes, alias="e")
        hp, hpp = self._happened_clause(as_of, alias="e")
        sql = (
            "SELECT f.episode_id AS eid, bm25(episodes_fts) AS s "
            "FROM episodes_fts f JOIN episodes e ON e.id = f.episode_id "
            f"WHERE episodes_fts MATCH ? AND {sc} AND {hp} "
            "ORDER BY s ASC LIMIT ?"
        )
        with self._read() as conn:
            rows = conn.execute(sql, [m] + sp + hpp + [limit]).fetchall()
        return [(r["eid"], -float(r["s"])) for r in rows]

    def vector_search(self, qvec: np.ndarray, scopes: Sequence[Scope], limit: int,
                      as_of: datetime | None = None,
                      include_invalidated: bool = False) -> list[tuple[str, float]]:
        allowed = self.candidate_ids(scopes, as_of, include_invalidated)
        if not allowed:
            return []
        self._ensure_index()
        return self._vec.search(qvec, allowed, limit)

    def vector_search_episodes(self, qvec: np.ndarray, scopes: Sequence[Scope],
                               limit: int,
                               as_of: datetime | None = None) -> list[tuple[str, float]]:
        """Cosine over turn vectors, restricted to the ids this scope may see.

        Same matrix and same code path as `vector_search`; the candidate set is what
        differs, and it is what enforces isolation — an episode id the scope filter did
        not return is not passed to the index at all.
        """
        allowed = self.episode_candidate_ids(scopes, as_of)
        if not allowed:
            return []
        self._ensure_index()
        return self._vec.search(qvec, allowed, limit)

    # -- maintenance ---------------------------------------------------------

    def iter_claims(self, tenant: str | None = None,
                    include_invalidated: bool = False) -> Iterable[Claim]:
        sql = "SELECT * FROM claims"
        params: list = []
        conds = []
        if tenant is not None:
            conds.append("tenant=?")
            params.append(tenant)
        if not include_invalidated:
            conds.append("invalidated_at IS NULL")
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        with self._read() as conn:
            rows = conn.execute(sql, params).fetchall()
        for r in rows:
            yield self._row_to_claim(r)

    def stats(self, tenant: str | None = None) -> dict[str, int]:
        """Row counts, optionally for one tenant.

        Scoping matters on a shared store: unfiltered counts disclose how much data
        other tenants hold, which is a real signal even without their content.
        `embeddings` stays global — it is a property of the store, not of a tenant —
        and is counted in SQLite rather than in the index, because the index is a cache
        of a table that other processes also write to. It counts every vector in the
        matrix, claims' and episodes' alike, since that is what the file on disk holds
        and what a caller sizing the store is asking about.
        """
        where = " WHERE tenant = ?" if tenant is not None else ""
        params: tuple = (tenant,) if tenant is not None else ()
        and_ = " AND" if tenant is not None else " WHERE"

        with self._read() as conn:
            def q(sql: str) -> int:
                return int(conn.execute(sql, params).fetchone()[0])

            return {
                "episodes": q(f"SELECT COUNT(*) FROM episodes{where}"),
                "claims": q(f"SELECT COUNT(*) FROM claims{where}"),
                "live_claims": q(
                    f"SELECT COUNT(*) FROM claims{where}{and_} invalidated_at IS NULL"),
                "invalidated": q(
                    f"SELECT COUNT(*) FROM claims{where}{and_} invalidated_at IS NOT NULL"),
                "embeddings": self._count_vectors(),
            }

    def close(self) -> None:
        with self._lock:
            # Commit unconditionally: closing inside an open batch would otherwise
            # discard it, and an explicit close is a stronger signal than the batch.
            self._db.commit()
            with self._readers_lock:
                # Flag and drain together, so a thread that has just opened a connection
                # either lands in this list or is told the store is shut, never neither.
                self._closed = True
                readers, self._readers = self._readers, []
            # Every reading thread's connection, closed from here rather than left to
            # each thread: an open one holds a WAL read mark, which pins the log at the
            # oldest live snapshot and stops checkpointing — so a forgotten reader shows
            # up as a `-wal` file that grows without bound rather than as an error.
            for conn in readers:
                conn.close()
            self._db.close()
            self._vec.close()

    def __enter__(self) -> "SQLiteStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
