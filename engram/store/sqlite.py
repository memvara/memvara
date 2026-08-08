"""Default store: a single SQLite file.

Design notes:

* Time is persisted as epoch floats, not ISO strings. Both time axes get compared and
  range-scanned constantly, and float comparison is unambiguous across timezones and
  formats.
* Lexical search is FTS5/BM25 joined against the claims table, so scope and liveness
  filtering happen inside the query rather than by over-fetching and filtering after.
* Vector search is a numpy matmul over the in-process index, restricted to the
  candidate rows for the scope. Exact, not approximate: at the scale a local memory
  store actually operates at, ANN buys nothing and costs recall.

The whole thing runs in WAL mode behind a lock so the consolidation worker can write
while readers are live.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import Iterable, Iterator, Sequence

import numpy as np

from ..types import Claim, Derivation, Episode, MemoryType, Scope

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
CREATE INDEX IF NOT EXISTS ep_hash ON episodes(tenant, hash);

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
    vec      BLOB NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS claims_fts
    USING fts5(claim_id UNINDEXED, text, tokenize='porter unicode61');
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
# SQLite's parameter limit is 999 on older builds; chunk bulk lookups below it.
_MAX_SQL_PARAMS = 900


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
    """In-process exact cosine index with amortized-O(1) append."""

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim
        self._ids: list[str] = []
        self._row: dict[str, int] = {}
        self._mat: np.ndarray | None = None
        self._n = 0

    def _init_mat(self, dim: int) -> None:
        self.dim = dim
        self._mat = np.zeros((256, dim), dtype=np.float32)

    def add(self, claim_id: str, vec: np.ndarray) -> None:
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        if n > 0.0:
            v = v / n
        if self._mat is None:
            self._init_mat(v.shape[0])
        assert self._mat is not None
        if v.shape[0] != self.dim:
            raise ValueError(
                f"embedding dim {v.shape[0]} != index dim {self.dim}; "
                "the store was built with a different embedder"
            )
        existing = self._row.get(claim_id)
        if existing is not None:
            self._mat[existing] = v
            return
        if self._n == self._mat.shape[0]:
            self._mat = np.vstack([self._mat, np.zeros_like(self._mat)])
        self._mat[self._n] = v
        self._row[claim_id] = self._n
        self._ids.append(claim_id)
        self._n += 1

    def search(self, q: np.ndarray, allowed: Sequence[str], limit: int) -> list[tuple[str, float]]:
        if self._mat is None or self._n == 0 or limit <= 0:
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
        qq = np.asarray(q, dtype=np.float32).reshape(-1)
        nq = float(np.linalg.norm(qq))
        if nq > 0.0:
            qq = qq / nq
        if qq.shape[0] != self.dim:
            raise ValueError(f"query dim {qq.shape[0]} != index dim {self.dim}")
        scores = self._mat[np.asarray(rows, dtype=np.int64)] @ qq
        k = min(limit, scores.shape[0])
        part = np.argpartition(-scores, k - 1)[:k] if k < scores.shape[0] else np.arange(scores.shape[0])
        order = part[np.argsort(-scores[part])]
        return [(cids[int(i)], float(scores[int(i)])) for i in order]

    def get(self, claim_id: str) -> np.ndarray | None:
        """The stored unit vector, or None if this claim was never embedded."""
        row = self._row.get(claim_id)
        if row is None or self._mat is None:
            return None
        return self._mat[row].copy()

    def __len__(self) -> int:
        return self._n


class SQLiteStore:
    """Reference `Store` implementation. Single file, no server, no Docker."""

    def __init__(self, path: str = ":memory:") -> None:
        self.path = path
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._batch_depth = 0
        self._vec = _VecIndex()
        with self._lock:
            self._db.executescript(SCHEMA)
            self._db.commit()
        self._load_vectors()

    # -- transaction batching ------------------------------------------------

    def _maybe_commit(self) -> None:
        if self._batch_depth == 0:
            self._db.commit()

    @contextmanager
    def batch(self) -> Iterator["SQLiteStore"]:
        """Defer commits until the block exits, then commit once.

        Per-statement commits are the right default for a memory store — a crash must
        not lose an acknowledged write — but bulk paths (ingesting a transcript, a
        consolidation sweep) pay a durability round-trip per claim for no benefit, since
        the whole sweep is one logical operation. Reentrant, so nesting is harmless.
        """
        with self._lock:
            self._batch_depth += 1
            try:
                yield self
            except BaseException:
                if self._batch_depth == 1:
                    self._db.rollback()
                raise
            finally:
                self._batch_depth -= 1
                if self._batch_depth == 0:
                    self._maybe_commit()

    def _load_vectors(self) -> None:
        with self._lock:
            rows = self._db.execute("SELECT claim_id, dim, vec FROM embeddings").fetchall()
        for row in rows:
            try:
                self._vec.add(row["claim_id"], np.frombuffer(row["vec"], dtype=np.float32))
            except ValueError as e:
                raise ValueError(
                    f"{self.path}: embeddings have mixed dimensions ({e}). This store was "
                    "written with more than one embedder; re-embed it with a single one."
                ) from e

    # -- scope / liveness SQL ------------------------------------------------

    @staticmethod
    def _scope_clause(scopes: Sequence[Scope], alias: str = "") -> tuple[str, list]:
        a = f"{alias}." if alias else ""
        if not scopes:
            return "1=1", []
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

    # -- episodes ------------------------------------------------------------

    def add_episode(self, ep: Episode) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO episodes "
                "(id, tenant, usr, agent, session, role, content, ts, hash, meta) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (ep.id, ep.scope.tenant, ep.scope.user, ep.scope.agent, ep.scope.session,
                 ep.role, ep.content, _ts(ep.ts), ep.hash, json.dumps(ep.meta)),
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
        with self._lock:
            r = self._db.execute("SELECT * FROM episodes WHERE id=?", (episode_id,)).fetchone()
        return self._row_to_episode(r) if r else None

    def find_episode_by_hash(self, tenant: str, ep_hash: str) -> Episode | None:
        with self._lock:
            r = self._db.execute(
                "SELECT * FROM episodes WHERE tenant=? AND hash=? LIMIT 1", (tenant, ep_hash)
            ).fetchone()
        return self._row_to_episode(r) if r else None

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
        with self._lock:
            r = self._db.execute("SELECT * FROM claims WHERE id=?", (claim_id,)).fetchone()
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
        with self._lock:
            for i in range(0, len(ids), _MAX_SQL_PARAMS):
                chunk = ids[i:i + _MAX_SQL_PARAMS]
                q = f"SELECT * FROM claims WHERE id IN ({','.join('?' * len(chunk))})"
                for r in self._db.execute(q, chunk):
                    out[r["id"]] = self._row_to_claim(r)
        return out

    def competing_claims(self, tenant: str, fact_key: str,
                         as_of: datetime | None = None) -> list[Claim]:
        """Every live claim in the same (subject, predicate) slot.

        One indexed lookup. No embeddings, no top-k, so a contradiction cannot hide
        below a similarity cutoff the way it can in a vector-search-based updater.
        """
        live, lp = self._live_clause(as_of, include_invalidated=False)
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM claims WHERE tenant=? AND fact_key=? AND {live}",
                [tenant, fact_key] + lp,
            ).fetchall()
        return [self._row_to_claim(r) for r in rows]

    def slot_history(self, tenant: str, fact_key: str) -> list[Claim]:
        """Every claim ever recorded in one (subject, predicate) slot, oldest first.

        This is the audit trail for a single fact: what we believed, when we believed it,
        and what replaced it. Free here only because invalidation never deletes.
        """
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM claims WHERE tenant=? AND fact_key=? "
                "ORDER BY recorded_at ASC, id ASC",
                (tenant, fact_key),
            ).fetchall()
        return [self._row_to_claim(r) for r in rows]

    def find_by_value(self, tenant: str, value_key: str) -> list[Claim]:
        with self._lock:
            rows = self._db.execute(
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

    def set_embedding(self, claim_id: str, vec: np.ndarray) -> None:
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        # Normalize once, here, so the persisted bytes and the in-memory index hold the
        # same thing. Otherwise `get_embedding` returns a unit vector when the claim is
        # cached and a raw one when it is read from disk, and every cosine computed
        # against it silently depends on cache state.
        norm = float(np.linalg.norm(v))
        if norm > 0.0:
            v = v / norm
        # Validate against the live index *before* touching the DB. Writing first would
        # leave a row the index rejects, and the store would then fail to reopen.
        if self._vec.dim is not None and v.shape[0] != self._vec.dim:
            raise ValueError(
                f"embedding dim {v.shape[0]} != store dim {self._vec.dim}; "
                "this store was built with a different embedder"
            )
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO embeddings (claim_id, dim, vec) VALUES (?,?,?)",
                (claim_id, int(v.shape[0]), v.tobytes()),
            )
            self._maybe_commit()
        self._vec.add(claim_id, v)

    def get_embedding(self, claim_id: str) -> np.ndarray | None:
        """Read back a stored vector.

        Without this, anything needing claim similarity outside of search has to
        re-encode the text — which turns a background consolidation sweep into an
        embedding call per claim, i.e. a network round trip per claim against a hosted
        embedder. The vectors are already on disk; read them.
        """
        cached = self._vec.get(claim_id)
        if cached is not None:
            return cached
        with self._lock:
            r = self._db.execute(
                "SELECT vec FROM embeddings WHERE claim_id=?", (claim_id,)
            ).fetchone()
        return np.frombuffer(r["vec"], dtype=np.float32).copy() if r else None

    def candidate_ids(self, scopes: Sequence[Scope], as_of: datetime | None = None,
                      include_invalidated: bool = False) -> list[str]:
        sc, sp = self._scope_clause(scopes)
        lv, lp = self._live_clause(as_of, include_invalidated)
        with self._lock:
            rows = self._db.execute(
                f"SELECT id FROM claims WHERE {sc} AND {lv}", sp + lp
            ).fetchall()
        return [r["id"] for r in rows]

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
        with self._lock:
            rows = self._db.execute(sql, [m] + sp + lp + [limit]).fetchall()
        # bm25() is negative-is-better; flip it so callers see a normal ascending score.
        return [(r["cid"], -float(r["s"])) for r in rows]

    def vector_search(self, qvec: np.ndarray, scopes: Sequence[Scope], limit: int,
                      as_of: datetime | None = None,
                      include_invalidated: bool = False) -> list[tuple[str, float]]:
        allowed = self.candidate_ids(scopes, as_of, include_invalidated)
        if not allowed:
            return []
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
        with self._lock:
            rows = self._db.execute(sql, params).fetchall()
        for r in rows:
            yield self._row_to_claim(r)

    def stats(self) -> dict[str, int]:
        with self._lock:
            q = lambda s: int(self._db.execute(s).fetchone()[0])  # noqa: E731
            return {
                "episodes": q("SELECT COUNT(*) FROM episodes"),
                "claims": q("SELECT COUNT(*) FROM claims"),
                "live_claims": q("SELECT COUNT(*) FROM claims WHERE invalidated_at IS NULL"),
                "invalidated": q("SELECT COUNT(*) FROM claims WHERE invalidated_at IS NOT NULL"),
                "embeddings": len(self._vec),
            }

    def close(self) -> None:
        with self._lock:
            # Commit unconditionally: closing inside an open batch would otherwise
            # discard it, and an explicit close is a stronger signal than the batch.
            self._db.commit()
            self._db.close()

    def __enter__(self) -> "SQLiteStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
