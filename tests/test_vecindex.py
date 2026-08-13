"""The vector matrix: its backing file, the no-copy query, cross-process coherence,
and the transaction boundary between the matrix and the rows it mirrors.

These are the properties that let the store hold more vectors than the process holds
memory, and that stop a second worker's writes from being silently unsearchable.
"""

import os
import sqlite3
import struct
import threading

import numpy as np
import pytest

from memvara.embed import HashingEmbedder
from memvara.schema import Cardinality, PredicateSpec, Volatility
from memvara.store import SQLiteStore
from memvara.store.sqlite import SCHEMA_VERSION, _VecIndex, _vec_path
from memvara.types import Claim, Episode, MemoryType, Scope

SCOPE = Scope("acme", "alice")

# The v1 shape, spelled out rather than imported: the point of a migration test is to
# open a file this build did not write.
V1_SCHEMA = """
CREATE TABLE claims (
    id TEXT PRIMARY KEY, tenant TEXT NOT NULL, usr TEXT, agent TEXT, session TEXT,
    subject TEXT NOT NULL, predicate TEXT NOT NULL, object TEXT NOT NULL,
    text TEXT NOT NULL, polarity INTEGER NOT NULL DEFAULT 1, memory_type TEXT NOT NULL,
    valid_from REAL NOT NULL, valid_to REAL, recorded_at REAL NOT NULL,
    invalidated_at REAL, invalidated_by TEXT, confidence REAL NOT NULL DEFAULT 1.0,
    salience REAL NOT NULL DEFAULT 1.0, obs_count INTEGER NOT NULL DEFAULT 1,
    sources TEXT NOT NULL DEFAULT '[]', derivation TEXT NOT NULL,
    extractor TEXT NOT NULL DEFAULT '', meta TEXT NOT NULL DEFAULT '{}',
    fact_key TEXT NOT NULL, value_key TEXT NOT NULL);
CREATE TABLE embeddings (
    claim_id TEXT PRIMARY KEY, dim INTEGER NOT NULL, vec BLOB NOT NULL);
CREATE VIRTUAL TABLE claims_fts
    USING fts5(claim_id UNINDEXED, text, tokenize='porter unicode61');
CREATE TABLE predicates (
    name TEXT PRIMARY KEY, cardinality TEXT NOT NULL, volatility TEXT NOT NULL,
    memory_type TEXT NOT NULL, aliases TEXT NOT NULL DEFAULT '[]',
    supersedes TEXT NOT NULL DEFAULT '[]', learned INTEGER NOT NULL DEFAULT 1);
"""


def claim(**kw) -> Claim:
    base = dict(subject="user", predicate="lives_in", object="Berlin", scope=SCOPE)
    base.update(kw)
    return Claim(**base)


def onehot(dim: int, i: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[i % dim] = 1.0
    return v


def embed(store: SQLiteStore, vec: np.ndarray, **kw) -> Claim:
    c = claim(**kw)
    store.put_claim(c)
    store.set_embedding(c.id, vec)
    return c


def write_v1(path: str, *, dim: int = 4, n: int = 5) -> list[str]:
    """A store file as the previous release left it: no slots, global predicates."""
    raw = sqlite3.connect(path)
    raw.executescript(V1_SCHEMA)
    raw.execute("PRAGMA user_version = 1")
    raw.execute("INSERT INTO predicates (name, cardinality, volatility, memory_type) "
                "VALUES (?,?,?,?)", ("works_at", "one", "slow", "semantic"))
    ids = []
    for i in range(n):
        cid = f"cl_v1_{i}"
        raw.execute("INSERT INTO claims VALUES (" + ",".join("?" * 25) + ")",
                    (cid, "acme", "alice", None, None, "user", f"p{i}", f"o{i}",
                     f"text {i}", 1, "semantic", 0.0, None, 0.0, None, None, 1.0, 1.0,
                     1, "[]", "extraction", "v1", "{}", f"fk{i}", f"vk{i}"))
        raw.execute("INSERT INTO embeddings (claim_id, dim, vec) VALUES (?,?,?)",
                    (cid, dim, onehot(dim, i).tobytes()))
        ids.append(cid)
    raw.commit()
    raw.close()
    return ids


# --- The backing file -------------------------------------------------------

def test_the_matrix_lives_beside_the_database_not_on_the_heap(tmp_path):
    path = str(tmp_path / "m.db")
    with SQLiteStore(path) as s:
        embed(s, onehot(8, 1))
        assert s._vec.path == path + ".vecs"
        assert os.path.exists(path + ".vecs")


def test_an_in_memory_store_has_nowhere_to_map_and_says_so():
    assert _vec_path(":memory:") is None
    assert _vec_path("") is None
    assert _vec_path("/tmp/x.db") == "/tmp/x.db.vecs"
    with SQLiteStore(":memory:") as s:
        embed(s, onehot(8, 1))
        assert s._vec.path is None
        assert s._vec._fh is None


def test_reopening_reads_the_header_not_the_vectors(tmp_path, monkeypatch):
    """The point of the mapped file: opening a large store must not deserialize it.

    A store that has to materialize every blob to open takes seconds at 100k rows and
    is unopenable — and therefore unrepairable — once it outgrows RAM.
    """
    path = str(tmp_path / "m.db")
    with SQLiteStore(path) as s:
        for i in range(20):
            embed(s, onehot(8, i), predicate=f"p{i}")

    def explode(self):  # pragma: no cover - the assertion is that it never runs
        raise AssertionError("reopening must not rebuild the matrix")

    monkeypatch.setattr(SQLiteStore, "_rebuild_matrix", explode)
    with SQLiteStore(path) as s2:
        assert s2._index_loaded is False, "the name-to-row map is loaded on demand"
        hits = s2.vector_search(onehot(8, 3), [SCOPE], limit=1)
        assert hits[0][1] == pytest.approx(1.0)
        assert s2._index_loaded is True


def test_a_deleted_matrix_file_is_rebuilt_from_the_stored_vectors(tmp_path):
    """The file is a view of what SQLite holds, so losing it is recoverable."""
    path = str(tmp_path / "m.db")
    with SQLiteStore(path) as s:
        kept = [embed(s, onehot(8, i), predicate=f"p{i}") for i in range(6)]
    os.remove(path + ".vecs")
    with SQLiteStore(path) as s2:
        hits = s2.vector_search(onehot(8, 2), [SCOPE], limit=1)
        assert hits[0][0] == kept[2].id
        assert hits[0][1] == pytest.approx(1.0)


def test_a_matrix_file_written_by_another_embedder_is_rebuilt(tmp_path):
    """Width is stamped in the header, so a file from a different embedder is detected
    rather than reinterpreted as garbage rows."""
    path = str(tmp_path / "m.db")
    with SQLiteStore(path) as s:
        c = embed(s, onehot(8, 1))
    with open(path + ".vecs", "r+b") as fh:
        fh.seek(8)
        fh.write(struct.pack("<II", 1, 384))  # claim a 384-wide matrix
    with SQLiteStore(path) as s2:
        assert s2.vector_search(onehot(8, 1), [SCOPE], limit=1)[0][0] == c.id


def test_a_truncated_matrix_file_is_rebuilt(tmp_path):
    path = str(tmp_path / "m.db")
    with SQLiteStore(path) as s:
        c = embed(s, onehot(8, 1))
    with open(path + ".vecs", "r+b") as fh:
        fh.truncate(3)
    with SQLiteStore(path) as s2:
        assert s2.vector_search(onehot(8, 1), [SCOPE], limit=1)[0][0] == c.id


def test_growth_across_the_file_boundary_preserves_every_vector(tmp_path):
    """Capacity starts at 256 rows. Extending the file must not disturb what is in it,
    and must not cost a copy of the old matrix."""
    path = str(tmp_path / "g.db")
    with SQLiteStore(path) as s:
        made = [embed(s, onehot(16, i), predicate=f"p{i}") for i in range(600)]
        assert s._vec._rows >= 600
        for i in (0, 255, 256, 599):
            hits = s.vector_search(onehot(16, i), [SCOPE], limit=600)
            assert made[i].id in [h[0] for h in hits]
    with SQLiteStore(path) as s2:
        assert s2.stats()["embeddings"] == 600
        assert s2.vector_search(onehot(16, 300), [SCOPE], limit=1)[0][1] == pytest.approx(1.0)


def test_reopening_never_grows_the_matrix_file(tmp_path):
    """Sizing the file from what it already holds rather than from what is missing
    doubles it on every open. It is sparse, so nothing complains — until the apparent
    size outgrows what can be mapped and the store cannot be opened at all, which is
    the exact failure the mapped file exists to prevent."""
    path = str(tmp_path / "grow.db")
    with SQLiteStore(path) as s:
        for i in range(400):
            embed(s, onehot(16, i), predicate=f"p{i}")
    settled = os.path.getsize(path + ".vecs")
    for _ in range(6):
        with SQLiteStore(path) as s:
            assert s.vector_search(onehot(16, 5), [SCOPE], limit=1)[0][1] \
                == pytest.approx(1.0)
    assert os.path.getsize(path + ".vecs") == settled


def test_the_file_grows_only_when_a_row_is_missing(tmp_path):
    path = str(tmp_path / "g2.db")
    with SQLiteStore(path) as s:
        embed(s, onehot(16, 0), predicate="first")
        start = os.path.getsize(path + ".vecs")
        for i in range(1, 200):  # still inside the initial 256 rows
            embed(s, onehot(16, i), predicate=f"p{i}")
        assert os.path.getsize(path + ".vecs") == start
        for i in range(200, 400):  # now past it
            embed(s, onehot(16, i), predicate=f"q{i}")
        assert os.path.getsize(path + ".vecs") > start


def test_a_bare_index_grows_on_the_heap_and_keeps_every_row():
    """`_VecIndex` with nowhere to map still works — it is what an in-memory store uses."""
    idx = _VecIndex()
    for i in range(700):
        idx.add(f"c{i}", onehot(4, i))
    assert len(idx) == 700
    assert np.allclose(idx.get("c257"), onehot(4, 257))
    idx.close()


# --- The query --------------------------------------------------------------

@pytest.mark.parametrize("share", [0.02, 0.2, 0.5, 1.0])
def test_dense_and_chunked_scoring_agree(share):
    """Two implementations of one product: gathering rows in reusable chunks when the
    candidate set is sparse, one contiguous pass when it is not. They must not be
    distinguishable from the outside."""
    rng = np.random.default_rng(3)
    idx = _VecIndex()
    for i in range(400):
        idx.add(f"c{i}", rng.standard_normal(32).astype(np.float32))
    allowed = [f"c{i}" for i in range(int(400 * share))]
    q = rng.standard_normal(32).astype(np.float32)

    got = idx.search(q, allowed, 10)
    rows = np.asarray([idx._row[c] for c in allowed], dtype=np.int64)
    reference = idx._mat[rows] @ (q / np.linalg.norm(q))
    order = np.argsort(-reference)[:10]
    assert [h[0] for h in got] == [allowed[int(i)] for i in order]
    assert [h[1] for h in got] == pytest.approx([float(reference[int(i)]) for i in order],
                                                abs=1e-6)
    idx.close()


def test_a_sparse_candidate_set_does_not_touch_the_rest_of_the_matrix(tmp_path):
    """The old spelling copied every candidate row per query — 307 MB allocated and
    discarded at 100k. A narrow scope must stay cheap regardless of store size."""
    path = str(tmp_path / "s.db")
    with SQLiteStore(path) as s:
        mine = embed(s, onehot(32, 7), scope=Scope("acme", "alice"))
        for i in range(500):
            embed(s, onehot(32, i), scope=Scope("acme", "bob"), predicate=f"p{i}")
        hits = s.vector_search(onehot(32, 7), [Scope("acme", "alice")], limit=5)
        assert [h[0] for h in hits] == [mine.id]


def test_search_below_the_dense_threshold_still_finds_the_best_row():
    idx = _VecIndex()
    for i in range(300):
        idx.add(f"c{i}", onehot(8, i))
    hits = idx.search(onehot(8, 3), ["c3", "c11"], 2)
    assert hits[0][0] in ("c3", "c11")
    assert hits[0][1] == pytest.approx(1.0)
    idx.close()


# --- Cross-process coherence ------------------------------------------------

def test_a_vector_written_by_another_worker_becomes_visible(tmp_path):
    """The failure this prevents is silent: the vector leg simply never sees the claim,
    while BM25 still finds it, so fusion only ranks it worse and nothing looks broken."""
    path = str(tmp_path / "c.db")
    a, b = SQLiteStore(path), SQLiteStore(path)
    a.vector_search(onehot(8, 0), [SCOPE], limit=1)  # A's index is now warm

    c = embed(b, onehot(8, 5), predicate="written_by_b")
    hits = a.vector_search(onehot(8, 5), [SCOPE], limit=1)
    assert [h[0] for h in hits] == [c.id]
    assert hits[0][1] == pytest.approx(1.0)
    a.close()
    b.close()


def test_a_vector_replaced_by_another_worker_is_the_one_that_is_searched(tmp_path):
    """Re-embedding keeps the row, so the update arrives through the shared mapping
    with no re-read at all."""
    path = str(tmp_path / "c.db")
    a, b = SQLiteStore(path), SQLiteStore(path)
    c = embed(a, onehot(8, 1))
    assert a.vector_search(onehot(8, 1), [SCOPE], limit=1)[0][1] == pytest.approx(1.0)

    b.set_embedding(c.id, onehot(8, 6))
    assert a.vector_search(onehot(8, 1), [SCOPE], limit=1)[0][1] == pytest.approx(0.0)
    assert a.vector_search(onehot(8, 6), [SCOPE], limit=1)[0][1] == pytest.approx(1.0)
    a.close()
    b.close()


def test_an_unchanged_store_is_not_re_read_on_every_query(tmp_path, monkeypatch):
    """Refreshing has to be cheap enough to do on the read path, so the generation
    check must short-circuit before any row is touched."""
    path = str(tmp_path / "c.db")
    with SQLiteStore(path) as s:
        embed(s, onehot(8, 1))
        s.vector_search(onehot(8, 1), [SCOPE], limit=1)

        reads = []
        real = SQLiteStore._read_map
        monkeypatch.setattr(SQLiteStore, "_read_map",
                            lambda self: (reads.append(1), real(self))[1])
        for _ in range(5):
            s.vector_search(onehot(8, 1), [SCOPE], limit=1)
        assert reads == [], "no other connection committed; there is nothing to re-read"


def test_a_store_backed_index_counts_the_store_not_its_own_cache(tmp_path):
    """`len(index)` is how the construction-time embedder check decides whether a store
    holds any vectors at all. Reporting "none" merely because this process has not
    needed the name-to-row map yet sends that check down a full scan of the claims
    table — on every open, which is the cost the mapped file exists to remove."""
    path = str(tmp_path / "l.db")
    with SQLiteStore(path) as s:
        for i in range(4):
            embed(s, onehot(8, i), predicate=f"p{i}")
    with SQLiteStore(path) as s2:
        assert s2._index_loaded is False
        assert len(s2._vec) == 4
        assert s2._vec.dim == 8


def test_a_bare_index_counts_what_it_holds():
    idx = _VecIndex()
    assert len(idx) == 0
    idx.add("a", onehot(4, 0))
    assert len(idx) == 1
    idx.remove("a")
    assert len(idx) == 0, "a removed row is still allocated but no longer held"
    idx.close()


def test_a_store_opened_after_another_wrote_the_first_vector_learns_its_width(tmp_path):
    """The width is discovered lazily, so a process that opened an empty store must
    still reject a mismatched vector once someone else has set the width."""
    path = str(tmp_path / "w.db")
    a = SQLiteStore(path)          # opened while empty: width unknown
    b = SQLiteStore(path)
    embed(b, np.ones(8, dtype=np.float32))
    c = claim(predicate="late")
    a.put_claim(c)
    with pytest.raises(ValueError, match="dim"):
        a.set_embedding(c.id, np.ones(3, dtype=np.float32))
    a.close()
    b.close()


# --- The matrix and the transaction -----------------------------------------

def test_a_rolled_back_batch_leaves_no_phantom_vector():
    """The matrix takes no part in SQLite's transaction, so without reconciliation a
    failed batch leaves vectors no claim owns: still matched by search, still counted,
    and no longer erasable by purging the claim that is gone."""
    store = SQLiteStore(":memory:")
    keeper = embed(store, onehot(8, 1), predicate="keep")
    with pytest.raises(RuntimeError):
        with store.batch():
            embed(store, onehot(8, 2), predicate="ghost")
            raise RuntimeError("boom")

    assert store.stats()["embeddings"] == 1
    hits = store.vector_search(onehot(8, 2), [SCOPE], limit=5)
    assert [h[0] for h in hits] == [keeper.id]
    assert hits[0][1] == pytest.approx(0.0), "the ghost row must not be reachable"
    store.close()


def test_a_rolled_back_re_embedding_restores_the_previous_vector(tmp_path):
    path = str(tmp_path / "r.db")
    with SQLiteStore(path) as store:
        c = embed(store, onehot(8, 1))
        with pytest.raises(RuntimeError):
            with store.batch():
                store.set_embedding(c.id, onehot(8, 4))
                raise RuntimeError("boom")

        assert np.allclose(store.get_embedding(c.id), onehot(8, 1))
        assert store.vector_search(onehot(8, 1), [SCOPE], limit=1)[0][1] == pytest.approx(1.0)


def test_a_rolled_back_purge_puts_the_vectors_back(tmp_path):
    path = str(tmp_path / "p.db")
    with SQLiteStore(path) as store:
        c = embed(store, onehot(8, 3))
        with pytest.raises(RuntimeError):
            with store.batch():
                store.purge(SCOPE)
                raise RuntimeError("boom")

        assert store.stats()["embeddings"] == 1
        assert store.vector_search(onehot(8, 3), [SCOPE], limit=1)[0][0] == c.id


def test_stats_counts_the_vectors_the_database_holds(tmp_path):
    """Counting the in-process index instead made `embeddings` a claim about this
    process's cache rather than about the store."""
    path = str(tmp_path / "n.db")
    a, b = SQLiteStore(path), SQLiteStore(path)
    embed(a, onehot(8, 1), predicate="from_a")
    embed(b, onehot(8, 2), predicate="from_b")
    assert a.stats()["embeddings"] == 2
    assert b.stats()["embeddings"] == 2
    a.close()
    b.close()


# --- Dropping every vector (the re-embedding hook) --------------------------

def test_clear_embeddings_drops_the_vectors_and_keeps_the_memory(tmp_path):
    """Derived data goes; claims, episodes and history do not. Re-embedding is a
    migration of the index, not of the store."""
    path = str(tmp_path / "c.db")
    with SQLiteStore(path) as s:
        kept = [embed(s, onehot(8, i), predicate=f"p{i}") for i in range(4)]
        s.add_episode(Episode(content="hello", scope=SCOPE))

        assert s.clear_embeddings() == 4
        assert s.stats() == {"episodes": 1, "claims": 4, "live_claims": 4,
                             "ended_claims": 0, "invalidated": 0, "embeddings": 0}
        assert s.get_embedding(kept[0].id) is None
        assert s.get_claim(kept[0].id).object == "Berlin"
        assert s.lexical_search("berlin", [SCOPE], limit=10)


def test_clearing_releases_the_width_so_a_new_embedder_can_own_the_store(tmp_path):
    """The reason the hook has to exist: the index fixes its dimension on the first
    vector it sees, so without this a migration fails on its first write having
    replaced nothing."""
    path = str(tmp_path / "w.db")
    with SQLiteStore(path) as s:
        c = embed(s, onehot(8, 1))
        with pytest.raises(ValueError, match="dim"):
            s.set_embedding(c.id, onehot(4, 1))

        s.clear_embeddings()
        s.set_embedding(c.id, onehot(4, 1))  # a narrower model now owns the store
        assert s._vec.dim == 4
        assert s.vector_search(onehot(4, 1), [SCOPE], limit=1)[0][0] == c.id
    with SQLiteStore(path) as s2:
        assert s2._vec.dim == 4
        assert s2.vector_search(onehot(4, 1), [SCOPE], limit=1)[0][1] == pytest.approx(1.0)


def test_clearing_empties_the_matrix_file_rather_than_unmapping_it(tmp_path):
    """"Drop every vector" that leaves the bytes on disk is not what it says — the
    file outlives the process and an embedding is invertible back to its text."""
    path = str(tmp_path / "e.db")
    with SQLiteStore(path) as s:
        embed(s, onehot(8, 1))
        assert os.path.getsize(path + ".vecs") > 0
        s.clear_embeddings()
        assert os.path.getsize(path + ".vecs") == 0


def test_clearing_hands_the_rows_back_from_the_start(tmp_path):
    path = str(tmp_path / "r.db")
    with SQLiteStore(path) as s:
        for i in range(3):
            embed(s, onehot(8, i), predicate=f"p{i}")
        s.purge(SCOPE)
        assert s._db.execute("SELECT COUNT(*) FROM vec_free").fetchone()[0] == 3

        s.clear_embeddings()
        assert s._db.execute("SELECT COUNT(*) FROM vec_free").fetchone()[0] == 0
        fresh = embed(s, onehot(8, 0), predicate="after")
        assert s._db.execute("SELECT slot FROM embeddings").fetchone()[0] == 0
        assert s.vector_search(onehot(8, 0), [SCOPE], limit=1)[0][0] == fresh.id


def test_clearing_an_empty_store_is_a_no_op():
    with SQLiteStore(":memory:") as s:
        assert s.clear_embeddings() == 0
        assert s.stats()["embeddings"] == 0


def test_a_rolled_back_clear_puts_every_vector_back(tmp_path):
    """A re-embedding that fails part way must leave the old index intact. Repairing
    row by row cannot do it — the rows all came back and the matrix holds none of
    them — so this is the one rollback that rebuilds."""
    path = str(tmp_path / "rb.db")
    with SQLiteStore(path) as s:
        made = [embed(s, onehot(8, i), predicate=f"p{i}") for i in range(3)]
        with pytest.raises(RuntimeError):
            with s.batch():
                s.clear_embeddings()
                raise RuntimeError("the new embedder blew up")

        assert s.stats()["embeddings"] == 3
        for i, c in enumerate(made):
            hits = s.vector_search(onehot(8, i), [SCOPE], limit=1)
            assert hits[0][0] == c.id
            assert hits[0][1] == pytest.approx(1.0)


def test_clearing_an_in_memory_store_releases_its_width_too():
    with SQLiteStore(":memory:") as s:
        c = embed(s, onehot(8, 1))
        assert s.clear_embeddings() == 1
        s.set_embedding(c.id, onehot(16, 1))
        assert s.vector_search(onehot(16, 1), [SCOPE], limit=1)[0][0] == c.id


# --- Slot reuse -------------------------------------------------------------

def test_purged_rows_are_reused_rather_than_leaked(tmp_path):
    """Without a free list a store that churns grows its matrix forever, because a
    removed row is only blanked and every new vector appends past it."""
    path = str(tmp_path / "f.db")
    with SQLiteStore(path) as store:
        for i in range(5):
            embed(store, onehot(8, i), predicate=f"p{i}")
        store.purge(SCOPE)
        assert [r[0] for r in store._db.execute("SELECT slot FROM vec_free ORDER BY slot")] \
            == [0, 1, 2, 3, 4]

        fresh = [embed(store, onehot(8, i), predicate=f"q{i}") for i in range(5)]
        assert store._db.execute("SELECT COUNT(*) FROM vec_free").fetchone()[0] == 0
        slots = {r[0] for r in store._db.execute("SELECT slot FROM embeddings")}
        assert slots == {0, 1, 2, 3, 4}, "the freed rows must be the ones handed out"
        assert store.vector_search(onehot(8, 2), [SCOPE], limit=1)[0][0] == fresh[2].id


def test_a_bare_index_reuses_the_slot_it_removed():
    idx = _VecIndex()
    idx.add("a", onehot(4, 0))
    idx.add("b", onehot(4, 1))
    assert idx.remove("a") is True
    idx.add("c", onehot(4, 2))
    assert idx._row["c"] == 0
    assert len(idx) == 2
    idx.close()


# --- Migration from v1 ------------------------------------------------------

def test_a_v1_file_is_migrated_and_stamped(tmp_path):
    path = str(tmp_path / "v1.db")
    write_v1(path)
    with SQLiteStore(path) as s:
        stamped = int(s._db.execute("PRAGMA user_version").fetchone()[0])
        # Compared against the constant, not a literal. What this test is for is that a
        # v1 file gets migrated and re-stamped at all; pinning the number instead means
        # every future schema change fails here for no reason, which trains people to
        # bump the literal without reading what broke.
        assert stamped == SCHEMA_VERSION
        assert SCHEMA_VERSION > 1, "a v1 file must be migrated to something newer"


def test_v1_predicate_specs_become_the_default_tenants(tmp_path):
    """The table was global, so its rows belong to the tenant `Memvara` uses when the
    caller names none — anything else silently loses a learned schema on upgrade."""
    path = str(tmp_path / "v1.db")
    write_v1(path)
    with SQLiteStore(path) as s:
        assert [spec.name for spec in s.all_specs("default")] == ["works_at"]
        assert s.all_specs("acme") == []


def test_v1_embeddings_become_addressable_and_stay_searchable(tmp_path):
    path = str(tmp_path / "v1.db")
    ids = write_v1(path, dim=4, n=5)
    with SQLiteStore(path) as s:
        slots = dict(s._db.execute("SELECT claim_id, slot FROM embeddings"))
        assert sorted(slots.values()) == [0, 1, 2, 3, 4]
        hits = s.vector_search(onehot(4, 1), [SCOPE], limit=1)
        assert hits[0][0] == ids[1]
        assert hits[0][1] == pytest.approx(1.0)
    with SQLiteStore(path) as s2:
        assert s2.vector_search(onehot(4, 2), [SCOPE], limit=1)[0][0] == ids[2]


def test_a_migrated_store_still_takes_writes(tmp_path):
    path = str(tmp_path / "v1.db")
    write_v1(path, dim=4, n=3)
    with SQLiteStore(path) as s:
        fresh = embed(s, onehot(4, 3), predicate="after_migration")
        assert s.stats()["embeddings"] == 4
        assert s.vector_search(onehot(4, 3), [SCOPE], limit=1)[0][0] in (fresh.id, "cl_v1_3")


def test_an_embedding_inserted_by_hand_is_given_a_row(tmp_path):
    """A row written by something other than this class has no slot, and an unmapped
    vector is invisible to search rather than merely unranked."""
    path = str(tmp_path / "h.db")
    with SQLiteStore(path) as s:
        embed(s, onehot(8, 0))
    raw = sqlite3.connect(path)
    # Columns named rather than positional. The point of the row is that this class did
    # not write it, not that it happens to know today's column count — spelled
    # positionally, every future column addition breaks a test about vector slots.
    raw.execute(
        "INSERT INTO claims (id, tenant, usr, agent, session, subject, predicate, "
        "object, text, polarity, memory_type, valid_from, recorded_at, fact_key, "
        "value_key, subject_key, object_key, derivation) "
        "VALUES (" + ",".join("?" * 18) + ")",
        ("cl_hand", "acme", "alice", None, None, "user", "p", "o", "hand", 1,
         "semantic", 0.0, 0.0, "fk", "vk", "user", "o", "extraction"))
    raw.execute("INSERT INTO embeddings (claim_id, dim, vec) VALUES (?,?,?)",
                ("cl_hand", 8, onehot(8, 5).tobytes()))
    raw.commit()
    raw.close()
    with SQLiteStore(path) as s2:
        assert s2.vector_search(onehot(8, 5), [SCOPE], limit=1)[0][0] == "cl_hand"


def test_a_blob_that_disagrees_with_its_recorded_width_is_diagnosed(tmp_path):
    """Caught while refilling the matrix, where numpy would otherwise raise a broadcast
    error naming neither the store nor the cause."""
    path = str(tmp_path / "bad.db")
    with SQLiteStore(path) as s:
        c = embed(s, onehot(8, 0))
    raw = sqlite3.connect(path)
    raw.execute("UPDATE embeddings SET vec=? WHERE claim_id=?",
                (onehot(3, 0).tobytes(), c.id))
    raw.commit()
    raw.close()
    os.remove(path + ".vecs")
    with pytest.raises(ValueError, match="mixed dimensions"):
        SQLiteStore(path)


# --- Contract A: tenant-scoped predicate specs ------------------------------

def spec(name: str, **kw) -> PredicateSpec:
    base = dict(name=name, cardinality=Cardinality.ONE, volatility=Volatility.SLOW,
                memory_type=MemoryType.SEMANTIC)
    base.update(kw)
    return PredicateSpec(**base)


def test_specs_do_not_leak_between_tenants():
    """A global table let one tenant's classification set another tenant's
    contradiction behaviour and decay half-life."""
    with SQLiteStore(":memory:") as s:
        s.put_spec(spec("works_at", cardinality=Cardinality.ONE), "t_a")
        s.put_spec(spec("works_at", cardinality=Cardinality.MANY), "t_b")
        assert [x.cardinality for x in s.all_specs("t_a")] == [Cardinality.ONE]
        assert [x.cardinality for x in s.all_specs("t_b")] == [Cardinality.MANY]
        assert s.all_specs("t_c") == []


def test_specs_round_trip_every_field():
    with SQLiteStore(":memory:") as s:
        original = spec("collects", volatility=Volatility.STATIC,
                        memory_type=MemoryType.PROCEDURAL,
                        aliases=("gathers", "hoards"), supersedes=("owns",),
                        learned=False)
        s.put_spec(original, "acme")
        got = s.all_specs("acme")[0]
        assert got == original


def test_put_spec_upserts_within_a_tenant():
    with SQLiteStore(":memory:") as s:
        s.put_spec(spec("works_at", cardinality=Cardinality.ONE), "acme")
        s.put_spec(spec("works_at", cardinality=Cardinality.MANY), "acme")
        assert [x.cardinality for x in s.all_specs("acme")] == [Cardinality.MANY]


def test_specs_default_to_the_tenant_memvara_uses_when_none_is_named():
    with SQLiteStore(":memory:") as s:
        s.put_spec(spec("works_at"))
        assert [x.name for x in s.all_specs()] == ["works_at"]
        assert [x.name for x in s.all_specs("default")] == ["works_at"]


def test_specs_survive_a_reopen_under_their_tenant(tmp_path):
    path = str(tmp_path / "spec.db")
    with SQLiteStore(path) as s:
        s.put_spec(spec("works_at"), "acme")
    with SQLiteStore(path) as s2:
        assert [x.name for x in s2.all_specs("acme")] == ["works_at"]


# --- Concurrency ------------------------------------------------------------

def test_concurrent_embedding_writes_never_share_a_row(tmp_path):
    """Slots come from SQLite so two writers cannot be handed the same row; if they
    were, each would read back the other's vector."""
    path = str(tmp_path / "t.db")
    store = SQLiteStore(path)
    emb = HashingEmbedder(dim=32)
    errors: list[BaseException] = []

    def worker(n: int) -> None:
        try:
            for i in range(20):
                c = claim(predicate=f"p_{n}_{i}", object=f"v{n}_{i}")
                store.put_claim(c)
                store.set_embedding(c.id, emb.encode([c.text])[0])
        except BaseException as e:  # noqa: BLE001 - surfaced via assert below
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    slots = [r[0] for r in store._db.execute("SELECT slot FROM embeddings")]
    assert len(slots) == 120
    assert len(set(slots)) == 120, "one row per claim"
    store.close()
