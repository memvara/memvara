"""Erasure has to remove the text from the file, not merely from the queries.

`erase_claim` deleted the row, the FTS entry and the vector, reported per-table counts,
and left the erased claim's words readable in the file with `grep`. Two independent
reasons, and the second is the one nobody would guess:

1. **Ordinary tables.** SQLite marks a deleted row's space free and leaves the bytes.
   That half was known — `SECURITY.md` listed it as out of scope and named `VACUUM` as
   the deployment's lever.
2. **The text indexes.** `DELETE FROM claims_fts` does *not* remove the document's terms.
   FTS5 writes a delete marker and keeps the terms as live rows in the `claims_fts_data`
   shadow table. They are not residue in a freed page; they are current content of a
   current table, so **`VACUUM` does not touch them**. The lever the documentation named
   did not work on the half it was most needed for, and `docs/DEPLOY.md` said in as many
   words that the FTS entry was erased "(which stores the tokens directly)".

So every test here greps the **file on disk** rather than asking the store a question.
Asking the store was exactly what made this invisible for so long: every query already
answered correctly.

`VACUUM` is deliberately *not* run before the assertions. Running it would mask half of
what is being tested, and nothing in normal operation runs one.
"""

import sqlite3

import pytest

from memvara import Memvara, NullLLM
from memvara.embed import HashingEmbedder
from memvara.store import SQLiteStore
from memvara.store.sqlite import SCHEMA_VERSION

#: A token that cannot occur by accident in an empty store, a schema string, or a
#: predicate name — so finding it in the file means finding the erased claim.
SECRET = "Huntingtons"

#: What to actually grep for, and it is not `SECRET`. The text index is built with
#: `tokenize='porter unicode61'`, so what reaches `claims_fts_data` is the *stem* —
#: `huntington`, not `huntingtons`. Searching for the whole word finds nothing in the
#: index and the control tests below fail, which is how this was noticed. A substring
#: shared by the word and its stem matches both the index and the raw row.
PROBE = b"untington"


def store_at(path):
    return Memvara(str(path), llm=NullLLM(), embedder=HashingEmbedder(dim=64), user="d")


def on_disk(path) -> bool:
    """Is the erased text still in the file? The only question that matters here."""
    with open(path, "rb") as fh:
        return PROBE in fh.read().lower()


def in_index(path, table: str) -> bool:
    """Is it in this FTS5 shadow table? Live rows, which no VACUUM reclaims."""
    db = sqlite3.connect(str(path))
    try:
        return any(blob and PROBE in bytes(blob).lower()
                   for (blob,) in db.execute(f"SELECT block FROM {table}"))
    finally:
        db.close()


# --- the claim path -----------------------------------------------------------


def test_an_erased_claim_leaves_none_of_its_text_in_the_file(tmp_path):
    path = tmp_path / "m.db"
    mem = store_at(path)
    claim_id = mem.remember("Dara", "diagnosed_with", SECRET).added[0].id
    assert mem.erase(claim_id) is True
    mem.close()

    assert not in_index(path, "claims_fts"  + "_data"), "terms survived in the text index"
    assert not on_disk(path), "the erased text is still greppable in the file"


def test_an_erased_source_turn_leaves_none_of_its_text_either(tmp_path):
    """`sources=True` erases the turn behind the claim; the episode index is a second
    FTS5 table with the same shadow-table behaviour, and it is the one holding verbatim
    user text rather than a normalized triple."""
    path = tmp_path / "m.db"
    mem = store_at(path)
    claim_id = mem.remember("Dara", "diagnosed_with", SECRET,
                            sources=[f"Dara mentioned {SECRET} at the appointment"]
                            ).added[0].id
    assert mem.erase(claim_id, sources=True) is True
    mem.close()

    assert not in_index(path, "episodes_fts" + "_data")
    assert not on_disk(path)


def test_purge_and_reset_scrub_as_thoroughly_as_erase(tmp_path):
    """The scope-level paths delete through the same helpers, so they inherited the same
    defect. Asserted separately because they are the ones a subject-erasure request
    actually reaches."""
    for op in ("purge", "reset"):
        path = tmp_path / f"{op}.db"
        mem = store_at(path)
        mem.remember("Dara", "diagnosed_with", SECRET,
                     sources=[f"Dara mentioned {SECRET} once"])
        getattr(mem, op)()
        mem.close()
        assert not on_disk(path), f"{op}() left the text in the file"


def test_the_text_really_is_there_until_it_is_erased(tmp_path):
    """The control. Without it every assertion above could pass against a store that
    never wrote the text in the first place."""
    path = tmp_path / "m.db"
    mem = store_at(path)
    mem.remember("Dara", "diagnosed_with", SECRET)
    mem.close()
    assert on_disk(path)
    assert in_index(path, "claims_fts" + "_data")


# --- what makes it work, asserted so a later change cannot quietly undo it -----


def test_secure_delete_is_set_on_both_text_indexes_and_persists(tmp_path):
    """FTS5's option is stored in the table's own config, so it is set once at migration
    rather than on every open — which is also what keeps a read-only open from needing a
    write. If it is ever moved to the open path, this test still passes; if it is dropped,
    nothing else here would explain why the leak came back."""
    path = tmp_path / "m.db"
    store_at(path).close()
    db = sqlite3.connect(str(path))
    try:
        for table in ("claims_fts", "episodes_fts"):
            config = dict(db.execute(f"SELECT k, v FROM {table}_config").fetchall())
            assert config.get("secure-delete") == 1, f"{table}: {config}"
    finally:
        db.close()


def test_ordinary_rows_are_overwritten_rather_than_left_in_a_free_page(tmp_path):
    """The other half, and the reason the FTS fix alone is not enough.

    Turned off explicitly here, so this fails if `PRAGMA secure_delete` is ever dropped
    from `SCHEMA` — the FTS option would still hide the index half and the file would
    still be leaking.
    """
    path = tmp_path / "m.db"
    mem = store_at(path)
    mem.store._db.execute("PRAGMA secure_delete=OFF")
    claim_id = mem.remember("Dara", "diagnosed_with", SECRET).added[0].id
    mem.erase(claim_id)
    mem.close()
    assert on_disk(path), (
        "with the pragma off the bytes should survive in a free page — if they do not, "
        "this test has stopped proving that the pragma is what removes them"
    )


# --- the upgrade --------------------------------------------------------------


def test_a_store_written_before_the_fix_is_scrubbed_when_it_is_opened(tmp_path):
    """The migration's whole job. FTS5's option is not retroactive, so an existing file
    keeps its markers — and the terms they hide — until something merges the index.

    The pre-fix state is reconstructed rather than imported: the option is cleared and the
    version wound back, which is exactly the file a user upgrading from 0.2.x has.
    """
    path = tmp_path / "legacy.db"
    mem = store_at(path)
    claim_id = mem.remember("Dara", "diagnosed_with", SECRET).added[0].id
    # Wind the file back to what the previous release produced.
    mem.store._db.execute("INSERT INTO claims_fts(claims_fts, rank) VALUES('secure-delete', 0)")
    mem.store._db.execute("INSERT INTO episodes_fts(episodes_fts, rank) VALUES('secure-delete', 0)")
    mem.store._db.execute("PRAGMA secure_delete=OFF")
    mem.erase(claim_id)
    mem.store._db.execute(f"PRAGMA user_version = {SCHEMA_VERSION - 1}")
    mem.store._db.commit()
    mem.close()

    assert in_index(path, "claims_fts" + "_data"), (
        "the legacy state was not reproduced, so the migration below proves nothing"
    )

    reopened = store_at(path)                      # the upgrade
    reopened.close()
    assert not in_index(path, "claims_fts" + "_data"), "the migration did not clear it"


def test_the_migration_keeps_every_row_it_found(tmp_path):
    """An index merge that lost rows would be a far worse bug than the one it fixes."""
    path = tmp_path / "legacy.db"
    mem = store_at(path)
    for i in range(20):
        mem.remember(f"subject-{i}", "noted", f"ordinary words number {i}",
                     sources=[f"turn about number {i}"])
    before = {(c.subject, c.predicate, c.object) for c in mem.get_all()}
    mem.store._db.execute(f"PRAGMA user_version = {SCHEMA_VERSION - 1}")
    mem.store._db.commit()
    mem.close()

    reopened = store_at(path)
    try:
        assert {(c.subject, c.predicate, c.object)
                for c in reopened.get_all()} == before
        assert reopened.search("ordinary words number 7", k=5), "the text index still works"
        assert len(reopened.recall("number 3", k=5)) > 0
    finally:
        reopened.close()


def test_the_schema_version_moved_so_an_older_build_refuses_the_file():
    """The setting is durable state in the file, so the stamp has to move with it —
    otherwise an older build opens a version-7 file, sees 6, and writes to a text index
    whose format it does not understand."""
    assert SCHEMA_VERSION == 7
    store = SQLiteStore(":memory:")
    try:
        assert store._db.execute("PRAGMA user_version").fetchone()[0] == 7
    finally:
        store.close()


def test_an_in_memory_store_migrates_without_a_file_to_scrub(tmp_path):
    """`:memory:` takes the same path and has no shadow table worth merging. It is here
    because the migration runs unconditionally and a failure would be a crash on the most
    common configuration in the test suite."""
    mem = Memvara(llm=NullLLM(), embedder=HashingEmbedder(dim=64), user="d")
    try:
        claim_id = mem.remember("Dara", "diagnosed_with", SECRET).added[0].id
        assert mem.erase(claim_id) is True
    finally:
        mem.close()


@pytest.mark.parametrize("table", ["claims_fts", "episodes_fts"])
def test_optimize_left_the_index_usable(tmp_path, table):
    """A merged FTS5 index still has to answer. Cheap to assert, and the failure mode of
    a bad merge is a store that silently stops matching."""
    path = tmp_path / "m.db"
    mem = store_at(path)
    try:
        mem.add("I live in Lisbon and I like sardines", role="user")
        mem.remember("user", "lives_in", "Lisbon")
        assert mem.search("Lisbon", k=5, include_episodes=True)
    finally:
        mem.close()
