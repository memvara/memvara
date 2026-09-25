"""Stores written by old releases, opened by this checkout's code.

Each store under `tests/fixtures/stores/<tag>/` was written by release `<tag>`'s own
code, one store for each schema version a release has shipped; `build_stores.py` says
how. Each is committed with `golden.json`, the dump `golden.dump` took of it before
anything else opened it. Every test unpacks a copy into its own temporary directory, so
no test can change a committed file.
"""

from __future__ import annotations

import pathlib
import warnings

import pytest

from memvara.store.sqlite import SCHEMA_VERSION, SQLiteStore

from harness import stores
from harness.invariants import check_store_integrity

from . import golden


@pytest.mark.parametrize("tag", golden.TAGS)
def test_a_committed_store_holds_what_its_golden_dump_says(
        tag: str, tmp_path: pathlib.Path) -> None:
    """A check on the fixtures themselves. A store and its golden dump are written
    together, so the unopened store must read exactly as the dump says."""
    db = golden.unpack(tag, tmp_path)
    record = golden.load(tag)
    assert record["tag"] == tag
    assert golden.schema_version(db) == record["schema_version"] == golden.RELEASES[tag]
    assert golden.compare(record["data"], golden.dump(db)) == []


@pytest.mark.parametrize("tag", golden.TAGS)
def test_a_store_from_an_old_release_upgrades_without_loss_and_only_once(
        tag: str, tmp_path: pathlib.Path) -> None:
    db = golden.unpack(tag, tmp_path)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        stores.file(db).close()
    # A ResourceWarning here comes from garbage collection of whatever an earlier test
    # left open, not from this open, so it is not counted.
    assert [f"{w.category.__name__}: {w.message}" for w in seen
            if not issubclass(w.category, ResourceWarning)] == [], (
        "opening an old store with the embedder that wrote it warned")
    assert golden.schema_version(db) == SCHEMA_VERSION
    assert golden.compare(golden.load(tag)["data"], golden.dump(db)) == []
    assert check_store_integrity(db) == []
    first = golden.snapshot(db)
    stores.file(db).close()
    assert golden.changes(first, golden.snapshot(db)) == [], (
        "a second open changed the upgraded store")


@pytest.mark.parametrize("tag", golden.TAGS)
def test_every_claim_turn_and_document_of_an_upgraded_store_can_be_read(
        tag: str, tmp_path: pathlib.Path) -> None:
    db = golden.unpack(tag, tmp_path)
    data = golden.load(tag)["data"]
    live = golden.live(data["claims"])
    with stores.file(db) as mem:
        for row in data["claims"]:
            claim = mem.store.get_claim(row["id"])
            assert claim is not None, f"claim {row['id']} ({row['object']!r}) is unreadable"
            read = (claim.subject, claim.predicate, claim.object, claim.polarity,
                    golden.iso(claim.valid_from), golden.iso(claim.valid_to),
                    golden.iso(claim.recorded_at), golden.iso(claim.invalidated_at))
            held = (row["subject"], row["predicate"], row["object"], row["polarity"],
                    row["valid_from"], row["valid_to"], row["recorded_at"],
                    row["invalidated_at"])
            assert read == held
        for row in live:
            if row["project"] is None:
                found = [r.claim.id for r in mem.search(row["object"], k=10,
                                                        **golden.scope(row))]
                assert row["id"] in found, f"search({row['object']!r}) lost {row['id']}"
        for row in data["episodes"]:
            found = [r.episode.id for r in mem.search(row["content"], k=10,
                                                      include_episodes=True,
                                                      **golden.scope(row))
                     if hasattr(r, "episode")]
            assert row["id"] in found, f"search({row['content']!r}) lost turn {row['id']}"
        for row in data["claims"]:
            if row["sources"]:
                why = mem.why(row["id"], **golden.scope(row))
                assert why is not None
                assert set(row["sources"]) <= {e.id for e in why.episodes}
        for row in data["erasures"]:
            assert mem.store.get_claim(row["claim_id"]) is None
            assert mem.store.erasure_record(row["claim_id"]) is not None
        for row in data["documents"]:
            assert mem.get_document(row["id"], tenant=row["tenant"],
                                    user=row["usr"]) is not None
        # No reader sees another user's or another tenant's claims.
        for row in live:
            if row["tenant"] != "default" or row["usr"] != golden.USER:
                seen = [r.claim.id for r in mem.search(row["object"], k=10,
                                                       user=golden.USER)]
                assert row["id"] not in seen
    for row in live:
        if row["project"] is not None:
            with stores.file(db, project=row["project"]) as mem:
                found = [r.claim.id for r in mem.search(row["object"], k=10,
                                                        **golden.scope(row))]
                assert row["id"] in found


@pytest.mark.parametrize("tag", golden.TAGS)
def test_an_upgraded_store_recognises_the_facts_it_already_holds(
        tag: str, tmp_path: pathlib.Path) -> None:
    """Versions 6, 12 and 16 re-derive every claim's entity keys and both hashes, which
    the golden dump leaves out. This checks what they are for: restating a stored value
    reinforces the stored claim instead of adding a second one, and a new value in a
    single-valued slot ends the stored one."""
    db = golden.unpack(tag, tmp_path)
    mine = [row for row in golden.live(golden.load(tag)["data"]["claims"])
            if golden.in_default_scope(row)]
    with stores.file(db) as mem:
        for row in mine:
            receipt = mem.remember(row["subject"], row["predicate"], row["object"],
                                   user=golden.USER)
            assert ([c.id for c in receipt.reinforced], receipt.added) == ([row["id"]], []), (
                f"restating {row['predicate']} {row['object']!r} did not reinforce "
                f"{row['id']}")
        home = next(row for row in mine if row["predicate"] == "lives_in")
        receipt = mem.remember("user", "lives_in", "Porto", user=golden.USER)
        assert [c.id for c in receipt.ended] == [home["id"]]


@pytest.mark.parametrize("tag", golden.TAGS)
def test_a_claim_an_old_release_erased_stays_unreadable_after_the_upgrade(
        tag: str, tmp_path: pathlib.Path) -> None:
    db = golden.unpack(tag, tmp_path)
    before = golden.tables_holding(db, golden.ERASED_WORD)
    if golden.RELEASES[tag] < 7:
        # Before version 7 an erasure left the words as live rows of the text index's
        # shadow table. The fixture must still show that, or the check below is empty.
        assert before == ["claims_fts_data"]
    else:
        assert before == []
    stores.file(db).close()
    assert golden.tables_holding(db, golden.ERASED_WORD) == []
    # docs/UPGRADING.md, "Erasure now actually removes the text, and the schema is
    # version 7": the first open cleans the text index but does not rewrite pages an
    # older release freed, and one VACUUM after that open finishes the job.
    with stores.file(db) as mem:
        assert isinstance(mem.store, SQLiteStore)
        mem.store._db.execute("VACUUM")
    holding = [p.name for p in tmp_path.iterdir()
               if p.is_file() and golden.ERASED_WORD.encode() in p.read_bytes()]
    assert holding == []
