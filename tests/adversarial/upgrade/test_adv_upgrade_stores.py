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
from typing import Any

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
    # The rule build_stores holds every release to: a close checkpoints the log, so the
    # database file holds every write.
    assert not first["logs"]["-wal"], (
        f"closing the upgraded store left a {first['logs']['-wal']}-byte write-ahead log")
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
    for row in live:
        if row["project"] is not None:
            with stores.file(db, project=row["project"]) as mem:
                found = [r.claim.id for r in mem.search(row["object"], k=10,
                                                        **golden.scope(row))]
                assert row["id"] in found


#: The readers the visibility test asks as: one in each scope the fixture writes claims
#: in, and a sibling session and a sibling agent that it writes nothing in. The project
#: reader asks through a handle bound to the project.
READERS: dict[str, dict[str, Any]] = {
    "user": {"tenant": "default", "user": golden.USER},
    "session": {"tenant": "default", "user": golden.USER, "session": "s1"},
    "sibling session": {"tenant": "default", "user": golden.USER, "session": "s2"},
    "agent": {"tenant": "default", "user": golden.USER, "agent": "a1"},
    "sibling agent": {"tenant": "default", "user": golden.USER, "agent": "a2"},
    "other user": {"tenant": "default", "user": "u2"},
    "other tenant": {"tenant": "acme", "user": golden.USER},
    "project": {"tenant": "default", "user": golden.USER},
}
#: The scopes whose claims each reader sees, written out by hand for the fixture's
#: scopes. The rule is `Scope.sees` in memvara/types.py: "a handle sees its own scope and
#: every broader one, and never a deeper one", which `search` and `get_all` apply
#: through `Scope.ancestors()`. So a user-wide reader does not see a session's claims.
SEES: dict[str, set[str]] = {
    "user": {"user"},
    "session": {"user", "session"},
    "sibling session": {"user"},
    "agent": {"user", "agent"},
    "sibling agent": {"user"},
    "other user": {"other user"},
    "other tenant": {"other tenant"},
    "project": {"user", "project"},
}


def written_in(row: dict[str, object]) -> str:
    """The scope, as one of the names `SEES` uses, that a claim was written in."""
    if row["tenant"] != "default":
        return "other tenant"
    if row["usr"] != golden.USER:
        return "other user"
    for field in ("project", "session", "agent"):
        if row[field] is not None:
            return field
    return "user"


def by_project(live: list[dict[str, object]]) -> list[str | None]:
    """The projects the live claims were recorded against, `None` first."""
    projects = {str(row["project"]) for row in live if row["project"] is not None}
    return [None, *sorted(projects)]


@pytest.mark.parametrize("tag", golden.TAGS)
def test_each_reader_of_an_upgraded_store_sees_exactly_the_scopes_it_should(
        tag: str, tmp_path: pathlib.Path) -> None:
    """Version 12 re-derived the hash that partitions slots by tenant, user and project,
    and a read must still see a claim exactly when its scope allows. Every live claim is
    asked for by id, by every reader. A reader that must not see it must not find it by
    searching for its text either. Search is not used to show that a reader does see a
    claim: its top ten can leave out a real match, while `get` cannot."""
    db = golden.unpack(tag, tmp_path)
    live = golden.live(golden.load(tag)["data"]["claims"])
    for project in by_project(live):
        readers = [name for name in READERS if (name == "project") == (project is not None)]
        with stores.file(db, project=project) as mem:
            for reader in readers:
                for row in live:
                    should = written_in(row) in SEES[reader]
                    got = mem.get(row["id"], **READERS[reader])
                    assert (got is not None) == should, (
                        f"a {reader} reader {'cannot' if should else 'can'} get the "
                        f"{written_in(row)} claim {row['object']!r}")
                    if not should:
                        found = [r.claim.id for r in mem.search(row["object"], k=10,
                                                                **READERS[reader])]
                        assert row["id"] not in found, (
                            f"a {reader} reader found the {written_in(row)} claim "
                            f"{row['object']!r}")


@pytest.mark.parametrize("tag", golden.TAGS)
def test_an_upgraded_store_recognises_the_facts_it_already_holds(
        tag: str, tmp_path: pathlib.Path) -> None:
    """Versions 6, 12 and 16 re-derive every claim's entity keys and both hashes, which
    the golden dump leaves out. This checks what the value hash is for, in every scope the
    fixture writes in: restating a stored value in its own scope reinforces the stored
    claim instead of adding a second one."""
    db = golden.unpack(tag, tmp_path)
    live = golden.live(golden.load(tag)["data"]["claims"])
    for project in by_project(live):
        with stores.file(db, project=project) as mem:
            for row in live:
                if row["project"] != project:
                    continue
                receipt = mem.remember(row["subject"], row["predicate"], row["object"],
                                       **golden.scope(row))
                assert ([c.id for c in receipt.reinforced], receipt.added) == (
                    [row["id"]], []), (
                    f"restating the {written_in(row)} claim {row['predicate']} "
                    f"{row['object']!r} did not reinforce {row['id']}")


@pytest.mark.parametrize("tag", golden.TAGS)
def test_a_new_value_in_an_upgraded_store_ends_only_its_own_scopes_value(
        tag: str, tmp_path: pathlib.Path) -> None:
    """Version 12 mixes the tenant, user and project into the hash that names a slot. A
    new home, written in each scope that has one, must end that scope's home and nothing
    else. This runs on a store nothing has restated: a write re-saves the claim it
    touches with freshly computed keys, which would hide a wrong hash the migration
    stored."""
    db = golden.unpack(tag, tmp_path)
    live = golden.live(golden.load(tag)["data"]["claims"])
    for project in by_project(live):
        with stores.file(db, project=project) as mem:
            for home in live:
                if home["project"] != project or home["predicate"] != "lives_in":
                    continue
                receipt = mem.remember("user", "lives_in", "Porto", **golden.scope(home))
                assert [c.id for c in receipt.ended] == [home["id"]], (
                    f"a new home in the {written_in(home)} scope ended "
                    f"{[c.object for c in receipt.ended]}, not {home['object']!r}")


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
