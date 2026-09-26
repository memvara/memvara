"""The upgrade tests that need the release tags, or are too slow for every pull request.

They rebuild each committed store from its tag and compare the two, check that every
schema version a release shipped has a committed store, and kill a child process in the
middle of a migration. Two more cover what a user upgrading is likeliest to have: an
encrypted store, which the MCP server creates by default, and a store whose last process
was killed with writes still in its write-ahead log.

They need a clone with the release tags (`git fetch --tags`), and fail when it has none.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

from memvara.store.sqlite import SCHEMA_VERSION

from harness import stores
from harness.crash import Child, after_crash, kill_at
from harness.env import child_env
from harness.invariants import check_store_integrity

from .. import build_stores, golden

#: The committed stores older than this code, which its first open migrates.
OLD = tuple(tag for tag, version in golden.RELEASES.items() if version < SCHEMA_VERSION)
#: The key of the encrypted store: a test key, used for nothing else.
KEY = bytes(range(32))


def released(home: pathlib.Path) -> dict[int, str]:
    """The first release tag that shipped each schema version, read from the tags."""
    env = child_env(home)

    def git(*args: str) -> str:
        done = build_stores._git(*args, env=env)
        assert done.returncode == 0, (
            f"git {' '.join(args)} failed: {done.stderr.decode(errors='replace')}")
        return done.stdout.decode()

    first: dict[int, str] = {}
    for tag in git("tag", "--list", "v*", "--sort=v:refname").split():
        found = re.search(r"^SCHEMA_VERSION = (\d+)",
                          git("show", f"{tag}:memvara/store/sqlite.py"), re.M)
        if found:
            first.setdefault(int(found.group(1)), tag)
    return first


def test_every_released_schema_version_has_a_committed_store(home: pathlib.Path) -> None:
    first = released(home)
    assert first, "this clone has no release tags; `git fetch --tags` fetches them"
    assert {tag: version for version, tag in first.items()} == golden.RELEASES, (
        "a release shipped a schema version with no committed store; add the release "
        "to golden.RELEASES and run tests/adversarial/upgrade/build_stores.py")


@pytest.mark.parametrize("tag", golden.TAGS)
def test_a_store_rebuilt_from_its_tag_matches_the_committed_one(
        tag: str, tmp_path: pathlib.Path, home: pathlib.Path) -> None:
    root = tmp_path / "rebuilt"
    rebuilt = build_stores.build(tag, root, env=child_env(home))
    committed = golden.load(tag)
    assert (rebuilt["commit"], rebuilt["schema_version"]) == (
        committed["commit"], committed["schema_version"])
    assert golden.compare(golden.mask_clock(committed["data"]),
                          golden.mask_clock(rebuilt["data"])) == []
    ours = golden.unpack(tag, tmp_path / "committed")
    theirs = golden.unpack(tag, tmp_path / "fresh", root=root)
    assert golden.snapshot(ours)["schema"] == golden.snapshot(theirs)["schema"]
    record = golden.RECORD
    assert (ours.parent / record).read_bytes() == (theirs.parent / record).read_bytes()


def test_a_build_that_fails_leaves_the_committed_store_as_it_was(
        tmp_path: pathlib.Path, home: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The builder checks the schema version the release wrote only after the release
    has run. A failed check must not have replaced the committed files already, or the
    store and its golden record would no longer match."""
    tag = golden.TAGS[-1]
    root = tmp_path / "root"
    committed = golden.FIXTURES / tag
    (root / tag).mkdir(parents=True)
    for item in committed.iterdir():
        (root / tag / item.name).write_bytes(item.read_bytes())
    monkeypatch.setitem(golden.RELEASES, tag, golden.RELEASES[tag] + 1)
    with pytest.raises(build_stores.BuildError, match="golden.RELEASES says"):
        build_stores.build(tag, root, env=child_env(home))
    assert {item.name: item.read_bytes() for item in (root / tag).iterdir()} == {
        item.name: item.read_bytes() for item in committed.iterdir()}


@pytest.mark.parametrize("tag", OLD)
def test_a_kill_during_the_upgrade_of_an_old_store_leaves_it_whole(
        tag: str, tmp_path: pathlib.Path, home: pathlib.Path) -> None:
    db = golden.unpack(tag, tmp_path)
    record = golden.load(tag)
    before = golden.snapshot(db)
    kill_at({"db": str(db), "user": golden.USER, "setup": [],
             "point": "between-migrations", "action": ["open", {}], "hold": False}, home)
    # All or nothing: the killed migration committed nothing. Every open runs the
    # schema's CREATE TABLE IF NOT EXISTS statements before it migrates, and those commit
    # at once, so a table the upgrade adds may exist, but it must be empty.
    after = golden.snapshot(db)
    assert after["user_version"] == before["user_version"] == record["schema_version"]
    assert [name for name, rows in before["rows"].items()
            if after["rows"].get(name) != rows] == []
    assert [name for name, rows in after["rows"].items()
            if name not in before["rows"] and rows != golden.EMPTY] == []
    assert after["files"] == before["files"]
    assert golden.compare(record["data"], golden.dump(db)) == []
    stores.file(db).close()
    assert golden.schema_version(db) == SCHEMA_VERSION
    assert golden.compare(record["data"], golden.dump(db)) == []
    live = {row["id"]: row["object"] for row in golden.live(record["data"]["claims"])
            if golden.in_default_scope(row)}
    after_crash(db, golden.USER, live).close()


@pytest.mark.parametrize("tag", OLD)
def test_writes_an_old_release_left_in_its_write_ahead_log_survive_the_upgrade(
        tag: str, tmp_path: pathlib.Path, home: pathlib.Path) -> None:
    """The release's process commits a write and is killed before it checkpoints, so
    the write exists only in `-wal` when this code first opens the store."""
    db = golden.unpack(tag, tmp_path / "store")
    release = build_stores.extract(tag, tmp_path / "release", env=child_env(home))
    program = {"db": str(db), "user": golden.USER, "setup": [], "point": "after-commit",
               "action": ["remember", {"predicate": "visited", "object": "Porto"}],
               "hold": False}
    with Child(program, home=home, env={"PYTHONPATH": str(release)}) as child:
        done = json.loads(child.wait_for("DONE"))
        child.wait_for("POINT after-commit")
        child.kill()
    assert db.with_name(db.name + "-wal").stat().st_size > 0
    porto = done["ids"][0]
    # This code's first open recovers the log and migrates. Only then do the tables the
    # integrity checks read exist in a store older than version 8.
    stores.file(db).close()
    assert golden.schema_version(db) == SCHEMA_VERSION
    record = golden.load(tag)
    live = {row["id"]: row["object"] for row in golden.live(record["data"]["claims"])
            if golden.in_default_scope(row)}
    after_crash(db, golden.USER, {**live, porto: "Porto"}).close()


def test_an_encrypted_store_from_the_first_release_with_encryption_upgrades(
        tmp_path: pathlib.Path, home: pathlib.Path) -> None:
    tag = "v0.15.0"
    db = tmp_path / "store" / golden.DB
    db.parent.mkdir()
    build_stores.write_with_release(
        tag, db, env=child_env(home, {"MEMVARA_DB_KEY": KEY.hex()}), encrypted=True)
    assert golden.schema_version(db, key=KEY) == golden.RELEASES[tag]
    before = golden.dump(db, key=KEY)
    key_env = {"MEMVARA_DB_KEY": KEY.hex()}
    with stores.file(db, encryption=True, key_env=key_env) as mem:
        for row in golden.live(before["claims"]):
            if golden.in_default_scope(row):
                found = [r.claim.id for r in mem.search(row["object"], k=10,
                                                        user=golden.USER)]
                assert row["id"] in found
    assert golden.schema_version(db, key=KEY) == SCHEMA_VERSION
    assert golden.compare(before, golden.dump(db, key=KEY)) == []
    assert check_store_integrity(db, key=KEY) == []
