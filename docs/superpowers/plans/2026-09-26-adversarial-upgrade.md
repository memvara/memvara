# Stores from old releases (A6) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show that a store written by any memvara release opens with today's code, migrates, keeps everything it held, and opens a second time without changing, and that a store this code cannot read is refused with a clear message instead of a traceback.

**Architecture:**
- `tests/fixtures/stores/<tag>/` holds one small store for each schema version a release has shipped, written by that release's own code, with a `golden.json` that records what the store holds.
- `tests/adversarial/upgrade/build_stores.py` builds them. It extracts a release's `memvara` package with `git archive`, runs itself again as a child process with `PYTHONPATH` set to that release, and writes a fixed program of memory operations with the hashing embedder and no model. It then reads the closed store with `sqlite3` alone and writes the fixture.
- `tests/adversarial/upgrade/golden.py` is the dump both sides use: the builder under the old release's store, and the tests under the migrated one. It reads only what a migration must keep, so the two dumps must be equal.
- The fast tests unpack a copy of each store into a temporary directory, open it with today's code, and compare. The nightly tests rebuild each store from its tag and compare, and kill a child process in the middle of a migration.
- No library code changes.

**Tech Stack:** Python 3.10–3.13, pytest, `sqlite3`, `gzip`, `git archive`, `harness.env.child_env`, `harness.stores.file`, `harness.crash`, `harness.invariants.check_store_integrity`.

**Spec:** `docs/superpowers/specs/2026-09-25-adversarial-test-suite-design.md`, the A6 row of "Phase 2: The agent-facing deterministic tiers", the "Upgrade" rows of "First targets", and done criterion 9, "Every released schema version upgrades without loss."

## Global Constraints

- Everything runs offline, with no API key. Stores use `HashingEmbedder(dim=512)` and `NullLLM`, as `harness.stores.file` and a server started with `child_env` do.
- Every child process, including `git`, gets its environment from `harness.env.child_env`.
- One committed store per distinct schema version since v0.1.0, at most 256 KB each including its side files (`.vecs`, `.embedder.json`).
- No test changes a committed fixture: every test unpacks a copy into its temporary directory first.
- The fast tier of this workstream takes about 5 seconds.
- A bug these tests find is pinned as a strict expected failure that cites its issue and accepts only its own symptom, which is the suite's rule. This PR pins three that way, B22, B23 and B24, in `tests/adversarial/upgrade/test_adv_upgrade_known_bugs.py`. A bug in the scope of `SECURITY.md` is written in no committed file.
- A test that meets documented behaviour asserts that behaviour and cites where it is documented.
- Test files are named `test_adv_*.py`; every folder has an `__init__.py`; nightly tests live in `tests/adversarial/upgrade/nightly/`.
- No AI attribution and no AI or model name in any commit, code, comment or document.
- Write plainly: every sentence must be understood on its first reading.

## What the exploration found

These facts come from reading the tags and running each release's code before this plan was written.

- **The released schema versions.** v0.1.0 wrote version 5; v0.2.0 version 6; v0.3.0 to v0.9.0 version 8; v0.10.0 to v0.11.3 version 9; v0.12.0 to v0.14.0 version 12; v0.15.0 version 15; v0.16.0 version 16, which is also today's `SCHEMA_VERSION`. Versions 7, 10, 11, 13 and 14 never shipped. The fixtures use the first release of each version: v0.1.0, v0.2.0, v0.3.0, v0.10.0, v0.12.0, v0.15.0 and v0.16.0. All seven import and run under Python 3.13.
- **Size.** Uncompressed, a store is 750 to 870 KB. At SQLite's default 4 KB page size, today's schema alone takes about 230 KB for an almost empty database, and the vector file reserves room for 256 vectors, which is 512 KB at width 512. Compressed with gzip, each store is 16 to 24 KB. The fixtures are therefore committed compressed; the alternative, a smaller page size or embedder width, would make them unlike any store a real user has.
- **Reproducible builds.** Every release mints claim, episode and document ids through `uuid.uuid4`, so the child replaces it with a seeded generator before memvara is imported, and two builds then have the same ids. Two builds of v0.16.0 then differed only in instants the release read from the clock: a fast-path claim's `recorded_at`, a retraction's two closing instants, `erased_at`, and a link's and a document's times. Every instant the writer passes is a whole day in 2024 or 1 January 2100, so `golden.mask_clock` can hide exactly the others.
- **Erased words before version 7.** A v0.1.0 store keeps an erased claim's words as live rows of `claims_fts_data`, but only when the erasure is the last write; a later write merges the index and drops them. The writer therefore erases last. Today's first open clears those rows. Pages the old release freed still hold the words until one `VACUUM`, which `docs/UPGRADING.md` documents ("Erasure now actually removes the text, and the schema is version 7"), and that `VACUUM` removes them.
- **The newer-version refusal.** `Memvara(path)` on a store stamped newer than this build raises `RuntimeError` naming both versions. `python -m memvara.server` on the same store exits with status 1 and a full Python traceback, as the design predicted, because `memvara/server/cli.py` catches `ConfigError` and `EmbedderMismatchError` but not this `RuntimeError`.
- **The embedder refusal.** An old store opened with an embedder of another width is refused with `EmbedderMismatchError`, but only after the migration has committed: the refused v0.1.0 store is already at version 16. The docstrings of `Memvara._check_embedder` and `Memvara._unasked_swap_hint` say the refusal comes "before anything writes".
- **Encrypted stores.** v0.15.0 is the first release that wrote encrypted stores, and the MCP server creates new stores encrypted, so an encrypted version 15 store is the likeliest one a real user upgrades. It cannot be committed within the size limit, because encrypted pages do not compress, so the nightly tier builds one from the tag.

## Review Focus

These are the conditions the spec implies but does not name, most likely to bite first. Each has a test in the task named.

1. **An encrypted store written by v0.15.0**, the MCP server's default kind of store, must upgrade without loss when opened with its key. Task 5.
2. **A store whose release was killed with committed writes still in its write-ahead log** must upgrade with those writes. Task 5.
3. **Accented names** (Zürich, São Paulo, Kraków) and names the entity fold changed ("C++", "C#", "the the band") must be recognised after versions 6, 12 and 16 re-derive the keys: restating one must reinforce the stored claim rather than add a second one. Task 3.
4. **A reader in one scope must not see another scope's claims after the upgrade**, because version 12 re-derives the hash that partitions slots by tenant and user. Task 3.
5. **A refusal must leave the file as it was**, so that the release that wrote it can still open it. Task 4.

## Files

| File | Responsibility |
|---|---|
| `tests/adversarial/upgrade/__init__.py` | Makes the folder a package |
| `tests/adversarial/upgrade/conftest.py` | The `home` fixture for child processes |
| `tests/adversarial/upgrade/golden.py` | The list of fixtures, unpacking, the dump, the snapshot, comparing, masking the clock |
| `tests/adversarial/upgrade/build_stores.py` | The builder: extract a tag, run the writer under it, write the fixture |
| `tests/adversarial/upgrade/test_adv_upgrade_golden.py` | Shows `golden`'s comparisons catching what they must |
| `tests/adversarial/upgrade/test_adv_upgrade_stores.py` | The fast upgrade tests |
| `tests/adversarial/upgrade/test_adv_upgrade_refusals.py` | The fast refusal tests |
| `tests/adversarial/upgrade/nightly/__init__.py` | Makes the nightly folder a package |
| `tests/adversarial/upgrade/nightly/test_adv_upgrade_nightly.py` | Rebuilds, kills during migration, the write-ahead log and the encrypted store |
| `tests/fixtures/stores/<tag>/` | `store.db.gz`, `store.db.vecs.gz`, `store.db.embedder.json`, `golden.json`, for seven tags |
| `docs/claude/testing.md` | A section "Stores from old releases", before the final `Next:` line |

---

### Task 1: The golden dump

**Files:**
- Create: `tests/adversarial/upgrade/__init__.py`, `tests/adversarial/upgrade/conftest.py`, `tests/adversarial/upgrade/golden.py`
- Test: `tests/adversarial/upgrade/test_adv_upgrade_golden.py`

**Interfaces:**
- Produces: `golden.FIXTURES`, `RELEASES: dict[str, int]`, `TAGS`, `DB`, `COMPRESSED`, `RECORD`, `USER`, `PROJECT`, `ERASED_WORD`, `T0`, `FAR_FUTURE`, `CLOCK`, `at(day) -> datetime`, `TABLES`, `unpack(tag, dest, *, root=FIXTURES) -> Path`, `load(tag, *, root=FIXTURES) -> dict`, `connect(db, key=None)`, `schema_version(db, *, key=None) -> int`, `instant(float | None) -> str | None`, `iso(datetime | None) -> str | None`, `dump(db, *, key=None) -> dict[str, list[dict]]`, `compare(expected, actual) -> list[str]`, `is_scripted(str) -> bool`, `mask_clock(data) -> dict`, `snapshot(db, *, key=None) -> dict`, `changes(before, after) -> list[str]`, `tables_holding(db, word, *, key=None) -> list[str]`, `live(claims, *, now=None) -> list[dict]`, `scope(row) -> dict`, `in_default_scope(row) -> bool`. The fixture `home(tmp_path) -> Path`.

- [ ] **Step 1: Write the failing tests**

`tests/adversarial/upgrade/test_adv_upgrade_golden.py`:

```python
"""The comparisons the upgrade tests rely on, each shown catching what it must.

`golden.dump` reads a store with `sqlite3` alone, and the upgrade tests compare two dumps
of one store, taken before and after this code migrated it. If `compare` or `changes`
missed a difference, every upgrade test would pass whatever the migration did, so each
is shown here finding one.
"""

from __future__ import annotations

import pathlib
import sqlite3

from harness import stores

from . import golden


def written(path: pathlib.Path) -> pathlib.Path:
    """A closed store written by this checkout, holding one claim and one turn."""
    with stores.file(path) as mem:
        mem.remember("user", "lives_in", "Berlin", user=golden.USER,
                     valid_from=golden.at(0), recorded_at=golden.at(0))
        mem.add("The quarterly report is due on Friday.", user=golden.USER,
                ts=golden.at(1))
    return path


def test_a_dump_reads_a_table_or_column_the_store_lacks_as_empty(
        tmp_path: pathlib.Path) -> None:
    db = written(tmp_path / "s.db")
    raw = sqlite3.connect(db)
    raw.execute("DROP TABLE claim_links")
    raw.execute("DROP INDEX cl_expiry")  # SQLite refuses to drop an indexed column
    raw.execute("ALTER TABLE claims DROP COLUMN expires_at")
    raw.commit()
    raw.close()
    data = golden.dump(db)
    assert data["claim_links"] == []
    assert [row["expires_at"] for row in data["claims"]] == [None]
    assert [row["valid_from"] for row in data["claims"]] == ["2024-01-01T00:00:00+00:00"]


def test_compare_names_a_changed_field_a_missing_row_and_a_new_row(
        tmp_path: pathlib.Path) -> None:
    db = written(tmp_path / "s.db")
    before = golden.dump(db)
    claim = before["claims"][0]["id"]
    raw = sqlite3.connect(db)
    raw.execute("UPDATE claims SET valid_to = valid_from WHERE id = ?", (claim,))
    raw.execute("DELETE FROM episodes")
    raw.execute("INSERT INTO claim_sources VALUES ('ep_x', ?)", (claim,))
    raw.commit()
    raw.close()
    found = golden.compare(before, golden.dump(db))
    assert f"claims {claim}: valid_to was None, now '2024-01-01T00:00:00+00:00'" in found
    assert any(line.startswith("episodes ") and line.endswith(" is missing")
               for line in found)
    assert f"claim_sources ep_x/{claim} is new" in found
    assert golden.compare(before, before) == []


def test_changes_names_a_changed_row_a_changed_schema_and_a_changed_side_file(
        tmp_path: pathlib.Path) -> None:
    db = written(tmp_path / "s.db")
    before = golden.snapshot(db)
    assert golden.changes(before, golden.snapshot(db)) == []
    raw = sqlite3.connect(db)
    raw.execute("UPDATE claims SET salience = 0.5")
    raw.execute("CREATE INDEX extra ON claims(object)")
    raw.execute("PRAGMA user_version = 99")
    raw.commit()
    raw.close()
    record = tmp_path / "s.db.embedder.json"
    record.write_text(record.read_text() + " ")
    found = golden.changes(before, golden.snapshot(db))
    assert "the rows of claims changed" in found
    assert any(line.startswith("the schema gained ") and "extra" in line for line in found)
    assert "the side file .embedder.json changed" in found
    assert f"the schema version went from {before['user_version']} to 99" in found


def test_only_instants_the_writer_did_not_choose_are_masked() -> None:
    assert golden.is_scripted("2024-01-01T00:00:00+00:00")
    assert golden.is_scripted("2024-12-31T00:00:00+00:00")
    assert golden.is_scripted("2100-01-01T00:00:00+00:00")
    assert not golden.is_scripted("2024-01-01T00:00:01+00:00")
    assert not golden.is_scripted("2026-09-26T08:15:02.123456+00:00")
    data = {"claims": [dict.fromkeys(golden.TABLES["claims"].columns)]}
    data["claims"][0].update(id="cl_1", valid_from="2024-02-01T00:00:00+00:00",
                             recorded_at="2026-09-26T08:15:02.123456+00:00")
    masked = golden.mask_clock(data)["claims"][0]
    assert masked["valid_from"] == "2024-02-01T00:00:00+00:00"
    assert masked["recorded_at"] == golden.CLOCK
    assert data["claims"][0]["recorded_at"] != golden.CLOCK, "the input was changed"


def test_tables_holding_finds_a_word_in_text_and_in_bytes(tmp_path: pathlib.Path) -> None:
    db = written(tmp_path / "s.db")
    raw = sqlite3.connect(db)
    raw.execute("CREATE TABLE blobs (b BLOB)")
    raw.execute("INSERT INTO blobs VALUES (?)", (b"\x00quartz\x00",))
    raw.commit()
    raw.close()
    assert golden.tables_holding(db, "quartz") == ["blobs"]
    assert "claims" in golden.tables_holding(db, "Berlin")
```

- [ ] **Step 2: Run them to see them fail**

Run: `PYTHONPATH=$PWD TMPDIR=<own tmp> <python> -m pytest -q -p no:cacheprovider tests/adversarial/upgrade/test_adv_upgrade_golden.py`
Expected: every test fails with `ImportError: cannot import name 'golden'`.

- [ ] **Step 3: Write `golden.py`, the package file and the conftest**

`tests/adversarial/upgrade/__init__.py`:

```python
"""Stores written by old releases, opened by this checkout (docs/claude/testing.md)."""
```

`tests/adversarial/upgrade/conftest.py`:

```python
"""Fixtures shared by the upgrade tests."""

from __future__ import annotations

import pathlib

import pytest


@pytest.fixture()
def home(tmp_path: pathlib.Path) -> pathlib.Path:
    """A home directory of the test's own, for the child processes it starts."""
    path = tmp_path / "home"
    path.mkdir()
    return path
```

`tests/adversarial/upgrade/golden.py`:

```python
"""What an upgrade test compares: a store's contents, read with `sqlite3` alone.

A committed store under `tests/fixtures/stores/<tag>/` was written by release `<tag>`
(`build_stores.py` says how), and its `golden.json` holds `dump` of that store, taken
before anything else opened it. An upgrade test opens a copy with this checkout's code,
which migrates it, and dumps it again. The two dumps must be equal, because no migration
may change what a store holds.

`dump` reads what a migration must keep: every claim with its scope, text, both clocks
and sources, every episode, and the provenance edges, erasure records, links, documents,
entities and predicates. It leaves out what a migration recomputes on purpose: the entity
keys and the two hashes that versions 6, 12 and 16 re-derive, and the two type columns
version 12 derives from the keys. It also leaves out the vectors, which the integrity
checks and the search tests cover. A table or column the store's version does not have
reads as an empty list or as `None`, so a store dumps the same before and after its
migration.

Nothing here imports memvara, so the builder can dump a store any release wrote. Nothing
here writes to a committed file: the tests `unpack` a copy first.
"""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import pathlib
import shutil
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

#: tests/fixtures/stores, where the committed stores live.
FIXTURES = pathlib.Path(__file__).resolve().parents[2] / "fixtures" / "stores"

#: The first release that wrote each schema version, oldest first, with that version.
#: One store per version is enough, because the releases in between write the same
#: tables. Versions 7, 10, 11, 13 and 14 were never in a release.
RELEASES: dict[str, int] = {"v0.1.0": 5, "v0.2.0": 6, "v0.3.0": 8, "v0.10.0": 9,
                            "v0.12.0": 12, "v0.15.0": 15, "v0.16.0": 16}
TAGS = tuple(RELEASES)

#: The database's file name in every fixture, and the files committed gzip-compressed.
#: The embedder record `RECORD` is a few bytes of text and is committed as it is.
DB = "store.db"
COMPRESSED = (DB, DB + ".vecs")
RECORD = DB + ".embedder.json"

#: The user every claim belongs to unless the writer names another scope.
USER = "u1"
#: The project a claim is recorded against in stores from 0.12.0 on.
PROJECT = "repo-one"
#: The word in the one claim the writer erases, as the last of its writes.
ERASED_WORD = "xylophonequartz"
#: Every instant the writer passes is a whole number of days after T0 within one year,
#: or FAR_FUTURE. Any other instant in a store was read from the clock.
T0 = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
FAR_FUTURE = dt.datetime(2100, 1, 1, tzinfo=dt.timezone.utc)
#: What `mask_clock` writes in place of an instant read from the clock.
CLOCK = "<clock>"


def at(day: int) -> dt.datetime:
    """The instant `day` whole days after T0."""
    return T0 + dt.timedelta(days=day)


@dataclass(frozen=True)
class Table:
    """How `dump` reads one table: the columns it keeps, the ones that order the rows,
    the ones that hold instants as seconds since the epoch, and the ones that hold JSON."""

    columns: tuple[str, ...]
    key: tuple[str, ...]
    instants: tuple[str, ...] = ()
    json: tuple[str, ...] = ()


TABLES: dict[str, Table] = {
    "claims": Table(
        ("id", "tenant", "usr", "agent", "session", "project", "subject", "predicate",
         "object", "text", "polarity", "memory_type", "confidence", "salience",
         "obs_count", "derivation", "extractor", "meta", "sources", "valid_from",
         "valid_to", "recorded_at", "invalidated_at", "invalidated_by",
         "temporal_precision", "amount", "unit", "object_kind", "expires_at",
         "expire_reason"),
        key=("id",),
        instants=("valid_from", "valid_to", "recorded_at", "invalidated_at",
                  "expires_at"),
        json=("meta", "sources")),
    "episodes": Table(
        ("id", "tenant", "usr", "agent", "session", "project", "role", "content", "ts",
         "hash", "meta"),
        key=("id",), instants=("ts",), json=("meta",)),
    "claim_sources": Table(("episode_id", "claim_id"), key=("episode_id", "claim_id")),
    "erasures": Table(
        ("claim_id", "tenant", "scope", "erased_at", "sources", "counts"),
        key=("claim_id", "erased_at"), instants=("erased_at",), json=("counts",)),
    "claim_links": Table(
        ("tenant", "from_id", "to_id", "relation", "created_at", "by"),
        key=("tenant", "from_id", "to_id", "relation"), instants=("created_at",)),
    "documents": Table(
        ("tenant", "id", "custom_id", "scope_key", "usr", "agent", "session", "project",
         "title", "filepath", "source_uri", "mime", "content_hash", "status", "error",
         "meta", "created_at", "updated_at"),
        key=("tenant", "id"), instants=("created_at", "updated_at"), json=("meta",)),
    "document_chunks": Table(
        ("tenant", "document_id", "position", "hash", "episode_id"),
        key=("tenant", "document_id", "position")),
    "entities": Table(("tenant", "id", "canonical", "aliases"), key=("tenant", "id"),
                      json=("aliases",)),
    "predicates": Table(
        ("tenant", "name", "cardinality", "volatility", "memory_type", "aliases",
         "supersedes", "learned"),
        key=("tenant", "name"), json=("aliases", "supersedes")),
}


def unpack(tag: str, dest: pathlib.Path, *,
           root: pathlib.Path = FIXTURES) -> pathlib.Path:
    """Copy `tag`'s store into the directory `dest`, decompressed, and return the path
    of its database. The committed files are only read."""
    source = root / tag
    dest.mkdir(parents=True, exist_ok=True)
    for name in COMPRESSED:
        (dest / name).write_bytes(gzip.decompress((source / f"{name}.gz").read_bytes()))
    shutil.copyfile(source / RECORD, dest / RECORD)
    return dest / DB


def load(tag: str, *, root: pathlib.Path = FIXTURES) -> dict[str, Any]:
    """The golden record committed with `tag`'s store: the tag, the commit it names,
    the schema version the store was written with, and `dump` of the store as `data`."""
    record: dict[str, Any] = json.loads(
        (root / tag / "golden.json").read_text(encoding="utf-8"))
    return record


def connect(db: pathlib.Path, key: bytes | None = None) -> Any:
    """A connection to the store at `db`, through `sqlcipher3` when `key` is the key
    of an encrypted store. The functions here only read through it."""
    if key is None:
        return sqlite3.connect(db)
    import sqlcipher3  # type: ignore[import-untyped,import-not-found,unused-ignore]  # noqa: PLC0415
    conn = sqlcipher3.connect(str(db))
    # The raw-key form memvara itself uses, so no password stretching is applied.
    conn.execute(f"PRAGMA key = \"x'{key.hex()}'\"")
    return conn


def schema_version(db: pathlib.Path, *, key: bytes | None = None) -> int:
    """The schema version stamped in the store's file."""
    conn = connect(db, key)
    try:
        return int(conn.execute("PRAGMA user_version").fetchone()[0])
    finally:
        conn.close()


def instant(value: float | None) -> str | None:
    """A stored instant, seconds since the epoch, as ISO 8601 in UTC."""
    if value is None:
        return None
    return dt.datetime.fromtimestamp(value, dt.timezone.utc).isoformat()


def iso(value: dt.datetime | None) -> str | None:
    """An instant memvara returned, in the form `dump` writes instants."""
    return None if value is None else value.astimezone(dt.timezone.utc).isoformat()


def dump(db: pathlib.Path, *, key: bytes | None = None) -> dict[str, list[dict[str, Any]]]:
    """What the store holds, table by table, in the form `golden.json` records."""
    conn = connect(db, key)
    try:
        return {name: _rows(conn, name, table) for name, table in TABLES.items()}
    finally:
        conn.close()


def _rows(conn: Any, name: str, table: Table) -> list[dict[str, Any]]:
    present = {row[1] for row in conn.execute(f'PRAGMA table_info("{name}")')}
    wanted = [column for column in table.columns if column in present]
    if not wanted:
        return []  # a table this version does not have holds nothing
    rows = []
    for values in conn.execute(f'SELECT {", ".join(wanted)} FROM "{name}"'):
        row: dict[str, Any] = dict.fromkeys(table.columns)
        row.update(zip(wanted, values))
        for column in table.instants:
            row[column] = instant(row[column])
        for column in table.json:
            if row[column] is not None:
                row[column] = json.loads(row[column])
        rows.append(row)
    rows.sort(key=lambda row: _key(row, table))
    return rows


def _key(row: Mapping[str, Any], table: Table) -> str:
    return "/".join(str(row[column]) for column in table.key)


def compare(expected: Mapping[str, list[dict[str, Any]]],
            actual: Mapping[str, list[dict[str, Any]]]) -> list[str]:
    """Every difference between two dumps, one sentence each; empty when they agree."""
    problems: list[str] = []
    for name, table in TABLES.items():
        want = {_key(row, table): row for row in expected.get(name, [])}
        got = {_key(row, table): row for row in actual.get(name, [])}
        problems.extend(f"{name} {key} is missing" for key in sorted(want.keys() - got.keys()))
        problems.extend(f"{name} {key} is new" for key in sorted(got.keys() - want.keys()))
        for key in sorted(want.keys() & got.keys()):
            for column in table.columns:
                if want[key][column] != got[key][column]:
                    problems.append(f"{name} {key}: {column} was {want[key][column]!r}, "
                                    f"now {got[key][column]!r}")
    return problems


def is_scripted(value: str) -> bool:
    """Whether an instant is one the writer passes: a whole day after T0 within a year,
    or FAR_FUTURE. Any other instant was read from the clock while the store was
    written, so it differs between two builds."""
    moment = dt.datetime.fromisoformat(value)
    offset = moment - T0
    whole_day = offset % dt.timedelta(days=1) == dt.timedelta(0)
    return moment == FAR_FUTURE or (
        dt.timedelta(0) <= offset < dt.timedelta(days=366) and whole_day)


def mask_clock(data: Mapping[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """A copy of a dump with every instant read from the clock replaced by CLOCK. Two
    builds of one release differ only there, so the nightly rebuild compares these."""
    masked: dict[str, list[dict[str, Any]]] = {}
    for name, table in TABLES.items():
        rows = []
        for original in data.get(name, []):
            row = dict(original)
            for column in table.instants:
                if row[column] is not None and not is_scripted(row[column]):
                    row[column] = CLOCK
            rows.append(row)
        masked[name] = rows
    return masked


def _table_names(conn: Any) -> list[str]:
    return [row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]


def _digest(lines: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def snapshot(db: pathlib.Path, *, key: bytes | None = None) -> dict[str, Any]:
    """Everything an open could change, to compare two opens: the schema version, the
    schema, a digest of every row of every table, and a digest of each side file."""
    conn = connect(db, key)
    try:
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        schema = sorted(repr(tuple(row)) for row in conn.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master"))
        rows = {name: _digest(sorted(repr(tuple(row)) for row in
                                     conn.execute(f'SELECT * FROM "{name}"')))
                for name in _table_names(conn)}
    finally:
        conn.close()
    files: dict[str, str | None] = {}
    for suffix in (".vecs", ".embedder.json"):
        side = db.with_name(db.name + suffix)
        files[suffix] = hashlib.sha256(side.read_bytes()).hexdigest() if side.exists() else None
    return {"user_version": version, "schema": schema, "rows": rows, "files": files}


def changes(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    """What differs between two snapshots, one sentence each; empty when nothing does."""
    problems: list[str] = []
    if before["user_version"] != after["user_version"]:
        problems.append(f"the schema version went from {before['user_version']} to "
                        f"{after['user_version']}")
    old, new = set(before["schema"]), set(after["schema"])
    problems.extend(f"the schema lost {line}" for line in sorted(old - new))
    problems.extend(f"the schema gained {line}" for line in sorted(new - old))
    for name in sorted(before["rows"].keys() | after["rows"].keys()):
        if before["rows"].get(name) != after["rows"].get(name):
            problems.append(f"the rows of {name} changed")
    for suffix in sorted(before["files"]):
        if before["files"][suffix] != after["files"][suffix]:
            problems.append(f"the side file {suffix} changed")
    return problems


def tables_holding(db: pathlib.Path, word: str, *, key: bytes | None = None) -> list[str]:
    """The tables with at least one row that holds `word` in any column, as text or as
    bytes. It reads rows, not the file: see `docs/INTERNALS.md` for reading the file."""
    needle = word.encode()
    conn = connect(db, key)
    try:
        found = []
        for name in _table_names(conn):
            for row in conn.execute(f'SELECT * FROM "{name}"'):
                if any(needle in (value if isinstance(value, bytes) else str(value).encode())
                       for value in row if value is not None):
                    found.append(name)
                    break
        return found
    finally:
        conn.close()


def live(claims: Iterable[Mapping[str, Any]], *,
         now: dt.datetime | None = None) -> list[dict[str, Any]]:
    """The claims a present-tense read returns: asserted, never retired, in force now
    and not expired."""
    moment = now or dt.datetime.now(dt.timezone.utc)

    def later(value: str | None) -> bool:
        return value is None or dt.datetime.fromisoformat(value) > moment

    return [dict(row) for row in claims
            if row["polarity"] == 1 and row["invalidated_at"] is None
            and dt.datetime.fromisoformat(row["valid_from"]) <= moment
            and later(row["valid_to"]) and later(row["expires_at"])]


def scope(row: Mapping[str, Any]) -> dict[str, Any]:
    """The keyword arguments that name a row's scope in a memvara read or write."""
    return {"tenant": row["tenant"], "user": row["usr"], "agent": row["agent"],
            "session": row["session"]}


def in_default_scope(row: Mapping[str, Any]) -> bool:
    """Whether a row belongs to USER in the default tenant, with no agent, session or
    project: the scope a plain `user=USER` read covers."""
    return (row["tenant"] == "default" and row["usr"] == USER and row["agent"] is None
            and row["session"] is None and row["project"] is None)
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `PYTHONPATH=$PWD TMPDIR=<own tmp> <python> -m pytest -q -p no:cacheprovider tests/adversarial/upgrade/test_adv_upgrade_golden.py`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/adversarial/upgrade/__init__.py tests/adversarial/upgrade/conftest.py \
  tests/adversarial/upgrade/golden.py tests/adversarial/upgrade/test_adv_upgrade_golden.py \
  docs/claude/testing.md
git commit -m "Add the dump the upgrade tests compare, and show it catching changes"
```

`docs/claude/testing.md` gets the section header and its first paragraph in this commit (the section's full text is under Task 6).

---

### Task 2: The builder and the seven committed stores

**Files:**
- Create: `tests/adversarial/upgrade/build_stores.py`
- Create: `tests/fixtures/stores/<tag>/{store.db.gz,store.db.vecs.gz,store.db.embedder.json,golden.json}` for the seven tags
- Test: `tests/adversarial/upgrade/test_adv_upgrade_stores.py` (its first test)

**Interfaces:**
- Consumes: everything `golden` produces; `harness.env.child_env`.
- Produces: `build_stores.extract(tag, dest, *, env) -> Path`, `commit_of(tag, *, env) -> str`, `write_with_release(tag, db, *, env, encrypted=False) -> dict`, `build(tag, root, *, env) -> dict`, `write(db, *, encrypted=False) -> dict`, `BuildError`, `main(argv) -> int`.

- [ ] **Step 1: Write the failing test**

`tests/adversarial/upgrade/test_adv_upgrade_stores.py`, first version:

```python
"""Stores written by old releases, opened by this checkout's code.

Each store under `tests/fixtures/stores/<tag>/` was written by release `<tag>`'s own
code, one store for each schema version a release has shipped; `build_stores.py` says
how. Each is committed with `golden.json`, the dump `golden.dump` took of it before
anything else opened it. Every test unpacks a copy into its own temporary directory, so
no test can change a committed file.
"""

from __future__ import annotations

import pathlib

import pytest

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
```

- [ ] **Step 2: Run it to see it fail**

Run: `PYTHONPATH=$PWD TMPDIR=<own tmp> <python> -m pytest -q -p no:cacheprovider tests/adversarial/upgrade/test_adv_upgrade_stores.py`
Expected: 7 failures, each `FileNotFoundError` for `tests/fixtures/stores/<tag>/store.db.gz`.

- [ ] **Step 3: Write the builder**

`tests/adversarial/upgrade/build_stores.py`:

```python
"""Build the committed stores in tests/fixtures/stores/ by running each release's own code.

Run it from the repository root, in a clone that has the release tags:

    python tests/adversarial/upgrade/build_stores.py            # every store
    python tests/adversarial/upgrade/build_stores.py v0.16.0    # one store

For each tag in `golden.RELEASES` it does four things:

1. It extracts the tag's `memvara` package with `git archive` into a temporary directory.
2. It runs this file again as a child process, with `PYTHONPATH` set to that directory,
   so that `write` runs with the release's own code, the hashing embedder and no model.
   The child reports which memvara it imported, and the build stops if that is not the
   release.
3. It compresses the closed store into `tests/fixtures/stores/<tag>/`.
4. It dumps that committed copy with `golden.dump`, which reads with `sqlite3` alone,
   because opening it with memvara would migrate it, and writes the dump to
   `golden.json` beside it.

Two things make a rebuild reproducible, so that the nightly tier can compare it with the
committed store. Claim, episode and document ids come from `uuid.uuid4`, which the child
replaces with a generator seeded with a constant before memvara is imported; every release
mints its ids there, and nothing in memvara reads meaning from an id. And every instant
the program passes is a whole day in 2024, or 1 January 2100. The instants a release
reads from the clock, such as when a fast-path claim was recorded or when a claim was
erased, differ between builds, and `golden.mask_clock` hides exactly those.

The database and the vector file are committed gzip-compressed. Uncompressed, even an
almost empty store is over the fixture size limit of 256 KB: at SQLite's default page
size each of today's sixty or so tables and indexes takes at least one 4 KB page, and
the vector file reserves room for 256 vectors, 512 KB at width 512. Compressed, a store
is about 20 KB. A smaller page size or embedder width would also fit, but would make the
fixtures unlike any store a real user has.
"""

from __future__ import annotations

import gzip
import inspect
import io
import json
import os
import pathlib
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
import uuid
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

try:
    from . import golden
except ImportError:  # run as a script: this file's folder is the first entry of sys.path
    import golden  # type: ignore[no-redef]

#: The repository root, and tests/, which holds the harness package.
REPO = pathlib.Path(__file__).resolve().parents[3]
TESTS = REPO / "tests"
#: Seeds the generator that stands in for `uuid.uuid4`, so a rebuild mints the same ids.
SEED = 20240101
#: How long one release may take to write its store, in seconds.
WRITE_SECONDS = 300


class BuildError(RuntimeError):
    """A store could not be built. The message says why."""


def _git(*args: str, env: Mapping[str, str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                          env=dict(env), check=False)


def extract(tag: str, dest: pathlib.Path, *, env: Mapping[str, str]) -> pathlib.Path:
    """Extract release `tag`'s `memvara` package into `dest`, so that `PYTHONPATH=dest`
    imports that release. Returns `dest`."""
    done = _git("archive", "--format=tar", tag, "memvara", env=env)
    if done.returncode != 0:
        raise BuildError(f"`git archive {tag}` failed. Is the tag in this clone? "
                         f"`git fetch --tags` fetches it.\n"
                         f"{done.stderr.decode(errors='replace')}")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(done.stdout)) as archive:
        if hasattr(tarfile, "data_filter"):
            archive.extractall(dest, filter="data")
        else:  # Python before 3.10.12 has no extraction filters
            archive.extractall(dest)
    return dest


def commit_of(tag: str, *, env: Mapping[str, str]) -> str:
    """The commit `tag` points at."""
    done = _git("rev-parse", f"{tag}^{{commit}}", env=env)
    if done.returncode != 0:
        raise BuildError(f"{tag} is not a tag in this clone")
    return done.stdout.decode().strip()


def write_with_release(tag: str, db: pathlib.Path, *, env: Mapping[str, str],
                       encrypted: bool = False) -> dict[str, Any]:
    """Write the fixture program into a new store at `db` with release `tag`'s own code,
    in a child process, and return what the child reported.

    `env` is the child's environment, from `harness.env.child_env`; its PYTHONPATH is
    replaced with the extracted release. With `encrypted`, the store is encrypted with
    the key in `env`'s MEMVARA_DB_KEY.
    """
    with tempfile.TemporaryDirectory(prefix="memvara-release-") as scratch:
        release = extract(tag, pathlib.Path(scratch) / "release", env=env)
        argv = [sys.executable, str(pathlib.Path(__file__).resolve()), "--write", str(db)]
        if encrypted:
            argv.append("--encrypted")
        done = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                              env={**env, "PYTHONPATH": str(release)}, cwd=scratch,
                              timeout=WRITE_SECONDS, check=False)
        if done.returncode != 0:
            raise BuildError(f"{tag} could not write its store (exit status "
                             f"{done.returncode}):\n{done.stderr[-3000:]}")
        reports = [line for line in done.stdout.splitlines() if line.startswith("WROTE ")]
        if not reports:
            raise BuildError(f"{tag}'s writer printed no WROTE line:\n{done.stdout[-3000:]}")
        info: dict[str, Any] = json.loads(reports[-1][len("WROTE "):])
        imported = pathlib.Path(info["memvara"]).resolve()
        if not imported.is_relative_to(release.resolve()):
            raise BuildError(f"the writer imported memvara from {imported}, not from "
                             f"{tag}; an editable install is shadowing PYTHONPATH")
    return info


def _check_closed(store: pathlib.Path) -> None:
    """The release closed its store: no write-ahead log or shared-memory file is left,
    so the database file holds every write, and both side files exist."""
    known = {*golden.COMPRESSED, golden.RECORD, golden.DB + ".lock"}
    left = sorted(p.name for p in store.iterdir() if p.name not in known)
    if left:
        raise BuildError(f"the release left {left} beside its store, so the database "
                         "file may not hold every write")
    missing = [name for name in (*golden.COMPRESSED, golden.RECORD)
               if not (store / name).exists()]
    if missing:
        raise BuildError(f"the release wrote no {missing}")


def build(tag: str, root: pathlib.Path, *, env: Mapping[str, str]) -> dict[str, Any]:
    """Write release `tag`'s store and its golden record into `root/<tag>/`, replacing
    any that are there, and return the record."""
    if tag not in golden.RELEASES:
        raise BuildError(f"{tag} is not in golden.RELEASES")
    out = root / tag
    with tempfile.TemporaryDirectory(prefix="memvara-store-") as scratch:
        store = pathlib.Path(scratch) / "store"
        store.mkdir()
        info = write_with_release(tag, store / golden.DB, env=env)
        _check_closed(store)
        out.mkdir(parents=True, exist_ok=True)
        for name in golden.COMPRESSED:
            (out / f"{name}.gz").write_bytes(
                gzip.compress((store / name).read_bytes(), compresslevel=9, mtime=0))
        shutil.copyfile(store / golden.RECORD, out / golden.RECORD)
        # Dumped from the committed files, the way every test reads them.
        db = golden.unpack(tag, pathlib.Path(scratch) / "committed", root=root)
        record = {"tag": tag, "commit": commit_of(tag, env=env),
                  "schema_version": golden.schema_version(db), "data": golden.dump(db)}
    if {info["schema"], record["schema_version"]} != {golden.RELEASES[tag]}:
        raise BuildError(f"{tag} wrote schema version {record['schema_version']}, and "
                         f"golden.RELEASES says {golden.RELEASES[tag]}")
    (out / "golden.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n",
                                     encoding="utf-8", newline="\n")
    return record


def write(db: str, *, encrypted: bool = False) -> dict[str, Any]:
    """Write the fixture program into a new store at `db`, with whichever memvara this
    process imports; the builder arranges for that to be one release's own code.

    It runs only in the builder's child process, because it replaces `uuid.uuid4`
    before it imports memvara. A step that needs a feature the release lacks is skipped;
    the release's own signatures say which features it has. Each comment names the
    part of the schema the step gives the migrations to carry.
    """
    generator = random.Random(SEED)
    uuid.uuid4 = lambda: uuid.UUID(int=generator.getrandbits(128), version=4)
    warnings.simplefilter("ignore")  # an old release's warnings say nothing about its store

    import memvara  # noqa: PLC0415 - imported only after uuid4 is replaced
    from memvara import Memvara, NullLLM  # noqa: PLC0415
    from memvara.embed import HashingEmbedder  # noqa: PLC0415
    from memvara.store.sqlite import SCHEMA_VERSION  # noqa: PLC0415

    def takes(method: str, parameter: str) -> bool:
        return parameter in inspect.signature(getattr(Memvara, method)).parameters

    security: dict[str, Any] = {}
    if encrypted:
        security = {"encryption": True,
                    "key_env": {"MEMVARA_DB_KEY": os.environ["MEMVARA_DB_KEY"]}}

    def open_store(**options: Any) -> Any:
        return Memvara(db, embedder=HashingEmbedder(dim=512), llm=NullLLM(), **security,
                       **options)

    mem = open_store()

    def remember(predicate: str, obj: str, day: int, **options: Any) -> str | None:
        """Assert `user <predicate> <obj>`, true and recorded at `day`, for USER unless
        `options` names another scope. The id of the claim stored or reinforced."""
        options.setdefault("user", golden.USER)
        receipt = mem.remember("user", predicate, obj, valid_from=golden.at(day),
                               recorded_at=golden.at(day), **options)
        claims = list(receipt.added) + list(receipt.reinforced)
        return claims[0].id if claims else None

    # A single-valued slot whose first value is superseded.
    remember("lives_in", "Berlin", 0)
    remember("lives_in", "Paris", 30)
    # A multi-valued slot. One value is ended here, and the other is retracted below.
    remember("likes", "green tea", 1)
    coffee = remember("likes", "black coffee", 2)
    mem.delete(coffee, at=golden.at(40), user=golden.USER,
               **({"close": "ended"} if takes("delete", "close") else {}))
    # A retired slot.
    remember("has_pet", "a cat named Tom", 3)
    mem.forget("user", "has_pet", at=golden.at(41), user=golden.USER,
               **({"close": "retired"} if takes("forget", "close") else {}))
    # A claim that cites the turn it came from: the provenance edge version 5 indexes.
    turn = mem.add("We moved the whole team into the new office downtown.",
                   user=golden.USER, ts=golden.at(4))
    remember("works_at", "Acme Corp", 5, sources=[turn.episode_ids[0]])
    # Two turns: the fast path extracts a claim from the first and nothing from the
    # second. The extracted claim's `recorded_at` comes from the clock.
    mem.add("I live in Lisbon", user=golden.USER, ts=golden.at(50))
    mem.add("The quarterly report is due on Friday.", user=golden.USER, ts=golden.at(51))
    # Other scopes: another user, a session, an agent and another tenant.
    remember("lives_in", "Madrid", 7, user="u2")
    remember("prefers", "dark mode", 8, session="s1")
    remember("prefers", "short answers", 9, agent="a1")
    remember("lives_in", "Oslo", 10, tenant="acme")
    # Names the entity fold treats differently from version 16 on, and accented names,
    # all of which versions 6, 12 and 16 re-key.
    for day, language in enumerate(("C++", "C#", "C"), start=11):
        remember("knows_language", language, day)
    for day, place in enumerate(("Zürich", "São Paulo", "Kraków"), start=21):
        remember("visited", place, day)
    remember("likes_band", "the the band", 24)
    # Caller metadata, and a retraction of one value.
    remember("visited", "Rome", 14, note="written by the fixture")
    remember("likes", "green tea", 60, polarity=-1)
    if hasattr(Memvara, "link"):
        # 0.15.0 and later: a link (version 13), a document (14), an expiry (15), and a
        # replacement named by the caller.
        florence = remember("visited", "Florence", 15)
        uffizi = remember("visited", "the Uffizi gallery in Florence", 16)
        mem.link(uffizi, florence, relation="extends", user=golden.USER)
        remember("subscribes_to", "a jazz magazine", 17, expires_at=golden.FAR_FUTURE,
                 expire_reason="the trial ends")
        green = remember("favorite_color", "green", 18)
        remember("favorite_color", "blue", 19, replaces=green, reason="the user said blue")
        mem.add_document("Onboarding notes. The staging database is PostgreSQL 16.",
                         title="Onboarding notes", custom_id="onboarding", extract=False,
                         user=golden.USER)
    mem.close()
    if takes("__init__", "project"):
        # 0.12.0 and later: an undeclared predicate is recorded against the project the
        # store was opened in (version 12).
        mem = open_store(project=golden.PROJECT)
        remember("uses_database", "PostgreSQL", 20)
        mem.close()
    # Last, so that no later write merges the text index. Before version 7 the erased
    # claim's words then stay in the index's shadow table, which is what the version 7
    # migration has to clear.
    mem = open_store()
    secret = remember("codeword", golden.ERASED_WORD, 6)
    if not mem.erase(secret, user=golden.USER):
        raise RuntimeError("the release did not erase the claim it was asked to erase")
    mem.close()
    return {"memvara": memvara.__file__, "schema": SCHEMA_VERSION,
            "version": getattr(memvara, "__version__", None)}


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["--write"]:
        info = write(args[1], encrypted="--encrypted" in args[2:])
        print("WROTE " + json.dumps(info), flush=True)
        return 0
    unknown = [tag for tag in args if tag not in golden.RELEASES]
    if unknown:
        print(f"unknown tags {unknown}; choose from {list(golden.TAGS)}", file=sys.stderr)
        return 2
    sys.path.insert(0, str(TESTS))
    from harness.env import child_env  # noqa: PLC0415 - tests/ is on sys.path only now

    with tempfile.TemporaryDirectory(prefix="memvara-build-home-") as home:
        env = child_env(pathlib.Path(home))
        for tag in args or golden.TAGS:
            record = build(tag, golden.FIXTURES, env=env)
            data = record["data"]
            print(f"{tag}: schema version {record['schema_version']}, "
                  f"{len(data['claims'])} claims, {len(data['episodes'])} episodes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Build every store, twice, and check the second build agrees**

Run: `python tests/adversarial/upgrade/build_stores.py` from the worktree root.
Expected: seven lines such as `v0.1.0: schema version 5, 24 claims, 3 episodes`.

Then build a second copy into a scratch root and compare it with the committed one, to confirm that a rebuild differs only in instants read from the clock (the nightly test in Task 5 makes this permanent). The script, saved under `/private/tmp` and run with `PYTHONPATH` at the worktree and `tests/` on `sys.path`:

```python
import pathlib, sys, tempfile
sys.path.insert(0, "tests")
from adversarial.upgrade import build_stores, golden
from harness.env import child_env

with tempfile.TemporaryDirectory() as scratch:
    home = pathlib.Path(scratch) / "home"
    home.mkdir()
    for tag in golden.TAGS:
        rebuilt = build_stores.build(tag, pathlib.Path(scratch) / "root", env=child_env(home))
        found = golden.compare(golden.mask_clock(golden.load(tag)["data"]),
                               golden.mask_clock(rebuilt["data"]))
        print(tag, "agrees" if not found else found)
```

Expected: seven lines ending in `agrees`. Confirm each fixture directory is under 256 KB with `du -k tests/fixtures/stores/*`.

- [ ] **Step 5: Run the test to see it pass**

Run: `PYTHONPATH=$PWD TMPDIR=<own tmp> <python> -m pytest -q -p no:cacheprovider tests/adversarial/upgrade/test_adv_upgrade_stores.py`
Expected: `7 passed`.

- [ ] **Step 6: Commit**

Every file is named; the loop only spells out the 28 fixture paths.

```bash
files="tests/adversarial/upgrade/build_stores.py tests/adversarial/upgrade/test_adv_upgrade_stores.py docs/claude/testing.md"
for tag in v0.1.0 v0.2.0 v0.3.0 v0.10.0 v0.12.0 v0.15.0 v0.16.0; do
  for name in store.db.gz store.db.vecs.gz store.db.embedder.json golden.json; do
    files="$files tests/fixtures/stores/$tag/$name"
  done
done
git add $files
git commit -m "Add a store written by the first release of each schema version, and the script that builds them"
```

---

### Task 3: The upgrade tests

**Files:**
- Modify: `tests/adversarial/upgrade/test_adv_upgrade_stores.py`

**Interfaces:**
- Consumes: `golden.*`, `harness.stores.file`, `harness.invariants.check_store_integrity`, `memvara.store.sqlite.SCHEMA_VERSION`.

- [ ] **Step 1: Write the tests**

Append to `tests/adversarial/upgrade/test_adv_upgrade_stores.py` (and add `import warnings`, `from memvara.store.sqlite import SCHEMA_VERSION`, `from harness import stores` and `from harness.invariants import check_store_integrity` to its imports):

```python
@pytest.mark.parametrize("tag", golden.TAGS)
def test_a_store_from_an_old_release_upgrades_without_loss_and_only_once(
        tag: str, tmp_path: pathlib.Path) -> None:
    db = golden.unpack(tag, tmp_path)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        stores.file(db).close()
    assert [f"{w.category.__name__}: {w.message}" for w in seen] == [], (
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
            assert mem.get_document(row["id"], tenant=row["tenant"], user=row["usr"]) is not None
        # No reader sees another user's or another tenant's claims (Review Focus 4).
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
        mem.store._db.execute("VACUUM")
    holding = [p.name for p in tmp_path.iterdir()
               if p.is_file() and golden.ERASED_WORD.encode() in p.read_bytes()]
    assert holding == []
```

- [ ] **Step 2: Run them and read each failure**

Run: `PYTHONPATH=$PWD TMPDIR=<own tmp> <python> -m pytest -q -p no:cacheprovider tests/adversarial/upgrade/test_adv_upgrade_stores.py`
Expected: the tests pass against today's code. A failure is either a mistake in the test, which is fixed, or a bug in memvara, which is classified as the common brief says. A bug's failing test stays out of the commit.

- [ ] **Step 3: Show the tests catch what they claim (scratch copies, never committed)**

In a scratch copy of the worktree under `/private/tmp`, make each change below in `memvara/store/sqlite.py` and confirm that the named test fails:
- remove the `optimize` line from `_migrate_to_v7`: the erased-word test fails for v0.1.0 and v0.2.0;
- make `_migrate_to_v12` skip its three `UPDATE`s: the recognising test fails for the stores older than version 12;
- skip `_migrate_to_v15`: the upgrade test fails for every store older than 15;
- change `_migrate` to stamp `SCHEMA_VERSION - 1`: the upgrade test fails for every store.

- [ ] **Step 4: Commit**

```bash
git add tests/adversarial/upgrade/test_adv_upgrade_stores.py docs/claude/testing.md
git commit -m "Open each old release's store with today's code and check nothing is lost"
```

---

### Task 4: The refusals

**Files:**
- Create: `tests/adversarial/upgrade/test_adv_upgrade_refusals.py`

**Interfaces:**
- Consumes: `golden.*`, `harness.stores.file`, `harness.env.child_env`, `memvara.EmbedderMismatchError`, `memvara.EmbedderChangedWarning`.

- [ ] **Step 1: Write the tests**

`tests/adversarial/upgrade/test_adv_upgrade_refusals.py`:

```python
"""Stores this code cannot read are refused, with a message and without a traceback.

Two kinds: a store written by a newer version than this code knows, and a store opened
with an embedder that did not write it. The newer store is made by raising the schema
version of the newest committed store, as a later release's migration would.
"""

from __future__ import annotations

import pathlib
import sqlite3
import subprocess
import sys

import pytest

from memvara import EmbedderChangedWarning, EmbedderMismatchError, Memvara, NullLLM
from memvara.embed import HashingEmbedder
from memvara.store.sqlite import SCHEMA_VERSION

from harness import stores
from harness.env import child_env

from . import golden

NEWER = SCHEMA_VERSION + 1


def newer_store(tmp_path: pathlib.Path) -> pathlib.Path:
    """The newest committed store, stamped with a schema version past this code's."""
    db = golden.unpack(golden.TAGS[-1], tmp_path)
    raw = sqlite3.connect(db)
    raw.execute(f"PRAGMA user_version = {NEWER}")
    raw.commit()
    raw.close()
    return db


def serve(db: pathlib.Path, home: pathlib.Path,
          **extra: str) -> subprocess.CompletedProcess[str]:
    """Start the MCP server on `db` with nothing on its input, and wait for it to exit."""
    return subprocess.run([sys.executable, "-m", "memvara.server"], input="",
                          capture_output=True, text=True, encoding="utf-8",
                          env=child_env(home, {"MEMVARA_DB": str(db), **extra}),
                          timeout=60, check=False)


def test_a_store_from_a_newer_version_is_refused_and_left_as_it_was(
        tmp_path: pathlib.Path) -> None:
    db = newer_store(tmp_path)
    before = golden.snapshot(db)
    for attempt in ("first", "second"):
        with pytest.raises(RuntimeError) as refused:
            stores.file(db)
        message = str(refused.value)
        assert f"schema version {NEWER} was written by a newer Memvara" in message, attempt
        assert f"this build understands {SCHEMA_VERSION}" in message, attempt
    assert golden.changes(before, golden.snapshot(db)) == []


@pytest.mark.parametrize("tag", golden.TAGS)
def test_an_old_store_is_refused_by_an_embedder_of_another_width(
        tag: str, tmp_path: pathlib.Path) -> None:
    db = golden.unpack(tag, tmp_path)
    with pytest.raises(EmbedderMismatchError,
                       match="512-dimensional vectors, written by hashing:512"):
        Memvara(str(db), embedder=HashingEmbedder(dim=256), llm=NullLLM())


@pytest.mark.parametrize("tag", golden.TAGS)
def test_an_old_store_warns_when_another_embedder_of_its_width_opens_it(
        tag: str, tmp_path: pathlib.Path) -> None:
    """Documented: an embedder of the same width but another vector space is warned
    about, not refused, because nothing can raise on it (`EmbedderChangedWarning` and
    `Memvara._check_embedder` in memvara/core.py). The old release's embedder record
    must still be read for the warning to fire."""
    db = golden.unpack(tag, tmp_path)
    with pytest.warns(EmbedderChangedWarning, match="unrelated vector spaces"):
        Memvara(str(db), embedder=HashingEmbedder(dim=512, ngram=(2, 4)),
                llm=NullLLM()).close()


def test_the_server_refuses_an_old_store_with_an_embedder_of_another_width(
        tmp_path: pathlib.Path, home: pathlib.Path) -> None:
    db = golden.unpack(golden.TAGS[0], tmp_path)
    done = serve(db, home, MEMVARA_EMBEDDER="hashing:256")
    assert done.returncode == 2, done.stderr
    assert done.stderr.startswith("memvara-mcp: ")
    assert "512-dimensional" in done.stderr
    assert "Traceback" not in done.stderr
```

Two more tests are written the same way and are expected to fail today:

```python
def test_the_server_refuses_a_store_from_a_newer_version_without_a_traceback(
        tmp_path: pathlib.Path, home: pathlib.Path) -> None:
    db = newer_store(tmp_path)
    done = serve(db, home)
    assert done.returncode == 2, done.stderr
    assert done.stderr.startswith("memvara-mcp: ")
    assert f"schema version {NEWER}" in done.stderr
    assert "Traceback" not in done.stderr


@pytest.mark.parametrize("tag", [t for t, v in golden.RELEASES.items() if v < SCHEMA_VERSION])
def test_a_store_refused_for_its_embedder_is_left_at_its_old_version(
        tag: str, tmp_path: pathlib.Path) -> None:
    db = golden.unpack(tag, tmp_path)
    with pytest.raises(EmbedderMismatchError):
        Memvara(str(db), embedder=HashingEmbedder(dim=256), llm=NullLLM())
    assert golden.schema_version(db) == golden.RELEASES[tag]
```

- [ ] **Step 2: Run them**

Run: `PYTHONPATH=$PWD TMPDIR=<own tmp> <python> -m pytest -q -p no:cacheprovider tests/adversarial/upgrade/test_adv_upgrade_refusals.py`
Expected: every test passes except the last two. The first of those fails because the server exits with status 1 and a traceback; the second fails because the refused store is at version 16. Both are bugs outside `SECURITY.md`'s scope: move the two tests out of this file, report them, and confirm the rest pass. They are pinned as B22 and B23 in `tests/adversarial/upgrade/test_adv_upgrade_known_bugs.py`, beside B24, a connection a refused open leaves open, which the verification step found.

- [ ] **Step 3: Commit**

```bash
git add tests/adversarial/upgrade/test_adv_upgrade_refusals.py docs/claude/testing.md
git commit -m "Check that a store from a newer version or another embedder is refused with a message"
```

---

### Task 5: The nightly tier

**Files:**
- Create: `tests/adversarial/upgrade/nightly/__init__.py`, `tests/adversarial/upgrade/nightly/test_adv_upgrade_nightly.py`

**Interfaces:**
- Consumes: `golden.*`, `build_stores.build`, `build_stores.extract`, `build_stores.write_with_release`, `build_stores.commit_of`, `harness.crash.Child`, `harness.crash.kill_at`, `harness.crash.after_crash`, `harness.env.child_env`, `harness.stores.file`, `harness.invariants.check_store_integrity`.

- [ ] **Step 1: Write the tests**

`tests/adversarial/upgrade/nightly/__init__.py`:

```python
"""The upgrade tests too slow for every pull request, or needing the release tags."""
```

`tests/adversarial/upgrade/nightly/test_adv_upgrade_nightly.py`:

```python
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
import subprocess

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
        return subprocess.run(["git", "-C", str(build_stores.REPO), *args],
                              capture_output=True, text=True, env=env,
                              check=True).stdout

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


@pytest.mark.parametrize("tag", OLD)
def test_a_kill_during_the_upgrade_of_an_old_store_leaves_it_whole(
        tag: str, tmp_path: pathlib.Path, home: pathlib.Path) -> None:
    db = golden.unpack(tag, tmp_path)
    record = golden.load(tag)
    kill_at({"db": str(db), "user": golden.USER, "setup": [],
             "point": "between-migrations", "action": ["open", {}], "hold": False}, home)
    # All or nothing: the killed migration committed nothing.
    assert golden.schema_version(db) == record["schema_version"]
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
```

- [ ] **Step 2: Run them**

Run: `PYTHONPATH=$PWD TMPDIR=<own tmp> <python> -m pytest -q -p no:cacheprovider tests/adversarial/upgrade/nightly --tier nightly`
Expected: every test passes. Then confirm a plain run leaves the folder out: `pytest -q tests/adversarial/upgrade` collects none of them.

- [ ] **Step 3: Commit**

```bash
git add tests/adversarial/upgrade/nightly/__init__.py \
  tests/adversarial/upgrade/nightly/test_adv_upgrade_nightly.py docs/claude/testing.md
git commit -m "Rebuild the old stores from their tags nightly, and kill a child during an upgrade"
```

---

### Task 6: Documentation and verification

**Files:**
- Modify: `docs/claude/testing.md` (the section, completed across Tasks 1 to 5; its final text is below)

The section, placed just before the final line that starts with `Next:`:

```markdown
## Stores from old releases

`tests/adversarial/upgrade/` checks that a store written by any release opens with this code, migrates, and keeps everything it held. The plan is `docs/superpowers/plans/2026-09-26-adversarial-upgrade.md`.

- **The committed stores.** `tests/fixtures/stores/<tag>/` holds one store for each schema version a release has shipped, written by the first release that wrote it: v0.1.0 (version 5), v0.2.0 (6), v0.3.0 (8), v0.10.0 (9), v0.12.0 (12), v0.15.0 (15) and v0.16.0 (16). Versions 7, 10, 11, 13 and 14 never shipped. Each store holds the same program of about thirty operations: superseded, ended, retired, retracted and erased claims, claims in other users, tenants, agents and sessions, a claim with a source turn, turns the fast path read, accented names and names the entity fold changed, and, where the release has them, a project, a link, an expiry and a document.
- **How a store is built.** `tests/adversarial/upgrade/build_stores.py` extracts the release's `memvara` package with `git archive`, runs the program with that code in a child process, and reads the closed store with `sqlite3`, never with memvara, which would migrate it. Run it from the repository root; it needs the release tags. Ids come from a seeded generator, so a rebuild mints the same ones.
- **The golden dump.** `golden.json` holds `golden.dump` of the store: every claim with its scope, text, both clocks and sources, every turn, and the provenance edges, erasure records, links, documents, entities and predicates. It leaves out what a migration recomputes on purpose: the entity keys, the two hashes and the type columns.
- **Why the files are compressed.** The database and the vector file are committed gzip-compressed, about 20 KB a store. Uncompressed, even an almost empty store is over the 256 KB limit, because of SQLite's 4 KB pages and the vector file's room for 256 vectors.
- **What every pull request checks.** Each store is unpacked into the test's temporary directory, so no test can change a committed file. It must open without a warning, reach today's schema version, dump exactly as `golden.json` says, pass the integrity checks, and not change on a second open. Every claim, turn and document must be readable and found by search in its own scope and not in another user's; restating a stored value must reinforce the stored claim; and a claim an old release erased must stay unreadable. A store from a newer version, and a store opened with an embedder of another width, must be refused with a message.
- **What the nightly run checks.** It rebuilds every store from its tag and compares it with the committed one, hiding only the instants the release read from the clock. It checks that every schema version a release shipped has a store, kills a child process between two migrations of each old store, upgrades a store whose release was killed with a write still in its write-ahead log, and upgrades an encrypted store that v0.15.0 writes on the spot, because encrypted pages do not compress enough to commit.
- **When a release changes the schema.** Add the release to `golden.RELEASES` and run `python tests/adversarial/upgrade/build_stores.py <tag>`. The nightly test `test_every_released_schema_version_has_a_committed_store` fails until you do.
- **Two support modules.** Besides its tests, the folder holds `golden.py` and `build_stores.py`, which are not tests.
```

- [ ] **Step 1: Run each new test file 20 times in a row**

Run, for each of the four files, a loop of 20 runs of `PYTHONPATH=$PWD TMPDIR=<own tmp> <python> -m pytest -q -p no:cacheprovider <file>` (with `--tier nightly` for the nightly file), counting passes.
Expected: 20 of 20 for each.

- [ ] **Step 2: Run the fast tier's time for this workstream**

Run: `pytest -q -p no:cacheprovider tests/adversarial/upgrade --durations=10`
Expected: about 5 seconds in total.

- [ ] **Step 3: Run the full gate and both type checks**

```bash
COVERAGE_FILE=$PWD/local/cov/.coverage.upgrade <python> -m coverage run -m pytest -q -p no:cacheprovider
COVERAGE_FILE=$PWD/local/cov/.coverage.upgrade <python> -m coverage report
<python> -m mypy -p memvara
<python> -m mypy tests/harness
<python> -m mypy tests/harness --ignore-missing-imports
```

Expected: the gate passes with 100% coverage of `memvara/`, and mypy reports no issues.

- [ ] **Step 4: Report**

The final message names the branch, each commit, the files, the test counts, the 20-repeat results, the gate's result line and coverage total, the mypy results, every bug found with its classification and reproduction, and every departure from the design with its reason.
