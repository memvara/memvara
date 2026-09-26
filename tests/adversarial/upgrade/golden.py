"""What an upgrade test compares: a store's contents, read with `sqlite3` alone.

A committed store under `tests/fixtures/stores/<tag>/` was written by release `<tag>`
(`build_stores.py` says how), and its `golden.json` holds `dump` of that store, taken
before anything else opened it. An upgrade test opens a copy with this checkout's code,
which migrates it, and dumps it again. The two dumps must be equal, because no migration
may change what a store holds.

`dump` reads what a migration must keep: every claim with its scope, text, both clocks
and sources, every episode, and the provenance edges, erasure records, links, documents,
entities and predicates, each predicate with the graph declaration version 10 added. It
leaves out what a migration recomputes on purpose: the entity keys and the two hashes
that versions 6, 12 and 16 re-derive, and the two type columns version 12 derives from
the keys. It also leaves out the vectors, which the integrity checks and the search tests
cover.

A table the store's version does not have reads as an empty list. A column it does not
have reads as the value the migration that adds the column gives existing rows: `None`
for the nullable columns, and version 10's defaults for a predicate's graph declaration.
So a store dumps the same before and after its migration, and a migration that writes
anything else into those columns shows as a change.

Nothing here imports memvara, so the builder can dump a store any release wrote. Nothing
here writes to a committed file: the tests `unpack` a copy first.
"""

from __future__ import annotations

import copy
import datetime as dt
import gzip
import hashlib
import json
import pathlib
import shutil
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
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
    the ones that hold instants as seconds since the epoch, and the ones that hold JSON.

    `absent` gives, for a column that a migration adds with a constant default, that
    default: a store written before the column existed reads as if the migration had
    already added it. A column missing from `absent` reads as `None`, which is what the
    migrations that add nullable columns give existing rows."""

    columns: tuple[str, ...]
    key: tuple[str, ...]
    instants: tuple[str, ...] = ()
    json: tuple[str, ...] = ()
    absent: Mapping[str, Any] = field(default_factory=dict)


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
         "supersedes", "learned", "subject_type", "object_type", "graph", "inverse",
         "inverse_cardinality", "traversal_cost"),
        key=("tenant", "name"),
        json=("aliases", "supersedes", "subject_type", "object_type"),
        # The graph declaration version 10 added, with the defaults it gives existing rows.
        absent={"subject_type": [], "object_type": [], "graph": 0, "inverse": None,
                "inverse_cardinality": None, "traversal_cost": 1.0}),
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
            if column in present and row[column] is not None:
                row[column] = json.loads(row[column])
        for column, default in table.absent.items():
            if column not in present:
                row[column] = copy.deepcopy(default)
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


#: The digest `snapshot` gives a table with no rows.
EMPTY = _digest([])


#: The two files SQLite keeps beside a database in write-ahead-log mode, by suffix.
LOGS = {"-wal": "write-ahead log", "-shm": "shared-memory file"}


def snapshot(db: pathlib.Path, *, key: bytes | None = None) -> dict[str, Any]:
    """Everything an open could change, to compare two opens: the schema version, the
    schema, a digest of every row of every table, a digest of each side file, and the
    size of the write-ahead log and the shared-memory file, or `None` when one does not
    exist.

    The two logs are looked at first. The connection this opens would checkpoint a
    write-ahead log that a close left behind, and delete both files when it closes."""
    logs = {suffix: _size(db.with_name(db.name + suffix)) for suffix in LOGS}
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
    return {"user_version": version, "schema": schema, "rows": rows, "files": files,
            "logs": logs}


def _size(path: pathlib.Path) -> int | None:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return None


def _bytes(size: int | None) -> str:
    return "absent" if size is None else f"{size} bytes"


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
    for suffix, name in LOGS.items():
        was, now = before["logs"][suffix], after["logs"][suffix]
        if was != now:
            problems.append(f"the {name} was {_bytes(was)} and is now {_bytes(now)}")
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
