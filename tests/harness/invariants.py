"""Checks on a closed store file, for the damage a crash or a bad write can leave.

`check_store_integrity(path)` opens the SQLite file directly, with no memvara code in
between, and returns a list of the problems it finds, in plain sentences. An empty list
means the file passed every check. The state machine calls it once a run is over, and
the crash tests will call it after reopening a store that a killed process left behind.

It checks:
- SQLite's own `PRAGMA integrity_check`, and FTS5's `integrity-check` on both text indexes;
- that every claim has exactly one row in the claim text index and one embedding row, and
  that neither index names a claim that is gone;
- that no vector slot is both in use and on the free list;
- that no provenance edge (`claim_sources`) names a claim or an episode that is gone;
- that no claim with an erasure record still exists.

It does not check an encrypted store, which needs its key, or the `.vecs` file's bytes.
"""

from __future__ import annotations

import pathlib
import sqlite3


def check_store_integrity(path: pathlib.Path | str) -> list[str]:
    """The problems found in the store file at `path`, or `[]` for a healthy one."""
    problems: list[str] = []
    # Not read-only: FTS5's integrity check is issued as an INSERT, although it writes
    # nothing. The connection is rolled back before it closes.
    conn = sqlite3.connect(path)
    try:
        result = [row[0] for row in conn.execute("PRAGMA integrity_check")]
        if result != ["ok"]:
            problems.extend(f"SQLite integrity check: {line}" for line in result)
        for table in ("claims_fts", "episodes_fts"):
            try:
                # FTS5's own consistency check between its index and its content.
                conn.execute(f"INSERT INTO {table}({table}) VALUES ('integrity-check')")
            except sqlite3.DatabaseError as exc:
                problems.append(f"the {table} index is damaged: {exc}")
        problems.extend(_claims(conn))
    finally:
        conn.rollback()
        conn.close()
    return problems


def _claims(conn: sqlite3.Connection) -> list[str]:
    problems: list[str] = []
    claims = {row[0] for row in conn.execute("SELECT id FROM claims")}
    text: dict[str, int] = {}
    for (claim_id,) in conn.execute("SELECT claim_id FROM claims_fts"):
        text[claim_id] = text.get(claim_id, 0) + 1
    embedded = {row[0] for row in conn.execute("SELECT claim_id FROM embeddings")}
    for claim_id in sorted(claims):
        if text.get(claim_id, 0) != 1:
            problems.append(
                f"claim {claim_id} has {text.get(claim_id, 0)} text-index rows, not 1")
        if claim_id not in embedded:
            problems.append(f"claim {claim_id} has no embedding row")
    for claim_id in sorted(set(text) - claims):
        problems.append(f"the text index names claim {claim_id}, which is gone")
    for claim_id in sorted(embedded - claims):
        problems.append(f"an embedding row names claim {claim_id}, which is gone")
    free = {row[0] for row in conn.execute("SELECT slot FROM vec_free")}
    for (slot,) in conn.execute(
            "SELECT slot FROM embeddings WHERE slot IS NOT NULL "
            "UNION ALL SELECT slot FROM episode_embeddings WHERE slot IS NOT NULL"):
        if slot in free:
            problems.append(f"vector slot {slot} is in use and on the free list")
    for (claim_id,) in conn.execute("SELECT DISTINCT claim_id FROM claim_sources"):
        if claim_id not in claims:
            problems.append(f"a source edge names claim {claim_id}, which is gone")
    episodes = {row[0] for row in conn.execute("SELECT id FROM episodes")}
    for (episode_id,) in conn.execute("SELECT DISTINCT episode_id FROM claim_sources"):
        if episode_id not in episodes:
            problems.append(f"a source edge names episode {episode_id}, which is gone")
    for (claim_id,) in conn.execute("SELECT claim_id FROM erasures"):
        if claim_id in claims:
            problems.append(f"claim {claim_id} has an erasure record and still exists")
    return problems
