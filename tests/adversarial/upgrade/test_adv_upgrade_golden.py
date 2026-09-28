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
    # As a store written before versions 13 and 15 lacks them. `expire_reason` is in
    # no index, so SQLite lets it be dropped.
    raw.execute("DROP TABLE claim_links")
    raw.execute("ALTER TABLE claims DROP COLUMN expire_reason")
    raw.commit()
    raw.close()
    data = golden.dump(db)
    assert data["claim_links"] == []
    assert [row["expire_reason"] for row in data["claims"]] == [None]
    assert [row["valid_from"] for row in data["claims"]] == ["2024-01-01T00:00:00+00:00"]


def test_a_predicates_graph_declaration_is_dumped_and_reads_as_version_10s_defaults_when_absent(
        tmp_path: pathlib.Path) -> None:
    """Version 10 gave predicates a graph declaration. A store written before it has no
    such columns, and the migration adds them with constant defaults, so the dump reads
    an absent column as that default and a corrupted value still shows as a change."""
    db = written(tmp_path / "s.db")
    raw = sqlite3.connect(db)
    raw.execute("INSERT INTO predicates (tenant, name, cardinality, volatility, memory_type,"
                " aliases, supersedes, learned, subject_type, object_type, graph, inverse,"
                " inverse_cardinality, traversal_cost) VALUES ('default', 'deploys_to',"
                " 'one', 'slow', 'semantic', '[\"ships_to\"]', '[]', 0, '[\"component\"]',"
                " '[\"environment\"]', 1, 'hosts', 'many', 2.5)")
    raw.commit()
    [row] = golden.dump(db)["predicates"]
    assert (row["subject_type"], row["object_type"], row["graph"], row["inverse"],
            row["inverse_cardinality"], row["traversal_cost"]) == (
        ["component"], ["environment"], 1, "hosts", "many", 2.5)
    for column in ("subject_type", "object_type", "graph", "inverse",
                   "inverse_cardinality", "traversal_cost"):
        raw.execute(f"ALTER TABLE predicates DROP COLUMN {column}")
    raw.commit()
    raw.close()
    [row] = golden.dump(db)["predicates"]
    assert (row["subject_type"], row["object_type"], row["graph"], row["inverse"],
            row["inverse_cardinality"], row["traversal_cost"]) == ([], [], 0, None, None, 1.0)


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


def test_a_snapshot_records_a_write_ahead_log_left_behind_and_changes_names_it(
        tmp_path: pathlib.Path) -> None:
    db = written(tmp_path / "s.db")
    before = golden.snapshot(db)
    assert before["logs"] == {"-wal": None, "-shm": None}, "a clean close left a log"
    # Stands in for a close that never checkpointed. The snapshot must look before its
    # own connection checkpoints the log and deletes it.
    (tmp_path / "s.db-wal").write_bytes(b"\0" * 100)
    after = golden.snapshot(db)
    assert after["logs"]["-wal"] == 100
    assert "the write-ahead log was absent and is now 100 bytes" in golden.changes(
        before, after)


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
