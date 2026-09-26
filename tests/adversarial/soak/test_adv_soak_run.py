"""A tiny soak end to end: the command line, its record, its history, and the store it leaves.

The nightly and weekly runs call the same `main()` at 10,000 and 100,000 turns, so a
200-turn run here is what keeps the long runs from breaking unnoticed between nights.
"""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
from typing import Any

import pytest

import soak
from harness.invariants import check_store_integrity
from memvara.store import SQLiteStore

TURNS = 200


@pytest.fixture(scope="module")
def cli_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    folder = tmp_path_factory.mktemp("soak-cli")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = soak.main(["--turns", str(TURNS), "--store", str(folder / "store"),
                          "--out", str(folder / "record.json")])
    return {"code": code, "lines": out.getvalue().splitlines(),
            "record": json.loads((folder / "record.json").read_text(encoding="utf-8")),
            "stores": sorted((folder / "store").glob("*.db"))}


def test_the_command_line_runs_a_healthy_soak_and_passes(cli_run: dict[str, Any]) -> None:
    assert cli_run["code"] == 0
    for finding in cli_run["record"]["findings"]:
        assert finding["status"] in ("ok", "tracked"), finding


def test_the_command_line_prints_one_line_per_detector(cli_run: dict[str, Any]) -> None:
    names = [finding["detector"] for finding in cli_run["record"]["findings"]]
    assert len(names) == 8
    for name in names:
        assert sum(f" {name}: " in line for line in cli_run["lines"]) == 1, name


def test_the_record_says_what_ran_where(cli_run: dict[str, Any]) -> None:
    record = cli_run["record"]
    assert (record["kind"], record["version"], record["turns"], record["seed"]) == (
        "memvara-soak", 1, TURNS, 0)
    assert record["bytes_per_turn"] > 0 and record["fingerprint"]["id"]
    assert record["counts"]["merged"] >= 1
    assert record["counts"]["retraction"]["noop"] == 0
    assert record["counts"]["reconcile"]["reinforce"] > 0


def test_the_store_a_soak_leaves_passes_the_integrity_checks(cli_run: dict[str, Any]) -> None:
    [store] = cli_run["stores"]
    assert check_store_integrity(store) == []


def write_record(folder: pathlib.Path, name: str, **fields: Any) -> None:
    record = {"kind": "memvara-soak", "version": 1, "turns": TURNS, "seed": 0,
              "started": "2026-09-01T00:00:00+00:00", "bytes_per_turn": 1000.0, **fields}
    (folder / name).write_text(json.dumps(record), encoding="utf-8")


def test_history_holds_only_earlier_soaks_of_the_same_length_and_seed(
        tmp_path: pathlib.Path) -> None:
    write_record(tmp_path, "b.json", started="2026-09-03T00:00:00+00:00", bytes_per_turn=3.0)
    write_record(tmp_path, "a.json", started="2026-09-02T00:00:00+00:00", bytes_per_turn=2.0)
    write_record(tmp_path, "longer.json", turns=TURNS * 10, bytes_per_turn=99.0)
    write_record(tmp_path, "other-seed.json", seed=1, bytes_per_turn=99.0)
    write_record(tmp_path, "in-memory.json", bytes_per_turn=None)
    (tmp_path / "timing.json").write_text(json.dumps({"kind": "memvara-perf"}),
                                          encoding="utf-8")
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    assert soak.load_history(tmp_path, turns=TURNS, seed=0) == [2.0, 3.0]
    assert soak.load_history(tmp_path / "missing", turns=TURNS, seed=0) == []


def test_store_growth_catches_a_store_that_stops_recognising_repeats(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = soak.SoakConfig(TURNS)
    baseline = soak.run(config, tmp_path / "baseline.db")
    assert baseline.store_bytes is not None
    history = [baseline.store_bytes / TURNS] * 7

    # Neither a repeated turn nor a restated fact is recognised any more, so each one is
    # stored again. Either lookup alone grows a 200-turn store by about a tenth, less than
    # the rule's 1.20, because the vector file grows in steps of 256 vectors.
    monkeypatch.setattr(SQLiteStore, "find_by_value", lambda self, tenant, value_key: [])
    monkeypatch.setattr(SQLiteStore, "find_episode_by_hash", lambda self, tenant, key: None)
    runs = iter(range(10))

    def again() -> float:
        observed = soak.run(config, tmp_path / f"again-{next(runs)}.db")
        assert observed.store_bytes is not None
        return observed.store_bytes / TURNS

    grown = soak.run(config, tmp_path / "grown.db")
    finding = soak.store_growth(grown, history, again)
    assert finding.status == "fail" and "regression" in finding.detail

    monkeypatch.undo()
    healthy = soak.run(config, tmp_path / "healthy.db")
    assert soak.store_growth(healthy, history, again).status == "ok"


def test_the_command_line_fails_when_its_store_outgrows_its_history(
        tmp_path: pathlib.Path) -> None:
    history = tmp_path / "history"
    history.mkdir()
    for night in range(7):
        write_record(history, f"night-{night}.json", bytes_per_turn=1_000.0,
                     started=f"2026-09-0{night + 1}T00:00:00+00:00")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = soak.main(["--turns", str(TURNS), "--history", str(history)])
    assert code == 1
    assert any(line.startswith("fail") and " store growth: " in line
               for line in out.getvalue().splitlines())


def test_a_soak_refuses_a_store_that_already_exists(tmp_path: pathlib.Path) -> None:
    old = tmp_path / "old.db"
    old.write_bytes(b"somebody's store")
    with pytest.raises(FileExistsError, match="already exists"):
        soak.run(soak.SoakConfig(TURNS), old)
    assert old.read_bytes() == b"somebody's store"
