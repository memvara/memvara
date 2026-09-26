"""The timing run end to end at a tiny size, and the rules around it.

Nothing here asserts how long anything took: the machine is shared, and the design keeps
absolute latencies out of the fast tier. What is asserted is that every series measures
what it names, in a real child process or a real hook, that a run is judged only when it
is valid, and that a series slower than its history is measured again before it counts.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Sequence

import pytest

import perf_budget as pb
from harness.hooks import HookRunner, HookTimeout
from harness.skips import explained
from memvara import Memvara, NullLLM
from memvara.embed import HashingEmbedder

LIBRARY = ("search", "recall", "remember")
HOOKS = ("hook.session_start", "hook.recall")
KEYS = {pb.series_key(operation, 40, temperature)
        for operation in LIBRARY + HOOKS for temperature in ("cold", "warm")}
VALID = pb.Conditions(on_battery=False, load_per_cpu=0.1)
BUSY = pb.Conditions(on_battery=False, load_per_cpu=0.9)


def test_a_built_store_holds_exactly_the_claims_asked_for(tmp_path: pathlib.Path) -> None:
    names = pb.build_store(tmp_path / "store.db", 40)
    assert len(names) == 10 and len(set(names)) == 10
    mem = Memvara(str(tmp_path / "store.db"), embedder=HashingEmbedder(dim=512),
                  llm=NullLLM(), user=pb.PERF_USER)
    assert mem.stats()["live_claims"] == 40
    mem.close()


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One real run at 40 claims: every series in real child processes and real hooks.
    The conditions are forced valid so that the record is judged whatever else is
    running on this machine."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pb, "read_conditions", lambda: VALID)
        return pb.measure(pb.PerfConfig(sizes=(40,), cold=1, warm=2, resamples=50),
                          tmp_path_factory.mktemp("perf"))


def test_a_tiny_run_measures_every_series_it_names(tiny: dict[str, Any]) -> None:
    assert set(tiny["series"]) == KEYS
    for key, series in tiny["series"].items():
        expected = 1 if key.endswith("/cold") else 2
        assert series["n"] == len(series["samples_ms"]) == expected, key
        assert all(sample > 0 for sample in series["samples_ms"]), key
        assert series["timeouts"] == 0, key


def test_the_report_names_the_machine_and_every_series(tiny: dict[str, Any]) -> None:
    text = pb.report(tiny)
    first, second = text.splitlines()[:2]
    assert tiny["fingerprint"]["cpu"] in first and second == "valid"
    for key in KEYS:
        assert sum(line.startswith(key + " ") for line in text.splitlines()) == 1, key


def test_a_tiny_run_is_recorded_and_judged(tiny: dict[str, Any]) -> None:
    assert (tiny["kind"], tiny["version"], tiny["valid"], tiny["invalid_reasons"]) == (
        "memvara-perf", 1, True, [])
    assert tiny["fingerprint"]["id"] and tiny["date"] == tiny["started"][:10]
    assert len(tiny["ceilings"]) == 6
    assert {verdict["outcome"] for verdict in tiny["regressions"].values()} == {"no history"}
    assert set(tiny["regressions"]) == KEYS and tiny["budgets"] is None


# --- judging a run, with the measurement replaced by fixed samples ------------------------


class FixedSeries:
    """Stands in for measuring a series: every series takes `ms` on every sample.

    It replaces a method on the class, and an object that is not a function is not bound
    to the instance, so it is called without one.
    """

    def __init__(self, ms: float) -> None:
        self.ms = ms
        self.calls: list[str] = []

    def __call__(self, operation: str, size: int,
                 temperature: str) -> tuple[list[float], int]:
        self.calls.append(pb.series_key(operation, size, temperature))
        return [self.ms, self.ms], 0


def fake_run(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, *,
             conditions: pb.Conditions, history: Sequence[dict[str, Any]] = (),
             ms: float = 5.0) -> tuple[dict[str, Any], FixedSeries]:
    series = FixedSeries(ms)
    monkeypatch.setattr(pb, "read_conditions", lambda: conditions)
    monkeypatch.setattr(pb._Bench, "series", series)
    record = pb.measure(pb.PerfConfig(sizes=(40,), cold=1, warm=1, resamples=20), tmp_path,
                        history=history)
    return record, series


def night(date: str, p95: float, *, fingerprint: str, valid: bool = True) -> dict[str, Any]:
    return {"kind": "memvara-perf", "version": 1, "date": date, "started": f"{date}T02:00:00",
            "valid": valid, "fingerprint": {"id": fingerprint},
            "series": {key: {"p95": p95} for key in KEYS}}


def test_a_run_under_load_is_reported_invalid_and_not_judged(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    history = [night(f"2026-09-{day:02d}", 1.0, fingerprint=pb.machine_fingerprint()["id"])
               for day in range(1, 8)]
    record, series = fake_run(monkeypatch, tmp_path, conditions=BUSY, history=history)
    assert record["valid"] is False and "0.90" in record["invalid_reasons"][0]
    assert record["regressions"] == {}
    assert len(series.calls) == len(KEYS), "nothing is measured again on an invalid run"
    assert pb.exit_code(record) == pb.EXIT_INVALID


def test_a_series_slower_than_its_history_is_measured_again_and_fails(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    mine = pb.machine_fingerprint()["id"]
    history = [night(f"2026-09-{day:02d}", 10.0, fingerprint=mine) for day in range(1, 8)]
    record, series = fake_run(monkeypatch, tmp_path, conditions=VALID, history=history,
                              ms=50.0)
    assert {v["outcome"] for v in record["regressions"].values()} == {"regression"}
    assert sorted(series.calls) == sorted([*KEYS, *KEYS]), "each series is measured twice"
    assert pb.exit_code(record) == 1


def test_history_from_another_machine_is_not_compared(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    history = [night(f"2026-09-{day:02d}", 10.0, fingerprint="another-machine")
               for day in range(1, 8)]
    record, _ = fake_run(monkeypatch, tmp_path, conditions=VALID, history=history, ms=50.0)
    assert {v["outcome"] for v in record["regressions"].values()} == {"no history"}


def test_a_child_is_given_a_bounded_command_line_whatever_the_store_size(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    # A child's queries travel on its command line, which macOS caps at 1 MB together
    # with the environment. Naming all 25,000 people of a 100,000-claim store came close
    # to that; a run needs only as many queries as it makes calls.
    given: list[str] = []
    monkeypatch.setattr(pb, "_run_child",
                        lambda spec, home, *, timeout: given.append(json.dumps(spec)) or [1.0])
    bench = pb._Bench(pb.PerfConfig(), tmp_path, None)
    bench.stores[100_000] = bench.copies[100_000] = tmp_path / "store.db"
    bench.names[100_000] = [f"Person{n:05d}" for n in range(25_000)]
    bench.series("search", 100_000, "warm")
    bench.series("remember", 100_000, "cold")
    assert max(len(spec) for spec in given) < 20_000


# --- hooks that did not read the store ----------------------------------------------------


def test_a_hook_with_no_store_configured_is_refused_rather_than_timed(
        tmp_path: pathlib.Path) -> None:
    def unconfigured(host: str, **options: Any) -> HookRunner:
        return HookRunner(host, home=options["home"], cwd=options["cwd"])

    with pytest.raises(pb.PerfError, match="not configured"):
        pb.time_hook(tmp_path / "store.db", "recall", 1, cold=True, workdir=tmp_path,
                     names=["Talovimar"], runner_factory=unconfigured)


@pytest.mark.parametrize("hook, message", [("recall", "never recalled"),
                                           ("session_start", "no claim")])
def test_a_hook_pointed_at_a_missing_store_is_refused_rather_than_timed(
        tmp_path: pathlib.Path, hook: str, message: str) -> None:
    # The hooks create an empty store at a path that does not exist and answer as though
    # nothing matched, which is faster than a real read and would pass for one.
    with pytest.raises(pb.PerfError, match=message):
        pb.time_hook(tmp_path / "missing.db", hook, 1, cold=True, workdir=tmp_path,
                     names=["Talovimar"])


class TimingOut:
    """A HookRunner whose every run passes the host's time limit."""

    def __init__(self, host: str, **options: Any) -> None:
        self.host = HookRunner(host, home=options["home"], cwd=options["cwd"]).host

    def run(self, hook: str, **fields: Any) -> Any:
        raise HookTimeout(f"{hook} ran past its limit")


def test_a_hook_that_times_out_is_recorded_at_its_limit_and_breaches_the_maximum(
        tmp_path: pathlib.Path) -> None:
    samples, timeouts = pb.time_hook(tmp_path / "store.db", "recall", 2, cold=False,
                                     workdir=tmp_path, names=["Talovimar"],
                                     runner_factory=TimingOut)
    assert (samples, timeouts) == ([10_000.0, 10_000.0], 2)
    series = {pb.series_key("hook.recall", 40, "warm"):
              {**pb.summarize(samples, resamples=10), "timeouts": timeouts}}
    # Every run sat at the 10-second limit, so the p95 of 10,000 ms breaches its 7,500 ms
    # ceiling too; the maximum is breached by the timeouts themselves.
    assert [e["stat"] for e in pb.check_ceilings(series) if e["breached"]] == ["p95", "max"]


# --- nights, budgets and exit codes -------------------------------------------------------


def test_only_timing_records_are_loaded_as_runs(tmp_path: pathlib.Path) -> None:
    (tmp_path / "b.json").write_text(json.dumps(night("2026-09-02", 1.0, fingerprint="m")),
                                     encoding="utf-8")
    (tmp_path / "a.json").write_text(json.dumps(night("2026-09-01", 1.0, fingerprint="m")),
                                     encoding="utf-8")
    (tmp_path / "soak.json").write_text(json.dumps({"kind": "memvara-soak"}),
                                        encoding="utf-8")
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    assert [run["date"] for run in pb.load_runs(tmp_path)] == ["2026-09-01", "2026-09-02"]
    assert pb.load_runs(tmp_path / "missing") == []


def test_only_valid_nights_on_this_machine_count_and_the_last_run_of_a_night_wins() -> None:
    runs = [night("2026-09-01", 1.0, fingerprint="m"),
            night("2026-09-02", 2.0, fingerprint="m"),
            {**night("2026-09-02", 3.0, fingerprint="m"), "started": "2026-09-02T05:00:00"},
            night("2026-09-03", 4.0, fingerprint="m", valid=False),
            night("2026-09-04", 5.0, fingerprint="other")]
    kept = pb.nights_for(runs, "m")
    assert [(n["date"], n["series"]["search@40/warm"]["p95"]) for n in kept] == [
        ("2026-09-01", 1.0), ("2026-09-02", 3.0)]


def test_budgets_need_fourteen_valid_nights_and_are_written_once(
        tmp_path: pathlib.Path) -> None:
    fingerprint = {"id": "m", "cpu": "made-up"}
    runs = [night(f"2026-09-{day:02d}", 8.0, fingerprint="m") for day in range(1, 14)]
    path = tmp_path / "perf_budgets.json"
    with pytest.raises(ValueError, match="14"):
        pb.write_budgets(runs, fingerprint, path)
    runs.append(night("2026-09-14", 8.0, fingerprint="m"))
    written = pb.write_budgets(runs, fingerprint, path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == written and saved["fingerprint"] == fingerprint
    assert len(saved["nights"]) == 14
    # 1.5 x 8 = 12, rounded up to 20, for each library series and no hook series.
    assert saved["budgets_ms"] == {key: 20.0 for key in KEYS if not key.startswith("hook.")}
    with pytest.raises(FileExistsError):
        pb.write_budgets(runs, fingerprint, path)


def test_a_committed_budget_is_checked_only_on_the_machine_that_measured_it(
        tmp_path: pathlib.Path) -> None:
    path = tmp_path / "perf_budgets.json"
    path.write_text(json.dumps({"fingerprint": {"id": "m"},
                                "budgets_ms": {"search@40/warm": 20.0}}), encoding="utf-8")
    series = {"search@40/warm": {"p95": 25.0}}
    assert pb.check_budgets(series, {"id": "m"}, path) == {
        "search@40/warm": {"p95": 25.0, "budget_ms": 20.0, "over": True}}
    assert pb.check_budgets(series, {"id": "other"}, path) is None
    assert pb.check_budgets(series, {"id": "m"}, tmp_path / "missing.json") is None


def test_an_invalid_night_skips_with_a_reason_the_skip_ledger_explains() -> None:
    reason = pb.skip_reason({"invalid_reasons": ["the machine ran on battery at 2 of 5 checks"]})
    assert reason == ("the performance run is invalid: the machine ran on battery at 2 of 5 "
                      "checks")
    assert explained(reason)


@pytest.mark.parametrize("changes, code", [
    ({}, 0),
    ({"ceilings": [{"breached": True}]}, 1),
    ({"regressions": {"search@40/warm": {"outcome": "regression"}}}, 1),
    ({"budgets": {"search@40/warm": {"over": True}}}, 1),
    ({"valid": False, "ceilings": [{"breached": True}]}, pb.EXIT_INVALID),
])
def test_the_exit_code_fails_a_valid_run_only(changes: dict[str, Any], code: int) -> None:
    record = {"valid": True, "ceilings": [{"breached": False}],
              "regressions": {"search@40/warm": {"outcome": "not reproduced"}},
              "budgets": None, **changes}
    assert pb.exit_code(record) == code
