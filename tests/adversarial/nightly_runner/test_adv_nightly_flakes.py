"""Flakes: a test that failed is rerun twice before anything is filed.

The majority of the three runs decides: a test that passes both reruns is a flake, and
only a test that fails both reruns is a confirmed break that may be filed. A test that
passes one rerun and fails the other is intermittent. It still counts as a failure, but it
is never filed, because a strict expected failure on a test that sometimes passes would
itself make the suite flaky. The flake rate per layer over the last fourteen nights is what
the design's flake budget (0.5% per layer) is measured against.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
from typing import Any

import pytest

from harness.env import REPO, child_env

if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))
from nightly import flakes  # noqa: E402 - scripts/ is not on the path until above


@pytest.mark.parametrize("results, verdict, flaky", [
    (("passed", "passed"), "flake", True),
    (("failed", "failed"), "confirmed", False),
    (("passed", "failed"), "intermittent", True),
    (("failed", "passed"), "intermittent", True),
    (("failed", "error"), "unconfirmed", False),
    (("passed", "error"), "unconfirmed", True),
    ((), "unconfirmed", False),
])
def test_the_reruns_decide_what_a_failure_was(
        results: tuple[str, ...], verdict: str, flaky: bool) -> None:
    """Only a failure that repeats every time is confirmed: a rerun that could not run the
    test, or a test that was never rerun, confirms nothing. A test that passed after
    failing is flaky whatever else happened."""
    assert flakes.verdict(results) == verdict
    assert flakes.flaky(results) is flaky


@pytest.mark.parametrize("code, outcome", [
    (0, "passed"), (1, "failed"), (2, "error"), (4, "error"), (5, "error"), (None, "error")])
def test_only_a_run_that_ran_the_test_counts_as_a_pass_or_a_failure(
        code: int | None, outcome: str) -> None:
    """pytest exits 1 when a test failed. It exits 2, 4 or 5 when it was interrupted,
    misused, or collected nothing, which says nothing about the test, and None means the
    rerun was stopped at its cap."""
    assert flakes.outcome_of(code) == outcome


def test_a_failed_test_is_rerun_twice_with_the_nights_tier() -> None:
    """The rerun must run the same test, in the same tier, as the night did."""
    commands: list[list[str]] = []
    codes = iter([1, 0])

    def run(argv: list[str]) -> int | None:
        commands.append(argv)
        return next(codes)

    nodeid = "tests/adversarial/model/test_adv_model_machine.py::test_x[a b]"
    rerun = flakes.rerun(nodeid, run, python="/venv/bin/python")
    expected = ["/venv/bin/python", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                "--tier", "nightly", nodeid]
    assert commands == [expected, expected]
    assert (rerun.nodeid, rerun.results) == (nodeid, ("failed", "passed"))
    assert (rerun.verdict, rerun.flaky) == ("intermittent", True)


@pytest.mark.parametrize("nodeid, layer", [
    ("tests/adversarial/model/test_adv_model_machine.py::test_x", "model"),
    ("tests/adversarial/concurrency/nightly/test_adv_full_disk.py::test_x", "concurrency"),
    ("tests/adversarial/nightly/test_adv_nightly_tier_guard.py::test_x", "adversarial"),
    ("tests/adversarial/test_adv_env.py::test_x", "adversarial"),
    ("tests/live/replay.py::test_x", "live"),
    ("tests/test_api.py::test_x", "unit"),
    ("memvara/core.py::memvara.core.Memvara.recall", "doctest"),
    ("bench/test_x.py::test_x", "other"),
])
def test_a_tests_layer_is_the_folder_that_names_what_it_attacks(nodeid: str,
                                                                layer: str) -> None:
    """The flake budget is per layer. A tier folder is not a layer: a nightly concurrency
    test belongs to concurrency, or the nightly folder's flakes would hide in one bucket."""
    assert flakes.layer_of(nodeid) == layer


def _night(date: str, **layers: tuple[int, int]) -> dict[str, Any]:
    return {"kind": "night", "date": date,
            "layers": {name: {"run": run, "failed": flaky, "flaky": flaky}
                       for name, (run, flaky) in layers.items()}}


def test_the_flake_rate_counts_each_night_once_and_only_nights_that_ran() -> None:
    """A second run of one night replaces the first, and a night with no test results,
    or a filing record, adds nothing, or the rate would be counted against the wrong total."""
    history = [
        _night("2026-09-20", model=(100, 1), unit=(1000, 0)),
        _night("2026-09-21", model=(100, 0)),
        _night("2026-09-21", model=(100, 2)),
        {"kind": "filed", "date": "2026-09-21", "fingerprint": "0" * 64},
        {"kind": "night", "date": "2026-09-22", "status": "crashed"},
    ]
    rates = flakes.rates(history)
    assert sorted(rates) == ["model", "unit"]
    model, unit = rates["model"], rates["unit"]
    assert (model.run, model.flaky, model.nights) == (200, 3, 2)
    assert model.rate == pytest.approx(0.015)
    assert model.over_budget
    assert (unit.run, unit.flaky, unit.nights, unit.rate, unit.over_budget) == (
        1000, 0, 1, 0.0, False)


def test_the_flake_rate_looks_at_the_last_fourteen_nights_only() -> None:
    """The budget is measured over fourteen nights, so a flake fixed a month ago no longer
    counts against its layer."""
    history = [_night("2026-09-01", model=(100, 50))]
    history += [_night(f"2026-09-{day:02d}", model=(100, 0)) for day in range(2, 16)]
    assert flakes.rates(history)["model"].flaky == 0
    assert flakes.rates(history, window=15)["model"].flaky == 50


#: A test that fails the first time it runs and passes every time after that.
FAILS_ONCE = '''
import pathlib


def test_fails_only_on_its_first_run():
    marker = pathlib.Path(__file__).with_name("ran-once")
    if not marker.exists():
        marker.write_text("")
        raise AssertionError("this is the first run")
'''

#: Stands in for the checkout's own conftest.py, which registers --tier.
ACCEPTS_TIER = '''
def pytest_addoption(parser):
    parser.addoption("--tier", default="fast")
'''


def test_a_test_that_failed_once_and_then_passes_twice_is_reported_as_a_flake(
        tmp_path: pathlib.Path, tmp_path_factory: pytest.TempPathFactory,
        capsys: pytest.CaptureFixture[str]) -> None:
    """The whole rerun path, with a real pytest in a child process: the command line the
    rerun builds must select the one test, and its exit codes must be read the right way."""
    (tmp_path / "test_flaky.py").write_text(FAILS_ONCE)
    (tmp_path / "conftest.py").write_text(ACCEPTS_TIER)
    nodeid = "test_flaky.py::test_fails_only_on_its_first_run"
    home = tmp_path_factory.mktemp("home")
    first = subprocess.run(flakes.rerun_command(sys.executable, nodeid), cwd=tmp_path,
                           env=child_env(home), capture_output=True, timeout=120)
    assert first.returncode == 1, first.stdout
    assert flakes.main(["rerun", "--worktree", str(tmp_path), "--python", sys.executable,
                        "--home", str(home), nodeid]) == 0
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert printed == [{"nodeid": nodeid, "results": ["passed", "passed"],
                        "verdict": "flake", "flaky": True}]


def test_the_rates_command_reads_the_history_and_marks_a_layer_over_budget(
        tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    history = tmp_path / "local" / "nightly" / "history.jsonl"
    history.parent.mkdir(parents=True)
    history.write_text("\n".join(json.dumps(record) for record in [
        _night("2026-09-20", model=(1000, 6), unit=(1000, 0))]) + "\n")
    assert flakes.main(["rates", "--checkout", str(tmp_path)]) == 0
    lines = {line.split()[0]: line for line in capsys.readouterr().out.splitlines()[1:]}
    assert "0.60%" in lines["model"] and "over" in lines["model"]
    assert "0.00%" in lines["unit"] and "over" not in lines["unit"]
