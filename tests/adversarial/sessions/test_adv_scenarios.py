"""The scripted scenarios in tests/scenarios/scripted, played over the real stdio pipe and
the plugin's real hook scripts.

Each scenario plays once, the first time a test asks for it, and every test here reads
that one play: one test per gold item, one for the script, one for forbidden tool calls
and one negative control. docs/claude/testing.md explains the format and how to add a
scenario.
"""

from __future__ import annotations

import pathlib
from typing import Any, Callable, Mapping

import pytest

from . import runner

Outcomes = Callable[[Mapping[str, Any]], runner.Outcome]


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Give each test the scenarios, or the gold items, that the run's tier selects."""
    name = metafunc.definition.name
    if name == "test_the_file_follows_the_format":
        files = sorted(runner.SCRIPTED.glob("*.json"))
        metafunc.parametrize("path", files, ids=[path.stem for path in files])
        return
    scenarios = runner.selected(runner.load_all(), metafunc.config.getoption("--tier"))
    if name == "test_gold":
        metafunc.parametrize("gold", runner.gold_params(scenarios))
        return
    if name == "test_no_forbidden_tool_was_called":
        scenarios = [scenario for scenario in scenarios if scenario.get("forbidden")]
    elif name == "test_the_gold_fails_without_memvara":
        scenarios = [scenario for scenario in scenarios if scenario["negative_control"]]
    metafunc.parametrize("scenario", scenarios,
                         ids=[scenario["id"] for scenario in scenarios])


@pytest.fixture(scope="module")
def outcomes(tmp_path_factory: pytest.TempPathFactory) -> Outcomes:
    """Play each scenario once and hand that play to every test that asks for it.

    A scenario that stops early is remembered as stopped, so each of its tests reports the
    same error instead of playing it again.
    """
    played: dict[str, runner.Outcome | Exception] = {}

    def get(scenario: Mapping[str, Any]) -> runner.Outcome:
        key = scenario["id"]
        if key not in played:
            try:
                played[key] = runner.run(scenario, tmp_path_factory.mktemp(key))
            except Exception as exc:  # noqa: BLE001 - every test of the scenario reports it
                played[key] = exc
        found = played[key]
        if isinstance(found, Exception):
            raise RuntimeError(f"the scenario {key} stopped before its end: {found}") from found
        return found

    return get


def test_the_file_follows_the_format(path: pathlib.Path) -> None:
    runner.load(path)


def test_gold(gold: runner.Gold, outcomes: Outcomes) -> None:
    runner.judge(gold, outcomes(gold.scenario))


def test_the_script_ran_as_written(scenario: Mapping[str, Any], outcomes: Outcomes) -> None:
    assert outcomes(scenario).problems == []


def test_no_forbidden_tool_was_called(scenario: Mapping[str, Any],
                                      outcomes: Outcomes) -> None:
    assert runner.forbidden_calls(scenario["forbidden"], outcomes(scenario).calls) == []


def test_the_gold_fails_without_memvara(scenario: Mapping[str, Any]) -> None:
    assert runner.fails_without_memvara(scenario), (
        "every gold item also passes for an agent with no memory, so this scenario cannot "
        "tell memvara working from memvara absent")
