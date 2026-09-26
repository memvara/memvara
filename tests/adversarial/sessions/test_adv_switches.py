"""The tool surface in this process: the oracle in switches.py against a server built for
each of the 2,048 combinations of the settings that change it, and the arrays of
combinations the real-server tests start.

The section "Scripted sessions and the tool surface" in docs/claude/testing.md explains
the oracle and the arrays.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any, Callable

import pytest

from harness import stores
from memvara.server import mcp
from memvara.server.config import FEATURES, FEATURES_OFF_BY_DEFAULT
from memvara.server.mcp import MemvaraMCPServer
from memvara.server.tools import BY_NAME, FEATURE_ARGUMENTS, TOOLS, without_filters
from memvara.server.validate import validate

from . import switches

Combination = switches.Combination


@pytest.fixture
def base(tmp_path: pathlib.Path) -> dict[str, str]:
    return switches.base_env(tmp_path)


# -- the settings and the oracle -------------------------------------------------------

def test_the_switches_are_the_settings_that_change_the_tool_list() -> None:
    """Every feature is switched away from its default one at a time, and read-only and
    anchored mode are switched on. The ones that change the list must be exactly the
    eleven in SWITCHES, so a new setting that changes it cannot go unchecked."""
    memory = stores.memory()

    def listing(**options: Any) -> list[dict[str, Any]]:
        memory.metadata_filters = True
        return switches.InProcess(MemvaraMCPServer(memory, user=switches.USER,
                                                   **options)).list_tools()

    try:
        default = listing()
        changed = {feature for feature in FEATURES
                   if listing(features_off=FEATURES_OFF_BY_DEFAULT ^ {feature}) != default}
        changed |= {mode for mode in ("read_only", "anchored")
                    if listing(**{mode: True}) != default}
    finally:
        memory.close()
    assert changed == set(switches.SWITCHES)


def test_a_combination_sets_one_variable_for_each_setting() -> None:
    env = Combination.of("documents", "read_only").env()
    assert len(env) == len(switches.SWITCHES)
    assert env["MEMVARA_FEATURE_DOCUMENTS"] == "0" and env["MEMVARA_FEATURE_LINKS"] == "1"
    assert env["MEMVARA_READ_ONLY"] == "1" and env["MEMVARA_ANCHORED"] == "0"


def test_a_combination_refuses_a_setting_that_does_not_change_the_list() -> None:
    with pytest.raises(ValueError, match="encryption"):
        Combination.of("encryption")


def test_every_combination_lists_and_refuses_what_the_oracle_predicts(
        base: dict[str, str]) -> None:
    runs = switches.every_combination()
    failures = switches.in_process_failures(runs, base)
    assert not failures, switches.summary(failures, len(runs))


# -- the check finds the faults it is for ----------------------------------------------

def _keeps_synthesize(patch: pytest.MonkeyPatch) -> None:
    """The server forgets that `synthesis` owns an argument, so `synthesize` stays."""
    patch.setattr(mcp, "FEATURE_ARGUMENTS",
                  {k: v for k, v in FEATURE_ARGUMENTS.items() if k != "synthesis"})


def _keeps_filter_descriptions(patch: pytest.MonkeyPatch) -> None:
    """With metadata filters off, the filter arguments keep their usual descriptions."""
    patch.setattr(mcp, "without_filters", lambda tools: tools)


def _rewrites_filters_from_the_table(patch: pytest.MonkeyPatch) -> None:
    """The filter rewrite starts from each tool's entry in TOOLS instead of from the tool
    it was given, so it undoes the anchored rewrite and brings removed arguments back."""
    patch.setattr(mcp, "without_filters", lambda tools: without_filters(tuple(
        BY_NAME[tool.name] if "filters" in tool.properties else tool for tool in tools)))


def _marks_a_write_tool_read_only(patch: pytest.MonkeyPatch) -> None:
    """memory_link is marked as a tool that does not write. Its annotation is then wrong
    wherever it is listed, and a read-only server lists it."""
    patch.setattr(mcp, "TOOLS", tuple(dataclasses.replace(tool, writes=False)
                                      if tool.name == "memory_link" else tool
                                      for tool in TOOLS))


FAULTS: dict[str, tuple[Callable[[pytest.MonkeyPatch], None],
                        Callable[[Combination], bool]]] = {
    "synthesize-is-kept": (_keeps_synthesize, lambda c: not c.feature_on("synthesis")),
    "filter-descriptions-are-kept": (
        _keeps_filter_descriptions, lambda c: not c.feature_on("metadata_filters")),
    "filters-rewritten-from-the-table": (
        _rewrites_filters_from_the_table,
        lambda c: not c.feature_on("metadata_filters") and (
            c.anchored or not c.feature_on("query_rewrite")
            or not c.feature_on("synthesis"))),
    "a-write-tool-marked-read-only": (_marks_a_write_tool_read_only,
                                      lambda c: c.feature_on("links")),
}


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_the_check_finds_a_planted_fault_on_exactly_the_servers_it_affects(
        fault: str, base: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Each fault is planted in the server's composition code in memvara/server/mcp.py,
    and the listing check must fail on every combination the fault touches and on no
    other. The oracle reads memvara/server/tools.py, which stays as it is."""
    plant, affected = FAULTS[fault]
    plant(monkeypatch)
    runs = switches.every_combination()
    found = set(switches.in_process_failures(runs, base, refusals=False))
    assert found == {combination for combination in runs if affected(combination)}


# -- the minimal calls -----------------------------------------------------------------

def test_every_tool_has_a_minimal_call_that_its_schema_accepts() -> None:
    """Every call goes through this validator, so a minimal call it refused would fail on
    every server. No minimal call may send an argument that a switch removes, and each
    such argument needs a value to be probed with."""
    calls = switches.minimal_calls(switches.NOWHERE)
    assert sorted(calls) == sorted(BY_NAME)
    for tool in TOOLS:
        validate(tool.properties, tool.required, calls[tool.name], tool=tool.name)
        assert not set(calls[tool.name]) & set(switches.PROBES), tool.name
    assert set(switches.PROBES) == {argument for arguments in FEATURE_ARGUMENTS.values()
                                    for argument in arguments}


# -- the arrays ------------------------------------------------------------------------

def test_the_fast_array_is_orthogonal_and_covers_every_three_settings() -> None:
    runs = switches.fast_runs()
    assert len(runs) == len(set(runs)) == 12
    assert Combination.of() in runs
    pairs = switches.interaction_counts(runs, 2)
    assert all(len(seen) == 4 and set(seen.values()) == {3} for seen in pairs.values())
    triples = switches.interaction_counts(runs, 3)
    assert all(len(seen) == 8 for seen in triples.values())


def test_the_nightly_array_covers_every_three_settings_and_completes_strength_three() -> None:
    nightly = switches.nightly_runs()
    assert len(nightly) == len(set(nightly)) == 12
    assert not set(nightly) & set(switches.fast_runs())
    assert all(len(seen) == 8 for seen in switches.interaction_counts(nightly, 3).values())
    both = switches.interaction_counts(switches.fast_runs() + nightly, 3)
    assert all(len(seen) == 8 and set(seen.values()) == {3} for seen in both.values())


def test_the_weekly_array_is_every_combination() -> None:
    runs = switches.every_combination()
    assert len(runs) == len(set(runs)) == 2 ** len(switches.SWITCHES) == 2048
