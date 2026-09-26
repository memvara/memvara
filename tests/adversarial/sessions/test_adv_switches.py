"""The tool surface in this process: the oracle in switches.py against a server built for
each of the 2,048 combinations of the settings that change it, and the arrays of
combinations the real-server tests start.

The section "Scripted sessions and the tool surface" in docs/claude/testing.md explains
the oracle and the arrays.
"""

from __future__ import annotations

import dataclasses
import pathlib
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

import pytest

from harness import stores
from memvara.server import mcp
from memvara.server import tools as server_tools
from memvara.server.config import FEATURES, FEATURES_OFF_BY_DEFAULT
from memvara.server.mcp import MemvaraMCPServer
from memvara.server.tools import (BY_NAME, FEATURE_ARGUMENTS, TOOLS, ToolContext, ToolError,
                                  without_filters)
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


#: A planted fault: how to plant it, and which combinations it affects.
Fault = tuple[Callable[[pytest.MonkeyPatch], None], Callable[[Combination], bool]]

FAULTS: dict[str, Fault] = {
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


def _runs_hidden_tools(patch: pytest.MonkeyPatch) -> None:
    """A tool the server does not list still runs when it is called by name, so hiding it
    only changes the list."""
    original = MemvaraMCPServer._call_tool

    def call_tool(self: MemvaraMCPServer, params: Mapping[str, Any]) -> dict[str, Any]:
        tool = BY_NAME.get(str(params.get("name")))
        if tool is None or tool.name in self._tools:
            return original(self, params)
        try:
            return mcp._text(tool.run(self._ctx, params.get("arguments", {})))
        except ToolError as exc:
            return mcp._text(str(exc), is_error=True)

    patch.setattr(MemvaraMCPServer, "_call_tool", call_tool)


def _ignores_filepath_prefix(patch: pytest.MonkeyPatch) -> None:
    """With metadata filters off, memory_search drops filepath_prefix and answers without
    the filter, instead of refusing the call."""
    search = BY_NAME["memory_search"]

    def unfiltered(ctx: ToolContext, args: dict[str, Any]) -> str:
        return search.handler(ctx, {**args, "filepath_prefix": None})

    patch.setattr(mcp, "TOOLS", tuple(dataclasses.replace(tool, handler=unfiltered)
                                      if tool.name == "memory_search" else tool
                                      for tool in TOOLS))


REFUSAL_FAULTS: dict[str, Fault] = {
    "a-hidden-tool-still-runs": (
        _runs_hidden_tools, lambda c: bool(switches.Expected.of(c).unavailable)),
    "filepath-prefix-ignored-without-filters": (
        _ignores_filepath_prefix, lambda c: not c.feature_on("metadata_filters")),
}


@pytest.mark.parametrize("fault", sorted(REFUSAL_FAULTS))
def test_the_refusal_check_finds_a_planted_fault_on_exactly_the_servers_it_affects(
        fault: str, base: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Neither fault changes the list, so only the refusal check can see them."""
    plant, affected = REFUSAL_FAULTS[fault]
    plant(monkeypatch)
    runs = switches.every_combination()
    assert switches.in_process_failures(runs, base, refusals=False) == {}
    found = set(switches.in_process_failures(runs, base))
    assert found == {combination for combination in runs if affected(combination)}


def test_the_fast_rows_run_every_listed_tool_in_process(base: dict[str, str]) -> None:
    """Every tool each fast row's server lists runs with its minimal call. The nightly tier
    does the same for all 2,048 combinations."""
    runs = switches.fast_runs()
    failures = switches.in_process_failures(runs, base, calls=True)
    assert not failures, switches.summary(failures, len(runs))


def test_the_call_check_finds_a_handler_that_needs_an_argument_a_switch_removed(
        base: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    """memory_recall's handler is made to read query_rewrite with args[...]. A server that
    removed that argument then fails the call with a KeyError. The list stays the same, so
    only the call check can see it, and it must see it on exactly the fast rows that
    switch query rewriting off.

    The refusal check is left out. With metadata filters off, the engine refuses a filter
    from inside the handler, so on those rows the refusal check reaches the fault too."""
    recall = BY_NAME["memory_recall"]

    def strict(ctx: ToolContext, args: dict[str, Any]) -> str:
        _ = args["query_rewrite"]
        return recall.handler(ctx, args)

    monkeypatch.setattr(mcp, "TOOLS", tuple(dataclasses.replace(tool, handler=strict)
                                            if tool.name == "memory_recall" else tool
                                            for tool in TOOLS))
    runs = switches.fast_runs()
    assert switches.in_process_failures(runs, base, refusals=False) == {}
    found = switches.in_process_failures(runs, base, refusals=False, calls=True)
    assert set(found) == {run for run in runs if not run.feature_on("query_rewrite")}
    assert all(problem.startswith("memory_recall ") and "KeyError" in problem
               for problems in found.values() for problem in problems)


@pytest.mark.parametrize("pairs, wanted", [
    ((), "does not pair its schema with a replacement"),
    (switches.ANCHORED_ON + ((server_tools._ANCHORED, {"type": "boolean"}),),
     "pairs its schema with 2 replacements"),
])
def test_the_oracle_needs_exactly_one_anchored_replacement_for_each_schema(
        pairs: tuple[Any, ...], wanted: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """With none, the oracle cannot predict the anchored schema. With several, it cannot
    tell which one the server uses, so it must not pick the first."""
    monkeypatch.setattr(switches, "ANCHORED_ON", pairs)
    with pytest.raises(LookupError, match=wanted):
        switches.served(Combination.of("anchored"))


# -- the store a real server starts on -------------------------------------------------

def test_the_store_check_sees_a_write_that_changes_no_claim(
        surface_template: switches.Template, tmp_path: pathlib.Path) -> None:
    """A link changes no claim and no document, so only a check of every row can see it.
    A copy that nothing touched must compare equal, or every read-only run would fail."""
    untouched = surface_template.copy(tmp_path / "untouched")
    assert switches.store_dump(untouched) == surface_template.dump
    linked = surface_template.copy(tmp_path / "linked")
    with stores.file(linked, expiry_erasure=False, sweep_expired=False) as memory:
        memory.scope(user=switches.USER).link(
            surface_template.ids.linked_from, surface_template.ids.linked_to, "extends")
    assert switches.store_dump(linked) != surface_template.dump


def test_the_template_holds_an_expired_fact_that_only_a_writable_open_erases(
        surface_template: switches.Template, tmp_path: pathlib.Path) -> None:
    """Every server starts on a store that holds a fact whose expiry has passed. A read-only
    server must hide it and keep it, so the store check also sees a read-only server that
    erases. A writable open erases it, which changes the dump."""
    [expired] = [claim for claim in switches.read_claims(surface_template.directory
                                                         / switches.STORE)
                 if claim.object == "4417"]
    assert expired.expires_at is not None and expired.expires_at < datetime.now(timezone.utc)
    copy = surface_template.copy(tmp_path / "copy")
    with stores.file(copy):
        pass  # opening a store for writing erases every fact whose expiry has passed
    assert [claim for claim in switches.read_claims(copy) if claim.object == "4417"] == []
    assert switches.store_dump(copy) != surface_template.dump


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
