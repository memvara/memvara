"""Every host approves exactly the memvara tools that only read.

The approve hook answers a host's permission check for a memvara tool. It allows a tool
in its READ_ONLY list and says nothing about any other, which leaves the host to ask the
person (plugin/hooks/approve.py). That list must be the tools the server marks read-only,
`readOnlyHint` in its `tools/list`, as a client sees it over stdio. The list used to
miss the two document readers (#267, B3).
"""

from __future__ import annotations

import importlib
import re
import sys
from types import ModuleType
from typing import Callable

import pytest

from harness import known_bugs
from harness.hooks import HOOKS_DIR, HookRunner, host_record
from harness.stdio import McpProcess

from . import support

def _approve_module() -> ModuleType:
    """plugin/hooks/approve.py. plugin/hooks is not a package, so its folder goes on the
    path first."""
    if str(HOOKS_DIR) not in sys.path:
        sys.path.insert(0, str(HOOKS_DIR))
    return importlib.import_module("approve")


@pytest.fixture(scope="module")
def server_tools(tmp_path_factory: pytest.TempPathFactory) -> dict[str, bool]:
    """Every tool the real server lists with every feature at its default, and whether it
    marks the tool read-only."""
    base = tmp_path_factory.mktemp("server")
    with McpProcess(base / "memory.db", home=tmp_path_factory.mktemp("home")) as server:
        server.initialize()
        return {tool["name"]: bool(tool["annotations"]["readOnlyHint"])
                for tool in server.list_tools()}


@pytest.fixture(scope="module")
def approvals(server_tools: dict[str, bool],
              tmp_path_factory: pytest.TempPathFactory) -> support.Runs:
    """The approve hook, asked on every host about every tool the server lists, under
    every name that host gives it (support.TOOL_NAMES)."""
    work = tmp_path_factory.mktemp("approvals")
    jobs: support.Jobs = {}
    with support.runner_factory(work) as make:
        for host in support.HOSTS:
            for form in support.TOOL_NAMES[host]:
                for tool in server_tools:
                    name = form.format(tool=tool)
                    jobs[host, name, tool] = support.job(
                        make(host), "approve",
                        stdin=support.host_json(host, "approve", session="approve", cwd=work,
                                                tool_name=name))
        results = support.run_all(jobs)
    return support.Runs(results)


def test_the_approve_list_is_the_servers_read_only_tools(server_tools: dict[str, bool]) -> None:
    read_only = {name for name, only_reads in server_tools.items() if only_reads}
    allowed = set(_approve_module().READ_ONLY)
    assert allowed - read_only == set(), "approved, but the server does not mark it read-only"
    assert read_only <= allowed, "read-only on the server, but not approved"


@pytest.mark.parametrize("host", support.HOSTS)
def test_support_tool_names_are_the_names_the_host_record_describes(host: str) -> None:
    """support.TOOL_NAMES restates the records by hand. Each name it gives a tool on `host`
    must match the record's matcher, which decides the names that reach the approve hook,
    and the names must be exactly the record's approve prefixes, each followed by the tool.
    When a record changes, this names the table that has gone stale."""
    approve = host_record(host).approve
    for form in support.TOOL_NAMES[host]:
        assert re.search(approve.matcher, form.format(tool="memory_search")), (
            f"support.TOOL_NAMES is stale for {host}: {form!r} does not match "
            f"{approve.matcher!r}")
    forms = {form.removesuffix("{tool}") for form in support.TOOL_NAMES[host]}
    assert forms == set(approve.prefixes), (
        f"support.TOOL_NAMES is stale for {host}: it names {sorted(forms)}, and the record "
        f"approves {sorted(approve.prefixes)}")


#: How each host spells a memvara tool's name in the event its approve hook answers,
#: measured on 2026-09-28 with the real clients (#340).
MEASURED_NAMES = (("cursor", "MCP:memory_search"), ("opencode", "memvara_memory_search"))


@pytest.mark.parametrize(("host", "name"), MEASURED_NAMES)
@known_bugs.xfail("B58")
def test_a_read_only_tool_is_approved_under_the_name_its_host_sends(
        hooks: Callable[..., HookRunner], host: str, name: str) -> None:
    """Measured on 2026-09-28. Cursor 2026.09.15 sends `MCP:memory_search` to preToolUse,
    which does not name the server, so approving it there would approve any server's
    `memory_search`. Its beforeMCPExecution names the server (`mcp_server_name`), but
    headless Cursor ignored an `allow` from that hook: the call ran only with `--force`,
    while a `deny` was honoured. OpenCode 1.18.20 sends `memvara_memory_search`, and
    approving by that prefix would approve a tool of any server whose name starts with
    `memvara_`; its permission.ask never fired in a headless run. Both records therefore
    still approve only `mcp__memvara__`, and a memvara read prompts on both hosts. This
    stays open until each host's approval path is measured in an interactive session."""
    result = hooks(host).run("approve", tool_name=name)
    if (result.exit_code, result.reply, support.crashes(result)) == (0, None, []):
        raise known_bugs.Reproduced(f"B58: approve on {host} says nothing about {name!r}")
    assert support.decision_of(host, result.reply) == "allow"


@pytest.mark.parametrize("host", support.HOSTS)
def test_every_host_approves_exactly_the_read_only_tools(
        approvals: support.Runs, server_tools: dict[str, bool], host: str) -> None:
    wrong = []
    for (asked_on, name, tool), outcome in approvals.results.items():
        if asked_on != host:
            continue
        result = support.result_of(outcome)
        decision = support.decision_of(host, result.reply)
        expected = "allow" if server_tools[tool] else None
        if (result.exit_code, decision) != (0, expected):
            wrong.append(f"{name}: exit {result.exit_code}, {decision!r} where {expected!r}")
    assert not wrong, wrong
