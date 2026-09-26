"""Every host approves exactly the memvara tools that only read.

The approve hook answers a host's permission check for a memvara tool. It allows a tool
in its READ_ONLY list and says nothing about any other, which leaves the host to ask the
person (plugin/hooks/approve.py). That list must be the tools the server marks read-only,
`readOnlyHint` in its `tools/list`, as a client sees it over stdio.

#267 (B3) is already pinned for the two document tools the list misses
(test_adv_known_bugs.py). While B3 is registered, those two are left out here, and any
other difference between the lists fails.
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

#: The two read-only tools that #267 is about.
B3_MISSING = frozenset({"memory_get_document", "memory_list_documents"})


def _approve_module() -> ModuleType:
    """plugin/hooks/approve.py. plugin/hooks is not a package, so its folder goes on the
    path first."""
    if str(HOOKS_DIR) not in sys.path:
        sys.path.insert(0, str(HOOKS_DIR))
    return importlib.import_module("approve")


def _known_gap() -> frozenset[str]:
    """The tools B3 leaves out of the approve list, while B3 is registered."""
    return B3_MISSING if "B3" in known_bugs.KNOWN_BUGS else frozenset()


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
    assert read_only - allowed <= _known_gap(), "read-only on the server, but not approved"


@pytest.mark.parametrize("host", support.HOSTS)
def test_support_tool_names_are_the_names_the_host_record_describes(host: str) -> None:
    """support.TOOL_NAMES restates the records by hand. Each name it gives a tool on `host`
    must match the record's matcher, which decides the names that reach the approve hook,
    and must join the tool on with the first separator the record lists. When a record
    changes, this names the table that has gone stale."""
    approve = host_record(host).approve
    for form in support.TOOL_NAMES[host]:
        assert re.search(approve.matcher, form.format(tool="memory_search")), (
            f"support.TOOL_NAMES is stale for {host}: {form!r} does not match "
            f"{approve.matcher!r}")
        assert form.endswith(approve.separators[0] + "{tool}"), (
            f"support.TOOL_NAMES is stale for {host}: {form!r} does not join the tool on "
            f"with {approve.separators[0]!r}")


@pytest.mark.parametrize("host", ("cursor", "opencode"))
@pytest.mark.parametrize("name", ("memvara_memory_search", "memory_search"))
@known_bugs.xfail("B58")
def test_a_read_only_tool_named_with_a_single_underscore_is_approved(
        hooks: Callable[..., HookRunner], host: str, name: str) -> None:
    """Cursor's and OpenCode's records list "_" as an approve separator, and approve.py,
    `_tool_leaf`, splits a name at the last separator it holds. Every memvara tool name has
    an underscore of its own, so a name joined with one underscore, or the bare tool name,
    splits to its last word, which no read-only tool is."""
    result = hooks(host).run("approve", tool_name=name)
    leaf = _approve_module()._tool_leaf(name, host_record(host).approve.separators)
    if (result.exit_code, result.reply, support.crashes(result), leaf) == (0, None, [], "search"):
        raise known_bugs.Reproduced(
            f"B58: approve on {host} splits {name!r} to {leaf!r} and says nothing")
    assert support.decision_of(host, result.reply) == "allow"


@pytest.mark.parametrize("host", support.HOSTS)
def test_every_host_approves_exactly_the_read_only_tools(
        approvals: support.Runs, server_tools: dict[str, bool], host: str) -> None:
    wrong = []
    for (asked_on, name, tool), outcome in approvals.results.items():
        if asked_on != host or tool in _known_gap():
            continue
        result = support.result_of(outcome)
        decision = support.decision_of(host, result.reply)
        expected = "allow" if server_tools[tool] else None
        if (result.exit_code, decision) != (0, expected):
            wrong.append(f"{name}: exit {result.exit_code}, {decision!r} where {expected!r}")
    assert not wrong, wrong
