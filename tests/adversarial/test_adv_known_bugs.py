"""Confirmed bugs, each pinned by a strict expected failure that cites its issue.

Each test states the behaviour the fix must produce. Until then it fails in exactly the
way the bug fails, and `known_bugs.xfail` names that failure.
"""

from __future__ import annotations

import sys
from typing import Callable

import pytest

from harness import known_bugs, stores
from harness.hooks import HOOKS_DIR, HookRunner
from harness.stdio import McpProcess, McpProcessError
from memvara import MemoryType
from memvara.server.tools import TOOLS

if str(HOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(HOOKS_DIR))
import approve  # noqa: E402 - plugin/hooks is not a package; the path is set just above

LEVELS = ("session", "agent")


def _live(scoped: object) -> list[str]:
    return sorted(claim.object for claim in scoped.get_all())  # type: ignore[attr-defined]


# -- B2: a bound write ends the user-wide value --------------------------------------

@pytest.mark.parametrize("level", LEVELS)
@known_bugs.xfail("B2")
def test_a_bound_write_leaves_the_user_wide_value_live(level: str) -> None:
    mem = stores.memory()
    mem.scope(user="u").remember("user", "lives_in", "Berlin")
    mem.scope(user="u", **{level: "one"}).remember("user", "lives_in", "Paris")
    assert _live(mem.scope(user="u")) == ["Berlin"]


@pytest.mark.parametrize("level", LEVELS)
@known_bugs.xfail("B2")
def test_a_bound_write_leaves_its_siblings_on_the_user_wide_value(level: str) -> None:
    mem = stores.memory()
    mem.scope(user="u").remember("user", "lives_in", "Berlin")
    mem.scope(user="u", **{level: "one"}).remember("user", "lives_in", "Paris")
    assert _live(mem.scope(user="u", **{level: "two"})) == ["Berlin"]


@pytest.mark.parametrize("level", LEVELS)
def test_a_bound_scope_reads_its_own_value_in_place_of_the_user_wide_one(level: str) -> None:
    """Passes today and must keep passing after the B2 fix: the shadow side of the rule."""
    mem = stores.memory()
    mem.scope(user="u").remember("user", "lives_in", "Berlin")
    mem.scope(user="u", **{level: "one"}).remember("user", "lives_in", "Paris")
    assert _live(mem.scope(user="u", **{level: "one"})) == ["Paris"]


# -- B3: auto-approve misses two read-only tools --------------------------------------

@known_bugs.xfail("B3")
def test_the_approve_hook_allows_exactly_the_servers_read_only_tools() -> None:
    assert set(approve.READ_ONLY) == {tool.name for tool in TOOLS if not tool.writes}


@pytest.mark.parametrize("tool", ["memory_get_document", "memory_list_documents"])
@known_bugs.xfail("B3")
def test_the_approve_hook_allows_the_read_only_document_tools(
        hook_runner: Callable[..., HookRunner], tool: str) -> None:
    result = hook_runner("claude").run("approve", tool_name=f"mcp__memvara__{tool}")
    assert result.exit_code == 0
    assert result.reply is not None, f"the approve hook printed no decision for {tool}"
    assert result.reply["hookSpecificOutput"]["permissionDecision"] == "allow"


# -- B4: deeply nested JSON kills the server ------------------------------------------

@known_bugs.xfail("B4", raises=McpProcessError)
def test_one_deeply_nested_request_does_not_kill_the_server(
        mcp: Callable[..., McpProcess]) -> None:
    server = mcp()
    server.initialize()
    depth = 100_000
    server.send_raw('{"jsonrpc":"2.0","id":99,"method":"ping","params":'
                    + "[" * depth + "]" * depth + "}")
    reply = server.recv(timeout=20)
    assert "error" in reply and reply.get("id") in (None, 99), reply
    assert server.request("ping") == {}


# -- B5: memory_standing with k=0 --------------------------------------------------------

@known_bugs.xfail("B5")
def test_memory_standing_with_k_zero_never_reports_an_empty_store(
        mcp: Callable[..., McpProcess]) -> None:
    server = mcp()
    server.initialize()
    stored = server.call("memory_remember", subject="user", predicate="prefers",
                         object="tabs for indentation", memory_type="procedural")
    assert not stored.is_error, stored.text
    reply = server.call("memory_standing", k=0)
    assert "No standing preferences are stored" not in reply.text, reply.text


# -- B6: a string memory_type -----------------------------------------------------------

@known_bugs.xfail("B6", raises=AttributeError)
def test_remember_takes_a_string_memory_type_or_refuses_it_by_name() -> None:
    user = stores.memory().scope(user="u")
    try:
        # A string on purpose: it is the spelling the MCP tool accepts.
        user.remember("user", "prefers", "tabs", memory_type="procedural")  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        assert "memory_type" in str(exc), exc
        return
    [claim] = user.get_all()
    assert claim.memory_type is MemoryType.PROCEDURAL
