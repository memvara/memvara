"""Confirmed bugs, each pinned by a strict expected failure that cites its issue.

Each test states the behaviour the fix must produce. It raises known_bugs.Reproduced only
when it has seen that bug's own symptom, and the bug's marker accepts nothing else, so a
different failure in the same test fails loudly instead of passing for the known bug.
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

#: The two read-only tools that #267 is about.
B3_MISSING = {"memory_get_document", "memory_list_documents"}


def _live(scoped: object) -> list[str]:
    return sorted(claim.object for claim in scoped.get_all())  # type: ignore[attr-defined]


def _ended(scoped: object) -> list[str]:
    return sorted(claim.object  # type: ignore[attr-defined]
                  for claim in scoped.get_all(states=("ended",)))  # type: ignore[attr-defined]


def _read_only_tools() -> set[str]:
    return {tool.name for tool in TOOLS if not tool.writes}


def test_a_known_bug_marker_accepts_only_the_bugs_own_symptom() -> None:
    """Each marker absorbs only known_bugs.Reproduced, which a test raises after it has
    seen the bug's exact symptom. Any other failure in the same test fails loudly."""
    mark = known_bugs.xfail("B2").mark
    assert mark.kwargs["strict"] is True
    assert mark.kwargs["raises"] is known_bugs.Reproduced


# -- B2: a bound write ends the user-wide value --------------------------------------

@pytest.mark.parametrize("level", LEVELS)
@known_bugs.xfail("B2")
def test_a_bound_write_leaves_the_user_wide_value_live(level: str) -> None:
    mem = stores.memory()
    user = mem.scope(user="u")
    user.remember("user", "lives_in", "Berlin")
    mem.scope(user="u", **{level: "one"}).remember("user", "lives_in", "Paris")
    live = _live(user)
    if not live and _ended(user) == ["Berlin"]:
        raise known_bugs.Reproduced(f"B2: the {level}-bound write ended the user-wide value")
    assert live == ["Berlin"]


@pytest.mark.parametrize("level", LEVELS)
@known_bugs.xfail("B2")
def test_a_bound_write_leaves_its_siblings_on_the_user_wide_value(level: str) -> None:
    mem = stores.memory()
    user = mem.scope(user="u")
    user.remember("user", "lives_in", "Berlin")
    mem.scope(user="u", **{level: "one"}).remember("user", "lives_in", "Paris")
    sibling = _live(mem.scope(user="u", **{level: "two"}))
    if not sibling and _ended(user) == ["Berlin"]:
        raise known_bugs.Reproduced(f"B2: a sibling {level} lost the user-wide value")
    assert sibling == ["Berlin"]


@pytest.mark.parametrize("level", LEVELS)
def test_a_bound_scope_reads_its_own_value_in_place_of_the_user_wide_one(level: str) -> None:
    """Passes today and must keep passing after the B2 fix: the shadow side of the rule."""
    mem = stores.memory()
    mem.scope(user="u").remember("user", "lives_in", "Berlin")
    mem.scope(user="u", **{level: "one"}).remember("user", "lives_in", "Paris")
    assert _live(mem.scope(user="u", **{level: "one"})) == ["Paris"]


# -- B3: auto-approve misses two read-only tools --------------------------------------

def test_the_approve_hook_differs_from_the_server_only_by_the_known_bug() -> None:
    """Passes today and after the fix for #267. Any other difference between the hook's
    list and the server's read-only tools fails here at once."""
    allowed, read_only = set(approve.READ_ONLY), _read_only_tools()
    assert allowed <= read_only, allowed - read_only
    assert read_only - allowed <= B3_MISSING, read_only - allowed


@known_bugs.xfail("B3")
def test_the_approve_hook_allows_exactly_the_servers_read_only_tools() -> None:
    if _read_only_tools() - set(approve.READ_ONLY) == B3_MISSING:
        raise known_bugs.Reproduced("B3: the approve list misses the two document tools")
    assert set(approve.READ_ONLY) == _read_only_tools()


@pytest.mark.parametrize("tool", sorted(B3_MISSING))
@known_bugs.xfail("B3")
def test_the_approve_hook_allows_the_read_only_document_tools(
        hook_runner: Callable[..., HookRunner], tool: str) -> None:
    result = hook_runner("claude").run("approve", tool_name=f"mcp__memvara__{tool}")
    assert result.exit_code == 0
    if result.reply is None:
        raise known_bugs.Reproduced(f"B3: the approve hook printed no decision for {tool}")
    assert result.reply["hookSpecificOutput"]["permissionDecision"] == "allow"


# -- B4: deeply nested JSON kills the server ------------------------------------------

@known_bugs.xfail("B4")
def test_one_deeply_nested_request_does_not_kill_the_server(
        mcp: Callable[..., McpProcess]) -> None:
    server = mcp()
    server.initialize()
    depth = 100_000
    server.send_raw('{"jsonrpc":"2.0","id":99,"method":"ping","params":'
                    + "[" * depth + "]" * depth + "}")
    try:
        reply = server.recv(timeout=20)
    except McpProcessError as exc:
        if "RecursionError" in str(exc):
            raise known_bugs.Reproduced("B4: the server died of a RecursionError") from exc
        raise
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
    if reply.text.startswith("No standing preferences are stored"):
        raise known_bugs.Reproduced("B5: k=0 reported an empty store")
    # #269 accepts either fix: refusing k below 1, or any answer that is not the
    # empty-store one. Nothing more is asserted, so either fix makes this test pass.


# -- B6: a string memory_type -----------------------------------------------------------

@known_bugs.xfail("B6")
def test_remember_takes_a_string_memory_type_or_refuses_it_by_name() -> None:
    user = stores.memory().scope(user="u")
    try:
        # A string on purpose: it is the spelling the MCP tool accepts.
        user.remember("user", "prefers", "tabs", memory_type="procedural")  # type: ignore[arg-type]
    except AttributeError as exc:
        if "'str' object has no attribute 'value'" in str(exc):
            raise known_bugs.Reproduced("B6: remember() used the string as an enum") from exc
        raise
    except (TypeError, ValueError) as exc:
        assert "memory_type" in str(exc), exc
        return
    [claim] = user.get_all()
    assert claim.memory_type is MemoryType.PROCEDURAL
