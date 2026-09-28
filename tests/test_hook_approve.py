"""The approve hook lets memvara's own read-only tools run without a prompt, and nothing else.

A host hands the hook a tool's whole name, and the only part of it that says which MCP server
the tool belongs to is the prefix. The hook used to read the name's last segment and nothing
before it, so any server whose name contained `memvara` could name a tool `memory_recall` and
have it run without a prompt. Each host now lists the exact prefixes its memvara server's
tools arrive with, and a name has to be one of those followed by a read-only tool.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

HOOKS = pathlib.Path(__file__).resolve().parent.parent / "plugin" / "hooks"
if str(HOOKS) not in sys.path:
    sys.path.insert(0, str(HOOKS))

import approve  # noqa: E402
from hosts import claude, codex, copilot, cursor, opencode  # noqa: E402

HOSTS = {h.HOST.id: h.HOST for h in (claude, codex, copilot, cursor, opencode)}


def _approved(monkeypatch, host: str, tool: str) -> bool:
    replies: list = []
    monkeypatch.setenv("MEMVARA_FEATURE_STATUS_LINE", "0")
    monkeypatch.setattr(approve, "active", lambda: HOSTS[host])
    monkeypatch.setattr(approve, "payload", lambda: {"session_id": "s", "tool_name": tool})
    monkeypatch.setattr(approve, "write", lambda _host, reply: replies.append(reply))
    assert approve.main() == 0
    return [r.decision for r in replies] == [HOSTS[host].approve.allow]


@pytest.mark.parametrize("host, tool", [
    ("claude", "mcp__memvara__memory_recall"),
    ("claude", "mcp__plugin_memvara_memvara__memory_search"),
    ("codex", "mcp__memvara__memory_why"),
    ("copilot", "memvara-memory_recall"),
    ("cursor", "mcp__memvara__memory_recall"),
    ("opencode", "mcp__memvara__memory_standing"),
])
def test_a_read_only_tool_of_memvaras_own_server_is_approved(monkeypatch, host, tool):
    assert _approved(monkeypatch, host, tool)


@pytest.mark.parametrize("host, tool", [
    ("claude", "mcp__not-memvara__memory_recall"),
    ("claude", "mcp__memvara-helper__memory_why"),
    ("claude", "mcp__evil__memvara__memory_search"),
    ("claude", "mcp__plugin_evil_memvara__memory_search"),
    ("claude", "memory_recall"),
    ("codex", "mcp__memvara_tools__memory_recall"),
    ("codex", "memory_recall"),
    ("copilot", "not-memvara-memory_recall"),
    ("copilot", "memvara-helper-memory_recall"),
    ("copilot", "memory_recall"),
    ("cursor", "mcp__memvara-evil__memory_why"),
    ("cursor", "memory_recall"),
    ("opencode", "mcp__memvara-evil__memory_why"),
    ("opencode", "memory_recall"),
])
def test_a_tool_of_any_other_server_is_not_approved(monkeypatch, host, tool):
    """Every one of these was approved before approval was pinned to a prefix."""
    assert not _approved(monkeypatch, host, tool)


@pytest.mark.parametrize("host, prefix", [
    (host, prefix) for host in sorted(HOSTS) for prefix in HOSTS[host].approve.prefixes])
def test_a_write_tool_of_memvaras_own_server_is_not_approved(monkeypatch, host, prefix):
    assert not _approved(monkeypatch, host, prefix + "memory_forget")


#: The spelling each host gives the tools of the server keyed `memvara`, which is the key
#: every installer writes: the plugin's `mcp.json`, `memvara-mcp init`, and the headless
#: capture command. Claude Code's plugin form is how this plugin's own server is named in a
#: live Claude Code session. Cursor's and OpenCode's own spellings are not measured, so they
#: keep only the one form approved there before.
PREFIXES = {
    "claude": ("mcp__memvara__", "mcp__plugin_memvara_memvara__"),
    "codex": ("mcp__memvara__",),
    "copilot": ("memvara-",),
    "cursor": ("mcp__memvara__",),
    "opencode": ("mcp__memvara__",),
}


@pytest.mark.parametrize("host", sorted(HOSTS))
def test_every_host_approves_exactly_its_own_spelling_of_the_memvara_server(host):
    """Exact per host, so one host given another's spelling fails here."""
    assert HOSTS[host].approve.prefixes == PREFIXES[host]


def test_the_capture_command_and_the_approve_hook_spell_the_server_alike():
    """The headless capture command allows memvara's tools by name. If the server key it
    builds those names from ever changed, the approve hook would stop recognising the same
    server, so the two spellings are tied together here."""
    from lib import agentic

    assert agentic.tool_name("memory_search") == PREFIXES["claude"][0] + "memory_search"
