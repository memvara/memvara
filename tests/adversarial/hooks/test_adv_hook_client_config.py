"""Where each host's hooks look for the store.

The hooks run in the client's environment, not the MCP server's, so they find the store
by reading the memvara server block in the client config files the host record lists
(plugin/hooks/lib/ipc.py, `server_env`), and a variable set in their own environment wins
over that block (`client_env`). `HookRunner(server_env=...)` writes the block where and
how the host itself keeps its MCP servers (`harness.hooks.CLIENT_CONFIGS`), which is where
a user who runs a local store configures it. Claude Code and Copilot keep their MCP
servers in JSON under `mcpServers`, Cursor in `~/.cursor/mcp.json`, Codex in TOML under
`mcp_servers`, and OpenCode under `mcp` with the variables in `environment`. The hooks
used to read only the first shape, and only from files Cursor does not keep its servers
in, so on Codex, Cursor and OpenCode a local store was never found (#341, B59).
"""

from __future__ import annotations

import pathlib
from typing import Callable

import pytest

from harness.hooks import HookRunner

from . import support

Make = Callable[..., HookRunner]

@pytest.mark.parametrize("host", ["claude", "copilot", "codex", "cursor", "opencode"])
def test_the_hooks_find_the_store_their_hosts_own_mcp_config_names(
        hooks: Make, store_env: dict[str, str], host: str) -> None:
    result = hooks(host, server_env=store_env).run("session_start")
    assert result.exit_code == 0
    assert support.MEMORY in support.context_of(host, result.reply)


def test_a_store_named_in_the_hooks_own_environment_wins_over_the_client_config(
        hooks: Make, store_env: dict[str, str], tmp_path: pathlib.Path) -> None:
    """Someone who exports MEMVARA_DB to point a session at another store means it."""
    empty = support.make_store(tmp_path / "empty.db", memory=False)
    runner = hooks("claude", env=store_env,
                   server_env=support.store_env(empty))
    result = runner.run("session_start")
    assert support.MEMORY in support.context_of("claude", result.reply)


def test_codex_finds_a_store_named_in_its_hooks_environment(
        hooks: Make, store_env: dict[str, str]) -> None:
    result = hooks("codex", env=store_env).run("session_start")
    assert result.exit_code == 0
    assert support.MEMORY in support.context_of("codex", result.reply)
