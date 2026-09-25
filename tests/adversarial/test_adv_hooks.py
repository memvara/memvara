"""HookRunner runs the plugin's real hook scripts, the way a client does."""

from __future__ import annotations

import pathlib
from types import SimpleNamespace
from typing import Callable

import pytest

from harness import stores
from harness.hooks import HookOutputError, HookRunner, HookTimeout, host_record, parse_reply
from memvara import MemoryType

Make = Callable[..., HookRunner]
HOSTS = ("claude", "codex", "copilot", "cursor", "opencode")


@pytest.mark.parametrize("host", HOSTS)
def test_every_host_gives_each_of_its_hooks_a_timeout(host: str) -> None:
    record = host_record(host)
    for hook in record.events:
        assert record.timeouts[hook] > 0, (host, hook)


def test_session_start_without_a_store_says_not_configured(hook_runner: Make) -> None:
    result = hook_runner("claude").run("session_start")
    assert result.exit_code == 0
    assert result.reply is not None
    assert "not configured" in result.reply["systemMessage"]


def test_session_start_reads_the_store_the_client_config_names(
        hook_runner: Make, tmp_path: pathlib.Path) -> None:
    db = tmp_path / "memory.db"
    mem = stores.file(db)
    mem.scope(user="tester").remember("user", "prefers", "tabs for indentation",
                                      memory_type=MemoryType.PROCEDURAL)
    mem.close()
    runner = hook_runner("claude", server_env={"MEMVARA_DB": str(db), "MEMVARA_USER": "tester"})
    result = runner.run("session_start")
    assert result.exit_code == 0
    assert result.reply is not None
    assert "session opened with" in result.reply["systemMessage"]
    assert "tabs for indentation" in result.reply["hookSpecificOutput"]["additionalContext"]


def test_the_approve_hook_allows_a_read_only_memvara_tool(hook_runner: Make) -> None:
    result = hook_runner("claude").run("approve", tool_name="mcp__memvara__memory_search")
    assert result.exit_code == 0
    assert result.reply is not None
    assert result.reply["hookSpecificOutput"]["permissionDecision"] == "allow"


def test_the_approve_hook_says_nothing_about_a_write_tool(hook_runner: Make) -> None:
    result = hook_runner("claude").run("approve", tool_name="mcp__memvara__memory_forget")
    assert result.exit_code == 0
    assert result.reply is None


def test_output_that_is_not_json_is_reported_with_its_text() -> None:
    with pytest.raises(HookOutputError, match="Traceback"):
        parse_reply("Traceback (most recent call last): boom", what="recall on claude")
    assert parse_reply("", what="recall on claude") is None


def test_a_toml_client_config_is_refused_with_the_reason(hook_runner: Make) -> None:
    """Codex keeps its client config in TOML, and HookRunner writes JSON only. The
    refusal is deliberate and must say so, so nobody mistakes it for a harness bug."""
    with pytest.raises(NotImplementedError, match="writes JSON client configs only"):
        hook_runner("codex", server_env={"MEMVARA_DB": "unused.db"})


def test_a_hook_that_runs_past_its_limit_is_reported_as_a_timeout(hook_runner: Make) -> None:
    with pytest.raises(HookTimeout, match="ran past"):
        hook_runner("claude").run("session_start", timeout=0.001)


def test_capture_is_refused_until_the_agent_clis_are_stubbed(hook_runner: Make) -> None:
    """capture can start the real agent CLI, which would reach the network and spend
    money. HookRunner refuses it until the hook-conformance tests put stubs on PATH."""
    with pytest.raises(NotImplementedError, match="stub"):
        hook_runner("claude").run("capture")


def test_json_that_is_not_an_object_is_reported_with_stderr() -> None:
    with pytest.raises(HookOutputError, match="the stderr text"):
        parse_reply("[1, 2]", what="recall on claude", stderr="the stderr text")


def test_a_client_config_outside_the_home_directory_is_refused(
        hook_runner: Make, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = hook_runner("claude")
    monkeypatch.setattr(runner, "host", SimpleNamespace(
        id="claude", config_format="json", client_configs=("/etc/memvara.json",)))
    with pytest.raises(ValueError, match="outside the test's home"):
        runner.write_client_config({"MEMVARA_DB": "unused.db"})
