"""HookRunner runs the plugin's real hook scripts, the way a client does."""

from __future__ import annotations

import json
import os
import pathlib
import sys
from types import SimpleNamespace
from typing import Callable

import pytest

from harness import stores
from harness.fakes.cli import FakeClis
from harness.hooks import (HookOutputError, HookRunner, HookTimeout, agent_clis, host_ids,
                           host_record, parse_reply)
from memvara import MemoryType

Make = Callable[..., HookRunner]
HOSTS = ("claude", "codex", "copilot", "cursor", "opencode")


@pytest.mark.parametrize("host", HOSTS)
def test_every_host_gives_each_of_its_hooks_a_timeout(host: str) -> None:
    record = host_record(host)
    for hook in record.events:
        assert record.timeouts[hook] > 0, (host, hook)


@pytest.mark.covers("hook:claude/session_start")
def test_session_start_without_a_store_says_not_configured(hook_runner: Make) -> None:
    result = hook_runner("claude").run("session_start")
    assert result.exit_code == 0
    assert result.reply is not None
    assert "not configured" in result.reply["systemMessage"]


@pytest.mark.covers("hook:claude/session_start")
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


@pytest.mark.covers("hook:claude/approve")
def test_the_approve_hook_allows_a_read_only_memvara_tool(hook_runner: Make) -> None:
    result = hook_runner("claude").run("approve", tool_name="mcp__memvara__memory_search")
    assert result.exit_code == 0
    assert result.reply is not None
    assert result.reply["hookSpecificOutput"]["permissionDecision"] == "allow"


@pytest.mark.covers("hook:claude/approve")
def test_the_approve_hook_says_nothing_about_a_write_tool(hook_runner: Make) -> None:
    result = hook_runner("claude").run("approve", tool_name="mcp__memvara__memory_forget")
    assert result.exit_code == 0
    assert result.reply is None


def test_output_that_is_not_json_is_reported_with_its_text() -> None:
    with pytest.raises(HookOutputError, match="Traceback"):
        parse_reply("Traceback (most recent call last): boom", what="recall on claude")
    assert parse_reply("", what="recall on claude") is None


def test_a_codex_client_config_is_written_as_toml(hook_runner: Make) -> None:
    """Codex keeps its MCP servers in ~/.codex/config.toml, one `[mcp_servers.<name>]`
    table each. A value with a backslash and a quote must survive the round trip."""
    if sys.version_info < (3, 11):
        pytest.skip("tomllib arrives in 3.11")
    import tomllib  # noqa: PLC0415 - Python 3.11 and later

    store = 'C:\\stores\\a "quoted" name.db'
    runner = hook_runner("codex", server_env={"MEMVARA_DB": store, "MEMVARA_USER": "tester"})
    config = tomllib.loads((runner.home / ".codex" / "config.toml").read_text())
    block = config["mcp_servers"]["memvara"]
    assert block["args"] == ["-m", "memvara.server"]
    assert block["env"] == {"MEMVARA_DB": store, "MEMVARA_USER": "tester"}


def test_a_client_config_format_the_runner_cannot_write_is_refused(
        hook_runner: Make, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = hook_runner("claude")
    monkeypatch.setattr(runner, "host", SimpleNamespace(
        id="claude", config_format="yaml", client_configs=("~/.claude.json",)))
    with pytest.raises(NotImplementedError, match="yaml"):
        runner.write_client_config({"MEMVARA_DB": "unused.db"})


def test_a_hook_that_runs_past_its_limit_is_reported_as_a_timeout(hook_runner: Make) -> None:
    with pytest.raises(HookTimeout, match="ran past"):
        hook_runner("claude").run("session_start", timeout=0.001)


def test_capture_is_refused_without_stub_agent_clis(hook_runner: Make) -> None:
    """capture starts an agent CLI to mine the turn, which would reach the network and
    spend money. HookRunner refuses it unless the test gives it stub CLIs."""
    with pytest.raises(NotImplementedError, match="stub"):
        hook_runner("claude").run("capture")


def test_the_host_ids_are_the_records_in_the_hosts_folder() -> None:
    assert host_ids() == HOSTS


def test_every_extractor_a_host_names_counts_as_an_agent_cli() -> None:
    assert {"claude", "codex", "cursor-agent", "copilot", "opencode"} <= agent_clis()


def test_no_directory_that_holds_a_real_agent_cli_is_on_the_hooks_path(
        hook_runner: Make, tmp_path: pathlib.Path) -> None:
    """The stubs stand in for claude and codex only. A real cursor-agent, copilot or
    opencode further along PATH would still be found, so its whole directory goes."""
    real = tmp_path / "real-bin"
    real.mkdir()
    (real / "cursor-agent").write_text("#!/bin/sh\nexit 0\n")
    plain = tmp_path / "plain-bin"
    plain.mkdir()
    runner = hook_runner("claude", env={"PATH": os.pathsep.join([str(real), str(plain)])})
    assert runner.environment["PATH"].split(os.pathsep) == [str(plain)]


def test_a_run_reports_the_log_lines_it_added_without_their_timestamps(
        hook_runner: Make) -> None:
    runner = hook_runner("cursor")
    first = runner.run("recall", stdin="{}", timeout=10)
    assert first.log("hooks") == ("skipped=cursor has no event for recall",)
    again = runner.run("recall", stdin="{}", timeout=10)
    assert again.log("hooks") == ("skipped=cursor has no event for recall",)
    assert again.log("capture") == ()


def _clis(tmp_path: pathlib.Path) -> FakeClis:
    if sys.platform == "win32":
        pytest.skip("the fake agent CLIs are POSIX shell scripts")
    return FakeClis(tmp_path / "clis")


def test_capture_runs_against_the_stub_clis(hook_runner: Make, tmp_path: pathlib.Path) -> None:
    clis = _clis(tmp_path)
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(json.dumps({"type": "user", "message": {"content": "ok"}}) + "\n")
    result = hook_runner("claude", stubs=clis).run("capture", transcript_path=str(transcript))
    assert result.exit_code == 0
    assert result.detached_pid is None
    assert result.log("capture") == ("turn=8c skipped=continuation",)


def test_a_detached_capture_is_waited_for(hook_runner: Make, tmp_path: pathlib.Path) -> None:
    """Codex hands capture to a child in a new session and returns at once. The runner
    waits for that child, so the logs hold what the capture did."""
    clis = _clis(tmp_path)
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(json.dumps({"type": "response_item", "payload": {
        "type": "message", "role": "user",
        "content": [{"type": "input_text", "text": "ok"}]}}) + "\n")
    result = hook_runner("codex", stubs=clis).run("capture", transcript_path=str(transcript))
    assert result.exit_code == 0
    assert result.detached_pid is not None
    assert result.log("capture") == ("turn=8c skipped=continuation",)


def test_output_that_is_not_utf8_is_reported(hook_runner: Make,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    """A client decodes a hook's stdout as UTF-8, so bytes that are not UTF-8 are a
    failure in their own right, like output that is not JSON."""
    import subprocess  # noqa: PLC0415 - only this test replaces it

    class Done:
        returncode, stdout, stderr = 0, b"\xff\xfe{}", b""

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: Done())
    with pytest.raises(HookOutputError, match="not UTF-8"):
        hook_runner("claude").run("approve", tool_name="mcp__memvara__memory_search")


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
