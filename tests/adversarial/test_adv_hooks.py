"""HookRunner runs the plugin's real hook scripts, the way a client does."""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import socket
import sys
import tempfile
from types import SimpleNamespace
from typing import Callable

import pytest

from harness import stores
from harness.fakes.cli import FakeClis
from harness.hooks import (HookOutputError, HookRunner, HookTimeout, agent_clis, host_ids,
                           host_record, parse_reply, socket_peer_pid)
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


#: Why a test of the recall daemon skips on Windows. It has a rule in harness/skips.py.
NO_UNIX_SOCKETS = "the recall daemon listens on a unix socket, which Windows lacks"


def _short_dir() -> pathlib.Path:
    """A private directory with a short path. macOS refuses a unix socket path longer
    than 104 bytes, which a directory under a long TMPDIR can pass on its own."""
    return pathlib.Path(tempfile.mkdtemp(prefix="mv-hooks-", dir="/tmp"))


def test_the_peer_pid_of_a_socket_is_the_process_listening_on_it() -> None:
    if sys.platform == "win32":
        pytest.skip(NO_UNIX_SOCKETS)
    directory = _short_dir()
    path = directory / "s.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(path))
        server.listen(1)
        assert socket_peer_pid(path) == os.getpid()
    finally:
        server.close()
        shutil.rmtree(directory, ignore_errors=True)
    assert socket_peer_pid(path) is None


def test_the_daemon_option_lets_the_recall_hook_start_its_daemon(hook_runner: Make) -> None:
    """child_env forbids the daemon, because it outlives the hook. daemon=True lifts that."""
    assert hook_runner("claude").environment["MEMVARA_DAEMON"] == "1"
    if sys.platform == "win32":
        pytest.skip(NO_UNIX_SOCKETS)
    assert "MEMVARA_DAEMON" not in hook_runner("claude", daemon=True).environment


def test_the_daemon_option_is_refused_where_there_are_no_unix_sockets(
        hook_runner: Make, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(socket, "AF_UNIX", raising=False)
    with pytest.raises(NotImplementedError, match="unix socket"):
        hook_runner("claude", daemon=True)


def test_close_stops_the_daemon_a_recall_started(tmp_path: pathlib.Path) -> None:
    if sys.platform == "win32":
        pytest.skip(NO_UNIX_SOCKETS)
    db = tmp_path / "memory.db"
    stores.file(db).close()
    home = _short_dir()
    runner = HookRunner("claude", home=home, cwd=tmp_path, daemon=True,
                        env={"MEMVARA_DB": str(db), "MEMVARA_USER": "tester"})
    try:
        runner.run("recall", prompt="where does the user live")
        sock, pid = runner.wait_for_daemon()
        assert runner.daemon_sockets() == [sock]
    finally:
        runner.close()
        shutil.rmtree(home, ignore_errors=True)
    assert socket_peer_pid(sock) is None
    assert not sock.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_a_patch_reaches_the_hook_process(hook_runner: Make) -> None:
    """With recall's budget for optional work patched to nothing, the hook skips that work
    and says so, which shows the patch was in place before the hook ran."""
    runner = hook_runner("claude", patches={"recall.OVERALL_BUDGET_SEC": 0.0})
    result = runner.run("recall", prompt="where does the user live")
    assert result.exit_code == 0
    assert "skipped=standing refresh, budget exhausted" in result.log("recall")


def test_a_patch_that_names_nothing_the_hooks_have_is_refused(hook_runner: Make) -> None:
    """A limit that was renamed would otherwise leave a test waiting out the real one,
    or passing without having shrunk anything."""
    runner = hook_runner("claude", patches={"recall.NO_SUCH_LIMIT": 1.0})
    with pytest.raises(ValueError, match="NO_SUCH_LIMIT"):
        runner.run("recall", prompt="where does the user live")


def test_patches_are_refused_for_a_capture_the_host_hands_to_a_child(
        hook_runner: Make, tmp_path: pathlib.Path) -> None:
    """run.py starts that child afresh, so a patch would not reach the capture."""
    runner = hook_runner("codex", stubs=_clis(tmp_path),
                         patches={"lib.extract.TIMEOUT_SEC": 1.0})
    with pytest.raises(ValueError, match="child"):
        runner.run("capture", transcript_path=str(tmp_path / "t.jsonl"))


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
