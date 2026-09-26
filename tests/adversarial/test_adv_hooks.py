"""HookRunner runs the plugin's real hook scripts, the way a client does."""

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Callable, Iterator

import pytest

from harness import hooks as hooks_module
from harness import skips, stores
from harness.env import child_env
from harness.fakes.cli import NO_FAKES, FakeClis, HangingClis
from harness.hooks import (NO_UNIX_SOCKETS, HookOutputError, HookRunner, HookTimeout,
                           agent_clis, daemon_socket_path, host_ids, host_record,
                           parse_reply, process_alive, short_dir, socket_peer_pid)
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


def test_a_host_whose_mcp_config_the_runner_does_not_know_is_refused(
        hook_runner: Make, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = hook_runner("claude")
    monkeypatch.setattr(runner, "host", SimpleNamespace(id="nohost"))
    with pytest.raises(NotImplementedError, match="nohost"):
        runner.write_client_config({"MEMVARA_DB": "unused.db"})


#: Where and how each host keeps its MCP servers, written out by hand.
MCP_CONFIGS = {
    "claude": (".claude.json", lambda data: data["mcpServers"]["memvara"]["env"]),
    "copilot": (".copilot/mcp-config.json",
                lambda data: data["mcpServers"]["memvara"]["env"]),
    "cursor": (".cursor/mcp.json", lambda data: data["mcpServers"]["memvara"]["env"]),
    "opencode": (".config/opencode/opencode.json",
                 lambda data: data["mcp"]["memvara"]["environment"]),
}


@pytest.mark.parametrize("host", sorted(MCP_CONFIGS))
def test_the_store_is_written_where_and_how_the_host_keeps_its_mcp_servers(
        hook_runner: Make, host: str) -> None:
    """Where a user who runs a local store configures it, so a test can check that the
    hooks look there too. Codex's TOML is checked above."""
    server_env = {"MEMVARA_DB": "store.db", "MEMVARA_USER": "tester"}
    runner = hook_runner(host, server_env=server_env)
    relative, env_of = MCP_CONFIGS[host]
    assert env_of(json.loads((runner.home / relative).read_text())) == server_env


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


def test_capture_runs_against_the_stub_clis(hook_runner: Make, clis: FakeClis,
                                            tmp_path: pathlib.Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(json.dumps({"type": "user", "message": {"content": "ok"}}) + "\n")
    result = hook_runner("claude", stubs=clis).run("capture", transcript_path=str(transcript))
    assert result.exit_code == 0
    assert result.detached_pid is None
    assert result.log("capture") == ("turn=8c skipped=continuation",)


def test_a_detached_capture_is_waited_for(hook_runner: Make, clis: FakeClis,
                                          tmp_path: pathlib.Path) -> None:
    """Codex hands capture to a child in a new session and returns at once. The runner
    waits for that child, so the logs hold what the capture did."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(json.dumps({"type": "response_item", "payload": {
        "type": "message", "role": "user",
        "content": [{"type": "input_text", "text": "ok"}]}}) + "\n")
    result = hook_runner("codex", stubs=clis).run("capture", transcript_path=str(transcript))
    assert result.exit_code == 0
    assert result.detached_pid is not None
    assert result.log("capture") == ("turn=8c skipped=continuation",)


def test_the_peer_pid_of_a_socket_is_the_process_listening_on_it() -> None:
    if sys.platform == "win32":
        pytest.skip(NO_UNIX_SOCKETS)
    directory = short_dir("hooks")
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


def _socket_fits_on_macos(home: pathlib.Path) -> bool:
    """Whether macOS accepts the daemon's socket path under `home`, as the hooks see it.

    macOS's `sockaddr_un.sun_path` holds 104 bytes, and the path must end with a NUL byte
    inside them. The hooks get the home through `child_env`, which resolves it.
    """
    return len(os.fsencode(daemon_socket_path(home.resolve()))) < 104


@pytest.mark.parametrize("prefix", ["home", "hooks"])
@pytest.mark.parametrize("length", [30, 36, 37, 38, 40, 41])
def test_a_short_dir_leaves_room_for_the_daemon_socket_under_any_temporary_directory(
        prefix: str, length: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """A daemon test's home comes from short_dir, and the recall daemon's socket goes under
    it. short_dir must fall back to /tmp whenever a home in the system's temporary directory
    would be too long for the socket, and only then. It used to fall back only for a
    temporary directory longer than 40 characters, so a TMPDIR of 38 to 40 characters made
    every daemon test time out."""
    if sys.platform == "win32":
        pytest.skip(NO_UNIX_SOCKETS)
    root = pathlib.Path(tempfile.mkdtemp(prefix="b", dir="/tmp")).resolve()
    home = None
    try:
        base = root / ("p" * (length - len(str(root)) - 1))
        base.mkdir()
        assert len(str(base)) == length
        monkeypatch.setattr(tempfile, "tempdir", str(base))
        home = short_dir(prefix)
        assert _socket_fits_on_macos(home), (base, home)
        if home.parent.resolve() != base:
            assert not _socket_fits_on_macos(base / home.name), (
                f"{base} was short enough for the socket, and short_dir did not use it")
    finally:
        shutil.rmtree(root, ignore_errors=True)
        if home is not None and root not in home.resolve().parents:
            shutil.rmtree(home, ignore_errors=True)


def test_a_short_dir_measures_the_directory_a_symbolic_link_leads_to(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A short TMPDIR can be a symbolic link to a long directory, as /tmp is a link to
    /private/tmp on macOS. child_env resolves a home before the hooks see it, so the
    socket's path is as long as the resolved home, and short_dir must measure that."""
    if sys.platform == "win32":
        pytest.skip(NO_UNIX_SOCKETS)
    root = pathlib.Path(tempfile.mkdtemp(prefix="b", dir="/tmp")).resolve()
    home = None
    try:
        target = root / ("t" * 50)
        target.mkdir()
        link = root / "l"
        link.symlink_to(target, target_is_directory=True)
        monkeypatch.setattr(tempfile, "tempdir", str(link))
        home = short_dir("home")
        assert _socket_fits_on_macos(home), (home, home.resolve())
    finally:
        shutil.rmtree(root, ignore_errors=True)
        if home is not None and root not in home.resolve().parents:
            shutil.rmtree(home, ignore_errors=True)


def test_daemon_socket_path_is_where_the_hooks_put_the_daemons_socket() -> None:
    """daemon_socket_path copies how plugin/hooks/lib/ipc.py names the recall daemon's
    socket. This asks the hooks' own `socket_path` for the path, in a child process started
    the way the hooks are, and compares the two: the same directory, and a name of the same
    length."""
    if sys.platform == "win32":
        pytest.skip(NO_UNIX_SOCKETS)
    home = short_dir("hooks")
    try:
        ask = ("import sys; sys.path.insert(0, sys.argv[1]); from lib import ipc; "
               "print(ipc.socket_path('a store'))")
        done = subprocess.run([sys.executable, "-c", ask, str(hooks_module.HOOKS_DIR)],
                              env=child_env(home), capture_output=True, text=True,
                              timeout=60, check=True)
        real = pathlib.Path(done.stdout.strip())
        copy = daemon_socket_path(home.resolve())
        assert real.parent == copy.parent
        assert re.fullmatch(r"recall-[0-9a-f]{16}\.sock", real.name), real.name
        assert len(real.name) == len(copy.name)
    finally:
        shutil.rmtree(home, ignore_errors=True)


def test_the_daemon_option_lets_the_recall_hook_start_its_daemon(hook_runner: Make) -> None:
    """child_env forbids the daemon, because it outlives the hook. daemon=True lifts that."""
    assert hook_runner("claude").environment["MEMVARA_DAEMON"] == "1"
    if sys.platform == "win32":
        pytest.skip(NO_UNIX_SOCKETS)
    assert "MEMVARA_DAEMON" not in hook_runner("claude", daemon=True).environment


def test_the_daemon_option_is_refused_where_there_are_no_unix_sockets(
        hook_runner: Make, monkeypatch: pytest.MonkeyPatch) -> None:
    """The refusal uses the words of the skip ledger's rule for that reason, so a test that
    skips with those words is explained on Windows."""
    monkeypatch.delattr(socket, "AF_UNIX", raising=False)
    with pytest.raises(NotImplementedError, match="unix socket") as refused:
        hook_runner("claude", daemon=True)
    assert skips.explained(str(refused.value), platform="win32")


def _daemon_runner(tmp_path: pathlib.Path, home: pathlib.Path) -> HookRunner:
    """A runner that allows the daemon, over a store whose one memory the prompt
    "user lives in Lisbon" matches, so one recall reads the store once."""
    db = tmp_path / "memory.db"
    with stores.file(db) as mem:
        mem.scope(user="tester").remember("user", "lives_in", "Lisbon")
    return HookRunner("claude", home=home, cwd=tmp_path, daemon=True,
                      env={"MEMVARA_DB": str(db), "MEMVARA_USER": "tester"})


def test_close_stops_the_daemon_a_recall_started(tmp_path: pathlib.Path) -> None:
    if sys.platform == "win32":
        pytest.skip(NO_UNIX_SOCKETS)
    home = short_dir("hooks")
    runner = _daemon_runner(tmp_path, home)
    try:
        runner.run("recall", prompt="user lives in Lisbon")
        sock, pid = runner.wait_for_daemon()
        assert runner.daemon_sockets() == [sock]
        assert runner.daemon_pids() == [pid]
    finally:
        runner.close()
        shutil.rmtree(home, ignore_errors=True)
    assert socket_peer_pid(sock) is None
    assert not sock.exists()
    assert not process_alive(pid)


def test_close_stops_a_daemon_that_no_socket_path_leads_to(tmp_path: pathlib.Path) -> None:
    """A daemon can keep running on a socket whose path was removed, where close() cannot
    find it by its socket. The runner records every daemon the hooks start, so close()
    stops that one too."""
    if sys.platform == "win32":
        pytest.skip(NO_UNIX_SOCKETS)
    home = short_dir("hooks")
    runner = _daemon_runner(tmp_path, home)
    try:
        runner.run("recall", prompt="user lives in Lisbon")
        sock, pid = runner.wait_for_daemon()
        sock.unlink()
    finally:
        runner.close()
        shutil.rmtree(home, ignore_errors=True)
    assert not process_alive(pid)


@pytest.fixture
def stopped_by_teardown() -> Iterator[list[int]]:
    """Pids that must have stopped once the other fixtures are torn down. Requested before
    them, this fixture is set up first and so torn down last, after them."""
    pids: list[int] = []
    yield pids
    assert [pid for pid in pids if process_alive(pid)] == []


def test_the_hook_runner_fixture_closes_the_runners_it_made(
        stopped_by_teardown: list[int], hook_runner: Make, tmp_path: pathlib.Path) -> None:
    """A capture handed to a child and not waited for keeps running after its hook
    returns, here with a codex that never answers. The fixture closes its runners when
    the test ends, and closing kills the child and the stub it started."""
    if sys.platform == "win32":
        pytest.skip(NO_FAKES)
    stubs = HangingClis(tmp_path / "hanging")
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(json.dumps({"type": "response_item", "payload": {
        "type": "message", "role": "user",
        "content": [{"type": "input_text", "text": "Please remember that I live in Lisbon."}]}})
        + "\n")
    db = tmp_path / "memory.db"
    stores.file(db).close()
    runner = hook_runner("codex", stubs=stubs,
                         env={"MEMVARA_DB": str(db), "MEMVARA_USER": "tester"})
    result = runner.run("capture", transcript_path=str(transcript), wait_detached=False)
    assert result.detached_pid is not None and process_alive(result.detached_pid)
    stopped_by_teardown.append(result.detached_pid)


def test_a_process_that_has_ended_but_is_not_reaped_counts_as_ended() -> None:
    """A killed daemon or capture child is reaped by whatever adopted it, which can be
    slow, or never happen when the test runs as process 1 in a container. Until then it
    is a zombie, which still answers signal 0, so the harness reads its state instead."""
    if sys.platform == "win32":
        pytest.skip(NO_UNIX_SOCKETS)
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        deadline = time.monotonic() + 10
        while "Z" not in subprocess.run(["ps", "-o", "stat=", "-p", str(child.pid)],
                                        capture_output=True, text=True).stdout:
            assert time.monotonic() < deadline, "the child never became a zombie"
            time.sleep(0.02)
        assert not process_alive(child.pid)
        assert process_alive(os.getpid())
    finally:
        child.wait()


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
        hook_runner: Make, clis: FakeClis, tmp_path: pathlib.Path) -> None:
    """run.py starts that child afresh, so a patch would not reach the capture."""
    runner = hook_runner("codex", stubs=clis, patches={"lib.extract.TIMEOUT_SEC": 1.0})
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
    monkeypatch.setitem(hooks_module.CLIENT_CONFIGS, "claude",
                        ("/etc/memvara.json", "mcpServers"))
    with pytest.raises(ValueError, match="outside the test's home"):
        runner.write_client_config({"MEMVARA_DB": "unused.db"})
