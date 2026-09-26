"""A recall daemon that was killed leaves its socket file behind, and the next one takes
the address over (plugin/hooks/daemon.py, `Daemon.run`)."""

from __future__ import annotations

import os
import signal
import stat
import sys
import time
from typing import Callable

import pytest

from harness.hooks import NO_UNIX_SOCKETS, HookRunner, socket_peer_pid

from .. import support

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason=NO_UNIX_SOCKETS)


def test_the_next_daemon_takes_over_the_socket_a_killed_one_left(
        hooks: Callable[..., HookRunner], store_env: dict[str, str]) -> None:
    runner = hooks("claude", env=store_env, daemon=True)
    runner.run("recall", session="one", prompt=support.PROMPT)
    sock, pid = runner.wait_for_daemon()
    os.kill(pid, signal.SIGKILL)
    deadline = time.monotonic() + 5
    while socket_peer_pid(sock) is not None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert socket_peer_pid(sock) is None
    assert sock.exists(), "a killed daemon leaves its socket file behind"

    result = runner.run("recall", session="two", prompt=support.PROMPT)
    assert support.MEMORY in support.context_of("claude", result.reply)
    new_sock, new_pid = runner.wait_for_daemon()
    assert (new_sock, new_pid != pid) == (sock, True)
    assert stat.S_IMODE(new_sock.stat().st_mode) == 0o600
    assert runner.daemon_pids() == [pid, new_pid]
