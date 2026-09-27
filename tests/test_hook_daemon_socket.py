"""The recall daemon's socket: it listens as soon as it binds, and it removes only its own.

Two daemons for one store can start at once, and `Daemon.run` settles which one serves:
the one whose `bind()` fails probes the path, and takes a refused probe to mean the owner
is dead. That holds only if the owner is listening by then, so the daemon listens straight
after it binds, before its slow sweep of other sockets. And a daemon that lost its path to
another daemon must not remove that daemon's socket when it exits (#344).
"""

from __future__ import annotations

import os
import pathlib
import socket
import sys
import tempfile
from typing import Callable, Iterator

import pytest

HOOKS = pathlib.Path(__file__).resolve().parents[1] / "plugin" / "hooks"
if str(HOOKS) not in sys.path:
    sys.path.insert(0, str(HOOKS))

import daemon  # noqa: E402

pytestmark = pytest.mark.skipif(not hasattr(socket, "AF_UNIX"),
                                reason="unix sockets are unavailable on this platform")


@pytest.fixture
def path() -> Iterator[str]:
    """A socket path short enough for AF_UNIX, which allows about 100 bytes."""
    folder = tempfile.mkdtemp(dir="/tmp")
    yield os.path.join(folder, "d.sock")
    for name in os.listdir(folder):
        os.unlink(os.path.join(folder, name))
    os.rmdir(folder)


def _run_with_sweep(path: str, monkeypatch: pytest.MonkeyPatch,
                    sweep: Callable[[daemon.Daemon], None]) -> int:
    """Run a daemon at `path` whose sweep is `sweep`. The sweep leaves one connection
    waiting and sets the failure count to its limit, so the daemon's first `accept`
    returns it and the daemon exits, through its own cleanup."""
    served = daemon.Daemon(path, object())

    def swept(self: daemon.Daemon) -> None:
        sweep(self)
        self.failures = daemon.MAX_CONSECUTIVE_FAILURES

    monkeypatch.setattr(daemon.Daemon, "_sweep_stale", swept)
    return served.run()


def test_the_daemon_listens_before_it_sweeps(path: str,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    """A connection made during the sweep is accepted, so a second daemon's probe in that
    window reaches a live owner rather than taking it for a dead one."""
    waiting: list[socket.socket] = []

    def connect(self: daemon.Daemon) -> None:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(self.path)
        waiting.append(client)

    try:
        assert _run_with_sweep(path, monkeypatch, connect) == 0
    finally:
        for client in waiting:
            client.close()
    assert len(waiting) == 1
    assert not os.path.exists(path), "the daemon left its own socket behind"


def test_a_daemon_leaves_a_path_it_no_longer_owns(path: str,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """Another daemon took the path while this one was running. When this one exits, the
    other daemon's socket is still there."""
    others: list[socket.socket] = []
    inodes: list[int] = []

    def replace(self: daemon.Daemon) -> None:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(self.path)
        others.append(client)
        os.unlink(self.path)
        other = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        other.bind(self.path)
        others.append(other)
        inodes.append(os.stat(self.path).st_ino)

    try:
        assert _run_with_sweep(path, monkeypatch, replace) == 0
        assert os.path.exists(path), "the daemon removed another daemon's socket"
        assert [os.stat(path).st_ino] == inodes
    finally:
        for sock in others:
            sock.close()
