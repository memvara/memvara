"""The recall daemon: a private socket, one daemon, and later hooks that reuse it.

The recall hook starts a daemon after it has answered, so that the next prompt costs a
socket round trip instead of an import (plugin/hooks/daemon.py). The socket is a read
interface to everything the user has stored, so it is created 0600 inside a 0700
directory, and the directory is made private again on every call, because an older
version may have left it looser (plugin/hooks/lib/ipc.py, `runtime_dir`).

A hook served by the daemon never imports the memvara library, because only the
in-process route needs it. `PYTHONPROFILEIMPORTTIME=1` makes Python list every import on
stderr, which is how these tests tell the two routes apart. The second recall is in the
same session as the first, because the first prompt of a session refreshes the standing
preferences in the hook's own process, whichever route serves the recall itself.
"""

from __future__ import annotations

import stat
import sys
from typing import Callable

import pytest

from harness.hooks import HOOKS_HOME, HookRunner, socket_peer_pid

from . import support

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason=support.NO_UNIX_SOCKETS)

Make = Callable[..., HookRunner]

#: Makes a hook list every module it imports on stderr.
TRACE = {"PYTHONPROFILEIMPORTTIME": "1"}


def imports_memvara(stderr: str) -> bool:
    """Whether a hook's import trace shows it importing the memvara package."""
    return any(line.rstrip().endswith("| memvara") for line in stderr.splitlines())


def test_one_daemon_on_a_private_socket_answers_the_recalls_after_the_first(
        hooks: Make, store_env: dict[str, str]) -> None:
    first = hooks("claude", env=store_env, daemon=True)
    run_dir = first.home / HOOKS_HOME / "run"
    run_dir.mkdir(parents=True)
    run_dir.chmod(0o755)  # as an older version of the hooks left it
    started = first.run("recall", session="one", prompt=support.PROMPT)
    assert support.MEMORY in support.context_of("claude", started.reply)
    sock, pid = first.wait_for_daemon()
    assert stat.S_IMODE(sock.stat().st_mode) == 0o600
    assert stat.S_IMODE(run_dir.stat().st_mode) == 0o700

    traced = hooks("claude", home=first.home, env={**store_env, **TRACE}, daemon=True)
    answered = traced.run("recall", session="one", prompt=support.PROMPT)
    # The daemon found the memory, and recall knew it was already in the session.
    assert support.status_of("claude", answered.reply) == (
        "⋈ Memvara · 1 already in context")
    assert not imports_memvara(answered.stderr), "the hook read the store itself"
    assert first.daemon_sockets() == [sock]
    assert socket_peer_pid(sock) == pid
    assert first.daemon_pids() == [pid], "a second daemon was started"


def test_a_hook_with_no_daemon_imports_the_library_itself(
        hooks: Make, store_env: dict[str, str]) -> None:
    """The control for the test above: the import trace does show the library when the
    hook reads the store in its own process."""
    runner = hooks("claude", env={**store_env, **TRACE})
    result = runner.run("recall", session="one", prompt=support.PROMPT)
    assert support.MEMORY in support.context_of("claude", result.reply)
    assert imports_memvara(result.stderr)
    assert runner.daemon_sockets() == []
