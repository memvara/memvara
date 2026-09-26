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

A recall whose first read finds nothing starts two daemons, which B62 (#344) pins.
"""

from __future__ import annotations

import pathlib
import stat
import sys
import time
from typing import Callable

import pytest

from harness import known_bugs
from harness.hooks import HOOKS_HOME, NO_UNIX_SOCKETS, HookRunner, socket_peer_pid

from . import support

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason=NO_UNIX_SOCKETS)

Make = Callable[..., HookRunner]

#: Makes a hook list every module it imports on stderr.
TRACE = {"PYTHONPROFILEIMPORTTIME": "1"}


def imports_memvara(stderr: str) -> bool:
    """Whether a hook's import trace shows it importing the memvara package, at any
    depth: the trace indents a module imported from inside another one."""
    return any(line.startswith("import time:") and line.rsplit("|", 1)[-1].strip() == "memvara"
               for line in stderr.splitlines())


def test_the_import_trace_check_sees_the_package_imported_at_any_depth() -> None:
    """The trace indents a module imported from inside another one. If the hooks ever
    imported memvara that way, a check that saw only top-level imports would pass a hook
    that read the store itself."""
    trace = "import time: self [us] | cumulative | imported package\n{}\n"
    assert imports_memvara(trace.format("import time:       168 |     140683 | memvara"))
    assert imports_memvara(trace.format("import time:       168 |     140683 |   memvara"))
    assert not imports_memvara(trace.format("import time:        15 |       6418 | memvara_x"))


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


# -- known bugs --------------------------------------------------------------------------

@known_bugs.xfail("B62")
def test_a_recall_whose_first_read_finds_nothing_starts_one_daemon(
        hooks: Make, tmp_path: pathlib.Path) -> None:
    """lib/fast.py, `recall`, starts the daemon after every read it makes in its own
    process, and recall.py reads a second time, wider, when the first read finds nothing
    fresh. So a prompt that matches nothing, with no daemon running, starts two daemons
    for one store. This counts the starts, which does not depend on which of the two ends
    up listening.

    Each daemon records its pid as soon as its Python starts, long before either daemon
    listens. In 40 of 40 measured runs, 20 of them with four extra processes keeping the
    cores busy, both pids were recorded by the time `wait_for_daemon` returned. The test
    still waits up to 5 seconds for a second pid, so that a start delayed by load cannot
    make the strict pin pass. While the bug is present, the wait ends as soon as the second
    pid appears and costs nothing. Once the bug is fixed, only one pid ever appears, so the
    test waits the whole 5 seconds on every run. The fix should then shorten the wait or
    move the test to the nightly tier."""
    db = support.make_store(tmp_path / "empty.db", memory=False)
    runner = hooks("claude", daemon=True, env=support.store_env(db))
    result = runner.run("recall", session="one", prompt=support.UNRELATED)
    assert support.status_of("claude", result.reply) == support.status_line(
        "no matching memories")
    runner.wait_for_daemon()
    # A second daemon, if the hook started one, records its pid as its Python starts. The
    # docstring says what this wait costs once the bug is fixed.
    deadline = time.monotonic() + 5.0
    while len(runner.daemon_pids()) < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    pids = runner.daemon_pids()
    if len(pids) == 2:
        raise known_bugs.Reproduced(f"B62: one recall started two daemons, pids {pids}")
    assert len(pids) == 1, pids
