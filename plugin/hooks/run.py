#!/usr/bin/env python3
"""`run.py <hook> --host <id>` — the one entry point every client's config names.

The four bodies beside this file are host-neutral now: they read an `Event` and answer
with a `Reply`, and the client's spelling of both lives in a `Host` record under
`hosts/`. This resolves that record, binds it, and hands off.

Nothing here may raise. A hook that fails a prompt is worse than a hook that does
nothing, so every path out of `main` returns 0 -- and every path that decides to do
nothing says so in `~/.memvara/.hooks/hooks.log`, because "skipped" and "never ran" are
the pair that must not look alike.
"""

from __future__ import annotations

import importlib
import os.path
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import host as _host  # noqa: E402


def _note(text: str) -> None:
    """Write one line to the hook log, and never raise doing it.

    Called on three paths *outside* the try below, and once inside the handler that
    exists so a body's failure cannot reach the client. It imports `lib.ipc` at call
    time, so a tree that is missing or half-copied -- an interrupted vendor, a sync that
    stopped between files -- made the logging turn a handled failure into an unhandled
    one. Measured: with `lib/ipc.py` moved aside, `run.py recall --host claude` exited 1
    with a traceback out of the handler itself, and a non-zero UserPromptSubmit hook
    blocks the turn.

    Losing the line is the right trade when the alternative is losing the prompt. A tree
    too broken to write to `~/.memvara/.hooks/hooks.log` is a tree too broken to have run.
    """
    try:
        from lib.ipc import log_line

        log_line("hooks", text)
    except Exception:  # noqa: BLE001 -- see above; a hook must never fail a prompt
        pass


#: Set on the detached child so it does not fork again. An environment variable rather
#: than an argv flag because the child is invoked with the same argv on purpose -- one
#: spelling of the command, and a `ps` line that reads the same as the parent's.
_DETACHED = "MEMVARA_HOOK_DETACHED"


def _detach(hook: str, host_id: str) -> int:
    """Re-run this same command in a new session and return immediately.

    For a host that offers no working async: Codex's registration accepts `async: true`
    and its hook then does not run at all -- measured on codex-cli 0.151.0, an async Stop
    wrote no receipt though writing one is the script's first statement. Declared
    synchronous it fires, and a child started with `start_new_session=True` outlives the
    `codex exec` process and finishes twelve seconds after the turn ended.

    So the fork happens here rather than in the body: `capture` stays one straight-line
    program that mines a turn, and the question of who waits for it stays a property of
    the host. stdin is read here and handed on, because it is the payload and a child that
    inherited the parent's stdin would find it already consumed -- or worse, still open,
    holding the turn on a pipe the client is waiting to close.

    Every failure returns 0. A capture that could not be started is a lost turn; a hook
    that raises is a broken one.
    """
    import subprocess  # noqa: PLC0415 -- off the per-prompt path, paid only on capture
    import tempfile  # noqa: PLC0415

    try:
        payload = sys.stdin.read()
    except (OSError, ValueError):
        payload = ""

    # A FILE rather than a pipe, and that is the whole point of these six lines. Writing
    # the payload into `stdin=PIPE` blocks once the 64KB buffer fills, and the child does
    # not drain it for ~95ms while it imports -- so the parent, which is a SYNCHRONOUS
    # Stop hook whose only job is to return immediately, would sit holding the turn open
    # on exactly the large payloads worth capturing. `Stop` carries the assistant's whole
    # last message, so payload size is the host's to choose, not ours.
    #
    # Unlinked as soon as it is open: on POSIX the child keeps a valid descriptor to an
    # inode with no name, so nothing is left behind however the child ends -- no temp file
    # to clean up on a path where the parent is already gone.
    try:
        handle = tempfile.TemporaryFile()
        handle.write(payload.encode("utf-8"))
        handle.seek(0)
    except (OSError, ValueError) as exc:
        _note(f"failed hook={hook} host={host_id} detach-payload: {type(exc).__name__}")
        return 0

    try:
        child = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), hook, "--host", host_id],
            stdin=handle,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env={**os.environ, _DETACHED: "1"},
        )
    except (OSError, ValueError) as exc:
        _note(f"failed hook={hook} host={host_id} detach: {type(exc).__name__}: {exc}")
        return 0
    finally:
        # Ours to close either way: the child has its own descriptor once spawned, and on
        # the failure path nobody else will.
        try:
            handle.close()
        except OSError:
            pass

    _note(f"detached hook={hook} host={host_id} pid={child.pid} bytes={len(payload)}")
    return 0


def main(argv: "list[str]") -> int:
    hook = argv[0] if argv and not argv[0].startswith("-") else ""
    host_id = argv[argv.index("--host") + 1] if "--host" in argv[:-1] else ""
    if hook not in _host.HOOKS or not host_id:
        _note(f"skipped=bad invocation argv={argv}")
        return 0
    try:
        record = importlib.import_module(f"hosts.{host_id}").HOST
    except (ImportError, AttributeError, ValueError):
        _note(f"skipped=unknown host {host_id!r} hook={hook}")
        return 0
    if hook not in record.events:
        _note(f"skipped={host_id} has no event for {hook}")
        return 0
    if hook == "capture" and record.detach_capture and not os.environ.get(_DETACHED):
        return _detach(hook, host_id)

    # Bound before the body is imported, not after: `lib.transcript` resolves this host's
    # noise markers at import time.
    _host.use(record)
    try:
        return importlib.import_module(hook).main()
    except Exception as exc:  # noqa: BLE001 -- a hook must never fail a prompt
        _note(f"failed hook={hook} host={host_id} {type(exc).__name__}: {exc}"[:400])
        return 0


if __name__ == "__main__":
    # The last guard, and deliberately not `raise SystemExit(main(...))` alone. `main`
    # returns 0 on every path it knows about; this catches the ones it does not -- an
    # import that fails before the try, an interpreter that cannot resolve `core.host`,
    # anything a future edit adds above the handler. Zero is the only exit code that
    # leaves the turn alone.
    try:
        _status = main(sys.argv[1:])
    except BaseException:  # noqa: BLE001 -- nothing may fail the prompt
        _status = 0
    raise SystemExit(_status)
