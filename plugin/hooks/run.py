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
