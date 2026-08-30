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
    from lib.ipc import log_line

    log_line("hooks", text)


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
    raise SystemExit(main(sys.argv[1:]))
