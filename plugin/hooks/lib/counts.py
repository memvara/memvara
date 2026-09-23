"""Per-session counts of memory activity, for the status line.

The status line shows `⋈ memvara · 12 recalled · 3 searched · 5 captured` for the session
in front of the user. Three hooks keep the numbers, one file per session, in
`~/.memvara/.hooks/counts/<session>.json`:

- `recall.py` adds the number of memory lines it injected into a prompt (`recalled`);
- `approve.py` adds one for each read-only memory tool the model calls (`searched`);
- `capture.py` adds the number of facts a turn stored, after the write succeeds
  (`captured`).

The file holds `{"recalled": int, "searched": int, "captured": int, "updated_at": str}`,
where `updated_at` is an ISO-8601 UTC time. Files untouched for 14 days are removed, as the
recall hook's per-session state is.

**This module imports nothing from the rest of the hooks.** The status-line script in the
plugin repository vendors this one file and calls `read()`, and it must finish in under
50ms, so this file cannot pull in the rest of the tree. `read()` has no side effects. The
callers decide whether counting is switched on, through the `status_line` setting.
"""

from __future__ import annotations

import json
import os
import os.path
import time

#: One file per session. Beside the other hook state, not in the plugin, which is replaced
#: on update.
COUNTS_DIR = os.path.join(os.path.expanduser("~"), ".memvara", ".hooks", "counts")

#: The counters, in the order the status line prints them.
FIELDS = ("recalled", "searched", "captured")

#: A session nobody has touched in a fortnight will not be resumed. The same lifetime as
#: the recall hook's `SEEN_TTL_SECONDS`.
TTL_SECONDS = 14 * 24 * 3600


def _path(session_id: str) -> "str | None":
    """The session's file, or `None` for an id that could name a file outside the directory."""
    if (not session_id or "/" in session_id or "\\" in session_id
            or session_id in (".", "..")):
        return None
    return os.path.join(COUNTS_DIR, f"{session_id}.json")


def read(session_id: str) -> dict:
    """The session's counts, with zeros for anything missing or unreadable.

    Never raises and never writes. A session with no file yet reads as all zeros and
    `updated_at` of `None`.
    """
    out: dict = {field: 0 for field in FIELDS}
    out["updated_at"] = None
    path = _path(session_id)
    if path is None:
        return out
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return out
    if not isinstance(data, dict):
        return out
    for field in FIELDS:
        value = data.get(field)
        # `bool` is an `int` in Python, and `true` is not a count.
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            out[field] = value
    stamp = data.get("updated_at")
    out["updated_at"] = stamp if isinstance(stamp, str) else None
    return out


def _prune(now: float) -> None:
    try:
        names = os.listdir(COUNTS_DIR)
    except OSError:
        return
    for name in names:
        if not name.endswith(".json"):
            continue
        path = os.path.join(COUNTS_DIR, name)
        try:
            if now - os.path.getmtime(path) > TTL_SECONDS:
                os.unlink(path)
        except OSError:
            continue


def bump(session_id: str, field: str, n: int = 1, now: "float | None" = None) -> None:
    """Add `n` to one counter for this session. Silent on every failure.

    The write goes through a temporary file and a rename, so the status line never reads a
    half-written file. Two hooks for one session can run at the same moment, for example
    two tool calls approved in parallel, so the read and the write are held under an
    exclusive lock where the platform has one. Without it the second write would replace
    the first and one count would be lost.
    """
    path = _path(session_id)
    if path is None or field not in FIELDS or n <= 0:
        return
    import tempfile

    now = time.time() if now is None else now
    try:
        os.makedirs(COUNTS_DIR, exist_ok=True)
        with open(os.path.join(COUNTS_DIR, ".lock"), "a", encoding="utf-8") as lock:
            try:
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            except (ImportError, OSError):
                # No `fcntl` on Windows. A lost count there is the cost, not a failed hook.
                pass
            counts = read(session_id)
            counts[field] += n
            counts["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
            fd, tmp = tempfile.mkstemp(dir=COUNTS_DIR, prefix=".counts-")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(counts, fh)
                os.replace(tmp, path)
            except OSError:
                os.unlink(tmp)
        _prune(now)
    except OSError:
        pass
