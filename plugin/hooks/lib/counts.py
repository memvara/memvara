"""Per-session counts of memory activity, for the status line.

The status line shows `⋈ memvara · 12 recalled · 3 searched · 5 captured` for the session
in front of the user. Three hooks keep the numbers, one file per session, in
`~/.memvara/.hooks/counts/<session>.json`:

- `recall.py` adds the number of memory lines it injected into a prompt (`recalled`);
- `approve.py` adds one for each read-only memory tool the model calls (`searched`);
- `capture.py` adds the number of facts a turn stored, after the write succeeds
  (`captured`).

The file holds `{"recalled": int, "searched": int, "captured": int, "updated_at": str}`,
where `updated_at` is an ISO-8601 UTC time. `session_start.py` removes files untouched for
14 days, once per session, as it opens.

**`read()` imports nothing from the rest of the hooks.** The status-line script in the
plugin repository vendors this one file and calls `read()`, and it must finish in under
50ms. The writing side (`bump`, `prune`, `enabled`) imports `lib.state_file` and
`lib.settings` when it is first called, so a script that only reads never needs them.
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

#: The switch in `~/.memvara/settings.json` that turns counting off.
FEATURE = "status_line"

#: A session nobody has touched in a fortnight will not be resumed. The same lifetime as
#: the recall hook's `SEEN_TTL_SECONDS`.
TTL_SECONDS = 14 * 24 * 3600


def _path(session_id: str) -> "str | None":
    """The session's file, or `None` for an id that could name a file outside the directory.

    A NUL byte is refused as well: every `os` call raises `ValueError` on one, which is not
    the `OSError` a file operation is normally guarded against.
    """
    if (not session_id or "/" in session_id or "\\" in session_id or "\0" in session_id
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


def enabled() -> bool:
    """Whether the hooks should count at all: the `status_line` setting."""
    from .settings import enabled as setting

    return setting(FEATURE)


def bump(session_id: str, field: str, n: int = 1, now: "float | None" = None) -> None:
    """Add `n` to one counter for this session. Silent on every failure.

    The read and the write happen under one lock, because two hooks for one session can run
    at the same moment, for example two tool calls approved in parallel, and without it
    the second write would replace the first and lose a count. The write is atomic, so the
    status line never reads half a file. See `lib.state_file`.
    """
    path = _path(session_id)
    if path is None or field not in FIELDS or n <= 0:
        return
    from .state_file import update_json

    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                          time.gmtime(time.time() if now is None else now))

    def change(_: object) -> dict:
        counts = read(session_id)
        counts[field] += n
        counts["updated_at"] = stamp
        return counts

    update_json(path, change, lock_path=os.path.join(COUNTS_DIR, ".lock"),
                prefix=".counts-")


def prune(now: "float | None" = None) -> None:
    """Remove the files of sessions untouched for `TTL_SECONDS`. Called once per session."""
    from .state_file import prune as prune_dir

    prune_dir(COUNTS_DIR, TTL_SECONDS, time.time() if now is None else now)
