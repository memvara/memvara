"""One deadline for the whole hook process, which every hosted call respects.

The host stops a hook at its time limit: 10 seconds for recall and 20 for session start on
every host. The hosted client waits up to `hosted.TIMEOUT_SEC` for each request and retries
a request that got no answer once, and one hook makes several calls, so an endpoint that
accepted connections and never answered kept session start for about 60 seconds and recall
for about 36 (#345). The host killed the hook first, so the turn got no memories and not
even the status line that would have said why.

A reading hook sets the deadline once: `MARGIN_SEC` seconds before its host's limit, which
leaves that long for writing its reply. Every hosted call then waits at most the time left, and none is started or retried
once it is spent. A process that sets no deadline, such as the daemon or the capture hook,
waits as it always has.

Only `time` is imported, because recall imports this on every prompt.
"""

from __future__ import annotations

import time

#: What a hook keeps back from its host's limit, for the work after its last hosted call:
#: writing the reply and its log lines. The interpreter's start is inside the host's limit
#: too, before the hook can set anything.
MARGIN_SEC = 1.5

_at: "float | None" = None


def set_from_limit(limit: float) -> None:
    """Set the deadline to `MARGIN_SEC` seconds before a limit of `limit` seconds from
    now. With a 10-second limit, the deadline is 8.5 seconds from now."""
    global _at
    _at = time.monotonic() + max(0.0, limit - MARGIN_SEC)


def left() -> "float | None":
    """Seconds until the deadline, which may be 0 or less, or None when none is set."""
    return None if _at is None else _at - time.monotonic()


def clear() -> None:
    """Remove the deadline."""
    global _at
    _at = None
