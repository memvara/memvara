#!/usr/bin/env python3
"""PreToolUse — let read-only memory_* tools run without a permission prompt.

SuperMemory auto-allows search; writes still ask. Same split here. A silent
no-op on any other tool, so this matcher can be wide (`mcp__.*memvara.*`)
without approving a forget, or a tool of another server whose name contains
`memvara`: a tool is approved only when its whole name is one of the host's
`ApproveSpec.prefixes` followed by a read-only tool's name.

Each read it approves is also counted as one `searched` for the status line, in
`~/.memvara/.hooks/counts/<session>.json`, unless the `status_line` setting is off.
"""

from __future__ import annotations

import os.path
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.envelope import read_event, write  # noqa: E402
from core.host import Reply, active  # noqa: E402
from lib import counts  # noqa: E402
from lib.ipc import payload  # noqa: E402

#: Every memory_* tool the server marks `readOnlyHint`. A read that prompts is a read the
#: model learns to avoid, and the two graph tools were missing for no reason other than
#: that they were added after this list. `memory_standing` and `memory_ask` were missing for
#: the same reason, and the memory-research subagent calls both, so it stopped at a
#: permission prompt on its first search. `memory_profile` is listed before the server
#: ships it so that the subagent can call it the day it does. The two document readers,
#: `memory_get_document` and `memory_list_documents`, were missing too (#267).
READ_ONLY = frozenset({
    "memory_recall",
    "memory_search",
    "memory_since",
    "memory_history",
    "memory_why",
    "memory_stats",
    "memory_neighborhood",
    "memory_paths",
    "memory_standing",
    "memory_ask",
    "memory_profile",
    "memory_get_document",
    "memory_list_documents",
})


def _ours(name: str, prefixes: "tuple[str, ...]") -> bool:
    """Whether `name` is a read-only tool of memvara's own server on this host.

    `mcp__memvara__memory_search` is; `mcp__not-memvara__memory_search` is not, and
    neither is `mcp__memvara__memory_forget`.
    """
    return any(name.startswith(p) and name[len(p):] in READ_ONLY for p in prefixes)


def main() -> int:
    host = active()
    if host.approve is None:
        # No pre-tool event on this client: there is no prompt to pre-empt.
        return 0
    event = read_event(host, "approve", payload())
    if not _ours(event.tool_name, host.approve.prefixes):
        return 0
    write(host, Reply("approve", decision=host.approve.allow,
                      reason="Memvara recall is read-only."))
    if counts.enabled():
        counts.bump(event.session, "searched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
