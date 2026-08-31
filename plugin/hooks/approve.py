#!/usr/bin/env python3
"""PreToolUse — let read-only memory_* tools run without a permission prompt.

SuperMemory auto-allows search; writes still ask. Same split here. A silent
no-op on any other tool, so this matcher can be wide (`mcp__.*memvara.*`)
without approving a forget.
"""

from __future__ import annotations

import os.path
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.envelope import read_event, write  # noqa: E402
from core.host import Reply, active  # noqa: E402
from lib.ipc import payload  # noqa: E402

#: Every memory_* tool the server marks `readOnlyHint`. A read that prompts is a read the
#: model learns to avoid, and the two graph tools were missing for no reason other than
#: that they were added after this list.
READ_ONLY = frozenset({
    "memory_recall",
    "memory_search",
    "memory_since",
    "memory_history",
    "memory_why",
    "memory_stats",
    "memory_neighborhood",
    "memory_paths",
})


def _tool_leaf(name: str, separators: "tuple[str, ...]") -> str:
    # mcp__memvara__memory_search or mcp__plugin_memvara_memvara__memory_search
    for sep in separators:
        if sep in name:
            return name.rsplit(sep, 1)[-1]
    return name


def main() -> int:
    host = active()
    if host.approve is None:
        # No pre-tool event on this client: there is no prompt to pre-empt.
        return 0
    leaf = _tool_leaf(read_event(host, "approve", payload()).tool_name,
                      host.approve.separators)
    if leaf not in READ_ONLY:
        return 0
    write(host, Reply("approve", decision=host.approve.allow,
                      reason="Memvara recall is read-only."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
