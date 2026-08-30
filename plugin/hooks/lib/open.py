"""Resolve the store a hook should read, or decide there isn't one.

Hooks run in the *client's* environment, not the MCP server's. Claude Code launches
`python3 -m memvara.server` with an env block — `MEMVARA_DB`, `PYTHONPATH` and the rest —
and none of that reaches a hook process. So the hook has to rediscover the same
configuration, and the only way to guarantee it lands on the same store, with the same
embedder, is to read the client's own server block and reuse the library's
`ServerConfig.from_env()` rather than re-deriving any of it here.

Everything in this module is written so a failure is silent. A hook that raises on a
missing store, an unreadable settings file or a half-installed library turns every prompt
into an error banner; the memory is an enhancement, and its absence must look like
nothing at all.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from .ipc import emit, server_env  # noqa: F401  (re-exported; they live
# in ipc so the fast path can use them without importing pathlib)

#: Written by `memvara-mcp login`, read when there is no local store to open.
_CREDENTIALS = Path.home() / ".memvara" / "credentials.json"


def _import_memvara(env: Mapping[str, str]) -> Any:
    """Import the library, honouring a PYTHONPATH that only the server block knows.

    A source checkout is a normal way to run this — the server block carries the path —
    and without this the hook would import nothing while the server imports fine, which
    presents as memory that works in tool calls and is silently absent from prompts.
    """
    extra = env.get("PYTHONPATH", "")
    for part in reversed([p for p in extra.split(os.pathsep) if p]):
        if part not in sys.path:
            sys.path.insert(0, part)
    import memvara  # noqa: F401  (imported for its side effect of being importable)

    return memvara


def open_store() -> Any | None:
    """The `Memvara` a hook should read, or `None` to do nothing at all.

    `None` is a normal outcome, not an error: no store is configured, the library is not
    installed, the embedder does not match, the credentials expired, or the deployment is
    hosted and belongs to `lib.hosted` instead. Every one of those means this prompt gets
    no memory block *from here*, and the last one means it gets a better one elsewhere.

    There is no caller-kind parameter here any more, and the absence is the point: on a
    hosted deployment `lib.hosted` now serves reads *and* writes, so there is nothing left
    for a second client to be better at. It briefly took a `recalls` flag, when the MCP
    surface could not carry `sources=` and the library's client could.
    """
    env = dict(os.environ)
    # The client's block loses to a real environment variable. Someone who exports
    # MEMVARA_DB to point a session at a scratch store means it.
    for key, value in server_env().items():
        env.setdefault(key, value)

    if not env.get("MEMVARA_DB") and env.get("MEMVARA_MODE") != "cloud":
        # No local store named. Cloud mode is still possible if a key was written by
        # `memvara-mcp login`, and is the only case where we invent configuration.
        if _CREDENTIALS.is_file():
            env["MEMVARA_MODE"] = "cloud"
        else:
            return None

    if env.get("MEMVARA_MODE") == "cloud":
        # The same refusal as below, reached before paying for it. `import memvara` is
        # ~95ms and this function runs whenever the daemon is not warm -- the first prompt
        # of every session, and every prompt after a daemon dies -- so importing the whole
        # library to then discard it is the one cost here worth avoiding.
        #
        # Deliberately the raw string, matching the check above that may have just set it,
        # rather than a second copy of the library's normalisation. An unnormalised value
        # like `Cloud` misses this and is caught by the normalised check below, so this can
        # only ever be an early exit for a case that was already decided, never a decision
        # of its own.
        return None

    try:
        _import_memvara(env)
        from memvara.server.config import ServerConfig, build_memvara

        config = ServerConfig.from_env(env)
        if config.mode != "local":
            # Not a local engine, so not ours to hand any hook. `build_memvara` has
            # returned a `RemoteMemvara` for a cloud config since memvara/memvara@2a3bb48,
            # and no hook here can use one: its `recall()` takes no `header=` at all and
            # *refuses* a `budget=` rather than dropping it -- deliberately, because it
            # cannot re-derive the local truncation from a server-rendered string.
            #
            # Writers used to want the opposite handle, for the `sources=` that carries a
            # claim back to the turn it came from. memvara/memvara#76 shipped that across
            # the MCP transport, so `lib.hosted` carries it too and the exception is gone.
            # Measured against the live endpoint rather than assumed: `memory_remember`
            # reports `accepts("sources") is True`, the `memory_add` receipt renders
            # `turn id(s): ep_...`, and `memory_why` on a claim written this way resolves
            # to the turn.
            #
            # This is not a new rule; it is the one every docstring here already states
            # ("open_store() answers None on a hosted install"). It stopped being true by
            # accident, upstream, and cost every prompt its memory block for a day while
            # the fallback chain sat intact and unreached.
            #
            # Spelled `!= "local"` rather than `== "cloud"`: a mode added later is far
            # likelier to be another remote than another engine, and this direction
            # degrades to the route that works instead of to a silent outage.
            return None
        return build_memvara(config)
    except Exception:
        # Deliberately bare. ConfigError, ImportError, EmbedderMismatchError, a corrupt
        # SQLite file and a revoked API key are all the same event from here: no memory
        # this turn.
        return None
