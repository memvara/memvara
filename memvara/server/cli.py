"""`python -m memvara.server` — process startup, and the one place that touches stdio.

There are no options. Everything is environment configuration, because that is what an
MCP client can actually set: the settings file gives a command, an argument list and an
env block, and the env block is the only part a user edits per machine. `--help` exists
for the moment someone runs the command by hand to find out why the client says it
failed, and prints the variables rather than a flag list.
"""

from __future__ import annotations

import os
import sys
from typing import Mapping, Sequence, TextIO

from .. import __version__
from .config import EXAMPLE_CONFIG, ConfigError, ServerConfig, build_memvara
from .mcp import MemvaraMCPServer

__all__ = ["main"]

USAGE = f"""\
memvara-mcp {__version__} — Memvara memory as an MCP server over stdio.

This program speaks JSON-RPC on stdin/stdout and is meant to be launched by an MCP
client, not run interactively. Configured entirely by environment:

  MEMVARA_DB          required. Path to the SQLite file; created on first use.
                     ':memory:' for a throwaway store that dies with the process.
  MEMVARA_USER        who this server remembers for. Unset means the whole tenant.
  MEMVARA_TENANT      isolation boundary above the user. Default 'default'.
  MEMVARA_AGENT       narrows further; unset is usually right.
  MEMVARA_SESSION     narrows further still. Memory written here is not visible to
                     other sessions, so leave it unset for durable facts.
  MEMVARA_LLM         'none' (default, offline, extracts only recognised sentence
                     forms) or 'anthropic' (needs ANTHROPIC_API_KEY).
  MEMVARA_READ_ONLY   '1' to hide every tool that writes.

The scope above is bound at startup and cannot be changed by a tool call, which is
what stops a model reaching another user's memory.

Client configuration:

{EXAMPLE_CONFIG}
"""


def main(argv: Sequence[str] | None = None, *, env: Mapping[str, str] | None = None,
         stdin: TextIO | None = None, stdout: TextIO | None = None,
         stderr: TextIO | None = None) -> int:
    """Serve until stdin closes. Returns a process exit status."""
    args = list(sys.argv[1:] if argv is None else argv)
    env = os.environ if env is None else env
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr

    if args:
        if args == ["--version"]:
            print(__version__, file=out)
            return 0
        if args in (["--help"], ["-h"]):
            print(USAGE, file=out)
            return 0
        print(f"memvara-mcp: unexpected argument {args[0]!r}\n\n{USAGE}", file=err)
        return 2

    try:
        config = ServerConfig.from_env(env)
        memory = build_memvara(config)
    except ConfigError as exc:
        # The client shows this to the user as the reason the server would not start,
        # which is the only moment they are looking. Exit 2, as for a usage error: the
        # invocation was wrong, not the program.
        print(f"memvara-mcp: {exc}", file=err)
        return 2

    server = MemvaraMCPServer(memory, read_only=config.read_only, **config.scope_kwargs)
    try:
        server.serve(sys.stdin if stdin is None else stdin, out)
    finally:
        # Closing the store matters even on the way out: the vector index is a file this
        # process may have been extending, and other processes share it.
        server.close()
    return 0
