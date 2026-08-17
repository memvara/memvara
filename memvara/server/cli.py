"""`python -m memvara.server` — process startup, and the one place that touches stdio.

The server takes no options. Everything is environment configuration, because that is
what an MCP client can actually set: the settings file gives a command, an argument list
and an env block, and the env block is the only part a user edits per machine. `--help`
exists for the moment someone runs the command by hand to find out why the client says
it failed, and prints the variables rather than a flag list.

`init` is the one subcommand, and it does not weaken that. It is not a way to configure
this process — it writes the settings file the *client* will launch this process from,
which is a different program's configuration and the only place flags could have helped
anyone. See `init.py`.
"""

from __future__ import annotations

import os
import sys
from typing import Mapping, Sequence, TextIO

from .. import __version__
from ..core import EmbedderMismatchError
from .config import EXAMPLE_CONFIG, ConfigError, ServerConfig, build_memvara
from .init import init
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
  MEMVARA_EMBEDDER    'hashing' (default, offline, 512-dimensional), 'hashing:<dim>',
                     'local' or 'local:<model>' (needs memvara[local-embed]), or
                     'auto' for whichever of those happens to be installed. A store
                     can only be opened by the embedder that wrote it; if this server
                     refuses to start with a dimension mismatch, this is the variable
                     that fixes it, and the message names the width to give it.
  MEMVARA_READ_ONLY   '1' to hide every tool that writes.

The scope above is bound at startup and cannot be changed by a tool call, which is
what stops a model reaching another user's memory.

Rather than writing that block by hand:

  memvara-mcp init --agent claude
                     Writes the client's server block, the memvara skill and a
                     CLAUDE.md snippet into a project, with MEMVARA_DB already
                     absolute. `memvara-mcp init --help` for its options.

  memvara-mcp login --project NAME
                     Signs in to a memvara-cloud deployment over the device-code
                     flow and writes an API key to ~/.memvara/credentials.json,
                     for MEMVARA_MODE=cloud. Needs the `cloud` extra. `memvara-mcp
                     login --help` for its options.

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
        # Dispatched before the flags, and by first word rather than by whole argv,
        # because `init` has a command line of its own and this one has none.
        if args[0] == "init":
            return init(args[1:], env=env, stdout=out, stderr=err)
        if args[0] == "login":
            # Imported here, not at module top: `login.py` belongs to the `cloud` extra
            # and requires `httpx`, and this file has to stay importable with no extras
            # installed (CONTRIBUTING.md's "no extras" CI job) for everyone who never
            # calls `login`.
            from .login import login

            return login(args[1:], env=env, stdout=out, stderr=err)
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
    except EmbedderMismatchError as exc:
        # Not a ConfigError — the library raised it about the store — but from here it is
        # one: this environment names an embedder that cannot read this store, and the
        # remedy is a variable, not a code change. Reported the same way for the same
        # reason, because the alternative under stdio is a traceback in a log the user
        # has to go looking for. The message already carries the store's width and the
        # name of whatever wrote it; all this adds is where to type them.
        print(f"memvara-mcp: {exc}\nFrom this server, set MEMVARA_EMBEDDER to match the "
              "store — 'hashing:<dim>' or 'local:<model>', spelled as above — rather "
              "than editing code. See docs/DEPLOY.md.", file=err)
        return 2

    server = MemvaraMCPServer(memory, read_only=config.read_only, **config.scope_kwargs)
    try:
        server.serve(sys.stdin if stdin is None else stdin, out)
    finally:
        # Closing the store matters even on the way out: the vector index is a file this
        # process may have been extending, and other processes share it.
        server.close()
    return 0
