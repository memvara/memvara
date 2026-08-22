"""Memvara as an MCP server: memory that any MCP client can use, over stdio.

    $ MEMVARA_DB=~/.memvara/memory.db MEMVARA_USER=alice python3 -m memvara.server

Why this exists before the HTTP API, which is the thing that was originally planned:

* **`recall()` was already an MCP tool result.** It returns numbered plain facts under a
  header that frames them as data — no scores, no JSON, because retrieval metadata in a
  prompt is noise the model has to ignore. That is the MCP tool-result contract exactly.
  The library was built for this shape before anyone was aiming at it.
* **stdio needs no authentication infrastructure.** The process *is* the user: the client
  launched it, with their environment, and it exits with them. So this ships before the
  token layer. REST cannot: scope there is a caller-supplied string with no enforcement
  behind it, which is not an API, it is a suggestion.
* **No per-host integration.** Claude Code, Cursor and the rest speak MCP natively.

Two design choices worth knowing before reading further:

**No third-party dependency.** The official SDK is not installed here, and the transport
it would provide is a hundred lines — one JSON object per line — against a dependency
tree of a dozen packages. The core's single hard dependency is a selling point; spending
it on framing would be a poor trade. See `protocol.py`.

**Twelve tools, and no way to erase anything.** `consolidate` is an operator action that an
agent, given it, will call in a loop. `purge` and `reset` are irreversible erasure, which
must never be one tool call away from a model that read "forget that" as "delete
everything". The two closures are offered instead, as two tools: `memory_forget` retires
a record that was wrong, `memory_end` closes out a fact that stopped being true, and both
stay visible to `memory_history`. They are separate tools rather than one with a
`closure` flag because a model commits to a tool by its name before it reads a parameter,
and "forget" already asserts one of the two answers — see `tools.py`.
"""

from .cli import main
from .config import ConfigError, ServerConfig, build_memvara
from .mcp import INSTRUCTIONS, PROTOCOL_VERSION, SUPPORTED_PROTOCOLS, MemvaraMCPServer
from .protocol import ProtocolError, serve_stdio
from .tools import TOOLS, Tool, ToolContext, ToolError

__all__ = [
    "MemvaraMCPServer",
    "ServerConfig", "ConfigError", "build_memvara", "main",
    "TOOLS", "Tool", "ToolContext", "ToolError",
    "ProtocolError", "serve_stdio",
    "PROTOCOL_VERSION", "SUPPORTED_PROTOCOLS", "INSTRUCTIONS",
]
