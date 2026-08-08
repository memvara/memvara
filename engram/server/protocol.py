"""JSON-RPC 2.0 over newline-delimited JSON — the MCP stdio transport, and nothing else.

This is deliberately not the official `mcp` SDK. The SDK is not installed here, and
pulling it in would cost the library its one-hard-dependency property for a transport
that is a hundred lines: one JSON object per line, requests carry an `id` and get
exactly one response, notifications have no `id` and get none. The interesting part of
an MCP server is its tool descriptions, not its framing.

Two rules the rest of the package depends on:

* **stdout is the wire.** Anything printed there that is not a JSON-RPC message
  desynchronises the client, so diagnostics go to stderr and nowhere else.
* **Encoding is pure ASCII.** `json.dumps` escapes non-ASCII by default and that default
  is kept on purpose: it makes the byte stream independent of whatever locale the client
  launched this process with, which on a stdio transport is not knowable from in here.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Iterator, Mapping, TextIO

# The subset of the JSON-RPC 2.0 error space this server can produce, and it is a small
# subset on purpose: every one of these means "the client sent something structurally
# wrong". A tool that ran and failed is never reported here — it comes back as a normal
# result carrying `isError`, because that is the one a model can read. There is
# deliberately no INTERNAL_ERROR (-32603): see `EngramMCPServer._call_tool`.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602


class ProtocolError(Exception):
    """A JSON-RPC-level failure: the message itself was malformed.

    Distinct from a tool that ran and failed, which is a successful response carrying
    `isError`. Conflating the two is how a model loses the ability to see, and correct,
    its own mistakes — the client eats protocol errors.
    """

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def success(request_id: Any, result: Mapping[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": dict(result)}


def failure(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def encode(message: Mapping[str, Any]) -> str:
    """One message as one line. Never contains a newline, which is what frames the wire."""
    return json.dumps(message, ensure_ascii=True, separators=(",", ":"))


def decode(line: str) -> Any:
    try:
        return json.loads(line)
    except ValueError as exc:
        # The id is unknowable — the message did not parse — and JSON-RPC says to answer
        # a parse error with a null id rather than staying silent, so the client learns
        # its request died instead of waiting for it.
        raise ProtocolError(PARSE_ERROR, f"invalid JSON: {exc}") from exc


def iter_messages(stream: TextIO) -> Iterator[str]:
    """Non-empty lines, with framing whitespace removed.

    Blank lines are skipped rather than rejected: a client that writes `\\r\\n` or pads
    between messages is not making a protocol error worth failing a session over.
    """
    for raw in stream:
        line = raw.strip()
        if line:
            yield line


def serve_stdio(handle: Callable[[str], str | None], stdin: TextIO, stdout: TextIO) -> int:
    """Pump lines from `stdin` through `handle` to `stdout` until the client closes.

    Flushing after every message is not optional: the client is blocked waiting for the
    response to the request it just sent, so a buffered reply is a hung session rather
    than a slow one. Returns the number of messages handled, which is what makes the
    loop assertable without a subprocess.
    """
    handled = 0
    for line in iter_messages(stdin):
        handled += 1
        response = handle(line)
        if response is None:
            continue                    # a notification: JSON-RPC forbids answering it
        stdout.write(response + "\n")
        stdout.flush()
    return handled
