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
import math
from typing import Any, Callable, Iterator, Mapping, TextIO

# The subset of the JSON-RPC 2.0 error space this server can produce, and it is a small
# subset on purpose. The first four mean "the client sent something structurally wrong".
# A tool that ran and failed is never reported here — it comes back as a normal result
# carrying `isError`, because that is the one a model can read (see
# `MemvaraMCPServer._call_tool`). INTERNAL_ERROR is not a tool's failure either: it answers
# a request whose reply the server could not write as JSON, which `encode` refuses rather
# than writing a line a strict parser cannot read (`MemvaraMCPServer.handle_line`).
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


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
    """One message as one line. Never contains a newline, which is what frames the wire.

    `allow_nan=False`, so a reply can never carry a number that is not JSON: the standard
    library would otherwise write a non-finite float as a bare `NaN` or `Infinity`, which
    a strict parser, such as JavaScript's `JSON.parse`, cannot read. The id is the only
    value a reply copies from its request, `decode` refuses an id that holds such a
    number, and every result is text, so none should reach this. If one does, this raises
    `ValueError` instead of writing the line, and `MemvaraMCPServer.handle_line` answers
    the request with an internal error (-32603) and keeps serving.
    """
    return json.dumps(message, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def _holds_a_non_finite_number(value: Any) -> bool:
    """Whether `value`, or anything nested inside it, is a float that is not finite.

    It walks with a list of pending items rather than by recursion, because `json.loads`
    accepts nesting almost as deep as the interpreter's stack allows, and a recursive
    walk started from inside the server would run out of stack on such a value.
    """
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, float):
            if not math.isfinite(item):
                return True
        elif isinstance(item, Mapping):
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return False


def decode(line: str) -> Any:
    """One line as one message, or `ProtocolError(PARSE_ERROR)`.

    Python's decoder accepts the tokens `NaN`, `Infinity` and `-Infinity`, which are not
    JSON, and it reads a number too large for a double, such as `1e400`, as infinity. A
    reply copies its request's id, of any JSON type, so an id like that came back as a
    bare `NaN` or `Infinity`, and the reply line was then not JSON. A message whose id
    holds such a number, at any depth of an object or array id, is therefore a parse
    error here. The same numbers anywhere else in a request are read as before: no reply
    copies them, and a tool's own argument check decides what they mean.
    """
    try:
        message = json.loads(line)
    except RecursionError:
        # Nesting deeper than the interpreter's stack allows. The decoder raises this
        # rather than a ValueError, and uncaught it ended the stdio loop, so one broken or
        # hostile line cost the agent its memory for the rest of the session (#268).
        raise ProtocolError(PARSE_ERROR, "invalid JSON: nested too deeply to parse") from None
    except ValueError as exc:
        # The id is unknowable — the message did not parse — and JSON-RPC says to answer
        # a parse error with a null id rather than staying silent, so the client learns
        # its request died instead of waiting for it.
        raise ProtocolError(PARSE_ERROR, f"invalid JSON: {exc}") from exc
    if isinstance(message, Mapping) and _holds_a_non_finite_number(message.get("id")):
        # Answered like a line that does not parse, and for the same reason: the id
        # cannot be copied into the reply, so the reply carries a null one.
        raise ProtocolError(PARSE_ERROR, "invalid JSON: the id holds a number that JSON "
                            "cannot carry, such as NaN, Infinity or one too large for a "
                            "double")
    return message


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
