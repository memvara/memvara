"""The stdio server's framing and request ids, over the real pipe.

Each test writes raw lines to a real `python -m memvara.server` and counts what comes
back with `exchange`, which ends every send with a ping. So every test also checks that
the server still answers after the input it was given.
"""

from __future__ import annotations

import collections
import json
import random
import sqlite3
import sys
from typing import Any, Callable

import pytest

from harness.stdio import McpProcess

from . import call_line, changed, exchange, notification_line, request_line, rows, text_of
from . import shared_server  # noqa: F401 - a fixture; importing it lets pytest find it here

Start = Callable[..., McpProcess]

#: Deeper than any Python version's JSON decoder allows.
DEEP = 100_000


def _same(first: Any, second: Any) -> bool:
    """Equal as JSON, so that true is not 1 and 1.0 is not 1."""
    return json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def _nested(depth: int) -> str:
    return "[" * depth + "]" * depth


def _stored_objects(server: McpProcess, predicate: str) -> list[str]:
    connection = sqlite3.connect(f"{server.db.resolve().as_uri()}?mode=ro", uri=True)
    try:
        return sorted(value for (value,) in connection.execute(
            "SELECT object FROM claims WHERE predicate = ?", (predicate,)))
    finally:
        connection.close()


def _one_error(replies: list[dict[str, Any]], code: int) -> None:
    """Exactly one reply, and it is a JSON-RPC error with `code` and a null id: the id
    of a line that could not be read as one request is unknowable."""
    assert len(replies) == 1, replies
    assert replies[0]["id"] is None, replies[0]
    assert replies[0]["error"]["code"] == code, replies[0]


# -- many requests at once ------------------------------------------------------------

def test_500_pipelined_requests_with_shuffled_ids_each_get_exactly_one_reply(
        mcp: Start) -> None:
    """A client may write many requests before it reads any reply. Each request with an
    id gets exactly one reply with that id, whatever order the ids come in and whatever
    the request asks for. A notification between them gets no reply, and every write in
    the stream is stored exactly once."""
    server = mcp()
    server.initialize()
    rng = random.Random(20260926)
    ids: list[Any] = [*rng.sample(range(1, 2**53), 350), *(f"r{n}" for n in range(150))]
    rng.shuffle(ids)
    lines: list[str] = []
    kinds: dict[Any, int] = {}
    written: list[str] = []
    for n, request_id in enumerate(ids):
        kind = n % 5
        kinds[request_id] = kind
        if kind == 0:
            lines.append(request_line(request_id, "ping"))
        elif kind == 1:
            written.append(f"pipelined value {n}")
            lines.append(call_line(request_id, "memory_remember",
                                   {"predicate": "likes", "object": written[-1]}))
        elif kind == 2:
            lines.append(call_line(request_id, "memory_search",
                                   {"query": f"value {n}", "k": 3}))
        elif kind == 3:
            lines.append(call_line(request_id, "memory_recall", {"query": "tea", "k": "8"}))
        else:
            lines.append(request_line(request_id, "no/such/method"))
        if n % 7 == 0:
            lines.append(notification_line("notifications/initialized"))
    replies = exchange(server, "\n".join(lines))
    assert collections.Counter(reply.get("id") for reply in replies) == \
        collections.Counter(ids)
    for reply in replies:
        kind = kinds[reply["id"]]
        if kind == 4:
            assert reply["error"]["code"] == -32601, reply
        else:
            assert reply["result"].get("isError", False) is (kind == 3), reply
    assert _stored_objects(server, "likes") == sorted(written)


# -- ids ------------------------------------------------------------------------------

#: An id of every JSON type. JSON-RPC allows a string or a number, and MCP narrows that
#: to a string or an integer, but the server does not police the type: it answers with
#: whatever id the request carried, which lets any client match its reply.
ODD_IDS: dict[str, Any] = {
    "a float": 1.5,
    "true": True,
    "false": False,
    "zero": 0,
    "a negative integer": -7,
    "an object": {"a": [1, None]},
    "an array": [1, "two"],
    "an integer past 64 bits": 10**30,
    "an integer of 4000 digits": int("9" * 4000),
    "an empty string": "",
    "a 10000-character string": "x" * 10_000,
    "a string outside ASCII": "é\U0001f600",
}


@pytest.mark.parametrize("request_id", list(ODD_IDS.values()), ids=list(ODD_IDS))
def test_an_id_of_any_json_type_comes_back_unchanged_on_exactly_one_reply(
        shared_server: McpProcess, request_id: Any) -> None:
    replies = exchange(shared_server, request_line(request_id, "ping"))
    assert len(replies) == 1, replies
    assert _same(replies[0].get("id"), request_id)
    assert replies[0].get("result") == {}


def test_a_null_id_is_treated_as_a_notification_and_gets_no_reply(
        shared_server: McpProcess) -> None:
    """`MemvaraMCPServer.handle_message` treats a missing id as a notification, and an
    explicit null one too, because a reply addressed to null could not be matched to
    anything. So a ping with a null id gets no reply."""
    assert exchange(shared_server, request_line(None, "ping")) == []


@pytest.mark.parametrize("method", ["notifications/initialized", "notifications/cancelled",
                                    "ping", "no/such/method"])
def test_a_notification_gets_no_reply_whatever_its_method(
        shared_server: McpProcess, method: str) -> None:
    """JSON-RPC forbids answering a notification, and the server ignores one it does not
    know rather than failing on it (`MemvaraMCPServer._dispatch`)."""
    assert exchange(shared_server, notification_line(method)) == []


# -- lines that are not one request ----------------------------------------------------

BATCHES: dict[str, list[Any]] = {
    "requests": [json.loads(request_line(1, "ping")),
                 json.loads(call_line(2, "memory_remember",
                                      {"predicate": "likes", "object": "batched"})),
                 json.loads(call_line(3, "memory_forget", {"predicate": "lives_in"}))],
    "empty": [],
    "notifications": [json.loads(notification_line("notifications/initialized"))],
    "nested": [[json.loads(request_line(4, "ping"))]],
}


@pytest.mark.parametrize("batch", list(BATCHES.values()), ids=list(BATCHES))
def test_a_batch_is_refused_whole_and_nothing_in_it_runs(
        shared_server: McpProcess, batch: list[Any]) -> None:
    """MCP removed JSON-RPC batches in its 2025-06-18 revision, and the server refuses a
    batch outright rather than half-implementing it (`MemvaraMCPServer.handle_message`;
    the design's list of documented behaviour says "Batches are refused"). So the line
    gets one Invalid Request error with a null id, no request inside it is answered, and
    none of them runs: the write and the retirement in the first batch change nothing."""
    before = rows(shared_server.db)
    _one_error(exchange(shared_server, json.dumps(batch)), -32600)
    assert changed(before, rows(shared_server.db)) == []


@pytest.mark.parametrize("line", ["1", "3.5", '"ping"', "true", "null"])
def test_a_line_that_is_json_but_not_an_object_is_refused_as_an_invalid_request(
        shared_server: McpProcess, line: str) -> None:
    _one_error(exchange(shared_server, line), -32600)


@pytest.mark.parametrize("gap", ["", " "], ids=["touching", "a space apart"])
def test_two_requests_on_one_line_get_one_parse_error_and_neither_runs(
        shared_server: McpProcess, gap: str) -> None:
    """The newline is what frames a message, so two requests on one line are one line
    that does not parse. It gets one parse error with a null id, because no id can be
    read from a line that did not parse, and neither write runs."""
    before = rows(shared_server.db)
    line = (call_line(1, "memory_remember", {"predicate": "likes", "object": "first"})
            + gap + call_line(2, "memory_remember", {"predicate": "likes", "object": "second"}))
    _one_error(exchange(shared_server, line), -32700)
    assert changed(before, rows(shared_server.db)) == []


def test_blank_lines_and_crlf_line_endings_are_framing_and_get_no_reply(
        shared_server: McpProcess) -> None:
    """`iter_messages` in `memvara/server/protocol.py` skips blank lines and strips the
    whitespace around a message, so a client that ends its lines with CRLF, or pads
    between messages, is not failed for it."""
    replies = exchange(shared_server, "", "   ", "\t", "\r", request_line(7, "ping") + "\r")
    assert replies == [{"jsonrpc": "2.0", "id": 7, "result": {}}]


# -- nesting ----------------------------------------------------------------------------

PLACES: dict[str, Callable[[int], str]] = {
    "params": lambda depth: ('{"jsonrpc":"2.0","id":5,"method":"ping","params":{"x":'
                             + _nested(depth) + "}}"),
    "an argument": lambda depth: (
        '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"memory_remember",'
        '"arguments":{"predicate":"likes","object":' + _nested(depth) + "}}}"),
    "the id": lambda depth: '{"jsonrpc":"2.0","id":' + _nested(depth) + ',"method":"ping"}',
    "the top level": _nested,
    "objects": lambda depth: ('{"jsonrpc":"2.0","id":5,"method":"ping","params":'
                              + '{"a":' * depth + "1" + "}" * depth + "}"),
}


@pytest.mark.parametrize("place", list(PLACES))
def test_a_line_nested_too_deeply_gets_one_parse_error_and_nothing_runs(
        shared_server: McpProcess, place: str) -> None:
    """The fix for #268: Python's JSON decoder raises RecursionError, not ValueError, on
    nesting deeper than the interpreter allows, and `decode` in
    `memvara/server/protocol.py` turns it into a parse error. Its id is null, because no
    id can be read from a line that did not parse. Wherever the nesting sits, the server
    answers once, runs nothing and carries on.

    The same line nested three deep parses, which shows that the parse error comes from
    the depth and not from a line this test built wrongly."""
    build = PLACES[place]
    shallow = exchange(shared_server, build(3))
    assert len(shallow) == 1 and shallow[0].get("error", {}).get("code") != -32700, shallow
    before = rows(shared_server.db)
    _one_error(exchange(shared_server, build(DEEP)), -32700)
    assert changed(before, rows(shared_server.db)) == []


def test_the_deepest_nesting_that_still_parses_is_answered_and_the_server_carries_on(
        shared_server: McpProcess) -> None:
    """Just inside the decoder's limit, a line parses, and the server must still build a
    reply from it: here, a refusal that describes the whole nested value it was sent
    where an integer goes. That reply is built by recursion too, so this is the depth at
    which a line could parse and then fail on the way out. The limit depends on the
    Python version, so the test searches for the deepest nesting that is not a parse
    error, and checks it and the three depths just inside it."""
    def reply_to(depth: int) -> dict[str, Any]:
        line = ('{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":'
                '"memory_recall","arguments":{"query":"tea","k":' + _nested(depth) + "}}}")
        replies = exchange(shared_server, line)
        assert len(replies) == 1, replies
        return replies[0]

    low, high = 1, DEEP
    assert "error" not in reply_to(low) and "error" in reply_to(high)
    while high - low > 1:
        middle = (low + high) // 2
        if "error" in reply_to(middle):
            high = middle
        else:
            low = middle
    for depth in range(max(1, low - 3), low + 1):
        reply = reply_to(depth)
        assert reply["id"] == 6 and reply["result"]["isError"] is True, reply
        assert text_of(reply).startswith("memory_recall.k "), text_of(reply)[:200]


def test_an_integer_past_pythons_digit_limit_gets_a_parse_error_and_the_server_carries_on(
        shared_server: McpProcess) -> None:
    """Python 3.11, and 3.10 from 3.10.7, refuse to parse an integer of more than 4,300
    digits, which guards against a conversion that takes quadratic time. The server
    reports that as a parse error with a null id, whether the integer is the id or an
    argument. An interpreter without the limit parses it, and then the request is
    answered as usual."""
    huge = "9" * 5000
    limited = hasattr(sys, "get_int_max_str_digits")
    for line in ('{"jsonrpc":"2.0","id":' + huge + ',"method":"ping"}',
                 '{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":'
                 '"memory_recall","arguments":{"query":"tea","budget":' + huge + "}}}"):
        replies = exchange(shared_server, line)
        if limited:
            _one_error(replies, -32700)
        else:
            assert len(replies) == 1 and "result" in replies[0], replies
