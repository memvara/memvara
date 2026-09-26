"""Lines of 20 MB, over the real pipe.

A client can write a line of any length, and the server reads a whole line before it
parses it. Each test checks that a 20 MB line gets exactly one reply, or none when it is
blank, and that the server still answers a ping afterwards. Storing a 20 MB fact takes
about a minute on a laptop, which is why these tests run nightly.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Callable

from harness.stdio import McpProcess

from .. import call_line, changed, exchange, rows

Start = Callable[..., McpProcess]

MB_20 = 20 * 1024 * 1024

#: Seconds to wait for one reply. The slowest line here takes about a minute.
PATIENCE = 600.0


def _server(mcp: Start) -> McpProcess:
    server = mcp(timeout=PATIENCE)
    server.initialize()
    return server


def _one_reply(server: McpProcess, line: str) -> dict[str, Any]:
    replies = exchange(server, line, timeout=PATIENCE)
    assert len(replies) == 1, [str(reply)[:300] for reply in replies]
    return replies[0]


def test_a_request_padded_with_20_mb_of_whitespace_is_answered_once(mcp: Start) -> None:
    server = _server(mcp)
    reply = _one_reply(server, '{"jsonrpc":"2.0","id":1,' + " " * MB_20 + '"method":"ping"}')
    assert reply == {"jsonrpc": "2.0", "id": 1, "result": {}}


def test_a_20_mb_line_that_is_not_json_gets_one_parse_error(mcp: Start) -> None:
    server = _server(mcp)
    reply = _one_reply(server, "x" * MB_20)
    assert reply["id"] is None and reply["error"]["code"] == -32700


def test_a_20_mb_line_of_spaces_is_a_blank_line_and_gets_no_reply(mcp: Start) -> None:
    """`iter_messages` strips a line before it looks at it, so this is a blank line."""
    server = _server(mcp)
    assert exchange(server, " " * MB_20, timeout=PATIENCE) == []


def test_20_mb_where_an_integer_goes_is_refused_and_changes_nothing(mcp: Start) -> None:
    server = _server(mcp)
    before = rows(server.db)
    reply = _one_reply(server, call_line(2, "memory_recall", {"query": "tea", "k": "9" * MB_20}))
    assert reply["id"] == 2 and reply["result"]["isError"] is True
    assert changed(before, rows(server.db)) == []


def test_a_20_mb_argument_name_is_refused_and_changes_nothing(mcp: Start) -> None:
    server = _server(mcp)
    before = rows(server.db)
    reply = _one_reply(server, call_line(3, "memory_recall", {"query": "tea", "z" * MB_20: 1}))
    assert reply["id"] == 3 and reply["result"]["isError"] is True
    assert changed(before, rows(server.db)) == []


def test_a_20_mb_fact_is_stored_whole_and_the_server_carries_on(mcp: Start) -> None:
    """`object` has no length cap on purpose: it is the fact itself (see `_SUBJECT_CHARS`
    in `memvara/server/tools.py`). So a 20 MB fact is stored whole. The value has no
    whitespace at either end, which the store trims."""
    server = _server(mcp)
    value = " ".join(["word"] * (MB_20 // 5))
    reply = _one_reply(server, call_line(4, "memory_remember",
                                         {"predicate": "notes", "object": value}))
    assert reply["id"] == 4 and reply["result"]["isError"] is False
    connection = sqlite3.connect(f"{server.db.resolve().as_uri()}?mode=ro", uri=True)
    try:
        lengths = [length for (length,) in connection.execute(
            "SELECT length(object) FROM claims WHERE predicate = 'notes'")]
    finally:
        connection.close()
    assert lengths == [len(value)]


def test_a_20_mb_query_is_answered_once(mcp: Start) -> None:
    server = _server(mcp)
    reply = _one_reply(server, call_line(5, "memory_search", {"query": "word " * (MB_20 // 5)}))
    assert reply["id"] == 5 and "result" in reply
