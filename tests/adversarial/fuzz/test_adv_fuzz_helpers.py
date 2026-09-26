"""The helpers in this folder catch what they exist to catch.

A helper that cannot see a fault makes every test built on it pass for the wrong reason.
So each helper is shown catching its fault here, without a server wherever one is not
needed.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest
from hypothesis import given, settings

from harness import stores
from harness.stdio import McpProcessError
from memvara.server.tools import TOOLS

from . import SCHEMA_KEYWORDS, arguments, changed, conforming, exchange, request_line, rows


class Scripted:
    """A stand-in for McpProcess that answers each line with the replies a test gave it,
    and answers the fence ping that `exchange` sends last."""

    def __init__(self, replies: dict[str, list[dict[str, Any]]],
                 fence_result: Any = None) -> None:
        self.replies = replies
        self.fence_result = {} if fence_result is None else fence_result
        self.waiting: list[dict[str, Any]] = []

    def send_raw(self, data: str | bytes) -> None:
        line = data.decode() if isinstance(data, bytes) else data
        message = json.loads(line)
        if str(message.get("id", "")).startswith("fence-"):
            self.waiting.append({"jsonrpc": "2.0", "id": message["id"],
                                 "result": self.fence_result})
        else:
            self.waiting.extend(self.replies.get(line, []))

    def recv(self, timeout: float | None = None) -> dict[str, Any]:
        return self.waiting.pop(0)


def test_exchange_returns_both_replies_when_one_request_gets_two() -> None:
    line = request_line(1, "ping")
    twice = [{"jsonrpc": "2.0", "id": 1, "result": {}}] * 2
    assert exchange(Scripted({line: twice}), line) == twice  # type: ignore[arg-type]


def test_exchange_returns_nothing_for_a_request_that_got_no_reply() -> None:
    assert exchange(Scripted({}), request_line(1, "ping")) == []  # type: ignore[arg-type]


def test_exchange_fails_when_the_fence_ping_is_not_answered_with_an_empty_result() -> None:
    with pytest.raises(McpProcessError, match="fence ping"):
        exchange(Scripted({}, fence_result={"surprise": 1}),  # type: ignore[arg-type]
                 request_line(1, "ping"))


def test_rows_sees_a_write_in_the_tables_and_in_the_vector_file(
        tmp_path: pathlib.Path) -> None:
    db = tmp_path / "memory.db"
    memory = stores.file(db)
    try:
        user = memory.scope(user="tester")
        user.remember("user", "lives_in", "Berlin")
        before = rows(db)
        assert changed(before, rows(db)) == []
        user.remember("user", "likes", "green tea")
        after = rows(db)
    finally:
        memory.close()
    assert {"claims", "embeddings", ".vecs"} <= set(changed(before, after))


def test_every_schema_keyword_a_tool_uses_is_one_the_strategies_understand() -> None:
    """A keyword the strategies did not know would be ignored while values are drawn, so
    the fuzzer would send values the schema forbids as if they were allowed, and never
    try the ones it forbids."""
    used: set[str] = set()

    def walk(spec: dict[str, Any]) -> None:
        used.update(spec)
        for key in ("items", "additionalProperties", "propertyNames"):
            if isinstance(spec.get(key), dict):
                walk(spec[key])

    for tool in TOOLS:
        for spec in tool.properties.values():
            walk(spec)
    assert used <= SCHEMA_KEYWORDS, sorted(used - SCHEMA_KEYWORDS)


def test_a_number_range_with_no_whole_number_in_it_draws_numbers_inside_it() -> None:
    """A range such as 0.1 to 0.2 holds no integer, so the strategy must not try to draw
    one from an empty integer range."""

    @settings(max_examples=30, database=None, derandomize=True)
    @given(conforming({"type": "number", "minimum": 0.1, "maximum": 0.2}))
    def draws(value: float) -> None:
        assert 0.1 <= value <= 0.2

    draws()


def test_an_argument_that_must_not_be_sent_cannot_also_be_required() -> None:
    """`leave_out` promises that an argument is never drawn, and a required argument is
    always drawn, so asking for both is a mistake in the test and is refused."""
    with pytest.raises(ValueError, match="url"):
        arguments({"url": {"type": "string"}}, ["url"], leave_out={"url"})

