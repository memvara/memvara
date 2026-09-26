"""The bugs the fuzz tests found, each pinned by a strict expected failure that cites its
issue.

Each test states the behaviour the fix must produce. It raises known_bugs.Reproduced only
when it has seen that bug's own symptom, so a different failure in the same test fails
loudly instead of passing for the known bug.
"""

from __future__ import annotations

import json
import math
from typing import Any, Callable

import pytest

from harness import known_bugs
from harness.stdio import McpProcess, McpProcessError
from memvara.server.tools import BY_NAME, TOOLS, Tool
from memvara.server.validate import ToolError, validate

from . import call_line, changed, exchange, rows, text_of
from . import shared_server  # noqa: F401 - a fixture; importing it lets pytest find it here

Start = Callable[..., McpProcess]

#: An argument far longer than any reply should quote.
LONG = "y" * 100_000

#: Every argument whose schema type is a number, as (tool, argument name).
NUMBERS = [(tool, name) for tool in TOOLS for name, spec in tool.properties.items()
           if spec["type"] == "number"]


def _label(pair: tuple[Tool, str]) -> str:
    return f"{pair[0].name}.{pair[1]}"


# -- B34: the server reads its input in the locale's encoding ----------------------------
# PYTHONIOENCODING sets the stream encoding the same way on every platform, so these
# tests do not depend on which locales a machine has installed. "utf-8:strict" is what
# LC_ALL=en_US.UTF-8 gives on macOS, and cp1252 is a Windows pipe's default.

@known_bugs.xfail("B34")
def test_a_byte_that_is_not_utf8_gets_a_parse_error_and_the_server_carries_on(
        mcp: Start) -> None:
    server = mcp(env={"PYTHONIOENCODING": "utf-8:strict"})
    server.initialize()
    try:
        replies = exchange(
            server, b'{"jsonrpc":"2.0","id":7,"method":"ping","params":{"x":"\xff"}}')
    except McpProcessError as exc:
        if "UnicodeDecodeError" in str(exc):
            raise known_bugs.Reproduced(str(exc)[:300]) from exc
        raise
    assert len(replies) == 1, replies
    assert replies[0]["error"]["code"] == -32700, replies


@known_bugs.xfail("B34")
def test_the_server_reads_utf8_input_whatever_its_stream_encoding(mcp: Start) -> None:
    """A Node client writes "Zürich" as UTF-8 bytes, because JSON.stringify does not
    escape non-ASCII characters."""
    server = mcp(env={"PYTHONIOENCODING": "cp1252"})
    server.initialize()
    request = json.loads(call_line(3, "memory_remember",
                                   {"predicate": "lives_in", "object": "Zürich"}))
    replies = exchange(server, json.dumps(request, ensure_ascii=False).encode("utf-8"))
    assert len(replies) == 1, replies
    text = text_of(replies[0])
    if "user lives in ZÃ¼rich" in text:
        raise known_bugs.Reproduced(text)
    assert "user lives in Zürich" in text, text


# -- B35: NaN passes the bounds -----------------------------------------------------------

@pytest.mark.parametrize("pair", NUMBERS, ids=_label)
@known_bugs.xfail("B35")
def test_nan_is_refused_by_the_bounds_of_every_number_argument(
        pair: tuple[Tool, str]) -> None:
    """Every comparison with NaN is false, so a bound written as `value < low` or
    `value > high` lets it through."""
    tool, name = pair
    arguments: dict[str, Any] = {required: "tea" for required in tool.required}
    arguments[name] = math.nan
    try:
        validate(tool.properties, tool.required, arguments, tool=tool.name)
    except ToolError as exc:
        assert f"{tool.name}.{name}" in str(exc), exc
        return
    raise known_bugs.Reproduced(f"{tool.name}.{name} accepted NaN")


# -- B36: a refusal or a no-match reply quotes the whole argument -------------------------

@known_bugs.xfail("B36")
def test_a_refusal_quotes_only_a_short_part_of_the_value_it_refuses(
        shared_server: McpProcess) -> None:
    """A refusal is read by a model. `safe_detail` caps the detail of other failures at
    300 characters."""
    before = rows(shared_server.db)
    replies = exchange(shared_server, call_line(1, "memory_recall", {"query": "tea", "k": LONG}))
    assert len(replies) == 1 and replies[0]["result"]["isError"] is True, replies
    assert changed(before, rows(shared_server.db)) == []
    text = text_of(replies[0])
    if LONG in text:
        raise known_bugs.Reproduced(f"a refusal of {len(text)} characters")
    assert len(text) < 2_000, len(text)


@known_bugs.xfail("B36")
def test_a_read_that_finds_nothing_quotes_only_a_short_part_of_the_query(
        mcp: Start) -> None:
    server = mcp()
    server.initialize()
    texts = {}
    for tool in ("memory_search", "memory_recall"):
        replies = exchange(server, call_line(1, tool, {"query": LONG}))
        assert len(replies) == 1 and replies[0]["result"]["isError"] is False, replies
        texts[tool] = text_of(replies[0])
    whole = sorted(tool for tool, text in texts.items() if LONG in text)
    if whole:
        raise known_bugs.Reproduced(f"{whole} quote the whole query")
    assert all(len(text) < 2_000 for text in texts.values()), {
        tool: len(text) for tool, text in texts.items()}


# -- B37: the key pattern's $ matches before a final newline -----------------------------

@known_bugs.xfail("B37")
def test_a_filter_key_that_ends_in_a_newline_is_refused() -> None:
    """The schema's key pattern is ^[A-Za-z0-9_.-]{1,64}$. In Python, $ also matches just
    before a newline at the end of a string, so re.search lets "team\\n" through."""
    tool = BY_NAME["memory_search"]
    try:
        validate(tool.properties, tool.required,
                 {"query": "tea", "filters": {"team\n": "support"}}, tool=tool.name)
    except ToolError as exc:
        assert "memory_search.filters" in str(exc), exc
        return
    raise known_bugs.Reproduced("memory_search accepted the filter key 'team\\n'")


# -- B38: a lone surrogate in an object key is stored ------------------------------------

@known_bugs.xfail("B38")
def test_a_lone_surrogate_in_an_object_key_is_refused_like_one_in_a_value(
        shared_server: McpProcess) -> None:
    before = rows(shared_server.db)
    replies = exchange(shared_server, call_line(1, "memory_add_document", {
        "content": "Refunds last thirty days.", "metadata": {"a\ud800b": "support"}}))
    assert len(replies) == 1, replies
    text = text_of(replies[0])
    if replies[0]["result"]["isError"] is False and text.startswith("Stored"):
        raise known_bugs.Reproduced(text)
    assert replies[0]["result"]["isError"] is True, text
    assert "unpaired surrogate" in text, text
    assert changed(before, rows(shared_server.db)) == []


# -- B39: memory_recall refuses ranked without turns from the catch-all ------------------

@pytest.mark.parametrize("arguments", [
    {"query": "tea", "ranked": True},
    {"query": "tea", "ranked": True, "include_episodes": True,
     "memory_types": ["semantic"]},
], ids=["without include_episodes", "with memory_types"])
@known_bugs.xfail("B39")
def test_ranked_recall_without_turns_is_refused_as_an_argument_error(
        shared_server: McpProcess, arguments: dict[str, Any]) -> None:
    """memory_search refuses its own invalid combination, as_of with valid_at, at the
    tool boundary, so the model reads an argument error rather than a Python exception."""
    replies = exchange(shared_server, call_line(2, "memory_recall", arguments))
    assert len(replies) == 1 and replies[0]["result"]["isError"] is True, replies
    text = text_of(replies[0])
    if text.startswith("memory_recall failed: ValueError: ranked=True needs turns to rank"):
        raise known_bugs.Reproduced(text)
    assert "ranked" in text and not text.startswith("memory_recall failed:"), text
