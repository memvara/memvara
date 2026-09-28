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


@pytest.mark.parametrize("pair", NUMBERS, ids=_label)
def test_an_integer_too_large_for_a_float_is_refused_by_its_bound_not_by_the_nan_check(
        pair: tuple[Tool, str]) -> None:
    """The NaN check must look only at floats. `math.isnan` converts an integer to a
    float first, and an integer of 400 digits, which JSON allows, raises OverflowError
    there instead of reaching the bound it breaks."""
    tool, name = pair
    arguments: dict[str, Any] = {required: "tea" for required in tool.required}
    arguments[name] = 10 ** 400
    with pytest.raises(ToolError, match=rf"^{tool.name}\.{name} must be <= "):
        validate(tool.properties, tool.required, arguments, tool=tool.name)


@pytest.mark.parametrize("tool", ["memory_search", "memory_recall"])
def test_a_nan_floor_sent_over_the_pipe_is_refused(shared_server: McpProcess,
                                                    tool: str) -> None:
    """The validator lets NaN through, and a NaN `min_score` then acts as no floor at
    all, so the read answers as if no floor had been given. It must be refused instead."""
    replies = exchange(shared_server, call_line(1, tool, {"query": "tea",
                                                          "min_score": math.nan}))
    assert len(replies) == 1, replies
    reply = replies[0]
    if "result" in reply and reply["result"]["isError"] is False:
        raise known_bugs.Reproduced(f"{tool} accepted a NaN floor: {text_of(reply)[:120]}")
    assert "error" in reply or reply["result"]["isError"] is True, reply


# -- B36, fixed: a refusal or a no-match reply quoted the whole argument ------------------

def test_a_refusal_quotes_only_a_short_part_of_the_value_it_refuses(
        shared_server: McpProcess) -> None:
    """A refusal is read by a model. `safe_detail` caps the detail of other failures at
    300 characters."""
    before = rows(shared_server.db)
    replies = exchange(shared_server, call_line(1, "memory_recall", {"query": "tea", "k": LONG}))
    assert len(replies) == 1 and replies[0]["result"]["isError"] is True, replies
    assert changed(before, rows(shared_server.db)) == []
    text = text_of(replies[0])
    assert LONG not in text and "(shortened from 100,000 characters)" in text, text[:300]
    assert len(text) < 2_000, len(text)


def test_a_read_that_finds_nothing_quotes_only_a_short_part_of_the_query(
        mcp: Start) -> None:
    server = mcp()
    server.initialize()
    texts = {}
    for tool in ("memory_search", "memory_recall"):
        replies = exchange(server, call_line(1, tool, {"query": LONG}))
        assert len(replies) == 1 and replies[0]["result"]["isError"] is False, replies
        texts[tool] = text_of(replies[0])
    assert not any(LONG in text for text in texts.values()), texts.keys()
    assert all("(shortened from 100,000 characters)" in text for text in texts.values())
    assert all(len(text) < 2_000 for text in texts.values()), {
        tool: len(text) for tool, text in texts.items()}


@pytest.mark.parametrize("tool, arguments", [
    ("memory_since", {"since": LONG}),
    ("memory_profile", {"since": LONG}),
    ("memory_forget", {"claim_id": LONG}),
    ("memory_end", {"claim_id": LONG}),
    ("memory_why", {"claim_id": LONG}),
    ("memory_remember", {"predicate": "likes", "object": "tea", "replaces": LONG}),
    ("memory_profile", {"query": "q" * 2000, "buckets": {LONG: ["likes"]}}),
    ("memory_forget_matching", {"query": "m" * 500, "reason": "r" * 500}),
], ids=["a timestamp", "a timestamp on profile", "memory_forget's id", "memory_end's id",
        "memory_why's id", "memory_remember's replaces", "a profile's query and bucket",
        "a matching query and reason"])
def test_every_reply_that_quotes_an_argument_quotes_only_a_short_part_of_it(
        mcp: Start, tool: str, arguments: dict[str, Any]) -> None:
    """The review of the fix for #313 found the same whole quote in these replies, and
    in the parser's own message about a timestamp it cannot read."""
    server = mcp()
    server.initialize()
    replies = exchange(server, call_line(1, tool, arguments))
    assert len(replies) == 1, replies
    text = text_of(replies[0])
    long_values = [value for value in (*arguments.values(), *arguments.get("buckets", {}))
                   if isinstance(value, str) and len(value) > 80]
    assert long_values
    for value in long_values:
        assert value not in text, (tool, text[:300])
    assert "(shortened from " in text, text[:300]
    assert len(text) < 3_000, len(text)


# -- B37: the key pattern's $ matches before a final newline -----------------------------

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


def _patterns(spec: Any) -> list[str]:
    """Every `pattern` anywhere inside one argument's schema."""
    if isinstance(spec, dict):
        found = [spec["pattern"]] if isinstance(spec.get("pattern"), str) else []
        return found + [p for value in spec.values() for p in _patterns(value)]
    if isinstance(spec, list):
        return [p for value in spec for p in _patterns(value)]
    return []


def test_every_pattern_a_tool_declares_is_anchored_at_both_ends() -> None:
    """The validator matches a pattern against the whole value, which means the same as
    the schema's own pattern only when the pattern is written ^...$. An unanchored
    pattern added later would be applied more strictly than a client reading the schema
    expects, so this test names it."""
    # A list of pairs rather than a dict keyed by argument, because one argument's
    # schema can declare more than one pattern, and a dict would keep only the last.
    patterns = [(f"{tool.name}.{name}", pattern) for tool in TOOLS
                for name, spec in tool.properties.items() for pattern in _patterns(spec)]
    assert patterns, "no tool declares a pattern, so this test checks nothing"
    loose = [(label, pattern) for label, pattern in patterns
             if not (pattern.startswith("^") and pattern.endswith("$"))]
    assert loose == [], loose


# -- B38: a lone surrogate in an object key is stored ------------------------------------

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


# -- B39, fixed: memory_recall refused ranked without turns from the catch-all -----------

@pytest.mark.parametrize("arguments", [
    {"query": "tea", "ranked": True},
    {"query": "tea", "ranked": True, "include_episodes": True,
     "memory_types": ["semantic"]},
], ids=["without include_episodes", "with memory_types"])
def test_ranked_recall_without_turns_is_refused_as_an_argument_error(
        shared_server: McpProcess, arguments: dict[str, Any]) -> None:
    """ranked needs conversation turns to rank. The library refuses a call without them
    with a ValueError, which reached the model through the server's catch-all as a Python
    exception; the tool now refuses it first, as an argument error, the way memory_search
    refuses as_of with valid_at. tests/test_server.py keeps the two checks in step."""
    replies = exchange(shared_server, call_line(2, "memory_recall", arguments))
    assert len(replies) == 1 and replies[0]["result"]["isError"] is True, replies
    text = text_of(replies[0])
    assert text.startswith("memory_recall ranked=true needs turns to rank"), text
