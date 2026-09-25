"""Argument values a tool's schema does not allow, sent over the real pipe to every tool.

The validator in `memvara/server/validate.py` refuses each of these before the tool runs.
So each test checks the three things a refusal must do: come back as one tool error that
names the argument, change nothing in the store, and leave the server answering.
"""

from __future__ import annotations

import math
import re
from typing import Any

import pytest

from harness.stdio import McpProcess
from memvara.server.tools import TOOLS, Tool

from . import call_line, changed, exchange, rows, text_of
from . import shared_server  # noqa: F401 - a fixture; importing it lets pytest find it here

#: A valid value for each required argument, so that the one argument a test breaks is
#: the only thing wrong with the call. A tool that gains a required argument fails here
#: with a KeyError until it is added.
REQUIRED: dict[str, Any] = {
    "query": "tea", "question": "what do I like", "entity": "user", "source": "user",
    "target": "tea", "since": "2024-01-01", "text": "I like tea.", "predicate": "likes",
    "object": "tea", "from_id": "cl_0", "to_id": "cl_1", "relation": "extends",
    "claim_id": "cl_0", "id": "doc_0",
}

#: A lone surrogate: half of a character, which cannot be encoded as UTF-8.
LONE = "a\ud800b"


def _minimal(tool: Tool) -> dict[str, Any]:
    return {name: REQUIRED[name] for name in tool.required}


def _types(spec: dict[str, Any]) -> list[str]:
    return spec["type"] if isinstance(spec["type"], list) else [spec["type"]]


def _of_type(kind: str) -> list[tuple[Tool, str]]:
    return [(tool, name) for tool in TOOLS for name, spec in tool.properties.items()
            if _types(spec) == [kind]]


def _label(pair: tuple[Tool, str]) -> str:
    return f"{pair[0].name}.{pair[1]}"


def _refused(server: McpProcess, tool: str, arguments: Any) -> str:
    """Send one call, check that it got exactly one reply, that the reply is a tool error,
    and that the store did not change, and return the reply's text."""
    before = rows(server.db)
    replies = exchange(server, call_line(1, tool, arguments))
    assert len(replies) == 1 and replies[0].get("id") == 1, replies
    assert replies[0]["result"]["isError"] is True, replies[0]
    assert changed(before, rows(server.db)) == []
    return text_of(replies[0])


BOOLEANS = _of_type("boolean")
INTEGERS = _of_type("integer")
NUMBERS = _of_type("number")


@pytest.mark.parametrize("value", ["false", "true", 0, 1], ids=repr)
@pytest.mark.parametrize("pair", BOOLEANS, ids=_label)
def test_a_string_or_a_number_where_a_boolean_goes_is_refused_by_name(
        shared_server: McpProcess, pair: tuple[Tool, str], value: Any) -> None:
    """Every handler reads its flags through `bool(...)`, where the string "false" is
    True, so the validator accepts only a real boolean. Its docstring tells how "false"
    once reached production as True."""
    tool, name = pair
    text = _refused(shared_server, tool.name, {**_minimal(tool), name: value})
    assert text.startswith(f"{tool.name}.{name} must be a boolean"), text


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf],
                         ids=["NaN", "Infinity", "-Infinity"])
@pytest.mark.parametrize("pair", INTEGERS, ids=_label)
def test_nan_and_the_infinities_are_refused_where_an_integer_goes(
        shared_server: McpProcess, pair: tuple[Tool, str], value: float) -> None:
    """The server's JSON parser accepts the tokens NaN, Infinity and -Infinity, so they
    reach the validator as floats, and a float is not an integer."""
    tool, name = pair
    text = _refused(shared_server, tool.name, {**_minimal(tool), name: value})
    assert text.startswith(f"{tool.name}.{name} must be an integer"), text


@pytest.mark.parametrize("value", [math.inf, -math.inf], ids=["Infinity", "-Infinity"])
@pytest.mark.parametrize("pair", NUMBERS, ids=_label)
def test_an_infinite_number_is_refused_by_the_bound_it_breaks(
        shared_server: McpProcess, pair: tuple[Tool, str], value: float) -> None:
    """The refusal names the bound, which is what tells the caller the value to send."""
    tool, name = pair
    spec = tool.properties[name]
    bound = f"<= {spec['maximum']}" if value > 0 else f">= {spec['minimum']}"
    text = _refused(shared_server, tool.name, {**_minimal(tool), name: value})
    assert text.startswith(f"{tool.name}.{name} must be {bound}"), text


@pytest.mark.parametrize("case", ["a new slot", "a slot that holds a value",
                                  "a named replacement"])
def test_a_nan_confidence_is_refused_and_leaves_the_slot_as_it_was(
        shared_server: McpProcess, case: str) -> None:
    """A NaN confidence must never be stored. A write refused for one must leave the store
    as it was, including the value it would otherwise have ended: confidence decides
    whether a write may end the value already in its slot, and every comparison with NaN
    is false."""
    arguments: dict[str, Any] = {"predicate": "lives_in", "object": "Oslo",
                                 "confidence": math.nan}
    if case == "a new slot":
        arguments["predicate"] = "works_in"
    elif case == "a named replacement":
        found = shared_server.call("memory_search", query="Berlin").text
        berlin = re.search(r"id=(cl_[0-9a-f]+)[^\n]*lives in Berlin", found)
        assert berlin is not None, found
        arguments["replaces"] = berlin.group(1)
    _refused(shared_server, "memory_remember", arguments)


def _surrogate_cases() -> list[Any]:
    """Each place in a tool's arguments where its schema says a string goes, holding a
    lone surrogate: a string argument, a string item of an array, a string value of an
    object, and an object's key where the schema declares `propertyNames` for its keys.
    Each case carries the label the refusal must start with."""
    cases = []
    for tool in TOOLS:
        for name, spec in tool.properties.items():
            label = f"{tool.name}.{name}"
            kinds = _types(spec)
            if "string" in kinds:
                cases.append(pytest.param(tool, name, LONE, label, id=label))
            if "array" in kinds and "string" in _types(spec["items"]):
                # The first item is one the schema allows, so that the lone surrogate in
                # the second is the only thing wrong.
                allowed = spec["items"].get("enum", ["tea"])[0]
                cases.append(pytest.param(tool, name, [allowed, LONE], f"{label}[1]",
                                          id=f"{label} item"))
            if "object" in kinds:
                values = spec["additionalProperties"]
                if "string" in _types(values):
                    cases.append(pytest.param(tool, name, {"team": LONE}, f"{label}.team",
                                              id=f"{label} value"))
                elif "array" in _types(values):
                    cases.append(pytest.param(tool, name, {"team": [LONE]},
                                              f"{label}.team[0]", id=f"{label} value item"))
                if "propertyNames" in spec:
                    cases.append(pytest.param(tool, name, {LONE: "tea"},
                                              f"{label} key {LONE!r}", id=f"{label} key"))
    return cases


@pytest.mark.parametrize(("tool", "name", "value", "label"), _surrogate_cases())
def test_a_lone_surrogate_is_refused_wherever_the_schema_says_a_string_goes(
        shared_server: McpProcess, tool: Tool, name: str, value: Any, label: str) -> None:
    """Python's JSON parser turns the escape \\ud800 into half of a character, which no
    store can encode. The validator refuses it wherever the schema says a string goes,
    and names the argument, rather than the exception a store would raise later."""
    text = _refused(shared_server, tool.name, {**_minimal(tool), name: value})
    assert text.startswith(label), text
    assert "unpaired surrogate" in text, text


def test_a_lone_surrogate_outside_the_arguments_is_answered_by_the_protocol(
        shared_server: McpProcess) -> None:
    """In an argument's name, it is an unknown argument. In a tool's name or a method's,
    it is an unknown tool or an unknown method, answered with the request's id."""
    assert _refused(shared_server, "memory_search", {"query": "tea", LONE: 1}).startswith(
        "memory_search: unknown argument")
    replies = exchange(shared_server, call_line(2, LONE, {}))
    assert [(reply["id"], reply["error"]["code"]) for reply in replies] == [(2, -32602)]
    replies = exchange(shared_server, '{"jsonrpc":"2.0","id":3,"method":"a\\ud800b"}')
    assert [(reply["id"], reply["error"]["code"]) for reply in replies] == [(3, -32601)]
