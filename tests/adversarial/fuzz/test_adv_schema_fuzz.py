"""Arguments drawn by Hypothesis from each tool's own input schema.

In the test process, every tool's validator must accept every call that follows its
schema, refuse every call that breaks one rule of it with a message naming the rule, and
never raise anything but a refusal, whatever JSON it is handed. Over the real pipe, a
sample of the same calls must each get one reply, a refusal must be the validator's own
message and change nothing in the store, and the server must still answer a ping.
"""

from __future__ import annotations

import itertools
import json
from typing import Any, Callable

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from harness.stdio import McpProcess
from memvara.server.tools import TOOLS, Tool
from memvara.server.validate import ToolError, validate

from . import (ANY_TEXT, arguments, broken_arguments, call_line, changed, exchange,
               json_values, rows, seed, text_of)

Start = Callable[..., McpProcess]

#: Examples per tool. The tier's profile sets a count for a whole test, and each property
#: here runs once for every one of the 22 tools, so the nightly and weekly tiers take a
#: tenth of their count per tool (300 and 2,000), and the fast tier keeps its 30.
PER_TOOL = settings(max_examples=max(30, settings().max_examples // 10))

#: Arguments never sent to a real server. memory_add_document fetches a url, and this
#: suite does not reach the network.
NEVER_SENT = frozenset({"url"})


def _as_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


@pytest.mark.parametrize("tool", TOOLS, ids=lambda tool: tool.name)
@PER_TOOL
@given(data=st.data())
def test_a_call_that_follows_the_schema_is_accepted_and_its_defaults_filled_in(
        tool: Tool, data: st.DataObject) -> None:
    """Every argument the caller sent comes back unchanged, every argument it left out
    that has a default comes back as that default, and nothing else is added. Running the
    result through the validator again changes nothing."""
    sent = data.draw(arguments(tool.properties, tool.required))
    out = validate(tool.properties, tool.required, sent, tool=tool.name)
    expected = {name: spec["default"] for name, spec in tool.properties.items()
                if "default" in spec}
    expected.update(sent)
    assert _as_json(out) == _as_json(expected)
    again = validate(tool.properties, tool.required, out, tool=tool.name)
    assert _as_json(again) == _as_json(out)


@pytest.mark.parametrize("tool", TOOLS, ids=lambda tool: tool.name)
@PER_TOOL
@given(data=st.data())
def test_a_call_that_breaks_one_rule_is_refused_with_one_line_that_names_it(
        tool: Tool, data: st.DataObject) -> None:
    """The refusal starts with the tool's name and the argument at fault, which is what
    lets a model correct its call, and it is one line, because a value quoted in it
    must not be able to start a line of its own."""
    sent, start = data.draw(broken_arguments(tool.name, tool.properties, tool.required))
    with pytest.raises(ToolError) as refused:
        validate(tool.properties, tool.required, sent, tool=tool.name)
    message = str(refused.value)
    assert message.startswith(start), (message[:300], start)
    assert message.splitlines() == [message], message[:300]


@pytest.mark.parametrize("tool", TOOLS, ids=lambda tool: tool.name)
@PER_TOOL
@given(data=st.data())
def test_any_json_as_arguments_is_accepted_or_refused_and_nothing_else_is_raised(
        tool: Tool, data: st.DataObject) -> None:
    """Whatever JSON arrives, the validator answers with arguments or with a refusal. Any
    other exception would reach the model as a Python error instead of something it can
    act on, which `validate.py`'s docstring tells happened once for booleans."""
    names = (st.sampled_from(sorted(tool.properties)) | ANY_TEXT if tool.properties
             else ANY_TEXT)
    values = json_values(max_leaves=6)
    sent = data.draw(st.dictionaries(names, values, max_size=4) | values)
    try:
        validate(tool.properties, tool.required, sent, tool=tool.name)
    except ToolError:
        pass


def test_a_sample_of_generated_calls_each_get_one_reply_and_a_refusal_changes_nothing(
        mcp: Start) -> None:
    """Calls drawn from the schemas the server itself lists, some following them and some
    breaking one rule, sent over the real pipe:

    * each call gets exactly one reply, with its id, and the server then answers a ping;
    * a call the validator refuses gets the validator's own message, word for word;
    * a call that comes back as an error has changed nothing in the store.
    """
    server = mcp()
    server.initialize()
    seed(server)
    schemas = {spec["name"]: spec["inputSchema"] for spec in server.list_tools()}
    ids = itertools.count(1)

    @given(data=st.data())
    def check(data: st.DataObject) -> None:
        name = data.draw(st.sampled_from(sorted(schemas)), label="tool")
        properties, required = schemas[name]["properties"], schemas[name]["required"]
        drawn = data.draw(st.one_of(
            arguments(properties, required, leave_out=NEVER_SENT),
            broken_arguments(name, properties, required, leave_out=NEVER_SENT).map(
                lambda pair: pair[0])), label="arguments")
        request_id = next(ids)
        line = call_line(request_id, name, drawn)
        received = json.loads(line)["params"]["arguments"]  # what the server will parse
        before = rows(server.db)
        replies = exchange(server, line)
        assert [reply.get("id") for reply in replies] == [request_id], replies
        result = replies[0]["result"]
        try:
            validate(properties, required, received, tool=name)
        except ToolError as refusal:
            assert result["isError"] is True and text_of(replies[0]) == str(refusal)
        if result["isError"]:
            assert changed(before, rows(server.db)) == []

    check()
