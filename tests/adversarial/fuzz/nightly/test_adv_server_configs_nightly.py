"""Generated calls against servers configured in the ways that change which tools are
listed and what their schemas say: read-only, anchored by default, and with every
feature that owns a tool or an argument switched off.

A call to a tool the server does not list must be refused with a reason and change
nothing. Otherwise the properties are the fast tier's (test_adv_schema_fuzz.py): one
reply per call, a validator refusal word for word, and no change after any refusal.
"""

from __future__ import annotations

import itertools
import json
from typing import Any, Callable

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from harness.stdio import McpProcess
from memvara.server.tools import TOOLS
from memvara.server.validate import ToolError, validate

from .. import arguments, broken_arguments, call_line, changed, exchange, rows, seed, text_of
from ..test_adv_schema_fuzz import NEVER_SENT

Start = Callable[..., McpProcess]

#: Every feature that owns a tool or an argument.
OWNING = ("profile", "forget_matching", "end_reason", "links", "documents",
          "query_rewrite", "synthesis", "metadata_filters", "expiry_erasure")

CONFIGURATIONS: dict[str, dict[str, Any]] = {
    "read-only": {"read_only": True},
    "anchored": {"env": {"MEMVARA_ANCHORED": "1"}},
    "features off": {"features": {name: False for name in OWNING}},
}

#: Each configuration takes a tenth of the tier's example count.
EACH = settings(max_examples=max(30, settings().max_examples // 10))


@pytest.mark.parametrize("options", list(CONFIGURATIONS.values()), ids=list(CONFIGURATIONS))
def test_generated_calls_to_a_server_configured_another_way_follow_the_same_rules(
        mcp: Start, options: dict[str, Any]) -> None:
    seeder = mcp()
    seeder.initialize()
    seed(seeder)
    seeder.close()
    server = mcp(seeder.db, **options)
    server.initialize()
    schemas = {spec["name"]: spec["inputSchema"] for spec in server.list_tools()}
    ids = itertools.count(1)

    @EACH
    @given(data=st.data())
    def check(data: st.DataObject) -> None:
        name = data.draw(st.sampled_from(sorted(tool.name for tool in TOOLS)), label="tool")
        before = rows(server.db)
        if name not in schemas:
            reply = exchange(server, call_line(next(ids), name, {}))
            assert len(reply) == 1 and reply[0]["result"]["isError"] is True, reply
            assert text_of(reply[0]).startswith(f"{name} is unavailable"), reply
            assert changed(before, rows(server.db)) == []
            return
        properties, required = schemas[name]["properties"], schemas[name]["required"]
        drawn = data.draw(st.one_of(
            arguments(properties, required, leave_out=NEVER_SENT),
            broken_arguments(name, properties, required, leave_out=NEVER_SENT).map(
                lambda pair: pair[0])), label="arguments")
        request_id = next(ids)
        line = call_line(request_id, name, drawn)
        replies = exchange(server, line)
        assert [reply.get("id") for reply in replies] == [request_id], replies
        result = replies[0]["result"]
        try:
            validate(properties, required, json.loads(line)["params"]["arguments"], tool=name)
        except ToolError as refusal:
            assert result["isError"] is True and text_of(replies[0]) == str(refusal)
        if result["isError"]:
            assert changed(before, rows(server.db)) == []

    check()
