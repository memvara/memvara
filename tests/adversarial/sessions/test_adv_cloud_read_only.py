"""A cloud-mode server whose API key may only read, started over a real pipe against the
FakeV1 fake of the hosted API.

The server asks the deployment about its credential once, at startup, with GET /v1/stats,
and a read-only answer hides every write tool as MEMVARA_READ_ONLY would
(`_service_facts` in memvara/server/mcp.py). A write asked for by name is refused by the
server itself, so the deployment never receives it.
"""

from __future__ import annotations

from typing import Callable

from harness.fakes.fake_v1 import FakeV1
from harness.stdio import McpProcess

from . import switches


def cloud(fake: FakeV1) -> dict[str, str]:
    """The variables that point a server at the fake in cloud mode. The mcp fixture also
    sets MEMVARA_DB, which cloud mode ignores."""
    return {"MEMVARA_MODE": "cloud", "MEMVARA_API_KEY": fake.api_key,
            "MEMVARA_SERVER_URL": fake.serve()}


def test_a_read_only_key_lists_only_read_tools_and_refuses_every_write_by_name(
        mcp: Callable[..., McpProcess]) -> None:
    read_only = switches.Combination.of("read_only")
    with FakeV1(read_only=True) as fake:
        server = mcp(env=cloud(fake))
        server.initialize()
        problems = (switches.listing_problems(server, read_only)
                    + switches.refusal_problems(server, read_only,
                                                switches.minimal_calls(switches.NOWHERE)))
        found = server.call("memory_search", query="where does the user live")
        stats = server.call("memory_stats").text
        assert server.close() == 0
    # First, because it names the likely cause of the failures below: when the startup
    # probe of the credential times out, the server treats the key as one that may write.
    assert "writes: disabled — this server is read-only" in stats
    assert problems == []
    assert not found.is_error, found.text
    routes = {request.route for request in fake.requests}
    assert {"GET /v1/stats", "POST /v1/search"} <= routes
    assert [request.route for request in fake.requests if request.status != 200] == []


def test_a_writable_key_lists_every_tool(mcp: Callable[..., McpProcess]) -> None:
    """The control for the test above: with a key that may write, the same server lists
    the write tools, so the difference comes from the credential."""
    with FakeV1() as fake:
        server = mcp(env=cloud(fake))
        server.initialize()
        problems = switches.listing_problems(server, switches.Combination.of())
        assert server.close() == 0
    assert problems == []
