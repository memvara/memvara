"""The three MCP protocol versions the server speaks, each agreed over a real pipe.

The server agrees to whichever supported version a client asks for, and answers every
version the same way otherwise. A version it does not speak gets its newest, as the MCP
lifecycle rule asks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from harness import stores
from harness.stdio import McpProcess
from memvara.server.mcp import PROTOCOL_VERSION, SUPPORTED_PROTOCOLS, MemvaraMCPServer


@dataclass(frozen=True)
class Handshake:
    """What one server answered, in the order a client asks."""

    agreed: dict[str, Any]
    tools: list[dict[str, Any]]
    stats: str
    batch: dict[str, Any]
    exit_code: int


@pytest.fixture(scope="module")
def handshakes(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Handshake]:
    """One real server for each supported version, each on a new store and asked the
    same things in the same order."""
    home = tmp_path_factory.mktemp("protocol-home")
    found = {}
    for version in SUPPORTED_PROTOCOLS:
        server = McpProcess(tmp_path_factory.mktemp("protocol") / "memory.db", home=home)
        try:
            agreed = server.initialize(version)
            tools = server.list_tools()
            stats = server.call("memory_stats").text
            server.send_raw(json.dumps([{"jsonrpc": "2.0", "id": 90, "method": "ping"}]))
            batch = server.recv()
            server.request("ping")
            code = server.close()
        finally:
            server.kill()
        found[version] = Handshake(agreed, tools, stats, batch, code)
    return found


@pytest.mark.parametrize("version", SUPPORTED_PROTOCOLS)
def test_the_server_agrees_to_each_version_it_supports(
        version: str, handshakes: dict[str, Handshake]) -> None:
    assert handshakes[version].agreed["protocolVersion"] == version
    assert handshakes[version].exit_code == 0


def test_the_versions_differ_only_in_the_version_agreed(
        handshakes: dict[str, Handshake]) -> None:
    """The same capabilities, server information and instructions, the same tools and the
    same tool result under every version: a client on any of the three sees one server."""
    def rest(handshake: Handshake) -> dict[str, Any]:
        return {k: v for k, v in handshake.agreed.items() if k != "protocolVersion"}

    first, *others = SUPPORTED_PROTOCOLS
    for version in others:
        assert rest(handshakes[version]) == rest(handshakes[first])
        assert handshakes[version].tools == handshakes[first].tools
        assert handshakes[version].stats == handshakes[first].stats


@pytest.mark.parametrize("version", SUPPORTED_PROTOCOLS)
def test_a_batch_is_refused_whichever_version_was_agreed(
        version: str, handshakes: dict[str, Handshake]) -> None:
    """JSON-RPC batches belong to revision 2025-03-26 and were removed in 2025-06-18. The
    server refuses a batch under every version, which the design lists as documented
    behaviour ("Batches are refused", docs/superpowers/specs/
    2026-09-25-adversarial-test-suite-design.md). The server keeps answering afterwards,
    because the fixture's ping after the batch was answered."""
    assert handshakes[version].batch == {
        "jsonrpc": "2.0", "id": None,
        "error": {"code": -32600,
                  "message": "expected a single JSON-RPC request object per line"}}


@pytest.mark.parametrize("asked", ["2099-01-01", "2024-10-07", 20250618, None])
def test_a_version_the_server_does_not_speak_gets_its_newest(asked: Any) -> None:
    """MCP's lifecycle rule: a server that does not support the version a client asks for
    answers with one it does, and should answer with its newest. A version that is not a
    string, or no version at all, is treated the same way."""
    params: dict[str, Any] = {"capabilities": {},
                              "clientInfo": {"name": "adversarial", "version": "0"}}
    if asked is not None:
        params["protocolVersion"] = asked
    server = MemvaraMCPServer(stores.memory(), user="tester")
    try:
        reply = server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                       "params": params})
    finally:
        server.close()
    assert reply is not None
    assert reply["result"]["protocolVersion"] == PROTOCOL_VERSION == SUPPORTED_PROTOCOLS[0]
