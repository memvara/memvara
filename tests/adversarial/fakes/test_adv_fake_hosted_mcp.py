"""FakeHostedMcp answers both clients that reach the hosted /mcp endpoint: the plugin
hooks' hosted client and the npm bridge."""

from __future__ import annotations

import importlib
import json
import pathlib
import shutil
import subprocess
import sys
from types import ModuleType

import httpx
import pytest

from harness.env import REPO, child_env
from harness.fakes.hosted_mcp import API_KEY, FakeHostedMcp

HOOKS = REPO / "plugin" / "hooks"
BRIDGE = REPO / "npm" / "memvara" / "bin" / "memvara.js"


@pytest.fixture
def hosted() -> ModuleType:
    """The hooks' own hosted client, `plugin/hooks/lib/hosted.py`."""
    if str(HOOKS) not in sys.path:
        sys.path.insert(0, str(HOOKS))
    return importlib.import_module("lib.hosted")


def _methods(fake: FakeHostedMcp) -> list[str | None]:
    return [json.loads(r.body).get("method") if r.body else None for r in fake.requests]


def _node_runs() -> bool:
    """Whether node starts at all. `which node` is not enough, as
    tests/test_npm_release.py found on a machine whose node aborted at start."""
    if shutil.which("node") is None:
        return False
    try:
        subprocess.run(["node", "-e", "process.exit(0)"], check=True, capture_output=True,
                       timeout=8)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return True


def test_the_hooks_client_remembers_and_recalls_through_the_fake(
        hosted_mcp: FakeHostedMcp, hosted: ModuleType) -> None:
    client = hosted.HostedRecall(API_KEY, hosted_mcp.serve())
    try:
        assert "Lisbon" in client.remember("user", "lives_in", "Lisbon")
        assert "Lisbon" in client.recall("where does the user live")
    finally:
        client.close()
    assert _methods(hosted_mcp)[:3] == ["initialize", "notifications/initialized",
                                        "tools/call"]
    first, *rest = hosted_mcp.requests
    assert first.header("mcp-session-id") is None
    assert {r.header("mcp-session-id") for r in rest} == {hosted_mcp.issued[0]}
    assert {r.header("authorization") for r in hosted_mcp.requests} == {f"Bearer {API_KEY}"}
    assert {r.header("user-agent") for r in hosted_mcp.requests} == {hosted.USER_AGENT}
    assert [c.object for c in hosted_mcp.memvara.scope(user="tester").get_all()] \
        == ["Lisbon"]


def test_a_wrong_key_is_refused_with_the_header_a_client_signs_in_from(
        hosted_mcp: FakeHostedMcp, hosted: ModuleType) -> None:
    client = hosted.HostedRecall("mv_wrong", hosted_mcp.serve())
    try:
        with pytest.raises(hosted.HostedError) as caught:
            client.recall("anything")
    finally:
        client.close()
    assert caught.value.status == 401
    ping = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
    with httpx.Client(base_url=hosted_mcp.MOCK_URL, transport=hosted_mcp.transport()) as raw:
        missing = raw.post("/mcp", json=ping)
        wrong = raw.post("/mcp", json=ping, headers={"Authorization": "Bearer mv_wrong"})
    metadata = 'resource_metadata="http://fake.invalid/.well-known/oauth-protected-resource"'
    assert missing.status_code == 401 and metadata in missing.headers["www-authenticate"]
    assert wrong.status_code == 401
    assert wrong.headers["www-authenticate"].endswith('error="invalid_token"')


def test_every_request_after_initialize_needs_a_session_the_fake_issued(
        hosted_mcp: FakeHostedMcp) -> None:
    ping = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
    auth = {"Authorization": f"Bearer {API_KEY}"}
    with httpx.Client(base_url=hosted_mcp.MOCK_URL, transport=hosted_mcp.transport(),
                      headers=auth) as raw:
        assert raw.post("/mcp", json=ping).status_code == 400
        assert raw.post("/mcp", json=ping,
                        headers={"mcp-session-id": "made-up"}).status_code == 404
        hello = raw.post("/mcp", json={"jsonrpc": "2.0", "id": 2, "method": "initialize",
                                       "params": {"protocolVersion": "2025-06-18",
                                                  "capabilities": {}}})
        session = {"mcp-session-id": hello.headers["mcp-session-id"]}
        notified = raw.post("/mcp", json={"jsonrpc": "2.0",
                                          "method": "notifications/initialized"},
                            headers=session)
        stream = raw.get("/mcp", headers=session)
    assert hello.status_code == 200 and hello.json()["result"]["serverInfo"]["name"]
    assert (notified.status_code, notified.content) == (202, b"")
    assert stream.status_code == 405


def test_a_client_holding_an_expired_session_shakes_hands_again(
        hosted_mcp: FakeHostedMcp, hosted: ModuleType) -> None:
    client = hosted.HostedRecall(API_KEY, hosted_mcp.serve())
    try:
        client.remember("user", "lives_in", "Lisbon")
        hosted_mcp.expire_sessions()
        assert "Lisbon" in client.recall("where does the user live")
    finally:
        client.close()
    assert _methods(hosted_mcp).count("initialize") == 2
    assert 404 in [r.status for r in hosted_mcp.requests]
    assert len(hosted_mcp.issued) == 2


def test_one_tool_can_fail_while_the_handshake_still_works(
        hosted_mcp: FakeHostedMcp, hosted: ModuleType) -> None:
    hosted_mcp.fail("tools/call memory_recall", 402, headers={"Retry-After": "3600"})
    client = hosted.HostedRecall(API_KEY, hosted_mcp.serve())
    try:
        with pytest.raises(hosted.HostedError) as caught:
            client.recall("anything")
    finally:
        client.close()
    assert (caught.value.status, caught.value.code, caught.value.retry_after) == (
        402, "quota_exhausted", 3600)
    assert _methods(hosted_mcp)[:2] == ["initialize", "notifications/initialized"]


def test_a_project_header_binds_the_project_and_is_reported_back(
        hosted_mcp: FakeHostedMcp, hosted: ModuleType,
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(hosted.PROJECT_ENV, "github.com/acme/app")
    client = hosted.HostedRecall(API_KEY, hosted_mcp.serve())
    try:
        client.remember("api", "depends_on", "postgres")
    finally:
        client.close()
    assert {r.header("memvara-project") for r in hosted_mcp.requests} \
        == {"github.com/acme/app"}
    in_project = hosted_mcp.memvara.scope(user="tester", project="github.com/acme/app")
    assert [c.object for c in in_project.get_all()] == ["postgres"]
    assert hosted_mcp.memvara.scope(user="tester", project="github.com/acme/web").get_all() \
        == []


def test_the_hooks_client_reads_a_reply_sent_as_an_event_stream(hosted: ModuleType) -> None:
    with FakeHostedMcp(sse=True) as fake:
        client = hosted.HostedRecall(API_KEY, fake.serve())
        try:
            client.remember("user", "lives_in", "Lisbon")
            assert "Lisbon" in client.recall("where does the user live")
        finally:
            client.close()


@pytest.mark.skipif(not _node_runs(), reason="node/npm missing or unloadable")
@pytest.mark.parametrize("sse", [False, True], ids=["json", "event-stream"])
def test_the_npm_bridge_carries_its_key_and_session_to_the_fake(
        sse: bool, tmp_path: pathlib.Path) -> None:
    lines = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": {"name": "memvara-adversarial-suite", "version": "0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    ]
    with FakeHostedMcp(sse=sse) as fake:
        done = subprocess.run(
            ["node", str(BRIDGE), "--server", fake.serve()],
            input="".join(json.dumps(line) + "\n" for line in lines), capture_output=True,
            text=True, encoding="utf-8", timeout=60, cwd=str(tmp_path),
            env=child_env(tmp_path, {"MEMVARA_API_KEY": API_KEY}))
    assert done.returncode == 0, done.stderr
    replies = [json.loads(line) for line in done.stdout.splitlines()]
    assert [reply["id"] for reply in replies] == [1, 2]
    assert "memory_recall" in {tool["name"] for tool in replies[1]["result"]["tools"]}
    assert {r.header("authorization") for r in fake.requests} == {f"Bearer {API_KEY}"}
    agents = {r.header("user-agent") or "" for r in fake.requests}
    assert len(agents) == 1 and agents.pop().startswith("memvara-npm/")
    assert [r.header("mcp-session-id") for r in fake.requests] == [
        None, fake.issued[0], fake.issued[0]]


def test_a_body_that_is_not_a_json_rpc_object_gets_the_json_rpc_error_envelope(
        hosted_mcp: FakeHostedMcp) -> None:
    """The cloud answers any body it cannot decode, including one nested too deeply for
    the decoder, as invalid JSON, and a body that decodes to something other than an
    object as an invalid request (rest/mcp.py, `_dispatch`)."""
    auth = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    with httpx.Client(base_url=hosted_mcp.MOCK_URL, transport=hosted_mcp.transport(),
                      headers=auth) as raw:
        broken = raw.post("/mcp", content=b"{not json")
        nested = raw.post("/mcp", content=b"[" * 100_000 + b"]" * 100_000)
        listed = raw.post("/mcp", content=b"[1, 2]")
    parse_error = {"jsonrpc": "2.0", "id": None,
                   "error": {"code": -32700, "message": "invalid JSON"}}
    assert (broken.status_code, broken.json()) == (400, parse_error)
    assert (nested.status_code, nested.json()) == (400, parse_error)
    assert (listed.status_code, listed.json()) == (400, {
        "jsonrpc": "2.0", "id": None,
        "error": {"code": -32600, "message": "expected a JSON-RPC object"}})
