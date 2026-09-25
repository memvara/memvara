"""Faults injected into FakeV1 reach the remote client as the client handles them, over a
mock transport and over a real socket."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

import httpx
import pytest

from harness.fakes import fake_v1 as v1
from harness.fakes._http import Request
from harness.fakes.fake_v1 import FakeV1
from harness.stdio import McpProcess
from memvara.core import ScopedMemvara
from memvara.remote.api import RemoteMemvara
from memvara.remote.client import DEFAULT_ATTEMPTS
from memvara.remote.errors import (AuthError, InvalidRequest, RateLimited, ReadOnly,
                                   RemoteError, ScopeError)

Start = Callable[..., McpProcess]


def _sent(fake: FakeV1, route: str) -> list[Request]:
    return [r for r in fake.requests if r.route == route]


def test_a_retryable_status_is_retried_with_the_same_key_and_the_retry_lands(
        fake_v1: FakeV1) -> None:
    fake_v1.fail("POST /v1/facts", 503, times=1, body={"error": {
        "code": "unavailable", "message": "a deadlock lost", "retryable": True}})
    receipt = fake_v1.remote(user="alice").remember("user", "lives_in", "Lisbon")
    assert [c.object for c in receipt.added] == ["Lisbon"]
    attempts = _sent(fake_v1, "POST /v1/facts")
    assert [r.status for r in attempts] == [503, 200]
    assert attempts[0].header("idempotency-key") == attempts[1].header("idempotency-key")
    assert len(fake_v1.memvara.scope(user="alice").get_all()) == 1


def test_a_status_the_server_does_not_call_retryable_is_raised_at_once(
        fake_v1: FakeV1) -> None:
    fake_v1.fail("GET /v1/memories/{id}", 403)
    with pytest.raises(ScopeError):
        fake_v1.remote(user="alice").get("cl_00000000000000000000")
    assert [r.status for r in fake_v1.requests] == [403]


def test_a_rate_limit_is_waited_out_for_as_long_as_the_server_asks(fake_v1: FakeV1) -> None:
    fake_v1.fail("POST /v1/search", 429, headers={"Retry-After": "0"}, times=2)
    assert fake_v1.remote(user="alice").search("anything") == []
    assert [r.status for r in _sent(fake_v1, "POST /v1/search")] == [429, 429, 200]


def test_a_rate_limit_longer_than_the_client_will_wait_is_raised_with_the_wait(
        fake_v1: FakeV1) -> None:
    fake_v1.fail("POST /v1/search", 429, headers={"Retry-After": "3600"})
    with pytest.raises(RateLimited) as caught:
        fake_v1.remote(user="alice").search("anything")
    assert caught.value.retry_after == 3600
    assert len(fake_v1.requests) == 1


def test_a_proxy_page_with_no_envelope_is_classified_by_its_status(fake_v1: FakeV1) -> None:
    fake_v1.fail("GET /v1/stats", 502, body="<html>Bad gateway</html>")
    with pytest.raises(RemoteError) as caught:
        fake_v1.remote().stats()
    assert caught.value.status_code == 502 and caught.value.retryable
    assert len(fake_v1.requests) == DEFAULT_ATTEMPTS


def test_a_delay_within_the_client_s_timeout_is_waited_out(fake_v1: FakeV1) -> None:
    fake_v1.delay("GET /v1/stats", 0.3)
    started = time.monotonic()
    assert fake_v1.remote(timeout=5).stats()["claims"] == 0
    assert time.monotonic() - started >= 0.3


def test_a_hang_is_cut_off_by_the_client_s_timeout_on_every_attempt(fake_v1: FakeV1) -> None:
    fake_v1.hang("GET /v1/stats")
    started = time.monotonic()
    with pytest.raises(RemoteError) as caught:
        fake_v1.remote(timeout=0.2).stats()
    assert caught.value.code == "transport"
    assert [r.status for r in fake_v1.requests] == [None] * DEFAULT_ATTEMPTS
    assert time.monotonic() - started >= 0.2 * DEFAULT_ATTEMPTS


def test_the_async_client_times_out_on_a_hang_too(fake_v1: FakeV1) -> None:
    import asyncio

    fake_v1.hang("GET /v1/stats", times=1)

    async def main() -> dict[str, int]:
        client = fake_v1.aremote(timeout=0.2)
        try:
            return await client.stats()
        finally:
            await client.aclose()

    assert asyncio.run(main())["claims"] == 0
    assert [r.status for r in fake_v1.requests] == [None, 200]


def test_over_a_socket_the_client_s_own_timeout_cuts_off_a_hang(fake_v1: FakeV1) -> None:
    fake_v1.hang("GET /v1/stats", times=1)
    with RemoteMemvara(api_key=fake_v1.api_key, base_url=fake_v1.serve(),
                       timeout=0.3) as remote:
        assert remote.stats()["claims"] == 0
    assert [r.status for r in fake_v1.requests] == [None, 200]


def test_a_write_retried_after_its_first_attempt_timed_out_lands_once(
        fake_v1: FakeV1) -> None:
    """Over a socket the first attempt is carried out late, after the client has already
    retried. The idempotency key is what stops it writing a second time."""
    fake_v1.delay("POST /v1/facts", 0.8, times=1)
    with RemoteMemvara(api_key=fake_v1.api_key, base_url=fake_v1.serve(), timeout=0.3,
                       user="alice") as remote:
        assert [c.object for c in remote.remember("user", "lives_in", "Lisbon").added] \
            == ["Lisbon"]
    deadline = time.monotonic() + 5
    while (any(r.status is None for r in _sent(fake_v1, "POST /v1/facts"))
           and time.monotonic() < deadline):
        time.sleep(0.02)
    attempts = _sent(fake_v1, "POST /v1/facts")
    assert len(attempts) == 2
    assert sorted(r.replayed for r in attempts) == [False, True]
    # Written twice, the same value would be one claim observed twice, so the count of
    # observations is what shows the write landed once.
    assert [c.observation_count for c in fake_v1.memvara.scope(user="alice").get_all()] \
        == [1]


def test_a_write_waits_only_for_an_earlier_write_with_the_same_key_method_and_path(
        fake_v1: FakeV1) -> None:
    """The fake stores a write's reply under its idempotency key, method and path
    together. A write that shares only the key with one still being carried out is a
    different write, so it must go ahead without waiting for it."""
    entered, release = threading.Event(), threading.Event()
    remember = fake_v1._handlers["POST /v1/facts"]

    def held(view: ScopedMemvara, request: Request) -> Any:
        entered.set()
        release.wait(10)
        return remember(view, request)

    fake_v1._handlers["POST /v1/facts"] = held
    statuses: dict[str, int] = {}

    def post(path: str, body: dict[str, Any]) -> None:
        with httpx.Client(base_url=fake_v1.MOCK_URL, transport=fake_v1.transport(),
                          timeout=30, headers={"Authorization": f"Bearer {fake_v1.api_key}",
                                               "Idempotency-Key": "the-same-key"}) as client:
            statuses[path] = client.post(path, json=body,
                                         params={"user": "alice"}).status_code

    first = threading.Thread(target=post, daemon=True, args=(
        "/v1/facts", {"subject": "user", "predicate": "lives_in", "object": "Lisbon"}))
    second = threading.Thread(target=post, daemon=True, args=(
        "/v1/memories", {"messages": [{"role": "user", "content": "I like green tea"}]}))
    try:
        first.start()
        assert entered.wait(5)
        second.start()
        second.join(timeout=3)
        assert not second.is_alive(), "the second write waited for the first"
    finally:
        release.set()
        first.join(timeout=5)
    assert statuses == {"/v1/facts": 200, "/v1/memories": 200}
    assert [r.replayed for r in fake_v1.requests] == [False, False]


def test_a_wrong_key_is_refused_and_not_retried(fake_v1: FakeV1) -> None:
    with pytest.raises(AuthError):
        fake_v1.remote(api_key="mv_wrong").stats()
    assert [r.status for r in fake_v1.requests] == [401]


def test_each_user_reads_only_its_own_memories(fake_v1: FakeV1) -> None:
    fake_v1.remote(user="alice").remember("user", "lives_in", "Lisbon")
    assert fake_v1.remote(user="bob").get_all() == []
    assert [c.object for c in fake_v1.remote(user="alice").get_all()] == ["Lisbon"]


def test_every_answer_in_a_project_names_the_project_it_ran_at(fake_v1: FakeV1) -> None:
    """The cloud sets `Memvara-Project-Applied` on the response every route shares
    (`deps._context`), so the header is on the consolidation's 202 as well as on an
    ordinary 200. A request that names no project gets no header."""
    auth = {"Authorization": f"Bearer {fake_v1.api_key}"}
    project = {**auth, "Memvara-Project": "github.com/acme/app"}
    with httpx.Client(base_url=fake_v1.MOCK_URL, transport=fake_v1.transport()) as client:
        read = client.get("/v1/stats", headers=project)
        accepted = client.post("/v1/maintenance/consolidate", headers=project)
        plain = client.post("/v1/maintenance/consolidate", headers=auth)
    assert (read.status_code, read.headers.get("memvara-project-applied")) \
        == (200, "github.com/acme/app")
    assert (accepted.status_code, accepted.headers.get("memvara-project-applied")) \
        == (202, "github.com/acme/app")
    assert accepted.headers["location"] == f"/v1/jobs/{accepted.json()['id']}"
    assert (plain.status_code, plain.headers.get("memvara-project-applied")) == (202, None)


def test_an_agent_named_without_a_user_is_refused(fake_v1: FakeV1) -> None:
    with pytest.raises(ScopeError):
        fake_v1.remote(agent="a1").get_all()


def test_a_read_only_deployment_says_so_and_refuses_every_write() -> None:
    with FakeV1(read_only=True) as fake:
        remote = fake.remote(user="alice")
        assert remote.service()["read_only"] is True
        with pytest.raises(ReadOnly):
            remote.remember("user", "lives_in", "Lisbon")


def test_a_field_the_route_does_not_take_is_refused_as_the_cloud_refuses_it(
        fake_v1: FakeV1, monkeypatch: pytest.MonkeyPatch) -> None:
    """The client counts on this: a read that sends `query_rewrite` to a deployment that
    predates the field gets a 422, and is sent again without it."""
    with httpx.Client(base_url=fake_v1.MOCK_URL, transport=fake_v1.transport(),
                      headers={"Authorization": f"Bearer {fake_v1.api_key}"}) as client:
        refused = client.post("/v1/facts", json={"predicate": "lives_in", "object": "Lisbon",
                                                 "no_such_field": 1})
        missing = client.get("/v1/no-such-route")
    assert refused.status_code == 422
    assert refused.json()["error"]["code"] == "invalid_request"
    assert missing.status_code == 404
    with pytest.raises(InvalidRequest):
        fake_v1.remote().search("")

    # A deployment from before `query_rewrite`: neither read's request model has it.
    for fields in ("_SEARCH", "_RECALL"):
        monkeypatch.setattr(v1, fields, getattr(v1, fields) - {"query_rewrite"})
    first = len(fake_v1.requests)
    remote = fake_v1.remote(user="alice")
    assert remote.search("where does the user live", query_rewrite=False) == []
    assert isinstance(remote.recall("where does the user live", query_rewrite=False), str)
    assert [(r.route, r.status, "query_rewrite" in r.json())
            for r in fake_v1.requests[first:]] == [
        ("POST /v1/search", 422, True), ("POST /v1/search", 200, False),
        ("POST /v1/recall", 422, True), ("POST /v1/recall", 200, False)]


def test_a_cloud_mode_server_process_reaches_the_fake_over_a_real_url(
        fake_v1: FakeV1, mcp: Start) -> None:
    server = mcp(env={"MEMVARA_MODE": "cloud", "MEMVARA_API_KEY": fake_v1.api_key,
                      "MEMVARA_SERVER_URL": fake_v1.serve()})
    server.initialize()
    stored = server.call("memory_remember", subject="user", predicate="lives_in",
                         object="Lisbon")
    assert not stored.is_error, stored.text
    assert "Lisbon" in server.call("memory_recall", query="where does the user live").text
    assert {"POST /v1/facts", "POST /v1/recall"} <= {r.route for r in fake_v1.requests}
    assert all(r.header("authorization") == f"Bearer {fake_v1.api_key}"
               and r.param("user") == "tester"
               for r in fake_v1.requests if r.route != "GET /v1/health")
    assert server.close() == 0
