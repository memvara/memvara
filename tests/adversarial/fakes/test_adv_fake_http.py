"""The machinery every HTTP fake shares: the request log, the faults, and the three ways
to reach a fake."""

from __future__ import annotations

import asyncio
import time
from typing import Iterator

import httpx
import pytest

from harness.fakes._http import HttpFake, Reply, Request, json_reply


class Echo(HttpFake):
    """The smallest fake: it answers `GET /echo` and `POST /echo` with what it was sent,
    and any other path with a 404."""

    ROUTES = ("GET /echo", "POST /echo")

    def route(self, request: Request) -> str | None:
        name = f"{request.method} {request.path}"
        return name if name in self.ROUTES else None

    def respond(self, request: Request) -> Reply:
        if request.route is None:
            return json_reply(404, {"error": "no such route"})
        return json_reply(200, {"method": request.method, "query": request.query,
                                "body": request.json()})


@pytest.fixture
def echo() -> Iterator[Echo]:
    with Echo() as fake:
        yield fake


def _client(fake: HttpFake, timeout: float = 5.0) -> httpx.Client:
    return httpx.Client(base_url=fake.MOCK_URL, transport=fake.transport(), timeout=timeout)


def _wait_for_answer(request: Request) -> None:
    deadline = time.monotonic() + 5
    while request.status is None and time.monotonic() < deadline:
        time.sleep(0.02)


def test_every_request_is_recorded_with_what_it_carried(echo: Echo) -> None:
    with _client(echo) as client:
        answer = client.post("/echo", params={"user": "alice", "tag": ["a", "b"]},
                             json={"k": 1}, headers={"X-Probe": "yes"})
    assert answer.json() == {"method": "POST", "body": {"k": 1},
                             "query": {"user": ["alice"], "tag": ["a", "b"]}}
    (seen,) = echo.requests
    assert (seen.method, seen.path, seen.route, seen.status) == ("POST", "/echo",
                                                                  "POST /echo", 200)
    assert seen.param("tag") == "a" and seen.param("absent") is None
    assert seen.header("X-PROBE") == "yes" and seen.json() == {"k": 1}


def test_a_request_the_fake_has_no_route_for_reaches_it_without_a_route(echo: Echo) -> None:
    with _client(echo) as client:
        assert client.get("/nowhere").status_code == 404
    assert echo.requests[0].route is None


def test_a_fault_on_a_route_the_fake_does_not_have_is_refused(echo: Echo) -> None:
    with pytest.raises(ValueError, match="has no route 'GET /ecko'"):
        echo.fail("GET /ecko", 500)
    with pytest.raises(ValueError, match="has no route 'GET /ecko'"):
        echo.delay("GET /ecko", 1)
    with pytest.raises(ValueError, match="has no route 'GET /ecko'"):
        echo.hang("GET /ecko")


def test_an_injected_status_answers_instead_of_the_fake_for_as_long_as_asked(
        echo: Echo) -> None:
    echo.fail("GET /echo", 503, times=2)
    echo.fail("POST /echo", 502, body="<html>Bad gateway</html>",
              headers={"Retry-After": "7"})
    with _client(echo) as client:
        assert [client.get("/echo").status_code for _ in range(3)] == [503, 503, 200]
        proxied = client.post("/echo", json={})
        assert client.post("/echo", json={}).status_code == 502
    assert proxied.text == "<html>Bad gateway</html>"
    assert proxied.headers["retry-after"] == "7"
    echo.clear_faults()
    with _client(echo) as client:
        assert client.post("/echo", json={}).status_code == 200


def test_a_delay_shorter_than_the_timeout_is_waited_out(echo: Echo) -> None:
    echo.delay("GET /echo", 0.2)
    started = time.monotonic()
    with _client(echo, timeout=5) as client:
        assert client.get("/echo").status_code == 200
    assert time.monotonic() - started >= 0.2


def test_over_a_mock_transport_a_hang_or_a_long_delay_ends_at_the_client_s_timeout(
        echo: Echo) -> None:
    echo.hang("GET /echo", times=1)
    echo.delay("GET /echo", 30, times=1)
    with _client(echo, timeout=0.2) as client:
        for _ in range(2):
            started = time.monotonic()
            with pytest.raises(httpx.ReadTimeout):
                client.get("/echo")
            assert 0.2 <= time.monotonic() - started < 5
        assert client.get("/echo").status_code == 200
    assert [r.status for r in echo.requests] == [None, None, 200]


def test_the_async_transport_waits_without_blocking_the_event_loop(echo: Echo) -> None:
    echo.delay("GET /echo", 0.3, times=1)
    echo.hang("POST /echo", times=1)

    async def main() -> tuple[int, list[float]]:
        started = time.monotonic()
        ticks: list[float] = []

        async def tick() -> None:
            for _ in range(3):
                await asyncio.sleep(0.05)
                ticks.append(time.monotonic() - started)

        async with httpx.AsyncClient(base_url=echo.MOCK_URL, timeout=0.2,
                                     transport=echo.async_transport()) as client:
            waited, _ = await asyncio.gather(client.get("/echo", timeout=5), tick())
            with pytest.raises(httpx.ReadTimeout):
                await client.post("/echo", json={})
        return waited.status_code, ticks

    status, ticks = asyncio.run(main())
    assert status == 200
    assert len(ticks) == 3 and ticks[0] < 0.25, "the delay blocked the event loop"


def test_over_a_socket_a_hang_ends_at_the_client_s_timeout_and_close_releases_it(
        echo: Echo) -> None:
    url = echo.serve()
    assert echo.serve() == url
    echo.hang("GET /echo", times=1)
    with httpx.Client(base_url=url, timeout=0.3) as client:
        with pytest.raises(httpx.ReadTimeout):
            client.get("/echo")
        assert client.post("/echo", json={"k": 2}).json()["body"] == {"k": 2}
    started = time.monotonic()
    echo.close()
    assert time.monotonic() - started < 5
    assert [r.status for r in echo.requests] == [None, 200]


def test_over_a_socket_a_request_the_client_gave_up_on_is_carried_out_late(
        echo: Echo) -> None:
    echo.delay("POST /echo", 0.5, times=1)
    with httpx.Client(base_url=echo.serve(), timeout=0.2) as client:
        with pytest.raises(httpx.ReadTimeout):
            client.post("/echo", json={"late": True})
    _wait_for_answer(echo.requests[0])
    assert echo.requests[0].status == 200
