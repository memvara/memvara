"""The one HTTP layer: auth, retries, and the line between a write that can be repeated
and one that cannot.

Uses httpx's own MockTransport rather than monkeypatching the client, so the code under
test builds and drives a real `httpx.Client` and a change to how it is constructed cannot
pass by being mocked away.

The section at the end of this file re-runs every case here against `AsyncHttpClient`.
`HttpClient.request` and `AsyncHttpClient.request` share their classification
(`_outcome`/`_delay` in `memvara/remote/client.py`), but the attempt loop around that
classification is written twice — once with a blocking call and `time.sleep`, once with
`await` and `await asyncio.sleep` — and a duplicated loop is exactly the kind of place two
implementations drift. Mutating either loop (moving the `Idempotency-Key` header inside
the loop so a retry mints a new one, say) should turn one of these tests red; a mutation
that only the sync tests would catch is this file failing at its one job. Written with
`asyncio.run` rather than `pytest-asyncio`, matching this repository's stated reason in
`tests/test_aio.py`: nothing here needs a fixture-scoped event loop, so the plugin would
be a larger dependency than the code it tests.
"""
import asyncio

import httpx
import pytest

from memvara.remote.client import AsyncHttpClient, HttpClient
from memvara.remote.errors import AuthError, RateLimited, ServerError


def _client(handler, **kw):
    """Build a real `HttpClient` and swap only its transport for a mock one.

    `__init__` is what sets the bearer header, the base url, and the timeout — replacing
    the whole `httpx.Client` the way an earlier version of this helper did would rebuild
    those from scratch in the fixture instead of exercising what `__init__` produced, and
    `test_the_bearer_token_is_sent_on_every_request` would pass even if `__init__` sent no
    auth header at all. Swapping only `_transport` keeps everything `__init__` set intact.
    """
    c = HttpClient("k", "https://example.test", sleep=lambda _: None, **kw)
    c._client._transport = httpx.MockTransport(handler)
    return c


def test_the_bearer_token_is_sent_on_every_request():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"ok": True})

    _client(handler).request("GET", "/v1/health")
    assert seen["auth"] == "Bearer k"


def test_a_failure_becomes_a_typed_error_rather_than_an_httpx_status_error():
    handler = lambda r: httpx.Response(401, json={"error": {"code": "unauthorized",
                                                            "message": "bad key"}})
    with pytest.raises(AuthError):
        _client(handler).request("GET", "/v1/stats")


def test_a_retryable_server_error_is_retried_and_then_succeeds():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(500, json={"error": {"code": "internal",
                                                       "message": "deadlock",
                                                       "retryable": True}})
        return httpx.Response(200, json={"ok": True})

    assert _client(handler).request("GET", "/v1/stats") == {"ok": True}
    assert len(calls) == 3


def test_a_non_retryable_server_error_is_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(500, json={"error": {"code": "internal", "message": "x"}})

    with pytest.raises(ServerError):
        _client(handler).request("GET", "/v1/stats")
    assert len(calls) == 1


def test_attempts_are_bounded_and_the_last_error_is_raised():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429, json={"error": {"code": "rate_limited", "message": "x"}},
                              headers={"Retry-After": "0"})

    with pytest.raises(RateLimited):
        _client(handler, attempts=3).request("GET", "/v1/stats")
    assert len(calls) == 3


def test_every_write_carries_an_idempotency_key():
    seen = {}

    def handler(request):
        seen["key"] = request.headers.get("idempotency-key")
        return httpx.Response(200, json={"ok": True})

    _client(handler).request("POST", "/v1/facts", json={"subject": "user"}, write=True)
    assert seen["key"]


def test_a_retried_write_repeats_the_same_idempotency_key():
    keys = []

    def handler(request):
        keys.append(request.headers.get("idempotency-key"))
        if len(keys) < 2:
            return httpx.Response(429, json={"error": {"code": "rate_limited", "message": "x"}},
                                  headers={"Retry-After": "0"})
        return httpx.Response(200, json={"ok": True})

    _client(handler).request("POST", "/v1/facts", json={}, write=True)
    assert len(keys) == 2 and keys[0] == keys[1]


def test_two_separate_writes_do_not_share_a_key():
    keys = []

    def handler(request):
        keys.append(request.headers.get("idempotency-key"))
        return httpx.Response(200, json={"ok": True})

    c = _client(handler)
    c.request("POST", "/v1/facts", json={}, write=True)
    c.request("POST", "/v1/facts", json={}, write=True)
    assert keys[0] != keys[1]


def test_a_read_timeout_on_a_read_is_retried():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 2:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json={"ok": True})

    assert _client(handler).request("GET", "/v1/stats") == {"ok": True}
    assert len(calls) == 2


def test_a_connect_error_on_a_write_is_retried_because_nothing_was_sent():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 2:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json={"ok": True})

    assert _client(handler).request("POST", "/v1/facts", json={}, write=True) == {"ok": True}
    assert len(calls) == 2


def test_an_unset_scope_parameter_is_omitted_rather_than_sent_empty():
    seen = {}

    def handler(request):
        seen["query"] = request.url.params
        return httpx.Response(200, json={"ok": True})

    _client(handler).request("GET", "/v1/facts", params={"user": "alice", "agent": None})
    assert seen["query"]["user"] == "alice"
    assert "agent" not in seen["query"]


def test_an_empty_response_body_is_none_rather_than_a_decode_error():
    def handler(request):
        return httpx.Response(204)

    assert _client(handler).request("DELETE", "/v1/facts/1") is None


# -- the same rules, on AsyncHttpClient --------------------------------------------
#
# Every test above pins a rule about what is retryable, what carries an idempotency key,
# and how a response decodes. `AsyncHttpClient` shares the classification that answers
# those questions (`_outcome`/`_delay`), but the loop that calls them is its own
# `async def` written separately from `HttpClient.request`'s — so each rule needs its own
# proof on this side too, or a hand-written loop could drift with nothing to catch it.


async def _noop_sleep(_seconds: float) -> None:
    return None


def _aclient(handler, **kw):
    c = AsyncHttpClient("k", "https://example.test", sleep=_noop_sleep, **kw)
    c._client._transport = httpx.MockTransport(handler)
    return c


def run(coro):
    return asyncio.run(coro)


def test_async_the_bearer_token_is_sent_on_every_request():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"ok": True})

    run(_aclient(handler).request("GET", "/v1/health"))
    assert seen["auth"] == "Bearer k"


def test_async_a_failure_becomes_a_typed_error_rather_than_an_httpx_status_error():
    handler = lambda r: httpx.Response(401, json={"error": {"code": "unauthorized",
                                                            "message": "bad key"}})
    with pytest.raises(AuthError):
        run(_aclient(handler).request("GET", "/v1/stats"))


def test_async_a_retryable_server_error_is_retried_and_then_succeeds():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(500, json={"error": {"code": "internal",
                                                       "message": "deadlock",
                                                       "retryable": True}})
        return httpx.Response(200, json={"ok": True})

    assert run(_aclient(handler).request("GET", "/v1/stats")) == {"ok": True}
    assert len(calls) == 3


def test_async_a_non_retryable_server_error_is_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(500, json={"error": {"code": "internal", "message": "x"}})

    with pytest.raises(ServerError):
        run(_aclient(handler).request("GET", "/v1/stats"))
    assert len(calls) == 1


def test_async_attempts_are_bounded_and_the_last_error_is_raised():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429, json={"error": {"code": "rate_limited", "message": "x"}},
                              headers={"Retry-After": "0"})

    with pytest.raises(RateLimited):
        run(_aclient(handler, attempts=3).request("GET", "/v1/stats"))
    assert len(calls) == 3


def test_async_every_write_carries_an_idempotency_key():
    seen = {}

    def handler(request):
        seen["key"] = request.headers.get("idempotency-key")
        return httpx.Response(200, json={"ok": True})

    run(_aclient(handler).request("POST", "/v1/facts", json={"subject": "user"},
                                  write=True))
    assert seen["key"]


def test_async_a_retried_write_repeats_the_same_idempotency_key():
    keys = []

    def handler(request):
        keys.append(request.headers.get("idempotency-key"))
        if len(keys) < 2:
            return httpx.Response(429, json={"error": {"code": "rate_limited", "message": "x"}},
                                  headers={"Retry-After": "0"})
        return httpx.Response(200, json={"ok": True})

    run(_aclient(handler).request("POST", "/v1/facts", json={}, write=True))
    assert len(keys) == 2 and keys[0] == keys[1]


def test_async_two_separate_writes_do_not_share_a_key():
    keys = []

    def handler(request):
        keys.append(request.headers.get("idempotency-key"))
        return httpx.Response(200, json={"ok": True})

    c = _aclient(handler)

    async def main():
        await c.request("POST", "/v1/facts", json={}, write=True)
        await c.request("POST", "/v1/facts", json={}, write=True)

    run(main())
    assert keys[0] != keys[1]


def test_async_a_read_timeout_on_a_read_is_retried():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 2:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json={"ok": True})

    assert run(_aclient(handler).request("GET", "/v1/stats")) == {"ok": True}
    assert len(calls) == 2


def test_async_a_connect_error_on_a_write_is_retried_because_nothing_was_sent():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 2:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json={"ok": True})

    assert run(_aclient(handler).request("POST", "/v1/facts", json={},
                                         write=True)) == {"ok": True}
    assert len(calls) == 2


def test_async_an_unset_scope_parameter_is_omitted_rather_than_sent_empty():
    seen = {}

    def handler(request):
        seen["query"] = request.url.params
        return httpx.Response(200, json={"ok": True})

    run(_aclient(handler).request("GET", "/v1/facts",
                                  params={"user": "alice", "agent": None}))
    assert seen["query"]["user"] == "alice"
    assert "agent" not in seen["query"]


def test_async_an_empty_response_body_is_none_rather_than_a_decode_error():
    def handler(request):
        return httpx.Response(204)

    assert run(_aclient(handler).request("DELETE", "/v1/facts/1")) is None
