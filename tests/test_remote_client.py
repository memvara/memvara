"""The one HTTP layer: auth, retries, and the line between a write that can be repeated
and one that cannot.

Uses httpx's own MockTransport rather than monkeypatching the client, so the code under
test builds and drives a real `httpx.Client` and a change to how it is constructed cannot
pass by being mocked away.
"""
import httpx
import pytest

from memvara.remote.client import HttpClient
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
