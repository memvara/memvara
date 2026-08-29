"""The envelope memvara-cloud returns, turned into an exception a caller can branch on.

An unrecognised code must raise the base class rather than the nearest neighbour: guessing
that an unknown `foo_exhausted` is a `QuotaExhausted` would have a caller handle a failure
it has never seen as one it has.
"""
import pytest

from memvara.remote.errors import (
    AuthError, Conflict, InvalidRequest, LegalHold, NotFound, QuotaExhausted,
    RateLimited, ReadOnly, RemoteError, ScopeError, ServerError, error_from_response,
)


def _envelope(code, message="nope"):
    return {"error": {"code": code, "message": message}}


@pytest.mark.parametrize("status, code, expected", [
    (401, "unauthorized", AuthError),
    (403, "forbidden_scope", ScopeError),
    (403, "forbidden_privilege", ScopeError),
    (400, "bad_scope", ScopeError),
    (404, "not_found", NotFound),
    (409, "conflict", Conflict),
    (402, "quota_exhausted", QuotaExhausted),
    (429, "rate_limited", RateLimited),
    (409, "legal_hold", LegalHold),
    (403, "read_only", ReadOnly),
    (422, "invalid_request", InvalidRequest),
    (500, "internal", ServerError),
])
def test_each_code_maps_to_its_own_class(status, code, expected):
    err = error_from_response(status, _envelope(code), None)
    assert isinstance(err, expected)
    assert err.code == code
    assert err.status_code == status


def test_an_unknown_code_raises_the_base_class_and_not_a_guess():
    err = error_from_response(418, _envelope("teapot_unavailable"), None)
    assert type(err) is RemoteError
    assert err.code == "teapot_unavailable"


def test_rate_limited_carries_retry_after_as_a_number():
    err = error_from_response(429, _envelope("rate_limited"), "12")
    assert isinstance(err, RateLimited)
    assert err.retry_after == 12.0


def test_retryable_comes_from_the_envelope_when_the_server_states_it():
    body = {"error": {"code": "internal", "message": "deadlock", "retryable": True}}
    assert error_from_response(500, body, None).retryable is True


def test_a_body_that_is_not_an_envelope_still_produces_an_error():
    err = error_from_response(502, {}, None)
    assert isinstance(err, ServerError)
    assert err.status_code == 502
