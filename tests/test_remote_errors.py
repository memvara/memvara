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


# --- no envelope at all: what the status alone is allowed to say --------------
#
# Not every failure comes from the facade. An edge proxy rate-limits before the request
# reaches it and answers with an HTML page; a gateway does the same for an expired
# credential. Classifying those from the body alone made every one of them a
# `bad_request`, so a 429 arrived as a non-retryable `InvalidRequest` with its
# `Retry-After` discarded — while `HttpClient.request` documented itself as retrying on
# 429. The three tests below pin the two statuses that now decide their own code, and the
# fourth pins the fact that the list stops there.


def test_a_proxy_429_with_no_envelope_is_a_retryable_rate_limit():
    """The case that made this necessary: nothing in the body, everything in the status.

    All three assertions matter and each fails on its own. The class is what a caller
    branches on, `retryable` is what the transport's loop reads, and `retry_after` is the
    number that decides how long it waits — dropping any one of them turns a rate limit
    into a hard failure somewhere further down.
    """
    err = error_from_response(429, {}, "3600")
    assert isinstance(err, RateLimited)
    assert err.retryable is True
    assert err.retry_after == 3600.0


def test_a_gateway_401_with_no_envelope_is_an_auth_error():
    err = error_from_response(401, {}, None)
    assert isinstance(err, AuthError)
    assert err.retryable is False


def test_an_envelope_still_outranks_the_status_it_arrived_with():
    """The status is a fallback, not an override. A facade that answers 429 with
    `quota_exhausted` is saying waiting will not help, and that is the answer to keep."""
    err = error_from_response(429, _envelope("quota_exhausted"), "3600")
    assert isinstance(err, QuotaExhausted)
    assert err.retryable is False


@pytest.mark.parametrize("status", [403, 404, 409])
def test_an_ambiguous_status_is_not_guessed_at(status):
    """Deliberately short. A 403 is `forbidden_scope`, `forbidden_privilege`, `legal_hold`
    or `read_only` depending on what the facade meant, and a 404 read as `not_found` would
    make `get()` return None for a base url pointing at the wrong host — an empty answer
    where the caller needed an error. Unless the status settles the code, it does not
    supply one.
    """
    assert type(error_from_response(status, {}, None)) is InvalidRequest


def test_a_retry_after_that_is_not_a_number_is_dropped_rather_than_guessed():
    """`Retry-After` also has a date form, which this does not parse: the facade sends the
    delta form, and reading `Wed, 21 Oct 2026 07:28:00 GMT` as anything numeric would
    produce a wrong sleep instead of no sleep."""
    err = error_from_response(429, _envelope("rate_limited"),
                              "Wed, 21 Oct 2026 07:28:00 GMT")
    assert isinstance(err, RateLimited)
    assert err.retry_after is None


@pytest.mark.parametrize("status", [502, 504])
def test_a_gateway_failure_with_no_envelope_is_retryable(status):
    """A gateway saying it could not reach the application is the same class of failure as
    a connect error, which the transport already retries.

    Neither carries an envelope, so before this the absent `retryable` read as `False` and
    the call was raised on its first attempt — while the identical outage arriving as a
    dropped connection was retried.
    """
    err = error_from_response(status, {}, None)
    assert err.retryable is True
    assert err.status_code == status


def test_a_service_unavailable_is_not_guessed_at():
    """503 is the one an application sends about itself, for a maintenance window and for
    an exhausted quota alike — and retrying the second makes it worse. A deployment that
    means it is retryable can say so in the envelope."""
    assert error_from_response(503, {}, None).retryable is False


def test_an_envelope_that_says_not_retryable_wins_over_the_status():
    """The status is a fallback for silence, never an override. A server that says a 502
    is not worth retrying is answering about itself, and knows."""
    body = {"error": {"code": "internal", "message": "no", "retryable": False}}
    assert error_from_response(502, body, None).retryable is False


def test_an_envelope_that_says_retryable_wins_for_a_status_not_in_the_table():
    body = {"error": {"code": "internal", "message": "deadlock", "retryable": True}}
    assert error_from_response(503, body, None).retryable is True
