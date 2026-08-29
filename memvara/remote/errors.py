"""One exception class per error code the `/v1` facade returns.

The envelope already carries the classification — a `code` naming what went wrong and,
where the server can tell, whether retrying could help. This module turns that into types
a caller can branch on and adds nothing of its own.

**A response with no envelope is classified from its status, for the two statuses that
say enough on their own.** Not every failure comes from the facade: an edge proxy answers
429 with an HTML page, and a gateway answers 401 the same way. `_BY_STATUS` below is
deliberately two entries long, because a status that could mean several codes is one this
module must not guess at.

**An unrecognised code raises `RemoteError` itself.** Coercing it into the nearest known
class would have a caller handle a failure they have never seen as one they have, and the
whole reason for a code is that the server is the thing that knows.
"""
from __future__ import annotations

from typing import Any


class RemoteError(Exception):
    """A `/v1` request that did not succeed."""

    def __init__(self, status_code: int, code: str, message: str,
                 retryable: bool = False) -> None:
        super().__init__(f"{status_code} {code}: {message}")
        self.status_code = status_code
        self.code = code
        self.message = message
        #: Whether the *server* said retrying could help. Never inferred here.
        self.retryable = retryable


class AuthError(RemoteError):
    """The credential is missing, malformed, or not accepted."""


class ScopeError(RemoteError):
    """The credential cannot address the tenant, user, agent or session asked for."""


class NotFound(RemoteError):
    """No such resource — which the facade also returns for one belonging elsewhere."""


class Conflict(RemoteError):
    """The request collides with the stored state."""


class QuotaExhausted(RemoteError):
    """A period allowance is spent. Not retryable; waiting does not help."""


class RateLimited(RemoteError):
    """Too much work too fast. Retryable, and `retry_after` says when."""

    def __init__(self, status_code: int, code: str, message: str,
                 retryable: bool = True, retry_after: float | None = None) -> None:
        super().__init__(status_code, code, message, retryable)
        self.retry_after = retry_after


class LegalHold(RemoteError):
    """A hold forbids this write."""


class ReadOnly(RemoteError):
    """The deployment is not accepting writes."""


class InvalidRequest(RemoteError):
    """The request body did not validate."""


class ServerError(RemoteError):
    """The deployment failed. Retryable only when it says so."""


#: status -> the code the facade would have sent, for a response that carried no envelope.
#: Only statuses whose meaning the status line already fixes are here, because a guess is
#: worse than the generic fallback: a 403 could be `forbidden_scope`, `forbidden_privilege`,
#: `legal_hold` or `read_only`, and a 404 read as `not_found` would turn a misdirected
#: `base_url` into `get()` quietly returning None instead of raising.
#:
#: 429 is the one that had to be here. An edge proxy rate-limits before the request ever
#: reaches the facade, so its response carries an HTML page and no envelope — and without
#: this the generic fallback made it an `InvalidRequest(retryable=False)`, which the
#: transport does not retry and which discards `Retry-After`.
_BY_STATUS: dict[int, str] = {
    401: "unauthorized",
    429: "rate_limited",
}

#: Statuses that are retryable when the envelope does not say. A separate table from
#: `_BY_STATUS` because these are two different questions: that one asks *what went
#: wrong*, this one asks *whether trying again could help*, and a status can be
#: unambiguous about the second while saying little about the first.
#:
#: A gateway answering 502 or 504 is reporting that it could not get an answer from the
#: application — the request either never arrived or produced nothing, which puts it in
#: the same class as the connect-phase failures the transport already retries. Both carry
#: no envelope, so without this they fall to `retryable` absent, read as `False`, and are
#: raised on the first attempt.
#:
#: **503 is deliberately absent.** It is the one an application sends about itself, and it
#: is used for a maintenance window and for an exhausted quota alike — the second of which
#: retrying makes worse. A facade that means it is retryable can say so in the envelope,
#: and this module does not guess where the status genuinely carries more than one meaning.
#:
#: An explicit `retryable` in the envelope always wins, in either direction: a server that
#: says a 502 is not worth retrying is answering about itself, and knows.
_RETRYABLE_WHEN_UNSTATED: frozenset[int] = frozenset({502, 504})

#: code -> class. Codes absent here raise `RemoteError`, deliberately.
_BY_CODE: dict[str, type[RemoteError]] = {
    "unauthorized": AuthError,
    "bad_scope": ScopeError,
    "forbidden_scope": ScopeError,
    "forbidden_privilege": ScopeError,
    "not_found": NotFound,
    "conflict": Conflict,
    "quota_exhausted": QuotaExhausted,
    "rate_limited": RateLimited,
    "legal_hold": LegalHold,
    "read_only": ReadOnly,
    "invalid_request": InvalidRequest,
    "bad_request": InvalidRequest,
    "method_not_allowed": InvalidRequest,
    "internal": ServerError,
}


def _retry_after(value: str | None) -> float | None:
    """`Retry-After` as seconds, or None. A date-form header is not parsed: the facade
    sends the delta form, and guessing at the other would produce a wrong sleep."""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def error_from_response(status_code: int, body: Any,
                        retry_after: str | None) -> RemoteError:
    """Build the exception for one failed response.

    `body` is whatever decoded, including `{}` for a response that carried no envelope —
    a proxy 502, say. That case must still produce an error rather than a KeyError, which
    is why nothing here indexes into the payload.

    **With no envelope the status decides the code, where the status is unambiguous.** The
    envelope is still preferred whenever there is one; `_BY_STATUS` above says which
    statuses stand on their own and why the list is short. Falling straight through to
    `bad_request` classified an edge proxy's 429 as a non-retryable `InvalidRequest` and
    dropped its `Retry-After` — the failure the client's retry rule exists for.

    >>> err = error_from_response(429, {}, "12")
    >>> type(err).__name__, err.retryable, err.retry_after
    ('RateLimited', True, 12.0)
    """
    envelope = body.get("error") if isinstance(body, dict) else None
    envelope = envelope if isinstance(envelope, dict) else {}
    default = _BY_STATUS.get(status_code,
                             "internal" if status_code >= 500 else "bad_request")
    code = str(envelope.get("code") or default)
    message = str(envelope.get("message") or "no message")
    stated = envelope.get("retryable")
    cls = _BY_CODE.get(code, RemoteError)
    if cls is RateLimited:
        return RateLimited(status_code, code, message,
                           bool(stated) if stated is not None else True,
                           _retry_after(retry_after))
    retryable = (bool(stated) if stated is not None
                 else status_code in _RETRYABLE_WHEN_UNSTATED)
    return cls(status_code, code, message, retryable)
