"""One exception class per error code the `/v1` facade returns.

The envelope already carries the classification — a `code` naming what went wrong and,
where the server can tell, whether retrying could help. This module turns that into types
a caller can branch on and adds nothing of its own.

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
    """
    envelope = body.get("error") if isinstance(body, dict) else None
    envelope = envelope if isinstance(envelope, dict) else {}
    code = str(envelope.get("code") or ("internal" if status_code >= 500 else "bad_request"))
    message = str(envelope.get("message") or "no message")
    stated = envelope.get("retryable")
    cls = _BY_CODE.get(code, RemoteError)
    if cls is RateLimited:
        return RateLimited(status_code, code, message,
                           bool(stated) if stated is not None else True,
                           _retry_after(retry_after))
    return cls(status_code, code, message, bool(stated))
