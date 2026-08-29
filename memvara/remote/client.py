"""One HTTP layer for every remote call, so retry and error behaviour cannot differ
between two clients of the same API.

**`httpx` is imported inside `__init__` and inside `request`, never at module level.**
`import memvara` must work with numpy alone — the core promise in `docs/INTERNALS.md`,
invariant 5 — and a module-level import here would break it the moment anything imports
this file, even without constructing a client.

**This client does not resolve credentials.** `memvara.remote.creds.resolve` turns an
explicit key, an environment variable, and a credentials file into one `(key, url)` pair,
but that happens once, in `RemoteMemvara.__init__` (later in this plan) — a transport
that resolved its own credentials could not be constructed with an explicit key for a
test, and `memvara/store/remote.py` builds one from a key it already holds. `HttpClient`
only ever takes an already-resolved key and base url.

**Why writes are retried at all.** A write that fails *after* the request was sent may
have committed, so repeating it can write the fact twice. The facade accepts an
`Idempotency-Key`, and this client sends one on every write and repeats it verbatim across
retries of that same write, which is what makes the repeat safe. A new write gets a new
key.

**What the idempotency guarantee actually covers.** The server's idempotency store lives
in the process that first saw the key, not in a shared location every worker can read.
A retry that a load balancer routes back to that same worker is deduplicated there and
does not re-execute. A retry routed to a *different* worker finds no record of the first
attempt and re-executes it — the server's own `deploy/README.md` documents this. The
guarantee therefore holds unconditionally only for a single-worker deployment; behind a
load balancer with more than one worker, a retried write is deduplicated when luck (or
session affinity) routes it back to the worker that saw the original attempt, and
duplicates it otherwise. This client does not — cannot, from here — make that stronger.
"""
from __future__ import annotations

import random
import time
import uuid
from typing import Any, Callable

from .errors import RemoteError, error_from_response

#: Attempts, total, per call. Three is two retries — enough for a redeploy or a deadlock,
#: short enough that a caller does not wait a minute to be told something is down.
DEFAULT_ATTEMPTS = 3
#: Seconds. Matches `store/remote.py`, so the two clients time out alike.
DEFAULT_TIMEOUT = 30.0
#: Base for exponential backoff, in seconds, before jitter.
_BACKOFF = 0.25


def install_hint() -> str:
    """The message to show when `httpx` is not installed."""
    return ('httpx is required to talk to a hosted deployment. Install it with '
            'pip install "memvara[cloud]".')


class HttpClient:
    """A `/v1` transport bound to one deployment and one already-resolved credential."""

    def __init__(self, api_key: str, base_url: str, *, timeout: float = DEFAULT_TIMEOUT,
                 attempts: int = DEFAULT_ATTEMPTS,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(install_hint()) from exc
        self._attempts = max(1, attempts)
        self._sleep = sleep
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def close(self) -> None:
        """Release the underlying connection pool."""
        self._client.close()

    def request(self, method: str, path: str, *, params: dict[str, Any] | None = None,
                json: Any = None, write: bool = False) -> Any:
        """One call, decoded, retried where retrying is safe.

        Retries on a server-classified `retryable` error, on 429, and on a connect-phase
        failure (`httpx.ConnectError`, `httpx.ConnectTimeout`) — none of these ever
        reached the server, so repeating them is safe regardless of method. A
        `httpx.ReadTimeout` did reach the server and is retried too, but only because
        `write=True` attaches an `Idempotency-Key` that is held constant across this
        call's own retries; callers pass `write=True` for every route that mutates.
        A non-retryable error, or the last error once attempts are exhausted, is raised.
        """
        import httpx

        headers = {"Idempotency-Key": uuid.uuid4().hex} if write else None
        last: Exception | None = None
        for attempt in range(self._attempts):
            try:
                response = self._client.request(
                    method, path,
                    params=_drop_none(params), json=json, headers=headers)
            except httpx.TransportError as exc:
                last = exc
            else:
                if response.status_code < 400:
                    return response.json() if response.content else None
                last = error_from_response(
                    response.status_code, _body(response),
                    response.headers.get("Retry-After"))
                if not last.retryable:
                    raise last
            if attempt + 1 < self._attempts:
                self._sleep(self._delay(attempt, last))
        raise last if isinstance(last, RemoteError) else _wrapped(last)

    def _delay(self, attempt: int, last: Exception | None) -> float:
        """Exponential backoff with jitter, or what the server asked for.

        Jitter is not decoration: without it every client that failed against one
        deployment retries in the same millisecond and reproduces the load that caused it.
        """
        stated = getattr(last, "retry_after", None)
        if stated is not None:
            return float(stated)
        return _BACKOFF * (2 ** attempt) * (0.5 + random.random())


def _drop_none(params: dict[str, Any] | None) -> dict[str, Any] | None:
    """Omit unset query parameters rather than sending `?user=`. The facade reads an
    empty `user` as a user literally named empty string, which silently creates a
    second, invisible partition of the store — see `memvara_cloud/rest/scope.py`."""
    if params is None:
        return None
    return {k: v for k, v in params.items() if v is not None}


def _body(response: Any) -> Any:
    """The decoded envelope, or `{}` for a response that carried none — a proxy's 502
    is still an error and must not become a JSONDecodeError."""
    try:
        return response.json()
    except ValueError:
        return {}


def _wrapped(exc: Exception | None) -> Exception:
    """A transport failure that outlived every attempt, named as one."""
    return RemoteError(0, "transport", f"could not reach the deployment: {exc}", False)
