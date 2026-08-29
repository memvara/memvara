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

**`HttpClient` and `AsyncHttpClient` are two transports sharing one set of rules.**
`_outcome()` and `_delay()` below decide what is retryable and how long to wait before
trying again, and neither function touches a socket — one reads an already-received
`httpx.Response`, the other reads jitter and a clock. Both clients call the same two
functions from their own attempt loop (a blocking `for` with `time.sleep` in one, the
same `for` with `await asyncio.sleep` in the other), so the retry *rule* cannot drift
between a sync and an async caller of the same API — only the mechanics of waiting
differ, which is the one thing that has to.
"""
from __future__ import annotations

import asyncio
import random
import time
import uuid
from typing import Any, Callable, Coroutine

from .errors import RemoteError, error_from_response

#: Attempts, total, per call. Three is two retries — enough for a redeploy or a deadlock,
#: short enough that a caller does not wait a minute to be told something is down.
DEFAULT_ATTEMPTS = 3
#: Seconds. Matches `store/remote.py`, so the two clients time out alike.
DEFAULT_TIMEOUT = 30.0
#: Base for exponential backoff, in seconds, before jitter.
_BACKOFF = 0.25
#: Seconds. The longest this client will wait on a server-supplied `Retry-After` before it
#: stops waiting and raises instead. Set to the default timeout, because a client that will
#: abandon a whole request after 30 seconds has no business blocking for an hour between
#: two of them: a rate limiter is entitled to say "come back in an hour", and the honest
#: way to pass that on is the `RateLimited` exception carrying `retry_after`, which the
#: caller can act on. Sleeping it would be a two-hour hang on a 30-second client, and on
#: `AsyncHttpClient` it would be that hang inside an event loop.
MAX_RETRY_AFTER = DEFAULT_TIMEOUT


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
                json: Any = None, write: bool = False,
                attempts: int | None = None, timeout: float | None = None) -> Any:
        """One call, decoded, retried where retrying is safe.

        `attempts` and `timeout` override this client's own for one call, and exist for
        one caller: a probe whose value is that failing is cheap. A server asking a
        deployment what it is at startup wants an answer in seconds or not at all — three
        attempts at a 30-second timeout, plus backoff, is a minute and a half of silent
        stdio before it gives up and degrades, which is the opposite of what degrading
        gracefully is for. Every ordinary call leaves both `None` and keeps the client's.

        Retries on a server-classified `retryable` error, on 429, and on a connect-phase
        failure (`httpx.ConnectError`, `httpx.ConnectTimeout`) — none of these ever
        reached the server, so repeating them is safe regardless of method. A
        `httpx.ReadTimeout` did reach the server and is retried too, but only because
        `write=True` attaches an `Idempotency-Key` that is held constant across this
        call's own retries; callers pass `write=True` for every route that mutates.
        A non-retryable error, or the last error once attempts are exhausted, is raised.

        The 429 case holds for a rate limit the facade classified *and* for one an edge
        proxy returned with no envelope at all — `errors._BY_STATUS` is what makes the
        second one a `RateLimited` rather than an `InvalidRequest`. What the server asked
        to wait is waited, up to `MAX_RETRY_AFTER`; a longer `Retry-After` raises the
        `RateLimited` immediately rather than blocking on it, with the server's own number
        on the exception.
        """
        import httpx

        headers = {"Idempotency-Key": uuid.uuid4().hex} if write else None
        # Only when given: httpx reads `timeout=None` as "no timeout at all", which is the
        # opposite of "use the client's", so an unconditional keyword would turn an
        # override into a hang.
        extra: dict[str, Any] = {} if timeout is None else {"timeout": timeout}
        tries = self._attempts if attempts is None else max(1, attempts)
        last: Exception | None = None
        for attempt in range(tries):
            try:
                response = self._client.request(
                    method, path,
                    params=_drop_none(params), json=json, headers=headers, **extra)
            except httpx.TransportError as exc:
                last = exc
            else:
                value, err = _outcome(response)
                if err is None:
                    return value
                last = err
                if not err.retryable:
                    raise err
            if attempt + 1 < tries:
                self._sleep(_delay(attempt, last))
        raise last if isinstance(last, RemoteError) else _wrapped(last)


class AsyncHttpClient:
    """`HttpClient`'s twin on `httpx.AsyncClient`, for a caller already on an event loop.

    Every rule is the same rule — same attempts, same backoff, same idempotency-key
    handling — because both clients call `_outcome()` and `_delay()` below rather than
    each keeping its own copy. What differs is only what has to: `await`ing the request
    and `await asyncio.sleep(...)` instead of a blocking call and `time.sleep(...)`.
    """

    def __init__(self, api_key: str, base_url: str, *, timeout: float = DEFAULT_TIMEOUT,
                 attempts: int = DEFAULT_ATTEMPTS,
                 sleep: Callable[[float], Coroutine[Any, Any, None]] = asyncio.sleep,
                 ) -> None:
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(install_hint()) from exc
        self._attempts = max(1, attempts)
        self._sleep = sleep
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    async def aclose(self) -> None:
        """Release the underlying connection pool."""
        await self._client.aclose()

    async def request(self, method: str, path: str, *,
                      params: dict[str, Any] | None = None,
                      json: Any = None, write: bool = False,
                      attempts: int | None = None, timeout: float | None = None) -> Any:
        """`HttpClient.request`, awaited. See there for the retry rule itself, and for
        what the two per-call overrides are for."""
        import httpx

        headers = {"Idempotency-Key": uuid.uuid4().hex} if write else None
        extra: dict[str, Any] = {} if timeout is None else {"timeout": timeout}
        tries = self._attempts if attempts is None else max(1, attempts)
        last: Exception | None = None
        for attempt in range(tries):
            try:
                response = await self._client.request(
                    method, path,
                    params=_drop_none(params), json=json, headers=headers, **extra)
            except httpx.TransportError as exc:
                last = exc
            else:
                value, err = _outcome(response)
                if err is None:
                    return value
                last = err
                if not err.retryable:
                    raise err
            if attempt + 1 < tries:
                await self._sleep(_delay(attempt, last))
        raise last if isinstance(last, RemoteError) else _wrapped(last)


def _outcome(response: Any) -> tuple[Any, RemoteError | None]:
    """One completed response, classified: `(decoded_value, None)` on success, or
    `(None, error)` on failure, where `error.retryable` says whether trying again could
    help. Shared by both clients so that classification cannot drift between them —
    only how each one waits and re-sends differs.
    """
    if response.status_code < 400:
        return (response.json() if response.content else None), None
    return None, error_from_response(
        response.status_code, _body(response), response.headers.get("Retry-After"))


def _delay(attempt: int, last: Exception | None) -> float:
    """Exponential backoff with jitter, or what the server asked for, up to a ceiling.

    Jitter is not decoration: without it every client that failed against one
    deployment retries in the same millisecond and reproduces the load that caused it.

    **A `Retry-After` above `MAX_RETRY_AFTER` raises `last` instead of being slept on.**
    The header is a number the server chose and this client has no say in — `Retry-After:
    3600` is a legitimate answer from a rate limiter, and obeying it here would block the
    calling thread for an hour, twice, inside a client documented to give up on any one
    request after thirty seconds. Raising hands the caller the same instruction in a form
    they can act on: `RateLimited.retry_after` still carries the server's number, so a
    caller who genuinely wants to wait an hour can, deliberately, at a level that knows
    whether an hour is acceptable. Both clients call this from inside their attempt loop,
    so the raise leaves `request()` with the server's own error rather than a timeout.
    """
    if last is not None:
        stated = getattr(last, "retry_after", None)
        if stated is not None:
            wait = float(stated)
            if wait > MAX_RETRY_AFTER:
                raise last
            return wait
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
