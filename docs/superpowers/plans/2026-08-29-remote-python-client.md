# Remote Python Client Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `Memvara(api_key=...)` returns a client that talks to a hosted memvara-cloud deployment over its `/v1` REST API, and `memvara-mcp --mode cloud` starts instead of refusing.

**Architecture:** A parallel class, `RemoteMemvara`, that calls the `/v1` facade with the engine running server-side. It is not a `Store` and not a subclass of `Memvara`. `Memvara.__new__` dispatches to it when the caller passes credentials. A `MemoryAPI` protocol lets the existing MCP tool table serve from either implementation without a second tool table.

**Tech Stack:** Python 3.10+, `httpx` (lazy import, `memvara[cloud]` extra), `pytest`, `mypy`.

**Spec:** `docs/superpowers/specs/2026-08-29-remote-python-client-design.md` — read it before Task 1. The plan argues from the spec and does not repeat its reasoning.

## Global Constraints

- **Run tests with `PYTHONPATH` pinned to this worktree.** The editable install points at the main checkout, so a bare `pytest` in this worktree silently imports the *other* copy of `memvara` and every result is about code you did not change. Every test command in this plan is written as `PYTHONPATH=$PWD python -m pytest ...`. Do not shorten them.
- **Coverage runs as two commands, never one.** A combined run exits 139 and kills the report, which reads as success. `PYTHONPATH=$PWD COVERAGE_FILE=.coverage.remote python -m coverage run -m pytest ...` then `COVERAGE_FILE=.coverage.remote python -m coverage report`.
- **Use a private `COVERAGE_FILE`.** Another session sharing `.coverage` produces a report that is wrong in the direction that looks fine.
- **`--doctest-modules` is on** and `testpaths = ["tests", "memvara"]`. Any example you write in a docstring runs as a test. A `>>>` block that would make a network call must not be written as a doctest.
- **numpy and nothing else in the core.** `import memvara` must never require `httpx`. Import it lazily, inside a function or method, with a message naming `pip install "memvara[cloud]"`.
- **Commit files by name.** Never `git add -A`, `git add .`, or `git commit -a`. Other agents are working in this checkout.
- **No AI attribution anywhere** — not in a commit message, a PR title, a PR body, or an issue. No `Co-Authored-By` trailer, no "Generated with" footer. Absolute.
- **Documentation ships in the same commit as the code that changes its meaning.** Tasks below name the exact files; they are not optional and not a follow-up.
- **Never delete or weaken a test to make a gate pass.** If finished work depends on something that does not exist, build the missing thing or ship it visibly disabled and say so in plain words.
- **Two endpoints do not exist yet.** `POST /v1/end` and `Idempotency-Key` are specified in `local/END-ENDPOINT-SPEC.md` and land in `memvara-cloud`. Every task here mocks the HTTP layer, so all of it is testable now. Only the live round-trip tests skip.

---

## File Structure

**Create:**

| file | responsibility |
|---|---|
| `memvara/remote/__init__.py` | public exports |
| `memvara/remote/errors.py` | exception hierarchy, envelope → exception |
| `memvara/remote/creds.py` | resolve api key and base url |
| `memvara/remote/client.py` | one HTTP layer: auth, timeout, retries, idempotency |
| `memvara/remote/hydrate.py` | JSON → the library's own dataclasses |
| `memvara/remote/api.py` | `RemoteMemvara`, `ScopedRemoteMemvara` |
| `memvara/remote/aio.py` | `AsyncRemoteMemvara`, `AsyncScopedRemoteMemvara` |
| `tests/test_remote_errors.py` … `tests/test_remote_cloud_mode.py` | one per task |

**Modify:** `memvara/core.py` (dispatch), `memvara/__init__.py` (exports), `memvara/aio.py` (docstring), `memvara/store/remote.py` (share the transport), `memvara/server/config.py` (cloud branch), `memvara/server/tools.py` (protocol type, `_standing`), `tests/test_config_cloud.py`, `pyproject.toml`, and the documentation named per task.

---

### Task 1: Error hierarchy

**Files:**
- Create: `memvara/remote/__init__.py`, `memvara/remote/errors.py`
- Test: `tests/test_remote_errors.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `RemoteError(status_code: int, code: str, message: str, retryable: bool)`; subclasses `AuthError`, `ScopeError`, `NotFound`, `Conflict`, `QuotaExhausted`, `RateLimited` (extra attribute `retry_after: float | None`), `LegalHold`, `ReadOnly`, `InvalidRequest`, `ServerError`; and `error_from_response(status_code: int, body: dict, retry_after: str | None) -> RemoteError`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_remote_errors.py
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=$PWD python -m pytest tests/test_remote_errors.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'memvara.remote'`

- [ ] **Step 3: Write the implementation**

```python
# memvara/remote/__init__.py
"""Client for a hosted memvara-cloud deployment.

`memvara.store.remote` is a `Store` — the low-level surface the *engine* calls. This
package is the other seam: a client of the `/v1` facade, where the engine runs
server-side. See `docs/OPEN-CORE.md` for why the two do not converge.
"""
from .errors import (
    AuthError, Conflict, InvalidRequest, LegalHold, NotFound, QuotaExhausted,
    RateLimited, ReadOnly, RemoteError, ScopeError, ServerError,
)

__all__ = [
    "RemoteError", "AuthError", "ScopeError", "NotFound", "Conflict",
    "QuotaExhausted", "RateLimited", "LegalHold", "ReadOnly", "InvalidRequest",
    "ServerError",
]
```

```python
# memvara/remote/errors.py
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONPATH=$PWD python -m pytest tests/test_remote_errors.py -v`
Expected: PASS, 17 tests

- [ ] **Step 5: Commit**

```bash
git add memvara/remote/__init__.py memvara/remote/errors.py tests/test_remote_errors.py
git commit -m "feat(remote): map the /v1 error envelope to exception types"
```

---

### Task 2: Credential and base-URL resolution

**Files:**
- Create: `memvara/remote/creds.py`
- Modify: `memvara/server/config.py` (name the existing resolution as a reusable function)
- Test: `tests/test_remote_creds.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `resolve(api_key: str | None, base_url: str | None, env: Mapping[str, str] | None = None) -> tuple[str, str]`, returning `(api_key, base_url)` and raising `MissingCredential` (a subclass of `RuntimeError`) when no key can be found.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_remote_creds.py
"""Where an api key and a base url come from, in what order.

The order matters more than it looks. An explicit argument beating the environment is
ordinary; the credentials file coming last is what makes `memvara-mcp login` usable
without making it authoritative over a key the caller passed in this process.
"""
import json

import pytest

from memvara.remote.creds import MissingCredential, resolve


def test_an_explicit_key_wins_over_everything():
    key, url = resolve("explicit", None, env={"MEMVARA_API_KEY": "from-env"})
    assert key == "explicit"


def test_the_environment_supplies_the_key_when_the_caller_did_not():
    key, _ = resolve(None, "https://example.test", env={"MEMVARA_API_KEY": "from-env"})
    assert key == "from-env"


def test_the_credentials_file_is_the_last_resort(tmp_path, monkeypatch):
    path = tmp_path / "credentials.json"
    path.write_text(json.dumps({"api_key": "from-file"}))
    monkeypatch.setattr("memvara.remote.creds.CREDENTIALS_PATH", path)
    key, _ = resolve(None, "https://example.test", env={})
    assert key == "from-file"


def test_no_key_anywhere_names_the_command_that_writes_one(tmp_path, monkeypatch):
    monkeypatch.setattr("memvara.remote.creds.CREDENTIALS_PATH", tmp_path / "absent.json")
    with pytest.raises(MissingCredential) as caught:
        resolve(None, "https://example.test", env={})
    assert "memvara-mcp login" in str(caught.value)


def test_the_base_url_defaults_to_the_hosted_deployment():
    _, url = resolve("k", None, env={})
    assert url == "https://app.memvara.dev"


def test_the_environment_can_point_at_another_deployment():
    _, url = resolve("k", None, env={"MEMVARA_SERVER_URL": "https://self.hosted.test"})
    assert url == "https://self.hosted.test"


def test_a_trailing_slash_is_removed_so_paths_do_not_double_up():
    _, url = resolve("k", "https://example.test/", env={})
    assert url == "https://example.test"


def test_a_blank_environment_value_is_treated_as_unset_rather_than_as_a_url():
    _, url = resolve("k", None, env={"MEMVARA_SERVER_URL": "   "})
    assert url == "https://app.memvara.dev"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=$PWD python -m pytest tests/test_remote_creds.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'memvara.remote.creds'`

- [ ] **Step 3: Write the implementation**

```python
# memvara/remote/creds.py
"""Resolve the two things a remote client needs: a key, and somewhere to send it.

The constants come from `memvara.server.config` rather than being declared again. There
were already two copies of the default URL — `config.py` and `login.py` — and a third
would be the one that drifts.
"""
from __future__ import annotations

import json
import os
from typing import Mapping

from ..server.config import CREDENTIALS_PATH, DEFAULT_SERVER_URL

__all__ = ["MissingCredential", "resolve", "CREDENTIALS_PATH"]


class MissingCredential(RuntimeError):
    """No api key was passed, exported, or written by `memvara-mcp login`."""


def _clean(value: str | None) -> str | None:
    """A blank or whitespace-only environment variable means unset, not empty. An
    exported `MEMVARA_SERVER_URL=` would otherwise be a base url of `""`."""
    return value.strip() if value and value.strip() else None


def _from_file() -> str | None:
    """The key `memvara-mcp login` wrote, or None. A malformed file is None rather than
    an exception: the caller's next stop is the same "run login" message either way."""
    try:
        body = json.loads(CREDENTIALS_PATH.read_text())
    except (OSError, ValueError):
        return None
    key = body.get("api_key") if isinstance(body, dict) else None
    return _clean(key if isinstance(key, str) else None)


def resolve(api_key: str | None, base_url: str | None,
            env: Mapping[str, str] | None = None) -> tuple[str, str]:
    """`(api_key, base_url)` for a remote client, or raise naming how to get one."""
    environ = os.environ if env is None else env
    key = _clean(api_key) or _clean(environ.get("MEMVARA_API_KEY")) or _from_file()
    if key is None:
        raise MissingCredential(
            "No memvara api key. Pass Memvara(api_key=...), export MEMVARA_API_KEY, or "
            f'run "memvara-mcp login" to write one to {CREDENTIALS_PATH}.')
    url = (_clean(base_url) or _clean(environ.get("MEMVARA_SERVER_URL"))
           or DEFAULT_SERVER_URL)
    return key, url.rstrip("/")
```

- [ ] **Step 4: Add `DEFAULT_SERVER_URL` to `memvara/server/config.py`**

`config.py` spells `"https://app.memvara.dev"` inline in three places (the `server_url` field default and twice in `from_env`). Add the constant next to `CREDENTIALS_PATH` and use it in those three places. Do not touch anything else in that file.

```python
#: Where a client goes absent MEMVARA_SERVER_URL. `login.py` declares its own copy as
#: `_DEFAULT_SERVER_URL`; leaving that alone is deliberate, since collapsing them is a
#: change to a module this work has no other reason to touch.
DEFAULT_SERVER_URL = "https://app.memvara.dev"
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `PYTHONPATH=$PWD python -m pytest tests/test_remote_creds.py tests/test_config.py -v`
Expected: PASS. `test_config.py` is included because Step 4 edited `config.py`; it must be green.

- [ ] **Step 6: Commit**

```bash
git add memvara/remote/creds.py memvara/server/config.py tests/test_remote_creds.py
git commit -m "feat(remote): resolve api key and base url from arg, env, or credentials file"
```

---

### Task 3: The HTTP transport

**Files:**
- Create: `memvara/remote/client.py`
- Modify: `pyproject.toml` (the `cloud` extra's comment)
- Test: `tests/test_remote_client.py`

**Interfaces:**
- Consumes: `memvara.remote.errors.error_from_response`, `memvara.remote.creds.resolve`.
- Produces: `HttpClient(api_key: str, base_url: str, *, timeout: float = 30.0, attempts: int = 3, sleep: Callable[[float], None] = time.sleep)` with methods `request(method: str, path: str, *, params: dict | None = None, json: Any = None, write: bool = False) -> Any` and `close() -> None`; and `install_hint() -> str`.

**Retry rule, from the spec:** retry on a server-classified `retryable`, on 429, and on connect-phase failures. A write that fails after the request was sent is retried only when it carried an `Idempotency-Key`, which this client always sends on writes. `httpx.ConnectError` and `httpx.ConnectTimeout` are connect-phase; `httpx.ReadTimeout` is not.

**Say what the guarantee actually is, in the docstring.** The server's idempotency store lives in the serving process, so a retry a load balancer routes to a second worker finds no record of the first attempt and re-executes. Do not write "retries are safe". Write that a retried write is deduplicated when it reaches the worker that saw the first attempt, and that a single-worker deployment is the only one where that always holds. This was found in the final review of `memvara-cloud`'s `feat/v1-end-and-idempotency`; the server documents it in `deploy/README.md`, and a client that repeated the unqualified claim would be the place the promise broke.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_remote_client.py
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
    c = HttpClient("k", "https://example.test", sleep=lambda _: None, **kw)
    c._client = httpx.Client(base_url="https://example.test",
                             headers={"Authorization": "Bearer k"},
                             transport=httpx.MockTransport(handler))
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=$PWD python -m pytest tests/test_remote_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'memvara.remote.client'`

- [ ] **Step 3: Write the implementation**

```python
# memvara/remote/client.py
"""One HTTP layer for every remote call, so retry and error behaviour cannot differ
between two clients of the same API.

**`httpx` is imported inside `__init__`, never at module level.** `import memvara` must
work with numpy alone — the core promise in `docs/INTERNALS.md`, invariant 5 — and a
module-level import here would break it the moment anything imports this file.

**Why writes are retried at all.** A write that fails *after* the request was sent may
have committed, so repeating it can write the fact twice. The facade accepts an
`Idempotency-Key`, and this client sends one on every write and repeats it verbatim across
retries of that write, which is what makes the repeat safe. A new write gets a new key.
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
    return ('httpx is required to talk to a hosted deployment. Install it with '
            '`pip install "memvara[cloud]"`.')


class HttpClient:
    """A `/v1` transport bound to one deployment and one credential."""

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
        self._client.close()

    def request(self, method: str, path: str, *, params: dict[str, Any] | None = None,
                json: Any = None, write: bool = False) -> Any:
        """One call, decoded, retried where retrying is safe.

        `write=True` attaches an `Idempotency-Key` held constant across this call's own
        retries. Callers pass it for every route that mutates.
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
                # Connect-phase failures never reached the server, so repeating them is
                # safe whatever the method. A read timeout did reach it, and is safe to
                # repeat only because of the idempotency key above.
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
    empty `user` as a user named empty string, which is a second invisible partition of
    the store — see `memvara_cloud/rest/scope.py`."""
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONPATH=$PWD python -m pytest tests/test_remote_client.py -v`
Expected: PASS, 10 tests

- [ ] **Step 5: Correct the `cloud` extra's comment in `pyproject.toml`**

It currently says login and `RemoteStore` are the only things in the distribution making an outbound call. Replace that clause so it names this client too. Change only the comment lines directly above `cloud = ["httpx>=0.27"]`.

- [ ] **Step 6: Commit**

```bash
git add memvara/remote/client.py tests/test_remote_client.py pyproject.toml
git commit -m "feat(remote): add the HTTP transport with idempotent write retries"
```

---

### Task 4: Hydration

**Files:**
- Create: `memvara/remote/hydrate.py`
- Test: `tests/test_remote_hydrate.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `claim(body: dict) -> Claim`, `episode(body: dict) -> Episode`, `result(body: dict) -> Result`, `explanation(body: dict) -> Explanation`, `receipt(body: dict) -> WriteReceipt`, `provenance(body: dict) -> Provenance`, `reading(body: dict) -> Reading`, `answer(body: dict) -> Answer`, `delta(body: dict) -> Delta`, `edge(body: dict) -> Edge`, `path(body: dict) -> Path`, `scope(body: dict) -> Scope`.

**Authority:** `memvara_cloud/rest/render.py`. Each function here is the inverse of the function of the same name there. Read that file before writing this one; where the two disagree, `render.py` is right and this is the bug.

The three traps, from the spec: `extractor` is `""` in the library and `null` on the wire; `salience_base` and `last_observed` are top-level on the wire and `meta` keys under `SALIENCE_BASE` / `LAST_OBSERVED` in the library; `state` and `links` are derived and must never be stored back.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_remote_hydrate.py
"""JSON from /v1 turned back into the library's own dataclasses.

The round trip is the assertion that matters. A hand-written expectation agrees with
whatever the author misread; a claim that survives render-then-hydrate unchanged does not.
"""
import pytest

from memvara import Claim, Derivation, MemoryType, Scope, utcnow
from memvara.remote import hydrate
from memvara.types import LAST_OBSERVED, SALIENCE_BASE

pytest.importorskip("memvara_cloud", reason="the renderer is the authority for this test")


def _wire(claim):
    from memvara_cloud.rest import render
    return render.memory(claim).model_dump(mode="json")


def test_a_claim_survives_render_then_hydrate_unchanged():
    original = Claim(subject="user", predicate="lives_in", object="Berlin",
                     scope=Scope("t", "u"), text="user lives in Berlin",
                     memory_type=MemoryType.SEMANTIC, confidence=0.9,
                     derivation=Derivation.LLM_EXTRACT, extractor="rules/1",
                     sources=["ep_1"], meta={"note": "kept"})
    restored = hydrate.claim(_wire(original))
    for field in ("subject", "predicate", "object", "text", "polarity", "memory_type",
                  "confidence", "salience", "observation_count", "derivation",
                  "extractor", "id", "valid_from", "valid_to", "recorded_at",
                  "invalidated_at", "invalidated_by"):
        assert getattr(restored, field) == getattr(original, field), field
    assert restored.scope == original.scope
    assert restored.sources == original.sources
    assert restored.meta["note"] == "kept"


def test_an_unrecorded_extractor_comes_back_as_the_empty_string_not_none():
    original = Claim(subject="user", predicate="likes", object="tea", extractor="")
    assert hydrate.claim(_wire(original)).extractor == ""


def test_salience_base_and_last_observed_return_to_meta():
    original = Claim(subject="user", predicate="likes", object="tea")
    original.meta[SALIENCE_BASE] = 2.5
    original.meta[LAST_OBSERVED] = 1700000000.0
    restored = hydrate.claim(_wire(original))
    assert restored.meta[SALIENCE_BASE] == 2.5
    assert restored.meta[LAST_OBSERVED] == 1700000000.0


def test_the_closure_record_survives_because_it_is_not_a_reserved_key():
    from memvara.types import CLOSURE, close_out
    original = Claim(subject="user", predicate="works_at", object="Acme")
    close_out(original, utcnow(), None, "ended")
    assert CLOSURE in hydrate.claim(_wire(original)).meta


def test_a_missing_required_field_raises_rather_than_defaulting():
    body = _wire(Claim(subject="user", predicate="likes", object="tea"))
    del body["predicate"]
    with pytest.raises(KeyError):
        hydrate.claim(body)


def test_state_is_recomputed_and_never_read_from_the_wire():
    original = Claim(subject="user", predicate="likes", object="tea")
    body = _wire(original)
    body["state"] = "retired"          # a lie the hydrator must ignore
    assert hydrate.claim(body).is_live()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=$PWD python -m pytest tests/test_remote_hydrate.py -v`
Expected: FAIL — `ImportError: cannot import name 'hydrate'` (or a skip if `memvara_cloud` is not importable; make it importable for this task by adding `/Applications/workstation/memvara-cloud` to `PYTHONPATH` for the run)

- [ ] **Step 3: Write the implementation**

```python
# memvara/remote/hydrate.py
"""JSON from `/v1` back into the library's own dataclasses.

Each function here is the inverse of the function of the same name in
`memvara_cloud/rest/render.py`. That module is the authority: where the two disagree, this
one is wrong.

**Required fields are indexed, not `.get()`.** A server that renamed a field should raise
on the first call rather than hand back a claim carrying a plausible zero, which nothing
downstream can tell from a real one.

Three asymmetries the renderer introduces and this must undo:

* `extractor` is `""` in the library and `null` on the wire.
* `salience_base` and `last_observed` are top-level on the wire and `meta` keys here.
* `state` and `links` are derived server-side. They are dropped, never stored: a claim
  carrying a `state` that disagreed with its own timestamps would be unfixable.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from ..retrieve.traverse import Edge, Path
from ..types import (
    LAST_OBSERVED, SALIENCE_BASE, Answer, Claim, Delta, Derivation, Episode,
    Explanation, MemoryType, Provenance, Reading, Result, Scope, WriteReceipt,
)

__all__ = ["claim", "episode", "result", "explanation", "receipt", "provenance",
           "reading", "answer", "delta", "edge", "path", "scope"]


def _dt(value: Any) -> datetime | None:
    return None if value is None else datetime.fromisoformat(str(value))


def scope(body: dict[str, Any]) -> Scope:
    return Scope(body["tenant"], body.get("user"), body.get("agent"),
                 body.get("session"))


def claim(body: dict[str, Any]) -> Claim:
    valid, txn = body["valid_time"], body["transaction_time"]
    meta = dict(body.get("metadata") or {})
    if body.get("salience_base") is not None:
        meta[SALIENCE_BASE] = body["salience_base"]
    if body.get("last_observed") is not None:
        meta[LAST_OBSERVED] = body["last_observed"]
    out = Claim(
        subject=body["subject"],
        predicate=body["predicate"],
        object=body["object"],
        scope=scope(body["scope"]),
        text=body["text"],
        polarity=body["polarity"],
        memory_type=MemoryType(body["memory_type"]),
        confidence=body["confidence"],
        salience=body["salience"],
        observation_count=body["observation_count"],
        sources=list(body.get("source_ids") or []),
        derivation=Derivation(body["derivation"]),
        # The wire says null for what the library spells as the empty string.
        extractor=body.get("extractor") or "",
        id=body["id"],
        meta=meta,
    )
    out.valid_from = _dt(valid["valid_from"])
    out.valid_to = _dt(valid.get("valid_to"))
    out.recorded_at = _dt(txn["recorded_at"])
    out.invalidated_at = _dt(txn.get("invalidated_at"))
    out.invalidated_by = txn.get("invalidated_by")
    return out


def episode(body: dict[str, Any]) -> Episode:
    return Episode(content=body["content"], role=body["role"], ts=_dt(body["ts"]),
                   id=body["id"], meta=dict(body.get("metadata") or {}))


def explanation(body: dict[str, Any] | None) -> Explanation:
    """`render.ranking`, backwards. `None` for a response that carried no ranking — a
    listing rather than a search — and an all-defaults `Explanation` is the right answer
    there, because nothing ranked it."""
    if not body:
        return Explanation()
    out = Explanation()
    for name in ("vector_rank", "vector_score", "lexical_rank", "lexical_score",
                 "fusion_score", "recency", "confidence", "salience", "rerank_score",
                 "raw_score", "final_score", "graph_rank", "graph_score",
                 "temporal_rank", "temporal_score"):
        if name in body and body[name] is not None:
            setattr(out, name, body[name])
    return out


def result(body: dict[str, Any]) -> Result:
    return Result(claim=claim(body["memory"]), score=body["score"],
                  explain=explanation(body.get("ranking")))


def receipt(body: dict[str, Any]) -> WriteReceipt:
    return WriteReceipt(
        episode_ids=list(body.get("episode_ids") or []),
        added=[claim(c) for c in body.get("added") or []],
        closed=[claim(c) for c in body.get("invalidated") or []],
        reinforced=[claim(c) for c in body.get("reinforced") or []],
        skipped=body.get("skipped", 0),
        unextracted=body.get("unextracted", 0),
    )


def provenance(body: dict[str, Any]) -> Provenance:
    return Provenance(
        claim=claim(body["memory"]),
        episodes=[episode(e) for e in body.get("sources") or []],
        derivation=Derivation(body["derivation"]),
        extractor=body.get("extractor") or "",
        superseded=[claim(c) for c in body.get("superseded") or []],
    )


def reading(body: dict[str, Any]) -> Reading:
    return Reading(
        subject=body["subject"], predicate=body["predicate"],
        now=tuple(claim(c) for c in body.get("now") or []),
        then=tuple(claim(c) for c in body.get("then") or []),
        stated=tuple(claim(c) for c in body.get("stated") or []),
        timeline=tuple(claim(c) for c in body.get("timeline") or []),
        single_valued=body.get("single_valued", False),
    )


def answer(body: dict[str, Any]) -> Answer:
    return Answer(question=body["question"], at=_dt(body["at"]),
                  readings=tuple(reading(r) for r in body.get("readings") or []),
                  text=body.get("text", ""))


def delta(body: dict[str, Any]) -> Delta:
    return Delta(since=_dt(body["since"]),
                 added=tuple(claim(c) for c in body.get("added") or []),
                 gone=tuple(claim(c) for c in body.get("gone") or []))


def edge(body: dict[str, Any]) -> Edge:
    """`render.edge`, backwards.

    `backward` is carried rather than recomputed. A claim read object-to-subject is a
    different statement — `Acme founded_by Bob` reached from Bob is still "Acme was
    founded by Bob" — and inferring the direction here would assert something nobody
    stored.
    """
    return Edge(claim=claim(body["memory"]), backward=body["backward"],
                strength=body["strength"])


def path(body: dict[str, Any]) -> Path:
    """`render.path`, backwards — `nodes`, `edges` and `score` only.

    `labels` and `hops` are on the wire and are **not** passed back: both are properties
    computed from the edges the walk crossed. Storing the rendered spellings would be a
    second implementation of the fold that identity is stored under, and a second one can
    disagree.
    """
    return Path(nodes=tuple(body["nodes"]),
                edges=tuple(edge(e) for e in body.get("edges") or []),
                score=body["score"])
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONPATH=$PWD:/Applications/workstation/memvara-cloud python -m pytest tests/test_remote_hydrate.py -v`
Expected: PASS, 6 tests

If a field name in the code above does not match `render.py`, `render.py` wins — fix the code, not the test.

- [ ] **Step 5: Commit**

```bash
git add memvara/remote/hydrate.py tests/test_remote_hydrate.py
git commit -m "feat(remote): hydrate /v1 responses into the library's dataclasses"
```

---

### Task 5: `RemoteMemvara` — the read surface

**Files:**
- Create: `memvara/remote/api.py`
- Test: `tests/test_remote_reads.py`

**Interfaces:**
- Consumes: `HttpClient`, `hydrate`, `creds.resolve`.
- Produces: `RemoteMemvara(*, api_key=None, base_url=None, tenant="default", user=None, agent=None, session=None, timeout=30.0, redactor=None)` with the read methods below, plus `close()` and the attribute `default_scope: Scope`.

**The mapping. Implement every row; no row is optional.**

| method | HTTP | path | notes |
|---|---|---|---|
| `search(query, k, min_score, memory_types, as_of, valid_at, known_at, include_episodes)` | POST | `/v1/search` | `hydrate.result` per hit |
| `recall(query, k, min_score, memory_types, include_episodes, with_ids)` | POST | `/v1/recall` | returns the rendered string, or results when `with_ids` |
| `get(claim_id)` | GET | `/v1/memories/{id}` | `None` on `NotFound` |
| `get_all(states, memory_types, limit, offset, as_of, valid_at, known_at)` | GET | `/v1/memories` | pages; `hydrate.claim` per item |
| `count(...)` | GET | `/v1/memories` | `limit=1`, return `total` |
| `history(subject, predicate)` | GET | `/v1/history` | |
| `why(claim_id)` | GET | `/v1/memories/{id}/why` | `hydrate.provenance` |
| `ask(question, at, k, min_score)` | POST | `/v1/ask` | `hydrate.answer` |
| `since(when)` | GET | `/v1/since` | `hydrate.delta` |
| `produced(episode_id)` | GET | `/v1/episodes/{id}/produced` | |
| `neighborhood(key, depth, limit, as_of, valid_at)` | GET | `/v1/neighborhood` | |
| `paths_between(source, target, depth, limit)` | GET | `/v1/paths` | |
| `standing(k)` | GET | `/v1/standing` | Task 11 depends on this existing |
| `stats()` | GET | `/v1/stats` | returns the dict as-is |
| `connectivity()` | GET | `/v1/stats` | `{live_claims, joinable_claims}`, or `{}` when the facade omits them |
| `whoami()` | GET | `/v1/whoami` | |
| `health()` | GET | `/v1/health` | |

Scope goes on every call as `user` / `agent` / `session` query parameters, dropped when `None` by `_drop_none`. `tenant` is never sent: the bearer token decides it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_remote_reads.py
"""Every read method reaches the endpoint it claims to, with the scope it was bound to.

The assertions are on the request the client made, not only on the value returned. A
method that hit the wrong path and happened to decode the fixture would otherwise pass.
"""
import httpx
import pytest

from memvara.remote.api import RemoteMemvara
from memvara.remote.client import HttpClient


@pytest.fixture
def recorded():
    calls = []

    def build(payload, **kw):
        def handler(request):
            calls.append(request)
            return httpx.Response(200, json=payload)

        mem = RemoteMemvara(api_key="k", base_url="https://example.test", **kw)
        mem._http = HttpClient("k", "https://example.test", sleep=lambda _: None)
        mem._http._client = httpx.Client(base_url="https://example.test",
                                         transport=httpx.MockTransport(handler))
        return mem

    build.calls = calls
    return build


def test_stats_reaches_the_stats_endpoint(recorded):
    mem = recorded({"claims": 3})
    assert mem.stats()["claims"] == 3
    assert recorded.calls[-1].url.path == "/v1/stats"


def test_get_returns_none_for_a_memory_that_is_not_there(recorded):
    def handler(request):
        return httpx.Response(404, json={"error": {"code": "not_found", "message": "x"}})

    mem = RemoteMemvara(api_key="k", base_url="https://example.test")
    mem._http = HttpClient("k", "https://example.test", sleep=lambda _: None)
    mem._http._client = httpx.Client(base_url="https://example.test",
                                     transport=httpx.MockTransport(handler))
    assert mem.get("cl_missing") is None


def test_the_bound_scope_is_sent_as_query_parameters(recorded):
    mem = recorded({"count": 0, "total": 0, "limit": 1, "offset": 0, "memories": []},
                   user="alice", agent="a1")
    mem.count()
    params = dict(recorded.calls[-1].url.params)
    assert params["user"] == "alice"
    assert params["agent"] == "a1"


def test_an_unbound_scope_field_is_omitted_rather_than_sent_empty(recorded):
    mem = recorded({"count": 0, "total": 0, "limit": 1, "offset": 0, "memories": []},
                   user="alice")
    mem.count()
    assert "agent" not in dict(recorded.calls[-1].url.params)


def test_the_tenant_is_never_sent_because_the_token_decides_it(recorded):
    mem = recorded({"claims": 0}, tenant="not-mine")
    mem.stats()
    assert "tenant" not in dict(recorded.calls[-1].url.params)


def test_count_asks_for_one_row_and_reports_the_total(recorded):
    mem = recorded({"count": 1, "total": 47, "limit": 1, "offset": 0, "memories": []})
    assert mem.count() == 47
    assert dict(recorded.calls[-1].url.params)["limit"] == "1"


def test_connectivity_is_empty_when_the_facade_does_not_report_joins(recorded):
    mem = recorded({"claims": 5})
    assert mem.connectivity() == {}


def test_connectivity_reports_the_two_counts_when_present(recorded):
    mem = recorded({"live_claims": 10, "joinable_claims": 4})
    assert mem.connectivity() == {"live_claims": 10, "joinable_claims": 4}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=$PWD python -m pytest tests/test_remote_reads.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'memvara.remote.api'`

- [ ] **Step 3: Write the implementation**

Write `memvara/remote/api.py` with the class docstring below, `__init__`, `_params`, `close`, and every read method in the mapping table. Two worked examples fix the shape; the rest follow it exactly.

```python
# memvara/remote/api.py
"""`RemoteMemvara`: the library's own API, served by a hosted deployment.

Not a `Store` and not a subclass of `Memvara`. The engine runs server-side; this class
turns a method call into one `/v1` request and hydrates what comes back into the same
dataclasses the local engine returns, so calling code cannot tell which it holds.

**What is absent is absent, not raising.** `reembed`, `pending_extraction`, `reextract`
and `reset` have no endpoint, so they are not methods here. A caller reaching for one gets
an `AttributeError` at the call site and mypy catches it before that — where a method that
raised would compile, ship, and fail in production.

Two divergences that are real and documented rather than hidden: `consolidate()` returns a
job handle because the endpoint is asynchronous, and there is no `prove_erased()` because
`/v1/erasures` returns its per-table counts as evidence in the erasure response itself.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from ..types import Answer, Claim, Delta, Provenance, Result, Scope
from . import hydrate
from .client import DEFAULT_TIMEOUT, HttpClient
from .creds import resolve
from .errors import NotFound


class RemoteMemvara:
    def __init__(self, *, api_key: str | None = None, base_url: str | None = None,
                 tenant: str = "default", user: str | None = None,
                 agent: str | None = None, session: str | None = None,
                 timeout: float = DEFAULT_TIMEOUT, redactor: Any = None) -> None:
        key, url = resolve(api_key, base_url)
        self._http = HttpClient(key, url, timeout=timeout)
        #: The scope this client narrows to. `tenant` is held for `default_scope`'s sake
        #: and never sent: the facade resolves it from the bearer token, and a `tenant`
        #: parameter a caller could set would be a request to be trusted about identity.
        self.default_scope = Scope(tenant, user, agent, session)
        #: Rewrites text on its way out, or None. Runs here rather than server-side on
        #: purpose: redaction that happens after the text has left the process is not
        #: redaction.
        self.redactor = redactor

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "RemoteMemvara":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _params(self, **extra: Any) -> dict[str, Any]:
        """Scope on every call, plus whatever this call adds. `None` values are dropped
        by the transport rather than sent as empty strings."""
        scope = self.default_scope
        return {"user": scope.user, "agent": scope.agent, "session": scope.session,
                **extra}

    # -- reading ---------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        return self._http.request("GET", "/v1/stats", params=self._params())

    def connectivity(self) -> dict[str, int]:
        """`live_claims` and `joinable_claims`, or `{}` when the deployment does not
        report them.

        `{}` is not the same as a store with nothing in it, and the distinction is the
        whole reason the local method documents it: an empty store answers with two zeros,
        a backend that cannot measure says nothing at all, and a caller reading a missing
        key as zero would report a star it never measured.
        """
        body = self.stats()
        if "joinable_claims" not in body or "live_claims" not in body:
            return {}
        return {"live_claims": body["live_claims"],
                "joinable_claims": body["joinable_claims"]}

    def get(self, claim_id: str) -> Claim | None:
        """One memory, or None. None rather than raising for an id in another tenant as
        well as one that never existed — the facade gives the same answer for both so
        that this cannot be used to test whether an id exists elsewhere."""
        try:
            body = self._http.request("GET", f"/v1/memories/{claim_id}",
                                      params=self._params())
        except NotFound:
            return None
        return hydrate.claim(body)

    def count(self, **filters: Any) -> int:
        body = self._http.request("GET", "/v1/memories",
                                  params=self._params(limit=1, **filters))
        return int(body["total"])
```

Now write the remaining read methods from the mapping table against that shape. Each is: build `params` or a JSON body, call `self._http.request`, hand the result to the matching `hydrate` function. `search` returns `[hydrate.result(h) for h in body["hits"]]`; `history` returns `[hydrate.claim(c) for c in body["memories"]]`; `ask` returns `hydrate.answer(body)`; `why` returns `hydrate.provenance(body)`; `since` returns `hydrate.delta(body)`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONPATH=$PWD python -m pytest tests/test_remote_reads.py -v`
Expected: PASS, 8 tests

- [ ] **Step 5: Add one test per remaining mapping-table row**

Each asserts the path reached and the type returned, in the shape of `test_stats_reaches_the_stats_endpoint`. Seventeen rows, seventeen tests. A row without a test is a row that will be wrong.

- [ ] **Step 6: Run the whole file**

Run: `PYTHONPATH=$PWD python -m pytest tests/test_remote_reads.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add memvara/remote/api.py tests/test_remote_reads.py
git commit -m "feat(remote): add the read surface over /v1"
```

---

### Task 6: `RemoteMemvara` — the write surface

**Files:**
- Modify: `memvara/remote/api.py`
- Test: `tests/test_remote_writes.py`

**Interfaces:**
- Consumes: Task 5's `RemoteMemvara`.
- Produces: `add`, `remember`, `supersede`, `forget`, `delete`, `end`, `erase`, `purge`, `consolidate`.

**The mapping. Every write passes `write=True` to the transport.**

| method | HTTP | path | notes |
|---|---|---|---|
| `add(messages, role, ts)` | POST | `/v1/memories` | `hydrate.receipt` |
| `remember(subject, predicate, obj, ...)` | POST | `/v1/facts` | `hydrate.receipt` |
| `supersede(old_claim_id, ..., close)` | POST | `/v1/memories/{id}/supersede` | `close` forwarded verbatim |
| `forget(subject, predicate, at)` | POST | `/v1/forget` | returns `list[Claim]` |
| `delete(claim_id, at, close)` | DELETE or POST | `/v1/memories/{id}` when `close="retired"`; `/v1/end` when `close="ended"` | see below |
| `end(claim_id=None, subject=None, predicate=None, at=None)` | POST | `/v1/end` | the two addressing modes |
| `erase(claim_id, sources)` | POST | `/v1/erasures` | |
| `purge(scope)` | POST | `/v1/erasures` | scope form |
| `consolidate()` | POST | `/v1/maintenance/consolidate` | returns the job body |

**`delete` routes on `close`, and that is the point of this task.** `close="retired"` is `DELETE /v1/memories/{id}`. `close="ended"` is `POST /v1/end`, because the retirement route records that we stopped believing a record and ending records that the world moved — filing one as the other is the mistake `memvara/types.py:195` calls the one that cannot be found by reading the data afterwards. Validate `close` through `memvara.types.closure()` so a typo raises with both readings spelled out rather than silently taking the default.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_remote_writes.py
"""Writes, and the one routing decision that must not be got wrong.

`close="ended"` and `close="retired"` are different statements about whether a stored fact
was ever true. They go to different endpoints and the test that they do is the point of
this file.
"""
import httpx
import pytest

from memvara.remote.api import RemoteMemvara
from memvara.remote.client import HttpClient


@pytest.fixture
def recorded():
    calls = []

    def build(payload=None):
        def handler(request):
            calls.append(request)
            return httpx.Response(200, json=payload if payload is not None else {})

        mem = RemoteMemvara(api_key="k", base_url="https://example.test")
        mem._http = HttpClient("k", "https://example.test", sleep=lambda _: None)
        mem._http._client = httpx.Client(base_url="https://example.test",
                                         transport=httpx.MockTransport(handler))
        return mem

    build.calls = calls
    return build


def test_retiring_goes_to_the_delete_route(recorded):
    mem = recorded({"id": "cl_1", "retired": True, "erased": False})
    mem.delete("cl_1", close="retired")
    assert recorded.calls[-1].method == "DELETE"
    assert recorded.calls[-1].url.path == "/v1/memories/cl_1"


def test_ending_goes_to_the_end_route_and_never_to_delete(recorded):
    mem = recorded({"memory_id": "cl_1", "count": 1, "ended": [], "erased": False})
    mem.delete("cl_1", close="ended")
    assert recorded.calls[-1].url.path == "/v1/end"
    assert recorded.calls[-1].method == "POST"


def test_an_unknown_closure_raises_with_both_readings_named(recorded):
    mem = recorded()
    with pytest.raises(ValueError) as caught:
        mem.delete("cl_1", close="deleted")
    assert "ended" in str(caught.value) and "retired" in str(caught.value)


def test_end_by_slot_sends_subject_and_predicate_and_no_id(recorded):
    mem = recorded({"count": 0, "ended": [], "erased": False})
    mem.end(subject="user", predicate="works_at")
    body = recorded.calls[-1].read().decode()
    assert '"predicate":"works_at"' in body.replace(" ", "")
    assert "memory_id" not in body


def test_end_needs_exactly_one_addressing_mode(recorded):
    mem = recorded()
    with pytest.raises(TypeError):
        mem.end()
    with pytest.raises(TypeError):
        mem.end(claim_id="cl_1", predicate="works_at")


def test_every_write_carries_an_idempotency_key(recorded):
    mem = recorded({"added": [], "episode_ids": []})
    mem.remember("user", "likes", "tea")
    assert recorded.calls[-1].headers.get("idempotency-key")


def test_supersede_forwards_close_verbatim_rather_than_defaulting(recorded):
    mem = recorded({"added": [], "episode_ids": []})
    mem.supersede("cl_old", "user", "lives_in", "Berlin", close="retired")
    body = recorded.calls[-1].read().decode().replace(" ", "")
    assert '"close":"retired"' in body
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=$PWD python -m pytest tests/test_remote_writes.py -v`
Expected: FAIL — `AttributeError: 'RemoteMemvara' object has no attribute 'delete'`

- [ ] **Step 3: Write the implementation**

```python
    # -- writing ---------------------------------------------------------

    def remember(self, subject: str, predicate: str, obj: str, *,
                 confidence: float = 1.0, memory_type: Any = None, polarity: int = 1,
                 valid_from: datetime | None = None, valid_to: datetime | None = None,
                 recorded_at: datetime | None = None,
                 sources: list[str] | None = None, text: str | None = None,
                 extractor: str = "api", **meta: Any) -> Any:
        body = {
            "subject": subject, "predicate": predicate, "object": obj,
            "confidence": confidence, "polarity": polarity, "extractor": extractor,
            "text": self.redactor(text) if self.redactor and text else text,
            "memory_type": memory_type.value if memory_type is not None else None,
            "valid_from": _iso(valid_from), "valid_to": _iso(valid_to),
            "recorded_at": _iso(recorded_at), "source_ids": list(sources or []),
            "metadata": meta,
        }
        return hydrate.receipt(self._http.request(
            "POST", "/v1/facts", params=self._params(),
            json={k: v for k, v in body.items() if v is not None}, write=True))

    def delete(self, claim_id: str, *, at: datetime | None = None,
               close: str = "retired") -> bool:
        """Close one memory by id.

        **Routes on `close`, and the two destinations are not interchangeable.**
        `"retired"` says the record was wrong and goes to `DELETE /v1/memories/{id}`.
        `"ended"` says the world moved on from something true and goes to `POST /v1/end`.
        Sending one to the other's route records a false reason for the change, and
        nothing downstream can detect it — see `memvara/types.py:195`.
        """
        how = closure(close)                      # raises on a typo, with both readings
        if how == "ended":
            return self.end(claim_id=claim_id, at=at)
        body = self._http.request("DELETE", f"/v1/memories/{claim_id}",
                                  params=self._params(), write=True)
        return bool(body.get("retired"))

    def end(self, *, claim_id: str | None = None, subject: str | None = None,
            predicate: str | None = None, at: datetime | None = None) -> bool:
        """Close a fact that stopped being true, with nothing replacing it.

        Exactly one addressing mode: `claim_id` for one memory, or `predicate` (with
        `subject`, default `"user"`) for every current value in that slot. Both or
        neither is a `TypeError`, deliberately — the two have different blast radii and a
        silent default on that choice is not a convenience.
        """
        if (claim_id is None) == (predicate is None):
            raise TypeError(
                "end() needs exactly one of: claim_id, to end one memory, or predicate "
                "(with optional subject), to end every current value of that fact.")
        body: dict[str, Any] = {"at": _iso(at)}
        if claim_id is not None:
            body["memory_id"] = claim_id
        else:
            body["subject"] = subject or "user"
            body["predicate"] = predicate
        out = self._http.request("POST", "/v1/end", params=self._params(),
                                 json={k: v for k, v in body.items() if v is not None},
                                 write=True)
        return bool(out.get("count"))
```

Add `from ..types import closure` to the imports and this helper at module level:

```python
def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()
```

Then write `add`, `supersede`, `forget`, `erase`, `purge` and `consolidate` against the mapping table in the same shape. Every one passes `write=True`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONPATH=$PWD python -m pytest tests/test_remote_writes.py -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Add one test per remaining write in the mapping table**

Same shape: assert the method, the path, and that `Idempotency-Key` is present.

- [ ] **Step 6: Commit**

```bash
git add memvara/remote/api.py tests/test_remote_writes.py
git commit -m "feat(remote): add the write surface, routing ended and retired apart"
```

---

### Task 7: `ScopedRemoteMemvara`

**Files:**
- Modify: `memvara/remote/api.py`
- Test: `tests/test_remote_scope.py`

**Interfaces:**
- Consumes: `RemoteMemvara`.
- Produces: `RemoteMemvara.scope(*, user=None, agent=None, session=None) -> ScopedRemoteMemvara`, exposing every method of `RemoteMemvara` with no scope parameters, plus a `memvara` property returning the underlying client.

The security property to preserve, from `ToolContext`'s docstring: a handler holding a scoped view has no *argument* with which to address another tenant, because the scope was bound once at construction. So no method on `ScopedRemoteMemvara` takes `tenant=`, `user=`, `agent=` or `session=`.

The `memvara` property is required, not a leak — `ScopedMemvara` has one, `tools.py` calls `ctx.memory.memvara`, and Task 11's protocol declares it. It is not a way around the scope: against a hosted deployment the credential itself binds the tenant, so the underlying client cannot address another one either.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_remote_scope.py
"""A scoped view narrows and cannot widen.

`ScopedMemvara` exists so that an MCP handler has no way to address another tenant. The
remote twin has to hold that property the same way or cloud mode is a weaker server than
the local one.
"""
import inspect

import pytest

from memvara.remote.api import RemoteMemvara, ScopedRemoteMemvara


def test_scope_returns_a_scoped_view():
    mem = RemoteMemvara(api_key="k", base_url="https://example.test")
    assert isinstance(mem.scope(user="alice"), ScopedRemoteMemvara)


def test_the_scoped_view_carries_the_narrowed_scope():
    mem = RemoteMemvara(api_key="k", base_url="https://example.test", user="alice")
    assert mem.scope(agent="a1")._scope.agent == "a1"
    assert mem.scope(agent="a1")._scope.user == "alice"


@pytest.mark.parametrize("name", [
    "add", "remember", "recall", "search", "forget", "delete", "end", "get", "get_all",
    "history", "why", "ask", "since", "count", "stats",
])
def test_no_scoped_method_accepts_a_scope_argument(name):
    params = inspect.signature(getattr(ScopedRemoteMemvara, name)).parameters
    assert not {"tenant", "user", "agent", "session"} & set(params)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=$PWD python -m pytest tests/test_remote_scope.py -v`
Expected: FAIL — `ImportError: cannot import name 'ScopedRemoteMemvara'`

- [ ] **Step 3: Write the implementation**

Add `ScopedRemoteMemvara` to `api.py`. It holds a `RemoteMemvara` under `_mem` and a `Scope` under `_scope`, and each method forwards with the bound scope applied. Give `RemoteMemvara.scope()` this docstring:

```python
    def scope(self, *, user: str | None = None, agent: str | None = None,
              session: str | None = None) -> "ScopedRemoteMemvara":
        """A view bound to a narrower scope, with no way back out.

        The narrowing is against this client's own scope, and the facade enforces the
        same rule again from the credential — naming an agent or a session requires
        naming a user, because an agent under *every* user is a read across users dressed
        up as a narrowing.
        """
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONPATH=$PWD python -m pytest tests/test_remote_scope.py -v`
Expected: PASS, 17 tests

- [ ] **Step 5: Commit**

```bash
git add memvara/remote/api.py tests/test_remote_scope.py
git commit -m "feat(remote): add the scoped view"
```

---

### Task 8: The public entry point

**Files:**
- Modify: `memvara/core.py`, `memvara/__init__.py`, `README.md`, `docs/API.md`, `docs/ROADMAP.md`, `CHANGELOG.md`
- Test: `tests/test_remote_constructor.py`

**Interfaces:**
- Consumes: `RemoteMemvara`.
- Produces: `Memvara(api_key=..., base_url=...) -> RemoteMemvara`; `Memvara.connect(*, api_key=None, base_url=None, **scope) -> RemoteMemvara`.

**The safety property this task exists to guarantee:** a bare `Memvara()` must never become remote because the environment happens to hold a key.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_remote_constructor.py
"""How a caller asks for a hosted deployment, and how they cannot ask for one by accident.

The last test is the one that matters. Dispatch keys on the explicit argument and never on
the environment, so a script that has always written to a local file cannot start posting
to a hosted store because someone ran `memvara-mcp login` on that machine last month.
"""
import pytest

from memvara import Memvara, NullLLM
from memvara.remote.api import RemoteMemvara


def test_an_api_key_returns_a_remote_client():
    assert isinstance(Memvara(api_key="k", base_url="https://example.test"),
                      RemoteMemvara)


def test_a_base_url_alone_is_also_a_request_for_a_remote_client(monkeypatch):
    monkeypatch.setenv("MEMVARA_API_KEY", "from-env")
    assert isinstance(Memvara(base_url="https://example.test"), RemoteMemvara)


def test_a_bare_constructor_stays_local_even_when_the_environment_holds_a_key(monkeypatch):
    monkeypatch.setenv("MEMVARA_API_KEY", "from-env")
    monkeypatch.setenv("MEMVARA_SERVER_URL", "https://example.test")
    mem = Memvara(":memory:", llm=NullLLM())
    assert not isinstance(mem, RemoteMemvara)
    mem.close()


def test_connect_is_the_door_for_ambient_credentials(monkeypatch):
    monkeypatch.setenv("MEMVARA_API_KEY", "from-env")
    assert isinstance(Memvara.connect(), RemoteMemvara)


@pytest.mark.parametrize("kwargs", [
    {"path": ":memory:"},
    {"store": object()},
])
def test_credentials_are_refused_alongside_a_local_store(kwargs):
    with pytest.raises(TypeError):
        Memvara(api_key="k", **kwargs)


@pytest.mark.parametrize("name", ["embedder", "llm", "registry"])
def test_server_side_subsystems_are_refused_rather_than_ignored(name):
    with pytest.raises(TypeError) as caught:
        Memvara(api_key="k", base_url="https://example.test", **{name: object()})
    assert name in str(caught.value)


def test_scope_is_passed_through_to_the_remote_client():
    mem = Memvara(api_key="k", base_url="https://example.test", user="alice")
    assert mem.default_scope.user == "alice"


def test_the_constructor_makes_no_network_call():
    # No transport is mocked here. Constructing must not touch the network; if it did,
    # this test would hang or fail against a DNS lookup for a host that does not exist.
    Memvara(api_key="k", base_url="https://nonexistent.invalid")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=$PWD python -m pytest tests/test_remote_constructor.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'api_key'`

- [ ] **Step 3: Add dispatch to `memvara/core.py`**

Insert immediately above `Memvara.__init__`:

```python
    #: Constructor arguments that name a local engine's subsystems. Meaningless against a
    #: hosted deployment, where extraction, embedding and the predicate vocabulary all run
    #: server-side — so they are refused rather than accepted and ignored, which is the
    #: same trade the `path=`/`store=` guard below makes.
    _LOCAL_ONLY = ("path", "store", "embedder", "llm", "registry", "reembed")

    def __new__(cls, path: str | None = None, *, api_key: str | None = None,
                base_url: str | None = None, **kwargs: Any) -> "Memvara":
        """Return a local engine, or a client for a hosted deployment.

        **Dispatch keys on the explicit argument and never on the environment.** If a bare
        `Memvara()` could turn remote because `MEMVARA_API_KEY` happens to be exported,
        then a script that has always written to a local file would start posting to a
        hosted store on any machine where somebody ran `memvara-mcp login`. The
        environment supplies the *value*, once the caller has asked for remote; see
        `Memvara.connect` for the ambient-credential door.

        Returning an object that is not an instance of `cls` means Python does not call
        `__init__`, which is what keeps a half-built local engine from existing here.
        """
        if api_key is None and base_url is None:
            return super().__new__(cls)
        named = [n for n in cls._LOCAL_ONLY
                 if kwargs.get(n) is not None or (n == "path" and path is not None)]
        if named:
            raise TypeError(
                f"{', '.join(named)} cannot be combined with api_key= or base_url=: a "
                "hosted deployment runs extraction, embedding and the predicate "
                "vocabulary itself, so these would be accepted and never used. Pass "
                "either Memvara(path) or Memvara(api_key=...).")
        from .remote.api import RemoteMemvara
        return RemoteMemvara(api_key=api_key, base_url=base_url, **kwargs)

    @classmethod
    def connect(cls, *, api_key: str | None = None, base_url: str | None = None,
                **kwargs: Any) -> Any:
        """A client for a hosted deployment, using whatever credentials are available.

        The door for the case `memvara-mcp login` sets up: it reads `MEMVARA_API_KEY` and
        then `~/.memvara/credentials.json`. `Memvara(api_key=...)` is the same thing with
        the key named explicitly; this exists so that the post-login path has a supported
        spelling rather than sending people to the private resolver.
        """
        from .remote.api import RemoteMemvara
        return RemoteMemvara(api_key=api_key, base_url=base_url, **kwargs)
```

`Memvara.__init__` must also accept and ignore `api_key`/`base_url` in its signature, because Python passes the original arguments to `__init__` whenever `__new__` did return an instance of `cls` — which it does for every local construction. Add them as keyword-only parameters defaulting to `None` and assert they are `None`.

- [ ] **Step 4: Export from `memvara/__init__.py`**

Add `RemoteMemvara`, `AsyncRemoteMemvara` (Task 9 provides it), and the error classes to `__all__`, and reach them through the existing `__getattr__` lazy hook rather than an eager import — `import memvara` must not import `httpx`.

- [ ] **Step 5: Run the tests**

Run: `PYTHONPATH=$PWD python -m pytest tests/test_remote_constructor.py tests/test_core.py -v`
Expected: PASS. `test_core.py` must be green — `__new__` is on the path of every construction in the suite.

- [ ] **Step 6: Update the documentation this changes**

- `README.md` — the surfaces section gains the Python client, with the two-line example.
- `docs/API.md` — `RemoteMemvara`, what is absent and why, and the two divergences.
- `docs/ROADMAP.md` — the entry *"Declined: a REST client library… nobody has asked for that"* moves out of the declined list. Say who asked and when (2026-08-29) and what shipped, in the manner the JavaScript-client entry uses.
- `CHANGELOG.md` — user-visible.

- [ ] **Step 7: Commit**

```bash
git add memvara/core.py memvara/__init__.py tests/test_remote_constructor.py \
        README.md docs/API.md docs/ROADMAP.md CHANGELOG.md
git commit -m "feat: reach a hosted deployment with Memvara(api_key=...)"
```

---

### Task 9: The async client

**Files:**
- Create: `memvara/remote/aio.py`
- Modify: `memvara/aio.py` (docstring), `CHANGELOG.md`
- Test: `tests/test_remote_aio.py`

**Interfaces:**
- Consumes: `RemoteMemvara`'s mapping, `hydrate`.
- Produces: `AsyncRemoteMemvara` and `AsyncScopedRemoteMemvara` with the same method names as their sync twins, each `async def`, plus `aclose()`, `__aenter__`, `__aexit__`.

**This does not follow `memvara/aio.py`'s pattern, and the reason is `aio.py`'s own argument.** That module wraps sync calls in `asyncio.to_thread` because there is no async SQLite and colouring would have to propagate through the whole engine. Neither holds here: `httpx` has a real async client and there is no engine below the transport. Use `httpx.AsyncClient` directly.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_remote_aio.py
"""The async client uses a real async transport, not a thread.

The second test is the one with teeth: it asserts no thread pool is involved, because
wrapping a blocking client in `asyncio.to_thread` would pass every behavioural test here
while being strictly worse than the thing httpx already provides.
"""
import httpx
import pytest

from memvara.remote.aio import AsyncRemoteMemvara

pytestmark = pytest.mark.asyncio


async def _client(handler):
    mem = AsyncRemoteMemvara(api_key="k", base_url="https://example.test")
    mem._http._client = httpx.AsyncClient(base_url="https://example.test",
                                          transport=httpx.MockTransport(handler))
    return mem


async def test_a_read_awaits_and_returns_the_decoded_body():
    mem = await _client(lambda r: httpx.Response(200, json={"claims": 3}))
    assert (await mem.stats())["claims"] == 3
    await mem.aclose()


async def test_the_transport_is_a_real_async_client_and_not_a_thread_wrapper():
    mem = await _client(lambda r: httpx.Response(200, json={}))
    assert isinstance(mem._http._client, httpx.AsyncClient)
    await mem.aclose()


async def test_it_works_as_an_async_context_manager():
    mem = await _client(lambda r: httpx.Response(200, json={}))
    async with mem as m:
        assert m is mem


async def test_every_sync_method_has_an_async_twin_of_the_same_name():
    from memvara.remote.api import RemoteMemvara
    sync = {n for n in dir(RemoteMemvara) if not n.startswith("_")}
    asyn = {n for n in dir(AsyncRemoteMemvara) if not n.startswith("_")}
    missing = sync - asyn - {"close"}
    assert not missing, f"async client is missing: {sorted(missing)}"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=$PWD python -m pytest tests/test_remote_aio.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'memvara.remote.aio'`

- [ ] **Step 3: Write the implementation**

Add `AsyncHttpClient` to `memvara/remote/client.py` — the same retry logic, built on `httpx.AsyncClient`, with `await`ed calls and `asyncio.sleep`. Then write `memvara/remote/aio.py` mirroring `api.py` method for method. The module docstring must state why it does not follow `memvara/aio.py`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONPATH=$PWD python -m pytest tests/test_remote_aio.py -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Amend `memvara/aio.py`'s docstring**

It opens with *"The library is not async, and deliberately stays that way"* and argues from `asyncio.to_thread`. Add a short paragraph saying where that stops applying: the remote client has a genuinely async transport below it and no engine to colour, so `memvara.remote.aio` uses `httpx.AsyncClient` directly rather than this module's pattern.

- [ ] **Step 6: Commit**

```bash
git add memvara/remote/aio.py memvara/remote/client.py memvara/aio.py \
        tests/test_remote_aio.py CHANGELOG.md
git commit -m "feat(remote): add the async client on a native async transport"
```

---

### Task 10: One HTTP layer

**Files:**
- Modify: `memvara/store/remote.py` (`__init__` and `_request` only)
- Test: `tests/test_store_remote.py` (existing — must stay green)

- [ ] **Step 1: Read `memvara/store/remote.py` and run its suite first**

Run: `PYTHONPATH=$PWD python -m pytest tests/test_store_remote.py -v`
Expected: PASS. Establish the baseline before changing anything.

- [ ] **Step 2: Replace its plumbing with `HttpClient`**

`RemoteStore.__init__` builds an `HttpClient` and holds it; `_request` delegates. Keep the class's public behaviour identical, including its own install hint. Change nothing else in the file — the module docstring's argument about which methods raise is unaffected and must not be touched.

- [ ] **Step 3: Run both suites**

Run: `PYTHONPATH=$PWD python -m pytest tests/test_store_remote.py tests/test_remote_client.py -v`
Expected: PASS, both

- [ ] **Step 4: Commit**

```bash
git add memvara/store/remote.py
git commit -m "refactor(store): share one HTTP layer with the remote client"
```

---

### Task 11: The `MemoryAPI` protocol

**Files:**
- Modify: `memvara/server/tools.py` (the `ToolContext.memory` annotation, `_standing`)
- Create: `memvara/server/protocol.py`
- Test: `tests/test_memory_api_protocol.py`

**Interfaces:**
- Consumes: `ScopedMemvara`, `ScopedRemoteMemvara`.
- Produces: `MemoryAPI`, a `typing.Protocol` declaring the eighteen members `tools.py` calls — `add`, `ask`, `count`, `delete`, `forget`, `get`, `get_all`, `history`, `memvara`, `neighborhood`, `paths_between`, `recall`, `remember`, `scope`, `search`, `since`, `stats`, `why` — plus an optional `standing`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_memory_api_protocol.py
"""Both scoped views satisfy the protocol the MCP tools are written against.

Derived from what `tools.py` actually calls rather than from a list somebody maintains: a
name drifting off that list is the failure that matters, because it would let the server
believe a capability exists.
"""
import re
from pathlib import Path

import pytest

from memvara.core import ScopedMemvara
from memvara.remote.api import ScopedRemoteMemvara
from memvara.server.protocol import MemoryAPI

CALLED = set(re.findall(r"ctx\.memory\.([a-z_]+)", Path("memvara/server/tools.py").read_text()))


def test_the_protocol_declares_everything_the_tools_call():
    declared = {n for n in dir(MemoryAPI) if not n.startswith("_")}
    assert CALLED <= declared, f"tools.py calls undeclared members: {sorted(CALLED - declared)}"


@pytest.mark.parametrize("impl", [ScopedMemvara, ScopedRemoteMemvara])
def test_both_implementations_provide_every_declared_member(impl):
    declared = {n for n in dir(MemoryAPI) if not n.startswith("_")}
    missing = {n for n in declared if not hasattr(impl, n)} - {"standing"}
    assert not missing, f"{impl.__name__} is missing: {sorted(missing)}"


def test_standing_is_optional_and_only_the_remote_view_has_it():
    assert hasattr(ScopedRemoteMemvara, "standing")
    assert not hasattr(ScopedMemvara, "standing")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=$PWD python -m pytest tests/test_memory_api_protocol.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'memvara.server.protocol'`

- [ ] **Step 3: Write `memvara/server/protocol.py`**

A `typing.Protocol` with the eighteen members, each with the signature `ScopedMemvara` already has. The module docstring explains that the protocol is what lets one tool table serve a local engine and a hosted deployment without a second table.

- [ ] **Step 4: Retype `ToolContext.memory` and extend its docstring**

Change the annotation from `ScopedMemvara` to `MemoryAPI`. The existing docstring says *"`memory` is a `ScopedMemvara`, never an `Memvara`, and that is the whole security model of this server."* Rewrite that sentence to name the property rather than the class: scope is bound once at construction, so a handler has no argument and no attribute with which to address another tenant — which both implementations hold, the remote one by sending a bound scope the credential must already contain.

- [ ] **Step 5: Make `_standing` prefer the endpoint**

```python
    # `GET /v1/standing` does this filter server-side. Against a hosted deployment the
    # fallback below pages every live memory in the scope across the network to keep the
    # procedural ones -- and this is the tool a session calls at startup. Local behaviour
    # is unchanged: `ScopedMemvara` has no `standing`, so it takes the same path it always
    # did.
    server_side = getattr(ctx.memory, "standing", None)
    if server_side is not None:
        claims = list(server_side(k=cap))
    else:
        claims = [c for c in ctx.memory.get_all(states=["live"])
                  if c.memory_type is MemoryType.PROCEDURAL]
```

Leave the sort in place and applying to both branches: order is confidence, then recency, then id, and the last makes the order total so two claims written in the same instant cannot swap places between calls.

- [ ] **Step 6: Run the tests**

Run: `PYTHONPATH=$PWD python -m pytest tests/test_memory_api_protocol.py tests/test_tools.py tests/test_mcp.py -v`
Expected: PASS. The tool tests must be green — `_standing` changed.

- [ ] **Step 7: Commit**

```bash
git add memvara/server/protocol.py memvara/server/tools.py \
        tests/test_memory_api_protocol.py
git commit -m "feat(server): type the tool context to a protocol both engines satisfy"
```

---

### Task 12: Cloud mode

**Files:**
- Modify: `memvara/server/config.py`, `tests/test_config_cloud.py`, `docs/OPEN-CORE.md`, `docs/UPGRADING.md`, `CHANGELOG.md`
- Test: `tests/test_remote_cloud_mode.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_remote_cloud_mode.py
"""`MEMVARA_MODE=cloud` starts a server that serves its tools from a hosted deployment.

The guard this replaces refused at construction because a `RemoteStore`-backed engine
would list fourteen tools and fail on the first one reached for. That reasoning was right
and is not being overturned: the engine is still never run against a remote store. The
server is now a client of the facade instead, which is what `docs/OPEN-CORE.md` said the
answer was.
"""
import pytest

from memvara.remote.api import RemoteMemvara
from memvara.server.config import ConfigError, ServerConfig, build_memvara


def _cloud(**kw):
    return ServerConfig(mode="cloud", api_key="k",
                        server_url="https://example.test", **kw)


def test_cloud_mode_builds_a_remote_client():
    assert isinstance(build_memvara(_cloud()), RemoteMemvara)


def test_cloud_mode_no_longer_raises_about_unwired_store_methods():
    build_memvara(_cloud())          # must not raise


def test_a_cloud_config_without_a_key_still_fails_at_construction():
    with pytest.raises(ConfigError):
        build_memvara(ServerConfig(mode="cloud", api_key=None,
                                   server_url="https://example.test"))


@pytest.mark.parametrize("field, value", [("llm", "anthropic"), ("embedder", "local")])
def test_naming_a_server_side_subsystem_under_cloud_mode_is_refused(field, value):
    with pytest.raises(ConfigError) as caught:
        build_memvara(_cloud(**{field: value}))
    assert field in str(caught.value).lower()


def test_the_scope_reaches_the_client():
    assert build_memvara(_cloud(user="alice")).default_scope.user == "alice"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=$PWD python -m pytest tests/test_remote_cloud_mode.py -v`
Expected: FAIL — `ConfigError: MEMVARA_MODE=cloud cannot start a server yet...`

- [ ] **Step 3: Rewrite the cloud branch of `build_memvara`**

Delete the `cloud_gap()` check and `_CLOUD_NOT_WIRED`. Keep the `api_key is None` check. Build a `RemoteMemvara` from `config.server_url`, `config.api_key` and `config.scope_kwargs`. Raise `ConfigError` naming `MEMVARA_LLM` or `MEMVARA_EMBEDDER` if either was set to a non-default under cloud mode.

Also delete `cloud_gap()` and `_ENGINE_NEEDS`. Their docstrings record why the guard existed and that it was built to un-refuse itself when `RemoteStore.WIRED` grew — which never happens now, because this bypasses the `Store` seam rather than completing it. Removing them deliberately is the point; leaving a dead set difference in place would tell the next reader the gate is still live.

- [ ] **Step 4: Replace the guard's test in `tests/test_config_cloud.py`**

`test_the_cloud_guard_is_derived_from_the_store_rather_than_hardcoded` asserts a guard that no longer exists. Replace it with a test asserting the thing that now has to stay true: cloud mode builds a client of the facade and never a `Memvara` over a `RemoteStore`. Do not simply delete it — a removed test is a removed guarantee.

- [ ] **Step 5: Run the tests**

Run: `PYTHONPATH=$PWD python -m pytest tests/test_remote_cloud_mode.py tests/test_config_cloud.py tests/test_config.py -v`
Expected: PASS

- [ ] **Step 6: Update the documentation this makes wrong**

- `docs/OPEN-CORE.md` — one sentence is now false: *"A hosted deployment is reached by pointing an MCP client at its own URL. It is not proxied through a local server."* Rewrite it. The seam table's last row, *"running the engine against a remote store — neither, for now"*, stays exactly as it is; add a row for the facade client. Replace the paragraph describing the refusal and how it un-refuses itself with what actually happened.
- `docs/UPGRADING.md` — `MEMVARA_MODE=cloud` now starts. Say what changes for anyone who had configured it and been refused.
- `CHANGELOG.md`.

- [ ] **Step 7: Run the full suite**

```bash
PYTHONPATH=$PWD python -m pytest -q
```

Expected: PASS, no new failures. Note the pre-existing `platform_grants` flake — a wall-clock assertion, not contention. Do not re-run past it or try to fix it here.

- [ ] **Step 8: Commit**

```bash
git add memvara/server/config.py tests/test_config_cloud.py \
        tests/test_remote_cloud_mode.py docs/OPEN-CORE.md docs/UPGRADING.md CHANGELOG.md
git commit -m "feat(server): serve cloud mode from the facade instead of refusing"
```

---

### Task 13: The packaged skill

**Files:**
- Modify: `memvara/skills/memvara/SKILL.md`

**This gets its own commit and touches nothing else.** `memvara/skills/memvara/` is vendored into seven downstream plugin repositories that pin it by sha and diff against it in CI. A drive-by edit here is a change in all of them.

- [ ] **Step 1: Read the surfaces section**

It currently says: *"A loop in any other language: hosted MCP as a client, or the commercial REST API."* And the skill's own rule is that it does not repeat what a tool description says, so text moving between the two has to move in both directions.

- [ ] **Step 2: Rewrite the surface guidance**

Python callers now have a fourth answer, and it is the right one for application code. Say when to reach for the library against a local file, when for `Memvara(api_key=...)`, and when for MCP. Keep it to the length the surrounding entries use.

- [ ] **Step 3: Verify the packaged skill still ships**

Run: `PYTHONPATH=$PWD python -m pytest tests/test_packaging.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add memvara/skills/memvara/SKILL.md
git commit -m "docs(skill): name the Python client as a surface"
```

---

### Task 14: The public API reference on the website

**Repository:** `memvara-web` at `/Applications/workstation/memvara-web`. **Not this one.**

- Modify: `src/content/api.ts`

Missed when this plan was first written, and confirmed by inspection afterwards.
`memvara-web` documents this library's public API in `src/content/api.ts`, rendered by
`src/routes/ApiReference.tsx`. It enumerates the library's own methods — `Memvara(...)`,
`mem.add`, `mem.remember`, `mem.supersede`, `AsyncMemvara`, `scope()` — grouped as *Open a
store* / *Write* / *Read*. A new public entry point belongs there.

Also confirmed: `memvara-web` needs **nothing** for the `memvara-cloud` half of this work.
The site documents the library, not the REST surface; the only `/v1` routes it names are
three GET routes in `Guide.tsx`, for time-travel parameters. A `CloudApi.tsx` exists but only
inside a feature worktree, not on `main`.

- [ ] **Step 1: Work in a fresh worktree of `memvara-web`**

That repository's `main` had uncommitted changes to `worker/generated/searchIndex.ts` when
this was checked — somebody else's work. Branch in a worktree; do not touch `main`.

- [ ] **Step 2: Add the two entry points under *Open a store***

`Memvara(api_key=...)` and `Memvara.connect()`, in the shape the existing `api-memvara`
entry uses. Say plainly that dispatch keys on the explicit argument and never on the
environment, because that is the property a reader most needs and least expects.

- [ ] **Step 3: Record the divergences where the affected methods are described**

`reembed`, `pending_extraction`, `reextract` and `reset` do not exist on a remote client;
`consolidate()` returns a job handle; there is no `prove_erased()`. Put each note against the
method it concerns rather than in one list, matching how that file already attaches notes.

- [ ] **Step 4: Run the examples**

That page asserts its examples produced the output shown. Run each new one against a real
deployment and paste what it actually printed. A plausible-looking transcript on a page whose
whole claim is that the outputs are real is worse than no example.

- [ ] **Step 5: Commit and open a PR in that repository**

Files by name. No AI attribution.

---

## Final gate

- [ ] **Full suite, with a private coverage file, as two commands**

```bash
PYTHONPATH=$PWD COVERAGE_FILE=.coverage.remote python -m coverage run -m pytest -q
```

```bash
COVERAGE_FILE=.coverage.remote python -m coverage report --include="memvara/remote/*"
```

Running these as one command exits 139 and kills the report, which reads as success.

- [ ] **Type check**

```bash
PYTHONPATH=$PWD python -m mypy memvara/remote memvara/core.py memvara/server
```

- [ ] **Confirm the core stays importable without httpx**

```bash
PYTHONPATH=$PWD python -c "import sys; sys.modules['httpx']=None; import memvara; print(memvara.__version__)"
```

Expected: the version prints. If this fails, an `httpx` import escaped to module level.

- [ ] **Open the PR, then review it, then fix what the review finds**

```bash
/code-review high <PR number>
```

Run it on `claude-sonnet-5` — switch the session model before and back after, or name the reviewing model in the PR body. Use `high`, not `ultra`: `ultra` is user-triggered and an agent cannot launch it. Fix everything it finds on the same branch and re-run. Where a finding is wrong, write the reason in the PR body.

The PR body must state that `end()` and write retries depend on `POST /v1/end` and `Idempotency-Key` landing in `memvara-cloud`, and link the spec. A deferral somebody can see is a deferral; a silent one is a defect with a delay on it.

---

## Spec coverage

| spec section | tasks |
|---|---|
| §1 construction and credentials | 2, 8 |
| §2 the surface | 5, 6, 7 |
| §3 transport, hydration, errors, retries, async | 1, 3, 4, 9, 10 |
| §4 un-refusing cloud mode | 11, 12 |
| §5 memvara-cloud changes | **not in this plan** — separate repo, separate plan; `local/END-ENDPOINT-SPEC.md` |
| §6 verification | every task's test steps, plus the final gate |
| §7 documentation | folded into 3, 8, 9, 12, 13; plus Task 14 (`memvara-web`) |
| §8 deliberately not in scope | no tasks, by design |
