# Adversarial suite F3: the fakes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the test doubles that later workstreams of the adversarial suite need: a fake of the hosted `/v1` REST API, a fake of the hosted `/mcp` endpoint, a fake OpenAI-compatible model endpoint, and fake `claude` and `codex` executables for the capture hook. Each fake is checked by driving it with the real client code it stands in for.

**Architecture:**

- **One shared mechanism.** `tests/harness/fakes/_http.py` holds what the three HTTP fakes share: a request log, per-route faults (`fail`, `delay`, `hang`), and three ways to reach a fake (`transport()` for an `httpx.Client`, `async_transport()` for an `httpx.AsyncClient`, and `serve()` on 127.0.0.1 in a thread).
- **Real code behind each fake.** `FakeV1` answers from a real local `Memvara` and renders like memvara-cloud's `rest/render.py`. `FakeHostedMcp` answers with the real `MemvaraMCPServer`. `FakeOpenAI` answers from a script. The fake CLIs are shell scripts that run one standard-library program.
- **Self-tests drive the real clients.** `RemoteMemvara`, `AsyncRemoteMemvara`, a cloud-mode server process, the hooks' `lib/hosted.py`, the npm bridge, `OpenAILLM`, and the capture hook's `lib/extract.py` and `lib/agentic.py`. The routes `FakeV1` serves are compared with the routes the two remote clients call, read from their source.

**Tech Stack:** Python 3.10 to 3.13, pytest, httpx (the `cloud` extra CI installs), `http.server` and `threading` from the standard library, and node for the npm bridge test when it is installed.

**Spec:** `docs/superpowers/specs/2026-09-25-adversarial-test-suite-design.md`: the `fakes` bullets of "The shared foundation", and the F3 row of "Phase 0: Foundation". The workstream brief adds what the self-tests must show.

## Global Constraints

- **Platforms.** Python `>=3.10`. CI runs 3.10 to 3.13 on Ubuntu, and 3.13 on macOS and Windows. Every test here must pass on all of them or skip under a rule in `tests/harness/skips.py`.
- **Offline.** Every fake listens on 127.0.0.1 only. No test reaches the network. `FakeV1` refuses a document added by `url` rather than fetch it.
- **Child processes** get their environment from `harness.env.child_env`, never the real home directory.
- **Embedder.** Every `Memvara(...)` built in `tests/` passes `embedder=`, or `tests/conftest.py` fails the run.
- **Skips.** Every skip reason matches a rule in `tests/harness/skips.py`.
- **Deprecations are errors** (`filterwarnings = ["error::DeprecationWarning"]`).
- **Fast tier.** Every test here is in the fast tier, and each test file takes a few seconds at most.
- **Type checks.** `python -m mypy tests/harness` passes with and without `--ignore-missing-imports`.
- **Prose.** Docstrings, comments, docs and commit messages are plain English that a reader with no context understands on the first read.
- **No AI attribution** anywhere: no trailer, no "generated with" line, no model name.
- **Commits** name their files. Never `git add -A`, `git add .` or `git commit -a`. Never stash. Documentation ships in the same commit as the code it describes.
- **Scope.** This plan owns `tests/harness/fakes/*`, `tests/adversarial/fakes/*`, this plan file, and the fakes section of `docs/claude/testing.md`. It adds one rule to `tests/harness/skips.py` for the Windows skip of the fake CLIs. It does not change `HookRunner`'s refusal of `run("capture")`; the hook-conformance workstream lifts it.
- **Bugs found in memvara** are classified against `SECURITY.md`'s "In scope" section before anything is written. A security-class bug is reported only in the final message. Any other bug is reported with an offline reproduction, and its failing test stays out of the commits.

## Review Focus

1. **A hung request must never hang the suite.** Every hang ends at the client's timeout, or when the fake closes. Pinned by `test_over_a_socket_a_hang_ends_at_the_client_s_timeout_and_close_releases_it` and `test_over_a_mock_transport_a_hang_or_a_long_delay_ends_at_the_client_s_timeout` (Task 1).
2. **A typo in a route name must not inject a fault that never fires**, because the test would then pass without testing anything. Pinned by `test_a_fault_on_a_route_the_fake_does_not_have_is_refused` (Task 1).
3. **A route a client calls and the fake does not serve.** The fake would drift from the client with no signal. Pinned by `test_the_fake_serves_exactly_the_routes_the_clients_call` (Task 2), which reads the clients' source.
4. **A retried write that lands twice.** A fake without idempotency would report a duplicate that the real deployment does not produce. Pinned by `test_a_write_retried_after_its_first_attempt_timed_out_lands_once` (Task 2).
5. **A test that makes one call more than it scripted must fail loudly.** Pinned by `test_replies_come_back_in_the_order_they_were_scripted` (Task 4, a 500 that says so) and `test_a_run_with_no_reply_left_fails_loudly` (Task 5, exit status 3).

## File structure

| File | Responsibility |
|---|---|
| `tests/harness/fakes/__init__.py` | Says what each fake is. Imports nothing. |
| `tests/harness/fakes/_http.py` | The request log, faults and three transports that the HTTP fakes share. |
| `tests/harness/fakes/fake_v1.py` | `FakeV1`: the 34 `/v1` routes, the renderer, the refusals, idempotency. |
| `tests/harness/fakes/hosted_mcp.py` | `FakeHostedMcp`: `/mcp` with auth, sessions, 202s, the project header, event streams. |
| `tests/harness/fakes/openai_compat.py` | `FakeOpenAI`: the scripted chat-completions endpoint and an SDK-shaped client. |
| `tests/harness/fakes/cli.py` | `FakeClis`: fake `claude` and `codex` executables with scripts and a call log. |
| `tests/adversarial/fakes/__init__.py`, `conftest.py` | The test package, and one fixture per fake. |
| `tests/adversarial/fakes/test_adv_fake_http.py` | The shared mechanism, through a minimal `Echo` fake. |
| `tests/adversarial/fakes/test_adv_fake_v1_parity.py` | Routes, parity with a local store, the round trip, a tour of every route, the async client. |
| `tests/adversarial/fakes/test_adv_fake_v1_faults.py` | Faults through the real client, over both transports, and a cloud-mode server process. |
| `tests/adversarial/fakes/test_adv_fake_hosted_mcp.py` | The hooks' hosted client and the npm bridge against `FakeHostedMcp`. |
| `tests/adversarial/fakes/test_adv_fake_openai.py` | `OpenAILLM` and a store with a model against `FakeOpenAI`. |
| `tests/adversarial/fakes/test_adv_fake_cli.py` | The capture hook's extraction code, in a child process, against the fake CLIs. |

The brief names one self-test file, `tests/adversarial/test_adv_fakes.py`, to be split into `tests/adversarial/fakes/test_adv_*.py` once it passes about 400 lines. The tests below come to well over 400 lines, so they start split, one file per fake.

## Commands

Every command runs from the worktree root, `/Applications/workstation/agent-memory/.claude/worktrees/agent-a9a81dd973c76f719`, written `$W` below, with the CI virtual environment's Python, written `$PY`:

```bash
W=/Applications/workstation/agent-memory/.claude/worktrees/agent-a9a81dd973c76f719
PY=/Applications/workstation/agent-memory/.claude/worktrees/friendly-einstein-53c8da/local/venv-ci/bin/python
export PYTHONPATH=$W TMPDIR=/private/tmp/claude-501/f3-tmp
```

`TEST <paths>` below means `$PY -m pytest -q -p no:cacheprovider <paths>`.

---

## Task 0: Commit this plan

- [ ] **Step 1:** Create the branch from `origin/main` and remove its upstream, so a stray push cannot reach `main`.

```bash
git fetch origin
git switch -c claude/adversarial-fakes origin/main
git branch --unset-upstream
```

- [ ] **Step 2:** Commit this file by name.

```bash
git add docs/superpowers/plans/2026-09-26-adversarial-fakes.md
git commit -m "Plan the adversarial suite's fakes: the hosted APIs, a model endpoint and the agent CLIs"
```

---

## Task 1: The shared mechanism

**Files:**
- Create: `tests/harness/fakes/__init__.py`, `tests/harness/fakes/_http.py`
- Create: `tests/adversarial/fakes/__init__.py`, `tests/adversarial/fakes/test_adv_fake_http.py`
- Modify: `docs/claude/testing.md` (a new section just before the line that starts with `Next:`)

**Interfaces:**
- Produces: `Request` (method, raw path, query, headers, body, `route`, `params`, `status`, `replayed`; `param()`, `header()`, `json()`), `Reply(status, body, headers)`, `json_reply(status, body, headers)`, `Step(hang, wait, reply)`, and `HttpFake` with `ROUTES`, `MOCK_URL`, `requests`, `route()`, `respond()`, `error_body()`, `plan()`, `fail()`, `delay()`, `hang()`, `clear_faults()`, `transport()`, `async_transport()`, `serve()`, `close()`, and the context manager.

- [ ] **Step 1: Write the failing test.** Create `tests/adversarial/fakes/__init__.py` holding one docstring line, and the test file:

`tests/adversarial/fakes/test_adv_fake_http.py`:

```python
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
```

- [ ] **Step 2: Run it and see it fail** because the module does not exist.

Run: `TEST tests/adversarial/fakes/test_adv_fake_http.py`
Expected: a collection error, `ModuleNotFoundError: No module named 'harness.fakes'`.

- [ ] **Step 3: Write the package and the mechanism.**

`tests/harness/fakes/__init__.py`:

```python
"""Test doubles for the services a memvara client talks to. Nothing in this package is a
test, and importing it has no side effects.

Each fake stands in for one thing a client reaches over a network or starts as a
program, and each is checked in tests/adversarial/fakes/ by driving it with the real
client code it stands in for. A fake that drifts from what its client sends or reads
therefore fails its own tests before it can mislead any other test.

* `fake_v1.FakeV1` is the hosted `/v1` REST API that `memvara.remote` calls, answered by
  a real local `Memvara`.
* `hosted_mcp.FakeHostedMcp` is the hosted `/mcp` endpoint that the plugin's hooks and
  the npm bridge call, answered by the real MCP server code.
* `openai_compat.FakeOpenAI` is an OpenAI-compatible chat-completions endpoint that
  returns scripted replies.
* `cli.FakeClis` writes executables named `claude` and `codex` for the capture hook.

`_http` holds what the three HTTP fakes share. Every server here listens on 127.0.0.1
only, so nothing reaches the network. docs/claude/testing.md explains how to use them.
"""
```

`tests/harness/fakes/_http.py`:

```python
"""What the HTTP fakes share: a request log, injected faults, and three ways to reach one.

A test reaches a fake in whichever of three ways its client needs:

* `transport()` is an `httpx.MockTransport` for an `httpx.Client`. Nothing listens on a
  socket, so it is the fastest way for a test in this process to reach a fake.
* `async_transport()` is the same for an `httpx.AsyncClient`.
* `serve()` answers on 127.0.0.1 from a background thread and returns the base URL, for a
  child process, or for a client that does not use httpx.

A test can inject three faults into a named route, for every request to it or for the
next `times` of them:

* `fail(route, status)` answers with that status and does nothing else;
* `delay(route, seconds)` waits, then answers as usual;
* `hang(route)` never answers.

A delay or a hang has to reach the client the way a slow server would reach it, whichever
way the fake is reached. Over a socket that happens by itself: the client's own timeout
fires. A mock transport has no network under it, so there the timeout is emulated. A hang,
or a delay at least as long as the request's read timeout, waits for that timeout and then
raises `httpx.ReadTimeout`, which is what the client would have seen. One difference
remains, and a test has to choose for it: over a mock transport a request the client gave
up on is never carried out, while over a socket it is carried out late, after the client
has stopped waiting, as on a real server.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping, TypeVar

import httpx

_Fake = TypeVar("_Fake", bound="HttpFake")


@dataclass
class Request:
    """One request a fake received."""

    method: str
    #: The path as it arrived, still percent-encoded, without the query string.
    path: str
    #: Every query parameter, each with all of its values in order.
    query: dict[str, list[str]]
    #: Header names in lower case.
    headers: dict[str, str]
    body: bytes
    #: The route the fake matched, or None for a request it has no route for.
    route: str | None = None
    #: The matched route's path parameters, percent-decoded.
    params: dict[str, str] = field(default_factory=dict)
    #: The status the fake answered with. None until it answers, and for good when the
    #: request hung or the client gave up before the answer.
    status: int | None = None
    #: True when the fake answered with the reply it stored for an earlier request, as a
    #: retried write that carries the same idempotency key is answered.
    replayed: bool = False

    @classmethod
    def parse(cls, method: str, target: str, headers: Mapping[str, str],
              body: bytes) -> Request:
        """A request from its method, its request target (path and query), its headers
        and its body."""
        path, _, query = target.partition("?")
        return cls(method=method.upper(), path=path,
                   query=urllib.parse.parse_qs(query, keep_blank_values=True),
                   headers={name.lower(): value for name, value in headers.items()},
                   body=body)

    def param(self, name: str) -> str | None:
        """The first value of the query parameter `name`, or None when it was not sent."""
        values = self.query.get(name)
        return values[0] if values else None

    def header(self, name: str) -> str | None:
        return self.headers.get(name.lower())

    def json(self) -> Any:
        """The body decoded as JSON, or None when the body is empty. Raises ValueError
        when the body is not JSON."""
        return json.loads(self.body) if self.body else None


@dataclass(frozen=True)
class Reply:
    """One answer a fake sends."""

    status: int
    body: bytes = b""
    headers: Mapping[str, str] = field(default_factory=dict)


def json_reply(status: int, body: Any, headers: Mapping[str, str] | None = None) -> Reply:
    """`body` sent as JSON with `status`."""
    return Reply(status, json.dumps(body).encode("utf-8"),
                 {"Content-Type": "application/json", **(headers or {})})


@dataclass(frozen=True)
class Step:
    """What a fake does with one request before it answers. The default is nothing."""

    #: Never answer.
    hang: bool = False
    #: Wait this many seconds, then answer.
    wait: float = 0.0
    #: Answer with this instead of asking the fake.
    reply: Reply | None = None


@dataclass
class _Fault:
    kind: str                  # "fail", "delay" or "hang"
    times: int | None          # None means every matching request
    reply: Reply | None = None
    seconds: float = 0.0


class HttpFake:
    """The part every HTTP fake shares.

    A subclass lists its route names in `ROUTES`, names the route a request addresses in
    `route()`, and answers it in `respond()`. Everything else is here: the request log,
    the faults, the three transports and closing.
    """

    #: The route names a test may inject a fault into, as `route()` spells them.
    ROUTES: tuple[str, ...] = ()
    #: The origin a client reached through `transport()` believes it is talking to.
    #: Nothing ever resolves it.
    MOCK_URL = "http://fake.invalid"

    def __init__(self) -> None:
        #: Every request received, in the order it arrived.
        self.requests: list[Request] = []
        self._lock = threading.Lock()
        self._faults: dict[str, list[_Fault]] = {}
        #: Set by `close()`. It releases every request that is hanging or waiting.
        self._closed = threading.Event()
        self._server: _Server | None = None
        self._thread: threading.Thread | None = None

    # -- what a subclass provides ---------------------------------------------------

    def route(self, request: Request) -> str | None:
        """The name of the route `request` addresses, or None when there is none. It may
        also fill in `request.params`."""
        raise NotImplementedError

    def respond(self, request: Request) -> Reply:
        """The answer to `request`, when no fault replaces it."""
        raise NotImplementedError

    def error_body(self, status: int, route: str) -> Any:
        """The body an injected `fail()` sends when the test names none."""
        return {"error": f"{type(self).__name__} injected a {status} on {route}"}

    # -- faults ---------------------------------------------------------------------

    def fail(self, route: str, status: int, *, body: Any = None,
             headers: Mapping[str, str] | None = None, times: int | None = None) -> None:
        """Answer requests to `route` with `status`, and do nothing else for them.

        `body` is sent as JSON when it is a dict or a list, and exactly as given when it is
        a `str` or `bytes`, which is how a proxy's HTML error page is sent. Left out, it is
        this fake's own error body for `status`.
        """
        self._check(route)
        if body is None:
            body = self.error_body(status, route)
        if isinstance(body, (str, bytes)):
            raw = body.encode("utf-8") if isinstance(body, str) else body
            reply = Reply(status, raw, dict(headers or {}))
        else:
            reply = json_reply(status, body, headers)
        self._add(route, _Fault("fail", times, reply=reply))

    def delay(self, route: str, seconds: float, *, times: int | None = None) -> None:
        """Wait `seconds` before answering requests to `route`."""
        self._check(route)
        self._add(route, _Fault("delay", times, seconds=seconds))

    def hang(self, route: str, *, times: int | None = None) -> None:
        """Never answer requests to `route`. The fake does not act on them either: each
        one is held until the client gives up or the fake closes."""
        self._check(route)
        self._add(route, _Fault("hang", times))

    def clear_faults(self) -> None:
        """Remove every fault that has not been used up."""
        with self._lock:
            self._faults.clear()

    def _check(self, route: str) -> None:
        # A typo in a route name would inject a fault that never fires, and the test would
        # pass without testing anything.
        if route not in self.ROUTES:
            raise ValueError(f"{type(self).__name__} has no route {route!r}. Its routes "
                             f"are: {', '.join(self.ROUTES)}")

    def _add(self, route: str, fault: _Fault) -> None:
        with self._lock:
            self._faults.setdefault(route, []).append(fault)

    def plan(self, request: Request) -> Step:
        """What to do with `request` before answering it: the oldest fault queued on its
        route, which is used up after its `times`. A subclass may add its own reasons."""
        with self._lock:
            queue = self._faults.get(request.route or "")
            if not queue:
                return Step()
            fault = queue[0]
            if fault.times is not None:
                fault.times -= 1
                if fault.times <= 0:
                    queue.pop(0)
        if fault.kind == "hang":
            return Step(hang=True)
        if fault.kind == "delay":
            return Step(wait=fault.seconds)
        return Step(reply=fault.reply)

    # -- one request, whichever transport carried it --------------------------------

    def _receive(self, request: Request) -> Step:
        request.route = self.route(request)
        with self._lock:
            self.requests.append(request)
        return self.plan(request)

    def _answer(self, request: Request, step: Step) -> Reply:
        reply = step.reply if step.reply is not None else self.respond(request)
        request.status = reply.status
        return reply

    # -- the three transports -------------------------------------------------------

    def transport(self) -> httpx.MockTransport:
        """An `httpx.MockTransport` that answers from this fake, for an `httpx.Client`."""

        def handle(outgoing: httpx.Request) -> httpx.Response:
            request = _from_httpx(outgoing)
            step = self._receive(request)
            limit = _read_timeout(outgoing)
            if step.hang or (limit is not None and step.wait >= limit):
                if limit is None:
                    self._closed.wait()
                else:
                    time.sleep(limit)
                raise httpx.ReadTimeout("the fake did not answer in time", request=outgoing)
            if step.wait:
                time.sleep(step.wait)
            return _to_httpx(self._answer(request, step))

        return httpx.MockTransport(handle)

    def async_transport(self) -> httpx.MockTransport:
        """`transport()` for an `httpx.AsyncClient`. It waits with `asyncio.sleep`, so a
        delay or a hang does not block the event loop."""

        async def handle(outgoing: httpx.Request) -> httpx.Response:
            request = _from_httpx(outgoing)
            step = self._receive(request)
            limit = _read_timeout(outgoing)
            if step.hang or (limit is not None and step.wait >= limit):
                if limit is None:
                    while not self._closed.is_set():
                        await asyncio.sleep(0.05)
                else:
                    await asyncio.sleep(limit)
                raise httpx.ReadTimeout("the fake did not answer in time", request=outgoing)
            if step.wait:
                await asyncio.sleep(step.wait)
            return _to_httpx(self._answer(request, step))

        return httpx.MockTransport(handle)

    def serve(self) -> str:
        """Answer on 127.0.0.1 from a background thread, and return the base URL.

        Calling it again returns the same URL. `close()` stops it.
        """
        with self._lock:
            if self._server is None:
                self._server = _Server(self)
                self._thread = threading.Thread(
                    target=self._server.serve_forever,
                    name=f"{type(self).__name__} on 127.0.0.1", daemon=True)
                self._thread.start()
            host, port = self._server.server_address[:2]
        return f"http://{host!s}:{port}"

    def close(self) -> None:
        """Release every request that is hanging or waiting, and stop serving. Calling it
        twice is harmless."""
        self._closed.set()
        with self._lock:
            server, thread = self._server, self._thread
            self._server = self._thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=5)

    def __enter__(self: _Fake) -> _Fake:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _from_httpx(outgoing: httpx.Request) -> Request:
    return Request.parse(outgoing.method, outgoing.url.raw_path.decode("ascii"),
                         dict(outgoing.headers.items()), outgoing.read())


def _to_httpx(reply: Reply) -> httpx.Response:
    return httpx.Response(reply.status, headers=dict(reply.headers), content=reply.body)


def _read_timeout(outgoing: httpx.Request) -> float | None:
    """How long the client waits for an answer, or None when it waits forever."""
    timeout = outgoing.extensions.get("timeout") or {}
    value = timeout.get("read")
    return None if value is None else float(value)


class _Server(ThreadingHTTPServer):
    """One fake's socket on 127.0.0.1. Each request runs on its own daemon thread, so a
    request that hangs never holds up the next one or the end of the test run."""

    daemon_threads = True

    def __init__(self, fake: HttpFake) -> None:
        self.fake = fake
        super().__init__(("127.0.0.1", 0), _Handler)

    def handle_error(self, request: Any, client_address: Any) -> None:
        # A client that gave up first closes its socket, and answering it then fails. That
        # is what a hang or a long delay is for, so it is not worth a traceback.
        if isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            return
        super().handle_error(request, client_address)


class _Handler(BaseHTTPRequestHandler):
    # HTTP/1.1 keeps a connection open between requests, as a real deployment does, so a
    # client that holds a connection, such as the hooks' hosted client, reuses it.
    protocol_version = "HTTP/1.1"
    server: _Server

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - the base's name
        """Stay quiet: a test's output is not the place for an access log."""

    def _handle(self) -> None:
        fake = self.server.fake
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        request = Request.parse(self.command, self.path, dict(self.headers.items()), body)
        step = fake._receive(request)
        if step.hang:
            fake._closed.wait()
            self.close_connection = True
            return
        if step.wait and fake._closed.wait(step.wait):
            self.close_connection = True
            return
        reply = fake._answer(request, step)
        self.send_response(reply.status)
        for name, value in reply.headers.items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(reply.body)))
        self.end_headers()
        self.wfile.write(reply.body)

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _handle
```

- [ ] **Step 4: Run the test and see it pass.**

Run: `TEST tests/adversarial/fakes/test_adv_fake_http.py`
Expected: `9 passed`.

- [ ] **Step 5: Document it.** Add this section to `docs/claude/testing.md`, just before the line that starts with `Next:`:

```markdown
## Fakes for the services a client talks to

`tests/harness/fakes/` holds test doubles for what a memvara client reaches over a network or starts as a program: the hosted REST API, the hosted MCP endpoint, an OpenAI-compatible model, and the agent CLIs that the capture hook runs. With them a test drives the real client code offline, with no login and no bill. Every fake listens on 127.0.0.1 only. Each one is checked in `tests/adversarial/fakes/` by driving it with the real client it stands in for, so a fake that drifts from what its client sends or reads fails its own tests first.

**The three HTTP fakes share one mechanism,** in `fakes/_http.py`:

- **A test reaches a fake in one of three ways.** `transport()` is an `httpx.MockTransport` for an `httpx.Client`, and `async_transport()` is the same for an `httpx.AsyncClient`. `serve()` answers on 127.0.0.1 from a background thread and returns the base URL, for a child process or for a client that does not use httpx. `close()` stops a fake, and each fake is a context manager that closes itself.
- **Every request is recorded** in `fake.requests`, with its method, raw path, query, headers and body, the route it matched, and the status it was answered with.
- **A test can inject a fault into one route,** for every request or for the next `times`: `fail(route, status)` answers with that status and does nothing else, `delay(route, seconds)` waits and then answers, and `hang(route)` never answers. A route name the fake does not have is refused, so a typo cannot inject a fault that never fires.
- **A slow answer reaches the client the way a real server's would.** Over a socket, the client's own timeout fires. A mock transport has no network under it, so there a hang, or a delay at least as long as the request's read timeout, waits out that timeout and then raises `httpx.ReadTimeout`. One difference remains. Over a mock transport a request the client gave up on is never carried out, while over a socket it is carried out late, as on a real server. A test about a write that lands after its client gave up therefore uses `serve()`.
```

- [ ] **Step 6: Type-check and commit.**

```bash
$PY -m mypy tests/harness
$PY -m mypy tests/harness --ignore-missing-imports
git add tests/harness/fakes/__init__.py tests/harness/fakes/_http.py \
    tests/adversarial/fakes/__init__.py tests/adversarial/fakes/test_adv_fake_http.py \
    docs/claude/testing.md
git commit -m "Add the fakes' shared request log, fault injection and transports"
```

---

## Task 2: `FakeV1`, the hosted `/v1` API

**Files:**
- Create: `tests/harness/fakes/fake_v1.py`
- Create: `tests/adversarial/fakes/conftest.py` (the `fake_v1` fixture only; tasks 3 and 4 add theirs)
- Create: `tests/adversarial/fakes/test_adv_fake_v1_parity.py`, `tests/adversarial/fakes/test_adv_fake_v1_faults.py`
- Modify: `docs/claude/testing.md`

**Interfaces:**
- Consumes: `HttpFake`, `Request`, `Reply`, `json_reply` from Task 1; `harness.stores.memory()`; the `mcp` fixture from `tests/adversarial/conftest.py`.
- Produces: `FakeV1(memvara=None, *, tenant="default", api_key=API_KEY, read_only=False)` with `memvara`, `api_key`, `ROUTES` (34 names such as `POST /v1/facts` and `GET /v1/memories/{id}`), `remote(**options) -> RemoteMemvara`, `aremote(**options) -> AsyncRemoteMemvara`, and everything `HttpFake` offers; `API_KEY`, `APPLIED_HEADER`, `TOKEN_ID`, `ApiError`.

The routes were read from `memvara/remote/api.py` and `memvara/remote/aio.py`, and each answer follows memvara-cloud's `rest/app.py` and `rest/render.py` on its `origin/main` (read, not changed). The client's `hydrate.py` says that renderer is the authority for the wire format.

- [ ] **Step 1: Write the failing tests.** The fixture file, with only the `fake_v1` fixture for now:

```python
"""Fixtures for the fakes' self-tests. Each one yields a fake and closes it afterwards,
which releases any request still hanging and stops its server."""

from __future__ import annotations

from typing import Iterator

import pytest

from harness.fakes.fake_v1 import FakeV1


@pytest.fixture
def fake_v1() -> Iterator[FakeV1]:
    with FakeV1() as fake:
        yield fake
```

The parity tests:

`tests/adversarial/fakes/test_adv_fake_v1_parity.py`:

```python
"""FakeV1 serves every route the remote clients call, and the real client reads back from
it the answers a local store gives for the same calls."""

from __future__ import annotations

import ast
import asyncio
import dataclasses
import re
from datetime import timedelta
from typing import Any, Callable

import pytest

from harness import stores
from harness.env import REPO
from harness.fakes.fake_v1 import FakeV1
from memvara import MemoryType
from memvara.schema import BUILTIN_PREDICATES, PredicateRegistry, PredicateSpec
from memvara.types import (ENTITY_REKEY, LAST_OBSERVED, OBJECT_ENTITY, SALIENCE_BASE,
                           SUBJECT_ENTITY, Claim, utcnow)

#: The two clients whose calls define the routes.
CLIENTS = (REPO / "memvara" / "remote" / "api.py", REPO / "memvara" / "remote" / "aio.py")

#: An id no store holds.
MISSING = "cl_00000000000000000000"


def _template(node: ast.expr) -> str | None:
    """A path argument as a route template with every interpolated part written `{}`, or
    None for a plain name, which is a helper forwarding a path its caller chose."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(part.value if isinstance(part, ast.Constant) else "{}"
                       for part in node.values)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "_document_path"):
        tail = node.args[1] if len(node.args) > 1 else ast.Constant("")
        assert isinstance(tail, ast.Constant), ast.dump(node)
        return "/v1/documents/{}" + str(tail.value)
    assert isinstance(node, ast.Name), f"a path this test cannot read: {ast.dump(node)}"
    return None


def client_routes() -> set[str]:
    """Every `METHOD /v1/...` route the two clients call, read from their source."""
    found: set[str] = set()
    for source in CLIENTS:
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr == "_request" and len(node.args) >= 2:
                method, path = node.args[0], node.args[1]
            elif node.func.attr == "_read" and node.args:
                # `_read` always POSTs: it is the helper for the two reads that carry a
                # query in their body.
                method, path = ast.Constant("POST"), node.args[0]
            else:
                continue
            template = _template(path)
            if template is not None and isinstance(method, ast.Constant):
                found.add(f"{method.value} {template}")
    return found


def test_the_fake_serves_exactly_the_routes_the_clients_call() -> None:
    called = client_routes()
    served = {re.sub(r"\{\w+\}", "{}", name) for name in FakeV1.ROUTES}
    assert called, "the scan found no calls, so it no longer reads the clients"
    assert sorted(called - served) == [], "routes the clients call that FakeV1 lacks"
    assert sorted(served - called) == [], "routes FakeV1 serves that no client calls"


def _shape(claim: Claim) -> tuple[Any, ...]:
    """What a claim says and its state, without its id and its instants, which differ
    between two stores that were told the same things."""
    state = ("retired" if claim.invalidated_at is not None
             else "ended" if claim.valid_to is not None else "live")
    return (claim.subject, claim.predicate, claim.object, claim.text, state,
            claim.memory_type, claim.polarity, claim.confidence, claim.derivation,
            claim.extractor, claim.scope)


def _program(mem: Any) -> dict[str, Any]:
    """remember, get, get_all, search, forget, delete and erase, on one small program,
    returning every answer in a form that two stores can be compared by."""
    out: dict[str, Any] = {}
    berlin = mem.remember("user", "lives_in", "Berlin").added[0]
    moved = mem.remember("user", "lives_in", "Lisbon")
    lisbon = moved.added[0]
    out["correction"] = ([_shape(c) for c in moved.added],
                         [_shape(c) for c in moved.closed])
    mem.remember("user", "prefers", "tabs for indentation",
                 memory_type=MemoryType.PROCEDURAL)
    out["get"] = [_shape(mem.get(lisbon.id)), _shape(mem.get(berlin.id)),
                  mem.get(MISSING)]
    out["get_all"] = sorted(_shape(c) for c in mem.get_all())
    out["get_all_every_state"] = sorted(
        _shape(c) for c in mem.get_all(states=["live", "ended", "retired"]))
    out["search"] = [(_shape(r.claim), round(r.score, 6))
                     for r in mem.search("where does the user live", k=5)]
    out["forget"] = sorted(_shape(c) for c in mem.forget("user", "prefers"))
    out["delete"] = [mem.delete(lisbon.id), mem.delete(lisbon.id), mem.delete(MISSING)]
    out["after_delete"] = _shape(mem.get(lisbon.id))
    out["erase"] = [mem.erase(berlin.id), mem.erase(berlin.id)]
    out["after_erase"] = mem.get(berlin.id)
    out["end"] = sorted(_shape(c) for c in mem.get_all(states=["live", "ended", "retired"]))
    return out


def test_the_remote_client_gets_the_answers_a_local_store_gives(fake_v1: FakeV1) -> None:
    local = stores.memory()
    try:
        assert _program(fake_v1.remote(user="alice")) == _program(local.scope(user="alice"))
    finally:
        local.close()


class _Blocking:
    """Runs an async client's calls to completion one at a time, so the same program can
    drive it."""

    def __init__(self, client: Any, loop: asyncio.AbstractEventLoop) -> None:
        self._client = client
        self._loop = loop

    def __getattr__(self, name: str) -> Callable[..., Any]:
        method = getattr(self._client, name)
        return lambda *args, **kwargs: self._loop.run_until_complete(method(*args, **kwargs))


def test_the_async_client_gets_the_same_answers(fake_v1: FakeV1) -> None:
    loop = asyncio.new_event_loop()
    local = stores.memory()
    client = fake_v1.aremote(user="alice")
    try:
        assert _program(_Blocking(client, loop)) == _program(local.scope(user="alice"))
    finally:
        loop.run_until_complete(client.aclose())
        loop.close()
        local.close()


#: Fields `/v1` does not carry. The cloud's wire model has no place for them, so the
#: client fills in each one's default. The parity tests are where a difference would show.
_NOT_ON_THE_WIRE = {"temporal_precision", "object_kind", "amount", "unit"}

#: `Claim.meta` keys the wire does not carry as they are stored. Three are left out of
#: `metadata`, and two travel as the top-level `salience_base` and `last_observed`, whose
#: properties this test compares instead.
_BOOKKEEPING = {SALIENCE_BASE, LAST_OBSERVED, SUBJECT_ENTITY, OBJECT_ENTITY, ENTITY_REKEY}


def test_every_field_the_wire_carries_survives_the_round_trip(fake_v1: FakeV1) -> None:
    """The client's hydration is the inverse of the fake's rendering, field by field, for
    claims in all three states, with a restatement, sources, metadata, an expiry and a
    closure reason."""
    remote = fake_v1.remote(user="alice")
    remote.remember("user", "lives_in", "Berlin", valid_from=utcnow() - timedelta(days=30))
    remote.remember("user", "lives_in", "Berlin")
    moved = remote.remember("user", "lives_in", "Lisbon", topic="relocation",
                            sources=[{"role": "user", "content": "I moved to Lisbon"}])
    remote.remember("user", "prefers", "short answers", memory_type=MemoryType.PROCEDURAL,
                    expires_at=utcnow() + timedelta(days=3), expire_reason="a trial")
    remote.delete(moved.added[0].id, reason="it was Porto")
    local = fake_v1.memvara.scope(user="alice")
    hydrated = remote.get_all(states=["live", "ended", "retired"])
    assert sorted(c.object for c in hydrated) == ["Berlin", "Lisbon", "short answers"]
    assert max(c.observation_count for c in hydrated) == 2
    for claim in hydrated:
        stored = local.get(claim.id)
        assert stored is not None
        for field in dataclasses.fields(Claim):
            if field.name in _NOT_ON_THE_WIRE or field.name == "meta":
                continue
            assert getattr(claim, field.name) == getattr(stored, field.name), field.name
        assert claim.salience_base == stored.salience_base
        if stored.last_observed is None:
            assert claim.last_observed is None
        else:
            # Epoch seconds travel as an ISO instant, so they come back to the microsecond.
            assert claim.last_observed is not None
            assert claim.last_observed.timestamp() == pytest.approx(
                stored.last_observed.timestamp(), abs=1e-5)
        assert ({k: v for k, v in claim.meta.items() if k not in _BOOKKEEPING}
                == {k: v for k, v in stored.meta.items() if k not in _BOOKKEEPING})


def _walkable() -> PredicateRegistry:
    """The builtins plus one entity-valued predicate, so the store has a graph to walk."""
    return PredicateRegistry(BUILTIN_PREDICATES + (
        PredicateSpec("reports_to", object_type=("entity",), graph=True),))


def test_every_route_answers_the_client_that_calls_it() -> None:
    """One call through the real client for every route the fake serves, each answered
    with a 2xx that the client could read back into its own types."""
    with FakeV1(stores.memory(registry=_walkable())) as fake:
        remote = fake.remote(user="alice")
        assert remote.health()["status"] == "ok"
        assert remote.whoami()["scope"]["tenant"] == "default"

        added = remote.add("I moved to Lisbon last year.")
        assert [c.object for c in added.added] == ["Lisbon"]
        lisbon = added.added[0]
        assert remote.get(lisbon.id) == lisbon
        assert remote.stats()["claims"] == 1
        assert remote.service()["extractor"] == "fast-path-only"
        assert remote.connectivity()["live_claims"] == 1
        assert remote.count() == 1
        assert "Lisbon" in remote.recall("where does the user live")
        assert [c.id for c in remote.history("user", "lives_in")] == [lisbon.id]
        provenance = remote.why(lisbon.id)
        assert provenance is not None and provenance.episodes[0].content.startswith("I moved")
        assert [c.id for c in remote.produced(added.episode_ids[0])] == [lisbon.id]
        assert remote.ask("where does the user live").readings
        assert [c.id for c in remote.since(utcnow() - timedelta(hours=1)).added] == [lisbon.id]

        remote.remember("alice", "reports_to", "bob")
        remote.remember("bob", "reports_to", "carol")
        assert remote.neighborhood("alice", depth=2)
        assert remote.paths_between("alice", "carol")[0].nodes[-1] == "carol"

        rule = remote.remember("user", "prefers", "tabs", memory_type=MemoryType.PROCEDURAL)
        assert [c.object for c in remote.standing()] == ["tabs"]
        assert [r.text for r in remote.profile().standing] == [rule.added[0].text]
        newer = remote.supersede(rule.added[0].id, "user", "prefers", "spaces",
                                 memory_type=MemoryType.PROCEDURAL)
        assert [c.id for c in newer.closed] == [rule.added[0].id]
        assert remote.link(newer.added[0].id, lisbon.id, "extends").relation == "extends"
        assert [k.to_id for k in remote.links(newer.added[0].id)] == [lisbon.id]

        city = remote.remember("user", "works_at", "Acme").added[0]
        assert remote.end(claim_id=city.id)
        assert remote.delete(lisbon.id, close="ended")
        assert [c.object for c in remote.forget("user", "prefers", close="ended")] == ["spaces"]
        tea = remote.remember("user", "likes", "green tea").added[0]
        assert remote.delete(tea.id)
        remote.remember("user", "likes", "coffee")
        assert [c.object for c in remote.forget("user", "likes")] == ["coffee"]
        preview = remote.forget_matching("reports_to", close="retired", k=5)
        confirmed = remote.forget_matching("reports_to", close="retired", k=5,
                                           confirm=preview.confirm)
        assert sorted(c.id for c in confirmed.closed) == sorted(preview.matches)

        doc = remote.add_document("A runbook. Restart the service.", custom_id="docs/runbook",
                                  title="Runbook")
        assert remote.get_document("docs/runbook").id == doc.id
        assert [d.id for d in remote.list_documents().items] == [doc.id]
        assert remote.update_document(doc.id, title="The runbook").title == "The runbook"
        assert remote.document_status("docs/runbook").id == doc.id
        assert remote.delete_document("docs/runbook").deleted
        other = remote.add_document("Second note.", custom_id="docs/other")
        assert [r.deleted for r in remote.delete_documents([other.id, "docs/none"])] == [True, False]

        assert remote.erase(city.id) and remote.get(city.id) is None
        assert remote.purge()["claims"] > 0
        assert remote.consolidate()["status"] == "succeeded"
        assert remote.search("anything") == []

    answered = {r.route for r in fake.requests if r.status is not None and r.status < 300}
    assert sorted(set(FakeV1.ROUTES) - answered) == []
```

The fault tests:

`tests/adversarial/fakes/test_adv_fake_v1_faults.py`:

```python
"""Faults injected into FakeV1 reach the remote client as the client handles them, over a
mock transport and over a real socket."""

from __future__ import annotations

import time
from typing import Callable

import httpx
import pytest

from harness.fakes._http import Request
from harness.fakes.fake_v1 import FakeV1
from harness.stdio import McpProcess
from memvara.remote.api import RemoteMemvara
from memvara.remote.client import DEFAULT_ATTEMPTS
from memvara.remote.errors import (AuthError, InvalidRequest, RateLimited, ReadOnly,
                                   RemoteError, ScopeError)

Start = Callable[..., McpProcess]


def _sent(fake: FakeV1, route: str) -> list[Request]:
    return [r for r in fake.requests if r.route == route]


def test_a_retryable_status_is_retried_with_the_same_key_and_the_retry_lands(
        fake_v1: FakeV1) -> None:
    fake_v1.fail("POST /v1/facts", 503, times=1, body={"error": {
        "code": "unavailable", "message": "a deadlock lost", "retryable": True}})
    receipt = fake_v1.remote(user="alice").remember("user", "lives_in", "Lisbon")
    assert [c.object for c in receipt.added] == ["Lisbon"]
    attempts = _sent(fake_v1, "POST /v1/facts")
    assert [r.status for r in attempts] == [503, 200]
    assert attempts[0].header("idempotency-key") == attempts[1].header("idempotency-key")
    assert len(fake_v1.memvara.scope(user="alice").get_all()) == 1


def test_a_status_the_server_does_not_call_retryable_is_raised_at_once(
        fake_v1: FakeV1) -> None:
    fake_v1.fail("GET /v1/memories/{id}", 403)
    with pytest.raises(ScopeError):
        fake_v1.remote(user="alice").get("cl_00000000000000000000")
    assert [r.status for r in fake_v1.requests] == [403]


def test_a_rate_limit_is_waited_out_for_as_long_as_the_server_asks(fake_v1: FakeV1) -> None:
    fake_v1.fail("POST /v1/search", 429, headers={"Retry-After": "0"}, times=2)
    assert fake_v1.remote(user="alice").search("anything") == []
    assert [r.status for r in _sent(fake_v1, "POST /v1/search")] == [429, 429, 200]


def test_a_rate_limit_longer_than_the_client_will_wait_is_raised_with_the_wait(
        fake_v1: FakeV1) -> None:
    fake_v1.fail("POST /v1/search", 429, headers={"Retry-After": "3600"})
    with pytest.raises(RateLimited) as caught:
        fake_v1.remote(user="alice").search("anything")
    assert caught.value.retry_after == 3600
    assert len(fake_v1.requests) == 1


def test_a_proxy_page_with_no_envelope_is_classified_by_its_status(fake_v1: FakeV1) -> None:
    fake_v1.fail("GET /v1/stats", 502, body="<html>Bad gateway</html>")
    with pytest.raises(RemoteError) as caught:
        fake_v1.remote().stats()
    assert caught.value.status_code == 502 and caught.value.retryable
    assert len(fake_v1.requests) == DEFAULT_ATTEMPTS


def test_a_delay_within_the_client_s_timeout_is_waited_out(fake_v1: FakeV1) -> None:
    fake_v1.delay("GET /v1/stats", 0.3)
    started = time.monotonic()
    assert fake_v1.remote(timeout=5).stats()["claims"] == 0
    assert time.monotonic() - started >= 0.3


def test_a_hang_is_cut_off_by_the_client_s_timeout_on_every_attempt(fake_v1: FakeV1) -> None:
    fake_v1.hang("GET /v1/stats")
    started = time.monotonic()
    with pytest.raises(RemoteError) as caught:
        fake_v1.remote(timeout=0.2).stats()
    assert caught.value.code == "transport"
    assert [r.status for r in fake_v1.requests] == [None] * DEFAULT_ATTEMPTS
    assert time.monotonic() - started >= 0.2 * DEFAULT_ATTEMPTS


def test_the_async_client_times_out_on_a_hang_too(fake_v1: FakeV1) -> None:
    import asyncio

    fake_v1.hang("GET /v1/stats", times=1)

    async def main() -> dict[str, int]:
        client = fake_v1.aremote(timeout=0.2)
        try:
            return await client.stats()
        finally:
            await client.aclose()

    assert asyncio.run(main())["claims"] == 0
    assert [r.status for r in fake_v1.requests] == [None, 200]


def test_over_a_socket_the_client_s_own_timeout_cuts_off_a_hang(fake_v1: FakeV1) -> None:
    fake_v1.hang("GET /v1/stats", times=1)
    with RemoteMemvara(api_key=fake_v1.api_key, base_url=fake_v1.serve(),
                       timeout=0.3) as remote:
        assert remote.stats()["claims"] == 0
    assert [r.status for r in fake_v1.requests] == [None, 200]


def test_a_write_retried_after_its_first_attempt_timed_out_lands_once(
        fake_v1: FakeV1) -> None:
    """Over a socket the first attempt is carried out late, after the client has already
    retried. The idempotency key is what stops it writing a second time."""
    fake_v1.delay("POST /v1/facts", 0.8, times=1)
    with RemoteMemvara(api_key=fake_v1.api_key, base_url=fake_v1.serve(), timeout=0.3,
                       user="alice") as remote:
        assert [c.object for c in remote.remember("user", "lives_in", "Lisbon").added] \
            == ["Lisbon"]
    deadline = time.monotonic() + 5
    while (any(r.status is None for r in _sent(fake_v1, "POST /v1/facts"))
           and time.monotonic() < deadline):
        time.sleep(0.02)
    attempts = _sent(fake_v1, "POST /v1/facts")
    assert len(attempts) == 2
    assert sorted(r.replayed for r in attempts) == [False, True]
    assert len(fake_v1.memvara.scope(user="alice").get_all()) == 1


def test_a_wrong_key_is_refused_and_not_retried(fake_v1: FakeV1) -> None:
    with pytest.raises(AuthError):
        fake_v1.remote(api_key="mv_wrong").stats()
    assert [r.status for r in fake_v1.requests] == [401]


def test_each_user_reads_only_its_own_memories(fake_v1: FakeV1) -> None:
    fake_v1.remote(user="alice").remember("user", "lives_in", "Lisbon")
    assert fake_v1.remote(user="bob").get_all() == []
    assert [c.object for c in fake_v1.remote(user="alice").get_all()] == ["Lisbon"]


def test_an_agent_named_without_a_user_is_refused(fake_v1: FakeV1) -> None:
    with pytest.raises(ScopeError):
        fake_v1.remote(agent="a1").get_all()


def test_a_read_only_deployment_says_so_and_refuses_every_write() -> None:
    with FakeV1(read_only=True) as fake:
        remote = fake.remote(user="alice")
        assert remote.service()["read_only"] is True
        with pytest.raises(ReadOnly):
            remote.remember("user", "lives_in", "Lisbon")


def test_a_field_the_route_does_not_take_is_refused_as_the_cloud_refuses_it(
        fake_v1: FakeV1) -> None:
    """The client counts on this: a read that sends `query_rewrite` to a deployment that
    predates the field gets a 422, and is sent again without it."""
    with httpx.Client(base_url=fake_v1.MOCK_URL, transport=fake_v1.transport(),
                      headers={"Authorization": f"Bearer {fake_v1.api_key}"}) as client:
        refused = client.post("/v1/facts", json={"predicate": "lives_in", "object": "Lisbon",
                                                 "no_such_field": 1})
        missing = client.get("/v1/no-such-route")
    assert refused.status_code == 422
    assert refused.json()["error"]["code"] == "invalid_request"
    assert missing.status_code == 404
    with pytest.raises(InvalidRequest):
        fake_v1.remote().search("")


def test_a_cloud_mode_server_process_reaches_the_fake_over_a_real_url(
        fake_v1: FakeV1, mcp: Start) -> None:
    server = mcp(env={"MEMVARA_MODE": "cloud", "MEMVARA_API_KEY": fake_v1.api_key,
                      "MEMVARA_SERVER_URL": fake_v1.serve()})
    server.initialize()
    stored = server.call("memory_remember", subject="user", predicate="lives_in",
                         object="Lisbon")
    assert not stored.is_error, stored.text
    assert "Lisbon" in server.call("memory_recall", query="where does the user live").text
    assert {"POST /v1/facts", "POST /v1/recall"} <= {r.route for r in fake_v1.requests}
    assert all(r.header("authorization") == f"Bearer {fake_v1.api_key}"
               and r.param("user") == "tester"
               for r in fake_v1.requests if r.route != "GET /v1/health")
    assert server.close() == 0
```

- [ ] **Step 2: Run them and see them fail** because `harness.fakes.fake_v1` does not exist.

Run: `TEST tests/adversarial/fakes/test_adv_fake_v1_parity.py tests/adversarial/fakes/test_adv_fake_v1_faults.py`
Expected: collection errors, `ModuleNotFoundError: No module named 'harness.fakes.fake_v1'`.

- [ ] **Step 3: Write the fake.**

`tests/harness/fakes/fake_v1.py`:

```python
"""A fake of the hosted `/v1` REST API, answered by a real local `Memvara`.

`RemoteMemvara` and `AsyncRemoteMemvara` (`memvara/remote/`) turn each method call into one
`/v1` request, and hydrate the JSON that comes back into the library's own dataclasses
(`memvara/remote/hydrate.py`). `FakeV1` answers those requests from an in-process
`Memvara` with the hashing embedder and no model. It renders each answer the way
memvara-cloud's `rest/render.py` does, because `hydrate.py` is written as the inverse of
that module, and it refuses a request the way the cloud's routes refuse it. A test can
therefore point the real client at it and compare the answers with a local store's.

**The routes** are the 34 that the two clients call, read from `memvara/remote/api.py` and
`memvara/remote/aio.py`. A self-test reads those two files and fails when a client calls a
route this fake does not serve, or when this fake serves one that no client calls.
`RemoteStore` (`memvara/store/remote.py`) calls three of the same routes.

**The credential** is one API key, `API_KEY` unless a test names another. It is bound to
the whole tenant with the `admin` privilege, so a request may narrow to any user, agent or
session with the query parameters the client sends, and to a project with the
`Memvara-Project` header. A wrong or missing key is a 401. With `read_only=True`, every
write is a 403 `read_only`, as on a read-only deployment.

**A retried write is carried out once.** A write that carries an `Idempotency-Key` is
answered from the stored reply when the same key arrives again for the same method and
path, as on a deployment with one worker. That is what makes the client's retry of a
write safe, and a test can watch it happen.

What this fake leaves out on purpose: allowances and rate limits, so a 402 or a 429 appears
only when a test injects one; legal holds; the audit trail; OAuth; the two routes no client
calls (`GET /v1/jobs/{id}` and `POST /v1/erasures/shred`); and a document added by `url`,
which the cloud fetches and this fake refuses, because the suite runs offline. One route
behaves differently: `POST /v1/maintenance/consolidate` runs the pass before it answers,
so the job it returns has already finished, where the cloud answers first.
"""

from __future__ import annotations

import base64
import binascii
import inspect
import re
import threading
import urllib.parse
import uuid
from datetime import datetime
from typing import Any, Callable, Collection, Sequence

import memvara
from memvara import Memvara
from memvara.confirm import ConfirmationRefused
from memvara.core import ScopedMemvara
from memvara.filters import FilterError
from memvara.ingest import IngestError
from memvara.project import check_project
from memvara.remote.aio import AsyncRemoteMemvara
from memvara.remote.api import PROJECT_HEADER, RemoteMemvara
from memvara.retrieve import Edge, EpisodeResult, Path
from memvara.select.base import Rewrite, Selection, Synthesis
from memvara.store import resolve_states
from memvara.types import (
    ENTITY_REKEY, LAST_OBSERVED, OBJECT_ENTITY, SALIENCE_BASE, SUBJECT_ENTITY, Answer,
    Claim, DeleteResult, Derivation, Document, Episode, Explanation,
    ForgetPreview, Link, MemoryType, Page, Profile, Provenance, Reading, Result, Row,
    Scope, WriteReceipt, as_utc, time_axes, utcnow,
)

from .. import stores
from ._http import HttpFake, Reply, Request, json_reply

#: The key a `FakeV1` accepts unless a test names another. It opens only this in-process
#: fake, so it is not a secret.
API_KEY = "mv_fake_v1_key"

#: The credential's handle, as `GET /v1/whoami` reports it.
TOKEN_ID = "tok_fake_v1"

#: The header that tells a client which project a request ran at.
APPLIED_HEADER = "Memvara-Project-Applied"

#: `Claim.meta` keys that memvara keeps for itself. The cloud leaves them out of every
#: rendered `metadata`, and refuses them in a request's (`render.RESERVED_META`).
RESERVED_META = frozenset({SALIENCE_BASE, LAST_OBSERVED, SUBJECT_ENTITY, OBJECT_ENTITY,
                           ENTITY_REKEY})

#: The named parameters of `Memvara.remember`. A metadata key with one of these names
#: would arrive as a duplicate keyword, so the cloud refuses it with 400.
_REMEMBER_PARAMS = frozenset(inspect.signature(Memvara.remember).parameters) - {"self", "meta"}

#: Every route: its method, its path pattern, its name, and what it needs. "open" needs no
#: credential, "read" any valid one, and "write" and "admin" one that a read-only
#: deployment refuses. The patterns match the path as it arrived, still percent-encoded,
#: so a `custom_id` holding a `/` stays one path segment.
_ROUTES: tuple[tuple[str, str, str, str], ...] = (
    ("GET", r"/v1/health", "GET /v1/health", "open"),
    ("GET", r"/v1/whoami", "GET /v1/whoami", "read"),
    ("GET", r"/v1/stats", "GET /v1/stats", "read"),
    ("POST", r"/v1/search", "POST /v1/search", "read"),
    ("POST", r"/v1/recall", "POST /v1/recall", "read"),
    ("GET", r"/v1/memories", "GET /v1/memories", "read"),
    ("POST", r"/v1/memories", "POST /v1/memories", "write"),
    ("GET", r"/v1/memories/(?P<id>[^/]+)", "GET /v1/memories/{id}", "read"),
    ("DELETE", r"/v1/memories/(?P<id>[^/]+)", "DELETE /v1/memories/{id}", "write"),
    ("GET", r"/v1/memories/(?P<id>[^/]+)/why", "GET /v1/memories/{id}/why", "read"),
    ("GET", r"/v1/memories/(?P<id>[^/]+)/links", "GET /v1/memories/{id}/links", "read"),
    ("POST", r"/v1/memories/(?P<id>[^/]+)/supersede", "POST /v1/memories/{id}/supersede",
     "write"),
    ("GET", r"/v1/history", "GET /v1/history", "read"),
    ("POST", r"/v1/ask", "POST /v1/ask", "read"),
    ("GET", r"/v1/since", "GET /v1/since", "read"),
    ("GET", r"/v1/episodes/(?P<id>[^/]+)/produced", "GET /v1/episodes/{id}/produced",
     "read"),
    ("GET", r"/v1/neighborhood", "GET /v1/neighborhood", "read"),
    ("GET", r"/v1/paths", "GET /v1/paths", "read"),
    ("GET", r"/v1/standing", "GET /v1/standing", "read"),
    ("POST", r"/v1/profile", "POST /v1/profile", "read"),
    ("POST", r"/v1/facts", "POST /v1/facts", "write"),
    ("POST", r"/v1/forget", "POST /v1/forget", "write"),
    ("POST", r"/v1/end", "POST /v1/end", "write"),
    ("POST", r"/v1/forget-matching", "POST /v1/forget-matching", "write"),
    ("POST", r"/v1/links", "POST /v1/links", "write"),
    ("POST", r"/v1/documents", "POST /v1/documents", "write"),
    ("GET", r"/v1/documents", "GET /v1/documents", "read"),
    ("POST", r"/v1/documents/delete", "POST /v1/documents/delete", "write"),
    ("GET", r"/v1/documents/(?P<ref>[^/]+)/status", "GET /v1/documents/{ref}/status",
     "read"),
    ("GET", r"/v1/documents/(?P<ref>[^/]+)", "GET /v1/documents/{ref}", "read"),
    ("PATCH", r"/v1/documents/(?P<ref>[^/]+)", "PATCH /v1/documents/{ref}", "write"),
    ("DELETE", r"/v1/documents/(?P<ref>[^/]+)", "DELETE /v1/documents/{ref}", "write"),
    ("POST", r"/v1/erasures", "POST /v1/erasures", "admin"),
    ("POST", r"/v1/maintenance/consolidate", "POST /v1/maintenance/consolidate", "admin"),
)
_MATCHERS = tuple((method, re.compile(pattern), name) for method, pattern, name, _ in _ROUTES)
_NEEDS = {name: needs for _method, _pattern, name, needs in _ROUTES}

#: The code a status carries when a test injects it without a body, as the cloud spells
#: the code for that status.
_CODES = {400: "bad_request", 401: "unauthorized", 402: "quota_exhausted",
          403: "forbidden_scope", 404: "not_found", 405: "method_not_allowed",
          409: "conflict", 422: "invalid_request", 429: "rate_limited", 503: "unavailable"}

# The fields each request body may carry. The cloud's request models forbid any other
# field, so an unknown one is a 422, which is what lets the client detect a deployment
# that predates a field it sends.
_SEARCH = {"query", "k", "min_score", "anchored", "ranked", "query_rewrite", "as_of",
           "valid_at", "known_at", "states", "include_invalidated", "memory_types",
           "filters", "filepath_prefix", "include_episodes"}
_RECALL = {"query", "k", "min_score", "anchored", "ranked", "query_rewrite", "synthesize",
           "memory_types", "include_episodes", "valid_at", "filters", "filepath_prefix"}
_ASK = {"question", "at", "k", "min_score", "anchored"}
_PROFILE = {"query", "k", "since", "buckets"}
_ADD = {"messages", "role", "ts"}
_MESSAGE = {"role", "content", "ts", "metadata"}
_FACT_BODY = {"subject", "predicate", "object", "text", "polarity", "confidence",
              "memory_type", "valid_from", "valid_to", "recorded_at", "source_ids",
              "sources", "extractor", "metadata"}
_FACT = _FACT_BODY | {"until_reason", "replaces", "reason", "expires_at", "expire_reason"}
_SUPERSEDE = _FACT_BODY | {"at", "close", "reason"}
_FORGET = {"subject", "predicate", "at", "reason"}
_END = {"memory_id", "subject", "predicate", "at", "reason"}
_FORGET_MATCHING = {"query", "close", "k", "reason", "confirm"}
_LINK = {"from_id", "to_id", "relation", "by"}
_DOCUMENT_TEXT = {"content", "content_base64", "title", "filepath", "mime", "metadata",
                  "extract"}
_ERASURE = {"memory_id", "scope", "sources", "confirm_tenant"}


class ApiError(Exception):
    """A refusal, sent as the cloud's error envelope:
    `{"error": {"code": ..., "message": ..., "detail": ...}}`."""

    def __init__(self, status: int, code: str, message: str, *,
                 detail: dict[str, Any] | None = None,
                 headers: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.detail = detail
        self.headers = headers or {}

    def reply(self) -> Reply:
        return json_reply(self.status, _envelope(self.code, self.message, self.detail),
                          self.headers)


def _envelope(code: str, message: str, detail: dict[str, Any] | None = None) -> Any:
    return {"error": {"code": code, "message": message, "detail": detail}}


class FakeV1(HttpFake):
    """The `/v1` routes the remote clients call, answered by a real local `Memvara`.

    `memvara` is the store to answer from. Left out, the fake makes an in-memory one with
    the hashing embedder and no model, and closes it when the fake closes. `tenant` is the
    tenant the credential is bound to, `api_key` the only key it accepts, and `read_only`
    makes every write a 403.
    """

    ROUTES = tuple(name for _method, _pattern, name, _needs in _ROUTES)

    def __init__(self, memvara: Memvara | None = None, *, tenant: str = "default",
                 api_key: str = API_KEY, read_only: bool = False) -> None:
        super().__init__()
        self._owns_store = memvara is None
        #: The store every request is answered from.
        self.memvara = memvara if memvara is not None else stores.memory()
        self.tenant = tenant
        self.api_key = api_key
        self.read_only = read_only
        self._stored: dict[tuple[str, str, str], Reply] = {}
        self._key_locks: dict[str, threading.Lock] = {}
        self._handlers: dict[str, Callable[[ScopedMemvara, Request], Any]] = {
            "GET /v1/whoami": self._whoami,
            "GET /v1/stats": self._stats,
            "POST /v1/search": self._search,
            "POST /v1/recall": self._recall,
            "GET /v1/memories": self._list,
            "POST /v1/memories": self._add,
            "GET /v1/memories/{id}": self._get,
            "DELETE /v1/memories/{id}": self._delete,
            "GET /v1/memories/{id}/why": self._why,
            "GET /v1/memories/{id}/links": self._links,
            "POST /v1/memories/{id}/supersede": self._supersede,
            "GET /v1/history": self._history,
            "POST /v1/ask": self._ask,
            "GET /v1/since": self._since,
            "GET /v1/episodes/{id}/produced": self._produced,
            "GET /v1/neighborhood": self._neighborhood,
            "GET /v1/paths": self._paths,
            "GET /v1/standing": self._standing,
            "POST /v1/profile": self._profile,
            "POST /v1/facts": self._fact,
            "POST /v1/forget": self._forget,
            "POST /v1/end": self._end,
            "POST /v1/forget-matching": self._forget_matching,
            "POST /v1/links": self._link,
            "POST /v1/documents": self._add_document,
            "GET /v1/documents": self._list_documents,
            "POST /v1/documents/delete": self._delete_documents,
            "GET /v1/documents/{ref}/status": self._document_status,
            "GET /v1/documents/{ref}": self._get_document,
            "PATCH /v1/documents/{ref}": self._update_document,
            "DELETE /v1/documents/{ref}": self._delete_document,
            "POST /v1/erasures": self._erase,
            "POST /v1/maintenance/consolidate": self._consolidate,
        }

    # -- clients --------------------------------------------------------------------

    def remote(self, **options: Any) -> RemoteMemvara:
        """A `RemoteMemvara` that reaches this fake through `transport()`.

        `options` go to its constructor: `user`, `agent`, `session`, `project`, `timeout`,
        `redactor`, `metadata_filters`, and `api_key` to send a different key. Only the
        transport under its `httpx.Client` is replaced, so the client keeps its own
        headers, timeout and retry rule.
        """
        options.setdefault("api_key", self.api_key)
        client = RemoteMemvara(base_url=self.MOCK_URL, **options)
        client._http._client._transport = self.transport()
        return client

    def aremote(self, **options: Any) -> AsyncRemoteMemvara:
        """`remote()` for `AsyncRemoteMemvara`, through `async_transport()`."""
        options.setdefault("api_key", self.api_key)
        client = AsyncRemoteMemvara(base_url=self.MOCK_URL, **options)
        client._http._client._transport = self.async_transport()
        return client

    def close(self) -> None:
        super().close()
        if self._owns_store:
            self.memvara.close()

    # -- routing --------------------------------------------------------------------

    def route(self, request: Request) -> str | None:
        for method, pattern, name in _MATCHERS:
            found = pattern.fullmatch(request.path)
            if found is not None and method == request.method:
                request.params = {key: urllib.parse.unquote(value)
                                  for key, value in found.groupdict().items()}
                return name
        return None

    def error_body(self, status: int, route: str) -> Any:
        code = _CODES.get(status, "internal" if status >= 500 else "bad_request")
        return _envelope(code, f"FakeV1 injected a {status} on {route}")

    def respond(self, request: Request) -> Reply:
        try:
            return self._respond(request)
        except ApiError as refusal:
            return refusal.reply()
        except Exception as exc:  # noqa: BLE001 - a crash is a 500, as on a deployment
            return ApiError(500, "internal",
                            f"FakeV1 raised {type(exc).__name__}: {exc}").reply()

    def _respond(self, request: Request) -> Reply:
        name = request.route
        if name is None:
            if any(pattern.fullmatch(request.path) for _m, pattern, _n in _MATCHERS):
                raise ApiError(405, "method_not_allowed", "Method Not Allowed")
            raise ApiError(404, "not_found", "Not Found")
        needs = _NEEDS[name]
        if needs == "open":
            return json_reply(200, {"status": "ok", "memvara_version": memvara.__version__})
        self._authorize(request)
        if needs in ("write", "admin") and self.read_only:
            raise ApiError(403, "read_only",
                           "this deployment is read-only, and refuses every write")
        scope = _resolve(Scope(self.tenant), user=request.param("user"),
                         agent=request.param("agent"), session=request.param("session"),
                         project=request.header(PROJECT_HEADER))
        view = self.memvara.scope(tenant=scope.tenant, user=scope.user, agent=scope.agent,
                                  session=scope.session, project=scope.project)
        headers = {} if scope.project is None else {APPLIED_HEADER: scope.project}
        handler = self._handlers[name]

        def answer() -> Reply:
            body = handler(view, request)
            return body if isinstance(body, Reply) else json_reply(200, body, headers)

        key = request.header("idempotency-key")
        if needs in ("write", "admin") and key:
            return self._once((key, request.method, request.path), request, answer)
        return answer()

    def _authorize(self, request: Request) -> None:
        scheme, _, token = (request.header("authorization") or "").partition(" ")
        if scheme.lower() != "bearer" or token.strip() != self.api_key:
            raise ApiError(401, "unauthorized",
                           "no bearer token, or one this deployment does not know",
                           headers={"WWW-Authenticate": 'Bearer realm="memvara"'})

    def _once(self, key: tuple[str, str, str], request: Request,
              answer: Callable[[], Reply]) -> Reply:
        """Answer a write once per idempotency key, method and path.

        One lock per key, so a retry that arrives while the first attempt is still being
        answered waits for it and then gets its reply, rather than writing a second time.
        """
        with self._lock:
            lock = self._key_locks.setdefault(key[0], threading.Lock())
        with lock:
            stored = self._stored.get(key)
            if stored is not None:
                request.replayed = True
                return stored
            reply = answer()
            if reply.status < 500:
                self._stored[key] = reply
            return reply

    # -- service --------------------------------------------------------------------

    def _whoami(self, view: ScopedMemvara, request: Request) -> Any:
        return {"token_id": TOKEN_ID, "scope": _scope(Scope(self.tenant)),
                "granted_privilege": "admin",
                "effective_privilege": "read" if self.read_only else "admin",
                "expires_at": None, "read_only": self.read_only}

    def _stats(self, view: ScopedMemvara, request: Request) -> Any:
        counts = dict(view.stats())
        counts.update(view.connectivity())
        return {"scope": _scope(view.scope), "visible": view.count(),
                "tenant_counts": counts, "extractor": self.memvara.extractor,
                "read_only": self.read_only}

    # -- reading --------------------------------------------------------------------

    def _search(self, view: ScopedMemvara, request: Request) -> Any:
        body = _query_body(request, _SEARCH)
        valid_at, known_at = _axes(body.get("as_of"), body.get("valid_at"),
                                   body.get("known_at"))
        states = _states(body.get("states"), body.get("include_invalidated"))
        try:
            found = view.search(
                body["query"], k=body.get("k", 10), min_score=body.get("min_score", 0.0),
                anchored=bool(body.get("anchored")), ranked=bool(body.get("ranked")),
                query_rewrite=body.get("query_rewrite", True), valid_at=valid_at,
                known_at=known_at, states=states,
                memory_types=_memory_types(body.get("memory_types")),
                filters=body.get("filters"), filepath_prefix=body.get("filepath_prefix"),
                include_episodes=bool(body.get("include_episodes")))
        except FilterError as exc:
            raise ApiError(400, "bad_request", str(exc)) from None
        return {"as_of": body.get("as_of"), "valid_at": _instant(valid_at),
                "known_at": _instant(known_at), "states": list(states),
                "count": len(found), "results": [_hit(item) for item in found],
                "selection": _selection(getattr(found, "selection", None)),
                "rewrite": _rewrite(getattr(found, "rewrite", None))}

    def _recall(self, view: ScopedMemvara, request: Request) -> Any:
        body = _query_body(request, _RECALL)
        valid_at = _instant_in(body.get("valid_at"), "valid_at")
        try:
            result = view.recall(
                body["query"], k=body.get("k", 8), min_score=body.get("min_score", 0.0),
                anchored=bool(body.get("anchored")), ranked=bool(body.get("ranked")),
                query_rewrite=body.get("query_rewrite", True),
                synthesize=bool(body.get("synthesize")),
                memory_types=_memory_types(body.get("memory_types")),
                include_episodes=bool(body.get("include_episodes")), valid_at=valid_at,
                filters=body.get("filters"), filepath_prefix=body.get("filepath_prefix"),
                with_ids=True)
        except FilterError as exc:
            raise ApiError(400, "bad_request", str(exc)) from None
        return {"text": result.text, "empty": not result.text,
                "valid_at": _instant(valid_at), "selection": _selection(result.selection),
                "rewrite": _rewrite(result.rewrite),
                "synthesis": _synthesis(result.synthesis)}

    def _list(self, view: ScopedMemvara, request: Request) -> Any:
        limit = _int(request, "limit", 100, low=1, high=500)
        offset = _int(request, "offset", 0, low=0)
        valid_at, known_at = _axes(request.param("as_of"), request.param("valid_at"),
                                   request.param("known_at"))
        states = _states(request.query.get("states"),
                         _flag(request, "include_invalidated"))
        claims = view.get_all(states=states, valid_at=valid_at, known_at=known_at)
        page = claims[offset:offset + limit]
        return {"count": len(page), "total": len(claims), "limit": limit,
                "offset": offset, "as_of": request.param("as_of"),
                "valid_at": _instant(valid_at), "known_at": _instant(known_at),
                "states": list(states), "memories": [_memory(c) for c in page]}

    def _get(self, view: ScopedMemvara, request: Request) -> Any:
        claim = view.get(request.params["id"])
        if claim is None:
            raise _no_memory(request.params["id"])
        return _memory(claim)

    def _why(self, view: ScopedMemvara, request: Request) -> Any:
        valid_at, known_at = _axes(request.param("as_of"), request.param("valid_at"),
                                   request.param("known_at"))
        found = view.why(request.params["id"], valid_at=valid_at, known_at=known_at)
        if found is None:
            raise _no_memory(request.params["id"])
        return _provenance(found)

    def _history(self, view: ScopedMemvara, request: Request) -> Any:
        predicate = _required(request, "predicate")
        subject = request.param("subject") or "user"
        valid_at, known_at = _axes(request.param("as_of"), request.param("valid_at"),
                                   request.param("known_at"))
        claims = view.history(subject, predicate, valid_at=valid_at, known_at=known_at)
        return {"subject": subject,
                "predicate": self.memvara.registry.normalize(predicate),
                "scope": _scope(view.scope), "as_of": request.param("as_of"),
                "valid_at": _instant(valid_at), "known_at": _instant(known_at),
                "count": len(claims), "timeline": [_memory(c) for c in claims]}

    def _ask(self, view: ScopedMemvara, request: Request) -> Any:
        body = _body(request, _ASK, required=("question",))
        found = view.ask(body["question"], at=_instant_in(body.get("at"), "at"),
                         k=body.get("k", 3), min_score=body.get("min_score", 0.0),
                         anchored=bool(body.get("anchored")))
        return _answer(found)

    def _since(self, view: ScopedMemvara, request: Request) -> Any:
        when = _instant_in(_required(request, "since"), "since")
        assert when is not None  # _required refuses an absent value
        delta = view.since(when)
        return {"since": _instant(delta.since), "added": [_memory(c) for c in delta.added],
                "gone": [_memory(c) for c in delta.gone]}

    def _produced(self, view: ScopedMemvara, request: Request) -> Any:
        valid_at, known_at = _axes(request.param("as_of"), request.param("valid_at"),
                                   request.param("known_at"))
        claims = view.produced(request.params["id"], valid_at=valid_at, known_at=known_at)
        return {"episode_id": request.params["id"], "as_of": request.param("as_of"),
                "valid_at": _instant(valid_at), "known_at": _instant(known_at),
                "count": len(claims), "memories": [_memory(c) for c in claims]}

    def _neighborhood(self, view: ScopedMemvara, request: Request) -> Any:
        valid_at, known_at = _axes(request.param("as_of"), request.param("valid_at"),
                                   request.param("known_at"))
        found = view.neighborhood(
            _required(request, "entity"), depth=_int(request, "depth", 2, low=1, high=4),
            k=_int(request, "k", 10, low=1, high=50),
            min_hops=_int(request, "min_hops", 1, low=1, high=4),
            predicates=request.query.get("predicates"),
            min_score=_float(request, "min_score", 0.0), valid_at=valid_at,
            known_at=known_at)
        return _paths_body(request, valid_at, known_at, found)

    def _paths(self, view: ScopedMemvara, request: Request) -> Any:
        valid_at, known_at = _axes(request.param("as_of"), request.param("valid_at"),
                                   request.param("known_at"))
        found = view.paths_between(
            _required(request, "source"), _required(request, "target"),
            depth=_int(request, "depth", 3, low=1, high=4),
            k=_int(request, "k", 3, low=1, high=50),
            predicates=request.query.get("predicates"),
            min_score=_float(request, "min_score", 0.0), valid_at=valid_at,
            known_at=known_at)
        return _paths_body(request, valid_at, known_at, found)

    def _standing(self, view: ScopedMemvara, request: Request) -> Any:
        limit = _int(request, "limit", 64, low=1, high=200)
        every = view.standing()
        return {"count": min(len(every), limit), "limit": limit,
                "truncated": len(every) > limit,
                "memories": [_memory(c) for c in every[:limit]]}

    def _profile(self, view: ScopedMemvara, request: Request) -> Any:
        body = _body(request, _PROFILE)
        found = view.profile(body.get("query"), k=body.get("k", 8),
                             since=_instant_in(body.get("since"), "since"),
                             buckets=body.get("buckets"))
        return _profile_body(found)

    def _links(self, view: ScopedMemvara, request: Request) -> Any:
        if view.get(request.params["id"]) is None:
            raise _no_memory(request.params["id"])
        return {"claim_links": [_link(k) for k in view.links(request.params["id"])]}

    # -- writing --------------------------------------------------------------------

    def _add(self, view: ScopedMemvara, request: Request) -> Any:
        body = _body(request, _ADD, required=("messages",))
        messages = body["messages"]
        payload: Any
        if isinstance(messages, str):
            payload = messages
        else:
            if not isinstance(messages, list):
                raise ApiError(422, "invalid_request",
                               "messages is neither a string nor a list of messages")
            payload = []
            for raw in messages:
                message = _fields(raw, _MESSAGE, ("content",), "a message")
                meta = _message_metadata(message)
                item: dict[str, Any] = {"role": message.get("role") or "user",
                                        "content": message["content"]}
                if message.get("ts") is not None:
                    item["ts"] = _instant_in(message["ts"], "ts")
                item.update(meta)
                payload.append(item)
        receipt = view.add(payload, role=body.get("role", "user"),
                           ts=_instant_in(body.get("ts"), "ts"))
        return _receipt(receipt, self.memvara.extractor)

    def _fact(self, view: ScopedMemvara, request: Request) -> Any:
        body = _body(request, _FACT, required=("predicate", "object"))
        meta = _metadata(body)
        sources = list(body.get("source_ids") or []) + _episodes(body.get("sources"),
                                                                  view.scope)
        replaces = body.get("replaces")
        try:
            receipt = view.remember(
                body.get("subject") or "user", body["predicate"], body["object"],
                confidence=body.get("confidence", 1.0),
                memory_type=_memory_type(body.get("memory_type")),
                polarity=body.get("polarity", 1),
                valid_from=_instant_in(body.get("valid_from"), "valid_from"),
                valid_to=_instant_in(body.get("valid_to"), "valid_to"),
                recorded_at=_instant_in(body.get("recorded_at"), "recorded_at"),
                sources=sources or None, text=body.get("text"),
                extractor=body.get("extractor", "api"),
                until_reason=body.get("until_reason"), replaces=replaces,
                reason=body.get("reason"),
                expires_at=_instant_in(body.get("expires_at"), "expires_at"),
                expire_reason=body.get("expire_reason"), **meta)
        except KeyError:
            if replaces is None:
                raise
            raise _no_memory(replaces) from None
        except ValueError as exc:
            if replaces is None:
                raise ApiError(400, "bad_request", str(exc)) from None
            raise ApiError(409, "conflict", str(exc)) from None
        return _receipt(receipt, self.memvara.extractor)

    def _supersede(self, view: ScopedMemvara, request: Request) -> Any:
        body = _body(request, _SUPERSEDE, required=("predicate", "object"))
        meta = _metadata(body)
        registry = self.memvara.registry
        predicate = registry.normalize(body["predicate"])
        recorded = _instant_in(body.get("recorded_at"), "recorded_at") or utcnow()
        claim = Claim(
            subject=body.get("subject") or "user", predicate=predicate,
            object=body["object"], scope=view.scope, polarity=body.get("polarity", 1),
            confidence=body.get("confidence", 1.0),
            memory_type=(_memory_type(body.get("memory_type"))
                         or registry.spec(predicate).memory_type),
            valid_from=_instant_in(body.get("valid_from"), "valid_from") or recorded,
            valid_to=_instant_in(body.get("valid_to"), "valid_to"), recorded_at=recorded,
            text=body.get("text") or "", derivation=Derivation.USER,
            extractor=body.get("extractor", "api"), meta=meta)
        sources = list(body.get("source_ids") or []) + _episodes(body.get("sources"),
                                                                  view.scope)
        try:
            receipt = view.supersede(request.params["id"], claim,
                                     at=_instant_in(body.get("at"), "at"),
                                     sources=sources or None,
                                     close=body.get("close", "ended"),
                                     reason=body.get("reason"))
        except KeyError:
            raise _no_memory(request.params["id"]) from None
        except ValueError as exc:
            raise ApiError(409, "conflict", str(exc)) from None
        return _receipt(receipt, self.memvara.extractor)

    def _delete(self, view: ScopedMemvara, request: Request) -> Any:
        body = _body(request, {"reason"})
        done = view.delete(request.params["id"], reason=body.get("reason"))
        return {"id": request.params["id"], "retired": done, "erased": False}

    def _forget(self, view: ScopedMemvara, request: Request) -> Any:
        body = _body(request, _FORGET, required=("predicate",))
        subject = body.get("subject") or "user"
        retired = view.forget(subject, body["predicate"],
                              at=_instant_in(body.get("at"), "at"),
                              reason=body.get("reason"))
        return {"subject": subject, "predicate": body["predicate"],
                "count": len(retired),
                "retired": [_memory(c) for c in self._reread(retired)], "erased": False}

    def _end(self, view: ScopedMemvara, request: Request) -> Any:
        body = _body(request, _END)
        memory_id, predicate = body.get("memory_id"), body.get("predicate")
        if (memory_id is None) == (predicate is None):
            raise ApiError(422, "invalid_request",
                           "exactly one of memory_id and predicate is required")
        at, reason = _instant_in(body.get("at"), "at"), body.get("reason")
        subject = body.get("subject") or "user"
        claims: list[Claim] = []
        if memory_id is not None:
            found = view.get(memory_id)
            if found is not None and view.delete(memory_id, at=at, close="ended",
                                                 reason=reason):
                claims = self._reread([found])
        else:
            claims = self._reread(view.forget(subject, predicate, at=at, close="ended",
                                              reason=reason))
        return {"memory_id": memory_id,
                "subject": subject if memory_id is None else None,
                "predicate": predicate, "count": len(claims),
                "ended": [_memory(c) for c in claims], "erased": False}

    def _forget_matching(self, view: ScopedMemvara, request: Request) -> Any:
        body = _body(request, _FORGET_MATCHING, required=("close",))
        try:
            outcome = view.forget_matching(body.get("query") or "", close=body["close"],
                                           k=body.get("k", 20), reason=body.get("reason"),
                                           confirm=body.get("confirm"))
        except ConfirmationRefused as exc:
            raise ApiError(409, "conflict", str(exc)) from None
        if isinstance(outcome, ForgetPreview):
            return _forget_preview(outcome)
        return {"close": outcome.close,
                "closed": [_memory(c) for c in self._reread(outcome.closed)],
                "reason": outcome.reason}

    def _link(self, view: ScopedMemvara, request: Request) -> Any:
        body = _body(request, _LINK, required=("from_id", "to_id", "relation"))
        try:
            made = view.link(body["from_id"], body["to_id"], body["relation"],
                             by=body.get("by", "api"))
        except KeyError:
            raise ApiError(404, "not_found",
                           f"no memory {body['from_id']!r} or {body['to_id']!r} is "
                           "visible to this credential.") from None
        return _link(made)

    # -- documents ------------------------------------------------------------------

    def _add_document(self, view: ScopedMemvara, request: Request) -> Any:
        body = _body(request, _DOCUMENT_TEXT | {"url", "custom_id"})
        if body.get("url") is not None:
            raise ApiError(400, "bad_request",
                           "FakeV1 does not fetch a document by url, because the suite "
                           "runs offline. Send its content instead.")
        custom_id = body.get("custom_id")
        if custom_id is not None and custom_id.endswith("/status"):
            raise ApiError(400, "bad_request",
                           f"custom_id {custom_id!r} ends in '/status', which "
                           "GET /v1/documents/{id}/status reserves.")
        try:
            doc = view.add_document(_document_content(body), custom_id=custom_id,
                                    title=body.get("title"), filepath=body.get("filepath"),
                                    mime=body.get("mime"), meta=body.get("metadata"),
                                    extract=body.get("extract", True))
        except IngestError as exc:
            raise ApiError(400, "bad_request", str(exc),
                           detail={"ingest": exc.code}) from None
        except ValueError as exc:
            raise ApiError(400, "bad_request", str(exc)) from None
        return _document(doc)

    def _list_documents(self, view: ScopedMemvara, request: Request) -> Any:
        try:
            page = view.list_documents(filepath_prefix=request.param("filepath_prefix"),
                                       status=request.param("status"),
                                       limit=_int(request, "limit", 50, low=1, high=1000),
                                       cursor=request.param("cursor"))
        except ValueError as exc:
            raise ApiError(400, "bad_request", str(exc)) from None
        return _document_page(page)

    def _get_document(self, view: ScopedMemvara, request: Request) -> Any:
        doc = view.get_document(request.params["ref"])
        if doc is None:
            raise _no_document(request.params["ref"])
        return _document(doc)

    def _update_document(self, view: ScopedMemvara, request: Request) -> Any:
        body = _body(request, _DOCUMENT_TEXT)
        try:
            doc = view.update_document(request.params["ref"],
                                       content=_document_content(body),
                                       title=body.get("title"), meta=body.get("metadata"),
                                       filepath=body.get("filepath"), mime=body.get("mime"),
                                       extract=body.get("extract", True))
        except KeyError:
            raise _no_document(request.params["ref"]) from None
        except IngestError as exc:
            raise ApiError(400, "bad_request", str(exc),
                           detail={"ingest": exc.code}) from None
        except ValueError as exc:
            raise ApiError(400, "bad_request", str(exc)) from None
        return _document(doc)

    def _delete_document(self, view: ScopedMemvara, request: Request) -> Any:
        return _deleted(view.delete_document(request.params["ref"]))

    def _delete_documents(self, view: ScopedMemvara, request: Request) -> Any:
        body = _body(request, {"ids"}, required=("ids",))
        return {"results": [_deleted(r) for r in view.delete_documents(body["ids"])]}

    def _document_status(self, view: ScopedMemvara, request: Request) -> Any:
        try:
            found = view.document_status(request.params["ref"])
        except KeyError:
            raise _no_document(request.params["ref"]) from None
        return {"id": found.id, "status": found.status, "error": found.error,
                "chunks": found.chunks, "updated_at": _instant(found.updated_at)}

    # -- erasure and maintenance ------------------------------------------------------

    def _erase(self, view: ScopedMemvara, request: Request) -> Any:
        body = _body(request, _ERASURE)
        memory_id, target = body.get("memory_id"), body.get("scope")
        if (memory_id is None) == (target is None):
            raise ApiError(400, "bad_request",
                           "send exactly one of 'memory_id', to erase one memory, or "
                           "'scope', to erase everything at a scope and beneath it.")
        if memory_id is not None:
            sources = bool(body.get("sources", False))
            erased = view.erase(memory_id, sources=sources)
            return {"target": "memory", "memory_id": memory_id, "scope": None,
                    "erased": erased, "counts": None,
                    "sources_erased": sources if erased else False,
                    "audit_subject_linkable": None}
        wanted = _fields(target, {"user", "agent", "session"}, (), "scope")
        scope = _resolve(view.scope, user=wanted.get("user"), agent=wanted.get("agent"),
                         session=wanted.get("session"))
        if scope.user is None and body.get("confirm_tenant") != scope.tenant:
            raise ApiError(400, "bad_request",
                           "erasing a scope that names no user erases the whole tenant. "
                           f"Send confirm_tenant={scope.tenant!r} if that is what you "
                           "mean, or name a user.")
        counts = self.memvara.scope(tenant=scope.tenant, user=scope.user,
                                    agent=scope.agent, session=scope.session,
                                    project=scope.project).purge()
        return {"target": "scope", "memory_id": None, "scope": _scope(scope),
                "erased": True, "counts": counts, "sources_erased": None,
                "audit_subject_linkable": None}

    def _consolidate(self, view: ScopedMemvara, request: Request) -> Any:
        job_id = f"job_{uuid.uuid4().hex[:16]}"
        started = utcnow()
        result: dict[str, int] | None = None
        error: str | None = None
        try:
            result = self.memvara.consolidate(tenant=self.tenant)
        except Exception as exc:  # noqa: BLE001 - a failed pass is reported on the job
            error = f"{type(exc).__name__}: {exc}"
        job = {"id": job_id, "kind": "consolidate", "tenant": self.tenant,
               "status": "failed" if error else "succeeded",
               "created_at": _instant(started), "started_at": _instant(started),
               "finished_at": _instant(utcnow()), "result": result, "error": error,
               "links": {"self": f"/v1/jobs/{job_id}"}}
        return json_reply(202, job, {"Location": f"/v1/jobs/{job_id}", "Retry-After": "2"})

    # -- shared by several routes -------------------------------------------------------

    def _reread(self, claims: Sequence[Claim]) -> list[Claim]:
        """The claims as the store now holds them. A route that closes claims echoes the
        rows it wrote, re-read, as the cloud's `_reread` does."""
        fresh = self.memvara.store.get_claims([c.id for c in claims])
        return [fresh.get(c.id, c) for c in claims]


# -- the scope a request runs at -----------------------------------------------------


def _resolve(credential: Scope, *, user: str | None = None, agent: str | None = None,
             session: str | None = None, project: str | None = None) -> Scope:
    """The credential's scope, narrowed by what a request asked for.

    memvara-cloud's `rest/scope.py::resolve`: a field the request leaves out keeps the
    credential's value, a field it sends must be one the credential permits, and an agent
    or a session named without a user is refused, because it would reach that agent or
    session under every user.
    """
    wanted = Scope(credential.tenant, _clean(user) or credential.user,
                   _clean(agent) or credential.agent,
                   _clean(session) or credential.session,
                   project=_project(project) or credential.project)
    if not credential.contains(wanted):
        raise ApiError(403, "forbidden_scope",
                       f"this credential is scoped to {credential.key()} and cannot "
                       f"address {wanted.key()}")
    if wanted.user is None and (wanted.agent is not None or wanted.session is not None):
        named = "agent" if wanted.agent is not None else "session"
        raise ApiError(400, "bad_scope",
                       f"naming an {named} without a user addresses that {named} under "
                       "every user in the tenant. Send user= as well.")
    return wanted


def _clean(value: str | None) -> str | None:
    """Blank means absent, and a NUL is refused, as the cloud refuses it."""
    text = (value or "").strip()
    if "\x00" in text:
        raise ApiError(400, "bad_scope", "a scope identifier contains a NUL character")
    return text or None


def _project(value: str | None) -> str | None:
    text = _clean(value)
    if text is None:
        return None
    try:
        return check_project(text)
    except ValueError as exc:
        raise ApiError(400, "bad_scope",
                       f"the {PROJECT_HEADER} header is not usable: {exc}") from None


# -- reading a request -----------------------------------------------------------------


def _body(request: Request, allowed: Collection[str],
          required: Sequence[str] = ()) -> dict[str, Any]:
    """The request's JSON body, checked as the cloud's request models check it."""
    try:
        value = request.json()
    except ValueError:
        raise ApiError(422, "invalid_request", "the body is not JSON") from None
    return _fields({} if value is None else value, allowed, required, "the body")


def _query_body(request: Request, allowed: Collection[str]) -> dict[str, Any]:
    """The body of a read that carries a query, which the cloud requires to be a
    non-empty string."""
    body = _body(request, allowed, required=("query",))
    if not isinstance(body["query"], str) or not body["query"]:
        raise ApiError(422, "invalid_request", "query must be a non-empty string")
    return body


def _fields(value: Any, allowed: Collection[str], required: Sequence[str],
            what: str) -> dict[str, Any]:
    """`value` as an object, refused with 422 when it is not an object, carries a field
    that is not allowed, or lacks a required one."""
    if not isinstance(value, dict):
        raise ApiError(422, "invalid_request", f"{what} is not a JSON object")
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ApiError(422, "invalid_request",
                       f"{what} has fields this route does not take: {unknown}")
    missing = [name for name in required if value.get(name) is None]
    if missing:
        raise ApiError(422, "invalid_request", f"{what} is missing {missing}")
    return value


def _instant_in(value: Any, name: str) -> datetime | None:
    """A timestamp from a request, in UTC, or None when it was not sent."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ApiError(422, "invalid_request", f"{name} is not a timestamp")
    text = value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value
    try:
        return as_utc(datetime.fromisoformat(text))
    except ValueError:
        raise ApiError(422, "invalid_request",
                       f"{name} {value!r} is not an ISO 8601 timestamp") from None


def _axes(as_of: Any, valid_at: Any,
          known_at: Any) -> tuple[datetime | None, datetime | None]:
    """The two clocks a read runs on, by `memvara.types.time_axes`, the rule the cloud
    calls. `as_of` beside either of the others is a 400."""
    try:
        return time_axes(_instant_in(as_of, "as_of"), _instant_in(valid_at, "valid_at"),
                         _instant_in(known_at, "known_at"))
    except ValueError as exc:
        raise ApiError(400, "bad_request", str(exc)) from None


def _states(states: Any, include_invalidated: bool | None) -> tuple[str, ...]:
    """The claim states a read covers, by `memvara.store.resolve_states`."""
    try:
        return tuple(resolve_states(states, include_invalidated))
    except ValueError as exc:
        raise ApiError(400, "bad_request", str(exc)) from None


def _memory_type(value: Any) -> MemoryType | None:
    if value is None:
        return None
    try:
        return MemoryType(value)
    except ValueError:
        raise ApiError(422, "invalid_request",
                       f"memory_type {value!r} is not a memory type") from None


def _memory_types(values: Any) -> list[MemoryType] | None:
    if values is None:
        return None
    if not isinstance(values, list):
        raise ApiError(422, "invalid_request", "memory_types is not a list")
    return [t for t in (_memory_type(v) for v in values) if t is not None]


def _required(request: Request, name: str) -> str:
    value = request.param(name)
    if not value:
        raise ApiError(422, "invalid_request", f"the query parameter {name} is required")
    return value


def _int(request: Request, name: str, default: int, *, low: int,
         high: int | None = None) -> int:
    raw = request.param(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ApiError(422, "invalid_request", f"{name}={raw!r} is not a whole number") \
            from None
    if value < low or (high is not None and value > high):
        raise ApiError(422, "invalid_request",
                       f"{name}={value} is outside {low} to {high or 'any'}")
    return value


def _float(request: Request, name: str, default: float) -> float:
    raw = request.param(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ApiError(422, "invalid_request", f"{name}={raw!r} is not a number") from None


def _flag(request: Request, name: str) -> bool | None:
    raw = request.param(name)
    if raw is None:
        return None
    if raw not in ("true", "false"):
        raise ApiError(422, "invalid_request", f"{name}={raw!r} is not true or false")
    return raw == "true"


def _metadata(body: dict[str, Any]) -> dict[str, Any]:
    """A fact's metadata, refused as the cloud's `_check_metadata` refuses it: a key
    memvara reserves, or one that collides with a named field."""
    meta = body.get("metadata") or {}
    if not isinstance(meta, dict):
        raise ApiError(422, "invalid_request", "metadata is not a JSON object")
    reserved = sorted(set(meta) & RESERVED_META)
    if reserved:
        raise ApiError(400, "bad_request",
                       f"metadata key {reserved[0]!r} is reserved by memvara")
    collide = sorted(set(meta) & _REMEMBER_PARAMS)
    if collide:
        raise ApiError(400, "bad_request",
                       f"metadata key(s) {collide} collide with named fields of this "
                       "request")
    return dict(meta)


def _message_metadata(message: dict[str, Any]) -> dict[str, Any]:
    meta = message.get("metadata") or {}
    if not isinstance(meta, dict):
        raise ApiError(422, "invalid_request", "a message's metadata is not an object")
    clash = sorted(set(meta) & {"content", "role", "ts"})
    if clash:
        raise ApiError(400, "bad_request",
                       f"message metadata key(s) {clash} collide with the message's own "
                       "fields")
    return dict(meta)


def _episodes(messages: Any, scope: Scope) -> list[Episode]:
    """Source turns sent with a fact, stored in the request's scope, as the cloud's
    `_episodes` builds them."""
    out = []
    for raw in messages or []:
        message = _fields(raw, _MESSAGE, ("content",), "a source message")
        out.append(Episode(content=message["content"], scope=scope,
                           role=message.get("role") or "user",
                           ts=_instant_in(message.get("ts"), "ts") or utcnow(),
                           meta=_message_metadata(message)))
    return out


def _document_content(body: dict[str, Any]) -> str | bytes | None:
    text, encoded = body.get("content"), body.get("content_base64")
    if text is not None and encoded is not None:
        raise ApiError(400, "bad_request", "send content or content_base64, not both")
    if encoded is None:
        return text
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise ApiError(400, "bad_request", "content_base64 is not base64") from None


def _no_memory(memory_id: str) -> ApiError:
    return ApiError(404, "not_found", f"no memory {memory_id!r} is visible to this "
                                      "credential")


def _no_document(ref: str) -> ApiError:
    return ApiError(404, "not_found", f"no document {ref!r} is visible to this credential")


# -- writing an answer, as memvara-cloud's rest/render.py writes it ----------------------


def _instant(value: datetime | None) -> str | None:
    """An instant as the cloud writes one: ISO 8601, with UTC written as `Z`."""
    if value is None:
        return None
    text = value.isoformat()
    return text[:-6] + "Z" if text.endswith("+00:00") else text


def _scope(scope: Scope) -> dict[str, Any]:
    return {"tenant": scope.tenant, "user": scope.user, "project": scope.project,
            "agent": scope.agent, "session": scope.session}


def _state(claim: Claim) -> str:
    """Retired first, for the cloud's reason: a claim closed on both clocks is one we
    stopped believing, which is the stronger statement."""
    if claim.invalidated_at is not None:
        return "retired"
    if claim.valid_to is not None:
        return "ended"
    return "live"


def _memory(claim: Claim) -> dict[str, Any]:
    history = {"subject": claim.subject, "predicate": claim.predicate}
    for field in ("user", "project", "agent", "session"):
        value = getattr(claim.scope, field)
        if value is not None:
            history[field] = value
    return {
        "id": claim.id, "text": claim.text, "subject": claim.subject,
        "predicate": claim.predicate, "object": claim.object, "polarity": claim.polarity,
        "memory_type": claim.memory_type.value, "scope": _scope(claim.scope),
        "state": _state(claim),
        "valid_time": {"valid_from": _instant(claim.valid_from),
                       "valid_to": _instant(claim.valid_to)},
        "transaction_time": {"recorded_at": _instant(claim.recorded_at),
                             "invalidated_at": _instant(claim.invalidated_at),
                             "invalidated_by": claim.invalidated_by},
        "confidence": claim.confidence, "salience": claim.salience,
        "salience_base": claim.salience_base,
        "observation_count": claim.observation_count,
        "last_observed": _instant(claim.last_observed),
        "expires_at": _instant(claim.expires_at), "expire_reason": claim.expire_reason,
        "derivation": claim.derivation.value, "extractor": claim.extractor or None,
        "source_ids": list(claim.sources),
        "metadata": {k: v for k, v in claim.meta.items() if k not in RESERVED_META},
        "links": {"self": f"/v1/memories/{claim.id}",
                  "why": f"/v1/memories/{claim.id}/why",
                  "history": f"/v1/history?{urllib.parse.urlencode(history)}"},
    }


def _episode(episode: Episode) -> dict[str, Any]:
    return {"id": episode.id, "role": episode.role, "ts": _instant(episode.ts),
            "content": episode.content, "scope": _scope(episode.scope),
            "metadata": dict(episode.meta)}


def _ranking(explain: Explanation, *, applicable: bool = True) -> dict[str, Any]:
    """The ranking explanation. For a turn, which has no recency, confidence or salience,
    those three are null rather than the library's placeholder 1.0."""
    return {"vector_rank": explain.vector_rank, "vector_score": explain.vector_score,
            "lexical_rank": explain.lexical_rank, "lexical_score": explain.lexical_score,
            "fusion_score": explain.fusion_score,
            "recency": explain.recency if applicable else None,
            "confidence": explain.confidence if applicable else None,
            "salience": explain.salience if applicable else None,
            "rerank_score": explain.rerank_score, "raw_score": explain.raw_score,
            "final_score": explain.final_score, "summary": explain.summary(),
            "anchor": explain.anchor, "selected": explain.selected, "span": explain.span}


def _hit(item: Result | EpisodeResult) -> dict[str, Any]:
    if isinstance(item, EpisodeResult):
        return {"kind": "episode", "score": item.score,
                "ranking": _ranking(item.explain, applicable=False),
                "episode": _episode(item.episode)}
    return {"kind": "claim", "score": item.score, "ranking": _ranking(item.explain),
            "memory": _memory(item.claim)}


def _selection(value: Selection | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {"outcome": value.outcome, "reason": value.reason, "status": value.status,
            "candidates": value.candidates, "kept": value.kept}


def _rewrite(value: Rewrite | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {"outcome": value.outcome, "reason": value.reason, "status": value.status,
            "queries": list(value.queries),
            "date_from": None if value.date_from is None else value.date_from.isoformat(),
            "date_to": None if value.date_to is None else value.date_to.isoformat(),
            "valid_at": _instant(value.valid_at)}


def _synthesis(value: Synthesis | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {"outcome": value.outcome, "reason": value.reason, "status": value.status,
            "text": value.text}


def _receipt(value: WriteReceipt, extractor: str) -> dict[str, Any]:
    """A write receipt, with the `note` the cloud adds when a write stored nothing."""
    note = None
    if value.unextracted:
        note = (f"{value.unextracted} turn(s) carried something extraction did not "
                f"recognise and were not stored (extractor: {extractor}).")
        if extractor == "fast-path-only":
            note += (" This deployment has no extraction model, so only a fixed set of "
                     "sentence forms is recognised. Use POST /v1/facts to state the fact "
                     "explicitly, or ask the operator to configure one.")
    elif value.deferred and not value.added:
        note = (f"{len(value.episode_ids)} turn(s) are queued for this deployment's "
                "extraction worker; claims from them arrive after it has read them, not "
                "in this response.")
    return {"episode_ids": list(value.episode_ids),
            "added": [_memory(c) for c in value.added],
            "invalidated": [_memory(c) for c in value.invalidated],
            "reinforced": [_memory(c) for c in value.reinforced],
            "skipped": value.skipped, "unextracted": value.unextracted,
            "ungrounded": value.ungrounded, "polluted": value.polluted,
            "unregistered": value.unregistered, "llm_calls": value.llm_calls,
            "latency_ms": value.latency_ms, "deferred": value.deferred, "note": note,
            "may_replace": [_memory(c) for c in value.may_replace]}


def _link(value: Link) -> dict[str, Any]:
    return {"from_id": value.from_id, "to_id": value.to_id, "relation": value.relation,
            "created_at": _instant(value.created_at), "by": value.by}


def _provenance(value: Provenance) -> dict[str, Any]:
    return {"memory": _memory(value.claim), "derivation": value.derivation.value,
            "extractor": value.extractor or None,
            "sources": [_episode(e) for e in value.episodes],
            "superseded": [_memory(c) for c in value.superseded],
            "claim_links": [_link(k) for k in value.links]}


def _forget_preview(value: ForgetPreview) -> dict[str, Any]:
    return {"close": value.close,
            "matches": [{"memory_id": claim_id, "text": text}
                        for claim_id, text in value.matches.items()],
            "confirm": value.confirm, "expires_at": _instant(value.expires_at)}


def _edge(value: Edge) -> dict[str, Any]:
    return {"memory": _memory(value.claim), "backward": value.backward,
            "strength": value.strength}


def _path(value: Path) -> dict[str, Any]:
    return {"nodes": list(value.nodes), "labels": list(value.labels), "hops": value.hops,
            "score": value.score, "edges": [_edge(e) for e in value.edges]}


def _paths_body(request: Request, valid_at: datetime | None, known_at: datetime | None,
                found: Sequence[Path]) -> dict[str, Any]:
    return {"as_of": request.param("as_of"), "valid_at": _instant(valid_at),
            "known_at": _instant(known_at), "count": len(found),
            "paths": [_path(p) for p in found]}


def _reading(value: Reading) -> dict[str, Any]:
    return {"subject": value.subject, "predicate": value.predicate,
            "now": [_memory(c) for c in value.now],
            "then": [_memory(c) for c in value.then],
            "stated": [_memory(c) for c in value.stated],
            "diverged": value.diverged, "moved": value.moved}


def _answer(value: Answer) -> dict[str, Any]:
    return {"question": value.question, "at": _instant(value.at),
            "count": len(value.readings),
            "readings": [_reading(r) for r in value.readings], "text": value.text}


def _row(value: Row) -> dict[str, Any]:
    return {"claim_id": value.claim_id, "text": value.text, "inferred": value.inferred}


def _profile_body(value: Profile) -> dict[str, Any]:
    return {"standing": [_row(r) for r in value.standing],
            "recent": [_row(r) for r in value.recent],
            "relevant": [_row(r) for r in value.relevant],
            "buckets": {name: [_row(r) for r in rows]
                        for name, rows in value.buckets.items()},
            "warnings": list(value.warnings)}


def _document(doc: Document) -> dict[str, Any]:
    return {"id": doc.id, "scope": _scope(doc.scope), "custom_id": doc.custom_id,
            "title": doc.title, "filepath": doc.filepath, "source_uri": doc.source_uri,
            "mime": doc.mime, "content_hash": doc.content_hash, "status": doc.status,
            "error": doc.error, "metadata": dict(doc.meta),
            "created_at": _instant(doc.created_at), "updated_at": _instant(doc.updated_at),
            "chunks": doc.chunks}


def _document_page(page: Page[Document]) -> dict[str, Any]:
    return {"documents": [_document(d) for d in page.items],
            "next_cursor": page.next_cursor}


def _deleted(value: DeleteResult) -> dict[str, Any]:
    return {"id": value.id, "deleted": value.deleted, "custom_id": value.custom_id,
            "chunks": value.chunks, "episodes": value.episodes,
            "retired": list(value.retired), "unlinked": list(value.unlinked)}


__all__ = ["API_KEY", "APPLIED_HEADER", "ApiError", "FakeV1", "TOKEN_ID"]
```

- [ ] **Step 4: Run the tests and see them pass.**

Run: `TEST tests/adversarial/fakes/test_adv_fake_v1_parity.py tests/adversarial/fakes/test_adv_fake_v1_faults.py`
Expected: `21 passed`.

- [ ] **Step 5: Document it.** Add this after the shared-mechanism paragraphs of the fakes section:

```markdown
**`FakeV1` is the hosted `/v1` REST API.** It answers the 34 routes that `RemoteMemvara` and `AsyncRemoteMemvara` call, from a real local `Memvara` with the hashing embedder and no model. It writes each answer the way memvara-cloud's `rest/render.py` does, so the client's own hydration code reads it back. A self-test reads the two clients' source and fails when either one calls a route the fake does not serve.

- `fake.remote(user="alice")` and `fake.aremote(...)` return a client wired to the fake through a mock transport. For a real URL, pass `fake.serve()` as `base_url` with `api_key=fake.api_key`. That is also how to start an MCP server in cloud mode against it: `MEMVARA_MODE=cloud`, `MEMVARA_API_KEY` and `MEMVARA_SERVER_URL`.
- A route's name is its method and path template, such as `POST /v1/facts` or `GET /v1/memories/{id}`. `FakeV1.ROUTES` lists them.
- The credential is one key, bound to the whole tenant with the admin privilege, so a client may narrow to any user. A wrong key is a 401, an agent or a session named without a user is a 400, and `FakeV1(read_only=True)` refuses every write with a 403.
- A write retried with the same `Idempotency-Key` is carried out once, as on a deployment with one worker.
- What it leaves out: allowances and rate limits (inject a 402 or a 429 instead), legal holds, the audit trail, OAuth, and a document added by `url`, which it refuses because the suite runs offline. `POST /v1/maintenance/consolidate` runs the pass before it answers, where the cloud answers first.
- `/v1` does not carry a claim's `temporal_precision`, `object_kind`, `amount` or `unit`, so a claim read through the remote client has the default in each of them. A test that compares a local store with a remote one leaves those four out.
```

- [ ] **Step 6: Type-check and commit.**

```bash
$PY -m mypy tests/harness
$PY -m mypy tests/harness --ignore-missing-imports
git add tests/harness/fakes/fake_v1.py tests/adversarial/fakes/conftest.py \
    tests/adversarial/fakes/test_adv_fake_v1_parity.py \
    tests/adversarial/fakes/test_adv_fake_v1_faults.py docs/claude/testing.md
git commit -m "Add FakeV1, the hosted REST API answered by a local store"
```

---

## Task 3: `FakeHostedMcp`, the hosted `/mcp` endpoint

**Files:**
- Create: `tests/harness/fakes/hosted_mcp.py`, `tests/adversarial/fakes/test_adv_fake_hosted_mcp.py`
- Modify: `tests/adversarial/fakes/conftest.py` (the `hosted_mcp` fixture), `docs/claude/testing.md`

**Interfaces:**
- Consumes: Task 1's mechanism; `MemvaraMCPServer` and `TOOLS` from `memvara.server`.
- Produces: `FakeHostedMcp(memvara=None, *, tenant="default", user="tester", api_key=API_KEY, read_only=False, sse=False)` with `memvara`, `issued` (every session id, in order), `expire_sessions()`, and route names `GET /mcp`, each JSON-RPC method, and `tools/call <tool>` for every tool; `API_KEY`, `SESSION_HEADER`, `PROJECT_HEADER`, `APPLIED_HEADER`.

The behaviour follows memvara-cloud's `rest/mcp.py`. What the clients send was read from `plugin/hooks/lib/hosted.py` and `npm/memvara/lib/transport.js` and `bin/memvara.js`.

- [ ] **Step 1: Write the failing test.** Add the fixture to `conftest.py`:

```python
from harness.fakes.hosted_mcp import FakeHostedMcp


@pytest.fixture
def hosted_mcp() -> Iterator[FakeHostedMcp]:
    with FakeHostedMcp() as fake:
        yield fake
```

and the test file:

`tests/adversarial/fakes/test_adv_fake_hosted_mcp.py`:

```python
"""FakeHostedMcp answers both clients that reach the hosted /mcp endpoint: the plugin
hooks' hosted client and the npm bridge."""

from __future__ import annotations

import importlib
import json
import pathlib
import shutil
import subprocess
import sys
from types import ModuleType

import httpx
import pytest

from harness.env import REPO, child_env
from harness.fakes.hosted_mcp import API_KEY, FakeHostedMcp

HOOKS = REPO / "plugin" / "hooks"
BRIDGE = REPO / "npm" / "memvara" / "bin" / "memvara.js"


@pytest.fixture
def hosted() -> ModuleType:
    """The hooks' own hosted client, `plugin/hooks/lib/hosted.py`."""
    if str(HOOKS) not in sys.path:
        sys.path.insert(0, str(HOOKS))
    return importlib.import_module("lib.hosted")


def _methods(fake: FakeHostedMcp) -> list[str | None]:
    return [json.loads(r.body).get("method") if r.body else None for r in fake.requests]


def _node_runs() -> bool:
    """Whether node starts at all. `which node` is not enough, as
    tests/test_npm_release.py found on a machine whose node aborted at start."""
    if shutil.which("node") is None:
        return False
    try:
        subprocess.run(["node", "-e", "process.exit(0)"], check=True, capture_output=True,
                       timeout=8)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return True


def test_the_hooks_client_remembers_and_recalls_through_the_fake(
        hosted_mcp: FakeHostedMcp, hosted: ModuleType) -> None:
    client = hosted.HostedRecall(API_KEY, hosted_mcp.serve())
    try:
        assert "Lisbon" in client.remember("user", "lives_in", "Lisbon")
        assert "Lisbon" in client.recall("where does the user live")
    finally:
        client.close()
    assert _methods(hosted_mcp)[:3] == ["initialize", "notifications/initialized",
                                        "tools/call"]
    first, *rest = hosted_mcp.requests
    assert first.header("mcp-session-id") is None
    assert {r.header("mcp-session-id") for r in rest} == {hosted_mcp.issued[0]}
    assert {r.header("authorization") for r in hosted_mcp.requests} == {f"Bearer {API_KEY}"}
    assert {r.header("user-agent") for r in hosted_mcp.requests} == {hosted.USER_AGENT}
    assert [c.object for c in hosted_mcp.memvara.scope(user="tester").get_all()] \
        == ["Lisbon"]


def test_a_wrong_key_is_refused_with_the_header_a_client_signs_in_from(
        hosted_mcp: FakeHostedMcp, hosted: ModuleType) -> None:
    client = hosted.HostedRecall("mv_wrong", hosted_mcp.serve())
    try:
        with pytest.raises(hosted.HostedError) as caught:
            client.recall("anything")
    finally:
        client.close()
    assert caught.value.status == 401
    ping = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
    with httpx.Client(base_url=hosted_mcp.MOCK_URL, transport=hosted_mcp.transport()) as raw:
        missing = raw.post("/mcp", json=ping)
        wrong = raw.post("/mcp", json=ping, headers={"Authorization": "Bearer mv_wrong"})
    metadata = 'resource_metadata="http://fake.invalid/.well-known/oauth-protected-resource"'
    assert missing.status_code == 401 and metadata in missing.headers["www-authenticate"]
    assert wrong.status_code == 401
    assert wrong.headers["www-authenticate"].endswith('error="invalid_token"')


def test_every_request_after_initialize_needs_a_session_the_fake_issued(
        hosted_mcp: FakeHostedMcp) -> None:
    ping = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
    auth = {"Authorization": f"Bearer {API_KEY}"}
    with httpx.Client(base_url=hosted_mcp.MOCK_URL, transport=hosted_mcp.transport(),
                      headers=auth) as raw:
        assert raw.post("/mcp", json=ping).status_code == 400
        assert raw.post("/mcp", json=ping,
                        headers={"mcp-session-id": "made-up"}).status_code == 404
        hello = raw.post("/mcp", json={"jsonrpc": "2.0", "id": 2, "method": "initialize",
                                       "params": {"protocolVersion": "2025-06-18",
                                                  "capabilities": {}}})
        session = {"mcp-session-id": hello.headers["mcp-session-id"]}
        notified = raw.post("/mcp", json={"jsonrpc": "2.0",
                                          "method": "notifications/initialized"},
                            headers=session)
        stream = raw.get("/mcp", headers=session)
    assert hello.status_code == 200 and hello.json()["result"]["serverInfo"]["name"]
    assert (notified.status_code, notified.content) == (202, b"")
    assert stream.status_code == 405


def test_a_client_holding_an_expired_session_shakes_hands_again(
        hosted_mcp: FakeHostedMcp, hosted: ModuleType) -> None:
    client = hosted.HostedRecall(API_KEY, hosted_mcp.serve())
    try:
        client.remember("user", "lives_in", "Lisbon")
        hosted_mcp.expire_sessions()
        assert "Lisbon" in client.recall("where does the user live")
    finally:
        client.close()
    assert _methods(hosted_mcp).count("initialize") == 2
    assert 404 in [r.status for r in hosted_mcp.requests]
    assert len(hosted_mcp.issued) == 2


def test_one_tool_can_fail_while_the_handshake_still_works(
        hosted_mcp: FakeHostedMcp, hosted: ModuleType) -> None:
    hosted_mcp.fail("tools/call memory_recall", 402, headers={"Retry-After": "3600"})
    client = hosted.HostedRecall(API_KEY, hosted_mcp.serve())
    try:
        with pytest.raises(hosted.HostedError) as caught:
            client.recall("anything")
    finally:
        client.close()
    assert (caught.value.status, caught.value.code, caught.value.retry_after) == (
        402, "quota_exhausted", 3600)
    assert _methods(hosted_mcp)[:2] == ["initialize", "notifications/initialized"]


def test_a_project_header_binds_the_project_and_is_reported_back(
        hosted_mcp: FakeHostedMcp, hosted: ModuleType,
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(hosted.PROJECT_ENV, "github.com/acme/app")
    client = hosted.HostedRecall(API_KEY, hosted_mcp.serve())
    try:
        client.remember("api", "depends_on", "postgres")
    finally:
        client.close()
    assert {r.header("memvara-project") for r in hosted_mcp.requests} \
        == {"github.com/acme/app"}
    in_project = hosted_mcp.memvara.scope(user="tester", project="github.com/acme/app")
    assert [c.object for c in in_project.get_all()] == ["postgres"]
    assert hosted_mcp.memvara.scope(user="tester", project="github.com/acme/web").get_all() \
        == []


def test_the_hooks_client_reads_a_reply_sent_as_an_event_stream(hosted: ModuleType) -> None:
    with FakeHostedMcp(sse=True) as fake:
        client = hosted.HostedRecall(API_KEY, fake.serve())
        try:
            client.remember("user", "lives_in", "Lisbon")
            assert "Lisbon" in client.recall("where does the user live")
        finally:
            client.close()


@pytest.mark.skipif(not _node_runs(), reason="node/npm missing or unloadable")
@pytest.mark.parametrize("sse", [False, True], ids=["json", "event-stream"])
def test_the_npm_bridge_carries_its_key_and_session_to_the_fake(
        sse: bool, tmp_path: pathlib.Path) -> None:
    lines = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": {"name": "memvara-adversarial-suite", "version": "0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    ]
    with FakeHostedMcp(sse=sse) as fake:
        done = subprocess.run(
            ["node", str(BRIDGE), "--server", fake.serve()],
            input="".join(json.dumps(line) + "\n" for line in lines), capture_output=True,
            text=True, encoding="utf-8", timeout=60, cwd=str(tmp_path),
            env=child_env(tmp_path, {"MEMVARA_API_KEY": API_KEY}))
    assert done.returncode == 0, done.stderr
    replies = [json.loads(line) for line in done.stdout.splitlines()]
    assert [reply["id"] for reply in replies] == [1, 2]
    assert "memory_recall" in {tool["name"] for tool in replies[1]["result"]["tools"]}
    assert {r.header("authorization") for r in fake.requests} == {f"Bearer {API_KEY}"}
    agents = {r.header("user-agent") or "" for r in fake.requests}
    assert len(agents) == 1 and agents.pop().startswith("memvara-npm/")
    assert [r.header("mcp-session-id") for r in fake.requests] == [
        None, fake.issued[0], fake.issued[0]]
```

- [ ] **Step 2: Run it and see it fail.**

Run: `TEST tests/adversarial/fakes/test_adv_fake_hosted_mcp.py`
Expected: a collection error, `ModuleNotFoundError: No module named 'harness.fakes.hosted_mcp'`.

- [ ] **Step 3: Write the fake.**

`tests/harness/fakes/hosted_mcp.py`:

```python
"""A fake of the hosted `/mcp` endpoint, answered by the real MCP server code.

Two clients reach app.memvara.dev's `/mcp`:

* the plugin hooks' hosted client, `plugin/hooks/lib/hosted.py`, which the hooks use on an
  install signed in to the hosted service; and
* the npm bridge, `npm/memvara/lib/transport.js`, which `npx memvara` runs.

Both POST one JSON-RPC message per request, and both send `Authorization: Bearer <key>`,
a `User-Agent` of their own, `Accept: application/json, text/event-stream`, and, after
`initialize`, the `mcp-session-id` the server issued. The hooks also send
`memvara-project` when project scope is on. Both read a reply that arrives as plain JSON
or as server-sent events.

`FakeHostedMcp` answers the way memvara-cloud's `rest/mcp.py` does, with a
`MemvaraMCPServer` over a local `Memvara`, bound to the credential's scope:

* No bearer token, or a wrong one, is a 401. It carries the `WWW-Authenticate` header an
  MCP client reads to find the authorization server.
* `initialize` issues a session id in the `mcp-session-id` header. Every later request must
  carry an id this fake issued: none is a 400, and an unknown one is a 404.
* A notification is a 202 with no body, and `GET /mcp` is a 405.
* A `memvara-project` header binds the project, and the reply names it in
  `Memvara-Project-Applied`.

With `sse=True` each reply is sent as one server-sent event instead of plain JSON.
`expire_sessions()` forgets every issued session, as a restarted deployment does, so a
test can check that a client shakes hands again.

A fault is injected by JSON-RPC method (`initialize`, `tools/list`, ...), a tool call by
`tools/call <tool>`, and the stream request by `GET /mcp`, so a test can fail one tool and
leave the handshake alone.
"""

from __future__ import annotations

import json
import secrets
from typing import Any

from memvara import Memvara
from memvara.project import check_project
from memvara.server.mcp import MemvaraMCPServer
from memvara.server.tools import TOOLS

from .. import stores
from ._http import HttpFake, Reply, Request, json_reply

#: The key a `FakeHostedMcp` accepts unless a test names another. It opens only this
#: in-process fake, so it is not a secret.
API_KEY = "mv_fake_mcp_key"

#: The header that carries a session, as MCP's Streamable HTTP transport names it.
SESSION_HEADER = "mcp-session-id"

#: The header that carries the project, and the one that reports it was applied.
PROJECT_HEADER = "memvara-project"
APPLIED_HEADER = "Memvara-Project-Applied"

#: The code an injected status carries when a test names no body. `/mcp` shares the
#: limiter with `/v1`, so a quota or rate refusal arrives in the `/v1` envelope, which is
#: the shape the hooks' client reads a refusal's code and detail from.
_CODES = {400: "bad_request", 401: "unauthorized", 402: "quota_exhausted",
          403: "forbidden", 404: "not_found", 409: "conflict", 429: "rate_limited",
          503: "unavailable"}


def _rpc_error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


class FakeHostedMcp(HttpFake):
    """The hosted `/mcp` endpoint, answered by `MemvaraMCPServer` over a local store.

    `memvara` is the store to answer from. Left out, the fake makes an in-memory one and
    closes it when the fake closes. `tenant` and `user` are the scope the credential
    binds, `api_key` the only key it accepts, `read_only` hides the write tools as a
    read-only credential does, and `sse` sends every reply as a server-sent event.
    """

    ROUTES = ("GET /mcp", "initialize", "notifications/initialized", "ping", "tools/list",
              *(f"tools/call {tool.name}" for tool in TOOLS))

    def __init__(self, memvara: Memvara | None = None, *, tenant: str = "default",
                 user: str | None = "tester", api_key: str = API_KEY,
                 read_only: bool = False, sse: bool = False) -> None:
        super().__init__()
        self._owns_store = memvara is None
        #: The store every tool call reads and writes.
        self.memvara = memvara if memvara is not None else stores.memory()
        self.tenant = tenant
        self.user = user
        self.api_key = api_key
        self.read_only = read_only
        self.sse = sse
        #: Every session id this fake issued, in order, including expired ones.
        self.issued: list[str] = []
        self._live: set[str] = set()

    def expire_sessions(self) -> None:
        """Forget every session issued so far, as a deployment does when it restarts."""
        with self._lock:
            self._live.clear()

    def close(self) -> None:
        super().close()
        if self._owns_store:
            self.memvara.close()

    # -- routing --------------------------------------------------------------------

    def route(self, request: Request) -> str | None:
        if request.path != "/mcp":
            return None
        if request.method == "GET":
            return "GET /mcp"
        try:
            message = request.json()
        except ValueError:
            return None
        if request.method != "POST" or not isinstance(message, dict):
            return None
        method = message.get("method")
        if method == "tools/call":
            params = message.get("params")
            return f"tools/call {params.get('name') if isinstance(params, dict) else None}"
        return method if isinstance(method, str) else None

    def error_body(self, status: int, route: str) -> Any:
        code = _CODES.get(status, "internal" if status >= 500 else "bad_request")
        return {"error": {"code": code, "message": f"FakeHostedMcp injected a {status} on "
                                                   f"{route}", "detail": None}}

    def respond(self, request: Request) -> Reply:
        if request.path != "/mcp":
            return json_reply(404, {"error": "not_found",
                                    "error_description": f"no route {request.path}"})
        if request.method not in ("GET", "POST"):
            return json_reply(405, {"error": "method_not_allowed",
                                    "error_description": f"{request.method} /mcp"})
        refused = self._authenticate(request)
        if refused is not None:
            return refused
        raw = (request.header(PROJECT_HEADER) or "").strip()
        try:
            project = check_project(raw) if raw else None
        except ValueError as exc:
            return json_reply(400, {"error": "bad_scope",
                                    "error_description": f"the Memvara-Project header is "
                                                         f"not usable: {exc}"})
        reply = self._get(request) if request.method == "GET" else self._post(request,
                                                                              project)
        if project is None:
            return reply
        return Reply(reply.status, reply.body, {**reply.headers, APPLIED_HEADER: project})

    def _authenticate(self, request: Request) -> Reply | None:
        """A 401 for a missing or wrong bearer token, or None. The header names where the
        protected resource's metadata is, which is how an MCP client finds the
        authorization server to sign in with."""
        origin = f"http://{request.header('host') or '127.0.0.1'}"
        challenge = f'Bearer resource_metadata="{origin}/.well-known/oauth-protected-resource"'
        header = request.header("authorization") or ""
        if not header.lower().startswith("bearer "):
            return json_reply(401, {"error": "unauthorized",
                                    "error_description": "this endpoint needs "
                                                         "'Authorization: Bearer <token>'."},
                              {"WWW-Authenticate": challenge})
        if header[len("Bearer "):].strip() != self.api_key:
            return json_reply(401, {"error": "unauthorized",
                                    "error_description": "this token is not one this "
                                                         "deployment issued"},
                              {"WWW-Authenticate": f'{challenge}, error="invalid_token"'})
        return None

    def _post(self, request: Request, project: str | None) -> Reply:
        try:
            message = request.json()
        except ValueError:
            return json_reply(400, _rpc_error(None, -32700, "invalid JSON"))
        if not isinstance(message, dict):
            return json_reply(400, _rpc_error(None, -32600, "expected a JSON-RPC object"))
        # A server per request, as the cloud builds one: it holds no state between
        # requests except the store, so there is nothing to keep alive. It is never
        # closed, because closing a server closes the store it was given.
        server = MemvaraMCPServer(self.memvara, tenant=self.tenant, user=self.user,
                                  project=project, read_only=self.read_only)
        if message.get("method") == "initialize":
            session = secrets.token_hex(16)
            with self._lock:
                self.issued.append(session)
                self._live.add(session)
            return self._message(server.handle_message(message), {SESSION_HEADER: session})
        session = request.header(SESSION_HEADER)
        if session is None:
            return json_reply(400, _rpc_error(
                message.get("id"), -32600,
                f"every request after 'initialize' needs a {SESSION_HEADER!r} header."))
        with self._lock:
            known = session in self._live
        if not known:
            return json_reply(404, _rpc_error(
                message.get("id"), -32001,
                "unrecognised or expired session; call 'initialize' again."))
        return self._message(server.handle_message(message))

    def _get(self, request: Request) -> Reply:
        session = request.header(SESSION_HEADER)
        if session is None:
            return json_reply(400, {"error": "bad_request",
                                    "error_description": f"{SESSION_HEADER!r} header "
                                                         "required."})
        with self._lock:
            known = session in self._live
        if not known:
            return json_reply(404, {"error": "not_found",
                                    "error_description": "unrecognised session."})
        return json_reply(405, {"error": "not_supported",
                                "error_description": "this deployment sends no "
                                                     "server-initiated messages; there is "
                                                     "no stream to open."})

    def _message(self, answer: dict[str, Any] | None,
                 headers: dict[str, str] | None = None) -> Reply:
        """One JSON-RPC reply as JSON or as one server-sent event, or a 202 with no body
        for a notification, which JSON-RPC forbids answering."""
        if answer is None:
            return Reply(202, b"", dict(headers or {}))
        if self.sse:
            event = f"event: message\ndata: {json.dumps(answer)}\n\n".encode("utf-8")
            return Reply(200, event, {"Content-Type": "text/event-stream",
                                      **(headers or {})})
        return json_reply(200, answer, headers)


__all__ = ["API_KEY", "APPLIED_HEADER", "FakeHostedMcp", "PROJECT_HEADER",
           "SESSION_HEADER"]
```

- [ ] **Step 4: Run the test and see it pass.**

Run: `TEST tests/adversarial/fakes/test_adv_fake_hosted_mcp.py`
Expected: `9 passed` where node runs, and `7 passed, 2 skipped` where it does not.

- [ ] **Step 5: Document it:**

```markdown
**`FakeHostedMcp` is the hosted `/mcp` endpoint** that the hooks' hosted client (`plugin/hooks/lib/hosted.py`) and the npm bridge reach. It answers with the real `MemvaraMCPServer` over a local store, bound to the credential's scope, which is the user `tester` unless the test names another. It checks what those clients send the way memvara-cloud's `rest/mcp.py` does. A request needs a bearer token, and without one it gets a 401 whose `WWW-Authenticate` header tells an MCP client where to sign in. `initialize` issues a session id, and every later request must carry one: none is a 400 and an unknown one a 404. A notification gets a 202 with no body, and a `memvara-project` header binds the project. A hook run reaches the fake when its environment sets `MEMVARA_API_KEY` to `fake.api_key` and `MEMVARA_SERVER_URL` to `fake.serve()`.

- `FakeHostedMcp(sse=True)` sends each reply as a server-sent event, which both clients must be able to read.
- `expire_sessions()` forgets every session, as a restarted deployment does, so a test can watch a client shake hands again. `fake.issued` lists every session id the fake has issued.
- A fault is keyed by JSON-RPC method, and a tool call by `tools/call <tool>`, so `fake.fail("tools/call memory_recall", 402)` refuses one tool and leaves the handshake alone.
```

- [ ] **Step 6: Type-check and commit.**

```bash
$PY -m mypy tests/harness
$PY -m mypy tests/harness --ignore-missing-imports
git add tests/harness/fakes/hosted_mcp.py tests/adversarial/fakes/conftest.py \
    tests/adversarial/fakes/test_adv_fake_hosted_mcp.py docs/claude/testing.md
git commit -m "Add FakeHostedMcp, the hosted MCP endpoint the hooks and the npm bridge reach"
```

---

## Task 4: `FakeOpenAI`, an OpenAI-compatible model endpoint

**Files:**
- Create: `tests/harness/fakes/openai_compat.py`, `tests/adversarial/fakes/test_adv_fake_openai.py`
- Modify: `tests/adversarial/fakes/conftest.py` (the `fake_openai` fixture), `docs/claude/testing.md`

**Interfaces:**
- Consumes: Task 1's mechanism; `OpenAILLM` from `memvara.llm.openai`.
- Produces: `FakeOpenAI()` with `add_reply`, `add_json`, `add_tool_calls`, `add_raw`, `add_rate_limit`, `add_hang`, `pending`, `base_url` and `client(api_key=..., timeout=...)`; `FakeOpenAIError(status, body, retry_after)`; `COMPLETIONS`, `MODEL`.

The requests and replies were read from `memvara/llm/openai.py` and `memvara/llm/_shape.py`. CI does not install the `openai` package, so `client()` is a stand-in for the SDK's transport that sends the same body over HTTP.

- [ ] **Step 1: Write the failing test.** Add the fixture to `conftest.py`:

```python
from harness.fakes.openai_compat import FakeOpenAI


@pytest.fixture
def fake_openai() -> Iterator[FakeOpenAI]:
    with FakeOpenAI() as fake:
        yield fake
```

and the test file:

`tests/adversarial/fakes/test_adv_fake_openai.py`:

```python
"""FakeOpenAI answers in the order it was scripted, records what was sent, and memvara's
own OpenAI client talks to it."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx
import pytest

from harness.fakes.openai_compat import FakeOpenAI, FakeOpenAIError
from memvara import Memvara
from memvara.embed import HashingEmbedder
from memvara.llm.base import Message, ToolSpec, TruncatedResponse
from memvara.llm.openai import OpenAILLM
from memvara.types import Episode

#: A turn the fast path does not recognise, so a store with a model has to ask it.
TURN = "I have been practising the cello every evening since spring."

#: What a model answers the extraction call with, in the shape `OpenAILLM` asks for.
CLAIMS = {"claims": [{"subject": "user", "predicate": "likes", "object": "the cello",
                      "polarity": 1, "memory_type": "semantic", "confidence": 0.9,
                      "source_index": 0, "when": None, "amount": None, "unit": None}]}


def _post(fake: FakeOpenAI, content: str) -> httpx.Response:
    return httpx.post(fake.base_url + "/chat/completions",
                      json={"model": "m", "messages": [{"role": "user", "content": content}]})


def test_replies_come_back_in_the_order_they_were_scripted(fake_openai: FakeOpenAI) -> None:
    fake_openai.add_reply("first")
    fake_openai.add_reply("second")
    answers = [_post(fake_openai, str(n)).json() for n in (1, 2)]
    assert [a["choices"][0]["message"]["content"] for a in answers] == ["first", "second"]
    assert [a["model"] for a in answers] == ["m", "m"]
    assert [r.json()["messages"][0]["content"] for r in fake_openai.requests] == ["1", "2"]
    assert fake_openai.pending == 0
    exhausted = _post(fake_openai, "3")
    assert exhausted.status_code == 500 and "no scripted reply left" in exhausted.text


def test_memvara_extracts_a_claim_through_the_fake(fake_openai: FakeOpenAI) -> None:
    fake_openai.add_json(CLAIMS, prompt_tokens=12, completion_tokens=7)
    mem = Memvara(embedder=HashingEmbedder(dim=512),
                  llm=OpenAILLM(client=fake_openai.client(), model="fake-model"))
    try:
        receipt = mem.scope(user="alice").add(TURN)
    finally:
        mem.close()
    assert [(c.predicate, c.object) for c in receipt.added] == [("likes", "the cello")]
    assert (receipt.llm_calls, receipt.tokens_in, receipt.tokens_out) == (1, 12, 7)
    (sent,) = fake_openai.requests
    body = sent.json()
    assert body["model"] == "fake-model"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert TURN in body["messages"][1]["content"]
    assert sent.header("authorization") == "Bearer sk-fake"


def test_output_the_client_cannot_use_extracts_nothing(fake_openai: FakeOpenAI) -> None:
    fake_openai.add_reply("Sure! Here are the facts you asked for.")
    fake_openai.add_raw({"id": "x", "choices": []})
    fake_openai.add_reply(json.dumps(CLAIMS), finish_reason="length")
    fake_openai.add_raw("<html>not a completion</html>")
    llm = OpenAILLM(client=fake_openai.client())
    turn = [Episode(content=TURN)]
    assert llm.extract(turn, []) == []
    assert llm.extract(turn, []) == []
    with pytest.raises(TruncatedResponse):
        llm.extract(turn, [])
    with pytest.raises(ValueError):
        llm.extract(turn, [])


def test_a_429_reaches_the_client_and_the_store_still_keeps_the_turn(
        fake_openai: FakeOpenAI) -> None:
    fake_openai.add_rate_limit(retry_after=7)
    fake_openai.add_rate_limit(retry_after=None)
    llm = OpenAILLM(client=fake_openai.client())
    with pytest.raises(FakeOpenAIError) as caught:
        llm.chat("system", "prompt", json_object=False, max_completion_tokens=10, timeout=5)
    assert (caught.value.status, caught.value.retry_after) == (429, "7")
    mem = Memvara(embedder=HashingEmbedder(dim=512), llm=llm)
    try:
        receipt = mem.scope(user="alice").add(TURN)
        assert (receipt.added, receipt.unextracted, len(receipt.episode_ids)) == ([], 1, 1)
        assert mem.stats()["episodes"] == 1
    finally:
        mem.close()


def test_a_hang_is_cut_off_by_the_client_s_timeout(fake_openai: FakeOpenAI) -> None:
    fake_openai.add_hang()
    fake_openai.add_reply("after the hang")
    llm = OpenAILLM(client=fake_openai.client())
    started = time.monotonic()
    with pytest.raises(httpx.ReadTimeout):
        llm.chat("system", "prompt", json_object=False, max_completion_tokens=10,
                 timeout=0.3)
    assert 0.3 <= time.monotonic() - started < 5
    assert llm.chat("system", "prompt", json_object=False, max_completion_tokens=10,
                    timeout=5) == "after the hang"


def test_a_tool_call_runs_the_tool_and_its_result_goes_back_to_the_model(
        fake_openai: FakeOpenAI) -> None:
    fake_openai.add_tool_calls(("lookup", {"key": "city"}))
    fake_openai.add_reply("The user lives in Lisbon.")
    asked: list[dict[str, Any]] = []

    def lookup(arguments: dict[str, Any]) -> str:
        asked.append(arguments)
        return "Lisbon"

    tool = ToolSpec("lookup", "Look a stored value up by its key.",
                    {"type": "object", "properties": {"key": {"type": "string"}},
                     "required": ["key"], "additionalProperties": False}, lookup)
    run = OpenAILLM(client=fake_openai.client()).run_tools(
        "system", [Message("user", "Where does the user live?")], [tool], max_steps=3,
        timeout=10)
    assert (run.steps, run.requests, run.finished, run.text) == (
        2, 2, True, "The user lives in Lisbon.")
    assert asked == [{"key": "city"}]
    assert fake_openai.requests[1].json()["messages"][-1] == {
        "role": "tool", "tool_call_id": "call_0", "content": "Lisbon"}


def test_a_route_fault_applies_before_the_script_and_leaves_it_alone(
        fake_openai: FakeOpenAI) -> None:
    fake_openai.fail("POST /v1/chat/completions", 503, times=1)
    fake_openai.add_reply("still first")
    assert _post(fake_openai, "1").status_code == 503
    assert _post(fake_openai, "2").json()["choices"][0]["message"]["content"] == "still first"
```

- [ ] **Step 2: Run it and see it fail.**

Run: `TEST tests/adversarial/fakes/test_adv_fake_openai.py`
Expected: a collection error, `ModuleNotFoundError: No module named 'harness.fakes.openai_compat'`.

- [ ] **Step 3: Write the fake.**

`tests/harness/fakes/openai_compat.py`:

```python
"""A fake OpenAI-compatible chat-completions endpoint that returns scripted replies.

memvara reaches an OpenAI-compatible model through `memvara.llm.openai.OpenAILLM`. It
calls `client.chat.completions.create(...)` with `model`, `messages`, and, depending on the
call, `response_format`, `tools`, `temperature`, `max_completion_tokens`, `timeout` and
`extra_body`. From the reply it reads `choices[0].message` (its `content`, `refusal` and
`tool_calls`), `choices[0].finish_reason`, and `usage.prompt_tokens` and
`usage.completion_tokens`. With no client given, `OpenAILLM` builds an `openai.OpenAI`
pointed at `OPENAI_BASE_URL`, which is how a server started with `MEMVARA_LLM=openai`
reaches a self-hosted model.

`FakeOpenAI` is such an endpoint: `POST /v1/chat/completions` on 127.0.0.1. It answers
each request with the next reply a test scripted, in order, and records every request.
A scripted reply can be a completion, tool calls, a raw body the client cannot parse, a
429 or a hang. A request that finds no scripted reply left gets a 500 that says so, so a
test that made one call more than it expected fails loudly.

CI does not install the `openai` package, so `client()` returns a stand-in for its
transport. `OpenAILLM(client=fake.client())` sends each call over HTTP to this fake, as
the SDK would send it, and gets back the decoded JSON, which `OpenAILLM` reads the way it
reads the SDK's objects. Unlike the SDK, the stand-in does not retry: the SDK retries a
429 or a timeout twice by default, so a test that wants to see a retry scripts one.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import httpx

from ._http import HttpFake, Reply, Request, Step, json_reply

#: The model a completion names when the request named none.
MODEL = "fake-model"

#: The route every completion is requested on, relative to the base URL.
COMPLETIONS = "POST /v1/chat/completions"


@dataclass(frozen=True)
class _Scripted:
    #: None for a hang; otherwise builds the reply from the request it answers.
    build: Callable[[Request, int], Reply] | None


class FakeOpenAIError(RuntimeError):
    """The fake answered with an error status. `client()` raises it where the SDK would
    raise one of its `APIStatusError` subclasses."""

    def __init__(self, status: int, body: str, retry_after: str | None) -> None:
        super().__init__(f"the model endpoint answered {status}: {body[:300]}")
        self.status = status
        self.body = body
        self.retry_after = retry_after


class FakeOpenAI(HttpFake):
    """An OpenAI-compatible chat-completions endpoint with scripted replies.

    Reached over a socket only: `serve()` starts it, and `client()` and `base_url` start it
    as well. A route fault injected with `fail`, `delay` or `hang` applies before the
    script, and does not use up a scripted reply.
    """

    ROUTES = (COMPLETIONS,)

    def __init__(self) -> None:
        super().__init__()
        self._script: deque[_Scripted] = deque()
        self._answered = 0
        self._script_lock = threading.Lock()

    @property
    def base_url(self) -> str:
        """What to set `OPENAI_BASE_URL` to. Starts serving if it has not."""
        return self.serve() + "/v1"

    @property
    def pending(self) -> int:
        """How many scripted replies have not been used yet."""
        with self._script_lock:
            return len(self._script)

    # -- the script -----------------------------------------------------------------

    def add_reply(self, content: str, *, finish_reason: str = "stop",
                  prompt_tokens: int = 10, completion_tokens: int = 5) -> None:
        """A completion whose message says `content`."""
        message = {"role": "assistant", "content": content, "refusal": None}
        self._push(lambda request, number: _completion(
            request, number, message, finish_reason, prompt_tokens, completion_tokens))

    def add_json(self, value: Any, **options: Any) -> None:
        """A completion whose content is `value` written as JSON, which is the shape a
        structured-output call is answered in."""
        self.add_reply(json.dumps(value), **options)

    def add_tool_calls(self, *calls: tuple[str, Mapping[str, Any]],
                       prompt_tokens: int = 10, completion_tokens: int = 5) -> None:
        """A completion that asks for tool calls: each `(name, arguments)` pair."""
        message = {"role": "assistant", "content": None, "refusal": None,
                   "tool_calls": [{"id": f"call_{index}", "type": "function",
                                   "function": {"name": name,
                                                "arguments": json.dumps(dict(arguments))}}
                                  for index, (name, arguments) in enumerate(calls)]}
        self._push(lambda request, number: _completion(
            request, number, message, "tool_calls", prompt_tokens, completion_tokens))

    def add_raw(self, body: Any, *, status: int = 200,
                headers: Mapping[str, str] | None = None) -> None:
        """Exactly this body: JSON for a dict or a list, raw for `str` or `bytes`. For a
        reply the client cannot parse, or one shaped wrongly."""
        if isinstance(body, (str, bytes)):
            raw = body.encode("utf-8") if isinstance(body, str) else body
            reply = Reply(status, raw, dict(headers or {}))
        else:
            reply = json_reply(status, body, headers)
        self._push(lambda request, number: reply)

    def add_rate_limit(self, *, retry_after: float | None = 1.0,
                       message: str = "Rate limit reached for requests") -> None:
        """A 429 in OpenAI's error shape, with `Retry-After` unless it is None."""
        headers = {} if retry_after is None else {"Retry-After": f"{retry_after:g}"}
        body = {"error": {"message": message, "type": "requests", "param": None,
                          "code": "rate_limit_exceeded"}}
        self._push(lambda request, number: json_reply(429, body, headers))

    def add_hang(self) -> None:
        """No answer at all: the request is held until the client gives up or the fake
        closes."""
        with self._script_lock:
            self._script.append(_Scripted(None))

    def _push(self, build: Callable[[Request, int], Reply]) -> None:
        with self._script_lock:
            self._script.append(_Scripted(build))

    # -- serving --------------------------------------------------------------------

    def route(self, request: Request) -> str | None:
        return COMPLETIONS if (request.method, request.path) == (
            "POST", "/v1/chat/completions") else None

    def error_body(self, status: int, route: str) -> Any:
        kind = "server_error" if status >= 500 else "invalid_request_error"
        return {"error": {"message": f"FakeOpenAI injected a {status} on {route}",
                          "type": kind, "param": None, "code": None}}

    def plan(self, request: Request) -> Step:
        fault = super().plan(request)
        if fault != Step() or request.route is None:
            return fault
        with self._script_lock:
            self._answered += 1
            number = self._answered
            scripted = self._script.popleft() if self._script else None
        if scripted is None:
            return Step(reply=json_reply(500, {"error": {
                "message": f"FakeOpenAI has no scripted reply left for request {number}",
                "type": "server_error", "param": None, "code": "script_exhausted"}}))
        if scripted.build is None:
            return Step(hang=True)
        return Step(reply=scripted.build(request, number))

    def respond(self, request: Request) -> Reply:
        return json_reply(404, {"error": {"message": f"no route {request.method} "
                                                     f"{request.path}",
                                          "type": "invalid_request_error", "param": None,
                                          "code": "not_found"}})

    # -- a client -------------------------------------------------------------------

    def client(self, *, api_key: str = "sk-fake", timeout: float = 10.0) -> Any:
        """An object shaped like an `openai.OpenAI` client, for `OpenAILLM(client=...)`.

        Its `chat.completions.create(**kwargs)` sends the keyword arguments as the JSON
        body, the way the SDK does: `timeout` sets how long to wait, `extra_body` is
        merged into the body, and everything else is sent as given. It returns the
        decoded reply, raises `FakeOpenAIError` for an error status, and lets
        `httpx.ReadTimeout` through when the reply does not come in time.
        """
        return _Client(self.base_url, api_key, timeout)


def _completion(request: Request, number: int, message: Mapping[str, Any],
                finish_reason: str, prompt_tokens: int, completion_tokens: int) -> Reply:
    """A chat completion in OpenAI's shape, naming the model the request asked for."""
    try:
        sent = request.json()
    except ValueError:
        sent = None
    model = sent.get("model") if isinstance(sent, dict) else None
    return json_reply(200, {
        "id": f"chatcmpl-fake-{number}", "object": "chat.completion",
        "created": int(time.time()), "model": model or MODEL,
        "choices": [{"index": 0, "message": dict(message), "finish_reason": finish_reason,
                     "logprobs": None}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens}})


class _Completions:
    def __init__(self, base_url: str, api_key: str, timeout: float) -> None:
        self._url = base_url + "/chat/completions"
        self._key = api_key
        self._timeout = timeout

    def create(self, **kwargs: Any) -> Any:
        timeout = kwargs.pop("timeout", None)
        body = dict(kwargs)
        body.update(body.pop("extra_body", None) or {})
        response = httpx.post(self._url, json=body,
                              headers={"Authorization": f"Bearer {self._key}"},
                              timeout=self._timeout if timeout is None else timeout)
        if response.status_code >= 400:
            raise FakeOpenAIError(response.status_code, response.text,
                                  response.headers.get("retry-after"))
        return response.json()


class _Chat:
    def __init__(self, completions: _Completions) -> None:
        self.completions = completions


class _Client:
    """`client.chat.completions.create`, and nothing else of the SDK."""

    def __init__(self, base_url: str, api_key: str, timeout: float) -> None:
        self.chat = _Chat(_Completions(base_url, api_key, timeout))


__all__ = ["COMPLETIONS", "FakeOpenAI", "FakeOpenAIError", "MODEL"]
```

- [ ] **Step 4: Run the test and see it pass.**

Run: `TEST tests/adversarial/fakes/test_adv_fake_openai.py`
Expected: `7 passed`.

- [ ] **Step 5: Document it:**

```markdown
**`FakeOpenAI` is an OpenAI-compatible chat-completions endpoint.** It answers each request with the next reply a test scripted, in order, and records every request. `add_reply`, `add_json` and `add_tool_calls` script a completion; `add_raw` scripts a body the client cannot use; `add_rate_limit` scripts a 429; and `add_hang` scripts no answer at all. A request that finds no reply left gets a 500 that says so, and `fake.pending` counts the replies not used yet.

- **In the test process,** `OpenAILLM(client=fake.client())` talks to it. CI does not install the `openai` package, so `client()` is a small stand-in for the SDK's transport. It sends each call over HTTP the way the SDK would, and hands back the decoded JSON, which `OpenAILLM` reads as it reads the SDK's own objects. It does not retry, whereas the SDK retries a 429 or a timeout twice by default, so a test that wants a retry scripts it.
- **In a child process,** a server started with `MEMVARA_LLM=openai` reaches the fake through `OPENAI_BASE_URL` set to `fake.base_url`, with any `OPENAI_API_KEY`. That needs the `openai` package installed in the environment the child runs in.
```

- [ ] **Step 6: Type-check and commit.**

```bash
$PY -m mypy tests/harness
$PY -m mypy tests/harness --ignore-missing-imports
git add tests/harness/fakes/openai_compat.py tests/adversarial/fakes/conftest.py \
    tests/adversarial/fakes/test_adv_fake_openai.py docs/claude/testing.md
git commit -m "Add FakeOpenAI, a scripted OpenAI-compatible model endpoint"
```

---

## Task 5: The fake agent CLIs

**Files:**
- Create: `tests/harness/fakes/cli.py`, `tests/adversarial/fakes/test_adv_fake_cli.py`
- Modify: `tests/harness/skips.py` (one rule), `docs/claude/testing.md`

**Interfaces:**
- Consumes: `harness.env.child_env`, `harness.hooks.host_record`.
- Produces: `FakeClis(directory)` with `bin`, `script(name, *replies)`, `calls(name) -> list[CliCall]` and `path(rest=None)`; `CliReply(text, stdout, stderr, exit_code, is_error, usage, events)`; `CliCall(argv, stdin)`; `NAMES`, `USAGE`, `EXHAUSTED`.

How capture runs a CLI and reads its output was read from `plugin/hooks/lib/extract.py`, `lib/agentic.py` and `core/host.py`.

- [ ] **Step 1: Write the failing test.**

`tests/adversarial/fakes/test_adv_fake_cli.py`:

```python
"""The fake agent CLIs are the ones a child process finds on PATH, record how they were
started, and print their scripted reply in the format the capture hook reads."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
from typing import Any

import pytest

from harness.env import REPO, child_env
from harness.fakes.cli import EXHAUSTED, USAGE, CliCall, CliReply, FakeClis
from harness.hooks import host_record

pytestmark = pytest.mark.skipif(sys.platform == "win32",
                                reason="the fake agent CLIs are POSIX shell scripts")

HOOKS = REPO / "plugin" / "hooks"

#: A reply holding what a careless envelope would mangle: a newline, quotes, a backslash,
#: characters outside ASCII, and JSON inside the text.
REPLY = 'Line one\n"quoted" \\ back — ünïcode ✓\n{"facts": []}'

#: Runs in a child process. It binds a host the way `plugin/hooks/run.py` does, then asks
#: the capture hook's own extraction code to start that host's CLI and read its reply.
EXTRACT = r"""
import json, shutil, sys
sys.path.insert(0, sys.argv[1])
from core import host
host.use(__import__("hosts." + sys.argv[2], fromlist=["HOST"]).HOST)
from lib import extract
reply, usage, model = extract._payload("TURN", "PROMPT ")
print(json.dumps({"found": shutil.which(sys.argv[2]), "reply": reply, "usage": usage}))
"""

#: Runs in a child process: the agentic capture run, which reads `claude`'s stream-json.
AGENTIC = r"""
import json, os, sys
sys.path.insert(0, sys.argv[1])
from core import host
from hosts import claude
host.use(claude.HOST)
from lib import agentic
command = agentic.argv("RULES", "/no/such/config.json", "DATA")
run = agentic._run(command, dict(os.environ))
print(json.dumps({"failure": run.failure, "argv": command[1:],
                  "result": (run.watch.result or {}).get("result")}))
"""


@pytest.fixture
def fakes(tmp_path: pathlib.Path) -> FakeClis:
    return FakeClis(tmp_path / "bin")


@pytest.fixture
def home(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    return tmp_path_factory.mktemp("cli-home")


def _child(script: str, *args: str, fakes: FakeClis, home: pathlib.Path,
           cwd: pathlib.Path) -> Any:
    done = subprocess.run([sys.executable, "-c", script, str(HOOKS), *args],
                          capture_output=True, text=True, encoding="utf-8", timeout=60,
                          env=child_env(home, {"PATH": fakes.path()}), cwd=str(cwd),
                          stdin=subprocess.DEVNULL)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.splitlines()[-1])


def _run(fakes: FakeClis, name: str, home: pathlib.Path, *argv: str,
         stdin: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(fakes.bin / name), *argv], input=stdin, capture_output=True,
                          text=True, encoding="utf-8", timeout=30, env=child_env(home))


def test_the_claude_capture_starts_is_the_fake_and_its_reply_arrives_exactly(
        fakes: FakeClis, home: pathlib.Path, tmp_path: pathlib.Path) -> None:
    fakes.script("claude", REPLY)
    got = _child(EXTRACT, "claude", fakes=fakes, home=home, cwd=tmp_path)
    assert got["found"] == str(fakes.bin / "claude")
    assert (got["reply"], got["usage"]) == (REPLY, dict(USAGE))
    assert fakes.calls("claude") == [
        CliCall(argv=[*host_record("claude").extractor.argv[1:], "PROMPT TURN"], stdin="")]
    assert fakes.calls("codex") == []


def test_the_codex_capture_starts_is_the_fake_and_its_events_are_read(
        fakes: FakeClis, home: pathlib.Path, tmp_path: pathlib.Path) -> None:
    fakes.script("codex", CliReply(text=REPLY, usage={"input_tokens": 3, "output_tokens": 4}))
    got = _child(EXTRACT, "codex", fakes=fakes, home=home, cwd=tmp_path)
    assert got["found"] == str(fakes.bin / "codex")
    assert (got["reply"], got["usage"]) == (REPLY, {"input_tokens": 3, "output_tokens": 4})
    assert fakes.calls("codex") == [
        CliCall(argv=[*host_record("codex").extractor.argv[1:], "PROMPT TURN"], stdin="")]
    assert fakes.calls("claude") == []


def test_the_agentic_run_reads_the_fake_s_stream_of_events(
        fakes: FakeClis, home: pathlib.Path, tmp_path: pathlib.Path) -> None:
    fakes.script("claude", '{"proposals": []}')
    got = _child(AGENTIC, fakes=fakes, home=home, cwd=tmp_path)
    assert (got["failure"], got["result"]) == ("", '{"proposals": []}')
    assert fakes.calls("claude") == [CliCall(argv=got["argv"], stdin="")]


def test_a_raw_reply_is_printed_byte_for_byte_with_its_stderr_and_status(
        fakes: FakeClis, home: pathlib.Path) -> None:
    fakes.script("claude", CliReply(stdout="not json {\n", stderr="login expired\n",
                                    exit_code=1))
    done = _run(fakes, "claude", home, "-p", "hello", stdin="piped text")
    assert (done.stdout, done.stderr, done.returncode) == ("not json {\n",
                                                           "login expired\n", 1)
    assert fakes.calls("claude") == [CliCall(argv=["-p", "hello"], stdin="piped text")]


def test_a_run_with_no_reply_left_fails_loudly(fakes: FakeClis, home: pathlib.Path) -> None:
    fakes.script("codex", "only one")
    assert _run(fakes, "codex", home, "exec").returncode == 0
    second = _run(fakes, "codex", home, "exec")
    assert second.returncode == EXHAUSTED
    assert "no scripted reply left for run 2" in second.stderr
    assert len(fakes.calls("codex")) == 2


def test_a_name_that_is_not_one_of_the_fakes_is_refused(fakes: FakeClis) -> None:
    with pytest.raises(ValueError, match="no fake 'cursor-agent'"):
        fakes.script("cursor-agent", "a reply")


def test_windows_is_refused_with_the_reason(monkeypatch: pytest.MonkeyPatch,
                                            tmp_path: pathlib.Path) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(NotImplementedError, match="only as an .exe"):
        FakeClis(tmp_path / "bin")
```

- [ ] **Step 2: Run it and see it fail.**

Run: `TEST tests/adversarial/fakes/test_adv_fake_cli.py`
Expected: a collection error, `ModuleNotFoundError: No module named 'harness.fakes.cli'`.

- [ ] **Step 3: Write the fakes.**

`tests/harness/fakes/cli.py`:

```python
"""Fake `claude` and `codex` executables for the plugin's capture hook.

The capture hook mines a turn by starting a headless agent CLI that the user is already
signed in to (`plugin/hooks/lib/extract.py` and `lib/agentic.py`). It finds the CLI on
`PATH`, passes the prompt as the last argument, and reads what the CLI prints:

* `claude -p ... --output-format json <prompt>` prints one JSON object, and the hook reads
  its `result`, `usage` and `is_error` (`CLAUDE_CLI` in `core/host.py`).
* `claude -p ... --output-format stream-json --verbose ... <data>`, the agentic run,
  prints one JSON event per line: an `init` event that names the connected MCP servers and
  the tools, then the run, then a `result` event (`_Watch` in `lib/agentic.py`).
* `codex exec --skip-git-repo-check --json <prompt>` prints one JSON event per line, and
  the hook reads the text of the `item.completed` event whose item is an `agent_message`,
  and the `usage` of the `turn.completed` event (`CODEX_CLI` in `core/host.py`).

`FakeClis(directory)` writes a `claude` and a `codex` into `directory`. Put `path()` in a
child process's environment as `PATH`, and a hook in that process starts the fakes instead
of the real CLIs. Each run takes the next reply from its script, prints it in the format
its arguments ask for, and appends its arguments and its stdin to a log that `calls()`
reads. A run that finds no reply left says so on stderr and exits with status 3, so a
hook that starts a CLI once more than the test expected fails loudly.

POSIX only. The executables are shell scripts, and on Windows a program that another
starts without a shell is found on `PATH` only as an `.exe`.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import shlex
import stat
import sys
from typing import Any, Mapping

#: The CLIs the capture hook can start.
NAMES = ("claude", "codex")

#: The token counts a reply reports when the test names none.
USAGE: Mapping[str, int] = {"input_tokens": 10, "output_tokens": 5}

#: Exit status of a run that found no scripted reply left.
EXHAUSTED = 3


@dataclasses.dataclass(frozen=True)
class CliReply:
    """What one run of a fake CLI prints."""

    #: What the model answered. The fake wraps it in the format the arguments ask for.
    text: str = ""
    #: When set, printed exactly as given instead of any envelope, for output the hook
    #: cannot parse.
    stdout: str | None = None
    stderr: str = ""
    exit_code: int = 0
    #: For `claude`, the envelope's `is_error` flag. For `codex`, a `turn.failed` event is
    #: printed instead of the reply.
    is_error: bool = False
    usage: Mapping[str, int] = dataclasses.field(default_factory=lambda: dict(USAGE))
    #: `claude --output-format stream-json` only: the events to print instead of the
    #: default `init`, `assistant` and `result`, for an agentic run that calls tools.
    events: tuple[Mapping[str, Any], ...] | None = None


@dataclasses.dataclass(frozen=True)
class CliCall:
    """One run of a fake CLI: its arguments without the program name, and its stdin."""

    argv: list[str]
    stdin: str


class FakeClis:
    """A `claude` and a `codex` in one directory, each with its own script and log."""

    def __init__(self, directory: pathlib.Path) -> None:
        if sys.platform == "win32":
            raise NotImplementedError(
                "the fake agent CLIs are POSIX shell scripts, and on Windows a program "
                "started without a shell is found on PATH only as an .exe")
        #: The directory to put first on `PATH`.
        self.bin = pathlib.Path(directory)
        self.bin.mkdir(parents=True, exist_ok=True)
        runner = self.bin / "_fake_cli.py"
        runner.write_text(_RUNNER, encoding="utf-8")
        for name in NAMES:
            script = self.bin / name
            # `-I` runs the runner isolated from the child's PYTHON* variables and user
            # site, so what the hook's environment holds cannot change what it does.
            script.write_text(f"#!/bin/sh\nexec {shlex.quote(sys.executable)} -I "
                              f"{shlex.quote(str(runner))} {name} \"$@\"\n",
                              encoding="utf-8")
            script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            self.script(name)

    def script(self, name: str, *replies: CliReply | str) -> None:
        """Replace what `name` answers with `replies`, one per run, in order. A plain
        string is a reply with that text. With no replies, every run fails as exhausted."""
        self._check(name)
        queue = [dataclasses.asdict(r if isinstance(r, CliReply) else CliReply(text=r))
                 for r in replies]
        (self.bin / f"{name}.replies.json").write_text(json.dumps(queue), encoding="utf-8")

    def calls(self, name: str) -> list[CliCall]:
        """Every run of `name` so far, in order."""
        self._check(name)
        log = self.bin / f"{name}.calls.jsonl"
        if not log.exists():
            return []
        return [CliCall(**json.loads(line))
                for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]

    def path(self, rest: str | None = None) -> str:
        """A `PATH` value with the fakes first. `rest` follows them, and defaults to this
        process's own `PATH`, so a child still finds everything else it runs."""
        rest = os.environ.get("PATH", "") if rest is None else rest
        return os.pathsep.join(part for part in (str(self.bin), rest) if part)

    def _check(self, name: str) -> None:
        if name not in NAMES:
            raise ValueError(f"there is no fake {name!r}; the fakes are {', '.join(NAMES)}")


#: The program both executables run. Standard library only, so it runs under any Python.
_RUNNER = r'''"""A fake agent CLI for the memvara test suite, written by tests/harness/fakes/cli.py."""
import fcntl
import json
import os
import sys

name, argv = sys.argv[1], sys.argv[2:]
here = os.path.dirname(os.path.abspath(__file__))
stdin = "" if sys.stdin is None or sys.stdin.isatty() else sys.stdin.read()

# The log is read and appended under a lock, so two runs at once each get their own reply.
with open(os.path.join(here, name + ".calls.jsonl"), "a+", encoding="utf-8") as log:
    fcntl.flock(log, fcntl.LOCK_EX)
    log.seek(0)
    number = sum(1 for line in log if line.strip())
    log.write(json.dumps({"argv": argv, "stdin": stdin}) + "\n")
    log.flush()

with open(os.path.join(here, name + ".replies.json"), encoding="utf-8") as script:
    replies = json.load(script)
if number >= len(replies):
    sys.stderr.write(f"fake {name}: no scripted reply left for run {number + 1}\n")
    sys.exit(3)
reply = replies[number]


def emit(event):
    sys.stdout.write(json.dumps(event) + "\n")


def option(flag):
    return argv[argv.index(flag) + 1] if flag in argv[:-1] else None


if reply["stdout"] is not None:
    sys.stdout.write(reply["stdout"])
elif name == "codex":
    emit({"type": "thread.started", "thread_id": "fake-thread"})
    emit({"type": "turn.started"})
    if reply["is_error"]:
        emit({"type": "turn.failed", "error": {"message": reply["text"]}})
    else:
        emit({"type": "item.completed",
              "item": {"id": "item_0", "type": "agent_message", "text": reply["text"]}})
        emit({"type": "turn.completed", "usage": reply["usage"]})
elif option("--output-format") == "stream-json":
    events = reply["events"]
    if events is None:
        events = [
            {"type": "system", "subtype": "init", "session_id": "fake-session",
             "mcp_servers": [{"name": "memvara", "status": "connected"}],
             "tools": ["mcp__memvara__memory_search", "mcp__memvara__memory_recall",
                       "mcp__memvara__memory_why", "mcp__memvara__memory_profile"]},
            {"type": "assistant",
             "message": {"id": "msg_fake", "role": "assistant", "usage": reply["usage"],
                         "content": [{"type": "text", "text": reply["text"]}]}},
            {"type": "result",
             "subtype": "error_during_execution" if reply["is_error"] else "success",
             "is_error": reply["is_error"], "result": reply["text"],
             "usage": reply["usage"]},
        ]
    for event in events:
        emit(event)
else:
    emit({"type": "result",
          "subtype": "error_during_execution" if reply["is_error"] else "success",
          "is_error": reply["is_error"], "result": reply["text"], "usage": reply["usage"],
          "session_id": "fake-session"})
sys.stdout.flush()
sys.stderr.write(reply["stderr"])
sys.exit(reply["exit_code"])
'''

__all__ = ["CliCall", "CliReply", "EXHAUSTED", "FakeClis", "NAMES", "USAGE"]
```

- [ ] **Step 4: Add the skip rule** to `RULES` in `tests/harness/skips.py`, after the `SIGSTOP` rule:

```python
    SkipRule(r"^the fake agent CLIs are POSIX shell scripts$",
             "On Windows a program started without a shell is found on PATH only as an "
             ".exe, and the fake claude and codex are shell scripts. Linux and macOS run "
             "these tests.", platforms=("win32",)),
```

- [ ] **Step 5: Run the test and see it pass.**

Run: `TEST tests/adversarial/fakes/test_adv_fake_cli.py tests/adversarial/test_adv_skips.py`
Expected: every test passes; `test_adv_fake_cli.py` alone is `7 passed`.

- [ ] **Step 6: Document it:**

```markdown
**`FakeClis(directory)` writes fake `claude` and `codex` executables** for the capture hook, which mines a turn by starting one of them. Give a child process `PATH` set to `fakes.path()`, and it starts the fakes instead of the real CLIs. `fakes.script("claude", "reply text", CliReply(...))` sets what each run prints, one reply per run, and `fakes.calls("claude")` lists each run's arguments and stdin.

- Each fake prints its reply in the format its arguments ask for: the single JSON object of `claude --output-format json`, the event stream of `claude --output-format stream-json` that the agentic capture run reads, or the event stream of `codex exec --json`. `CliReply(stdout=...)` prints exactly what it is given instead, for output the hook cannot parse.
- A run that finds no reply left says so on stderr and exits with status 3.
- `HookRunner` still refuses `run("capture")`. The hook-conformance tests lift that refusal and use these fakes.
- The fakes are POSIX shell scripts. On Windows a program that another starts without a shell is found on `PATH` only as an `.exe`, so their tests skip there.
```

- [ ] **Step 7: Type-check and commit.**

```bash
$PY -m mypy tests/harness
$PY -m mypy tests/harness --ignore-missing-imports
git add tests/harness/fakes/cli.py tests/adversarial/fakes/test_adv_fake_cli.py \
    tests/harness/skips.py docs/claude/testing.md
git commit -m "Add fake claude and codex executables for the capture hook"
```

---

## Task 6: Verification (no commit unless something needs fixing)

- [ ] **Step 1: Twenty runs in a row of every new test file.**

```bash
for f in tests/adversarial/fakes/test_adv_*.py; do
  for i in $(seq 20); do TEST "$f" > /private/tmp/claude-501/f3-tmp/last.txt 2>&1 \
    || { echo "FLAKE $f run $i"; tail -20 /private/tmp/claude-501/f3-tmp/last.txt; }; done
  echo "$f: 20 runs done"; done
```

Expected: no line starting with `FLAKE`.

- [ ] **Step 2: The full gate, as two commands, with a private coverage file.**

```bash
COVERAGE_FILE=$W/local/cov/.coverage.fakes $PY -m coverage run -m pytest -q -p no:cacheprovider
COVERAGE_FILE=$W/local/cov/.coverage.fakes $PY -m coverage report | tail -3
```

Expected: the last line of the first command reports no failures, and the coverage total is `100%`.

- [ ] **Step 3: Both type checks.**

```bash
$PY -m mypy -p memvara
$PY -m mypy tests/harness
$PY -m mypy tests/harness --ignore-missing-imports
```

Expected: `Success: no issues found` three times.

---

## Self-review against the brief

- `FakeV1` on the routes of `memvara/remote/`, served by a local `Memvara`, as a mock transport and on 127.0.0.1, with a status, a delay and a hang per route: Tasks 1 and 2.
- `FakeHostedMcp`, enough for the hooks' hosted client and the npm bridge, including their authentication headers: Task 3.
- `FakeOpenAI` with ordered scripted replies, a request log, malformed output, a 429 and a hang, and memvara's LLM client talking to it: Task 4.
- Fake `claude` and `codex` that a test puts first on `PATH`, recording argv and stdin, printing a scripted reply in the format the hook parses: Task 5. `HookRunner`'s refusal of `capture` is left as it is.
- The self-tests: parity for `remember`, `get`, `get_all`, `search`, `forget`, `delete` and `erase` (Task 2); each fault reaching the client as the client handles it (Tasks 1 and 2); `FakeOpenAI` in order, recorded, and reachable by `OpenAILLM` (Task 4); a fake CLI found on `PATH`, recording argv, printing its reply exactly (Task 5); everything offline and in the fast tier (every task).

## Changes made while executing this plan

The committed files are the reference. They differ from the code above in these places, each for the reason given:

- **`_http.py`, `serve()`** runs `serve_forever` with a poll interval of 0.05 seconds instead of the default half second. `close()` waits for the loop to notice the shutdown, so the default cost every test that serves a fake up to half a second.
- **`_http.py`, the fault helpers** are named `_queue_fault` and `_check_route` instead of `_add` and `_check`. `FakeV1`'s handler for `POST /v1/memories` is also named `_add`, and it overrode the helper, so `fail()` and `delay()` called the route handler. The first run of Task 2's tests showed it.
- **`fake_v1.py`, `_end`** tests its two addressing modes as two explicit branches, so the type checker can see which variable each one uses. The answer it gives is unchanged.
- **`hosted_mcp.py`, `_post`** calls the presented session header `presented`, because reusing the name `session` gave one variable two types.
- **`openai_compat.py`**: the class docstring no longer says the fake is reached over a socket only, because the inherited mock transports reach it too.
- **`cli.py`**: the module docstring says that a run reads its stdin to the end, so a caller that leaves stdin open holds the run.
- **`test_adv_fake_v1_faults.py`**: the retried-write test also checks that the claim was observed once. A write that lands twice leaves one claim observed twice, so counting claims alone could not catch it.
- **`test_adv_fake_http.py`**: the async test compares the first tick with the moment the delayed answer arrived, instead of with a fixed 0.25 seconds that a loaded machine could miss.
- **`test_adv_fake_v1_parity.py`**, from the final review: the route scan stops on a client call whose method it cannot read, instead of skipping it, and `test_a_call_the_scan_cannot_read_stops_it` shows that on a synthetic client. A route called that way would otherwise have slipped past the check.
- **`docs/claude/testing.md`**, from the final review: the fakes section says that the hooks use a hosted endpoint only when no local store is configured, which is when a hook run reaches `FakeHostedMcp`.
