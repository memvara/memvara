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

`close()` releases every request that is still hanging or waiting, however it arrived, so
a test that ends with a request held never waits out the delay or the client's timeout. A
released request is not carried out. Over a socket its connection is closed without an
answer; over a mock transport it raises `httpx.ReadTimeout`.
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
    #: request hung, the client gave up, or the fake closed before the answer.
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
        self._check_route(route)
        if body is None:
            body = self.error_body(status, route)
        if isinstance(body, (str, bytes)):
            raw = body.encode("utf-8") if isinstance(body, str) else body
            reply = Reply(status, raw, dict(headers or {}))
        else:
            reply = json_reply(status, body, headers)
        self._queue_fault(route, _Fault("fail", times, reply=reply))

    def delay(self, route: str, seconds: float, *, times: int | None = None) -> None:
        """Wait `seconds` before answering requests to `route`."""
        self._check_route(route)
        self._queue_fault(route, _Fault("delay", times, seconds=seconds))

    def hang(self, route: str, *, times: int | None = None) -> None:
        """Never answer requests to `route`. The fake does not act on them either: each
        one is held until the client gives up or the fake closes."""
        self._check_route(route)
        self._queue_fault(route, _Fault("hang", times))

    def clear_faults(self) -> None:
        """Remove every fault that has not been used up."""
        with self._lock:
            self._faults.clear()

    def _check_route(self, route: str) -> None:
        # A typo in a route name would inject a fault that never fires, and the test would
        # pass without testing anything.
        if route not in self.ROUTES:
            raise ValueError(f"{type(self).__name__} has no route {route!r}. Its routes "
                             f"are: {', '.join(self.ROUTES)}")

    def _queue_fault(self, route: str, fault: _Fault) -> None:
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
                # Waiting on the event rather than sleeping lets `close()` end the wait.
                self._closed.wait(limit)
                raise httpx.ReadTimeout("the fake did not answer in time", request=outgoing)
            if step.wait and self._closed.wait(step.wait):
                raise httpx.ReadTimeout("the fake closed before it answered",
                                        request=outgoing)
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
                await self._wait_for_close(limit)
                raise httpx.ReadTimeout("the fake did not answer in time", request=outgoing)
            if step.wait and await self._wait_for_close(step.wait):
                raise httpx.ReadTimeout("the fake closed before it answered",
                                        request=outgoing)
            return _to_httpx(self._answer(request, step))

        return httpx.MockTransport(handle)

    async def _wait_for_close(self, seconds: float | None) -> bool:
        """Wait `seconds`, or until the fake closes, without blocking the event loop.
        None waits only for the close. True when the fake closed first."""
        deadline = None if seconds is None else time.monotonic() + seconds
        while not self._closed.is_set():
            left = None if deadline is None else deadline - time.monotonic()
            if left is not None and left <= 0:
                return False
            # The event is a thread's event, so the loop cannot await it; it checks it
            # every twentieth of a second instead.
            await asyncio.sleep(0.05 if left is None else min(0.05, left))
        return True

    def serve(self) -> str:
        """Answer on 127.0.0.1 from a background thread, and return the base URL.

        Calling it again returns the same URL. `close()` stops it.
        """
        with self._lock:
            if self._server is None:
                self._server = _Server(self)
                # A short poll interval, because `close()` waits for the loop to notice
                # the shutdown, and the default half second would be paid by every test
                # that serves a fake.
                self._thread = threading.Thread(
                    target=self._server.serve_forever, kwargs={"poll_interval": 0.05},
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
