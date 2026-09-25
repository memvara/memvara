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
        presented = request.header(SESSION_HEADER)
        if presented is None:
            return json_reply(400, _rpc_error(
                message.get("id"), -32600,
                f"every request after 'initialize' needs a {SESSION_HEADER!r} header."))
        with self._lock:
            known = presented in self._live
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
