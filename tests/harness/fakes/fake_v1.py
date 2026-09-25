"""A fake of the hosted `/v1` REST API, answered by a real local `Memvara`.

`RemoteMemvara` and `AsyncRemoteMemvara` (`memvara/remote/`) turn each method call into one
`/v1` request, and hydrate the JSON that comes back into the library's own dataclasses
(`memvara/remote/hydrate.py`). `FakeV1` answers those requests from an in-process
`Memvara` with the hashing embedder and no model. It renders each answer the way
memvara-cloud's `rest/render.py` does, because `hydrate.py` is written as the inverse of
that module, and it refuses a request the way the cloud's routes refuse it. A test can
therefore point the real client at it and compare the answers with a local store's.

**How current it is.** The renderers and the refusals were written to match memvara-cloud
at origin/main e8940be (2026-09-25). Nothing checks that automatically, because this suite
never reads memvara-cloud: when the cloud changes, this fake keeps the old behaviour until
somebody compares the two again by hand. Only the list of routes is checked, against the
clients, as the next paragraph says.

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

One header differs as well, and not on purpose. An answer in a project carries
`Memvara-Project-Applied`, but a refusal never does. The cloud also puts the header on a
refusal raised after it has resolved the request's project (`rest/errors.py`,
`_stamped`), such as a 404 for a missing memory, though not on a request that fails
schema validation. No client in this repository reads the header, so nothing depends on
the difference yet.
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
        self._key_locks: dict[tuple[str, str, str], threading.Lock] = {}
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
            if isinstance(body, Reply):
                # A handler that sets its own status and headers still names the project.
                # The cloud sets the header on the response every route shares
                # (`deps._context`), so its consolidation's 202 carries it too.
                return Reply(body.status, body.body, {**body.headers, **headers})
            return json_reply(200, body, headers)

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

        One lock for each key, method and path, the same triple the reply is stored under.
        A retry that arrives while the first attempt is still being answered waits for it
        and then gets its reply, rather than writing a second time, and a write that
        shares only the key does not wait at all.
        """
        with self._lock:
            lock = self._key_locks.setdefault(key, threading.Lock())
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
        valid_at, known_at = _query_axes(request)
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
        valid_at, known_at = _query_axes(request)
        found = view.why(request.params["id"], valid_at=valid_at, known_at=known_at)
        if found is None:
            raise _no_memory(request.params["id"])
        return _provenance(found)

    def _history(self, view: ScopedMemvara, request: Request) -> Any:
        predicate = _required(request, "predicate")
        subject = request.param("subject") or "user"
        valid_at, known_at = _query_axes(request)
        claims = view.history(subject, predicate, valid_at=valid_at, known_at=known_at)
        return {"subject": subject,
                "predicate": self.memvara.registry.normalize(predicate),
                "scope": _scope(view.scope), "as_of": request.param("as_of"),
                "valid_at": _instant(valid_at), "known_at": _instant(known_at),
                "count": len(claims), "timeline": [_memory(c) for c in claims]}

    def _ask(self, view: ScopedMemvara, request: Request) -> Any:
        body = _body(request, _ASK, required=("question",))
        _non_empty_text(body, "question")
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
        valid_at, known_at = _query_axes(request)
        claims = view.produced(request.params["id"], valid_at=valid_at, known_at=known_at)
        return {"episode_id": request.params["id"], "as_of": request.param("as_of"),
                "valid_at": _instant(valid_at), "known_at": _instant(known_at),
                "count": len(claims), "memories": [_memory(c) for c in claims]}

    def _neighborhood(self, view: ScopedMemvara, request: Request) -> Any:
        valid_at, known_at = _query_axes(request)
        found = view.neighborhood(
            _required(request, "entity"), depth=_int(request, "depth", 2, low=1, high=4),
            k=_int(request, "k", 10, low=1, high=50),
            min_hops=_int(request, "min_hops", 1, low=1, high=4),
            predicates=request.query.get("predicates"),
            min_score=_float(request, "min_score", 0.0), valid_at=valid_at,
            known_at=known_at)
        return _paths_body(request, valid_at, known_at, found)

    def _paths(self, view: ScopedMemvara, request: Request) -> Any:
        valid_at, known_at = _query_axes(request)
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
        at, reason = _instant_in(body.get("at"), "at"), body.get("reason")
        subject = body.get("subject") or "user"
        claims: list[Claim] = []
        if memory_id is not None and predicate is None:
            found = view.get(memory_id)
            if found is not None and view.delete(memory_id, at=at, close="ended",
                                                 reason=reason):
                claims = self._reread([found])
        elif predicate is not None and memory_id is None:
            claims = self._reread(view.forget(subject, predicate, at=at, close="ended",
                                              reason=reason))
        else:
            raise ApiError(422, "invalid_request",
                           "exactly one of memory_id and predicate is required")
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
    _non_empty_text(body, "query")
    return body


def _non_empty_text(body: dict[str, Any], name: str) -> None:
    """Refuse the body unless `body[name]` is a non-empty string, as the cloud's request
    models do for a `str` field declared with `Field(min_length=1)`."""
    if not isinstance(body[name], str) or not body[name]:
        raise ApiError(422, "invalid_request", f"{name} must be a non-empty string")


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


def _query_axes(request: Request) -> tuple[datetime | None, datetime | None]:
    """`_axes` for a read that takes the clocks as query parameters."""
    return _axes(request.param("as_of"), request.param("valid_at"),
                 request.param("known_at"))


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
