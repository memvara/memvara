"""`AsyncRemoteMemvara`: `RemoteMemvara`, awaited, on a real async transport.

**This does not follow `memvara/aio.py`'s pattern, and the reason is that module's own
argument.** `memvara.aio` wraps every synchronous call in `asyncio.to_thread` because
there is no async SQLite and coroutine-colouring would have to propagate from `Memvara`
down through `Store`, `WritePipeline`, `Reconciler`, `HybridRetriever` and `Consolidator`
— see that module's docstring, which spends a full paragraph on why a thread is the
right tool there. Neither half of that argument holds here. `httpx` ships a real
`AsyncClient` that speaks the same protocol without blocking a thread to do it, and there
is no engine underneath this class to colour — `RemoteMemvara` already does nothing but
turn a method call into one `/v1` request. Wrapping `httpx.Client` in `asyncio.to_thread`
here would spend a thread-pool slot to get worse behaviour than the async client this
class uses directly already provides for free.

This class is otherwise `RemoteMemvara`'s twin: same methods, same arguments, same
hydrated return values, same absences (`reembed`, `pending_extraction`, `reextract`,
`reset`, `recall(budget=...)`) for the same reasons — see `memvara.remote.api`'s module
docstring, which this one does not repeat. The only difference method-by-method is
`async def` and an `await` in front of the one call to the transport.
"""
from __future__ import annotations

from contextlib import nullcontext
from copy import copy
from datetime import datetime
from typing import Any, Collection, Literal, Mapping, Sequence, overload

from ..confirm import ConfirmationRefused
from ..core import _check_k
from ..filters import FilterValue
from ..redact import CLAIM_OBJECT, CLAIM_SUBJECT, CLAIM_TEXT, EPISODE, Redactor
from ..retrieve import Path, Retrieved
from ..types import (
    Answer, Claim, DeleteResult, Delta, Document, DocumentStatus, Episode,
    ForgetPreview, ForgetResult, Link, MemoryType, Page, Profile, Provenance, Result,
    Scope, SearchResults, WriteReceipt, closure, closure_reason, link_relation,
    one_source, refuse_self_link,
)
from . import hydrate
from .api import (PROJECT_HEADER, _as_local_refusal, _document_body,
                  _document_path, _expire_reason, _filter_fields, _hit,
                  _iso, _refuse_project_meta, _sent, _states, _type, _types)
from .client import DEFAULT_TIMEOUT, AsyncHttpClient
from .creds import resolve
from .errors import Conflict, InvalidRequest, NotFound, refuse_project_purge


class AsyncRemoteMemvara:
    """`RemoteMemvara` against a hosted deployment, on `httpx.AsyncClient`.

    Constructing one performs no network call, for the same reason `RemoteMemvara`'s
    constructor does not: resolving a credential and building a connection pool is not
    a side effect a library performs on your behalf. The first request is the first
    `await`ed method call.
    """

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None,
                 tenant: str = "default", user: str | None = None,
                 agent: str | None = None, session: str | None = None,
                 project: str | None = None,
                 timeout: float = DEFAULT_TIMEOUT,
                 redactor: Redactor | None = None,
                 metadata_filters: bool = True) -> None:
        key, url = resolve(api_key, base_url)
        #: The `metadata_filters` switch, as on `Memvara`: off, a read that passes
        #: `filters` or `filepath_prefix` is refused before anything is sent.
        self.metadata_filters = metadata_filters
        self._http = AsyncHttpClient(key, url, timeout=timeout)
        #: See `RemoteMemvara.default_scope`. The tenant is held, never sent; the project
        #: is sent as the `Memvara-Project` header.
        self.default_scope = Scope(tenant, user, agent, session, project=project)
        #: See `RemoteMemvara.redactor`.
        self.redactor = redactor

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "AsyncRemoteMemvara":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def __repr__(self) -> str:
        return f"<AsyncRemoteMemvara {self.default_scope.key()}>"

    def scope(self, *, user: str | None = None, agent: str | None = None,
              session: str | None = None,
              project: str | None = None) -> "AsyncScopedRemoteMemvara":
        """See `RemoteMemvara.scope`."""
        current = self.default_scope
        narrowed = Scope(
            current.tenant,
            user if user is not None else current.user,
            agent if agent is not None else current.agent,
            session if session is not None else current.session,
            project=project if project is not None else current.project,
        )
        return AsyncScopedRemoteMemvara(self, narrowed)

    def _at(self, scope: Scope) -> "AsyncRemoteMemvara":
        """See `RemoteMemvara._at`. No `await` in here: copying an attribute does not
        touch the transport, so there is nothing to make async."""
        twin = copy(self)
        twin.default_scope = scope
        return twin

    async def _request(self, method: str, path: str, **kw: Any) -> Any:
        """See `RemoteMemvara._request`."""
        project = self.default_scope.project
        if project is not None:
            kw["headers"] = {PROJECT_HEADER: project}
        return await self._http.request(method, path, **kw)

    async def _read(self, path: str, body: dict[str, Any]) -> Any:
        """POST one read, retrying once without `query_rewrite` for an older deployment.

        `query_rewrite` is sent only as `false`, when the caller opted out. A deployment
        from before the field refuses it as unknown (422), and such a deployment never
        rewrites a query, so the opt-out already holds there: the read is sent again
        without the field. A 422 for any other reason fails the second time as well and
        is raised. `synthesize` gets no retry, because an older deployment cannot write
        the summary the caller asked for, and saying so is the honest answer.
        """
        try:
            return await self._request("POST", path, params=self._params(), json=body)
        except InvalidRequest:
            if body.get("query_rewrite") is not False:
                raise
            body = {k: v for k, v in body.items() if k != "query_rewrite"}
            return await self._request("POST", path, params=self._params(), json=body)

    def _params(self, **extra: Any) -> dict[str, Any]:
        scope = self.default_scope
        return {"user": scope.user, "agent": scope.agent, "session": scope.session,
                **extra}

    def _redact(self, text: str | None, field: str) -> str | None:
        if self.redactor is None or text is None:
            return text
        return self.redactor.redact(text, field=field, scope=self.default_scope)

    def _turn(self, message: Episode | Mapping[str, Any] | str) -> dict[str, Any]:
        if isinstance(message, str):
            return _sent({"content": self._redact(message, EPISODE)})
        if isinstance(message, Episode):
            return _sent({"role": message.role,
                          "content": self._redact(message.content, EPISODE),
                          "ts": _iso(message.ts), "metadata": dict(message.meta)})
        known = {"role", "content", "ts", "metadata"}
        meta = dict(message.get("metadata") or {})
        meta.update({k: v for k, v in message.items() if k not in known})
        return _sent({"role": message.get("role"),
                      "content": self._redact(message["content"], EPISODE),
                      "ts": _iso(message.get("ts")), "metadata": meta})

    def _cite(self, sources: Sequence[Episode | Mapping[str, Any] | str] | None,
              ) -> tuple[list[str], list[dict[str, Any]]]:
        ids = [s for s in sources or [] if isinstance(s, str)]
        turns = [self._turn(s) for s in sources or [] if not isinstance(s, str)]
        return ids, turns

    async def _end(self, body: dict[str, Any]) -> list[Claim]:
        out = await self._request("POST", "/v1/end", params=self._params(),
                                  json=_sent(body), write=True)
        return [hydrate.claim(c) for c in out["ended"]]

    # -- service -------------------------------------------------------------

    async def health(self) -> dict[str, Any]:
        return await self._request("GET", "/v1/health")

    async def whoami(self) -> dict[str, Any]:
        return await self._request("GET", "/v1/whoami", params=self._params())

    async def stats(self) -> dict[str, int]:
        body = await self._request("GET", "/v1/stats", params=self._params())
        return dict(body["tenant_counts"])

    async def service(self, *, attempts: int | None = None,
                      timeout: float | None = None) -> dict[str, Any]:
        """The whole `/v1/stats` envelope. See `RemoteMemvara.service`."""
        return dict(await self._request("GET", "/v1/stats", params=self._params(),
                                        attempts=attempts, timeout=timeout))

    async def connectivity(self) -> dict[str, int]:
        body = await self.stats()
        if "joinable_claims" not in body or "live_claims" not in body:
            return {}
        return {"live_claims": body["live_claims"],
                "joinable_claims": body["joinable_claims"]}

    # -- reading -------------------------------------------------------------

    # The same three variants as `RemoteMemvara.search`, and they carry the same weight:
    # they are what makes "calling code cannot tell which it holds" true of the *type* as
    # well as of the value. Without them `await mem.search(q)` types as `list[Retrieved]`
    # here and `list[Result]` on `AsyncMemvara`, so the same expression reading `.claim`
    # off a row checks against one engine and not the other -- and this class's own
    # docstring promises it is `RemoteMemvara`'s twin down to the arguments.
    @overload
    async def search(self, query: str, *, k: int = ..., min_score: float = ...,
                     anchored: bool = ..., ranked: bool = ...,
                     query_rewrite: bool = ...,
                     as_of: datetime | None = ..., valid_at: datetime | None = ...,
                     known_at: datetime | None = ...,
                     states: Collection[str] | None = ...,
                     include_invalidated: bool | None = ...,
                     memory_types: Sequence[MemoryType | str] | None = ...,
                     filters: Mapping[str, FilterValue] | None = ...,
                     filepath_prefix: str | None = ...,
                     include_episodes: Literal[False] = ...) -> list[Result]: ...

    @overload
    async def search(self, query: str, *, k: int = ..., min_score: float = ...,
                     anchored: bool = ..., ranked: bool = ...,
                     query_rewrite: bool = ...,
                     as_of: datetime | None = ..., valid_at: datetime | None = ...,
                     known_at: datetime | None = ...,
                     states: Collection[str] | None = ...,
                     include_invalidated: bool | None = ...,
                     memory_types: Sequence[MemoryType | str] | None = ...,
                     filters: Mapping[str, FilterValue] | None = ...,
                     filepath_prefix: str | None = ...,
                     include_episodes: Literal[True]) -> list[Retrieved]: ...

    @overload
    async def search(self, query: str, *, k: int = ..., min_score: float = ...,
                     anchored: bool = ..., ranked: bool = ...,
                     query_rewrite: bool = ...,
                     as_of: datetime | None = ..., valid_at: datetime | None = ...,
                     known_at: datetime | None = ...,
                     states: Collection[str] | None = ...,
                     include_invalidated: bool | None = ...,
                     memory_types: Sequence[MemoryType | str] | None = ...,
                     filters: Mapping[str, FilterValue] | None = ...,
                     filepath_prefix: str | None = ...,
                     include_episodes: bool) -> list[Retrieved]: ...

    async def search(self, query: str, *, k: int = 10, min_score: float = 0.0,
                     anchored: bool = False, ranked: bool = False,
                     query_rewrite: bool = True,
                     as_of: datetime | None = None, valid_at: datetime | None = None,
                     known_at: datetime | None = None,
                     states: Collection[str] | None = None,
                     include_invalidated: bool | None = None,
                     memory_types: Sequence[MemoryType | str] | None = None,
                     filters: Mapping[str, FilterValue] | None = None,
                     filepath_prefix: str | None = None,
                     include_episodes: bool = False) -> list[Any]:
        body = await self._read(
            "/v1/search",
            body=_sent({"query": query, "k": k, "min_score": min_score,
                        "anchored": anchored or None, "ranked": ranked or None,
                        "query_rewrite": None if query_rewrite else False,
                        "as_of": _iso(as_of), "valid_at": _iso(valid_at),
                        "known_at": _iso(known_at), "states": _states(states),
                        "include_invalidated": include_invalidated,
                        "memory_types": _types(memory_types),
                        **_filter_fields(filters, filepath_prefix, self.metadata_filters),
                        "include_episodes": include_episodes}))
        return SearchResults([_hit(h) for h in body["results"]],
                             selection=hydrate.selection(body.get("selection")),
                             rewrite=hydrate.rewrite(body.get("rewrite")))

    async def recall(self, query: str, *, k: int = 8, min_score: float = 0.0,
                     anchored: bool = False, ranked: bool = False,
                     query_rewrite: bool = True, synthesize: bool = False,
                     memory_types: Sequence[MemoryType | str] | None = None,
                     include_episodes: bool = False,
                     budget: int | None = None,
                     valid_at: datetime | None = None,
                     filters: Mapping[str, FilterValue] | None = None,
                     filepath_prefix: str | None = None) -> str:
        if budget is not None:
            raise ValueError(
                "recall(budget=...) is not available against a hosted deployment: "
                "POST /v1/recall renders the block server-side and takes no budget. Use "
                "a smaller k, or render your own block from search().")
        if valid_at is not None:
            raise ValueError(
                "recall(valid_at=...) is not available against a hosted deployment: "
                "POST /v1/recall has no time axis. Use search(valid_at=...) and render "
                "your own block.")
        body = await self._read(
            "/v1/recall",
            body=_sent({"query": query, "k": k, "min_score": min_score,
                        "anchored": anchored or None, "ranked": ranked or None,
                        "query_rewrite": None if query_rewrite else False,
                        "synthesize": synthesize or None,
                        "memory_types": _types(memory_types),
                        **_filter_fields(filters, filepath_prefix, self.metadata_filters),
                        "include_episodes": include_episodes}))
        return str(body["text"])

    async def get(self, claim_id: str) -> Claim | None:
        try:
            body = await self._request("GET", f"/v1/memories/{claim_id}",
                                       params=self._params())
        except NotFound:
            return None
        return hydrate.claim(body)

    async def get_all(self, *, states: Collection[str] | None = None,
                      include_invalidated: bool | None = None,
                      limit: int = 100, offset: int = 0,
                      as_of: datetime | None = None, valid_at: datetime | None = None,
                      known_at: datetime | None = None) -> list[Claim]:
        body = await self._request(
            "GET", "/v1/memories",
            params=self._params(limit=limit, offset=offset, states=_states(states),
                                include_invalidated=include_invalidated,
                                as_of=_iso(as_of), valid_at=_iso(valid_at),
                                known_at=_iso(known_at)))
        return [hydrate.claim(c) for c in body["memories"]]

    async def count(self, *, states: Collection[str] | None = None,
                    include_invalidated: bool | None = None,
                    as_of: datetime | None = None, valid_at: datetime | None = None,
                    known_at: datetime | None = None) -> int:
        body = await self._request(
            "GET", "/v1/memories",
            params=self._params(limit=1, states=_states(states),
                                include_invalidated=include_invalidated,
                                as_of=_iso(as_of), valid_at=_iso(valid_at),
                                known_at=_iso(known_at)))
        return int(body["total"])

    async def history(self, subject: str, predicate: str, *,
                      as_of: datetime | None = None, valid_at: datetime | None = None,
                      known_at: datetime | None = None) -> list[Claim]:
        body = await self._request(
            "GET", "/v1/history",
            params=self._params(subject=subject, predicate=predicate, as_of=_iso(as_of),
                                valid_at=_iso(valid_at), known_at=_iso(known_at)))
        return [hydrate.claim(c) for c in body["timeline"]]

    async def why(self, claim_id: str, *, as_of: datetime | None = None,
                  valid_at: datetime | None = None,
                  known_at: datetime | None = None) -> Provenance | None:
        try:
            body = await self._request(
                "GET", f"/v1/memories/{claim_id}/why",
                params=self._params(as_of=_iso(as_of), valid_at=_iso(valid_at),
                                    known_at=_iso(known_at)))
        except NotFound:
            return None
        return hydrate.provenance(body)

    async def ask(self, question: str, *, at: datetime | None = None, k: int = 3,
                 min_score: float = 0.0, anchored: bool = False) -> Answer:
        body = await self._request(
            "POST", "/v1/ask", params=self._params(),
            json=_sent({"question": question, "at": _iso(at), "k": k,
                        "min_score": min_score, "anchored": anchored or None}))
        return hydrate.answer(body)

    async def since(self, when: datetime) -> Delta:
        body = await self._request("GET", "/v1/since",
                                   params=self._params(since=_iso(when)))
        return hydrate.delta(body)

    async def produced(self, episode_id: str, *, as_of: datetime | None = None,
                       valid_at: datetime | None = None,
                       known_at: datetime | None = None) -> list[Claim]:
        body = await self._request(
            "GET", f"/v1/episodes/{episode_id}/produced",
            params=self._params(as_of=_iso(as_of), valid_at=_iso(valid_at),
                                known_at=_iso(known_at)))
        return [hydrate.claim(c) for c in body["memories"]]

    async def neighborhood(self, entity: str, *, depth: int = 2, k: int = 10,
                           min_hops: int = 1, predicates: Sequence[str] | None = None,
                           as_of: datetime | None = None,
                           valid_at: datetime | None = None,
                           known_at: datetime | None = None,
                           min_score: float = 0.0) -> list[Path]:
        body = await self._request(
            "GET", "/v1/neighborhood",
            params=self._params(entity=entity, depth=depth, k=k, min_hops=min_hops,
                                predicates=list(predicates) if predicates else None,
                                min_score=min_score, as_of=_iso(as_of),
                                valid_at=_iso(valid_at), known_at=_iso(known_at)))
        return [hydrate.path(p) for p in body["paths"]]

    async def paths_between(self, source: str, target: str, *, depth: int = 3,
                            k: int = 3, predicates: Sequence[str] | None = None,
                            as_of: datetime | None = None,
                            valid_at: datetime | None = None,
                            known_at: datetime | None = None,
                            min_score: float = 0.0) -> list[Path]:
        body = await self._request(
            "GET", "/v1/paths",
            params=self._params(source=source, target=target, depth=depth, k=k,
                                predicates=list(predicates) if predicates else None,
                                min_score=min_score, as_of=_iso(as_of),
                                valid_at=_iso(valid_at), known_at=_iso(known_at)))
        return [hydrate.path(p) for p in body["paths"]]

    async def standing(self, *, k: int | None = None) -> list[Claim]:
        if k is not None:
            _check_k(k)
        body = await self._request("GET", "/v1/standing",
                                   params=self._params(limit=k))
        return [hydrate.claim(c) for c in body["memories"]]

    async def profile(self, query: str | None = None, *, k: int = 8,
                      since: datetime | None = None,
                      buckets: Mapping[str, Sequence[str]] | None = None) -> Profile:
        """See `RemoteMemvara.profile`, which documents the request and the reply."""
        _check_k(k)
        body = await self._request(
            "POST", "/v1/profile", params=self._params(),
            json=_sent({"query": query, "k": k, "since": _iso(since),
                        "buckets": ({name: list(predicates)
                                     for name, predicates in buckets.items()}
                                    if buckets is not None else None)}))
        return hydrate.profile(body)

    # -- writing -------------------------------------------------------------

    async def add(self, messages: Any, *, role: str = "user",
                  ts: datetime | None = None) -> WriteReceipt:
        payload: Any
        if isinstance(messages, str):
            payload = self._redact(messages, EPISODE)
        elif isinstance(messages, (Episode, Mapping)):
            payload = [self._turn(messages)]
        else:
            payload = [self._turn(m) for m in messages]
        body = await self._request(
            "POST", "/v1/memories", params=self._params(),
            json=_sent({"messages": payload, "role": role, "ts": _iso(ts)}),
            write=True)
        return hydrate.receipt(body)

    async def remember(self, subject: str, predicate: str, obj: str, *,
                       confidence: float = 1.0,
                       memory_type: MemoryType | str | None = None, polarity: int = 1,
                       valid_from: datetime | None = None,
                       valid_to: datetime | None = None,
                       recorded_at: datetime | None = None,
                       sources: Sequence[Episode | Mapping[str, Any] | str] | None = None,
                       text: str | None = None, extractor: str = "api",
                       until_reason: str | None = None, replaces: str | None = None,
                       reason: str | None = None,
                       expires_at: datetime | None = None,
                       expire_reason: str | None = None,
                       **meta: Any) -> WriteReceipt:
        _refuse_project_meta(meta, "remember()")
        ids, turns = self._cite(sources)
        body = {
            "subject": self._redact(subject, CLAIM_SUBJECT),
            "predicate": predicate,
            "object": self._redact(obj, CLAIM_OBJECT),
            "text": self._redact(text, CLAIM_TEXT),
            "confidence": confidence, "polarity": polarity, "extractor": extractor,
            "memory_type": _type(memory_type),
            "valid_from": _iso(valid_from), "valid_to": _iso(valid_to),
            "recorded_at": _iso(recorded_at),
            "source_ids": ids, "sources": turns, "metadata": meta,
            "until_reason": closure_reason(until_reason),
            "replaces": replaces, "reason": closure_reason(reason),
            "expires_at": _iso(expires_at),
            "expire_reason": _expire_reason(expire_reason, expires_at),
        }
        with _as_local_refusal(replaces) if replaces is not None else nullcontext():
            out = await self._request(
                "POST", "/v1/facts", params=self._params(), json=_sent(body), write=True)
        return hydrate.receipt(out)

    async def supersede(self, old_claim_id: str, subject: str, predicate: str, obj: str,
                        *, at: datetime | None = None, close: str = "ended",
                        confidence: float = 1.0,
                        memory_type: MemoryType | str | None = None, polarity: int = 1,
                        valid_from: datetime | None = None,
                        valid_to: datetime | None = None,
                        recorded_at: datetime | None = None,
                        sources: Sequence[Episode | Mapping[str, Any] | str] | None = None,
                        text: str | None = None, extractor: str = "api",
                        reason: str | None = None,
                        **meta: Any) -> WriteReceipt:
        _refuse_project_meta(meta, "supersede()")
        ids, turns = self._cite(sources)
        body = {
            "subject": self._redact(subject, CLAIM_SUBJECT),
            "predicate": predicate,
            "object": self._redact(obj, CLAIM_OBJECT),
            "text": self._redact(text, CLAIM_TEXT),
            "at": _iso(at), "close": closure(close), "reason": closure_reason(reason),
            "confidence": confidence, "polarity": polarity, "extractor": extractor,
            "memory_type": _type(memory_type),
            "valid_from": _iso(valid_from), "valid_to": _iso(valid_to),
            "recorded_at": _iso(recorded_at),
            "source_ids": ids, "sources": turns, "metadata": meta,
        }
        with _as_local_refusal(old_claim_id):
            out = await self._request(
                "POST", f"/v1/memories/{old_claim_id}/supersede", params=self._params(),
                json=_sent(body), write=True)
        return hydrate.receipt(out)

    async def forget(self, subject: str, predicate: str, *, at: datetime | None = None,
                     close: str = "retired", reason: str | None = None) -> list[Claim]:
        why = closure_reason(reason)
        if closure(close) == "ended":
            return await self._end(
                {"subject": subject, "predicate": predicate, "at": _iso(at),
                 "reason": why})
        body = await self._request(
            "POST", "/v1/forget", params=self._params(),
            json=_sent({"subject": subject, "predicate": predicate, "at": _iso(at),
                        "reason": why}),
            write=True)
        return [hydrate.claim(c) for c in body["retired"]]

    async def delete(self, claim_id: str, *, at: datetime | None = None,
                     close: str = "retired", reason: str | None = None) -> bool:
        why = closure_reason(reason)
        if closure(close) == "ended":
            return await self.end(claim_id=claim_id, at=at, reason=why)
        body = await self._request("DELETE", f"/v1/memories/{claim_id}",
                                        params=self._params(),
                                        json=None if why is None else {"reason": why},
                                        write=True)
        return bool(body["retired"])

    async def forget_matching(self, query: str, *, close: str, k: int = 20,
                              reason: str | None = None,
                              confirm: str | None = None) -> ForgetPreview | ForgetResult:
        body = _sent({"query": query, "close": closure(close), "k": k,
                      "reason": closure_reason(reason), "confirm": confirm})
        try:
            out = await self._request("POST", "/v1/forget-matching",
                                           params=self._params(), json=body, write=True)
        except Conflict as exc:
            raise ConfirmationRefused(exc.message) from exc
        if "closed" in out:
            return hydrate.forget_result(out)
        return hydrate.forget_preview(out)

    async def link(self, from_id: str, to_id: str, relation: str, *,
                   by: str = "api") -> Link:
        refuse_self_link(from_id, to_id)
        try:
            out = await self._request(
                "POST", "/v1/links", params=self._params(),
                json={"from_id": from_id, "to_id": to_id,
                      "relation": link_relation(relation), "by": by},
                write=True)
        except NotFound:
            raise KeyError(f"no claim {from_id!r} or {to_id!r} is visible here") from None
        return hydrate.link(out)

    async def links(self, claim_id: str) -> list[Link]:
        try:
            out = await self._request("GET", f"/v1/memories/{claim_id}/links",
                                           params=self._params())
        except NotFound:
            return []
        return [hydrate.link(k) for k in out["claim_links"]]

    async def end(self, *, claim_id: str | None = None, subject: str | None = None,
                 predicate: str | None = None, at: datetime | None = None,
                 reason: str | None = None) -> bool:
        if (claim_id is None) == (predicate is None):
            raise TypeError(
                "end() needs exactly one of: claim_id, to end one memory, or predicate "
                "(with optional subject), to end every current value of that fact.")
        body: dict[str, Any] = {"at": _iso(at), "reason": closure_reason(reason)}
        if claim_id is not None:
            body["memory_id"] = claim_id
        else:
            body["subject"] = subject or "user"
            body["predicate"] = predicate
        return bool(await self._end(body))

    # -- documents -----------------------------------------------------------

    async def add_document(self, content: str | bytes | None = None, *,
                           url: str | None = None, custom_id: str | None = None,
                           title: str | None = None, filepath: str | None = None,
                           mime: str | None = None, meta: Mapping[str, Any] | None = None,
                           extract: bool = True) -> Document:
        """See `RemoteMemvara.add_document`."""
        one_source(content, url)
        body = _document_body(self.redactor, self._redact, content, url=url,
                              custom_id=custom_id, title=title, filepath=filepath,
                              mime=mime, meta=meta, extract=extract)
        return hydrate.document(await self._request(
            "POST", "/v1/documents", params=self._params(), json=body, write=True))

    async def get_document(self, id_or_custom_id: str) -> Document | None:
        try:
            return hydrate.document(await self._request(
                "GET", _document_path(id_or_custom_id), params=self._params()))
        except NotFound:
            return None

    async def list_documents(self, *, filepath_prefix: str | None = None,
                             status: str | None = None, limit: int = 50,
                             cursor: str | None = None) -> Page[Document]:
        return hydrate.document_page(await self._request(
            "GET", "/v1/documents",
            params=self._params(filepath_prefix=filepath_prefix, status=status,
                                limit=limit, cursor=cursor)))

    async def update_document(self, id_or_custom_id: str, *,
                              content: str | bytes | None = None,
                              title: str | None = None,
                              meta: Mapping[str, Any] | None = None,
                              filepath: str | None = None, mime: str | None = None,
                              extract: bool = True) -> Document:
        body = _document_body(self.redactor, self._redact, content, title=title,
                              filepath=filepath, mime=mime, meta=meta, extract=extract)
        try:
            return hydrate.document(await self._request(
                "PATCH", _document_path(id_or_custom_id), params=self._params(),
                json=body, write=True))
        except NotFound:
            raise KeyError(f"no document {id_or_custom_id!r} is visible here") from None

    async def delete_document(self, id_or_custom_id: str) -> DeleteResult:
        try:
            return hydrate.delete_result(await self._request(
                "DELETE", _document_path(id_or_custom_id), params=self._params(),
                write=True))
        except NotFound:
            return DeleteResult(id=id_or_custom_id, deleted=False)

    async def delete_documents(self, ids_or_custom_ids: Sequence[str]) -> list[DeleteResult]:
        body = await self._request("POST", "/v1/documents/delete", params=self._params(),
                                   json={"ids": list(ids_or_custom_ids)}, write=True)
        return [hydrate.delete_result(r) for r in body["results"]]

    async def document_status(self, id_or_custom_id: str) -> DocumentStatus:
        try:
            return hydrate.document_status(await self._request(
                "GET", _document_path(id_or_custom_id, "/status"), params=self._params()))
        except NotFound:
            raise KeyError(f"no document {id_or_custom_id!r} is visible here") from None

    # -- erasure -------------------------------------------------------------

    async def erase(self, claim_id: str, *, sources: bool = False) -> bool:
        body = await self._request(
            "POST", "/v1/erasures", params=self._params(),
            json={"memory_id": claim_id, "sources": sources}, write=True)
        return bool(body["erased"])

    async def purge(self, *, confirm_tenant: str | None = None) -> dict[str, int]:
        """See `RemoteMemvara.purge`, including its refusal while a project is bound."""
        scope = self.default_scope
        refuse_project_purge(scope.project)
        body = await self._request(
            "POST", "/v1/erasures", params=self._params(),
            json=_sent({"scope": _sent({"user": scope.user, "agent": scope.agent,
                                   "session": scope.session}),
                        "confirm_tenant": confirm_tenant}),
            write=True)
        return dict(body["counts"] or {})

    # -- maintenance ---------------------------------------------------------

    async def consolidate(self) -> dict[str, Any]:
        return await self._request("POST", "/v1/maintenance/consolidate",
                                   params=self._params(), write=True)


class AsyncScopedRemoteMemvara:
    """`ScopedRemoteMemvara`'s twin: a scope already filled in, every call awaited.

    Same security property as `ScopedRemoteMemvara` and `ScopedMemvara`: a handler
    holding one of these has no argument with which to address another tenant, user,
    agent or session, because no method here takes one.
    """

    __slots__ = ("_mem", "_scope")

    def __init__(self, mem: AsyncRemoteMemvara, scope: Scope) -> None:
        self._mem = mem._at(scope)
        self._scope = scope

    @property
    def memvara(self) -> AsyncRemoteMemvara:
        """See `ScopedRemoteMemvara.memvara`."""
        return self._mem

    @property
    def scope(self) -> Scope:
        return self._scope

    def __repr__(self) -> str:
        return f"<AsyncScopedRemoteMemvara {self._scope.key()}>"

    # -- service -------------------------------------------------------------

    async def health(self) -> dict[str, Any]:
        return await self._mem.health()

    async def whoami(self) -> dict[str, Any]:
        return await self._mem.whoami()

    async def stats(self) -> dict[str, int]:
        return await self._mem.stats()

    async def connectivity(self) -> dict[str, int]:
        return await self._mem.connectivity()

    # -- reading -------------------------------------------------------------

    # The same three variants as `ScopedRemoteMemvara.search`, and they carry the same
    # weight:
    # they are what makes "calling code cannot tell which it holds" true of the *type* as
    # well as of the value. Without them `await mem.search(q)` types as `list[Retrieved]`
    # here and `list[Result]` on `AsyncMemvara`, so the same expression reading `.claim`
    # off a row checks against one engine and not the other -- and this class's own
    # docstring promises it is `RemoteMemvara`'s twin down to the arguments.
    @overload
    async def search(self, query: str, *, k: int = ..., min_score: float = ...,
                     anchored: bool = ..., ranked: bool = ...,
                     query_rewrite: bool = ...,
                     as_of: datetime | None = ..., valid_at: datetime | None = ...,
                     known_at: datetime | None = ...,
                     states: Collection[str] | None = ...,
                     include_invalidated: bool | None = ...,
                     memory_types: Sequence[MemoryType | str] | None = ...,
                     filters: Mapping[str, FilterValue] | None = ...,
                     filepath_prefix: str | None = ...,
                     include_episodes: Literal[False] = ...) -> list[Result]: ...

    @overload
    async def search(self, query: str, *, k: int = ..., min_score: float = ...,
                     anchored: bool = ..., ranked: bool = ...,
                     query_rewrite: bool = ...,
                     as_of: datetime | None = ..., valid_at: datetime | None = ...,
                     known_at: datetime | None = ...,
                     states: Collection[str] | None = ...,
                     include_invalidated: bool | None = ...,
                     memory_types: Sequence[MemoryType | str] | None = ...,
                     filters: Mapping[str, FilterValue] | None = ...,
                     filepath_prefix: str | None = ...,
                     include_episodes: Literal[True]) -> list[Retrieved]: ...

    @overload
    async def search(self, query: str, *, k: int = ..., min_score: float = ...,
                     anchored: bool = ..., ranked: bool = ...,
                     query_rewrite: bool = ...,
                     as_of: datetime | None = ..., valid_at: datetime | None = ...,
                     known_at: datetime | None = ...,
                     states: Collection[str] | None = ...,
                     include_invalidated: bool | None = ...,
                     memory_types: Sequence[MemoryType | str] | None = ...,
                     filters: Mapping[str, FilterValue] | None = ...,
                     filepath_prefix: str | None = ...,
                     include_episodes: bool) -> list[Retrieved]: ...

    async def search(self, query: str, *, k: int = 10, min_score: float = 0.0,
                     anchored: bool = False, ranked: bool = False,
                     query_rewrite: bool = True,
                     as_of: datetime | None = None, valid_at: datetime | None = None,
                     known_at: datetime | None = None,
                     states: Collection[str] | None = None,
                     include_invalidated: bool | None = None,
                     memory_types: Sequence[MemoryType | str] | None = None,
                     filters: Mapping[str, FilterValue] | None = None,
                     filepath_prefix: str | None = None,
                     include_episodes: bool = False) -> list[Any]:
        return await self._mem.search(query, k=k, min_score=min_score, as_of=as_of,
                                      anchored=anchored, ranked=ranked,
                                      query_rewrite=query_rewrite,
                                      valid_at=valid_at, known_at=known_at,
                                      states=states,
                                      include_invalidated=include_invalidated,
                                      memory_types=memory_types, filters=filters,
                                      filepath_prefix=filepath_prefix,
                                      include_episodes=include_episodes)

    async def recall(self, query: str, *, k: int = 8, min_score: float = 0.0,
                     anchored: bool = False, ranked: bool = False,
                     query_rewrite: bool = True, synthesize: bool = False,
                     memory_types: Sequence[MemoryType | str] | None = None,
                     include_episodes: bool = False,
                     budget: int | None = None,
                     valid_at: datetime | None = None,
                     filters: Mapping[str, FilterValue] | None = None,
                     filepath_prefix: str | None = None) -> str:
        return await self._mem.recall(query, k=k, min_score=min_score, anchored=anchored,
                                      ranked=ranked,
                                      query_rewrite=query_rewrite, synthesize=synthesize,
                                      memory_types=memory_types,
                                      include_episodes=include_episodes,
                                      budget=budget, valid_at=valid_at,
                                      filters=filters, filepath_prefix=filepath_prefix)

    async def get(self, claim_id: str) -> Claim | None:
        return await self._mem.get(claim_id)

    async def get_all(self, *, states: Collection[str] | None = None,
                      include_invalidated: bool | None = None,
                      limit: int = 100, offset: int = 0,
                      as_of: datetime | None = None, valid_at: datetime | None = None,
                      known_at: datetime | None = None) -> list[Claim]:
        return await self._mem.get_all(states=states,
                                       include_invalidated=include_invalidated,
                                       limit=limit, offset=offset, as_of=as_of,
                                       valid_at=valid_at, known_at=known_at)

    async def count(self, *, states: Collection[str] | None = None,
                    include_invalidated: bool | None = None,
                    as_of: datetime | None = None, valid_at: datetime | None = None,
                    known_at: datetime | None = None) -> int:
        return await self._mem.count(states=states,
                                     include_invalidated=include_invalidated,
                                     as_of=as_of, valid_at=valid_at, known_at=known_at)

    async def history(self, subject: str, predicate: str, *,
                      as_of: datetime | None = None, valid_at: datetime | None = None,
                      known_at: datetime | None = None) -> list[Claim]:
        return await self._mem.history(subject, predicate, as_of=as_of,
                                       valid_at=valid_at, known_at=known_at)

    async def why(self, claim_id: str, *, as_of: datetime | None = None,
                  valid_at: datetime | None = None,
                  known_at: datetime | None = None) -> Provenance | None:
        return await self._mem.why(claim_id, as_of=as_of, valid_at=valid_at,
                                   known_at=known_at)

    async def ask(self, question: str, *, at: datetime | None = None, k: int = 3,
                 min_score: float = 0.0, anchored: bool = False) -> Answer:
        return await self._mem.ask(question, at=at, k=k, min_score=min_score,
                                   anchored=anchored)

    async def since(self, when: datetime) -> Delta:
        return await self._mem.since(when)

    async def produced(self, episode_id: str, *, as_of: datetime | None = None,
                       valid_at: datetime | None = None,
                       known_at: datetime | None = None) -> list[Claim]:
        return await self._mem.produced(episode_id, as_of=as_of, valid_at=valid_at,
                                        known_at=known_at)

    async def neighborhood(self, entity: str, *, depth: int = 2, k: int = 10,
                           min_hops: int = 1, predicates: Sequence[str] | None = None,
                           as_of: datetime | None = None,
                           valid_at: datetime | None = None,
                           known_at: datetime | None = None,
                           min_score: float = 0.0) -> list[Path]:
        return await self._mem.neighborhood(entity, depth=depth, k=k, min_hops=min_hops,
                                            predicates=predicates, as_of=as_of,
                                            valid_at=valid_at, known_at=known_at,
                                            min_score=min_score)

    async def paths_between(self, source: str, target: str, *, depth: int = 3,
                            k: int = 3, predicates: Sequence[str] | None = None,
                            as_of: datetime | None = None,
                            valid_at: datetime | None = None,
                            known_at: datetime | None = None,
                            min_score: float = 0.0) -> list[Path]:
        return await self._mem.paths_between(source, target, depth=depth, k=k,
                                             predicates=predicates, as_of=as_of,
                                             valid_at=valid_at, known_at=known_at,
                                             min_score=min_score)

    async def standing(self, *, k: int | None = None) -> list[Claim]:
        return await self._mem.standing(k=k)

    async def profile(self, query: str | None = None, *, k: int = 8,
                      since: datetime | None = None,
                      buckets: Mapping[str, Sequence[str]] | None = None) -> Profile:
        return await self._mem.profile(query, k=k, since=since, buckets=buckets)

    # -- writing -------------------------------------------------------------

    async def add(self, messages: Any, *, role: str = "user",
                  ts: datetime | None = None) -> WriteReceipt:
        return await self._mem.add(messages, role=role, ts=ts)

    async def remember(self, subject: str, predicate: str, obj: str,
                       **kw: Any) -> WriteReceipt:
        return await self._mem.remember(subject, predicate, obj, **kw)

    async def supersede(self, old_claim_id: str, subject: str, predicate: str, obj: str,
                        **kw: Any) -> WriteReceipt:
        return await self._mem.supersede(old_claim_id, subject, predicate, obj, **kw)

    async def forget(self, subject: str, predicate: str, *, at: datetime | None = None,
                     close: str = "retired", reason: str | None = None) -> list[Claim]:
        return await self._mem.forget(subject, predicate, at=at, close=close,
                                      reason=reason)

    async def delete(self, claim_id: str, *, at: datetime | None = None,
                     close: str = "retired", reason: str | None = None) -> bool:
        return await self._mem.delete(claim_id, at=at, close=close, reason=reason)

    async def forget_matching(self, query: str, *, close: str, k: int = 20,
                              reason: str | None = None,
                              confirm: str | None = None) -> ForgetPreview | ForgetResult:
        return await self._mem.forget_matching(query, close=close, k=k, reason=reason,
                                               confirm=confirm)

    async def link(self, from_id: str, to_id: str, relation: str, *,
                   by: str = "api") -> Link:
        return await self._mem.link(from_id, to_id, relation, by=by)

    async def links(self, claim_id: str) -> list[Link]:
        return await self._mem.links(claim_id)

    async def end(self, *, claim_id: str | None = None, subject: str | None = None,
                 predicate: str | None = None, at: datetime | None = None,
                 reason: str | None = None) -> bool:
        return await self._mem.end(claim_id=claim_id, subject=subject,
                                   predicate=predicate, at=at, reason=reason)

    async def erase(self, claim_id: str, *, sources: bool = False) -> bool:
        return await self._mem.erase(claim_id, sources=sources)

    async def add_document(self, content: str | bytes | None = None, *,
                           url: str | None = None, custom_id: str | None = None,
                           title: str | None = None, filepath: str | None = None,
                           mime: str | None = None, meta: Mapping[str, Any] | None = None,
                           extract: bool = True) -> Document:
        return await self._mem.add_document(
            content, url=url, custom_id=custom_id, title=title, filepath=filepath,
            mime=mime, meta=meta, extract=extract)

    async def get_document(self, id_or_custom_id: str) -> Document | None:
        return await self._mem.get_document(id_or_custom_id)

    async def list_documents(self, *, filepath_prefix: str | None = None,
                             status: str | None = None, limit: int = 50,
                             cursor: str | None = None) -> Page[Document]:
        return await self._mem.list_documents(filepath_prefix=filepath_prefix,
                                              status=status, limit=limit, cursor=cursor)

    async def update_document(self, id_or_custom_id: str, *,
                              content: str | bytes | None = None,
                              title: str | None = None,
                              meta: Mapping[str, Any] | None = None,
                              filepath: str | None = None, mime: str | None = None,
                              extract: bool = True) -> Document:
        return await self._mem.update_document(
            id_or_custom_id, content=content, title=title, meta=meta, filepath=filepath,
            mime=mime, extract=extract)

    async def delete_document(self, id_or_custom_id: str) -> DeleteResult:
        return await self._mem.delete_document(id_or_custom_id)

    async def delete_documents(self, ids_or_custom_ids: Sequence[str]) -> list[DeleteResult]:
        return await self._mem.delete_documents(ids_or_custom_ids)

    async def document_status(self, id_or_custom_id: str) -> DocumentStatus:
        return await self._mem.document_status(id_or_custom_id)

    async def purge(self, *, confirm_tenant: str | None = None) -> dict[str, int]:
        return await self._mem.purge(confirm_tenant=confirm_tenant)

    async def consolidate(self) -> dict[str, Any]:
        return await self._mem.consolidate()
