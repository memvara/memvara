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

from copy import copy
from datetime import datetime
from typing import Any, Collection, Mapping, Sequence

from ..redact import CLAIM_OBJECT, CLAIM_SUBJECT, CLAIM_TEXT, EPISODE, Redactor
from ..retrieve import Path, Retrieved
from ..types import (
    Answer, Claim, Delta, Episode, MemoryType, Provenance, Scope, WriteReceipt, closure,
)
from . import hydrate
from .api import _hit, _iso, _sent, _states, _type, _types
from .client import DEFAULT_TIMEOUT, AsyncHttpClient
from .creds import resolve
from .errors import NotFound


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
                 timeout: float = DEFAULT_TIMEOUT,
                 redactor: Redactor | None = None) -> None:
        key, url = resolve(api_key, base_url)
        self._http = AsyncHttpClient(key, url, timeout=timeout)
        #: See `RemoteMemvara.default_scope`: held for `default_scope`'s sake, never sent.
        self.default_scope = Scope(tenant, user, agent, session)
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
              session: str | None = None) -> "AsyncScopedRemoteMemvara":
        """See `RemoteMemvara.scope`."""
        current = self.default_scope
        narrowed = Scope(
            current.tenant,
            user if user is not None else current.user,
            agent if agent is not None else current.agent,
            session if session is not None else current.session,
        )
        return AsyncScopedRemoteMemvara(self, narrowed)

    def _at(self, scope: Scope) -> "AsyncRemoteMemvara":
        """See `RemoteMemvara._at`. No `await` in here: copying an attribute does not
        touch the transport, so there is nothing to make async."""
        twin = copy(self)
        twin.default_scope = scope
        return twin

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
        out = await self._http.request("POST", "/v1/end", params=self._params(),
                                       json=_sent(body), write=True)
        return [hydrate.claim(c) for c in out["ended"]]

    # -- service -------------------------------------------------------------

    async def health(self) -> dict[str, Any]:
        return await self._http.request("GET", "/v1/health")

    async def whoami(self) -> dict[str, Any]:
        return await self._http.request("GET", "/v1/whoami", params=self._params())

    async def stats(self) -> dict[str, int]:
        body = await self._http.request("GET", "/v1/stats", params=self._params())
        return dict(body["tenant_counts"])

    async def connectivity(self) -> dict[str, int]:
        body = await self.stats()
        if "joinable_claims" not in body or "live_claims" not in body:
            return {}
        return {"live_claims": body["live_claims"],
                "joinable_claims": body["joinable_claims"]}

    # -- reading -------------------------------------------------------------

    async def search(self, query: str, *, k: int = 10, min_score: float = 0.0,
                     as_of: datetime | None = None, valid_at: datetime | None = None,
                     known_at: datetime | None = None,
                     states: Collection[str] | None = None,
                     include_invalidated: bool | None = None,
                     memory_types: Sequence[MemoryType | str] | None = None,
                     include_episodes: bool = False) -> list[Retrieved]:
        body = await self._http.request(
            "POST", "/v1/search", params=self._params(),
            json=_sent({"query": query, "k": k, "min_score": min_score,
                        "as_of": _iso(as_of), "valid_at": _iso(valid_at),
                        "known_at": _iso(known_at), "states": _states(states),
                        "include_invalidated": include_invalidated,
                        "memory_types": _types(memory_types),
                        "include_episodes": include_episodes}))
        return [_hit(h) for h in body["results"]]

    async def recall(self, query: str, *, k: int = 8, min_score: float = 0.0,
                     memory_types: Sequence[MemoryType | str] | None = None,
                     include_episodes: bool = False,
                     budget: int | None = None) -> str:
        if budget is not None:
            raise ValueError(
                "recall(budget=...) is not available against a hosted deployment: "
                "POST /v1/recall renders the block server-side and takes no budget. Use "
                "a smaller k, or render your own block from search().")
        body = await self._http.request(
            "POST", "/v1/recall", params=self._params(),
            json=_sent({"query": query, "k": k, "min_score": min_score,
                        "memory_types": _types(memory_types),
                        "include_episodes": include_episodes}))
        return str(body["text"])

    async def get(self, claim_id: str) -> Claim | None:
        try:
            body = await self._http.request("GET", f"/v1/memories/{claim_id}",
                                            params=self._params())
        except NotFound:
            return None
        return hydrate.claim(body)

    async def get_all(self, *, states: Collection[str] | None = None,
                      include_invalidated: bool | None = None,
                      limit: int = 100, offset: int = 0,
                      as_of: datetime | None = None, valid_at: datetime | None = None,
                      known_at: datetime | None = None) -> list[Claim]:
        body = await self._http.request(
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
        body = await self._http.request(
            "GET", "/v1/memories",
            params=self._params(limit=1, states=_states(states),
                                include_invalidated=include_invalidated,
                                as_of=_iso(as_of), valid_at=_iso(valid_at),
                                known_at=_iso(known_at)))
        return int(body["total"])

    async def history(self, subject: str, predicate: str, *,
                      as_of: datetime | None = None, valid_at: datetime | None = None,
                      known_at: datetime | None = None) -> list[Claim]:
        body = await self._http.request(
            "GET", "/v1/history",
            params=self._params(subject=subject, predicate=predicate, as_of=_iso(as_of),
                                valid_at=_iso(valid_at), known_at=_iso(known_at)))
        return [hydrate.claim(c) for c in body["timeline"]]

    async def why(self, claim_id: str, *, as_of: datetime | None = None,
                  valid_at: datetime | None = None,
                  known_at: datetime | None = None) -> Provenance | None:
        try:
            body = await self._http.request(
                "GET", f"/v1/memories/{claim_id}/why",
                params=self._params(as_of=_iso(as_of), valid_at=_iso(valid_at),
                                    known_at=_iso(known_at)))
        except NotFound:
            return None
        return hydrate.provenance(body)

    async def ask(self, question: str, *, at: datetime | None = None, k: int = 3,
                 min_score: float = 0.0) -> Answer:
        body = await self._http.request(
            "POST", "/v1/ask", params=self._params(),
            json=_sent({"question": question, "at": _iso(at), "k": k,
                        "min_score": min_score}))
        return hydrate.answer(body)

    async def since(self, when: datetime) -> Delta:
        body = await self._http.request("GET", "/v1/since",
                                        params=self._params(since=_iso(when)))
        return hydrate.delta(body)

    async def produced(self, episode_id: str, *, as_of: datetime | None = None,
                       valid_at: datetime | None = None,
                       known_at: datetime | None = None) -> list[Claim]:
        body = await self._http.request(
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
        body = await self._http.request(
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
        body = await self._http.request(
            "GET", "/v1/paths",
            params=self._params(source=source, target=target, depth=depth, k=k,
                                predicates=list(predicates) if predicates else None,
                                min_score=min_score, as_of=_iso(as_of),
                                valid_at=_iso(valid_at), known_at=_iso(known_at)))
        return [hydrate.path(p) for p in body["paths"]]

    async def standing(self, *, k: int | None = None) -> list[Claim]:
        body = await self._http.request("GET", "/v1/standing",
                                        params=self._params(limit=k))
        return [hydrate.claim(c) for c in body["memories"]]

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
        body = await self._http.request(
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
                       **meta: Any) -> WriteReceipt:
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
        }
        return hydrate.receipt(await self._http.request(
            "POST", "/v1/facts", params=self._params(), json=_sent(body), write=True))

    async def supersede(self, old_claim_id: str, subject: str, predicate: str, obj: str,
                        *, at: datetime | None = None, close: str = "ended",
                        confidence: float = 1.0,
                        memory_type: MemoryType | str | None = None, polarity: int = 1,
                        valid_from: datetime | None = None,
                        valid_to: datetime | None = None,
                        recorded_at: datetime | None = None,
                        sources: Sequence[Episode | Mapping[str, Any] | str] | None = None,
                        text: str | None = None, extractor: str = "api",
                        **meta: Any) -> WriteReceipt:
        ids, turns = self._cite(sources)
        body = {
            "subject": self._redact(subject, CLAIM_SUBJECT),
            "predicate": predicate,
            "object": self._redact(obj, CLAIM_OBJECT),
            "text": self._redact(text, CLAIM_TEXT),
            "at": _iso(at), "close": closure(close),
            "confidence": confidence, "polarity": polarity, "extractor": extractor,
            "memory_type": _type(memory_type),
            "valid_from": _iso(valid_from), "valid_to": _iso(valid_to),
            "recorded_at": _iso(recorded_at),
            "source_ids": ids, "sources": turns, "metadata": meta,
        }
        return hydrate.receipt(await self._http.request(
            "POST", f"/v1/memories/{old_claim_id}/supersede", params=self._params(),
            json=_sent(body), write=True))

    async def forget(self, subject: str, predicate: str, *, at: datetime | None = None,
                     close: str = "retired") -> list[Claim]:
        if closure(close) == "ended":
            return await self._end(
                {"subject": subject, "predicate": predicate, "at": _iso(at)})
        body = await self._http.request(
            "POST", "/v1/forget", params=self._params(),
            json=_sent({"subject": subject, "predicate": predicate, "at": _iso(at)}),
            write=True)
        return [hydrate.claim(c) for c in body["retired"]]

    async def delete(self, claim_id: str, *, at: datetime | None = None,
                     close: str = "retired") -> bool:
        if closure(close) == "ended":
            return await self.end(claim_id=claim_id, at=at)
        body = await self._http.request("DELETE", f"/v1/memories/{claim_id}",
                                        params=self._params(), write=True)
        return bool(body["retired"])

    async def end(self, *, claim_id: str | None = None, subject: str | None = None,
                 predicate: str | None = None, at: datetime | None = None) -> bool:
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
        return bool(await self._end(body))

    # -- erasure -------------------------------------------------------------

    async def erase(self, claim_id: str, *, sources: bool = False) -> bool:
        body = await self._http.request(
            "POST", "/v1/erasures", params=self._params(),
            json={"memory_id": claim_id, "sources": sources}, write=True)
        return bool(body["erased"])

    async def purge(self, *, confirm_tenant: str | None = None) -> dict[str, int]:
        scope = self.default_scope
        body = await self._http.request(
            "POST", "/v1/erasures", params=self._params(),
            json=_sent({"scope": _sent({"user": scope.user, "agent": scope.agent,
                                        "session": scope.session}),
                        "confirm_tenant": confirm_tenant}),
            write=True)
        return dict(body["counts"] or {})

    # -- maintenance ---------------------------------------------------------

    async def consolidate(self) -> dict[str, Any]:
        return await self._http.request("POST", "/v1/maintenance/consolidate",
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

    async def search(self, query: str, *, k: int = 10, min_score: float = 0.0,
                     as_of: datetime | None = None, valid_at: datetime | None = None,
                     known_at: datetime | None = None,
                     states: Collection[str] | None = None,
                     include_invalidated: bool | None = None,
                     memory_types: Sequence[MemoryType | str] | None = None,
                     include_episodes: bool = False) -> list[Retrieved]:
        return await self._mem.search(query, k=k, min_score=min_score, as_of=as_of,
                                      valid_at=valid_at, known_at=known_at,
                                      states=states,
                                      include_invalidated=include_invalidated,
                                      memory_types=memory_types,
                                      include_episodes=include_episodes)

    async def recall(self, query: str, *, k: int = 8, min_score: float = 0.0,
                     memory_types: Sequence[MemoryType | str] | None = None,
                     include_episodes: bool = False,
                     budget: int | None = None) -> str:
        return await self._mem.recall(query, k=k, min_score=min_score,
                                      memory_types=memory_types,
                                      include_episodes=include_episodes,
                                      budget=budget)

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
                 min_score: float = 0.0) -> Answer:
        return await self._mem.ask(question, at=at, k=k, min_score=min_score)

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
                     close: str = "retired") -> list[Claim]:
        return await self._mem.forget(subject, predicate, at=at, close=close)

    async def delete(self, claim_id: str, *, at: datetime | None = None,
                     close: str = "retired") -> bool:
        return await self._mem.delete(claim_id, at=at, close=close)

    async def end(self, *, claim_id: str | None = None, subject: str | None = None,
                 predicate: str | None = None, at: datetime | None = None) -> bool:
        return await self._mem.end(claim_id=claim_id, subject=subject,
                                   predicate=predicate, at=at)

    async def erase(self, claim_id: str, *, sources: bool = False) -> bool:
        return await self._mem.erase(claim_id, sources=sources)

    async def purge(self, *, confirm_tenant: str | None = None) -> dict[str, int]:
        return await self._mem.purge(confirm_tenant=confirm_tenant)

    async def consolidate(self) -> dict[str, Any]:
        return await self._mem.consolidate()
