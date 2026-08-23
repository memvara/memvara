"""`AsyncMemvara`: the whole public surface, awaitable.

**The library is not async, and deliberately stays that way.** `async def` is colouring:
it would have to propagate from `Memvara` down through `Store`, `WritePipeline`,
`Reconciler`, `HybridRetriever` and `Consolidator`, because a coroutine can only be
awaited by a coroutine. That would end the one property this project sells hardest —
a synchronous core with a single dependency, runnable from a script, a notebook, a cron
job or a thread — and it would buy nothing back, because there is no async SQLite. The C
API is blocking; `aiosqlite` is a thread running the same blocking calls behind a queue,
which is exactly what this module is, minus a dependency and a per-connection worker
thread.

So the async surface is a wrapper: every method hands the synchronous one to
`asyncio.to_thread` and awaits the result. What that fixes is real and is the only thing
anyone was actually asking for — `encode()` on a sentence-transformer and a SQLite write
lock are both hundreds of milliseconds of blocked event loop, and blocking the loop for
that long stalls every other request the process is serving. Awaiting instead means the
loop keeps serving.

An earlier version of this paragraph also claimed the LangChain, LlamaIndex and CrewAI
adapters declare async methods that fall back to running the sync one on the loop
thread. That is not true of any of the three — LangChain's `aget_messages` uses
`run_in_executor`, LlamaIndex's `BaseMemory.aget` uses `asyncio.to_thread`, its
`BaseMemoryBlock` is async-primary with no sync fallback at all, and CrewAI's
`StorageBackend` is a bare `Protocol`, so an omitted `asave` is an `AttributeError`
rather than a silent sync call. The argument above stands without it.

Two things worth knowing before relying on it:

* **It is safe to call concurrently.** `SQLiteStore` guards its connection with an
  `RLock`, so overlapping calls serialize inside the store rather than corrupting it;
  reads mostly do not even do that (see `store.sqlite._read`). Concurrency here buys
  overlap between waiting and computing, not parallel writes — SQLite has one writer.
* **It runs on the default executor**, which is shared with every other `to_thread` in
  the process and holds `min(32, cpu_count + 4)` threads. A burst of writes larger than
  that queues behind itself; `loop.set_default_executor(...)` is the knob if that
  matters.

Constructing the `Memvara` is left to the caller, and synchronously, on purpose: opening
the store and loading an embedding model is blocking work that belongs in application
startup, not hidden inside the first `await`.

    >>> import asyncio
    >>> from memvara import Memvara, NullLLM
    >>> async def main():
    ...     async with AsyncMemvara(Memvara(llm=NullLLM(), user="alice")) as mem:
    ...         await mem.add("I live in Berlin")
    ...         return [c.text for c in await mem.get_all()]
    >>> asyncio.run(main())
    ['user lives in Berlin']
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Callable, Collection, Literal, Sequence, overload

from .core import Memvara, Messages, ScopedMemvara, _approx_tokens
from .embed import Embedder
from .retrieve import Path, Retrieved
from .types import (Claim, Delta, Episode, ErasureProof, MemoryType, Provenance,
                    RecallResult, Result, Scope, WriteReceipt)


class AsyncMemvara:
    """An `Memvara` whose methods are coroutines. Same semantics, off the loop thread.

    Every method mirrors the synchronous one exactly — same name, same arguments, same
    return value — so the sync docstring is the documentation for both, and there is no
    second set of semantics to keep in step.

    Nothing is omitted any more. `scope()` used to be, on the argument that every method
    here already takes the four scope keywords so nothing was unreachable without it —
    which was true, and which answered the wrong question: `ScopedMemvara` does not exist
    to reach anything, it exists so that the four keywords are written once instead of on
    every line, because the call site that repeats them is the call site that eventually
    writes one user's fact into another user's scope.

    The workaround for that omission also had a price, and this docstring used to
    understate it. Holding one `AsyncMemvara` per scope means holding one `Memvara` per
    scope — this class has no scope of its own, it forwards the one it is given — and
    constructing a `Memvara` over an existing store is *not* free even though the store
    is shared: it re-reads the persisted predicate specs, re-runs the embedder
    fingerprint check, and builds a fresh `PredicateRegistry` that starts empty of
    anything learned since. Per request, that is several queries and a schema the process
    has already paid for. `scope()` costs a `Scope` and two attribute writes; `registry=`
    and `store=` are still there for the case that genuinely wants two instances.
    """

    __slots__ = ("memvara",)

    def __init__(self, memvara: Memvara) -> None:
        self.memvara = memvara

    # -- pass-through state --------------------------------------------------
    #
    # Attributes, not coroutines: reading them touches no I/O, and making them awaitable
    # would imply otherwise.

    @property
    def store(self) -> Any:
        return self.memvara.store

    @property
    def embedder(self) -> Embedder:
        return self.memvara.embedder

    @property
    def default_scope(self) -> Scope:
        return self.memvara.default_scope

    @property
    def extractor(self) -> str:
        return self.memvara.extractor

    # -- writing -------------------------------------------------------------

    async def add(self, messages: Messages, *, tenant=None, user=None, agent=None,
                  session=None, role: str = "user",
                  ts: datetime | None = None) -> WriteReceipt:
        """See `Memvara.add`. The one that most needs to be off the loop: it encodes."""
        return await asyncio.to_thread(
            self.memvara.add, messages, tenant=tenant, user=user, agent=agent,
            session=session, role=role, ts=ts)

    async def remember(self, subject: str, predicate: str, obj: str,
                       **kw: Any) -> WriteReceipt:
        """See `Memvara.remember`."""
        return await asyncio.to_thread(
            self.memvara.remember, subject, predicate, obj, **kw)

    async def supersede(self, old_claim_id: str, new_claim: Claim, *,
                        at: datetime | None = None,
                        sources: Sequence[str | Episode] | None = None,
                        close: str = "ended",
                        tenant=None, user=None, agent=None,
                        session=None) -> WriteReceipt:
        """See `Memvara.supersede`."""
        return await asyncio.to_thread(
            self.memvara.supersede, old_claim_id, new_claim, at=at, sources=sources,
            close=close, tenant=tenant, user=user, agent=agent, session=session)

    async def forget(self, subject: str, predicate: str, *, tenant=None, user=None,
                     agent=None, session=None, at: datetime | None = None,
                     close: str = "retired") -> list[Claim]:
        """See `Memvara.forget`."""
        return await asyncio.to_thread(
            self.memvara.forget, subject, predicate, tenant=tenant, user=user,
            agent=agent, session=session, at=at, close=close)

    async def delete(self, claim_id: str, *, at: datetime | None = None,
                     close: str = "retired", tenant=None,
                     user=None, agent=None, session=None) -> bool:
        """See `Memvara.delete` — retires, does not erase."""
        return await asyncio.to_thread(
            self.memvara.delete, claim_id, at=at, close=close, tenant=tenant, user=user,
            agent=agent, session=session)

    async def erase(self, claim_id: str, *, sources: bool = False, tenant=None,
                    user=None, agent=None, session=None) -> bool:
        """See `Memvara.erase` — irreversible."""
        return await asyncio.to_thread(
            self.memvara.erase, claim_id, sources=sources, tenant=tenant, user=user,
            agent=agent, session=session)

    async def prove_erased(self, claim_id: str) -> "ErasureProof":
        """See `Memvara.prove_erased`."""
        return await asyncio.to_thread(self.memvara.prove_erased, claim_id)

    async def purge(self, *, tenant=None, user=None, agent=None,
                    session=None) -> dict[str, int]:
        """See `Memvara.purge` — irreversible."""
        return await asyncio.to_thread(
            self.memvara.purge, tenant=tenant, user=user, agent=agent, session=session)

    async def reset(self, *, tenant=None, user=None, agent=None,
                    session=None) -> dict[str, int]:
        """See `Memvara.reset` — irreversible."""
        return await asyncio.to_thread(
            self.memvara.reset, tenant=tenant, user=user, agent=agent, session=session)

    # -- reading -------------------------------------------------------------

    # Mirrored from `Memvara.search`, overloads included — "same name, same arguments,
    # same return value" is the promise this class makes, and a return type that is
    # precise on one facade and a union on the other breaks it in the way that costs
    # most: awaiting the wrapper would be the version you have to narrow.
    @overload
    async def search(self, query: str, *, k: int = ..., min_score: float = ...,
                     tenant=..., user=..., agent=..., session=...,
                     as_of: datetime | None = ..., valid_at: datetime | None = ...,
                     known_at: datetime | None = ...,
                     states: Collection[str] | None = ...,
                     include_invalidated: bool | None = ...,
                     memory_types: Sequence[MemoryType] | None = ...,
                     include_episodes: Literal[False] = ...) -> list[Result]: ...

    @overload
    async def search(self, query: str, *, k: int = ..., min_score: float = ...,
                     tenant=..., user=..., agent=..., session=...,
                     as_of: datetime | None = ..., valid_at: datetime | None = ...,
                     known_at: datetime | None = ...,
                     states: Collection[str] | None = ...,
                     include_invalidated: bool | None = ...,
                     memory_types: Sequence[MemoryType] | None = ...,
                     include_episodes: Literal[True]) -> list[Retrieved]: ...

    @overload
    async def search(self, query: str, *, k: int = ..., min_score: float = ...,
                     tenant=..., user=..., agent=..., session=...,
                     as_of: datetime | None = ..., valid_at: datetime | None = ...,
                     known_at: datetime | None = ...,
                     states: Collection[str] | None = ...,
                     include_invalidated: bool | None = ...,
                     memory_types: Sequence[MemoryType] | None = ...,
                     include_episodes: bool) -> list[Retrieved]: ...

    async def search(self, query: str, *, k: int = 10, min_score: float = 0.0,
                     tenant=None, user=None, agent=None, session=None,
                     as_of: datetime | None = None, valid_at: datetime | None = None,
                     known_at: datetime | None = None,
                     states: Collection[str] | None = None,
                     include_invalidated: bool | None = None,
                     memory_types: Sequence[MemoryType] | None = None,
                     include_episodes: bool = False) -> list[Any]:
        """See `Memvara.search`. Encodes the query, so it belongs off the loop too."""
        return await asyncio.to_thread(
            self.memvara.search, query, k=k, min_score=min_score, tenant=tenant,
            user=user, agent=agent, session=session, as_of=as_of, valid_at=valid_at,
            known_at=known_at, states=states,
            include_invalidated=include_invalidated,
            memory_types=memory_types, include_episodes=include_episodes)

    # Mirrored from `Memvara.recall`, overloads included, for the reason given above
    # `search`: a return type that is precise on one facade and a union on the other is
    # the promise broken where it costs most.
    @overload
    async def recall(self, query: str, *, k: int = ..., min_score: float = ...,
                     header: str | None = ..., tenant=..., user=..., agent=...,
                     session=..., memory_types: Sequence[MemoryType] | None = ...,
                     include_episodes: bool = ..., episode_header: str | None = ...,
                     include_history: bool = ..., history_header: str | None = ...,
                     budget: int | None = ..., counter: Callable[[str], int] = ...,
                     with_ids: Literal[False] = ...) -> str: ...

    @overload
    async def recall(self, query: str, *, k: int = ..., min_score: float = ...,
                     header: str | None = ..., tenant=..., user=..., agent=...,
                     session=..., memory_types: Sequence[MemoryType] | None = ...,
                     include_episodes: bool = ..., episode_header: str | None = ...,
                     include_history: bool = ..., history_header: str | None = ...,
                     budget: int | None = ..., counter: Callable[[str], int] = ...,
                     with_ids: Literal[True]) -> RecallResult: ...

    @overload
    async def recall(self, query: str, *, k: int = ..., min_score: float = ...,
                     header: str | None = ..., tenant=..., user=..., agent=...,
                     session=..., memory_types: Sequence[MemoryType] | None = ...,
                     include_episodes: bool = ..., episode_header: str | None = ...,
                     include_history: bool = ..., history_header: str | None = ...,
                     budget: int | None = ..., counter: Callable[[str], int] = ...,
                     with_ids: bool) -> str | RecallResult: ...

    async def recall(self, query: str, *, k: int = 8, min_score: float = 0.0,
                     header: str | None = None, tenant=None, user=None, agent=None,
                     session=None, memory_types: Sequence[MemoryType] | None = None,
                     include_episodes: bool = False,
                     episode_header: str | None = None,
                     include_history: bool = False,
                     history_header: str | None = None,
                     budget: int | None = None,
                     counter: Callable[[str], int] = _approx_tokens,
                     with_ids: bool = False) -> Any:
        """See `Memvara.recall`."""
        return await asyncio.to_thread(
            self.memvara.recall, query, k=k, min_score=min_score, header=header,
            tenant=tenant, user=user, agent=agent, session=session,
            memory_types=memory_types, include_episodes=include_episodes,
            episode_header=episode_header, include_history=include_history,
            history_header=history_header, budget=budget, counter=counter,
            with_ids=with_ids)

    async def since(self, when: datetime, *, tenant=None, user=None, agent=None,
                    session=None) -> Delta:
        """See `Memvara.since`. Two scope-wide id scans, so it belongs off the loop."""
        return await asyncio.to_thread(
            self.memvara.since, when, tenant=tenant, user=user, agent=agent,
            session=session)

    async def get(self, claim_id: str, *, tenant=None, user=None, agent=None,
                  session=None) -> Claim | None:
        """See `Memvara.get`."""
        return await asyncio.to_thread(
            self.memvara.get, claim_id, tenant=tenant, user=user, agent=agent,
            session=session)

    async def get_all(self, *, tenant=None, user=None, agent=None, session=None,
                      states: Collection[str] | None = None,
                      include_invalidated: bool | None = None,
                      as_of: datetime | None = None, valid_at: datetime | None = None,
                      known_at: datetime | None = None) -> list[Claim]:
        """See `Memvara.get_all`."""
        return await asyncio.to_thread(
            self.memvara.get_all, tenant=tenant, user=user, agent=agent, session=session,
            states=states, include_invalidated=include_invalidated, as_of=as_of,
            valid_at=valid_at, known_at=known_at)

    async def history(self, subject: str, predicate: str, *, tenant=None, user=None,
                      agent=None, session=None, as_of: datetime | None = None,
                      valid_at: datetime | None = None,
                      known_at: datetime | None = None) -> list[Claim]:
        """See `Memvara.history`."""
        return await asyncio.to_thread(
            self.memvara.history, subject, predicate, tenant=tenant, user=user,
            agent=agent, session=session, as_of=as_of, valid_at=valid_at,
            known_at=known_at)

    async def why(self, claim_id: str, *, tenant=None, user=None, agent=None,
                  session=None, as_of: datetime | None = None,
                  valid_at: datetime | None = None,
                  known_at: datetime | None = None) -> Provenance | None:
        """See `Memvara.why`."""
        return await asyncio.to_thread(
            self.memvara.why, claim_id, tenant=tenant, user=user, agent=agent,
            session=session, as_of=as_of, valid_at=valid_at, known_at=known_at)

    async def produced(self, episode_id: str, *, tenant=None, user=None, agent=None,
                       session=None, as_of: datetime | None = None,
                       valid_at: datetime | None = None,
                       known_at: datetime | None = None) -> list[Claim]:
        """See `Memvara.produced`."""
        return await asyncio.to_thread(
            self.memvara.produced, episode_id, tenant=tenant, user=user, agent=agent,
            session=session, as_of=as_of, valid_at=valid_at, known_at=known_at)

    async def count(self, *, tenant=None, user=None, agent=None, session=None,
                    as_of: datetime | None = None, valid_at: datetime | None = None,
                    known_at: datetime | None = None,
                    states: Collection[str] | None = None,
                    include_invalidated: bool | None = None) -> int:
        """See `Memvara.count`."""
        return await asyncio.to_thread(
            self.memvara.count, tenant=tenant, user=user, agent=agent, session=session,
            as_of=as_of, valid_at=valid_at, known_at=known_at, states=states,
            include_invalidated=include_invalidated)

    async def neighborhood(self, entity: str, *, depth: int = 2, k: int = 10,
                           min_hops: int = 1,
                           predicates: Sequence[str] | None = None,
                           as_of: datetime | None = None,
                           valid_at: datetime | None = None,
                           known_at: datetime | None = None, min_score: float = 0.0,
                           tenant=None, user=None, agent=None,
                           session=None) -> list[Path]:
        """See `Memvara.neighborhood`. One store round trip per hop, so it belongs off
        the loop for the same reason `search` does — more so at depth."""
        return await asyncio.to_thread(
            self.memvara.neighborhood, entity, depth=depth, k=k, min_hops=min_hops,
            predicates=predicates, as_of=as_of, valid_at=valid_at, known_at=known_at,
            min_score=min_score, tenant=tenant, user=user, agent=agent,
            session=session)

    async def paths_between(self, source: str, target: str, *, depth: int = 3,
                            k: int = 3, predicates: Sequence[str] | None = None,
                            as_of: datetime | None = None,
                            valid_at: datetime | None = None,
                            known_at: datetime | None = None, min_score: float = 0.0,
                            tenant=None, user=None, agent=None,
                            session=None) -> list[Path]:
        """See `Memvara.paths_between`."""
        return await asyncio.to_thread(
            self.memvara.paths_between, source, target, depth=depth, k=k,
            predicates=predicates, as_of=as_of, valid_at=valid_at, known_at=known_at,
            min_score=min_score, tenant=tenant, user=user, agent=agent,
            session=session)

    # -- maintenance ---------------------------------------------------------

    async def consolidate(self, *, tenant: str | None = None) -> dict[str, int]:
        """See `Memvara.consolidate`. Minutes of work on a large store; await it."""
        return await asyncio.to_thread(self.memvara.consolidate, tenant=tenant)

    async def reembed(self, embedder: Embedder | None = None, *,
                      batch_size: int = 256) -> int:
        """See `Memvara.reembed`."""
        return await asyncio.to_thread(
            self.memvara.reembed, embedder, batch_size=batch_size)

    async def stats(self, *, tenant: str | None = None) -> dict[str, int]:
        """See `Memvara.stats`."""
        return await asyncio.to_thread(self.memvara.stats, tenant=tenant)

    async def connectivity(self, *, tenant: str | None = None) -> dict[str, int]:
        """See `Memvara.connectivity`. Off the loop because it is a semi-join over the
        whole claim table, which is the one counting call that is not cheap."""
        return await asyncio.to_thread(self.memvara.connectivity, tenant=tenant)

    async def close(self) -> None:
        """See `Memvara.close`. Awaitable because it commits and fsyncs."""
        await asyncio.to_thread(self.memvara.close)

    async def __aenter__(self) -> "AsyncMemvara":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    def __repr__(self) -> str:
        return f"<AsyncMemvara of {self.memvara!r}>"

    # -- scoped views --------------------------------------------------------

    def scope(self, *, tenant=None, user=None, agent=None,
              session=None) -> "AsyncScopedMemvara":
        """A view of this facade bound to one scope. See `Memvara.scope`.

        Not a coroutine, and the one method here that is not: it binds four strings and
        touches no store, so awaiting it would advertise I/O that does not happen.

        This used to be the deliberate omission, on the grounds that every method here
        already takes the four keywords so nothing was unreachable. Reachable is not the
        same as safe. The argument for `ScopedMemvara` — that a call site repeating
        `tenant/user/agent/session` on every line eventually gets one wrong, and in a
        memory store that means writing one user's fact into another user's scope —
        is not weakened by the methods being awaitable. It is strengthened: the async
        facade exists for servers, and a server is precisely where one handle per request
        per user is the shape, and where a mistake is someone else's data.
        """
        inner = self.memvara._scope(tenant, user, agent, session)
        return AsyncScopedMemvara(self, inner)


class AsyncScopedMemvara:
    """An `AsyncMemvara` with its scope already filled in. See `ScopedMemvara`.

    The async twin, method for method: same names, same arguments, same return values,
    minus the four scope keywords and plus `await`. `ScopedMemvara`'s docstring is the
    documentation for both.

    It wraps the `AsyncMemvara` rather than the `Memvara`, so every call goes through the
    one `asyncio.to_thread` that already exists on the facade. Wrapping the synchronous
    object instead would mean a second copy of the threading, and two places for the two
    facades to drift apart about what runs off the loop.
    """

    __slots__ = ("_amem", "scope")

    def __init__(self, memvara: AsyncMemvara, scope: Scope) -> None:
        # Deliberately **not** named `_mem`: `integrations._common.bind` sniffs for that
        # attribute to recognise a `ScopedMemvara`, and finding one here would hand a
        # synchronous adapter an `AsyncMemvara` typed as a `Memvara` — every call a
        # coroutine nobody awaits, and nothing raised. Under this name that adapter
        # raises `AttributeError` on `default_scope` instead, which is the right outcome
        # for passing an async handle to a synchronous adapter.
        self._amem = memvara
        self.scope = scope

    @property
    def memvara(self) -> Memvara:
        """The unscoped, **synchronous** `Memvara` underneath.

        The same accessor `ScopedMemvara.memvara` is, resolving to the same kind of
        object, so one name means one thing across all three facades: the real instance,
        with the store and the registry on it, for the server layer that needs them.
        `unscoped` is the async facade.
        """
        return self._amem.memvara

    @property
    def unscoped(self) -> AsyncMemvara:
        """The `AsyncMemvara` this is a view of.

        Has no synchronous counterpart, and is here because `close()` and `reembed()`
        are deliberately absent from a scoped view (they are not scoped operations —
        see `ScopedMemvara`), which on the sync side leaves `view.memvara.close()` as
        the way to reach them. Doing that here would run a commit and an fsync on the
        loop thread, which is the single thing this module exists to prevent. So:
        `await view.unscoped.close()`.
        """
        return self._amem

    # -- narrowing -----------------------------------------------------------

    def bind(self, *, tenant=None, user=None, agent=None,
             session=None) -> "AsyncScopedMemvara":
        """A narrower view. Fields not given keep this view's values."""
        s = self.scope
        return AsyncScopedMemvara(self._amem, Scope(
            tenant if tenant is not None else s.tenant,
            user if user is not None else s.user,
            agent if agent is not None else s.agent,
            session if session is not None else s.session,
        ))

    @property
    def _kw(self) -> dict[str, Any]:
        s = self.scope
        return {"tenant": s.tenant, "user": s.user, "agent": s.agent, "session": s.session}

    # -- writing -------------------------------------------------------------

    async def add(self, messages: Messages, *, role: str = "user",
                  ts: datetime | None = None) -> WriteReceipt:
        return await self._amem.add(messages, role=role, ts=ts, **self._kw)

    async def remember(self, subject: str, predicate: str, obj: str,
                       **kw: Any) -> WriteReceipt:
        return await self._amem.remember(subject, predicate, obj, **self._kw, **kw)

    async def forget(self, subject: str, predicate: str, *,
                     at: datetime | None = None,
                     close: str = "retired") -> list[Claim]:
        return await self._amem.forget(subject, predicate, at=at, close=close, **self._kw)

    async def delete(self, claim_id: str, *, at: datetime | None = None,
                     close: str = "retired") -> bool:
        return await self._amem.delete(claim_id, at=at, close=close, **self._kw)

    async def erase(self, claim_id: str, *, sources: bool = False) -> bool:
        return await self._amem.erase(claim_id, sources=sources, **self._kw)

    async def prove_erased(self, claim_id: str) -> "ErasureProof":
        return await self._amem.prove_erased(claim_id)

    async def supersede(self, old_claim_id: str, new_claim: Claim, *,
                        at: datetime | None = None,
                        sources: Sequence[str | Episode] | None = None,
                        close: str = "ended") -> WriteReceipt:
        return await self._amem.supersede(old_claim_id, new_claim, at=at,
                                          sources=sources, close=close, **self._kw)

    async def purge(self) -> dict[str, int]:
        return await self._amem.purge(**self._kw)

    async def reset(self) -> dict[str, int]:
        return await self._amem.reset(**self._kw)

    # -- reading -------------------------------------------------------------

    # The same three variants as `ScopedMemvara.search`, for the reason given there and
    # again on `AsyncMemvara.search`: this is the object a server layer holds, and it is
    # the one that must not be the more-convenient facade that types worse.
    @overload
    async def search(self, query: str, *, k: int = ..., min_score: float = ...,
                     as_of: datetime | None = ..., valid_at: datetime | None = ...,
                     known_at: datetime | None = ...,
                     states: Collection[str] | None = ...,
                     include_invalidated: bool | None = ...,
                     memory_types: Sequence[MemoryType] | None = ...,
                     include_episodes: Literal[False] = ...) -> list[Result]: ...

    @overload
    async def search(self, query: str, *, k: int = ..., min_score: float = ...,
                     as_of: datetime | None = ..., valid_at: datetime | None = ...,
                     known_at: datetime | None = ...,
                     states: Collection[str] | None = ...,
                     include_invalidated: bool | None = ...,
                     memory_types: Sequence[MemoryType] | None = ...,
                     include_episodes: Literal[True]) -> list[Retrieved]: ...

    @overload
    async def search(self, query: str, *, k: int = ..., min_score: float = ...,
                     as_of: datetime | None = ..., valid_at: datetime | None = ...,
                     known_at: datetime | None = ...,
                     states: Collection[str] | None = ...,
                     include_invalidated: bool | None = ...,
                     memory_types: Sequence[MemoryType] | None = ...,
                     include_episodes: bool) -> list[Retrieved]: ...

    async def search(self, query: str, *, k: int = 10, min_score: float = 0.0,
                     as_of: datetime | None = None, valid_at: datetime | None = None,
                     known_at: datetime | None = None,
                     states: Collection[str] | None = None,
                     include_invalidated: bool | None = None,
                     memory_types: Sequence[MemoryType] | None = None,
                     include_episodes: bool = False) -> list[Any]:
        return await self._amem.search(
            query, k=k, min_score=min_score, as_of=as_of, valid_at=valid_at,
            known_at=known_at, states=states,
            include_invalidated=include_invalidated,
            memory_types=memory_types, include_episodes=include_episodes, **self._kw)

    # The same three variants again, for the reason given on `ScopedMemvara.recall`.
    @overload
    async def recall(self, query: str, *, k: int = ..., min_score: float = ...,
                     header: str | None = ...,
                     memory_types: Sequence[MemoryType] | None = ...,
                     include_episodes: bool = ..., episode_header: str | None = ...,
                     include_history: bool = ..., history_header: str | None = ...,
                     budget: int | None = ..., counter: Callable[[str], int] = ...,
                     with_ids: Literal[False] = ...) -> str: ...

    @overload
    async def recall(self, query: str, *, k: int = ..., min_score: float = ...,
                     header: str | None = ...,
                     memory_types: Sequence[MemoryType] | None = ...,
                     include_episodes: bool = ..., episode_header: str | None = ...,
                     include_history: bool = ..., history_header: str | None = ...,
                     budget: int | None = ..., counter: Callable[[str], int] = ...,
                     with_ids: Literal[True]) -> RecallResult: ...

    @overload
    async def recall(self, query: str, *, k: int = ..., min_score: float = ...,
                     header: str | None = ...,
                     memory_types: Sequence[MemoryType] | None = ...,
                     include_episodes: bool = ..., episode_header: str | None = ...,
                     include_history: bool = ..., history_header: str | None = ...,
                     budget: int | None = ..., counter: Callable[[str], int] = ...,
                     with_ids: bool) -> str | RecallResult: ...

    async def recall(self, query: str, *, k: int = 8, min_score: float = 0.0,
                     header: str | None = None,
                     memory_types: Sequence[MemoryType] | None = None,
                     include_episodes: bool = False,
                     episode_header: str | None = None,
                     include_history: bool = False,
                     history_header: str | None = None,
                     budget: int | None = None,
                     counter: Callable[[str], int] = _approx_tokens,
                     with_ids: bool = False) -> Any:
        return await self._amem.recall(
            query, k=k, min_score=min_score, header=header, memory_types=memory_types,
            include_episodes=include_episodes, episode_header=episode_header,
            include_history=include_history, history_header=history_header,
            budget=budget, counter=counter, with_ids=with_ids, **self._kw)

    async def since(self, when: datetime) -> Delta:
        return await self._amem.since(when, **self._kw)

    async def get(self, claim_id: str) -> Claim | None:
        return await self._amem.get(claim_id, **self._kw)

    async def get_all(self, *, states: Collection[str] | None = None,
                      include_invalidated: bool | None = None,
                      as_of: datetime | None = None, valid_at: datetime | None = None,
                      known_at: datetime | None = None) -> list[Claim]:
        return await self._amem.get_all(
            states=states, include_invalidated=include_invalidated, as_of=as_of,
            valid_at=valid_at, known_at=known_at, **self._kw)

    async def history(self, subject: str, predicate: str, *,
                      as_of: datetime | None = None,
                      valid_at: datetime | None = None,
                      known_at: datetime | None = None) -> list[Claim]:
        return await self._amem.history(subject, predicate, as_of=as_of,
                                        valid_at=valid_at, known_at=known_at, **self._kw)

    async def why(self, claim_id: str, *, as_of: datetime | None = None,
                  valid_at: datetime | None = None,
                  known_at: datetime | None = None) -> Provenance | None:
        return await self._amem.why(claim_id, as_of=as_of, valid_at=valid_at,
                                    known_at=known_at, **self._kw)

    async def produced(self, episode_id: str, *, as_of: datetime | None = None,
                       valid_at: datetime | None = None,
                       known_at: datetime | None = None) -> list[Claim]:
        return await self._amem.produced(episode_id, as_of=as_of, valid_at=valid_at,
                                         known_at=known_at, **self._kw)

    async def neighborhood(self, entity: str, *, depth: int = 2, k: int = 10,
                           min_hops: int = 1,
                           predicates: Sequence[str] | None = None,
                           as_of: datetime | None = None,
                           valid_at: datetime | None = None,
                           known_at: datetime | None = None,
                           min_score: float = 0.0) -> list[Path]:
        return await self._amem.neighborhood(
            entity, depth=depth, k=k, min_hops=min_hops, predicates=predicates,
            as_of=as_of, valid_at=valid_at, known_at=known_at, min_score=min_score,
            **self._kw)

    async def paths_between(self, source: str, target: str, *, depth: int = 3,
                            k: int = 3, predicates: Sequence[str] | None = None,
                            as_of: datetime | None = None,
                            valid_at: datetime | None = None,
                            known_at: datetime | None = None,
                            min_score: float = 0.0) -> list[Path]:
        return await self._amem.paths_between(
            source, target, depth=depth, k=k, predicates=predicates, as_of=as_of,
            valid_at=valid_at, known_at=known_at, min_score=min_score, **self._kw)

    async def count(self, *, as_of: datetime | None = None,
                    valid_at: datetime | None = None,
                    known_at: datetime | None = None,
                    states: Collection[str] | None = None,
                    include_invalidated: bool | None = None) -> int:
        return await self._amem.count(as_of=as_of, valid_at=valid_at, known_at=known_at,
                                      states=states,
                                      include_invalidated=include_invalidated,
                                      **self._kw)

    # -- maintenance ---------------------------------------------------------

    async def consolidate(self) -> dict[str, int]:
        return await self._amem.consolidate(tenant=self.scope.tenant)

    async def stats(self) -> dict[str, int]:
        return await self._amem.stats(tenant=self.scope.tenant)

    async def connectivity(self) -> dict[str, int]:
        return await self._amem.connectivity(tenant=self.scope.tenant)

    def __repr__(self) -> str:
        return f"<AsyncScopedMemvara {self.scope.key()} of {self._amem!r}>"


# --- surface parity, by introspection ------------------------------------------
# Two facades wrap two others, and the failure mode of all four is the same: a method
# lands on one and is forgotten on the next, so the promise "the whole public surface"
# quietly becomes "most of it". A hand-written list of method names would have to be
# edited by the same person who just forgot to edit the class, so both checks below
# derive the surface from the classes themselves.

def _public(obj: type) -> set[str]:
    """The public callables of a class: its method surface, as names."""
    return {n for n in dir(obj)
            if not n.startswith("_") and callable(getattr(obj, n, None))}


def _scoped_omissions() -> set[str]:
    """What a scoped view legitimately does not carry, taken from the synchronous pair.

    `close`, `reembed` and `scope` are not scoped operations — closing is per-store,
    re-embedding rebuilds one index shared by every tenant, and a view that could hand
    out further unscoped views would not be a narrowing. Read off `ScopedMemvara` rather
    than written out, so the async view is held to whatever the sync one actually does:
    the day `ScopedMemvara` grows `reembed`, this stops excusing its absence here.
    """
    return _public(Memvara) - _public(ScopedMemvara)


#: Names `AsyncMemvara` deliberately does not wrap, checked by the test suite so that a
#: method added to `Memvara` cannot quietly go missing here. Empty since `scope` landed;
#: kept because the check needs somewhere to record a deliberate omission, and an empty
#: set is the honest current answer.
NOT_WRAPPED: frozenset[str] = frozenset()


def _unwrapped(memvara_type: type = Memvara) -> set[str]:
    """Public `Memvara` methods with no `AsyncMemvara` counterpart. For the suite.

    Lives here rather than in the test so the answer comes from the module under test,
    and so a reader of this file can see what the promise "the whole public surface" is
    actually checked against.
    """
    return _public(memvara_type) - set(dir(AsyncMemvara)) - NOT_WRAPPED


def _unbound(async_type: type = AsyncMemvara,
             scoped_type: type = AsyncScopedMemvara) -> set[str]:
    """Public `async_type` methods with no `scoped_type` counterpart. For the suite.

    The `_unwrapped` of the scoping axis. Called twice by the suite, because
    `AsyncScopedMemvara` sits at the corner of a square and can fall off either edge: it
    must cover `AsyncMemvara` (or a method is awaitable only unscoped) and it must cover
    `ScopedMemvara` (or a method is scopable only synchronously).
    """
    return _public(async_type) - set(dir(scoped_type)) - _scoped_omissions()


__all__ = ["AsyncMemvara", "AsyncScopedMemvara"]
