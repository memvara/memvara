"""`AsyncEngram`: the whole public surface, awaitable.

**The library is not async, and deliberately stays that way.** `async def` is colouring:
it would have to propagate from `Engram` down through `Store`, `WritePipeline`,
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
lock are both hundreds of milliseconds of blocked event loop, and the LangChain,
LlamaIndex and CrewAI adapters all declare async methods whose default implementation
calls the sync one *on the loop thread*. Awaiting instead means the loop keeps serving.

Two things worth knowing before relying on it:

* **It is safe to call concurrently.** `SQLiteStore` guards its connection with an
  `RLock`, so overlapping calls serialize inside the store rather than corrupting it;
  reads mostly do not even do that (see `store.sqlite._read`). Concurrency here buys
  overlap between waiting and computing, not parallel writes — SQLite has one writer.
* **It runs on the default executor**, which is shared with every other `to_thread` in
  the process and holds `min(32, cpu_count + 4)` threads. A burst of writes larger than
  that queues behind itself; `loop.set_default_executor(...)` is the knob if that
  matters.

Constructing the `Engram` is left to the caller, and synchronously, on purpose: opening
the store and loading an embedding model is blocking work that belongs in application
startup, not hidden inside the first `await`.

    >>> import asyncio
    >>> from engram import Engram, NullLLM
    >>> async def main():
    ...     async with AsyncEngram(Engram(llm=NullLLM(), user="alice")) as mem:
    ...         await mem.add("I live in Berlin")
    ...         return [c.text for c in await mem.get_all()]
    >>> asyncio.run(main())
    ['user lives in Berlin']
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Sequence

from .core import Engram, Messages
from .embed import Embedder
from .retrieve import Retrieved
from .types import Claim, Episode, MemoryType, Provenance, Scope, WriteReceipt


class AsyncEngram:
    """An `Engram` whose methods are coroutines. Same semantics, off the loop thread.

    Every method mirrors the synchronous one exactly — same name, same arguments, same
    return value — so the sync docstring is the documentation for both, and there is no
    second set of semantics to keep in step.

    The one omission is `scope()`. Every method here already takes the four scope
    keywords, and an async binding would mean a third facade to keep aligned with the
    other two for no new capability; a server that wants per-request scoping either
    passes `user=` on the call or holds one `AsyncEngram` per scope, both of which are
    free (they share the store).
    """

    __slots__ = ("engram",)

    def __init__(self, engram: Engram) -> None:
        self.engram = engram

    # -- pass-through state --------------------------------------------------
    #
    # Attributes, not coroutines: reading them touches no I/O, and making them awaitable
    # would imply otherwise.

    @property
    def store(self) -> Any:
        return self.engram.store

    @property
    def embedder(self) -> Embedder:
        return self.engram.embedder

    @property
    def default_scope(self) -> Scope:
        return self.engram.default_scope

    @property
    def extractor(self) -> str:
        return self.engram.extractor

    # -- writing -------------------------------------------------------------

    async def add(self, messages: Messages, *, tenant=None, user=None, agent=None,
                  session=None, role: str = "user",
                  ts: datetime | None = None) -> WriteReceipt:
        """See `Engram.add`. The one that most needs to be off the loop: it encodes."""
        return await asyncio.to_thread(
            self.engram.add, messages, tenant=tenant, user=user, agent=agent,
            session=session, role=role, ts=ts)

    async def remember(self, subject: str, predicate: str, obj: str,
                       **kw: Any) -> WriteReceipt:
        """See `Engram.remember`."""
        return await asyncio.to_thread(
            self.engram.remember, subject, predicate, obj, **kw)

    async def supersede(self, old_claim_id: str, new_claim: Claim, *,
                        at: datetime | None = None,
                        sources: Sequence[str | Episode] | None = None,
                        tenant=None, user=None, agent=None,
                        session=None) -> WriteReceipt:
        """See `Engram.supersede`."""
        return await asyncio.to_thread(
            self.engram.supersede, old_claim_id, new_claim, at=at, sources=sources,
            tenant=tenant, user=user, agent=agent, session=session)

    async def forget(self, subject: str, predicate: str, *, tenant=None, user=None,
                     agent=None, session=None,
                     at: datetime | None = None) -> list[Claim]:
        """See `Engram.forget`."""
        return await asyncio.to_thread(
            self.engram.forget, subject, predicate, tenant=tenant, user=user,
            agent=agent, session=session, at=at)

    async def delete(self, claim_id: str, *, at: datetime | None = None, tenant=None,
                     user=None, agent=None, session=None) -> bool:
        """See `Engram.delete` — retires, does not erase."""
        return await asyncio.to_thread(
            self.engram.delete, claim_id, at=at, tenant=tenant, user=user, agent=agent,
            session=session)

    async def erase(self, claim_id: str, *, sources: bool = False, tenant=None,
                    user=None, agent=None, session=None) -> bool:
        """See `Engram.erase` — irreversible."""
        return await asyncio.to_thread(
            self.engram.erase, claim_id, sources=sources, tenant=tenant, user=user,
            agent=agent, session=session)

    async def purge(self, *, tenant=None, user=None, agent=None,
                    session=None) -> dict[str, int]:
        """See `Engram.purge` — irreversible."""
        return await asyncio.to_thread(
            self.engram.purge, tenant=tenant, user=user, agent=agent, session=session)

    async def reset(self, *, tenant=None, user=None, agent=None,
                    session=None) -> dict[str, int]:
        """See `Engram.reset` — irreversible."""
        return await asyncio.to_thread(
            self.engram.reset, tenant=tenant, user=user, agent=agent, session=session)

    # -- reading -------------------------------------------------------------

    async def search(self, query: str, *, k: int = 10, min_score: float = 0.0,
                     tenant=None, user=None, agent=None, session=None,
                     as_of: datetime | None = None, include_invalidated: bool = False,
                     memory_types: Sequence[MemoryType] | None = None,
                     include_episodes: bool = False) -> list[Retrieved]:
        """See `Engram.search`. Encodes the query, so it belongs off the loop too."""
        return await asyncio.to_thread(
            self.engram.search, query, k=k, min_score=min_score, tenant=tenant,
            user=user, agent=agent, session=session, as_of=as_of,
            include_invalidated=include_invalidated, memory_types=memory_types,
            include_episodes=include_episodes)

    async def recall(self, query: str, *, k: int = 8, min_score: float = 0.0,
                     header: str | None = None, tenant=None, user=None, agent=None,
                     session=None, memory_types: Sequence[MemoryType] | None = None,
                     include_episodes: bool = False,
                     episode_header: str | None = None) -> str:
        """See `Engram.recall`."""
        return await asyncio.to_thread(
            self.engram.recall, query, k=k, min_score=min_score, header=header,
            tenant=tenant, user=user, agent=agent, session=session,
            memory_types=memory_types, include_episodes=include_episodes,
            episode_header=episode_header)

    async def get(self, claim_id: str, *, tenant=None, user=None, agent=None,
                  session=None) -> Claim | None:
        """See `Engram.get`."""
        return await asyncio.to_thread(
            self.engram.get, claim_id, tenant=tenant, user=user, agent=agent,
            session=session)

    async def get_all(self, *, tenant=None, user=None, agent=None, session=None,
                      include_invalidated: bool = False,
                      as_of: datetime | None = None) -> list[Claim]:
        """See `Engram.get_all`."""
        return await asyncio.to_thread(
            self.engram.get_all, tenant=tenant, user=user, agent=agent, session=session,
            include_invalidated=include_invalidated, as_of=as_of)

    async def history(self, subject: str, predicate: str, *, tenant=None, user=None,
                      agent=None, session=None) -> list[Claim]:
        """See `Engram.history`."""
        return await asyncio.to_thread(
            self.engram.history, subject, predicate, tenant=tenant, user=user,
            agent=agent, session=session)

    async def why(self, claim_id: str, *, tenant=None, user=None, agent=None,
                  session=None) -> Provenance | None:
        """See `Engram.why`."""
        return await asyncio.to_thread(
            self.engram.why, claim_id, tenant=tenant, user=user, agent=agent,
            session=session)

    async def count(self, *, tenant=None, user=None, agent=None, session=None,
                    as_of: datetime | None = None,
                    include_invalidated: bool = False) -> int:
        """See `Engram.count`."""
        return await asyncio.to_thread(
            self.engram.count, tenant=tenant, user=user, agent=agent, session=session,
            as_of=as_of, include_invalidated=include_invalidated)

    # -- maintenance ---------------------------------------------------------

    async def consolidate(self, *, tenant: str | None = None) -> dict[str, int]:
        """See `Engram.consolidate`. Minutes of work on a large store; await it."""
        return await asyncio.to_thread(self.engram.consolidate, tenant=tenant)

    async def reembed(self, embedder: Embedder | None = None, *,
                      batch_size: int = 256) -> int:
        """See `Engram.reembed`."""
        return await asyncio.to_thread(
            self.engram.reembed, embedder, batch_size=batch_size)

    async def stats(self, *, tenant: str | None = None) -> dict[str, int]:
        """See `Engram.stats`."""
        return await asyncio.to_thread(self.engram.stats, tenant=tenant)

    async def close(self) -> None:
        """See `Engram.close`. Awaitable because it commits and fsyncs."""
        await asyncio.to_thread(self.engram.close)

    async def __aenter__(self) -> "AsyncEngram":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    def __repr__(self) -> str:
        return f"<AsyncEngram of {self.engram!r}>"


#: Names `AsyncEngram` deliberately does not wrap, checked by the test suite so that a
#: method added to `Engram` cannot quietly go missing here. `scope` is explained in the
#: class docstring; the rest are not I/O and would only be `await`ed for symmetry.
NOT_WRAPPED = frozenset({"scope"})


def _unwrapped(engram_type: type = Engram) -> set[str]:
    """Public `Engram` methods with no `AsyncEngram` counterpart. For the suite.

    Lives here rather than in the test so the answer comes from the module under test,
    and so a reader of this file can see what the promise "the whole public surface" is
    actually checked against.
    """
    theirs = {n for n in dir(engram_type)
              if not n.startswith("_") and callable(getattr(engram_type, n, None))}
    return theirs - set(dir(AsyncEngram)) - NOT_WRAPPED


__all__ = ["AsyncEngram"]
