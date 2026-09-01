"""What `server/tools.py` needs from the memory it is handed, and nothing more.

One tool table serves a local engine and a hosted deployment because both scoped views —
`ScopedMemvara` over a SQLite file, `ScopedRemoteMemvara` over `/v1` — satisfy this
protocol. A second table would be two descriptions of the same nineteen tools, drifting
apart at whichever one a change forgets.

**Every member is derived from a call site, not from a wish list.** Eighteen are reached
as `ctx.memory.<name>`; `connectivity` is reached through a parameter, because `_stats`
hands `ctx.memory` to `_join_rate`. `tests/test_memory_api_protocol.py` reads
`tools.py` and fails in both directions — a call this does not declare, and a member no
call uses.

**Signatures are the intersection of the two views, not the union.** `search` and
`recall` take `Sequence[MemoryType]` rather than the remote client's wider
`Sequence[MemoryType | str]`, because a protocol that promised the wider type would be a
promise `ScopedMemvara` cannot keep. `get_all` declares no `limit`/`offset` even though
the remote view has them, because the tools do not pass them — and a member declared here
is a member every implementation is then held to.

**`standing` is deliberately absent.** A `Protocol` has no optional members: declaring it
would stop `ScopedMemvara` satisfying the protocol its own server is typed against.
`_standing` asks for it with `getattr` and keeps the paging path when it is missing, so
the local engine behaves exactly as it did.

**There is no `memvara` member, and its absence is what makes `ToolContext`'s
security claim checkable.** One existed, typed `Any`, because `_fold_note` reached through
it for the engine's predicate registry — which `RemoteMemvara` does not have, so that read
raised `AttributeError` against a hosted deployment. `_fold_note` now reads the canonical
predicate off the claim the store wrote back, so nothing needs the unscoped client, and
declaring it would hand every handler `.scope(tenant=...)`: precisely the attribute
`ToolContext` says a handler does not have. A protocol that omits it makes that a type
error rather than a promise.
"""

from __future__ import annotations

from datetime import datetime
from typing import (TYPE_CHECKING, Any, Collection, Literal, Protocol, Sequence,
                    runtime_checkable)

from ..retrieve import Path
from ..types import (Answer, Claim, Delta, MemoryType, Provenance, Result, Scope,
                     WriteReceipt)

__all__ = ["MemoryAPI"]


@runtime_checkable
class MemoryAPI(Protocol):
    """A memory already bound to one scope, whatever is serving it.

    `runtime_checkable` so `isinstance` answers the coarse question — does this object
    have the members — which is what a caller wiring an alternative implementation wants
    at startup. It checks names and not signatures, so it is a smoke test rather than the
    guarantee; the guarantee is mypy, over the two assignments at the foot of this file.
    """

    # -- what the view knows about itself ------------------------------------

    @property
    def scope(self) -> Scope:
        """The scope this view is bound to.

        A property rather than an argument, and that is the security model rather than a
        convenience: there is no parameter anywhere below that names a tenant, a user, an
        agent or a session, so a handler holding one of these cannot address another one.
        Read-only here on purpose — `ScopedMemvara` holds it as a plain attribute and
        `ScopedRemoteMemvara` as a property, and only the read is common to both.
        """

    # -- reading -------------------------------------------------------------

    def search(self, query: str, *, k: int = 10, min_score: float = 0.0,
               anchored: bool = False,
               as_of: datetime | None = None, valid_at: datetime | None = None,
               known_at: datetime | None = None,
               states: Collection[str] | None = None,
               include_invalidated: bool | None = None,
               memory_types: Sequence[MemoryType] | None = None,
               include_episodes: Literal[False] = False) -> list[Result]:
        """Hybrid retrieval over claims.

        `include_episodes` is pinned to `False` because that is the only call `tools.py`
        makes, and it is what makes the return `list[Result]` rather than the wider
        `list[Retrieved]`: an episode hit has no `.claim`, and `_search` reads `.claim` on
        every row.
        """

    def recall(self, query: str, *, k: int = 8, min_score: float = 0.0,
               anchored: bool = False,
               memory_types: Sequence[MemoryType] | None = None,
               include_episodes: bool = False,
               budget: int | None = None) -> str:
        """Retrieval already rendered for a system prompt.

        `budget` is declared because `_recall` passes it on every call, as `None` unless
        the model asked for a ceiling. `ScopedRemoteMemvara.recall` accepts the parameter
        and raises for any value other than `None`: `POST /v1/recall` renders server-side
        and takes no budget, and a ceiling silently not applied is an oversized prompt
        with nothing to notice it by.
        """

    def get(self, claim_id: str) -> Claim | None: ...

    def get_all(self, *, states: Collection[str] | None = None,
                include_invalidated: bool | None = None,
                as_of: datetime | None = None, valid_at: datetime | None = None,
                known_at: datetime | None = None) -> list[Claim]:
        """The memories visible at this scope — how many depends on what is serving it.

        `ScopedMemvara` returns the whole scope. `ScopedRemoteMemvara` returns one page,
        because `GET /v1/memories` materializes server-side and its `limit` defaults to
        100 — a parameter this protocol does not declare, so no caller reaching it through
        here can raise it or page past it.

        Harmless today, and named so it stays that way. The one call site is `_standing`'s
        fallback, and it never runs against a hosted deployment: `ScopedRemoteMemvara` has
        `standing`, so `_standing` takes `GET /v1/standing` instead. A new caller that
        needs more than the first hundred memories of a cloud scope has to reach past this
        protocol for the paging arguments, rather than getting a short answer with nothing
        saying it was short.
        """

    def count(self, *, states: Collection[str] | None = None,
              include_invalidated: bool | None = None,
              as_of: datetime | None = None, valid_at: datetime | None = None,
              known_at: datetime | None = None) -> int: ...

    def history(self, subject: str, predicate: str, *,
                as_of: datetime | None = None, valid_at: datetime | None = None,
                known_at: datetime | None = None) -> list[Claim]: ...

    def why(self, claim_id: str, *, as_of: datetime | None = None,
            valid_at: datetime | None = None,
            known_at: datetime | None = None) -> Provenance | None: ...

    def ask(self, question: str, *, at: datetime | None = None, k: int = 3,
            min_score: float = 0.0) -> Answer: ...

    def since(self, when: datetime) -> Delta: ...

    def neighborhood(self, entity: str, *, depth: int = 2, k: int = 10,
                     min_hops: int = 1, predicates: Sequence[str] | None = None,
                     as_of: datetime | None = None, valid_at: datetime | None = None,
                     known_at: datetime | None = None,
                     min_score: float = 0.0) -> list[Path]: ...

    def paths_between(self, source: str, target: str, *, depth: int = 3, k: int = 3,
                      predicates: Sequence[str] | None = None,
                      as_of: datetime | None = None, valid_at: datetime | None = None,
                      known_at: datetime | None = None,
                      min_score: float = 0.0) -> list[Path]: ...

    # -- writing -------------------------------------------------------------

    def add(self, messages: Any, *, role: str = "user",
            ts: datetime | None = None) -> WriteReceipt: ...

    def remember(self, subject: str, predicate: str, obj: str,
                 **kw: Any) -> WriteReceipt:
        """The exact-fact write.

        `**kw` rather than the eight named arguments both views accept, because the two
        spell their optional arguments differently below the surface and `tools.py` passes
        the same six to either. Widening it here would make the protocol assert a
        keyword-by-keyword agreement neither view has been checked for.
        """

    def forget(self, subject: str, predicate: str, *, at: datetime | None = None,
               close: str = "retired") -> list[Claim]: ...

    def delete(self, claim_id: str, *, at: datetime | None = None,
               close: str = "retired") -> bool: ...

    # -- reporting -----------------------------------------------------------

    def stats(self) -> dict[str, int]:
        """Row counts for the whole tenant: `claims`, `live_claims`, `episodes`,
        `embeddings`. Not the scope's own count — `count()` is that one."""

    def connectivity(self) -> dict[str, int]:
        """`live_claims` and `joinable_claims`, or `{}` when the backend cannot measure
        the join. `{}` is not a store with nothing in it, and `_join_rate` prints no line
        rather than a zero nobody measured."""


if TYPE_CHECKING:
    # The real conformance check. `dir()` in the test file compares names; this compares
    # signatures, which is the half that catches a parameter renamed on one view only.
    from ..core import ScopedMemvara
    from ..remote.api import ScopedRemoteMemvara

    def _both_views_satisfy_this(local: ScopedMemvara,
                                 remote: ScopedRemoteMemvara) -> None:
        _local: MemoryAPI = local
        _remote: MemoryAPI = remote
