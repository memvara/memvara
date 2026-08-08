"""Storage protocol.

Everything the engine needs from persistence, and nothing more. The default
implementation is SQLite so the library works with no infrastructure, but the surface
is deliberately narrow enough that pgvector, Qdrant or LanceDB slot in behind it.

Note the shape of `competing_claims`: it is a keyed lookup, not a similarity search.
That signature is the whole reason contradiction detection can be exact rather than
best-effort.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Protocol, Sequence, runtime_checkable

import numpy as np

from ..types import Claim, Episode, Scope


@runtime_checkable
class Store(Protocol):
    # --- episodes ---------------------------------------------------------
    def add_episode(self, ep: Episode) -> None: ...
    def get_episode(self, episode_id: str) -> Episode | None: ...
    def find_episode_by_hash(self, tenant: str, ep_hash: str) -> Episode | None: ...

    def get_episodes(self, episode_ids: Sequence[str]) -> dict[str, Episode]:
        """Bulk fetch, so hydrating a result set is one query rather than one per hit."""
        ...

    def iter_episodes(self, tenant: str | None = None) -> Iterable[Episode]:
        """Every stored turn, optionally for one tenant. What re-embedding walks."""
        ...

    # --- claims -----------------------------------------------------------
    def put_claim(self, claim: Claim) -> None: ...
    def get_claim(self, claim_id: str) -> Claim | None: ...

    def get_claims(self, claim_ids: Sequence[str]) -> dict[str, Claim]:
        """Bulk fetch. Avoids the N+1 that otherwise makes retrieval scale with
        result count rather than with the query."""
        ...

    def batch(self):
        """Context manager deferring commits to one transaction for bulk work."""
        ...

    def competing_claims(self, tenant: str, fact_key: str, as_of: datetime | None = None) -> list[Claim]:
        """Live claims occupying the same (subject, predicate) slot. Exact, indexed."""
        ...

    def find_by_value(self, tenant: str, value_key: str) -> list[Claim]: ...

    def slot_history(self, tenant: str, fact_key: str) -> list[Claim]:
        """Every claim ever recorded in one slot, oldest first — the audit trail."""
        ...

    def invalidate(self, claim_id: str, at: datetime, by: str | None) -> None: ...

    def set_valid_to(self, claim_id: str, valid_to: datetime | None) -> None:
        """Close (or reopen) a claim's valid-time interval.

        Part of the protocol because `Engram.forget` calls it directly: both time axes
        must move together or an `as_of` query lands between them and sees an
        inconsistent world.
        """
        ...

    def reinforce(self, claim_id: str, salience: float, observation_count: int,
                  sources: Sequence[str]) -> None:
        """Record that we saw an already-known fact again."""
        ...

    # --- retrieval --------------------------------------------------------
    def set_embedding(self, claim_id: str, vec: np.ndarray) -> None: ...

    def get_embedding(self, claim_id: str) -> np.ndarray | None:
        """Read a stored vector back, so background work never re-embeds text that has
        already been embedded."""
        ...

    def set_episode_embedding(self, episode_id: str, vec: np.ndarray) -> None:
        """Index a raw turn for semantic recall.

        Its own method rather than an overload of `set_embedding`, because claims and
        episodes are separate objects with separate lifecycles and inferring which one
        an id names is exactly the guess that puts a turn in the claims index.
        """
        ...

    def get_episode_embedding(self, episode_id: str) -> np.ndarray | None:
        """A turn's vector, or None if it was never indexed. Also the cheap probe for
        "has this already been embedded?"."""
        ...

    def clear_embeddings(self) -> int:
        """Drop every stored vector and release the dimension they fixed; return how
        many went.

        Part of the protocol because re-embedding cannot be expressed without it: an
        index fixes its width on the first vector it sees, so migrating to a new model
        must empty the store of vectors before writing the first new one. Claims and
        episodes are untouched — this drops derived data, not memory.
        """
        ...

    def candidate_ids(self, scopes: Sequence[Scope], as_of: datetime | None = None,
                      include_invalidated: bool = False) -> list[str]: ...

    def lexical_search(self, query: str, scopes: Sequence[Scope], limit: int,
                       as_of: datetime | None = None,
                       include_invalidated: bool = False) -> list[tuple[str, float]]: ...

    def vector_search(self, qvec: np.ndarray, scopes: Sequence[Scope], limit: int,
                      as_of: datetime | None = None,
                      include_invalidated: bool = False) -> list[tuple[str, float]]: ...

    # --- episode retrieval ------------------------------------------------
    #
    # The same three primitives over raw turns. They take no `include_invalidated`:
    # episodes are not bitemporal — nothing retires or supersedes them — so there is no
    # end-of-life to lift. `as_of` still applies, and means the one thing it can mean
    # for a turn: it had already happened.
    #
    # Scope filtering is *not* relaxed. It is the same question for raw text as for a
    # derived belief, and raw text is the more sensitive of the two: an unfiltered
    # episode search would hand one session's transcript to a sibling session and one
    # tenant's to another.

    def episode_candidate_ids(self, scopes: Sequence[Scope],
                              as_of: datetime | None = None) -> list[str]: ...

    def lexical_search_episodes(self, query: str, scopes: Sequence[Scope], limit: int,
                                as_of: datetime | None = None
                                ) -> list[tuple[str, float]]: ...

    def vector_search_episodes(self, qvec: np.ndarray, scopes: Sequence[Scope],
                               limit: int, as_of: datetime | None = None
                               ) -> list[tuple[str, float]]: ...

    def purge(self, scope: Scope) -> dict[str, int]:
        """Irreversibly erase a scope: claims, episodes, every vector, both text indexes.

        The one place deletion is correct. Retirement cannot satisfy a legal erasure
        request, because the text stays readable — and an index entry outliving the row
        it describes leaves the purged text searchable, which is the same failure with
        an extra step.
        """
        ...

    # --- learned schema ---------------------------------------------------
    def put_spec(self, spec, tenant: str = "default") -> None:
        """Persist a learned predicate specification. Must survive restart: cardinality
        is what makes a contradiction detectable.

        Scoped to a tenant, because cardinality and volatility are what decide whether a
        claim retires another and how fast it decays — decisions one tenant must not be
        able to make on another's behalf. The default is the tenant `Engram` uses when
        the caller names none, so a single-tenant caller need not think about it.
        """
        ...

    def all_specs(self, tenant: str = "default") -> list:
        """Every persisted predicate specification for one tenant."""
        ...

    # --- maintenance ------------------------------------------------------
    def iter_claims(self, tenant: str | None = None,
                    include_invalidated: bool = False) -> Iterable[Claim]: ...

    def stats(self, tenant: str | None = None) -> dict[str, int]:
        """Row counts, optionally for one tenant. `embeddings` counts what the store
        holds, not what any one process has mapped."""
        ...

    def close(self) -> None: ...
