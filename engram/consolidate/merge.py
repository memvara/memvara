"""Duplicate merging and episodic -> semantic promotion.

Both operations exist because the write path is deliberately cheap. `Reconciler` catches
exact duplicates by `value_key` and cardinality conflicts by `fact_key`, but two
phrasings of the same fact ("works at Acme" / "works at Acme Corp") occupy the same slot
with different values and are indistinguishable without embeddings. Doing that
comparison on the write path would put an embedding search in front of every turn; doing
it here costs nothing per write and catches the drift in the background.

Neither operation deletes. A merged-away claim is invalidated in *transaction* time only
- we stopped believing it separately, but it was never false - so `valid_to` stays
unset and an `as_of` query from before the merge still returns the pre-merge world.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

import numpy as np

from ..embed.base import Embedder
from ..store.base import Store
from ..types import Claim, Derivation, MemoryType, utcnow


def _as_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def survivor_rank(claim: Claim) -> tuple[int, float, str]:
    """Total order over merge candidates. Lowest sorts first and wins.

    Three keys, and all three are needed: `observation_count` picks the best-attested
    claim, `recorded_at` breaks the common tie toward the one we have believed longest
    (its id is the one already referenced elsewhere), and `id` makes the order total so
    two runs over the same data can never disagree.
    """
    return (-claim.observation_count, _as_utc(claim.recorded_at).timestamp(), claim.id)


def _similarity(store: Store, embedder: Embedder, claims: Sequence[Claim]) -> np.ndarray:
    """Pairwise cosine, reusing stored vectors and encoding only what is missing.

    Re-encoding every claim on every sweep is the expensive way to do this: against a
    hosted embedder it is one network round trip per claim, per scheduled run, for text
    that was already embedded at write time. `store.get_embedding` reads them back, and
    only claims written by a path that skipped embedding fall through to `encode` — in a
    single batched call, not one per claim.
    """
    stored: list[np.ndarray | None] = [store.get_embedding(c.id) for c in claims]
    missing = [i for i, v in enumerate(stored) if v is None]
    if missing:
        encoded = np.asarray(
            embedder.encode([claims[i].text for i in missing]), dtype=np.float32
        )
        for slot, i in enumerate(missing):
            stored[i] = encoded[slot]
    vecs = np.asarray(np.stack(stored), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1)
    unit = vecs / np.where(norms > 0.0, norms, 1.0)[:, None]
    # A zero-norm row has no direction to compare, so it is similar to nothing - that is
    # what keeps an empty-text or un-embeddable claim from merging into an arbitrary
    # neighbour instead of crashing.
    unit[norms == 0.0] = 0.0
    return unit @ unit.T


def merge_duplicates(
    store: Store,
    embedder: Embedder,
    tenant: str | None = None,
    threshold: float = 0.97,
    now: datetime | None = None,
) -> int:
    """Fold near-identical live claims in the same slot into one. Returns claims retired.

    Only claims sharing a `fact_key` are ever compared - two claims answering different
    questions are not duplicates however similar their text reads.
    """
    at = _as_utc(now or utcnow())
    groups: dict[str, list[Claim]] = {}
    for claim in store.iter_claims(tenant, include_invalidated=False):
        if not claim.is_live(at):
            continue
        groups.setdefault(claim.fact_key, []).append(claim)

    retired = 0
    for group in groups.values():
        if len(group) < 2:
            continue
        group.sort(key=survivor_rank)
        sim = _similarity(store, embedder, group)

        absorbed: set[int] = set()
        for i in range(len(group)):
            if i in absorbed:
                continue
            losers = [
                j for j in range(i + 1, len(group))
                if j not in absorbed and float(sim[i, j]) >= threshold
            ]
            if not losers:
                continue
            survivor = group[i]
            for j in losers:
                loser = group[j]
                absorbed.add(j)
                # Provenance is the point: the survivor must be able to answer "why do
                # you believe this?" with every episode that ever supported either claim.
                survivor.sources = list(dict.fromkeys(survivor.sources + loser.sources))
                survivor.observation_count += loser.observation_count
                store.invalidate(loser.id, at=at, by=survivor.id)
                retired += 1
            store.put_claim(survivor)
    return retired


def promote(
    store: Store,
    tenant: str | None = None,
    min_observations: int = 3,
    now: datetime | None = None,
) -> int:
    """Reclassify repeatedly-observed EPISODIC claims as SEMANTIC. Returns count promoted.

    Seeing something happen once is an event; seeing it `min_observations` times is a
    pattern. The distinction is load-bearing at read time - a caller asking only for
    SEMANTIC memory wants the pattern and not the individual occurrences - and it is only
    knowable in hindsight, which is why it belongs here rather than on the write path.
    Promotion is in place: the claim's identity has not changed, only our reading of what
    kind of thing it is.
    """
    at = _as_utc(now or utcnow())
    promoted = 0
    for claim in store.iter_claims(tenant, include_invalidated=False):
        if claim.memory_type is not MemoryType.EPISODIC:
            continue
        if claim.observation_count < min_observations or not claim.is_live(at):
            continue
        claim.meta["promoted_from"] = MemoryType.EPISODIC.value
        claim.memory_type = MemoryType.SEMANTIC
        claim.derivation = Derivation.CONSOLIDATION
        store.put_claim(claim)
        promoted += 1
    return promoted
