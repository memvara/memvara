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

The comparison is blocked rather than exhaustive. Every pair in a `fact_key` group is
O(g²) in time *and* memory, and MANY-cardinality predicates (`likes`, `prefers`, `goal`,
`never_do`) grow monotonically for the life of an account with nothing capping them:
measured 0.21 s at 500 claims in one slot, 3.2 s at 2,000, and 49 s and 289 MB at 8,000,
where most of the cost is a dense 8000x8000 float matrix nobody needed. Blocked, the
same three are 0.07 s, 0.29 s and 1.2 s at 33 MB, and on a planted-duplicate corpus the
two find exactly the same merges. See `NEIGHBOURHOOD`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

import numpy as np

from ..embed.base import Embedder
from ..schema import PredicateRegistry
from ..store.base import Store
from ..types import MAX_SALIENCE, Claim, Derivation, MemoryType, as_utc
from .decay import BASE_KEY, decayed_salience
from .sweep import Sweep

#: How many neighbours in one slot a claim is compared against.
#:
#: Sorted-neighbourhood blocking, the standard answer to quadratic entity resolution.
#: Near-duplicates differ by casing, punctuation or a trailing qualifier ("Acme" /
#: "acme" / "Acme Corp"), so sorting the group by normalized text puts them next to each
#: other; comparing each claim only against the next `NEIGHBOURHOOD - 1` bounds the pass
#: at O(g * NEIGHBOURHOOD) instead of O(g²) and removes the dense matrix entirely.
#:
#: It bounds work rather than losing merges. A cluster larger than the window collapses
#: to one survivor per window in this pass and those survivors are adjacent in the next
#: one, so convergence is logarithmic rather than absent - which a plain "compare only
#: the top N claims" cap cannot say, since it would re-examine the same N forever and
#: never touch the tail.
NEIGHBOURHOOD = 64


def survivor_rank(claim: Claim) -> tuple[int, float, str]:
    """Total order over merge candidates. Lowest sorts first and wins.

    Three keys, and all three are needed: `observation_count` picks the best-attested
    claim, `recorded_at` breaks the common tie toward the one we have believed longest
    (its id is the one already referenced elsewhere), and `id` makes the order total so
    two runs over the same data can never disagree.
    """
    return (-claim.observation_count, as_utc(claim.recorded_at).timestamp(), claim.id)


def _blocking_key(claim: Claim) -> str:
    """Sort key that puts near-duplicates next to each other, for free.

    Casing and whitespace are the two ways the same fact reaches the store twice, and
    both are removable without an embedding. Everything the key fails to bring together
    is still caught on a later sweep once the group around it has thinned.
    """
    return " ".join(claim.text.split()).casefold()


def _unit_vectors(store: Store, embedder: Embedder,
                  claims: Sequence[Claim]) -> np.ndarray:
    """Unit-length rows for `claims`, reusing stored vectors and encoding only gaps.

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
    return unit


def _absorb(survivor: Claim, loser: Claim, registry: PredicateRegistry,
            at: datetime) -> None:
    """Fold a duplicate into the claim that outranks it."""
    # Provenance is the point: the survivor must be able to answer "why do you believe
    # this?" with every episode that ever supported either claim.
    survivor.sources = list(dict.fromkeys(survivor.sources + loser.sources))
    survivor.observation_count += loser.observation_count
    # Salience pooled too, and it is the half that used to be dropped: merging two
    # claims at 1.0 left a survivor at 1.0, so consolidation *reduced* the ranking mass
    # of a fact by finding more evidence for it.
    survivor.meta[BASE_KEY] = min(
        MAX_SALIENCE, survivor.salience_base + loser.salience_base)
    # Pooled on the storage strengths and then re-derived through the same function
    # decay uses - including its rounding, without which the sum of two rounded bases
    # carries float noise. The survivor therefore lands exactly on the curve, and the
    # next pass finds nothing to write instead of correcting this one.
    survivor.meta[BASE_KEY], survivor.salience = decayed_salience(survivor, registry, at)

    loser.invalidated_at = at
    loser.invalidated_by = survivor.id


def merge_pass(sweep: Sweep, embedder: Embedder, registry: PredicateRegistry, *,
               threshold: float = 0.97, neighbourhood: int = NEIGHBOURHOOD) -> int:
    """Fold near-identical live claims over a snapshot in hand. Returns claims retired."""
    at = sweep.now
    groups: dict[str, list[Claim]] = {}
    for claim in sweep.claims:
        if claim.is_live(at):
            groups.setdefault(claim.fact_key, []).append(claim)

    retired = 0
    for group in groups.values():
        if len(group) < 2:
            # The common case by far, and the one that must cost nothing: no vectors are
            # read and no comparison is set up for a slot holding a single answer.
            continue
        group.sort(key=survivor_rank)
        unit = _unit_vectors(sweep.store, embedder, group)
        # Index into `group`, i.e. into survivor_rank order, ordered by blocking key.
        # Keeping the two orders separate is what lets the blocking decide *who is
        # compared* without letting it decide *who survives*.
        order = sorted(range(len(group)), key=lambda i: (_blocking_key(group[i]), i))

        absorbed: set[int] = set()
        for position, i in enumerate(order):
            if i in absorbed:
                continue
            window = [j for j in order[position + 1:position + neighbourhood]
                      if j not in absorbed]
            if not window:
                continue
            sims = unit[window] @ unit[i]
            cluster = [i] + [j for j, s in zip(window, sims) if float(s) >= threshold]
            if len(cluster) < 2:
                continue
            # Lowest index is the best `survivor_rank`. Taken over the whole cluster
            # rather than "whoever the blocking visited first", so the documented order
            # decides the winner even though text order decided the comparison.
            keeper = group[min(cluster)]
            for j in cluster:
                if group[j] is keeper:
                    continue
                absorbed.add(j)
                _absorb(keeper, group[j], registry, at)
                sweep.touch(group[j])
                retired += 1
            sweep.touch(keeper)
    return retired


def promote_pass(sweep: Sweep, min_observations: int = 3) -> int:
    """Reclassify repeatedly-observed EPISODIC claims as SEMANTIC over a snapshot."""
    promoted = 0
    for claim in sweep.claims:
        if claim.memory_type is not MemoryType.EPISODIC:
            continue
        if claim.observation_count < min_observations or not claim.is_live(sweep.now):
            continue
        claim.meta["promoted_from"] = MemoryType.EPISODIC.value
        claim.memory_type = MemoryType.SEMANTIC
        claim.derivation = Derivation.CONSOLIDATION
        sweep.touch(claim)
        promoted += 1
    return promoted


def merge_duplicates(
    store: Store,
    embedder: Embedder,
    registry: PredicateRegistry,
    tenant: str | None = None,
    threshold: float = 0.97,
    now: datetime | None = None,
    *,
    neighbourhood: int = NEIGHBOURHOOD,
    window: int | None = None,
) -> int:
    """Fold near-identical live claims in the same slot into one. Returns claims retired.

    Only claims sharing a `fact_key` are ever compared - two claims answering different
    questions are not duplicates however similar their text reads.
    """
    sweep = Sweep(store, tenant, now=now, window=window)
    retired = merge_pass(sweep, embedder, registry, threshold=threshold,
                         neighbourhood=neighbourhood)
    sweep.flush()
    return retired


def promote(
    store: Store,
    tenant: str | None = None,
    min_observations: int = 3,
    now: datetime | None = None,
    window: int | None = None,
) -> int:
    """Reclassify repeatedly-observed EPISODIC claims as SEMANTIC. Returns count promoted.

    Seeing something happen once is an event; seeing it `min_observations` times is a
    pattern. The distinction is load-bearing at read time - a caller asking only for
    SEMANTIC memory wants the pattern and not the individual occurrences - and it is only
    knowable in hindsight, which is why it belongs here rather than on the write path.
    Promotion is in place: the claim's identity has not changed, only our reading of what
    kind of thing it is.
    """
    sweep = Sweep(store, tenant, now=now, window=window)
    promoted = promote_pass(sweep, min_observations)
    sweep.flush()
    return promoted
