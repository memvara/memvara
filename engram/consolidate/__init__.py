"""Background maintenance: decay, merge, promote.

This is the subsystem that keeps a year-old store from degrading into noise. Without it
a memory layer only ever accumulates: nothing fades, near-duplicates pile up in the same
slot, and a thing that happened once is indistinguishable from a stable pattern. Those
three failures compound - more claims per slot means worse ranking, and worse ranking
means the write path reinforces the wrong ones.

Everything here is deterministic and off the write path. No stage calls an LLM, and
every stage is idempotent, because this runs on a schedule and a scheduler that fires
twice must not leave a different store than one that fires once.
"""

from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime

from ..embed.base import Embedder
from ..schema import PredicateRegistry
from ..store.base import Store
from ..types import utcnow
from .decay import BASE_KEY, SALIENCE_FLOOR
from .decay import decay as _decay
from .merge import merge_duplicates as _merge_duplicates
from .merge import promote as _promote

__all__ = ["Consolidator", "SALIENCE_FLOOR", "BASE_KEY"]


class Consolidator:
    """The scheduled maintenance pass over a store.

    Stage counts are "claims affected": rows this pass actually wrote. On a settled
    store every count is zero, which is the signal that consolidation has converged.
    """

    def __init__(self, store: Store, embedder: Embedder, registry: PredicateRegistry) -> None:
        self.store = store
        self.embedder = embedder
        self.registry = registry

    def decay(self, tenant: str | None = None, now: datetime | None = None) -> int:
        return _decay(self.store, self.registry, tenant, now)

    def merge_duplicates(self, tenant: str | None = None, threshold: float = 0.97) -> int:
        return _merge_duplicates(self.store, self.embedder, tenant, threshold)

    def promote(self, tenant: str | None = None, min_observations: int = 3) -> int:
        return _promote(self.store, tenant, min_observations)

    def run(self, tenant: str | None = None) -> dict[str, int]:
        """All three stages, in the order their outputs feed each other.

        Merge runs before promote so a claim only crosses `min_observations` after its
        duplicates' counts have been folded in - otherwise the same evidence split across
        two rows would never promote either of them.
        """
        now = utcnow()
        # One transaction for the whole sweep. Each stage rewrites many claims, and a
        # durability round-trip per claim buys nothing here: the sweep is one logical
        # operation, and a crash halfway through should leave the pre-sweep store rather
        # than a half-decayed one. `nullcontext` keeps third-party stores working.
        batch = getattr(self.store, "batch", None)
        with (batch() if batch is not None else nullcontext()):
            return {
                "decayed": self.decay(tenant, now=now),
                "merged": self.merge_duplicates(tenant),
                "promoted": self.promote(tenant),
            }
