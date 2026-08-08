"""One pass's snapshot of a store, and the bounded transactions it writes back through.

Two measured problems live here, and they are the same problem seen from either end.

*Duration.* The sweep used to run inside a single `store.batch()`. SQLite holds the
write lock for a transaction's whole life, so on a 100k-claim store an external writer
did not get slow, it got `OperationalError: database is locked` after 5.4 s — a write
outage rather than backpressure, and one that grows linearly with the store. Committing
every `DEFAULT_WINDOW` rows leaves a gap between windows for anyone else to take the
lock: measured on the same store, every one of 346 concurrent writes landed, worst wait
71 ms, median 10 ms.

*Volume.* Each of decay, merge and promote used to scan and materialize the whole table
for itself. Three scans of everything, plus up to three `put_claim` calls for a claim
that all three touched, on a store where the common case is that nothing changed at all.
The snapshot is read once and the writes are deduplicated by claim id: measured over
20k claims, a settled pass went from 4.8 s and 163 MB to 0.6 s and 40 MB, and the full
sweep over 100k from 107 s to 13 s.

`iter_claims` materializing its rows is deliberate and is respected here rather than
worked around: consolidation mutates rows while iterating, and streaming a live SQLite
cursor through that is undefined behaviour. The snapshot is that materialization, taken
once, outside every transaction this module opens.
"""

from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime

from ..store.base import Store
from ..types import Claim, as_utc, utcnow

#: Rows per transaction. Small enough that a concurrent writer never waits long for the
#: lock - measured at 71 ms worst case, 10 ms median, against 100k claims - and large
#: enough that the sweep still amortizes the commit, which is the win that made batching
#: worth having in the first place.
DEFAULT_WINDOW = 500


class Sweep:
    """A snapshot of one tenant's live claims plus a windowed writer for them.

    Stages mutate the `Claim` objects in `claims` and call `touch` on whatever they
    changed; nothing reaches the store until `flush`. That ordering is what makes a
    multi-stage pass cost one write per claim instead of one per stage, and it keeps
    every stage a pure function of the snapshot - which is the property idempotence
    rests on.
    """

    def __init__(self, store: Store, tenant: str | None = None, *,
                 now: datetime | None = None, window: int | None = None) -> None:
        self.store = store
        self.tenant = tenant
        self.now = as_utc(now or utcnow())
        self.window = max(1, DEFAULT_WINDOW if window is None else int(window))
        self.claims: list[Claim] = list(
            store.iter_claims(tenant, include_invalidated=False))
        # Keyed by id, so a claim decay and merge both touched is written once. Insertion
        # order is preserved, which keeps the write order a function of the data rather
        # than of dict iteration.
        self._dirty: dict[str, Claim] = {}

    def touch(self, claim: Claim) -> None:
        """Mark a claim as needing to be written back at the end of the pass."""
        self._dirty[claim.id] = claim

    def flush(self) -> int:
        """Write every touched claim, committing once per window. Returns rows written."""
        queued = list(self._dirty.values())
        self._dirty.clear()
        # `getattr` so a third-party Store that never heard of batching still works; it
        # just commits per statement, as it did before windowing existed.
        batch = getattr(self.store, "batch", None)
        for start in range(0, len(queued), self.window):
            with (batch() if batch is not None else nullcontext()):
                for claim in queued[start:start + self.window]:
                    self.store.put_claim(claim)
        return len(queued)
