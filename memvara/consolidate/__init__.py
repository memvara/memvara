"""Background maintenance: decay, merge, promote.

This is the subsystem that keeps a year-old store from degrading into noise. Without it
a memory layer only ever accumulates: nothing fades, near-duplicates pile up in the same
slot, and a thing that happened once is indistinguishable from a stable pattern. Those
three failures compound - more claims per slot means worse ranking, and worse ranking
means the write path reinforces the wrong ones.

Everything here is deterministic and off the write path. No stage calls an LLM, and
every stage is idempotent, because this runs on a schedule and a scheduler that fires
twice must not leave a different store than one that fires once.

Off the write path is not the same as out of the way, which is what `Sweep` exists to
fix: the pass reads its snapshot once and writes it back in bounded transactions, so it
neither scans the table three times nor holds the write lock for its own duration.
"""

from __future__ import annotations

from datetime import datetime
from time import perf_counter

from ..embed.base import Embedder
from ..schema import PredicateRegistry
from ..store.base import Store
from ..telemetry import CONSOLIDATE_LATENCY_MS, Recorder
from .decay import BASE_KEY, SALIENCE_FLOOR
from .decay import decay as _decay
from .decay import decay_pass
from .merge import NEIGHBOURHOOD, merge_pass, promote_pass
from .merge import merge_duplicates as _merge_duplicates
from .merge import promote as _promote
from .sweep import DEFAULT_WINDOW, Sweep

__all__ = ["Consolidator", "SALIENCE_FLOOR", "BASE_KEY", "Sweep", "DEFAULT_WINDOW"]


class Consolidator:
    """The scheduled maintenance pass over a store.

    Stage counts are "claims affected": rows this pass changed. On a settled store every
    count is zero, which is the signal that consolidation has converged.
    """

    def __init__(self, store: Store, embedder: Embedder, registry: PredicateRegistry,
                 *, window: int = DEFAULT_WINDOW,
                 telemetry: Recorder | None = None) -> None:
        self.store = store
        self.embedder = embedder
        self.registry = registry
        #: Rows per transaction. Lower it on a store with heavy concurrent write
        #: traffic; the sweep gets slower and the writers wait less.
        self.window = window
        #: Aggregate metrics sink, or `None` (the default and the fast path). This is
        #: the subsystem whose numbers matter most over a year - see
        #: `Sweep._observe_slots` - and the one nobody watches, because a scheduled pass
        #: that has silently stopped running looks exactly like a settled store.
        self.telemetry = telemetry

    def decay(self, tenant: str | None = None, now: datetime | None = None) -> int:
        return _decay(self.store, self.registry, tenant, now, self.window,
                      telemetry=self.telemetry)

    def merge_duplicates(self, tenant: str | None = None, threshold: float = 0.97,
                         *, neighbourhood: int = NEIGHBOURHOOD) -> int:
        return _merge_duplicates(self.store, self.embedder, self.registry, tenant,
                                 threshold, neighbourhood=neighbourhood,
                                 window=self.window, telemetry=self.telemetry)

    def promote(self, tenant: str | None = None, min_observations: int = 3) -> int:
        return _promote(self.store, tenant, min_observations, window=self.window,
                        telemetry=self.telemetry)

    def run(self, tenant: str | None = None,
            now: datetime | None = None) -> dict[str, int]:
        """All three stages, in the order their outputs feed each other.

        Merge runs before promote so a claim only crosses `min_observations` after its
        duplicates' counts have been folded in - otherwise the same evidence split across
        two rows would never promote either of them.

        `now` defaults to the wall clock, read once for the whole pass. Pass it to
        evaluate two passes at the same instant. The decay target is a function of stored
        state *and* the instant it is measured at, so a claim whose recomputed salience
        sits within a pass-duration of a rounding boundary changes on the second call
        over a store nothing has touched. That is the schedule working as intended - a
        nightly pass is meant to move as time passes - but it is not what a caller
        comparing two passes is asking for. `decay()` has taken `now` for the same reason
        since it existed.

        One snapshot serves all three, and one windowed flush writes the result. The
        alternative - a scan and a transaction per stage - cost three full reads of the
        table and up to three writes of any claim more than one stage touched, most of
        it on stores where nothing had changed since the last pass. Holding a single
        transaction across the whole thing was worse still: it is a write outage for
        every other connection, measured as a hard `database is locked` at 5.2 s rather
        than as backpressure.
        """
        t0 = perf_counter() if self.telemetry is not None else 0.0
        sweep = Sweep(self.store, tenant, now=now, window=self.window,
                      telemetry=self.telemetry)
        counts = {
            "decayed": decay_pass(sweep, self.registry),
            "merged": merge_pass(sweep, self.embedder, self.registry),
            "promoted": promote_pass(sweep),
        }
        sweep.flush()
        if self.telemetry is not None:
            self.telemetry.timing(CONSOLIDATE_LATENCY_MS, (perf_counter() - t0) * 1000.0)
        return counts
