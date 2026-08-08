"""Salience decay.

Salience is the "how much does this still matter" term in ranking. Left alone it is a
one-way ratchet: every reinforcement pushes it up and nothing ever pushes it back, so
after a year of writes every claim is maximally salient and the signal is gone.

The hard constraint is that this runs on a schedule. A naive `salience *= factor` is
wrong for exactly that reason - it compounds, so the same store decays differently
depending on how many times the worker happened to fire. Instead salience is stored as a
*derived* value: an undecayed base is kept in `meta` (`Claim.salience_base`), and each
pass recomputes `base * factor(age)` from scratch. Two passes a microsecond apart
therefore write the same number, and a pass that is skipped or run twice changes nothing.

That arrangement only works if the two halves stay in their own lanes. Decay owns
`salience`; the *write* path owns the base, and raises it through
`Claim.record_observation`. A reinforcement written straight onto `salience` survives
only while `age < 0.415 * half_life` — 2.9 days for a FAST predicate — and is then
erased permanently, because age only ever grows. That is the bug this module's comments
used to promise was impossible, and the reason age is now measured from
`Claim.trace_from` rather than from `valid_from`.
"""

from __future__ import annotations

from datetime import datetime

from ..schema import PredicateRegistry
from ..store.base import Store
from ..telemetry import CONSOLIDATE_DECAYED, Recorder
from ..types import (
    SALIENCE_BASE,
    SALIENCE_PRECISION,
    Claim,
    as_utc,
)
from .sweep import Sweep

# Nothing decays to zero. A claim at 0.0 salience is unrankable and therefore
# unrecoverable - it can never be retrieved, so it can never be reinforced back up.
# The floor keeps a cold fact reachable by a sufficiently specific query.
SALIENCE_FLOOR = 0.05

# Where the undecayed salience lives. Kept in `meta` rather than as a column so the
# store schema (owned elsewhere) does not have to know consolidation exists. The name
# is defined next to the `Claim` field it qualifies; re-exported here because it has
# been part of this module's public surface since before the split.
BASE_KEY = SALIENCE_BASE


def decay_factor(claim: Claim, registry: PredicateRegistry, now: datetime) -> float:
    """Exponential decay on the predicate's half-life: `0.5 ** (age / half_life)`.

    Age runs from `Claim.trace_from` - the later of "when the fact became true" and
    "when we last saw it restated" - and never from `recorded_at`, so back-dating a
    fact we learned late does not hand it artificial freshness while restating one
    still does. A STATIC predicate's 100-year half-life leaves this at ~1.0, which is
    why a birthplace never fades while "what I'm working on today" is halved every week.
    """
    half_life = registry.half_life_days(claim.predicate)
    age_days = (as_utc(now) - claim.trace_from).total_seconds() / 86400.0
    if age_days <= 0.0:
        return 1.0  # future-dated or just-written: nothing to decay yet
    return 0.5 ** (age_days / half_life)


def decayed_salience(claim: Claim, registry: PredicateRegistry,
                     now: datetime) -> tuple[float, float]:
    """The (base, salience) pair this claim should hold at `now`. Pure function."""
    # A decayed value is always <= its base, so `max` is a no-op on a settled claim and
    # re-anchors only when something outside consolidation - a third-party writer, or a
    # store that cannot persist `meta` - pushed salience above the curve. Without this,
    # the next pass would silently undo that bump.
    base = max(claim.salience_base, claim.salience)
    value = max(SALIENCE_FLOOR, base * decay_factor(claim, registry, now))
    return round(base, SALIENCE_PRECISION), round(value, SALIENCE_PRECISION)


def decay_pass(sweep: Sweep, registry: PredicateRegistry) -> int:
    """Recompute salience over a snapshot already in hand. Returns claims changed.

    Split from `decay` so a full sweep pays for one scan of the table instead of one
    per stage - three scans of everything, each materialized, was most of the cost of a
    pass over a store that had nothing left to do.
    """
    changed = 0
    for claim in sweep.claims:
        base, value = decayed_salience(claim, registry, sweep.now)
        if value == claim.salience and claim.meta.get(BASE_KEY) == base:
            continue
        claim.meta[BASE_KEY] = base
        claim.salience = value
        sweep.touch(claim)
        changed += 1
    if sweep.telemetry is not None:
        # Emitted even at zero. "Decay has reported 0 for three months" is a settled
        # store; "no decay series at all" is a scheduler nobody noticed had stopped, and
        # only reporting the zero tells those two apart.
        sweep.telemetry.counter(CONSOLIDATE_DECAYED, changed)
    return changed


def decay(
    store: Store,
    registry: PredicateRegistry,
    tenant: str | None = None,
    now: datetime | None = None,
    window: int | None = None,
    telemetry: Recorder | None = None,
) -> int:
    """Recompute salience for every live claim. Returns the number actually changed.

    A second consecutive call returns 0: the target is a function of stored state and
    `now`, so once written there is nothing left to write.
    """
    sweep = Sweep(store, tenant, now=now, window=window, telemetry=telemetry)
    changed = decay_pass(sweep, registry)
    sweep.flush()
    return changed
