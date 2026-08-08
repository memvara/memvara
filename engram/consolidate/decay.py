"""Salience decay.

Salience is the "how much does this still matter" term in ranking. Left alone it is a
one-way ratchet: every reinforcement pushes it up and nothing ever pushes it back, so
after a year of writes every claim is maximally salient and the signal is gone.

The hard constraint is that this runs on a schedule. A naive `salience *= factor` is
wrong for exactly that reason - it compounds, so the same store decays differently
depending on how many times the worker happened to fire. Instead salience is stored as a
*derived* value: an undecayed base is kept in `meta`, and each pass recomputes
`base * factor(age)` from scratch. Two passes a microsecond apart therefore write the
same number, and a pass that is skipped or run twice changes nothing.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from ..schema import PredicateRegistry
from ..store.base import Store
from ..types import Claim, utcnow

# Nothing decays to zero. A claim at 0.0 salience is unrankable and therefore
# unrecoverable - it can never be retrieved, so it can never be reinforced back up.
# The floor keeps a cold fact reachable by a sufficiently specific query.
SALIENCE_FLOOR = 0.05

# Where the undecayed salience lives. Kept in `meta` rather than as a column so the
# store schema (owned elsewhere) does not have to know consolidation exists.
BASE_KEY = "salience_base"

# Salience is a ranking weight, not an accounting figure. Quantizing kills the
# sub-nanosecond drift between two scheduler ticks that would otherwise make an
# idempotent pass look like a change.
_PRECISION = 6


def _as_utc(dt: datetime) -> datetime:
    # The store round-trips timestamps through epoch floats and hands back aware
    # datetimes, but a Claim constructed in memory may still carry a naive one.
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def decay_factor(claim: Claim, registry: PredicateRegistry, now: datetime) -> float:
    """Exponential decay on the predicate's half-life: `0.5 ** (age / half_life)`.

    Age is measured from `valid_from` - when the fact became true - not from
    `recorded_at`, so back-dating a fact we learned late does not hand it artificial
    freshness. A STATIC predicate's 100-year half-life leaves this at ~1.0, which is why
    a birthplace never fades while "what I'm working on today" is halved every week.
    """
    half_life = registry.half_life_days(claim.predicate)
    age_days = (now - _as_utc(claim.valid_from)).total_seconds() / 86400.0
    if age_days <= 0.0:
        return 1.0  # future-dated or just-written: nothing to decay yet
    return 0.5 ** (age_days / half_life)


def decayed_salience(claim: Claim, registry: PredicateRegistry, now: datetime) -> tuple[float, float]:
    """The (base, salience) pair this claim should hold at `now`. Pure function."""
    stored = claim.meta.get(BASE_KEY)
    base = float(stored) if isinstance(stored, (int, float)) else claim.salience
    # A decayed value is always <= its base, so `max` is a no-op on a settled claim and
    # re-anchors only when something outside consolidation (a reinforcement on the write
    # path) pushed salience above the curve. Without this, the next pass would silently
    # undo that bump.
    base = max(base, claim.salience)
    value = max(SALIENCE_FLOOR, base * decay_factor(claim, registry, now))
    return round(base, _PRECISION), round(value, _PRECISION)


def decay(
    store: Store,
    registry: PredicateRegistry,
    tenant: str | None = None,
    now: datetime | None = None,
) -> int:
    """Recompute salience for every live claim. Returns the number actually changed.

    A second consecutive call returns 0: the target is a function of stored state and
    `now`, so once written there is nothing left to write.
    """
    at = _as_utc(now or utcnow())
    changed = 0
    claims: Iterable[Claim] = list(store.iter_claims(tenant, include_invalidated=False))
    for claim in claims:
        base, value = decayed_salience(claim, registry, at)
        if value == claim.salience and claim.meta.get(BASE_KEY) == base:
            continue
        claim.meta[BASE_KEY] = base
        claim.salience = value
        store.put_claim(claim)
        changed += 1
    return changed
