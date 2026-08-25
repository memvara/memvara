#!/usr/bin/env python3
"""Demonstrate why a consolidation pass needs an explicit `now`.

Run this to see the failure that `Consolidator.run(now=...)` exists to prevent:

  PYTHONPATH=. python3 scripts/decay_flake_repro.py

`Sweep` reads the wall clock once per pass, so two back-to-back `run()` calls evaluate
decay at two instants a few milliseconds apart. `decay_pass` compares salience already
rounded to `SALIENCE_PRECISION`, which normally hides that gap - both passes write the
same number and the second reports `decayed: 0`. It stops hiding it when a claim's value
sits within one pass-duration of a rounding boundary: the second pass crosses the
boundary and rewrites a row nothing had touched.

That made `tests/test_decay.py::test_run_twice_leaves_identical_state` fail once on a CI
runner and never in 55 local runs. The three sections below show the mechanism, the rate,
and the fix:

1. A claim placed on a rounding boundary by brute force, then swept twice a millisecond
   apart. The second sweep reports `decayed=1`. Nothing is patched and the comparison in
   `decay_pass` is untouched - only the clock differs between the two passes.
2. How often an arbitrary claim sits that close to an edge, which is why this reached CI
   once rather than never or always.
3. The same claim under `run(now=...)`, where the second pass reports zero.

Kept in the tree because a rate this low is expensive to re-derive: the test it explains
looks unremarkable, and the obvious "fix" - widening the comparison in `decay_pass` - is
the one change that must not be made, since exact equality on the rounded value is what
makes a skipped or doubled pass a no-op.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from memvara.consolidate import Consolidator
from memvara.consolidate.decay import decay_pass, decayed_salience
from memvara.consolidate.sweep import Sweep
from memvara.embed.base import HashingEmbedder
from memvara.schema import PredicateRegistry
from memvara.store.sqlite import SQLiteStore
from memvara.types import Claim, Scope, utcnow

REG = PredicateRegistry()

#: The gap between two passes, which is the first pass's own duration. Measured at 2.3 ms
#: over the six claims `test_run_twice_leaves_identical_state` builds; 1 ms is the
#: conservative round number and the rate scales with it.
GAP = timedelta(milliseconds=1)


def claim_at(now: datetime, age_days: float,
             predicate: str = "working_on") -> Claim:
    """One live claim whose fact became true `age_days` ago."""
    ts = now - timedelta(days=age_days)
    return Claim(subject="user", predicate=predicate, object="the migration",
                 scope=Scope(tenant="acme", user="u1"), valid_from=ts, recorded_at=ts)


def crosses(claim: Claim, at: datetime) -> bool:
    """Does this claim's rounded salience change between `at` and `at + GAP`?"""
    return decayed_salience(claim, REG, at)[1] != decayed_salience(claim, REG, at + GAP)[1]


def on_a_boundary(now: datetime) -> Claim:
    """A claim whose salience crosses a rounding boundary within `GAP`.

    Found by nudging the age a microsecond at a time. Both the age and the base are
    quantized, so for a FAST claim at this age the reachable salience values step by
    about 2.9e-13, and sweeping the 1e-6 gap between two boundaries takes roughly 3.4
    million steps. An edge therefore turns up after a million or two, which is most of
    this script's ~23 s runtime.
    """
    for step in range(2_000_000):
        candidate = claim_at(now, 14.0 + step * 1e-6 / 86400.0)   # 14d, 7d half-life
        if crosses(candidate, now):
            print(f"  found after {step} microsecond nudges of the claim's age")
            return candidate
    raise SystemExit("no rounding boundary within 2e6 microseconds of age - "
                     "check SALIENCE_PRECISION")


def two_unpinned_sweeps(claim: Claim, now: datetime) -> None:
    """What two back-to-back `run()` calls did before `run` took a `now`."""
    store = SQLiteStore(":memory:")
    store.put_claim(claim)
    first = Sweep(store, "acme", now=now)
    print(f"  sweep 1 at T:         decayed={decay_pass(first, REG)}")
    first.flush()
    second = Sweep(store, "acme", now=now + GAP)
    changed = decay_pass(second, REG)
    print(f"  sweep 2 at T + 1ms:   decayed={changed}   "
          f"{'<-- the flake' if changed else '(no change)'}")
    store.close()


def rate(trials: int = 20_000) -> None:
    """How many arbitrary claims sit within `GAP` of a boundary."""
    rng = random.Random(11)
    now = utcnow()
    # Ages stay off SALIENCE_FLOOR, where a claim is pinned and can never cross anything.
    # A FAST claim reaches the floor at about 30 days old.
    for predicate, half_life, lo, hi in (("working_on", 7, 0.1, 29.0),
                                         ("works_at", 730, 30.0, 3000.0)):
        hits = sum(crosses(claim_at(now, rng.uniform(lo, hi), predicate), now)
                   for _ in range(trials))
        print(f"  {predicate:11s} ({half_life:3d}d half-life, {lo}-{hi}d old): "
              f"{hits:5d}/{trials} = {hits / trials:.4%} per claim")


def two_pinned_passes(claim: Claim, now: datetime) -> None:
    """The same claim through the shipped fix: one instant for both passes."""
    store = SQLiteStore(":memory:")
    store.put_claim(claim)
    con = Consolidator(store, HashingEmbedder(dim=64), REG)
    print(f"  run(now=T):           {con.run('acme', now=now)}")
    print(f"  run(now=T) again:     {con.run('acme', now=now)}")
    store.close()


def main() -> None:
    now = utcnow()
    print("1. a claim on a rounding boundary, two real sweeps 1ms apart:")
    claim = on_a_boundary(now)
    print(f"    salience at T       = {decayed_salience(claim, REG, now)[1]}")
    print(f"    salience at T + 1ms = {decayed_salience(claim, REG, now + GAP)[1]}")
    two_unpinned_sweeps(claim, now)
    print("2. how often a claim sits that close to an edge:")
    rate()
    print("3. the same claim, one instant for both passes:")
    two_pinned_passes(claim, now)


if __name__ == "__main__":
    main()
