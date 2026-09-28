"""Fixed instants to write and read at, so a test controls time by passing it.

Nothing here patches the clock. A test that needs a past instant passes one of these as
`valid_from`, `recorded_at` or `valid_at`; everything it does not pass comes from the wall
clock, and a test checks those stamps against a window rather than predicting them.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

#: The first instant of the pool.
EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)

#: Six past instants, one calendar month apart, starting at `EPOCH`. Few enough that two
#: operations drawn at random often land on the same one, which is where the boundary
#: rules (inclusive starts, exclusive ends) are tested.
INSTANTS: tuple[datetime, ...] = tuple(EPOCH.replace(month=m) for m in range(1, 7))

#: An instant no run will reach: a fact in force from here is recorded now and true later.
FAR_FUTURE = datetime(2100, 1, 1, tzinfo=timezone.utc)


def within(value: datetime, before: datetime, after: datetime) -> bool:
    """Whether `value` lies in the closed window from `before` to `after`."""
    return before <= value <= after


def moments_around(t: datetime) -> tuple[datetime, datetime, datetime]:
    """`t` itself and one microsecond either side, the smallest steps a store records."""
    tick = timedelta(microseconds=1)
    return t - tick, t, t + tick
