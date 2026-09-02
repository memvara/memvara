"""Turning a temporal phrase into a world-clock boundary, deterministically.

This is the only place in the library allowed to decide what "last month" means. Both
write tiers route through it, and neither the regex in `fast.py` nor the extraction model
is permitted to invent a world-clock instant of its own — the model reports the expression
it saw and this module interprets it. Models are unreliable at date arithmetic and
reliable at spotting "three weeks ago", so the error surface lives here, where a test can
pin it against a fixed anchor.

**It declines rather than guesses.** `resolve` returns `None` for anything it does not
recognise, and the caller falls back to the episode's timestamp with no precision. A
misresolved expression writes a `valid_from` that is wrong and looks exactly like one that
is right, which is the silent failure this library's telemetry module exists to catch;
an unresolved one merely writes what the store already writes today.

**It returns a precision as well as an instant**, because normalising "last month" to the
1st invents a day the speaker never said. The instant orders claims and the precision
records how much of it was invented — see `reconcile` for what that buys.

    >>> from datetime import datetime, timezone
    >>> anchor = datetime(2026, 9, 3, 14, 30, tzinfo=timezone.utc)
    >>> resolve("last month", anchor)
    (datetime.datetime(2026, 8, 1, 0, 0, tzinfo=datetime.timezone.utc), 'month')
    >>> resolve("three weeks ago", anchor)[1]
    'day'
    >>> resolve("sometime last year", anchor) is None
    True
"""

from __future__ import annotations

import re
from calendar import monthrange
from datetime import datetime, timedelta, timezone
from typing import Literal

from ..types import as_utc

Precision = Literal["instant", "day", "week", "month", "season", "year"]

#: Meteorological seasons, northern hemisphere, by the month each begins. Meteorological
#: rather than astronomical because the boundaries are calendar dates rather than a
#: computed solstice, and a memory store that answers "last summer" differently depending
#: on an ephemeris is answering a different question than the one it was asked.
#:
#: Winter is the awkward one and the reason this is a dict of start months rather than a
#: range: it spans the year boundary, so winter 2025 runs December 2025 through February
#: 2026 and its *start* is in the earlier year.
_SEASON_START = {"spring": 3, "summer": 6, "autumn": 9, "fall": 9, "winter": 12}

_NUMBER = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "a": 1, "an": 1,
}

_AGO = re.compile(
    r"^(?P<count>\d+|" + "|".join(_NUMBER) + r")\s+"
    r"(?P<unit>day|week|month|year)s?\s+ago$",
    re.IGNORECASE,
)
_RELATIVE_PERIOD = re.compile(
    r"^(?P<which>last|this|next)\s+"
    r"(?P<period>week|month|year|" + "|".join(_SEASON_START) + r")$",
    re.IGNORECASE,
)
_YEAR = re.compile(r"^in\s+(?P<year>\d{4})$", re.IGNORECASE)


def _midnight(moment: datetime) -> datetime:
    return moment.replace(hour=0, minute=0, second=0, microsecond=0)


def _add_months(moment: datetime, months: int) -> datetime:
    """Calendar-month arithmetic, clamped to the last valid day of the destination.

    There is no 31st of February, and the alternatives — overflowing into March, or
    refusing — are both worse than landing on the 28th. Stated in the design so two
    implementations cannot pick different defensible answers.
    """
    total = (moment.year * 12 + moment.month - 1) + months
    year, month = divmod(total, 12)
    month += 1
    return moment.replace(year=year, month=month,
                          day=min(moment.day, monthrange(year, month)[1]))


def _named_season(at: datetime, name: str, step: int) -> datetime:
    """The start of the named season that `last`/`this`/`next` picks out.

    Not "N seasons from here": "last summer" means the most recent summer, which from
    September is three months back, while "last winter" is nine. The name selects the
    month and the qualifier selects the year.

    A winter is named for the December it starts in, so winter 2025 runs from
    2025-12-01 through February 2026 — the one season whose name and calendar year
    disagree for two of its three months.
    """
    start = _midnight(at.replace(month=_SEASON_START[name], day=1))
    if step < 0 and start >= at:
        start = start.replace(year=start.year - 1)
    elif step > 0 and start <= at:
        start = start.replace(year=start.year + 1)
    return start


def resolve(expression: str, anchor: datetime) -> tuple[datetime, Precision] | None:
    """A temporal expression and an anchor, to a boundary and the precision it holds.

    Calendar arithmetic runs in the anchor's timezone when it is aware; a naive anchor is
    UTC, which is `types.as_utc` rather than a second convention invented here. Returns
    `None` for anything unrecognised.

    This is a temporal resolver and not a truth-semantic one: it resolves past, present
    and future alike. Whether a future boundary means anything for a given predicate is
    decided elsewhere, and `reconcile._observed_at` already clamps one to the
    reconciliation instant.
    """
    if not expression or not expression.strip():
        return None
    text = " ".join(expression.strip().lower().split())
    at = as_utc(anchor)

    if text == "yesterday":
        return _midnight(at - timedelta(days=1)), "day"
    if text in {"today", "now"}:
        return _midnight(at), "day"
    if text == "tomorrow":
        return _midnight(at + timedelta(days=1)), "day"

    if (m := _AGO.match(text)) is not None:
        raw = m.group("count")
        count = int(raw) if raw.isdigit() else _NUMBER[raw]
        if count == 0:
            # "0 days ago" is not something a person says; treating it as "today" would
            # be inventing an intent from what is far more likely a parse artifact.
            return None
        unit = m.group("unit")
        if unit == "day":
            return _midnight(at - timedelta(days=count)), "day"
        if unit == "week":
            return _midnight(at - timedelta(weeks=count)), "day"
        if unit == "month":
            return _midnight(_add_months(at, -count)), "day"
        return _midnight(_add_months(at, -12 * count)), "day"

    if (m := _RELATIVE_PERIOD.match(text)) is not None:
        step = {"last": -1, "this": 0, "next": 1}[m.group("which")]
        period = m.group("period")
        if period == "week":
            monday = _midnight(at - timedelta(days=at.weekday()))
            return monday + timedelta(weeks=step), "week"
        if period == "month":
            return _midnight(_add_months(at, step)).replace(day=1), "month"
        if period == "year":
            return _midnight(at).replace(year=at.year + step, month=1, day=1), "year"
        return _named_season(at, period, step), "season"

    if (m := _YEAR.match(text)) is not None:
        return _midnight(at).replace(year=int(m.group("year")), month=1, day=1), "year"

    return None


#: Canonical unit spellings. Deliberately small: this folds spellings of the same unit and
#: does not convert between units, so 120 minutes never becomes 2 hours.
_UNITS = {
    "minute": ("min", "mins", "minute", "minutes"),
    "hour": ("h", "hr", "hrs", "hour", "hours"),
    "day": ("day", "days"),
    "kilometer": ("km", "kms", "kilometer", "kilometers", "kilometre", "kilometres"),
    "mile": ("mi", "mile", "miles"),
    "kilogram": ("kg", "kgs", "kilogram", "kilograms"),
    "usd": ("$", "usd", "dollar", "dollars"),
    "eur": ("eur", "euro", "euros"),
    "gbp": ("gbp", "pound", "pounds"),
}
_CANONICAL = {spelling: canon for canon, group in _UNITS.items() for spelling in group}


def normalize_unit(unit: str | None) -> str | None:
    """Fold a unit to its canonical singular lowercase spelling, or leave it alone.

    The model reports the unit as it appears and this folds it, for the same reason the
    model never computes a date: correctness-critical normalisation does not belong in
    probabilistic output. An unrecognised unit is returned lowercased and stripped rather
    than guessed at, so the failure mode is an uncanonical unit and never a wrong one.

        >>> normalize_unit("mins"), normalize_unit("KM"), normalize_unit("furlongs")
        ('minute', 'kilometer', 'furlongs')
    """
    if unit is None:
        return None
    cleaned = unit.strip().lower()
    if not cleaned:
        return None
    return _CANONICAL.get(cleaned, cleaned)
