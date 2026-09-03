"""The temporal resolver: expression plus anchor to a boundary and a precision.

Everything here is a fixed anchor and a pure call. The resolver is the one place
allowed to turn a phrase into a world-clock instant, so its edges are the edges of
the whole feature: a wrong answer here writes a false `valid_from` that reads
exactly like a true one.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from memvara.write.when import Precision, normalize_unit, resolve

# A Tuesday, so "last week" and ISO week edges are unambiguous.
ANCHOR = datetime(2026, 9, 3, 14, 30, tzinfo=timezone.utc)


def boundary(expression: str, anchor: datetime = ANCHOR) -> datetime | None:
    got = resolve(expression, anchor)
    return None if got is None else got[0]


def precision(expression: str, anchor: datetime = ANCHOR) -> "Precision | None":
    got = resolve(expression, anchor)
    return None if got is None else got[1]


# -- days ------------------------------------------------------------------------

def test_yesterday_is_the_previous_day_at_midnight() -> None:
    assert boundary("yesterday") == datetime(2026, 9, 2, tzinfo=timezone.utc)
    assert precision("yesterday") == "day"


@pytest.mark.parametrize("expression, expected", [
    ("3 days ago", datetime(2026, 8, 31, tzinfo=timezone.utc)),
    ("three days ago", datetime(2026, 8, 31, tzinfo=timezone.utc)),
    ("1 day ago", datetime(2026, 9, 2, tzinfo=timezone.utc)),
    ("3 weeks ago", datetime(2026, 8, 13, tzinfo=timezone.utc)),
])
def test_counted_days_and_weeks_subtract_calendar_days(expression, expected) -> None:
    """`N weeks ago` is 7N days, and both keep `day` precision: the speaker named a
    particular day, however loosely. `last week` names a period instead — see below."""
    assert boundary(expression) == expected
    assert precision(expression) == "day"


# -- calendar-month and -year subtraction, and the clamp ------------------------

def test_months_ago_subtracts_calendar_months() -> None:
    assert boundary("2 months ago") == datetime(2026, 7, 3, tzinfo=timezone.utc)
    assert precision("2 months ago") == "day"


def test_a_month_before_the_31st_clamps_to_the_end_of_a_shorter_month() -> None:
    """There is no 31st of February. Clamping is stated in the design so that two
    implementations cannot pick different defensible answers."""
    march = datetime(2026, 3, 31, 9, 0, tzinfo=timezone.utc)
    assert boundary("1 month ago", march) == datetime(2026, 2, 28, tzinfo=timezone.utc)


def test_the_clamp_lands_on_the_29th_in_a_leap_year() -> None:
    march = datetime(2024, 3, 31, 9, 0, tzinfo=timezone.utc)
    assert boundary("1 month ago", march) == datetime(2024, 2, 29, tzinfo=timezone.utc)


def test_years_ago_clamps_the_same_way() -> None:
    leap = datetime(2024, 2, 29, 9, 0, tzinfo=timezone.utc)
    assert boundary("1 year ago", leap) == datetime(2023, 2, 28, tzinfo=timezone.utc)


# -- periods, which take the period's precision ----------------------------------

def test_last_week_is_the_iso_week_and_takes_week_precision() -> None:
    """ISO weeks begin Monday. The anchor is Tuesday 2026-09-03, so this week began
    2026-08-31 and last week began 2026-08-24."""
    assert boundary("last week") == datetime(2026, 8, 24, tzinfo=timezone.utc)
    assert precision("last week") == "week"


def test_this_week_is_the_monday_of_the_anchors_week() -> None:
    assert boundary("this week") == datetime(2026, 8, 31, tzinfo=timezone.utc)


def test_a_sunday_anchor_still_belongs_to_the_week_that_began_on_monday() -> None:
    sunday = datetime(2026, 9, 6, 23, 0, tzinfo=timezone.utc)
    assert boundary("this week", sunday) == datetime(2026, 8, 31, tzinfo=timezone.utc)


def test_last_month_is_the_first_of_the_previous_month() -> None:
    assert boundary("last month") == datetime(2026, 8, 1, tzinfo=timezone.utc)
    assert precision("last month") == "month"


def test_last_year_and_a_named_year_both_take_year_precision() -> None:
    assert boundary("last year") == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert precision("last year") == "year"
    assert boundary("in 2024") == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert precision("in 2024") == "year"


# -- seasons ---------------------------------------------------------------------

def test_seasons_are_meteorological() -> None:
    assert boundary("last summer") == datetime(2026, 6, 1, tzinfo=timezone.utc)
    assert precision("last summer") == "season"
    assert boundary("last spring") == datetime(2026, 3, 1, tzinfo=timezone.utc)


def test_fall_and_autumn_are_the_same_season() -> None:
    assert boundary("last fall") == boundary("last autumn")


def test_winter_spans_the_year_boundary() -> None:
    """Winter 2025 is December 2025 through February 2026, so the boundary is in
    December of the *earlier* year — the one case where a season's name and its start
    year disagree."""
    assert boundary("last winter") == datetime(2025, 12, 1, tzinfo=timezone.utc)


# -- future, which resolves: the resolver is temporal, not truth-semantic --------

def test_this_winter_is_the_winter_the_speaker_is_standing_in() -> None:
    """Winter starts in December and runs into the next year, so in January and February
    the current season began *last* December. Resolving to the coming December would put
    the boundary eleven months in the future."""
    january = datetime(2026, 1, 15, 12, tzinfo=timezone.utc)
    assert boundary("this winter", january) == datetime(2025, 12, 1, tzinfo=timezone.utc)
    # December is still the winter that starts that month, not the one before it.
    december = datetime(2026, 12, 20, 12, tzinfo=timezone.utc)
    assert boundary("this winter", december) == datetime(2026, 12, 1, tzinfo=timezone.utc)


def test_future_expressions_resolve_rather_than_being_refused() -> None:
    """Whether a future boundary is meaningful for a predicate is decided elsewhere;
    `reconcile._observed_at` already clamps one to the reconciliation instant."""
    assert boundary("tomorrow") == datetime(2026, 9, 4, tzinfo=timezone.utc)
    assert boundary("next month") == datetime(2026, 10, 1, tzinfo=timezone.utc)
    assert boundary("next year") == datetime(2027, 1, 1, tzinfo=timezone.utc)


@pytest.mark.parametrize("expression", ["now", "today", "currently", "these days"])
def test_a_present_tense_marker_declines(expression: str) -> None:
    """These locate a claim at the speaking moment, which the anchor already is to the
    second. Resolving them would replace a precise instant with midnight and stamp a
    `day` precision on it — strictly less information, presented as more. `fast.py`
    excludes them from the tail it captures, and this keeps the two tiers agreeing."""
    assert resolve(expression, ANCHOR) is None


# -- adversarial: everything here must decline rather than guess -----------------

@pytest.mark.parametrize("expression", [
    "not yesterday",
    "I don't remember when",
    "sometime last year",
    "around March",
    "a few weeks ago",
    "recently",
    "lately",
    "",
    "   ",
    "the day the music died",
    "in 20XX",
    "0 days ago",
])
def test_an_expression_it_cannot_resolve_returns_none(expression: str) -> None:
    """`None` is a first-class outcome: the caller falls back to the episode timestamp
    with no precision. A confidently wrong boundary is the failure this guards."""
    assert resolve(expression, ANCHOR) is None


# -- timezone --------------------------------------------------------------------

def test_calendar_arithmetic_runs_in_the_anchors_timezone() -> None:
    """22:30 at +05:30 on the 3rd is 17:00 UTC, and "yesterday" there is the 2nd in that
    offset — not the 2nd in UTC, which is a different instant.

    A fixed offset rather than `ZoneInfo("Asia/Kolkata")`: the zone's identity is not
    what is under test, and Windows ships no system tz database, so a named zone makes
    this a test of whether `tzdata` happens to be installed.
    """
    kolkata = timezone(timedelta(hours=5, minutes=30))
    anchor = datetime(2026, 9, 3, 22, 30, tzinfo=kolkata)
    assert boundary("yesterday", anchor) == datetime(2026, 9, 2, tzinfo=kolkata)


def test_a_naive_anchor_is_utc_which_is_the_conventions_this_library_already_has() -> None:
    """`types.as_utc` treats naive as UTC everywhere else; a second convention here
    would be a silent disagreement."""
    naive = datetime(2026, 9, 3, 14, 30)
    assert boundary("yesterday", naive) == datetime(2026, 9, 2, tzinfo=timezone.utc)


def test_a_dst_transition_does_not_shift_the_day_boundary() -> None:
    """US clocks went forward on 2026-03-08. The day before it is still a whole day, and
    midnight is still midnight, which is what `zoneinfo` arithmetic gives and a 24-hour
    subtraction does not.

    This one genuinely needs a named zone, so it skips where the tz database is absent —
    Windows ships none, and `tzdata` is not a dependency this library imposes for a
    behaviour that only a real zone can exercise.
    """
    ZoneInfo = pytest.importorskip("zoneinfo").ZoneInfo
    try:
        ny = ZoneInfo("America/New_York")
    except Exception:  # pragma: no cover - only where the tz database is missing
        pytest.skip("no system tz database and no tzdata package")
    anchor = datetime(2026, 3, 9, 12, 0, tzinfo=ny)
    assert boundary("yesterday", anchor) == datetime(2026, 3, 8, tzinfo=ny)


# -- units -----------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("mins", "minute"), ("min", "minute"), ("minutes", "minute"), ("Minute", "minute"),
    ("hrs", "hour"), ("hours", "hour"), ("h", "hour"),
    ("km", "kilometer"), ("kilometres", "kilometer"), ("kilometers", "kilometer"),
    ("USD", "usd"), ("dollars", "usd"), ("$", "usd"),
    ("kg", "kilogram"), ("kgs", "kilogram"),
])
def test_units_fold_to_a_canonical_singular_lowercase_form(raw, expected) -> None:
    assert normalize_unit(raw) == expected


def test_an_unrecognised_unit_is_kept_rather_than_guessed_at() -> None:
    """The failure mode has to be an uncanonical unit, never a wrong one. Folding
    `furlongs` to something plausible would be inventing data."""
    assert normalize_unit("furlongs") == "furlongs"
    assert normalize_unit("  Widgets ") == "widgets"


def test_normalizing_nothing_gives_nothing() -> None:
    assert normalize_unit(None) is None
    assert normalize_unit("") is None


@pytest.mark.parametrize("expression, expected", [
    ("next summer", datetime(2027, 6, 1, tzinfo=timezone.utc)),
    ("next winter", datetime(2026, 12, 1, tzinfo=timezone.utc)),
    ("this summer", datetime(2026, 6, 1, tzinfo=timezone.utc)),
])
def test_the_coming_season_is_the_next_one_to_start(expression, expected) -> None:
    """From September 2026: the next summer has not begun, so it is 2027's; the next
    winter begins this December. "this summer" is the one already past this year, which
    is what a person means by it in September."""
    assert boundary(expression) == expected
