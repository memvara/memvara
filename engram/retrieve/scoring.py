"""Post-fusion rescoring: recency, confidence, salience.

The gap this closes: a pure vector store ranks "works at Initech" (recorded 2023,
superseded 2026) exactly as highly as "works at Acme" today, because cosine
similarity has no opinion about time. Ranking by recency alone is equally wrong -
it would bury a birthplace under this morning's mood.

The fix is to key the decay to the *predicate*, not to the memory. `born_in` has a
100-year half-life and never meaningfully decays; `working_on` has a one-week
half-life and falls out of the ranking within days. Volatility is a schema property
we already know, so this stays a pure function - no LLM on the read path.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..schema import PredicateRegistry
from ..types import Claim


def _as_utc(dt: datetime) -> datetime:
    """Naive datetimes are treated as UTC rather than rejected.

    Callers construct these by hand in tests and at API edges; a TypeError deep inside
    ranking is a worse outcome than assuming the convention the rest of the store uses.
    """
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def recency_factor(claim: Claim, registry: PredicateRegistry, now: datetime) -> float:
    """Exponential decay on the predicate's half-life, in [0, 1].

    Age is measured from `valid_from` - when the fact became true in the world - not
    from `recorded_at`. Backfilling a 2019 fact today should not make it look fresh;
    staleness is a property of the fact, not of our bookkeeping.

    `now` is supplied rather than read from the clock so that a time-travel query
    decays relative to the moment being asked about. Asking "what did we believe in
    2024" and getting 2024's facts scored as two years stale would defeat the point.
    """
    half_life = registry.half_life_days(claim.predicate)
    if half_life <= 0.0:
        # No sane spec produces this, but a learned predicate could carry a bad value
        # and a ZeroDivisionError inside ranking is a terrible way to find out.
        return 0.0

    age_days = (_as_utc(now) - _as_utc(claim.valid_from)).total_seconds() / 86400.0
    if age_days <= 0.0:
        # Not yet true, or true as of exactly now. Clamp instead of letting the
        # exponent go positive: a fact scheduled for next month must not outscore a
        # current one by virtue of being in the future.
        return 1.0
    return 0.5 ** (age_days / half_life)


def final_score(
    fusion: float,
    *,
    recency: float,
    confidence: float,
    salience: float,
    w_recency: float,
    w_confidence: float,
    w_salience: float,
) -> float:
    """Scale the fusion score by a quality multiplier.

    Multiplicative, not additive, and that is the load-bearing decision. RRF scores
    live around `1 / (k + 1)` - roughly 0.016 at the default k - while recency,
    confidence and salience all live around 1.0. Adding them would swamp the fusion
    term by two orders of magnitude and turn "search" into "sort every claim in scope
    by salience", which is precisely the ranking pathology this module exists to avoid.

    Scaling keeps relevance primary and lets quality reorder neighbours. With the
    retriever's default weights the multiplier spans 1.25x between a fully decayed
    claim and a fresh one, against an RRF gap of about 1.02x between adjacent ranks -
    enough for freshness to climb roughly a dozen positions, not enough to drag an
    irrelevant claim up from the tail.

    Every factor is monotone increasing and each weight is independent, so setting a
    weight to 0 removes that signal exactly rather than re-baselining the others.
    Salience above 1.0 (heavily reinforced facts) is intentionally not clamped - it is
    headroom the write path earns by observing something repeatedly.
    """
    boost = 1.0 + w_recency * recency + w_confidence * confidence + w_salience * salience
    return fusion * boost
