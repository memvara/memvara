"""The temporal leg: time as something that *produces* candidates.

Bitemporality is what this library is for, and until this module existed time appeared on
the read path twice, in both cases too late to matter:

* as a **filter** — `valid_at` and `known_at` narrow what the store will return, so a
  candidate has to survive them, and one that no other leg found is never a candidate to
  begin with;
* as a **multiplier** — `recency_factor` scales a claim already in the fused list, so it
  reorders the answer and cannot add to it.

Neither can answer "what was going on around then". That question is answered by a turn
which need share no vocabulary with it at all: the words in it are `when`, `around` and
`then`, every one of which the lexical analyzer drops as a stopword and the embedder maps
onto nothing in particular. So the leg here ranks on *when* and reads no text.

**The anchor is given, never parsed.** A date parser on the read path would be a second
extractor with its own locale bugs, and the question it answers — what instant does this
sentence mean — is exactly the one a caller who wrote `valid_at=` has already answered.
So the anchor is the instant the search was asked about (see `anchor_for`), and when no
instant was given it is the present, which makes the leg "the turns nearest to now" — the
right reading of "recently" and the wrong one of "in 2019", which is why the second is
expected to be spelled with `valid_at`.

**And it abstains rather than voting weakly.** If nothing in scope is within a half-life
of the anchor, time has no opinion about this question, and a leg with no opinion that
votes anyway is worse than one that does not run — fusion reads positions, so a leg
ranking a set of equally-irrelevant turns still contributes rank 0, rank 1, rank 2. The
other two legs have the same guard for the same reason: a zero-norm query and a
stopword-only query both abstain. See `MIN_PROXIMITY`.

**Episodes, not claims.** A claim already carries the predicate-keyed half-life that
`recency_factor` applies, which is a better time signal than raw proximity: a `born_in`
from 2019 is as current as it will ever be, and a `working_on` from 2019 is not, and only
the predicate knows which. A turn has no predicate and no decay, and it is the thing a
"what was happening then" question is actually about.
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

from ..types import as_utc

#: Fusion leg name, shared with `hybrid.py` so a rename cannot split them.
TEMPORAL = "temporal"

#: Days apart at which a turn is judged half as close to the anchor.
#:
#: The one fitted constant here, and it is a pure calibration knob rather than a ranking
#: one: `proximity` is monotone decreasing in the distance for **every** positive value of
#: it, so the order this leg produces is the same at 1 day and at 1,000. What moves is the
#: number in `Explanation.temporal_score` and therefore the leg's weight in the relevance
#: average — how *convinced* time is, not what it prefers.
#:
#: 30 days, because that is the scale a personal store's "around then" tends to mean: a
#: turn from the same week scores above 0.8 and one from six months ago below 0.15.
PROXIMITY_HALF_LIFE_DAYS = 30.0

#: How close the *nearest* turn has to be before this leg votes at all, as a proximity.
#:
#: 0.5 is exactly one half-life: if nothing in scope is within thirty days of the anchor,
#: time has no opinion about this question and the leg abstains.
#:
#: **This is the same guard the other two legs already have**, and it exists for the same
#: reason rather than for tuning. The vector leg abstains on a zero-norm query and the
#: lexical leg on a query with no content terms, because fusion reads *positions* and a
#: leg that ranks weakly still contributes rank 0, rank 1, rank 2 — a ranking assembled
#: from nothing is not a weak ranking, it is a fabricated one, and fusion cannot tell.
#:
#: It is what separates a live store from an archive, which is the case that made it
#: necessary. `add()` stamps a turn with the wall clock, so a store somebody is actually
#: using has its recent turns near the anchor and "what happened recently" is answerable.
#: A benchmark that replays a transcript from three years ago has *every* turn far from
#: now, so the nearest one is no more about the present than the furthest — and measured
#: on LongMemEval, a leg that voted anyway cost 2.4 points of temporal-reasoning R@12 and
#: 4.6 of MRR. See `docs/BENCHMARKS.md`.
MIN_PROXIMITY = 0.5


def anchor_for(valid_at: datetime | None, known_at: datetime | None,
               now: datetime) -> datetime:
    """The instant this leg measures distance from.

    `valid_at` first: it is the world clock, and "what was going on then" is a question
    about the world. `known_at` second, because a belief-time query is asking what the
    store looked like at that moment and the turns near it are the ones it was made of.
    `now` when neither was given — which makes the leg mean "the most recent turns", and
    is the reading `recently`, `lately` and `still` all want.

    >>> from datetime import datetime, timezone
    >>> june = datetime(2026, 6, 1, tzinfo=timezone.utc)
    >>> march = datetime(2026, 3, 1, tzinfo=timezone.utc)
    >>> anchor_for(june, march, march).month
    6
    >>> anchor_for(None, march, june).month
    3
    >>> anchor_for(None, None, june).month
    6
    """
    if valid_at is not None:
        return as_utc(valid_at)
    if known_at is not None:
        return as_utc(known_at)
    return as_utc(now)


def proximity(ts: float, anchor: datetime,
              half_life_days: float = PROXIMITY_HALF_LIFE_DAYS) -> float:
    """How close one turn is to the anchor, in (0, 1]. Absolute, like a cosine.

    Symmetric on purpose: a turn a week before the asked instant and a turn a week after
    it are equally about that time. The alternative — only looking backwards — would make
    the leg a second liveness filter, and the store already has two of those.

    Exponential rather than a window, so there is no cliff for a caller to land on the
    wrong side of and no threshold to tune. A turn exactly at the anchor scores 1.0.

    >>> from datetime import datetime, timezone
    >>> at = datetime(2026, 6, 1, tzinfo=timezone.utc)
    >>> proximity(at.timestamp(), at)
    1.0
    >>> round(proximity(at.timestamp() - 30 * 86400, at), 3)
    0.5
    >>> round(proximity(at.timestamp() + 30 * 86400, at), 3)
    0.5
    """
    days = abs(ts - anchor.timestamp()) / 86400.0
    return 0.5 ** (days / half_life_days)


def rank(hits: Sequence[tuple[str, float]], anchor: datetime,
         half_life_days: float = PROXIMITY_HALF_LIFE_DAYS,
         floor: float = MIN_PROXIMITY) -> list[tuple[str, float]]:
    """`(episode_id, ts)` from the store, as a ranked `(id, closeness)` list — or nothing.

    The store already returned them nearest-first — the ordering and the cap live in one
    statement there, which is design invariant 7 — so this re-sorts nothing and only
    turns the timestamps into scores. Sorting here would silently make the store's `LIMIT`
    mean something different from what came back.

    **Abstains wholesale when even the nearest turn is further than `floor`.** See
    `MIN_PROXIMITY`: a leg with no opinion that votes anyway is worse than one that does
    not run, because fusion reads positions and cannot tell the two apart.

    >>> from datetime import datetime, timezone
    >>> at = datetime(2026, 6, 1, tzinfo=timezone.utc)
    >>> rank([("ep_1", at.timestamp())], at)
    [('ep_1', 1.0)]
    >>> rank([], at)
    []

    Nothing within a half-life of the anchor, so nothing to say:

    >>> year_ago = at.timestamp() - 365 * 86400
    >>> rank([("ep_1", year_ago)], at)
    []

    The floor is on the *nearest* hit, not on each of them: once time has an opinion, the
    tail of the list is the ordering it produced and dropping part of it would leave a
    ranking with holes in it.

    >>> tail = at.timestamp() - 200 * 86400
    >>> [item for item, _ in rank([("ep_1", at.timestamp()), ("ep_2", tail)], at)]
    ['ep_1', 'ep_2']
    """
    scored = [(item_id, proximity(ts, anchor, half_life_days)) for item_id, ts in hits]
    if not scored or scored[0][1] < floor:
        return []
    return scored
