"""Measuring a store's own relevance floor, because no constant can be one.

`min_score` needs a number, and the obvious move is to ship a good default. That was
tried and it does not survive contact with a second corpus size. Measured with the
shipped `HashingEmbedder`, four fixed gold claims and increasing unrelated filler, the
window between "the weakest correct answer" and "the best wrong answer" moves like
this:

    claims   weakest correct   best wrong   usable floors
         5             0.266        0.152   (0.152, 0.266)
        20             0.391        0.197   (0.197, 0.391)
        50             0.424        0.191   (0.191, 0.424)
       200             0.451        0.232   (0.232, 0.451)
     1,000             0.421        0.268   (0.268, 0.421)

The windows at 5 and at 1,000 do not intersect - 0.266 < 0.268 - so no single constant
is correct at both ends, and one calibrated in the middle silences correct answers on a
small store while admitting junk on a large one. Two independent drifts cause it:

  * **The lexical leg drifts up with the corpus.** IDF grows roughly with log N, so the
    *same* claim matching the *same* query scored BM25 1.31 at 5 claims and 8.88 at
    1,000. Normalizing that away is tempting and is the wrong move: it is the drift
    that keeps correct answers climbing, and removing it flattens them at ~0.27 while
    the noise below keeps rising.
  * **The vector leg's noise ceiling drifts up and cannot be normalized away.** The
    best wrong answer is a maximum over N similarities, so it grows with N by
    construction. With the vector leg alone the weakest correct answer is *dead flat*
    at 0.248 - cosine is a property of a fixed pair of texts - while the best wrong
    answer climbs from 0.162 to 0.298 and overtakes it at around twenty claims.

Relative criteria do not rescue it either. Top-to-median ratio and a median/MAD z-score
both invert on gibberish, where the whole pool sits near zero and the best of the noise
looks like an enormous standout: at 50 claims "quetzalcoatl bandersnatch" scored ratio
24.5 and z 23.5, against 7.2 and 24.4 for the weakest *correct* answer. Top-to-runner-up
overlaps too (1.43 correct against 1.46 wrong at 50 claims). Deriving the floor from the
store's own claims - querying with each claim's text and reading the best non-self hit -
tracks the noise on some corpora and badly overestimates it on redundant ones, where
genuinely related claims are indistinguishable from noise by construction.

Which leaves the honest answer: a floor is a property of *this* store with *this*
embedder at *this* size, it is cheap to measure, and it must be re-measured as the store
grows. `calibrate_min_score` measures it from questions the deployment already knows the
answers to, and reports the evidence alongside the number so it is never quoted without
its provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from ..types import Result

#: A function that runs one query and returns ranked results - `mem.search` with its
#: scope already bound, or `HybridRetriever.search` in a lambda. Taking the search as a
#: callable keeps calibration out of the scope-plumbing business.
Search = Callable[[str], Sequence[Result]]


@dataclass(frozen=True, slots=True)
class FloorReport:
    """A measured `min_score`, with what it costs and what it buys.

    `separable` is the part worth reading first. When it is False no threshold exists
    that both keeps every answerable probe and silences every unanswerable one; `floor`
    is then the best available compromise rather than a clean cut, and the right
    response is usually better retrieval rather than a better threshold.
    """

    floor: float
    separable: bool
    kept: int
    answerable: int
    silenced: int
    unanswerable: int

    def __str__(self) -> str:
        verdict = "clean" if self.separable else "OVERLAPPING"
        return (f"<FloorReport min_score={self.floor:.3f} {verdict} "
                f"keeps {self.kept}/{self.answerable} answerable, "
                f"silences {self.silenced}/{self.unanswerable} unanswerable>")

    __repr__ = __str__


def _top_score(search: Search, query: str) -> float:
    results = search(query)
    # No results at all is the strongest possible "nothing relevant", and scores are
    # non-negative, so 0.0 is the honest reading rather than a missing value.
    return results[0].score if results else 0.0


def calibrate_min_score(
    search: Search,
    *,
    answerable: Sequence[str],
    unanswerable: Sequence[str],
    margin: float = 0.5,
) -> FloorReport:
    """Measure the `min_score` that separates answerable questions from unanswerable ones.

    `answerable` are questions this store *should* answer; `unanswerable` are questions
    it should not - plausible questions about things it does not contain, not gibberish.
    Gibberish is the wrong probe: it scores far below a real question on an absent
    topic, so calibrating on it produces a floor that admits exactly the confident
    wrong answers you were trying to stop.

    Both classes are required. A floor derived from one of them is a guess with a
    number attached: the answerable set alone gives no idea what noise looks like here,
    and the unanswerable set alone gives no idea what it costs to exclude it.

    `margin` places the floor between the best wrong answer (0.0) and the weakest
    correct one (1.0). The midpoint default is the least-assumption choice; move it
    down if a missed memory is worse than a spurious one, and up if the reverse. When
    the two classes overlap, `margin` does not apply - the floor that maximizes
    (kept + silenced) is chosen instead, preferring the lower one on a tie so that
    recall is favoured over silence.

    >>> from engram.types import Claim, Explanation, Result
    >>> def fake(query):
    ...     score = {"where do they live": 0.44, "who is Mira": 0.51,
    ...              "capital of France": 0.19, "the CFO's phone number": 0.23}[query]
    ...     claim = Claim(subject="user", predicate="lives_in", object="Lisbon")
    ...     return [Result(claim=claim, score=score, explain=Explanation())]
    >>> report = calibrate_min_score(
    ...     fake,
    ...     answerable=["where do they live", "who is Mira"],
    ...     unanswerable=["capital of France", "the CFO's phone number"])
    >>> round(report.floor, 3)
    0.335
    >>> report.separable
    True
    """
    if not answerable or not unanswerable:
        raise ValueError(
            "calibration needs both answerable and unanswerable probes; a floor "
            "measured against one class only is a guess with a number attached"
        )

    hits = [_top_score(search, q) for q in answerable]
    misses = [_top_score(search, q) for q in unanswerable]
    weakest_hit, best_miss = min(hits), max(misses)

    if best_miss < weakest_hit:
        floor = best_miss + margin * (weakest_hit - best_miss)
    else:
        # Overlapping. Score every observed value as a candidate cut and take the one
        # that gets the most probes right, lowest first - a floor that silences a real
        # memory is the more expensive error, since the caller cannot tell it happened.
        floor = min(
            set(hits + misses),
            key=lambda c: (-(sum(h >= c for h in hits) + sum(m < c for m in misses)), c),
        )

    return FloorReport(
        floor=floor,
        separable=best_miss < weakest_hit,
        kept=sum(h >= floor for h in hits),
        answerable=len(hits),
        silenced=sum(m < floor for m in misses),
        unanswerable=len(misses),
    )
