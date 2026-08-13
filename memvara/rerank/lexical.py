"""A reranker that needs no model, and is honest about what that costs.

Cross-encoders are the reason to have a reranking stage, and they are also a model
download, so a tree that only ships one leaves the stage untested in every default
install and unmeasurable by anyone without the extra. This is the other kind: it reads
the query and the candidate text together — which is the thing neither retrieval leg
does — using nothing but the analyzer the lexical leg already uses.

Two signals, both genuinely absent from the fused ranking it reorders:

* **Coverage.** BM25 sums per-term weights, so a candidate matching one rare query term
  three times can outrank one matching all four terms once. Which of those actually
  answers the question is not a close call, and coverage is the set-valued measure that
  says so. It is not a rescaling of BM25; it deliberately ignores IDF.
* **Proximity.** A candidate where the matched terms sit in one clause is more likely to
  be about their conjunction than one where they are forty tokens apart in an unrelated
  aside. BM25 has no notion of position at all, and a bag-of-n-grams embedding has only
  a smeared one. On conversational turns, which are long and cover several subjects,
  this is the difference between a mention and a statement.

**What it is not.** It is lexical. It will not put "physician" near "doctor", it cannot
tell that a passage answers a question it shares no vocabulary with, and it will not do
what a cross-encoder does. Use `CrossEncoderReranker` for that and pay for the model.
What this one is good for: reordering a candidate list with no download, no network and
no new dependency, and being a control that isolates how much of a measured gain came
from the reranking *stage* rather than from the model plugged into it.

**It has been measured, and it is a control rather than a recommendation.** On LOCOMO it
nets **−0.1 R@12** (62.0 → 61.9): it moves multi-hop, open-domain and temporal up as the
theory above predicts, and pays for all of it on single-hop. A cross-encoder over the
identical candidates is **+4.5**. That contrast is the useful thing this class produces —
it is the row that proves the later gain belongs to the model and not to the stage, which
is exactly the confusion it was written to prevent. Do not reach for it expecting the
cross-encoder's numbers without the cross-encoder's dependency; that trade is not
available. See `docs/ROADMAP.md`.
"""

from __future__ import annotations

from math import inf
from typing import Sequence

from ..retrieve.analyze import analyze, tokenize


class CoverageReranker:
    """Scores a candidate on how much of the query it covers, and how tightly.

    `proximity_weight` is how much of the score the tightness term may move, as a
    fraction. At the default 0.25 a candidate can never overtake one that covers 25%
    more of the query, however tightly packed it is: coverage is the signal and
    proximity only orders within it. At 0.0 this is coverage alone.
    """

    def __init__(self, proximity_weight: float = 0.25) -> None:
        if not 0.0 <= proximity_weight <= 1.0:
            raise ValueError(
                f"proximity_weight must be in [0, 1], got {proximity_weight}: it is the "
                "share of the score the proximity term may move, and outside that range "
                "it either inverts the ranking or swamps coverage entirely."
            )
        self.proximity_weight = proximity_weight
        self.name = f"coverage:{proximity_weight:g}"

    def __repr__(self) -> str:
        return f"<CoverageReranker {self.name}>"

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """Coverage in [0, 1], discounted by how spread out the matches are.

        The candidate covering both query terms outranks the one covering half of
        them, and neither number is a rescaling of what BM25 gave the same pair:

        >>> r = CoverageReranker()
        >>> r.score("dog flat", ["a calm dog suits a small flat", "I saw a dog"])
        [0.875, 0.5]

        Abstains — every score 0.0, so the stage's stable sort leaves the fused order
        exactly as it found it — when the query reduces to no content terms, which is
        the same guard `retrieve.analyze` puts on the lexical leg and for the same
        reason: a ranking built from stopwords is fabricated, not weak.

        >>> r.score("what is it about?", ["anything", "at all"])
        [0.0, 0.0]
        """
        wanted = frozenset(analyze(query).terms)
        if not wanted:
            return [0.0] * len(documents)
        return [self._one(wanted, tokenize(doc)) for doc in documents]

    def _one(self, wanted: frozenset[str], tokens: Sequence[str]) -> float:
        hits = [(i, t) for i, t in enumerate(tokens) if t in wanted]
        found = {t for _, t in hits}
        if not found:
            return 0.0
        coverage = len(found) / len(wanted)
        span = _min_window(hits, len(found))
        # `len(found) / span` is 1.0 when the matched terms are adjacent and falls
        # towards 0 as they scatter. A single matched term has a span of 1 and is
        # trivially tight, which is right: there is no distance to measure.
        tightness = len(found) / span
        return coverage * (1.0 - self.proximity_weight
                           + self.proximity_weight * tightness)


def _min_window(hits: Sequence[tuple[int, str]], distinct: int) -> int:
    """Length of the shortest run of tokens containing every matched term at least once.

    The standard two-pointer sweep, over the matched positions rather than over the
    whole document, so it costs O(matches) rather than O(tokens) — which matters because
    the candidate here is a whole conversation turn and the stage runs it per query.

    >>> _min_window([(0, "a"), (9, "b"), (10, "a")], 2)
    2
    """
    counts: dict[str, int] = {}
    best = inf
    left = 0
    for right, (position, term) in enumerate(hits):
        counts[term] = counts.get(term, 0) + 1
        while len(counts) == distinct:
            start, dropped = hits[left]
            best = min(best, position - start + 1)
            counts[dropped] -= 1
            if not counts[dropped]:
                del counts[dropped]
            left += 1
    # `best` is always assigned: `distinct` is the number of distinct terms in `hits`,
    # so the full sweep necessarily closes the window at least once.
    return int(best)
