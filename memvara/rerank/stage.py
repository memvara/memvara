"""The reranking stage: reorder a fused candidate list, and say why.

Three properties, each of which is a decision rather than an implementation detail.

**It reorders, it does not rescore.** `Result.score` stays the normalized relevance the
retriever computed, because that is the number `min_score` thresholds against and the
number every integration reads as a 0-1 relevance. Folding a cross-encoder logit into it
would silently change what a floor calibrated by `calibrate_min_score` means, and the
change would be invisible — the same field, the same range, a different meaning. So the
reranker's opinion lands in its own field and the ordering is the only thing it moves.
A consequence worth knowing before it surprises you: a reranked list is **not** in
descending `score` order.

**It is bounded.** Reranking is O(candidates) model calls, which is the whole objection
to it, so the stage sees `top_n` candidates and never the tail. Everything past `top_n`
keeps its fused position and keeps `rerank_score=None` — which is the accurate record:
the reranker did not see it. `None` for "not scored" against `0.0` for "scored zero" is
the distinction `Explanation.rerank_score` was reserved to preserve.

**It is explainable.** Every candidate the reranker saw carries the number that moved
it. A reordering nobody can account for is worse than no reordering at all in a library
whose pitch is that retrieval explains itself; `Explanation.summary()` already renders
the field, so a reranked result prints its own reason.

Ties break to the fused order, because the sort is stable and the fused order is already
deterministic to a content hash. Two candidates a reranker cannot separate therefore
come back in the order two ingests of the same corpus would both produce.
"""

from __future__ import annotations

from typing import Protocol, Sequence, TypeVar, runtime_checkable

from ..types import Explanation
from .base import Reranker


@runtime_checkable
class Rankable(Protocol):
    """What the stage needs from a retrieved item: text to score, and a place to say so.

    `Result` and `EpisodeResult` both satisfy this, which is the point — a search that
    returns claims and raw turns interleaved has to be reranked as one list, or the
    reranker's ordering would only apply within whichever half it was handed.
    """

    explain: Explanation

    @property
    def text(self) -> str: ...


R = TypeVar("R", bound=Rankable)


def rerank(reranker: Reranker, query: str, items: Sequence[R], *, top_n: int) -> list[R]:
    """Reorder the first `top_n` of `items` by `reranker`, leaving the tail in place.

    Mutates each scored item's `explain.rerank_score` — the items are the ones the
    search just built, and copying them would break the identity a caller holding a
    `Result` relies on.

    >>> from dataclasses import dataclass, field
    >>> from memvara.types import Explanation
    >>> @dataclass
    ... class Row:
    ...     text: str
    ...     explain: Explanation = field(default_factory=Explanation)
    >>> class Dogs:
    ...     def score(self, query, documents):
    ...         return [1.0 if "dog" in d else 0.0 for d in documents]
    >>> rows = [Row("the cat sat"), Row("a dog barked"), Row("never scored")]
    >>> [r.text for r in rerank(Dogs(), "dog", rows, top_n=2)]
    ['a dog barked', 'the cat sat', 'never scored']

    The tail was never shown to the model, and its explanation says so rather than
    claiming a zero:

    >>> rows[1].explain.rerank_score, rows[2].explain.rerank_score
    (1.0, None)
    """
    head = list(items[:top_n]) if top_n > 0 else []
    tail = list(items[len(head):])
    if not head:
        # No candidates to reorder. Skipped rather than handed an empty list, so a
        # backend that charges per call is not billed for a query with no results.
        return tail

    scores = reranker.score(query, [item.text for item in head])
    if len(scores) != len(head):
        raise ValueError(
            f"{type(reranker).__name__}.score returned {len(scores)} scores for "
            f"{len(head)} documents. A reranker must return exactly one score per "
            "document, in the order given — a shorter list silently truncates the "
            "candidate set and a longer one silently drops the extras."
        )
    scored = [(item, float(score)) for item, score in zip(head, scores)]
    for item, score in scored:
        item.explain.rerank_score = score
    # Sorted on the scores as a parallel list rather than by reading the field back,
    # which keeps the key a plain `float` instead of the `float | None` the dataclass
    # declares. Stable, so candidates the reranker scored equally keep the fused
    # ranking's own deterministic order rather than an arbitrary one.
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [item for item, _ in scored] + tail
