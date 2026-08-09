"""Rank fusion.

Combining retrievers by *rank* rather than by score is the whole point. BM25 is an
unbounded, corpus-dependent log-scaled number; cosine similarity is bounded in
[-1, 1] and geometrically distributed. Putting them on a common scale requires
normalization constants that are guesswork on day one and drift as the corpus grows,
and the failure mode is silent: a mis-scaled weight quietly deletes one retriever's
contribution and nobody notices because results still come back.

Ranks are already commensurable. RRF's harmonic falloff also gives the property this
system is built around: a top hit in one retriever survives total absence from the
other. That is what lets BM25 rescue an exact token - an error code, a version
string, a surname - that the embedding blurred into its neighbours.
"""

from __future__ import annotations

from typing import Mapping, Sequence


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[tuple[str, float]]],
    *,
    k: int = 60,
    weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Fuse several ranked `(item_id, score)` lists into one score per item.

    Each list must already be ordered best-first by its own retriever. An item at
    0-based rank `r` in list `L` contributes `weights[L] / (k + r + 1)`.

    The per-item scores in the input are deliberately ignored - only position is read.
    They travel along because callers need them for the `Explanation`, not for fusion.

    `k` damps the head of the curve: at the default 60 the gap between rank 0 and
    rank 1 is under 2%, so fusion expresses "both retrievers liked this" more strongly
    than "one retriever liked this slightly more than that". Lowering `k` sharpens
    trust in each retriever's own ordering.

    Returns a dict ordered best-first, ties broken by item id. Items absent from every
    list are simply absent.

    Read that tiebreak as *deterministic*, not as *content-addressed* — it is stable for
    a given set of ids and no further. This function is handed ids and positions and
    nothing else, so an id is the only total order available to it, and a memvara claim
    id is a `uuid4` minted at ingest: two ingests of one corpus tie differently here.
    Nothing downstream inherits that, because `HybridRetriever._rank` and `._episodes`
    both re-sort on a content digest, which is where reproducibility across two stores
    of the same data is actually made. It is stated because this is exported, and a
    caller fusing its own rankings on ids that mean something would be right to expect
    more from the sentence above than it can give.
    """
    if k < 0:
        raise ValueError(f"rrf k must be non-negative, got {k}")

    fused: dict[str, float] = {}
    for name, ranked in rankings.items():
        weight = 1.0 if weights is None else float(weights.get(name, 1.0))
        if weight == 0.0:
            # A retriever weighted to zero must not even reach the output. Adding 0.0
            # would still create the key, which would let a disabled retriever inject
            # candidates that no enabled retriever ever found.
            continue
        seen: set[str] = set()
        for rank, (item_id, _score) in enumerate(ranked):
            if item_id in seen:
                continue  # a repeated id keeps its best (first) position in that list
            seen.add(item_id)
            fused[item_id] = fused.get(item_id, 0.0) + weight / (k + rank + 1)

    return dict(sorted(fused.items(), key=lambda kv: (-kv[1], kv[0])))
