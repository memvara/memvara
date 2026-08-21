"""The graph leg of retrieval: what to walk from, and what a walk is worth.

`GraphTraverser` has been able to answer "who does Alice's manager report to" since it
landed, and `bench/multihop.py` measured what that is worth — **34.7% against 4.7%** for
search-then-search at three hops. Nothing on the read path called it. `search()` fused two
legs, both of them lookups, and a question whose evidence is two rows with a join between
them was answered by whichever row happened to embed closest.

This module is the join between the two, and it is deliberately thin, because the
expensive design decision is the one it *declines* to make.

**Seeds come from the other legs, not from the query.** Zep's φ_bfs is seeded from the
top-n of φ_cos and φ_bm25, and the reason is not economy — it is that entity extraction
over free-text queries is a second extractor on the read path, with its own failure modes,
its own vocabulary and (in every implementation that works well) its own model call. The
top of the fused list already names entities, exactly, in the form the graph is keyed on:
`Claim.subject_key` and `Claim.object_key` are folded identities that the *write* path
resolved through the entity registry. Reading them off a row costs nothing and cannot
disagree with the store, which is more than a query-side extractor could promise.

**A path's score is already an absolute relevance.** `GraphTraverser._extend` composes
edge strengths multiplicatively with a damping factor, both bounded by 1.0, so a path
scores in [0, 1] and a longer path can never outscore its own prefix. That is the same
shape `vector_relevance` produces, so the graph leg needs no calibration constant of its
own to sit beside the other two in `scoring.relevance`.

Everything here is order-stable on content rather than on ids. Fusion breaks ties on the
item id, and a memvara claim id is a `uuid4` minted at ingest — so seeding straight off
the fused order would make *which entities get walked* a property of which ingest ran.
Both functions below re-sort on `value_key` before anything is truncated.
"""

from __future__ import annotations

from typing import Sequence

from ..types import Claim
from .traverse import Path


def seed_keys(ranked: Sequence[tuple[Claim, float]], limit: int) -> tuple[str, ...]:
    """The entities to walk from: the folded ends of the best-scoring claims.

    `ranked` is `(claim, fusion score)` in any order; the sort happens here. `limit`
    bounds the number of **keys**, not the number of claims, because the key list is what
    reaches `Store.adjacent` and the frontier width is what a walk costs. One claim
    contributes up to two keys, so a limit of 5 is somewhere between two and five claims
    deep depending on how much the head of the list overlaps.

    Both ends of each claim, and in that order. A question answered two hops out reaches
    its answer through the object as often as through the subject — "where is Alice's
    employer headquartered" starts at `alice` and turns at `acme` — and the fused list
    that named the first row named both of them.

    >>> from memvara.types import Claim
    >>> def c(s, o):
    ...     return Claim(subject=s, predicate="works_at", object=o)
    >>> seed_keys([(c("Alice", "Acme"), 0.02), (c("Bob", "Acme"), 0.03)], 3)
    ('bob', 'acme', 'alice')
    >>> seed_keys([(c("Alice", ""), 0.02)], 5)
    ('alice',)
    """
    # `value_key` and not `id`: the id tier of the fused order is a `uuid4`, so two
    # stores holding identical data would seed identical queries from different entities
    # and the graph leg alone would make retrieval unreproducible across ingests.
    ordered = sorted(ranked, key=lambda pair: (-pair[1], pair[0].value_key, pair[0].id))
    out: dict[str, None] = {}
    for claim, _score in ordered:
        for key in (claim.subject_key, claim.object_key):
            # An empty key is how a retraction spells "clear the slot", so it is a real
            # stored value rather than a missing one. `GraphTraverser` drops it too; it is
            # dropped here as well so the seed list never *asks* for the hub that would
            # connect every retraction in the tenant to every other.
            if key and key not in out:
                out[key] = None
                if len(out) >= limit:
                    return tuple(out)
    return tuple(out)


def rank_paths(paths: Sequence[Path]) -> list[tuple[str, float]]:
    """One ranked `(claim_id, score)` list over every claim any path passes through.

    The third ranking, in the shape `reciprocal_rank_fusion` and `_positions` read. A
    claim reached by more than one path takes the **best** path's score, not the sum:
    scores here are relevances rather than evidence counts, and a hub that sits on nine
    weak chains is not thereby a better answer than one lying on a single strong one.

    Every claim on a path is emitted, not only the far end. The middle of a chain is what
    makes the answer checkable — a caller handed "Acme is in Tallinn" with no "Alice works
    at Acme" beside it has an assertion rather than a derivation — and `Explanation`
    carries the rank that says which leg found it.

    >>> from memvara.retrieve.traverse import Edge, Path
    >>> from memvara.types import Claim
    >>> a = Claim(subject="Alice", predicate="works_at", object="Acme")
    >>> b = Claim(subject="Acme", predicate="based_in", object="Tallinn")
    >>> one = Path(nodes=("alice", "acme"), edges=(Edge(a, False, 1.0),), score=1.0)
    >>> two = Path(nodes=("alice", "acme", "tallinn"),
    ...            edges=(Edge(a, False, 1.0), Edge(b, False, 1.0)), score=0.75)
    >>> [round(score, 2) for _cid, score in rank_paths([two, one])]
    [1.0, 0.75]
    """
    best: dict[str, float] = {}
    content: dict[str, str] = {}
    for path in paths:
        for claim in path.claims:
            if path.score > best.get(claim.id, -1.0):
                best[claim.id] = path.score
            content.setdefault(claim.id, claim.value_key)
    return sorted(best.items(), key=lambda kv: (-kv[1], content[kv[0]], kv[0]))
