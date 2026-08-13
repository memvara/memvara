"""Reranker protocol, plus the no-op case.

A reranker is the one component in this library that reads the query and a candidate
*together*. Both retrieval legs score a document without ever seeing the other side:
BM25 sums per-term weights, cosine compares two independently produced vectors. That
independence is what makes them fast enough to run over a whole scope, and it is also
their ceiling — neither can tell that "the flat that only suits a calm dog" answers a
question about pets while "I walked past a dog" does not.

Reranking buys that at a price the rest of the library refuses to pay by default: a
model, downloaded, resident in memory, and run once per candidate per query. So the
shape here is the shape `memvara.llm` and `memvara.embed` already use — a protocol two
lines wide, a null implementation, and every real backend behind an extra that the
default install does not have. `HybridRetriever(reranker=None)` is the default, and
`None` means the stage does not exist rather than that it runs and does nothing.

The protocol asks for exactly one method because that is exactly what the stage calls.
A `name` is deliberately *not* required, for the same reason `Embedder` does not require
one: every implementation in this package carries one, and requiring it would break the
one-method adapters people actually write around a hosted rerank endpoint.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class Reranker(Protocol):
    def score(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        """Relevance of each document to `query`, in the order given, higher is better.

        Exactly `len(documents)` values, and the scale is the backend's own: nothing
        downstream compares two rerankers' numbers or thresholds on them, so a
        cross-encoder's unbounded logit and a coverage fraction in [0, 1] are both
        valid. What the stage requires is that the ordering be meaningful and that the
        count match, and it raises rather than guesses when it does not.

        Must be deterministic for a given (query, documents) pair. Retrieval in this
        library is reproducible by construction — ties break on a content hash so two
        ingests of one corpus rank identically — and a sampling reranker would spend
        that property for nothing.
        """
        ...


class NullReranker:
    """Scores everything 0.0, which leaves the fused order untouched.

    Not the default — the default is no reranker at all. This exists for the case the
    `Explanation.rerank_score` field was designed around: a reranker that ran and
    scored zero is a different fact from no reranker having run, and the first is
    `0.0` while the second stays `None`. It is also the honest way to measure what the
    stage itself costs, separately from what a model costs.
    """

    name = "null"

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """
        >>> NullReranker().score("anything", ["a", "b"])
        [0.0, 0.0]
        """
        return [0.0] * len(documents)
