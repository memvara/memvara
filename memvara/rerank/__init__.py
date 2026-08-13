"""Reranking. Opt-in at every level, and "off" means the stage does not exist.

    from memvara import Memvara
    from memvara.rerank import CoverageReranker

    mem = Memvara("memory.db", read_reranker=CoverageReranker(), read_rerank_top_n=20)

`HybridRetriever` takes `reranker=None` by default and `Memvara` passes nothing, so a
default install runs precisely the code it ran before this package existed: no import of
a backend, no model, no network, no measurable cost. That is a headline property of the
library rather than a nicety, and `tests/test_rerank.py` asserts it in a subprocess.

Three implementations, in increasing order of what they cost you:

* `NullReranker` — scores everything 0.0. For proving the wiring, and for measuring what
  the stage itself costs separately from what a model costs.
* `CoverageReranker` — query-term coverage and proximity, no dependency, no download.
  Lexical, and says so; see its module docstring for what that rules out.
* `CrossEncoderReranker` — a real cross-encoder behind `pip install 'memvara[rerank]'`.
  The reason to have a reranking stage, and the reason it cannot be the default.
"""

from __future__ import annotations

from typing import Any

from .base import NullReranker, Reranker
from .stage import Rankable, rerank

__all__ = [
    "Reranker", "NullReranker", "Rankable", "rerank",
    "CoverageReranker", "CrossEncoderReranker",
]


def __getattr__(name: str) -> Any:
    # PEP 562 lazy attributes, same as `memvara.llm`. `CrossEncoderReranker` is here for
    # the usual reason — naming it must not import sentence-transformers, let alone
    # torch. `CoverageReranker` is here for a different one: it imports the query
    # analyzer out of `memvara.retrieve`, and `memvara.retrieve.hybrid` imports this
    # package, so an eager import would close a cycle. Deferring it to first use means
    # `memvara.retrieve` has always finished importing by the time it runs.
    if name == "CoverageReranker":
        from .lexical import CoverageReranker

        return CoverageReranker
    if name == "CrossEncoderReranker":
        from .cross import CrossEncoderReranker

        return CrossEncoderReranker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
