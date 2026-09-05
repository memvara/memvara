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
  Lexical, and says so; see its module docstring for what that rules out. **A control,
  not a recommendation** — measured at −0.1 R@12 on LOCOMO.
* `CrossEncoderReranker` — a real cross-encoder behind `pip install 'memvara[rerank]'`.
  The reason to have a reranking stage, and the reason it cannot be the default.

**Which one to use: the cross-encoder, or none.** On LOCOMO's 1,531 evidence-labelled
questions, over an identical candidate list, `ms-marco-MiniLM-L-6-v2` at `top_n=20` moves
R@12 from 62.0 to 66.5 and R@1 from 30.5 to 44.9 — while `CoverageReranker` nets −0.1.
The gap between those two rows is the whole point of this package: the stage is cheap and
the model is where the accuracy lives. R@20 is unchanged either way, because reranking
the top 20 can only reorder within it — so the entire effect is evidence moving *upward*,
which is what a caller with a token budget is paying for.

What keeps the default `None` is cost, not accuracy: a cross-encoder is roughly 84 ms per
query at `top_n=20` against a ~3 ms search, so with reranking on the reranker *is* the
query latency. `docs/ROADMAP.md` carries the full table, the per-category breakdown, and
the commands to reproduce it.

A reranker orders candidates; it never drops one, and it never sees the question and the
candidate weighed against each other by a model that can be asked to *judge*. That is a
different stage, opt-in a level further out — `memvara.select`, a `read_selector` that
takes a `ranked=True` read's reranked turns and names the ones that actually bear on the
question. It builds on this package (the judged configuration reranks first, at
`rerank_top_n=200`, then selects) rather than replacing it: the two answer different
questions, "which order" and "which ones", and the second costs a real chat call where
this package's most expensive backend still costs none.
"""

from __future__ import annotations

from typing import Any

from .base import NullReranker, Reranker
from .stage import Rankable, rerank

__all__ = [
    "Reranker", "NullReranker", "Rankable", "rerank",
    "CoverageReranker", "CrossEncoderReranker", "DEFAULT_MODEL",
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
    if name == "DEFAULT_MODEL":
        # Lazy for consistency rather than necessity — `cross` defers its own SDK import
        # into `_load`, so reading the constant costs nothing either way. Exported so a
        # caller naming a model can say which one it is *not* using without hardcoding
        # the string a second time.
        from .cross import DEFAULT_MODEL

        return DEFAULT_MODEL
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
