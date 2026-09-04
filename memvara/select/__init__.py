"""Model-ranked recall: opt-in per-query model consultation on top of the reranker.

    from memvara import Memvara
    from memvara.select import ModelSelector

    mem = Memvara("memory.db", read_selector=ModelSelector(llm=openai_llm, top_n=40))
    mem.recall("what did they say about the trip", ranked=True)

A default install imports none of this: `Memvara(read_selector=None)` is the default,
and no `read_*` option calls a model unless it is explicitly configured — see
`docs/INTERNALS.md`, invariant 1. `ModelSelector`, the one real implementation here,
needs the `openai` or `anthropic` extra; naming it must not import either SDK, the same
promise `memvara.rerank.CrossEncoderReranker` makes about `sentence-transformers`, and
`tests/test_rerank.py`'s subprocess assertion covers this package too.
"""

from __future__ import annotations

from typing import Any

from .base import Candidate, Selected, Selection, Selector, SelectorBusy, SelectorRefused

__all__ = [
    "Candidate", "Selected", "Selection", "Selector", "SelectorBusy", "SelectorRefused",
    "ModelSelector",
]


def __getattr__(name: str) -> Any:
    # PEP 562 lazy attribute, the reason `memvara.rerank` defers `CrossEncoderReranker`:
    # naming `ModelSelector` must not cost an import a caller who never asked for it did
    # not agree to pay.
    if name == "ModelSelector":
        from .model import ModelSelector

        return ModelSelector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
