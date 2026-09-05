"""Model-ranked recall: opt-in per-query model consultation on top of the reranker.

A configured `read_selector` sends the reranked turns to a model that names which ones
actually bear on the question; every other claim and turn keeps its plain order. A real
deployment passes `ModelSelector(llm=...)`, but the example below stands in for it with
a fake `Selector` so the doctest touches no network and needs no API key.

    >>> from contextlib import nullcontext
    >>> from memvara import Memvara
    >>> from memvara.embed import HashingEmbedder
    >>> from memvara.select import Selected

    >>> class FakeSelector:
    ...     '''Keeps whichever candidates mention "trip" — stands in for a real model.'''
    ...     top_n = 40
    ...     def admit(self):
    ...         return nullcontext()
    ...     def select(self, question, candidates, *, asked_on=None, usage=None):
    ...         return [Selected(id=c.id, span=None) for c in candidates
    ...                 if "trip" in c.text]

    >>> mem = Memvara(embedder=HashingEmbedder(dim=8), read_selector=FakeSelector())
    >>> _ = mem.add("Loved the trip to Lisbon last spring", user="alice")
    >>> _ = mem.add("Started a new job on Monday", user="alice")
    >>> result = mem.recall("what did they say about the trip", user="alice",
    ...                     ranked=True, include_episodes=True, with_ids=True)
    >>> result.selection.outcome
    'applied'
    >>> "Loved the trip to Lisbon last spring" in result.text
    True

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
