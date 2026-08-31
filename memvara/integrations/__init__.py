"""Adapters onto the agent frameworks people already use.

Developers meet a memory layer through the framework they already use, not through its
README, so these exist for distribution. That does not make them decoration: most of
them model memory as **a list of messages** or **a vector store**, and memvara is
neither — it stores resolved bitemporal facts, ends contradicted ones, and can answer
what was believed on a date. Some of that has nowhere to go, and the standard set by
`memvara.compat.mem0` applies here too: where a call has no honest translation it raises
and names the alternative, rather than returning something plausible.

| | surface | what it maps to | what does not survive |
|---|---|---|---|
| **LangChain** | `MemvaraRetriever` | `search()` → `Document` | nothing — `as_of=` even survives |
| | `MemvaraChatMessageHistory` | turns in, transcript out | claims, supersession, provenance |
| **LlamaIndex** | `MemvaraMemoryBlock` | `recall()` / `add()` | scores and provenance (`_aget` returns a string) |
| | `MemvaraRetriever` | `search()` → `NodeWithScore` | nothing |
| **CrewAI** | `MemvaraStorage` | the `StorageBackend` protocol | the query text, unless you pass `embedder=storage.embedder` |
| **LangGraph** | `MemvaraStore` | the `BaseStore` protocol | field names never reach the predicate registry; no provenance; no TTL |

**LangGraph is the one that loses least, and the reason is instructive.** `BaseStore` is
the only framework interface that hands over the *query text* natively, and its
`put(namespace, key, value)` supplies all three parts of a triple — so an item is stored
as one claim **per field**, and changing `city` ends exactly `city` while an unchanged
`food` is recognised as a re-observation. That is per-field contradiction resolution,
which the CrewAI adapter cannot do for a nameable reason: its unit of memory is a
sentence, which contains no subject and no predicate to key on.

Three refusals are worth knowing before you go looking:

* `LangChain.MemvaraChatMessageHistory.clear()` raises. LangChain means "erase"; memvara's
  two candidate meanings are retirement and `purge()`, and they are not variants of one
  operation. `on_clear=` picks.
* `llamaindex.as_vector_store()` raises. A vector store is handed an embedding and never
  the query text; memvara retrieves from text. `MemvaraRetriever` is the answer.
* `llamaindex.as_chat_memory()` raises. `BaseMemory` is the chat buffer; memvara is
  long-term memory *beside* the buffer, which is what a memory block is for.

**No framework is a dependency of memvara.** Importing this package imports none of them
— every class here is a PEP 562 lazy attribute on its module, exactly as
`memvara.llm.OpenAILLM` is, so `import memvara` keeps working with numpy alone and CI
proves it by walking every module with nothing else installed.

    from memvara.integrations.langchain import MemvaraRetriever      # pip install 'memvara[langchain]'
    from memvara.integrations.llamaindex import MemvaraMemoryBlock   # pip install 'memvara[llama-index]'
    from memvara.integrations.crewai import MemvaraStorage           # pip install 'memvara[crewai]'
    from memvara.integrations.langgraph import MemvaraStore           # pip install 'memvara[langgraph]'

`MemvaraStore` and `MemvaraRetriever` are deliberately **not** re-exported from this
package, and both for the same reason: a name that is ambiguous in one namespace is worse
than a longer import. `MemvaraRetriever` exists twice, in LangChain and LlamaIndex. And
`MemvaraStore` (LangGraph's `BaseStore`) sits one letter from `MemvaraStorage` (CrewAI's
`StorageBackend`) — close enough that a typo would import a working class for the wrong
framework, which fails somewhere far from the mistake. Import both from their own module.
"""

from __future__ import annotations

import importlib
from typing import Any

from ._common import IntegrationError

#: Adapter name -> (module, attribute). Also what `__getattr__` dispatches on, so the
#: table and the behaviour cannot disagree.
_ADAPTERS = {
    "MemvaraChatMessageHistory": ("langchain", "MemvaraChatMessageHistory"),
    "MemvaraMemoryBlock": ("llamaindex", "MemvaraMemoryBlock"),
    "MemvaraStorage": ("crewai", "MemvaraStorage"),
}

__all__ = [
    "IntegrationError",
    "MemvaraChatMessageHistory", "MemvaraMemoryBlock", "MemvaraStorage",
]


def __getattr__(name: str) -> Any:
    # `MemvaraRetriever` is deliberately absent: LangChain and LlamaIndex both have one
    # and they are different classes, so a single unqualified name here would resolve to
    # whichever framework happened to be installed. Import those from their own modules.
    target = _ADAPTERS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = target
    return getattr(importlib.import_module(f".{module}", __name__), attribute)
