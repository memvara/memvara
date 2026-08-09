"""Adapters onto the agent frameworks people already use.

Developers meet a memory layer through LangChain, LlamaIndex or CrewAI, not through its
README, so these exist for distribution. That does not make them decoration: each of the
three models memory as **a list of messages** or **a vector store**, and memvara is
neither — it stores resolved bitemporal facts, retires contradictions, and can answer
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
