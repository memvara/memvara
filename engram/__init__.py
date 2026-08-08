"""Engram — bitemporal memory for AI agents.

What it does differently from mem0 and friends:

* **Facts are structured and bitemporal.** A memory is a `Claim` — a (subject, predicate,
  object) triple carrying both when it was true in the world and when we believed it.
  That is what makes `search(..., as_of=T)` answer "what did we think in March?"
* **Contradictions resolve deterministically.** A predicate schema says whether a
  relation is single-valued, so a conflict is an indexed lookup on (subject, predicate)
  rather than a top-k similarity search plus an LLM adjudication that may miss it.
* **The write path avoids the LLM.** Hash dedupe, near-duplicate detection, a salience
  gate, and rule-based extraction run first; the model is consulted only for turns that
  survive all of them, batched. `WriteReceipt.llm_calls` reports the cost every time.
  With no `llm=` there is no fourth tier at all, so turns the rules do not recognise are
  not stored — `Engram()` warns once about that, and `WriteReceipt.unextracted` counts
  it per write.
* **Retrieval is hybrid and time-aware.** BM25 fused with vector search, reranked by
  recency decay tuned per predicate, and every result explains why it surfaced.
* **Nothing is silently lost.** Superseded facts are retired, never deleted, and every
  claim traces back to the source turns via `why()`.

    from engram import Engram

    mem = Engram("memory.db", user="alice")
    mem.add("I live in Berlin and work at Acme")
    mem.add("Actually I moved to Lisbon last month")

    mem.search("where do they live?")   # -> Lisbon
    mem.history("user", "lives_in")     # -> Berlin (retired), Lisbon (current)
"""

from .consolidate import Consolidator
from .core import (
    DegradedExtractionWarning,
    EmbedderChangedWarning,
    EmbedderMismatchError,
    Engram,
    ScopedEngram,
)
from .embed import (
    CachedEmbedder,
    Embedder,
    EmbedderFingerprint,
    HashingEmbedder,
    default_embedder,
)
from .llm import LLM, NullLLM
from .retrieve import FloorReport, HybridRetriever, calibrate_min_score
from .schema import Cardinality, PredicateRegistry, PredicateSpec, Volatility
from .store import SQLiteStore, Store
from .types import (
    Claim,
    Derivation,
    Episode,
    Explanation,
    MemoryType,
    Provenance,
    Result,
    Scope,
    WriteReceipt,
    utcnow,
)
from .write import FastExtractor, Reconciler, SalienceGate, WritePipeline

__version__ = "0.1.0"

__all__ = [
    "Engram", "ScopedEngram",
    # data model
    "Claim", "Episode", "Scope", "Result", "Explanation", "Provenance",
    "WriteReceipt", "MemoryType", "Derivation", "utcnow",
    # schema
    "PredicateRegistry", "PredicateSpec", "Cardinality", "Volatility",
    # pluggable backends
    "Store", "SQLiteStore",
    "Embedder", "HashingEmbedder", "CachedEmbedder", "default_embedder",
    "EmbedderFingerprint",
    "LLM", "NullLLM",
    # diagnostics: importable so they can be filtered or caught by category
    "DegradedExtractionWarning", "EmbedderChangedWarning", "EmbedderMismatchError",
    # subsystems
    "WritePipeline", "SalienceGate", "FastExtractor", "Reconciler",
    "HybridRetriever", "Consolidator",
    # relevance floors are measured per deployment, never assumed
    "calibrate_min_score", "FloorReport",
    "__version__",
]


def __getattr__(name: str):
    # Kept out of the eager imports so `import engram` works without the anthropic SDK.
    if name == "AnthropicLLM":
        from .llm.anthropic import AnthropicLLM

        return AnthropicLLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
