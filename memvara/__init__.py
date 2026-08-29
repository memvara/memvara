"""Memvara — bitemporal memory for AI agents.

What it does differently from mem0 and friends:

* **Facts are structured and bitemporal.** A memory is a `Claim` — a (subject, predicate,
  object) triple carrying both when it was true in the world and when we believed it.
  Two clocks, asked independently: `known_at=T` answers "what did we think in March?",
  `valid_at=T` answers "what do we think *today* about how March was?", and `as_of=T`
  moves both at once.
* **Contradictions resolve deterministically.** A predicate schema says whether a
  relation is single-valued, so a conflict is an indexed lookup on (subject, predicate)
  rather than a top-k similarity search plus an LLM adjudication.
* **The write path avoids the LLM.** Hash dedupe, a salience gate, and rule-based
  extraction run first; the model is consulted only for turns that need it.
* **Retrieval is hybrid and time-aware.** BM25 fused with vector search, reranked by
  recency decay tuned per predicate, and every result explains why it surfaced.
* **Nothing is silently lost.** Superseded facts are retired, never deleted, and every
  claim traces back to its source turns via `why()`.

The `memvara.code` package builds on those same primitives for change-aware code memory.
It indexes Python symbols deterministically, keeps path as metadata rather than identity,
and stores semantic context as ordinary bitemporal Memvara claims.
"""

from .aio import AsyncMemvara, AsyncScopedMemvara
from .consolidate import Consolidator
from .entities import EntityRegistry, EntityResolution, EntitySpec, entity_key
from .core import (
    DegradedExtractionWarning,
    EmbedderChangedWarning,
    EmbedderMismatchError,
    ErasureIncomplete,
    Memvara,
    ScopedMemvara,
)
from .code import CodeIndex, CodeMemory, CodeSnapshot, ContextRecord, Symbol, SymbolChange, SymbolKind
from .embed import (
    CachedEmbedder,
    Embedder,
    EmbedderFingerprint,
    HashingEmbedder,
    default_embedder,
)
from .llm import LLM, NullLLM
from .retrieve import (
    HOP_DAMPING,
    Edge,
    EpisodeResult,
    FloorReport,
    GraphTraverser,
    HybridRetriever,
    Path,
    Retrieved,
    calibrate_min_score,
)
from .schema import Cardinality, PredicateRegistry, PredicateSpec, Volatility
from .store import SQLiteStore, Store
from .types import (
    Accumulation,
    Answer,
    Claim,
    Closure,
    Collapse,
    Delta,
    Derivation,
    Dispute,
    Retype,
    Episode,
    ErasureProof,
    Explanation,
    MemoryType,
    Provenance,
    Reading,
    RecallResult,
    Result,
    Scope,
    WriteReceipt,
    time_axes,
    utcnow,
)
from .telemetry import MemoryRecorder, NullRecorder, Recorder
from .write import (
    FastExtractor,
    Reconciler,
    SalienceGate,
    UnembeddableTextWarning,
    WritePipeline,
)
from .write.reconcile import SplitReport, backfill_entities, split_entity

__version__ = "0.8.1"

__all__ = [
    "Memvara", "ScopedMemvara", "AsyncMemvara", "AsyncScopedMemvara",
    "CodeIndex", "CodeMemory", "CodeSnapshot", "ContextRecord", "Symbol", "SymbolChange", "SymbolKind",
    "Claim", "Episode", "Scope", "Result", "Explanation", "Provenance",
    "WriteReceipt", "MemoryType", "Derivation", "utcnow", "time_axes",
    "Accumulation", "Dispute", "Collapse", "Retype", "Closure", "RecallResult", "Delta",
    "Answer", "Reading",
    "PredicateRegistry", "PredicateSpec", "Cardinality", "Volatility",
    "Store", "SQLiteStore",
    "Embedder", "HashingEmbedder", "CachedEmbedder", "default_embedder", "EmbedderFingerprint",
    "LLM", "NullLLM", "AnthropicLLM", "OpenAILLM",
    "DegradedExtractionWarning", "EmbedderChangedWarning", "EmbedderMismatchError",
    "ErasureIncomplete", "ErasureProof",
    "WritePipeline", "SalienceGate", "FastExtractor", "Reconciler", "UnembeddableTextWarning",
    "HybridRetriever", "Retrieved", "EpisodeResult", "GraphTraverser", "Path", "Edge", "HOP_DAMPING",
    "EntityRegistry", "EntityResolution", "EntitySpec", "entity_key", "backfill_entities", "split_entity", "SplitReport",
    "Recorder", "NullRecorder", "MemoryRecorder", "Consolidator", "calibrate_min_score", "FloorReport",
    "__version__",
]


def __getattr__(name: str):
    # Kept out of the eager imports so `import memvara` works with neither hosted SDK.
    if name == "AnthropicLLM":
        from .llm.anthropic import AnthropicLLM
        return AnthropicLLM
    if name == "OpenAILLM":
        from .llm.openai import OpenAILLM
        return OpenAILLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
