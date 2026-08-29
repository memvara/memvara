"""Memvara — bitemporal memory for AI agents.

What it does differently from mem0 and friends:

* **Facts are structured and bitemporal.** A memory is a `Claim` — a (subject, predicate,
  object) triple carrying both when it was true in the world and when we believed it.
  Two clocks, asked independently: `known_at=T` answers "what did we think in March?",
  `valid_at=T` answers "what do we think *today* about how March was?", and `as_of=T`
  moves both at once. The middle one is why the axes are separate — a correction that
  arrives in August about June cannot be seen from a belief clock rewound to June.
* **Contradictions resolve deterministically.** A predicate schema says whether a
  relation is single-valued, so a conflict is an indexed lookup on (subject, predicate)
  rather than a top-k similarity search plus an LLM adjudication that may miss it.
* **The write path avoids the LLM.** Hash dedupe, near-duplicate detection, a salience
  gate, and rule-based extraction run first; the model is consulted only for turns that
  survive all of them, batched. `WriteReceipt.llm_calls` reports the cost every time.
  With no `llm=` there is no fourth tier at all, so turns the rules do not recognise are
  not stored — `Memvara()` warns once about that, and `WriteReceipt.unextracted` counts
  it per write.
* **Retrieval is hybrid and time-aware.** BM25 fused with vector search, reranked by
  recency decay tuned per predicate, and every result explains why it surfaced.
* **Nothing is silently lost.** Superseded facts are retired, never deleted, and every
  claim traces back to the source turns via `why()`.

    from memvara import Memvara

    mem = Memvara("memory.db", user="alice")
    mem.add("I live in Berlin and work at Acme")
    mem.add("Actually I moved to Lisbon last month")

    mem.search("where do they live?")   # -> Lisbon
    mem.history("user", "lives_in")     # -> Berlin (retired), Lisbon (current)
"""

from .aio import AsyncMemvara, AsyncScopedMemvara
from .consolidate import Consolidator
from .entities import EntityRegistry, EntityResolution, EntitySpec, entity_key
from .redact import PatternRedactor, Redactor
from .core import (
    DegradedExtractionWarning,
    EmbedderChangedWarning,
    EmbedderMismatchError,
    ErasureIncomplete,
    Memvara,
    ScopedMemvara,
)
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
    # data model
    "Claim", "Episode", "Scope", "Result", "Explanation", "Provenance",
    "WriteReceipt", "MemoryType", "Derivation", "utcnow", "time_axes",
    # One entry of `WriteReceipt.accumulated`: a value written beside values already live
    # in the same slot, under a predicate whose cardinality nobody has declared. Exported
    # because a caller reading that field needs to be able to name its element type.
    "Accumulation",
    # The element types of `WriteReceipt.disputed` and `.collapsed`, exported for the
    # reason above and not a new one: a caller reading either field has to be able to
    # name what is in it, and this package ships `py.typed`, so the caller who cannot is
    # the one whose type checker is doing its job.
    "Dispute", "Collapse", "Retype",
    # Which clock a write stops when it ends a claim: "ended" (the world changed) or
    # "retired" (the record was wrong). Exported because it is in `Reconciler.apply`'s
    # signature and in four facade methods, so a typed caller needs to be able to name it.
    "Closure",
    # What `recall(with_ids=True)` and `since()` hand back. Exported for the same reason
    # `Closure` is: both are return types on four facade methods each, so a caller who
    # annotates anything cannot name them otherwise.
    "RecallResult", "Delta",
    # What `ask()` hands back, and the per-slot element inside it. Exported for the same
    # reason: both are return types on four facade methods, and a `Reading` is the object
    # a caller reads `then` and `stated` off when they need the structure rather than the
    # rendered narrative.
    "Answer", "Reading",
    # schema
    "PredicateRegistry", "PredicateSpec", "Cardinality", "Volatility",
    # pluggable backends
    "Store", "SQLiteStore",
    "Embedder", "HashingEmbedder", "CachedEmbedder", "default_embedder",
    "EmbedderFingerprint",
    "LLM", "NullLLM", "AnthropicLLM", "OpenAILLM",
    # diagnostics: importable so they can be filtered or caught by category
    "DegradedExtractionWarning", "EmbedderChangedWarning", "EmbedderMismatchError",
    # erasure, and the evidence for it
    "ErasureIncomplete", "ErasureProof",
    # subsystems
    "WritePipeline", "SalienceGate", "FastExtractor", "Reconciler",
    "UnembeddableTextWarning",
    "HybridRetriever", "Retrieved", "EpisodeResult",
    # multi-hop traversal. `Path` is a chain of claims, not a filesystem path.
    "GraphTraverser", "Path", "Edge", "HOP_DAMPING",
    "EntityRegistry", "EntityResolution", "EntitySpec", "entity_key",
    "backfill_entities",
    # The other direction: `learn_alias` says two spellings are one thing, and this says
    # one spelling has been two. Exported with its report type for `backfill_entities`'
    # reason — an operator reaching for one reaches for the other, and a caller reading
    # the result has to be able to name it.
    "split_entity", "SplitReport",
    "Recorder", "NullRecorder", "MemoryRecorder",
    "Redactor", "PatternRedactor", "Consolidator",
    # relevance floors are measured per deployment, never assumed
    "calibrate_min_score", "FloorReport",
    # The hosted client and the errors its calls raise. Reached lazily below, so
    # `import memvara` still works without httpx installed.
    "RemoteMemvara", "AsyncRemoteMemvara",
    "RemoteError", "AuthError", "ScopeError", "NotFound", "Conflict", "QuotaExhausted",
    "RateLimited", "LegalHold", "ReadOnly", "InvalidRequest", "ServerError",
    "__version__",
]


def __getattr__(name: str):
    # Kept out of the eager imports so `import memvara` works with neither hosted SDK,
    # and with no httpx: the remote client reaches a hosted deployment over HTTP, and
    # `import memvara` must still be a two-package install.
    if name == "AnthropicLLM":
        from .llm.anthropic import AnthropicLLM

        return AnthropicLLM
    if name == "OpenAILLM":
        from .llm.openai import OpenAILLM

        return OpenAILLM
    if name == "RemoteMemvara":
        from .remote.api import RemoteMemvara
        return RemoteMemvara
    if name == "AsyncRemoteMemvara":
        from .remote.aio import AsyncRemoteMemvara
        return AsyncRemoteMemvara
    if name in ("RemoteError", "AuthError", "ScopeError", "NotFound", "Conflict",
                "QuotaExhausted", "RateLimited", "LegalHold", "ReadOnly",
                "InvalidRequest", "ServerError"):
        from . import remote
        return getattr(remote, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
