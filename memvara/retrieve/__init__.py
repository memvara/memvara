"""Read path: hybrid retrieval, rank fusion, time-aware rescoring, and traversal."""

from .analyze import STOPWORDS, LexicalQuery, analyze, tokenize
from .calibrate import FloorReport, calibrate_min_score
from .fusion import reciprocal_rank_fusion
from .hybrid import (
    CLAIM,
    EPISODE,
    DegradedRetrievalWarning,
    EpisodeResult,
    HybridRetriever,
    Retrieved,
    kind_of,
)
from .intent import Intent, classify
from .scoring import (
    final_score,
    lexical_relevance,
    normalized_score,
    quality_boost,
    recency_factor,
    relevance,
    vector_relevance,
)
from .spread import rank_paths, seed_keys
from .traverse import HOP_DAMPING, Edge, GraphTraverser, Path

__all__ = [
    "reciprocal_rank_fusion",
    "recency_factor",
    "final_score",
    "quality_boost",
    "normalized_score",
    "relevance",
    "lexical_relevance",
    "vector_relevance",
    "analyze",
    "tokenize",
    "LexicalQuery",
    "STOPWORDS",
    "calibrate_min_score",
    "FloorReport",
    "HybridRetriever",
    "DegradedRetrievalWarning",
    # the graph leg: which entities to walk from, and what a walk is worth
    "seed_keys",
    "rank_paths",
    # query shape, and the weights it selects
    "Intent",
    "classify",
    # retrieved episodes, and how to tell one from a claim
    "EpisodeResult",
    "Retrieved",
    "kind_of",
    "CLAIM",
    "EPISODE",
    # multi-hop traversal
    "GraphTraverser",
    "Path",
    "Edge",
    "HOP_DAMPING",
]
