"""Read path: hybrid retrieval, rank fusion, and time-aware rescoring."""

from .analyze import STOPWORDS, LexicalQuery, analyze, tokenize
from .calibrate import FloorReport, calibrate_min_score
from .fusion import reciprocal_rank_fusion
from .hybrid import CLAIM, EPISODE, EpisodeResult, HybridRetriever, Retrieved, kind_of
from .scoring import (
    final_score,
    lexical_relevance,
    normalized_score,
    quality_boost,
    recency_factor,
    relevance,
    vector_relevance,
)

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
    # retrieved episodes, and how to tell one from a claim
    "EpisodeResult",
    "Retrieved",
    "kind_of",
    "CLAIM",
    "EPISODE",
]
