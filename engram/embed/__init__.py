from .base import CachedEmbedder, Embedder, HashingEmbedder

__all__ = ["Embedder", "HashingEmbedder", "CachedEmbedder"]


def default_embedder(dim: int = 512) -> Embedder:
    """Best embedder available without forcing a dependency.

    Prefers a real sentence-transformers model when installed; otherwise falls back to
    the offline hashing embedder so `Engram()` always constructs.
    """
    try:  # pragma: no cover - depends on optional install
        from .local import LocalEmbedder

        return CachedEmbedder(LocalEmbedder())
    except Exception:
        return CachedEmbedder(HashingEmbedder(dim=dim))
