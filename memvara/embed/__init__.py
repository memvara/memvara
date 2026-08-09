from .base import CachedEmbedder, Embedder, HashingEmbedder
from .fingerprint import (
    EmbedderFingerprint,
    embedder_name,
    fingerprint_of,
    read_fingerprint,
    stored_dim,
    write_fingerprint,
)

__all__ = [
    "Embedder", "HashingEmbedder", "CachedEmbedder", "default_embedder",
    "EmbedderFingerprint", "embedder_name", "fingerprint_of", "read_fingerprint",
    "write_fingerprint", "stored_dim",
]


def default_embedder(dim: int = 512) -> Embedder:
    """Best embedder available without forcing a dependency.

    Prefers a real sentence-transformers model when installed; otherwise falls back to
    the offline hashing embedder so `Memvara()` always constructs.

    Note what this means for an *existing* store: installing `memvara[local-embed]`
    changes what this returns, and the new vectors are incomparable with the old ones.
    That is why `Memvara()` checks the embedder against the store it opens instead of
    discovering the swap on the first read — see `memvara/embed/fingerprint.py` and
    `Memvara.reembed()`.
    """
    try:  # pragma: no cover - depends on optional install
        from .local import LocalEmbedder

        return CachedEmbedder(LocalEmbedder())
    except Exception:
        return CachedEmbedder(HashingEmbedder(dim=dim))
