"""Embedder protocol plus a dependency-free fallback.

The fallback matters more than it looks. A memory library whose import-to-first-write
path requires downloading a 400MB model, or an API key, is a library people evaluate
by reading the README instead of running it. `HashingEmbedder` makes the whole system
runnable and testable offline in milliseconds; it is a lexical approximation, not a
semantic model, and the code says so where it counts.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

_WORD = re.compile(r"[a-z0-9']+")


def _name_of(embedder: object) -> str:
    """Stable identity string for any embedder, including third-party ones.

    Degrades rather than demands: an embedder that declares a `name` gets to define its
    own identity, a wrapper delegates to what it wraps, and anything else falls back to
    its class name. The fallback is weaker (two configurations of one class look
    identical) but it is never wrong about the case that matters, which is two
    *different* classes writing into one store.
    """
    name = getattr(embedder, "name", None)
    if isinstance(name, str) and name:
        return name
    inner = getattr(embedder, "inner", None)
    if inner is not None:
        return _name_of(inner)
    return type(embedder).__name__


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dim) float32 array. Rows need not be normalized."""
        ...

    # A `name` is deliberately *not* part of this protocol even though everything in
    # this package carries one, because vectors written by two different models are not
    # comparable and a store needs to know which one wrote it (see `fingerprint.py`).
    # Requiring it here would break every third-party embedder that already satisfies
    # the protocol — including the two-line lambda wrappers people actually write — for
    # a check that degrades gracefully to the class name instead.


class HashingEmbedder:
    """Deterministic character-n-gram + word hashing into a fixed-dim space.

    Properties that make it a good default: no dependencies, no network, no model
    download, identical vectors across processes and machines, and fast enough that
    tests do not need a mock. Character n-grams give it some robustness to morphology
    and typos ("lisbon"/"lisbonne"), which pure word hashing lacks.

    What it is not: a semantic model. It will not put "physician" near "doctor". Swap in
    a real embedder for production recall; the interface is one method.
    """

    def __init__(self, dim: int = 512, ngram: tuple[int, int] = (3, 5)) -> None:
        self.dim = dim
        self.ngram = ngram

    @property
    def name(self) -> str:
        # The dimension is part of the identity: two HashingEmbedders with different
        # dims produce incomparable vectors, and the n-gram range changes the feature
        # space too, so both belong in the fingerprint a store is checked against.
        return f"hashing:{self.dim}:{self.ngram[0]}-{self.ngram[1]}"

    def __repr__(self) -> str:
        return f"<HashingEmbedder {self.name}>"

    def _feat(self, text: str) -> dict[int, float]:
        t = text.lower().strip()
        buckets: dict[int, float] = {}

        def bump(tok: str, w: float) -> None:
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
            idx = h % self.dim
            # Signed hashing keeps collisions from systematically inflating similarity.
            sign = 1.0 if (h >> 63) & 1 else -1.0
            buckets[idx] = buckets.get(idx, 0.0) + sign * w

        words = _WORD.findall(t)
        for w in words:
            bump(f"w:{w}", 1.0)
        padded = f" {' '.join(words)} "
        lo, hi = self.ngram
        for n in range(lo, hi + 1):
            for i in range(max(0, len(padded) - n + 1)):
                bump(f"c{n}:{padded[i:i + n]}", 0.5)
        return buckets

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for idx, val in self._feat(t).items():
                out[i, idx] = val
            n = float(np.linalg.norm(out[i]))
            if n > 0.0:
                out[i] /= n
        return out


class CachedEmbedder:
    """Memoizes encodes by text hash.

    Agent transcripts are extraordinarily repetitive - the same system preamble, the
    same acknowledgements, the same restated facts. Caching at this layer removes a
    surprising share of embedding spend for free.
    """

    def __init__(self, inner: Embedder, max_items: int = 50_000) -> None:
        self.inner = inner
        self.dim = inner.dim
        self.max_items = max_items
        self._cache: dict[str, np.ndarray] = {}
        self.hits = 0
        self.misses = 0

    @property
    def name(self) -> str:
        # A cache is not a different embedding space, so it must not change the
        # identity a store is fingerprinted with — otherwise wrapping an embedder in
        # `CachedEmbedder` would look like an embedder swap and demand a re-embed.
        return _name_of(self.inner)

    def __repr__(self) -> str:
        return (f"<CachedEmbedder {self.name} cached={len(self._cache)} "
                f"hits={self.hits} misses={self.misses}>")

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        keys = [hashlib.blake2b(t.encode(), digest_size=16).hexdigest() for t in texts]
        missing = [(i, t) for i, (t, k) in enumerate(zip(texts, keys)) if k not in self._cache]
        if missing:
            self.misses += len(missing)
            vecs = self.inner.encode([t for _, t in missing])
            for (i, _), v in zip(missing, vecs):
                if len(self._cache) >= self.max_items:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[keys[i]] = v
        self.hits += len(texts) - len(missing)
        return np.stack([self._cache[k] for k in keys]).astype(np.float32)
