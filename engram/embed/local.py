"""Optional local embedder backed by sentence-transformers.

Kept in its own module so importing `engram` never pays the import cost of torch.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class LocalEmbedder:
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self._m = SentenceTransformer(model)
        self.dim = int(self._m.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self._m.encode(list(texts), normalize_embeddings=True), dtype=np.float32
        )
