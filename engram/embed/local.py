"""Optional local embedder backed by sentence-transformers.

Kept in its own module so importing `engram` never pays the import cost of torch.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class LocalEmbedder:
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError as exc:
            # Naming the extra matters more here than anywhere else it is done: this is
            # the class a user reaches for after reading that the default embedder is a
            # lexical fallback, so a bare ModuleNotFoundError lands on someone who has
            # just been told to fix exactly this and is not told how.
            raise ImportError(
                "LocalEmbedder needs the `sentence-transformers` package: "
                "pip install 'engram[local-embed]'. The default HashingEmbedder needs "
                "nothing and works offline, at the cost of semantic recall."
            ) from exc

        self._m = SentenceTransformer(model)
        self.dim = int(self._m.get_sentence_embedding_dimension())
        # The model id, not the class, is the identity: two sentence-transformers
        # models of the same width produce vectors that are not comparable, and that
        # swap is invisible to a dimension check. See `embed/fingerprint.py`.
        self.name = f"local:{model}"

    def __repr__(self) -> str:
        return f"<LocalEmbedder {self.name} dim={self.dim}>"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self._m.encode(list(texts), normalize_embeddings=True), dtype=np.float32
        )
