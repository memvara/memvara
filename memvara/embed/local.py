"""Optional local embedder backed by sentence-transformers.

Kept in its own module so importing `memvara` never pays the import cost of torch.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

#: What `LocalEmbedder()` loads. bge-small-en-v1.5 has the same width as the model before
#: it, 384, and finds more of the evidence: over the 1,531 evidence-labelled LOCOMO
#: questions, R@12 rose from 62.7 to 68.2 and R@1 from 28.5 to 33.2
#: (`bench/locomo.py --score retrieval --embedder local`), for about twice the encoding
#: time. Its cosines run higher, so the thresholds that read them are per model; see
#: `embed/calibration.py`.
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

#: What `LocalEmbedder()` loaded through 0.15. A store it wrote names this model in its
#: fingerprint, and `Memvara` goes on opening that store with it rather than with
#: `DEFAULT_MODEL`: the two share a width, so nothing but the name tells their vectors
#: apart.
PREVIOUS_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
#: Its width, which is also `DEFAULT_MODEL`'s.
PREVIOUS_DEFAULT_DIM = 384


class LocalEmbedder:
    def __init__(self, model: str | None = None) -> None:
        #: Whether the caller named the model. `LocalEmbedder()` asks for the default,
        #: and the default changed under stores that already exist, so `Memvara` refuses
        #: to open a store another local model wrote with an unnamed one, where a named
        #: one only earns a warning.
        self.chosen = model is not None
        model = model if model is not None else DEFAULT_MODEL
        try:
            # `type: ignore` because the SDK is an extra: a checker run in an
            # environment that has not installed `memvara[local-embed]` — CI, and most
            # contributors — cannot resolve the module, and the alternative is a global
            # `ignore_missing_imports` that would also hide a *real* missing import
            # anywhere else in the package.
            from sentence_transformers import (  # type: ignore[import-not-found] # noqa: PLC0415
                SentenceTransformer,
            )
        except ImportError as exc:
            # Naming the extra matters more here than anywhere else it is done: this is
            # the class a user reaches for after reading that the default embedder is a
            # lexical fallback, so a bare ModuleNotFoundError lands on someone who has
            # just been told to fix exactly this and is not told how.
            raise ImportError(
                "LocalEmbedder needs the `sentence-transformers` package: "
                "pip install 'memvara[local-embed]'. The default HashingEmbedder needs "
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
