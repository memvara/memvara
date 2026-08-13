"""Optional cross-encoder reranker, backed by sentence-transformers.

Kept in its own module, and reachable from `memvara.rerank` only as a PEP 562 lazy
attribute, so that importing `memvara` never pays the import cost of torch and never
touches a model. That is the same arrangement `memvara/embed/local.py` has with
`LocalEmbedder`, for the same reason: the default configuration of this library is
"numpy and nothing else, offline, no API key", and a reranker that broke it would be
worth less than the recall it bought.

A cross-encoder is a different animal from the bi-encoder in `memvara/embed/`. A
bi-encoder embeds the query and the document separately and compares the two vectors,
which is what makes a whole-scope search affordable and also what caps its accuracy —
neither side ever sees the other. A cross-encoder concatenates query and document into
one sequence and attends across the join, so it can judge whether a passage *answers* a
question rather than whether it is about the same topic. It cannot be precomputed, which
is exactly why it belongs at the end of a pipeline over a bounded candidate list rather
than at the start over the corpus.
"""

from __future__ import annotations

from typing import Any, Sequence

#: ms-marco MiniLM-L6: 22M parameters, ~90MB, and the standard baseline cross-encoder
#: for passage reranking. Named as a default rather than hardcoded because the whole
#: point of the protocol is that a deployment picks its own trade of cost for accuracy.
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Scores (query, document) pairs with a sentence-transformers `CrossEncoder`.

    The model is loaded in `__init__`, not on the first `score()`, and that placement is
    the same fail-fast decision the hosted LLM backends make about a missing API key: a
    server that starts clean and then dies on the first query that reaches reranking has
    hidden the failure until after the deployment looked healthy. There is no key to
    check here, so the equivalent check is the one that can actually fail — the package
    being absent, and the model being resolvable — and both happen at construction.

    Pass `encoder=` to inject an already-loaded model. That is how a process that shares
    one model across several `Memvara` instances avoids loading it twice, and it is how
    the test suite exercises this class without the extra installed.
    """

    def __init__(self, model: str = DEFAULT_MODEL, *, encoder: Any = None,
                 batch_size: int = 32) -> None:
        self.batch_size = batch_size
        #: The model id, not the class, is the identity — two cross-encoders of the same
        #: architecture rank differently and a caller reading `rerank_score` in a log
        #: needs to know which one produced it. Same argument as `LocalEmbedder.name`.
        self.name = f"cross-encoder:{model}"
        self._encoder = self._load(model) if encoder is None else encoder

    @staticmethod
    def _load(model: str) -> Any:
        try:
            # `type: ignore` because the SDK is an extra: a checker run in an
            # environment that has not installed `memvara[rerank]` — CI, and most
            # contributors — cannot resolve the module, and the alternative is a global
            # `ignore_missing_imports` that would also hide a *real* missing import
            # anywhere else in the package.
            from sentence_transformers import (  # type: ignore[import-not-found] # noqa: PLC0415
                CrossEncoder,
            )
        except ImportError as exc:
            # Naming the extra rather than letting a bare ModuleNotFoundError through:
            # the person reading this traceback has just been told that reranking is
            # opt-in and has no way to guess that the extra is spelled `rerank` while
            # the package is spelled `sentence-transformers`.
            raise ImportError(
                "CrossEncoderReranker needs the `sentence-transformers` package: "
                "pip install 'memvara[rerank]'. Note that it also downloads a model on "
                "first use, which is why reranking is opt-in — memvara's default "
                "configuration retrieves offline with no model at all."
            ) from exc
        return CrossEncoder(model)

    def __repr__(self) -> str:
        return f"<CrossEncoderReranker {self.name} batch={self.batch_size}>"

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """One logit per document. Unbounded and not comparable across models.

        Nothing downstream thresholds on the value — the stage only sorts by it — so it
        is passed through unsquashed. A sigmoid here would look tidier in a log and
        would throw away the resolution that separates two near-identical passages,
        which is the entire reason a cross-encoder is being paid for.
        """
        if not documents:
            # `CrossEncoder.predict([])` is a wasted forward-pass setup at best and a
            # shape error at worst, depending on the version.
            return []
        pairs = [(query, doc) for doc in documents]
        return [float(s) for s in
                self._encoder.predict(pairs, batch_size=self.batch_size)]
