"""`LocalSelector`: a `Selector` that runs a small cross-encoder in this process.

`ModelSelector` asks a chat model, on a key the deployment holds, which of a ranked
read's candidate turns bear on the question. This class asks a cross-encoder instead. It
scores each candidate against the question, turns each score into a probability with a
calibration measured on labelled turns, and keeps the candidates whose probability
reaches a threshold, at most `max_keep` of them. No text leaves the process, nothing is
billed, and the same candidates always get the same answer. The design is
`docs/superpowers/specs/2026-10-01-local-selector-design.md`.

The model loads when the selector is built, as `CrossEncoderReranker`'s does, so a
missing extra or a missing model fails where it is configured and not on the first
ranked read. Naming the class costs nothing heavy: `memvara.select` reaches this module
through a PEP 562 attribute, and `sentence_transformers` is imported inside
`load_encoder()`.

`select()` raises `ValueError` when the encoder returns a different number of scores
than it was given candidates, or a score that is not a finite number. The ranked stage
serves the plain read for that, counted as `retrieval.local_fallback` with reason
`malformed`. Anything else the encoder raises is served the same way, with reason
`error`.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from ..llm.base import Usage
from .base import Candidate, Selected

#: The cross-encoder memvara already reranks with (`memvara.rerank.cross.DEFAULT_MODEL`),
#: pinned to a commit. A Hugging Face commit fixes the weights, because it records each
#: large file's SHA-256 in that file's pointer.
STOCK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
STOCK_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"

#: The file a trained selector model keeps beside its weights. `Calibration.read` reads
#: it and `bench/selector_train.py` writes it.
CALIBRATION_FILE = "memvara_selector.json"

#: The only format `Calibration.read` accepts. A file with another value is refused
#: rather than half-read.
CALIBRATION_FORMAT = 1


@dataclass(frozen=True, slots=True)
class Calibration:
    """How a raw cross-encoder score becomes a keep decision.

    `probability(score)` is `sigmoid(scale * score + shift)`, which is Platt scaling: `scale`
    and `shift` are fitted by logistic regression on labelled candidates the model was not
    trained on. A candidate is kept when its probability is at least `threshold`, and at
    most `max_keep` candidates are kept, the highest-scoring first. `max_length` is the
    token limit the scores were measured at; `None` means the model's own limit.
    """

    scale: float
    shift: float
    threshold: float
    max_keep: int
    max_length: int | None = None

    def __post_init__(self) -> None:
        if not (math.isfinite(self.scale) and self.scale > 0):
            raise ValueError(f"scale must be a positive finite number, not {self.scale!r}")
        if not math.isfinite(self.shift):
            raise ValueError(f"shift must be a finite number, not {self.shift!r}")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError(f"threshold must be between 0 and 1, not {self.threshold!r}")
        if self.max_keep < 1:
            raise ValueError(f"max_keep must be at least 1, not {self.max_keep!r}")
        if self.max_length is not None and self.max_length < 1:
            raise ValueError(f"max_length must be positive, not {self.max_length!r}")

    def probability(self, score: float) -> float:
        z = self.scale * score + self.shift
        # Two branches so that `exp` never sees a large positive number.
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        e = math.exp(z)
        return e / (1.0 + e)

    @classmethod
    def read(cls, directory: Path) -> "tuple[Calibration, str | None]":
        """The calibration in `directory/memvara_selector.json`, and the SHA-256 of
        `model.safetensors` that the file records, or `None` when it records none."""
        path = directory / CALIBRATION_FILE
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("format") != CALIBRATION_FORMAT:
            raise ValueError(f"{path} has format {data.get('format')!r}; this version of "
                             f"memvara reads format {CALIBRATION_FORMAT}")
        max_length = data.get("max_length")
        calibration = cls(scale=float(data["scale"]), shift=float(data["shift"]),
                          threshold=float(data["threshold"]),
                          max_keep=int(data["max_keep"]),
                          max_length=None if max_length is None else int(max_length))
        digest = data.get("weights_sha256")
        return calibration, None if digest is None else str(digest)


#: Calibrations memvara ships for Hugging Face models, keyed by model id and commit.
_BUILT_IN: dict[tuple[str, str], Calibration] = {}


def keep_positions(scores: Sequence[float], calibration: Calibration) -> list[int]:
    """Which of `scores` the keep rule keeps, as positions in ascending order: every score
    whose probability reaches the threshold, at most `max_keep` of them, the highest
    first, a tie going to the earlier position. `bench/selector_metrics.py` replays the
    ranked stage with this same function, so what is measured is what ships."""
    passing = [i for i, s in enumerate(scores)
               if calibration.probability(s) >= calibration.threshold]
    best = sorted(passing, key=lambda i: (-scores[i], i))[:calibration.max_keep]
    return sorted(best)


def load_encoder(model: str, revision: str | None = None,
                 max_length: int | None = None) -> Any:
    """A sentence-transformers `CrossEncoder` for `model` at `revision`."""
    try:
        # `type: ignore` because the package is an extra, as in `memvara.rerank.cross`.
        from sentence_transformers import (  # type: ignore[import-not-found] # noqa: PLC0415
            CrossEncoder,
        )
    except ImportError as exc:
        raise ImportError(
            "LocalSelector needs the `sentence-transformers` package: pip install "
            "'memvara[rerank]'. The model is downloaded once, on first use, unless it is "
            "already in the Hugging Face cache or `model` is a local directory."
        ) from exc
    return CrossEncoder(model, revision=revision, max_length=max_length)


def _check_weights(directory: Path, expected: str) -> None:
    path = directory / "model.safetensors"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(
            f"{path} has SHA-256 {digest.hexdigest()}, but {directory / CALIBRATION_FILE} "
            f"records {expected}. These weights are not the ones the calibration was "
            "measured on.")


class LocalSelector:
    """A `Selector` backed by a cross-encoder in this process. See the module docstring.

    `model` is a Hugging Face model id or a local directory. A Hub id needs a calibration
    memvara ships, which is keyed by id and commit, unless `calibration=` is given; the
    stock model's commit is filled in when `revision` is not. A directory needs its
    `memvara_selector.json` unless `calibration=` is given, and when that file records the
    weights' SHA-256, the weights are checked before they are loaded.

    `encoder=` injects an already-loaded `CrossEncoder`, as `CrossEncoderReranker` allows,
    so a process that reranks and selects with the same model loads it once. The caller is
    then responsible for it being the model `calibration` was measured on, and no digest
    is checked.
    """

    #: Read by the ranked stage to pick its series. A local selector's reads are counted
    #: as `retrieval.local_query`, never as `retrieval.model_query`, which counts calls a
    #: model provider answered.
    kind = "local"

    def __init__(self, model: str = STOCK_MODEL, *, revision: str | None = None,
                 calibration: Calibration | None = None, encoder: Any = None,
                 top_n: int = 40, batch_size: int = 32) -> None:
        if top_n < 1:
            raise ValueError(f"top_n must be at least 1, not {top_n!r}")
        directory = Path(model)
        digest: str | None = None
        if directory.is_dir():
            revision = None
            if calibration is None:
                calibration, digest = Calibration.read(directory)
        else:
            if revision is None and model == STOCK_MODEL:
                revision = STOCK_REVISION
            if calibration is None:
                calibration = _BUILT_IN.get((model, revision or ""))
            if calibration is None:
                raise ValueError(
                    f"memvara ships no calibration for {model!r} at revision {revision!r}. "
                    "Pass calibration=, or a model directory written by "
                    "bench/selector_train.py.")
        self.model = model
        self.revision = revision
        self.calibration = calibration
        #: How many reranked, routed turns the ranked stage hands `select()`. Read by
        #: `hybrid.py`, as for `ModelSelector`.
        self.top_n = top_n
        self.batch_size = batch_size
        if encoder is None:
            if digest is not None:
                _check_weights(directory, digest)
            encoder = load_encoder(model, revision, calibration.max_length)
        self._encoder = encoder
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"<LocalSelector {self.model} top_n={self.top_n}>"

    @property
    def encoder(self) -> Any:
        """The loaded cross-encoder, for a caller that reranks with the same model and
        should not load it twice (`memvara.server.config`)."""
        return self._encoder

    def admit(self) -> AbstractContextManager[None]:
        """Never refuses. A deployment that caps concurrent ranked reads wraps this, as
        the hosted service wraps `ModelSelector`."""
        return nullcontext()

    def scores(self, question: str, texts: Sequence[str]) -> list[float]:
        """One raw score per text, in order. Scoring holds a lock per selector, so two
        ranked reads at once take turns on the CPU rather than oversubscribing it."""
        if not texts:
            return []
        pairs = [(question, text) for text in texts]
        with self._lock:
            raw = self._encoder.predict(pairs, batch_size=self.batch_size,
                                        show_progress_bar=False)
        out = [float(s) for s in raw]
        if len(out) != len(texts):
            raise ValueError(f"the encoder returned {len(out)} scores for {len(texts)} "
                             "candidates")
        if not all(math.isfinite(s) for s in out):
            raise ValueError("the encoder returned a score that is not a finite number")
        return out

    def select(self, question: str, candidates: Sequence[Candidate], *,
               asked_on: datetime | None = None,
               usage: Usage | None = None) -> list[Selected]:
        """The candidates to keep, in the order they were handed in.

        `asked_on` and `usage` are accepted for the protocol and not used: the scores do
        not depend on the date, and no tokens are billed.
        """
        if not candidates:
            return []
        kept = keep_positions(self.scores(question, [c.text for c in candidates]),
                              self.calibration)
        return [Selected(id=candidates[i].id, span=None) for i in kept]
