"""`memvara` — the adapter for this repository's own library, through its public API.

Nothing here reaches into `memvara`'s internals. The writes go through `remember()`, the
reads through `history()`, `search()` and `why()`, and the predicate schema through
`PredicateRegistry` — which is the same route an application takes. Benchmarking a
library through a private door produces a number nobody can reproduce with the published
one.

## Two choices in this adapter that are worth arguing with

**The write path uses `remember()`, not `add()`.** The benchmark hands every system a
structured `(subject, predicate, object)` alongside the sentence, so extraction is out of
scope for all of them (`dataset.py` says why). `remember()` is the API for a fact the
application already has as structured data. The consequence is that this adapter's
`llm_calls` is zero *by construction* rather than as a finding, and the report says so
rather than presenting it as a result.

**A same-instant conflict is written as a retirement.** When an incoming event describes
the same `valid_from` as a value already held and disagrees with it, this adapter passes
`close="retired"` — the record was wrong — instead of the default `close="ended"`, which
would say the world had changed. That is supersession rule 2 from `timeline.py`, which is
published with the dataset and available to every adapter; `vector-rag` implements the
same rule. It is an integration decision a real caller makes, and getting it wrong is the
mistake `memvara/server/tools.py` warns about at length: reporting a correction as an
ending records a false reason for the change.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Sequence

import numpy as np

from memvara import (
    Cardinality,
    Claim,
    Episode,
    HashingEmbedder,
    Memvara,
    NullLLM,
    PredicateRegistry,
    PredicateSpec,
    Volatility,
    __version__ as memvara_version,
)

from ..dataset import MemoryEvent, PredicateDecl
from .base import Ask, MemoryAnswer, Usage, indexable, wants_a_date

#: Fixed rather than `default_embedder()`, which returns a sentence-transformer whenever
#: one happens to be installed. A benchmark whose vector leg depends on what is in the
#: environment is not reproducible, and `memvara[rerank]` installs one as a side effect.
EMBED_DIM = 512

#: How many claims an unprobed question pulls back before a slot is chosen.
SEARCH_K = 10


class _CountingEmbedder:
    """`HashingEmbedder`, plus a tally of how many texts it was asked to encode.

    memvara embeds on write and again on every unprobed read, and reports neither: the
    library has no cost counter for it, so the benchmark's `texts_embedded` column read
    `-` for memvara while both baselines reported a number. `-` means *not measured* and
    was honest, but it left the one system doing the most embedding as the one with no
    figure.

    Counting texts rather than calls, because a batched call and a loop of single ones do
    the same work and should read the same. `name` and `dim` are delegated so the
    embedder's identity is unchanged: `memvara.embed.fingerprint` derives a store's
    recorded identity from exactly those two, and a wrapper that shadowed the name would
    make a file-backed store refuse to reopen with the embedder that wrote it.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.texts_embedded = 0
        # Plain attributes rather than properties: `Embedder` declares `dim` as a
        # settable variable, and a read-only property does not satisfy the protocol.
        self.dim: int = int(inner.dim)
        self.name: str = str(getattr(inner, "name", type(inner).__name__))

    def encode(self, texts: Sequence[str]) -> "np.ndarray":
        self.texts_embedded += len(texts)
        return self._inner.encode(texts)


class MemvaraMemory:
    """memvara, driven the way an application drives it."""

    name = "memvara"
    version = memvara_version

    def __init__(self, path: str = ":memory:", user: str = "benchmark") -> None:
        self._path = path
        self._user = user
        self._mem: Memvara | None = None
        self._single: dict[str, bool] = {}
        self._embedder: _CountingEmbedder | None = None
        self._reads = 0

    @property
    def mem(self) -> Memvara:
        assert self._mem is not None, "reset() must run before any other call"
        return self._mem

    def reset(self, predicates: Mapping[str, PredicateDecl]) -> None:
        if self._mem is not None:
            self._mem.close()
        registry = PredicateRegistry()
        self._single = {name: decl.single_valued for name, decl in predicates.items()}
        for name, decl in predicates.items():
            registry.register(PredicateSpec(
                name=name,
                cardinality=Cardinality(decl.cardinality),
                volatility=Volatility(decl.volatility),
            ))
        # `NullLLM` because this adapter never calls `add()`: there is no text to
        # extract from, only structured facts to record.
        self._embedder = _CountingEmbedder(HashingEmbedder(dim=EMBED_DIM))
        self._mem = Memvara(self._path, user=self._user, registry=registry,
                            embedder=self._embedder, llm=NullLLM())
        self._reads = 0

    # -- write --------------------------------------------------------------

    def _is_correction(self, event: MemoryEvent) -> bool:
        """Same fact slot, same `valid_from`, different value: a correction, not a change.

        **Only on a single-valued predicate.** Two values that begin on the same day are
        the ordinary case for a multi-valued relation, not a contradiction — somebody
        speaks two languages and we learned both at once. Without the guard this returned
        True for 30 writes to `speaks` in the shipped dataset and told `remember()` the
        earlier record was *wrong*. memvara accumulates on a MANY predicate whatever
        `close=` says, so nothing was corrupted; that is the library defending against
        this method rather than this method being right, and `memvara/server/tools.py` is
        explicit that a false reason for a change cannot be found by reading the data
        afterwards.
        """
        if not self._single.get(event.predicate, True):
            return False
        held = self.mem.history(event.subject, event.predicate)
        return any(claim.valid_from == event.valid_from and claim.object != event.object
                   and claim.state != "retired"
                   for claim in held)

    def remember(self, event: MemoryEvent) -> None:
        source = Episode(content=event.text, ts=event.recorded_at,
                         meta={"source": event.source, "event_id": event.id})
        self.mem.remember(
            event.subject, event.predicate, event.object,
            valid_from=event.valid_from, valid_to=event.valid_to,
            recorded_at=event.recorded_at, confidence=event.confidence,
            text=indexable(event), sources=[source],
            close="retired" if self._is_correction(event) else "ended",
        )

    # -- read ---------------------------------------------------------------

    def _resolve(self, question: str, ask: Ask) -> tuple[str, str] | None:
        """Find the slot an unprobed question is about, using memvara's own retrieval.

        **Search the population the question is about.** A present-tense question is
        answered from live claims; only a question that rewinds a clock needs the values
        that have finished or been retracted.

        This used to pass all three states unconditionally, and it was the single largest
        cause of memvara's `retrieval` score. Asked *"Where does Frank live?"*, the right
        claim was BM25 rank 0 and still lost the fused ranking to `alice lives_in Berlin`
        — a value Alice has not held since March, competing for a question about now. The
        symptom read as weak ranking; the cause was asking the wrong population.
        """
        past = ask.known_at is not None or ask.at < ask.evaluated_at
        states = ["live", "ended", "retired"] if past else None
        hits = self.mem.search(question, k=SEARCH_K, states=states)
        return (hits[0].claim.subject, hits[0].claim.predicate) if hits else None

    def _first_source(self, claim: Claim) -> str | None:
        """The label of the earliest turn this claim cites.

        Earliest rather than any, because a fact restated four times cites four turns and
        the question asks which one *first* reported it. `why()` is the provenance API,
        and using anything else here would benchmark a shortcut.
        """
        provenance = self.mem.why(claim.id)
        if provenance is None or not provenance.episodes:
            return None
        first = min(provenance.episodes, key=lambda e: e.ts)
        value = first.meta.get("source")
        return str(value) if value is not None else None

    def query(self, ask: Ask) -> MemoryAnswer:
        self._reads += 1
        slot = ask.probe or self._resolve(ask.question, ask)
        if slot is None:
            return MemoryAnswer()
        versions = self.mem.history(*slot)
        if not versions:
            return MemoryAnswer()

        if ask.category == "provenance":
            claim = self._pick(versions, ask.about)
            if claim is None:
                return MemoryAnswer()
            source = self._first_source(claim)
            return MemoryAnswer(value=source, support=(claim.id,)) if source else MemoryAnswer()

        if wants_a_date(ask):
            claim = self._pick(versions, ask.about)
            if claim is None:
                return MemoryAnswer()
            when = claim.valid_from if ask.category == "change_time" else claim.recorded_at
            return MemoryAnswer(value=when.date().isoformat(), support=(claim.id,))

        if ask.category == "change_detection":
            # Values the record still stands behind: `ended` is a value that stopped
            # being true, `retired` is one we stopped believing, and only the first is a
            # value the subject ever had.
            kept = [c for c in versions if c.state != "retired"]
            return MemoryAnswer(values=tuple(dict.fromkeys(c.object for c in kept)),
                                support=tuple(c.id for c in kept))

        live = (self._as_of(slot, ask.at, ask.known_at) if ask.known_at is not None
                else [c for c in versions if c.is_live(valid_at=ask.at)])
        if not live:
            return MemoryAnswer()
        support = tuple(c.id for c in live)
        if len(live) > 1:
            return MemoryAnswer(values=tuple(dict.fromkeys(c.object for c in live)),
                                support=support)
        return MemoryAnswer(value=live[0].object, support=support)

    def _as_of(self, slot: tuple[str, str], at: datetime,
               known_at: datetime) -> list[Claim]:
        """What this store would have answered at `known_at`, about `at`.

        `is_live(valid_at=..., known_at=...)` is the wrong read for this question, and
        the reason is worth stating because it looks right. It evaluates a row on its
        own, and a superseded row carries a `valid_to` stamped by a successor that the
        belief clock has not yet reached — so the row reads as finished at an instant
        when the store had not heard it was. `Memvara.ask()` says the same thing about
        `Reading.stated`: a row read on its own cannot date its own ending.

        `history(known_at=T)` can, because it withholds the successor entirely. What is
        left is every version recorded by T and not retracted by T, and the answer is the
        latest of them to have started — which is what the store would have said.

        `Memvara.ask(question, at=T).readings[].stated` is the one-call form of this and
        composes the same three reads. It is not used here because it selects its slots by
        searching the question text, and a probed question already names the slot; going
        through the ranker would make a temporal result depend on retrieval.
        """
        seen = self.mem.history(*slot, known_at=known_at)
        believed = [c for c in seen
                    if c.invalidated_at is None or c.invalidated_at > known_at]
        started = [c for c in believed if c.valid_from <= at]
        if not started:
            return []
        if not self._single.get(slot[1], True):
            return [c for c in started if c.valid_to is None or at < c.valid_to]
        newest = max(c.valid_from for c in started)
        return [c for c in started if c.valid_from == newest]

    @staticmethod
    def _pick(versions: list[Claim], value: str | None) -> Claim | None:
        """The earliest surviving claim carrying `value`."""
        matching = [c for c in versions if c.object == value and c.state != "retired"]
        if not matching:
            matching = [c for c in versions if c.object == value]
        return min(matching, key=lambda c: c.recorded_at) if matching else None

    def usage(self) -> Usage:
        stats = self.mem.stats()
        return Usage(
            llm_calls=0,
            # Zero calls is zero tokens; both are measured, not absent.
            tokens=0,
            texts_embedded=(self._embedder.texts_embedded
                            if self._embedder is not None else None),
            rows_stored=int(stats.get("claims", 0)),
            db_reads=self._reads,
            extra={"episodes": int(stats.get("episodes", 0))},
        )

    def close(self) -> None:
        if self._mem is not None:
            self._mem.close()
            self._mem = None


def build(**kwargs: object) -> MemvaraMemory:
    return MemvaraMemory(**kwargs)  # type: ignore[arg-type]
