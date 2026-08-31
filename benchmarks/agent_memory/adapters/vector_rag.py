"""`vector-rag` — retrieval over the whole write log, with one clock.

The strongest baseline that is not bitemporal, and the one worth beating. It keeps every
observation rather than overwriting, indexes each one's sentence for retrieval, and
carries the metadata it was given. Asked about a past instant it does the sensible thing a
single-clock store can do: take the most recent write it had received by then.

That design answers a great deal correctly, and the report shows it doing so. It gets
current state right, it gets provenance right, it reconstructs history correctly wherever
a fact was recorded on the day it became true — which is most of the dataset — and it
enumerates a slot's past values. A benchmark on which this scored near zero would be a
benchmark measuring the wrong thing.

It comes apart in exactly two places, and they are the two the dataset was built around:

* **Delayed knowledge.** With one clock it cannot separate *when a thing became true* from
  *when we were told*, so on the `atlas_deploy` and `auth_migration` scenarios its answer
  to one of those questions is necessarily the other's.
* **Correction.** A later record about the same instant is a retraction, not a second
  value. Keeping every write means keeping the retracted one, so it enumerates values the
  record no longer stands behind.

## The vector leg

Hashed bag-of-words with inverse document frequency, cosine-compared, in numpy. That is a
real sparse retriever and not a neural embedder, which is a limitation stated here and in
`README.md` rather than glossed: a sentence-transformer would rank paraphrase better than
this does. It buys the two things the benchmark needs from a baseline — no API key and
byte-identical results on every run — and the questions are not written to defeat lexical
matching, so the gap it leaves is small relative to what the temporal categories measure.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence

import numpy as np

from ..dataset import MemoryEvent, PredicateDecl
from ..normalization import tokens
from .base import Ask, MemoryAnswer, Usage, indexable, wants_a_date

#: Width of the hashed vocabulary. Large enough that collisions between the dataset's few
#: hundred distinct tokens are rare, small enough that the index is trivial.
DIM = 4096

#: How many events a query pulls back before the answer is chosen from them.
TOP_K = 12


@dataclass
class Record:
    """One observation, kept forever."""

    id: str
    subject: str
    predicate: str
    object: str
    text: str
    source: str
    recorded_at: datetime
    valid_from: datetime
    #: When a later record about the same instant contradicted this one. Supersession
    #: rule 2 from `timeline.py`, which is published with the dataset and implemented
    #: here as well as in the memvara adapter — a rule available to one system and not
    #: the others would be the benchmark rigging itself.
    retracted_at: datetime | None = None


def _hash(token: str) -> int:
    # Python's `hash()` is salted per process, so it would make two runs differ. This is
    # FNV-1a, which is not.
    h = 0x811C9DC5
    for byte in token.encode("utf-8"):
        h = ((h ^ byte) * 0x01000193) & 0xFFFFFFFF
    return h % DIM


class VectorRAGMemory:
    """Full write log, vector retrieval, one clock."""

    name = "vector-rag"
    version = "1.0"

    def __init__(self, top_k: int = TOP_K) -> None:
        self.top_k = top_k
        self._records: list[Record] = []
        self._counts: list[dict[int, int]] = []
        self._df = np.zeros(DIM, dtype=np.float64)
        self._matrix: np.ndarray | None = None
        self._single: dict[str, bool] = {}
        self._writes = self._reads = self._embeds = 0

    def reset(self, predicates: Mapping[str, PredicateDecl]) -> None:
        self._records.clear()
        self._counts.clear()
        self._df = np.zeros(DIM, dtype=np.float64)
        self._matrix = None
        self._single = {name: decl.single_valued for name, decl in predicates.items()}
        self._writes = self._reads = self._embeds = 0

    def remember(self, event: MemoryEvent) -> None:
        self._writes += 1
        self._embeds += 1
        if self._single.get(event.predicate, True):
            for held in self._records:
                if ((held.subject, held.predicate) == (event.subject, event.predicate)
                        and held.valid_from == event.valid_from
                        and held.object != event.object
                        and held.retracted_at is None
                        and held.recorded_at < event.recorded_at):
                    held.retracted_at = event.recorded_at
        self._records.append(Record(
            id=event.id, subject=event.subject, predicate=event.predicate,
            object=event.object, text=event.text, source=event.source,
            recorded_at=event.recorded_at, valid_from=event.valid_from))
        counts: dict[int, int] = {}
        for token in tokens(indexable(event)):
            counts[_hash(token)] = counts.get(_hash(token), 0) + 1
        self._counts.append(counts)
        for index in counts:
            self._df[index] += 1
        self._matrix = None            # invalidated; rebuilt on the next query

    # -- retrieval ----------------------------------------------------------

    def _idf(self) -> np.ndarray:
        n = max(len(self._records), 1)
        return np.log((1.0 + n) / (1.0 + self._df)) + 1.0

    def _index(self) -> np.ndarray:
        if self._matrix is None:
            idf = self._idf()
            matrix = np.zeros((len(self._records), DIM), dtype=np.float64)
            for row, counts in enumerate(self._counts):
                for index, count in counts.items():
                    matrix[row, index] = count * idf[index]
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            self._matrix = matrix / np.where(norms == 0.0, 1.0, norms)
        return self._matrix

    def _search(self, query: str, k: int) -> list[Record]:
        if not self._records:
            return []
        self._embeds += 1
        idf = self._idf()
        vector = np.zeros(DIM, dtype=np.float64)
        for token in tokens(query):
            vector[_hash(token)] += idf[_hash(token)]
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return []
        scores = self._index() @ (vector / norm)
        # `-scores` then id order, so ties do not depend on numpy's sort stability.
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], self._records[i].id))
        return [self._records[i] for i in order[:k] if scores[i] > 0.0]

    # -- answering ----------------------------------------------------------

    def _slot_records(self, slot: tuple[str, str]) -> list[Record]:
        return [r for r in self._records if (r.subject, r.predicate) == slot]

    def _resolve(self, question: str) -> tuple[str, str] | None:
        hits = self._search(question, self.top_k)
        return (hits[0].subject, hits[0].predicate) if hits else None

    @staticmethod
    def _latest(records: Sequence[Record], known_at: datetime | None) -> list[Record]:
        """The most recent write, on the only clock this store has.

        `known_at` is the belief instant when the question supplies one and the world
        instant otherwise, which is the collapse a single-clock store makes and cannot
        avoid. It is the source of every wrong answer this baseline gives on the
        delayed-knowledge scenarios, and it is not a bug in the implementation.
        """
        visible = [r for r in records
                   if (known_at is None or r.recorded_at <= known_at)
                   and (r.retracted_at is None
                        or (known_at is not None and known_at < r.retracted_at))]
        if not visible:
            return []
        newest = max(r.recorded_at for r in visible)
        return [r for r in visible if r.recorded_at == newest]

    def query(self, ask: Ask) -> MemoryAnswer:
        self._reads += 1
        slot = ask.probe or self._resolve(ask.question)
        if slot is None:
            return MemoryAnswer()
        records = self._slot_records(slot)
        if not records:
            return MemoryAnswer()

        if ask.category == "provenance":
            match = [r for r in records if r.object == ask.about and r.retracted_at is None] \
                or [r for r in records if r.object == ask.about]
            first = min(match, key=lambda r: r.recorded_at) if match else None
            return MemoryAnswer(value=first.source, support=(first.id,)) if first else MemoryAnswer()

        if wants_a_date(ask):
            match = [r for r in records if r.object == ask.about and r.retracted_at is None] \
                or [r for r in records if r.object == ask.about]
            if not match:
                return MemoryAnswer()
            first = min(match, key=lambda r: r.recorded_at)
            when = first.valid_from if ask.category == "change_time" else first.recorded_at
            return MemoryAnswer(value=when.date().isoformat(), support=(first.id,))

        if ask.category == "change_detection":
            kept = [r for r in records if r.retracted_at is None]
            seen = tuple(dict.fromkeys(r.object for r in kept))
            return MemoryAnswer(values=seen, support=tuple(r.id for r in kept))

        # One clock: a question about a world instant is answered from the write log as
        # it stood at that instant.
        cutoff = ask.known_at or ask.at
        if self._single.get(slot[1], True):
            latest = self._latest(records, cutoff)
            if not latest:
                return MemoryAnswer()
            chosen = max(latest, key=lambda r: (r.valid_from, r.id))
            return MemoryAnswer(value=chosen.object, support=(chosen.id,))

        visible = [r for r in records
                   if (cutoff is None or r.recorded_at <= cutoff)
                   and (r.retracted_at is None
                        or (cutoff is not None and cutoff < r.retracted_at))]
        return MemoryAnswer(values=tuple(dict.fromkeys(r.object for r in visible)),
                            support=tuple(r.id for r in visible))

    def usage(self) -> Usage:
        # Rows held. This store appends every observation, so it equals the event count
        # — which is itself the finding: it keeps everything, and pays for it.
        return Usage(llm_calls=0, embedding_calls=self._embeds,
                     rows_stored=len(self._records), db_reads=self._reads)

    def close(self) -> None:
        return None


def build(**_: object) -> VectorRAGMemory:
    return VectorRAGMemory()
