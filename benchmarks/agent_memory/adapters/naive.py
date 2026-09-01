"""`naive` — a dictionary of current values, which is what most agent memory actually is.

One entry per `(subject, predicate)`, overwritten when a newer value arrives, with the
source and the date of the write kept beside it. Multi-valued predicates accumulate,
because the dataset declares which ones are and ignoring that would be a strawman rather
than a baseline.

This is not a weakened system built to lose. It is the design that most memory layers
converge on, and it is genuinely good at the thing it is for: it answers *what is true
now* perfectly, it answers *who told us* perfectly for anything it still holds, and it
correctly says nothing about entities it has never heard of. What it cannot do is
reconstruct a past state, because it overwrote it — and the benchmark's job is to make
that difference legible rather than to hide it inside an average.

Where it has no answer it abstains rather than guessing. Abstaining is scored as a
failure everywhere except `negative`; guessing would be scored as a failure too, and
would additionally make the failure report useless. Silence is the honest output of a
store that does not know.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

from ..dataset import MemoryEvent, PredicateDecl
from ..normalization import tokens
from .base import Ask, MemoryAnswer, Usage, indexable, pick_slot, wants_a_date

#: How many distinct slots the overlap ranking offers `base.pick_slot`. Ten, the same as
#: memvara's `SEARCH_K`, so the three systems choose from comparably deep lists and the
#: retrieval dimension is not measuring who was allowed to look furthest.
TOP_K = 10


@dataclass
class Entry:
    """One value currently held, with the little metadata a key-value store keeps."""

    object: str
    source: str
    recorded_at: datetime
    valid_from: datetime


def overlap(question: set[str], text: set[str]) -> float:
    """Token overlap, normalized by the question's length. The whole of this store's
    retrieval, and deliberately so: a dictionary has no index.

    Takes token *sets* rather than strings because the caller has 262 texts to compare
    one question against, and tokenizing the question inside this function meant doing it
    262 times. That cost nothing in answers and a great deal in the benchmark's own
    latency table, where it made a dictionary lookup read as slower than a vector search.
    """
    return len(question & text) / len(question) if question else 0.0


class NaiveMemory:
    """Latest value wins, one slot at a time."""

    name = "naive"
    version = "1.0"

    def __init__(self) -> None:
        self._slots: dict[tuple[str, str], list[Entry]] = {}
        #: (tokens of the indexed text, subject, predicate). Tokenized once, on write.
        #: Not an index — the query still scans all of it — just not re-derived per query.
        self._texts: list[tuple[set[str], str, str]] = []
        self._single: dict[str, bool] = {}
        self._writes = 0
        self._reads = 0

    def reset(self, predicates: Mapping[str, PredicateDecl]) -> None:
        self._slots.clear()
        self._texts.clear()
        self._single = {name: decl.single_valued for name, decl in predicates.items()}
        self._writes = self._reads = 0

    def remember(self, event: MemoryEvent) -> None:
        self._writes += 1
        slot = (event.subject, event.predicate)
        entry = Entry(event.object, event.source, event.recorded_at, event.valid_from)
        held = self._slots.setdefault(slot, [])
        if self._single.get(event.predicate, True):
            held.clear()
            held.append(entry)
        elif not any(e.object == entry.object for e in held):
            held.append(entry)
        self._texts.append((set(tokens(indexable(event))), event.subject, event.predicate))

    def _resolve(self, question: str) -> tuple[str, str] | None:
        """Find the slot an unprobed question is about, by token overlap.

        The ranking is this baseline's own and stays crude. Choosing among the top of it
        is `base.pick_slot`, which every system shares — see there for why the choice is
        not each adapter's to make. Ties go to the first stored text, which makes the
        order deterministic rather than arbitrary.
        """
        asked = set(tokens(question))
        scored = [(overlap(asked, text), index, subject, predicate)
                  for index, (text, subject, predicate) in enumerate(self._texts)]
        ranked = sorted((s for s in scored if s[0] > 0.0), key=lambda s: (-s[0], s[1]))
        slots = list(dict.fromkeys((s[2], s[3]) for s in ranked))[:TOP_K]
        return pick_slot(question, slots)

    def query(self, ask: Ask) -> MemoryAnswer:
        self._reads += 1
        slot = ask.probe or self._resolve(ask.question)
        if slot is None:
            return MemoryAnswer()
        held = self._slots.get(slot)
        if not held:
            return MemoryAnswer()

        support = tuple(f"{slot[0]}/{slot[1]}={e.object}" for e in held)
        if ask.category == "provenance":
            entry = next((e for e in held if e.object == ask.about), None)
            # The source of a value it no longer holds went with the value.
            return MemoryAnswer(value=entry.source, support=support) if entry else MemoryAnswer()
        if wants_a_date(ask):
            entry = next((e for e in held if e.object == ask.about), None)
            if entry is None:
                return MemoryAnswer()
            when = entry.valid_from if ask.category == "change_time" else entry.recorded_at
            return MemoryAnswer(value=when.date().isoformat(), support=support)
        if ask.category == "change_detection":
            # It can only report what it still holds, which for a single-valued slot is
            # one value however many times the slot changed.
            return MemoryAnswer(values=tuple(e.object for e in held), support=support)
        if len(held) > 1:
            return MemoryAnswer(values=tuple(e.object for e in held), support=support)
        return MemoryAnswer(value=held[0].object, support=support)

    def usage(self) -> Usage:
        # Rows held, not calls made: this store overwrites, so it ends up holding far
        # fewer rows than it was handed — which is the whole difference between it and a
        # store that keeps history, and the number the report is for.
        rows = sum(len(entries) for entries in self._slots.values())
        return Usage(llm_calls=0, tokens=0, texts_embedded=0, rows_stored=rows,
                     db_reads=self._reads)

    def close(self) -> None:
        return None


def build(**_: object) -> NaiveMemory:
    return NaiveMemory()
