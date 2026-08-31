"""The interface a memory system implements to be benchmarked.

Four methods. `reset` clears the memory and hands over the predicate schema, `remember`
delivers one observation, `query` answers one question, and `close` releases whatever
was opened. Nothing else is required, and nothing about the benchmark's internals is
exposed to an implementation.

## What the answer has to be

`MemoryAnswer` carries a string, a set of strings, or neither — the third case being an
explicit "I do not know", which is a real answer here and the only correct one for the
`negative` category. Scoring is deterministic (`scoring.py`): normalized comparison
against the gold value and its published aliases. No model reads the answer, so an
adapter that returns a sentence rather than a value is not penalised for style — but it
is not rewarded for hedging either, because `contains the gold string` is not how any
category is scored. Return the value.

## What an adapter may and may not do

It may use anything its system offers: structured writes, temporal queries, ranking,
graph traversal, whatever. It may read `Question.probe`, `Question.at` and
`Question.known_at`, because every adapter gets those and the dataset decides which
questions carry them.

It may not read `Question.gold`, and it is never given it: the runner strips it before
the adapter is called. That is the one rule the code enforces rather than asks for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Protocol, Sequence, runtime_checkable

from ..dataset import MemoryEvent, PredicateDecl


@dataclass(frozen=True, slots=True)
class Ask:
    """A question as an adapter sees it — the gold answer removed.

    Built by the runner from a `Question`. An adapter cannot reach the answer through
    this object, which is what makes "no system-specific hidden test knowledge" a
    property of the code rather than a promise in a README.
    """

    id: str
    category: str
    question: str
    #: `(subject, predicate)` when the dataset names the slot, else `None` and the system
    #: has to find it.
    probe: tuple[str, str] | None
    #: The world instant the question is about. Always set by the runner — `None` in the
    #: dataset means "now", and "now" is the dataset's fixed `evaluated_at`.
    at: datetime
    #: The belief instant, when the question asks what the system would have said then.
    #: `None` means "as we understand it today".
    known_at: datetime | None = None
    #: The value the question is about, for the `provenance`, `change_time` and
    #: `knowledge_time` categories: "which source reported *London*", "when did *London*
    #: begin". `None` everywhere else. It is in the data rather than only in the prose so
    #: that answering does not require parsing the question — see `dataset.Question`.
    about: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryAnswer:
    """What an adapter returns.

    `value=None` with an empty `values` is an abstention, and it is scored as one.
    Returning a wrong value and returning nothing are different outcomes and the report
    keeps them apart.
    """

    #: The answer to a single-valued or date question. ISO-8601 for a date.
    value: str | None = None
    #: The answer to a set-valued question. Order is ignored.
    values: tuple[str, ...] = ()
    #: Ids — of events, claims, rows, whatever the system calls them — that justify the
    #: answer. Never scored. Printed in the failure report, where knowing which memory a
    #: wrong answer came from is most of the debugging.
    support: tuple[str, ...] = ()

    @property
    def abstained(self) -> bool:
        return self.value is None and not self.values


@dataclass
class Usage:
    """Cost counters, reported by whichever system can count them.

    Every field defaults to `None`, meaning *not measured*, which the report prints as
    `-` rather than as zero. A system that does not count its database reads should say
    so; reporting an unmeasured quantity as zero is the one way this section could lie.
    """

    llm_calls: int | None = None
    embedding_calls: int | None = None
    db_writes: int | None = None
    db_reads: int | None = None
    #: Anything a system wants recorded that has no field here. Serialized as-is.
    extra: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        row: dict[str, object] = {
            "llm_calls": self.llm_calls,
            "embedding_calls": self.embedding_calls,
            "db_writes": self.db_writes,
            "db_reads": self.db_reads,
        }
        if self.extra:
            row["extra"] = dict(self.extra)
        return row


@runtime_checkable
class MemorySystem(Protocol):
    """What the runner needs from a memory system.

    Implementations live in this package or anywhere importable; `--system` accepts a
    dotted path, so a third-party system can be benchmarked without this repository
    knowing it exists.
    """

    #: Short identifier used on the command line and in the result file.
    name: str
    #: The version of the *system under test*, not of this adapter. Recorded in the
    #: result so a score can be traced to what produced it.
    version: str

    def reset(self, predicates: Mapping[str, PredicateDecl]) -> None:
        """Discard all memory and adopt the dataset's predicate schema.

        Called **once per run**, before the first event. Every scenario's events go into
        one memory and every question is asked against all of them — scenario is a label
        for reporting, not an isolation boundary, and entities are unique across the
        dataset so nothing collides. `runner.py` says why: a benchmark that gave each
        scenario its own three-fact store would make retrieval trivial.
        """

    def remember(self, event: MemoryEvent) -> None:
        """Take delivery of one observation.

        Events arrive in `recorded_at` order. `event.valid_from` may be earlier than
        `event.recorded_at`, and for the delayed-knowledge scenarios it is.
        """

    def query(self, ask: Ask) -> MemoryAnswer:
        """Answer one question."""

    def usage(self) -> Usage:
        """Counters accumulated since the last `reset`, or an all-`None` `Usage`."""

    def close(self) -> None:
        """Release anything held. Called once at the end of a run."""


def wants_a_date(ask: Ask) -> bool:
    """Does this question ask *when*, rather than *what*?

    `knowledge_time` holds both kinds. "On what date did the system learn X" wants a
    date and names the value it means in `about`. "What would the system have said on
    5 March" wants a value, sets `known_at`, and carries no `about` because it is not
    about one value — it is about whichever value was believed.

    This lives here rather than in each adapter because getting it wrong is silent: the
    date branch looks for a value named in `about`, finds `None`, and abstains. Three
    adapters made that mistake independently, which is three too many for a distinction
    the benchmark can state once.
    """
    return ask.category in ("change_time", "knowledge_time") and ask.about is not None


def default_usage() -> Usage:
    """An all-unmeasured `Usage`, for a system that counts nothing."""
    return Usage()


def sort_events(events: Sequence[MemoryEvent]) -> list[MemoryEvent]:
    """Delivery order: by `recorded_at`, ties broken by id so it is total.

    Ties are the common case, not the corner — a scenario often records several facts on
    the same day — and an order that depended on the JSONL's line order would make the
    benchmark sensitive to a file edit that changed nothing.
    """
    return sorted(events, key=lambda e: (e.recorded_at, e.id))
