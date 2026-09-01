"""The benchmark's data model: what a system is told, and what it is asked.

Three record types, all JSON, all versioned together under `datasets/<version>/`:

    events.jsonl     what the memory system is told, in the order it is told
    questions.jsonl  what it is then asked, with the gold answer
    metadata.json    the predicate schema, the scenarios, and the dimension map

`events.jsonl` is deliberately not called `facts.jsonl`. It holds *observations* — the
things a system is handed, one of which may be wrong and several of which are superseded
later. The facts are in the gold answers, and conflating the two is the mistake that
makes a memory benchmark score its own input.

## Every adapter receives the same event, structure included

A `MemoryEvent` carries both a sentence and a `(subject, predicate, object)` triple, and
every adapter is handed both. That is a deliberate scope limit, not an oversight: it
holds *extraction quality* constant so that what is measured is what a system does with a
fact once it has it. A benchmark that handed raw text to everyone would be measuring
whichever extractor happened to be configured, and the temporal result would move when
nobody touched the temporal code. `README.md` states this under *Limitations*, because it
is the largest one.

## The two clocks are in the data, not in the reader's head

Every event has `recorded_at` — when the system is told — and `valid_from` — when the
fact became true in the world. They are usually equal and sometimes are not, which is the
whole of the `knowledge_time` category. A system with one clock has to pick one, and the
questions will find out which.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from . import DEFAULT_DATASET

#: Where the shipped datasets live. A caller may point `load()` anywhere else.
DATASETS = Path(__file__).resolve().parent / "datasets"

#: The question categories, and the order they are reported in. A question naming
#: anything else is a load error rather than an uncounted row — an unknown category
#: silently scoring nothing is the failure mode this list exists to prevent.
CATEGORIES: tuple[str, ...] = (
    "current_state",
    "historical_state",
    "change_detection",
    "change_time",
    "knowledge_time",
    "provenance",
    "contradiction",
    "multi_hop",
    "distractor",
    "negative",
)

#: How a gold answer is compared. `value` is one string, `set` is an unordered group of
#: them, `date` is a calendar day, and `none` means the system is expected to say it does
#: not know. See `benchmarks.agent_memory.scoring`.
ANSWER_KINDS: tuple[str, ...] = ("value", "set", "date", "none")


def parse_instant(text: str) -> datetime:
    """Read an ISO-8601 instant from the dataset, always as UTC.

    A naive timestamp is read as UTC rather than as local time, so a run in Berlin and a
    run in San Francisco score the same dataset identically.

    >>> parse_instant("2026-03-15").isoformat()
    '2026-03-15T00:00:00+00:00'
    >>> parse_instant("2026-03-15T09:30:00Z").isoformat()
    '2026-03-15T09:30:00+00:00'
    """
    raw = text.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    when = datetime.fromisoformat(raw)
    return when.replace(tzinfo=timezone.utc) if when.tzinfo is None else when.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class PredicateDecl:
    """One relation, and how many values it holds at once.

    Published with the dataset and handed to every adapter, which is what keeps it
    fair. Cardinality decides whether a second value supersedes the first or joins it,
    and a system that had to guess would be measured on its guess rather than on its
    memory. Guessing is a real and interesting problem; it is not this benchmark's.
    """

    name: str
    #: `"one"` — a new value replaces the old. `"many"` — values accumulate.
    cardinality: str = "one"
    #: How fast the fact goes stale: `"static"`, `"slow"` or `"fast"`. Advisory. A system
    #: with no notion of decay ignores it and loses nothing on any scored question.
    volatility: str = "slow"

    @property
    def single_valued(self) -> bool:
        return self.cardinality == "one"


@dataclass(frozen=True, slots=True)
class MemoryEvent:
    """One thing the memory system is told.

    `recorded_at` is when it is told; `valid_from` is when the fact began to hold in the
    world. `valid_to` is set only where the dataset says the fact stops holding without
    anything replacing it — a supersession leaves it `None`, because the successor is
    what ends it and a system that cannot work that out should lose the point.
    """

    id: str
    recorded_at: datetime
    valid_from: datetime
    subject: str
    predicate: str
    object: str
    #: The sentence a person would have said. Indexed by the retrieval baselines, and by
    #: memvara, which embeds it rather than the rendered triple.
    text: str
    #: Who said it — `"alice"`, `"hr_directory"`, `"deploy_log"`. The gold answer to every
    #: `provenance` question is one of these labels, so provenance is answerable by any
    #: system that stores the label beside the value. It requires nothing memvara-shaped.
    source: str
    #: The reporter's reliability in [0, 1]. Used by the `contradiction` scenarios where
    #: two sources disagree; a system with no confidence notion ignores it.
    confidence: float = 1.0
    valid_to: datetime | None = None
    #: The scenario this belongs to. Runners keep scenarios in separate memory scopes so
    #: that one scenario's Alice is not another's.
    scenario: str = "misc"

    @classmethod
    def from_json(cls, row: Mapping[str, Any]) -> "MemoryEvent":
        return cls(
            id=str(row["id"]),
            recorded_at=parse_instant(row["recorded_at"]),
            valid_from=parse_instant(row["valid_from"]),
            valid_to=parse_instant(row["valid_to"]) if row.get("valid_to") else None,
            subject=str(row["subject"]),
            predicate=str(row["predicate"]),
            object=str(row["object"]),
            text=str(row["text"]),
            source=str(row["source"]),
            confidence=float(row.get("confidence", 1.0)),
            scenario=str(row.get("scenario", "misc")),
        )

    def to_json(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "id": self.id,
            "scenario": self.scenario,
            "recorded_at": self.recorded_at.isoformat(),
            "valid_from": self.valid_from.isoformat(),
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "text": self.text,
            "source": self.source,
            "confidence": self.confidence,
        }
        if self.valid_to is not None:
            row["valid_to"] = self.valid_to.isoformat()
        return row


@dataclass(frozen=True, slots=True)
class Gold:
    """The right answer, and how to tell whether it was given.

    `aliases` are additional spellings that count as correct — `"NYC"` for
    `"New York"`. They are part of the published dataset so that a system is never
    marked wrong for formatting, and so that nobody can add one after seeing a result.
    """

    kind: str
    value: str | None = None
    values: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()

    @classmethod
    def from_json(cls, row: Mapping[str, Any]) -> "Gold":
        kind = str(row["kind"])
        if kind not in ANSWER_KINDS:
            raise ValueError(f"unknown answer kind {kind!r}; expected one of {ANSWER_KINDS}")
        return cls(
            kind=kind,
            value=None if row.get("value") is None else str(row["value"]),
            values=tuple(str(v) for v in row.get("values", ())),
            aliases=tuple(str(a) for a in row.get("aliases", ())),
        )

    def to_json(self) -> dict[str, Any]:
        row: dict[str, Any] = {"kind": self.kind}
        if self.value is not None:
            row["value"] = self.value
        if self.values:
            row["values"] = list(self.values)
        if self.aliases:
            row["aliases"] = list(self.aliases)
        return row


@dataclass(frozen=True, slots=True)
class Question:
    """One question, with everything needed to score it without a model.

    `probe` is the fact slot the question is about, as `(subject, predicate)`, or `None`.
    Where it is given, every system gets it and the question measures the temporal model
    rather than the ranker. Where it is `None` the system has to find the slot itself
    from the wording, which is what the `multi_hop`, `distractor` and `negative`
    categories are for. Which questions carry a probe is fixed by the dataset, so no
    system can choose the easier path on a question another system had to search for.
    """

    id: str
    category: str
    question: str
    gold: Gold
    scenario: str = "misc"
    probe: tuple[str, str] | None = None
    #: The world instant the question is about — `valid_at`. `None` means "now", which
    #: the runner resolves to the dataset's `evaluated_at`.
    at: datetime | None = None
    #: The belief instant — `known_at`. Set only where the question asks what the system
    #: would have said at a past moment rather than what it believes today.
    known_at: datetime | None = None
    #: Which *value* the question is about, where naming it is the difference between a
    #: memory question and a reading-comprehension one. "Which source reported that Alice
    #: lives in London?" carries `about="London"`; without it a system would have to parse
    #: the sentence to know which of three cities was meant, and the benchmark would be
    #: scoring its parser. Every system gets this field, and only `provenance`,
    #: `change_time` and `knowledge_time` questions carry it.
    about: str | None = None
    #: Free text for the failure report. Never scored.
    note: str = ""

    @classmethod
    def from_json(cls, row: Mapping[str, Any]) -> "Question":
        category = str(row["category"])
        if category not in CATEGORIES:
            raise ValueError(f"unknown category {category!r}; expected one of {CATEGORIES}")
        probe = row.get("probe")
        return cls(
            id=str(row["id"]),
            category=category,
            question=str(row["question"]),
            gold=Gold.from_json(row["gold"]),
            scenario=str(row.get("scenario", "misc")),
            probe=(str(probe[0]), str(probe[1])) if probe else None,
            at=parse_instant(row["at"]) if row.get("at") else None,
            known_at=parse_instant(row["known_at"]) if row.get("known_at") else None,
            about=str(row["about"]) if row.get("about") else None,
            note=str(row.get("note", "")),
        )

    def to_json(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "id": self.id,
            "scenario": self.scenario,
            "category": self.category,
            "question": self.question,
        }
        if self.probe is not None:
            row["probe"] = list(self.probe)
        if self.at is not None:
            row["at"] = self.at.isoformat()
        if self.known_at is not None:
            row["known_at"] = self.known_at.isoformat()
        if self.about is not None:
            row["about"] = self.about
        row["gold"] = self.gold.to_json()
        if self.note:
            row["note"] = self.note
        return row


@dataclass(frozen=True, slots=True)
class Dataset:
    """A loaded dataset: the schema, the events in delivery order, and the questions."""

    version: str
    #: The instant "now" resolves to. Fixed in the data rather than read from the clock,
    #: because a benchmark whose answers change overnight cannot be reproduced.
    evaluated_at: datetime
    predicates: Mapping[str, PredicateDecl]
    events: tuple[MemoryEvent, ...]
    questions: tuple[Question, ...]
    #: Which categories make up each reported dimension, in report order.
    dimensions: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    description: str = ""

    @property
    def scenarios(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for event in self.events:
            seen.setdefault(event.scenario, None)
        return tuple(seen)

    def events_for(self, scenario: str) -> tuple[MemoryEvent, ...]:
        return tuple(e for e in self.events if e.scenario == scenario)

    def questions_for(self, scenario: str) -> tuple[Question, ...]:
        return tuple(q for q in self.questions if q.scenario == scenario)

    def filter(self, *, categories: Sequence[str] | None = None,
               limit: int | None = None) -> "Dataset":
        """A narrower view for a quick run, keeping every event.

        Events are never dropped. Filtering them would change what each system was told
        and make a partial run's numbers incomparable with a full one's, which is exactly
        the confusion a `--quick` flag invites. A filtered run reports fewer questions
        over the same memory.

        `limit` takes the first N questions of the already-category-filtered set, in
        dataset order. Dataset order is round-robin across categories (see
        `datasets/build_v1.py`), so a limit is a spread rather than a slice of one
        category.
        """
        picked = self.questions
        if categories:
            wanted = set(categories)
            unknown = wanted - set(CATEGORIES)
            if unknown:
                raise ValueError(
                    f"unknown categor{'y' if len(unknown) == 1 else 'ies'} "
                    f"{sorted(unknown)}; expected from {list(CATEGORIES)}")
            picked = tuple(q for q in picked if q.category in wanted)
        if limit is not None:
            picked = picked[:limit]
        return Dataset(version=self.version, evaluated_at=self.evaluated_at,
                       predicates=self.predicates, events=self.events,
                       questions=picked, dimensions=self.dimensions,
                       description=self.description)


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:      # pragma: no cover - defensive
                raise ValueError(f"{path}:{number}: {exc}") from exc


def load(version: str = DEFAULT_DATASET, root: Path | None = None) -> Dataset:
    """Load a shipped dataset by version, validating as it goes.

    Validation is not a formality here. The defects a benchmark can carry that are
    invisible in a passing run are load-time checks, because there is no later moment at
    which they announce themselves: a question probing a slot no event ever wrote, a
    duplicate question id quietly overwriting another, a predicate used by an event but
    absent from the declared schema — which would hand one system a cardinality and leave
    the next to guess — and a question filed under a scenario with no events, which turns
    off the lenient rule's ambiguity check for that question alone. `validate` has the
    whole list; it is deliberately not restated here as a number, because the number went
    stale the first time the list grew.
    """
    base = (root or DATASETS) / version
    meta = json.loads((base / "metadata.json").read_text(encoding="utf-8"))
    predicates = {
        name: PredicateDecl(name=name, cardinality=spec.get("cardinality", "one"),
                            volatility=spec.get("volatility", "slow"))
        for name, spec in meta["predicates"].items()
    }
    events = tuple(MemoryEvent.from_json(row) for row in _read_jsonl(base / "events.jsonl"))
    questions = tuple(Question.from_json(row) for row in _read_jsonl(base / "questions.jsonl"))
    dataset = Dataset(
        version=str(meta["version"]),
        evaluated_at=parse_instant(meta["evaluated_at"]),
        predicates=predicates,
        events=events,
        questions=questions,
        dimensions={k: tuple(v) for k, v in meta.get("dimensions", {}).items()},
        description=str(meta.get("description", "")),
    )
    validate(dataset)
    return dataset


def validate(dataset: Dataset) -> None:
    """Raise on anything that would make a score wrong rather than low."""
    seen_events: set[str] = set()
    for event in dataset.events:
        if event.id in seen_events:
            raise ValueError(f"duplicate event id {event.id!r}")
        seen_events.add(event.id)
        if event.predicate not in dataset.predicates:
            raise ValueError(
                f"event {event.id!r} uses predicate {event.predicate!r}, which "
                "metadata.json does not declare; every adapter would have to guess its "
                "cardinality and they would not all guess alike")
        if event.valid_to is not None and event.valid_to <= event.valid_from:
            raise ValueError(
                f"event {event.id!r} has valid_to <= valid_from, so it holds at no "
                "instant and no query on either clock can return it")

    slots = {(e.subject, e.predicate) for e in dataset.events}
    populated = {e.scenario for e in dataset.events}
    seen_questions: set[str] = set()
    for question in dataset.questions:
        if question.id in seen_questions:
            raise ValueError(f"duplicate question id {question.id!r}")
        seen_questions.add(question.id)
        if question.scenario not in populated:
            # Not cosmetic. `timeline.Truth.competitors` answers an unprobed question
            # with *every value in its scenario*, and that set is built from events —
            # so a question filed under a scenario with none gets an empty competitor
            # set, and the lenient rule's ambiguity check silently stops running. It
            # then accepts an answer naming two candidate values, which is exactly the
            # abuse `normalization` was written to refuse. Dataset v2 shipped twelve
            # chained questions this way and they were graded more leniently than the
            # six v1 questions beside them in the same category.
            raise ValueError(
                f"question {question.id!r} is filed under scenario "
                f"{question.scenario!r}, which no event writes to. An unprobed question "
                "there has no competing values, so `--match lenient` would accept an "
                "ambiguous answer that the same question in a populated scenario would "
                "refuse. File it under the scenario whose events it is about.")
        if question.probe is not None and question.probe not in slots:
            # A `negative` question may legitimately probe an empty slot; anything else
            # probing one is asking about a fact the dataset never delivered.
            if question.category != "negative":
                raise ValueError(
                    f"question {question.id!r} probes {question.probe}, which no event "
                    "writes; it can only ever be scored wrong")
        if question.gold.kind == "set" and not question.gold.values:
            raise ValueError(f"question {question.id!r} is set-valued with no gold values")
        if question.gold.kind == "set" and question.gold.aliases:
            raise ValueError(
                f"question {question.id!r} is set-valued and publishes aliases, which "
                "the scorer does not honour: a flat alias list cannot say which member "
                "it stands in for. See normalization.matches_set")
        if question.gold.kind in ("value", "date") and question.gold.value is None:
            raise ValueError(f"question {question.id!r} is {question.gold.kind}-valued with no gold value")
        needs_about = question.category == "provenance" or question.gold.kind == "date"
        if needs_about and question.about is None:
            raise ValueError(
                f"question {question.id!r} asks for a source or a date and does not name "
                "the value it is about, so which fact it means is only in the prose; "
                "every system would have to parse the sentence and the benchmark would "
                "be scoring parsers. Set `about`.")
        if question.gold.kind == "date" and question.probe is not None:
            starts = {e.valid_from for e in dataset.events
                      if (e.subject, e.predicate) == question.probe
                      and e.object == question.about}
            if len(starts) > 1:
                raise ValueError(
                    f"question {question.id!r} asks for a date about "
                    f"{question.about!r}, which {question.probe[0]} held over "
                    f"{len(starts)} separate intervals; `about` cannot say which one is "
                    "meant, so the question has more than one defensible answer")
        if question.gold.kind == "none" and (question.gold.value or question.gold.values):
            raise ValueError(
                f"question {question.id!r} expects an abstention but carries a gold "
                "value, so a system could be marked wrong for agreeing with it")
