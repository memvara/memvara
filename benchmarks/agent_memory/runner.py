"""The run loop: deliver every event, ask every question, score, time.

Deliberately dull. Every decision that could favour one system is made in the dataset or
in `scoring.py`, and this module's only job is to hand each system the same input in the
same order and measure what comes back.

## One memory, not one per scenario

Every scenario's events go into a single memory, and the questions are asked against all
262 of them. Scenario is a label for reporting, not an isolation boundary. Entities are
globally unique across the dataset, so nothing collides — and a benchmark that gave each
scenario its own three-fact store would make retrieval trivial and would measure nothing
about working at any size at all.

## What is timed, and what is not

The clock starts and stops around the adapter call and nothing else. Building the system,
loading the dataset, judging the answer and rendering the report are all outside it.
`time.perf_counter` throughout — the monotonic one, so a clock adjustment mid-run cannot
produce a negative interval.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from .adapters.base import Ask, MemoryAnswer, MemorySystem, Usage, sort_events
from .dataset import Dataset, Question
from .results import RunResult, latency_of
from .scoring import Judgement, judge, score
from .timeline import Truth


def ask_of(question: Question, evaluated_at: datetime) -> Ask:
    """The question as an adapter sees it, with the gold answer removed.

    `at` is resolved here rather than left `None`, so an adapter never has to know what
    the dataset's "now" is — and cannot accidentally use the wall clock, which would make
    its answers depend on the day the benchmark was run.
    """
    return Ask(id=question.id, category=question.category, question=question.question,
               probe=question.probe, at=question.at or evaluated_at,
               evaluated_at=evaluated_at, known_at=question.known_at,
               about=question.about)


def ingest(system: MemorySystem, dataset: Dataset,
           progress: Callable[[int, int], None] | None = None) -> float:
    """Deliver every event in `recorded_at` order. Returns seconds spent inside the adapter."""
    events = sort_events(dataset.events)
    system.reset(dataset.predicates)
    spent = 0.0
    for index, event in enumerate(events, start=1):
        start = time.perf_counter()
        system.remember(event)
        spent += time.perf_counter() - start
        if progress is not None:
            progress(index, len(events))
    return spent


def interrogate(system: MemorySystem, dataset: Dataset, truth: Truth, *,
                lenient: bool = True) -> tuple[list[Judgement], list[float]]:
    judgements: list[Judgement] = []
    times: list[float] = []
    for question in dataset.questions:
        ask = ask_of(question, dataset.evaluated_at)
        start = time.perf_counter()
        try:
            answer = system.query(ask)
        except Exception as exc:                    # pragma: no cover - adapter defect
            raise RuntimeError(
                f"{system.name} raised on question {question.id}: {exc!r}. An adapter "
                "that cannot answer should return an empty MemoryAnswer, which is scored "
                "as an abstention.") from exc
        elapsed = time.perf_counter() - start
        if not isinstance(answer, MemoryAnswer):    # pragma: no cover - adapter defect
            raise TypeError(
                f"{system.name}.query returned {type(answer).__name__} for {question.id}; "
                "adapters must return adapters.base.MemoryAnswer.")
        times.append(elapsed)
        judgements.append(judge(question, answer, truth, dataset.evaluated_at,
                                lenient=lenient, latency_s=elapsed))
    return judgements, times


def run(system: MemorySystem, dataset: Dataset, *, lenient: bool = True,
        config: Mapping[str, Any] | None = None,
        timestamp: str | None = None,
        progress: Callable[[int, int], None] | None = None) -> RunResult:
    """One system against one dataset. The whole benchmark, in twenty lines."""
    truth = Truth(dataset)
    write_seconds = ingest(system, dataset, progress)
    judgements, times = interrogate(system, dataset, truth, lenient=lenient)
    usage: Usage = system.usage()
    return RunResult(
        system=system.name,
        system_version=system.version,
        dataset_version=dataset.version,
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
        scorecard=score(judgements, dataset),
        latency=latency_of(write_seconds, len(dataset.events), times),
        usage=usage,
        judgements=tuple(judgements),
        config=dict(config or {}),
        counts={
            "events": len(dataset.events),
            "questions": len(dataset.questions),
            "scenarios": len(dataset.scenarios),
        },
    )


def failures(result: RunResult, dataset: Dataset,
             limit: int | None = None) -> Sequence[Judgement]:
    """Wrong answers, worst category first, then in dataset order.

    Ordered so that reading the top of the list tells you which *kind* of memory failure
    dominates rather than which question happens to be first in the file.
    """
    wrong = [j for j in result.judgements if not j.correct]
    weight = {c: t.total - t.correct for c, t in result.scorecard.by_category.items()}
    order = {q.id: i for i, q in enumerate(dataset.questions)}
    wrong.sort(key=lambda j: (-weight.get(j.category, 0), j.category, order.get(j.question_id, 0)))
    return wrong[:limit] if limit is not None else wrong
