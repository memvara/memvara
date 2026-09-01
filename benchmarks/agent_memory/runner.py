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

Ingestion is timed once. Repeating it means emptying the store and refilling it, and the
second fill is no longer the cold path the number is supposed to describe.

Queries can be timed more than once: `--latency-repeats N` asks the whole question set N
times. Only the first pass is scored and only the first pass is charged for, and the
timings come from the passes after it. A single timing of a hundred questions on a laptop
with a browser open measures the machine's mood as much as the system, and the repeat
spread reported beside the percentiles is what lets a reader see which they are looking
at.
"""

from __future__ import annotations

import time
from dataclasses import replace
from datetime import datetime, timezone
from statistics import median_low
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


def _one_pass(system: MemorySystem, dataset: Dataset) -> tuple[list[MemoryAnswer],
                                                                list[float]]:
    """Ask every question once. Answers and per-question seconds, in dataset order."""
    answers: list[MemoryAnswer] = []
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
        answers.append(answer)
        times.append(elapsed)
    return answers, times


def interrogate(system: MemorySystem, dataset: Dataset, truth: Truth, *,
                lenient: bool = True) -> tuple[list[Judgement], list[float]]:
    """Ask and score every question once. This is the pass the scores come from."""
    answers, times = _one_pass(system, dataset)
    judgements = [
        judge(question, answer, truth, dataset.evaluated_at,
              lenient=lenient, latency_s=elapsed)
        for question, answer, elapsed in zip(dataset.questions, answers, times)
    ]
    return judgements, times


def retime(system: MemorySystem, dataset: Dataset, passes: int) -> list[list[float]]:
    """Ask every question `passes` more times, for timing only. Nothing is scored."""
    return [_one_pass(system, dataset)[1] for _ in range(passes)]


def restamp(judgements: Sequence[Judgement],
            passes: Sequence[Sequence[float]]) -> list[Judgement]:
    """Re-time each judgement from `passes`, keeping every verdict as it was scored.

    The median across passes rather than the mean, matching `query_p50_ms`: one slow
    question on one pass should not move the figure a reader compares against the block.

    `median_low`, not `median`. An even number of passes — four is what
    `--latency-repeats 5` produces — makes the plain median average the two middle
    readings into a duration nothing ever measured, which is the thing
    `results._percentile` picks nearest-rank to avoid. Every `latency_s` published here
    is a timing that actually happened.
    """
    return [
        replace(judgement, latency_s=median_low(column))
        for judgement, column in zip(judgements, zip(*passes))
    ]


def run(system: MemorySystem, dataset: Dataset, *, lenient: bool = True,
        config: Mapping[str, Any] | None = None,
        timestamp: str | None = None,
        latency_repeats: int = 1,
        progress: Callable[[int, int], None] | None = None) -> RunResult:
    """One system against one dataset. The whole benchmark, in twenty lines."""
    truth = Truth(dataset)
    write_seconds = ingest(system, dataset, progress)
    judgements, first = interrogate(system, dataset, truth, lenient=lenient)
    # Read here, between the scored pass and any repeat, and the order is the point: the
    # cost columns describe one pass over the dataset however many times the questions
    # are asked for timing. Read after the repeats, `db_reads` would multiply by
    # `--latency-repeats` and a system would look more expensive because the benchmark
    # got more careful about its clock.
    usage: Usage = system.usage()
    # With repeats, the scored pass is discarded from the timings: it is where a system
    # does whatever it deferred — `vector-rag` builds its index on first search — and
    # that belongs to the cold path rather than to a warm-path p95.
    extra = retime(system, dataset, latency_repeats - 1) if latency_repeats > 1 else []
    times = extra or [first]
    if extra:
        # And the per-question figures move with them. `questions[].latency_s` used to
        # keep the discarded cold pass while the summary block reported the warm ones,
        # so one result file published two answers to "how long does a query take" —
        # measured at 0.38 ms against 0.09 ms on the same run. Whatever a reader
        # averages has to agree with what the block above it says.
        judgements = restamp(judgements, times)
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
