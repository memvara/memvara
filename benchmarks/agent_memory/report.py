"""Rendering a run for a person to read.

Two outputs. The scorecard is the table; the failure report is the part that is actually
useful while you are fixing something. A benchmark that prints `73.0%` and stops has told
a developer that there is a problem and nothing about where it is.

Only measured quantities are printed. An unmeasured cost counter renders as `-`, never as
`0`, and a category with no questions renders as `-`, never as `0.0%`. The two mistakes
this rule prevents are the same mistake: a number that reads as a measurement and is not
one.
"""

from __future__ import annotations

from typing import Sequence

from . import BENCHMARK_VERSION
from .dataset import CATEGORIES, Dataset
from .results import RunResult
from .scoring import Judgement, Tally
from .timeline import Truth

RULE = "─" * 68

#: One line each, printed under the failure report so a reader does not have to infer
#: what a reason means from its name.
REASON_HELP: dict[str, str] = {
    "abstained": "returned no answer where the dataset has one",
    "answered_when_it_should_abstain": "answered a question about a fact it was never told",
    "answered_current_state": "gave today's value to a question about a past instant",
    "answered_stale_state": "gave a superseded value to a question about now",
    "answered_other_interval": "gave a value this slot held, but not at the instant asked",
    "answered_before_it_knew": "crossed the two clocks: gave one clock's instant for the other's",
    "answered_other_entity": "gave a value belonging to a different subject",
    "over_answered": "returned more values than the question has answers",
    "under_answered": "returned part of the gold set",
    "wrong_set": "returned a set that overlaps the gold set but is neither",
    "unparsable_date": "answered, but not as an ISO-8601 date",
    "unknown_value": "returned a value that appears nowhere in the scenario",
    "wrong_value": "returned a real value, for no reason the dataset can name",
}


def pct(tally: Tally) -> str:
    accuracy = tally.accuracy
    return "-" if accuracy is None else f"{accuracy * 100:5.1f}%"


def _row(label: str, tally: Tally, width: int) -> str:
    counts = f"{tally.correct}/{tally.total}" if tally.total else ""
    return f"  {label:<{width}}  {pct(tally):>7}  {counts:>8}"


def scorecard(result: RunResult, dataset: Dataset) -> str:
    """The table. Dimensions first, because that is what the categories are for."""
    lines = [
        "",
        f"Agent Memory Benchmark v{BENCHMARK_VERSION}   dataset {result.dataset_version}",
        f"System: {result.system} {result.system_version}",
        f"{result.counts.get('events', 0)} events, {result.counts.get('questions', 0)} "
        f"questions, {result.counts.get('scenarios', 0)} scenarios",
        RULE,
    ]
    width = max([len(d) for d in result.scorecard.by_dimension] + [len(c) for c in CATEGORIES] + [12])

    if result.scorecard.by_dimension:
        lines.append("  Dimension")
        for name, tally in result.scorecard.by_dimension.items():
            lines.append(_row(name, tally, width))
        lines.append("")

    lines.append("  Category")
    for category in CATEGORIES:
        lines.append(_row(category, result.scorecard.by_category[category], width))

    lines += [RULE, _row("OVERALL", result.scorecard.overall, width), RULE, ""]
    lines += _timing_block(result)
    lines += _cost_block(result)
    if result.scorecard.reasons:
        lines.append("  Failures by reason")
        for reason, count in result.scorecard.reasons.items():
            lines.append(f"    {count:>3}  {reason:<32} {REASON_HELP.get(reason, '')}")
        lines.append("")
    return "\n".join(lines)


def _timing_block(result: RunResult) -> list[str]:
    latency = result.latency
    return [
        "  Latency",
        f"    write, mean per event        {latency.write_mean_ms:8.3f} ms",
        f"    write, whole corpus          {latency.write_total_ms:8.1f} ms",
        f"    query, mean                  {latency.query_mean_ms:8.3f} ms",
        f"    query, p50 / p95 / max       {latency.query_p50_ms:.3f} / "
        f"{latency.query_p95_ms:.3f} / {latency.query_max_ms:.3f} ms",
        "",
    ]


def _cost_block(result: RunResult) -> list[str]:
    usage = result.usage

    def show(value: int | None) -> str:
        return "-" if value is None else f"{value}"

    lines = [
        "  Cost, as counted by the system itself",
        f"    LLM calls                    {show(usage.llm_calls):>8}",
        f"    texts embedded               {show(usage.texts_embedded):>8}",
        f"    rows stored                  {show(usage.rows_stored):>8}",
        f"    read calls                   {show(usage.db_reads):>8}",
    ]
    for key, value in sorted(usage.extra.items()):
        lines.append(f"    {key:<28} {value:>8}")
    lines += ["    '-' means not measured, not zero.", ""]
    return lines


def failure_report(judgements: Sequence[Judgement], dataset: Dataset,
                   truth: Truth) -> str:
    """Every wrong answer, with the slot's history and a named reason.

    The timeline is printed from the dataset, not from the system: it is what the system
    *should* have been able to reconstruct, with the instant asked about marked in it.
    Seeing the answer beside the timeline is usually the whole diagnosis.
    """
    if not judgements:
        return "\nNo failures.\n"
    by_id = {q.id: q for q in dataset.questions}
    blocks = ["", f"{len(judgements)} failure(s)", RULE]
    for judgement in judgements:
        question = by_id[judgement.question_id]
        blocks += [
            "",
            f"FAIL  {judgement.question_id}   [{judgement.category}]",
            f"  Question:  {question.question}",
            f"  Expected:  {judgement.expected}",
            f"  Answered:  {judgement.given if judgement.given is not None else '(nothing)'}",
            f"  Reason:    {judgement.reason} — {REASON_HELP.get(judgement.reason or '', '')}",
        ]
        if question.probe is not None:
            timeline = truth.slot(*question.probe)
            if timeline is not None:
                blocks.append(f"  Timeline for {question.probe[0]} / {question.probe[1]}:")
                blocks += timeline.render(question.at or dataset.evaluated_at)
        if question.known_at is not None:
            blocks.append(f"  Asked as the record stood on "
                          f"{question.known_at.date().isoformat()}.")
        if question.note:
            blocks.append(f"  Note:      {question.note}")
    blocks.append("")
    return "\n".join(blocks)


def leaderboard(results: Sequence[RunResult]) -> str:
    """Several systems, one table. The shape a public benchmark page would render.

    Ordered by overall accuracy, highest first. Ties keep the order they were run in,
    which is the order the caller named them.
    """
    if not results:
        return ""
    dimensions = list(results[0].scorecard.by_dimension)
    # Width from the data, not a constant: slicing to 13 rendered `knowledge_time` as
    # `knowledge_tim`, so the primary output disagreed with every table in the docs about
    # the name of one of the seven things being measured.
    column = max((len(d) for d in dimensions), default=8)
    header = (f"  {'System':<16} {'Overall':>8}"
              + "".join(f" {d:>{column}}" for d in dimensions))
    lines = ["", f"Agent Memory Benchmark v{BENCHMARK_VERSION}", RULE, header, RULE]
    for result in sorted(results, key=lambda r: -(r.scorecard.overall.accuracy or 0.0)):
        row = f"  {result.system:<16} {pct(result.scorecard.overall):>8}"
        for dimension in dimensions:
            tally = result.scorecard.by_dimension.get(dimension)
            row += f" {(pct(tally) if tally else '-'):>{column}}"
        lines.append(row)
    lines += [RULE, ""]
    return "\n".join(lines)
