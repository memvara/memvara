"""Scoring: one rule per answer kind, and a named reason for every failure.

Accuracy here is `correct / total`, per category and overall. There is no weighting, no
partial credit and no composite index, for one reason: a single number that mixes
temporal reasoning with retrieval hides exactly the thing a reader wants — which half a
system is bad at. `docs/benchmarks/agent-memory-benchmark.md` reports the categories and
prints the overall figure last.

Partial credit is not awarded anywhere. A set answer is right when it is the gold set and
wrong otherwise; a date answer is right on the right calendar day. Half-credit for a
half-remembered fact would be a defensible choice in a different benchmark, and it is not
one here because an agent that half-remembers a customer's plan tells the customer
something false.

## The failure reason

Every wrong answer gets a `reason` from a closed list, derived from the dataset rather
than inferred. `answered_current_state` — the system gave today's value to a question
about March — is the finding this whole benchmark exists to surface, and it is worth more
to a developer than the score is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Sequence

from .adapters.base import MemoryAnswer
from .dataset import CATEGORIES, Dataset, Question
from .normalization import matches_date, matches_set, matches_value, normalize, parse_date
from .timeline import Truth

#: Every reason a question can be marked wrong. Closed so that the report cannot grow a
#: silent "other" bucket, and so a reader can be told what each one means once.
REASONS: tuple[str, ...] = (
    "abstained",                    # no answer where one was expected
    "answered_when_it_should_abstain",
    "answered_current_state",       # gave today's value to a question about the past
    "answered_stale_state",         # gave a superseded value to a question about now
    "answered_other_interval",      # a value this slot held, but not at the instant asked
    "answered_before_it_knew",      # right about the world, wrong about what it had heard
    "answered_other_entity",        # a value belonging to a different subject
    "over_answered",                # set answer is a strict superset of the gold
    "under_answered",               # set answer is a strict subset of the gold
    "wrong_set",                    # set answer overlaps the gold but is neither
    "unparsable_date",              # answered, but not in ISO-8601
    "unknown_value",                # a value that appears nowhere in the scenario
    "wrong_value",                  # a real value, none of the above
)


@dataclass(frozen=True, slots=True)
class Judgement:
    """One question, scored."""

    question_id: str
    category: str
    scenario: str
    correct: bool
    #: What the system said, flattened for the report. `None` for an abstention.
    given: str | None
    #: What the dataset expected, flattened the same way.
    expected: str
    reason: str | None
    #: Seconds spent inside the adapter's `query`, excluding everything else.
    latency_s: float = 0.0
    support: tuple[str, ...] = ()

    def to_json(self) -> dict[str, object]:
        return {
            "id": self.question_id, "category": self.category, "scenario": self.scenario,
            "correct": self.correct, "given": self.given, "expected": self.expected,
            "reason": self.reason, "latency_s": round(self.latency_s, 6),
            "support": list(self.support),
        }


def flatten(answer: MemoryAnswer) -> str | None:
    """The answer as one string for the report, or `None` if it abstained."""
    if answer.values:
        return ", ".join(answer.values)
    return answer.value


def expected_text(question: Question) -> str:
    gold = question.gold
    if gold.kind == "none":
        return "(no answer)"
    if gold.kind == "set":
        return ", ".join(gold.values)
    return gold.value or ""


def grade(question: Question, answer: MemoryAnswer, truth: Truth, *,
          lenient: bool = True) -> bool:
    """Is this answer correct? One rule per gold kind, no model involved."""
    gold = question.gold
    if gold.kind == "none":
        return answer.abstained
    if answer.abstained:
        return False
    if gold.kind == "set":
        values = answer.values or ((answer.value,) if answer.value else ())
        return matches_set([v for v in values if v], gold.values)
    if gold.kind == "date":
        return matches_date(answer.value, gold.value or "")
    competitors = truth.competitors(question.probe, question.scenario)
    return matches_value(answer.value, gold.value or "", gold.aliases, competitors,
                         lenient=lenient)


def diagnose(question: Question, answer: MemoryAnswer, truth: Truth,
             evaluated_at: datetime) -> str:
    """Name why an answer was wrong, from the dataset alone.

    Every branch is a lookup against the timelines built from `events.jsonl`. Nothing
    here asks the system under test why it said what it said, because a system that could
    answer that reliably would not have got the question wrong.
    """
    gold = question.gold
    if gold.kind == "none":
        return "answered_when_it_should_abstain"
    if answer.abstained:
        return "abstained"
    if gold.kind in ("value", "date") and len(answer.values) > 1:
        # A set where one value was asked for. Without this branch the reason is derived
        # from `answer.value`, which is `None`, and the report says `unknown_value` about
        # an answer that named several values the dataset knows perfectly well.
        return "over_answered"

    if gold.kind == "set":
        given = {normalize(v) for v in (answer.values or ((answer.value,) if answer.value else ())) if v}
        want = {normalize(v) for v in gold.values}
        if given > want:
            return "over_answered"
        if given < want:
            return "under_answered"
        return "wrong_set"

    if gold.kind == "date":
        if parse_date(answer.value) is None:
            return "unparsable_date"
        return _date_reason(question, answer, truth)

    return _value_reason(question, answer, truth, evaluated_at)


def _date_reason(question: Question, answer: MemoryAnswer, truth: Truth) -> str:
    """A parsable but wrong date. The interesting case is the two clocks crossed."""
    given = parse_date(answer.value)
    if question.probe is not None and given is not None:
        timeline = truth.slot(*question.probe)
        if timeline is not None:
            for version in timeline.versions:
                # `change_time` asks when the fact became true, `knowledge_time` when we
                # heard. Answering the other clock's instant is the specific confusion,
                # and naming it is worth far more than "wrong date".
                if question.category == "knowledge_time" and version.valid_from.date() == given:
                    return "answered_before_it_knew"
                if question.category == "change_time" and version.recorded_at.date() == given:
                    return "answered_before_it_knew"
    return "wrong_value"


def _value_reason(question: Question, answer: MemoryAnswer, truth: Truth,
                  evaluated_at: datetime) -> str:
    given = answer.value or ""
    normalized = normalize(given)
    # `sorted` because `known_values` is a frozenset and string hashing is randomised
    # per process: without it, two values normalizing to one key would let the winner
    # depend on the hash seed, and `reason` is compared by `results.comparable`.
    scenario_values = {normalize(v): v for v in sorted(truth.known_values(question.scenario))}
    if normalized not in scenario_values:
        return "unknown_value"

    if question.probe is not None:
        timeline = truth.slot(*question.probe)
        if timeline is not None:
            asked_at = question.at or evaluated_at
            current = timeline.values_at(evaluated_at)
            if normalized in {normalize(v) for v in current} and asked_at < evaluated_at:
                return "answered_current_state"
            if normalized in {normalize(v) for v in timeline.objects}:
                if asked_at >= evaluated_at:
                    return "answered_stale_state"
                if question.known_at is not None:
                    # The value was right about the world and wrong about what had been
                    # heard by the belief instant — the delayed-knowledge failure.
                    believed = timeline.values_at(asked_at, known_at=question.known_at)
                    if normalized not in {normalize(v) for v in believed}:
                        return "answered_before_it_knew"
                return "answered_other_interval"
            others = truth.other_subjects_holding(
                scenario_values[normalized], question.probe[1], question.scenario,
                excluding=question.probe[0])
            if others:
                return "answered_other_entity"
    return "wrong_value"


def judge(question: Question, answer: MemoryAnswer, truth: Truth, evaluated_at: datetime,
          *, lenient: bool = True, latency_s: float = 0.0) -> Judgement:
    correct = grade(question, answer, truth, lenient=lenient)
    return Judgement(
        question_id=question.id, category=question.category, scenario=question.scenario,
        correct=correct, given=flatten(answer), expected=expected_text(question),
        reason=None if correct else diagnose(question, answer, truth, evaluated_at),
        latency_s=latency_s, support=answer.support,
    )


@dataclass(frozen=True, slots=True)
class Tally:
    """`correct / total` for one group, and nothing else."""

    correct: int
    total: int

    @property
    def accuracy(self) -> float | None:
        """`None` when the group is empty, so an unasked category cannot print 0%."""
        return self.correct / self.total if self.total else None

    def to_json(self) -> dict[str, object]:
        return {"correct": self.correct, "total": self.total, "accuracy": self.accuracy}


@dataclass(frozen=True, slots=True)
class Scorecard:
    """Every accuracy figure a run produces."""

    overall: Tally
    by_category: Mapping[str, Tally]
    by_dimension: Mapping[str, Tally]
    by_scenario: Mapping[str, Tally]
    #: How often each failure reason fired, most common first.
    reasons: Mapping[str, int] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "overall": self.overall.to_json(),
            "by_category": {k: v.to_json() for k, v in self.by_category.items()},
            "by_dimension": {k: v.to_json() for k, v in self.by_dimension.items()},
            "by_scenario": {k: v.to_json() for k, v in self.by_scenario.items()},
            "failure_reasons": dict(self.reasons),
        }


def _tally(judgements: Sequence[Judgement]) -> Tally:
    return Tally(correct=sum(1 for j in judgements if j.correct), total=len(judgements))


def score(judgements: Sequence[Judgement], dataset: Dataset) -> Scorecard:
    """Aggregate judgements into the figures the report and the JSON both use.

    Empty groups are kept with `total=0` rather than dropped, so a run limited to one
    category still shows which categories exist and that they were not asked. A missing
    row and a zero row mean different things and neither should be inferred.
    """
    asked = {q.id for q in dataset.questions}
    judgements = [j for j in judgements if j.question_id in asked]

    by_category = {
        category: _tally([j for j in judgements if j.category == category])
        for category in CATEGORIES
    }
    by_dimension = {}
    for dimension, categories in dataset.dimensions.items():
        members = [j for j in judgements if j.category in set(categories)]
        by_dimension[dimension] = _tally(members)
    by_scenario = {
        scenario: _tally([j for j in judgements if j.scenario == scenario])
        for scenario in sorted({j.scenario for j in judgements})
    }

    counts: dict[str, int] = {}
    for judgement in judgements:
        if judgement.reason:
            counts[judgement.reason] = counts.get(judgement.reason, 0) + 1
    reasons = dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    return Scorecard(overall=_tally(judgements), by_category=by_category,
                     by_dimension=by_dimension, by_scenario=by_scenario, reasons=reasons)
