"""What the dataset says was true, and when — derived from the events, once.

Two jobs, both of which the benchmark has to do for itself. The first is to know a fact
slot's competing values, which is what makes lenient answer matching safe (see
`normalization.matches_value`). The second is to explain a failure: when a system answers
`New York` to a question about March, the useful output is not `FAIL` but the slot's
history with March marked on it.

Nothing here is read back out of any memory system. It is built from `events.jsonl`
before a single system is constructed — the same discipline `bench/temporal.py` states
for its gold sets, and for the same reason: a gold derived from the thing under test
measures only that thing's self-consistency.

## The supersession rule, stated once

Every gold answer in this benchmark follows from four sentences. They are published here
rather than left implicit, because a scoring rule a system cannot read is a rule it
cannot be expected to implement.

1. **Later valid time wins.** A value whose `valid_from` is later replaces one whose
   `valid_from` is earlier, on a single-valued predicate. The earlier value is *ended*:
   it stops being in force and it keeps answering about the period it held.
2. **At equal valid time, later record wins.** Two reports about the same instant are a
   contradiction, and the one recorded later is the correction. The earlier one is
   *retired*: we stopped believing it, so it answers nothing on the world clock at all.
3. **An ending is a belief, and it is dated.** A value stops being in force *only from
   the moment its successor was recorded*. Before that instant the store had not heard,
   and a `known_at` query rewound to before it must see the value still open-ended. This
   is the sentence a store with one clock cannot express.
4. **Repeating a value is not a change.** A second report of the same object neither ends
   the first nor adds a version to the slot's list of distinct values.

Nothing is scored on `confidence`. It is carried in the data because the conflicting-
source scenarios are more honest with it than without, and it is deliberately not part of
any gold: resolving a conflict by source reliability is a policy, and picking one here
would score systems on whether they had implemented this benchmark's policy rather than
on whether they had a memory. `README.md` lists that under *Limitations*.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence

from .dataset import Dataset, MemoryEvent


@dataclass(frozen=True, slots=True)
class Version:
    """One value a slot held, on both clocks.

    `valid_to` is when the value stopped being in force in the world.
    `ended_by_recorded_at` is when we *found out* it had — the successor's `recorded_at`,
    which is later than `valid_to` whenever the news arrived late. `retired_at` is set
    when a later record contradicted this one about the same instant, and a retired
    version answers nothing on the world clock as we understand it today.
    """

    event_id: str
    object: str
    valid_from: datetime
    valid_to: datetime | None
    ended_by_recorded_at: datetime | None
    retired_at: datetime | None
    recorded_at: datetime
    source: str
    confidence: float

    @property
    def retired(self) -> bool:
        return self.retired_at is not None

    def known_by(self, known_at: datetime | None) -> bool:
        """Had we heard this by `known_at`? `None` means "today", which has heard all."""
        return known_at is None or self.recorded_at <= known_at

    def believed_at(self, known_at: datetime | None) -> bool:
        """Did we still believe this record at `known_at`? Retirement is a belief event."""
        if not self.known_by(known_at):
            return False
        if self.retired_at is None:
            return True
        return known_at is not None and known_at < self.retired_at

    def ends_by(self, known_at: datetime | None) -> datetime | None:
        """This version's `valid_to` *as the store knew it* at `known_at`.

        `None` where the successor had not been recorded yet: on that day the value was
        in force with no end in sight, and any query rewound to it must say so.
        """
        if self.valid_to is None:
            return None
        if known_at is None or self.ended_by_recorded_at is None:
            return self.valid_to
        return self.valid_to if self.ended_by_recorded_at <= known_at else None

    def held_at(self, valid_at: datetime, known_at: datetime | None = None) -> bool:
        if valid_at < self.valid_from:
            return False
        ends = self.ends_by(known_at)
        return ends is None or valid_at < ends


class SlotTimeline:
    """Every version of one `(subject, predicate)` slot, oldest first."""

    def __init__(self, subject: str, predicate: str, versions: Sequence[Version],
                 single_valued: bool) -> None:
        self.subject = subject
        self.predicate = predicate
        self.versions = tuple(versions)
        self.single_valued = single_valued

    def values_at(self, valid_at: datetime,
                  known_at: datetime | None = None) -> tuple[str, ...]:
        """The values in force at `valid_at`, as believed at `known_at`.

        `known_at=None` means "as we understand it today", which is the default every
        question uses unless it says otherwise. For a single-valued slot the result holds
        at most one string; for a multi-valued one it holds every value covering the
        instant, which is what declaring `many` is for.
        """
        live = [v for v in self.versions
                if v.believed_at(known_at) and v.held_at(valid_at, known_at)]
        return tuple(dict.fromkeys(v.object for v in live))

    @property
    def objects(self) -> tuple[str, ...]:
        """Every value the slot has ever held, oldest first, without repeats.

        Retired versions are included: they are values the slot held in the record, and
        a system answering with one is confused about belief rather than about strings.
        `distinct_values` is the gold for `change_detection` and excludes them.
        """
        return tuple(dict.fromkeys(v.object for v in self.versions))

    @property
    def distinct_values(self) -> tuple[str, ...]:
        """The values we still believe the slot ever held, oldest first.

        Rule 4 above in one property: three reports of London make one entry, and a
        retired mistake makes none.
        """
        return tuple(dict.fromkeys(v.object for v in self.versions if not v.retired))

    def version_of(self, value: str) -> Version | None:
        """The first surviving version carrying `value`, which is the one that introduced it."""
        return next((v for v in self.versions if v.object == value and not v.retired), None)

    def render(self, mark: datetime | None = None) -> list[str]:
        """The slot as lines for a failure report, with `mark` placed in the sequence."""
        lines: list[str] = []
        placed = mark is None
        for version in sorted(self.versions, key=lambda v: (v.valid_from, v.recorded_at)):
            if not placed and mark is not None and mark < version.valid_from:
                lines.append(f"    ---------- asked about {mark.date().isoformat()} ----------")
                placed = True
            flags = []
            if version.recorded_at.date() != version.valid_from.date():
                flags.append(f"learned {version.recorded_at.date().isoformat()}")
            if version.retired:
                flags.append(f"retired {version.retired_at.date().isoformat()}")  # type: ignore[union-attr]
            suffix = f"  ({'; '.join(flags)})" if flags else ""
            lines.append(f"    {version.valid_from.date().isoformat()}  {version.object}"
                         f"  [{version.source}]{suffix}")
        if not placed and mark is not None:
            lines.append(f"    ---------- asked about {mark.date().isoformat()} ----------")
        return lines


def _build_versions(events: Sequence[MemoryEvent], single_valued: bool) -> list[Version]:
    """Apply the four supersession rules to one slot's events."""
    ordered = sorted(events, key=lambda e: (e.valid_from, e.recorded_at, e.id))
    if not single_valued:
        return [Version(event_id=e.id, object=e.object, valid_from=e.valid_from,
                        valid_to=e.valid_to, ended_by_recorded_at=e.recorded_at if e.valid_to else None,
                        retired_at=None, recorded_at=e.recorded_at, source=e.source,
                        confidence=e.confidence)
                for e in ordered]

    # Rule 2: at equal valid time, the later record retires the earlier one — but only
    # when it says something different. Two identical reports about the same instant are
    # rule 4, not a contradiction.
    retired_at: dict[str, datetime] = {}
    for index, event in enumerate(ordered):
        for later in ordered[index + 1:]:
            if later.valid_from != event.valid_from:
                break
            if later.object != event.object and later.recorded_at > event.recorded_at:
                retired_at[event.id] = later.recorded_at
                break

    surviving = [e for e in ordered if e.id not in retired_at]
    versions: list[Version] = []
    for event in ordered:
        valid_to = event.valid_to
        ended_by = event.recorded_at if event.valid_to is not None else None
        if valid_to is None and event.id not in retired_at:
            # Rules 1 and 4: the next surviving value that is actually different.
            successor = next((later for later in surviving
                              if later.valid_from > event.valid_from
                              and later.object != event.object), None)
            if successor is not None:
                valid_to = successor.valid_from
                # Rule 3: the ending is dated by when we heard about the successor.
                ended_by = successor.recorded_at
        versions.append(Version(
            event_id=event.id, object=event.object, valid_from=event.valid_from,
            valid_to=valid_to, ended_by_recorded_at=ended_by,
            retired_at=retired_at.get(event.id), recorded_at=event.recorded_at,
            source=event.source, confidence=event.confidence))
    return versions


class Truth:
    """The dataset's own model of what was true, indexed for lookup."""

    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset
        grouped: dict[tuple[str, str], list[MemoryEvent]] = {}
        for event in dataset.events:
            grouped.setdefault((event.subject, event.predicate), []).append(event)

        self._slots: dict[tuple[str, str], SlotTimeline] = {
            slot: SlotTimeline(slot[0], slot[1],
                               _build_versions(events, dataset.predicates[slot[1]].single_valued),
                               dataset.predicates[slot[1]].single_valued)
            for slot, events in grouped.items()
        }

        #: Every value asserted anywhere in a scenario, used as the competitor set for a
        #: question that names no slot.
        by_scenario: dict[str, list[str]] = {}
        self._subject_scenarios: dict[str, set[str]] = {}
        for event in dataset.events:
            by_scenario.setdefault(event.scenario, []).append(event.object)
            self._subject_scenarios.setdefault(event.subject, set()).add(event.scenario)
        self._scenario_values = {k: tuple(dict.fromkeys(v)) for k, v in by_scenario.items()}

    def slot(self, subject: str, predicate: str) -> SlotTimeline | None:
        return self._slots.get((subject, predicate))

    def competitors(self, probe: tuple[str, str] | None, scenario: str) -> tuple[str, ...]:
        """Values a correct answer has to be distinguished from.

        With a probe, that is the slot's own history — the values the question asks the
        system to choose between. Without one it is every value in the scenario, which is
        broader and therefore stricter: an answer naming two of them is ambiguous, and
        ambiguity scores wrong.
        """
        if probe is not None:
            timeline = self._slots.get(probe)
            if timeline is not None:
                return timeline.objects
        return self._scenario_values.get(scenario, ())

    def other_subjects_holding(self, value: str, predicate: str, scenario: str,
                               excluding: str | None) -> tuple[str, ...]:
        """Which *other* entities in this scenario have held `value` for `predicate`.

        This is how a distractor failure gets named rather than guessed: an answer that
        is wrong for Alice but right for Bob is a retrieval failure, and saying so is
        worth more than reporting a wrong string.
        """
        return tuple(sorted(
            subject for (subject, pred), timeline in self._slots.items()
            if pred == predicate and subject != excluding and value in timeline.objects
            and scenario in self._subject_scenarios.get(subject, set())))

    def known_values(self, scenario: str) -> frozenset[str]:
        return frozenset(self._scenario_values.get(scenario, ()))

    @property
    def slots(self) -> Mapping[tuple[str, str], SlotTimeline]:
        return dict(self._slots)
