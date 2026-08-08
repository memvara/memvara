"""Predicate schema: the contradiction engine.

mem0 detects contradictions by embedding the new fact, pulling the top-k most similar
existing memories, and asking an LLM to emit ADD / UPDATE / DELETE / NOOP. That fails
in three ways that matter in production:

  1. If the contradicting memory is not in the top-k, the contradiction is never seen
     and both facts survive. Recall of the conflict is bounded by recall of the search.
  2. It costs an LLM call on the write path, every time.
  3. It is non-deterministic. The same pair of facts can resolve differently on two runs.

The observation that fixes all three: contradiction is mostly a *schema* property, not a
semantic one. "Lives in" takes one value at a time. "Likes" takes many. If you know the
predicate's cardinality, a conflict is an index lookup on (subject, predicate) - exact,
free, and total. No embedding search, no LLM, no top-k cliff.

The LLM's job moves off the write path and onto schema acquisition: when we meet an
unfamiliar predicate we ask about it *once* and cache the answer forever.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .types import MemoryType


class Cardinality(str, Enum):
    """How many values a predicate can hold at once for one subject."""

    ONE = "one"    # single-valued: a new value invalidates the old one
    MANY = "many"  # multi-valued: values accumulate


class Volatility(str, Enum):
    """How fast a fact goes stale, which sets its recency half-life in ranking."""

    STATIC = "static"  # birthplace, date of birth - effectively never decays
    SLOW = "slow"      # employer, city, marital status - changes over years
    FAST = "fast"      # current task, mood, what they're working on today


HALF_LIFE_DAYS: dict[Volatility, float] = {
    Volatility.STATIC: 36500.0,  # 100y - decay term stays ~1.0
    Volatility.SLOW: 730.0,      # 2y
    Volatility.FAST: 7.0,        # 1w
}


@dataclass(frozen=True, slots=True)
class PredicateSpec:
    name: str
    cardinality: Cardinality = Cardinality.MANY
    volatility: Volatility = Volatility.SLOW
    memory_type: MemoryType = MemoryType.SEMANTIC
    aliases: tuple[str, ...] = ()
    # Asserting this predicate also retires these others. Handles the case where two
    # different predicates cover the same slot, e.g. asserting `unemployed` should
    # retire `works_at` even though the predicate names differ.
    supersedes: tuple[str, ...] = ()
    learned: bool = False  # True if acquired at runtime rather than declared up front

    @property
    def half_life_days(self) -> float:
        return HALF_LIFE_DAYS[self.volatility]

    @property
    def functional(self) -> bool:
        return self.cardinality is Cardinality.ONE


def _p(
    name: str,
    card: Cardinality,
    vol: Volatility,
    *aliases: str,
    mtype: MemoryType = MemoryType.SEMANTIC,
    supersedes: tuple[str, ...] = (),
) -> PredicateSpec:
    return PredicateSpec(name, card, vol, mtype, tuple(aliases), supersedes)


# A starter schema for the personal-assistant domain. Deliberately small: it is meant to
# cover the common cases out of the box and then grow by learning, not to be exhaustive.
BUILTIN_PREDICATES: tuple[PredicateSpec, ...] = (
    # --- identity: single-valued and effectively permanent ---
    _p("name", Cardinality.ONE, Volatility.STATIC, "is_named", "called", "full_name"),
    _p("born_on", Cardinality.ONE, Volatility.STATIC, "birthday", "date_of_birth", "dob"),
    _p("born_in", Cardinality.ONE, Volatility.STATIC, "birthplace", "place_of_birth"),
    _p("pronouns", Cardinality.ONE, Volatility.STATIC, "uses_pronouns"),
    _p("speaks", Cardinality.MANY, Volatility.STATIC, "speaks_language", "language"),

    # --- situation: single-valued but genuinely changes ---
    _p("lives_in", Cardinality.ONE, Volatility.SLOW,
       "resides_in", "located_in", "based_in", "lives_at", "home_is", "moved_to"),
    _p("works_at", Cardinality.ONE, Volatility.SLOW,
       "employed_at", "employer", "employed_by", "company", "works_for", "joined"),
    _p("job_title", Cardinality.ONE, Volatility.SLOW, "role", "title", "position", "works_as"),
    _p("timezone", Cardinality.ONE, Volatility.SLOW, "tz", "in_timezone"),
    _p("relationship_status", Cardinality.ONE, Volatility.SLOW, "marital_status"),
    _p("owns_pet", Cardinality.MANY, Volatility.SLOW, "has_pet", "pet"),

    # --- preferences: mostly multi-valued, and procedural rather than factual ---
    _p("likes", Cardinality.MANY, Volatility.SLOW, "enjoys", "loves", "is_a_fan_of",
       mtype=MemoryType.SEMANTIC),
    _p("dislikes", Cardinality.MANY, Volatility.SLOW, "hates", "avoids",
       mtype=MemoryType.SEMANTIC),
    _p("allergic_to", Cardinality.MANY, Volatility.STATIC, "allergy", "is_allergic_to"),
    _p("dietary_restriction", Cardinality.MANY, Volatility.SLOW, "diet", "eats"),

    # --- procedural: how this user wants the agent to behave ---
    _p("prefers", Cardinality.MANY, Volatility.SLOW, "wants", "would_rather",
       mtype=MemoryType.PROCEDURAL),
    _p("prefers_tool", Cardinality.ONE, Volatility.SLOW, "uses_tool", "tool_of_choice",
       mtype=MemoryType.PROCEDURAL),
    _p("communication_style", Cardinality.ONE, Volatility.SLOW, "tone", "style",
       mtype=MemoryType.PROCEDURAL),
    _p("never_do", Cardinality.MANY, Volatility.SLOW, "avoid_doing", "do_not",
       mtype=MemoryType.PROCEDURAL),

    # --- fast-moving state: decays out of the ranking within days ---
    _p("working_on", Cardinality.ONE, Volatility.FAST, "current_task", "current_project",
       mtype=MemoryType.EPISODIC),
    _p("goal", Cardinality.MANY, Volatility.FAST, "wants_to", "plans_to",
       mtype=MemoryType.EPISODIC),
    _p("mood", Cardinality.ONE, Volatility.FAST, "feeling", "feels"),
    _p("located_now", Cardinality.ONE, Volatility.FAST, "currently_in", "travelling_to"),
)


class PredicateRegistry:
    """Normalizes predicate names and answers the cardinality question.

    Unknown predicates default to MANY. That is the conservative direction: keeping two
    facts that turn out to conflict degrades ranking, while dropping one that turns out
    not to conflict destroys information. Errors should fall on the recoverable side.
    """

    def __init__(self, specs: tuple[PredicateSpec, ...] = BUILTIN_PREDICATES) -> None:
        self._specs: dict[str, PredicateSpec] = {}
        self._alias: dict[str, str] = {}
        for s in specs:
            self.register(s)

    # -- normalization -------------------------------------------------------

    @staticmethod
    def _slug(raw: str) -> str:
        out = []
        prev_us = False
        for ch in raw.strip().lower():
            if ch.isalnum():
                out.append(ch)
                prev_us = False
            elif not prev_us:
                out.append("_")
                prev_us = True
        return "".join(out).strip("_")

    def normalize(self, raw: str) -> str:
        """Map a surface predicate onto its canonical name.

        Without this, `lives_in` and `resides_in` are different slots and the
        contradiction between them is invisible - which is exactly how free-text memory
        stores end up holding two cities for one person.
        """
        s = self._slug(raw)
        if s in self._specs:
            return s
        if s in self._alias:
            return self._alias[s]
        return s

    # -- registry ------------------------------------------------------------

    def register(self, spec: PredicateSpec) -> PredicateSpec:
        self._specs[spec.name] = spec
        for a in spec.aliases:
            self._alias[self._slug(a)] = spec.name
        return spec

    def spec(self, predicate: str) -> PredicateSpec:
        name = self.normalize(predicate)
        found = self._specs.get(name)
        if found is not None:
            return found
        # Unknown: synthesize a conservative default. Not registered, so a later
        # `learn()` can still install a real spec for it.
        return PredicateSpec(name=name, cardinality=Cardinality.MANY, volatility=Volatility.SLOW)

    def known(self, predicate: str) -> bool:
        return self.normalize(predicate) in self._specs

    def learn(
        self,
        predicate: str,
        cardinality: Cardinality,
        volatility: Volatility = Volatility.SLOW,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        aliases: tuple[str, ...] = (),
    ) -> PredicateSpec:
        """Install a spec discovered at runtime (typically proposed by an LLM, once).

        This is the amortization that keeps the write path cheap: the Nth occurrence of
        a predicate costs nothing, because the 1st one paid for the schema.
        """
        return self.register(
            PredicateSpec(
                name=self.normalize(predicate),
                cardinality=cardinality,
                volatility=volatility,
                memory_type=memory_type,
                aliases=aliases,
                learned=True,
            )
        )

    def functional(self, predicate: str) -> bool:
        return self.spec(predicate).functional

    def half_life_days(self, predicate: str) -> float:
        return self.spec(predicate).half_life_days

    def superseded_by(self, predicate: str) -> tuple[str, ...]:
        return self.spec(predicate).supersedes

    def all_specs(self) -> list[PredicateSpec]:
        return sorted(self._specs.values(), key=lambda s: s.name)

    def __len__(self) -> int:
        return len(self._specs)
