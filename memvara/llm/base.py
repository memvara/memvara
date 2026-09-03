"""LLM protocol.

The interface is deliberately tiny because the whole architecture is built to call an
LLM as rarely as possible. Everything an LLM is genuinely needed for falls into two
buckets:

  extract()            - turn unstructured text into structured claims
  resolve_predicate()  - answer "have we already got a predicate for this?" once per
                         novel surface form, ever, then cache the answer in the registry

`resolve_predicate` replaced `classify_predicate` as the acquisition call, and the
difference is the point. Classification asked "how should this new predicate behave?",
which quietly accepted the premise that every phrasing a model invents deserves a slot;
a measured run produced 41 predicates for six questions and thirteen live answers to
"where do you work?". Resolution asks the question that was actually load-bearing -
"which existing predicate is this?" - and spends the same one-off call on *merging*.
`classify_predicate` stays for backends that predate the change; the pipeline falls back
to it and the conservative default keeps holding.

Contradiction detection, deduplication, ranking, decay, and time travel are all
deterministic and never call this interface. That is the design.

`NullLLM` is the default. The library must be fully functional with no API key: you
get the deterministic fast path, and `add()` tells you honestly what it did - including
that it cost nothing, which is why a no-op backend advertises itself via `is_noop`
rather than being detected by class name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

from ..types import Episode


@dataclass(slots=True)
class Usage:
    """Tokens a backend consumed, accumulated across one write.

    **The caller allocates this and passes it in; the backend fills it.** A `last_usage`
    attribute read after the call would be smaller, and would be wrong: `pipeline.py`
    deliberately moved the model round trip *outside* the store's transaction so reads
    are not blocked by it, which means two `add()` calls can be inside `extract()` on one
    backend instance at the same time. Shared mutable state on the backend would then
    bill one caller for the other's tokens, intermittently, with no way to notice. A
    per-call object cannot race because it is not shared.

    One `Usage` spans a whole tier-2 batch — the extraction *and* any predicate
    acquisition it triggers — because the unit a caller is billed for is the write, not
    the round trip. `reported` is what separates "the model consumed nothing", which
    cannot happen, from "this backend does not report usage", which is common; the write
    path publishes no token series at all in the second case rather than a run of zeros.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    #: Calls that came back carrying a usage block. See the class docstring.
    reported: int = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.reported += 1


@runtime_checkable
class LLM(Protocol):
    name: str
    #: True for a backend that consults no model. Suppresses `llm_calls` billing, which
    #: must count model consultations rather than method invocations - otherwise the one
    #: number the write path exists to minimize reports spend that never happened.
    is_noop: bool = False
    #: True if this backend accepts a `usage=` accumulator and fills it. Advertised
    #: rather than detected, for the reason `is_noop` is: a class-name check or a
    #: signature probe guesses, and this is a claim the backend should have to make.
    #:
    #: The write path only passes `usage=` to a backend that sets this, so a third-party
    #: implementation written against the older three-argument signature keeps working
    #: untouched — the same courtesy `classify_predicate` still gets. What such a backend
    #: loses is only the token series; `write.llm_calls` is unaffected.
    reports_usage: bool = False

    def extract(self, episodes: Sequence[Episode], known_predicates: Sequence[str],
                *, usage: "Usage | None" = None) -> list[dict[str, Any]]:
        """Return claim dicts: subject, predicate, object, polarity, memory_type,
        confidence, source_index (index into `episodes`, for provenance).

        Add this call's tokens to `usage` when it is not None and this backend sets
        `reports_usage`. See `Usage` for why it arrives as an argument.
        """
        ...

    def resolve_predicate(self, surface: str, candidates: Sequence[str],
                          *, usage: "Usage | None" = None) -> dict[str, Any]:
        """Decide whether `surface` is a new predicate or a spelling of an existing one.

        Returns {'canonical': str | None, 'cardinality': 'one'|'many',
        'volatility': 'static'|'slow'|'fast',
        'memory_type': 'episodic'|'semantic'|'procedural'}, where `canonical` names one
        of `candidates` the surface form is a synonym of, or None if it is genuinely
        new. The cardinality fields describe the *new* predicate and are ignored when a
        canonical is returned.
        """
        ...

    def classify_predicate(self, predicate: str, example: str,
                           *, usage: "Usage | None" = None) -> dict[str, str]:
        """Legacy acquisition call. Return {'cardinality': 'one'|'many',
        'volatility': 'static'|'slow'|'fast',
        'memory_type': 'episodic'|'semantic'|'procedural'}."""
        ...


class NullLLM:
    """No-op backend. Deterministic paths still work; extraction simply yields nothing."""

    name = "null"
    is_noop = True
    # Nothing is consumed, so there is nothing to report. Left False deliberately rather
    # than set True with a zero: `reported` distinguishes a backend that measured nothing
    # from one that cannot measure, and a no-op backend is neither — it never runs.
    reports_usage = False

    def extract(self, episodes: Sequence[Episode], known_predicates: Sequence[str],
                *, usage: Usage | None = None) -> list[dict[str, Any]]:
        return []

    def resolve_predicate(self, surface: str, candidates: Sequence[str],
                          *, usage: Usage | None = None) -> dict[str, Any]:
        # With no model there is no evidence that two spellings mean the same thing, and
        # guessing one would merge slots on nothing but a hunch. The deterministic
        # pre-pass has already had its turn; whatever reaches here stays separate.
        return {"canonical": None, "cardinality": "many", "volatility": "slow",
                "memory_type": "semantic"}

    def classify_predicate(self, predicate: str, example: str,
                           *, usage: Usage | None = None) -> dict[str, str]:
        # Conservative default: multi-valued predicates never wrongly retire a fact.
        return {"cardinality": "many", "volatility": "slow", "memory_type": "semantic"}


# --- JSON schemas for structured extraction ---------------------------------
# Constrained decoding rather than "please reply with JSON": no parse-retry loop, no
# markdown fences to strip, no partially-valid objects to defend against.

CLAIM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "polarity": {"type": "integer", "enum": [1, -1]},
                    "memory_type": {
                        "type": "string",
                        "enum": ["episodic", "semantic", "procedural"],
                    },
                    "confidence": {"type": "number"},
                    "source_index": {"type": "integer"},
                    # Nullable rather than optional: strict `json_schema` requires every
                    # declared property in `required`, so "the turn stated no time" is
                    # spelled `null` rather than by omitting the key.
                    "when": {"type": ["string", "null"]},
                    "amount": {"type": ["number", "null"]},
                    "unit": {"type": ["string", "null"]},
                },
                "required": [
                    "subject", "predicate", "object", "polarity",
                    "memory_type", "confidence", "source_index",
                    "when", "amount", "unit",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}

PREDICATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "cardinality": {"type": "string", "enum": ["one", "many"]},
        "volatility": {"type": "string", "enum": ["static", "slow", "fast"]},
        "memory_type": {"type": "string", "enum": ["episodic", "semantic", "procedural"]},
    },
    "required": ["cardinality", "volatility", "memory_type"],
    "additionalProperties": False,
}

EXTRACT_SYSTEM = """\
You extract durable facts from conversation turns and emit them as (subject, predicate, object) triples.

Rules:
- Only extract facts that will still be worth knowing in a later, unrelated conversation. \
Skip pleasantries, acknowledgements, questions, and anything transient to the current exchange.
- A thing that happened counts as durable when the turn gives it a time or a measured \
quantity: "I ran 30 minutes yesterday", "I spent $40 on groceries last week", "I finished \
the course in March". Record those, with the time in `when` and the measurement in \
`amount`/`unit`. An undated, unmeasured happening is transient and is still skipped - \
"I went for a run" alone is not worth a later conversation, and "I ran 5k on Tuesday" is.
- subject: use "user" for the person speaking, or a lowercase entity name.
- predicate: lowercase snake_case. Reuse a predicate from the known list whenever it fits \
- consistency matters far more than precision here, because predicates are how \
contradictions get detected.
- object: the value alone, no articles or filler.
- polarity: 1 for an assertion, -1 for a retraction ("I no longer work at Acme" -> \
works_at / Acme / polarity -1).
- memory_type: "semantic" for durable facts, "episodic" for things that happened at a \
point in time, "procedural" for how the user wants an assistant to behave.
- confidence: 0.0-1.0. Use lower values for facts that were implied rather than stated.
- source_index: the 0-based index of the turn the fact came from. This is load-bearing \
provenance, so it must be exact.
- when: the temporal expression exactly as it appears in the turn, naming the single \
earliest time this fact is tied to - "yesterday", "last month", "three weeks ago", "in 2019". \
Copy the words; never compute a date, and never sharpen a vague expression into a precise \
one. null when the turn states no time, which is the common case.
- amount and unit: a measured quantity, if the turn states one - amount 30, unit "minutes". \
At most one per fact: "I ran 5 km in 30 minutes" is two facts, one for the distance and one \
for the duration, never one fact with a compound object. null for both when nothing is \
measured.

Return an empty list when a turn carries no durable fact. That is the common case, and \
an empty list is a correct answer."""

COMPOSE_SYSTEM = (
    "You are given the predicate names a memory store uses. Name the English relation "
    "terms a person would use that are NOT one of those predicates but are a composition "
    "of two or more of them, and give the number of predicates each composes from.\n"
    "Example: given father, mother, spouse — 'grandfather' is father-of-father or "
    "father-of-mother, so it composes from 2. 'father-in-law' is father-of-spouse, 2.\n"
    "Do not list a term that is already one of the predicates. Do not list a term that "
    "composes from only one. Terms of at most three words. If none apply, answer with an "
    "empty object."
)

#: The answer is a term -> arity map. Free-form keys, because the terms are the thing
#: being asked for and enumerating them in a schema would be asking the question twice.
#: Everything is filtered on the way back in `retrieve/compose.acquire`, which is where a
#: bad answer costs the questions it would have helped and nothing else.
COMPOSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "derived": {
            "type": "object",
            "additionalProperties": {"type": "integer", "minimum": 2},
        },
    },
    "required": ["derived"],
    "additionalProperties": False,
}

RESOLVE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        # Nullable rather than optional: "this is new" has to be sayable in one token,
        # or a model with nothing to merge onto will reach for the nearest candidate.
        "canonical": {"type": ["string", "null"]},
        "cardinality": {"type": "string", "enum": ["one", "many"]},
        "volatility": {"type": "string", "enum": ["static", "slow", "fast"]},
        "memory_type": {"type": "string", "enum": ["episodic", "semantic", "procedural"]},
    },
    "required": ["canonical", "cardinality", "volatility", "memory_type"],
    "additionalProperties": False,
}

RESOLVE_SYSTEM = """\
You decide whether a predicate a memory system just saw is a new relation or another \
spelling of one it already has.

canonical: if the new predicate asks the same question about a subject as one of the \
listed existing predicates - so that a new value should *replace* the old one rather \
than sit beside it - return that existing predicate's name exactly. Otherwise return \
null. Examples of the same question: employer_name / works_at, current_city / lives_in, \
emotional_state / mood. Examples of different questions: works_at / working_on \
(employer vs current task), born_in / lives_in (origin vs residence), job_title / \
works_at (role vs company).

Answer null when unsure. Two predicates that should have been merged only cost some \
ranking quality; merging two that should not have been permanently destroys the \
distinction between them and silently retires true facts.

The remaining fields describe the new predicate and are used only when canonical is null.

cardinality: "one" if a subject can only have one value at a time, so a new value \
replaces the old (lives_in, works_at, date_of_birth). "many" if values accumulate \
(likes, speaks, allergic_to). When genuinely unsure answer "many".

volatility: "static" never changes (born_in). "slow" changes over years (works_at). \
"fast" changes within days (current_task, mood).

memory_type: "semantic" for facts, "episodic" for events, "procedural" for behavioral \
preferences directed at an assistant."""

PREDICATE_SYSTEM = """\
You classify a relational predicate so a memory system knows how to store it.

cardinality: "one" if a subject can only have one value at a time, so a new value \
replaces the old (lives_in, works_at, date_of_birth). "many" if values accumulate \
(likes, speaks, allergic_to). When genuinely unsure answer "many" - wrongly retiring a \
true fact is far more damaging than keeping two.

volatility: "static" never changes (born_in). "slow" changes over years (works_at). \
"fast" changes within days (current_task, mood).

memory_type: "semantic" for facts, "episodic" for events, "procedural" for behavioral \
preferences directed at an assistant."""
