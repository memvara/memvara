"""LLM protocol.

The interface is deliberately tiny — two methods — because the whole architecture is
built to call an LLM as rarely as possible. Everything an LLM is genuinely needed for
falls into two buckets:

  extract()            - turn unstructured text into structured claims
  classify_predicate() - answer "is this predicate single-valued?" once per predicate,
                         ever, then cache the answer in the schema registry

Contradiction detection, deduplication, ranking, decay, and time travel are all
deterministic and never call this interface. That is the design.

`NullLLM` is the default. The library must be fully functional with no API key: you
get the deterministic fast path, and `add()` tells you honestly what it did.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from ..types import Episode


@runtime_checkable
class LLM(Protocol):
    name: str

    def extract(self, episodes: Sequence[Episode], known_predicates: Sequence[str]) -> list[dict[str, Any]]:
        """Return claim dicts: subject, predicate, object, polarity, memory_type,
        confidence, source_index (index into `episodes`, for provenance)."""
        ...

    def classify_predicate(self, predicate: str, example: str) -> dict[str, str]:
        """Return {'cardinality': 'one'|'many', 'volatility': 'static'|'slow'|'fast',
        'memory_type': 'episodic'|'semantic'|'procedural'}."""
        ...


class NullLLM:
    """No-op backend. Deterministic paths still work; extraction simply yields nothing."""

    name = "null"

    def extract(self, episodes: Sequence[Episode], known_predicates: Sequence[str]) -> list[dict[str, Any]]:
        return []

    def classify_predicate(self, predicate: str, example: str) -> dict[str, str]:
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
                },
                "required": [
                    "subject", "predicate", "object", "polarity",
                    "memory_type", "confidence", "source_index",
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

Return an empty list when a turn carries no durable fact. That is the common case, and \
an empty list is a correct answer."""

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
