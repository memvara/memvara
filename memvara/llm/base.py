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

import copy

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


@runtime_checkable
class Chat(Protocol):
    """A backend that can hold one unstructured system+user exchange and hand back text.

    Its own protocol rather than a third method on `LLM`, for the reason `RelationComposer`
    (`retrieve/compose.py`) is its own protocol: adding a member to a `runtime_checkable`
    protocol breaks `isinstance` for every implementation that predates it, and a
    downstream backend finds out when its type checker does. `LLM` calls a model for
    exactly two things, and this is not one of them — it exists for `memvara.select`,
    which needs a plain chat completion, not the schema-constrained extraction the two
    `LLM` methods make.

    A backend that does not have this is not broken. `ModelSelector.__init__` refuses to
    construct on one, with a `TypeError` naming the extra to install — the same courtesy
    `OpenAILLM._default_client` gives an absent SDK.
    """

    def chat(self, system: str, prompt: str, *, json_object: bool,
             max_completion_tokens: int, timeout: float,
             usage: "Usage | None" = None) -> str:
        """One request: a system message, a user message, the reply's text back.

        `json_object` asks the backend for its provider's loosest JSON-shaped response
        mode where one exists (OpenAI's `response_format: {"type": "json_object"}`); a
        backend with no such mode relies on the prompt asking for JSON in prose, as
        `memvara.select`'s does, and treats the flag as informational.

        `timeout` is the caller's remaining budget for this one call, in seconds — not a
        per-socket-operation limit, the whole round trip. The caller starts and reads its
        own clock around this call; `timeout` is a hint the backend forwards to its
        transport, not something this method enforces itself.

        Add this call's tokens to `usage` when it is not None and this backend sets
        `reports_usage`, exactly as `extract` and `resolve_predicate` do.
        """
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

#: Default ceiling for `bounded_claim_schema`. Deliberately not applied to `CLAIM_SCHEMA`
#: itself, for the reason that function documents.
#:
#: 32 because a bound is there to stop a runaway, not to edit a good answer. Measured
#: against phi-4-mini through llama.cpp on 2026-09-03: the runaway passed 35 claims, while
#: every well-formed response produced 19 or fewer, so 32 sits above anything real and
#: below the failure.
MAX_CLAIMS = 32

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


def bounded_claim_schema(max_claims: int = MAX_CLAIMS) -> dict[str, Any]:
    """`CLAIM_SCHEMA` with the claims array capped, for a backend that constrains decoding.

    Unbounded, "one more claim" is forever a legal continuation, which is only safe for a
    model that stops on its own. A backend that compiles this schema to a grammar —
    llama.cpp, vLLM — has no legal way to end a response the grammar still permits to
    continue: a model that starts restating itself runs to its token limit still emitting
    well-formed claim objects, and the reply arrives as truncated JSON that parses as
    nothing, losing the real claims that preceded the restatements along with it. Measured
    against phi-4-mini through llama.cpp, one extraction in three was lost this way.

    **The cap is opt-in rather than part of `CLAIM_SCHEMA` because OpenAI's strict-mode
    structured output rejects `maxItems`.** It sits on the documented list of keywords
    strict mode does not permit, alongside `minItems`, `uniqueItems` and `pattern`, and an
    unsupported keyword is a 400 rather than something ignored — so putting the cap in the
    shared schema would take the hosted OpenAI backend off the air in order to protect a
    local one. `test_every_schema_satisfies_strict_mode` pins that boundary.

    A hosted model closes the array itself and needs no cap; a self-hosted server reached
    through the same `openai` backend does. That split is why this is a parameter and not
    a constant.

    What exceeding the cap costs is the backend's business rather than this function's: a
    grammar backend stops at `max_claims`, so the tail of an unusually rich turn is lost,
    while a backend that ignores `maxItems` is unaffected and the cap is inert.
    """
    schema = copy.deepcopy(CLAIM_SCHEMA)
    schema["properties"]["claims"]["maxItems"] = max_claims
    return schema


#: The claim fields nothing downstream can supply a default for. `_shape.shape_claims`
#: drops a claim missing any of these rather than repairing it: without `source_index` the
#: claim cannot be traced back to a turn, and without a subject, predicate and object
#: there is no fact to store. Every other field in `CLAIM_SCHEMA` already has a documented
#: fallback in that function, which is what makes `self_hosted_claim_schema` safe.
LOAD_BEARING_CLAIM_FIELDS = ("subject", "predicate", "object", "source_index")


def self_hosted_claim_schema(max_claims: int = MAX_CLAIMS) -> dict[str, Any]:
    """`bounded_claim_schema`, with the fields the model may leave out taken out of `required`.

    On a CPU-hosted model, generating the response is most of the wall time, and much of
    what gets generated is field names rather than facts. Measured against the shipped
    shape: one claim serializes to 52 tokens, of which 44 are keys, punctuation and the
    three forced nulls.

    **Measured end to end on the production box, this saves 27% of the generated tokens.**
    Three episodes, the deployment's own prompt and predicate vocabulary, a 12-claim cap on
    both arms, phi-4-mini Q8_0: 2,401 output tokens under the shipped schema against 1,756
    under this one, and the same 10 of 15 key facts found either way. Prefill did not move —
    2,894 tokens on both — because the prompt is identical.

    Serialization alone predicts 45%, and the gap between that and 27% is the honest
    correction: the model still spends tokens on the values and on deciding what to write,
    and only the field names went away. Take 27% as the number and re-measure with
    `bench/extract_cost.py` for any other model, because it is a property of the model's
    habits and not of the schema.

    At the rates production runs at — Q8_0, four threads, sharing the box with the API and
    Postgres, 21.0 tokens per second prefill and 5.53 generation — 27% off generation is a
    typical extraction going from 98 seconds to 79, a 20% cut in wall time. Those are
    measured production rates, not `llama-bench` figures, which are roughly twice as fast
    because the bench runs with everything else stopped on a short context.

    **The long turn gains most, and there it is a reliability fix rather than a speedup.** A
    production call on a 4,117-token turn spent 220 seconds on prefill and 333 generating
    1,618 tokens, 554 in total, against the OpenAI SDK's default 600-second client timeout;
    `max_tokens` is 8,192, so nothing in this library stops a long response first. 27% off
    generation takes that call to about 464 seconds — from 92% of the timeout budget to
    77%. A cancelled call is a turn deferred and retried next pass.

    **It does not bound a runaway, and nothing here should be read as claiming it does.**
    Measured on the same box, an *uncapped* claims array reached 7,197 generated tokens on a
    900-character turn, ran for 1,957 seconds, and found 7 of 15 facts against the capped
    arm's 10 — the restatements crowd out the answer. A fixed fraction of a runaway is still
    a runaway. `bounded_claim_schema` is what stops it, which is why this function builds on
    it rather than beside it.

    Five fields move out of `required`, and each one is safe because `shape_claims` already
    reads it with `.get()` and documents what an absent value means. `polarity` defaults to
    1, since anything that is not an explicit -1 is an assertion. `when`, `amount` and
    `unit` default to `None`, which is what the shipped schema spells as an explicit null on
    the common turn that states no time and no measurement.

    **`confidence` is the one field whose behaviour changes, and it changes ranking.**
    `clamp_confidence` returns `UNKNOWN_CONFIDENCE` for a value it cannot read, so an
    omitted confidence puts every claim at 0.5 instead of a number the model chose. On a
    small model that number is close to noise — the field is the one thing in a claim that
    no validation can check — but a store written under this schema ranks differently from
    one written without it. `write/pollution.py`'s R4 still sorts a discounted claim below a
    clean one, at 0.4 against 0.5, on a narrower margin than before. Decide it per
    deployment rather than per call.

    Opt-in for the same reason `bounded_claim_schema` is. OpenAI's strict mode requires
    every declared property to appear in `required`, so this schema is a 400 on the hosted
    path rather than something that degrades quietly. `additionalProperties` stays False
    and every property stays declared, so a model that does send `confidence` is still
    understood — the fields become optional, not forbidden.

        >>> schema = self_hosted_claim_schema(8)
        >>> schema["properties"]["claims"]["maxItems"]
        8
        >>> schema["properties"]["claims"]["items"]["required"]
        ['subject', 'predicate', 'object', 'source_index']
        >>> len(schema["properties"]["claims"]["items"]["properties"])
        10
    """
    schema = bounded_claim_schema(max_claims)
    schema["properties"]["claims"]["items"]["required"] = list(LOAD_BEARING_CLAIM_FIELDS)
    return schema


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
