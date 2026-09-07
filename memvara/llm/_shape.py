"""Validation shared by every model backend. The trust boundary lives here.

Everything downstream — the reconciler, the ranker, the schema registry — treats the
dicts a backend returns as structured facts and acts on them without re-checking. So a
hallucinated `source_index` becomes a claim attributed to the wrong conversation, and a
`confidence` of 5.0 silently outranks every honest claim in the store. Constrained
decoding makes malformed output rare, not impossible; validation is what makes it
harmless.

None of that reasoning is provider-specific, which is why it is here rather than in
`anthropic.py`. A second backend that reimplemented these rules would drift from the
first, and the drift would show up as *data* — differently-shaped claims in one store,
depending on which model happened to be configured the day a turn was written. Backends
own their transport and their response shape. They do not own what counts as a valid
claim.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Sequence

from ..types import Episode, MemoryType
from .base import TruncatedResponse, Usage

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")

CARDINALITY = ("one", "many")
VOLATILITY = ("static", "slow", "fast")
MEMORY_TYPES = tuple(m.value for m in MemoryType)

# Matches NullLLM: unknown predicates default to multi-valued, because wrongly retiring a
# true fact is worse than keeping two competing ones.
PREDICATE_FALLBACK: dict[str, str] = {
    "cardinality": "many",
    "volatility": "slow",
    "memory_type": "semantic",
}

# A model that ignored the schema tells us nothing about how sure it is. Neither extreme
# is safe - 1.0 lets a malformed claim outrank well-formed ones, 0.0 makes it
# unretrievable - so an unreadable confidence lands in the middle.
UNKNOWN_CONFIDENCE = 0.5

# Ceiling on the known-predicate list sent with every extraction. Sending the whole
# vocabulary is an unbounded per-write token tax that grows fastest exactly when the
# vocabulary is growing fastest, and each new predicate shifts the bytes of the prompt
# prefix and throws away the cache at the same moment. The list is a reuse hint, not an
# enumeration, so a bounded head of it does the same job.
MAX_KNOWN_PREDICATES = 64

# Ceiling on the candidate list sent when resolving a surface form. Same reasoning, and
# a longer list measurably makes the merge decision worse rather than better.
MAX_CANDIDATES = 48


def snake_case(raw: str) -> str:
    """`Lives In` / `livesIn` / `LIVES-IN` -> `lives_in`.

    Predicate identity *is* slot identity: `lives_in` and `livesIn` arriving as different
    predicates means the contradiction between them is invisible and the store quietly
    holds two cities for one person. `PredicateRegistry.normalize()` still runs
    downstream to resolve aliases onto canonical names - this only guarantees it receives
    a well-formed key to look up.
    """
    return _NON_ALNUM.sub("_", _CAMEL.sub("_", raw).strip().lower()).strip("_")


def parse_json_object(text: str) -> dict[str, Any]:
    """A JSON object, or `{}` for anything else.

    Backends hand over already-extracted text, because "where is the text in this
    response" is the one part that really is provider-specific.
    """
    text = (text or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def finite_amount(value: Any) -> float | None:
    """A measured quantity, or `None` for anything that cannot be one.

    The last unguarded field in this module, and it is guarded for the reasons its
    neighbours are. `isinstance(True, int)` is True in Python, so a stray boolean would
    land as `amount=1.0` — a measurement nobody took, the same slip `source_index` refuses
    a few lines down.

    Two more arrive from the wire rather than from a typo. `json.loads('{"a": 1e400}')`
    yields `inf` rather than raising, and `inf` is a perfectly valid `float` that
    round-trips through the store and reads afterwards as a real distance. And an integer
    of a few hundred digits raises `OverflowError` from `float()` — which `WritePipeline`
    catches around the whole `extract` call, so one malformed number silently discards
    every claim the model returned for that batch, recorded exactly as a provider 429 is.

    A value that cannot become a finite float is not a quantity, so it is dropped and the
    claim keeps everything else.

        >>> finite_amount(30), finite_amount(True), finite_amount(float("inf"))
        (30.0, None, None)
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def clamp_confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return UNKNOWN_CONFIDENCE
    # NaN loses every comparison, so clamping it silently yields 0.0 and buries the claim
    # at the bottom of the ranking rather than admitting we could not read the field.
    if not math.isfinite(value):
        return UNKNOWN_CONFIDENCE
    return min(1.0, max(0.0, float(value)))


def source_index(value: Any, n: int) -> int | None:
    # `isinstance(True, int)` is True in Python, and a claim sourced from episode `True`
    # would be attributed to episode 1 - a real provenance bug from a one-character slip.
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 0 <= value < n else None


def bounded(names: Sequence[str], limit: int) -> list[str]:
    """Dedupe preserving order, then truncate.

    Order is preserved rather than sorted on purpose. `PredicateRegistry` hands the
    vocabulary over declared-first, which is a head that does not move as predicates
    are learned; re-sorting would interleave every newly acquired predicate into the
    middle and invalidate the cached prompt prefix on each acquisition - the one
    moment the cache is worth the most, because that is when writes are busiest.
    """
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name and name not in seen:
            seen.add(name)
            out.append(name)
            if len(out) == limit:
                break
    return out


def extract_prompt(episodes: Sequence[Episode], known_predicates: Sequence[str]) -> str:
    turns = "\n".join(f"[{i}] {ep.role}: {ep.content}" for i, ep in enumerate(episodes))
    known = ", ".join(bounded(known_predicates, MAX_KNOWN_PREDICATES))
    return (
        f"Known predicates, reuse one whenever it fits:\n{known or '(none yet)'}\n\n"
        f"Turns:\n{turns}"
    )


def resolve_prompt(surface: str, offered: Sequence[str]) -> str:
    return (
        f"new predicate: {snake_case(surface)}\n"
        f"existing predicates:\n{', '.join(offered) or '(none yet)'}"
    )


def compose_prompt(predicates: Sequence[str]) -> str:
    return f"predicates:\n{', '.join(predicates) or '(none)'}"


def shape_composition(parsed: dict[str, Any]) -> dict[str, int]:
    """The `derived` map, with anything that is not a term-and-arity dropped.

    Shaped here rather than trusted, like every other model answer in this package. The
    caller filters again — `retrieve/compose.acquire` drops terms that collide with a real
    predicate or run to a whole phrase — because this function knows the response's shape
    and that one knows the store's vocabulary, and neither can do the other's job.
    """
    # Both shapes, because a model that is not being held to the schema returns the bare
    # map. Measured: `nvidia/nemotron-3-ultra` answered with 21 correct kinship terms and
    # no `derived` wrapper, and this function dropped every one of them and returned
    # nothing — the acquisition looked like a model with no opinion rather than a parser
    # reading the wrong shape. Anthropic's structured output does enforce the wrapper, so
    # the unit test could not have caught it: the fake client returned the shape this
    # code was written to expect.
    # No guard on `derived` being a dict: `parse_json_object` returns one or `{}`, so
    # both branches above already are.
    inner = parsed.get("derived")
    derived = inner if isinstance(inner, dict) else parsed
    out: dict[str, int] = {}
    for term, arity in derived.items():
        if isinstance(term, str) and isinstance(arity, int) and not isinstance(arity, bool):
            out[term] = arity
    return out


def shape_claims(parsed: dict[str, Any], n_episodes: int) -> list[dict[str, Any]]:
    """Validated claim dicts from a parsed model response. Anything doubtful is dropped."""
    raw = parsed.get("claims")
    if not isinstance(raw, list):
        return []

    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        # Provenance first: a claim we cannot trace back to a turn is unusable, and
        # guessing an episode for it would corrupt `why()` for every reader after us.
        index = source_index(item.get("source_index"), n_episodes)
        if index is None:
            continue
        subject = str(item.get("subject") or "").strip()
        predicate = snake_case(str(item.get("predicate") or ""))
        obj = str(item.get("object") or "").strip()
        if not (subject and predicate and obj):
            continue
        memory_type = str(item.get("memory_type") or "")
        raw_when = item.get("when")
        when = raw_when.strip() if isinstance(raw_when, str) and raw_when.strip() else None
        amount = finite_amount(item.get("amount"))
        out.append(
            {
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                # Anything other than an explicit -1 is an assertion. A garbled value
                # must not be read as a retraction, which would invalidate a live fact.
                "polarity": -1 if item.get("polarity") == -1 else 1,
                "memory_type": (
                    memory_type if memory_type in MEMORY_TYPES else MemoryType.SEMANTIC.value
                ),
                "confidence": clamp_confidence(item.get("confidence")),
                "source_index": index,
                # The temporal expression as the model saw it, never a date it computed:
                # `write.when` is the only thing allowed to decide what a phrase means.
                # Anything that is not a non-empty string becomes `None`, and the caller
                # then falls back to the episode's timestamp exactly as before.
                "when": when,
                # A quantity is kept only as a pair. `bool` is excluded before the numeric
                # test because `isinstance(True, int)` is True in Python and `amount=1`
                # from a stray boolean is a measurement nobody made — the same slip
                # `source_index` guards against a few lines up.
                "amount": amount,
                "unit": str(item.get("unit") or "").strip().lower() or None
                        if amount is not None else None,
            }
        )
    return out


def spec_fields(parsed: dict[str, Any]) -> dict[str, str]:
    # This answer is cached in the registry forever, so a bad field here is a
    # permanent mistake for that predicate. Any value outside the enum falls back to
    # the conservative default rather than being coerced into a neighbour.
    result = dict(PREDICATE_FALLBACK)
    for field, allowed in (
        ("cardinality", CARDINALITY),
        ("volatility", VOLATILITY),
        ("memory_type", MEMORY_TYPES),
    ):
        value = parsed.get(field)
        if isinstance(value, str) and value in allowed:
            result[field] = value
    return result


def shape_resolution(parsed: dict[str, Any], offered: Sequence[str]) -> dict[str, Any]:
    """The most consequential validation here.

    `canonical` is echoed straight into `PredicateSpec.aliases`, so a hallucinated name
    would permanently reroute every claim using that surface form into a slot nobody
    looks up. Only a name the caller actually offered is accepted - a model that invents
    one is treated as having said "new", which is the recoverable direction.
    """
    canonical = parsed.get("canonical")
    if not isinstance(canonical, str) or snake_case(canonical) not in offered:
        canonical = None
    else:
        canonical = snake_case(canonical)
    return {"canonical": canonical, **spec_fields(parsed)}


def record_usage(response: Any, usage: "Usage | None",
                 input_field: str, output_field: str) -> None:
    """Add one provider response's token counts to `usage`, if both are present.

    Tolerates SDK objects and plain dicts, for the reason each backend's `_first_text`
    does: a test double should not have to reimplement the SDK's types to be exercised.

    **A response whose usage block is missing or unreadable records nothing** rather than
    recording a zero, which is what keeps `Usage.reported` meaningful — see its docstring.
    A provider that changed a field name would otherwise silently start reporting free
    writes, and free is the direction that flatters us.

    Cached-prompt tokens are deliberately *not* folded in. Providers report them in
    separate fields and bill them at a different rate (an order of magnitude cheaper to
    read), so adding them to `input_field` would overstate cost. Leaving them out
    understates it instead, which is the same direction the metering layer chose: a
    number that is low is revenue absorbed, and a number that is high is a false charge.
    """
    if usage is None:
        return
    block = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    if block is None:
        return
    def _field(name: str) -> int | None:
        raw = block.get(name) if isinstance(block, dict) else getattr(block, name, None)
        return raw if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0 else None
    got_in, got_out = _field(input_field), _field(output_field)
    if got_in is None or got_out is None:
        return
    usage.add(got_in, got_out)


def refuse_if_truncated(reason: Any, cutoff: str, *, model: str, budget: int) -> None:
    """Raise `TruncatedResponse` when a provider says the token budget ran out.

    A backend passes the value of its own field and its own name for the cutoff, because
    those two things are the genuinely provider-specific part: OpenAI puts
    `finish_reason` on each choice and calls a truncation `"length"`, while Anthropic
    puts `stop_reason` on the response and calls it `"max_tokens"`. What to do about a
    truncation is not provider-specific, so it is decided once, here, for the same reason
    every other rule in this module is.

    Every call a backend makes through its `_call` reaches this, not extraction alone —
    predicate resolution and `compose_relations` too — so the message names raising the
    budget first and scopes the claims cap to the call it actually applies to. An
    operator reading this out of a `DerivedTermsUnavailable` warning must not be sent to
    tune a setting that does nothing for the call that failed.

    **A reason this cannot read is not treated as a truncation.** Absent, `None`, or a
    name some provider adds after this was written all mean "carry on". That direction is
    deliberate and it matches `record_usage`: guessing that an unfamiliar value means
    "cut off" would turn a working extraction into a failed write, and a response with no
    reason on it is a test double far more often than it is a real answer.

        >>> refuse_if_truncated("stop", "length", model="gpt-4.1", budget=8192)
        >>> refuse_if_truncated(None, "length", model="gpt-4.1", budget=8192)
    """
    if reason != cutoff:
        return
    raise TruncatedResponse(
        f"{model} stopped generating at its {budget}-token limit, so the answer is "
        f"incomplete and none of it can be used. Raise the backend's max_tokens above "
        f"{budget}, or ask for a shorter answer — on an extraction that means capping "
        f"the claims array with MEMVARA_LLM_MAX_CLAIMS (or max_claims=)."
    )
