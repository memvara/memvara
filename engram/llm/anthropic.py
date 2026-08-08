"""Anthropic backend for the two calls that genuinely need a model.

This file is a trust boundary. Everything downstream - the reconciler, the ranker, the
schema registry - treats the dicts returned here as structured facts and acts on them
without re-checking, so a hallucinated `source_index` becomes a claim attributed to the
wrong conversation and a `confidence` of 5.0 silently outranks every honest claim in the
store. Constrained decoding makes malformed output rare, not impossible; validation here
is what makes it harmless.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Sequence

from ..types import Episode, MemoryType
from .base import (
    CLAIM_SCHEMA,
    EXTRACT_SYSTEM,
    PREDICATE_SCHEMA,
    PREDICATE_SYSTEM,
    RESOLVE_SCHEMA,
    RESOLVE_SYSTEM,
)

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")

_CARDINALITY = ("one", "many")
_VOLATILITY = ("static", "slow", "fast")
_MEMORY_TYPES = tuple(m.value for m in MemoryType)

# Matches NullLLM: unknown predicates default to multi-valued, because wrongly retiring a
# true fact is worse than keeping two competing ones.
_PREDICATE_FALLBACK: dict[str, str] = {
    "cardinality": "many",
    "volatility": "slow",
    "memory_type": "semantic",
}

# A model that ignored the schema tells us nothing about how sure it is. Neither extreme
# is safe - 1.0 lets a malformed claim outrank well-formed ones, 0.0 makes it
# unretrievable - so an unreadable confidence lands in the middle.
_UNKNOWN_CONFIDENCE = 0.5

# Ceiling on the known-predicate list sent with every extraction. Sending the whole
# vocabulary is an unbounded per-write token tax that grows fastest exactly when the
# vocabulary is growing fastest, and each new predicate shifts the bytes of the prompt
# prefix and throws away the cache at the same moment. The list is a reuse hint, not an
# enumeration, so a bounded head of it does the same job.
_MAX_KNOWN_PREDICATES = 64

# Ceiling on the candidate list sent when resolving a surface form. Same reasoning, and
# a longer list measurably makes the merge decision worse rather than better.
_MAX_CANDIDATES = 48


def _snake_case(raw: str) -> str:
    """`Lives In` / `livesIn` / `LIVES-IN` -> `lives_in`.

    Predicate identity *is* slot identity: `lives_in` and `livesIn` arriving as different
    predicates means the contradiction between them is invisible and the store quietly
    holds two cities for one person. `PredicateRegistry.normalize()` still runs
    downstream to resolve aliases onto canonical names - this only guarantees it receives
    a well-formed key to look up.
    """
    return _NON_ALNUM.sub("_", _CAMEL.sub("_", raw).strip().lower()).strip("_")


def _first_text(response: Any) -> str:
    """The first text block of a Messages response.

    Tolerates both SDK objects and plain dicts so a test double does not have to
    reimplement the SDK's block types.
    """
    for block in getattr(response, "content", None) or []:
        if isinstance(block, dict):
            if block.get("type") == "text":
                return str(block.get("text") or "")
        elif getattr(block, "type", None) == "text":
            return str(getattr(block, "text", "") or "")
    return ""


def _parse_json_object(response: Any) -> dict[str, Any]:
    text = _first_text(response).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _clamp_confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return _UNKNOWN_CONFIDENCE
    # NaN loses every comparison, so clamping it silently yields 0.0 and buries the claim
    # at the bottom of the ranking rather than admitting we could not read the field.
    if not math.isfinite(value):
        return _UNKNOWN_CONFIDENCE
    return min(1.0, max(0.0, float(value)))


def _source_index(value: Any, n: int) -> int | None:
    # `isinstance(True, int)` is True in Python, and a claim sourced from episode `True`
    # would be attributed to episode 1 - a real provenance bug from a one-character slip.
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 0 <= value < n else None


class AnthropicLLM:
    """Structured extraction and predicate resolution via the Messages API."""

    #: A real backend, so every call it makes is billed to `WriteReceipt.llm_calls`.
    is_noop = False

    def __init__(
        self,
        model: str = "claude-opus-5",
        client: Any = None,
        effort: str = "low",
        max_tokens: int = 8192,
    ) -> None:
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.name = f"anthropic/{model}"
        if client is None:
            client = self._default_client()
        self._client = client

    @staticmethod
    def _default_client() -> Any:
        # Imported here, not at module scope, so `import engram` works in the default
        # offline configuration where the SDK is not installed at all.
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "AnthropicLLM needs the `anthropic` package: pip install 'engram[anthropic]'. "
                "Pass `client=` to inject one, or use NullLLM to run without a model."
            ) from exc
        return anthropic.Anthropic()

    # -- request ------------------------------------------------------------

    def _call(self, system: str, prompt: str, schema: dict[str, Any]) -> Any:
        """One Messages request with constrained decoding.

        The parameter set here is load-bearing and narrower than it looks:

        * structured output goes in `output_config.format`; the top-level `output_format`
          parameter is deprecated;
        * `effort` rides inside the same `output_config`;
        * `temperature` / `top_p` / `top_k` are rejected by this model - passing any of
          them is a 400, not a soft ignore;
        * `thinking` is omitted so adaptive thinking stays on, which is the default and
          the setting extraction accuracy depends on.
        """
        return self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": schema},
            },
        )

    @staticmethod
    def _bounded(names: Sequence[str], limit: int) -> list[str]:
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

    @classmethod
    def _extract_prompt(cls, episodes: Sequence[Episode],
                        known_predicates: Sequence[str]) -> str:
        turns = "\n".join(f"[{i}] {ep.role}: {ep.content}" for i, ep in enumerate(episodes))
        known = ", ".join(cls._bounded(known_predicates, _MAX_KNOWN_PREDICATES))
        return (
            f"Known predicates, reuse one whenever it fits:\n{known or '(none yet)'}\n\n"
            f"Turns:\n{turns}"
        )

    # -- LLM protocol -------------------------------------------------------

    def extract(
        self, episodes: Sequence[Episode], known_predicates: Sequence[str]
    ) -> list[dict[str, Any]]:
        if not episodes:
            return []  # nothing to extract from, and a call we should not pay for
        response = self._call(
            EXTRACT_SYSTEM, self._extract_prompt(episodes, known_predicates), CLAIM_SCHEMA
        )
        raw = _parse_json_object(response).get("claims")
        if not isinstance(raw, list):
            return []

        out: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            # Provenance first: a claim we cannot trace back to a turn is unusable, and
            # guessing an episode for it would corrupt `why()` for every reader after us.
            index = _source_index(item.get("source_index"), len(episodes))
            if index is None:
                continue
            subject = str(item.get("subject") or "").strip()
            predicate = _snake_case(str(item.get("predicate") or ""))
            obj = str(item.get("object") or "").strip()
            if not (subject and predicate and obj):
                continue
            memory_type = str(item.get("memory_type") or "")
            out.append(
                {
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    # Anything other than an explicit -1 is an assertion. A garbled value
                    # must not be read as a retraction, which would invalidate a live fact.
                    "polarity": -1 if item.get("polarity") == -1 else 1,
                    "memory_type": (
                        memory_type if memory_type in _MEMORY_TYPES else MemoryType.SEMANTIC.value
                    ),
                    "confidence": _clamp_confidence(item.get("confidence")),
                    "source_index": index,
                }
            )
        return out

    @staticmethod
    def _spec_fields(parsed: dict[str, Any]) -> dict[str, str]:
        # This answer is cached in the registry forever, so a bad field here is a
        # permanent mistake for that predicate. Any value outside the enum falls back to
        # the conservative default rather than being coerced into a neighbour.
        result = dict(_PREDICATE_FALLBACK)
        for field, allowed in (
            ("cardinality", _CARDINALITY),
            ("volatility", _VOLATILITY),
            ("memory_type", _MEMORY_TYPES),
        ):
            value = parsed.get(field)
            if isinstance(value, str) and value in allowed:
                result[field] = value
        return result

    def resolve_predicate(self, surface: str, candidates: Sequence[str]) -> dict[str, Any]:
        """Merge a novel surface form onto an existing predicate, or declare it new.

        The most consequential validation in this file: `canonical` is echoed straight
        into `PredicateSpec.aliases`, so a hallucinated name would permanently reroute
        every claim using that surface form into a slot nobody looks up. Only a name the
        caller actually offered is accepted - a model that invents one is treated as
        having said "new", which is the recoverable direction.
        """
        offered = self._bounded(candidates, _MAX_CANDIDATES)
        prompt = (
            f"new predicate: {_snake_case(surface)}\n"
            f"existing predicates:\n{', '.join(offered) or '(none yet)'}"
        )
        parsed = _parse_json_object(self._call(RESOLVE_SYSTEM, prompt, RESOLVE_SCHEMA))

        canonical = parsed.get("canonical")
        if not isinstance(canonical, str) or _snake_case(canonical) not in offered:
            canonical = None
        else:
            canonical = _snake_case(canonical)
        return {"canonical": canonical, **self._spec_fields(parsed)}

    def classify_predicate(self, predicate: str, example: str) -> dict[str, str]:
        """Legacy acquisition call, kept for backends and callers that still use it."""
        prompt = f"predicate: {_snake_case(predicate)}\nexample usage: {example}"
        return self._spec_fields(_parse_json_object(
            self._call(PREDICATE_SYSTEM, prompt, PREDICATE_SCHEMA)
        ))
