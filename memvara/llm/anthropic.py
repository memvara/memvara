"""Anthropic backend: transport and response shape, nothing else.

Every rule about what counts as a valid claim lives in `_shape`, shared with the other
backends — see that module for why validation is not a per-provider decision. What is
genuinely Anthropic-specific is here: the Messages request, and finding the text in a
response made of typed content blocks.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

from ..types import Episode
from . import _shape
from .base import (
    CLAIM_SCHEMA,
    EXTRACT_SYSTEM,
    PREDICATE_SCHEMA,
    PREDICATE_SYSTEM,
    COMPOSE_SCHEMA,
    COMPOSE_SYSTEM,
    RESOLVE_SCHEMA,
    RESOLVE_SYSTEM,
    Usage,
)


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


class AnthropicLLM:
    """Structured extraction and predicate resolution via the Messages API."""

    #: A real backend, so every call it makes is billed to `WriteReceipt.llm_calls`.
    is_noop = False
    reports_usage = True

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
        # Imported here, not at module scope, so `import memvara` works in the default
        # offline configuration where the SDK is not installed at all.
        try:
            # The whole point is that this is absent in the default install.
            import anthropic  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "AnthropicLLM needs the `anthropic` package: pip install 'memvara[anthropic]'. "
                "Pass `client=` to inject one, or use NullLLM to run without a model."
            ) from exc
        # Validated here rather than left to the first write. `openai.OpenAI()` refuses
        # to construct without a key and this did not, so the same documented line —
        # `Memvara(..., llm=XLLM())` — failed at two different moments depending on the
        # backend: one at startup, one on the first turn that reached tier 2, which in a
        # server is after the deployment looked healthy. Fail-fast is the better of the
        # two behaviours, so this is the one that moved.
        #
        # Checked rather than caught: the SDK reads the variable lazily, so there is no
        # exception to catch until a request is already in flight.
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError(
                "AnthropicLLM found no ANTHROPIC_API_KEY. Set it, pass `client=` to "
                "inject a configured client, or use NullLLM to run without a model. "
                "Checked at construction on purpose: the alternative is a server that "
                "starts clean and fails on the first turn that needs extraction."
            )
        return anthropic.Anthropic()

    # -- request ------------------------------------------------------------

    def _call(self, system: str, prompt: str, schema: dict[str, Any],
              usage: Usage | None = None) -> Any:
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
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": schema},
            },
        )
        _shape.record_usage(response, usage, "input_tokens", "output_tokens")
        return response

    # -- Chat protocol --------------------------------------------------------

    def chat(self, system: str, prompt: str, *, json_object: bool,
             max_completion_tokens: int, timeout: float,
             usage: Usage | None = None) -> str:
        """Plain chat completion for `memvara.select` — no schema, no `output_config`.

        The Messages API has no `response_format` equivalent to ask for loosely-typed
        JSON; `memvara.select`'s prompt already asks for JSON in prose, the same way it
        does for OpenAI, so `json_object` is accepted for protocol symmetry and does not
        change the request.
        """
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_completion_tokens,
            timeout=timeout,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        _shape.record_usage(response, usage, "input_tokens", "output_tokens")
        return _first_text(response)

    # -- LLM protocol -------------------------------------------------------

    def extract(
        self, episodes: Sequence[Episode], known_predicates: Sequence[str],
        *, usage: Usage | None = None,
    ) -> list[dict[str, Any]]:
        if not episodes:
            return []  # nothing to extract from, and a call we should not pay for
        response = self._call(
            EXTRACT_SYSTEM,
            _shape.extract_prompt(episodes, known_predicates),
            CLAIM_SCHEMA,
            usage,
        )
        return _shape.shape_claims(
            _shape.parse_json_object(_first_text(response)), len(episodes))

    def resolve_predicate(self, surface: str, candidates: Sequence[str],
                          *, usage: Usage | None = None) -> dict[str, Any]:
        """Merge a novel surface form onto an existing predicate, or declare it new."""
        offered = _shape.bounded(candidates, _shape.MAX_CANDIDATES)
        response = self._call(
            RESOLVE_SYSTEM, _shape.resolve_prompt(surface, offered), RESOLVE_SCHEMA,
            usage)
        return _shape.shape_resolution(
            _shape.parse_json_object(_first_text(response)), offered)

    def compose_relations(self, predicates: Sequence[str]) -> dict[str, int]:
        """Relation terms that are a composition of two or more of these predicates.

        Asked once per vocabulary and never per query — `retrieve/compose` explains why,
        and `retrieve/intent.py` promises to be model-free. What comes back is a term to
        arity map; the caller filters it against the store's own predicate names.
        """
        offered = _shape.bounded(predicates, _shape.MAX_KNOWN_PREDICATES)
        response = self._call(
            COMPOSE_SYSTEM, _shape.compose_prompt(offered), COMPOSE_SCHEMA, None)
        return _shape.shape_composition(
            _shape.parse_json_object(_first_text(response)))

    def classify_predicate(self, predicate: str, example: str,
                           *, usage: Usage | None = None) -> dict[str, str]:
        """Legacy acquisition call, kept for backends and callers that still use it."""
        prompt = f"predicate: {_shape.snake_case(predicate)}\nexample usage: {example}"
        response = self._call(PREDICATE_SYSTEM, prompt, PREDICATE_SCHEMA, usage)
        return _shape.spec_fields(_shape.parse_json_object(_first_text(response)))
