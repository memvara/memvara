"""OpenAI backend: transport and response shape, nothing else.

`pyproject.toml` has declared an `memvara[openai]` extra since the first commit and this
file did not exist, so installing it got you a dependency and no adapter. That is the gap
this closes.

Every rule about what counts as a valid claim lives in `_shape`, shared with
`AnthropicLLM` — see that module for why validation is not a per-provider decision. Two
things here are genuinely OpenAI-specific and neither is optional:

* **`strict: True` structured output.** Without it `response_format` is a suggestion, and
  a schema-shaped suggestion is worse than none: it produces output that looks parseable
  often enough that the validation below stops being exercised in testing and starts
  being load-bearing in production. The schemas in `base.py` already satisfy strict
  mode's requirements — every property `required`, `additionalProperties: false` —
  because those are good ideas independently.
* **Refusals are a first-class field.** `message.refusal` is populated instead of
  `content` when the model declines, and reading `content` alone turns a refusal into an
  empty extraction: zero claims, no error, no receipt entry. The turn silently carries no
  memory. So a refusal is surfaced as text that fails to parse, which lands in the same
  "returned nothing usable" path as any other malformed response rather than pretending
  the model answered.
"""

from __future__ import annotations

from typing import Any, Sequence

from ..types import Episode
from . import _shape
from .base import (
    CLAIM_SCHEMA,
    EXTRACT_SYSTEM,
    MAX_CLAIMS,
    PREDICATE_SCHEMA,
    PREDICATE_SYSTEM,
    RESOLVE_SCHEMA,
    RESOLVE_SYSTEM,
    Usage,
    bounded_claim_schema,
    self_hosted_claim_schema,
)

#: `json_schema` requires a name. It is echoed back in nothing we read, but the API
#: rejects the request without one.
_SCHEMA_NAMES = {
    id(CLAIM_SCHEMA): "claims",
    id(RESOLVE_SCHEMA): "predicate_resolution",
    id(PREDICATE_SCHEMA): "predicate_spec",
}


def _first_text(response: Any) -> str:
    """The assistant message text, or `""` for anything we should not act on.

    Tolerates SDK objects and plain dicts so a test double need not reimplement the SDK's
    model classes. A refusal returns `""` deliberately: see the module docstring.
    """
    choices = _get(response, "choices") or []
    if not choices:
        return ""
    message = _get(choices[0], "message")
    if message is None:
        return ""
    if _get(message, "refusal"):
        return ""
    return str(_get(message, "content") or "")


def _get(obj: Any, name: str) -> Any:
    """Attribute or key, whichever this object has."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


class OpenAILLM:
    """Structured extraction and predicate resolution via Chat Completions."""

    #: A real backend, so every call it makes is billed to `WriteReceipt.llm_calls`.
    is_noop = False
    reports_usage = True

    def __init__(
        self,
        model: str = "gpt-4.1",
        client: Any = None,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        max_claims: int | None = None,
        base_url: str | None = None,
        extract_system: str | None = None,
        terse: bool = False,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        # Replacement extraction instructions, for the same self-hosted case `max_claims`
        # serves. `EXTRACT_SYSTEM` closes by saying an empty list is a correct answer and
        # the common case, which is true and is what a model able to weigh salience across
        # a long turn needs to hear. A small one reads it as permission: measured
        # 2026-09-03, phi-4-mini-instruct returned nothing at all on inputs past roughly
        # 1,300 tokens until that sentence was removed, on the same prompt and episodes.
        # An override rather than a rewrite, because the shipped wording is right for the
        # models it was written for and a small-model accommodation applied to every
        # provider is a change nobody asked for.
        #
        # `or` rather than `is not None`, so an empty string means "use the shipped
        # prompt" here. That is deliberately *not* what the server does: it refuses an
        # empty file, because an operator who named one meant to change something. The two
        # answers differ because the inputs do — a caller passing `""` in Python has an
        # object it can inspect, and sending a model an empty system message is never what
        # it wanted. A caller that must distinguish "unset" from "empty" should not
        # compute the argument down to a string first.
        self._extract_system = extract_system or EXTRACT_SYSTEM
        # Cap the claims array, for a self-hosted server reached through this backend. Off
        # by default because hosted OpenAI rejects `maxItems` under `strict: True` — see
        # `bounded_claim_schema`, which carries the reasoning and the measurement. Built
        # once here rather than per call, since it is the same dict every time.
        # `terse` additionally drops the five defaultable fields out of `required`, so the
        # model stops spending tokens on `"when":null,"amount":null,"unit":null` for every
        # claim. Measured on the shipped shape, that is 413 tokens for eight claims against
        # 229 — a little under half the generation, which on a CPU-hosted model is most of
        # the wall time. `self_hosted_claim_schema` carries the measurement and the one
        # consequence, which is that `confidence` stops being a number the model chose.
        #
        # A separate argument from `max_claims` rather than a second meaning for it. Both
        # describe the same self-hosted case, but they are different trades: the cap
        # protects against a runaway and costs nothing, while this one buys latency and
        # moves ranking. An operator who set `MEMVARA_LLM_MAX_CLAIMS` asked for the first
        # and must not silently receive the second.
        if terse:
            self._claim_schema = self_hosted_claim_schema(
                MAX_CLAIMS if max_claims is None else max_claims)
        else:
            self._claim_schema = (
                CLAIM_SCHEMA if max_claims is None else bounded_claim_schema(max_claims))
        # Extraction is a parsing task, not a creative one, and the same turn arriving
        # twice should produce the same claim rather than two spellings of it that the
        # reconciler then has to treat as competing values.
        self.temperature = temperature
        self.name = f"openai/{model}"
        if client is None:
            client = self._default_client(base_url)
        self._client = client

    @staticmethod
    def _default_client(base_url: str | None = None) -> Any:
        # Imported here, not at module scope, so `import memvara` works in the default
        # offline configuration where the SDK is not installed at all.
        try:
            # The whole point is that this is absent in the default install.
            import openai  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "OpenAILLM needs the `openai` package: pip install 'memvara[openai]'. "
                "Pass `client=` to inject one, or use NullLLM to run without a model."
            ) from exc
        # `base_url` matters to the hosted service, which stores one per organisation
        # rather than always reaching api.openai.com — an Azure or self-hosted
        # OpenAI-compatible endpoint. `None` keeps the SDK's own default.
        return openai.OpenAI(base_url=base_url) if base_url else openai.OpenAI()

    # -- request ------------------------------------------------------------

    def _call(self, system: str, prompt: str, schema: dict[str, Any],
              usage: Usage | None = None, *, name: str | None = None) -> Any:
        response = self._client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": name or _SCHEMA_NAMES.get(id(schema), "result"),
                    "strict": True,
                    "schema": schema,
                },
            },
        )
        # OpenAI names the same two quantities differently from Anthropic; the reading and
        # the refusal-to-guess live in one place so the two backends cannot drift.
        _shape.record_usage(response, usage, "prompt_tokens", "completion_tokens")
        return response

    # -- Chat protocol --------------------------------------------------------

    def chat(self, system: str, prompt: str, *, json_object: bool,
             max_completion_tokens: int, timeout: float,
             usage: Usage | None = None) -> str:
        """Plain chat completion for `memvara.select` — no schema, no temperature.

        Deliberately not routed through `_call`: `_call` hard-codes strict `json_schema`
        and a `temperature`, and the selector's prompt was measured with neither — the
        request this sends has to match `extract.py`'s, byte for byte in the messages and
        exactly in these two parameters.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_completion_tokens": max_completion_tokens,
            "timeout": timeout,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        if json_object:
            kwargs["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(**kwargs)
        _shape.record_usage(response, usage, "prompt_tokens", "completion_tokens")
        return _first_text(response)

    # -- LLM protocol -------------------------------------------------------

    def extract(
        self, episodes: Sequence[Episode], known_predicates: Sequence[str],
        *, usage: Usage | None = None,
    ) -> list[dict[str, Any]]:
        if not episodes:
            return []  # nothing to extract from, and a call we should not pay for
        response = self._call(
            self._extract_system,
            _shape.extract_prompt(episodes, known_predicates),
            self._claim_schema,
            usage,
            name="claims",
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

    def classify_predicate(self, predicate: str, example: str,
                           *, usage: Usage | None = None) -> dict[str, str]:
        """Legacy acquisition call, kept for backends and callers that still use it."""
        prompt = f"predicate: {_shape.snake_case(predicate)}\nexample usage: {example}"
        response = self._call(PREDICATE_SYSTEM, prompt, PREDICATE_SCHEMA, usage)
        return _shape.spec_fields(_shape.parse_json_object(_first_text(response)))

    def __repr__(self) -> str:
        return f"<OpenAILLM {self.model}>"
