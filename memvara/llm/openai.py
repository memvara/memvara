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

import base64
import json
from typing import Any, Mapping, Sequence

from ..ingest.errors import MediaUnsupported
from ..types import Episode
from . import _shape, _tools
from .guidance import Guidance, with_guidance
from .base import (
    TOOL_STEP_MAX_TOKENS,
    MalformedToolOutput,
    Message,
    ToolRun,
    ToolSpec,
    CLAIM_SCHEMA,
    DESCRIBE_IMAGE_MAX_TOKENS,
    DESCRIBE_IMAGE_PROMPT,
    DESCRIBE_IMAGE_SYSTEM,
    EXTRACT_SYSTEM,
    JUDGE_SCHEMA,
    JUDGE_SYSTEM,
    MAX_CLAIMS,
    PREDICATE_SCHEMA,
    PREDICATE_SYSTEM,
    RESOLVE_SCHEMA,
    RESOLVE_SYSTEM,
    Usage,
    bounded_claim_schema,
    self_hosted_claim_schema,
)

#: The image types Chat Completions accepts as image input.
IMAGE_TYPES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})

#: The largest image sent. OpenAI documents 20 MB per image.
MAX_IMAGE_BYTES = 20 * 1024 * 1024

#: The media types the transcription endpoint accepts, with the file extension it reads
#: the format from. The video types are here because the endpoint takes those containers
#: and transcribes their audio track. Other video containers (QuickTime, AVI, Matroska)
#: are refused: turning them into one of these needs a video decoder, and this package
#: does not ship one.
AUDIO_TYPES = {
    "audio/flac": "flac",
    "audio/mpeg": "mp3",
    "audio/mp4": "m4a",
    "audio/m4a": "m4a",
    "audio/ogg": "ogg",
    "audio/wav": "wav",
    "audio/webm": "webm",
    "video/mp4": "mp4",
    "video/mpeg": "mpeg",
    "video/webm": "webm",
}

#: The transcription endpoint refuses a file larger than this.
MAX_AUDIO_BYTES = 25 * 1024 * 1024


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


def _finish_reason(response: Any) -> Any:
    """Why generation stopped, off the first choice, or `None` if it does not say."""
    choices = _get(response, "choices") or []
    return _get(choices[0], "finish_reason") if choices else None


def _get(obj: Any, name: str) -> Any:
    """Attribute or key, whichever this object has."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _arguments(raw: Any) -> Any:
    """A tool call's arguments parsed from their JSON string, or the raw value.

    The raw value is returned when it is not a string or does not parse, and
    `_tools.run_loop` then refuses anything that is not an object, which is the one place
    that decision is made.

        >>> _arguments('{"k": 3}'), _arguments("{not json"), _arguments({"k": 3})
        ({'k': 3}, '{not json', {'k': 3})
    """
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except ValueError:
        return raw


class OpenAILLM:
    """Structured extraction and predicate resolution via Chat Completions."""

    #: A real backend, so every call it makes is billed to `WriteReceipt.llm_calls`.
    is_noop = False
    reports_usage = True
    #: `extract` appends `guidance=` to its system message. See `llm.guidance`.
    accepts_guidance = True

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
        extra_body: Mapping[str, Any] | None = None,
        timeout: float | None = None,
        transcription_model: str = "whisper-1",
    ) -> None:
        self.model = model
        # The model `transcribe` asks for, since a chat model cannot transcribe. Every
        # OpenAI-compatible server that transcribes accepts `whisper-1`, including Azure
        # and the common self-hosted Whisper servers.
        self.transcription_model = transcription_model
        self.max_tokens = max_tokens
        # Seconds one call may take. `None` keeps the SDK's default of 600, which is too
        # short for a slow self-hosted model on a long turn. It is sent with each request,
        # so it also applies to a client the caller injected. `chat()` passes its own
        # timeout and does not use this one.
        self.timeout = timeout
        # Extra request fields the SDK does not name, sent on every request. A Qwen3
        # server needs `{"chat_template_kwargs": {"enable_thinking": false}}`, or it
        # spends the whole token budget thinking and returns an empty message.
        self.extra_body = dict(extra_body) if extra_body else None
        # Replacement extraction instructions. Small self-hosted models can read the
        # shipped prompt's "an empty list is correct" as permission to return nothing;
        # `docs/DEPLOY.md` has the measurement. An empty string means the shipped prompt
        # here, whereas the server refuses an empty prompt file.
        self._extract_system = extract_system or EXTRACT_SYSTEM
        # `max_claims` caps the claims array. It is off by default because hosted OpenAI
        # rejects `maxItems` under `strict: True`; see `bounded_claim_schema`. `terse`
        # also makes four claim fields optional; see `self_hosted_claim_schema`. They are
        # separate arguments because the cap only guards against a runaway, while `terse`
        # changes what the model is asked to write.
        if terse:
            self._claim_schema = self_hosted_claim_schema(
                MAX_CLAIMS if max_claims is None else max_claims)
        else:
            self._claim_schema = (
                CLAIM_SCHEMA if max_claims is None else bounded_claim_schema(max_claims))
        # Zero by default: the same turn extracted twice should give the same claim, not
        # two spellings that the reconciler would treat as competing values.
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
              usage: Usage | None = None, *, name: str) -> Any:
        """One Chat Completions request with strict structured output.

        `name` is the schema's name. The API rejects a `json_schema` without one, although
        nothing in the response repeats it.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_completion_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": name,
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        if self.extra_body is not None:
            kwargs["extra_body"] = self.extra_body
        # Absent unless asked for, so a deployment that never set it sends the request it
        # sent before this option existed and keeps the SDK's own default.
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        response = self._client.chat.completions.create(**kwargs)
        # OpenAI names the same two quantities differently from Anthropic; the reading and
        # the refusal-to-guess live in one place so the two backends cannot drift.
        _shape.record_usage(response, usage, "prompt_tokens", "completion_tokens")
        # Checked after the usage is recorded, so a truncated call still appears on the
        # bill. Checked here rather than in `_first_text`, which `chat()` shares, because
        # only the schema path is silent: there an unparseable reply becomes an empty claim
        # list that looks like a turn with no claims. `memvara.select` already raises
        # `ValueError` on a `chat()` reply that will not parse.
        _shape.refuse_if_truncated(
            _finish_reason(response), "length", model=self.model, budget=self.max_tokens)
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
        if self.extra_body is not None:
            kwargs["extra_body"] = self.extra_body
        response = self._client.chat.completions.create(**kwargs)
        _shape.record_usage(response, usage, "prompt_tokens", "completion_tokens")
        return _first_text(response)

    # -- ToolChat protocol ----------------------------------------------------

    def run_tools(self, system: str, messages: Sequence[Message],
                  tools: Sequence[ToolSpec], *, max_steps: int, timeout: float,
                  usage: Usage | None = None) -> ToolRun:
        """A tool-using conversation through Chat Completions' native function calling.

        Each tool is sent as a `strict` function, so the model's arguments match the
        tool's schema; the arguments still arrive as a JSON string, and one that does not
        parse to an object cannot be used. So cannot an answer cut off at
        `TOOL_STEP_MAX_TOKENS` (`finish_reason` of `"length"`) or a refusal, and each of
        those raises `MalformedToolOutput`. `temperature` and `extra_body` are sent as on
        every other request this backend makes. The loop itself, with its step limit,
        deadline and retry, is `_tools.run_loop`.
        """
        convo: list[dict[str, Any]] = [{"role": "system", "content": system}]
        convo += [{"role": m.role, "content": m.content} for m in messages]
        specs = [{"type": "function", "function": {
            "name": t.name, "description": t.description,
            "parameters": dict(t.parameters), "strict": True}} for t in tools]

        def send(remaining: float) -> _tools.Step:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "max_completion_tokens": TOOL_STEP_MAX_TOKENS,
                "temperature": self.temperature,
                "timeout": remaining,
                "messages": convo,
                "tools": specs,
            }
            if self.extra_body is not None:
                kwargs["extra_body"] = self.extra_body
            response = self._client.chat.completions.create(**kwargs)
            _shape.record_usage(response, usage, "prompt_tokens", "completion_tokens")
            choices = _get(response, "choices") or []
            message = _get(choices[0], "message") if choices else None
            if message is None or _get(message, "refusal"):
                raise MalformedToolOutput(f"{self.model} gave no usable message")
            if _finish_reason(response) == "length":
                raise MalformedToolOutput(
                    f"{self.model} stopped at its {TOOL_STEP_MAX_TOKENS}-token limit")
            raw_calls = list(_get(message, "tool_calls") or [])
            calls = [_tools.Call(str(_get(c, "id")), str(_get(_get(c, "function"), "name")),
                                 _arguments(_get(_get(c, "function"), "arguments")))
                     for c in raw_calls]
            return _tools.Step(str(_get(message, "content") or ""), calls, raw_calls)

        def append(step: _tools.Step, results: list[tuple[str, str]]) -> None:
            convo.append({"role": "assistant", "content": step.text or None,
                          "tool_calls": [
                              {"id": call.id, "type": "function",
                               "function": {"name": call.name,
                                            "arguments": json.dumps(call.arguments)}}
                              for call in step.calls]})
            convo.extend({"role": "tool", "tool_call_id": call_id, "content": text}
                         for call_id, text in results)

        return _tools.run_loop(send, append, tools, max_steps=max_steps, timeout=timeout)

    # -- Multimodal protocol ------------------------------------------------

    def describe_image(self, data: bytes, mime: str) -> str:
        """A text description of an image, through one Chat Completions request.

        Refuses, without a request, an image type the API does not accept and an image
        over its size limit, so the caller gets the reason rather than a 400. A refusal by
        the model comes back as an empty string, which `memvara.ingest.extract` reports
        as `no_text`.
        """
        if mime not in IMAGE_TYPES:
            raise MediaUnsupported(
                f"OpenAILLM reads JPEG, PNG, GIF and WebP images, not {mime}")
        if len(data) > MAX_IMAGE_BYTES:
            raise MediaUnsupported(
                f"the image is {len(data)} bytes, and OpenAILLM sends at most "
                f"{MAX_IMAGE_BYTES} bytes per image")
        encoded = base64.b64encode(data).decode("ascii")
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_completion_tokens": DESCRIBE_IMAGE_MAX_TOKENS,
            "messages": [
                {"role": "system", "content": DESCRIBE_IMAGE_SYSTEM},
                {"role": "user", "content": [
                    {"type": "text", "text": DESCRIBE_IMAGE_PROMPT},
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                ]},
            ],
        }
        if self.extra_body is not None:
            kwargs["extra_body"] = self.extra_body
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        return _first_text(self._client.chat.completions.create(**kwargs)).strip()

    def transcribe(self, data: bytes, mime: str) -> str:
        """A transcript of audio, or of a video's audio track, from the transcription
        endpoint.

        Only the sound is transcribed. Frames of a video are not sampled or described,
        because that needs a video decoder and this package does not ship one.
        """
        extension = AUDIO_TYPES.get(mime)
        if extension is None:
            raise MediaUnsupported(
                f"OpenAILLM cannot transcribe {mime}. The transcription endpoint accepts "
                "FLAC, MP3, M4A, Ogg, WAV and WebM audio, and MP4, MPEG and WebM video; "
                "convert the file to one of those first")
        if len(data) > MAX_AUDIO_BYTES:
            raise MediaUnsupported(
                f"the recording is {len(data)} bytes, and the transcription endpoint "
                f"accepts at most {MAX_AUDIO_BYTES} bytes")
        kwargs: dict[str, Any] = {
            "model": self.transcription_model,
            "file": (f"upload.{extension}", data, mime),
        }
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        response = self._client.audio.transcriptions.create(**kwargs)
        # The SDK returns an object with `.text`; a plain-text response format returns a
        # string, and a test double may return a dict.
        text = response if isinstance(response, str) else _get(response, "text")
        return str(text or "").strip()

    # -- LLM protocol -------------------------------------------------------

    def extract(
        self, episodes: Sequence[Episode], known_predicates: Sequence[str],
        *, usage: Usage | None = None, guidance: Guidance | None = None,
    ) -> list[dict[str, Any]]:
        if not episodes:
            return []  # nothing to extract from, and a call we should not pay for
        response = self._call(
            # Appended to whichever prompt is in use, the shipped one or the replacement
            # `extract_system` names: guidance adds a project's rules and never decides
            # which base prompt a deployment runs.
            with_guidance(self._extract_system, guidance),
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
            usage, name="predicate_resolution")
        return _shape.shape_resolution(
            _shape.parse_json_object(_first_text(response)), offered)

    def classify_predicate(self, predicate: str, example: str,
                           *, usage: Usage | None = None) -> dict[str, str]:
        """Legacy acquisition call, kept for backends and callers that still use it."""
        prompt = f"predicate: {_shape.snake_case(predicate)}\nexample usage: {example}"
        response = self._call(PREDICATE_SYSTEM, prompt, PREDICATE_SCHEMA, usage,
                              name="predicate_spec")
        return _shape.spec_fields(_shape.parse_json_object(_first_text(response)))

    # -- ReplacementJudge protocol -------------------------------------------

    def judge_replacement(self, new_text: str, old_text: str,
                          *, usage: Usage | None = None) -> dict[str, bool]:
        """Is `new_text` a newer version of `old_text`? See `JUDGE_SYSTEM`."""
        response = self._call(
            JUDGE_SYSTEM, _shape.judge_prompt(new_text, old_text), JUDGE_SCHEMA, usage,
            name="replacement_verdict")
        return _shape.shape_verdict(_shape.parse_json_object(_first_text(response)))

    def __repr__(self) -> str:
        return f"<OpenAILLM {self.model}>"
