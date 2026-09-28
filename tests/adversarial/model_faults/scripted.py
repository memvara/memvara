"""A scripted stand-in for a model, for the tests of a model that misbehaves.

`ScriptedModel` implements the four protocols in `memvara/llm/base.py` that memvara calls,
as both shipped backends (`AnthropicLLM` and `OpenAILLM`) do: `LLM` (`extract`,
`resolve_predicate`, `classify_predicate`), `Chat` (`chat`), `ToolChat` (`run_tools`) and
`ReplacementJudge` (`judge_replacement`). A store built with it therefore takes the same
paths as a store built with a real backend. It runs in the test's own process and never
reaches a network.

Each method answers with the next reply scripted for it, in order, and every call it
answers is recorded in `calls`. A call that arrives after its method's script has run out
raises `Unscripted` and is recorded in `unscripted` instead. Memvara catches most model
failures and carries on, so that exception alone would go unnoticed. `check_scripts` turns
it into a test failure, and the `scripted` fixture calls it when a test ends. For the same
reason every exception the model raises, scripted or from shaping a reply, is kept in
`failures` with the method that raised it, so a test can name what failed and where.

The model stands in for the provider and nothing more. What a shipped backend does with
the provider's answer is done here by memvara's own code, so a malformed answer meets the
code a real one would meet:

- A `Text` reply is what the provider sent back. `extract`, `resolve_predicate`,
  `classify_predicate` and `judge_replacement` turn it into their return value with
  `memvara.llm._shape`, the validation both shipped backends run. `chat` returns it
  unchanged, as both backends do.
- A `Truncated` reply is text the provider cut off at its token limit, and said so. The
  four methods above raise `TruncatedResponse` for it through
  `_shape.refuse_if_truncated`, as both backends do, and `chat` returns the cut text, as
  both backends do. In `run_tools` it is an answer that cannot be used.
- `run_tools` runs `memvara.llm._tools.run_loop`, the tool loop both backends share, and
  each scripted `Answer` stands in for one provider response. The step limit, the retry
  and the time budget are memvara's own.

Anything else in a script is returned as it is, which is how a test plays a backend that
does no validation of its own. An exception in a script is raised.

Provider errors are raised the way the provider SDKs raise them. Neither SDK is installed
in the test environment, so the classes below copy the three things memvara reads from
the real ones: the class name, which is how the tool loop recognises a timeout
(`memvara.llm._tools.is_timeout`); the `status_code` attribute, which is how a read stage
tells a rejected key from another provider error (`memvara.select.chat.call_chat`); and
the class hierarchy, in which a timeout is not a subclass of Python's `TimeoutError`. The
`anthropic` and `openai` SDKs both use these class names.

The model keeps its own clock, `clock()`, in seconds from zero. It moves only when a
`Late` reply is answered. Memvara reads it wherever it is handed it: the tool loop, and a
query rewriter or synthesizer built with `clock=model.clock`, as `handles.with_model`
builds them. No test sleeps.

    >>> model = ScriptedModel(chat=[Text('{"queries": []}')])
    >>> model.chat("system", "prompt", json_object=True, max_completion_tokens=300,
    ...            timeout=10.0)
    '{"queries": []}'
    >>> model.count("chat"), model.unscripted
    (1, [])
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from memvara.llm import _shape, _tools
from memvara.llm.base import MalformedToolOutput, Message, ToolRun, ToolSpec, Usage
from memvara.llm.guidance import Guidance
from memvara.types import Episode

#: The token budget a `Truncated` reply says the provider ran out of: the default
#: `max_tokens` of both shipped backends.
BUDGET = 8192

#: The methods the model answers, as `counts()` and `queue()` name them.
METHODS = ("extract", "resolve_predicate", "classify_predicate", "chat", "run_tools",
           "judge_replacement")


class APIError(Exception):
    """A failed request to the provider. Stands in for `anthropic.APIError` and
    `openai.APIError`, which every request error in both SDKs is an instance of."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class APIConnectionError(APIError):
    """The request got no answer, because the connection was refused or reset."""


class APITimeoutError(APIConnectionError):
    """The provider's client stopped waiting for an answer.

    As in both SDKs, this is not a subclass of Python's `TimeoutError` and it has no
    `status_code`. Memvara's tool loop recognises it by its class name.
    """

    def __init__(self) -> None:
        super().__init__("Request timed out.")


class APIStatusError(APIError):
    """The provider answered with an HTTP error. `status_code` is its code."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class RateLimitError(APIStatusError):
    """HTTP 429: the provider's rate limit."""

    def __init__(self, message: str = "rate limited") -> None:
        super().__init__(message, status_code=429)


class AuthenticationError(APIStatusError):
    """HTTP 401: the provider rejected the key."""

    def __init__(self, message: str = "invalid api key") -> None:
        super().__init__(message, status_code=401)


@dataclass(frozen=True)
class Text:
    """What the provider sent back, as text."""

    text: str


@dataclass(frozen=True)
class Truncated:
    """Text the provider cut off at its token limit, and said so."""

    text: str = ""


@dataclass(frozen=True)
class Late:
    """`reply`, arriving `seconds` after its request, as the model's clock counts."""

    reply: object
    seconds: float


@dataclass(frozen=True)
class Answer:
    """One provider response inside `run_tools`: its text, and its tool calls.

    Each call is a pair of a tool's name and its arguments, and both go to memvara's tool
    loop exactly as they are written, so a script can name a tool that was not offered or
    pass arguments that are not an object.
    """

    text: str = ""
    calls: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True)
class Forever:
    """`reply` to this call and to every later call of the same method.

    `Forever(Late(x, s))` and `Late(Forever(x), s)` mean the same thing: `x` answers every
    later call, and each answer arrives `s` seconds late.
    """

    reply: object


@dataclass(frozen=True)
class Call:
    """One call the model answered, with the arguments a test may want to read."""

    method: str
    args: Mapping[str, Any]


class Unscripted(Exception):
    """A call arrived after its method's script had run out."""


#: What a reply in the `run_tools` script may be, once `Late` and `Forever` are unwrapped.
_TOOL_REPLIES = (Answer, Text, Truncated, BaseException)


def _unwrapped(reply: object) -> object:
    while isinstance(reply, (Late, Forever)):
        reply = reply.reply
    return reply


def check_scripts(models: Sequence["ScriptedModel"]) -> None:
    """Raise `AssertionError` naming every call any of `models` got after its script ran
    out. The `scripted` fixture calls this when a test ends."""
    stray = [call for model in models for call in model.unscripted]
    if stray:
        raise AssertionError(f"a scripted model was called after its script ran out: "
                             f"{stray}")


class ScriptedModel:
    """A model whose every answer is scripted. See the module docstring."""

    name = "scripted/model"
    #: A model as far as memvara can tell, so every call it answers is billed.
    is_noop = False
    #: It reports no token usage, which a backend may do.
    reports_usage = False
    #: `extract` takes `guidance=`, as both shipped backends do.
    accepts_guidance = True

    def __init__(self, *, extract: Sequence[object] = (), resolve: Sequence[object] = (),
                 classify: Sequence[object] = (), chat: Sequence[object] = (),
                 tools: Sequence[object] = (), judge: Sequence[object] = ()) -> None:
        self._scripts: dict[str, list[object]] = {method: [] for method in METHODS}
        #: Every call answered from a script, in order.
        self.calls: list[Call] = []
        #: Every call that arrived after its method's script had run out.
        self.unscripted: list[Call] = []
        #: One entry per `run_tools` call: the system message, the opening messages, the
        #: names of the tools offered, `max_steps` and `timeout`.
        self.runs: list[dict[str, Any]] = []
        #: The text each tool answered, in order: what a real model would read back.
        self.tool_results: list[str] = []
        #: Every exception the model raised, with the method that raised it, in order.
        #: Memvara catches most of them, so this is where a test finds them.
        self.failures: list[tuple[str, BaseException]] = []
        self._now = 0.0
        for method, replies in zip(METHODS, (extract, resolve, classify, chat, tools,
                                             judge)):
            self.queue(method, *replies)

    def queue(self, method: str, *replies: object) -> None:
        """Add `replies` to the end of `method`'s script."""
        if method not in self._scripts:
            raise ValueError(f"{method!r} is not one of {METHODS}")
        if method == "run_tools":
            for reply in replies:
                if not isinstance(_unwrapped(reply), _TOOL_REPLIES):
                    raise TypeError(f"run_tools cannot send {reply!r}. Script an Answer, "
                                    "a Text, a Truncated or an exception.")
        self._scripts[method].extend(replies)

    def clock(self) -> float:
        """Seconds since the model was made, as the model counts them."""
        return self._now

    def count(self, method: str | None = None) -> int:
        """How many calls the model answered: of `method`, or of every method."""
        return sum(1 for call in self.calls if method in (None, call.method))

    def counts(self) -> Counter[str]:
        """How many calls the model answered, by method."""
        return Counter(call.method for call in self.calls)

    # -- the protocols ---------------------------------------------------------------

    def extract(self, episodes: Sequence[Episode], known_predicates: Sequence[str], *,
                usage: Usage | None = None, guidance: Guidance | None = None) -> Any:
        reply = self._next("extract", turns=[ep.content for ep in episodes],
                           known_predicates=list(known_predicates), guidance=guidance)
        return self._shaped("extract", reply,
                            lambda parsed: _shape.shape_claims(parsed, len(episodes)))

    def resolve_predicate(self, surface: str, candidates: Sequence[str], *,
                          usage: Usage | None = None) -> Any:
        offered = _shape.bounded(candidates, _shape.MAX_CANDIDATES)
        reply = self._next("resolve_predicate", surface=surface, candidates=offered)
        return self._shaped("resolve_predicate", reply,
                            lambda parsed: _shape.shape_resolution(parsed, offered))

    def classify_predicate(self, predicate: str, example: str, *,
                           usage: Usage | None = None) -> Any:
        reply = self._next("classify_predicate", predicate=predicate, example=example)
        return self._shaped("classify_predicate", reply, _shape.spec_fields)

    def judge_replacement(self, new_text: str, old_text: str, *,
                          usage: Usage | None = None) -> Any:
        reply = self._next("judge_replacement", new=new_text, old=old_text)
        return self._shaped("judge_replacement", reply, _shape.shape_verdict)

    def chat(self, system: str, prompt: str, *, json_object: bool,
             max_completion_tokens: int, timeout: float,
             usage: Usage | None = None) -> Any:
        reply = self._next("chat", system=system, prompt=prompt, timeout=timeout)
        # Neither backend checks why a chat reply stopped, so a cut reply arrives as it is.
        if isinstance(reply, (Text, Truncated)):
            return reply.text
        return reply

    def run_tools(self, system: str, messages: Sequence[Message],
                  tools: Sequence[ToolSpec], *, max_steps: int, timeout: float,
                  usage: Usage | None = None) -> ToolRun:
        self.runs.append({"system": system, "messages": list(messages),
                          "tools": [tool.name for tool in tools],
                          "max_steps": max_steps, "timeout": timeout})

        def send(remaining: float) -> _tools.Step:
            reply = self._next("run_tools", remaining=remaining)
            if isinstance(reply, Truncated):
                # What both backends raise for an answer cut off at its token limit.
                unusable = MalformedToolOutput(f"{self.name} stopped at its token limit")
                self.failures.append(("run_tools", unusable))
                raise unusable
            if isinstance(reply, Text):
                return _tools.Step(reply.text)
            assert isinstance(reply, Answer)  # `queue` refused anything else
            calls = [_tools.Call(f"call_{len(self.calls)}_{i}", str(name), arguments)
                     for i, (name, arguments) in enumerate(reply.calls)]
            return _tools.Step(reply.text, calls, reply)

        def append(step: _tools.Step, results: list[tuple[str, str]]) -> None:
            self.tool_results.extend(text for _, text in results)

        return _tools.run_loop(send, append, tools, max_steps=max_steps, timeout=timeout,
                               clock=self.clock)

    # -- the script ----------------------------------------------------------------

    def _next(self, method: str, **args: Any) -> object:
        call = Call(method, args)
        script = self._scripts[method]
        if not script:
            self.unscripted.append(call)
            raise Unscripted(f"{method} was called after its script ran out, with {args}")
        self.calls.append(call)
        # `Forever` and `Late` may wrap a reply in either order, and both orders mean the
        # same: the entry stays at the head of the script when any wrapper is `Forever`,
        # and every `Late` around the reply moves the clock on each call it answers.
        reply, forever = script[0], False
        while isinstance(reply, (Forever, Late)):
            if isinstance(reply, Forever):
                forever = True
            else:
                self._now += reply.seconds
            reply = reply.reply
        if not forever:
            script.pop(0)
        if isinstance(reply, BaseException):
            self.failures.append((method, reply))
            raise reply
        return reply

    def _shaped(self, method: str, reply: object,
                shape: Callable[[dict[str, Any]], Any]) -> Any:
        """A schema-constrained method's return value for `reply`, made the way a
        shipped backend makes it. An exception from memvara's shaping is kept in
        `failures` before it is raised, because the write path will not show it."""
        try:
            if isinstance(reply, Truncated):
                # Anthropic's word for the event. `refuse_if_truncated` raises for any
                # backend's own word, so which one is used here changes nothing.
                _shape.refuse_if_truncated("max_tokens", "max_tokens", model=self.name,
                                           budget=BUDGET)
            if isinstance(reply, Text):
                return shape(_shape.parse_json_object(reply.text))
        except Exception as error:
            self.failures.append((method, error))
            raise
        return reply
