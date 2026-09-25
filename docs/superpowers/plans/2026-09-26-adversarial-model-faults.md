# A model that misbehaves (A7) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show that whatever a model sends back, memvara keeps every turn it was given, retires and erases nothing on the model's word alone, serves the plain read when a read stage fails, and makes exactly the number of model calls `docs/INTERNALS.md` states.

**Architecture:**
- `tests/adversarial/model_faults/scripted.py` holds `ScriptedModel`, an object in the test's own process that implements the four protocols in `memvara/llm/base.py` that memvara calls: `LLM`, `Chat`, `ToolChat` and `ReplacementJudge`. Both shipped backends implement the same four, so a store built with the scripted model takes the same paths as one built with a real backend. Each method answers with the next reply scripted for it, and every call is recorded.
- The scripted model replaces the provider's round trip and nothing more. A text reply is turned into a return value by `memvara.llm._shape`, the validation both shipped backends run, and `run_tools` runs `memvara.llm._tools.run_loop`, the tool loop both backends share. So a malformed reply meets the code a real one would meet, and the step limit, the retry and the time budget of an agentic run are memvara's own.
- Provider errors are raised the way the `anthropic` and `openai` SDKs raise them. Neither SDK is installed in the test environment, so stand-in classes copy the three things memvara reads from the real ones: the class name, the `status_code` attribute, and the class hierarchy.
- `tests/adversarial/model_faults/handles.py` builds two handles on one store, one with the scripted model and one with no model, so every failed read can be compared with the read a store with no model serves. It also records the state of every claim before a write and names what the write did to each one afterwards.
- No library code changes. The tests call the public API (`Memvara.add`, `reextract`, `remember`, `search`, `recall`, `forget_matching`) and the in-process MCP server (`MemvaraMCPServer`).

**Tech Stack:** Python 3.10–3.13, pytest, `memvara.llm._shape`, `memvara.llm._tools.run_loop`, `memvara.select.stages.QueryRewriter` and `Synthesizer`, `memvara.select.model.ModelSelector`, `memvara.server.MemvaraMCPServer`, `harness.stores`.

**Spec:** `docs/superpowers/specs/2026-09-25-adversarial-test-suite-design.md`, section "Phase 2: The agent-facing deterministic tiers", row A7. The rules the tests check are stated in `docs/INTERNALS.md` (invariant 1, and the section on `write/pipeline.py`), `docs/claude/write-pipeline.md` and `docs/claude/retrieval.md`.

## Global Constraints

- Everything runs offline, with no API key. The scripted model is the only model any test uses, and no test reaches a provider or the network.
- Every `Memvara` a test builds passes `embedder=HashingEmbedder(dim=512)`.
- No test sleeps. Time on the model's side is the scripted model's own clock, which memvara reads only where it is handed it.
- The files are everything under `tests/adversarial/model_faults/`, this plan, and one section of `docs/claude/testing.md`, placed just before that page's final line that starts with `Next:`. Every folder has an `__init__.py`, every test file is named `test_adv_*.py`, and anything slow goes in `tests/adversarial/model_faults/nightly/`.
- The fast tier of this plan takes about 5 seconds in total on a laptop.
- Agentic extraction and extraction chunks are off by default, which the design lists as documented behaviour. The tests that exercise them switch them on with `write_agentic_extraction=True` and `write_extraction_chunks=True`.
- A test that meets documented behaviour asserts that behaviour and cites where it is documented.
- A failure is a finding. A finding that matches the "In scope" section of `SECURITY.md` is written into no committed file, and its test is left out. Any other finding's failing test is left out of the commit and kept under `local/` for the maintainer, who files the issue and pins the test. No test is skipped or weakened so that it passes.
- `tests/harness/checklist.py` does not exist on `origin/main`, so the tests carry no `covers` marks. `tests/harness/known_bugs.py` is not edited.
- Commits name their files. There is no AI attribution anywhere.
- Write plainly: every sentence must be understood on its first reading.

## What must hold

1. **The episode is always stored, whatever the model does.** Episodes commit before tier 2 calls a model ("Episodes commit on their own, first", `write/pipeline.py`), and a failed extraction keeps them: "A provider 429 is not a reason to lose a transcript" (`WritePipeline._extraction_failed`). The fast-path claims of the same batch are kept too.
2. **Nothing is retired or erased because of model output alone.** INTERNALS invariant 1: "A model still cannot retire or erase anything, because a proposed end is a retraction with `close="ended"`." `docs/claude/write-pipeline.md`: "a model can end a memory but never retire or erase one, and a replacement the reconciler does not accept leaves the old memory live". Replacement advice "closes nothing".
3. **A read stage that fails serves the plain read, byte for byte, as a store with no model would.** INTERNALS invariant 1 says each of the three read stages "serves the plain read on every outcome but `applied`". Two lines are the documented exceptions, both in `Memvara.recall`'s docstring: a `synthesize=True` block starts with a `RECALL_UNSYNTHESIZED` line naming why there is no summary, and a `ranked=True` block ends with a `RECALL_UNRANKED` line naming why the model did not rank. Everything else in the block must match.
4. **The number of model calls per operation matches INTERNALS.** Task 7 holds the table, with the sentence of INTERNALS each row checks. Every write's `receipt.llm_calls` must equal the number of calls the scripted model answered.

## Review Focus

1. A model that restates one fact hundreds or thousands of times in one reply must leave one claim, reinforced, and not one row per restatement. Pinned in Task 2 (500 restatements) and Task 8 (10,000).
2. A query rewrite that returns more alternatives than allowed, or repeats the question, must still cost one model call and at most four retrievals. Pinned in Task 6.
3. A selector reply that is only partly readable (an index out of range, a repeated index, an index that is not an integer, an empty span) must keep only the entries it can read, in the reranked order, and mark no other turn as selected. Pinned in Task 6.
4. An agentic model that asks for another user's claim by id must learn nothing about it, and must not be able to end it. Pinned in Task 4.
5. A tool call with an unusable `k` (0, negative, 10,000, text, a boolean) must be clamped or defaulted, and must not fail the run. Pinned in Task 5.

## Files

- `tests/adversarial/model_faults/__init__.py`
- `tests/adversarial/model_faults/scripted.py`: the scripted model, its reply kinds, and the provider error stand-ins.
- `tests/adversarial/model_faults/handles.py`: handles on one store with and without the model, `seed`, `ledger`, `fates`, `turns`, `rendered` and `tool_text`.
- `tests/adversarial/model_faults/conftest.py`: the `scripted` fixture, which fails a test whose model was called after its script ran out.
- `tests/adversarial/model_faults/test_adv_scripted_model.py`: the scripted model's own tests, and the tests of `handles.fates`.
- `tests/adversarial/model_faults/test_adv_write_output.py`: malformed output, invented predicates, and many claims in one reply.
- `tests/adversarial/model_faults/test_adv_write_errors.py`: timeouts and 429s on the write path.
- `tests/adversarial/model_faults/test_adv_retire_erase.py`: proposals to retire or erase a stored claim.
- `tests/adversarial/model_faults/test_adv_tool_loops.py`: runaway tool loops, and the other ways an agentic run fails.
- `tests/adversarial/model_faults/test_adv_read_stages.py`: read stages that fail.
- `tests/adversarial/model_faults/test_adv_call_counts.py`: model calls per operation, against INTERNALS.
- `tests/adversarial/model_faults/nightly/__init__.py` and `nightly/test_adv_ten_thousand_claims.py`: 10,000 claims in one reply.

## How each test is shown to fail for the right reason

The tests in Tasks 2 to 8 check behaviour memvara already has, so a new test that passes proves nothing until it has been seen to catch the fault it exists for. Each of those tasks names planted faults. For each one, copy `memvara/` into a scratch directory under `$TMPDIR`, make the one change there, and run the task's tests with `PYTHONPATH` set to the scratch directory first. The tests named for that fault must fail, with a message that points at the fault. Then run them against the real tree, where they must pass. A test that no planted fault can make fail is rewritten until one does.

---

### Task 1: The scripted model, its handles, and its own tests

**Files:**
- Create: `tests/adversarial/model_faults/__init__.py`, `scripted.py`, `handles.py`, `conftest.py`
- Test: `tests/adversarial/model_faults/test_adv_scripted_model.py`
- Modify: `docs/claude/testing.md` (the new section "A model that misbehaves", first paragraphs)

**Interfaces:**
- Produces, in `scripted.py`:
  - Provider errors: `APIError(message)`, `APIConnectionError(message)`, `APITimeoutError()`, `APIStatusError(message, *, status_code)`, `RateLimitError(message="rate limited")` with `status_code` 429, and `AuthenticationError(message="invalid api key")` with `status_code` 401.
  - Reply kinds: `Text(text)`, `Truncated(text="")`, `Late(reply, seconds)`, `Answer(text="", calls=())` where `calls` is a tuple of `(tool name, arguments)`, and `Forever(reply)`. Any other object is returned as it is, and an exception is raised.
  - `Call(method, args)`, `Unscripted`, `METHODS`, `BUDGET = 8192`, and `check_scripts(models)`.
  - `ScriptedModel(*, extract=(), resolve=(), classify=(), chat=(), tools=(), judge=())` with `calls`, `unscripted`, `runs`, `tool_results`, `queue(method, *replies)`, `clock()`, `count(method=None)` and `counts()`, and the protocol methods `extract`, `resolve_predicate`, `classify_predicate`, `chat`, `run_tools` and `judge_replacement`.
- Produces, in `handles.py`: `USER = "u1"`, `FAST_TURN = "My name is Ada."`, `MODEL_TURN = "The team relocated the whole office to Porto over the summer."`, `with_model(model, path=None, *, store=None, user=USER, ranked=False, **options) -> Memvara`, `without_model(store, *, user=USER) -> Memvara`, `Row(state, valid_to, invalidated_at)`, `ledger(mem) -> dict[str, Row]`, `fates(before, mem) -> dict[str, str]`, `seed(mem) -> dict[str, str]` (keys `berlin`, `tea`, `acme`), `turns(mem, receipt) -> list[str | None]`, `rendered(results) -> str`, and `tool_text(server, name, arguments) -> tuple[str, bool]`.
- Produces, in `conftest.py`: the fixture `scripted`, a factory with `ScriptedModel`'s keyword arguments.

- [ ] **Step 1: Write the failing tests** in `test_adv_scripted_model.py`:
  - The model is an instance of `LLM`, `Chat`, `ToolChat` and `ReplacementJudge`, and `is_noop` is false, so memvara bills every call it answers.
  - Each method answers with its own script, in order, and records one `Call` per answer with the method's name and its arguments (for `extract`, the turns' text).
  - A call after a script has run out raises `Unscripted`, is recorded in `unscripted` and not in `calls`, and makes `check_scripts` raise an `AssertionError` that names the call.
  - A `Text` reply to `extract` is shaped exactly as `_shape.shape_claims(_shape.parse_json_object(text), n)` shapes it: a claim naming turn 9 of one turn is dropped, and `"not json"` gives `[]`. A `Text` reply to `resolve_predicate` that names a predicate it was not offered gives `canonical` `None`.
  - A `Truncated` reply makes `extract`, `resolve_predicate`, `classify_predicate` and `judge_replacement` raise `TruncatedResponse`, and makes `chat` return the cut text.
  - An exception in a script is raised. `RateLimitError().status_code` is 429 and `AuthenticationError().status_code` is 401. `APITimeoutError()` is not an instance of `TimeoutError`, has no `status_code`, and `memvara.llm._tools.is_timeout` recognises it. Every provider error is an `APIError`.
  - A `Late(Text("x"), 11.0)` reply returns `"x"` and moves `clock()` from 0.0 to 11.0. Nothing else moves the clock.
  - A `Forever(Text("x"))` reply answers three calls in a row, and nothing is unscripted.
  - `run_tools` with `[Answer(calls=(("t", {"a": 1}),)), Answer("done")]` and one tool `t` runs `t`'s handler with `{"a": 1}`, records the handler's text in `tool_results`, records the run's `max_steps` and `timeout` in `runs`, and returns `ToolRun(steps=2, requests=2, finished=True, text="done")`.
  - `run_tools` with `[Forever(Answer(calls=(("t", {}),)))]` and `max_steps=12` returns `finished=False` after 12 requests.
  - `run_tools` with `[Truncated(), Answer("done")]` sends two requests and finishes. With `[Truncated(), Truncated()]` it raises `MalformedToolOutput` whose `requests` is 2. Both are `run_loop`'s own retry rule.
  - `ScriptedModel(tools=[{"not": "an answer"}])` raises `TypeError` when the script is made, not when a run sends it.
  - `handles.fates`: seed three facts and one more, then end one (a new `lives_in`), retire one (`forget`), erase one (`erase`) and leave one alone. `fates` names them `ended`, `retired`, `erased` and `unchanged`. These are the names every later task's "nothing retired or erased" check relies on, so the check is shown to see all three before it is trusted.
  - `with_model` and `without_model` on one `SQLiteStore(":memory:")` read the same claims, and only the first has a model.
- [ ] **Step 2: Run them and watch them fail** on the missing modules:
  `PYTHONPATH=$PWD TMPDIR=$TMPDIR python -m pytest -q -p no:cacheprovider tests/adversarial/model_faults/test_adv_scripted_model.py`
- [ ] **Step 3: Write `scripted.py`.**

```python
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
it into a test failure, and the `scripted` fixture calls it when a test ends.

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
    """`reply` to this call and to every later call of the same method."""

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
        return self._shaped(reply, lambda parsed: _shape.shape_claims(parsed,
                                                                      len(episodes)))

    def resolve_predicate(self, surface: str, candidates: Sequence[str], *,
                          usage: Usage | None = None) -> Any:
        offered = _shape.bounded(candidates, _shape.MAX_CANDIDATES)
        reply = self._next("resolve_predicate", surface=surface, candidates=offered)
        return self._shaped(reply, lambda parsed: _shape.shape_resolution(parsed, offered))

    def classify_predicate(self, predicate: str, example: str, *,
                           usage: Usage | None = None) -> Any:
        reply = self._next("classify_predicate", predicate=predicate, example=example)
        return self._shaped(reply, _shape.spec_fields)

    def judge_replacement(self, new_text: str, old_text: str, *,
                          usage: Usage | None = None) -> Any:
        reply = self._next("judge_replacement", new=new_text, old=old_text)
        return self._shaped(reply, _shape.shape_verdict)

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
                raise MalformedToolOutput(f"{self.name} stopped at its token limit")
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
        reply = script[0]
        if isinstance(reply, Forever):
            reply = reply.reply
        else:
            script.pop(0)
        if isinstance(reply, Late):
            self._now += reply.seconds
            reply = reply.reply
        if isinstance(reply, BaseException):
            raise reply
        return reply

    def _shaped(self, reply: object, shape: Callable[[dict[str, Any]], Any]) -> Any:
        """A schema-constrained method's return value for `reply`, made the way a
        shipped backend makes it."""
        if isinstance(reply, Truncated):
            # Anthropic's word for the event. `refuse_if_truncated` raises for any
            # backend's own word, so which one is used here changes nothing.
            _shape.refuse_if_truncated("max_tokens", "max_tokens", model=self.name,
                                       budget=BUDGET)
        if isinstance(reply, Text):
            return shape(_shape.parse_json_object(reply.text))
        return reply
```

- [ ] **Step 4: Write `handles.py` and `conftest.py`.** `handles.py` holds the constants and functions named in Interfaces. `seed(mem)` makes two `remember()` calls, `user lives_in Berlin` and `user likes green tea`, and one `add("I work at Acme.")`, which the fast path reads with no model call, and returns their ids. `ledger(mem)` reads `mem.store.iter_claims(tenant, states=("live", "ended", "retired"))` into `Row`s. `fates(before, mem)` names each claim of `before` as `unchanged`, `ended` (only its `valid_to` was set), `retired` (its `invalidated_at` changed), `erased` (its row is gone and `store.erasure_record` names it), `missing` (its row is gone with no record) or `changed`. `rendered(results)` writes one line per result with its type, the id of its claim or turn, its `score!r` and its `explain!r`. `tool_text` sends one `tools/call` to an in-process `MemvaraMCPServer` and returns the text and the error flag. `conftest.py`:

```python
"""The fixture every model-faults test uses to make a scripted model."""

from __future__ import annotations

from typing import Callable, Iterator, Sequence

import pytest

from .scripted import ScriptedModel, check_scripts


@pytest.fixture
def scripted() -> Iterator[Callable[..., ScriptedModel]]:
    """Make scripted models, and fail the test if any of them was called after its script
    ran out. Memvara swallows most exceptions a model raises, so the extra call would
    otherwise pass unnoticed."""
    made: list[ScriptedModel] = []

    def make(**scripts: Sequence[object]) -> ScriptedModel:
        model = ScriptedModel(**scripts)
        made.append(model)
        return model

    yield make
    check_scripts(made)
```

- [ ] **Step 5: Run the tests and watch them pass.** Run the doctests of the two helper modules too: `python -m pytest -q -p no:cacheprovider --doctest-modules tests/adversarial/model_faults/scripted.py tests/adversarial/model_faults/handles.py`.
- [ ] **Step 6: Write the first paragraphs of the testing.md section:** what the scripted model is, what it replaces and what it leaves to memvara, the provider error stand-ins, the model's clock, and the `scripted` fixture.
- [ ] **Step 7: Commit** the five new files and `docs/claude/testing.md`, by name.

### Task 2: Malformed output, invented predicates, and many claims on the write path

**Files:**
- Test: `tests/adversarial/model_faults/test_adv_write_output.py`
- Modify: `docs/claude/testing.md` (the section's paragraph on the write path)

**Interfaces:**
- Consumes: `ScriptedModel`, `Text`, `Truncated`, the `scripted` fixture, `with_model`, `seed`, `ledger`, `fates`, `turns`, `FAST_TURN`, `MODEL_TURN`.

A model claim in this file is a dict with all ten fields of `CLAIM_SCHEMA`. The one the malformed table varies is `team based_in Porto`, citing turn 0 with confidence 0.9. `based_in` is an alias of the builtin `lives_in`, so the claim costs no acquisition call, and its subject is `team`, so its slot is not the user's.

- [ ] **Step 1: Write the tests.** Every test seeds the store, records `before = ledger(mem)`, and then calls `mem.add([FAST_TURN, MODEL_TURN])`, so the model is shown one turn.
  - **The malformed table**, one test per row. Each row gives the reply to `extract`, the objects of the model claims that must be stored, and the expected `deferred` and `unextracted`. For every row: `turns(mem, receipt)` equals the two turns as given; the fast path's `name = Ada` is live; `fates(before, mem)` is `unchanged` for every seeded claim; and `receipt.llm_calls == model.count() == 1`.
    - Text a shipped backend would receive: prose that is not JSON; an empty string; JSON cut off mid-claim with no stop reason; a `Truncated` reply, which must set `deferred`; a top-level list; an object with no `claims` key; `claims` that is an object; `claims` whose items are numbers, strings, `null` and lists; a claim citing turn 7, turn `"0"`, turn `true` or turn `-1`; a claim with no subject, an empty predicate, or a blank object. None of these stores a model claim, and `unextracted` is 1.
    - Text whose fields shaping repairs: `confidence` `1e400`, `polarity` `"yes"`, `memory_type` `"bogus"`, `when` `42` and `amount` `NaN`. The claim is stored with confidence 0.5, polarity 1, type `semantic`, no amount, and `valid_from` equal to the turn's timestamp.
    - Values a backend that does no validation would return: `[]`; claims citing turn `"0"`, `True`, `-1` or 7; an empty or missing predicate; an empty or blank object. None is stored. Five more rows each garble one field, and each claim is stored with that field repaired: `confidence` `"high"` is stored as 0.7, `polarity` `"yes"` as 1, `memory_type` `"bogus"` as `semantic`, `amount` `float("nan")` as no amount, and `when` `42` leaves `valid_from` at the turn's timestamp.
  - **Invented predicates are paid for once.** On a file store, one reply holds three claims under three invented predicates, each grounded in the turn. Three `resolve_predicate` replies say "new, holds many". `llm_calls` is 4. A second `add()` whose reply uses the same three predicates costs 1. A new handle on the same file, with a new scripted model, costs 1 for a third reply that uses them. INTERNALS: "the answer is learned, persisted through `store.put_spec()` and never asked again, including after a restart and including by another process".
  - **A model cannot end the user's value by renaming a slot.** The reply holds `user home_base_city Lisbon` at confidence 0.95, and the acquisition reply names `lives_in` as its canonical predicate. The merge is recorded (`registry.normalize("home_base_city") == "lives_in"`), the new claim is stored at confidence 0.4 beside the user's asserted Berlin, `receipt.disputed` names Berlin, and `fates` says Berlin is `unchanged`. `write/pollution.py` R4 is the rule: a novel predicate is stored at `min(confidence, 0.4)`, so the reconciler's half rule stores it beside the incumbent.
  - **A closed vocabulary pays nothing for invented predicates.** With `write_closed_vocabulary=True`, a reply of three invented predicates and one known one stores only the known one, `receipt.unregistered` is 3, and `llm_calls` is 1.
  - **Past the learned cap an invented predicate costs no call.** A reply of 205 claims under 205 invented predicates costs 201 calls (one extraction and 200 acquisitions, the default cap). A second reply of 5 new invented predicates costs 1. INTERNALS: past the cap "a novel form folds onto its nearest existing predicate ... it costs nothing".
  - **Many claims, one call.** A reply of 500 distinct claims under `likes` stores all 500, every one citing the one turn it came from, with `llm_calls` 1, and the seeded claims `unchanged`.
  - **Review Focus 1.** A reply that restates one claim 500 times stores one claim. `receipt.added` has one entry, `receipt.reinforced` has 499, the slot holds one live value, and its `observation_count` is 500.
- [ ] **Step 2: Show the tests catch the faults they exist for.** Planted faults, each in a scratch copy: in `WritePipeline._claim_from_dict`, accept a `source_index` given as a digit string (the "turn `"0"`" rows must fail); in `pollution.guard`, skip R4's discount (the renaming test must fail, with Berlin `ended`); in `WritePipeline._persist`, write nothing (the restart half of the first invented-predicate test must fail).
- [ ] **Step 3: Run them against the real tree.** A failure is a finding: handle it as the Global Constraints say.
- [ ] **Step 4: Add the section's paragraph on malformed output and invented predicates.**
- [ ] **Step 5: Commit** the test file and `docs/claude/testing.md`, by name.

### Task 3: Timeouts and 429s on the write path

**Files:**
- Test: `tests/adversarial/model_faults/test_adv_write_errors.py`
- Modify: `docs/claude/testing.md`

**Interfaces:**
- Consumes: the provider errors, `Truncated`, `Forever`, `with_model`, `seed`, `ledger`, `fates`, `turns`, `FAST_TURN`, `MODEL_TURN`.

- [ ] **Step 1: Write the tests.**
  - **A failed extraction keeps the turns and the fast path's facts.** One test per error raised by `extract`: `APITimeoutError()`, `RateLimitError()`, `AuthenticationError()`, `APIStatusError("overloaded", status_code=529)`, `APIConnectionError("connection reset")`, Python's `TimeoutError()`, and a `Truncated` reply. For each: both turns are stored as given, `name = Ada` is live, `receipt.deferred` is true, `receipt.unextracted` is 1, `receipt.llm_calls == model.count() == 1`, and every seeded claim is `unchanged`.
  - **A deferred turn is read by `reextract()` once the model answers.** After a 429 on `add()`, `reextract()` with a model that answers `team based_in Porto` costs 1 call, and the new claim's `sources` is the id of the turn the first receipt stored. A second `reextract()` costs nothing, because the turn now has a claim. INTERNALS: `reextract()` is for "a batch a provider failure left `deferred`".
  - **A failed acquisition is paid once and never retried.** `resolve_predicate` raises `RateLimitError()`. The claim is stored anyway, under its invented predicate, `llm_calls` is 2, and the predicate holds many values. A second `add()` using the same predicate costs 1 call and asks nothing more. `WritePipeline._acquire`: "A rate limit or a network blip must not cost the caller the whole batch of facts", and the form is "Marked resolved above, so we do not retry in a hot loop."
  - **A failed judge leaves the write whole.** With `advise_replacements=True`, two live claims stand in other slots near a new `remember()`, and `judge_replacement` raises `RateLimitError()`. The new claim is stored, `receipt.may_replace` is empty, one `RuntimeWarning` says the advice failed, `receipt.llm_calls` is 1, and every other claim is `unchanged`. INTERNALS: "A judge that raises warns once per instance and leaves the list empty".
  - **A failed piece of a long turn defers that turn only.** With `write_extraction_chunks=True`, one `add()` holds a turn of about 13,000 characters that splits into three pieces, and a short turn. The call for the short turn answers a claim, the first piece's call answers a claim, and the second piece's call raises `RateLimitError()`. The third piece is not sent, so `llm_calls` is 3. The short turn's claim is stored, no claim cites the long turn, `deferred` is true, and both turns are stored. INTERNALS: "when a call carrying a turn fails, that turn's later pieces are not sent, what its earlier pieces returned is dropped, and the turn is deferred".
- [ ] **Step 2: Show the tests catch their faults:** `_tier2` without its `try` around the extraction call (the first test must fail, with `add()` raising); `_extraction_failed` not setting `deferred`; `_acquire` returning 0 for a call that raised (the billing assertion must fail).
- [ ] **Step 3: Run them against the real tree.** A failure is a finding.
- [ ] **Step 4: Add the section's paragraph on provider errors.**
- [ ] **Step 5: Commit** by name.

### Task 4: Proposals to retire or erase a stored claim

**Files:**
- Test: `tests/adversarial/model_faults/test_adv_retire_erase.py`
- Modify: `docs/claude/testing.md`

**Interfaces:**
- Consumes: `ScriptedModel`, `Answer`, `Forever`, `Text`, `with_model`, `seed`, `ledger`, `fates`.

- [ ] **Step 1: Write the tests.** Each checks `fates(before, mem)` for every seeded claim, and no seeded claim may ever be `retired`, `erased` or `missing`.
  - **A retraction ends and never retires.** The turn is "Berlin is behind me now, the flat there is gone." and the reply is `user lives_in Berlin` with polarity -1. Berlin is `ended`, it is still in `history()`, `receipt.ended` names it and `receipt.retired` is empty. The same holds when the claim dict also carries each of these: `"close": "retired"`; `"state": "retired"` and an `invalidated_at`; `"erase": true` and `"purge": true`. INTERNALS: "The matches are **ended**, not retired".
  - **A new value ends the old one and never retires it.** The reply `user lives_in Lisbon` at 0.9, from a turn that names Lisbon: Berlin is `ended` and Lisbon is live.
  - **A model cannot clear a whole slot.** A retraction with an empty object would mean "clear the whole slot" to the reconciler. From a model it is dropped at the trust boundary, and green tea is `unchanged`.
  - **An agentic end ends, with the model's reason.** With `write_agentic_extraction=True`, the run reads Berlin with `get_claim`, proposes `propose_end` with the reason "moved away", and stops. Berlin is `ended`, `why()` or `history()` shows "moved away", nothing is refused, and `llm_calls` is 3.
  - **An end for a claim the model never read is refused.** `propose_end` on Berlin with no read first: `receipt.proposals_refused` holds one `not_read`, and Berlin is `unchanged`.
  - **An end for a broader scope's claim is refused.** The write is in session `s1`, and Berlin is user-wide. After `get_claim`, `propose_end` is refused as `broader_scope`, and Berlin is `unchanged`.
  - **A replacement the reconciler does not accept changes nothing.** After `get_claim`, `propose_supersede` names Berlin with a new value in another slot (`likes`, "Lisbon trams"). The new claim is stored, the proposal is refused as `not_applied`, and Berlin is `unchanged`.
  - **A replacement the reconciler accepts ends the named claim.** `propose_supersede` names Berlin with `user lives_in Lisbon` at 0.9: Berlin is `ended`, not retired.
  - **Review Focus 4.** Another user, `u2`, has a claim. The run calls `get_claim` with its id and with a made-up id, and gets the same text for both, "No stored memory with that id is visible to this write." Its `propose_end` for the other user's claim is refused as `not_read`, and that claim is `unchanged`.
  - **No tool retires or erases.** The six tools offered (`runs[0]["tools"]`) are `search_memories`, `get_claim`, `propose_claim`, `propose_end`, `propose_supersede` and `propose_link`. A run that calls `erase_claim` and then `retire_claim` sends two unusable answers, falls back as `malformed`, and changes nothing. `llm_calls` is 2 requests plus 1 fallback extraction.
  - **Replacement advice closes nothing.** With `advise_replacements=True` and a judge that answers "replaces" to every question, `receipt.may_replace` lists the advised claims and every one of them is `unchanged`. INTERNALS: replacement advice "closes nothing".
  - **A forget preview asks no model.** The model's chat script is `Forever` of a rewrite that would add Berlin, Acme and tea as alternative queries. `forget_matching("green tea", close="retired")` previews only the tea claim, and confirming it retires only that claim: the caller asked for it. Berlin and Acme are `unchanged`, and the model answered no call. `core.py` passes `**PLAIN_READ` to the preview's search.
- [ ] **Step 2: Show the tests catch their faults:** `Reconciler._retract` closing with `"retired"` whatever `close` says; `WritePipeline._apply_proposals` ending with `close="retired"`; `Memvara._advise_replacements` retiring each advised claim; `forget_matching`'s preview search without `**PLAIN_READ`.
- [ ] **Step 3: Run them against the real tree.** A failure is a finding.
- [ ] **Step 4: Add the section's paragraph on proposals to retire or erase.**
- [ ] **Step 5: Commit** by name.

### Task 5: Runaway tool loops, and the other ways an agentic run fails

**Files:**
- Test: `tests/adversarial/model_faults/test_adv_tool_loops.py`
- Modify: `docs/claude/testing.md`

**Interfaces:**
- Consumes: `Answer`, `Forever`, `Late`, `Truncated`, the provider errors, `with_model(..., write_agentic_extraction=True)`, `seed`, `ledger`, `fates`, `MODEL_TURN`.

The fallback extraction in every test answers `team based_in Porto`.

- [ ] **Step 1: Write the tests.**
  - **A model that never stops is stopped at twelve answers, and each is billed.** `Forever(Answer(calls=(("search_memories", {"query": "office", "k": 3}),)))`: the receipt says `agentic_fallback == "step_limit"`, the model answered 12 `run_tools` requests and 1 `extract`, `llm_calls` is 13, the fallback's claim is stored, the run was given `max_steps` 12 and `timeout` 25.0, and every seeded claim is `unchanged`. INTERNALS: "At most `AGENTIC_MAX_STEPS` (12) answers", and "Every request the run sent is billed in `llm_calls`, fallback or not."
  - **A run stopped at the step limit loses its proposals.** Every answer reads Berlin and proposes to end it. After twelve, the run is abandoned, and Berlin is `unchanged`. INTERNALS: `step_limit`, "and its proposals are discarded".
  - **An unusable answer is retried once and billed twice.** `[Truncated(), Answer("done")]`: no fallback, `llm_calls` 2. Twice in a row, for each of a `Truncated` answer, a call to a tool that was not offered, and arguments that are not an object: fallback `malformed`, `llm_calls` 2 plus 1.
  - **A rate-limited request is retried once.** `[RateLimitError(), Answer("done")]`: no fallback, `llm_calls` 2. `[RateLimitError(), RateLimitError()]`: fallback `error`, `llm_calls` 2 plus 1.
  - **A provider timeout is not retried.** `[APITimeoutError()]`: fallback `timeout`, one `run_tools` request, `llm_calls` 1 plus 1.
  - **A slow provider runs out a write's 25 seconds.** `Forever(Late(Answer(calls=(search,)), 10.0))`: the fourth request is never sent, so the fallback is `timeout` after 3 requests, and `llm_calls` is 3 plus 1.
  - **The same slow provider gets 180 seconds in `reextract()`.** After an `add()` whose run times out at once and whose fallback extracts nothing, `reextract()` runs the slow model to the step limit rather than out of time: fallback `step_limit`, `runs[-1]["timeout"]` 180.0, `llm_calls` 12 plus 1. INTERNALS: `AGENTIC_SYNC_TIMEOUT` (25 s) in `add()` and `AGENTIC_TIMEOUT` (180 s) in `reextract()`.
  - **Review Focus 5.** One answer calls `search_memories` five times, with `k` 0, -5, 10,000, `"ten"` and `True`, and the next answer stops. The run finishes with no fallback, every tool result starts "Stored memories.", and the results for 0 and -5 list exactly one memory.
- [ ] **Step 2: Show the tests catch their faults:** `AGENTIC_MAX_STEPS` raised to 100 (the first test must fail on the count); `WritePipeline._agentic` not billing the requests of a failed run; `run_loop` retrying a timeout.
- [ ] **Step 3: Run them against the real tree.** A failure is a finding.
- [ ] **Step 4: Add the section's paragraph on tool loops.**
- [ ] **Step 5: Commit** by name.

### Task 6: Read stages that fail

**Files:**
- Test: `tests/adversarial/model_faults/test_adv_read_stages.py`
- Modify: `docs/claude/testing.md`

**Interfaces:**
- Consumes: `ScriptedModel`, the provider errors, `Text`, `Truncated`, `Late`, `with_model(..., store=shared, ranked=True)`, `without_model(shared)`, `rendered`, `tool_text`, `Memvara.RECALL_UNSYNTHESIZED`, `Memvara.RECALL_UNRANKED`.

The store is one `SQLiteStore(":memory:")`. Its content is written through the handle with no model: three facts with `remember()` and eight turns with `add(..., role="system")`, so no model is asked and no predicate is learned. The handle with the model is made after that, so both handles hold the same registry. The query is "Lisbon trip". A `search()` compared byte for byte passes `known_at`, one fixed instant after every write, because recency is measured at `known_at` and would otherwise differ between two reads in its last digits.

- [ ] **Step 1: Write the tests.**
  - **A failed rewrite serves the plain read.** One test per reply to the rewrite's chat call, each with its documented outcome:
    - `APITimeoutError()` raised at once: `fallback`, `error`, because the call ended before its deadline.
    - `Late(APITimeoutError(), 10.5)`: `fallback`, `timeout`.
    - Python's `TimeoutError()`: `fallback`, `timeout`.
    - `RateLimitError()`: `fallback`, `provider`, status 429. `APIStatusError("unavailable", status_code=503)`: `fallback`, `provider`, status 503.
    - `AuthenticationError()`: `key_rejected`, status 401.
    - `APIConnectionError("connection reset")`: `fallback`, `error`.
    - A good rewrite that arrives after 11 seconds: `fallback`, `timeout`.
    - Prose that is not JSON, an empty string, cut-off JSON, a `Truncated` reply, a top-level list, `queries` that is text, a date range that ends before it starts, and a date range written in words: `fallback`, `malformed`.

    For each: `model.recall(query, include_episodes=True)` equals the no-model handle's, byte for byte; `rendered(model.search(query, include_episodes=True, known_at=T))` equals the no-model handle's; `.rewrite` has the outcome, reason and status above; and the model answered exactly one chat call per read. For three of the replies (the timeout at once, the 429 and the prose), `memory_recall` and `memory_search` on an in-process `MemvaraMCPServer` over each handle return the same text.
  - **A failed synthesis names itself and keeps every note.** One test per reply to the synthesis call: the provider errors above, prose, an empty string, cut-off JSON, `{"synthesis": ""}`, `{"synthesis": 42}`, `{"summary": "..."}` and a list. `model.recall(query, include_episodes=True, synthesize=True, query_rewrite=False)` starts with `RECALL_UNSYNTHESIZED` naming the outcome (for example `fallback: provider`), the no-model handle's block starts with the same line naming `unconfigured`, and every line after the first is the same in both, and equal to the block that `synthesize=False` renders. The model answered one call.
  - **A recall that finds nothing asks for no summary.** On an empty store, `recall(..., synthesize=True, query_rewrite=False)` returns an empty block and the model answered no call. `Memvara.recall`: "A recall that found no notes makes no call and stays empty."
  - **A failed ranking selects nothing and says so.** One test per reply to the selector's call (the provider errors and three malformed replies). `recall(..., ranked=True, include_episodes=True, query_rewrite=False, with_ids=True)` reports `selection.outcome` as documented, its last line is `RECALL_UNRANKED` naming that outcome, no turn in `search(..., ranked=True)` has `explain.selected` set, and the model answered one call.
  - **A failed ranking serves the plain read.** The same ranked recall and search equal the no-model handle's ranked recall and search, except for the `RECALL_UNRANKED` line's outcome. INTERNALS invariant 1 says a failed stage serves the plain read.
  - **Review Focus 2.** A rewrite that returns `["Lisbon trip", "LISBON TRIP", "a", "b", "c", "d", "e"]` is applied with the queries `("a", "b", "c")`, costs one call, and runs four retrievals, counted by wrapping the retriever's `_search_once` on the instance.
  - **Review Focus 3.** A selector reply keeps `i` 2 with span "tram", repeats `i` 2, names `i` 99, `i` `"1"` and `i` `true`, and gives `i` 1 an empty span and `i` 3 no span. The read is `applied`, exactly one turn is selected, and it is the turn the prompt numbered 2.
- [ ] **Step 2: Show the tests catch their faults:** `QueryRewriter.rewrite` returning `applied` with the query "Berlin" when a reply cannot be parsed; `recall()` dropping the notes when a synthesis fails; a failed ranking marking every turn selected.
- [ ] **Step 3: Run them against the real tree.** A failure is a finding.
- [ ] **Step 4: Add the section's paragraph on read stages.**
- [ ] **Step 5: Commit** by name.

### Task 7: Model calls per operation

**Files:**
- Test: `tests/adversarial/model_faults/test_adv_call_counts.py`
- Modify: `docs/claude/testing.md`

**Interfaces:**
- Consumes: `ScriptedModel`, `Forever`, `Text`, `Answer`, `with_model`, `counts()`.

One table, one test per row. A row sets the store up (not counted), then runs one operation, and compares the model's calls made by that operation, by method, with the row's expected counts. For a write, `receipt.llm_calls` must equal the calls counted. The model answers every call properly, each script a `Forever` of one reply:
- extraction: one claim, `team based_in Porto`, citing turn 0, from turns that name Porto;
- acquisition: `Text('{"canonical": null, "cardinality": "many", "volatility": "slow", "memory_type": "semantic"}')`;
- chat: `Text('{"queries": [], "date_range": null, "synthesis": "The notes mention Lisbon.", "kept": []}')`, one object that the rewrite, the synthesis and the selector can all read;
- the judge: `Text('{"same_thing": false, "same_property": false, "newer_value": false, "replaces": false}')`.

- [ ] **Step 1: Write the table.** Each row quotes the INTERNALS sentence it checks.

  | Operation | Calls |
  |---|---|
  | `add()` of a turn the fast path reads | none |
  | `add()` of "thanks!" | none |
  | `add()` of a turn already stored | none |
  | `add()` of three turns that reach the model | 1 `extract` |
  | `add()` whose claim uses a new predicate | 1 `extract`, 1 `resolve_predicate` |
  | the same predicate in the next `add()` | 1 `extract` |
  | the same predicate after reopening the file | 1 `extract` |
  | a new predicate under a closed vocabulary | 1 `extract` |
  | `remember()` | none |
  | `remember()` with replacement advice, four live claims in other slots | 3 `judge_replacement` |
  | `reextract()` of a stored turn | 1 `extract` |
  | `reextract()` of the same turn again | none |
  | a turn of about 13,000 characters with extraction chunks | 3 `extract` |
  | an agentic run that proposes one claim and stops | 2 `run_tools` |
  | `search()` | 1 `chat` |
  | `search(query_rewrite=False)` and `search(k=0)` | none |
  | `recall(synthesize=True)` | 2 `chat` |
  | `recall(synthesize=True)` on an empty store | 1 `chat` |
  | `recall(ranked=True, include_episodes=True)` | 2 `chat` |
  | `forget_matching()` preview | none |
- [ ] **Step 2: Show the table catches its faults:** `_tier2` calling `extract` once per turn (the three-turn row must fail); `recall()` asking for a summary with no notes (the empty-store row must fail).
- [ ] **Step 3: Run it against the real tree.** A failure is a finding.
- [ ] **Step 4: Add the section's paragraph on call counts.**
- [ ] **Step 5: Commit** by name.

### Task 8: 10,000 claims in one reply (nightly)

**Files:**
- Create: `tests/adversarial/model_faults/nightly/__init__.py`
- Test: `tests/adversarial/model_faults/nightly/test_adv_ten_thousand_claims.py`
- Modify: `docs/claude/testing.md`

- [ ] **Step 1: Write the tests.**
  - A reply of 10,000 distinct claims under `likes` stores all of them from one call, each citing the turn, the seeded claims are `unchanged`, and a search finds one of them by its object.
  - The same 10,000 claims sent as JSON text, shaped as a shipped backend shapes them, are stored the same way.
  - **Review Focus 1.** A reply that restates one claim 10,000 times stores one claim, with an `observation_count` of 10,000.
  - A reply of 10,000 claims under 10,000 invented predicates stores every claim, costs 201 calls, and finishes within 60 seconds on a laptop.
- [ ] **Step 2: Show the tests catch their fault:** `_tier2` keeping only the first 1,000 items.
- [ ] **Step 3: Run them with `--tier nightly` and time them.** A failure is a finding.
- [ ] **Step 4: Add the section's paragraph on the nightly tier.**
- [ ] **Step 5: Commit** by name.

### Task 9: Flakes, the gate, and the report

- [ ] **Step 1: Run each new test file 20 times in a row**, and `nightly/test_adv_ten_thousand_claims.py` with `--tier nightly` as well. Quote each result line.
- [ ] **Step 2: Time the fast tier of this plan** with `--durations=10` and check it is about 5 seconds.
- [ ] **Step 3: Run the full gate as two commands with a private coverage file:** `COVERAGE_FILE=$PWD/local/cov/.coverage.a7 python -m coverage run -m pytest -q -p no:cacheprovider`, then `python -m coverage report`. Coverage of `memvara/` must stay at 100%.
- [ ] **Step 4: Run `python -m mypy -p memvara`, and `python -m mypy tests/harness` with and without `--ignore-missing-imports`.** Quote the result lines.
- [ ] **Step 5: Write the report:** the branch, each commit, the files, the counts, the flake runs, the gate, coverage, mypy, every finding with its classification and an offline reproduction, and every departure from the design with its reason.
