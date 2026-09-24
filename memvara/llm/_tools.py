"""The tool-calling loop shared by every backend that implements `ToolChat`.

What differs between providers is the shape of a request and of an answer: Anthropic sends
`tool_use` blocks and takes `tool_result` blocks back, OpenAI sends `tool_calls` and takes
`tool` messages back. What does not differ is the loop around them: how many steps, how
long, what to retry, and what counts as an answer that cannot be used. That part is here,
once, so the two backends cannot come to disagree about it, for the reason `_shape` holds
the rules for a valid claim.

A backend hands `run_loop` two functions. `send(remaining)` makes one request with at most
`remaining` seconds and returns a `Step`, raising `MalformedToolOutput` for an answer it
cannot read. `append(step, results)` adds the model's answer and the tool results to the
backend's own copy of the conversation. The loop validates each tool call, runs its
handler, and decides when to stop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .base import MalformedToolOutput, ToolRun, ToolRunError, ToolRunTimeout, ToolSpec


@dataclass(slots=True)
class Call:
    """One tool call from a model's answer, as the provider sent it."""

    id: str
    name: str
    #: The parsed arguments, or whatever the provider sent when it could not be parsed.
    #: `run_loop` refuses anything that is not a dict.
    arguments: Any


@dataclass(slots=True)
class Step:
    """One answer from the model: its text, its tool calls, and the raw answer.

    `payload` is the provider's own representation of the answer, which `append` puts back
    into the conversation unchanged. Anthropic needs that: an answer can carry thinking
    blocks, and they must be returned exactly as they came.
    """

    text: str
    calls: list[Call] = field(default_factory=list)
    payload: Any = None


def is_timeout(exc: BaseException) -> bool:
    """True for an exception that means a request ran out of time.

    Matched by name as well as by type, because the provider SDKs are optional imports and
    their timeout errors (`anthropic.APITimeoutError`, `openai.APITimeoutError`) cannot be
    named here without importing them.

        >>> is_timeout(TimeoutError()), is_timeout(ValueError())
        (True, False)
    """
    return isinstance(exc, TimeoutError) or "Timeout" in type(exc).__name__


def run_loop(send: Callable[[float], Step],
             append: Callable[[Step, list[tuple[str, str]]], None],
             tools: Sequence[ToolSpec], *, max_steps: int, timeout: float,
             clock: Callable[[], float] = time.monotonic) -> ToolRun:
    """Run a model through its tool calls. See `ToolChat.run_tools` for the contract.

    One retry per step, for an unusable answer or for a failed request, because both are
    usually transient and a whole run should not be lost to one. A timeout is not retried.
    The deadline is checked before every request, and each request is given only the time
    that is left. Whatever stops the run, the exception it raises says how many requests
    were sent, because each of them may have been billed.
    """
    deadline = clock() + timeout
    by_name = {tool.name: tool for tool in tools}
    steps = requests = 0
    text = ""
    while steps < max_steps:
        attempt = 1
        while True:
            remaining = deadline - clock()
            if remaining <= 0:
                raise ToolRunTimeout("the tool run's time budget ran out",
                                     requests=requests)
            requests += 1
            try:
                answer = send(remaining)
                _check(answer, by_name)
            except MalformedToolOutput as exc:
                if attempt == 2:
                    raise MalformedToolOutput(str(exc), requests=requests) from exc
            except Exception as exc:
                if is_timeout(exc):
                    raise ToolRunTimeout(str(exc) or type(exc).__name__,
                                         requests=requests) from exc
                if attempt == 2:
                    raise ToolRunError(f"the request failed twice: {exc!r}",
                                       requests=requests) from exc
            else:
                break
            attempt += 1
        steps += 1
        text = answer.text
        if not answer.calls:
            return ToolRun(steps=steps, requests=requests, finished=True, text=text)
        results = [(call.id, by_name[call.name].handler(call.arguments))
                   for call in answer.calls]
        append(answer, results)
    return ToolRun(steps=steps, requests=requests, finished=False, text=text)


def _check(answer: Step, by_name: dict[str, ToolSpec]) -> None:
    """Refuse a tool call that names no offered tool or carries no argument object."""
    for call in answer.calls:
        if call.name not in by_name:
            raise MalformedToolOutput(f"the model called {call.name!r}, which was not offered")
        if not isinstance(call.arguments, dict):
            raise MalformedToolOutput(
                f"the arguments to {call.name!r} were not a JSON object")
