"""The one chat call every read-path model stage makes, and how its failures are sorted.

`ModelSelector` (ranked reads), `QueryRewriter` and `Synthesizer` all send one system and
one user message to a `Chat` backend under a deadline, and all three sort a failure the
same way. This module holds that call and that sorting once, so the three stages cannot
disagree about what a 401 or a late reply means.

The deadline covers the whole call. `clock` is read before the call and again when it
returns or raises; a reply that arrives after the deadline counts as a timeout even
though it arrived, because the provider billed for it and the caller waited for it.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable, Iterator

from ..llm.base import Chat, Usage

#: The deadline for one read-path model call, in seconds, for every stage.
DEFAULT_TIMEOUT = 10.0


class ChatFailed(Exception):
    """A chat call that produced no usable reply, already sorted into an outcome.

    `outcome` is `key_rejected` (the provider answered 401 or 403) or `fallback`. For
    `fallback`, `reason` is `timeout` (the call ended after its deadline, however it
    ended), `provider` (the exception carried an HTTP status) or `error` (anything else).
    `status` is the provider's HTTP status when there was one. The exception the backend
    raised, when it raised one, is `__cause__`.
    """

    def __init__(self, outcome: str, reason: str | None = None,
                 status: int | None = None) -> None:
        super().__init__(outcome if reason is None else f"{outcome}: {reason}")
        self.outcome = outcome
        self.reason = reason
        self.status = status


def require_chat(llm: object, name: str) -> None:
    """Refuse a backend that cannot chat, naming the extras that provide one."""
    if not isinstance(llm, Chat):
        raise TypeError(
            f"{name} needs a backend with .chat() — OpenAILLM or AnthropicLLM "
            "(pip install 'memvara[openai]' or 'memvara[anthropic]'), "
            f"not {type(llm).__name__}. NullLLM has no model to consult.")


def call_chat(llm: Chat, system: str, prompt: str, *, timeout: float,
              max_completion_tokens: int, usage: Usage | None = None,
              clock: Callable[[], float] = time.monotonic) -> str:
    """One chat call in JSON mode. Returns the reply's text or raises `ChatFailed`."""
    deadline = clock() + timeout
    try:
        text = llm.chat(system, prompt, json_object=True,
                        max_completion_tokens=max_completion_tokens, timeout=timeout,
                        usage=usage)
    except Exception as exc:                                  # noqa: BLE001 - deliberate
        status = getattr(exc, "status_code", None)
        if status in (401, 403):
            raise ChatFailed("key_rejected", status=status) from exc
        # The backend's own timeout can fire before ours as an SDK-specific exception
        # rather than Python's `TimeoutError`; after the deadline it is a timeout either
        # way, because it was billed either way.
        if isinstance(exc, TimeoutError) or clock() > deadline:
            raise ChatFailed("fallback", "timeout", status) from exc
        raise ChatFailed("fallback", "provider" if status is not None else "error",
                         status) from exc
    if clock() > deadline:
        raise ChatFailed("fallback", "timeout")
    return text


class ChatStage:
    """What every read-path model stage holds: a chat backend, a deadline, a clock.

    `ModelSelector`, `QueryRewriter` and `Synthesizer` are all one of these. `admit()` is
    the hook a deployment uses to bound concurrent model calls: it is a context manager
    held around the call, and a wrapper that holds a cap raises `SelectorBusy` from it
    when the cap is full (or `SelectorRefused("disabled")` for an operator's switch). The
    implementation here never refuses. A ranked read that is refused admission is not
    served (see `Selector`); a rewrite or a synthesis that is refused is recorded as a
    `fallback` with reason `busy` (or as `disabled`) and the read goes on without it.

    `clock` is the clock the deadline is measured on. It is a parameter so a test can
    move time forward without sleeping.
    """

    def __init__(self, llm: Chat, *, timeout: float = DEFAULT_TIMEOUT,
                 clock: Callable[[], float] = time.monotonic) -> None:
        require_chat(llm, type(self).__name__)
        self._llm = llm
        self.timeout = timeout
        self._clock = clock

    @contextmanager
    def admit(self) -> Iterator[None]:
        """Never refuses. See the class docstring for what a wrapper may raise here."""
        yield

    def _call(self, system: str, prompt: str, max_completion_tokens: int,
              usage: Usage | None) -> str:
        return call_chat(self._llm, system, prompt, timeout=self.timeout,
                         max_completion_tokens=max_completion_tokens, usage=usage,
                         clock=self._clock)
