"""A fake OpenAI-compatible chat-completions endpoint that returns scripted replies.

memvara reaches an OpenAI-compatible model through `memvara.llm.openai.OpenAILLM`. It
calls `client.chat.completions.create(...)` with `model`, `messages`, and, depending on the
call, `response_format`, `tools`, `temperature`, `max_completion_tokens`, `timeout` and
`extra_body`. From the reply it reads `choices[0].message` (its `content`, `refusal` and
`tool_calls`), `choices[0].finish_reason`, and `usage.prompt_tokens` and
`usage.completion_tokens`. With no client given, `OpenAILLM` builds an `openai.OpenAI`
pointed at `OPENAI_BASE_URL`, which is how a server started with `MEMVARA_LLM=openai`
reaches a self-hosted model.

`FakeOpenAI` is such an endpoint: `POST /v1/chat/completions` on 127.0.0.1. It answers
each request with the next reply a test scripted, in order, and records every request.
A scripted reply can be a completion, tool calls, a raw body the client cannot parse, a
429 or a hang. A request that finds no scripted reply left gets a 500 that says so, so a
test that made one call more than it expected fails loudly.

CI does not install the `openai` package, so `client()` returns a stand-in for its
transport. `OpenAILLM(client=fake.client())` sends each call over HTTP to this fake, as
the SDK would send it, and gets back the decoded JSON, which `OpenAILLM` reads the way it
reads the SDK's objects. Unlike the SDK, the stand-in does not retry: the SDK retries a
429 or a timeout twice by default, so a test that wants to see a retry scripts one.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import httpx

from ._http import HttpFake, Reply, Request, Step, json_reply

#: The model a completion names when the request named none.
MODEL = "fake-model"

#: The route every completion is requested on, relative to the base URL.
COMPLETIONS = "POST /v1/chat/completions"


@dataclass(frozen=True)
class _Scripted:
    #: None for a hang; otherwise builds the reply from the request it answers.
    build: Callable[[Request, int], Reply] | None


class FakeOpenAIError(RuntimeError):
    """The fake answered with an error status. `client()` raises it where the SDK would
    raise one of its `APIStatusError` subclasses."""

    def __init__(self, status: int, body: str, retry_after: str | None) -> None:
        super().__init__(f"the model endpoint answered {status}: {body[:300]}")
        self.status = status
        self.body = body
        self.retry_after = retry_after


class FakeOpenAI(HttpFake):
    """An OpenAI-compatible chat-completions endpoint with scripted replies.

    `client()` and `base_url` reach it over a socket, and start serving it if it is not
    serving yet. A route fault from `fail` or `hang` answers, or holds, the request in
    place of the script, and does not use up a scripted reply. A `delay` waits, and the
    request is then answered with the next scripted reply. Replies are handed out in the
    order requests arrive, so a request whose client gives up during a delay still uses
    up its reply.
    """

    ROUTES = (COMPLETIONS,)

    def __init__(self) -> None:
        super().__init__()
        self._script: deque[_Scripted] = deque()
        self._answered = 0
        self._script_lock = threading.Lock()

    @property
    def base_url(self) -> str:
        """What to set `OPENAI_BASE_URL` to. Starts serving if it has not."""
        return self.serve() + "/v1"

    @property
    def pending(self) -> int:
        """How many scripted replies have not been used yet."""
        with self._script_lock:
            return len(self._script)

    # -- the script -----------------------------------------------------------------

    def add_reply(self, content: str, *, finish_reason: str = "stop",
                  prompt_tokens: int = 10, completion_tokens: int = 5) -> None:
        """A completion whose message says `content`."""
        message = {"role": "assistant", "content": content, "refusal": None}
        self._push(lambda request, number: _completion(
            request, number, message, finish_reason, prompt_tokens, completion_tokens))

    def add_json(self, value: Any, **options: Any) -> None:
        """A completion whose content is `value` written as JSON, which is the shape a
        structured-output call is answered in."""
        self.add_reply(json.dumps(value), **options)

    def add_tool_calls(self, *calls: tuple[str, Mapping[str, Any]],
                       prompt_tokens: int = 10, completion_tokens: int = 5) -> None:
        """A completion that asks for tool calls: each `(name, arguments)` pair."""
        message = {"role": "assistant", "content": None, "refusal": None,
                   "tool_calls": [{"id": f"call_{index}", "type": "function",
                                   "function": {"name": name,
                                                "arguments": json.dumps(dict(arguments))}}
                                  for index, (name, arguments) in enumerate(calls)]}
        self._push(lambda request, number: _completion(
            request, number, message, "tool_calls", prompt_tokens, completion_tokens))

    def add_raw(self, body: Any, *, status: int = 200,
                headers: Mapping[str, str] | None = None) -> None:
        """Exactly this body: JSON for a dict or a list, raw for `str` or `bytes`. For a
        reply the client cannot parse, or one shaped wrongly."""
        if isinstance(body, (str, bytes)):
            raw = body.encode("utf-8") if isinstance(body, str) else body
            reply = Reply(status, raw, dict(headers or {}))
        else:
            reply = json_reply(status, body, headers)
        self._push(lambda request, number: reply)

    def add_rate_limit(self, *, retry_after: float | None = 1.0,
                       message: str = "Rate limit reached for requests") -> None:
        """A 429 in OpenAI's error shape, with `Retry-After` unless it is None."""
        headers = {} if retry_after is None else {"Retry-After": f"{retry_after:g}"}
        body = {"error": {"message": message, "type": "requests", "param": None,
                          "code": "rate_limit_exceeded"}}
        self._push(lambda request, number: json_reply(429, body, headers))

    def add_hang(self) -> None:
        """No answer at all: the request is held until the client gives up or the fake
        closes."""
        with self._script_lock:
            self._script.append(_Scripted(None))

    def _push(self, build: Callable[[Request, int], Reply]) -> None:
        with self._script_lock:
            self._script.append(_Scripted(build))

    # -- serving --------------------------------------------------------------------

    def route(self, request: Request) -> str | None:
        return COMPLETIONS if (request.method, request.path) == (
            "POST", "/v1/chat/completions") else None

    def error_body(self, status: int, route: str) -> Any:
        kind = "server_error" if status >= 500 else "invalid_request_error"
        return {"error": {"message": f"FakeOpenAI injected a {status} on {route}",
                          "type": kind, "param": None, "code": None}}

    def plan(self, request: Request) -> Step:
        fault = super().plan(request)
        # A fault that answers, or never answers, takes the request's turn in the script.
        # A delay only waits, and the request is then answered from the script as usual.
        if fault.hang or fault.reply is not None or request.route is None:
            return fault
        scripted = self._next_scripted(request)
        return Step(hang=scripted.hang, wait=fault.wait, reply=scripted.reply)

    def _next_scripted(self, request: Request) -> Step:
        """The next scripted answer, used up by this request."""
        with self._script_lock:
            self._answered += 1
            number = self._answered
            scripted = self._script.popleft() if self._script else None
        if scripted is None:
            return Step(reply=json_reply(500, {"error": {
                "message": f"FakeOpenAI has no scripted reply left for request {number}",
                "type": "server_error", "param": None, "code": "script_exhausted"}}))
        if scripted.build is None:
            return Step(hang=True)
        return Step(reply=scripted.build(request, number))

    def respond(self, request: Request) -> Reply:
        return json_reply(404, {"error": {"message": f"no route {request.method} "
                                                     f"{request.path}",
                                          "type": "invalid_request_error", "param": None,
                                          "code": "not_found"}})

    # -- a client -------------------------------------------------------------------

    def client(self, *, api_key: str = "sk-fake", timeout: float = 10.0) -> Any:
        """An object shaped like an `openai.OpenAI` client, for `OpenAILLM(client=...)`.

        Its `chat.completions.create(**kwargs)` sends the keyword arguments as the JSON
        body, the way the SDK does: `timeout` sets how long to wait, `extra_body` is
        merged into the body, and everything else is sent as given. It returns the
        decoded reply, raises `FakeOpenAIError` for an error status, and lets
        `httpx.ReadTimeout` through when the reply does not come in time.
        """
        return _Client(self.base_url, api_key, timeout)


def _completion(request: Request, number: int, message: Mapping[str, Any],
                finish_reason: str, prompt_tokens: int, completion_tokens: int) -> Reply:
    """A chat completion in OpenAI's shape, naming the model the request asked for."""
    try:
        sent = request.json()
    except ValueError:
        sent = None
    model = sent.get("model") if isinstance(sent, dict) else None
    return json_reply(200, {
        "id": f"chatcmpl-fake-{number}", "object": "chat.completion",
        "created": int(time.time()), "model": model or MODEL,
        "choices": [{"index": 0, "message": dict(message), "finish_reason": finish_reason,
                     "logprobs": None}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens}})


class _Completions:
    def __init__(self, base_url: str, api_key: str, timeout: float) -> None:
        self._url = base_url + "/chat/completions"
        self._key = api_key
        self._timeout = timeout

    def create(self, **kwargs: Any) -> Any:
        timeout = kwargs.pop("timeout", None)
        body = dict(kwargs)
        body.update(body.pop("extra_body", None) or {})
        response = httpx.post(self._url, json=body,
                              headers={"Authorization": f"Bearer {self._key}"},
                              timeout=self._timeout if timeout is None else timeout)
        if response.status_code >= 400:
            raise FakeOpenAIError(response.status_code, response.text,
                                  response.headers.get("retry-after"))
        return response.json()


class _Chat:
    def __init__(self, completions: _Completions) -> None:
        self.completions = completions


class _Client:
    """`client.chat.completions.create`, and nothing else of the SDK."""

    def __init__(self, base_url: str, api_key: str, timeout: float) -> None:
        self.chat = _Chat(_Completions(base_url, api_key, timeout))


__all__ = ["COMPLETIONS", "FakeOpenAI", "FakeOpenAIError", "MODEL"]
