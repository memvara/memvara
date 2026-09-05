"""`ModelSelector`: the one real `Selector` this package ships.

A plain chat call to a configured `Chat` backend, sending exactly the two messages
`local/compress/extract.py` sent — the prompt the 182-question measurement ran against.
The request parameters are not `extract.py`'s: `json_object` rather than a constrained
schema, no `temperature`, and `max_completion_tokens=400` rather than 4,000 — every
gpt-5.4 answer in the sample fit under 400, and a cap cannot change an answer that
already fits under it.

`timeout` is a deadline on the whole call — connect, request and response together, read
off the clock at the start and again on return — not the per-socket-operation limit
`timeout=` ordinarily means to an SDK's own client. An answer that lands after the
deadline is a `timeout` outcome even though it arrived, because the provider already
billed for it and the caller waited through it.

What this raises, and what it means:

* `SelectorRefused("key_rejected", status)` — the provider answered 401 or 403. Not a
  fallback: a revoked key served unranked for a month with nothing saying so is the
  failure that must not hide, so the caller has to treat this differently from a
  transient error.
* `TimeoutError` — the reply arrived, or the call failed, after the deadline.
* `ValueError` — the reply was not JSON, or carried no usable `"kept"` list. Matches
  `local/compress/extract.py`'s own rule: a reply `json.loads(txt)["kept"]` cannot read
  is a parse failure there too, whether the text is not JSON at all or is a JSON object
  missing the key.
* anything else the backend's `chat()` raised — a connection failure, a 429, a 5xx —
  propagates unchanged. The caller reads a `status_code` off it when there is one.

A clean call returns a `Sequence[Selected]`: the candidates the model named, in the
order they were handed in (which is the reranker's order — this class does not reorder
them), each with its span kept when it is a genuine substring of the turn and `None`
otherwise. `admit()` never refuses; the process-wide cap this protocol exists to make
room for belongs to a wrapper held around this class, such as the hosted service's, not
to this class itself.
"""

from __future__ import annotations

import re
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator, Sequence

from ..llm import _shape
from ..llm.base import Chat, Usage
from ..types import utcnow
from .base import Candidate, Selected, SelectorRefused

#: Byte-identical to `local/compress/extract.py`'s `SYSTEM` — the prompt the 182-question
#: measurement ran against. A drift here is a silent change to a measured thing, so
#: `tests/test_select.py` asserts it against a fixture copied from that file rather than
#: trusting this copy to stay in sync by eye.
SYSTEM = (
    "You filter conversation excerpts for a question-answering system. For each numbered "
    "excerpt, copy out VERBATIM the shortest span or spans that could help answer the question: "
    "every number, name, date, place, quantity, duration, price, product or decision that bears "
    "on it, with just enough surrounding words to keep its meaning. Never paraphrase, never add "
    "words, never answer the question, never merge excerpts. If an excerpt has nothing that bears "
    "on the question, omit it. Be inclusive on the borderline: a partial mention that might "
    "combine with other excerpts is worth keeping. Respond with JSON only: "
    "{\"kept\": [{\"i\": <excerpt number>, \"span\": \"<verbatim text>\"}, ...]}."
)

#: Not `extract.py`'s 4,000. Every gpt-5.4 answer in the 182-question sample fit under
#: this (maximum 259; mini 315; nano 446 — nano is not offered for exactly that reason),
#: and a cap cannot change an answer that already fits under it.
MAX_COMPLETION_TOKENS = 400

#: Matches the timestamp prefix an excerpt is rendered with, e.g. `(2024-05-01 09:30) `.
#: 34 of 528 measured spans copied the excerpt's own timestamp in; stripping it before
#: the substring check is what keeps those 34 as kept turns with a real span rather than
#: `None`.
_TS_PREFIX = re.compile(r"^\(\d{4}-\d{2}-\d{2} \d{2}:\d{2}\) ")


def _ts(when: datetime) -> str:
    """`YYYY-MM-DD HH:MM` — to the minute, `T` replaced by a space.

    The first 16 characters of `isoformat()` are always the date and the hour and
    minute, whatever timezone offset or fractional seconds follow, so this needs no
    format string of its own.
    """
    return when.isoformat().replace("T", " ")[:16]


def _prompt(question: str, candidates: Sequence[Candidate], asked_on: datetime) -> str:
    body = "\n\n".join(
        f"[{i + 1}] ({_ts(c.when)}) {c.text}" for i, c in enumerate(candidates))
    return f"Question (asked on {_ts(asked_on)}): {question}\n\nExcerpts:\n\n{body}"


def _clean_span(span: str, text: str) -> str | None:
    """`span` as returned when it is a substring of `text`; stripped of a leading
    `(YYYY-MM-DD HH:MM) ` when stripping is what makes it one; `None` otherwise."""
    if span in text:
        return span
    stripped = _TS_PREFIX.sub("", span, count=1)
    if stripped != span and stripped in text:
        return stripped
    return None


def _parse_reply(text: str, candidates: Sequence[Candidate]) -> list[Selected]:
    """The model's `kept` list, shaped through the same helpers `extract.py`'s uses.

    Raises `ValueError` when the reply is not JSON, or is JSON with no usable `"kept"`
    list — the two ways `local/compress/extract.py`'s own `json.loads(txt)["kept"]`
    raises, whether the text fails to parse at all or parses to an object missing the
    key. Everything past that point is filtered rather than trusted: a malformed entry,
    an out-of-range excerpt number, or an empty span drops that one entry and keeps
    processing the rest, matching `extract.py`'s per-entry `try`/`except`.
    """
    parsed = _shape.parse_json_object(text)
    kept_raw = parsed.get("kept")
    if not isinstance(kept_raw, list):
        raise ValueError("selector reply carried no usable 'kept' list")
    n = len(candidates)
    spans: dict[int, str] = {}
    for entry in kept_raw:
        if not isinstance(entry, dict):
            continue
        raw_i = entry.get("i")
        if isinstance(raw_i, bool) or not isinstance(raw_i, int):
            continue
        idx = _shape.source_index(raw_i - 1, n)
        if idx is None or idx in spans:
            continue
        span = entry.get("span")
        if not isinstance(span, str):
            continue
        span = span.strip()
        if not span:
            continue
        spans[idx] = span
    # Reranked order, not the order the model listed them in — arm B rendered "kept
    # turns whole, in rank order" and that is the measured thing.
    return [Selected(id=c.id, span=_clean_span(spans[i], c.text))
            for i, c in enumerate(candidates) if i in spans]


class ModelSelector:
    """A `Selector` backed by a `Chat` implementation. See the module docstring."""

    def __init__(self, llm: Chat, *, top_n: int = 40, timeout: float = 10.0) -> None:
        if not isinstance(llm, Chat):
            raise TypeError(
                "ModelSelector needs a backend with .chat() — OpenAILLM or "
                "AnthropicLLM (pip install 'memvara[openai]' or 'memvara[anthropic]'), "
                f"not {type(llm).__name__}. NullLLM has no model to consult."
            )
        self._llm = llm
        #: How many of the reranked candidates the caller should hand to `select()`.
        #: Read by the caller (`hybrid.py`'s ranked stage), not applied inside
        #: `select()` itself — this class processes whatever list it is given.
        self.top_n = top_n
        self.timeout = timeout

    @contextmanager
    def admit(self) -> Iterator[None]:
        """Never refuses. The cap this protocol exists to bound belongs to a wrapper
        held around this class — the hosted service's — not to this class itself."""
        yield

    def select(self, question: str, candidates: Sequence[Candidate], *,
               asked_on: datetime | None = None,
               usage: Usage | None = None) -> Sequence[Selected]:
        if not candidates:
            return []  # nothing to ask about, and a call we should not pay for
        asked_on = asked_on if asked_on is not None else utcnow()
        prompt = _prompt(question, candidates, asked_on)
        deadline = time.monotonic() + self.timeout
        try:
            text = self._llm.chat(
                SYSTEM, prompt, json_object=True,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                timeout=self.timeout, usage=usage)
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status in (401, 403):
                raise SelectorRefused("key_rejected", status) from exc
            if time.monotonic() > deadline:
                # The backend's own timeout fired before ours did — an SDK-level
                # exception (e.g. an `APITimeoutError`), not Python's builtin
                # `TimeoutError`, so it would otherwise propagate as `exc` unchanged
                # and be counted as `reason=error` rather than `reason=timeout`. The
                # module docstring's "or the call failed, after the deadline" is this
                # branch: billed either way, so it counts as a timeout either way.
                raise TimeoutError("selector call failed after its deadline") from exc
            raise
        if time.monotonic() > deadline:
            # The call succeeded, but not within the deadline. Billed either way — see
            # the module docstring.
            raise TimeoutError("selector reply arrived after its deadline")
        return _parse_reply(text, candidates)
