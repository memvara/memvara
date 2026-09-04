"""`memvara.select`: the selector protocol, `ModelSelector`, the `Chat` transport, and
the ranked stage in `HybridRetriever.search`/`Memvara.recall`.

The first half covers what Step 1 of the model-ranked-recall design can be tested
without a `HybridRetriever` in the room: the protocol's records and exceptions,
`ModelSelector`'s prompt (byte-identical to `local/compress/extract.py`'s), its parsing
and span-cleaning, its deadline and error handling, the `Chat` protocol, and `chat()` on
both hosted backends.

The second half is the read order this feeds: where `admit()` and `select()` are
actually called from `hybrid.py`, every one of the six ways a ranked read can end (§3,
"The outcomes"), the telemetry each one emits, and what a ranked `search()`/`recall()`
returns. It runs a real `HybridRetriever` over an in-memory `SQLiteStore`, exactly the
harness `tests/test_hybrid.py` uses, with a fake `Selector` standing in for the model.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Iterator, Sequence

import pytest

from memvara.embed import HashingEmbedder
from memvara.llm import Chat, NullLLM
from memvara.llm.anthropic import AnthropicLLM
from memvara.llm.base import Usage
from memvara.llm.openai import OpenAILLM
from memvara.retrieve import EpisodeResult, HybridRetriever
from memvara.schema import PredicateRegistry
from memvara.select import (
    Candidate,
    Selected,
    Selection,
    Selector,
    SelectorBusy,
    SelectorRefused,
)
from memvara.select.model import MAX_COMPLETION_TOKENS, SYSTEM, ModelSelector, _clean_span, _ts
from memvara.store import SQLiteStore
from memvara.telemetry import (
    RETRIEVAL_MODEL_FALLBACK,
    RETRIEVAL_MODEL_QUERY,
    RETRIEVAL_MODEL_REFUSED,
    RETRIEVAL_QUERY,
    RETRIEVAL_SELECT_MS,
    RETRIEVAL_TOKENS_IN,
    RETRIEVAL_TOKENS_OUT,
    MemoryRecorder,
)
from memvara.types import Episode, Scope

UTC = timezone.utc


# --- a fake Chat backend, for exercising ModelSelector without a real client ----------


class FakeChat:
    """A minimal `Chat` implementation whose one call is fully controlled by the test.

    `reply` is returned as-is; `raises`, when set, is raised instead. `delay` sleeps
    before responding — real, short sleeps, used only to push a reply past a very small
    `timeout` deterministically rather than mocking the clock.
    """

    def __init__(self, reply: str | None = None, raises: Exception | None = None,
                 delay: float = 0.0) -> None:
        self.reply = reply
        self.raises = raises
        self.delay = delay
        self.calls: list[dict] = []

    def chat(self, system: str, prompt: str, *, json_object: bool,
             max_completion_tokens: int, timeout: float,
             usage: Usage | None = None) -> str:
        self.calls.append({
            "system": system, "prompt": prompt, "json_object": json_object,
            "max_completion_tokens": max_completion_tokens, "timeout": timeout,
        })
        if self.delay:
            time.sleep(self.delay)
        if self.raises is not None:
            raise self.raises
        if usage is not None:
            usage.add(10, 5)
        assert self.reply is not None
        return self.reply


def _status(status_code: int) -> Exception:
    exc = RuntimeError(f"provider status {status_code}")
    exc.status_code = status_code  # type: ignore[attr-defined]
    return exc


def cands(*texts: str, start: datetime = datetime(2024, 5, 1, 9, 0, tzinfo=UTC)) -> list[Candidate]:
    return [Candidate(id=f"e{i}", when=start, text=t) for i, t in enumerate(texts)]


# --- the Chat protocol ------------------------------------------------------------


def test_chat_is_runtime_checkable_and_a_backend_without_it_fails_the_check() -> None:
    assert isinstance(FakeChat(), Chat)
    assert not isinstance(NullLLM(), Chat)
    assert not isinstance(object(), Chat)


# --- the byte-identical prompt -----------------------------------------------------


#: Copied verbatim from `local/compress/extract.py`'s `SYSTEM` constant. This is the
#: fixture the design spec asks for: a drift here is a silent change to the prompt the
#: 182-question measurement ran against.
_EXTRACT_SYSTEM_FIXTURE = (
    "You filter conversation excerpts for a question-answering system. For each numbered "
    "excerpt, copy out VERBATIM the shortest span or spans that could help answer the question: "
    "every number, name, date, place, quantity, duration, price, product or decision that bears "
    "on it, with just enough surrounding words to keep its meaning. Never paraphrase, never add "
    "words, never answer the question, never merge excerpts. If an excerpt has nothing that bears "
    "on the question, omit it. Be inclusive on the borderline: a partial mention that might "
    "combine with other excerpts is worth keeping. Respond with JSON only: "
    "{\"kept\": [{\"i\": <excerpt number>, \"span\": \"<verbatim text>\"}, ...]}."
)


def test_the_system_message_is_byte_identical_to_extract_pys() -> None:
    assert SYSTEM == _EXTRACT_SYSTEM_FIXTURE


def _extract_py_user_message(question_date: str, question: str, excerpts: list[dict]) -> str:
    """Rebuilds `local/compress/extract.py`'s `call()` user message, verbatim."""
    body = "\n\n".join(
        "[%d] (%s) %s" % (i + 1, e["ts"].replace("T", " ")[:16], e["content"])
        for i, e in enumerate(excerpts))
    return "Question (asked on %s): %s\n\nExcerpts:\n\n%s" % (question_date, question, body)


def test_the_user_message_matches_extract_py_for_a_fixed_candidate_list() -> None:
    excerpts = [
        {"ts": "2024-05-01T09:30:00Z", "content": "I moved to Lisbon last spring."},
        {"ts": "2024-05-02T10:05:00Z", "content": "Started the new job on Monday."},
    ]
    expected = _extract_py_user_message("2024-05-01 09:00", "where do they live?", excerpts)

    candidates = [
        Candidate(id="e0", when=datetime(2024, 5, 1, 9, 30, tzinfo=UTC),
                  text="I moved to Lisbon last spring."),
        Candidate(id="e1", when=datetime(2024, 5, 2, 10, 5, tzinfo=UTC),
                  text="Started the new job on Monday."),
    ]
    asked_on = datetime(2024, 5, 1, 9, 0, tzinfo=UTC)

    llm = FakeChat(reply='{"kept": []}')
    ModelSelector(llm=llm, timeout=5.0).select("where do they live?", candidates, asked_on=asked_on)
    assert llm.calls[0]["system"] == SYSTEM
    assert llm.calls[0]["prompt"] == expected


def test_ts_renders_to_the_minute_with_t_replaced_by_a_space() -> None:
    assert _ts(datetime(2024, 5, 1, 9, 30, 12, tzinfo=UTC)) == "2024-05-01 09:30"


# --- construction --------------------------------------------------------------------


def test_model_selector_refuses_a_backend_with_no_chat_method() -> None:
    with pytest.raises(TypeError, match=r"memvara\[openai\]"):
        ModelSelector(llm=NullLLM())


def test_model_selector_stores_top_n_and_timeout() -> None:
    selector = ModelSelector(llm=FakeChat(), top_n=17, timeout=3.5)
    assert (selector.top_n, selector.timeout) == (17, 3.5)


def test_model_selector_satisfies_the_selector_protocol() -> None:
    assert isinstance(ModelSelector(llm=FakeChat()), Selector)


# --- admit(): never refuses ------------------------------------------------------


def test_admit_never_refuses() -> None:
    selector = ModelSelector(llm=FakeChat())
    with selector.admit():
        pass  # did not raise


# --- select(): the empty case ------------------------------------------------------


def test_select_with_no_candidates_makes_no_call() -> None:
    llm = FakeChat()
    out = ModelSelector(llm=llm).select("anything?", [])
    assert out == []
    assert llm.calls == []


# --- select(): the request itself --------------------------------------------------


def test_select_sends_json_object_and_the_400_token_cap() -> None:
    llm = FakeChat(reply='{"kept": []}')
    ModelSelector(llm=llm, timeout=7.0).select("q", cands("a turn"))
    call = llm.calls[0]
    assert call["json_object"] is True
    assert call["max_completion_tokens"] == MAX_COMPLETION_TOKENS == 400
    assert call["timeout"] == 7.0


def test_select_defaults_asked_on_to_now_when_not_given() -> None:
    import re

    llm = FakeChat(reply='{"kept": []}')
    before = datetime.now(UTC)
    ModelSelector(llm=llm).select("q", cands("a turn"))
    after = datetime.now(UTC)
    match = re.match(r"^Question \(asked on ([\d-]+ [\d:]+)\):", llm.calls[0]["prompt"])
    assert match is not None
    rendered = match.group(1)
    assert before.strftime("%Y-%m-%d %H:%M") <= rendered <= after.strftime("%Y-%m-%d %H:%M")


def test_select_forwards_usage_to_chat() -> None:
    llm = FakeChat(reply='{"kept": []}')
    usage = Usage()
    ModelSelector(llm=llm).select("q", cands("a turn"), usage=usage)
    assert (usage.input_tokens, usage.output_tokens, usage.reported) == (10, 5, 1)


# --- select(): a clean answer ------------------------------------------------------


def test_select_keeps_turns_in_candidate_order_with_their_spans() -> None:
    candidates = cands("The rent is 1200 euros.", "Nothing relevant here.",
                        "They moved to Lisbon in May.")
    reply = json.dumps({"kept": [
        {"i": 3, "span": "moved to Lisbon in May"},
        {"i": 1, "span": "1200 euros"},
    ]})
    out = ModelSelector(llm=FakeChat(reply=reply)).select("where do they live?", candidates)
    # Candidate order (0, then 2), not the order the model listed them in (3, then 1).
    assert [s.id for s in out] == ["e0", "e2"]
    assert out[0].span == "1200 euros"
    assert out[1].span == "moved to Lisbon in May"


def test_select_returns_nothing_kept_when_the_model_keeps_nothing() -> None:
    out = ModelSelector(llm=FakeChat(reply='{"kept": []}')).select("q", cands("irrelevant"))
    assert out == []


# --- span cleaning: the timestamp-prefix strip and the substring rule --------------


def test_clean_span_keeps_a_genuine_substring_untouched() -> None:
    assert _clean_span("moved to Lisbon", "I moved to Lisbon last year.") == "moved to Lisbon"


def test_clean_span_strips_a_copied_timestamp_prefix_that_makes_it_a_substring() -> None:
    text = "I moved to Lisbon last year."
    span = "(2024-05-01 09:30) moved to Lisbon"
    # The raw span, prefix included, is not itself a substring of `text` — but stripping
    # the excerpt timestamp the model copied in front of it makes it one.
    assert span not in text
    assert _clean_span(span, text) == "moved to Lisbon"


def test_clean_span_becomes_none_when_not_a_substring_even_after_stripping() -> None:
    assert _clean_span("a paraphrase of the turn", "the actual turn text") is None


def test_select_keeps_the_turn_even_when_its_span_becomes_none() -> None:
    candidates = cands("the actual turn text")
    reply = json.dumps({"kept": [{"i": 1, "span": "a total paraphrase"}]})
    out = ModelSelector(llm=FakeChat(reply=reply)).select("q", candidates)
    assert len(out) == 1
    assert out[0].id == "e0"
    assert out[0].span is None


# --- select(): malformed entries are skipped, not fatal -----------------------------


@pytest.mark.parametrize("kept_entry", [
    "not-a-dict",
    {"i": "1", "span": "x"},          # i is a string, not an int
    {"i": True, "span": "x"},          # a bool, which is an int in Python
    {"i": 0, "span": "x"},             # excerpt numbers are 1-based; 0 is out of range
    {"i": 99, "span": "x"},            # out of range the other way
    {"span": "x"},                     # no i at all
    {"i": 1, "span": 7},                # span is not a string
    {"i": 1, "span": "   "},           # blank after stripping
    {"i": 1},                          # no span at all
])
def test_a_malformed_kept_entry_is_skipped_not_fatal(kept_entry) -> None:
    candidates = cands("only turn")
    reply = json.dumps({"kept": [kept_entry]})
    out = ModelSelector(llm=FakeChat(reply=reply)).select("q", candidates)
    assert out == []


def test_a_duplicate_index_keeps_only_the_first_occurrence() -> None:
    candidates = cands("only turn")
    reply = json.dumps({"kept": [{"i": 1, "span": "only turn"}, {"i": 1, "span": "second"}]})
    out = ModelSelector(llm=FakeChat(reply=reply)).select("q", candidates)
    assert len(out) == 1 and out[0].span == "only turn"


def test_an_invented_id_cannot_appear_selected_is_built_from_the_candidate_list() -> None:
    # There is no "id" field in the model's reply at all — only "i", an index into the
    # candidate list this call was given. A model cannot invent an id that was not
    # already in `candidates`.
    candidates = cands("only turn")
    reply = json.dumps({"kept": [{"i": 1, "span": "only turn"}]})
    out = ModelSelector(llm=FakeChat(reply=reply)).select("q", candidates)
    assert out[0].id == "e0"


# --- select(): malformed replies raise ValueError -----------------------------------


def test_a_reply_that_is_not_json_raises_value_error() -> None:
    with pytest.raises(ValueError):
        ModelSelector(llm=FakeChat(reply="not json at all")).select("q", cands("x"))


def test_a_reply_with_no_kept_key_raises_value_error() -> None:
    with pytest.raises(ValueError):
        ModelSelector(llm=FakeChat(reply='{"nothing": "here"}')).select("q", cands("x"))


def test_a_reply_whose_kept_is_not_a_list_raises_value_error() -> None:
    with pytest.raises(ValueError):
        ModelSelector(llm=FakeChat(reply='{"kept": "not-a-list"}')).select("q", cands("x"))


# --- select(): the deadline, whatever the call returns ------------------------------


def test_a_reply_that_arrives_after_the_deadline_is_a_timeout_even_though_it_answered() -> None:
    llm = FakeChat(reply='{"kept": []}', delay=0.05)
    with pytest.raises(TimeoutError):
        ModelSelector(llm=llm, timeout=0.01).select("q", cands("x"))


def test_a_reply_within_the_deadline_is_not_a_timeout() -> None:
    llm = FakeChat(reply='{"kept": []}', delay=0.0)
    ModelSelector(llm=llm, timeout=5.0).select("q", cands("x"))  # did not raise


# --- select(): provider errors -------------------------------------------------------


def test_a_401_raises_selector_refused_key_rejected() -> None:
    llm = FakeChat(raises=_status(401))
    with pytest.raises(SelectorRefused) as exc_info:
        ModelSelector(llm=llm).select("q", cands("x"))
    assert (exc_info.value.reason, exc_info.value.status) == ("key_rejected", 401)


def test_a_403_raises_selector_refused_key_rejected() -> None:
    llm = FakeChat(raises=_status(403))
    with pytest.raises(SelectorRefused) as exc_info:
        ModelSelector(llm=llm).select("q", cands("x"))
    assert (exc_info.value.reason, exc_info.value.status) == ("key_rejected", 403)


def test_a_429_propagates_unchanged_it_is_not_key_rejected() -> None:
    original = _status(429)
    llm = FakeChat(raises=original)
    with pytest.raises(RuntimeError) as exc_info:
        ModelSelector(llm=llm).select("q", cands("x"))
    assert exc_info.value is original
    assert exc_info.value.status_code == 429


def test_a_connection_failure_with_no_status_propagates_unchanged() -> None:
    original = ConnectionError("could not connect")
    llm = FakeChat(raises=original)
    with pytest.raises(ConnectionError) as exc_info:
        ModelSelector(llm=llm).select("q", cands("x"))
    assert exc_info.value is original


# --- the protocol's records and exceptions ------------------------------------------


def test_candidate_selected_and_selection_hold_their_fields() -> None:
    c = Candidate(id="e1", when=datetime(2024, 1, 1, tzinfo=UTC), text="hello")
    assert (c.id, c.text) == ("e1", "hello")
    s = Selected(id="e1", span="hello")
    assert (s.id, s.span) == ("e1", "hello")
    sel = Selection(outcome="applied", candidates=40, kept=5)
    assert (sel.outcome, sel.reason, sel.status, sel.candidates, sel.kept) == (
        "applied", None, None, 40, 5)


def test_candidate_is_frozen() -> None:
    c = Candidate(id="e1", when=datetime(2024, 1, 1, tzinfo=UTC), text="hello")
    with pytest.raises((AttributeError, TypeError)):
        c.id = "e2"  # type: ignore[misc]


def test_selector_refused_carries_reason_and_optional_status() -> None:
    disabled = SelectorRefused("disabled")
    assert (disabled.reason, disabled.status) == ("disabled", None)
    rejected = SelectorRefused("key_rejected", 401)
    assert (rejected.reason, rejected.status) == ("key_rejected", 401)


def test_selector_busy_is_a_plain_exception() -> None:
    with pytest.raises(SelectorBusy):
        raise SelectorBusy("cap full")


# --- lazy attribute access on the package ---------------------------------------


def test_model_selector_is_reachable_from_the_package() -> None:
    # `memvara.select.model` is already imported by this file's own top-level import, so
    # this only checks that the PEP 562 `__getattr__` resolves to the same class — the
    # promise that naming it costs no *extra* import is `test_rerank.py`'s subprocess
    # assertion, which starts a fresh interpreter to check it honestly.
    import memvara.select as pkg

    assert pkg.ModelSelector is ModelSelector
    assert "ModelSelector" in pkg.__all__


def test_an_unknown_attribute_on_the_select_package_raises() -> None:
    import memvara.select as pkg

    with pytest.raises(AttributeError, match="no attribute 'Nope'"):
        pkg.Nope


# --- the two hosted backends: chat() ------------------------------------------------


class _FakeOpenAICompletions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.content, refusal=None)
        usage = SimpleNamespace(prompt_tokens=11, completion_tokens=3)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


class _FakeOpenAIClient:
    def __init__(self, content: str = '{"kept": []}') -> None:
        self.completions = _FakeOpenAICompletions(content)
        self.chat = SimpleNamespace(completions=self.completions)


def test_openai_chat_sends_the_messages_json_object_and_cap_and_records_usage() -> None:
    client = _FakeOpenAIClient()
    llm = OpenAILLM(client=client, model="gpt-5.4-mini")
    usage = Usage()
    text = llm.chat("SYS", "PROMPT", json_object=True, max_completion_tokens=400,
                    timeout=9.0, usage=usage)
    assert text == '{"kept": []}'
    call = client.completions.calls[0]
    assert call["model"] == "gpt-5.4-mini"
    assert call["max_completion_tokens"] == 400
    assert call["timeout"] == 9.0
    assert call["messages"] == [
        {"role": "system", "content": "SYS"}, {"role": "user", "content": "PROMPT"}]
    assert call["response_format"] == {"type": "json_object"}
    assert "temperature" not in call
    assert (usage.input_tokens, usage.output_tokens, usage.reported) == (11, 3, 1)


def test_openai_chat_omits_response_format_when_json_object_is_false() -> None:
    client = _FakeOpenAIClient()
    OpenAILLM(client=client).chat("SYS", "PROMPT", json_object=False,
                                  max_completion_tokens=400, timeout=9.0)
    assert "response_format" not in client.completions.calls[0]


def test_openai_chat_surfaces_a_refusal_as_empty_text_not_a_crash() -> None:
    client = _FakeOpenAIClient()

    def refusing_create(**kwargs):
        client.completions.calls.append(kwargs)
        message = SimpleNamespace(content=None, refusal="cannot help with that")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

    client.completions.create = refusing_create
    text = OpenAILLM(client=client).chat("SYS", "PROMPT", json_object=True,
                                         max_completion_tokens=400, timeout=1.0)
    assert text == ""


def test_openai_llm_satisfies_the_chat_protocol() -> None:
    assert isinstance(OpenAILLM(client=_FakeOpenAIClient()), Chat)


def test_openai_llm_base_url_reaches_the_sdk_client(monkeypatch) -> None:
    import sys

    calls: list[dict] = []

    def factory(**kwargs):
        calls.append(kwargs)
        return _FakeOpenAIClient()

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=factory))
    OpenAILLM(base_url="https://hosted.example/v1")
    assert calls == [{"base_url": "https://hosted.example/v1"}]


class _FakeAnthropicMessages:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        block = SimpleNamespace(type="text", text=self.text)
        usage = SimpleNamespace(input_tokens=6, output_tokens=2)
        return SimpleNamespace(content=[block], usage=usage)


class _FakeAnthropicClient:
    def __init__(self, text: str = '{"kept": []}') -> None:
        self.messages = _FakeAnthropicMessages(text)


def test_anthropic_chat_sends_the_messages_and_records_usage() -> None:
    client = _FakeAnthropicClient()
    llm = AnthropicLLM(client=client, model="claude-opus-5")
    usage = Usage()
    text = llm.chat("SYS", "PROMPT", json_object=True, max_completion_tokens=400,
                    timeout=9.0, usage=usage)
    assert text == '{"kept": []}'
    call = client.messages.calls[0]
    assert call["model"] == "claude-opus-5"
    assert call["max_tokens"] == 400
    assert call["timeout"] == 9.0
    assert call["system"] == "SYS"
    assert call["messages"] == [{"role": "user", "content": "PROMPT"}]
    assert (usage.input_tokens, usage.output_tokens, usage.reported) == (6, 2, 1)


def test_anthropic_llm_satisfies_the_chat_protocol() -> None:
    assert isinstance(AnthropicLLM(client=_FakeAnthropicClient()), Chat)


# --- the two backends actually driving ModelSelector end to end --------------------


def test_model_selector_against_the_openai_backend() -> None:
    reply = json.dumps({"kept": [{"i": 1, "span": "Lisbon"}]})
    llm = OpenAILLM(client=_FakeOpenAIClient(reply))
    out = ModelSelector(llm=llm).select("where?", cands("They live in Lisbon."))
    assert out == [Selected(id="e0", span="Lisbon")]


def test_model_selector_against_the_anthropic_backend() -> None:
    reply = json.dumps({"kept": [{"i": 1, "span": "Lisbon"}]})
    llm = AnthropicLLM(client=_FakeAnthropicClient(reply))
    out = ModelSelector(llm=llm).select("where?", cands("They live in Lisbon."))
    assert out == [Selected(id="e0", span="Lisbon")]


# --- the six telemetry constants -----------------------------------------------------


def test_the_six_retrieval_selector_series_appear_in_series_names() -> None:
    from memvara.telemetry import series_names

    names = set(series_names())
    for series in (
        "retrieval.model_query", "retrieval.model_fallback", "retrieval.model_refused",
        "retrieval.tokens_in", "retrieval.tokens_out", "retrieval.select_ms",
    ):
        assert series in names


# =======================================================================================
# Part two: the ranked stage in HybridRetriever.search and Memvara.recall
# =======================================================================================
#
# A real HybridRetriever over an in-memory SQLiteStore — the harness tests/test_hybrid.py
# uses — with a fake Selector standing in for the model call. `ScoreReranker` gives full
# control over turn order without depending on BM25/vector fusion, which is what lets
# these tests assert exact ordering.

EP_SCOPE = Scope("acme", "alice")


def _turn(store: SQLiteStore, embedder: HashingEmbedder, content: str, **kw) -> Episode:
    ep = Episode(content=content, scope=EP_SCOPE, **kw)
    store.add_episode(ep)
    store.set_episode_embedding(ep.id, embedder.encode([content])[0])
    return ep


def _seed(store: SQLiteStore, embedder: HashingEmbedder,
         texts: Sequence[str] = ("one about kayaks", "two about kayaks",
                                  "three about kayaks")) -> list[Episode]:
    return [_turn(store, embedder, t) for t in texts]


def _add_claim(store: SQLiteStore, embedder: HashingEmbedder, text: str):
    from memvara.types import Claim

    claim = Claim(subject="user", predicate="reported", object=text, text=text,
                  scope=EP_SCOPE)
    store.put_claim(claim)
    store.set_embedding(claim.id, embedder.encode([text])[0])
    return claim


class ScoreReranker:
    """Deterministic reranker: orders documents by their position in `order`, first
    highest. Records every call's document list, for the admission-ordering test."""

    def __init__(self, order: Sequence[str]) -> None:
        self._rank = {text: len(order) - i for i, text in enumerate(order)}
        self.calls: list[list[str]] = []

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        self.calls.append(list(documents))
        return [float(self._rank.get(d, 0)) for d in documents]


class FakeSelector:
    """A `Selector` whose `admit()` and `select()` are both fully scripted.

    `admit_raises` is what `admit()` itself raises — `SelectorRefused("disabled")` or
    `SelectorBusy`. `select_raises` is what `select()` raises — `SelectorRefused`,
    `TimeoutError`, `ValueError`, or any other exception (`status_code` optional, for the
    provider-error case). `keep` is how many of the candidates it names when it does not
    raise (the first `keep`, in the order it was handed them), `spans` an id -> span map
    for the ones it does. `usage_tokens`, when set, is added to the `usage` accumulator
    hybrid.py passes in — the shape a real backend fills. `log`, when given, records
    `"admit"` and `"select"` in call order, for the admission-precedes-reranking test.
    """

    def __init__(self, *, top_n: int = 5, admit_raises: Exception | None = None,
                 select_raises: Exception | None = None, keep: int = 0,
                 spans: dict | None = None, usage_tokens: tuple | None = None,
                 log: list | None = None) -> None:
        self.top_n = top_n
        self._admit_raises = admit_raises
        self._select_raises = select_raises
        self._keep = keep
        self._spans = spans or {}
        self._usage_tokens = usage_tokens
        self._log = log
        self.admit_calls = 0
        self.select_calls = 0
        self.seen: list[Candidate] = []

    @contextmanager
    def admit(self) -> Iterator[None]:
        self.admit_calls += 1
        if self._log is not None:
            self._log.append("admit")
        if self._admit_raises is not None:
            raise self._admit_raises
        yield

    def select(self, question: str, candidates: Sequence[Candidate], *,
               asked_on: datetime | None = None, usage: Usage | None = None
               ) -> Sequence[Selected]:
        self.select_calls += 1
        self.seen = list(candidates)
        if self._log is not None:
            self._log.append("select")
        if self._usage_tokens is not None and usage is not None:
            usage.add(*self._usage_tokens)
        if self._select_raises is not None:
            raise self._select_raises
        kept = candidates[:self._keep]
        return [Selected(id=c.id, span=self._spans.get(c.id)) for c in kept]


def _status_exc(status: int) -> Exception:
    exc = RuntimeError(f"provider status {status}")
    exc.status_code = status  # type: ignore[attr-defined]
    return exc


@pytest.fixture
def store() -> SQLiteStore:
    return SQLiteStore(":memory:")


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=64)


def _engine(store: SQLiteStore, embedder: HashingEmbedder, selector, *,
           reranker=None, rerank_top_n: int = 20, max_episodes: int = 3,
           rerank_ranked_only: bool = False, telemetry=None) -> HybridRetriever:
    return HybridRetriever(
        store, embedder, PredicateRegistry(), reranker=reranker,
        rerank_top_n=rerank_top_n, max_episodes=max_episodes, selector=selector,
        rerank_ranked_only=rerank_ranked_only, telemetry=telemetry)


# --- the counting fake: top_n candidates, reranked order, admission ordering ----------


def test_the_selector_sees_exactly_top_n_candidates_in_reranked_order(store, embedder) -> None:
    eps = _seed(store, embedder)
    reranker = ScoreReranker([e.content for e in reversed(eps)])
    selector = FakeSelector(top_n=2)
    r = _engine(store, embedder, selector, reranker=reranker, rerank_top_n=20)

    r.search("kayaks", EP_SCOPE, k=8, include_episodes=True, ranked=True)

    assert selector.select_calls == 1
    assert len(selector.seen) == 2
    assert all(isinstance(c, Candidate) for c in selector.seen)
    assert [c.text for c in selector.seen] == [e.content for e in reversed(eps)][:2]


def test_the_selector_is_never_called_on_a_plain_read(store, embedder) -> None:
    _seed(store, embedder)
    selector = FakeSelector(top_n=3)
    r = _engine(store, embedder, selector, rerank_top_n=20)

    r.search("kayaks", EP_SCOPE, k=8, include_episodes=True)  # ranked=False (default)

    assert selector.admit_calls == 0
    assert selector.select_calls == 0


def test_admission_precedes_the_reranker_call(store, embedder) -> None:
    eps = _seed(store, embedder)
    log: list[str] = []
    inner = ScoreReranker([e.content for e in eps])

    class LoggingReranker:
        def score(self, query: str, documents):
            log.append("rerank")
            return inner.score(query, documents)

    selector = FakeSelector(top_n=3, log=log)
    r = _engine(store, embedder, selector, reranker=LoggingReranker(), rerank_top_n=20)

    r.search("kayaks", EP_SCOPE, k=8, include_episodes=True, ranked=True)

    assert log == ["admit", "rerank", "select"]


# --- the episode cap: rerank_top_n on a ranked call, max_episodes on a plain one -------


def test_ranked_call_gathers_episodes_at_rerank_top_n_not_max_episodes(store, embedder) -> None:
    _seed(store, embedder, texts=[f"kayak turn {i}" for i in range(10)])
    selector = FakeSelector(top_n=8)
    r = _engine(store, embedder, selector, rerank_top_n=8, max_episodes=2)

    r.search("kayak", EP_SCOPE, k=20, include_episodes=True, ranked=True)

    assert selector.select_calls == 1
    assert len(selector.seen) == 8  # rerank_top_n, not max_episodes=2


def test_plain_call_still_caps_episodes_at_max_episodes(store, embedder) -> None:
    _seed(store, embedder, texts=[f"kayak turn {i}" for i in range(10)])
    selector = FakeSelector(top_n=8)
    r = _engine(store, embedder, selector, rerank_top_n=8, max_episodes=2)

    result = r.search("kayak", EP_SCOPE, k=20, include_episodes=True)  # ranked=False

    episodes = [x for x in result if isinstance(x, EpisodeResult)]
    assert len(episodes) == 2


def test_claim_density_does_not_shrink_the_selectors_turn_pool(store, embedder) -> None:
    """The rationale in "Where it sits" step 2: lifting the episode cap alone would not
    give the selector its turns if claims crowded the shared cut. `_episodes` no longer
    goes through `_interleave` on a ranked call, so a dense claim population changes
    nothing about how many turns the selector sees."""
    _seed(store, embedder, texts=[f"kayak turn {i}" for i in range(8)])
    for i in range(20):
        _add_claim(store, embedder, f"kayak claim {i}")
    selector = FakeSelector(top_n=8)
    r = _engine(store, embedder, selector, rerank_top_n=8, max_episodes=2)

    r.search("kayak", EP_SCOPE, k=20, include_episodes=True, ranked=True)

    assert len(selector.seen) == 8


# --- read_rerank_ranked_only ------------------------------------------------------------


def test_ranked_only_switch_skips_the_reranker_on_a_plain_read_and_runs_it_on_a_ranked_one(
    store, embedder,
) -> None:
    eps = _seed(store, embedder)
    reranker = ScoreReranker([e.content for e in eps])
    selector = FakeSelector(top_n=3)
    r = _engine(store, embedder, selector, reranker=reranker, rerank_top_n=20,
               rerank_ranked_only=True)

    r.search("kayaks", EP_SCOPE, k=5, include_episodes=True)  # ranked=False
    assert reranker.calls == []

    r.search("kayaks", EP_SCOPE, k=5, include_episodes=True, ranked=True)
    assert len(reranker.calls) == 1


def test_ranked_only_false_reranks_both_kinds_of_read(store, embedder) -> None:
    eps = _seed(store, embedder)
    reranker = ScoreReranker([e.content for e in eps])
    selector = FakeSelector(top_n=3)
    r = _engine(store, embedder, selector, reranker=reranker, rerank_top_n=20,
               rerank_ranked_only=False)

    r.search("kayaks", EP_SCOPE, k=5, include_episodes=True)
    assert len(reranker.calls) == 1

    r.search("kayaks", EP_SCOPE, k=5, include_episodes=True, ranked=True)
    assert len(reranker.calls) == 2


class _CountingStore:
    """Wraps a `Store`, recording the `limit` each episode-search call was given."""

    def __init__(self, inner: SQLiteStore) -> None:
        self._inner = inner
        self.limits: list[int] = []

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def vector_search_episodes(self, qvec, scopes, limit, *a, **kw):
        self.limits.append(limit)
        return self._inner.vector_search_episodes(qvec, scopes, limit, *a, **kw)


def test_ranked_only_plain_read_gathers_exactly_as_a_retriever_with_no_reranker(
    embedder,
) -> None:
    store_a = SQLiteStore(":memory:")
    _seed(store_a, embedder)
    counting_a = _CountingStore(store_a)
    r_switch = HybridRetriever(counting_a, embedder, PredicateRegistry(),
                               reranker=ScoreReranker([]), rerank_top_n=20,
                               rerank_ranked_only=True)
    r_switch.search("kayaks", EP_SCOPE, k=5, include_episodes=True)

    store_b = SQLiteStore(":memory:")
    _seed(store_b, embedder)
    counting_b = _CountingStore(store_b)
    r_none = HybridRetriever(counting_b, embedder, PredicateRegistry(), reranker=None)
    r_none.search("kayaks", EP_SCOPE, k=5, include_episodes=True)

    assert counting_a.limits == counting_b.limits
    assert counting_a.limits, "the vector leg should have been asked at least once"


def test_ranked_read_gathers_at_rerank_top_n(store, embedder) -> None:
    _seed(store, embedder, texts=[f"kayak turn {i}" for i in range(10)])
    counting = _CountingStore(store)
    selector = FakeSelector(top_n=5)
    r = HybridRetriever(counting, embedder, PredicateRegistry(),
                        reranker=ScoreReranker([]), rerank_top_n=15,
                        rerank_ranked_only=True, selector=selector)

    r.search("kayak", EP_SCOPE, k=5, include_episodes=True, ranked=True)

    # depth = max(k, rerank_top_n) = 15 with a ranked call, whatever the switch says.
    assert counting.limits and max(counting.limits) >= 15


# --- applied: kept turns first, whole, outside k, selected/span set -------------------


def test_applied_kept_turns_first_whole_outside_k_with_selected_and_span(store, embedder) -> None:
    eps = _seed(store, embedder, texts=["alpha kayak", "beta kayak", "gamma kayak"])
    reranker = ScoreReranker([e.content for e in eps])  # alpha, beta, gamma order
    selector = FakeSelector(top_n=3, keep=1, spans={eps[0].id: "alpha span"})
    _add_claim(store, embedder, "a claim about kayak")
    r = _engine(store, embedder, selector, reranker=reranker, rerank_top_n=20)

    result = r.search("kayak", EP_SCOPE, k=1, include_episodes=True, ranked=True)

    assert result.selection == Selection(outcome="applied", candidates=3, kept=1)
    assert isinstance(result[0], EpisodeResult) and result[0].episode.id == eps[0].id
    assert result[0].explain.selected is True
    assert result[0].explain.span == "alpha span"
    # The kept turn arrived outside k=1, so k still bounds everything that follows it.
    assert len(result) == 2


def test_unkept_seen_turns_get_selected_false_claims_stay_none(store, embedder) -> None:
    eps = _seed(store, embedder, texts=["alpha kayak", "beta kayak"])
    reranker = ScoreReranker([e.content for e in eps])
    selector = FakeSelector(top_n=2, keep=1)  # keeps eps[0] only
    _add_claim(store, embedder, "a kayak claim")
    r = _engine(store, embedder, selector, reranker=reranker, rerank_top_n=20)

    result = r.search("kayak", EP_SCOPE, k=5, include_episodes=True, ranked=True)

    by_id = {x.episode.id: x for x in result if isinstance(x, EpisodeResult)}
    assert by_id[eps[0].id].explain.selected is True
    assert by_id[eps[1].id].explain.selected is False
    claims = [x for x in result if not isinstance(x, EpisodeResult)]
    assert claims and all(x.explain.selected is None for x in claims)


# --- fallback: every failure mode, the reranked order, and its telemetry --------------


@pytest.mark.parametrize("exc,reason,status", [
    (ValueError("not json"), "malformed", None),
    (TimeoutError("late"), "timeout", None),
    (_status_exc(429), "provider", 429),
    (ConnectionError("down"), "error", None),
])
def test_a_failed_model_call_falls_back_with_the_reranked_order(
    store, embedder, exc, reason, status,
) -> None:
    eps = _seed(store, embedder, texts=["alpha kayak", "beta kayak"])
    reranker = ScoreReranker([e.content for e in eps])
    selector = FakeSelector(top_n=2, select_raises=exc)
    telemetry = MemoryRecorder()
    r = _engine(store, embedder, selector, reranker=reranker, rerank_top_n=20,
               telemetry=telemetry)

    result = r.search("kayak", EP_SCOPE, k=5, include_episodes=True, ranked=True)

    assert result.selection == Selection(outcome="fallback", reason=reason,
                                         status=status, candidates=2)
    episodes = [x for x in result if isinstance(x, EpisodeResult)]
    assert [x.episode.id for x in episodes] == [e.id for e in eps]  # reranked order
    assert all(x.explain.selected is None for x in episodes)
    assert telemetry.total(RETRIEVAL_MODEL_FALLBACK, reason=reason) == 1
    assert len(telemetry.values(RETRIEVAL_SELECT_MS)) == 1
    assert telemetry.total(RETRIEVAL_MODEL_QUERY) == 0
    assert telemetry.values(RETRIEVAL_TOKENS_IN) == []


# --- disabled, key_rejected, unconfigured, busy, and the empty-result guarantee -------


def test_disabled_serves_unranked_with_no_reranker_call_and_no_select_ms(store, embedder) -> None:
    eps = _seed(store, embedder)
    reranker = ScoreReranker([e.content for e in eps])
    selector = FakeSelector(top_n=3, admit_raises=SelectorRefused("disabled"))
    telemetry = MemoryRecorder()
    r = _engine(store, embedder, selector, reranker=reranker, rerank_top_n=20,
               telemetry=telemetry)

    result = r.search("kayaks", EP_SCOPE, k=5, include_episodes=True, ranked=True)

    assert result.selection == Selection(outcome="disabled", candidates=0)
    assert reranker.calls == [], "nothing is spent on the cross-encoder"
    assert telemetry.total(RETRIEVAL_MODEL_REFUSED, reason="disabled") == 1
    assert telemetry.values(RETRIEVAL_SELECT_MS) == []


def test_key_rejected_401_carries_its_status_and_still_made_the_call(store, embedder) -> None:
    _seed(store, embedder, texts=["alpha", "beta", "gamma"])
    selector = FakeSelector(top_n=3, select_raises=SelectorRefused("key_rejected", 401))
    telemetry = MemoryRecorder()
    r = _engine(store, embedder, selector, rerank_top_n=20, telemetry=telemetry)

    result = r.search("alpha beta gamma", EP_SCOPE, k=5, include_episodes=True,
                      ranked=True)

    assert result.selection.outcome == "key_rejected"
    assert result.selection.status == 401
    assert telemetry.total(RETRIEVAL_MODEL_REFUSED, reason="key_rejected") == 1
    assert len(telemetry.values(RETRIEVAL_SELECT_MS)) == 1, "the call was made"


def test_unconfigured_retriever_serves_unranked_and_no_leg_runs_differently(
    store, embedder,
) -> None:
    _seed(store, embedder, texts=[f"kayak turn {i}" for i in range(10)])
    telemetry = MemoryRecorder()
    plain = HybridRetriever(store, embedder, PredicateRegistry(), max_episodes=2,
                            telemetry=telemetry)
    unconfigured_result = plain.search("kayak", EP_SCOPE, k=5, include_episodes=True,
                                       ranked=True)
    plain_result = plain.search("kayak", EP_SCOPE, k=5, include_episodes=True)

    assert unconfigured_result.selection == Selection(outcome="unconfigured", candidates=0)
    assert telemetry.total(RETRIEVAL_MODEL_REFUSED, reason="unconfigured") == 1
    assert telemetry.values(RETRIEVAL_SELECT_MS) == []
    # "No leg run differently": the episode cap is still `max_episodes`, unaffected by
    # `ranked=True` on an unconfigured retriever.
    assert (len([x for x in unconfigured_result if isinstance(x, EpisodeResult)])
           == len([x for x in plain_result if isinstance(x, EpisodeResult)]) == 2)


def test_busy_propagates_and_emits_no_retrieval_query(store, embedder) -> None:
    _seed(store, embedder)
    selector = FakeSelector(top_n=3, admit_raises=SelectorBusy("full"))
    telemetry = MemoryRecorder()
    r = _engine(store, embedder, selector, telemetry=telemetry)

    with pytest.raises(SelectorBusy):
        r.search("kayaks", EP_SCOPE, k=5, include_episodes=True, ranked=True)

    assert telemetry.total(RETRIEVAL_MODEL_REFUSED, reason="inflight") == 1
    assert telemetry.total(RETRIEVAL_QUERY) == 0


def test_an_empty_ranked_result_still_carries_its_selection(store, embedder) -> None:
    # No turns and no claims at all — the store is empty.
    selector = FakeSelector(top_n=3)
    r = _engine(store, embedder, selector, rerank_top_n=20)

    result = r.search("nothing stored here", EP_SCOPE, k=5, include_episodes=True,
                      ranked=True)

    assert list(result) == []
    assert result.selection == Selection(outcome="applied", candidates=0, kept=0)


# --- token series: only when Usage.reported > 0, and retrieval.model_query -------------


def test_tokens_and_model_query_emitted_when_usage_reports_something(store, embedder) -> None:
    _seed(store, embedder)
    selector = FakeSelector(top_n=3, keep=1, usage_tokens=(10, 5))
    telemetry = MemoryRecorder()
    r = _engine(store, embedder, selector, rerank_top_n=20, telemetry=telemetry)

    r.search("kayaks", EP_SCOPE, k=5, include_episodes=True, ranked=True)

    assert telemetry.total(RETRIEVAL_TOKENS_IN) == 10
    assert telemetry.total(RETRIEVAL_TOKENS_OUT) == 5
    assert telemetry.total(RETRIEVAL_MODEL_QUERY) == 1


def test_no_token_series_when_usage_reports_nothing(store, embedder) -> None:
    _seed(store, embedder)
    selector = FakeSelector(top_n=3, keep=1)  # usage_tokens left unset
    telemetry = MemoryRecorder()
    r = _engine(store, embedder, selector, rerank_top_n=20, telemetry=telemetry)

    r.search("kayaks", EP_SCOPE, k=5, include_episodes=True, ranked=True)

    assert telemetry.total(RETRIEVAL_TOKENS_IN) == 0
    assert telemetry.total(RETRIEVAL_TOKENS_OUT) == 0
    assert telemetry.total(RETRIEVAL_MODEL_QUERY) == 1


# --- the two ValueErrors: no turns to rank ----------------------------------------------


def test_ranked_with_include_episodes_false_raises_value_error(store, embedder) -> None:
    selector = FakeSelector(top_n=3)
    r = _engine(store, embedder, selector)
    with pytest.raises(ValueError, match="ranked=True"):
        r.search("q", EP_SCOPE, k=5, ranked=True, include_episodes=False)


def test_ranked_with_memory_types_raises_value_error(store, embedder) -> None:
    selector = FakeSelector(top_n=3)
    r = _engine(store, embedder, selector)
    with pytest.raises(ValueError, match="ranked=True"):
        r.search("q", EP_SCOPE, k=5, ranked=True, include_episodes=True,
                memory_types=["semantic"])


# --- recall(ranked=True) -----------------------------------------------------------------

from memvara import Memvara  # noqa: E402 - after the fixtures that build the fakes it needs

LONG_TURN = ("x" * 400) + " kayak"  # longer than Memvara.RECALL_EPISODE_CHARS (280)


def test_recall_ranked_keeps_k_claims_when_the_model_keeps_k_or_more_turns() -> None:
    embedder = HashingEmbedder(dim=64)
    selector = FakeSelector(top_n=5, keep=5)
    mem = Memvara(llm=NullLLM(), user="alice", embedder=embedder,
                  read_selector=selector, read_rerank_top_n=20)
    for i in range(6):
        mem.add(f"turn number {i} about kayaking")
    # Both share the query's vocabulary, so the lexical leg finds both of them —
    # `remember("user", "lives_in", "Lisbon")` shares nothing with "kayaking" and the
    # lexical leg simply never returns it, which made an earlier version of this test
    # flaky on exactly the axis it means to check.
    mem.remember("user", "likes", "kayaking")
    mem.remember("user", "avoids", "kayaking in storms")

    block = mem.recall("kayaking", include_episodes=True, ranked=True, k=2, with_ids=True)

    assert len(block.claim_ids) == 2
    assert block.selection.outcome == "applied"


def test_recall_ranked_renders_the_kept_turn_whole() -> None:
    embedder = HashingEmbedder(dim=64)
    selector = FakeSelector(top_n=5, keep=1)
    mem = Memvara(llm=NullLLM(), user="alice", embedder=embedder,
                  read_selector=selector, read_rerank_top_n=20)
    mem.add(LONG_TURN)

    block = mem.recall("kayak", include_episodes=True, ranked=True)

    assert LONG_TURN in block


def test_recall_ranked_renders_an_unkept_turn_at_280_characters() -> None:
    embedder = HashingEmbedder(dim=64)
    selector = FakeSelector(top_n=5, keep=0)  # the model kept nothing
    mem = Memvara(llm=NullLLM(), user="alice", embedder=embedder,
                  read_selector=selector, read_rerank_top_n=20)
    mem.add(LONG_TURN)

    block = mem.recall("kayak", include_episodes=True, ranked=True)

    assert LONG_TURN not in block


def test_recall_ranked_budget_can_still_cut_a_kept_turn_and_recall_dropped_counts_it() -> None:
    embedder = HashingEmbedder(dim=64)
    selector = FakeSelector(top_n=5, keep=2)
    mem = Memvara(llm=NullLLM(), user="alice", embedder=embedder,
                  read_selector=selector, read_rerank_top_n=20)
    mem.add("first kayak turn here")
    mem.add("second kayak turn here")

    unbudgeted = mem.recall("kayak", include_episodes=True, ranked=True, with_ids=True)
    assert unbudgeted.dropped == 0

    tiny = mem.recall("kayak", include_episodes=True, ranked=True, budget=1, with_ids=True)
    assert tiny.dropped == 2
    assert "did not fit" in tiny.text


def test_recall_ranked_ends_with_the_unranked_line_when_the_model_did_not_rank_it() -> None:
    embedder = HashingEmbedder(dim=64)
    selector = FakeSelector(top_n=5, select_raises=TimeoutError("late"))
    mem = Memvara(llm=NullLLM(), user="alice", embedder=embedder,
                  read_selector=selector, read_rerank_top_n=20)
    mem.add("a kayak turn")

    block = mem.recall("kayak", include_episodes=True, ranked=True, with_ids=True)

    assert block.text.rstrip().endswith("fallback.)")
    assert block.selection.outcome == "fallback"


def test_recall_ranked_carries_no_unranked_line_when_applied() -> None:
    embedder = HashingEmbedder(dim=64)
    selector = FakeSelector(top_n=5, keep=1)
    mem = Memvara(llm=NullLLM(), user="alice", embedder=embedder,
                  read_selector=selector, read_rerank_top_n=20)
    mem.add("a kayak turn")

    block = mem.recall("kayak", include_episodes=True, ranked=True)

    assert "model ranking not applied" not in block


def test_a_plain_recall_carries_no_selection() -> None:
    embedder = HashingEmbedder(dim=64)
    mem = Memvara(llm=NullLLM(), user="alice", embedder=embedder)
    mem.remember("user", "lives_in", "Lisbon")

    block = mem.recall("where do they live", with_ids=True)

    assert block.selection is None
