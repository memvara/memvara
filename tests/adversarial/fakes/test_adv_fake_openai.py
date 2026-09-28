"""FakeOpenAI answers in the order it was scripted, records what was sent, and memvara's
own OpenAI client talks to it."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx
import pytest

from harness.fakes.openai_compat import COMPLETIONS, FakeOpenAI, FakeOpenAIError
from memvara import Memvara
from memvara.embed import HashingEmbedder
from memvara.llm.base import Message, ToolSpec, TruncatedResponse
from memvara.llm.openai import OpenAILLM
from memvara.types import Episode

#: Windows' monotonic clock ticks every 15.6 ms, so a wait can measure up to one tick
#: shorter than the time it waited. Every lower bound on a measured wait allows for it.
CLOCK_TICK = 0.016


#: A turn the fast path does not recognise, so a store with a model has to ask it.
TURN = "I have been practising the cello every evening since spring."

#: What a model answers the extraction call with, in the shape `OpenAILLM` asks for.
CLAIMS = {"claims": [{"subject": "user", "predicate": "likes", "object": "the cello",
                      "polarity": 1, "memory_type": "semantic", "confidence": 0.9,
                      "source_index": 0, "when": None, "amount": None, "unit": None}]}


def _post(fake: FakeOpenAI, content: str) -> httpx.Response:
    return httpx.post(fake.base_url + "/chat/completions",
                      json={"model": "m", "messages": [{"role": "user", "content": content}]})


def test_replies_come_back_in_the_order_they_were_scripted(fake_openai: FakeOpenAI) -> None:
    fake_openai.add_reply("first")
    fake_openai.add_reply("second")
    answers = [_post(fake_openai, str(n)).json() for n in (1, 2)]
    assert [a["choices"][0]["message"]["content"] for a in answers] == ["first", "second"]
    assert [a["model"] for a in answers] == ["m", "m"]
    assert [r.json()["messages"][0]["content"] for r in fake_openai.requests] == ["1", "2"]
    assert fake_openai.pending == 0
    exhausted = _post(fake_openai, "3")
    assert exhausted.status_code == 500 and "no scripted reply left" in exhausted.text


def test_memvara_extracts_a_claim_through_the_fake(fake_openai: FakeOpenAI) -> None:
    fake_openai.add_json(CLAIMS, prompt_tokens=12, completion_tokens=7)
    mem = Memvara(embedder=HashingEmbedder(dim=512),
                  llm=OpenAILLM(client=fake_openai.client(), model="fake-model"))
    try:
        receipt = mem.scope(user="alice").add(TURN)
    finally:
        mem.close()
    assert [(c.predicate, c.object) for c in receipt.added] == [("likes", "the cello")]
    assert (receipt.llm_calls, receipt.tokens_in, receipt.tokens_out) == (1, 12, 7)
    (sent,) = fake_openai.requests
    body = sent.json()
    assert body["model"] == "fake-model"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert TURN in body["messages"][1]["content"]
    assert sent.header("authorization") == "Bearer sk-fake"


def test_output_the_client_cannot_use_extracts_nothing(fake_openai: FakeOpenAI) -> None:
    fake_openai.add_reply("Sure! Here are the facts you asked for.")
    fake_openai.add_raw({"id": "x", "choices": []})
    fake_openai.add_reply(json.dumps(CLAIMS), finish_reason="length")
    fake_openai.add_raw("<html>not a completion</html>")
    llm = OpenAILLM(client=fake_openai.client())
    turn = [Episode(content=TURN)]
    assert llm.extract(turn, []) == []
    assert llm.extract(turn, []) == []
    with pytest.raises(TruncatedResponse):
        llm.extract(turn, [])
    with pytest.raises(ValueError):
        llm.extract(turn, [])


def test_a_429_reaches_the_client_and_the_store_still_keeps_the_turn(
        fake_openai: FakeOpenAI) -> None:
    fake_openai.add_rate_limit(retry_after=7)
    fake_openai.add_rate_limit(retry_after=None)
    llm = OpenAILLM(client=fake_openai.client())
    with pytest.raises(FakeOpenAIError) as caught:
        llm.chat("system", "prompt", json_object=False, max_completion_tokens=10, timeout=5)
    assert (caught.value.status, caught.value.retry_after) == (429, "7")
    mem = Memvara(embedder=HashingEmbedder(dim=512), llm=llm)
    try:
        receipt = mem.scope(user="alice").add(TURN)
        assert (receipt.added, receipt.unextracted, len(receipt.episode_ids)) == ([], 1, 1)
        assert mem.stats()["episodes"] == 1
    finally:
        mem.close()


def test_a_hang_is_cut_off_by_the_client_s_timeout(fake_openai: FakeOpenAI) -> None:
    fake_openai.add_hang()
    fake_openai.add_reply("after the hang")
    llm = OpenAILLM(client=fake_openai.client())
    started = time.monotonic()
    with pytest.raises(httpx.ReadTimeout):
        llm.chat("system", "prompt", json_object=False, max_completion_tokens=10,
                 timeout=0.3)
    assert 0.3 - CLOCK_TICK <= time.monotonic() - started < 5
    assert llm.chat("system", "prompt", json_object=False, max_completion_tokens=10,
                    timeout=5) == "after the hang"


def test_a_tool_call_runs_the_tool_and_its_result_goes_back_to_the_model(
        fake_openai: FakeOpenAI) -> None:
    fake_openai.add_tool_calls(("lookup", {"key": "city"}))
    fake_openai.add_reply("The user lives in Lisbon.")
    asked: list[dict[str, Any]] = []

    def lookup(arguments: dict[str, Any]) -> str:
        asked.append(arguments)
        return "Lisbon"

    tool = ToolSpec("lookup", "Look a stored value up by its key.",
                    {"type": "object", "properties": {"key": {"type": "string"}},
                     "required": ["key"], "additionalProperties": False}, lookup)
    run = OpenAILLM(client=fake_openai.client()).run_tools(
        "system", [Message("user", "Where does the user live?")], [tool], max_steps=3,
        timeout=10)
    assert (run.steps, run.requests, run.finished, run.text) == (
        2, 2, True, "The user lives in Lisbon.")
    assert asked == [{"key": "city"}]
    assert fake_openai.requests[1].json()["messages"][-1] == {
        "role": "tool", "tool_call_id": "call_0", "content": "Lisbon"}


def test_a_route_fault_applies_before_the_script_and_leaves_it_alone(
        fake_openai: FakeOpenAI) -> None:
    fake_openai.fail("POST /v1/chat/completions", 503, times=1)
    fake_openai.add_reply("still first")
    assert _post(fake_openai, "1").status_code == 503
    assert _post(fake_openai, "2").json()["choices"][0]["message"]["content"] == "still first"


def test_a_delay_waits_and_then_answers_with_the_next_scripted_reply(
        fake_openai: FakeOpenAI) -> None:
    fake_openai.delay(COMPLETIONS, 0.2, times=1)
    fake_openai.add_reply("after the wait")
    started = time.monotonic()
    answer = _post(fake_openai, "1")
    assert time.monotonic() - started >= 0.2
    assert answer.status_code == 200
    assert answer.json()["choices"][0]["message"]["content"] == "after the wait"
    assert fake_openai.pending == 0


def test_a_rate_limit_writes_a_whole_wait_as_whole_seconds(fake_openai: FakeOpenAI) -> None:
    fake_openai.add_rate_limit(retry_after=1_000_000)
    fake_openai.add_rate_limit(retry_after=2.5)
    assert _post(fake_openai, "1").headers["retry-after"] == "1000000"
    assert _post(fake_openai, "2").headers["retry-after"] == "2.5"
