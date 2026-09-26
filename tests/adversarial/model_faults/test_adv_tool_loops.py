"""Agentic extraction when the model runs away, or its run fails in any other way.

Agentic extraction is off by default, which the suite's design lists as documented
behaviour, so every test here switches it on. The model's tool loop is memvara's own
(`memvara.llm._tools.run_loop`, which the scripted model runs as both shipped backends
do), so the step limit, the retry and the time budget below are the shipped ones.
INTERNALS, the `agentic_extraction` entry: "At most `AGENTIC_MAX_STEPS` (12) answers", one
retry per answer, a budget of 25 seconds in `add()` and 180 in `reextract()`, a fallback
to single-call extraction that says why, and "Every request the run sent is billed in
`llm_calls`, fallback or not."
"""

from __future__ import annotations

from typing import Callable

import pytest

from .handles import MODEL_TURN, PORTO, agentic, fates, ledger, model_claims, seed
from .scripted import (
    APITimeoutError, Answer, Forever, Late, RateLimitError, ScriptedModel, Truncated,
)

Make = Callable[..., ScriptedModel]

SEARCH = ("search_memories", {"query": "office", "k": 3})

#: What single-call extraction answers when a run falls back to it.
FALLBACK = [PORTO]


def test_a_model_that_never_stops_is_stopped_at_twelve_answers_and_each_is_billed(
        scripted: Make) -> None:
    model = scripted(tools=[Forever(Answer(calls=(SEARCH,)))], extract=[FALLBACK])
    mem = agentic(model)
    seed(mem)
    before = ledger(mem)
    receipt = mem.add(MODEL_TURN)
    assert receipt.agentic_fallback == "step_limit"
    assert model.counts() == {"run_tools": 12, "extract": 1}
    assert receipt.llm_calls == model.count() == 13
    assert model_claims(mem) == ["Porto"]
    assert (model.runs[0]["max_steps"], model.runs[0]["timeout"]) == (12, 25.0)
    assert set(fates(before, mem).values()) == {"unchanged"}


def test_a_run_stopped_at_the_step_limit_loses_its_proposals(scripted: Make) -> None:
    """Every answer reads Berlin and proposes to end it. INTERNALS: a run past 12 answers
    is `step_limit`, "and its proposals are discarded"."""
    model = scripted(extract=[FALLBACK])
    mem = agentic(model)
    ids = seed(mem)
    before = ledger(mem)
    ending = ("propose_end", {"claim_id": ids["berlin"], "reason": "gone",
                              "source_index": 0})
    model.queue("run_tools", Forever(Answer(calls=(
        ("get_claim", {"claim_id": ids["berlin"]}), ending))))
    receipt = mem.add("Berlin is behind me now, the flat there is gone.")
    assert receipt.agentic_fallback == "step_limit"
    assert set(fates(before, mem).values()) == {"unchanged"}
    assert receipt.llm_calls == model.count() == 13


def test_an_unusable_answer_is_retried_once_and_billed_twice(scripted: Make) -> None:
    model = scripted(tools=[Truncated(), Answer("done")])
    mem = agentic(model)
    receipt = mem.add(MODEL_TURN)
    assert receipt.agentic_fallback is None
    assert receipt.llm_calls == model.count("run_tools") == 2


@pytest.mark.parametrize("unusable", [
    pytest.param(Truncated(), id="cut-off-at-the-limit"),
    pytest.param(Answer(calls=(("erase_claim", {}),)), id="a-tool-not-offered"),
    pytest.param(Answer(calls=(("search_memories", "office"),)),
                 id="arguments-not-an-object"),
])
def test_two_unusable_answers_in_a_row_fall_back_to_one_extraction_call(
        scripted: Make, unusable: object) -> None:
    model = scripted(tools=[unusable, unusable], extract=[FALLBACK])
    mem = agentic(model)
    receipt = mem.add(MODEL_TURN)
    assert receipt.agentic_fallback == "malformed"
    assert model.counts() == {"run_tools": 2, "extract": 1}
    assert receipt.llm_calls == model.count() == 3
    assert model_claims(mem) == ["Porto"]


def test_a_rate_limited_request_is_retried_once(scripted: Make) -> None:
    once = scripted(tools=[RateLimitError(), Answer("done")])
    receipt = agentic(once).add(MODEL_TURN)
    assert receipt.agentic_fallback is None
    assert receipt.llm_calls == once.count() == 2
    twice = scripted(tools=[RateLimitError(), RateLimitError()], extract=[FALLBACK])
    receipt = agentic(twice).add(MODEL_TURN)
    assert receipt.agentic_fallback == "error"
    assert receipt.llm_calls == twice.count() == 3


def test_a_provider_timeout_is_not_retried(scripted: Make) -> None:
    """`run_loop` raises a timeout at once: "the time a retry would need is the time
    that has run out" (`ToolRunTimeout`)."""
    model = scripted(tools=[APITimeoutError()], extract=[FALLBACK])
    receipt = agentic(model).add(MODEL_TURN)
    assert receipt.agentic_fallback == "timeout"
    assert model.count("run_tools") == 1
    assert receipt.llm_calls == model.count() == 2


def test_a_slow_provider_runs_out_the_twenty_five_seconds_of_a_write(scripted: Make) -> None:
    """Each answer takes 10 seconds on the model's clock. The third request starts with 5
    seconds left and the fourth is never sent."""
    model = scripted(tools=[Forever(Late(Answer(calls=(SEARCH,)), 10.0))],
                     extract=[FALLBACK])
    receipt = agentic(model).add(MODEL_TURN)
    assert receipt.agentic_fallback == "timeout"
    assert model.count("run_tools") == 3
    assert receipt.llm_calls == model.count() == 4


def test_the_same_slow_provider_runs_to_the_step_limit_in_reextract(scripted: Make) -> None:
    """`reextract()` is run by a background worker, so its budget is 180 seconds: twelve
    answers of 10 seconds each fit, and the run ends at the step limit instead."""
    model = scripted(tools=[APITimeoutError()], extract=[[]])
    mem = agentic(model)
    first = mem.add(MODEL_TURN)
    assert first.agentic_fallback == "timeout" and first.unextracted == 1
    model.queue("run_tools", Forever(Late(Answer(calls=(SEARCH,)), 10.0)))
    model.queue("extract", FALLBACK)
    swept = mem.reextract()
    assert swept.agentic_fallback == "step_limit"
    assert model.runs[-1]["timeout"] == 180.0
    assert swept.llm_calls == 12 + 1
    assert model_claims(mem) == ["Porto"]


def test_an_unusable_k_in_a_search_is_clamped_or_defaulted(scripted: Make) -> None:
    """Review Focus 5 of the plan: `search_memories` clamps `k` to between 1 and 20, and
    reads a `k` that is not an integer as 8, so no value of it fails the run."""
    searches = tuple(("search_memories", {"query": "Berlin", "k": k})
                     for k in (0, -5, 10_000, "ten", True))
    model = scripted(tools=[Answer(calls=searches), Answer("done")])
    mem = agentic(model)
    seed(mem)
    receipt = mem.add(MODEL_TURN)
    assert receipt.agentic_fallback is None
    assert len(model.tool_results) == 5
    assert all(r.startswith("Stored memories.") for r in model.tool_results)
    assert [r.count("\n- claim_id=") for r in model.tool_results[:2]] == [1, 1]
    assert receipt.llm_calls == model.count() == 2
