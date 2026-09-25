"""The scripted model's own tests, and the tests of the helpers that judge what a write did.

Every other test in this folder trusts two things: that the scripted model behaves like a
shipped backend in every way memvara can observe, and that `handles.fates` sees a claim
being ended, retired or erased. Both are checked here before anything relies on them.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

from memvara.llm import LLM, Chat, ReplacementJudge, ToolChat
from memvara.llm import _tools
from memvara.llm.base import (
    MalformedToolOutput, Message, ToolRun, ToolRunTimeout, ToolSpec, TruncatedResponse,
)
from memvara.store import SQLiteStore

from .handles import fates, ledger, seed, with_model, without_model
from .scripted import (
    APIConnectionError, APIError, APIStatusError, APITimeoutError, Answer,
    AuthenticationError, Call, Forever, Late, RateLimitError, ScriptedModel, Text,
    Truncated, Unscripted, check_scripts,
)

Make = Callable[..., ScriptedModel]

CHAT = {"json_object": True, "max_completion_tokens": 300, "timeout": 10.0}


def chat(model: ScriptedModel) -> Any:
    return model.chat("system", "prompt", **CHAT)


def tool(seen: list[Any]) -> ToolSpec:
    """A tool named `t` that records what it was called with."""
    def handler(arguments: dict[str, Any]) -> str:
        seen.append(arguments)
        return f"t answered {arguments}"
    return ToolSpec("t", "a tool", {"type": "object", "properties": {}, "required": [],
                                    "additionalProperties": False}, handler)


def run(model: ScriptedModel, seen: list[Any], *, max_steps: int = 12) -> ToolRun:
    return model.run_tools("system", [Message("user", "turns")], [tool(seen)],
                           max_steps=max_steps, timeout=25.0)


# -- the model is shaped like a shipped backend ----------------------------------------


def test_the_model_implements_every_protocol_a_shipped_backend_does(scripted: Make) -> None:
    model = scripted()
    assert isinstance(model, LLM) and isinstance(model, Chat)
    assert isinstance(model, ToolChat) and isinstance(model, ReplacementJudge)
    # A no-op backend is not billed, so a model that said it was one would make every
    # call-count assertion in this folder pass for the wrong reason.
    assert model.is_noop is False


def test_each_method_answers_from_its_own_script_in_order_and_records_the_call(
        scripted: Make) -> None:
    model = scripted(extract=[[{"a": 1}], [{"b": 2}]], chat=[Text("first")])
    assert model.extract([], ["lives_in"]) == [{"a": 1}]
    assert chat(model) == "first"
    assert model.extract([], []) == [{"b": 2}]
    assert [call.method for call in model.calls] == ["extract", "chat", "extract"]
    assert model.calls[0].args["known_predicates"] == ["lives_in"]
    assert model.counts() == {"extract": 2, "chat": 1}
    assert model.count() == 3 and model.count("chat") == 1


def test_extract_records_the_text_of_the_turns_it_was_shown(scripted: Make) -> None:
    from memvara.types import Episode, Scope
    model = scripted(extract=[[]])
    model.extract([Episode(content="one", scope=Scope("t", "u")),
                   Episode(content="two", scope=Scope("t", "u"))], [])
    assert model.calls[0].args["turns"] == ["one", "two"]


def test_a_call_past_the_script_raises_and_fails_the_check() -> None:
    model = ScriptedModel(chat=[Text("only one")])
    chat(model)
    with pytest.raises(Unscripted):
        chat(model)
    assert model.count() == 1
    assert [call.method for call in model.unscripted] == ["chat"]
    with pytest.raises(AssertionError, match="chat"):
        check_scripts([model])
    check_scripts([ScriptedModel()])  # a model nothing called is fine


# -- replies are shaped the way a shipped backend shapes them --------------------------


def test_a_text_reply_to_extract_goes_through_the_backends_own_validation(
        scripted: Make) -> None:
    """The confidence is clamped, the predicate snake-cased, the time trimmed, and a
    claim naming a turn that does not exist dropped. A fake that returned the parsed JSON
    as it was would fail every one of those."""
    reply = ('{"claims": [{"subject": "team", "predicate": "Office City", '
             '"object": "Porto", "polarity": 1, "memory_type": "semantic", '
             '"confidence": 1.7, "source_index": 0, "when": " last summer ", '
             '"amount": null, "unit": null}, {"subject": "team", "predicate": "x", '
             '"object": "y", "polarity": 1, "memory_type": "semantic", '
             '"confidence": 0.5, "source_index": 9, "when": null, "amount": null, '
             '"unit": null}]}')
    from memvara.types import Episode, Scope
    model = scripted(extract=[Text(reply), Text("not json")])
    turn = [Episode(content="The office moved to Porto.", scope=Scope("t", "u"))]
    assert model.extract(turn, []) == [{
        "subject": "team", "predicate": "office_city", "object": "Porto", "polarity": 1,
        "memory_type": "semantic", "confidence": 1.0, "source_index": 0,
        "when": "last summer", "amount": None, "unit": None}]
    assert model.extract(turn, []) == []


def test_text_replies_to_the_other_schema_methods_are_validated_too(scripted: Make) -> None:
    model = scripted(
        resolve=[Text('{"canonical": "made_up_slot", "cardinality": "one", '
                      '"volatility": "fast", "memory_type": "episodic"}'),
                 Text('{"canonical": "Lives In", "cardinality": "many", '
                      '"volatility": "slow", "memory_type": "semantic"}')],
        classify=[Text('{"cardinality": "sideways"}')],
        judge=[Text('{"same_thing": true, "same_property": true, "newer_value": false, '
                    '"replaces": true}')])
    # A canonical name that was not offered is read as "this is new".
    assert model.resolve_predicate("office_city", ["lives_in", "works_at"]) == {
        "canonical": None, "cardinality": "one", "volatility": "fast",
        "memory_type": "episodic"}
    assert model.resolve_predicate("home_city", ["lives_in"])["canonical"] == "lives_in"
    assert model.calls[0].args["candidates"] == ["lives_in", "works_at"]
    assert model.classify_predicate("office_city", "example") == {
        "cardinality": "many", "volatility": "slow", "memory_type": "semantic"}
    # A verdict that denies one of its own premises is not believed.
    assert model.judge_replacement("new", "old") == {
        "same_thing": True, "same_property": True, "newer_value": False,
        "replaces": False}


def test_a_truncated_reply_raises_what_a_shipped_backend_raises(scripted: Make) -> None:
    model = scripted(extract=[Truncated('{"claims": [')], resolve=[Truncated()],
                     classify=[Truncated()], judge=[Truncated()],
                     chat=[Truncated('{"queries": ["Lis')])
    with pytest.raises(TruncatedResponse):
        model.extract([], [])
    with pytest.raises(TruncatedResponse):
        model.resolve_predicate("x", [])
    with pytest.raises(TruncatedResponse):
        model.classify_predicate("x", "y")
    with pytest.raises(TruncatedResponse):
        model.judge_replacement("new", "old")
    # Neither backend checks why a chat reply stopped, so the cut text arrives.
    assert chat(model) == '{"queries": ["Lis'


# -- provider errors and time -----------------------------------------------------------


def test_an_exception_in_a_script_is_raised(scripted: Make) -> None:
    model = scripted(chat=[RateLimitError()], extract=[AuthenticationError()])
    with pytest.raises(RateLimitError):
        chat(model)
    with pytest.raises(AuthenticationError):
        model.extract([], [])
    assert model.count() == 2  # a call that raised was still answered, and billed


def test_the_provider_errors_carry_what_memvara_reads_from_the_sdk_ones() -> None:
    assert RateLimitError().status_code == 429
    assert AuthenticationError().status_code == 401
    assert APIStatusError("unavailable", status_code=503).status_code == 503
    timeout = APITimeoutError()
    # The SDKs' timeout is not Python's, and has no status: memvara's tool loop knows it
    # by its name, and a read stage knows it only by the clock.
    assert not isinstance(timeout, TimeoutError)
    assert not hasattr(timeout, "status_code")
    assert _tools.is_timeout(timeout)
    assert not _tools.is_timeout(RateLimitError())
    for error in (timeout, RateLimitError(), AuthenticationError(),
                  APIConnectionError("connection reset"),
                  APIStatusError("unavailable", status_code=503)):
        assert isinstance(error, APIError)
    assert str(timeout) == "Request timed out."


def test_a_late_reply_moves_the_models_clock_and_nothing_else_does(scripted: Make) -> None:
    model = scripted(chat=[Text("on time"), Late(Text("late"), 11.0), Text("again")])
    assert model.clock() == 0.0
    assert chat(model) == "on time" and model.clock() == 0.0
    assert chat(model) == "late" and model.clock() == 11.0
    assert chat(model) == "again" and model.clock() == 11.0


def test_a_late_exception_moves_the_clock_before_it_is_raised(scripted: Make) -> None:
    model = scripted(chat=[Late(APITimeoutError(), 10.5)])
    with pytest.raises(APITimeoutError):
        chat(model)
    assert model.clock() == 10.5


def test_forever_answers_every_later_call(scripted: Make) -> None:
    model = scripted(chat=[Forever(Text("again"))])
    assert [chat(model) for _ in range(3)] == ["again", "again", "again"]
    assert model.unscripted == []


# -- run_tools is memvara's own tool loop ------------------------------------------------


def test_run_tools_runs_each_tool_call_through_its_handler(scripted: Make) -> None:
    seen: list[Any] = []
    model = scripted(tools=[Answer(calls=(("t", {"a": 1}),)), Answer("done")])
    result = run(model, seen, max_steps=5)
    assert result == ToolRun(steps=2, requests=2, finished=True, text="done")
    assert seen == [{"a": 1}]
    assert model.tool_results == ["t answered {'a': 1}"]
    assert model.runs == [{"system": "system", "messages": [Message("user", "turns")],
                           "tools": ["t"], "max_steps": 5, "timeout": 25.0}]
    assert model.count("run_tools") == 2  # one call recorded per provider request


def test_a_model_that_never_stops_is_stopped_by_the_step_limit_it_is_given(
        scripted: Make) -> None:
    seen: list[Any] = []
    model = scripted(tools=[Forever(Answer(calls=(("t", {}),)))])
    assert run(model, seen, max_steps=12) == ToolRun(steps=12, requests=12,
                                                     finished=False, text="")
    assert len(seen) == 12


def test_an_answer_that_cannot_be_used_is_retried_once(scripted: Make) -> None:
    seen: list[Any] = []
    once = scripted(tools=[Truncated(), Answer("done")])
    assert run(once, seen) == ToolRun(steps=1, requests=2, finished=True, text="done")
    twice = scripted(tools=[Truncated(), Truncated()])
    with pytest.raises(MalformedToolOutput) as caught:
        run(twice, seen)
    assert caught.value.requests == 2


def test_a_slow_answer_spends_the_runs_time_on_the_models_clock(scripted: Make) -> None:
    """Three answers of 10 seconds each: the third starts with 5 of the 25 seconds left,
    and the fourth is never sent."""
    seen: list[Any] = []
    model = scripted(tools=[Forever(Late(Answer(calls=(("t", {}),)), 10.0))])
    with pytest.raises(ToolRunTimeout) as caught:
        run(model, seen)
    assert caught.value.requests == 3 and model.clock() == 30.0


def test_a_tool_script_that_cannot_be_sent_is_refused_when_it_is_written() -> None:
    with pytest.raises(TypeError, match="run_tools cannot send"):
        ScriptedModel(tools=[{"not": "an answer"}])
    with pytest.raises(ValueError, match="not one of"):
        ScriptedModel().queue("summarise", Text("x"))


# -- the helpers that judge a write -----------------------------------------------------


def test_fates_name_every_way_a_claim_can_leave_the_live_set(scripted: Make) -> None:
    """The check every later test relies on for "nothing retired or erased" is shown here
    to see each of the three, so it cannot pass by seeing none of them."""
    mem = with_model(scripted())
    ids = seed(mem)
    ids["speaks"] = mem.remember("user", "speaks", "Portuguese").added[0].id
    before = ledger(mem)
    mem.remember("user", "lives_in", "Lisbon")          # ends Berlin: the world changed
    mem.forget("user", "works_at")                      # retires Acme: it was wrong
    assert mem.erase(ids["tea"])                        # erases green tea: gone
    assert fates(before, mem) == {ids["berlin"]: "ended", ids["acme"]: "retired",
                                  ids["tea"]: "erased", ids["speaks"]: "unchanged"}


def test_the_two_handles_share_one_store_and_only_one_has_a_model(scripted: Make) -> None:
    store = SQLiteStore(":memory:")
    plain = without_model(store)
    plain.remember("user", "lives_in", "Berlin")
    model = scripted()
    mem = with_model(model, store=store)
    assert [c.object for c in mem.get_all()] == ["Berlin"]
    assert mem.llm is model and plain.llm.is_noop
