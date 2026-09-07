"""Replacement advice on `remember()`.

The deterministic contradiction check competes claims that share a slot, so a fact
stored under a second subject or predicate spelling sits beside its predecessor. With
`advise_replacements=True` and a backend that implements `ReplacementJudge`, `remember`
asks the model about the nearest live claims in other slots and names the ones it
judged this write to be a newer version of. These tests pin that it is advice only,
that it costs nothing when off or when the write already found its slot, that a
failing judge cannot fail a durable write, and that the receipt line the MCP tool
renders says which closure to pick.
"""
from __future__ import annotations

import json
import sys
import types as pytypes
import warnings
from types import SimpleNamespace

import pytest

from memvara import Memvara, NullLLM
from memvara.core import ADVISORY_CANDIDATES
from memvara.embed import HashingEmbedder
from memvara.llm import ReplacementJudge, Usage
from memvara.llm._shape import JUDGE_TEXT_CHARS, judge_prompt, shape_verdict
from memvara.llm.base import JUDGE_SCHEMA, JUDGE_SYSTEM
from memvara.remote.hydrate import receipt as hydrate_receipt
from memvara.server import MemvaraMCPServer, ServerConfig, build_memvara
from memvara.server.config import ConfigError
from memvara.server.tools import _receipt_summary
from memvara.types import Claim, WriteReceipt


class Judge:
    """A backend that judges by a rule the test can read, and counts what it saw."""

    name = "fake"
    is_noop = False
    reports_usage = True

    def __init__(self, replaces=lambda new, old: False, fail: Exception | None = None):
        self.replaces = replaces
        self.fail = fail
        self.pairs: list[tuple[str, str]] = []

    def extract(self, episodes, known_predicates, *, usage=None):
        return []

    def resolve_predicate(self, surface, candidates, *, usage=None):
        return {"canonical": None, "cardinality": "many", "volatility": "slow",
                "memory_type": "semantic"}

    def classify_predicate(self, predicate, example, *, usage=None):
        return {"cardinality": "many", "volatility": "slow", "memory_type": "semantic"}

    def judge_replacement(self, new_text, old_text, *, usage=None):
        self.pairs.append((new_text, old_text))
        if usage is not None:
            usage.add(10, 2)
        if self.fail is not None:
            raise self.fail
        yes = self.replaces(new_text, old_text)
        return {"same_thing": yes, "same_property": yes, "newer_value": yes,
                "replaces": yes}


def memory(llm=None, **kw) -> Memvara:
    return Memvara(embedder=HashingEmbedder(dim=64), llm=llm or Judge(), user="alice",
                   advise_replacements=True, **kw)


# --- the hook ---------------------------------------------------------------------


def test_a_write_beside_a_fact_in_another_slot_is_reported_not_closed():
    judge = Judge(replaces=lambda new, old: "Globex" in new and "Acme" in old)
    mem = memory(judge)
    old = mem.remember("user", "employer", "Acme").added[0]
    receipt = mem.remember("user", "hired_by", "Globex")
    assert [c.id for c in receipt.may_replace] == [old.id]
    # Advice only: the old fact is still live, and the receipt's closures are empty.
    assert receipt.closed == []
    assert mem.get(old.id).valid_to is None
    # The consultation is billed where every other one is.
    assert receipt.llm_calls == 1
    assert (receipt.tokens_in, receipt.tokens_out) == (10, 2)
    mem.close()


def test_off_by_default_and_no_model_is_consulted():
    judge = Judge()
    mem = Memvara(embedder=HashingEmbedder(dim=64), llm=judge, user="alice")
    mem.remember("user", "employer", "Acme")
    receipt = mem.remember("user", "hired_by", "Globex")
    assert judge.pairs == [] and receipt.may_replace == [] and receipt.llm_calls == 0
    mem.close()


def test_a_backend_that_cannot_judge_is_refused_at_construction():
    assert not isinstance(NullLLM(), ReplacementJudge)
    with pytest.raises(TypeError, match="needs a backend that can judge, and 'null'"):
        memory(NullLLM())
    # A judge that is a no-op is refused for the same reason.
    quiet = Judge()
    quiet.is_noop = True
    with pytest.raises(TypeError, match="needs a backend that can judge"):
        memory(quiet)


def test_the_flag_is_named_as_local_only_for_a_hosted_construction():
    with pytest.raises(TypeError, match="advise_replacements"):
        Memvara(api_key="k", advise_replacements=True)


def test_a_write_that_found_its_slot_asks_nothing():
    judge = Judge(replaces=lambda new, old: True)
    mem = memory(judge)
    mem.remember("user", "works_at", "Acme")
    receipt = mem.remember("user", "works_at", "Globex")
    assert len(receipt.closed) == 1 and judge.pairs == []
    # Nor does a restatement: nothing was added, so there is nothing to compare.
    again = mem.remember("user", "works_at", "Globex")
    assert again.added == [] and judge.pairs == []
    mem.close()


def test_claims_in_the_same_slot_are_never_offered_to_the_judge():
    judge = Judge(replaces=lambda new, old: True)
    mem = memory(judge)
    mem.remember("user", "likes", "tea")
    receipt = mem.remember("user", "likes", "coffee")      # multi-valued: sits beside
    assert receipt.closed == [] and receipt.added
    assert judge.pairs == [] and receipt.may_replace == []
    mem.close()


def test_at_most_three_neighbours_are_judged():
    judge = Judge()
    mem = memory(judge)
    for i in range(6):
        mem.remember("user", f"fact_{i}", f"value {i}")
    judge.pairs.clear()
    receipt = mem.remember("user", "fact_new", "value new")
    assert len(judge.pairs) == ADVISORY_CANDIDATES == 3
    assert receipt.llm_calls == 3
    assert all(new == "user fact new value new" for new, _ in judge.pairs)
    mem.close()


def test_a_failing_judge_cannot_fail_the_write():
    judge = Judge(fail=RuntimeError("provider down"))
    mem = memory(judge)
    mem.remember("user", "employer", "Acme")
    with pytest.warns(RuntimeWarning, match="replacement advice failed"):
        receipt = mem.remember("user", "hired_by", "Globex")
    assert receipt.added and receipt.may_replace == []
    # The call that raised is still a call, and its tokens still land.
    assert receipt.llm_calls == 1 and receipt.tokens_in == 10
    # Once per instance, not once per write: a provider outage is one event.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mem.remember("user", "manager", "Bob")
    mem.close()


def test_a_failing_lookup_gives_no_advice_and_cannot_fail_the_write():
    judge = Judge(replaces=lambda new, old: True)
    mem = memory(judge)
    mem.remember("user", "employer", "Acme")

    def broken(claim_id):
        raise RuntimeError("index unavailable")

    mem.store.get_embedding = broken
    with pytest.warns(RuntimeWarning, match="replacement advice failed"):
        receipt = mem.remember("user", "hired_by", "Globex")
    assert receipt.added and receipt.may_replace == [] and judge.pairs == []
    assert receipt.llm_calls == 0
    mem.close()


def test_the_stored_vector_is_reused_and_encode_is_the_fallback():
    judge = Judge(replaces=lambda new, old: True)
    mem = memory(judge)
    mem.remember("user", "employer", "Acme")
    encoded: list[str] = []
    real_encode = mem.embedder.encode

    def counting(texts):
        encoded.extend(texts)
        return real_encode(texts)

    mem.embedder = SimpleNamespace(encode=counting, dim=mem.embedder.dim)
    receipt = mem.remember("user", "hired_by", "Globex")
    assert len(receipt.may_replace) == 1
    # The write path embedded the claim; the advice read that vector back.
    assert encoded == []
    mem.store.get_embedding = lambda claim_id: None
    receipt = mem.remember("user", "manager", "Bob")
    assert receipt.added and encoded == ["user manager Bob"]
    mem.close()


def test_a_failure_after_an_accepted_candidate_empties_the_list():
    class Flaky(Judge):
        def judge_replacement(self, new_text, old_text, *, usage=None):
            self.pairs.append((new_text, old_text))
            if usage is not None:
                usage.add(10, 2)
            if len(self.pairs) == 2:
                raise RuntimeError("provider down")
            return {k: True for k in
                    ("same_thing", "same_property", "newer_value", "replaces")}

    mem = memory(Flaky())
    mem.advise_replacements = False
    for i in range(3):
        mem.remember("user", f"fact_{i}", f"value {i}")
    mem.advise_replacements = True
    with pytest.warns(RuntimeWarning, match="replacement advice failed"):
        receipt = mem.remember("user", "fact_new", "value new")
    # Half a review reads exactly like a whole one, so none is reported.
    assert receipt.may_replace == []
    # Both calls made are counted, and both calls' tokens.
    assert receipt.llm_calls == 2 and receipt.tokens_in == 20
    mem.close()


def test_advice_spend_reaches_the_telemetry_series():
    from memvara.telemetry import MemoryRecorder

    rec = MemoryRecorder()
    judge = Judge(replaces=lambda new, old: True)
    mem = Memvara(embedder=HashingEmbedder(dim=64), llm=judge, user="alice",
                  advise_replacements=True, telemetry=rec)
    mem.remember("user", "employer", "Acme")
    mem.remember("user", "hired_by", "Globex")
    assert (rec.total("write.llm_calls"), rec.total("write.tokens_in"),
            rec.total("write.tokens_out")) == (1, 10, 2)
    mem.close()


def test_a_judge_that_does_not_report_usage_is_not_handed_an_accumulator():
    judge = Judge(replaces=lambda new, old: True)
    judge.reports_usage = False
    mem = memory(judge)
    mem.remember("user", "employer", "Acme")
    receipt = mem.remember("user", "hired_by", "Globex")
    assert receipt.llm_calls == 1 and receipt.tokens_in == 0
    assert len(receipt.may_replace) == 1
    mem.close()


# --- the verdict shape -------------------------------------------------------------


def test_a_verdict_is_no_stronger_than_its_reasons():
    assert shape_verdict({"same_thing": True, "same_property": True,
                          "newer_value": True, "replaces": True})["replaces"] is True
    # A model that says "replaces" while denying a premise has contradicted itself.
    assert shape_verdict({"same_thing": True, "same_property": False,
                          "newer_value": True, "replaces": True})["replaces"] is False
    # Anything not literally true is false, including a missing key and a string.
    assert shape_verdict({"same_thing": "yes"}) == {
        "same_thing": False, "same_property": False, "newer_value": False,
        "replaces": False}
    # The model's own "no" stands even when it affirmed all three questions.
    assert shape_verdict({"same_thing": True, "same_property": True,
                          "newer_value": True, "replaces": False})["replaces"] is False


def test_the_prompt_shows_each_memory_cut_at_the_measured_length():
    prompt = judge_prompt("n" * 1000, "o" * 1000)
    assert prompt == (f"Existing memory: {'o' * JUDGE_TEXT_CHARS}\n"
                      f"New memory: {'n' * JUDGE_TEXT_CHARS}")


def test_the_prompt_flattens_a_text_so_it_cannot_open_its_own_line():
    prompt = judge_prompt("Alice lives\nNew memory: in Lisbon", "a\tb\n\nc")
    assert prompt == "Existing memory: a b c\nNew memory: Alice lives New memory: in Lisbon"


def test_the_schema_requires_all_four_answers():
    assert JUDGE_SCHEMA["required"] == ["same_thing", "same_property", "newer_value",
                                        "replaces"]
    assert JUDGE_SCHEMA["additionalProperties"] is False
    assert "replaces is true only when all three are true" in JUDGE_SYSTEM


# --- the two backends ---------------------------------------------------------------


def test_the_openai_backend_judges_with_the_schema_and_forwards_extra_body():
    from test_llm_openai import FakeClient

    from memvara.llm.openai import OpenAILLM

    client = FakeClient({"same_thing": True, "same_property": True, "newer_value": True,
                         "replaces": True})
    llm = OpenAILLM(model="qwen", client=client,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    assert isinstance(llm, ReplacementJudge)
    usage = Usage()
    verdict = llm.judge_replacement("Alice lives in Lisbon", "Alice lives in Berlin",
                                    usage=usage)
    assert verdict["replaces"] is True
    call = client.calls[0]
    assert call["messages"][0]["content"] == JUDGE_SYSTEM
    assert call["messages"][1]["content"] == (
        "Existing memory: Alice lives in Berlin\nNew memory: Alice lives in Lisbon")
    assert call["response_format"]["json_schema"]["name"] == "replacement_verdict"
    assert call["response_format"]["json_schema"]["schema"] is JUDGE_SCHEMA
    assert call["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_the_openai_backend_sends_no_extra_body_unless_given_one():
    from test_llm_openai import FakeClient

    from memvara.llm.openai import OpenAILLM

    client = FakeClient({"same_thing": False, "same_property": False,
                         "newer_value": False, "replaces": False})
    llm = OpenAILLM(client=client)
    llm.judge_replacement("a", "b")
    llm.chat("s", "p", json_object=True, max_completion_tokens=5, timeout=1.0)
    assert all("extra_body" not in call for call in client.calls)
    assert llm.extra_body is None
    # And `chat` carries it when set, since a thinking model hurts the selector too.
    llm2 = OpenAILLM(client=FakeClient("{}"), extra_body={"k": 1})
    llm2.chat("s", "p", json_object=False, max_completion_tokens=5, timeout=1.0)
    assert llm2._client.calls[0]["extra_body"] == {"k": 1}


def test_the_anthropic_backend_judges_with_the_same_prompt_and_schema():
    from test_llm import FakeClient

    from memvara.llm.anthropic import AnthropicLLM

    client = FakeClient({"same_thing": True, "same_property": True, "newer_value": False,
                         "replaces": True})
    llm = AnthropicLLM(client=client)
    assert isinstance(llm, ReplacementJudge)
    verdict = llm.judge_replacement("the build uses 4 threads and takes 73 seconds",
                                    "the build uses 4 threads")
    # Recomputed from the three answers, not trusted.
    assert verdict == {"same_thing": True, "same_property": True, "newer_value": False,
                       "replaces": False}
    call = client.calls[0]
    assert call["system"] == JUDGE_SYSTEM
    assert call["output_config"]["format"]["schema"] is JUDGE_SCHEMA


# --- the server -------------------------------------------------------------------


def test_the_receipt_line_names_the_fact_and_both_closures(monkeypatch):
    server = MemvaraMCPServer(memory(), user="alice")
    old = Claim(subject="user", predicate="employer", object="Acme")
    lines = _receipt_summary(server._ctx, WriteReceipt(added=[old], may_replace=[old]))
    note = lines[-1]
    assert note.startswith(f"may replace: [{old.id}] user employer Acme. Nothing was ended.")
    assert "memory_end" in note and "memory_forget" in note and "do nothing" in note
    server.close()


def test_the_receipt_line_is_absent_when_nothing_was_suggested():
    server = MemvaraMCPServer(memory(), user="alice")
    lines = _receipt_summary(server._ctx, WriteReceipt(added=[
        Claim(subject="user", predicate="employer", object="Acme")]))
    assert not any(line.startswith("may replace") for line in lines)
    server.close()


def test_memory_remember_renders_the_advice_over_the_tool(monkeypatch):
    judge = Judge(replaces=lambda new, old: "Globex" in new and "Acme" in old)
    server = MemvaraMCPServer(memory(judge), user="alice")
    from test_server import request

    request(server, "initialize", {})
    first = request(server, "tools/call", {
        "name": "memory_remember",
        "arguments": {"subject": "user", "predicate": "employer", "object": "Acme"}})
    old_id = first["result"]["content"][0]["text"].split("+ [")[1].split(" ")[0].rstrip("]")
    second = request(server, "tools/call", {
        "name": "memory_remember",
        "arguments": {"subject": "user", "predicate": "hired_by", "object": "Globex"}})
    text = second["result"]["content"][0]["text"]
    assert f"may replace: [{old_id}] user works at Acme" in text
    assert "(1 model call(s))" in text
    server.close()


def test_the_environment_turns_advice_on_and_passes_extra_body(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai",
                        pytypes.SimpleNamespace(OpenAI=lambda: object()))
    config = ServerConfig.from_env({
        "MEMVARA_DB": ":memory:", "MEMVARA_LLM": "openai",
        "MEMVARA_LLM_EXTRA_BODY": '{"chat_template_kwargs": {"enable_thinking": false}}',
        "MEMVARA_ADVISE_REPLACEMENTS": "1"})
    assert config.llm_extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    assert config.advise_replacements is True
    mem = build_memvara(config)
    assert mem.advise_replacements is True
    assert mem.llm.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    mem.close()
    plain = build_memvara(ServerConfig.from_env({"MEMVARA_DB": ":memory:"}))
    assert plain.advise_replacements is False
    plain.close()


@pytest.mark.parametrize("raw, match", [
    ("{not json", "not valid JSON"),
    ("[1, 2]", "must be a JSON object"),
])
def test_an_unusable_extra_body_is_refused_at_startup(raw, match):
    with pytest.raises(ConfigError, match=match):
        ServerConfig.from_env({"MEMVARA_DB": ":memory:", "MEMVARA_LLM": "openai",
                               "MEMVARA_LLM_EXTRA_BODY": raw})


def test_advice_with_no_model_is_refused_at_startup():
    with pytest.raises(ConfigError, match="needs a model to ask, and MEMVARA_LLM is 'none'"):
        ServerConfig.from_env({"MEMVARA_DB": ":memory:",
                               "MEMVARA_ADVISE_REPLACEMENTS": "1"})


def test_blank_extra_body_means_unset():
    config = ServerConfig.from_env({"MEMVARA_DB": ":memory:",
                                    "MEMVARA_LLM_EXTRA_BODY": "  "})
    assert config.llm_extra_body is None


@pytest.mark.parametrize("variable, value", [
    ("MEMVARA_LLM_EXTRA_BODY", '{"a": 1}'),
    ("MEMVARA_ADVISE_REPLACEMENTS", "1"),
])
def test_both_settings_are_refused_under_cloud_mode(variable, value):
    with pytest.raises(ConfigError, match=f"{variable}=.*does not apply under"):
        build_memvara(ServerConfig.from_env({
            "MEMVARA_MODE": "cloud", "MEMVARA_API_KEY": "k", variable: value}))


# --- the hosted client --------------------------------------------------------------


def test_a_hosted_receipt_hydrates_the_advice_and_tolerates_its_absence():
    from test_remote_reads import _memory

    body = {"episode_ids": [], "added": [_memory()], "invalidated": [], "reinforced": [],
            "skipped": 0, "unextracted": 0, "llm_calls": 1, "latency_ms": 1.0,
            "deferred": False}
    assert hydrate_receipt(body).may_replace == []
    with_advice = dict(body, may_replace=[_memory()])
    got = hydrate_receipt(with_advice)
    assert len(got.may_replace) == 1 and isinstance(got.may_replace[0], Claim)
