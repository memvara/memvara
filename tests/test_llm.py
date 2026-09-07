"""`AnthropicLLM` request shape and output validation.

Everything here runs offline against an injected fake client: no API key, no network,
and the `anthropic` package is not installed.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from memvara.llm import LLM, AnthropicLLM, NullLLM, TruncatedResponse
from memvara.llm import _shape
from memvara.llm.anthropic import _stop_reason
from memvara.llm.base import (
    CLAIM_SCHEMA,
    EXTRACT_SYSTEM,
    PREDICATE_SCHEMA,
    PREDICATE_SYSTEM,
    RESOLVE_SCHEMA,
    RESOLVE_SYSTEM,
    Usage,
)
from memvara.types import Episode, Scope


class FakeMessages:
    def __init__(self, payloads: list[object], stop_reason: str = "end_turn") -> None:
        self.payloads = payloads
        # A model that finished its answer, which is what every test here but the
        # truncation ones is about.
        self.stop_reason = stop_reason
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        payload = self.payloads[min(len(self.calls) - 1, len(self.payloads) - 1)]
        if isinstance(payload, str):
            blocks = [SimpleNamespace(type="text", text=payload)]
        elif payload is None:
            blocks = []
        else:
            blocks = [SimpleNamespace(type="text", text=json.dumps(payload))]
        return SimpleNamespace(content=blocks, stop_reason=self.stop_reason)


class FakeClient:
    """Stands in for `anthropic.Anthropic()` and records every request verbatim."""

    def __init__(self, *payloads: object, stop_reason: str = "end_turn") -> None:
        self.messages = FakeMessages(list(payloads) or [{"claims": []}], stop_reason)

    @property
    def calls(self) -> list[dict]:
        return self.messages.calls


def episodes(*contents: str) -> list[Episode]:
    return [Episode(content=c, scope=Scope(tenant="acme")) for c in contents]


def claim(**overrides) -> dict:
    base = {
        "subject": "user",
        "predicate": "lives_in",
        "object": "lisbon",
        "polarity": 1,
        "memory_type": "semantic",
        "confidence": 0.9,
        "source_index": 0,
        # `shape_claims` emits these for every claim, `None` when the turn stated no time
        # and measured nothing — which is the common case and what most of these tests
        # assert. They are present rather than omitted so that a shaped claim has one
        # shape, and a reader never has to ask whether a missing key means "no value" or
        # "an older extractor".
        "when": None,
        "amount": None,
        "unit": None,
    }
    base.update(overrides)
    return base


def extract_once(payload, *, eps=None):
    client = FakeClient(payload)
    llm = AnthropicLLM(client=client)
    return llm.extract(eps or episodes("I live in Lisbon"), ["lives_in"]), client


# -- protocol / wiring ------------------------------------------------------


def test_satisfies_the_llm_protocol():
    assert isinstance(AnthropicLLM(client=FakeClient()), LLM)
    assert AnthropicLLM(client=FakeClient(), model="claude-opus-5").name == (
        "anthropic/claude-opus-5"
    )


def test_only_the_null_backend_advertises_itself_as_a_no_op():
    """`llm_calls` is billed off this flag, so a real backend claiming it would report
    a write path that costs nothing while spending money on every turn."""
    assert NullLLM.is_noop is True
    assert AnthropicLLM(client=FakeClient()).is_noop is False


def test_the_null_backend_merges_nothing():
    """No model means no evidence that two spellings are the same question."""
    assert NullLLM().resolve_predicate("employer_name", ["works_at"]) == {
        "canonical": None, "cardinality": "many", "volatility": "slow",
        "memory_type": "semantic",
    }


def test_exported_from_the_package_without_the_sdk_installed():
    import memvara.llm as pkg

    assert pkg.NullLLM is NullLLM
    assert set(pkg.__all__) == {
        "LLM", "Chat", "NullLLM", "TruncatedResponse", "Usage", "AnthropicLLM",
        "OpenAILLM"}
    with pytest.raises(AttributeError):
        pkg.NotAThing


def test_missing_sdk_raises_a_clear_install_hint():
    """The SDK is genuinely absent here, which is the configuration this must survive."""
    with pytest.raises(ImportError, match="pip install"):
        AnthropicLLM()


# -- request shape ----------------------------------------------------------


def test_extract_request_carries_schema_and_effort():
    _, client = extract_once({"claims": []})
    assert len(client.calls) == 1
    kwargs = client.calls[0]

    assert kwargs["model"] == "claude-opus-5"
    assert kwargs["max_tokens"] == 8192
    assert kwargs["system"] is EXTRACT_SYSTEM
    assert kwargs["output_config"]["effort"] == "low"
    assert kwargs["output_config"]["format"] == {
        "type": "json_schema",
        "schema": CLAIM_SCHEMA,
    }
    assert kwargs["messages"][0]["role"] == "user"


@pytest.mark.parametrize("banned", ["temperature", "top_p", "top_k", "output_format"])
def test_rejected_parameters_are_never_sent(banned):
    """`temperature`/`top_p`/`top_k` are a 400 on this model; `output_format` is the
    deprecated spelling of `output_config.format`."""
    client = FakeClient({"claims": []}, {"cardinality": "one", "volatility": "slow",
                                         "memory_type": "semantic"})
    llm = AnthropicLLM(client=client)
    llm.extract(episodes("I live in Lisbon"), [])
    llm.classify_predicate("drives", "I drive a Volvo")

    assert len(client.calls) == 2
    for kwargs in client.calls:
        assert banned not in kwargs


def test_adaptive_thinking_is_left_on():
    """Not passing `thinking` at all is what keeps the model's default in force."""
    _, client = extract_once({"claims": []})
    assert "thinking" not in client.calls[0]


def test_effort_and_model_are_configurable():
    client = FakeClient({"claims": []})
    AnthropicLLM(client=client, model="claude-sonnet-5", effort="high",
                 max_tokens=1024).extract(episodes("hi"), [])

    kwargs = client.calls[0]
    assert kwargs["model"] == "claude-sonnet-5"
    assert kwargs["max_tokens"] == 1024
    assert kwargs["output_config"]["effort"] == "high"


def test_prompt_numbers_the_turns_and_lists_known_predicates_deterministically():
    client = FakeClient({"claims": []})
    llm = AnthropicLLM(client=client)
    eps = episodes("I live in Lisbon", "I work at Acme")
    llm.extract(eps, ["works_at", "lives_in", "works_at"])
    llm.extract(eps, ["works_at", "lives_in"])

    prompt = client.calls[0]["messages"][0]["content"]
    assert "[0] user: I live in Lisbon" in prompt
    assert "[1] user: I work at Acme" in prompt
    # Deduped, and in the order the registry supplied rather than re-sorted: sorting
    # would slot each newly learned predicate into the middle of the cacheable prefix.
    assert "works_at, lives_in" in prompt
    # Same predicate set, same bytes - the prompt prefix has to stay cacheable.
    assert client.calls[1]["messages"][0]["content"] == prompt


def test_the_known_predicate_list_is_bounded():
    """Sending the whole vocabulary is an unbounded per-write token tax, and it grows
    fastest exactly when the vocabulary is growing fastest."""
    client = FakeClient({"claims": []})
    vocabulary = [f"predicate_{i:03d}" for i in range(500)]
    AnthropicLLM(client=client).extract(episodes("hi"), vocabulary)

    listed = client.calls[0]["messages"][0]["content"].split("\n")[1].split(", ")
    assert len(listed) == 64
    # The head is preserved, so the prefix a growing vocabulary shares stays identical.
    assert listed == vocabulary[:64]


def test_an_empty_vocabulary_is_stated_rather_than_left_blank():
    client = FakeClient({"claims": []})
    AnthropicLLM(client=client).extract(episodes("hi"), ["", None])
    assert "(none yet)" in client.calls[0]["messages"][0]["content"]


def test_empty_episode_batch_costs_no_call():
    client = FakeClient({"claims": []})
    assert AnthropicLLM(client=client).extract([], ["lives_in"]) == []
    assert client.calls == []


# -- output validation: the trust boundary ----------------------------------


def test_wellformed_claim_passes_through():
    out, _ = extract_once({"claims": [claim()]})
    assert out == [claim()]


@pytest.mark.parametrize(
    "bad_index",
    [None, 1, 7, -1, "0", 0.0, True, [0]],
    ids=["missing", "off_by_one", "far_out", "negative", "string", "float", "bool", "list"],
)
def test_bad_source_index_drops_the_claim(bad_index):
    """Provenance is not guessable: a claim we cannot trace to a turn is unusable."""
    out, _ = extract_once({"claims": [claim(source_index=bad_index)]})
    assert out == []


def test_source_index_maps_onto_the_batch_it_was_given():
    eps = episodes("a", "b", "c")
    out, _ = extract_once({"claims": [claim(source_index=2)]}, eps=eps)
    assert out[0]["source_index"] == 2


@pytest.mark.parametrize(
    ("given", "expected"),
    [(5.0, 1.0), (1.5, 1.0), (-3, 0.0), (1, 1.0), (0.42, 0.42), (0.0, 0.0)],
)
def test_confidence_is_clamped(given, expected):
    out, _ = extract_once({"claims": [claim(confidence=given)]})
    assert out[0]["confidence"] == expected


@pytest.mark.parametrize("given", [None, "high", float("nan"), True, {}])
def test_unreadable_confidence_lands_in_the_middle(given):
    out, _ = extract_once({"claims": [claim(confidence=given)]})
    assert out[0]["confidence"] == 0.5


@pytest.mark.parametrize(
    ("given", "expected"),
    [("Lives In", "lives_in"), ("livesIn", "lives_in"), ("LIVES-IN", "lives_in"),
     ("  works at  ", "works_at"), ("worksAt", "works_at"), ("has/pet", "has_pet")],
)
def test_predicates_are_normalized_to_snake_case(given, expected):
    out, _ = extract_once({"claims": [claim(predicate=given)]})
    assert out[0]["predicate"] == expected


@pytest.mark.parametrize("field", ["subject", "predicate", "object"])
@pytest.mark.parametrize("value", [None, "", "   "])
def test_empty_triple_slots_drop_the_claim(field, value):
    out, _ = extract_once({"claims": [claim(**{field: value})]})
    assert out == []


@pytest.mark.parametrize("given", ["!!!", "___", "-"])
def test_a_predicate_that_normalizes_away_to_nothing_drops_the_claim(given):
    out, _ = extract_once({"claims": [claim(predicate=given)]})
    assert out == []


@pytest.mark.parametrize("field", ["subject", "predicate", "object", "source_index"])
def test_missing_required_keys_drop_the_claim(field):
    partial = claim()
    del partial[field]
    out, _ = extract_once({"claims": [partial]})
    assert out == []


@pytest.mark.parametrize("given", [0, 2, "-1", None, "negative"])
def test_garbled_polarity_is_read_as_an_assertion_not_a_retraction(given):
    """A misread retraction silently invalidates a true fact - fail toward keeping it."""
    out, _ = extract_once({"claims": [claim(polarity=given)]})
    assert out[0]["polarity"] == 1


def test_explicit_retraction_survives():
    out, _ = extract_once({"claims": [claim(polarity=-1)]})
    assert out[0]["polarity"] == -1


@pytest.mark.parametrize("given", ["procedural", "episodic", "semantic"])
def test_valid_memory_types_pass_through(given):
    out, _ = extract_once({"claims": [claim(memory_type=given)]})
    assert out[0]["memory_type"] == given


@pytest.mark.parametrize("given", ["EPISODIC", "fact", "", None, 3])
def test_unknown_memory_type_falls_back_to_semantic(given):
    out, _ = extract_once({"claims": [claim(memory_type=given)]})
    assert out[0]["memory_type"] == "semantic"


def test_one_malformed_entry_does_not_take_the_batch_with_it():
    payload = {
        "claims": [
            claim(object="lisbon"),
            claim(source_index=99, object="mars"),
            "not even an object",
            {"subject": "user"},
            claim(predicate="works_at", object="acme", confidence=8.0),
        ]
    }
    out, _ = extract_once(payload)
    assert [(c["object"], c["confidence"]) for c in out] == [("lisbon", 0.9), ("acme", 1.0)]


@pytest.mark.parametrize(
    "payload",
    ["", "   ", "not json at all", "[1, 2, 3]", '{"claims": "nope"}', '{"claims": null}',
     "{}", '{"claims": {}}', None],
    ids=["empty", "blank", "prose", "top_level_array", "string_claims", "null_claims",
         "no_key", "object_claims", "no_content_blocks"],
)
def test_unusable_responses_yield_nothing_rather_than_raising(payload):
    client = FakeClient(payload)
    assert AnthropicLLM(client=client).extract(episodes("hi"), []) == []
    assert len(client.calls) == 1


def test_text_is_found_past_a_leading_thinking_block():
    """Adaptive thinking is on, so the JSON is rarely the first block in the response."""
    blocks = [
        {"type": "thinking", "thinking": "weighing whether this is durable..."},
        {"type": "text", "text": json.dumps({"claims": [claim()]})},
    ]
    client = SimpleNamespace(
        messages=SimpleNamespace(create=lambda **kw: SimpleNamespace(content=blocks))
    )
    assert AnthropicLLM(client=client).extract(episodes("hi"), []) == [claim()]


# -- resolve_predicate: the acquisition call --------------------------------


def resolve_once(payload, *, surface="employer_name", candidates=("works_at", "mood")):
    client = FakeClient(payload)
    return AnthropicLLM(client=client).resolve_predicate(surface, list(candidates)), client


def test_resolve_predicate_request_shape():
    out, client = resolve_once({"canonical": "works_at", "cardinality": "one",
                                "volatility": "slow", "memory_type": "semantic"})
    kwargs = client.calls[0]
    assert kwargs["system"] is RESOLVE_SYSTEM
    assert kwargs["output_config"]["format"]["schema"] == RESOLVE_SCHEMA
    assert "effort" in kwargs["output_config"]
    prompt = kwargs["messages"][0]["content"]
    assert "employer_name" in prompt and "works_at, mood" in prompt
    assert out["canonical"] == "works_at"


def test_resolution_costs_exactly_one_call():
    _, client = resolve_once({"canonical": None, "cardinality": "many",
                              "volatility": "slow", "memory_type": "semantic"})
    assert len(client.calls) == 1


def test_a_surface_form_the_model_calls_new_stays_new():
    out, _ = resolve_once({"canonical": None, "cardinality": "one",
                           "volatility": "fast", "memory_type": "episodic"})
    assert out == {"canonical": None, "cardinality": "one", "volatility": "fast",
                   "memory_type": "episodic"}


@pytest.mark.parametrize(
    "given",
    ["not_offered", "", None, 7, ["works_at"], {"name": "works_at"}, "   "],
    ids=["invented", "empty", "null", "int", "list", "dict", "blank"],
)
def test_a_canonical_that_was_not_offered_is_read_as_new(given):
    """The riskiest field in the file: `canonical` is echoed into `PredicateSpec.aliases`,
    so a hallucinated name permanently reroutes a surface form into a slot nobody looks
    up. Only a name we offered is accepted; anything else means "new", which is the
    recoverable direction."""
    out, _ = resolve_once({"canonical": given, "cardinality": "many",
                           "volatility": "slow", "memory_type": "semantic"})
    assert out["canonical"] is None


@pytest.mark.parametrize(("given", "expected"),
                         [("Works At", "works_at"), ("worksAt", "works_at"),
                          ("WORKS-AT", "works_at")])
def test_a_canonical_is_matched_after_snake_casing(given, expected):
    out, _ = resolve_once({"canonical": given, "cardinality": "many",
                           "volatility": "slow", "memory_type": "semantic"})
    assert out["canonical"] == expected


@pytest.mark.parametrize(
    "payload",
    [{}, "garbage", {"canonical": "works_at", "cardinality": "several"}, None],
    ids=["empty", "unparseable", "bad_enum", "no_content"],
)
def test_an_unusable_resolution_falls_back_to_the_conservative_default(payload):
    out, _ = resolve_once(payload)
    assert out["cardinality"] == "many"
    assert out["volatility"] == "slow"
    assert out["memory_type"] == "semantic"


def test_the_candidate_list_is_bounded():
    """Same token tax as the extraction vocabulary, and a longer list measurably makes
    the merge decision worse rather than better."""
    candidates = [f"predicate_{i:03d}" for i in range(500)]
    _, client = resolve_once({"canonical": None, "cardinality": "many",
                              "volatility": "slow", "memory_type": "semantic"},
                             candidates=candidates)
    listed = client.calls[0]["messages"][0]["content"].split("\n")[-1].split(", ")
    assert listed == candidates[:48]


def test_resolving_against_an_empty_registry_still_asks():
    out, client = resolve_once({"canonical": None, "cardinality": "many",
                                "volatility": "slow", "memory_type": "semantic"},
                               candidates=())
    assert "(none yet)" in client.calls[0]["messages"][0]["content"]
    assert out["canonical"] is None


# -- classify_predicate -----------------------------------------------------


def test_classify_predicate_request_shape():
    client = FakeClient({"cardinality": "one", "volatility": "fast",
                         "memory_type": "episodic"})
    out = AnthropicLLM(client=client).classify_predicate("Currently Reading", "reading Dune")

    kwargs = client.calls[0]
    assert kwargs["system"] is PREDICATE_SYSTEM
    assert kwargs["output_config"]["format"]["schema"] == PREDICATE_SCHEMA
    assert "effort" in kwargs["output_config"]
    assert "currently_reading" in kwargs["messages"][0]["content"]
    assert out == {"cardinality": "one", "volatility": "fast", "memory_type": "episodic"}


@pytest.mark.parametrize(
    "payload",
    [{}, "garbage", {"cardinality": "several"}, {"cardinality": None},
     {"cardinality": "ONE", "volatility": "slow", "memory_type": "semantic"}],
    ids=["empty", "unparseable", "bad_enum", "null", "wrong_case"],
)
def test_unusable_classification_falls_back_to_the_conservative_default(payload):
    """This answer is cached forever, so a bad value must not become permanent."""
    client = FakeClient(payload)
    assert AnthropicLLM(client=client).classify_predicate("drives", "I drive") == {
        "cardinality": "many",
        "volatility": "slow",
        "memory_type": "semantic",
    }


def test_partially_valid_classification_keeps_the_good_fields():
    client = FakeClient({"cardinality": "one", "volatility": "weekly",
                         "memory_type": "procedural"})
    assert AnthropicLLM(client=client).classify_predicate("prefers_editor", "I use vim") == {
        "cardinality": "one",
        "volatility": "slow",
        "memory_type": "procedural",
    }


def test_classification_costs_exactly_one_call():
    client = FakeClient({"cardinality": "many", "volatility": "slow",
                         "memory_type": "semantic"})
    llm = AnthropicLLM(client=client)
    llm.classify_predicate("drives", "I drive a Volvo")
    assert len(client.calls) == 1


# ===========================================================================
# record_usage — the token counts a bill is computed from
# ===========================================================================

@pytest.mark.parametrize("block", [
    SimpleNamespace(input_tokens=120, output_tokens=8),   # an SDK object
    {"input_tokens": 120, "output_tokens": 8},            # a plain dict
])
def test_usage_is_read_from_an_sdk_object_and_from_a_dict_alike(block):
    # Same tolerance `_first_text` has, for the same reason: a test double should not
    # have to reimplement the SDK's types to be exercised.
    usage = Usage()
    _shape.record_usage(SimpleNamespace(usage=block), usage, "input_tokens",
                        "output_tokens")
    assert (usage.input_tokens, usage.output_tokens, usage.reported) == (120, 8, 1)


def test_usage_accumulates_across_the_calls_of_one_write():
    usage = Usage()
    for _ in range(3):
        _shape.record_usage({"usage": {"input_tokens": 10, "output_tokens": 2}}, usage,
                            "input_tokens", "output_tokens")
    assert (usage.input_tokens, usage.output_tokens, usage.reported) == (30, 6, 3)


def test_a_none_accumulator_is_the_no_usage_wanted_path_and_costs_nothing():
    # What the pipeline passes for a backend that did not advertise `reports_usage`.
    _shape.record_usage({"usage": {"input_tokens": 1, "output_tokens": 1}}, None,
                        "input_tokens", "output_tokens")


@pytest.mark.parametrize("response, why", [
    (SimpleNamespace(), "no usage attribute at all"),
    (SimpleNamespace(usage=None), "usage present but empty"),
    ({"usage": {"output_tokens": 8}}, "input field missing"),
    ({"usage": {"input_tokens": 120}}, "output field missing"),
    ({"usage": {"input_tokens": "120", "output_tokens": 8}}, "a string, not an int"),
    ({"usage": {"input_tokens": -1, "output_tokens": 8}}, "negative"),
    ({"usage": {"input_tokens": True, "output_tokens": 8}}, "a bool, which is an int"),
])
def test_an_unreadable_usage_block_records_nothing_rather_than_a_zero(response, why):
    """**This is the one that matters.** A provider that renames a field would otherwise
    silently start reporting free writes, and free is the direction that flatters us.
    `reported` staying 0 is what makes the write path publish no series at all instead of
    a run of zeros — see `Usage` and `telemetry.WRITE_TOKENS_IN`."""
    usage = Usage()
    _shape.record_usage(response, usage, "input_tokens", "output_tokens")
    assert usage.reported == 0, why
    assert (usage.input_tokens, usage.output_tokens) == (0, 0), why


def test_a_genuine_zero_output_is_recorded_rather_than_discarded():
    # 0 output tokens is possible (a refusal, an empty structured result); 0 *reported*
    # is not the same thing. The guard rejects unreadable, not small.
    usage = Usage()
    _shape.record_usage({"usage": {"input_tokens": 40, "output_tokens": 0}}, usage,
                        "input_tokens", "output_tokens")
    assert (usage.input_tokens, usage.output_tokens, usage.reported) == (40, 0, 1)


def test_both_hosted_backends_advertise_that_they_report_usage():
    """`reports_usage` is what the write path gates on. A backend that fills the
    accumulator but forgets to advertise is never handed one, and reports nothing."""
    from memvara.llm.anthropic import AnthropicLLM
    from memvara.llm.openai import OpenAILLM

    assert AnthropicLLM.reports_usage and OpenAILLM.reports_usage
    assert NullLLM.reports_usage is False


def test_a_missing_key_is_refused_at_construction_not_on_the_first_write(monkeypatch):
    """`openai.OpenAI()` refuses to construct without a key and this did not, so the same
    documented line — `Memvara(..., llm=XLLM())` — failed at two different moments
    depending on the backend: one at startup, one on the first turn that reached tier 2,
    which in a server is well after the deployment looked healthy."""
    import sys
    from types import SimpleNamespace as NS

    monkeypatch.setitem(sys.modules, "anthropic", NS(Anthropic=lambda: NS(messages=None)))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="no ANTHROPIC_API_KEY"):
        AnthropicLLM()

    # With a key it constructs, and an injected client skips the check entirely — that is
    # the escape hatch every test in this file already relies on.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert AnthropicLLM().name.startswith("anthropic/")
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert AnthropicLLM(client=FakeClient()).name.startswith("anthropic/")


def test_compose_relations_asks_once_about_a_vocabulary_and_shapes_what_returns(
        monkeypatch) -> None:
    """The acquisition call behind `retrieve/compose`, tested the way the other two are.

    Not verified against the live API in the session that wrote it — no key was available
    — so what this pins is the request shape and the parsing, which is what the fake
    client exists to check. The prompt carries the predicates and nothing else, because
    the question is about a vocabulary rather than about any query.
    """
    from memvara.llm.anthropic import AnthropicLLM
    from memvara.llm.base import COMPOSE_SCHEMA, COMPOSE_SYSTEM

    client = FakeClient({"derived": {"grandfather": 2, "uncle": 2, "father": 1}})
    got = AnthropicLLM(client=client).compose_relations(["father", "mother", "spouse"])

    assert got == {"grandfather": 2, "uncle": 2, "father": 1}, (
        "the backend shapes the response and leaves the filtering to the caller, which "
        "is the half that knows the store's own predicate names"
    )
    kwargs = client.calls[0]
    assert kwargs["system"] is COMPOSE_SYSTEM
    assert kwargs["output_config"]["format"]["schema"] == COMPOSE_SCHEMA
    assert "father, mother, spouse" in kwargs["messages"][0]["content"], (
        "the vocabulary is what was asked about, and nothing else is"
    )


def test_compose_relations_survives_a_response_that_is_not_the_shape_asked_for() -> None:
    """A missing or malformed `derived` map is an empty answer, not an exception. The
    feature is an enrichment; a model having a bad day must not reach the caller."""
    from memvara.llm.anthropic import AnthropicLLM

    for body in ({"derived": "not a map"}, {}, {"derived": {"x": "two"}}):
        llm = AnthropicLLM(client=FakeClient(body))
        assert llm.compose_relations(["father"]) == {}


def test_compose_relations_reads_a_bare_map_as_well_as_the_wrapped_one() -> None:
    """A live model returned 21 correct terms and this parser dropped every one.

    `COMPOSE_SCHEMA` declares `{"derived": {...}}`, and a backend whose structured output
    enforces the schema sends that. One that is only *asked* for it — any OpenAI-compatible
    endpoint, which is most of them — answers with the bare term-to-arity map instead. The
    parser required the wrapper and returned `{}` without complaint, so the acquisition
    looked like a model with no opinion rather than a reader of the wrong shape.

    The unit tests could not have caught it: the fake client returns whatever shape the
    test author expected, and the test author had written the schema. It took a call to a
    real endpoint — `nvidia/nemotron-3-ultra-550b-a55b` via OpenRouter — to see it.
    """
    from memvara.llm.anthropic import AnthropicLLM

    wrapped = FakeClient({"derived": {"grandfather": 2, "uncle": 2}})
    bare = FakeClient({"grandfather": 2, "uncle": 2})

    assert AnthropicLLM(client=wrapped).compose_relations(["father"]) == {
        "grandfather": 2, "uncle": 2}
    assert AnthropicLLM(client=bare).compose_relations(["father"]) == {
        "grandfather": 2, "uncle": 2}


def test_the_durability_rule_admits_dated_and_measured_happenings() -> None:
    """The extraction prompt used to rule events out by construction.

    "Skip anything transient to the current exchange" is the rule that keeps a memory
    store from filling with chatter, and it is right — but it also classified "I ran 30
    minutes yesterday" as transient, so no amount of event *vocabulary* could produce an
    event *claim*. A predicate pack supplies names; it does not supply intent.

    The carve-out is deliberately conditional rather than a relaxation: a happening earns
    durability by carrying a time or a measurement, and an undated, unmeasured one is
    still skipped. Asserted here because the prompt is the only place this is stated and
    a later edit could quietly drop the clause.
    """
    from memvara.llm.base import EXTRACT_SYSTEM

    assert "transient to the current exchange" in EXTRACT_SYSTEM, \
        "the durability rule itself must survive; this is a carve-out, not a removal"
    assert "counts as durable when the turn gives it a time or a measured quantity" \
        in EXTRACT_SYSTEM
    assert "still skipped" in EXTRACT_SYSTEM, \
        "the undated case must stay excluded, or the store fills with chatter"
    for field in ("`when`", "`amount`/`unit`"):
        assert field in EXTRACT_SYSTEM, f"the prompt must name {field} for it to be filled"


def test_shape_claims_carries_the_event_fields_through() -> None:
    """The normaliser rebuilds each claim from an explicit key list, so a field the
    schema gains and this function does not name is dropped between the model and the
    write path — silently, because the result is still a well-formed claim.

    That is the third place in this change with the same shape: the extraction schema,
    this allowlist, and each store's field tuple all enumerate a claim's fields
    independently, and none of them fails when one is missing. It cost a benchmark arm
    to find the store's copy and a second probe run to find this one.
    """
    from memvara.llm._shape import shape_claims

    base = {"subject": "user", "predicate": "ran", "object": "5k", "polarity": 1,
            "memory_type": "episodic", "confidence": 0.9, "source_index": 0}
    [got] = shape_claims({"claims": [dict(base, when="last month", amount=5,
                                          unit="Kilometres")]}, 1)
    assert (got["when"], got["amount"], got["unit"]) == ("last month", 5.0, "kilometres")


def test_shape_claims_declines_a_quantity_it_cannot_trust() -> None:
    """`isinstance(True, int)` is True in Python, so a stray boolean would land as
    `amount=1.0` — a measurement nobody made, indistinguishable afterwards from one
    somebody did. The same slip `source_index` already guards against.

    A blank `when` is `None` rather than an empty string, and a unit without an amount is
    dropped: a unit alone measures nothing, and storing it invites a later reader to take
    it for a value.
    """
    from memvara.llm._shape import shape_claims

    base = {"subject": "user", "predicate": "ran", "object": "5k", "polarity": 1,
            "memory_type": "episodic", "confidence": 0.9, "source_index": 0}
    [got] = shape_claims({"claims": [dict(base, when="   ", amount=True, unit="min")]}, 1)
    assert (got["when"], got["amount"], got["unit"]) == (None, None, None)


@pytest.mark.parametrize("raw, why", [
    (float("inf"), "a JSON 1e400 parses to inf rather than raising"),
    (float("-inf"), "and the same in the other direction"),
    (float("nan"), "nan compares false against everything, including itself"),
])
def test_a_non_finite_amount_is_not_a_measurement(raw, why) -> None:
    """`inf` is not a distance, and storing it makes a measurement nobody took
    indistinguishable from one somebody did.

    It reaches here intact: `json.loads('{"amount": 1e400}')` yields `inf` rather than
    raising, `isinstance(inf, float)` is True, and the bool guard does not fire. Without a
    check it round-trips through `put_claim` and sits in the store reading as real, and
    nothing downstream rejects it — `amount` is not part of claim identity, so it does not
    even change which slot the claim occupies.
    """
    from memvara.llm._shape import shape_claims

    base = {"subject": "user", "predicate": "ran", "object": "5k", "polarity": 1,
            "memory_type": "episodic", "confidence": 0.9, "source_index": 0,
            "when": None, "unit": "km"}
    [got] = shape_claims({"claims": [dict(base, amount=raw)]}, 1)
    assert (got["amount"], got["unit"]) == (None, None), why


def test_an_unconvertible_amount_does_not_cost_the_whole_batch() -> None:
    """A 400-digit integer raises `OverflowError` from `float()`, and `WritePipeline`
    catches it around the whole `extract` call — so one malformed number discards every
    claim the model returned for that batch, recorded the same way a provider 429 is. An
    operator watching extraction yield drop cannot tell the two apart.

    A value that cannot become a float cannot be a measurement, so it is dropped like any
    other unusable one and the claim survives without a quantity.
    """
    import json

    from memvara.llm._shape import shape_claims

    huge = json.loads('{"a": 1' + "0" * 400 + "}")["a"]
    base = {"subject": "user", "predicate": "ran", "object": "5k", "polarity": 1,
            "memory_type": "episodic", "confidence": 0.9, "source_index": 0,
            "when": None, "unit": "km"}
    [got] = shape_claims({"claims": [dict(base, amount=huge)]}, 1)
    assert (got["amount"], got["unit"]) == (None, None)
    assert got["object"] == "5k", "the claim itself must survive a bad quantity"


# -- truncation ------------------------------------------------------------------------


def test_a_truncated_response_raises_rather_than_returning_no_claims():
    """The same silence `test_llm_openai.py` describes at length, under this provider's
    name for it. Anthropic reports it as `stop_reason="max_tokens"` on the response
    rather than per choice, which is the whole of the difference."""
    client = FakeClient({"claims": [claim()]}, stop_reason="max_tokens")
    with pytest.raises(TruncatedResponse, match=r"claude-opus-5.*8192-token"):
        AnthropicLLM(client=client).extract(episodes("hi"), [])


@pytest.mark.parametrize("reason", ["end_turn", "stop_sequence", "tool_use"])
def test_a_model_that_finished_its_answer_does_not_raise(reason):
    client = FakeClient({"claims": [claim()]}, stop_reason=reason)
    assert AnthropicLLM(client=client).extract(episodes("hi"), []) == [claim()]


@pytest.mark.parametrize("response, expected, why", [
    (SimpleNamespace(stop_reason="max_tokens"), "max_tokens", "an SDK object"),
    ({"stop_reason": "max_tokens"}, "max_tokens", "a plain dict"),
    (SimpleNamespace(), None, "a response that does not say"),
    ({}, None, "a dict that does not say"),
])
def test_the_stop_reason_is_read_from_whatever_shape_arrives(response, expected, why):
    assert _stop_reason(response) == expected, why
