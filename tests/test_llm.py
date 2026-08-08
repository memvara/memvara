"""`AnthropicLLM` request shape and output validation.

Everything here runs offline against an injected fake client: no API key, no network,
and the `anthropic` package is not installed.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from engram.llm import LLM, AnthropicLLM, NullLLM
from engram.llm.base import CLAIM_SCHEMA, EXTRACT_SYSTEM, PREDICATE_SCHEMA, PREDICATE_SYSTEM
from engram.types import Episode, Scope


class FakeMessages:
    def __init__(self, payloads: list[object]) -> None:
        self.payloads = payloads
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
        return SimpleNamespace(content=blocks, stop_reason="end_turn")


class FakeClient:
    """Stands in for `anthropic.Anthropic()` and records every request verbatim."""

    def __init__(self, *payloads: object) -> None:
        self.messages = FakeMessages(list(payloads) or [{"claims": []}])

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


def test_exported_from_the_package_without_the_sdk_installed():
    import engram.llm as pkg

    assert pkg.NullLLM is NullLLM
    assert set(pkg.__all__) == {"LLM", "NullLLM", "AnthropicLLM"}
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
    llm.extract(eps, ["lives_in", "works_at"])

    prompt = client.calls[0]["messages"][0]["content"]
    assert "[0] user: I live in Lisbon" in prompt
    assert "[1] user: I work at Acme" in prompt
    assert "lives_in, works_at" in prompt
    # Same predicate set, same bytes - the prompt prefix has to stay cacheable.
    assert client.calls[1]["messages"][0]["content"] == prompt


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
