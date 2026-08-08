"""`OpenAILLM` request shape and response handling.

Offline against an injected fake client: no API key, no network, and the `openai` package
is not installed. The validation itself is not retested here — it lives in `_shape` and is
covered once, through `AnthropicLLM`, in `test_llm.py`. What *is* tested here is
everything that made this a separate file: strict structured output, the Chat Completions
response shape, and refusals.

The last of those is the reason this suite exists at all. A refusal populates
`message.refusal` and leaves `content` null, so a backend that reads `content` alone turns
"the model declined" into "the model found no facts" — zero claims, no error, nothing in
the receipt, and a turn that silently carries no memory.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from engram.llm import LLM
from engram.llm.base import CLAIM_SCHEMA, EXTRACT_SYSTEM, PREDICATE_SCHEMA, RESOLVE_SCHEMA
from engram.llm.openai import OpenAILLM, _first_text
from engram.types import Episode, Scope


class FakeCompletions:
    def __init__(self, payloads: list[object]) -> None:
        self.payloads = payloads
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        payload = self.payloads[min(len(self.calls) - 1, len(self.payloads) - 1)]
        if isinstance(payload, SimpleNamespace):      # a hand-built message
            message = payload
        elif isinstance(payload, str):
            message = SimpleNamespace(content=payload, refusal=None)
        else:
            message = SimpleNamespace(content=json.dumps(payload), refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    """Stands in for `openai.OpenAI()` and records every request verbatim."""

    def __init__(self, *payloads: object) -> None:
        self.completions = FakeCompletions(list(payloads) or [{"claims": []}])
        self.chat = SimpleNamespace(completions=self.completions)

    @property
    def calls(self) -> list[dict]:
        return self.completions.calls


def episodes(*contents: str) -> list[Episode]:
    return [Episode(content=c, scope=Scope(tenant="acme")) for c in contents]


def claim(**overrides) -> dict:
    base = {
        "subject": "user", "predicate": "lives_in", "object": "lisbon",
        "polarity": 1, "memory_type": "semantic", "confidence": 0.9, "source_index": 0,
    }
    base.update(overrides)
    return base


# --- the protocol ---------------------------------------------------------------

def test_it_satisfies_the_llm_protocol():
    assert isinstance(OpenAILLM(client=FakeClient()), LLM)


def test_it_reports_itself_as_a_real_backend():
    """`is_noop` is what `WriteReceipt.llm_calls` uses to decide whether a call was
    billable. A real backend claiming to be a no-op would make the cost reporting — this
    library's central claim — quietly wrong."""
    llm = OpenAILLM(client=FakeClient())
    assert llm.is_noop is False
    assert llm.name == "openai/gpt-4.1"
    assert "gpt-4.1" in repr(llm)


# --- strict structured output ---------------------------------------------------

def test_the_request_asks_for_strict_structured_output():
    """Without `strict`, `response_format` is a suggestion. Schema-shaped output that is
    *usually* right is the worst case: it passes in testing and fails in production."""
    client = FakeClient({"claims": []})
    OpenAILLM(client=client).extract(episodes("I live in Lisbon"), [])
    fmt = client.calls[0]["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"] is CLAIM_SCHEMA
    assert fmt["json_schema"]["name"]        # the API rejects a request without one


@pytest.mark.parametrize("schema", [CLAIM_SCHEMA, RESOLVE_SCHEMA, PREDICATE_SCHEMA])
def test_every_schema_satisfies_strict_mode(schema):
    """Strict mode requires every property listed in `required` and
    `additionalProperties: false` at each object level. The schemas already comply, and
    this pins it so a later edit cannot break the OpenAI path without failing here."""
    def check(node):
        if node.get("type") == "object":
            assert node.get("additionalProperties") is False
            assert set(node.get("required", [])) == set(node.get("properties", {}))
            for child in node.get("properties", {}).values():
                check(child)
        elif node.get("type") == "array":
            check(node["items"])

    check(schema)


def test_the_system_prompt_and_temperature_ride_on_the_request():
    """Temperature 0 because extraction is parsing, not writing: the same turn twice
    should give one claim, not two spellings the reconciler treats as competing."""
    client = FakeClient({"claims": []})
    OpenAILLM(client=client, temperature=0.0).extract(episodes("hi"), [])
    call = client.calls[0]
    assert call["messages"][0] == {"role": "system", "content": EXTRACT_SYSTEM}
    assert call["messages"][1]["role"] == "user"
    assert call["temperature"] == 0.0


# --- refusals -------------------------------------------------------------------

def test_a_refusal_yields_no_claims_rather_than_a_silent_empty_extraction():
    refusal = SimpleNamespace(content=None, refusal="I can't help with that.")
    llm = OpenAILLM(client=FakeClient(refusal))
    assert llm.extract(episodes("something the model declines"), []) == []


def test_a_refusal_does_not_get_read_as_content():
    """The specific bug: `content` is null on a refusal, but a backend that falls back to
    `str(content or refusal)` would try to JSON-parse the refusal text. It must be treated
    as "nothing usable", not as a response."""
    assert _first_text(SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content='{"claims": []}', refusal="I can't help with that."))])) == ""


def test_a_refusal_on_predicate_resolution_falls_back_to_new_rather_than_guessing():
    refusal = SimpleNamespace(content=None, refusal="no")
    out = OpenAILLM(client=FakeClient(refusal)).resolve_predicate("resides_in", ["lives_in"])
    # `canonical=None` means "treat it as a new predicate" — the recoverable direction.
    assert out["canonical"] is None
    assert out["cardinality"] == "many"


# --- response shape -------------------------------------------------------------

@pytest.mark.parametrize("response, why", [
    (SimpleNamespace(choices=[]), "no choices"),
    (SimpleNamespace(choices=[SimpleNamespace(message=None)]), "no message"),
    (SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content=None, refusal=None))]), "null content"),
    ({"choices": [{"message": {"content": None, "refusal": None}}]}, "plain dicts"),
])
def test_a_response_with_nothing_readable_yields_empty_text(response, why):
    assert _first_text(response) == "", why


def test_plain_dicts_work_so_a_test_double_need_not_import_the_sdk():
    assert _first_text({"choices": [{"message": {"content": "hello", "refusal": None}}]}) \
        == "hello"


def test_a_well_formed_extraction_round_trips():
    client = FakeClient({"claims": [claim(), claim(object="berlin", source_index=1)]})
    out = OpenAILLM(client=client).extract(episodes("a", "b"), ["lives_in"])
    assert [(c["object"], c["source_index"]) for c in out] == [("lisbon", 0), ("berlin", 1)]


def test_validation_is_shared_not_reimplemented():
    """A hallucinated `source_index` is dropped here exactly as it is for Anthropic. If
    this ever diverges, one store holds differently-shaped claims depending on which model
    was configured the day a turn was written."""
    client = FakeClient({"claims": [claim(source_index=99), claim(source_index=0)]})
    out = OpenAILLM(client=client).extract(episodes("only one turn"), [])
    assert len(out) == 1 and out[0]["source_index"] == 0


def test_no_episodes_makes_no_request():
    """A call we should not pay for."""
    client = FakeClient()
    assert OpenAILLM(client=client).extract([], ["lives_in"]) == []
    assert client.calls == []


def test_resolve_and_classify_go_through_the_shared_shaping():
    client = FakeClient(
        {"canonical": "lives_in", "cardinality": "one", "volatility": "slow",
         "memory_type": "semantic"},
        {"cardinality": "one", "volatility": "fast", "memory_type": "episodic"},
    )
    llm = OpenAILLM(client=client)
    assert llm.resolve_predicate("resides_in", ["lives_in"])["canonical"] == "lives_in"
    assert llm.classify_predicate("working_on", "I'm working on the parser") == {
        "cardinality": "one", "volatility": "fast", "memory_type": "episodic"}


# --- construction ---------------------------------------------------------------

def test_a_missing_sdk_names_the_extra_and_the_two_ways_out(monkeypatch):
    """The error a user hits first, so it has to say what to install *and* that the
    library runs without it at all."""
    import builtins
    real = builtins.__import__

    def no_openai(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("no module named openai")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_openai)
    with pytest.raises(ImportError, match=r"engram\[openai\]"):
        OpenAILLM()


def test_an_injected_client_skips_construction_entirely():
    client = FakeClient()
    assert OpenAILLM(client=client, model="gpt-4o")._client is client


def test_the_default_client_is_built_from_the_sdk(monkeypatch):
    import sys

    sentinel = object()
    fake_sdk = SimpleNamespace(OpenAI=lambda: sentinel)
    monkeypatch.setitem(sys.modules, "openai", fake_sdk)
    assert OpenAILLM()._client is sentinel


def test_reachable_from_both_packages_without_the_sdk_installed():
    """PEP 562 lazy attributes. The `openai` package is genuinely absent here, which is
    the configuration this has to survive: naming the class must not import the SDK, or
    the default offline install stops being able to `import engram` at all."""
    import engram
    import engram.llm as pkg

    assert pkg.OpenAILLM is OpenAILLM
    assert engram.OpenAILLM is OpenAILLM
    assert "OpenAILLM" in pkg.__all__ and "OpenAILLM" in engram.__all__
    with pytest.raises(AttributeError):
        engram.NotAThing
