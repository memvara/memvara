"""The model settings a server reads from its environment reach the model it calls.

`memvara/server/config.py` reads eleven variables that shape what the server asks an
extraction model: which backend, which model, how many tokens and claims, which claim
shape, how long to wait, which extra request fields, which instructions, which project
guidance, whether to ask for replacement advice, and whether to drop claims under an
unregistered predicate. `docs/DEPLOY.md` documents each of them for an operator.

Each test here configures a store the way `python -m memvara.server` does, with
`ServerConfig.from_env` and then `build_memvara`, and points the OpenAI backend at
`FakeOpenAI`. The test then looks at the request the fake received, so it checks what
the model is actually sent rather than an attribute on the backend object. The refusals
of invalid values are checked by the tests in `tests/test_server.py`,
`tests/test_advisory.py` and `tests/test_extraction_guidance.py` that carry the same
covers marks.
"""

from __future__ import annotations

import pathlib
import sys
import time
import types
from typing import Any, Iterator

import pytest

from harness.fakes.openai_compat import FakeOpenAI
from harness.skips import needs_toml
from memvara import Memvara
from memvara.llm.base import EXTRACT_SYSTEM
from memvara.server.config import ServerConfig, build_memvara

#: A turn the deterministic fast path does not recognise, so a store with a model has to
#: ask the model about it.
TURN = "I have been practising the cello every evening since spring."

#: How long the fake client waits for an answer when the request names no timeout. It is
#: far longer than any test here should take, so a timeout that did not reach the request
#: shows up as a test that took too long.
CLIENT_TIMEOUT = 20.0

#: The claim the model proposes in the closed-vocabulary test. `build_commit` is not a
#: registered predicate.
UNREGISTERED = {"subject": "memvara", "predicate": "build_commit", "object": "127f6eb",
                "polarity": 1, "memory_type": "semantic", "confidence": 0.9,
                "source_index": 0, "when": None, "amount": None, "unit": None}


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeOpenAI]:
    """A fake OpenAI-compatible endpoint that the `openai` backend reaches.

    CI does not install the `openai` package, so the module is replaced by one whose
    `OpenAI()` returns the fake's client. `OpenAILLM` builds its client through exactly
    that call when the server gives it none.
    """
    with FakeOpenAI() as fake:
        monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(
            OpenAI=lambda **_: fake.client(timeout=CLIENT_TIMEOUT)))
        yield fake


def _open(**variables: str) -> Memvara:
    """A store configured from `variables` the way the server's command line does it."""
    memory = build_memvara(ServerConfig.from_env({
        "MEMVARA_DB": ":memory:", "MEMVARA_USER": "alice",
        "MEMVARA_FEATURE_PROJECT_SCOPE": "0", **variables}))
    assert isinstance(memory, Memvara), "a local configuration builds a local engine"
    return memory


def _extraction_request(model: FakeOpenAI, **variables: str) -> dict[str, Any]:
    """The body of the one extraction request a new store sends for `TURN`."""
    model.add_json({"claims": []})
    before = len(model.requests)
    with _open(MEMVARA_LLM="openai", **variables) as memory:
        memory.add(TURN)
    (request,) = model.requests[before:]
    body = request.json()
    assert isinstance(body, dict)
    return body


def _claims_schema(body: dict[str, Any]) -> dict[str, Any]:
    schema = body["response_format"]["json_schema"]["schema"]["properties"]["claims"]
    assert isinstance(schema, dict)
    return schema


@pytest.mark.covers("env:MEMVARA_LLM")
def test_the_backend_variable_decides_whether_a_model_is_asked(model: FakeOpenAI) -> None:
    """`MEMVARA_LLM=openai` sends a turn the fast path cannot read to the model and stores
    what the model answers. Unset, it means `none`: no model is asked and the turn yields
    no claim. `docs/DEPLOY.md` documents both values."""
    model.add_json({"claims": [{**UNREGISTERED, "subject": "user", "predicate": "likes",
                                "object": "the cello"}]})
    with _open(MEMVARA_LLM="openai") as memory:
        receipt = memory.add(TURN)
    assert [(c.predicate, c.object) for c in receipt.added] == [("likes", "the cello")]
    assert len(model.requests) == 1

    with _open() as memory:
        receipt = memory.add(TURN)
    assert receipt.added == [] and receipt.llm_calls == 0
    assert len(model.requests) == 1, "with no backend named, nothing reaches a model"


@pytest.mark.covers("env:MEMVARA_LLM_MODEL")
def test_the_model_variable_names_the_model_on_every_request(model: FakeOpenAI) -> None:
    """The model an operator names is the model the request asks for, which is how a
    self-hosted server behind `OPENAI_BASE_URL` picks its weights."""
    body = _extraction_request(model, MEMVARA_LLM_MODEL="Qwen/Qwen3.5-4B-Instruct")
    assert body["model"] == "Qwen/Qwen3.5-4B-Instruct"
    assert _extraction_request(model)["model"] == "gpt-4.1", "the backend's own default"


@pytest.mark.covers("env:MEMVARA_LLM_MAX_TOKENS")
def test_the_response_budget_is_sent_on_the_request(model: FakeOpenAI) -> None:
    """`MEMVARA_LLM_MAX_TOKENS` bounds how long one response may run. It has to arrive
    as the request's `max_completion_tokens`; unset, the backend's 8,192 is sent."""
    body = _extraction_request(model, MEMVARA_LLM_MAX_TOKENS="2048")
    assert body["max_completion_tokens"] == 2048
    assert _extraction_request(model)["max_completion_tokens"] == 8192


@pytest.mark.covers("env:MEMVARA_LLM_MAX_CLAIMS")
def test_the_claim_cap_is_sent_in_the_response_schema(model: FakeOpenAI) -> None:
    """`MEMVARA_LLM_MAX_CLAIMS` caps the claims array for a server that compiles the
    schema into a grammar. The cap has to be in the schema the model is sent, and it is
    absent by default because hosted OpenAI refuses `maxItems` under strict mode."""
    assert _claims_schema(_extraction_request(
        model, MEMVARA_LLM_MAX_CLAIMS="12"))["maxItems"] == 12
    assert "maxItems" not in _claims_schema(_extraction_request(model))


@pytest.mark.covers("env:MEMVARA_LLM_TERSE_CLAIMS")
def test_the_terse_claim_shape_is_the_schema_the_model_is_sent(model: FakeOpenAI) -> None:
    """`MEMVARA_LLM_TERSE_CLAIMS=1` stops requiring the four fields a turn rarely states,
    so a slow self-hosted model writes less. The shorter `required` list has to be in the
    schema on the request; the default keeps all ten fields required."""
    terse = _claims_schema(_extraction_request(model, MEMVARA_LLM_TERSE_CLAIMS="1"))
    assert terse["items"]["required"] == ["subject", "predicate", "object", "source_index",
                                          "memory_type", "confidence"]
    assert len(_claims_schema(_extraction_request(model))["items"]["required"]) == 10


@pytest.mark.covers("env:MEMVARA_LLM_EXTRA_BODY")
def test_the_extra_body_fields_are_sent_on_the_request(model: FakeOpenAI) -> None:
    """`MEMVARA_LLM_EXTRA_BODY` carries request fields the SDK does not name, such as the
    switch that stops a Qwen3 server from thinking through its whole budget. The fields
    have to arrive in the request body; unset, nothing extra is sent."""
    body = _extraction_request(
        model, MEMVARA_LLM_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}')
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "chat_template_kwargs" not in _extraction_request(model)


@pytest.mark.covers("env:MEMVARA_LLM_EXTRACT_SYSTEM")
def test_the_replacement_instructions_are_the_system_message(
        model: FakeOpenAI, tmp_path: pathlib.Path) -> None:
    """`MEMVARA_LLM_EXTRACT_SYSTEM` names a file whose text replaces memvara's own
    extraction instructions. The file's text, stripped, has to be the system message the
    model receives; unset, the shipped instructions are sent."""
    prompt = tmp_path / "extract.txt"
    prompt.write_text("  Only the facts.\n", encoding="utf-8")
    body = _extraction_request(model, MEMVARA_LLM_EXTRACT_SYSTEM=str(prompt))
    assert body["messages"][0] == {"role": "system", "content": "Only the facts."}
    assert _extraction_request(model)["messages"][0]["content"] == EXTRACT_SYSTEM


@needs_toml
@pytest.mark.covers("env:MEMVARA_EXTRACT_GUIDANCE")
def test_the_project_guidance_is_appended_to_the_system_message(
        model: FakeOpenAI, tmp_path: pathlib.Path) -> None:
    """`MEMVARA_EXTRACT_GUIDANCE` names a TOML file of project rules. The rules have to
    be added to the system message, after the instructions, and never to the user
    message, where the conversation is. Unset, the system message is the instructions
    alone."""
    guide = tmp_path / "guidance.toml"
    guide.write_text('context = "A payments service."\n'
                     'include = ["decisions about retries"]\n', encoding="utf-8")
    body = _extraction_request(model, MEMVARA_EXTRACT_GUIDANCE=str(guide))
    system, user = (message["content"] for message in body["messages"])
    assert system.startswith(EXTRACT_SYSTEM) and system != EXTRACT_SYSTEM
    assert "A payments service." in system and "decisions about retries" in system
    assert "decisions about retries" not in user
    assert _extraction_request(model)["messages"][0]["content"] == EXTRACT_SYSTEM


@pytest.mark.covers("env:MEMVARA_LLM_TIMEOUT")
def test_the_extraction_timeout_cuts_off_a_model_that_does_not_answer(
        model: FakeOpenAI) -> None:
    """`MEMVARA_LLM_TIMEOUT` is how long one extraction may wait. A model that never
    answers has to be given up on after that time, and the turn is kept to be extracted
    again rather than lost. Without the setting the client would wait its own default,
    which here is twenty seconds."""
    model.add_hang()
    with _open(MEMVARA_LLM="openai", MEMVARA_LLM_TIMEOUT="0.3") as memory:
        started = time.monotonic()
        receipt = memory.add(TURN)
        waited = time.monotonic() - started
    assert waited < 5, f"the write waited {waited:.1f}s for a model that never answered"
    assert (receipt.added, receipt.unextracted, receipt.deferred) == ([], 1, True)
    assert len(model.requests) == 1


@pytest.mark.covers("env:MEMVARA_ADVISE_REPLACEMENTS")
def test_replacement_advice_asks_the_model_only_when_switched_on(model: FakeOpenAI) -> None:
    """`MEMVARA_ADVISE_REPLACEMENTS=1` makes `remember` ask the model whether a new fact
    in another slot is a newer version of a stored one, and name that fact on the
    receipt. Off, which is the default, no model is asked and nothing is named."""
    model.add_json({"same_thing": True, "same_property": True, "newer_value": True,
                    "replaces": True})
    with _open(MEMVARA_LLM="openai", MEMVARA_ADVISE_REPLACEMENTS="1") as memory:
        old = memory.remember("user", "employer", "Acme").added[0]
        receipt = memory.remember("user", "hired_by", "Globex")
    assert [claim.id for claim in receipt.may_replace] == [old.id]
    assert [r.json()["response_format"]["json_schema"]["name"]
            for r in model.requests] == ["replacement_verdict"]

    with _open(MEMVARA_LLM="openai") as memory:
        memory.remember("user", "employer", "Acme")
        receipt = memory.remember("user", "hired_by", "Globex")
    assert receipt.may_replace == [] and len(model.requests) == 1


@pytest.mark.covers("env:MEMVARA_CLOSED_VOCABULARY")
def test_a_closed_vocabulary_drops_a_claim_under_an_unregistered_predicate(
        model: FakeOpenAI) -> None:
    """`MEMVARA_CLOSED_VOCABULARY=1` makes the write path drop a claim the model proposes
    under a predicate nobody registered, and count it on the receipt as unregistered,
    instead of learning the new predicate."""
    model.add_json({"claims": [UNREGISTERED]})
    with _open(MEMVARA_LLM="openai", MEMVARA_CLOSED_VOCABULARY="1") as memory:
        receipt = memory.add("The build is at commit 127f6eb on the release branch.")
        assert memory.get_all() == []
    assert (receipt.added, receipt.unregistered) == ([], 1)
    assert len(model.requests) == 1, "a refused predicate costs no call to learn it"
