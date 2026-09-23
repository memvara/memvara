"""Per-project extraction guidance: checked, appended to the system message, and wired.

Guidance adds a project's own rules to the shipped extraction prompt. The tests here hold
four things. It is refused when it is too long, rather than cut. It goes in the system
message and never in the user message, where the conversation turns are. Both backends
append it to whichever prompt they use, including a full replacement from
`MEMVARA_LLM_EXTRACT_SYSTEM`. And the MCP server reads it from `MEMVARA_EXTRACT_GUIDANCE`
at startup, refusing a bad file then rather than at the first write.
"""
from __future__ import annotations

import sys
from typing import Any

import pytest

from memvara import Guidance, HashingEmbedder, Memvara, NullLLM
from memvara.llm import AnthropicLLM, OpenAILLM
from memvara.llm.base import EXTRACT_SYSTEM
from memvara.llm.guidance import (GUIDANCE_HEADING, MAX_CONTEXT_CHARS, MAX_RULE_CHARS,
                                  MAX_RULES, GuidanceError, load_guidance, with_guidance)
from memvara.server.config import (FEATURE_DEFAULTS, ConfigError, ServerConfig,
                                   build_memvara)

import test_llm
import test_llm_openai

needs_toml = pytest.mark.skipif(sys.version_info < (3, 11),
                                reason="tomllib arrives in 3.11; load_guidance refuses below it")

GUIDE = Guidance(context="A payments service.",
                 include=["decisions about retries"], exclude=["stack traces"])


# --- the value --------------------------------------------------------------

def test_a_context_over_the_limit_is_refused_rather_than_cut():
    Guidance(context="x" * MAX_CONTEXT_CHARS)
    with pytest.raises(GuidanceError, match="1501 characters, over the 1500"):
        Guidance(context="x" * (MAX_CONTEXT_CHARS + 1))


def test_more_than_twenty_rules_in_either_list_is_refused():
    Guidance(include=["r"] * MAX_RULES, exclude=["r"] * MAX_RULES)
    with pytest.raises(GuidanceError, match="include has 21 rules, over the 20"):
        Guidance(include=["r"] * (MAX_RULES + 1))
    with pytest.raises(GuidanceError, match="exclude has 21 rules"):
        Guidance(exclude=["r"] * (MAX_RULES + 1))


def test_a_rule_over_two_hundred_characters_is_refused_and_named_by_position():
    Guidance(include=["x" * MAX_RULE_CHARS])
    with pytest.raises(GuidanceError, match="include rule 2 is 201 characters"):
        Guidance(include=["fine", "x" * (MAX_RULE_CHARS + 1)])


@pytest.mark.parametrize("bad, message", [
    ({"include": "stack traces"}, "include must be a list of rules, not str"),
    ({"exclude": None}, "exclude must be a list of rules, not NoneType"),
    ({"include": [3]}, "include rule 1 must be text, not int"),
    ({"exclude": ["ok", "   "]}, "exclude rule 2 is blank"),
    ({"context": 12}, "context must be text, not int"),
])
def test_a_wrong_shape_is_refused_with_the_field_named(bad, message):
    """A bare string is the sharp case: iterated, every character would become a rule."""
    with pytest.raises(GuidanceError, match=message):
        Guidance(**bad)


def test_rules_are_stripped_and_stored_as_tuples_so_the_value_is_hashable():
    g = Guidance(context="  A service. ", include=[" retries "], exclude=("traces",))
    assert (g.context, g.include, g.exclude) == ("A service.", ("retries",), ("traces",))
    assert hash(g) == hash(Guidance(context="A service.", include=["retries"],
                                    exclude=["traces"]))


def test_no_guidance_and_empty_guidance_leave_the_prompt_byte_for_byte_unchanged():
    assert with_guidance("rules", None) == "rules"
    assert with_guidance("rules", Guidance()) == "rules"
    assert Guidance().render() == ""


def test_guidance_is_appended_after_the_shipped_rules_under_the_fixed_heading():
    system = with_guidance(EXTRACT_SYSTEM, GUIDE)
    assert system.startswith(EXTRACT_SYSTEM + "\n\n" + GUIDANCE_HEADING + ":\n")
    tail = system[len(EXTRACT_SYSTEM):]
    assert "About this project: A payments service." in tail
    assert "Extract:\n- decisions about retries" in tail
    assert "Do not extract:\n- stack traces" in tail
    assert "does not change the output format" in tail


def test_a_section_with_nothing_in_it_is_left_out():
    rendered = Guidance(include=["deadlines"]).render()
    assert "About this project" not in rendered and "Do not extract" not in rendered


# --- both backends ------------------------------------------------------------

def _anthropic_system(guidance: Guidance | None) -> tuple[str, Any]:
    client = test_llm.FakeClient({"claims": []})
    AnthropicLLM(client=client).extract(test_llm.episodes("I live in Lisbon"), [],
                                        guidance=guidance)
    return client.calls[0]["system"], client.calls[0]["messages"]


def test_anthropic_appends_guidance_to_its_system_message_and_nowhere_else():
    """The turns are data in the user message; the guidance is an instruction and stays
    in the system message, so a turn that quotes rules cannot pass for one."""
    system, messages = _anthropic_system(GUIDE)
    assert system == with_guidance(EXTRACT_SYSTEM, GUIDE)
    assert "stack traces" not in str(messages)
    assert AnthropicLLM.accepts_guidance is True


def test_anthropic_without_guidance_sends_the_shipped_prompt_exactly():
    assert _anthropic_system(None)[0] == EXTRACT_SYSTEM


def _openai_system(llm: OpenAILLM, client: Any, guidance: Guidance | None) -> tuple[str, str]:
    llm.extract(test_llm_openai.episodes("I live in Lisbon"), [], guidance=guidance)
    system, user = client.calls[-1]["messages"]
    return system["content"], user["content"]


def test_openai_appends_guidance_to_the_shipped_prompt():
    client = test_llm_openai.FakeClient({"claims": []})
    system, user = _openai_system(OpenAILLM(client=client), client, GUIDE)
    assert system == with_guidance(EXTRACT_SYSTEM, GUIDE)
    assert "stack traces" not in user
    assert _openai_system(OpenAILLM(client=client), client, None)[0] == EXTRACT_SYSTEM


def test_a_replacement_prompt_keeps_its_meaning_and_gets_the_guidance_appended():
    """`MEMVARA_LLM_EXTRACT_SYSTEM` replaces the shipped prompt; guidance adds to whichever
    prompt is in use and does not bring the shipped one back."""
    client = test_llm_openai.FakeClient({"claims": []})
    llm = OpenAILLM(client=client, extract_system="Small-model rules.")
    system, _ = _openai_system(llm, client, GUIDE)
    assert system == "Small-model rules.\n\n" + GUIDE.render()
    assert EXTRACT_SYSTEM not in system


def test_null_llm_takes_the_argument_and_calls_nothing():
    assert NullLLM().extract([], [], guidance=GUIDE) == []
    assert NullLLM.accepts_guidance is True


# --- the write path -------------------------------------------------------------

class Recording:
    """A backend that records the keyword arguments of every `extract` call."""

    name = "recording"
    is_noop = False
    reports_usage = False

    def __init__(self, accepts: bool = True) -> None:
        self.accepts_guidance = accepts
        self.calls: list[dict[str, Any]] = []

    def extract(self, episodes, known_predicates, **kw):
        self.calls.append(kw)
        return []

    def resolve_predicate(self, surface, candidates, **kw):
        return {"canonical": None, "cardinality": "many", "volatility": "slow",
                "memory_type": "semantic"}

    def classify_predicate(self, predicate, example, **kw):
        return {"cardinality": "many", "volatility": "slow", "memory_type": "semantic"}


def _mem(llm, **kw) -> Memvara:
    return Memvara(llm=llm, embedder=HashingEmbedder(dim=64), user="alice", **kw)


def test_every_extraction_call_carries_the_guidance_the_store_was_opened_with():
    llm = Recording()
    mem = _mem(llm, write_guidance=GUIDE)
    assert mem.writer.guidance is GUIDE
    mem.add("We decided to retry payments three times before paging anyone.")
    assert llm.calls and all(call["guidance"] is GUIDE for call in llm.calls)


def test_without_guidance_the_argument_is_not_sent_so_older_backends_keep_working():
    llm = Recording(accepts=False)
    _mem(llm).add("We decided to retry payments three times before paging anyone.")
    assert llm.calls and all("guidance" not in call for call in llm.calls)


def test_empty_guidance_is_no_guidance_even_for_a_backend_that_cannot_take_it():
    llm = Recording(accepts=False)
    assert _mem(llm, write_guidance=Guidance()).writer.guidance is None


def test_guidance_for_a_backend_that_does_not_accept_it_is_refused_at_construction():
    """The other outcome is a guidance file the operator wrote that no extraction sees."""
    with pytest.raises(TypeError, match="recording does not accept extraction guidance"):
        _mem(Recording(accepts=False), write_guidance=GUIDE)


# --- the file -------------------------------------------------------------------

@needs_toml
def test_a_guidance_file_is_read_into_a_guidance(tmp_path):
    path = tmp_path / "guidance.toml"
    path.write_text('context = "A payments service."\n'
                    'include = ["decisions about retries"]\n'
                    'exclude = ["stack traces"]\n', encoding="utf-8")
    assert load_guidance(path) == GUIDE
    empty = tmp_path / "empty.toml"
    empty.write_text("", encoding="utf-8")
    assert load_guidance(empty).is_empty


@needs_toml
@pytest.mark.parametrize("body, message", [
    ('exlude = ["x"]\n', "unknown key\\(s\\) exlude"),
    ("include = [\n", "is not valid TOML"),
    ('include = ["' + "x" * 201 + '"]\n', "include rule 1 is 201 characters"),
])
def test_a_bad_guidance_file_is_refused_with_the_file_named(tmp_path, body, message):
    """An unknown key is refused rather than ignored: `exlude` would otherwise leave the
    rule out of every extraction with nothing saying why."""
    path = tmp_path / "guidance.toml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(GuidanceError, match=message) as caught:
        load_guidance(path)
    assert str(path) in str(caught.value)


@needs_toml
def test_a_missing_huge_or_binary_file_is_refused(tmp_path):
    with pytest.raises(GuidanceError, match="cannot be read"):
        load_guidance(tmp_path / "absent.toml")
    huge = tmp_path / "model.bin"
    huge.write_bytes(b"x" * (64 * 1024 + 1))
    with pytest.raises(GuidanceError, match="over the 65536 this reads"):
        load_guidance(huge)
    binary = tmp_path / "binary.toml"
    binary.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(GuidanceError, match="is not valid TOML"):
        load_guidance(binary)


@pytest.mark.skipif(sys.version_info >= (3, 11), reason="3.10 only")
def test_on_python_3_10_a_guidance_file_is_refused_and_names_the_way_round(tmp_path):
    path = tmp_path / "guidance.toml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(GuidanceError, match="Python 3.11"):
        load_guidance(path)


# --- the MCP server's configuration --------------------------------------------

def test_both_switches_are_on_by_default():
    assert FEATURE_DEFAULTS["extraction_guidance"] is True
    assert FEATURE_DEFAULTS["expiry_erasure"] is True


@pytest.fixture
def guidance_file(tmp_path):
    path = tmp_path / "guidance.toml"
    path.write_text('include = ["decisions about retries"]\n', encoding="utf-8")
    return str(path)


def _env(tmp_path, **extra: str) -> dict[str, str]:
    return {"MEMVARA_DB": str(tmp_path / "m.db"), "MEMVARA_PROJECT": "path:" + "0" * 16,
            **extra}


@needs_toml
def test_the_server_reads_the_guidance_path_and_checks_the_file_at_startup(
        tmp_path, guidance_file):
    config = ServerConfig.from_env(_env(tmp_path, MEMVARA_LLM="openai",
                                        MEMVARA_EXTRACT_GUIDANCE=guidance_file))
    assert config.extract_guidance == guidance_file
    broken = tmp_path / "broken.toml"
    broken.write_text("include = 3\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="MEMVARA_EXTRACT_GUIDANCE: .*include must be"):
        ServerConfig.from_env(_env(tmp_path, MEMVARA_LLM="openai",
                                   MEMVARA_EXTRACT_GUIDANCE=str(broken)))


@needs_toml
def test_guidance_with_no_model_is_refused_unless_the_switch_is_off(tmp_path, guidance_file):
    """Read and never used is the silent failure the other refusals here exist to stop."""
    with pytest.raises(ConfigError, match="MEMVARA_LLM is 'none', so no model"):
        ServerConfig.from_env(_env(tmp_path, MEMVARA_EXTRACT_GUIDANCE=guidance_file))
    config = ServerConfig.from_env(_env(tmp_path, MEMVARA_EXTRACT_GUIDANCE=guidance_file,
                                        MEMVARA_FEATURE_EXTRACTION_GUIDANCE="0"))
    assert "extraction_guidance" in config.features_off


@needs_toml
def test_the_local_store_is_opened_with_the_guidance_and_without_it_when_switched_off(
        tmp_path, guidance_file):
    config = ServerConfig(path=str(tmp_path / "m.db"), extract_guidance=guidance_file,
                          embedder="hashing:64", features_off=frozenset({"encryption"}))
    with build_memvara(config) as mem:
        assert mem.writer.guidance == Guidance(include=["decisions about retries"])
    off = ServerConfig(path=str(tmp_path / "m.db"), extract_guidance=guidance_file,
                       embedder="hashing:64",
                       features_off=frozenset({"encryption", "extraction_guidance"}))
    with build_memvara(off) as mem:
        assert mem.writer.guidance is None


def test_guidance_is_refused_under_cloud_mode_where_the_deployment_extracts():
    config = ServerConfig(mode="cloud", api_key="k", extract_guidance="/g.toml")
    with pytest.raises(ConfigError, match="MEMVARA_EXTRACT_GUIDANCE='/g.toml' does not apply"):
        build_memvara(config)
