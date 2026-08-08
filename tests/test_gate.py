"""SalienceGate: the tier that decides whether a turn is worth an extraction call.

Everything here is offline and deterministic — the gate never touches a model, and a
test that needed one would be testing the wrong thing.
"""

from __future__ import annotations

import pytest

from engram.types import Episode, Scope
from engram.write import SalienceGate


def ep(content: str, role: str = "user", **kw) -> Episode:
    return Episode(content=content, role=role, **kw)


@pytest.fixture()
def gate() -> SalienceGate:
    return SalienceGate()


# --- negatives: the turns that must never cost a call ------------------------

@pytest.mark.parametrize("content", ["", "   ", "\n\n", "\t \n  \t"])
def test_empty_and_whitespace_are_dropped(gate, content):
    assert gate.carries_fact(ep(content)) == (False, "no_content")


@pytest.mark.parametrize(
    "content",
    ["ok", "OK!", "thanks", "Thanks!", "thank you", "Thank you :)", "sounds good",
     "Got it.", "perfect", "great, thanks", "ok cool thanks", "yep", "no problem",
     "hi", "Hello!", "makes sense", "will do", "haha", "  Sure.  "],
)
def test_acknowledgements_are_dropped(gate, content):
    ok, reason = gate.carries_fact(ep(content))
    assert (ok, reason) == (False, "ack_only")


@pytest.mark.parametrize(
    "content",
    ["sure thing", "no worries", "ok sounds good to me", "yeah ok thanks",
     "that makes sense", "all good", "see you later", "lgtm", "yep will do",
     "ah ok cool thanks so much"],
)
def test_compositional_acknowledgements_are_dropped(gate, content):
    # These are combinations, not phrases. A phrase list would have to enumerate every
    # permutation; a filler-token vocabulary covers them by construction.
    assert gate.carries_fact(ep(content)) == (False, "ack_only")


@pytest.mark.parametrize(
    "content",
    ["sounds good, I moved to Berlin",
     "ok, my new employer is Globex",
     "thanks - also I'm allergic to shellfish",
     "sure thing, remind me that I live in Lisbon"],
)
def test_a_single_content_token_defeats_the_ack_rule(gate, content):
    # The recall bias in one test: one word outside the filler vocabulary and the turn
    # goes through, because a missed fact is unrecoverable and a spurious call is not.
    assert gate.carries_fact(ep(content))[0] is True


def test_long_all_filler_turn_still_passes(gate):
    # Past the token cap the "it is only pleasantries" assumption stops being safe.
    assert gate.carries_fact(ep("ok yes sure thanks fine good cool right"))[0] is True


@pytest.mark.parametrize(
    "content",
    ["What's the weather in Berlin?", "Where do I live?",
     "Do you remember my name? Are you sure?", "why?"],
)
def test_bare_questions_are_dropped(gate, content):
    ok, reason = gate.carries_fact(ep(content))
    assert not ok
    assert reason in ("question", "too_short")


def test_assistant_and_system_turns_are_dropped(gate):
    # Assistant text is generated, not observed. Treating it as evidence is how a store
    # ends up believing its own output.
    assert gate.carries_fact(ep("You live in Berlin.", role="assistant")) == (
        False, "assistant_turn")
    assert gate.carries_fact(ep("You are a helpful assistant.", role="system")) == (
        False, "system_turn")


def test_single_token_is_dropped(gate):
    assert gate.carries_fact(ep("Berlin")) == (False, "too_short")
    assert gate.carries_fact(ep("🙂")) == (False, "too_short")


@pytest.mark.parametrize(
    "content",
    ["私は東京に住んでいます。",   # Japanese
     "我住在北京",                # Chinese, 5 characters and a complete declarative
     "ฉันอาศัยอยู่ที่กรุงเทพ",           # Thai
     "저는 서울에 살아요"],        # Korean
)
def test_unspaced_scripts_are_not_mistaken_for_single_tokens(gate, content):
    # These scripts put no spaces between words, so a token count reads a whole sentence
    # as one token. Getting it wrong drops every fact those users ever state.
    assert gate.carries_fact(ep(content)) == (True, "has_declarative")


@pytest.mark.parametrize("content", ["私", "。", "、", "🙂", "…", "！"])
def test_short_non_latin_fragments_are_still_rejected(gate, content):
    # The negative control on the character floor: relaxing it for unspaced scripts must
    # not turn the gate into a pass-through for punctuation and single glyphs.
    assert gate.carries_fact(ep(content))[0] is False


def test_latin_diacritics_were_never_the_problem(gate):
    assert gate.carries_fact(ep("Je habite à Berlin")) == (True, "has_declarative")


def test_non_latin_text_cannot_be_read_as_an_acknowledgement(gate):
    # The filler vocabulary is English; a CJK turn must never reach `ack_only`.
    assert gate.carries_fact(ep("わかりました"))[1] != "ack_only"


# --- positives: recall bias --------------------------------------------------

@pytest.mark.parametrize(
    "content",
    ["I live in Berlin.",
     "My name is Goldy.",
     "I moved to Lisbon last month.",
     "I no longer work at Acme.",
     "Ich wohne in München und arbeite bei Siemens.",
     "私は東京に住んでいます。",
     "The team standardized on Rust for the billing service."],
)
def test_declarative_turns_pass(gate, content):
    assert gate.carries_fact(ep(content)) == (True, "has_declarative")


def test_question_with_a_declarative_clause_passes(gate):
    # A false positive costs one call; a false negative loses the fact forever, so a
    # mixed turn must always go through.
    assert gate.carries_fact(ep("I live in Berlin. What's good to eat nearby?")) == (
        True, "has_declarative")


def test_ack_prefix_with_real_content_passes(gate):
    assert gate.carries_fact(ep("Thanks! By the way I moved to Lisbon."))[0] is True


def test_50kb_turn_passes_and_is_cheap(gate):
    big = "I reviewed the deployment logs and everything looked fine. " * 900
    assert len(big) > 50_000
    ok, reason = gate.carries_fact(ep(big))
    assert (ok, reason) == (True, "has_declarative")


def test_unicode_only_punctuation_is_not_a_fact(gate):
    assert gate.carries_fact(ep("…!!! ??? —"))[0] is False


# --- determinism -------------------------------------------------------------

def test_gate_is_deterministic(gate):
    turns = ["I live in Berlin.", "ok", "What time is it?", "", "Thanks a lot",
             "I work at Acme.", "🙂", "Ich wohne in München."]
    first = [gate.carries_fact(ep(t)) for t in turns]
    second = [SalienceGate().carries_fact(ep(t)) for t in turns]
    assert first == second


def test_scope_does_not_affect_the_decision(gate):
    a = ep("I live in Berlin.", scope=Scope("t", "u1"))
    b = ep("I live in Berlin.", scope=Scope("t", "u2", "agent", "sess"))
    assert gate.carries_fact(a) == gate.carries_fact(b)
