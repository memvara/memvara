"""FastExtractor: the zero-LLM tier.

The property under test is precision. Recall failures are cheap (the LLM tier picks the
turn up one call later); a wrong triple is a lie the store repeats for months. So roughly
half of these tests assert that nothing is emitted.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from memvara.schema import Cardinality, PredicateRegistry
from memvara.types import Derivation, Episode, Scope, utcnow
from memvara.write import FastExtractor
from memvara.write.fast import EXTRACTOR


@pytest.fixture()
def fast() -> FastExtractor:
    return FastExtractor(PredicateRegistry())


def triples(claims):
    return [(c.predicate, c.object, c.polarity) for c in claims]


def ep(content: str, role: str = "user", **kw) -> Episode:
    return Episode(content=content, role=role, **kw)


# --- the forms that are common and unambiguous -------------------------------

@pytest.mark.parametrize(
    "content,expected",
    [
        ("My name is Goldy.", [("name", "Goldy", 1)]),
        ("I live in Berlin.", [("lives_in", "Berlin", 1)]),
        ("I'm based in Lisbon.", [("lives_in", "Lisbon", 1)]),
        ("I moved to Lisbon last month.", [("lives_in", "Lisbon", 1)]),
        ("I work at Acme.", [("works_at", "Acme", 1)]),
        ("I work for Globex.", [("works_at", "Globex", 1)]),
        ("I prefer dark roast.", [("prefers", "dark roast", 1)]),
        ("I like coffee.", [("likes", "coffee", 1)]),
        ("I hate cilantro.", [("dislikes", "cilantro", 1)]),
        ("I'm allergic to peanuts.", [("allergic_to", "peanuts", 1)]),
        ("I speak Portuguese.", [("speaks", "Portuguese", 1)]),
        ("My pronouns are she/her.", [("pronouns", "she/her", 1)]),
        ("I'm vegan.", [("dietary_restriction", "vegan", 1)]),
    ],
)
def test_unambiguous_forms_extract(fast, content, expected):
    assert triples(fast.extract(ep(content))) == expected


@pytest.mark.parametrize(
    "content,expected",
    [
        ("I no longer work at Acme.", [("works_at", "Acme", -1)]),
        ("I used to work at Globex.", [("works_at", "Globex", -1)]),
        ("I don't work at Acme anymore.", [("works_at", "Acme", -1)]),
        ("I no longer live in Berlin.", [("lives_in", "Berlin", -1)]),
        ("I used to live in Berlin.", [("lives_in", "Berlin", -1)]),
        ("I no longer like coffee.", [("likes", "coffee", -1)]),
    ],
)
def test_retractions_get_negative_polarity(fast, content, expected):
    assert triples(fast.extract(ep(content))) == expected


def test_multiple_clauses_yield_multiple_claims(fast):
    got = triples(fast.extract(ep("My name is Goldy and I work at Acme, I live in Berlin.")))
    assert got == [("name", "Goldy", 1), ("works_at", "Acme", 1), ("lives_in", "Berlin", 1)]


def test_one_clause_can_carry_two_facts(fast):
    # "I work as a designer at Acme" is genuinely a title and an employer.
    assert triples(fast.extract(ep("I work as a designer at Acme."))) == [
        ("job_title", "designer", 1),
        ("works_at", "Acme", 1),
    ]


def test_adverbial_tails_are_stripped_so_the_slot_does_not_fragment(fast):
    # "Lisbon" and "Lisbon last month" would otherwise be two competing facts.
    a = fast.extract(ep("I moved to Lisbon last month."))[0]
    b = fast.extract(ep("I moved to Lisbon."))[0]
    assert a.object == b.object == "Lisbon"
    assert a.value_key == b.value_key


# --- precision: everything below must emit nothing ---------------------------

@pytest.mark.parametrize(
    "content",
    [
        "Where do I live?",
        "Do I work at Acme?",
        "Would I like coffee?",
        "If I lived in Berlin I would be closer.",
        "I think I live in Berlin.",
        "She said I work at Acme.",
        "I might move to Lisbon.",
        "I'm not allergic to peanuts.",
        "I don't live in Berlin.",
        "I like it.",
        "I like to think about it.",
        "I live in the same place we were talking about during that long call earlier.",
    ],
)
def test_ambiguous_turns_emit_nothing(fast, content):
    assert fast.extract(ep(content)) == []


def test_coordinated_objects_are_handed_to_the_llm(fast):
    # Splitting "coffee and tea" correctly needs real parsing; guessing produces a fact
    # about a drink called "coffee and tea".
    assert fast.extract(ep("I like coffee and tea.")) == []


def test_assistant_turns_emit_nothing(fast):
    # Every rule binds "I" to the user, which is simply false on an assistant turn.
    assert fast.extract(ep("I live in Berlin.", role="assistant")) == []
    assert fast.extract(ep("I live in Berlin.", role="system")) == []


@pytest.mark.parametrize("content", ["", "   ", "\n\t ", "!!!", "…"])
def test_empty_and_junk_input_is_safe(fast, content):
    assert fast.extract(ep(content)) == []


# --- provenance and metadata -------------------------------------------------

def test_claims_carry_full_provenance(fast):
    scope = Scope("acme", "u1", "agent", "sess")
    ts = utcnow() - timedelta(days=3)
    e = ep("I live in Berlin.", scope=scope, ts=ts)
    claim = fast.extract(e)[0]

    assert claim.derivation is Derivation.FAST_PATH
    assert claim.extractor == EXTRACTOR
    assert claim.sources == [e.id]
    assert claim.scope == scope
    assert claim.subject == "user"
    # Valid time is when the user said it, not when we happened to process the batch.
    assert claim.valid_from == ts
    assert 0.0 < claim.confidence <= 1.0
    assert claim.text == "user lives in Berlin"


def test_memory_type_comes_from_the_registry(fast):
    assert fast.extract(ep("I prefer dark roast."))[0].memory_type.value == "procedural"
    assert fast.extract(ep("I live in Berlin."))[0].memory_type.value == "semantic"


def test_predicates_are_canonical(fast):
    # "moved to" is an alias of lives_in; if it stayed a distinct predicate the move
    # would never contradict the old city.
    claim = fast.extract(ep("I moved to Lisbon."))[0]
    assert claim.predicate == "lives_in"
    assert PredicateRegistry().spec(claim.predicate).cardinality is Cardinality.ONE


def test_unicode_objects_survive_intact(fast):
    assert triples(fast.extract(ep("I live in München."))) == [("lives_in", "München", 1)]
    assert triples(fast.extract(ep("I like 日本茶."))) == [("likes", "日本茶", 1)]


def test_duplicate_clauses_within_one_turn_collapse(fast):
    got = fast.extract(ep("I live in Berlin. I live in Berlin."))
    assert triples(got) == [("lives_in", "Berlin", 1)]


# --- robustness --------------------------------------------------------------

def test_50kb_turn_is_handled(fast):
    filler = "I reviewed the deployment logs and everything looked fine. " * 850
    content = filler + " I live in Berlin."
    assert len(content) > 50_000
    assert triples(fast.extract(ep(content))) == [("lives_in", "Berlin", 1)]


def test_extraction_is_deterministic(fast):
    e = ep("My name is Goldy and I live in Berlin, I work at Acme.")
    first = triples(fast.extract(e))
    second = triples(FastExtractor(PredicateRegistry()).extract(e))
    assert first == second == [("name", "Goldy", 1), ("lives_in", "Berlin", 1),
                               ("works_at", "Acme", 1)]
