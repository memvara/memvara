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


# --- contact directives ------------------------------------------------------
#
# The one rule family here that is second person and imperative rather than first-person
# declarative, and the reason `_HAS_SUBJECT` is no longer a first-person filter. It is
# also the family where the value is not in the text: "ring me" means the phone and does
# not contain the word, so the object is fixed by which pattern matched.


@pytest.mark.parametrize(
    "content,expected",
    [
        ("Please ring me.", [("contact_preference", "phone", 1)]),
        ("Call me.", [("contact_preference", "phone", 1)]),
        ("And please do ring.", [("contact_preference", "phone", 1)]),
        ("Email me.", [("contact_preference", "email", 1)]),
        ("Email from now on please.", [("contact_preference", "email", 1)]),
        ("Text me from now on.", [("contact_preference", "text", 1)]),
        ("Don't email me.", [("contact_preference", "email", -1)]),
        ("Can you stop ringing me.", [("contact_preference", "phone", -1)]),
        ("Please stop emailing me.", [("contact_preference", "email", -1)]),
    ],
)
def test_a_contact_directive_records_the_channel_not_the_verb(fast, content, expected):
    assert triples(fast.extract(ep(content))) == expected


def test_the_channel_is_one_slot_so_a_reversal_supersedes_rather_than_accumulates():
    """`contact_preference` is `ONE` in `BUILTIN_PREDICATES`, and that is what makes
    "email from now on" close "please do ring" instead of leaving a store that believes
    both. A deployment that genuinely accepts two channels declares its own spec."""
    assert PredicateRegistry().spec("contact_preference").cardinality is Cardinality.ONE


@pytest.mark.parametrize("content", [
    # A verb with an object is an errand, not a standing instruction.
    "Please call the office.",
    "Email the invoice to my accountant.",
    "Ring the doorbell twice.",
    # Reported and hypothetical, exactly as for every other rule.
    "She said to ring me.",
    "If it breaks, call me.",
])
def test_a_one_off_request_is_not_a_standing_contact_preference(fast, content):
    assert fast.extract(ep(content)) == []


# --- values that contain the punctuation the clause splitter cuts on ----------


@pytest.mark.parametrize(
    "content,expected",
    [
        ("Everything comes to 41 Coldharbour Road, Lewes, BN7 2GT.",
         [("address", "41 Coldharbour Road, Lewes, BN7 2GT", 1)]),
        ("As of Friday everything comes to Bramble Cottage, Westmeston, BN6 8XA.",
         [("address", "Bramble Cottage, Westmeston, BN6 8XA", 1)]),
        ("My delivery address is 9 Mill Lane, Hove.",
         [("address", "9 Mill Lane, Hove", 1)]),
        ("Send it to 9 Mill Lane, Hove.", [("address", "9 Mill Lane, Hove", 1)]),
        ("Invoices go to 9 Mill Lane.", [("address", "9 Mill Lane", 1)]),
    ],
)
def test_an_address_survives_its_own_commas(fast, content, expected):
    """A postal address is one fact with commas in it, not four clauses.

    Run through the clause splitter, "41 Coldharbour Road, Lewes, BN7 2GT" becomes three
    fragments and a postcode, none of which is an address. That is why the address rules
    are evaluated on the whole sentence, before the split.
    """
    assert triples(fast.extract(ep(content))) == expected


def test_a_negated_delivery_instruction_extracts_nothing(fast):
    """"Ship them to Coldharbour Road, not the Yard" is the case that makes the
    whole-sentence tier need the negation guard as much as the clause tier does: the
    object runs to the end of the sentence, so a rule that fired here would store the
    address with ", not the Yard" welded onto it."""
    assert fast.extract(ep("Ship them to Coldharbour Road, not the Yard.")) == []


@pytest.mark.parametrize("content,expected", [
    ("07700 900 118.", [("phone", "07700 900 118", 1)]),
    ("+44 7700 900118", [("phone", "+44 7700 900118", 1)]),
    ("020 7946 0958", [("phone", "020 7946 0958", 1)]),
])
def test_a_bare_phone_number_is_how_anyone_answers_whats_your_number(fast, content,
                                                                     expected):
    assert triples(fast.extract(ep(content))) == expected


@pytest.mark.parametrize("content", [
    "HX2-4419-B.",          # a serial: letters
    "2026.",                # a year: four digits
    "79.",                  # a price
    "Thirty metres.",       # words
    "1 2 3 4 5 6 7 8 9 10 11",   # too many digits to be a number anyone dials
    "(020) 7946 0958",      # a bracketed area code: the string is not all number
    "My number is 07700 900 118.",   # not bare — a captured object, not this rule
])
def test_only_a_bare_number_of_dialable_length_is_read_as_a_phone(fast, content):
    """The digit count is the whole of the guard, and it is a lookahead over the entire
    string rather than a length: `HX2-4419-B` is the number off a power brick in the demo
    corpus and reads as an identifier to a person and to this rule alike."""
    assert [c for c in fast.extract(ep(content)) if c.predicate == "phone"] == []


# --- the clause splitter, widened --------------------------------------------


def test_a_coordinated_clause_with_its_own_subject_is_split_off(fast):
    """"I'm redoing the schedule and it wants the serial" is one fact and one aside.

    Read as a coordinated *object* the whole clause is rejected and the fact is lost,
    which is what happened while the split only recognised a following `i` or `my`.
    """
    assert triples(fast.extract(ep("I live in Hove and it is cold."))) == [
        ("lives_in", "Hove", 1)]


def test_a_genuinely_coordinated_object_is_still_refused(fast):
    """The other half of the same rule: nothing in the subject list can head an object,
    so "coffee and tea" is untouched and still rejected as two facts in one hat."""
    assert fast.extract(ep("I like coffee and tea.")) == []

