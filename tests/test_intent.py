"""Query-shape classification: the four classes, and the order they are checked in.

The classifier is a marker vocabulary, so the interesting cases are not the ones it gets
right by construction — the doctests carry those — but the ones where two vocabularies
both match and something has to win. Every test here is about a decision:

* **Time outranks relation.** "Who was my manager in 2023" is both. A wrong answer from
  the wrong instant is wrong in a way extra recall does not repair; a missed hop is an
  answer that is merely incomplete.
* **`who` is not enough on its own.** "Who did I meet on Tuesday" names a person and
  wants a row; routing every `who` into a walk would gate the graph leg on by far the
  most common interrogative in a personal store.
* **Not recognising something is `open`, never an error.** Both lookup legs abstain on a
  contentless query; this must not be the one stage that raises.
"""

import pytest

from memvara.retrieve.intent import (
    Intent,
    LOOKUP_MARKERS,
    RELATIONAL_MARKERS,
    TEMPORAL_MARKERS,
    classify,
    weights,
)


@pytest.mark.parametrize("query, intent", [
    ("what is my name", Intent.LOOKUP),
    ("where do I live", Intent.LOOKUP),
    ("which plan am I on", Intent.LOOKUP),
    ("when did I move", Intent.TEMPORAL),
    ("what did I do before the move", Intent.TEMPORAL),
    ("what plan am I on now", Intent.TEMPORAL),
    ("what was true in 2023", Intent.TEMPORAL),
    ("who does Alice report to", Intent.RELATIONAL),
    ("how is Acme connected to Tallinn", Intent.RELATIONAL),
    ("tell me everything about the migration", Intent.OPEN),
    ("summarise the project", Intent.OPEN),
])
def test_the_four_classes(query, intent):
    assert classify(query) is intent


def test_time_outranks_relation_when_a_query_is_both():
    """`valid_at` is the axis a wrong answer here is wrong on."""
    assert classify("who was my manager in 2023") is Intent.TEMPORAL
    assert classify("who is my manager") is Intent.RELATIONAL


def test_who_alone_is_a_lookup_and_who_beside_a_relation_is_not():
    assert classify("who founded Acme") is Intent.LOOKUP
    assert classify("who is Alice's employer") is Intent.RELATIONAL


def test_a_serial_number_is_not_a_year():
    """The year pattern is narrow on purpose.

    `4419` is the number off a power brick in `demo/scenario.py`, and a store full of
    part numbers would otherwise route every lookup about one through the time leg.
    """
    assert classify("what is serial 4419") is Intent.LOOKUP
    assert classify("what happened in 2026") is Intent.TEMPORAL


def test_between_and_is_read_as_a_question_about_a_path():
    """`paths_between`, written in English. `between` alone is not enough for it."""
    assert classify("what is between the two readings") is Intent.RELATIONAL
    assert classify("what connects Alice and Acme") is Intent.RELATIONAL


def test_an_unreadable_query_is_open_rather_than_an_error():
    for query in ("", "   ", "?!", "***"):
        assert classify(query) is Intent.OPEN


def leg_weights(intent):
    return weights(intent, vector=1.0, lexical=1.0, graph=0.8, temporal=0.4)


def test_a_lookup_pays_for_neither_extra_leg_and_the_others_pay_for_one_each():
    """The gates are zero weights, checked *before* the walk and before the time query.

    A multiplier applied afterwards would buy the same ranking and none of the latency
    saving, which is the whole reason the table has zeroes in it. And the two are never
    on together: a question about a chain and a question about an instant are different
    questions, and running the graph leg on the second would zero every claim it did not
    reach on a question that was never about a join.
    """
    assert leg_weights(Intent.LOOKUP)[2:] == (0.0, 0.0)
    assert leg_weights(Intent.RELATIONAL)[2:] == (0.8, 0.0)
    assert leg_weights(Intent.TEMPORAL)[2:] == (0.0, 0.4)
    assert leg_weights(Intent.OPEN)[2:] == (0.8, 0.4), (
        "open is the one shape that could be either, so it runs both and lets fusion "
        "decide"
    )


def test_configured_weights_are_scaled_rather_than_replaced():
    """A deployment that tuned `w_vector` keeps its tuning.

    A table of absolute weights would discard every configured value silently, which is
    the failure mode of every routing layer that ends up switched off in production.
    """
    for intent in Intent:
        vector, lexical, _graph, _temporal = weights(
            intent, vector=0.3, lexical=2.5, graph=0.0, temporal=0.0)
        assert (vector, lexical) == (0.3, 2.5)


def test_the_vocabularies_do_not_claim_words_another_stage_already_decided():
    """A marker listed in two vocabularies is a race the ordering has settled.

    Leaving it in both reads as though the later check could win, and the next person to
    add a word has no way to tell which list is authoritative for it.
    """
    assert not (LOOKUP_MARKERS & TEMPORAL_MARKERS)
    assert not (LOOKUP_MARKERS & RELATIONAL_MARKERS)
    assert not (TEMPORAL_MARKERS & RELATIONAL_MARKERS)


@pytest.mark.parametrize("query", [
    "what was I working on last summer",
    "what did we ship in Q3",
    "what happened in the fourth quarter",
    "where was I living in the winter",
    "what did I do in spring",
])
def test_seasons_and_quarters_are_time_words(query: str) -> None:
    """Months were in the marker set and the coarser units were not.

    So "what did I do in March" routed as temporal and "what did I do in spring" routed
    as a lookup — answered with the leg that ranks on *when* switched off, on a question
    that is about nothing else. `spring` is also a coil and `fall` is also a verb, which
    is the objection; it is outweighed because this set only weights legs, and the
    temporal leg abstains anyway when nothing in scope is near the anchor. That is
    exactly the shape a misread `fall` produces.
    """
    assert classify(query) is Intent.TEMPORAL


def test_adding_the_coarse_units_did_not_pull_in_ordinary_questions() -> None:
    """The check on the other side: a relation question and a lookup stay where they
    were, so the widened vocabulary bought recall without spending precision."""
    assert classify("who does Alice report to") is Intent.RELATIONAL
    assert classify("what is my name") is Intent.LOOKUP
