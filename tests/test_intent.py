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


def test_two_predicates_in_one_question_is_a_chain_and_one_is_a_lookup() -> None:
    """The vocabulary gap the hand-written marker list cannot close.

    "Which city is the company Ada works at based in?" names `works_at` and `lives_in`,
    shares no word with `RELATIONAL_MARKERS`, and is a two-hop question by construction.
    Adding "works at" to the list would fit the classifier to the corpus that needed it —
    the module comment says so — so the signal comes from the store's own registry
    instead, and the rule is structural: one predicate is a question about one slot, two
    is a question that passes through one fact to reach another.

    The lookups are the half that matters more. `name`, `city` and `birthday` are all
    predicates, so a rule that fired on *any* predicate would route the purest lookup
    there is into the graph leg.
    """
    from memvara.schema import PredicateRegistry

    registry = PredicateRegistry()
    chains = ["Which city is the company Ada works at based in?",
              "what city does the company Bruno works at operate from"]
    lookups = ["what is my name", "where do I live", "what languages do I speak",
               "what is my email"]

    for query in chains:
        assert classify(query, registry) is Intent.RELATIONAL, query
    for query in lookups:
        assert classify(query, registry) is not Intent.RELATIONAL, query


def test_the_rule_is_additive_and_off_without_a_registry() -> None:
    """No registry, no change: `classify(query)` answers exactly what it used to.

    The parameter is optional because `classify` is public and callers exist that have no
    registry to hand. Nothing that was relational stops being relational either — the new
    rule is another way to reach `RELATIONAL`, never a way to leave it.
    """
    from memvara.schema import PredicateRegistry

    registry = PredicateRegistry()
    assert classify("Which city is the company Ada works at based in?") is Intent.LOOKUP
    assert classify("who does Alice report to", registry) is Intent.RELATIONAL
    assert classify("what happened between 2019 and 2021", registry) is Intent.TEMPORAL


def test_predicates_are_matched_as_phrases_not_as_tokens() -> None:
    """`lives_in` splits into `lives` and `in`, and `in` is in most English questions.

    A token index would make almost every query look like it named two predicates, which
    would route the whole workload into the graph leg — the opposite failure to the one
    being fixed, and harder to notice because it only shows up as latency.
    """
    from memvara.retrieve.intent import predicate_refs
    from memvara.schema import PredicateRegistry

    registry = PredicateRegistry()
    assert predicate_refs("what is in the box and in the bag", registry) == set()
    assert predicate_refs("she lives in Lisbon", registry) == {"lives_in"}


def test_aliases_fold_before_they_are_counted() -> None:
    """Otherwise a question that spells one relation two ways looks like a chain.

    `based_in` and `located_in` are both `lives_in`; counting them separately would make
    "is the office located in the city it is based in" a two-predicate question about one
    predicate.
    """
    from memvara.retrieve.intent import predicate_refs
    from memvara.schema import PredicateRegistry

    registry = PredicateRegistry()
    assert predicate_refs("is it based in or located in Lisbon", registry) == {"lives_in"}


# --- derived relation terms: the chain a question names without naming a predicate ---


def test_a_derived_term_is_a_chain_even_though_it_names_no_predicate() -> None:
    """"Maternal grandfather" is `mother` composed with `father` and is neither of them.

    Every rule in this module counts predicates a question says out loud, so this shape
    was invisible to all of them: on 2WikiMultihopQA's `inference` family the gate ran the
    walk on none of the 1,549 questions. `grandfather` is not a synonym for `father` and
    no string match gets from one to the other — what is needed is the fact that the term
    *is* a composition, which only a model can supply and which is therefore supplied once
    per vocabulary rather than once per query.
    """
    from memvara.retrieve.compose import names_derived

    terms = {"grandfather", "father-in-law", "uncle"}
    assert names_derived("Who is the paternal grandfather of Reginald I?", terms)
    assert names_derived("Who is Marie Luisa's father-in-law?", terms)
    assert not names_derived("Where was the director of film Nagarahole born?", terms)


def test_no_terms_means_the_behaviour_of_every_release_before_this_one() -> None:
    """The feature is opt-in and its absence is not a degradation.

    A backend without `compose_relations` yields no terms, `names_derived` is False for
    every query, and the gate keeps the rule it had. That is the state this shipped in
    before the acquisition existed, so a store with no model is no worse off than it was.
    """
    from memvara.retrieve.compose import names_derived

    assert not names_derived("Who is the paternal grandfather of Reginald I?", set())
    assert not names_derived("anything at all", ())


def test_the_model_is_filtered_rather_than_trusted() -> None:
    """Four ways an answer is wrong, and none of them raises.

    This is an optional enrichment on a path that works without it, so a model having a
    bad day should cost the questions it would have helped and never the search that was
    going to succeed anyway.
    """
    from memvara.retrieve.compose import acquire

    class Model:
        def compose_relations(self, predicates):
            return {
                "grandfather": 2,            # kept
                "stepmother": 1,             # arity 1 — the store has a predicate for it
                "father": 3,                 # is itself a predicate; a contradiction
                "the person who married": 2,  # a phrase, not a relation name
                "uncle": True,               # bool is not an arity
                7: 2,                        # not a term
            }

    assert acquire(Model(), ["father", "mother", "spouse"]) == {"grandfather"}


def test_a_backend_that_cannot_compose_costs_nothing() -> None:
    """No method, or a method that raises — including the network. An enrichment that
    raised into `Memvara.__init__` would make an optional feature a startup dependency."""
    from memvara.retrieve.compose import acquire

    class Silent:
        pass

    class Broken:
        def compose_relations(self, predicates):
            raise RuntimeError("no route to host")

    class Nonsense:
        def compose_relations(self, predicates):
            return ["grandfather"]

    for backend in (Silent(), Broken(), Nonsense()):
        assert acquire(backend, ["father"]) == frozenset()
