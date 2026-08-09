"""Predicate registry: normalization, cardinality, and runtime schema learning."""

import pytest

from memvara.schema import (
    BUILTIN_PREDICATES,
    Cardinality,
    PredicateRegistry,
    PredicateSpec,
    Volatility,
)
from memvara.types import MemoryType


@pytest.fixture()
def reg() -> PredicateRegistry:
    return PredicateRegistry()


# --- Normalization ----------------------------------------------------------

@pytest.mark.parametrize(
    "surface,canonical",
    [
        ("resides_in", "lives_in"),
        ("based_in", "lives_in"),
        ("moved_to", "lives_in"),
        ("employed_at", "works_at"),
        ("works_for", "works_at"),
        ("is_named", "name"),
        ("dob", "born_on"),
        ("hates", "dislikes"),
    ],
)
def test_aliases_collapse_onto_one_canonical_slot(reg, surface, canonical):
    """Without this, 'lives_in' and 'resides_in' are different slots and the
    contradiction between them is invisible."""
    assert reg.normalize(surface) == canonical


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Lives In", "lives_in"),
        ("  WORKS AT!!  ", "works_at"),
        ("works---at", "works_at"),
        ("__lives_in__", "lives_in"),
        ("Works At", "works_at"),
    ],
)
def test_surface_forms_slugify_before_lookup(reg, raw, expected):
    assert reg.normalize(raw) == expected


def test_normalize_passes_through_unknown_predicates(reg):
    assert reg.normalize("enjoys_hiking_at_dawn") == "enjoys_hiking_at_dawn"


def test_normalize_handles_degenerate_input(reg):
    assert reg.normalize("") == ""
    assert reg.normalize("!!!") == ""
    assert reg.normalize("   ") == ""


# --- Cardinality ------------------------------------------------------------

@pytest.mark.parametrize("p", ["lives_in", "works_at", "name", "born_on", "mood"])
def test_single_valued_predicates_are_functional(reg, p):
    assert reg.functional(p)


@pytest.mark.parametrize("p", ["likes", "dislikes", "speaks", "allergic_to", "goal"])
def test_multi_valued_predicates_accumulate(reg, p):
    assert not reg.functional(p)


def test_functional_check_follows_aliases(reg):
    assert reg.functional("resides_in")
    assert reg.functional("employed_at")


def test_unknown_predicates_default_to_multi_valued(reg):
    """The conservative direction: keeping two facts degrades ranking, dropping a true
    one destroys information. Errors must fall on the recoverable side."""
    assert not reg.functional("some_predicate_nobody_declared")
    assert not reg.known("some_predicate_nobody_declared")


def test_spec_for_unknown_predicate_is_synthesized_not_registered(reg):
    before = len(reg)
    spec = reg.spec("brand_new_thing")
    assert spec.name == "brand_new_thing"
    assert spec.cardinality is Cardinality.MANY
    assert len(reg) == before, "reading a spec must not mutate the registry"


# --- Volatility / half-life -------------------------------------------------

def test_half_lives_are_ordered_by_volatility(reg):
    assert reg.half_life_days("born_in") > reg.half_life_days("lives_in")
    assert reg.half_life_days("lives_in") > reg.half_life_days("working_on")


def test_static_predicates_effectively_never_decay(reg):
    assert reg.half_life_days("born_on") > 365 * 50


def test_fast_predicates_decay_within_days(reg):
    assert reg.half_life_days("working_on") <= 30


# --- Memory typing ----------------------------------------------------------

def test_behavioral_preferences_are_procedural(reg):
    assert reg.spec("prefers").memory_type is MemoryType.PROCEDURAL
    assert reg.spec("never_do").memory_type is MemoryType.PROCEDURAL


def test_transient_state_is_episodic(reg):
    assert reg.spec("working_on").memory_type is MemoryType.EPISODIC


# --- Learning ---------------------------------------------------------------

def test_learning_installs_a_spec_permanently(reg):
    assert not reg.functional("drives_car")
    reg.learn("drives_car", Cardinality.ONE, Volatility.SLOW)
    assert reg.functional("drives_car")
    assert reg.known("drives_car")
    assert reg.spec("drives_car").learned


def test_learned_predicates_can_carry_aliases(reg):
    reg.learn("drives_car", Cardinality.ONE, aliases=("owns_vehicle", "car_is"))
    assert reg.normalize("owns_vehicle") == "drives_car"
    assert reg.functional("car_is")


def test_learning_normalizes_the_name_it_registers(reg):
    reg.learn("Drives Car", Cardinality.ONE)
    assert reg.known("drives_car")


def test_learning_can_override_a_builtin(reg):
    assert not reg.functional("likes")
    reg.learn("likes", Cardinality.ONE)
    assert reg.functional("likes")


# --- Registry integrity -----------------------------------------------------

def test_builtin_predicate_names_are_unique():
    names = [s.name for s in BUILTIN_PREDICATES]
    assert len(names) == len(set(names))


def test_no_alias_shadows_a_canonical_name():
    """An alias pointing at a different predicate's canonical name would silently
    reroute every claim using it."""
    canonical = {s.name for s in BUILTIN_PREDICATES}
    for spec in BUILTIN_PREDICATES:
        for alias in spec.aliases:
            assert alias not in canonical, f"{alias} is both an alias and a canonical name"


def test_no_alias_is_claimed_by_two_predicates():
    seen: dict[str, str] = {}
    for spec in BUILTIN_PREDICATES:
        for alias in spec.aliases:
            assert alias not in seen, f"{alias} claimed by {seen.get(alias)} and {spec.name}"
            seen[alias] = spec.name


def test_registry_can_be_built_empty(reg):
    empty = PredicateRegistry(specs=())
    assert len(empty) == 0
    assert not empty.functional("lives_in")
    assert empty.normalize("lives_in") == "lives_in"


def test_custom_registry_is_isolated_from_builtins():
    custom = PredicateRegistry(specs=(PredicateSpec("owns", Cardinality.ONE),))
    assert custom.functional("owns")
    assert not custom.known("lives_in")


# --- Resolution: the pre-pass that keeps the schema from exploding ----------
# The full simulation lives in test_predicates.py; these are the individual rules.


def test_derivational_matching_folds_a_word_family(reg):
    """`employer`, `employed_by` and `employment` are one relation wearing three
    suffixes. Only reached when nothing stricter matched, and only when exactly one
    predicate claims the stem."""
    assert reg.resolve("employment").method == "derivational"
    assert reg.normalize("employment") == "works_at"


def test_resolution_reports_how_it_decided(reg):
    """The method is not decoration: "morphological" and "novel" are the difference
    between a free fold and a model call, so it has to be inspectable."""
    assert [reg.resolve(p).method for p in
            ("", "works_at", "employer", "employer_name", "employment", "brand_new")] == [
        "empty", "canonical", "alias", "morphological", "derivational", "novel"]


def test_an_alias_that_slugifies_to_nothing_is_ignored():
    """Punctuation-only aliases come from real model output, and indexing one would
    make every unparseable predicate resolve to whichever spec declared it."""
    reg = PredicateRegistry(specs=(
        PredicateSpec("owns", Cardinality.ONE, aliases=("!!!", "possesses")),
    ))
    assert reg.normalize("!!!") == ""
    assert reg.normalize("possesses") == "owns"


def test_nearest_has_nothing_to_go_on_for_an_empty_surface(reg):
    assert reg.nearest("") is None
    assert reg.nearest("!!!") is None


@pytest.mark.parametrize("surface", ["works_at", "employer", "", "!!!"])
def test_aliasing_a_form_the_predicate_already_owns_is_a_no_op(reg, surface):
    """Re-recording an alias must not grow the tuple every time it is seen."""
    before = reg.spec("works_at")
    assert reg.learn_alias("works_at", surface) is before


def test_learn_alias_follows_an_alias_to_the_canonical(reg):
    reg.learn_alias("employer", "paycheck_source")
    assert reg.normalize("paycheck_source") == "works_at"


def test_a_replaced_spec_does_not_leave_its_old_aliases_behind(reg):
    """Specs are swapped wholesale, so an index patched in place would keep serving a
    predecessor's aliases forever — silently routing claims into a dead slot."""
    reg.learn("drives_car", Cardinality.ONE, aliases=("owns_vehicle",))
    assert reg.normalize("owns_vehicle") == "drives_car"
    reg.learn("drives_car", Cardinality.ONE, aliases=("car_is",))
    assert reg.normalize("owns_vehicle") == "owns_vehicle"
    assert reg.normalize("car_is") == "drives_car"


def test_the_learned_cap_is_configurable_and_counts_only_learned_specs():
    reg = PredicateRegistry(max_learned=2)
    assert reg.learned_count == 0 and not reg.at_capacity
    reg.learn("first_new", Cardinality.MANY)
    reg.learn("second_new", Cardinality.MANY)
    assert reg.at_capacity
    reg.learn("third_new", Cardinality.MANY)
    assert not reg.known("third_new")
    # Re-learning something already installed is a correction, not growth.
    reg.learn("first_new", Cardinality.ONE)
    assert reg.functional("first_new")


def test_the_prompt_vocabulary_puts_declared_predicates_first(reg):
    reg.learn("aaa_learned_sorts_first_alphabetically", Cardinality.MANY)
    vocabulary = reg.prompt_vocabulary()
    assert vocabulary[0] == "allergic_to"
    assert vocabulary[-1] == "aaa_learned_sorts_first_alphabetically"
    assert reg.prompt_vocabulary(limit=3) == vocabulary[:3]


def test_candidates_are_bounded_and_ranked_by_overlap(reg):
    assert reg.candidates("previous_employer")[0] == "works_at"
    assert len(reg.candidates("anything", limit=5)) == 5
    assert reg.candidates("previous_employer") == reg.candidates("previous_employer")


def test_a_declared_predicate_outranks_a_learned_one_on_a_tie(reg):
    """Otherwise the first novel phrasing a deployment ever saw becomes the attractor
    every later phrasing collapses into, purely because it sorts earlier."""
    reg.learn("aa_employer_guess", Cardinality.ONE)
    assert reg.nearest("previous_employer") == "works_at"
    assert reg.candidates("previous_employer")[0] == "works_at"
