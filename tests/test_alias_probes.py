"""Alias-stamped seeds: what a *probe* is allowed to reach.

`entities.py` folds a surface form onto an identity deterministically, and separately
learns that two folds are one entity ("Big Blue" is IBM). A written claim gets the
learned answer pinned into its `meta` at write time (`Reconciler._stamp`). A probe — the
string a caller hands `neighborhood()` or `history()` — is not a written claim and
carries no stamp, so it only ever got the deterministic fold and missed the other half
of its own entity.

The property under test throughout is that resolving a probe **widens** what is found
and never narrows it: a merge does not rewrite the past (that is `backfill_entities`,
dated and opt-in), so after "Big Blue" is learned to be IBM the store holds claims under
`big blue` *and* under `ibm`, and a probe spelled either way has to reach both.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memvara import Memvara, NullLLM
from memvara.embed import HashingEmbedder
from memvara.entities import EntityRegistry
from memvara.retrieve.traverse import GraphTraverser
from memvara.schema import PredicateRegistry
from memvara.store import SQLiteStore
from memvara.types import Claim, Scope, owner_key

ALICE = Scope("acme", "alice")
T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
T1 = datetime(2024, 6, 1, tzinfo=timezone.utc)


@pytest.fixture()
def mem():
    with Memvara(llm=NullLLM(), embedder=HashingEmbedder(dim=64), tenant="acme",
                 user="alice") as m:
        yield m


def learn(mem, canonical, surface, scope=ALICE):
    """Record the merge a model (or an operator) decided on, in the live registry.

    The same object `Reconciler._stamp` resolves against, which is the point: probe and
    stamp have to agree by construction rather than by being kept in step.
    """
    return mem.writer.reconciler.entities.learn_alias(
        owner_key(scope), canonical, surface)


def objects(paths):
    return sorted(c.object for p in paths for c in p.claims)


# --- neighborhood -------------------------------------------------------------

def test_neighborhood_reaches_claims_folded_onto_the_learned_canonical(mem):
    """The headline defect: `neighborhood("Big Blue")` folded to `big blue` and never
    looked at the alias, so every claim about IBM was invisible to it."""
    mem.remember("IBM", "headquartered_in", "Armonk")
    learn(mem, "IBM", "Big Blue")
    assert objects(mem.neighborhood("Big Blue")) == ["Armonk"]


def test_neighborhood_reaches_claims_carrying_the_write_time_stamp(mem):
    """The same miss one step further on: a claim written *after* the merge is stamped
    `ibm` in its own meta, so it is not even under the fold of the words it was written
    with."""
    mem.remember("IBM", "headquartered_in", "Armonk")
    learn(mem, "IBM", "Big Blue")
    mem.remember("Big Blue", "listed_on", "NYSE")
    stamped = [c for c in mem.get_all() if c.object == "NYSE"][0]
    assert stamped.subject_key == "ibm"          # stamped, not folded
    assert objects(mem.neighborhood("Big Blue")) == ["Armonk", "NYSE"]


def test_neighborhood_still_finds_claims_written_before_the_merge(mem):
    """No narrowing. A merge does not re-key the past, so the pre-merge claims are still
    under `big blue` — resolving the probe *to* `ibm` instead of the fold would have
    traded one half of the entity for the other."""
    mem.remember("Big Blue", "listed_on", "NYSE")     # keyed `big blue`
    mem.remember("IBM", "headquartered_in", "Armonk")  # keyed `ibm`
    learn(mem, "IBM", "Big Blue")
    assert objects(mem.neighborhood("Big Blue")) == ["Armonk", "NYSE"]


def test_neighborhood_probed_by_the_canonical_sees_the_same_entity(mem):
    """Symmetry. Once the two names are one entity, which spelling was used to ask
    cannot change the answer, or `neighborhood` would report a different graph for
    "IBM" than for "Big Blue"."""
    mem.remember("Big Blue", "listed_on", "NYSE")
    mem.remember("IBM", "headquartered_in", "Armonk")
    learn(mem, "IBM", "Big Blue")
    assert objects(mem.neighborhood("IBM")) == objects(mem.neighborhood("Big Blue"))


def test_a_probe_with_nothing_learned_about_it_is_unchanged(mem):
    mem.remember("IBM", "headquartered_in", "Armonk")
    assert objects(mem.neighborhood("IBM")) == ["Armonk"]
    assert mem.neighborhood("Big Blue") == []


def test_an_unfoldable_probe_keeps_its_raw_text(mem):
    """`default_entity` refuses to collapse "..." onto the empty key, and probe
    resolution must not undo that by folding it to "" and matching every retraction."""
    mem.remember("...", "listed_on", "NYSE")
    assert objects(mem.neighborhood("...")) == ["NYSE"]
    assert mem.neighborhood("   ") == []


# --- history ------------------------------------------------------------------

def test_history_reaches_the_slot_the_learned_canonical_keys(mem):
    mem.remember("IBM", "headquartered_in", "Armonk")
    learn(mem, "IBM", "Big Blue")
    assert [c.object for c in mem.history("Big Blue", "headquartered_in")] == ["Armonk"]


def test_history_merges_both_slots_oldest_first(mem):
    """Two fact keys are two slots, and two slots concatenated are not a timeline.

    Deliberately written so that concatenation gives the *wrong* answer: the older
    version lives in the slot that is queried second, so a merge that only appended
    would report the timeline backwards.
    """
    mem.remember("IBM", "headquartered_in", "Armonk", recorded_at=T0)    # slot `ibm`
    mem.remember("Big Blue", "headquartered_in", "Endicott",
                 recorded_at=T1)                                          # slot `big blue`
    learn(mem, "IBM", "Big Blue")
    rows = mem.history("Big Blue", "headquartered_in")
    assert [c.object for c in rows] == ["Armonk", "Endicott"]
    assert [c.recorded_at for c in rows] == sorted(c.recorded_at for c in rows)


def test_history_of_an_unaliased_subject_is_unchanged(mem):
    mem.remember("user", "lives_in", "Berlin")
    mem.remember("user", "lives_in", "Lisbon")
    assert [c.object for c in mem.history("user", "lives_in")] == ["Berlin", "Lisbon"]


# --- ambiguity ----------------------------------------------------------------

def test_a_probe_that_names_several_learned_keys_takes_all_of_them(mem):
    """The rule, stated once: a probe resolves to *every* key that names its entity —
    the deterministic fold, the canonical, and every sibling alias — and never picks
    between them. Refusing when there is more than one would make the answer depend on
    how many spellings had happened to be merged, which is not a fact about the world."""
    mem.remember("Big Blue", "listed_on", "NYSE")
    mem.remember("IBM", "headquartered_in", "Armonk")
    mem.remember("Armonk Giant", "founded_in", "Endicott")
    learn(mem, "IBM", "Big Blue")
    learn(mem, "IBM", "Armonk Giant")
    assert objects(mem.neighborhood("Big Blue")) == ["Armonk", "Endicott", "NYSE"]


def test_an_edge_between_two_names_of_one_entity_is_a_self_loop(mem):
    """`_edges` already drops `subject == object`. After a merge, an edge from `big
    blue` to `ibm` is that same self-loop spelled two ways, and walking it would return
    every one-hop path twice — once direct, once through the entity's other name."""
    mem.remember("Big Blue", "renamed_as", "IBM")
    mem.remember("IBM", "headquartered_in", "Armonk")
    learn(mem, "IBM", "Big Blue")
    assert objects(mem.neighborhood("Big Blue")) == ["Armonk"]


# --- scope --------------------------------------------------------------------

def test_an_alias_learned_in_one_tenant_does_not_fold_another_tenants_probe():
    """Identity is owner-scoped — `owner_key` is tenant plus user — and a probe is
    resolved under the *reader's* owner, the same one `Reconciler._stamp` wrote with.
    One tenant deciding two names are one thing must not decide it for another."""
    with Memvara(llm=NullLLM(), embedder=HashingEmbedder(dim=64)) as mem:
        mem.remember("IBM", "headquartered_in", "Yorktown", tenant="t_a", user="alice")
        learn(mem, "IBM", "Big Blue", scope=Scope("t_a", "alice"))

        mem.remember("Big Blue", "listed_on", "NYSE", tenant="t_b", user="alice")
        mem.remember("IBM", "headquartered_in", "Armonk", tenant="t_b", user="alice")
        seen = mem.neighborhood("Big Blue", tenant="t_b", user="alice")
        assert objects(seen) == ["NYSE"]
        assert [c.object for c in mem.history("Big Blue", "headquartered_in",
                                              tenant="t_b", user="alice")] == []


def test_an_alias_learned_by_one_user_does_not_fold_a_siblings_probe():
    with Memvara(llm=NullLLM(), embedder=HashingEmbedder(dim=64), tenant="acme") as mem:
        mem.remember("IBM", "headquartered_in", "Yorktown", user="alice")
        learn(mem, "IBM", "Big Blue")

        mem.remember("Big Blue", "listed_on", "NYSE", user="bob")
        mem.remember("IBM", "headquartered_in", "Armonk", user="bob")
        assert objects(mem.neighborhood("Big Blue", user="bob")) == ["NYSE"]
        assert objects(mem.neighborhood("Big Blue", user="alice")) == ["Yorktown"]


# --- the registry lookup itself -----------------------------------------------

def test_probe_keys_of_an_unknown_surface_is_the_fold_alone():
    assert EntityRegistry().probe_keys("acme\x1falice", "Big Blue") == ("big blue",)


def test_probe_keys_never_registers_what_it_was_asked_about():
    """A read must not teach the registry. `resolve` registers a novel fold on purpose;
    doing that here would let anyone populate an owner's entity table by querying it."""
    reg = EntityRegistry()
    reg.probe_keys("acme\x1falice", "Big Blue")
    assert reg.all("acme\x1falice") == []


def test_probe_keys_puts_the_deterministic_fold_first():
    reg = EntityRegistry()
    reg.resolve("acme\x1falice", "IBM")
    reg.learn_alias("acme\x1falice", "IBM", "Big Blue")
    assert reg.probe_keys("acme\x1falice", "Big Blue") == ("big blue", "ibm")
    assert reg.probe_keys("acme\x1falice", "IBM") == ("ibm", "big blue")


# --- the traverser on its own -------------------------------------------------

def test_between_resolves_both_ends_of_the_question():
    """`GraphTraverser.between` takes the same widening on either end. Exercised here
    directly because `Memvara.paths_between` does not pass it yet."""
    store = SQLiteStore(":memory:")
    try:
        for subject, predicate, obj in (
            ("Big Blue", "employs", "Dana"),
            ("Dana", "lives_in", "Armonk"),
        ):
            store.put_claim(Claim(subject=subject, predicate=predicate, object=obj,
                                  scope=ALICE))
        walker = GraphTraverser(store, PredicateRegistry())
        assert walker.between("IBM", "Armonk", ALICE) == []
        found = walker.between("IBM", "Armonk", ALICE, source_keys=("ibm", "big blue"))
        assert [p.render() for p in found] == [
            "Big Blue -employs-> Dana -lives_in-> Armonk"]
    finally:
        store.close()


def test_a_target_that_is_the_source_under_another_name_is_not_a_path():
    """"How is IBM connected to Big Blue" is a question about one entity, and the answer
    is not the loop through everything it touches.

    The early return in `_walk` is not the only thing producing this: the cycle guard
    reaches the same `[]` on its own, exactly as the pre-merge `goal == seed` check was
    already redundant with it. Removing either still passes; this pins the behaviour, and
    the guard is there so the rule is stated rather than emergent.
    """
    store = SQLiteStore(":memory:")
    try:
        store.put_claim(Claim(subject="Big Blue", predicate="employs", object="Dana",
                              scope=ALICE))
        walker = GraphTraverser(store, PredicateRegistry())
        assert walker.between("IBM", "Big Blue", ALICE,
                              source_keys=("ibm", "big blue")) == []
    finally:
        store.close()


def test_the_traverser_never_lets_a_caller_replace_the_deterministic_fold():
    """`entity_keys` may only add. A caller passing the canonical alone still walks from
    the fold of the words it actually asked with, so a merge cannot trade the claims
    written before it for the ones written after."""
    store = SQLiteStore(":memory:")
    try:
        for subject, obj in (("Big Blue", "NYSE"), ("IBM", "Armonk")):
            store.put_claim(Claim(subject=subject, predicate="listed_on", object=obj,
                                  scope=ALICE))
        walker = GraphTraverser(store, PredicateRegistry())
        found = walker.neighborhood("Big Blue", ALICE, entity_keys=("ibm",))
        assert objects(found) == ["Armonk", "NYSE"]
    finally:
        store.close()
