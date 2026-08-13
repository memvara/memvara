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

import asyncio
from datetime import datetime, timezone

import pytest

from memvara import Memvara, NullLLM
from memvara.aio import AsyncMemvara
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


def rendered(paths):
    """Whole chains rather than loose objects: `paths_between` is asked about a route,
    so which spellings it ran through is part of the answer being checked."""
    return [p.render() for p in paths]


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


# --- paths_between ------------------------------------------------------------

def test_paths_between_reaches_a_chain_hanging_off_the_learned_canonical(mem):
    """The same defect one method further on, and the one with the worst failure mode:
    a walk that starts nowhere near its own entity returns `[]`, and `[]` from
    `paths_between` reads as "not connected" rather than as "asked under the wrong
    name"."""
    mem.remember("Dana", "works_at", "IBM")
    mem.remember("Dana", "lives_in", "Armonk")
    learn(mem, "IBM", "Big Blue")
    assert rendered(mem.paths_between("Big Blue", "Armonk")) == [
        "IBM <-works_at- Dana -lives_in-> Armonk"]


def test_paths_between_resolves_the_target_end_as_well(mem):
    """Both ends of this question are probes and neither carries a stamp, so resolving
    only the source would fix half of it."""
    mem.remember("Alice", "reports_to", "Dana")
    mem.remember("Dana", "works_at", "IBM")
    learn(mem, "IBM", "Big Blue")
    assert rendered(mem.paths_between("Alice", "Big Blue")) == [
        "Alice -reports_to-> Dana -works_at-> IBM"]


def test_paths_between_still_reaches_an_end_keyed_before_the_merge(mem):
    """No narrowing, on either end. The chain finishes at `big blue` because that is
    what it was written under, and a target resolved *to* the canonical would have
    reported the two as unconnected on the strength of having learned they are one
    company."""
    mem.remember("IBM", "headquartered_in", "Armonk")
    mem.remember("Alice", "reports_to", "Dana")
    mem.remember("Dana", "works_at", "Big Blue")        # keyed `big blue`
    learn(mem, "IBM", "Big Blue")
    assert rendered(mem.paths_between("Alice", "IBM")) == [
        "Alice -reports_to-> Dana -works_at-> Big Blue"]


def test_paths_between_ends_the_path_at_whichever_name_it_arrives_by(mem):
    """Arrival is terminal, and a merged target does not make it twice-terminal: the
    connection comes back once, at the spelling the store actually holds, rather than
    once more with `-renamed_as-> IBM` glued on the end of it."""
    mem.remember("IBM", "headquartered_in", "Armonk")
    mem.remember("Big Blue", "renamed_as", "IBM")
    mem.remember("Alice", "works_at", "Big Blue")
    learn(mem, "IBM", "Big Blue")
    assert rendered(mem.paths_between("Alice", "IBM")) == ["Alice -works_at-> Big Blue"]


def test_paths_between_with_nothing_learned_about_either_end_is_unchanged(mem):
    """The deterministic fold stays the fallback. This is a widening of what a probe
    reaches and never a replacement of it, so a name nothing has been learned about
    connects to exactly what it connected to before."""
    mem.remember("Dana", "works_at", "IBM")
    mem.remember("Dana", "lives_in", "Armonk")
    assert rendered(mem.paths_between("IBM", "Armonk")) == [
        "IBM <-works_at- Dana -lives_in-> Armonk"]
    assert mem.paths_between("Big Blue", "Armonk") == []


def test_the_scoped_and_async_views_resolve_a_probe_the_same_way(mem):
    """All three facades forward to `Memvara.paths_between` and add nothing to it, which
    is what makes one fix cover them — pinned rather than assumed, because "it only
    delegates" is exactly the claim that stops being true without anyone noticing."""
    mem.remember("Dana", "works_at", "IBM")
    learn(mem, "IBM", "Big Blue")
    expected = ["IBM <-works_at- Dana"]

    assert rendered(mem.scope(user="alice").paths_between("Big Blue", "Dana")) == expected

    amem = AsyncMemvara(mem)
    assert rendered(asyncio.run(amem.paths_between("Big Blue", "Dana"))) == expected
    assert rendered(asyncio.run(
        amem.scope(user="alice").paths_between("Big Blue", "Dana"))) == expected


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


def test_paths_between_walks_from_every_key_a_probe_names(mem):
    """Same rule at the ends of a route: the chain is filed under the one spelling
    nobody asked with, and is still the answer. Taking the fold and the canonical but
    not the siblings would make "are these connected" depend on which of three names
    for one company the caller happened to type."""
    mem.remember("IBM", "headquartered_in", "Armonk")
    mem.remember("Dana", "works_at", "Armonk Giant")
    learn(mem, "IBM", "Big Blue")
    learn(mem, "IBM", "Armonk Giant")
    assert rendered(mem.paths_between("Big Blue", "Dana")) == [
        "Armonk Giant <-works_at- Dana"]


def test_an_edge_between_two_names_of_one_entity_is_a_self_loop(mem):
    """`_edges` already drops `subject == object`. After a merge, an edge from `big
    blue` to `ibm` is that same self-loop spelled two ways, and walking it would return
    every one-hop path twice — once direct, once through the entity's other name."""
    mem.remember("Big Blue", "renamed_as", "IBM")
    mem.remember("IBM", "headquartered_in", "Armonk")
    learn(mem, "IBM", "Big Blue")
    assert objects(mem.neighborhood("Big Blue")) == ["Armonk"]


def test_paths_between_two_names_of_one_entity_is_not_a_path(mem):
    """The self-loop question asked of a route: once the merge is learned, "how is IBM
    connected to Big Blue" has one entity in it, and every chain that could be returned
    for it is a loop leaving that entity and coming back — an answer about the size of
    the graph rather than about the two names. `[]` is the honest answer, and it is the
    same `[]` `paths_between(x, x)` has always given.

    Widening the probe is what makes this reachable at all: before the merge these were
    two entities, and the edge that says so was a real one-hop answer.
    """
    mem.remember("IBM", "headquartered_in", "Armonk")
    mem.remember("Big Blue", "renamed_as", "IBM")
    assert rendered(mem.paths_between("Big Blue", "IBM")) == ["Big Blue -renamed_as-> IBM"]

    learn(mem, "IBM", "Big Blue")
    assert mem.paths_between("IBM", "Big Blue") == []
    assert mem.paths_between("Big Blue", "IBM") == []


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


def test_an_alias_learned_in_one_tenant_does_not_join_another_tenants_route():
    """Both ends of `paths_between` are resolved under the reader's own owner and no
    further. Identical graphs in two tenants, and only the tenant that learned the merge
    is connected — the alias is what joins the two halves, so a probe that could reach
    across owners would be inventing a connection out of a sibling's decision."""
    with Memvara(llm=NullLLM(), embedder=HashingEmbedder(dim=64)) as mem:
        for tenant in ("t_a", "t_b"):
            mem.remember("Dana", "works_at", "IBM", tenant=tenant, user="alice")
        learn(mem, "IBM", "Big Blue", scope=Scope("t_a", "alice"))

        assert rendered(mem.paths_between("Big Blue", "Dana", tenant="t_a",
                                          user="alice")) == ["IBM <-works_at- Dana"]
        assert mem.paths_between("Big Blue", "Dana", tenant="t_b", user="alice") == []


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
    """`GraphTraverser.between` takes the same widening on either end. Exercised
    directly as well as through `Memvara.paths_between`, because the traverser holds no
    registry and is usable against any `Store` on its own: the keys are the whole of its
    contract, and a caller that has some other way of knowing what an entity is called
    gets the same widening without owning an `EntityRegistry`."""
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
