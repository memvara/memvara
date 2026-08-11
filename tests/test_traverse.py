"""Multi-hop traversal: the store's entity index, and the walk built on it.

Grouped by the property under test rather than by module, because the properties are
what the feature is: one instant for the whole path, a negation is not a link, every hop
is scope-checked, and the same graph always answers the same way.
"""

import asyncio
import sqlite3
from datetime import datetime, timezone

import pytest

from memvara import Memvara, NullLLM
from memvara.aio import AsyncMemvara
from memvara.embed import HashingEmbedder
from memvara.retrieve.traverse import HOP_DAMPING, Edge, GraphTraverser, Path
from memvara.schema import PredicateRegistry
from memvara.store import SQLiteStore
from memvara.store.sqlite import SCHEMA_VERSION
from memvara.types import SUBJECT_ENTITY, Claim, Scope

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
T1 = datetime(2024, 6, 1, tzinfo=timezone.utc)
T2 = datetime(2025, 1, 1, tzinfo=timezone.utc)
T3 = datetime(2025, 6, 1, tzinfo=timezone.utc)

SCOPE = Scope("acme", "alice")


@pytest.fixture()
def store():
    s = SQLiteStore(":memory:")
    yield s
    s.close()


@pytest.fixture()
def walker(store):
    return GraphTraverser(store, PredicateRegistry())


@pytest.fixture()
def mem():
    with Memvara(llm=NullLLM(), embedder=HashingEmbedder(dim=64), tenant="acme",
                 user="alice") as m:
        yield m


def edge(store, subject, predicate, obj, *, scope=SCOPE, at=T0, **kw):
    """Write one claim straight to the store, bypassing extraction and reconciliation.

    Deliberately not `remember()`: several tests here need two rows the write path would
    have refused to leave side by side (a duplicate value, a retired edge whose successor
    starts later), and the traversal has to be correct about whatever is on disk.
    """
    claim = Claim(subject=subject, predicate=predicate, object=obj, scope=scope,
                  recorded_at=at, valid_from=kw.pop("valid_from", at), **kw)
    store.put_claim(claim)
    return claim


def rendered(paths):
    return [p.render() for p in paths]


# --- one instant for the whole traversal -------------------------------------


def test_a_path_is_never_stitched_from_edges_that_were_never_believed_together(walker,
                                                                              store):
    """The failure this whole feature could most easily produce, and the one nobody else
    can refuse: two edges that each existed, but never at the same time, joined into a
    chain and reported as a connection.

    Alice reported to Dana until Dana left; Dana joined Acme afterwards. "Alice is
    connected to Acme through Dana" was true on no day of the year, and a traversal that
    evaluates its hops one clock-read apart — or worse, at whatever instant each store
    call happened to run at — says it confidently and shows its work."""
    edge(store, "Alice", "reports_to", "Dana", at=T0,
         invalidated_at=T1, valid_to=T1)
    edge(store, "Dana", "works_at", "Acme", at=T2)

    for instant in (T0, T1, T2, T3, None):
        found = walker.between("Alice", "Acme", SCOPE, as_of=instant)
        assert found == [], f"connected them at {instant}"


def test_an_ended_edge_is_not_walked_as_live(mem):
    """A hop whose *world* clock has closed is over, whatever its belief clock says.

    Since supersession stopped closing transaction time, every superseded edge in a store
    the library wrote itself carries `invalidated_at=None` — so a traversal that read
    liveness off that column alone would happily walk "Alice reports to Dana" years after
    Alice started reporting to someone else, and present it as a current connection.
    `Store.adjacent` applies the full liveness predicate, which is what makes this hold;
    the point of the test is that the write path can now *produce* the row that would
    catch a backend that did not.

    Written through `remember()` on purpose, unlike everything else in this file: the
    shape being checked is exactly the one the ordinary write path emits.
    """
    from datetime import timedelta
    from memvara.types import utcnow

    then = utcnow() - timedelta(days=800)
    moved = utcnow() - timedelta(days=100)
    mem.remember("Alice", "reports_to", "Dana", valid_from=then, recorded_at=then)
    mem.remember("Dana", "works_at", "Acme", valid_from=then, recorded_at=then)
    assert mem.paths_between("Alice", "Acme"), "the chain held while Dana was at Acme"

    mem.remember("Dana", "works_at", "Globex", valid_from=moved, recorded_at=moved)
    acme = [c for c in mem.history("Dana", "works_at") if c.object == "Acme"][0]
    assert acme.state == "ended" and acme.invalidated_at is None

    assert mem.paths_between("Alice", "Acme") == [], "walked an edge that had finished"
    assert rendered(mem.paths_between("Alice", "Globex")) == [
        "Alice -reports_to-> Dana -works_at-> Globex"], "and the live one still walks"
    # Still reachable where it genuinely held, which is the other half of the rule:
    # ended is not deleted.
    assert rendered(mem.paths_between("Alice", "Acme",
                                      valid_at=moved - timedelta(days=1))) == [
        "Alice -reports_to-> Dana -works_at-> Acme"]


def test_a_chain_that_did_hold_at_once_is_returned_at_that_instant_and_not_after(walker,
                                                                                store):
    """The other half of the same rule: `as_of` is not a filter on the newest edge, it is
    the instant the *whole* path is evaluated at. The chain below is real between T1 and
    T2 and nowhere else, and a traversal that quietly used wall-clock now for the hops it
    had already taken would answer the same at every instant."""
    edge(store, "Alice", "reports_to", "Dana", at=T1, invalidated_at=T2, valid_to=T2)
    edge(store, "Dana", "works_at", "Acme", at=T1)

    assert rendered(walker.between("Alice", "Acme", SCOPE, as_of=T1)) == [
        "Alice -reports_to-> Dana -works_at-> Acme"]
    assert walker.between("Alice", "Acme", SCOPE, as_of=T0) == [], "before either edge"
    assert walker.between("Alice", "Acme", SCOPE, as_of=T3) == [], "after the first ended"


def test_the_clock_pair_is_pinned_before_the_first_hop_rather_than_read_per_hop(walker):
    """Neither axis may reach the store as None. `_live_clause` substitutes its own
    `now()` for a missing instant, so a three-hop walk would evaluate three hops at three
    instants — and a claim recorded between hop one and hop two would join a path it was
    not part of when the question was asked. Pinning is what makes that impossible, and
    both axes have to be real datetimes by the time the store sees them."""
    assert walker._pin(T1, T2) == (T1, T2)
    assert walker._pin(datetime(2024, 6, 1), None)[0] == T1, \
        "naive input is UTC, as everywhere"

    before = datetime.now(timezone.utc)
    valid_at, known_at = walker._pin(None, None)
    after = datetime.now(timezone.utc)
    assert before <= valid_at <= after and before <= known_at <= after


def test_an_unset_axis_is_pinned_to_the_same_now_as_the_other_one(walker):
    """One clock read fills both defaults. Two reads would put the axes microseconds
    apart, so `neighborhood()` with no arguments would be evaluating "the world" and
    "our belief" at two different moments — a split nothing can reproduce and nothing in
    the result would show."""
    valid_at, known_at = walker._pin(None, None)
    assert valid_at == known_at
    assert walker._pin(T1, None)[0] == T1, "a supplied axis is never overwritten"
    assert walker._pin(None, T1)[1] == T1


def test_the_pinned_pair_survives_every_hop_of_a_walk(walker, store, monkeypatch):
    """The invariant the whole multi-hop feature rests on, asserted on the wire rather
    than on `_pin` alone: every `adjacent` call a walk makes is handed the *same* pair.
    A walk that re-read either clock per hop could return a chain no single moment ever
    contained, and nothing in a `Path` would say so."""
    edge(store, "Alice", "reports_to", "Dana")
    edge(store, "Dana", "works_at", "Acme")
    edge(store, "Acme", "founded_by", "Bob")

    seen: list[tuple] = []
    real = store.adjacent

    def spy(*a, **kw):
        seen.append((kw["valid_at"], kw["known_at"]))
        return real(*a, **kw)

    monkeypatch.setattr(store, "adjacent", spy)
    assert walker.neighborhood("Alice", SCOPE, depth=3, k=10)
    assert len(seen) == 3, "one store call per hop"
    assert len(set(seen)) == 1, "and one clock pair across all of them"


def test_a_claim_recorded_after_the_pinned_instant_cannot_join_the_path(walker, store):
    """Knowledge from the future, which is the one way a bitemporal read can actively
    lie. The transaction-time floor is inside the liveness predicate the store applies to
    every `adjacent` call, and the pin is what makes "after" mean the same thing on hop
    three as on hop one."""
    edge(store, "Alice", "reports_to", "Dana", at=T0)
    edge(store, "Dana", "works_at", "Acme", at=T3)

    assert walker.between("Alice", "Acme", SCOPE, as_of=T1) == []
    assert rendered(walker.between("Alice", "Acme", SCOPE, as_of=T3)) != []


# --- a negation is not a link ------------------------------------------------


def test_a_negated_claim_is_adjacent_in_the_store_and_is_never_walked(walker, store):
    """"Alice does not work at Acme" is genuinely a claim about Alice and about Acme, so
    the store's index returns it — the layer that knows about entities should not also be
    deciding what counts as a relationship. Walking it would report the *absence* of a
    link as its presence, which is the plainest confident lie available here, so the
    refusal lives in the traversal where every store implementation inherits it."""
    negative = edge(store, "Alice", "works_at", "Acme", polarity=-1)
    edge(store, "Acme", "founded_by", "Bob")

    assert [c.id for c in store.adjacent("acme", ["alice"])] == [negative.id]
    assert walker.neighborhood("Alice", SCOPE, depth=2) == []
    assert walker.between("Alice", "Bob", SCOPE) == []


# --- scope ---------------------------------------------------------------


def test_a_deeper_scopes_claim_cannot_become_an_intermediate_hop(walker, store):
    """The leak this feature could introduce: a hop is not shown to the caller as a
    result, so an unchecked one would let a traversal *use* a claim the caller cannot
    read and hand back the far side of it. Here the middle edge belongs to one agent and
    the caller is the user; `Scope.sees` refuses downward, so the chain does not exist for
    them — and neither does Carol, whom only that edge names."""
    edge(store, "Alice", "reports_to", "Dana")
    edge(store, "Dana", "mentors", "Carol", scope=Scope("acme", "alice", "agent-1"))
    edge(store, "Carol", "works_at", "Acme")

    assert walker.between("Alice", "Acme", SCOPE) == []
    assert [p.nodes[-1] for p in walker.neighborhood("Alice", SCOPE, depth=3)] == ["dana"]


def test_a_sibling_users_claim_is_not_a_hop_even_though_it_shares_the_tenant(walker,
                                                                            store):
    """`Store.adjacent` is tenant-scoped and nothing finer, exactly like
    `competing_claims` — so on a shared tenant the rows come back and this layer is the
    only thing standing between them and the caller."""
    edge(store, "Alice", "reports_to", "Dana")
    edge(store, "Dana", "works_at", "Initech", scope=Scope("acme", "bob"))

    assert rendered(walker.neighborhood("Dana", SCOPE, depth=2)) == [
        "Dana <-reports_to- Alice"]
    assert walker.between("Alice", "Initech", SCOPE) == []


def test_a_traversal_reaches_the_tenants_shared_knowledge_upward(walker, store):
    """The direction that *must* open, and the reason the stored keys are the bare fold
    rather than owner-qualified. Scopes inherit upward, so a user's own fact chaining
    into the organization's shared graph is the main thing multi-hop is for; keys
    qualified by owner would have made that join impossible to express."""
    edge(store, "Alice", "reports_to", "Dana")
    edge(store, "Dana", "works_at", "Acme", scope=Scope("acme"))

    assert rendered(walker.between("Alice", "Acme", SCOPE)) == [
        "Alice -reports_to-> Dana -works_at-> Acme"]


def test_every_claim_on_every_returned_path_is_one_get_all_would_have_returned(mem):
    """The invariant the whole scope argument reduces to: traversal composes what is
    readable, it does not widen what is readable. Stated as a property over a graph that
    mixes three scopes, because the per-case tests above can only check the cases someone
    thought of."""
    for subject, predicate, obj, scope in (
        ("Alice", "reports_to", "Dana", dict()),
        ("Dana", "works_at", "Acme", dict(agent="agent-1")),
        ("Acme", "founded_by", "Bob", dict(user=None)),
        ("Bob", "lives_in", "Lisbon", dict(user="bob")),
        ("Alice", "lives_in", "Berlin", dict()),
    ):
        mem.remember(subject, predicate, obj, **scope)

    readable = {c.id for c in mem.get_all()}
    walked = {c.id for p in mem.neighborhood("Alice", depth=4, k=50) for c in p.claims}

    assert walked, "a graph this connected must produce some path"
    assert walked <= readable


def test_a_hub_of_unreadable_claims_cannot_crowd_out_the_one_readable_edge(store):
    """A neighbour's claims about one entity must not consume the page and take the
    caller's own edge with them — that makes one caller's answer a function of another's
    write volume, which is both wrong and faintly informative about them.

    This was first fixed by re-asking wider when the page came back full and filtering
    had dropped something, the shape `HybridRetriever` uses for filter starvation. That
    only moved the threshold: measured on a shared tenant with 20 readable claims about
    a hub, 15,000 competing claims still returned 19 and 40,000 returned 8. The scope now
    goes to `Store.adjacent` and is applied in the same statement as `limit`, so the
    unreadable rows are never on the page to crowd anything out.

    Which makes the property testable at its limit rather than at a tuned ratio: an
    `edge_limit` of **1**, six decoys, and the readable edge deliberately the *worst*
    scoring of the seven. Under the old fix this needed the retry and a multiplier big
    enough; under this one no multiplier exists and the decoys are not competing."""
    for i in range(6):
        edge(store, "Hub", "linked_to", f"decoy-{i}", scope=Scope("acme", "bob"),
             confidence=1.0)
    edge(store, "Hub", "linked_to", "Target", confidence=0.5)

    starved = GraphTraverser(store, PredicateRegistry(), edge_limit=1)
    assert rendered(starved.neighborhood("Hub", SCOPE, depth=1)) == [
        "Hub -linked_to-> Target"]


def test_the_scope_reaches_the_store_rather_than_being_applied_after_it(store):
    """The line that makes the test above hold, pinned directly.

    A store honouring `scopes` returns only readable rows, so the count coming back is
    the count after filtering. Asserting on the traverser alone would still pass if the
    argument were dropped and the Python-side `sees` check did all the work — which is
    exactly the regression this guards, because that is the unsound arrangement it
    replaced."""
    for i in range(6):
        edge(store, "Hub", "linked_to", f"decoy-{i}", scope=Scope("acme", "bob"))
    edge(store, "Hub", "linked_to", "Target")

    assert len(store.adjacent("acme", ["hub"])) == 7           # unscoped: everything
    assert len(store.adjacent("acme", ["hub"], scopes=SCOPE.ancestors())) == 1
    # An unresolved scope fails closed, as `candidate_ids` does. Matching everything
    # would be the worst possible response to a caller bug about *authorization*.
    assert store.adjacent("acme", ["hub"], scopes=[]) == []


def test_an_empty_key_is_not_an_entity_and_names_no_neighbours(store):
    """A retraction stores `''` for the object it retracts, so a store that treats the
    empty string as an ordinary key answers `adjacent(t, [""])` with every retraction in
    the tenant, joined into one hub that does not exist. SQLite stores `''` and Postgres
    stores `NULL` for this, and that difference must not be observable through the
    protocol — so the contract is stated on `Store.adjacent` and both ends enforce it."""
    edge(store, "Alice", "likes", "tea", polarity=-1)
    edge(store, "Bob", "likes", "coffee", polarity=-1)

    assert store.adjacent("acme", [""]) == []
    # And the empty key is ignored rather than poisoning a batch it appears in.
    assert len(store.adjacent("acme", ["", "alice"])) == 1


# --- bounded and deterministic -----------------------------------------------


def test_a_cycle_is_walked_once_rather_than_forever(walker, store):
    """Three mutual edges are a loop, and a walk with no visited check follows it until
    the depth cap — returning `Alice -> Bob -> Alice` as a discovery about Alice."""
    edge(store, "Alice", "knows", "Bob")
    edge(store, "Bob", "knows", "Carol")
    edge(store, "Carol", "knows", "Alice")

    paths = walker.neighborhood("Alice", SCOPE, depth=6, k=50)

    assert all(len(set(p.nodes)) == len(p.nodes) for p in paths)
    assert max(p.hops for p in paths) == 2, "the loop closes on the third hop"


def test_a_self_referential_claim_is_not_an_edge(walker, store):
    """`Acme Corp` and `ACME` fold to one entity, so a claim between two spellings of one
    name is a self-loop. It cannot extend any path — its far end is where the walk is
    standing — and admitting it would only make the cycle check do the work twice."""
    edge(store, "Acme Corp", "also_known_as", "ACME")
    edge(store, "Acme", "founded_by", "Bob")

    assert rendered(walker.neighborhood("Acme", SCOPE, depth=2)) == [
        "Acme -founded_by-> Bob"]


def test_an_empty_object_is_not_a_node_every_retraction_shares(walker, store):
    """An object of `""` is how a retraction says "clear the whole slot", so the empty
    key is a real stored value rather than a missing one. Treated as an entity it becomes
    a hub every retraction in the tenant passes through, and any two unrelated facts
    become two hops apart."""
    edge(store, "Alice", "works_at", "")
    edge(store, "Bob", "works_at", "")

    assert walker.neighborhood("Alice", SCOPE, depth=2) == []
    assert walker.between("Alice", "Bob", SCOPE) == []


def test_the_beam_bounds_a_hub_instead_of_letting_it_multiply(store):
    """A node with 40 edges reached at depth 3 is 64,000 partial paths with no frontier
    cap, and the store call at each level grows with it. The beam holds the frontier flat
    and prunes by the same total order the results are ranked by, so what it drops is the
    worst of the level rather than whatever the dict happened to hold."""
    for i in range(40):
        edge(store, "Hub", "linked_to", f"n-{i}")
        edge(store, f"n-{i}", "linked_to", f"leaf-{i}")

    narrow = GraphTraverser(store, PredicateRegistry(), beam=4)
    paths = narrow.neighborhood("Hub", SCOPE, depth=3, k=100)

    assert len({p.nodes[1] for p in paths if p.hops > 1}) <= 4
    assert len(paths) <= 100


def test_two_stores_holding_the_same_graph_return_the_same_order(store):
    """Ranking that depends on `uuid4` has bitten this project three times. Two ingests
    of one graph mint different claim ids, so an order broken on `id` is stable within a
    store and a coin flip across two — exactly the comparison a benchmark, a regression
    test and a bisect all make."""
    def build(target):
        for subject, predicate, obj in (("Alice", "knows", "Bob"),
                                        ("Bob", "knows", "Carol"),
                                        ("Alice", "knows", "Dana"),
                                        ("Dana", "knows", "Carol")):
            edge(target, subject, predicate, obj)
        return GraphTraverser(target, PredicateRegistry())

    first = build(store)
    with SQLiteStore(":memory:") as other:
        second = build(other)
        left = rendered(first.neighborhood("Alice", SCOPE, depth=3, k=20))
        right = rendered(second.neighborhood("Alice", SCOPE, depth=3, k=20))

    assert left == right
    assert len({c.id for p in first.neighborhood("Alice", SCOPE, depth=3, k=20)
                for c in p.claims}) == 4, "different ids, same order"


def test_the_same_edge_asserted_twice_is_one_path_not_two(walker, store):
    """A duplicate value can sit in the store legitimately — the same fact written at two
    scopes, or a row a merge has not reached yet. Returning the identical chain twice
    spends the caller's `k` on noise, so paths de-duplicate on content (`value_key`) and
    keep the better-scoring row."""
    edge(store, "Alice", "knows", "Bob", confidence=0.9)
    edge(store, "Alice", "knows", "Bob", confidence=0.4)

    paths = walker.neighborhood("Alice", SCOPE, depth=1, as_of=T0)

    assert len(paths) == 1
    assert paths[0].score == pytest.approx(0.9, abs=1e-3)


def test_four_spellings_of_one_company_are_one_node(walker, store):
    """The reason traversal is possible at all: entity resolution already folded the
    surface forms, so `Acme, Inc.` and `ACME` are one node and the chain through them
    joins. Keyed on the raw text, this graph is three disconnected pairs."""
    edge(store, "Alice", "works_at", "Acme, Inc.")
    edge(store, "ACME", "founded_by", "Bob")
    edge(store, "Bob", "lives_in", "Lisbon")

    assert rendered(walker.between("Alice", "the acme corporation", SCOPE)) == [
        "Alice -works_at-> Acme, Inc."]
    assert walker.between("Alice", "Lisbon", SCOPE)[0].hops == 3


# --- scoring -----------------------------------------------------------------


def test_a_path_can_never_outscore_its_own_prefix(walker, store):
    """The invariant every other scoring claim rests on: both factors `_extend`
    multiplies by are at most 1.0. It is what makes `min_score` prunable mid-walk, and
    what makes "prefer the short route unless it is worse" fall out of the ranking rather
    than being imposed on top of it."""
    edge(store, "Alice", "knows", "Bob")
    edge(store, "Bob", "knows", "Carol")
    edge(store, "Carol", "knows", "Dana")

    by_length = {p.hops: p.score
                 for p in walker.neighborhood("Alice", SCOPE, depth=3, as_of=T0)}

    assert by_length[1] >= by_length[2] >= by_length[3]
    assert by_length[2] == pytest.approx(by_length[1] * HOP_DAMPING, rel=1e-6)


def test_three_weak_hops_rank_below_one_strong_edge(walker, store):
    """The case the damping constant was chosen on. Composition is an inference nobody
    asserted, so a chain of three 0.6-confidence links has to land far below a single
    0.99 one — 0.12 against 0.99 at the shipped 0.75."""
    for subject, obj in (("Alice", "Bob"), ("Bob", "Carol"), ("Carol", "Dana")):
        edge(store, subject, "knows", obj, confidence=0.6)
    edge(store, "Alice", "married_to", "Eve", confidence=0.99)

    best = walker.neighborhood("Alice", SCOPE, depth=3, k=10, as_of=T0)

    assert best[0].labels[-1] == "Eve"
    assert best[0].score == pytest.approx(0.99, abs=1e-3)
    assert min(p.score for p in best) == pytest.approx(0.6 ** 3 * 0.75 ** 2, abs=1e-3)


def test_a_stale_edge_is_weaker_than_a_fresh_one_on_the_predicates_own_half_life(store):
    """Liveness is a hard filter, so an employer nobody ever superseded is *live* and is
    still weak evidence about where someone works today — while a birthplace of the same
    age is as good as it will ever be. Only a predicate-keyed half-life separates those,
    which is why `recency_factor` is in the edge strength and not merely in search."""
    edge(store, "Alice", "works_at", "Acme", at=T0)
    edge(store, "Alice", "born_in", "Berlin", at=T0)

    walker = GraphTraverser(store, PredicateRegistry())
    scores = {p.labels[-1]: p.score
              for p in walker.neighborhood("Alice", SCOPE, depth=1, as_of=T3)}

    assert scores["Berlin"] > scores["Acme"]
    assert scores["Berlin"] > 0.98, "a 100-year half-life barely moves in 18 months"


def test_confidence_outside_the_unit_range_cannot_promote_a_longer_path(walker, store):
    """Nothing enforces `Claim.confidence`, and an importer or a third-party extractor
    can write 3.0. Unclamped it multiplies straight into the score, so a three-hop chain
    would outrank the one-hop edge beside it and the prefix invariant — the thing
    `min_score` pruning depends on — would be quietly false."""
    edge(store, "Alice", "knows", "Bob", confidence=5.0)
    edge(store, "Bob", "knows", "Carol", confidence=5.0)

    paths = walker.neighborhood("Alice", SCOPE, depth=2, as_of=T0)

    assert paths[0].score == pytest.approx(1.0, abs=1e-3)
    assert paths[1].score == pytest.approx(HOP_DAMPING, abs=1e-3)


def test_min_score_prunes_prefixes_and_still_answers_exactly(walker, store):
    """Pruning mid-walk is exact rather than a heuristic, because a path's score can
    never rise: a prefix already under the floor cannot produce a suffix over it. If that
    ever stops being true the floor starts silently dropping good long paths, which is
    invisible in the result."""
    for subject, obj in (("Alice", "Bob"), ("Bob", "Carol"), ("Carol", "Dana")):
        edge(store, subject, "knows", obj, confidence=0.8)

    everything = walker.neighborhood("Alice", SCOPE, depth=3, k=20, as_of=T0)
    floored = walker.neighborhood("Alice", SCOPE, depth=3, k=20, min_score=0.5,
                                  as_of=T0)

    assert rendered(floored) == [p.render() for p in everything if p.score >= 0.5]
    assert 0 < len(floored) < len(everything)


def test_short_paths_crowd_out_the_answer_unless_min_hops_says_otherwise(walker, store):
    """Score is non-increasing along a path, so *every* one-hop path outranks *every*
    two-hop one at equal edge quality — and a person with four relations spends a `k` of
    four before a two-hop answer is even considered. Measured on `bench/multihop.py`,
    two-hop questions go from 45.3% to 100% at k=12 with this. The short paths are still
    walked, because they are the only way to reach the long ones."""
    # Three distinct relations, so the diversity pass has nothing to demote and the only
    # thing deciding the top three is path length.
    edge(store, "Alice", "knows", "Bob")
    edge(store, "Alice", "lives_in", "Berlin")
    edge(store, "Alice", "works_at", "Acme")
    edge(store, "Acme", "founded_by", "Dana")

    crowded = walker.neighborhood("Alice", SCOPE, depth=2, k=3, as_of=T0)
    focused = walker.neighborhood("Alice", SCOPE, depth=2, k=3, min_hops=2, as_of=T0)

    assert {p.hops for p in crowded} == {1}
    assert rendered(focused) == ["Alice -works_at-> Acme -founded_by-> Dana"]


def test_one_relation_out_of_a_hub_cannot_take_the_whole_answer(walker, store):
    """Ties are the rule here, not the exception: a graph of asserted facts has
    confidence 1.0 everywhere, so most paths of one length score identically and the
    tie-break runs on a content hash. Whichever relation a hub happens to have most of
    then takes every slot — six ways of saying "a colleague" while the one path that
    answers the question sits outside `k`. Demoted rather than dropped, like
    `HybridRetriever._rank`, so `k` still means the same thing."""
    edge(store, "Alice", "works_at", "Acme")
    for i in range(20):
        edge(store, f"Colleague {i}", "works_at", "Acme")
    edge(store, "Acme", "founded_by", "Bob")

    spread = walker.neighborhood("Alice", SCOPE, depth=2, k=5, min_hops=2, as_of=T0)
    flat = GraphTraverser(store, PredicateRegistry(), max_per_relation=0).neighborhood(
        "Alice", SCOPE, depth=2, k=25, min_hops=2, as_of=T0)

    assert "Bob" in [p.labels[-1] for p in spread]
    assert [p.labels[-1] for p in flat].index("Bob") > 5, "the failure being fixed"
    assert len(spread) == 5, "demotion, not deletion — `k` still means `k`"


def test_the_diversity_pass_survives_the_final_sort(walker, store):
    """It did not, at first. The pass ran on each level and the walk then re-sorted
    everything it had collected by score — which put the ranking straight back, so the
    answer only moved when a level happened to be wider than `k`. It looked like it
    worked because the benchmark it was written for truncates."""
    edge(store, "Alice", "works_at", "Acme")
    for i in range(6):
        edge(store, f"Colleague {i}", "works_at", "Acme")
    edge(store, "Acme", "founded_by", "Bob")

    # `k` far larger than the frontier, so nothing truncates anywhere and the final
    # ordering is the only thing that can be under test.
    ranked = walker.neighborhood("Alice", SCOPE, depth=2, k=100, min_hops=2, as_of=T0)

    assert [p.labels[-1] for p in ranked].index("Bob") <= 2


# --- shape of the answer -----------------------------------------------------


def test_a_path_carries_its_edges_its_nodes_and_a_direction_for_each_hop(walker, store):
    """A path the caller cannot inspect is an answer they cannot check. Every hop has to
    be recoverable as the claim it came from — including which way it was walked, because
    `Acme founded_by Bob` read from Bob is still a fact about Acme and must not be
    rendered as though Bob had founded something called `founded_by`."""
    edge(store, "Acme", "founded_by", "Bob")
    edge(store, "Alice", "works_at", "Acme")

    path = walker.between("Bob", "Alice", SCOPE)[0]

    assert path.nodes == ("bob", "acme", "alice")
    assert path.labels == ("Bob", "Acme", "Alice")
    assert path.hops == 2 and len(path.edges) == 2 and len(path.claims) == 2
    assert [e.backward for e in path.edges] == [True, True]
    assert path.render() == "Bob <-founded_by- Acme <-works_at- Alice"
    assert [e.predicate for e in path.edges] == ["founded_by", "works_at"]


def test_an_edge_reports_the_end_it_came_from_and_the_end_it_reached(walker, store):
    """`source`/`target` follow the walk; `subject`/`object` stay the claim's own. Both
    are needed and conflating them is how a rendered chain reverses a relationship."""
    edge(store, "Acme", "founded_by", "Bob")

    forward = walker.neighborhood("Acme", SCOPE, depth=1)[0].edges[0]
    backward = walker.neighborhood("Bob", SCOPE, depth=1)[0].edges[0]

    assert (forward.source, forward.target) == ("Acme", "Bob")
    assert (forward.source_key, forward.target_key) == ("acme", "bob")
    assert (backward.source, backward.target) == ("Bob", "Acme")
    assert (backward.source_key, backward.target_key) == ("bob", "acme")
    assert "founded_by" in repr(forward) and "Bob" in repr(forward)


def test_a_paths_repr_shows_the_chain_rather_than_the_dataclass(walker, store):
    """A list of these is read at a REPL and in a failing assertion; the dataclass repr
    dumps every field of every claim on every hop."""
    edge(store, "Alice", "knows", "Bob")

    assert repr(walker.neighborhood("Alice", SCOPE, depth=1, as_of=T0)[0]) == (
        "<Path 1.0000 1h Alice -knows-> Bob>")


def test_a_zero_hop_path_falls_back_to_its_keys_rather_than_raising():
    """`labels` reads the node spellings off the edges, and a path with none has no
    spellings to read. Unreachable through the public API, which never returns a zero-hop
    path — but an IndexError from a rendering helper is a worse answer than the keys."""
    assert Path(nodes=("alice",), edges=(), score=1.0).labels == ("alice",)


# --- the questions the API is for --------------------------------------------


def test_who_does_alices_manager_report_to(mem):
    """The question the feature exists for, and the one the store could not express:
    "where does Alice work" was one indexed lookup, and this was not expressible at any
    cost, because the two facts that answer it are two rows with no join between them."""
    mem.remember("Alice", "reports_to", "Dana")
    mem.remember("Dana", "reports_to", "Priya")

    assert [p.labels[-1] for p in mem.paths_between("Alice", "Priya")] == ["Priya"]
    assert mem.paths_between("Alice", "Priya")[0].hops == 2


def test_predicates_narrow_the_walk_through_the_registrys_canonical_names(mem):
    """A filter for `works_at` that missed claims stored as `employed_by_company` would
    be wrong in the silent direction — an empty result, not an error — and it would undo
    exactly the fold `schema.py` performs on the way in."""
    mem.remember("Alice", "employed_by_company", "Acme")
    mem.remember("Alice", "lives_in", "Berlin")

    assert [p.labels[-1] for p in mem.neighborhood("Alice", predicates=["works_at"])] == [
        "Acme"]
    assert mem.neighborhood("Alice", predicates=["born_in"]) == []


def test_an_entity_is_not_connected_to_itself(walker, store):
    """`paths_between(x, x)` has no honest answer other than none: every cycle through x
    ends where it began, so returning them would answer a question nobody asked with a
    list that grows with the store."""
    edge(store, "Alice", "knows", "Bob")
    edge(store, "Bob", "knows", "Alice")

    assert walker.between("Alice", "Alice", SCOPE) == []
    assert walker.between("Alice", "  ", SCOPE) == [], "a target that folds to nothing"


def test_a_walk_with_nothing_to_walk_returns_nothing_rather_than_guessing(walker, store):
    """Every bound is expressible as zero, and each of them means "no answer" rather than
    "no limit" — the same fail-closed reading `_scope_clause` gives an empty scope list
    and `scope_episodes` gives a negative one."""
    edge(store, "Alice", "knows", "Bob")

    assert walker.neighborhood("", SCOPE) == []
    assert walker.neighborhood("Alice", SCOPE, depth=0) == []
    assert walker.neighborhood("Alice", SCOPE, k=0) == []
    assert walker.neighborhood("Alice", SCOPE, predicates=[]) == []


def test_a_walk_stops_when_the_frontier_runs_out_before_the_depth_cap(walker, store):
    """A two-edge graph asked for six hops must not make six store calls, and must not
    return a path padded to the depth it was asked for."""
    edge(store, "Alice", "knows", "Bob")

    assert [p.hops for p in walker.neighborhood("Alice", SCOPE, depth=6)] == [1]
    assert walker.between("Alice", "Nobody", SCOPE, depth=6) == []


def test_reaching_the_target_ends_that_path_rather_than_continuing_through_it(walker,
                                                                             store):
    """Every result of `between` has to *end* at the target. A path that passes through
    it and carries on is answering a different question, and shipping it as an answer to
    "how are these connected" is a chain with an irrelevant tail."""
    edge(store, "Alice", "knows", "Bob")
    edge(store, "Bob", "knows", "Carol")
    edge(store, "Carol", "knows", "Dana")

    found = walker.between("Alice", "Carol", SCOPE, depth=3)

    assert [p.labels[-1] for p in found] == ["Carol"]


# --- store: the index the walk is built on -----------------------------------


def test_adjacent_answers_in_both_directions_on_the_folded_identity(store):
    """The lookup no existing index could serve: `fact_key` and `value_key` both hash the
    predicate in, so neither can be asked "which claims touch this entity" in either
    direction — and `subject`/`object` hold the raw text, which is four spellings of one
    employer."""
    out = edge(store, "Alice", "works_at", "Acme, Inc.")
    into = edge(store, "The Acme Corporation", "founded_by", "Bob")

    assert {c.id for c in store.adjacent("acme", ["acme"])} == {out.id, into.id}
    assert [c.id for c in store.adjacent("acme", ["acme"], incoming=False)] == [into.id]
    assert [c.id for c in store.adjacent("acme", ["acme"], outgoing=False)] == [out.id]
    assert store.adjacent("other-tenant", ["acme"]) == []


def test_adjacent_returns_one_row_for_a_claim_matched_from_both_ends(store):
    """`Acme rival_of Initech` is in both legs when both entities are in the frontier.
    Two rows for one claim would double it in every path built from it and would spend
    the caller's cap twice on the same edge."""
    both = edge(store, "Acme", "rival_of", "Initech")

    assert [c.id for c in store.adjacent("acme", ["acme", "initech"])] == [both.id]


def test_adjacent_excludes_what_was_retired_or_not_yet_recorded(store):
    """Same liveness predicate as every other read: both axes have to agree. A retired
    edge walked as live is a path through a fact we stopped believing."""
    edge(store, "Alice", "knows", "Bob", at=T0, invalidated_at=T1)
    edge(store, "Alice", "knows", "Carol", at=T2)

    def at(t):
        return [c.object for c in store.adjacent("acme", ["alice"],
                                                 valid_at=t, known_at=t)]

    assert at(T0) == ["Bob"]
    assert at(T1) == []
    assert at(T3) == ["Carol"]


def test_adjacent_reads_each_axis_from_its_own_clock(store):
    """The two axes are not interchangeable inside one edge lookup. `Bob` is retired on
    the belief clock at T1 and untouched on the world clock; `Carol` starts on the world
    clock at T2 and was recorded at T0. Sharing one instant, no argument returns both;
    with the clocks apart, `valid_at=T3, known_at=T0` does — the world as it is now,
    judged with what we knew then."""
    edge(store, "Alice", "knows", "Bob", at=T0, invalidated_at=T1)
    edge(store, "Alice", "knows", "Carol", at=T0, valid_from=T2)

    both = store.adjacent("acme", ["alice"], valid_at=T3, known_at=T0)
    assert sorted(c.object for c in both) == ["Bob", "Carol"]
    for t in (T0, T1, T2, T3):
        assert len(store.adjacent("acme", ["alice"], valid_at=t, known_at=t)) < 2


def test_adjacent_fails_closed_on_every_way_of_asking_for_nothing(store):
    """An empty `predicates` is "these relations and no others", of which there are none —
    not "no filter". Reading it the other way turns a narrowing argument into a widening
    one, which is the direction that hands back more than the caller asked for."""
    edge(store, "Alice", "knows", "Bob")

    assert store.adjacent("acme", []) == []
    assert store.adjacent("acme", ["alice"], limit=0) == []
    assert store.adjacent("acme", ["alice"], predicates=[]) == []
    assert store.adjacent("acme", ["alice"], outgoing=False, incoming=False) == []


def test_adjacent_truncates_on_content_rather_than_on_insertion_order(store):
    """`limit` is applied in SQL, before the caller can filter by scope, so *which* rows
    survive must be a function of the data and not of who wrote first — otherwise one
    tenant's answers move when a neighbour writes. Confidence first so a hub that has to
    be cut keeps its best-evidenced edges, then a content hash; the `uuid4` id decides
    nothing that is visible."""
    for i in range(5):
        edge(store, "Hub", "linked_to", f"n-{i}", confidence=0.1 * i)

    kept = store.adjacent("acme", ["hub"], limit=2)

    assert [c.object for c in kept] == ["n-4", "n-3"]
    assert [c.id for c in store.adjacent("acme", ["hub"], limit=2)] == [c.id for c in kept]


def test_adjacent_chunks_a_key_list_past_the_parameter_limit_without_losing_rows(store):
    """SQLite's parameter ceiling is 999 on older builds, and a frontier is not bounded by
    it. Chunking is exact rather than approximate — a row cut from its own chunk had
    `limit` better rows in that same chunk — and the merge has to preserve that."""
    for i in range(3):
        edge(store, f"alpha{i}", "knows", "Bob")
    keys = [f"alpha{i}" for i in range(3)] + [f"filler{i}" for i in range(2000)]

    assert len(store.adjacent("acme", keys)) == 3


def test_adjacent_filters_on_the_stored_predicate_exactly(store):
    """No registry down here, so no normalization: the store matches the column. The
    traversal folds the caller's names first, which is where the registry lives."""
    edge(store, "Alice", "knows", "Bob")
    edge(store, "Alice", "works_at", "Acme")

    assert [c.object for c in store.adjacent("acme", ["alice"],
                                             predicates=["works_at"])] == ["Acme"]
    assert store.adjacent("acme", ["alice"], predicates=["employed_by_company"]) == []


# --- migration ---------------------------------------------------------------


def regress_to_v5(path):
    """Turn a file this build wrote back into the shape version 5 left behind.

    Better than a hand-copied v5 `CREATE TABLE`: that copy would drift from the real one
    and the migration would then be tested against a schema no released build ever
    wrote.
    """
    db = sqlite3.connect(path)
    db.execute("DROP INDEX cl_subj")
    db.execute("DROP INDEX cl_obj")
    db.execute("ALTER TABLE claims DROP COLUMN subject_key")
    db.execute("ALTER TABLE claims DROP COLUMN object_key")
    db.execute("PRAGMA user_version = 5")
    db.commit()
    db.close()


def test_a_store_written_before_traversal_existed_becomes_walkable_on_first_open(tmp_path):
    """`ALTER TABLE ADD COLUMN` with a constant default touches no row, which is what
    keeps opening a large store cheap — and leaves every existing claim keyed to the
    empty string. An empty key is a *real* stored value (a retraction's object is one),
    so nothing downstream could have told "not backfilled" from "mentions nothing", and
    every pre-v6 claim would simply be absent from the graph. "Not connected" is the
    wrong answer in the direction that looks like an answer."""
    path = str(tmp_path / "old.db")
    with SQLiteStore(path) as s:
        edge(s, "Alice", "works_at", "Acme, Inc.")
        edge(s, "ACME", "founded_by", "Bob")
    regress_to_v5(path)

    with SQLiteStore(path) as upgraded:
        assert upgraded._db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        walker = GraphTraverser(upgraded, PredicateRegistry())
        assert rendered(walker.between("Alice", "Bob", SCOPE)) == [
            "Alice -works_at-> Acme, Inc. -founded_by-> Bob"]


def test_the_backfill_reads_the_write_time_stamp_rather_than_only_the_fold(tmp_path):
    """A claim whose subject was resolved through a learned alias carries the identity it
    was written with in `meta`, and that stamp is what every key derivation reads. A
    backfill that re-folded the raw text instead would silently re-key exactly the claims
    an operator had already paid a model to merge — the one thing `backfill_entities`
    exists to keep explicit and dated."""
    path = str(tmp_path / "stamped.db")
    with SQLiteStore(path) as s:
        edge(s, "Big Blue", "rival_of", "Acme", meta={SUBJECT_ENTITY: "ibm"})
    regress_to_v5(path)

    with SQLiteStore(path) as upgraded:
        assert [c.subject for c in upgraded.adjacent("acme", ["ibm"])] == ["Big Blue"]
        assert upgraded.adjacent("acme", ["big blue"]) == []


def test_reopening_an_upgraded_store_does_not_run_the_backfill_again(tmp_path):
    """The stamp is what makes the pass once-only: it rewrites every row of the claims
    table, which is seconds on a large store and must not be paid on every open. Opening
    a file already at this version has to be free."""
    path = str(tmp_path / "twice.db")
    with SQLiteStore(path) as s:
        edge(s, "Alice", "works_at", "Acme")
    regress_to_v5(path)

    with SQLiteStore(path):
        pass
    with SQLiteStore(path) as third:
        # The ALTER branch is skipped and the UPDATE is not reached at all; what proves
        # it is that the columns survive untouched and are still correct.
        assert [c.object for c in third.adjacent("acme", ["alice"])] == ["Acme"]
        assert third._db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_a_fresh_database_arrives_at_the_backfill_with_nothing_to_do(tmp_path):
    """A brand-new file is version 0, so it walks the whole ladder — including a pass
    that would rewrite every claim if there were any. Shape-driven and idempotent, like
    the three migrations before it."""
    path = str(tmp_path / "new.db")
    with SQLiteStore(path) as s:
        assert s._db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        edge(s, "Alice", "works_at", "Acme")
        assert [c.object for c in s.adjacent("acme", ["alice"])] == ["Acme"]


# --- facades -----------------------------------------------------------------


def test_traversal_is_reachable_from_the_scoped_view_and_the_async_facade(mem):
    """Both wrappers forward, which is the whole of what they do — and a public method
    missing from either is the gap `_unwrapped()` and `ScopedMemvara` exist to close."""
    mem.remember("Alice", "reports_to", "Dana")
    mem.remember("Dana", "works_at", "Acme")

    view = mem.scope(user="alice")
    assert rendered(view.neighborhood("Alice", depth=1)) == ["Alice -reports_to-> Dana"]
    assert view.paths_between("Alice", "Acme")[0].hops == 2

    amem = AsyncMemvara(mem)
    assert asyncio.run(amem.neighborhood("Alice", depth=1))[0].hops == 1
    assert asyncio.run(amem.paths_between("Alice", "Acme"))[0].hops == 2


def test_graph_options_reach_the_traverser_and_a_typo_names_the_real_one():
    """`graph_*` is the third tuning prefix, and the point at which routing them with a
    chain of `elif`s started duplicating itself. A rejected option has to suggest the
    real name — the failure being avoided is an option accepted by prefix, forwarded, and
    rejected deep inside a subsystem by a message naming a parameter nobody typed."""
    with Memvara(llm=NullLLM(), embedder=HashingEmbedder(dim=64), graph_beam=7,
                 graph_damping=0.5) as m:
        assert (m.traverser.beam, m.traverser.damping) == (7, 0.5)

    with pytest.raises(TypeError, match="graph_beam"):
        Memvara(llm=NullLLM(), embedder=HashingEmbedder(dim=64), graph_beem=7)


def test_a_store_that_cannot_index_entities_refuses_instead_of_scanning():
    """Degrading to a scan of the tenant per hop is not a graceful fallback, it is a
    denial of service with the right answer. `Memvara.erase` refuses the same way and for
    the same reason: a capability that cannot be faked should say so."""
    class Bare:
        pass

    walker = GraphTraverser(Bare(), PredicateRegistry())

    with pytest.raises(NotImplementedError, match="adjacent"):
        walker.neighborhood("Alice", SCOPE)


def test_an_edge_and_a_path_are_immutable_so_a_caller_cannot_rewrite_the_answer(walker,
                                                                               store):
    """Frozen for the reason `Scope` is: these are handed out of a read and a caller that
    mutated one would be editing the explanation of a result rather than a copy of it."""
    edge(store, "Alice", "knows", "Bob")
    path = walker.neighborhood("Alice", SCOPE, depth=1)[0]

    with pytest.raises(AttributeError):
        path.score = 0.0
    with pytest.raises(AttributeError):
        path.edges[0].strength = 0.0
    assert isinstance(path.edges[0], Edge)
