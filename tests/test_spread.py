"""The graph leg of retrieval: seeding it, ranking it, fusing it, and turning it off.

Grouped by the decision each test defends, because this feature is mostly decisions:

1. **Seeds come off the head of the fused list, not out of the query.** So the seeding
   rule has to be stable across two ingests of one corpus — the fused order breaks ties
   on a `uuid4` and `seed_keys` must not inherit that — and it has to bound the *keys*,
   which is what reaches the store.
2. **Seeds are different entities, not one entity's aliases.** `GraphTraverser` has
   always blocked arriving at a seed, because `neighborhood`'s seeds are several
   spellings of one thing. Applied to `spread` that rule deletes the answer: the edges
   between two seeds are exactly the join the leg exists to make.
3. **A leg that cannot run degrades and says so.** `RemoteStore.adjacent` exists and
   raises, so a `getattr` guard cannot see it, and a search that quietly ran two legs
   for a month is the failure this warning exists to prevent.
4. **Gating happens before the walk, not after it.** A `lookup` query must cost nothing,
   not cost a walk whose score is then multiplied by zero.
"""

from datetime import datetime, timezone

import pytest

from memvara import Memvara, NullLLM
from memvara.embed import HashingEmbedder
from memvara.retrieve.hybrid import DegradedRetrievalWarning, HybridRetriever
from memvara.retrieve.intent import Intent
from memvara.retrieve.spread import rank_paths, seed_keys
from memvara.retrieve.traverse import Edge, GraphTraverser, Path
from memvara.schema import PredicateRegistry
from memvara.store import SQLiteStore
from memvara.types import Claim, Scope

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
SCOPE = Scope("acme", "alice")


@pytest.fixture()
def store():
    s = SQLiteStore(":memory:")
    yield s
    s.close()


@pytest.fixture()
def mem():
    """A store with an org graph in it, and the graph leg switched on.

    `remember()` rather than `add()`: the rule extractor's vocabulary is first-person
    declaratives, so a transcript of these facts would produce no claims and the leg
    under test would have nothing to walk. That gap is real and is measured elsewhere;
    it is not what these tests are about.
    """
    with Memvara(llm=NullLLM(), embedder=HashingEmbedder(dim=64), tenant="acme",
                 user="alice", read_w_graph=1.0) as m:
        for subject, predicate, obj in (
            ("Alice", "reports_to", "Dana"),
            ("Dana", "works_at", "Acme"),
            ("Acme", "headquartered_in", "Tallinn"),
            ("Bruno", "works_at", "Acme"),
            ("Carol", "lives_in", "Lisbon"),
        ):
            m.remember(subject, predicate, obj, recorded_at=T0)
        yield m


def claim(subject: str, obj: str, predicate: str = "works_at") -> Claim:
    return Claim(subject=subject, predicate=predicate, object=obj, scope=SCOPE,
                 recorded_at=T0, valid_from=T0)


def edge(store, subject, predicate, obj, *, scope=SCOPE):
    c = Claim(subject=subject, predicate=predicate, object=obj, scope=scope,
              recorded_at=T0, valid_from=T0)
    store.put_claim(c)
    return c


# --- seeding ------------------------------------------------------------------


def test_seeds_are_ordered_by_content_so_two_ingests_of_one_corpus_agree():
    """The fused order breaks ties on the item id, and an id here is a fresh `uuid4`.

    Inheriting that would make *which entities get walked* a property of which ingest
    ran, so two stores holding identical data would answer the same question with
    different graph legs — the exact failure that made repeated LOCOMO runs drift by
    0.07 points before ties moved onto a content hash.
    """
    first = [(claim("Alice", "Acme"), 0.02), (claim("Bruno", "Acme"), 0.02)]
    # Same content, same scores, fresh ids, opposite order.
    second = [(claim("Bruno", "Acme"), 0.02), (claim("Alice", "Acme"), 0.02)]
    assert {c.id for c, _ in first}.isdisjoint({c.id for c, _ in second})
    assert seed_keys(first, 9) == seed_keys(second, 9)


def test_a_higher_scoring_claim_seeds_first():
    ranked = [(claim("Alice", "Acme"), 0.01), (claim("Bruno", "Zeta"), 0.09)]
    assert seed_keys(ranked, 9) == ("bruno", "zeta", "alice", "acme")


def test_the_limit_bounds_keys_and_not_claims():
    """The key list is what `Store.adjacent` is called with, so it is what costs.

    One claim contributes two keys, which is why a limit of 3 cuts inside the second
    claim rather than after it.
    """
    ranked = [(claim("Alice", "Acme"), 0.09), (claim("Bruno", "Zeta"), 0.08)]
    assert seed_keys(ranked, 3) == ("alice", "acme", "bruno")


def test_an_empty_end_never_becomes_a_seed():
    """`object=""` is how a retraction clears a slot, so it is a stored value.

    Seeding on it would ask the store for everything adjacent to the empty key, which is
    every retraction in the tenant — one hub joining everything to everything.
    """
    assert seed_keys([(claim("Alice", ""), 0.5)], 9) == ("alice",)
    assert seed_keys([(claim("", ""), 0.5)], 9) == ()


# --- ranking a walk -----------------------------------------------------------


def path_of(*claims: Claim, score: float = 1.0) -> Path:
    edges = tuple(Edge(c, False, 1.0) for c in claims)
    nodes = (claims[0].subject_key,) + tuple(c.object_key for c in claims)
    return Path(nodes=nodes, edges=edges, score=score)


def test_a_claim_on_two_paths_takes_the_better_one_rather_than_the_sum():
    """Path scores are relevances, not evidence counts.

    Summing would make a hub that sits on nine weak chains outrank a claim lying on one
    strong one, which is how a graph leg turns into a popularity ranking.
    """
    hub = claim("Acme", "Tallinn", "headquartered_in")
    weak = path_of(claim("Alice", "Acme"), hub, score=0.2)
    strong = path_of(claim("Bruno", "Acme"), hub, score=0.9)
    scores = dict(rank_paths([weak, strong, weak]))
    assert scores[hub.id] == pytest.approx(0.9)


def test_every_claim_on_a_path_is_returned_not_only_the_far_end():
    """The middle of a chain is what makes the answer checkable.

    "Acme is in Tallinn" handed over without "Alice works at Acme" beside it is an
    assertion rather than a derivation, and the caller has no way to audit the hop.
    """
    first, second = claim("Alice", "Acme"), claim("Acme", "Tallinn", "headquartered_in")
    assert {cid for cid, _ in rank_paths([path_of(first, second)])} == {first.id,
                                                                       second.id}


def test_no_paths_ranks_nothing_rather_than_raising():
    assert rank_paths([]) == []


# --- walking from several distinct entities -----------------------------------


def test_an_edge_between_two_seeds_is_the_answer_rather_than_a_cycle(store):
    """The regression this entry point was written around.

    `GraphTraverser` blocks arriving at any seed, because `neighborhood`'s seeds are
    several spellings of one entity and an edge onto another spelling is a self-loop.
    `spread`'s seeds are five different entities off the head of a fused list, so most
    edges have *both* ends in the set — and with the rule applied the walk returned zero
    paths, which reads exactly like a corpus with no structure in it.
    """
    written = edge(store, "Alice", "reports_to", "Dana")
    walker = GraphTraverser(store, PredicateRegistry())
    paths = walker.spread(("alice", "dana"), SCOPE)
    assert sorted(p.render() for p in paths) == ["Alice -reports_to-> Dana",
                                                 "Dana <-reports_to- Alice"]
    # Two seeds on one edge produce it once from each end, which is the honest answer to
    # "what is around each of these entities" and is not double-counting downstream:
    # `rank_paths` keys on the claim, so the leg ranks one row.
    assert [cid for cid, _ in rank_paths(paths)] == [written.id]

    # And the single-entity entry point still refuses the same edge, walked from both of
    # one entity's names.
    assert walker.neighborhood("Alice", SCOPE, entity_keys=("dana",)) == []


def test_spread_takes_keys_as_given_and_folds_nothing(store):
    """A retrieval seed is `Claim.subject_key`, already folded by the write path.

    Handing it back to the surface-form fold is at best a no-op and at worst derives a
    different key from the canonical one, which would walk from an entity nothing is
    filed under.
    """
    edge(store, "Acme, Inc.", "headquartered_in", "Tallinn")
    walker = GraphTraverser(store, PredicateRegistry())
    assert walker.spread(("acme",), SCOPE)
    assert walker.spread(("Acme, Inc.",), SCOPE) == [], (
        "an unfolded surface form is not an entity key and must not be treated as one"
    )


def test_a_walk_with_no_usable_keys_returns_nothing(store):
    edge(store, "Alice", "reports_to", "Dana")
    walker = GraphTraverser(store, PredicateRegistry())
    assert walker.spread((), SCOPE) == []
    assert walker.spread(("",), SCOPE) == []


# --- the leg inside search ----------------------------------------------------


def test_the_graph_leg_promotes_a_claim_two_hops_from_the_query(mem):
    """The question names Alice; the answer is a fact about Acme.

    Neither lookup leg can connect them — the claim holding the answer shares no
    vocabulary with the question — so this is the case the leg exists for.
    """
    question = "which city is the employer of Alice's manager based in?"
    found = {r.claim.text: r for r in mem.search(question, k=5)}
    answer = found["Acme headquartered in Tallinn"]
    assert answer.explain.graph_rank is not None
    assert answer.explain.intent == Intent.RELATIONAL.value

    off = Memvara(llm=NullLLM(), embedder=HashingEmbedder(dim=64), tenant="acme",
                  user="alice")
    for subject, predicate, obj in (("Alice", "reports_to", "Dana"),
                                    ("Dana", "works_at", "Acme"),
                                    ("Acme", "headquartered_in", "Tallinn"),
                                    ("Bruno", "works_at", "Acme"),
                                    ("Carol", "lives_in", "Lisbon")):
        off.remember(subject, predicate, obj, recorded_at=T0)
    plain = {r.claim.text: r for r in off.search(question, k=5)}
    assert plain["Acme headquartered in Tallinn"].explain.graph_rank is None
    assert (answer.score > plain["Acme headquartered in Tallinn"].score), (
        "the leg has to change the score, not merely populate a field"
    )
    off.close()


def test_a_claim_the_walk_did_not_reach_says_so_rather_than_going_unmentioned(mem):
    """`graph_rank=None` beside a populated `graph_rank` elsewhere is the finding.

    Carol is in the store and connected to nothing, so she is the control: the leg ran,
    and it did not rank her.
    """
    results = {r.claim.text: r for r in mem.search(
        "which city is the employer of Alice's manager based in?", k=9)}
    assert results["Carol lives in Lisbon"].explain.graph_rank is None
    assert "graph#" in repr(results["Acme headquartered in Tallinn"])
    assert "graph#" not in repr(results["Carol lives in Lisbon"])


def test_the_walk_is_skipped_entirely_on_a_lookup_query(mem):
    """Gated before the traverser is called, not scored to zero afterwards.

    A `lookup` question is answered by one row, and paying a two-hop expansion to
    confirm it is latency spent on a question that was already answered. Counting store
    calls is the only way to tell "gated" from "ran and was weighted away".
    """
    calls = []
    real = mem.store.adjacent
    mem.store.adjacent = lambda *a, **kw: (calls.append(1), real(*a, **kw))[1]

    lookup = mem.search("where does Bruno work?", k=5)
    assert [r.explain.intent for r in lookup] == [Intent.LOOKUP.value] * len(lookup)
    assert calls == []

    mem.search("which city is the employer of Alice's manager based in?", k=5)
    assert calls, "a relational query must still reach the store"


def test_turning_intent_weighting_off_runs_every_query_at_the_configured_weights(mem):
    """The escape hatch, and the thing that makes the stage attributable.

    A ranking difference between these two calls is this stage's; without the switch it
    would have to be argued about.
    """
    reader = HybridRetriever(mem.store, mem.embedder, mem.registry, w_graph=1.0,
                             traverser=mem.traverser, intent_weighting=False)
    results = reader.search("where does Bruno work?", mem.default_scope, k=5)
    assert all(r.explain.intent is None for r in results)
    assert any(r.explain.graph_rank is not None for r in results), (
        "ungated, the walk runs on a lookup query too"
    )


def test_a_store_that_cannot_traverse_degrades_once_and_says_so(mem):
    """`RemoteStore.adjacent` is present and raises, which no `getattr` guard can see.

    Warned once per retriever rather than once per query: a store that cannot traverse
    cannot traverse for the whole process, and a warning per search buries the finding
    under itself.
    """
    def refuse(*_a, **_kw):
        raise NotImplementedError("cloud mode has no adjacency endpoint")

    mem.store.adjacent = refuse
    question = "which city is the employer of Alice's manager based in?"
    with pytest.warns(DegradedRetrievalWarning, match="two legs instead of three"):
        first = mem.search(question, k=5)
    assert first, "degrading means a narrower answer, never no answer"
    assert all(r.explain.graph_rank is None for r in first)

    import warnings as _warnings
    with _warnings.catch_warnings(record=True) as later:
        _warnings.simplefilter("always")
        mem.search(question, k=5)
    assert [w for w in later if issubclass(w.category, DegradedRetrievalWarning)] == []


def test_a_candidate_set_of_retractions_alone_seeds_nothing(store):
    """Every early return in `_graph_search` is a different fact; this is the last one.

    A retraction's object is the empty string, so a candidate set made entirely of them
    folds to no seeds at all — and the leg must return without calling the traverser
    rather than walking from the empty key.
    """
    walked = []
    walker = GraphTraverser(store, PredicateRegistry())
    walker.spread = lambda *a, **kw: walked.append(a) or []
    reader = HybridRetriever(store, HashingEmbedder(dim=64), PredicateRegistry(),
                             w_graph=1.0, traverser=walker, intent_weighting=False)
    store.put_claim(Claim(subject="", predicate="lives_in", object="", scope=SCOPE,
                          recorded_at=T0, valid_from=T0, text="lisbon"))
    assert reader.search("lisbon", SCOPE, k=5) is not None
    assert walked == []
