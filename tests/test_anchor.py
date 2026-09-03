"""Anchoring: what tied a result to the question, and the filter built on it.

Grouped by the decision each test defends.

1. **The signal is read off the rows, not off the score.** A question about an entity
   the store has never heard of scores like any other question — the Agent Memory
   Benchmark's Project Chronos case scores *above* two genuine answers — so no floor can
   catch it. Whether the question names the claim's subject or object can.
2. **A derivation counts, and only a derivation.** A claim the graph leg reached by
   walking out of a claim the question names is tied to the question by that path. A
   claim reached by walking out of an unanchored seed is not: the walk proves nothing
   about a question it did not start from.
3. **`anchored=True` is a filter with the same retry as `memory_types`.** A named claim
   with little vocabulary in common with the question sits past the first cut exactly as
   a filtered memory type does, and the second pass is what finds it.
4. **What a key comparison cannot see is handled once.** The self subject and a
   possessive are the two, and a learned alias is the third.
"""
from datetime import datetime, timezone

import pytest

from memvara import Memvara, NullLLM
from memvara.embed import HashingEmbedder
from memvara.retrieve.anchor import SELF_SUBJECT, anchor_of, query_tokens
from memvara.retrieve.hybrid import EpisodeResult, HybridRetriever
from memvara.types import Claim, owner_key

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)

#: The org chart the Agent Memory Benchmark's chained questions are asked over, reduced
#: to the claims those questions need, plus the neighbours that make the negatives hard.
ORG = (
    ("alice", "works_on", "Project Atlas"),
    ("Project Atlas", "deploy_region", "eu-west-1"),
    ("bob", "works_at", "Globex"),
    ("Globex", "hq_city", "Munich"),
    ("Initech", "hq_city", "Austin"),
    ("ivan", "lives_in", "Lisbon"),
    ("judy", "lives_in", "London"),
    ("grace", "lives_in", "London"),
)


def build(**kw) -> Memvara:
    mem = Memvara(llm=NullLLM(), embedder=HashingEmbedder(dim=128), tenant="acme",
                  user="alice", **kw)
    for subject, predicate, obj in ORG:
        mem.remember(subject, predicate, obj, recorded_at=T0)
    return mem


@pytest.fixture()
def mem():
    with build() as m:
        yield m


@pytest.fixture()
def graph_mem():
    with build(read_w_graph=1.0) as m:
        yield m


# -- 1. the signal ------------------------------------------------------------

def test_every_result_says_which_end_of_the_claim_the_question_named(mem):
    """`Explanation.anchor` is populated on every result, filter or no filter."""
    rows = {r.claim.subject: r.explain.anchor
            for r in mem.search("Which region is Project Atlas deployed to?", k=5)}
    assert rows["Project Atlas"] == "subject"
    assert rows["alice"] == "object"          # alice works_on Project Atlas
    stranger = mem.search("Where does Oscar live?", k=1)[0]
    assert stranger.explain.anchor is None


def test_a_question_about_an_unknown_entity_returns_nothing_when_anchored(mem):
    """The benchmark's `negative` category, at the shipped `min_score` of 0.0.

    Without the filter the store answers "where does Oscar live" with somebody else's
    city, at a score that looks like every other answer. The filter returns nothing,
    because no row it holds is about Oscar.
    """
    assert mem.search("Where does Oscar live?", k=5)          # answers regardless
    assert mem.search("Where does Oscar live?", k=5, anchored=True) == []


def test_a_sibling_that_shares_a_word_is_not_the_entity_asked_about(mem):
    """The case no threshold reaches, and the reason this exists.

    "Which region is Project Chronos deployed to" is answered from Project Atlas's row
    at 0.450 on the benchmark — above genuine answers at 0.410 — because it looks exactly
    like a question the store can answer. Every token of the key has to be present:
    `project` alone does not name `Project Atlas`.
    """
    plain = mem.search("Which region is Project Chronos deployed to?", k=5)
    assert plain and plain[0].claim.subject == "Project Atlas"
    assert plain[0].explain.anchor is None
    assert mem.search("Which region is Project Chronos deployed to?", k=5,
                      anchored=True) == []


def test_the_filter_keeps_every_answerable_question_answerable(mem):
    """A question about an entity the store holds loses nothing to the filter."""
    for question, subject in (("Where does Ivan live?", "ivan"),
                              ("Where does Judy live?", "judy"),
                              ("Which region is Project Atlas deployed to?",
                               "Project Atlas")):
        rows = mem.search(question, k=5, anchored=True)
        assert rows and rows[0].claim.subject == subject, question
        assert all(r.explain.anchor is not None for r in rows)


def test_a_lucky_vocabulary_hit_is_not_a_grounded_answer(mem):
    """The one benchmark question the filter costs at the shipped defaults, on purpose.

    On the benchmark corpus "In which city is Bob's employer headquartered" is answered
    correctly by plain search — `Globex/hq_city=Munich` ranks first — but by vocabulary
    alone: the row shares no entity with the question, and `Initech/hq_city=Austin` sits
    right behind it at almost the same score. A caller who asked for grounded answers is
    right to be handed Bob's own rows instead. The next test is where the walk earns it
    back.
    """
    plain = {r.claim.subject: r.explain.anchor
             for r in mem.search("In which city is Bob's employer headquartered?", k=5)}
    assert plain["Globex"] is None and plain["Initech"] is None
    grounded = mem.search("In which city is Bob's employer headquartered?", k=5,
                          anchored=True)
    assert grounded and all(r.claim.subject == "bob" for r in grounded)
    assert all(r.explain.anchor == "subject" for r in grounded)


# -- 2. derivations -----------------------------------------------------------

def test_a_claim_the_walk_reached_from_a_named_entity_is_anchored_by_the_path(graph_mem):
    """The multi-hop half of #129, read off the same signal.

    With the graph leg on, `Globex/hq_city=Munich` is still not named by the question —
    and it is now tied to it, because the walk reached it from `bob/works_at=Globex`,
    which is. `anchor="path"` is that derivation, and the filter keeps it.
    """
    rows = graph_mem.search("In which city is Bob's employer headquartered?", k=5,
                            anchored=True)
    by_subject = {r.claim.subject: r for r in rows}
    assert by_subject["Globex"].explain.anchor == "path"
    assert by_subject["Globex"].explain.graph_rank is not None
    assert by_subject["bob"].explain.anchor == "subject"


def test_a_walk_out_of_an_unanchored_seed_ties_nothing_to_the_question():
    """A walk that started nowhere the question named proves nothing about it.

    The intent gate is off so the walk runs on the Chronos question at all — its seeds
    come off the fused head, which is Project Atlas — and nothing it reaches counts,
    because the seed itself was not named. Otherwise every negative would be answered
    from the neighbourhood of whatever the lookup legs guessed.
    """
    with build(read_w_graph=1.0, read_intent_weighting=False) as mem:
        rows = mem.search("Which region is Project Chronos deployed to?", k=5)
        assert any(r.explain.graph_rank is not None for r in rows), "premise: walked"
        assert all(r.explain.anchor is None for r in rows)
        assert mem.search("Which region is Project Chronos deployed to?", k=5,
                          anchored=True) == []


# -- 3. the filter ------------------------------------------------------------

def test_an_anchored_claim_past_the_first_cut_is_found_on_the_retry(monkeypatch):
    """Filter starvation, and the retry `memory_types` already gets.

    Three rows about Bea say "city" and "live" over and over; Ada's one row says
    neither and only its subject names her. At `candidate_multiplier=1`, the candidate
    floor off and `k=1`, the first pass fetches one candidate per leg, both Bea's, and
    filters both. Without the retry the answer would be an empty list with Ada's row
    one position past the cut.
    """
    with Memvara(llm=NullLLM(), embedder=HashingEmbedder(dim=128), tenant="acme",
                 user="alice", read_candidate_multiplier=1, read_candidate_floor=0) as mem:
        for n in range(3):
            mem.remember("Bea", f"note_{n}", f"note {n}",
                         text="the city where Bea lives is a city she lives in",
                         recorded_at=T0)
        mem.remember("Ada", "lives_in", "Porto", text="I live in Porto now",
                     recorded_at=T0)

        limits: list[int] = []
        original = HybridRetriever._gather

        def spy(self, query, scope, limit, *args, **kw):
            limits.append(limit)
            return original(self, query, scope, limit, *args, **kw)

        monkeypatch.setattr(HybridRetriever, "_gather", spy)
        rows = mem.search("which city does Ada live in?", k=1, anchored=True)

    assert [r.claim.subject for r in rows] == ["Ada"]
    assert len(limits) == 2 and limits[1] > limits[0], (
        "the second, wider pass is what found the row"
    )


def test_episodes_are_not_filtered_because_a_turn_has_no_subject(mem):
    """`anchored` is a filter on claims. A raw turn names nobody in the structured
    sense, so the episode leg returns what it always returned."""
    mem.add("Oscar said the deploy is on Friday.", role="system")
    rows = mem.search("what did Oscar say?", k=5, include_episodes=True, anchored=True)
    assert rows and all(isinstance(r, EpisodeResult) for r in rows)


def test_recall_and_ask_carry_the_flag(mem):
    """The two composed reads say nothing rather than something about somebody else."""
    assert mem.recall("Where does Oscar live?")                 # renders a stranger
    assert mem.recall("Where does Oscar live?", anchored=True) == ""
    assert mem.ask("Where does Oscar live?").readings
    assert mem.ask("Where does Oscar live?", anchored=True).readings == ()
    assert mem.recall("Where does Ivan live?", anchored=True)   # still answers


# -- 4. what a key comparison cannot see ---------------------------------------

def test_the_self_subject_is_named_by_a_pronoun():
    """A first-person statement is filed under `user` and asked about as "I".

    Pinned end to end through the fast path rather than against the constant, so the
    two places that spell the subject — `write/fast.py` and `write/pipeline.py` — cannot
    drift from `anchor.SELF_SUBJECT` without this going red.
    """
    with Memvara(llm=NullLLM(), embedder=HashingEmbedder(dim=128), user="alice") as mem:
        receipt = mem.add("I live in Lisbon.")
        assert receipt.added and receipt.added[0].subject == SELF_SUBJECT
        rows = mem.search("where do I live?", k=3, anchored=True)
        assert [r.explain.anchor for r in rows] == ["subject"]
        assert mem.search("where does Oscar live?", k=3, anchored=True) == []


def test_a_possessive_is_a_mention():
    """`entities._tokens` drops apostrophes, so "Bob's" would fold to `bobs`."""
    claim = Claim(subject="Bob", predicate="works_at", object="Globex")
    assert anchor_of(claim, query_tokens("where is Bob's office?")) == "subject"
    assert anchor_of(claim, query_tokens("where is Bob’s office?")) == "subject"


def test_a_learned_alias_anchors_the_claims_filed_under_the_canonical_key(mem):
    """The registry's spellings widen the match, under the reader's own owner."""
    mem.remember("IBM", "based_in", "Armonk", recorded_at=T0)
    before = mem.search("where is Big Blue based?", k=3, anchored=True)
    assert all(r.claim.subject != "IBM" for r in before)

    owner = owner_key(mem.default_scope)
    mem.writer.reconciler.entities.learn_alias(owner, "IBM", "Big Blue")
    after = mem.search("where is Big Blue based?", k=3, anchored=True)
    assert [r.claim.subject for r in after] == ["IBM"]
    assert after[0].explain.anchor == "subject"


def test_a_retriever_built_without_a_registry_anchors_on_the_key_alone(mem):
    """`bench/` and a third-party `Store` construct `HybridRetriever` directly."""
    reader = HybridRetriever(mem.store, mem.embedder, mem.registry)
    rows = reader.search("Where does Ivan live?", mem.default_scope, k=3, anchored=True)
    assert [r.claim.subject for r in rows] == ["ivan"]
    assert reader.search("Where does Oscar live?", mem.default_scope, k=3,
                         anchored=True) == []


# -- what the review found -----------------------------------------------------

def test_a_derivation_starts_at_the_named_entity_and_not_at_its_value():
    """A walk out of the *value* end of a named claim reaches the rows that share it.

    From `Project Atlas/deploy_region=eu-west-1`, every other project deployed to
    `eu-west-1` is one hop from the value, scored 1.0, on the very predicate asked — and
    each was coming back marked `"path"`, ranked by the walk above the genuine answer.
    Those are derivations from the answer, not from the question. Only the named end is
    an origin now. The siblings are still reachable the long way round — out of Project
    Atlas and back through the region, two hops — and are labelled as the derivations
    they are at that distance; what changes is that the row the question is about ranks
    first, and a sibling one hop from the value no longer counts as tied to the question
    by that hop alone.
    """
    with build(read_w_graph=1.0, read_intent_weighting=False) as mem:
        for sibling in ("Project Zeta", "Project Omega"):
            mem.remember(sibling, "deploy_region", "eu-west-1", recorded_at=T0)
        rows = mem.search("Which region is Project Atlas deployed to?", k=5,
                          anchored=True)
        assert rows[0].claim.subject == "Project Atlas"
        assert rows[0].explain.anchor == "subject"
        siblings = [r for r in rows if r.claim.subject in ("Project Zeta", "Project Omega")]
        assert siblings and all(r.explain.anchor == "path" for r in siblings)
        assert all(r.score < rows[0].score for r in siblings)


def test_the_second_person_does_not_name_the_self_subject():
    """"Do you know where Oscar lives" addresses the agent, and `us` is a country.

    Both anchored every `user` row on exactly the questions the filter exists to return
    nothing for; the pronoun set is first person only.
    """
    with Memvara(llm=NullLLM(), embedder=HashingEmbedder(dim=128), user="alice") as mem:
        mem.add("I live in Lisbon.")
        assert mem.search("Do you know where Oscar lives?", k=5, anchored=True) == []
        assert mem.search("Which US region is Project Chronos deployed to?", k=5,
                          anchored=True) == []
        assert mem.search("where do I live?", k=5, anchored=True)


def test_every_apostrophe_the_fold_drops_is_a_possessive_here():
    """iOS and Word emit U+02BC; the fold already treats it as an apostrophe."""
    claim = Claim(subject="Bob", predicate="works_at", object="Globex")
    for apostrophe in "'‘’ʼ´`":
        question = f"where is Bob{apostrophe}s office?"
        assert anchor_of(claim, query_tokens(question)) == "subject", repr(apostrophe)


def test_the_walk_is_not_run_under_the_filter_when_nothing_is_named(monkeypatch):
    """No path can start from an entity the question did not name, so nothing the walk
    finds could survive `anchored=True`. It used to run anyway — and again on the
    widened retry — on every question about a stranger."""
    with build(read_w_graph=1.0, read_intent_weighting=False) as mem:
        calls: list[int] = []
        original = type(mem.traverser).spread

        def spy(self, *args, **kw):
            calls.append(1)
            return original(self, *args, **kw)

        monkeypatch.setattr(type(mem.traverser), "spread", spy)
        assert mem.search("Where does Oscar live?", k=5, anchored=True) == []
        assert calls == [], "the walk ran with nothing to start from"
        mem.search("Where does Oscar live?", k=5)
        assert calls, "premise: without the filter the walk still runs"
