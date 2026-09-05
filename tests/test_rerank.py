"""Reranking: the stage, the backends, and the promise the stage must not break.

The load-bearing test in this file is not any of the ranking ones. It is
`test_the_default_configuration_never_imports_a_reranker_backend`, plus its sibling that
runs a whole write-and-read cycle with the network unplugged. The library's default
configuration is "numpy and nothing else, offline, no API key", and a reranker is a
model — so the way this feature fails is not by ranking badly, it is by quietly costing
every user who never asked for it an import of torch or a call to a model host.

Everything here runs with `sentence-transformers` absent, which is also how CI runs it:
the cross-encoder is exercised through an injected fake and through a fake module in
`sys.modules`, so the import path is covered without the extra being installed.
"""

from __future__ import annotations

import ast
import socket
import subprocess
import sys
import types
from dataclasses import dataclass, field

import pytest

from memvara import Memvara
from memvara.compat.mem0 import Mem0CompatError, Memory
from memvara.embed import HashingEmbedder
from memvara.llm import NullLLM
from memvara.rerank import CoverageReranker, NullReranker, Reranker, rerank
from memvara.rerank.cross import DEFAULT_MODEL, CrossEncoderReranker
from memvara.rerank.lexical import _min_window
from memvara.retrieve import HybridRetriever
from memvara.schema import PredicateRegistry
from memvara.store import SQLiteStore
from memvara.types import Explanation, Scope

REPO_ROOT = str(__import__("pathlib").Path(__file__).resolve().parent.parent)


# -- doubles ---------------------------------------------------------------------------


@dataclass
class Row:
    """The least thing `rerank` accepts: text to score and somewhere to record it."""

    text: str
    explain: Explanation = field(default_factory=Explanation)


class Keyword:
    """Scores 1.0 for documents containing `word`, 0.0 otherwise. Deterministic."""

    name = "keyword"

    def __init__(self, word: str) -> None:
        self.word = word
        self.calls: list[tuple[str, int]] = []

    def score(self, query: str, documents):
        self.calls.append((query, len(documents)))
        return [1.0 if self.word in d.lower() else 0.0 for d in documents]


class FakeCrossEncoder:
    """Stands in for `sentence_transformers.CrossEncoder`, which is not installed."""

    def __init__(self, model: str = "fake") -> None:
        self.model = model
        self.batches: list[int] = []

    def predict(self, pairs, batch_size: int = 32):
        self.batches.append(batch_size)
        return [float(len(doc)) for _, doc in pairs]


# -- fixtures --------------------------------------------------------------------------


@pytest.fixture
def store() -> SQLiteStore:
    return SQLiteStore(":memory:")


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=256)


def claims_memvara(**kw) -> Memvara:
    """A store holding real claims, which is what the mem0 shim searches.

    `Memory.search` does not pass `include_episodes`, so a store built the way
    `episodes_memvara` builds one answers it with nothing at all — every turn is an
    episode and no claim was ever extracted. `remember()` writes claims directly, with
    no model, which is the offline way to get some.

    Pinned for the reason `episodes_memvara` sets out at length, which applies here too
    and was simply missed when that one was fixed: `default_embedder()` returns a
    sentence-transformers model as soon as the package is importable, and
    `memvara[rerank]` installs one. Nothing below depends on the vector ordering — three
    claims are asked for at `top_k=4`, so the set comes back whole either way — but
    "either way" is the point: an unpinned factory loads a transformer per call and makes
    the run a property of the machine. `dim=512` is not a new choice, it is exactly what
    `default_embedder()` returns when sentence-transformers is absent.
    """
    kw.setdefault("embedder", HashingEmbedder(dim=512))
    mem = Memvara(llm=NullLLM(), user="alice", **kw)
    for predicate, obj in (("has_pet", "a calm greyhound dog"),
                           ("lives_in", "a small flat"),
                           ("dislikes", "loud dog parks")):
        mem.remember("user", predicate, obj)
    return mem


#: The question every episode test below asks. Its content terms after `analyze` are
#: dog / suits / small / flat.
QUESTION = "which dog suits a small flat?"

TURNS = (
    "The greyhound is a calm dog and suits a small flat.",
    "I keep meaning to read more about dog training.",
    "My flat gets very little light in winter.",
    "A dog, a dog, a dog. Mostly I think about dogs.",
    "Dog owners in this city complain a lot.",
    "The flat below mine has a dog that barks.",
)


def episodes_memvara(**kw) -> Memvara:
    """Six turns, ranked by the shipped offline configuration and nothing else.

    `read_max_episodes` is raised because the library's default of 3 assumes raw turns
    are a short tail on a list of extracted facts; with `NullLLM` nothing is extracted,
    so here they are the entire candidate pool and a cap of 3 would make "rank 4" a
    thing that cannot exist.

    **The embedder is pinned, not defaulted.** `default_embedder()` returns
    `LocalEmbedder` when sentence-transformers is importable and `HashingEmbedder` when
    it is not, so leaving it out made the fused ranking — and therefore which candidate
    sits outside a given `k` — depend on what happened to be installed. That is not "the
    shipped offline configuration", it is whichever one the machine has, and it silently
    broke `test_the_reranker_can_promote_a_candidate_from_below_the_callers_k` the day
    `memvara[rerank]` was installed to measure the cross-encoder.
    """
    kw.setdefault("read_max_episodes", 12)
    kw.setdefault("embedder", HashingEmbedder(dim=512))
    mem = Memvara(llm=NullLLM(), user="alice", **kw)
    mem.add([{"role": "user", "content": t} for t in TURNS])
    return mem


# -- the stage -------------------------------------------------------------------------


def test_the_stage_reorders_only_the_head_and_leaves_the_tail_alone() -> None:
    rows = [Row("nothing here"), Row("a DOG"), Row("also a dog, but out of reach")]
    out = rerank(Keyword("dog"), "dog", rows, top_n=2)
    assert [r.text for r in out] == ["a DOG", "nothing here",
                                     "also a dog, but out of reach"]


def test_only_the_candidates_the_reranker_saw_carry_a_rerank_score() -> None:
    """`None` means "not scored" and `0.0` means "scored zero", which is the whole
    reason `Explanation.rerank_score` was reserved as an optional rather than a float."""
    rows = [Row("nothing here"), Row("a dog"), Row("unseen")]
    rerank(Keyword("dog"), "dog", rows, top_n=2)
    assert [r.explain.rerank_score for r in rows] == [0.0, 1.0, None]


def test_the_stage_never_shows_the_backend_more_than_top_n_documents() -> None:
    """The bounded cost is the entire answer to the objection to reranking."""
    backend = Keyword("dog")
    rerank(backend, "dog", [Row(f"row {i}") for i in range(500)], top_n=10)
    assert backend.calls == [("dog", 10)]


def test_a_tie_keeps_the_fused_order_rather_than_an_arbitrary_one() -> None:
    """Retrieval here is reproducible to a content hash; an unstable sort would spend
    that for nothing, and the drift would only show up as a benchmark that wobbles."""
    rows = [Row("dog one"), Row("dog two"), Row("dog three")]
    out = rerank(Keyword("dog"), "dog", rows, top_n=3)
    assert [r.text for r in out] == ["dog one", "dog two", "dog three"]


def test_the_stage_does_not_touch_scores_only_order() -> None:
    """`Result.score` is what `min_score` thresholds on, so folding a reranker's logit
    into it would silently change what a calibrated floor means."""
    rows = [Row("nothing"), Row("a dog")]
    for row, value in zip(rows, (0.9, 0.1)):
        row.explain.final_score = value
    rerank(Keyword("dog"), "dog", rows, top_n=2)
    assert [r.explain.final_score for r in rows] == [0.9, 0.1]


@pytest.mark.parametrize("top_n", [0, -1])
def test_a_non_positive_top_n_reranks_nothing_and_calls_no_backend(top_n: int) -> None:
    """A backend that bills per call must not be billed for a query it never scored."""
    backend = Keyword("dog")
    rows = [Row("a dog"), Row("nothing")]
    assert [r.text for r in rerank(backend, "dog", rows, top_n=top_n)] == [
        "a dog", "nothing"]
    assert backend.calls == []


def test_an_empty_candidate_list_calls_no_backend() -> None:
    backend = Keyword("dog")
    assert rerank(backend, "dog", [], top_n=10) == []
    assert backend.calls == []


def test_a_backend_returning_the_wrong_number_of_scores_is_refused() -> None:
    """Silently zipping a short list truncates the candidate set, which would look like
    a ranking change rather than like the bug it is."""

    class Short:
        def score(self, query, documents):
            return [1.0]

    with pytest.raises(ValueError, match="1 scores for 2 documents"):
        rerank(Short(), "q", [Row("a"), Row("b")], top_n=2)


def test_null_reranker_scores_zero_and_changes_nothing() -> None:
    rows = [Row("a"), Row("b"), Row("c")]
    out = rerank(NullReranker(), "q", rows, top_n=3)
    assert [r.text for r in out] == ["a", "b", "c"]
    assert [r.explain.rerank_score for r in rows] == [0.0, 0.0, 0.0]


def test_the_protocol_is_satisfied_by_anything_with_a_score_method() -> None:
    """One method wide on purpose: the adapters people write around a hosted rerank
    endpoint are two lines, and a wider protocol would exclude them for nothing."""
    assert isinstance(NullReranker(), Reranker)
    assert isinstance(Keyword("x"), Reranker)
    assert not isinstance(object(), Reranker)


# -- the coverage reranker -------------------------------------------------------------


def test_coverage_beats_repetition() -> None:
    """The signal BM25 structurally cannot produce: a candidate matching one query term
    three times loses to one matching both terms once."""
    scores = CoverageReranker().score(
        "dog flat", ["dog dog dog", "a calm dog suits a small flat"])
    assert scores[1] > scores[0]


def test_proximity_orders_candidates_that_cover_the_query_equally() -> None:
    tight, loose = CoverageReranker().score(
        "dog flat",
        ["a dog in a flat",
         "a dog " + " ".join(["and"] * 40) + " and separately a flat"],
    )
    assert tight > loose


def test_proximity_weight_zero_is_coverage_alone() -> None:
    tight, loose = CoverageReranker(proximity_weight=0.0).score(
        "dog flat", ["a dog in a flat", "a dog " + "x " * 40 + "a flat"])
    assert tight == loose == 1.0


def test_a_candidate_matching_nothing_scores_zero() -> None:
    assert CoverageReranker().score("dog flat", ["entirely unrelated"]) == [0.0]


def test_a_query_with_no_content_terms_abstains_rather_than_fabricating_an_order() -> None:
    """The same guard the lexical leg has: a ranking assembled from stopwords is not a
    weak ranking, it is a fabricated one.

    The candidates deliberately *do* contain the query's stopwords, so an implementation
    that forgot to abstain would score them well above zero and be caught. Candidates
    sharing nothing with the query would score zero either way and prove nothing.
    """
    assert CoverageReranker().score(
        "what is it about?", ["what is it about", "about what it is"]) == [0.0, 0.0]


@pytest.mark.parametrize("weight", [-0.01, 1.01])
def test_an_out_of_range_proximity_weight_is_refused_at_construction(weight: float) -> None:
    with pytest.raises(ValueError, match=r"proximity_weight must be in \[0, 1\]"):
        CoverageReranker(proximity_weight=weight)


def test_the_coverage_reranker_names_its_configuration() -> None:
    assert repr(CoverageReranker(0.5)) == "<CoverageReranker coverage:0.5>"


def test_the_minimum_window_is_the_shortest_span_holding_every_matched_term() -> None:
    assert _min_window([(0, "a"), (1, "b")], 2) == 2
    assert _min_window([(0, "a"), (5, "b"), (6, "a")], 2) == 2
    assert _min_window([(3, "a")], 1) == 1


# -- the cross-encoder, without the extra installed ------------------------------------


def test_the_cross_encoder_accepts_an_injected_model() -> None:
    """The injection point exists so one loaded model can be shared across instances;
    it is also the only way to test this class without the extra."""
    encoder = FakeCrossEncoder()
    ranker = CrossEncoderReranker(encoder=encoder, batch_size=8)
    assert ranker.score("q", ["ab", "abcd"]) == [2.0, 4.0]
    assert encoder.batches == [8]


def test_the_cross_encoder_scores_nothing_for_an_empty_candidate_list() -> None:
    encoder = FakeCrossEncoder()
    assert CrossEncoderReranker(encoder=encoder).score("q", []) == []
    assert encoder.batches == []


def test_the_cross_encoder_identity_names_the_model_not_the_class() -> None:
    """Two cross-encoders of one architecture rank differently, and a `rerank_score` in
    a log is unreadable without knowing which produced it."""
    ranker = CrossEncoderReranker("some/model", encoder=FakeCrossEncoder())
    assert ranker.name == "cross-encoder:some/model"
    assert repr(ranker) == "<CrossEncoderReranker cross-encoder:some/model batch=32>"


def test_the_cross_encoder_loads_its_model_at_construction_not_at_first_query(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Same fail-fast placement as the hosted LLM backends' API-key check: a server that
    starts clean and dies on the first query that reaches reranking has hidden the
    failure until after the deployment looked healthy."""
    fake = types.ModuleType("sentence_transformers")
    loaded: list[str] = []

    def CrossEncoder(model: str):  # noqa: N802 - mirrors the SDK's class name
        loaded.append(model)
        return FakeCrossEncoder(model)

    fake.CrossEncoder = CrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

    ranker = CrossEncoderReranker()
    assert loaded == [DEFAULT_MODEL]          # before any score() call
    assert ranker.score("q", ["abc"]) == [3.0]


def test_the_cross_encoder_without_its_extra_names_the_extra(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """`ModuleNotFoundError: No module named 'sentence_transformers'` sends the reader to
    the SDK's install page, where they never learn the extra exists. `None` in
    `sys.modules` is CPython's own "this import is blocked" sentinel, so this asserts the
    same thing whether or not the package happens to be installed."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError, match=r"memvara\[rerank\]"):
        CrossEncoderReranker()


# -- the package surface ---------------------------------------------------------------


def test_the_backends_are_lazy_attributes_rather_than_eager_imports() -> None:
    import memvara.rerank as pkg

    assert pkg.CoverageReranker is CoverageReranker
    assert pkg.CrossEncoderReranker is CrossEncoderReranker
    assert pkg.DEFAULT_MODEL == DEFAULT_MODEL
    assert set(pkg.__all__) == {"Reranker", "NullReranker", "Rankable", "rerank",
                                "CoverageReranker", "CrossEncoderReranker",
                                "DEFAULT_MODEL"}


def test_an_unknown_attribute_on_the_package_is_an_attribute_error() -> None:
    import memvara.rerank as pkg

    with pytest.raises(AttributeError, match="no attribute 'Nope'"):
        pkg.Nope


# -- the promise -----------------------------------------------------------------------


def test_the_default_configuration_never_imports_a_reranker_backend() -> None:
    """**The test this feature exists to not break** — split into the two properties it
    used to conflate under one name.

    A subprocess because by the time this file runs, pytest has already imported half the
    package and every other test in it.

    **The reranker half is unconditional.** A fresh interpreter that imports memvara,
    writes and reads must not have pulled in `memvara.rerank.cross` or
    `memvara.rerank.lexical`, whatever else is installed. Verified by breaking it: making
    `CrossEncoderReranker` an eager import in `memvara/rerank/__init__.py` turns this red.

    **The torch half is conditional, and pretending otherwise hid a real interaction.**
    It holds only while nothing in the environment provides sentence-transformers — and
    `memvara[rerank]` provides it, because a cross-encoder is one. `default_embedder()`
    keys on whether the package is *importable* rather than on which extra was asked for,
    so installing the reranker extra also makes the **default embedder** a torch model.
    That is an interaction between two extras, not a reranker defect, so it is asserted
    where it belongs: with the embedder pinned, the default read path still reaches no
    model at all. The unpinned case is asserted as what it actually is.
    """
    body = (
        "mem.add('I live in Lisbon')\n"
        "mem.search('where do they live?', include_episodes=True)\n"
        "mem.close()\n"
        "watched = {'sentence_transformers', 'torch', 'transformers',\n"
        "           'memvara.rerank.cross', 'memvara.rerank.lexical',\n"
        "           'memvara.select.model'}\n"
        "print(sorted(watched & set(sys.modules)))\n"
    )
    head = ("import sys\n"
            "from memvara import Memvara\n"
            "from memvara.llm import NullLLM\n")

    def run(construct: str) -> str:
        done = subprocess.run([sys.executable, "-c", head + construct + body],
                              cwd=REPO_ROOT, check=False, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        return done.stdout.strip()

    # Pinned: the whole watched set must be absent. This is the claim the README makes —
    # "numpy and nothing else, offline, no API key" — and it is exactly true.
    assert run("from memvara.embed import HashingEmbedder\n"
               "mem = Memvara(llm=NullLLM(), user='alice',\n"
               "              embedder=HashingEmbedder(dim=512))\n") == "[]"

    # Unpinned: no reranker backend either way. Torch may or may not be there, depending
    # on whether something in the environment installed sentence-transformers — so that
    # is read off the environment rather than asserted as a constant.
    reranker_backends = {"memvara.rerank.cross", "memvara.rerank.lexical"}
    seen = set(ast.literal_eval(run("mem = Memvara(llm=NullLLM(), user='alice')\n")))
    assert seen & reranker_backends == set(), seen


def test_the_default_configuration_opens_no_socket(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """"Offline" is a claim about the network, so this asserts it about the network
    rather than about an import. Any attempt to construct a socket fails the test.

    Verified by breaking it: a `socket.socket()` anywhere under the write or read path
    turns this red with the message below.
    """

    def refuse(*args, **kw):
        raise AssertionError("the default configuration opened a socket")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)

    mem = episodes_memvara()
    try:
        assert mem.reader.reranker is None
        assert mem.search("which dog suits a small flat?", include_episodes=True)
    finally:
        mem.close()


def test_no_reranker_leaves_every_explanation_unreranked() -> None:
    """The seam `Explanation.rerank_score` was reserved for, still saying what it said."""
    mem = episodes_memvara()
    try:
        results = mem.search("which dog suits a small flat?", include_episodes=True)
        assert results
        assert all(r.explain.rerank_score is None for r in results)
    finally:
        mem.close()


# -- wiring into retrieval -------------------------------------------------------------


def test_a_configured_reranker_reorders_the_search_and_explains_itself() -> None:
    """Reordering that cannot be accounted for is worse than none in a library whose
    pitch is that retrieval explains itself."""
    mem = episodes_memvara(read_reranker=CoverageReranker(), read_rerank_top_n=8)
    try:
        results = mem.search("which dog suits a small flat?", k=4,
                             include_episodes=True)
        assert results
        assert results[0].explain.rerank_score is not None
        assert "suits a small flat" in results[0].text
        assert "rerank=" in results[0].explain.summary()
    finally:
        mem.close()


def test_the_reranker_can_promote_a_candidate_from_below_the_callers_k() -> None:
    """The stage ranks to `rerank_top_n` and cuts to `k` *after*, so a candidate fusion
    left outside the caller's `k` can reach them.

    That ordering is the whole reason the stage is worth having. Reranking the same `k`
    items the caller was going to receive anyway can only change what they read first;
    it cannot change what is present, which is what a recall number measures.

    Same query, same `k`, same corpus in both halves — the reranker is the only
    difference, so nothing else can explain the change.
    """
    baseline = episodes_memvara()
    try:
        deep = [r.text for r in baseline.search(QUESTION, k=6, include_episodes=True)]
        shallow = [r.text for r in baseline.search(QUESTION, k=2, include_episodes=True)]
    finally:
        baseline.close()

    target = "The flat below mine has a dog that barks."
    assert target in deep and target not in shallow      # fusion left it outside k=2

    reranked = episodes_memvara(read_reranker=CoverageReranker(), read_rerank_top_n=6)
    try:
        top = [r.text for r in reranked.search(QUESTION, k=2, include_episodes=True)]
    finally:
        reranked.close()
    assert target in top


def test_the_reranker_runs_after_fusion_not_instead_of_it(
        store: SQLiteStore, embedder: HashingEmbedder) -> None:
    """It reorders what the existing scoring produced; it does not replace it. Every
    result still carries its fusion score, its recency and its normalized `score`."""
    mem = Memvara(store=store, embedder=embedder, llm=NullLLM(), user="alice",
                  read_reranker=NullReranker())
    try:
        mem.add([{"role": "user", "content": "I live in Lisbon"},
                 {"role": "user", "content": "I work at Acme"}])
        results = mem.search("where do they live?", include_episodes=True)
        assert results
        for r in results:
            assert r.explain.fusion_score > 0.0
            assert r.explain.rerank_score == 0.0
            assert r.score == pytest.approx(r.explain.final_score)
    finally:
        mem.close()


def test_the_reranked_ordering_is_reproducible_across_stores() -> None:
    """Two ingests of one corpus must rank identically, reranker included — otherwise a
    retrieval regression cannot be bisected."""
    runs = []
    for _ in range(2):
        mem = episodes_memvara(read_reranker=CoverageReranker(), read_rerank_top_n=8)
        try:
            runs.append([(r.text, r.explain.rerank_score)
                         for r in mem.search("dog flat", k=4, include_episodes=True)])
        finally:
            mem.close()
    assert runs[0] == runs[1]


def test_the_retriever_takes_a_reranker_directly_as_well(
        store: SQLiteStore, embedder: HashingEmbedder) -> None:
    """`Memvara(read_...)` routes to this signature; a caller wiring the engine up by
    hand should not have to go through the facade to reach it."""
    reader = HybridRetriever(store, embedder, PredicateRegistry(),
                             reranker=NullReranker(), rerank_top_n=5)
    assert reader.reranker is not None
    assert reader.search("anything", Scope("default", "alice")) == []


# -- the mem0 shim -------------------------------------------------------------------


def test_the_shim_still_refuses_rerank_when_nothing_is_configured() -> None:
    """The refusal stays, and the message has to stay *true*: it now says how to satisfy
    the request rather than that the library cannot."""
    api = Memory(claims_memvara())
    try:
        with pytest.raises(Mem0CompatError, match="read_reranker"):
            api.search("anything", rerank=True)
    finally:
        api.memvara.close()


def test_the_shim_honours_rerank_when_one_is_configured() -> None:
    api = Memory(claims_memvara(read_reranker=CoverageReranker(),
                                read_rerank_top_n=8))
    try:
        rows = api.search("which dog?", top_k=4, rerank=True, explain=True)
        assert rows["results"]
        assert "rerank=" in rows["results"][0]["explanation"]
    finally:
        api.memvara.close()


def test_the_shim_leaves_a_configured_reranker_alone_when_rerank_is_not_asked_for() -> None:
    """mem0's default is `rerank=False`, which every unmodified call site passes. It has
    to mean "no opinion" rather than "switch the instance's reranker off"."""
    api = Memory(claims_memvara(read_reranker=CoverageReranker(),
                                read_rerank_top_n=8))
    try:
        rows = api.search("which dog?", top_k=4, explain=True)
        assert "rerank=" in rows["results"][0]["explanation"]
    finally:
        api.memvara.close()
