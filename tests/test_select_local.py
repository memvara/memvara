"""`LocalSelector` (memvara/select/local.py): the keep rule, the protocol, loading a
model, and how the ranked stage counts a local selection."""

from __future__ import annotations

import hashlib
import json
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from memvara.llm.base import Usage
from memvara.select import Candidate, Selected, Selector
from memvara.select.local import (
    CALIBRATION_FILE, SELECTOR_MODEL, SELECTOR_REVISION, STOCK_MODEL, STOCK_REVISION,
    Calibration, LocalSelector,
    keep_positions, load_encoder,
)

WHEN = datetime(2026, 1, 1, tzinfo=timezone.utc)
CAL = Calibration(scale=1.0, shift=0.0, threshold=0.5, max_keep=3)


class FakeEncoder:
    """Scores each (question, text) pair from a text -> score map. A text it does not know
    scores -5.0, which no calibration in this file keeps. `result`, when given, is
    returned as it is, for the tests of a misbehaving model. Records every pair."""

    def __init__(self, scores: dict[str, float] | None = None, *, result=None) -> None:
        self._scores = scores or {}
        self._result = result
        self.calls = 0
        self.pairs: list[tuple[str, str]] = []

    def predict(self, pairs, batch_size=32, show_progress_bar=None):
        self.calls += 1
        self.pairs += list(pairs)
        if self._result is not None:
            return self._result
        return [self._scores.get(text, -5.0) for _question, text in pairs]


def _candidates(*texts: str) -> list[Candidate]:
    return [Candidate(id=f"ep{i}", when=WHEN, text=text) for i, text in enumerate(texts)]


def _selector(scores: dict[str, float] | None = None, *, calibration: Calibration = CAL,
              result=None) -> LocalSelector:
    return LocalSelector(encoder=FakeEncoder(scores, result=result), calibration=calibration)


def _fake_sentence_transformers(monkeypatch) -> list:
    """A stand-in `sentence_transformers` whose `CrossEncoder` records how it was built, so
    loading is tested with neither the extra nor the network."""
    built: list = []

    class CrossEncoder(FakeEncoder):
        def __init__(self, model, **kwargs):
            super().__init__()
            built.append((model, kwargs))

    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        SimpleNamespace(CrossEncoder=CrossEncoder))
    return built


def _model_dir(path: Path, *, weights: bytes = b"weights", digest: str | None = "auto",
               fmt: int = 1, max_length: int | None = 256) -> Path:
    (path / "model.safetensors").write_bytes(weights)
    record = {"format": fmt, "scale": 2.0, "shift": -1.0, "threshold": 0.4, "max_keep": 5,
              "max_length": max_length}
    if digest == "auto":
        record["weights_sha256"] = hashlib.sha256(weights).hexdigest()
    elif digest is not None:
        record["weights_sha256"] = digest
    (path / CALIBRATION_FILE).write_text(json.dumps(record), encoding="utf-8")
    return path


# --- Calibration ------------------------------------------------------------------------


def test_probability_is_a_sigmoid_that_cannot_overflow() -> None:
    assert CAL.probability(0.0) == 0.5
    assert Calibration(scale=2.0, shift=-1.0, threshold=0.5, max_keep=1).probability(0.5) == 0.5
    assert CAL.probability(1000.0) == 1.0
    assert CAL.probability(-1000.0) == 0.0


@pytest.mark.parametrize("change", [
    {"scale": 0.0}, {"scale": -1.0}, {"scale": float("nan")}, {"scale": float("inf")},
    {"shift": float("nan")}, {"threshold": 0.0}, {"threshold": 1.0}, {"max_keep": 0},
    {"max_length": 0},
])
def test_a_calibration_that_cannot_work_is_refused(change) -> None:
    values = {"scale": 1.0, "shift": 0.0, "threshold": 0.5, "max_keep": 3} | change
    with pytest.raises(ValueError):
        Calibration(**values)


# --- the keep rule ----------------------------------------------------------------------


def test_keep_positions_keeps_every_score_at_or_above_the_threshold() -> None:
    # probability(0.0) is exactly the 0.5 threshold, so position 2 is kept.
    assert keep_positions([2.0, -1.0, 0.0, 3.0], CAL) == [0, 2, 3]


def test_keep_positions_caps_at_max_keep_by_score_and_breaks_ties_by_position() -> None:
    assert keep_positions([1.0, 5.0, 1.0, 4.0, 1.0], CAL) == [0, 1, 3]


def test_keep_positions_of_nothing_or_of_nothing_passing_is_empty() -> None:
    assert keep_positions([], CAL) == []
    assert keep_positions([-3.0, -9.0], CAL) == []


# --- the selector -----------------------------------------------------------------------


def test_it_is_a_selector_that_never_refuses_admission() -> None:
    sel = _selector()
    assert isinstance(sel, Selector)
    assert sel.kind == "local"
    assert sel.top_n == 40
    with sel.admit() as admitted:
        assert admitted is None


def test_it_keeps_passing_candidates_in_the_order_it_was_handed_them() -> None:
    sel = _selector({"booked the flight": 1.0, "flight leaves at 9": 3.0})
    kept = sel.select("when is my flight?",
                      _candidates("booked the flight", "weather is nice", "flight leaves at 9"))
    assert kept == [Selected(id="ep0", span=None), Selected(id="ep2", span=None)]


def test_no_candidates_means_no_encoder_call() -> None:
    encoder = FakeEncoder({"a": 9.0})
    sel = LocalSelector(encoder=encoder, calibration=CAL)
    assert sel.select("q", []) == []
    assert sel.scores("q", []) == []
    assert encoder.calls == 0


def test_when_every_candidate_passes_it_keeps_the_max_keep_highest() -> None:
    texts = ["t10", "t14", "t11", "t13", "t12"]
    sel = _selector({t: float(t[1:]) for t in texts})
    kept = sel.select("q", _candidates(*texts))
    assert [k.id for k in kept] == ["ep1", "ep3", "ep4"]      # 14, 13 and 12, in position order


def test_when_nothing_passes_it_keeps_nothing() -> None:
    assert _selector({}).select("q", _candidates("a", "b")) == []


def test_two_turns_with_the_same_text_are_judged_alike() -> None:
    kept = _selector({"same words": 2.0}).select("q", _candidates("same words", "same words"))
    assert [k.id for k in kept] == ["ep0", "ep1"]


def test_a_long_question_reaches_the_encoder_whole() -> None:
    encoder = FakeEncoder({"a": 2.0})
    question = "why does the build fail? " * 160                # 4,000 characters
    LocalSelector(encoder=encoder, calibration=CAL).select(question, _candidates("a"))
    assert encoder.pairs == [(question, "a")]


@pytest.mark.parametrize("result, message", [
    ([1.0], "returned 1 scores for 2 candidates"),
    ([1.0, float("nan")], "not a finite number"),
    ([float("inf"), 1.0], "not a finite number"),
])
def test_a_misbehaving_encoder_is_refused(result, message) -> None:
    with pytest.raises(ValueError, match=message):
        _selector(result=result).select("q", _candidates("a", "b"))


def test_the_date_and_the_usage_accumulator_change_nothing() -> None:
    sel = _selector({"a": 2.0})
    usage = Usage()
    assert (sel.select("q", _candidates("a", "b"), asked_on=WHEN, usage=usage)
            == sel.select("q", _candidates("a", "b")))
    assert usage.reported == 0


def test_scoring_holds_the_selector_lock_so_concurrent_reads_take_turns() -> None:
    held: list[bool] = []

    class LockWatcher(FakeEncoder):
        def predict(self, pairs, batch_size=32, show_progress_bar=None):
            held.append(sel._lock.locked())
            return super().predict(pairs, batch_size, show_progress_bar)

    sel = LocalSelector(encoder=LockWatcher({"a": 2.0}), calibration=CAL)
    threads = [threading.Thread(target=sel.select, args=("q", _candidates("a")))
               for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert held == [True] * 4


def test_top_n_must_be_positive() -> None:
    with pytest.raises(ValueError, match="top_n"):
        LocalSelector(encoder=FakeEncoder(), calibration=CAL, top_n=0)


def test_repr_and_encoder() -> None:
    encoder = FakeEncoder()
    sel = LocalSelector(encoder=encoder, calibration=CAL)
    assert repr(sel) == f"<LocalSelector {SELECTOR_MODEL} top_n=40>"
    assert sel.encoder is encoder


# --- where the calibration comes from, and loading --------------------------------------


def test_a_hub_model_without_a_shipped_calibration_is_refused() -> None:
    with pytest.raises(ValueError, match="no calibration"):
        LocalSelector("someone/else", encoder=FakeEncoder())
    with pytest.raises(ValueError, match="no calibration"):    # the stock model ships none
        LocalSelector(STOCK_MODEL, encoder=FakeEncoder())
    with pytest.raises(ValueError, match="no calibration"):    # nor does another commit
        LocalSelector(SELECTOR_MODEL, revision="0" * 40, encoder=FakeEncoder())


def test_the_default_is_the_published_model_at_its_commit_with_its_calibration(
        monkeypatch) -> None:
    built = _fake_sentence_transformers(monkeypatch)
    sel = LocalSelector()
    assert (sel.model, sel.revision) == (SELECTOR_MODEL, SELECTOR_REVISION)
    # The memvara_selector.json published beside the weights at that commit.
    assert sel.calibration == Calibration(scale=0.5170408164055298, shift=-2.690818832603243,
                                          threshold=0.01, max_keep=6, max_length=256)
    assert built == [(SELECTOR_MODEL, {"revision": SELECTOR_REVISION, "max_length": 256})]


def test_the_stock_model_is_pinned_to_its_commit(monkeypatch) -> None:
    built = _fake_sentence_transformers(monkeypatch)
    sel = LocalSelector(STOCK_MODEL, calibration=CAL)
    assert sel.revision == STOCK_REVISION
    assert built == [(STOCK_MODEL, {"revision": STOCK_REVISION, "max_length": None})]


def test_another_hub_model_loads_at_the_revision_given(monkeypatch) -> None:
    built = _fake_sentence_transformers(monkeypatch)
    LocalSelector("org/selector", revision="abc123", calibration=CAL)
    assert built == [("org/selector", {"revision": "abc123", "max_length": None})]


def test_a_model_directory_brings_its_own_calibration(tmp_path, monkeypatch) -> None:
    built = _fake_sentence_transformers(monkeypatch)
    sel = LocalSelector(str(_model_dir(tmp_path)))
    assert sel.calibration == Calibration(scale=2.0, shift=-1.0, threshold=0.4, max_keep=5,
                                          max_length=256)
    assert sel.revision is None
    assert built == [(str(tmp_path), {"revision": None, "max_length": 256})]


def test_a_directory_with_no_recorded_digest_or_length_loads_as_it_is(tmp_path, monkeypatch) -> None:
    built = _fake_sentence_transformers(monkeypatch)
    sel = LocalSelector(str(_model_dir(tmp_path, digest=None, max_length=None)))
    assert sel.calibration.max_length is None
    assert len(built) == 1


def test_weights_that_do_not_match_the_recorded_digest_are_refused(tmp_path, monkeypatch) -> None:
    built = _fake_sentence_transformers(monkeypatch)
    with pytest.raises(ValueError, match="not the ones the calibration was measured on"):
        LocalSelector(str(_model_dir(tmp_path, digest="0" * 64)))
    assert built == []


def test_a_calibration_file_of_another_format_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="format 2"):
        LocalSelector(str(_model_dir(tmp_path, fmt=2)), encoder=FakeEncoder())


def test_an_explicit_calibration_wins_over_the_directory(tmp_path) -> None:
    sel = LocalSelector(str(tmp_path), calibration=CAL, encoder=FakeEncoder())
    assert sel.calibration is CAL


def test_a_missing_extra_is_named(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError, match=r"memvara\[rerank\]"):
        load_encoder(STOCK_MODEL)


# --- the ranked stage --------------------------------------------------------------------

from memvara import Memvara  # noqa: E402
from memvara.embed import HashingEmbedder  # noqa: E402
from memvara.llm import NullLLM  # noqa: E402
from memvara.retrieve import EpisodeResult, HybridRetriever  # noqa: E402
from memvara.schema import PredicateRegistry  # noqa: E402
from memvara.select import Selection  # noqa: E402
from memvara.store import SQLiteStore  # noqa: E402
from memvara.telemetry import (  # noqa: E402
    RETRIEVAL_LOCAL_FALLBACK, RETRIEVAL_LOCAL_QUERY, RETRIEVAL_LOCAL_SELECT_MS,
    RETRIEVAL_MODEL_FALLBACK, RETRIEVAL_MODEL_QUERY, RETRIEVAL_SELECT_MS, MemoryRecorder,
)
from memvara.types import Episode, Scope  # noqa: E402

SCOPE = Scope("acme", "alice")


def _engine(selector, texts):
    store = SQLiteStore(":memory:")
    embedder = HashingEmbedder(dim=64)
    for text in texts:
        episode = Episode(content=text, scope=SCOPE)
        store.add_episode(episode)
        store.set_episode_embedding(episode.id, embedder.encode([text])[0])
    telemetry = MemoryRecorder()
    engine = HybridRetriever(store, embedder, PredicateRegistry(), selector=selector,
                             rerank_top_n=20, max_episodes=3, telemetry=telemetry)
    return engine, telemetry


@pytest.mark.covers("inv:TB9")
def test_a_local_selection_is_counted_on_the_local_series_only() -> None:
    engine, telemetry = _engine(_selector({"booked the kayak trip": 2.0}),
                                ["booked the kayak trip", "kayak shop opens late"])
    result = engine.search("kayak trip", SCOPE, k=5, include_episodes=True, ranked=True)
    assert result.selection == Selection(outcome="applied", candidates=2, kept=1)
    assert telemetry.total(RETRIEVAL_LOCAL_QUERY) == 1
    assert len(telemetry.values(RETRIEVAL_LOCAL_SELECT_MS)) == 1
    assert telemetry.total(RETRIEVAL_MODEL_QUERY) == 0
    assert telemetry.values(RETRIEVAL_SELECT_MS) == []


def test_keeping_nothing_is_applied_and_shows_every_turn_unkept() -> None:
    engine, telemetry = _engine(_selector({}), ["alpha kayak", "beta kayak"])
    result = engine.search("kayak", SCOPE, k=5, include_episodes=True, ranked=True)
    assert result.selection == Selection(outcome="applied", candidates=2, kept=0)
    episodes = [x for x in result if isinstance(x, EpisodeResult)]
    assert {x.text for x in episodes} == {"alpha kayak", "beta kayak"}
    assert all(x.explain.selected is False for x in episodes)
    assert telemetry.total(RETRIEVAL_LOCAL_QUERY) == 1


@pytest.mark.parametrize("scores", [[1.0], [1.0, float("nan")]])
def test_a_misbehaving_encoder_serves_the_plain_read_as_a_local_fallback(scores) -> None:
    engine, telemetry = _engine(_selector(result=scores), ["alpha kayak", "beta kayak"])
    result = engine.search("kayak", SCOPE, k=5, include_episodes=True, ranked=True)
    assert result.selection == Selection(outcome="fallback", reason="malformed", candidates=2)
    assert telemetry.total(RETRIEVAL_LOCAL_FALLBACK, reason="malformed") == 1
    assert telemetry.total(RETRIEVAL_MODEL_FALLBACK) == 0
    assert len(telemetry.values(RETRIEVAL_LOCAL_SELECT_MS)) == 1


def test_an_encoder_that_raises_is_counted_as_a_local_error() -> None:
    class Broken(FakeEncoder):
        def predict(self, pairs, batch_size=32, show_progress_bar=None):
            raise RuntimeError("out of memory")

    engine, telemetry = _engine(LocalSelector(encoder=Broken(), calibration=CAL),
                                ["alpha kayak"])
    result = engine.search("kayak", SCOPE, k=5, include_episodes=True, ranked=True)
    assert (result.selection.outcome, result.selection.reason) == ("fallback", "error")
    assert telemetry.total(RETRIEVAL_LOCAL_FALLBACK, reason="error") == 1


def test_a_local_failure_with_a_status_keeps_it_on_the_local_series() -> None:
    # A local model fetched over HTTP on first use can fail with a status code; the
    # failure is still a local one, never a model-provider fallback.
    class Unreachable(FakeEncoder):
        def predict(self, pairs, batch_size=32, show_progress_bar=None):
            exc = RuntimeError("hub unavailable")
            exc.status_code = 503                        # type: ignore[attr-defined]
            raise exc

    engine, telemetry = _engine(LocalSelector(encoder=Unreachable(), calibration=CAL),
                                ["alpha kayak"])
    result = engine.search("kayak", SCOPE, k=5, include_episodes=True, ranked=True)
    assert (result.selection.outcome, result.selection.reason) == ("fallback", "provider")
    assert telemetry.total(RETRIEVAL_LOCAL_FALLBACK, reason="provider", status="503") == 1
    assert telemetry.total(RETRIEVAL_MODEL_FALLBACK) == 0


def test_recall_with_nothing_kept_is_still_a_ranked_block() -> None:
    mem = Memvara(llm=NullLLM(), user="alice", embedder=HashingEmbedder(dim=8),
                  read_selector=_selector({}))
    mem.add("Loved the trip to Lisbon last spring", user="alice")
    out = mem.recall("the trip", user="alice", ranked=True, include_episodes=True,
                     with_ids=True)
    assert (out.selection.outcome, out.selection.kept) == ("applied", 0)
    assert "model ranking not applied" not in out.text


def test_recall_says_when_a_local_ranking_failed() -> None:
    mem = Memvara(llm=NullLLM(), user="alice", embedder=HashingEmbedder(dim=8),
                  read_selector=_selector(result=[float("inf")]))
    mem.add("Loved the trip to Lisbon last spring", user="alice")
    out = mem.recall("the trip", user="alice", ranked=True, include_episodes=True,
                     with_ids=True)
    assert out.selection.outcome == "fallback"
    assert Memvara.RECALL_UNRANKED.format(outcome="fallback") in out.text


def test_select_ordered_returns_the_kept_turns_and_every_candidate_by_score() -> None:
    sel = _selector({"b": 3.0, "c": 1.0, "a": -1.0})
    kept, order = sel.select_ordered("q", _candidates("a", "b", "c", "d"))
    assert [s.id for s in kept] == ["ep1", "ep2"]            # probability >= 0.5
    assert order == ["ep1", "ep2", "ep0", "ep3"]             # "d" ties nothing; -5.0 last
    assert sel.select("q", _candidates("a", "b", "c", "d")) == kept
    assert sel.select_ordered("q", []) == ([], [])


def test_a_tie_keeps_the_order_the_candidates_were_handed_in() -> None:
    _kept, order = _selector({}).select_ordered("q", _candidates("x", "y", "z"))
    assert order == ["ep0", "ep1", "ep2"]


def test_a_local_selection_shows_its_unkept_candidates_in_its_own_order() -> None:
    # The reranker here is absent, so the turn order is the episode leg's own; the local
    # model keeps "kayak one" and ranks "kayak three" above "kayak two".
    texts = ["kayak one", "kayak two", "kayak three"]
    engine, _ = _engine(_selector({"kayak one": 2.0, "kayak three": -0.5, "kayak two": -1.0}),
                        texts)
    engine.max_episodes = 3
    result = engine.search("kayak", SCOPE, k=5, include_episodes=True, ranked=True)
    episodes = [x.text for x in result if isinstance(x, EpisodeResult)]
    assert episodes == ["kayak one", "kayak three", "kayak two"]


def test_a_selector_without_select_ordered_keeps_the_reranked_tail() -> None:
    class KeepFirst:
        top_n = 40

        def admit(self):
            from contextlib import nullcontext
            return nullcontext()

        def select(self, question, candidates, *, asked_on=None, usage=None):
            return [Selected(id=candidates[0].id, span=None)]

    texts = ["kayak one", "kayak two", "kayak three"]
    engine, _ = _engine(KeepFirst(), texts)
    plain = engine.search("kayak", SCOPE, k=5, include_episodes=True)
    ranked = engine.search("kayak", SCOPE, k=5, include_episodes=True, ranked=True)
    order = [x.text for x in plain if isinstance(x, EpisodeResult)]
    shown = [x.text for x in ranked if isinstance(x, EpisodeResult)]
    assert shown[1:] == [t for t in order if t != shown[0]]


def test_candidates_a_selector_leaves_out_of_its_order_follow_the_ones_it_names() -> None:
    sel = _selector({})
    sel.select_ordered = lambda q, c, **kw: ([], [c[2].id])        # names one candidate
    engine, _ = _engine(sel, ["kayak one", "kayak two", "kayak three"])
    plain = [x.text for x in engine.search("kayak", SCOPE, k=5, include_episodes=True)
             if isinstance(x, EpisodeResult)]
    result = engine.search("kayak", SCOPE, k=5, include_episodes=True, ranked=True)
    shown = [x.text for x in result if isinstance(x, EpisodeResult)]
    assert len(shown) == 3 and set(shown) == set(plain)


@pytest.mark.parametrize("fields, message", [
    ({"scale": 1.0, "shift": 0.0, "max_keep": 6}, "has no 'threshold'"),
    ({"scale": None, "shift": 0.0, "threshold": 0.5, "max_keep": 6}, "not a number"),
])
def test_an_incomplete_calibration_file_names_what_is_wrong(tmp_path, fields, message) -> None:
    (tmp_path / CALIBRATION_FILE).write_text(json.dumps({"format": 1, **fields}),
                                             encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        Calibration.read(tmp_path)


def test_the_package_names_the_local_selector_lazily() -> None:
    import memvara.select as select_package
    from memvara.select import local

    assert select_package.LocalSelector is local.LocalSelector
    assert select_package.Calibration is local.Calibration
