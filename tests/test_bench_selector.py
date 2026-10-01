"""The local selector's bench scripts: pools, replay, scoring, calibration, synthesis,
training data, the MemoryBench arm and the screen. Only pure functions are tested here;
nothing loads a model or reaches the network."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# `bench/` is a directory of scripts that import each other by bare name, as in
# tests/test_bench_eval.py.
BENCH = Path(__file__).resolve().parent.parent / "bench"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import locomo  # noqa: E402
import longmemeval as lme  # noqa: E402
import selector_pools as sp  # noqa: E402

from memvara.embed import HashingEmbedder  # noqa: E402


def _embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=64)


def test_longmemeval_pools_mark_the_gold_turns_by_their_text() -> None:
    items = json.loads(json.dumps(lme.FIXTURE))
    pools = sp.from_longmemeval(items, embedder=_embedder())
    assert [p.qid for p in pools] == [str(raw["question_id"]) for raw in items]
    for raw, pool in zip(items, pools):
        marked = sp.answer_turns(raw)
        assert pool.turns
        assert all(t.gold == (t.text in marked) for t in pool.turns)
    assert sum(p.gold_count for p in pools) >= 1


def test_longmemeval_pool_order_does_not_depend_on_the_labels() -> None:
    marked = json.loads(json.dumps(lme.FIXTURE[:1]))
    moved = json.loads(json.dumps(lme.FIXTURE[:1]))
    for session in moved[0]["haystack_sessions"]:
        for turn in session:
            turn["has_answer"] = True
    a = sp.from_longmemeval(marked, embedder=_embedder())[0]
    b = sp.from_longmemeval(moved, embedder=_embedder())[0]
    assert [t.text for t in a.turns] == [t.text for t in b.turns]
    assert all(t.gold for t in b.turns)


def test_locomo_pools_skip_adversarial_questions_and_label_by_evidence() -> None:
    samples = locomo.fixture()
    by_label = {t.label: t.text for s in samples[0].sessions for t in s.turns}
    pools = sp.from_locomo(samples, embedder=_embedder())
    assert [p.qtype for p in pools] == ["single-hop", "multi-hop", "temporal", "open-domain"]
    answerable = [qa for qa in samples[0].qa if not qa.is_adversarial]
    for qa, pool in zip(answerable, pools):
        assert {t.text for t in pool.turns if t.gold} <= {by_label[e] for e in qa.evidence_ids}
    assert sum(p.gold_count for p in pools) >= 1


def test_memorybench_pools_read_the_turns_and_the_model_selectors_flags(tmp_path) -> None:
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "q1.json").write_text(json.dumps({"questionId": "q1", "results": [
        {"kind": "turn", "role": "user", "content": "I adopted a cat", "ts": "2023-05-01",
         "score": 0.9, "selected": True},
        {"kind": "memory", "text": "user adopted cat"},
        {"kind": "turn", "role": "assistant", "content": "Cats are great", "score": 0.5,
         "selected": False},
        {"kind": "turn", "role": "user", "content": "Weather", "score": 0.1, "selected": None},
    ]}), encoding="utf-8")
    items = {"q1": {"question": "What pet did I adopt?", "question_type": "single-session-user",
                    "haystack_sessions": [[{"role": "user", "content": "I adopted a cat",
                                            "has_answer": True}]]}}
    [pool] = sp.from_memorybench(tmp_path, items)
    assert [t.id for t in pool.turns] == ["q1:0", "q1:2", "q1:3"]
    assert [t.gold for t in pool.turns] == [True, False, False]
    assert [t.paid_kept for t in pool.turns] == [True, False, None]


def test_pools_and_scores_round_trip(tmp_path) -> None:
    pools = sp.from_longmemeval(json.loads(json.dumps(lme.FIXTURE)), embedder=_embedder())
    assert sp.write_pools(tmp_path / "p.jsonl", pools) == len(pools)
    assert sp.read_pools(tmp_path / "p.jsonl") == pools
    sp.write_scores(tmp_path / "s.jsonl", {"q": [1.0, None]}, model="m")
    assert sp.read_scores(tmp_path / "s.jsonl") == {"q": [1.0, None]}
    assert sp.full_scores({"q": [1.0, 2.0]}) == {"q": [1.0, 2.0]}
    with pytest.raises(ValueError, match="some turns have no score"):
        sp.full_scores({"q": [1.0, None]})


def _pool(source: str, qid: str) -> sp.Pool:
    return sp.Pool(source=source, qid=qid, question="q", qtype="t", asked_on=None,
                   abstention=False, turns=[])


def test_splits_keep_given_test_ids_test_only_sources_and_whole_scenarios() -> None:
    pools = ([_pool("longmemeval", f"l{i}") for i in range(20)]
             + [_pool("locomo", "conv:1")]
             + [_pool("synth-coding", f"syn-coding-agent-{s:04d}-q{q}")
                for s in range(20) for q in range(3)])
    splits = sp.assign_splits(pools, test_ids={"l0", "l1"}, seed=0)
    assert splits["l0"] == splits["l1"] == "test" and splits["conv:1"] == "test"
    assert {splits[f"l{i}"] for i in range(2, 20)} <= {"train", "validation"}
    for s in range(20):
        assert len({splits[f"syn-coding-agent-{s:04d}-q{q}"] for q in range(3)}) == 1
    assert {"train", "validation", "test"} <= set(splits.values())
    assert splits == sp.assign_splits(pools, test_ids={"l0", "l1"}, seed=0)


import selector_metrics as sm  # noqa: E402

from memvara.select.local import Calibration  # noqa: E402

WORDS = lambda text: len(text.split())  # noqa: E731 - a token counter for small tests


def _turn(text: str, role: str = "user", gold: bool = False, paid=None) -> sp.PoolTurn:
    return sp.PoolTurn(id=text, role=role, text=text, ts="", fused=0.0, gold=gold,
                       paid_kept=paid)


def _toy(question: str = "where did I park?") -> sp.Pool:
    return sp.Pool(source="toy", qid="toy1", question=question, qtype="t", asked_on=None,
                   abstention=False, turns=[
                       _turn("a1 a1 a1 a1", role="assistant"),
                       _turn("u1 u1", gold=False),
                       _turn("u2 u2 u2", gold=True),
                       _turn("u3", gold=False)])


def test_replay_reranks_routes_and_renders_kept_turns_first() -> None:
    pool = _toy()
    rerank = [4.0, 3.0, 2.0, 1.0]
    plain = sm.replay(pool, rerank, None)
    assert plain.order == [0, 1, 2, 3]
    assert plain.scope == [1, 2, 3]                 # the question is the user's: user turns only
    assert plain.rendered == plain.order
    kept = sm.replay(pool, rerank, {2, 0})          # 0 is outside the scope and is ignored
    assert kept.rendered == [2, 0, 1, 3]


def test_replay_hands_over_every_role_when_the_routed_one_has_no_turn() -> None:
    pool = _toy("remind me what you said about dinner")  # routed to the assistant
    pool.turns[0].role = "user"                           # ... who said nothing here
    assert sm.replay(pool, [4.0, 3.0, 2.0, 1.0], None).scope == [0, 1, 2, 3]


def test_coverage_fills_the_budget_greedily_and_skips_what_does_not_fit() -> None:
    pool = _toy()
    assert sm.coverage(pool, [0, 1, 2, 3], WORDS, budget=6) == (0, 1)     # 4 + 2 fill it
    assert sm.coverage(pool, [0, 2, 1, 3], WORDS, budget=8) == (1, 1)     # 4 + 3, then 1 fits


def test_local_replay_keeps_by_the_shipped_rule() -> None:
    pool = _toy()
    cal = Calibration(scale=1.0, shift=0.0, threshold=0.5, max_keep=1)
    rep, kept = sm.local_replay(pool, [4.0, 3.0, 2.0, 1.0], [None, -2.0, 3.0, 1.0], cal)
    assert kept == {2}
    assert rep.rendered[0] == 2
    with pytest.raises(ValueError, match="do not cover every candidate"):
        sm.local_replay(pool, [4.0, 3.0, 2.0, 1.0], [None, None, 3.0, 1.0], cal)


def test_kept_recall_counts_gold_kept_among_gold_shown() -> None:
    assert sm.kept_recall(_toy(), [1, 2, 3], {2}) == (1, 1)
    assert sm.kept_recall(_toy(), [1, 2, 3], set()) == (0, 1)


def test_paired_bootstrap_is_deterministic_and_counts_wins() -> None:
    a = {"q1": (1, 1), "q2": (2, 2), "q3": (0, 1)}
    b = {"q1": (0, 1), "q2": (2, 2), "q3": (0, 1)}
    first = sm.paired_bootstrap(a, b, resamples=200, seed=0)
    assert first == sm.paired_bootstrap(a, b, resamples=200, seed=0)
    assert (first.better, first.worse) == (1, 0)
    assert first.diff == pytest.approx(0.25)
    assert first.low <= first.diff <= first.high


def test_evaluate_reports_plain_routed_paid_and_local_orderings() -> None:
    pool = _toy()
    for turn, paid in zip(pool.turns, [None, False, True, False]):
        turn.paid_kept = paid
    cal = Calibration(scale=1.0, shift=0.0, threshold=0.5, max_keep=1)
    results = sm.evaluate([pool], {"toy1": [4.0, 3.0, 2.0, 1.0]},
                          {"local": ({"toy1": [None, -2.0, 3.0, 1.0]}, cal)}, WORDS,
                          paid=True, budget=8)
    assert set(results) == {"plain", "routed", "paid", "local"}
    assert results["local"]["toy1"] == (1, 1, 1, 1, 1)
    assert "| local | 1 |" in sm.table(results, baseline="routed")
    assert "| t | 1 |" in sm.type_table([pool], results)


import math  # noqa: E402
import random  # noqa: E402

import selector_calibrate as sc  # noqa: E402
import selector_score as ss  # noqa: E402

from memvara.select.local import CALIBRATION_FILE  # noqa: E402


def test_score_pools_scores_only_the_candidates_when_given_a_scope() -> None:
    pool = _toy()
    seen: list = []

    def predict(pairs):
        seen.extend(pairs)
        return [float(len(text)) for _q, text in pairs]

    full = ss.score_pools([pool], predict)
    assert full["toy1"] == [11.0, 5.0, 8.0, 2.0]
    seen.clear()
    scoped = ss.score_pools([pool], predict, scope_of={"toy1": [4.0, 3.0, 2.0, 1.0]})
    assert scoped["toy1"] == [None, 5.0, 8.0, 2.0]
    assert [text for _q, text in seen] == ["u1 u1", "u2 u2 u2", "u3"]


def test_fit_platt_recovers_a_known_logistic() -> None:
    rng = random.Random(0)
    xs = [i / 100 for i in range(-400, 401)]
    ys = [rng.random() < 1 / (1 + math.exp(-(2 * x - 1))) for x in xs]
    scale, shift = sc.fit_platt(xs, ys)
    assert scale == pytest.approx(2.0, abs=0.35)
    assert shift == pytest.approx(-1.0, abs=0.35)


def test_fit_platt_refuses_data_it_cannot_calibrate() -> None:
    with pytest.raises(ValueError, match="both gold and non-gold"):
        sc.fit_platt([1.0, 2.0], [True, True])
    with pytest.raises(ValueError, match="slope"):
        sc.fit_platt([float(x) for x in range(-50, 50)], [x < 0 for x in range(-50, 50)])


def test_choose_keep_prefers_the_threshold_that_lets_the_gold_turn_in() -> None:
    pool = sp.Pool(source="toy", qid="t", question="q", qtype="t", asked_on=None,
                   abstention=False, turns=[_turn("n " * 10), _turn("m " * 10),
                                            _turn("g g", gold=True)])
    rerank = {"t": [3.0, 2.0, 1.0]}
    select = {"t": [0.0, -5.0, 5.0]}      # the first non-gold turn sits at probability 0.5
    cal, cov, mean_kept = sc.choose_keep([pool], rerank, select, 1.0, 0.0, WORDS, budget=11)
    assert cov == 1.0
    assert cal.threshold == 0.55          # the first grid value above 0.5 keeps only the gold
    assert mean_kept == 1.0


def test_write_calibration_is_read_back_by_the_selector(tmp_path) -> None:
    cal = Calibration(scale=1.5, shift=-0.5, threshold=0.35, max_keep=8, max_length=256)
    path = sc.write_calibration(tmp_path, cal, base_model="m")
    assert path == tmp_path / CALIBRATION_FILE
    assert Calibration.read(tmp_path) == (cal, None)


def _log_loss(scale: float, shift: float, xs, ys) -> float:
    total = 0.0
    for x, y in zip(xs, ys):
        z = scale * x + shift
        # log(1 + exp(-z)) for gold, log(1 + exp(z)) otherwise, written to stay finite.
        m = -z if y else z
        total += max(m, 0.0) + math.log1p(math.exp(-abs(m)))
    return total / len(xs)


def test_fit_platt_converges_on_wide_imbalanced_scores_like_a_real_reranker() -> None:
    # The shape that broke the first stock-model fit, as 25 quantiles of each class from
    # the stock model's scores on LongMemEval's 301 calibration questions (2026-10-01):
    # gold spread from -11 to +9, the rest packed near -11, 4% gold. Undamped Newton from
    # (1, 0) diverged to a slope of -26,900 on exactly these numbers.
    gold = [-11.0, -10.1, -9.3, -8.6, -7.6, -6.9, -5.7, -4.8, -4.0, -3.0, -2.2, -1.5, -0.8,
            0.0, 0.7, 1.2, 2.0, 2.9, 3.8, 4.3, 5.1, 5.8, 6.6, 7.4, 8.8]
    other = [-11.4, -11.4, -11.3, -11.3, -11.3, -11.3, -11.2, -11.2, -11.2, -11.1, -11.1,
             -11.0, -11.0, -10.9, -10.8, -10.7, -10.5, -10.3, -10.1, -9.7, -9.3, -8.7, -7.8,
             -6.1, 1.4]
    xs = other * 22 + gold
    ys = [False] * (len(other) * 22) + [True] * len(gold)
    scale, shift = sc.fit_platt(xs, ys)
    assert scale > 0
    # The fit must be at least as good as a brute-force search over a grid of slopes and
    # intercepts, which is the reference this test trusts.
    best = min(_log_loss(a / 20, b / 4, xs, ys) for a in range(1, 61) for b in range(-60, 21))
    assert _log_loss(scale, shift, xs, ys) <= best + 1e-6


def test_the_keep_rule_search_can_reach_low_thresholds_and_keep_every_candidate() -> None:
    # Step 2a's grid (spec section 12): a gold turn whose calibrated probability is only
    # 0.013 is kept at threshold 0.01, and max_keep can go up to all 40 candidates.
    assert sc.GRID_THRESHOLDS[:3] == (0.01, 0.02, 0.05)
    assert sc.GRID_MAX_KEEP[-2:] == (24, 40)
    pool = sp.Pool(source="toy", qid="t", question="q", qtype="t", asked_on=None,
                   abstention=False, turns=[_turn("n " * 10), _turn("m " * 10),
                                            _turn("g g", gold=True)])
    cal, cov, _ = sc.choose_keep([pool], {"t": [3.0, 2.0, 1.0]}, {"t": [-9.0, -9.0, -4.3]},
                                 1.0, 0.0, WORDS, budget=11)
    assert (cal.threshold, cov) == (0.01, 1.0)


import hashlib  # noqa: E402

import selector_train as st  # noqa: E402


def _labelled_pool(qid: str, source: str = "toy") -> sp.Pool:
    turns = [_turn(f"{qid} gold", gold=True, paid=True)] + [
        _turn(f"{qid} other {i}", paid=False if i < 3 else None) for i in range(8)]
    return sp.Pool(source=source, qid=qid, question="q", qtype="t", asked_on=None,
                   abstention=False, turns=turns)


def test_training_pairs_use_the_train_split_candidates_and_cap_negatives() -> None:
    pools = [_labelled_pool("a"), _labelled_pool("b"), _labelled_pool("c", source="held")]
    splits = {"a": "train", "b": "validation", "c": "train"}
    rerank = {p.qid: [float(-i) for i in range(len(p.turns))] for p in pools}
    pairs = st.training_pairs(pools, splits, rerank, negatives=3, seed=0, hold_out="held")
    assert {text.split()[0] for _q, text, _y in pairs} == {"a"}
    assert sum(y for _q, _t, y in pairs) == 1.0 and len(pairs) == 4
    assert pairs == st.training_pairs(pools, splits, rerank, negatives=3, seed=0, hold_out="held")


def test_training_pairs_can_use_the_model_selectors_labels_for_the_comparison() -> None:
    pools = [_labelled_pool("a")]
    rerank = {"a": [float(-i) for i in range(9)]}
    pairs = st.training_pairs(pools, {"a": "train"}, rerank, label="paid", negatives=10)
    assert len(pairs) == 4                       # one kept, three omitted; unshown turns unused
    assert sum(y for _q, _t, y in pairs) == 1.0


def test_write_selector_json_records_the_weights_digest(tmp_path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"trained")
    cal = Calibration(scale=1.2, shift=-2.0, threshold=0.4, max_keep=6, max_length=256)
    st.write_selector_json(tmp_path, cal, base_model="m", seed=0)
    read, digest = Calibration.read(tmp_path)
    assert read == cal
    assert digest == hashlib.sha256(b"trained").hexdigest()


import selector_public as pub  # noqa: E402


def _hotpot_row() -> dict:
    return {"id": "h1", "question": "who built it?",
            "context": {"title": ["A", "B"],
                        "sentences": [[" A was built.", " By Ann."], ["B is far.", "  "]]},
            "supporting_facts": {"title": ["A", "A", "B"], "sent_id": [1, 7, 0]}}


def test_supporting_facts_mark_sentences_and_ignore_facts_past_the_paragraph() -> None:
    ex = pub.from_supporting_facts("hotpot", _hotpot_row())
    assert ex is not None
    assert ex.candidates == ["A was built.", "By Ann.", "B is far."]   # blank sentence dropped
    assert ex.gold == {1, 2}                                           # (A, 7) does not exist
    row = _hotpot_row()
    row["supporting_facts"] = {"title": ["C"], "sent_id": [0]}
    assert pub.from_supporting_facts("hotpot", row) is None


def test_musique_uses_paragraphs_and_skips_questions_with_no_support() -> None:
    row = {"id": "m1", "question": "q", "paragraphs": [
        {"paragraph_text": "p0", "is_supporting": False},
        {"paragraph_text": "p1", "is_supporting": True}]}
    ex = pub.from_musique(row)
    assert ex is not None and ex.candidates == ["p0", "p1"] and ex.gold == {1}
    row["paragraphs"][1]["is_supporting"] = False
    assert pub.from_musique(row) is None


def test_quac_marks_the_sentence_where_the_answer_starts_and_skips_unanswerable() -> None:
    context = "Ann was born in Rome. She moved to Oslo in 1990! Why? Nobody knows. CANNOTANSWER"
    start = context.index("Oslo")
    article = {"paragraphs": [{"context": context, "qas": [
        {"id": "q1", "question": "where did she move?",
         "orig_answer": {"text": "Oslo", "answer_start": start}},
        {"id": "q2", "question": "her dog?",
         "orig_answer": {"text": "CANNOTANSWER", "answer_start": len(context) - 12}}]}]}
    [ex] = list(pub.from_quac(article))
    assert ex.candidates == ["Ann was born in Rome.", "She moved to Oslo in 1990!", "Why?",
                             "Nobody knows."]
    assert ex.gold == {1} and ex.qid == "q1"


def test_build_draws_per_dataset_and_three_negatives_per_gold(tmp_path) -> None:
    examples = [pub.Example("hotpot", f"h{i}", f"q{i}", [f"c{j}" for j in range(10)], {0, 1})
                for i in range(20)]
    rows = pub.build({"hotpot": examples}, {"hotpot": 5}, negatives=3, seed=0)
    assert len({r["qid"] for r in rows}) == 5
    assert len(rows) == 5 * 8 and sum(r["label"] for r in rows) == 10
    assert rows == pub.build({"hotpot": examples}, {"hotpot": 5}, negatives=3, seed=0)
    # Adding another dataset does not change this one's draws.
    other = [pub.Example("quac", "x", "q", ["a", "b"], {0})]
    both = pub.build({"hotpot": examples, "quac": other}, {"hotpot": 5}, seed=0)
    assert [r for r in both if r["source"] == "hotpot"] == rows
    path = tmp_path / "pairs.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert pub.read_pairs(path)[0] == (rows[0]["query"], rows[0]["text"], rows[0]["label"])


def test_take_accepts_known_datasets_only() -> None:
    assert pub.parse_take(["quac=5"])["quac"] == 5
    assert pub.parse_take([]) == pub.DEFAULT_TAKE
    with pytest.raises(ValueError):
        pub.parse_take(["nq=5"])


def test_renderings_reorder_only_a_local_selection() -> None:
    # Candidates are the user turns in reranked order; the assistant turn is not routed.
    turns = [_turn("u0"), _turn("u1", gold=True), _turn("u2"), _turn("a0", role="assistant")]
    pool = sp.Pool(source="toy", qid="r", question="where did I park?", qtype="t",
                   asked_on=None, abstention=False, turns=turns)
    rerank = [4.0, 3.0, 2.0, 5.0]
    select = [0.0, 1.0, 9.0, None]
    cal = Calibration(scale=1.0, shift=0.0, threshold=0.99, max_keep=1)   # keeps u2 only
    server, kept = sm.local_replay(pool, rerank, select, cal)
    assert kept == {2}
    assert server.rendered == [2, 3, 0, 1]           # kept, then everything in reranked order
    first, _ = sm.local_replay(pool, rerank, select, cal, render="routed-first")
    assert first.rendered == [2, 0, 1, 3]            # kept, unkept candidates, then the rest
    ordered, _ = sm.local_replay(pool, rerank, select, cal, render="selector-order")
    assert ordered.rendered == [2, 1, 0, 3]          # candidates by the selector's score
    with pytest.raises(ValueError):
        sm.local_replay(pool, rerank, select, cal, render="sideways")
