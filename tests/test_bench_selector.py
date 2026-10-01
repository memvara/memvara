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
