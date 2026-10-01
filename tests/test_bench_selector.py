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
