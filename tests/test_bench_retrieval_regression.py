"""`bench/retrieval_regression.py`, the CI gate on LOCOMO retrieval figures.

The gate itself runs in CI against the real file. These tests pin the parts that decide
whether it fails: what it measures, how far a figure may move, and that a changed dataset
is reported as a changed dataset.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parent.parent / "bench"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import evalkit as ek  # noqa: E402
import locomo  # noqa: E402
import retrieval_regression as gate  # noqa: E402


def _score(category: str, *, in_ctx: bool | None, rank: int | None,
           evidence: bool = True) -> ek.RetrievalScore:
    s = ek.RetrievalScore(qid=f"q{id(object())}", category=category)
    s.answer_in_context = in_ctx
    if evidence:
        s.evidence_rank = rank
        s.evidence_recall_at = {k: float(rank is not None and rank <= k) for k in (1, 5)}
    return s


def test_the_figures_are_the_ones_the_published_table_prints():
    """The gate recomputes the table's numbers rather than parsing its text, so it is
    checked against the table: the same populations, the same rates."""
    scores = [_score("temporal", in_ctx=True, rank=1),
              _score("temporal", in_ctx=False, rank=3),
              _score("temporal", in_ctx=None, rank=None),
              _score("multi-hop", in_ctx=True, rank=None, evidence=False)]
    got = gate.figures(scores, (1, 5))
    assert got["temporal"] == {"n answered": 2, "n evidenced": 3, "in ctx": 50.0,
                               "R@1": 33.33, "R@5": 66.67, "MRR": 44.44}
    assert got["multi-hop"] == {"n answered": 1, "n evidenced": 0, "in ctx": 100.0}
    assert got["all"]["n answered"] == 3 and got["all"]["in ctx"] == 66.67
    table = ek.retrieval_tables(scores, ek.RetrievalPlan(ks=(1, 5)), ek.RetrievalBudget(),
                                ["temporal", "multi-hop"])
    assert "33.3" in table and "66.7" in table and "44.4" in table


def test_a_figure_fails_only_once_it_moves_past_its_bound():
    want = {"all": {"R@1": 30.0}, "temporal": {"R@1": 40.0}}
    inside = {"all": {"R@1": 30.1}, "temporal": {"R@1": 38.9}}
    assert gate.compare(want, inside) == []
    outside = {"all": {"R@1": 29.8}, "temporal": {"R@1": 41.2}}
    assert gate.compare(want, outside) == [
        "all R@1: expected 30.00, measured 29.80 (-0.20, bound 0.1)",
        "temporal R@1: expected 40.00, measured 41.20 (+1.20, bound 1.1)",
    ]


def test_a_changed_question_count_or_a_missing_figure_fails_whatever_else_matches():
    want = {"all": {"n answered": 10, "R@1": 30.0}}
    assert gate.compare(want, {"all": {"n answered": 11, "R@1": 30.0}}) == [
        "all n answered: expected 10, measured 11"]
    assert gate.compare(want, {"all": {"n answered": 10}}) == [
        "all R@1: expected 30.0, measured nothing"]


def test_a_dataset_the_figures_were_not_measured_on_is_reported_as_one(
        tmp_path, monkeypatch, capsys):
    """Checked before anything runs: a file changed upstream moves every figure, and the
    report should say which of the two things happened."""
    (tmp_path / "locomo10.json").write_text("[]")
    expected = tmp_path / "expected.json"
    expected.write_text(json.dumps({"dataset sha256": "0" * 64, "figures": {}}))
    monkeypatch.setattr(gate, "EXPECTED", expected)
    monkeypatch.setattr(gate, "measure", lambda samples: pytest.fail("measured anyway"))
    assert gate.main(["--cache", str(tmp_path)]) == 1
    assert "The dataset changed upstream" in capsys.readouterr().out


def test_the_gate_passes_on_the_figures_it_wrote_and_fails_once_they_move(
        tmp_path, monkeypatch, capsys):
    """End to end on the five built-in questions `bench/locomo.py --dry-run` uses, which
    need no download: `--update` writes what a run measures, the next run matches it,
    and a figure edited in the file fails the run."""
    (tmp_path / "locomo10.json").write_text("[]")
    monkeypatch.setattr(locomo, "load", lambda path: locomo.fixture())
    expected = tmp_path / "expected" / "locomo_retrieval.json"
    monkeypatch.setattr(gate, "EXPECTED", expected)
    assert gate.main(["--cache", str(tmp_path), "--update"]) == 0
    assert gate.main(["--cache", str(tmp_path)]) == 0
    assert "matches the committed figures" in capsys.readouterr().out
    written = json.loads(expected.read_text())
    written["figures"]["all"]["in ctx"] += 5.0
    expected.write_text(json.dumps(written))
    assert gate.main(["--cache", str(tmp_path)]) == 1
    assert "all in ctx: expected" in capsys.readouterr().out
