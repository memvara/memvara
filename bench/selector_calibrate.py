"""Fit a selector model's calibration and keep rule on labelled pools.

    PYTHONPATH=. python3 bench/selector_calibrate.py --pools P.jsonl [...] --splits S.json \
        --split validation [--split train] --rerank STOCK.jsonl [...] --select MODEL.jsonl [...] \
        [--max-length N] --out-dir DIR

The fit has two steps:

1. Logistic regression of the gold flag on the raw score of every candidate (Platt
   scaling). This gives `scale` and `shift`.
2. A grid search over `threshold` and `max_keep` for the highest coverage at 720 tokens,
   with ties going to fewer turns kept.

Fit on the validation split for a model that was trained. Fit on train and validation
together for the stock model, which was not trained on either. The script prints the
`Calibration(...)` line for memvara/select/local.py and writes the same numbers to
DIR/memvara_selector.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from selector_metrics import BUDGET, TokenCounter, coverage, local_replay, replay, tiktoken_counter
from selector_pools import Pool, full_scores, read_pools, read_scores

from memvara.select.local import CALIBRATION_FILE, CALIBRATION_FORMAT, Calibration

#: The keep-rule grid. Step 2a (the spec's section 12) widened it after G0, where the
#: stock model's calibrated probabilities fell below 0.05 after three or four turns:
#: 0.01 and 0.02 let a model keep turns it is unsure of, and a max_keep of 24 or 40 lets
#: it keep most or all of the 40 candidates.
GRID_THRESHOLDS = (0.01, 0.02) + tuple(round(0.05 * i, 2) for i in range(1, 20))
GRID_MAX_KEEP = (4, 6, 8, 10, 12, 16, 24, 40)


def fit_platt(scores: Sequence[float], labels: Sequence[bool], *, l2: float = 1e-3,
              iterations: int = 100) -> tuple[float, float]:
    x = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    if x.size == 0 or y.min() == y.max():
        raise ValueError("calibration needs both gold and non-gold candidates")
    def loss(a: float, b: float) -> float:
        # Mean log loss plus the L2 term, written with logaddexp so it stays finite.
        z = a * x + b
        return float(np.mean(np.logaddexp(0.0, -z) * y + np.logaddexp(0.0, z) * (1.0 - y))
                     + 0.5 * l2 * a * a / x.size)

    a, b = 1.0, 0.0
    current = loss(a, b)
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-np.clip(a * x + b, -50.0, 50.0)))
        w = p * (1.0 - p)
        grad = np.array([np.dot(p - y, x) + l2 * a, np.sum(p - y)])
        hess = np.array([[np.dot(w, x * x) + l2, np.dot(w, x)],
                         [np.dot(w, x), np.sum(w) + 1e-9]])
        step = np.linalg.solve(hess, grad)
        # A full Newton step overshoots when the scores span a wide range and few are gold,
        # which is exactly a reranker's shape: halve the step until the loss goes down.
        t = 1.0
        while t > 1e-8 and loss(a - t * step[0], b - t * step[1]) > current:
            t /= 2
        a, b = a - t * step[0], b - t * step[1]
        previous, current = current, loss(a, b)
        if abs(previous - current) < 1e-12:
            break
    if not a > 0:
        raise ValueError(f"the fitted slope is {a:.3f}: on this data a higher score does not "
                         "mean a more relevant turn, so the model cannot be calibrated")
    return float(a), float(b)


def scope_examples(pools: Sequence[Pool], rerank: Mapping[str, Sequence[float]],
                   select: Mapping[str, Sequence[float | None]]) -> tuple[list[float], list[bool]]:
    xs: list[float] = []
    ys: list[bool] = []
    for pool in pools:
        for i in replay(pool, rerank[pool.qid], None).scope:
            value = select[pool.qid][i]
            if value is None:
                raise ValueError(f"{pool.qid}: candidate {i} has no selector score")
            xs.append(float(value))
            ys.append(pool.turns[i].gold)
    return xs, ys


def choose_keep(pools: Sequence[Pool], rerank: Mapping[str, Sequence[float]],
                select: Mapping[str, Sequence[float | None]], scale: float, shift: float,
                count: TokenCounter, *, max_length: int | None = None,
                budget: int = BUDGET) -> tuple[Calibration, float, float]:
    answerable = [p for p in pools if p.gold_count]
    if not answerable:
        raise ValueError("no pool has a gold turn to cover")
    best: tuple[tuple[float, int], Calibration, float] | None = None
    for threshold in GRID_THRESHOLDS:
        for max_keep in GRID_MAX_KEEP:
            calibration = Calibration(scale, shift, threshold, max_keep, max_length)
            got = total = kept = 0
            for pool in answerable:
                rep, chosen = local_replay(pool, rerank[pool.qid], select[pool.qid], calibration)
                g, n = coverage(pool, rep.rendered, count, budget)
                got, total, kept = got + g, total + n, kept + len(chosen)
            key = (got / total, -kept)
            if best is None or key > best[0]:
                best = (key, calibration, kept / len(answerable))
    assert best is not None
    (cov, _), calibration, mean_kept = best
    return calibration, cov, mean_kept


def write_calibration(directory: Path, calibration: Calibration, **extra: Any) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    record = {"format": CALIBRATION_FORMAT, "scale": calibration.scale,
              "shift": calibration.shift, "threshold": calibration.threshold,
              "max_keep": calibration.max_keep, "max_length": calibration.max_length, **extra}
    path = directory / CALIBRATION_FILE
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit a selector's calibration and keep rule.")
    parser.add_argument("--pools", type=Path, nargs="+", required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--split", action="append", required=True)
    parser.add_argument("--rerank", type=Path, nargs="+", required=True)
    parser.add_argument("--select", type=Path, nargs="+", required=True)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    pools = [p for path in args.pools for p in read_pools(path) if splits.get(p.qid) in args.split]
    rerank_raw: dict[str, list[float | None]] = {}
    for path in args.rerank:
        rerank_raw.update(read_scores(path))
    rerank = full_scores(rerank_raw)
    select: dict[str, list[float | None]] = {}
    for path in args.select:
        select.update(read_scores(path))
    xs, ys = scope_examples(pools, rerank, select)
    scale, shift = fit_platt(xs, ys)
    calibration, cov, mean_kept = choose_keep(pools, rerank, select, scale, shift,
                                              tiktoken_counter(), max_length=args.max_length)
    write_calibration(args.out_dir, calibration, fitted_on=sorted(set(args.split)),
                      questions=len(pools), candidates=len(xs), gold=int(sum(ys)))
    print(f"fitted on {len(pools)} questions, {len(xs)} candidates, {int(sum(ys))} gold")
    print(f"coverage {cov:.3f} at {mean_kept:.1f} turns kept on average")
    print(repr(calibration))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
