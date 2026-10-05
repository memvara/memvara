"""Score pool turns with a cross-encoder once, and save the raw scores.

    PYTHONPATH=. python3 bench/selector_score.py --pools P.jsonl [...] --model ID_OR_DIR \
        [--revision SHA] [--max-length N] [--device cpu|mps|cuda] \
        [--scope-of STOCK.jsonl [...]] --out S.jsonl

Scoring is the slow step, so it runs once per model and the other scripts read the file.
`--scope-of` scores only the turns a selector would be handed (the routed first 40 under
the reranker scores in those files), which is all a selector model ever sees. Without it,
every turn is scored, which is what a reranker needs.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

from selector_metrics import TOP_N, replay
from selector_pools import Pool, full_scores, read_pools, read_scores, write_scores

Predict = Callable[[list[tuple[str, str]]], Sequence[float]]


def score_pools(pools: Sequence[Pool], predict: Predict, *,
                scope_of: Mapping[str, Sequence[float]] | None = None,
                top_n: int = TOP_N) -> dict[str, list[float | None]]:
    out: dict[str, list[float | None]] = {}
    for pool in pools:
        indices = (list(range(len(pool.turns))) if scope_of is None
                   else replay(pool, scope_of[pool.qid], None, top_n=top_n).scope)
        values = list(predict([(pool.question, pool.turns[i].text) for i in indices])) \
            if indices else []
        full: list[float | None] = [None] * len(pool.turns)
        for i, value in zip(indices, values):
            full[i] = float(value)
        out[pool.qid] = full
    return out


def encoder_predict(model: str, *, revision: str | None = None, max_length: int | None = None,
                    device: str | None = None, batch_size: int = 32) -> Predict:
    from sentence_transformers import CrossEncoder

    encoder = CrossEncoder(model, revision=revision, max_length=max_length, device=device)
    return lambda pairs: encoder.predict(pairs, batch_size=batch_size, show_progress_bar=False)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score pools once with one cross-encoder.")
    parser.add_argument("--pools", type=Path, nargs="+", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--device")
    parser.add_argument("--scope-of", type=Path, nargs="*")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    pools = [p for path in args.pools for p in read_pools(path)]
    scope_of = None
    if args.scope_of:
        merged: dict[str, list[float | None]] = {}
        for path in args.scope_of:
            merged.update(read_scores(path))
        scope_of = full_scores(merged)
    predict = encoder_predict(args.model, revision=args.revision, max_length=args.max_length,
                              device=args.device)
    started = time.perf_counter()
    scores = score_pools(pools, predict, scope_of=scope_of)
    elapsed = time.perf_counter() - started
    pairs = sum(v is not None for values in scores.values() for v in values)
    write_scores(args.out, scores, model=args.model, revision=args.revision,
                 max_length=args.max_length, pairs=pairs, seconds=round(elapsed, 1))
    print(f"{pairs} pairs in {elapsed:.0f} s ({1000 * elapsed / max(1, pairs):.1f} ms a pair) "
          f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
