"""Replay a ranked read on a pool, and measure what share of the evidence it shows.

    PYTHONPATH=. python3 bench/selector_metrics.py --pools P.jsonl [...] --splits S.json \
        --split test --rerank STOCK.jsonl [...] [--local NAME=SCORES.jsonl,CALIBRATION_DIR] \
        [--paid] [--baseline routed] [--by-type]

`replay` follows memvara/retrieve/hybrid.py's ranked stage step for step. It sorts the
pool by the reranker's scores; the sort is stable, so ties keep the plain read's order. It
keeps the questioner's role when routing is on and that role has any turn (`routed_role`),
and takes the first `top_n` as the selector's candidates. It then renders the kept ones
first, in candidate order, followed by every other turn in reranked order.

`coverage` counts gold turns rendered whole inside the first `budget` tokens, greedily: a
turn that does not fit is skipped and the next one is tried. That is how the MemoryBench
harness fills its 720-token block from a ranked read's turns. Token counts use tiktoken's
cl100k_base, as the 2026-09-27 measurements did (`pip install tiktoken`).

The orderings in a report:
- `plain`: the plain read's own order (the fused score).
- `routed`: the stock reranker's order with routing, every candidate kept. This is the
  reranker-and-routing order the judged routed-720 arm used.
- `paid`: the order a MemoryBench ranked run returned (`--paid`, MemoryBench pools only).
- one row per `--local`: a local selector's keep decisions, replayed with
  `memvara.select.local.keep_positions`, the function the selector itself uses.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Collection, Mapping, Sequence

from selector_pools import Pool, full_scores, read_pools, read_scores

from memvara.retrieve.intent import routed_role
from memvara.select.local import Calibration, keep_positions

TokenCounter = Callable[[str], int]
TOP_N = 40
BUDGET = 720


@dataclass(frozen=True, slots=True)
class Replay:
    order: list[int]
    scope: list[int]
    rendered: list[int]


def replay(pool: Pool, rerank: Sequence[float], kept: Collection[int] | None, *,
           route: bool = True, top_n: int = TOP_N) -> Replay:
    order = sorted(range(len(pool.turns)), key=lambda i: -rerank[i])
    role = routed_role(pool.question)
    routed = [i for i in order if pool.turns[i].role == role] if route else []
    scope = (routed or order)[:top_n]
    if kept is None:
        return Replay(order, scope, order)
    keep = set(kept) & set(scope)
    head = [i for i in scope if i in keep]
    return Replay(order, scope, head + [i for i in order if i not in keep])


def local_replay(pool: Pool, rerank: Sequence[float], select: Sequence[float | None],
                 calibration: Calibration, *, route: bool = True,
                 top_n: int = TOP_N) -> tuple[Replay, set[int]]:
    scope = replay(pool, rerank, None, route=route, top_n=top_n).scope
    values = [select[i] for i in scope]
    if any(v is None for v in values):
        raise ValueError(f"{pool.qid}: the selector scores do not cover every candidate. "
                         "Score with --scope-of the same reranker scores.")
    scores = [float(v) for v in values]      # every value is set: checked just above
    kept = {scope[p] for p in keep_positions(scores, calibration)}
    return replay(pool, rerank, kept, route=route, top_n=top_n), kept


def coverage(pool: Pool, rendered: Sequence[int], count: TokenCounter,
             budget: int = BUDGET) -> tuple[int, int]:
    used = got = 0
    for i in rendered:
        n = count(pool.turns[i].text)
        if used + n <= budget:
            used += n
            got += pool.turns[i].gold
    return got, pool.gold_count


def kept_recall(pool: Pool, scope: Sequence[int], kept: Collection[int]) -> tuple[int, int]:
    gold = [i for i in scope if pool.turns[i].gold]
    return sum(1 for i in gold if i in kept), len(gold)


@dataclass(frozen=True, slots=True)
class Paired:
    diff: float
    low: float
    high: float
    better: int
    worse: int


def _share(per_q: Mapping[str, tuple[int, int]], qids: Sequence[str]) -> float:
    total = sum(per_q[q][1] for q in qids)
    return sum(per_q[q][0] for q in qids) / total if total else 0.0


def paired_bootstrap(a: Mapping[str, tuple[int, int]], b: Mapping[str, tuple[int, int]], *,
                     resamples: int = 5000, seed: int = 0) -> Paired:
    """`a` minus `b` in pooled coverage, with a 95% interval from resampling questions."""
    qids = sorted(q for q in a if a[q][1] > 0)
    rng = random.Random(seed)
    diffs = sorted(_share(a, s) - _share(b, s)
                   for s in ([rng.choice(qids) for _ in qids] for _ in range(resamples)))
    return Paired(diff=_share(a, qids) - _share(b, qids),
                  low=diffs[int(0.025 * resamples)], high=diffs[int(0.975 * resamples) - 1],
                  better=sum(a[q][0] > b[q][0] for q in qids),
                  worse=sum(a[q][0] < b[q][0] for q in qids))


def tiktoken_counter() -> TokenCounter:
    try:
        import tiktoken
    except ImportError as exc:
        raise SystemExit("this script counts tokens with tiktoken: pip install tiktoken") from exc
    encoding = tiktoken.get_encoding("cl100k_base")
    cache: dict[str, int] = {}

    def count(text: str) -> int:
        if text not in cache:
            cache[text] = len(encoding.encode(text, disallowed_special=()))
        return cache[text]

    return count


Row = tuple[int, int, int, int, int]  # gold rendered, gold in pool, gold kept, gold shown, kept


def evaluate(pools: Sequence[Pool], rerank: Mapping[str, Sequence[float]],
             selectors: Mapping[str, tuple[Mapping[str, Sequence[float | None]], Calibration]],
             count: TokenCounter, *, paid: bool = False,
             budget: int = BUDGET) -> dict[str, dict[str, Row]]:
    """Per ordering, per question with a gold turn in its pool, a `Row`."""
    out: dict[str, dict[str, Row]] = {}

    def put(name: str, pool: Pool, rendered: Sequence[int], scope: Sequence[int],
            kept: Collection[int]) -> None:
        got, total = coverage(pool, rendered, count, budget)
        hit, shown = kept_recall(pool, scope, kept)
        out.setdefault(name, {})[pool.qid] = (got, total, hit, shown, len(kept))

    for pool in pools:
        if pool.gold_count == 0:
            continue
        plain = replay(pool, [t.fused for t in pool.turns], None, route=False)
        put("plain", pool, plain.rendered, [], ())
        stock = rerank[pool.qid]
        scope = replay(pool, stock, None).scope
        put("routed", pool, replay(pool, stock, scope).rendered, scope, scope)
        if paid:
            shown = [i for i, t in enumerate(pool.turns) if t.paid_kept is not None]
            kept = {i for i, t in enumerate(pool.turns) if t.paid_kept}
            put("paid", pool, list(range(len(pool.turns))), shown, kept)
        for name, (scores, calibration) in selectors.items():
            rep, kept = local_replay(pool, stock, scores[pool.qid], calibration)
            put(name, pool, rep.rendered, rep.scope, kept)
    return out


def table(results: Mapping[str, Mapping[str, Row]], *, baseline: str | None) -> str:
    lines = [f"| ordering | questions | coverage | kept recall | mean kept | vs {baseline} |",
             "|---|---|---|---|---|---|"]
    for name, per_q in results.items():
        got = sum(r[0] for r in per_q.values())
        total = sum(r[1] for r in per_q.values())
        hit = sum(r[2] for r in per_q.values())
        shown = sum(r[3] for r in per_q.values())
        mean_kept = sum(r[4] for r in per_q.values()) / len(per_q)
        versus = "-"
        if baseline and name != baseline and baseline in results:
            p = paired_bootstrap({q: r[:2] for q, r in per_q.items()},
                                 {q: r[:2] for q, r in results[baseline].items()})
            versus = f"{p.diff:+.3f} [{p.low:+.3f}, {p.high:+.3f}] {p.better}/{p.worse}"
        recall = f"{hit / shown:.3f}" if shown else "-"
        lines.append(f"| {name} | {len(per_q)} | {got / total:.3f} | {recall} | "
                     f"{mean_kept:.1f} | {versus} |")
    return "\n".join(lines)


def type_table(pools: Sequence[Pool], results: Mapping[str, Mapping[str, Row]]) -> str:
    qtype = {p.qid: p.qtype for p in pools}
    first = next(iter(results.values()))
    lines = ["| type | n | " + " | ".join(results) + " |", "|" + "---|" * (len(results) + 2)]
    for kind in sorted({qtype[q] for q in first}):
        qids = [q for q in first if qtype[q] == kind]
        cells = []
        for per_q in results.values():
            total = sum(per_q[q][1] for q in qids)
            cells.append(f"{sum(per_q[q][0] for q in qids) / total:.3f}")
        lines.append(f"| {kind} | {len(qids)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay ranked reads on pools and report coverage.")
    parser.add_argument("--pools", type=Path, nargs="+", required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--rerank", type=Path, nargs="+", required=True,
                        help="stock-model score files covering every pool")
    parser.add_argument("--local", action="append", default=[],
                        metavar="NAME=SCORES.jsonl,CALIBRATION_DIR")
    parser.add_argument("--paid", action="store_true")
    parser.add_argument("--baseline", default="routed")
    parser.add_argument("--by-type", action="store_true")
    args = parser.parse_args(argv)

    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    pools = [p for path in args.pools for p in read_pools(path)
             if splits.get(p.qid) == args.split]
    rerank: dict[str, list[float | None]] = {}
    for path in args.rerank:
        rerank.update(read_scores(path))
    selectors = {}
    for spec in args.local:
        name, _, rest = spec.partition("=")
        scores_path, _, calibration_dir = rest.partition(",")
        calibration, _digest = Calibration.read(Path(calibration_dir))
        selectors[name] = (read_scores(Path(scores_path)), calibration)
    results = evaluate(pools, full_scores(rerank), selectors, tiktoken_counter(),
                       paid=args.paid)
    print(table(results, baseline=args.baseline))
    if args.by_type:
        print()
        print(type_table(pools, results))
    left_out = sum(1 for p in pools if p.gold_count == 0)
    print(f"\n{len(pools)} questions in split {args.split!r}; {left_out} have no gold turn "
          "in their pool and are left out of every row.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
