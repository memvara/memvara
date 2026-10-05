"""Time `LocalSelector.select()` and measure the memory it adds, for gate G6.

    OMP_NUM_THREADS=4 PYTHONPATH=. python3 bench/selector_latency.py [--model ID_OR_DIR] \
        [--pools P.jsonl] [--reads 50]

G6 passes when `select_ordered()` (what the ranked stage calls) p95 is at most 1.0 s for 40 candidates, including a question
4,000 characters long, and resident memory grows by at most 300 MB with the reranker
already loaded. Run it on 4 threads, which is the production host's core count, with
nothing else busy. The candidates come from real pools when `--pools` is given, and are
otherwise 40 copies of a 400-token assistant-length turn, which is the slow case. The
encoder is kept on the CPU with 4 threads even where a GPU is available, because G6 is a
CPU gate.

`--steady with|without` is the steady-state memory measure of the spec's section 18: a fresh
process serves `--reads` production-shaped ranked reads, each reranking one test pool's
turns and, with `with`, running `select_ordered()` on its 40 routed candidates. It prints
the median current resident memory over the last 10 reads. Run each kind three times, in
fresh processes, and compare the medians.
"""

from __future__ import annotations

import argparse
import resource
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from selector_metrics import replay
from selector_pools import read_pools

from memvara.rerank.cross import CrossEncoderReranker
from memvara.select import Candidate
from memvara.select.local import STOCK_MODEL, Calibration, LocalSelector

WHEN = datetime(2026, 1, 1, tzinfo=timezone.utc)
LONG_TURN = ("I set up the build so that the integration tests run against a fresh "
             "database each time, and the flaky one turned out to be a timezone issue. ") * 12


def rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def steady(args: argparse.Namespace) -> int:
    import json

    import psutil
    import torch
    from sentence_transformers import CrossEncoder

    torch.set_num_threads(4)
    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    pools = [p for p in read_pools(args.pools) if splits.get(p.qid) == "test"][:args.reads]
    reranker = CrossEncoderReranker(encoder=CrossEncoder(STOCK_MODEL, device="cpu"))
    selector = None
    if args.steady == "with":
        calibration = Calibration.read(Path(args.model))[0]
        selector = LocalSelector(args.model, calibration=calibration, encoder=CrossEncoder(
            args.model, device="cpu", max_length=calibration.max_length))
    process = psutil.Process()
    samples = []
    for pool in pools:
        texts = [t.text for t in pool.turns]
        scores = reranker.score(pool.question, texts)
        scope = replay(pool, scores, None).scope
        if selector is not None:
            selector.select_ordered(pool.question, [
                Candidate(id=str(i), when=WHEN, text=texts[i]) for i in scope])
        samples.append(process.memory_info().rss / (1024 * 1024))
    print(f"{args.steady}: {len(pools)} reads, median resident memory over the last 10: "
          f"{statistics.median(samples[-10:]):.0f} MB")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Time the local selector.")
    parser.add_argument("--model", default=STOCK_MODEL)
    parser.add_argument("--pools", type=Path)
    parser.add_argument("--reads", type=int, default=50)
    parser.add_argument("--steady", choices=["with", "without"])
    parser.add_argument("--splits", type=Path)
    args = parser.parse_args(argv)
    if args.steady:
        return steady(args)

    import torch
    torch.set_num_threads(4)

    from sentence_transformers import CrossEncoder

    # The reranker a ranked read already holds, on the CPU, and already used once: a
    # production process has run it before any selection, so torch's first-inference
    # buffers are not the selector's. Both models load straight onto the CPU, because a
    # GPU this machine happens to have is not the production host.
    reranker = CrossEncoderReranker(encoder=CrossEncoder(STOCK_MODEL, device="cpu"))
    reranker.score("warm up", ["the reranker has run once"] * 40)
    before = rss_mb()
    calibration = None
    if Path(args.model).is_dir():
        calibration = Calibration.read(Path(args.model))[0]
    selector = LocalSelector(
        args.model, calibration=calibration,
        encoder=CrossEncoder(args.model, device="cpu",
                             max_length=calibration.max_length if calibration else None))
    added = rss_mb() - before
    cases: list[tuple[str, list[Candidate]]] = []
    if args.pools:
        for pool in read_pools(args.pools)[:args.reads]:
            scope = replay(pool, [t.fused for t in pool.turns], None).scope
            cases.append((pool.question, [Candidate(id=str(i), when=WHEN,
                                                    text=pool.turns[i].text) for i in scope]))
    else:
        filler = [Candidate(id=str(i), when=WHEN, text=LONG_TURN) for i in range(40)]
        cases = [("why did the integration tests fail last week?", filler)] * args.reads
    cases.append(("why does the build fail? " * 160, cases[0][1]))     # 4,000 characters
    selector.select_ordered(*cases[0])                                  # warm-up
    timings = []
    for question, candidates in cases:
        started = time.perf_counter()
        selector.select_ordered(question, candidates)
        timings.append(time.perf_counter() - started)
    long_question = timings[-1]
    timings.sort()
    p50 = statistics.median(timings)
    p95 = timings[min(len(timings) - 1, int(0.95 * len(timings)))]
    print(f"{len(cases)} reads of up to 40 candidates: p50 {p50 * 1000:.0f} ms, "
          f"p95 {p95 * 1000:.0f} ms, max {timings[-1] * 1000:.0f} ms; the 4,000-character "
          f"question {long_question * 1000:.0f} ms")
    added = max(added, rss_mb() - before)                # the peak after every read
    print(f"device {selector._encoder.device}, {torch.get_num_threads()} threads")
    print(f"resident memory added by the selector: {added:.0f} MB (measured from the peak)")
    ok = p95 <= 1.0 and long_question <= 1.0 and added <= 300
    print("G6:", "pass" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
