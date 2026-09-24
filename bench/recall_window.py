"""Which part of a long turn `recall()` shows: its head, or the window the question names.

Run:  PYTHONPATH=. python3 bench/recall_window.py [--limit N] [--shuffle SEED]

LongMemEval-S, one store per question, the configuration `bench/longmemeval.py` runs by
default: the question's sessions are ingested, the question is searched with its own day
as `valid_at`, and the results are rendered by `recall()`'s own renderer twice — once with
the question, which shows each long turn as the window that matches it best
(`retrieve/excerpt.py`), and once without, which shows the first
`Memvara.RECALL_EPISODE_CHARS` characters, as every release before the window did. Both
renderings come from the same search, so retrieval cannot differ between the two columns.

The measure is the harness's string rule: whether at least 60% of the gold answer's content
words appear anywhere in the context, clipped to 4,000 characters as a reader would see it.
That is a proxy for whether a reader *could* answer, not a judged answer. The 30
unanswerable questions are left out, because their gold is a refusal sentence.

Needs the `s` file (`python3 bench/longmemeval.py --download --dataset s`). The shipped
`HashingEmbedder` and `NullLLM`, so nothing here calls a model and two runs are identical.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evalkit as ek  # noqa: E402
import longmemeval as lme  # noqa: E402

from memvara.retrieve import EpisodeResult  # noqa: E402
from memvara.select import PLAIN_READ  # noqa: E402


def contexts(mem, item, budget: ek.RetrievalBudget) -> tuple[str, str]:
    """The clipped block with the window, and the same block with the head cut."""
    results = mem.search(item.question, k=budget.k, include_episodes=True,
                         valid_at=item.anchor, **PLAIN_READ)
    claims = [r for r in results if not isinstance(r, EpisodeResult)]
    episodes = [r for r in results if isinstance(r, EpisodeResult)]
    headers = (mem._recall_header(item.anchor), mem.RECALL_HISTORY_HEADER,
               mem.RECALL_EPISODE_HEADER)
    keep = len(claims) + len(episodes)

    def render(query: str) -> str:
        block = mem._recall_block(claims, [[] for _ in claims], [], episodes, keep,
                                  headers, query=query)
        return ek.clip(block, budget.max_chars)

    return render(item.question), render("")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=0, help="stop after N questions")
    parser.add_argument("--shuffle", type=int, default=0, metavar="SEED",
                        help="shuffle questions with this seed before --limit")
    args = parser.parse_args()

    items = lme.load(ek.require(ek.LME_S))
    if args.shuffle:
        random.Random(args.shuffle).shuffle(items)
    if args.limit:
        items = items[: args.limit]
    budget = ek.RetrievalBudget()
    embedder = ek.build_embedder("hashing")
    found: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for item in items:
        if item.is_abstention:
            continue
        mem = lme.build_memory(item.qid, budget, embedder=embedder)
        try:
            ek.ingest(mem, item.sessions)
            window, head = contexts(mem, item, budget)
        finally:
            mem.close()
        gold = ek.content_tokens(item.answer)
        pair = (ek.coverage(window, gold) >= ek.DEFAULT_PRESENCE_THRESHOLD,
                ek.coverage(head, gold) >= ek.DEFAULT_PRESENCE_THRESHOLD)
        found[item.category].append(pair)
        found["all"].append(pair)

    print(f"\n  LongMemEval-S, one store per question: gold answer present in the "
          f"{budget.max_chars:,}-character context\n")
    print(f"  {'question type':<28} {'n':>4} {'head of turn':>13} {'window':>8}")
    for name in sorted(found, key=lambda c: (c == "all", c)):
        rows = found[name]
        head = 100 * sum(h for _, h in rows) / len(rows)
        window = 100 * sum(w for w, _ in rows) / len(rows)
        print(f"  {name:<28} {len(rows):>4} {head:>12.1f}% {window:>7.1f}%")
    print()
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by hand
    sys.exit(main())
