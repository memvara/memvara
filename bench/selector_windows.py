"""Show the part of a long turn that answers: measure answer windows (spec section 15).

    PYTHONPATH=. python3 bench/selector_windows.py --pools P.jsonl --splits S.json \
        --split test --rerank STOCK.jsonl --select MODEL.jsonl --model MODEL_DIR \
        --data longmemeval_s_cleaned.json [--max-length 256] [--device mps]

Every rendering here uses rendering (e): the selector's candidates in the local model's
score order, then every other turn in reranked order. Turns are flattened to one line,
as `recall()` shows them, and a 720-token block is filled greedily in that order. What
changes is how a turn longer than `LIMIT` characters is shown:

- `whole`: the whole turn.
- `lexical`: `memvara.retrieve.excerpt.excerpt`, the window `recall()` shows today.
- `model`: the *answer window* the local model scores highest. A turn's windows are runs of
  whole consecutive sentences of at most `LIMIT` characters, one starting at each sentence;
  a sentence longer than that is cut to its first `LIMIT - 1` characters and an ellipsis.

The paid selector's row renders its own order with whole turns, as MemoryBench saw it.

An answer is *shown* when at least 60% of the gold answer's content words appear in the
block: the presence rule `bench/recall_window.py` uses, a proxy for whether a reader could
answer and not a judged answer. Abstention questions are left out.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import evalkit as ek
from selector_metrics import BUDGET, TokenCounter, paired_bootstrap, replay, tiktoken_counter
from selector_pools import Pool, full_scores, read_pools, read_scores
from selector_score import Predict, encoder_predict

from memvara.retrieve.excerpt import ELLIPSIS, _framed, _sentences, excerpt

LIMIT = 280
MODES = ("whole", "lexical", "model")

#: Turns a block is filled from, at most. 720 tokens never holds more than this many.
DEPTH = 120


def flatten(text: str) -> str:
    return " ".join(text.split())


def answer_windows(text: str, limit: int = LIMIT) -> list[str]:
    """Every window of `text` the model chooses between, one starting at each sentence."""
    spans = _sentences(text)
    windows: list[str] = []
    for i, (a, b) in enumerate(spans):
        if (b - a) + (a > 0) + 1 > limit:
            windows.append((ELLIPSIS if a > 0 else "") + text[a:a + limit - 2].rstrip()
                           + ELLIPSIS)
            continue
        end = b
        for _a2, b2 in spans[i + 1:]:
            if (b2 - a) + (a > 0) + (b2 < len(text)) > limit:
                break
            end = b2
        windows.append(_framed(text, a, end))
    return windows


def selector_order(pool: Pool, rerank: Sequence[float], select: Sequence[float | None]
                   ) -> list[int]:
    rep = replay(pool, rerank, None)
    scope = sorted(rep.scope, key=lambda i: -float(select[i] or 0.0))
    inside = set(scope)
    return scope + [i for i in rep.order if i not in inside]


def shown_text(text: str, question: str, mode: str,
               choose: Callable[[str, list[str]], str] | None) -> str:
    if mode == "whole" or len(text) <= LIMIT:
        return text
    if mode == "lexical":
        return excerpt(text, question, LIMIT)
    assert choose is not None
    return choose(question, answer_windows(text))


def fill(texts: Sequence[str], count: TokenCounter, budget: int = BUDGET) -> str:
    used = 0
    block: list[str] = []
    for text in texts:
        n = count(text)
        if used + n <= budget:
            used += n
            block.append(text)
    return "\n".join(block)


def model_chooser(predict: Predict, tally: list[int]) -> Callable[[str, list[str]], str]:
    def choose(question: str, windows: list[str]) -> str:
        if len(windows) == 1:
            return windows[0]
        tally.append(len(windows))
        scores = list(predict([(question, w) for w in windows]))
        return windows[max(range(len(windows)), key=lambda k: scores[k])]
    return choose


def answer_shown(block: str, answer: str) -> bool:
    gold = ek.content_tokens(answer)
    return bool(gold) and ek.coverage(block, gold) >= ek.DEFAULT_PRESENCE_THRESHOLD


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure answer windows on pools.")
    parser.add_argument("--pools", type=Path, nargs="+", required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--rerank", type=Path, nargs="+", required=True)
    parser.add_argument("--select", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device")
    args = parser.parse_args(argv)

    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    answers = {d["question_id"]: d["answer"]
               for d in json.loads(args.data.read_text(encoding="utf-8"))}
    pools = [p for path in args.pools for p in read_pools(path)
             if splits.get(p.qid) == args.split and not p.abstention]
    raw: dict[str, list[float | None]] = {}
    for path in args.rerank:
        raw.update(read_scores(path))
    rerank = full_scores(raw)
    select = read_scores(args.select)
    count = tiktoken_counter()
    tally: list[int] = []
    choose = model_chooser(encoder_predict(args.model, max_length=args.max_length,
                                           device=args.device), tally)

    rows: dict[str, dict[str, tuple[int, int]]] = {m: {} for m in (*MODES, "paid")}
    kinds: dict[str, str] = {}
    for pool in pools:
        answer = str(answers[pool.qid])
        kinds[pool.qid] = pool.qtype
        texts = [flatten(t.text) for t in pool.turns]
        order = selector_order(pool, rerank[pool.qid], select[pool.qid])[:DEPTH]
        for mode in MODES:
            block = fill([shown_text(texts[i], pool.question, mode, choose) for i in order],
                         count)
            rows[mode][pool.qid] = (int(answer_shown(block, answer)), 1)
        # A MemoryBench pool holds the turns in the order the ranked run returned them,
        # which is how selector_metrics.py renders the paid row too.
        rows["paid"][pool.qid] = (int(answer_shown(fill(texts, count), answer)), 1)

    def share(name: str, qids: Sequence[str]) -> float:
        return sum(rows[name][q][0] for q in qids) / len(qids)

    qids = sorted(kinds)
    print(f"| rendering | shown, all {len(qids)} | vs whole |")
    print("|---|---|---|")
    for name in rows:
        versus = "-"
        if name != "whole":
            p = paired_bootstrap(rows[name], rows["whole"])
            versus = f"{p.diff:+.3f} [{p.low:+.3f}, {p.high:+.3f}] {p.better}/{p.worse}"
        print(f"| {name} | {share(name, qids):.3f} | {versus} |")
    print()
    print("| type | n | " + " | ".join(rows) + " |")
    print("|" + "---|" * (len(rows) + 2))
    for kind in sorted(set(kinds.values())):
        of = [q for q in qids if kinds[q] == kind]
        print(f"| {kind} | {len(of)} | " + " | ".join(f"{share(n, of):.3f}" for n in rows)
              + " |")
    if tally:
        print(f"\nThe model scored {sum(tally) / len(pools):.1f} windows per read on average "
              f"({len(tally)} long turns, {sum(tally) / len(tally):.1f} windows each).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
