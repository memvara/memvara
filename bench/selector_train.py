"""Fine-tune the selector's cross-encoder on gold-labelled candidates, then calibrate it.

    PYTHONPATH=. python3 bench/selector_train.py --pools P.jsonl [...] --splits S.json \
        --rerank STOCK.jsonl [...] --out local/selector/models/NAME [--hold-out SOURCE] \
        [--label gold|paid] [--epochs 1] [--lr 2e-5] [--max-length 256] [--batch-size 16] \
        [--negatives 3] [--pos-weight 1.0] [--seed 0] [--device mps] \
        [--pairs PUBLIC.jsonl [...]] [--no-pool-pairs] [--base MODEL_DIR] \
        [--calibrate-on validation [--calibrate-on train]]

**Training pairs** come from the `train` split only. For each question, the script takes
the candidates the selector would be handed: the stock reranker's order, routed, first 40.
It keeps every positive, with up to `--negatives` negatives per positive chosen at random.
Positives are the dataset's gold turns. With `--label paid` they are the model selector's
kept turns instead; that label is used only to measure what gold labels lose (Task 14).

**Calibration.** The negatives are subsampled, so the trained model's raw probability is
not calibrated. After training, the model scores the candidates of the `validation` split,
and Platt scaling and the keep-rule search run there.

**Public pairs.** `--pairs` adds pairs that selector_public.py wrote from public datasets,
and `--no-pool-pairs` trains on those alone (step 2b, arm C). `--base` starts from an
already fine-tuned model directory instead of the stock model (arm D continues from C).

**Calibrating on more questions.** A model that saw no pool during training can be
calibrated on the `train` split as well: `--calibrate-on validation --calibrate-on train`.

**Leaving a source out.** `--hold-out SOURCE` leaves that source out of training and
calibration both, for the leave-one-source-out gate.

**Output.** The output directory holds the model, its tokenizer and
`memvara_selector.json`, which `LocalSelector(<directory>)` reads and checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from selector_calibrate import choose_keep, fit_platt, scope_examples, write_calibration
from selector_metrics import replay, tiktoken_counter
from selector_pools import Pool, full_scores, read_pools, read_scores
from selector_public import read_jsonl, read_pairs
from selector_score import encoder_predict, score_pools

from memvara.select.local import STOCK_MODEL, STOCK_REVISION, Calibration


def training_pairs(pools: Sequence[Pool], splits: Mapping[str, str],
                   rerank: Mapping[str, Sequence[float]], *, label: str = "gold",
                   negatives: int = 3, seed: int = 0,
                   hold_out: str | None = None) -> list[tuple[str, str, float]]:
    rng = random.Random(seed)
    pairs: list[tuple[str, str, float]] = []
    for pool in sorted(pools, key=lambda p: p.qid):
        if splits.get(pool.qid) != "train" or pool.source == hold_out:
            continue
        scope = replay(pool, rerank[pool.qid], None).scope
        if label == "paid":
            labelled = [(i, bool(pool.turns[i].paid_kept)) for i in scope
                        if pool.turns[i].paid_kept is not None]
        else:
            labelled = [(i, pool.turns[i].gold) for i in scope]
        positives = [i for i, y in labelled if y]
        others = [i for i, y in labelled if not y]
        rng.shuffle(others)
        for i in positives + others[:negatives * len(positives)]:
            pairs.append((pool.question, pool.turns[i].text, 1.0 if i in positives else 0.0))
    return pairs


def write_selector_json(directory: Path, calibration: Calibration, **extra: Any) -> Path:
    digest = hashlib.sha256((directory / "model.safetensors").read_bytes()).hexdigest()
    return write_calibration(directory, calibration, weights_sha256=digest, **extra)


def train(pairs: Sequence[tuple[str, str, float]], base_model: str, revision: str | None,
          out_dir: Path, *, epochs: int = 1, lr: float = 2e-5, max_length: int = 256,
          batch_size: int = 16, seed: int = 0, device: str | None = None,
          pos_weight: float = 1.0) -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(seed)
    torch.manual_seed(seed)
    dev = device or ("cuda" if torch.cuda.is_available()
                     else "mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(base_model, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, revision=revision, num_labels=1).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    # `pos_weight` above 1 makes missing a gold turn cost more than keeping a wrong one,
    # which pushes the model to keep more turns (step 2a, arm B).
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=dev))
    steps = epochs * math.ceil(len(pairs) / batch_size)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / max(1.0, 0.1 * steps))
        * max(0.0, (steps - s) / steps))
    order = list(pairs)
    rng = random.Random(seed)
    model.train()
    for epoch in range(epochs):
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            encoded = tokenizer([q for q, _t, _y in batch], [t for _q, t, _y in batch],
                                truncation="only_second", max_length=max_length,
                                padding=True, return_tensors="pt").to(dev)
            target = torch.tensor([y for _q, _t, y in batch], device=dev)
            loss = loss_fn(model(**encoded).logits.squeeze(-1), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
            if (start // batch_size) % 50 == 0:
                print(f"epoch {epoch} batch {start // batch_size} loss {loss.item():.4f}",
                      flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune and calibrate a selector model.")
    parser.add_argument("--pools", type=Path, nargs="+", required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--rerank", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hold-out")
    parser.add_argument("--label", choices=["gold", "paid"], default="gold")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--negatives", type=int, default=3)
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--pairs", type=Path, nargs="*", default=[])
    parser.add_argument("--no-pool-pairs", action="store_true")
    parser.add_argument("--base", type=Path)
    parser.add_argument("--calibrate-on", action="append")
    args = parser.parse_args(argv)
    calibrate_on = set(args.calibrate_on or ["validation"])

    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    pools = [p for path in args.pools for p in read_pools(path)]
    raw: dict[str, list[float | None]] = {}
    for path in args.rerank:
        raw.update(read_scores(path))
    rerank = full_scores(raw)
    pairs = [] if args.no_pool_pairs else training_pairs(
        pools, splits, rerank, label=args.label, negatives=args.negatives, seed=args.seed,
        hold_out=args.hold_out)
    public = [pair for path in args.pairs for pair in read_pairs(path)]
    pairs += public
    print(f"{len(pairs)} training pairs ({len(public)} public), "
          f"{int(sum(y for *_x, y in pairs))} positive", flush=True)
    base, revision = (str(args.base), None) if args.base else (STOCK_MODEL, STOCK_REVISION)
    train(pairs, base, revision, args.out, epochs=args.epochs, lr=args.lr,
          max_length=args.max_length, batch_size=args.batch_size, seed=args.seed,
          device=args.device, pos_weight=args.pos_weight)
    validation = [p for p in pools
                  if splits.get(p.qid) in calibrate_on and p.source != args.hold_out]
    predict = encoder_predict(str(args.out), max_length=args.max_length, device=args.device)
    select = score_pools(validation, predict, scope_of=rerank)
    xs, ys = scope_examples(validation, rerank, select)
    scale, shift = fit_platt(xs, ys)
    calibration, cov, mean_kept = choose_keep(validation, rerank, select, scale, shift,
                                              tiktoken_counter(), max_length=args.max_length)
    sources: dict[str, int] = {}
    for p in pools:
        if (not args.no_pool_pairs and splits.get(p.qid) == "train"
                and p.source != args.hold_out):
            sources[p.source] = sources.get(p.source, 0) + 1
    for path in args.pairs:
        for row in {(r["source"], r["qid"]) for r in read_jsonl(path)}:
            sources[row[0]] = sources.get(row[0], 0) + 1
    write_selector_json(args.out, calibration, base_model=STOCK_MODEL,
                        base_revision=STOCK_REVISION, sources=sources, label=args.label,
                        hold_out=args.hold_out, seed=args.seed, epochs=args.epochs,
                        lr=args.lr, negatives=args.negatives, pos_weight=args.pos_weight,
                        started_from=str(args.base) if args.base else None,
                        calibrated_on=sorted(calibrate_on))
    print(f"{'+'.join(sorted(calibrate_on))} coverage {cov:.3f} at {mean_kept:.1f} kept; {calibration!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
