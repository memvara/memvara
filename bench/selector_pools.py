"""Candidate pools for the local selector: what a read retrieves, with gold labels.

    PYTHONPATH=. python3 bench/selector_pools.py longmemeval --data FILE.json --out OUT.jsonl
    PYTHONPATH=. python3 bench/selector_pools.py memorybench --run-dir DIR --data FILE.json --out OUT.jsonl
    PYTHONPATH=. python3 bench/selector_pools.py locomo --data locomo10.json --out OUT.jsonl
    PYTHONPATH=. python3 bench/selector_pools.py splits --pools A.jsonl [...] --test-ids IDS.txt --out S.json

A pool is the list of turns memvara's plain read returns for one question, up to 200 of
them, in the plain read's own order. That is the list the ranked stage reranks, routes and
cuts to the selector's 40 candidates (memvara/retrieve/hybrid.py:1751-1773), so any
selector can be replayed on it offline (selector_metrics.py).

Gold labels are attached after retrieval, by matching turn text against the turns the
dataset marks. Nothing here passes a label into ingestion or a query, which is the rule
tests/test_bench_eval.py pins for `longmemeval.parse_instance`.

Score files hold one model's raw score for every turn of every pool (`None` for a turn it
was not asked about), so the slow scoring step runs once per model (selector_score.py).
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Collection, Iterable, Mapping, Sequence

import evalkit as ek
import locomo
import longmemeval as lme

from memvara import Memvara, NullLLM
from memvara.retrieve import EpisodeResult
from memvara.select import PLAIN_READ

#: How many turns a pool holds: the ranked stage's rerank depth on the hosted service
#: (`_RERANK_TOP_N = 200` in memvara-cloud) and in `memvara-mcp` (memvara/server/config.py).
POOL_DEPTH = 200

#: LoCoMo's category numbers, as the dataset's own evaluation code names them. Category 5
#: is adversarial: its premise is false, so it has no evidence to find and is skipped.
LOCOMO_TYPES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}


@dataclass(slots=True)
class PoolTurn:
    id: str
    role: str
    text: str
    ts: str
    fused: float
    gold: bool
    #: The model selector's decision on this turn in a MemoryBench ranked run: True kept,
    #: False shown and omitted, None not shown or not known.
    paid_kept: bool | None = None


@dataclass(slots=True)
class Pool:
    source: str
    qid: str
    question: str
    qtype: str
    asked_on: str | None
    abstention: bool
    turns: list[PoolTurn] = field(default_factory=list)

    @property
    def gold_count(self) -> int:
        return sum(t.gold for t in self.turns)


def write_pools(path: Path, pools: Iterable[Pool]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as out:
        for pool in pools:
            out.write(json.dumps(asdict(pool), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_pools(path: Path) -> list[Pool]:
    pools = []
    with path.open(encoding="utf-8") as src:
        for line in src:
            if line.strip():
                raw = json.loads(line)
                raw["turns"] = [PoolTurn(**t) for t in raw["turns"]]
                pools.append(Pool(**raw))
    return pools


def write_scores(path: Path, scores: Mapping[str, Sequence[float | None]], **meta: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        out.write(json.dumps({"meta": meta}) + "\n")
        for qid, values in scores.items():
            out.write(json.dumps({"qid": qid, "scores": list(values)}) + "\n")


def read_scores(path: Path) -> dict[str, list[float | None]]:
    out: dict[str, list[float | None]] = {}
    with path.open(encoding="utf-8") as src:
        for line in src:
            row = json.loads(line)
            if "qid" in row:
                out[row["qid"]] = row["scores"]
    return out


def full_scores(scores: Mapping[str, Sequence[float | None]]) -> dict[str, list[float]]:
    """Scores that must cover every turn of every pool, as a reranker's do. A gap is
    refused: dropping it would shift every later score onto the wrong turn."""
    out: dict[str, list[float]] = {}
    for qid, values in scores.items():
        if any(v is None for v in values):
            raise ValueError(f"{qid}: some turns have no score; a reranker's scores must "
                             "cover every turn, so score them without --scope-of")
        out[qid] = [float(v) for v in values if v is not None]
    return out


def answer_turns(raw: Mapping[str, Any]) -> frozenset[str]:
    """The text of every turn a LongMemEval-schema item marks `has_answer`, stripped."""
    return frozenset(str(t.get("content") or "").strip()
                     for session in raw.get("haystack_sessions") or []
                     for t in session if t.get("has_answer"))


def _memory(embedder: Any, depth: int) -> Memvara:
    return Memvara(user="bench", llm=NullLLM(), embedder=embedder,
                   read_max_episodes=depth, **PLAIN_READ)


def _turns(results: Iterable[Any], is_gold: Callable[[Any], bool]) -> list[PoolTurn]:
    return [PoolTurn(id=r.episode.id, role=r.episode.role, text=r.episode.content,
                     ts=r.episode.ts.isoformat(), fused=float(r.score),
                     gold=bool(is_gold(r.episode)))
            for r in results if isinstance(r, EpisodeResult)]


def from_longmemeval(items: Sequence[Mapping[str, Any]], *, embedder: Any,
                     source: str = "longmemeval", depth: int = POOL_DEPTH) -> list[Pool]:
    """One pool per item, each from a fresh store holding only that item's haystack, read
    at the last second of the question's day as `bench/longmemeval.py` reads it."""
    pools = []
    for raw in items:
        inst = lme.parse_instance(dict(raw))
        mem = _memory(embedder, depth)
        ek.ingest(mem, inst.sessions)
        when = {"valid_at": inst.anchor} if inst.anchor is not None else {}
        results = mem.search(inst.question, k=depth, include_episodes=True, **when,
                             **PLAIN_READ)
        gold = answer_turns(raw)
        pools.append(Pool(
            source=source, qid=inst.qid, question=inst.question, qtype=inst.category,
            asked_on=None if inst.asked_on is None else inst.asked_on.isoformat(),
            abstention=inst.is_abstention,
            turns=_turns(results, lambda episode: episode.content.strip() in gold)))
        mem.close()
    return pools


def from_locomo(samples: Sequence[locomo.Sample], *, embedder: Any, source: str = "locomo",
                depth: int = POOL_DEPTH) -> list[Pool]:
    """One pool per answerable question, one store per conversation. A turn's role is its
    speaker's name, so the ranked stage's role routing finds no `user` turn and hands the
    selector both people's turns, which is what a two-person store needs."""
    pools = []
    for sample in samples:
        mem = _memory(embedder, depth)
        labels: dict[str, str] = {}
        ek.ingest(mem, [session.turns for session in sample.sessions], labels)
        known = sample.dia_ids
        for qa in sample.qa:
            evidence = qa.evidence_ids & known
            if qa.is_adversarial or not evidence:
                continue
            results = mem.search(qa.question, k=depth, include_episodes=True, **PLAIN_READ)
            pools.append(Pool(
                source=source, qid=f"{sample.sample_id}:{qa.index}", question=qa.question,
                qtype=LOCOMO_TYPES.get(qa.category, str(qa.category)), asked_on=None,
                abstention=False,
                turns=_turns(results, lambda episode: (
                    episode.meta.get(ek.LABEL_KEY) or labels.get(episode.id)) in evidence)))
        mem.close()
    return pools


def from_memorybench(run_dir: Path, items: Mapping[str, Mapping[str, Any]], *,
                     source: str = "longmemeval") -> list[Pool]:
    """Pools from a saved MemoryBench run. Each `results/<question id>.json` holds the
    server's turns in the order it returned them, with the model selector's `selected`
    flag on the 40 it was shown when the run was ranked. `items` maps each question id to
    its LongMemEval item, for the gold labels."""
    pools = []
    for path in sorted((run_dir / "results").glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        qid = str(result["questionId"])
        raw = items[qid]
        gold = answer_turns(raw)
        turns = []
        for position, item in enumerate(result.get("results") or []):
            if item.get("kind") != "turn":
                continue
            text = str(item.get("content") or "").strip()
            selected = item.get("selected")
            turns.append(PoolTurn(
                id=f"{qid}:{position}", role=str(item.get("role") or "user"), text=text,
                ts=str(item.get("ts") or ""), fused=float(item.get("score") or 0.0),
                gold=text in gold, paid_kept=None if selected is None else bool(selected)))
        pools.append(Pool(source=source, qid=qid, question=str(raw.get("question") or ""),
                          qtype=str(raw.get("question_type") or "unknown"), asked_on=None,
                          abstention=qid.endswith("_abs"), turns=turns))
    return pools


def scenario_of(qid: str) -> str:
    """A synthetic question's scenario: `syn-<domain>-<scenario>-q<n>` without `-q<n>`.
    Any other question is its own group."""
    return qid.rsplit("-q", 1)[0] if qid.startswith("syn-") else qid


def assign_splits(pools: Sequence[Pool], *, test_ids: Collection[str] = (),
                  test_only: Collection[str] = ("locomo",), validation_share: float = 0.15,
                  test_share: float = 0.15, seed: int = 0) -> dict[str, str]:
    """`train`, `validation` or `test` for every pool's question id.

    A question in `test_ids` is `test`, and a source in `test_only` is all `test`. Every
    other source is split by group (`scenario_of`), so no scenario has questions on both
    sides. A source with questions in `test_ids` gets no further test share: its test set
    is the one given."""
    fixed = set(test_ids)
    by_source: dict[str, list[Pool]] = {}
    for pool in pools:
        by_source.setdefault(pool.source, []).append(pool)
    out: dict[str, str] = {}
    for source, members in sorted(by_source.items()):
        if source in test_only:
            out.update({p.qid: "test" for p in members})
            continue
        groups = sorted({scenario_of(p.qid) for p in members if p.qid not in fixed})
        random.Random(f"{seed}:{source}").shuffle(groups)
        n_validation = round(len(groups) * validation_share)
        n_test = 0 if any(p.qid in fixed for p in members) else round(len(groups) * test_share)
        role = ({g: "validation" for g in groups[:n_validation]}
                | {g: "test" for g in groups[n_validation:n_validation + n_test]})
        for p in members:
            out[p.qid] = "test" if p.qid in fixed else role.get(scenario_of(p.qid), "train")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build candidate pools for the local selector.")
    sub = parser.add_subparsers(dest="command", required=True)
    lme_p = sub.add_parser("longmemeval", help="a LongMemEval-schema file, retrieved here")
    lme_p.add_argument("--data", type=Path, required=True)
    lme_p.add_argument("--source", default="longmemeval")
    lme_p.add_argument("--embedder", default="local", choices=["hashing", "local"])
    lme_p.add_argument("--limit", type=int)
    lme_p.add_argument("--out", type=Path, required=True)
    mb_p = sub.add_parser("memorybench", help="a saved MemoryBench run")
    mb_p.add_argument("--run-dir", type=Path, required=True)
    mb_p.add_argument("--data", type=Path, required=True)
    mb_p.add_argument("--out", type=Path, required=True)
    lc_p = sub.add_parser("locomo", help="the LoCoMo file, retrieved here (test only)")
    lc_p.add_argument("--data", type=Path, required=True)
    lc_p.add_argument("--embedder", default="local", choices=["hashing", "local"])
    lc_p.add_argument("--out", type=Path, required=True)
    sp_p = sub.add_parser("splits", help="assign train, validation and test")
    sp_p.add_argument("--pools", type=Path, nargs="+", required=True)
    sp_p.add_argument("--test-ids", type=Path)
    sp_p.add_argument("--seed", type=int, default=0)
    sp_p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "longmemeval":
        items = json.loads(args.data.read_text(encoding="utf-8"))[:args.limit]
        pools = from_longmemeval(items, embedder=ek.build_embedder(args.embedder),
                                 source=args.source)
    elif args.command == "memorybench":
        items = {str(r["question_id"]): r
                 for r in json.loads(args.data.read_text(encoding="utf-8"))}
        pools = from_memorybench(args.run_dir, items)
    elif args.command == "locomo":
        pools = from_locomo(locomo.load(args.data), embedder=ek.build_embedder(args.embedder))
    else:
        every = [p for path in args.pools for p in read_pools(path)]
        test_ids = (set(args.test_ids.read_text(encoding="utf-8").split())
                    if args.test_ids else set())
        splits = assign_splits(every, test_ids=test_ids, seed=args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(splits, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        counts: dict[str, int] = {}
        for split in splits.values():
            counts[split] = counts.get(split, 0) + 1
        print(f"{len(splits)} questions: {counts} -> {args.out}")
        return 0
    n = write_pools(args.out, pools)
    gold = sum(p.gold_count for p in pools)
    empty = sum(1 for p in pools if p.gold_count == 0)
    print(f"{n} pools, {gold} gold turns, {empty} with no gold turn retrieved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
