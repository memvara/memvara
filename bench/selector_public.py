"""Turn public question-answering datasets into training pairs for the selector.

    PYTHONPATH=. python3 bench/selector_public.py --hotpot H1.parquet [...] \
        --twowiki W1.parquet [...] --musique M.jsonl --quac Q.json \
        [--take hotpot=30000 --take twowiki=30000 --take musique=0 --take quac=20000] \
        [--negatives 3] [--seed 0] --out local/selector/public/pairs.jsonl

Each dataset marks which sentence or paragraph holds the answer to a question. That is the
same shape as the selector's job: a question, a list of candidate turns, and the turns that
carry the answer. The spec's section 13 fixes how each dataset is read:

- **HotpotQA** (distractor setting) and **2WikiMultihopQA**: a candidate is one sentence of
  the ten context paragraphs, and gold is a sentence listed in `supporting_facts`.
- **MuSiQue** (answerable set): a candidate is one of the twenty paragraphs, and gold is a
  paragraph marked `is_supporting`.
- **QuAC**: a candidate is one sentence of the Wikipedia section, and gold is the sentence
  where the answer starts. Unanswerable questions are skipped.

The query is the question alone: no paragraph title and no earlier dialogue turn, because
a real read has only the question and the turn. For each gold candidate, `--negatives`
non-gold candidates are drawn at random from the same question's context, the ratio
selector_train.py uses on LongMemEval. `--take NAME=N` draws N questions at random from a
dataset, and 0 keeps them all. Every random draw is seeded by `--seed` and the dataset's
name, so adding a dataset does not change the draws from another.

The output holds one JSON object per pair, with `source`, `qid`, `query`, `text` and
`label`. `selector_train.py --pairs` reads it.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

#: The questions the spec's section 13 draws from each dataset. 0 keeps every question.
DEFAULT_TAKE = {"hotpot": 30_000, "twowiki": 30_000, "musique": 0, "quac": 20_000}

#: QuAC appends this marker to every section's text, and gives it as the answer to a
#: question the section cannot answer.
QUAC_NO_ANSWER = "CANNOTANSWER"

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=\S)")


@dataclass(slots=True)
class Example:
    source: str
    qid: str
    question: str
    candidates: list[str]
    gold: set[int] = field(default_factory=set)


def split_sentences(text: str) -> list[tuple[int, int, str]]:
    """Split text after `.`, `!` or `?` followed by whitespace. Returns each sentence's
    start and end offsets in `text`, so an answer offset can be mapped to its sentence."""
    spans = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        spans.append((start, match.start(), text[start:match.start()]))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text), text[start:]))
    return [(s, e, t.strip()) for s, e, t in spans if t.strip()]


def from_supporting_facts(source: str, row: Mapping[str, Any]) -> Example | None:
    """A HotpotQA or 2WikiMultihopQA row. A supporting fact names a paragraph by its title
    and a sentence by its position; a fact that points past the paragraph's end, which a
    few HotpotQA rows do, is ignored."""
    candidates: list[str] = []
    where: dict[tuple[str, int], int] = {}
    context = row["context"]
    for title, sentences in zip(context["title"], context["sentences"]):
        for n, sentence in enumerate(sentences):
            if sentence.strip():
                where[(title, n)] = len(candidates)
                candidates.append(sentence.strip())
    facts = row["supporting_facts"]
    gold = {where[(t, n)] for t, n in zip(facts["title"], facts["sent_id"]) if (t, n) in where}
    if not gold:
        return None
    return Example(source, str(row["id"]), row["question"], candidates, gold)


def from_musique(row: Mapping[str, Any]) -> Example | None:
    paragraphs = row["paragraphs"]
    gold = {i for i, p in enumerate(paragraphs) if p["is_supporting"]}
    if not gold:
        return None
    return Example("musique", str(row["id"]), row["question"],
                   [p["paragraph_text"] for p in paragraphs], gold)


def from_quac(article: Mapping[str, Any]) -> Iterator[Example]:
    """Every answerable question of one QuAC dialogue. The candidates are the sentences of
    the section, with QuAC's no-answer marker removed from its end."""
    for paragraph in article["paragraphs"]:
        context = paragraph["context"]
        if context.endswith(QUAC_NO_ANSWER):
            context = context[:-len(QUAC_NO_ANSWER)].rstrip()
        spans = split_sentences(context)
        candidates = [text for _s, _e, text in spans]
        for qa in paragraph["qas"]:
            answer = qa["orig_answer"]
            if answer["text"] == QUAC_NO_ANSWER:
                continue
            offset = answer["answer_start"]
            gold = {i for i, (s, e, _t) in enumerate(spans) if s <= offset < e}
            if gold:
                yield Example("quac", qa["id"], qa["question"], candidates, gold)


def draw(examples: Sequence[Example], take: int, rng: random.Random) -> list[Example]:
    if take <= 0 or take >= len(examples):
        return list(examples)
    return rng.sample(list(examples), take)


def pairs_of(example: Example, negatives: int,
             rng: random.Random) -> list[tuple[str, str, float]]:
    others = [i for i in range(len(example.candidates)) if i not in example.gold]
    rng.shuffle(others)
    chosen = sorted(example.gold) + others[:negatives * len(example.gold)]
    return [(example.question, example.candidates[i], 1.0 if i in example.gold else 0.0)
            for i in chosen]


def build(sources: Mapping[str, Sequence[Example]], take: Mapping[str, int], *,
          negatives: int = 3, seed: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in sorted(sources):
        rng = random.Random(f"{seed}:{name}")
        for example in draw(sources[name], take.get(name, 0), rng):
            for query, text, label in pairs_of(example, negatives, rng):
                rows.append({"source": name, "qid": example.qid, "query": query,
                             "text": text, "label": label})
    return rows


def read_parquet(paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    for path in paths:
        yield from pq.read_table(path).to_pylist()


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as src:
        for line in src:
            if line.strip():
                yield json.loads(line)


def read_pairs(path: Path) -> list[tuple[str, str, float]]:
    return [(row["query"], row["text"], float(row["label"])) for row in read_jsonl(path)]


def parse_take(values: Sequence[str]) -> dict[str, int]:
    take = dict(DEFAULT_TAKE)
    for value in values:
        name, _, n = value.partition("=")
        if name not in DEFAULT_TAKE or not n.isdigit():
            raise ValueError(f"--take expects NAME=N with NAME one of {sorted(DEFAULT_TAKE)}, "
                             f"got {value!r}")
        take[name] = int(n)
    return take


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Turn public QA datasets into selector pairs.")
    parser.add_argument("--hotpot", type=Path, nargs="*", default=[])
    parser.add_argument("--twowiki", type=Path, nargs="*", default=[])
    parser.add_argument("--musique", type=Path)
    parser.add_argument("--quac", type=Path)
    parser.add_argument("--take", action="append", default=[])
    parser.add_argument("--negatives", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    sources: dict[str, list[Example]] = {}
    if args.hotpot:
        sources["hotpot"] = [e for r in read_parquet(args.hotpot)
                             if (e := from_supporting_facts("hotpot", r))]
    if args.twowiki:
        sources["twowiki"] = [e for r in read_parquet(args.twowiki)
                              if (e := from_supporting_facts("twowiki", r))]
    if args.musique:
        sources["musique"] = [e for r in read_jsonl(args.musique) if (e := from_musique(r))]
    if args.quac:
        data = json.loads(args.quac.read_text(encoding="utf-8"))["data"]
        sources["quac"] = [e for article in data for e in from_quac(article)]
    rows = build(sources, parse_take(args.take), negatives=args.negatives, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    for name in sorted(sources):
        mine = [r for r in rows if r["source"] == name]
        print(f"{name}: {len(sources[name])} usable questions, "
              f"{len({r['qid'] for r in mine})} drawn, {len(mine)} pairs, "
              f"{int(sum(r['label'] for r in mine))} gold")
    print(f"{len(rows)} pairs written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
