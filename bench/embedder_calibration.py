"""Where each cosine threshold in `embed/calibration.py` sits in each embedding space.

Run:  PYTHONPATH=. python3 bench/embedder_calibration.py [--model ID ...]

Needs sentence-transformers (`pip install 'memvara[local-embed]'`) and the LongMemEval
`s` file (`python3 bench/longmemeval.py --download --dataset s`), which supplies turns
that the invented values below have nothing to do with.

Two thresholds read a cosine, and a cosine is not portable between models:

* **The grounding rescue** (`write/pipeline.py`) keeps a model-proposed claim that shares
  no word with its source when its best 1,200-character chunk cosine reaches
  `grounding_rescue`. It should keep paraphrases and refuse inventions. The pairs: 15
  paraphrases, each of a fact its source turn states in other words, and 15 invented
  values of the kind a small model emits on a turn that states no such fact, each
  against 20 turns it has nothing to do with, 300 pairs.
* **The duplicate merge** (`consolidate/merge.py`) folds two live claims in one slot
  when their cosine reaches `merge`. It should fold restatements and never two different
  values. The pairs: 14 claims that differ in one value, often one digit, and 8
  restatements of one claim.

The original eval behind 0.40, 33 inventions from two 4B-class models over real turns,
is not in this repository. These pairs are a reconstruction written for this script, so
a threshold taken from them should be read as that.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import evalkit as ek  # noqa: E402
import longmemeval as lme  # noqa: E402

CHUNK = 1200
MODELS = ("sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5")

# (source turn, a fact it states, in words it does not use)
PARAPHRASES = [
    ("I prefer short answers, please don't write essays.", "keep replies brief"),
    ("My flight to Lisbon leaves on Friday morning.",
     "travelling to Portugal at the end of the week"),
    ("I've been vegetarian for about six years now.", "does not eat meat"),
    ("We adopted a greyhound from the rescue last spring.", "has a retired racing dog"),
    ("I'm allergic to penicillin, found out the hard way.", "cannot take that antibiotic"),
    ("Our team moved the standup to 9:30 because of the new hires in Berlin.",
     "daily sync starts half past nine"),
    ("I usually cycle to the office unless it's raining.", "commutes by bike"),
    ("I just finished my PhD in marine biology.", "holds a doctorate studying ocean life"),
    ("My daughter starts kindergarten in September.", "child begins school this autumn"),
    ("I can't stand cilantro, it tastes like soap to me.", "dislikes coriander"),
    ("We're renting a flat near the canal in Amsterdam.", "lives in a Dutch city by the water"),
    ("I play bass in a jazz trio on weekends.", "performs music as a hobby"),
    ("The deploy failed because the database migration timed out.",
     "release broke on a slow schema change"),
    ("I switched from an iPhone to a Pixel last month.", "now uses an Android phone"),
    ("I get migraines when I skip coffee.", "headaches without caffeine"),
]

# Placeholder values a small model emits on a turn that states no such fact.
INVENTIONS = ["Acme", "Acme Corp", "unknown", "software engineer at Google",
              "a golden retriever named Max", "pollen allergy", "lives in New York",
              "likes pizza", "John Smith", "enjoys hiking and photography",
              "prefers email communication", "works remotely", "has two children",
              "favorite color is blue", "speaks Spanish"]

# One slot, two different values: must never merge.
DIFFERENT = [
    ("server runs on port 8080", "server runs on port 8081"),
    ("app version is 1.2.3", "app version is 1.2.4"),
    ("user's room is 101", "user's room is 102"),
    ("user's phone extension is 4521", "user's phone extension is 4522"),
    ("appointment is on 2023-05-01", "appointment is on 2023-05-02"),
    ("dose is 5 mg", "dose is 10 mg"),
    ("budget is $50", "budget is $60"),
    ("user's locker code is 3481", "user's locker code is 3418"),
    ("meeting is in room B12", "meeting is in room B21"),
    ("user's flight is LH 400", "user's flight is LH 401"),
    ("project deadline is March 3", "project deadline is March 13"),
    ("user lives at 12 Main Street", "user lives at 21 Main Street"),
    ("the API key prefix is sk-ab", "the API key prefix is sk-ba"),
    ("user's employee id is E-10293", "user's employee id is E-10239"),
]

# One claim, written twice: should merge.
RESTATED = [
    ("user lives in Berlin", "User lives in Berlin."),
    ("user works at Google", "the user works at Google"),
    ("server runs on port 8080", "Server runs on port 8080"),
    ("user's dog is named Max", "user's dog is called Max"),
    ("user prefers tea", "User prefers tea"),
    ("app version is 1.2.3", "the app version is 1.2.3"),
    ("dose is 5 mg", "Dose is 5 mg."),
    ("user is allergic to peanuts", "user has a peanut allergy"),
]


def unrelated_turns() -> list[str]:
    """Twenty real turns, over 200 characters, that none of the inventions describe."""
    episodes = json.loads((ROOT / "tests/fixtures/phi4_spike/episodes.json").read_text())
    turns = [e["content"] for e in episodes]
    items = lme.load(ek.require(ek.LME_S), limit=3)
    turns += [t.text for item in items for s in item.sessions[:4] for t in s[:3]]
    return [t for t in turns if len(t) > 200][:20]


def measure(model_id: str, sources: list[str]) -> dict[str, np.ndarray]:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

    model = SentenceTransformer(model_id, device="cpu")

    def unit(texts: list[str]) -> np.ndarray:
        return np.asarray(model.encode(texts, normalize_embeddings=True), dtype=np.float32)

    def best_chunk(obj: str, source: str) -> float:
        chunks = [source[i:i + CHUNK] for i in range(0, max(len(source), 1), CHUNK)]
        v = unit([obj] + chunks)
        return float(max(v[1:] @ v[0]))

    def pair(a: str, b: str) -> float:
        v = unit([a, b])
        return float(v[0] @ v[1])

    return {
        "paraphrase": np.array([best_chunk(obj, src) for src, obj in PARAPHRASES]),
        "invention": np.array([best_chunk(obj, src) for obj in INVENTIONS for src in sources]),
        "different": np.array([pair(a, b) for a, b in DIFFERENT]),
        "restated": np.array([pair(a, b) for a, b in RESTATED]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", action="append", default=[],
                        help="a sentence-transformers model id; repeatable")
    args = parser.parse_args()
    sources = unrelated_turns()
    for model_id in args.model or MODELS:
        m = measure(model_id, sources)
        inv, para = m["invention"], m["paraphrase"]
        print(f"\n  {model_id}")
        print(f"    inventions: median {np.median(inv):.3f}, highest {inv.max():.3f}; "
              f"paraphrases: median {np.median(para):.3f}, lowest {para.min():.3f}")
        print(f"    {'rescue at':<12} {'inventions kept':>16} {'paraphrases kept':>17}")
        for t in (0.40, 0.55, 0.60, 0.65):
            print(f"    {t:<12.2f} {np.mean(inv >= t):>15.1%} {np.mean(para >= t):>16.1%}")
        diff, same = m["different"], m["restated"]
        print(f"    different values: highest {diff.max():.5f}")
        print(f"    {'merge at':<12} {'different merged':>17} {'restated merged':>16}")
        for t in (0.97, 0.98, 0.985, 0.99):
            print(f"    {t:<12} {int((diff >= t).sum()):>11} of {len(diff):<3} "
                  f"{int((same >= t).sum()):>10} of {len(same)}")
    print()
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by hand
    sys.exit(main())
