"""How long each read takes when one scope holds a large store.

Run:  PYTHONPATH=. python3 bench/scale.py [--path FILE] [--claims N] [--questions N]

`bench/perf.py` measures stores of up to 8,000 claims spread over fifty users, so no
scope it searches holds more than a few hundred rows. This script measures the other
end: one user's scope holding every haystack turn of LongMemEval-S, deduplicated by
session as `bench/longmemeval.py --share-store` does, and N synthetic claims beside them.
Each store read that a search runs is timed on its own, and then `Memvara.search()` as a
whole, so a change to one read shows up in its own row.

Everything is written straight through `SQLiteStore`, with the `HashingEmbedder` that
`evalkit.build_embedder("hashing")` configures. No model runs, and nothing is extracted
from the turns, so building the store takes minutes rather than the hours the ingest
pipeline would. The episode reads and `search()` are timed over the first `--questions`
LongMemEval-S questions. The claim reads are timed over five fixed queries that each
match many of the synthetic claims, because the questions match almost none of them. The
lexical legs are handed what `HybridRetriever` hands them, the query reduced to its
content words by `retrieve.analyze`, because a question's stopwords match most turns.

Without `--path` the store is built in a temporary directory and deleted afterwards. With
it, the store is built at that path, or timed as it is if the file already exists, so two
builds can be timed on one store. Time the older build first, or on a copy: opening a
store with a newer build can add an index that an older build then uses.

Needs the `s` file (`python3 bench/longmemeval.py --download --dataset s`).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evalkit as ek  # noqa: E402
import longmemeval as lme  # noqa: E402

from memvara import Memvara  # noqa: E402
from memvara.embed import HashingEmbedder  # noqa: E402
from memvara.llm import NullLLM  # noqa: E402
from memvara.retrieve.analyze import analyze  # noqa: E402
from memvara.select import PLAIN_READ  # noqa: E402
from memvara.store import SQLiteStore  # noqa: E402
from memvara.types import Claim, Episode, Scope  # noqa: E402

SCOPE = Scope("default", "shared")
CITIES = ["Berlin", "Lisbon", "Osaka", "Nairobi", "Lima", "Oslo", "Cairo", "Perth"]
CLAIM_QUERIES = ["where does the user live", "Berlin project notes", "what about Osaka",
                 "project 42 status", "Lisbon"]
# What a search at k=12 asks each leg for, with the default candidate multiplier.
LEG_LIMIT = 60
CHUNK = 2_000


def build(path: str, items: list[Any], n_claims: int, embedder: HashingEmbedder) -> None:
    """Write every distinct haystack session's turns, then `n_claims` claims."""
    store = SQLiteStore(path)
    seen: set[str] = set()
    turns: list[Episode] = []
    for item in items:
        for i, session in enumerate(item.sessions):
            label = lme.session_label(item.qid, item.session_ids, i)
            if label in seen:
                continue
            seen.add(label)
            turns += [Episode(content=t.text, scope=SCOPE, role=t.role, ts=t.ts)
                      for t in session]
    claims = [Claim(subject="user", predicate=f"pred_{i % 400}",
                    object=f"{CITIES[i % 8]} note {i} about project {i % 97}", scope=SCOPE)
              for i in range(n_claims)]
    for start in range(0, len(turns), CHUNK):
        chunk = turns[start:start + CHUNK]
        vectors = embedder.encode([ep.content for ep in chunk])
        with store.batch():
            for ep, vec in zip(chunk, vectors):
                store.add_episode(ep)
                store.set_episode_embedding(ep.id, vec)
    for start in range(0, len(claims), CHUNK):
        chunk = claims[start:start + CHUNK]
        vectors = embedder.encode([c.text for c in chunk])
        with store.batch():
            for c, vec in zip(chunk, vectors):
                store.put_claim(c)
                store.set_embedding(c.id, vec)
    store.close()


def timed(calls: list[Callable[[], Any]]) -> tuple[float, float]:
    """Median and 95th percentile, in milliseconds, one sample per call."""
    samples = []
    for call in calls:
        t0 = time.perf_counter()
        call()
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    return statistics.median(samples), samples[min(len(samples) - 1,
                                                   int(len(samples) * 0.95))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", default="",
                        help="build the store here and keep it, or time the one already here")
    parser.add_argument("--claims", type=int, default=100_000,
                        help="synthetic claims beside the turns (default 100,000)")
    parser.add_argument("--questions", type=int, default=50,
                        help="LongMemEval-S questions to time the episode reads over")
    parser.add_argument("--reps", type=int, default=20,
                        help="samples for the reads that take no query")
    parser.add_argument("--limit", type=int, default=0,
                        help="build from the first N questions' haystacks only")
    args = parser.parse_args()

    items = lme.load(ek.require(ek.LME_S), limit=args.limit)
    questions = [item.question for item in items if not item.is_abstention]
    questions = questions[:args.questions]
    embedder = HashingEmbedder(dim=ek.BASELINE_EMBED_DIM)

    with tempfile.TemporaryDirectory() as tmp:
        path = args.path or str(Path(tmp) / "scale.db")
        built = None
        if not Path(path).exists():
            t0 = time.perf_counter()
            build(path, items, args.claims, embedder)
            built = time.perf_counter() - t0
        rows, (n_turns, n_claims) = measure(path, questions, embedder, args.reps)

    source = f"built in {built:.0f} s" if built is not None else f"read from {path}"
    print(f"\n  One scope: {n_turns:,} LongMemEval-S turns and {n_claims:,} claims, "
          f"{source}\n")
    print(f"  {'read':<32} {'median':>9} {'p95':>9}")
    for name, (median, p95) in rows:
        print(f"  {name:<32} {median:>6.1f} ms {p95:>6.1f} ms")
    print()
    return 0


def measure(path: str, questions: list[str], embedder: HashingEmbedder,
            reps: int) -> tuple[list[tuple[str, tuple[float, float]]], tuple[int, int]]:
    """Each read's median and p95 in milliseconds, and the store's turn and claim counts."""
    mem = Memvara(path, embedder=embedder, llm=NullLLM(), user=SCOPE.user)
    store = mem.store
    scopes = SCOPE.ancestors()
    # Warm the vector index and the page cache, so the first timed call is not the one
    # that maps the matrix file.
    mem.search(questions[0], k=12, include_episodes=True, **PLAIN_READ)

    claim_terms = [analyze(q).text for q in CLAIM_QUERIES]
    question_terms = [r.text for r in map(analyze, questions) if not r.abstains]
    claim_vecs = embedder.encode(CLAIM_QUERIES)
    question_vecs = embedder.encode(questions)
    rows = [
        ("candidate_ids", timed(
            [lambda: store.candidate_ids(scopes)] * reps)),
        ("episode_candidate_ids", timed(
            [lambda: store.episode_candidate_ids(scopes)] * reps)),
        ("lexical_search", timed(
            [lambda q=q: store.lexical_search(q, scopes, LEG_LIMIT)
             for q in claim_terms] * 4)),
        ("lexical_search_episodes", timed(
            [lambda q=q: store.lexical_search_episodes(q, scopes, LEG_LIMIT)
             for q in question_terms])),
        ("vector_search", timed(
            [lambda v=v: store.vector_search(v, scopes, LEG_LIMIT)
             for v in claim_vecs] * 4)),
        # The same read with the scope's cached claim list dropped first, as every
        # commit drops it: what the first search after a write pays.
        ("  after a write", timed(
            [lambda v=v: (store._changed(), store.vector_search(v, scopes, LEG_LIMIT))
             for v in claim_vecs] * 4)),
        ("vector_search_episodes", timed(
            [lambda v=v: store.vector_search_episodes(v, scopes, LEG_LIMIT)
             for v in question_vecs])),
        # The same read with the scope's cached turn list dropped first, as every commit
        # drops it: what the first search after a write pays.
        ("  after a write", timed(
            [lambda v=v: (store._changed(),
                          store.vector_search_episodes(v, scopes, LEG_LIMIT))
             for v in question_vecs])),
        ("search(k=12, include_episodes)", timed(
            [lambda q=q: mem.search(q, k=12, include_episodes=True, **PLAIN_READ)
             for q in questions])),
        ("  after a write", timed(
            [lambda q=q: (store._changed(),
                          mem.search(q, k=12, include_episodes=True, **PLAIN_READ))
             for q in questions])),
    ]
    counts = store.stats()
    mem.close()
    return rows, (counts["episodes"], counts["claims"])


if __name__ == "__main__":  # pragma: no cover - exercised by hand
    sys.exit(main())
