"""Measure the graph leg on a public dataset, with the extractor taken out of the loop.

    PYTHONPATH=. python3 bench/twowiki.py --download        # 56 MB, once
    PYTHONPATH=. python3 bench/twowiki.py --limit 2000
    PYTHONPATH=. python3 bench/twowiki.py                   # all 12,576

**This exists because the other two public benchmarks cannot see the graph leg.** The leg
walks claims; LOCOMO and LongMemEval are prose, and the offline extractor turns 5,882
LOCOMO turns into **zero** claims. A retrieval number from those runs measures extraction
and reports it as retrieval. Until now the only instrument that could see the leg was
`bench/multihop.py`, which this repository wrote — an illustration of a mechanism, not
evidence for it.

2WikiMultihopQA ships its evidence as `[subject, relation, object]` triples from Wikidata.
That is `Claim(subject, predicate, object)` with a rename, so the triples load through
`remember()` and no extractor runs. Two questions that were tangled come apart:

* *does the graph leg retrieve chained facts* — this file, on public data;
* *does ingestion extract claims from conversation* — still open, still the bottleneck.

**Say which one you are quoting.** A number here is retrieval given claims. It is not
end-to-end memory quality and does not become that by being quoted next to one.

**One shared store, not one per question.** All ~31k triples from all questions go into a
single scope, so every query is answered against everyone else's facts. That is the harder
setting and the honest one for a memory: the dataset's own baselines retrieve from a
per-question candidate set, which is a reading comprehension task rather than a recall
task. Expect lower numbers here than the 2Wiki leaderboard reports, and do not compare
them.

**Two of the four question types are not transitive**, and the split below reports them
apart rather than averaging them away. `compositional` and `inference` chain one fact into
the next — "who is the mother of the director of X" is `director` then `mother`, and that
is what a walk is for. `comparison` and `bridge_comparison` ask which of two independent
entities came first; the evidence has two ends and no join between them, so the graph leg
has nothing to walk and should not be expected to help. A benchmark that hid that
distinction inside an average would credit the leg for questions it cannot answer.

Contamination is a smaller problem here than `evalkit`'s note describes, structurally
rather than by luck: scoring is R@k against gold evidence under `NullLLM`, so there is no
reader that could have memorised an answer. What a contaminated *reader* would inflate is
end-to-end accuracy, which this file does not measure.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bench.evalkit as ek                                          # noqa: E402
from memvara import Memvara, NullLLM                                # noqa: E402
from memvara.embed import HashingEmbedder                           # noqa: E402
from memvara.retrieve.hybrid import HybridRetriever                 # noqa: E402
from memvara.schema import PredicateRegistry                        # noqa: E402

#: The transitive types. A walk can only help where the evidence has a join in it.
CHAINED = ("compositional", "inference")

#: Hash embedder rather than a model, for the same reason `multihop.py` uses one: this
#: measures whether the *walk* reaches a row the lookup legs missed, and a real embedder
#: would make the vector leg strong enough to blur which leg did the finding. It also
#: keeps the run offline and deterministic.
DIM = 64


class Sample:
    """One 2Wiki question, its answer, and the evidence chain behind it."""

    __slots__ = ("qid", "question", "answer", "kind", "triples")

    def __init__(self, raw: dict) -> None:
        self.qid: str = raw["_id"]
        self.question: str = raw["question"]
        self.answer: str = raw["answer"]
        self.kind: str = raw["type"]
        # Wikidata relations arrive spaced ("date of birth"); the store keys predicates by
        # name, so they are underscored here. **Not folded yet** — see `fold_to_store`.
        self.triples: list[tuple[str, str, str]] = [
            (s, p.strip().replace(" ", "_"), o) for s, p, o in raw["evidences"]
        ]

    def fold_to_store(self, registry: "PredicateRegistry") -> None:
        """Rewrite the gold predicates the way the store will have written them.

        `place_of_birth` is an alias of `born_in` and `date_of_birth` of `born_on`, so a
        claim written from this evidence is *stored* under the canonical name. Comparing
        the raw gold triple against a returned row therefore never matched for either of
        them — and they are 6,624 of this corpus's evidence triples.

        The effect was one-sided and quiet: `answer` matched on the object alone and kept
        scoring, while `chain` required the predicate too and silently failed. So chain
        recall read far below the truth and the gap between the two columns looked like a
        finding about retrieval — "the walk brings back answers without their evidence" —
        when it was an artifact of this file comparing two spellings of one predicate.
        """
        self.triples = [(s, registry.normalize(p), o) for s, p, o in self.triples]

    @property
    def chained(self) -> bool:
        return self.kind in CHAINED


def load(path: Path, limit: int | None = None) -> list[Sample]:
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    samples = [Sample(r) for r in raw if r.get("evidences")]
    return samples[:limit] if limit else samples


def ingest(samples: Sequence[Sample]) -> Memvara:
    """Every triple from every question, deduplicated, in one scope.

    Deduplicated because 2Wiki reuses entities heavily — `director` alone appears 7,592
    times — and writing the same triple twice would spend the run's time in reconciliation
    proving that a fact equals itself. What survives is one claim per distinct
    `(subject, relation, object)`, which is what the store would hold anyway.
    """
    mem = Memvara(llm=NullLLM(), embedder=HashingEmbedder(dim=DIM),
                  tenant="2wiki", user="reader")
    seen: set[tuple[str, str, str]] = set()
    written = 0
    started = time.perf_counter()
    with mem.store.batch():
        for sample in samples:
            for triple in sample.triples:
                if triple in seen:
                    continue
                seen.add(triple)
                mem.remember(*triple, valid_from=WRITTEN_AT,
                             recorded_at=WRITTEN_AT)
                written += 1
                if written % 5_000 == 0:
                    rate = written / (time.perf_counter() - started)
                    print(f"    {written:,} claims  ({rate:,.0f}/s)", flush=True)
    took = time.perf_counter() - started
    # Fold the gold predicates the way the store just wrote them, or `chain` compares
    # `place_of_birth` against the `born_in` it was stored as and never matches.
    for sample in samples:
        sample.fold_to_store(mem.registry)
    print(f"  ingested {written:,} distinct claims from {len(samples):,} questions "
          f"in {took:,.1f}s")
    return mem


#: When this corpus is written, and when it is read. Both fixed, and the gap between
#: them is the point.
#:
#: Retrieval decays a claim's score from the moment it is asked, so an unpinned run
#: scores every question differently on every pass — measured at 3,000 of 3,000, in the
#: low-order digits, which is enough to flip a near-tie at the `k` boundary and move a
#: published figure by a tenth of a point with no code change behind it.
#:
#: Pinning the read alone is not enough and is actively wrong: `remember()` stamps a
#: claim with the wall clock, so a read pinned to a distant instant makes every claim
#: years old, `recency_factor` collapses to zero for all of them, and the quality
#: multiplier loses the spread the shipped configuration has. Measured: reading at 2030
#: against claims written today moved ungated `chained` from 76.7% to 54.5%. That is not
#: a benchmark of the same thing.
#:
#: So both ends are pinned and the gap is one minute, which is the regime an unpinned
#: run is already in — a corpus ingested seconds ago and queried over the following
#: minute or two. 2WikiMultihopQA carries no timestamps of its own, so nothing here is
#: being overridden; the wall clock was arbitrary and this is arbitrary and repeatable.
WRITTEN_AT = datetime(2026, 1, 1, tzinfo=timezone.utc)
NOW = WRITTEN_AT + timedelta(minutes=1)


def _reader(mem: Memvara, *, w_graph: float, gated: bool) -> HybridRetriever:
    """A retriever differing from the shipped one in at most two arguments.

    Borrowing `mem.traverser` rather than building one, so the walk here is bounded
    exactly as `mem.neighborhood()`'s is and the column is about the leg rather than
    about a differently-configured copy of it.
    """
    return HybridRetriever(mem.store, mem.embedder, mem.registry, w_graph=w_graph,
                           graph_depth=2, traverser=mem.traverser,
                           intent_weighting=gated)


def _found(results: Iterable, sample: Sample) -> tuple[bool, bool]:
    """Did the answer come back, and did the whole chain come back?

    Two questions, because they fail differently. `answer` is the criterion
    `multihop.py` uses and it is the one a caller cares about — the gold entity is
    somewhere in the rows. `chain` asks whether *every* evidence triple is there, which
    is what makes the answer checkable rather than lucky: a row naming Małgorzata Braunek
    with no "Xawery Żuławski, mother, Małgorzata Braunek" behind it is a coincidence that
    scores the same as a derivation.
    """
    rows = list(results)
    hay = {f"{r.claim.subject}\t{r.claim.predicate}\t{r.claim.object}" for r in rows}
    answer = any(sample.answer.lower() in f"{r.claim.subject} {r.claim.object}".lower()
                 for r in rows)
    chain = all(f"{s}\t{p}\t{o}" in hay for s, p, o in sample.triples)
    return answer, chain


def evaluate(mem: Memvara, samples: Sequence[Sample], *, k: int,
             plain: HybridRetriever, graph: HybridRetriever,
             ungated: HybridRetriever) -> dict[str, Counter]:
    """One pass over the questions, three readers, split by question type.

    **Every arm reads the clock at `NOW`, and that is what makes two runs comparable.**
    Retrieval decays a claim's score from the moment it is asked, so without a pinned
    instant every question scores differently on every pass — measured here at 3,000 of
    3,000, in the low-order digits. That is correct for a live store and useless for a
    benchmark: on a near-tie at the `k` boundary the drift flips which row lands inside
    the cut, and a re-run then differs from the published table by a tenth of a point
    with no code change behind it. An hour was spent attributing one such diff to a
    change that had not caused it.

    `mem.search()` is not used for the `search` arm any more, because it has no way to
    take the instant. `plain` is the same read path with the same defaults — `w_graph`
    is 0.0 there, which is what `mem.search()` was — reached through the object that can
    be pinned.
    """
    hits: dict[str, Counter] = {
        arm: Counter() for arm in ("search", "+graph", "+graph!", "n")
    }
    for sample in samples:
        buckets = (sample.kind, "chained" if sample.chained else "flat", "all")
        for bucket in buckets:
            hits["n"][bucket] += 1
        for arm, rows in (
            ("search", plain.search(sample.question, mem.default_scope, k=k, now=NOW)),
            ("+graph", graph.search(sample.question, mem.default_scope, k=k, now=NOW)),
            ("+graph!",
             ungated.search(sample.question, mem.default_scope, k=k, now=NOW)),
        ):
            answer, chain = _found(rows, sample)
            for bucket in buckets:
                hits[arm][bucket] += answer
                hits[arm][f"{bucket}\tchain"] += chain
    return hits


def report(hits: dict[str, Counter], k: int, buckets: Sequence[str]) -> None:
    n = hits["n"]
    print(f"\n  k={k}   answer found in the returned rows / full evidence chain returned")
    print(f"  {'set':<18} {'n':>6} {'search':>14} {'+graph':>14} {'+graph!':>14}")
    for bucket in buckets:
        total = n[bucket]
        if not total:
            continue
        cells = []
        for arm in ("search", "+graph", "+graph!"):
            ans = 100.0 * hits[arm][bucket] / total
            chain = 100.0 * hits[arm][f"{bucket}\tchain"] / total
            cells.append(f"{ans:>6.1f}% /{chain:>5.1f}%")
        print(f"  {bucket:<18} {total:>6,} " + " ".join(f"{c:>14}" for c in cells))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Graph-leg retrieval on 2WikiMultihopQA, one shared store.")
    parser.add_argument("--download", action="store_true",
                        help=f"fetch the dev set ({ek.TWOWIKI_DEV.size_mb:.0f} MB) and exit")
    parser.add_argument("--limit", type=int, default=None,
                        help="use the first N questions (default: all 12,576)")
    parser.add_argument("--k", default="5,12,25",
                        help="comma-separated cut-offs (default: 5,12,25)")
    args = parser.parse_args(argv)

    if args.download:
        ek.fetch(ek.TWOWIKI_DEV)
        return 0

    try:
        path = ek.require(ek.TWOWIKI_DEV)
    except ek.DatasetMissing as exc:
        print(exc, file=sys.stderr)
        return 2

    samples = load(path, args.limit)
    kinds = Counter(s.kind for s in samples)
    print(f"  {len(samples):,} questions: "
          + ", ".join(f"{kind} {count:,}" for kind, count in kinds.most_common()))
    print(f"  chained (a walk can help): "
          f"{sum(1 for s in samples if s.chained):,}    "
          f"flat (it cannot): {sum(1 for s in samples if not s.chained):,}")

    mem = ingest(samples)
    try:
        plain = _reader(mem, w_graph=0.0, gated=True)
        graph = _reader(mem, w_graph=1.0, gated=True)
        ungated = _reader(mem, w_graph=1.0, gated=False)
        buckets = ["all", "chained", "flat", *sorted(kinds)]
        for k in (int(x) for x in args.k.split(",")):
            report(evaluate(mem, samples, k=k, plain=plain, graph=graph,
                            ungated=ungated), k, buckets)
    finally:
        mem.close()

    print("\n  `search` is the shipped read path, `+graph` the same with the leg turned "
          "on,\n   `+graph!` the same again with intent weighting off. Each cell is "
          "\"answer found /\n   whole evidence chain found\" — the second is what makes "
          "the first checkable.\n\n   One shared store: every question is answered "
          "against every other question's facts.\n   The 2Wiki leaderboard retrieves "
          "from a per-question candidate set, so these are not\n   comparable numbers "
          "and are not meant to be.\n\n   `flat` is `comparison` and `bridge_comparison`, "
          "whose evidence has two ends and no\n   join between them. The leg has nothing "
          "to walk there; it is reported to show that,\n   not because a walk was "
          "expected to help.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
