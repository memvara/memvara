"""What `anchored=True` costs and what it buys, measured on public questions.

    PYTHONPATH=. python3 bench/anchoring.py --download
    PYTHONPATH=. python3 bench/anchoring.py --ingest 3000 --holdout 1500

`search(anchored=True)` keeps only the rows whose subject or object the question names, or
that the graph leg reached by walking out of such a row. It exists so that a question about
something the store was never told returns nothing instead of the nearest row about
somebody else. The Agent Memory Benchmark measures that on six negatives this repository
wrote. This measures it on a thousand it did not.

## Where the negatives come from

Nothing here is authored. 2WikiMultihopQA is loaded as `bench/twowiki.py` loads it, the
questions are split, and only the first half's triples are written to the store. A question
from the held-out half is then a real question, from a public set, whose facts the store
genuinely does not hold — which is what "a question the store was never told the answer to"
means outside a benchmark somebody wrote to have one.

One filter makes that honest. 2Wiki reuses entities heavily, so a held-out question often
names somebody the store knows from another question's triples, and refusing to answer that
would be wrong rather than right. A held-out question counts as a negative only when it
names no entity the store holds at all.

## What the two columns mean

`answer found` is `twowiki._found`'s answer criterion over questions whose facts *are*
stored: the cost of anchoring, in legitimate answers it drops.

`correctly silent` is the share of negatives that return no rows: the buy, in questions it
stops answering from the nearest match.

## The finding, and the condition on it

The shipped configuration is silent on **none** of them. It returns rows for every question
it was never told the answer to, at relevances indistinguishable from a real match.

Anchoring alone is expensive here — it drops around a sixth of the legitimate answers —
because a 2Wiki question names entities that the anchor then has to match exactly. The graph
leg pays most of that back, because a question whose answer sits one hop away reaches it
through the entity the question does name.

**That recovery is a property of this corpus, not of the setting.** 2Wiki loads clean
triples through `remember()`, so its join rate is high and the walk has somewhere to go. On a
store whose claims all hang off one subject — the shape `memory_stats` calls a star, and the
shape a single user's own sentences produce — the walk has nowhere to go and pays nothing
back. Read the join rate before reading these numbers as advice.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evalkit as ek                                          # noqa: E402
import twowiki as tw                                          # noqa: E402
from memvara import HashingEmbedder, Memvara, NullLLM         # noqa: E402

#: The depths `twowiki.py` reports, so the two arms can be read side by side.
DEPTHS = (5, 12, 25)

#: Below this length an entity name matches too much prose to mean anything: "de" and
#: "of" appear in most questions, and a negative filtered on those would filter everything.
MIN_ENTITY_CHARS = 4


def build(samples: Sequence[tw.Sample], *, w_graph: float) -> Memvara:
    """`twowiki.ingest`, with the retrieval configuration a deployment would set.

    Built per configuration rather than mutated between passes: `read_w_graph` is a
    constructor argument, and reaching into the retriever afterwards would measure
    something no deployment can ask for.
    """
    mem = Memvara(llm=NullLLM(), embedder=HashingEmbedder(dim=tw.DIM),
                  tenant="2wiki", user="reader", read_w_graph=w_graph)
    seen: set[tuple[str, str, str]] = set()
    with mem.store.batch():
        for sample in samples:
            for triple in sample.triples:
                if triple in seen:
                    continue
                seen.add(triple)
                mem.remember(*triple, valid_from=tw.WRITTEN_AT, recorded_at=tw.WRITTEN_AT)
    for sample in samples:
        sample.fold_to_store(mem.registry)
    return mem


def negatives(held: Sequence[tw.Sample], stored: Sequence[tw.Sample]) -> list[tw.Sample]:
    """Held-out questions that name nothing the store holds.

    The filter is what makes these negatives rather than merely unanswered questions: a
    held-out question naming an entity the store knows from elsewhere is one the store has
    a reason to answer, and counting a refusal there as a success would flatter the number.
    """
    known = {entity.lower()
             for sample in stored
             for triple in sample.triples
             for entity in (triple[0], triple[2])}
    known = {e for e in known if len(e) >= MIN_ENTITY_CHARS}
    return [s for s in held if not any(e in s.question.lower() for e in known)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true",
                        help="fetch the 2Wiki dev set, as bench/twowiki.py does")
    parser.add_argument("--ingest", type=int, default=3000,
                        help="questions whose triples are written (default: 3000)")
    parser.add_argument("--holdout", type=int, default=1500,
                        help="questions held back to draw negatives from (default: 1500)")
    parser.add_argument("--score", type=int, default=1000,
                        help="stored questions to score the cost on (default: 1000)")
    args = parser.parse_args(argv)

    if args.download:
        ek.fetch(ek.TWOWIKI_DEV)
        return 0
    try:
        path = ek.require(ek.TWOWIKI_DEV)
    except ek.DatasetMissing as exc:
        print(exc, file=sys.stderr)
        return 2

    samples = tw.load(path)
    stored = samples[:args.ingest]
    held = samples[args.ingest:args.ingest + args.holdout]
    unheard = negatives(held, stored)
    scored = stored[:args.score]

    print(f"  {len(stored):,} questions ingested, {len(held):,} held out")
    print(f"  {len(scored):,} answerable questions scored, "
          f"{len(unheard):,} of the held-out questions name nothing the store holds\n")
    if not unheard:
        print("  no negatives survived the filter; nothing to measure")
        return 1

    print(f"{'configuration':24}{'k':>4}{'answer found':>15}{'correctly silent':>19}")
    for name, w_graph, anchored in (("shipped", 0.0, False),
                                    ("anchored", 0.0, True),
                                    ("anchored + graph leg", 1.0, True)):
        mem = build(stored, w_graph=w_graph)
        try:
            for k in DEPTHS:
                found = sum(tw._found(mem.search(s.question, k=k, anchored=anchored), s)[0]
                            for s in scored)
                silent = sum(1 for s in unheard
                             if not mem.search(s.question, k=k, anchored=anchored))
                print(f"{name:24}{k:>4}{found / len(scored) * 100:>14.1f}%"
                      f"{silent / len(unheard) * 100:>18.1f}%")
        finally:
            mem.close()
    return 0


if __name__ == "__main__":                                    # pragma: no cover
    raise SystemExit(main())
