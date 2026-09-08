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
stored. It is the cost of anchoring: the legitimate answers the filter drops.

`correctly silent` is the share of negatives that return no rows. It is what anchoring buys:
the questions the store stops answering from the nearest row it happens to hold.

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
from memvara import Memvara                                   # noqa: E402
from memvara.entities import entity_key, key_words            # noqa: E402
from memvara.retrieve.anchor import query_tokens              # noqa: E402
from memvara.retrieve.hybrid import HybridRetriever           # noqa: E402

#: The depths `twowiki.py` reports, so the two arms can be read side by side.
DEPTHS = (5, 12, 25)

def reader(mem: Memvara, *, w_graph: float) -> HybridRetriever:
    """A retriever over the ingested store, differing only in the graph weight.

    One store and two readers rather than one store per arm, which is how
    `bench/twowiki.py` measures its own arms: `w_graph` is a retriever argument, so a
    second ingest would re-pay the corpus to change a number the retriever holds. The
    traverser is borrowed from `mem` so the walk is bounded exactly as `neighborhood()`'s
    is, rather than being a differently-configured copy of it.
    """
    return HybridRetriever(mem.store, mem.embedder, mem.registry, w_graph=w_graph,
                           graph_depth=2, traverser=mem.traverser,
                           # `Memvara` always wires this into its own reader, and it is
                           # what lets an anchor widen onto a learned alias. Inert while
                           # the ingest runs on `NullLLM`, which never resolves two
                           # spellings into one entity, and wrong to omit the moment a
                           # real backend is put behind this script.
                           entities=mem.writer.reconciler.entities)


def negatives(held: Sequence[tw.Sample], stored: Sequence[tw.Sample]) -> list[tw.Sample]:
    """Held-out questions that name nothing the store holds.

    The filter is what makes these negatives rather than merely unanswered questions: a
    held-out question naming an entity the store knows from elsewhere is one the store has
    a reason to answer, and counting a refusal there as a success would flatter the number.

    **It decides "names" the way the thing being measured decides it**, which is the whole
    reason this function is not two lines of substring matching. `anchor.anchor_of` folds a
    stored entity to a key, splits the key into words, and asks whether the question's own
    folded tokens contain every one of them. A substring test agrees with that on neither
    side: it matches "Mark" inside "Denmark", and it misses "Atlas Project" for a key stored
    as "project atlas". Either disagreement puts questions in this set that the filter under
    test would have anchored, which moves the column this script exists to report. Measured
    on the shipped split: the substring version admitted 1,018 negatives and the folded one
    admits 332, and the extra 686 were questions the store could answer.

    One branch of `anchor_of` is deliberately not mirrored. It also anchors a claim whose
    subject is the self subject, `user`, when the question uses a first-person pronoun, with
    no word overlap required. No 2Wiki entity folds to `user`, so replicating it here would
    be code for a case this corpus cannot produce.
    """
    known = {parts
             for sample in stored
             for triple in sample.triples
             for entity in (triple[0], triple[2])
             if (parts := tuple(key_words(entity_key(entity))))}
    kept = []
    for sample in held:
        # Folded once per question rather than once per entity: `known` runs to thousands,
        # and a true negative is exactly the case that cannot short-circuit.
        tokens = query_tokens(sample.question)
        if not any(tokens.issuperset(parts) for parts in known):
            kept.append(sample)
    return kept


def main(argv: Sequence[str] | None = None) -> int:
    # The first line of the docstring, or nothing at all: `python -OO` strips docstrings,
    # so both the attribute and the first element have to be allowed to be absent. A
    # benchmark that cannot print its own usage under an optimised interpreter would be a
    # poor trade for one line of help text.
    parser = argparse.ArgumentParser(
        description=next(iter((__doc__ or "").splitlines()), None))
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
    if not scored:
        # Guarded beside `unheard` rather than left to divide by zero: --score 0 and
        # --ingest 0 both reach here with negatives still to report, so the run would
        # otherwise end in a traceback after printing the counts that suggest it worked.
        print("  no answerable questions to score; nothing to measure")
        return 1

    mem = tw.ingest(stored)
    rows: list[tuple[str, int, str, str]] = []
    try:
        scope = mem.default_scope
        plain, walked = reader(mem, w_graph=0.0), reader(mem, w_graph=1.0)
        for name, rdr, anchored in (("shipped", plain, False),
                                    ("anchored", plain, True),
                                    ("anchored + graph leg", walked, True)):
            for k in DEPTHS:
                # `now=tw.NOW` for the reason `bench/twowiki.py` pins it: retrieval decays
                # a claim's score from the instant it is asked, so an unpinned run scores
                # every question differently on every pass and a re-run differs from the
                # published table with no code change behind it.
                found = sum(tw._found(rdr.search(s.question, scope, k=k, now=tw.NOW,
                                                 anchored=anchored), s)[0]
                            for s in scored)
                silent = sum(1 for s in unheard
                             if not rdr.search(s.question, scope, k=k, now=tw.NOW,
                                               anchored=anchored))
                rows.append((name, k, f"{found / len(scored) * 100:.1f}%",
                             f"{silent / len(unheard) * 100:.1f}%"))
    finally:
        mem.close()
    print(ek.render_table(["configuration", "k", "answer found", "correctly silent"],
                          rows))
    return 0


if __name__ == "__main__":                                    # pragma: no cover
    raise SystemExit(main())
