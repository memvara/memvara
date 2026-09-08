"""Which of a corpus's relations a vocabulary declares, and what that costs the graph.

    PYTHONPATH=. python3 bench/predicate_audit.py --dataset twowiki_dev
    PYTHONPATH=. python3 bench/predicate_audit.py --dataset twowiki_dev --packs engineering

**Why this exists.** `docs/SUBJECT-CONVENTIONS.md` decision 3 makes an object's kind a
property of its predicate: a predicate whose `object_type` is undeclared takes *values*, and
a value carries no graph edge. That default is deliberate — a value wrongly treated as an
entity creates false joins that degrade retrieval invisibly, while an entity wrongly treated
as a value only costs a join a later declaration recovers.

The consequence is that connectivity becomes exactly as large as the declared vocabulary and
not one edge larger, and *that* is what this script measures. `BUILTIN_PREDICATES` is a
23-predicate personal-assistant vocabulary; 2WikiMultihopQA's evidence is Wikidata relations
like `director` and `mother`. None of them are declared, so once the classification rule is
wired into retrieval, every object in that corpus becomes a value and the corpus goes from
its measured 40.6% joinable to zero. The primary regression test for the graph leg would
report that the graph leg had stopped working — correctly, and for reasons having nothing to
do with whether the design is any good.

So a predicate pack per benchmark corpus is a prerequisite rather than an enhancement, and
this script is what tells you which relations the pack has to cover and in what order of
importance. It is also the audit to re-run after writing one, where a clean result is
`undeclared: 0`.

**It measures rather than predicts.** The classification rule is not wired into retrieval
yet, so today's joinable count is unaffected by any of this. The `projected` figure below is
what the count *becomes* once it is, computed from the same claims by the same rule, so the
two numbers can be compared before and after that change rather than argued about.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bench.evalkit as ek                                          # noqa: E402
from memvara.schema import (BUILTIN_PREDICATES, PredicateRegistry,  # noqa: E402
                            load_all_specs)


def relation_counts(dataset_key: str, limit: int | None) -> Counter[str]:
    """Every relation in the corpus's evidence, by how often it is asserted.

    Frequency is the ordering a pack should be written in: declaring the head of this
    distribution buys most of the connectivity, and the tail is where a missing declaration
    costs one edge rather than thousands.
    """
    if dataset_key != "twowiki_dev":
        raise SystemExit(
            f"{dataset_key!r} has no triple-shaped evidence to audit. Only 2WikiMultihopQA "
            "ships its evidence as [subject, relation, object]; LOCOMO and LongMemEval are "
            "prose, and their relations do not exist until an extractor invents them — "
            "which is a different measurement, and bench/extract_cost.py is the one for it.")

    import bench.twowiki as tw

    path = ek.require(ek.DATASETS[dataset_key])
    counts: Counter[str] = Counter()
    for sample in tw.load(path, limit):
        for _subject, relation, _object in sample.triples:
            counts[relation] += 1
    return counts


def audit(counts: Counter[str], registry: PredicateRegistry) -> dict[str, object]:
    """Split the corpus's relations three ways, because two of them are not the same.

    A relation declared to take values and a relation nobody declared at all both end up
    carrying no edge, so a two-way split reports them together and hides the only one that
    is a gap. `born_on` taking a date is the vocabulary working; `has_part` falling through
    to the default is a relation somebody has not got to yet.
    """
    entities: list[tuple[str, int]] = []
    values: list[tuple[str, int]] = []
    undeclared: list[tuple[str, int]] = []
    for relation, n in counts.most_common():
        # Normalized, because that is the name a claim is stored under: `bench/twowiki.py`
        # folds every relation through the registry on the way in, and three of this
        # corpus's spellings are aliases of builtins. Auditing the raw spelling would
        # report a declaration that is working as missing.
        name = registry.normalize(relation)
        if not registry.spec_is_declared(name):
            undeclared.append((relation, n))
        elif registry.spec(name).objects_are_entities:
            entities.append((relation, n))
        else:
            values.append((relation, n))

    asserted = sum(counts.values())
    entity_valued = sum(n for _, n in entities)
    return {
        "relations": len(counts),
        "asserted": asserted,
        "declared": entities,
        "values": values,
        "undeclared": undeclared,
        "entity_valued": entity_valued,
        # Claims that could still carry an edge once the rule is wired in. Not the join
        # rate: a claim with an entity-valued object joins only if that object is also some
        # other claim's subject, which this cannot know without building the store.
        "projected_share": (entity_valued / asserted) if asserted else 0.0,
    }


def render(report: dict[str, object], top: int) -> None:
    undeclared = report["undeclared"]                     # type: ignore[index]
    declared = report["declared"]                         # type: ignore[index]
    values = report["values"]                             # type: ignore[index]
    print(f"  relations: {report['relations']:,}   "
          f"triples asserted: {report['asserted']:,}")
    print(f"  declared entity-valued: {len(declared):,} relations, "
          f"{report['entity_valued']:,} triples")
    print(f"  declared value-valued:  {len(values):,} relations, "
          f"{sum(n for _, n in values):,} triples   (deliberate: no edge)")
    print(f"  undeclared:             {len(undeclared):,} relations, "
          f"{sum(n for _, n in undeclared):,} triples   (the gap)")
    print(f"  share of triples that can carry an edge: "
          f"{report['projected_share']:.1%}")
    if undeclared:
        print(f"\n  the {min(top, len(undeclared))} costliest undeclared relations:")
        for relation, n in undeclared[:top]:
            print(f"    {n:>7,}  {relation}")
    else:
        print("\n  no gap: every relation in this corpus carries a declaration.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Which of a corpus's relations a vocabulary declares.")
    parser.add_argument("--dataset", default="twowiki_dev",
                        help="dataset key (default: twowiki_dev)")
    parser.add_argument("--packs", default="",
                        help="MEMVARA_PREDICATES-style list: pack names, paths, or both")
    parser.add_argument("--limit", type=int, default=None,
                        help="use the first N questions (default: all)")
    parser.add_argument("--top", type=int, default=25,
                        help="how many undeclared relations to list (default: 25)")
    args = parser.parse_args(argv)

    try:
        counts = relation_counts(args.dataset, args.limit)
    except ek.DatasetMissing as exc:
        print(exc, file=sys.stderr)
        return 2

    # Builtins first, pack appended, exactly as `server/config._registry` builds it. A
    # registry holding the pack alone would lose the builtin aliases that fold three of
    # this corpus's relations onto canonical names, and would report two working
    # declarations as missing.
    specs = load_all_specs(args.packs) if args.packs.strip() else ()
    registry = PredicateRegistry(BUILTIN_PREDICATES + specs)
    print(f"  vocabulary: {args.packs or 'builtins only'}")
    render(audit(counts, registry), args.top)
    return 0


if __name__ == "__main__":                                    # pragma: no cover
    raise SystemExit(main())
