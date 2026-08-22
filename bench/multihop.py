"""Multi-hop retrieval: can the store answer a question no single claim contains?

Run:  PYTHONPATH=. python3 bench/multihop.py [--sizes 1000,10000,100000]

**This is a synthetic, self-authored workload**, in the same category as
`bench/compare.py` and to be read the same way: an illustration of a mechanism, not
evidence of superiority over anything. It exists because no external set measures the
thing. LOCOMO's `multi-hop` category was the obvious candidate and does not fit — its
questions ("What is Caroline's identity?", "What did Caroline research?") are single-fact
lookups whose evidence happens to span one or two turns, not transitive relations over
entities, so a graph walk is not what that 36% row in the README is short of.

## What is measured

Four ways of answering the same question, over one store:

    search              mem.search(question, k)              — the shipped retriever
    search+graph        the same call at w_graph=1.0          — the shipped retriever with
                        the graph leg on, which is the only column here that measures an
                        end-to-end system: the walk is seeded from the head of its own
                        fused list, so nothing hands it the seed entity. Compare it with
                        `linked`, which is the same idea done by the caller.
    search x2           search, then search again on the top hit's object — the agent
                        loop traversal is meant to replace, and the baseline that
                        matters. A single-shot search cannot answer these questions by
                        construction: the claim holding the answer shares no vocabulary
                        with the question, so comparing against it alone would be
                        comparing against nothing.
    traverse            mem.neighborhood(seed, depth, k)     — every relation, ranked
    traverse+min_hops   ... with min_hops set to the answer's distance
    traverse+both       ... and narrowed to the relations the question names. This is a
                        *guided* upper bound, not a result: it is handed the exact
                        relation chain, which a real agent would have to derive.

A question counts as answered when the gold entity is where one of the returned paths
*ends*; for `search`, when it appears in one of the returned claims. Both are "the answer
is in what you were handed", and for traversal the path itself is the witness — the
caller can read every hop and take any of them to `why()`.

`seed` is the entity the question is about. Traversal is given it and `search` is not, so
those numbers are **not a like-for-like comparison** and the traversal columns are an
upper bound on an end-to-end system. `search+graph` is the like-for-like one. `linked` reports the honest version: the seed is
taken from the top hit of the same search the baseline ran, so the gap between it and
`traverse` is entity linking rather than traversal.

## How this benchmark could flatter us, and what stops it

Three deliberate constraints, because the graph and the questions come from one
generator and that is exactly where a benchmark starts measuring itself:

1. **Every question is verified to need the hops it claims.** The harness walks the store
   to depth `hops - 1` with a very large `k` and drops any question whose gold answer is
   already reachable. A "two-hop" question answerable in one hop measures nothing.
2. **Roles use disjoint entity pools and disjoint relations.** Founders never work
   anywhere, office cities are never anyone's home. Without that, a person can reach
   their employer's city through their own `lives_in` by coincidence, and the metric
   counts a right answer reached by a route that has nothing to do with the question.
   The office relation is `headquartered_in` rather than the obvious `located_in`
   because the shipped registry folds `located_in` onto `lives_in` — so a company and a
   person would have shared one predicate and the two routes would have merged again
   one level down.
3. **The store is padded with claims that cannot answer anything.** At `k=25` over a
   400-claim store, a retriever is handed a large fraction of everything and `recall@k`
   stops meaning much.

The first version of this file had none of 1–3, and reported 45.6% for traversal at
`k=25` — a number produced by `k` being smaller than the two-hop frontier, which is a
fact about the harness and not about the store.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, "bench")

from evalkit import mean, percentile                        # noqa: E402

from memvara import HashingEmbedder, Memvara, NullLLM       # noqa: E402
from memvara.retrieve import HybridRetriever                # noqa: E402

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)

FIRST = ["Ada", "Bruno", "Cai", "Dara", "Elif", "Farid", "Gita", "Hugo", "Iris",
         "Jonas", "Kira", "Luc", "Mira", "Nils", "Oona", "Piotr", "Quinn", "Rosa"]
LAST = ["Ahmed", "Bennett", "Costa", "Duarte", "Eriksson", "Fontaine", "Gruber",
        "Haddad", "Ibarra", "Jansen", "Kovac", "Lindqvist", "Moreau", "Novak"]
HOME = ["Lisbon", "Berlin", "Osaka", "Nairobi", "Lima", "Oslo", "Cairo", "Perth"]
OFFICE = ["Tallinn", "Bogota", "Dakar", "Hanoi", "Zagreb", "Quito", "Bergen", "Utrecht"]
SUFFIX = ["Systems", "Labs", "Works", "Dynamics", "Analytics", "Robotics", "Foundry"]


class Corpus:
    """An org graph, with the roles kept in disjoint pools. See constraint 2 above."""

    def __init__(self, staff: int, companies: int) -> None:
        self.staff = [f"{FIRST[i % len(FIRST)]} {LAST[(i // 3) % len(LAST)]}-{i}"
                      for i in range(staff)]
        self.companies = [f"{LAST[i % len(LAST)]} {SUFFIX[i % len(SUFFIX)]}-{i}"
                          for i in range(companies)]
        # Founders exist only as the object of `founded_by`, so a founder is reachable
        # from a person by exactly one route: through the company.
        self.founders = [f"{FIRST[(i * 5) % len(FIRST)]} Halvorsen-f{i}"
                         for i in range(companies)]
        self.office = {c: OFFICE[i % len(OFFICE)] for i, c in enumerate(self.companies)}
        self.founder = dict(zip(self.companies, self.founders))
        self.employer = {p: self.companies[i % len(self.companies)]
                         for i, p in enumerate(self.staff)}
        self.home = {p: HOME[(i * 3) % len(HOME)] for i, p in enumerate(self.staff)}
        # A management chain that crosses companies, so "my manager's employer" is a
        # different company from mine and the three-hop question has one route.
        self.manager = {}
        for i, person in enumerate(self.staff):
            boss = self.staff[(i * 7 + 1) % len(self.staff)]
            if boss != person and self.employer[boss] != self.employer[person]:
                self.manager[person] = boss

    def facts(self) -> list[tuple[str, str, str]]:
        out: list[tuple[str, str, str]] = []
        for company in self.companies:
            # A second spelling of the same employer, so the graph joins only if entity
            # resolution folds them — which is the property traversal is built on.
            out.append((f"{company}, Inc.", "headquartered_in", self.office[company]))
            out.append((company, "founded_by", self.founder[company]))
        for person in self.staff:
            out.append((person, "works_at", self.employer[person]))
            out.append((person, "lives_in", self.home[person]))
            if person in self.manager:
                out.append((person, "reports_to", self.manager[person]))
        return out

    def questions(self) -> list[tuple[str, str, str, int, list[str]]]:
        """(question, seed, gold, hops, the relations the question names)."""
        out = []
        for person in self.staff:
            company = self.employer[person]
            out.append((f"Which city is the company {person} works at based in?",
                        person, self.office[company], 2, ["works_at", "headquartered_in"]))
            out.append((f"Who founded the company that {person} works at?",
                        person, self.founder[company], 2, ["works_at", "founded_by"]))
            boss = self.manager.get(person)
            if boss is not None:
                out.append((f"Which city does the employer of {person}'s manager sit in?",
                            person, self.office[self.employer[boss]], 3,
                            ["reports_to", "works_at", "headquartered_in"]))
        return out


def load(corpus: Corpus, *, padding: int, path: str = ":memory:") -> Memvara:
    mem = Memvara(path, llm=NullLLM(), embedder=HashingEmbedder(dim=256), user="alice")
    with mem.store.batch():
        for subject, predicate, obj in corpus.facts():
            mem.remember(subject, predicate, obj, recorded_at=T0)
        for i in range(padding):
            mem.remember(f"record-{i}", "noted", f"value-{i}", recorded_at=T0)
    return mem


def genuinely_multi_hop(mem: Memvara, questions, limit: int) -> list:
    """Constraint 1: drop any question whose answer is already closer than it claims."""
    kept = []
    for item in questions[:limit]:
        _question, seed, gold, hops, _preds = item
        if hops > 1:
            closer = mem.neighborhood(seed, depth=hops - 1, k=2000, as_of=T0)
            if any(gold in path.labels[1:] for path in closer):
                continue
        kept.append(item)
    return kept


def graph_reader(mem: Memvara, *, w_graph: float = 1.0, depth: int = 2,
                 gated: bool = True) -> HybridRetriever:
    """The shipped retriever over the same store, with the graph leg switched on.

    A second reader rather than a second store: the corpus costs seconds to build at
    100k claims and the two configurations have to see byte-identical data or the column
    is measuring the loader. `Memvara` builds the traverser, and this borrows it, so the
    walk here is bounded exactly as `mem.neighborhood()`'s is.

    `gated` is `intent_weighting`, and it is a column of its own below rather than a
    footnote: the gap between the two columns is what the gate costs on this workload.

    **That gap used to have the wrong explanation, including here.** This docstring said
    the cause was vocabulary — that two of the three families contain no relational word.
    They do not, and it was not the cause: `evaluate` passes `as_of=T0` on every call, so
    `_weights` took its `timed` branch, `classify` was never reached, and
    `Intent.TEMPORAL`'s multipliers set the graph weight to zero. The whole `+graph`
    column measured a configuration in which the leg could not run at all.

    Both halves are fixed. The classifier now counts distinct predicates, so
    "which city is the company X works at based in" reads as a chain without any word
    being added to the relational list; and naming an instant no longer switches the walk
    off. What remains gated is "who founded the company that X works at", and that one is
    morphology rather than vocabulary — the store holds `founded_by` and the question
    says "founded the", so the phrase never matches. A stemmer would close it; a longer
    word list would only close it here.
    """
    return HybridRetriever(mem.store, mem.embedder, mem.registry, w_graph=w_graph,
                           graph_depth=depth, traverser=mem.traverser,
                           intent_weighting=gated)


def evaluate(mem: Memvara, questions, *, k: int, graph: HybridRetriever,
             ungated: HybridRetriever) -> dict[str, float]:
    hits = dict.fromkeys(
        ("search", "search+graph", "search+graph!", "search x2", "traverse",
         "traverse+min_hops", "traverse+both", "linked"), 0)
    for question, seed, gold, hops, preds in questions:
        results = mem.search(question, k=k, as_of=T0)
        if any(gold in f"{r.claim.subject} {r.claim.object}" for r in results):
            hits["search"] += 1

        # The same call, one constructor argument different. This is the column that says
        # what the *shipped read path* can do, as opposed to what a caller who already
        # knows the seed entity can do with `neighborhood()` — the seed here comes from
        # the head of the fused list, which is the `linked` row's handicap paid inside
        # the retriever instead of by the caller.
        fused = graph.search(question, mem.default_scope, k=k, as_of=T0)
        if any(gold in f"{r.claim.subject} {r.claim.object}" for r in fused):
            hits["search+graph"] += 1

        # The same again with the intent gate off, so the gate's cost is a number rather
        # than an argument. Nothing else differs between the two readers.
        raw = ungated.search(question, mem.default_scope, k=k, as_of=T0)
        if any(gold in f"{r.claim.subject} {r.claim.object}" for r in raw):
            hits["search+graph!"] += 1

        # The agent loop: re-query on what the first search brought back. Given the same
        # `k` budget per step, so the two-step system sees at most 2k claims against
        # traversal's k paths.
        second = []
        for first in results[:1]:
            second = mem.search(f"{first.claim.object} {question}", k=k, as_of=T0)
        if any(gold in f"{r.claim.subject} {r.claim.object}"
               for r in list(results) + list(second)):
            hits["search x2"] += 1

        def ends_at(paths):
            return any(p.labels[-1] == gold for p in paths)

        if ends_at(mem.neighborhood(seed, depth=hops, k=k, as_of=T0)):
            hits["traverse"] += 1
        if ends_at(mem.neighborhood(seed, depth=hops, k=k, min_hops=hops, as_of=T0)):
            hits["traverse+min_hops"] += 1
        if ends_at(mem.neighborhood(seed, depth=hops, k=k, min_hops=hops,
                                    predicates=preds, as_of=T0)):
            hits["traverse+both"] += 1
        if results and ends_at(mem.neighborhood(results[0].claim.subject, depth=hops,
                                                k=k, min_hops=hops, as_of=T0)):
            hits["linked"] += 1
    n = max(1, len(questions))
    return {name: 100.0 * count / n for name, count in hits.items()}


def accuracy() -> None:
    print("\n=== multi-hop answerability (synthetic, self-authored) ===")
    corpus = Corpus(staff=150, companies=24)
    mem = load(corpus, padding=4000)
    asked = corpus.questions()
    questions = genuinely_multi_hop(mem, asked, limit=450)
    two = [q for q in questions if q[3] == 2]
    three = [q for q in questions if q[3] == 3]
    print(f"  {len(questions)} of {len(asked[:450])} questions verified to need the hops "
          f"they claim ({len(two)} two-hop, {len(three)} three-hop)")
    print(f"  store holds {mem.count():,} claims; the one-hop frontier of a person is "
          f"{len(mem.neighborhood(corpus.staff[0], depth=1, k=999, as_of=T0))} paths, "
          f"the two-hop frontier "
          f"{len(mem.neighborhood(corpus.staff[0], depth=2, k=999, as_of=T0))}")

    graph = graph_reader(mem)
    ungated = graph_reader(mem, gated=False)
    header = (f"\n  {'set':<10} {'k':>4} {'search':>8} {'+graph':>8} {'+graph!':>8} "
              f"{'search x2':>10} {'traverse':>9} {'+min_hops':>10} {'+both':>8} "
              f"{'linked':>8}")
    print(header)
    for label, subset in (("two-hop", two), ("three-hop", three), ("all", questions)):
        for k in (5, 12, 25):
            got = evaluate(mem, subset, k=k, graph=graph, ungated=ungated)
            print(f"  {label:<10} {k:>4} {got['search']:>7.1f}% "
                  f"{got['search+graph']:>7.1f}% {got['search+graph!']:>7.1f}% "
                  f"{got['search x2']:>9.1f}% {got['traverse']:>8.1f}%"
                  f" {got['traverse+min_hops']:>9.1f}% {got['traverse+both']:>7.1f}%"
                  f" {got['linked']:>7.1f}%")
    print("  `+graph` is the shipped configuration and `+graph!` the same with "
          "intent_weighting off.\n   The gap between them is what the query-shape gate "
          "still costs here. It was larger, and\n   for a reason this note used to get "
          "wrong: every call below passes `as_of=T0`, which\n   made the intent "
          "`temporal` before the classifier ran, and the temporal row sets the\n   "
          "graph weight to zero. What is left is one family — \"who founded the company "
          "that\n   X works at\" — where the store holds `founded_by` and the question "
          "says \"founded the\".\n   Both walk two hops, the shipped `graph_depth`, "
          "which is why the three-hop rows\n   measure that bound rather than "
          "traversal.")

    question, seed, gold, hops, preds = two[0]
    print(f"\n  example: {question}\n    gold: {gold}")
    for path in mem.neighborhood(seed, depth=hops, k=3, min_hops=hops, predicates=preds,
                                 as_of=T0):
        print(f"    {path.score:.3f}  {path.render()}")
    mem.close()


def interleaving() -> None:
    """The property recall cannot measure, and the reason the feature is worth building.

    At two hops a search-then-search loop scores about as well as a traversal, so recall
    alone says the feature is barely worth it. What recall cannot see is that the loop
    makes two independent reads with two independent clocks, and the store keeps moving
    between them. Below, the write that lands mid-loop *retires* the fact step one
    returned and *creates* the fact step two returns — so the chain the loop reports held
    at no instant whatsoever, and it reports it with full provenance on both hops.

    Deterministic here rather than left as a race: the write is placed between the two
    steps on purpose, which is what a concurrent writer does at a rate that grows with
    traffic.

    A caller who passes the same `as_of` to both searches closes this, and that is the
    honest statement of the difference: the loop is correct only if its author thought of
    it, with no affordance anywhere reminding them, while a traversal pins once before
    its first hop and cannot be called any other way.
    """
    print("\n=== the same question, asked across a write ===")
    mem = Memvara(llm=NullLLM(), embedder=HashingEmbedder(dim=256), user="alice")
    mem.remember("Dana Novak", "works_at", "Kovac Labs", recorded_at=T0)
    mem.remember("Petrov Foundry", "headquartered_in", "Bergen", recorded_at=T0)

    step_one = mem.search("Where does Dana Novak work?", k=5)
    employer = next((r.claim.object for r in step_one if "Kovac" in r.claim.object), None)

    # The interleaved write: Dana changed jobs, and the old employer was acquired.
    mem.remember("Dana Novak", "works_at", "Petrov Foundry")
    mem.remember("Kovac Labs", "acquired_by", "Ahmed Systems")

    step_two = mem.search(f"{employer}", k=5)
    chained = any("Ahmed Systems" in r.claim.object for r in step_two)

    walked_now = mem.paths_between("Dana Novak", "Ahmed Systems", depth=3)
    walked_then = mem.paths_between("Dana Novak", "Ahmed Systems", depth=3, as_of=T0)

    print(f"  loop step 1 answered {employer!r}; step 2 chains it to Ahmed Systems: "
          f"{chained}")
    print(f"  traversal, unpinned:        {len(walked_now)} paths")
    print(f"  traversal, as_of the T0 world: {len(walked_then)} paths")
    print("  Dana's employment at Kovac Labs was retired by the same write that created "
          "the\n   acquisition, so the loop's chain was true at no instant on either "
          "side of it.")
    mem.close()


def latency(sizes: list[int]) -> None:
    """Two sweeps, because the obvious one alone is misleading.

    `graph grows` scales the org with the store, so node degree grows too — at 100k
    claims a home city has 300 residents where at 1k it has three. `graph fixed` holds
    the org at one shape and pads the store with claims nothing in it touches. If the
    first grows and the second is flat, the cost is fan-out rather than store size, which
    is what an indexed lookup should look like; if the second grows too, the index is not
    doing its job and the number is a store problem.
    """
    for label, shape in (("graph grows", True), ("graph fixed", False)):
        print(f"\n=== traversal latency ({label}) ===")
        print(f"  {'claims':>9} {'op':<26} {'p50':>10} {'p95':>10} {'mean':>10}")
        for size in sizes:
            corpus = (Corpus(staff=max(60, size // 40), companies=max(12, size // 200))
                      if shape else Corpus(staff=150, companies=24))
            mem = load(corpus, padding=max(0, size - len(corpus.facts())))
            seeds = corpus.staff[:200]

            def run(op, call):
                samples = []
                for i, seed in enumerate(seeds):
                    t0 = time.perf_counter()
                    call(seed, OFFICE[i % len(OFFICE)])
                    samples.append((time.perf_counter() - t0) * 1000.0)
                print(f"  {mem.count():>9,} {op:<26} {percentile(samples, 0.5):>9.3f}ms "
                      f"{percentile(samples, 0.95):>9.3f}ms {mean(samples):>9.3f}ms")

            for depth in (1, 2, 3):
                run(f"neighborhood depth={depth}",
                    lambda s, _t, d=depth: mem.neighborhood(s, depth=d, k=10, as_of=T0))
            run("neighborhood d=2 +preds",
                lambda s, _t: mem.neighborhood(s, depth=2, k=10, as_of=T0,
                                               predicates=["works_at",
                                                           "headquartered_in"]))
            run("paths_between depth=3",
                lambda s, t: mem.paths_between(s, t, depth=3, k=3, as_of=T0))
            run("search k=10 (for scale)", lambda s, _t: mem.search(s, k=10, as_of=T0))
            mem.close()


if __name__ == "__main__":
    argv = sys.argv[1:]
    sizes = [1_000, 10_000, 100_000]
    if "--sizes" in argv:
        sizes = [int(s) for s in argv[argv.index("--sizes") + 1].split(",")]
    accuracy()
    interleaving()
    latency(sizes)
