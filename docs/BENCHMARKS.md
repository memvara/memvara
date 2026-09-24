# Benchmarks

Every number here is reproducible from this repository; the harnesses are in
`bench/` and `demo/`. Where a result is synthetic or self-authored it says so in
its own heading, because that is the part a reader is entitled to discount.

**The Agent Memory Benchmark is on its own page.** It is the one measurement here that is
not about memvara: a system-neutral dataset and adapter interface that any memory system
can implement, with baselines that beat memvara in one category.
See [the report](benchmarks/agent-memory-benchmark.md), or run it:

```bash
python -m benchmarks.agent_memory --system memvara --system naive --system vector-rag --compare
```

## The Plugin Recall Benchmark is also on its own page

`benchmarks/agent_memory` grades memory *systems*. `benchmarks/plugin_recall` grades what a
memory **plugin** puts in a model's context, driven through the editor's own hook protocol
so it needs no per-vendor code — see [its README](../benchmarks/plugin_recall/README.md).

```bash
python -m benchmarks.plugin_recall --plugin memvara
```

Half its corpus is prompts with **no** right answer, because a plugin that injects its whole
store on every prompt scores 100% on a hit-only benchmark. On the seeded reference store,
the shipped hook scored 100% on hits and **0% on silence** — it answered every bare
acknowledgement, every arithmetic question, and all eight lexical traps. Passing the
`min_score` floor that `Memvara.recall` has always accepted took it to 100%/100%, and cut
mean injected tokens from 111 to 23.

The floor is store-specific: scores are not comparable between embedders, and the value
that separates cleanly on the reference store filters nothing on the hosted one. Calibrate
against your own store rather than trusting the default:

```bash
python -m benchmarks.plugin_recall.calibrate --db ~/.memvara/store.db
```

## Measured against the real mem0 package

`pip install mem0ai && PYTHONPATH=. python3 bench/mem0_real.py` — mem0 **2.0.17**, not a
reimplementation of it. Same 105-turn transcript, same perfect extraction oracle, same
`HashingEmbedder`, Qdrant in `:memory:`. Fully offline. Five runs each:

| metric | mem0 2.0.17 | memvara |
|---|---:|---:|
| LLM calls on the write path | 105 | **2** |
| Current value stored correctly | 9–10 / 10 | **10 / 10** |
| Stale values left live | 10–11 | **0** |
| Live rows in the store | 20 | **10** |
| **Identical result every run** | **no** | **yes** |
| Wall clock, median | 108 ms | **11 ms** |
| Install size | 33 packages | **2 packages** |

The row that matters is not the stale count — it is **`no`**. The oracle returns
byte-identical JSON on every run and both systems use the same deterministic embedder, so
there is no model variance in this harness at all. mem0 still reaches a different final
state between runs on identical input. We did not isolate the cause inside mem0, only
established that it is not the model and not the embeddings, because neither varies here.

That is the "a keyed lookup has no threshold to get wrong" claim, measured against the
real package instead of argued against something we wrote.

**Two caveats that cut against these numbers.** mem0 is charged per turn while memvara
receives the transcript in one `add()`, so the call-count row is partly an
ingestion-granularity choice — the equal-granularity figure is 126 vs 17, below. And the
oracle gives mem0 *perfect* extraction, which no real deployment gets; the stale count is
therefore a floor for mem0, not a typical case.

**The first version of this benchmark was wrong, in memvara's favour.** Its oracle
string-matched the whole prompt for known turns, and mem0's additive prompt embeds
`last_k_messages` — so every earlier turn in the window matched and was re-extracted,
emitting each fact eleven times and measuring mem0 under a firehose no real extractor
would produce. It reported 6/10 for mem0. A benchmark whose bug flatters its author is the
one to distrust most, so the mechanism is documented in `bench/mem0_real.py`.

---

## The two clocks, measured (synthetic, self-authored)

`PYTHONPATH=. python3 bench/temporal.py` — no model, no network, no reader, no judge, and
byte-identical on every run because every instant in it is a module constant.

Everything else on this page measures *retrieval*. That is the commodity half, and it is
the half benchmarked against competitors. The differentiator — two independent clocks,
supersession that closes exactly one of them, source authority — had **no number at all**
until this harness existed, and the cost of that was not hypothetical: two defects lived
on the write path while 3,448 tests passed.

Six families over 48 authored scenarios and 160 writes, scored as exact set matches
against golds the generator builds before anything is written:

| family | n | memvara | `no-clocks` | `disc` |
|---|---:|---:|---:|---:|
| `point_in_time` — `valid_at=T` | 24 | **100.0%** | 33.3% | 66.7% |
| `delayed_knowledge` — `known_at` against `valid_at` | 16 | **100.0%** | 50.0% | 50.0% |
| `as_of_audit` — both clocks together | 24 | **100.0%** | 33.3% | 66.7% |
| `contradiction` — ONE resolving, MANY not | 24 | **100.0%** | 66.7% | 33.3% |
| `correction` — `ended` against `retired` | 16 | **100.0%** | 50.0% | 50.0% |
| `source_authority` — a guess meeting a statement | 16 | **100.0%** | 100.0% | 0.0% |
| **all** | **120** | **100.0%** | **53.3%** | **46.7%** |

`no-clocks` answers every question with the present-tense live set: what a store with one
clock can say, asked the same questions. `disc` is the share of a family's questions it
gets wrong. **A family at 0.0% is one the baseline handles**, and the table says so rather
than hiding it — `source_authority` is present-tense by construction, so a read-side
baseline ties there and the comparator has to be this repository's own history.

### The before-and-after, which is the point

The same file against `origin/main` at `7b91a9a`, before the two write-path fixes:

| | `7b91a9a` | after |
|---|---:|---:|
| `source_authority` | **50.0%** | 100.0% |
| all | 93.3% | **100.0%** |
| `ended` claims that answer at no instant | 8 of 56 | 8 of 48 |
| ...of which the write path reported | **0** | **8** |

`source_authority` at 50.0% is eight of eight scenarios in which a 0.10-confidence guess
displaced a 1.00-confidence statement — and stamped it `ended`, which asserts the world
changed. The `ended` totals differ (56 against 48) for the same reason: on `7b91a9a` each
of those eight guesses ended a claim that should not have been ended.

The last row is not an accuracy question and cannot be one. A claim closed at or before
the instant it began holds for no interval, so it is absent from every answer at every
instant on either clock, and no gold can name a row no query returns. It is checked
against the rows instead: how many `ended` claims answer nothing, and how many of those
the write said so about while making them. The corpus produces them deliberately — the
`contradiction` family writes the same-instant case that every import stamping dates
rather than timestamps produces.

### What it is not

**Synthetic and self-authored**, in the same category as `bench/multihop.py` and
`bench/compare.py` and to be discounted the same way. It is an illustration of a
mechanism, not evidence against another system; nothing here is a head-to-head. The
scenarios are triples with instants and contain no English, so none of the extraction path
is exercised and none of its cost or failure modes appear.

**It probes `get_all`, not `search`.** That is deliberate: it measures the temporal axes
and not the ranker, so a temporal regression cannot be confused with a ranking one. The
retrieval side of temporal questions is [the temporal leg](#the-temporal-leg-and-the-abstention-that-is-the-actual-finding),
below, and it is a different measurement.

**One gold in it was wrong before the harness caught it.** The `contradiction` family used
`prefers_tool` for its accumulating slot on the strength of the name; it is
`Cardinality.ONE`, and the family reported 33.3% until the gold was corrected to use
`speaks`. The four anti-flattery constraints in the file's docstring exist because that is
the normal failure mode of a self-authored benchmark, and this one hit it on the first run.

---

## Answer accuracy, judged, in the MemoryBench harness

Everywhere else on this page, "measured" means retrieval — did the right evidence come
back — because that needs no model and the reader is a shared confound either system
would carry. This section is the one exception: an LLM judges whether the *answer* is
right, on the shipped 0.11.0 read path, through a harness neither party controls.

**Method.** LongMemEval-S, a 199-question stratified sample (seed `20260903`) of the
public 500, run inside the [MemoryBench harness](https://github.com/supermemoryai/memorybench)
— open source, published by Supermemory, not written for this project. Reader and judge
are both `gpt-5.4`, and the metric is LLM-judged answer accuracy. Reader
self-disagreement on identical prompts is **7.8%**, so a single-run difference under
about 8 questions is noise, not a finding.

**The shipped path** is memvara 0.11.0 as released, read through the hosted service with
`ranked=True` and `gpt-5.4-mini` as the selector, billed on the customer's own key — one
run:

| path | selector | median context | correct |
|---|---|---:|---:|
| Shipped (0.11.0, `ranked=True`) | gpt-5.4-mini | 549 tok (p90 693) | **177 / 199 (88.9%)** |
| Twin — same retrieval, no selector, same budget | none | 720 tok | 135 / 199 (67.8%) |
| Control — no selector, wide budget | none | 4,089 tok | 172 / 199 (86.4%) |
| Selector swapped — same retrieval, rendered offline, 2 runs | gpt-5.4 | 672 tok (both runs) | **182 / 199 (91.5%)** |

The twin isolates what the selector buys at a fixed budget: paired against it, the
shipped path answers **42 more questions correctly** on the same 199. Against the control
the shipped path uses about a seventh of the tokens (549 against 4,089) and lands 5
questions ahead, which is inside the noise floor above: the selector matches the wide
context, it does not beat it.

**Per type, shipped path:** single-session-user 27/28, single-session-assistant 22/22,
single-session-preference 7/12, multi-session 45/53, temporal-reasoning 47/53,
knowledge-update 29/31. Preference is the weak row here for the same reason it is weak in
the retrieval table below: the golds are meta-descriptions no single turn contains. It is
the row to check before trusting the headline number in production. Whether a stronger
selector lifts it is not known; the `gpt-5.4` run (91.5% overall) is not broken out by
type here.

**Off the tuning sample.** A one-run, one-sample number invites overfitting to that
sample; the way to check is to look at what the selector keeps on data it was never tuned
against. Screened over all 500 LongMemEval questions, the shipped selector keeps
**93.7% of gold turns (819 of 874)** at a non-gold keep rate of **6.0%**. The 199-question
sample it was scored on above reads 93.5% and 6.8% on the same two measures — close enough
that the headline number is not an artifact of the sample it was picked to look good on.

**Cost, on the customer's own provider key:** on average about **0.35 cents per ranked
recall** with `gpt-5.4-mini` as the selector, about **1.1 cents** with `gpt-5.4`. The
median call is cheaper, 0.27 and 0.87 cents, because the prompt distribution is
right-skewed: median 3,325 prompt tokens, p95 14,609. Memvara pays nothing for the model call: the hosted service makes it on the
customer's key, and the provider bills the customer.

**What this is not.** No same-harness comparison against Supermemory's own hosted service
exists. Ingesting the 199-question sample into it was quoted at roughly $160, and their
open-source engine could not take the bulk ingest this comparison needed, so the run was
never completed — this is not a result being withheld, it is a result that does not
exist. Supermemory's published 95% is also not a like-for-like number to reach for: it is
Recall@15, a retrieval metric, and this section's numbers are judged answer accuracy.
The two measure different things and are not comparable.

The retrieval-only figures in the rest of this page — R@12 70.4 on the LongMemEval oracle
split and the rest — are unaffected by any of the above and remain true. They are a
different measurement, made without a reader, and the next section is exactly that one.

---

## LOCOMO and LongMemEval — retrieval, measured

Not answer accuracy — that number is judged, separately, [above](#answer-accuracy-judged-in-the-memorybench-harness)
— and **not comparable to published LOCOMO/LongMemEval scores**, which are end-to-end judged
accuracy on the full public sets. This measures the thing a memory layer is actually
responsible for: *did retrieval surface the evidence the annotators marked?* It needs no
model, so it runs the full question sets for nothing, and it removes the reader — which
both systems would share anyway — as a confound.

```bash
PYTHONPATH=. python3 bench/locomo.py       --score retrieval
PYTHONPATH=. python3 bench/longmemeval.py  --score retrieval --share-store
```

`k=12`, 4000-char budget, `HashingEmbedder`, `NullLLM` — **no extraction ran**, so this is
episode retrieval alone. `chance` is the share of the haystack marked as evidence: what
random retrieval would score.

**LOCOMO, all 1,531 evidence-labelled questions** — recall of annotator-marked evidence:

| category | n | R@1 | R@5 | **R@12** | R@20 | MRR | chance |
|---|---:|---:|---:|---:|---:|---:|---:|
| single-hop | 840 | 35.7 | 60.0 | **70.7** | 75.5 | 48.1 | 0.2 |
| temporal | 320 | 41.5 | 63.1 | **71.0** | 76.2 | 54.0 | 0.2 |
| multi-hop | 279 | 7.4 | 22.9 | **36.0** | 44.0 | 31.6 | 0.5 |
| open-domain | 92 | 13.9 | 22.4 | **30.7** | 34.1 | 24.7 | 0.4 |
| **all** | **1531** | **30.5** | **51.7** | **62.0** | **67.4** | **44.9** | **0.3** |

**LongMemEval, all 500, one shared 940-session store** so there are distractors:

| category | n | R@1 | R@5 | **R@12** | MRR | chance |
|---|---:|---:|---:|---:|---:|---:|
| single-session-assistant | 56 | 96.4 | 98.2 | **100.0** | 97.6 | 0.1 |
| single-session-user | 64 | 56.2 | 76.6 | **92.2** | 66.0 | 0.1 |
| knowledge-update | 72 | 39.6 | 79.9 | **91.0** | 85.3 | 0.2 |
| temporal-reasoning | 127 | 23.6 | 52.1 | **66.6** | 56.4 | 0.3 |
| multi-session | 121 | 22.4 | 45.1 | **65.5** | 61.7 | 0.3 |
| single-session-preference | 30 | 13.3 | 20.0 | **23.3** | 17.4 | 0.1 |
| abstention | 30 | 0.0 | 1.7 | **1.7** | 0.7 | 0.2 |
| **all** | **500** | **35.9** | **57.7** | **70.4** | **62.0** | **0.2** |

**Read the weak rows first.** Multi-hop LOCOMO is 36% and open-domain is 31% — questions
needing evidence stitched across sessions are where a top-k budget hurts most, and no
amount of contradiction resolution helps. A reranker does, though: see below. LongMemEval abstention is **1.7%**, essentially
never: unanswerable questions retrieve nothing relevant, which is the right *outcome* by
accident rather than by design. Preference questions score 23% because their golds are
30-token meta-descriptions no single turn can contain — a metric artifact, visible in the
`best cov` column the report prints.

`knowledge-update` at **91.0%** is the row that matters for the thesis: it is the category
where a fact changes and the old value must not win.

### What a reranker buys

Every number above is the **shipped default, which has no reranker**. Turning one on is
one constructor argument and an optional install, and on LOCOMO it is the largest single
improvement available:

| LOCOMO, 1,531 questions | R@1 | R@5 | **R@12** | R@20 | MRR |
|---|---:|---:|---:|---:|---:|
| default (no reranker) | 30.5 | 51.7 | **62.0** | 67.4 | 44.9 |
| `+ cross-encoder/ms-marco-MiniLM-L-6-v2`, `top_n=20` | **44.9** | **62.1** | **66.5** | 67.4 | **59.2** |

```python
from memvara import Memvara
from memvara.rerank import CrossEncoderReranker      # pip install 'memvara[rerank]'

mem = Memvara("memory.db", read_reranker=CrossEncoderReranker(), read_rerank_top_n=20)
```

**R@12 understates it.** A reranker over the top 20 cannot find evidence retrieval
missed — R@20 is identical in both rows, and must be — so the entire effect is moving the
right evidence *upward*. That is why R@1 gains 14.4 points and MRR gains 14.3: the win
lands exactly where a token budget spends. Multi-hop R@1 more than doubles, 7.4 → 16.2.

Two things worth knowing before you reach for a bigger model. `BAAI/bge-reranker-base` is
12× the parameters and scores **lower** on every metric at 5× the runtime. And a
reranker is the query latency once it is on — roughly 84 ms at `top_n=20` against a ~3 ms
search. That cost, not the accuracy, is why the default is still `None`.

The dependency-free `CoverageReranker` is a **control, not a recommendation**: it is
lexical, it measures what the *stage* does without a model, and on this suite it nets
−0.1. Full table, per-category breakdown and the reproduce commands are in
[docs/ROADMAP.md](ROADMAP.md).

Two findings from building this. **LongMemEval's `oracle` split cannot measure evidence
retrieval at all** — in all 500 instances every haystack session *is* an evidence session,
so recall there is 99.2% by arithmetic. The harness now computes `chance` and warns loudly
above 50%; `--share-store` is the offline workaround. And **retrieval was not reproducible
until this run**: `HybridRetriever` broke score ties on `claim.id`, a fresh `uuid4` per
ingest, so two ingests of one corpus ranked differently and the numbers drifted 0.07
points. Ties now break on a content hash and three full runs are byte-identical.

**That fix was one layer short, and `bench/twowiki.py` found the rest of it.** A score tie
no longer depends on ingest, but the *score* still did: `remember()` stamps each claim
with the wall clock, so 1,239 claims carried 1,239 distinct `valid_from` values and
`recency_factor` turned write order into a strict ranking. On top of that a search decays
from the moment it is asked, so two identical passes scored **3,000 of 3,000** questions
differently — in the low-order digits, but enough to flip a near-tie at the `k` boundary.

`HybridRetriever.search()` and `GraphTraverser.spread()` now take `now=`, the parameter
`Consolidator.run()` already had, and that harness pins both the instant it writes at and
the instant it reads at. Two runs of it are byte-identical. The 2Wiki table below moved by
up to 1.6 points when this landed, and the new figures are the ones without write order in
them.

`bench/locomo.py`, `bench/longmemeval.py` and `bench/multihop.py` were checked and report
identical figures across runs without a pin: their claims are either absent or carry
timestamps years old, which is the flat part of the decay curve.

### The graph leg, and what it costs on the corpora above

`w_graph > 0` adds a third retrieval leg: a bounded walk out of the entities the vector
and lexical legs just named (`memvara/retrieve/spread.py`). **It ships at `w_graph=0.0`,
because neither corpus above holds enough of a graph for the walk to pay for itself:
LOCOMO cannot see the leg at all, and LongMemEval sees it lose.**

```bash
PYTHONPATH=. python3 bench/locomo.py      --score retrieval --w-graph 1.0
PYTHONPATH=. python3 bench/longmemeval.py --score retrieval --share-store --w-graph 1.0
```

| instrument | claims in the store | what the leg changed |
|---|---:|---|
| LOCOMO, 1,531 questions | **0** | nothing — the two reports are byte-identical |
| LongMemEval oracle, 500, `--share-store` | **78** | **a loss**: single-session-user R@12 92.2 → 90.6, all 70.4 → 70.1, nothing gained |
| `bench/multihop.py` (synthetic), gate off | 4,498 | **2.9% → 20.0%** at k=12, **7.6% → 50.0%** at k=25 |
| `bench/multihop.py`, **as shipped** | 4,498 | **2.9% → 20.0%** at k=12, **7.6% → 50.0%** at k=25 — the same as with the gate off |
| `bench/twowiki.py`, gate off, **public** | 26,403 | **28.2% → 67.3%** at k=12 on chained questions; **−14.7** on flat ones |
| `bench/twowiki.py`, **as shipped** | 26,403 | **28.2% → 48.3%** answer and **25.5% → 45.8%** chain on chained questions; **−1.4** on flat ones |

The leg walks *claims*, and both public runs are episode retrieval: `SalienceGate` drops
any turn whose role is not `user`, LOCOMO writes each turn under the speaker's name, and
the deterministic extractor's vocabulary is first-person declaratives. LOCOMO extracts
**0 claims from 5,882 turns** and LongMemEval **78 from 10,866**. With no claims the
candidate set is empty and the leg is never reached, so the LOCOMO figure is not a null
result — it is the leg being inert by construction.

`bench/multihop.py` already said the other half of this, before the leg existed: LOCOMO's
`multi-hop` category is single-fact lookups whose evidence happens to span one or two
turns, not transitive relations over entities, "so a graph walk is not what that 36% row
is short of."

What the one instrument that *can* see it measures — `search` is the shipped read path,
`+graph` the same call with one constructor argument changed, `linked` the best a caller
could previously get by hand (take the seed entity off the top hit and call
`neighborhood()` yourself):

```
  set           k   search   +graph  +graph!  search x2  traverse  +min_hops    +both   linked
  two-hop      12     4.0%    29.7%    29.7%      64.3%     69.7%     100.0%   100.0%    99.7%
  two-hop      25     9.3%    72.7%    72.7%      96.3%    100.0%     100.0%   100.0%    99.7%
  three-hop    25     4.0%     4.7%     4.7%       4.7%     34.7%      48.7%   100.0%    46.7%
  all          12     2.9%    20.0%    20.0%      43.1%     46.4%      78.7%    83.1%    77.8%
  all          25     7.6%    50.0%    50.0%      65.8%     78.2%      82.9%   100.0%    82.0%
```

**`+graph` is the shipped configuration and `+graph!` is the same with
`intent_weighting=False`.** The two columns are equal in every row, measured on
2026-09-13: on this workload the gate now costs nothing. The `+graph` column used to read
`2.9%` and `7.6%` — exactly `search`, as though the leg were not installed — and then
`6.4%` and `21.8%`, with one question family still gated. Every other column reproduced
the previously published figure exactly, which is the check that only the gate moved.

**The published reason for that was wrong, and finding out why is the more useful half of
this entry.** This document, the benchmark's own footnote and the classifier's source
comment all said the same thing: two of the three question families contain no word in
`intent.RELATIONAL_MARKERS`, so the gate reads them as `lookup`. They do not contain one,
and that was not the cause. `evaluate()` passes `as_of=T0` on every call. `_weights` takes
its `timed` branch whenever an axis is given, the classifier is never consulted, and
`Intent.TEMPORAL`'s multipliers set the graph weight to **zero**. The column measured a
configuration in which the leg could not run at all, and three separate documents
explained the resulting number in terms of a mechanism that never executed.

The `timed` override is right for the temporal leg — a caller who resolved an instant has
said more about time than any word could — and it was never meant to say anything about
chains. It said the strongest possible thing silently. *"Where was Alice's employer based
in 2019"* is the query this library exists for, and it was the shape that lost the walk.

Three things are fixed and the numbers above are after all of them:

* **Naming an instant no longer switches the walk off.** The temporal row still decides
  the other three legs; the graph leg keeps the weight the query shape asked for.
* **A question names a predicate in whatever form it inflects it** (#150). The store
  holds `founded_by` and the question says "founded the company"; both sides of the match
  now fold through `schema.word_stem`, so that family opens the walk like the other two.
  Before this, "who founded the company that X works at" was the one family still gated,
  and it was the whole of the gap between `6.4%` and `20.0%`.
* **The classifier counts predicates instead of matching a longer word list.**
  `intent.predicate_refs` counts how many *distinct* predicates a question names, folded
  onto canonical names, and two of them is a chain — one predicate is a question about one
  slot. Derived from `PredicateRegistry`, so no word was added because this benchmark
  needed it. **How far it reaches is narrower than "derived from the registry" suggests:**
  `PredicateRegistry.learn()` is called only from the LLM-assisted resolution in
  `write/pipeline.py`, so an offline store never teaches it and the rule sees the 23
  builtins alone. `bench/twowiki.py` exposed that — every predicate in that corpus is a
  learned one, so the rule does not fire there. `bench/packs/twowiki.toml` now declares all
  34 of them, which is a prerequisite rather than a tuning knob; see below. Matched as phrases
  and never as tokens: `lives_in` splits into `lives` and `in`, and a token index would
  read almost every question as a chain.

## The graph benchmark needs a vocabulary of its own, and this is why

`docs/SUBJECT-CONVENTIONS.md` decision 3 makes an object's kind a property of its predicate:
a predicate whose `object_type` is undeclared takes *values*, and a value carries no graph
edge. Connectivity becomes exactly as large as the declared vocabulary and not one edge
larger.

2WikiMultihopQA's evidence is Wikidata relations, and the 23 builtins are a
personal-assistant vocabulary. `bench/predicate_audit.py` measures the overlap:

```
$ PYTHONPATH=. python3 bench/predicate_audit.py
  vocabulary: builtins only
  relations: 34   triples asserted: 31,120
  declared entity-valued: 0 relations, 0 triples
  declared value-valued:  3 relations, 6,772 triples   (deliberate: no edge)
  undeclared:             31 relations, 24,348 triples   (the gap)
  share of triples that can carry an edge: 0.0%
```

So once that rule reaches retrieval, every triple in the corpus is value-valued and the
40.6% joinable figure this file reports elsewhere becomes 0.0% — the graph benchmark
reporting that the graph leg had stopped working, correctly, and for reasons having nothing
to do with whether the design is any good. `bench/packs/twowiki.toml` closes it:

```
$ PYTHONPATH=. python3 bench/predicate_audit.py --packs bench/packs/twowiki.toml
  declared entity-valued: 30 relations, 21,266 triples
  declared value-valued:  4 relations, 9,854 triples   (deliberate: no edge)
  undeclared:             0 relations, 0 triples   (the gap)
  share of triples that can carry an edge: 68.3%
```

**The remaining 31.7% is not a gap.** Four relations — `date_of_birth`, `date_of_death`,
`publication_date` and `inception` — take dates, and 9,854 of the corpus's 31,120 triples
are theirs. Declaring them entity-valued would connect every person born in 1935 to every
work published in 1935, which is the false join the classification rule exists to prevent.
A third of this corpus is a value, and the audit reports declared-as-value separately from
undeclared for exactly that reason: both carry no edge, and only one of them is a problem.

**Two things about the pack are load-bearing and neither is obvious.** Three of the corpus's
relations are already aliases of builtins, so `bench/twowiki.py` stores them under the
canonical names — `date_of_birth` as `born_on`, `place_of_birth` as `born_in`, `employer` as
`works_at` — and the pack has to declare those names rather than the corpus spelling. And a
declared spec *replaces* a builtin of the same name rather than extending it, so the
override has to repeat the builtin's alias list: a bare `born_on` declaration made all three
relations resolve to nothing, and the audit reported them as undeclared *because* they had
been declared. `tests/test_predicate_packs.py` pins both.

**The same rule reached `bench/multihop.py`, and for a day the harness measured nothing.**
Its five relations were never declared, so once only a declared relation carries an edge
every object in its store was a value: the one-hop frontier of a person was 0 paths, every
traversal column read 0.0%, and `+graph` equalled `search` in every row — which the
footnote then explained as the gate's cost. `bench/multihop.py` now declares its relations
in `vocabulary()`, extending the two builtins it uses rather than replacing them, and
`tests/test_predicate_packs.py` pins both harnesses' registries so the next change of that
kind fails in a test rather than in a table. The `+graph!` column reproduced its published
figure to the decimal once the edges were back, so the padding claims, which stay values,
never contributed to the walk.

**The last gated family was morphology rather than vocabulary, and it is closed.** "Who
founded the company that X works at" names `works_at` and `founded_by`, but the store
holds `founded_by` and the question says "founded the", so the phrase never matched.
Matching the head token instead was measured and rejected: the head tokens of this
registry's predicates include `in`, `is`, `do`, `has`, `date` and `place`, which turns
"what is my name" into a two-predicate chain. A stemmer closed it (#150): both sides fold
through `schema.word_stem`, and the fold is the registry's own, so it only has to agree
with itself.

One false positive came out of that fold and is fixed here. `works_at` and `job_title`'s
alias `works_as` both reduce to `work` once the prepositions are gone, and the count took
each word on its own, so "what company does Ada work at" — `work` to `job_title`, `company`
to `works_at` — read as a chain and opened the walk on the plainest lookup there is. The
count is now the fewest predicates that account for everything the question said. It
changes no row above: every question here names two relations that only one predicate
each can explain.

**The standing advice needs a condition on it, which `bench/twowiki.py` supplied.** It
used to read: a deployment turning the graph leg on should turn `intent_weighting` off
with it. On public multi-hop data that buys 39 points on chained questions and **costs
15 on flat ones**, so it is right for a workload of relationship questions and wrong for
a workload of lookups. Net on a corpus that is 54% chained it is +14.4 points at k=12;
invert the mix and it inverts.

The honest statement is that the gate is right in principle and half calibrated. As
shipped it captures 20.1 of the 39.1 points on chained questions and pays 1.4 of the 14.7
on flat ones, measured on 2026-09-13 with the corpus's relations declared; it used to
capture almost none of the gain. What it still pays on flat questions comes from the
hand-written markers rather than from the predicate count. Of the first 3,000 questions,
the walk ran on 29.1% of the flat ones with the gate on, and 22.3% of them classify as
relational through `same`, `both` or `whose` — comparison questions written without a
disjunction, so `is_comparison` does not see them — against 0.1% that open through two
declared predicates. A deployment should turn `intent_weighting` off if its traffic is
mostly relationship questions, and leave the graph leg off entirely if it is mostly
lookups. Neither is a default this repository can pick for you, which is why `w_graph`
ships at 0.0.

**The store now asks itself.** Where no live claim's object is another live claim's
subject, the graph leg does not run whatever `w_graph` says — so turning it on costs
nothing on a store that cannot use it. Measured on LongMemEval with `w_graph=1.0`: every
category exactly baseline, where it previously lost 1.6 points of single-session-user
R@12. On 2Wiki the gate closed the leg on 0 of 3,000 searches and no returned row moved.
See `UnjoinedStoreWarning`, which says so out loud once per retriever.

That is a floor, not a recommendation. **Ask the store before you guess at the traffic**,
because the store is the half you can measure. `memory_stats` reports a **join rate** — the share of live claims whose object is
the subject of another live claim, which is the share that leads anywhere at all. The two
corpora below sit at 29.0% and 0.0% and the leg gains 20 points on one and loses 1.6 on
the other, so the rate predicts the sign where a guess about query mix does not. Under
about 1% the store is a *star*, every fact hanging off one subject, and there is no
second hop to find however the traffic is shaped. `Memvara.connectivity()` is the same
two counts in the library.

The three-hop rows barely move because `graph_depth` ships at 2; that row measures the
bound, not the traversal. And this benchmark is synthetic and self-authored — read it as
an illustration of a mechanism, which is not evidence for a default.

**So it is opt-in, and on a store with almost no graph in it, turning it on costs
something.** On LongMemEval the leg loses 1.6 points of single-session-user R@12
(92.2 → 90.6) and 0.3 overall (70.4 → 70.1), and no category gains. Both runs ingest the
same 78 claims and 12 reinforcements, so every part of that difference is the read path:
a third leg that reaches almost nothing still votes, and fusion reads positions, so it
puts a real zero on every candidate the walk did not touch. That is the same failure the
temporal leg's `MIN_PROXIMITY` floor exists to prevent, and the graph leg has no
equivalent. The precedent for shipping a measured stage at zero is the MMR rejection
recorded in `hybrid.py`.

<div data-type="panel-warning">

**This paragraph used to claim the opposite, and the number moved under it.** It read
"every R@k in both public runs held exactly ... single-session-user 92.2 → 92.2", which
was true when it was written and stopped being true three commits later.

The cause is the gate work in this section. Before it, `evaluate()` passing an instant
forced `Intent.TEMPORAL`, whose multipliers zero the graph weight, so the leg barely ran
on LongMemEval and could not cost anything. Fixing that, and then teaching the classifier
to read vocabulary off retrieved rows, let the leg fire on queries it used to skip — on a
store holding 78 claims, where there is nothing to walk to.

Nothing caught it, because no test asserts a benchmark figure and the commits that moved
this number edited a different file. Re-measured on `016afbf`; raw output is the two
commands above.

</div>

```python
mem = Memvara("memory.db", read_w_graph=1.0)
```

`memvara/retrieve/intent.py` is what makes turning it on affordable — a deterministic,
model-free classifier routes `lookup` and `temporal` queries past the walk entirely — and
the table above is also what it currently costs. Every multiplier in it other than the two
gates is 1.0 and stays 1.0 until a per-category sweep moves it.

### The graph leg on public data, with the extractor out of the loop

**The leg is worth 2.4x on multi-hop questions, and costs 15 points on questions that are
not.** Both halves are new information, and the second is the more useful one.

Everything above this section says the leg is unmeasurable on public data, because LOCOMO
and LongMemEval are prose and the offline extractor gets 0 claims from LOCOMO's 5,882
turns. That is a fact about *extraction*, and it was being reported as a fact about
retrieval. `bench/twowiki.py` separates them: 2WikiMultihopQA ships its evidence as
`[subject, relation, object]` triples from Wikidata, so they load through `remember()`
with no extractor running.

Full dev set, 12,576 questions, 26,403 distinct claims in **one shared scope** — every
question answered against every other question's facts. Each cell is *answer found in the
returned rows / whole evidence chain returned*:

```
  k=12
  set                     n         search         +graph        +graph!
  all                12,576   50.5% / 37.2%   60.8% / 46.2%   64.9% / 51.4%
  chained             6,785   28.2% / 25.5%   48.3% / 45.8%   67.3% / 65.5%
  flat                5,791   76.7% / 50.8%   75.3% / 46.8%   62.0% / 34.9%
  compositional       5,236   22.8% / 20.5%   48.8% / 46.8%   63.5% / 61.7%
  inference           1,549   46.6% / 42.3%   46.7% / 42.4%   80.4% / 78.4%
  comparison          3,040   73.9% / 96.8%   73.4% / 87.1%   58.1% / 64.5%
  bridge_comparison   2,751   79.8% /  0.0%   77.5% /  2.2%   66.4% /  2.2%
```

Measured on 2026-09-13, with the harness loading `bench/packs/twowiki.toml`. Since 0.12
only a declared relation carries an edge. This harness declared nothing, so for a day it
measured a store with no edges: `search`, `+graph` and `+graph!` were equal in every row.
`ingest()` now builds its registry from the builtins and the pack, and the store's
join rate with it is 29.0% (7,663 of 26,402 live claims lead to another). Three things
moved against the previously published table, and none of them is the gate. The four
date relations are values now — 9,854 triples that used to be edges — so every walk
reaches less: with the gate off, `chained` fell from 72.3% to 67.3% and
`bridge_comparison` chain recall from 23.8% to 2.2%, which was the year-hub join that
decision 3 exists to prevent. With the gate on, `chained` rose from 43.8% to 48.3%,
because the declared vocabulary is visible to `classify` where before only the rows a
query happened to retrieve were. And `flat` gives up 1.4 points with the gate on where it
gave up none, because the walks it still runs reach different rows. The matcher fix
described above moved no cell: the table with it and without it is identical to the
decimal, here and on `bench/multihop.py`. `search` moved by 0.1 on two rows; the pack
declares every relation `static`, and a near-tie at the `k` boundary is the likely
reason.

The previous table was after the gate repair filed as
[#150](https://github.com/memvara/memvara/issues/150): a question names a predicate in
whatever form it inflects it, and a chain that also names an instant keeps the walk. That
repair moved `chained` from 42.1 / 39.5 to 43.8 / 41.3 with `search` and `+graph!`
unmoved, which was the check that only the gate had changed.

**`chained` is the result.** `compositional` and `inference` questions chain one fact into
the next — "who is the mother of the director of X" is `director` then `mother` — and the
leg takes them from **28.2% to 67.3%**. `inference` also carries its derivation: chain
recall 42.3% → 78.4%, so most answers arrive with every triple that supports them rather
than with the gold entity alone.

**`flat` is the control, and it did what a control is for.** `comparison` and
`bridge_comparison` ask which of two independent entities came first. The evidence has two
ends and no join, the leg has nothing to walk, and turning it on **costs 14.7 points**
(76.7% → 62.0%) because the walk spends `k` on neighbours of a hub. Had that row improved,
the `chained` row would be worth much less: it would suggest the leg helps by adding rows
rather than by following edges.

**The intent gate is right in principle, and it now captures half of what it was
blocking.** It exists to route flat questions past the walk, and on `flat` it mostly
does: 75.3% against search's 76.7%, where the gate off costs 14.7. On `chained` it
captures 20.1 of the 39.1 points available. It used to block almost the entire gain —
29.1% where 72.2% was available, 0.9 points of 43.9.

The reason was vocabulary, and not the kind a word list fixes. `classify` counts the
predicates a question names, drawn from `PredicateRegistry.all_specs()`, which lists what
somebody **declared**. A predicate written through `remember()` is never declared — the
registry synthesizes a spec on demand and does not remember — so on a store whose
vocabulary arrived that way the count is one or zero and every chain question reads as a
lookup. All 34 of this corpus's relations are of that kind.

The fix reads the vocabulary off the rows instead. The lookup legs run first, so by the
time the graph weight matters the candidates are in hand, and their predicates are the
store's vocabulary — observed rather than declared, and already narrowed to this query.
A question naming two of them is a chain. `chained` goes **29.1% → 35.4%** and chain
recall 19.8% → 26.1%; `flat` is unchanged at 75.3%, so the discrimination holds.

**Teaching the registry instead was tried first and reverted**, and the reason is worth
recording. Recording an observed predicate means recording a cardinality; the only one
available is the default; the store would then hold `MANY` chosen by nobody, and
`memory_remember`'s note — *"this store has no cardinality recorded for that predicate"* —
would stop firing because the sentence had been made false rather than because anyone had
answered the question. That note is the only warning that two live values might be a
contradiction rather than a legitimate multi-valued slot. Three tests in
`tests/test_server.py` caught the trade.

**Predicates are matched on content tokens, and comparison frames are excluded.** Two
refinements measured after the above, worth their own paragraph because the first is not
what "entailment" suggested.

`date of birth` folds to `born_on`, whose spoken form is "born on" — and questions say
"when was X born". The predicate was in the question and the *preposition* was not, which
failed 79% of compositional questions. Matching the content tokens of a predicate name
(`STOPWORDS` dropped, and **all** remaining tokens required, so `country_of_citizenship`
still needs both) took the trigger rate from 21% to 38.8%.

That alone would have been a net loss. It also fired on a third of `bridge_comparison`,
where the walk costs 13.7 points: "which film has the director died later, A or B" names
`director` and `died_on` — two predicates, a chain by that measure — while being two
independent lookups whose answers are compared. `intent.is_comparison()` suppresses on the
**disjunction** rather than on a list of comparative words, because "earlier", "first" and
"younger" are what this corpus happens to say and a rule built from them would be fitted
to it. That took `bridge_comparison` back to 0.0% with no cost to `compositional`.

One false positive found on the way: `born_in` and `born_on` share the content token
`born`, so "when was Alice born" named two predicates and read as a chain — from one word.
Matches are now deduplicated by what the question said rather than by how many predicates
answer to it.

**Answers and derivations move together.** On `chained`, the leg is worth +20.1 points of
answer recall and **+20.3 of chain recall** — 28.2% → 48.3% and 25.5% → 45.8%. Ungated the
two columns nearly meet, 67.3% against 65.5%: almost every answer the walk finds arrives
with every triple that supports it. That is the property the library is for, and it is the
one worth quoting.

<div data-type="panel-warning">

**This paragraph previously said the opposite, and the error was in this harness.**
`place_of_birth` is an alias of `born_in` and `date_of_birth` of `born_on`, so a claim
written from 2Wiki evidence is *stored* under the canonical name. `chain` compared the raw
gold predicate against the returned row and never matched for either — 6,624 of this
corpus's triples. The failure was one-sided: `answer` matched on the object alone and kept
scoring, `chain` needed the predicate too and silently failed.

So chain recall read ~13 points low everywhere, and the gap between the two columns looked
like a finding about retrieval — "the walk brings back answers without their evidence" —
when it was this file comparing two spellings of one predicate. `Sample.fold_to_store()`
now folds the gold predicates the way the store wrote them.

The tell was there and was misread: chain recall sat still through three changes while
answer recall climbed. That pattern was evidence about the measurement, not about the
product.

</div>

**The one question type no rule could reach, and the model call that reaches it.**
`inference` questions ask "who is the maternal grandfather of X" over evidence
`(X, mother, Y)` and `(Y, father, Z)`. They name a *derived* relation and no stored
predicate at all, so every rule above — which counts predicates a question says out loud —
found at most one and never ran the walk. On that family the leg was worth nothing.

`grandfather` is not a synonym for `father`; it is `father` composed with `father`, and no
string match gets from one to the other. What the gate needs is the single fact that the
term **is** a composition — not which predicates it composes from, since its question is
only ever "is this a chain".

`retrieve/compose.py` asks a model that once, about a **vocabulary**, and never about a
query: given the predicates a store uses, which English relation terms compose from two or
more of them. The read path does a set-membership test against the answer.
`retrieve/intent.py` promises to be model-free and `hybrid.py` promises reproducible
retrieval; a search that could block on an API call breaks both, which is why the
acquisition is shaped like `resolve_predicate` — pay once per vocabulary, keep it, never
pay again.

Measured on **all 1,549** `inference` questions at k=12, terms acquired from a live model
(`nvidia/nemotron-3-ultra-550b-a55b` via OpenRouter) against the store's own seven
predicates:

```
  no terms                    49.0% answer / 45.1% chain
  terms from the live model   80.3%        / 78.6%
```

**The floor matters more than that number.** A minimal list of four words any model would
produce — `grandfather, grandmother, uncle, aunt` — is worth 73.9% / 71.8% on its own. The
feature does not need a good list, only a plausible one. A hand-written full kinship list
reaches 81.4% / 79.7%, so the live model landed within a point of it.

The model was given the store's actual vocabulary, not a kinship prompt, and returned
`academic grandfather` for `doctoral_advisor` — the generalisation past kinship that no
hand-written list would have contained.

**False positives are negligible**: the terms appear in 0.10% of `comparison`, 0.62% of
`bridge_comparison` and 0.13% of `compositional` questions, and the `bridge_comparison`
ones are disjunctions that `is_comparison` catches first.

`compositional`, `comparison` and `bridge_comparison` are unchanged, so it reaches the
family it was built for and nothing else. A disjunction is still a comparison even when it
names a derived relation — "whose grandfather was born earlier, A or B" is two two-hop
lookups compared — so `is_comparison` runs first.

It is **opt-in and absent by default**: a backend without `compose_relations` yields no
terms and the gate keeps the rule it had, which is what every release before this shipped.
The terms are not persisted, so a server pays once at startup; `docs/ROADMAP.md` carries
why, and it is that the two obvious places to put them are both wrong.

**What it still does not reach.**
`inference` gains **nothing** — 46.6% through every change in this series, 46.7% on the
current table — and the reason is not a bug. Those
questions ask "who is the maternal grandfather of X"; the evidence is `(X, mother, Y)` and
`(Y, father, Z)`, and the question names neither `mother` nor `father`. It names a
*derived* relation. Matching a question's words against stored predicate names cannot
bridge `grandfather` to `mother` + `father`, and no longer word list closes that — it
needs synonymy or entailment, which is a model rather than a lookup. All of the gain here
is in `compositional`, where the question does say the predicates out loud: 24.0% → 32.1%.

`bridge_comparison` chain recall is 0.0% for `search` and 2.2% ungated. Those chains are
four hops and `graph_depth` ships at 2, so that row measures the depth bound rather than
traversal — the same caveat the synthetic benchmark's three-hop rows carry. It read 23.8%
while every claim carried an edge. The only edges removed since are the four date
relations, so that recall was reached through a shared date rather than through the
chain, and with dates as values it cannot be.

**What this does not measure.** Retrieval given claims. The write path never runs, so
nothing here says anything about extraction, which remains the bottleneck. Quote this as
evidence about the graph leg or not at all.

Contamination is a smaller problem here than the note in `bench/evalkit.py` describes, and
structurally rather than by luck: scoring is R@k against gold evidence under `NullLLM`, so
there is no reader that could have memorised an answer. A contaminated reader inflates
end-to-end accuracy, which this file does not compute.

These numbers are not comparable to the 2Wiki leaderboard, which retrieves from a
per-question candidate set. That is reading comprehension; this is recall against 26,403
competing facts.

### Anchoring: abstention without a threshold, measured on the Agent Memory Benchmark

`search(anchored=True)` keeps only the rows the question names an entity of — the folded
subject or object key, every content token present — or that the graph leg reached by
walking out of such a row, and `Explanation.anchor` reports which on every result
(`memvara/retrieve/anchor.py`). It is the answer to the `irrelevance` half of
[#129](https://github.com/memvara/memvara/issues/129). The rows below come from the same
adapter at four settings of the two switches, over dataset v1, byte-identical on repeat.
The bottom row is now published as its own system, `--system memvara-anchored`, beside the
default one — `--system memvara` still means the library's shipped defaults, so the number
somebody may already have quoted keeps meaning what it meant.

```
  configuration                 overall  retrieval  irrelevance  multi_hop  negative
  shipped                         92.0%      64.3%        50.0%        1/6       3/6
  shipped, anchored=True          93.0%      57.1%        83.3%        0/6       5/6
  read_w_graph=1.0                92.0%      64.3%        50.0%        1/6       3/6
  read_w_graph=1.0, anchored      94.0%      64.3%        83.3%        1/6       5/6
```

**Two of the three open negatives are caught, and no threshold reaches either.** *Where
does Oscar live* returns nothing because no row is about Oscar. *Which region is Project
Chronos deployed to* is the case the issue singled out — it scores 0.450, above two
genuine answers — and it returns nothing because `project` alone does not name `Project
Atlas`. The third, the reporting service's authentication strategy, is correctly *not*
caught: the store holds who owns the reporting service, that row is about the entity
asked about, and telling it from the answer is a question about the predicate, which
anchoring does not judge.

**The one point it costs at the shipped defaults is a lucky hit, and the walk earns it
back.** *In which city is Bob's employer headquartered* is answered by plain search from
`Globex/hq_city=Munich`, which shares no entity with the question and sits a hair above
`Initech/hq_city=Austin`; anchoring hands back Bob's own rows instead, and the adapter
answers `Globex`. With the graph leg on, the walk out of `bob/works_at=Globex` reaches the
same row and marks it `"path"`, so the filter keeps it. That is the shape the two halves
of the issue share: on a negative "the top hit is not about what you asked" means *never
told*, and on a chain it means *one more hop*, and the path anchor is what tells them
apart.

A deployment can now reach that bottom row without writing Python. `MEMVARA_ANCHORED=1`
makes `anchored` the default on the three read tools of an MCP server, and
`MEMVARA_READ_W_GRAPH=1.0` switches on the leg that pays for it; before those existed both
switches were constructor arguments, so every MCP deployment — the hosted product included
— ran the top row and had no way not to. `docs/DEPLOY.md` has the operator's version of
the trade.

`multi_hop` does not move from anything in the read path, and the issue's own measurement
says why: the answer is already retrieved — rank 1 for the Atlas question at the shipped
defaults — and the adapter takes rank 0 and stops. The intent-gate repair in this same
series (#150: a question names a predicate in whatever form it inflects it, and a chain
that also names an instant keeps the walk) changes which questions walk and not that row.
`bench/multihop.py` at 1,000 staff is byte-identical before and after, because its
questions already name their predicates in the stored form.

### Anchoring on public questions, with a holdout split

The section above measures anchoring on six negatives this repository wrote. `bench/anchoring.py`
measures it on a thousand it did not.

Nothing in it is authored. 2WikiMultihopQA is loaded as `bench/twowiki.py` loads it, the
questions are split, and only the first half's triples are written. A question from the
held-out half is then a real question from a public set whose facts the store genuinely does
not hold. One filter keeps that honest: 2Wiki reuses entities heavily, so a held-out question
counts as a negative only when it names no entity the store holds at all — refusing a question
about somebody the store knows would be wrong rather than right.

3,000 questions ingested, 1,000 of them scored for cost, 332 negatives. Every arm reads at a
pinned instant, and the table is identical on repeat. The three `graph leg` rows were re-measured on
2026-09-14, when `bench/twowiki.py` began declaring the corpus's relations: the walk now
follows the declared edges only, so those rows moved by about a point and the other six did
not move at all. The run takes about 85 minutes on a developer machine.

```
  configuration          k  answer found  correctly silent
  shipped                5         49.8%              0.0%
  shipped               12         55.3%              0.0%
  shipped               25         57.6%              0.0%
  anchored               5         39.0%            100.0%
  anchored              12         39.1%            100.0%
  anchored              25         39.1%            100.0%
  anchored + graph leg   5         55.0%            100.0%
  anchored + graph leg  12         54.8%            100.0%
  anchored + graph leg  25         55.0%            100.0%
```

**The shipped configuration is silent on none of them, and anchoring is silent on all of
them.** Every one of the 332 questions the store was never told the answer to comes back with
rows by default, at relevances that look like any other match; with `anchored` set, none of
them does. That is the behaviour anchoring exists to produce, and this is the first
measurement of it on questions written by somebody else.

**Anchoring alone costs about a sixth of the legitimate answers**, and the graph leg pays most
of it back: 39.1% against 54.8% at k=12, where the shipped figure is 55.3%. A 2Wiki question
names entities the anchor then has to match, and a question whose answer sits one hop away
reaches it through the entity the question does name. At k=5 the pair is ahead of the shipped
configuration on both columns at once.

**The recovery is a property of this corpus, not of the setting, and that is the condition to
read the table under.** 2Wiki loads clean triples through `remember()`, so its join rate is
high and the walk has somewhere to go. On a store whose claims all hang off one subject —
what `memory_stats` calls a star, and what one user's own sentences produce — the walk has
nowhere to go and pays nothing back. A deployment should read its own join rate before reading
these numbers as advice, which is why neither setting ships on.

**100% is a statement about this filter, not about abstention in general.** These negatives
name no entity the store holds, which is the case anchoring is built to catch and the reason
it catches all of them at every depth. The case it does not catch is a question that names an
entity the store *does* hold and asks about an attribute it does not — the reporting service's
authentication strategy, in the Agent Memory Benchmark's wording. Nothing here measures that,
and anchoring is not the mechanism that would answer it.

An earlier version of this table read 42.0% to 50.6% instead, from a negative set of 1,018.
That filter decided "the question names this entity" by substring, where the mechanism it was
measuring folds the entity to a key and requires every word of the key to appear as a token.
The two rules disagree in both directions, and the sets are not nested:

- **698 questions the substring rule called negatives are not.** A key stored as
  `project atlas` is named by a question saying "Atlas Project", which contains no such
  substring. Those questions anchored, returned rows, and were counted as failures to
  abstain — which is the whole of the missing abstention.
- **12 genuine negatives the substring rule threw away.** `iran` matches inside `Piranha`
  and `france` inside `Francesco`, so a question naming neither was treated as naming both.
  The four-character floor did not stop either, since both are longer than four characters.

`negatives()` now calls the same `entity_key` / `key_words` / `query_tokens` path that
`anchor_of` does, which is the only way the filter and the thing it measures can agree.

### The temporal leg, and the abstention that is the actual finding

`w_temporal > 0` adds a fourth leg over **raw turns**: the ones nearest in time to the
instant the search was asked about, ranked on *when* and reading no text at all. It is the
answer to "what was going on around then", whose only content words — `when`, `around`,
`then` — the analyzer drops and the embedder maps onto nothing. **It also ships at 0.0.**

```bash
PYTHONPATH=. python3 bench/longmemeval.py --score retrieval --share-store --w-temporal 1.0
```

| LongMemEval oracle, R@12 | baseline | + temporal, no floor | + temporal, with floor |
|---|---:|---:|---:|
| temporal-reasoning | **66.6** | 64.2 | **66.6** |
| knowledge-update | 91.0 | 91.0 | 91.0 |
| multi-session | 65.5 | 65.2 | 65.0 |
| **all** | **70.4** | 69.7 | 70.3 |
| all MRR | 62.0 | 57.4 | 62.0 |

**The middle column is the finding, and it is about fusion rather than about time.** With
no instant given the anchor is *now*, and these transcripts are dated years earlier, so
every turn scored a proximity around 0.005 — and RRF reads *positions*, so a leg with no
opinion still contributed rank 0, rank 1, rank 2. A ranking assembled from nothing is not
a weak ranking, it is a fabricated one, and fusion cannot tell the difference. That cost
**2.4 points of temporal-reasoning R@12 and 4.6 of MRR**.

The other two legs have had the matching guard all along: the vector leg abstains on a
zero-norm query, the lexical leg on a query with no content terms. `MIN_PROXIMITY` gives
this one the same rule — nothing within a half-life of the anchor and it does not vote —
and the loss goes to zero.

What it does not do is clear the bar, and the reason it could not be read off these rows
was that nothing passed an instant: both runners called `search(question, k)` with the
question as prose, so the anchor was the wall clock, every archived turn sat years from
it, and the leg abstained on every question. `bench/longmemeval.py` now hands retrieval
the question's own day, and the next section measures what that is worth. The leg still
ships at `0.0`.

**The blocking dependency here is ingestion, not retrieval.** Both public instruments are
blind to the graph leg for the same reason the `memvara` demo arm produces zero claims —
see [What the fast path does not
catch](DESIGN.md#what-the-fast-path-does-not-catch-measured). Until the offline write path extracts
from ordinary prose, no public retrieval number can move on this.

### The anchor the leg never had, measured

Every row above was produced with no instant passed to retrieval. LongMemEval dates every
question, and the harness put that date in the reader's prompt and nowhere else, so the
memory was read as of now while the reader was told it was 2023. `bench/longmemeval.py`
now passes the last second of the question's day as `valid_at` on both reads a question
makes, and `--no-anchor` withholds it and reproduces every row published before this.

Measured on 2026-09-14 on the `s` split with one shared store: 199,499 turns ingested once
in 490 seconds, then six configurations scored on that one store. Each configuration is
what this command prints for the same flags, and the harness ingests once per invocation,
so scoring them together only saves time:

```bash
PYTHONPATH=. python3 bench/longmemeval.py --dataset s --score retrieval \
    --recall-at 1,5,12,20 --share-store --embedder hashing [--no-anchor] [--w-temporal W]
```

| configuration | all R@12 | temporal-reasoning R@12 | all MRR | questions the leg voted on |
|---|---:|---:|---:|---:|
| `--no-anchor --w-temporal 0` | 35.1 | 23.1 | 24.6 | 0 of 500 |
| anchor, `--w-temporal 0` | **40.9** | **38.9** | **28.8** | 0 of 500 |
| anchor, `--w-temporal 0.25` | 41.2 | 38.9 | 25.8 | 52 of 500 |
| anchor, `--w-temporal 0.5` | 40.7 | 38.4 | 24.0 | 87 of 500 |
| anchor, `--w-temporal 1.0` | 28.3 | 19.2 | 16.6 | 447 of 500 |
| `--no-anchor --w-temporal 1.0` | 35.1 | 23.1 | 24.6 | 0 of 500 |

**Read the first row first.** It reproduces [the shared-store
baseline](#episode-retrieval-on-a-shared-store) to the decimal, which is what makes the
five rows below it attributable to the configuration rather than to the shortcut.

**Without an anchor the leg ranks nothing**, on any of the 500 questions, at any weight.
The last row is identical to the first in every cell. The section above could only say
that this was expected; here it is counted.

**Most of the gain comes from the anchor rather than from the leg.** With the leg still
off, the question's day takes overall evidence recall from 35.1 to 40.9 and
temporal-reasoning from 23.1 to 38.9. `valid_at` is the world clock every leg filters on,
so most of that gain is a shared store no longer returning sessions dated after the
question was asked — other questions' sessions leaving the pool, rather than better
ranking. A per-question store has far less for the filter to remove, and that
configuration is not measured here.

**The leg itself still does not pay.** At 0.25 it ranks something on 52 questions, buys
0.3 of recall and loses 3.0 of mean reciprocal rank. At 1.0 it ranks something on 447 and
overall recall falls to 28.3. The shipped default stays `0.0`.

**What it costs.** Scoring the 500 questions took 302 seconds with the leg off and 354
with it at 1.0 on the same store, about 17% more, because every temporal or open query
runs `episodes_near`, whose `ORDER BY ABS(ts - ?)` sorts everything the scope matched. On
LOCOMO the leg is inert for a different reason — no question there carries a date — and
`--w-temporal 1.0` changes no cell of that table at all, at a median retrieval latency of
2.9 ms against 2.7.

### Episode retrieval on a shared store

The configuration below had never been measured before 2026-09-06, and it is the baseline
any future change to the episode index should be read against. It is **not** comparable to
the `66.6` temporal-reasoning figure above: that one is the `oracle` split, which hands the
system only sessions containing the answer, while this is `s`, where 19,195 sessions are
loaded into one store and distractors are the point.

```bash
PYTHONPATH=. python3 bench/longmemeval.py --dataset s --score retrieval \
    --recall-at 1,5,12,20 --share-store --embedder hashing
```

Evidence recall — the annotator-marked table, indifferent to wording. 500 questions,
199,499 turns ingested, 1,168 claims:

| category | n | R@1 | R@5 | R@12 | R@20 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| single-session-user | 64 | 12.5 | 32.8 | 46.9 | 56.2 | 23.2 |
| single-session-assistant | 56 | 44.6 | 48.2 | 55.4 | 57.1 | 46.9 |
| single-session-preference | 30 | 3.3 | 3.3 | 6.7 | 13.3 | 4.1 |
| multi-session | 121 | 4.6 | 19.7 | 29.2 | 35.7 | 21.0 |
| knowledge-update | 72 | 19.4 | 50.7 | 66.7 | 71.5 | 52.0 |
| temporal-reasoning | 127 | 3.8 | 14.6 | 23.1 | 32.7 | 14.1 |
| abstention | 30 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **all** | **500** | **11.7** | **25.6** | **35.1** | **41.6** | **24.6** |

The `abstention` row is 30 questions that have no answer in the haystack, so there is no
evidence to retrieve and 0.0 is the correct score rather than a failure. It is listed
because the harness reports it separately and never folds it into the type it was drawn
from, and because leaving it out would make the six named categories sum to 470 against an
`all` row of 500.

**`--share-store` means one store for every question**, so retrieval can reach another
question's sessions. The harness prints that warning itself and it is repeated here: this is
a number for comparing two builds of memvara to each other, not for quoting beside a
published LongMemEval score.

Adding the cross-encoder (`--rerank 20 --reranker cross-encoder`) takes overall R@12 from
**35.1 to 40.1** and temporal-reasoning from **23.1 to 30.2**, on the same store. `--rerank`
reached this runner on 2026-09-06; before that it existed only on `bench/locomo.py`, so the
reranker had never been measured on LongMemEval at all.

### Dating the episode index, which did not pay

Rendering each turn's date into the text the retriever ranks — the episode FTS row, the
episode vector and the cross-encoder's input — was built against the baseline above and
reverted. Two formats, six arms, ingest byte-identical across all of them.

What the cross-encoder is worth on temporal-reasoning R@12, by how much date text sits in
front of the turn:

| index | cross-encoder gain |
|---|---:|
| undated | **+7.1** |
| `June 2023`, two tokens | +5.8 |
| `Thursday, 15 June 2023 (2023-06-15)`, eight tokens | +4.3 |

Monotonic in prefix length, which is the mechanism rather than a coincidence: BM25
normalises for document length, so every token added to the index makes every turn a
slightly worse match for everything, while the date tokens are matched by only a few
questions. Overall R@12 moved +0.5 with the reranker off and −0.3 to −0.7 with it on.
Temporal-reasoning rose at R@12 and fell at R@1, R@5, R@20 and MRR in both formats.

The reasoning behind the change and what it does *not* rule out — the claim side was never
built — are in [`docs/ROADMAP.md`](ROADMAP.md).

---

## A design comparison (synthetic, self-authored)

Not an external benchmark. One workload, n=1, written by the same people who wrote the
system being measured and the system it is measured against. Read this section as an
illustration of a mechanism, not as evidence of superiority.

`PYTHONPATH=. python3 bench/compare.py` — 105-turn transcript, 21 turns carrying a
durable fact, 10 distinct facts, several revised two or three times:

| metric | mem0-style | memvara |
|---|---:|---:|
| LLM calls on the write path | 126 | **2** |
| Current value stored correctly | 10/10 | 10/10 |
| **Stale values left live** | **7** | **0** |
| Local compute | **4 ms** | 11 ms |

**Where the stale-value result actually comes from.** An earlier version of this document
claimed those seven contradictions were "invisible to top-k adjudication." That was
wrong, and the benchmark disproves it: sweeping the baseline's `top_k` from 1 to 1000
changes nothing, because the conflicting memory is returned in the candidate list every
time. What kills them is the baseline's similarity **threshold** (0.75) — competing
values embed at 0.52–0.74, just under it. That threshold is a tuning choice and the
result is sensitive to it: at 0.5 the baseline also holds zero stale values; at 0.9 it
holds eleven. The honest claim is not "top-k loses conflicts" but **"a keyed lookup has
no threshold to get wrong"** — which is a claim about determinism, not recall.

**The call-count gap is mostly an ingestion-granularity choice.** Memvara receives the
whole transcript in one `add()` and batches extraction; the baseline is charged per turn.
At equal per-turn granularity it is 126 vs 17, not 126 vs 2. The gap also scales linearly
with the chitchat ratio, which is a parameter we picked: 1:0 → 21x, 1:4 → 63x, 1:12 →
147x, 1:100 → 1071x, with identical information content at every point.

**Memvara loses the local-compute row** — roughly 3x slower per operation, because it does
strictly more work (FTS indexing, reconciliation, bitemporal filtering). That trade is
worth it only when model calls dominate, which is the normal case but not a universal one.

**What this does not measure:** end-to-end answer quality. Both systems are driven by the
same perfect extraction oracle, which neither would have in production. 9 of the 10
predicates ship pre-seeded in the registry with the right cardinality, so the benchmark
never exercises the path where an unknown predicate defaults to multi-valued and
accumulates. The LOCOMO and LongMemEval numbers above do not close that gap either: they
measure retrieval, not answers. The apparatus for scoring answers end to end is
[below](#answer-quality-end-to-end-an-authored-corpus-an-agent-as-the-reader); it exists
now, it has been run once, and the run is a sanity check rather than a benchmark.

### Throughput

`PYTHONPATH=. python3 bench/perf.py` — single process, in-memory store, no LLM:

Single-shot point estimates on one loaded developer machine, no warmup, no repetition,
no variance reported — treat as order-of-magnitude, not as a regression baseline.

| @ 8,000 claims | per op | scaling per 4x data |
|---|---:|---|
| `remember()` (structured write) | 0.12 ms | flat |
| `add()` (fast path, no LLM) | 0.50 ms | flat |
| search k=10 | 2.1 ms | sub-linear |
| consolidation, cold sweep | 457 ms | linear |
| consolidation, steady state | 273 ms | linear |

Two algorithmic fixes got it there, both found by profiling rather than guessing:

- **The FTS index was keyed on an `UNINDEXED` column.** `DELETE FROM claims_fts WHERE
  claim_id = ?` on every write was a full scan of the text index, making N writes over N
  rows **O(n²)** — it dominated everything else at 80% of consolidation time. Mirroring
  the claim's rowid into the FTS table makes the delete an indexed lookup. Consolidation
  went from 4.8 s to ~460 ms at 8k claims, and from degrading ~11x per 4x of data to
  ~4x — i.e. from quadratic to about linear, which is the floor for a full sweep.
  (This required switching `INSERT OR REPLACE` to an upsert: REPLACE assigns a *new*
  rowid, which would orphan the index entry it is keyed on.)
- **N+1 query patterns.** Retrieval hydrated every fused candidate with its own
  `SELECT`, and consolidation re-embedded every claim's text on every sweep — against a
  hosted embedder that is one network round trip per claim, per run. Both now read in
  bulk, and consolidation reuses the vectors already on disk.

Exact vector search over a scope is O(|scope| · d) and that is the floor — the matmul is
already BLAS. Beating it requires an approximate index (HNSW/IVF), which trades recall
for speed and belongs behind the `Store` protocol, not in the default path.

Read [`bench/baseline.py`](../bench/baseline.py) before quoting any of this: the comparison
target is a reimplementation of mem0's *documented architecture*, not the mem0 package,
and both systems are driven by the same extraction oracle so the comparison isolates
architecture from model quality. The benchmark does **not** demonstrate the hybrid-retrieval
advantage — the offline `HashingEmbedder` is character-n-gram based and therefore unusually
good at exact tokens, so the vector-only baseline finds them too. That claim needs a real
semantic embedder to test, and is stated here rather than claimed.

### One large scope

`bench/perf.py` spreads its claims over fifty users, so no scope it searches holds more than
a few hundred rows. `bench/scale.py` measures the other end: one user's scope holding all
199,499 LongMemEval-S haystack turns, deduplicated by session, and 100,000 synthetic claims,
written straight through the store with the hashing embedder. It times each store read a
search runs, over 50 questions or five fixed claim queries, and then
`search(k=12, include_episodes=True)` as a whole.

```bash
PYTHONPATH=. python3 bench/scale.py --path /tmp/scale.db
```

Both columns time copies of one store file, built once, on a 4-core Linux container, one
run after the other. "Before" is `main` on 2026-09-24. "After" is the change that asks for
each scope's candidates separately, reads the turn list from the covering index `ep_cover`,
joins the text index to its table on rowid, and looks up vector rows in one vectorised pass:

| read | before, median | after, median | before, p95 | after, p95 |
|---|---:|---:|---:|---:|
| `candidate_ids` | 86.7 ms | 70.6 ms | 124.5 ms | 77.4 ms |
| `episode_candidate_ids` | 265.5 ms | 88.5 ms | 288.2 ms | 106.4 ms |
| `lexical_search` | 323.3 ms | 120.0 ms | 360.6 ms | 131.8 ms |
| `lexical_search_episodes` | 140.0 ms | 55.4 ms | 423.0 ms | 161.0 ms |
| `vector_search` | 176.2 ms | 158.9 ms | 274.0 ms | 216.1 ms |
| `vector_search_episodes` | 391.5 ms | 215.1 ms | 447.5 ms | 253.3 ms |
| **`search()`** | **737.6 ms** | **469.8 ms** | **1,071.0 ms** | **556.5 ms** |

Every read returns the same rows before and after; only how SQLite reaches them changed.
The claim vector leg moved least. Measured on its own, about half its time is the
candidate list, which no covering index answers because it reads the claim's state
columns, and most of the rest is the product over 100,000 vectors, which is the floor for
an exact index. The lexical legs are handed what
`HybridRetriever` hands them, the query reduced to its content words. The mechanism behind
each row, and the first open that builds `ep_cover`, are in
[`docs/INTERNALS.md`](INTERNALS.md) under *Reading a whole scope, and joining the text
index*.

**Each scope's turns, kept in memory.** The next change keeps each scope's turn list and
its matrix rows between searches, and empties them on every commit. Two stores this time:
the one above, and the LongMemEval-S turns written with the benchmark harness, 189,520 in
one scope with 1,168 claims, where the turns are nearly all of the work. "After a write"
empties the lists before each search, which is what the first search after any commit
pays. "Before" is the change above, timed at the start and again at the end of the run, and
both columns give the two runs where they differ:

| read | before, median | after, median | before, p95 | after, p95 |
|---|---:|---:|---:|---:|
| 100,000 claims: `vector_search_episodes` | 219.8 / 229.8 ms | 53.7 ms | 249.8 / 263.3 ms | 66.0 ms |
| &nbsp;&nbsp;after a write | | 238.2 ms | | 270.0 ms |
| 100,000 claims: **`search()`** | **449.7 / 471.5 ms** | **284.1 ms** | **567.6 / 570.9 ms** | **399.0 ms** |
| &nbsp;&nbsp;after a write | | 471.4 ms | | 592.1 ms |
| 1,168 claims: `vector_search_episodes` | 182.8 / 181.0 ms | 36.5 ms | 221.8 / 218.8 ms | 40.2 ms |
| &nbsp;&nbsp;after a write | | 202.7 ms | | 217.3 ms |
| 1,168 claims: **`search()`** | **257.5 / 258.0 ms** | **107.2 ms** | **360.3 / 353.1 ms** | **223.4 ms** |
| &nbsp;&nbsp;after a write | | 267.9 ms | | 381.5 ms |

The rows that come back are the same: 50 searches on each store returned the same 600
rows with the same scores under both builds. The first search after a write costs slightly
more than a search did before, because rebuilding a list also reads each turn's `ts` and
builds three arrays, and every search after it, until the next write, costs the
"after" column. On the turn-heavy store, what is left of a search is mostly the lexical leg
over turns: 54 ms at the median and 158 ms at the 95th percentile, when a question's
content words are common.

**Two legs at once.** The change after that runs each stage's vector leg on a pool thread
while the stage runs its lexical leg on the calling thread. Same two stores, timed three
times in a row: this change, the change above, and this change again.

| read | before, median | after, median | before, p95 | after, p95 |
|---|---:|---:|---:|---:|
| 100,000 claims: **`search()`** | **285.7 ms** | **254.1 / 245.1 ms** | **392.5 ms** | **369.0 / 351.9 ms** |
| &nbsp;&nbsp;after a write | 473.8 ms | 479.9 / 461.4 ms | 612.4 ms | 526.8 / 503.1 ms |
| 1,168 claims: **`search()`** | **100.2 ms** | **65.1 / 63.6 ms** | **205.1 ms** | **188.6 / 170.3 ms** |
| &nbsp;&nbsp;after a write | 266.4 ms | 239.6 / 243.1 ms | 389.5 ms | 256.6 / 270.4 ms |

The rows that come back are the same: 50 searches on each store returned the same 600 rows
with the same scores. A stage now costs about its longer leg. Timed on their own, the turn
stage's two legs took 54 and 56 ms on the first store, 111 ms one after the other and
61 ms together, and the claim stage's took 167 and 126 ms for five claim queries, 291 ms
one after the other and 175 ms together. A whole search gains less than that on the first
store because its claim stage has little to overlap there: the LongMemEval-S questions
share almost no words with the synthetic claims, so the claim lexical leg takes 0.1 ms.

The first search after a write gains least. Timed on its own in the same runs, the vector
leg that rebuilds the scope's turn list took 270 to 282 ms in this build against 250 ms in
the one before, and `episode_candidate_ids` 110 to 118 ms against 96 ms. That difference is
not the legs. `bench/scale.py` runs on one thread until a search starts the pool, and both
reads run 11 to 16% slower in any process that has started a second thread, even one that
opened no connection and has exited. A host that serves requests from more than one thread
pays that with or without this change.

---

## Answer quality, end to end (an authored corpus, an agent as the reader)

Every number above measures **retrieval** — did the right claim come back, ranked where it
should be. None of them measures **answers**: whether an agent reading memvara's output
tells the customer the right thing. [`demo/`](../demo/) is the apparatus for that, and
[`demo/README.md`](../demo/README.md) is its full documentation.

```
demo/scenario.py    64 turns of one customer's support history, and 20 questions
demo/distractors.py generated tickets that scale the history without moving any fact
demo/baselines.py   five context-building arms
demo/hosted.py      the two memvara arms against a memvara-cloud project
demo/competitors.py two more arms, mem0 and Supermemory, off unless asked for
demo/harness.py     a blinded dump/answer round trip over those arms, and the scoring
```

The corpus is one customer's account from January to August 2026. Six facts move across
seven changes, and **they do not all move for the same reason.** Five of the changes are
`ended` — the plan (twice), the delivery address, the billing address, the contact
preference: true once, then true no longer. Two are `retired` — a mistyped mobile number
and a misread serial: never true at all. Every superseded value is deliberately
re-surfaced *after* the value that replaced it, so recency and emphasis both point at the
wrong answer. Each question carries an authored `gold`, the specific wrong answer a
single-clock store gives as `trap`, and which clock closed as `closure`, so the two
failures can be counted apart. The golds were written by hand from the transcript, never
recorded from a memvara run.

### The offline run, which is one command and repeats exactly

```bash
PYTHONPATH=. python3 demo/harness.py --reader stub
```

Every arm, every question, in one process, with no key. It is deterministic, so two runs
of it differ only where the library does — which is what makes the apparatus something a
test can hold and a bisect can walk. `test_the_offline_run_is_identical_twice` pins it.

**Read nothing about answer quality out of it.** The reader is `evalkit.StubReader`: it
returns the line of the retrieved context with the most words in common with the question.
Its `correct` column is a property of the corpus and the arms, and the run prints two
banners saying so. The rows below, and the table further down, are the numbers.

### Context size, which is deterministic and reproducible

Either command builds the contexts. This table is a property of the corpus and the arms
and comes out the same on every run:

```
  arm                 mean chars  max chars  mean ~tokens  items used / turns seen
  ------------------  ----------  ---------  ------------  -----------------------
  none                         0          0             0               0.0 / 60.8
  full_transcript           9803      10263          2451              60.8 / 60.8
  naive_rag                 2329       2846           582              12.0 / 60.8
  memvara                   2074       2489           519              12.0 / 60.8
  memvara_structured        1772       2241           443              12.0 / 60.8
```

`~tokens` is characters ÷ 4, an estimate and not a tokenizer. The `memvara_structured`
row grew from 1,721 to 1,772 characters when the arm moved to `recall(valid_at=)`; the
agent run below was made at the earlier size.

### The hosted run, which is the measurement

```bash
export ANTHROPIC_API_KEY=...
PYTHONPATH=. python3 demo/harness.py --reader anthropic --judge llm \
    --model claude-opus-5 --effort low --max-tokens 4096 --thinking adaptive \
    --checkpoint runs/hosted.checkpoint.jsonl --concurrency 4 --out runs/hosted.jsonl
```

The same arms and questions, with a model behind an API as the reader and a second call
to it as the judge. The model id, effort, output budget and thinking setting are printed
under the report's title exactly as sent, the cost is priced from the usage the provider
reported, and answers that never finished are counted apart from wrong ones.
`--checkpoint` makes the run resumable and `--concurrency` shortens it;
[`demo/README.md`](../demo/README.md) has every flag. No run with it has been recorded
yet: the scores below are the agent run.

### The second corpus size

The token argument is a slope — retrieval context flat in corpus length, transcript
context linear — and the authored corpus is one point on it. `--corpus-scale 10` pads the
history with generated support tickets (`demo/distractors.py`) that never name a value a
question is about, so the questions, golds and traps are unchanged and only the haystack
grows. At that scale, deterministically:

```
  arm                 mean chars  max chars  mean ~tokens  items used / turns seen
  ------------------  ----------  ---------  ------------  -----------------------
  none                         0          0             0              0.0 / 607.5
  full_transcript          92053      96897         23013            607.5 / 607.5
  naive_rag                 1891       2838           473             12.0 / 607.5
  memvara                   1596       2424           399             12.0 / 607.5
  memvara_structured        1530       2204           383             12.0 / 607.5
```

The transcript arm grows 9.4× while the three retrieval arms stay under their cap.
Whether they still surface the evidence among ten times more turns is the hosted run's
question at this scale, and that run has not been made yet.
[`demo/README.md`](../demo/README.md#two-corpus-sizes) has the constraints the generated
turns are tested against.

### The memvara arms against the hosted service

`--memory hosted` points the two memvara arms at a memvara-cloud project, through the
client a customer uses, so the run can measure the service rather than the library alone.
The other three arms use no store and are unchanged. It writes to a project made for it:
the credentials file is refused if it is the machine's default one, holds the same key, or
holds a key that the server says reaches the same tenant. The tenant is looked up only
when the two files name the same project, and a failed lookup also refuses the file.

Two differences are not incidental and are printed above the report's tables. A hosted
project cannot be sent the support schema, so the arm closes single-valued slots itself
rather than relying on a declared cardinality, and the built-in vocabulary files `plan`
under its alias `goal`. And `POST /v1/recall` has no time axis, so the four dated questions
are read with `search(valid_at=)` and rendered by the library's own recall renderer.
Extraction and the episode cap are the deployment's, so each context records how many
claims its scope held when it was read.
[`demo/README.md`](../demo/README.md#the-memvara-arms-against-the-hosted-service) has the
whole of it.

### Two other systems, as arms

`--arm-mem0` and `--arm-supermemory` add a competitor arm each. Both are off by default
and each needs something a fresh checkout does not have, so neither can affect the offline
run.

**mem0** needs only the package (`pip install mem0ai`). It is driven by the oracle
`bench/mem0_real.py` uses, so it receives exactly the ground-truth facts
`memvara_structured` receives, with perfect extraction recall — better than any real model
— which leaves architecture as the only thing that differs. mem0 2.x's add path emits only
`ADD`, so a value that moves is held beside the value that replaced it.

Measured on the authored corpus with mem0ai 2.1.0, over the fifteen questions whose answer
has a superseded value to be wrong with, mem0 asserts that superseded value as a current
fact in **13 of 15**; both memvara arms assert it in **0 of 15** and carry it only under
the episode header, which tells the reader those lines are things that were said and are
unverified. That is a statement about where a value appears in a prompt, not about whether
a reader was fooled — no model was asked. The judged comparison is the missing half and is
[item 1 of what is still missing](ROADMAP.md#what-is-still-missing).

**Supermemory** needs an account, which this repository does not have. The only endpoint
anything here has ever called is `POST /v3/documents/list`, in the importer, and it is a
read; an arm has to write a corpus and query it. So the arm ships with no default write or
search path and refuses without them rather than guessing, and it refuses without an
explicit container tag so that a run cannot land in whatever space an account defaults to.
**No Supermemory number is published in this repository**, because nobody here has run it.

### The scores, and everything that makes them less than they look

One run has been done. **The reader was an agent, not a model behind an API** — there is
no key in this repository — and the answers were then audited by hand, correcting for the
containment judge's known false positives (it marks a correct answer trapped for reciting
the history it corrects) and false negatives (it marks a correct paraphrase wrong).

| arm | context | correct | genuine traps |
|---|---:|---:|---:|
| `none` (floor) | 0 tok | 10% | 0 |
| `full_transcript` | 2,451 tok | **100%** | 0 |
| `naive_rag` | 582 tok | 80% | 0 |
| `memvara` | 519 tok | 95% | 0 |
| `memvara_structured` | 430 tok | 95% | 0 |

**This is not a benchmark and must not be quoted as one.** Twenty questions, on a corpus
we wrote, answered by an agent that is the same party that wrote the library. It is **not
reproducible**: there is no model id, no seed and no temperature to put beside it, and the
same contexts answered again will not give the same answers. `evalkit.FileReader` and
`demo/harness.py` both print that banner above their own tables, and it is the correct
reading of them. What a run like this can do is show the pipeline produces sane answers
from real retrieval. It cannot rank systems.

With that said, four things in it are worth reading:

- **A careful reader with the whole transcript scored 100%.** At this corpus size the
  memory layer earns nothing on accuracy — it is beaten, and by the simplest possible
  baseline. What it earns is the size column: **5.6× fewer tokens for 95%**
  (2,451 → 440; the `memvara` arm is 4.7×). That is a claim about a *slope* — retrieval
  context is flat in corpus length while transcript context is linear — and this run has
  exactly one corpus size, so the slope is argued rather than measured. A second corpus
  ten times longer is what would turn it into evidence.
- **`naive_rag` was the only arm that genuinely lost information**, and its four failures
  were exactly the bitemporal ones. That is the comparison the corpus was built for: it
  runs the same embedder, at the same `k`, over the same visible turns, so a difference
  between it and the memvara arms cannot be explained by vector quality.
- **The trap metric produced no signal at all**, because the reader never fell for one:
  0 genuine traps in every arm, `naive_rag` included — so its four misses were wrong in
  some other way rather than by reciting the superseded value. The failure mode the
  product describes needs a reader that skims. Reported as a null result rather than
  dropped, because `trapped` is the column a before/after claim would rest on and it is
  the column that did not move.
- **The floor is 10%, which is 2 questions of 20** — and the harness warns, on every run,
  that an arm with no context abstains on the two `unanswerable` questions by
  construction and would score that kind on any corpus. Read the floor as "at or near
  zero on the eighteen questions that have an answer", which is what makes the other rows
  mean anything.

### The finding that matters more than the score

The `memvara` arm — the shipped defaults, a transcript dropped in with no `llm=` —
produced **zero claims from those 64 turns**. Its prompt block has no
`Known about the user` header in it at all, only the episode tail. The rule extractor's
vocabulary is first-person declaratives and a support history is not written that way, so
in that configuration there is no supersession and no bitemporal reasoning: it is lexical
episode retrieval with a different ranker, and its 95% is not a measurement of the thing
this comparison exists to test. `memvara_structured`'s is. The mechanism, the receipt
counts and the way out are in
[What the fast path does not catch](DESIGN.md#what-the-fast-path-does-not-catch-measured).

That is why there are two memvara arms and why neither may be deleted: the first is what
an evaluator meets on a weekend, and the second is what a deployment ships.

### Measuring against your own store: `bench/hosted.py`

Every corpus above was built for its benchmark. None has the shape a real
store develops — the roadmap's census of one production store found ~95% of
claims on predicates outside the declared vocabulary and a join rate of 0.5%.
`bench/hosted.py` measures the read path against the store you actually have:

    # hosted store (credentials from ~/.memvara), the default
    PYTHONPATH=. python3 bench/hosted.py --probes ~/.memvara/probes.jsonl

    # a local store file — `--db` is the only way to reach one
    PYTHONPATH=. python3 bench/hosted.py --db ~/.memvara/store.db \
        --tenant workstation --probes ~/.memvara/probes.jsonl

`--tenant` and `--user` say which scope to read the file at, and default to
`default` and the whole tenant. A store file is not one population: scopes
inherit upward, so a run reads the claims visible at the scope it opened, and a
file whose claims all live under one named tenant answers *nothing* at
`default`. That is not hypothetical — it is what a real store did, silently:
`--draft` printed no rows, the run exited 0, and the fingerprint said `claims:
0`, which is exactly what an empty store prints.

So a `--db` run that can see nothing while the file itself holds live claims
now says so, on **stderr**, naming both numbers and the scope:

    WARNING: this store holds 1240 live claims and none of them are visible at
    scope default/*/*/* — every probe will miss and --draft will print nothing.
    Re-run with --tenant/--user naming the scope the claims are under.

Both numbers are *live* claims. `count()` resolves its states to `("live",)`,
so the whole-file figure is `stats(None)["live_claims"]` and not `claims`,
which counts retired and ended rows too. Comparing against `claims` would fire
on a store whose facts are all at the right tenant but retired — telling
someone to re-scope when nothing is misscoped. A backend whose `stats` predates
`live_claims` gets silence rather than a fallback, for the same reason.

stderr rather than stdout because `--draft` and the seeding phases write JSONL
to stdout and you redirect that into a file; a warning on stdout would corrupt
the probe file it exists to help you build. The exit code does not change —
this tool has no pass/fail status — and the warning fires on both the `--draft`
path and a scoring run whose table would read `0 claims`.

It is asked only of a local store. The whole-file count comes from
`SQLiteStore.stats(None)`, the unfiltered call, and unfiltered counts disclose
how much data other tenants hold; `RemoteMemvara` talks to a shared
multi-tenant server, so that question is never put to it and no warning is
issued there. `--tenant` is likewise refused without `--db`, rather than
accepted and ignored: the hosted facade resolves the tenant from your
credential and the client never sends the parameter, so honouring the flag
there would change what a run records about itself without changing a byte of
what it read. `--user` is a real narrowing and applies to both routes.

The scope is recorded in the run's fingerprint alongside the claim count,
embedder and surface, and `--compare` warns on it: two runs at different scopes
read two different populations out of one file and their deltas are not a
before/after.

`--db` opens the store with the embedder that **wrote** it, read back from the
fingerprint the library records beside the file, not with `default_embedder()` —
which returns a 384-dimensional sentence-transformers model as soon as that
package is importable, and so could not open a store built by the 512-dimensional
fallback at all. A hashing embedder is reconstructed from its recorded name
(`hashing:<dim>:<lo>-<hi>` carries its whole vector space); a store with no
fingerprint on record — or none that binds, because it holds no vectors for a
fingerprint to describe, which is the same condition `Memvara` itself skips its
compatibility check on — takes the library default as before. Any other name is used
only when `default_embedder()` is itself what wrote the store, and is otherwise
refused, naming the embedder on record — install it and re-run, or open the store
yourself and pass it in as `mem=`. Measuring with the wrong embedder either raises
on every read or, at equal width, silently scores two unrelated vector spaces
against each other.

You author the probes once — `hit` (a question whose answer you know is
stored, gold = its claim id), `abstain` (a question the store cannot answer,
gold = nothing), `verbatim` (a claim's own text, which must return that claim
first), `ambiguous` (real prompts from your logs, judged) — and the run
reports hit@k, mean gold-rank, false-injection rate with per-failure score
headroom, and self-retrieval@1. `--draft` and `--seed-from-recalled` help
author; both refuse to produce a probe no person has reviewed.

#### The probe file

One JSON object per line, at `--probes PATH` (default `~/.memvara/probes.jsonl`).
The runner refuses the whole file on the first bad row, naming the line — it
never skips one.

```jsonl
{"id": "p001", "class": "hit",       "query": "what suite must run with -j1?", "gold": ["cl_..."]}
{"id": "p002", "class": "abstain",   "query": "write a haiku about rain",      "gold": []}
{"id": "p003", "class": "verbatim",  "query": "<the claim's own text>",        "gold": ["cl_..."]}
{"id": "p004", "class": "ambiguous", "query": "<a real prompt from your log>", "gold": ["cl_..."], "judged": "2026-09-01"}
```

| field | required | meaning |
|---|---|---|
| `id` | yes | non-empty, unique within the file; it is what a result row is addressed by |
| `class` | yes | one of `hit`, `abstain`, `verbatim`, `ambiguous` |
| `query` | yes | non-empty; what you would actually type |
| `gold` | yes | list of claim ids. Empty **only** for `abstain`; non-empty for every other class |
| `judged` | no | date a human judged an `ambiguous` row, because a judgment ages as the store changes |
| `draft` | no | `true` marks a row `--draft` emitted; the runner refuses it until you rewrite the query and remove the mark |

Probe files are private to a store and never belong in this repository. The
numbers are per-store and are not memvara scores: nothing measured here is
comparable between two stores, let alone publishable against another system.

One thing the hosted route cannot see, stated because hosted is the default
above. `RemoteMemvara.recall` returns prose and names no claim ids, so on that
route the injected set is inferred from `search()` at the same `k` and
`min_score` — which is what `recall()` renders from, so the two agree by
construction on the local engine. What it cannot catch is a server-side
divergence between what `POST /v1/recall` renders and what `POST /v1/search`
returns. `recall()` is still called, so a hosted recall failure still fails the
run; it is only a difference in *which* claims the two endpoints pick that
would pass unnoticed. Run against `--db` for a number read off the recall call
itself.

#### Seeding probes from the recall hook's own state: `--seed-from-recalled`

    PYTHONPATH=. python3 bench/hosted.py \
        --seed-from-recalled ~/.memvara/.hooks/recalled --dump pairs.jsonl
    # judge pairs.jsonl into {"id", "gold": [claim ids]} rows, then
    PYTHONPATH=. python3 bench/hosted.py \
        --seed-from-recalled ~/.memvara/.hooks/recalled \
        --dump pairs.jsonl --answers answers.jsonl --judged 2026-09-01

The first pass dumps real queries, blinded — shuffled by `--seed`, carrying the
query text and nothing else, so a judge answers from the store rather than from
what the hook happened to return that day. The second turns the judgments into
`ambiguous` probes (and `abstain` probes, where the judgment was "the store has
nothing for this"). `--judged` is required: a judgment ages as the store
changes.

**Be clear about what that directory holds, because it is narrower than "real
recall traffic".** `plugin/hooks/recall.py` keys **one file per session**, named
for the session id, and rewrites it on every turn. Each file therefore carries a
single query: that session's **most recent** substantive prompt, **truncated to
300 characters** (`MAX_CARRY_CHARS`). Files older than **fourteen days**
(`SEEN_TTL_SECONDS`) are pruned. No append-only log of recall events exists on
disk, and this does not reconstruct one.

So the sample is one query per recent session. It skews toward whatever each
session last asked, it cannot see anything asked earlier in a session, and a
long prompt reaches the judge clipped. On a machine in daily use that is still
on the order of a thousand distinct real queries and worth judging — but when
you read the resulting `ambiguous` rows, that is the population they came from.
Unreadable files are counted and the count is printed, so a directory that
failed to parse does not look like an empty one.

Both read surfaces are queried at `--min-score`, whose default **resolves
exactly as the recall hook's own `_min_score()` does**: `MEMVARA_RECALL_MIN_SCORE`
when that is set (clamped to `[0, 1]`), otherwise the hook's `MIN_SCORE`. So a
store owner who recalibrated and exported that variable — the workflow the hook's
own comment recommends — measures their shipped configuration, not the constant,
and the run measures what that hook actually injects on this machine.
So false-injection is not predetermined: it counts how often the shipped floor
still injects on a question the store cannot answer, and it can come back at
zero or well above it depending on the store and the embedder. Pass
`--min-score 0` to measure the unfloored read path instead — that number is
100% by construction on any non-empty store, which is why it is not the
default.

---

Previous: [How it works](DESIGN.md) · Next: [Roadmap](ROADMAP.md) · [Documentation index](README.md)
