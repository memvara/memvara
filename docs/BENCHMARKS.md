# Benchmarks

Every number here is reproducible from this repository; the harnesses are in
`bench/` and `demo/`. Where a result is synthetic or self-authored it says so in
its own heading, because that is the part a reader is entitled to discount.

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

## LOCOMO and LongMemEval — retrieval, measured

Not answer accuracy, and **not comparable to published LOCOMO/LongMemEval scores**, which
are end-to-end judged accuracy. This measures the thing a memory layer is actually
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
| `bench/multihop.py`, **as shipped** | 4,498 | **2.9% → 6.4%** at k=12, **7.6% → 21.8%** at k=25 |
| `bench/twowiki.py`, gate off, **public** | 26,403 | **28.3% → 72.2%** at k=12 on chained questions; **−13.7** on flat ones |
| `bench/twowiki.py`, **as shipped** | 26,403 | **28.3% → 42.1%** answer and **25.5% → 39.5%** chain on chained questions; −0.4 on flat |

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
  two-hop      12     4.0%     9.3%    29.7%      64.3%     69.7%     100.0%   100.0%    99.7%
  two-hop      25     9.3%    30.3%    72.7%      96.3%    100.0%     100.0%   100.0%    99.7%
  three-hop    25     4.0%     4.7%     4.7%       4.7%     34.7%      48.7%   100.0%    46.7%
  all          12     2.9%     6.4%    20.0%      43.1%     46.4%      78.7%    83.1%    77.8%
  all          25     7.6%    21.8%    50.0%      65.8%     78.2%      82.9%   100.0%    82.0%
```

**`+graph` is the shipped configuration and `+graph!` is the same with
`intent_weighting=False`.** The `+graph` column used to read `2.9%` and `7.6%` — exactly
`search`, as though the leg were not installed.

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

Both halves are now fixed and the numbers above are after both:

* **Naming an instant no longer switches the walk off.** The temporal row still decides
  the other three legs; the graph leg keeps the weight the query shape asked for.
* **The classifier counts predicates instead of matching a longer word list.**
  `intent.predicate_refs` counts how many *distinct* predicates a question names, folded
  onto canonical names, and two of them is a chain — one predicate is a question about one
  slot. Derived from `PredicateRegistry`, so no word was added because this benchmark
  needed it. **How far it reaches is narrower than "derived from the registry" suggests:**
  `PredicateRegistry.learn()` is called only from the LLM-assisted resolution in
  `write/pipeline.py`, so an offline store never teaches it and the rule sees the 23
  builtins alone. `bench/twowiki.py` exposed that — every predicate in that corpus is a
  learned one, so the rule does not fire there. Matched as phrases
  and never as tokens: `lives_in` splits into `lives` and `in`, and a token index would
  read almost every question as a chain.

**What is still gated is one family, and it is morphology rather than vocabulary.** "Who
founded the company that X works at" names `works_at` and `founded_by`, but the store
holds `founded_by` and the question says "founded the", so the phrase never matches.
Matching the head token instead was measured and rejected: the head tokens of this
registry's predicates include `in`, `is`, `do`, `has`, `date` and `place`, which turns
"what is my name" into a two-predicate chain. A stemmer would close that gap; a longer
word list would only close it here.

**The standing advice needs a condition on it, which `bench/twowiki.py` supplied.** It
used to read: a deployment turning the graph leg on should turn `intent_weighting` off
with it. On public multi-hop data that buys 44 points on chained questions and **costs
14 on flat ones**, so it is right for a workload of relationship questions and wrong for
a workload of lookups. Net on a corpus that is 54% chained it is +17.4 points at k=12;
invert the mix and it inverts.

The honest statement is that the gate is right in principle and badly calibrated: it
captures almost none of the gain and still pays part of the cost. A deployment should
turn `intent_weighting` off if its traffic is mostly relationship questions, and leave
the graph leg off entirely if it is mostly lookups. Neither is a default this repository
can pick for you, which is why `w_graph` ships at 0.0.

**The store now asks itself.** Where no live claim's object is another live claim's
subject, the graph leg does not run whatever `w_graph` says — so turning it on costs
nothing on a store that cannot use it. Measured on LongMemEval with `w_graph=1.0`: every
category exactly baseline, where it previously lost 1.6 points of single-session-user
R@12. On 2Wiki the gate closed the leg on 0 of 3,000 searches and no returned row moved.
See `UnjoinedStoreWarning`, which says so out loud once per retriever.

That is a floor, not a recommendation. **Ask the store before you guess at the traffic**,
because the store is the half you can measure. `memory_stats` reports a **join rate** — the share of live claims whose object is
the subject of another live claim, which is the share that leads anywhere at all. The two
corpora below sit at 40.6% and 0.0% and the leg gains 13 points on one and loses 1.6 on
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

**The leg is worth 2.6x on multi-hop questions, and costs 14 points on questions that are
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
  all                12,576   50.6% / 37.2%   57.9% / 43.8%   68.0% / 57.2%
  chained             6,785   28.3% / 25.5%   42.1% / 39.5%   72.2% / 70.3%
  flat                5,791   76.7% / 50.8%   76.3% / 48.9%   63.0% / 41.9%
  compositional       5,236   22.8% / 20.5%   40.8% / 38.7%   70.1% / 68.3%
  inference           1,549   46.6% / 42.3%   46.6% / 42.3%   79.3% / 77.3%
  comparison          3,040   73.9% / 96.8%   74.9% / 88.7%   60.7% / 58.2%
  bridge_comparison   2,751   79.8% /  0.0%   77.8% /  4.9%   65.5% / 23.8%
```

**`chained` is the result.** `compositional` and `inference` questions chain one fact into
the next — "who is the mother of the director of X" is `director` then `mother` — and the
leg takes them from **28.3% to 72.2%**. `inference` also carries its derivation: chain
recall 42.2% → 76.4%, so most answers arrive with every triple that supports them rather
than with the gold entity alone.

**`flat` is the control, and it did what a control is for.** `comparison` and
`bridge_comparison` ask which of two independent entities came first. The evidence has two
ends and no join, the leg has nothing to walk, and turning it on **costs 13.7 points**
(76.7% → 63.0%) because the walk spends `k` on neighbours of a hub. Had that row improved,
the `chained` row would be worth much less: it would suggest the leg helps by adding rows
rather than by following edges.

**The intent gate is right in principle, and it now captures some of what it was
blocking.** It exists to route flat questions past the walk, and on `flat` it does:
76.3% against search's 76.7%. On `chained` it used to block almost the entire gain —
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

**Answers and derivations move together.** On `chained`, the leg is worth +13.8 points of
answer recall and **+14.0 of chain recall** — 28.3% → 42.1% and 25.5% → 39.5%. Ungated the
two columns nearly meet, 72.2% against 70.3%: almost every answer the walk finds arrives
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
`inference` gains **nothing** — 46.6% through every change in this series — and the reason
is not a bug. Those
questions ask "who is the maternal grandfather of X"; the evidence is `(X, mother, Y)` and
`(Y, father, Z)`, and the question names neither `mother` nor `father`. It names a
*derived* relation. Matching a question's words against stored predicate names cannot
bridge `grandfather` to `mother` + `father`, and no longer word list closes that — it
needs synonymy or entailment, which is a model rather than a lookup. All of the gain here
is in `compositional`, where the question does say the predicates out loud: 24.0% → 32.1%.

`bridge_comparison` chain recall is 0.0% for `search` and 6.7% ungated. Those chains are
four hops and `graph_depth` ships at 2, so that row measures the depth bound rather than
traversal — the same caveat the synthetic benchmark's three-hop rows carry.

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

What it does not do is clear the bar. Temporal-reasoning is unchanged and multi-session
loses 0.5, so the default stays off. The reason is the same shape as the graph leg's: the
leg is strongest when a caller passes `valid_at`, and no benchmark here passes one — both
call `search(question, k)` with the question as prose. It is second strongest on a **live**
store, where `add()` stamps turns with the wall clock and the recent ones genuinely are
near the anchor; LongMemEval replays an archive, so the abstention fires nearly everywhere,
which is right and also leaves nothing to measure.

**The blocking dependency here is ingestion, not retrieval.** Both public instruments are
blind to the graph leg for the same reason the `memvara` demo arm produces zero claims —
see [What the fast path does not
catch](DESIGN.md#what-the-fast-path-does-not-catch-measured). Until the offline write path extracts
from ordinary prose, no public retrieval number can move on this.

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

---

## Answer quality, end to end (an authored corpus, an agent as the reader)

Every number above measures **retrieval** — did the right claim come back, ranked where it
should be. None of them measures **answers**: whether an agent reading memvara's output
tells the customer the right thing. [`demo/`](../demo/) is the apparatus for that, and
[`demo/README.md`](../demo/README.md) is its full documentation.

```
demo/scenario.py    64 turns of one customer's support history, and 20 questions
demo/baselines.py   five context-building arms
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
  memvara_structured        1721       2151           430              12.0 / 60.8
```

`~tokens` is characters ÷ 4, an estimate and not a tokenizer.

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
  baseline. What it earns is the size column: **5.7× fewer tokens for 95%**
  (2,451 → 430; the `memvara` arm is 4.7×). That is a claim about a *slope* — retrieval
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

---

