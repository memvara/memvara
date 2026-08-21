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
[docs/ROADMAP.md](docs/ROADMAP.md).

Two findings from building this. **LongMemEval's `oracle` split cannot measure evidence
retrieval at all** — in all 500 instances every haystack session *is* an evidence session,
so recall there is 99.2% by arithmetic. The harness now computes `chance` and warns loudly
above 50%; `--share-store` is the offline workaround. And **retrieval was not reproducible
until this run**: `HybridRetriever` broke score ties on `claim.id`, a fresh `uuid4` per
ingest, so two ingests of one corpus ranked differently and the numbers drifted 0.07
points. Ties now break on a content hash and three full runs are byte-identical.

### The graph leg, and why no row above moves

`w_graph > 0` adds a third retrieval leg: a bounded walk out of the entities the vector
and lexical legs just named (`memvara/retrieve/spread.py`). **It ships at `w_graph=0.0`,
and the reason is that neither benchmark above can see it.**

```bash
PYTHONPATH=. python3 bench/locomo.py      --score retrieval --w-graph 1.0
PYTHONPATH=. python3 bench/longmemeval.py --score retrieval --share-store --w-graph 1.0
```

| instrument | claims in the store | what the leg changed |
|---|---:|---|
| LOCOMO, 1,531 questions | **0** | nothing — the two reports are byte-identical |
| LongMemEval oracle, 500, `--share-store` | **81** | R@12 **70.4 → 70.0**; single-session-user **92.2 → 90.6**, multi-session **65.5 → 64.6** |
| `bench/multihop.py` (synthetic), gate off | 4,498 | **2.9% → 21.6%** at k=12, **7.6% → 50.4%** at k=25 |
| `bench/multihop.py`, **as shipped** | 4,498 | **nothing** — the intent gate routes 2 of the 3 question families past the walk |

The LongMemEval row is a **loss**, and it is the decisive one: 1.6 points off
single-session-user is above any guardrail worth setting. It is what a third leg costs
when it has almost nothing to walk — where it fires at all it puts a real zero on every
candidate it did not reach, and 81 claims across a 940-session store is not a graph.

The leg walks *claims*, and both public runs are essentially episode retrieval:
`SalienceGate` drops any turn whose role is not `user`, LOCOMO writes each turn under the
speaker's name, and the deterministic extractor reads a small set of sentence forms.
LOCOMO extracts **0 claims from 5,882 turns** and LongMemEval **81 from 10,866**. With no
claims at all the candidate set is empty and the leg is never reached, so the LOCOMO
figure is not a null result — it is the leg being inert by construction.

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
  two-hop      12     4.0%     4.0%    31.7%      64.3%     69.7%     100.0%   100.0%    99.7%
  two-hop      25     9.3%     9.3%    73.3%      96.3%    100.0%     100.0%   100.0%    99.7%
  three-hop    25     4.0%     4.0%     4.7%       4.7%     34.7%      48.7%   100.0%    46.7%
  all          12     2.9%     2.9%    21.6%      43.1%     46.4%      78.7%    83.1%    77.8%
  all          25     7.6%     7.6%    50.4%      65.8%     78.2%      82.9%   100.0%    82.0%
```

**`+graph` is the shipped configuration and `+graph!` is the same with
`intent_weighting=False`. The gap between them is the second finding, and it is not a
small one.** Two of the three question families here contain no word in
`intent.RELATIONAL_MARKERS` — "which city is the company X works at based in", "who
founded the company that X works at" — so the gate reads them as `lookup` and routes them
past the walk. The leg is switched off on exactly the questions it was built for, and the
column collapses onto `search`.

That is a gap in the vocabulary rather than in the gate: `works at` and `founded` are
relations by any reading, and both are *predicates in this store's own registry*. The fix
is to derive the relational markers from the registry's predicate names rather than from a
hand-written list, which is written down here rather than done, because widening the list
by hand against a benchmark this repository wrote is how a classifier gets fitted to its
own corpus. Until then, a deployment turning the graph leg on should turn
`intent_weighting` off with it and pay the walk on every query.

The three-hop rows barely move because `graph_depth` ships at 2; that row measures the
bound, not the traversal. And this benchmark is synthetic and self-authored — read it as
an illustration of a mechanism, which is not evidence for a default.

**So it is opt-in, and it stays opt-in until ingestion changes.** knowledge-update, the
row the thesis rests on, held exactly at 91.0; single-session-user did not. The precedent
for shipping a measured stage at zero is the MMR rejection recorded in `hybrid.py`, and
this is the same call: a large gain on a workload we wrote, a small loss on one we did
not, and no honest way to make a default out of that pair.

```python
mem = Memvara("memory.db", read_w_graph=1.0)
```

`memvara/retrieve/intent.py` is what makes turning it on affordable — a deterministic,
model-free classifier routes `lookup` and `temporal` queries past the walk entirely — and
the table above is also what it currently costs. Every multiplier in it other than the two
gates is 1.0 and stays 1.0 until a per-category sweep moves it.

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
catch](#what-the-fast-path-does-not-catch-measured). Until the offline write path extracts
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

Read [`bench/baseline.py`](bench/baseline.py) before quoting any of this: the comparison
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
tells the customer the right thing. [`demo/`](demo/) is the apparatus for that, and
[`demo/README.md`](demo/README.md) is its full documentation.

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
  memvara                   2043       2331           511              12.0 / 60.8
  memvara_structured        1671       2032           418              12.0 / 60.8
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

**These context sizes are the ones that run was answered against, and they are no longer
what the arms produce.** The offline write path was widened afterwards, so the `memvara`
arm now builds 511 tokens and `memvara_structured` 418 — the table above is the current
apparatus, this one is a record of a past run. The accuracy column belongs to the contexts
in *this* table and cannot be re-attached to the new ones without answering them again.

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
produced **zero claims from those 64 turns**, so its prompt block had no
`Known about the user` header in it at all, only the episode tail. No claim means no
`(subject, predicate)` slot, so there was no supersession and no bitemporal reasoning of
any kind: it was lexical episode retrieval with a different ranker, and its 95% was not a
measurement of the thing this comparison exists to test.

**It now produces six**, and two of them supersede on the world clock — the delivery
address when the customer moves, the contact preference when it reverses. That is the
machine working offline, on prose, with no key. It is also still four facts short of what
the corpus contains: the plan, the serial, the mobile correction and the billing address
are in sentence forms no rule reads, so `memvara_structured` remains the arm that
exercises the whole of it. The mechanism, the receipt counts and the way out are in
[What the fast path does not catch](#what-the-fast-path-does-not-catch-measured).

That is why there are two memvara arms and why neither may be deleted: the first is what
an evaluator meets on a weekend, and the second is what a deployment ships.

---

