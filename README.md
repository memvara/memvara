# Memvara

**Bitemporal memory for AI agents.** Structured facts, deterministic contradiction
resolution, hybrid retrieval, and a write path that mostly doesn't call an LLM.

```bash
pip install -e .
```

```python
from datetime import datetime, timedelta, timezone
from memvara import Memvara

now = datetime.now(timezone.utc)
mem = Memvara("memory.db", user="alice")

# Two independent axes. `valid_from` is when it was true in the world; `recorded_at`
# is when we learned it. Both are set here so the time-travel query below has a past
# to travel to — a plain mem.add() would record both facts as of now.
mem.remember("user", "lives_in", "Berlin",
             valid_from=now - timedelta(days=800), recorded_at=now - timedelta(days=800))
mem.remember("user", "lives_in", "Lisbon",
             valid_from=now - timedelta(days=30), recorded_at=now - timedelta(days=30))

[r.text for r in mem.search("where do they live?")]
# -> ['user lives in Lisbon']

[(c.object, c.valid_to) for c in mem.history("user", "lives_in")]
# -> [('Berlin', datetime(... 30 days ago ...)), ('Lisbon', None)]

[c.object for c in mem.get_all(as_of=now - timedelta(days=365))]
# -> ['Berlin']      # what was true a year ago
```

Two axes means two clocks, and they move independently:

```python
mem.get_all(valid_at=T)   # what we believe TODAY about how the world was at T
mem.get_all(known_at=T)   # what we believed at T, about the world as it is now
mem.get_all(as_of=T)      # both clocks at T — what we believed at T, about T
```

The middle two are the ones a single instant cannot ask. A correction that arrives in
August about June is invisible to `as_of=June`, because that call rewinds the belief
clock past the correction; `valid_at=June` is how you see it. Every read that took
`as_of` takes all three — `search`, `get_all`, `count`, `history`, `why`, `produced`,
`neighborhood`, `paths_between` — and `as_of` is exact sugar for
`valid_at=known_at=T`. Passing it alongside either axis raises rather than quietly
picking one.

Core requires **numpy and nothing else**. It runs offline, with no API key, no Docker,
and no vector database.

---

## Open core, and exactly where the line is

Everything in this repository is Apache-2.0, and it is the whole library: the bitemporal
store, deterministic contradiction resolution, hybrid retrieval, consolidation,
provenance, entity resolution, multi-hop traversal, the MCP server, the mem0 shim and
importer, the LangChain / LlamaIndex / CrewAI / LangGraph adapters, and the SQLite
backend. Nothing in it is gated, time-limited, keyed, or degraded into a demo. There is
no free tier here, because there is no tier — the library runs on numpy, offline, with no
account and no network call, and it is the same code we build everything else on.

A commercial product is built around it. It is not a better version of this one:

| in this repository, Apache-2.0 | commercial, separate product |
|---|---|
| SQLite store | Postgres / pgvector store |
| in-process library; MCP over stdio | REST API and auth |
| the `Redactor` and `Recorder` seams | governance: policy, retention, tamper-evident audit chain, RBAC |
| one process, one store | multi-tenant control plane |
| `WriteReceipt`, telemetry counters | usage metering, quotas, rate limiting, hosted console |

The right column is the commercial product's *scope*, not a shipping manifest — some of it
exists today and some is being built. The left column is what matters here, and the left
column is complete.

The pattern is that **the library is the product and the commercial layer is the
operations around it.** Nothing in the right column changes what a claim is, how a
contradiction resolves, what `why()` returns, or what `search()` finds — that is a
constraint on what may be built there, not a slogan, because a paid layer that altered the
semantics of the free one would make the free one untrustworthy. What is over there is what
you need when memory becomes several machines' problem and several people's. For one
application on one machine, nothing is missing.

The uncomfortable half, stated here rather than discovered three weeks in: if you need
Postgres or an HTTP endpoint, this repository does not have one and is not scheduled to
grow one. That is a commercial boundary, not a backlog — saying "planned" would be the
dishonest version. The line is drawn there because SQLite is genuinely sufficient for a
single node, and needing more than one node correlates closely with being able to pay for
it. The storage half of that sits behind the `Store` protocol in
[`memvara/store/base.py`](memvara/store/base.py), which is public, documented, and
implementable by anyone — a third-party Postgres backend is a legitimate thing to write,
and neither the license nor the design objects to one.

Two things stay open on purpose and are worth naming, because they are the ones a
commercial reading would have closed. **The mem0 shim and the `history.db` importer are
Apache-2.0** — they are the reason anyone can leave mem0 in an afternoon, and putting them
behind a paywall would mean charging for the exit. **`erase()` and `purge()` are
Apache-2.0** — real, irreversible deletion including the FTS tokens and the vectors. A
GDPR Article 17 obligation is not a feature to upsell.

The one thing to know before you count on "offline": with no `llm=`, `add()` runs the
deterministic fast path only and drops the turns its rules do not recognise. `remember()`,
retrieval, contradiction resolution and everything else are unaffected and need no model
ever. See [Honest limitations](#honest-limitations).

---

## Why this exists

mem0 and its descendants store a memory as an opaque string with an embedding, and every
`add()` costs a model call on the critical path. Retrieval is vector top-k.

> **Corrected against mem0 2.0.17.** An earlier version of this section said `add()` costs
> *two* LLM calls — extract, then adjudicate ADD/UPDATE/DELETE. That described mem0 1.x.
> 2.x makes **one** call, with existing memories passed into a single additive extraction
> prompt; `DEFAULT_UPDATE_MEMORY_PROMPT` is still in the source and no longer reached from
> the add path. The correction cuts against us, so it is stated rather than quietly
> dropped — but the contradiction problem it was cited for got *larger*, not smaller: 2.x's
> add path emits only `ADD` events, and its prompt says "Your sole operation is ADD".
> Conflicting values are **linked**, never retired. `update()` and `delete()` are calls
> your application has to know to make.

That design has four consequences that show up in production:

1. **Contradictions accumulate.** In 2.x this is explicit: nothing on the write path
   retires anything. Six months in, the store holds three cities for one person and
   returns whichever embeds closest to the question.
2. **Writes are slow and expensive.** A model call per turn, on the critical path,
   including for "ok, thanks."
3. **There is no time.** One `updated_at` column can't answer "where did she live in
   March?" or absorb a fact that arrives late about the past.
4. **Nothing explains itself.** When the agent says something wrong, you cannot ask which
   memory caused it, where that memory came from, or why it ranked first.

Memvara is built around the observation that **most of this doesn't need a model at all.**

---

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

**Where the stale-value result actually comes from.** An earlier version of this README
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
measure retrieval, not answers. Nothing here has been scored end to end against a reader
model, and that remains the honest hole in the evidence.

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

## What's different

### Facts are structured and bitemporal

A memory is a `Claim` — a `(subject, predicate, object)` triple with two independent time
axes:

| axis | fields | answers |
|---|---|---|
| **valid time** | `valid_from`, `valid_to` | when was this true in the world? |
| **transaction time** | `recorded_at`, `invalidated_at` | when did *we* believe it? |

Collapsing those into one timestamp is the mistake almost every agent-memory layer makes.
Keeping them apart is what lets you ask both "where does she live now?" and "on March 1st,
what did we think?" — and lets a late-arriving fact correct the past without rewriting
history.

```python
mem.remember("user", "born_in", "Osaka", valid_from=datetime(1990, 1, 1))
# true since 1990, known since today — both recorded honestly
```

Ending a claim moves **one** of those clocks, and which one is the whole distinction:

```python
mem.remember("user", "lives_in", "Lisbon")                    # she moved
# -> Berlin: valid_to set, still believed          state == "ended"
mem.remember("user", "lives_in", "Lisbon", close="retired")   # we misheard her
# -> Berlin: invalidated_at set, interval untouched  state == "retired"
```

`ended` is the default, because a new value is news about the world, not a complaint
about the record — so `get_all(valid_at=<while Berlin held>)` keeps answering `Berlin`.
`close="retired"` is the caller stating a correction, and only a caller can know that.
`forget()` and `delete()` default the other way: forgetting is something the holder of a
memory does, so they stop belief and assert nothing about the world.

#### Reading one population: `states=`

Three states, so the read filter takes the three words rather than a boolean. `search`,
`get_all` and `count` accept `states=`, any non-empty subset of `("live", "ended",
"retired")`, defaulting to `["live"]`:

```python
mem.remember("user", "lives_in", "Berlin", valid_from=JAN, recorded_at=JAN)
mem.remember("user", "lives_in", "Lisbon", valid_from=JUN, recorded_at=JUN)  # she moved
mem.remember("user", "works_at", "Acme",   valid_from=JAN, recorded_at=JAN)
mem.forget("user", "works_at")                       # we stopped believing it

[c.object for c in mem.get_all(states=["live"])]     # ['Lisbon']
[c.object for c in mem.get_all(states=["ended"])]    # ['Berlin']  — true once, still believed
[c.object for c in mem.get_all(states=["retired"])]  # ['Acme']    — the correction audit
```

`states=["retired"]` is the one a boolean could never express, and it is the query a
correction audit is made of. It cannot be recovered by filtering afterwards either:
`search` is capped at `k`, so a client-side filter returns an empty audit whenever enough
live claims fill the page, with nothing in the result to say the answer was truncated.

`include_invalidated=` remains an exact alias — `False` is `["live"]`, `True` is all
three — and is not deprecated. Passing both raises rather than picking one.

**Asking for all three states is not the union of the three parts.** It is the audit
view, and under it `valid_at` stops narrowing anything:

```python
from memvara.store import STATES                     # ("live", "ended", "retired")

# `.object` of each result, in the order get_all returns them (newest recorded first)
mem.get_all(valid_at=MARCH,  states=["live"])   # ['Berlin'] — where she lived in March
mem.get_all(valid_at=MARCH,  states=STATES)     # ['Lisbon', 'Acme', 'Berlin']
mem.get_all(valid_at=AUGUST, states=STATES)     # ['Lisbon', 'Acme', 'Berlin'] — same
```

The reason is that `Claim.state` is absolute while the query is as-of, so the three do
not tile the store: a fact recorded but not yet in force at `valid_at` — scheduled to
start next month — is named by none of them. The complete set therefore compiles to the
belief floor alone, which readmits that row and leaves the world clock nothing to
constrain. That is exactly what `include_invalidated=True` has always meant.

#### Counting claims

`stats()` reports each population separately because none of them is derivable from the
others. Take a store holding four claims — one live, one ended, one that ended and was
*later* retired, and one recorded now but not in force until next year:

```python
mem.stats()
# {'episodes': 0, 'claims': 4, 'live_claims': 1, 'ended_claims': 1,
#  'invalidated': 1, 'embeddings': 0}
```

Both claim filters are the full state predicate, not a column test, and on that store
every cheaper spelling is wrong:

| you might write | gives | truth | why |
|---|---|---|---|
| `invalidated_at IS NULL` | 3 | `live_claims` = 1 | counts every superseded version as live |
| `valid_to IS NOT NULL` | 2 | `ended_claims` = 1 | counts the ended-then-retired row, already inside `invalidated` |
| `claims - live_claims - invalidated` | 2 | `ended_claims` = 1 | the residual also holds the scheduled claim, which is in no state at all |

`ended_claims` and `invalidated` are disjoint, and **the counts do not sum**: `1 + 1 + 1`
against `claims = 4`. `claims` is the only total that covers everything, and a backend
that "corrects" the arithmetic has put the conflation back.

### Contradictions resolve without an LLM

The insight: **contradiction is mostly a schema property, not a semantic one.** "Lives in"
takes one value at a time. "Likes" takes many. Given the predicate's cardinality, a
conflict is an indexed lookup on `(subject, predicate)` — exact, free, and total.

```python
Cardinality.ONE   # lives_in, works_at, name  -> a new value retires the old
Cardinality.MANY  # likes, speaks, allergic_to -> values accumulate
```

No embedding search, no top-k cutoff a conflict can hide beneath, no non-determinism. The
same two facts resolve the same way every run. Unknown predicates default to `MANY`,
because keeping two facts degrades ranking while dropping a true one destroys
information — errors should fall on the recoverable side.

The model's job moves off the write path and onto *schema acquisition*: the first time an
unfamiliar predicate appears, one call asks whether it's single-valued; the answer is
cached forever. The thousandth occurrence costs nothing.

Aliases collapse too, so `lives_in` / `resides_in` / `based_in` / `moved_to` are one slot.
Without that, the contradiction between them is invisible — which is exactly how free-text
stores end up holding two cities for one person.

### Entities are folded before they are keyed

A keyed lookup only works if both facts land on the same key, and `Acme`, `Acme Corp` and
`acme, inc.` are the same employer written three ways. So the key is computed from a pure
fold — Unicode NFKD, casefold, punctuation and legal-suffix stripping — applied to subject
and object *before* the `(subject, predicate)` key exists:

```python
from memvara import entity_key
entity_key("Acme Corp.") == entity_key("ACME, Inc.") == entity_key("acme")   # True
```

Over a 258-write simulation across 6 employers and 3 drinks: 516 resolutions, **98.1%
settled by the fold alone, zero model calls**, and 41 distinct surface forms collapsed to
exactly the 9 real entities. `history("user", "works_at")` went from 22 rows to 6 — five
retirements and one live value, which is what actually happened.

The fold is *total*, so it needs no acquisition step and no cache: an entity seen for the
first time still gets a correct, stable identity for free. That is why `resolve_entity`
(the LLM path, for genuine aliases like `Big Blue` → `IBM`) ships **opt-in and unset** —
unlike predicates, entity surface forms never saturate, so acquisition would be a
per-entity tax forever rather than a one-time cost. The honest limit is that
`Stark` and `Stark Industries` are indistinguishable from two different companies without
one.

Learning an alias later does **not** rewrite history. A claim keeps the identity it was
written with, so `history()` doesn't silently restructure itself the day the model learns
something; applying an alias retroactively is `backfill_entities()`, dry-run by default,
which stamps every touched claim so `why()` can explain why history changed.

### The write path avoids the model

Four tiers, in order, each cheaper than the next one down:

| tier | what it does | cost |
|---|---|---|
| 0 | content-hash dedupe, then near-duplicate detection by embedding | no LLM |
| 1 | salience gate — does this turn contain a durable fact at all? | no LLM |
| 1b | rule-based extraction for common unambiguous forms | no LLM |
| 2 | batched structured extraction for what survives | **one** call per batch |

Most conversational turns carry nothing durable. mem0 pays a model call for "sounds good";
memvara pays zero — the salience gate drops it on a string comparison before anything is
embedded or sent. That is the whole of the 105-vs-2 row measured above.

(This sentence said *two* model calls until the correction at the top of this file landed.
Two was mem0 1.x. 2.x makes one, which is still one more than zero, and quoting the older
number here while correcting it forty lines earlier would have been the kind of thing that
makes a reader stop trusting the rest.)

Every `add()` returns a receipt that reports the cost, because a number you can't see is a
number nobody optimizes:

```python
receipt = mem.add(transcript)
print(receipt)   # <WriteReceipt +3 ~1 -1 skip=17 llm=1 42.3ms>
#                    added ─┘  │  │      │      └─ one batched call for 21 turns
#                 reinforced ──┘  │      └─ carried no durable fact
#                 closed out ─────┘
```

That third number is `receipt.closed`, and it is **not** a retirement count. A write
closes one clock or the other, so it holds both kinds — `receipt.ended` (the world
changed) and `receipt.retired` (the record was wrong) split it, and `Claim.state` says
which on any one claim. The label here read "retired" until the two axes were separated,
which named the rarer of the two for a number that is almost always the other one: the
write above superseded, and superseding *ends*.

The field is spelled `closed`. `receipt.invalidated` still works and is the same list —
the old name, kept because it is on the published API, to be removed at `1.0.0`.

### Retrieval is hybrid, time-aware, and explains itself

BM25 (SQLite FTS5) and vector search run in parallel and fuse with Reciprocal Rank
Fusion — rank fusion rather than score fusion, because BM25 scores and cosine similarities
aren't on comparable scales and normalizing them is guesswork.

Lexical retrieval isn't a nicety. Embeddings blur exactly the tokens agents most need
verbatim: error codes, version numbers, IDs, surnames. A query for `ERR_7734_TLSHANDSHAKE`
is a BM25 bullseye and a cosine near-miss.

Results are then rescored by **recency decay keyed to how volatile the predicate actually
is**:

| volatility | half-life | example |
|---|---|---|
| `STATIC` | ~never | `born_in` — a 10-year-old fact ranks undiminished |
| `SLOW` | 2 years | `works_at` |
| `FAST` | 7 days | `working_on` — last week's task stops crowding out this week's |

And every result carries an `Explanation`:

```python
r = mem.search("where do they live?")[0]
print(r.explain.summary())
# vector#1(0.812) bm25#2(6.44) recency=0.98 conf=0.90 sal=1.25 -> 0.7431
```

### The claims are a graph, and it can be walked at a point in time

A claim is `(subject, predicate, object)` and entity resolution folds every spelling of a
name onto one identity — so the store has been a labelled directed graph all along.
`neighborhood()` and `paths_between()` query it transitively.

```python
mem.remember("alice", "reports_to", "Dana")
mem.remember("dana", "works_at", "Kovac Labs")

for path in mem.paths_between("Alice", "Kovac Labs"):
    print(path.render(), round(path.score, 3))
# -> alice -reports_to-> Dana -works_at-> Kovac Labs 0.75

path.claims        # every hop, each one a claim you can pass to why()
path.nodes         # ('alice', 'dana', 'kovac labs') — folded, so Acme / Acme Corp /
                   # acme, inc. are one node. path.labels has the spellings as stored.
```

**Every edge on a path is evaluated at the same instant.** That is the point, and it is
what a bitemporal store is uniquely able to offer. An agent that searches, then searches
again on the result, is stitching two reads taken at two different times: if a write
lands in between, the chain it reports was true at no instant. `bench/multihop.py`
demonstrates exactly that — a write retires hop 1 and creates hop 2, and the loop happily
reports a connection that never existed. A traversal pins one `(valid_at, known_at)` pair
before its first hop and passes it unchanged to every hop after, so it returns nothing at
every instant.
A caller can close the same hole by passing one instant to both searches; the difference
is that traversal cannot be called any other way.

Negative polarity is never walked as a link — "Alice does *not* work at Acme" is a claim
about Alice and Acme and is not a path between them. Scope is checked on every hop with
the same rule `get()` uses, so a path can only ever be built from facts you could already
have enumerated yourself; traversal joins what is readable, it does not widen it.

Where it actually helps, measured on a synthetic set rather than asserted: at two hops a
search-then-search loop already reaches 96.3%, so recall alone barely justifies the
feature. At three hops that loop collapses to **4.7%**, against **34.7%** for traversal at its
defaults and **48.7%** once `min_hops` stops one-hop answers spending the whole of `k`.
LOCOMO's `multi-hop` category is *not* transitive multi-hop — its questions are
single-fact lookups whose evidence spans a couple of turns — so the number in the table
above cannot be improved by this, and is not claimed to be.

### Nothing is silently lost

Superseding sets an end timestamp; it never deletes, and it never records the old value
as an error. So the audit trail is free:

```python
for c in mem.history("user", "works_at"):
    print(c.object, c.recorded_at.date(), c.state, "-> replaced by", c.invalidated_by)

prov = mem.why(claim.id)
prov.episodes      # the exact source turns this was derived from
prov.superseded    # what it replaced
prov.extractor     # which model/rule version produced it
```

The deliberate exceptions are `erase()` and `purge()` — one claim and one scope. Erasure
is a separate, explicit, irreversible call rather than a flag on `forget`, and it removes
everything derived from the text. Purging a user takes their agents and sessions with
them, and both return per-table counts as evidence. See
[Two meanings of "delete"](#two-meanings-of-delete-kept-apart).

### The learned schema is durable

Predicate classifications are persisted, not held in process memory. This matters more
than it sounds: a serverless or CLI agent is a fresh process per invocation, so a
process-local registry would re-pay the model on *every* run — and, worse, treat every
learned predicate as multi-valued until it did, silently disabling contradiction
detection for anything written in that window. "Classified once, ever" has to mean across
processes to mean anything.

### Consolidation

Runs off the write path: decays salience toward a floor, merges near-duplicate claims into
a deterministic survivor (folding in their sources and observation counts), and promotes
repeatedly-observed episodic claims to semantic ones — seeing something once is an event,
seeing it five times is a pattern.

```python
mem.consolidate()   # {'decayed': 128, 'merged': 4, 'promoted': 2}
```

It is idempotent, which matters because it runs on a schedule. It also runs **windowed** —
committing every 500 rows rather than holding one transaction over the whole sweep, which
is what stops a large store's maintenance pass from locking out its own writes.

Salience follows Bjork & Bjork's new theory of disuse: storage strength (`salience_base`,
which never decays) is kept separate from retrieval strength (`salience`, derived from it).
A reinforcement bumps storage *inversely* to current retrievability, so re-encountering a
fact you were about to forget is worth more than re-encountering one that's already top of
mind — the spacing effect, which an exponential-decay-plus-flat-bump scheme gets backwards.

### It says when it is failing

Seven things can go wrong here without raising anything: predicate explosion,
reinforcement that never refreshes recency, flip-flop growth, salience overriding
relevance, a gate tuned for English silently dropping other scripts, a retraction that
quietly no-ops, and a **redaction policy that stops matching**. Each has a metric series.

The last one is the nastiest and the newest. A deployment configures a `Redactor`, it
works, and then the data drifts — a new phone format, a different locale, a vendor
changing an id shape. Nothing raises, nothing logs, and the write path gets *faster*. The
only symptom is unredacted PII on disk, found by an auditor. So `redact.inspected` and
`redact.changed` are emitted as a **pair**, tagged by field and by script: a count of
redactions alone cannot be read, because "zero today" is the silent failure and the
normal case at once. It is the *ratio*, sliced by script, that shows a rule set matching
a steady fraction of one population and nothing of another —
`私の電話は090-1234-5678です` is punctuated exactly like `555-123-4567` but grouped
3-4-4, so a rule written for the second misses the first entirely.

```python
from memvara import Memvara, MemoryRecorder

rec = MemoryRecorder()
mem = Memvara("memory.db", telemetry=rec)
mem.add(["I live in Berlin", "你好，我住在北京", "ok thanks"])

rec.total("fast.hit",  script="latin")   # 1  — extracted by rule, no model
rec.total("fast.miss", script="han")     # 1  — fell through to the model
rec.total("gate.drop", reason="ack_only")  # 1  — "ok thanks" carried nothing
```

Tags filter by subset, so `total("fast.miss")` is the whole series and
`total("fast.miss", script="han")` is one slice of it. The example above is the
English-centrism limitation showing up as a number: the Latin sentence is free, the Han
one costs a model call.

Two design choices make it honest. `retrieval.quality_factor` is emitted **unclamped**,
because a value above 1.0 is the alarm — only an over-reinforced salience can produce one,
and clamping it before recording would hide exactly the failure it exists to catch. And
`consolidate.merged` is emitted **at zero**, so "nothing to merge" is distinguishable from
"the scheduler stopped running."

The default is `None`, not a no-op recorder, and every metric that requires *computing*
something sits inside the `is not None` guard. Measured against a control built from this
tree with the emission points deleted: unset costs **+0.8% on write and −0.4% on read** —
inside the launch-to-launch spread rather than merely small.

---

## API

Every method takes `tenant=`/`user=`/`agent=`/`session=` to override the default scope,
omitted below for readability.

```python
mem = Memvara(path=":memory:", *, store=, embedder=, llm=, registry=, telemetry=,
             redactor=, tenant=, user=, agent=, session=, reembed=False, **tuning)

# write
mem.add(messages, *, role="user", ts=None)        -> WriteReceipt
mem.remember(subject, predicate, obj, *, valid_from=, valid_to=, recorded_at=, sources=,
             text=, confidence=, memory_type=, polarity=, extractor=, **meta)
                                                  -> WriteReceipt
mem.supersede(old_claim_id, new_claim, *, at=, sources=)   -> WriteReceipt

# retire — reversible, keeps history
mem.forget(subject, predicate, *, at=None)        -> list[Claim]    # a whole slot
mem.delete(claim_id, *, at=None)                  -> bool           # one claim

# erase — irreversible, removes the text itself
mem.erase(claim_id, *, sources=False)             -> bool           # one claim
mem.purge()                                       -> dict[str, int] # a whole scope
mem.reset()                                       -> dict[str, int] # scope + schema
#   `store.erase_claim` returns purge's four counts instead; `mem.erase` stays a bool
#   so that `if mem.erase(id):` keeps working — a dict of zeroes is truthy

# read
# every read below takes the same three time keywords, written `T=` here for width:
#   valid_at=  the world clock   known_at=  the belief clock   as_of=  both at once
# the first three also take `states=`, any non-empty subset of ("live", "ended",
# "retired"), defaulting to ["live"]; `include_invalidated=` is its two-valued alias.
mem.search(query, *, k=10, min_score=0.0, T=None, memory_types=None,
           states=None, include_invalidated=None, include_episodes=False)
                                                  -> list[Retrieved]
mem.recall(query, *, k=8, min_score=0.0, header=None, include_episodes=False,
           episode_header=None)                   -> str
mem.get(claim_id)                                 -> Claim | None
mem.get_all(*, T=None, states=None, include_invalidated=None)  -> list[Claim]
mem.count(*, T=None, states=None, include_invalidated=None)    -> int
mem.history(subject, predicate, *, T=None)        -> list[Claim]    # timeline of one slot
mem.why(claim_id, *, T=None)                      -> Provenance | None
mem.produced(episode_id, *, T=None)               -> list[Claim]    # why(), backwards

# traverse — the claims are a graph; walk it
mem.neighborhood(entity, *, depth=2, k=10, min_hops=1, predicates=None,
                 T=None, min_score=0.0)                    -> list[Path]
mem.paths_between(source, target, *, depth=3, k=3, predicates=None,
                  T=None, min_score=0.0)                   -> list[Path]

# maintenance
mem.consolidate()                                 -> dict[str, int]
mem.reembed(embedder=None)                        -> int            # after a model change
mem.stats()                                       -> dict[str, int]
#   episodes, claims, live_claims, ended_claims, invalidated, embeddings
#   these do not sum — see "Counting claims" above; `claims` is the only total
mem.scope(user="bob")                             -> ScopedMemvara   # same API, scope bound
mem.close()                                       -> None           # or use as a context manager
```

`add()` takes a string, a list of strings, pre-built `Episode`s, or OpenAI/mem0-style
`{"role": ..., "content": ...}` transcripts, so an existing agent loop can pass its
messages straight through.

`recall()` is the one you put in a prompt. It returns a framed block that labels itself as
retrieved data rather than instructions, and flattens each claim to a single line — a
memory whose text contains newlines and a fake section header cannot forge prompt
structure around itself.

### Two meanings of "delete", kept apart

`forget`/`delete` **retire**: the claim stops answering present-tense queries, and
`history()` and `as_of` still see it. That is the right default for correcting a belief,
and the wrong answer to "delete my data" — the text stays readable, which does not satisfy
a GDPR Article 17 request.

`erase`/`purge` **erase**, irreversibly, including everything derived from the text:
the claim, the FTS entry (which stores the tokens directly), the embedding (which leaks
content under inversion) and — with `sources=True`, or always for `purge` — the source
turns. `erase(sources=True)` only removes turns that no surviving claim still cites,
because one turn can source several claims.

### Scoping

`tenant > user > agent > session`, with inheritance. A query at session scope also sees
that user's durable memory, but never a sibling session's scratch space or another user's
anything. mem0's flat `user_id`/`agent_id`/`run_id` triple can't express that.

```python
bob = mem.scope(user="bob")     # the whole API, with the scope bound
bob.add("I live in Oslo")
```

Scope filters fail **closed**: a scope that resolves to nothing matches nothing, rather
than degrading into an unfiltered query across every user.

### Swapping backends

Everything is a protocol:

```python
Memvara(embedder=MyEmbedder(),      # anything with .dim and .encode(texts) -> (n, dim)
       llm=AnthropicLLM(),         # or your own .extract() / .classify_predicate()
       store=MyPgVectorStore())    # see memvara/store/base.py
```

Defaults are `HashingEmbedder` + `NullLLM` + `SQLiteStore` — so `Memvara()` constructs and
works with zero configuration. To use a real model:

```python
from memvara import AnthropicLLM, Memvara          # pip install 'memvara[anthropic]'
mem = Memvara("memory.db", llm=AnthropicLLM(model="claude-opus-5"))

from memvara import OpenAILLM                     # pip install 'memvara[openai]'
mem = Memvara("memory.db", llm=OpenAILLM(model="gpt-4.1"))
```

Both are lazy attributes: naming one does not import its SDK, so the default offline
install stays a two-package install (`memvara` and `numpy`, verified in CI). Each backend
is transport and response-shape only — every rule about what counts as a valid claim is
shared in `memvara/llm/_shape.py`, so the same turn produces the same claim regardless of
which model wrote it.

### Concurrency

The library is synchronous, and reads no longer queue behind writes. Read statements use a
per-thread connection, and the slow half of a write — the near-duplicate encode and the
model call — runs with no transaction open, so the store's write lock is held for the
database work and nothing else.

One reader thread against a 20,000-claim consolidation sweep:

| | before | after |
|---|---:|---:|
| reads completed during the sweep | 1,470 | **13,728** |
| p95 | 3.44 ms | **0.31 ms** |
| p99 | 30.4 ms | **2.01 ms** |

Idle read latency is unchanged (12.7 µs → 13.0 µs), so this was not taken from the write
path. The sweep itself goes 2.2 s → 2.8 s *with a reader beside it*, because the reader is
now doing about 9× the work instead of waiting.

For an asyncio application, `AsyncMemvara` wraps each method over `asyncio.to_thread`:

```python
from memvara import AsyncMemvara, Memvara

mem = AsyncMemvara(Memvara("memory.db", user="alice"))
await mem.add("I live in Berlin")
[r.text for r in await mem.search("where do they live?")]

bob = mem.scope(user="bob")          # -> AsyncScopedMemvara, the same API, scope bound
await bob.add("I live in Oslo")
```

It wraps an `Memvara` rather than constructing one, so the sync object stays available for
setup and for the calls that have no async form.

`scope()` is the one method that is not a coroutine — it binds four strings and touches
no store — and it is the shape a server wants: one handle per request, with the four
scope keywords written once instead of on every call.

It is a thread-pool wrapper, not an async rewrite, and says so: SQLite has no async
driver worth the name, and the work here is CPU and disk rather than network.

---

## Beyond the library

### MCP server

```bash
MEMVARA_DB=/path/to/memory.db memvara-mcp                   # JSON-RPC 2.0 over stdio
MEMVARA_DB=/path/to/memory.db python3 -m memvara.server    # the same thing, no console script
```

Eight tools — `memory_add`, `memory_remember`, `memory_recall`, `memory_search`,
`memory_history`, `memory_why`, `memory_forget`, `memory_stats`. Hand-rolled against the
MCP wire format rather than taking an SDK dependency, so the library's "numpy and nothing
else" claim survives. It refuses to start without `MEMVARA_DB` and prints the client config
block, rather than silently remembering into a store that vanishes on exit.

`consolidate`, `purge`, `reset` and `erase` are deliberately **absent**, and a test
asserts their absence: a model that can be talked into calling a tool should not be able
to reach one that irreversibly erases a scope. Run those from the library, on a schedule
you control. `memory_forget` is present because retirement is recoverable.

### Running an existing mem0 app

```python
from memvara.compat import Memory          # mem0's method surface, backed by memvara
api = Memory(user_id="alice")
api.add("I live in Berlin")
api.search("where do they live?")
```

Written against mem0 2.x. Calls with no honest translation — `update()`, `from_config()` —
raise and explain why, rather than returning something plausible. A shim that quietly means
something else is worse than no shim, because the difference surfaces as data loss months
later.

### Importing a mem0 store

```python
from memvara.compat import import_mem0
receipt = import_mem0(mem, history_db="~/.mem0/history.db")
```

The interesting part is that `history.db` — mem0's own mutation log — is a complete
transaction-time history that mem0 itself cannot query. Replaying it through a bitemporal
store turns it into `search(as_of=…)`, `history()` and `why()`. **Phase 1 is lossless and
costs zero tokens**; extraction into real triples is opt-in.

The receipt names every slot left holding more than one live value, undeclared predicates
first. mem0 cannot produce that list — its conflicts are settled per-write by a model
looking at a top-k, and nothing ever looks again.

---

## Honest limitations

- **`HashingEmbedder` is a lexical fallback, not a semantic model.** It's the default so
  the library runs offline in milliseconds with no download, and it makes tests
  deterministic. It will not put "physician" near "doctor". Install
  `memvara[local-embed]` or pass your own embedder for real semantic recall.
- **Two benchmarks, and only one of them runs the real thing.** `bench/mem0_real.py`
  drives the actual `mem0ai` package; `bench/compare.py` drives `bench/baseline.py`, a
  reimplementation of mem0's documented architecture, and is kept because it can vary
  parameters (top-k, threshold, chitchat ratio) that the real package does not expose.
  Both share one extraction oracle, so both isolate architecture from model quality — and
  neither says anything about end-to-end answer quality.
- **The LOCOMO / LongMemEval numbers above are retrieval, not accuracy.** They are real
  and they run free, but they are not the metric those papers report and must never be
  quoted as if they were. Closing that gap needs a reader model: ~$17.50 for LOCOMO and
  ~$5 for LongMemEval on `claude-opus-5`, or ~$2 for a stratified sample.
  The harness reports a `none` / `memory` / `full` triple when a reader *is* configured,
  on purpose: a memory score with no reader-only floor and no whole-haystack ceiling
  beside it is uninterpretable, and stuffing the transcript into the reader is measurable
  as `full`, labelled a reader ceiling rather than a result.
- **The vector index is exact and in-process.** A numpy matmul over the candidate set —
  correct and fast to roughly a million claims, at which point the `Store` protocol is
  where pgvector or Qdrant goes.
- **Predicate schema, the salience gate and the fast extractor are English-centric.** The
  schema grows by learning, but the seed set is small on purpose, and the gate's and
  extractor's rules are English sentence forms. On other scripts they fall through to the
  model — which is correct behavior and a real cost. This is the one limitation the
  telemetry measures directly: `gate.drop` and `fast.miss` are tagged by script, so the
  gap is visible rather than assumed.
- **Entity resolution folds surface forms, it does not know the world.** `Acme Corp` and
  `acme, inc.` collapse; `Big Blue` and `IBM` do not, unless you enable the opt-in model
  path or declare the alias. `Stark` versus `Stark Industries` is genuinely ambiguous and
  is left that way.
- **`AsyncMemvara` is a thread-pool wrapper, not an async rewrite.** It keeps an asyncio
  event loop unblocked, which is what it is for; it does not make the store itself async.
- **With no `llm=`, `add()` keeps only what its rules recognise.** The default `NullLLM`
  runs tiers 0, 1 and 1b and then stops, so high-precision sentence forms ("I live in X",
  "I work at X") are extracted for nothing and an employer mentioned in passing is
  dropped. It is loud rather than silent — `Memvara()` warns once with a
  `DegradedExtractionWarning`, and `WriteReceipt.unextracted` counts the dropped turns on
  every write — but it is the qualifier on the offline claim: the *library* runs with no
  API key, extraction from arbitrary prose does not. `remember()`, retrieval,
  contradiction resolution and consolidation never needed a model.
- **No REST server in the open core.** MCP over stdio is the shipped remote surface. A
  REST API is a component of the commercial product rather than a gap in this one — see
  [Open core](#open-core-and-exactly-where-the-line-is), which says where that line is and
  why it does not move.
- **The framework adapters do not all preserve what makes memvara different.** LangChain
  and LlamaIndex *retrievers* keep everything, including `as_of=`, because "query in,
  documents out" is what `search()` already is. A LangChain `ChatMessageHistory` keeps
  the write path and loses the rest: a `list[BaseMessage]` has nowhere to put a
  supersession, a valid-time interval or a source id, and tier-0 dedupe means it is not
  a faithful transcript either. CrewAI loses the headline feature outright — its unit of
  memory is an opaque sentence with no subject or predicate, so the keyed lookup has
  nothing to key on and "Alice lives in Berlin" and "Alice moved to Lisbon" both stay
  live. **LangGraph loses least of the four**, and instructively: `BaseStore` is the only
  interface that hands over the query text natively, *and* `put(namespace, key, value)`
  supplies all three parts of a triple — so an item is stored as one claim per field and
  changing `city` retires exactly `city`, which is contradiction resolution surviving a
  foreign interface intact. What it loses is the predicate registry: a stored `home_city`
  does not contradict an extracted `lives_in`. Each adapter says which it is; see
  `memvara/integrations/`.
- **No encryption at rest.** `purge()`, `erase()` and the redaction hook cover the
  deletion and ingestion halves of a privacy story; the storage half is the deployment's
  problem, and full-disk encryption is the honest answer today. It is not laziness:
  SQLCipher works here — measured, +43–48% on writes, search unchanged, and FTS5 keeps
  working because page-level encryption sits *beneath* SQLite — but the mmap-backed
  `.vecs` sidecar stays plaintext outside that boundary, and a plaintext vector is a
  confirmation oracle. Encoding a guess and taking the cosine against that file returns
  exactly 1.0000 for the right text and 0.87 for a one-digit-different phone number, so
  it is not merely confirmable, it is hill-climbable. Encrypting the text and not the
  vectors would be theatre.
- **The built-in redactor is not compliance-grade** and says so in its own docstring. It
  is a default, not a product: the seam is the deliverable, and a serious deployment
  brings its own `Redactor`.

---

## Development

```bash
python3 -m pytest -q                              # 2,329 tests, offline, no API key
python3 -m coverage run -m pytest && python3 -m coverage report   # gated at 100%
PYTHONPATH=. python3 bench/compare.py             # architecture comparison
PYTHONPATH=. python3 bench/perf.py                # throughput and scaling
```

**100% statement coverage, enforced** (`fail_under = 100`), and `mypy -p memvara` is
clean in CI. The suite runs in about 21
seconds with no network, no API key, and almost no sleeping — time is controlled by
passing explicit `datetime` values rather than patching the clock, and the handful of
tests that do sleep are measuring concurrency, where the wall clock is the thing under
test.

Coverage of the *lines* is the floor, not the goal. What the suite actually pins down:

- **Behavior** — contradictions resolve, history survives, users are isolated in all
  three directions (sibling session, sibling agent, other tenant), and the LLM stays idle.
  Fakes count their own calls, and the tests assert on those counts — the design claim is
  that the model is rarely consulted, so a test that doesn't count calls doesn't test it.
- **Failure paths** — dimension mismatches, transaction rollback (including nested),
  a classifier that raises, a store that loses rows mid-query, and model output that
  violates every field contract at once. These only run during an incident, which is
  exactly why they can't ship unexercised.
- **Adversarial input** — a fuzz corpus (SQL and FTS5 injection, path traversal, template
  injection, control characters, astral-plane codepoints, 5KB strings, combining marks)
  driven through every public method and a persistence round trip, plus randomized
  transcripts asserting the store never ends up internally inconsistent.
- **Executable docs** — the README walkthrough and the `Memvara` docstring run as tests, so
  the examples can't drift from the code.

The twelve remaining *branch* partials are verified-unreachable defensive guards — mostly
`if valid_to is None or valid_to > t`, where a live claim always satisfies the first
disjunct, so the second can never decide the branch. They are kept as guards rather than
deleted, and documented as such.

Design notes and the module-by-module contract live in [docs/INTERNALS.md](docs/INTERNALS.md).
[docs/UPGRADING.md](docs/UPGRADING.md) is the short list of changes that do not announce
themselves — read it before upgrading, starting with the one where `invalidated_at is
None` stopped meaning "live" without breaking anything.
[CONTRIBUTING.md](CONTRIBUTING.md) covers the bar a patch has to clear and what will and
will not be accepted; [SECURITY.md](SECURITY.md) covers private vulnerability reporting.

## License

Apache-2.0, for everything in this repository. See
[Open core](#open-core-and-exactly-where-the-line-is) for what is and is not in it.
