# Engram

**Bitemporal memory for AI agents.** Structured facts, deterministic contradiction
resolution, hybrid retrieval, and a write path that mostly doesn't call an LLM.

```bash
pip install -e .
```

```python
from datetime import datetime, timedelta, timezone
from engram import Engram

now = datetime.now(timezone.utc)
mem = Engram("memory.db", user="alice")

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

Core requires **numpy and nothing else**. It runs offline, with no API key, no Docker,
and no vector database.

---

## Why this exists

mem0 and its descendants store a memory as an opaque string with an embedding. Every
`add()` costs two LLM calls — one to extract facts, one to decide ADD/UPDATE/DELETE
against whatever the vector search happened to return. Retrieval is vector top-k.

That design has four consequences that show up in production:

1. **Contradictions leak.** Conflict detection is bounded by retrieval recall. If the
   memory that contradicts the new fact isn't in the top-k, nothing catches it and both
   survive. Six months in, the store holds three cities for one person and returns
   whichever embeds closest to the question.
2. **Writes are slow and expensive.** Two model calls per turn, on the critical path,
   including for "ok, thanks."
3. **There is no time.** One `updated_at` column can't answer "where did she live in
   March?" or absorb a fact that arrives late about the past.
4. **Nothing explains itself.** When the agent says something wrong, you cannot ask which
   memory caused it, where that memory came from, or why it ranked first.

Engram is built around the observation that **most of this doesn't need a model at all.**

---

## A design comparison (synthetic, self-authored)

Not an external benchmark. One workload, n=1, written by the same people who wrote the
system being measured and the system it is measured against. Read this section as an
illustration of a mechanism, not as evidence of superiority.

`PYTHONPATH=. python3 bench/compare.py` — 105-turn transcript, 21 turns carrying a
durable fact, 10 distinct facts, several revised two or three times:

| metric | mem0-style | engram |
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

**The call-count gap is mostly an ingestion-granularity choice.** Engram receives the
whole transcript in one `add()` and batches extraction; the baseline is charged per turn.
At equal per-turn granularity it is 126 vs 17, not 126 vs 2. The gap also scales linearly
with the chitchat ratio, which is a parameter we picked: 1:0 → 21x, 1:4 → 63x, 1:12 →
147x, 1:100 → 1071x, with identical information content at every point.

**Engram loses the local-compute row** — roughly 3x slower per operation, because it does
strictly more work (FTS indexing, reconciliation, bitemporal filtering). That trade is
worth it only when model calls dominate, which is the normal case but not a universal one.

**What this does not measure:** end-to-end answer quality. Both systems are driven by the
same perfect extraction oracle, which neither would have in production. 9 of the 10
predicates ship pre-seeded in the registry with the right cardinality, so the benchmark
never exercises the path where an unknown predicate defaults to multi-valued and
accumulates. And there are no LOCOMO or LongMemEval numbers yet — that is a gap in the
work, not a limitation of the environment.

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
from engram import entity_key
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

Most conversational turns carry nothing durable. mem0 pays two model calls for "sounds
good"; Engram pays zero. Every `add()` returns a receipt that reports the cost, because a
number you can't see is a number nobody optimizes:

```python
receipt = mem.add(transcript)
print(receipt)   # <WriteReceipt +3 ~1 -1 skip=17 llm=1 42.3ms>
#                    added ─┘  │  │      │      └─ one batched call for 21 turns
#                 reinforced ──┘  │      └─ carried no durable fact
#                    retired ─────┘
```

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

### Nothing is silently lost

Superseding sets an end timestamp; it never deletes. So the audit trail is free:

```python
for c in mem.history("user", "works_at"):
    print(c.object, c.recorded_at.date(), "-> retired by", c.invalidated_by)

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

Six things can go wrong here without raising anything: predicate explosion, reinforcement
that never refreshes recency, flip-flop growth, salience overriding relevance, a gate
tuned for English silently dropping other scripts, and a retraction that quietly no-ops.
Each now has a metric series.

```python
from engram import Engram, MemoryRecorder

rec = MemoryRecorder()
mem = Engram("memory.db", telemetry=rec)
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
mem = Engram(path=":memory:", *, store=, embedder=, llm=, registry=, telemetry=,
             tenant=, user=, agent=, session=)

# write
mem.add(messages, *, role="user", ts=None)        -> WriteReceipt
mem.remember(subject, predicate, obj, *, valid_from=, recorded_at=, sources=,
             text=, confidence=, memory_type=, polarity=, **meta)  -> WriteReceipt
mem.supersede(old_claim_id, new_claim, *, at=, sources=)   -> WriteReceipt

# retire — reversible, keeps history
mem.forget(subject, predicate, *, at=None)        -> list[Claim]    # a whole slot
mem.delete(claim_id, *, at=None)                  -> bool           # one claim

# erase — irreversible, removes the text itself
mem.erase(claim_id, *, sources=False)             -> bool           # one claim
mem.purge()                                       -> dict[str, int] # a whole scope
mem.reset()                                       -> dict[str, int] # scope + schema

# read
mem.search(query, *, k=10, min_score=0.0, as_of=None, memory_types=None,
           include_invalidated=False, include_episodes=False)  -> list[Retrieved]
mem.recall(query, *, k=8, header=None, include_episodes=False)  -> str
mem.get(claim_id)                                 -> Claim | None
mem.get_all(*, as_of=None, include_invalidated=False)      -> list[Claim]
mem.count(*, as_of=None, include_invalidated=False)        -> int
mem.history(subject, predicate)                   -> list[Claim]    # timeline of one slot
mem.why(claim_id)                                 -> Provenance | None

# maintenance
mem.consolidate()                                 -> dict[str, int]
mem.reembed(embedder=None)                        -> int            # after a model change
mem.stats()                                       -> dict[str, int]
mem.scope(user="bob")                             -> ScopedEngram   # same API, scope bound
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
Engram(embedder=MyEmbedder(),      # anything with .dim and .encode(texts) -> (n, dim)
       llm=AnthropicLLM(),         # or your own .extract() / .classify_predicate()
       store=MyPgVectorStore())    # see engram/store/base.py
```

Defaults are `HashingEmbedder` + `NullLLM` + `SQLiteStore` — so `Engram()` constructs and
works with zero configuration. To use a real model:

```python
from engram import Engram
from engram.llm.anthropic import AnthropicLLM      # pip install 'engram[anthropic]'

mem = Engram("memory.db", llm=AnthropicLLM(model="claude-opus-5"))
```

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

For an asyncio application, `AsyncEngram` wraps each method over `asyncio.to_thread`:

```python
from engram import AsyncEngram, Engram

mem = AsyncEngram(Engram("memory.db", user="alice"))
await mem.add("I live in Berlin")
[r.text for r in await mem.search("where do they live?")]
```

It wraps an `Engram` rather than constructing one, so the sync object stays available for
setup and for the calls that have no async form.

It is a thread-pool wrapper, not an async rewrite, and says so: SQLite has no async
driver worth the name, and the work here is CPU and disk rather than network.

---

## Beyond the library

### MCP server

```bash
ENGRAM_DB=/path/to/memory.db python3 -m engram.server    # JSON-RPC 2.0 over stdio
```

Eight tools — `memory_add`, `memory_remember`, `memory_recall`, `memory_search`,
`memory_history`, `memory_why`, `memory_forget`, `memory_stats`. Hand-rolled against the
MCP wire format rather than taking an SDK dependency, so the library's "numpy and nothing
else" claim survives. It refuses to start without `ENGRAM_DB` and prints the client config
block, rather than silently remembering into a store that vanishes on exit.

`consolidate`, `purge`, `reset` and `erase` are deliberately **absent**, and a test
asserts their absence: a model that can be talked into calling a tool should not be able
to reach one that irreversibly erases a scope. Run those from the library, on a schedule
you control. `memory_forget` is present because retirement is recoverable.

### Running an existing mem0 app

```python
from engram.compat import Memory          # mem0's method surface, backed by engram
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
from engram.compat import import_mem0
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
  `engram[local-embed]` or pass your own embedder for real semantic recall.
- **The benchmark's comparison target is a reimplementation** of mem0's documented
  architecture (`bench/baseline.py`), not the mem0 package — so the numbers characterize
  the design difference, not a head-to-head against a running install. Both systems share
  one extraction oracle so the comparison isolates architecture from model quality.
- **No LOCOMo / LongMemEval numbers yet.** Those need network access and an API key.
  The harness is the natural next step, not a completed result.
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
- **`AsyncEngram` is a thread-pool wrapper, not an async rewrite.** It keeps an asyncio
  event loop unblocked, which is what it is for; it does not make the store itself async.
- **No REST server yet** — MCP over stdio is the shipped remote surface. The library is
  the supported integration point, and framework adapters (LangChain, LlamaIndex, CrewAI)
  are not written.
- **No encryption at rest and no PII redaction hook.** `purge()` and `erase()` cover the
  deletion half of a privacy story; the storage half is the deployment's problem today.

---

## Development

```bash
python3 -m pytest -q                              # 1,657 tests, offline, no API key
python3 -m coverage run -m pytest && python3 -m coverage report   # gated at 100%
PYTHONPATH=. python3 bench/compare.py             # architecture comparison
PYTHONPATH=. python3 bench/perf.py                # throughput and scaling
```

**100% statement coverage, enforced** (`fail_under = 100`). The suite runs in about 17
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
- **Executable docs** — the README walkthrough and the `Engram` docstring run as tests, so
  the examples can't drift from the code.

The ten remaining *branch* partials are verified-unreachable defensive guards — mostly
`if valid_to is None or valid_to > t`, where a live claim always satisfies the first
disjunct, so the second can never decide the branch. They are kept as guards rather than
deleted, and documented as such.

Design notes and the module-by-module contract live in [docs/INTERNALS.md](docs/INTERNALS.md).

## License

Apache-2.0
