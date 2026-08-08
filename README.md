# Engram

**Bitemporal memory for AI agents.** Structured facts, deterministic contradiction
resolution, hybrid retrieval, and a write path that mostly doesn't call an LLM.

```bash
pip install -e .
```

```python
from engram import Engram

mem = Engram("memory.db", user="alice")

mem.add("I live in Berlin and work at Acme")
mem.add("Actually, I moved to Lisbon last month")

mem.search("where do they live?")     # -> Lisbon
mem.history("user", "lives_in")       # -> Berlin (retired 2026-08-08), Lisbon (current)
mem.search("where do they live?", as_of=last_year)   # -> Berlin
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

## Measured

`python3 bench/compare.py` — 105-turn transcript, 21 turns carrying a durable fact,
10 distinct facts, several revised two or three times over the conversation:

| metric | mem0-style | engram |
|---|---:|---:|
| LLM calls on the write path | 126 | **2** |
| Current value stored correctly | 10/10 | 10/10 |
| **Stale values left live** | **7** | **0** |
| Live memories (want 10) | 17 | **10** |
| End-to-end @ 800ms/call | 101 s | **2 s** |

The row that matters is the third. Both systems know the right answer — but the baseline
also still holds seven superseded values, so it answers the same question correctly and
incorrectly at once, and which one you get depends on what embeds closest. Those seven
contradictions were invisible to top-k adjudication; a keyed lookup catches them by
construction.

At a more realistic 1:12 chitchat ratio the write-path gap widens to **294 calls vs 2**.

### Throughput

`PYTHONPATH=. python3 bench/perf.py` — single process, in-memory store, no LLM:

| @ 8,000 claims | per op | scaling per 4x data |
|---|---:|---|
| write | 0.13 ms (**~8,000/s**) | flat |
| search k=10 | 1.3 ms (~770/s) | sub-linear |
| consolidation sweep | 336 ms | linear |

Two algorithmic fixes got it there, both found by profiling rather than guessing:

- **The FTS index was keyed on an `UNINDEXED` column.** `DELETE FROM claims_fts WHERE
  claim_id = ?` on every write was a full scan of the text index, making N writes over N
  rows **O(n²)** — it dominated everything else at 80% of consolidation time. Mirroring
  the claim's rowid into the FTS table makes the delete an indexed lookup. Consolidation
  went from 4.8 s to 336 ms at 8k claims, and from degrading 11x per 4x of data to
  exactly 4x — i.e. from quadratic to linear, which is optimal for a full sweep.
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

### Consolidation

Runs off the write path: decays salience toward a floor, merges near-duplicate claims into
a deterministic survivor (folding in their sources and observation counts), and promotes
repeatedly-observed episodic claims to semantic ones — seeing something once is an event,
seeing it five times is a pattern.

```python
mem.consolidate()   # {'decayed': 128, 'merged': 4, 'promoted': 2}
```

It is idempotent, which matters because it runs on a schedule.

---

## API

```python
mem = Engram(path=":memory:", *, store=, embedder=, llm=, registry=,
             tenant=, user=, agent=, session=)

# write
mem.add(messages, *, user=, session=, ...)        -> WriteReceipt
mem.remember(subject, predicate, obj, ...)        -> WriteReceipt   # structured, no LLM
mem.forget(subject, predicate)                    -> list[Claim]    # retire, keep history

# read
mem.search(query, *, k=10, as_of=None, memory_types=None)  -> list[Result]
mem.recall(query)                                 -> str            # prompt-ready block
mem.get_all(*, as_of=None, include_invalidated=False)      -> list[Claim]
mem.history(subject, predicate)                   -> list[Claim]    # timeline of one slot
mem.why(claim_id)                                 -> Provenance

# maintenance
mem.consolidate()                                 -> dict[str, int]
mem.stats()                                       -> dict[str, int]
```

`add()` takes a string, a list of strings, pre-built `Episode`s, or OpenAI/mem0-style
`{"role": ..., "content": ...}` transcripts, so an existing agent loop can pass its
messages straight through.

### Scoping

`tenant > user > agent > session`, with inheritance. A query at session scope also sees
that user's durable memory, but never a sibling session's scratch space or another user's
anything. mem0's flat `user_id`/`agent_id`/`run_id` triple can't express that.

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
- **Predicate schema is English-centric** and seeded for the personal-assistant domain. It
  grows by learning, but the built-in seed set is small on purpose.

---

## Development

```bash
python3 -m pytest -q                              # 775 tests, offline, no API key
python3 -m coverage run -m pytest && python3 -m coverage report   # gated at 100%
PYTHONPATH=. python3 bench/compare.py             # architecture comparison
PYTHONPATH=. python3 bench/perf.py                # throughput and scaling
```

**100% statement coverage, enforced** (`fail_under = 100`). The suite runs in under two
seconds with no network, no API key, and no sleeping — time is controlled by passing
explicit `datetime` values rather than patching the clock.

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

The seven remaining *branch* partials are verified-unreachable defensive guards (for
example: a live claim always has `valid_to` unset or in the future, so that check can
never be false). They are kept as guards rather than deleted, and documented as such.

Design notes and the module-by-module contract live in [docs/INTERNALS.md](docs/INTERNALS.md).

## License

Apache-2.0
