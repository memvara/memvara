# Memvara

The hosted service lives at:

> ## 🔗 [memvara.dev](https://memvara.dev)
>
> ### 🤖 MCP client setup — [memvara.dev/docs/agents](https://memvara.dev/docs/agents)
>
> Plugin (Claude Code): `/plugin marketplace add memvara/claude-memvara` then `/plugin install memvara`. Other agents: [memvara.dev/docs/agents](https://memvara.dev/docs/agents).

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

## What it does

| | |
|---|---|
| 🕰️ **Two clocks, not one** | When it was true, and when you learned it — independently. Ask what you believed in March about June, and get an answer that is not a guess. |
| ⚖️ **Contradictions resolve without a model** | Cardinality is a schema property, so a conflict is an indexed lookup. Same two facts, same result, every run. |
| 🔌 **Offline by default** | numpy and nothing else. No API key, no Docker, no vector database, no network on the write path. |
| 🧾 **Nothing is silently lost** | Every write returns a receipt saying what it did — including what it could *not* extract. |
| 🔍 **Hybrid retrieval that explains itself** | Vector and BM25, time-aware, and every score is inspectable rather than a ranking you have to trust. |
| 🧬 **Claims are a graph** | Walk relationships, at a point in time — and optionally fuse that walk into search as a third retrieval leg. |

## Where to start

<table>
<tr>
<td width="50%" valign="top">

### 🧑‍💻 I use an AI coding tool

Nothing to install and nothing to run.

```
/plugin marketplace add memvara/claude-memvara
/plugin install memvara
```

Cursor, Codex, Grok, VS Code and OpenCode have their own
one-liners at [memvara.dev/docs/agents](https://memvara.dev/docs/agents).
Claude Desktop and ChatGPT paste the same URL.

</td>
<td width="50%" valign="top">

### 🔧 I'm building on it

```bash
pip install memvara
```

The library is the product; the hosted service runs this
same code. Start with [`docs/API.md`](docs/API.md), then
[`docs/DESIGN.md`](docs/DESIGN.md) for why it is shaped
that way.

</td>
</tr>
<tr>
<td colspan="2" valign="top">

### 🖥️ I want it on my own machine

```bash
MEMVARA_DB=~/memory.db memvara-mcp        # JSON-RPC 2.0 over stdio
memvara-mcp init --agent claude           # writes the client block, skill tree and note
```

Or, with no Python at all — `npx memvara` bridges a stdio MCP client to the
hosted service and signs you in on first run. It is a way *in*, not a second
implementation: the engine is this library.

Thirteen tools — `memory_add`, `memory_remember`, `memory_recall`, `memory_search`,
`memory_neighborhood`, `memory_paths`, `memory_since`, `memory_standing`,
`memory_history`, `memory_why`, `memory_forget`, `memory_end`, `memory_stats`.
Hand-rolled against the MCP wire format
rather than taking an SDK
dependency, so the "numpy and nothing else" claim survives the server too.
[`docs/DEPLOY.md`](docs/DEPLOY.md) covers running it for other people.

</td>
</tr>
</table>

## Teach it your vocabulary

The built-in predicates are a personal-assistant vocabulary — where someone lives, where
they work. A store of engineering facts matches none of them, and an unknown predicate
takes the safe default twice over: multi-valued, so nothing supersedes, and slow-decaying,
so this morning's deploy still ranks as fresh in two years.

```bash
MEMVARA_PREDICATES=engineering memvara-mcp        # or: engineering,./ours.toml
```

```toml
[[predicate]]
name = "git_state"
cardinality = "one"     # supersedes; "many" accumulates
volatility = "fast"     # static | slow | fast -> 36500 | 730 | 7 day half-life
```

A declaration outranks a guess, so a pack corrects a store that already classified
something wrongly rather than only shaping a fresh one.

## Coming from somewhere else

```python
from memvara.compat import import_mem0, import_supermemory

import_mem0(mem, history_db="~/.mem0/history.db")   # replays mem0's own mutation log
import_supermemory(mem)                             # reads the Supermemory export API
```

mem0 records what changed and when, so that import rebuilds supersession and answers
`as_of` afterwards. Supermemory records current state, so its documents arrive as episodes
on their original timestamps and nothing invents a history it was never told. There is
also a method-level mem0 shim if you want its call surface on this store.

## Measured

| | |
|---|---|
| Against the real `mem0ai` package | [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) |
| LOCOMO and LongMemEval, retrieval | [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) |
| Answer quality, end to end | [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) |

The harnesses are in `bench/` and `demo/` and every number is reproducible from this
repository. Where a result is synthetic or self-authored, its own heading says so.

## Documentation

| | |
|---|---|
| [`docs/API.md`](docs/API.md) | The whole surface, in the order you meet it |
| [`docs/DESIGN.md`](docs/DESIGN.md) | How it works, and the failure each decision prevents |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | Every measured claim, with its method |
| [`docs/INTERNALS.md`](docs/INTERNALS.md) | Module map and invariants |
| [`docs/DEPLOY.md`](docs/DEPLOY.md) | Running it for other people |
| [`docs/UPGRADING.md`](docs/UPGRADING.md) | What changed under you |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | What is done, deferred, and still missing |
| [`docs/OPEN-CORE.md`](docs/OPEN-CORE.md) | What is Apache-2.0, and what is not |

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
  neither says anything about end-to-end answer quality. That is what `demo/` is for, and
  its one run is a sanity check with an agent as the reader, not a benchmark.
- **The LOCOMO / LongMemEval numbers above are retrieval, not accuracy.** They are real
  and they run free, but they are not the metric those papers report and must never be
  quoted as if they were. Closing that gap needs a reader model. Measured, on
  `claude-opus-5`: **$7–$31 for LOCOMO** and **$3–$9 for LongMemEval**, the spread being
  thinking tokens rather than answers, plus a few dollars for `--judge llm`; a
  stratified `--shuffle 7 --limit 200` sample is about a tenth of that and finishes in
  twenty minutes rather than hours. The full procedure — flags, key variable, order of
  operations, worked example, and where each number came from — is in one place, the
  module docstring of `bench/evalkit.py`. It is deliberately not restated here, because
  it was previously stated in four places and three of them drifted: the "$17.50" this
  bullet used to carry assumed twice the input tokens the harness actually sends.
  The harness reports a `none` / `memory` / `full` triple when a reader *is* configured,
  on purpose: a memory score with no reader-only floor and no whole-haystack ceiling
  beside it is uninterpretable, and stuffing the transcript into the reader is measurable
  as `full`, labelled a reader ceiling rather than a result.
- **LOCOMO and LongMemEval are public, and a good end-to-end score on them proves less
  than it looks.** Any reader model may have seen them in training, and nothing in the
  harness can distinguish a retrieved answer from a remembered one. The asymmetry is
  the usable part: a *strong* score is weak evidence (contamination inflates), and a
  *weak* score is strong evidence against us. This is why the `--context none` floor is
  reported beside every score, and why a purpose-written scenario with no such confound
  is built separately rather than instead.
- **The vector index is exact and in-process.** A numpy matmul over the candidate set —
  correct and fast to roughly a million claims, at which point the `Store` protocol is
  where pgvector or Qdrant goes.
- **Predicate schema, the salience gate and the fast extractor are English-centric.** The
  schema grows by learning, but the seed set is small on purpose, and the gate's and
  extractor's rules are English sentence forms. On other scripts they fall through to the
  model — which is correct behavior and a real cost. This is the limitation the telemetry
  measures directly: `gate.drop` and `fast.miss` are tagged by script, so the gap is
  visible rather than assumed.
- **The default embedder is worse than English-centric — it is Latin-only.**
  `HashingEmbedder`, what you get with no extras installed, tokenises `[a-z0-9']+` and
  builds its character n-grams over the rejoined word list, so text in Han, Kana, Hangul,
  Arabic or Hebrew produces an **all-zero vector**. Retrieval handles that honestly — it
  abstains on a zero norm rather than inventing a rank — so such a claim is stored, is
  reachable by predicate, and is never returned by meaning. A write like that now warns
  (`UnembeddableTextWarning`) and counts (`write.embedding_unusable`, tagged by script).
  Mixed text is affected without being caught: `user lives in 里斯本` embeds fine, from
  the Latin half alone. Installing `memvara[local-embed]` gets a real model and non-zero
  vectors; genuine cross-language retrieval needs a multilingual model and is not claimed.
- **Entity resolution folds surface forms, it does not know the world.** `Acme Corp` and
  `acme, inc.` collapse; `Big Blue` and `IBM` do not, unless you enable the opt-in model
  path or declare the alias. `Stark` versus `Stark Industries` is genuinely ambiguous and
  is left that way.
- **`AsyncMemvara` is a thread-pool wrapper, not an async rewrite.** It keeps an asyncio
  event loop unblocked, which is what it is for; it does not make the store itself async.
- **With no `llm=`, `add()` keeps only what its rules recognise — and on some corpora that
  is nothing.** The default `NullLLM` runs tiers 0, 1 and 1b and then stops, so
  high-precision sentence forms ("I live in X", "I work at X") are extracted for nothing
  and an employer mentioned in passing is dropped. Measured on `demo/`'s 64-turn support
  history: **64 episodes, 0 claims** — the rules matched not one turn, so that store does
  no supersession and no bitemporal reasoning at all. It is loud rather than silent —
  `Memvara()` warns once with a `DegradedExtractionWarning`, and
  `WriteReceipt.unextracted` counts the dropped turns on every write — but it is the
  qualifier on the offline claim: the *library* runs with no API key, extraction from
  arbitrary prose does not. `remember()` with a declared `PredicateSpec` is the offline
  way to get the full machine, and it is what a real integration does; see
  [What the fast path does not catch](#what-the-fast-path-does-not-catch-measured).
  Retrieval, contradiction resolution and consolidation never needed a model.
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
python3 -m pytest -q                              # 3,492 tests, offline, no API key
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
