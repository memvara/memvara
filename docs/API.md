# API

The whole surface, in the order you meet it. `docs/DESIGN.md` explains why it is
shaped this way; this file is the reference.

## API

Every method takes `tenant=`/`user=`/`agent=`/`session=` to override the default scope,
omitted below for readability.

```python
mem = Memvara(path=":memory:", *, store=, embedder=, llm=, registry=, telemetry=,
             redactor=, tenant=, user=, agent=, session=, reembed=False, **tuning)
# api_key= or base_url= instead returns a RemoteMemvara — see "A hosted deployment" below

# write
mem.add(messages, *, role="user", ts=None)        -> WriteReceipt
    # role decides what is EXTRACTED, not just attribution: only "user" is read.
    # Pass role="system" for a document, a log or a paste -- stored and cited, nothing
    # extracted. See Memvara.add's docstring for why quoting is not a defence.
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
# a read shown with `T=` below takes the same three time keywords, written `T=` for
# width; a read shown without one takes no time keyword at all:
#   valid_at=  the world clock   known_at=  the belief clock   as_of=  both at once
# the first three also take `states=`, any non-empty subset of ("live", "ended",
# "retired"), defaulting to ["live"]; `include_invalidated=` is its two-valued alias.
mem.search(query, *, k=10, min_score=0.0, anchored=False, ranked=False, T=None,
           memory_types=None, states=None, include_invalidated=None,
           include_episodes=False)
                                                  -> SearchResults  # a list, plus .selection
#   anchored=True keeps only the results the query names an entity of — a claim whose
#     subject or object the query names, or one the graph leg reached from such a
#     claim — so a question about an entity the store has never heard of returns []
#     rather than the nearest row about somebody else. Every result reports which as
#     `Explanation.anchor` (subject | object | path | None). Needs no number; combines
#     with min_score. Against a hosted deployment it is sent only when set, so a server
#     from before the field refuses it rather than quietly answering unfiltered.
#   ranked=True runs a configured read_selector over the reranked turns and returns the
#     ones it named first, whole, with Explanation.selected/.span set — see
#     memvara.select. Needs include_episodes=True and no memory_types (ValueError
#     otherwise). SearchResults.selection records what happened: None on a plain read,
#     else an outcome of applied | fallback | unconfigured | disabled | key_rejected.
mem.recall(query, *, k=8, min_score=0.0, anchored=False, ranked=False, header=None,
           include_episodes=False,
           episode_header=None, include_history=False, history_header=None,
           budget=None, counter=<internal>, with_ids=False)
                                                  -> str | RecallResult
#   no `T=`, no `states=`, no `include_invalidated=` — deliberately; see recall() below
#   budget= caps the block by size rather than by count: `k` bounds how many notes,
#     this bounds how much text. Notes drop whole and the block says how many did not
#     fit. `counter=` is any `(str) -> int`; pass `tiktoken`'s or Anthropic's to
#     measure exactly. The default is deliberately not exported — it is a length
#     heuristic that under-counts CJK, so a budget it meets can still overflow the
#     real one, and code that reaches for it by name is usually code that wanted a
#     real tokenizer.
#   ranked=True renders every kept turn whole, first in the turn block, ahead of the
#     unkept turns at their usual char cut. k still bounds the facts; budget still bounds
#     the kept turns (they arrived outside k). A block the model did not actually rank
#     ends with a RECALL_UNRANKED line naming why; with_ids=True puts the same outcome
#     on RecallResult.selection instead of making you parse the line.
#   with_ids=True returns RecallResult(text, claim_ids, dropped, selection) instead of
#     `str`. `text` is byte-identical to what you would have got; `claim_ids` is in
#     render order, 1:1 with the notes, so note n is claim_ids[n - 1]. Live facts only —
#     an episode has no claim id, and a past value is not the source of a
#     present-tense answer.
mem.get(claim_id)                                 -> Claim | None
mem.get_all(*, T=None, states=None, include_invalidated=None)  -> list[Claim]
mem.count(*, T=None, states=None, include_invalidated=None)    -> int
mem.history(subject, predicate, *, T=None)        -> list[Claim]    # timeline of one slot
mem.why(claim_id, *, T=None)                      -> Provenance | None
mem.produced(episode_id, *, T=None)               -> list[Claim]    # why(), backwards
mem.since(when)                                   -> Delta          # what changed since
mem.ask(question, *, at=None, k=3, min_score=0.0, anchored=False)  -> Answer  # narrated
#   Answer(question, at, readings, text). One `Reading` per fact slot, each holding
#   `now`, `then` (what we believe today was true at `at`) and `stated` (what this
#   store would have answered at `at`). The last two differing is the finding, and
#   `text` is the composed narrative — no model, every sentence from a stored column.
#   `Reading.stated` deliberately disagrees with `get_all(as_of=T)`: a row's `valid_to`
#     is written in place by the write that displaces it, so a row read on its own
#     applies an ending that had not been recorded yet. `ask()` has the supersession
#     chain and dates the ending at the successor's `recorded_at`, which is `why()`'s
#     rule already. See `Reading` for the one case it cannot recover.
#   `min_score` defaults to 0.0 exactly as on search() and recall(): this ranks, it
#     does not judge relevance, so on a store that knows nothing about the question it
#     answers confidently from the nearest slot it has. Every Reading names the slot.
#   Delta(since, added, gone): believed now and not then, believed then and not now.
#   A supersession lands in **both** halves, which is the point — an agent coming back
#   to a delta that showed only the arrival would hold the replaced value as well.
#   Both clocks pin to `when`: the belief clock alone leaves `valid_at` at now, so a
#   claim whose world-interval has since closed never enters the "then" set at all.

# traverse — the claims are a graph; walk it
mem.neighborhood(entity, *, depth=2, k=10, min_hops=1, predicates=None,
                 T=None, min_score=0.0)                    -> list[Path]
mem.paths_between(source, target, *, depth=3, k=3, predicates=None,
                  T=None, min_score=0.0)                   -> list[Path]

# identity repair — both dry-run by default, both rewrite history when they are not
from memvara import backfill_entities, split_entity
backfill_entities(mem.writer.reconciler, tenant)           -> RekeyReport
#   Applies aliases learned since a claim was written. A claim keeps the identity it
#   was written with, so learning "Big Blue" is IBM in month six does not re-key
#   month one; this is how you ask for that, dated and attributable.
split_entity(mem.writer.reconciler, scope, surface, at)    -> SplitReport
#   The inverse: one surface form that has been two different things either side of
#   `at`. Two people who share a name are one entity, and on a single-valued predicate
#   the later one's employment retires the earlier one's — a job change nobody wrote.
#   Re-stamps the earlier claims onto a distinct identity and undoes the supersessions
#   that crossed the boundary. Retirements move but are never un-retired: ending a
#   claim is something the write path inferred, retiring one is a caller's statement.

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

`ask()` is the one that uses both clocks. `recall()` renders the current answer;
`ask()` renders the current answer, what we believe today was true at some past instant,
and what this store *would have said* at that instant — and says so when the last two
differ. That difference is not an inconsistency: it means somebody corrected the record
after the moment being asked about, so the answer a person acted on is not the answer
they would get today. It is the sentence the two-clock model exists to produce, and
nothing else in this API composes it.

`recall()` is the one you put in a prompt. It returns a framed block that labels itself as
retrieved data rather than instructions, and flattens each claim to a single line — a
memory whose text contains newlines and a fake section header cannot forge prompt
structure around itself. Its signature is explicit rather than `**kwargs` for the same
reason: the time and state keywords are not reachable from here, and `include_history=True`
is the one bounded exception — see
[What a prompt block may carry from the past](DESIGN.md#what-a-prompt-block-may-carry-from-the-past).

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

### A hosted deployment

`Memvara(api_key=...)` returns a `RemoteMemvara`: the same methods, served by the `/v1`
API of a hosted deployment instead of by a store on this machine. Every response is
hydrated back into the `Claim`, `Episode`, `Result` and `WriteReceipt` dataclasses above,
so a function that takes a `Memvara` and calls `search()` cannot tell which it was handed.

```python
from memvara import Memvara                        # pip install 'memvara[cloud]'

mem = Memvara(api_key="mv_…", user="alice")        # or Memvara.connect()
mem.remember("Alice", "lives_in", "Lisbon")
[r.text for r in mem.search("where do they live?")]
```

`Memvara.connect()` is the same client using whatever credentials are around:
`MEMVARA_API_KEY`, then the file `memvara-mcp login` writes. `base_url=` (or
`MEMVARA_SERVER_URL`) points at a deployment other than the default one.

**A bare `Memvara()` never becomes remote.** The dispatch reads the explicit `api_key=` or
`base_url=` argument and nothing else — never the environment — because a script that has
always written to a local file must not start posting to a hosted store on the day
somebody runs `memvara-mcp login` on that machine. The environment supplies the *value*,
once the caller has asked for remote. Constructing performs no network call: it resolves a
credential, builds a connection pool, and stops. The first request is the first method
call, and `close()` (or a `with` block) releases the pool.

Naming a local subsystem alongside a credential is a `TypeError` rather than a silent
no-op — `path=`, `store=`, `embedder=`, `llm=`, `registry=` and `reembed=True` all
describe an engine that runs server-side. `reembed=False` is accepted and does nothing,
because it asks for nothing.

**What is absent is absent, not raising.** `reembed()`, `pending_extraction()`,
`reextract()` and `reset()` have no endpoint, so they are not methods: reaching for one is
an `AttributeError` at the call site and a mypy error before that, where a method that
raised would compile, ship and fail in production. The same rule decides which *arguments*
exist. `recall()` takes no `with_ids`, because `POST /v1/recall` returns a rendered string
and carries no ids at all; `get_all()` takes no `memory_types`, because the endpoint has no
such filter and would answer with an unfiltered page. `budget` is the one refusal rather
than omission: it stays in `recall()`'s signature so that `None` works, and a value raises
`ValueError`, because a budget silently ignored is an oversized prompt with no signal.

Two divergences are real and worth knowing before you write against them:

- **`consolidate()` returns a job handle, not counts.** The endpoint answers 202 before
  the pass starts, because a real store takes seconds to walk. Poll the job's `status` for
  `succeeded` and its `result` for the per-operation counts. A 202 is not a promise the
  work succeeded.
- **There is no `prove_erased()`.** The local one re-queries the tables a claim's content
  can survive in, which is a physical check this side of the wire cannot perform.
  `erase()` returns whether anything was erased, and `purge()` returns the per-table rows
  removed as the deployment's own count.

**One method exists here and has no local twin: `service()`.** It returns the whole
`GET /v1/stats` envelope — `scope`, `visible`, `tenant_counts`, `extractor`, `read_only` —
where `stats()` returns `tenant_counts` alone so that `stats()["claims"]` is a number
against either engine. Two of those fields have no local answer at all: `extractor` names a
pipeline running on the far side of the wire, and `read_only` is what the presented
*credential* authorizes rather than a setting. Anything deciding what it may offer a user
needs the second one before it offers anything — `memvara-mcp` in cloud mode calls this
once at startup and hides its write tools when it comes back true. `attempts=` and
`timeout=` override the client's own for that one call, because a probe whose value is
that it is cheap should not inherit three attempts at a thirty-second timeout.

`scope()` returns a `ScopedRemoteMemvara`, the twin of `ScopedMemvara`, and the credential
binds the tenant a second time from the other side: the facade resolves it from the bearer
token, and no `/v1` request parameter names one, so a narrowing cannot widen. Errors arrive
as the exception types in `memvara.remote.errors` — `AuthError`, `ScopeError`, `NotFound`,
`Conflict`, `QuotaExhausted`, `RateLimited`, `LegalHold`, `ReadOnly`, `InvalidRequest` and
`ServerError`, all `RemoteError` — and writes carry an `Idempotency-Key` that is held
constant across their own retries.

Three attempts per call. A call is retried on an error the deployment marked retryable, on
a 429 — including one an edge proxy returned with no envelope, which is classified from the
status — and on a connect-phase failure that never reached the server. A `Retry-After` is
waited for as asked, up to thirty seconds; a longer one raises `RateLimited` straight away
with the server's own number on `retry_after`, rather than blocking the call (or the event
loop) for as long as the header says. Waiting an hour is a decision for the caller, who
knows whether an hour is acceptable.

**`memvara.remote.aio.AsyncRemoteMemvara` is the same client, awaited.**

```python
from memvara import AsyncRemoteMemvara               # pip install 'memvara[cloud]'

async def main():
    async with AsyncRemoteMemvara(api_key="mv_…", user="alice") as mem:
        await mem.remember("Alice", "lives_in", "Lisbon")
        return [r.text for r in await mem.search("where do they live?")]
```

Every method on `RemoteMemvara` has an `async def` twin of the same name taking the same
arguments — `scope()` returns an `AsyncScopedRemoteMemvara` rather than binding four
strings, so it stays synchronous, exactly as `AsyncMemvara.scope()` does below. `aclose()`
replaces `close()`; `__aenter__`/`__aexit__` replace the plain context manager.

This one is **not** built on `AsyncMemvara`'s pattern below — it does not run
`RemoteMemvara` inside `asyncio.to_thread`. It talks to `/v1` through `httpx.AsyncClient`
directly, because that transport already has a real async implementation and there is no
local engine underneath this class for coroutine-colouring to propagate through. See
`memvara/aio.py`'s module docstring for the fuller argument and exactly where it stops
applying.

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

`AsyncMemvara` is a thread-pool wrapper around the *local* engine, not an async
rewrite, and says so: SQLite has no async driver worth the name, and the work it hands to
a thread is CPU and disk rather than network. That reasoning is specific to the local
engine — `AsyncRemoteMemvara` above talks to a hosted deployment over a transport that is
already async, and uses it directly instead.

---

Previous: [Quickstart](getting-started/quickstart.md) · Next: [Architecture](reference/architecture.md) · [How it works](DESIGN.md) · [Internals](INTERNALS.md)
