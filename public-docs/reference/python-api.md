# Python API reference

This page lists the full public surface of the `memvara` package: what you can import,
and what every method on the main `Memvara` class does. It assumes you've already read
the [tutorial](../tutorials/getting-started.md) or a how-to guide — this is a lookup
page, not an introduction.

Every method below accepts `tenant=`, `user=`, `agent=`, and `session=` to override the
scope you set when the store was created; they're omitted below for brevity.

## Creating a store

```python
from memvara import Memvara

mem = Memvara(
    path=None,            # a SQLite filename, or None for in-memory (not persisted)
    *,
    store=None,            # a custom Store implementation, instead of path=
    embedder=None,         # a custom Embedder; defaults to HashingEmbedder (keyword-based)
    llm=None,              # an LLM for extracting facts from free text; None = offline
    registry=None,         # a PredicateRegistry; defaults to the built-in vocabulary
    telemetry=None,        # a Recorder for observability
    redactor=None,         # a Redactor for stripping sensitive content before storage
    tenant=None, user=None, agent=None, session=None,   # default scope
    reembed=False,         # re-embed all existing claims if the embedder changed
    **tuning,
)
```

Pass `api_key=` or `base_url=` instead of `path=` to connect to a hosted deployment. This
returns a `RemoteMemvara` with the same methods, described in
[Deployment reference](cli-and-configuration.md#connecting-to-a-hosted-deployment).

## Writing

| Method | Returns | Calls a model? |
|---|---|---|
| `add(messages, *, role="user", ts=None)` | `WriteReceipt` | Only if you configured `llm=` and the text doesn't match a recognized simple sentence pattern |
| `remember(subject, predicate, object, *, valid_from=, valid_to=, recorded_at=, sources=, confidence=, memory_type=, **meta)` | `WriteReceipt` | Never |
| `supersede(old_claim_id, new_claim, *, at=, sources=)` | `WriteReceipt` | Never |

`role="user"` on `add()` is scanned for facts; `role="system"` (for documents, logs, or
pasted transcripts) is stored and citable, but never scanned. See
[Record and correct a fact](../how-to-guides/record-and-correct-a-fact.md).

## Correcting (reversible — nothing is deleted)

| Method | Returns | What it means |
|---|---|---|
| `forget(subject, predicate, *, at=None, close="retired")` | `list[Claim]` | Close every current value in this slot. `close="retired"` (default) means the record was wrong; `close="ended"` means the world changed. |
| `delete(claim_id, *, at=None, close="retired")` | `bool` | The same choice, for a single claim by ID. |

## Deleting (irreversible — text is actually removed)

| Method | Returns | What it removes |
|---|---|---|
| `erase(claim_id, *, sources=False)` | `bool` | One claim's text, search index entry, and embedding. `sources=True` also removes its source message, if nothing else cites it. |
| `purge()` | `dict[str, int]` | Everything in the current scope. |
| `reset()` | `dict[str, int]` | Everything in the current scope, plus the schema itself. |
| `prove_erased(claim_id)` | `ErasureProof` | Re-checks every table for leftover content, and returns proof it's gone. |

See [Delete personal data](../how-to-guides/delete-personal-data.md) for why these are
kept separate from the reversible corrections above.

## Reading

Reads marked with a clock icon below accept the three time-travel keywords —
`valid_at=`, `known_at=`, and `as_of=` — described in
[Ask about the past](../how-to-guides/ask-about-the-past.md).

| Method | Returns | Time travel |
|---|---|---|
| `search(query, *, k=10, min_score=0.0, states=("live",))` | `list[Result]` | ⏱ |
| `recall(query, *, k=8, min_score=0.0, budget=None, include_history=False)` | `str` | — (deliberately not supported; see below) |
| `get(claim_id)` | `Claim \| None` | — |
| `get_all(*, states=None)` | `list[Claim]` | ⏱ |
| `count(*, states=None)` | `int` | ⏱ |
| `history(subject, predicate)` | `list[Claim]` | ⏱ (via `as_of=`) |
| `since(when)` | `Delta` | — |
| `ask(question, *, at=None, k=3)` | `Answer` | via `at=` |
| `why(claim_id)` | `Provenance` | ⏱ |
| `produced(episode_id)` | `list[Claim]` | ⏱ |
| `neighborhood(entity, *, depth=2, k=10)` | `list[Path]` | ⏱ |
| `paths_between(source, target, *, depth=3, k=3)` | `list[Path]` | ⏱ |

`recall()` deliberately has no time-travel parameters. It produces text meant to be
pasted straight into a prompt, and a call that silently returned outdated information
would look identical, in the output, to a normal one — so the choice is made explicit
instead: pass `include_history=True` to include past values, clearly labeled under their
own heading, alongside the current ones.

## Every claim's fields

A `Claim` is what every read above returns. The fields worth knowing:

| Field | Meaning |
|---|---|
| `id` | Unique claim ID |
| `subject`, `predicate`, `object` | The fact itself |
| `text` | A human-readable rendering of the fact |
| `valid_from`, `valid_to` | The world-time interval this fact held (world clock) |
| `recorded_at`, `invalidated_at` | When this store came to believe / stop believing it (belief clock) |
| `state` | `"live"`, `"ended"` (world changed), or `"retired"` (record was wrong) |
| `confidence` | How sure the writer was, 0 to 1 |
| `salience` | How important this fact is, feeds into search ranking |

`live`, `ended`, and `retired` are not mutually exclusive categories that add up to the
total — a claim can be both ended and retired at once (the world moved on, *and* it later
turned out the record was wrong even before that). `claims` (the total count) is the only
number guaranteed to add up.

## Working with a fixed scope

```python
scoped = mem.scope(tenant="acme", user="alice")
scoped.remember("alice", "lives_in", "Berlin")
```

`ScopedMemvara`, returned by `mem.scope(...)`, exposes the same methods as `Memvara` but
with the scope already fixed, so you don't have to repeat `tenant=`/`user=`/etc. on every
call.

## Diagnostics

| Method | Returns |
|---|---|
| `stats(*, tenant=None)` | Claim and message counts for a scope |
| `connectivity(*, tenant=None)` | How connected the fact graph is — useful for judging whether multi-hop search (`neighborhood`, `paths_between`) will find anything |
| `extractor` (property) | The name of the currently active fact extractor, e.g. `"api"` for an offline store |
| `reembed(embedder=None, batch_size=256)` | Re-computes every stored embedding, e.g. after switching to a real semantic embedder |
| `consolidate(*, tenant=None)` | Runs housekeeping — merging near-duplicate facts, decaying stale ones |

## Everything you can import from `memvara`

```python
from memvara import (
    Memvara, ScopedMemvara, AsyncMemvara, AsyncScopedMemvara,

    # the data model
    Claim, Episode, Scope, Result, Explanation, Provenance,
    WriteReceipt, MemoryType, Derivation, Answer, Reading, Delta, RecallResult,
    Closure,   # "ended" vs "retired" — which clock a correction closes

    # schema (declaring your own fact types)
    PredicateRegistry, PredicateSpec, Cardinality, Volatility,

    # pluggable backends
    Store, SQLiteStore,
    Embedder, HashingEmbedder, CachedEmbedder, default_embedder, EmbedderFingerprint,
    LLM, NullLLM, AnthropicLLM, OpenAILLM,

    # erasure
    ErasureIncomplete, ErasureProof,

    # retrieval internals, if you need to reason about a score directly
    HybridRetriever, Retrieved, EpisodeResult, GraphTraverser, Path, Edge,

    # entity handling
    EntityRegistry, EntityResolution, EntitySpec, entity_key,
    backfill_entities, split_entity, SplitReport,

    # observability
    Recorder, NullRecorder, MemoryRecorder,

    # redaction and consolidation
    Redactor, PatternRedactor, Consolidator,

    # a hosted deployment
    RemoteMemvara, AsyncRemoteMemvara,
    RemoteError, AuthError, ScopeError, NotFound, Conflict, QuotaExhausted,
    RateLimited, LegalHold, ReadOnly, InvalidRequest, ServerError,
)
```

## Async

`AsyncMemvara` and `AsyncScopedMemvara` mirror the same methods as their synchronous
counterparts (`async def add(...)`, `async def search(...)`, and so on) for use inside
an `asyncio` application. They run the same underlying store on a background thread pool
rather than being a separate async-native implementation — the point is to keep your
event loop from blocking, not to change what Memvara does.

## Related pages

- [Predicate schema reference](predicate-schema.md) — the full detail on declaring your
  own fact types.
- [MCP tools reference](mcp-tools.md) — the same capabilities, exposed as tools for an AI
  assistant rather than as a Python API.
