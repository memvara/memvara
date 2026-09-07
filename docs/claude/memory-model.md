# The memory model

Memvara stores facts as **claims**: a subject, a predicate and an object, each carried by a
`Claim` in `memvara/types.py`. Every claim sits on two independent time axes, and the whole
product follows from keeping them apart. Valid time (`valid_from`, `valid_to`) is when the
fact was true in the world. Transaction time (`recorded_at`, `invalidated_at`) is when this
store believed it. A memory layer with one `updated_at` column can answer "where does she
live now?" but not "on 1 March, where did we think she lived?", and it cannot let a
late-arriving fact correct the past without rewriting history.

Claims that answer the same question about the same subject share a **slot**, keyed by
`fact_key_for()` in `memvara/types.py`. The slot is where contradiction handling happens: a
new value about the same subject and predicate meets the values already there rather than
landing beside them unnoticed.

## Where the code is

- Primary: `memvara/types.py` — `Claim`, `Scope`, `Episode`, `WriteReceipt`, `Result`,
  `Explanation`, `ErasureProof`, `MemoryType`, `Derivation`, `CLOSURES`, `close_out()`,
  `fact_key_for()`, `owner_key()`.
- Primary: `memvara/core.py` — `Memvara`, the whole public API, plus `ScopedMemvara` for a
  view narrowed to one tenant, user, agent or session.
- Storage: `memvara/store/base.py` — the `Store` protocol, `resolve_states()`,
  `state_predicate()`, `stored_state_predicate()`, `live_predicate()`.
- Storage backends: `memvara/store/sqlite.py` — `SQLiteStore`, the default;
  `memvara/store/remote.py` — `RemoteStore`, the same protocol against a hosted deployment.
- Entities and predicates: `memvara/entities.py` — `EntityRegistry`, `entity_key()`;
  `memvara/schema.py` — `PredicateRegistry`, `PredicateSpec`, `Cardinality`, `Volatility`.
- Tests: `tests/test_bitemporal.py`, `tests/test_types.py`, `tests/test_store.py`,
  `tests/test_erasure.py`, `tests/test_erasure_residue.py`.
- Documentation: [INTERNALS.md](../INTERNALS.md), sections *`memvara/store/`*, *The two time
  axes*, *The three states* and *Erasure removes the bytes, not just the rows*. The reader's
  version is [bitemporal memory](../concepts/bitemporal-memory.md).

## How the pieces fit

`Memvara` in `memvara/core.py` is the only class most callers touch. It owns a `Store`, an
`Embedder`, a `PredicateRegistry` and an `LLM`, and it exposes the writes (`add()`,
`remember()`, `forget()`, `supersede()`, `delete()`, `erase()`), the reads (`search()`,
`recall()`, `ask()`, `get()`, `get_all()`, `since()`, `history()`, `why()`, `produced()`,
`neighborhood()`, `paths_between()`) and the maintenance calls (`consolidate()`, `stats()`,
`connectivity()`, `reembed()`, `merge_predicate()`).

A write becomes an `Episode` (the raw turn) and zero or more `Claim` rows that cite it. A
read returns `Result` objects that wrap a claim with its score and an `Explanation`, so the
reason a claim was returned is inspectable rather than implied. Every write returns a
`WriteReceipt`, which counts what happened: claims written, claims reinforced, values ended,
values retired, model calls made, and whether extraction was deferred.

Scope is a four-part key — tenant, user, agent, session — held by `Scope` and flattened by
`owner_key()`. It is bound where the store is opened, not passed per call by a model, which
is what stops a tool call reaching another user's memory.

## Three states, and three different endings

A claim is `live` when neither clock is closed, `ended` when valid time is closed because the
world moved on, and `retired` when transaction time is closed because the record was wrong.
`Claim.state` reports the stored answer; `state_predicate()` in `memvara/store/base.py`
compiles the as-of answer for a given instant.

The distinction is the product, and it appears in three places that must agree.

- **Ended** means the fact was true and has stopped being true. `delete()` and the closure
  value `"ended"` write it, at the instant it stopped. The claim keeps answering questions
  about the period it held.
- **Retired** means the fact was never true and the record was a mistake. `forget()` and the
  closure value `"retired"` write it. The claim stops answering present-tense questions but
  stays visible to `history()` and `why()`.
- **Erased** is the only one that removes bytes. `erase()` deletes the row and its residue,
  and `prove_erased()` returns an `ErasureProof` with per-table counts as evidence.

The MCP tool descriptions in `memvara/server/tools.py` state the same three words for a
model that cannot read this page; `.claude/rules/tool-descriptions.md` covers that side and
the write receipt that once got it wrong. If the definitions above change, change them there
in the same commit.

`CLOSURES` in `memvara/types.py` is the pair `("ended", "retired")`, and `close_out()` is the
one function that applies either. Choosing the wrong one records a false reason for the
change, and nothing downstream can detect it afterwards.

## Invariants and assumptions

- **Deterministic paths never call a model.** Deduplication, contradiction resolution,
  ranking, decay and time travel are pure functions of stored state. `NullLLM` in
  `memvara/llm/base.py` is the default, and `Reconciler` and `Consolidator` take no `llm`
  parameter at all. This is invariant 1 in [INTERNALS.md](../INTERNALS.md).
- **An unknown predicate defaults to `Cardinality.MANY`.** Wrongly retiring a true fact is
  worse than keeping two competing ones, so nothing supersedes until a `PredicateSpec` says
  the slot holds one value. `MEMVARA_PREDICATES` is how a deployment revises that.
- **`resolve_states()` is the single place either spelling of a state filter is
  interpreted.** Passing both `states=` and `include_invalidated=` raises, because there is
  no reading of the mix in which one of them is not being ignored.
- **The three states do not tile the store.** Asking for all three collapses to the belief
  floor alone, which readmits a claim recorded but not yet in force. That is deliberate and
  `tests/test_bitemporal.py` pins it.
- **No engine write deletes a row, and no write closes both clocks.** Superseding a value
  sets `valid_to` and leaves `invalidated_at` unset, because the old value stopped being
  true and we were never mistaken about it. `erase()`, `purge()` and `reset()` delete, on
  purpose and by name, and they are the caller's decision rather than the engine's. This is
  invariant 3 in [INTERNALS.md](../INTERNALS.md).
- **A filter and a limit may not live in different layers.** Whatever narrows rows has to
  run where the truncation runs, or the top of the list is wrong with nothing saying it was
  partial. This is invariant 7, and it is why `states=` is a store parameter rather than a
  comprehension in `memvara/core.py`.
- **A sort that sits above a limit ends in a content key, never an id.** A claim id is a
  `uuid4` minted at ingest, so a tie broken on it would make the result a property of which
  ingest ran. `memvara/store/sqlite.py` orders on `value_key` before `id`.
- **Scope is bound at startup and cannot be widened by a call.** `ScopedMemvara.bind()`
  narrows only.

## Read next

[INTERNALS.md](../INTERNALS.md) has the module contracts and the numbered invariants with
the code that makes each one true. The docstrings of `Claim` and `WriteReceipt` in
`memvara/types.py` explain the field-level decisions, and `memvara/store/base.py` explains
the state vocabulary at the point where it is defined.

Next: [the write pipeline](write-pipeline.md).
