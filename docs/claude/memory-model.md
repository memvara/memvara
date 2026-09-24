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
  `closure_reason()`, `closure_reasons()`, `Link`, `ForgetPreview`, `ForgetResult`,
  `fact_key_for()`, `owner_key()`.
- The confirmation token behind `forget_matching`: `memvara/confirm.py` — `Confirmer`,
  `ConfirmationRefused`, `CONFIRM_TTL`.
- Primary: `memvara/core.py` — `Memvara`, the whole public API, plus `ScopedMemvara` for a
  view narrowed to one tenant, user, agent or session.
- Storage: `memvara/store/base.py` — the `Store` protocol, `resolve_states()`,
  `state_predicate()`, `stored_state_predicate()`, `live_predicate()`.
- Storage backends: `memvara/store/sqlite.py` — `SQLiteStore`, the default;
  `memvara/store/remote.py` — `RemoteStore`, the same protocol against a hosted deployment.
- Encryption at rest for `SQLiteStore`: `memvara/store/encryption.py` — `resolve_key()`,
  `VectorSealer` (the encrypted vector file), `encrypt_store()` (behind
  `memvara encrypt`), `EncryptionError`, `EncryptionWarning`. Tests:
  `tests/test_encryption.py`. INTERNALS.md section *Encryption at rest*.
- Entities and predicates: `memvara/entities.py` — `EntityRegistry`, `entity_key()` and
  `typed_entity_key()`, which keeps a `type:` namespace so `company:apple` and `fruit:apple`
  are two entities;
  `memvara/schema.py` — `PredicateRegistry`, `PredicateSpec`, `Cardinality`, `Volatility`.
- Tests: `tests/test_bitemporal.py`, `tests/test_types.py`, `tests/test_store.py`,
  `tests/test_erasure.py`, `tests/test_erasure_residue.py`.
- Documentation: [INTERNALS.md](../INTERNALS.md), sections *`memvara/store/`*, *The two time
  axes*, *The three states* and *Erasure removes the bytes, not just the rows*. The reader's
  version is [bitemporal memory](../concepts/bitemporal-memory.md).

## How the pieces fit

`Memvara` in `memvara/core.py` is the only class most callers touch. It owns a `Store`, an
`Embedder`, a `PredicateRegistry` and an `LLM`, and it exposes the writes (`add()`,
`remember()`, `forget()`, `forget_matching()`, `supersede()`, `delete()`, `link()`,
`erase()`), the reads (`search()`,
`recall()`, `ask()`, `get()`, `get_all()`, `since()`, `history()`, `why()`, `produced()`,
`neighborhood()`, `paths_between()`), the document calls (`add_document()`,
`get_document()`, `list_documents()`, `update_document()`, `delete_document()`,
`delete_documents()`, `document_status()`) and the maintenance calls (`consolidate()`,
`stats()`, `connectivity()`, `reembed()`, `merge_predicate()`).

A write becomes an `Episode` (the raw turn) and zero or more `Claim` rows that cite it. A
read returns `Result` objects that wrap a claim with its score and an `Explanation`, so the
reason a claim was returned is inspectable rather than implied. Every write returns a
`WriteReceipt`, which counts what happened: claims written, claims reinforced, values ended,
values retired, model calls made, and whether extraction was deferred.

Scope is a five-part key — tenant, user, project, agent, session — held by `Scope` and
flattened by `owner_key()`, which folds only tenant and user because it also scopes entity
identity. The project is mixed into `fact_key_for()` directly instead, so that two repositories
keep separate slots while `software:postgresql` stays one entity across both. It is bound where the store is opened, not passed per call by a model, which
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
  and `prove_erased()` returns an `ErasureProof` with per-table counts as evidence. The
  engine erases on its own in one case only: a claim written with `expires_at`, once that
  instant has passed. From that instant no read returns the claim, and `erase_expired()`
  deletes it through the same path as `erase()`, when the store opens and hourly in the
  MCP server. `expires_at` is not `valid_to`: a claim
  whose `valid_to` has passed is ended and kept.

A document sits beside this rather than inside it. `add_document()` stores a document's
text as chunks, each an episode, and `delete_document()` erases that text. It never
erases a claim: a claim whose every source was one of the document's chunks is retired,
with the closure reason "source document deleted", and a claim with another source keeps
it. `docs/INTERNALS.md` has the details under *Documents*.

The MCP tool descriptions in `memvara/server/tools.py` state the same three words for a
model that cannot read this page; `.claude/rules/tool-descriptions.md` covers that side and
the write receipt that once got it wrong. If the definitions above change, change them there
in the same commit.

`CLOSURES` in `memvara/types.py` is the pair `("ended", "retired")`, and `close_out()` is the
one function that applies either. Choosing the wrong one records a false reason for the
change, and nothing downstream can detect it afterwards.

A closure can also carry the caller's own reason, at most 500 characters, which
`close_out()` writes onto the closure witness in `meta["closure"]` beside the clock that
stopped. `history()` and `why()` return it; nothing on the read path filters on it.
`forget_matching()` applies either closure to every claim a query matched, in two calls
bound by a signed token, and never erases. Links between claims (`extends`, `derives`) are
records about two records rather than a third state: they have no closure, and erasing
either claim removes them. `docs/INTERNALS.md` has all three under *`memvara/store/`*.

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
- **The engine deletes a row only when a claim's explicit expiry has passed, and no write
  closes both clocks.** Superseding a value sets `valid_to` and leaves `invalidated_at`
  unset, because the old value stopped being true and we were never mistaken about it.
  Ending and superseding never delete. The one deletion the engine makes by itself is
  `erase_expired()`, which erases only claims carrying an `expires_at` the caller set, only
  after it passes, and always with a proof. `erase()`, `purge()` and `reset()` delete on
  purpose and by name, as the caller's decision. This is invariant 3 in
  [INTERNALS.md](../INTERNALS.md).
- **A filter and a limit may not live in different layers.** Whatever narrows rows has to
  run where the truncation runs, or the top of the list is wrong with nothing saying it was
  partial. This is invariant 7, and it is why `states=` is a store parameter rather than a
  comprehension in `memvara/core.py`.
- **A sort that sits above a limit ends in a content key, never an id.** A claim id is a
  `uuid4` minted at ingest, so a tie broken on it would make the result a property of which
  ingest ran. `memvara/store/sqlite.py` orders on `value_key` before `id`.
- **Scope is bound at startup and cannot be widened by a call.** `ScopedMemvara.bind()`
  narrows only.
- **Each text index row sits at the rowid of the row it indexes.** Erasure deletes a row's
  index entry by rowid, and both lexical legs join the index to its table on rowid, so a
  write that gave a row a new rowid would break both. That is why `put_claim` and
  `add_episode` upsert instead of `INSERT OR REPLACE`. `tests/test_store.py` checks the
  invariant after every write that moves or frees a rowid, and after a `VACUUM`.
- **Every commit `SQLiteStore` makes goes through `_maybe_commit`.** That is what empties
  `_scope_turns`, the vector leg's in-memory list of each scope's turns and their matrix
  rows. A commit that went around it would leave searches missing a new turn, or returning
  an erased one scored by the vector that took its row, until the next commit.
  `tests/test_store.py` checks a turn written, erased, rolled back and written by another
  process between two searches.

## Read next

[INTERNALS.md](../INTERNALS.md) has the module contracts and the numbered invariants with
the code that makes each one true. The docstrings of `Claim` and `WriteReceipt` in
`memvara/types.py` explain the field-level decisions, and `memvara/store/base.py` explains
the state vocabulary at the point where it is defined.

Next: [the write pipeline](write-pipeline.md).
