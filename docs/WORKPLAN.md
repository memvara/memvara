# Wave 1 — contracts between parallel workstreams

Four agents are working simultaneously. **File ownership is exclusive.** If you need a
change in a file you do not own, it is specified below — code against it and trust it to
land. Do not edit another workstream's files, and do not "fix" a call to something that
does not exist yet if this document says it will.

Non-negotiable for everyone: `python3 -m pytest -q` must be green and
`python3 -m coverage run -m pytest && python3 -m coverage report` must hold at
**100%** (`fail_under = 100`) before you report. The suite runs offline, no API key.

| workstream | owns |
|---|---|
| **W1 predicate identity** | `memvara/schema.py`, `memvara/write/pipeline.py`, `memvara/llm/base.py`, `memvara/llm/anthropic.py`, `memvara/llm/__init__.py`, `tests/test_schema.py`, `tests/test_pipeline.py`, `tests/test_llm.py`, `tests/test_predicates.py` (new) |
| **W2 store scale** | `memvara/store/sqlite.py`, `memvara/store/base.py`, `memvara/store/__init__.py`, `tests/test_store.py`, `tests/test_vecindex.py` (new) |
| **W3 retrieval quality** | `memvara/retrieve/*`, `tests/test_fusion.py`, `tests/test_scoring.py`, `tests/test_hybrid.py` |
| **W4 defaults & API** | `memvara/core.py`, `memvara/types.py`, `memvara/embed/*`, `memvara/__init__.py`, `tests/test_types.py`, `tests/test_integration.py`, `tests/test_api.py` (new) |

Nobody owns `tests/test_edges.py`, `tests/test_internals.py`, `tests/test_decay.py`,
`tests/test_merge.py`, `tests/test_fast.py`, `tests/test_gate.py`, `README.md`,
`bench/*` — I (the orchestrator) will reconcile those. If a change of yours breaks a test
in a file you do not own, **say so in your report with the test name**; do not edit it.

---

## Pinned interface changes

### A. Store protocol — predicate specs become tenant-scoped (W2 implements, W1 calls)

```python
store.put_spec(spec: PredicateSpec, tenant: str) -> None
store.all_specs(tenant: str) -> list[PredicateSpec]
```

Both currently take no tenant. W2 adds the parameter and a `tenant` column to the
`predicates` table, bumps `SCHEMA_VERSION` to 2, and writes the migration (existing rows
get tenant `'default'`). W1 passes `claim.scope.tenant` / the pipeline's tenant.

Rationale: the table is global today, so one tenant's classification silently sets another
tenant's contradiction behaviour and decay half-life.

### B. `Result` / `Explanation` (W4 owns the dataclasses, W3 populates them)

`Result.score` becomes a **normalized relevance in [0, 1]**. The raw internal value moves
to `Explanation.raw_score`.

```python
@dataclass
class Result:
    claim: Claim
    score: float          # normalized [0,1] — what callers threshold on
    explain: Explanation

@dataclass
class Explanation:
    ...                   # existing fields unchanged
    raw_score: float = 0.0        # NEW: the pre-normalization product
    final_score: float = 0.0      # stays == Result.score (normalized)
```

Why: `Result.score` currently caps at ~0.049 because RRF at `k=60` maxes at `2/61` and the
quality multiplier maxes at ~1.9. Nothing can threshold on it — mem0's default
`threshold=0.1` returns nothing, CrewAI's `min_score` is unusable, and `recall()` has no
relevance floor. W3 owns the normalization function and must document how it is computed.

W3 also adds to `HybridRetriever.search`: `min_score: float = 0.0`, filtering on the
normalized value.

### C. `WriteReceipt` (W4 owns)

```python
llm_calls: int      # MUST NOT increment when no model was actually consulted
unextracted: int    # NEW: turns that reached tier 2 and produced no claim
```

W1 must not increment `llm_calls` for a `NullLLM` (or any backend advertising itself as a
no-op — see D). W1 sets `unextracted`.

### D. `LLM` protocol gains predicate resolution (W1 owns entirely)

```python
class LLM(Protocol):
    name: str
    is_noop: bool = False          # NEW: True for NullLLM; suppresses llm_calls billing
    def extract(...) -> list[dict]: ...
    def resolve_predicate(self, surface: str, candidates: Sequence[str]) -> dict: ...
```

`resolve_predicate` returns
`{"canonical": str | None, "cardinality": ..., "volatility": ..., "memory_type": ...}`.
`canonical` names an existing predicate the surface form is a synonym of, or `None` if it
is genuinely new. `classify_predicate` stays for backward compatibility; W1 decides
whether to keep it as a thin wrapper.

---

## W1 — Predicate identity (the highest-severity item in the project)

A red-team simulation of 10,000 extractions over **six** real-world concepts produced
**41 distinct predicates** and thirteen simultaneously-live claims for "where does the
user work?", including four different employers all pinned at max salience. `recall()`
spent four of eight prompt slots restating a stale employer and never returned the
current one.

Mechanism: `fact_key = hash(owner, subject, predicate)` keys on the predicate *string*, so
`works_at`, `employed_by_company`, `job_employer` and `workplace` are four different slots
and cardinality never applies across them. `registry.learn()` is called with **no aliases**
(`pipeline.py`), so a learned predicate can never fold onto a builtin. Nothing prunes,
merges, caps or ages out the registry.

Build, in order:

1. **A deterministic resolution pre-pass before any model call.** Normalize, then try:
   exact canonical, existing alias, slug match, and a small morphological pass (strip
   `_name`/`_company`/`is_`/`_is`, singular/plural, `employer`↔`employed_by`). This must
   catch the easy majority for free — the model call is the fallback, not the first step.
2. **`resolve_predicate` replaces `classify_predicate` as the acquisition call.** Same
   once-per-novel-surface-form amortization, but spent on *merging* rather than
   classifying. When it returns a `canonical`, register the surface form as an **alias**
   (`PredicateSpec.aliases`, a field that already exists and is never populated) and
   persist it via `store.put_spec(spec, tenant)`.
3. **Bound the registry.** A per-tenant cap on *learned* predicates (default 200). At the
   cap, novel surface forms must fold onto their nearest existing predicate rather than
   registering new ones. Unbounded schema growth is the root cause; the cap is the
   backstop for when resolution is wrong.
4. **Do not bill for a no-op model.** `receipt.llm_calls` must stay 0 with `NullLLM`, and
   `receipt.unextracted` must report turns that reached tier 2 and yielded nothing.

Prove it with a simulation test in `tests/test_predicates.py`: drive ≥2,000 extractions
over ~6 concepts with a fake extractor that varies surface forms, and assert that live
claims per concept stays ≤ a small constant and that the *current* value is what
`get_all()` returns. That test is the deliverable; the code is how you pass it.

Also fix (found in the same simulation): `AnthropicLLM._extract_prompt` sends the entire
known-predicate list on every call, which is an unbounded per-write token tax and
invalidates the cached prompt prefix exactly when the vocabulary is growing fastest. Send
a bounded, stably-ordered subset.

## W2 — Store scale and integrity

All numbers below were measured, not estimated.

1. **Memory-map the vector matrix.** `_VecIndex` holds a dense float32 matrix built
   eagerly at open: 505 MB steady and **1,066 MB peak** at 100k×768, 2.40 s to open, and
   `np.vstack` doubling that transiently holds 4× the old matrix — a measured **950 ms
   stall inside a user's write** at 262k rows, which OOMs at exactly the moment the store
   is largest. A store that outgrows RAM cannot be opened to be repaired.
   Move to an mmap'd, rowid-addressed file. Measured effect: open 2.40 s → 5.6 ms, RSS
   505 MB → 0 (file-backed, and *shared* between processes), query 28 ms → 10 ms. This is
   also the prerequisite for any future ANN backend, which addresses by integer label.
2. **Stop copying the whole candidate set per query.** `self._mat[np.asarray(rows)] @ qq`
   uses advanced indexing, which copies — **307 MB allocated and discarded per query** at
   100k. Two-thirds of query time is not the matmul. Use a mask or a chunked product.
3. **Cross-process coherence.** The index never refreshes, so a claim written by worker B
   is permanently invisible to worker A's vector leg — silently, because BM25 still finds
   it and RRF just ranks it worse. Add a generation counter and a cheap tail re-read.
4. **Tenant-scope the predicates table** (contract A) with `SCHEMA_VERSION = 2` and a
   migration.
5. **`set_embedding` mutates the index outside the transaction**, so a rolled-back batch
   leaves phantom vectors and `stats()['embeddings']` lies. Make index mutation
   transaction-aware or reconcile on rollback.
6. **`purge()` issues two DELETEs per claim in a Python loop** — make it set-based.
7. **`_VecIndex.remove` never compacts**, so a churning store grows its matrix forever.

Do 1–3 first; they are the ceiling-raisers. If you cannot finish all seven, say which and
why rather than rushing them.

## W3 — Retrieval quality

1. **Normalize `Result.score` into [0,1]** and add `min_score` (contract B). Document the
   normalization.
2. **Stopword-aware lexical queries.** `_fts_query` ORs every alphanumeric token, and at
   personal-memory scale stopwords are *rare*, so BM25 gives them enormous IDF. Measured:
   `"what do you know about me?"` becomes six full-weight terms with zero content tokens,
   and a claim containing "do" was returned as the #1 answer for three unrelated queries.
   The vector leg already abstains on a zero-norm query — the lexical leg needs the same
   guard when no content tokens survive. **Note `_fts_query` lives in `store/sqlite.py`,
   which W2 owns.** Implement the stopword logic in `memvara/retrieve/` and have the
   retriever pre-check the query; coordinate through me if you need the store to change.
   Measured payoff: P@1 0.556 → 0.722 with the shipped embedder, 0.778 → 0.833 with a
   semantic one.
3. **Make absence representable.** 6 of 6 unanswerable queries returned confident results;
   `"what is my mother's maiden name?"` scored *identically* to the best answerable query.
   Add an explicit signal that nothing cleared the bar, computed from raw retriever scores
   rather than fused ranks.
4. **Filter starvation:** `search(k=10, memory_types=[PROCEDURAL])` can return 0 while
   matches exist, because filtering happens after fusion truncation. Retry once with a
   larger candidate limit.
5. **Per-slot diversity cap.** A cluster of near-identical claims can take 5 of 8 slots.
   Diversify on `fact_key`, not on embedding similarity — the shipped embedder is lexical
   and MMR over it measurably *reduced* topical coverage.

`rrf_k=60` makes the fusion head flat (rank0/rank1 gap 1.016×) while the quality
multiplier spans 1.66×, so freshness can outrank relevance by ~41 positions. Retune or
justify, and fix the `final_score` docstring arithmetic (it says 1.25×; it is 1.66×).

## W4 — Honest defaults and the missing API surface

1. **The default configuration silently extracts nothing.** A realistic 14-turn
   conversation produces **0 claims** under `NullLLM` + `HashingEmbedder`, with no signal.
   A new user concludes the library is broken in ten minutes. Fix the *honesty*, not by
   pretending: warn once at construction naming what is lost, surface
   `receipt.unextracted`, and make `repr(Memvara)` show the extractor. Consider
   auto-detecting `ANTHROPIC_API_KEY` and say clearly in the warning what to pass.
2. **The `local-embed` upgrade path bricks the store.** `default_embedder()` prefers
   MiniLM (dim 384) over `HashingEmbedder` (dim 512), so week two every read raises
   `ValueError: query dim 384 != index dim 512`. Persist the embedder identity and
   dimension with the store, detect the mismatch **at construction** with an actionable
   message, and ship the `reembed()` the existing error message already tells people to
   run — it does not exist.
3. **Missing API surface**, needed by mem0 compat, LangGraph, LlamaIndex and CrewAI alike:
   `get(claim_id)`, `delete(claim_id)`, `count()`, `reset()`. Scope-check every one of
   them the way `why()` now is.
4. **A scope view object.** `tenant/user/agent/session` is repeated, unannotated, on six
   methods. `mem.scope(user="alice", session="s1")` returning a bound facade removes the
   biggest source of call-site noise and is the shape a server layer needs.
5. **`__repr__` for `Claim`, `Result`, `Episode`, `Explanation`, `Scope`, `Provenance`.**
   `WriteReceipt` has an excellent one; the others dump 1,400-character dataclass reprs,
   which makes the README's own `history()` example illegible.
6. Constructor errors: accept `user_id`/`agent_id`/`run_id` as deprecated aliases with a
   `DeprecationWarning`, use `difflib.get_close_matches` on unknown kwargs, and raise
   rather than silently ignoring `path` when `store=` is also given.

---

# Wave 2 — contracts

Wave 1 is merged: 1,124 tests, 100% coverage, `fail_under = 100`. Same rules — exclusive
file ownership, whole suite green and coverage at 100% before reporting, and if you break
a test in a file you do not own, **report the name rather than editing it**.

| workstream | owns |
|---|---|
| **A — memory dynamics** (W5+W8) | `memvara/consolidate/*`, `memvara/write/reconcile.py`, `memvara/retrieve/scoring.py`, `memvara/types.py`, `tests/test_decay.py`, `tests/test_merge.py`, `tests/test_reconcile.py`, `tests/test_scoring.py` |
| **B — retrievable episodes** (W7) | `memvara/store/*`, `memvara/retrieve/hybrid.py`, `memvara/retrieve/__init__.py`, `memvara/core.py`, `tests/test_store.py`, `tests/test_hybrid.py`, `tests/test_api.py` |
| **C — mem0 compatibility** (W11) | `memvara/compat/` (new), `tests/test_compat.py` (new). **Public API only** — do not edit any existing module. |
| **D — MCP server** (W10) | `memvara/server/` (new), `tests/test_server.py` (new). **Public API only** — do not edit any existing module. |

`memvara/__init__.py`, `README.md`, `docs/*`, `bench/*`, `tests/test_edges.py`,
`tests/test_internals.py`, `tests/test_integration.py`, `tests/test_predicates.py` are
mine; I will wire exports and reconcile.

## Cross-workstream note (A and B)

A needs a per-claim "when was this last observed". **Prefer `Claim.meta`** — it is already
persisted as JSON and needs no schema change, so A can land without waiting on B. If you
conclude a real column is required, say so in your report and I will schedule it; do not
edit `memvara/store/*` (B owns it this wave).

---

# Wave 3 — contracts

Wave 2 merged: 1,414 tests, 100% coverage. Same rules — exclusive ownership, whole suite
green and coverage at 100% before reporting, and if you break a test in a file you do not
own, **report the name rather than editing it**.

| workstream | owns |
|---|---|
| **E — entity resolution** | `memvara/entities.py` (new), `memvara/types.py`, `memvara/write/reconcile.py`, `tests/test_entities.py` (new), `tests/test_reconcile.py`, `tests/test_types.py` |
| **F — store + API completeness + async** | `memvara/store/*`, `memvara/core.py`, `memvara/aio.py` (new), `tests/test_store.py`, `tests/test_api.py`, `tests/test_aio.py` (new) |
| **G — observability + lock hoisting** | `memvara/telemetry.py` (new), `memvara/write/pipeline.py`, `memvara/consolidate/*`, `memvara/retrieve/hybrid.py`, `tests/test_telemetry.py` (new), `tests/test_pipeline.py`, `tests/test_hybrid.py`, `tests/test_decay.py`, `tests/test_merge.py` |

Mine: `memvara/__init__.py`, `memvara/compat/*`, `memvara/server/*`, `README.md`, `docs/*`,
`bench/*`, `tests/test_edges.py`, `tests/test_internals.py`, `tests/test_integration.py`,
`tests/test_compat.py`, `tests/test_server.py`, `tests/test_vecindex.py`.

## Pinned interfaces

### E-1. Entity persistence (F implements, E calls)

Mirrors the predicate-spec pattern that already works:

```python
store.put_entity(entity_id: str, canonical: str, aliases: Sequence[str], tenant: str) -> None
store.all_entities(tenant: str) -> list[tuple[str, str, tuple[str, ...]]]
```

Tenant-scoped for the same reason predicates are: one tenant deciding that "Acme" and
"Acme Corp" are one entity must not decide it for another. Bump `SCHEMA_VERSION` and
migrate.

### E-2. `fact_key` / `value_key` may key on entity ids (E owns `types.py`)

No signature changes — `Claim.fact_key` stays a property. F's indexes are on the value,
so nothing in the store changes shape. **`fact_key_for()` remains the only supported way
to derive a key for a predicate other than a claim's own.**

### F-1. Per-claim erasure (C's highest-value gap)

```python
store.erase_claim(claim_id: str) -> bool      # irreversible; claim + FTS + vector
Memvara.erase(claim_id, *, tenant=, user=, ...) -> bool   # scope-checked like why()
```

Distinct from `delete()`, which retires. The mem0 shim currently warns that it cannot
honour `delete(memory_id)` because retirement leaves the text readable — that warning
should become unnecessary.

### F-2. Provenance-preserving writes (C had to bypass the facade for these)

```python
Memvara.remember(..., sources=Sequence[str] | None, text: str | None)
Memvara.supersede(old_claim_id, new_claim) -> WriteReceipt   # sets invalidated_by
```

C currently reaches `store.add_episode()` + `writer.assert_claim()` + `store.invalidate()`
directly — public objects, but below the facade, so a refactor breaks compat silently.

### G-1. Telemetry (G owns the module, F wires it)

```python
Memvara(..., telemetry=Recorder | None)
```

A `Recorder` protocol with a no-op default. **Must impose no measurable cost when unset** —
this library's whole argument is about cost, so an always-on hook would be self-defeating.
G defines the protocol and the emission points; F adds the one constructor parameter.
