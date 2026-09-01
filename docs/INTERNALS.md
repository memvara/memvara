# Memvara internals — module contracts

This file is the interface contract between subsystems. `core.py` wires them together
against exactly these signatures, so treat them as fixed. Everything here is already
importable from the foundation modules:

- `memvara/types.py` — `Claim`, `Episode`, `Scope`, `Result`, `Explanation`, `WriteReceipt`,
  `MemoryType`, `Derivation`, `utcnow()`, `content_hash()`
- `memvara/compat/supermemory_import.py` — `import_supermemory`, `SupermemoryReceipt`
- `memvara/schema.py` — `PredicateRegistry`, `PredicateSpec`, `Cardinality`, `Volatility`
- `memvara/store/` — `Store` and `SQLStore` protocols, `SQLiteStore`, `STATES`,
  `ClaimState`, `resolve_states()`, `state_predicate()`, `stored_state_predicate()`,
  `live_predicate()`
- `memvara/embed/` — `Embedder` protocol, `HashingEmbedder`, `CachedEmbedder`, `default_embedder()`
- `memvara/llm/base.py` — `LLM` protocol, `NullLLM`, `CLAIM_SCHEMA`, `PREDICATE_SCHEMA`,
  `EXTRACT_SYSTEM`, `PREDICATE_SYSTEM`

## Design invariants (do not violate)

Each one is stated as **Claim / Scope / Sketch / Measured**, borrowed from the Verified
Design Invariant format in SuperLocalMemory V4. The format earns its place through the
last two lines rather than the first: *Sketch* names the code that makes the claim true,
so a reader can check it, and *Measured* is either a number this repository produced or an
explicit statement that no measurement exists and only a test stands behind it. **Where
nothing was measured the line says so.** An invariant with an invented number beside it is
worse than one with none, and the temptation to supply one is exactly what the format is
for.

*Scope* is the line that is easiest to leave off and does the most work. Every claim here
holds somewhere and not everywhere, and the eighth invariant exists because one of them
was being read as holding further than it does.

1. **Deterministic paths never call an LLM.**

   > **Claim.** Deduplication, contradiction resolution, ranking, decay and time travel
   > are pure functions of stored state.
   > **Scope.** The library. Only `extract()` and `resolve_predicate()` may touch a model,
   > and both are on the write path. Nothing on the read path calls one at all — a
   > reranker is a cross-encoder rather than a generative model, and it is off by default.
   > **Sketch.** `NullLLM` is the default `llm=`, so the shipped configuration has no
   > model to call; `HybridRetriever`, `Reconciler` and `Consolidator` take no `llm`
   > parameter at all.
   > **Measured.** `bench/mem0_real.py`: 2 write-path LLM calls against mem0's 105 on the
   > same 105-turn transcript, and **identical final state on every run** where mem0's
   > differs. `tests/test_packaging.py::test_nothing_but_numpy_is_imported_while_the_
   > package_is_being_imported` holds the import side.

2. **Unknown predicates default to `Cardinality.MANY`.** Wrongly retiring a true fact is
   worse than keeping two competing ones. The default is deliberate and stays; what
   `MEMVARA_PREDICATES` adds is a way to *revise* it, since before it a server-backed
   store had none. A declared spec outranks a persisted learned one — rehydration skips
   any learned spec whose name a declaration already holds, so a pack corrects a store
   that guessed rather than only describing a fresh one. Forward-only: it changes what
   supersedes on the next write and retires nothing already stored.

   A vocabulary is TOML, one `[[predicate]]` table each, `name`, `cardinality` and
   `volatility` required, `memory_type`, `aliases` and `supersedes` optional:

   ```toml
   [[predicate]]
   name = "git_state"
   cardinality = "one"     # "one" supersedes, "many" accumulates
   volatility = "fast"     # static | slow | fast -> 36500 | 730 | 7 day half-life
   aliases = ["git_status"]
   ```

   Needs Python 3.11 or later, which is where `tomllib` arrives; the reader is
   imported lazily so 3.10 keeps working for everything else.

   Malformed entries raise rather than being skipped: a vocabulary that half-loads leaves
   some predicates superseding and others accumulating with nothing recording which.

   > **Claim.** A predicate nobody declared accumulates rather than superseding.
   > **Scope.** Detection only. It makes a missed contradiction the failure mode instead
   > of a wrongly retired fact; it does not make either one visible, because accumulating
   > is what `MANY` is *for* and nothing can tell an intended `MANY` from a forgotten
   > declaration.
   > **Sketch.** `PredicateRegistry.spec` returns a `MANY` default for an unknown name;
   > `Reconciler` only supersedes on `ONE`.
   > **Measured.** Not measured, and the cost of the default is instead recorded from the
   > other side: `tests/test_demo.py::test_a_predicate_left_at_the_default_cardinality_
   > stops_superseding_silently` removes one declaration from a working configuration and
   > watches the slot come back with two answers.

3. **Nothing is ever hard-deleted by the engine, and end-of-life moves exactly one
   clock.** Closing valid time (`valid_to`) says *the world changed*; closing transaction
   time (`invalidated_at`) says *the record was wrong*. They are different events and no
   write may assert both. Superseding a claim therefore sets `valid_to` and
   `invalidated_by` and leaves `invalidated_at` unset — the old value stopped being true,
   and we were never mistaken about it. `Claim.state` names the outcome: `live`, `ended`,
   `retired`. History must stay queryable via `known_at` **and** `valid_at`; the second
   of those returned nothing on any history the engine wrote itself for as long as
   supersession closed both clocks. The correcting reading is reachable, never guessed:
   `close="retired"` on `remember`, `supersede`, `forget` and `delete`, defaulting to
   `"ended"` everywhere except `forget`/`delete`, which are belief operations by name.

   > **Claim.** No engine write deletes a row, and no write closes both clocks.
   > **Scope.** The *engine*. `erase()`, `purge()` and `reset()` delete, on purpose and by
   > name, and they are the caller's decision rather than the engine's — see invariant 8's
   > neighbour below and `Memvara.prove_erased`.
   > **Sketch.** `close_out` is the single place any claim ends and takes one `Closure`;
   > `Claim.state` derives `live`/`ended`/`retired` from which column is set.
   > **Measured.** `bench/compare.py`: **0 stale values left live** against 7 for a
   > mem0-style baseline, on a transcript where 10 facts are revised. `tests/
   > test_bitemporal.py` holds the two-clock reads.

4. **Every claim carries provenance.**

   > **Claim.** `sources` holds the episode ids the claim came from, and `derivation`
   > reflects how it was produced.
   > **Scope.** Claims the engine writes. A `Claim` a caller constructs by hand and hands
   > to `remember()` carries what the caller put in it.
   > **Sketch.** `FastExtractor._claim` and the LLM tier both stamp `sources=[ep.id]` and
   > a `Derivation`; `claim_sources` indexes the reverse direction so `why()` is a lookup.
   > **Measured.** Not measured — there is no number here to produce.
   > `tests/test_fast.py::test_claims_carry_full_provenance` and
   > `tests/test_redact.py::test_provenance_still_resolves_after_the_turn_it_points_at_
   > was_redacted` are the enforcement.

5. **The library must run with no API key and no network.** `NullLLM` + `HashingEmbedder`
   is the default configuration and the one the tests use, and this invariant is about
   `import memvara` and the modules under it — every file this document is a contract
   for. It is not a claim about `memvara-mcp init`'s default output: with the optional
   `cloud` extra installed, that CLI now defaults to authenticating against a hosted
   console (`memvara-mcp login`), a decision made one layer up, in `memvara/server/`,
   and reversible per-invocation with `--mode local`. Nothing below `memvara/server/`
   knows that mode exists.
   > **Claim.** `import memvara` and everything under it works with no key and no
   > outbound connection.
   > **Scope.** The library. **Not** `memvara-mcp init`, which with the optional `cloud`
   > extra defaults to authenticating against a hosted console — a decision one layer up,
   > in `memvara/server/`, reversible with `--mode local`, and invisible below it.
   > **Sketch.** One hard dependency (`numpy`); every optional backend is imported lazily
   > inside the function that needs it.
   > **Measured.** `tests/test_packaging.py::test_every_module_imports_cleanly_in_a_
   > process_that_has_only_numpy`, and `..._the_only_sdks_the_package_names_anywhere_
   > are_the_ones_an_extra_installs`. The install-size figure that follows from it is 2
   > packages against mem0's 33 (`docs/BENCHMARKS.md`).

6. **A multi-hop answer is evaluated at one clock pair.** Every edge on a returned path
   must be checked against the same `(valid_at, known_at)`, pinned once before the walk.
   A path stitched from edges believed at different times is a connection that never
   simultaneously held, and reporting it as a fact is the worst thing traversal can do —
   it is invisible in any result that does not carry its timestamps. Two axes widened
   what has to be pinned; they did not weaken the rule.

   > **Claim.** Every edge on a returned path held at the same instant on both clocks.
   > **Scope.** One call. It says nothing about two calls: a caller who searches, reads an
   > entity out of the result and searches again has two clock reads and no affordance
   > anywhere reminding them, which is the difference the invariant exists to name.
   > **Sketch.** `GraphTraverser._pin` fills both defaults from **one** `utcnow()` before
   > the first hop and passes the pair to every `Store.adjacent` call; no axis is ever
   > forwarded as `None`.
   > **Measured.** `bench/multihop.py`'s interleaving section constructs the failure it
   > prevents: a write placed between two steps of a search-then-search loop retires the
   > fact step one returned and creates the fact step two returns, so the loop reports a
   > chain that held at **no instant**, with full provenance on both hops. The pinned walk
   > returns nothing for the same question. `tests/test_traverse.py::test_a_path_is_never_
   > stitched_from_edges_that_were_never_believed_together` is the assertion.

7. **A filter and a limit may not live in different layers.** Whatever narrows rows has to
   run where the truncation runs, or the top-k is wrong: `Store.adjacent` shipped without
   a `scopes` argument and with the caller filtering afterwards, and on a shared tenant a
   question with 20 answers returned 8. This applies to any future store method that caps
   rows the caller is expected to authorize. It is why `states=` is a store parameter and
   not a comprehension in the facade: `search` over-fetches `k * candidate_multiplier` and
   ranks those, so a state filter applied afterwards finds a retired claim only when it
   happens to land inside that window — twelve live rows against `k=1` is a window of
   five, and the audit comes back empty with nothing saying it was truncated.

   > **Claim.** Whatever narrows rows runs where the truncation runs.
   > **Scope.** Store methods that cap rows the caller is expected to authorize or filter.
   > Not a general rule about filtering: `HybridRetriever` applies `memory_types` after
   > fusion on purpose, and pays for it with a bounded retry when the pool came back full.
   > **Sketch.** `Store.adjacent` takes `scopes`; `state_predicate` is a store parameter
   > rather than a comprehension in the facade.
   > **Measured.** With one user holding 20 readable claims about a hub, a Python-side
   > filter over a store-side page returned **19** of them against 15,000 competing claims
   > and **8** against 40,000, with nothing in the result to say it was partial. The
   > `states=` half is measured too: twelve live rows against `k=1` is a window of five.
   > `tests/test_traverse.py::test_the_scope_reaches_the_store_rather_than_being_applied_
   > after_it` is the assertion.

8. **No MCP client can backdate the transaction clock.**

   > **Claim.** Nothing reachable from the MCP tool surface can make the store record that
   > a fact was believed earlier than it was.
   > **Scope.** **The MCP tool surface only.** `Memvara.remember(recorded_at=...)` is a
   > public Python parameter that writes the record clock directly, and
   > `Reconciler.apply` clamps forward-dating only — backdating is permitted deliberately,
   > because replaying an archived history and importing from another store both need it.
   > A deployment that needs this end to end must not expose the Python API to untrusted
   > callers. This is the invariant most likely to be read as holding further than it
   > does, which is why it is written down.
   > **Sketch.** `_remember` in `memvara/server/tools.py` passes only `valid_from` and
   > `valid_to` through to the library, and no schema in `TOOLS` accepts a transaction-time
   > argument at all — so there is nothing for a model to fill in.
   > **Measured.** Not a number: the falsifiable part is
   > `tests/test_server.py::test_no_tool_schema_exposes_a_transaction_clock_argument`,
   > which walks every property of every tool schema and fails if one appears. That test
   > is what stops the gap reopening silently — a new tool that takes `recorded_at`
   > because it seemed harmless would otherwise ship green.

---

## `memvara/write/`

### `write/gate.py`

```python
class SalienceGate:
    def carries_fact(self, ep: Episode) -> tuple[bool, str]
```
Cheap, deterministic triage: does this turn plausibly contain a durable fact? Returns
`(should_extract, reason)`. Reason is a short slug used in receipts and tests
(`"no_content"`, `"ack_only"`, `"question"`, `"assistant_turn"`, `"has_declarative"`, ...).

Bias toward recall: a false positive costs one extraction call, a false negative loses a
memory permanently. Cheap negatives worth catching: empty/whitespace, pure
acknowledgements ("ok", "thanks", "sounds good"), bare questions with no declarative
clause. This gate is what removes most of the write-path LLM spend, because the majority
of conversational turns carry nothing durable.

### `write/fast.py`

```python
class FastExtractor:
    def __init__(self, registry: PredicateRegistry) -> None
    def extract(self, ep: Episode) -> list[Claim]
```
High-precision, zero-LLM pattern extraction for the handful of statement forms that are
both common and unambiguous — "my name is X", "I live in X", "I work at X", "I prefer X",
"I'm allergic to X", "I no longer work at X" (polarity -1). Precision over recall: emit
nothing rather than a wrong triple; the LLM tier is the fallback. Set
`derivation=Derivation.FAST_PATH`, `extractor="fast/v1"`, and `sources=[ep.id]`.

#### Erasure, and the evidence for it

```python
def residue(self, claim_id: str) -> dict[str, int]           # Store, optional
def erasure_record(self, claim_id: str) -> dict | None       # Store, optional
def prove_erased(self, claim_id: str) -> ErasureProof        # Memvara
```

`erase()` reported success from `erase_claim`'s return code, which proves the code took
the branch it thought it took — the same statement the return value already made, and one
that cannot disagree with it. `residue` is a **live query**: four `SELECT COUNT(*)`s over
the tables a claim's content can survive in (`claims`, `claims_fts`, `embeddings`,
`claim_sources`). A re-hash of what was returned, or a cached count, would not be evidence.

`prove_erased` fails closed. A store with no `residue`, or one whose `residue` raises —
`RemoteStore`, which a `getattr` guard cannot see — yields `proven=False` with a reason,
and `erase()` raises `ErasureIncomplete` rather than returning `True`. Unproven and
proven-gone are different answers and only one of them is an erasure certificate.

Schema 8 adds `erasures`, one row per `erase_claim`, and three properties matter (schema
7 is the FTS scrub that makes erasure remove the text from the file, not only from the
queries):

- **Written before the delete, in the same transaction.** If the audit write raises, the
  exception leaves `erase_claim` before any delete runs and the claim is still there. The
  other order lets a delete succeed and its record fail, which is exactly the state
  nothing downstream can detect.
- **Compensated if the delete then fails.** The ordering above opens the mirror hole: a
  record of an erasure that never happened, which reads as proof to precisely the audit
  that would otherwise notice the claim survived. `erase_claim` removes its own row
  before re-raising. Not a `SAVEPOINT` — `RELEASE` commits into the enclosing
  transaction, so an erasure inside an abandoned `batch()` stopped rolling back with it.
- **It holds no text, subject, predicate or object.** `(claim_id, tenant, scope,
  erased_at, sources, counts)` and nothing else — an audit trail the erased fact can be
  read out of is a copy of it wearing a different name. Keyed on `(claim_id, erased_at)`,
  so erase, restore from backup and erase again is two records rather than one.

**Ordering and durability, not tamper-evidence.** Nothing here is chained or signed, so an
operator with write access can remove a row. A hash-chained log is a different feature and
is commercial (`docs/ROADMAP.md`); what this defends against is a delete that no record was
ever written for.
### `write/reconcile.py`

```python
class Reconciler:
    def __init__(self, store: Store, registry: PredicateRegistry) -> None
    def apply(self, claim: Claim, *, now: datetime | None = None,
              close: Closure = "ended") -> ReconcileResult
```
The contradiction engine. `close` decides which clock stops on whatever the candidate
displaces, and `"ended"` is the only answer this class could reach on its own: it is
told "here is the new value" and never "the old one was a mistake". For a candidate
claim:

1. **Exact duplicate** — a live claim with the same `value_key` exists: do not insert.
   Bump `observation_count`, raise the *storage* strength (`Claim.salience_base`) and
   stamp `last_observed`, merge `sources`, return `action="reinforce"`. The bump goes
   on the base, not on `salience`: the nightly pass recomputes `salience` from the
   base, so writing it there was erased once a claim aged past `0.415 * half_life`
   — 2.9 days for a FAST predicate, and permanently, since age only grows.
2. **Conflict** — the predicate is `Cardinality.ONE` and live claims share the candidate's
   `fact_key` with a different `value_key`: insert the new claim, and for each superseded
   claim set `invalidated_by=<new id>` plus `valid_to=<the new claim's valid_from>`,
   leaving `invalidated_at` unset. The old value stopped being true where the new one
   begins; it was not an error, so nothing on the belief clock moves and
   `get_all(valid_at=<back then>)` still returns it. Under `close="retired"` the axes
   swap: `invalidated_at=now` and `valid_to` untouched, because a correction witnessed
   no world event. Return `action="supersede"` with the list.

   Two things about that step are worth stating separately, because both were silent
   until they were not.

   **A candidate closes a victim only if it is worth at least half of it**, measured on
   `confidence` — `AUTHORITY_SHARE`. Below that the incumbent stays live, the candidate is
   stored beside it, the action is `add`, and a `Dispute` names both values. The rule
   reads `confidence` and not `Derivation` because the write paths already encode source
   authority as a number and say so — `write.fast.CONFIDENCE` is 0.95 rather than 1.0,
   with a comment explaining that the headroom is what keeps user-asserted claims above
   rule output. Ranking by `Derivation` instead would stop a conversational extraction
   ever displacing an application-asserted fact, which is a store that stops learning.
   Every confidence the shipped paths produce — 1.00, 0.95, 0.70, 0.50 — clears half of
   every other, so ordinary traffic passes untouched.

   **The rule binds this step and not `Memvara.supersede`**, which closes its target
   before the reconciler is asked anything — there is no comparison to make when the
   caller has named the victim. Same boundary `close="retired"` sits on: this arbitrates
   an inference, and an instruction is not one. `forget()` and `delete()` are outside it
   too, having no candidate to weigh.

   **A closure clamped to the victim's own start empties its interval**, and the write
   reports a `Collapse`. `close_out` never inverts an interval, so superseding a claim at
   or before the instant it began leaves `valid_from == valid_to`: it survives in
   `history()` and is returned by no `valid_at`, at any instant. It is not nudged forward
   by a tick, because that would invent an interval nothing witnessed. `Memvara.supersede`
   reports the same outcome from its own path.
3. **Retraction** — candidate has `polarity == -1`: close out matching live claims and
   store the negative claim as a tombstone (invalidated *and* ended at `now`, so it can
   never be live) rather than as a live fact. The matches are **ended**, not retired:
   every negative form the write path produces is "no longer" / "used to" / "not any
   more", which is the world moving on. `close="retired"` is the caller saying the
   original was never true.
4. **Accumulate** — otherwise insert. `action="add"`.

```python
@dataclass
class ReconcileResult:
    action: str                  # "add" | "reinforce" | "supersede" | "retract" | "noop"
    claim: Claim | None          # the stored/updated claim
    invalidated: list[Claim]     # claims this one closed out — on whichever clock
                                 # `close=` stopped, so `ended` by default, not retired
    accumulated: Accumulation | None   # landed beside live values under a predicate
                                       # nobody has declared a cardinality for
    disputed: list[Dispute]      # live claims this candidate was not confident enough
                                 # to close; they stayed, it was stored beside them
    collapsed: list[Collapse]    # claims closed at or before their own start, so their
                                 # interval is empty and answers nothing on either clock
    retyped: Retype | None       # an already-known claim re-filed under an asserted
                                 # memory_type; None unless the caller sent one
```

**Re-filing a claim's `memory_type`.** An identical triple is the same fact, so a
re-assertion reinforces the record rather than forking it — and until `Retype` existed the
`memory_type` on that write was dropped, so a claim filed wrongly could not be moved.
Writing it again with the corrected type reported `already-known 1`, left the type alone
and raised the confidence, which made the wrong filing more strongly believed.

`Reconciler._retype` runs immediately before `reinforce`, mutating the claim so that
`reinforce`'s single `put_claim` carries the re-filing and the reinforcement together. It
stamps `meta["retyped_from"]`, mirroring `consolidate.promote_pass`, which has always
reclassified a live claim in place — the operation is not new, only the caller's route to
it.

Two things it deliberately does not do. It does not touch `derivation`: where the fact
came from has not changed, only which drawer it is in, and `promote_pass` re-derives only
because consolidation authored its reclassification rather than re-filing someone else's
fact. And it moves nothing unless the caller **asserted** a type — `Memvara.remember`
forwards the `memory_type` argument it was given and nothing when it was given none, so
the predicate's default never counts as an opinion. That asymmetry is the safety property:
agents re-assert known facts constantly without a view about filing, and treating any
difference as a correction would let the last writer win when the last writer is usually
the one who said nothing.

`memory_type` stays out of `value_key` and `fact_key`, so none of this forks a record.

`WritePipeline` copies that list onto `WriteReceipt.closed`, where `receipt.ended` and
`receipt.retired` split it by `Claim.state`. Anything rendering the list as one word is
wrong for one of the two closures; a supersession is `ended`.

The reconciler is not the only contributor. `Memvara._write_claim` and
`compat/_notes.write_note` close their predecessor *themselves*, before `assert_claim`,
so that the reconciler cannot stamp the wall clock over a caller's `at` — which means the
reconciler then finds no live victim there and reports none. Both therefore append what
they closed onto the receipt after the transaction commits, and both do it conditionally:
a supersession dated in the *future* leaves the predecessor in force at `now`, so the
reconciler does reach it and has already recorded it, and an unconditional append would
name one claim twice. A write path that closes a claim outside `assert_claim` owns saying
so.

### `write/pipeline.py`

```python
class WritePipeline:
    def __init__(self, store, embedder, registry, llm, *,
                 near_dup_threshold: float = 0.97,
                 reinforce_bump: float = 0.25,
                 reject_ungrounded: bool | str = "auto") -> None

    def add(self, episodes: Sequence[Episode]) -> WriteReceipt
    def reextract(self, episodes: Sequence[Episode]) -> WriteReceipt
    def assert_claim(self, claim: Claim) -> WriteReceipt
```

`add()` runs the tiers in order and must populate every field of `WriteReceipt`,
including `llm_calls` (0 whenever the LLM is not consulted) and `latency_ms`:

- **Tier 0 (no LLM):** store the episode; skip content-hash duplicates
  (`store.find_episode_by_hash`). For surviving episodes, embed and check near-duplicates
  against existing claim embeddings — cosine >= `near_dup_threshold` reinforces the
  existing claim instead of writing a new one.
- **Tier 1 (no LLM):** `SalienceGate` drops turns carrying no durable fact
  (count them in `receipt.skipped`), then `FastExtractor` handles what it can.
- **Tier 2 (LLM):** only the turns that survived both and produced no fast-path claim are
  batched into a single `llm.extract(...)` call. Map `source_index` back to the
  originating episode for provenance. Unknown predicates trigger one
  `llm.resolve_predicate(...)` per *new surface form*, cached via `registry.learn_alias`
  / `registry.learn` and persisted through `store.put_spec(spec, tenant)` so it is never
  asked again — including after a restart, and including by another process.

  `reject_ungrounded` guards this tier's output, defaulting to `"auto"`: a proposed
  claim whose object shares not one content word with the episode it cites is a
  fabrication candidate, and the embedder then gets a veto — kept if the best
  chunk-cosine against the source reaches `_GROUNDING_RESCUE_COSINE` (0.40, measured;
  the constant's docstring carries the distributions), refused and counted on
  `receipt.ungrounded` otherwise. `True` is the lexical check alone; `False` is off.
  Only model-proposed claims are ever checked — `remember()` and the fast path do not
  pass through `_claim_from_dict` — and the reason the default is on rather than off is
  that the destructive direction is storing: a fabricated value in a ONE-cardinality
  slot supersedes and ends the true fact that was there. It remains a precision filter
  for wholesale fabrication only — a claim that reuses real vocabulary with an inverted
  or misattributed meaning passes clean — and an embedder failure during the rescue
  fails open, keeping the claim and warning once. Under the default `HashingEmbedder`
  nothing is ever rescued (n-gram cosines on zero-overlap pairs measure 0.0–0.11,
  far under the floor), so `"auto"` degrades to the strict check there.

`reextract()` is `add()` with tier 0 removed, for turns already in the store: a
deployment that ran without a model, or a batch a provider failure left `deferred`. Tier 1
runs and runs first, because the gate is free and the model is not — `add()` commits
episodes *before* gating them, so chitchat in the store is indistinguishable from an
unextracted fact from the outside. An episode that already has claims is skipped and
counted on `receipt.already_extracted`: re-reading stored text is not new evidence, but an
identical claim reconciles to `reinforce`, so a sweep run twice would silently promote what
it had already stored. `Memvara.pending_extraction()` is the work list and applies the same
gate; what it cannot see is a turn a model read and declined, so `reextract()` reports what
it read on `receipt.episode_ids` and the caller feeds those back as `exclude=`.

  Resolution, not classification, is the point. Asking "what cardinality is this?" lets
  `works_at`, `employed_by_company`, `job_employer` and `workplace` become four separate
  slots that can never contradict each other; a red-team simulation of 10k extractions
  over six concepts produced 41 predicates and four simultaneously-live employers that
  way. Asking "which existing predicate is this?" spends the same one-per-form call on
  merging instead, and a deterministic morphological pre-pass answers most of them for
  free before any model is consulted.

Every produced claim goes through `Reconciler.apply()`, and every stored claim gets its
embedding written via `store.set_embedding()`.

---

## `memvara/retrieve/`

### `retrieve/fusion.py`

```python
def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[tuple[str, float]]],
    *, k: int = 60, weights: Mapping[str, float] | None = None,
) -> dict[str, float]
```
Standard RRF: an item at rank `r` (0-based) in list `L` contributes
`weights[L] / (k + r + 1)`. Rank fusion rather than score fusion, because BM25 scores and
cosine similarities are not on comparable scales and normalizing them is guesswork.

### `retrieve/scoring.py`

```python
def recency_factor(claim: Claim, registry: PredicateRegistry, now: datetime) -> float
def normalized_score(...) -> float   # Result.score, in [0, 1]
def final_score(fusion: float, *, recency: float, confidence: float, salience: float,
                w_recency: float, w_confidence: float, w_salience: float) -> float
```
`recency_factor` is exponential decay on the predicate's half-life:
`0.5 ** (age_days / half_life_days)`, age measured from `claim.trace_from`
(`max(valid_from, last_observed)`) — from `valid_from` alone, a fact restated daily
for ninety days still scored as ninety days stale. A `STATIC`
predicate's 100-year half-life keeps its factor at ~1.0, so birthplaces do not decay out
of the ranking while "what I'm working on today" does.

### `retrieve/hybrid.py`

```python
class HybridRetriever:
    def __init__(self, store, embedder, registry, *,
                 w_vector: float = 1.0, w_lexical: float = 1.0, rrf_k: int = 60,
                 w_recency: float = 0.25, w_confidence: float = 0.15,
                 w_salience: float = 0.10, candidate_multiplier: int = 5,
                 w_graph: float = 0.0, graph_seeds: int = 5, graph_depth: int = 2,
                 w_temporal: float = 0.0, traverser: GraphTraverser | None = None,
                 intent_weighting: bool = True,
                 entities: EntityRegistry | None = None) -> None

    def search(self, query: str, scope: Scope, *, k: int = 10,
               as_of: datetime | None = None, valid_at: datetime | None = None,
               known_at: datetime | None = None,
               states: Collection[str] | None = None,
               include_invalidated: bool | None = None,
               memory_types: Sequence[MemoryType] | None = None,
               min_score: float = 0.0, anchored: bool = False,
               include_episodes: bool = False) -> list[Result]
```

Search must:
- expand `scope` via `scope.ancestors()` so a session query also sees user-level memory;
- run vector and lexical retrieval over `k * candidate_multiplier` candidates each;
- fuse with RRF, then rescore with recency/confidence/salience;
- resolve the three time keywords through `types.time_axes` **before anything else**, so
  `as_of` + `valid_at` raises whatever else the call would have done;
- pass `valid_at` and `known_at` through to the store so **time travel returns what we
  believed then, and what we now believe was true then**, including claims later
  invalidated. Decay is measured at `known_at`, not `valid_at`: recency asks how long ago
  we last heard something, and that is a question about the belief clock;
- populate `Explanation` on every `Result` — per-retriever rank and raw score, the fusion
  score, each scoring factor, and the final score. A result with no explanation is a bug;
- say on every `Result` what tied it to the question (`Explanation.anchor`), and with
  `anchored=True` return only the results something did. See
  [`retrieve/anchor.py`](#retrieveanchorpy).

#### The third leg

At `w_graph > 0` a graph leg runs **after** the first fusion and the whole list is fused
again, three-way. It cannot run before: its seeds are the folded entity keys of the
best-scoring claims (`retrieve/spread.seed_keys`), which is Zep's φ_bfs and is the
decision that keeps the leg cheap — no entity extractor over free-text queries, and no
second vocabulary to disagree with the store's.

The leg must:
- **seed on content, not on ids.** The fused order breaks ties on the item id and a claim
  id is a `uuid4` minted at ingest, so seeding straight off it would make which entities
  get walked a property of which ingest ran. `seed_keys` re-sorts on `value_key`. The
  same rule binds the store underneath: **every `ORDER BY` that sits above a `LIMIT`
  ends in a content key before `id`** — `s, value_key, id` for claims, `s, hash, id` for
  turns, `ABS(ts - anchor), hash, id` for `episodes_near`. Without it the tie is settled
  by rowid, and because the cap is in the same statement that decides which rows come
  back at all, not merely how they are arranged;
- **bound seeds by key count**, not by claim count: the key list is what reaches
  `Store.adjacent`, and frontier width is what a hop costs;
- **pass `valid_at`/`known_at` through unchanged**, so the walk pins the pair `search()`
  was asked about (invariant 6) rather than reading the clock again;
- **run only when `live` is among the wanted `states`.** `Store.adjacent` walks the live
  edges at the pinned instant and cannot be asked for anything else, so every row this
  leg can produce belongs to the live population. Gated, not post-filtered: a post-filter
  would have to test `claim.state`, which is the state *now*, and at a historical
  `known_at` the lookup legs correctly return rows that were live then;
- **take the best path's score per claim**, never the sum — a path score is a relevance,
  and summing would rank a hub on nine weak chains above a claim on one strong one;
- **collect one path per *undirected* identity** (`Path.undirected`), before `k` is
  spent rather than after. `seed_keys` emits both ends of each top-ranked claim, so the
  same row read from two ends is the normal case for the head of the list. The dedup is
  at collection only — the frontier keeps both readings, because they extend to
  different places;
- **abstain, not vote zero, when it did not run.** `_Legs.graph_active` is the same
  distinction the other two legs carry, and it is what keeps a two-leg query from being
  scored as though a third leg had rejected everything;
- **degrade rather than raise.** `RemoteStore.adjacent` exists and raises, so a `getattr`
  guard cannot see it: the `NotImplementedError` is caught, `DegradedRetrievalWarning`
  fires once per retriever, and the leg stays off for that retriever's life.

#### The fourth leg

At `w_temporal > 0` a fourth leg runs over **raw turns**: `Store.episodes_near` returns
the `limit` turns closest to the anchor, nearest first, and `retrieve/temporal.py` turns
their timestamps into an absolute [0, 1] closeness. The anchor is `valid_at`, else
`known_at`, else now — **given, never parsed**, because a date parser on the read path is
a second extractor answering a question the caller who wrote `valid_at=` already answered.

Episodes and not claims: a claim carries a predicate-keyed half-life, which knows what raw
proximity cannot — whether a fact from 2019 is stale. A `born_in` from 2019 is as current
as it will ever be.

Two properties are load-bearing.

- **The sort and the cap are one SQL statement.** Design invariant 7. Listing a scope's
  turns and dropping the ones after `valid_at` in Python filters a page the store already
  truncated, so a time-travel query comes back short with nothing saying it was partial.
- **The leg abstains when nothing is within a half-life of the anchor.** Measured: without
  it, a query with no instant anchors on *now*, an archival corpus scores every turn at
  ~0.005 proximity, and fusion — which reads positions — still takes rank 0, rank 1, rank
  2 from it. That cost 2.4 points of LongMemEval temporal-reasoning R@12. The vector and
  lexical legs have had the same guard from the start.

#### `retrieve/intent.py`

```python
def classify(query: str, registry=None) -> Intent   # lookup | temporal | relational | open
def is_relational(query: str, registry=None) -> bool
def weights(intent, *, vector, lexical, graph, temporal) -> tuple[float, ...]
```

Deterministic, model-free, and read off the *raw* tokens rather than `analyze()`'s terms —
`when`, `whose`, `between` are all stopwords, and they are exactly the words that say what
kind of question this is. The classes are checked in priority order, not as a taxonomy:
time first, because a wrong instant is wrong in a way extra recall does not repair.

`is_relational` is the reading the priority discards. A question can be about an instant
*and* about a chain — "who currently leads the team that owns the checkout service" — and
`classify` returns `temporal`, whose row zeroes the graph weight. `HybridRetriever._weights`
asks this second question and keeps the graph weight when the answer is yes, exactly as it
already did for a caller who named the instant as an argument; the comparison guard applies
to both, and `Explanation.intent` still reports the primary reading.

A question names a predicate in whatever form it inflects it. `predicate_refs` and
`observed_refs` fold both the predicate's content tokens and the question's through
`schema.word_stem` — the fold the registry uses to decide that `employer` and `employed_by`
are one predicate — so "who *leads* the team" names `team_lead` and "where is it
*deployed*" names `deploy_region`. Every content token still has to be present, which is
what keeps the match from becoming a token index.

`MULTIPLIERS` scales the *configured* weights rather than replacing them, so a deployment
that tuned `w_vector` keeps its tuning. Every entry is 1.0 except the graph column, where
`lookup` and `temporal` are 0.0 — and that zero is a **gate**, checked before the traverser
is called, so those queries pay nothing rather than paying for a walk that is then
multiplied away. `intent_weighting=False` runs every query at the configured weights and
leaves `Explanation.intent` unset, which is how a ranking difference is attributed to this
stage rather than argued about.

Any multiplier that is not 1.0 must come from a per-category sweep recorded in
`docs/BENCHMARKS.md`. A number picked because it sounds right is a ranking change with no
evidence behind it.

### `retrieve/anchor.py`

```python
SUBJECT, OBJECT, PATH = "subject", "object", "path"
SELF_SUBJECT = "user"

def query_tokens(query: str) -> frozenset[str]
def anchor_of(claim: Claim, tokens: frozenset[str], spellings=...) -> str | None
```

What tied a result to the question, read off the rows rather than off the score. A claim
is *anchored* when the question names one of its ends — `Claim.subject_key` or
`Claim.object_key`, the folded identities the write path stamped, against the question
folded the same way by `entity_key` — and *derived* when the graph leg reached it by walking
out of an anchored claim. `Explanation.anchor` reports which; `None` is the finding: the row
surfaced on vocabulary alone, which on a question the store cannot answer is what the best
available row looks like from inside a ranker.

Invariants:

- **No extractor runs over the query.** The candidates supply the entities, exactly as the
  graph leg's seeds do, and the question is only asked whether it contains them. Every
  content token of a key has to be present, so `Project Chronos` does not anchor a row
  about `Project Atlas` on the strength of `project`.
- **A derivation starts at the entity the question named.** `_graph_search` returns the
  ids on paths whose first node is the *named* end of an anchored candidate. Not its
  other end: from `Project Atlas/deploy_region=eu-west-1` a walk out of the value reaches
  every project in `eu-west-1`, one hop away at score 1.0, on the very predicate asked —
  derivations from the answer, not from the question. Not an unanchored seed either —
  the lookup legs' best guess on a question about nothing the store holds — or every
  negative would be answered from that guess's neighbourhood. Under `anchored=True` with
  nothing named the walk is not run at all, since nothing it found could survive.
- **The self subject is named by a pronoun, a possessive is a mention, and an alias is a
  spelling.** `user` is what `write/fast.py` and `write/pipeline.py` file a first-person
  statement under, and "where do I live" has to reach it; `entities._tokens` drops
  apostrophes so "Bob's" would fold to `bobs`; and `EntityRegistry.spellings(owner, key)`
  returns the key and its learned aliases, resolved under the reader's own owner and no
  wider, for the reason `Memvara._probe_entities` gives.
- **`anchored=True` filters claims and retries like `memory_types`.** An anchored claim
  with little vocabulary in common with the question sits past the first cut exactly as a
  filtered memory type does, so the same widened second pass runs. `min_score` deliberately
  gets no retry — deeper candidates have less evidence, not more. Episodes are untouched: a
  turn has no subject to name.

It is a filter on the *entity*, not on the slot. Asked about the reporting service's
authentication strategy, a store holding only who owns the reporting service correctly
keeps that row — the question is about an entity the store knows — and telling that row
from the answer is a question about the predicate, which nothing here judges.

### `retrieve/traverse.py`

```python
class GraphTraverser:
    def __init__(self, store, registry, *, damping: float = HOP_DAMPING,
                 beam: int = 64, edge_limit: int = 1000) -> None

    def neighborhood(self, entity: str, scope: Scope, *, depth: int = 2, k: int = 10,
                     min_hops: int = 1, predicates: Sequence[str] | None = None,
                     as_of: datetime | None = None, valid_at: datetime | None = None,
                     known_at: datetime | None = None,
                     min_score: float = 0.0) -> list[Path]
    def paths_between(self, source: str, target: str, scope: Scope, *,
                      depth: int = 3, k: int = 3, ...) -> list[Path]
```

Traversal must:
- **pin one clock pair before the first hop** and pass that same
  `(valid_at, known_at)` to every `Store.adjacent` call. Neither axis may be forwarded as
  `None` — the store substitutes its own clock per call, so a 3-hop walk would evaluate 3
  hops at 3 instants and could return a path that was true at none of them. One clock read
  fills both defaults, so an argument-free walk is still a single coherent moment. This is
  invariant 6 below;
- drop `polarity <= 0` before a claim becomes an edge, so the guarantee holds for every
  store rather than for the ones that remembered;
- pass `scope.ancestors()` to `adjacent` **and** re-check `Scope.sees` on what comes
  back. The first is what makes the answer correct under a cap; the second is what makes
  the guarantee ours rather than a third-party store's;
- keep the score non-increasing along a path — a path may never outscore its own prefix,
  which is what makes `min_score` prunable mid-walk exactly rather than approximately.
  Salience is therefore excluded: it is unbounded above 1.0 by design;
- bound everything (depth, beam, per-hop `edge_limit`, cycle check) and order totally,
  with no `uuid4` deciding anything observable.

---

## `memvara/core.py` — the prompt rendering boundary

Only one method in this library renders stored text into something a model is asked to
treat as its own knowledge, and its contract is a security contract rather than a
formatting one. `SECURITY.md` names it as an in-scope attack surface; this is the
implementation side of the same rule.

```python
class Memvara:
    RECALL_HEADER: str            # frames the live block as data, not instructions
    RECALL_HISTORY_HEADER: str    # "No longer true — ..." in the first three words
    RECALL_EPISODE_HEADER: str    # says "said", not "true"
    RECALL_EPISODE_CHARS: int     # 280 — a pasted stack trace cannot become the prompt

    def _safe_line(self, text, limit=None) -> str
    def recall(self, query, *, k=8, min_score=0.0, header=None, ...,
               include_episodes=False, episode_header=None,
               include_history=False, history_header=None,
               budget=None, counter=_approx_tokens,
               with_ids=False) -> str | RecallResult
    def _past_by_claim(self, claims, tenant=None, user=None, agent=None,
                       session=None) -> list[list[str]]
```

`recall()` must:

- **take an explicit signature, never `**kwargs`.** Forwarding arbitrary keywords into
  `search()` would expose `as_of`, `states` and `include_invalidated` here, and the latter
  two resurrect retired claims into a live prompt — an un-delete reachable by anyone who
  can influence a parameter. `states=["retired"]` is the sharper form: a prompt built from
  nothing but the records we stopped believing. Time travel and audit reads stay on
  `search()`;
- flatten every rendered line through `_safe_line`, which collapses whitespace, strips
  leading list and heading markers, and maps `[`/`]` to their fullwidth forms
  (`_FORGEABLE`), so stored text cannot open its own bullet list, repeat a header, or
  finish a line with something that parses as the next result row. The first two defend
  the gaps *between* lines; the third defends the rest of the line a claim is already on,
  which ordering metadata-first cannot reach. `memvara/server/tools.py:safe_line` calls
  this method rather than reimplementing it — it was a copy once, the two sets drifted,
  and the same stored value was then neutralised differently depending on which surface
  replayed it. Episodes are additionally truncated to `RECALL_EPISODE_CHARS`;
- keep the three blocks in order — claims, then history, then episodes — each under its own
  header, and emit a header only when its block is non-empty;
- under `budget=`, **drop whole notes and never part of one**, filling downward from the
  complete block rather than upward from nothing. Downward because the line that reports
  the drop is itself a line: a block one note short can be *larger* than the complete one,
  so filling upward stops at the first overshoot and can render three notes where five
  would have fitted. The drop-notice is also the floor — a budget too small for even the
  first note returns the notice alone, over budget, because an empty block is
  indistinguishable from "nothing is stored". Content never overruns; only the sentence
  saying there was content can.

`_past_by_claim` is the whole of `include_history`, and its contract is one line:

> **The filter is `state == "ended"`, never `state != "live"`.**

That is a security boundary, not a tidying step. `history()` returns every value a slot
ever held, retired ones included. An `ended` value is the fact's own past and we still
believe it was true while it was in force; a `retired` value is one we were wrong about or
were asked to delete, and rendering it is exactly the un-delete the explicit signature
exists to prevent. A claim that ended and was *later* retired reports `retired` and stays
out — which is precisely the case the looser spelling would admit.
`tests/test_api.py::test_recall_can_carry_the_past_of_a_fact_without_carrying_a_retired_one`
holds a live, an ended and a retired value **in one slot**, so a `!= "live"` filter cannot
pass it.

It is also keyed on `fact_key` and deduplicated, so a multi-valued predicate returning
four live values costs one `history()` call rather than four and renders its past once
rather than four times.

**It returns one list per claim, index-aligned, rather than one flat list** — which is
what the name says and is why the name changed. Flat, the past of note three could
outlive note three: `budget=` drops notes from the end, and history lines that did not
know which claim they belonged to stayed behind, leaving a fact's former values rendered
under a fact no longer in the block. Grouping costs nothing — the same one lookup per
slot — and makes the drop take the two together.

---

## `memvara/store/`

### The two time axes

`as_of` exists on the public facade and **nowhere below it**. `Memvara`, `ScopedMemvara`
and `AsyncMemvara` accept all three keywords and resolve them through
`types.time_axes(as_of, valid_at, known_at)`, which returns the pair and raises if
`as_of` was combined with either axis. Everything under the facade speaks in the pair.

Every `Store` method that used to take `as_of` now takes `valid_at` and `known_at`, both
**keyword-only**. That is deliberate: they replaced a positional argument, and a caller
still passing an instant third would otherwise be silently reinterpreted as `valid_at` —
a wrong answer with no error, which is the failure the split exists to remove.

```python
competing_claims(tenant, fact_key, *, valid_at=None, known_at=None)
adjacent(tenant, keys, *, outgoing=True, incoming=True, predicates=None,
         valid_at=None, known_at=None, scopes=None, limit=1000)
candidate_ids(scopes, *, valid_at=None, known_at=None, states=None,
              include_invalidated=None)
lexical_search(query, scopes, limit, *, valid_at=None, known_at=None, states=None,
               include_invalidated=None)
vector_search(qvec, scopes, limit, *, valid_at=None, known_at=None, states=None,
              include_invalidated=None)
episode_candidate_ids(scopes, *, valid_at=None, known_at=None)
lexical_search_episodes(query, scopes, limit, *, valid_at=None, known_at=None)
vector_search_episodes(qvec, scopes, limit, *, valid_at=None, known_at=None)
```

A SQL-backed store additionally implements `SQLStore`, a **second** protocol declared in
`store/base.py` and deliberately not folded into `Store` — the clause builders are SQL
generation, and `Store` promises a Qdrant or LanceDB backend can implement it without
them:

```python
_state_clause(valid_at, known_at, states=None, alias="")        -> tuple[str, list]
_live_clause(valid_at, known_at, include_invalidated, alias="") -> tuple[str, list]
_happened_clause(valid_at, known_at, alias="")                  -> tuple[str, list]
```

### `ask()` reconstructs an ending the row cannot date

`Reading.stated` — "what would this store have answered on T" — is the one read in the
API that does **not** reduce to the four columns, and it is the one that disagrees with
`get_all(as_of=T)`. The disagreement is deliberate and it is the point of the method.

A row's `valid_to` is written **in place** by the write that displaces it. So the row
carries its own ending but not the instant that ending came to be believed, and any
predicate over the four columns applies an ending that had not been recorded at `T`:

```
Rome    valid 2026-01-01 → 2026-03-01,  recorded 2026-01-01
Berlin  valid 2026-03-01 → open,        recorded 2026-03-22

get_all(as_of=2026-03-15)         -> []       Rome's ending applied a week early
ask(..., at=2026-03-15).stated    -> [Rome]   what the store actually held that day
```

`core._stated_at` closes the gap with the supersession chain, which the row does carry:
an ending is dated at `invalidated_by`'s `recorded_at`, the instant the pointer was
written. That is `_displaced_by`'s rule, unchanged — `why()` has used it since a July
view started reporting an August replacement, and its docstring argues it at length,
including why `invalidated_at` is the wrong stamp on a double-closed row. `invalidated_at`
*is* consulted here, for the retirement, where it dates exactly the right event.

The case it cannot recover is an ending whose successor has since been erased: the
pointer survives and its target does not, so the closure falls back to the row's own
`recorded_at` — the earliest instant it could have been known, which makes the claim stop
answering sooner rather than later. Under-reporting a past answer is the safe direction
in a store somebody is auditing, and it is the direction `_displaced_by` already chose.

`get_all(as_of=T)` is not being fixed to match. It is a scope-wide predicate over rows
and has no timeline in front of it; making it walk the chain would turn every read into a
per-slot join. The two answer different questions and both are documented as doing so.

### The three states

A claim is `live` (neither clock closed), `ended` (valid time closed — the world moved
on) or `retired` (transaction time closed — the record was wrong). That is `Claim.state`,
and `store/base.py` exports the vocabulary and the SQL that selects it:

```python
STATES            # ("live", "ended", "retired") — also the canonical order
ClaimState        # Literal of the same three
resolve_states(states=None, include_invalidated=None, *, default=("live",))
state_predicate(at="?", *, states=None, alias="")   -> (sql, axes)
stored_state_predicate(states=None, *, prefix="")   -> sql
live_predicate(at="?", *, include_invalidated=False, alias="") -> sql
```

`resolve_states` is **the one place either spelling is interpreted**, so no surface can
invent its own reading of the older flag. It returns a canonical tuple in `STATES` order,
so one requested population compiles to one string however the caller spelled it.
Passing `states=` and `include_invalidated=` together raises: there is no reading of the
mix in which one of the two is not being ignored. Nothing is deprecated and nothing
warns — `filterwarnings = ["error::DeprecationWarning"]` would turn a warning into a
failure at every existing call site.

`default=` is what the method returns when neither argument is given, *and* what
`include_invalidated=False` means on it — one parameter names both, so they cannot drift
apart. It is `("live",)` on the read path and `("live", "ended")` on `iter_claims`.

`_state_clause` is the parameterised form of `state_predicate` and the method every read
filter in a SQL backend routes through. **It is the binding site** — the only place in
this repository that binds the state predicate's markers. `state_predicate` returns the
SQL *and* an axis list naming the clock behind each marker in order (`("known", "known",
"valid", "valid")` for the live-only case), so binding is a comprehension over that list
rather than a remembered order. That is what makes the one silent error unwritable: a
belief instant bound onto a world column answers identically to a correct one on every
`as_of` call, because those pass the two axes equal.

`_live_clause` is now `_state_clause` with the two-valued alias applied, and is neither
deprecated nor changed in meaning. `live_predicate` is likewise the two-state alias of
`state_predicate` — it drops the axis list, so a caller that binds markers should prefer
the general form. Both remain exported.

`stored_state_predicate` is the member of the family for a walk with no clock to read.
`iter_claims` pages over rows rather than answering a question about a moment, so its
filter is the *stored* state — what `Claim.state` reports — which differs from
`state_predicate` exactly where a timestamp is in the future: a claim retired next
October is `retired` here and still believed there. It returns `""` for the complete set,
meaning "no filter", so the caller drops it rather than emitting a tautology into a paged
scan. Its `prefix` is load-bearing, not decorative: `iter_claims` passes `"+"` to keep
the planner off `cl_live`, which would otherwise sort every page back into rowid order.

**`iter_claims`'s unflagged view is `("live", "ended")`, not live-only.** It has always
meant "every row we still believe", and it must stay that way: `reembed()` walks it, and
narrowing the default would silently stop re-encoding every superseded version in the
store. `include_invalidated=True` is unchanged and still means all three.
`include_invalidated` also stays *positional* there, because it always was.

### The three states do not tile the store

Asking for all three is not the union of the three parts. `Claim.state` is absolute while
`state_predicate` is as-of, so a claim recorded but not yet in force at `valid_at` — a
fact scheduled to start next month — is named by none of the three. The complete set
therefore collapses to the belief floor alone:

```python
state_predicate("?", states=STATES)
# ('(recorded_at <= ?)', ('known',))
```

which readmits that row and leaves `valid_at` with nothing to constrain. That is exactly,
and deliberately, the semantics `include_invalidated=True` has always had, and
`tests/test_bitemporal.py::test_asking_for_all_three_states_is_the_audit_view_valid_at_cannot_narrow`
pins it so it cannot drift into a surprise.

### The clauses themselves

`_live_clause` is four constraints, two per axis, each reading only its own clock:
`recorded_at <= known_at`, `invalidated_at > known_at`, `valid_from <= valid_at`,
`valid_to > valid_at`. `None` on an axis means that axis reads the clock, substituted
once per call — and one read fills both defaults so the two cannot land microseconds
apart.

The SQL itself is not written in either clause builder. `at` is the SQL *expression* for
the instant, substituted at every axis — a bind marker (`"?"`, `"%s"`) or a server clock
(`"now()"`) — so a counter with no store instance, on a raw connection, in another
repository, can still ask the same question. One expression rather than one per axis
because such a counter is always counting *now*, and two markers would let a caller bind
the pair transposed, which no `as_of` query can reveal. Where markers are used the four
bind **known, known, valid, valid**.

`include_invalidated=True` — equivalently `states=STATES` — lifts the **whole valid-time
interval** plus the retirement, leaving only `recorded_at <= known_at`. That is more than
the name promises, it is existing behaviour, and it must stay identical across backends:
under it, `valid_at` has no effect at all. The belief floor is the one clause that never
lifts, under any subset — returning something first heard in July when asked what we
believed in March is the only way a bitemporal read can actively lie.

A subset containing `retired` is written with the retired disjunct **first**, so its
belief marker stays ahead of the world markers and the axis discipline generalises rather
than changing shape per subset. `base._either` parenthesises each disjunction inside
*and* out, because `AND` binds tighter than `OR` and a bare disjunction dropped into a
conjunction re-associates silently — `floor AND retired OR in_force` is
`(floor AND retired) OR in_force`, which drops the belief floor in the direction that
answers "what did we believe in March" with something first heard in July.

`_happened_clause` is the episode form. A turn has no separate record time, so its single
`ts` is both its `valid_from` and its `recorded_at`; substituting that into the four
clauses above leaves `ts <= min(valid_at, known_at)`. There is no `include_invalidated`
and no `states` because nothing retires a turn.

`types.Claim.is_live(as_of=None, *, valid_at=None, known_at=None)` is the Python mirror
of `_live_clause` and is held to the same wording clause for clause. Three copies of one
predicate is three chances to disagree; `tests/test_bitemporal.py` checks the Python one
against the SQL one row for row.

### Erasure removes the bytes, not just the rows

Two settings, covering two different halves, and neither is SQLite's default.

`PRAGMA secure_delete=ON` (in `SCHEMA`, so it applies to every writer connection) covers
ordinary tables: without it a deleted row's bytes sit in a free page, readable in the file.

FTS5's own `secure-delete` option (set once in `_migrate_to_v7`, persistent in the table's
config) covers the text indexes, and this is the half that is easy to miss.
`DELETE FROM claims_fts` does **not** remove the document's terms — FTS5 writes a delete
marker and keeps the terms as *live rows* in the `claims_fts_data` shadow table. They are
not residue in freed space, so `VACUUM` never reclaimed them, and an erased claim's words
stayed in the file indefinitely while `erase_claim` reported per-table counts as evidence.

The option is not retroactive, which is why the migration also runs one `optimize`: only a
merge discards what the existing markers hide.

The in-place rewrite has a second consequence, on the write path rather than the erasure
one. A delete under `secure-delete` is not the cheap append a delete marker would be, so
`put_claim` rewrites a claim's FTS row **only when its text has changed** — it reads the
stored `text` in the same SELECT that fetches `rowid` and `sources`, before the upsert
overwrites it. Without that guard every reinforcement rewrote a doclist to reproduce what
was already there, and enough of those inside one uncommitted transaction made FTS5 raise
`SQLITE_CORRUPT_VTAB`, reported as `database disk image is malformed` on a file that was
sound and that a `rollback()` restored to working order. On one measured store the
transaction that used to die at its 3,520th write instead ran to 19,420 and committed —
and every one of those 19,420 writes was of unchanged text.

Anything testing this must **read the file**, not query the store. Every query already
answered correctly — that is precisely why it went unnoticed. See
`tests/test_erasure_residue.py`.

The remaining residue is the write-ahead log, which a checkpoint or a clean `close()`
clears. `SECURITY.md` states that as the boundary.

### `stats()`

```python
{"episodes", "claims", "live_claims", "ended_claims", "invalidated", "embeddings"}
```

`live_claims` and `ended_claims` are both taken from `_state_clause`, and so from
`state_predicate`, rather than spelled out — a counter that writes its own copy of the
predicate is exactly how the cheap version got into three files. Neither is a column
test, and neither is derivable from the others. On a store holding one live claim, one
ended claim, one that ended and was *later* retired, and one recorded but not in force
until next year, `stats()` reports `claims=4, live_claims=1, ended_claims=1,
invalidated=1` — and:

| cheaper spelling | gives | truth | why |
|---|---|---|---|
| `invalidated_at IS NULL` | 3 | `live_claims` = 1 | counts every superseded version as live |
| `valid_to IS NOT NULL` | 2 | `ended_claims` = 1 | the ended-then-retired row is already inside `invalidated` |
| `claims - live_claims - invalidated` | 2 | `ended_claims` = 1 | the residual also holds the scheduled claim, which is in no state at all |

`ended_claims` and `invalidated` are **disjoint** — the state predicate excludes the
ended-then-retired row from the first, so it is counted once — which is why the key had
to be added rather than left to subtraction. It was the largest non-live population and
the only one with no key.

**The counts do not sum**, and the leftover is not the ended rows: `1 + 1 + 1` against
`claims = 4`. `claims` is the only total that covers everything, and a backend that
"corrects" the arithmetic has reintroduced the conflation.

That is the one change in this project that is wrong *silently* downstream: the old
one-column test is still valid SQL and still valid Python, it used to be right, and it
now over-counts with nothing to raise. [`docs/UPGRADING.md`](UPGRADING.md) carries the
grep list for finding copies of it.

### The store-level graph gate

`HybridRetriever` closes the graph leg — after the intent weighting, so it can undo what
`classify` opened — when `connectivity()` says nothing in the tenant chains. The condition
is `joinable_claims == 0`, not a threshold: a store with literally no joins provably has
nowhere for a walk to go, whereas any percentage picked from the two corpora that exist
would be a constant fitted to two points.

**Why the leg hurts on such a store**, given that `joinable_claims == 0` does not stop the
walk returning rows. It stops it returning *paths*. At depth 1 it still fans out from
whichever hub the seeds share and returns other claims about it, ranked by a path score
that is near-uniform when every path is one hop — and fusion reads positions. Measured on
LongMemEval: 1.6 points of single-session-user R@12, and `graph_depth=1` and `2` cost
exactly the same, so all of it is the fan-out and none of it is the second hop.

**Placement is the design.** The second-chance rule in `_gather` can only widen: its guard
is `weights.graph <= 0.0 < self.w_graph`. A veto wired into it returned False on all 802
of LongMemEval's gate calls and the run still lost the same 1.6 points, because `classify`
had already opened the leg. The gate therefore sits after `intent_weights`, not inside it.

`_store_has_joins` caches per tenant and re-measures every `GATE_RECHECK_EVERY` searches,
on a counter rather than a clock — a retriever that behaved differently at 3am would be
untestable, and this repository pins `now=` everywhere for that reason. The staleness is
one-directional: a store that gains joins stays gated for at most that many searches,
which degrades to `w_graph=0.0`, the shipped default. It cannot fail the other way,
because claims do not un-join except by retirement and the liveness predicate already
excludes those.

`{}` keeps the leg. A backend without `connectivity`, or a hosted facade too old to report
the counts, has not measured anything — and reading that as "no joins" would switch a
working graph leg off on every third-party store at once.

### `connectivity()`

```python
{"live_claims", "joinable_claims"}
```

A separate method, and separate on purpose. `joinable_claims` counts live claims whose
`object_key` is the `subject_key` of another live claim; the ratio against `live_claims`
is the **join rate**, which is the number that decides whether `read_w_graph > 0` can
pay for itself.

It is not in `stats()` because `Memvara.__repr__` calls `stats()`, and the join is a
semi-join over the whole claim table — about 60 ms on 26,403 claims against that call's
69 ms, so folding it in would roughly double the cost of printing a store.

**The spelling is load-bearing, and the reason is not the one this section first gave.**
`IN (SELECT ...)` is chosen over `EXISTS (SELECT ...)` because the method takes
`tenant: str | None` and only `IN` is fast for both. Median on 26,403 claims:

| form | `tenant='2wiki'` | `tenant=None` |
|---|---:|---:|
| `IN`, uncorrelated | 39 ms | 34 ms |
| `EXISTS`, correlated | **27 ms** | **42,302 ms** |

`EXISTS` is the faster form when it has an index and unusable when it does not. It
correlates on `c.object_key`, so it runs per outer row, and whether that is cheap depends
on `cl_subj` — which is `(tenant, subject_key, invalidated_at)`. A composite index is only
usable from its leading column, so the subquery reaches it only when `tenant` is bound.
`EXPLAIN QUERY PLAN` shows `SEARCH d USING INDEX cl_subj (tenant=? AND subject_key=?)`
in one case and a bare `SCAN d` in the other: 26,401 rows visited 26,401 times.

`IN` does not correlate. SQLite evaluates the subquery once into a transient list behind a
bloom filter, index or no index, so it gives up 12 ms on the tenant path to be immune on
the other. **The rule is not "correlated subqueries are slow"** — it is that a correlated
subquery inherits its index's leading column as a requirement, and an optional `tenant`
cannot always supply one.

Both sides of the join must be edges the traverser would follow: the liveness
predicate, for `adjacent`'s reason that a path through a retired claim is not a path,
plus `GraphTraverser._edges`' three rules — no negations, no empty ends, no self-loops.
`_WALKABLE` in `store/sqlite.py` is those three written once, because a rate built from
edges the walk refuses is a rate that promises hops which will not happen. The
denominator stays `live_claims`: a leaf is an answer to "what share of this leads
somewhere", not a row to exclude.

It is in `OMITTABLE`. A backend that leaves it out costs `memory_stats` its join-rate
line and nothing else; retrieval is unaffected. `Memvara.connectivity()` returns `{}` in
that case, which is **not** `{"live_claims": 0, "joinable_claims": 0}` — the first is a
backend that did not look, the second is a measured star, and only the second is a
finding.

Why the distinction earns a method at all: every claim is already an edge, so "is this
store a graph" is always yes and predicts nothing. Connectivity is what varies. Measured
on the two public corpora with identical retrieval code, 2Wiki joins at 40.6% and the
graph leg takes chained questions from 28.3% to 43.8%; LongMemEval joins at **0.0%** —
one subject, 78 leaf objects, no two-hop path in the store at all — and the leg loses 1.6
points. See [`docs/BENCHMARKS.md`](BENCHMARKS.md).

`invalidate()` and `set_valid_to()` are in `Store` and **no engine path calls either**.
Closing a claim moves one clock; `invalidate` writes `invalidated_at` and
`invalidated_by` together, which is the conflation that was removed, and `set_valid_to`
writes no pointer. They remain as the protocol's only single-statement writes, and
because `set_valid_to(id, None)` reopens an interval — the one thing `close_out` cannot
do, deliberately, since a write path able to un-end a fact would do it by accident.

---

## `memvara/consolidate/`

```python
class Consolidator:
    def __init__(self, store, embedder, registry) -> None

    def decay(self, tenant: str | None = None, now: datetime | None = None) -> int
    def merge_duplicates(self, tenant: str | None = None, threshold: float = 0.97) -> int
    def promote(self, tenant: str | None = None, min_observations: int = 3) -> int
    def run(self, tenant: str | None = None,
            now: datetime | None = None) -> dict[str, int]
```

- `decay` multiplies `salience` by the predicate's recency factor, floored at `0.05` so
  nothing decays to zero and disappears from ranking entirely.
- `merge_duplicates` finds live claims sharing a `fact_key` whose embeddings exceed
  `threshold`, keeps the one with the highest `observation_count` (ties broken by earliest
  `recorded_at` for determinism), folds the others' `sources` and `observation_count` into
  it, and invalidates them with `invalidated_by` pointing at the survivor.
- `promote` turns a repeatedly-observed `EPISODIC` claim into a `SEMANTIC` one: seeing
  something happen once is an event, seeing it `min_observations` times is a pattern.
  The promoted claim gets `derivation=Derivation.CONSOLIDATION`.
- `run` executes all three and returns the per-stage counts. `now` defaults to the wall
  clock, read once for the whole pass. Pass it to evaluate two passes at the same instant:
  the decay target depends on that instant, so a claim sitting within a pass-duration of a
  rounding boundary otherwise changes on the second call.

All counts are "number of claims affected". These run off the write path.

---

## `memvara/llm/anthropic.py`

```python
class AnthropicLLM:
    name: str                    # e.g. "anthropic/claude-opus-5"
    def __init__(self, model: str = "claude-opus-5", client=None,
                 effort: str = "low", max_tokens: int = 8192) -> None
    def extract(self, episodes, known_predicates) -> list[dict]
    def resolve_predicate(self, surface: str, candidates: Sequence[str]) -> dict
    def classify_predicate(self, predicate: str, example: str) -> dict  # legacy fallback
```

Hard API requirements — these are current and getting them wrong is a 400:

- Structured output goes in `output_config={"format": {"type": "json_schema", "schema": ...}}`.
  The top-level `output_format` parameter is deprecated; do not use it.
- **Never** pass `temperature`, `top_p`, or `top_k` — they are rejected on this model.
- Control depth with `output_config={"effort": "low"}` alongside `format`. Leave adaptive
  thinking on (the default); do not pass `thinking={"type": "disabled"}`.
- Import `anthropic` lazily inside `__init__` so `import memvara` works without it, and
  raise a clear install hint if it is missing.
- Use `CLAIM_SCHEMA` / `PREDICATE_SCHEMA` / `EXTRACT_SYSTEM` / `PREDICATE_SYSTEM` from
  `llm/base.py` rather than redefining them.
- Validate and coerce the model's output before returning: drop claims with a missing or
  out-of-range `source_index`, clamp `confidence` to `[0, 1]`, and normalize predicates to
  snake_case. The engine trusts these dicts, so this is the trust boundary.

---

## `plugin/hooks/` — the client-side tree

Not part of the package. `pyproject.toml` sweeps `packages = ["memvara"]` into the wheel,
so this lives at the top level: in the sdist, out of the wheel, and at the same path the
plugin repositories vendor it to, which makes syncing a subtree copy with no rewriting and
the drift guard a plain byte compare.

**Every host difference is data, until it is a difference in kind.** A client is one
`Host` record in `hosts/<id>.py` — event names, the stdin keys each field may arrive
under, reply keys, timeouts, config paths, an `ApproveSpec`, an `ExtractorSpec`. The four
bodies (`recall`, `session_start`, `capture`, `approve`) read the record and never a
client. `run.py <hook> --host <id>` binds it before importing a body, because
`lib/transcript.py` resolves the bound host's noise markers at import time.

**The record says what a host CANNOT do, not only what it does.** A canonical hook absent
from `events` is a hook that client has no event for. `context_key = ""` is a host with no
per-turn injection channel; `status_key = ""` is one with no operator-visible line, and
the renderer then DROPS that half of a reply rather than addressing a key nobody reads.
`transcript = None` is a host where capture cannot run at all. An absent key is the one
spelling that cannot be mistaken for a working default.

**Three envelope shapes, measured, none inferable from the others.** Claude Code and Codex
read `{"hookSpecificOutput": {"hookEventName": ..., "additionalContext": ...}}`; Copilot
reads the same keys flat at the top level and ignores the nested form; Cursor reads flat
`additional_context`. A port that ships the wrong one installs cleanly, runs, logs
success, and delivers nothing.

**Three transcript shapes too, and one of them keeps two copies of the prompt.** Claude
Code and Cursor write `message.content` blocks and differ only in whether the speaker sits
under `type` or `role`, which `TranscriptSpec.role_key` covers. Codex writes
`response_item` payloads and Copilot writes `{"type": "user.message", "data": {...}}`;
each needs a reader, dispatched on `TranscriptSpec.format`. Copilot's is the one with a
hazard in it: `data.content` is what the person typed and `data.transformedContent` is
what the model saw — the same text plus the host's own markup *and this plugin's injected
recall*. The reader mines `content`, so our own output can never be read back in and
re-stored; the echo filter reads `transformedContent`, because it still has to know what
was shown. That split is why `Host.noise` is empty for Copilot and is not an omission.

**Capture must not hold a turn open, and how it avoids that is per host.**
`supports_async` says the client honours `async: true` on the registration.
`detach_capture` says the hook must fork itself — `run.py` re-execs into a new session and
returns. They are separate fields because Codex accepts the async flag and then does not
run the hook at all, and guessing either wrong fails in opposite directions: losing the
hook, or holding the turn for the whole 12–14 second extraction.

**Extraction never recurses, and the guard is one line in one place.**
`lib/extract.py` refuses to run when `MEMVARA_CAPTURE_ACTIVE` is set, and sets it on the
child it spawns. This matters more now that a host mines with its own CLI: `codex exec`
inside a Codex Stop hook starts a session that fires the same hooks. The read hooks stand
down on the same sentinel; capture's own extraction is what refuses.

**Extraction is a chain, and a host must not pin a model.** `ExtractorSpec` names the CLI
that mines a turn: the host's own first, `claude -p` second, and then a logged failure that
raises the capture alert. There is deliberately no third rung handing the prose to the
server — `memory_add` on an `MEMVARA_LLM=none` deployment would accept it and store
nothing while logging success, which is the shape of every defect in this repository's
history.

A host CLI declares no `--model`. The point of mining with the host's own is that the user
already configured and authenticated it; naming a model inside a hook nobody read can name
one their account cannot reach, and capture then fails on every turn for a reason only the
log shows. `ExtractorSpec.model` is empty for such a CLI, and the label recorded in
`usage.jsonl` follows the rung that actually answered — a hardcoded label would account a
Codex extraction against the model `claude -p` pins, which is wrong in the one file whose
whole job is to say what was spent.

**A hook may never fail a turn.** Every path out of `run.py` returns 0, including the ones
it does not know about — the `__main__` block catches `BaseException`. That is the rule
that outranks reporting a problem: a hook that fails a prompt is worse than a hook that
does nothing. The obligation to say something moves to `~/.memvara/.hooks/`, where every
path that reaches a decision writes a line, including the ones that decide to do nothing.

## Testing requirements

- `pytest`, no network, no API key, no sleeping. Use `SQLiteStore(":memory:")`,
  `HashingEmbedder`, and `NullLLM` or a local fake.
- Control time by passing explicit `datetime` values (`utcnow()` +/- `timedelta`) rather
  than patching the clock.
- Test behavior through the public surface, and cover the failure modes, not just the
  happy path: empty input, unicode, adversarial FTS5 query strings (`"a AND (b"`),
  concurrent writes, dimension mismatches, claims with no embedding.
- A fake LLM must assert on *call count*, not just output — the whole design claim is that
  the LLM is called rarely, so a test that does not count calls does not test the design.

---

Previous: [How it works](DESIGN.md) · Next: [Contributing](../CONTRIBUTING.md) · [Roadmap](ROADMAP.md)
