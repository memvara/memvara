# Memvara internals — module contracts

This file is the interface contract between subsystems. `core.py` wires them together
against exactly these signatures, so treat them as fixed. Everything here is already
importable from the foundation modules:

- `memvara/types.py` — `Claim`, `Episode`, `Scope`, `Result`, `Explanation`, `WriteReceipt`,
  `MemoryType`, `Derivation`, `utcnow()`, `content_hash()`, and for documents `Document`,
  `DocumentChunk`, `DocumentStatus`, `DeleteResult`, `Page`
- `memvara/documents/` — `split()` and `normalise()`, the retrieval chunker, and
  `DocumentService`, which the document methods on `Memvara` delegate to
- `memvara/compat/supermemory_import.py` — `import_supermemory`, `SupermemoryReceipt`
- `memvara/schema.py` — `PredicateRegistry`, `PredicateSpec`, `Cardinality`, `Volatility`
- `memvara/store/` — `Store` and `SQLStore` protocols, `SQLiteStore`, `STATES`,
  `ClaimState`, `resolve_states()`, `state_predicate()`, `stored_state_predicate()`,
  `live_predicate()`, `unexpired_predicate()`
- `memvara/embed/` — `Embedder` protocol, `HashingEmbedder`, `CachedEmbedder`, `default_embedder()`,
  `encode_queries()`, which embeds a search query in the form its model expects, and
  `calibration_of()`, the cosine thresholds measured for each embedding space
- `memvara/llm/base.py` — `LLM` protocol, `NullLLM`, `CLAIM_SCHEMA`, `RESOLVE_SCHEMA`,
  `PREDICATE_SCHEMA`, `EXTRACT_SYSTEM`, `RESOLVE_SYSTEM`, `PREDICATE_SYSTEM`,
  `MAX_CLAIMS`, `bounded_claim_schema()`, and for agentic extraction the `ToolChat`
  protocol with `Message`, `ToolSpec`, `ToolRun`, `ToolRunError`, `ToolRunTimeout`,
  `MalformedToolOutput` and `TOOL_STEP_MAX_TOKENS`
- `memvara/write/agentic.py` — `AgenticExtractor`, `AGENTIC_SYSTEM`, the proposal types and
  `ProposalPlan`

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

1. **Contradiction resolution is decided by the deterministic reconciler, and a model is
   reached only through named stages.**

   > **Claim.** Deterministic stages (deduplication, contradiction resolution, ranking,
   > decay, time travel) never call a model. Contradiction resolution is decided by the
   > deterministic reconciler; a model may *propose* changes on the write path, and every
   > applied change is one of the reconciler's recorded outcomes. The read path may call a
   > model only through the named stages `ranked`, `query_rewrite` and `synthesis`, each
   > with a recorded outcome and a model-free fallback.
   > **Scope.** The library. On the write path, a model is reached by `extract()`,
   > `resolve_predicate()` and, when `agentic_extraction` is on and the backend implements
   > `llm.ToolChat`, by `run_tools()` in `memvara.write.agentic`. That loop's tools read
   > the store and record proposals; none of them writes. Its proposed memories pass the
   > same pollution guard, closed-vocabulary filter, predicate acquisition and grounding
   > check as `extract()` output and then `Reconciler.apply`; a proposed end becomes a
   > retraction through `Reconciler.apply(close="ended")`; a proposed link becomes a
   > `claim_links` row; and a proposal naming a memory the model did not read in that run,
   > or asking to end or replace one in a broader scope than the write, is refused and
   > recorded as a `RefusedProposal`. A
   > model can therefore choose *which* candidates the reconciler sees, and cannot choose
   > what the reconciler does with them. On the read path, a reranker is a
   > cross-encoder rather than a generative model, and it is off by default. The three
   > model stages are: `search(ranked=True)` against a retriever configured with a
   > `read_selector` (`memvara.select`), one chat call per read over the turns of the
   > role the question asks about (`retrieve.intent.routed_role`, model-free;
   > `read_route_roles=False` hands it both roles); `query_rewrite`, one chat call before
   > retrieval that returns up to three other phrasings and an optional date range; and
   > `synthesis`, one chat call after `recall(synthesize=True)` has rendered its notes.
   > Each runs on the caller's own chat backend with a 10-second deadline, records its
   > outcome (`applied`, `fallback`, `key_rejected`, `disabled`, `unconfigured`) on the
   > result, and serves the plain read on every outcome but `applied`. The model's
   > answer never changes what is stored and never reaches a deterministic stage: a
   > rewrite chooses which queries and which `valid_at` the ordinary pipeline runs with,
   > and a synthesis is text placed above notes that are still returned in full.
   > **Sketch.** `NullLLM` is the default `llm=` and cannot chat, so the shipped
   > configuration has no model to call on either path. `Reconciler` and `Consolidator`
   > take no `llm` parameter at all. `AgenticExtractor.run` returns a list of proposals and
   > writes nothing; `WritePipeline._reconcile` and `_apply_proposals` apply them inside
   > the claim transaction, and every change they make comes back as a `ReconcileResult`
   > action or a link row. `agentic_extraction` is off by default. `HybridRetriever` takes a model only as `selector=`
   > and `rewriter=`, both `None` by default; `Memvara` builds a `QueryRewriter` and a
   > `Synthesizer` (`memvara.select.stages`) only when its `llm=` implements `Chat`.
   > `query_rewrite=False` and `synthesis=False` on the constructor are the switches,
   > and `search(query_rewrite=False)` is the per-call opt-out; `memvara.select.PLAIN_READ`
   > spells it for the reads inside this repository that must stay deterministic.
   > **Measured.** `bench/mem0_real.py`: 2 write-path LLM calls against mem0's 105 on the
   > same 105-turn transcript, and **identical final state on every run** where mem0's
   > differs. `tests/test_packaging.py::test_nothing_but_numpy_is_imported_while_the_
   > package_is_being_imported` holds the import side.
   > `tests/test_read_stages.py::test_the_default_read_path_makes_no_model_call_without_a_chat_backend`
   > holds the read side: with a backend that cannot chat, `search()` and
   > `recall(synthesize=True)` make zero model calls.
   > `tests/test_read_stages.py::test_a_model_is_reached_only_from_the_places_invariant_1_names`
   > lists every call in the package that can reach a model and fails on a new one, and
   > `test_every_read_in_this_repository_says_whether_it_may_call_a_model` fails on a
   > `search()` or `recall()` call that does not say whether it may rewrite. No
   > measurement of answer quality with the two new stages exists yet.
   > `tests/test_agentic_extraction.py` holds the write side of the agentic loop: a
   > proposal naming an unread memory, or closing a broader-scope one, changes nothing,
   > a replacement
   > the reconciler does not accept leaves the named memory live, and a turn that quotes
   > the extractor's instructions yields no memory that restates them. The
   > identical-final-state result above was measured on the single-call path. With
   > `agentic_extraction` on, which candidates reach the reconciler depends on the model's
   > proposals, so two runs over the same turns end in the same state only when the model
   > proposes the same things. No measurement of agentic extraction's claim counts,
   > duplicates or answer accuracy exists yet; its release bar is in the "Reversed" list
   > of `docs/ROADMAP.md`.

   **What changed on 2026-09-23.** Until then this invariant said that nothing on the read
   path calls a model unless the caller opts in, and `ranked=True` was the one opt-in.
   The phase 2 parity design
   (`docs/superpowers/specs/2026-09-23-parity-phase-2-documents-and-retrieval-design.md`,
   §4.5) added query rewrite and synthesis, and made query rewrite **on by default**
   whenever the configured `llm=` can chat. So a caller who configured an extraction
   model now pays one read-path model call per `search()` and `recall()` unless they
   pass `query_rewrite=False`. What did not change is the part the measurement above
   stands on: the stages that decide what is stored, what contradicts what, and how
   results are ordered are still pure functions of stored state, and a store opened
   with the default `NullLLM` still makes no model call anywhere.

   **What changed on 2026-09-24.** Until then this invariant said that contradiction
   resolution is a pure function of stored state, and that on the write path only
   `extract()` and `resolve_predicate()` may touch a model. The phase 3 parity design
   (`docs/superpowers/specs/2026-09-23-parity-phase-3-extraction-and-cloud-design.md`,
   §3.1) added agentic extraction, in which a model reads the store through tools and
   proposes ends, replacements and links as well as new memories. What a model proposes
   now decides which candidates the reconciler is shown, so the write path's input is no
   longer only the turns and what is stored. What did not change: the reconciler still
   takes no model, still decides every duplicate, conflict and supersession by the same
   rules, and every change it applies is one of its recorded outcomes (`add`,
   `reinforce`, `supersede`, `retract`, `noop`). A model still cannot retire or erase
   anything, because a proposed end is a retraction with `close="ended"`. The switch
   ships off by default. The old wording and the reason it was reversed are in the
   "Reversed" list of `docs/ROADMAP.md`.

2. **Unknown predicates default to `Cardinality.MANY`.** Wrongly retiring a true fact is
   worse than keeping two competing ones. The default is deliberate and stays; what
   `MEMVARA_PREDICATES` adds is a way to *revise* it, since before it a server-backed
   store had none. A declared spec outranks a persisted learned one — rehydration skips
   any learned spec whose name a declaration already holds, so a pack corrects a store
   that guessed rather than only describing a fresh one. Forward-only: it changes what
   supersedes on the next write and retires nothing already stored.

   A vocabulary is TOML, one `[[predicate]]` table each, `name`, `cardinality` and
   `volatility` required, everything else optional:

   ```toml
   [[predicate]]
   name = "git_state"
   cardinality = "one"     # "one" supersedes, "many" accumulates
   volatility = "fast"     # static | slow | fast -> 36500 | 730 | 7 day half-life
   aliases = ["git_status"]

   [[predicate]]
   name = "depends_on"
   cardinality = "many"
   volatility = "slow"
   subject_type = ["project", "software"]   # what may hold this relation
   object_type = ["software", "service"]    # what its objects are
   graph = true                             # this relation is worth walking
   inverse = "depended_on_by"
   inverse_cardinality = "many"             # not the same as this predicate's own
   traversal_cost = 1.0                     # edge weight; nothing consumes it yet
   ```

   The second block is the **graph declaration**, and its six fields are declaration-only:
   nothing infers them, and a learned predicate leaves them at defaults that mean "takes
   values, walks nowhere". `object_type` is what classifies an object as an entity or a
   scalar, so `PredicateSpec.objects_are_entities` is False for any predicate nobody has
   declared — connectivity is opt-in. The reserved type `value` names a scalar, and a
   declaration that mixes it with an entity type resolves to a value, because such a
   predicate cannot decide per claim and the undecidable case takes the safe direction.

   `inverse_cardinality` is declared beside `inverse` rather than assumed, because the two
   sides are not symmetric: `owned_by` holds one value and `owns` holds many. A walk that
   assumed the forward cardinality would treat several true facts as competing answers to
   one question and end all but the last, so the loader refuses one without the other.

   Needs Python 3.11 or later, which is where `tomllib` arrives; the reader is
   imported lazily so 3.10 keeps working for everything else.

   Malformed entries raise rather than being skipped: a vocabulary that half-loads leaves
   some predicates superseding and others accumulating with nothing recording which. For
   the same reason an **unrecognised key is refused** rather than ignored. A pack is read
   once, at startup, by nobody; `graph_traversable = true` instead of `graph = true` would
   leave the predicate non-traversable, the store with no edges, and nothing anywhere
   saying why. `graph = true` without an `object_type`, with a `value` object type, or a
   `traversal_cost` at or below zero are refused on the same grounds.

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

3. **The engine hard-deletes only claims that carry an explicit `expires_at`, only after
   it passes, and always with a proof record; ending and superseding never delete. And
   end-of-life moves exactly one clock.** Closing valid time (`valid_to`) says *the world
   changed*; closing transaction time (`invalidated_at`) says *the record was wrong*. They
   are different events and no write may assert both. Superseding a claim therefore sets
   `valid_to` and `invalidated_by` and leaves `invalidated_at` unset — the old value
   stopped being true, and we were never mistaken about it. `Claim.state` names the
   outcome: `live`, `ended`, `retired`. History must stay queryable via `known_at` **and**
   `valid_at`; the second of those returned nothing on any history the engine wrote itself
   for as long as supersession closed both clocks. The correcting reading is reachable,
   never guessed: `close="retired"` on `remember`, `supersede`, `forget` and `delete`,
   defaulting to `"ended"` everywhere except `forget`/`delete`, which are belief operations
   by name.

   **Expiry is not `valid_to`.** A caller who writes a fact with `expires_at` asks for it to
   be **erased** once that instant passes: the row, its text index entry and its vector are
   deleted, the erasure is recorded in `erasures`, and `prove_erased` checks the disk. That
   is a third ending, beside ended and retired, and it is the only one the engine carries
   out by itself. It cannot be keyed on `valid_to`, because the engine also sets `valid_to`
   when it ends or supersedes a claim, and erasing on it would erase history. A claim
   without an `expires_at` is never erased by the engine, however old it is and whatever
   its `valid_to` says.

   > **Claim.** No engine write deletes a row except `erase_expired`, which deletes only
   > claims whose explicit `expires_at` is at or before the sweep's instant, and records
   > an erasure row and an `ErasureProof` for each. No write closes both clocks.
   > **Scope.** The *engine*. `erase()`, `purge()` and `reset()` delete, on purpose and by
   > name, and they are the caller's decision rather than the engine's — see invariant 8's
   > neighbour below and `Memvara.prove_erased`. `delete_document()` is the fourth: it
   > erases one document's text, and it deletes no claim row, because a claim whose only
   > source was the document is retired (see *Documents* under `memvara/store/`). A
   > re-ingest by `custom_id` erases the text of the chunks the new version no longer
   > has, under the same rule. The expiry sweep runs when a `Memvara` opens a store and
   > hourly in the MCP server, and not at all on a read-only server. Reads do not wait for
   > it: from the instant `expires_at` passes the claim is left out of every read, in
   > `SQLiteStore._state_clause` for the searched ones and in `Memvara._gone` for the ones
   > addressed by id, so the sweep only deletes. `expiry_erasure=False`
   > (`MEMVARA_FEATURE_EXPIRY_ERASURE=0`) stops the sweeps and the hiding and leaves
   > `expires_at` stored. The sweep keeps a claim's source turns, as `erase()` does by
   > default.
   > **Sketch.** `close_out` is the single place any claim ends and takes one `Closure`;
   > `Claim.state` derives `live`/`ended`/`retired` from which column is set.
   > `Memvara.erase_expired` lists due claims with `Store.expired_claims`, which reads
   > `expires_at` and never `valid_to`, re-reads each one, and erases it through
   > `_erase_proved`, the same code `erase()` runs after its scope check.
   > **Measured.** `bench/compare.py`: **0 stale values left live** against 7 for a
   > mem0-style baseline, on a transcript where 10 facts are revised. `tests/
   > test_bitemporal.py` holds the two-clock reads. The expiry side is not measured; there
   > is no number to produce. `tests/test_expiry.py::test_ended_superseded_and_retired_
   > claims_are_kept_however_old_they_are` holds that ended, superseded and retired claims
   > survive a sweep dated a century ahead, and `test_nothing_is_erased_before_the_instant_
   > and_everything_due_is_erased_after` holds the instant and the proof.

   **What changed on 2026-09-24.** Until then this invariant said that nothing is ever
   hard-deleted by the engine. The phase 3 parity design
   (`docs/superpowers/specs/2026-09-23-parity-phase-3-extraction-and-cloud-design.md`,
   §3.4) added erasure on expiry, on by default with a switch, and reversed the sentence
   for this one case. What did not change: ending and superseding still keep every row,
   the two clocks still close separately, and the measurement above still stands, because
   a sweep erases nothing that was written without an `expires_at`.

4. **Every claim carries provenance.**

   > **Claim.** `sources` holds the episode ids the claim came from, and `derivation`
   > reflects how it was produced.
   > **Scope.** Claims the engine writes. A `Claim` a caller constructs by hand and hands
   > to `remember()` carries what the caller put in it. Deleting a document removes its
   > erased episodes from `sources`, so a claim whose only source was the document ends
   > with none; it is retired in the same write, and its closure reason, "source document
   > deleted", is what `why()` shows in place of the source.
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
   > rather than a comprehension in the facade; the caller's metadata and file-path filter
   > reaches every capped store method as `where`, and
   > `tests/test_metadata_filters.py::test_a_filtered_search_returns_k_matches_when_many_
   > non_matching_rows_rank_higher` asserts it.
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
def expired_claims(self, now: datetime) -> list[Claim]       # Store, optional
def prove_erased(self, claim_id: str) -> ErasureProof        # Memvara
def erase_expired(self, now=None) -> list[ErasedClaim]       # Memvara
```

`erase()` reported success from `erase_claim`'s return code, which proves the code took
the branch it thought it took — the same statement the return value already made, and one
that cannot disagree with it. `residue` is a **live query**: five `SELECT COUNT(*)`s over
the tables a claim can survive in (`claims`, `claims_fts`, `embeddings`, `claim_sources`,
and since schema 13 `claim_links`). A re-hash of what was returned, or a cached count,
would not be evidence.

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

**Erasure on expiry uses the same path.** `erase_expired(now)` lists every claim whose
`expires_at` is at or before `now` with `Store.expired_claims`, in every tenant, re-reads
each one, and erases it with `_erase_proved`, which is the part of `erase()` after the
scope check: `erase_claim` (audit row, then delete, in one transaction) and then
`prove_erased`. Each erased claim comes back as an `ErasedClaim` with its id, scope,
`expires_at`, `expire_reason` and proof, and no subject, predicate, object or text. The
re-read is there because a write between the listing and the delete can move the expiry
later; that claim is left alone. A failed proof raises `ErasureIncomplete`, and the claims
erased before it stay erased and recorded. `erase_expired` runs when a `Memvara` opens a
store, unless `expiry_erasure=False` or `sweep_expired=False`, and hourly while the MCP
server's `serve()` loop runs (`server.mcp.ExpirySweeper`, on a daemon thread, which warns
and carries on when a sweep fails). A read-only server runs neither, because erasing is a
write. A store with no `expired_claims` is skipped at open, and so is one whose
`expired_claims` raises `NotImplementedError`, as `RemoteStore`'s does: the hosted
deployment runs its own sweep. Called by name, `erase_expired` raises in both cases.

**Reads do not wait for the sweep.** From the instant a claim's `expires_at` passes, no
read returns it. `SQLiteStore._state_clause`, the clause every limited claim query runs,
ANDs on `base.unexpired_predicate`, `expires_at IS NULL OR expires_at > <wall clock
now>`, whatever instants the read
asked about, so `search`, `recall`, `get_all`, `count`, the graph leg and the slot lookups
of the write path all leave it out and `k` still counts visible rows (invariant 7).
`Memvara._gone` does the same for the reads addressed by id: `get`, `why`, `history`,
`produced`, and the claims `forget_matching` and `links` hydrate. The reconciler does not
reinforce such a claim, so writing the fact again stores a new claim rather than one the
sweep is about to erase. `erase()` by name still finds and erases it. With
`expiry_erasure=False` the store's `hide_expired` is false and an expiry does nothing.

**A repeat with an expiry stays in its own scope.** Writing a fact the store already
holds normally reinforces the claim on record, which `value_key` finds by owner (tenant
and user) and not by project, agent or session. A repeat that names an `expires_at`
reinforces only a claim in exactly its own scope, and puts the expiry on it. With no such
claim, it is written as its own claim in its own scope, so the expiry never reaches a
claim another project or session relies on, and the claim beside it is not reported as an
accumulation. Refusing the write was the other choice; it was not taken because it would
lose the fact the caller asked to keep until the expiry, and its message would tell one
project what another project holds.
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

   **`Memvara.supersede` refuses to close a claim twice, except for an exact replay.** A
   named claim that is already retired, or already ended when `close="ended"` asks, is a
   `ValueError` and nothing is written. The exception, `Memvara._replayed`, is the same
   supersession again: the claim's `state` is the closure asked for, and the claim that
   closed it has the same `value_key`, the same project and the same `valid_from` as the
   new one. Then the call writes nothing and returns a receipt naming that successor under
   `reinforced`, with no salience change, so importing a mutation log twice completes.
   `remember(replaces=...)` goes through `supersede`, so it behaves the same way and so
   does `memory_remember`. The same value from another date, or in another project, is
   refused: the first is a correction of when the value began, and the second is a write
   the other project never received.

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
    retyped: Retype | None       # a claim filed under a different memory_type than it
                                 # arrived with: an asserted type on a known claim, or
                                 # procedural refused for a subject other than the user
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

**`procedural` is for the subject `user`, or a scope spelled `project:<key>`, and the
reconciler enforces it.**
`Reconciler.file_by_subject` runs on every candidate right after `_canonicalize`, and on
the claim on record when a candidate turns out to be a re-observation. A `procedural` claim
whose subject is not the user is filed as `semantic`, whoever supplied the type — a caller,
a model, or a predicate's declared default — and the result carries a `Retype` with
`reason="subject"`, so the receipt says what happened rather than silently filing
elsewhere. The reason it is a rule and not prompt guidance: `procedural` is the population
`memory_standing` returns and clients inject at the top of every session, and a repository
or a service cannot want anything, so a `procedural` claim about one is wrong however it
was produced. A claim already misfiled is moved the next time the same triple is seen,
even by a write that asserts no type; the safety property that an unopinionated write
cannot undo a correction still holds, because moving such a claim out of `procedural` is
never a correction anyone could have wanted to keep. A subject beginning `project:`
(`types.PROJECT_SUBJECT_TYPE`, read through `Claim.subject_type`) is exempt: it is a scope, not a thing, and
a preference scoped to one checkout is still how the user wants work done there — the
plugin's session-start block reads `project:<cwd>` beside `user`. A verbatim note is exempt — it is on
the `note` predicate (`types.NOTE_PREDICATE`, written by `compat/_notes.py` for the mem0
shim and the importer alike), and a note typed `procedural` is the owner's own standing
instruction in the owner's words, not a claim about a thing. The one write that does not
pass through `apply` — a restated turn, which `write/pipeline.py` reinforces directly —
calls `Reconciler.file_by_subject` on the claim it restates, so it heals and reports the
same way.

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

**Entity keys are bounded at 512 characters.** `subject_key`, `object_key` and the entity
id are all built from `entity_key`, and `entity_key` never returns more than
`ENTITY_KEY_MAX` (512) characters. A value longer than that, which in practice is a
pasted sentence or paragraph used as an object, is folded to the words that fit plus a
16-character digest of the whole folded key. The digest keeps two long values with the
same opening apart, and the result folds to itself: folding the bounded key again returns
the same key, which `fact_key_for` and `default_entity` depend on. The bound exists
because the hosted store keeps the entity id, `subject_key` and `object_key` in Postgres
btree indexes, which refuse a row over 2704 bytes, and a 3 KB value failed the whole write
there on 2026-09-06. A key that fits under the bound is returned unchanged, so every
entity id already inside a `fact_key` on disk stays what it was. Only values whose key
was longer than 512 characters get a new identity, and a store that holds such a value
from before this change will treat its next write as a new value rather than a
re-observation of the old row. Two other places follow the bound. `split_entity` folds
its `"{base} split {stamp}"` marker before storing it, so a split of an entity at the
bound stays under it. And `retrieve/anchor.py` compares a question against the words of
a key with the digest left out (`entities.key_words`), because a question repeats a
value by its words and never by the digest.

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
                 near_dup_threshold: float | None = None,
                 reinforce_bump: float = 0.25,
                 reject_ungrounded: bool | str = "auto",
                 closed_vocabulary: bool = False,
                 extraction_chunks: bool = False,
                 agentic_extraction: bool = False,
                 guidance: Guidance | None = None) -> None

    def add(self, episodes: Sequence[Episode]) -> WriteReceipt
    def reextract(self, episodes: Sequence[Episode]) -> WriteReceipt
    def assert_claim(self, claim: Claim) -> WriteReceipt
```

`guidance` is per-project extraction guidance (`memvara.llm.guidance`), reached from
`Memvara(write_guidance=...)`. Every tier-2 call passes it to `llm.extract(guidance=...)`,
which appends it to the system message, and an extractor that builds its own system
message reads it from `WritePipeline.guidance` and appends it with `with_guidance`. An
empty `Guidance` is stored as `None`. A non-empty one is refused with `TypeError` for a
backend whose `accepts_guidance` is not true, because an older backend's `extract` has no
such argument and the guidance would otherwise reach no extraction. Without guidance the
argument is not passed at all, so such a backend keeps working.

`add()` runs the tiers in order and must populate every field of `WriteReceipt`,
including `llm_calls` (0 whenever the LLM is not consulted) and `latency_ms`. One field
the pipeline never fills is `may_replace`: `Memvara.remember` fills it after the write,
and only when the instance was built with `advise_replacements=True`, the backend
implements `llm.ReplacementJudge`, and the write added a claim without closing one. It
asks the judge about up to `ADVISORY_CANDIDATES` of the nearest live claims in *other*
slots, counts each consultation in `llm_calls`, and closes nothing. A judge that raises
warns once per instance and leaves the list empty; the claim is already durable and a
suggestion must not turn it into an exception the caller retries.

- **Tier 0 (no LLM):** store the episode; skip content-hash duplicates
  (`store.find_episode_by_hash`). For surviving episodes, embed and find the nearest live
  claim. A cosine at or above `near_dup_threshold` reinforces that claim instead of
  extracting from the turn, unless the turn and the claim's text hold different numbers
  (`embed/calibration.numbers`, the rule the merge also applies). `None`, the default,
  reads the merge's threshold for the embedder's space from `embed/calibration.py`,
  because a turn worded like a claim embeds exactly as that claim would. Over the 69 pairs
  of different values in `bench/embedder_calibration.py`, a flat 0.97 would read the
  second of a pair as a restatement of the first 22 times under MiniLM and 19 times under
  bge-small, losing the value each time; the check now reads none of them that way. The
  bench's 16 first-person turns score at most 0.944 against their claim in all three
  spaces, so on them the check never fires: what it skips are turns worded like a claim,
  such as an exact repeat of one.
- **Tier 1 (no LLM):** `SalienceGate` drops turns carrying no durable fact
  (count them in `receipt.skipped`), then `FastExtractor` handles what it can.
- **Tier 2 (LLM):** only the turns that survived both and produced no fast-path claim are
  batched into a single `llm.extract(...)` call. Map `source_index` back to the
  originating episode for provenance. Unknown predicates trigger one
  `llm.resolve_predicate(...)` per *new surface form*, cached via `registry.learn_alias`
  / `registry.learn` and persisted through `store.put_spec(spec, tenant)` so it is never
  asked again — including after a restart, and including by another process. With
  `agentic_extraction` on, the single call is replaced by a tool loop; see the
  `agentic_extraction` entry below.

  `reject_ungrounded` guards this tier's output, defaulting to `"auto"`: a proposed
  claim whose object shares not one content word with the episode it cites is a
  fabrication candidate, and the embedder then gets a veto — kept if the best
  chunk-cosine against the source reaches the embedder's `grounding_rescue` threshold
  (`embed/calibration.py`: 0.40 as measured under MiniLM, 0.65 for bge-small-en-v1.5,
  whose cosines run higher; that module carries the distributions), refused and
  counted on `receipt.ungrounded` otherwise. `True` is the lexical check alone; `False` is off.
  Only model-proposed claims are ever checked — `remember()` and the fast path do not
  pass through `_claim_from_dict` — and the reason the default is on rather than off is
  that the destructive direction is storing: a fabricated value in a ONE-cardinality
  slot supersedes and ends the true fact that was there. It remains a precision filter
  for wholesale fabrication only — a claim that reuses real vocabulary with an inverted
  or misattributed meaning passes clean — and an embedder failure during the rescue
  fails open, keeping the claim and warning once. Under the default `HashingEmbedder`
  nothing is ever rescued (n-gram cosines on zero-overlap pairs measure 0.0–0.11,
  far under the floor), so `"auto"` degrades to the strict check there.
- **`reject_polluted`** (default `True`) is the other guard, and it catches what
  `reject_ungrounded` says it cannot: a real value under a slot it does not belong to.
  `write/pollution.py` carries the rules and the measurement. Within one turn, one
  (subject, object) under several predicates keeps every known predicate and drops the
  unknown ones beside them — the measured small-model failure, a found value forced into
  every available slot; a place predicate with a digit or URL in the object is refused, on
  any subject. Deliberately no rule about the subject: a speaker predicate on a named third
  party is that party's own slot, and a first draft that refused it was measured to catch
  nothing the place rule did not. A further rule stores a claim under a novel
  predicate, or a ONE-slot builtin (less `born_on` and `timezone`) with a digit or URL in
  it, at `min(confidence, 0.4)`, so the reconciler's half rule stores it beside the
  incumbent rather than ending it. Runs on `_tier2`'s raw output **before** predicate
  acquisition, so a refused duplicate's spelling is never acquired. Counted on
  `receipt.polluted`; a turn whose only claims were refused also counts on `unextracted`.
  Held to its measurement by `tests/test_pollution.py` over the 255-claim fixture:
  wrong-predicate 46 → 20, duplicates 32 → 0, keyed facts found unchanged at 60/90 in
  every configuration.
- **`closed_vocabulary`** (default `False`) refuses a model-proposed claim whose predicate
  the registry does not know — `registry.resolve(...)` not resolved, so a declared alias
  passes and a spelling nothing declared does not — and counts it on `receipt.unregistered`.
  It runs after the pollution guard and before acquisition, so a refused predicate is never
  learned and costs no model call. `remember()` and the fast path never reach it. The
  measurement behind it: a worker whose prompt let the model name relations itself wrote
  2,555 claims under about a hundred invented predicates in one afternoon, every one
  unregistered, multi-valued and retiring nothing, and recall for a short query about a
  repository's CI returned five commit hashes ahead of the note that answered it.
- **`extraction_chunks`** (default `False`) extracts a long turn in pieces. A turn longer
  than `write/split.EXTRACTION_CHUNK_CHARS` (6,000 characters) is cut by
  `split_for_extraction()` into pieces of at most that size. Whole paragraphs go into a
  piece where they fit, a longer paragraph is cut after a sentence or at a line break, and
  only a single sentence longer than a piece is cut mid-sentence. Turns under the limit
  still share the batch's one call, and each piece of a long turn is a call of its own, so
  `llm_calls` counts one per piece. A piece is a copy of the episode with the same id, and
  `source_index` from a piece's call is mapped back to the long turn's position in the
  batch, so every claim cites the whole episode and `why()` shows what the user wrote. An
  index that names no turn in its call is dropped, not moved. Nothing merges repeats
  across pieces: a fact stated in two pieces reaches `Reconciler.apply` twice and is
  stored once and reinforced, exactly as when one call states it twice. The calls run one
  after another because they share one `Usage` accumulator. A turn is extracted whole or
  not at all: when a call carrying a turn fails, that turn's later pieces are not sent,
  what its earlier pieces returned is dropped, and the turn is deferred, because
  `reextract()` skips a turn that has claims and would never read the missing piece. The
  other turns of the batch keep their claims; only when every turn failed is the batch
  reported as a failed extraction. A turn of only whitespace, which the splitter returns
  as no pieces, goes in the call for whole turns. The predicate vocabulary is built once
  per batch, not once per call. Off by default because its release bar is not met: see
  the "Reversed" list in `docs/ROADMAP.md` and `tests/test_extraction_chunks.py`. The MCP
  server turns it on with `MEMVARA_FEATURE_EXTRACTION_CHUNKS=1`; the feature is marked off
  by default in `FEATURE_DEFAULTS` in `server/config.py`, the one table of features and
  their defaults.
- **`agentic_extraction`** (default `False`) replaces tier 2's single `llm.extract()`
  call with `write/agentic.AgenticExtractor` when the backend implements `llm.ToolChat`.
  The rules go in the system message (`AGENTIC_SYSTEM`, with the project's extraction
  guidance appended by `llm.guidance.with_guidance` when there is some, exactly as the
  single call appends it); the turns go in one user message
  inside `<content>` tags, described as data, with any `<content` or `</content` inside a
  turn defused so a turn cannot close the wrapper. The model gets six tools:
  `search_memories(query, k)` (live memories this write's scope can see, lexical and vector
  hits fused by rank, `k` clamped to 1–20), `get_claim(claim_id)` (one memory in any state,
  the same answer for a missing id and another scope's id), `propose_claim`,
  `propose_end(claim_id, reason, source_index)`, `propose_supersede(claim_id, reason, …)`
  and `propose_link(from_ref, to_ref, relation)`. A proposed memory carries the fields of
  a single-call claim: `subject`, `predicate`, `object`, `source_index`, `confidence`,
  `memory_type`, `valid_from` (the turn's words, resolved by `write/when.py` like `when`),
  `amount` and `unit`, and `propose_claim` also takes `expires_at`: an ISO 8601 date or
  instant the turn names, read as UTC when it has no zone, refused as `invalid` when it
  is not in the future (the rule `remember()` applies), and passed to the reconciler on
  the claim, whose rule for a repeat that names an expiry then applies unchanged.
  `source_index` and `confidence` are not in the design's argument
  list and are required here, because provenance and the authority rule depend on them.
  At most `AGENTIC_MAX_STEPS` (12) answers, `TOOL_STEP_MAX_TOKENS` (8,192) output tokens
  per answer, one retry per answer, and a budget for the whole run of
  `AGENTIC_SYNC_TIMEOUT` (25 s) in `add()`, where a caller is waiting, or `AGENTIC_TIMEOUT`
  (180 s) in `reextract()`, which a background worker runs
  (`llm/_tools.run_loop`, shared by both backends). A proposal is refused at once, and
  recorded on `receipt.proposals_refused`, when it names a memory the model did not read
  in this run (`not_read`), asks to end or replace a memory whose scope is not exactly the
  write's scope (`broader_scope`: reads widen upward, so this is a user-wide memory seen
  from a project or a session, and closing it would close it everywhere, which the
  deterministic path never does from below), cannot be shaped into a claim or a link
  (`invalid`), or
  restates the instructions (`instruction_echo`, `write/agentic.echoes_instructions`).
  Accepted proposals do not write. Proposed memories go through the pollution guard, the
  closed vocabulary, acquisition and the grounding check like single-call output, then
  `Reconciler.apply`; a replacement passes its reason to `apply`, which records it on
  whatever the candidate closes, and a replacement whose new value closed nothing is
  reported as `not_applied` with the named memory left live. A proposed end becomes a
  retraction of exactly the named value through `Reconciler.apply(close="ended",
  reason=…)`, filed in the named memory's own scope and citing the turn the model named.
  A proposed link becomes a `claim_links` row when both sides name a stored claim. The
  batch falls back to the single call, and says why on `receipt.agentic_fallback`, when
  the backend is not a `ToolChat` (`unsupported`), the run times out (`timeout`), an answer
  cannot be used twice in a row (`malformed`), the model is still calling tools after 12
  answers (`step_limit`, and its proposals are discarded), a request fails twice
  (`error`), or the batch holds turns from two scopes (`mixed_scope`). Every request the
  run sent is billed in `llm_calls`, fallback or not. `extraction_chunks` applies to the
  single-call path only. Off by default because its release bar is not measured: see the
  "Reversed" list in `docs/ROADMAP.md`. The MCP server turns it on with
  `MEMVARA_FEATURE_AGENTIC_EXTRACTION=1`.

`reextract()` is `add()` with tier 0 removed, for turns already in the store: a
deployment that ran without a model, or a batch a provider failure left `deferred` — or
that `extraction_deferred=True` marked so on purpose, for exactly this sweep to find. Tier 1
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
- embed the query through `embed.encode_queries`, once per pass, so that both vector legs
  and every phrasing of a rewritten read compare a query's vector with the stored
  passages'. An embedder that embeds a query differently from a passage says so with an
  `encode_queries` method: `LocalEmbedder` puts the instruction bge's English models are
  trained on before the query. Every other embedder embeds it through `encode`, as before;
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

#### Two legs at once

Each stage, claims in `_gather` and turns in `_episodes`, hands its vector leg to a pool
thread (`_beside`) and runs its lexical leg, and on the turn side the time leg, on the
calling thread. The legs read the store independently, and on a large scope each spends
most of its time inside SQLite or in a matrix product, both of which release the GIL, so a
stage costs the longer of its legs rather than their sum. In `bench/scale.py`,
`search(k=12, include_episodes=True)` went from a median of 286 ms to 245 to 254 ms with
199,499 turns and 100,000 claims in one scope, and from 100 ms to 64 to 65 ms with 189,520
turns and few claims, and 50 searches on each store returned the same rows with the same
scores. `docs/BENCHMARKS.md` has why the first gains less than the legs alone suggest.

The query is embedded on the calling thread before the leg is handed over, and the pass's
vectors go with it, so the embedder is still called once per pass and only from the thread
that called `search()`. The leg runs on the calling thread instead, exactly where it ran
before, in two cases:

- the store does not answer `_parallel_reads()` with true. `SQLiteStore` answers false
  inside `batch()`, where the calling thread reads its own uncommitted rows through the
  writer's connection and another thread's connection cannot see them, and for a database
  with no file, whose one connection a second thread would only wait for. A third-party
  store has no such method and keeps its legs on one thread;
- every one of the `_LEG_THREADS` pool threads, three, is busy, so that a saturated pool
  costs a search what it cost before rather than a wait.

A leg that raises on its thread raises from the search when the stage collects it, and
frees its thread either way.

`_parallel_reads` is private and read with a guarded `getattr`, as `embed.fingerprint`
reads `_vec`. A public method would have to join `Store`, and a new protocol member stops
every existing backend from type-checking as one.

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

That rule still holds for this leg: `temporal.py` reads only the instant it is handed and
never looks at the words of the question. Since 2026-09-23 there is a model-backed way to
get an instant out of the words, and it sits in front of this leg rather than inside it.
`query_rewrite` (invariant 1, `memvara.select.stages`) asks a model for the date range a
question refers to, and `HybridRetriever.search` turns the range's last second into the
read's `valid_at`, which then anchors this leg like any caller's `valid_at`. A `valid_at`
or `as_of` the caller passed always wins over the model's range, and without a chat
backend nothing is parsed at all.

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
what keeps the match from becoming a token index. The count is the fewest predicates one
greedy pass needs to account for everything the question said: `works_at` and `job_title`'s alias `works_as`
both reduce to `work` once the prepositions are gone, so "what company does Ada work at"
is one relation said two ways, and a lookup, rather than a chain.

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

### `retrieve/excerpt.py`

```python
ELLIPSIS = "…"

def excerpt(text: str, query: str, limit: int) -> str
```

Which part of a long turn `recall()` shows. It returns `text` unchanged when it fits in
`limit`. Otherwise it splits the text into sentences, scores each by how many of the
query's content words it contains (both sides folded through `schema.word_stem`), and
returns the best sentence with as many neighbours as fit, trying the next sentence before
the previous one, with `ELLIPSIS` on each side where text was left out. A sentence longer
than `limit` on its own is cut around its first matching word.

It must:

- **never return more than `limit` characters**, whatever the text, the query or the
  limit. The limit is what stops a pasted stack trace from becoming the prompt;
  `tests/test_excerpt.py` checks it over a seeded sample of shapes;
- **return the old head cut when the query shares no word with the text**: the first
  `limit - 1` characters and `…`, byte for byte what `_safe_line(text, limit)` returns. A
  turn with nothing to aim at is not rendered differently from before;
- **be a pure function of its arguments**, so two renders of one search are identical.
  Ties between sentences go to the earlier one.

`bench/recall_window.py` measures what it changes on LongMemEval-S: whether the gold
answer's words reach the 4,000-character context, from the same search rendered both
ways. They did for 44.9% of the 470 answerable questions with the head cut and 50.4% with
the window; [BENCHMARKS.md](BENCHMARKS.md) has the breakdown by question type.

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
    RECALL_HEADER_AT: str         # the same, naming the day, for recall(valid_at=)
    RECALL_HISTORY_HEADER: str    # "No longer true — ..." in the first three words
    RECALL_EPISODE_HEADER: str    # says "said", not "true"
    RECALL_EPISODE_CHARS: int     # 280 — a pasted stack trace cannot become the prompt;
                                  # a longer turn shows the window the query names

    def _safe_line(self, text, limit=None) -> str
    def recall(self, query, *, k=8, min_score=0.0, header=None, ...,
               include_episodes=False, episode_header=None,
               include_history=False, history_header=None,
               budget=None, counter=_approx_tokens, valid_at=None,
               with_ids=False) -> str | RecallResult
    def _past_by_claim(self, claims, tenant=None, user=None, agent=None,
                       session=None, *, before=None) -> list[list[str]]
```

`recall()` must:

- **take an explicit signature, never `**kwargs`.** Forwarding arbitrary keywords into
  `search()` would expose `as_of`, `states` and `include_invalidated` here, and the latter
  two resurrect retired claims into a live prompt — an un-delete reachable by anyone who
  can influence a parameter. `states=["retired"]` is the sharper form: a prompt built from
  nothing but the records we stopped believing. `valid_at` is the one time keyword it
  takes, because it moves the world clock only and reaches no retired claim; the belief
  clock and audit reads stay on `search()`;
- flatten every rendered line through `_safe_line`, which collapses whitespace, strips
  leading list and heading markers, and maps `[`/`]` to their fullwidth forms
  (`_FORGEABLE`), so stored text cannot open its own bullet list, repeat a header, or
  finish a line with something that parses as the next result row. The first two defend
  the gaps *between* lines; the third defends the rest of the line a claim is already on,
  which ordering metadata-first cannot reach. `memvara/server/tools.py:safe_line` calls
  this method rather than reimplementing it — it was a copy once, the two sets drifted,
  and the same stored value was then neutralised differently depending on which surface
  replayed it. Episodes are additionally truncated to `RECALL_EPISODE_CHARS`, to the
  window of the turn that best matches the query (`retrieve/excerpt.py`), or to its first
  characters when the query shares no word with it;
- write each turn's day in front of it, `- [8 May 2023] …`, through `_day`, the formatter
  the dated header and the history tail use. The day is added after `_safe_line` has run,
  so its brackets are the renderer's own and stored text cannot forge them: every bracket
  inside a turn is already fullwidth. A turn is evidence about when only next to its date,
  and a turn saying "yesterday" is otherwise unanswerable;
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

## `memvara/project.py` — one project name per repository

```python
def canonical_project(cwd, *, run=None) -> str | None
def normalize_remote(url) -> str | None
def check_project(value) -> str          # raises ValueError with the rule it broke
```

`canonical_project` asks git two questions through `run`, which a test replaces: the
repository's common git directory (`rev-parse --path-format=absolute --git-common-dir`,
which needs git 2.31 or later), and the `origin` remote read through that directory
(`--git-dir <common> remote get-url origin`). Reading from the common directory is what
makes a linked worktree resolve to the same name as its main checkout. The remote is
normalised to `host/owner/repo`: whitespace, credentials, the query, the fragment, empty
segments, trailing slashes and one `.git` are dropped, the host is lower-cased, and a port
written in the URL is kept. The owner and repository are lower-cased only on `github.com`,
`gitlab.com` and `bitbucket.org`, where case does not distinguish repositories; elsewhere
folding could merge two projects. Percent-encoding is left as written.

The plugin hooks carry a copy of the normaliser in `plugin/hooks/lib/project.py`, because
they run without the library. The rules are pinned as data: `tests/fixtures/project_vectors.json`
is a byte-identical copy of the hooks' `project_vectors.json`, `tests/test_project.py` runs
the library against every row, and the hooks' tests run their copy against the same rows.
A change to either copy starts with a row in that file. The same file's `check_project`
rows pin which names the server accepts, which both copies use to decide when to fall
back to the path form. Git output that is not UTF-8, such as a remote holding the byte
`0xff`, is read as no answer, so such a remote gives the path form rather than an
exception at startup.

With no remote, or a remote that is a local path or would not pass `check_project`, the
name is `path:` plus the first 16 hexadecimal characters of the SHA-256 of the main working
tree's real path (`path_identity`). The path is put in one spelling first, so that Windows,
POSIX and the hooks' copy hash the same string: backslashes become forward slashes, a
drive letter is lower-cased, and trailing slashes are removed. `main_root` and
`path_identity` are pure and take Windows paths on any platform, which is how the Windows
behaviour is tested. It is provisional: it changes if the directory moves, and becomes the
remote form once the repository is pushed. Outside a git repository the answer is `None`.
SSH host aliases from `~/.ssh/config` are not resolved, so `git@work-github:o/r` is the
project `work-github/o/r`.

`ServerConfig.from_env` calls it once at startup when `MEMVARA_PROJECT` is unset and the
`project_scope` feature is on. The project then reaches the engine in two ways. Locally,
`Memvara.scope(project=...)` returns a view over a shallow copy of the `Memvara` whose
`default_scope.project` is that project, because every public method reads the project
from `default_scope` rather than taking it as an argument; the copy shares the store, the
models and the registry. No read or write takes a per-call `project=`, and `project` is in
`RESERVED_META`, so `remember(..., project="x")` is refused with the reason instead of
being stored as an annotation on a claim filed without a project. The hosted clients refuse
it the same way on `remember()` and `supersede()`. Against a hosted deployment,
`RemoteMemvara` sends the project as the `Memvara-Project` header on every request, and
sends no header when no project is bound. `Scope.contains()` compares the project like
every other field, so a slot operation's boundary does not rest on fact keys alone.

### A repository's own value shadows the user-wide one

Before project scope, the local server wrote every fact without a project. Inside a
repository, a new value for a single-valued, project-relative predicate now lands in the
repository's slot, whose key includes the project, so it cannot end the older user-wide
value. Ending it would be wrong anyway, because that value is still the answer in every
other repository. So both are stored, and `memvara/retrieve/shadow.py` decides at read
time. A present-tense read bound to project P leaves out a live claim with no project when
the same owner, subject and single-valued predicate has a live claim in P's slot. The check
is one store query per read, `Store.occupied_slots(tenant, fact_keys)`, over the distinct
slots of the candidates the read would otherwise return; a read with no project pays
nothing. Both slot lookups are optional on the store protocol: a store without
`occupied_slots` is asked one `count_competing` per slot, and a store with neither, or one
that raises `NotImplementedError` from them as `RemoteStore` does, returns the read
unshadowed rather than failing it.

It applies to `get_all()` (and so to `standing()`, `profile()` and the MCP tools built on
them), to `since()`'s `added` half, and to `search()` and `recall()` inside
`HybridRetriever`, after the graph walk and after every cheaper filter (memory type, belief
time, anchoring and the score floor), so a candidate those drop never costs a lookup. It does not apply to a read at another instant
(`valid_at`, `known_at` or `as_of`), because at that instant the repository may not have
had a value of its own; to many-valued predicates, where both values hold at once; to
predicates declared global, which are never written with a project; or to `count()` and
id-addressed reads such as `get()` and `why()`. When the repository's value ends, the
user-wide value answers there again. Nothing is written.

## `Memvara.standing()` and `Memvara.profile()`

```python
def standing(self, *, k=None, tenant=None, user=None, agent=None,
             session=None) -> list[Claim]
def profile(self, query=None, *, k=8, since=None, buckets=None, tenant=None, user=None,
            agent=None, session=None) -> Profile
```

`standing()` is every live `procedural` claim in the scope, sorted by `standing_order`:
stated before inferred (`is_derived`), then confidence, then newest, then id. The MCP tool
`memory_standing` sorts a hosted deployment's answer with the same key, so both engines
return one order.

`profile()` makes three reads: the scope's live claims (`get_all()`), the ids believed
at `since` on both clocks (`since()`'s "then" scan), and `search(query, k=k)` when there is
a query. `AsyncMemvara.profile()` runs the three on separate threads at once.
`_assemble_profile` builds every section from them without touching the store. `standing`
is `standing(k)`. `recent` is the live claims not believed at `since`, which equals
`since(since).added[:k]` without `since()`'s second scan for what left; `since` defaults to
seven days before now, and what left is not reported because a profile is read as current
context. Because `get_all()` already returns newest first, `recent` and the buckets are
filters over one list with no sort of their own.
`buckets` defaults to one bucket per pack in `PROFILE_PACKS` (`decisions`, `engineering`,
`events`), holding that pack's predicate names; each bucket lists the newest `k` live
claims whose predicate it names. A caller's bucket predicate is kept when the registry
knows it (its canonical spelling is added), a shipped pack declares it, or a live claim in
the scope uses it; anything else goes to `Profile.warnings`. The packs are read only when a
caller's predicate is neither registered nor stored. Python 3.10 has no `tomllib`, and by
the decision `schema._toml_reader` records there is no second reader and no `tomli`
fallback, so there the packs cannot be read: each default bucket is reported as
unavailable in `warnings`, and every other section is unaffected. A pack that cannot be
read is reported before any "nothing declares" warning it caused.

`RemoteMemvara.profile()` sends `POST /v1/profile` with a JSON body of `query`, `k`,
`since` and `buckets` (unset ones left out) and the scope as query parameters, and expects
`standing`, `recent`, `relevant` (lists of `{claim_id, text, inferred}`), `buckets` (name to
list of rows) and `warnings` (strings). `hydrate.profile` indexes every key, so a reply
missing a section raises instead of reading as empty.

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
              include_invalidated=None, where=None)
lexical_search(query, scopes, limit, *, valid_at=None, known_at=None, states=None,
               include_invalidated=None, where=None)
vector_search(qvec, scopes, limit, *, valid_at=None, known_at=None, states=None,
              include_invalidated=None, where=None)
episode_candidate_ids(scopes, *, valid_at=None, known_at=None, where=None)
lexical_search_episodes(query, scopes, limit, *, valid_at=None, known_at=None,
                        where=None)
vector_search_episodes(qvec, scopes, limit, *, valid_at=None, known_at=None,
                       where=None)
episodes_near(anchor, scopes, limit, *, valid_at=None, known_at=None, where=None)
```

`where` is the caller's metadata and file-path filter, a `memvara.filters.SearchFilter`
or `None`; see "Metadata and file-path filters" below.

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

### Reading a whole scope, and joining the text index

Most reads are capped, and they filter by scope with `_scope_clause`: one `OR` term per
ancestor scope, in the same statement as the `LIMIT` (design invariant 7). The two
candidate lists are not capped. `candidate_ids` and `episode_candidate_ids` return every
row a scope can see, because the vector leg ranks inside that list, so what a query costs
per row decides what they cost.

**One `SELECT` per scope.** SQLite plans the `OR` as a MULTI-INDEX OR: every rowid each
term returns goes into a temporary set first, so that a row two terms both match comes back
once. For a whole-scope list that set holds the whole scope. `_scoped_union` instead builds
one `SELECT` per scope and joins them with `UNION ALL`. That needs no set, because a row is
stored at exactly one scope and two distinct scopes never return the same row. Repeated
scopes are dropped before the SQL is built, since two copies of one scope would return its
rows twice. Over 100,000 claims in one user's scope this takes the claim list from 84 ms to
69 ms. `tests/test_store.py` checks the lists against the `OR`'s meaning over every shape a
scope can take.

**A covering index for the turn list.** `ep_cover` indexes a turn's five scope columns,
`ts` and `id`, which is every column `episode_candidate_ids` reads. SQLite answers the query
from the index and never reads a turn's row. Over the 189,520 LongMemEval-S turns in one
scope, the list took 244 ms before, 102 ms with the index, and 84 ms with one range per
scope. A plain scan of the table takes 70 ms. `ep_scope`, the older index, stays: an older
build creates it on every open, so dropping it would make a file both builds open rebuild
it each time. `ep_cover` is one of the late indexes, created on every open, so an existing
store builds it the first time this version opens it: 1.7 s and 10 MB for those 189,520
turns.

**The lexical legs join on rowid.** `lexical_search` and `lexical_search_episodes` join
each text index row to its table on rowid, not on the `claim_id` or `episode_id` column the
index row also stores. Reading that column back reads the index's copy of the row, text and
all, for every match. In `bench/scale.py` the claim leg's median went from 323 ms to 120 ms
over 100,000 claims, and the turn leg's from 140 ms to 55 ms over 199,499 turns. The join
is correct only because every index row sits at the rowid of the row it indexes, which has
been true since 0.1.0 and which erasure already relied on. `put_claim` and `add_episode`
upsert, so a row keeps its rowid, and write the index row with that rowid. Erasure and
`purge` delete the index row by rowid before the row. `VACUUM` and `VACUUM INTO` keep the
rowids of a table that has an index, as both tables do, although SQLite documents `VACUUM`
as free to renumber a table without an INTEGER PRIMARY KEY. The backup API copies pages, so
it keeps them too. A copy that re-inserts the rows into a new file need not keep them, and
a store copied that way can miss a lexical match or return another row in its place;
`docs/UPGRADING.md` has the repair. `tests/test_store.py` checks the invariant after every
write that moves or frees a rowid, after a `VACUUM`, and in a `VACUUM INTO` copy.

The vector leg then looks up each candidate's row in the matrix. `_VecIndex.search` does
that with `np.fromiter(map(dict.get, ...))`, which runs the lookups in C rather than in a
Python loop: 79 ms to 57 ms for 199,499 candidates, beside 45 ms for the product itself.

**Each scope's turns, kept in memory.** The turn list and those lookups are the same work
on every search until something is written, so the vector leg over turns keeps them.
`_scope_turns` holds, per scope, the ids of the turns that have a vector, their `ts` and
their matrix rows, sorted by `(ts, id)`. A search takes, for each of its scopes, the prefix
with `ts` at or before the instant asked about, found by binary search, and hands the rows
to `_VecIndex.search_rows`, which ranks them exactly as `search` does. The candidates are
the rows `_scoped_union` returns, in the order it returns them, so two turns whose cosines
tie keep the same order and the same rows come back. `tests/test_store.py` checks that
against the SQL path over every shape of scope, five pairs of instants and four limits,
with half the vectors shared. Over the 199,499 turns in one scope of `bench/scale.py`, the
vector leg's median went from 220 ms to 54 ms, and a whole search's from 450 ms to 284 ms.

A list lives until `_changed` empties the cache. That runs after every commit this store
makes, whatever it wrote, and when the first search after another connection's commit
sees it. That search asks one connection kept for the purpose, through `_notice_commits`,
because `PRAGMA data_version` can only be compared with an earlier answer from the same
connection: asked on each reading thread's own connection, it emptied the cache once per
thread per commit, and on every thread's first read with nothing committed at all. The
watch is read before `_ensure_index` refreshes the map, so a list rebuilt for a commit is
built from a map that already holds that commit's rows. A list built while a change
happened is returned to the search that built it and not kept. So the first search after
any write rebuilds the lists it needs,
which costs more than the SQL path it replaces: in `bench/scale.py` the vector leg took
238 ms after a write against 220 ms before this change, and a whole search 471 ms against
450 to 472 ms. Every search after it, until the next write, takes the 54 ms. Three reads
still take the SQL path: a filtered one, because a filter can name any metadata field
and the lists hold none; one inside `batch()`, because that thread must see its own
uncommitted rows and a list shared with other threads must never hold them; and one before
this process has seen any vector, because an index loaded then never learns a width.

A write in another thread can still move a row between a search reading its lists and
ranking them. `search_rows` therefore checks, under the index lock, that every row is
inside the matrix and that each turn it returns still holds the row it was scored by, and
returns `None` when one does not, which sends the search down the SQL path. A turn it
does not return cannot change the answer by having moved. The lists hold at most
`_SCOPE_TURNS_ROWS` turns, 1,000,000, across all scopes, dropping the least recently used
scope first. Each turn costs about 100 bytes: 19.3 MB for those 199,499.

**Each scope's claims, kept in memory for a read of the present.** The vector leg over
claims keeps its candidate list the same way, with one difference: a claim's state depends
on the instant read, and a read of the present reads a new instant every time.
`_scope_claims` holds, per scope, per set of states and per `hide_expired`, the ids and
matrix rows of the claims that have a vector and are in those states. It reads them with
the statement `candidate_ids` runs for that one scope, bound at the search's instant, so
they come back in the order the SQL path returns them. Every state predicate, and the
expiry clause, compares one of five columns with the instant read: `recorded_at`,
`invalidated_at`, `valid_from`, `valid_to` and `expires_at`. So without a write, a claim's
state can change only when the clock passes one of them. The same statement also asks for
the earliest such instant still ahead among the tenant's claims, and the list answers reads
from when it was built until then, or until the next commit empties it. `cl_last_change`
indexes each claim's latest time column (`_LAST_CHANGE`), so that question reads only the
claims whose last change is still ahead, usually none: 0.003 ms against 41 ms for a pass
over 100,000 claims. A claim in another scope of the tenant can end a list early, which
costs a rebuild and never a wrong list. The list answers from the clock read after its
statement rather than from the instant it bound, because the expiry clause reads the wall
clock a moment later than that instant.

`tests/test_store.py` checks the lists against the SQL path over every shape of scope,
every set of states and four limits, with the clock moved past a start, an end, a
retirement, an expiry and a late recording, and then back. It also reads the state
predicates and checks that every column they compare with the instant is one
`_LAST_CHANGE` and `_NEXT_CHANGE` read, so a state that came to read a sixth column would
fail there rather than leave a list in place past a change. A read pinned to an instant
still asks SQL, because it would need a list per instant, and so do a filtered read and a
read inside `batch()`, for the turn lists' reasons. Over 100,000 claims in one scope,
`vector_search` went from 168 ms to 69 ms and a whole search from 233 ms to 132 ms. The
first search after a write costs what it did before, because the list is the same SQL the
leg ran before and the next instant is one seek. What is left of the leg is mostly the
product over 100,000 vectors. Each claim held costs about 90 bytes: 8.9 MB for those
100,000.

**The lexical leg over turns ranks in the text index first.** The full query joins every
matching turn to its row, because the scope, the time bound and the tie-break are columns of
`episodes`. Those columns sit after `content`, so each long turn also costs its overflow
pages. Over the 189,520 turns in one scope, a question matching 73,719 of them took 187 ms,
71 ms of it inside the text index. `_episode_text_first` ranks the matches by `bm25` inside
the text index instead, keeps the best `top` of them, the larger of four per row asked for
and 100, and reads only those turns, by rowid. `CROSS JOIN` fixes that join order. Left to
choose, SQLite walked every turn of the scope in `ep_cover` and looked each one up in the
text index, which is slower than the full query.

It returns the full query's answer when the ranked rows prove it, and None otherwise, which
sends the leg to the full query. When fewer than `top` turns match, every match was ranked,
so the ranked rows that pass the scope and the time bound are all the rows the full query
would see. When `top` or more match, every match scoring strictly better than the worst
ranked row was ranked, because a match left out scores no better than that row. So once
`limit` of those pass the scope and the time bound, they are the full query's first
`limit`, in its order, tie-break included. The count, the worst score and the rows come
from one statement, so from one snapshot. The ranked rows are cut before the scope and the
time bound narrow them, which is the arrangement design invariant 7 forbids when nothing
notices the cut. Here the statement reports the cut, and the leg returns rows only when they
prove that the cut changed nothing. `tests/test_store.py` checks the leg against the full
query over every shape of scope, five pairs of instants and six limits, with every third
turn a copy of one text so that scores tie.

A search misses when too few ranked rows are turns it may see: its scopes hold few of the
store's matching turns, it reads an instant before most of them happened, or a tie runs
across the worst ranked row. A miss pays for both statements, so after one the next
`_TEXT_FIRST_BACKOFF` searches, 16, of the same scopes go straight to the full query. A
read pinned to an instant counts its misses apart from a read of the present, so reading
the past does not slow the present. A filtered read always runs the full query, because
its filter can name any metadata field. Over those 189,520 turns, all 50 LongMemEval-S
questions that `bench/scale.py` times proved their answers. The leg's median went from
55 ms to 23 to 26 ms, and its 95th percentile from 149 ms to 70 to 74 ms.

### Why a claim was closed

The reason for a closure is stored on the closure witness, `meta["closure"]`, which
`close_out` already appends to every time a claim ends or is retired. An entry gains a
`reason` key only when a caller gave one, so every closure written without a reason has
exactly the record it had before, and no existing row changes shape. The reason is at
most 500 characters (`types.REASON_CHARS`), stripped, and refused when blank, because an
empty reason on the record reads as a reason that said nothing. `types.closure_reason`
validates it and `types.closure_reasons(claim)` reads every one back, oldest first.

The reason lives in `meta` and not in a column, so it needs no migration, and the read
path never consults it: like the rest of the witness, it is evidence about a closure and
never an input to which rows a query returns.

A fact written with its end already known (`remember(valid_to=..., until_reason=...)`)
gets an `ended` witness at its `valid_to` when it is written, through
`types.planned_end`. Nothing else writes a witness for a `valid_to` set at write time, so
a planned end with no reason still has none.

`remember(replaces=<id>, reason=...)` builds the new claim and hands it to `supersede()`,
so there is one implementation of a named replacement and not two that can drift.
`invalidated_by` points from the old claim to the new one and the reason lands on the old
claim's witness. `supersede()` refuses, before anything is written, an id the caller
cannot see (`KeyError`), a claim already retired, and, under `close="ended"`, a claim
that is not live now (`ValueError`). Retiring an ended claim is allowed: a finished value
later found never to have been true is a correction, and the closure witness is a list so
that it can hold both events.

**The closure instant of a supersession has one rule.** An explicit `at` wins. Otherwise
`"ended"` closes the old claim where the world changed, which is where the new claim
begins: its `valid_from`, which `remember()` sets from `valid_from`, else `recorded_at`,
else now. `"retired"` closes belief in the old claim when the new record was made: its
`recorded_at`. This is the rule ending follows everywhere else, where a closure lands on
the axis its word names. Before this, `supersede()` closed both readings at the new
claim's `recorded_at`, so a replay whose new value began earlier than it was recorded
ended the old value at the recording instant instead of the instant the value changed.

### Closing everything that matches a query

`Memvara.forget_matching` is two calls, and the first one writes nothing. Without
`confirm` it runs an ordinary `search()` and returns the matching ids with their text and
a token. With `confirm` it checks the token, re-reads each listed claim through `get()`,
and closes them all in one `batch()` only if every one is still live and visible;
otherwise it raises `ConfirmationRefused` and writes nothing. The query is not run again
on the confirming call, so the set closed is the set the caller saw.

The token (`memvara/confirm.py`) is the sorted ids, the closure and an expiry ten minutes
out, serialised as JSON and followed by an HMAC-SHA256 of those bytes. The ids travel in
the token, which is what lets the confirming call avoid a second search. The key is
`Memvara(confirm_secret=...)`, which a server fills from `MEMVARA_CONFIRM_SECRET`; with
none, every `Memvara` in the process shares one key generated at import. So a token
survives neither a restart nor a hop to a process with another key, and that is the safe
direction to fail in. Scope is not in the token. It does not need to be: the confirming
call re-reads every id through `get()` in the caller's own scope, so a token minted for one
user closes nothing another user cannot see.

The MCP surface offers this as two tools, `memory_end_matching` and
`memory_forget_matching`, one per closure. The module docstring of `server/tools.py`
explains why a closure is chosen by a tool's name and never by an argument.

### Typed links between claims

`claim_links` (schema 13) holds one row per link: `tenant`, `from_id`, `to_id`,
`relation`, `created_at` and `by`, keyed on the first four. `relation` is `extends` (the
first claim adds detail to the second) or `derives` (the first was inferred from the
second), enforced by a `CHECK`. Supersession is not a relation here, because the
replacing claim is already recorded in `invalidated_by`.

The table has no `REFERENCES` clause. This connection does not turn on
`PRAGMA foreign_keys`, and a clause SQLite does not enforce would read as a guarantee it
is not. Both halves of the guarantee are kept in code instead. `Memvara.link` refuses an
id the caller cannot see. `erase_claim` and `purge` delete every link that touches an
erased claim in the same transaction as the claim, and `residue()` counts `claim_links`,
so a proof of erasure fails if one survives.

`put_link` writes with `ON CONFLICT DO NOTHING` rather than `INSERT OR IGNORE`. The second
also ignores `CHECK` failures, so a link with an unknown relation would have been dropped
without an error.

Links carry a creation instant and no closure. They record a relationship between two
records rather than a fact about the world, so they last exactly as long as both claims
are stored, including after either is ended or retired. `why()` dates them on the belief
clock only: `known_at` drops a link recorded after it.

`Memvara.links(claim_id)` returns both directions, and leaves out a link whose far end
the caller cannot see, because the link would otherwise disclose that id.

### Documents

Schema 14 adds two tables. `documents` holds one row per document, keyed on `(tenant,
id)`: `custom_id`, the scope both as its five parts and as `scope_key`, `title`,
`filepath`, `source_uri`, `mime`, `content_hash` (blake2b-16 of the normalised text),
`status` (`queued`, `extracting`, `done`, `stored` or `failed`, enforced by a `CHECK`),
`error`, `meta`, `created_at` and `updated_at`. A unique index on `(tenant, scope_key,
custom_id)` makes a caller's own id unique per scope. `document_chunks` holds one row per
chunk, keyed on `(tenant, document_id, position)`, with the chunk's `hash` and the
`episode_id` it was stored as. It holds no text: the text lives only in the episode, and
`document_chunks()` reads it through `episode_id`, so there is no second copy for an
erasure to miss. A read counts a document's chunks with one aggregate join.

**A chunk is an episode.** Each one is written as a `role="system"` episode with
`meta["document_id"]` set, so the episode text index, the episode vectors and
`search(include_episodes=True)` serve documents with no second index, and `why()` on a
claim extracted from a document quotes the chunk.

**What `episodes.hash` holds.** For an ordinary turn, as before: blake2b-16 of the scope
key, the role and the text. For a document chunk, the episode whose `meta` has a
`document_id`, the document id is mixed in as a fourth part. Without it, the same
paragraph in two documents would hash alike, exact-repeat detection would store it once,
and deleting either document would erase text the other still holds. No stored hash
changes on upgrade, because no episode before schema 14 carries the key. The column is a
dedupe key, not a digest of the text alone: two rows with the same text can differ in it.

**Chunking** (`memvara/documents/chunk.py`) splits the normalised text into runs of whole
sentences of at most 1,000 characters, each preceded by up to 150 characters repeated
from the end of the chunk before. A sentence is cut only when it is longer than a chunk
on its own. Cut points are chosen first, from the sentences alone: a sentence whose own
digest falls below a threshold proportional to its length (one every 600 characters on
average), at least 300 characters after the previous cut point. A chunk ends at each cut
point, and also before a sentence that would take it over 1,000 characters; such a forced
split moves no cut point. So an edit usually changes the chunks it touches and the one
after it. Measured over 400 single-sentence insertions, one per paragraph of ten
27,000-character documents of generated prose: 390 changed at most two chunks, 9 changed
three and 1 changed four. With the 300-character minimum counted from the start of the
chunk instead, 24 changed three or four. `Memvara(retrieval_chunks=False)` stores a
document as one chunk.

**Re-ingest matches chunks by content, not position.** `add_document` with a `custom_id`
that already exists at the same scope, and `update_document` with new content, chunk the
new text and match each chunk to an unused old chunk with the same `hash`. A match keeps
its episode, vector and citations, and only its position changes. An unmatched chunk
becomes a new episode. An old chunk with no match is released the way a delete releases
every chunk (next paragraph). The lookup of the `custom_id` and the rows it decides
between are written in one `batch()`, so two concurrent adds of one id cannot both miss
the lookup; indexing and extraction run after that transaction, so no model call holds
the write lock.

**Delete erases text and retires memory.** `delete_document` finds every claim citing
one of the document's episodes with one `claims_citing_any` query. A claim whose every
source is among them is retired first, with the closure reason `"source document
deleted"`; a claim with another source keeps it. Both lose the erased episodes from
`sources`. Then the document row and its chunk rows are deleted and the episodes erased
with one `erase_episodes` call, all in one `batch()`. **A chunk episode belongs to its
document**: while a document lists it, `erase_episode`, `erase_episodes` and
`erase_claim(sources=True)` keep it, whatever `cited` says, so no erasure path leaves a
document reporting a digest and a chunk count its text no longer has. `purge` deletes the
scope's documents and their chunk rows and counts both, as `documents` and
`document_chunks`, beside its four other keys.

**Extraction and status.** A document is written `queued` and becomes `extracting` while
`WritePipeline.reextract` reads its new chunks. The salience gate accepts a chunk
whatever its role; the fast path does not run on one, because it reads first-person
sentences as the user's own and runs only on user turns, so facts come from the model
tier. The outcome is `done` when every chunk has been read, `failed` with an `error`
when extraction raised or was deferred, or `stored` when a chunk was kept unread because
the call passed `extract=False`. Such a chunk carries `meta["extract"] = False`, which
the gate refuses, so a later `reextract()` sweep keeps the choice. Adding a `failed` or
`stored` document again with extraction on reads the kept chunks no claim cites yet, and
clears their mark. The chunks are stored and indexed before extraction runs, so a
document in any state is stored and searchable.

**Content that is not plain text** — a URL, `bytes`, HTML or any non-text mime — is handed
to `memvara.ingest.extract`, with the instance's `url_fetcher` (the MCP server passes
`ServerConfig.url_fetcher()`, which carries `MEMVARA_NAT64_PREFIXES`; unset, ingestion uses
its own `SafeFetcher`), its `llm` for images, audio and video, and its `ingest_urls` and
`ingest_media` switches as `allow_urls` and `allow_media`. The document store fetches
nothing itself. An `IngestError` is raised before anything is written. `update_document` reads new content under the `mime` it is given,
otherwise under the stored type for text, otherwise with none so ingestion detects it. A
configured redactor runs over the whole text and the title before chunking, so every
stored digest is of redacted text.

`list_documents` filters by scope, `filepath_prefix` (compared with `substr`, so `%` and
`_` match only themselves) and `status` in the same statement as its `LIMIT`, per
invariant 7, and pages on `(created_at, id)` newest first.

The nine store methods (`put_document`, `get_document`, `find_document`,
`list_documents`, `document_chunks`, `put_document_chunks`, `delete_document`,
`claims_citing_any`, `erase_episodes`) are optional as a group, and a store that has them
says so with `holds_documents = True`, which is what `Memvara` asks; see `OMITTABLE`.
`RemoteStore` has each as a stub that raises and names the `RemoteMemvara` method to use,
because the facade chunks, scope-checks and extracts server-side; it sets the marker to
false, so `Memvara(store=RemoteStore(...)).add_document()` is refused with that advice.

### Metadata and file-path filters

`search()` and `recall()` take `filters` and `filepath_prefix`, and `memvara/filters.py`
checks them and combines them into one `SearchFilter`. The retriever hands that object,
as `where=`, to every store method that caps rows: `candidate_ids`, `lexical_search`,
`vector_search` and the four episode methods. It is a store parameter for the reason
`states` is (invariant 7): a filter applied to a result the store had already cut to
`limit` would find a match only when it happened to land inside the cut.
`tests/test_metadata_filters.py` builds forty rows that do not match above five that do,
shows that an unfiltered read fifteen deep holds none of the five, and asserts that a
filtered read with `k=3` returns three of them, for the whole search and for each store
method. A read with a query rewrite retrieves every phrasing with the same filter before
the lists are fused.

**What matches.** A filter is an equality test on a top-level key of `meta`, a list means
any one of its values, and every key must match. A string matches only the same string,
a number any equal number, and `True` or `False` only a boolean; a stored list or object
never matches. Keys must match `[A-Za-z0-9_.-]{1,64}`, so a backend can put one inside a
JSON path without quoting rules of its own, and anything else is a `FilterError` (a
`ValueError`) before a query runs. **Each key is tested on its own**: a row matches a key
when its own `meta` **or** the `meta` of any document it came from holds a wanted value,
and different keys may be held by different sources. So a claim whose own `meta` says
`team: infra`, extracted from a document whose `meta` says `year: 2025`, matches
`{"team": "infra", "year": 2025}`. A chunk episode reaches its document through
`document_chunks`, and a claim through `claim_sources` to such a chunk. `filepath_prefix`
has only the document route. `_starts_with` compares it with `substr`, and
`list_documents` uses the same helper, so `%` and `_` match only themselves and case
matters; `LIKE` would treat both as wildcards and ignore ASCII case.

**How SQLite evaluates it.** `_where_clause` in `store/sqlite.py` adds one condition per
key, each "the row's own `meta` holds it, or `EXISTS` a source document whose `meta`
does", and one `EXISTS` for the prefix. The key test has two forms, and the store picks
one when it opens:

- **With SQLite's JSON functions**, which the design spec (§3) asked for: `json_type`
  keeps the JSON types apart the way `memvara.filters` does, so a boolean is never the
  number 1, and `json_extract` compares the value. The path `$."<key>"` is bound as a
  parameter like the values.
- **Without them**, `mv_meta_match(meta, ?)`, a Python function registered on the
  writer's connection and on every snapshot connection `_reader` opens, because a SQL
  function belongs to one connection. The JSON functions are built into SQLite only from
  3.38, and this library supports 3.35 (the reason `_migrate_to_v5` parses in Python), so
  a build without them still filters correctly, more slowly.

Nothing the caller wrote is placed in the SQL text. There is no index on `meta`: the key
test runs on the rows the scope, state and text clauses leave, which is the whole scope
for the vector leg's candidate list.

**What it costs.** `PYTHONPATH=. python3 bench/filters.py` builds a store of N competing
claims tagged `team=infra` plus 20 tagged `team=web`, and times `search(k=10)` (median of
15, query rewrite off) unfiltered, filtered to `team=web` with the JSON functions, and
filtered with the Python callback. Measured on SQLite 3.50.4, Python 3.13, an Apple
silicon Mac:

| Competing claims | Unfiltered | Filtered, JSON functions | Filtered, Python callback |
|---|---|---|---|
| 15,000 | 40.5 ms | 33.1 ms | 79.4 ms |
| 40,000 | 111.1 ms | 92.0 ms | 216.6 ms |

A filtered search with the JSON functions is slightly faster than an unfiltered one,
because the vector leg then ranks 20 candidates instead of every claim in scope. The
callback roughly doubles the unfiltered time, which is why it is only the fallback.

**What does not take the filter.** The graph leg does not run on a filtered search:
`Store.adjacent` takes no filter, a walk would step onto rows the filter excludes, and
removing them afterwards would mean reading each row's metadata and documents again
outside the store. `recall(include_history=True)` renders the past values of the facts the
filter kept without filtering them again, because they are the same fact at an earlier
time.

**A store without `where`.** The retriever passes `where` only when the caller filtered,
so a third-party store written before the parameter serves every unfiltered read as
before, and a filtered read against it raises `TypeError` naming the argument rather than
returning rows the filter would have excluded. `RemoteStore` accepts the argument on its
stubs, which still raise.

**The switch.** `Memvara(metadata_filters=False)`, `RemoteMemvara(metadata_filters=False)`
or `MEMVARA_FEATURE_METADATA_FILTERS=0` refuses a call that passes either argument with a
`FilterError` naming the switch; an empty `filters` mapping narrows nothing and is not
refused. The check and the parsing live in one function, `checked_filter`, which every
engine calls. A server started with the feature off sets its engine's switch, and the MCP
tools turn the engine's `FilterError` into an argument error. The two arguments stay in
the schema with a description saying they are refused; removing them would refuse the
call too, since `validate` refuses unknown arguments, but the message would not name the
switch.

**The hosted deployment.** `RemoteMemvara` sends `filters` and `filepath_prefix` only when
set, after the same checks. The deployment does not accept them yet; its request models
refuse an unknown field with 422, so a filtered call fails rather than being answered
unfiltered. The Postgres store and the cloud routes are stream P2-G of the phase 2 parity
design, and a Postgres implementation that uses `LIKE` must escape `%` and `_`.

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

The vector file is blanked by row number, and the row comes from the database, not from
the process's map of ids to rows. That map is loaded on the first search, so a process that
opened a store and erased a claim before searching used to leave the vector in `<db>.vecs`.
`tests/test_vecindex.py::test_an_erasure_before_any_search_still_blanks_the_row_on_disk`
reads the file.

### Encryption at rest

`SQLiteStore(path, encryption=True)` creates a new store encrypted. An existing file is
opened as whatever its first 16 bytes say it is: `SQLite format 3\0` is unencrypted, and
anything else is treated as encrypted and needs a key. `encryption` therefore decides only
what a new file becomes, and `:memory:` is never encrypted. `memvara/store/encryption.py`
holds the key lookup, the vector file's record format and the conversion.

**Two files, two mechanisms, one key.** The database is SQLCipher: every connection, the
writer's and each reading thread's, is opened through `SQLiteStore._connect`, which issues
`PRAGMA key` with the raw-key form (`x'<hex>'`, no password stretching) and
`PRAGMA temp_store = MEMORY`. The store keeps a `_sql` module reference, `sqlite3` or
`sqlcipher3.dbapi2`, because the two libraries have separate exception classes. The vector
file is `VectorSealer`: a 64-byte header (magic `MEMVASEA`, format, width, and the
database's 16-byte salt) and one record per row, `nonce(12) | AES-256-GCM(float32 row) |
tag(16)`. The row key is HKDF-SHA256 of the store key under that salt, with its own `info`
string, so the AES key is never a key SQLCipher uses. The associated data of a record is
the header, the row number and the owner's id. An all-zero record is an empty row.

**The salt comes from the database, so every process agrees on it.** `SQLiteStore` reads
`PRAGMA cipher_salt` (the first 16 bytes of the encrypted database file) after its first
commit and gives it to the sealer. Nothing in the header is chosen by the process writing
it, so two processes that each find the header missing, for example after the vector file
was deleted, write the same bytes and seal records under the same key. An earlier draft
picked a random salt per file. Two processes rewriting a missing file then each bound to
their own salt, the last header written won, and the next open refused the store because
the other process's records no longer authenticated. The salt is fixed for the life of the
database file, and a converted store is a new file, so a vector file written for the old
one is recognised as foreign and rebuilt.

**The matrix is on the heap, decrypted.** A memory-mapped matrix would expose the
ciphertext to every read, so `_VecIndex` keeps the path for the file and the matrix in
memory: `attach` checks the header and reads no rows; `put` writes one sealed record;
`forget` zeroes one. The cost is memory. Each process holds its own copy, where the mapped
file shared its pages between processes.

**Loading is where authentication is checked.** The first search runs
`SQLiteStore._load_sealed`: for every row the database lists, `_VecIndex.load` reads the
record and opens it against its row and owner. A record that does not open is looked up
again in the database before anything is decided, because another process may be changing
it: an owner that is gone or moved takes the database's answer; a blank record whose owner
is still there is an erasure in flight, and the vector is read from the database; a short
or failing record whose owner is still there, after one more read in case the first caught
a write half done, raises `EncryptionError` naming the file and the row. A later refresh,
after another process commits, reads the new rows' vectors from the database rather than
from the file, because the file may already hold that process's next, uncommitted write.

**The vector file is opened in binary mode on Windows.** `_VecIndex.attach` passes
`os.O_BINARY` to `os.open` wherever that flag exists. Without it, Windows opens the file in
text mode, and its C runtime deletes a final 0x1A byte from any file opened for reading and
writing. That happens inside `os.open`, before `os.fdopen` switches the descriptor to
binary. The last byte of the encrypted file is the last byte of a GCM tag, so it is 0x1A in
about one write in 256. When it was, the next open cut the file by one byte, and loading
raised `EncryptionError` because the last record ended early. The unencrypted file lost the
top byte of its last float the same way. `tests/test_encryption.py` and
`tests/test_vecindex.py` each write a file that ends in 0x1A and reopen it.

**A damaged header is not an error**: it is rewritten and every row is rebuilt from the
database, as a stale unencrypted file is. The database is the authority
and authenticates its own pages, so a rebuild cannot load anything wrong.

**The key** is looked up by `resolve_key`: the OS keychain through `keyring`, then
`MEMVARA_DB_KEY`, then `~/.memvara/db.key`. Only the creation of a new store may generate
one (`create=True`). The key is written to a temporary file in the same directory and
hard-linked into place. The link fails if a key is already there, so a key is never
replaced, and another process never sees the file half written. Where hard links are not
supported it falls back to `O_EXCL`, and a reader that finds a key file shorter than a key
reads it again for up to a second. A key from the file, generated or read, raises
`EncryptionWarning`. On POSIX the warning also names the file's mode and `chmod 600` when
the group or other users can read it. No message in the module includes a key, and
`StoreKey` keeps the key out of its `repr`.

**The server reads `MEMVARA_DB_KEY` from the environment it was configured from**, not
from `os.environ` at open. `ServerConfig.from_env(env)` checks it and keeps it as
`ServerConfig.db_key` (out of `repr`). `build_memvara` passes it to
`Memvara(key_env=...)`, which passes it to `SQLiteStore(key_env=...)` and on to
`resolve_key(env=...)`, so the keychain still comes first. This is what lets the plugin's
hooks open the store: they build their configuration from the MCP client's server block,
which never reaches `os.environ`. A key that lived only there used to open the server's
store and silently not the hooks'.

**Conversion** (`encrypt_store`, the `memvara encrypt` command) exports the database with
`sqlcipher_export` into a temporary file in the same directory, copies `user_version`
(which the export does not), opens the copy once to write its vector file and once more to
read every record back, compares row counts, checks the original has not changed and has
no `-wal` or `-shm` (the sign of another open connection), and only then renames the copy
over the original. The vector file is renamed second; if that rename fails, the next open
finds a header it did not write and rebuilds the file.

**The switch.** `encryption` is in `FEATURE_DEFAULTS`, on. `build_memvara` passes it to
`Memvara(encryption=...)` and turns `EncryptionUnavailable` into a `ConfigError` naming the
extra and `MEMVARA_FEATURE_ENCRYPTION=0`, because a new store created unencrypted would stay
that way, and under stdio a warning would reach nobody. An existing unencrypted store is
not refused; `memory_stats` reports it on its `storage:` line (`ToolContext.storage`, set
by `mcp._storage_fact`). The library's `Memvara` defaults to `encryption=False`, because a
library call should not read the OS keychain or write a key file unless asked.

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
on the two public corpora with identical retrieval code, 2Wiki joins at 29.0% with its
relations declared, and the graph leg takes chained questions from 28.2% to 48.3%;
LongMemEval joins at **0.0%** —
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
    def merge_duplicates(self, tenant: str | None = None,
                         threshold: float | None = None) -> int
    def promote(self, tenant: str | None = None, min_observations: int = 3) -> int
    def run(self, tenant: str | None = None,
            now: datetime | None = None) -> dict[str, int]
```

- `decay` multiplies `salience` by the predicate's recency factor, floored at `0.05` so
  nothing decays to zero and disappears from ranking entirely.
- `merge_duplicates` finds live claims sharing a `fact_key` whose embeddings reach
  `threshold`, keeps the one with the highest `observation_count` (ties broken by earliest
  `recorded_at` for determinism), folds the others' `sources` and `observation_count` into
  it, and invalidates them with `invalidated_by` pointing at the survivor. It writes no
  typed link. A merge is a supersession of near-duplicates, `invalidated_by` already
  records it, and `why(survivor).superseded` reports it; a `derives` link would be a
  second record of the same fact. `derives` is for a claim inferred from other claims,
  and nothing in consolidation creates one. `threshold=None`, the default, and what
  `run()` uses, is the value measured for the embedder's space (`embed/calibration.py`):
  0.985 for all-MiniLM-L6-v2, 0.99 for bge-small-en-v1.5 and 0.97 for any other embedder.
  The two measured values each sit above the closest pair of different values in their
  space: two booking references at 0.979 under MiniLM, and two availability zones at
  0.989 under bge-small. Two claims whose objects hold different numbers never merge,
  whatever `threshold` says. The numbers are the runs of digits, compared in order as
  strings without their leading zeros, so "09:30" and "9:30" can merge and "85,000" and
  "85000" cannot. No threshold does this job: MiniLM scores two dates a day apart at
  0.997 and bge-small two numpy versions at 0.995, above every restatement either model
  was measured on.
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
    def extract(self, episodes, known_predicates, *, usage=None,
                guidance=None) -> list[dict]
    def resolve_predicate(self, surface: str, candidates: Sequence[str]) -> dict
    def classify_predicate(self, predicate: str, example: str) -> dict  # legacy fallback
```

**Extraction guidance.** `Guidance` (`memvara/llm/guidance.py`) holds a project
description (`context`, at most 1,500 characters) and two lists of rules (`include` and
`exclude`, at most 20 rules of 200 characters each). Anything longer is refused with
`GuidanceError`, never cut. `with_guidance(system, guidance)` appends it to a system
message under the fixed heading `GUIDANCE_HEADING`, and returns the message unchanged for
`None` or an empty guidance. Both backends call it on their extraction prompt: the shipped
`EXTRACT_SYSTEM` for `AnthropicLLM`, and for `OpenAILLM` whichever prompt is in use,
including a full replacement from `MEMVARA_LLM_EXTRACT_SYSTEM`. The guidance stays in the
system message; the turns go in the user message as data. `accepts_guidance = True` on
a backend is the advertisement `WritePipeline` checks, and it is read with `getattr` rather
than declared on the `LLM` protocol, because a new protocol member would make every older
backend fail `isinstance`. `load_guidance(path)` reads the three fields from TOML and
refuses an unknown key; it needs `tomllib`, so on Python 3.10 it raises, for the reason
`schema._toml_reader` gives.

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
- **Do not add JSON Schema keywords to the shared schemas.** Strict mode rejects a long
  list of them — `maxItems`, `minItems`, `pattern`, `minimum`, `format`,
  `propertyNames` among others — and an unsupported keyword is a 400, not something
  ignored, so one added keyword takes the backend off the air. `FakeCompletions` does not
  validate the schema, so nothing in the suite catches it except the denylist in
  `test_every_schema_satisfies_strict_mode`.
- If your backend constrains decoding, cap the claims array with
  `bounded_claim_schema(n)` rather than the shared `CLAIM_SCHEMA`. Unbounded, "one more
  claim" stays legal forever and a grammar has no legal way to end a response: a model
  that starts restating itself runs to its token limit and the reply arrives as truncated
  JSON, losing the good claims that preceded the restatements too. A hosted model closes
  the array itself, which is why this is opt-in — `OpenAILLM(max_claims=...)` is the seam,
  because that backend serves both hosted OpenAI and self-hosted servers.
- **Raise `TruncatedResponse` when the provider says the token budget ran out**, by
  calling `_shape.refuse_if_truncated(reason, cutoff, model=..., budget=...)` with your
  own field's value and your own provider's word for it. This is not optional politeness:
  truncated JSON does not parse, `parse_json_object` returns `{}` for it, and `{}` shapes
  to an empty claim list — so without this a cut-off answer produces the same receipt as a
  turn that held no facts and the turn is never extracted again. Call it *after*
  `record_usage`, because a truncated call generated every token it is billed for and
  `WritePipeline` publishes the usage of a call that raised. A reason your backend cannot
  read must not count as a truncation; guessing turns a working extraction into a failed
  write.
- `_SCHEMA_NAMES` is keyed on the identity of the module-level schema dicts, so pass
  `_call(..., name=...)` explicitly for any schema you build at runtime. A copy falls
  through to `"result"`, which the API accepts, so there is nothing to notice.
- Validate and coerce the model's output before returning: drop claims with a missing or
  out-of-range `source_index`, clamp `confidence` to `[0, 1]`, and normalize predicates to
  snake_case. The engine trusts these dicts, so this is the trust boundary.

### The `Multimodal` protocol

```python
@runtime_checkable
class Multimodal(Protocol):          # memvara/llm/base.py
    def describe_image(self, data: bytes, mime: str) -> str
    def transcribe(self, data: bytes, mime: str) -> str
```

It is a separate protocol, like `Chat` and `ReplacementJudge`, so a backend written before
it still passes `isinstance(backend, LLM)`. A backend that cannot handle a media type
raises `memvara.ingest.MediaUnsupported` with the reason, **before** it sends a request,
so the caller gets a sentence rather than a provider's 400. `AnthropicLLM` describes JPEG,
PNG, GIF and WebP images up to 5 MB and refuses every `transcribe` call, because the
Messages API takes no audio. `OpenAILLM` describes the same image types up to 20 MB through
Chat Completions, and transcribes through `client.audio.transcriptions.create` with
`transcription_model` (default `whisper-1`, because every OpenAI-compatible transcription
server accepts that name). The transcription endpoint reads the format from the file name,
so the file is sent as `upload.<extension>` from the `AUDIO_TYPES` table; MP4, MPEG and WebM
video are in that table because the endpoint transcribes their audio track. Video frames
are not sampled: that needs a decoder, and no extra ships one.

The prompts are `DESCRIBE_IMAGE_SYSTEM` and `DESCRIBE_IMAGE_PROMPT` in `base.py`. The
description replaces the image in memory, so the prompt asks for the text in the image
first and then what it shows.

---

## `memvara/ingest/`

One entry point, `extract(content | url, mime, ...) -> Extracted(text, title, mime,
pages)`, and one module per source type: `plain`, `html_text`, `pdf`, `media` and `url`.
`_mime` parses a `Content-Type` value and recognises file signatures; `errors` holds
`IngestError` and its codes. The document store calls `extract` before chunking.

- **Choosing the reader.** The caller's `mime` wins, then the server's `Content-Type`, then
  the first bytes. `application/octet-stream` counts as no type. Bytes that match no
  signature must decode as UTF-8, or they are refused as `media_unsupported`; a `str` with
  no type is text, or HTML when it starts like a page.
- **Every failure is an `IngestError` with a `code`.** A server can put the code in its
  response unchanged. An empty result is `no_text`, never an empty `Extracted`, because a
  document with no text cannot be chunked or found and a silent empty one looks stored.
- **`pypdf` is imported inside `pdf_pages`** (invariant 5). Without it a PDF is refused
  with `media_unsupported` naming `memvara[ingest]`, not with an `ImportError`, because
  the upload is what cannot be read and a server should answer it like any other.
- **The network is reached only through `url.SafeFetcher`, and only when the caller passes
  `url=`.** `import memvara.ingest` opens nothing, so invariant 5 holds. The fetcher's
  rules, in the order they run on every hop: the scheme is `http` or `https`; the host is
  resolved and refused if *any* address is not public, with IPv4-mapped, NAT64, 6to4 and
  Teredo forms unwrapped to their IPv4 address first; the socket connects to the checked
  address while `Host` and TLS use the name, so DNS rebinding cannot move the request;
  redirects are followed by hand, at most `MAX_REDIRECTS` (5), each one checked again;
  one deadline of `TIMEOUT_SEC` (20 s) covers every hop, and each read's socket timeout is
  set to the time left; the body stops at `MAX_BYTES` (10 MB). The transport, resolver
  and clock are constructor arguments so `tests/test_ingest_url.py` needs no network and
  no sleep.
- **NAT64 prefixes.** A NAT64 address carries its IPv4 address in bytes that depend on the
  prefix length (RFC 6052): after a /96 it is the last four bytes, and after a /32 to /64
  it starts right after the prefix, skipping byte 8. `_nat64_ipv4` reads it that way.
  `NAT64_PREFIXES` holds the well-known `64:ff9b::/96` and the RFC 8215 local-use
  `64:ff9b:1::/48`; `SafeFetcher(nat64_prefixes=...)` adds an operator's own, and the MCP
  server reads those from `MEMVARA_NAT64_PREFIXES` in `ServerConfig` and hands them over
  through `ServerConfig.url_fetcher()`. A prefix of any other length is refused, because no
  layout is defined for it. An address behind a prefix nobody listed looks like an ordinary
  global IPv6 address, so it cannot be caught; `docs/LIMITATIONS.md` says so.
- **The switches.** `allow_urls` and `allow_media` are the `ingest_urls` and
  `ingest_media` features in `memvara/server/config.py`'s `FEATURES`. `extract` does not
  read the environment itself; the caller passes them.

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

**Agentic capture: the model proposes, the hook applies.** With the `agentic_capture`
switch on (the default), and only on a host whose first extractor is `claude`,
`lib/agentic.py` runs the headless agent command with read-only access to the user's
memory before anything is written. The command connects to one MCP server, the store this
hook writes to, through a config file written for the run (owner-only, deleted after):
the hosted endpoint with the hooks' own API key and project header, or the client's own
local server block. The run's searches are plain reads: the local server is started with
`MEMVARA_FEATURE_QUERY_REWRITE=0` and `MEMVARA_FEATURE_SYNTHESIS=0`, and the hosted config
sends `Memvara-Read-Stages: plain`, which the hosted service does not read yet. The hosted
config also sends `Memvara-Capture-Run` with a new random id per run (16 hex characters),
so that on the hosted service one capture turn counts as one recall, however many searches
it makes; that takes effect once the hosted service supports the header. The id is never
written to `capture.log`, and a local run sends nothing like it. A hook that
is killed never reaches the `finally` that deletes the config, so every capture and every
session start delete any `capture-mcp-*.json` in the runtime directory older than twice the
run's timeout, and log how many they removed. Only `memory_search`, `memory_recall`, `memory_why` and
`memory_profile` are in the model's context; every other memvara tool is denied by name,
no built-in tool is available, and `--permission-mode dontAsk` refuses anything not
allowed. The hook reads the command's event stream as it runs and stops it after four tool
calls or 60 seconds. The model returns JSON proposals of four kinds: a new fact, a
supersede of a claim id with a new value and a reason, an end of a claim id with a reason,
and a link (`extends` or `derives`) between two claims. Each proposal passes the same
checks as a fact from the single-call extractor (`extract.vet`), and three more: a claim id
must have appeared in a tool result during this run, the object must not repeat the
extractor's own rules, and it must come from the new turn rather than from the earlier
turns the model was shown for reference. Accepted proposals are written with
`remember` (with `replaces` and `reason` for a supersede), `delete(close="ended")` or
`memory_end`, and `link` or `memory_link`, so the reconciler still decides duplicates and
conflicts. `expires_at` is passed only when the store's `remember` takes it. A reply that is
not a proposal list writes nothing, and the turn still counts as mined. A run that cannot
use the store, fails, times out or goes over the search limit falls back to the
single-call extraction for that turn, with a `capture.log` line saying why; the
single-call path raises the capture alert if it fails as well. The capture hook's timeout
on Claude Code is 180 seconds to cover both runs.

**The project a hook works in reaches the server as a header, carried by the environment.**
`lib/project.py` turns the session's directory into a project: `host/owner/repo` from the
`origin` remote, which every worktree of a repository shares, or `path:` and 16 hex
characters of a hash of the main repository's root when there is no usable remote (the
root is put in one spelling first, with forward slashes, a lower-case drive letter and no
trailing slash, so Windows and POSIX agree with each other and with the library), or
nothing outside a repository. A remote that normalises to a name the server would refuse,
such as one with a `..` segment, falls back to the path form, as the library's copy does.
Output from git that is not UTF-8 gives nothing, rather than an exception that would fail
the turn. It is a copy of the
library's `memvara/project.py`, because the hooks cannot import the library, and the rules
both copies must agree on are data in `lib/project_vectors.json`. Each hook calls `bind(cwd)`
once, which puts the value in `MEMVARA_HOOK_PROJECT`. `lib/hosted.py` sends it as the
`memvara-project` header on every call (header names are case-insensitive, so this is the
`Memvara-Project` header the design names), `lib/ipc.store_key` makes it part of the daemon's
address on a hosted install, and a daemon the hook spawns inherits it. The environment is
the channel because the per-prompt path must not import `lib/hosted.py`, and because a
spawned process gets it with no extra plumbing. Working a project out costs one `git`
process for a repository with a remote and two for one without, about 15ms each, against a
per-prompt budget of about 30ms. So the answer is cached for an hour in
`~/.memvara/.hooks/projects/`, one small file per directory, named by a hash of its path.

**Every memory line the hooks inject starts with `⋈ `, and capture drops those lines.**
`lib/mark.py` puts the mark in front of each bullet: `- billing uses postgres` is injected as
`⋈ - billing uses postgres`. The recall hook's dedup hash is taken over the line without the
mark, so a session running across the upgrade keeps its record of what it has seen.
`lib/transcript.py` removes every line that starts with the mark before a turn is mined, and
still drops a text containing one of the block headers whole. The line rule catches marked
lines quoted back without their header, which the header rule cannot see.

**Three counters per session feed a status line.** `lib/counts.py` keeps
`~/.memvara/.hooks/counts/<session>.json` with `recalled` (memory lines the recall hook
injected), `searched` (read-only memory tools the pre-tool hook approved) and `captured`
(facts capture stored, counted after the write succeeds). The session-start hook removes
files older than 14 days, once per session. `read()` needs nothing else from the tree,
because the status-line script in the plugin repository vendors that one file and must
finish in under 50ms.

**Every hook state file goes through `lib/state_file.py`.** The recall hook's per-session
state, the counters, the project cache and the capture alerts all need an atomic write (a
temporary file renamed over the real one), and the first two also need a read-modify-write
under an exclusive lock (`fcntl` on POSIX, `msvcrt` on Windows), because two hooks for one
session can run at once. Without the lock the second writer replaced the first: a lost
count, or a seen-set missing the
memories another prompt had just injected. The module also prunes files by age. Nothing in
it raises, including for a path with a NUL byte, which makes `os` calls raise `ValueError`
rather than `OSError`.

**Each of those three features has a switch.** `lib/settings.py` reads
`~/.memvara/settings.json`, a flat object of `feature_name: true|false`, where a missing key
means the feature's default. `MEMVARA_FEATURE_<NAME>=0|1` overrides the file. The names are
`project_scope`, `status_line` and `recall_mark`, all on by default. Two more are on by
default: `agentic_capture`, described above, and `query_rewrite`, described below. The
hooks know the same
feature names and defaults as `ServerConfig`, kept as a copy in
`lib/settings.FEATURE_DEFAULTS` that a test compares with the library's. The file is read at
most once per process. Capture drops
marked lines whatever `recall_mark` says, because a transcript can hold lines injected before
the switch changed.

**The recall hook asks for a query rewrite only after setup verified a key.** A local
store whose model can chat rewrites every read by default (invariant 1 names `query_rewrite`
as one of the read path's model stages), and on the recall hook that is one model call per
prompt. `lib/read_model.allowed()` says yes only when the `query_rewrite` switch is on, the
state file `~/.memvara/.hooks/read_model.json` records that `/memvara:setup verify-key`
made one test rewrite through the library and the model answered it (outcome `applied`),
and the configured `MEMVARA_LLM` and `MEMVARA_LLM_MODEL` are the ones that were checked. A
rewrite that reports `key_rejected` during a recall marks that record failed. Every other
read from the hooks is plain: `lib/fast.read_kinds` decides once per store whether its
`recall()` takes `query_rewrite`, and a library released before query rewrite, which never
rewrites, is asked without it. A rewritten read gets `lib/fast.REWRITE_WAIT_SEC` (5
seconds), because the model call's own deadline is 10 seconds and so is the hook's
allowance. After that the plain read is served from the same handle while the abandoned
thread may still be reading it, which is safe because `SQLiteStore` gives each thread its
own reader connection. When a daemon took a rewrite and did
not serve it in time, the fallback read is plain, so one prompt is never billed twice. The
daemon runs a rewritten read outside its lock and keeps plain reads under it, for the
hosted client's one connection. The hooks' hosted client always asks for a plain read: a
hosted server would rewrite with the organisation's key, which the user's machine cannot
check. The daemon's address, the store the hooks open and the rewrite decision all read
the client's configuration through `lib/ipc.client_env`, which reads the file once per
process.

**The hooks' hosted client sends each recall once.** The hosted service counts every
`memory_recall` it answers against the plan's recall allowance, and that includes a call
the tool refused because of an argument, since the refusal is a tool result inside an
HTTP 200. So `lib/hosted.HostedRecall.recall` checks `query_rewrite`, `min_score` and
`include_episodes` against the server's `tools/list` schema before the call, and leaves
off any argument the server does not declare. The schema is fetched once per client and
kept; a fetch that fails is not kept, and the next call asks again. Only when the fetch
failed does the client fall back to resending: first without `query_rewrite` when the
refusal names it, then without `min_score`, then without `include_episodes`, and only
while the tool itself is the one refusing (status 200). A 429, a 402, a server error or
no reply is raised at once, because none of them is about an argument. A recall that goes
out without its `min_score` floor writes a line containing `UNFILTERED` to `recall.log`.

**The recall hook tells a spent allowance from a failure.** The hosted service refuses a
recall over a paid plan's daily allowance with HTTP 429 and code `rate_limited`, with
`Retry-After` set to the seconds until the allowance resets and a `detail` holding
`metric`, `limit`, `used`, `resets_at` and `reason: over_period_allowance`. A plain rate
limit is also 429 `rate_limited`, but its `detail` names the `rule` that bound. Free's
monthly allowance is refused with 402 `quota_exhausted` and `detail.resets_at`, and no
`Retry-After`. `lib/fast._reason` turns the first into the token `daily` and the last into
`quota`, and `recall._quota_line` turns the tokens into the banner: "today's recall
allowance is used up — resets in 3 h 30 min" (or "resets at 00:00 UTC" when there was no
`Retry-After`), and "retrieval quota spent — resets 1 Oct". Every other failure, including
a plain rate limit, is "recall failed".

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
