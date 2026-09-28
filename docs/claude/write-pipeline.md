# The write pipeline

Everything that puts a fact into the store runs through `memvara/write/`. The pipeline is
built so that the cheap, deterministic work happens first and a model is consulted only for
the turns nothing else could handle. That ordering is what makes the library usable with no
API key at all, and it is why a write costs a small number of model calls rather than one per
turn.

A write starts as an `Episode` — one raw turn, stored verbatim — and ends as zero or more
`Claim` rows that cite it, plus a `WriteReceipt` saying what happened.

## Where the code is

- Primary: `memvara/write/pipeline.py` — `WritePipeline`, with `add()`, `reextract()` and
  `assert_claim()`. This runs the tiers and fills in the receipt.
- Triage: `memvara/write/gate.py` — `SalienceGate.carries_fact()` returns
  `(should_extract, reason)`, where the reason is a short slug that shows up in receipts and
  in tests.
- Deterministic extraction: `memvara/write/fast.py` — `FastExtractor.extract()`, pattern
  matching for a handful of common, unambiguous sentence forms.
- Contradiction handling: `memvara/write/reconcile.py` — `Reconciler.apply()`,
  `Reconciler.reinforce()`, plus the backfill helpers `backfill_entities()` and
  `backfill_predicates()`.
- Fabrication guard: `memvara/write/pollution.py` — `guard()`, applied only to
  model-proposed claims.
- Cutting a long turn for extraction: `memvara/write/split.py` —
  `split_for_extraction()` and `EXTRACTION_CHUNK_CHARS`, used only when
  `extraction_chunks` is on.
- Agentic extraction: `memvara/write/agentic.py` — `AgenticExtractor`, `AGENTIC_SYSTEM`,
  the proposal types and `ProposalPlan`, used only when `agentic_extraction` is on. The
  tool loop both backends share is `memvara/llm/_tools.py`.
- Temporal expressions: `memvara/write/when.py` — `resolve()` turns "last March" into an
  instant and a `Precision`.
- Model backends: `memvara/llm/base.py` — the `LLM`, `Chat` and `ToolChat` protocols,
  `ReplacementJudge`, `Multimodal` (images, audio and video to text, used by
  `memvara/ingest/`), `NullLLM`, `Usage`, `TruncatedResponse`, `bounded_claim_schema()`. Implementations are
  `memvara/llm/anthropic.py` (`AnthropicLLM`) and `memvara/llm/openai.py` (`OpenAILLM`).
- Per-project extraction guidance: `memvara/llm/guidance.py` — `Guidance`,
  `with_guidance()` and `load_guidance()`. `WritePipeline.guidance` holds it, and every
  extraction call appends it to its system message. Tests in
  `tests/test_extraction_guidance.py`.
- Tests: `tests/test_pipeline.py`, `tests/test_gate.py`, `tests/test_fast.py`,
  `tests/test_reconcile.py`, `tests/test_pollution.py`, `tests/test_when.py`,
  `tests/test_llm.py`, `tests/test_advisory.py`, `tests/test_extraction_chunks.py`,
  `tests/test_agentic_extraction.py`.
- Documentation: [INTERNALS.md](../INTERNALS.md), section *`memvara/write/`*, which carries
  the tier-by-tier contract and the measured constants.

## How the pieces fit

`WritePipeline.add()` runs three tiers in order.

1. **Tier 0 needs no model.** The episode is stored, and an exact repeat of an earlier
   turn is skipped. The content hash covers the scope, the role and the text, not the
   time, so tier 0 compares the turn with the latest copy of it dated at or before the
   turn. The turn is a repeat, and reinforces that copy's live claims, when the claims
   extracted from the copy have not all ended by the turn's time. Otherwise it is a new
   statement, stored and extracted like any other turn. So "I live in Berlin." said again
   after "I moved to Paris." makes Berlin current again (#332), a copy dated before every
   earlier copy keeps its earlier date (#318), and a retry or a replay of a turn still
   converges on the rows it wrote the first time. A turn whose copy produced no claim,
   or only claims that were retired since, stays a repeat. Surviving episodes are
   embedded and compared with the nearest live claim. A turn whose cosine reaches
   `near_dup_threshold`, and which holds the same numbers as the claim's text, is a
   restatement: the claim is reinforced and nothing is extracted from the turn. The
   threshold defaults to the merge threshold measured for the embedder's space in
   `memvara/embed/calibration.py`: 0.985 for all-MiniLM-L6-v2, 0.99 for
   bge-small-en-v1.5 and 0.97 for any other embedder. A turn worded like a claim but
   holding another number, such as "user has appointment on 2023-05-02" against the
   claim for 2023-05-01, states another value and goes on to extraction. A restatement
   dated before the claim begins is not a repeat either (#318): tier 0 sends the claim's
   value, dated to the turn, to the reconciler, which stores it for the earlier period as
   it does for `remember()`.

   The two cases cost different amounts. A turn said again after its value ended is
   extracted again, so a turn that needed the model the first time costs a model call
   again. Its copy's value has ended, so the turn is read as a new statement rather than
   rebuilt from the old claims. A near-duplicate dated before its claim costs no call,
   because tier 0 already knows which claim the turn restates and only the date is new.
2. **Tier 1 needs no model.** `SalienceGate` drops turns that carry no durable fact and
   counts them on `receipt.skipped`. What survives goes to `FastExtractor`, which emits a
   claim only for a form it recognises with confidence and emits nothing otherwise.
3. **Tier 2 is the only tier that calls a model.** The turns that passed the gate and
   produced no fast-path claim are batched into one `llm.extract()` call. A predicate the
   registry has not seen costs one `llm.resolve_predicate()` per new surface form, and the
   answer is learned, persisted through `store.put_spec()` and never asked again, including
   after a restart and including by another process. With `extraction_chunks=True`, a turn
   over 6,000 characters is the exception to the one call: it is cut into pieces at
   paragraph and sentence ends and each piece is its own call. The claims still cite the
   whole episode, and a fact two pieces both state reaches the reconciler twice, as it
   would if the turn had stated it twice. The option is off by default, because the release bar recorded in the
   "Reversed" list of `docs/ROADMAP.md` has not been met.

**With `agentic_extraction=True`, tier 2 is a tool loop instead of one call.** When the
backend implements `llm.ToolChat`, `AgenticExtractor` gives the model six tools: it can
search what this write's scope can see, read one memory by id, and propose a new memory,
the end of a stored memory, a replacement for one, or a link between two. The system
message carries the project's extraction guidance when there is some, and a proposed
memory can carry an expiry the turn names. That expiry stays on the claim the proposal
creates; a proposal that repeats a claim on record reinforces it without touching its
expiry, because only a caller's repeat moves an expiry, and single-call output, which is
offered no expiry, has any it returns dropped. The proposals write nothing. A proposed memory goes through the same guards as single-call output and
then `Reconciler.apply()`; a proposed end becomes a retraction with `close="ended"`; a
proposed link becomes a `claim_links` row. A link from a proposal that restated a live
value with an earlier start lands on the live claim on record, not on the claim for the
earlier period (`ReconcileResult.restated`). A proposal naming a memory the model did not
read in the run, or asking to end or replace one in a broader scope than the write, is
refused, and every refusal is on
`receipt.proposals_refused`. A backend without tools, a timeout, an answer that cannot be
used, a run past 12 steps, a request that fails twice or a batch from two scopes sends the
batch to the single call, and `receipt.agentic_fallback` says which. Off by default,
because its release bar in the "Reversed" list of `docs/ROADMAP.md` has not been measured.

Every claim that reaches the store passes through `Reconciler.apply()`, which decides one of
four outcomes against the claims already in that slot: exact duplicate (do not insert,
unless the incoming claim begins before every live claim with its value that the writer
can see, in which case it is inserted for that earlier period only and ends where the
value's stored claims begin, or reinforces a stored claim that already holds that
period),
conflict (the predicate holds one value, so the incoming claim supersedes the old one),
retraction (the incoming claim has `polarity == -1`, so matching live claims are closed out),
or accumulate (insert alongside). An incoming claim that is already over, because the
caller gave its end or because a later value ends it, is also an exact duplicate when a
stored claim of its value that the writer can see holds its whole period; that claim is
reinforced and nothing is inserted.

A retraction is stored as a tombstone, closed on both clocks at the instant its write is
recorded, which is the given `recorded_at` for a backdated write, so no read at any
instant returns it. It faces the same authority rule as a new value (`AUTHORITY_SHARE`):
a match it is worth less than half of stays live and is reported as a `Dispute` with
`retraction=True`. It closes only a match it changes, so a second copy of a retraction
dated in the future closes nothing, reinforces the tombstone on record and reports
nothing ended. `Memvara.supersede()` is the explicit form of the second
outcome for a caller who already knows which claim is being replaced. In a conflict, the old
values are every value believed and neither retired nor ended yet, including one written to
begin later, that is true at some instant the incoming claim is true; `docs/INTERNALS.md` has
the three rules.

**An exact duplicate counts only when the writer can see it.** `value_key` leaves out the
project, agent and session, so the duplicate lookup also finds the same value in a sibling
project or session. That claim is left alone, and the repeat is stored in the writer's own
scope; otherwise the writer would read nothing back. INTERNALS, "A repeat reinforces only a
claim its writer can see", has the rule and the two other paths that follow it.

**Replacement advice is separate from all of this and closes nothing.** When a `Memvara` was
built with `advise_replacements=True`, when the backend implements `ReplacementJudge`, and
when a write added a claim without closing one, `Memvara.remember()` asks the judge about up
to `ADVISORY_CANDIDATES` (3) of the nearest live claims in *other* slots and fills
`WriteReceipt.may_replace` with what comes back. Each consultation is counted in
`llm_calls`. A judge that raises warns once per instance and leaves the list empty, because
the claim is already durable and a suggestion must not turn a completed write into an
exception the caller retries.

## Invariants and assumptions

- **A queued reinforcement reads its claim again before it writes.** Tier 0 finds the
  claims a repeated or restated turn supports, but the claim transaction reinforces them
  only after tiers 1 and 2. It reads each claim again there, under the write lock, and
  leaves alone one that another writer erased, ended or retired in the meantime, so the
  reinforcement cannot undo that change or bring an erased claim back.
- **Nothing read from a turn erased during extraction is written.** The claim transaction
  of `add()` and `reextract()` reads again which of its turns are still stored
  (`_erased`). It drops a candidate, and a reinforcement tier 0 queued, whose every turn is
  gone, cites only the stored turns from everything else, and refuses an agentic end read
  from a gone turn, so a purge or a document delete that lands during a model call is not
  undone by the claims that call produced.
- **The backfills write back only onto slots nobody changed.** `backfill_entities()`,
  `backfill_predicates()` and `split_entity()` read their claims, decide, and write at the
  end. `_write_back` reads the rows again under the write lock and writes a slot's claims
  only if every one of them is still exactly as the pass read it, so a pass cannot undo
  another writer's change or bring an erased claim back. A slot it leaves is applied by
  the next run.
- **A document chunk passes the role check.** `add_document()` stores each chunk as a
  system turn with `meta["document_id"]`, and the gate accepts it whatever its role,
  because the caller asked for the document to be read. The fast path still reads user
  turns only, so a chunk's facts come from the model tier. A chunk stored with
  `extract=False` carries `meta["extract"] = False` and the gate refuses it.
- **The gate biases toward recall.** A false positive costs one extraction call; a false
  negative loses a memory permanently. `SalienceGate.DEFAULT_EVIDENCE_ROLES` is the user
  role alone, and passing `evidence_roles=None` is the documented way to handle a transcript
  between two named people, where the default would otherwise drop every turn.
- **A closed vocabulary refuses what nothing declared.** `closed_vocabulary=True` drops a
  model-proposed claim whose predicate the registry cannot resolve, counts it on
  `receipt.unregistered`, and never spends a model call learning it. Off by default; on for
  a deployment whose predicates are deliberate, because an unregistered predicate is
  multi-valued forever and supersedes nothing, so all it adds is noise in every recall.
- **Guidance adds to the extraction prompt and never replaces it.** It is appended to the
  system message under a fixed heading, after the shipped rules or a replacement prompt,
  and never goes in the user message beside the turns. It is refused rather than cut when
  it is too long, and refused for a backend that does not set `accepts_guidance`.
- **The fast path chooses precision over recall.** It emits nothing rather than a wrong
  triple, sets `Derivation.FAST_PATH`, and leaves the rest to tier 2.
- **`reject_ungrounded` defaults to `"auto"`.** A model-proposed claim whose object shares
  no content word with the episode it cites is a fabrication candidate; the embedder gets a
  veto, and a claim that fails both is refused and counted on `receipt.ungrounded`. The
  default is on rather than off because the destructive direction is storing: a fabricated
  value in a one-value slot supersedes and ends the true fact that was there. Only
  model-proposed claims are checked — `remember()` and the fast path do not go through that
  code.
- **A truncated model response fails the write.** `AnthropicLLM` and `OpenAILLM` read the
  provider's own stop reason and raise `TruncatedResponse`, because a cut-off JSON object
  parses to an empty claim list and would otherwise be indistinguishable from a turn that
  genuinely held no facts.
- **`WriteReceipt` must be fully populated, including `llm_calls`.** It is zero whenever no
  model was consulted, and it counts predicate acquisition as well as extraction.
- **Extraction that cannot run is reported, and how depends on the deployment.**
  `extraction_deferred=False` counts the batch on `receipt.unextracted`, which says the
  content was accepted and nothing will ever extract from it. `extraction_deferred=True`
  counts it on `receipt.deferred` instead, for a deployment where a worker calls
  `reextract()` later. The option changes what is said, not what is stored.
- **A model proposes and the reconciler applies.** Under `agentic_extraction`, nothing a
  tool does writes to the store. Every change the write makes is one of the reconciler's
  outcomes or a link row, a model can end a memory but never retire or erase one, and a
  replacement the reconciler does not accept leaves the old memory live. The rules travel
  as the system message and the turns as fenced data, and a proposed memory that restates
  the rules is refused, so a turn quoting the extractor's prompt cannot become memories.
- **A long turn is extracted whole or not at all.** Under `extraction_chunks`, if the call
  for any piece of a turn fails, that turn keeps no claim from its other pieces and is
  deferred. `reextract()` skips any turn that already has claims, so keeping part of a
  turn's claims would mean the missing piece is never read. The other turns in the batch
  keep their claims.

## Read next

[INTERNALS.md](../INTERNALS.md)'s `memvara/write/` section gives the exact tier contract,
the measured grounding constants and the four reconcile outcomes with their receipt actions.
The docstrings in `memvara/write/pipeline.py` and `memvara/write/reconcile.py` explain the
decisions at the point where they are made.

Next: [retrieval](retrieval.md).
