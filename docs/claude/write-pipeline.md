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

1. **Tier 0 needs no model.** The episode is stored, and an episode whose content hash is
   already present is skipped. Surviving episodes are embedded and compared with the
   nearest live claim. A turn whose cosine reaches `near_dup_threshold`, and which holds
   the same numbers as the claim's text, is a restatement: the claim is reinforced and
   nothing is extracted from the turn. The threshold defaults to the merge threshold
   measured for the embedder's space in `memvara/embed/calibration.py`: 0.985 for
   all-MiniLM-L6-v2, 0.99 for bge-small-en-v1.5 and 0.97 for any other embedder. A turn
   worded like a claim but holding another number, such as "user has appointment on
   2023-05-02" against the claim for 2023-05-01, states another value and goes on to
   extraction.
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
memory can carry an expiry the turn names. The proposals write nothing. A proposed memory goes through the same guards as single-call output and
then `Reconciler.apply()`; a proposed end becomes a retraction with `close="ended"`; a
proposed link becomes a `claim_links` row. A proposal naming a memory the model did not
read in the run, or asking to end or replace one in a broader scope than the write, is
refused, and every refusal is on
`receipt.proposals_refused`. A backend without tools, a timeout, an answer that cannot be
used, a run past 12 steps, a request that fails twice or a batch from two scopes sends the
batch to the single call, and `receipt.agentic_fallback` says which. Off by default,
because its release bar in the "Reversed" list of `docs/ROADMAP.md` has not been measured.

Every claim that reaches the store passes through `Reconciler.apply()`, which decides one of
four outcomes against the claims already in that slot: exact duplicate (do not insert),
conflict (the predicate holds one value, so the incoming claim supersedes the old one),
retraction (the incoming claim has `polarity == -1`, so matching live claims are closed out),
or accumulate (insert alongside). `Memvara.supersede()` is the explicit form of the second
outcome for a caller who already knows which claim is being replaced.

**Replacement advice is separate from all of this and closes nothing.** When a `Memvara` was
built with `advise_replacements=True`, when the backend implements `ReplacementJudge`, and
when a write added a claim without closing one, `Memvara.remember()` asks the judge about up
to `ADVISORY_CANDIDATES` (3) of the nearest live claims in *other* slots and fills
`WriteReceipt.may_replace` with what comes back. Each consultation is counted in
`llm_calls`. A judge that raises warns once per instance and leaves the list empty, because
the claim is already durable and a suggestion must not turn a completed write into an
exception the caller retries.

## Invariants and assumptions

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
