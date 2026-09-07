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
- Temporal expressions: `memvara/write/when.py` — `resolve()` turns "last March" into an
  instant and a `Precision`.
- Model backends: `memvara/llm/base.py` — the `LLM` and `Chat` protocols, `ReplacementJudge`,
  `NullLLM`, `Usage`, `TruncatedResponse`, `bounded_claim_schema()`. Implementations are
  `memvara/llm/anthropic.py` (`AnthropicLLM`) and `memvara/llm/openai.py` (`OpenAILLM`).
- Tests: `tests/test_pipeline.py`, `tests/test_gate.py`, `tests/test_fast.py`,
  `tests/test_reconcile.py`, `tests/test_pollution.py`, `tests/test_when.py`,
  `tests/test_llm.py`, `tests/test_advisory.py`.
- Documentation: [INTERNALS.md](../INTERNALS.md), section *`memvara/write/`*, which carries
  the tier-by-tier contract and the measured constants.

## How the pieces fit

`WritePipeline.add()` runs three tiers in order.

1. **Tier 0 needs no model.** The episode is stored, and an episode whose content hash is
   already present is skipped. Surviving episodes are embedded and checked against existing
   claim embeddings; a cosine at or above `near_dup_threshold` (0.97) reinforces the claim
   that is already there instead of writing a second one.
2. **Tier 1 needs no model.** `SalienceGate` drops turns that carry no durable fact and
   counts them on `receipt.skipped`. What survives goes to `FastExtractor`, which emits a
   claim only for a form it recognises with confidence and emits nothing otherwise.
3. **Tier 2 is the only tier that calls a model.** The turns that passed the gate and
   produced no fast-path claim are batched into one `llm.extract()` call. A predicate the
   registry has not seen costs one `llm.resolve_predicate()` per new surface form, and the
   answer is learned, persisted through `store.put_spec()` and never asked again, including
   after a restart and including by another process.

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

- **The gate biases toward recall.** A false positive costs one extraction call; a false
  negative loses a memory permanently. `SalienceGate.DEFAULT_EVIDENCE_ROLES` is the user
  role alone, and passing `evidence_roles=None` is the documented way to handle a transcript
  between two named people, where the default would otherwise drop every turn.
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

## Read next

[INTERNALS.md](../INTERNALS.md)'s `memvara/write/` section gives the exact tier contract,
the measured grounding constants and the four reconcile outcomes with their receipt actions.
The docstrings in `memvara/write/pipeline.py` and `memvara/write/reconcile.py` explain the
decisions at the point where they are made.

Next: [retrieval](retrieval.md).
