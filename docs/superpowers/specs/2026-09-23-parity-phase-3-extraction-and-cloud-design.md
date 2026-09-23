# Parity phase 3: extraction and cloud

**Date:** 2026-09-23. **Status:** design approved in conversation on 2026-09-23, not built.
Line numbers are from agent-memory `origin/main` at `b57ef714` (**core**) and memvara-cloud
`origin/main` at `c1ab3ab` (**cloud**). Phases 1 and 2 are
`2026-09-23-parity-phase-1-coding-agent-experience-design.md` and
`2026-09-23-parity-phase-2-documents-and-retrieval-design.md`.

## 1. What this phase delivers, and which decisions it reverses

| # | Feature | Reverses |
|---|---|---|
| 1 | Agentic extraction: a tool-using model loop that proposes changes | invariant 1's claim that contradiction resolution is a pure function of stored state (`docs/INTERNALS.md:34-37`) |
| 2 | Per-project extraction guidance added to the shipped prompt | — |
| 3 | Project management: settings, description, rename, merge | — |
| 4 | Erasure when a memory's expiry date passes | invariant 3, "Nothing is ever hard-deleted by the engine" (`docs/INTERNALS.md:127-139`), for this one case |
| 5 | Connectors for Notion, Google Drive and GitHub | — |
| 6 | Per-project feature settings and a console settings screen | — |

Decisions taken on 2026-09-23: all on by default with a switch; in agentic extraction the model
**proposes** and the existing reconciler **applies**; a merged project's source is deleted after
a verified merge. Each reversal gets a Confluence decision page and rewrites the text it
reverses in the same commit.

Switch names: `agentic_extraction`, `extraction_guidance`, `project_descriptions`,
`expiry_erasure`, `connectors`.

## 2. Constraints every workstream keeps

- Phases 1 and 2's constraints apply.
- **A route the console mocks is a route the server owes** (cloud `CLAUDE.md:135-139`). Every new
  screen ships with its real route in the same pull request; anything that cannot be finished
  ships visibly disabled as a 501 with a reason.
- **`deploy/memvara_deploy/settings.py` is the only place the cloud reads environment
  variables.**
- The cloud gate is `scripts/gate.sh` (Python 3.13, about 4 minutes); Playwright layout specs run
  with `npm run test:layout` for any changed screen.

## 3. The features

### 3.1 Agentic extraction

**Interface to the model.** A new runtime-checkable protocol in core `memvara/llm/base.py`,
separate so older backends keep passing `isinstance` checks (`base.py:143-150`):

```python
class ToolChat(Protocol):
    def run_tools(self, system: str, messages: list[Message], tools: list[ToolSpec],
                  *, max_steps: int, timeout: float) -> ToolRun: ...
```

Implemented on `AnthropicLLM` and `OpenAILLM` with each provider's native tool calling.

**The loop.** A new `memvara/write/agentic.py` with `AgenticExtractor`, used by tier 2 of the
write pipeline (`memvara/write/pipeline.py:714-792`) in place of the single `llm.extract` call
when `agentic_extraction` is on and the backend is a `ToolChat`.

- **Tools given to the model:** `search_memories(query, k)`, `get_claim(claim_id)` (read only),
  and four proposals: `propose_claim(subject, predicate, object, valid_from?, memory_type?)`,
  `propose_end(claim_id, reason)`, `propose_supersede(claim_id, subject, predicate, object,
  reason)`, `propose_link(from_ref, to_ref, relation)`.
- **At most 12 steps**, 8,192 output tokens per step, one retry.
- **Proposals are not writes.** The loop returns a list of proposals. They pass the existing
  pollution guard, closed-vocabulary filter and predicate acquisition (`pipeline.py:830-1001`),
  then go to the existing `Reconciler.apply` (`memvara/write/reconcile.py:247-273`), which decides
  duplicates, conflicts and supersession as today. `propose_end` becomes a close with
  `close="ended"` and the reason; it never erases. `propose_link` becomes a phase 1 `claim_links`
  row.
- **Instructions and content are separate messages.** The system message holds the rules; the
  conversation turns go in a user message wrapped in `<content>` and described as data. A test
  feeds the extractor a turn that *quotes its own instructions* and asserts that no claim
  paraphrases an instruction. This is the failure seen in Supermemory, where its memory agent
  stored its own prompt as twenty memories.
- **Fallback.** Without a `ToolChat` backend, on a timeout, or on malformed tool output, tier 2
  runs today's single-call extraction for that batch and records the outcome
  (`agentic_fallback`) on the receipt.
- **Hosted worker.** `deploy/memvara_deploy/extract.py` uses the same path; its model is the
  deployment's (`extractor.py:161` `build_llm`).

**Release bar.** Before the switch defaults on in a release: no fewer claims and no more
duplicates than today's extractor on `demo/harness.py`, and judged accuracy on the 199-question
LongMemEval sample within the reader noise floor of the current production number (the ROADMAP
records the floor). If the bar is not met, it ships switched off, visibly, with the numbers in
the pull request body.

**Invariant text.** Invariant 1 is rewritten to: contradiction resolution is decided by the
deterministic reconciler; a model may *propose* changes on the write path, and every applied
change is one of the reconciler's recorded outcomes.

### 3.2 Per-project extraction guidance

- Core: `LLM.extract(…, guidance: Guidance | None = None)` on both backends, where `Guidance`
  has `context: str` (at most 1,500 characters), `include: list[str]` and `exclude: list[str]`
  (at most 20 rules of 200 characters each). The guidance is **appended** to the shipped
  `EXTRACT_SYSTEM` (`memvara/llm/base.py:433-466`) under a fixed heading, and to the agentic
  system message. `AnthropicLLM` currently hard-codes the prompt (`anthropic.py:171`); both
  backends now accept the addendum. `MEMVARA_LLM_EXTRACT_SYSTEM` keeps its meaning (full
  replacement).
- Local: `MEMVARA_EXTRACT_GUIDANCE` (a path to a TOML file with the three fields).
- Cloud: columns `projects.extraction_guidance jsonb` (cloud `control/schema.py:1045-1062`), read
  in `ProjectMemories._clone` (`memories.py:289`) with the existing 60-second cache, passed to the
  extractor. Route `PATCH /projects/{project}/extraction-guidance`, member-gated.

### 3.3 Project management

- **Rename:** `PATCH /projects/{project}` with `name` (the slug stays; keys and URLs keep
  working).
- **Description:** `projects.description text`. Written by the model on the write path when
  `project_descriptions` is on and the project gains 50 new episodes since the last description;
  editable by a member, and an edited description is not overwritten until the member clears it.
- **Merge:** `POST /projects/{source}/merge-into/{target}` with `dry_run` (default true).
  - The dry run counts claims, episodes, documents, vectors and links to move, and conflicts
    (same fact key with different live values).
  - The real run is a background job recorded in `job_history`: it copies rows into the target
    tenant, re-keys `fact_key`/`value_key` with the existing re-key tools
    (`reconcile.py:932`, :1014), re-runs reconciliation on the target so conflicts resolve through
    supersession, writes an audit record on both projects, then verifies counts.
  - Only after verification does it delete the source project through the existing delete path
    (`control/api/app.py:5837`). A failed verification leaves both projects intact and reports
    the difference.
  - Owner-only, and both projects must be in the same organisation.
- **Console:** a Project settings screen (`p/:project/settings`) with name, description,
  guidance, the feature switches (§3.6) and merge.

### 3.4 Erasure on expiry

- **A new marker, separate from `valid_to`.** `valid_to` also closes superseded and ended claims
  (invariant 3), so erasing on `valid_to` would erase history. Claims gain `expires_at` (nullable,
  a column in both stores: SQLite schema v15, Postgres v5) and `expire_reason`.
- **Write:** `remember(…, expires_at=…, expire_reason=…)`; MCP `memory_remember` gains
  `expires_at` and `expire_reason`; REST `FactRequest` the same.
- **Sweep:** `Memvara.erase_expired(now=None) -> list[ErasedClaim]` erases every claim with
  `expires_at <= now` through the existing `erase()` path (core.py:1982) and records proof with
  `prove_erased` (:2046). Locally it runs at store open and hourly in the MCP server; in the cloud
  it is a new compose loop `expiry.py` shaped like `prune.py` (:193, :244), recorded in
  `job_history` and watched by `/admin/health`.
- **Invariant text.** Invariant 3 is rewritten to: the engine hard-deletes only claims that carry
  an explicit `expires_at`, only after it passes, and always with a proof record; ending and
  superseding never delete.
- The cloud's manual retention sweep stays as it is; its "deliberate, stated deferral" docstring
  is unchanged, because this feature is expiry per claim, not retention per organisation.

### 3.5 Connectors

- **Providers:** Notion (pages and databases), Google Drive (Docs, text, PDF), GitHub (issues,
  pull requests, markdown files in chosen repositories).
- **OAuth out.** New `memvara_cloud/connectors/` with one module per provider: authorisation URL,
  token exchange, refresh. Client ids and secrets come from `settings.py`. Tokens are stored per
  project in `connector_accounts(project, provider, account_label, token_ciphertext, wrapped_dek,
  scopes, cursor, status, last_sync_at, last_error)`, encrypted exactly as `org_selector_settings`
  is (AES-256-GCM under `idp_key`, customer-managed key wrapping; `control/idp/crypto.py`,
  `control/cmk/`).
- **Sync worker.** A compose service `sync` (`deploy/memvara_deploy/sync.py`, same shape as
  `extract.py`): round-robin over connected accounts, incremental with each provider's change
  cursor, writing through phase 2's document API with `custom_id = "<provider>:<item id>"` and
  `filepath` set to the item's path in the source. A deletion at the source becomes a document
  delete (phase 2 §4.1). Recorded in `job_history`.
- **Routes:** `GET /projects/{p}/connectors`, `POST /projects/{p}/connectors/{provider}/connect`
  (returns the authorisation URL), the OAuth callback, `POST …/{account}/sync`, `DELETE
  …/{account}` (disconnect; documents it created are kept unless `purge=true`).
- **Console:** a Connectors screen under the project.
- "On by default" means the screen and routes are available; nothing syncs until a user connects
  an account.

### 3.6 Per-project feature settings

- Table `feature_settings(project, feature, enabled, updated_at, updated_by)`; a missing row means
  the default (on). Routes `GET /projects/{p}/features`, `PATCH /projects/{p}/features`.
- Every phase 1–3 switch name is valid here; an unknown name is refused with 400.
- The cloud resolves switches per request through the same cached path as the category
  (`memories.py:160`, 60 seconds).
- Console: the switches appear on the Project settings screen, and the onboarding flow
  (`Welcome.tsx:58-68`) gains a step that lists them with their defaults.
- `/memvara:setup` (phase 1 §4.10) reads and writes the hosted switches through these routes when
  the plugin is connected to the hosted service.

## 4. Workstreams for the fan-out

| Stream | Repository | Features | Depends on |
|---|---|---|---|
| P3-A | core | 3.1 `ToolChat`, `AgenticExtractor`, fallback, instruction-leak test, invariant rewrite | phase 1 `claim_links` merged |
| P3-B | core | 3.2 guidance on both backends; 3.4 `expires_at`, `erase_expired`, invariant rewrite | phase 2 merged (schema order) |
| P3-C | cloud | 3.2 column and route; 3.3 rename, description, merge job; 3.6 table and routes | P3-B merged |
| P3-D | cloud | 3.4 expiry loop; 3.5 connectors and sync worker | P3-B merged, phase 2 P2-G merged |
| P3-E | cloud `dashboard/` | Project settings, Connectors screens, onboarding step, layout specs | P3-C and P3-D routes merged |
| P3-F | cloud `deploy/` worker and core `bench/` | the extraction worker switch and the release-bar runs for 3.1 | P3-A merged |
| P3-G | memvara-web | customer documentation for every phase 1–3 feature, customer-first per the house rule | each feature's pull request merged |

## 5. Out of scope for phase 3

Connectors beyond the three named providers, and encryption of the cloud database, which is an
operator task.
