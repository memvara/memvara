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
`expiry_erasure`, `connectors`, and `agentic_capture` for the plugin (section 3.7).

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

### 3.7 Agentic capture in the plugin (local)

Added on 2026-09-24, when stream P3-H was approved. Section 3.1 makes extraction agentic on
the write path of the library and the hosted worker. This section does the same for the
plugin's capture hook, which extracts on the user's own machine with the headless agent
command and the user's own login. The principle is the same: the model proposes, and the
deterministic write path applies.

**What the hook runs.** `plugin/hooks/lib/agentic.py` runs the headless agent command once
per mined turn, with read-only access to the store the hook writes to:

- The command connects to one MCP server, named `memvara`, from a config file the hook
  writes for the run and deletes afterwards (owner-only, in `~/.memvara/.hooks/run/`). On a
  hosted install the file names the endpoint with the hooks' API key and the
  `memvara-project` header, so the model searches the project the proposals will be
  written to. On a local install it is the client's own memvara server block, with this
  process's `MEMVARA_*` variables winning over it. The plugin's own `.mcp.json` entry is
  not used because it may be signed in through a browser, which a headless run cannot do.
- The run's searches are plain reads. The local server is started with
  `MEMVARA_FEATURE_QUERY_REWRITE=0` and `MEMVARA_FEATURE_SYNTHESIS=0`, set after the user's
  own variables, so a configured `MEMVARA_LLM` cannot turn each search into a model call.
  The hosted config sends `Memvara-Read-Stages: plain`. The hosted service does not read
  that header yet; the cloud side adds it, and until then a search from an organisation
  with a model key may be rewritten.
- On the hosted service, one capture turn counts as one recall against the plan's
  allowance, however many searches it makes (decided 2026-09-24). The hosted config sends
  `Memvara-Capture-Run` with a new random id per run, 16 hex characters from `secrets`, so
  the service can tell which searches belong to one run. This takes effect once the hosted
  service supports the header, which the cloud side adds; until then each search counts
  as a recall. The id is not logged. A local run needs nothing, since no allowance
  applies.
- A hook that is killed never reaches the `finally` that deletes the config file, which
  holds a credential. Every capture and every session start delete any
  `capture-mcp-*.json` in the runtime directory older than twice the run's timeout, and
  write a `capture.log` line saying how many they removed.
- `--strict-mcp-config` excludes every other MCP server the user has. `--tools ""` removes
  the built-in tools. `--allowedTools` lists `memory_search`, `memory_recall`, `memory_why`
  and `memory_profile`; `--disallowedTools` names every other memvara tool so it is not in
  the model's context; `--permission-mode dontAsk` refuses anything not allowed, including
  a tool the server adds later. A test compares the denied list with the server's tool
  table.
- `--setting-sources ""` loads no settings files and so no plugins or instruction files,
  and `--system-prompt` replaces the default system prompt with the rules. Together they
  are most of the cost difference measured below. `--no-session-persistence` keeps the
  run's transcript, which contains the user's turn, off disk.
- The step limit is `--max-turns 6`. The hook reads the command's `stream-json` events as
  they arrive and stops the run at the fifth tool call or after 60 seconds.

**Proposals.** The model returns `{"proposals": [...]}` with four kinds: `fact`;
`supersede` of a claim id with a new value and a reason; `end` of a claim id with a reason;
and `link` (`extends` or `derives`) between two claim ids, where `new:N` names the Nth fact
or supersede in the same reply. A fact or supersede may carry `"standing": false`, which
files it as `episodic`, and `expires_at`. Each proposal passes the checks a single-call
fact passes (`extract.vet`: closed vocabulary, empty and thin objects, values absent from
the turn, the user's own wording for a standing instruction, echoes of recalled notes and
of the search results the model read), and three more: every claim id must have appeared
in a read tool's result in this run, which the hook reads from the event stream rather
than from the model's text; the object must not repeat the rules; and it must come from
the new turn rather than from the earlier ones. Accepted proposals go through the hook's
existing writes: `remember`, `remember` with `replaces` and `reason`, `memory_end` (locally
`delete(close="ended")`), and `memory_link`. The reconciler decides duplicates and
conflicts as before.

**Context without re-reading.** Capture stays per turn. The model also sees up to 4,000
characters of the turns before, marked as already processed, inside the same data block.

**Instructions are never content.** The rules are the system prompt. The earlier turns
and the new turn are the user message, inside delimiters that carry a random value per run
and a first line saying the block is data. A test pastes the rules into a turn and has the
fake model restate them; every such proposal is refused.

**Expiry and standing facts.** `expires_at` is passed only when the store's `remember`
takes it: the hosted server's tool schema is asked, and a local library's signature is
read. Otherwise it is dropped with a log note. Supermemory's static flag has no new field:
a standing fact keeps its predicate's type (`procedural` is the set every session starts
with and the profile shows), and a fact marked not standing is filed as `episodic`.

**Fallback and visibility.** No config to connect with, a missing command, a failed or
timed-out run, no memory access, or a run over the search limit falls back to the
single-call extraction for that turn, with a `capture.log` line naming the reason. A reply
that is not a proposal list writes nothing, and the turn counts as mined. The capture
alert is raised by the single-call path as before, so an expired login still reaches the
terminal. The capture hook's timeout on Claude Code went from 120 to 180 seconds to cover
both runs. Only a host whose first extractor is the headless agent command runs agentic
capture; the other hosts keep their own CLI.

**Switch.** `agentic_capture`, on by default, in `FEATURE_DEFAULTS` and the hooks' copy.

**Measured.** On 2026-09-24 on one machine, with
`tests/fixtures/agentic_capture/replay.py`: nine synthetic coding turns in
`turns.json` (no real transcript was used), each replayed twice through both paths against
a freshly seeded local store, 18 runs per path. The raw rows are in
`results-2026-09-24.json` beside the fixtures.

| | single call | agentic |
|---|---|---|
| Expected changes made (new facts and replaced values) | 7 of 14 | 12 of 14 |
| Stored values the turn changed that were ended (supersedes caught) | 2 of 6 | 6 of 6 |
| Duplicate of a stored fact on the restated-instruction turn | 2 of 2 runs | 1 of 2 runs |
| New claims the turn did not call for, all turns | 6 | 4 |
| Writes from the turn that pastes the extractor's prompt | 0 | 0 |
| Writes from a fact stated only in the earlier turns | 0 | 0 |
| Mean input tokens per turn (of which cache writes) | 45,258 (19,290) | 19,436 (3,905) |
| Mean output tokens per turn | 1,272 | 1,163 |
| Mean seconds per turn | 19.6 | 17.3 |

The agentic runs made 0 to 3 searches (6 runs with none, 7 with one, 4 with two, 1 with
three), and none fell back. Both paths missed the project defect in the "known defect"
turn in both rounds. Three of the four unwanted agentic claims are `located_now Lisbon`
on the temporary-timezone turn, which a reader may well count as correct; the fourth is a
restatement of the stored hook-test rule filed under `known_defect`, where the reconciler
cannot see it as a duplicate. The single-call cost is high on this machine because that
path loads the user's instruction files and plugins into every run; with small ones it is
about 21,000 input tokens, as `lib/extract.py` records, and the agentic run then costs
about the same. The bar in 3.1 (no fewer facts, no more duplicates) is met on this set, so
the switch ships on. Nine turns is a small set, and a larger replay over real transcripts
is left to P3-F. These numbers were measured before the searches were made plain reads,
and they still hold: the replay's local server had no model configured, so its searches
made no model call either way, and its costs count only the headless run's own tokens.
What the change removes is a cost the replay did not incur: up to four model calls per
turn on a store with `MEMVARA_LLM` set, which would have been billed on that key and not
counted in this table.

**Open.** The hosted connection was tested against a fake transport, not against the live
endpoint. `expires_at` now reaches a local store: the library has it since P3-B (#238),
and a test writes an expiring proposal to it. A hosted deployment gets it once it runs a
server that lists the argument; until then the hook drops it with a log note.

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
| P3-H | core `plugin/hooks/` | 3.7 agentic capture in the plugin: read-only search, checked proposals, fallback, `agentic_capture` switch | phase 1 `claim_links`; uses `expires_at` once P3-B ships it |

## 5. Out of scope for phase 3

Connectors beyond the three named providers, and encryption of the cloud database, which is an
operator task.
