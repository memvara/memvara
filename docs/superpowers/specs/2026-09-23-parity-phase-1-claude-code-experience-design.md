# Parity phase 1: the Claude Code experience

**Date:** 2026-09-23. **Status:** design approved in conversation on 2026-09-23, not built.
Line numbers are from agent-memory `origin/main` at `b57ef714` (called **core** below),
memvara-cloud `origin/main` at `c1ab3ab` (**cloud**), and `/Applications/workstation/claude-memvara`
(**plugin repo**, which vendors core's `plugin/hooks` at `eb25ea0`).

This is the first of three phases that bring memvara to parity with Supermemory's server and
Claude Code plugin, as inventoried in `local/supermemory-internals-2026-09-23.md` in the main
checkout. Phase 2 is documents and retrieval
(`2026-09-23-parity-phase-2-documents-and-retrieval-design.md`); phase 3 is extraction and
cloud (`2026-09-23-parity-phase-3-extraction-and-cloud-design.md`).

## 1. What this phase delivers

Nine features, all on by default, each switchable off in setup:

| # | Feature | Repositories |
|---|---|---|
| 1 | `/memvara:index` command that records facts about a codebase | plugin repo |
| 2 | A memory-research subagent | plugin repo, core `plugin/hooks` |
| 3 | One project scope derived from the git remote, including for MCP tool calls | core, cloud, plugin repo |
| 4 | A status line with memory activity counts | plugin repo, core `plugin/hooks` |
| 5 | A mark on every recalled line | core `plugin/hooks` |
| 6 | A combined profile call | core, cloud |
| 7 | Ending or retiring memories that match a query, with a preview | core, cloud |
| 8 | A reason stored when a memory is ended or set to end | core, cloud |
| 9 | Typed links between memories (`extends`, `derives`) | core, cloud |
| — | `/memvara:setup`, the switch for every feature in phases 1–3 | plugin repo |

Decisions taken on 2026-09-23: every feature is on by default and the user can switch each off
during setup; the project scope reaches MCP tool calls through a local proxy (§4.3, option (a));
the word for user-defined profile groups is **bucket**, because the cloud already uses
**category** for a project's predicate vocabulary (cloud `memvara_cloud/memories.py:30-35`).

## 2. Constraints every workstream keeps

- **No scope from tool arguments.** The MCP invariant (core `docs/claude/mcp-server.md:75-76`)
  stays. The project scope arrives as a transport header the server validates, never as a tool
  argument.
- **Ended, retired and erased stay three different events.** Nothing in this phase erases.
  Feature 7 ends or retires; it never erases.
- **No transaction-time argument on any tool schema** (invariant 8, `tests/test_server.py`).
- **Tool descriptions follow `.claude/rules/tool-descriptions.md`**, and the packaged skill
  `memvara/skills/memvara/SKILL.md` must not repeat a description.
- **Documentation ships in the same commit** as the behaviour: `README.md`, `CHANGELOG.md`,
  `docs/UPGRADING.md`, `docs/INTERNALS.md`, `SKILL.md`, and in the plugin repo its README
  command table (`README.md:449-452`).
- **Core tests use no network, no key and no sleep**: `SQLiteStore(":memory:")`,
  `HashingEmbedder`, `NullLLM` or a fake that asserts its call count. Coverage stays at 100% and
  `mypy -p memvara` stays clean.

## 3. The switch store

All phase 1–3 switches live in one place per surface.

- **Local plugin and hooks:** `~/.memvara/settings.json`, a flat object of
  `feature_name: true|false`. A missing key means the default, which is `true` for every feature
  in these specs. Environment variables of the form `MEMVARA_FEATURE_<NAME>=0|1` override the
  file, so a test or CI run can pin a value.
- **Core library:** the same `MEMVARA_FEATURE_<NAME>` variables are parsed in
  `ServerConfig.from_env` (core `memvara/server/config.py:247-339`, `_flag` pattern at :573) and
  echoed in the startup table (:752-763). Library callers pass constructor arguments instead.
- **Cloud:** phase 3 adds the `feature_settings` table and console screen. Until then the cloud
  honours the same names as deployment environment variables read in
  `deploy/memvara_deploy/settings.py`.

Phase 1 feature names: `index_command`, `research_agent`, `project_scope`, `status_line`,
`recall_mark`, `profile`, `forget_matching`, `end_reason`, `links`.

## 4. The features

### 4.1 `/memvara:index`

**What it does.** The command explores the current repository and records 10–30 facts about
it: purpose, stack, entry points, how to build, test and run, conventions, deploy targets and
key dependencies.

**How.** A new `plugin/commands/index.md` in the plugin repo. The command body instructs the
agent to:

1. Resolve the project subject `project:<host>/<owner>/<repo>` with the helper from §4.3
   (`memvara-project-id`, printed by a Bash snippet). This is the settled convention in core
   `docs/SUBJECT-CONVENTIONS.md:465-490`.
2. Call `memory_search` for that subject first, so facts already stored are not repeated.
3. Read README, manifests (`pyproject.toml`, `package.json`, `Cargo.toml`, `go.mod`, …),
   CI and Docker files, entry points and a sample of the git log.
4. Write each fact with `memory_remember`:
   `subject=<project subject>`, `memory_type="semantic"` (the tool re-files anything else;
   core `memvara/server/tools.py:2203-2214`), `extractor="memvara-index"` (so provenance does not
   read "Derived by user"; :2250-2259), and a predicate from the `engineering` pack
   (`depends_on`, `deploys_to`, `version`, …; core `memvara/packs/engineering.toml:14-103`).
   Phase 1 adds three predicates to that pack: `convention`, `entry_point`, `runs_with`.
5. When a stored fact is now different, call `memory_end` on the old claim with
   `reason="superseded by /memvara:index"` (§4.8) before writing the new one.

**Manifest and tests.** Adding a command changes three test guards in the plugin repo:
`_COMMAND_NAMES` (`test/test_plugin.py:7716`), the manifest-and-tree check (:7840-7880) and
`ALLOWED_PLUGIN_FILES` (:123-130). The manifest `plugin/.claude-plugin/plugin.json` lists the
new command.

**Acceptance.** Running it on this repository writes at least 10 claims with the project
subject; running it a second time writes no duplicates (`memory_stats` claim count unchanged
when nothing changed).

### 4.2 Memory-research subagent

**What it does.** A subagent the main agent can call for deep history. It runs 3–6 targeted
searches and returns a brief of at most 300 words that cites claim ids.

**How.** `plugin/agents/memory-researcher.md` in the plugin repo, with a `tools:` list limited
to the read tools: `memory_recall`, `memory_search`, `memory_ask`, `memory_since`,
`memory_standing`, `memory_history`, `memory_why`, `memory_neighborhood`, `memory_paths`,
`memory_stats`, `memory_profile` (§4.6). The manifest gains an agents entry; the same three
test guards as §4.1 change.

**Fix in the same change.** Core `plugin/hooks/approve.py:23-32` `READ_ONLY` omits
`memory_standing` and `memory_ask`, so the subagent would hit permission prompts. Add both, plus
`memory_profile`.

**Acceptance.** The agent file loads; a test asserts its tool list contains no write tool.

### 4.3 One project scope from the git remote

**What it does.** Every clone and worktree of one repository shares one project scope, and the
same scope applies to hook recall, hook capture and MCP tool calls.

**The identity.** A new pure function `canonical_project(cwd) -> str | None` in core
`memvara/project.py`:

- Read `git remote get-url origin` from the repository's common git dir (so worktrees resolve
  to their main repository).
- Normalise: lower-case host, drop credentials, port kept, strip `.git`, trailing slashes,
  query and fragment; `git@host:owner/repo` becomes `host/owner/repo`. Result
  `host/owner/repo`, the format `Scope.project` already documents (core `memvara/types.py:474-489`).
- No remote: `path:` + the first 16 hex characters of SHA-256 of the real path of the git root.
- Not a git repository: `None`, which means no project scope, exactly as today.

The plugin hooks vendor this function through core's `plugin/hooks/lib/` so they need no
library install.

**The channel to the server (decision (a)).** Scope must not come from tool arguments, and
Claude Code's MCP configuration is one static URL. So:

1. The plugin repo's `.mcp.json` points the `memvara` server at a local stdio process,
   `node ${CLAUDE_PLUGIN_ROOT}/hooks/js/mcp-proxy.js` (or the Python equivalent under
   `hooks/`), instead of the hosted URL directly.
2. The proxy forwards every JSON-RPC message to the hosted URL (`https://app.memvara.dev/mcp`
   or `MEMVARA_SERVER_URL`) and adds one header, `Memvara-Project: <canonical project>`,
   computed once at start from the proxy's working directory. It passes
   `Authorization` from the existing credential unchanged.
3. The hooks' REST calls (`plugin/hooks/lib/hosted.py`) send the same header.
4. Local library mode (stdio `memvara-mcp`) reads the project from the working directory at
   start through `ServerConfig` (new field `project`, env `MEMVARA_PROJECT`, default: derived).

**Server side.**

- Core: `ServerConfig.scope_kwargs` (config.py:343-345) includes `project`; `Memvara.scope()`
  (core.py:3243) gains `project=`.
- Cloud: `rest/scope.py` `resolve()` (:85-104) and the MCP binding (`rest/mcp.py:226-244`,
  `rest/deps.py:255-257`) read `Memvara-Project`. The header can only narrow: it sets
  `Scope.project` inside the tenant the credential already binds. A value that is not a valid
  canonical project is refused with 400 and the reason.
- Predicates declared `project_scoped = false` (core `memvara/schema.py:130`, :578-593) still
  clear the project, so user preferences stay visible from every repository.

**The mislabelled line.** `session_start.py:142-150` and the `memory_stats` scope line
(`tools.py:1681`) label a five-part key as four parts. Fix the label to
`tenant/user/project/agent/session` in the same change.

**Relation to `docs/SUBJECT-CONVENTIONS.md`.** This implements decision 4 and steps 8 and 12 of
that page (:431-463, :754, :761-765). Update the page's status lines in the same commit.

**Acceptance.** Two worktrees of one repository resolve the same project; an `https` and an
`ssh` remote for one repository resolve the same project; a header with another tenant's data
cannot read it (the tenant comes from the credential); a user-scoped preference written in
repository A is recalled in repository B.

### 4.4 Status line

**What it does.** Shows `⋈ memvara · 12 recalled · 3 searched · 5 captured` for the current
session, and `⋈ memvara · off` when the hooks are switched off.

**How.**

- Counters: a per-session file `~/.memvara/.hooks/counts/<session>.json` holding
  `{recalled, searched, captured, updated_at}`. `recall.py` increments `recalled` by the number
  of lines injected; `capture.py` increments `captured` after a successful write; `approve.py`
  (PreToolUse, matcher `mcp__.*memvara.*`) increments `searched` for read tools. Writes are
  atomic (temporary file and rename). Files older than 14 days are pruned, as `recalled/` is
  today (`recall.py:241`).
- Script: `plugin/statusline.py` in the plugin repo reads the session id from the status-line
  input on stdin and prints one line. It must finish in under 50 ms and print nothing on error.
- Installation: `session_start.py` installs `statusLine` into `~/.claude/settings.json` only
  when no `statusLine` is set, writing through a temporary file and rename and leaving every
  other key untouched. It never overwrites another tool's status line. `/memvara:setup` can
  remove it.

**Acceptance.** Counts match the hook logs over a scripted session; an existing `statusLine`
is left byte-identical.

### 4.5 A mark on every recalled line

**What it does.** Every line the hooks inject starts with `⋈ `, so a reader can tell recalled
memory from everything else.

**How.** In core `plugin/hooks/recall.py` (`HEADER` :283-286, `_split` :416-426, `_clip`
:409-413) and `lib/standing.py` `render()` (:317-345), prefix each bullet with `⋈ `.

- The dedupe hash (`recall.py:289`, :854) is computed over the line **without** the mark, so
  the seen-set stays valid across the upgrade. `test/test_plugin.py:1275` in the plugin repo is
  updated to pin that.
- Capture (`lib/transcript.py:40-59`) drops any line starting with `⋈ ` and the block headers,
  so injected memory is never extracted again. This is the failure Supermemory showed: its
  recall text was re-uploaded and re-extracted.

**Acceptance.** A transcript containing an injected block produces no claims from that block.

### 4.6 Combined profile call

**What it does.** One call returns standing preferences, recent changes and, optionally, search
hits, grouped into buckets.

**Interface.**

```python
Memvara.profile(
    query: str | None = None,        # adds a "relevant" section when given
    *, k: int = 8,                   # rows per section
    since: datetime | None = None,   # default: 7 days before now
    buckets: dict[str, list[str]] | None = None,  # bucket name -> predicates
    tenant=..., user=..., project=..., agent=..., session=...,
) -> Profile
```

`Profile` has `standing: list[Row]`, `recent: list[Row]`, `relevant: list[Row]` and
`buckets: dict[str, list[Row]]`, where each `Row` carries `claim_id`, `text`, `inferred`.

- `standing` reuses the `_standing` logic (core `tools.py:669-743`); this adds the missing local
  `Memvara.standing()` method so both backends share one path.
- `recent` reuses `Memvara.since` (core.py:2716-2776).
- Default buckets come from the packs: `decisions` (decisions pack), `engineering`
  (engineering pack), `events` (events pack). A caller-supplied `buckets` replaces them.
- Surfaces: MCP tool `memory_profile`; cloud `POST /v1/profile`; `RemoteMemvara.profile`.
- The plugin's `lib/standing.py` fallback chain (:348-369) calls `memory_profile` first.

**Acceptance.** One call returns the same rows as the separate `standing` and `since` calls it
replaces; an unknown predicate in a bucket is ignored and reported in `Profile.warnings`.

### 4.7 Ending or retiring memories that match a query

**What it does.** Ends or retires every live memory that matches a query, after the caller has
seen exactly which ones.

**Interface.** Two calls with the same tool:

1. `memory_forget_matching(query, close="ended"|"retired", k=20, reason=None)` returns the
   matching claims (id and text) and a `confirm` token. It changes nothing.
2. The same call plus `confirm=<token>` applies the close to exactly the listed ids.

The token is an HMAC over the sorted claim ids, the close kind and a 10-minute expiry, keyed by
a per-process secret. A token that does not match, has expired, or whose claims have changed
since the preview is refused with the reason; nothing is applied.

- Core: `Memvara.forget_matching(query, *, close, k, reason, confirm=None) -> ForgetPreview |
  ForgetResult`, built on `search` and the existing `delete(claim_id, at, close)` (core.py:1950-1980).
- Cloud: `POST /v1/forget-matching` with the same two-step contract.
- `close="erased"` does not exist on this tool. Erasure stays operator-only.

**Acceptance.** A preview changes nothing; a confirmed call closes exactly the previewed ids;
a stale token closes nothing.

### 4.8 A reason on ending and on a future end

**What it does.** Records why a memory ended, or why it will end on a date.

**How.** The engine-owned closure record `{at, close, by}` (core `memvara/types.py:179`,
:311-312) gains `reason: str | None` (at most 500 characters). It lives in `meta` JSON, so no
schema migration is needed.

- `memory_end(…, reason=)` and `memory_forget(…, reason=)`; `remember(…, true_until=…,
  until_reason=…)` for a planned end. The REST `EndRequest` and `ForgetRequest` (cloud
  `models.py:1404`, :1413) and `FactRequest` gain the same fields.
- `history()` and `why()` show the reason.

**Acceptance.** A reason written through each surface reads back through `history` and `why`.

### 4.9 Typed links between memories

**What it does.** Records that one memory adds detail to another (`extends`) or was inferred
from others (`derives`). Supersession keeps its existing pointer (`invalidated_by`,
types.py:681).

**Data.** A new table, `claim_links(tenant, from_id, to_id, relation, created_at, by)`, primary
key `(tenant, from_id, to_id, relation)`, relation `CHECK IN ('extends','derives')`, both ids
referencing claims. SQLite: `SCHEMA_VERSION` 12 → 13 with `_migrate_to_v13`
(core `memvara/store/sqlite.py:157`, :1244-1275). Postgres: `SCHEMA_VERSION` 2 → 3 with
`CREATE TABLE IF NOT EXISTS` (cloud `memvara_cloud/store/postgres.py:151`, :972-983).

**Interface.** `Memvara.link(from_id, to_id, relation)`, `Memvara.links(claim_id)`; MCP tool
`memory_link`; `memory_why` lists links. Consolidation (core `memvara/consolidate/`) writes
`derives` from each derived claim to the claims it came from. Erasing a claim removes its link
rows in the same transaction.

**Acceptance.** Links survive a round trip through both stores; erasing either end removes the
row; `memory_why` shows them.

### 4.10 `/memvara:setup`

A command in the plugin repo that prints every feature from phases 1–3 with its current value
and default, and sets one with `/memvara:setup <feature> on|off`. It writes
`~/.memvara/settings.json` (§3). Features from later phases appear as soon as they ship.

## 5. Workstreams for the fan-out

Each workstream is one branch and one pull request, written test-first, reviewed with
`/code-review high` before merge.

| Stream | Repository | Features | Depends on |
|---|---|---|---|
| P1-A | core | 4.3 core part (`memvara/project.py`, `ServerConfig.project`, `Memvara.scope(project=)`), 4.6, 4.7, 4.8, 4.9, switch parsing (§3) | — |
| P1-B | core `plugin/hooks` | 4.2 approve fix, 4.3 hook part (header), 4.4 counters, 4.5 mark and capture filter | P1-A's `memvara/project.py` (vendored copy) |
| P1-C | cloud | 4.3 header binding, `/v1/profile`, `/v1/forget-matching`, reason fields, `claim_links` Postgres migration | P1-A merged (the cloud installs core from main; `check_core.py`) |
| P1-D | plugin repo | 4.1, 4.2 agent, 4.3 proxy, 4.4 status line script and install, 4.10, vendor P1-B | P1-B merged |

P1-A and the design of P1-B's header can start together; P1-C starts when P1-A merges.

## 6. Out of scope for phase 1

Documents, ingestion, chunking, metadata filters, read-path model calls and encryption are
phase 2. Agentic extraction, per-project guidance, project management, expiry erasure,
connectors and the cloud settings screen are phase 3.
