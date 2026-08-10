# Changelog

Notable changes per release. Dates are the commit date; this project has not been
published to PyPI yet, so nothing below has a release tag.

The format is loosely [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
versions follow [semantic versioning](https://semver.org/) once `1.0.0` ships. Before
then, the `Store`, `Embedder` and `LLM` protocols may change in a minor release.

## [Unreleased]

### BREAKING

- **`get()` and `why()` are now scope-checked by the same rule as `get_all()`.** They
  authorized with `Scope.contains`, where an unset field is a wildcard reaching
  *downward*; enumeration uses `Scope.ancestors()`, which reaches only upward. The two
  disagreed in four of seven scope shapes, so a handle could `get()` a claim that
  `get_all()` on the identical handle would not return — and with agents isolated by
  `agent=`, a session-scoped handle could read a sibling agent's claim by id. Ids are not
  secret: receipts, `invalidated_by` pointers, results and logs all leak them.
  `get(id)`/`why(id)` now return `None` when the claim was written at a **deeper** scope
  than the reader's handle. `forget()` and `history()` keep the downward reach on
  purpose — a `fact_key` ignores agent and session, so a user-level `forget` is meant to
  retire what its sessions wrote.
- **The vector matrix header changed** from `ENGRMVEC` to `MEMVAVEC` with the rename. No
  action needed: an unrecognised magic is already treated as a stale file and the matrix
  is rebuilt from SQLite. Costs one O(n) rebuild the first time an existing store opens.

### Changed

- **Renamed `engram` to `memvara`** — package, class (`Engram` → `Memvara`), env vars
  (`ENGRAM_DB` → `MEMVARA_DB`), console script (`memvara-mcp`) and every identifier.
  Two reasons, both found during Phase 8 prep. `pip install engram` already resolved to
  an unrelated MIT rendering/vision library, so the name was not merely unregistered but
  actively pointing at someone else's code and `twine upload` would have been rejected.
  And `engram` is the standard neuroscience term for a memory trace, which makes it
  *descriptive* of the product's own function — the weakest and hardest-to-defend
  trademark class. `memvara` is coined, is a fanciful mark, and is verified free on PyPI,
  GitHub and npm.

### Added

- **`Memvara.produced(episode_id)`** — `why()` run backwards: which claims a turn
  produced. Cheap now that a reverse provenance index exists, and it had no public door.
  Scope-checked with `Scope.sees`, so a session handle cannot read what a sibling agent
  derived from a shared turn.
- **LangGraph adapter** (`memvara[langgraph]`). The best-fitting framework interface of
  the four: `BaseStore` hands over the query text natively *and* `put(namespace, key,
  value)` supplies all three parts of a triple, so an item is stored as one claim **per
  field** and changing `city` retires exactly `city`. Per-field contradiction resolution,
  which the CrewAI adapter cannot do because its unit of memory is a sentence with no
  subject or predicate in it.
- **`Store.erase_episode(episode_id, *, cited=False)`** — the turn a retention rule has
  to reach and `erase_claim` cannot. `erase_claim(sources=True)` finds a turn only
  *through* a claim, so a turn the extractor found nothing in ("ok thanks") was
  unreachable by any per-claim erasure and accumulated forever, with `purge` far too
  blunt as the alternative. Refuses a cited turn by default, because erasing it leaves
  `why()` resolving to nothing.
- **`Claim.state`** — `live` / `ended` / `retired`, deliberately absolute rather than
  relative to an `as_of`. Three surfaces had derived it independently.
- **A SQLite floor.** `RETURNING` needs 3.35 and every `set_embedding` runs it, while
  `requires-python = ">=3.10"` admits builds linked against 3.31. Checked at construction
  instead of failing on the first vector write.

- **`OpenAILLM`** — the `memvara[openai]` extra has been declared since the first commit
  and shipped no adapter. Chat Completions with `strict: true` structured output.
  Refusals (`message.refusal`) are handled explicitly, because reading `content` alone
  turns a declined request into a silent empty extraction.
- **`LICENSE`** — Apache-2.0 was declared in `pyproject.toml` and the README with no
  license file present.
- **CI** (`.github/workflows/ci.yml`) — Python 3.10–3.13 on Linux, plus 3.13 on macOS and
  Windows. A separate job gates coverage at 100%, and a third installs the package with
  no extras and imports every module, so an accidental top-level SDK import cannot pass.
- **`docs/ROADMAP.md`** — phases 4–8 and the monetization argument.

### Changed

- **Model-backend validation is now shared** (`memvara/llm/_shape.py`). It was private to
  `anthropic.py`; a second backend reimplementing those rules would drift, and the drift
  would show up as differently-shaped claims in one store depending on which model was
  configured the day a turn was written. `anthropic.py` went 296 → 127 lines and is now
  transport plus response shape only.

### Fixed

- **One accepted write permanently broke every later read of its scope.**
  `valid_to=datetime.max` stored fine and every subsequent `get_all()`/`search()` over
  that claim raised `year 10000 is out of range`: `datetime.max.timestamp()` is
  253402300800.0, float64 has no precision left there, so the value rounds *up* onto it
  and `fromtimestamp` cannot invert it. The write returned success and the damage was
  deferred, with nothing pointing at the row that caused it. `_ts` now clamps one ulp
  down — a timestamp the float could not represent anyway.
- **`remember(sources=…)` returned a receipt saying it stored no turns**, while the turn
  was on disk and the claim cited it. Anything reading `episode_ids` to evidence what a
  call wrote — a governance audit entry, an importer reconciling itself — evidenced
  nothing.
- **`remember(sources=[Episode(...)])` filed the turn under tenant `"default"`, and
  `purge()` then left it on disk.** `Episode.scope` defaults to `Scope()`, whose tenant is
  the literal `"default"`, so the documented way to attach provenance wrote raw user text
  into another tenant while its claim landed in the right one — and an erasure of that
  user reported `episodes: 0` with the sentence still there. Nothing surfaced it, because
  `get_episode` is unscoped so `why()` kept resolving. A caller-built episode that names
  no scope now adopts its claim's. **Existing stores may already hold orphaned turns**;
  `docs/RELEASING.md` carries a detection query.
- **`forget()` returned stale claims** — every one reported `invalidated_at=None` and read
  as live while the same row re-read from the store read as retired.
- **`remember(**meta)` accepted `salience_base`**, which the next `consolidate()` turned
  into a permanent 5.0 ranking override reachable through no documented argument. Reserved
  keys are now rejected at the boundary.
- **`iter_claims`/`iter_episodes` re-sorted the whole matching set once per page.** A
  plain `tenant=?` is indexable, so SQLite chose an index and then a temp b-tree to get
  back into rowid order — for every page. `iter_episodes` over one tenant, which is what
  `reembed()` walks: 26.7 / 169.1 / 626.6 ms at 5k / 20k / 50k → 17.4 / 69.2 / 173.3.
- **A backdated supersession left two live values for a single-valued predicate.**
  `Reconciler._retire` stamped the transaction instant on both time axes, so learning
  today that someone moved in July closed the old value *today* — leaving it valid across
  a window in which its replacement was also valid. `valid_to` now comes from the new
  claim's `valid_from`, clamped so it can never precede the victim's own `valid_from`.
  Invisible on any non-backdated write, which is why 1,653 tests passed over it; it
  affected imports and replays, which is exactly where bitemporality earns its keep.

## Wave 3

### Added

- **Entity resolution** (`memvara/entities.py`). `entity_key()` is a pure fold applied
  before any key exists, so `Acme`, `Acme Corp` and `acme, inc.` are one identity.
  Measured over 258 writes: 98.1% resolved by fold, zero model calls, 41 surface forms to
  the 9 real entities. The LLM path is opt-in and unset by default.
- **Per-claim erasure** — `store.erase_claim`, `Memvara.erase(claim_id, sources=False)`.
  Distinct from `delete()`, which retires.
- **Telemetry** (`memvara/telemetry.py`). A `Recorder` protocol, default `None` rather than
  a no-op object. All six silent failure modes have a live series. Unset costs +0.8% on
  write and −0.4% on read against a control with the emission points deleted.
- **`AsyncMemvara`** (`memvara/aio.py`) — each public method over `asyncio.to_thread`.
- **`Memvara.supersede`**, **`remember(sources=, text=, extractor=)`**, **`get`**,
  **`count`**, **`reset`**, **`scope`**.
- **`backfill_entities()`** — dry-run by default, for applying a late alias to history.

### Changed

- **Reads no longer queue behind writes.** Per-thread read connections, and the slow half
  of a write (near-duplicate encode, model call) moved outside the transaction. Reads
  during a 20k-claim consolidation sweep: 1,470 → 13,728 completed, p99 30.4 ms → 2.01 ms,
  idle read latency unchanged.
- `Memvara.add`'s outer batch narrowed to the indexing half — it had been holding the write
  lock across the whole call and defeating the hoist end to end (2 → 1,030 reads during a
  1 s extraction).

### Fixed

- The mem0 compat layer leaked memvara's internal bookkeeping keys as caller metadata.
- `Memory(on_delete="erase")` refused rather than erasing; it now performs a real erasure.
- An import crash between a retirement and its replacement could leave a note slot empty.

## Wave 2

### Added

- **MCP server** (`python -m memvara.server`) — eight tools over JSON-RPC 2.0 stdio, no SDK
  dependency. `consolidate`, `purge`, `reset` and `erase` are deliberately absent.
- **mem0 compatibility** (`memvara.compat.Memory`) and the **`history.db` importer**, which
  replays mem0's own mutation log into a queryable bitemporal history at zero token cost.
- **Retrievable episodes** — `search(include_episodes=True)`, down-weighted and capped.

### Changed

- **Spacing effect.** Storage strength (`salience_base`) separated from retrieval strength,
  with reinforcement bumping inversely to retrievability. A daily-mentioned fact went from
  decaying to a 0.05 floor to rising to a 5.00 ceiling.
- **Windowed consolidation** — commits every 500 rows instead of one transaction over the
  whole sweep, which had been locking out the store's own writes.

## Wave 1

### Added

- Bitemporal `Claim` model, deterministic contradiction resolution keyed on
  `(subject, predicate)`, hybrid BM25 + vector retrieval fused by RRF, tiered write path,
  consolidation and decay, `why()` provenance, scope hierarchy.
- mmap-backed vector index with cross-process coherence.

### Fixed

- **Predicate explosion** — resolution plus alias merging took a simulation from 31 live
  predicates to 6, and 6 employers to 1.
- **The FTS index was keyed on an `UNINDEXED` column**, making N writes over N rows O(n²)
  and dominating consolidation at 80% of its time. Consolidation went 4.8 s → ~460 ms at
  8k claims.
