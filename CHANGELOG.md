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
