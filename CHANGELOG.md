# Changelog

Notable changes per release. Dates are the commit date; this project has not been
published to PyPI yet, so nothing below has a release tag.

The format is loosely [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
versions follow [semantic versioning](https://semver.org/) once `1.0.0` ships. Before
then, the `Store`, `Embedder` and `LLM` protocols may change in a minor release.

## [Unreleased]

### Added

- **Two independent time axes: `valid_at` and `known_at`.** Bitemporal data answers four
  questions and this library could express one of them. `as_of` moves both clocks to the
  same instant, so it only ever asked "what did we believe *then*, about *then*" — and the
  reading it cannot reach is the one bitemporality exists for. A correction that arrives
  in August about June is invisible to `as_of=June`, because that call rewinds the belief
  clock past the correction it is asking about.

  ```python
  mem.get_all(valid_at=T)   # what we believe today about how the world was at T
  mem.get_all(known_at=T)   # what we believed at T, about the world as it is now
  mem.get_all(as_of=T)      # both clocks at T — unchanged, and still correct
  ```

  Every read that took `as_of` takes all three: `search`, `get_all`, `count`, `history`,
  `why`, `produced`, `neighborhood`, `paths_between`, on `Memvara`, `ScopedMemvara` and
  `AsyncMemvara` alike. `as_of` is **exact sugar** for `valid_at=known_at=T` — every
  existing call, test and benchmark is untouched — and passing it alongside either axis
  raises rather than quietly picking one, because there is no reading of the mix in which
  one of the two is not being ignored.

  On the three *record* reads — `history()`, `why()`, `produced()` — an unset axis means
  "no filter" rather than "now". A timeline whose default was now would drop every
  superseded version, which is the whole content of a timeline. `history(known_at=T)` is
  the audit query that was missing: the trail *as it looked* on T, which for a slot
  corrected later is a different document from the one you would read today.

  `include_invalidated` is untouched and still lifts end-of-life. One consequence is now
  documented rather than implicit: it lifts the whole valid-time interval, not only its
  end, so under that flag `valid_at` has no effect. The belief floor never lifts.

  Recency decay follows `known_at`, not `valid_at` — it asks how long ago we last heard
  something, which is a question about the belief clock. Traversal now pins the
  `(valid_at, known_at)` *pair* before its first hop, so the one-instant guarantee is
  unweakened: one clock read fills both defaults, and every hop of a walk sees the same
  pair.

- **Multi-hop traversal** — `Memvara.neighborhood(entity)` and
  `Memvara.paths_between(source, target)`, both returning `list[Path]`, plus
  `Store.adjacent()` and `memvara/retrieve/traverse.py`. The store has been a labelled
  directed graph since entity resolution landed — claims are `(subject, predicate,
  object)` and the fold makes every spelling of a name one identity — and nothing could
  query it transitively. "Where does Alice work" was one lookup; "who does Alice's
  manager report to" was not expressible at any cost.

  **Every edge on a path is evaluated at one `as_of` instant**, pinned before the first
  hop. This is the property the feature exists for. A search-then-search agent loop with
  a write landing between its two reads will report a chain that was true at no instant —
  `bench/multihop.py` demonstrates one, where the write retired hop 1 and created hop 2 —
  and traversal returns nothing at every instant. Negative polarity is never walked as a
  link, scope is checked on every hop with `Scope.sees`, and the score is the product of
  confidence and per-predicate recency damped 0.75 a hop, so a path can never outscore
  its own prefix.

  Honest about where the value is: at two hops a plain search-then-search loop already
  reaches 96.3% on a synthetic set, so recall alone barely justifies this. At three hops
  the loop collapses to 4.7%, against 34.7% for traversal at its defaults and 48.7% with
  `min_hops`. LOCOMO's `multi-hop` category is **not** transitive multi-hop — its
  questions are single-fact lookups — so the measured 36% there cannot be improved by
  this, and is not claimed to be.
- **`Path` and `Edge`** (`memvara.retrieve`) — a path carries its nodes, the spellings
  actually stored, each claim and the direction it was walked, so every hop goes to
  `why()`. A path the caller cannot inspect is an answer they cannot check.
- **Windows is actually supported.** It was listed in CI and had never run there; the
  first run reported 99 failures. See *Fixed*.
- **Token accounting on the write path** — `WriteReceipt.tokens_in` / `tokens_out`,
  `ImportReceipt.tokens_in` / `tokens_out`, and the `write.tokens_in` /
  `write.tokens_out` series. `write.llm_calls` was the only cost signal and it cannot be
  billed on: providers charge per token, and a one-line turn and a 40,000-token document
  are both exactly one call, so the ratio between calls and spend is unbounded. Input and
  output are separate because they are priced separately, usually several-fold apart.
  On the receipt as well as in telemetry for the reason `llm_calls` is: a cost a caller
  can only discover by configuring a metrics backend is a cost most callers never
  discover.
- **`LLM.Usage` and `LLM.reports_usage`** (`memvara.llm.Usage`). A backend that can report
  usage advertises it and fills a **caller-allocated** accumulator passed as `usage=`.
  Caller-allocated rather than a `last_usage` attribute because `pipeline.py` deliberately
  runs the model round trip outside the store transaction, so two `add()` calls can be
  inside `extract()` on one backend at once — shared mutable state would bill one caller
  for the other's tokens, intermittently. One accumulator spans a whole write, including
  the predicate acquisition an extraction triggers, because the unit billed is the write
  and not the round trip.

  **Backwards compatible**: the write path only sends `usage=` to a backend that sets
  `reports_usage`, so an implementation written against the older three-argument
  signature keeps working untouched and simply publishes no token series — the same
  courtesy `classify_predicate` still gets. `AnthropicLLM` and `OpenAILLM` report; a
  response whose usage block is missing or unreadable records **nothing rather than a
  zero**, because a call that reached a provider consumed something and a run of zeros
  would understate a bill while dragging a fleet-wide average toward it.
- **`write.extract_ms`** — the model round trip timed on its own. Extraction time was
  previously only recoverable as `write.latency_ms` minus `write.lock_held_ms`, and a
  difference of two aggregates is not a distribution: percentiles do not subtract, so
  that arithmetic had a mean and no recoverable p99, which is the shape worth alerting
  on. Includes the call that raised — excluding it makes the p99 *improve* during a
  provider outage. Emitted only when a model was actually consulted, so a `NullLLM`
  deployment reports no series rather than a series of zeros.

### Changed

- **The `Store` protocol speaks in two axes; `as_of` survives on the facade only.**
  Every protocol method that took `as_of` now takes `valid_at` and `known_at`, both
  keyword-only: `competing_claims`, `adjacent`, `candidate_ids`, `lexical_search`,
  `vector_search`, `episode_candidate_ids`, `lexical_search_episodes`,
  `vector_search_episodes`. Keyword-only is the point — they replaced a positional
  argument, and a call still passing an instant third would otherwise be silently
  reinterpreted as `valid_at`, which is a wrong answer with no error.

  A SQL-backed store also implements the new `SQLStore` protocol
  (`_live_clause(valid_at, known_at, include_invalidated, alias="")` and
  `_happened_clause(valid_at, known_at, alias="")`). It is deliberately a *second*
  protocol rather than a widening of `Store`: those two are SQL generation, and `Store`
  promises a Qdrant or LanceDB backend can implement it without them. `_happened_clause`
  takes both axes because a turn's single `ts` is its `valid_from` and its `recorded_at`
  at once, so the claim rule collapses to `ts <= min(valid_at, known_at)` rather than
  that bound being chosen.

  Third-party stores must update their signatures; third-party *callers* of the facade
  are unaffected. Pre-`1.0.0`, per the note at the top of this file.
- **`Store.adjacent` takes `scopes`, and implementations must apply it inside `limit`.**
  It shipped without one, with `GraphTraverser` filtering the page afterwards and
  re-asking ten times wider on starvation. That was unsound rather than slow: on a tenant
  two people share, another user's claims about the same entity fill the page and the
  caller's own edges are cut before the filter can keep them. Measured with 20 readable
  claims about one hub, 15,000 competing claims returned 19 and 40,000 returned **8**,
  with nothing saying the answer was partial and its size set by a *different* user's
  write volume. The retry was deleted rather than tuned, because no multiplier is
  correct: a filter and a limit cannot be split across two layers and still give a
  correct top-k. Also 44x faster at 15,000 competing claims, since the rows are no longer
  fetched to be discarded.
- **SQLite schema v6** — `claims.subject_key` / `object_key` with indexes. `fact_key` and
  `value_key` both hash the predicate in, so no existing index could answer "which claims
  touch entity X" in either direction. Backfilled on first open by one statement: **0.62 s
  at 100k claims, 9.9 s at 1M**, once. Lazy backfill was rejected because `''` is a real
  stored key, so an unfilled column would be indistinguishable from "mentions nothing"
  and pre-v6 claims would silently answer "not connected".

### Fixed

- **The vector index could not open on Windows at all.** `os.pread`/`os.pwrite` are
  POSIX-only and Python provides no equivalent there, so every store with a `.vecs`
  sidecar raised `AttributeError` — 95 of the 99 failures in the first CI run, which is
  one missing function rather than 95 bugs. Now `lseek` + `read`/`write`, as a single
  code path rather than a `hasattr` branch: a fallback only one platform exercises is a
  fallback nobody tests.
- **A date before 1970 was a write the store accepted and could not read back — on
  Windows.** `_ts` clamped only the upper bound, hard-coded to the POSIX year-9999 limit.
  Windows' CRT stops at year 3001 *and* rejects negative timestamps outright, so the exact
  defect the clamp exists to prevent — one accepted write permanently breaking every later
  read of its scope — was alive at both ends. Both bounds are now probed from the C
  library rather than assumed. Surfaced by an ordinary decay test dating a claim 600 years
  back; on POSIX the floor is year 1, so a suite at 100% coverage on three platforms never
  saw it.

## [0.1.0] — 2026-08-10

First release. The sections below are the whole history to date, kept rather than
flattened: several of these entries are the searchable record of a specific defect, and
"initial release" would erase exactly the detail worth keeping.

### BREAKING

Nothing to break — this is the first published version. The entries below describe
changes made against unreleased code and are recorded because they document behaviour
someone reading the store's semantics needs to know is deliberate.

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
