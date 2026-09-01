# Changelog

Notable changes per release. Dates are the commit date, not the upload date: `0.1.0` is
tagged `v0.1.0` and reached PyPI on 2026-08-14.

The format is loosely [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
versions follow [semantic versioning](https://semver.org/) once `1.0.0` ships. Before
then, the `Store`, `Embedder` and `LLM` protocols may change in a minor release.

## [Unreleased]

### Added

- **`MEMVARA_LLM=openai` selects the OpenAI-compatible backend, which was finished and
  unreachable.** `OpenAILLM` has shipped complete since the `memvara[openai]` extra
  landed — Chat Completions with `strict: true` and explicit refusal handling — but
  `_BACKENDS` never named it, so the one value that would have selected it was a startup
  error. Wiring it up costs one tuple entry and one helper.

  The endpoint needs no variable of memvara's own: the adapter constructs its client
  through the official SDK, which reads `OPENAI_BASE_URL` and `OPENAI_API_KEY` from the
  environment itself. So a local vLLM, llama.cpp server or Ollama shim is reachable by
  pointing `OPENAI_BASE_URL` at it. What that does need is a model name, since
  `OpenAILLM`'s default names nothing on a self-hosted server — `MEMVARA_LLM_MODEL`, the
  one variable added, refused under `MEMVARA_MODE=cloud` alongside `MEMVARA_LLM` and
  `MEMVARA_EMBEDDER` for the reason all three share.

  `OPENAI_API_KEY` must be set even against a server that ignores it, because the SDK
  refuses to construct a client without one. That refusal is now a startup `ConfigError`
  naming the variable rather than a traceback out of `build_memvara` on the first turn
  that reached extraction.


- **GitHub Copilot CLI is a supported hook host.** `plugin/hooks/hosts/copilot.py`, with
  every value measured against Copilot CLI 1.0.82. It registers Claude Code's four event
  names on purpose: Copilot fires either casing and the casing decides the payload, so
  PascalCase delivers `session_id`/`tool_name`/`transcript_path` and the field map is
  Codex's rather than a fifth vocabulary. The reply envelope is flat — the nested
  `hookSpecificOutput` form delivers nothing there — and capture is declared synchronous
  because an async hook holds the turn open rather than being deferred.

  `lib/transcript.py` grew a third reader for it. Copilot keeps the typed prompt and the
  model-facing copy in separate fields; mining the typed one makes this host immune by
  construction to reading our own injected recall back in as conversation.

### Fixed

- **`CachedEmbedder.encode` raised `KeyError` on a batch larger than the cache**, on keys
  it had written itself moments earlier. Once the cache is full each insert evicts the
  oldest entry, and within a single call the oldest entries are that same call's — so a
  batch of more than `max_items` distinct texts evicted its own early results before the
  final read-back, and a batch mixing cache hits with enough new texts evicted the hits
  too. Either way the row was still owed to the caller.

  The threshold is *distinct* texts rather than list length, so duplicates masked it
  entirely: a fixture that was half repeats worked at 48k and the same code crashed at
  100k. Shipped paths were mostly safe — `reembed` batches at 256 — and it bit bulk
  callers: a migration, an import, a benchmark harness.

  Every row a call has to return is now captured before any eviction can run. Eviction
  behaviour, cache bounds and the hit/miss counters are unchanged.

- **A long run of reinforcements could fail with `database disk image is malformed`, on a
  store that was never damaged.** `put_claim` deleted and re-inserted the claim's row in
  `claims_fts` on every write — including the overwhelmingly common one where the text had
  not changed and the rewrite reproduced exactly what was already indexed. FTS5's
  `secure-delete`, on for `claims_fts` since the v7 migration and so on for every store
  this library has written since, makes such a delete rewrite the doclist inside existing
  segment pages rather than append a marker. Accumulate enough of them inside one
  uncommitted transaction and FTS5 raises `SQLITE_CORRUPT_VTAB`, which reaches Python
  under that message.

  Nothing is wrong with the file, and the error names the wrong thing. At the instant of
  failure `PRAGMA integrity_check` returns `ok` on any other connection, the committed
  index passes FTS5's own integrity-check, and a `rollback()` clears the state outright:
  what goes inconsistent is the pending index inside the open transaction, which is also
  why a page-level check could never have caught it.

  `put_claim` now reads `text` alongside `rowid` and `sources` in the SELECT it was
  already running, and rewrites the FTS row only when the text has actually moved. The
  read has to happen before the upsert, which sets every column including `text` — after
  it, the stored and incoming values are equal every time and the skip would swallow real
  changes. Measured against a store that reproduces the failure: 19,420 of 19,420 rewrites
  in the first transaction were of unchanged text, and the run now gets past the write it
  used to die on.

### Added

- **The benchmark's cost columns say what they do not measure**, and `Usage` has a
  `tokens` field. Both were named in the plan the benchmark was built from and neither
  had shipped: storage operations and model tokens were absent, and an absent metric
  reads as a zero one.

  `tokens` counts prompt and completion together. Every system shipped here reports `0` —
  a measurement, not a blank, since all three report `llm_calls=0` — and the field exists
  for adapters that do use a model, which would otherwise have to hide the cost that
  dominates their bill in `extra`, where nothing compares it.

  **Storage operations are not measured and now say so.** They cannot be, through a
  system-neutral interface: SQL statements and page reads live below every adapter's
  public API, and instrumenting one system's internals would produce a column only that
  system could fill. `db_reads` is a different quantity and the docs distinguish them.
  Memory footprint is disclosed as unmeasured on the same grounds.

  Two guards landed with it, both for the class of defect rather than the case. One
  asserts the published result-schema table names exactly the fields of `Usage`, in both
  directions — a field added without a doc entry and a doc entry surviving a rename are
  the same defect from two sides, and the last two renames (`db_writes` to `rows_stored`,
  `embedding_calls` to `texts_embedded`) were each caught by a reviewer rather than by
  the suite. The other asserts the report has a label for every field: the cost block
  hardcoded one line each, so `tokens` was added to `Usage` and simply not printed.

  `test_a_result_carries_no_secrets` scanned for the substring `token`, which the new
  cost field tripped. It matches on word boundaries now: `\btoken\b` finds
  `"token": "sk-live-…"` and does not find `"tokens": 0`. The first attempt deleted
  `token` outright, which silenced the false positive and quietly dropped
  `refresh_token`, `api_token` and a bare `"token"` with it — a guard tuned by what it
  happens to reject rather than by what it is for. The patterns are a named constant now,
  and fourteen cases pin them: nine credentials that must be caught, five cost fields
  that must not.

  The schema guard had the same shape of defect in its own regex: `[a-z_]+` cannot match
  a field name containing a digit, so a correctly documented `p95_ms` would have failed
  it. The sibling `latency` row already carries `query_p50_ms` and `query_p95_ms`.

- **The Agent Memory Benchmark is reflected in the documents that describe this
  project's evidence, and memvara's adapter now counts what it embeds.** Three gaps found
  by auditing the shipped benchmark against the plan it was built from, rather than by
  anything going red.

  `docs/ROADMAP.md` did not mention the benchmark at all — 0 occurrences — which for the
  document whose organizing theme is credibility was the wrong silence. It now carries it
  as *4d*, beside the other Phase 4 measurements, with the result stated as measured:
  three points over a numpy baseline, the whole lead in `temporal`, `retrieval` lost, and
  `irrelevance` a three-way tie. *What is still missing* gains the two things that are
  honestly missing — nobody outside this repository has reproduced it, and two of its
  seven dimensions do not yet discriminate.

  `CONTRIBUTING.md` told contributors the `bench/` conventions and never mentioned
  `benchmarks/`, so somebody arriving to add an adapter had no route to the guide written
  for them.

  `Usage.embedding_calls` read `-` for memvara — *not measured*, honest, and it left the
  system doing the most embedding as the one with no figure. A counting wrapper around
  `HashingEmbedder` reports **520**, against `vector-rag`'s 279, and the split is exact:
  241 claims plus 262 source episodes on the way in, then one per unprobed question.
  memvara embeds the claim *and* the turn it came from, which is what lets `why()` answer
  later. The wrapper delegates `name` and `dim` so `memvara.embed.fingerprint` derives the
  same identity as the bare embedder and a file-backed store still reopens.

  A review of that work found two more, fixed with it. `Usage.embedding_calls` was the
  one cost field still without a docstring — the same undefined-field condition that had
  just cost a published number one commit earlier — and it immediately grew the same
  split: memvara counting texts, `vector-rag` counting requests, agreeing only because
  neither batches. It is now `texts_embedded`, defined as texts, because batching is an
  implementation detail of an embedding API and counting requests would report identical
  work as orders of magnitude apart. The rename is disclosed in the result-schema section
  rather than left to be discovered; it happened before anything consumed the schema, and
  `benchmark_version` is unchanged because no score moved.

  And the adapter interface is **five** methods, not four. `registry.build` has always
  required `reset`, `remember`, `query`, `usage` and `close`; `adapters/base.py`'s
  docstring named four and omitted `usage`, and four other documents copied the count —
  so a contributor following the prose would learn about `usage` from a `TypeError`.
  Wrong in every copy from the start, which is why nothing could disagree with it. A test
  now scans every document that states a count against the list the registry enforces,
  and fails naming the file and line.

  No score changed: 92.0 / 89.0 / 50.0, and `datasets/v1` is untouched.

- **`tests/test_docs.py` refuses a wording this project has already corrected.** Two
  entries, both earned: the sentence about what `add()` costs, and "every read takes all
  three". Between them they had six live copies across five files, none found by the
  suite and none by the author of either fix — who knew which file they were editing,
  which is exactly why the copy that survives is never that one.

  It scans **every shipped markdown file**, which is wider than the reader's path that
  `tests/test_doc_links.py` walks. That set is `README.md`, `docs/` and `examples/`; the
  copies off it matter as much — the packaged skill under `memvara/skills/memvara/`, its
  mirror under `plugin/skills/`, and the `README.md` files in `npm/`, `plugin/`, `demo/`
  and `release/`. The packaged skill is the sharpest of those: its tool list rotted this
  same way once, and it is vendored by sha into seven downstream repositories, so a wrong
  sentence there is a wrong sentence in all of them. `CHANGELOG.md` is the one exclusion,
  since it quotes each wrong form inside the entry recording the correction, and scanning
  it would go red on the evidence that the fix happened.

  **The patterns are loose on purpose, and spaces in them match line breaks.** These
  files are hard-wrapped, so a wording that sits on one line today lands across two the
  moment a word ahead of it changes. The first version of the second entry also required
  `read` and `takes` to be adjacent, and so matched neither *"Every read in the API
  takes"* nor *"every read below takes"* — two live copies, in the commit that added the
  guard. A pattern that only matches the exact sentence already fixed is a guard that
  reads as protection and is not.

  **What it does not do is worth stating, because the name suggests otherwise.** It
  catches a wording coming *back*. A fresh overstatement in fresh words passes it, and
  nothing in the suite sees that. The list is a record of mistakes made, not a model of
  the API, and the time to add an entry is after correcting a claim that turned out to
  have copies.

- **The Agent Memory Benchmark** — a public, reproducible benchmark for memory systems in
  general, in `benchmarks/agent_memory/`. It measures what happens to a fact that
  changes: current state, historical state, contradiction resolution, provenance,
  knowledge time, retrieval among distractors, cost and latency. 262 events, 100
  questions, 16 scenarios, no API key, no network, about a second per system.

  ```bash
  python -m benchmarks.agent_memory --system memvara --system naive --system vector-rag --compare
  ```

  **It is not a memvara benchmark, and it is deliberately possible to lose it.** The
  dataset, the questions and the scorer never mention a system; everything system-specific
  is behind a four-method adapter interface, and `--system` accepts a dotted import path so
  a memory system in another repository is scored without forking this one. Two baselines
  ship beside the memvara adapter: `naive`, a dictionary of current values, and
  `vector-rag`, retrieval over the whole write log with one clock. Neither is a strawman —
  `vector-rag` is completely correct on current state, provenance, change time and
  knowledge time.

  **Measured, at the commit this landed in:** memvara 92.0% overall, `vector-rag` 89.0%,
  `naive` 50.0%. The three points separating memvara from a numpy baseline are entirely
  the `temporal` dimension (100.0% against 91.5%), and the four questions that separate
  them are the four delayed-knowledge and correction scenarios — where news arrived after
  the fact, so "what was true then" and "what had we heard by then" are different answers
  and one clock has to give the same one to both. **memvara loses the `retrieval`
  dimension** (64.3% against 71.4%) and is the worst of the three on `multi_hop`, at 1 of
  6; `irrelevance` is a three-way tie that discriminates nothing. All of that is in the
  report rather than in a footnote.

  A second review of the branch found three more, all fixed here: `Usage.db_writes` was
  undefined, so memvara reported rows stored while the baselines reported write calls and
  the published table compared them under one heading — it is now `rows_stored`, means
  rows for all three, and reads in the right direction (naive 193, memvara 241,
  vector-rag 262, so the dictionary holds *fewer* rows and memvara's 48 extra are what
  buys the temporal result); `_is_correction` ignored cardinality and told `remember()`
  that 30 multi-valued writes were corrections, which memvara's own MANY handling
  absorbed rather than the adapter getting right; and the leaderboard sliced dimension
  names to 13 characters, printing `knowledge_tim`.

  Line endings are pinned on both sides. `datasets/build_v1.py` wrote its two `.jsonl`
  files with `newline="\n"` and `metadata.json` without it, so on Windows the generator
  emitted CRLF for one file of the three — one call out of three, in a file nobody
  rereads once it works. All three now go through one writer. `RunResult.write` had the
  same latent bug and is fixed with it: nothing compares a result file's bytes today,
  which is exactly why it would have gone unnoticed until two runs from different
  platforms were diffed.

  `.gitattributes` pins `benchmarks/agent_memory/datasets/` to `eol=lf`. The dataset is
  a byte-exact artefact — the suite regenerates it and compares bytes, and CI diffs the
  regenerated copy against the committed one — and nothing declared its line endings, so
  Git for Windows checked it out as CRLF and the comparison failed on the windows-latest
  job and nowhere else. It is the repository's first `.gitattributes`, deliberately
  scoped to these three files rather than declared repository-wide.

  Those figures replace an earlier set (90.0 / 88.0 / 49.0) that this branch briefly
  carried. Investigating memvara's `retrieval` loss found two defects in the harness
  rather than in any memory system: the memvara adapter searched ended and retired claims
  for present-tense questions, so a value nobody held any more outranked the right one;
  and the three adapters fed their retrievers three different strings, which made
  `retrieval` a comparison of adapters rather than of retrievers. Both are fixed, the
  index text now comes from one shared function, and both fixes raised **every** system's
  `retrieval` score. No question, gold answer or scoring rule changed.

  Gold answers are authored by hand in `datasets/build_v1.py` and derived independently in
  `timeline.py` from four published supersession rules; the suite asserts the two agree on
  every question. Two derivations that must match is the only defence against a scoring bug
  that every system fails identically, which nobody questions because the numbers look
  plausible.

  Scoring is deterministic — normalized matching, published aliases, exact set equality,
  ISO-8601 dates, and no LLM judge in any mode — so `--repeat-check` runs a system twice
  and asserts the verdicts are identical. `--show-failures` prints each wrong answer beside
  the fact's real timeline with a named reason, which is more use to a developer than the
  score.

  Two spellings reach it: `python -m benchmarks.agent_memory` is canonical and
  `python -m benchmarks.agent_memory.run` is an alias for people who reach for the
  longer name first. Both call `cli.main`, so there is no second implementation to
  drift, and `run.py` guards on `__name__` because — unlike `__main__.py` — it has an
  importable dotted name and would otherwise run the whole benchmark on import.

  Distinct from `bench/temporal.py`, which stays what it was: this repository's own
  regression suite for the two clocks, memvara-only, probing `get_all()` directly. The new
  benchmark is system-neutral, has a committed versioned dataset, an adapter interface, a
  published result schema and a public report at
  `docs/benchmarks/agent-memory-benchmark.md`.
- **`docs/LIMITATIONS.md`, holding what was *Honest limitations* in the README.** Every
  bullet moved unchanged; only the links were respelled relative, which is the convention
  inside `docs/` and the reason they were absolute in the README (that file is the PyPI
  project description, where a relative link resolves against pypi.org).

  Nothing was dropped or softened by the move. The README keeps two ways to reach it — a
  row in its documentation index, and a sentence under *Measured* saying that what those
  numbers do not cover is on that page, since the caveat most often quoted wrongly is
  that the LOCOMO and LongMemEval figures are retrieval rather than answer accuracy.
  `docs/FAQ.md` and `docs/concepts/why-memvara.md` both pointed at the README section and
  now point at the page.

- **The README shows the 90-second demo**, embedded from
  `releases/latest/download/demo.gif`. The three questions the demo answers — where does
  Alice live now, where did she live in March, where in January — are the point of the
  library, and a reader gets them before deciding whether to install anything.

- **The demo GIF is published as a release asset, at a URL that never changes.**
  `.github/workflows/demo-gif.yml` records the demo and attaches `demo.gif` to a GitHub
  Release when one is published, so `releases/latest/download/demo.gif` always resolves
  to the current demo and the README can embed it once. Nothing is committed: the GIF is
  1.1 MB against a 2.3 MiB packed history, so a commit would make one binary about a
  third of every clone and each regeneration would add another copy permanently. It is
  also now in `.gitignore`, because the command the demo's README gives writes it into
  the working directory.

  **It attaches to a release and never creates one.** Firing on a tag push and creating
  the release would announce a release before the human approval `release.yml`'s `pypi`
  environment exists to require — one that might never reach PyPI — and, with no
  `--prerelease` flag, would move `releases/latest` onto a release candidate. So a person
  publishes the release and this job only decorates it. `workflow_dispatch` with a tag
  attaches the asset to a release that already shipped.

  **On a dispatch the tag says where the asset goes, not what is recorded**: the GIF
  comes from the ref the run was started on. Those are the same thing on a release and
  come apart on a backfill, which is what dispatch is for — `v0.9.0` predates the demo,
  so there is no `record_gif.py` at that tag to run. The consequence to know is that
  backfilling an old release puts a recording of newer source on its release page, and
  at `releases/latest/download/demo.gif`. That is the trade for having one URL the README
  can embed, and it is the same reason the GIF is a build product rather than something
  a release carries a copy of.

  The job reads the file back before uploading it: frame count, how many frames carry
  ink, and whether the delays still add up to about ninety seconds. Those are the three
  ways this pipeline fails while exiting 0 — a font that resolved but drew nothing, a run
  that stopped part-way, and pacing that collapsed.

- **`examples/temporal_memory_demo/record_gif.py`, a third way to record the demo.** The
  two routes the README already gave both want tooling a bare container does not have:
  VHS needs a Go toolchain, `ttyd` and `ffmpeg`, and asciinema needs a terminal to attach
  to. This one needs Pillow and nothing else. It encodes **one frame per output event**
  rather than at a frame rate: the demo is ninety seconds of mostly still text, so 10fps
  would be nine hundred frames that are almost all identical, where fifty-six reproduce
  the pacing exactly at about a megabyte.

  It sizes the canvas to the content rather than to a nominal window, which is where it
  differs from `demo.tape`: 811×534 against the tape's 1000×760, which is four-fifths
  empty for the first half of the run. `--cols 98 --rows 37` makes the two match. Both
  failure modes that would otherwise surface as a blank frame or a screen of tofu — no
  Pillow, no monospace font carrying U+2500 — are refusals that name the fix.

  **It is deterministic, which is the property that makes it worth having.** The demo
  runs for real — a real store, real `memvara` calls — but on a virtual clock: `time.sleep`
  advances a counter instead of waiting, so the frame delays come from the schedule
  `demo.py` already declares (`BEATS`, and the per-line holds) rather than from a
  stopwatch. The same source therefore always produces the same bytes, which is what
  would let a check regenerate the GIF and tell whether it is stale. A measured recording
  cannot answer that: two ninety-second runs of the same script differ by a few
  milliseconds a frame, so they differ in bytes while being equally correct. The replay
  also takes about two seconds rather than ninety.

  `--live` keeps the measured recording for anyone who wants wall-clock evidence rather
  than a replay. That one runs under a pty rather than a pipe, because a program writing
  to a pipe is block-buffered and would collapse ninety seconds of pacing into one burst
  at exit. It is POSIX-only for the same reason — a pty is — and refuses on Windows
  naming the flag to drop; the default replay runs everywhere `requires-python` does. The two were checked against each other and produce
  **pixel-identical frames** — the virtual clock is a statement about time, not about
  what the program printed. `tests/test_examples.py` pins both halves: that two replays
  agree, and that what they replay is the transcript `expected-output.txt` holds. Both
  assert on the event stream rather than on an encoded GIF, so they hold without Pillow,
  which is not a dependency of memvara or of its `dev` extra.

  The section intro said a recording "needs a terminal" and that none had been generated;
  the first half is no longer true and the second never belonged in a file that outlives
  the session that wrote it. It now says what is true: no GIF is checked in because a GIF
  is a build product, and here are three ways to make one.

- **`examples/`, and a test suite that runs it.** Three programs a developer can run
  straight after `pip install memvara`: `temporal_memory.py` (one person moves twice, and
  the same question gets three different correct answers), `coding_agent.py` (an
  engineering decision, its supersession, its date and the turn it came from), and
  `temporal_memory_demo/` (the same story paced for a 90-second screen recording).
  `tests/test_examples.py` runs all three in a subprocess — as a reader would, without the
  repository on `PYTHONPATH` — and asserts on what they print, so an example cannot drift
  from the library the way a snippet in a document can. All three run on every interpreter
  `requires-python` names: `coding_agent.py` declares its vocabulary as `PredicateSpec`s
  rather than loading the shipped `decisions` pack, because a pack is TOML and `tomllib`
  arrives in 3.11 — an example a supported interpreter cannot run is worse than one that
  spells out four extra lines. The demo sets its own stdout encoding to UTF-8 for the same
  class of reason: its rules are box-drawing characters, and on Windows `sys.stdout`
  defaults to cp1252, which has none of them. The demo's transcript is a golden
  file (`expected-output.txt`) rather than strings in the test, so the script, its README
  and the test cannot become three versions of one thing. **No GIF or video is checked
  in**: `demo.tape` records one deterministically with VHS, and a binary regenerated by
  hand goes stale without going red.

- **A documentation tree with a route through it.** `docs/README.md` is the index,
  `docs/FAQ.md` sits beside it, and thirteen new pages sit under `getting-started/`,
  `concepts/`, `guides/`, `integrations/` and `reference/` — including the two
  explanations the repository did not have anywhere:
  [Why Memvara?](docs/concepts/why-memvara.md), which states the five questions a memory
  layer has to answer that retrieval does not, and
  [RAG and memory](docs/concepts/rag-vs-memory.md), which is about how the two compose
  rather than which one wins. The FAQ answers eleven questions against the
  implementation. `docs/reference/architecture.md` draws the real module map in Mermaid.
  Every page a reader can arrive at now ends with a navigation footer, and
  `tests/test_docs.py` fails if a new one does not.

- **`tests/test_docs.py` also executes the getting-started pages.** Nothing ran the
  documentation before this, and one page did not survive being run: `first-memory.md` had
  an `added[0]` after a write that reinforced rather than added, a history printed in the
  wrong order because one write in the slot was undated, and two `forget()` variants shown
  as alternatives but written as a sequence. All three read perfectly and none of them ran.
  Every python block on a page in `RUNNABLE_PAGES` now executes in reading order, in a
  subprocess from a temporary directory; a block that is an illustration rather than a step
  says so in the source with `<!-- runnable: no — <reason> -->`, which renders as nothing.

- **`tests/test_docs.py`,** which checks three things nothing checked before. The README's
  links are absolute — they have to be, because `README.md` is also the PyPI project page,
  where a relative link resolves against `pypi.org` and 404s — so `test_doc_links.py`
  skipped every one of them by design, leaving the most-read file in the repository as the
  only one with unchecked links. These resolve them back to paths in the checkout and
  check the file and the heading. It also pins the fourteen-tool table in
  `docs/integrations/mcp.md` against `memvara/server/tools.py`, the same guard
  `tests/test_init.py` already puts on the packaged skill's copy, and refuses a relative
  `href=` in the README's raw-HTML badge row — which is where the licence badge's broken
  PyPI link was sitting.

- **`keywords` and `classifiers` in `pyproject.toml`, and a `Documentation` URL.** The
  project page carried neither, so PyPI's search matched it on its own name and its
  description line, on an index where the terms people type are "agent memory" and
  "bitemporal". `tests/test_packaging.py`'s hand parser learned multi-line arrays to read
  them, pinned against `tomllib` by the agreement test as before.

### Changed

- **`README.md`'s *Quickstart* is two columns: run it yourself, or use the hosted service
  at memvara.dev.** The README described the hosted product in one cell of an *Other ways
  in* table below the fold, which understated it — the hosted path is the one that needs
  no install, no key and no Python at all.

  The left column is `pip install memvara` and a SQLite file you own, plus `memvara-mcp`
  as a local server. The right column leads with the MCP address,
  `https://app.memvara.dev/mcp`, approved once in a browser over OAuth so the client holds
  a revocable grant rather than a stored secret; then the Claude Code plugin, which wires
  that same URL and the skill in one step; then `pip install 'memvara[cloud]'` and
  `Memvara(api_key=…)` for your own code, with `/v1` and a bearer key for anyone not
  writing Python. It states the free tier — 12,000 memories and one project, held rather
  than granted monthly, plus 2,000 recalls a month that do refill, and a refusal naming
  the next plan rather than a surprise bill — read off `memvara.dev/pricing` as rendered,
  not off the source that generates it.

  *Other ways in* is removed because both columns now say what it said, at greater length
  and in the right place: its editor cell is the hosted plugin, its `memvara-mcp` cell is
  self-serve, and its hosted-client cell is hosted. The safety property that cell carried
  — a bare `Memvara()` never becomes remote, because the dispatch reads the explicit
  argument and never the environment — is kept, in the hosted column.

  **Two things were wrong and are corrected by the move.** The editor plugin was presented
  as a local option; it wires `https://app.memvara.dev/mcp`, and its own marketplace entry
  calls it "Hosted MCP plus the skill for using it". And the client list pointed at
  `memvara.dev/docs/agents`, a retired URL that redirects to `/docs/cloud` — so a reader
  after a *local* editor one-liner was sent to the hosted client list. The columns now
  point at `/docs/self-hosted` and `/docs/cloud` respectively.

- **The top of `README.md` is a landing page, so a reader can decide inside the first
  screen.** A navigation row under the badges links the demo, the quickstart, the
  documentation index, PyPI, the site and the issue tracker. A five-row strip under
  `pip install memvara` names what the library does — bitemporal, deterministic,
  auditable, historical, LLM-light — one line each. *The problem* lost a third of its
  prose and gained a picture of the two designs side by side: values ranked by
  similarity, against values carrying the interval each held. It now closes by pointing
  at `docs/concepts/why-memvara.md` for the long version rather than telling that story
  twice, and says in one sentence that Memvara is neither a vector database nor a
  replacement for RAG, with a link to the section that explains where it does fit.

  *Use cases* moved above *Why Memvara* and opens on the coding-agent case, quoted from
  `examples/coding_agent.py`. A new test in `tests/test_docs.py` matches that block
  against what the program actually prints, so the quotation cannot go stale without the
  suite going red — three of its lines were already pinned by `tests/test_examples.py`,
  and the questions and spacing around them were not. *At a glance*,
  under *Architecture*, gives a reader who will not read the whole file the nine
  implementation facts in a table: the unit of memory, the two axes, how a conflict is
  decided, what provenance holds, how retrieval scores, the store, the dependency, where
  a model is and is not involved, and the Python floor.

  No section was removed and no claim was softened. The one thing deleted is a
  duplication this change created: the *Coding agents* row in the use-case table, which
  the new lede above it now says at greater length.

  Three rows of the new strip were checked against the code and tightened before they
  shipped. *Historical* said "every read" takes the three time keywords; `recall()`,
  `get()` and `since()` take none of them, deliberately — `recall()`'s docstring says why
  — and `ask()` spells it `at=`, so the row now names the eight reads that do. *Auditable*
  said every claim keeps the episodes it came from, which is not true of a `remember()`
  called without `sources=`, as the block at the top of the file is; it now says a claim
  carries the episodes cited for it. *LLM-light* said `add()` batches what survives into
  one call, and a surface form the registry has not seen costs a second for predicate
  acquisition; it now says one *extraction* call, which is the one being counted.

- **`tests/test_docs.py` checks the README's links back into its own headings.** The
  navigation row is three of them, spelled absolutely because this file is also the PyPI
  project description — which put them between the two guards that existed:
  `test_doc_links` skips an absolute URL by design, and `test_docs`'s pattern for a link
  into this repository requires a `/blob/main/` path these do not have. GitHub answers an
  unknown fragment with the page and no scroll, so a renamed heading breaks the first
  thing a reader clicks and nothing anywhere goes red.

- **`README.md` is rewritten for a developer arriving from a search result.** It opens
  with the claim, the three-answer example and `pip install memvara` above the fold, then
  the problem, the demo, the quickstart, and one section each for the three things that
  distinguish this library — the two clocks, contradiction resolution without a model, and
  provenance — followed by an architecture diagram, the use cases, the integration table
  and a RAG comparison. The mem0 comparison and *Development* are kept as they were,
  because they were the parts already doing their job. Nothing was softened: the
  qualifiers on every measured number are still attached to it. *Honest limitations* was
  kept intact too and has since moved to its own page — see the entry below.

- **`tests/test_doc_links.py` walks `docs/` and `examples/` recursively.** It globbed
  `docs/*.md`, which was correct while every document sat directly in `docs/` and silently
  stopped covering anything the moment a subdirectory existed. A guard that quietly narrows
  is worse than one that was never written, because the suite still reports it as passing.


- **`memory_add`'s `role` now says what it decides, and the skill stops calling the tool
  inert.** The skill said prose handed to `memory_add` was often accepted and not stored.
  True of the model tier, false of the write, and a reader who concludes the tool is inert
  stops thinking about what they hand it. The deterministic matcher runs on every
  `role="user"` turn whatever `MEMVARA_LLM` is, searches rather than anchors, and removes
  quotation marks before matching, so a first-person statement quoted inside a log, a
  docstring or a pasted document cannot be told from one the person wrote — and it lands at
  0.95, above `AUTHORITY_SHARE` against a value they stated outright. On 2026-08-26 a
  pasted log quoting `write/fast.py`'s own docstring wrote four claims and replaced a real
  stored name; nothing raised, because a matched write is an ordinary successful write.
  The mechanism now lives in the `role` description, which every MCP caller loads, and in
  `Memvara.add`'s docstring and `docs/API.md` for the library and REST surfaces, which the
  first draft of this change left out. It says three things it did not before: that a
  configured model reads the same `user` turn for anything else in the vocabulary; that
  `system` stops extraction but not near-duplicate reinforcement, which runs at every role
  and can still attach a pasted turn to an existing claim as a source; and that `assistant`
  is declined by the gate exactly as `system` is, so the old "stored but trusted less" was
  describing a mechanism that does not exist. The skill keeps only what no description can
  reach: whose voice the text is, that one call carries one role so a mixed turn takes two,
  and that a note invented this way held at no instant — `memory_forget` takes it back,
  `memory_end` would record a change that never happened. No behaviour change: `role` has
  accepted these three values since the rename.

- **The skill now tells an agent to close a note its own work disproved.** The
  correction sequence opened "when they say a memory is wrong", so every path through it
  was gated on the person raising it — and the commoner case is that nobody does: a note
  comes back in recall, the turn's own work makes it false, and nothing notices. The
  person cannot see the store and the next session reads the same note and believes it.
  Measured on a live store on 2026-08-30: six claims closed by hand in one sitting, five
  stale by mechanism rather than by anyone's mistake, one stale for four days *with its
  own closing instruction stored beside it* — naming the claim id and the right verb, at
  confidence 1.0. Knowing how to close a claim was never the gap.

  Prose only, and deliberately not a tool or a poller. `recall(with_ids=True)` already
  exists and `server/tools.py` deliberately does not pass it, because an agent that needs
  a handle is sent to `memory_search`; the missing piece was the instruction to go and get
  one, plus the reminder to check a claim against the thing it describes rather than
  against another note in the same store.

### Fixed

- **Twelve links still pointed at `/docs/agents`, and one of them was a real 404.**
  `memvara.dev/docs/agents` is retired; it 301s to `/docs/cloud`, so eleven of the twelve
  were working links costing a reader an extra hop. The twelfth was not: the tool
  description in `memvara/server/mcp.py` sent an agent to
  `memvara.dev/docs/agents/skill`, which has no redirect entry and no route, and the
  Worker's asset binding runs `not_found_handling: "none"` — so it answers 404. That is
  the one that mattered, because the reader is an agent following a pointer at runtime
  and it has no second way to find the page. It now names `/docs/cloud`, which is a live
  page that describes the skill and the plugin that ships it.

  The other eleven are the `homepage` field of all seven plugin manifests — the three
  root `marketplace.json` files and the four `plugin.json` files under `plugin/` — plus
  `plugin/README.md`, `npm/memvara/README.md`, and the packaged skill's
  `references/hosted-mcp.md` with its mirror. `README.md`'s client list was moved to
  `/docs/cloud` in an earlier entry above; these were the copies that move did not reach.

- **The packaged skill said a hosted OAuth grant lasts 90 days. It lasts ten years.**
  `DEFAULT_OAUTH_REFRESH_TTL` is `timedelta(days=3650)`. The 90-day figure was corrected
  in `docs/integrations/mcp.md` and left standing in five other files:
  `memvara/skills/memvara/SKILL.md`, that skill's `references/hosted-mcp.md`, both of
  their mirrors under `plugin/skills/`, and `plugin/README.md`. The skill is the
  expensive one — it is vendored by sha into seven downstream repositories, and it is
  read by an agent that cannot go and check.

  **The number dragged an inverted instruction behind it**, which is the part worth
  reading twice. At 90 days, *"a forgotten connector does not stay authorized forever"*
  is true and "open the approval page and click Allow again" is something a reader will
  need to do. At ten years both are wrong, and the second sends somebody to re-approve a
  grant that never lapsed. All three files now say the grant lasts until you revoke it,
  or ten years, whichever comes first, and that it does not lapse on its own — so a
  connector you stop using stays authorized until somebody revokes it in the console.

  `tests/test_docs.py` gains a third `RETIRED_WORDINGS` entry so the figure cannot come
  back. Its pattern tolerates the markup and the line break the live copies actually had
  — `lasts **90 days**` and `It lasts\n90 days` both match — and it leaves alone the
  unrelated 90-day spans elsewhere in the repository, such as the decay half-life in
  `memvara/retrieve/scoring.py`.

- **Two more copies of "every read takes the same three time keywords".**
  `docs/concepts/bitemporal-memory.md` now names the eight reads that do take them and
  the three that do not — and its heading, *The three reads*, is now *The three time
  keywords*, which is what the section is actually about and no longer a number
  contradicting the sentence under it. `docs/API.md` said "every read below takes" over a
  listing that includes `recall`, `get` and `since` and annotates each four lines later
  as taking no `T=`, a page contradicting itself within a screen; it now says that a read
  shown with `T=` takes them and a read shown without one takes no time keyword at all.

- **`CONTRIBUTING.md` gave the suite as 3,539 tests where the README gives 4,044**, for
  the identical command. One claim, two files, one of them updated — the failure this
  guard was written to stop, found inside the commit that added it. Neither pattern in
  `RETIRED_WORDINGS` catches it, and neither should: a stale number is not a wording, and
  a guard that pretended to cover it would be the reassurance without the check.

- **`docs/concepts/temporal-retrieval.md` said "Every read takes the same three time
  keywords".** The third copy of a claim corrected twice already: `recall()`, `get()` and
  `since()` take none of them and `ask()` spells it `at=`, so a reader following the
  general form gets a `TypeError`. It now says eight, which is the number that do. The
  guard added above is what found it.

- **`docs/integrations/mcp.md` said a hosted OAuth grant is "good for 90 days".** It is
  ten years, or until you revoke it. `memvara-cloud`'s `control/store.py` sets
  `DEFAULT_OAUTH_REFRESH_TTL = timedelta(days=3650)`, and no 90-day TTL exists anywhere in
  that service — the session is 14 days, the device grant 15 minutes, the access token an
  hour, and the only 90-day numbers are job and governance retention, which are not this.
  The page understated the grant by about 40x, in the direction that costs a reader work:
  somebody reading it re-approves every quarter for no reason. `memvara.dev` has said ten
  years all along, so the site and the backend agreed with each other and this file
  disagreed with both.

- **The same paragraph sent readers to `memvara.dev/docs/agents`**, a retired URL that
  redirects to `/docs/cloud`, and listed five clients where there are nine. Both are
  corrected, and this file is where it matters most now: the README's new *Quickstart*
  links here directly.

- **The README and `docs/FAQ.md` said `add()` batches what survives into "a single
  call".** It is a single *extraction* call. A predicate the registry has not seen before
  costs a second one for acquisition, and `WriteReceipt.llm_calls` counts both — so the
  sentence understated the cost of the first write to a novel vocabulary, which is
  exactly the write somebody measures. The FAQ copy is the one that mattered most: it sat
  four lines above that page's own promise that "`WriteReceipt.llm_calls` reports the
  cost, so the claim is checkable", which invites a reader to go and check it and then
  find two.

- **The README's *Temporal memory* section said "every read takes all three".** The list
  of eight that follows it was right; the sentence introducing them was not. `recall()`,
  `get()` and `since()` take none of the time keywords, and `ask()` spells it `at=`, so a
  reader acting on the general form gets a `TypeError`. It now says eight, and names the
  three reads that are not among them.

- **`examples/coding_agent.py` printed its timeline one column out of true.** The value
  field was padded to 27 and `OAuth 2.0 client credentials` is 28 characters, so the
  longer row's date started a character to the right of the shorter one's. Widened to 28.
  The output was always correct and never lined up; it is quoted in the README now, which
  is what made a cosmetic defect worth a commit.

- **Eight more places in the library said the same thing, and they are corrected too.**
  `memvara/integrations/__init__.py`, `langgraph.py`, `langchain.py`, `llamaindex.py` and
  `tests/test_integrations_langgraph.py` described contradiction resolution as retiring
  the value it displaces. One of them is a tool description a model reads at runtime
  (`llamaindex`'s memory-block `description`), which is the category where imprecision
  costs most; it now uses the phrasing `langchain.py` already had, "contradictions
  already resolved", which is both accurate and the word a test pins. Three docstrings in
  `tests/test_integrations.py` said the same thing, one of them contradicting the CrewAI
  adapter's own "Ended rather than retired" two files away. No behaviour changed; the code was always right and only the prose was
  wrong. `test_a_changed_field_supersedes_only_itself_and_leaves_the_others_alone` now
  asserts `state == "ended"` and `invalidated_at is None`, so the distinction is a tested
  property rather than a described one — it asserted the supersession pointer but never
  which clock closed, which is exactly the gap the wrong word lived in.

- **Six places said a superseded fact is *retired*. It is *ended*.** That is the
  distinction this repository treats as the product, and the wrong word records a false
  reason for a change that nothing downstream can detect. The README's feature table,
  `examples/coding_agent.py`, `examples/README.md`, `docs/integrations/frameworks.md`,
  `docs/guides/coding-agents.md` and `docs/ROADMAP.md` all now say *ended*. Two of those
  were pre-existing rather than new. The LangGraph section gained the distinction it was
  eliding: a field whose **value changes** is an ordinary supersession and is stamped
  `ended`, while a field that **disappears** from the item is `delete()`d and so is
  `retired`, with no `invalidated_by` because nothing replaced it — verified against
  `_write_field` and `_apply_put` rather than against the prose around them.

- **`docs/README.md` said the FAQ had ten questions; it has eleven.** The changelog's own
  page count double-counted `docs/README.md` and `docs/FAQ.md` as being under the five
  subdirectories, where thirteen pages sit.

- **The README's hero block showed `mem.get_all()  # -> ['New York']`.** It returns
  `Claim` objects; every other block in the file writes `[c.object for c in …]`. The one
  snippet a reader meets first was the one that would not have done what it showed.

- **The Development section's closing paragraph duplicated the new Contributing section**
  eight lines below it.

- **`docs/getting-started/first-memory.md` did not run.** Rewritten so it does, with every
  printed value taken from a real run: the stores are in-memory so a section that says it
  starts fresh genuinely does, the three corrections are shown as alternatives from one
  starting state rather than as a sequence that closes the slot on the first, and the page
  now says what `at=` earlier than a claim's own `valid_from` does — `valid_to` clamps
  forward to `valid_from`, giving a zero-length interval that answers at no instant, with
  nothing raised.

- **Two architecture diagrams named a method that does not exist.** `README.md` and
  `docs/reference/architecture.md` both listed `end` on the `Memvara` facade. There is no
  `Memvara.end`; ending a fact is `forget(close="ended")` or `delete(close="ended")`. Both
  diagrams now name `erase`, and `test_every_method_the_architecture_diagram_names_exists`
  reads the facade node out of each diagram and asserts every method in it is real, so a
  picture that grows a method the class does not have fails.

- **Two new guards did not guard.** `test_the_readme_names_no_repository_path_relatively`
  inspected raw-HTML `href=` only, so an ordinary relative markdown link — the commoner way
  to write that mistake — passed it and passed `test_doc_links`, whose "does this resolve"
  question a relative path answers correctly from the repository root and wrongly from
  pypi.org. And `test_every_example_is_listed_in_the_examples_index` fell back to
  `script.parent.name`, which is `examples` for anything at the top level and appears
  throughout the index, so it passed for every conceivable file. Both now fail on the case
  they are named for.

- **`test_examples.py`'s `run()` promised an environment it did not build.** Its docstring
  said "no `PYTHONPATH`" while the subprocess inherited it, so on a machine exporting
  `PYTHONPATH=.` — which `bench/` and `demo/` both need — the examples would have been
  tested against the checkout rather than the installed package. It now passes an explicit
  `env` with `PYTHONPATH` removed.

- **The `pyproject.toml` hand parser mis-read comments.** Once arrays could span lines, a
  comment line whose prose ended in `]` closed the array early and an inline comment after
  an entry became an entry — both producing a silently wrong table rather than an error, on
  3.10, which is the interpreter where the `tomllib` agreement test skips. Comment
  stripping is now quote-aware, and a second agreement test — against a fixture holding a
  bracket-ending comment, an inline comment and a `#` inside a string, none of which
  `pyproject.toml` contains — pins it. Without that fixture the new code path was
  unexercised on 3.11+ and skipped entirely on 3.10, which is the interpreter the hand
  parser exists for.

- **`examples/temporal_memory.py` claimed a test that does not exist.** Its docstring said
  the output matches `examples/README.md` line for line and that the suite asserts it;
  the index holds no transcript and nothing compared them. It now states what is actually
  asserted.

- **A `### Fixed` heading was inserted above two pre-existing `Unreleased` entries**,
  silently reclassifying a behaviour change as a fix. Moved below them.

- **`docs/DESIGN.md` printed an `Explanation.summary()` the code no longer produces.** The
  line was missing `raw=` and `intent=`, both of which the retriever has emitted for some
  time. Replaced with a real one, reproduced from a store the surrounding example
  describes, plus a sentence on why the arrow points at the normalised score rather than
  at `raw`.

### Documentation

- **The comment justifying `listChanged: false` claimed something true only of stdio.**
  `MemvaraMCPServer._initialize` (`memvara/server/mcp.py`) says the tool set "never
  changes while the process lives," which is honest for a stdio client — it spawns the
  process and dies with it, so the promise and the client's connection have the same
  lifetime. A hosted deployment reuses this same class per request behind one long-lived
  client connection instead, so a client there can outlive many processes across a
  redeploy that adds a tool, and sits on a stale `tools/list` with no signal that
  anything changed. Nothing in this repository was wrong — the hosted transport and its
  own reasoning about the missing notification channel live in `memvara-cloud` — but the
  comment here read as a blanket guarantee rather than one scoped to stdio, which is
  exactly the gap `memvara/memvara#94` traces back to it. The comment now says which
  transport the promise is true of, and names the file to change first if this class
  ever grows a hosted-aware caller.

  `docs/integrations/mcp.md` now says the same thing where the people affected are
  actually reading — under the tool table, since a missing tool is the symptom — and
  `docs/RELEASING.md` asks any release that adds or renames a tool to carry one line
  saying connected sessions keep the old list until they reconnect. The comment fix
  alone would have reached contributors and nobody else; a hosted user in a long-lived
  editor session is the case #94 measures, and they do not read source comments.

## [0.9.0] — 2026-08-30

### Added

- **`Memvara(api_key=...)` returns a client for a hosted deployment.** `RemoteMemvara`
  serves the library's own read and write surface out of the `/v1` API instead of a local
  store, and hydrates every response into the same `Claim`, `Episode` and `WriteReceipt`
  dataclasses, so calling code cannot tell which it holds. `Memvara.connect()` is the same
  client using whatever credentials are around — `MEMVARA_API_KEY`, then the file
  `memvara-mcp login` writes. Install `httpx` with it: `pip install "memvara[cloud]"`.

  **A bare `Memvara()` never becomes remote.** Dispatch keys on the explicit `api_key=` or
  `base_url=` argument and never on the environment, so a script that has always written
  to a local file cannot start posting to a hosted store because somebody ran
  `memvara-mcp login` on that machine. The environment supplies the *value*, once the
  caller has asked for remote.

  Two behaviour differences from the local engine, both deliberate and both documented in
  `docs/API.md`: `consolidate()` returns a job handle rather than per-operation counts,
  because the endpoint answers 202 before the pass starts, and there is no
  `prove_erased()`, because `erase()` returns its per-table evidence itself. Arguments the
  API cannot honour are absent rather than ignored — `recall()` has no `with_ids`, and
  raises on a `budget` it cannot enforce.

  Naming a local subsystem alongside a credential is a `TypeError`, not a silent no-op:
  `path=`, `store=`, `embedder=`, `llm=`, `registry=` and `reembed=True` all describe an
  engine that runs server-side.

  **Three attempts per call, and the wait between them is bounded.** A call is retried on
  an error the deployment marked retryable, on a 429 — including one an edge proxy returned
  with no envelope, which is classified from the status rather than falling through to
  `InvalidRequest` — and on a connect-phase failure that never reached the server. A
  `Retry-After` is waited for as asked up to thirty seconds; a longer one raises
  `RateLimited` immediately, carrying the server's own number on `retry_after`, instead of
  blocking the call — or, on `AsyncRemoteMemvara`, the event loop — for as long as the
  header says. Whether an hour is an acceptable wait is the caller's decision to make.

- **`AsyncRemoteMemvara`: the hosted client, awaited, on a real async transport.**
  Importable as `from memvara import AsyncRemoteMemvara`, same as `RemoteMemvara`. Every
  method on `RemoteMemvara` has an `async def` twin of the same name taking the same
  arguments, plus `aclose()`, `__aenter__` and `__aexit__`. It does not use
  `memvara.aio`'s `asyncio.to_thread` wrapper — `httpx.AsyncClient` already speaks `/v1`
  without blocking a thread to do it, and there is no engine underneath the transport for
  coroutine-colouring to propagate through, so wrapping the blocking client in a thread
  would only be worse. See `memvara/aio.py`'s module docstring for where that module's
  own async argument stops applying.

- **`MEMVARA_MODE=cloud` starts a server instead of refusing to.** `memvara-mcp` in cloud
  mode now builds a `RemoteMemvara` — a client of the `/v1` facade — and serves the same
  fourteen tools from a hosted deployment. It reads `MEMVARA_API_KEY`, or the credential
  `memvara-mcp login` wrote, exactly as `ServerConfig.from_env` already did. Needs `httpx`:
  `pip install "memvara[cloud]"`.

  **The engine is still never run against a remote store**, which is what the old refusal
  protected and is not being overturned. `docs/OPEN-CORE.md` records why, and the guard
  that enforced it — `config.cloud_gap()`, a set difference over `RemoteStore.WIRED` — has
  been **deleted**, along with `_ENGINE_NEEDS` and `_CLOUD_NOT_WIRED`. It was built to
  empty out on its own the day the facade grew the low-level endpoints; that day does not
  arrive under this design, because cloud mode bypasses the `Store` seam rather than
  completing it, and a dead gate left in place reads as a live one. The test that guarded
  it was replaced rather than removed: `test_no_cloud_path_anywhere_constructs_an_engine_
  over_a_remote_store` reads `config.py`'s syntax and fails if a `RemoteStore` import or a
  `store=` argument returns.

  **`MEMVARA_LLM` and `MEMVARA_EMBEDDER` are now refused under cloud mode** when set to
  anything but their defaults. Extraction and embedding run inside the deployment, so this
  process would read the setting and never use it — and an operator who set
  `MEMVARA_LLM=anthropic`, saw a server start and believed their writes were being
  extracted has been told something false by a program that stayed silent. The error names
  the variable. `memory_stats` reports the deployment's own answer.

  **A read-only API key hides the write tools, and `memory_stats` reports the deployment's
  own extractor.** The server reads `GET /v1/stats` once at startup, through the new
  `RemoteMemvara.service()`. Without it a cloud server started with a read-only credential
  listed every write tool and the deployment refused them mid-conversation as a 403 — to a
  model that cannot act on it, which is the failure shape the old refusal existed to
  prevent, one layer along. `MEMVARA_READ_ONLY` and the credential are **OR-ed, never
  overridden**: a server configured read-only stays read-only whatever the token allows,
  and the credential can only narrow. Any failure of that one call degrades to the
  previous behaviour — `extractor` reports `unknown`, `read_only` falls back to the
  environment — so a deployment that is briefly down does not stop the server starting.

- **`RemoteMemvara.service()` and `AsyncRemoteMemvara.service()`.** The `GET /v1/stats`
  envelope whole — `{scope, visible, tenant_counts, extractor, read_only}` — beside
  `stats()`, which keeps returning `tenant_counts` alone so that `stats()["claims"]` is a
  number against either engine. Both take `attempts=` and `timeout=`, which override the
  client's own for one call: a startup probe wants an answer in seconds or not at all, and
  the client's three attempts at a thirty-second timeout plus backoff is about ninety
  seconds of silent startup before a hanging deployment reaches the safe default. The two
  overrides are on `HttpClient.request` and `AsyncHttpClient.request` for any caller with
  the same need; every other call leaves them unset and keeps the client's.

- **`memory_standing` is answered server-side against a hosted deployment.** The tool used
  to page every live memory in the scope and filter for procedural ones in Python — across
  the network, for a handful of rows, in the tool a session calls at startup.
  `GET /v1/standing` does the filter at the source. Local behaviour is unchanged: the local
  scoped view has no `standing()`, so it takes the same path it always did. One divergence,
  in the last line of the reply rather than in the facts — the endpoint caps at `k` and
  reports no total, so the "(N more not shown)" hint does not appear against a hosted
  deployment.

### Changed

- **`memvara-mcp init --mode cloud` refuses on a missing `httpx` rather than on an unwired
  store.** The reason a cloud server could not start has changed, so the gate did.
  `init` writes a config it never launches, so it and the server have to answer the same
  question the same way or the disagreement is silent — that was the point of
  `cloud_gap()` and it is the point of this.

- **The predicate fold note is read off the claim the store wrote back, not off a
  registry.** `memory_remember`, `memory_forget` and `memory_end` each say when the
  predicate acted on is not the one asked for — `uses_tool` held as `prefers_tool`. That
  answer came from `Memvara.registry`, which a hosted deployment does not have, so all
  three write tools raised `AttributeError` under `MEMVARA_MODE=cloud`. Every claim these
  tools already hold carries the canonical predicate, so comparing it against the caller's
  own spelling is exact on both engines — and is the stronger statement of the two, being
  what the store did rather than what a registry says it would do.

  **One sentence was dropped and its absence is a decision.** The note used to add, on a
  write folding onto a single-valued predicate, that the slot now keeps one value where an
  unseen predicate would have accumulated. Cardinality is a property of the predicate's
  spec and nothing on the wire carries it. Restoring it needs the deployment to answer for
  its own vocabulary. `_fold_note`'s docstring and
  `test_the_fold_note_no_longer_says_what_the_fold_did_to_the_cardinality` both say so, so
  the loss is visible rather than quiet.

- **`ToolContext.memory` is typed `MemoryAPI` rather than `ScopedMemvara`.** A protocol in
  `memvara/server/memory_api.py` declaring the nineteen members `tools.py` calls, which
  both `ScopedMemvara` and `ScopedRemoteMemvara` satisfy — this is what lets one tool table
  serve either engine. The security property is unchanged and now stated as a property
  rather than a class: scope is bound once at construction, so a handler has no argument
  and no attribute with which to address another tenant. The remote view holds it twice
  over, since the deployment resolves the tenant from the bearer token rather than from
  anything a request can name.

- **`RemoteMemvara.search` and `ScopedRemoteMemvara.search` carry the same three overloads
  `Memvara.search` has.** Without them the identical expression typed as `list[Retrieved]`
  against a hosted deployment and `list[Result]` locally, so code reading `.claim` off a
  row type-checked against one engine and not the other. No runtime behaviour changes.

- **`RemoteStore` raises on a null `valid_from` or `recorded_at` instead of substituting
  the current time.** Both are declared required and non-nullable on the wire model, so a
  null in either is the server disagreeing with its own schema. The old fallback was
  `datetime.now()`, which is *naive*: the `Claim` came back carrying one naive instant
  among aware ones, looked well-formed, and made `Claim.is_live()` raise
  `TypeError: can't compare offset-naive and offset-aware datetimes` — a call nowhere near
  the response that caused it. `RemoteMemvara` already raises here
  (`remote/hydrate.py:_required_dt`); the two decode one wire model and must not disagree
  about a malformed response. No working deployment reaches this: it needs a response that
  violates the schema `/v1` publishes.

### Fixed

- **`RemoteStore.get_claim()` and `get_claims()` no longer raise on Python 3.10.** They
  parsed the facade's instants with a bare `datetime.fromisoformat`, which did not accept
  a trailing `Z` before 3.11 — and `/v1` renders every instant in exactly that form. So
  every memory read through a `RemoteStore` raised `ValueError` on the oldest interpreter
  `requires-python` claims to support, while working on 3.11 and later. A trailing `Z` is
  now rewritten to `+00:00` before parsing, matching `remote/hydrate.py`,
  `server/tools.py` and both importers in `compat/`.

  The tests could not see it: their fixtures spelled instants `+00:00`, which is what
  `datetime.isoformat()` produces and not what the server sends. They now carry the `Z`,
  and one test patches in a parser that rejects `Z` so the guard fails on every
  interpreter rather than only on the one leg of the matrix where the bug was real.

## [0.8.1] — 2026-08-27

### Added

- **`memory_standing` and `memory_since` say which rows a machine derived.** 0.8.0 gave
  `recall()` a per-row `(inferred)` marker; the tools that render through `_delta_lines`
  carried none, so the marker reached a ranked per-turn excerpt and not the enumerated set
  a client reads at session start — the surface where "which of these did a machine write"
  is actually the question. `memory_standing` already ordered stated rules above inferred
  ones by confidence, but ordering tells a reader the list is sorted without telling them
  where the boundary falls; in a block of twenty-odd rows the twelfth is unknowable.

  The marker is a bracket field, `[id=… procedural live inferred]`, not a suffix. That row
  format puts metadata first and the untrusted span last precisely so nothing trusted can
  follow text a claim could impersonate; `recall()` can suffix its rows because they carry
  no metadata at all. Note the bracket was already variable-width — `_state` appends an
  instant for `ended` and `retired` — so a consumer pinning a field count was already
  wrong for those rows. Read it as a set of tokens.

- **The skill says what the `(inferred)` marker means.** The tools emit it and nothing
  told a reader what to do about it, which is worse than not marking at all: the reader
  either discounts every row or ignores the marker on all of them, the exact failure the
  per-row marker was added to end. `references/project-instructions.md` now says to treat
  a marked row as the weaker of two that disagree, never to quote one back as something
  the user stated, and to reach for `memory_why` to check it — plus the half that is easy
  to miss, that an unmarked row is not thereby the user's own words, only one no component
  claimed to have derived.

### Changed

- **The stated-versus-derived rule has one home, `core.is_derived`.** Lifted out of
  `Memvara._derived_suffix`, which now calls it, because two renderers read it and a rule
  restated in two places is a rule that will disagree with itself. That is not
  hypothetical: two agents reimplemented this predicate from prose on the same afternoon
  and both got it wrong in opposite directions — one used a substring test over the
  rendered `Derived by` line and silently classed a third extractor as user-stated, the
  other described it as "marked unless the extractor is api", which drops `""` and would
  mark every claim written before `extractor` existed. The unmarked set is the tuple
  `("", "api")`. The extractor's name is still never rendered.


Nothing yet.

## [npm 0.1.1] — 2026-08-26

The npm package versions independently of the Python one and ships on `npm-v*`. It is
recorded here because 0.0.2 and 0.1.0 are, and a history readable up to the point it
stopped being kept is worse than none.

### Fixed

- **The bridge's tool table was two short.** It listed the twelve tools of the deployment
  it was written against and had not gained `memory_ask` or `memory_standing`. The heading
  no longer states a count at all: what a hosted deployment serves is a property of that
  deployment, `tools/list` is the authority for a given day, and a number here is a second
  copy of a fact this package cannot check.

## [0.8.0] — 2026-08-26

### Fixed

- **`ask()` said "the same day" between two different dates.** The lag was counted by
  truncating the timedelta while `_when` renders whole days, so a two-hour gap across
  midnight produced `True since 2026-01-05, recorded 2026-01-06 the same day.` — two dates
  and a clause denying they differ, in the method whose subject is when things happened.
  `late` had already accepted the claim as lagging, so the filter and the sentence
  disagreed as well. Counted between the calendar dates the sentence prints.

- **`recall()` marks a note nobody stated.** Every row rendered as the same shape of line,
  so a fact the user stated outright and one a capture hook mined from an assistant's own
  prose arrived in a model's context indistinguishable. The block header asserts authority
  over the whole set and notes that some of it was inferred, which leaves a reader unable
  to tell *which* — so the qualifier either discounts every row or is ignored for all of
  them. A derived note now ends ` (inferred)`.

  **Keyed on provenance, not on a confidence floor.** This repository's position is that
  such constants do not survive a change in store size — `min_score` defaults to no floor
  for exactly that reason — and a threshold tuned on one store marks everything or nothing
  on another. Two things count as derived: a `derivation` other than `USER`, and an
  `extractor` other than `api`. The second is the one that bites, because `remember()`
  stamps `USER` whatever called it, so a hook writing through it looks exactly like the
  user speaking until you read the extractor.

  **Cost, measured rather than asserted.** Nothing is marked on a store of stated facts.
  On `demo/`'s corpus, where every fact was extracted from a support transcript, the
  structured arm's prompt goes from **430 to 440 tokens** — so the demo's size result moves
  from 5.7× to **5.6×** fewer tokens than the whole-transcript arm, restated in
  `docs/BENCHMARKS.md`, `demo/README.md` and `docs/ROADMAP.md`. A marker that also named
  the deriver was measured at 24% inflation and rejected; `memory_why` reports it on
  demand, and leaving the caller-supplied extractor out of the block removes an injection
  surface as well as the cost.

- **`ask()` rendered a single-valued slot holding two live values as a conjunction.**
  `user lives_in: Berlin, Lisbon.` — commas mean "all of these are true at once", which is
  right for a multi-valued predicate and false for one the schema says holds exactly one.

  It happens whenever `AUTHORITY_SHARE` refuses a displacement: a candidate worth less
  than half the incumbent is stored beside it rather than ending it, deliberately, because
  ending a true claim destroys information where keeping two only degrades ranking. That
  trade is right; what was missing is that the reader was never told a contest had
  happened. `ask()` now says which value holds the slot and which did not displace it,
  named by confidence — the axis the refusal was decided on:

  ```
  user lives_in: Berlin, Lisbon.
    Single-valued, so only one of those can be true. 'Berlin' (1.00) holds it;
    'Lisbon' (0.49) did not displace it.
  ```

  Multi-valued slots are untouched, where several live values are the design rather than a
  contest. `Reading` gains `single_valued`, a bool rather than the `Cardinality` it derives
  from because `schema` imports `types` and the enum cannot travel back the other way.

- **A claim filed under the wrong `memory_type` can now be corrected, and asking no longer
  reinforces the mistake.** Re-asserting a triple this store already holds is a
  re-observation, so it reinforces rather than forking the record — and the `memory_type`
  on that write was dropped. Correcting a filing therefore reported `already-known 1`, left
  the type alone, and *raised the claim's confidence*, making the wrong filing more
  strongly believed. The only route out was `forget()` and re-create, which records that
  the record was wrong when the content is right and only the filing was.

  `remember(..., memory_type=...)` now re-files the stored claim and the receipt says so,
  on `WriteReceipt.retyped` and in the `memory_remember` receipt over MCP. The move is
  stamped `meta["retyped_from"]`, mirroring `consolidate.promote_pass` — which has always
  reclassified a live claim in place, so the operation is not new, only the caller's route
  to it.

  This is not decoration: `memory_standing` returns `procedural` and nothing else, and
  clients inject that set at the top of every session, so a project fact misfiled as
  `procedural` is carried into every later conversation until it is corrected.

  **Only an asserted type moves a claim.** `remember()` with no `memory_type` takes the
  predicate's default, which is nobody's opinion, and extraction never reaches this path at
  all. Agents re-assert known facts constantly without a view about filing, so treating any
  difference as a correction would let the last writer win — and the last writer is usually
  the one who said nothing. `derivation` is left alone for the same reason: only the drawer
  moved, and where the fact came from is the more important of the two.

- **`ask()` attached one value's dates to a whole slot, in the method whose job is dating
  belief.** On a multi-valued predicate every live value is rendered as one sentence, and
  the provenance line beneath it named no value at all:

  ```
  user prefers: dark mode, tabs over spaces, no AI attribution.
    True since 2026-01-05, recorded 2026-01-06 — 1 days later.
  ```

  Three values, three different date pairs, one line reading as a statement about all
  three. On a production store it was thirteen values under a date pair roughly three
  weeks wrong for the one being audited. `why()` returned the right dates for each claim
  throughout, so the store held them and the renderer discarded them.

  **The selection was never the bug and has not changed.** `ask()` reports the widest gap
  between the two clocks in a slot, deliberately, because the value that waited longest to
  be recorded is the one worth surfacing. What was wrong is that the sentence claimed a
  scope it did not have. It now names the value when the slot holds more than one:

  ```
    'dark mode' is the widest gap here: true since 2026-01-05, recorded 2026-01-06 — 1 day later.
  ```

  A slot with a single value keeps the sentence it had, because there is nothing to
  disambiguate. Rendering only — no stored data was wrong and nothing needs migrating.

- **`— 1 days later`** is now `— 1 day later`. Same expression, and it was in the output
  of the method that exists to report dates.

## [0.7.0] — 2026-08-26

### Added

- **`split_entity` — one surface form that has been two different things.** The inverse of
  `EntityRegistry.learn_alias`, and the repair `backfill_entities` is for the other
  direction. Identity is a fold over the surface form and nothing else, so two people who
  share a name are one entity:

  ```
  John Smith works_at Acme    (2018)
  John Smith works_at Globex  (2026)
  → Acme ended, why(Globex).superseded == [Acme]
  ```

  A job change nobody wrote, reported by `history()` as a timeline and explained by
  `why()` with a supersession pointer — the same failure `entities.py` was built to fix on
  the *spelling* axis, arrived at from the other side.

  `split_entity(reconciler, scope, "John Smith", at)` re-stamps every claim before `at`
  onto a distinct identity, undoes the closures that crossed the boundary so both
  employments are live again, and leaves a dated `ENTITY_REKEY` record on each moved claim
  so `why()` can say why history changed. Retired claims move but are never *un*-retired:
  ending a claim is something the write path inferred from the fold, and retiring one is a
  caller saying it was never true. Those are counted in `retired_left` rather than silently
  kept. `dry_run=True` by default, for `backfill_entities`' reason.

  Two details decide whether it repairs what the operator was actually shown. The surface
  form is resolved through `EntityRegistry.probe_keys`, the same widening `history()`
  reads through, so a name an alias has already merged is found under every key its claims
  were written with rather than only under its own fold. And a crossing closure is matched
  whether or not it carries an `invalidated_by`: a *backdated* write — the 2018 job
  recorded after the 2026 one, which is what importing somebody's history looks like —
  takes the closure on itself in `Reconciler.apply`'s `newer` branch and names no
  successor at all. An end the caller wrote is left alone, because the fold had no part
  in it.

  **A repair and deliberately not a detector.** Nothing in the data separates "one person
  changed jobs after eight years" from "two people share a name" — not the gap, since
  `works_at` is `SLOW` and eight years is four half-lives of an ordinary job change; not
  provenance, confidence or predicate. A write-time signal would fire on every long-gap
  supersession, and noise in that position teaches a reader to ignore the notes that mean
  something. What a person knows, this records.
- **`MEMVARA_PREDICATES=decisions` — a vocabulary for what an agent writes about its own
  work.** Two predicates, `decided` and `observed`, both multi-valued, both `episodic`,
  differing on volatility: a decision is `static` because one made in March was made in
  March for ever, and an observation is `slow` because it is a reading of a world that
  moves.

  This is the answer to the agent-state brief's ask for `decision` and `observation` as
  **memory types**, and the answer is that they are vocabulary. `MemoryType` is persisted
  and hydrated with `MemoryType(value)`, so a fourth member is a `SCHEMA_VERSION` bump and
  a store older builds cannot open — a cost that recurs for the fifth. The only rule a new
  member would have to enter is a hardcoded `EPISODIC -> SEMANTIC` promotion, which a
  further value cannot join without becoming a policy table nothing needs; the two other
  places that read `memory_type` to decide something, the `memory_types=` filter and
  `memory_standing`'s procedural-only selection, take a new member or ignore it without
  changing at all. And `observation` is a statement
  about *where a claim came from*, which is `Derivation`, already recorded and already
  reported by `why()`.

  The predicates were chosen from a measurement rather than from the taxonomy:
  `docs/ROADMAP.md` records a production store with **95% of its claims on undeclared
  predicates**, and `rejected` and `known_defect` from that list are already in the
  `engineering` pack — deliberately not redeclared here, because two shipped packs
  declaring one predicate would let load order decide its `memory_type` silently.

  Reversible, which is the argument that settles it: nothing here forecloses adding the
  enum member later, with usage behind it.


- **`ask()` — the question the two clocks exist for now has a method.** `recall()`
  renders the current answer. Nothing composed the one a bitemporal store can give and a
  single-clock one cannot:

  ```python
  answer = mem.ask("where do they live?", at=datetime(2026, 3, 15, tzinfo=utc))
  print(answer.text)

    where do they live?
      asked about 2026-03-15

    user lives_in: Berlin.
      On 2026-03-15 this store would have said Rome, and that is what anyone acting
      on it then acted on. The difference was recorded 2026-03-22, 7 days after the
      instant you asked about.
  ```

  Three readings per fact slot — `now`, `then` (what we believe today was true at that
  instant) and `stated` (what this store would have answered at it). The last two
  differing is the finding, not an inconsistency: it means the record was corrected after
  the moment being asked about, so the answer somebody acted on is not the answer they
  would get today. `Answer.text` is the composed narrative and every sentence in it is
  rendered from a stored column — no model is consulted and nothing is inferred.

  **`Reading.stated` deliberately disagrees with `get_all(as_of=T)`, and that is the one
  thing to know before quoting either.** A row's `valid_to` is written *in place* by the
  write that displaces it, so a row read on its own cannot say when its own ending came
  to be believed — `get_all(as_of=2026-03-15)` returns `[]` above, applying an ending a
  week before it was recorded. `ask()` has the supersession chain in front of it and
  dates the ending at the successor's `recorded_at`, which is `why()`'s rule already. The
  full argument, including the one case it cannot recover, is in `docs/INTERNALS.md`.

  On all four facades, and over MCP as `memory_ask`, which takes the tool count to
  fourteen.

  It ranks; it does not judge relevance. `min_score` defaults to 0.0 exactly as on
  `search()` and `recall()`, for the reason argued there — so on a store that knows
  nothing about the question, `ask()` answers confidently from the nearest slot it has.
  Every `Reading` names the subject and predicate it answered from, which is what lets a
  caller see that.


- **`bench/temporal.py` — the differentiator has a number.** Every other harness in
  `bench/` measures retrieval: LOCOMO, LongMemEval, 2WikiMultihop, mem0, the multi-hop
  walk. That is the commodity half, and it is the half benchmarked against competitors.
  Two independent clocks, supersession that closes exactly one of them, and source
  authority were measured by nothing at all.

  Six families over 48 authored scenarios and 160 writes — point-in-time, delayed
  knowledge, `as_of` audit, contradiction, correction, source authority — scored as exact
  set matches against golds the generator builds before anything is written. No model, no
  network, no reader, no judge, and byte-identical on every run because every instant in
  it is a module constant.

  It earned its place on the first run. Against `origin/main` at `7b91a9a` it scores
  `source_authority` at **50.0%** and reports 8 `ended` claims that answer at no instant
  with **0** of them named by the write path — the two defects fixed below, neither of
  which moved any number in a suite of 3,448 tests. Figures, the `no-clocks` baseline
  column and the four anti-flattery constraints are in `docs/BENCHMARKS.md` and the
  file's own docstring.

  Wired into `tests/test_bench_eval.py` as a gate rather than left as a script, because
  everything it scores is a promise the library makes in prose, and prose does not go red.

### Documentation

- **The skill explains `memory_ask` and `sources`, which it had been shipping without.**
  Both arrived with the tool surface and neither reached the agent-facing docs: `ask()`
  was named once, in the tool list in `references/hosted-mcp.md`, and `sources` on
  `memory_remember` appeared nowhere at all. The count and the list agreed with each
  other the whole time, which is why nothing looked wrong.

  `references/time.md` now says when to reach for `memory_ask` rather than a single-clock
  reading — the case where the *disagreement between two readings* is the question, which
  `valid_at` alone answers by hiding. `references/write-and-correct.md` carries the part
  no single tool description can: `memory_add` reports turn ids, `memory_remember` takes
  them in `sources`, and dropping them between the two fails silently. The write
  succeeds, the claim looks ordinary, and the cost lands on the one later occasion
  someone challenges the fact and `memory_why` has nothing to show. Extraction wires it
  up on its own; a hand-composed triple does not, which is the normal case on a
  `fast-path-only` deployment.

  Written to the rule `test_the_skill_does_not_restate_a_tool_description` enforces —
  the first draft transcribed both descriptions and went red, correctly.

- **`plugin-claude.md` carries the code-review rule**, so it reaches the seven plugin
  repositories that compose their `CLAUDE.md` from it. It was already in `CLAUDE.md`
  here, in `memvara-cloud` and in `memvara-web`, and in none of the plugin repos — which
  is where PRs had been merging unreviewed.

- **The README works as a PyPI page, and ten documentation links that pointed at nothing
  were repaired.** `README.md` is the body of the PyPI project page, so its nineteen
  relative links — `](docs/API.md)` and the rest — resolved against pypi.org and returned
  404 for exactly the reader `pip install memvara` produces. They are absolute GitHub URLs
  now, checked with `twine check --strict` against a built sdist and wheel.

  Ten more resolved to nothing anywhere, on GitHub as much as on PyPI, in two kinds.
  Anchors outlived their heading: `README.md` pointed three times at sections that had
  moved out to `docs/OPEN-CORE.md` and `docs/DESIGN.md`, and `docs/API.md`,
  `docs/BENCHMARKS.md` and `docs/DESIGN.md` each carried a cross-file anchor written as a
  same-page one. And six links in `docs/` were spelled from the repository root inside a
  file one level down, so `bench/baseline.py` meant `docs/bench/baseline.py` — including
  the one `docs/BENCHMARKS.md` tells you to read *before quoting any of this*.

  `tests/test_doc_links.py` pins it: every relative link in `README.md`, `CHANGELOG.md`,
  `CONTRIBUTING.md`, `SECURITY.md` and `docs/*.md` must resolve to a file that exists, and
  every anchor to a heading that exists, checked across files because a heading can be
  renamed out from under a link in either direction. Absolute URLs are not checked — a
  network call would trade a real guarantee for a flaky one. Nothing in the suite had ever
  looked at a link, which is how ten of them accumulated in the files read first.

- **The honesty sections said things the code had stopped supporting.** `docs/ROADMAP.md`
  opened *"Status as of `v0.1.0`. 2,734 tests"* six releases later, and repeated 2,734
  inside *"N tests prove the code does what we said it does"* — an honesty claim resting on
  a number wrong by 838. The README's entity-resolution limitation covered only the
  spelling axis and now states the confident direction, naming `split_entity()` as the
  repair and saying why there is deliberately no detector. *"Two benchmarks, and only one
  of them runs the real thing"* read as a count of the repository's harnesses, contradicting
  the *Measured* table sixty lines above it, and is scoped to the mem0 pair it describes.
  *"No REST server in the open core"* was true and could be read as "no HTTP anything": the
  client half, `memvara/store/remote.py`, ships here and is deliberately partial.

### Fixed

- **The authority rule's documentation invited the wrong inference, and now says what it
  does not cover.** `AUTHORITY_SHARE` was introduced around "a 0.10-confidence guess
  replaced a 1.00-confidence statement", which is true and is not the common case. 0.70 is
  the documented default for an extraction whose model gave no figure, and
  `0.70 >= 0.5 * 1.00` — so on a single-valued predicate a mined paraphrase still closes a
  fact a person asserted outright, and still stamps it `ended`. Measured: 0.70 against 1.00
  closes it, 0.49 does not.

  That stays, because the alternative is worse — raising the share far enough to block it
  would stop the store learning from conversation, which "I moved to Lisbon" depends on.
  What changes is that the boundary is now stated, and pinned by a test rather than a
  sentence. The related case, a paraphrase *outranking* a stated preference on a
  multi-valued predicate, is not this rule's to solve: nothing is displaced there and both
  values stay live, so what goes wrong is ranking. That is issue #62.

- **A low-confidence guess no longer displaces a high-confidence statement, or records a
  world event as the reason.** Contradiction resolution was predicate cardinality plus
  write order and nothing else — `confidence` appeared nowhere in
  `memvara/write/reconcile.py`:

  ```python
  m.remember("user", "lives_in", "London", confidence=1.00, extractor="api")
  m.remember("user", "lives_in", "Paris",  confidence=0.10, extractor="llm-guess")

    London   conf=1.0   via api          state=ended
    Paris    conf=0.1   via llm-guess    state=live
    live now → ['Paris']
  ```

  The bad ranking is the smaller half of it. `ended` asserts that **the world changed**,
  and nothing about the world had changed — a machine had guessed. The correct reading is
  that the record is disputed, and the store had no way to say so. `CLAUDE.md` calls the
  `ended`/`retired` distinction the one mistake here that cannot be found by reading the
  data afterwards; this wrote it automatically, on every low-confidence extraction that
  collided with a known fact.

  A candidate now closes a claim only if it is worth at least half of it, measured on
  `confidence` — `write.reconcile.AUTHORITY_SHARE`. Below that share the incumbent stays
  live, the candidate is stored beside it, and the write reports a `Dispute` naming both
  values on `WriteReceipt.disputed`, on the `write.disputed` counter, and in the
  `memory_remember` receipt.

  **Confidence, not `Derivation`, and deliberately.** The write paths already encode
  source authority as a number: `write.fast.CONFIDENCE` is 0.95 rather than 1.0, and its
  comment says the headroom exists "to keep LLM- and user-asserted claims rankable above
  rule output when they disagree". Ranking by `Derivation` instead would mean a
  conversational extraction could never displace anything an application asserted, which
  stops the store learning — "I moved to Lisbon" arrives as `LLM_EXTRACT` and has to be
  able to end a `USER` claim written last year.

  **Half, and not some other fraction.** The share has to sit below every confidence the
  shipped write paths produce, or ordinary writes would trip it: 1.00 from `remember()`,
  0.95 from the fast path, 0.70 from an extraction whose model gave no figure, 0.50 from
  one that ignored the schema. The lowest is exactly half the highest. What is left is a
  claim whose extractor said, in the one field provided for saying it, that this is a
  guess.

  The retraction path deliberately does not consult it. A retraction writes a tombstone
  that is born invalidated, and leaving its target live beside that would put both
  sentences in the store at once.

  **Nor does `supersede()`, `forget()` or `delete()`.** All three close a claim the
  caller named, before the reconciler is asked anything, so there is no candidate to
  weigh against it — the rule arbitrates an inference the write path drew, and naming
  the row to close is an instruction rather than an inference. So a store audited after
  this release can conclude that no low-confidence *extraction* ended a high-confidence
  fact; it cannot conclude that no low-confidence claim did, because `supersede()` can
  still be told to do exactly that.

- **A supersession that leaves a value true at no instant now says so.** Two writes
  sharing a `valid_from` — any same-day correction, and every import that stamps dates
  rather than timestamps — left the older claim with `valid_from == valid_to`:

  ```
  Delhi  state=ended   valid_from == valid_to == 2026-01-10T00:00:00Z
    valid_at=2026-01-09 → []
    valid_at=2026-01-10 → ['Mumbai']      Delhi answers nothing
    valid_at=2026-01-11 → ['Mumbai']
  ```

  The clamp in `close_out` is right and stays: an interval that ends before it begins is
  not a shorter fact, it is a row no `as_of` window can return consistently. What it
  cannot do is make the row answer anything, and the receipt still read `ended 1`, which
  is what an ordinary supersession reads — while `core.py` promises that `ended` means
  "`get_all(valid_at=<while it held>)` still answers". `server/tools._interval` already
  *refused* this exact shape one module over, calling it "true at no instant, returned by
  no query"; the reconciler produced it in silence.

  It is reported rather than prevented, on `WriteReceipt.collapsed`, the `write.collapsed`
  counter and the `memory_remember` receipt. The alternative — nudging the edge forward by
  a tick — would give the displaced claim an interval that nothing witnessed, in a store
  whose argument is that its intervals come from evidence. `Memvara.supersede()` reports
  the same outcome from its own path.

- **`remember()` refuses an interval of no length**, matching what `memory_remember` has
  always done with the same input. `remember(valid_from=T, valid_to=T)` used to store a
  claim that `added 1` reported and no query on either clock returned. It is now a
  `ValueError`. `forget()` and `delete()` still clamp rather than refuse, because there
  the row already exists and refusing would leave no way to close it at all.

- **`remember()` refuses `true_since` and `true_until` instead of filing them as
  metadata.** The two surfaces spell the valid interval differently — `memory_remember`
  takes `true_since`/`true_until`, the Python API takes `valid_from`/`valid_to` — and
  `**meta` accepted the tool's spelling from either. The failure had two halves and only
  one of them was loud:

  ```python
  m.remember("user", "lives_in", "Bangalore", true_since=datetime(2023, 1, 1, tzinfo=utc))
  # TypeError: Object of type datetime is not JSON serializable
  #   ...raised in memvara/store/sqlite.py, in put_claim, naming no key
  ```

  A `datetime` reached `json.dumps` in the storage layer and died four frames from the
  call, with the offending argument nowhere in the message. An ISO **string** — which is
  what the tool actually sends — serialized cleanly, so the claim stored dated *now*,
  with the interval the caller asked for sitting beside it as an annotation nothing
  reads. That one is silent, and it is the one an agent hits: the likeliest caller here
  is a model that read the tool description and then wrote Python, which is this
  library's own primary user.

  Both are now a `TypeError` at the call, naming the key and the keyword it meant.
  `MCP_ALIASES` in `memvara/core.py` is the map, so a third spelling cannot be added on
  one surface without the other refusing it.

- **`remember()` rejects a `meta` value the store cannot persist.** `Claim.meta` is a
  JSON column, so anything `json.dumps` refuses never reaches disk — it raises inside
  `put_claim` with the key absent from the message. The check now runs at the boundary,
  where `RESERVED_META` is already checked, and names the key and its type. The docstring
  already argued for exactly this: *"a silently dropped argument is how a caller comes to
  believe something untrue about what they wrote."*

## [0.6.0] — 2026-08-25

### Added

- **`memory_remember` takes `sources`, so a fact written over MCP can say where it
  came from.** `memory_why` exists to put the excerpt in front of the user when they
  challenge a memory, and for every claim a hosted client had ever written it answered
  "No source turns are retained for this claim".

  Nothing in the library was missing — `Memvara.remember` has always taken `sources`
  and `_cite` has always accepted ids or `Episode`s. Two halves of the *transport* were
  absent, and each made the other useless: the tool did not declare the argument, so no
  caller could pass one; and `WriteReceipt.episode_ids` existed while `_receipt_summary`
  did not render it, so a caller that stored a turn could not learn what it had stored
  and had nothing to cite even if the argument had existed.

  **Ids, not text.** `_cite` stores anything handed to it as an `Episode` and merely
  links a string, so accepting turn text would duplicate a turn the caller has usually
  just stored through `memory_add` — which is exactly what the plugin does. The
  description says ids and a test pins that it says so, because a model reading
  "sources" and sending a sentence is the mistake worth preventing.

  Absence stays a degradation rather than a refusal: every client that exists writes
  without provenance today, and a tool that began rejecting those writes would break
  them all to fix a gap they do not know about.

- **`plugin-claude.md` — the instructions every plugin repository shares, held where they
  can be synced.** All seven repositories in `plugin-repos.txt` carry the same `CLAUDE.md`
  and nothing carried it between them: it was hand-copied, and it drifted, so a section
  written in one of them reached none of the others.

  A whole-file copy would have been a regression rather than a sync. Eleven of the fourteen
  sections were already byte-identical; the two that differ are about what each repository
  *ships* — its own runtime facts, and hook rules only one plugin needs — so copying
  wholesale would overwrite six repositories' correct, lighter sections. Those two sit in
  the middle of the document, so a head/tail split cannot express it without reordering a
  file seven repositories read.

  Hence one splice point, `@@LOCAL@@`. A sync replaces everything around it and preserves
  what is inside, and refuses to splice at all when the markers are missing rather than
  guessing — guessing there loses text no sync can put back.

### Fixed

- **`sqlite3.connect` reports a missing file and an unreadable one identically**, and
  does not expand `~` — so the documented `~/.mem0/history.db` import path failed in a
  way that read like a permissions problem. The re-raised error now says which of the
  two it is and names the absolute path it resolved to.

- **Windows CI: `os.path.expanduser` reads `USERPROFILE`, not `HOME`.** Monkeypatching
  `HOME` alone left `~` resolving to the real Windows profile directory instead of the
  test's `tmp_path`, so the tilde-expansion tests were asserting against whatever that
  machine happened to contain.

## [0.5.0] — 2026-08-25

### Added

- **`memory_standing` — the standing set, with no query and no ranking.** A client that
  wanted a user's standing preferences had to invent a sentence to rank them against, and
  ranking them is a category error: a `procedural` claim applies to every turn by
  definition, so the set is what is wanted and there is nothing to rank it against.

  It was measured rather than argued. `claude-memvara` asked with
  `recall("who is this user, how do they want work done, what are they working on",
  memory_types=["procedural"])`. A rule stored at confidence 1.00 — never put an AI
  attribution in a commit, a PR or an issue — scored **0.760** against a query about
  attribution and **did not place in the top eight** against that sentence, so it never
  reached a session. What did reach sessions was a hook's paraphrase of the same rule at
  **0.70**, which had turned "Claude name" into "user name" — and it ranked *because* it
  was wrong: "user name" matches "who is this **user**". Twenty-six of forty-five commits
  made after the rule was stored still broke it.

  Ordering is confidence, then recency, then claim id — so what the user stated outranks
  what a model inferred, and two claims written in the same instant cannot swap places
  between calls. It reuses `_delta_lines`' row format rather than inventing a second one:
  metadata first, untrusted text last, brackets neutralised by `safe_line`, and one client
  parser serving both tools. Read-only deployments keep it, because asking how the user
  works is a read.

  The tool count moves from twelve to thirteen in `README.md`, `docs/DEPLOY.md` and
  `memvara/server/__init__.py`. Downstream, `memvara-cloud` prices it as `since` — both
  enumerate the live claims in a scope, and this then filters to the procedural ones, so
  it does strictly less work than a bitemporal delta across two clocks.

- **`npx memvara` — the npm package is a program now.** `memvara` on npm was a name
  reservation from 2026-08-14 to `0.0.3`: four keys, `implemented: false`, nothing to
  run. `0.1.0` makes it a CLI that bridges a stdio MCP client to the hosted server at
  `app.memvara.dev/mcp`, so a JavaScript developer with **no Python at all** gets a
  working memory in one command.

  Asked why this had never been built, the answer was that nobody had decided against
  it — a JS client appeared in neither *Deliberately deferred* nor *What is still
  missing*, and the package's own README parked the question on "the number of people
  who ask", a signal nobody was collecting. `docs/ROADMAP.md` now records the decision
  that was actually made, including the two shapes declined: a JavaScript engine, which
  would have to re-derive every invariant in `INTERNALS.md` identically, and a REST
  client library, which serves a case nobody has asked for.

  **The value is narrower than it looks, and saying so is part of shipping it.**
  `app.memvara.dev/mcp` advertises standard MCP OAuth — dynamic registration, PKCE
  S256, refresh — so any client that speaks to remote servers connects directly and
  needs no bridge. This is for clients that only spawn a command over stdio. Its one
  advantage over the generic `mcp-remote` is that it needs no configuration and finds
  an existing `~/.memvara/credentials.json`, so anyone who has run `memvara-mcp login`
  is never shown a browser.

  Credentials resolve `MEMVARA_API_KEY` → `~/.memvara/credentials.json` →
  `~/.memvara/oauth.json`, and with none of them present it signs in with a browser.
  The login file is **read and never written**: `memvara/server/config.py` owns that
  schema and a token pair does not fit in it, so writing our shape there would break
  the Python server on the same machine, silently, at its next start.

  **Zero runtime dependencies**, asserted in CI rather than intended — this process
  holds a bearer token, and "we will review what we add" is a policy, not a control.
  Everything is Node stdlib, which is the same argument `memvara/server/` makes for
  being hand-rolled against the MCP wire format.

### Fixed

- **The npm trusted publisher named the workflow the publish job had just left.**
  Splitting npm onto `release-npm.yml` invalidated the registration on npmjs.com, which
  names exactly one workflow filename and compares it against the OIDC token's
  `job_workflow_ref`. Repository, environment and package were all still correct, and
  the publish still failed. `docs/RELEASING.md` said `release.yml` and now says
  `release-npm.yml`.

  **The error gives no hint what happened**, which is the part worth recording: npm
  answers `E404 Not Found - PUT https://registry.npmjs.org/memvara` for a package that
  plainly exists and that `npm view` returns — not `ENEEDAUTH`, not a 403. It declines to
  distinguish "no such package" from "not yours to write" so a probe cannot enumerate
  private names. Read a 404 on `PUT` as unauthorized. Nothing is published when it
  happens, so recovery is to fix the registration and re-run the failed job on the
  original run; a `workflow_dispatch` re-run will not publish, by design.

  A test now pins `RELEASING.md`'s stated filename to the workflow that actually
  contains `npm publish`, and was confirmed red against the old value before being kept.

### Changed

- **The tool count is stated where it can be guarded and dropped where it cannot.**
  Four surfaces still said "twelve tools" after `memory_standing` landed, and two of
  them were right: a present-tense claim about the tool surface gets the current
  number, and a past-tense record of an incident keeps the number that was true when
  it happened. So `_CLOUD_NOT_WIRED` — a live error message read at the moment a
  deployment refuses to start — and `build_memvara`'s docstring now say thirteen,
  while `docs/OPEN-CORE.md` and `tests/test_config_cloud.py`, which narrate the defect
  that motivated that refusal, keep theirs.

  The hosted pages now state no number at all. `references/hosted-mcp.md` said "the
  twelve tools appear" directly above a section headed "The thirteen tools", and had
  said "ten" before that — each edit a guess about somebody else's rollout, because
  how many tools a hosted deployment serves is a property of that deployment. There is
  no source of truth in this tree for what a remote server advertises, so a number
  there guards nothing. The library's own set stays pinned positively, by the test
  that reads `TOOLS`.

- **The skill says how to choose a memory type, and shows the call beside the wrong one.**
  The packaged skill did not contain the words `memory_type` or `procedural` once, across
  `SKILL.md` and ten reference files. The only guidance anywhere was a clause in the
  parameter schema — "for how the user wants work done" — and an agent recording "three
  traps when redeploying the box" reads that as how work is done and files it as a standing
  instruction.

  On a live store, **10 of 32 procedural claims were facts about repositories** rather than
  instructions from the person: 3,606 characters, a quarter of what every session opened
  with once `memory_standing` existed to return them. Every one of the ten had a container
  as its subject, so the test that catches all of them is that a claim whose subject is not
  the person is almost never `procedural`. That rule now lives in the parameter description,
  where the choice is made; `references/examples.md` gains the same discovery written twice
  differing only in the type, and one call per tool with the adjacent call to avoid.

  The split is also a cost decision: tool descriptions and schemas are paid on every
  `tools/list`, and the skill is read on demand.

- **npm releases on `npm-v*`, not on the Python tag.** The coupling cost a real
  release: `0.4.1` was a PyPI version containing no Python changes at all, cut only
  because npm serves the README of the *published* version and there is no way to edit
  one in place. One code-less release is a curiosity; a package with its own cadence
  would have produced one per fix. `release.yml` matches `v*`, which `npm-v0.1.0` does
  not, so the two trains cannot start each other — extracted into `release-npm.yml`
  rather than conditioned, so no job in either file carries a clause about the other's
  package.

- **Node runs in CI for the first time.** `ci.yml` was four Python jobs; the npm tests
  were Python assertions about files, spawned through the matrix and skipped whenever
  node was unloadable. That arrangement let a real bug reach a release. `npm-bridge`
  now runs `node --test` on 20, 22 and 24 — a matrix because this package's entire
  dependency story is the standard library, so the standard library is the dependency.

- **`tests/test_npm_release.py` pins the opposite of what it used to.** It enforced the
  reservation: four keys, no client surface, nothing to run. It now enforces the CLI —
  the bin exists and has a shebang, the tarball ships `bin/` and `lib/` and not `test/`,
  there are no dependencies, and no public document still calls the package a name
  reservation. That last check found three (`RELEASING`, `DEPLOY`, `SECURITY`) on its
  first run, which is the drift the 0.0.2 listing was corrected for a day earlier.

## [0.4.1] — 2026-08-25

### Changed

- **`memvara@0.0.3` on npm — the listing now describes the product, not just its
  absence.** 0.0.2 spent its whole description saying what the package is not, which
  is honest and is also the least useful thing a reader can be told. The npm page is
  rendered from the package's own README, and that README was forty-seven lines while
  PyPI's is the full project one — so a reader arriving from npm search met a
  disclaimer and a link, and a reader arriving from PyPI met the product.

  The README now carries what memvara is (two independent clocks, contradiction
  resolution without a model, hybrid retrieval, the graph), the twelve MCP tools and
  what each is for, predicate packs, and the honest limitations — with **both** routes
  a JavaScript reader can actually take: the hosted endpoint at
  `app.memvara.dev/mcp`, and `memvara-mcp` over stdio needing no account.

  The reservation notice moves to the top and stays blunt, because that is the risk a
  richer page creates: a reader who skims a product page and concludes that `npm
  install memvara` gives them something to call. The first line still says it exposes
  no API, and `implemented` is still `false`.

  `description` is a one-line summary on npm as on PyPI — it is what search results
  show, not the page body — so it leads with what memvara does and keeps the "not a
  JS client" caveat inside the length search will display. Keywords gain `mcp`,
  `model-context-protocol`, `agent-memory`, `ai-memory`, `knowledge-graph` and
  `temporal`, and keep `placeholder`, which is still true.

  No code changed. `index.js` is the same four keys and the same notice, and the tests
  pinning that shape are untouched. This release exists because npm serves the
  description and README of the **published version**: there is no way to edit either
  in place, and `publish-npm` only runs on a `v*` tag.

## [0.4.0] — 2026-08-25

### Added

- **`Memvara.reextract()` and `Memvara.pending_extraction()`: extract from turns that are
  already stored.** A turn reaching the store with no claim ever derived from it is
  ordinary in two ways, and until now neither had a way back. A deployment running with no
  model keeps every turn and extracts from almost none of them — 96 of 129 episodes on one
  measured store. And a provider failure mid-write sets `receipt.deferred`, keeps the
  episodes and returns, so the facts in that batch were lost while the text sat on disk.

  `reextract()` is `add()` minus tier 0: the episodes are stored and embedded already, so
  re-running it would find each as an exact repeat of itself. With no argument it sweeps
  `pending_extraction(limit=...)`, which makes a scheduled pass `mem.reextract(limit=20)`;
  given episodes or ids it does those, which is what retrying one known-failed batch wants.

  **A turn that already has claims is skipped, and that is the idempotency guarantee.**
  Re-reading stored text is not new evidence about anything, but the reconciler cannot
  tell that from a genuine repeat — an identical claim arriving twice reconciles to
  `reinforce` and bumps salience, measured. A sweep run twice would otherwise quietly
  promote what it had already extracted, which nothing would report. Counted on the new
  `WriteReceipt.already_extracted`.

  `pending_extraction()` applies the salience gate, because `add()` commits episodes
  *before* gating them: every "thanks" in the store has no claims either, and without this
  a sweep would pay a model call to rediscover that on every pass. The gate is
  deterministic and free, so it runs in the query.

  What no filter here can see is a turn a model read and produced nothing from — including
  one whose only claims `reject_ungrounded` refused, which that feature actively creates.
  Nothing on the episode records the attempt. So `reextract()` reports every turn it read
  on `receipt.episode_ids` and `pending_extraction(exclude=...)` takes them back: the
  durable set of what has been tried belongs to whatever is doing the scheduling. Found by
  running a sweep twice against a local 4B model and watching a rejected turn cost another
  250s of CPU to be rejected again.

- **`WritePipeline(reject_ungrounded=...)`, a grounding check on model-proposed claims,
  on by default as `"auto"`.** A claim whose object shares not one content word with the
  episode it cites as its source is treated as a fabrication candidate; under `"auto"`
  the pipeline's own embedder then gets a veto — the claim is kept if its best
  chunk-cosine against the source reaches 0.40, which is what a genuine paraphrase looks
  like and a wholesale invention does not — and only a claim failing both checks is
  refused. Refusals are counted on the new `WriteReceipt.ungrounded` and surfaced on the
  MCP transport the same way `unextracted` is, via a new note in `memory_add`'s receipt.
  `True` runs the lexical check alone; `False` turns the whole thing off.

  Built from a measured failure, not a hypothetical one: two 4B-class instruct models
  run over 20 real conversational episodes, under `extract()`'s real prompt and schema,
  both invented a placeholder `works_at: "Acme"` on turns containing no such fact at all
  — 36% and 18% of their respective usable outputs — rather than returning the empty
  list `EXTRACT_SYSTEM` explicitly permits. The rescue floor is measured too, on those
  33 fabrications plus 8 hand-built zero-vocabulary paraphrases under the MiniLM
  embedder: every wholesale invention scored 0.33 or below, the paraphrases the rescue
  exists for scored 0.45 and up, and 0.40 sits inside the separating region. Under the
  default `HashingEmbedder` the same pairs score 0.0–0.11, so nothing is ever rescued
  and `"auto"` degrades to the strict lexical check — gracefully, since character
  n-grams have nothing to say about meaning. Substring-matched, not exact-token matched,
  because this store's own content is full of paths and hyphenated identifiers where
  exact-token matching produced false positives in testing.

  Defaulting on is a considered decision, argued from three facts. The check only ever
  runs on claims a model proposed — `remember()` and the deterministic fast path never
  reach it, so nothing a caller asserts is filtered. The destructive direction is
  storing, not rejecting: a fabricated value in a ONE-cardinality slot supersedes and
  *ends* the true fact that was there (`works_at: "Acme"` retires the user's real
  employer — there is a test that measures exactly this with the filter off). And the
  residual false-positive class — a genuine paraphrase with no shared vocabulary that
  the embedder also cannot connect — was observed zero times in 144 real claims and
  costs one claim from one turn, never anything already stored. An embedder that fails
  during the rescue fails open: the claim is kept and the pipeline warns once.

- **Two counters for the memories a write actually landed**, `write.memory_claims` and
  `write.memory_episodes`. Between them they answer "what did this store gain", which no
  existing series does.

  `write.memory_claims` is `len(WriteReceipt.added)` — the `add` and `supersede`
  reconciliation outcomes, emitted from both `add()` and `assert_claim()`. `reinforce`,
  `retract` and `noop` create no row and move nothing, which makes "writing the same thing
  twice is free" a property of the meter rather than a claim about it: two differently
  worded sentences carrying one fact reconcile to `reinforce`, and neither is counted.

  `write.memory_episodes` counts episodes that reached `store.add_episode`, which is not
  what `write.turns` counts. `write.turns` is the whole input batch, and exact repeats are
  dropped without being stored. It exists because an episode is retrievable on its own
  through `include_episodes`: on a deployment with no extraction model, prose matching no
  rule is stored and answers queries while producing no claim, so a meter counting claims
  alone would report almost nothing for most of that deployment's traffic.

  Both are deliberately separate from `write.claims`, which counts one per `assert_claim`
  *call* whatever the call displaced, and which says in its own docstring that it is not a
  billing series. These two are.

- **npm on the same release workflow as PyPI.** A `v*` tag now runs `check-npm`
  (does this `package.json` version already exist?), `build-npm` (pack once, hash
  the tarball), and `publish-npm` (download those bytes, verify the hash, upload
  the tarball). Unlike `publish-pypi`, no reviewer wait: the `npm` environment
  exists so the trusted publisher can name it, not as an approval gate, and the
  tag push is the publish. npm versions stay independent of the Python tag: the
  job compares `npm/memvara/package.json` against the registry and skips a number
  that is already there, so a tag that does not touch the package is green and
  publishes nothing. There is no JavaScript client; `npm/memvara` remains a
  four-field notice object.

- **`memvara@0.0.2` on npm — the reservation, saying something useful.** `0.0.1`
  told a JavaScript reader the library is Python and stopped, which reads as *come
  back later*. That is not the situation: memvara ships an MCP server, MCP is the
  interface a JavaScript agent already speaks, and a JS binding would sit between
  two things that are already connected. The notice and the README now say so, and
  name both routes — `memvara-mcp` over stdio against a local file, needing no
  account, and the hosted endpoint at `app.memvara.dev/mcp`, needing one. The
  README had the hosted half and not the local half, which is the half a reader
  can use in the next minute.

  Still four keys, still `implemented: false`, still no runtime surface — the
  package's promise is unchanged and the tests that pin it are unchanged. This is
  the first version to go up through `release.yml` rather than a local script.

### Fixed

- **`publish-npm` handed npm a git repository instead of a tarball.** `npm publish
  npm-dist/memvara-0.0.2.tgz` does not publish that file: to npm, a spec containing a
  slash and no leading `./` is `owner/repo`, so it ran `git ls-remote
  ssh://git@github.com/npm-dist/memvara-0.0.2.tgz.git` and died on `Permission denied
  (publickey)` — an authentication error naming a repository nobody meant to reach,
  for what was punctuation. It cost the first `v0.4.0` tag; nothing reached either
  index, because the failure is before the upload.

  The step now runs in `npm-dist` and publishes a bare filename, matching the hash
  check immediately above it, and refuses outright if the spec ever grows a slash.

  What let it through is the more useful half. `release/rehearse_npm.py` exists to be
  the integration proof for exactly this job, and it passed — because it handed npm
  `Path` objects, which are absolute, and npm reads an absolute path as a file. The
  rehearsal and the workflow were never running the same test. The rehearsal now
  publishes a bare name from the tarball's own directory, and a new test in
  `tests/test_npm_release.py` follows the workflow's shell variable back to its
  assignment and fails if what it yields is a slashed spec. Both were confirmed to
  fail against the old workflow before being kept.

- **`include_episodes` on `memory_recall` had never worked, and the wrong value worked
  better than the right one.** `validate.py` handled `string`, `integer`, `number` and
  `array`, and `tools.py` declared `include_episodes` as `boolean` — a type the validator
  did not know. It has been the only boolean in the twelve-tool surface since it shipped,
  so nothing else was affected and nothing surfaced it.

  Both halves are worth stating, because the second is the dangerous one. Sending the
  argument as the schema asks — `true` or `false` — reached the type-mismatch branch,
  which looked its article up in a table with no `boolean` key and raised
  `KeyError: 'boolean'` *from inside the error path*: an unhandled exception rather than a
  tool error, for a correctly-typed argument. Sending it as a **string** was accepted
  instead, because a boolean fell through to the `not isinstance(value, str)` check — and
  handlers read flags through `bool(...)`, where every non-empty string is truthy. So
  `"false"` turned episodes on.

  The validator now knows `boolean` and accepts only a real one: not `1`, and not the
  string `"false"`. A new test asserts that every type any tool declares is a type the
  validator handles, which is the link that did not exist — the schema was covered (one
  test already asserted `include_episodes` was listed as accepted) while the code path
  behind it had never run.

- **The release workflow could not collect the test suite.** Its `build` job installed
  `.[dev]` where `ci.yml` installs `.[dev,cloud]`, and `httpx` is in the `cloud` extra —
  so `tests/test_store_remote.py` failed at import and the suite ended with two collection
  errors before running a test. `publish-pypi` is gated on that job, so the release
  stopped exactly where it should have.

  Latent since the workflow was written: `0.2.0` was published by hand before the file
  existed, so the `v0.3.0` tag is the first time any of it ran. The cause is the one
  `release.yml` already avoids for the matrix — it *calls* `ci.yml` rather than restating
  it, "so there is one matrix in this repository and it cannot drift" — and then restated
  the install list a few lines further down.

- **The docs still described a world in which nothing had been published.**
  `docs/RELEASING.md`, `docs/DEPLOY.md`, `SECURITY.md`, `docs/ROADMAP.md` and
  `release/README.md` said the PyPI and npm names were free, or that
  `pip install memvara` was not the install. Both names were claimed on
  2026-08-14. A reader acting on those sentences would have treated a live
  package as takeable, or cloned a tree to get a library that has been on
  the index for days.

## [0.3.0] — 2026-08-23

### Fixed

- **Two identical searches scored differently, and the read path now takes the instant it
  is evaluated at.** `HybridRetriever.search()` and `GraphTraverser.spread()` gain
  `now=`, the parameter `Consolidator.run()` already had — this was the same defect on
  the read side and it went unnoticed longer.

  A search decays every claim from the moment it is asked, and with no instant named that
  moment is `utcnow()`. So two identical searches seconds apart return the same rows with
  different scores. Measured on 2WikiMultihopQA: **3,000 of 3,000 questions differ between
  two back-to-back passes**, in the low-order digits, ids and ranks unchanged. It matters
  twice over — a claim sitting near `min_score` is returned on one turn and dropped on the
  next with no write in between, and a benchmark re-run diverges from its own published
  table with no code change behind it.

  It was *two* clock reads, not one: the retriever's, feeding `recency_factor` in the
  quality multiplier, and the traverser's own inside `_pin`, feeding `_strength`. One
  search decayed its quality and its edge weights from two instants microseconds apart.
  `now` is threaded from the first to the second, so a search is now evaluated at one
  moment throughout.

  **`now` replaces the clock read and neither time axis.** A caller who named `known_at`
  still decays from `known_at`; time travel is unchanged and pinned by test.

  **`bench/twowiki.py` pins both ends, and its published table moves.** Pinning the read
  alone would have been worse than not pinning it: `remember()` stamps a claim with the
  wall clock, so reading at a distant instant makes every claim years old and collapses
  `recency_factor` to zero for all of them. Measured — reading at 2030 against claims
  written today took ungated `chained` from 76.7% to 54.5%, which is a benchmark of
  something else. Both ends are pinned instead, one minute apart, which is the regime an
  unpinned run was already in.

  That surfaced a second defect and is why the numbers moved rather than merely settling.
  With ingest unpinned, **1,239 claims carried 1,239 distinct `valid_from` values** — one
  per write, microseconds apart — so `recency_factor` gave a strict ordering that encoded
  *ingest order*. This is the defect fixed once already for score ties, which used to
  break on `claim.id`, a fresh `uuid4` per ingest; it survived one layer up, in the score
  itself rather than in the tie-break. Pinning collapses those into real ties, which the
  content hash resolves as intended.

  The re-baselined figures are within 1.6 points of the old ones everywhere. Two runs of
  the harness are now byte-identical, which three were not before.

  **Figures elsewhere in this release follow one rule.** A number recording what *that*
  change moved — `chained 35.4% → 41.4%` for the content-token match, `29.1% → 35.4%` for
  the observed-predicate rule — is left as it was measured, because it is the record of
  that measurement and re-running the intermediate configurations to restate it would be
  archaeology. A number stating what the leg is worth *now* tracks the table and has been
  updated. The two differ by at most 1.6 points, and `docs/BENCHMARKS.md` is the one to
  quote.

  **Only this harness needed it.** `bench/longmemeval.py` run twice differs in nothing but
  its two timing lines: its claims carry transcript timestamps years old, so they sit on
  the flat part of the decay curve where seconds of wall clock change nothing.

- **`erase()` left the erased text readable in the database file.** It deleted the row,
  the index entry and the vector, returned per-table counts as evidence, and the words
  were still there — findable with `grep`, with no `VACUUM` needed to expose them and no
  `VACUUM` run anywhere in normal operation. In a 3,000-claim store an erased term
  survived 2,600 subsequent writes. `purge()` and `reset()` leaked the same way.

  Two independent causes, and the second is the one that makes this worth a release of its
  own:

  - **Ordinary rows.** SQLite frees a deleted row's space and leaves the bytes. This half
    was known and disclosed. `PRAGMA secure_delete=ON` now overwrites them.
  - **The text indexes.** `DELETE FROM claims_fts` does not remove the document's terms.
    FTS5 writes a *delete marker* and keeps the terms as **live rows** in the
    `claims_fts_data` shadow table — not residue in a freed page, but current content of a
    current table, which is why `VACUUM` never touched them. FTS5's own `secure-delete`
    option fixes it, is persistent once set, and is **not** retroactive; schema 7 therefore
    also runs a one-time `optimize` to clear what is already on disk.

  **Two documents said the opposite of what the code did**, and both are corrected here:
  `SECURITY.md` listed the residue as out of scope while asserting that "erasure deletes
  rows and index entries", and named `VACUUM` as the deployment's lever — which cannot
  reach a shadow table. `docs/DEPLOY.md` said erasure covered "the FTS entry (which stores
  the tokens directly)". An operator following either would have believed the text was
  gone.

  Costs, measured: +6% on a 5,000-claim write run, +9% on `erase_claim`, and 0.01 s for
  the one-time `optimize` over a 20,000-claim index. The FTS5 option needs SQLite 3.35,
  which is already `_MIN_SQLITE` — the store has refused to open on anything older since
  before this change, so no build that can read the file lacks the option.

  **What is still not scrubbed is the write-ahead log.** An erased claim's bytes can
  remain in `-wal` until it checkpoints; a checkpoint or a clean close clears it.
  `SECURITY.md` now says that instead of the sentence it used to carry.

  SQLite schema **6 → 7**. The stamp moves because the setting is durable state in the
  file, and an older build must not open it and write to a text index whose format it does
  not understand.

- **A published guardrail figure was stale, and the correction argues harder for the
  default than the original claim did.** `docs/BENCHMARKS.md` said that turning the graph
  leg on moved no `R@k` in either public run, and named single-session-user 92.2 → 92.2 as
  one of the two rows the thesis rests on. Re-measured on this tree it is **92.2 → 90.6**,
  with the overall figure 70.4 → 70.1 and no category gaining.

  No code changed to cause it and no behaviour is wrong. The gate work earlier in this
  release is what moved it: a query that named an instant used to force `Intent.TEMPORAL`,
  whose multipliers zero the graph weight, so on LongMemEval the leg mostly did not run.
  Once it could run, and once the classifier could read vocabulary off retrieved rows, it
  began firing on a store that holds 78 claims across 940 sessions — where a walk reaches
  almost nothing and still votes, and fusion reads positions. Nothing caught it because no
  test asserts a benchmark figure and those commits edited a different file.

  `w_graph` still ships at **0.0**, now on evidence rather than on an absence of it, and
  `docs/ROADMAP.md`'s summary of the leg is corrected to match: what the leg is worth
  depends on how much graph the store holds, and the public corpora disagree because they
  hold 26,403 claims and 78.

### Added

- **`write.claims` counts the writes that skip extraction**, which nothing counted before.
  Emitted once per call from `WritePipeline.assert_claim`, so it covers `remember()`,
  `supersede()`, `assert_claim()` and the importer — every path that asserts a fact
  directly.

  `write.turns` counts turns handed to `WritePipeline.add` and nothing else, which is
  correct for what it measures and leaves a hole for anyone reading it as write activity.
  **On a deployment configured with no extraction model the hole is the whole picture**:
  prose sent to `add()` matches only a fixed set of sentence forms and is otherwise stored
  as nothing, so the direct write is the only reliable path and `write.turns` sits flat
  while the store fills up. Measured on a hosted deployment on 2026-08-23: 26 `remember()`
  calls took a store from 120 claims to 145 with `write.turns` unchanged at 88.

  The two series are not interchangeable and this is deliberately a second series rather
  than a wider definition of the first. `write.turns` answers "how much conversation was
  ingested" and is what a turn allowance is spent against; `write.claims` answers "how
  many facts were asserted" and spends no allowance. Summing them would bill for writes
  the API documents as free.

- **A store where nothing chains no longer runs the graph leg**, whatever `w_graph` is set
  to and whatever the query says. Closes memvara/memvara#42.

  A walk needs somewhere to go. Where no live claim's object is another live claim's
  subject there is nowhere, and the leg degenerates into returning other facts about
  whichever hub the seeds hang off — ranked by a path score that is near-uniform when
  every path is one hop. Fusion reads *positions*, so that is a fabricated ranking. It is
  the failure `MIN_PROXIMITY` prevents for the temporal leg, and the graph leg had no
  equivalent.

  Measured on LongMemEval, whose 78 claims all take `user` as their subject. With
  `w_graph=1.0`:

  | R@12 | baseline | before | after |
  |---|---:|---:|---:|
  | all | 70.4 | 70.1 | **70.4** |
  | single-session-user | 92.2 | 90.6 | **92.2** |
  | multi-session | 65.5 | 65.0 | **65.5** |

  Every category exactly baseline, so **turning the graph leg on is now free where it
  cannot help.** On 2WikiMultihopQA the gate closed the leg on 0 of 3,000 searches and
  every returned row is unchanged, which is the other half of the claim.

  Four decisions worth reading before changing it:

  - **It sits after the intent weighting, not inside it.** The second-chance rule added
    in 0.3.0 can only *widen* — its guard is `weights.graph <= 0.0` — so it cannot undo a
    walk `classify` opened. Measured: a veto wired into that hook returned False on all
    802 of LongMemEval's gate calls and the run still lost 1.6 points.
  - **`{}` from a backend is not zero.** A store without `connectivity`, or a hosted
    facade too old to report the counts, keeps its graph leg. Reading "did not look" as
    "no joins" would switch the leg off on every third-party store at once.
  - **No threshold.** The condition is `joinable_claims == 0`, a structural fact about the
    data. A tuned floor picked from the two corpora available — 0.0% and 40.6% — would be
    a constant fitted to two points.
  - **Cached per tenant, re-measured every 256 searches**, on a counter rather than a
    clock so the same sequence of searches re-measures at the same points on every run.
    The staleness only ever runs the leg *less*, which degrades to the shipped default.

  New `UnjoinedStoreWarning`, a subclass of `DegradedRetrievalWarning` so existing filters
  still catch it, raised once per retriever. It says the *data* has no chains, which is
  fixed in the write path — distinct from the parent's "this backend cannot traverse",
  which is fixed by changing store.

  **`w_graph` still defaults to `0.0`.** What this changes is that turning it on is now
  safe; whether it should be on by default needs a corpus between 0.0% and 40.6% join
  rate, and no public one exists.

- **`memory_stats` reports a join rate**, and it is the number that says whether the
  graph leg can do anything for you. `joinable_claims / live_claims` — the share of
  stored facts whose object is the subject of another fact — measured over one read
  snapshot. Both sides of the join are held to the rules `GraphTraverser._edges` uses,
  liveness included: no negations, no empty ends, no self-loops, nothing retired. A rate
  built from edges the walk would refuse promises hops that will not happen.

  It exists because "is this store a graph" is not a question worth asking: every claim
  carries `subject_key`, `predicate` and `object_key`, so the answer is always yes and it
  predicts nothing. Connectivity is what varies, and it varies enormously. Same retrieval
  code on the two public corpora: 2WikiMultihopQA joins at **40.6%** and `read_w_graph=1.0`
  takes chained questions from 28.3% to 42.1%; LongMemEval joins at **0.0%** — one
  subject, 78 leaf objects, not one two-hop path in the store — and the same setting
  *loses* 1.6 points. `docs/BENCHMARKS.md` has both.

  A rate near zero is usually a **star** and usually correct. Facts extracted from a
  user's own sentences all have that user as their subject, so their objects are leaves
  and no two of them connect. Raising it is a write-path question — store facts whose
  subject is not the user — not a retrieval one.

  Three details that are decisions rather than defaults:

  - **A backend that cannot measure it prints no line at all**, rather than 0.0%.
    `Memvara.connectivity()` returns `{}` for such a store and
    `{"live_claims": 0, "joinable_claims": 0}` for an empty one. A measured star is a
    finding an operator should act on; an unmeasured one would send them to the write
    path over a deployment lag. `RemoteStore` returns `{}` against a facade that does not
    yet report the counts.
  - **It is not folded into `stats()`.** `Memvara.__repr__` calls `stats()`, and the join
    is a semi-join over the whole claim table: about 60 ms on 26,403 claims against that
    call's 69 ms.
  - **`IN (SELECT ...)`, not `EXISTS (SELECT ...)`**, because `tenant` is optional and
    only `IN` is fast for both calls. Median on that store: `IN` 39 ms filtered and 34 ms
    unfiltered; `EXISTS` **27 ms** filtered and **42,302 ms** unfiltered. `cl_subj` leads
    on `tenant`, and a correlated subquery that cannot bind it gets no index at all.

  New optional `Store` member `connectivity`, listed in `OMITTABLE`; `connectivity()` on
  `Memvara`, `ScopedMemvara`, `AsyncMemvara` and `AsyncScopedMemvara`. Leaving it out
  costs the `memory_stats` line and nothing else.

- **`now` on `Consolidator.run()`**, which evaluates the whole pass at one instant the way
  `decay()` already could. `Sweep` reads the wall clock once per pass, and the decay target
  depends on that instant as well as on stored state, so two back-to-back passes land
  microseconds apart. `decay_pass` compares salience already rounded to six decimals, which
  hides that gap until a claim sits within one pass-duration of a rounding boundary; the
  second pass then crosses the boundary and rewrites the row. Measured on a FAST predicate,
  that is 0.045% of claims per millisecond of gap, and the gap is the first pass's own
  duration. It made `test_run_twice_leaves_identical_state` fail once on a CI runner and
  never in 55 local runs.

  The comparison in `decay_pass` is unchanged. Exact equality on the rounded value is what
  makes a skipped or doubled pass a no-op, so widening it would hide drift rather than
  prevent it.

- **A temporal leg over raw turns**, off by default. Bitemporality is what this library is
  for and time appeared on the read path twice, both times too late to matter: as a
  *filter*, which narrows what the store returns and so cannot add a candidate no other
  leg found, and as a *multiplier*, which reorders the fused list and so cannot add to it.
  Neither answers "what was going on around then" — a question whose only content words
  are `when`, `around` and `then`, every one of which the analyzer drops as a stopword.

  `Store.episodes_near` is one new optional method returning the turns closest to an
  anchor, nearest first; `memvara/retrieve/temporal.py` turns their timestamps into an
  absolute [0, 1] closeness. `HybridRetriever` gains `w_temporal`, `Explanation` gains
  `temporal_rank` and `temporal_score`, and `intent.MULTIPLIERS` gains a `temporal`
  column — where it is exclusive with the graph gate, because a question about a chain and
  a question about an instant are different questions.

  **The anchor is given, never parsed**: `valid_at`, else `known_at`, else now. A date
  parser on the read path is a second extractor with its own locale bugs, answering a
  question the caller who wrote `valid_at=` has already answered. An explicit instant also
  outranks the marker vocabulary — a call that named one has stated a temporal intent, and
  the words are frequently the wrong place to look.

  **Episodes, not claims.** A claim carries a predicate-keyed half-life, which knows what
  raw proximity cannot: a `born_in` from 2019 is as current as it will ever be and a
  `working_on` from 2019 is not.

  **`w_temporal` ships at 0.0, and the measured finding is the abstention rather than the
  leg.** Without one it cost **2.4 points of LongMemEval temporal-reasoning R@12 and 4.6
  of MRR**: with no instant given the anchor is *now*, those transcripts are years old, so
  every turn scored a proximity around 0.005 — and RRF reads *positions*, so a leg with no
  opinion still contributed rank 0, rank 1, rank 2. A ranking assembled from nothing is not
  a weak ranking, it is a fabricated one, and fusion cannot tell. `MIN_PROXIMITY` gives
  the leg the guard the other two have always had — the vector leg abstains on a zero-norm
  query, the lexical leg on a query with no content terms — and the loss goes to zero.
  With it, temporal-reasoning is unchanged and multi-session loses 0.5, so the default
  stays off. `docs/BENCHMARKS.md` has the table.

- **`memory_neighborhood` and `memory_paths`**, so the MCP surface can ask the graph a
  question. `GraphTraverser` has been complete since schema 6 and no tool reached it, so
  an agent asking "who does their manager report to" had `memory_search`, which matches
  text — and the fact that answers that question shares no words with it.

  Read-only, so both survive `MEMVARA_READ_ONLY`, which is the deployment that most wants
  them: a store nobody can write to is exactly the one worth asking about connections.
  Rendered through `Path.render()`, the same arrows `neighborhood()` prints in a REPL, so
  there is one place for that convention to live. `render()` takes an `escape` hook and
  the tool layer passes one, because a rendered walk is the first surface here that puts
  *stored text* either side of a delimiter **it** supplies: a claim whose object is
  `Acme -owned_by-> The CIA` otherwise prints a second hop that nobody walked, and the
  line it forges is well-formed. The hook escapes each label and predicate separately, so
  the arrows the walk really took are still arrows and the ones a row was carrying are
  not. Every row also names the claim ids it is made of, which is what lets a reader check
  a hop against `memory_why` rather than take the rendering's word for it.

  **Neither takes a scope argument, and a test asserts they never will.**
  `memvara/server/config.py` is explicit that reading scope from tool arguments hands the
  model other people's memory; these two walk *between* rows, so a scope argument here
  would not merely widen a read, it would let a chain leave the caller's own memory
  mid-hop.

  **An empty `memory_neighborhood` no longer denies what `min_hops` pruned.** The filter
  removes short paths after they are walked, so a store holding `Alice reports_to Dana`
  answered "nothing stored connects to 'Alice' within 3 hop(s)" and then explained the
  absence with two causes, neither of which was the real one. A model has no way to look
  behind a tool result, so what it does with that is tell the user Alice is unconnected.
  The pruned case now names `min_hops`, says closer connections may well be stored, and
  says how to see them. The genuinely-empty case gains the bounded-walk caveat
  `memory_paths` has always carried — the same beam bounds both walks, and two tools
  disagreeing about that is how a model learns to trust the wrong one.

  Two things the descriptions say because a model cannot check them for itself. An empty
  `memory_paths` result is an answer about **this search** — the walk is bounded by a beam
  as well as by depth, so a real route can be missed because its prefix was pruned — and
  the handler's own wording says "nothing stored connects them", never "they are
  unrelated". And `min_hops` is a correctness knob rather than a tuning one: a path's
  strength never rises with length, so every one-hop connection outranks every two-hop
  one and a crowded first hop spends the whole of `k`. Measured on questions whose answer
  is exactly two hops away, at `k=5`: **5.3% at the default against 41.0% with
  `min_hops=2`**.

  `memvara/skills/memvara/SKILL.md` gets the complementary half, since it states outright
  that it does not repeat what a tool description says: when a connection question beats a
  recall question, and to ask one *after* a thin recall rather than instead of one.

- **Ties are broken on content, so two stores holding the same data answer the same.**
  `lexical_search` ordered by BM25 alone. Ties are not exotic there — eight claims
  differing only in subject score identically for a query on the object — and with
  nothing after the score SQLite returned rowid order. The `LIMIT` is inside the same
  statement, so that decided *which rows came back at all*: two stores holding the same
  eight facts, filled in opposite orders, returned disjoint top-3s. Now
  `s, value_key, id`, and `s, hash, id` for the episode half. `episodes_near` had the
  same hole one level down — it broke ties on `id`, which is minted at ingest, while its
  docstring said equidistant turns "come back in the same order on every file". Now the
  content hash first, and the docstring says what the code does.

- **`Explanation`'s new fields are appended rather than slotted in beside their kin.**
  `graph_*` and `temporal_*` read best next to `vector_*` and `lexical_*`; putting them
  there shifted `fusion_score` and everything after it four positions right, so
  `Explanation(0, 0.9, 1, 0.8, 0.5)` written against 0.2.x put the fusion score into
  `graph_rank` and left `fusion_score` at its default — no exception, an `int | None`
  field holding 0.5, and a ranking explanation quietly reporting the wrong number about
  itself. Pickle is unaffected either way: `slots=True` keys state by name, so an older
  pickle restores its own fields and leaves the new ones unset.

- **`Store.OMITTABLE` names the five members a backend may leave out**, and what each
  costs. `Store` is `@runtime_checkable` and `isinstance` on a Protocol is all-or-nothing,
  so it cannot answer "can this store walk a graph" — which is why the capability check
  here is `getattr` per member, and why that list needed writing down.

  **`store.base.bulk_claims()` is the new one place claims are hydrated by id.**
  `get_claims` was added after the first third-party backends existed, and the suite
  pins that those keep working — but only `search()` actually fell back. `Memvara.get_all`
  and `produced` called straight through, so a store predating `get_claims` searched
  perfectly well and raised `TypeError: 'NoneType' object is not callable` the moment
  anybody listed their memories. A compatibility guarantee honoured at one of three call
  sites is not a guarantee, so now there is one call site.

- **Time-shifted walks say which clock they were answered at.** `memory_search` has
  echoed it since time travel existed; `memory_neighborhood` and `memory_paths` took the
  same axes and said nothing, so a walk of the graph as it stood in 2019 came back
  looking exactly like a walk of it as it stands now. All three share `_when()` now, and
  the two axes stay worded apart: `as_of` is what was *believed* then, `valid_at` what
  was *true* then judged by what is known today.

- **An unpaired surrogate is rejected by name.** JSON accepts `"\ud800"` and Python hands
  back a `str` that cannot be encoded, so it arrived looking like any other string and
  failed at whatever line first tried to write it — reaching the model as "failed:
  UnicodeEncodeError: 'utf-8' codec can't encode character". Now every string argument is
  checked, and the error names the argument and the position.

- **The length error stops recommending an argument the tool does not have.** Five of the
  six tools carrying a `maxLength` have no `object`; "put the detail in 'object'" earned
  their callers a second rejection for an unknown argument.

- **Seasons and quarters are time words.** `march` was in the classifier's marker set and
  `spring` was not, so "what did I do in March" routed as temporal and "what did I do in
  spring" routed as a lookup — answered with the leg that ranks on *when* switched off.
  Added `spring summer autumn fall winter season(s) quarter(s) q1-q4 h1 h2`. `between`
  was checked and is fine: `TEMPORAL` is tested before `RELATIONAL`, so "between 2019 and
  2021" and "between March and June" already routed on time.

- **`memvara-mcp init` no longer writes a config the server refuses to start.** With
  `httpx` importable — which is a great many environments that never installed the cloud
  extra — `init` defaulted to cloud mode, wrote `MEMVARA_MODE: cloud`, printed "restart
  your client" and exited 0. `build_memvara` then refused, because the REST facade has no
  endpoint for the surface the engine calls on every turn. Two commands answering the
  same question differently, and the gap between them was silent by construction: the one
  that writes the config never starts the thing it configured, so nothing in the
  successful run could notice. What reached the user was a client with no memvara tools
  in it and nothing anywhere connecting that to anything they had done.

  Cloud named explicitly, by `--mode` or by `MEMVARA_MODE`, is now refused with the
  server's own text, so the reason arrives while there is still something to do about it.
  The httpx heuristic asks whether cloud works before preferring it. Both callers derive
  the answer from `RemoteStore.WIRED` via the new `config.cloud_gap()`, so the day the
  endpoints land this lifts itself rather than becoming a flag somebody has to remember.

- **`Memvara.prove_erased()`, and `erase()` refusing to report a success it cannot
  support.** `erase()` returned `True` when `Store.erase_claim` said it had deleted a row.
  That proves the code took the branch it thought it took — which is the statement the
  return value already made and cannot disagree with — and "told the caller the memory was
  deleted while the text is still on disk" is the exact failure the method was added to
  remove. It was left open at the last step.

  `Store.residue(claim_id)` is a **physical re-query**: four `SELECT COUNT(*)`s over the
  tables a claim's content can survive in — the row, the text index, the vector, the
  provenance edges. `prove_erased` returns an `ErasureProof` built from it, `erase()` calls
  it after the delete, and raises `ErasureIncomplete` if anything survived.

  It **fails closed**, and against the shape of the answer rather than against a list of
  known failures. A store with no `residue`; one whose `residue` raises anything at all —
  `RemoteStore`, the shape a `getattr` guard cannot see; one that returns something that
  is not a mapping; and one that returns an *empty* mapping all yield `proven=False` with
  a reason naming what happened. The empty case is the one worth stating outright:
  `all(n == 0 for n in {})` is `True`, so a store that counted nothing would otherwise
  have produced the strongest possible certificate from the weakest possible evidence.
  Unproven and proven-gone are different answers and only one of them is an erasure
  certificate.

  **What it cannot check, said plainly here because the docstring says it too.** A store
  that returns well-formed zero counts for the *wrong* tables gets a passing proof. The
  protocol lets each store name the tables its own schema keeps content in — that is what
  makes `residue()` implementable by a backend this repository has never seen — and
  nothing generic can tell a short key set from an honest one. So the trust boundary is
  the store, and it is pinned where it can be: a test asserts the shipped SQLite store
  counts all four of its tables, and `Store.residue`'s docstring tells an implementer that
  a forgotten table is a passing proof rather than a missing one. This is a behaviour change on
  cloud mode: `erase()` there now raises rather than returning `True` for an erasure it
  could not verify.

- **SQLite schema 8: an `erasures` audit table.** One row per `erase_claim`, and two
  things about it are the whole design.

  It is written **before** the delete and in the same transaction, so a failed audit write
  aborts the delete: a claim cannot be gone without a record of it. The other order lets a
  delete succeed and its record fail, which is precisely the state nothing downstream can
  detect. `tests/test_erasure.py` asserts it by dropping the table and watching the claim
  survive.

  That ordering opens the mirror-image hole, and it is closed by hand: if the *delete*
  then fails, the audit row is a record of an erasure that never happened, which is a
  worse lie than no record at all — it reads as proof to exactly the audit that would
  otherwise catch the survival. `erase_claim` compensates by deleting its own row before
  re-raising. Deliberately **not** a `SAVEPOINT`: `RELEASE` commits the savepoint's work
  into the enclosing transaction, so an erasure inside an abandoned `batch()` stopped
  rolling back with it. The suite caught that; the comment in the code says so, because
  the savepoint is the version that looks tidier.

  Keyed on `(claim_id, erased_at)` rather than on the claim, so the trail is append-only.
  An id can be erased, restored from a backup and erased again — two events, and a table
  keyed on the id alone would let the second overwrite the record of the first.

  It holds **nothing the erased fact could be read back out of** — `claim_id`, `tenant`,
  `scope`, `erased_at`, how many source turns were cited, and the per-table counts. No
  text, subject, predicate or object. An audit trail you can read the erased memory out of
  is a copy of it wearing a different name, and would make `erase()` a rename.

  **Ordering and durability, not tamper-evidence.** Nothing here is chained or signed; an
  operator with write access can remove a row. The hash-chained log is a different feature
  and is commercial. What this defends against is a delete no record was ever written for.

- **`retrieve/compose.py`: relation terms that name a chain rather than a predicate.**
  "Who is the maternal grandfather of X" is a two-hop question over `mother` and `father`
  that names neither. Every rule in `intent.py` counts predicates a question says out
  loud, so this shape was invisible to all of them — on 2WikiMultihopQA's `inference`
  family the walk ran on none of 1,549 questions.

  `compose.acquire()` asks a model **once, about a vocabulary**: given the predicates a
  store uses, which English relation terms compose from two or more of them. The read
  path does a set-membership test against the answer. `intent.py` promises to be
  model-free and `hybrid.py` promises reproducible retrieval, and a search that could
  block on an API call breaks both — so this is shaped like `resolve_predicate`, paid once
  per vocabulary rather than once per query.

  Measured on all 1,549 `inference` questions at k=12, with terms acquired from a live
  model against the store's own vocabulary: **49.0% → 80.3%** answer and
  **45.1% → 78.6%** chain. A minimal four-word list any model would produce is worth
  73.9% / 71.8% on its own, so the feature needs a plausible list rather than a good one.
  `compositional`, `comparison` and `bridge_comparison` are unchanged. A disjunction is
  still a comparison when it names a derived relation, so `is_comparison` runs first.

  `HybridRetriever(derived_terms=...)` is empty by default and a backend without
  `compose_relations` yields nothing, so a store with no model behaves exactly as it did.
  `RelationComposer` is its own Protocol rather than a third method on `LLM`: adding a
  member to a `runtime_checkable` protocol makes every implementation that predates it
  fail `isinstance`, which is how #26 broke a downstream type check.

  The model's answer is filtered rather than trusted — arity below two, a term that is
  itself a predicate, a phrase longer than three tokens, a non-integer arity — and a
  backend that raises costs nothing, because an enrichment that raised into
  `Memvara.__init__` would make an optional feature a startup dependency.

  **`_shape.shape_composition` reads a bare term-to-arity map as well as the wrapped
  one.** `COMPOSE_SCHEMA` declares `{"derived": {...}}` and a backend whose structured
  output enforces the schema sends that; one that is only *asked* for it answers with the
  bare map. The parser required the wrapper and returned `{}` silently, so a live model's
  21 correct terms all disappeared and the acquisition looked like a model with no
  opinion. The unit tests could not have caught it — the fake client returns the shape the
  test author expected, and the test author had written the schema.

- **Predicates are matched on content tokens, and comparison frames are excluded from
  the chain rule.** `date of birth` folds to `born_on`, whose spoken form is "born on",
  and questions say "when was X born" — the predicate was in the question and the
  preposition was not. `intent._content()` drops `STOPWORDS` from a predicate name and
  requires **all** remaining tokens, so `born_on` is named by "born" while
  `country_of_citizenship` still needs both `country` and `citizenship`. A bare token
  index would read almost every query as a chain; that is the failure this avoids.

  `intent.is_comparison()` suppresses the rule on a disjunction. "Which film has the
  director died later, A or B" names `director` and `died_on` — two predicates, a chain by
  that measure — and is two independent lookups whose answers are compared. Without the
  guard, content matching fired on a third of that family, where the walk costs 15.4
  points. Suppression is on the disjunction rather than on comparative words, because
  "earlier", "first" and "younger" are what one corpus happens to say.

  Matches are deduplicated by what the question said: `born_in` and `born_on` share the
  content token `born`, so "when was Alice born" named two predicates and read as a chain
  from one word.

  Measured on 2WikiMultihopQA, 12,576 questions, k=12: chained **35.4% → 41.4%**,
  compositional **32.1% → 40.0%**, and `flat`, `comparison` and `bridge_comparison` all
  unchanged, so the walk still does not run where it cannot help. The gate captured 0.9
  points of a 43.9-point gain three releases ago and now captures 13.8.

  **Answers and derivations move together.** The leg is worth +13.8 points of answer
  recall on chained questions and **+14.0 of chain recall** — 28.3% → 42.1% and
  25.5% → 39.5%. Ungated the two columns nearly meet, 72.2% against 70.3%: almost every
  answer the walk finds arrives with every triple that supports it.

  **An earlier version of this entry said the opposite, and the error was in the
  benchmark.** `place_of_birth` is an alias of `born_in`, so a claim written from 2Wiki
  evidence is stored under the canonical name; `bench/twowiki.py` compared the raw gold
  predicate against the returned row and never matched, for 6,624 of that corpus's
  triples. `answer` matched on the object alone and kept scoring while `chain` needed the
  predicate and silently failed, so chain recall read about 13 points low everywhere. The
  numbers in the merged commits for #32, #33 and #34 are understated for that reason;
  `docs/BENCHMARKS.md` carries the corrected table and the explanation.

  `inference` is unchanged at 46.4% through this whole series, as it has been through all
  of them: "who is the maternal grandfather of X" over `(X, mother, Y)` and
  `(Y, father, Z)` names a derived relation the question never says, and no lexical rule
  reaches it.

- **The graph leg's gate can see predicates nobody declared.** `classify` decides a
  question is a chain by counting the predicates it names, drawn from
  `PredicateRegistry.all_specs()` — which lists what somebody *declared*. A predicate
  written through `remember()` is never declared: the registry synthesizes a spec on
  demand and does not remember. So on a store whose vocabulary arrived that way the count
  was always one or zero and every chain question read as a lookup.

  `HybridRetriever` now takes a second look after the lookup legs have run. Their
  candidates are the store's vocabulary — observed rather than declared, and already
  narrowed to this query — and a question naming two of *those* predicates gets the walk.
  It only ever widens: it fires where intent weighting closed the leg on a store that
  configured it open, and never narrows what the classifier allowed.
  `intent.observed_refs()` is the same phrase matching over an explicit vocabulary.

  Measured on 2WikiMultihopQA, 12,576 questions, k=12: chained questions **29.1% →
  35.4%** answer and 19.8% → 26.1% chain recall, `compositional` 24.0% → 32.1%, and
  `flat` unchanged at 75.3% — so the discrimination holds and the leg still does not run
  where it cannot help. The gate captured 0.9 points of a 42.5-point gain and now
  captures 7.2.

  **Teaching the registry instead was tried and reverted.** Recording an observed
  predicate means recording a cardinality, the only one available is the default, and the
  store would then hold `MANY` chosen by nobody — silencing `memory_remember`'s note that
  it has *no* cardinality recorded, which is the only warning that two live values may be
  a contradiction. Three tests caught it. A read-path signal is not worth a write-path
  warning.

  **`inference` questions gain nothing, and that is the ceiling rather than a defect.**
  They ask "who is the maternal grandfather of X" over evidence `(X, mother, Y)` and
  `(Y, father, Z)` — a *derived* relation the question never names. Matching words against
  stored predicate names cannot bridge `grandfather` to `mother` + `father`; that needs
  synonymy or entailment, which is a model rather than a lookup. 83% of the available gain
  is still behind `intent_weighting=False`.

- **The graph leg now actually runs in the shipped configuration.** It was installed,
  weighted and switched off by two separate things, and the published explanation named
  neither of them.

  **Naming an instant switched the walk off.** `HybridRetriever._weights` overrides the
  classifier whenever `as_of` / `valid_at` / `known_at` is given — right for the temporal
  leg, since a caller who resolved an instant has said more about time than any word
  could. But the override sets the whole intent, and `Intent.TEMPORAL`'s multipliers put
  the graph weight at zero, so every time-anchored query ran two legs instead of three.
  "Where was Alice's employer based in 2019" is the query a bitemporal memory exists for
  and it was the shape that lost the walk. The temporal row still decides the other three
  legs; the graph leg now keeps the weight the query shape asked for. A plain temporal
  question still does not walk.

  **The classifier now counts predicates rather than matching a word list.**
  `intent.predicate_refs()` counts how many *distinct* predicates a question names, and
  two of them is a chain — one predicate is a question about one slot. Derived from
  `PredicateRegistry`, so it grows with the registry rather than with this file. **How far that
  reaches is narrower than it sounds, and worth stating**: `PredicateRegistry.learn()` is
  called only from the LLM-assisted predicate resolution in `write/pipeline.py`, so an
  offline store — or any predicate written straight through `remember()` — never teaches
  it. On such a store the rule sees the 23 builtins and nothing else. That is a real
  limit on the gain, and `bench/twowiki.py` is what made it visible: its predicates
  (`director`, `mother`, `date of birth`) are all learned ones, so the rule does not fire
  there at all. No word was added because a benchmark needed one. `RELATIONAL_MARKERS` is unchanged and still runs:
  "who is Alice's manager" names a relation in English and no predicate at all. The two
  signals fail in opposite directions. Matched as phrases, never tokens — `lives_in`
  splits into `lives` and `in`, and a token index would read almost every question as a
  chain. `classify()` takes an optional `registry` and behaves exactly as before without
  one.

  Measured on `bench/multihop.py`, shipped configuration: **2.9% → 6.4% at k=12 and
  7.6% → 21.8% at k=25**; two-hop questions **9.3% → 30.3% at k=25**. The ungated ceiling
  is 50.0%, and the remaining gap is one question family where the store holds
  `founded_by` and the question says "founded the" — morphology, not vocabulary. Matching
  head tokens instead was measured and rejected: this registry's head tokens include
  `in`, `is`, `do` and `has`, which makes "what is my name" a chain. `docs/BENCHMARKS.md`
  carries the table and the correction.

- **A graph leg in `search()`**, off by default. `GraphTraverser` has been able to answer
  "who does Alice's manager report to" since it landed, and `bench/multihop.py` measured
  what that is worth — 34.7% against 4.7% for a search-then-search loop at three hops.
  Nothing on the read path called it. `search()` fused two legs, both lookups, and a
  question whose evidence is two rows with a join between them was answered by whichever
  row happened to embed closest.

  The leg is Zep's φ_bfs: seeds are the folded `subject_key`/`object_key` of the
  best-scoring claims from the first fusion, so no entity extractor runs over the query
  text. `HybridRetriever` gains `w_graph`, `graph_seeds`, `graph_depth`, `traverser` and
  `intent_weighting`; `Explanation` gains `graph_rank`, `graph_score` and `intent`;
  `scoring.relevance()` gains `graph`/`w_graph`; `GraphTraverser` gains `spread()`, a
  keys-only entry that does not re-fold what the write path already folded.

  **The leg is gated on `states`, because it can only ever produce live rows.**
  `Store.adjacent` walks the live edges at the pinned instant and takes no `states`
  argument — a graph of retracted edges is not a graph, since a retraction says the
  connection was never there. So a search asking only for `ended` or `retired` must not
  run it, and it was running: on a store where one retired claim had a live neighbour,
  `search(states=["retired"])` returned that neighbour ranked *above* the retired row,
  because the seeds come from the lookup legs and the retired row was a good seed. An
  audit query answered with current facts, silently. Gated rather than post-filtered: a
  post-filter would test `claim.state`, which is the state **now**, and at a historical
  `known_at` the lookup legs correctly return rows that were live then — filtering those
  would fix this leg by breaking time travel in the other two. A search naming `live`
  among several states still gets the leg.

  **One stored claim is one path, whichever end was seeded.** `Path.signature` starts at
  `nodes[0]`, so a walk reaching Acme from Alice and one reaching Alice from Acme signed
  differently while being one row read from two ends. `seed_keys` emits *both* ends of
  each top-ranked claim, so for the head of the list the pair was guaranteed rather than
  possible, and the second reading spent one of the caller's `k` on a claim already in
  the answer — one slot in six at `k=6`, measured on seeds shaped the way `seed_keys`
  emits them. `Path.undirected` is the mirror-insensitive identity and collection keeps
  one per value of it, lexicographically, so which direction survives is a property of
  the content rather than of seed order.

  The dedup is at collection and deliberately **not** in the frontier: the two readings
  extend to different places — `alice→acme` grows towards Acme's neighbours and
  `acme→alice` towards Alice's — so deduping earlier would make the walk cheaper by
  making it reach less.

  Two smaller ones. `graph_seeds=0` now means no seeds rather than one: the cap is
  checked after a key is inserted, because the loop has to insert before it can know it
  is full, so zero read as "stop once you have at least none". And a third-party
  `Store.get_claims` that returns an id nobody asked for costs the graph leg a seed
  instead of taking the whole search down with a `KeyError`.

  **`w_graph` ships at 0.0, and the measured table is in `docs/BENCHMARKS.md`.** Neither
  public retrieval benchmark can see the leg: it walks claims, and both runs are episode
  retrieval — LOCOMO extracts 0 claims from 5,882 turns and LongMemEval 78 from 10,866,
  because `SalienceGate` drops any turn whose role is not `user` and the deterministic
  extractor's vocabulary is first-person declaratives. The LOCOMO reports with and without
  the leg are byte-identical. What moves is `bench/multihop.py`, over a store of asserted
  claims: **2.9% → 21.6% at k=12 and 7.6% → 50.4% at k=25**, with no seed entity supplied
  by the caller. That benchmark is synthetic and self-authored, which is an illustration
  of a mechanism and not evidence for a default — the precedent is the MMR rejection
  recorded in `hybrid.py`.

  **Those numbers are with `intent_weighting=False`, and the shipped configuration scores
  nothing at all.** Two of the three question families there contain no word in
  `RELATIONAL_MARKERS` — "who founded the company that X works at" — so the gate reads
  them as `lookup` and switches the leg off on exactly the questions it was built for.
  That is a gap in the vocabulary rather than in the gate: `works at` and `founded` are
  relations by any reading, and both are predicates in the store's own registry. Deriving
  the markers from the registry is the fix and it is **not done** — widening the list by
  hand against a benchmark this repository wrote is how a classifier gets fitted to its
  own corpus. A deployment turning the graph leg on should turn `intent_weighting` off
  with it; `bench/multihop.py` prints both columns so the cost is a number.

  It is opt-in rather than rejected because nothing regressed where it cannot help: with
  78 claims across LongMemEval's 940 sessions no R@k moved at all, knowledge-update
  included. The measured table and the reproduce commands are in `docs/BENCHMARKS.md`.

  A `Store` without a working `adjacent()` degrades to the two legs it had and raises
  `DegradedRetrievalWarning` once per retriever. That is caught rather than guarded,
  because `RemoteStore.adjacent` is present on the object and raises when called, which a
  `getattr` check cannot see.

- **Deterministic query-intent gating**, `memvara/retrieve/intent.py`. Four classes —
  `lookup`, `temporal`, `relational`, `open` — matching the categories LOCOMO reports
  separately, decided by a marker vocabulary over the query's raw tokens with no model
  anywhere. It is what makes the graph leg affordable to turn on: `lookup` and `temporal`
  queries skip the walk *before* the traverser is called, so they pay nothing rather than
  paying for a walk that is then weighted away.

  It reads the raw tokens rather than `analyze()`'s reduced terms, and that is the one
  subtlety: `when`, `whose` and `between` are all stopwords, and they are exactly the
  words that say what kind of question is being asked. The two functions share a tokenizer
  and want opposite halves of it.

  Multipliers scale the *configured* weights rather than replacing them, so a deployment
  that tuned `w_vector` keeps its tuning. Every entry is 1.0 except the graph gate, and
  every entry stays 1.0 until a per-category sweep moves it. `intent_weighting=False`
  turns the stage off wholesale and leaves `Explanation.intent` unset, which is how a
  ranking difference is attributed to it rather than argued about.

- **`--w-graph` on both benchmark runners**, so the table above reproduces, and a
  `search+graph` column in `bench/multihop.py` measuring the shipped read path rather than
  a caller who already knows the seed entity.

- **`demo/harness.py --reader stub`**, one offline command that runs all five arms end to
  end and reports them. The answer-quality apparatus existed and could not be run without
  a person in the loop: the blinded dump/answers round trip stops halfway by design,
  because the answerer is outside the process. That made it unprotected — nothing in CI
  touched it, and a change to the arms or to `recall()`'s rendering would have been found
  by whoever next ran it by hand, which on the record is once.

  The new path plans, answers with `evalkit.StubReader`, judges with `ContainmentJudge`
  and prints the same report, in about three seconds and with no key. It is deterministic
  end to end, which is the property that makes it worth wiring up rather than a
  convenience: `test_the_offline_run_is_identical_twice` compares two rendered reports, and
  `test_two_ingest_orders_produce_the_same_context` ingests the same turns in opposite
  order and compares the rendered prompt — the property `HybridRetriever` breaks score ties
  on `value_key` for, asserted where an evaluator would notice it breaking.

  **Its accuracy column is not a measurement of answers**, and the run says so twice: the
  stub picks the retrieved line with the most words in common with the question. The
  harness appends a line to `evalkit.stub_caveat`'s banner because that banner ends by
  naming `--reader anthropic`, which is right for the `bench/` runners it was written for
  and does not exist here — the reader that measures answers is `--reader file`, a person
  or an agent. `--dump` is now required by `--reader file` rather than unconditionally.

- **`valid_at` on `memory_search`**, so the MCP surface can move the world clock without
  the belief clock. It previously took `as_of` only, which is exact sugar for
  `valid_at=known_at=T` — one instant on both axes — and that made two questions
  unaskable from a tool. The first is the row the axes were split for: a correction
  learned in August about June is invisible to `as_of=June`, because that call rewinds
  belief past the correction. The second is worse, because it has no workaround: a fact
  written now whose valid interval is **already over** needs `recorded_at <= known_at`
  and `valid_from <= valid_at < valid_to`, and no single instant satisfies both — so a
  backfilled closed interval was stored correctly, shown by `memory_history`, and
  reachable by no `memory_search` call at all.

  Additive plumbing rather than new retrieval behaviour: `Memvara.search`,
  `ScopedMemvara.search`, `HybridRetriever` and `state_predicate` have always threaded
  the two axes independently, with per-clause axis tags. Only the tool surface collapsed
  them.

  `known_at` is deliberately **not** exposed. It is the belief clock, the axis a caller
  can use to misread an audit trail, and audit reads stay a library and REST job — which
  is what the packaged skill already tells an agent to say. Passing `as_of` and
  `valid_at` together is refused at the tool boundary rather than by a bare `ValueError`
  from inside the library, so the message arrives in the same voice as the other argument
  errors and names which one answers which question.

  The write note for a closed backfilled interval moves with it. It was corrected in the
  previous release to promise only `memory_history`, because the `as_of` it used to name
  could not work; it now names `valid_at`, which can. `memvara/skills/memvara/` changes
  too — it stated the `as_of`-only limitation outright — and so does the vendored copy
  under `plugin/`, which is asserted byte-identical.

- **A write that embeds to nothing now says so**, via `UnembeddableTextWarning` and the
  `write.embedding_unusable` counter, tagged by script. `HashingEmbedder` — the default
  with no extras — tokenises `[a-z0-9']+` and builds its character n-grams over the
  rejoined word list, so a claim with no Latin in it reduces to an all-zero vector. Every
  layer then behaves correctly: the store accepts the row, retrieval abstains on a zero
  norm rather than inventing a rank, and the claim still answers by predicate. Nothing
  raised, and the result was a fact vector search could never return, with no signal
  anywhere. Detection is one norm the embedder had already computed and discarded.

  The claim is still stored — refusing the write would lose data over a retrieval
  limitation. The warning fires once per pipeline and the counter counts every claim,
  the same split `write.embedding_rejected` already uses: one line answers *is this
  happening*, and only the counter answers *how much of this store is affected*.

  **This is a floor, not a fix, and it does not catch mixed text.** A norm is a
  whole-string measure, and `remember("user", "lives_in", "里斯本")` renders as
  `user lives in 里斯本`, which embeds perfectly well from the Latin scaffolding alone —
  so the object is invisible to vector search while the claim looks healthy. The fix is
  an embedder that tokenises the script. `README.md`'s limitations section now says this
  outright; it previously named the schema, gate and extractor as English-centric and
  omitted the embedder, which is the larger of the two.

- **`import_supermemory`**, in `memvara.compat`, moving a Supermemory
  account into a store over stdlib HTTP with no SDK. Supermemory keeps
  documents rather than a mutation log, so unlike `import_mem0` nothing
  can reconstruct supersession it was never told: documents arrive as
  episodes on their original `createdAt`, and the receipt reports
  episodes and claims separately so a store with no extractor cannot
  mistake "nothing was structured" for "nothing arrived".
- **`include_episodes` on `memory_recall`.** Recall answers from claims,
  so a store holding imported or unextracted text looked empty through
  the tool surface however much it held — there was no way to ask for
  the excerpts at all. Default stays false: a fact is a settled reading
  of what was said and an excerpt is not.

- **Declared predicate vocabularies**, via `MEMVARA_PREDICATES`. It takes shipped
  pack names, paths to TOML files, or a comma-separated mix with later
  entries winning, and an `engineering` pack ships with the package.
  `PredicateRegistry` has always accepted `specs=`, but an MCP client can
  only set environment variables, so every server-backed store was pinned to
  the built-in vocabulary — and a predicate outside it accumulated instead of
  superseding, and decayed at the slow default rather than its own. A
  malformed pack is a startup error naming its own fix, not a surprise at the
  first write.
- **A declared spec now outranks a persisted learned one.** Rehydration skips
  a learned spec whose name a declaration already holds, so a vocabulary can
  correct a store that guessed rather than only describe a fresh one.
  Forward-only: it changes what supersedes on the next write and retires
  nothing already stored.

- **A real agent skill**, at `memvara/skills/memvara/`. `SKILL.md` is a
  dispatcher (which surface, then which job) plus `references/` for
  integrate / hosted MCP / write-and-correct / time / scopes / governance /
  mem0, worked turns, and a short paste for clients that have no skill
  directory. `memvara-mcp init --agent` accepts `claude`, `cursor` and
  `grok` — those names pick the destination, not different prose — and
  `--skill-only` writes the tree without touching `.mcp.json`.
- **A plugin** at `plugin/` is the source layout. Claude Code installs from
  the dedicated public marketplace
  [`memvara/claude-memvara`](https://github.com/memvara/claude-memvara)
  (`/plugin marketplace add memvara/claude-memvara`). No `npx`, no local
  stdio. A loop you wrote still uses the library, REST, or MCP as a client.

### Changed

- **`MEMVARA_MODE=cloud` refuses at construction instead of failing on the first tool
  call.** It built a `Memvara` over a `RemoteStore` and started a server; that server
  listed twelve tools and raised `NotImplementedError` on the first one a model reached
  for. `RemoteStore` wires seven `Store` methods and the engine calls a different set on
  every turn — `put_claim`, `add_episode`, `candidate_ids`, `lexical_search`,
  `vector_search`, `competing_claims` — none of which the REST facade has an endpoint
  for.

  A failure that arrives mid-conversation as a tool error is in the one place it cannot be
  acted on: the model cannot fix a deployment and whoever configured it is not in the
  room. `build_memvara` now raises a `ConfigError` naming exactly which methods are
  missing and what to run instead.

  The check is `_ENGINE_NEEDS - RemoteStore.WIRED`, a set difference rather than a
  literal refusal, so it **un-refuses itself** the day those endpoints exist and `WIRED`
  grows — and a test fails on that day and says what to delete. `RemoteStore.WIRED` is
  kept honest by another test that derives the same list from the source rather than
  restating it.

  `docs/OPEN-CORE.md` carries the decision behind it: **diverge, and gate**, with a table
  of which side of the open-core line each seam is on and why converging the two shapes
  would move contradiction resolution and scope enforcement to the client.

- **The seven design invariants in `docs/INTERNALS.md` are restated as Claim / Scope /
  Sketch / Measured**, and an eighth is added. The format earns its place on the last two
  lines: *Sketch* names the code that makes the claim true, and *Measured* is either a
  number this repository produced or an explicit statement that none exists and only a
  test stands behind it. Several already carried measurements in prose and those moved to
  the Measured line; **where nothing was measured the line says so and names the enforcing
  test.** No numbers were invented.

  The eighth is the one that needed writing down, because it was being read as holding
  further than it does:

  > **Claim.** No MCP client can backdate the transaction clock.
  > **Scope.** The MCP tool surface only. `Memvara.remember(recorded_at=...)` is a public
  > Python parameter that writes the record clock directly, and `Reconciler.apply` clamps
  > forward-dating only — backdating is permitted deliberately, for replays and imports. A
  > deployment needing this end to end must not expose the Python API to untrusted callers.

  The falsifiable part is a test that walks every property of every tool schema and fails
  if a record-clock argument appears, plus one that writes a fact dated to 2019 through
  `memory_remember` and asserts it is *recorded* today. That is what stops the gap
  reopening silently — a new tool taking `recorded_at` because it seemed harmless would
  otherwise ship green.

### Fixed

- **A blank part of a triple was stored as nothing, in silence.** `memory_remember` with
  an empty or whitespace `subject`, `predicate` or `object` wrote nothing and reported
  `added 0, ended 0, retired 0, already-known 0, no-fact 0` with `isError` false — which
  is also exactly what a legitimate already-known write looks like. A model had no way to
  tell "you sent nothing" from "there was nothing to do", so it either believed the fact
  was on record or repeated the call. Now refused, naming the field, like every other
  rejection on this surface.

- **`memory_end` on an already-retired claim contradicted itself in one line.** It
  rendered the state as `retired` and then asserted, in the next sentence, that
  `memory_history` shows the claim as *ended, not retired*. Stored state was never wrong —
  the store keeps `retired`, the stronger statement and the one made first — so this was a
  message defect. But an agent that believed it would report a false reason for a change,
  which is the one mistake the two-tool split exists to make unmakeable, because nothing
  downstream can detect it. It now says the claim is already retired, stays that way, and
  that nothing changed.

- **`memory_since` with a future instant answered "what you knew then still stands".**
  True of the future and useless: unqualified it reads as *you are up to date*, so a model
  stops asking, having learned nothing about the period it meant to ask about. It now says
  the instant has not arrived, and names the usual cause — a local time sent as UTC lands
  ahead of now for anyone west of Greenwich.

- **`recall(budget=)`'s cut notice implied a total it never counted.** It said "n further
  notes **matched**", counted over the pool `search()` had already truncated to `k`. So it
  reported how many *retrieved* notes the budget dropped and nothing about how many more
  the store holds, while reading as the complete remainder. Reworded to name the second
  cap. Deliberately three characters *shorter* than the sentence it replaces: the notice
  is counted against the budget and is the floor of a squeezed block, so a longer one
  silently lowers how many real notes fit — the first rewrite was 29 characters longer and
  cost a note, which is now pinned by its own test.

- **Two tool descriptions claimed more than the code does.** `memory_search` published
  `relevance` as a bare number without saying it is match strength adjusted by recency,
  writer-set confidence and reinforcement — so a smaller number reads as a worse match
  when it may not be. And `memory_remember.memory_type` said omitting it lets "the
  predicate's own classification decide, which is usually right"; a predicate this store
  has never seen has no classification and becomes `semantic`, and nothing infers a type
  from the words. Both now say what actually happens.

- **A failing tool replayed its exception message into the model's context unflattened
  and uncapped.** Stored claims have been treated as untrusted at the rendering boundary
  since the beginning; the failure path was the one line that was not, and it is the same
  kind of text arriving through a different door. An exception message is not this
  process's to trust — a store error can quote a value somebody wrote, and against a
  hosted backend it can carry an upstream body verbatim — so it could open its own line,
  spell something that reads as a result row, or run to the length of an HTML error page
  inside a context window.

  It now goes through `safe_detail`: `safe_line` for the structure, and a 300-character
  cap for the volume. The exception *class* is kept whole — it is a Python identifier and
  it is the half that says what went wrong. `SECURITY.md` now names the failure path as in
  scope on the same terms as the result path.

- **`memvara-mcp login` echoed an upstream error body whole.** A different audience and so
  a different risk: this reaches a terminal rather than a model, and the problem is
  volume. A gateway answering with an HTML error page put five kilobytes into stderr —
  which for a login run in CI is a build log, and on a public repository that log is
  public. Bounded to 200 characters, the same cap `compat/supermemory_import.py` already
  applies to the same kind of text. A short body, the normal case, is unchanged.

- **A predicate folded onto its canonical name silently, changing how many values the
  slot holds.** `uses_tool` is an alias of `prefers_tool`; a predicate the store has never
  seen is `MANY` and accumulates, and `prefers_tool` is `ONE`, where the next write ends
  the last. So writing two values under `uses_tool` keeps one, and writing them under a
  name the store does not know keeps both — a different outcome for the data, decided by a
  rename the caller was never told about. `memory_remember`'s own schema offers
  `uses_tool` as an example spelling, so this was reached by following the tool
  description rather than by getting it wrong.

  `memory_remember`, `memory_forget` and `memory_end` now say when the predicate they
  acted on is not the one they were given, and on a write they say what the fold decided
  about cardinality. The fold itself is unchanged and remains the right behaviour: without
  it two spellings of one fact become two slots that cannot contradict each other.

  **Addressing was never affected**, and the note says so rather than implying otherwise —
  every predicate-addressed tool resolves through the same registry, so the original
  spelling still finds the fact.

- **`subject` and `predicate` had no length bound.** A 2,000-character subject and a
  1,000-character predicate were both accepted, echoed back by the write, and re-rendered
  by every later `memory_search` and `memory_recall` that matched them. The validator
  implemented `minimum`/`maximum` for numbers and `enum` for strings but never checked
  string length, and no tool schema declared one. It now supports `maxLength`, and the two
  arguments that name a *slot* declare it: 128 characters for `subject`, 64 for
  `predicate`. `object` is deliberately left uncapped — it carries the fact itself, and a
  caller who needs a long one is not misusing the tool.

- **A write note sent the model to a search that can never succeed.** Storing a fact whose
  valid interval was already over emitted `memory_history shows it, and so does
  memory_search with as_of inside that period`. The second half is false at every instant:
  reaching a closed interval needs a moment *inside* it, reaching a claim recorded just now
  needs a moment at or after the write, and `as_of` moves both clocks together, so no
  single value satisfies both. The claim is stored and correct and simply not reachable by
  search from this surface.

  That made the note worse than saying nothing. `_interval_note` exists because a correct
  write whose effect is invisible gets "fixed" by a second write with the argument dropped
  — and this pointed at a query that comes back empty, with the server's authority behind
  it. It fires on the single-call closed-interval write, which `memory_remember` actively
  recommends over write-then-`memory_end`, so the recommended path was the one being
  misinformed. The note now names `memory_history`, says search will not find it, and says
  why. Exposing the two time axes separately on `memory_search`, which would make the
  original promise keepable, is #16.

- **`memory_history` printed only one of the two clocks.** Rows carried `recorded_at` and
  the closing instant, never `valid_from`, under a header that said "oldest first" —
  ordering by recording time, as every backend's protocol declares. A value backfilled
  today about two years ago was therefore listed last while being the earliest thing the
  slot had ever held, with nothing in the output saying so. Rows now carry `true from`, and
  the header names which clock the order is in. The `ORDER BY` is unchanged: printing the
  clock the order is *not* in is what makes the order safe to read.

- **Stored text could forge a result row without opening a line.** `_safe_line` flattens
  a claim so it cannot start its own block, and every surface writes its metadata before
  the untrusted span so nothing can *follow* a claim and impersonate this system. Neither
  rule covers the rest of the line the claim is already on: a value containing
  `[id=cl_… relevance=0.99] …` rendered as what read like an additional, higher-scoring
  result. It reached `memory_search`, `memory_recall`, `memory_history`, `memory_since`
  and `memory_why` — including the two surfaces the dispute flow tells an agent to consult
  when a user challenges a memory, which is what made it worth a fix rather than a note.
  Write and read need not be the same session, so the payload is planted once and read
  later by an agent that never saw it arrive.

  `_safe_line` now maps `[` and `]` to their fullwidth forms wherever they occur, rather
  than only stripping markers from the head. A substitution and not a deletion: a note
  about `arr[0]` still reads as `arr［0］`, where dropping the brackets would quietly
  rewrite the fact into a different one. **Rendered output changes for any stored value
  containing a square bracket** — see `docs/UPGRADING.md`. Nothing on disk is touched, and
  because the fix is at render time it also covers rows already in the store, which a
  write-time fix could not: `remember(..., text=…)` lets a caller supply the rendered line
  directly and the reconciler deliberately preserves it.

  `memvara.server.tools.safe_line` now calls `Memvara._safe_line` instead of holding a
  second copy of it. The two had already drifted — the library's set had stopped stripping
  `>` and backticks — so the same stored value was neutralised one way through
  `memory_recall` and another through `memory_search`.

- **`memvara-mcp login` never completed against the hosted console.** Two independent
  refusals stacked: `POST /api/auth/device/authorize` answers 403 `csrf_failed` unless
  `X-Memvara-CSRF` is present (presence is the whole check with no session cookie, which
  a CLI never has), and a successful mint is 201, which the client treated as "the
  server refused to start." Either one alone was enough for login to exit 1 before a
  browser opened. The client now sends the header and accepts 200 or 201.

- **`supersede()` reported closing nothing, from the one call whose whole purpose is
  closing a claim out.** `receipt.closed` came back empty — and with it `receipt.ended`
  and `receipt.retired` — while the predecessor was closed correctly on disk. Only the
  report of it was missing.

  The cause is an ordering the transaction requires. `_write_claim` closes the
  predecessor **before** `assert_claim`, because afterwards the reconciler gets there
  first and stamps the wall clock over `at`, which turns a backdated import into a pile
  of things that all changed today. But `closed` is filled in by that same reconciler
  from the victims it finds, and by then the predecessor is no longer live: it finds
  none, reports none, and the receipt went back to the caller as `assert_claim` built it.

  Two closure surfaces disagreed as a result. `forget()` returns its closed claims
  directly, so a caller replaying somebody else's mutation log had the retraction
  confirmed through one door and got silence from the other, with no way to tell that
  silence from "nothing was closed" short of re-reading the store. `WriteReceipt.closed`
  is documented as "claims this write closed out, as they read *after* the write", and
  that is now what it returns: `close_out` has already stamped the object, so
  `Claim.state` names the axis that actually stopped and `ended`/`retired` split it.

  **If you meter on `len(receipt.closed)`**, supersede-heavy workloads will report more
  closures than before. The old number was an undercount, not a different definition.

- **`compat/_notes.write_note` had the same defect, and dropped `episode_ids` as well.**
  Same ordering, one layer below the facade — so *every* note write, including ones that
  retire nothing, reported storing no turn while the turn sat committed on disk. That is
  the door `Mem0Memory.add(infer=False)` and every row of a mem0 import go through. It
  was latent rather than live: the importer counts its own rows and reads only `added`
  and `reinforced`, which is the argument for fixing it before something trusts it
  rather than after.

  Both receipts are now completed after their transaction commits, so neither can
  describe a write that rolled back.

## [0.2.0] — 2026-08-16

The first release with an agent-facing surface: a prompt block that can say what it
rested on and how big it is, a delta for a resumed session, and a skill that ships with
the package. Ten MCP tools, up from eight.

**`0.1.0` shipped eight tools and none of the below.** A reader who installed it before
today has `true_since`/`true_until` on `memory_remember` and no `close`, no `memory_end`,
no `memory_since`, and no `memvara-mcp init`.

### Upgrading

The long form of everything in this section is [`docs/UPGRADING.md`](docs/UPGRADING.md).

- **"Live" is no longer `invalidated_at IS NULL`.** This is the sharp corner of the
  closure split below, and the one thing to read before upgrading.

  Superseding closes `valid_to` and leaves `invalidated_at` unset, so a superseded claim
  is `ended`: **neither live nor invalidated**. `stats()` now reports that population as
  `ended_claims`; the claim counts a store returns still do not sum, and `claims` remains
  the only total that covers everything.

  ```sql
  -- was equivalent to "live". It is not any more.
  SELECT count(*) FROM claims WHERE invalidated_at IS NULL

  -- live: two clocks, four columns, both bounds on each
  SELECT count(*) FROM claims
   WHERE recorded_at <= now()
     AND (invalidated_at IS NULL OR invalidated_at > now())
     AND valid_from  <= now()
     AND (valid_to IS NULL OR valid_to > now())
  ```

  **The shape of the mistake.** Every surface that had ever spelled "live" as the
  one-column test was *correct before this release and wrong after it* — silently, with
  no exception, no migration error and no red test, because the two spellings selected
  the same rows for as long as superseding closed both clocks. Nothing about the code
  changed; what it means did. The error is always in the same direction, counting too
  many: a store whose one address has changed four times reports **five** live claims
  instead of one, and every such number steps up on the day of the upgrade for something
  no user did. On a usage-metered or billed surface that is money, and the step is
  unfalsifiable from inside the data, because no other series moves with it.

  **The Python spelling has the same problem and is easier to miss**, because it reads
  like an attribute check rather than a query:

  ```python
  # was equivalent to "live". It is not any more.
  live = [c for c in claims if c.invalidated_at is None]

  # live: ask the claim, which knows about both clocks
  live = [c for c in claims if c.is_live()]
  ```

  `Claim.invalidated_at`'s own docstring now says so, since that is where a reader who
  suspects the bug will look first.

  **What to grep for** — in application code, dashboards, saved queries, alert
  thresholds, notebooks, and any third-party `Store`:

  ```
  invalidated_at is None        # Python
  invalidated_at is not None    # Python — the mirror, and *not* the complement any more
  invalidated_at IS NULL        # SQL, also `is null`, `ISNULL(`, `= NULL`
  invalidated_at IS NOT NULL
  ```

  Every hit is one of three things. A **liveness** test is now wrong and must become the
  four-column predicate. A **retirement** test — "which records did we stop believing" —
  is still exactly right, and now selects a strictly smaller set than "not live". An
  **audit** view that wants everything ever displaced should read `invalidated_by IS NOT
  NULL`: the pointer is written under either closure, which is the whole reason it is a
  separate column.

  Three copies of the wrong test existed across this project's own repositories when the
  change landed, one of them a billing gauge, and finding them was manual.
  `memvara.store.live_predicate` is now the single exported home for the right one.

- **`WriteReceipt.invalidated` is now `WriteReceipt.closed`.** Same list, same object;
  the old name stays as an alias and still works. Rename at your call sites when
  convenient, and read the entry under *Changed* for why the number it holds needed
  splitting rather than just renaming.

- **The MCP server's write summary changed shape.** `added N, retired N, ...` became
  `added N, ended N, retired N, ...`. Tool output rather than API, so there is nothing to
  migrate — but an eval fixture or a saved transcript that pins the old string will not
  match.

- **`states=` is additive; `include_invalidated` keeps working forever.** Nothing to
  migrate on the facade. Two things will otherwise surprise you: asking for all three
  states makes `valid_at` inert, and `iter_claims`'s unflagged view is `("live", "ended")`
  rather than live-only. Both are pre-existing semantics now written down.

- **Third-party `Store` implementations must widen four read-path signatures** —
  `candidate_ids`, `lexical_search`, `vector_search` and `iter_claims` — and add
  `stats()["ended_claims"]`. `Store.erase_claim` returns `dict[str, int]` rather than
  `bool`. `Memvara.erase()` is unchanged and still returns `bool`.

### Added

- **`recall(with_ids=True)` returns the claim ids it rendered**, as `RecallResult`, in
  render order and 1:1 with the numbered notes. `recall()` returns `str` exactly as before
  unless asked; the overloads follow `search()`'s. Previously the prompt-shaped surface
  returned text and nothing else, so a caller who wanted to cite what an answer rested on
  had to re-run retrieval through `search()` and hope the two agreed. Ids only, never
  scores — and live facts only, since episodes have no claim id and citing a past value as
  the source of a present-tense answer is worse than citing nothing.

- **`recall(budget=…)` bounds the block by size rather than by count.** `k` bounds the
  number of notes and claim text is variable, so `k=8` was a context budget by convention
  only. Notes are dropped **whole** — half a fact in a prompt is a false fact — and the
  block says how many did not fit, so a model can tell a bounded answer from a complete
  one. `counter=` takes a real tokenizer; the default is a length heuristic that is roughly
  right for English prose and **under-counts CJK**, which is stated where it is defined
  rather than discovered in production. Core's dependencies remain `numpy` and nothing else.

- **`since(when)` — what changed while the agent was away**, as a `Delta` of `added` and
  `gone`. A supersession lands in both halves. This needs no new `Store` method: it is two
  `candidate_ids` calls with **both clocks** pinned to `when`, differenced. Pinning the
  belief clock alone is the tempting version and is wrong — `valid_at` then defaults to
  now, so a claim whose world-interval has since closed never enters the "then" set, and
  the supersession reports an arrival with nothing beside it.

- **`memory_since` on the MCP surface**, bringing the server to nine tools. It returns
  structured rows rather than a prompt block, and the reason is the same one that keeps
  `states=` off `recall()`: a delta necessarily carries claims that stopped being believed,
  so a `recall`-shaped twin would be an un-delete reached through another door. The two
  halves carry both a directional heading and a per-line `+`/`-`, because they fail
  differently — the heading is what a model reads, the mark is what survives the heading
  scrolling out of view.

- **`memvara-mcp init --agent claude`** writes the client's server block, the packaged
  skill and a CLAUDE.md snippet into a project, with `MEMVARA_DB` already absolute. It
  never rewrites an existing `.mcp.json`: where one exists and names another server, it
  prints the entry *without* enclosing braces, indented to paste inside `mcpServers`,
  because a self-valid block is the one people paste whole and break the file. `command` is
  `sys.executable` rather than the documented `python3` — the client launches without a
  login profile, so `PATH` finds the wrong interpreter for anyone in a virtualenv.

- **A packaged agent skill** (`memvara/skills/claude/SKILL.md`) carrying only what no single
  tool description can: the correction protocol as a sequence, scope hygiene and the
  `MEMVARA_SESSION` trap, what is worth storing at all, and what changes on a server with no
  extraction model. A test asserts it shares no 6-gram with any tool or property
  description, so the two cannot drift into two sources for one fact.

- **`RecallResult` and `Delta` are exported** from the top-level package, for the same
  reason `Closure` is: both are return types on four facade methods each.

- **`true_since` and `true_until` on `memory_remember`: the other end of the interval.**
  `memory_end` has always taken an `at` — the instant a fact stopped being true — and
  nothing on this surface could say when one *started*. So `Claim.valid_from` took its
  `default_factory=utcnow` on every tool write, and a store whose entire pitch is two
  independent clocks let an agent set one end of the valid interval and not the other.

  The failure it caused, with its real numbers. Recording project state, an agent wrote
  `quota_gate status "not installed"` at 00:50:13Z to represent a belief it had held
  earlier that morning. The gate had been installed at 00:04:07Z, 46 minutes before.
  `valid_from` defaulted to the write instant, so the stored claim asserted *"not
  installed, from 00:50:13Z onward"* — false at every instant of the interval it claimed.
  Nothing warned, because `added 1` is also what a correct write says. The symptom
  surfaced later and was misread: closing the claim at the install instant put `valid_to`
  before `valid_from` and `close_out` clamped it, which was the store correctly refusing
  to represent a fact that ended before it began, read as an obstacle rather than as the
  diagnosis.

  This is a tool-layer gap only. `Memvara.remember` has always accepted `valid_from` and
  `valid_to`, and the reconciler has always superseded along valid time rather than
  arrival order; nothing underneath changed.

  **The name is the decision, and it is deliberately not `at`.** Symmetry with
  `memory_end` argues for `at`, and the verb each name attaches to is why that is wrong.
  On a tool whose verb is *end*, "at" can only mean the instant of the ending — the tool
  name has already fixed which event is being timed. On a tool whose verb is *record*,
  "at" attaches to the recording, so it reads as transaction time at least as readily as
  valid time. The same spelling on the two tools would name two different clocks, and a
  model that took the second reading would forge history with a correctly-spelled call
  that nothing downstream could detect. `true_since` cannot take that reading: a record
  is not "true since". The symmetry given up is one of spelling; the symmetry kept is one
  of meaning.

  **`recorded_at` is still not reachable from any tool, on purpose.** Valid time is a
  claim about the world and the caller is entitled to it. Transaction time is a claim
  about the *record* — when this system came to believe something — and a caller who can
  backdate it can write an audit trail nothing downstream can falsify. `true_since`'s own
  description says so, which is where a model meets the boundary rather than a changelog.

  **`true_until` closes the interval in the same call, and that is a correctness gain
  rather than a keystroke one.** A backfill of a finished fact written as a write plus a
  `memory_end` leaves the store answering the present-tense question with a value already
  known to be false for as long as the two calls are apart — and permanently, if the turn
  ends before the second one. An interval that ends before or exactly when it begins is
  **refused** rather than clamped, unlike `memory_end.at`: there the claim already exists
  and the instant is second-hand, so refusing would leave no way to close it at all; here
  both ends arrive in one sentence, and clamping would store a zero-length claim that is
  true at no instant, returned by no query and identical at the call site to a successful
  write — the original defect in a second costume.

  **A past-dated write ends what it displaces where the new value began**, not now — the
  reconciler's existing rule, now reachable: ending it at "now" would leave a window in
  which both values were true, which for a single-valued slot is two answers to one
  question. A write dated *behind* an existing later value is closed at that value's start
  and becomes history rather than news, so an import cannot rewrite the present.

  **A future-dated write is stored, in force from its instant, and says so.** It ends the
  current value at that future instant — so the old value keeps answering until then and
  the new one takes over — and the reply names both halves, because `added 1` beside a
  `memory_recall` that returns the *old* value is the outcome most easily mistaken for a
  call that did nothing, and the obvious retry is the same write with the argument
  removed.

  Omitting both arguments behaves exactly as before: one clock read, both axes, and a
  test pins it.

- **`memory_end` on the MCP server: the closure an agent could not ask for.** The core has
  had two closures since the axes were separated — `"ended"` means the world changed,
  `"retired"` means the record was wrong — and `forget()`, `delete()` and `supersede()` all
  take one. The MCP surface offered only `memory_forget`, which retires. So an agent
  closing out a fact that had genuinely stopped being true had to assert that the record
  had been an error, which it had not.

  Found by using the server on a real task. Two `memory_remember` calls recorded
  `quota_gate status "not installed"` and then `"installed"`; `status` is an unknown
  predicate and therefore multi-valued, so nothing was displaced and `memory_recall`
  returned both, adjacent, with no ordering signal. The only closure available would have
  written a false statement into the audit trail to stop the store contradicting itself.

  `memory_end` takes the same two addressing modes as `memory_forget` — `predicate` (with
  `subject`) for a whole slot, or `claim_id` for one value — plus an optional ISO-8601
  `at`, the instant the fact stopped being true, defaulting to now. `at` is what makes the
  tool worth having separately: a fact that stopped last Tuesday must close on Tuesday, or
  every later `as_of` and `valid_at` query reports a week of believing something already
  false. An instant before the fact began is clamped to its start rather than inverting the
  interval, and the reply reports where the closure **landed** rather than where it was
  asked for. A future instant is allowed, means the fact is true until then, and says so —
  otherwise the value goes on answering `memory_recall`, which reads as a failed call and
  sends a model back to `memory_forget`.

  **Two tools rather than a `closure` enum on `memory_forget`, and the argument is about
  where a model commits.** It picks a tool by name, from a list, before it opens a schema —
  and `memory_forget` already asserts one of the two answers, so a flag on it would be
  asking the model to overrule the word it had just chosen. Splitting them puts the fork at
  the point the choice is actually made, and matches the shape `delete`/`erase` and
  `forget`/`purge` already have in `core`: operations that mean different things get
  different names, not a flag. The discoverability cost of a second tool is paid off in
  each description, which names the other and states its own reading, so the only route to
  the wrong closure runs through a paragraph that points at the right one.

  Ending is `destructiveHint: true` like retiring, hidden on a read-only server like every
  other write, and reversible by an operator: the claim stays on disk, stays in
  `memory_history` — rendered `ended`, distinguishably from `retired` — and stays visible
  to `memory_search` with `as_of`.

  `memory_forget`'s own description said "storing the new value already retires the old
  one". It does not; it **ends** it, which is right for a change and wrong for a mistake.
  That sentence told a model a genuine correction was handled automatically when what was
  recorded was a world event, so it is now corrected rather than merely joined.

- **`neighborhood()`, `history()` and `paths_between()` resolve learned aliases on the
  probe.** Previously a probe carried no alias stamp, so `neighborhood("Big Blue")` folded
  deterministically and missed every claim stamped `ibm`.

  `paths_between()` resolves **both** ends, since both are probes and neither is more
  stamped than the other; `[]` from it reads as "not connected", which makes asking under
  the wrong name the quietest of the three failures. Where the two ends turn out to name
  one entity the answer is `[]` — the same answer `paths_between(x, x)` has always given,
  for the same reason. Intermediate nodes are **not** alias-resolved: an entity appearing
  as `big blue` on one hop and `ibm` on another still does not join.

  Resolving the probe *to* the canonical — the obvious fix — would have made it worse: a
  merge never re-keys the past, so the store holds claims under both spellings and taking
  only the canonical trades one half of the entity for the other. `EntityRegistry.probe_keys`
  returns **every** key naming the entity instead — the deterministic fold first, then the
  canonical, then sibling aliases — and the deterministic fold remains the fallback, so a
  probe with no learned alias behaves exactly as before. Nothing on disk is re-keyed; only
  the read is widened. Probes resolve under the reader's own `owner_key(scope)` and no
  further, so a merge in one tenant cannot redefine an entity for another.

- **`MEMVARA_EMBEDDER` on the MCP server, and its default is `hashing`.** `ServerConfig`
  had an `llm` lever and no embedder field, so `build_memvara` took whatever
  `default_embedder()` returned — a sentence-transformers model as soon as that package is
  importable, and `memvara[rerank]` installs one because a cross-encoder is one.

  Since `Memvara.__init__` fingerprints the embedder against the store, **installing an
  extra made the server refuse to open its own store at startup.** The error names the
  extra as the likely cause and tells the reader to pass the original embedder explicitly;
  through this door there was no way to. That unreachable remedy is what made it a bug
  rather than a rough edge.

  Values: `hashing` · `hashing:<dim>` · `local` · `local:<model>` · `auto`. They are
  spelled the way `EmbedderMismatchError` and `memory.db.embedder.json` already spell
  them, so an operator **copies** rather than composes — `written by hashing:512:3-5`
  becomes `MEMVARA_EMBEDDER=hashing:512`. An unknown value is a `ConfigError` at startup
  listing all five, matching `MEMVARA_LLM`.

  **`hashing` rather than `auto` is the fix; `auto` would only have been a workaround** —
  the defect is that the store's vector space was a property of the machine's package set,
  and only a named default removes that. A deployment with no extras that sets nothing is
  byte-identical to before: same class, same width, same `hashing:512:3-5` fingerprint.

  **A deployment that deliberately installed `local-embed` now fails at startup** with the
  dimension mismatch naming its own store's width, and recovers with
  `MEMVARA_EMBEDDER=local` (or `auto` for the exact prior behaviour). Nothing is damaged —
  the refusal happens before any write — and the CLI now catches that error and prints
  where to type the fix, rather than the traceback that was the original complaint.

- **`recall(include_history=True)`** appends, for each fact the call surfaced, the values
  it used to have — under their own header, after the live block.

  Found by running the thing end to end: an agent asked "what plan were they on before?"
  from a `recall()` prompt got the current value and no signal that the past was missing
  rather than absent. `history()` had always answered it, but only for a caller who knew
  to ask a second, differently-shaped question — which an agent reading a prompt cannot.

  **Only `ended` values are rendered, never `retired` ones.** That bound is why this is
  safe on a surface which deliberately refuses `states=`: an `ended` value is the fact's
  own past and we still believe it was true while in force, whereas a `retired` one is
  something we were wrong about, and putting that in a prompt is the un-delete the
  signature exists to prevent. A claim that ended and was later retired is `retired` and
  stays out. The filter is `state == "ended"`, never `state != "live"`, and a test holds
  all three states in one slot so the looser spelling cannot pass. See `SECURITY.md`.

  History is fetched once per fact slot, so a multi-valued predicate with four live
  values costs one lookup rather than four.

- **An opt-in cross-encoder reranker: `memvara.rerank`.** Off by default, and "off" means
  the stage does not exist — no import of a backend, no model, no network. A subprocess
  test asserts that and goes red if the backend is ever made an eager import.

  ```python
  from memvara.rerank import CrossEncoderReranker      # pip install 'memvara[rerank]'
  mem = Memvara("memory.db", read_reranker=CrossEncoderReranker(), read_rerank_top_n=20)
  ```

  Measured on LOCOMO's 1,531 evidence-labelled questions, vector leg pinned, identical
  candidates: **R@12 62.0 → 66.5 and R@1 30.5 → 44.9**, MRR 44.9 → 59.2. R@20 is
  unchanged at 67.4 and must be — reranking the top 20 can only reorder within it — so
  the whole effect is evidence moving *upward*, which is where a token budget spends.
  Every category improves; multi-hop R@1 more than doubles.

  Two results worth carrying: `BAAI/bge-reranker-base` is 12× the parameters of the
  default `ms-marco-MiniLM-L-6-v2` and scores **lower** on every metric at 5× the
  runtime, and the dependency-free `CoverageReranker` nets **−0.1** — it is a control
  that proves the gain belongs to the model rather than the stage, not a recommendation.
  The default stays `None` because a cross-encoder is roughly 84 ms per query against a
  ~3 ms search, which is a cost decision and no longer an accuracy one.

- **`bench/locomo.py --embedder {hashing,local}` and `--rerank-model ID`.** The harness
  never pinned its vector leg: it fell through to `default_embedder()`, which prefers
  sentence-transformers when installed. Since `memvara[rerank]` *installs*
  sentence-transformers, installing the extra in order to measure the reranker also
  swapped the embedder underneath the measurement, and the whole difference landed on the
  reranker. It now defaults to `hashing` — the configuration every published number was
  produced with — and prints the embedder in its report unconditionally. Every other
  bench in that directory already pinned one explicitly.

- **A three-state read filter: `states=`.** `Memvara.search`, `get_all` and `count`, their
  `ScopedMemvara` mirrors and the `AsyncMemvara` / `AsyncScopedMemvara` ones take
  `states=`, any non-empty subset of `("live", "ended", "retired")` — `Claim.state`'s own
  three words — defaulting to `["live"]`.

  It replaces arithmetic nobody could do with a boolean. One flag over three states can
  say "live" and "all of them" and nothing else, and the population it cannot name is the
  one a correction audit is made of:

  ```python
  mem.get_all(states=["retired"])   # everything we stopped believing, and nothing that
                                    # merely stopped being true
  mem.get_all(states=["ended"])     # the other half: still believed, no longer true
  ```

  Client-side filtering was not the way out. These reads are capped, so a filter applied
  after `limit` under-returns silently: `search` over-fetches `k * candidate_multiplier`
  and ranks those, so with twelve live claims and `k=1` the window is five and the audit
  comes back empty with nothing saying it was truncated.

  `include_invalidated` remains an **exact alias** — `False` is `["live"]`, `True` is all
  three — and is not deprecated and does not warn: `filterwarnings =
  ["error::DeprecationWarning"]` would turn a warning into a failure at every existing
  call site. Passing both raises, because there is no reading of the mix in which one of
  the two is not being ignored.

  **Asking for all three states is not the union of the parts.** `Claim.state` is absolute
  while the query is as-of, so the three do not tile the store — a claim recorded but not
  yet in force at `valid_at` is named by none of them. The complete set therefore compiles
  to the belief floor alone, which readmits that row and leaves the world clock nothing to
  constrain. That is exactly what `include_invalidated=True` has always meant, and it is
  now pinned by a test rather than left to be rediscovered.

- **`memvara.store.STATES`, `ClaimState`, `resolve_states`, `state_predicate` and
  `stored_state_predicate`.** The vocabulary and the SQL that selects it.

  `resolve_states(states=None, include_invalidated=None, *, default=("live",))` is the
  single place either spelling is interpreted, so no surface can invent its own reading of
  the older flag; it returns a canonical tuple in `STATES` order, so one requested
  population compiles to one string however the caller spelled it. `default=` is both the
  unflagged view *and* what `include_invalidated=False` means on that method — one
  parameter names both, so they cannot drift apart.

  `state_predicate(at, *, states=None, alias="")` is the general form of `live_predicate`,
  and it returns something that function could only document: the **axis behind each bind
  marker**, in order.

  ```python
  state_predicate("?")                      # ('(recorded_at <= ? AND ...)',
                                            #  ('known', 'known', 'valid', 'valid'))
  state_predicate("?", states=STATES)       # ('(recorded_at <= ?)', ('known',))
  ```

  A backend therefore binds by *reading* that list rather than by remembering an order,
  which makes the one silent error unwritable — a belief instant bound onto a world column
  answers identically to a correct one on every `as_of` call, because those pass the two
  axes equal. Every subset keeps the belief markers ahead of the world markers, so the
  discipline generalises instead of changing shape per subset.

  `stored_state_predicate(states=None, *, prefix="")` is the member of the family for a
  walk with no clock to read. `iter_claims` pages over rows rather than answering a
  question about a moment, so it filters the *stored* state — what `Claim.state` reports —
  which differs from `state_predicate` exactly where a timestamp is in the future: a claim
  retired next October is `retired` here and still believed there. It returns `""` for the
  complete set, meaning "no filter", so a paged scan drops the term instead of emitting a
  tautology.

  `live_predicate` and `_live_clause` are now the two-state aliases of these and are
  neither deprecated nor changed in meaning.

- **`SQLStore._state_clause(valid_at, known_at, states=None, alias="")`** — the
  parameterised form of `state_predicate`, and the method every read filter in a SQL
  backend routes through. It is **the binding site**: the only place in this repository
  that binds the state predicate's markers, and `_live_clause` is now this with the
  two-valued alias applied.

- **`stats()["ended_claims"]`** — claims the world moved past and we still believe.

  It was the largest non-live population and the only one with no key, and it is **not
  derivable**. `claims - live_claims - invalidated` does not give it: that residual also
  contains claims in no state at all. Nor is it `valid_to IS NOT NULL`, which counts a
  claim that ended and was *later* retired a second time, that row already being inside
  `invalidated`. The state predicate excludes it from `ended_claims`, so the two
  populations are disjoint.

  On a store with one live claim, one ended, one ended-then-retired and one scheduled for
  next year — `claims=4, live_claims=1, ended_claims=1, invalidated=1` — both cheap
  spellings give 2 and both are wrong. The counts still do not sum, and the leftover is no
  longer the ended rows but the scheduled one; `claims` is the only total that covers
  everything.

### Changed

- **`Store.erase_claim` returns per-table counts instead of a `bool`.** The same four keys
  `purge` returns — `claims`, `episodes`, `embeddings`, `entities` — so the two erasure
  paths evidence themselves the same way. Of the two it was the weaker witness, and it is
  the path an erasure request naming one memory actually takes. A missing id returns **all
  zeroes rather than an absent key**, so a caller totalling an erasure campaign never
  special-cases it; `counts["claims"]` is 0 or 1 and carries what the boolean carried.

  **`Memvara.erase()` deliberately still returns `bool`.** Widening it would change a
  published v0.1.0 signature from a flag to a mapping, and every `if mem.erase(id):` in
  existence would start taking the branch unconditionally, because a dict of zeroes is
  truthy. A caller wanting the evidence calls `store.erase_claim` or `purge()`.

- **The `Store` read path takes `states` alongside a widened `include_invalidated`.**
  `candidate_ids`, `lexical_search` and `vector_search` now take
  `states: Collection[str] | None = None` and `include_invalidated: bool | None = None`;
  `None` on the flag means "not passed", and passing both raises. `iter_claims` gains
  `states` as a keyword while `include_invalidated` stays positional there, because it
  always was — and its unflagged view remains `("live", "ended")` rather than live-only.
  That default is load-bearing: `reembed()` walks this, and narrowing it would silently
  stop re-encoding every superseded version in the store.

  The episode reads take neither. Episodes are not bitemporal — nothing retires or
  supersedes them — so there is no end-of-life to lift.

- **`WriteReceipt.invalidated` → `closed`, plus `ended` and `retired`.** The field holds
  the claims a write closed out, and after the closure split that is claims on *either*
  clock: `close="ended"` puts still-believed claims in it, `close="retired"` puts
  no-longer-believed ones. One name could not say which, so every consumer had to
  re-derive it from `Claim.state` — and of the three that existed, two did not, and
  called every closure a retirement.

  ```python
  receipt.closed        # what this write closed out, either clock
  receipt.ended         # the world moved past these  — still believed
  receipt.retired       # we stopped believing these
  receipt.invalidated   # the old name, same list object, removed at 1.0.0
  ```

  `ended` and `retired` are derived from `Claim.state` rather than stored, so they cannot
  disagree with the claims themselves. The alias raises **no** `DeprecationWarning` on
  purpose: `filterwarnings = ["error::DeprecationWarning"]` in `pyproject.toml` would
  turn a warning into a failure at every existing call site, including this package's own
  write path.

- **The MCP server distinguishes the two closures.** `memory_add` and `memory_remember`
  reported `retired N` for every claim a write displaced — so a supersession, which is
  the common case and which *ends* a claim, was announced to the model as a correction.
  The same server rendered that claim as `ended` under `memory_history` and used
  "retired" correctly under `memory_forget`, leaving three names for two events on one
  transport. The summary line now carries both counts, and each displaced claim is
  listed with its own closure and timestamp:

  ```
  added 1, ended 1, retired 0, already-known 0, no-fact 0 (0 model call(s))
  + [cl_44b2c5ad486f491d9d43] user lives in Lisbon
  - [cl_047bac579e6d4ed680cc ended 2026-08-13 06:58Z] user lives in Berlin
  ```

### Fixed

- **Two more tool descriptions called one closure by the other's name.** `memory_forget`'s
  is fixed in the `memory_end` entry above; these two were still standing after it.
  `memory_remember` said an exact predicate lets the store "*retire* the previous value" —
  it **ends** it, and that sentence was on the one tool whose whole job is writing the
  replacement. `memory_history` described every past value as "*retired*", though
  `_history` renders `_state()`, which emits both words, so supersession — the common case
  — was mislabelled to every model that read it.

  This is the same bug `_receipt_summary`'s docstring was written about ("a model reading
  its own memory tool had three names for two events"), now on its third and fourth
  instance: that fix corrected the receipt line, `memory_end` corrected `memory_forget`,
  and neither swept the rest. So two guards now exist rather than a third correction —
  one asserting every handler reads every property its own schema declares, and one
  asserting no description uses a retire-word for an operation that ends or the reverse.
  Both were confirmed to fail against the pre-fix code before being kept, and the second
  one is what found these two.

- **A fact's past no longer outlives the fact under a budget.** `recall(include_history=…)`
  built its past values in a flat list that was not index-aligned to the claims, so a
  budget that dropped a note could leave that note's history rendered beneath a fact no
  longer there.

- **Superseding a claim no longer records it as an error.** `Reconciler._retire` closed
  *both* clocks when one value replaced another. `valid_to` was right — Berlin stopped
  being true when Lisbon began — but `invalidated_at` means *we no longer believe this
  record*, and the record was never wrong. Every superseded claim in every store this
  library wrote was marked as a mistake.

  The consequence was that one of the two readings the axes had just gained did not work
  at all: **"what do we now believe was true in June" returned nothing on any history the
  write path produced.**

  ```python
  mem.remember("user", "lives_in", "Berlin", valid_from=J23, recorded_at=J23)
  mem.remember("user", "lives_in", "Lisbon", valid_from=J26, recorded_at=J26)
  mem.get_all(as_of=MID)      # ['Berlin']  — worked, and hid the bug
  mem.get_all(valid_at=MID)   # []          — now ['Berlin']
  ```

  `as_of` kept answering the whole time, because it rewinds the belief clock past the
  supersession and so never reads the stamp that was wrong. `valid_at` worked only on
  stores built by calling `store.put_claim` directly — which is what
  `tests/test_bitemporal.py`'s fixture did, and its docstring said why.

  The rule now, one line: **closing valid time says the world changed; closing
  transaction time says the record was wrong, and no write asserts both.** Supersession
  and retraction are always the first kind — the reconciler is told "here is the new
  value", never "the old one was a mistake" — so they leave `invalidated_at` unset,
  keep `invalidated_by`, and produce `Claim.state == "ended"`. `retired` now means what
  its name says.

### Added

- **`AsyncMemvara.scope()` and `AsyncScopedMemvara`.** The async facade was the one of
  the three that could not bind a scope. It was documented as a deliberate omission on
  the grounds that every method there already takes the four scope keywords, so nothing
  was unreachable — which was true and beside the point. `ScopedMemvara` does not exist
  to make anything reachable; it exists so the four keywords are written once instead of
  on every line, because the call site that repeats them is the one that eventually
  writes one user's fact into another user's scope. That argument is *stronger* for the
  async facade, which exists for servers, where one handle per request per user is the
  shape and where a mistake is somebody else's data.

  ```python
  amem = AsyncMemvara(Memvara("memory.db"))
  bob = amem.scope(user="bob")          # not a coroutine: it binds four strings
  await bob.add("I live in Oslo")
  ```

  `scope()` is the only method on either async class that is not awaitable, because it
  touches no store. The view mirrors `ScopedMemvara` exactly — same methods, same
  arguments minus the scope keywords, plus `bind()`, minus `close`/`reembed`/`scope`,
  which are not scoped operations. `unscoped` reaches the `AsyncMemvara` for those (going
  through `memvara.close()` instead would put an fsync back on the loop thread).

  The suite checks the surface by introspection in both directions — the view must cover
  `AsyncMemvara` and it must cover `ScopedMemvara` — and derives what a scoped view is
  allowed to omit from the synchronous pair rather than from a hand-written list, so
  none of the four classes can gain a method that the other three quietly miss.

- **`memvara.store.live_predicate(at="?", *, include_invalidated=False, alias="")`** —
  the liveness predicate as SQL text, built without a store instance. `at` is the SQL
  *expression* for the instant and is substituted at every axis: a bind marker (`"?"`,
  `"%s"`) or a server clock (`"now()"`). It returns the parenthesised clause;
  `SQLiteStore._live_clause` is now that call plus the binding.

  It takes one expression rather than one per axis on purpose. Every caller that cannot
  reach a store is counting *now*, and two markers would let one bind the belief instant
  onto the world columns — the single error a diagonal query cannot reveal, since
  `valid_at == known_at` answers the same either way. Where markers are used the four
  bind in the order **known, known, valid, valid**, and `_live_clause` is the only place
  in this repository that performs that binding.

  A function rather than a constant because the paramstyle and the join alias both vary;
  a module-level function rather than a method on `SQLStore` because the copy that
  motivated it samples a tenant through a caller-supplied raw connection and has no
  instance to borrow a builder from.
- **`close=` on the four writes that end a claim**, so both readings are expressible and
  neither is guessed: `remember`, `supersede`, `forget` and `delete`, on `Memvara`,
  `ScopedMemvara` and `AsyncMemvara`, plus `Reconciler.apply`, `WritePipeline.assert_claim`
  and `compat._notes.write_note`. It takes the two words `Claim.state` already uses.

  `remember` and `supersede` default to `"ended"`: a new value is news about the world.
  `forget` and `delete` default to `"retired"`: forgetting is something the holder of a
  memory does, and neither call names a successor, an end date, or any evidence that
  something changed out in the world — so closing valid time would assert a world event
  on the caller's behalf. An unknown value raises rather than resolving to either, since
  the two mean opposite things about whether a stored fact was ever true.

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
- **`Store.invalidate` and `Store.set_valid_to` stay in the protocol, and the engine
  calls neither.** Every write that ends a claim now goes through `types.close_out` plus
  `put_claim`, because closing a claim moves exactly one clock and neither method can
  express that: `invalidate` writes `invalidated_at` and `invalidated_by` in one
  statement — that pairing *is* the bug above, written into a signature — and
  `set_valid_to` writes no pointer at all. They are kept because they are the protocol's
  only single-statement writes, because `set_valid_to(id, None)` **reopens** an interval
  and no write path can (`close_out` only ever moves an end earlier), and because both
  store suites use them to build fixtures. Their docstrings now say which of the two
  clocks each one stops, and a test refuses a store whose targeted writes are called
  from any engine path — the row a mistaken caller leaves behind is almost right, and
  it is the extra column that is wrong.
- **`Store.stats` states that its claim counts do not sum.** `live_claims` is the full
  predicate; a backend that "corrects" the arithmetic has reintroduced the conflation.
  (`ended_claims` joined them later in this release — see *Added*.)
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

- **A later retirement re-dated the supersession that preceded it.** `why()`'s
  `superseded` list read `invalidated_at` when the row had one, which dates a different
  event: when we stopped believing the *predecessor*, not when the successor displaced
  it. On a row carrying both closures those are months apart — Berlin ended in August
  when Lisbon replaced it, was deleted in October, and `why(Lisbon, known_at=September)`
  then reported nothing superseded. An October write silently changed what the audit
  said about September, in the method whose promise is that the trail survives, by the
  operation documented as the reversible one. The instant is now the successor's
  `recorded_at` throughout — the row that carries the moment we decided — which can only
  move a supersession's date earlier and so never drops a link a dated view used to
  show. Double-closed rows became an ordinary shape when the closures split, which is
  why this surfaced now.
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
  `docs/UPGRADING.md` carries a detection query, and what it means for an
  erasure already answered.
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
