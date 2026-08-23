# Changelog

Notable changes per release. Dates are the commit date, not the upload date: `0.1.0` is
tagged `v0.1.0` and reached PyPI on 2026-08-14.

The format is loosely [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
versions follow [semantic versioning](https://semver.org/) once `1.0.0` ships. Before
then, the `Store`, `Embedder` and `LLM` protocols may change in a minor release.

## [Unreleased]

### Fixed

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
