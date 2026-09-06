# Roadmap

Where the project is, what is deliberately not being built here, and how the open core and
the commercial layer relate. Kept honest about status: an item is `done` only when
something in the tree does it, `deferred` when it was considered and declined with a
reason, and `next` when it is actually queued.

**Status as of `0.8.0`.** 3,593 tests, 100% statement coverage, mypy clean, CI on
3.10–3.13 across Linux, macOS and Windows. The library does what the README says. Phase 4
— the evidence phase that gated everything below — is **done**, which changes the shape of
this document: the organizing risk was credibility, and it is no longer that every
comparative number is self-authored.

What has changed since that line first read "nothing here has been scored end to end":
`demo/` is an authored corpus, five arms and a blinded harness, and it has been run once
with an agent as the reader. That is an apparatus and a sanity check, not a reader-model
benchmark — the run is not reproducible and cannot rank systems. See
[What is still missing](#what-is-still-missing).

---

## Phase 4 — Prove it — **done**

The only phase that changed memvara's position rather than its surface area.

### 4a. Head-to-head against the real mem0 package — done

`bench/mem0_real.py` drives the actual `mem0ai` package (2.0.17), not a reimplementation:
same 105-turn transcript, same extraction oracle, same `HashingEmbedder`, Qdrant in
`:memory:`, fully offline, five runs each. The table is in the README.

Three things came out of it that were not the expected result, and all three are in the
README because they cut against us:

- **The original benchmark was wrong in memvara's favour.** Its oracle string-matched the
  whole prompt, and mem0's additive prompt embeds `last_k_messages`, so every earlier turn
  in the window was re-extracted and mem0 was measured under a firehose. It reported 6/10;
  the true figure is 9–10/10. The mechanism is documented in the harness.
- **A README claim about mem0 was corrected against the source.** `add()` costs *one* LLM
  call in 2.x, not two — the two-call description was 1.x. The contradiction problem it
  was cited for got larger rather than smaller, but the correction is stated rather than
  quietly dropped.
- **The call-count row is partly an ingestion-granularity choice.** 105 vs 2 becomes 126 vs
  17 at equal granularity, and that is now said next to the number.

### 4b. LOCOMO and LongMemEval — done for retrieval, not for accuracy

The blocker was recorded as "an API key", and that turned out to be the wrong framing.
Scoring *retrieval* — did the store surface the evidence the annotators marked? — needs no
model at all, so `bench/locomo.py` and `bench/longmemeval.py` run the full question sets
(1,531 and 500) offline, for nothing. Results are in the README, weak rows first.

Two findings worth keeping:

- **LongMemEval's `oracle` split cannot measure evidence retrieval.** Every haystack
  session in all 500 instances *is* an evidence session, so recall there is 99.2% by
  arithmetic. The harness now computes a `chance` baseline and warns above 50%.
- **Retrieval was not reproducible until that run.** `HybridRetriever` broke score ties on
  `claim.id`, a fresh `uuid4` per ingest, so two ingests of one corpus ranked differently.
  Ties now break on a content hash and three full runs are byte-identical.

### 4c. The differentiator, measured — done

Both of the above measure *retrieval*, which is the commodity half. `bench/temporal.py`
measures the other one: six families — point-in-time, delayed knowledge, `as_of` audit,
contradiction, correction, source authority — over 48 authored scenarios, scored as exact
set matches, with no model and no network, byte-identical on every run.

It was built because the moat had no number at all, and it earned its place immediately:
against `origin/main` at `7b91a9a` it scored `source_authority` at **50.0%** — a
0.10-confidence guess displaced a 1.00-confidence statement in every scenario — and
reported 8 `ended` claims that answer at no instant with 0 of them named by the write path.
Both are defects that had lived on the write path while 3,448 tests passed. Figures, the
baseline column and the anti-flattery constraints are in `docs/BENCHMARKS.md`.

The suite is wired into `tests/test_bench_eval.py` as a gate rather than left as a script,
because everything it scores is a promise the library makes in prose and prose does not go
red.

**The half that is still open** is end-to-end judged accuracy on this project's own
harness, which does need a reader model and does cost money. The harness already reports a `none` / `memory` / `full`
triple when a reader is configured, because a memory score with no reader-only floor and no
whole-haystack ceiling beside it is uninterpretable. On this harness it remains the single most
valuable remaining item in the repository; a judged number does exist for the hosted read
path, on LongMemEval-S through the MemoryBench harness
([`docs/BENCHMARKS.md`](BENCHMARKS.md#answer-accuracy-judged-in-the-memorybench-harness)),
but not on this apparatus. `demo/` built the apparatus and the corpus for it (see
[What is still missing](#what-is-still-missing) and
[`demo/README.md`](../demo/README.md)), but a run with an agent in the reader's seat is a
sanity check, not the measurement.

Exercising that path end to end offline — `--reader stub` and `--reader file` over both
suites — found four defects in code that had never run, all now fixed:

- **`--embedder` and `--rerank` were wired only into `--score retrieval`.** `run()` called
  `build_memory(sample, budget, llm)` with neither, so the answer path silently used
  `default_embedder()` — sentence-transformers wherever it happened to be installed — and
  ignored `--rerank` outright while the banner said it had been applied. Two runs differing
  only in those flags were byte-identical, which is how it was found; with the fix they
  differ on 4 of 8 and 5 of 12 answers respectively.
- **`bench/longmemeval.py` pinned no embedder at all** and had no flag to. So a LOCOMO
  figure and a LongMemEval figure quoted in one paragraph were produced by different
  vector legs. `--embedder` now lives in `evalkit.add_common_arguments` and both runners
  print what they used.
- **`stop_reason` was never read.** A `max_tokens` truncation or a `refusal` arrives as a
  short or empty string, scores 0.0, and is averaged into a figure presented as answer
  quality — so a budget that was too small reads as a memory layer that surfaced bad
  evidence. Now counted and printed. Relatedly, `AnthropicReader` sent no `thinking` and
  `max_tokens=1024`; on `claude-opus-5` thinking is on by default and shares that budget.
- **The cost estimate was ~2× high on LOCOMO's input side and ignored thinking on the
  output side.** Corrected against measured prompt sizes; the procedure and the numbers now
  live in exactly one place, `bench/evalkit.py`'s module docstring.

What a hosted run still cannot do is resume: `run()` issues its calls one at a time with no
concurrency and no checkpoint, so LOCOMO is a 2–3 hour foreground process that writes
nothing until it finishes. Slice it with `--shuffle SEED --limit N`, or use the
`--reader file` dump/answers round trip, which is a resume mechanism that already exists.

---

### 4d. A benchmark other systems can run — done, and memvara does not win it outright

Everything else in Phase 4 measures memvara against something we wrote or something we
chose. `benchmarks/agent_memory/` is the first measurement here that a *different* memory
system can run on the same dataset, by the same rules, without forking this repository:
262 events, 100 questions, 16 scenarios, a five-method adapter interface, a versioned
dataset, a published result schema, and `--system` accepting a dotted import path. Offline
and deterministic — `--repeat-check` runs a system twice and fails on a single differing
verdict.

The result is the reason it earns a line here rather than a bullet in the README:

| | memvara | vector-rag | naive |
|---|---:|---:|---:|
| overall | **92.0%** | 89.0% | 50.0% |
| temporal | **100.0%** | 91.5% | 34.0% |
| retrieval | 64.3% | **71.4%** | 64.3% |
| irrelevance | 50.0% | 50.0% | 50.0% |

**Three points separate memvara from a baseline built out of numpy**, and the whole of the
lead is the `temporal` dimension — where the four questions that separate them are the four
delayed-knowledge and same-instant-correction scenarios. That is the narrow, specific claim
the two clocks support, and it is narrower than the pitch: everywhere the two clocks
coincide, a single-clock store is exactly right. memvara **loses** `retrieval`, and
`irrelevance` is a three-way tie. Both are reported in the tables rather than a footnote,
and the core weaknesses behind them are tracked in
[#129](https://github.com/memvara/memvara/issues/129).

The `irrelevance` half has a core answer now, and the table above does not show it because
the shipped adapter does not use it: `search(anchored=True)` keeps only the rows the
question names an entity of, or that the graph leg reached from one, and through the
adapter with that flag set `irrelevance` goes 3/6 → 5/6 at both configurations. The
`multi_hop` half does not move from anything in the read path — the issue's own
measurement is that the answer is already retrieved and nothing consumes the pair — so
that row is the adapter's to change, and `docs/BENCHMARKS.md` carries both measurements.

`docs/benchmarks/agent-memory-benchmark.md` is the public report and
`benchmarks/agent_memory/README.md` the methodology. What this does **not** close is
[4b](#4b-locomo-and-longmemeval-done-for-retrieval-not-for-accuracy)'s open half: every
system here is handed a structured fact, so extraction is held constant and nothing
measures answers.

---

## Phase 5 — Keep the promises the package already makes — **done, one item excepted**

| gap | status |
|---|---|
| `memvara[openai]` extra with no adapter | **done** — `memvara/llm/openai.py`, Chat Completions with `strict: true`, refusals handled explicitly |
| Python floor unverified | **done** — CI runs 3.10–3.13 on Linux, 3.13 on macOS and Windows |
| No CI | **done** — `.github/workflows/ci.yml`: matrix, a coverage job gated at 100%, a mypy job, and a no-extras job that imports every module |
| No `LICENSE` file | **done** — Apache-2.0 |
| No `CHANGELOG.md` | **done** |
| Hosted embedders (OpenAI, Voyage) | **deferred** — see below |

**Windows support** belongs on this list retroactively and was the largest single item in
it. It was listed in CI and had never run there; the first run reported 99 failures. 95 of
them were one missing function — `os.pread`/`os.pwrite` are POSIX-only, so every store with
a `.vecs` sidecar raised `AttributeError`. The other notable one was a date before 1970: the
timestamp clamp was hard-coded to the POSIX year-9999 limit, while Windows' CRT stops at
year 3001 *and* rejects negative timestamps, so the exact defect the clamp exists to
prevent — one accepted write permanently breaking every later read of its scope — was alive
at both ends. Both bounds are now probed from the C library. A suite at 100% coverage on
three platforms never saw it, which is the useful lesson.

---

## Phase 6 — Deployment surface — **split**

This phase is where the open/commercial line actually falls, so it no longer reads as one
list.

**Shipped in the open core:**

- **Docker image** — multi-stage `python:3.13-slim`, 292 MB unpacked / 63.2 MB pulled, of
  which 1.4 MB is memvara and the rest is the base image and numpy. Runs as uid 10001 with
  a read-only root filesystem and `--cap-drop=ALL`. `docs/DEPLOY.md` has the whole story,
  including why there is no `EXPOSE` and no `HEALTHCHECK`.
- **Framework adapters** — LangChain, LlamaIndex, CrewAI, and LangGraph. Each says in its
  own module docstring what it *loses*, because "works with LangChain" without that is a
  claim that quietly means four different things. LangGraph turned out to be the best fit
  of the four: `BaseStore` hands over the query text natively *and* `put(namespace, key,
  value)` supplies all three parts of a triple, so an item is stored as one claim per field
  and changing `city` ends exactly `city` — contradiction resolution surviving a foreign
  interface intact.
- **Multi-hop traversal** — `neighborhood()`, `paths_between()`, `Store.adjacent()`, and
  SQLite schema v6 with `subject_key`/`object_key` indexes. Not on the original roadmap at
  all, and it is the feature that made the "the store has been a graph all along" claim
  true rather than rhetorical. Every edge on a path is evaluated at one pinned `as_of`,
  which is the property a search-then-search loop cannot have.
- **The graph leg of retrieval** — `GraphTraverser.spread()`, `retrieve/spread.py`, and a
  third leg in `HybridRetriever` seeded from the head of the fused vector+lexical list
  (Zep's φ_bfs). It closes the gap between what `neighborhood()` can answer and what
  `search()` can: the caller no longer has to know the seed entity. **It ships at
  `w_graph=0.0`** — the measured table is in `docs/BENCHMARKS.md`, and the short version
  is that what the leg is worth depends on how much graph the store holds, and the two
  numbers point opposite ways. On 2WikiMultihopQA, where 26,403 claims load through
  `remember()` with no extractor running, it takes chained questions from 28.3% to 43.8%
  at k=12. On LOCOMO and LongMemEval the offline write path extracts almost nothing —
  0 claims and 78 — and there the leg is inert on the first and a small loss on the
  second, 92.2% to 90.6% on single-session-user. A default cannot be right for both.
- **The temporal leg of retrieval** — `Store.episodes_near()`, `retrieve/temporal.py`, and
  a fourth leg over raw turns ranked on proximity to the instant the search was asked
  about. Time was a filter and a multiplier on the read path and never a candidate
  producer, so "what was going on around then" — whose only content words the analyzer
  drops — had no leg that could answer it. **It ships at `w_temporal=0.0`.** The measured
  finding is the abstention rather than the leg: without one it cost 2.4 points of
  LongMemEval temporal-reasoning R@12, because a query with no instant anchors on *now*,
  an archival corpus scores every turn at ~0.005, and fusion reads positions. With the
  guard the other two legs already had, the loss goes to zero — and so does the gain, on an
  instrument that never passes `valid_at`.
- **Query-intent gating** — `retrieve/intent.py`, deterministic and model-free, four
  classes matching the categories LOCOMO reports separately. It is what makes the graph
  leg affordable to switch on: `lookup` and `temporal` queries skip the walk before the
  traverser is called. Every multiplier that is not a gate is 1.0 and stays 1.0 until a
  per-category sweep moves it.

  **Its relational vocabulary is a hand-written list and it is too narrow, measured.** On
  `bench/multihop.py` the gate routes two of the three question families past the walk —
  "who founded the company that X works at" contains no word in it — so the shipped
  configuration scores exactly what plain `search` scores and the leg's whole gain is
  gated away. `works at` and `founded` are relations by any reading and both are
  predicates in the store's own registry, so **deriving the markers from the registry is
  the fix**. Not done, deliberately: widening the list by hand against a benchmark this
  repository wrote is how a classifier gets fitted to its own corpus.
- **Token accounting** — `WriteReceipt.tokens_in`/`tokens_out`, `LLM.Usage` with a
  caller-allocated accumulator, and the `write.tokens_in` / `write.tokens_out` /
  `write.extract_ms` series. `llm_calls` was the only cost signal and cannot be billed on:
  a one-line turn and a 40,000-token document are both one call.

**Moved to the commercial layer, and not planned here:**

- **REST API and auth.** The `http` extra stays declared and reserved in `pyproject.toml`
  and nothing in this repository will implement it.
- **Postgres / pgvector store.** This is the clean commercial boundary: SQLite is genuinely
  sufficient for a single node, and needing more than one node correlates almost exactly
  with willingness to pay. The `Store` protocol is public and documented precisely so this
  is implementable by anyone who wants to; the license permits it and so does the design.

Saying "planned" about either of those would be the dishonest version, and the README now
states the boundary in the same terms.

---

## Phase 7 — Governance — **the seams are open, the policy is not**

The deletion half was already built and remains open source, which is unusual in this
category and worth saying out loud: `erase()` and `purge()` are real, irreversible removal
of the claim, the FTS entries that store the tokens directly, the vectors (zeroed in place,
because an embedding leaks content under inversion), the entity rows that keep the first
spelling ever seen, and optionally the source turns. Both return per-table counts as
evidence. Retirement that leaves the text on disk is the normal behaviour in this category
and does not satisfy an erasure request.

| item | status |
|---|---|
| **Redaction seam** — one injectable hook, upstream of the hash, the store, the embedder and the model | **done, open** (`memvara/redact.py`) |
| **`Recorder` seam** — every silent failure mode has a live series | **done, open** (`memvara/telemetry.py`) |
| **`Store.erase_episode`** — the primitive a retention rule needs; reaches a turn no claim cites | **done, open** |
| PII ruleset, compliance mode, per-role policy, audit report | **commercial** |
| Tamper-evident hash-chained audit log | **commercial** |
| Retention policies on a schedule | **commercial** |
| RBAC / SSO | **commercial** |
| Encryption at rest | **deferred** — see below |

The dividing rule, stated once because it decides every future case: **a seam is worth
nothing to a competitor and everything to a deployment; a policy is the opposite.** A
library you have to fork in order to comply is worse than one that ships no policy at all,
so the extension point and one honest default live here, and the ruleset does not. The
built-in `PatternRedactor` says in its own docstring that it is not compliance-grade,
because it is a demonstration of the seam rather than a product.

---

## Phase 8 — Release — **done except the publish**

- **`CHANGELOG.md`** — done, and kept specific: "a backdated supersession left two live
  values for a single-valued predicate" is the entry someone searches for.
- **A version policy** — done, in `docs/RELEASING.md`. `0.x` means the protocols may change
  in a minor release; `1.0` will mean exactly one thing, that everything behind
  `Memvara(store=, embedder=, llm=)` is a contract we will not break in a minor version.
  **Open question before 1.0:** `Recorder` and `Redactor` are injectable extension points on
  the same terms and are not currently named in that promise. Either they are in it or the
  promise says why not; leaving it ambiguous is the one outcome to avoid, because a closed
  layer and a third-party backend both build against them.
- **The name is settled.** The project was `engram` until Phase 8 prep found that
  `pip install engram` already resolved to an unrelated MIT rendering/vision library — so
  the name was not merely unregistered, it was pointing at someone else's code, and
  `twine upload` would have been rejected. `engram` was a weak mark for a second reason: it
  is the standard neuroscience term for a memory trace, which makes it *descriptive* of the
  product's own function, the hardest class to register or defend. `memvara` is coined,
  means nothing in any language, and is therefore a **fanciful mark** — the strongest class.
- **The registry names are claimed.** An organization is still not a reservation — that
  is why the first uploads had to happen — but they did: PyPI `memvara` 0.1.0 on
  2026-08-14, npm `memvara` 0.0.1 the same day. An npm org still only reserves
  `@memvara/*`. The npm package was a placeholder until `0.1.0`, which made it a
  CLI — see below.
- **npm trusted publisher** — registered 2026-08-25. `release-npm.yml` packs,
  hashes and publishes over OIDC with no stored token and no reviewer wait, on its
  own `npm-v*` tag rather than the Python one.
- **Community files** — `CONTRIBUTING.md`, `SECURITY.md` and issue templates are in place,
  and the README states the open-core boundary rather than leaving a reader to infer it
  from a pricing page.

---

## The cross-encoder reranker — landed, opt-in, and worth +4.5 R@12 / +14.4 R@1

This sat under "deliberately deferred" until now, on the argument that a reranker is a
model and the default configuration is "numpy and nothing else, offline, no API key". That
argument was right about the constraint and wrong about the conclusion: the constraint is
satisfiable. `memvara.rerank` ships a `Reranker` protocol, a `NullReranker`, a
dependency-free `CoverageReranker`, and a `CrossEncoderReranker` behind
`pip install 'memvara[rerank]'`. `HybridRetriever(reranker=None)` is the default and
`None` means the stage does not exist — no import, no model, no network. A subprocess test
asserts exactly that, and it goes red if the backend is made an eager import.

The stage runs **after** fusion, the recency half-lives and the diversity cap, reorders
the top `rerank_top_n` and cuts to `k` afterwards — so it can promote a candidate fusion
left outside the caller's `k`. Every candidate it scores carries the number in
`Explanation.rerank_score`; everything past `top_n` keeps `None`, which is the accurate
record that the reranker never saw it.

**What was measured**, on the LOCOMO retrieval harness that produced the R@12 = 62.0 in
the README, all 1,531 evidence-labelled questions, **vector leg pinned to
`--embedder hashing`** so the only thing varying is the reranker:

| configuration | R@1 | R@5 | **R@12** | R@20 | MRR | run |
|---|---:|---:|---:|---:|---:|---:|
| no reranker (shipped default) | 30.5 | 51.7 | **62.0** | 67.4 | 44.9 | 10s |
| `NullReranker`, `top_n=20` | 30.5 | 51.7 | **62.0** | 67.4 | 44.9 | 12s |
| `CoverageReranker`, `top_n=20` | 31.5 | 50.5 | **61.9** | 67.4 | 45.0 | 12s |
| `CoverageReranker`, `top_n=50` | 31.6 | 49.8 | **59.7** | 66.2 | 44.5 | 12s |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 44.9 | 62.1 | **66.5** | 67.4 | 59.2 | 2m19s |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | 46.0 | 62.9 | **66.6** | 67.4 | 60.2 | 4m07s |
| `BAAI/bge-reranker-base` | 42.8 | 60.3 | **65.9** | 67.4 | 56.9 | 12m33s |

```bash
PYTHONPATH=. python3 bench/locomo.py --score retrieval --recall-at 1,5,12,20
pip install 'memvara[rerank]'
PYTHONPATH=. python3 bench/locomo.py --score retrieval --recall-at 1,5,12,20 \
    --rerank 20 --reranker cross-encoder
# the other rows of the table: --rerank-model picks which cross-encoder
PYTHONPATH=. python3 bench/locomo.py --score retrieval --recall-at 1,5,12,20 \
    --rerank 20 --reranker cross-encoder --rerank-model BAAI/bge-reranker-base
```

Every run above pins `--embedder hashing`, which is now that flag's default. That pin is
the reason these rows are comparable at all: `memvara[rerank]` installs
sentence-transformers, and before the flag existed `bench/locomo.py` fell through to
`default_embedder()` — so installing the extra in order to measure the reranker also
swapped the embedder underneath the measurement, and the whole difference landed on the
reranker. `bench/longmemeval.py` had no such flag at all. Both runners now print the
embedder they used, unconditionally.

**Read `NullReranker` first.** It reproduces the baseline in every cell, which is what
makes every row below it attributable to the reranker rather than to the plumbing.
**Then check R@20**: it is 67.4 in all seven rows. Reranking the top 20 can only reorder
within the top 20, so recall *at* 20 cannot move — and it does not, in any configuration.
That invariant is the strongest self-check this harness has, and it holds.

**The lever is real, and it is the model.** `CoverageReranker` — lexical, no download —
nets **−0.1**, robbing single-hop to pay multi-hop. A cross-encoder over the identical
stage, the identical candidates, is **+4.5 on R@12 and +14.4 on R@1**. An earlier revision
of this section concluded from the lexical row alone that reranking "has not been shown to
move" the number. That conclusion was wrong: it measured a keyword-overlap heuristic and
generalised it to a class of model it never ran.

**R@12 understates it.** Reranking cannot find evidence the retriever missed — R@20 proves
that — so its whole effect is pulling the right evidence *upward*. That shows up at the
top of the list, which is the part a caller actually puts in a prompt:

| category | n | R@1 base → CE | R@12 base → CE | MRR base → CE |
|---|---:|---:|---:|---:|
| multi-hop | 279 | 7.4 → **16.2** | 36.0 → **42.2** | 31.6 → **51.5** |
| temporal | 320 | 41.5 → **56.3** | 71.0 → **75.9** | 54.0 → **66.2** |
| open-domain | 92 | 13.9 → **15.3** | 30.7 → **32.5** | 24.7 → **27.5** |
| single-hop | 840 | 35.7 → **53.4** | 70.7 → **74.8** | 48.1 → **62.5** |
| **all** | **1531** | 30.5 → **44.9** | 62.0 → **66.5** | 44.9 → **59.2** |

**Every category improves.** That is the qualitative difference from the lexical row,
which could only redistribute: multi-hop R@1 more than doubles and its MRR goes up
twenty points.

**Bigger is not better here.** `bge-reranker-base` is 278M parameters against
MiniLM-L-6's 22M, and it is worse on every metric at **5× the runtime**. L-12 buys +0.1
R@12 and +1.1 R@1 for 1.8× the time. `ms-marco-MiniLM-L-6-v2` stays `DEFAULT_MODEL`: it
takes essentially all of the available gain at the lowest cost.

**What it costs.** Roughly **84 ms per query** at `top_n=20` for L-6, against a ~3 ms
search — so with reranking on, the reranker *is* the query latency, by more than an order
of magnitude. (L-12 ≈ 155 ms, `bge-reranker-base` ≈ 485 ms. CPU only, no GPU, on a
contended developer machine; treat them as ratios rather than absolutes.) The stage itself
is still ~3 µs and `CoverageReranker` ~250 µs. This cost is the reason the default stays
`None` — not the accuracy question, which is now settled.

**Still not measured for this specific comparison: whether reranking on versus off
reaches an answer.** These are retrieval numbers, and no reader has been run over either
configuration here, so "+14.4 R@1" is a claim about evidence placement and not about
answer accuracy. A reader has since been run elsewhere in the read path — 0.11.0's ranked
recall, judged on LongMemEval-S, is in
[`docs/BENCHMARKS.md`](BENCHMARKS.md#answer-accuracy-judged-in-the-memorybench-harness) —
but that run does not isolate the reranker the way this section's numbers do, so it does
not close this specific gap.

---

## Deliberately deferred

**A memvara-paid allowance on ranked reads.** `search(ranked=True)` (`memvara.select`)
puts one per-query model call on the *customer's* key, so memvara's own cost per call is
$0 and nothing here is metered. Phase 2 — memvara fronting the call on its own key, with
an allowance sized from usage — is deferred to thirty days of the `retrieval.model_query`
and `retrieval.model_fallback` series a phase-1 deployment emits; see
`docs/superpowers/specs/2026-09-04-model-ranked-recall-design.md` for the full design and
what it watches for. Before this the per-query model call the roadmap's pricing section
discusses below was entirely hypothetical; it is not any more, and the cost on the
customer's own key is now measured, in
[`docs/BENCHMARKS.md`](BENCHMARKS.md#answer-accuracy-judged-in-the-memorybench-harness).
What memvara would charge for a phase-2 fronted allowance is a separate decision and is
still not made.

**Persisting derived relation terms.** `retrieve/compose.acquire()` pays one model call
per vocabulary and the answer lives for the life of a `Memvara`, so a long-running server
pays once at startup and a short script pays once per run. Making it "asked once, ever" —
the standard `resolve_predicate` already meets — needs somewhere to put the terms, and
both candidates are wrong. `put_spec` stores predicates and a derived term is not one:
recording `grandfather` as a predicate would give it a cardinality and a decay half-life
it has no business having, and `all_specs()` would then offer it to the extractor as a
slot to write into. A new `Store` method is a protocol change, and #26 demonstrated what
those cost — three members added there broke `mypy` in a downstream repository whose CI
was switched off, and nothing went red.

The honest shape is probably a small tenant-scoped key/value on the store, which is a
surface this protocol does not have and should not grow for one feature. Until then the
cost is one call per process, which is the same order as loading an embedding model.


**`decision` and `observation` as memory types.** The agent-state brief asks for eight
memory kinds where this library has three and names these two as the pair missing for agent
work. They ship as a predicate pack instead — `MEMVARA_PREDICATES=decisions` — and the
reasoning is worth keeping, because "add an enum member" reads like a smaller change than
it is.

**`MemoryType` is a one-way door, every time.** It is persisted as its string value and
hydrated with `MemoryType(value)`, which raises on a name it does not know. So a fourth
member makes a store that an older build can no longer open, permanently — a
`SCHEMA_VERSION` bump of the same kind `docs/UPGRADING.md` already records for the FTS5
option. That cost recurs for the fifth member and the sixth.

**And the one rule a new member would have to enter does not generalise.** Three things
read `memory_type` to decide something: `consolidate.merge.promote_pass`, the
`memory_types=` filter on `recall()` and `search()`, and `memory_standing`, which returns
the live claims whose type is `PROCEDURAL` and nothing else. Everything else that touches
it renders it or stores it. Decay is `Volatility`, not this. The filter takes any member
for free and `memory_standing` would go on ignoring a fourth one, so the only *rule* a new
member has to enter is a hardcoded `EPISODIC -> SEMANTIC` promotion — a pairwise machine
that a further value cannot join without becoming a policy table nothing currently needs,
or a documented exemption that reads as an oversight later.

**`observation` is the wrong axis outright.** "An agent noticed this" rather than "a person
said it" is a statement about where a claim came from, and that axis exists: `Derivation`,
which every write path sets and `why()` reports. Putting it on `MemoryType` would record
the same fact in two columns that nothing keeps in step.

What was actually missing was vocabulary, and the evidence for that is in this file: the
join-rate measurement below found **95% of a production store's claims on predicates
outside the declared set** — `known_defect`, `deploy_gotcha`, `version`, `rejected` — so
the store was already writing decisions and observations and getting the unregistered
default for them. `packs/decisions.toml` declares the two that vocabulary genuinely lacked,
with the two `engineering` already had left alone so load order cannot decide the answer.

This is the reversible choice, which is the argument that settles it. A pack can be
corrected, extended or dropped; a schema bump cannot. If `memory_types=["decision"]` turns
out to be the filter people actually reach for, nothing here forecloses adding the member
then, with usage behind it rather than a taxonomy.

Each of these was considered and declined for a reason. They are recorded here so they stop
reading as things that are coming.

**Encryption at rest.** SQLCipher works — measured at +43–48% on writes, search unchanged,
and FTS5 keeps working because page-level encryption sits *beneath* SQLite. It is not
shipped because the mmap-backed `.vecs` sidecar stays plaintext outside that boundary, and
a plaintext vector is a confirmation oracle: encoding a guess and taking the cosine against
that file returns exactly 1.0000 for the right text and 0.87 for a one-digit-different
phone number, so it is not merely confirmable, it is hill-climbable. Encrypting the text and
not the vectors would be theatre. Full-disk encryption is the honest answer for the open
core; a storage-layer answer belongs with the backend that has one.

**Database-enforced row-level security.** Scope isolation here is enforced in the query
layer — `Scope.sees` for reads, `Scope.ancestors()` in SQL for enumeration — and it fails
closed. SQLite has no row-level security to enforce it a second time, so defence in depth
at the database layer is not available in this repository at all, and pretending it is
queued would misrepresent where the boundary is. The query-layer rule is the one that has to
be right, which is why it is named in `SECURITY.md` as an in-scope surface.

**Scrubbing on-disk residue after erasure.** `erase()` and `purge()` delete rows and index
entries and zero the vector slots. They do not scrub the SQLite pages those rows occupied,
and the `-wal` may still hold them. `VACUUM` and `PRAGMA secure_delete` are the deployment's
levers and `docs/DEPLOY.md` says so. Doing it in the library would mean either a
`secure_delete` pragma that taxes every write in the store or a `VACUUM` that rewrites the
whole file inside what a caller thinks is a per-claim call — both are the deployment's
trade to make, not ours.

**Hosted embedders (OpenAI, Voyage).** `memvara[local-embed]` ships `LocalEmbedder` and the
`Embedder` protocol is two members wide — `dim`, and `encode(texts) -> (n, dim)` — so a
hosted one is a small amount of code that mostly duplicates an SDK call. It has stayed
undone because nothing in the library needs it and every user who wants one can write it in
an afternoon. That is a weak reason and this is the item most likely to move back onto the
list; a contributed implementation would be accepted.

**Making a personal or project store joinable.** The graph leg ships at `w_graph=0.0`
because it does not pay for itself on the corpora in `docs/BENCHMARKS.md`. What was not
written down is why a *real* store looks the same way, and the reason is not the one that
suggests itself.

Measured on a production store of 387 live claims, join rate **0.5%** — two claims lead to
another. The tempting explanation is the extraction rules, which require rich self-contained
prose objects (`the object IS the memory`, `MIN_RICH_OBJECT_CHARS`): a paragraph can never be
another claim's subject, so rich objects and joinable objects look like opposites. **That is
not what is happening here**, and acting on it would reopen guidance that has nothing to do
with the problem.

The numbers say something else. **95% of that store's claims use a predicate outside the
declared vocabulary** — `known_defect`, `deploy_gotcha`, `version`, `rejected`, invented at
write time — because `remember()` bypasses predicate registration, as `CLAUDE.md` says. Only
18 claims use a declared predicate at all, and **exactly one** sits on an entity-valued one.
So the least invasive fix, emitting a short entity object alongside the prose for predicates
that have an entity, has a ceiling of one claim. It is not a trade-off worth weighing.

What the store is instead is an annotated log — `<component> <observation> <description>` —
and the shape is honest rather than deficient. 32% of its claims *mention* another claim's
subject somewhere in their prose, which looks like latent linkage and is not: those are
references inside a description, not the value of the relation, and promoting one to the
object position would misstate the claim it came from.

Two things worth knowing before anyone measures this again. Subject spellings fragment —
`memvara_cloud` (80 claims) and `memvara-cloud` (17) — and it is cosmetic: both normalise to
`subject_key='memvara cloud'`, so the store already counts them as one entity. And
`UnjoinedStoreWarning` already names this exact case and says where it is fixed: "in the
write path, by storing facts whose subject is not the one hub everything hangs off". That
remains true. It is a description of what such a store would have to become, not a defect in
this one.

**An approximate vector index (HNSW/IVF).** Exact search over a scope is O(|scope| · d) and
the matmul is already BLAS, so this is the floor, and it is correct and fast to roughly a
million claims. Beating it trades recall for speed, which belongs behind the `Store`
protocol as a choice a deployment makes, not in the default path.

**Hook code shared from here rather than owned by one plugin.** ~~Deferred.~~ **Reversed,
and done** — `plugin/hooks/` is in this repository and vendored by the plugins that use it.

The entry that stood here argued the opposite, and it is left described rather than
deleted because the argument was right when it was written and the thing that changed it
is worth naming. It said the hook code was Claude-specific *in fact* — it parsed one
editor's JSONL, `type: user`/`assistant`, `message.content` blocks — and that there was no
second consumer, so the abstraction would be paid for by anticipation rather than
evidence. It also named the price: a second vendored tree is a second lock, a second drift
test, and a sync PR for every hook change.

Every one of those held until six hosts were measured. What changed the answer is the
condition the entry itself set: a second client grew hooks. `opencode-memvara` ships them,
`codex-memvara` ships them too, and the measuring turned up that no two hosts agree on
anything that matters — three different reply envelopes (nested, flat, and flat
snake_case), three different transcript formats, and three different answers to whether an
async hook runs at all. Seven copies of that would have been seven places to get each
difference wrong.

The price was paid as quoted. `hooks.lock` is the second lock, the drift and freshness
guards are the second drift test, and re-vendoring is a PR per plugin per hook change.

What is here is deliberately only the half that is not an editor's format. Each host is a
`Host` record — event names, field names, reply keys, timeouts, an `ApproveSpec`, an
`ExtractorSpec` — and the four hook bodies read the record rather than the client. Where a
host genuinely differs in kind rather than in spelling, that is code and it says so:
`core/envelope.py` renders two shapes, `lib/transcript.py` reads two, and `js/` exists
because OpenCode cannot run a shell hook at all.

**What is still Claude-specific, and stays so.** `lib/transcript.py` still speaks Claude
Code's entry shape as its default, and the formatter still writes `Claude: ` in front of an
assistant line whatever host produced it. Neither is load-bearing for extraction, and
neither is worth a rename that would re-label history in users' stores.

## A JavaScript client, and what was built instead

Recorded here because this list is where *considered* belongs, and until 2026-08-25 a JS
client appeared in neither section — it had not been declined, it had never been decided.
The npm package asked readers to open an issue if they wanted one, which made the
decision wait on a demand signal nobody was collecting.

**Declined: a JavaScript engine.** A second implementation would have to re-derive the
invariants in `INTERNALS.md` identically, and getting one wrong is not hypothetical — the
paper below conflates the two clocks on supersession and measures its own time-travel
retrieval scoring *worse* than plain search as a result.

**Reversed on 2026-08-29, and built: a Python client for a hosted deployment.** This was
declined — *"MCP already covers the agent case… A library would serve calling memvara from
ordinary application code, and nobody has asked for that"* — and absence of demand was the
only reason given. This project's maintainer then asked, naming two callers: application
code in Python, which none of the three hosted surfaces serves, and `memvara-mcp --mode
cloud`, which refuses to start for want of exactly this client. What shipped is
`Memvara(api_key=...)`, the library's own API served by a deployment's `/v1` facade with
the engine still running server-side, plus `Memvara.connect()` for the credentials
`memvara-mcp login` leaves behind. It is not `RemoteStore` completed and does not bring
that closer: the `Store` protocol and the REST facade stay diverged for `OPEN-CORE.md`'s
reasons, and this takes the other seam that document had already named. The shape worth
noticing is the one this entry shares with the JS client above it — a decision resting on a
demand signal nobody was collecting stays decided until somebody asks out loud.

**Built: `npx memvara`**, a stdio-to-HTTP MCP bridge to the hosted server, zero runtime
dependencies. Its value is narrower than it looks and the narrowness is worth stating:
`app.memvara.dev/mcp` advertises standard MCP OAuth, so any client that speaks to remote
servers needs no bridge at all. This is for clients that only spawn a command over stdio,
and its one advantage over the generic `mcp-remote` is that it needs no configuration and
reuses an existing `~/.memvara/credentials.json` without opening a browser.

## What is still missing

Stated plainly, because a roadmap that only lists what is done is an advertisement.

1. **End-to-end answer quality has an apparatus and one non-reproducible run, and no
   benchmark.** `demo/` closed the first half of this: an authored support corpus, twenty
   questions with authored golds, five arms from a no-context floor to a whole-transcript
   ceiling, and a blinded dump/answer harness. What it does not have is a reader behind an
   API. The one run used an agent as the reader, so there is no model id, no seed and no
   temperature to quote, and it cannot be repeated. Two things it did establish are worth
   carrying: **at this corpus size the whole-transcript arm scored 100%**, so the memory
   layer's argument here is 5.6× fewer tokens rather than a better answer; and the trap
   metric — the column a before/after claim would rest on — produced **no signal at all**,
   because the reader never gave a superseded value. Still missing on *this* corpus: a
   hosted reader behind an API, a second corpus size to turn the token argument from a
   slope into a measurement, and any comparison against mem0 on answers rather than on
   architecture. A hosted reader has since been run on a different corpus — 0.11.0's
   ranked recall, judged on LongMemEval-S through the MemoryBench harness, in
   [`docs/BENCHMARKS.md`](BENCHMARKS.md#answer-accuracy-judged-in-the-memorybench-harness)
   — but it says nothing about this authored support scenario, which still needs its own
   run.

   What did land is the *guarded* half: `python3 demo/harness.py --reader stub` runs all
   five arms end to end in one offline, deterministic process, and
   `test_the_offline_run_is_identical_twice` pins it. That makes the apparatus runnable in
   CI and a change to it bisectable. It does **not** move this item, and the distinction is
   the point: a stub reader picks the retrieved line with the most words in common with the
   question, so its accuracy column measures the corpus and the arms and nothing about
   answers.
2. **Nobody outside this repository has reproduced the Agent Memory Benchmark**, and two
   of its seven dimensions do not yet discriminate. Every number in
   [4d](#4d-a-benchmark-other-systems-can-run-done-and-memvara-does-not-win-it-outright)
   was measured by the people who wrote both the benchmark and one of the systems in it —
   which is the objection the whole design tries to answer and cannot answer alone. The
   answer is somebody else's adapter, and the contributor guide exists for that.

   `irrelevance` is a three-way tie at 50%, and `multi_hop` is 16.7% / 33.3% / 33.3%: no
   system tested abstains when it knows nothing, and none joins two facts. A dimension all
   three systems fail identically measures the field rather than the systems, and until
   either the questions get harder or a system improves, those two rows carry no
   information. memvara can now abstain — `search(anchored=True)` — and the published
   row will move once the adapter asks it to; `multi_hop` is still nobody's.
3. **No external user has run this in production.** 3,593 tests prove the code does what we
   said it does. They prove nothing about what happens on someone else's data.
4. **The English-centrism is measured, not fixed.** The salience gate and the fast extractor
   are English sentence forms; other scripts fall through to the model, which is correct
   behaviour and a real cost. `gate.drop` and `fast.miss` are tagged by script so the gap is
   visible rather than assumed — but visible is not closed.
5. **Entity resolution folds surface forms, it does not know the world.** `Stark` versus
   `Stark Industries` is genuinely ambiguous and is left that way. So is the other
   direction — one surface form that has been two different things — and it is worse,
   because the fold is confident: two people who share a name are one entity, and on a
   single-valued predicate the later one's employment retires the earlier one's,
   manufacturing a job change that `history()` reports as a timeline and `why()` explains
   with a supersession pointer.

   **Nothing in the data can detect it**, which is why there is no warning for it and why
   there should not be. The gap does not separate the cases: `works_at` is
   `Volatility.SLOW`, a two-year half-life, so eight years between employers is four
   half-lives and an entirely ordinary job change. Neither does provenance, confidence or
   predicate. The distinction is knowledge the store does not have, and a signal that
   fired on every long-gap supersession would be noise that teaches a reader to ignore the
   notes that do mean something.

   What ships is the repair rather than the detector: `split_entity` records what a person
   knows, dated, dry-run by default, undoing the closures that crossed the boundary and
   leaving retirements alone. It is the inverse of `learn_alias` and the mirror of
   `backfill_entities`.

---

## Related work

Phase 4 existed because every comparative number here was self-authored. Two external papers
bear directly on this design, and both are recorded for the same reason: they are evidence
nobody in this repository wrote. The first appeared after `v0.1.0` and reached this project's
thesis on its own; the second predates it and is the closest published system on the
retrieval side.

**"A Graph-Native Bitemporal Memory Store for Conversational AI Agents"** — Alp Niksarli and
Gopesh Baheti, Davidson College, [arXiv:2607.26520](https://arxiv.org/abs/2607.26520)
[cs.DB], 29 July 2026. A Neo4j property graph with HNSW vector indexes, memory identity
nodes separated from versioned content nodes, two closed-open intervals on each version, and
an evaluation on LongMemEval. It is an unreviewed five-page preprint scoring 60 of the 500
questions, ten per category, so a single question moves any cell by ten points. None of what
follows rests on its scores.

**It is this project's argument, reached independently.** The paper's motivation is that
context-stuffing does not scale and that third-party memory services — it names Mem0 and Zep
— take custody of a record of everything the user said, so the store belongs next to the
agent. Its answer is the same one: valid time for when a fact held in the world, transaction
time for when the database believed it, closed-open intervals, nothing physically
overwritten. Its related-work section states outright that a full two-axis temporal model has
not previously been applied to a vector-indexed store for agent memory. That claim is theirs,
not ours, and it is the first external statement of the gap this library was built into.

**It is also a published instance of the conflation `INTERNALS.md` invariant 3 forbids, with
the regression measured.** Per its §III.B, `update_memory` closes **transaction time only**
and never sets `valid_to` on the superseded version, while `delete_memory` closes **both
clocks at once**. Memvara does neither: a supersession sets `valid_to` and `invalidated_by`
and leaves `invalidated_at` unset, because the old value stopped being true and we were not
mistaken about it. The consequence in their schema is that every historical version of an
updated fact stays valid-time-open forever, so the post-filter `valid_to IS NULL OR valid_to
> $valid_at` admits all of them and the valid-time axis cannot discriminate on exactly the
facts it was built for. Their §V-E reports the symptom without connecting it to the cause:
time-travel retrieval scores **worse** than current-state search on temporal reasoning, 3/8
against 5/8 on the same questions, which they attribute to over-fetch dilution and propose to
fix by re-ranking on proximity to `valid_at`. Their retrieval also over-fetches `10 × k` from
the full-history index and applies the temporal filter *afterwards*, which is the shape
invariant 7 forbids — a filter and a limit in different layers — and is why `states=` is a
`Store` parameter here rather than a comprehension in the facade.

That reading is taken from five pages of prose rather than from their code, and it should be
held that loosely. What supports it is their own result: a correctly closed valid interval
could not have made the time-travel path lose to current-state search on the category that
tests it.

**Two things in it are better than what is here.** It builds `RELATED_TO` edges at write time
from the top five neighbours above cosine 0.75, one extra ANN lookup per write, giving a
related-memories operation that needs no extraction at all; memvara's `GraphTraverser` walks
predicate edges, which is more precise and cannot exist until something has been extracted.
And its future work proposes counter nodes maintained during ingestion for questions that
count events across sessions — the class this repository's multi-session row at 65.5% R@12
cannot reach by retrieval, and which the reranker above does not reach either, because
counting is an aggregation over the evidence rather than an ordering of it.

**Its numbers are not comparable to the tables above and should not be quoted beside them.**
It clears the database per example and scores a hit as 50% token overlap with the ground-truth
*answer*; the LongMemEval table above uses one shared 940-session store so distractors exist,
and scores recall of annotator-marked evidence. Two of its weak rows are also structural
rather than informative: single-session-assistant at 20% because it stores only user turns by
policy, so the assistant's own text is never written down at all, and single-session-preference
at 10% for the same metric artifact documented above, where the golds are multi-sentence
summaries no single turn can contain. The first of those is worth separating from a choice
that looks like it. `SalienceGate.DEFAULT_EVIDENCE_ROLES` is `{"user"}` here too, but it
governs *extraction* and runs in tier 1 — every episode is already stored in tier 0
regardless of role, which is why the assistant row in the table above is 100.0% R@12 rather
than a structural zero. Declining to mine a turn for claims and declining to keep it are
different decisions, and only one of them is recoverable later.

**"Hindsight is 20/20: Building Agent Memory that Retains, Recalls, and Reflects"** — Chris
Latimer, Nicoló Boschi, Andrew Neeser, Chris Bartholomew, Gaurav Srivastava, Xuan Wang and
Naren Ramakrishnan; Vectorize.io, The Washington Post and Virginia Tech;
[arXiv:2512.12818](https://arxiv.org/abs/2512.12818) [cs.CL], 14 December 2025. Code at
[github.com/vectorize-io/hindsight](https://github.com/vectorize-io/hindsight), per-question
results at [hindsight-benchmarks.vercel.app](https://hindsight-benchmarks.vercel.app/). It
predates this project's first release by eight months; it was read on 2026-09-06 and every
statement below about this repository was checked against the tree at `151d994`.

What it is: memory organised into four networks (world facts, the agent's own experiences,
opinions carrying a confidence score, and synthesised entity summaries) and three operations
(retain, recall, reflect). Retain is narrative extraction by a model: two to five
self-contained facts per exchange, each carrying what, when, where, who and *why*, with
coreference resolved and relative times normalised to an occurrence interval plus a mention
time; then entity resolution and four kinds of link built at write time (temporal, decaying
with distance; semantic, cosine above a threshold; shared entity; and causal, extracted by
the model). Recall runs four channels in parallel (vectors, BM25, spreading activation seeded
from the top vector hits, and a temporal channel that parses the query into a date range and
keeps facts whose interval overlaps it), fuses them with RRF at k=60, reranks with
`ms-marco-MiniLM-L-6-v2`, and packs the result to a token budget. Reflect is a persona layer;
it ran at a neutral setting for every reported number and has no bearing on a memory store.

Its scores, on LongMemEval-S over all 500 questions: 83.6% with GPT-OSS-20B doing both the
extraction and the answering, 89.0% with OSS-120B, and 91.4% with OSS-120B building the
memory and Gemini-3 Pro writing the answer. The same 20B model handed the whole transcript
scores 39.0%. On LoCoMo the three configurations score 83.18, 85.67 and 89.61.

**Read those scores with four things in mind.** There is no ablation anywhere in the paper,
so nothing isolates which channel, the reranker, or the narrative extraction produced the
gain; every attribution of a gain to a component is the authors' reading. The retrieval token
budgets are literally `<add>` in its §7.3. The rows in its tables were judged by different
judges: its own rows by GPT-OSS-120B, the Supermemory and Zep rows copied from Supermemory's
report under a GPT-4o judge, and the Backboard row self-reported. And its Table 3 has no
abstention row, although the benchmark holds 30 abstention questions; the paper does not
say what became of them. Its judged numbers sit in the same range as this repository's one
judged run (177/199 shipped, 182/199 with the larger selector, in `docs/BENCHMARKS.md`), and
the two are not comparable: a different judge, a different sample and a different reader.

**Most of its retrieval stack is already here.** BM25 and vectors fused by RRF at k=60
(`retrieve/fusion.py`); the same cross-encoder, opt-in, measured above at +4.5 R@12; a graph
leg seeded from the head of the fused list (`retrieve/spread.py`), shipped at `w_graph=0.0`; a
temporal leg (`retrieve/temporal.py`), shipped at `w_temporal=0.0`; `recall(budget=)`; and a
deterministic write-side date resolver (`write/when.py`). On the temporal model this library
is ahead: their memory unit carries an occurrence interval and one mention time and no clock
for when a record stopped being believed, so it cannot tell `ended` from `retired`. And the
opinion trajectory the paper illustrates, a belief held at 0.70, reinforced to 0.85, then
rewritten at 0.55, is exactly what `history()` records here. Hindsight adjusts the confidence
in place and keeps only the formation time, so it loses the trajectory it draws.

**What is worth measuring here, in order.** None of these is queued. Each names the row it
would move and the instrument that would show it.

1. **Render the resolved time into the indexed text.** Hindsight prefixes each fact with a
   human-readable time before embedding it and includes the same string in the reranker's
   input. Here nothing that ranks can see time: `Claim.render()` is subject, predicate and
   object; the episode FTS index holds `content` alone; `rerank/stage.py` scores
   `item.text`. So BM25 cannot match "June 2024" in a question to a turn dated then, and the
   cross-encoder is asked whether an undated passage answers a dated question. The change is
   a write-time rendering of `valid_from` at the precision `when.resolve` returned, with no
   model on the read path, so invariants 1 and 7 hold. The row is temporal-reasoning: 66.6
   R@12 in the retrieval table and 47/53 judged. Instrument: `bench/longmemeval.py --score
   retrieval --share-store`, with and without the reranker.

2. **Give the temporal leg an anchor read off the question.** The leg ships at zero because
   no benchmark passes `valid_at`: `bench/evalkit.py` calls `search(question, k)`, and
   `bench/longmemeval.py` puts the question date into the reader's prompt and nowhere else,
   so the anchor is always *now* and the abstention floor fires on every archival turn.
   Hindsight's temporal channel parses the query into a range and keeps facts whose
   occurrence interval overlaps it, with a small seq2seq model as the fallback. The fallback
   is out (invariant 1). The rule-based half already exists: `write/when.py` is the one place
   in the library allowed to say what "last month" means, and it takes an anchor.
   `retrieve/temporal.py`'s docstring refuses a read-path date parser because it would be a
   second extractor with its own locale bugs; reusing the first one, gated on
   `intent.classify()` returning `temporal`, is not that. Two things come first, measured on
   2026-09-06 against the anchor 2023-06-01: `resolve` handles "last month", "yesterday",
   "three weeks ago" and "in 2019", and returns `None` for "last weekend", "in March", "June
   2024" and for any whole sentence, so it needs a phrase locator and those three forms
   before it can read a question. And the match should be interval overlap at the precision
   it returns, not the proximity to a point that the leg ranks on today.

3. **Ask the extractor for the why.** Hindsight's extraction prompt demands motivations,
   preferences and emotional context on every fact, and resolves "my roommate" and "Emily"
   to one entity. Its largest per-category gain with the same 20B model is
   single-session-preference, 20.0 to 66.7. `EXTRACT_SYSTEM` in `llm/base.py` asks for a
   triple with `when`, `amount` and `unit` and says nothing about motivation or coreference,
   and preference is this repository's weakest judged row, 7/12, for the reason the retrieval
   table gives: the golds are meta-descriptions no single turn contains, and only extraction
   can write a claim that does. The instrument is the judged MemoryBench run in
   `docs/BENCHMARKS.md`; a retrieval R@k cannot see this row move. Since #178 a self-hosted
   deployment can replace the instructions, so the change can be tried without touching the
   default.

4. **A same-judge comparison, which the Supermemory attempt could not get.** The stack is
   open source. Running it through the MemoryBench harness on the 199-question seed
   `20260903` sample with `gpt-5.4` as reader and judge puts a like-for-like number beside
   the 177/199 above, and running memvara on the full 500 puts one beside their 83.6, 89.0
   and 91.4. The paper's own tables mix judges; this is how to stop doing that.

5. **Entity summaries as a consolidation pass.** Their observation network is one
   model-written summary per entity, regenerated in the background when a fact about the
   entity changes, so that "tell me about Alice" costs one note rather than twelve. That is
   not the `observation` memory type the deferred list above declines, which was a provenance
   axis in the wrong column. It is a derived claim: `Consolidator` would gain a fourth pass
   that takes a model, writes the summary with `Derivation.CONSOLIDATION` and `sources`
   naming the claims it summarised, and ends it when any source ends. Off the write path, so
   invariant 1 holds; useful only to a deployment that has a model, which on 2026-09-06 the
   hosted one did not.

6. **Write-time edges that need no extraction.** Semantic links above a cosine threshold and
   temporal links that decay with distance, built as facts are stored, are what let
   Hindsight's graph channel run over narrative facts. Here the graph leg walks predicate
   edges and is inert on LOCOMO by construction, 0 claims from 5,882 turns. This is the
   second system after the Davidson paper above to build such edges, and neither ablates
   them, so it is a hypothesis with two examples rather than a finding.

**Not worth borrowing.** Confidence adjusted in place, for the reason above. The persona
parameters, which are a prompt on the reader and belong to the caller, and which at their
neutral setting contributed nothing to the paper's numbers. A seq2seq date parser on the
read path. And the four networks as `MemoryType` members, which the deferred list above
already declines as a one-way door.

**One thing it says about a bet already placed.** The 83.6% is a 20B open model doing the
extraction as well as the answering, which is evidence that the extraction layer rather than
the frontier model carries most of the result. #179 queues extraction on a self-hosted small
model on the same reasoning. The paper does not isolate extraction quality from anything
else, and 20B is not 4B, so it supports the direction and says nothing about the size.

---

# The commercial layer

## Why the split exists, and why the core is permissive

The core library is Apache-2.0 and stays that way. Every surface built around it — REST API
and auth, the Postgres/pgvector store, governance, the multi-tenant control plane, usage
metering, quotas, rate limiting and the hosted console — is closed and lives in a separate
private repository that depends on `memvara` as a published package. **The split has
happened**; it was sequenced to follow Phase 4 and Phase 4 is done.

The framing that keeps this honest, and the one the README uses: **the library is the
product and the commercial layer is the operations around it.** Nothing on the closed side
changes what a claim is, how a contradiction resolves, what `why()` returns or what
`search()` finds. That is not a marketing line, it is a constraint on what may be built
there — a paid layer that altered the semantics of the free one would make the free one
untrustworthy, which costs more than it could ever earn.

### Why not a protective license

Apache-2.0 permits our closed layer and everyone else's. A funded competitor can take the
core and ship the exact product we intend to sell, and the license will not stop them. That
risk is **accepted deliberately**, because the usual remedy is worse here.

AGPL plus a commercial dual license is what MongoDB and Elastic did, and it works for them
because they ship **servers** — the copyleft boundary is a socket. Memvara is a **library**
imported into someone's agent process, where AGPL arguably reaches the whole application. In
practice nobody `pip install`s an AGPL memory layer into a commercial product. That would
close the embedding path, and with it the migration wedge that makes the mem0 shim the most
commercially valuable thing built so far. Protection bought at the cost of the adoption
funnel is not protection.

BSL 1.1 protects better and is not OSI open source, which conflicts with the core being
genuinely open.

So the moat is the closed layer, the brand, and execution speed — not the license.

### The line

| open (Apache-2.0) | closed (private repo) |
|---|---|
| the library: store, retrieval, write path, bitemporal model | REST API and auth |
| MCP server — a thin adapter that drives adoption | multi-tenant control plane, hosted console |
| **mem0 shim and importer** — the wedge; closing it removes the reason to migrate | usage metering, quotas, rate limiting |
| **`erase()` / `purge()`** — real deletion; an Article 17 obligation is not an upsell | governance: policy, retention, audit chain, RBAC |
| `Recorder` and `Redactor` protocols (the seams) | the dashboard and the rulesets consuming them |
| SQLite store | **Postgres / pgvector store** |
| `Store` / `Embedder` / `LLM` protocols | — |

Four of these are deliberate and worth defending. The **mem0 importer stays open** because
it is worthless without the core and is the only reason anyone switches — putting it behind
a paywall would mean charging for the exit. **Erasure stays open** for the same class of
reason: a deletion guarantee sold separately is not a guarantee. **Postgres goes closed**
because it is a clean commercial boundary. And **the seams stay open while the policies do
not**, which is the rule stated in Phase 7.

### What is actually differentiated

Not the retrieval, and not the cost savings — those are a quantifiable nice-to-have that any
funded competitor can copy in a quarter.

**The bitemporal audit trail is the asset.** `why()`, `history()`, `as_of` and deterministic
contradiction resolution answer a question no vector store can:

> *What did this agent believe on March 3rd, where did that belief come from, and what
> replaced it?*

That is an **audit requirement** in every industry currently too scared to deploy agents:
healthcare, finance, insurance, legal, and anything touching the EU AI Act's logging
obligations or a GDPR Article 17 erasure request. So the strategy is not "better mem0":

> **The memory layer you move to when someone starts asking what your agent knew.**

Migration is free (the shim and the importer, both open), and the reason to migrate is a
question the incumbent cannot answer.

### Protections that do the work the license doesn't

- **CLA on the open core**, in place before the first outside contribution. Without it,
  every external patch is a veto on ever relicensing. `CONTRIBUTING.md` states the
  requirement and is honest about why.
- **Trademark.** `memvara` is a coined, fanciful mark — the strongest class — and that is
  the thing that stops someone selling a competing "Memvara Cloud", not the license. (An
  earlier version of this document said the opposite, calling `memvara` descriptive and weak.
  That was a leftover from the `engram` era and it was simply wrong: `engram` was the
  descriptive neuroscience term, and replacing it is most of why the rename happened.)
- **Nothing from the closed side ever enters this repository.** Not "moved later" — never
  committed. `git filter-repo` can remove a file from a public repository's tip; it cannot
  remove it from every clone, fork and archive that already fetched it. A commit-then-revert
  is a publication.

## The model

**Free, Apache-2.0, forever:** everything in this repository. Adoption is the moat for an
infrastructure library, and crippling the core to force upgrades is how you lose to the
thing that didn't.

**Commercial, per-deployment:** governance — PII policy, retention, tamper-evident audit
export, RBAC/SSO — and support with an SLA. These are exactly what a compliance officer
signs off on and an individual developer never wants.

**Hosted, usage-based:** the standard managed offering on the Postgres backend, priced per
memory stored and per query. The larger revenue line eventually and the *weaker* strategic
position, because it is where the funded competitors already are. It follows the governance
tier rather than leading it.

### Why not the alternatives

- **Open-core with a crippled library.** Kills the adoption that is the only asset a new
  entrant has, and invites a fork.
- **AGPL + commercial dual license.** Maximum capture, but AGPL on the *core* scares off
  exactly the enterprise legal departments this strategy targets. If dual licensing is
  wanted, put AGPL on the governance layer and leave the core Apache-2.0.
- **Charging a share of measured LLM savings.** Elegant, and unsellable: metering it
  credibly requires trust we have not earned, and it prices the product against a number the
  customer can dispute every month.

## The risk worth stating plainly

The compliance market is slow, procurement-heavy, and reference-driven. It rewards a product
with three named customers and punishes one with none. Phase 4 removed the "compared to
what?" objection, but it did not produce a customer, and the item under
[What is still missing](#what-is-still-missing) about no external user having run this in
production is the one that matters commercially: nobody outside this repository has run it
on real data.

If a faster route to first revenue matters more than the defensible position, the honest
alternative is to chase developer adoption first — adapters, hosting, distribution — and
accept competing head-on with better-funded incumbents. That is a real trade, and it is a
business decision rather than a technical one: it belongs to whoever is funding the runway,
not to the architecture.

---

Previous: [Internals](INTERNALS.md) · Next: [Open core](OPEN-CORE.md) · [Contributing](../CONTRIBUTING.md)
