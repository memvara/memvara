# Master plan: first on the agent-memory benchmarks

**Status:** the programme as agreed on 2026-09-02. Each numbered step below gets its own
spec and implementation plan; step 1's are written. This document is the picture the
steps are drawn from, and the place a later session checks before deciding anything the
steps do not cover.

## The goal, stated so it can be checked

Memvara scores higher than Supermemory, Mem0 and Zep on **LLM-judged answer accuracy on
LongMemEval-S inside MemoryBench**, Supermemory's own open-source harness, with the same
reader and the same judge (GPT-4o) for every provider, and the number is reproducible by
anyone with the fork and the keys. Second, the same shape of result on LoCoMo, which the
harness also runs. Third, the hosted service, not only the engine, is what produced the
published number.

What "beat them at their game" does not mean: matching Supermemory's Recall@15 figure,
which is a retrieval metric with an unpublished aggregation rule; or improving memvara's
own retrieval tables, which compare to nobody. Those stay as gates for iteration, not as
the claim.

## Where everyone stands, as of 2026-09-02

| system | what it publishes | number | notes |
|---|---|---|---|
| Supermemory | Recall@15 "with aggregation", LongMemEval-S | 95% (@5 86%, @10 91%) | model at ingest rewrites sessions into atomic, dated memories; ~720 context tokens; shown beside Zep's and full-context's *judged accuracy* |
| Mem0 | judged accuracy, LongMemEval | 94.4% | single-pass extraction, entity linking, hybrid retrieval; ~7k tokens per call; also 92.5% LoCoMo |
| Zep | judged accuracy, LongMemEval-S, GPT-4o | 71.2% | from its own paper; full-context baseline 60.2% |
| memvara | retrieval R@12, LongMemEval **oracle** file, shared store | 70.4% | hashing embedder, no model, episode retrieval only; not comparable to any row above |

Memvara's own retrieval tables say where it is weak before any redesign: temporal
reasoning 66.6, multi-session 65.5, preference 23.3, abstention 1.7 (retrieval R@12,
oracle file). Single-session rows are already 92 to 100.

## What is already known about the read path

Investigated on 2026-09-02 while fixing #155, and load-bearing for step 2:

- The final ordering is not the fused rank. Each candidate gets an absolute evidence
  score, the average of its raw cosine and its saturated BM25, scaled down by up to a
  third for staleness, confidence and salience. **A leg that ran but did not return the
  candidate within the window contributes zero.** The window is therefore a hidden score
  threshold as well as a membership cut, and no constant sets both correctly.
- A candidate floor of 50 recovers the hook's lost probe on the real store (+5 points
  hit@4 on 40 probes) and costs about 5 points of R@5 on 2WikiMultihopQA at k=4 and 5;
  floor 25 recovers the probe at rank 4 with no measurable 2Wiki cost at k=4. The
  displacement the floor covers is a score band `[c, 1.5c]` whose population grows with
  the store, so any constant is a measured value for one store size (#160).
- PR #159 is a draft carrying that measurement. It does not merge as a constant; step 2
  replaces it.

## The steps

Each step ends with the harness number moving, or a written reason it did not.

### 1. Baseline harness (spec and plan written)

A `memvara` provider in a fork of MemoryBench over memvara's REST API; memvara-cloud's
compose stack on this machine with the core built from a clean checkout; one judged run
of memvara exactly as shipped. Output: the baseline row and the per-type table in
`docs/BENCHMARKS.md`, and the list of failure shapes from `show-failures` that seeds
step 2.

### 2. Read path: scores that do not depend on the window

Goal: a candidate's score depends only on the candidate and the query, never on how
deep a list was cut or how big the store is; the window becomes a pure recall knob set
generously. Then the things every leader has and memvara ships off: a reranker over the
top of the fused pool, a token-budgeted context assembly that puts dates in front of the
reader, `k` chosen by question intent (aggregation questions need more), and a relevance
threshold that lets the reader abstain. Gates: the harness (accuracy per type), 2Wiki at
k=3,4,5, the 40-probe hosted suite, and the agent-memory benchmark at 92%. Closes #155
and #160 together and retires the floor.

Candidate mechanisms, to be measured rather than argued: completing the missing leg's
score for every candidate (cosine is cheap, the embeddings are loaded; BM25 needs the
store to score named claims); ordering by fused rank and thresholding by absolute score
instead of using one number for both; a band-reading re-gather when a leg came back full
with its tail still within `1/span` of its head.

### 3. Ingest: a model by default, the fast path as fallback

Decided: one model call per session becomes memvara's default write behaviour, hosted
and local, when a model is configured; the deterministic fast path remains and is what
runs with no key. The call produces self-contained memories with references resolved and
dates made absolute, stored as claims with the session date as `recorded_at`, the
resolved event date as `valid_from`, and the source turns as provenance. Knowledge
updates then fall out of supersession, which memvara already does and Supermemory
imitates with "updates" links. Preferences are tagged procedural. Everything the model
writes carries the same provenance and bitemporal fields as a fast-path claim, so the
audit and time-travel properties that are memvara's differentiator hold for it.
Cheap model at ingest (the full benchmark ingests about 20,000 sessions), GPT-4o only
as reader and judge. Gate: the harness, per type, especially knowledge-update and
single-session-preference.

### 4. Category passes

Whatever the per-type table still shows short after 2 and 3. Expected: temporal
reasoning (render both clocks, turn the temporal leg back on with its floor, resolve
relative dates at ingest), multi-session (intent-driven `k`, slot deduplication that
`max_per_slot` already provides, aggregation prompts), preference (procedural claims
surfaced and the prompt told to apply them), abstention (threshold plus prompt).
Then LoCoMo in the same harness.

### 5. Hosted

A benchmark tenant on app.memvara.dev with its own allowance, the branch deployed by
the provision script's `release` stage, the same run against the hosted URL, and the
published comparison table: memvara, Supermemory, Mem0 and Zep, one judge, one harness,
with their rows produced by us and their keys, not copied from their pages. The upstream
pull request to MemoryBench adds the provider so the result is reproducible from their
repository.

## Rules the steps inherit

- Every change is measured on the harness before it is described as an improvement,
  and a change that helps one benchmark and hurts another is reported with both numbers.
  The floor at 50 was rejected once, shipped once without seeing the rejection, and
  caught only because the store was checked. The check is not optional.
- Model-free operation stays a supported mode. "Works offline with no key" becomes
  "works offline, better with a key", never "needs a key".
- No AI attribution in anything that reaches GitHub, in any of the three repositories.
- Results live in the main checkout's `local/`, the docs carry every input a number
  depends on, and the docs ship in the same commit as the code.
- The hosted run is authorised per run, never taken on a session's own judgment.

## What each step needs from the owner

Step 1: an OpenAI key, Bun, the dataset download, the fork under the memvara
organisation. Step 3: a decision on the ingest model and its key. Step 5: the benchmark
tenant, the deploy, and keys for Supermemory, Mem0 and Zep if their rows are to be run
rather than cited.
