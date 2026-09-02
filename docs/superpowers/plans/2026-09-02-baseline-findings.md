# What the first MemoryBench baseline told us

**Measured 2026-09-02.** memvara as a provider in
[MemoryBench](https://github.com/supermemoryai/memorybench) on LongMemEval-S, 500
questions, `gpt-5.4` as both reader and judge, core `18e3626`, memvara-cloud `fe17d5f`,
`MEMVARA_LLM=none`. Run id `memvara-baseline-18e3626`; results under
`local/memorybench/` in the main checkout. This is the evidence base for step 2 of
`2026-09-02-benchmark-leadership-master-plan.md`, and it reorders that plan.

## The number

**54.4% judged accuracy (272/500)**, success rate 100%, 764 context tokens per question,
search p50 281 ms. Retrieval: hit@8 88.6%, MRR 0.79, NDCG 0.81.

| category | n | correct | accuracy | retrieval hit | MRR |
|---|---:|---:|---:|---:|---:|
| single-session-user | 70 | 64 | 91.4% | 97% | 0.85 |
| single-session-assistant | 56 | 49 | 87.5% | 100% | 0.96 |
| knowledge-update | 78 | 49 | 62.8% | 94% | 0.84 |
| single-session-preference | 30 | 16 | 53.3% | 87% | 0.76 |
| temporal-reasoning | 133 | 59 | 44.4% | 80% | 0.70 |
| multi-session | 133 | 35 | **26.3%** | 85% | 0.76 |

For scale, third-party *judged accuracy* measurements put Supermemory at 81.6%, Zep at
71–75% and Mem0 at 66.9%. Those vendors' own 94–95% headlines are retrieval metrics and
are not this number.

## Six findings

**1. Retrieval is not the bottleneck. Synthesis is.** Retrieval put the evidence in front
of the reader 88.6% of the time and the reader answered correctly 54.4% of the time. The
34-point gap opens *after* the right material is in the context. This is the single most
important thing the run says, and it was not what we expected going in: the work that
started this programme was a candidate-window fix (#155, PR #159), and the window is not
where the points are.

**2. Two categories decide the benchmark.** temporal-reasoning and multi-session are 133
questions each — 53% of the set — and score 44.4% and 26.3%. Together they yield 94 of
266. Everything else totals 234 questions already scoring 53–91%. Taking those two rows to
70% would move the overall from 54.4% to roughly 73%; nothing else available comes close.

**3. Most failures are refusals, not errors.** Every failure carries a judge explanation;
all 228 were read.

| category | failures | reader abstained | reader answered wrongly |
|---|---:|---:|---:|
| temporal-reasoning | 74 | **30 (41%)** | 44 |
| multi-session | 98 | 21 (21%) | 77 |
| knowledge-update | 29 | 3 (10%) | 26 |
| single-session-preference | 14 | 0 | 14 |

"The response says it cannot determine which trip came first." "The response says the
amount is unknown." The reader is refusing on questions whose evidence was retrieved.

**4. Multi-session failures are aggregation failures.** The wrong answers there are
overwhelmingly sums and counts: *the correct total is $65, the response found only the $15
car wash*; *four festivals, the response says it cannot determine*; *the correct answer is
four properties viewed*. These need every relevant turn, not a sample. memvara returned a
**median of 6 results** per multi-session question (max 18) against a request for 30,
because `HybridRetriever.max_episodes` defaults to 3 and is not exposed on the REST wire.
A question that asks "how much in total" cannot be answered from three turns.

**5. Our own thesis is untested.** knowledge-update scored 62.8%, above the overall
average, while the store held **1,444 claims against 211,240 episodes** — 146 raw turns per
extracted claim. memvara's bitemporal supersession machinery was almost entirely inactive,
so dated raw turns earned that score. We cannot yet claim the claim layer is what makes
knowledge-update work, because it barely ran.

**6. The pre-set decision rules earned their keep.** The 20-question smoke scored
knowledge-update 0/2 and, read on its own, said "an ingest model is mandatory". The full
run says 62.8%, which says it is not. The thresholds were fixed in writing before the
result existed (`.superpowers/sdd/.../progress.md`), which is the only reason that
reversal was visible rather than absorbed into a story about the smoke.

## What this changes in the master plan

The plan had step 2 (read path) leading with window and scoring work, and step 3 (an
ingest model) as the biggest lever. Both need reordering.

- **The candidate window drops down the list.** Finding 1 says the pool is not what is
  costing us. PR #159's floor remains a real fix for a real defect and its own
  measurement stands; it is simply not the lever here.
- **The episode cap rises to the top.** Finding 4 gives it a mechanism, a category, and a
  cheap test (below).
- **The abstention rate becomes its own work item.** Finding 3: 30 temporal failures are
  refusals on retrieved evidence. That is prompt and threshold work, not retrieval work.
- **The ingest model stays the biggest structural lever but is no longer proven
  necessary.** Finding 5 cuts both ways: the claim layer is inactive, and the benchmark
  still scored 62.8% on the category that layer exists for. Step 3 should follow the two
  cheap tests, not precede them.

## The episode cap test

The next experiment, stated before it is run.

**Hypothesis.** Multi-session accuracy is limited by the number of raw turns reaching the
reader, not by which turns are found. `max_episodes` caps episodes at 3 regardless of `k`.

**Change.** Raise the episode cap on the local benchmark stack. It is a constructor
argument (`read_max_episodes`) that memvara-cloud does not pass and the REST API does not
expose, so the test sets it where the API builds its `Memvara` and rebuilds the image. Two
arms: cap 3 (the baseline, already measured) and cap 15.

**Method, and why it is cheap.** The store is already ingested and the container tags are
`<questionId>-<dataSourceRunId>`. A checkpoint whose `dataSourceRunId` still points at
`memvara-baseline-18e3626`, with search, answer and evaluate reset to pending, re-runs
against the same data with no re-ingest. The harness does this for its web UI
(`src/orchestrator/checkpoint.ts`, the copy path) but exposes no CLI flag, so the test
writes the copied checkpoint directly. Cost is the answer and judge calls only.

**What each outcome means, fixed in advance.**

- Multi-session rises by **10 points or more**: the cap is the dominant constraint. Make
  the episode budget configurable through the REST API and choose it by measurement; this
  becomes the first item of step 2.
- It rises by **3 to 10 points**: the cap contributes but is not sufficient. Keep the
  change, and go after the aggregation prompt next, since the failures name sums and counts.
- It moves by **less than 3 points**: the cap is not the constraint. Drop this line
  entirely and spend the next measurement on the aggregation prompt and the abstention
  threshold, which finding 3 supports independently.
- Any category **regresses by more than 3 points**: more turns are crowding out claims or
  diluting the context. Report it, keep the baseline configuration, and treat context
  composition rather than context size as the step-2 problem.

**Validity caveat.** More episodes means more context tokens. The token count must be
reported beside the accuracy in every arm — a gain bought by tripling the context is a
different trade than a gain bought for free, and Supermemory's ~720 tokens is the bar
we are also competing on.

## Caveats on the baseline itself

- The local stack ran with the rate-limit rules in `memvara-cloud/memvara_cloud/ratelimit/policy.py`
  multiplied by 1000 to let the ingest finish. Rate limits govern write throughput only
  and cannot change what retrieval returns. That patch is uncommitted and must be reverted.
- The gateway blocks `gpt-4o`, so no published competitor row shares this reader or judge.
  A comparison against Supermemory, Mem0 and Zep requires running them ourselves in this
  harness under one judge, which is step 5.
- The 30 unanswerable (`*_abs`) questions are folded into their source categories rather
  than reported as a row, so abstention accuracy is not yet separable. Scoring it is a
  small piece of work worth doing before the next measurement.
