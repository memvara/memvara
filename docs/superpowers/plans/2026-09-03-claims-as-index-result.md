# Claims as an index: measured, and it does not help

**Date:** 2026-09-03
**Sample:** 199 questions, stratified proportionally to LongMemEval-S's own category mix,
seed `20260903`. Baseline accuracy on it is 54.8% against 54.4% for the full 500. Powered
to detect roughly 9 points.
**Method:** both arms at `max_episodes=15`, reusing the same ingest, differing only in
`claims_as_index`. The live configuration was probed through the API before each arm rather
than read from the source.

## Result

| arm | accuracy | median ctx | turns/question | claims/question |
| --- | ---: | ---: | ---: | ---: |
| cap 15, claims as results | **86.4%** | 4,089 | 15.0 | 3.3 |
| cap 15, claims as index | 83.9% | 4,167 | **18.1** | 0.0 |

Paired across the 199: the index arm wins 2, the control wins 7, exact p = 0.180. The
difference is inside the noise, so the honest reading is **no improvement**, not harm.

What makes the null informative is the middle column. **The index arm handed the model 3.1
more turns per question and scored no better.** Every earlier arm moved with turn count —
2.2 turns at 41.8%, 6.1 at 61.0%, 10.0 at 70.5%, and 15.0 at 86.4% here. This is the first
arm where more turns did not help, and the only thing different about them is that they were
chosen by claim provenance rather than by the retrievers.

Per category, with no arm meaningfully ahead:

| type | control | index |
| --- | ---: | ---: |
| single-session-assistant | 100.0% | 100.0% |
| single-session-user | 96.4% | 96.4% |
| knowledge-update | 96.8% | 90.3% |
| temporal-reasoning | 83.0% | 81.1% |
| multi-session | 75.5% | 69.8% |
| single-session-preference | 75.0% | 83.3% |

## What the claim layer is worth on this benchmark

Three routes have now been measured, and none of them pays:

1. **Claims as results.** Removing them entirely changed nothing — a turns-only arm scored
   46.2% against 46.2% with them in.
2. **Better claims.** Event times and quantities, built and verified writing real rows, made
   no measurable difference.
3. **Claims as an index.** This arm: more turns, no gain.

What does pay is retrieving more raw conversation and letting the retrievers rank it: cap 3
to cap 15 is 54.8% to 86.4% on this sample.

## The number that was wrong all day

**memvara at cap 15 scores 86.4%, not 70.5%.** The 70.5% figure quoted throughout this work
came from the two hardest categories alone — multi-session and temporal reasoning — and was
being compared against Supermemory's full-mix 95%.

On the same mix they report, the gap is **8.6 points, not 25**, and three of six categories
are already at parity:

| type | memvara cap 15 | Supermemory |
| --- | ---: | ---: |
| single-session-assistant | 100.0% | 100% |
| knowledge-update | 96.8% | 99% |
| single-session-user | 96.4% | 97% |
| temporal-reasoning | 83.0% | 91% |
| multi-session | 75.5% | 93% |
| single-session-preference | 75.0% | 90% |
| **overall** | **86.4%** | **95%** |

The remaining gap is concentrated in multi-session (−17.5) and preference (−15.0).

**The cost side is where the difference is stark: 4,089 median context tokens against their
~720.** memvara buys 86.4% with 5.7 times the context. That is the real distance, and it is
not an accuracy problem — it is that every point above cap 3 has been bought with turns.

## What follows

The claim layer does not earn its slots on LongMemEval by any of the three routes tested,
and saying so is more useful than testing a fourth. Two questions are worth more than
another arm:

1. **Where do the last 8.6 points live?** Multi-session and preference, in a corpus where
   accuracy is bought with context. Whatever closes them is unlikely to be a ranking change,
   since ranking has been measured five ways now.
2. **Is the token ratio the actual product problem?** Answering at 5.7x the context is a
   cost and latency story before it is an accuracy one, and it is the one number where the
   gap is not close.

Nothing here argues against the write-path work on its own terms: a store whose world clock
and belief clock record the same instant is not bitemporal, and fixing that stands whatever
this benchmark says. It argues only that the benchmark will not show it.
