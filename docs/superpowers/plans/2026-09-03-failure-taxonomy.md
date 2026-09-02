# LongMemEval failure taxonomy, and what it says to build

> **Superseded in part on 2026-09-03 by the arm that tested it.** The central
> recommendation below — select for gold-*session* coverage — was implemented, run, and
> refuted. Coverage rose from 39.8% to 84.2% and judged accuracy *fell* to 41.8%, against
> 61.0% for simply taking more turns by score. Accuracy tracks gold **turns** retrieved,
> not sessions reached, and the "accuracy ≈ 0.85 × coverage" law below was a confound:
> reaching every gold session correlated with retrieving many turns, and this document
> read the correlation as the cause. The section "What this document got wrong" at the end
> records the corrected numbers. What still stands: the harness's retrieval metrics are
> broken, the offline coverage measurement is sound, and the failure-mode table.

**Date:** 2026-09-03
**Data:** `memvara-baseline-18e3626` (500 questions), `memvara-epcap8` (160), `memvara-epcap15` (266)
**Analysis scripts:** `local/failure-taxonomy/` (not committed)

## The headline

Retrieval is the bottleneck after all. The earlier conclusion — "hit@8 is 88.6%, so
retrieval is fine and the gap is synthesis" — was built on a metric that does not measure
what its name says. Corrected, accuracy tracks retrieval coverage almost exactly.

| gold sessions actually retrieved | n | accuracy |
| --- | ---: | ---: |
| all of them | 304 | **76.0%** |
| some but not all | 176 | 21.6% |
| none | 20 | 15.0% |

A 54-point spread, on a variable we control.

## Why the old number was wrong

`src/orchestrator/phases/retrieval-eval.ts` computes:

```ts
const totalRelevant = Math.max(1, relevantRetrieved)
const hitAtK   = relevantRetrieved > 0 ? 1 : 0
const recallAtK = relevantRetrieved > 0 ? 1 : 0   // identical expression to hitAtK
```

`recallAtK` is not recall. It is `hitAtK` under another name, and `totalRelevant` sets the
denominator to the numerator, so it can never fall below 1 once anything is found. `ndcg`
takes its ideal ranking from the retrieved set too, so it scores ordering within what came
back and never coverage of what was needed. The harness has no coverage metric at all.

What `hitAtK` does measure is "an LLM judged at least one returned item topically
relevant". On 150 of 500 questions it reported a hit while real coverage was partial or
zero; accuracy on those 150 was 26.0%.

LongMemEval ships the real labels in `answer_session_ids`. Mapping every returned turn back
to its haystack session by exact content match recovers true coverage offline, for free.
The mapping is exact: all 1,500 `kind: turn` results in the baseline matched a haystack
turn. The only unmapped items are memvara's own synthesised claims, which have no turn to
match.

## Coverage by question type

| type | n | accuracy | harness hit@k | **all gold sessions** |
| --- | ---: | ---: | ---: | ---: |
| single-session-assistant | 56 | 87.5% | 100.0% | 100.0% |
| single-session-user | 70 | 91.4% | 97.1% | 97.1% |
| single-session-preference | 30 | 53.3% | 86.7% | 83.3% |
| knowledge-update | 78 | 62.8% | 93.6% | 62.8% |
| temporal-reasoning | 133 | 44.4% | 80.5% | 45.9% |
| multi-session | 133 | 26.3% | 85.0% | 33.8% |

Accuracy sits within a few points of full-coverage rate everywhere, and knowledge-update
matches it to the decimal. The harness metric sits 20–50 points above both on exactly the
types we are losing.

The three episode-cap arms confirm the relationship holds as we change retrieval:

| arm | all-gold coverage | accuracy | median context tokens |
| --- | ---: | ---: | ---: |
| cap 3 | ~40% | 35.3% | 753 |
| cap 8 | 74.4% | 61.9% | 2,051 |
| cap 15 | 87.2% | 74.1% | 4,070 |

Accuracy runs at roughly 0.85 × coverage across all three, which looked at the time like a
law worth extrapolating: drive coverage to 100% and get about 85%, above Supermemory's
published 81.6%. **A later arm broke it.** All three of these arms vary coverage and turn
count together, so the relationship they show is not evidence about which one is causal.

## The 228 failures

| failure mode | full coverage | partial coverage | total |
| --- | ---: | ---: | ---: |
| abstained — answered "I don't know" | 45 | 92 | **137** |
| wrong specific value | 9 | 29 | 38 |
| wrong arithmetic or aggregation | 4 | 19 | 23 |
| stale value chosen over the update | 4 | 12 | 16 |
| preference retrieved but not used | 8 | 3 | 11 |
| wrong ordering | 3 | 0 | 3 |
| | 73 | 155 | 228 |

Sixty percent of failures are abstentions, and two thirds of those had incomplete evidence
— the model behaving correctly given what it was handed. There is no large pool of failures
where the evidence was present and the model fumbled it. The 73 full-coverage failures are
real but they are a quarter of the problem, and a chunk of them (`eaca4986` a chord
progression, `561fabcd` a chosen name, `e8a79c70` an egg quantity) are cases where the gold
*session* was reached but the answer-bearing *turn* within it was not.

## What to build: select for coverage, not for relevance

The candidate pool from a single query already contains every gold session 87.2% of the
time at k=15. The pool is not the problem — the cut is. Replaying those ranked lists under
different selection policies, at no cost:

| selection over the same cap-15 pool | all-gold coverage |
| --- | ---: |
| everything in the pool | 87.2% |
| top-3 by rank | 39.8% |
| **top-3, one per session** | **68.8%** |
| top-5 by rank | 60.9% |
| **top-5, one per session** | **82.3%** |
| top-8 by rank | 73.3% |
| top-8, one per session | 86.8% |

Five slots chosen one-per-session cover more gold than eight chosen by rank, and nearly as
much as the entire fifteen. Expected accuracy at 0.85 × 82.3% is about 70%, at roughly a
third of cap-15's tokens. **The measured answer was 41.8%.** The coverage prediction held
almost exactly — 84.2% against 82.3% predicted — and the accuracy prediction did not.

This also explains why the cross-encoder reranker bought nothing. A reranker scores each
item independently for topical relevance. On a question spanning three sessions, the eight
most topically relevant turns are frequently eight turns from the *same* session — the best
possible answer to the wrong question. Coverage is a property of the set, and no per-item
scorer can optimise it.

### The depth trade-off is real but cheap

Spreading costs depth within the winning session:

| policy at k=5 | coverage | avg gold turns retrieved |
| --- | ---: | ---: |
| pure rank | 60.9% | 4.16 |
| spread first 3, then rank | 71.8% | 3.73 |
| spread all 5 | 82.3% | 2.97 |

Losing 1.2 gold turns to gain 21 points of coverage is a good trade, because depth barely
moves accuracy. Within the full-coverage subset, accuracy was 78.9% at one gold turn, 79.2%
at two and 75.1% at three — flat.

### It does not need to be gated on question type

The worry that diversification would hurt single-session questions does not survive the
data. Those three types need exactly one gold session (0% have more than one), every policy
preserves the rank-1 item unchanged, and rank-1 is already a gold session 73–100% of the
time. Meanwhile knowledge-update needs two sessions in 100% of cases and sits at 62.8%
coverage, so it gains as much as multi-session does. The policy can be unconditional.

## On running a second query and comparing result sets

The instinct is right one level up and wrong one level down. Right: the thing worth scoring
is a *set*, not an item, and "which of these result sets is better" is the correct question
to be asking. Wrong: a second query is not where the missing coverage is. One query's pool
already holds all the gold 87.2% of the time, so query decomposition competes for the last
13 points while selection is worth 21 points on its own and costs nothing — no extra call,
no added latency, no model.

Order of work: fix selection first, measure, then decide whether decomposition earns the
remaining headroom.

## Claims are consuming half the retrieval budget for no measured gain

In the baseline, 52.6% of returned slots and 27.9% of returned bytes were memvara `memory`
claims rather than conversation turns. None carried gold evidence in a form that maps to a
gold session, and the separate ingest-model experiment measured their contribution at 0.0
points. Whatever else changes, claims should not be taking half the slots.

## Corrections to the earlier findings document

`2026-09-02-baseline-findings.md` states that retrieval is not the bottleneck and puts a
34-point gap down to synthesis. Both rest on `hitAtK`, and both are withdrawn. The reranker
result stands as measured; the explanation offered for it at the time — that precision was
already adequate — was wrong, and the correct explanation is that per-item relevance
scoring cannot improve a set-coverage objective.

## What this document got wrong

The recommendation above was implemented as `max_per_source` in `memvara` and run as the
`memvara-spread5` arm: `max_episodes=5`, one turn per source conversation, on the same 266
questions and the same ingested data as the other three arms. It reached 146 of them before
the gateway key was exhausted.

| arm | multi | temporal | total | ctx tokens | gold **sessions** | gold **turns** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cap 3 baseline | 18.9% | 45.8% | 32.2% | 758 | ~40% | 3.0 |
| cap 5 + spread | 31.1% | 52.8% | **41.8%** | 1,238 | **84.2%** | **2.0** |
| cap 8 | 48.6% | 73.6% | 61.0% | 2,050 | 74.4% | 7.0 |
| cap 15 | 56.8% | 84.7% | 70.5% | 4,051 | 87.2% | 11.0 |

The mechanism did exactly what the simulation said it would. Session coverage went from
39.8% to 84.2%, within two points of prediction. Accuracy did not follow it: at 84.2%
coverage the arm scores 41.8%, where cap 15 scores 70.5% at 87.2%.

Read the last two columns together. Accuracy tracks gold **turns** retrieved — 2.0 turns
at 41.8%, 7.0 at 61.0%, 11.0 at 70.5% — and is indifferent to how many distinct
conversations were touched. Reaching a conversation is not carrying the sentence that
answers the question, and one turn from each of five conversations loses the surrounding
turns that make any of them usable.

Against the token curve the arm sits about where plain ranking would at 1,238 tokens, so
the spread bought nothing; it bought the same accuracy differently. `max_per_source`
therefore ships disabled, and the missing control — `max_episodes=5` with the spread off —
is what would say whether it actively hurt at that budget or merely failed to help.

The depth argument earlier in this document ("accuracy was 78.9% at one gold turn, 79.2% at
two and 75.1% at three — flat") is the specific claim that misled. That measurement came
from the cap-3 baseline's full-coverage subset, which is dominated by questions needing a
single session, where one turn is often the whole answer. It does not generalise to
questions spanning several conversations, and it was used as though it did.
