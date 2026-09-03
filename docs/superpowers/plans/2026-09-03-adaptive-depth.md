# Where the last 8.6 points live, and why the token ratio is the same problem

**Date:** 2026-09-03
**Evidence:** the 199-question cap-15 control arm, analysed offline. No new spend.

## The two questions had one answer

memvara retrieves a **fixed** number of turns for every question. The questions differ in
how much evidence they need by a factor of three.

| type | gold evidence available | accuracy | score at the cut, vs the top |
| --- | ---: | ---: | ---: |
| multi-session | 24 turns | 75.5% | **0.60** |
| knowledge-update | 24 turns | 96.8% | 0.47 |
| temporal-reasoning | 22 turns | 83.0% | **0.55** |
| single-session-preference | 12 turns | 75.0% | **0.58** |
| single-session-user | 12 turns | 96.4% | 0.39 |
| single-session-assistant | 8 turns | 100.0% | 0.35 |

The last column is the 15th retrieved turn's score as a fraction of the first's. Read it
as: **is the cut landing on a cliff, or in the middle of a plateau?**

The three categories still on a plateau at the cut — multi-session, preference, temporal —
are exactly the three below parity. The three that have fallen off the cliff by turn 15 are
exactly the three at parity. That ordering is not approximate; it matches the accuracy
ordering across all six.

## Accuracy is a smooth function of how much of the evidence arrives

| fraction of gold-session content retrieved | n | accuracy |
| --- | ---: | ---: |
| 0–25% | 33 | 66.7% |
| 25–40% | 38 | 84.2% |
| 40–60% | 76 | 90.8% |
| 60%+ | 52 | **94.2%** |

**94.2% is Supermemory's 95%.** The questions where memvara already sees most of the
evidence already perform at their level. The deficit is entirely in the questions where a
fixed budget delivers a third of what the answer needs.

Within multi-session, the wrong answers are not worse-retrieved in absolute terms — they
retrieve 11.7 gold turns against 11.3 for the right ones. They simply need more: 38 turns
of gold evidence against 27.8. The same budget covers a smaller share, and the model
undercounts. Every multi-session failure text says so in its own words: "missing the coffee
maker", "excludes the Natural History Museum", "missing at least one album".

## The proposal: depth follows the score curve, not a constant

Keep a turn while its score is at least `T` times the top score; stop at the cliff. A
question with one clear answer has a sharp cliff and stops early. An aggregation question
has a long plateau and keeps going.

Simulated on the existing cap-15 lists — which can only stop *earlier*, never go deeper:

| rule | turns/question | est. tokens | change |
| --- | ---: | ---: | ---: |
| cap 15 (today) | 15.0 | 4,376 | — |
| T = 0.45 | 13.1 | 3,723 | −15% |
| **T = 0.55** | **11.3** | **3,214** | **−27%** |
| T = 0.65 | 8.8 | 2,423 | −45% |

And it allocates by need without being told the question type. At T = 0.55:

| type | turns kept | gold evidence it needs |
| --- | ---: | ---: |
| multi-session | 13.5 | 24 |
| temporal-reasoning | 12.4 | 22 |
| knowledge-update | 11.4 | 24 |
| single-session-user | 8.0 | 12 |
| single-session-assistant | 6.8 | 8 |

Assistant questions are currently served 15 turns and need 8; the rule gives them 6.8. That
is where the 27% comes from, and it costs nothing in accuracy because those categories are
already at 100% and 96.4%.

**The two questions are one problem.** The fixed cap over-serves the easy categories, which
is the token ratio, and under-serves the hard ones, which is the accuracy gap. A rule that
follows the curve fixes both, in opposite directions, at once.

## What to measure next

The simulation can only show the saving, because a cap-15 list cannot be extended past 15.
The gain needs one arm:

- **`read_max_episodes=40`** as a ceiling rather than a target, with `T = 0.55` deciding
  where to stop. The ceiling exists so a pathological plateau cannot run away.
- **Against the 86.4% control**, same 199 questions, same ingest, read-path only.
- **Prediction, stated before the run**: multi-session and temporal rise, because their
  plateaus continue past 15; assistant and user are unchanged, because they stop before it;
  median context tokens fall despite the higher ceiling, because four of six categories
  stop earlier than they do today.

If accuracy rises and tokens fall together, that is both questions answered. If tokens rise
without accuracy, the plateau is noise rather than evidence and the fixed cap was right.

One caveat worth stating before the arm rather than after: `T` is being chosen from the same
data it will be evaluated on. 0.55 sits between the two clusters in the table above, which
is a reasonable place for a boundary, but the arm measures a threshold fitted to this
corpus. A second corpus is what would show whether the rule generalises or whether 0.55 is
LongMemEval's number.
