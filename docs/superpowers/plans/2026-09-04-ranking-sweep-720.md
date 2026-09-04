# Ranking sweep at 720 tokens: the stop rule fires

**No gated configuration reaches mean coverage 0.93, so the stop rule fires and no judged run
is justified by this grid.** The 1,290-cell grid tops out at **0.9096** against routed-720's
**0.9032** — in whole gold turns, 320 of 360 against 318 — and it buys those two turns by
*losing* two questions on full coverage (162 of 192 against 164). Of the 14 questions routed-720
lost to the token budget, the best any configuration recovers is **3**, and the recovery sets are
not a superset of each other: `{09d032c9, c4f10528, gpt4_a1b77f9c}` is reached by 26 gated
configurations, best `rerank(bge_base) · no boost · depth 100 · best-fit`, and
`{bf659f65, c4f10528, gpt4_a1b77f9c}` by exactly 2, best
`rerank(bge_base) · all@2.0 · depth 50 · best-fit`.

The verifier reproduced the grid cell for cell and then attacked the inference. It holds, for a
stronger reason than the one the sweep gave: it is not merely that nothing reaches 0.93, it is
that **nothing in the grid can raise the judged score at all**. Scoring every gated cell as
(questions newly fully covered that the judge got wrong) minus (questions losing full coverage
that the judge got right), the best achievable net over all 774 gated configurations is **+0** —
0 cells positive, 3 cells at zero, 771 negative — and the coverage leader scores **−3**.

The finding worth carrying out of this sweep is not a configuration. It is that **13 of the 14
budget-cut questions are not budget failures.** Render only the gold turns each one needs, same
routing, same line format, and the block costs 210 to 643 tokens; every one fits inside 720.
They are ranking failures, and no ordering in this grid finds them.

---

## The stop rule

Stated in advance: stop unless a gated configuration reaches mean gold-turn coverage 0.93.

**Fired — stop.** Maximum gated mean coverage is 0.9096; maximum over the whole grid including
gated-out cells is the same 0.9096. Nothing comes within 0.02 of the threshold.

```
cd local/sweep
PYTHONPATH=<core checkout> python3 report_check.py
  grid 1290 gated_in 774 gated_out 516
  any gated >= 0.93: False
  any cell at all >= 0.93: False
  max gated mean_cov: 0.9096
```

The grid is 43 orderings x 3 depths x 2 fills x 5 extraction settings. `bge-reranker-v2-m3` was
excluded and remains excluded: its score file held 11,856 of 39,430 pairs when the sweep ran and
22,359 when the verifier read it, still being written. Adding it costs about ten seconds of grid
time and grows the grid to 1,890.

## Top configurations

The judged arm reproduces exactly as the grid cell `rerank(minilm_l6) · no boost · depth 200 ·
greedy · no extraction`, and its 28 judged misses split 14 / 13 / 1 as briefed.

| | ordering | temporal | depth | fill | mean cov | full | gold turns | med tok | p90 | cut14 |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | rerank(minilm_l6) | — | 200 | greedy | 0.9032 | **0.8542** (164/192) | 318/360 | 676.5 | 712 | 0 |
| 1 | rrf+pool(bge_base) | dated@0.5 | 50 | best-fit | **0.9096** | 0.8438 (162) | 320 | 707.0 | 717 | 1 |
| 2 | rrf+pool(bge_base) | — | 200 | best-fit | 0.9095 | 0.8490 (163) | 320 | 713.5 | 720 | 2 |
| 3 | rerank(bge_base) | — | 100 | best-fit | 0.9090 | 0.8490 (163) | 319 | 710.0 | 718 | **3** |
| 4 | rerank(minilm_l6) | — | 100 | best-fit | 0.9071 | **0.8542** (164) | 319 | 710.0 | 718 | 0 |
| 5 | rrf+pool(bge_base) | dated@0.5 | 50 | greedy | 0.9070 | 0.8385 (161) | 319 | 676.0 | 710 | 1 |

Every cell in the grid is extraction-free above; all 774 gated cells carry retention 1.000.

**The top of the table is flat and the baseline is inside it.** Nine gated cells sit within 0.005
of the best and 29 within 0.010; the spread across the top twenty is 0.0077, which is less than
one and a half questions (one question is 0.0052). The baseline ranks **11th of 774**. Ten gated
cells beat it on mean coverage, **none** beats it on full coverage, and none beats it on both.

## The 14 questions the budget cut

Three is the ceiling, from two different directions.

| set | recovered | gated configs | best of them | its mean cov |
|---|---|---|---|---|
| A | 09d032c9, c4f10528, gpt4_a1b77f9c | 26 | `rerank(bge_base) · — · 100 · best-fit` | 0.9090 |
| B | bf659f65, c4f10528, gpt4_a1b77f9c | 2 | `rerank(bge_base) · all@2.0 · 50 · best-fit` | 0.8902 |

Both sets contain `c4f10528` and `gpt4_a1b77f9c`; they disagree on the third. The sweep reported
a single maximal set — that was wrong, and the correction is the verifier's C1, reproduced here.

Now the part that matters. Rendering only each question's gold turns, under the same routing and
the same line format, every one of the fourteen fits inside 720 tokens:

| qid | gold turns | roles | route | reachable after routing | cheapest block holding them |
|---|---|---|---|---|---|
| 09d032c9 | 1 | user x1 | user | 1 | 58 |
| 88432d0a | 4 | user x4 | user | 4 | 344 |
| 9d25d4e0 | 5 | user x5 | user | 5 | 390 |
| ac031881 | 1 | assistant x1 | user | **0** | 13 |
| bf659f65 | 3 | user x3 | user | 3 | 223 |
| c4f10528 | 1 | assistant x1 | assistant | 1 | 273 |
| gpt4_31ff4165 | 6 | user x6 | user | 6 | 464 |
| gpt4_7abb270c | 6 | user x6 | user | 6 | 549 |
| gpt4_7f6b06db | 3 | user x3 | user | 3 | 230 |
| gpt4_a1b77f9c | 6 | user x6 | user | 6 | 463 |
| gpt4_a56e767c | 6 | user x6 | user | 6 | 643 |
| gpt4_ab202e7f | 5 | user x5 | user | 5 | 382 |
| gpt4_e061b84f | 3 | user x3 | user | 3 | 210 |
| gpt4_f420262d | 3 | user x3 | user | 3 | 210 |

Thirteen are lost to ordering alone. The fourteenth, `ac031881`, cannot be recovered by any
configuration here: its only gold turn is an assistant turn, the fourteen-phrase rule routes the
question user-only, and the role filter deletes the evidence before any ordering sees it.

The verifier found a second such question the sweep missed. **The role filter deletes 6 gold turns
across 6 questions, so the reachable ceiling is 353 of 360, not 359.** Two questions are
unreachable outright: `ac031881`, and **`dc439ea3`** — user gold, routed assistant, because
"did you say" fires on a question whose evidence is a user turn.

## What each axis is actually worth

The sweep's per-axis table was a max-of-max: best cell containing a value against best cell
without it. That measure credits an axis with whatever else the winning cell happened to carry.
The numbers below are matched pairs — cells differing in exactly one axis — and two of the three
credited axes reverse under them. This is the verifier's correction C2, recomputed here.

```
PYTHONPATH=<core checkout> python3 report_check2.py
```

**Best-fit fill is the only real effect, and it is small.** Against greedy, holding everything
else fixed, it gains a mean of +0.0057 (median +0.0052) and wins **21 of 21** matched pairs. That
is about one question, every time, with no exceptions. It is also the only change that never costs
full coverage.

**The reranker choice is weak and not clearly ordered.** `bge-reranker-base` beats
`ms-marco-MiniLM-L-6-v2` by a mean of +0.0032 (median +0.0023) and wins 9 of 12 pairs, and beats
`ms-marco-MiniLM-L-12-v2` by +0.0036 (median +0.0055), 8 of 12. Half a question either way, with
three or four pairs going the other direction.

**RRF against plain reranker score order is a coin flip.** Over 18 matched pairs it is mean
−0.0002, median +0.0019, winning 11 of 18. The sweep credited it +0.0006 and the verifier
−0.0019 median; **both signs were artefacts** — the sweep's from max-of-max, and the verifier's
from a pairing loop that computes score-order minus RRF under a label reading the other way
(`verify/v9_axes.py:26` filters `r["base"]=="rerank"` and differences `r - o` where `o` is the
RRF cell). The magnitudes agree to the fourth decimal and the conclusion — a coin flip — is
unaffected; only the sign printed in that one row is wrong.

**The temporal boost costs coverage on average, at every weight.** Over all matched pairs at
extraction-none: `dated@0.5` is −0.0044 and wins 10 of 36; `dated@1.0` −0.0126, 2 of 36;
`dated@2.0` −0.0366, 0 of 36. Applying it to every question rather than dated ones is worse
throughout: `all@0.5` −0.0268, `all@1.0` −0.0614, `all@2.0` −0.1605, and **0 of 36 pairs won at
any weight**. The +0.0002 the sweep credited to `dated@0.5` was its single luckiest cell.

**Depth is inert only when the boost is off, and this qualification matters.** With no boost,
depth 50 against depth 200 is −0.0008 and wins 2 of 14. Turn a boost on and a shallow head starts
buying real coverage back: **+0.0148 at w=0.5 (23 of 24), +0.0362 at w=1.0 (21 of 24), +0.1009 at
w=2.0 (24 of 24)**. Depth is not a retrieval parameter here; it is damage control for the boost.
Anyone who keeps a boost and reads "depth is inert" will be wrong by up to a tenth of the metric.

**The temporal boost is not even the same treatment across rerankers, so those two axes are not
separable in this grid.** Min-max normalisation is applied to scores on incompatible scales:
`bge-reranker-base` emits probabilities (median 3.7e-05, max 0.35) while both MiniLMs emit logits
(median −11.3). The share of the head where the w=0.5 recency term exceeds the normalised
relevance score is 78.2% for MiniLM-L-6, 79.6% for MiniLM-L-12 and **95.9% for bge-base**. On
bge-base, `dated@0.5` is close to a straight recency re-sort of the head rather than a tilt — and
bge-base is the reranker in every one of the top three cells.

**Both question detectors are weak, and one selects against its own purpose.** The temporal
detector fires on 55 of 199 at precision 0.618 and recall 0.642 (34 true, 21 false, 19 missed),
and its misses are systematic rather than random: the entire *"how many days passed between X and
Y"* family carries no `when` / `before` / `after` / `since` and no month or year token, so it is
missed wholesale — `gpt4_7abb270c`, `gpt4_7f6b06db`, `gpt4_e061b84f` and `gpt4_f420262d` among
them. **`gpt4_a1b77f9c`, one of the three cut questions the leaders "recover", is itself a missed
temporal question**: the boost never applies to it, so its recovery belongs to some other axis and
was credited to the wrong one. The count detector fires on 79 of 199, of which `how many` supplies
62; as a predictor of "needs three or more gold turns" it runs at precision 0.247 against a base
rate of 0.156, and 31 of its 77 scored firings are duration questions averaging 2.03 gold turns.
The dominant pattern selects questions needing *fewer* turns. That is inert for the stop rule —
the count arms lose by 0.21 to 0.24 anyway — but it means **count-aware fill has not been fairly
tested**, because it is wired to a detector that is not selecting count questions.

## The retention gate and its casualties

516 of 1,290 cells fall below the 0.85 mean-retention gate. **Every one is an extraction arm**:
258 `lex` and 258 `top2`, the complete population of both.

| extraction | best mean cov | gated | retention range |
|---|---|---|---|
| none | 0.9096 | in | 1.000 – 1.000 |
| lex | 0.4681 | **all 258 out** | 0.749 – 0.781 |
| top2 | 0.3972 | **all 258 out** | 0.764 – 0.786 |
| count_lex | 0.7039 | in | 0.860 – 0.897 |
| count_top2 | 0.6659 | in | 0.878 – 0.902 |

**The gate is inert for the decision.** It removes only cells that lose the objective outright by
0.21 or more, and the top table would be unchanged without it.

It is not inert for what we can conclude about extraction, and the collision is worth naming. A
gold turn counts as covered only when 85% of its tokens survive, so an extraction that keeps
exactly the sentence carrying the answer and drops the rest scores as a **miss**. Measured over
the 91 gold user turns with more than two sentences and an identifiable answer-bearing sentence,
the lexical filter keeps the answer sentence 78.0% of the time at median retention 0.679, and the
top-2 cross-encoder filter keeps it **91.2% of the time at median retention 0.709**. Both sit
under the gate by construction. This sweep can say extraction loses on the stated objective. It
cannot say a reader would answer worse from an extracted block, because coverage as defined does
not measure that.

One disclosure: the sweep's coverage rule adds the per-turn 0.85 retention requirement that
`score.py` does not have. At extraction-none the two rules coincide exactly — retention is 1.000
on all 258 such cells — so no number in the top table is affected.

## Fitted against general

The verifier attacked the winner five ways. Everything about *which cell is best* is fitted.
Everything about the *shape of the problem* survives.

**Fitted — do not carry any of this forward as a configuration recommendation.**

- The winner's whole advantage is 13 questions moved (8 up, 5 down), and the two best movers alone
  exceed the net gain. +0.0064 is **1.23 question-equivalents**.
- It is not distinguishable from zero. Paired bootstrap over 20,000 resamples: **+0.0064, 95% CI
  [−0.0113, +0.0264], P(Δ ≤ 0) = 0.251**. On full coverage the delta is **−0.0104**.
- Winner's curse, measured directly. Split-half selection over 400 splits — argmax on 96 questions,
  scored on the held-out 96 — gives a held-out gain over baseline of **mean −0.0076, at or below
  zero in 87% of splits**; on full coverage **mean −0.0201, at or below zero in 99%**. **Thirty
  different configurations win** across the 400 splits.
- Each axis of the winner rests on a handful of questions: flipping one axis at a time moves
  **1 question for fill, 2 for temporal, 7 for reranker, 10 for depth, 18 for ordering base**.
- The decisive one: **the gain lands on questions the judge already got right.** Of the 13 the
  winner moves, the two supplying the entire net gain (`561fabcd`, `gpt4_ec93e27f`, both 0→1) were
  judged **correct** on routed-720, and four it degrades from full coverage (`gpt4_2ba83207`,
  `gpt4_d84a3211`, `gpt4_2f8be40d`, `gpt4_194be4b3`) were **all judged correct**. Best achievable
  judged net over all 774 gated cells: **+0**. The three cells at zero are all MiniLM-L-6 at depth
  200 — routed-720's own family.

**General — carry these forward.**

- The grid reproduces exactly. An independent reimplementation that does not import `sweeplib.py`
  or read `prep.pkl`, and that counts tokens by whole-block encoding rather than the additive
  model, agrees on all 1,290 cells across seven statistics each: **largest disagreement 0, cells
  differing 0**, and the gated-out set identical.
- The additive token model is sound: 0 mismatches on 4,000 random blocks and 1,500 ellipsis-bearing
  blocks, plus 444 exact spot checks inside the verifier's own grid run.
- The arm really is routed-720: whole-block token counts sit at a constant −5 offset from the
  judged run's `contextTokens` on 176 of 199 questions and −4 on the other 23, which is fixed
  prompt overhead rather than a selection difference.
- **13 of the 14 cut questions are ranking failures, not budget failures.** Gold-only blocks cost
  210–643 tokens against a 720 budget.
- **The reachable ceiling is 353 of 360, not 359**, because routing deletes 6 gold turns; two
  questions (`ac031881`, `dc439ea3`) are unreachable while routing is held fixed.
- Best-fit fill is a real, small, reliable gain: 21 of 21 matched pairs.
- The extraction pipeline is clean. Assistant turns are never extracted (0 of 39,430 episodes
  differ from the unextracted line), the header is never touched, and the ellipsis marker is paid
  for in every case (**15,913 of 15,913** marked lines cost more than the same kept sentences
  joined without it, mean +1.20 tokens, 19,115 tokens in total).

## The one judged arm to run next

**Run top-2 sentence extraction on user turns, with routed-720's own selection unchanged:
`rerank(minilm_l6) · no boost · depth 200 · greedy fill · top2 extraction`.** One change against
the judged arm, so the delta is attributable to extraction alone.

The reason is not that it looks best. It is the only axis in this study where **the offline metric
is structurally blind**. Ranking is measured and measures nothing: 774 gated cells, best judged net
+0. Routing is also offline-measurable — at extraction-none the sweep's coverage rule and
`score.py`'s coincide exactly — so the role-filter question can be swept for free and should be,
before any judged time is spent. Extraction is the one thing that cannot be screened offline,
because the coverage rule scores a block that keeps precisely the answer sentence as a miss.

The mechanism that could pay: top-2 extraction cuts user lines to a median 0.709 of their tokens,
so roughly 40% more lines fit inside 720, which is exactly the constraint that keeps deeper-ranked
gold turns out of the thirteen ranking failures. The mechanism that could cost: 8.8% of
multi-sentence gold user turns lose their answer sentence outright — about 8 turns among the 91
measured — and the reader also loses surrounding context that coverage never scored.

**Prediction, stated now: +1 question against routed-720's 171, and I would be surprised to see it
outside 165 to 176.** That is a judgement from the two measured mechanisms above, not a computed
interval; the upside and the downside are of similar size and I cannot separate them offline.

**Pre-registered decision rule, because the prediction sits inside the noise.** At 7.8% reader
self-disagreement roughly 15 of 199 judgements change on a re-read of the same arm, so a single run
cannot resolve a few questions either way. Adopt extraction only at **+8 or better**; treat
anything from −7 to +7 as no evidence and drop the idea rather than re-running it; investigate only
if it comes back worse than −7, which would mean truncation is costing the reader something the
mechanism above does not predict.

Two things not to spend judged time on. The coverage leader
(`rrf+pool(bge_base) · dated@0.5 · 50 · best-fit`) is predicted at **−3** by the judged-upside
score and should not be run. Best-fit fill on its own is free and never costs full coverage, so
adopt it as the default if any selection change is adopted at all — but it moves one question and
running it judged would buy no information.

---

## Commands

Sweep (builder), from `local/sweep`:

```
PYTHONPATH=<core checkout> python3 prep.py
PYTHONPATH=<core checkout> python3 repro.py
PYTHONPATH=<core checkout> python3 score_sentences.py
PYTHONPATH=<core checkout> python3 sweep.py
PYTHONPATH=<core checkout> python3 report_sweep.py
PYTHONPATH=<core checkout> python3 cut_diag.py
PYTHONPATH=<core checkout> python3 extract_diag.py
```

Verification (independent reimplementation), from `.../local/sweep/verify`:

```
PYTHONPATH=<core checkout> python3 v1_baseline.py
... v2_tokenmodel.py v2b_rolecheck.py v3_grid.py v4_ellipsis.py v5_tables.py v6_detectors.py
... v7_count.py v8_fitting.py v9_axes.py v10_scale.py v11_misses.py v12_movers.py
... v13_judged_upside.py v14_gate.py
```

Recomputed for this report, from `.../local/sweep`:

```
PYTHONPATH=<core checkout> python3 report_check.py
PYTHONPATH=<core checkout> python3 report_check2.py
```

`report_check.py` re-derives the stop-rule verdict, the gate counts, the top cells, the baseline
row, the two maximal cut14 recovery sets and the beat-baseline counts from `sweep_results.json`.
`report_check2.py` re-derives every matched-pair axis effect quoted above. Both were run for this
report and their output is what appears here.

## Not established

No reranker was re-run, no judged arm was run, and no paid model was called for this report. The
7.8% reader self-disagreement is taken from the brief. `bge-reranker-v2-m3` is absent from the
grid; its file was incomplete both times it was read. Everything said about extraction's effect on
a reader is mechanism plus the answer-sentence keep rates — coverage cannot measure it, which is
the whole reason for the arm above.

## Addendum: session diversity, measured after the report

The thirteen ranking failures are mostly multi-session questions, so the obvious untried axis
was diversity: at most `cap` turns per session in a first pass over the ranked list, optionally
relaxed by refilling the leftover budget from the skipped turns, optionally followed by a tail of
the other role's turns that still fit. `local/sweep/diversity.py` ran 56 such configurations over
the same cached pool, keyed by the turn's timestamp (what the harness sees; keying by the
dataset's session id gives the same numbers on the best capped cell).

**Refuted at every cap.** Against the baseline's 164 fully covered questions, cap 3 gives 161 to
162 with a judged upside of −2, cap 2 gives 153 to 155 at −9, and cap 1 gives 114 to 130 at −30
to −46. No capped configuration recovers any of the fourteen cut questions. The rendered blocks
already span 4 to 9 distinct sessions across their 8 to 13 lines, so diversity was never the
constraint. The only non-negative cells are the uncapped ones: a tail of the other role's turns
after the routed fill gains one covered question at judged upside +0 (median 707 tokens), and
best-fit fill repeats its +0 from the main grid.

The diagnosis the same run makes explicit: in the routed MiniLM-L-6 order the cut questions' gold
turns sit at ranks 14 to 37 (`88432d0a` at 9, 16, 18, 37; `gpt4_7f6b06db` at 2, 29, 31;
`gpt4_7abb270c` at 2, 4, 8, 19, 20, 25) while a 720-token block holds 8 to 13 verbatim lines.
No ordering in the grid lifts them into the top ten, and no fill can render rank 30 at 65 tokens
a line. What is left at this budget is compression: fewer tokens per turn, so that rank 30 is
inside the block.
