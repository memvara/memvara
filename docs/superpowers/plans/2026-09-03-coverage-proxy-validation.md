# Does gold-turn coverage predict judged accuracy?

**2026-09-03 · validation of the free retrieval proxy, and what it licenses for the context-reduction sweep**

Yes for one question and no for the question the sweep asks. Whether the retrieved context contains
every `has_answer` turn predicts whether the answer is judged correct, strongly and not as an artefact
of easy questions: pooled over 2,251 scored (arm, question) rows, accuracy is 93.0% when coverage is 1,
26.1% when partial and 23.1% when zero, coverage alone separates correct from incorrect answers at
AUC 0.866, and holding the question fixed the odds ratio is 49.7 — larger, not smaller, than the
unstratified 39.7. But `has_answer` coverage is defined on *which turns were retrieved*, and every lever
that reduces context leaves that set alone. Three arms with byte-identical retrieval span 0.874 to 0.849
accuracy and 3,700 to 2,028 median tokens at one constant coverage value. Under a 720-token cap the
coverage objective is maximised by truncating every turn to about 120 characters, which holds coverage at
its uncapped maximum of 0.9165 while deleting 72% of the gold-turn text. So coverage is a screening tool
for coarse retrieval-budget changes and **cannot be the objective of a sweep whose target is
`median context <= 720 with accuracy held >= 86.4%`** — it is an objective the sweep can satisfy while
destroying the thing it stands for. Section 5 gives the objective to use instead, as a rule.

The answering model and the judge are both recorded as `gpt-5.4` in all ten checkpoints. Every accuracy
figure in this document is that judge's verdict, and every other number comes from the commands below.

**Reproduce.** No script here loads the 277 MB dataset; all three read a derived artefact built earlier.

```
cd .../local/proxy-validation/writeup-check   && python3 check.py        # from rows.csv
cd .../local/proxy-validation/verify-inference && for s in analyze.py analyze2.py analyze3.py analyze4.py analyze5.py; do python3 $s; done   # from derived.pkl
cd .../local/proxy-validation/verify-recompute && python3 ordering_check.py                                 # from rows_verify.csv
```

Full paths: `local/proxy-validation/ (this branch's checkout)`
holds `rows.csv` (2,333 rows, one per arm-question), `measure.py`, `summary.md`, and the three verification
directories `writeup-check/`, `verify-inference/`, `verify-recompute/`.

**Definitions, one term per concept.** *has_answer coverage* is the share of a question's `has_answer`
turns — counted as distinct whitespace-normalised contents — that appear among the retrieved turns.
Questions with no `has_answer` turn are excluded and counted separately. *Session content coverage* is
o200k_base tokens of retrieved turns belonging to an answer session, over tokens of every turn in those
answer sessions. *Context tokens* is the harness's own `answer.contextTokens`. *Gold-token retention*,
introduced in section 5, is the share of `has_answer`-turn tokens actually present in the rendered
context after truncation.

---

## 2. Per arm, on the shared 199-question set

Both coverages beside accuracy and median context. The last column is the correction that makes this
table readable: the finished cap-3 baseline scored on **the same questions as that row**, because the
arms did not all run on the same set and the sets differ sharply in difficulty.

| arm | n | accuracy | has_answer cov | session cov | median ctx tokens | turns/q | baseline on the same questions |
|---|---|---|---|---|---|---|---|
| memvara-baseline-18e3626 (cap 3, shipped default) | 199 | 0.548 | 0.577 | 0.132 | 751 | 3.00 | 0.548 |
| memvara-spread5 (cap 5, per-source spread) | 56 | 0.429 | 0.480 | 0.061 | 1266 | 5.00 | 0.393 |
| memvara-epcap8 (cap 8) | 59 | 0.610 | 0.703 | 0.250 | 2191 | 8.00 | 0.390 |
| memvara-epcap15 (cap 15) | 106 | 0.755 | 0.843 | 0.404 | 4108 | 15.00 | 0.377 |
| memvara-cap15-control (the 86.4% control) | 199 | 0.864 | 0.906 | 0.495 | 4089 | 15.00 | 0.548 |
| memvara-cap15-index (claims as index) | 199 | 0.839 | 0.906 | 0.496 | 4167 | 18.13 | 0.548 |
| memvara-adaptive (score floor) | 199 | 0.874 | 0.916 | 0.468 | 3700 | 14.32 | 0.548 |
| memvara-adaptive-trim (floor + truncation at 400) | 199 | 0.849 | 0.916 | 0.468 | 2028 | 14.32 | 0.548 |
| memvara-adaptive-trim800 (floor + truncation at 800) | 199 | 0.859 | 0.916 | 0.468 | 2409 | 14.32 | 0.548 |
| memvara-rerank8 (reranker) | 106 | 0.660 | 0.697 | 0.235 | 2166 | 8.00 | 0.377 |

**Corrected from the builder's summary.** The builder's per-arm table reported each arm on its own
completed question set and noted only that two runs were unfinished. Verification found the sets differ
in difficulty, and I confirmed it: the baseline scores **0.353** on the 266-question set that `epcap15`
and `rerank8` ran, against 0.544 on its own 500 (Fisher p = 3e-20); **0.322** on `spread5`'s 146
completed (p = 1.7e-10); **0.325** on `epcap8`'s 160 (p = 1.8e-11). The 199-question set is the one that
is representative of the 500 — baseline 0.548 against 0.544, p = 0.93 — which is why this table is the
199-subset.

**One correction beyond both verifications.** Restricting to the 199 set does not rescue `spread5` and
`epcap8`. Their 199-subsets are 56 and 59 questions drawn from the hard 266, and the baseline scores
0.393 and 0.390 there. `epcap15` and `rerank8` are in the same position at n = 106, baseline 0.377. Only
the six rows with n = 199 are directly comparable with each other; the other four are comparable only
against the baseline column beside them.

---

## 3. The prediction test

### Pooled, all 2,333 rows

| has_answer coverage | n | accuracy |
|---|---|---|
| 0 | 251 | 0.231 |
| between 0 and 1 | 498 | 0.261 |
| 1 | 1502 | 0.930 |
| (no `has_answer` turn in the haystack) | 82 | 0.768 |

Partial coverage behaves like zero, not like a midpoint. Missing one gold turn of two costs almost
everything that missing both costs.

| session content coverage quartile | n | accuracy |
|---|---|---|
| Q1 (<= 0.102) | 584 | 0.502 |
| Q2 (<= 0.252) | 585 | 0.619 |
| Q3 (<= 0.499) | 581 | 0.788 |
| Q4 | 583 | 0.918 |

| coverage | n | point-biserial r | p | AUC |
|---|---|---|---|---|
| has_answer | 2251 | 0.645 | 1.81e-265 | 0.866 |
| session content | 2333 | 0.338 | 1.28e-63 | 0.728 |

**The p-values overstate the certainty and should not be quoted.** Those 2,251 rows are on average 4.7
repeated measurements of 479 distinct questions. Resampling questions (cluster bootstrap, 2,000 draws)
puts the pooled has_answer AUC at **0.866, 95% CI [0.835, 0.897]**. The point estimate stands; the
certainty attached to it does not.

Logistic regression with arm fixed effects, `LogisticRegression(penalty=None)`. statsmodels is not
installed on this machine, so the AUC is in-sample and there are no standard errors.

| model | n | coef has_answer | coef session | in-sample AUC |
|---|---|---|---|---|
| has_answer + arm FE | 2251 | 4.42 | — | 0.877 |
| session + arm FE | 2333 | — | 2.93 | 0.746 |
| both + arm FE | 2251 | 4.21 | 1.54 | 0.878 |

Session coverage adds essentially nothing once the gold turns are accounted for: 0.878 against 0.877.
It is the weaker measure on every arm and in every framing, and the builder's ranking of the two holds.

### Within each arm

| arm | n scored | cov 0: n / acc | partial: n / acc | cov 1: n / acc | r has_answer | AUC has_answer | r session | AUC session |
|---|---|---|---|---|---|---|---|---|
| memvara-baseline-18e3626 | 479 | 105 / 0.267 | 163 / 0.184 | 211 / 0.934 | 0.624 | 0.852 | 0.256 | 0.627 |
| memvara-spread5 | 142 | 43 / 0.163 | 58 / 0.276 | 41 / 0.878 | 0.563 | 0.809 | −0.210 | 0.411 |
| memvara-epcap8 | 156 | 20 / 0.100 | 42 / 0.238 | 94 / 0.894 | 0.651 | 0.862 | 0.313 | 0.683 |
| memvara-epcap15 | 257 | 16 / 0.188 | 50 / 0.260 | 191 / 0.916 | 0.623 | 0.844 | 0.274 | 0.686 |
| memvara-cap15-control | 192 | 7 / 0.286 | 25 / 0.360 | 160 / 0.963 | 0.597 | 0.857 | 0.231 | 0.707 |
| memvara-cap15-index | 192 | 7 / 0.286 | 25 / 0.320 | 160 / 0.950 | 0.600 | 0.839 | 0.258 | 0.712 |
| memvara-adaptive | 192 | 7 / 0.429 | 20 / 0.450 | 165 / 0.945 | 0.476 | 0.777 | 0.132 | 0.630 |
| memvara-adaptive-trim | 192 | 7 / 0.429 | 20 / 0.400 | 165 / 0.927 | 0.446 | 0.750 | 0.113 | 0.599 |
| memvara-adaptive-trim800 | 192 | 7 / 0.429 | 20 / 0.450 | 165 / 0.927 | 0.437 | 0.741 | 0.122 | 0.612 |
| memvara-rerank8 | 257 | 32 / 0.156 | 75 / 0.240 | 150 / 0.913 | 0.641 | 0.863 | 0.278 | 0.659 |

has_answer coverage beats session coverage on all ten arms. Its own discrimination weakens as an arm
approaches the ceiling — AUC 0.741 to 0.777 on the three adaptive arms, where only 7 questions have zero
coverage — which is the same fact section 5 turns on.

### Is the association just question difficulty?

Partly, and it survives the control. Leave-one-out question difficulty alone predicts correctness at
AUC 0.863, almost the same as coverage's 0.869 on that sample, so the two are heavily entangled. But
conditioning on the question does not remove the effect:

- **Mantel-Haenszel odds ratio stratified by question: 49.7**, against 39.7 unstratified, over the 171
  questions where some arm covered the gold turns and another did not.
- **Difference in differences**, which removes both question difficulty and arm identity. On questions
  where `cap15-control` and the cap-3 baseline have *equal* coverage, control beats baseline by
  **+2.1 points**; on questions where control covers more, by **+62.2 points**. `adaptive` against the
  same baseline: +0.0 and +66.7. `epcap15`: +3.1 and +62.9.

Twelve extra turns buy almost nothing except through covering the gold turn. That is the causal reading,
and it holds under a harder test than the correlation.

### The resolution problem

86.4% of questions have two or fewer gold turns (36.4% have one, 45.8% have two), so **89.4% of scored
rows take one of exactly three coverage values: 0, 0.5, or 1**. Across the cap-3-to-cap-15 range, where
mean coverage moves from 0.577 to 0.916, that is enough resolution. Near the target, where the arms sit
within one point of each other, it is not.

---

## 4. Do the gold turns fit in 720 tokens?

Yes — they fit twice over, and it does not help, because nothing can select them.

Rendered the way the provider renders turns (`- [YYYY-MM-DD HH:MM] role: content` per line under a
13-token section header), the `has_answer` turns of the 192 `cap15-control` questions that have one
occupy **median 166 tokens, mean 186, p90 332, max 775; 99.5% fit inside 720**. The arm's actual median
context is 4,089.

**Corrected.** That figure counts the excerpts block only, and `contextTokens` is the whole context
delta. Adding the claims this arm actually retrieved — a block of median 130 tokens — gives
**median 308, mean 325, p90 519, max 825, and 97.9% inside 720**. Still feasible, but the claims eat 42%
of the budget, and the builder's stated figure silently assumes they are dropped.

The gold turns fit. Today's ranker cannot find them at that budget:

| what a 720-token cap buys on cap15-control | claims kept | claims dropped |
|---|---|---|
| mean turns admitted, of 15 | 2.49 | 3.17 |
| mean has_answer coverage (uncapped: 0.906) | 0.529 | 0.605 |
| share with coverage 1 (uncapped: 83.3%) | 40.6% | 47.4% |
| predicted accuracy from this arm's own coverage-to-accuracy map (target 0.864) | 0.596 | 0.641 |

The reason is rank, not size. Across the 308 gold turns this arm retrieved, the median rank is 2 and the
mean 3.22; only **28.6% are first, 45.5% are inside the top 2, 86.0% inside the top 8**. A 720-token
budget admits about 2.5 turns, so it admits about half the gold turns.

Relaxing the budget moves the prediction slowly: 0.688 at 1,000 tokens, 0.790 at 2,000, **0.833 at 3,000**
— still short of 0.864, all with claims dropped. The lever is ranking quality or compression inside the
turn, not cap size.

---

## 5. What this licenses for the sweep

### The three facts that constrain the objective

**Coverage is invariant to truncation.** `memvara-adaptive`, `-trim` and `-trim800` have identical
has_answer coverage on 192 of 192 scored questions — verification found their `search.results` byte-identical
on all 199 — and their accuracies are 0.874, 0.849, 0.859 at median contexts 3,700, 2,028, 2,409. Every
retrieval-side proxy is constant across them by construction. No sample size fixes this.

**Maximising coverage under a 720-token cap has a degenerate optimum.** Recovering the trim arms'
settings by matching `contextTokens` (`HEAD_WHOLE=5, TAIL_CHARS=400` for `-trim`, exact on 99.5% of
questions; `5, 800` for `-trim800`, exact on 100%) and then sweeping that knob with retrieval untouched:

| head whole / tail chars | median ctx | has_answer coverage | gold-token retention |
|---|---|---|---|
| 0 / 400 | 1298 | 0.9165 | 0.723 |
| 0 / 200 | 880 | 0.9165 | 0.475 |
| **0 / 120** | **668** | **0.9165** | **0.284** |
| 0 / 60 | 519 | 0.9165 | 0.142 |

A sweep told to reach median context <= 720 while maximising has_answer coverage picks `TAIL_CHARS ~ 120`:
668 tokens, coverage still at its uncapped maximum, 71.6% of the gold-turn text deleted. It reports success.

**Coverage also cannot rank arms that are close.** Over the 45 arm pairs, coverage gets 38 right, inverts
3, and cannot order 4 more (exact coverage ties with accuracy differences). Restricted to each pair's
common questions: 3 inversions and 7 ties, of which 9 pairs carry a real accuracy difference the proxy
misses. The inversions include `cap15-control` against both trim arms, and — the one case where the proxy
is actively wrong about a genuine retrieval change — the cap-3 baseline against `spread5` on their 146
common questions, where the baseline has *higher* coverage (0.494 vs 0.479) and *lower* accuracy
(0.322 vs 0.418). The ties are not rounding: `cap15-control` and `cap15-index` retrieve 15.00 vs 18.13
turns per question with different claim handling, have **identical coverage on 192 of 192 questions**, and
differ by 2.5 accuracy points.

**And 199 questions cannot enforce the accuracy side of the target.** The 86.4% control is 172/199, exact
95% CI **[0.809, 0.909]**. Measured discordance among the five near-target arms is 0.056. Paired questions
needed at 80% power: **175 for a 5-point drop, 489 for 3 points, 1,103 for 2 points.** At n = 199 only a
drop of roughly 8 points is detectable, and every pairwise McNemar test among those five arms is
non-significant (p >= 0.092).

### The rule the sweep should follow

1. **Optimise coverage at the budget, not uncapped coverage.** Measure recall of `has_answer` turns on the
   context as actually rendered at the token cap. It is still free and still discriminates, and today it
   reads 0.529 (claims kept) or 0.605 (claims dropped) at 720 against 0.906 uncapped. This is the right
   objective for the selection half of the search — the half that changes what is retrieved.
2. **Gate truncation on gold-token retention; do not maximise it.** Retention is the only measured
   quantity that responds to compression: 0.9049 / 0.9026 / 0.8931 for `adaptive` / `-trim800` / `-trim`,
   which orders those three the same way the judge does. Three arms whose accuracy differences are not
   significant is not validation, so use it as a floor — reject any configuration below about 0.85 — not
   as a thing to climb.
3. **Do not compare two configurations on coverage when the gap is under about 5 points, and do not use
   coverage at all between configurations that differ downstream of retrieval.** Coverage is blind to
   rendering, truncation and claim handling by construction.
4. **Treat a coverage win as a shortlist, never as an accuracy claim.** Judge the finalists, and size the
   judge run to the question: 199 questions cannot certify "accuracy held >= 86.4%". Use the full 500 for a
   3-point margin, and restate the constraint as a non-inferiority margin the data can support rather
   than as an equality the run cannot test.
5. **Expect 720 to fail on ranking, not on budget, and instrument for that.** The gold turns fit in 308
   median tokens including claims, but sit at median rank 2 with 45.5% inside the top 2, so the cap buys
   2.5 turns and a predicted 0.596. Coverage-at-the-budget is the metric that will show ranking quality
   improving; cap size will not move it.

---

## 6. Anomalies, corrections, and what could not be verified

**Corrected against the builder's summary.**

1. **The per-arm accuracy column is not comparable across question sets.** See section 2. The builder's
   anomaly 1 noted the two unfinished runs but not that their question sets are harder. Beyond both
   verifications: the 199-subsets of `spread5` (n=56) and `epcap8` (n=59) are also drawn from the hard
   266, baseline 0.393 and 0.390.
2. **"The ordering it produces matches the judged ordering across all ten arms" is false.** See section 5.
   The two verifications agree on the count for each arm's own set — 3 inversions plus 4 exact ties of 45
   — and differ only on the common-question count, 9 against my 10, because one further tied pair
   (`epcap15` vs `cap15-index`) also ties on accuracy and so costs nothing. I believe 10 pairs are
   unorderable or inverted, 9 of them consequentially.
3. **The 720-token feasibility figure omits the claims block.** Corrected from 166 median / 99.5% under 720
   to 308 median / 97.9% under 720 when the claims the arm actually retrieved are included. See section 4.
4. **The pooled p-values are not interpretable as stated.** Use the cluster bootstrap CI [0.835, 0.897],
   not p = 1.81e-265.
5. **Gold-token retention for `-trim800` is 0.9026, not 0.7991.** The verification's own intermediate run
   printed 0.7991 before the truncation parameters were recovered, and its `analyze2.py` still prints the
   superseded value in the "three real arms" block while `analyze3.py` prints the corrected one. I re-ran
   both. Use 0.9026.
6. **Session coverage counts tokens on raw text but keys them on normalised text.** `measure.py` line 109
   hashes `norm(content)` and line 112 stores `content` for the token count, so the reported figure counts
   whitespace the matching key discards. Recomputing on normalised text moves the Q3 quartile edge from
   0.4992 to 0.497, the quartile accuracies by at most 0.003, the p from 1.28e-63 to 1.66e-63 and the
   logistic coefficient from 2.93 to 2.91. Neither definition is wrong — numerator and denominator use the
   same table either way — and no conclusion moves. The figures here use the `rows.csv` convention.

**Anomalies that stand.**

7. **Two runs are unfinished.** `memvara-spread5` has 146 of 266 evaluated (114 pending, 6 failed) and
   `memvara-epcap8` 160 of 266 (98 pending, 8 failed); both are still `status: running`, and 226 questions
   have search results but no evaluate phase. Those rows are excluded.
8. **Three checkpoints declare 500 `targetQuestionIds` but hold 266 questions** (`epcap8`, `epcap15`,
   `rerank8`). The declared target list is stale; the question keys were used.
9. **The proxy's floor: 58 of 251 zero-coverage rows (23.1%) were judged correct.** Temporal reasoning is
   28 of the 58, which fits — a date stated in a neighbouring turn answers a temporal question without the
   turn the dataset marks as gold. Then multi-session 11, single-session-preference 7, single-session-user
   5, knowledge-update 4, single-session-assistant 3. Corpus accuracy over all 2,333 rows is 0.706.
10. **Abstention questions carry no coverage signal.** 21 question ids have no `has_answer` turn anywhere
    in their haystack (82 rows), and every one is an abstention variant; they are excluded from all
    coverage figures and score 0.768. A further 9 abstention ids do have gold turns (57 rows), and there
    coverage and correctness are uncorrelated (r = 0.098, p = 0.468). They dilute the pooled proxy rather
    than reversing it, which is expected for questions scored on abstaining.
11. **`spread5` is the only arm where session coverage correlates negatively with correctness**
    (r = −0.210, AUC 0.411) while its has_answer coverage still correlates positively (r = 0.563). It also
    has the lowest coverage of any arm on both measures and an incomplete run. The mechanism is not
    established; do not let this arm carry weight alone.
12. **Latent crash in `measure.py`.** It uses `ENC.encode_ordinary_batch` for the bulk token table but
    plain `ENC.encode` at lines 89, 139 and 222, and the corpus contains at least one turn whose text
    includes a literal special-token string. Neither call reaches that content today — the line-222
    fallback fires only for an unmatched turn, and there are none — so no reported number is affected, but
    the script would fail on a corpus where an unmatched or `has_answer` turn carried one.

**What could not be verified here.**

13. **The 13 questions whose session coverage cannot reach 1.0.** One verification reports that where an
    answer session repeats a turn's content, the metric is capped below 1 (worst case 0.813), because the
    numerator dedups retrieved turns by content while the denominator sums every turn including repeats.
    No saved script produces that number, so I could not reproduce it. The mechanism is visible in the code
    and is present identically in both implementations, so it explains no disagreement between them; it
    would bias session coverage down for a handful of questions, and session coverage is not the measure
    this document recommends.
14. **The builder's assistant-token-share table** — the share of retrieved-turn tokens sitting in
    assistant turns — was not re-derived by either verification, so it is not carried into this document.
15. **The trim arms' truncation parameters are inferred, not read from configuration.** `HEAD_WHOLE=5,
    TAIL_CHARS=400` and `5, 800`, recovered by reproducing `contextTokens` exactly on 99.5% and 100% of
    questions. The sweep table in section 5 rests on that inference. The render reconstruction itself is
    solid: it reproduces `answer.contextTokens` exactly on 199 of 199 questions for both
    `cap15-control` and `adaptive`.
16. **Content matching is clean, and I checked the weaker form of it.** 24,767 retrieved turns across the
    ten arms, 0 unmatched, which I recomputed from `rows.csv`. The stronger check — that every retrieved
    turn is byte-identical to a dataset turn before normalisation, that no retrieved content appears in two
    sessions, and that a 20-turn sample matches by role and date — was run once during verification and I
    did not re-run it, because it requires loading the dataset.
