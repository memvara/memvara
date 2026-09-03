# Widening the retrieval pool and reranking it: what it buys, measured offline

**What this is.** An offline coverage measurement. No reader, no judge, no answers generated,
no accuracy produced. The only accuracy number here is the control's, read out of the control
run's own checkpoint. Everything else counts whether a question's gold turns are present in a
selection and how many tokens that selection renders to.

**What it was run against.** The 199-question LongMemEval sample, served by a local server
built from core `6c0ff6c` with `read_max_episodes=200`, reporting version `0.10.0`. One search
per question at k=200, cached to disk, then every selection rule applied to that cache.

**The control.** The shipped k=15 arm, judged **86.4% (172 of 199)** at a **median recorded
context of 4,089 tokens**. Both numbers come from `memvara-cap15-control/checkpoint.json`.

Coverage is scored over the **192** questions that have a gold turn. Seven are `_abs`
abstention items with no gold turn and are excluded from every coverage row; their ids are in
`results.json` under `no_gold_qids`. The control's judged 86.4% is over all 199, including
those seven, which it answers correctly (11 of 11 abstention items in the run overall).

Commands, in the order run. Everything below comes from these:

```
cd local/pool/verify (this branch's checkout)
python3 v1_identity.py        # pool identity: is this the same retriever?
python3 v1b_scoredrift.py     # score agreement on shared turns, quantified
python3 v2_recompute.py       # rows, token accounting, gold matching, from the raw cache
python3 v4_endtoend.py        # contextTokens rebuilt from the harness prompt + paired bootstrap
python3 v5_details.py         # displacement counts, drift direction
python3 v6_pertype.py         # every published row, overall and per question type
python3 v7_stubrisk.py        # what survives the 20-word assistant stub
python3 v8_quota.py           # role quotas at 720 tokens; control's judged result by type
python3 v9_claimsblock.py     # size of the claims block on top of the turns block
python3 v10_budget.py         # the budget that actually lands recorded context under 720
python3 v11_assistcost.py     # what a gold assistant turn costs; turns kept per fill rule
PYTHONPATH=<the clean core checkout> python3 v3_ce.py
```

---

## 1. The answer

**Widening the pool to k=200 and reranking it does not measurably raise gold coverage at 15
turns.** The best reranked row is +0.009 mean coverage over the control, with a 95% paired
bootstrap interval of [−0.015, +0.032] — better on 12 questions, worse on 7, tied on 173. That
is not distinguishable from zero on this sample. It also costs tokens rather than saving them:
4,357 median against the control's 3,937, because the reranker promotes longer turns. Widening
alone, before reranking, is very slightly *negative* (−0.005), and that too rests on a single
question. Report this as **no measurable gain at 15 turns**, not as a small one.

**Selecting user turns only, inside a 720-token budget, does not reach the control's coverage
— and the entire shortfall is one question type.** The best whole-turn 720-token row is 0.825
mean coverage against the control's 0.906, a gap of −0.081 with a 95% interval of [−0.133,
−0.030], worse on 26 questions and better on 16. Remove the 22 `single-session-assistant`
questions, whose gold turn is by construction something the assistant said, and the same row is
**0.926 against the control's 0.893** on the remaining 170. Across the other five question
types, a user-only selection at 682 median tokens already covers more gold than the full
15-turn control at 3,937. The deficit is not general. It is a hole exactly where a user-only
rule predicts one.

The practical consequence: **the lever is which turns you select, not how you rank them and not
how big the budget is.** Filtering to user turns lifts the 720-token row from 0.602 to 0.799.
Reranking then lifts it from 0.799 to 0.825. The role filter is worth about 7.6× the reranker.
And the ceiling is far below either — gold turns and nothing else render at a median of 165
tokens, 4% of the control's context, for perfect coverage. Everything between 165 and 4,089
tokens is paid for not knowing which turns the gold ones are.

---

## 2. Is this the same retriever? Yes, and here is what that licenses

Before any of the coverage numbers mean anything, the k=200 pool has to be the same retrieval
the control ran, widened — not a different scorer or a different candidate set.

| check | result |
| --- | --- |
| control turns absent from the k=200 pool | **0 of 2,985** |
| pool top-15 identical as a set to the control's 15 | 179 of 199 (89.9%) — 164 identical set and order, 15 same set reordered, 20 different cut |
| questions where the pool returns a *different candidate* | **0** |
| depth of displaced control turns in the pool | rank 16 ×15, 17 ×6, 18 ×2, 26 ×1 |
| score agreement on shared turns | 2,947 of 2,985 bit-identical (98.7%) |
| the 38 that differ | all 38 **higher** in the pool, none lower; 26 questions; max delta 0.0234 |
| API return order vs score-descending order | identical on 199 of 199 pools |
| exact score ties straddling the 15/16 boundary | 0 |

Every difference between the pool and the control is a **cut moving**, not a candidate
changing. No control turn is missing; the displaced ones sit one to three ranks below the cut
(once at 26); and the boundary score gaps on the differing questions are 0.0003–0.0007 against
0.001–0.026 on the agreeing ones — the cut moves where the list is flattest. The 38 drifting
scores all move in one direction, upward, which is what a rank-fusion score does when more
candidates enter the fused list.

This licenses three things and no more. **First**, every selection rule below is being applied
to candidates the shipped retriever actually returns, so a coverage difference between rows is
a difference between selection rules. **Second**, the recomputed control row is a valid
baseline: rescoring the control's own 15 turns with this code gives 0.9056 mean coverage and
83.3% of questions fully covered, and the same 15 turns render to a median of 3,937 tokens.
**Third**, the token accounting is not self-referential. Porting the harness's own prompt
builder and rebuilding its `contextTokens` reproduces the recorded value on **199 of 199
questions, delta exactly 0**, median 4,089 — the published figure.

Gold matching is clean on the same evidence: **0 of 39,430** pooled turns fail to match a turn
in their question's haystack, 0 of 2,985 control turns fail, no question has two pooled turns
that whitespace-normalise together, and ten seeded spot checks read correctly by eye.

---

## 3. Coverage

n = 192. "Turns-block tokens" is the rendered excerpt block alone — see §5 for why that is not
the same as the harness's recorded context. `CE@N` means: take the first N turns in the pool's
own order, re-sort those by the reranker's score. The reranker never sees a turn the pool did
not return; its identifier and timings are in `ce.meta.json`.

### At 15 turns

| selection | mean coverage | full | zero | median tok | p90 tok |
| --- | ---: | ---: | ---: | ---: | ---: |
| control's own 15 (recomputed) | 0.9056 | 83.3% | 3.6% | 3,937 | 5,053 |
| widened pool, its own top 15 | 0.9004 | 82.8% | 4.2% | 3,934 | 5,042 |
| CE@50, top 15 | 0.9143 | 86.5% | 3.6% | 4,357 | 5,862 |
| CE@100, top 15 | 0.9069 | 85.9% | 4.7% | 4,166 | 5,732 |
| CE@200, top 15 | 0.9069 | 85.9% | 4.7% | 3,863 | 5,658 |
| first 15 user turns, pool order | 0.8336 | 78.1% | 12.5% | 1,120 | 1,595 |
| first 15 user turns, CE@200 | 0.8411 | 79.7% | 12.5% | 998 | 2,233 |

Paired bootstrap on the two claims this table invites, 20,000 resamples, n=192:

| comparison | delta | 95% CI | better / worse / tied |
| --- | ---: | --- | --- |
| CE@50 top 15 vs control's 15 | +0.0087 | [−0.0148, +0.0324] | 12 / 7 / 173 |
| CE@50 vs CE@200 (does depth hurt?) | +0.0074 | [−0.0035, +0.0217] | 4 / 1 / 187 |
| widened pool's top 15 vs control's 15 | −0.0052 | [−0.0156, +0.0000] | 0 / 1 / 191 |

None of the three is established. The first two intervals cross zero; the third is one
question (`38146c39`, 1.0 → 0.0, and 1/192 = 0.0052 exactly). The mechanism behind the third is
real and visible in §2 — widening lets a candidate ranked low by one leg and high by the other
enter the fused list and displace the 15th — but "widening costs coverage" is a claim resting
on a single question, so state the mechanism, not the effect.

### At a 720-token budget

Fill is greedy in the given order and stops at the first line that would push the block past
720; the first item is always kept. Cost is measured by encoding the whole rendered block, not
by summing per-line estimates.

| selection | mean coverage | full | zero | median tok | p90 tok |
| --- | ---: | ---: | ---: | ---: | ---: |
| pool order, all roles | 0.6021 | 46.9% | 24.0% | 566 | 701 |
| pool order, user turns only | 0.7991 | 72.9% | 14.6% | 679 | 713 |
| CE@200, all roles | 0.6981 | 58.3% | 19.3% | 574 | 704 |
| **CE@200, user turns only** | **0.8251** | **77.6%** | 13.0% | **682** | 712 |
| CE@200, assistant turns kept as a 20-word stub | 0.8920 | 84.9% | 5.7% | 694 | 716 |
| oracle: gold turns and nothing else | 1.0000 | 100.0% | 0.0% | **165** | 319 |

The stubbed row is an upper bound, not a like-for-like result — see §6, anomaly 6.

### Per question type, mean coverage

n per type: single-session-user 25, single-session-assistant 22, single-session-preference 12,
multi-session 52, knowledge-update 29, temporal-reasoning 52.

| selection | ss-user | ss-assistant | ss-pref | multi-session | knowledge-update | temporal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control's own 15 | 1.000 | 1.000 | 0.847 | 0.825 | 1.000 | 0.862 |
| widened pool, top 15 | 1.000 | 1.000 | 0.764 | 0.825 | 1.000 | 0.862 |
| CE@50, top 15 | 1.000 | 0.909 | 0.847 | 0.886 | 0.983 | 0.881 |
| CE@200, top 15 | 1.000 | 0.909 | 0.847 | 0.878 | 0.983 | 0.862 |
| first 15 user turns, CE@200 | 0.960 | 0.045 | 0.889 | 0.948 | 1.000 | 0.913 |
| 720, pool order, all roles | 0.880 | 0.818 | 0.681 | 0.374 | 0.621 | 0.577 |
| 720, pool order, user only | 0.920 | 0.091 | 0.889 | 0.832 | 1.000 | 0.875 |
| 720, CE@200, all roles | 1.000 | 0.818 | 0.556 | 0.562 | 0.828 | 0.599 |
| **720, CE@200, user only** | 0.960 | **0.045** | 0.889 | **0.918** | 1.000 | **0.885** |
| 720, CE@200, assistant stubbed | 1.000 | 0.909 | 0.847 | 0.848 | 0.983 | 0.837 |
| oracle | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Read the bold row across. At 682 median tokens, user-only selection **beats the 15-turn
control** on multi-session (0.918 vs 0.825) and temporal-reasoning (0.885 vs 0.862), ties it on
knowledge-update (1.000), beats it on preference (0.889 vs 0.847), loses one question on
single-session-user (0.960 vs 1.000) — and collapses to 0.045 on single-session-assistant,
where the control is 1.000. One column carries the whole deficit.

### Per question type, share of questions fully covered

| selection | ss-user | ss-assistant | ss-pref | multi-session | knowledge-update | temporal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control's own 15 | 100.0% | 100.0% | 75.0% | 65.4% | 100.0% | 78.8% |
| CE@50, top 15 | 100.0% | 90.9% | 75.0% | 78.8% | 96.6% | 82.7% |
| 720, CE@200, user only | 92.0% | 4.5% | 83.3% | 82.7% | 100.0% | 82.7% |
| 720, CE@200, assistant stubbed | 100.0% | 90.9% | 75.0% | 76.9% | 96.6% | 78.8% |
| oracle | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

### The same rows with single-session-assistant removed (n = 170)

| selection | all 192 | 170 without ss-assistant |
| --- | ---: | ---: |
| control's own 15 | 0.9056 | **0.8934** |
| widened pool, top 15 | 0.9004 | 0.8875 |
| CE@50, top 15 | 0.9143 | 0.9150 |
| first 15 user turns, CE@200 | 0.8411 | 0.9440 |
| 720, pool order, user only | 0.7991 | 0.8908 |
| **720, CE@200, user only** | 0.8251 | **0.9260** |
| 720, CE@200, assistant stubbed | 0.8920 | 0.8898 |

Two things fall out. Across the 170 questions that are not single-session-assistant, the
720-token user-only arm covers more gold than the control does at 3,937 tokens — 0.9260 against
0.8934, on 17% of the context. And the stubbed arm, which looks
best overall, is *worse* than the plain user-only arm once the assistant column is removed —
0.8898 against 0.9260 — because stubs eat budget that would otherwise hold user turns. Half of
what the stubbed arm keeps is assistant text: 1,299 of 2,543 turns kept, 51.1%.

### The ceiling

The oracle renders each question's gold turns and nothing else: **165 median tokens, p90 319**,
for perfect coverage. That is 4% of the control's 4,089. The 720-token budget is already more
than four times the oracle's median and the best whole-turn selection inside it still misses
17.5% of gold turns. The headroom is in selection. It is not in budget, and it is not in
ranking.

The pool is close to recall-saturated but is not a full ceiling: **99.48% of gold turns appear
somewhere in the k=200 pool** — one gold turn of 360 is never returned, so 191 of 192 questions
could in principle be fully covered by some reranking of the pool, and no reranked row can
reach 1.000.

---

## 4. Where the reranker puts gold user turns

This is the ranking claim the whole design rests on, and it is real, consistent, and modest.
Over the 336 gold user turns present in the pool, in user-only lists:

| ordering | gold user turns found | median rank | within top 8 | top 10 | top 15 |
| --- | ---: | ---: | ---: | ---: | ---: |
| pool order | 336 of 337 | 3.0 | 85.1% | 88.4% | 91.4% |
| CE@200 | 336 of 337 | 2.0 | **88.4%** | 90.5% | 93.5% |

The reranker moves the median gold user turn from rank 3 to rank 2 and moves about three points
of gold turns into the top 8. That is why reranking buys 2.6 points of coverage at 720 tokens
(0.799 → 0.825, CI [+0.003, +0.051]) rather than ten. The effect is genuine — that interval
excludes zero — and small.

The reranker's own score distribution says why deeper candidate lists do not help: 99.02% of
the 39,430 logits are negative and the median sits at −11.19 against a saturated floor of
−11.51. Roughly half of every pool is pinned at "irrelevant" with no usable separation, so the
extra candidates a deeper list admits are mostly noise. The cache is complete and
deterministic: 39,430 scores for 39,430 pool episodes, no duplicates, no gaps, and rescoring
five seeded questions reproduces the cached values bit-identically.

Where the user-only arm loses at 720 tokens, it is not usually a ranking failure. It holds a
median of **10** user turns (p10 7, p90 14); the gold user turns it misses sit at **median rank
18** in its own reranked list, p90 29, max 55. Those are budget losses, not ordering losses. Of
the 43 questions it leaves below full coverage, 21 are single-session-assistant; of the gold
turns it misses, 23 are assistant turns it can never select and 34 are user turns it could not
afford.

---

## 5. Against the target: median context ≤ 720 with accuracy held ≥ 86.4%

### The budget is tighter than it looks

The 720-token budget in this study is the **turns block**. The harness's recorded
`contextTokens` also includes a claims block, and that block costs a **median of 130 tokens per
question** (mean 140, p90 246, max 585; median 3 claims per question, 11 questions have none).
So an arm built to a 720-token turns block records a median context of about **797**, not 720.

Measured, with the claims block held at what the control's own run recorded per question:

| turns-block budget | mean coverage | full | turns-block median | projected recorded median | mean coverage, 170 without ss-assistant |
| --- | ---: | ---: | ---: | ---: | ---: |
| 720 | 0.8251 | 77.6% | 682 | 797 | 0.9260 |
| 720, best-fit fill | 0.8277 | 77.6% | 712 | 840 | 0.9289 |
| **590** | 0.8054 | 74.5% | 583 | **712** | **0.9037** |
| 560 | 0.7961 | 72.9% | 552 | 680 | 0.8932 |

To land recorded context at or below 720, build the turns block to about **590 tokens**. At
that budget the user-only reranked arm covers 0.8054 overall and 0.9037 on the 170 non-assistant
questions — still above the control's 0.8934 on the same 170.

### The target cannot be met by a whole-turn rule, and the reason is measurable

The control's judged result, by question type, from its own checkpoint:

| type | n | control judged |
| --- | ---: | ---: |
| single-session-user | 28 | 27 (96.4%) |
| **single-session-assistant** | **22** | **22 (100.0%)** |
| single-session-preference | 12 | 9 (75.0%) |
| multi-session | 53 | 40 (75.5%) |
| knowledge-update | 31 | 30 (96.8%) |
| temporal-reasoning | 53 | 44 (83.0%) |
| **all** | **199** | **172 (86.4%)** |

The control answers every single-session-assistant question correctly. Those 22 questions are
11.1% of the benchmark, and they are exactly the ones a user-only rule cannot reach: coverage
0.045, one question of 22 fully covered. So a user-only arm starts 11 points down before
anything else happens.

Letting assistant turns back in does not fix it, and I measured that rather than assuming it.
Gold assistant turns have a median length of 211 words (min 6, max 446; 95.7% are longer than
20 words). Rendered as excerpt lines they cost a **median of 264 tokens each, p90 521, max
593** — one of them takes **37% of a 720-token block** at the median, and the largest takes 82%.
Admitting them in reranked order, with a quota (best-fit fill throughout this table, so the
rows are comparable to each other rather than to the 682-token row above):

| rule at 720 tokens | mean coverage | ss-assistant fully covered | multi-session fully covered | temporal fully covered | median turns kept |
| --- | ---: | ---: | ---: | ---: | ---: |
| user turns only | 0.8277 | 1 / 22 | 43 / 52 | 43 / 52 | 12 |
| + at most 1 assistant turn | 0.7825 | 16 / 22 | 29 / 52 | 32 / 52 | 6 |
| + at most 2 assistant turns | 0.7981 | 19 / 22 | 29 / 52 | 32 / 52 | 6 |
| no role rule at all | 0.7955 | 19 / 22 | 28 / 52 | 32 / 52 | 6 |

Every quota setting is **worse overall** than the user-only rule. One assistant turn buys 15
single-session-assistant questions and costs 14 multi-session and 11 temporal ones. At 720
tokens the two needs are mutually exclusive, and no fixed whole-turn rule holds both.

Stubbing looked like the way out, and the measurement says it is not. A 20-word stub keeps
about 9% of the median gold assistant turn. Testing deterministically whether the dataset's own
answer string survives: it is present somewhere in the whole gold assistant turn for 11 of 23
gold assistant turns, and present in the 20-word stub for **3**. The stubbed arm's 0.892
coverage counts 22 truncated assistant turns as covered; on the evidence, roughly three of them
still contain the answer.

### The one judged arm to run next

**Run the user-only reranked arm judged, on the same 199 questions: search at k=200, rerank the
whole pool, keep user turns only, greedy-fill a 720-token turns block.**

It is the right one for three reasons. It is the only 720-token arm with no truncation artefact
in it, so its judged number means what it says. It is the best whole-turn row measured. And its
offline coverage on the 177 non-assistant questions is *above* the control's, so a judged run
answers the one question coverage cannot: how much of a coverage advantage converts into
correct answers when the context shrinks by a factor of five.

### Prediction, stated now

- **Overall judged: 77%, range 75–81% (149–161 of 199).** It will not reach 86.4%.
- **On the 177 questions that are not single-session-assistant: 86%, range 84–89%** — against
  the control's 150 of 177, 84.7%. That is the real finding to look for: at one sixth of the
  context, the arm matches or slightly beats the control everywhere except one question type.
- **On the 22 single-session-assistant questions: 2, range 0–4** — against the control's 22.
- **Recorded context median: about 797, not 720**, because of the claims block. If the target
  is measured on recorded context, this arm does not meet it either; the 590-token variant does,
  at 712, and costs 0.022 mean coverage.
- **The stubbed arm, if it is also run, will not beat this one by much and may lose to it.** Its
  coverage is 0.067 higher, but the answer survives the stub in 3 of 23 gold assistant turns and
  its non-assistant coverage is *lower* (0.8898 vs 0.9260). Predicted gap: within 3 points either
  way. This is the sharpest falsifiable claim in this document; if the stubbed arm wins clearly,
  the answering model is recovering more from 20 words than the string test suggests.

What would falsify the framing: if the user-only arm comes back at or above 84% overall, then
single-session-assistant questions are being answered without the assistant turn, and the role
model behind this entire analysis is wrong.

---

## 6. Anomalies, corrections and unverified items

Five claims in the original write-up did not survive re-measurement. They are corrected here,
and the corrections are already applied everywhere above.

1. **"Reranking adds about 0.008 on top of the control" — not established.** +0.0087, CI
   [−0.0148, +0.0324], tied on 173 of 192 questions. Also **"reranking deeper is worse"** —
   +0.0074, CI [−0.0035, +0.0217], resting on five questions. Both are reported above as no
   measurable effect.
2. **"Scores agree exactly on every shared turn" — false.** 38 of 2,985 differ (1.3%), across 26
   questions, max delta 0.0234. The conclusion it was supporting survives and is strengthened:
   all 38 move *upward*, which is what fusion under a wider candidate list must do.
3. **"The 20 differing questions differ by exactly one turn" — false.** 17 differ by one, 2 by
   two, 1 by three: 24 displaced turns. The original depth tally (15+6+2+1 = 24) was the correct
   one of the two.
4. **The per-question-type median-token table in the original message does not match the
   artefacts.** `results.md` and an independent recompute agree exactly with each other and not
   with that table. The values used above are the recomputed ones.
5. **The line-prefix constant is 15 tokens, not 16** — 191 of 200 sampled lines, 14 on 8, 16 on
   1; 16 is right only if the joining newline is counted in. It is used in one place, the cost
   of a gold turn absent from the pool in the oracle row, which is one turn across 192
   questions. Immaterial to every number here.
6. **The stubbed row scores selection, not rendered text.** A truncated assistant turn counts as
   covered. Measured, the answer string survives the stub in 3 of 23 gold assistant turns. Treat
   0.892 as an upper bound on what stubbing could buy, not as a result.
7. **The fill rule matters at the margin.** Stopping at the first over-budget line gives 0.8251
   at 682 median tokens; skipping it and trying the next gives 0.8277 at 712. Both are inside
   720. Worth fixing deliberately rather than by accident.
8. **The pool is not a full ceiling.** One gold turn of 360 is never returned at k=200, so one
   question of 192 caps every reranked row below 1.000.
9. **Claims occupy pool slots.** Every request returned exactly 200 items, but 370 of the 39,800
   were claims rather than turns, so 161 of 199 questions have fewer than 200 turns to rank
   (minimum 189). "Top N" counts turns, not raw result positions.
10. **Gold is overwhelmingly but not entirely in user turns.** 337 of 360 gold turns are user
    turns. The 23 assistant ones fall in 23 distinct questions: 20 single-session-assistant, 2
    single-session-user, 1 multi-session. So two questions outside the single-session-assistant
    type also need an assistant turn, and a user-only rule loses them too.

Unverified, and not to be read as established:

- **No accuracy was produced here.** Coverage is a proxy for whether an answer is reachable, not
  for whether it is given. The only judged number in this document is the control's own.
- **The embedder is not exposed.** `/v1/stats` reports scope, counts and `extractor:
  fast-path-only`; `/v1/health` reports the version. The embedder was recorded as not exposed
  rather than guessed. The `/v1/stats` sample during the pull also returned HTTP 429, so the
  store's own counts were never captured; the pull's tallies stand in.
- **The projected recorded-context figures in §5 assume the claims block is unchanged** — the
  same search returns the same claims either way. That is reasonable and was not measured
  against a re-run.
- **The reranker was run once.** Its cache was verified complete, duplicate-free and
  bit-reproducible on five seeded questions, but a second full pass was not made.
- **The prediction in §5 is a prediction.** It is grounded in the control's judged result by
  question type, the coverage rows, and a literal string test on the stubs — not in any judged
  run of the arm itself.
