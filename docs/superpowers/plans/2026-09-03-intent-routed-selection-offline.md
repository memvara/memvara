# Query-conditional role routing at 720 tokens: measured offline, then verified

**2026-09-03 · offline coverage on the 199-question sample over the cached k=200 pool (core 6c0ff6c), no reader, no judge. The measurement is the first part; the verification that corrects its inferences is the second, and the corrected conclusion at the end is the one to cite.**

## Part 1 — the measurement, as the builder reported it


Every number below comes from two scripts in this directory:

```
cd local/intent
python3 fit_rule.py --dump      # fit on the 301, freeze, score the 199 held out
python3 score_intent.py         # 720-token rows over the cached pool + cross-encoder scores
```

`fit_rule.py` reads the 277 MB dataset once, caches the 500 `(qid, type, question)` triples
in `qmeta.json`, and works from that cache afterwards. `score_intent.py` reads the dataset
once for gold turns and imports `../pool/score.py` for the rendering, the o200k_base token
count, the greedy 720-token fill and the whitespace-collapsed gold matching, so its rows are
comparable line for line with `../pool/results.md`. Sanity row (a) reproduces the reference
CE@200 user-only row exactly: mean coverage 0.8251, median 682 tokens, 0.9260 on the 170
non-single-session-assistant questions.

## The rule

Case-insensitive `re.search` against the question text. Rule A is the frozen routing rule —
the spec's draft (patterns 1-2) plus the phrasings the 301 fitting questions show.

```
A1   \byou (?:suggested|recommended|proposed|listed|mentioned|said|told me|gave me|shared)\b
A2   \byour (?:suggestion|recommendation|list|advice)s?\b
A3   \byou (?:provided|offered|advised|described|explained|outlined|presented|showed|showed me|
     sent|sent me|wrote|created|drafted|generated|produced|came up with|pointed out|pointed me|
     referred me|directed me|recall)\b
A4   \bwhat (?:did|had) you\b
A5   \bwhich .{0,40}\bdid you\b
A6   \bhow (?:did|had) you\b
A7   \b(?:did|do) you (?:suggest|recommend|mention|say|tell|give|provide|list|advise|propose|
     offer|name)\b
A8   \b(?:can|could|would|will) you remind me\b
A9   \bremind me (?:what|which|who|how|of|about|again)\b
A10  \byour (?:tip|idea|proposal|suggestion|recommendation|advice|list|plan|draft|summary|
     explanation|instruction|step|response|answer|reply|comment|remark|guidance|feedback|
     solution|example|analysis|breakdown|overview|note|point|insight|strategy|approach|method|
     technique|option|alternative)s?\b
```

Rule B is rule A plus the discourse frame the benchmark wraps these questions in. It is kept
separate because it is a template cue, not an intent cue: a real user can say "in our previous
conversation" about something *they* said, and a false fire costs a question its whole budget.

```
B1   \b(?:our|the) previous (?:conversation|chat|discussion|session|exchange)\b
B2   \b(?:looking|thinking|going|check(?:ing)?) back (?:at|on|to) (?:our|the|that)
     (?:previous |earlier |last )?(?:conversation|chat|discussion)\b
```

The line breaks above are for reading; the patterns are single-line strings in `fit_rule.py`.

### Precision and recall

Fit on the 301 questions outside the cached sample; the 199 sample was scored once, after
freezing. The label is `question_type == "single-session-assistant"`.

| rule | set | positives | fired | TP | FP | FN | precision | recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | fit (301) | 34 | 31 | 31 | 0 | 3 | 1.000 | 0.912 |
| A | held out (199) | 22 | 19 | 19 | 0 | 3 | 1.000 | **0.864** |
| B | fit (301) | 34 | 34 | 34 | 0 | 0 | 1.000 | 1.000 |
| B | held out (199) | 22 | 20 | 20 | 0 | 2 | 1.000 | **0.909** |

Both rules fire on zero non-assistant questions in either set — 267 negatives in the fitting
set, 177 in the sample. Precision 1.0 is not a rounding: no negative matches any pattern.

The three rule-A misses in the fitting set:

- `f523d9fe` — "…our previous conversation about Netflix… Do you remember what show I used as an example…"
- `58470ed2` — "…I wanted to confirm - what did Borges say about the center and circumference of the Library?"
- `4baee567` — "…I wanted to confirm, how many times did the Chiefs play the Jaguars at Arrowhead Stadium?"

The three rule-A misses in the held-out sample (rule B catches the middle one):

- `1568498a` — "I'm looking back at our previous chess game and I was wondering, what was the move you made after 27. Kg2 Bd5+?"
- `561fabcd` — "I was thinking back to our previous conversation about the Radiation Amplified zombie… what we finally decided to name it?"
- `ac031881` — "I'm trying to recall what the designation on my jumpsuit was that helped me find the file number in the records room?"

Recall cannot be pushed much past this without giving up precision. The residual misses name
no assistant act at all: their surface form is a plain factual question ("how many times did
the Chiefs play…", "what was the designation on my jumpsuit"), and what makes the answer live
in an assistant turn is who happened to state the fact, which the question text does not say.
The only signal left in them is the "our previous conversation" frame, which is rule B, and
that frame is a property of how the benchmark writes questions rather than of what the user
wants.

## The 720-token rows

192 of the 199 scored (7 `_abs` items have no gold turn); 22 single-session-assistant, 170
other. Rule A fires on 19 of the 192, rule B on 20. `full` is the share of questions at
coverage 1.0, `zero` the share at 0.0; tokens are the rendered excerpt block.

| row | n | mean cov | full | zero | median tok | SSA cov | SSA full | other cov |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| a. CE@200 user-only (sanity) | 192 | 0.8251 | 77.6% | 13.0% | 682 | 0.045 | 4.5% | 0.9260 |
| b. intent-routed (rule A), CE@200, assistant-only | 192 | **0.9032** | 85.4% | 5.2% | 676 | 0.727 | 72.7% | 0.9260 |
| b2. intent-routed (rule B), CE@200, assistant-only | 192 | 0.9032 | 85.4% | 5.2% | 676 | 0.727 | 72.7% | 0.9260 |
| c. intent-routed (rule A), CE@200, assistant-first all roles | 192 | 0.9032 | 85.4% | 5.2% | 676 | 0.727 | 72.7% | 0.9260 |
| c2. intent-routed (rule B), CE@200, assistant-first all roles | 192 | 0.9032 | 85.4% | 5.2% | 676 | 0.727 | 72.7% | 0.9260 |
| d1. intent-routed (rule A), pool order, assistant-only | 192 | 0.8773 | 80.7% | 6.8% | 677 | 0.773 | 77.3% | 0.8908 |
| d2. intent-routed (rule A), pool order, assistant-first | 192 | 0.8773 | 80.7% | 6.8% | 677 | 0.773 | 77.3% | 0.8908 |
| e1. oracle intent, CE@200, assistant-only | 192 | **0.9136** | 86.5% | 4.2% | 677 | 0.818 | 81.8% | 0.9260 |
| e2. oracle intent, CE@200, assistant-first | 192 | 0.9136 | 86.5% | 4.2% | 677 | 0.818 | 81.8% | 0.9260 |
| _reference: control's own 15 turns_ | 192 | _0.9056_ | _83.3%_ | _3.6%_ | _3938_ | _1.000_ | _100%_ | _0.8934_ |

Rows (c) and (c2) are identical to (b) and (b2) on **every question**, not just in aggregate
(`b_equals_c_per_question: true` in `results.json`). Putting assistant turns first and letting
user turns follow changes nothing because the budget never reaches them: on all 19 routed
questions the greedy fill kept **zero** user turns. Assistant turns are long enough that two or
three of them exhaust 720 tokens on their own.

### How much of the budget the routed questions get

| row | routed | fits ≥1 assistant turn | ≥2 | ≥3 |
| --- | ---: | ---: | ---: | ---: |
| b / c (rule A, CE order) | 19 | 100.0% | 84.2% | 42.1% |
| b2 / c2 (rule B, CE order) | 20 | 100.0% | 85.0% | 45.0% |
| d1 / d2 (rule A, pool order) | 19 | 100.0% | 73.7% | 31.6% |
| e1 / e2 (oracle intent, CE order) | 22 | 100.0% | 86.4% | 50.0% |

One assistant turn always fits. Three fit less than half the time — that is the whole story of
the budget at 720 tokens with turns that render at a median of 264.

### Mean coverage per question type

| row | single-session-user | single-session-assistant | single-session-preference | multi-session | knowledge-update | temporal-reasoning |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| _questions_ | 25 | 22 | 12 | 52 | 29 | 52 |
| a. CE@200 user-only (sanity) | 0.960 | 0.045 | 0.889 | 0.918 | 1.000 | 0.885 |
| b. rule A, CE@200, assistant-only | 0.960 | 0.727 | 0.889 | 0.918 | 1.000 | 0.885 |
| b2. rule B, CE@200, assistant-only | 0.960 | 0.727 | 0.889 | 0.918 | 1.000 | 0.885 |
| c. rule A, CE@200, assistant-first | 0.960 | 0.727 | 0.889 | 0.918 | 1.000 | 0.885 |
| c2. rule B, CE@200, assistant-first | 0.960 | 0.727 | 0.889 | 0.918 | 1.000 | 0.885 |
| d1. rule A, pool order, assistant-only | 0.920 | 0.773 | 0.889 | 0.832 | 1.000 | 0.875 |
| d2. rule A, pool order, assistant-first | 0.920 | 0.773 | 0.889 | 0.832 | 1.000 | 0.875 |
| e1. oracle intent, CE@200, assistant-only | 0.960 | 0.818 | 0.889 | 0.918 | 1.000 | 0.885 |
| e2. oracle intent, CE@200, assistant-first | 0.960 | 0.818 | 0.889 | 0.918 | 1.000 | 0.885 |

Outside single-session-assistant nothing moves: the rule never fires there, so every other
column is the user-only row it was before. The reranker is what separates rows b/c from d:
+0.035 mean coverage on the 170 other questions (0.9260 vs 0.8908).

### Median rendered tokens per question type

| row | single-session-user | single-session-assistant | single-session-preference | multi-session | knowledge-update | temporal-reasoning |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| a. CE@200 user-only (sanity) | 628 | 677 | 674 | 685 | 687 | 692 |
| b / c. rule A, CE@200 | 628 | 532 | 674 | 685 | 687 | 692 |
| b2 / c2. rule B, CE@200 | 628 | 530 | 674 | 685 | 687 | 692 |
| d1 / d2. rule A, pool order | 677 | 532 | 693 | 676 | 679 | 683 |
| e1 / e2. oracle intent, CE@200 | 628 | 532 | 674 | 685 | 687 | 692 |

Routing makes the routed questions *cheaper*, not dearer: 532 median tokens against 677,
because the fill stops on a long assistant turn rather than packing short user turns to the
line.

### The 22 single-session-assistant questions, one line each

`gold rank` is the position of the gold turn in the CE@200 order restricted to assistant
turns; `-` means the gold turn is not an assistant turn at all. `gold tok` is that turn's
rendered cost.

| qid | gold role | rule A | rule B | gold rank | gold tok | cov a | cov b | cov oracle | turns kept (b) |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1568498a | assistant | - | - | 2 | 25 | 0.00 | 0.00 | 1.00 | 3 |
| 16c90bf4 | assistant | fires | fires | 1 | 119 | 0.00 | 1.00 | 1.00 | 2 |
| 1d4da289 | assistant | fires | fires | 2 | 262 | 0.00 | 1.00 | 1.00 | 3 |
| 2bf43736 | assistant | fires | fires | 1 | 138 | 0.00 | 1.00 | 1.00 | 2 |
| 3249768e | assistant | fires | fires | 2 | 305 | 0.00 | 1.00 | 1.00 | 2 |
| 41275add | assistant | fires | fires | 1 | 242 | 0.00 | 1.00 | 1.00 | 2 |
| 51b23612 | assistant | fires | fires | 1 | 380 | 0.00 | 1.00 | 1.00 | 3 |
| 561fabcd | **user** | - | fires | - | 74 | 0.00 | 0.00 | 0.00 | 11 |
| 6222b6eb | assistant | fires | fires | 2 | 379 | 0.00 | 0.00 | 0.00 | 1 |
| 7e00a6cb | assistant | fires | fires | 1 | 232 | 0.00 | 1.00 | 1.00 | 3 |
| 89527b6b | assistant | fires | fires | 1 | 592 | 0.00 | 1.00 | 1.00 | 1 |
| 8aef76bc | assistant | fires | fires | 1 | 256 | 0.00 | 1.00 | 1.00 | 3 |
| 8b9d4367 | assistant | fires | fires | 1 | 325 | 0.00 | 1.00 | 1.00 | 2 |
| 8cf51dda | assistant | fires | fires | 1 | 378 | 0.00 | 1.00 | 1.00 | 3 |
| a40e080f | assistant | fires | fires | 1 | 237 | 0.00 | 1.00 | 1.00 | 3 |
| ac031881 | assistant | - | - | 1 | 99 | 0.00 | 0.00 | 1.00 | 11 |
| b759caee | assistant | fires | fires | 1 | 520 | 0.00 | 1.00 | 1.00 | 1 |
| c4f10528 | assistant | fires | fires | 15 | 260 | 0.00 | 0.00 | 0.00 | 2 |
| d596882b | assistant | fires | fires | 1 | 231 | 0.00 | 1.00 | 1.00 | 2 |
| dc439ea3 | **user** | fires | fires | - | 46 | **1.00** | **0.00** | 0.00 | 3 |
| e9327a54 | assistant | fires | fires | 1 | 338 | 0.00 | 1.00 | 1.00 | 2 |
| fea54f57 | assistant | fires | fires | 1 | 263 | 0.00 | 1.00 | 1.00 | 3 |

Every gold turn is present in the pool for all 22 — nothing here is a retrieval-depth failure.

## Answers

**Does a query-conditional rule reach the control's 0.906 at 720 tokens?** Effectively yes, on
5.8x less context. Routing on rule A scores **0.9032** mean coverage at a median of **676
tokens**; the control scores 0.9056 at 3,938. The gap is 0.0024 — under half a question out of
192. The two rows get there differently: the control takes every single-session-assistant
question (1.000 against 0.727) and the routed row takes multi-session (0.918 vs 0.825),
temporal-reasoning (0.885 vs 0.862) and single-session-preference (0.889 vs 0.847). The routed
row is ahead on share-at-full-coverage (85.4% vs 83.3%) and behind on share-at-zero (5.2% vs
3.6%). The honest statement is a tie on mean coverage at a fifth of the context, not a win.

**How far is it from the oracle?** 0.0104 mean coverage — 0.9032 against 0.9136, exactly two
questions of the 192 (`1568498a`, `ac031881`). A perfect intent classifier is worth two
questions here, and the rule already routes 18 of the 20 questions the oracle routes usefully
(the other 2 of the oracle's 22 have user gold turns, where routing helps nobody). Rule B does
not close the gap: on the sample it adds only `561fabcd`, a user-gold question where firing
changes nothing, which is why rows b2/c2 match b/c to four decimals. Rule B's extra recall is
real on the fitting set — it catches all three of rule A's misses there — but it buys no
coverage on the held-out sample.

**What is the residual made of?** Take the 6 single-session-assistant questions that routing
still leaves short, and it splits four ways:

1. **Rule misses — 2 questions** (`1568498a`, `ac031881`). Both would be full coverage under
   the oracle. This is the only part a better rule can recover, and it is worth +0.0104
   overall.
2. **Mislabelled role — 2 questions** (`561fabcd`, `dc439ea3`). Their gold turn is a *user*
   turn despite the single-session-assistant label, so role routing cannot help and the oracle
   scores 0.00 on both too. `dc439ea3` is the one place routing actively costs: user-only found
   its gold turn (coverage 1.00) and the rule sends it to assistant turns, where the answer is
   not (0.00). That single question is the whole reason 16 gains net out to 15.
3. **Budget — 1 question** (`6222b6eb`). The gold turn sits at assistant-rank 2 and renders at
   379 tokens; the rank-1 turn costs 349, so the block stands at 362 tokens with 358 left and
   the gold turn misses by 21. Nothing about the rule or the ranking is wrong here — the budget
   is.
4. **Ranking — 1 question** (`c4f10528`). The gold assistant turn is in the pool but at
   assistant-rank 15 under the cross-encoder, far past anything three turns of budget can
   reach.

So of the 0.0968 that separates row (b) from perfect coverage, only about a tenth of it
(0.0104) is the rule's fault. The rest is budget and ranking on the 170 non-assistant
questions plus the four irreducible cases above. Two more things follow from the tables. The
assistant-first variant is not worth building: it never once selected a user turn, so it is
row (b) under a different name. And the reranker earns its place on the questions it was
already earning it on — dropping to pool order costs 0.0259 overall (0.8773 vs 0.9032), all of
it on the 170 non-assistant questions.

---

## Part 2 — verification

**VERDICT: holds-with-corrections.** Every number in the builder's report reproduces from my own code. The measurements are sound; three inferences drawn from them are overstated.

## What I ran

```
cd local/intent/verify
python3 verify_rule.py            # rule metrics both splits + atom-level tuning audit + negative probe
python3 verify_rows.py            # rows a/b/e1 + control, re-implemented; false-fire cost; handcheck.txt
python3 verify_residual.py        # the 22 SSA questions: gold-in-pool, ranks, budget arithmetic
python3 verify_generalisation.py  # where the load-bearing cues occur; precision margin
python3 verify_claims.py          # paired bootstrap; type-mix sensitivity
python3 verify_failsafe.py        # is assistant-first a safety net for a false fire
python3 verify_minimal.py         # strip the rule to fit-justified alternatives, rescore
```
Files under `local/intent/verify/` (`.py` + matching `.out`, `handcheck.txt`, `verify_rows.json`). Nothing outside `local/` touched; `git status` clean. No imports from the builder's code — rendering, o200k_base counting, greedy fill, gold matching, both orderings rewritten from the stated conventions.

## Reproduction: exact, no corrections

| quantity | builder | mine |
|---|---|---|
| draft rule (A1+A2), all 500 | P 1.000 R 0.446 | P 1.0000 R 0.4464 |
| rule A fit(301) / held-out(199) | 31/34, 19/22, 0 FP | identical, same 3+3 FN qids |
| rule B fit / held-out | 34/34, 20/22 | identical |
| row a | 0.8251 / 77.6% / 13.0% / 682 / other 0.9260 | identical |
| row b | 0.9032 / 85.4% / 5.2% / 676 / SSA 0.7273 | identical |
| row e1 | 0.9136 / 86.5% / 4.2% / 677 / SSA 0.8182 | identical |
| control | 0.9056 / 83.3% / 3.6% / other 0.8934 | identical (median tok 3937.5) |
| residual: 22/22 gold in pool; `6222b6eb` misses by 21 tok; `c4f10528` at assistant-rank 15; `561fabcd`/`dc439ea3` user gold; 16 gains −1 loss = 15; b≡c per question | as claimed | all confirmed (741 > 720 on `6222b6eb`; 0 user turns kept on 19/19 routed) |

Hand check on 5 SSA questions (`16c90bf4`, `1d4da289`, `2bf43736`, `3249768e`, `6222b6eb`): rendered blocks and gold turns printed in `handcheck.txt`; my by-hand gold-in-block count equals the script's coverage in all 5.

**No tuning on the 199.** Atom-level ablation over all 109 concrete alternatives: zero atoms are the unique matcher for a held-out positive while firing on no fitting-set positive. Two atoms fire on a held-out positive without fit justification (`\byou said\b`→`fea54f57`, `\bwhich .{0,40}\bdid you\b`→`dc439ea3`); both are redundant. `qmeta.json` is byte-identical to the dataset; `heldout_dump.txt` (misnamed — it holds the *fitting* 301) contains 0 sample qids.

## Findings

**F1 — HIGH. Precision 1.0 measures the benchmark's question templates, not the intent.** Of 444 negatives, 15 (3.4%) contain "you"/"your" at all, 0 contain "your", 0 contain the discourse frame — and all 15 are single-session-preference present-tense requests ("Can you suggest a hotel for my upcoming trip to Miami?"). `remind me` occurs in 42 of 500 questions, **all 42 single-session-assistant**; a second-person past-tense speech verb occurs in 30 of 500, **all 30 single-session-assistant**. A9 alone fires on 17 of the 22 held-out positives. The rule survives the only 15 negatives that address the assistant purely on tense: `\b(?:can|could|would|will) you (?:suggest|recommend|…)\b`, a pattern a reasonable author would write, adds **9 false positives and 0 true positives**. "Remind me what the dose was" carries no information about who said it; it works here because LongMemEval never writes that sentence for a user-turn answer.

**F2 — HIGH. The cost of a false fire was never measured, and it is near-total.** Routing a non-SSA question to assistant turns: mean coverage **0.9260 → 0.0078**, loss **0.9181**, zero coverage in **163 of 170**. Gain per true fire is **+0.7895** (15 net over 19 fires). Break-even is **0.86 false fires per true fire — the rule needs precision above ~54%**. On this distribution 0/444 gives a rule-of-three bound of ≤0.68% FP rate, so it is safe here with wide margin; off it the margin is unmeasured and the failure is silent.

**F3 — HIGH. "Reaches the control" is a knife-edge on the benchmark's type mix.** Both rows are separable — routing 0.7273/0.9260, control 1.0000/0.8934 — so they **cross at an assistant-question share of 0.1066**; this sample sits at 0.1146. At 5% share routing is +0.0173 ahead, at 20% it is −0.0285 behind, at 30% −0.0590. The reported −0.0024 is where LongMemEval happens to sit, not a property of the method.

**F4 — MEDIUM. The tie is not measurable at n=192.** Paired bootstrap, b − control: **−0.0024, 95% CI [−0.0372, +0.0308]**, 16 wins / 11 losses / 165 ties. The interval is 14× the point estimate. The claim that *is* solid: b − a = **+0.0781, CI [+0.0365, +0.1198]**, 16 wins / 1 loss.

**F5 — MEDIUM. Assistant-first is not a failsafe, and the report frames its dismissal wrongly.** Measured on the 170 questions a false fire would hit: assistant-first scores **0.0078, identical to assistant-only**, and the fill reaches a user turn on **0 of 170**. So at 720 tokens there is no graceful-degradation variant; the design is all-or-nothing on the classifier. The report notes only that assistant-first "is not worth building".

**F6 — MEDIUM. 95 of the rule's 109 alternatives are untested surface area.** **14 alternatives reproduce every reported number on both splits** (fit 31/34 P1.0, held-out 19/22 P1.0): `you suggested|recommended|mentioned|told me|provided|wrote|created`, `did you say`, `can you remind me`, `remind me what|which|who|how|of`. Four whole patterns — A2 (the spec's own draft), A4, A6, A10 (38 noun alternatives) — fire on **zero positives in either split**. The description "the spec's draft plus the phrasings the 301 fitting questions show" is inaccurate for A4, A6, A10 and most of A3/A7: written from intuition, not fitted. This is exactly where an unmeasured false positive would come from.

**F7 — LOW. Coverage understates `6222b6eb`.** The one assistant turn that does fit states the answer ("6S … is implemented in the SIAC_GEE tool") but is not the `has_answer` turn, so it scores 0.00. The "budget miss" category is softer than it reads.

**F8 — COSMETIC.** Control median tokens is 3937.5; `intent/results.md` prints 3938, the spec quotes 3,937.

## Corrected conclusion

Query-conditional role routing is a real and well-measured win over user-only selection: it turns 16 of the 22 single-session-assistant questions from zero to full gold coverage and costs one, a net **+0.0781 mean coverage (CI +0.037 to +0.120) at 676 median tokens**, and the rule that triggers it was frozen on a disjoint 301 and did not touch the held-out 199. It does **not** reach the control in any defensible sense: the −0.0024 gap is unmeasurable at this sample size (CI ±0.035) and sits 0.008 from the point where the two curves cross as the assistant-question share moves, so it reports the benchmark's type mix rather than the method. And the rule generalises no further than LongMemEval's phrasing: 14 alternatives carry all of it, "remind me" and second-person past-tense speech verbs appear in that benchmark **only** in the class being detected, and every negative that addresses the assistant at all is a present-tense request from one question type. Since a false fire costs 0.918 coverage against a 0.790 gain — with no failsafe, because the budget never reaches the user turns behind the assistant ones — the design needs ~54% precision to break even, and nothing here measures what its precision would be on real user text.
