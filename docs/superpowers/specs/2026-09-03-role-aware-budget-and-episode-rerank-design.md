# Role-aware selection under a token budget — measured, and what to judge next

**Date:** 2026-09-03, second version. The first version of this document proposed a design
from arithmetic; an adversarial review found its target unreachable and its proxy unvalidated.
This version is written from three offline measurements made the same day, each verified by
a second pass, and it proposes two judged arms. Nothing in it is built.

**Target, as decided:** median injected context at or under **720 tokens** by the harness's own
count, with judged accuracy held at or above **86.4%** — the cap-15 control's number on the
199-question stratified sample (seed `20260903`), reader and judge **gpt-5.4** as the control's
checkpoint records. 95% was dropped: category-for-category parity with the published
competitor is 94.6% on this dataset's mix, and no route offered reaches it; it stays
conditional on question-side retrieval, which is deferred.

The three findings this rests on, all on this branch:

- `docs/superpowers/plans/2026-09-03-coverage-proxy-validation.md` — the free proxy, and its limits
- `docs/superpowers/plans/2026-09-03-widening-arm-offline.md` — what selection buys at 720 tokens
- `docs/superpowers/plans/2026-09-03-intent-routed-selection-offline.md` — the query-conditional rule

## What is established

**Gold-turn coverage predicts judged correctness, causally.** Over ten judged arms, 93.0% of
answers are correct when every `has_answer` turn was retrieved, 26.1% when some were, 23.1%
when none; AUC 0.866 (cluster-bootstrap 0.835–0.897); the odds ratio rises to 49.7 when
stratified by question; and on questions where cap 15 and cap 3 retrieved the same gold turns,
the twelve extra turns bought 2.1 points, where cap 15 covered more they bought 62.2.

**Coverage is blind to rendering.** Three arms with byte-identical retrieval span 87.4% to
84.9% accuracy at 3,700 to 2,028 tokens at one coverage value, and under a 720-token cap
coverage is maximised by cutting every turn to about 120 characters. So the sweep objective is
coverage of gold turns **on the context as rendered at the budget**, truncation is gated on
gold-token retention rather than optimised, and no two configurations within five coverage
points are ordered without a judge.

**The budget is never the constraint; selection is.** Gold turns alone render at 165 median
tokens. Today's ranker puts them at median rank 2, so a 720-token fill of the shipped order
admits 2.5 turns and covers 0.602 of gold — the cap-3 arm's 54.4% is what that buys.

**Selecting on role is worth 7.6 times reranking.** At a 720-token turns block: shipped order
0.602, user turns only 0.799, reranked user-only 0.825, against the control's 0.906 at 3,938
tokens. On the 170 questions that are not `single-session-assistant`, the 720-token user-only
arm covers **more** gold than the 15-turn control: 0.926 against 0.893.

**Widening the pool and reranking it buys nothing at 15 turns.** +0.009 coverage, interval
[−0.015, +0.032], tied on 173 of 192 questions, 420 more tokens. The cross-encoder's effect
at 720 tokens is real and small: +0.026, all of it on the non-assistant questions.

**Two repairs for the assistant-answer questions are refuted.** A 20-word stub keeps the
answer string in 3 of 23 gold assistant turns. A fixed quota of assistant turns is worse than
user-only at every setting, because a gold assistant turn renders at a median of 264 tokens:
one of them buys 15 assistant questions and costs 14 multi-session and 11 temporal ones.

**Routing by the question recovers them.** A deterministic rule on the question text — second-
person past-tense speech verbs, "remind me", "what did you" — fitted on the 301 questions
outside the sample and frozen fires on 19 of the 22 `single-session-assistant` questions and
none of the other 177. Routing fired questions to assistant turns and the rest to user turns,
in cross-encoder order under a 720-token turns block, gives **0.903 mean coverage at 676 median
tokens** — +0.078 over user-only (interval +0.037 to +0.120), 16 questions from zero to full
coverage, one lost. Oracle routing by the true label gives 0.914.

**And what routing does not establish.** Against the control's 0.906 the gap is −0.002 with an
interval of ±0.035 — not measurable at this sample. The two rows cross at an assistant-question
share of 10.7%; the sample sits at 11.5%, so the tie reports the benchmark's type mix. The rule
generalises no further than this benchmark's phrasing: "remind me" occurs 42 times in 500
questions, every one in the class being detected, and every negative that addresses the
assistant at all is a present-tense request. A false fire costs 0.918 coverage against a 0.790
gain with no failsafe at 720 tokens, so the design needs about 54% precision to break even, and
its precision on real user text is unmeasured.

## The design, reduced to what survived

For the benchmark arm, every piece lives where it can be measured, and none of it ships in
memvara core from this work.

**Selection.** Search at `k=200` against a server with `read_max_episodes=200` and the
cross-encoder configured (`read_reranker`, `read_rerank_top_n=200`), so the API returns the
pool in reranked order. The harness keeps user turns only, unless the routing rule fires on the
question, in which case it keeps assistant turns only. It fills a 720-token turns block
greedily in the returned order, encoding the whole rendered block with `o200k_base`, stopping
at the first line that would exceed the budget and always keeping the first item. Claims are
dropped (`MEMVARA_TURNS_ONLY=1`): they cost a median of 130 tokens and were measured to
contribute nothing.

**Nothing is truncated**, so the retention gate does not bind and the judged number means what
it says.

**Dropped from the first version, with the measurement that dropped it:** stub rendering (3 of
23); fixed assistant quotas (worse at every setting); widening for its own sake (no effect at
15); the token budget and role weights inside `HybridRetriever` (the budget has to be computed
on the rendered text the harness produces, and core has no tokenizer — the existing
`recall(counter=)` seam is the right home if it ever moves into core, and that is a product
decision to take after a judged result exists).

**The product boundary, stated plainly.** The routing rule is fitted to this benchmark's
question templates and its false-fire cost is near-total. It is a measured configuration for
the benchmark arm, not a feature. What would make it product-worthy is a precision measurement
on real user questions and a failsafe that the 720-token budget does not allow; neither exists.

## The judged arms

Two arms on the same 199 questions as the control, same ingest, `gpt-5.4` reader and judge,
`SKIP_RETRIEVAL_EVAL=1`, paired per question, McNemar exact test. Each is a copied checkpoint
with `dataSourceRunId` kept and the search, answer and evaluate phases reset, per the stack
notes.

| arm | selection | what it answers |
| --- | --- | --- |
| **routed-720** | routing rule; cross-encoder order; 720-token turns block; no claims | does the target hold |
| **user-only-720** | user turns only; cross-encoder order; 720-token block; no claims | is the role model right |

The second arm is the falsifier. If it lands at or above 84% overall, the assistant-answer
questions are being answered without assistant turns and the role model behind every
measurement here is wrong.

**Predictions, stated before either run.**

- routed-720: overall **86%, range 82–89%**; `single-session-assistant` **15–17 of 22** (control
  22); multi-session **44–46 of 53** (control 40); temporal **45–47 of 53** (control 44);
  recorded median context **about 690 tokens**, under 720 because claims are dropped.
- user-only-720: overall **77%, range 75–81%**; `single-session-assistant` 0–4 of 22; on the 177
  other questions **86%, range 84–89%** against the control's 84.7% there.
- routed minus user-only: about **+9 points overall**, almost entirely the assistant column.

**Stop rules, with what the sample can see.** 199 paired questions detect roughly an 8-point
drop at 80% power and nothing finer; every rule below is written on that.

1. routed-720 overall below **82%** — the arm has lost more than the coverage predicted and the
   coverage-to-accuracy map does not hold at this budget; stop and diagnose per category.
2. routed-720 `single-session-assistant` below **12 of 22** — the rule's recall does not survive
   the reader; the residual is the rule, not the budget.
3. routed-720 recorded median above **720** — the budget was mis-set; rerun with a 590-token
   block, which the widening study measured at 0.805 coverage.
4. user-only-720 at or above **84%** — the role model is wrong; every conclusion above is reopened.

**Then the 500.** If routed-720 clears rule 1, the claim "accuracy held at 86.4%" still cannot
be made on 199 questions: a 3-point non-inferiority margin needs about 489 paired questions.
The full-500 run compares routed-720 against the shipped cap-3 baseline that already exists on
all 500 — the like-for-like comparison at the same budget, 54.4% today — and against the
control unpaired. A 500-question cap-15 control does not exist and is not proposed; the
paired margin is stated against what has been run.

## Build

**Server** (`memvara-cloud`, uncommitted `BENCHMARK-LOCAL` lines in both `asgi.py:1371` and
`memories.py::_clone`, because the per-tenant clone drops every `read_*` option — filed as a
defect): `read_max_episodes=200`, `read_reranker=CrossEncoderReranker()`,
`read_rerank_top_n=200`. Rebuild from the clean core at `6c0ff6c` with the compose env the
stack notes list. The image already carries `sentence-transformers`, so the embedder does not
change; the arm records `all-MiniLM-L6-v2` and the core sha.

**Harness** (`memorybench`, provider `memvara`, on its branch, with tests): `SEARCH_K` read from
`MEMVARA_SEARCH_K` (200); `MEMVARA_ROLE_ROUTE=1` applying the frozen rule — the fourteen
alternatives that carry all of its measured behaviour, not the 109 — in `prompts.ts`;
`MEMVARA_TOKEN_BUDGET=720` with the greedy fill using the harness's own `js-tiktoken`
`o200k_base` so the count is the count it reports; the returned order preserved. Each knob off
by default so the shipped provider is unchanged.

**Sweep tooling** is not part of this step. The cached pool and cross-encoder scores under
`local/pool/` are the sweep's inputs when there is a judged result to calibrate it against.

## Cost

Each 199-question arm is 199 reader calls at about 700 tokens of context plus 199 judge calls;
well inside one gateway key's budget. The 500-question arm is two and a half times that and
should be ordered by question id so a partial run is still a stratified prefix.

## Deliberately deferred

- Question-side retrieval for multi-session aggregation — the only route to 95%.
- The sweep over fill rules, budgets and fusion weights — after a judged calibration point.
- Moving selection into memvara core — after a judged result and a precision measurement on
  real questions; the `recall(counter=)` seam is the home.
- A second corpus. Every constant here was chosen on the corpus it is evaluated on.

## Risks

- **The rule is the benchmark's phrasing.** Stated above; the falsifier arm and the product
  boundary are the response. Do not generalise from a routed-720 result to a product feature.
- **The proxy-to-accuracy map at 720 tokens is extrapolated.** It was measured across cap 3 to
  cap 15 on whole turns; a 700-token context of user turns is a different shape. Rule 1 exists
  for this.
- **Abstention questions.** Seven of the 199 have no gold turn and the control answers all of
  them; with a fifth of the context the reader may abstain differently. Reported separately.
- **Two runs marked `running` from 2026-09-02 are dead**, not live; restarting the stack under
  them was checked and is safe.
