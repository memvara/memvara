# Role-aware selection, a token budget, and reranking the episode pool

**Date:** 2026-09-03
**Status:** design, awaiting review. Nothing here is built.
**Target:** LLM-judged answer accuracy on LongMemEval-S inside Supermemory's MemoryBench
harness, GPT-4o reader and judge, on the 199-question stratified sample (seed `20260903`)
every prior arm used. Two numbers, both required, the second the real goal:

1. accuracy at or above **95%** (today 86.4%);
2. median injected context at or under **720 tokens** by the harness's own count
   (today 4,089), which is roughly what Supermemory reports.

This document was written by one author without the adversarial review it was meant to
get: the design panel was launched three times and every agent died on API overload. The
offline sweep and the judged arms below are the check that review would have been. Read
the risks section with that in mind.

## The finding this rests on

memvara hands the reader whole turns. Every one of the six selection interventions
measured so far varied *which* whole turns or *how many*, and none beat the token curve.
Nobody varied what a slot costs or which role gets one. The dataset says that is where the
gap is:

| | share of haystack tokens | share of gold turns |
| --- | ---: | ---: |
| user turns, 43 words on average | 14% | **94%** (842 of 896) |
| assistant turns, 273 words on average | **86%** | 6% (54 of 896) |

"Gold turn" throughout means a turn the dataset marks `has_answer: true`. There are 896
across 500 questions, a median of two per question, and 51 of the 54 assistant ones belong
to `single-session-assistant`, which memvara already answers at 100%. For the five
categories with any gap, the evidence is in user turns essentially without exception:
multi-session 326 user to 1 assistant, temporal 259 to 0, knowledge-update 144 to 0,
preference 44 to 0.

So a cap of fifteen whole turns spends most of 4,089 tokens on turns that cannot hold the
answer, and because those turns are six times longer they also outscore user turns for
slots on lexical and vector legs alike. The cost is a ranking defect as much as a token one.

The pure evidence is small: two gold user turns at about 70 tokens each is roughly 140
tokens. A budget of 720 buys eight to ten whole user turns, or six with a short stub of
each one's assistant reply. Either is more gold-bearing slots than today's fifteen. The
mechanism is the one the score floor tried to use — spend by need rather than by count —
except that a floor could only stop earlier, and this can go deeper for the same tokens.

### The counter-claim, and why it does not hold

The harness carries a comment (`src/providers/memvara/prompts.ts`) arguing assistant turns
cannot be dropped because "the answer text sits in an assistant turn 69% of the time". That
was measured by string presence over retrieved turns, on a run pre-filtered to questions
whose answer text had been retrieved. Assistant turns are six times longer and match strings
by volume. Against the gold labels the picture inverts: of 654 gold user turns with an answer
longer than two characters, the answer string appears in the following assistant turn 12% of
the time, and *only* there 2% of the time. The assistant echoes the user; the evidence is
the user's turn. This is stated as a prediction below rather than assumed, because the
judged arm is what settles it.

## What already exists and is not pointed at this benchmark

Three of the four pieces are in the tree, built and measured elsewhere.

- **Role at scoring time.** `Episode.role` is a stored column, and `HybridRetriever._episodes`
  (`memvara/retrieve/hybrid.py`) computes `score = evidence * self.w_episode` with the
  episode in hand. A role multiplier is a one-line hook.
- **A cross-encoder reranker.** `memvara/rerank/` ships a `Reranker` protocol, a
  `CrossEncoderReranker` behind the `memvara[rerank]` extra (`ms-marco-MiniLM-L-6-v2`), and
  two model-free controls. On LOCOMO at `top_n=20` it moves R@12 from 62.0 to 66.5 and R@1
  from 30.5 to 44.9. It has never been run on LongMemEval, and it is structurally unable to
  help here: `_episodes` cuts the pool to `max_episodes` before returning, and the reranker
  runs after `_interleave`, so it can reorder the fifteen it is handed and cannot promote a
  gold turn from rank forty.
- **Rank-aware truncation in the harness.** `MEMVARA_HEAD_WHOLE` and `MEMVARA_TAIL_CHARS`
  render the top N turns whole and cut the rest. Measured only by string presence, never
  judged, off by default.
- **An offline grader.** `bench/evalkit.score_retrieval` grades a retrieval against
  `has_answer` without the flag ever reaching ingest or the query path. The sweep builds on
  that separation.

## Design A: role-aware selection under a token budget

Three changes in `HybridRetriever`, all off by default so the shipped behaviour is
unchanged, plus one rendering change in the harness.

### A1. A role multiplier on episode scores

New constructor argument `episode_role_weights: Mapping[str, float] | None = None`. When
set, `_episodes` multiplies each candidate's score by `weights.get(episode.role, 1.0)`
immediately after `score = evidence * self.w_episode`, before the `min_score` check and the
sort. `None` is today's behaviour exactly. A weight of `0.0` removes that role from the pool
rather than merely demoting it, which is deliberate: the sweep needs the "user turns only"
corner to exist.

Validation: every value must be a finite float in `[0, 1]`, keys must be non-empty strings;
anything else raises `ValueError` at construction with a message that says what was passed.
The same shape as the guard `episode_score_floor` already has.

### A2. A token budget as the cap

New constructor argument `max_episode_tokens: int | None = None`. When set, the episode cut
that today is `out[:self.max_episodes]` becomes: walk the sorted pool, keep an episode while
the running total plus its cost stays within the budget, stop at the first one that does not
fit. `max_episodes` still applies as an outer ceiling. The best-scoring episode is always
kept, even if it alone exceeds the budget, for the reason the floor keeps it: an empty answer
is worse than a thin one.

Cost accounting in core is Unicode code points divided by four, the convention
`memory_recall`'s budget already uses, because core ships with no tokenizer. That
undercounts CJK by about 2.3× (measured, see the recall budget note); the sweep reports the
ratio between this count and the harness's tiktoken count on this corpus so the budget can
be calibrated to land at 720 by the harness's measure. Pinning the ratio is part of step 1,
not a guess made here.

### A3. An intent rule for assistant-authored answers

`single-session-assistant` asks what the assistant said, and its gold is assistant turns. A
role weight that demotes assistant turns would trade a category at 100% for the others. The
rule is deterministic and reads the query only: if the question matches a small pattern set
— the second person addressing the assistant with a speech or recommendation verb
(`you (suggested|recommended|proposed|listed|mentioned|said|told me|gave me|shared)`, and
"your (suggestion|recommendation|list|advice)") — the role weights are inverted for that
query: assistant weight becomes 1.0 and user weight becomes the configured assistant weight.
New constructor argument `episode_role_intent: bool = True`, meaningful only when
`episode_role_weights` is set. The dataset reader in step 1 measures the rule's precision and
recall against `question_type` before anything is built on it; if either is below 0.9 the
pattern set is revised there.

### A4. Rendering assistant turns as stubs

In the harness (`prompts.ts`), a role-aware variant of the existing truncation: user turns
render whole; assistant turns render as their first `MEMVARA_ASSISTANT_STUB_WORDS` words
followed by an ellipsis, default off. This is the harness's responsibility because the
harness renders; memvara returns turns. Porting the same policy into memvara's own renderers
(`recall` text, the MCP tools) is a named follow-up, in scope for the product and out of
scope for this measurement, so that a stub policy is not shipped to every memvara user on the
strength of one benchmark.

Rendering keeps rank order, as today. Knowledge-update questions need the latest value; each
turn already carries its date, and the reader is told to prefer later dates. The sweep
includes a "sort by date within the budget" variant so that choice is measured rather than
assumed.

## Design B: rerank the episode pool before the cut

One structural change so the existing reranker can reach episodes, and one combination rule.

### B1. Widen the pool the reranker sees

When `self.reranker` is set, `_episodes` returns `out[:max(self.max_episodes,
self.rerank_top_n)]` instead of `out[:self.max_episodes]`, and `_interleave` receives that
wider list. After `rerank(...)` reorders the head, a final episode cut applies `max_episodes`
and, when set, `max_episode_tokens` to the episodes in the reranked list. With no reranker
configured, nothing changes: the cut stays where it is and the same fifteen come back.

The reranker scores `(query, episode.content)`. Role is not in the text it sees; role is
handled by the multiplier in the next section, so the two effects stay separately
attributable.

### B2. How rerank score and role weight combine

For an episode, the ordering key after reranking is `rerank_score * role_weight`. The
multiplier is applied after the model scores so that an assistant turn must beat a user
turn by a margin the sweep chooses rather than one hardcoded here. `Explanation.rerank_score`
records the model's number unmultiplied, as it does for claims, so a reader of the
explanation can see both.

`rerank_top_n` is the depth of the pool the cross-encoder scores. The sweep tries 50, 100
and 200. Cost scales linearly: roughly 84 ms per query at 20 on the LOCOMO measurement, so
200 is on the order of a second per query. That is acceptable for a benchmark arm and is
not a default for anyone.

## Step 1: the offline sweep

The user asked for a very large number of experiments before concluding the direction was
wrong. Judged arms cost about 200 reader calls and 200 judge calls each; the honest way to
run tens of thousands of experiments is to make the objective free.

### Objective

For a configuration, on each of the 199 questions: render the context the configuration
would produce, count its tokens with tiktoken `o200k_base` (what the harness uses for
GPT-4o), and record which gold turns it contains. Report, per category and overall:

- gold-turn coverage: the fraction of a question's gold turns present in the context;
- questions with full coverage; questions with zero coverage;
- median and p90 tokens.

Accuracy is a smooth function of coverage (measured: 66.7% at 0–25% coverage of gold-session
content, 94.2% at 60% and above), so coverage at a budget is the proxy. What it cannot see
is stated below.

### Cached inputs, computed once

1. **One wide pull per question.** Against the same store the judged arms will run on,
   `search(question, k=200, include_episodes=True)` and cache, per candidate: episode id,
   role, timestamp, content, fusion score, and each leg's rank and score from `Explanation`.
   Claims are cached alongside but rendered only in configurations that include them; the
   measured contribution of claims on this benchmark is zero and the default sweep leaves
   them out.
2. **Cross-encoder scores over the cached pool.** `CrossEncoderReranker` from the
   `memvara[rerank]` extra, scoring the 199 × 200 pairs once locally — minutes on a laptop CPU
   — and cached beside the pool. Configurations without a reranker ignore the column.
3. **Gold labels** from the dataset via `bench/evalkit`, never from the store.

After this, every configuration is arithmetic over the cache and runs in milliseconds.

### The grid

| axis | values | count |
| --- | --- | ---: |
| assistant role weight | 0, 0.1, 0.2, … , 1.0 | 11 |
| intent rule | off, on | 2 |
| assistant stub length (words) | 0, 20, 40, 80, whole | 5 |
| token budget | 400, 500, 600, 720, 900, 1200 | 6 |
| reranker | none, cross@50, cross@100, cross@200 | 4 |
| score floor `T` | 0, 0.45, 0.55, 0.65 | 4 |
| fusion weights (vector, lexical, temporal) | shipped default; vector ×1.5; lexical ×1.5; temporal ×2; temporal 0 | 5 |
| ordering within the budget | by score, by date | 2 |

That is 11 × 2 × 5 × 6 × 4 × 4 × 5 × 2 = **105,600 configurations**, each evaluated on 199
questions from cache. An MMR diversity axis is deliberately left out of the first grid: with
user turns only 43 words long, near-duplicates are rare, and adding a third continuous axis
before the first two are understood makes the front harder to read. It is the first thing
to add if multi-session coverage stalls.

Fusion presets are re-scored from the cached per-leg ranks with `reciprocal_rank_fusion`, the
same function the retriever uses, so a preset in the sweep is the number a retriever
configured that way would produce.

### Output

The coverage-versus-tokens Pareto front, per category and overall, as a table and a plot
committed with the run; the single best configuration at each budget; and the tiktoken to
code-point ratio on this corpus, which sets `max_episode_tokens` for the judged arm.

### Stop rule, stated now

If no configuration reaches at least 60% gold-turn coverage for multi-session at a 720
budget, selection and rendering alone cannot reach 93% on that category, and question-side
retrieval (deferred, below) moves from deferred to next. Steps 2 and 3 still run, because
the token target and the other five categories are reachable without it, and the accuracy
number they report is stated with that category's shortfall named rather than averaged
away.

## Steps 2 and 3: judged confirmation

Both arms on the 199-question sample, GPT-4o reader and judge, same ingest as the cap-15
control, paired per question, McNemar exact test as before.

**Step 2 — Design A alone**, at the best A-only configuration the sweep finds at or under
720 tiktoken median. Prediction, stated before the run: median tokens at or under 720;
accuracy not below 86.4%; single-session-assistant unchanged at 100% because of the intent
rule; multi-session and temporal up, because they now get more gold-bearing slots per token.
Stop rule: accuracy down more than three points means the echo counter-claim was right after
all — assistant text was doing work the labels do not show — and the repair is a longer stub,
not abandoning the direction.

**Step 3 — Designs A and B together**, at the best configuration with a reranker. Prediction:
multi-session and temporal each up at least five points over step 2, because the
cross-encoder promotes gold turns the fused ranking left past the cut. Stop rule: if neither
moves five points, the reranker adds nothing on conversational turns and B is dropped; A
stands on its own.

Each arm is separately attributable: A changes what wins slots and what a slot costs; B
changes only which candidates the cut sees. A run that ships both at once would not say which
one worked.

## What the sweep cannot see

- Whether the *reader* can answer from a stub. A forty-word stub of an assistant turn may
  cut a number the question needs. Coverage counts the gold turn as present whether or not
  the answer survived truncation, when the gold turn is an assistant one. Steps 2 and 3 exist
  for this.
- Judge behaviour. The judge accepts paraphrase; the sweep counts turns.
- Anything about knowledge-update ordering beyond the date-sort variant.

## Build

**memvara**, all behind new keyword arguments defaulting to today's behaviour:

- `memvara/retrieve/hybrid.py`: `episode_role_weights`, `max_episode_tokens`,
  `episode_role_intent`; the multiplier in `_episodes`; the budgeted cut; the pool widening
  when a reranker is set; the post-rerank episode cut. Roughly 80 lines.
- `memvara/retrieve/intent.py` (new): the pattern set and a `wants_assistant(query) -> bool`.
  Roughly 40 lines, with a doctest that runs.
- `memvara/core.py`: the three arguments threaded through `Memvara(read_...)` as the
  reranker arguments already are.
- Tests in `tests/test_hybrid.py` and a new `tests/test_intent.py`: each argument proven able
  to fail by breaking what it watches — a weight of 0 removes the role; the budget stops where
  the arithmetic says; the best episode survives a budget it exceeds alone; the pool is wider
  only with a reranker set; `NullReranker` as the control that the widening alone changes
  nothing; the intent rule's precision and recall pinned against a fixture of question texts.
  Coverage stays at 100%.
- Documentation in the same commit: `CHANGELOG.md`, `docs/INTERNALS.md` (the episode
  pipeline section), the tool descriptions in `memvara/server/tools.py` if any read tool
  exposes the new arguments, and `README.md` only if a reader is told to set them.

**harness** (`memorybench`, provider `memvara`): `MEMVARA_ASSISTANT_STUB_WORDS` in
`prompts.ts` beside the existing truncation knobs, with a test; the read arguments passed
through the search request the way `read_max_episodes` already is.

**sweep tooling**, under `bench/` in memvara: `bench/longmemeval_pool.py` to do the wide
pull and the cross-encoder pass and write the cache; `bench/longmemeval_sweep.py` to run
the grid over the cache and emit the front. Both offline after the pull, both tested on a
fixture pool. The cache itself is a measurement artifact and lives under `local/`.

**Step 0, a decision before step 1.** The judged arms need a server that has these arguments.
The hosted deployment is hand-deployed while CI is down, so the recommendation is to run the
harness against a local memvara server for steps 1–3 and reserve the hosted tenant for the
final number once a configuration is chosen. The wide pull must come from whichever store
the judged arms will use, or the sweep predicts a different retriever than the one measured.

## Deliberately deferred

- **Question-side retrieval** — decomposition or retrieve-read-retrieve for multi-session
  aggregation. Highest ceiling for the one category that needs tens of turns of evidence,
  and the most build; costs model calls at query time. Taken up only if the step 1 stop rule
  fires or step 3 leaves multi-session short of 93%.
- **Sentence-level extraction inside user turns.** Worth about 2× on turns already 43 words
  long; role is worth about 5×. After A and B are measured.
- **Stub rendering inside memvara's own renderers.** Named above; a product decision, not a
  benchmark one.
- **MMR diversity.** First axis to add if multi-session coverage stalls in the sweep.

## Risks

- **The echo counter-claim.** The labels say assistant turns carry 2% of the evidence the
  user's turn does not; the harness comment says the answer string is in an assistant turn
  69% of the time. Both are measurements; they measure different things. Step 2's stop rule
  is the arbiter.
- **The cross-encoder on conversational turns.** Its numbers are from LOCOMO. MiniLM was
  trained on passages, not chat. Step 3's stop rule is the arbiter, and `NullReranker` in the
  sweep separates the stage's effect from the model's.
- **`T` and every other constant is fitted on the corpus it is evaluated on.** The 0.55
  caveat from the floor carries over to every axis here. A second corpus (LOCOMO, whose
  evidence labels the reranker was measured on) is the generalisation check, and it is not
  in this plan; it should be the plan after this one.
- **Token accounting mismatch.** Core counts code points over four; the harness counts
  tiktoken. The sweep pins the ratio; if it is unstable across categories the budget has to
  be set per the harness's count rather than core's, which means the harness truncates and
  core only orders.
- **A hosted server without the arguments.** Covered by step 0.

## Success criteria for the whole plan

Step 3 reports, on the 199-question sample, a configuration with median tiktoken context at
or under 720 and judged accuracy at or above 95%, with per-category numbers and the paired
test against the cap-15 control committed beside the run. Anything less is reported as what
it is, with the stop rule that fired and what it says to do next.
