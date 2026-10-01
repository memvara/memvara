# Local selector: ranked reads with no model provider

**Date:** 2026-10-01. **Status:** proposed; nothing in this document is built. It extends
[model-ranked recall](2026-09-04-model-ranked-recall-design.md), whose §3 ("Where it
sits", "The protocol", "The outcomes") is assumed knowledge here. The build is planned in
[`docs/superpowers/plans/2026-10-01-local-selector.md`](../plans/2026-10-01-local-selector.md).
Line numbers are from agent-memory `origin/main` at `50ccd6bd` (release 0.18.0).

> **Correction, 2026-10-01, after gate G0 ran.** §1 and §3 overstate what a local
> selector achieves. The 94.7% to 95.0% figures come from orderings that put *every*
> routed candidate first. A real ranked read renders only the kept turns first, and then
> the rest of both roles in reranked order. Replayed that way:
> - the stock model reaches 0.833, and G0 failed;
> - the 2026-09-27 fine-tuned model reaches 0.889;
> - the model selector reaches 0.958.
>
> Step 1 did not ship. `docs/BENCHMARKS.md`, "The local selector", has the tables. The
> choices this leaves are in §10, decision 6.

Terms used throughout:

- A **ranked read** is `search()` or `recall()` with `ranked=True`.
- The **selector** is the stage that is handed up to 40 candidate turns and names the ones
  that bear on the question.
- The **model selector** is the `ModelSelector` that ships today. It makes one chat call
  on a key the deployment holds.
- The **local selector** is the new `LocalSelector`. It runs a small cross-encoder inside
  the Memvara process.
- **Gold turns** are the turns a dataset marks as holding the answer.
- **Coverage** is the share of gold turns that fit whole inside a 720-token block when the
  read's turns are rendered in the order the server returns them. It is the measure the
  ranked read was tuned on.

## 1. The answer

Memvara will ship a second selector, `LocalSelector`. It works in three steps:

1. It scores each candidate turn against the question with a cross-encoder.
2. It turns each score into a probability, using a calibration measured on labelled data.
3. It keeps the candidates whose probability reaches a threshold, up to a fixed number.

A ranked read that uses it calls no model provider and needs no API key. It costs nothing
per read and sends no text out of the process. It also gives the same answer every time
for the same input.

It ships in two steps.

**Step 1 uses the stock model and needs no training.** The stock model is
`cross-encoder/ms-marco-MiniLM-L-6-v2`, which Memvara already uses as its reranker. On
LongMemEval, reranking with it and then routing by role gives 90.3% coverage. The plain
read, which a customer without a key gets today, gives 67.4%. Judged answers agree:
routed-720 scored 171 of 199 against 135 of 199 for the plain-read twin. Routed-720 is the
same reranker-and-routing order with no selector.

**Step 2 uses the same model fine-tuned on conversation data.** In that data, every label
comes from how the data was built. On LongMemEval a fine-tuned version reached 95.0%
coverage. The paid gpt-5.4-mini selector reaches 95.8%, and the difference is within noise.
That result comes from one dataset, so Step 2 ships only after it holds on data the model
was not trained on (§6).

The model selector stays. A customer who has set a key can keep using it.

## 2. What changes for users

- **Self-hosted `memvara-mcp`.** Today `ranked=true` always ends `unconfigured`, because the
  server has no read-side setting. The parent design deferred one (§9, "A selector setting
  for the self-hosted `memvara-mcp` server"). With `MEMVARA_SELECTOR=local`, the server
  builds a local selector and the cross-encoder reranker, and ranked reads work with no
  key. The reranker runs on ranked reads only.
- **Library users.** Pass `Memvara(read_selector=LocalSelector(), read_reranker=...)`.
  This needs the existing `rerank` extra.
- **memvara-cloud** gets its own plan (§11). Organisations without a key could get ranked
  reads; the choices that involves are in §10.
- **What does not change:**
  - plain reads;
  - the reranker's model and its default;
  - the routing rule;
  - `ModelSelector`;
  - the `Selector` protocol;
  - invariant 1 in `docs/INTERNALS.md`.

## 3. What is established

All coverage figures below come from saved search results for 199 LongMemEval-S questions
(MemoryBench run `memvara-ranked-parity2`). Each question has about 200 retrieved turns.
191 of the questions have at least one gold turn in the pool, and there are 359 gold
turns in all. The scripts and pools are under `local/laya-eval-2026-09-27/` in the
worktree that produced them; the plan rebuilds them inside `bench/` (Tasks 1 to 4).

| Ordering | Calls a provider? | Coverage at 720 tokens |
|---|---|---|
| Plain read (fused score) | no | 67.4% |
| Stock MiniLM over both roles, no routing | no | 76.3% |
| Stock MiniLM, then role routing (Step 1's order) | no | 90.3% |
| Fine-tuned MiniLM over both roles, no routing | no | 91.1% |
| Fine-tuned MiniLM as a second pass over the 40 routed turns | no | 94.7% |
| Fine-tuned MiniLM in one pass, then role routing | no | 95.0% |
| Model selector (gpt-5.4-mini, as shipped) | yes | 95.8% |

Paired bootstrap over the 191 questions, 5,000 resamples:

- **Fine-tuned in one pass, minus the model selector:** −0.8 points (95% interval −3.1 to
  +1.4). The fine-tuned model does better on 6 questions and worse on 8.
- **Fine-tuned in one pass, minus stock with routing:** +4.7 points (+2.3 to +7.4). It does
  better on 13 questions and worse on none.

Judged answers, with gpt-5.4 as reader and judge in a 720-token block:

| Arm | Correct of 199 |
|---|---|
| Plain-read twin | 135 |
| Reranker and routing, no selector (routed-720) | 171 |
| Model selector, gpt-5.4-mini | 177 |
| Model selector, gpt-5.4 | 182 |

Speed and cost:

- **Local selector:** the fine-tuned MiniLM scored a turn in 13.6 ms on a laptop CPU, so 40
  candidates take about half a second.
- **Model selector:** its call is estimated at 4 to 5 seconds and has never been timed in
  production. The 199 gpt-5.4-mini calls of the parity screen cost $0.72 in all.

**Limits of this evidence:**

- One generator wrote every conversation.
- Coverage is not judged accuracy.
- The fine-tuned model learned from the model selector's own keep or omit decisions, which
  §5 replaces.
- The routing rule was fitted on LongMemEval's phrasing.

The plan's gates (§6) exist to close each of these.

**Laya was considered and rejected on 2026-09-27, and rechecked on 2026-10-01.** Laya is an
open "decision model" released as an alternative to the hosted Jev.

- **Speed:** on a 4-core server CPU it takes 580 ms per question with the English
  checkpoint and 193 ms with the multilingual one. Its INT8 CPU export is about 2.5 times
  faster on Laya's own test.
- **Ranking:** untrained, it ranks worse than MiniLM. On BEIR NFCorpus its nDCG@10 moved
  −0.024 where MiniLM's moved +0.016.
- **Calibration:** on 240 "does NEW replace OLD's value?" pairs it reached AUROC 0.939, but
  its calibration error was 0.24 to 0.33.
- **Pinning:** its loader fetches the mutable `main` revision of the weights (Laya issue
  #332, open).

A 22M-parameter model Memvara already ships does the job better at a tenth of the cost.

## 4. The design

### Where it sits

It sits in the same place as the model selector (parent §3, "Where it sits"):

1. The read gathers up to 200 turns.
2. The cross-encoder reranker orders them.
3. `routed_role` cuts them to the questioner's side.
4. The first `top_n` (40) are handed to `select()` as `Candidate`s
   (`memvara/retrieve/hybrid.py:1751-1773`).

`select()` returns the kept ones. Kept turns come first, whole. The rest follow in reranked
order, and `recall()` cuts them to 280 characters (`memvara/core.py:3936-3987`).

### How it decides what to keep

- **Score.** One raw cross-encoder score (a logit) per candidate, for the pair (question,
  turn text). The turn text is used as stored, with no date in front of it. When date text
  entered the index, the cross-encoder's gain fell from +7.1 to +4.3 (the declined
  episode-time design, 2026-09-06).
- **Probability.** `p = sigmoid(scale × score + shift)`. `scale` and `shift` are fitted by
  logistic regression on labelled candidates the model was not trained on (Platt scaling).
- **Keep rule.** Keep every candidate with `p ≥ threshold`, at most `max_keep` of them, the
  highest-scoring first. Return them in the order they were handed in, as the protocol
  requires (`memvara/select/base.py:159-191`).
- **Choosing the threshold.** `threshold` and `max_keep` are chosen on a validation split.
  The rule is to maximise coverage, and to break ties toward fewer kept turns.
- **Keeping nothing is a valid answer.** The outcome is `applied` with `kept=0`, and the
  read renders the reranked turns of both roles with no line saying the read was unranked
  (`hybrid.py:1811-1836`). The model selector answering "none" produces the same result.

### The model and its calibration

- **The stock model** has no calibration file, because its repository is not Memvara's. Its
  calibration lives in code as `STOCK_CALIBRATION`, measured in the plan's Task 4. Its
  revision is pinned to commit `233902d25c440f23af6f7d6e94d2946bac0bee0a`, which is
  Hugging Face `main` as of 2026-10-01. A Hub commit fixes the weights, because the commit
  records each large file's SHA-256 in its pointer.
- **A trained model** is a directory, or a Hub repository pinned to a commit. Beside its
  weights it holds `memvara_selector.json`:

  ```json
  {"format": 1, "scale": 1.21, "shift": -2.4, "threshold": 0.35, "max_keep": 8,
   "max_length": 256, "weights_sha256": "…", "base_model": "…", "base_revision": "…",
   "sources": {"longmemeval": 250, "synth-coding": 120}, "seed": 0}
  ```

  The numbers in this example are made up. When the file names `weights_sha256`, the
  weights are checked before they are loaded, and a mismatch is refused.
- **Loading.** The model loads when the selector is built, as `CrossEncoderReranker`'s does
  (`memvara/rerank/cross.py:40-62`). A missing extra or a missing model therefore fails at
  startup and not on the first ranked read. Naming the class imports nothing heavy.

### Counting

The model selector's series are `retrieval.model_query`, `retrieval.select_ms` and
`retrieval.model_fallback`. They mean calls a model provider answered, and they are what a
quota sums by name (`memvara/telemetry.py:420-470`). Local selections must not land in
them, so a local selector gets three series of its own:

- `retrieval.local_query`: one per ranked read the local selector answered.
- `retrieval.local_select_ms`: the time spent scoring.
- `retrieval.local_fallback`: tagged `reason`, either `malformed` or `error`.

The ranked stage picks the set by reading `kind = "local"` on the selector. Any other
selector keeps today's series.

### Outcomes and failures

- **Scoring succeeds:** the outcome is `applied`.
- **The encoder misbehaves:** it might return the wrong number of scores, or a score that
  is not finite. Either way `select()` raises `ValueError`, and the stage serves the plain
  read with `fallback/malformed`.
- **The encoder raises anything else:** the stage serves the plain read with
  `fallback/error`. This is the existing behaviour for a failed ranked read (#308, PR #389).
- **`admit()`** never refuses, like `ModelSelector`'s. The hosted service already wraps
  selectors in its own admission gate.
- **Concurrency.** Scoring is serialised per selector by a lock, the same way the hosted
  service locks its reranker (`LockedReranker`), so concurrent ranked reads take turns on
  the CPU.
- **`usage`** is left untouched, because there are no tokens to bill.

### What does not change

Invariant 1 already allows a model on the read path only through the `ranked`,
`query_rewrite` and `synthesis` stages, each with a recorded outcome and a fallback that
needs no model. `docs/INTERNALS.md` already treats a reranker on the read path as a
cross-encoder, not a generative model. The local selector is a cross-encoder in the
`ranked` stage, so the invariant's text needs one sentence naming it, and no new
exception.

## 5. Training data

### Why it does not need "every type of data"

The model's job is narrow: does this turn help answer this question? The stock model has
already learned general relevance from about 500,000 real web-search queries (MS MARCO).
Fine-tuning only adapts it to conversation turns: short first-person user turns, long
assistant turns, questions about one's own past, and values that changed over time.

So it needs every shape of turn Memvara stores, not every topic in the world. What it does
not see in training is measured on sources held out of training, and the safety nets
below cover the rest.

### Sources

| Source | What it adds | Licence | Use |
|---|---|---|---|
| LongMemEval-S | Chats between a user and an assistant about personal life, with gold turns marked | MIT (dataset and code) | Training and validation on 301 questions. The 199 parity questions are test only. |
| Synthetic coding-agent sessions | Repository setup, decisions, bugs, configuration, tool choices and preferences: what Memvara's users actually store | Ours, generated by an open-weights model under Apache-2.0 or MIT | Training, validation and test, split by scenario |
| Synthetic personal and work conversations | Schedules, people, health, travel and money, including values that change and facts that are retracted | Ours, as above | Same |
| Synthetic two-person conversations | Two named people, both stored as `user` with the name in front, as `MEMVARA_NAMED_SPEAKERS` does in the harness | Ours, as above | Same |
| LoCoMo | Ten long two-person conversations with turn-level evidence | CC BY-NC 4.0 | **Test only.** Never trained on. |
| Your own transcripts | Real coding-agent traffic | Yours | A screen on your machine, with your consent. Never trained on. |

No customer's data, and no customer's selector decisions, are used for anything.

### Labels

- **LongMemEval** marks gold turns with `has_answer`. The pool builder reads this flag only
  after retrieval, by matching turn text. It never passes the flag into ingestion or the
  query, which `tests/test_bench_eval.py:1071` guards for the loader.
- **Synthetic data:** each scenario plants each fact in a known turn. Validation then checks
  three things:
  - the fact's value appears in the turn that plants it;
  - the value appears in no other turn;
  - every question names facts that exist.

  A scenario that fails any check is discarded, never repaired.
- **The model selector's decisions are not used as labels.** The 2026-09-27 experiment
  trained on them. Plan Task 15 measures what gold labels lose against them on LongMemEval,
  so the trade is known. A provider's terms may limit training on its output, and gold
  labels make that question moot.

### Hard negatives

Training pairs are built from Memvara's own retrieval of each question: the pool is
reranked by the stock model and routed, and the first 40 are taken. That is exactly the
list the selector will be handed in production. The negatives are therefore the turns
Memvara actually confuses with the answer. Each training question contributes its gold
candidates and up to three non-gold candidates per gold one. The negatives are subsampled,
so the training loss is not calibrated; Platt scaling on the validation split corrects
that, which is why calibration is fitted after training and never during it.

### Synthetic generation

Each scenario is generated as one JSON document with:

- one domain and one conversation shape;
- two to four facts of the following kinds: a preference, a decision, a number or date, a
  value that changes (old and new), a fact that is retracted ("I no longer…"), something
  the assistant recommended, or (in two-person scenarios) who said something;
- four to eight dated sessions;
- at least one near-miss turn per fact, which names the fact's subject without its value;
- three to five questions: single fact, two facts, current value, time-based, what the
  assistant said, and one question the conversation cannot answer.

The output uses LongMemEval's JSON schema, so the same pool builder reads every source.
Each question's haystack is its own scenario's sessions plus sessions from other scenarios,
sorted by date.

Planned volume: 260 scenarios and about 1,000 questions. The scenarios split as 80
coding-agent, 50 work, 70 personal and 60 two-person. The split within each source is
70/15/15, by scenario.

**Generator model.** An instruct model released under Apache-2.0 or MIT, served by an
OpenAI-compatible server on the GPU machine the extraction worker already uses. Using a
commercial API's output to train a model needs counsel first, so it is not the default.

## 6. Validation gates

These numbers are fixed now, before any result is seen. Every gate's result is recorded in
`docs/BENCHMARKS.md`, whether it passes or fails.

| Gate | What | Pass |
|---|---|---|
| G0 | Stock model as the local selector, protocol-faithful, LongMemEval test (199) | coverage ≥ 0.89, and gold-kept recall ≥ 0.80 |
| G1 | Fine-tuned model, LongMemEval test (199, never trained on) | coverage ≥ 0.93 |
| G2 | Leave one source out: train on all sources but one, test on the one left out | on every held-out source, fine-tuned ≥ stock − 0.5 points; mean gain over stock ≥ +2.0 points |
| G3 | LoCoMo, categories 1 to 4, with routing off (two people), test only | fine-tuned − stock ≥ +1.0 point, with the 95% interval above 0 |
| G4 | Judged LongMemEval (199), gpt-5.4 reader and judge, 720 tokens | ≥ 174 correct, and a paired net of at least −3 against the model selector's 177 |
| G5 | Your real transcripts: 100 recall prompts, local against model selector on the same candidates | the local selector keeps ≥ 75% of the turns the model selector keeps, and a hand review of 20 disagreements is written down |
| G6 | Speed and memory on 4 CPU threads | `select()` p95 ≤ 1.0 s for 40 candidates, including a 4,000-character prompt; resident memory grows ≤ 300 MB with the reranker already loaded |

Predictions, written down before the runs:

- G0: about 0.90.
- G1: about 0.945.
- G4: 175 (range 170 to 180).

**G4 is a screen, not a proof.** Showing that a model is no worse within 3 points needs
about 489 paired questions (parent §9). A 199-question run can only catch a large loss.

**How the gates are used:**

- Step 1 (stock model, opt-in) ships after G0 and G6.
- Step 2 (the fine-tuned model becomes the default) ships after G1 to G6.
- If G2 or G3 fails, Step 2 does not ship and Step 1 stays.

## 7. Rollout

1. **Core:** `LocalSelector` with the stock model, its telemetry, `MEMVARA_SELECTOR` in
   `memvara-mcp`, and every document that describes ranked reads. Opt-in. This is one
   pull request.
2. **Bench:** pool building, scoring, calibration, metrics, the synthetic generator,
   training and the gates. The tooling lands with step 1; the runs follow it.
3. **The fine-tuned model:** published to a Hugging Face repository the project controls,
   pinned by commit, and made the default for `LocalSelector()`. This changes behaviour,
   so `docs/UPGRADING.md` says so.
4. **Follow-up plans** for memvara-cloud, memvara-web and the plugin repositories (§11).

## 8. Alternatives considered

- **Laya, or the hosted Jev.** Rejected: see §3.
- **A larger cross-encoder**, such as `bge-reranker-v2-m3` at 568M parameters. It took about
  490 ms per turn on a laptop CPU, against 30 ms for MiniLM. The roadmap's reranker table
  already found that a bigger reranker was not a better one on this workload.
- **Putting the fine-tuned weights into the reranker and running one pass (95.0%).** This
  does less work, but it changes plain reads for every library user who reranks. It is
  revisited only if G6 shows the second pass matters.
- **No model, just reranker order and routing (90.3%).** This is Step 1 with every
  candidate kept. Step 1 adds a calibrated keep rule because a probability is also what
  abstention needs later (§9).
- **Training on the model selector's decisions.** This measured 94.7% to 95.0%, but its
  labels are a provider's output (§5).

## 9. Deliberately deferred

- **Abstention.** A calibrated relevance probability could replace the recall hook's
  per-route score floors (0.29 and 0.35, set by hand in #399) and the unset `min_score`
  (#129). It needs its own measurement: the 30 LongMemEval abstention questions and the
  332 negatives in `bench/anchoring.py`.
- **Languages other than English.** The base model is English-only. A multilingual one
  would need its own calibration. Separately, the default embedder handles Latin script
  only, so retrieval misses non-Latin turns before any selector sees them.
- **Using the local selector as the fallback when the model selector fails.** This would
  change the decision in #308 that a failed ranked read serves exactly the plain read. It
  is memvara-cloud's choice (§10).
- **Sharing scores between the reranker and a stock-model selector.** With the stock
  model, the selector re-scores 40 turns the reranker has just scored, about 0.2 to 0.5 s
  of repeated work. Sharing those scores needs a `score` field on `Candidate`, which is a
  protocol change, so it waits for G6 to show the time matters.
- **Spans.** The local selector returns `span=None`, which the protocol allows.
- **Write-side advice and a gate on the capture hook.** These are the evaluation's other
  candidates, and each needs its own labelled data.

## 10. Decisions the user owns

None blocks Step 1.

1. **Where the fine-tuned weights are published**, for example a `memvara` organisation on
   Hugging Face. It must be one the project controls, because the code pins its commit.
   Blocks Step 2.
2. **The generator model for the synthetic data, and where it runs.** Blocks Task 13.
3. **Budget for the paid runs.** Measured costs so far:
   - one judged LongMemEval arm (answer and evaluate, 199 questions): $1.08;
   - one judged LoCoMo run of 250 questions: $1.78;
   - 199 gpt-5.4-mini selector calls: $0.72.

   G4 and G5 together need about $2 at those rates. Blocks Tasks 17 and 18.
4. **memvara-cloud:** should organisations without a key get the local selector by
   default? Should organisations with a key fall back to it when their model call fails?
   The second question changes #308. Blocks the cloud plan.
5. **memvara-mcp:** should the local selector become the default once Step 2 ships, or stay
   opt-in? Blocks Task 16's docs.
6. **After G0 failed (2026-10-01): how should a local selector's read be rendered?** The
   keep rule leaves too many gold turns to a tail where long assistant turns crowd them
   out. The options:
   - **(a)** Have the ranked stage render the unkept *routed* candidates before the other
     role's turns. This changes the paid selector's rendering too, so it needs its own
     measurement.
   - **(b)** Train the selector to keep more of the gold, measuring with the faithful
     replay from the start.
   - **(c)** Ship the keep-everything-routed order (0.903, judged 171 of 199) under a
     different name, without the keep rule.
   - **(d)** Stop here.

   Blocks everything after Task 5.

## 11. Follow-up plans in other repositories

- **memvara-cloud.**
  - Bake the selector model into the image at the pinned revision; the runtime is offline
    (`deploy/Dockerfile:123-139, 278-284`).
  - Decide decision 4 and wire `LocalSelector` into `_OrgSelectors`
    (`deploy/memvara_deploy/asgi.py:683-742`), sharing the `LockedReranker`'s encoder.
  - Meter `retrieval.local_query` separately from `ranked_reads`
    (`memvara_cloud/metering/api.py:195-207`).
  - Time a ranked read on the 4-core production host.
- **memvara-web.** Rewrite `/docs/cloud/model-ranked-recall` customer-first. It says a
  ranked read needs a key, which stops being true for organisations that get the local
  selector.
- **Plugin repositories.** Run the packaged skill's pin bump through
  `scripts/sync_plugin_repos.py` after its text changes.
- **memorybench.** The LoCoMo loader reads `answer`, but 444 of the 446 category-5
  questions carry only `adversarial_answer`, so their ground truth becomes the string
  "undefined". G3 skips category 5, but a judged LoCoMo run needs the loader fixed first.

## 12. Step 2a: train the selector to keep more turns (pre-registered 2026-10-01)

Written before any model in this section was trained. The user chose decision 6's option
(b) after G0 failed.

**Question.** G0 failed because the stock model keeps three or four turns. The gold turns it
leaves out then lose the 720 tokens to long assistant turns, which the ranked read renders
after the kept ones. The question here is whether a MiniLM fine-tuned on gold labels keeps
enough of the gold turns to cover well under the server's real rendering.

**Data.** LongMemEval-S only, with the frozen splits from G0:
- 256 questions to train on;
- 45 to calibrate and choose on;
- the 199 parity questions as the test set.

The synthetic sources are not used, because the generator model (decision 2) is not chosen
yet. A result here is in-distribution only. It cannot ship Step 2 by itself: G2 and G3
still apply.

**Arms.** Both arms start from the stock model at its pinned commit. Both train for one
epoch, with learning rate 2e-5, 256 tokens, batches of 16 and seed 0. Both use gold labels
on the 40 routed candidates, with three non-gold turns per gold one.
- **A:** the loss is unweighted.
- **B:** a gold turn's loss is weighted 3, so missing a gold turn costs three times as much
  as keeping a wrong one.

**Calibration.** Platt scaling is fitted on the 45 validation questions. The keep rule is
chosen there too, from a grid wider than G0's:
- thresholds 0.01, 0.02, then 0.05 to 0.95 in steps of 0.05;
- `max_keep` of 4, 6, 8, 10, 12, 16, 24 or 40.

**Choice.** The arm with the higher validation coverage is the one this section reports as
the result. The other arm's test number is reported beside it, but it is not chosen after
the fact.

*Tie-break, added after training and before any test score was computed.* Both arms
printed a validation coverage of 0.931, so the rule above did not decide. The comparison
uses the exact fraction of gold turns covered on validation. If that is also equal, the
arm that keeps fewer turns on average on validation is chosen, which is the same tie-break
the keep-rule grid uses.

**Gate (G1 under the faithful replay).**
- **Pass:** test coverage of at least 0.93.
- **Partial:** between 0.903 (keeping every routed candidate) and 0.93. The model is then a
  free-tier improvement, not a stand-in for the model selector.
- **Fail:** below 0.903.

**Prediction:** 0.91, with a range of 0.89 to 0.93.

**Also reported:** kept recall, mean turns kept, coverage by question type, and the
paired bootstrap against the model selector and against keeping every routed candidate.

**Result (2026-10-01): partial, at the bottom of the band.** The chosen arm, B, covers 0.905
of the test gold turns. That is +0.003 against keeping every routed candidate (95% interval
−0.019 to +0.025) and −0.053 against the model selector (−0.082 to −0.026). Arm A covers
0.908. Both kept about five turns. The result is below the prediction of 0.91 and inside
its range. On validation, raising `max_keep` above 6 lowered coverage, so keeping more turns
is not the lever: every wrong turn kept spends budget a gold turn needed. The lever is a
better ordering, which needs far more training data than 256 questions. The numbers and the
commands are in `docs/BENCHMARKS.md`, "The local selector".

## 13. Step 2b: train on large public datasets first (pre-registered 2026-10-01)

Written before any model in this section was trained and before the public data was
converted. The user asked for public datasets of about 100,000 questions or more, and
approved this step.

**Question.** Step 2a showed that the selector is limited by how well it orders the
candidates, not by how many it keeps, and that 256 LongMemEval questions are too few to
teach the ordering. The question here is whether about 100,000 questions from public
datasets, whose labels mark the sentence or paragraph holding the answer, teach an ordering
that carries over to conversation turns.

**Public data.** Each dataset's licence allows training a model for commercial use. The
licences were checked against primary sources on 2026-10-01; the research report is in
`local/selector/datasets-2026-10-01.md`. Every file is pinned to a revision.

| Dataset | Questions used | A candidate is | Gold is | Licence |
|---|---|---|---|---|
| HotpotQA, distractor setting, train | 30,000 drawn at random | one sentence of the 10 paragraphs | a sentence in `supporting_facts` | CC BY-SA 4.0 |
| 2WikiMultihopQA, train | 30,000 drawn at random | one sentence of the 10 paragraphs | a sentence in `supporting_facts` | Apache-2.0 |
| MuSiQue, answerable, train | all 19,938 | one of the 20 paragraphs | `is_supporting` | CC BY 4.0 |
| QuAC, train | 20,000 questions drawn at random, skipping unanswerable ones | one sentence of the section | the sentence where the answer starts | CC BY-SA 4.0 |

That is 99,938 questions. The query is the question alone, with no earlier dialogue turns
and no paragraph title, because a real read has only the question and the turn. For each
gold candidate, three non-gold candidates are drawn at random from the same question's
context, the same ratio as step 2a. The random draws use seed 0. Natural Questions is left
out for now because its download is about 40 GB and the disk does not have the space.

**Arms.** Both start from the stock model at its pinned commit and train with the step 2a
settings: one epoch, learning rate 2e-5, 256 tokens, batches of 16, seed 0.
- **C:** the public pairs only, with an unweighted loss. It sees no LongMemEval data during
  training, so its keep rule is chosen on the 301 LongMemEval train and validation
  questions together, as the stock model's was.
- **D:** arm C's model, then one more epoch on the LongMemEval train split with step 2a
  arm B's recipe (gold weighted 3). Its keep rule is chosen on the 45 validation questions.

**Choice.** The arm with the higher coverage on the 45 validation questions is the result.
A tie goes to the arm that keeps fewer turns there. The other arm's test number is
reported beside it.

**Gate.** The same as step 2a, on the same 191 test questions:
- **Pass:** test coverage of at least 0.93.
- **Partial:** between 0.903 and 0.93.
- **Fail:** below 0.903.

**Prediction:** 0.92, with a range of 0.90 to 0.94.

**Also reported:** kept recall, mean turns kept, coverage by question type, and the paired
bootstrap against the paid selector, against keeping every routed candidate, and against
step 2a's chosen model (0.905).

**Result (2026-10-01): fail.** Arm D, chosen on validation (0.931 against C's 0.917),
covers 0.894 of the test gold turns, below the gate's 0.903 and below the predicted range.
Arm C covers 0.883. Against step 2a's model, D is −0.011 (−0.035 to +0.010). A diagnostic
added after the result shows why: trained on the public data alone, the model orders
conversation turns worse than the stock model does (0.770 of gold turns in its first five,
against 0.812), and continuing on LongMemEval only brings it back to where LongMemEval
training alone gets (0.852 against 0.858). The public datasets teach finding a sentence that
states a fact, which is not the same task as finding the chat turn where something was said.

The diagnostic also shows that a better ordering barely reaches the reader. Step 2a's model
orders the candidates clearly better than the stock model, yet covers only 0.002 more,
because the ranked read uses the selector's order only for the turns it keeps. That makes
decision 6, how a local selection is rendered, the next thing to settle. The numbers are in
`docs/BENCHMARKS.md`, "The local selector".

## 14. Decision 6: render a local selection differently (pre-registered 2026-10-02)

Written before any number in this section was computed. The user chose this after steps 2a
and 2b. Only option (a) has been measured before, on the stock model and on the 2026-09-27
fine-tuned model, and those numbers are known (0.903 and 0.919). Nothing else here has been.

**Question.** Steps 2a and 2b showed that the server's rendering throws away most of what a
better model knows: the selector's order is used only for the few turns it keeps, and every
other turn follows the stock reranker's order across both roles. Does a rendering that uses
more of the selector's information cover more of the gold?

**The renderings.** Each applies only when the selector is local. The paid selector's
rendering does not change.
- **server:** today's rendering. The kept turns come first, in candidate order, then every
  other turn in reranked order.
- **(a) routed first:** the kept turns first, in candidate order, then the unkept
  candidates in candidate order, then every other turn in reranked order.
- **(e) selector order:** all 40 candidates first, in the local model's score order, then
  every other turn in reranked order. The kept turns are the highest-scoring candidates, so
  they still come first.

For the stock model, (e) is the same order as keeping every routed candidate, because the
stock model's scores are the reranker's scores. It is reported as a check on the replay.

**Keep rule.** The Platt fit of each model is unchanged. The threshold and `max_keep` are
chosen again under each rendering, from the same grid and on the same questions as before:
train and validation together for the stock model and step 2b's arm C, validation alone
for step 2a's B and step 2b's D. Under (e) every candidate is rendered before the rest
whatever the keep rule says, so the grid's tie-break picks the fewest turns kept.

**Primary result.** Step 2a's model B, rendered by (e), on the same 191 test questions. It
was chosen before this section because it orders the candidates best of the models that a
pre-registered rule selected. The gate has the same bands as steps 2a and 2b:
- **Pass:** test coverage of at least 0.93.
- **Partial:** between 0.903 and 0.93.
- **Fail:** below 0.903.

**Prediction:** 0.935, with a range of 0.92 to 0.95.

**Also reported:** every model (stock, step 2a's B, step 2b's C and D) under each of the
three renderings, with kept recall, mean turns kept, the paired bootstrap against keeping
every routed candidate and against the paid selector, and coverage by question type for
the primary result.

**What each outcome leads to.**
- **Pass:** the ranked stage gets a rendering hook that a local selector can use to hand
  back its full order. The paid selector's path is unchanged. The fine-tuned model then
  goes on to gates G2 and G3.
- **Partial:** (e) is worth building only if it also beats (a) on the test set. Otherwise
  (a) is the simpler change.
- **Fail:** option (c) is the one left. It ships the routed order with no keep rule, under
  a different name.
