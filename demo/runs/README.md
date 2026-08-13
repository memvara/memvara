# Recorded runs

One run, kept so the numbers quoted elsewhere can be **audited** rather than believed.

It cannot be reproduced — see below — so the artefacts are the only evidence there is.
That is a reason to keep them, not a reason to trust the run more than it deserves.

## `2026-08-13-agent-reader`

| file | what |
|---|---|
| `*.answers.jsonl` | the 100 answers, `{"id", "answer"}` — one per question × arm |
| `*.scored.jsonl` | the same rows joined to `arm`, `kind`, `question`, `gold`, `trap`, `correct`, `trapped`, `context_chars` |

**The reader was an agent, not a model behind an API.** No `ANTHROPIC_API_KEY` or
`OPENAI_API_KEY` existed in the environment, so `--reader anthropic` was unavailable and
the blinded `FileReader` round trip was used instead: the harness wrote 100 prompts
carrying only `{id, system_prompt, prompt}`, the agent answered every one from its
context alone, and the answers came back through the scorer.

**It is not reproducible.** There is no model id, no seed and no temperature to quote
beside it, and the same contexts answered again will not give the same answers. Treat it
as one audited sample, never as a benchmark, and never beside a published LOCOMO or
LongMemEval score.

**The answerer wrote the library.** `evalkit.FileReader`'s docstring is explicit that
this is the weakest part of the arrangement and that some of it cannot be fixed: the
system name, dataset, question id, category, gold, retrieval statistics and order were
all withheld, but `recall()`'s headers are recognisable on sight to anyone who has read
this repository, and `full_transcript` is identifiable by being four times longer than
anything else. Read that docstring before quoting any of this.

What the corpus does remove is the confound LOCOMO and LongMemEval cannot: it was written
for this measurement and exists nowhere else, so no reader can have seen it in training.

## Reading the `correct` and `trapped` columns

**Both are scored by `ContainmentJudge`, a substring rule, and both are wrong in known
directions.** The run was audited by hand afterwards. The judge marked 16 rows incorrect
(excluding the context-free `none` arm) and 16 trapped, 9 of which are the same rows —
**23 distinct rows the audit disputes**, counted from the file rather than estimated:

* It marks a **correct paraphrase wrong**. Ten of the sixteen rows it scored incorrect
  were correct answers in different words — mostly on `correction` questions, where the
  gold is a sentence rather than a value. Six were genuine misses: four in `naive_rag`,
  one in `memvara`, one in `memvara_structured`.
* It marks an answer **trapped for containing the trap string in any role**. All sixteen
  `trapped` rows were artefacts; **no arm gave a genuine trap answer.** Two examples:
  `q_mobile_current` answered
  `07700 900 811` counts as giving the trap `07700 900 118`, because two phone numbers
  differing in two digits pass the token-F1 ≥ 0.6 fallback; and `q_pro_price` answered
  *"£79 is for an extra node, not the answer"* counts as answering £79.
* On `correction` questions the gold **contains** the trap by construction — a correct
  answer has to name the wrong value in order to say it was wrong. `demo/README.md` says
  not to count traps by containment there. It applies more widely than that.

Re-run with `--judge llm` once a key exists. Nothing about the arms changes, only the
instrument.

## Regenerating the prompts these answer

The dump is deterministic given the code and the seed, so it is not stored:

```bash
PYTHONPATH=.:bench python3 demo/harness.py --dump runs/dump.jsonl --seed 20260813
```

It will only match these ids while `demo/scenario.py` and `demo/baselines.py` are
unchanged — the id is a digest of the prompt, so editing a fact or a renderer renumbers
everything. If they no longer match, the run is stale and the honest thing is to discard
it rather than re-key it.
