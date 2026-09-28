# Recorded runs

The runs whose numbers are quoted in `demo/README.md` and `docs/BENCHMARKS.md`, kept so
those numbers can be **audited** rather than believed.

## `2026-09-25w`: a model as the reader, at both corpus sizes

Three runs over hosted run `2026-09-25w`. Its scopes were written on 2026-09-25 by
`--write-only` and read on 2026-09-28, 62.7 to 62.9 hours later.

| file | what |
|---|---|
| `2026-09-25w.hosted.jsonl` | the manifest: every scope, its turn and fact counts, and when its write started and finished |
| `2026-09-25w-qwen3.8-x1.scored.jsonl` | scale 1: 100 rows, one per question and arm, with `answer`, `correct`, `trapped`, `closure` and `stop_reason` |
| `2026-09-25w-qwen3.8-x1.checkpoint.jsonl` | scale 1: every reader and judge call, 207 rows |
| `2026-09-25w-qwen3.8-x1-repeat.*` | the same run again over the same scopes, with its own checkpoint, for the noise floor |
| `2026-09-25w-qwen3.8-x10.scored.jsonl` | scale 10 (`--corpus-scale 10`): 100 rows |
| `2026-09-25w-qwen3.8-x10.checkpoint.jsonl` | scale 10: every call, 199 rows |

Every pinned setting, as the report header printed it, for the reader and the judge alike:

* model `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_S`, served by llama.cpp on a private machine
  (`--base-url` pointed at it; the address is not recorded here)
* `--max-tokens 512`, `--temperature 0`, `--sampling-seed 7`
* `--extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'`
* `--timeout 1800`, `--concurrency 1`, `--judge llm` (the reader's twin, same model)
* `--memory hosted --hosted-run-id 2026-09-25w`, `--min-scope-age` left at 24 hours
* scale 1 and scale 10 with `--corpus-scale`; the questions, golds and traps are the same

The command, less the paths:

```bash
PYTHONPATH=. python3 demo/harness.py --reader openai \
    --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_S --max-tokens 512 --temperature 0 \
    --sampling-seed 7 --base-url http://HOST:8888/v1 --api-key-file KEYFILE \
    --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}' \
    --timeout 1800 --concurrency 1 --judge llm --corpus-scale 1 \
    --memory hosted --hosted-credentials CREDS --hosted-run-id 2026-09-25w \
    --checkpoint demo/runs/2026-09-25w-qwen3.8-x1.checkpoint.jsonl \
    --out demo/runs/2026-09-25w-qwen3.8-x1.scored.jsonl
```

**The plain `memvara` arm's rows are not a result.** Its scopes held 2 to 10 live claims
when they were read, and none was about a fact a question asks. `demo/README.md` says why.

**The runs can be repeated only while the hosted scopes exist, and only as they are
then.** Rerunning a command above with its checkpoint replays every call whose prompt is
unchanged. The two memvara arms read their scopes again, though, so if the service's
background worker has added claims since 2026-09-28, those prompts change and are sent to
the reader again. A run over new scopes will differ for the same reason.

## `2026-08-13-agent-reader`

The first run. It cannot be reproduced, as explained below, so these files are the only
evidence there is. That is a reason to keep them, not a reason to trust the run more than
it deserves.

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
