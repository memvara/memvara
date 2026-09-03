# Plan: build and run the two judged arms at 720 tokens

**Date:** 2026-09-03. Executes the "judged arms" section of
`docs/superpowers/specs/2026-09-03-role-aware-budget-and-episode-rerank-design.md`.
Every step names its check. A step whose check fails stops the plan; nothing downstream is
run on a failed check and reported as if it had passed.

## Task 1 — harness: selection and budget knobs, as a reviewed PR on the fork

Repository `/Applications/workstation/memorybench`, branch `memvara-budget-arms` off
`memvara-provider`. The uncommitted truncation knobs already in `prompts.ts` are committed
first, on their own, so this change does not absorb them.

Changes, all off by default so the shipped provider renders byte-identically:

1. `MEMVARA_SEARCH_K` — the `k` sent on every search; default stays 30. The arms send 200.
2. `MEMVARA_ROLE_SELECT` — `off` (default), `user`, or `route`. `user` keeps turns whose role
   is `user`. `route` keeps assistant turns only when the frozen rule fires on the question
   text, else user turns only. The rule is the fourteen alternatives the verification found
   to carry every measured number, verbatim, case-insensitive:
   `you suggested|you recommended|you mentioned|you told me|you provided|you wrote|you created|did you say|can you remind me|remind me what|remind me which|remind me who|remind me how|remind me of`,
   each on a word boundary.
3. `MEMVARA_TOKEN_BUDGET` — when set, the turns block is filled greedily in the order memvara
   returned: render each line, stop at the first line that would push the whole block —
   header included — past the budget, always keep the first item. Counted with the harness's
   own `js-tiktoken` `o200k_base`, the encoder it reports `contextTokens` with, so the
   budget is the number the run records.
4. The order memvara returned is never changed by the renderer. The routing needs the
   question, so `renderMemvaraContext` takes it from `buildMemvaraAnswerPrompt`.
5. The comment on the truncation knobs is corrected to say both measurements: answer-string
   presence favours assistant turns; the gold labels put 94% of evidence in user turns.

Tests (`bun test`): defaults render exactly today's text on a fixture; each knob parsed;
the rule fires on the fourteen phrasings and not on the present-tense requests
("Can you suggest a hotel…"); the fill stops where the encoder says and keeps the first item
even when it alone exceeds the budget; `user` and `route` keep the roles they claim.

**Check:** `bun test` green; a diff of the rendered context with all knobs unset against
`main` is empty. Then PR to `memvara-provider`, reviewed before merge, no attribution.

## Task 2 — server: the cross-encoder, as a documented local edit

`memvara-cloud`, uncommitted `BENCHMARK-LOCAL` lines: a lazily constructed
`CrossEncoderReranker` shared by the base instance and every per-tenant clone, passed as
`read_reranker` with `read_rerank_top_n=200` in both `asgi.py:1371` and
`memories.py::_clone` (the clone drops `read_*` options, filed separately). Rebuild from the
clean core at `6c0ff6c` with the compose env in the stack notes.

**Check:** `k=200` returns 200 items whose order differs from the pool order and whose
`ranking` carries a rerank score; a one-question `test` run through the harness with the arm
env records `contextTokens` near 690 and answers.

## Task 3 — the arms

For each arm, copy `memvara-cap15-control/checkpoint.json` to a new run id, keep
`dataSourceRunId`, reset `search`, `answer` and `evaluate` to pending in the shapes the stack
notes give, and run with `-m gpt-5.4 -j gpt-5.4`, `SKIP_RETRIEVAL_EVAL=1`,
`MEMVARA_TURNS_ONLY=1`, `MEMVARA_SEARCH_K=200`, `MEMVARA_TOKEN_BUDGET=720`, and:

| run id | `MEMVARA_ROLE_SELECT` |
| --- | --- |
| `memvara-routed720` | `route` |
| `memvara-useronly720` | `user` |

Sequential, routed first. Both record the core sha, the image build time, the embedder
(`all-MiniLM-L6-v2`), and the reranker model.

**Check:** every question reaches `evaluate: completed`; a run with a failed question is
re-run, never reported.

## Task 4 — analysis

Per arm against the control, paired on the 199: accuracy overall and per type, McNemar
exact, recorded context median and mean, wins/losses/ties; each prediction in the spec
beside its outcome; each stop rule stated as fired or not. Verified by a second pass before
the finding is written and committed beside the others.

**Check:** the numbers reproduce from the checkpoints with a script under `local/`.
