# Confirming the candidate floor on the real store

PR #159 floors the candidate window at 50 (#155). It stays a draft until a before/after on
the store the issue measured shows the floor helping and nothing regressing. This file is
the run sheet: what to run, in which order, and what decides.

Everything below is `bench/floor_e2e.py`, which wraps `bench/hosted.py` under one fixed
set of arguments so every run is comparable: the hook's `k=4`, the hook's own relevance
floor, one result file per run under `local/floor-e2e/` in this checkout.

## Before you start

- Check out `claude/memvara-retrieval-relevance-rm7t9p` in this checkout. Only `bench/`
  and this file matter on the client side; the retriever change runs on the box.
- Your probe file at `~/.memvara/probes.jsonl`, the 40-probe suite the issue used. Pass
  `--probes PATH` if it lives elsewhere.
- A credential for the store: `MEMVARA_API_KEY` and `MEMVARA_SERVER_URL` in the
  environment, or the file `memvara-mcp login` writes. Read-only is enough for every step.
- `MEMVARA_RECALL_MIN_SCORE` set to whatever your hook runs with, if you changed it. The
  bench resolves the floor exactly as the hook does, so leaving it unset measures 0.29.
- For `replay`: `pip install "memvara[local-embed]"` in the environment you run from, so
  the copy is embedded with `all-MiniLM-L6-v2`, the model the deployment uses. The step
  refuses to run under the hashing fallback.

## The run, in order

```bash
cd /Applications/workstation/agent-memory
git fetch origin && git checkout claude/memvara-retrieval-relevance-rm7t9p

# 1. Production as it is today. This should reproduce the issue: hit 85.0%, gold-rank 1.5,
#    and the version-number probe absent from its own results.
PYTHONPATH=. python3 bench/floor_e2e.py before

# 2. A copy of the store, here, at floor 0 and floor 50. Floor 0 should agree with step 1;
#    how closely says how far to trust floor 50 as a prediction of step 4.
PYTHONPATH=. python3 bench/floor_e2e.py replay
PYTHONPATH=. python3 bench/floor_e2e.py compare before replay-floor-0

# 3. Deploy the branch to the box. The provision script ships this checkout at HEAD, so the
#    checkout above is what goes out. Note the short sha it prints for agent-memory.
(cd ../memvara-cloud && deploy-scripts/memvara-provision.sh)

# 4. Production on the branch, same command as step 1.
PYTHONPATH=. python3 bench/floor_e2e.py after

# 5. The decision.
PYTHONPATH=. python3 bench/floor_e2e.py compare before after
```

Steps 1 and 2 need nothing deployed and can run today. If step 2 looks wrong, stop there
and say so before step 3 touches production.

## What decides

`compare before after` is the number. Read it per probe, not only the headline, and hold
it to three conditions:

- **The lost probe hits.** The issue's version-number probe is absent at `k=4` on `main`
  and rank 1 from `k=5`. It must appear in `after`.
- **No hit probe regresses.** A wider window can only add candidates, and only a claim
  that outscores what was there can displace one. A probe that hit before and misses
  after is a real finding, not noise, and the PR does not merge over it.
- **Abstain probes still abstain.** The false-injection rate on `abstain` must not rise.
  A deeper window offers `min_score` more candidates to accept; the floor is wrong if it
  starts letting them through.

Expected from the issue's own measurement: `hit` 85.0% → 90.0%, mean gold-rank 1.5 →
1.4. The `after` run does not have to match those figures, which were `k=6` on a store
that has moved since; it has to satisfy the three conditions above.

## Reading the compare output

- `compare before replay-floor-0` **will print a drift warning**: the two runs read
  different surfaces (hosted against a local file) and possibly a different claim count.
  That is expected here. What you are reading past it is whether the per-class rates
  agree. A gap of a probe or two is the SQLite lexical leg not being Postgres's; a large
  gap means the copy is not a fair stand-in and step 2 predicts nothing.
- `compare before after` should print **no** warning. If it does, the store moved between
  the runs — claims were written in between — and the deltas are not a before/after. The
  box is already on the branch, so `before` cannot be taken again; read the two files
  probe by probe instead, and trust only the probes whose gold claim exists in both.

## What to hand back

The four result files under `local/floor-e2e/`, the compare output for step 5, and the
agent-memory sha the provision script printed. That is enough to update the PR body with
a measured number in place of "50 was chosen, not measured", and to merge or not.

## If it does not help

The floor is the smallest change that fixes the shape; the number is the open question.
If `after` recovers the lost probe but regresses another, the next thing to try is a
lower floor (25 recovers the issue's probe by its own sweep) before any other direction
from the issue. `read_candidate_floor` is a constructor argument, so the box can run a
different value without a code change once memvara-cloud exposes it; today it does not,
and a different value is a one-line change to the default in `memvara/retrieve/hybrid.py`.

## Result, 2026-09-02

The floor merges. All three conditions held on the hosted store at `k=4`.

| run | hit@4 | mean gold-rank | verbatim@1 | abstain false-injection |
|---|---|---|---|---|
| `before`, box at core `3d3ab84` | 85.0% | 1.5 | 100% | 37.5% (3 of 8) |
| `replay-floor-0`, local copy | 85.0% | 1.5 | 100% | 37.5% |
| `replay-floor-50`, local copy | 90.0% | 1.6 | 100% | 37.5% |
| `after`, box at core `9bf0715` | 90.0% | 1.4 | 100% | 37.5% (the same 3) |

- **The lost probe hits.** `h017`, "why can't we trust the version number?", was absent
  at `k=4` in `before` and is rank 1 in `after`. The copy predicted rank 3.
- **No hit probe regressed.** The other 19 hit probes have the same rank in both runs.
- **Abstain probes still abstain.** The same three probes leak in both runs and the same
  five stay silent. The top score on the three that leak rose a little (0.29 → 0.33,
  0.31 → 0.31, 0.35 → 0.40), which is the deeper window handing `min_score` more
  candidates; none crossed from silent to injected.

Two things the compare output flagged, and how they were read.

1. **The store moved between `before` and `after`: 740 → 745 live claims.** Another
   session wrote five claims in the fifteen minutes between the runs, all `user` or
   `memvara_code` facts unrelated to any probe. Nothing was removed, so every gold claim
   exists in both. Two of the five appear in the `after` results of `h019` and `h020`
   below the gold claim, which stayed at rank 1 in both.
2. **The box did not go from `main` to `main` plus the floor.** It ran core `3d3ab84`,
   57 commits behind `main`, so the release carried those too (anchored search, the
   embed-cache fix, the openai backend, among others). memvara-cloud stayed at `fe17d5f`,
   the sha already deployed. The replay isolates the floor from that: the same code at
   floor 0 and floor 50 on the same copy gives the same one-probe delta, and floor 0 on
   the copy agreed with `before` on 38 of 40 probes, the other two off by one rank.

The four result files and the release log are under `local/floor-e2e/` in the checkout
that ran them, with a copy in the main checkout's `local/floor-e2e/`. The release was
`memvara-provision.sh release` with `CORE_REPO` pointed at the branch's worktree, and it
shipped agent-memory at `9bf0715`, which is the branch before this section was added.
