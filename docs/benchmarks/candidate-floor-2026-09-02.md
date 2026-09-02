# The candidate floor, measured on the real store

PR #159 floors the per-leg candidate window at 50 (#155). It merged on the strength of
the measurement recorded here: a before and an after on the hosted store the issue was
measured on, at the recall hook's `k=4`, made on 2026-09-02. This page is the run sheet
that produced it, kept so that the next retrieval change can be measured the same way,
and the result.

Everything below is `bench/floor_e2e.py`, which wraps `bench/hosted.py` under one fixed
set of arguments so every run is comparable: the hook's `k=4`, the hook's own relevance
floor (`min_score`, which is a different floor from the candidate one this page is
about), and one result file per run under `local/floor-e2e/` in the checkout that runs
it.

## What a run needs

- The branch under test checked out. Only `bench/` matters on the client side; the
  retriever change runs on the box.
- A probe file. `~/.memvara/probes.jsonl` is the 40-probe suite the issue used; pass
  `--probes PATH` for another, before or after the step name.
- A credential for the store: `MEMVARA_API_KEY` and `MEMVARA_SERVER_URL` in the
  environment, or the file `memvara-mcp login` writes. Read-only is enough for every
  step.
- `MEMVARA_RECALL_MIN_SCORE` set to whatever the hook runs with, if it was changed. The
  bench resolves the relevance floor exactly as the hook does, so leaving it unset
  measures 0.29.
- For `replay`, `pip install "memvara[local-embed]"` in the environment the bench runs
  from, so the copy is embedded with `all-MiniLM-L6-v2`, the model the deployment uses.
  The step refuses to run under the hashing fallback.

## The run, in order

```bash
# 1. Production as it is today.
PYTHONPATH=. python3 bench/floor_e2e.py before

# 2. A copy of the store, here, at floor 0 and floor 50. Floor 0 should agree with
#    step 1; how closely says how far to trust floor 50 as a prediction of step 4.
PYTHONPATH=. python3 bench/floor_e2e.py replay
PYTHONPATH=. python3 bench/floor_e2e.py compare before replay-floor-0

# 3. Deploy the branch to the box. memvara-cloud's deploy-scripts/memvara-provision.sh
#    ships the checkout CORE_REPO points at, at its committed HEAD; `release` records a
#    rollback point first. Note the agent-memory sha it prints.

# 4. Production on the branch, same command as step 1.
PYTHONPATH=. python3 bench/floor_e2e.py after

# 5. The decision.
PYTHONPATH=. python3 bench/floor_e2e.py compare before after
```

Steps 1 and 2 need nothing deployed. If step 2 looks wrong, stop there and say so before
step 3 touches production.

## What decides

`compare before after` is the number. Read it per probe, not only the headline, and hold
it to three conditions:

- **The lost probe hits.** The issue's version-number probe is absent at `k=4` on `main`
  and rank 1 from `k=5`. It must appear in `after`.
- **No hit probe regresses.** A wider window can only add candidates, and only a claim
  that outscores what was there can displace one. A probe that hit before and misses
  after is a real finding, not noise, and the change does not merge over it.
- **Abstain probes still abstain.** The false-injection rate on `abstain` must not rise.
  A deeper window offers `min_score` more candidates to accept; the candidate floor is
  wrong if it starts letting them through.

## Reading the compare output

- `compare before replay-floor-0` **prints a drift warning**: the two runs read
  different surfaces (hosted against a local file) and possibly a different claim count.
  That is expected. What you are reading past it is whether the per-class rates agree.
  A gap of a probe or two is the SQLite lexical leg not being Postgres's; a large gap
  means the copy is not a fair stand-in and step 2 predicts nothing.
- `compare before after` should print **no** warning. If it does, the store moved between
  the runs and the deltas are not a before/after. The box is already on the branch, so
  `before` cannot be taken again; read the two files probe by probe instead, and trust
  only the probes whose gold claim exists in both.

## Result, 2026-09-02

The floor merged. All three conditions held on the hosted store at `k=4`.

| run | hit@4 | mean gold-rank | verbatim@1 | abstain false-injection |
|---|---|---|---|---|
| `before`, box at core `3d3ab84` | 85.0% | 1.5 | 100% | 37.5% (3 of 8) |
| `replay-floor-0`, local copy | 85.0% | 1.5 | 100% | 37.5% |
| `replay-floor-50`, local copy | 90.0% | 1.6 | 100% | 37.5% |
| `after`, box at core `4fc97b0` | 90.0% | 1.4 | 100% | 37.5% (the same 3) |

- **The lost probe hits.** `h017`, "why can't we trust the version number?", was absent
  at `k=4` in `before` and is rank 1 in `after`. The copy predicted rank 3.
- **No hit probe regressed.** The other 19 hit probes have the same rank in both runs.
- **Abstain probes still abstain.** The same three probes leak in both runs and the same
  five stay silent. The top score on the three that leak rose a little (0.29 to 0.33,
  0.31 to 0.31, 0.35 to 0.40), which is the deeper window handing `min_score` more
  candidates; none crossed from silent to injected.

Two things the compare output flagged, and how they were read.

1. **The store moved between `before` and `after`: 740 to 745 live claims.** Another
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

The four result files and the release log are kept under `local/floor-e2e/` in the
checkout that ran them. The box's `DEPLOYED` file names the shipped core commit as
`9bf0715`; the branch's history was rewritten after the deploy to strip commit
trailers, and `4fc97b0` is the same tree under its new name.

## What the number is, and is not

50 recovered the issue's probe on a store of 730 claims and moved nothing else. It is a
measured value for a store of that size, not a bound: the displacement it covers is a
score band whose population grows with the store, which the `candidate_floor` attribute
comment in `memvara/retrieve/hybrid.py` explains, and #160 tracks the form that reads the
band instead of assuming it. A larger store that loses a probe the same way is a
one-line change to the default in `memvara/retrieve/hybrid.py` and a redeploy;
`read_candidate_floor` is a constructor argument, so a library caller can set it without
waiting for that.

---

Previous: [Benchmarks](../BENCHMARKS.md) · Next: [Limitations](../LIMITATIONS.md) · [Documentation index](../README.md)
