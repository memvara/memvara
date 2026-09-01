# Code review — PR 156 (`memvara/memvara`), commit `31d3193`

**Command run:** `Skill({skill: "code-review", args: "high 156"})`, then executed manually
(diff gathering, reading, and one empirical reproduction) rather than fanning out
background finder agents, per the dispatching task's instruction to keep the work in this
turn. Effort level: **high**. No `--comment`, no `--fix`, no `ultra`.

**Diff scope check (explicitly required by the task):** `git diff main...HEAD` pulls in
41 files / 6417 lines — the stacked base PR 153 (`fix/bench-hosted-db-embedder`, tip
`a49f27f`) plus several unrelated open branches merged into the shared worktree's view of
`main`. That is the wrong diff. The correct scope is `git diff a49f27f..31d3193` (equal to
`git show 31d3193`): 4 files, ~410 lines — `CHANGELOG.md`, `bench/hosted.py`,
`docs/BENCHMARKS.md`, `tests/test_bench_hosted.py`. All findings below are confined to
that diff.

---

## Finding 1 — CONFIRMED (reproduced empirically). Severity: high (correctness).

**`scope_blindspot()` compares two counts with mismatched state semantics, producing a
false-positive warning with actively wrong remediation advice on a store that has zero
*live* claims but nonzero retired/ended claim rows — even when the scope is entirely
correct.**

`bench/hosted.py`, `scope_blindspot()` (around line 117-133):

```python
if mem.count():
    return None
...
total = stats(None).get("claims", 0)
...
return (f"WARNING: 0 of this store's {total} claims are visible at scope "
        f"{key} — ... Re-run with --tenant/--user naming the scope the claims "
        f"are under.")
```

- `mem.count()` defaults to `states=("live",)` (`resolve_states()` default is
  `LIVE_ONLY`, see `memvara/store/base.py` and `memvara/core.py:1962`) — it counts only
  currently-believed-true claims, scope-filtered.
- `stats(None)["claims"]` (`memvara/store/sqlite.py:3069`) is `SELECT COUNT(*) FROM
  claims` with **no state filter at all** — it counts every row: live, ended, *and
  retired*, across every tenant, unfiltered.
- `SQLiteStore.stats()` already exposes a key that matches `count()`'s semantics:
  `live_claims`, computed with the same `("live",)` state clause. The code uses
  `"claims"` instead of `"live_claims"`.

**Reproduction** (run against this worktree, `PYTHONPATH=.`):

```python
with Memvara(db, embedder=HashingEmbedder(dim=64), llm=NullLLM()) as mem:
    mem.remember("larkspur", "test_flag", "the Larkspur suite needs -j1")
    mem.forget("larkspur", "test_flag")   # retire it — correct tenant, correct scope

store = SQLiteStore(db)
mem2 = Memvara(store=store, embedder=hosted._store_embedder(store),
               tenant="default", user=None)
mem2.count()          # -> 0   (correct: nothing live)
store.stats(None)     # -> {'claims': 1, 'live_claims': 0, 'ended_claims': 0,
                       #     'invalidated': 1, 'embeddings': 1}
hosted.scope_blindspot(mem2)
# -> "WARNING: 0 of this store's 1 claims are visible at scope default/*/*/* —
#     every probe will miss and --draft will print nothing. Re-run with
#     --tenant/--user naming the scope the claims are under."
```

This is exactly the ambiguity the PR exists to remove, reappearing on the other axis: a
store with a retired claim, correctly scoped, is told to re-run at a different
`--tenant`/`--user` — advice that is simply wrong, since no such scope exists; the claim
was retired, not misplaced. Any real store with correction history (superseded facts,
`forget()` calls) can trigger this the moment its *live* population (globally, not just
at the queried scope) happens to be empty while historical rows remain.

**Fix:** use `stats(None).get("live_claims", 0)` instead of `.get("claims", 0)`. That
key already exists on `SQLiteStore.stats()` and carries exactly the state semantics
`mem.count()` uses, so the comparison becomes apples-to-apples with no other change
needed.

**Test-coverage angle (same root cause):** `test_the_scope_warning_is_silent_on_a_file_
that_is_genuinely_empty` is the only "must not false-positive" test, and it uses a store
that was *never written to* (`with Memvara(...): pass`) — zero rows, not zero *live*
rows. It does not exercise the retired/ended-claims path and would not catch this
regression (or, in this case, did not catch the bug shipping in the first place). This
matches the review brief's explicit warning to watch for "a test that passes because a
fixture store is empty either way" — it does, on the one input that would have caught
this.

---

## Everything else checked — no further findings survived

- **Stream discipline:** the only emission site is `print(blindspot, file=sys.stderr)`
  in `main()` (one call site, gated once, before both the `--draft` branch and the
  scoring branch). `test_main_warns_on_stderr_when_the_scope_sees_nothing_the_file_
  holds` asserts `out.out == ""` positively, so a regression sending the warning to
  stdout would be caught. Clean.
- **Hosted route never asked for whole-store counts:** the check is gated by
  `if args.db:` in `main()`, before `scope_blindspot` is ever called; `RemoteMemvara`
  has no `.store` attribute at all (verified: no `self.store` anywhere in
  `memvara/remote/api.py`), so even a hypothetical bypass would hit `getattr(...,
  None)` and return `None` rather than leak. `--tenant` on the hosted route is refused
  before any client is constructed, confirmed by `_open_store`'s early
  `SystemExit`. Clean.
- **The three deliberate decisions** (refuse `--tenant` on hosted rather than
  forward it; no `--agent`/`--session`; `compare_runs` warns `tenant None -> default`
  on old result files) are all sound. Verified against `RemoteMemvara.__init__`
  (`memvara/remote/api.py:105-115`): `tenant` really is accepted and only stored on
  `default_scope`, never sent — so forwarding really would be a silent no-op on the
  fingerprint. `--user` really is forwarded and really does narrow both routes
  (`RemoteMemvara`'s `default_scope` carries it into every scoped request). No
  disagreement with any of the three.
- **No file under `memvara/` changed; no new dependency.** Confirmed by the 4-file
  diffstat above.
- **No real store content in `bench/`/`tests/`.** All new tests use `tmp_path` or
  `:memory:` with `HashingEmbedder`/`NullLLM`; nothing reads `~/.memvara/`.
- **No test reaches `default_embedder()` or the network.** Checked every new test in
  the diff individually; the one test that opens a store with no embedder recorded
  (`test_the_scope_warning_is_silent_on_a_file_that_is_genuinely_empty`) monkeypatches
  both `default_embedder` bindings first, following the pattern already established
  earlier in the same file for the identical reason.
- **Documentation.** `CHANGELOG.md` and `docs/BENCHMARKS.md` both shipped in the same
  commit and accurately describe the code *as written* — including, faithfully, the
  buggy `stats(None)["claims"]` behavior (the docs are not wrong relative to the code;
  the code itself is wrong relative to its own stated intent). No AI attribution
  anywhere in the commit message, `CHANGELOG.md`, or `docs/BENCHMARKS.md`.
- Cross-file check: no other file in the repository calls `bench.hosted._open_store`,
  `store_fingerprint`, `compare_runs`, or `main` — no external call sites to break.
- No leftover un-migrated `argparse.Namespace(...)` test call site skipped the new
  `_args()` helper (checked: exactly one definition, all call sites updated).

## Confirmation

- Nothing posted to GitHub (no `gh pr comment`, no `gh pr review`, no `--comment`).
- No files changed in the repository; no commits made; no pushes.
- This file is scratch output only, not committed.
