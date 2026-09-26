# Soak and performance (S1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run memvara for thousands of seeded turns and fail when one of the silent failure modes that `memvara/telemetry.py` lists appears, and time the reads and writes an agent depends on at three store sizes, so that a slow hook or a regression is caught by a rule fixed before any number was measured.

**Architecture:**
- `bench/soak.py` drives a real `Memvara` (a SQLite file, the hashing embedder, no model, the built-in redactor, a `MemoryRecorder`) through a seeded workload of user turns, direct writes and reads. The turns are spread over 21 simulated days that end when the run starts. Each detector is a pure function of what the run observed, so a test can show it firing on a hand-built observation as well as on a real run with a fault injected.
- `bench/perf_budget.py` builds stores of 1,000, 10,000 and 100,000 claims, times `recall`, `search` and `remember` in fresh child processes (the first call in each of 30 processes, then 200 calls in one process), and times the session-start and recall hooks through `harness.hooks.HookRunner`. It holds the statistics (percentiles, a bootstrap interval, the median absolute deviation, rounding up to a 1-2-5 step) and the budget decision rule, as pure functions.
- The fast tier (`tests/adversarial/soak/`) unit-tests the statistics and the rules, shows each detector firing on an injected fault and staying quiet on a healthy 200-turn soak, and runs both scripts end to end at a tiny size. The nightly tier runs the 10,000-turn soak and the full timing run; the weekly tier runs the 100,000-turn soak.

**Tech Stack:** Python 3.10–3.13, pytest, the standard library, numpy (already a dependency of memvara), `memvara.telemetry.MemoryRecorder`, `memvara.redact.PatternRedactor`, `bench/evalkit.py` (`percentile`, `build_embedder`), `harness.env.child_env`, `harness.hooks.HookRunner`, `harness.invariants.check_store_integrity`.

**Spec:** `docs/superpowers/specs/2026-09-25-adversarial-test-suite-design.md`, section "Phase 4: Long-horizon", S1.

## Global Constraints

- The soak runs 10,000 seeded turns nightly and 100,000 weekly, through the real library, with the hashing embedder and no model.
- Detectors, with the failure rule copied from the design's table:
  - predicate explosion: distinct predicates exceed 1.1 × the vocabulary;
  - recency not refreshed: the median rank correlation is ≤ 0;
  - flip-flop row growth: a single-valued slot has more than 1 live claim, or `merged` stays at 0;
  - salience over relevance: the relevant claim ranks first less than 95% of the time;
  - script bias in the gate: tracked only, a script's rate below 0.8 × the Latin rate;
  - a retraction that retires nothing: it happens at all;
  - redaction drift: the ratio falls below 0.99;
  - store growth: the relative rule triggers.
- Performance: store sizes 1,000, 10,000 and 100,000 claims; cold is the first call in 30 fresh processes and warm is 200 calls; p50, p90, p95 and p99 with a bootstrap interval, for recall, search, remember and both hooks; the machine it ran on is recorded with each run; a run on battery or under load is marked invalid, not failed.
- The budget decision rule, fixed before any number is measured:
  1. hard ceilings: the recall hook's p95 ≤ 7.5 s at every size, cold and warm; its maximum ≤ 10 s; session start's p95 ≤ 20 s;
  2. after 14 valid nights, each library budget is 1.5 × the median p95, rounded up to the next 1-2-5 step, committed once with the machine fingerprint;
  3. a regression needs all three of: p95 more than 1.20 × the rolling 7-night median; an increase of more than max(2 ms, 3 × MAD); an immediate re-measure that reproduces it.
- This workstream writes the code that computes the budgets and the regression rule, not the numbers. No budget numbers are committed.
- The machine is shared, so the fast tier never asserts an absolute latency. Timing is asserted only in the nightly tier, and only on a valid run.
- Every child process gets its environment from `harness.env.child_env`. Every skip has a rule in `tests/harness/skips.py`.
- Every `search()` and `recall()` call in `bench/` passes `**PLAIN_READ` (`tests/test_read_stages.py` scans for it).
- Committed data is made up. Names, cities and companies are generated from syllables; phone numbers are in the fictional 555-01xx range; email domains end in `.example`.
- Test files are named `test_adv_*.py`, and every folder has an `__init__.py`.
- Write plainly: every sentence must be understood on its first reading. No AI or model name anywhere.

## Decisions this plan makes where the design is silent

Each of these is reported in the pull request as an interpretation, with its reason.

1. **Simulated time.** The turns are spread over 21 days that end at the moment the run starts. Valid time comes from that clock; transaction time is the wall clock, because `add()` always records at the wall clock. Twenty-one days is three half-lives of a fast-moving predicate (seven days each), so a fast-moving fact still carries a recency signal at the start of the span. A fixed span means the weekly run is ten times denser than the nightly one, not ten times longer.
2. **Consolidation once per simulated day**, at the wall clock. A pass can only see claims recorded at or before its own instant, and every claim is recorded at the wall clock.
3. **Which reads the recency detector reads.** Only the "panel" reads: eight subjects share the fast-moving, single-valued `working_on`, four of them restating one value and four changing theirs. Their claims tie on relevance, so only freshness and salience can order them. A search whose order is decided by relevance (a question naming one entity) reports a positive correlation even when reinforcement is broken, which was measured while calibrating: the median stayed at 0.29 with reinforcement disabled.
4. **What "relevant ranks first" asks.** Probes are worded "tell me where {person} lives". With "where does {person} live?", the hashing embedder cannot tell "live" from "likes" and ranked the right claim first only 62% of the time on a healthy store, so the detector would have measured the embedder rather than salience.
5. **The gate's rate per script** is the share of fact-carrying turns that the gate passes, counting only turns that reached the gate. A turn that repeats an earlier one word for word never reaches the gate, because tier 0 drops it first.
6. **The redaction ratio** is the share of turns carrying planted personal data whose text the redactor changed, per simulated day. The data is planted only in formats `PatternRedactor`'s docstring says it catches. The run also fails when `redact.inspected` is absent while turns are written, which is the "policy dropped" failure the telemetry docstring names.
7. **Store growth** is bytes on disk per turn after the store is closed. The relative rule is the regression rule, applied to earlier soaks with the same turn count and seed, with one SQLite page (4,096 bytes) of total growth in place of the 2 ms floor.
8. **"Under load"** means the one-minute load average, divided by the number of logical CPUs, is above 0.5 at any of the checks made before, between and after the store sizes. "On battery" is read from `pmset` on macOS and `/sys/class/power_supply` on Linux; elsewhere it is recorded as unknown.
9. **Hooks run without their daemon.** `child_env` switches the daemon off, and `HookRunner`'s daemon option lands with the hook-conformance work (A2). So every hook call pays interpreter start-up and a store open. Cold means a fresh home directory for each of 30 runs; warm means 200 runs sharing one home directory.
10. **Library budgets are checked once they exist.** When `bench/expected/perf_budgets.json` exists and its fingerprint matches the machine, a valid run fails on a library p95 above its budget.
11. **Where the nightly tests keep their records.** In `$NIGHTLY_RECORDS_DIR`, or `local/nightly/records/` in the checkout when it is unset, so that the nightly run can point successive clean worktrees at one history.
12. **An invalid timing run skips its nightly test** with the reason "the performance run is invalid: …", which has a rule in the skip ledger. A pass would read as healthy, and a failure would blame memvara for the machine.

## Review Focus

1. **A detector whose series never arrives.** If the telemetry is disconnected, no rank correlation, retraction count or redaction count is recorded, and a detector that treats "nothing" as "fine" passes forever. Each detector fails with "not measured" when its evidence is missing. Task 4 pins it for every detector.
2. **History from a different run.** A 10,000-turn soak compared with a 100,000-turn one, or a timing night from another machine, would trigger or mask the rules. History is filtered by turn count and seed (soak) and by fingerprint and validity (timing). Tasks 5 and 6 pin it.
3. **A hook that never read the store.** A hook pointed at no store answers "not configured" faster than a real read, and a timing run would record a fast, healthy-looking number. Each hook reply is checked, and a series with no successful read is refused. Task 6 pins it.
4. **A soak started on an old store.** Reusing a file would add the old rows to the growth figure and the old values to every slot. The soak refuses a path that already exists. Task 5 pins it.
5. **A hook that times out.** A timed-out run has no elapsed time, and dropping it would hide the worst sample. It is recorded at the hook's limit and counted as a breach of the recall hook's 10-second maximum. Task 1 pins the rule and Task 6 the recording.

---

## File structure

| File | Responsibility |
|---|---|
| `bench/perf_budget.py` | The statistics and the budget rules (pure functions); the machine fingerprint and the run conditions; building stores, timing library calls in child processes and hooks through `HookRunner`; the run record and the command line. |
| `bench/soak.py` | The workload, the run, the detectors, the soak record and the command line. Imports the regression rule and the fingerprint from `perf_budget`. |
| `tests/adversarial/soak/conftest.py` | Puts `bench/` on the import path, as `tests/test_bench_eval.py` does. |
| `tests/adversarial/soak/test_adv_perf_stats.py` | Percentiles, the bootstrap interval, the median absolute deviation, rounding to 1-2-5, the regression rule, budgets and ceilings. |
| `tests/adversarial/soak/test_adv_perf_conditions.py` | The fingerprint and the rules for an invalid run. |
| `tests/adversarial/soak/test_adv_soak_workload.py` | The workload is seeded, stays inside its vocabulary, and plants what the detectors need. |
| `tests/adversarial/soak/test_adv_soak_detectors.py` | Each detector on hand-built observations, including "not measured". |
| `tests/adversarial/soak/test_adv_soak_faults.py` | A healthy 200-turn soak is quiet, and each detector fires on its injected fault. |
| `tests/adversarial/soak/test_adv_soak_run.py` | The tiny soak end to end: the record, the history, the command line, the store's integrity. |
| `tests/adversarial/soak/test_adv_perf_run.py` | A tiny timing run end to end, and the refusal of a hook that did not read the store. |
| `tests/adversarial/soak/nightly/test_adv_soak_nightly.py` | The 10,000-turn soak. |
| `tests/adversarial/soak/nightly/test_adv_perf_nightly.py` | The full timing run and the budget decision rule. |
| `tests/adversarial/soak/weekly/test_adv_soak_weekly.py` | The 100,000-turn soak. |
| `tests/harness/skips.py` | One rule: the skip of an invalid timing night. |
| `docs/claude/testing.md` | A section "Soak and performance", before the final `Next:` line. |
| `docs/claude/telemetry-and-benchmarks.md` | The two scripts in the list of benchmark scripts. |

---

### Task 1: The statistics and the budget rules

**Files:**
- Create: `bench/perf_budget.py` (the pure functions only), `tests/adversarial/soak/__init__.py`, `tests/adversarial/soak/conftest.py`
- Test: `tests/adversarial/soak/test_adv_perf_stats.py`

**Interfaces:**
- Produces, in `perf_budget`:
  - `summarize(samples: Sequence[float], *, resamples: int = 1000, seed: int = 0) -> dict[str, Any]` with keys `n`, `p50`, `p90`, `p95`, `p99`, `max` and `interval`, where `interval` maps `p50`…`p99` to `[low, high]`. Percentiles are `evalkit.percentile` (nearest rank).
  - `bootstrap_interval(samples, q, *, resamples=1000, confidence=0.95, seed=0) -> tuple[float, float]`.
  - `round_up_125(value: float) -> float`: the smallest value in {1, 2, 5} × 10^k at or above `value`.
  - `mad(values: Sequence[float]) -> float`: the median of absolute deviations from the median, unscaled.
  - `Verdict(outcome: str, current: float, median: float | None, mad: float | None, remeasured: float | None, detail: str)`, frozen, where `outcome` is `"ok"`, `"regression"`, `"not reproduced"` or `"no history"`.
  - `judge(current: float, history: Sequence[float], remeasure: Callable[[], float], *, floor: float) -> Verdict`. It uses the last `ROLLING_NIGHTS` (7) values of `history` and calls `remeasure` only when the first two conditions hold.
  - `derive_budgets(nights: Sequence[Mapping[str, float]]) -> dict[str, float]`: for each series, `round_up_125(1.5 × median p95)`. Refuses fewer than `BUDGET_NIGHTS` (14) with `ValueError`.
  - `CEILINGS`, and `check_ceilings(series: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]`, one entry per checked series and statistic with `series`, `stat`, `limit_ms`, `value_ms` and `breached`.
  - `series_key(operation: str, size: int, temperature: str) -> str`, for example `hook.recall@1000/cold`.
  - Constants: `LATENCY_FLOOR_MS = 2.0`, `ROLLING_NIGHTS = 7`, `REGRESSION_RATIO = 1.20`, `MAD_MULTIPLIER = 3.0`, `BUDGET_NIGHTS = 14`, `BUDGET_MULTIPLIER = 1.5`.

The rules, exactly:

```python
def round_up_125(value: float) -> float:
    if value <= 0:
        raise ValueError(f"a budget must be positive, not {value}")
    exponent = math.floor(math.log10(value))
    for decade in (exponent - 1, exponent, exponent + 1):
        for step in (1.0, 2.0, 5.0):
            candidate = step * 10.0 ** decade
            if candidate >= value * (1 - 1e-9):
                return candidate
    raise AssertionError("unreachable: 10 ** (exponent + 1) is above value")


def is_regression(current: float, history: Sequence[float], *, floor: float) -> bool:
    centre = statistics.median(history)
    return (current > REGRESSION_RATIO * centre
            and current - centre > max(floor, MAD_MULTIPLIER * mad(history)))
```

- [ ] **Step 1: Write the failing tests.**
  - `summarize([1..100])`: p50 is 50 or 51 by nearest rank (whatever `evalkit.percentile` returns, asserted against it), p99 is 99 or 100, `max` is 100, `n` is 100. Each interval contains its point estimate, and `low <= high`.
  - The same samples and seed give the same interval twice; a constant sample gives an interval of zero width.
  - `round_up_125`: 0.7 → 1, 1 → 1, 1.01 → 2, 2 → 2, 3 → 5, 5 → 5, 7.5 → 10, 13 → 20, 150 → 200, 0.012 → 0.02, and 0 and −1 raise `ValueError`. A value that is a step except for floating-point noise stays on the step: `1.5 * 4 / 3`, which is 2.0000000000000004, gives 2, while 2.01 gives 5.
  - `mad([1, 1, 2, 2, 4, 6, 9])` is 1.
  - `judge` with six nights of history: `"no history"`, and `remeasure` is not called.
  - With seven nights at 10 ms: 12.5 ms is `"ok"` (the ratio fails); 11 ms with a floor of 0.5 is `"ok"` (the ratio fails); with history 100 ms and a spread of 30 ms, 125 ms is `"ok"` (the increase is below 3 × MAD); 13 ms whose re-measure is 10 ms is `"not reproduced"`; 13 ms whose re-measure is 13 ms is `"regression"`. `remeasure` is called only in the last two cases.
  - An increase that meets the ratio but not the 2 ms floor (history 1 ms, current 2.5 ms) is `"ok"`.
  - `derive_budgets` with 13 nights raises `ValueError`. With 14 nights whose `search@1000/warm` p95 values have a median of 8 ms, the budget is 20 ms (1.5 × 8 = 12, rounded up to 20).
  - `check_ceilings`: a recall hook p95 of 7,500 ms is not a breach and 7,501 ms is; a recall hook maximum of 10,001 ms is a breach; a recall hook series with `timeouts > 0` breaches the maximum whatever its values; a session-start p95 of 20,001 ms is a breach; library series are not checked.
- [ ] **Step 2: Run them and watch them fail** on `ModuleNotFoundError: No module named 'perf_budget'`.
- [ ] **Step 3: Write the pure half of `bench/perf_budget.py`**, and `tests/adversarial/soak/conftest.py`, which inserts `bench/` into `sys.path`.
- [ ] **Step 4: Run them and watch them pass.**
- [ ] **Step 5: Commit** `bench/perf_budget.py`, the two new test-package files and the test file, by name.

### Task 2: The machine fingerprint and an invalid run

**Files:**
- Modify: `bench/perf_budget.py`
- Test: `tests/adversarial/soak/test_adv_perf_conditions.py`

**Interfaces:**
- Produces, in `perf_budget`:
  - `machine_fingerprint() -> dict[str, Any]` with `system`, `release`, `machine`, `cpu`, `logical_cpus`, `memory_bytes`, `python`, `sqlite`, `memvara`, `commit` and `id`. `id` is a short SHA-256 of the hardware fields only (`system`, `machine`, `cpu`, `logical_cpus`, `memory_bytes`), so an operating-system or Python update does not orphan a machine's budgets.
  - `Conditions(on_battery: bool | None, load_per_cpu: float)`, frozen; `read_conditions() -> Conditions`.
  - `invalid_reasons(checks: Sequence[Conditions]) -> list[str]`, empty for a valid run. `MAX_LOAD_PER_CPU = 0.5`.

- [ ] **Step 1: Write the failing tests.**
  - Two calls of `machine_fingerprint()` return the same `id`; every key above is present; `logical_cpus` equals `os.cpu_count()`.
  - `invalid_reasons([Conditions(False, 0.2)])` is empty; `on_battery=None` is valid; one check on battery gives one reason naming the battery; one check at 0.9 load per CPU gives one reason naming the load and the limit; both give two reasons.
  - `read_conditions()` returns a load that is a non-negative float, and `on_battery` is a bool or `None`.
- [ ] **Step 2: Run them and watch them fail** on the missing names.
- [ ] **Step 3: Write them.** `pmset -g batt` on macOS ("Battery Power" means on battery); `/sys/class/power_supply/*/online` and `type` on Linux; `None` elsewhere. `os.getloadavg()`. Where it does not exist (Windows), the load is recorded as 0.0, because the nightly tiers run only on the maintainer's Mac.
- [ ] **Step 4: Run them and watch them pass.**
- [ ] **Step 5: Commit.**

### Task 3: The soak's workload

**Files:**
- Create: `bench/soak.py` (the workload)
- Test: `tests/adversarial/soak/test_adv_soak_workload.py`

**Interfaces:**
- Produces, in `soak`:
  - `VOCABULARY: dict[str, bool]`: each predicate the workload writes, mapped to whether it holds one value at a time: `name`, `lives_in`, `works_at` and `working_on` are single-valued; `likes` and `prefers` are not.
  - `SoakConfig(turns: int, seed: int = 0)`, frozen, with `span = timedelta(days=21)` and the property `per_day` (turns per simulated day, at least 1). `MIN_TURNS = 100`.
  - `Turn`, frozen: `index`, `kind` (`"say"`, `"remember"`, `"probe"`, `"panel"` or `"recall"`), `text`, `subject`, `predicate`, `obj`, `polarity`, `script`, `fact`, `planted` and `gold` (`tuple[str, str, str] | None`).
  - `Workload(config: SoakConfig)`: iterating it yields `config.turns` turns. Two methods are the seams the fault tests patch: `pii_text(index: int) -> str` and `retraction_text(item: str) -> str`.
  - `PANEL_QUERY = "who is working on what"`, `PREFERENCES` (three sentences) and `variant(sentence, which) -> str`.

The world the workload describes: 16 people with a city, an employer and a much-restated "heavy" like; the user, with a name, a city, an employer, four favourite likes and one-off likes; eight panel subjects under `working_on`; three sentence preferences and their near-duplicates. The first turns introduce everything, in a fixed order: every person's city, every panel value, the user's name, city, employer and favourites, one one-off like and its retraction, one personal-data turn, and a preference with its near-duplicate. Then each turn draws its kind: a Latin fact 22%, filler 5%, a non-Latin fact 6%, planted personal data 3%, a person's fact 22% (spelled with a random alias of its predicate), a panel write 12%, a preference 3%, a probe 14%, a panel read 10%, a recall 3%.

- [ ] **Step 1: Write the failing tests.**
  - The same config yields the same turns twice; a different seed yields a different sequence; fewer than 100 turns raises `ValueError`.
  - Every predicate spelling a turn uses normalizes, through a fresh `PredicateRegistry()`, to a key of `VOCABULARY`, and every key is used.
  - Each preference and its two near-duplicates reach the hashing embedder's merge threshold (`calibration_of(HashingEmbedder(dim=512)).merge`) when rendered as `user prefers …`, so a healthy merge pass has work to do.
  - Every planted turn is changed by `PatternRedactor().redact(..., field="episode", ...)`, and no other turn is.
  - Replaying the turns: every retraction names a one-off the user likes and has not yet taken back, and every probe's gold is the city the workload last gave that person.
  - Every non-Latin fact turn is classified by `telemetry.script_of` as the script the turn says, and the seven scripts all appear in a 2,000-turn workload.
- [ ] **Step 2: Run them and watch them fail** on the missing module.
- [ ] **Step 3: Write the workload.** Names come from a seeded syllable generator, and a one-off like is always a new word, so that no sentence the workload retracts is ever said again word for word.
- [ ] **Step 4: Run them and watch them pass.**
- [ ] **Step 5: Commit.**

### Task 4: Running the soak, and the detectors

**Files:**
- Modify: `bench/soak.py`
- Test: `tests/adversarial/soak/test_adv_soak_detectors.py`, `tests/adversarial/soak/test_adv_soak_faults.py`

**Interfaces:**
- Consumes: `Workload`, `SoakConfig`, `VOCABULARY`; `evalkit.build_embedder("hashing")`; `perf_budget.judge`.
- Produces, in `soak`:
  - `SoakRecorder(MemoryRecorder)`: records everything `MemoryRecorder` does, and keeps the rank correlations that arrive while `capture` is a list.
  - `Observations`, a dataclass: `config`, `recorder`, `predicates: set[str]`, `crowded: list[tuple[str, str, int]]` (single-valued slots seen with more than one live claim, at any check), `panel_correlations: list[float]`, `probes: int`, `probe_hits: int`, `gate: dict[str, list[int]]` (script to `[reached, passed]` over fact-carrying turns), `planted: list[tuple[int, bool]]` (simulated day and whether the redactor changed the turn), `unplanted_changed: int`, `store_bytes: int | None`, `elapsed_s: float`.
  - `run(config: SoakConfig, path: Path | None, *, workload: Workload | None = None, options: Mapping[str, Any] | None = None) -> Observations`. `path=None` runs in memory, which measures everything except store growth. `options` override the `Memvara` keyword arguments (`redactor`, `registry`, `read_w_salience`, and so on), which is how a test injects a configuration fault. A path that exists raises `FileExistsError`.
  - `Finding(detector: str, status: str, value: float | None, threshold: float | None, detail: str)`, frozen; `status` is `"ok"`, `"fail"` or `"tracked"`.
  - The detectors, each `(obs: Observations) -> Finding`: `predicate_explosion`, `recency_refresh`, `flip_flop`, `salience_over_relevance`, `script_bias`, `retraction_noop`, `redaction_drift`; and `store_growth(obs, history: Sequence[float], remeasure: Callable[[], float]) -> Finding`.
  - `DETECTORS`, the first seven in that order, and `judge_soak(obs, *, history=(), remeasure=None) -> list[Finding]`.

How `run` executes a turn at simulated time `t`: a `say` turn is `mem.add(text, ts=t)`, with the gate and redaction counters read before and after; a `remember` turn is `mem.remember(subject, predicate, obj, polarity=..., valid_from=t)`; a `probe` is `mem.search(text, k=10, **PLAIN_READ)` and counts a hit when the first result's subject, predicate and object equal the gold; a `panel` read is `mem.search(PANEL_QUERY, k=8, **PLAIN_READ)` with the correlations captured; a `recall` is `mem.recall(text, **PLAIN_READ)`. After every `per_day` turns, `mem.consolidate()`, then every live claim in a single-valued slot is counted by `fact_key`. At the end, the distinct predicates are read from `get_all(states=("live", "ended", "retired"))`, the store is closed, and its files are measured.

Each detector's failure rule, and its "not measured" case (Review Focus 1):

| Detector | Fails when | Not measured, and so fails, when |
|---|---|---|
| predicate explosion | distinct predicates > 1.1 × `len(VOCABULARY)` | no claim is stored |
| recency refresh | median panel correlation ≤ 0 | no panel read produced a correlation |
| flip-flop | any single-valued slot held more than 1 live claim at any check, or `consolidate.merged` totals 0 | no consolidation pass ran |
| salience over relevance | probe hits / probes < 0.95 | no probe ran |
| script bias | never: `tracked`, naming each script whose rate is below 0.8 × the Latin rate | no Latin fact reached the gate (still `tracked`) |
| retraction | `write.retraction{outcome="noop"}` > 0 | no retraction was counted at all |
| redaction drift | the lowest per-day ratio is below 0.99 | nothing was planted, or `redact.inspected` is absent while `write.turns` is not |
| store growth | `judge` returns `"regression"` | the store was in memory, or fewer than 7 earlier soaks match (both `tracked`) |

- [ ] **Step 1: Write the failing detector tests** on hand-built `Observations` (a helper builds a healthy one, and each test changes one field): each detector's boundary (for example 6 distinct predicates with a vocabulary of 6 is `ok` and 7 fails; a median of 0.0 fails and 0.01 passes; 95 hits in 100 probes passes and 94 fails; one no-op retraction fails; a day at 0.98 fails), and each "not measured" row of the table.
- [ ] **Step 2: Write the failing fault tests.** One healthy 200-turn soak, shared by a module fixture, is `ok` or `tracked` on every detector, and script bias flags no script. Then one soak per fault, each asserting that the named detector fails:
  - predicate explosion: `PredicateRegistry.normalize` patched to return its argument slugified, so aliases stop folding;
  - recency refresh: `Reconciler.reinforce` patched to raise the count only, without refreshing the observation time or the storage strength (the bug `telemetry.py` describes);
  - flip-flop: a registry in which `lives_in`, `works_at` and `working_on` are declared as holding many values; and separately, the merge threshold patched above 1.0, so `merged` stays at 0;
  - salience over relevance: `options={"read_w_salience": 1.0}`;
  - script bias: `memvara.write.gate._MIN_UNSPACED_CHARS` patched to 10**6, so Han and kana turns are dropped as too short; the finding is `tracked` and names `han` and `kana`, and the run does not fail;
  - retraction: `Workload.retraction_text` patched to misspell the item, so the retraction matches nothing;
  - redaction drift: `Workload.pii_text` patched to write unpunctuated phone numbers after the midpoint, which the redactor's docstring says it does not catch; and separately, `options={"redactor": None}`.
- [ ] **Step 3: Run both files and watch them fail** on the missing names.
- [ ] **Step 4: Write `SoakRecorder`, `Observations`, `run`, `Finding` and the detectors.**
- [ ] **Step 5: Run them and watch them pass.** Then check each fault test's failure for the right reason by reading its finding's `detail`, not only its status.
- [ ] **Step 6: Commit.**

### Task 5: The soak's record, history and command line

**Files:**
- Modify: `bench/soak.py`, `docs/claude/testing.md`, `docs/claude/telemetry-and-benchmarks.md`
- Test: `tests/adversarial/soak/test_adv_soak_run.py`

**Interfaces:**
- Produces, in `soak`:
  - `record(obs: Observations, findings: Sequence[Finding]) -> dict[str, Any]`: `kind` (`"memvara-soak"`), `version` (1), `started`, `turns`, `seed`, `fingerprint`, `bytes_per_turn`, `elapsed_s`, `counts` (the recorder's reconcile, retraction and merge totals) and `findings`.
  - `load_history(directory: Path, *, turns: int, seed: int) -> list[float]`: `bytes_per_turn` from every record in `directory` with the same turns and seed, oldest first. A file that is not a soak record is skipped.
  - `main(argv: Sequence[str] | None = None) -> int`: `--turns`, `--seed`, `--store` (default: a temporary directory), `--out` (write the record), `--history` (compare store growth). Prints one line per finding. Returns 1 when any finding fails, otherwise 0.

- [ ] **Step 1: Write the failing tests.**
  - `main(["--turns", "200", "--out", <file>])` returns 0, prints one line per detector, and writes a record whose findings are all `ok` or `tracked`.
  - The store left by a 200-turn soak passes `harness.invariants.check_store_integrity`.
  - `load_history` returns the growth figures of matching records only: a record with other turns, another seed, or other JSON in the folder is left out (Review Focus 2).
  - Store growth end to end: a healthy soak gives a figure; seven copies of it are the history; a soak with exact-duplicate detection disabled (`SQLiteStore.find_by_value` patched to find nothing, so every restatement adds a row) fails `store_growth`, and its re-measure is a second faulty soak. A second healthy soak against the same history is `ok`.
  - `run()` refuses a store path that already exists (Review Focus 4).
- [ ] **Step 2: Run them and watch them fail.**
- [ ] **Step 3: Write `record`, `load_history` and `main`.**
- [ ] **Step 4: Write the documentation for the soak:** the first half of the "Soak and performance" section in `docs/claude/testing.md` (what a soak is, the detectors table, how to run one by hand, what the fault tests show), and `bench/soak.py` in the list in `docs/claude/telemetry-and-benchmarks.md`.
- [ ] **Step 5: Run the tests and the documentation tests** (`tests/test_docs.py`, `tests/test_doc_links.py`) **and watch them pass.**
- [ ] **Step 6: Commit.**

### Task 6: Timing library calls and hooks

**Files:**
- Modify: `bench/perf_budget.py`, `docs/claude/testing.md`, `docs/claude/telemetry-and-benchmarks.md`
- Test: `tests/adversarial/soak/test_adv_perf_run.py`

**Interfaces:**
- Consumes: the Task 1 and Task 2 functions; `harness.env.child_env`; `harness.hooks.HookRunner`, `HookTimeout`.
- Produces, in `perf_budget`:
  - `PerfConfig(sizes=(1_000, 10_000, 100_000), cold=30, warm=200, resamples=1000, seed=0)`, frozen.
  - `build_store(path: Path, claims: int, *, seed: int = 0) -> list[str]`: writes exactly `claims` live claims through `Memvara.remember` (four per made-up person: a city, an employer and two likes), in batches of 1,000, and returns the people's names for the queries.
  - `PerfError(RuntimeError)`: a measurement that did not measure what it names.
  - `measure(config: PerfConfig, workdir: Path, *, history: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]`: the run record: `kind` (`"memvara-perf"`), `version`, `date`, `started`, `finished`, `fingerprint`, `config`, `conditions`, `valid`, `invalid_reasons`, `series` (each a `summarize` result plus `samples_ms` and `timeouts`), `ceilings`, `regressions` (a `Verdict` per series, as a dict) and `budgets` (`None`, or each library series against a committed budget).
  - `load_runs(directory: Path) -> list[dict[str, Any]]`; `nights_for(runs, fingerprint_id) -> list[dict]`: valid runs on this machine, the last one of each date, oldest first.
  - `write_budgets(runs, fingerprint, path: Path) -> dict`: refuses when `path` exists or fewer than 14 nights qualify.
  - `main(argv) -> int`: `--sizes`, `--cold`, `--warm`, `--out`, `--history`, `--write-budgets PATH`. Exit 0 on a valid, passing run; 1 on a ceiling breach, a confirmed regression or a budget exceeded; 3 on an invalid run, which is reported and not failed.
  - Child mode: `python bench/perf_budget.py --child '<json>'` opens the store, makes `warmup` untimed calls and then `calls` timed calls of one operation, and prints `{"ms": [...]}`. Every child runs with `child_env` and a scratch home.

How each series is measured:
- Library, cold: 30 child processes, each timing its first call. Library, warm: one child, five untimed calls, then 200 timed. `search` and `recall` read the built store; `remember` writes to a copy of it, made after the reads, and each call writes a new fact.
- Hooks: `HookRunner("claude", server_env={"MEMVARA_DB": ..., "MEMVARA_USER": ...})`. Cold: 30 runs, each with a new home directory and session. Warm: 200 runs sharing one home and one session, each prompt naming a different person. A run past the hook's limit is recorded at the limit and counted in `timeouts`. Every reply's status must say the store was read (`recalled`, `already in context`, `no matching memories` or `standing preferences updated` for recall; `session opened` for session start), and at least one recall reply in each series must say `recalled`; otherwise `PerfError` (Review Focus 3).
- Conditions are read before the first size, after each size, and at the end.
- After measuring, each series is judged against the history with `judge`, whose re-measure measures that series again.

- [ ] **Step 1: Write the failing tests.**
  - `build_store(path, 40)` stores exactly 40 live claims and returns ten names.
  - A tiny run, `PerfConfig(sizes=(40,), cold=2, warm=3, resamples=50)`, produces all ten series (three library operations and two hooks, cold and warm), each with the right number of positive samples; a `valid` flag that agrees with `invalid_reasons`; six ceiling entries (the recall hook's p95 and maximum, and session start's p95, each cold and warm); and a `no history` verdict for every series. Nothing about how long a sample took is asserted.
  - A hook pointed at a store path that does not exist raises `PerfError` naming "not configured".
  - `nights_for` keeps only valid runs with the given fingerprint, and one per date (Review Focus 2).
  - `write_budgets` refuses 13 nights and refuses an existing file; with 14 nights it writes the fingerprint, the dates and the budgets.
  - A recall hook run that times out is recorded at the limit and counted, and `check_ceilings` reports it (Review Focus 5); the timeout is simulated by a `HookRunner` stub that raises `HookTimeout`.
- [ ] **Step 2: Run them and watch them fail.**
- [ ] **Step 3: Write the measurement half of `bench/perf_budget.py`.**
- [ ] **Step 4: Write the documentation for timing:** the second half of the "Soak and performance" section (what is timed, cold and warm, the three ceilings, the budget rule, an invalid night, how to run it by hand), and `bench/perf_budget.py` in the benchmark list.
- [ ] **Step 5: Run the tests and the documentation tests and watch them pass.**
- [ ] **Step 6: Commit.**

### Task 7: The nightly and weekly tiers

**Files:**
- Create: `tests/adversarial/soak/nightly/__init__.py`, `test_adv_soak_nightly.py`, `test_adv_perf_nightly.py`; `tests/adversarial/soak/weekly/__init__.py`, `test_adv_soak_weekly.py`
- Modify: `tests/harness/skips.py` (one rule), `docs/claude/testing.md` (the tiers paragraph of the section)

**Interfaces:**
- Consumes: `soak.run`, `soak.judge_soak`, `soak.record`, `soak.load_history`; `perf_budget.measure`, `perf_budget.load_runs`, `perf_budget.PerfConfig`.
- Produces: the record files `soak-<turns>-<seed>-<timestamp>.json` and `perf-<timestamp>.json` in the records folder.

- [ ] **Step 1: Write the tests.**
  - Nightly soak: 10,000 turns, seed 0, a file store in `tmp_path`; the record is written to the records folder before any assertion, so a failing night still leaves its evidence; then no finding may fail, and the message names each failing detector with its detail.
  - Weekly soak: the same at 100,000 turns.
  - Nightly timing: `measure(PerfConfig(), tmp_path, history=...)`; the record is written first; an invalid run skips with "the performance run is invalid: <reasons>"; a valid run fails on any breached ceiling, any `"regression"` verdict, and any budget exceeded.
  - The skip ledger rule: `SkipRule(r"^the performance run is invalid: ", "A run on battery or under load measures the machine, not memvara. The design marks it invalid rather than failed, and the record keeps the reason.")`.
- [ ] **Step 2: Run the nightly tier once on this machine** (`--tier nightly tests/adversarial/soak`) and the weekly soak once, and record their wall times and results for the pull request.
- [ ] **Step 3: Write the tiers paragraph** of the documentation section.
- [ ] **Step 4: Commit.**

### Task 8: Flakes, the gate and the type checks

- [ ] **Step 1:** Run each new fast-tier test file 20 times in a row; every run must pass.
- [ ] **Step 2:** Run the full gate as two commands with a private coverage file; coverage of `memvara/` must stay at 100%.
- [ ] **Step 3:** Run `mypy -p memvara`, and `mypy tests/harness` with and without `--ignore-missing-imports`.
- [ ] **Step 4:** Quote each result line in the report.
