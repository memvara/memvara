# Nightly orchestration (P5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task by task, with test-driven development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the adversarial suite's nightly steps in order, each within its own time cap, against a clean worktree of `origin/main`; write a report and a running history; turn each new confirmed break into one deduplicated filing (dry-run by default); and warn the maintainer when a night did not run.

**Architecture:**
- `scripts/nightly/` is a small package named `nightly`. A script run by path (`python3 scripts/nightly/run.py`) puts `scripts/` on the import path and imports its siblings as `nightly.<module>`. The tests do the same, so a module is loaded once and no generic name such as `run` or `flakes` enters `sys.modules`.
- `run.py` creates the worktree and the virtual environment, runs the step table, and writes `findings.jsonl`, `report.json` and `report.md` under `local/nightly/<date>/` in the main checkout, and one line per night in `local/nightly/history.jsonl`.
- Steps are child processes. Each has a cap in seconds; a step that runs past it is stopped with its whole process tree, and the run goes on to the next step. A step whose code has not landed on `main` is reported as "not built yet" with the reason and the file it waits for.
- `tests/harness/report.py` defines `Finding`, the one-JSON-line record every layer writes, and its `signature()`, the fingerprint that deduplicates a break across nights.
- Filing never runs unless the operator passes `--file`. The deterministic run files only findings a producer has already classified. A failed test from the regressions step is unclassified, so the scheduled session classifies it against `SECURITY.md` first (the prompt is `scripts/nightly/task.md`) and then files it with `scripts/nightly/filing.py`.
- `watchdog.py` uses no model and imports nothing from the package, so launchd can run it with any Python 3.10 or later.

**Tech Stack:** Python 3.10–3.13 standard library only in `scripts/nightly/` and `tests/harness/report.py`; pytest 8 or later; `git`; `gh` (only with `--file`, never in tests); `osascript` on macOS.

**Spec:** `docs/superpowers/specs/2026-09-25-adversarial-test-suite-design.md`, "Phase 5: Nightly orchestration", and the lines about `report.Finding`, flakes and notifications elsewhere in that document.

## Global Constraints

- Do not install anything: no scheduled task, no launchd job, no cron entry. The launchd file is a template with placeholders.
- Never call GitHub for real from code or tests. Filing is a dry run unless `--file` is given, and the tests replay recorded `gh` output through a fake runner.
- A security-class break (one that falls under `SECURITY.md`'s "In scope" section) never reaches a public issue, a public branch or a public pull request. It goes to a private draft advisory.
- Steps: preflight, regressions, agents, red team, hosted, production smoke, performance, soak, incremental mutation, replay, in that order, each with its own cap. Only preflight and regressions are built; `bench/soak.py`, `bench/perf_budget.py` and `bench/mutation.py` are not on `origin/main`.
- Notifications go out only for a new break, an isolation breach, or a dependency that has been down two nights in a row (and, from the watchdog, a night that did not run).
- The fast-tier tests live in `tests/adversarial/nightly_runner/`, are named `test_adv_*.py`, need no network, and pass unchanged on Linux, macOS and Windows under Python 3.10–3.13. Nothing may skip: this workstream cannot add rules to the skip ledger.
- Every child process a test starts gets its environment from `harness.env.child_env`.
- No AI or model name appears in any committed file or commit message, and no commit carries an attribution trailer.
- Write plainly: every sentence must be understood on its first reading.

## Review Focus

1. **A run killed partway through** (the Mac sleeps for good, the app quits): the heartbeat has a start and no finish. The watchdog must report the night as not finished, not stay silent because a heartbeat exists. Test in Task 6.
2. **A damaged line in `history.jsonl`** (a run killed while appending): every later night must still run. The reader skips the line and the report says so. Test in Task 2.
3. **The operator's paths in a failure's text** (`/Users/<name>/…`, `pytest-of-<name>`, the checkout path): they must be removed from anything sent to GitHub. Test in Task 5.
4. **A pytest run that fails with no failing test** (the skip ledger, a crash in a session hook): the step is failed and a failure that needs a person is reported, never a pass. Test in Task 4.
5. **A strict expected failure that starts to pass:** it is reported as "a known bug's test now passes", not filed as a new break. Test in Task 4.

---

### Task 1: Findings and their fingerprint

**Files:**
- Create: `tests/harness/report.py`, `tests/adversarial/nightly_runner/__init__.py`
- Test: `tests/adversarial/nightly_runner/test_adv_nightly_findings.py`
- Modify: the testing guide (a new section, "The nightly run", before its `Next:` line), describing `Finding`

**Interfaces:**
- Produces:
  - `SEVERITIES = ("unclassified", "security", "data-loss", "wrong-result", "crash")`. `"security"` means in scope for `SECURITY.md`; any other value except `"unclassified"` means someone checked and found it out of scope.
  - `@dataclass(frozen=True) class Finding: layer: str; surface: str; invariant: str; severity: str = "unclassified"; ops: tuple[Any, ...] = (); seed: str | None = None; artifacts: tuple[tuple[str, str], ...] = (); commit: str = ""; title: str = ""; detail: str = ""`. The constructor accepts a mapping or pairs for `artifacts` and any sequence for `ops`, and refuses an empty `layer`, `surface` or `invariant` and an unknown severity with `ValueError`.
  - `Finding.signature() -> str`: the SHA-256 hex digest of `json.dumps({"v": SIGNATURE_VERSION, "layer": ..., "surface": ..., "invariant": ..., "ops": list(ops)}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`. Severity, seed, artifacts, commit, title and detail are left out, because they change between nights or at triage.
  - `Finding.to_dict()`, `Finding.from_dict(data)` (refuses unknown keys; accepts a `fingerprint` key only when it equals the signature), `to_json()` (one line), `from_json(line)`.
  - `write(path, findings)`, `append(path, finding)`, `read(path) -> list[Finding]` (blank lines ignored; a bad line raises `ValueError` naming its line number).

- [ ] **Step 1: Write the failing tests.**
  - A finding with the same layer, surface, invariant and ops has the same signature whatever its seed, commit, artifacts, severity, title and detail: the property that makes a break seen on two nights one issue.
  - Changing the layer, the surface, the invariant or any op changes the signature.
  - A pinned golden signature for one fixed finding, so a change to the algorithm, which would re-file every open break, fails a test and has to be made on purpose.
  - A JSON round trip gives an equal finding; `to_json()` is one line; `read` of a file written by `write` and `append` gives the findings back in order.
  - Refusals: an unknown severity, an empty invariant, an unknown key, and a `fingerprint` key that does not match.
- [ ] **Step 2: Run them and see them fail** with `ModuleNotFoundError: No module named 'harness.report'`.
- [ ] **Step 3: Write `tests/harness/report.py`**, with one doctest that shows the seed not changing the signature.
- [ ] **Step 4: Run the tests and the module's doctest; they pass.** `mypy tests/harness` with and without `--ignore-missing-imports` is clean.
- [ ] **Step 5: Commit** `tests/harness/report.py`, the two test files and the testing guide.

### Task 2: The night folder, and steps with caps

**Files:**
- Create: `scripts/nightly/__init__.py`, `scripts/nightly/night.py`, `scripts/nightly/steps.py`
- Test: `tests/adversarial/nightly_runner/test_adv_nightly_steps.py`

**Interfaces:**
- Produces, in `night.py`:
  - `NIGHTLY = ("local", "nightly")`, `HEARTBEAT = "heartbeat.json"`, `HISTORY = "history.jsonl"`, `REPORT_JSON`, `REPORT_MD`, `FINDINGS`.
  - `@dataclass(frozen=True) class Layout: checkout: Path` with `root`, `history`, `home` (the persistent HOME the steps run with, so the Hypothesis example database survives from night to night) and `night(date) -> Path`.
  - `main_checkout(path) -> Path`: the checkout that owns `.git`, found with `git rev-parse --git-common-dir`, so a run started from any worktree writes to the main checkout.
  - `write_json(path, data)` (atomic: a temporary file, then `os.replace`), `read_json`, `append_jsonl(path, record)`, `read_jsonl(path) -> tuple[list[dict], list[int]]` (the records, and the numbers of the lines it could not read).
  - `write_heartbeat(path, *, night, started_at, finished_at=None, status="running")`.
  - `step_env(base, *, worktree, home, tmp, bin_dir=None) -> dict[str, str]`: `base` without the variable prefixes `harness.env` drops from every child, with `HOME` and `USERPROFILE` set to `home`, `TMPDIR`, `TMP` and `TEMP` to `tmp`, `PYTHONPATH` to the worktree and `bin_dir` first on `PATH`. It sets no `MEMVARA_` variable, so the suite runs as it does in CI.
  - `now() -> datetime`: the local time, with its offset.
- Produces, in `steps.py`:
  - Status names: `PASSED = "passed"`, `FAILED = "failed"`, `TIMED_OUT = "timed out"`, `NOT_BUILT = "not built yet"`, `NOT_RUN = "not run"`, `ERROR = "error"`.
  - `@dataclass(frozen=True) class Step: name: str; cap: float; run: Callable[[Any, float], Outcome] | None = None; waits_for: str = ""; not_built: str = ""; essential: bool = False`. `run(context, deadline)` gets the `time.monotonic()` instant its cap ends.
  - `@dataclass(frozen=True) class Outcome: status: str; summary: str = ""`.
  - `@dataclass(frozen=True) class StepResult: name, status, seconds, cap, summary`.
  - `run_command(argv, *, cwd, env, log, deadline) -> CommandResult(returncode: int | None, timed_out: bool, seconds: float)`: output appended to `log`, stdin closed. On POSIX the child starts a new session and the whole group is killed at the cap and after the child exits; on Windows it starts a new process group and `taskkill /F /T` stops the tree.
  - `run_steps(steps, context, *, worktree=None, clock=time.monotonic) -> list[StepResult]`: in order. A step with no `run` is `not built yet`, with its `not_built` reason, or, when `waits_for` exists in the worktree, a reason saying the file has landed and the step still needs wiring. After an essential step that did not pass, every later built step is `not run`, naming that step. A step that raises is `error` with the exception's last line; a step that returns after its deadline is `timed out`.

- [ ] **Step 1: Write the failing tests** with fake steps that are small Python child processes started with `harness.env.child_env`:
  - Three steps run in the order given, each after the previous one ended (each child appends its name to one file).
  - A step that would run for a minute, with a cap of one second, is reported `timed out` within a few seconds, its output so far is in its log, and the next step still runs.
  - A step whose child starts a grandchild and then hangs: after the cap, the grandchild is dead too (polled by process id, without a fixed sleep).
  - A step with no code is `not built yet` with its reason; with its `waits_for` file present, the reason says the file has landed.
  - An essential step that fails makes every later built step `not run`, and a not-built step stays `not built yet`.
  - A step that raises is `error`, and the run goes on.
  - `step_env` drops every prefix `harness.env` drops, keeps an unrelated variable, and sets `HOME`, `PYTHONPATH` and `TMPDIR` as described.
  - `read_jsonl` returns the good records of a file with a torn last line and the number of the bad line (Review Focus 2).
- [ ] **Step 2: Run them and see them fail** with `ModuleNotFoundError: No module named 'nightly'`.
- [ ] **Step 3: Write `scripts/nightly/__init__.py`, `night.py` and `steps.py`.**
- [ ] **Step 4: Run the tests; they pass.**
- [ ] **Step 5: Commit** the three modules, the test file and the testing guide's paragraph on steps and caps.

### Task 3: Flakes

**Files:**
- Create: `scripts/nightly/flakes.py`
- Test: `tests/adversarial/nightly_runner/test_adv_nightly_flakes.py`

**Interfaces:**
- Produces:
  - `layer_of(nodeid) -> str`: the first folder under `tests/adversarial/` that is not a tier folder (`model`, `concurrency`, …); `"adversarial"` for a file with no such folder; `"live"` under `tests/live/`; `"unit"` elsewhere under `tests/`; `"doctest"` under `memvara/`; `"other"` otherwise.
  - `outcome_of(returncode) -> str`: `0` is `"passed"`, `1` is `"failed"`, anything else (including a rerun stopped at its cap, passed as `None`) is `"error"`.
  - `verdict(results) -> str` for the results of the reruns of a test that failed once: `"flake"` when every rerun passed (the majority of the three runs passed); `"confirmed"` when every rerun failed; `"intermittent"` when they disagree; `"unconfirmed"` when a rerun could not run the test. Only `"confirmed"` may be filed, because a strict expected failure on a test that sometimes passes would itself be a flake.
  - `flaky(results) -> bool`: the test passed at least once after failing.
  - `@dataclass(frozen=True) class Rerun: nodeid: str; results: tuple[str, ...]` with `verdict` and `flaky`.
  - `rerun_command(python, nodeid, tier="nightly") -> list[str]`: `[python, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tier", tier, nodeid]`.
  - `rerun(nodeid, run, times=2) -> Rerun`, where `run(argv) -> int | None` runs one command.
  - `rates(nights, window=14) -> dict[str, dict[str, float]]`: from the `layers` of the last `window` night records (one per date, the last one wins), per layer the tests run, the flaky tests, the rate, and whether it is above `BUDGET = 0.005`.
  - A CLI: `flakes.py rerun --worktree W --python P [--tier T] NODEID…` prints one JSON line per test; `flakes.py rates [--checkout C] [--window N]` prints the table.

- [ ] **Step 1: Write the failing tests:** every verdict from scripted results; `flaky`; `outcome_of`; `rerun` with a fake runner that records the commands; `rates` over records including two for one date and one with no `layers`; the layer of representative node ids; and one real rerun: a test file in a temporary folder that fails on its first run and passes afterwards, rerun twice through `pytest` in a child process, is a flake.
- [ ] **Step 2: See them fail** (`ImportError` for `nightly.flakes`).
- [ ] **Step 3: Write `flakes.py`.**
- [ ] **Step 4: Tests pass.**
- [ ] **Step 5: Commit** with the testing guide's paragraph on flakes.

### Task 4: Reading the regressions run

**Files:**
- Create: `scripts/nightly/pytest_results.py`, `scripts/nightly/regressions.py`
- Test: `tests/adversarial/nightly_runner/test_adv_nightly_regressions.py`

**Interfaces:**
- Produces:
  - `pytest_results.py`, run as `python pytest_results.py --results FILE -- <pytest arguments>`: removes its own folder from the import path, puts the working directory first (as `python -m pytest` does), and runs pytest with a plugin that writes one JSON line per test: `{"nodeid", "outcome", "when", "message", "longrepr"}`, where `outcome` is one of `passed`, `failed`, `error`, `skipped`, `xfailed`, `xpassed`, `xpass-strict`, and a last line `{"exitstatus": N}`. A collection error is an `error` line for the file.
  - `regressions.read_results(path) -> Results(tests: list[dict], exitstatus: int | None)`.
  - `regressions.layers(results) -> dict[str, dict[str, int]]`: per layer, tests run (everything but skipped) and failed.
  - `regressions.program_of(text) -> tuple[str, ...]`: the operations of the last `drive.replay([...])` program in a failure's text; empty when there is none.
  - `regressions.seed_of(text) -> str | None`: the `@reproduce_failure(...)` call Hypothesis prints, when it does.
  - `@dataclass class Failure: finding: Finding; kind: str; reruns: tuple[str, ...]; verdict: str` where `kind` is `"test"`, `"xpass-strict"` or `"session"`.
  - `regressions.failures(results, *, commit, rerun, limit=20) -> list[Failure]`: one per failed test. `rerun(nodeid) -> tuple[str, ...] | None` reruns one test twice and returns `None` once the caller's rerun budget is spent; a test past the limit or the budget is reported `unconfirmed`. A strict unexpected pass is kind `xpass-strict` and is not rerun. A non-zero exit with no failed test adds one `session` failure. Each finding has layer `layer_of(nodeid)`, surface `"suite"`, invariant the node id, severity `"unclassified"`, ops `program_of`, seed `seed_of`, artifacts `{"log": "regressions/output.log", "results": "regressions/results.jsonl"}`, the commit, a title and the failure text.
  - `regressions.command(python, results_file) -> list[str]`.

- [ ] **Step 1: Write the failing tests:** the wrapper, run on a temporary test file holding a pass, a failure, a setup error, a skip, an expected failure, a strict unexpected pass and a file that fails to import, writes the right outcome for each and the exit status; `program_of` and `seed_of` on text shaped like the state machine's failure; `failures` with a fake rerun for each verdict, the rerun limit, a strict unexpected pass (Review Focus 5), and an exit status of 1 with no failed test (Review Focus 4).
- [ ] **Step 2: See them fail.**
- [ ] **Step 3: Write the two modules.**
- [ ] **Step 4: Tests pass.**
- [ ] **Step 5: Commit** with the testing guide's paragraph on the regressions step.

### Task 5: Filing, dry-run by default

**Files:**
- Create: `scripts/nightly/filing.py`, `tests/adversarial/nightly_runner/gh_recorded.json`
- Test: `tests/adversarial/nightly_runner/test_adv_nightly_filing.py`

**Interfaces:**
- Produces:
  - `REPO = "memvara/memvara"`, `LABEL = "nightly-break"`, `marker(fp) -> "<!-- memvara-nightly-fingerprint: <fp> -->"`, `MARKER` (the regular expression that reads one back), `branch(fp) -> "test/nightly-" + fp[:12]`.
  - `@dataclass(frozen=True) class Command: argv: tuple[str, ...]; stdin: str | None = None; cwd: str | None = None`; `@dataclass(frozen=True) class Completed: returncode: int; stdout: str = ""; stderr: str = ""`; `Runner = Callable[[Command], Completed]`; `subprocess_runner`, the real one.
  - `@dataclass class Filed: what: str; fingerprint: str; dry_run: bool; commands: list[Command]; number: int | None; url: str | None; state: str | None; existing: bool`.
  - `class FilingError(Exception)`.
  - `scrub(text, paths: Mapping[str, str]) -> str`: each given path replaced by its placeholder, and `pytest-of-<name>` by `pytest-of-<user>` (Review Focus 3).
  - `existing_issues(gh, *, repo, label) -> dict[str, dict]` and `existing_advisories(gh, *, repo) -> dict[str, dict]`: fingerprint to `{"number" or "ghsa_id", "state", "url"}`, read from the markers in the bodies.
  - `file_issue(finding, fp, *, night, gh, dry_run=True, triaged=False, repo, label, paths) -> Filed`: refuses a security-class finding always, and an unclassified one unless `triaged`; looks for an existing marker first (not in a dry run); then `gh issue create --repo R --title T --label L --body-file -`, with the body on stdin.
  - `file_advisory(finding, fp, *, night, gh, dry_run=True, repo, paths) -> Filed`: `gh api --method POST repos/R/security-advisories --input -` with a JSON draft advisory whose description carries the marker.
  - `open_pr(finding, fp, *, issue, worktree, night, gh, git, dry_run=True, repo) -> Filed`: refuses a security-class finding and a worktree with uncommitted changes; pushes `HEAD` to `refs/heads/test/nightly-<fp12>` (never anything else) and runs `gh pr create --repo R --draft --base main --head B --title T --body-file -`.
  - A CLI for the scheduled session: `filing.py {issue,advisory,pr} --night DATE --fingerprint FP [--worktree W] [--checkout C] [--file]`. It reads the night's `report.json`, refuses a fingerprint that is not a confirmed failure of that night, prints what it sent or would send, and with `--file` appends a `{"kind": "filed", ...}` record to the history.

- [ ] **Step 1: Write the failing tests** against a fake runner that checks each command and answers with the recorded output: in a dry run nothing is called and the planned commands carry the label, the marker and the draft flag; with the fake runner, a new issue is created and its number read from the URL, an existing marker means nothing is created, an advisory is created and its id read, a pull request pushes to the nightly branch only and is a draft; refusals for a security-class issue, an unclassified untriaged issue, a security-class pull request and a dirty worktree; `scrub`; and the CLI in a dry run on a small night folder.
- [ ] **Step 2: See them fail.**
- [ ] **Step 3: Write `filing.py`** and the recorded output.
- [ ] **Step 4: Tests pass.**
- [ ] **Step 5: Commit** with the testing guide's paragraph on filing.

### Task 6: The watchdog

**Files:**
- Create: `scripts/nightly/watchdog.py`, `scripts/nightly/com.memvara.nightly-watchdog.plist.template`
- Test: `tests/adversarial/nightly_runner/test_adv_nightly_watchdog.py`

**Interfaces:**
- Produces:
  - `night_to_check(now, start, deadline) -> date`: the latest night whose deadline has passed at `now`, where a night is named by the date its run starts.
  - `check(checkout, *, now, start, deadline, notify) -> Verdict(night, state, report, notified)`, with `state` one of `"finished"`, `"did-not-run"` (no heartbeat) and `"did-not-finish"` (a heartbeat with no finish, Review Focus 1). It writes `DID-NOT-RUN.md` and `.json`, or `DID-NOT-FINISH.md` and `.json`, in the night's folder, and notifies once: a report already there means no second notification.
  - `osascript_argv(title, message) -> list[str]`, which passes the text as arguments to an `on run argv` script so no text is ever parsed as AppleScript, and `notify(title, message, *, run=subprocess.run)`.
  - A CLI: `watchdog.py --checkout C --start HH:MM --deadline HH:MM [--now ISO] [--no-notify]`.
  - The plist template, with `__PYTHON__`, `__CHECKOUT__`, `__START__`, `__DEADLINE__`, `__HOUR__` and `__MINUTE__`.

- [ ] **Step 1: Write the failing tests** with a fixed clock: a night with no heartbeat, one started but not finished, one finished; a run of the watchdog before the deadline checks the night before; a second check sends no second notification; a start before midnight and a deadline after it; `osascript_argv`; the CLI; and the template, filled in with sample values, parses as a property list whose program arguments the watchdog's own parser accepts and whose hour and minute match the deadline. The watchdog's folder and heartbeat names equal `night.py`'s.
- [ ] **Step 2: See them fail.**
- [ ] **Step 3: Write `watchdog.py` and the template.**
- [ ] **Step 4: Tests pass.**
- [ ] **Step 5: Commit** with the testing guide's paragraph on the watchdog.

### Task 7: The nightly run

**Files:**
- Create: `scripts/nightly/run.py`, `scripts/nightly/render.py`
- Test: `tests/adversarial/nightly_runner/test_adv_nightly_run.py`

**Interfaces:**
- Consumes everything above.
- Produces:
  - `STEPS`, the design's ten steps with their caps: preflight (20 minutes, essential), regressions (75 minutes: the test run may use up to 60 of them, and the reruns of failed tests get the rest, at least 15), agents 20, red team 25, hosted 15, production smoke 5, performance 15, soak 20, mutation 15 and replay 10 minutes. The regressions cap is checked in Task 9 against a measured run and changed there if a normal night would use more than half of it. The caps of unbuilt steps are placeholders for the workstreams that build them. Each unbuilt step names the file it waits for and why it is not built.
  - Preflight: write the heartbeat (before anything that can fail), take the canary snapshot (`~/.memvara/credentials.json` and `~/.memvara/db.key` by default, more with `--canary`), remove earlier nights' worktrees when they are clean, fetch `origin`, add a detached worktree of `origin/main` at `local/nightly/<date>/worktree`, build `local/venv` in it and install `.[dev,cloud,ingest,encrypt]`, check the `gh` login only with `--file`, and say that model resolution is not built yet. `--worktree PATH` and `--python PATH` use an existing checkout and interpreter instead, and the report says so.
  - `triage(failures, nights, filed, remote) -> list[dict]`: each confirmed failure's fingerprint and its novelty: `"new"` (never seen before), `"known"`, or `"recurred"` (its issue is closed).
  - `plan(...)`: for each confirmed fingerprint with no filing yet, what filing needs and the exact commands. In file mode the run files only classified findings: an advisory for a security-class one, an issue for any other.
  - `notifications(report, previous_night) -> list[tuple[str, str]]`: new breaks, an isolation breach, a dependency down two nights in a row.
  - `main(argv=None, *, steps=STEPS, notify=watchdog.notify, gh=filing.subprocess_runner, clock=night.now) -> int`. It always writes `report.json`, `report.md`, `findings.jsonl`, a history record and the heartbeat's finish, even when a step raises.
  - `render.markdown(report) -> str`.

- [ ] **Step 1: Write the failing tests:** an end-to-end run in a temporary git repository with a bare `origin`, fake steps and a fake regressions result (the worktree is created at `origin/main`'s commit, and every file is written); every unbuilt step is reported with its reason; two nights with the same confirmed failure make one new break and one known break, one notification, and one filing plan per night while it stays unfiled; a filed record stops the plan; a classified security finding from a step's `findings.jsonl` plans an advisory and no issue; a canary file changed during the run is an isolation breach with a notification; a dependency down on two nights running is notified once; a step that raises still leaves a full report; the markdown names every step and every new break.
- [ ] **Step 2: See them fail.**
- [ ] **Step 3: Write `run.py` and `render.py`.**
- [ ] **Step 4: Tests pass.**
- [ ] **Step 5: Commit** with the testing guide's paragraph on running it by hand and the report.

### Task 8: The session prompt

**Files:**
- Create: `scripts/nightly/task.md`
- Test: add to `tests/adversarial/nightly_runner/test_adv_nightly_run.py`: every `scripts/nightly/<module>.py …` command in `task.md` parses with that module's own argument parser.

- [ ] **Step 1: Write the failing test.**
- [ ] **Step 2: See it fail** (`task.md` missing).
- [ ] **Step 3: Write `task.md`:** the settings the maintainer fills in (the checkout, and the filing mode, which stays `dry-run` until the supervised run), the run command, how to read the report, how to classify each new break against `SECURITY.md`'s "In scope" section, what to do for each class, the review every pull request needs before it merges, and the rules: nothing merges, nothing is pushed to `main`, no attribution, no other notification.
- [ ] **Step 4: Test passes.**
- [ ] **Step 5: Commit** with the testing guide's paragraph on the scheduled session and the operator's install steps.

### Task 9: Verification

- [ ] Run each new test file 20 times in a row and record the result.
- [ ] Run the full gate as two commands with a private coverage file, and both mypy runs on `tests/harness`, and `mypy -p memvara`. Quote the result lines.
- [ ] Run `run.py` for real once, against this worktree (`--worktree` and `--python`, with `--no-notify`), and read the report it writes.
