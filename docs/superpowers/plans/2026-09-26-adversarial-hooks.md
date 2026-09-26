# Hook conformance for all five hosts (A2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drive every plugin hook on every host the way that host's client does, through `plugin/hooks/run.py`, and check what each host depends on: the reply envelope it reads, exit code 0 for any payload, four outcomes a person can tell apart, deduplication and the recall daemon, the generated registration, the time limits, and an approve list that matches the server's read-only tools.

**Architecture:**
- `tests/harness/hooks.py` gains what the design assigns to this workstream: stub agent CLIs first on `PATH` (so `capture` can run), a `PATH` that never holds a real agent CLI, the log lines each run added, a wait for the capture that some hosts hand to a detached child, a TOML client config for Codex, a `daemon` option with a way to stop the daemon, and `patches`, which shrink one of a hook's time limits inside the child process so no fast test waits out a real limit.
- `tests/adversarial/hooks/` holds the tests, with `support.py` for what several modules share and `nightly/` for the full host matrices and the tests that take real seconds.
- A module that needs many hook runs makes them once, side by side in threads, in a module-scoped fixture, and its parametrized tests read the results. Each run has a home directory of its own, so no run sees another's logs or state.
- A test that shows a bug is kept out of the commits and reported to the maintainer with a reproduction, as the brief for this workstream says. What is committed passes.

**Tech Stack:** Python 3.10–3.13, pytest, `subprocess`, `concurrent.futures`, `socket`, `harness.hooks.HookRunner`, `harness.fakes.cli.FakeClis`, `harness.fakes.hosted_mcp.FakeHostedMcp`, `harness.stdio.McpProcess`, `harness.stores`.

**Spec:** `docs/superpowers/specs/2026-09-25-adversarial-test-suite-design.md`, the "A2 hooks" row of Phase 2, the predictions for hooks under "First targets", and the "One more rule" section.

## Global Constraints

- Everything runs offline, with no API key and no real agent CLI. Stores use `HashingEmbedder(dim=512)` and `NullLLM`.
- Every child process gets its environment from `harness.env.child_env`, through `HookRunner`.
- No library or plugin code changes. This workstream only observes the hooks.
- Tests live under `tests/adversarial/hooks/`, and each test file's name starts with `test_adv_`. Every folder has an `__init__.py`.
- The fast tier of this workstream takes about 20 seconds on a laptop.
- Every skip has a rule in `tests/harness/skips.py`.
- A test that fails because memvara misbehaves is left out of the commits and reported, with its classification against `SECURITY.md`. Predictions that the design tracks privately are handled outside this plan.
- Write plainly: every sentence must be understood on its first reading. No AI attribution anywhere.

## What each host reads

These facts come from the host records in `plugin/hooks/hosts/`, which record what each client was measured to do. The tests restate them by hand, so a record that drifts from its own measurements fails.

| Host | Envelope | Status line | Context key | Approval keys | Recall event | Capture |
|---|---|---|---|---|---|---|
| claude | nested in `hookSpecificOutput` | `systemMessage` | `additionalContext` | `permissionDecision`, `permissionDecisionReason` | `UserPromptSubmit` | async, in the hook process |
| codex | nested | none | `additionalContext` | `permissionDecision`, `permissionDecisionReason` | `UserPromptSubmit` | handed to a detached child |
| copilot | flat | none | `additionalContext` | `permissionDecision`, `permissionDecisionReason` | `UserPromptSubmit` | handed to a detached child |
| cursor | flat | none | `additional_context` | `permission`, `reason` | none | handed to a detached child |
| opencode | flat, read by `js/shim.mjs` | none | `additionalContext` | `status`, `reason` | `chat.message` | not awaited by the shim |

The limits, in seconds: session start 20, recall 10 with optional work stopped at 7.5, approve 5, capture 180 on claude and 120 elsewhere.

## Review Focus

1. **A payload the hook cannot read.** Empty, not JSON, not an object, or not UTF-8: the hook must answer exactly as it answers `{}`, and its body must not crash.
2. **A detached capture outliving its test.** On Codex, Copilot and Cursor the capture keeps running after the hook returns. The harness waits for it, or kills its whole process group, so no agent CLI stub is left sleeping.
3. **A daemon outliving its test.** A recall daemon idles for 30 minutes. Every runner that allows one is closed at teardown, which kills it by the pid the kernel reports for its socket.
4. **A real agent CLI on the test machine's `PATH`.** The harness removes every directory that holds one, so a capture can only ever start a stub.
5. **A socket path too long for macOS.** The daemon's socket lives under the hooks' home, and macOS refuses a unix socket path over 104 bytes, so homes here come from a short base directory.

---

### Task 1: Stub CLIs, a safe PATH and per-run log lines in HookRunner

**Files:**
- Modify: `tests/harness/hooks.py`
- Modify: `tests/adversarial/test_adv_hooks.py`
- Modify: `docs/claude/testing.md` (the "Hooks" section, and the `FakeClis` bullet that says capture is refused)

**Interfaces:**
- Produces: `HookRunner(host, *, home, cwd, server_env=None, env=None, stubs=None)`; `HookRunner.run(hook, *, stdin: str | bytes | None = None, timeout=None, wait_detached=True, **fields) -> HookResult`; `HookResult.logs: Mapping[str, tuple[str, ...]]`, `HookResult.log(name) -> tuple[str, ...]`, `HookResult.detached_pid: int | None`; `HookRunner.close()`; module functions `host_ids()`, `agent_clis()`, `path_without_agent_clis(path)`; the `Stubs` protocol (anything with `path(rest=None) -> str`, such as `FakeClis`).

- [ ] **Step 1: Write the failing self-tests** in `tests/adversarial/test_adv_hooks.py`: capture is refused without stubs and runs with them; a directory that holds a real agent CLI is left off `PATH`; a run reports the log lines it added, without timestamps; a detached capture is waited for.

```python
def test_capture_is_refused_without_stub_agent_clis(hook_runner: Make) -> None:
    """capture starts an agent CLI to mine the turn, which would reach the network and
    spend money. HookRunner refuses it unless the test gives it stub CLIs."""
    with pytest.raises(NotImplementedError, match="stub"):
        hook_runner("claude").run("capture")


def test_no_directory_that_holds_a_real_agent_cli_is_on_the_hooks_path(
        hook_runner: Make, tmp_path: pathlib.Path) -> None:
    real = tmp_path / "real-bin"
    real.mkdir()
    (real / "cursor-agent").write_text("#!/bin/sh\nexit 0\n")
    runner = hook_runner("claude", env={"PATH": os.pathsep.join([str(real), "/usr/bin"])})
    assert str(real) not in runner._env["PATH"].split(os.pathsep)
    assert "/usr/bin" in runner._env["PATH"].split(os.pathsep)


def test_every_extractor_a_host_names_counts_as_an_agent_cli() -> None:
    assert {"claude", "codex", "cursor-agent", "copilot", "opencode"} <= agent_clis()


def test_a_run_reports_the_log_lines_it_added_without_their_timestamps(
        hook_runner: Make) -> None:
    runner = hook_runner("cursor")
    result = runner.run("recall", stdin="{}", timeout=10)
    assert result.log("hooks") == ("skipped=cursor has no event for recall",)
    again = runner.run("recall", stdin="{}", timeout=10)
    assert again.log("hooks") == ("skipped=cursor has no event for recall",)
```

The capture tests need the fakes, which are POSIX only:

```python
def _clis(tmp_path: pathlib.Path) -> FakeClis:
    if sys.platform == "win32":
        pytest.skip("the fake agent CLIs are POSIX shell scripts")
    return FakeClis(tmp_path / "clis")


def test_capture_runs_against_the_stub_clis(hook_runner: Make, tmp_path: pathlib.Path) -> None:
    clis = _clis(tmp_path)
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(json.dumps({"type": "user", "message": {"content": "ok"}}) + "\n")
    result = hook_runner("claude", stubs=clis).run("capture", transcript_path=str(transcript))
    assert result.exit_code == 0
    assert result.log("capture") == ("turn=6c skipped=continuation",)


def test_a_detached_capture_is_waited_for(hook_runner: Make, tmp_path: pathlib.Path) -> None:
    """Codex hands capture to a child in a new session and returns at once. The runner
    waits for the child, so the logs hold what the capture did."""
    clis = _clis(tmp_path)
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(json.dumps({"type": "response_item", "payload": {
        "type": "message", "role": "user",
        "content": [{"type": "input_text", "text": "ok"}]}}) + "\n")
    result = hook_runner("codex", stubs=clis).run("capture", transcript_path=str(transcript))
    assert result.detached_pid is not None
    assert result.log("capture") == ("turn=6c skipped=continuation",)
```

- [ ] **Step 2: Run them and see them fail**

Run: `PYTHONPATH=$PWD TMPDIR=<own> <python> -m pytest -q -p no:cacheprovider tests/adversarial/test_adv_hooks.py`
Expected: FAIL. `HookRunner.__init__` has no `stubs` argument, `HookResult` has no `log`, and `agent_clis` does not exist.

- [ ] **Step 3: Implement** in `tests/harness/hooks.py`: the `Stubs` protocol; `host_ids()`, `agent_clis()` (every host record's `extractor.argv[0]` and `core.host.CLAUDE_CLI.argv[0]`), and `path_without_agent_clis(path)`, which drops every `PATH` entry that holds one of those names (with each `PATHEXT` suffix on Windows); in `HookRunner.__init__`, `PATH` becomes `stubs.path(path_without_agent_clis(...))` or the filtered value; `run` refuses capture without stubs, sends `stdin` as bytes, decodes stdout strictly as UTF-8 (raising `HookOutputError` otherwise), records the size of every `~/.memvara/.hooks/*.log` before the run and returns the lines each gained after it, with the leading timestamp removed and a truncated log read whole; for a host whose record sets `detach_capture`, it reads the `detached hook=capture host=<id> pid=<n>` line from `hooks.log`, waits for that pid to end within the same limit (a zombie on Linux counts as ended), and reads the logs again; with `wait_detached=False` it records the pid, and `close()` kills its process group.

- [ ] **Step 4: Run the self-tests and see them pass**, then run `tests/adversarial/sessions` too, because the scenario runner builds `HookRunner`s.

- [ ] **Step 5: Update the testing guide's "Hooks" section**, which says capture is refused and the runner returns no log lines, and the `FakeClis` bullet that says the same. Commit:

```bash
git add tests/harness/hooks.py tests/adversarial/test_adv_hooks.py docs/claude/testing.md
git commit -m "Let HookRunner run capture against stub agent CLIs and report each run's log lines"
```

### Task 2: A TOML client config for Codex

**Files:**
- Modify: `tests/harness/hooks.py`, `tests/adversarial/test_adv_hooks.py`, `docs/claude/testing.md`

**Interfaces:**
- Produces: `toml_string(text) -> str` and `toml_server_block(name, command, args, env) -> str` in `tests/harness/hooks.py`, each with a doctest; `HookRunner.write_client_config` writes `[mcp_servers.memvara]` with an `env` sub-table for a host whose `config_format` is `toml`.

- [ ] **Step 1: Replace the refusal test with a test of the TOML writer** (tomllib arrives in 3.11; the ledger already has that skip rule):

```python
def test_a_codex_client_config_is_written_as_toml(hook_runner: Make) -> None:
    if sys.version_info < (3, 11):
        pytest.skip("tomllib arrives in 3.11")
    import tomllib  # noqa: PLC0415 - 3.11 and later

    runner = hook_runner("codex", server_env={"MEMVARA_DB": 'C:\\stores\\a "b".db',
                                              "MEMVARA_USER": "tester"})
    config = tomllib.loads((runner.home / ".codex" / "config.toml").read_text())
    block = config["mcp_servers"]["memvara"]
    assert block["args"] == ["-m", "memvara.server"]
    assert block["env"] == {"MEMVARA_DB": 'C:\\stores\\a "b".db', "MEMVARA_USER": "tester"}
```

- [ ] **Step 2: Run it and see it fail** with `NotImplementedError: HookRunner writes JSON client configs only`.
- [ ] **Step 3: Implement `toml_string`, `toml_server_block` and the TOML branch.** A basic string escapes `"`, `\` and control characters, and refuses a lone surrogate, which TOML cannot hold.
- [ ] **Step 4: Run the self-tests and the module's doctests** (`pytest tests/harness/hooks.py tests/adversarial/test_adv_hooks.py`).
- [ ] **Step 5: Update the "Hooks" section, which says a Codex run with a store is refused, and commit:** `git commit -m "Write Codex's client config as TOML in HookRunner, the way Codex keeps it"`.

### Task 3: The daemon option

**Files:**
- Modify: `tests/harness/hooks.py`, `tests/adversarial/test_adv_hooks.py`, `tests/harness/skips.py`, `docs/claude/testing.md`

**Interfaces:**
- Produces: `HookRunner(..., daemon=False)`; `HookRunner.daemon_sockets() -> list[pathlib.Path]`; `HookRunner.wait_for_daemon(timeout=10.0) -> tuple[pathlib.Path, int]`; `socket_peer_pid(path) -> int | None`; `close()` also kills every daemon listening under the runner's home.
- A skip rule: `^the recall daemon listens on a unix socket, which Windows lacks$`, on `win32` only.

- [ ] **Step 1: Write the failing self-tests**: `socket_peer_pid` names the test process for a socket it listens on, and None for a path nothing listens on; `daemon=True` removes `MEMVARA_DAEMON` from the child's environment; `daemon=True` is refused where `socket.AF_UNIX` does not exist (monkeypatched away).

```python
def test_the_peer_pid_of_a_socket_is_the_process_listening_on_it() -> None:
    if sys.platform == "win32":
        pytest.skip("the recall daemon listens on a unix socket, which Windows lacks")
    # Not tmp_path: macOS refuses a unix socket path over 104 bytes.
    directory = pathlib.Path(tempfile.mkdtemp(prefix="mv-sock-", dir="/tmp"))
    path = directory / "s.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(1)
    try:
        assert socket_peer_pid(path) == os.getpid()
    finally:
        server.close()
        shutil.rmtree(directory, ignore_errors=True)
    assert socket_peer_pid(path) is None
```

- [ ] **Step 2: Run them and see them fail** (`socket_peer_pid` does not exist).
- [ ] **Step 3: Implement.** `socket_peer_pid` connects and reads `LOCAL_PEERPID` (level 0, option 2) on macOS or `SO_PEERCRED` on Linux. `close()` kills each daemon's process group and removes its socket file.
- [ ] **Step 4: Run the self-tests.**
- [ ] **Step 5: Add the skip rule, document the option, and commit:** `git commit -m "Add a daemon option to HookRunner, and stop the daemon when the runner closes"`.

### Task 4: Patches that shrink a time limit inside the hook process

**Files:**
- Modify: `tests/harness/hooks.py`, `tests/adversarial/test_adv_hooks.py`, `docs/claude/testing.md`

**Interfaces:**
- Produces: `HookRunner(..., patches: Mapping[str, float] | None = None)`. With patches, the child runs a small launcher (`python -c`) that binds the host, imports each named module, refuses an attribute that does not exist (exit status 97, which `run` turns into `ValueError`), sets the rest, and calls `run.main`. Patches are refused for a capture that a host detaches, because run.py starts that child afresh.

- [ ] **Step 1: Write the failing self-tests**: a patch reaches the hook (with `recall.OVERALL_BUDGET_SEC` at 0 the recall log says `skipped=standing refresh, budget exhausted`); a patch naming nothing is refused with `ValueError`.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement the launcher.**
- [ ] **Step 4: Run the self-tests.**
- [ ] **Step 5: Document and commit:** `git commit -m "Let HookRunner shrink a hook's time limit inside the hook process"`.

### Task 5: Shared support, and the envelope every host reads

**Files:**
- Create: `tests/adversarial/hooks/__init__.py`, `tests/adversarial/hooks/support.py`, `tests/adversarial/hooks/conftest.py`, `tests/adversarial/hooks/test_adv_hook_envelopes.py`
- Modify: `docs/claude/testing.md` (a new "Hook conformance" section before its last line)

**Interfaces:**
- Produces in `support.py`: `HOSTS`, `USER`, `MEMORY`, `PROMPT`, `UNRELATED`, `EVENTS`, `SHAPES`, `DETACHES`, `USER_TURN`, `ASSISTANT_TURN`, `FACTS_REPLY`, `PROPOSALS_REPLY`, `short_dir(prefix)`, `runner_factory(work)` (a context manager yielding `make(host, *, home=None, **options) -> HookRunner`), `make_store(path, memory=True)`, `host_payload(host, hook, *, session, cwd, **fields)`, `write_transcript(host, path, turns)`, `script_clis(clis, host)`, `run_all(jobs)`, `result_of(outcome)`, `context_of(host, reply)`, `status_of(host, reply)`, `decision_of(host, reply)`, `observed(result)`, `crashes(result)`, `HangingClis(directory)`.
- Produces in `conftest.py`: the fixtures `hooks` (a `runner_factory` for the test), `store`, `store_env`, `clis`.

- [ ] **Step 1: Write the envelope tests first.** A module-scoped fixture runs, side by side, every hook on every host against a store that holds `MEMORY` (session start, recall with `PROMPT`, approve for a read tool and for a write tool, and capture against the fakes with a transcript in that host's format). The tests compare each reply's shape with the table above, check that the memory sits in the context key, that the status line starts with `⋈ Memvara · `, that capture prints nothing and stores the fact, that a capture is detached exactly on codex, copilot and cursor, and that cursor's recall is skipped with the line `skipped=cursor has no event for recall`.

```python
@pytest.mark.parametrize("host", support.HOSTS)
def test_session_start_answers_in_the_envelope_the_host_reads(runs, host: str) -> None:
    result = support.result_of(runs["session_start", host])
    assert (result.exit_code, result.stderr) == (0, "")
    assert shape(result.reply) == expected(host, "session_start")
    assert support.MEMORY in support.context_of(host, result.reply)
```

- [ ] **Step 2: Run and see it fail** because `support` has none of those names yet.
- [ ] **Step 3: Write `support.py` and `conftest.py`.**
- [ ] **Step 4: Run and see it pass.** Then prove the assertions bite: change one row of the expected table (copilot to nested) and one context key, see the matching tests fail, and change them back.
- [ ] **Step 5: Add the "Hook conformance" section to the testing guide and commit:** `git commit -m "Check that every hook on every host answers in the envelope that host reads"`.

### Task 6: Where each host's hooks find the store

**Files:** Create `tests/adversarial/hooks/test_adv_hook_client_config.py`; modify the testing guide.

- [ ] **Step 1: Write the tests**: on every host whose client config is JSON, session start finds the store the host's first client config names; a `MEMVARA_DB` in the hook's own environment wins over the client config; on Codex a store named in the hook's environment is found. (Codex's own TOML config is the finding reported in Task 15.)
- [ ] **Steps 2–4: Run, see the bite by pointing the config at an empty store, and pass.**
- [ ] **Step 5: Commit:** `git commit -m "Check where each host's hooks look for the store"`.

### Task 7: Hostile payloads

**Files:** Create `tests/adversarial/hooks/test_adv_hook_hostile.py` and `tests/adversarial/hooks/nightly/__init__.py`, `tests/adversarial/hooks/nightly/test_adv_hook_matrix_nightly.py`; add to `support.py`: `HOSTILE`, `UNREADABLE`, `hostile_matrix(base, hosts)`; modify the testing guide.

Payloads: empty stdin; text that is not JSON; a JSON list; invalid UTF-8 read strictly and read with `surrogateescape` (the two ways a child's stdin can be set up); a prompt that is a lone surrogate; no fields; fields of the wrong type; unknown fields; an 8 MB payload; deep nesting. The fast tier runs them on claude and cursor, and the nightly tier on all five hosts, with a JSON string, `null` and a number added.

- [ ] **Step 1: Write the tests**: every payload leaves the turn alone (exit 0, stderr empty, stdout empty or one JSON object, within the host's limit); a payload that is not a JSON object is answered exactly as `{}` is, and the hook's body does not crash; no payload starts an extraction.
- [ ] **Steps 2–4: Run, check the bite (drop the `{}` baseline comparison to a wrong host), pass.**
- [ ] **Step 5: Commit:** `git commit -m "Send the hooks hostile payloads and check that none breaks the turn"`.

### Task 8: The four outcomes

**Files:** Create `tests/adversarial/hooks/test_adv_hook_outcomes.py`; add the outcome matrix to `support.py` and its nightly half to `nightly/test_adv_hook_matrix_nightly.py`; modify the testing guide.

Outcomes: not configured; a local store that cannot open (a file that is not a SQLite database); a hosted store that cannot be reached (`FakeHostedMcp` refusing `initialize` with 503); nothing matches (for session start, an empty store); memories injected. Fast tier: claude and copilot. Nightly: all hosts.

- [ ] **Step 1: Write the tests**: injected memories are told apart from every other outcome; a store that could not be asked is told apart from one that answered; recall that could not ask says so (`recall failed` on claude, and `failed reason=unknown` in `recall.log` on every host); on a host with a status line, each outcome has its own words. Those words, on claude:

| Hook | Outcome | `systemMessage` |
|---|---|---|
| session_start | not configured | `⋈ Memvara · not configured` |
| session_start | nothing matches (an empty store) | `⋈ Memvara · session opened` |
| session_start | memories injected | `⋈ Memvara · session opened with 1 memory` |
| recall | not configured | `⋈ Memvara · not configured` |
| recall | store unreachable | `⋈ Memvara · recall failed` |
| recall | nothing matches | `⋈ Memvara · no matching memories` |
| recall | memories injected | `⋈ Memvara · 1 memory recalled` |

A test compares two outcomes by what a person or a log reader can see of each run: the reply, and the lines it added to each log, without timestamps.
- [ ] **Steps 2–5: Run, check the bite, pass, commit:** `git commit -m "Check that the hooks tell their outcomes apart"`.

### Task 9: Deduplication

**Files:** Create `tests/adversarial/hooks/test_adv_hook_dedup.py`; modify the testing guide.

- [ ] **Step 1: Write the tests**: recall injects a memory once per session and again in a new one; capture mines a turn once however often `Stop` fires over it, and mines the next turn when the transcript grows.
- [ ] **Steps 2–5: Run, check the bite, pass, commit:** `git commit -m "Check that recall and capture do not repeat themselves within a session"`.

### Task 10: The recall daemon

**Files:** Create `tests/adversarial/hooks/test_adv_hook_daemon.py` and `tests/adversarial/hooks/nightly/test_adv_hook_daemon_nightly.py`; modify the testing guide.

- [ ] **Step 1: Write the tests** (POSIX only): the first recall starts one daemon, whose socket is 0600 inside a 0700 directory even when that directory already existed with looser modes; a second recall is answered by it, which shows as a hook process that never imported `memvara` (`PYTHONPROFILEIMPORTTIME=1`), the same peer pid, and one socket file; the control, a recall without a daemon, does import `memvara`. Nightly: a killed daemon's socket file is reclaimed by the next daemon.
- [ ] **Steps 2–5: Run, check the bite, pass, commit:** `git commit -m "Check the recall daemon's socket modes and that later hooks reuse it"`.

### Task 11: The generated registration

**Files:** Create `tests/adversarial/hooks/test_adv_hook_registration.py`; modify the testing guide.

This repository commits no `hooks.json`: `plugin/README.md` says the path is ignored here because seven plugin repositories generate their own. So the tests check what can be checked here: generating twice gives the same bytes as `registration()` does in process; the registration agrees with each shell host's record (events, command, limits, `async` only for claude's capture, `additionalContextLimit` 32000 only on Codex's context hooks, the approve matcher); a host that has no shell hooks (opencode) is refused and an existing file is left alone.

- [ ] **Steps 1–5: Write, run and see fail, pass, check the bite, commit:** `git commit -m "Check that generating a host's hook registration is repeatable and matches its record"`.

### Task 12: Time limits

**Files:** Create `tests/adversarial/hooks/test_adv_hook_timeouts.py` and `tests/adversarial/hooks/nightly/test_adv_hook_timeouts_nightly.py`; modify the testing guide.

- [ ] **Step 1: Write the tests**:
  - every host's limits are the contract's (the table under "What each host reads");
  - recall's optional-work budget, `recall.OVERALL_BUDGET_SEC`, is 7.5 seconds, above `lib.fast.REWRITE_WAIT_SEC` (5) and below recall's 10-second limit;
  - with that budget patched to 0, recall still answers, and `recall.log` says `skipped=standing refresh, budget exhausted`, and, for a prompt that matches nothing, `skipped=episode widen, budget exhausted`;
  - each host's capture limit covers every extraction its capture can run: on claude, `lib.agentic.TIMEOUT_SEC` (60) plus the 10 seconds `lib.agentic._run` waits for a killed run, plus `lib.extract.TIMEOUT_SEC` (90), is at most 180; elsewhere `lib.extract.TIMEOUT_SEC` is at most 120;
  - with `lib.agentic.TIMEOUT_SEC` and `lib.extract.TIMEOUT_SEC` patched to 1.0 and a `claude` that never answers (`support.HangingClis`), capture on claude exits 0 within 8 seconds, `capture.log` holds `agentic capture fell back to single-call extraction: no reply within 1.0s` and `extraction did not run via claude: no reply within 1.0s`, and the next recall's status line ends `capture failing: no reply within 1.0s`;
  - on Codex, a capture whose extractor never answers returns within 3 seconds, and its detached child is still running when it does (the runner kills it afterwards).

  Nightly: the real 7.5-second cut, against a hosted endpoint whose `memory_standing` and `memory_since` calls each take 3.9 seconds, with a prompt that matches nothing (`recall.log` says `skipped=episode widen, budget exhausted` and the hook finishes inside its 10 seconds); and the detached-capture check on copilot and cursor.
- [ ] **Steps 2–5: Run, check the bite, pass, commit:** `git commit -m "Check the hooks' time limits without waiting them out"`.

### Task 13: The approve list

**Files:** Create `tests/adversarial/hooks/test_adv_hook_approve.py`; modify the testing guide.

- [ ] **Step 1: Write the tests**: the approve hook's list holds no tool the real server (started over stdio, with every feature on) leaves unmarked as read-only, and misses no read-only tool other than the two #267 (B3) is about, while B3 is registered; every host approves exactly the read-only tools, each named the way that host names it.
- [ ] **Steps 2–5: Run, check the bite, pass, commit:** `git commit -m "Check that every host approves exactly the server's read-only tools"`.

### Task 14: Capture's log lines

**Files:** Create `tests/adversarial/hooks/test_adv_hook_capture_log.py`; modify the testing guide.

- [ ] **Step 1: Write the tests**: capture writes a line to `capture.log` when it decides something. On claude, with the fakes:

| The turn | The line capture writes |
|---|---|
| the user typed only `ok` | `turn=6c skipped=continuation` |
| the transcript holds no typed prompt | `no turn to mine` |
| no store is configured and nobody is signed in | `turn=<n>c stored=0 failed=no store or login` |
| the extractor proposes a fact the turn states | `turn=<n>c agentic searches=0 proposals=1 refused=0 stored=1 ...` |
- [ ] **Steps 2–5: Run, check the bite, pass, commit:** `git commit -m "Check that capture logs what it decided"`.

### Task 15: The findings

Every test that shows a bug is written as a normal test, run to see it fail for the reason predicted, and kept out of the commits, in a findings file outside the repository. Each finding is classified against `SECURITY.md` and reported to the maintainer with an offline reproduction.

The design's public predictions for hooks are settled here, each either confirmed and reported or disproved and kept as a passing guard in the task named:

| Prediction | Test |
|---|---|
| On hosts with no status line, "not configured" and "no matching memories" cannot be told apart. | recall on codex, copilot and opencode, not configured against nothing matches (Task 8's outcomes) |
| A configured local store that cannot open is reported as "not configured". | every host, both reading hooks (Task 8's outcomes) |
| The `_` approve separator can never recover a tool name. | cursor and opencode, each read-only tool named `memvara_<tool>` (Task 13's naming) |
| Codex's TOML client config is parsed as JSON. | session start on codex with only `~/.codex/config.toml` naming the store (Task 6) |
| Capture returns without a log line on three paths. | every early return in `capture.main` (Task 14) |

### Task 16: Verification

- [ ] Run each new test file 20 times in a row.
- [ ] Run the full gate once, with a private coverage file, and both mypy runs.
- [ ] Quote the result lines in the final report.
