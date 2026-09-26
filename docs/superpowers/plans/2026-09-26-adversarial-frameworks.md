# Framework adapters against the real packages (A8, nightly half) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run memvara's LangChain, LlamaIndex, CrewAI and LangGraph adapters and its mem0 shim against the real framework packages every night, at the oldest release memvara claims to support and at the newest release, and fail when an adapter breaks against either.

**Architecture:**
- **One virtual environment per framework and pin**, built with this interpreter's `venv` and pip under `~/.cache/memvara-adversarial/frameworks/`. Each holds memvara, installed from a wheel built from this checkout, and the framework pinned exactly. An environment is reused while its key is unchanged. The key covers the framework's pinned version, the requirement memvara declares on it, memvara's own dependencies and the interpreter. Environments the current pins no longer need are removed. `tests/adversarial/frameworks/environments.py` does all of this, and everything in it that runs pip goes through one class, `Pip`, which the fast tests replace with a fake.
- **A probe runs each framework's checks inside its environment.** `tests/adversarial/frameworks/probe.py` is started with the environment's own interpreter. It installs an audit hook that blocks and records every network access, then runs each `check_*` function of the framework's checks file and writes one JSON line per outcome. The suite reads the report back. The probe imports only the standard library at module level, because the environment has no pytest and the suite has no framework.
- **The nightly tests** in `tests/adversarial/frameworks/nightly/` make every check a test of its own, and add three tests per environment: it holds the release it pins, it runs the memvara in this checkout, and nothing in it reached the network.

**Tech Stack:** Python 3.10–3.13 for the fast tier, the interpreter running the suite for the environments (3.13 on the maintainer's Mac), pytest, the standard library's `venv`, `subprocess`, `sys.addaudithook` and `faulthandler`, and `packaging` (a pytest dependency) for requirement strings.

**Spec:** `docs/superpowers/specs/2026-09-25-adversarial-test-suite-design.md`: the A8 row of the Phase 2 table (the nightly half), and "The integrations and the mem0 compatibility shim, against the real framework packages" in the table of decisions.

## Global Constraints

- The framework tests run in the nightly tier only, under `tests/adversarial/frameworks/nightly/`. The environment and probe code has fast-tier tests that need no network and no framework.
- Downloading packages from PyPI is approved. Nothing may call a model or any other network service. Every store uses `HashingEmbedder` and `NullLLM`.
- Every framework's telemetry is switched off through its documented environment variables. A network access by anything in the probe's process fails a test; it never skips one.
- The environments live outside the repository, under the home directory of the process that runs the suite. The disk budget for all of them is 8 GB.
- memvara is installed into each environment from the checkout under test.
- Every child process gets its environment from `harness.env.child_env`, and may only remove or add variables on top of it.
- A check that finds a bug that is not security-class is pinned as a strict expected failure, as `tests/harness/known_bugs.py` describes, with a placeholder issue number. A security-class finding goes into no file, commit or comment; it is reported to the maintainer only.
- Do not change anything under `memvara/`. A broken adapter is a bug to report, not code to fix here.
- Every file in `tests/adversarial` must be importable by `--doctest-modules` in both tiers, so no module imports a framework at module level, and every folder has an `__init__.py`.
- Commits name files explicitly. No commit message, file or comment carries an AI or model attribution.
- Every command below sets `PYTHONPATH` to the worktree, because an editable install can point at another checkout, and `TMPDIR` to a short private folder, written `$SHORT_TMP`.
- Write plainly: every sentence must be understood on its first reading.

## The frameworks and their pins

| Framework | Distribution memvara declares | Where memvara declares it | Floor | Installed beside it |
|---|---|---|---|---|
| LangChain | `langchain-core` | the `langchain` extra, `>=0.3` | 0.3.0 | nothing |
| LlamaIndex | `llama-index-core` | the `llama-index` extra, `>=0.13` | 0.13.0 | nothing |
| CrewAI | `crewai` | the `crewai` extra, `>=1.10.1` | 1.10.1 | nothing |
| LangGraph | `langgraph-checkpoint` | the `langgraph` extra, `>=4.1` | 4.1.0 | `langgraph`, unpinned, because the adapter's docstring compiles a graph with the store |
| mem0 | `mem0ai` | the `bench` extra, `>=2.0`; the shim says it is written against mem0 2.x | 2.0.0 | nothing; memvara is installed without the bench extra, which also pulls in nltk |

- **The floor** is the requirement pinned to its lower bound: `langchain-core>=0.3` becomes `langchain-core==0.3`, which pip resolves to 0.3.0. A floor that was never released fails to install, and the environment's test says so.
- **The latest** is what pip resolves the requirement to on this interpreter, asked with `pip install --dry-run --no-deps --report`. Resolution goes through pip rather than through `urllib`, because the python.org build of Python on macOS cannot verify PyPI's certificate until its certificate installer has been run, and pip carries its own.
- **The requirements are read from the wheel's METADATA**, not from `pyproject.toml`. The wheel is what pip reads when it installs `memvara[langchain]`, and reading it needs no `tomllib`, which Python 3.10 lacks.

## Review Focus

1. **A framework that reaches the network from a background thread or an exit handler, after its check has passed.** CrewAI's telemetry does exactly this when it is left on. The audit hook records every attempt with the phase it happened in, the report file is opened once per record so an exit handler's record still lands, and the network test reads attempts from every phase. Task 2 plants an attempt in an exit handler and shows it recorded.
2. **A reused environment that runs an old copy of memvara.** memvara is reinstalled into every reused environment, and a test compares each installed memvara file with this checkout's through the hashes pip wrote in `RECORD`. Task 3 plants a file whose hash differs.
3. **A build that stopped partway.** The marker file is written last, so a folder without it is rebuilt from nothing, and so is an environment whose refresh fails or which no longer holds its pinned release. Task 3 plants all three.
4. **The floor and the newest release resolving to the same version.** They share one environment and one probe run instead of building twice. Task 3 pins the shared directory.
5. **A check that hangs.** `faulthandler` stops the probe after a time limit and writes every thread's stack, the report names the check that was running, and each check after it is reported as never run. Task 2 plants a hang.

---

### Task 1: The plan

**Files:**
- Create: `docs/superpowers/plans/2026-09-26-adversarial-frameworks.md`

- [ ] **Step 1: Commit the plan.**

```bash
git add docs/superpowers/plans/2026-09-26-adversarial-frameworks.md
git commit -m "Plan the nightly tests of the framework adapters against the real packages"
```

### Task 2: The probe

**Files:**
- Create: `tests/adversarial/frameworks/__init__.py`, `tests/adversarial/frameworks/probe.py`
- Test: `tests/adversarial/frameworks/test_adv_framework_probe.py`

**Interfaces:**
- Produces, in `probe.py`:
  - `CHECK_SECONDS = 120.0`, the time one check may run.
  - `BlockedNetworkAccess(OSError)`, raised by the audit hook in place of a network access.
  - `network_access(event: str, args: tuple) -> str | None`: `"connect to <host>:<port>"`, `"send to <host>:<port>"`, `"look up <name>"`, or `None`. A connection or datagram counts unless its socket is a unix socket. A lookup (`getaddrinfo`, `gethostbyname`, `gethostbyname_ex`, `gethostbyaddr`) counts unless the name is `None`, empty, `localhost`, a loopback address, or a numeric address given to anything but the reverse lookup `gethostbyaddr`.
  - `Report(path)` with `emit(record: dict)`, thread-safe, opening the file for each record.
  - `install_guard(report, phase: Callable[[], str])`.
  - `Context(folder: Path)` with `folder`, `memvara(**options) -> Memvara` (in memory, `HashingEmbedder(dim=512)`, `NullLLM`, user `"alice"` unless given) and `close()`.
  - `checks(module) -> list[tuple[str, Callable]]`: functions named `check_*` that the module itself defines, in definition order.
  - `load(path) -> ModuleType`: imports a checks file by path.
  - `run(checks_file, report_file, *, check_seconds=CHECK_SECONDS) -> int` and `main(argv=None) -> int`, the command line `probe.py <checks> <report> [--check-seconds N]`.
  - `Result(check, passed, seconds, error_type, message, traceback)` with `describe() -> str`.
  - `Run(start, results, begun, network, finished)` with `result(check) -> Result`, which for a check with no outcome returns a failed `Result` of type `"ProbeStopped"` saying why.
  - `read(path) -> Run`, which skips a line cut short by a killed process.
- The report's records, one JSON object per line, by `kind`: `start` (`python`, `executable`, `memvara`, `distributions`), `begin` (`check`), `result` (`check`, `passed`, `seconds`, and on failure `error_type`, `message`, `traceback`), `network` (`access`, `event`, `phase`, `thread`, `stack`) and `end`.

- [ ] **Step 1: Write the failing tests.** `test_adv_framework_probe.py` runs the probe with this interpreter, `-I -B`, on checks planted in a temporary file, with the environment `environments.probe_env` gives (Task 3 adds it, so until then build it as `child_env(home)` without `PYTHONPATH`). Two module-scoped runs, so the file costs about one second:
  - `PLANTED` holds `check_passes`, `check_fails` (raises `AssertionError("planted failure")`), `check_connects` (`socket.create_connection(("127.0.0.1", 9), timeout=1)`), `check_looks_up_a_name` (`socket.getaddrinfo("example.invalid", 443)`), `check_reaches_out_after_the_last_check` (registers `socket.create_connection` with `atexit`), and `helper_that_is_not_a_check`.
  - `HANGS` holds `check_hangs` (`time.sleep(60)`) and `check_after_the_hang`, run with `--check-seconds 0.5`.
  - `test_each_check_runs_once_in_the_order_its_file_defines_it`: the results are the five checks in order, the helper is not run, and the run finished.
  - `test_a_failed_check_is_reported_and_the_next_one_still_runs`.
  - `test_a_connection_is_refused_and_recorded_against_its_check`: the check failed with `BlockedNetworkAccess`, and the only network record in its phase is `"connect to 127.0.0.1:9"`.
  - `test_a_name_lookup_is_refused_and_recorded_against_its_check`: `"look up example.invalid"`.
  - `test_an_attempt_after_the_last_check_is_still_recorded`: the check passed, and the phase `"exit"` holds `"connect to 127.0.0.1:9"` (Review Focus 1).
  - `test_the_start_record_names_what_the_interpreter_can_import`: `pytest` is among the distributions.
  - `test_a_check_that_hangs_is_stopped_and_named`: `begun == ["check_hangs"]`, no results, not finished, a non-zero exit status, `"Timeout"` in stderr, and the next check's result says it never ran because the probe stopped while running `check_hangs` (Review Focus 5).
  - `test_network_access_is_recognised_in_each_form`: with real, unconnected sockets, a TCP connect and a UDP `sendto` to 192.0.2.1 count, a `bind` does not, a unix-socket connect does not; each lookup event counts for `pypi.org` (also as bytes) and not for `127.0.0.1`, `::1`, `localhost`, `""` or `None`; `getaddrinfo` of a numeric address does not count and `gethostbyaddr` of one does; an `open` event does not.
  - `test_a_report_cut_short_is_read_as_far_as_it_goes`: a hand-written report whose last line is cut off gives the results before it, `begun` of two checks, not finished, and a `"ProbeStopped"` result for the second; a missing report says the probe stopped before its first check.
  - `test_only_the_check_functions_a_file_defines_are_listed`: a file with `check_b`, a helper, `check_a`, a `check_*` name imported from elsewhere and a `check_*` constant lists `["check_b", "check_a"]`.
- [ ] **Step 2: Run the file and watch it fail on the missing module.**

```bash
PYTHONPATH=$PWD TMPDIR=$SHORT_TMP python -m pytest -q -p no:cacheprovider tests/adversarial/frameworks/test_adv_framework_probe.py
```

Expected: a collection error, `cannot import name 'probe'`.

- [ ] **Step 3: Write `probe.py`.** The core, as it must behave:

```python
def network_access(event: str, args: tuple[Any, ...]) -> str | None:
    if event in _SENDS:  # socket.connect, socket.sendto, socket.sendmsg
        sock, address = args[0], args[-1]
        if sock.family == getattr(socket, "AF_UNIX", object()):
            return None
        verb = "connect to" if event == "socket.connect" else "send to"
        return f"{verb} {_address(address)}"
    if event in _LOOKUPS:
        host = args[0]
        if isinstance(host, bytes):
            host = host.decode("ascii", "replace")
        if host is None or host.lower() in _LOCAL_NAMES or _loopback(host):
            return None
        if _numeric(host) and event != "socket.gethostbyaddr":
            return None  # a numeric address is answered without a name server
        return f"look up {host}"
    return None


def install_guard(report: Report, phase: Callable[[], str]) -> None:
    def hook(event: str, args: tuple[Any, ...]) -> None:
        if not event.startswith("socket."):
            return
        what = network_access(event, args)
        if what is None:
            return
        report.emit({"kind": "network", "access": what, "event": event, "phase": phase(),
                     "thread": threading.current_thread().name,
                     "stack": traceback.format_stack(limit=16)[:-1]})
        raise BlockedNetworkAccess(f"the probe blocked a network access: {what}")

    sys.addaudithook(hook)


def run(checks_file: Path, report_file: Path, *, check_seconds: float = CHECK_SECONDS) -> int:
    report = Report(report_file)
    phase = ["start"]
    install_guard(report, lambda: phase[0])        # before anything imports a framework
    report.emit({"kind": "start", ..., "memvara": <memvara.__file__ or None>,
                 "distributions": distributions()})
    phase[0] = "load"
    module = load(checks_file)
    for name, function in checks(module):
        phase[0] = name
        report.emit({"kind": "begin", "check": name})
        faulthandler.dump_traceback_later(check_seconds, exit=True)
        with tempfile.TemporaryDirectory(prefix="check-") as folder:
            context = Context(Path(folder))
            try:
                function(context)
            except Exception as exc:  # every failure is a result, and the next check runs
                ...record passed=False with the error's type, message and traceback
            finally:
                faulthandler.cancel_dump_traceback_later()
                context.close()
        report.emit(record)
    phase[0] = "exit"                               # exit handlers still run after this
    report.emit({"kind": "end"})
    return 0
```

- [ ] **Step 4: Run the file and watch it pass.** Expected: `10 passed`.
- [ ] **Step 5: Commit.**

```bash
git add tests/adversarial/frameworks/__init__.py tests/adversarial/frameworks/probe.py tests/adversarial/frameworks/test_adv_framework_probe.py
git commit -m "Add the probe that runs a framework's checks inside its own environment"
```

### Task 3: The environments

**Files:**
- Create: `tests/adversarial/frameworks/environments.py`
- Test: `tests/adversarial/frameworks/test_adv_framework_environments.py`
- Modify: `tests/adversarial/frameworks/test_adv_framework_probe.py` (use `environments.probe_env`)

**Interfaces:**
- Consumes: `probe.read`, `probe.Run`; `harness.env.REPO`, `harness.env.child_env`.
- Produces, in `environments.py`:
  - `CACHE = Path.home() / ".cache" / "memvara-adversarial" / "frameworks"`, read when the module is imported, which pytest does while it collects, before any test points `HOME` at a temporary folder.
  - `MARKER = "memvara-environment.json"`, `RECIPE = 1`, `PINS = ("floor", "latest")`, `PROBE` (the path of `probe.py`).
  - `Framework(name, dist, extra, install_extra=True, companions=())` with the property `checks -> Path` (`nightly/checks_<name>.py`), and `FRAMEWORKS`, the five rows of the table above. `framework(name) -> Framework`.
  - `BuildError(RuntimeError)`.
  - `Declared(dependencies: tuple[str, ...], requirements: dict[str, str])`; `declared(metadata: str) -> Declared`; `wheel_metadata(wheel: Path) -> str`.
  - `floor(requirement: str) -> str`, raising `ValueError` for a requirement with no single lower bound.
  - `interpreter() -> str`, one line naming this interpreter's implementation, version, platform and base executable.
  - `Wanted(framework, version, requirement, dependencies, interpreter)` with `key` (a SHA-256 over `RECIPE`, the framework, its distribution, the version, the extra installed, the companions, the requirement, the dependencies and the interpreter; the pin is not in it), `directory` (`<dist>-<version>-<key[:12]>`) and `install_arguments(wheel) -> list[str]`.
  - `python_in(path) -> Path`: `bin/python`, or `Scripts/python.exe` on Windows.
  - `Pip(home, *, python=sys.executable, timeout=1800.0)` with `wheel(source, folder) -> Path`, `resolve(requirements) -> dict[str, str]`, `create(path) -> Path`, `install(python, arguments)` and `describe(python) -> tuple[Path, dict[str, str]]`. Every command runs with `child_env(home)` minus `PYTHONPATH`, `PIP_DISABLE_PIP_VERSION_CHECK=1` and `PIP_NO_INPUT=1`, and raises `BuildError` with the end of its output when it fails or runs past its time limit.
  - `versions_from_report(report: Mapping) -> dict[str, str]`.
  - `Prepared(wanted, path, python, how, seconds, purelib, distributions, size)`, where `how` is `"built"`, `"rebuilt"` or `"reused"`.
  - `Cache(root, pip, wheel)` with `prepare(wanted) -> Prepared` and `prune(keep: Iterable[Wanted]) -> list[Path]`.
  - `disk_usage(path) -> int`, counting a hard-linked file once.
  - `stale_files(purelib, checkout=REPO) -> list[str]`.
  - `TELEMETRY_OFF`, the documented switches: `LANGSMITH_TRACING=false` and `LANGCHAIN_TRACING_V2=false` (LangSmith), `CREWAI_DISABLE_TELEMETRY=true`, `CREWAI_DISABLE_TRACKING=true`, `OTEL_SDK_DISABLED=true` and `CREWAI_TRACING_ENABLED=false` (CrewAI), `ANONYMIZED_TELEMETRY=False` (chromadb, which CrewAI installs), `MEM0_TELEMETRY=False` (mem0), `HF_HUB_DISABLE_TELEMETRY=1` and `HF_HUB_OFFLINE=1` (the Hugging Face hub, which CrewAI installs), and `DO_NOT_TRACK=1`. llama-index-core has no telemetry of its own.
  - `probe_env(home) -> dict[str, str]`: `child_env(home, TELEMETRY_OFF + every proxy variable at http://127.0.0.1:9)`, without `PYTHONPATH`, and without any variable whose name has a part `KEY`, `API_KEY`, `TOKEN`, `SECRET`, `PASSWORD` or `CREDENTIAL(S)`.
  - `ProbeRun(run: probe.Run, exit_code: int | None, output: str, seconds: float)` and `run_probe(prepared, work, *, timeout=900.0) -> ProbeRun`, which runs `<python> -I -B probe.py <checks> <report>` with `probe_env`, stdin closed and stdout and stderr in one log file.
  - `Session(cache, wanted, work, removed)` with `prepare(name, pin) -> Prepared` and `probe(name, pin) -> ProbeRun`, each done once per environment directory (a failure is remembered and raised again as `BuildError`), `summary() -> list[str]` and `record() -> dict`.
  - `session(root, work, pip=None)`, a context manager: under an exclusive lock on `root/.lock` (POSIX only), build the wheel, read `declared` from it, resolve the floors and the latest releases, build `wanted` for every framework and pin, prune the cache, yield the `Session`, and write `root/last-run.json` when it ends.

`Cache.prepare` is the part a reviewer should read most closely:

```python
def prepare(self, wanted: Wanted) -> Prepared:
    path = self.root / wanted.directory
    python = python_in(path)
    started = time.monotonic()
    how = "built"
    if _read_marker(path).get("key") == wanted.key:
        try:
            self.pip.install(python, ["--no-deps", "--force-reinstall", str(self.wheel)])
            how = "reused"
        except BuildError:
            how = "rebuilt"
    if how == "reused":
        purelib, distributions = self.pip.describe(python)
        if distributions.get(canonicalize_name(wanted.framework.dist)) != wanted.version:
            how = "rebuilt"                  # someone emptied it; build it again
    if how != "reused":
        if path.exists():
            shutil.rmtree(path)
        python = self.pip.create(path)
        self.pip.install(python, wanted.install_arguments(self.wheel))
        purelib, distributions = self.pip.describe(python)
    seconds = time.monotonic() - started
    if how != "reused":
        _write_marker(path, {...the key, the framework, the version, the time it took...})
    return Prepared(wanted, path, python, how, seconds, purelib, distributions,
                    disk_usage(path))
```

- [ ] **Step 1: Write the failing tests** with `FakePip`, which makes folders and small files where pip would download and install, records every call, and raises `BuildError` for any call kind in its `failing` set:
  - `test_the_floor_is_the_lowest_version_a_requirement_admits`: `>=0.3`, `>=1.10.1,<2` and `==2.0` give their version; `test_a_requirement_with_no_lower_bound_has_no_floor`: `<2` and a bare name raise `ValueError`.
  - `test_the_wheel_gives_each_framework_its_requirement_and_memvara_its_dependencies`, on the METADATA lines hatchling writes, including an unrelated extra; `test_a_framework_the_wheel_does_not_name_is_an_error`.
  - `test_pips_report_gives_one_version_per_distribution`.
  - `test_the_key_changes_with_everything_the_environment_depends_on`: each of the version, the requirement, the dependencies and the interpreter changes the key; the floor and the latest at one version share one directory (Review Focus 4).
  - `test_a_missing_environment_is_built_and_then_reused`: the second `prepare` makes only the refresh install and one `describe`, and the marker holds the key.
  - `test_a_folder_left_by_an_interrupted_build_is_built_again`: a folder with `pyvenv.cfg` and no marker (Review Focus 3); `FakePip.create` refuses a folder that exists, so the test also proves the folder was removed first.
  - `test_an_environment_whose_refresh_fails_is_built_again` and `test_an_environment_without_its_pinned_release_is_built_again`: both give `how == "rebuilt"`.
  - `test_a_failed_build_leaves_no_marker`: the install fails, `prepare` raises `BuildError`, no marker exists, and the next `prepare` builds.
  - `test_pruning_removes_only_the_environments_the_pins_no_longer_need`: a wanted environment, an unwanted one with a marker, an interrupted one with only `pyvenv.cfg`, an unrelated folder and a file; only the two unwanted environments go.
  - `test_disk_usage_counts_a_hard_linked_file_once`.
  - `test_a_stale_installed_memvara_is_named_file_by_file`: a `RECORD` with this checkout's real hash for `memvara/__init__.py` and a wrong one for `memvara/core.py` names only `memvara/core.py`; a `RECORD` that lists no memvara file, and two installed copies, are both reported (Review Focus 2).
  - `test_the_probe_environment_drops_credentials_and_switches_telemetry_off`: with `OPENAI_API_KEY`, `LANGSMITH_API_KEY`, `HF_TOKEN`, `AWS_SECRET_ACCESS_KEY` and `GITHUB_TOKEN` set, none reaches the probe; `PYTHON_KEYRING_BACKEND` does; every `TELEMETRY_OFF` switch and every proxy variable is set; there is no `PYTHONPATH`.
  - `test_a_session_pins_every_framework_twice_and_prunes_the_rest`: ten wanted environments with the floors from the requirements and the latest versions from `FakePip`, an old environment in the root is removed, the wheel is built once, and `last-run.json` is written when the session ends.
  - `test_a_session_prepares_each_environment_once_and_remembers_a_failure`.
- [ ] **Step 2: Run the file and watch it fail on the missing module.**
- [ ] **Step 3: Write `environments.py`,** and point `_probe` in the probe tests at `environments.probe_env`.
- [ ] **Step 4: Run both fast files and watch them pass.**

```bash
PYTHONPATH=$PWD TMPDIR=$SHORT_TMP python -m pytest -q -p no:cacheprovider tests/adversarial/frameworks
```

Expected: every test passes, in about two seconds.

- [ ] **Step 5: Commit.**

```bash
git add tests/adversarial/frameworks/environments.py tests/adversarial/frameworks/test_adv_framework_environments.py tests/adversarial/frameworks/test_adv_framework_probe.py
git commit -m "Build, reuse and prune one virtual environment per framework and pin"
```

### Task 4: The nightly tests, with the LangChain checks

**Files:**
- Create: `tests/adversarial/frameworks/nightly/__init__.py`, `tests/adversarial/frameworks/nightly/conftest.py`, `tests/adversarial/frameworks/nightly/test_adv_frameworks_nightly.py`, `tests/adversarial/frameworks/nightly/checks_langchain.py`

**Interfaces:**
- Consumes: `environments.session`, `environments.CACHE`, `environments.FRAMEWORKS`, `environments.PINS`, `environments.stale_files`; `probe.checks`, `probe.Result`; `known_bugs.xfail`, `known_bugs.Reproduced`.
- Produces:
  - In `conftest.py`, the session fixture `frameworks -> environments.Session`, and a `pytest_terminal_summary` hook that prints `Session.summary()` under the heading "framework environments".
  - In the test module, `Symptom(error_type: str, text: str)` with `seen_in(result) -> bool`, and `PINNED: dict[tuple[str, str, str], tuple[pytest.MarkDecorator, Symptom]]`, keyed by framework, pin and the check's name without `check_`. Each value holds a literal `known_bugs.xfail("B..")` call, so the coverage checklist reads the pin from the source.
  - Each checks module: functions `check_<promise>(ctx: Context) -> None`, the framework imported inside each through `importlib`, and `from ..probe import Context` under `TYPE_CHECKING` only, because the probe loads the file by path.

- [ ] **Step 1: Write the test module.** Tests, each parametrised with ids like `langchain-floor`:
  - `test_the_environment_holds_the_release_it_pins`: the probe's start record, which is what the checks' own process could import, holds the pinned version of the framework's distribution.
  - `test_the_environment_runs_the_memvara_in_this_checkout`: `stale_files` is empty, and the probe imported memvara from inside the environment.
  - `test_nothing_in_the_environment_reached_the_network`: the probe finished, so it saw every phase, and recorded no network access. The failure lists each access with its phase, thread and the last frames of its stack.
  - `test_the_adapter_keeps_its_promise`, parametrised with ids like `langchain-floor-messages_come_back_as_the_classes_they_were_written_as`: raises `known_bugs.Reproduced` when the check's pin sees its symptom, and otherwise asserts the check passed, with the probe's output added when the probe stopped.
- [ ] **Step 2: Write `conftest.py` and `checks_langchain.py`.** The checks, each against the real langchain-core:
  - `the_history_is_a_real_basechatmessagehistory`: `isinstance` of `BaseChatMessageHistory`, and the composed class is made once.
  - `messages_come_back_as_the_classes_they_were_written_as`: a `SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage` with its call id and `ChatMessage` with the role `"critic"` round-trip in order.
  - `the_base_class_helpers_write_through_add_messages`: `add_user_message`, `add_ai_message`, `add_message`, `aadd_messages` and `aget_messages`.
  - `reading_and_writing_text_raises_no_deprecation_warning`: the promise in `_text_of`'s docstring.
  - `reading_the_transcript_warns_once_that_the_memory_is_not_in_it`, and `transcript_warning=False` silences it.
  - `a_repeated_turn_is_stored_once`: the documented deduplication.
  - `the_transcript_holds_only_its_own_session`: two sessions, and a named agent inside the same session, see only their own turns.
  - `clear_refuses_and_each_opt_in_does_what_it_names`: `clear()` and langchain-core's `aclear()` refuse naming both options; `on_clear="ignore"` keeps the turns; `on_clear="purge"` leaves no turn in the store.
  - `runnable_with_message_history_reads_and_writes_the_history`: langchain-core's `RunnableWithMessageHistory` over a prompt and `FakeListChatModel`; the second prompt holds the first exchange, and the store holds all four turns.
  - `the_retriever_is_a_real_baseretriever_on_the_modern_path`: `_new_arg_supported` is true and `_expects_other_args` false, as the adapter's docstring says `run_manager` in its signature ensures.
  - `the_retriever_returns_documents_carrying_the_claim`: `invoke()` returns `Document`s whose metadata holds the triple, both time axes, `why`, `sources`, `scope` and `score`.
  - `the_retriever_answers_ainvoke_and_batch`.
  - `an_ended_value_is_absent_now_and_present_at_an_earlier_as_of`.
  - `an_episode_comes_back_labelled_as_an_episode`: `kind == "episode"` and no predicate.
  - `the_retriever_composes_into_a_chain`: `retriever | RunnableLambda(...)`.
  - `recall_and_search_on_the_history_reach_the_memory`: the recall block is framed as reference data, and the search result keeps its source turn.
- [ ] **Step 3: Run the nightly tier of the folder for LangChain only** and read every failure. A check that fails because the check is wrong is fixed in the check. A check that fails because of memvara is a finding: classify it against the "In scope" section of `SECURITY.md` before writing anything down.

```bash
PYTHONPATH=$PWD TMPDIR=$SHORT_TMP python -m pytest -q -p no:cacheprovider tests/adversarial/frameworks/nightly --tier nightly -k langchain
```

Expected: every LangChain test passes; nothing reaches the network.

- [ ] **Step 4: Commit.**

```bash
git add tests/adversarial/frameworks/nightly/__init__.py tests/adversarial/frameworks/nightly/conftest.py tests/adversarial/frameworks/nightly/test_adv_frameworks_nightly.py tests/adversarial/frameworks/nightly/checks_langchain.py
git commit -m "Run the LangChain adapter against the real langchain-core at its floor and newest release"
```

### Task 5: The LlamaIndex checks

**Files:**
- Create: `tests/adversarial/frameworks/nightly/checks_llamaindex.py`
- Modify: `tests/adversarial/frameworks/nightly/test_adv_frameworks_nightly.py`, `tests/harness/known_bugs.py`

- [ ] **Step 1: Write the checks.**
  - `the_block_is_a_real_basememoryblock_named_memvara`.
  - `the_docstring_memory_example_works_offline`: the `Memory.from_defaults(...)` example in `MemvaraMemoryBlock`'s docstring, taken from the docstring and evaluated, builds a working `Memory` with the default tokenizer. The example is code from this checkout, so evaluating it runs nothing that is not already in the repository.
  - `flushed_turns_go_through_memvaras_write_path`: a `Memory` with a short buffer and a whitespace tokenizer flushes older turns into the block; the store holds `lives_in Berlin`, and the receipt says no model was called.
  - `the_prompt_frames_memory_as_reference_data`: a stored value that spells out a newline, a list item, the recall header and a bracketed id reaches LlamaIndex's system message as one line under exactly one header, with the brackets replaced.
  - `a_block_that_refuses_short_term_memory_writes_nothing`, while a direct `aput` still writes.
  - `nothing_to_query_on_contributes_nothing`: `aget([])` and a message with no content give `""`.
  - `the_retriever_returns_nodes_carrying_the_claim`: `NodeWithScore` of a `TextNode` whose id is the claim id.
  - `the_retriever_answers_from_text_not_from_a_supplied_vector`, and `aretrieve`.
  - `the_retriever_works_inside_a_query_engine`: `RetrieverQueryEngine.from_args(..., llm=MockLLM())`.
  - `the_retriever_docstring_example_builds_a_query_engine`: the `as_query_engine(retriever=...)` example in `MemvaraRetriever`'s docstring, with a `SummaryIndex` as `index_or_engine` and `MockLLM` as the global model.
  - `time_travel_and_episode_labels_survive_the_retriever`.
  - `the_refusals_name_the_supported_shape`: `as_vector_store()` names `MemvaraRetriever`, `as_chat_memory()` names `Memory.from_defaults`.
  - `search_and_history_on_the_block_reach_the_structure`.
- [ ] **Step 2: Run the nightly tier with `-k llamaindex`.** Expected, measured while this plan was written: every check passes at 0.13.0 and 0.14.25 except the docstring example, which raises `TypeError: RetrieverQueryEngine.from_args() got multiple values for argument 'retriever'` at both, because `as_query_engine` builds its own retriever from its keyword arguments and passes them on to `from_args` too. It is not security-class.
- [ ] **Step 3: Register B83 in `known_bugs.py`** with the placeholder issue 9004, and pin the check at both pins with `Symptom("TypeError", "got multiple values for argument 'retriever'")`.
- [ ] **Step 4: Run it again.** Expected: the two docstring cases are xfailed, everything else passes.
- [ ] **Step 5: Commit** the checks file, the test module and `known_bugs.py`.

### Task 6: The LangGraph checks

**Files:**
- Create: `tests/adversarial/frameworks/nightly/checks_langgraph.py`
- Modify: `tests/adversarial/frameworks/nightly/test_adv_frameworks_nightly.py`

- [ ] **Step 1: Write the checks.** Stores get an injected clock one minute apart, so no two writes share an instant.
  - `the_store_is_a_real_basestore_without_ttl`.
  - `every_json_value_round_trips_through_put_and_get`, and the string "123" stays apart from the number 123.
  - `a_replaced_field_ends_and_a_dropped_field_is_retired`: the changed field's old value has `valid_to` set and `invalidated_at` unset, the dropped field's value the reverse, and both rows remain.
  - `an_unchanged_field_is_not_rewritten`: one claim, `updated_at` unmoved, `created_at` kept across a real change.
  - `delete_retires_by_default_and_warns_once`.
  - `erase_mode_removes_the_items_text`: `prove_erased` for each field's claim.
  - `search_ranks_by_the_query_text_on_the_normalised_scale`: a complete `SearchPage`, the relevant item first, scores in [0, 1].
  - `filters_select_what_inmemorystore_selects`: twelve filters, over three prefixes, against langgraph's own `InMemoryStore`, compared as sets, because the adapter documents newest-first order for an unranked search and `InMemoryStore` keeps insertion order.
  - `list_namespaces_answers_what_inmemorystore_answers`: ten questions.
  - `the_documented_differences_from_inmemorystore_hold`: `InMemoryStore` resets `created_at` on every put and memvara keeps it; a `$gt` on a field an item lacks makes `InMemoryStore` raise `TypeError` and matches nothing here; an emptied namespace is still listed by `InMemoryStore` and not here.
  - `ttl_is_refused_by_put_and_by_batch`.
  - `index_paths_are_parsed_by_langgraph_itself`: `index=["context[*].content"]` makes only those texts searchable, and `index=False` keeps a value out of the index.
  - `the_async_methods_answer_like_the_sync_ones`.
  - `a_compiled_graph_reads_and_writes_through_the_store`: `builder.compile(store=store)` from the docstring, with nodes that reach the store through `langgraph.config.get_store()`, under `invoke` and `ainvoke`.
  - `history_and_search_memory_reach_the_structure`.
- [ ] **Step 2: Run the nightly tier with `-k langgraph`.** Expected: everything passes at 4.1.0 and 4.2.0, with `langgraph` 1.2.12 beside both.
- [ ] **Step 3: Commit** the checks file and the test module.

### Task 7: The CrewAI checks

**Files:**
- Create: `tests/adversarial/frameworks/nightly/checks_crewai.py`
- Modify: `tests/adversarial/frameworks/nightly/test_adv_frameworks_nightly.py`, `tests/harness/known_bugs.py`

CrewAI's `Memory` builds its analysis model the first time it saves anything, even when every field is given and no analysis is needed, and the default model needs an OpenAI key. So each check that uses `Memory` passes `llm=` a `BaseLLM` subclass that fails the check if anything calls it. That also proves the adapter's own writes cost no model call.

- [ ] **Step 1: Write the checks.**
  - `the_storage_satisfies_the_storagebackend_protocol`: `isinstance` of the runtime-checkable protocol, and every protocol method takes the same parameters by name.
  - `crewais_memory_remembers_and_recalls_through_the_storage`: the documented wiring, `remember()` with scope, categories and importance given, and `recall(depth="shallow")`.
  - `a_record_round_trips_as_crewais_own_memoryrecord`.
  - `crewais_scorer_accepts_the_records_it_gets_back`: `compute_composite_score` on a record read back.
  - `the_embedder_satisfies_crewais_embed_helpers`: `embed_text` and `embed_texts` get plain lists of floats.
  - `a_vector_from_another_model_is_refused`, naming `embedder=storage.embedder`.
  - `update_ends_the_old_text_and_delete_retires_it`: `Memory.update()` ends the old text with one clock, and `Memory.forget()` retires the record with the other.
  - `forget_warns_once_that_it_retired`, naming `on_delete='erase'`.
  - `reset_leaves_nothing_behind`: `Memory.reset()`, then `prove_erased` for every claim the storage wrote.
  - `listing_and_scope_info_come_back_as_crewais_types`.
  - `a_metadata_filter_is_refused` by search and delete.
  - `the_async_methods_answer_like_the_sync_ones`.
  - `two_storages_on_one_memvara_cannot_see_each_other`.
  - `a_repeated_memory_reaches_crewais_consolidation`: remembering one sentence twice, with a model that answers CrewAI's consolidation question by deleting the old record, asks that question once and leaves one live record.
- [ ] **Step 2: Run the nightly tier with `-k crewai`.** Expected, measured while this plan was written:
  - At 1.10.1, the floor, `Memory.remember()` fails in both checks that call it with `AttributeError: 'MemvaraStorage' object has no attribute 'write_lock'`. CrewAI 1.10.1's `EncodingFlow.execute_plans` takes `storage.write_lock`, which its protocol does not declare; 1.11.0 and later do not. So memvara's declared floor cannot save anything through CrewAI's `Memory`.
  - At the newest release, the consolidation check fails: an exact duplicate scores 0.50 through `MemvaraStorage.search`, CrewAI consolidates only at 0.85, so it never asks, and two live copies remain.
  - Neither is security-class.
- [ ] **Step 3: Register B84 (placeholder 9005) and B85 (placeholder 9006)** and pin: both remember checks at the floor with `Symptom("AttributeError", "object has no attribute 'write_lock'")`, and the consolidation check at the newest release with `Symptom("AssertionError", "CrewAI asked its model to consolidate 0 times")`. The floor's consolidation check fails on B84 before it can reach B85; the comment on its pin says that the fix for B84 moves that pin to B85.
- [ ] **Step 4: Run it again.** Expected: three xfailed, everything else passes, nothing reaches the network.
- [ ] **Step 5: Commit** the checks file, the test module and `known_bugs.py`.

### Task 8: The mem0 checks

**Files:**
- Create: `tests/adversarial/frameworks/nightly/checks_mem0.py`
- Modify: `tests/adversarial/frameworks/nightly/test_adv_frameworks_nightly.py`, `tests/harness/known_bugs.py`

The shim says it is mem0 2.x's method surface, so that an existing call site keeps working. These checks compare it with the real package without starting any of mem0's backends: they read signatures and pydantic models, and call mem0's own methods only where mem0 refuses a call before it touches a backend. The importer checks use mem0's `SQLiteManager`, the class mem0 writes its history file with, because reading that file is the importer's whole job.

- [ ] **Step 1: Write the checks.**
  - `the_shim_takes_every_method_and_argument_mem0_takes`: every public method of mem0's `Memory` and the with-statement protocol exist on the shim, and every argument mem0's method names is named by the shim's. The entity ids have their own check; `chat()` is left out because mem0's own raises `NotImplementedError`. The message lists what is missing, sorted.
  - `add_and_delete_all_take_the_entity_ids_mem0_takes`: mem0's `add()` and `delete_all()` name `user_id`, `agent_id` and `run_id`; `shim.add(..., user_id="alice")` stores the fact for alice, and `shim.delete_all(user_id="alice")` erases it.
  - `search_and_get_all_refuse_entity_ids_as_mem0_does`: mem0 raises `ValueError` (called on an instance whose `__init__` never ran), and the shim must raise the same type.
  - `defaults_match_mem0s_except_the_documented_threshold`.
  - `every_row_carries_mem0s_memoryitem_fields`: search and get rows carry every `MemoryItem` field, get_all rows every field but `score`, as mem0's own do.
  - `history_rows_carry_mem0s_history_columns`.
  - `import_mem0_dates_each_event_when_mem0_recorded_it` and `import_mem0_replays_each_event_after_the_add_it_changes`: a history file written by `SQLiteManager` the way mem0 writes one, where an UPDATE or DELETE row holds its memory's creation time in `created_at` and the time of the event in `updated_at` (`_update_memory` and `_delete_memory` in `mem0/memory/main.py`). mem0 gives rows random ids; the check fixes them so each ADD sorts first in one check and last in the other, since either order happens.
  - `update_and_from_config_refuse_with_mem0compaterror`, for every argument mem0's own `update()` takes.
- [ ] **Step 2: Run the nightly tier with `-k mem0`.** Expected, measured while this plan was written at 2.0.0 and 2.2.1 (the history columns check passes):
  - `add()` refuses `user_id` with `TypeError: ... mem0 2.x moved entity ids into filters=`, though mem0 2.x's own `add()` requires one of the three ids and its `delete_all()` takes them; and `search()` refuses a top-level id with `TypeError` where mem0 raises `ValueError`.
  - Missing from the shim at 2.0.0: `close()`, `from_config(config_dict=)` and `update(metadata=)`. At 2.2.1 also the with statement, `add(timestamp=, expiration_date=)`, `search(reference_date=, show_expired=)`, `get_all(show_expired=)` and `update(expiration_date=)`. `update()` given `metadata=` raises `TypeError`, not the documented `Mem0CompatError`, and a `get()` row lacks the `score` key mem0's carries as `None`.
  - The defaults differ: `search(top_k=10)` and `get_all(top_k=100)` where mem0 has 20 for both.
  - `import_mem0` dates each event at its row's `created_at`, which for an UPDATE or DELETE is the memory's creation time. So the update is dated 2024-03-01 and the delete 2024-03-02, and with the ADD sorting last the import leaves the old value and the deleted memory live.
  - None is security-class under `SECURITY.md`: they are refusals, missing arguments, different defaults, and an import that dates and orders mem0's events wrongly.
- [ ] **Step 3: Register B80 (placeholder 9001, the entity ids), B81 (9002, the missing surface), B82 (9003, the defaults) and B86 (9007, the importer's dates),** and pin each failing check at the pins where it fails, with the exact symptom texts measured in Step 2.
- [ ] **Step 4: Run it again.** Expected: the pinned cases are xfailed, the rest pass, nothing reaches the network.
- [ ] **Step 5: Commit** the checks file, the test module and `known_bugs.py`.

### Task 9: The coverage checklist

**Files:**
- Modify: `tests/adversarial/frameworks/nightly/test_adv_frameworks_nightly.py`, `tests/harness/checklist_baseline.txt`

- [ ] **Step 1: Mark what the tests genuinely check.**
  - `inv:I5`, "The library must run with no API key and no network", on `test_nothing_in_the_environment_reached_the_network`: every adapter ran with no credential in its environment and with every network access blocked and recorded.
  - `inv:I3`, "end-of-life moves exactly one clock", on a test of its own, `test_ending_a_value_through_an_adapter_moves_exactly_one_clock`, which reports the LangGraph check `a_replaced_field_ends_and_a_dropped_field_is_retired` and the CrewAI check `update_ends_the_old_text_and_delete_retires_it`.
  - `inv:RT3`, "The recall header names the text as data", on a test of its own, `test_the_prompt_an_adapter_builds_frames_memory_as_data`, which reports the LlamaIndex check `the_prompt_frames_memory_as_reference_data`.
  - The general test leaves out the checks those two report, so no check is reported twice.
- [ ] **Step 2: Delete `inv:I3`, `inv:I5` and `inv:RT3` from the baseline,** and run `tests/adversarial/test_adv_checklist.py`. Expected: it passes; it fails if a line is left behind or a covered item is still listed.
- [ ] **Step 3: Commit** the test module and the baseline.

### Task 10: Documentation

**Files:**
- Modify: the testing guide (`testing.md` in the folder of context pages under `docs/`), that folder's `README.md` index, the root instructions file that repeats the index, and `CHANGELOG.md`.

- [ ] **Step 1: Add the section "Framework adapters against the real packages" to the testing guide,** before its last line: what runs, where the environments live and when each is reused, rebuilt or removed, the probe and what its guard does and does not see, the telemetry switches, the commands, the measured sizes and times, the bugs pinned, and how to add a check.
- [ ] **Step 2: Add "framework adapters against the real packages" to the suite's row** in the index and in the root instructions file.
- [ ] **Step 3: Add a bullet for the framework tests** to the CHANGELOG's entry for the suite.
- [ ] **Step 4: Commit** the four files, naming each; the commit message says "the testing guide" and "the index" rather than their paths.

### Task 11: Verification and review

- [ ] **Step 1: Run the fast tier of the folder 20 times** and record every result line.
- [ ] **Step 2: Run the nightly tier of the folder twice,** first with the environments built from nothing, then reusing them, and record each result line and the terminal summary's sizes and times.
- [ ] **Step 3: Type-check:** `mypy -p memvara`, `mypy tests/harness`, `mypy tests/harness --ignore-missing-imports`, and `mypy tests/adversarial/frameworks` with `MYPYPATH` naming the worktree and its `tests` folder, so that `memvara` and `harness` resolve to this checkout.
- [ ] **Step 4: Run the full gate** as two commands with a private coverage file, after checking that no other `coverage run` process is running: `coverage run -m pytest -q -p no:cacheprovider`, then `coverage report`. Coverage must stay at 100%.
- [ ] **Step 5: Update the count lines** in `README.md` and `CONTRIBUTING.md` to the gate's result line, and commit.
- [ ] **Step 6: Review the whole branch,** fix what the review finds in one pass, run the checks the fixes can move again, and commit.
