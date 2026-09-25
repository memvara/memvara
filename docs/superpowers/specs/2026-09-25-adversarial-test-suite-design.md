# An adversarial test suite that uses memvara the way agents do

**Status:** design, 2026-09-25. Agreed with the maintainer in an interview the same day.

**Order of work:** the shared foundation, then the data-loss hunt (property-based and
crash tests), then the agent-facing tests, the live tiers, the long-horizon tiers and the
nightly run. Each workstream lands as its own PR.

## Why this exists

The existing suite is large and strict, and it still misses a class of bug. On 2026-09-25,
at commit `d76e5813`:

- There are 5,005 test functions and a 100% statement-coverage gate on `memvara/`.
- Every test runs offline and in-process.
- No test starts the MCP server as a subprocess and talks to it over a real pipe.
- No test uses a real model, and nothing uses property-based testing.
- The plugin hooks (about 9,700 lines), `release/`, the npm bridge's `oauth.js` and several
  bench scripts are outside the coverage gate. Most of that code has no tests.

A few minutes of probing through the public API and the stdio server found five bugs that
no existing test catches. Four of them are listed below. The fifth is being handled through
the private channel described in `SECURITY.md`, so its details are not in this document.

| # | Bug | How it was found |
|---|---|---|
| B2 | A value written inside a session ends the user-wide value, but only that session can see the new value. User-wide reads and every other session then return nothing. | User-wide `lives_in Berlin`, then session s1 writes `lives_in Paris`. User-wide reads return `[]`, s1 returns `[Paris]`, s2 returns `[]`. |
| B3 | The plugin's auto-approve list leaves out two read-only tools, `memory_get_document` and `memory_list_documents`, so both ask for permission. | `plugin/hooks/approve.py`, the `READ_ONLY` set |
| B4 | One request line with deeply nested JSON kills the stdio MCP server, so the agent loses memory for the rest of the session. | `RecursionError` inside `json.decoder`; the server exits with code 1. |
| B5 | `memory_standing` with `k=0` says "No standing preferences are stored" when one exists. | Over stdio, `k=0` gives the empty answer, and the default `k` returns 1. |

The suite exists to find this kind of bug on purpose, before users do.

## What was decided

| Topic | Decision |
|---|---|
| Layers | **Scripted agent sessions**: deterministic, over the real stdio pipe and the real hook scripts, on every PR. **Real agents** doing multi-session tasks, graded, nightly. **Red-team agents** told to break memvara, nightly; every confirmed break becomes a deterministic regression test. |
| Surfaces | The library, the local server and the CLIs. The plugin hooks for all five hosts. The hosted path (the remote client, the npm bridge and cloud mode). The integrations and the mem0 compatibility shim, against the real framework packages. The release scripts are out of scope. |
| Hosted target | A throwaway local memvara-cloud Compose stack for every hosted test, including the destructive ones, plus a small read-mostly smoke set against app.memvara.dev under a dedicated test key. This suite tests the client side; memvara-cloud's own server code keeps its own suite. |
| Attack types | Property-based testing and fuzzing (Hypothesis). Concurrency and kill -9 crashes. Adversarial content and security. Long-horizon soak and performance. Also: upgrade and migration of old stores, parity across surfaces, a misbehaving extraction model, hook protocol conformance, docs-as-tests, and mutation testing. |
| Agent hosts | Claude Code headless on every nightly run. Codex, Cursor, OpenCode and Copilot weekly. |
| Models | The agent under test runs on the latest Sonnet, and the red-team attacker on the latest Opus. Both are resolved at run time through the Claude Code aliases `sonnet` and `opus`. Every report records the resolved model id and marks a trend break when it changes. OpenCode uses a self-hosted model; Codex uses its own default; Cursor and Copilot use the newest Sonnet they list. |
| Found bugs | The failing test lands at once as `xfail(strict=True)` and cites a GitHub issue. Fixes follow in separate PRs, data-loss bugs first. No test is ever skipped, deleted or weakened to make the gate green. |
| Security-class findings | Anything that falls under the in-scope list in `SECURITY.md` is never filed as a public issue and never lands as a public xfail. It goes to a private draft advisory, and its failing test lands together with its fix. |
| B2 | A session's or agent's value shadows the user-wide value and does not end it. This deliberately reverses the reasoning in `memvara/types.py` (`owner_key`), so the fix PR opens an issue first, as CONTRIBUTING asks for any change to scope resolution, and rewrites that reasoning and INTERNALS. |
| Cadence | The fast tier joins the existing PR CI and adds at most about 3 minutes, measured on the slowest job (Windows). Everything else runs nightly on the maintainer's Mac as a scheduled Claude Code session. The other agent hosts run weekly. |
| Nightly actions | Write a report. Send a notification only when something new breaks. File one deduplicated issue per new confirmed break, and open a draft PR that adds its strict-xfail test. Nothing merges without the maintainer's review and the code review. |
| Real data | Committed tests use only made-up data, because this repository is public. A local-only tier replays the maintainer's own Claude Code transcripts through the hooks against a throwaway store. Its reports keep counts and ids only, never text. |
| Build order | The foundation, then the data-loss hunt, then everything else. |

### The B2 rule in detail

It mirrors how project shadowing already works.

- **(a)** A user-wide write does not end a session's or agent's own value, because they are
  separate slots for supersession.
- **(b)** A present-tense read takes the value from the narrowest scope in the reader's
  ancestor chain that holds one, in `Scope.ancestors()` order. Reads at another instant,
  and `count()`, are not shadowed, as with projects.

### One more rule

**A configured local store that cannot open must not be reported as "not configured."**
The hooks keep four outcomes distinct: memories recalled, no matching memories, recall
failed, and not configured. A store that exists and fails to open is "recall failed".

## Architecture

### Layout

```
tests/conftest.py                +~20 lines: --tier option, pytest_ignore_collect, skip ledger
tests/harness/                   shared support package; holds no test files
  env.py clock.py stores.py stdio.py hooks.py scenarios.py report.py tiers.py skips.py
  known_bugs.py model.py invariants.py crash.py crash_child.py checklist.py
  invariant_ids.json checklist_baseline.txt
  fakes/  fake_v1.py hosted_mcp.py openai_compat.py cli.py
tests/scenarios/                 one JSON format for both layers, with schema.json
  scripted/*.json  agent/*.json
tests/adversarial/               the deterministic suite; every file is named test_adv_*.py
  test_adv_harness.py test_adv_known_bugs.py test_adv_checklist.py
  model/  concurrency/  sessions/  hooks/  fuzz/  security/  parity/  upgrade/
  model_faults/  docs/  frameworks/  regressions/     (each may hold nightly/ weekly/ local/)
tests/fixtures/stores/<tag>/     small stores from old releases, each with golden.json
tests/live/                      local tier only: sandbox, agents/, grading, judge,
                                 redteam/, stack, provision, prod_smoke, replay
bench/soak.py  bench/perf_budget.py  bench/mutation.py
bench/expected/perf_budgets.json  bench/mutation_equivalents.toml
scripts/nightly/                 run.py, watchdog.py, flakes.py, task.md, launchd template
docs/claude/testing.md           how to run and extend the suite
```

### Tiers

- **Where a test's tier comes from:** its path.
  - A folder named `nightly`, `weekly`, `local` or `quarantine` sets that tier.
  - `tests/live/**` is always `local`.
  - Everything else is `fast`.
- **How the higher tiers stay out of a plain run.** `pytest_ignore_collect` drops them at
  collection time, so a plain `pytest -q` never imports them and never reports them as
  skipped. `--tier` widens the run:

  | Flag | Runs |
  |---|---|
  | `--tier nightly` | fast and nightly |
  | `--tier weekly` | fast, nightly and weekly |
  | `--tier local` | only local |
  | `--tier quarantine` | only quarantine |

- **Why not markers, or a separate top-level folder?**
  - `--doctest-modules` imports every `.py` under `tests/`. A nightly module that imports
    docker or mem0 would therefore break collection on every PR.
  - Tests excluded by a marker still show up as skipped or deselected.
  - A folder outside `tests/` would lose the guards in `tests/conftest.py`: HOME
    redirection, the null keychain backend, and the embedder guard.
- **The skip ledger.** A run fails when any skip gives a reason that has no entry in
  `tests/harness/skips.py`. The file starts with every skip site that exists today, and
  each entry records the pattern, the reason and the platforms it applies to.
- **Hypothesis profiles.** `.hypothesis/` is added to `.gitignore`.

  | Profile | Settings |
  |---|---|
  | fast | derandomized, no example database, about 30 examples × 25 steps |
  | nightly | 3,000 examples × 100 steps, with an example database under `~/.cache/memvara-adversarial/` |
  | weekly | 20,000 examples × 200 steps |

### The shared foundation (`tests/harness`)

Each module lands in the first PR that uses it:

- `env`, `stdio`, `hooks`, `stores`, `tiers` and `skips` in F1;
- `known_bugs` in F2;
- the fakes in F3;
- `scenarios` in F4;
- `clock`, `model` and `invariants` in D1;
- `report` with the nightly run.

Their interfaces are written down here before any parallel work starts.

- **`env.child_env(home, extra)`** builds a subprocess environment from an allowlist.
  - HOME and USERPROFILE point at a temporary directory; the real HOME is refused.
  - `PYTHONPATH` points at the checkout under test. An editable install can point
    somewhere else entirely, and a child process would silently import that copy.
  - The hashing embedder is selected, encryption and project scope are off, and the
    keyring backend is the null one.
  - Every inherited `MEMVARA_*`, `ANTHROPIC_*` and `OPENAI_*` variable is removed. CI
    exports `MEMVARA_API_KEY`.
- **`stdio.McpProcess(db, home, scope, features, encrypted, env, cwd, timeout)`** starts
  `python -m memvara.server` and speaks newline-delimited JSON-RPC.
  - Protocol calls: `initialize`, `list_tools`, `call(name, **args) -> ToolResult`,
    `request`, `notify`, `send_raw`, `recv`.
  - Process control: `kill` works on every OS, `signal` on POSIX only, and `close`
    returns the exit code.
  - It records a `transcript` and the server's `stderr`.
  - On Windows, reader threads stand in for `select`, and the finalizer kills the process
    before temporary files are deleted.
- **`hooks.HookRunner(host, home, client_env, cwd, stubs, daemon)`** runs
  `plugin/hooks/run.py <hook> --host <host>` with a payload shaped the way that host sends
  it.
  - `.run()` returns the exit code, the parsed reply, the elapsed time and any new log
    lines.
  - It writes the host's client config, in JSON (TOML for Codex).
  - It puts stub agent CLIs first on PATH, and each stub records the arguments it was
    called with.
- **`stores`**: `memory()`, `file(tmp, encrypted)`, `second_handle()`, `reopen()` and
  `TEST_KEY`. Every factory passes `embedder=`.
- **`clock`**: `EPOCH`, `VirtualClock`, `WALL0` and `PRESENT`. Time is controlled by
  passing explicit datetimes, never by patching the clock.
- **`model.ReferenceStore`**: a pure-Python model of claims over both clocks and all five
  scope levels. It offers `apply(op)`, `expected(read)` and `diff(store)`.
- **`known_bugs.KnownBug(issue, xfail_nodeid, trigger)`**: the registry of open bugs. The
  state machines avoid a bug's trigger and log `avoided <id>` until its fix lands.
- **`invariants.check_store_integrity(db)`** checks four things:
  - SQLite's `integrity_check`;
  - the FTS integrity check;
  - that rowids line up with the `.vecs` file;
  - reference integrity.

  Every layer calls it after a session.
- **`report.Finding`**: one JSON line per finding, with the layer, surface, invariant,
  severity, replayable ops, seed, artifacts and commit. Its `signature()` is the key for
  deduplication.
- **`fakes`**:
  - `FakeV1`: the `/v1` routes, answered by a local `Memvara`. It runs as an
    `httpx.MockTransport` or on 127.0.0.1, and a test can inject a status, a delay or a
    hang.
  - `FakeHostedMcp`: a fake hosted `/mcp` endpoint.
  - `FakeOpenAI`: a fake chat-completions endpoint.
  - Fake agent CLIs that stand in for the hooks' extractor.

### One scenario format for both layers

`tests/scenarios/schema.json` defines one format for the scripted layer and the
real-agent layer. A scenario has these fields:

| Field | What it holds |
|---|---|
| `id`, `tier` | The scenario's name and tier |
| `surfaces` | Which surfaces it runs on |
| `env` | The user, feature switches, read-only mode and protocol version |
| `workspace` | Optional files for coding tasks |
| `seed` | Operations that populate the store before the sessions |
| `sessions` | A list of sessions. Each is a list of turns shaped `{user, script?}`, and the optional `script` is the deterministic agent. |
| `store_gold` | Expected claims, compared by text and state, never by id |
| `answer_gold` | Text the answer must contain, text it must not contain (the superseded "trap" value), or an expected abstention |
| `judge_rubric` | Optional rubric items for an LLM judge |
| `forbidden` | Tool calls that must not happen |
| `requires` | Capabilities the host must have |
| `negative_control` | Whether the scenario must fail when memvara is off |
| `known_bugs` | Which issue breaks which gold item |

- **The scripted layer** runs `script`.
- **The real-agent layer** sends only the `user` text and is graded on the same gold.
- **Each gold item is its own test id**, so a strict xfail marks exactly the assertion
  that a known bug breaks.

## Workstreams

Each PR works in its own worktree and branch and owns a disjoint set of files. Tests come
first. A test that finds a real bug lands as a strict xfail with a new issue, unless the
bug is security-class.

### Phase 0: Foundation

F1 comes first; everything else waits for it.

| PR | Content | Owns |
|---|---|---|
| F1 | The harness core, the tiers, the skip ledger, Hypothesis in `[dev]`, `.gitignore`, the tiers section of CONTRIBUTING, `docs/claude/testing.md` and its index rows, and this spec | `tests/conftest.py`, `tests/harness/{env,stores,stdio,hooks,tiers,skips}.py`, `tests/adversarial/conftest.py`, the `test_adv_*.py` self-tests beside them, and the tier guard folders |
| F2 | The public known bugs as strict xfails, with one issue each | `tests/harness/known_bugs.py` (append-only for later PRs), `tests/adversarial/test_adv_known_bugs.py` |
| F3 | The fakes | `tests/harness/fakes/*` |
| F4 | The scenario schema and runner, and the first scripted scenarios | `tests/scenarios/**`, `tests/adversarial/sessions/runner.py` |
| F5 | The checklist meta-test and its ratchet | `tests/harness/{checklist.py,invariant_ids.json,checklist_baseline.txt}`, `test_adv_checklist.py` |

**What the checklist enumerates.** Everything is taken from the code, not from lists kept
by hand:

- the tools in `TOOLS`;
- the feature switches in `FEATURES`, plus read-only and anchored mode;
- tool and switch pairs, found by diffing `tools/list` with each switch turned off;
- every `MEMVARA_*` variable that `config.py` reads, found through its AST;
- every hook on every host;
- the numbered INTERNALS invariants, and the invariant bullets in the `docs/claude`
  pages;
- the silent failure modes that `memvara/telemetry.py` lists;
- the known bugs.

**How a test gets counted.**

- A test declares what it covers with `@pytest.mark.covers("tool:…", "inv:…", …)`.
- A static scan collects those declarations. An xfail test covers nothing.
- `checklist_baseline.txt` lists the gaps that exist today, and the fast tier fails when a
  new gap appears.
- The work is done when the baseline is empty.

### Phase 1: The data-loss hunt

**D1: the reference model and state machines** (`tests/harness/{model,invariants}.py`,
`tests/adversarial/model/**`).

- **Self-check first.** The model must reproduce the examples INTERNALS documents before
  it is trusted: Rome/Berlin, project shadow, collapse, and the stats table.
- **Operations.** A Hypothesis `RuleBasedStateMachine` drives:
  - `remember`, with `valid_from`, `valid_to`, `close`, `expires_at` and `replaces`;
  - `delete`, `forget`, `erase` and `erase_expired`;
  - reads through `get_all`, `history` and `search` at many `valid_at`/`known_at`/`as_of`
    pairs and scopes.

  The nightly run adds `supersede`, disputes, polarity, links, two handles, reopen and an
  encrypted store.
- **Checked after every step:**
  - no step closes both clocks on a row;
  - rows disappear only through erasure;
  - reads match the model;
  - the belief floor never lifts;
  - a writer reads its own writes, and the narrowest scope wins;
  - an erased claim is gone from every read;
  - nothing leaks across users;
  - receipts and `stats()` are correct;
  - the offline configuration makes zero model calls;
  - an undeclared predicate never supersedes;
  - `search(k)` returns only visible rows.
- **What the model cannot check** (graph traversal and ranking quality among them) is
  listed in its docstring.
- **When a check fails,** the state machine prints a replayable op program.
  `model.replay(ops)` turns it into a fast regression test.

**D2: concurrency and crashes** (`tests/harness/{crash,crash_child}.py`,
`tests/adversarial/concurrency/**`).

| Scenario | What it checks | Tier |
|---|---|---|
| Two handles on one file | Commits are visible through the other handle, including the vector leg | fast |
| Reconcile race | The write is paused at the slot lookup, because nothing issues `BEGIN IMMEDIATE` | fast |
| Lost closure | `delete()` reads and then writes outside one transaction | fast |
| Reads while another writer holds the lock | Reads return the last commit without blocking | fast |
| N threads on one store | The invariants hold at the end | fast (small), nightly |
| kill -9 at named code points | A child process is paused at a named point and then killed (points listed below) | fast (6 points), nightly (all 10) |
| 200 random kills of a real server | The same crash checks | nightly |
| WAL recovery | Uncommitted work is gone and committed work is present | fast, nightly |
| Two real servers on one database file | No duplicates, no slot with two live values, no acknowledged write lost | fast, nightly |
| Server plus hook daemon | Both see each other's writes | nightly |
| Lock timeout | The tool returns an error, and the server stays up | nightly |
| Full disk | A loud failure, then a clean reopen. `RLIMIT_FSIZE` nightly; a macOS RAM disk in the local tier | nightly, local |
| Damaged `.vecs` or `<db>.embedder.json` | The file is rebuilt, or the damage is detected | fast, nightly |

The named kill points:

1. after the episode is written;
2. after the claim is written;
3. after the vector is written and before commit;
4. inside erase, between the audit row and the delete;
5. while `.vecs` grows;
6. while the fingerprint is written;
7. between two migrations;
8. inside `encrypt_store`;
9. inside `add_document`.

**After a crash**, the store is reopened in a fresh process, and these must hold:

- the integrity checks pass;
- the interrupted write is all-or-nothing;
- every acknowledged write is present;
- an erasure audit row exists exactly when its claim is gone;
- the next write succeeds.

No fault hooks are added to the library; the child process patches internals itself.

**Flake budget.**

- A fast-tier concurrency or crash test lands only after three things: 200 of 200 local
  runs pass, 100 of 100 runs under CPU load pass, and 3 full-matrix CI runs in a row are
  green.
- A test that fails in CI and does not reproduce moves to `quarantine/` with an issue. It
  comes back after 500 clean nightly repeats.

### Phase 2: The agent-facing deterministic tiers

These PRs run in parallel, after F3 and F4.

| PR | Content | Fast-tier time |
|---|---|---|
| A1 sessions | **Scripted agent workflows over the real pipe:**<br>• open a session;<br>• recall on each prompt;<br>• remember or add;<br>• the three kinds of correction;<br>• time travel;<br>• documents;<br>• bulk end or forget with a confirm token;<br>• the graph tools, profile, expiry and read-only mode.<br><br>**The tool surface:**<br>• an in-process oracle over every combination of switches that changes the tool list;<br>• over the real pipe, a 12-run orthogonal array (fast), a 3-way covering array (nightly), and every combination (weekly);<br>• the three protocol versions;<br>• read-only cloud credentials. | about 25 s |
| A2 hooks | **Conformance for all five hosts:**<br>• the reply envelope each host expects;<br>• exit code 0 for hostile payloads;<br>• the four outcomes stay distinct;<br>• deduplication;<br>• the daemon (socket 0600 inside a 0700 directory);<br>• `generate.py` output;<br>• the timeouts: 20 s start, 10 s recall with optional work stopped at 7.5 s, 5 s approve, 180 s capture;<br>• the approve list equals the server's read-only tools (B3). | about 20 s |
| A3 fuzz | **Protocol and validator fuzzing:**<br>• raw invalid UTF-8 and 20 MB lines;<br>• 500 pipelined requests with shuffled ids;<br>• odd id types and batches;<br>• deep nesting (B4);<br>• NaN and ±Infinity;<br>• the string `"false"` for booleans;<br>• lone surrogates.<br><br>**Properties:** every request gets one reply line, a refused call changes nothing, and the server always answers `ping` afterwards. | about 12 s |
| A4 security | • Prompt injection round-trips through every read tool and both reading hooks.<br>• Scope isolation between two processes on one database file. The reply about an id that exists elsewhere equals the reply about an id that never existed.<br>• An SSRF matrix.<br>• Encryption tampering.<br>• Redaction.<br>• Confirm-token forgery, replay and expiry.<br>• A sentinel API key must never appear in `repr`, CLI output, stderr, logs or argv.<br>• POSIX file modes.<br><br>Findings go through `SECURITY.md`. | about 15 s |
| A5 parity | The same operations through sync `Memvara`, `AsyncMemvara`, and sync and async `RemoteMemvara` against `FakeV1`. Then the MCP text from the in-process server, from stdio in local mode, and from stdio in cloud mode. | about 6 s |
| A6 upgrade | Committed stores, at most 256 KB each: one per distinct schema version since v0.1.0, each with a golden dump. Each must:<br>• open and migrate;<br>• match its dump;<br>• reopen idempotently.<br><br>A store from a newer version, and an embedder mismatch, are both refused cleanly.<br><br>**Nightly:** the stores are rebuilt from tags and wheels, and a kill -9 lands during migration. | about 5 s |
| A7 model faults | A scripted fake LLM sends:<br>• malformed output;<br>• invented predicates;<br>• 10,000 claims;<br>• timeouts and 429s;<br>• proposals to retire or erase;<br>• runaway tool loops.<br><br>**What must hold:**<br>• the episode is always stored;<br>• nothing is retired or erased because of model output;<br>• a failed read stage serves the plain read byte for byte;<br>• model call counts match INTERNALS. | about 5 s |
| A8 docs and frameworks | **Fast:**<br>• `--help` matches the config;<br>• the tool descriptions and `SKILL.md` name only tools and arguments that exist;<br>• defaults match the schemas;<br>• known drift lands as strict xfails.<br><br>**Nightly:** one virtual environment per framework (LangChain, LlamaIndex, CrewAI, LangGraph, mem0), at both the pinned floor and the latest release. | about 2 s |

Phases 1 and 2 together add about 2 minutes to the fast tier on Linux and macOS. Windows
runs a smaller subset, which also takes about 2 minutes. The budget is at most 3 minutes
on each CI job, measured on the Windows job. If the fast tier outgrows it,
`tests/adversarial` moves to its own parallel job on each OS.

### Phase 3: Live tiers (the `local` tier)

These run on the maintainer's Mac and in parallel with Phase 2.

**L1: sandbox and the Claude driver** (`tests/live/{sandbox,agents/claude,judge}.py`).

- **The sandbox.** Each attempt gets a scratch HOME outside every repository. It holds a
  copy of the plugin taken from this checkout, and a `.claude.json` that points at a local
  stdio server with a throwaway database and `MEMVARA_LLM=none`.
- **The environment.** A dedicated virtual environment is used, and the process
  environment is built from an allowlist.
- **An isolation canary.** It hashes the operator's real memvara and Claude Code settings
  and store before and after every run. Any change is a top-priority finding.
- **The invocation:**

  ```
  claude -p --model sonnet --plugin-dir … --output-format stream-json --verbose --include-hook-events
  ```

  It never uses `--no-session-persistence` or `--bare`, because capture reads the
  transcript file.
- **Probes that run first:**
  - the plugin loads;
  - the server connects;
  - the resolved model id is captured;
  - asynchronous capture survives the process exiting. If it does not, capture is made
    synchronous for the tests, labelled as such, and filed as a product question.

**L2: scenarios and grading** (`tests/scenarios/agent/*`, `tests/live/grading.py`).

- **30 scenarios:**
  - preference learning;
  - the three corrections;
  - flip-flop and restatement;
  - time travel;
  - project and worktree isolation;
  - documents;
  - poisoning: a pasted log, a README telling the agent to store something, echoed
    recall lines, quoted extractor rules, and forget-matching.
- **Grading, in order:**
  1. the store gold;
  2. the answer gold, using `benchmarks/agent_memory/normalization.py`;
  3. workspace checks;
  4. an LLM judge, for rubric items only. It is calibrated weekly against 20 hand-labelled
     answers, and below 90% agreement its verdicts are marked unreliable.
- **Negative control.** Every scenario must fail when memvara is off.
- **Flakes.** A failure is re-run twice, and the majority result counts.
- **Pass rates** are 14-night rolling figures with Wilson intervals.

**L3: hosted** (`tests/live/{stack,provision,prod_smoke}.py`, `compose.test.yaml`).

- **The stack.**
  - It runs from a private clone of memvara-cloud, detached at `origin/main`, and records
    that commit's SHA.
  - Each run gets a unique Compose project name and retagged images, so it never
    overwrites the operator's own.
  - Postgres is not published, and the API listens on a free 127.0.0.1 port.
- **Health checks,** in order:
  1. Compose's wait;
  2. `/v1/health`, which must report the same core version;
  3. the OAuth metadata;
  4. an MCP `initialize`.
- **Teardown** runs `down -v`, and a janitor removes projects and images left by earlier
  runs.
- **Suites:**
  - `RemoteMemvara` against the local gold;
  - the cloud-mode server;
  - the `/mcp` auth and session rules;
  - the npm bridge, including `oauth.js`;
  - the hooks in hosted mode.

  The destructive suites cover quota, 429 handling, cross-tenant reads, and erase and
  purge. They refuse to run against any origin other than the local stack.
- **The production smoke** makes at most 40 calls, with a dedicated key loaded through
  `demo/hosted.py::load_demo_credential`. It writes at most one claim and one episode, and
  ends them in the same run.

**L4: the red team** (`tests/live/redteam/*`). Its output goes to
`tests/adversarial/regressions/`.

- **The attacker** runs as `claude -p --model opus --restricted`, with no memvara plugin.
- **What it can touch.** It acts only through `attack_server.py`, an MCP server we write,
  against a throwaway store. Its tools are `session`, `api`, `mcp`, `rest` (local stack
  only), `inspect` and `check`.
- **Budget.** Each night covers two rotating attack classes, with 150 tool calls and 10
  minutes.
- **What happens to a candidate break:**
  1. It must fail on 3 out of 3 deterministic replays.
  2. It is reduced to a minimal trace.
  3. It is fingerprinted and deduplicated.
  4. It is emitted as `test_adv_rt_<fp>.py` with a strict xfail.
- **Security-class breaks** follow `SECURITY.md` instead.

**L5: transcript replay** (`tests/live/replay.py`).

- **Input:** the last 14 days of local Claude Code transcripts, sampled
  deterministically.
- **Two arms:** the fast path over up to 2,000 turns, and the hook's own extractor over 40
  turns.
- **What it measures:**
  - claims per turn;
  - reconcile outcomes;
  - gate drops, by reason and by script;
  - how often recall injects anything;
  - a "later hit" proxy;
  - hook latency.
- **Privacy:**
  - It never runs in CI.
  - Its temporary directories are 0700.
  - The report writer rejects any text.
  - It never files issues.

**L6: other hosts, weekly** (`tests/live/agents/{codex,cursor,opencode,copilot}.py`).

- Each host runs headless under a persistent test profile.
- A capability a host lacks is recorded as N/A.
- **On every host:** the hooks and the tools must write to the same store.

### Phase 4: Long-horizon

**S1: soak and performance** (`bench/soak.py`, `bench/perf_budget.py`).

The soak runs 10,000 seeded turns nightly and 100,000 weekly. Each silent failure mode
has a detector:

| Failure mode | Fails when |
|---|---|
| Predicate explosion | Distinct predicates exceed 1.1 × the vocabulary |
| Recency not refreshed | The median rank correlation is ≤ 0 |
| Flip-flop row growth | A single-valued slot has more than 1 live claim, or `merged` stays at 0 |
| Salience over relevance | The relevant claim ranks first less than 95% of the time |
| Script bias in the gate | Tracked only: a script's rate below 0.8 × the Latin rate |
| A retraction that retires nothing | It happens at all |
| Redaction drift | The ratio falls below 0.99 |
| Store growth | The relative rule below triggers |

**How performance is measured:**

- **Store sizes:** 1,000, 10,000 and 100,000 claims.
- **Cold:** the first call in 30 fresh processes. **Warm:** 200 calls.
- **Reported:** p50, p90, p95 and p99, with a bootstrap interval, for recall, search,
  remember and both hooks.
- **Recorded with each run:** the machine it ran on.
- **An invalid night:** a run on battery or under load is marked invalid, not failed.

**The budget decision rule.** It is fixed before any number is measured.

1. **Hard ceilings from the hook contract:**
   - the recall hook's p95 must be ≤ 7.5 s at every size, cold and warm;
   - its maximum must be ≤ 10 s;
   - session start's p95 must be ≤ 20 s.
2. **Library budgets.** After 14 valid nights, each budget is set to 1.5 × the median
   p95, rounded up to the next 1-2-5 step. The budgets are committed once, together with
   the machine fingerprint.
3. **A regression** needs all three of these:
   - p95 is more than 1.20 × the rolling 7-night median;
   - the increase is more than max(2 ms, 3 × MAD);
   - an immediate re-measure reproduces it.

**S2: mutation testing** (`bench/mutation.py`).

- **Tool:** mutmut 3. If it would need more than 10 exemptions, cosmic-ray replaces it.
- **Where it runs:** in a throwaway clone. It starts with a calibration run on
  `write/reconcile.py`.
- **Schedule:** a full baseline at weekends. Each night covers the changed functions plus
  a rotating seventh of the rest, capped at 15 minutes.
- **Floor:** 80% per module across `core.py`, `write/`, `store/` and `retrieve/`, with no
  drop of more than 2 points in a week.
- **Equivalent mutants** are listed in `mutation_equivalents.toml`, each with a reason.

### Phase 5: Nightly orchestration (`scripts/nightly/`)

**How it is scheduled.** A Desktop scheduled task runs the nightly Claude Code session. A
launchd watchdog, which uses no model, writes a **DID NOT RUN** report and sends a
notification when a night is missed, for example because the Mac slept, the app was
closed or a login expired.

**Where it runs.** Each run uses a clean worktree of `origin/main` and its own virtual
environment. Output goes to `local/nightly/<date>/` in the main checkout.

**Steps**, each with its own time cap:

1. preflight: resolve the models, check the logins, take the canary snapshot, write the
   heartbeat;
2. regressions;
3. agents;
4. red team;
5. hosted;
6. production smoke;
7. performance;
8. soak;
9. incremental mutation;
10. replay.

The target is about 90 minutes of wall time. The weekly run adds the other hosts, the
100,000-turn soak, the full mutation baseline, and every switch combination.

**For each new confirmed fingerprint**, the run files one issue labelled `nightly-break`,
with a hidden fingerprint marker. It then pushes a branch with the strict-xfail test, opens
a draft PR, and has the PR code-reviewed. Security-class breaks go to a private draft
advisory instead. Nothing is merged.

**Notifications** go out only for:

- a new break;
- an isolation breach;
- a dependency that has been down for two nights in a row.

### Phase 6: Fixes

Fixes run continuously, data loss first.

- Each confirmed bug gets its own fix PR, which removes its strict xfail.
- The order is:
  1. the privately reported finding (through its advisory), then B2;
  2. B4;
  3. B5 and B3;
  4. whatever the suite finds next.
- The B2 fix carries the design change: an issue first, then rules (a) and (b), then a
  rewrite of the reasoning in `types.py` and INTERNALS.

## First targets

These are predicted from reading the code and have not been confirmed. The suite writes a
failing test for each one first. Only a test that fails for the predicted reason becomes
an issue. Security-class predictions are tracked privately and are not listed here.

| Area | Prediction |
|---|---|
| Concurrency | A reconcile race across processes creates duplicate claims, because nothing takes the write lock before the lookup. |
| Concurrency | `delete()` loses a concurrent closure. |
| Write path | Restating a value with an earlier `valid_from` loses the earlier start. This needs a ruling if it is confirmed. |
| Validator | NaN passes the confidence bounds. |
| Validator | Error messages echo the whole invalid value, with no length cap. |
| Confirm tokens | A non-ASCII confirm token raises `TypeError` instead of being refused. |
| Hooks | On hosts with no status line, "not configured" and "no matching memories" cannot be told apart. |
| Hooks | The `_` approve separator can never recover a tool name. |
| Hooks | Codex's TOML client config is parsed as JSON. |
| Hooks | Capture returns without a log line on three paths. |
| Protocol | Invalid UTF-8 on stdin kills the line loop. On Windows, stdin is decoded with the locale code page. |
| Upgrade | A store from a newer version exits with a traceback. |
| Upgrade | A damaged `<db>.embedder.json` is never rewritten, so a later swap to another embedder of the same width goes unnoticed. |

## Documented behaviour the suite must not report as bugs

These come from `docs/LIMITATIONS.md`, the "Deliberately deferred" list in
`docs/ROADMAP.md`, INTERNALS, and the out-of-scope list in `SECURITY.md`. A test that
meets one asserts the documented behaviour and cites where it is documented.

- The hashing embedder is Latin-only, and the fast path knows only English forms.
- With no model configured, `add()` stores only rule-matched facts.
- Same-named entities fold together.
- Batches are refused.
- Project shadowing applies only to present-tense reads of single-valued predicates.
- The documented limits of encryption at rest.
- A NAT64 prefix nobody configured cannot be detected.
- The redactor misses what its docstring lists.
- Erasure does not wipe old SQLite pages or the WAL.
- There is no retraction prompt (#209).
- Agentic extraction and extraction chunks are off by default.

## Done criteria

1. **The checklist baseline is empty:**
   - every tool and switch pair is driven over the real pipe;
   - every invariant, environment variable, hook-and-host pair and silent failure mode
     has a test;
   - every known bug has an issue and a strict xfail, or an advisory.
2. **Mutation score** is at least 80% per target module, with no drop of more than 2
   points in a week.
3. **Agent pass rate** is at least 90% over 14 nights. It is tracked, not gating.
4. **The red team** goes 4 weeks in a row without a new confirmed fingerprint, at a fixed
   budget.
5. **Performance:**
   - the hard ceilings hold;
   - the derived budgets are committed;
   - the regression rule is live.
6. **The flake budget is met:**
   - the land-time repeat runs pass;
   - nightly flakiness is at most 0.5% per layer over 14 nights;
   - every quarantined test has an issue.
7. **The skip ledger is enforced**, so there are no unexplained skips.
8. **Every surface gives equivalent results** on the shared scenarios.
9. **Every released schema version upgrades without loss.**
10. **The docs contract holds:** the help text, the tool descriptions and `SKILL.md` all
    match the code.
11. **The fast tier stays within budget:** it adds at most 3 minutes to each CI job, and
    it is green on Python 3.10–3.13 on Linux, macOS and Windows.
12. **The isolation canary** has been clean on every nightly run.

## Operator setup

These steps are done once, by a person:

1. Create a headless Claude Code login that works from an isolated HOME, with
   `claude setup-token`.
2. Log in once to test profiles for the other agent CLIs.
3. Create a dedicated test tenant and API key on app.memvara.dev for the production
   smoke.
4. Approve installing the scheduled task and the launchd watchdog.

## Verification

**Every PR**, run from its own worktree:

```
PYTHONPATH=$PWD python3 -m pytest -q -p no:cacheprovider tests/adversarial --durations=20
PYTHONPATH=$PWD python3 -m pytest -q -p no:cacheprovider --runxfail -k <its xfail ids>
PYTHONPATH=$PWD python3 -m pytest -q -p no:cacheprovider tests/adversarial --tier nightly
PYTHONPATH=$PWD COVERAGE_FILE=$PWD/local/cov/.coverage.<pr> python3 -m coverage run -m pytest -q
PYTHONPATH=$PWD COVERAGE_FILE=$PWD/local/cov/.coverage.<pr> python3 -m coverage report
python3 -m mypy -p memvara && python3 -m mypy tests/harness --ignore-missing-imports
```

Each PR body reports:

- the passed, xfailed and skipped counts;
- coverage, which must be 100%;
- the fast-tier wall time.

The skip count stays the same unless the skip ledger explains the change.

**Proofs specific to some PRs:**

- **F2:** under `--runxfail`, each bug test fails for its stated reason.
- **D1:** catches three deliberate mutants:
  - drop the belief floor;
  - make `_is_after` non-strict;
  - make `close_out` set both clocks.
- **D2:** passes the 200-of-200 and 100-of-100 repeat loops.
- **F5:** a new feature switch fails the meta-test.
- **L1:** the probes pass on 3 runs.
- **L2:** every negative control fails.
- **L3:** two up and down cycles leave no volumes behind.
- **L4:** a bug planted in a scratch copy is found, confirmed, reduced and emitted.
- **S1:** each detector fires on an injected fault.

**End to end:** one supervised nightly dry run. It covers the report, the DID NOT RUN
path, the canary, and issue and PR creation against a test label.
