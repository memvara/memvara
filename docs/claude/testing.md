# The adversarial test suite

This page is for anyone adding to or running the adversarial suite. The suite tries to break memvara, and it uses memvara the way an agent does: through the real MCP server process, the real hook scripts and a real store, rather than by calling library functions directly. The design and the reasons behind it are in `docs/superpowers/specs/2026-09-25-adversarial-test-suite-design.md`.

The suite lives in two places:

- `tests/harness/` is support code, and it holds no tests.
- `tests/adversarial/` holds the tests. Every file there is named `test_adv_*.py`.

## Child processes

**Every child process the suite starts gets its environment from `harness.env.child_env(home)`.** That function does five things:

- It refuses the real home directory.
- It removes every `MEMVARA_`, `ANTHROPIC_`, `OPENAI_` and Claude Code variable.
- It points `PYTHONPATH` at this checkout.
- It selects the hashing embedder, with encryption and project detection off.
- It stops the hooks from starting their background daemon.

A test that builds its own environment for a child process is the kind of test that once overwrote a developer's real credentials, so do not write one.

**The suite must import the checkout it lives in.** In a git worktree, a stale editable install can make `import memvara` load another copy. `test_adv_env.py` then fails first and explains the fix. There are two ways to fix it:

- run with `PYTHONPATH` set to the checkout;
- create a virtual environment inside the worktree (the `local/` directory is ignored by git) and install the checkout into it in editable mode.

## Tiers

A test's tier comes from the folder its file lives in. You do not mark it.

| Folder | Tier | Who runs it |
|---|---|---|
| anything else | fast | every PR, in CI |
| `nightly/` | nightly | the nightly run on the maintainer's Mac |
| `weekly/` | weekly | the weekly run |
| `local/`, and all of `tests/live/` | local | only on a machine with the logins, Docker and transcripts it needs |
| `quarantine/` | quarantine | nobody by default; each test there has an issue |

`--tier` chooses what a run collects:

| Flag | Collects |
|---|---|
| none | fast |
| `--tier nightly` | fast and nightly |
| `--tier weekly` | fast, nightly and weekly |
| `--tier local` | only local |
| `--tier quarantine` | only quarantine |

A tier that a run does not select is left out when pytest collects, so it is never imported and never reported as skipped. A file you name on the command line is always collected, whatever its folder. A tier folder you name is not: run `pytest tests/adversarial/nightly --tier nightly`, because without `--tier` the folder is left out and nothing runs.

The `--tier` option, and the filter that leaves the other tiers out, are in the `conftest.py` at the repository root. pytest reads that file on every run, so `--tier` works whatever paths a run is given, `pytest memvara --tier nightly` included. The doctests in `memvara/` are fast tests, so a local or quarantine run leaves them out. The root file loads `tests/harness/tiers.py` from its path, because it is read before pytest puts `tests/` on the import path.

**The outermost tier folder decides.** Do not put one tier folder inside another, or anywhere under `tests/live`; `test_adv_tiers.py` refuses both. Every run prints its tier and the tier folders it left out, for example `tier fast; left out 3 tier folders: ...`, so a folder that happens to share a tier's name cannot drop out of the run unnoticed.

**Every folder of the suite needs an `__init__.py`,** not only the tier folders. Without one at every level, pytest imports the modules below that folder under names of their own: two test files with the same name collide, and a tier folder's tests fail to import. `test_adv_tiers.py` checks every folder under `tests/adversarial` and `tests/live`.

The files named `test_adv_*_tier_guard.py` fail if their folder is ever collected by a tier that should have left it out. The ordinary fast run is therefore the proof that the tiers work.

## Skips

**A skip needs a rule.** This applies to every test in the repository, not only the adversarial suite, because `tests/conftest.py` registers the ledger for every run that collects tests under `tests/`. A run given only `memvara/` has no ledger, which is harmless while no doctest skips. Every skip reason must match a rule in `tests/harness/skips.py`, and each rule says why that skip hides no failure. A skip with no matching rule fails the whole run, and the run lists the test and its reason.

A rule can be bound to platforms (`platforms=("win32",)`) or to Python versions (`python_below=(3, 11)`, `python_from=(3, 11)`). Outside those, the reason counts as unexplained, so a test that starts skipping where it should run turns the run red.

The ledger exists because most summaries show a skip as green, so a test that stops running for a new reason looks exactly like one that passes.

An expected failure (xfail) is not a skip, and the ledger ignores it.

## Property-based tests

Hypothesis runs under the profile of the selected tier.

| Tier | Examples | Behaviour |
|---|---|---|
| fast | 30 per test, 25 steps per state machine | Derandomized, with no example database, so a PR run gives the same answer every time |
| nightly | 3,000 | Keeps the examples it finds in `~/.cache/memvara-adversarial/hypothesis`, so a failure found one night is tried first the next night |
| weekly | 20,000 | Same database as nightly |

When a property test fails, Hypothesis prints a reproduction blob. Put it in a `@reproduce_failure` decorator to replay the exact case.

## The MCP server, in its own process

`harness.stdio.McpProcess` starts `python -m memvara.server` as a child process and speaks newline-delimited JSON-RPC to it, the way an agent's client does. Nothing about the server is faked: it imports this checkout, opens a real SQLite store, and reads its configuration from the environment.

To use it:

- **In a test,** use the `mcp` fixture. Each `mcp()` call opens a new store file in the test's temporary directory. `mcp(path)` opens a store you name, so passing the same path twice makes two servers share one store.
- **Scope and switches.** Pass `features={"documents": False}` to switch a feature off, `read_only=True` for a read-only server, and `scope={"session": "s1"}` to bind a scope field.
- **Calling tools.** `call(name, **arguments)` returns the tool's text and its error flag.
- **Raw input.** `send_raw` and `recv` send and read arbitrary lines, for protocol tests.

Failures are loud and quick:

- A server that exits raises `McpProcessError`, with its exit code and the tail of its stderr.
- A server that writes nothing within the timeout raises the same error, instead of hanging the suite.
- A server that stops reading its input does too: a write that does not finish within the timeout kills the server and raises the error.
- A line from the server that is not JSON raises the error with the line and the tail of stderr.

## Hooks

`harness.hooks.HookRunner` runs `plugin/hooks/run.py <hook> --host <host>` in a child process, with the stdin payload that host sends.

- **In a test,** use the `hook_runner` fixture: `hook_runner("claude").run("approve", tool_name="mcp__memvara__memory_search")`.
- **Giving the hooks a store.** Pass `server_env={"MEMVARA_DB": ..., "MEMVARA_USER": ...}` and the runner writes the host's client config, which is where the hooks look for the store. Without it, the hooks report "not configured". The runner writes the config in the host's own format: JSON with an `mcpServers` object, or for Codex, TOML with an `[mcp_servers.memvara]` table in `~/.codex/config.toml`. `env={"MEMVARA_DB": ...}` names a store another way, through the hook process's own environment, which wins over any client config.
- **What a run returns:** the exit code, the parsed reply, the elapsed time, and the lines the run added to each log in `~/.memvara/.hooks/`, without their timestamps: `result.log("recall")` gives the new lines of `recall.log`. A hook that runs past its host's time limit raises `HookTimeout`, with what it had printed so far.
- **`capture` needs stub agent CLIs.** It starts an agent CLI to extract facts, which would reach the network and spend money, so `run("capture")` raises `NotImplementedError` unless the runner was given `stubs=`, such as `FakeClis` (see "Fakes" below), which go first on `PATH`.
- **No real agent CLI is ever reachable.** The runner leaves out of `PATH` every directory that holds a program any host's capture could start (`claude`, `codex`, `cursor-agent`, `copilot`, `opencode`), whether or not the test gave it stubs. The fakes stand in for `claude` and `codex` only, so a real `cursor-agent` further along `PATH` would otherwise still be found.
- **A detached capture is waited for.** Codex, Copilot and Cursor hand capture to a child in a new session, and the hook returns at once. `run` waits for that child within the same limit, so the result's logs hold what the capture did, and `detached_pid` names it. `elapsed` is still what the host waited. With `wait_detached=False` it does not wait, and `close()` kills the child with everything it started.
- **The recall daemon is off unless a test asks for it.** `daemon=True` lets the recall hook start its background daemon, which `child_env` forbids because the daemon outlives the hook. `wait_for_daemon()` returns its socket and its pid once it accepts a connection, and `close()` kills it. The pid comes from the kernel, through the socket (`socket_peer_pid`), because every daemon runs the same command. The socket lives under the runner's home, and macOS refuses a unix socket path longer than 104 bytes, so a home for a daemon test needs a short path. Windows has no unix sockets, so these tests skip there.
- **Non-JSON output fails the test.** A hook that prints something other than JSON raises `HookOutputError`, because on a real client that output would desynchronise the conversation. So do bytes that are not UTF-8.

## Stores in the test process

`harness.stores.memory()` gives an in-memory store and `harness.stores.file(path)` a SQLite file. Both use the hashing embedder and no model, as every other test here does.

A server started on the same file with the child environment opens it in the same vector space. So a test can write through the library and then read through the server, or the other way round.

## Known bugs and security findings

**A bug the suite finds lands at once as a failing test marked `xfail(strict=True)`.** The marker cites a GitHub issue, and the fix follows in its own PR.

- **Strict mode keeps the marker honest.** When the fix lands, the test starts passing and strict mode fails the run until the fix PR removes the marker.
- **Registering a bug.** `tests/harness/known_bugs.py` lists each open bug, and `known_bugs.xfail("B2")` builds its marker.
- **A marker absorbs only its own bug.** The test raises `known_bugs.Reproduced` after it has seen that bug's exact symptom, and the marker accepts nothing else. Any other failure in the same test, even an exception of the same type, fails the run, so a new bug cannot hide behind a known one.
- **Nothing is weakened.** Never skip, delete or weaken a test to make the run green.

**A finding that falls under the in-scope list in `SECURITY.md` never goes into a public issue or a public test.** It goes to a private draft advisory on GitHub, and its failing test lands together with its fix.

## The reference model

`tests/harness/model.py` is a small, separate model of what a store must hold: a dictionary of rows over both clocks, with each rule copied from the code that states it (`state_predicate`, `Reconciler.apply`, `close_out`, and `Memvara`'s `remember`, `forget`, `delete`, `erase` and `erase_expired`). Which predicates hold one value, and which are declared at all, it reads from memvara's own predicate registry. Before it judges the store, it reproduces the examples INTERNALS documents: the stats table whose counts do not sum, the audit view that readmits a future-dated row, the Rome and Berlin gap, and a collapse that is true at no instant.

`tests/harness/drive.py` runs the same operations on a real store and on the model. The store is in memory for the hand-written cases and in a file for the state machine. After every operation, `Pair.apply` compares every row and every receipt field the model predicts. It also compares these reads: `get_all` in every combination of states, in the present and at six pairs of instants (one of them after every scheduled value begins), and once with `as_of`; `history` at the same pairs; `search` in the present for every value, and at each pair for one; and `count`, `get` and `stats`. A write the store refuses, such as an end at or before the start, must be one the model refuses too.

- **Wall-clock stamps are checked, not predicted.** A stamp the store takes from the clock, such as `recorded_at` when none is given or the instant of a retirement, must fall inside the window the operation ran in. It is then copied into the model.
- **A difference names the operation.** `drive.replay(ops)` raises `ModelDivergence` at the first operation where the two disagree, and `drive.format_program(ops)` prints a program that `replay` accepts. When the state machine diverges, its failure carries the whole program so far as `drive.replay([...])`, among the notes Hypothesis prints, so the failure can become a fixed test without rebuilding it by hand.
- **Some checks do not use the model.** They read only the store, so they still hold if the model and the store were ever wrong in the same way. No read may return a row recorded after the read's `known_at`, or a row that belongs to another user. No search may return a row twice or more than `k` rows. A writer must be able to read back a live row it has just written. `get` must return nothing for an erased row. A write under a predicate nobody declared must close no row. `test_adv_drive_checks.py` shows each of these checks catching its fault.
- **The state machine** (`tests/adversarial/model/test_adv_model_machine.py`) draws operations at random from small pools, so they often collide, and a second `remember` rule writes only `lives_in` at three instants: two in the past, so that two values often begin at the same moment, and one in the future, so that the slot often holds a value stored to begin later. `forget` runs with either closure, so ending a slot is driven too, including the clamp of an ending onto the start of a value that has not begun. Run against a `forget()` without that clamp, the machine at the nightly tier's size reported a divergence within 32 seconds in each of three tries. The fast tier's 30 short runs may miss the clamp, because Hypothesis switches a random subset of the rules off in each run and seeds the fast tier from the machine's source, so the hand-written model case that ends such a slot is what checks the clamp there. It also checks that no step closes both clocks of an existing row, that a closed clock never moves later and an ending is never cleared, and that rows disappear from the store only through erasure. Each run keeps its store in a file of its own, and when the run ends, that file must pass the integrity checks described at the end of this section. The machine makes 30 short runs in the fast tier, which take about 13 seconds on a laptop, and at most 300 runs in the nightly and weekly tiers; the nightly tier's 300 runs of 100 steps took 10 minutes 10 seconds on a laptop with nothing else running.
- **Known bugs are steered around.** An operation that would trigger an open bug is skipped, so the machine keeps finding new bugs instead of the known ones. It steers around none today. The last was #275, a retraction dated in the future whose tombstone ended before it began, which the machine found and which is now fixed; `STEERED` in the machine's module lists the bugs it avoids. The test prints how many steps it ran and how many it skipped for each bug (`pytest -s` shows the line). It fails if skips pass a tenth of all steps, and `STEERED` may name only a bug still in the registry, so a fixed bug's detour cannot outlive it.
- **It catches the mistakes it is meant to catch.** It was run against four deliberate faults, each in a scratch copy of the code, and the fast tier failed on every one. Dropping the belief floor from `state_predicate`, and dropping the parentheses in `_either` that keep the floor in force, were each caught at once by the belief-floor check. Making `_is_after` non-strict, and making `close_out` close both clocks, were each caught by the state machine.

What the model leaves out is listed in its module docstring. This first stage has one scope per user; projects, agents and sessions come with the #266 fix, which changes how those scopes interact.

`tests/harness/invariants.py` checks a closed store file for the damage a crash or a bad write leaves: SQLite's and FTS5's own integrity checks, a claim missing from the text index or the embeddings, a vector slot both used and free, a provenance edge that names a claim or an episode that is gone, and an erased claim that still exists. The state machine runs these checks on its store file at the end of every run, and the crash tests run them on the store a killed process leaves behind. With `key=`, they open an encrypted store through `sqlcipher3`.

## Concurrency and crashes

`tests/adversarial/concurrency/` checks what happens when two handles or two processes use one store file at once, and when a process is killed in the middle of a write. The plan is `docs/superpowers/plans/2026-09-25-adversarial-crashes.md`.

- **The crash child.** `tests/harness/crash_child.py` is a script that a test runs as a child process. It reads one JSON program on its standard input: some setup operations, a named point, and one more operation. It prints `ACK` for each setup operation it finishes. It then patches memvara in its own process so that the last operation stops at the point, and prints `POINT <name>` when it gets there. From there it either waits to be killed, or, when the program holds it, waits for the line `go` and carries on. The library has no fault hooks: every pause is a patch the child applies to itself. `tests/harness/crash.py` starts the child with the suite's child environment, and reads its lines with a timeout, so a child that never reaches its point fails the test instead of hanging it.
- **What recovery must look like.** `crash.after_crash` opens the store again in the test's own process, which never had it open. The file must pass `check_store_integrity`. Every acknowledged claim must be present and found by `search`. An erased claim must have its erasure record and nothing else. The next write must finish within a second, which shows that the dead process left no lock behind. Each test then checks that the interrupted operation is either whole or absent.
- **The ten kill points.** The fast tier kills the child after an episode is written inside `add()`, after a claim is written and after its vector is written inside `remember()`, inside `erase()` between the audit row and the delete, between two schema migrations, and after `remember()` has returned. It also kills a child inside an uncommitted batch, and while a brand-new store is being created. The nightly tier adds the vector file growing, the embedder record being written, `encrypt_store` between its two renames, and `add_document` between storing the chunks and extracting claims. The plan's table says what each must leave behind.
- **Two processes in a fixed order.** A held child stops at a point while the test writes through its own handle, and then the child carries on. That fixes the interleaving, so the tests give the same answer on every run. This is how the tests check that a reader is never blocked by a writer that holds the lock, and that two handles read each other's commits at once, through the vector index as well.
- **Threads and real servers.** Four threads write through one handle in the fast tier, and sixteen nightly. Two real MCP servers share one store file and read each other's writes. The nightly tier also kills a real server 200 times at random moments, runs two servers at once from two threads, holds the write lock past SQLite's five-second busy timeout, and fills the disk by lowering the child's largest file size (POSIX only), which fails a write with "file too large". The local tier fills a real RAM disk instead, which fails with "no space left on device", as a full laptop disk does (macOS only, because it uses `hdiutil`).
- **What they found.** #280: a damaged embedder record lets a same-width embedder change go unnoticed. #281: two processes opening a new store at once can make one of them fail at startup. Both are pinned as strict expected failures.
- **The flake budget.** Before these tests landed, every fast test here passed 200 of 200 runs on a laptop, and 100 of 100 runs while four processes kept every core busy.

To run one point by hand, write a program and pipe it into the child. It stops at the point and waits to be killed:

```bash
echo '{"db": "/tmp/s.db", "user": "u1", "setup": [], "point": "after-claim", "action": ["remember", {"predicate": "lives_in", "object": "Berlin"}]}' | PYTHONPATH=$PWD python tests/harness/crash_child.py
```

## Scenarios and the scripted layer

A scenario describes a few sessions of a user talking to an agent that has memvara, and what must be true afterwards. It is one JSON file. The same format serves two layers. The scripted layer, described here, plays a fixed script for every turn on every pull request. The real-agent layer, which comes later, sends only the user's words to a real agent and grades it against the same kind of gold.

- `tests/scenarios/schema.json` defines the format. Every field has a description there.
- `tests/scenarios/scripted/` holds the scripted scenarios, one per file, each named after its `id`.
- `tests/adversarial/sessions/runner.py` checks, plays and grades them.

The `jsonschema` package is not a dependency, so `runner.py` carries a small validator for the keywords the schema uses. It refuses a keyword it does not implement, so the schema cannot state a rule that nothing checks. A second check covers what a schema cannot express:

- every gold id is unique;
- a known bug names a gold item and a registered bug;
- a placeholder is set by an earlier step;
- a mark that an `expires_at` uses is at least four seconds ahead, so the steps before the expiry cannot race it on a slow machine;
- a hook or tool a step uses is declared in `surfaces` and `requires`;
- a `forbidden` rule names a tool memvara has, and an answer gold item checks a turn that has a tool or hook step, because otherwise the check could never fail;
- the tier is fast, nightly or weekly, because a run that selects only the local or quarantine tier never collects the scenario tests, so a scenario in either would never run.

### How a scenario plays

The runner writes the `seed` through the library first. The seed stands for memory from conversations before this one. Then every session starts its own server process on the same store file, the way a client starts one per conversation, and plays its turns in order. A turn holds the user's words and a `script`: the steps a careful agent would take for that turn.

| Step | What it does |
|---|---|
| `{"tool": "memory_…", "args": {…}}` | Calls a tool on the session's server. The call must succeed, unless the step says `"expect_error": true`. |
| `{"hook": "session_start"}` | Runs one of the plugin's hooks against the same store, with the payload and reply shape of the `claude` host unless `host` names another one. The recall hook is given the turn's words as its prompt. |
| `{"op": "erase", "claim_id": "…"}` | Erases a claim through the library. No tool can erase a memory, so this stands for the operator doing it. It erases that claim and nothing else: the store is opened without the expiry sweep, so it never does the server's expiry work for it. |
| `{"mark": "name", "offset_seconds": 4}` | Records the instant now, plus the offset, under a name. |
| `{"wait_until": "name"}` | Sleeps until that instant has passed. |

A tool or hook step can `capture` part of its output with a regular expression, and a later step can use it. An argument that is exactly `{name}` is replaced by the captured text or the marked instant, and one that is exactly `{file:path}` by that workspace file's contents. Nothing else in an argument changes, so text with braces in it is safe.

`env` sets how the server starts: the user, the project, the feature switches, read-only mode and the protocol version. A session can override any of them except the user with its own `env`, which is how a scenario moves the user from one project to another. The user stays the same in every session, because store gold reads every claim at the scenario's user.

After the last session the store is read once more with expiry switched off, so the read neither erases an expired claim nor hides one. The store gold therefore sees exactly what the server left on disk.

### What gets checked

`tests/adversarial/sessions/test_adv_scenarios.py` plays each scenario once, and every test below reads that one play. When a scenario stops early, for example because a capture found nothing, each of its tests fails with the same message.

| Test | Passes when |
|---|---|
| `test_the_file_follows_the_format[<id>]` | The file matches the schema and passes the checks the schema cannot express. |
| `test_gold[<id>/<gold id>]` | That one gold item holds. |
| `test_the_script_ran_as_written[<id>]` | Every step succeeded, or failed where it said `expect_error`. |
| `test_no_forbidden_tool_was_called[<id>]` | No step called a tool the scenario forbids. |
| `test_the_gold_fails_without_memvara[<id>]` | At least one gold item fails for an agent with no memory. |

**Store gold** names a claim by its text, such as `user lives in Lisbon`, never by its id, and says which state it must be in: `live`, `ended`, `retired`, or `absent` for no claim with that text in any state. `count` asks for an exact number. `project` reads at another project than the scenario's own, or at user level when it is `null`.

**Answer gold** checks the answer to one turn: the turn its `turn` names, or the last one. In the scripted layer, the answer is every memory memvara showed the agent in that turn: each tool's text and each hook's injected context, in order, leaving out any read that found nothing and the session-start hook's first line, which names the scope the store is bound to rather than any memory. `must_contain` and `must_not_contain` compare whole words and ignore case and punctuation, using `phrase_in` from `benchmarks/agent_memory/normalization.py`, which is the benchmark's own normalization and token rule. They do not apply the length ceiling or the competitor check that the benchmark's `matches_value` adds for a short answer, because memvara's replies are long by design and a history reply names every value a slot has held. `must_not_match` is a regular expression, for checks about lines, such as stored text that must not start a line of its own. `abstain` passes when the answer is empty: every tool in the turn replied that it found nothing, and no hook injected any memory.

**A known bug** is attached to the one gold item it breaks, with the symptom it causes: `"known_bugs": {"<gold id>": {"bug": "B2", "symptom": {"states": ["ended"]}}}`. That item's test gets the bug's strict expected-failure marker. The test raises `known_bugs.Reproduced` only when the failure shows exactly that symptom: the same states for a store item, or the given words in the answer for an answer item. Any other failure fails the run.

**The negative control** plays the scenario for an agent with no memory: no seed, no server and no hook, so every answer is empty and the store holds nothing. At least one gold item must fail then. If none does, the gold cannot tell memvara working from memvara absent.

### Adding a scenario

1. Write `tests/scenarios/scripted/<id>.json`. Use made-up people and data, because this repository is public.
2. Run `pytest tests/adversarial/sessions -k <id>` and read every failure. A scenario mistake is fixed in the scenario. A failure that shows memvara doing the wrong thing is a bug, handled as "Known bugs and security findings" above describes.
3. Keep it deterministic and offline. A scenario that needs a model, the network or the capture hook belongs to the real-agent layer.

A read that finds nothing repeats its query in its reply. That reply adds nothing to the answer, so the words of a query can neither satisfy `must_contain` nor trip `must_not_contain`.

Each session starts a server, which takes about 0.2 seconds on a laptop and longer on Windows. The scripted layer's budget on the fast tier is about 25 seconds, so use as few sessions as the story allows.

## Fakes for the services a client talks to

`tests/harness/fakes/` holds test doubles for what a memvara client reaches over a network or starts as a program: the hosted REST API, the hosted MCP endpoint, an OpenAI-compatible model, and the agent CLIs that the capture hook runs. With them a test drives the real client code offline, with no login and no bill. Every fake listens on 127.0.0.1 only. Each one is checked in `tests/adversarial/fakes/` by driving it with the real client it stands in for, so a fake that drifts from what its client sends or reads fails its own tests first. Drift from memvara-cloud is not caught that way. `FakeV1` and `FakeHostedMcp` were written to match memvara-cloud at origin/main e8940be (2026-09-25), and nothing compares them with it automatically, so when the cloud changes they keep the old behaviour until somebody compares them again by hand.

**The three HTTP fakes share one mechanism,** in `fakes/_http.py`:

- **A test reaches a fake in one of three ways.** `transport()` is an `httpx.MockTransport` for an `httpx.Client`, and `async_transport()` is the same for an `httpx.AsyncClient`. `serve()` answers on 127.0.0.1 from a background thread and returns the base URL, for a child process or for a client that does not use httpx. `close()` stops a fake, and each fake is a context manager that closes itself.
- **Every request is recorded** in `fake.requests`, with its method, raw path, query, headers and body, the route it matched, and the status it was answered with.
- **A test can inject a fault into one route,** for every request or for the next `times`: `fail(route, status)` answers with that status and does nothing else, `delay(route, seconds)` waits and then answers, and `hang(route)` never answers. A route name the fake does not have is refused, so a typo cannot inject a fault that never fires.
- **A slow answer reaches the client the way a real server's would.** Over a socket, the client's own timeout fires. A mock transport has no network under it, so there a hang, or a delay at least as long as the request's read timeout, waits out that timeout and then raises `httpx.ReadTimeout`. One difference remains. Over a mock transport a request the client gave up on is never carried out, while over a socket it is carried out late, as on a real server. A test about a write that lands after its client gave up therefore uses `serve()`.
- **Closing a fake releases every request it is still holding,** however the request arrived, so a test that ends with a hang or a delay in progress does not wait it out. A released request is not carried out. Over a socket its connection is closed without an answer, and over a mock transport it raises `httpx.ReadTimeout`.

**`FakeV1` is the hosted `/v1` REST API.** It answers the 34 routes that `RemoteMemvara` and `AsyncRemoteMemvara` call, from a real local `Memvara` with the hashing embedder and no model. It writes each answer the way memvara-cloud's `rest/render.py` does, so the client's own hydration code reads it back. A self-test reads the two clients' source and fails when either one calls a route the fake does not serve.

- `fake.remote(user="alice")` and `fake.aremote(...)` return a client wired to the fake through a mock transport. For a real URL, pass `fake.serve()` as `base_url` with `api_key=fake.api_key`. That is also how to start an MCP server in cloud mode against it: `MEMVARA_MODE=cloud`, `MEMVARA_API_KEY` and `MEMVARA_SERVER_URL`.
- A route's name is its method and path template, such as `POST /v1/facts` or `GET /v1/memories/{id}`. `FakeV1.ROUTES` lists them.
- The credential is one key, bound to the whole tenant with the admin privilege, so a client may narrow to any user. A wrong key is a 401, an agent or a session named without a user is a 400, and `FakeV1(read_only=True)` refuses every write with a 403.
- A write retried with the same `Idempotency-Key`, method and path is carried out once, as on a deployment with one worker.
- What it leaves out: allowances and rate limits (inject a 402 or a 429 instead), legal holds, the audit trail, OAuth, and a document added by `url`, which it refuses because the suite runs offline. `POST /v1/maintenance/consolidate` runs the pass before it answers, where the cloud answers first.
- A request that names a project with the `Memvara-Project` header gets `Memvara-Project-Applied` back on its answer, but never on a refusal. The cloud also sends it on a refusal raised after it has resolved the project, such as a 404 for a missing memory. No client in this repository reads that header, so nothing depends on the difference yet.
- `/v1` does not carry a claim's `temporal_precision`, `object_kind`, `amount` or `unit`, so a claim read through the remote client has the default in each of them. A test that compares a local store with a remote one leaves those four out.

**`FakeHostedMcp` is the hosted `/mcp` endpoint** that the hooks' hosted client (`plugin/hooks/lib/hosted.py`) and the npm bridge reach. It answers with the real `MemvaraMCPServer` over a local store, bound to the credential's scope, which is the user `tester` unless the test names another. It checks what those clients send the way memvara-cloud's `rest/mcp.py` does. A request needs a bearer token, and without one it gets a 401 whose `WWW-Authenticate` header tells an MCP client where to sign in. `initialize` issues a session id, and every later request must carry one: none is a 400 and an unknown one a 404. A notification gets a 202 with no body, and a `memvara-project` header binds the project. The hooks use a hosted endpoint only when no local store is configured, so a hook run reaches the fake when its client config names no store and its environment sets `MEMVARA_API_KEY` to `fake.api_key` and `MEMVARA_SERVER_URL` to `fake.serve()`.

- `FakeHostedMcp(sse=True)` sends each reply as a server-sent event, which both clients must be able to read.
- `expire_sessions()` forgets every session, as a restarted deployment does, so a test can watch a client shake hands again. `fake.issued` lists every session id the fake has issued.
- A fault is keyed by JSON-RPC method, and a tool call by `tools/call <tool>`, so `fake.fail("tools/call memory_recall", 402)` refuses one tool and leaves the handshake alone.

**`FakeOpenAI` is an OpenAI-compatible chat-completions endpoint.** It answers each request with the next reply a test scripted, in order, and records every request. `add_reply`, `add_json` and `add_tool_calls` script a completion; `add_raw` scripts a body the client cannot use; `add_rate_limit` scripts a 429; and `add_hang` scripts no answer at all. A request that finds no reply left gets a 500 that says so, and `fake.pending` counts the replies not used yet. A `fail` or a `hang` injected on the route answers in place of the script without using up a reply, and a `delay` waits and then answers with the next reply. It serves chat completions and nothing else. Describing an image is a chat completion, so it is answered from the script like any other call. Transcribing audio uses a different endpoint, which the fake answers with a 404.

- **In the test process,** `OpenAILLM(client=fake.client())` talks to it. CI does not install the `openai` package, so `client()` is a small stand-in for the SDK's transport. It sends each call over HTTP the way the SDK would, and hands back the decoded JSON, which `OpenAILLM` reads as it reads the SDK's own objects. It does not retry, whereas the SDK retries a 429 or a timeout twice by default, so a test that wants a retry scripts it.
- **In a child process,** a server started with `MEMVARA_LLM=openai` reaches the fake through `OPENAI_BASE_URL` set to `fake.base_url`, with any `OPENAI_API_KEY`. That needs the `openai` package installed in the environment the child runs in.

**`FakeClis(directory)` writes fake `claude` and `codex` executables** for the capture hook, which mines a turn by starting one of them. Give a child process `PATH` set to `fakes.path()`, and it starts the fakes instead of the real CLIs. `fakes.script("claude", "reply text", CliReply(...))` sets what each run prints, one reply per run, starting with the next run, and `fakes.calls("claude")` lists every run's arguments and stdin, including the runs made under an earlier script.

- Each fake prints its reply in the format its arguments ask for: the single JSON object of `claude --output-format json`, the event stream of `claude --output-format stream-json` that the agentic capture run reads, or the event stream of `codex exec --json`. `CliReply(stdout=...)` prints exactly what it is given instead, for output the hook cannot parse.
- A run that finds no reply left says so on stderr and exits with status 3.
- `HookRunner(..., stubs=fakes)` runs the whole capture hook against these fakes. The fakes' own tests call the capture hook's extraction code directly instead.
- The fakes are POSIX shell scripts. On Windows a program that another starts without a shell is found on `PATH` only as an `.exe`, so their tests skip there.

## Documentation that must match the code

`tests/adversarial/docs/` checks that the text a person or a model reads about memvara names only what the code has. Each check parses the text and reads the truth from the code, so no test keeps a list of what the text mentions, and a new tool, argument or switch is checked without anyone adding it. The plan is `docs/superpowers/plans/2026-09-26-adversarial-docs.md`.

- **Every server configuration.** A switch can hide a tool, remove an argument or rewrite a description, so a description that is right on the default server can be wrong on another. `surface.py` therefore starts a server in the test process for each configuration: the default, each feature switched away from its default, read-only, and anchored by default. It reads each server's `tools/list`. The configurations are built from `FEATURE_DEFAULTS`, so a new switch is covered without being added by hand.
- **Tool descriptions.** `test_adv_docs_tools.py` checks every description any configuration serves, and the instructions the server sends when a client connects. Every tool name in them must be a tool. Every snake_case word must be a tool, an argument of some tool, or a built-in predicate or alias, unless the text marks it as example data. A phrase that ties an argument to a tool, such as "memory_recall with include_episodes", must name an argument that tool takes. On each server, neither a tool's own description nor the instructions may name an argument that server's switches removed, because a call that passes it is refused as an unknown argument.
- **How a mention is recognised.** `mentions.py` holds the rules, and its docstring lists them together with what they cannot see. The main limit is that a one-word argument such as `reason` or `query`, written alone as a plain word, is not read as a mention, because it is usually English: "a false reason" is about a stored reason, not about the argument.
- **Defaults stated in words.** `test_adv_docs_defaults.py` reads every default a description states, such as "Default false." or "2 is the useful default", and compares it with the default the input schema declares, in every configuration. This matters because the validator fills a declared default before the handler runs, so the schema's default is what a call gets. A boolean must equal a boolean, because in Python `True == 1`. A default worked out at call time, such as "Defaults to now", must not be declared as a constant. `defaults.py` lists the phrases it reads and the ones it deliberately does not, such as "the default order".
- **The packaged skill.** `test_adv_docs_skill.py` reads `SKILL.md` and every page under `references/`, because they ship together and the skill sends the model to its reference pages. Every tool name must be a tool. Every call in an example, such as `memory_neighborhood(entity=..., depth=2)`, must pass only arguments its tool takes. `skill.py` reads each call with Python's own parser. It skips a call that comes after a `#` on its line, because the skill writes the calls it warns against as comments, such as `# avoid: memory_standing(query=...)`. A backticked argument tied to a backticked tool, as in "`role` on `memory_add`", must belong to that tool. A backticked word with no tie is not checked, because the skill also names library parameters, such as `known_at`, that no tool takes.
- **The command lines and their help.** `test_adv_docs_help.py` finds every command line from the code: the console scripts in `pyproject.toml`, `python -m memvara.server`, and each subcommand their dispatch reaches. `commandline.py` reads with `ast` which words and options each one accepts, and which help text it prints; the words that print the help, such as `-h`, are found the same way, so the help need not list them. Every option a subcommand accepts must be named in its help. Every `MEMVARA_*` variable a help names must be one `memvara/server/config.py` reads, and every feature it names must exist. A claim such as "PROFILE=0 hides memory_profile" must hold on the servers `surface.py` starts. A variable set to the value its help calls the default, and each feature set to its stated default, must leave the configuration unchanged.
- **Each rule is first shown catching its fault** on planted text, and the real-data tests also assert that the parse read something. A parser that stopped reading would therefore fail instead of passing on nothing.

## Stores from old releases

`tests/adversarial/upgrade/` checks that a store written by any release opens with this code, migrates, and keeps everything it held. The plan is `docs/superpowers/plans/2026-09-26-adversarial-upgrade.md`.

- **The committed stores.** `tests/fixtures/stores/<tag>/` holds one store for each schema version a release has shipped, written by the first release that wrote that version: v0.1.0 (version 5), v0.2.0 (6), v0.3.0 (8), v0.10.0 (9), v0.12.0 (12), v0.15.0 (15) and v0.16.0 (16). Versions 7, 10, 11, 13 and 14 never shipped. Every store holds the same program of about thirty operations: superseded, ended, retired, retracted and erased claims; claims of another user, another tenant, an agent and a session; a claim with a source turn; turns the fast path read; accented names and names the entity fold changed in version 16; a built-in predicate that learned an alias, which gives the predicates table a row; and, where the release has them, a graph predicate from the engineering pack that learned an alias too, a project, a link, an expiry and a document. `golden.json` beside each store records what it holds.
- **How a store is built.** `tests/adversarial/upgrade/build_stores.py` extracts the release's `memvara` package with `git archive`, runs the program with that code in a child process, and reads the closed store with `sqlite3` alone. Run it from the repository root, in a clone that has the release tags. The child replaces `uuid.uuid4` with a seeded generator before it imports memvara, so a rebuild mints the same ids, and every instant the program passes is a whole day in 2024 or 1 January 2100. A rebuild therefore differs from the committed store only in the instants the release read from the clock. The builder puts a finished store in place all at once: it copies the files into a directory beside the store, then swaps the two directories with two renames, so a copy that fails partway leaves the committed files as they were. A release whose writer runs past five minutes is stopped, and the build fails with the writer's output. `test_adv_upgrade_builder.py` checks both without needing a release.
- **The golden dump.** `tests/adversarial/upgrade/golden.py` reads a store with `sqlite3` alone, never through memvara, which would migrate it. `golden.dump` returns every claim with its scope, text, both clocks and sources, every turn, and the provenance edges, erasure records, links, documents, entities and predicates, each predicate with the graph declaration version 10 added. It leaves out what a migration recomputes on purpose: the entity keys, the two hashes and the two type columns. A table that a store's version lacks reads as empty. A column it lacks reads as the value the migration that adds it gives existing rows, which is `None` for the nullable columns and version 10's defaults for a predicate's graph declaration. So a store dumps the same before and after its migration, and a migration that writes anything else there shows as a change. `test_adv_upgrade_golden.py` shows each comparison the upgrade tests rely on catching a change.
- **Why the files are compressed.** The database and the vector file are committed gzip-compressed, about 20 KB a store. Uncompressed, every store is over the 256 KB limit. Each of today's 53 tables and indexes takes at least one 4 KB page, so an empty database is already 228 KB, and the vector file reserves room for 256 vectors, which is 512 KB at width 512. A smaller page size or embedder width would fit too, but would make the stores unlike any store a real user has.
- **What every pull request checks.** Each store is unpacked into the test's temporary directory, so no test can change a committed file. Opened with this code, it must open without a warning, reach today's schema version, dump exactly as `golden.json` says, pass the integrity checks, leave no write-ahead log behind when it closes, and not change on a second open, down to the size of the write-ahead log and the shared-memory file. Every claim must read back with the same subject, predicate, object and clocks. Every live claim and every turn must be found by search in its own scope. A reader in each scope the fixture writes in, and in a sibling session and a sibling agent, must get a claim by id exactly when `Scope.sees` in `memvara/types.py` allows it, and must not find by search a claim it may not see; so a user-wide reader sees no session's or agent's claims. Every document and every erasure record must still be there. In every scope, restating a stored value must reinforce the stored claim, and a new value in a single-valued slot must end that scope's value and nothing else, which shows that versions 6, 12 and 16 re-keyed the claims correctly. The second check runs on a store nothing has restated, because a write re-saves the claim it touches with fresh keys and would hide a wrong one. A claim an old release erased must stay unreadable, including the words a release before version 7 left in the text index.
- **What must be refused.** A store stamped with a schema version newer than this code's must be refused with a message that names both versions, twice in a row, and be left exactly as it was. An old store opened with an embedder of another width must be refused, by the library and by the MCP server, which prints one line and exits with status 2. An embedder of the same width but another vector space is warned about rather than refused, as `Memvara._check_embedder` documents, and each old release's embedder record must still trigger that warning.
- **The tests catch the breaks they are for.** They were run against deliberately broken copies of memvara, each in a scratch directory, and each break failed the test aimed at it, for exactly the stores it affects. For the upgrade: no `optimize` in `_migrate_to_v7`, no re-keying in `_migrate_to_v12`, no `_migrate_to_v15`, a version stamped one too low, a backfill of `object_kind`, a write on every open that migrates nothing, turns moved into a project, a predicate's graph declaration cleared on upgrade, a graph column added with the wrong default, a version 12 slot hash computed with one user's name for everyone, and a close that keeps its connection and so leaves a write-ahead log. For reading: a `Scope.sees` that also reaches into deeper scopes. For the refusals: a newer store that opens, a newer store written to before it is refused, no width check, no same-width warning, and a server that lets the embedder error out as a traceback.
- **What the nightly run checks.** It needs a clone with the release tags, and fails without them. It rebuilds every store from its tag and compares the rebuild with the committed store, hiding only the instants the release read from the clock, and checks that every schema version a release shipped has a store. A build that fails must leave the committed store and its golden record as they were. It kills a child process between two migrations of each old store: every row and side file must be as it was, a table the upgrade adds must be empty, and the next open must migrate cleanly. It upgrades a store whose release was killed with a committed write still in its write-ahead log, which must keep that write. It also has v0.15.0 write an encrypted store on the spot, since the MCP server creates encrypted stores and their pages do not compress enough to commit, and that store must upgrade with its key and lose nothing. Each of these was run against a deliberately broken copy and failed.
- **When a release changes the schema.** Add the release to `golden.RELEASES` and run `python tests/adversarial/upgrade/build_stores.py <tag>`. The nightly test `test_every_released_schema_version_has_a_committed_store` fails until you do.
- **Two support modules.** Besides its tests, the folder holds `golden.py` and `build_stores.py`, which are not tests.

## Protocol and validator fuzzing

`tests/adversarial/fuzz/` sends the MCP server input that a correct client would never send. After each input it checks three things: every request that has an id gets exactly one reply with that id, a call the server refuses changes nothing in the store, and the server still answers a ping. The plan is `docs/superpowers/plans/2026-09-26-adversarial-fuzz.md`.

- **How replies are counted.** `exchange(server, *lines)` sends the lines exactly as given, then a ping, and returns every message the server wrote before it answered the ping. The server handles one line at a time and in order, so a request that got no reply, or two, shows in that list. The ping's own reply is checked too, which is how every test checks that the server carries on.
- **How "changed nothing" is checked.** `rows(db)` reads every row of every table through a second, read-only SQLite connection, and a digest of the `.vecs` vector file, while the server keeps running. A test takes one reading before a refused call and one after, and `changed` names any table that differs.
- **Where the helpers live.** This folder may hold only test files, so the helpers are in `tests/adversarial/fuzz/__init__.py`. `test_adv_fuzz_helpers.py` shows each one catching the fault it exists to catch.
- **The wire** (`test_adv_wire.py`). 500 requests are written before any reply is read, with shuffled ids and a mix of methods, and every write among them must be stored exactly once. An id of every JSON type comes back unchanged. A null id is treated as a notification and gets no reply. A batch is refused whole, and nothing in it runs. Two requests on one line get one parse error. Blank lines and CRLF endings get no reply. A line nested 100,000 deep gets a parse error wherever the nesting sits, and the deepest nesting that still parses is found by a search and answered. An integer of more than 4,300 digits gets a parse error.
- **Arguments** (`test_adv_arguments.py`), for every tool that takes each kind of argument: the strings "false" and "true" and the numbers 0 and 1 where a boolean goes; NaN, Infinity and -Infinity where an integer goes; an infinity past a number's bounds; and a lone surrogate in every place the schema says a string goes. Each must be refused with a message that names the argument, and change nothing. A NaN confidence must also be refused and change nothing, even when the write would have ended the value already in its slot.
- **Generated calls** (`test_adv_schema_fuzz.py`). Hypothesis draws arguments from each tool's own input schema, so a new argument is fuzzed without anyone editing the suite. In the test process, a call that follows the schema must be accepted with its defaults filled in, a call that breaks one rule must be refused with one line that names the rule, and any JSON at all must be accepted or refused, never met with another exception. Over the real pipe, a sample of such calls must each get one reply, a refusal must be the validator's own message word for word, and a refused call must change nothing. Half of the broken calls break an argument's value, and the near misses a model sends, such as the string "false" for a boolean, are drawn on purpose rather than left to chance. Each property runs once per tool, so the nightly and weekly tiers give it a tenth of their example count per tool, 300 and 2,000, and the fast tier keeps 30.
- **Nightly** (`nightly/`). Lines of 20 MB: a request padded with whitespace, a line that is not JSON, a line of spaces, 20 MB where an integer goes, a 20 MB argument name, a 20 MB fact, which is stored whole, and a 20 MB query. On a laptop the fact took about a minute to store and the query between ten seconds and two minutes, and the server answers nothing else while it works on either. And generated calls against a read-only server, a server that anchors by default, and a server with every feature that owns a tool or an argument switched off, where a call to a tool the server does not list must be refused and change nothing. The nightly tier of this folder took about five minutes on a laptop.
- **What it found.** Six bugs, each pinned as a strict expected failure in `test_adv_fuzz_known_bugs.py`:
  - #311: the server reads its input in the locale's encoding rather than UTF-8. Under a strict UTF-8 stream, one byte that is not UTF-8 ends the server, and under another encoding, UTF-8 text is stored wrongly.
  - #312: NaN passes the bounds on `confidence` and `min_score`.
  - #313: a refusal, and the reply to a read that finds nothing, quote the whole argument, however long it is.
  - #314: a filter key that ends in a newline passes the key pattern.
  - #315: a lone surrogate in an object argument's key is stored.
  - #316: `memory_recall` reports `ranked` without `include_episodes` as a `ValueError` from the catch-all instead of an argument error.

  The tests for #311 set `PYTHONIOENCODING` rather than a locale, because it sets the stream encoding the same way on every platform, whatever locales a machine has installed.

## The coverage checklist

The checklist is a list of everything the suite has to test, together with the tests that cover each item. It is read from the code, so a new tool, switch or invariant is added to it as soon as it exists. `tests/harness/checklist.py` builds it, and `tests/adversarial/test_adv_checklist.py` checks it in every fast run.

**What is on the checklist.** Each item has an id of the form `kind:name`.

| Kind | One item for each | Example |
|---|---|---|
| `tool` | tool in `memvara.server.tools.TOOLS` | `tool:memory_recall` |
| `switch` | feature switch in `FEATURES` (`memvara/server/config.py`), and the read-only and anchored modes | `switch:documents`, `switch:read_only` |
| `tool-switch` | tool whose entry in `tools/list` changes when a switch is flipped from its default | `tool-switch:memory_add_document/documents` |
| `env` | `MEMVARA_*` variable that `memvara/server/config.py` reads | `env:MEMVARA_DB` |
| `hook` | hook that a host in `plugin/hooks/hosts/` fires | `hook:claude/recall` |
| `inv` | numbered invariant in `docs/INTERNALS.md`, and bullet under "Invariants and assumptions" on a `docs/claude/` page | `inv:I3`, `inv:MM5` |
| `silent` | silent failure mode that the `memvara/telemetry.py` docstring lists | `silent:predicate-explosion` |
| `bug` | open bug in `tests/harness/known_bugs.py` | `bug:B2` |

These details explain how some of the sources are read:

- **Tool-switch pairs.** The checklist compares the `tools/list` reply of a server with every setting at its default against the reply with one switch flipped. A switch is flipped rather than turned off because two features, and both modes, are off by default. The servers run inside the test process, which is enough: `python -m memvara.server` builds the same server from the same three settings.
- **Environment variables.** A variable counts only when `config.py` reads it, through `.get()`, `getenv()` or a subscript. A variable that appears only in an error message does not count. The `MEMVARA_FEATURE_*` variables are the `switch` items.
- **Silent failure modes.** The telemetry docstring lists six in a table and announces the seventh in a sentence of its own. The checklist reads both.
- **A source that yields nothing stops the run.** Otherwise a renamed heading, or a table that is missing or has no rows, would drop that source's items from the checklist, and nothing would report it. The known bugs are the exception, because running out of open bugs is the goal.

**Each source written as text is read by a parser that recognises particular forms.** For the environment variables, it recognises a read through `.get()`, `getenv()` or a subscript. For the silent failure modes, it reads the table whose header names the columns failure and signal, and any sentence that announces a mode on its own, in the form "A seventh arrived with the ...". For the invariants, it reads the numbered list under the design invariants heading in INTERNALS, and the bullets under "Invariants and assumptions" on each page. So a new form of a source the checklist already reads, such as a new way of reading an environment variable, is missed until its parser learns it: update the parser in the same change. The check above catches only a source that yields nothing at all, not one that has lost some of its items.

**Invariant ids.** An invariant's wording changes over time, so it needs an id that does not. `tests/harness/invariant_ids.json` records, for each document, each invariant's id and the bold sentence the invariant opens with. The sentence is how the checklist finds the invariant.

- The numbered invariants in INTERNALS take their number as their id, `I1` to `I8`.
- A bullet on a `docs/claude/` page gets its page's prefix and a number, such as `MM5` on `memory-model.md`. Give a new bullet the next unused number on its page.
- A bullet that restates an INTERNALS invariant carries that invariant's id, so that one test covers both. When the bullet says so, as in "This is invariant 3", the fast tier checks that it carries the right id.
- When you reword an invariant's opening sentence, change the sentence in the file and keep the id. Until you do, the fast tier fails and names the sentence.

**Declaring what a test covers.** Put a `covers` mark on the test, naming the items that its assertions check:

```python
@pytest.mark.covers("tool:memory_history", "env:MEMVARA_DB")
def test_the_store_outlives_the_server_process(mcp: Start, tmp_path: pathlib.Path) -> None:
```

Name only what the assertions check. A test that happens to start a server does not cover every variable the server reads.

The checklist reads the marks from each file's source instead of importing it, because importing a nightly or local test can need Docker or a package that is not installed. So the ids must be string literals, and the mark must sit on a test function, on a test class, or in a module's `pytestmark`. A mark anywhere else, or one built from a variable, fails the run with its file and line. Without that check, the test would cover nothing and nobody would be told.

Three kinds of test cover nothing, because none of them shows that anything works:

- a test marked `xfail`;
- a test marked `skip` without a condition;
- a test in `quarantine/`.

A known bug is covered differently. Its item is covered by the test that carries its strict expected failure, `known_bugs.xfail("B2")`, so registering a bug and pinning it is all the checklist needs. A `covers` mark cannot name a bug.

**The baseline.** `tests/harness/checklist_baseline.txt` lists the items that no test covers today, one per line. The fast tier fails in two cases, and each failure lists the items concerned:

- An item has no test and is not in the baseline, for example a tool that was added without a test. Write a test that covers it. Adding the item to the baseline instead would hide exactly what the checklist exists to catch.
- A line of the baseline is no longer a gap, because a test now covers the item or the item no longer exists. Delete the line.

These two checks keep the baseline equal to today's gaps, in both directions. They do not make it shrink: review does that, as the next paragraph explains. The checklist is complete when the baseline is empty. `test_a_new_feature_switch_is_a_gap_the_baseline_does_not_list` shows the first check working: it adds a switch to `FEATURES` and checks that the checklist reports that switch, and nothing else, as a new gap.

**Adding a line to the baseline excuses a gap instead of closing it.** The tests check only that the baseline and today's gaps hold the same items. So a pull request could add a tool with no test and, in the same diff, a baseline line for it, and the fast tier would pass. A pull request that adds a line to the baseline must therefore give the reason in its body, and the code review checks the baseline's diff for added lines.

**Exempt items.** Four invariant bullets are rules for people rather than behaviour of memvara, so no test can check them: `TB1` ("verify" means comparing an output), `TB2` (a number is reported with its caveat), `RC5` (this repository does not implement the hosted server) and `RP1` (a published version is final; the release process is outside the suite's scope). `EXEMPT` in `tests/harness/checklist.py` lists each with its reason. An exempt item stays on the checklist, so a reworded rule is still noticed, but it is never a gap. The fast tier fails if an exemption names an item that no longer exists, or an item that a test covers. Add an exemption only for a rule that no test could ever check, and give the reason.
## The nightly run

The nightly run turns what the slow tiers find into a report and, for each new break, one issue. The plan is `docs/superpowers/plans/2026-09-26-adversarial-nightly.md`.

### Running a night

`scripts/nightly/run.py` runs one night. Its first act is to write the night's heartbeat. Its preflight step then hashes the operator's protected files for the isolation canary, removes the worktrees earlier nights left (unless one holds uncommitted work), fetches `origin`, adds a clean, detached worktree of `origin/main` at `local/nightly/<date>/worktree`, and builds a virtual environment inside it with `.[dev,cloud,ingest,encrypt]`, as CI does. After that it runs the design's steps in order, each within its own cap:

| Step | Cap | Built |
|---|---|---|
| preflight | 20 minutes | yes |
| regressions | 75 minutes: the test run may use 60, and the reruns of failed tests get the rest | yes |
| agents | 20 minutes | no: waits for `tests/live/agents` |
| red team | 25 minutes | no: waits for `tests/live/redteam` |
| hosted | 15 minutes | no: waits for `tests/live/stack.py` |
| production smoke | 5 minutes | no: waits for `tests/live/prod_smoke.py` |
| performance | 15 minutes | no: waits for `bench/perf_budget.py` |
| soak | 20 minutes | no: waits for `bench/soak.py` |
| mutation | 15 minutes | no: waits for `bench/mutation.py` |
| replay | 10 minutes | no: waits for `tests/live/replay.py` |

A step that is not built appears in every report with its reason. When the file it waits for lands on `main`, the report says so, and the step's command still has to be added to `STEPS` in `scripts/nightly/run.py`. The caps of the unbuilt steps are placeholders for the work that builds them. A full `pytest --tier nightly` run took 16 minutes 37 seconds on a laptop on 2026-09-26, with other test suites running beside it, so the regressions cap leaves room.

The code under test always comes from the fresh worktree, but the run's own code comes from the checkout it was started in: the nightly scripts, and the harness they import, including `tests/harness/report.py`, which computes every fingerprint. When any of those files differs from the worktree's copy, the report warns and names the files, because the night was then run, and its breaks fingerprinted, by code that is not on `main`.

The run writes these files in `local/nightly/<date>/` in the main checkout, however it was started:

- `report.md` for a person and `report.json` for a program. The first line of `report.md` counts the breaks and names every step that did not pass, so a night whose preflight failed never reads as a quiet one. It also says how many steps are not built yet, and on a quiet night how many ran, for example "Nothing broke in the 2 steps that ran. 8 steps are not built yet.", so a quiet night never reads as if every step had checked the code. The reports list every step with its result, time and cap, every new, recurred and known break with what filing it still needs and the exact commands, the failures that need a person, the flakes, the flake rate of each layer, the canary, the dependencies and the notifications sent.
- `findings.jsonl`, every break the night saw, one `Finding` per line.
- One folder per step with its output; `regressions/results.jsonl` holds one line per test.
- `heartbeat.json`, which the watchdog reads.

It also adds one record per night to `local/nightly/history.jsonl`: the commit, each step's result, each layer's tests, failures and flaky tests, the fingerprints of the confirmed breaks, the dependencies and the canary. A later night reads it to recognise a break it has seen, to measure flake rates, and to notice a dependency that stays down. A step that raises costs only that step. If the run's own code crashes, it still compares the canary and writes the report, marked as crashed, and it leaves the heartbeat without a finish, so the watchdog reports the night too.

A step can report findings of its own by writing `findings.jsonl` into its folder. The run counts each as a confirmed break, because the step confirms a finding before it writes it, as the red team will by replaying it three times. Only a finding its step has classified can be filed without a person: with `--file`, a security-class one goes to a private draft advisory and any other one to an issue.

To run a night by hand, from the main checkout:

```bash
python3 scripts/nightly/run.py                      # tonight, filing as a dry run
python3 scripts/nightly/run.py --date 2026-09-27    # a named night, such as one the watchdog reported
python3 scripts/nightly/run.py --worktree <checkout> --python <interpreter> --no-notify
```

A night's name must be a real date written `YYYY-MM-DD`, because that is the folder the watchdog looks for. The run refuses any other form before it writes anything, and `filing.py` refuses it too. The last form tests an existing checkout with an existing interpreter, and sends no notification. It is for a supervised run, and the report says the checkout was given rather than fresh. `--canary PATH` adds a file to the canary, and `--file` turns filing on.

Notifications go out only for a new or recurred break, for an isolation breach (a file the canary watches changed during the night), and for a dependency such as `origin` that is down for the second night in a row. The canary watches `~/.memvara/credentials.json` and `~/.memvara/db.key` by default. It keeps only their hashes, never their contents.

### Findings

A finding is the record of one break. `tests/harness/report.py` defines it as `Finding`, and a file of findings holds one JSON line per finding. A finding has these fields:

| Field | What it holds |
|---|---|
| `layer` | The part of the suite that found the break, such as `model` or `concurrency`. |
| `surface` | What the break was seen through, such as `library` or `server`. |
| `invariant` | The property that failed: an invariant's name, or the node id of a failed test. |
| `severity` | Where the break may be filed; see below. |
| `ops` | The operations that replay the break, as JSON values. |
| `seed` | What reproduces a random search, such as the `@reproduce_failure(...)` call Hypothesis prints. |
| `artifacts` | Files kept with the finding, by name, as paths inside the night's folder. |
| `commit` | The commit that was tested. |
| `title`, `detail` | A one-line summary and the failure's text, for a person to read. |

**The fingerprint.** `Finding.signature()` is a SHA-256 hash of the layer, the surface, the invariant and the operations. The nightly run uses it to recognise a break it has seen before, so a break that fails on two nights becomes one issue. The seed, the commit, the artifacts, the title and the failure text are left out because they change from night to night, and the severity is left out because triage can change it. So a producer must give the operations in a minimal, deterministic form: a temporary path or a wall-clock time in them would give the same break a new fingerprint every night. A test pins the exact text the hash is taken over, because changing it would make every known break look new and be filed again.

**What the fingerprint cannot tell apart.** A failed test whose failure carries no replay program is fingerprinted by its layer, its surface and its node id alone. So when the same test later fails for a different reason, the second failure gets the first one's fingerprint. While the first break's issue is open, the second failure is counted as that known break, and nothing new is filed or notified. Once that issue is closed, a run with `--file` reports the second failure as recurred and notifies it, so it reaches a person. That person has to read the failure to see that it is a different bug. A dry run cannot tell, because only GitHub says whether an issue is closed.

**The severity decides where a break may go.** `unclassified` means nobody has checked the break against the "In scope" section of `SECURITY.md` yet, and the nightly run never files an unclassified break in public. `security` means it is in scope, so it goes to a private draft advisory. `data-loss`, `wrong-result` and `crash` mean someone checked and found it out of scope, so it can be filed as a public issue.

## A model that misbehaves

`tests/adversarial/model_faults/` checks what memvara does when the model it is configured with answers badly: output that is not JSON, is cut off or has the wrong shape; predicates nobody declared; 10,000 claims in one reply; a timeout or a rate limit; a proposal to retire or erase a stored claim; and a tool loop that never stops. Four things must hold whatever the model does. Every turn the caller gave is stored. Nothing already stored is retired or erased on the model's word alone. A read stage that fails serves the read a store with no model serves. And the number of model calls each operation makes is the number `docs/INTERNALS.md` states. The plan is `docs/superpowers/plans/2026-09-26-adversarial-model-faults.md`.

- **The scripted model.** `scripted.py` holds `ScriptedModel`, which implements the four model protocols memvara calls (`LLM`, `Chat`, `ToolChat` and `ReplacementJudge`), as both shipped backends do. Each method answers with the next reply in its script, and every call is recorded, so a test can count the calls an operation made. It stands in for the provider and nothing more. A `Text` reply goes through `memvara.llm._shape`, the validation both backends run, so a malformed reply meets the code a real one would meet. `run_tools` runs memvara's own tool loop, so the step limit and the retry of an agentic run are memvara's own.
- **Provider errors.** The `anthropic` and `openai` SDKs are not installed in the test environment, so `APITimeoutError`, `RateLimitError` and the others are stand-ins. They copy what memvara reads from the real exceptions: the class name, the `status_code`, and a class hierarchy in which a timeout is not Python's `TimeoutError`.
- **Time.** The model keeps its own clock, which moves only when a reply is scripted as `Late`. Memvara reads that clock in the tool loop, and in a query rewriter or synthesizer built with `clock=model.clock`, which is how `handles.with_model` builds them. No test sleeps.
- **A call nobody scripted fails the test.** Memvara catches most exceptions a model raises, so a call made after a script ran out would pass unnoticed. The `scripted` fixture fails the test instead, when the test ends. For the same reason the model keeps every exception it raises in `failures`, and `handles.raised_in` names the file and function inside memvara that an exception came from. A test that pins a bug uses the two to check the bug's exact symptom.
- **Two handles on one store.** `handles.py` builds one handle with the scripted model and one with no model on the same store. `ledger` records every claim before a write, and `fates` names what the write did to each one: `unchanged`, `ended`, `extended`, `reopened`, `retired`, `erased`, `missing` or `changed`. A claim is `ended` when its world clock closes or moves to an earlier end. It is `extended` when an end it already had moves later, and `reopened` when that end is cleared; both break the rule that a closed clock never moves later, so each has its own name. The scripted model's own tests check every one of these names against hand-built rows, and show `fates` seeing an end, a retirement and an erasure in a real store, before any other test relies on it.
- **Malformed output on the write path** (`test_adv_write_output.py`). Each reply that holds no usable claim is sent in two forms: as text, which goes through the same validation a shipped backend runs, and as values, the way a backend that does no validation of its own would return them. The replies include prose, an empty string, JSON cut off with and without a stop reason, the wrong top-level shape, claims that cite a turn that does not exist, and claims with a field missing. For every one, the batch's turns are stored, the fact the fast path read from the same batch is kept, nothing already stored changes, and the write costs one call. A reply the provider cut off at its token limit also marks the batch deferred. A garbled field that can be repaired, such as a confidence written as text, is stored repaired.
- **Invented predicates.** A new predicate spelling costs one acquisition call and never another, even after the store is reopened. A model that answers that its own spelling means `lives_in` gets the merge, but its claim is stored at confidence 0.4 beside the user's value, so it cannot end it. A closed vocabulary refuses invented predicates without a call, and past the cap of 200 learned predicates a new spelling costs no call. A reply of 500 claims costs one call, and a reply that restates one fact 500 times stores one claim, seen 500 times.
- **Provider errors on the write path** (`test_adv_write_errors.py`). When the extraction call raises a timeout, a 429, a 401, a 529, a connection reset or a truncation, the batch's turns and the fast path's fact are kept, the receipt says the batch was deferred, and the failed call is billed. `reextract()` then reads the deferred turn once the model answers, and a second sweep costs nothing. A failed acquisition call is billed once and never retried, and the claim is stored under its unresolved predicate. A failed replacement judge warns once and leaves the advice empty, and the write it advised on is kept. With extraction chunks on, a failed piece defers only its own turn, and the pieces after it are not sent.
- **Proposals to retire or erase** (`test_adv_retire_erase.py`). A model may end a stored fact, which says that the world changed. It may not retire or erase one. A retraction or a new value from single-call extraction ends the old value, and extra keys such as `"close": "retired"` or `"erase": true` in the model's claim change nothing. A retraction with no object, which the reconciler would read as clearing the whole slot, is dropped before it gets there. In agentic extraction, an end proposed after reading the claim ends it with the model's reason. An end for a claim the model never read, or for a user-wide claim from inside a session, is refused, and so is any tool the run was not offered, such as `erase_claim`. A replacement the reconciler does not accept leaves the named claim live. An agentic model gets the same answer for another user's claim id as for an id that never existed. Replacement advice that answers "replace" to every question closes nothing. The preview that `forget_matching` shows before it retires anything is made without asking a model, so a model cannot change what the caller confirms.
- **Tool loops** (`test_adv_tool_loops.py`). Agentic extraction is off by default, so these tests switch it on. A model that never stops calling tools is stopped after twelve answers, its proposals are discarded, the batch falls back to one extraction call, and every request is billed. An answer that cannot be used, or a 429, is retried once. A timeout is not retried. A provider that takes 10 seconds per answer uses up the 25 seconds that `add()` allows after three requests. The same provider reaches the step limit in `reextract()`, which allows 180 seconds. A `k` of 0, -5, 10,000, text or a boolean in a search call is clamped or replaced by the default, and the run carries on.
- **Read stages that fail** (`test_adv_read_stages.py`). Each test reads one store through a handle whose model fails and through a handle with no model. The store is written once for the whole file, because no test there writes to it, and each test fails if the store has changed by the time it ends. When the query rewrite fails, whether from a provider error, a late answer or a reply it cannot read, `recall()` returns the same text as the handle with no model, `search()` returns the same results field by field, the MCP tools return the same text, and the outcome names the reason. The `search()` comparison pins `known_at`, because recency is measured from it and two reads a moment apart would otherwise differ in their scores' last digits. When the synthesis fails, the block's first line says why, in the place where a store with no model says `unconfigured`, and every note after that line is the same. When the ranking fails, no turn is marked as selected and the block's last line names the outcome. A rewrite that returns seven alternatives still costs one call and four retrievals, and a selector reply that is only partly readable keeps only the entries it can read.
- **Model calls per operation** (`test_adv_call_counts.py`). One table, one row per operation, each row quoting the sentence of `docs/INTERNALS.md` it checks. A turn the fast path reads, a turn the gate drops, a repeated turn and `remember()` cost nothing. Three turns that reach the model cost one extraction call. A new predicate costs one more call the first time it appears and none after that, even after the store is reopened, and under a closed vocabulary it costs none. Replacement advice costs one call for each of up to three neighbours, a long turn with extraction chunks one call per piece, and an agentic run one call per request. A `search()` costs one call for the rewrite, `recall(synthesize=True)` two, and a ranked recall two. A read with `query_rewrite=False` or `k=0` costs nothing, and so does the preview before `forget_matching`. A recall that finds nothing to summarise costs only the rewrite's call. Every write's `receipt.llm_calls` equals the calls the model answered.
- **10,000 claims in one reply** (`nightly/test_adv_ten_thousand_claims.py`, nightly tier). A reply of 10,000 distinct claims costs one extraction call and stores every claim, each citing its turn, whether the model returns them as values or as about 2 MB of JSON text. A reply that restates one fact 10,000 times stores one claim, seen 10,000 times. Each of these three tests takes one to two seconds on a laptop, which is why they are nightly; the fast tier checks the same properties with 500 claims. A fourth nightly test pins #309 below. It writes 5,000 claims under as many invented predicates, and it takes about two minutes while that bug is open.
- **Pinned bugs.** These tests found seven bugs, each pinned as a strict expected failure that cites its issue: #303 (malformed model output makes `add()` raise instead of dropping the item), #304 (one claim with an overflowing confidence makes `add()` drop the whole batch), #305 (a claim the trust boundary drops still costs a model call and leaves a learned predicate), #306 (a claim with no subject is filed under the user, and a list object is stored as Python text), #307 (a model retraction at low confidence ends a fact the user asserted), #308 (a ranked read whose selector fails does not serve the plain read) and #309 (invented predicates past the learned cap make one write take quadratic time).

The fast tier of this folder has 207 tests, 19 of them strict expected failures, and takes about 3 seconds on a laptop. To run the nightly tests as well:

```bash
PYTHONPATH=$PWD python -m pytest -q -p no:cacheprovider tests/adversarial/model_faults --tier nightly
```
### Steps and time caps

A night is a list of steps, run one after another by `scripts/nightly/steps.py`, and each step has a time cap. The commands a step runs start in a process group of their own. At the cap the whole group is stopped, so a stuck step cannot use up the night or leave a server running into the morning; on POSIX, whatever a command leaves running in its group is stopped when the command exits, too. Each step runs even when the step before it failed, with one exception: when an essential step does not pass, every later step is reported as "not run", with the name of the step that failed. Preflight is essential, because it builds the worktree and the virtual environment the other steps use. A step whose code has not landed is reported as "not built yet", with the reason and the file it waits for. A step that raises is reported as an "error", and a step that returns after its cap as "timed out".

The steps run with the environment `night.step_env` builds. It is the run's own environment without the variables the harness keeps from every child process, with `HOME` pointed at `local/nightly/home/`, a private temporary folder, and the tested worktree on `PYTHONPATH`. The home folder is kept from night to night, because the nightly Hypothesis profile keeps the examples it finds under the home directory. No `MEMVARA_` variable is set, so the suite runs with the same defaults as in CI.

### Flakes

A test that fails during the night is run twice more by `scripts/nightly/flakes.py`, in the same tier, and the majority of the three runs decides what the failure was:

| Reruns | Verdict | What happens |
|---|---|---|
| both pass | flake | The majority passed, so nothing is filed. |
| both fail | confirmed | A break that may be filed, because its strict expected failure will fail every time too. |
| one passes, one fails | intermittent | It counts as a failure, but it is not filed: a strict expected failure on a test that sometimes passes would make the suite flaky. A person looks at it. |
| a rerun could not run the test, or it was never rerun | unconfirmed | It counts as a failure, and a person looks at it. |

A test that passed after failing counts as flaky whatever its verdict. The flake rate of a layer is its flaky tests over the tests it ran in the last fourteen nights, and the design's budget is 0.5% per layer. A test's layer is the first folder under `tests/adversarial/` that is not a tier folder, so a nightly concurrency test counts as `concurrency`; the rest of `tests/` is `unit`, and the doctests are `doctest`. To see the rates, or to rerun tests by hand:

```bash
python3 scripts/nightly/flakes.py rates
python3 scripts/nightly/flakes.py rerun --worktree <checkout> --python <interpreter> <node id>
```

### The regressions step

The regressions step runs `pytest --tier nightly` in the tested worktree through `scripts/nightly/pytest_results.py`, which records one JSON line per test in `regressions/results.jsonl` and pytest's exit status on the last line. The step passes `--continue-on-collection-errors`, so one test file that fails to import is recorded as an error and cannot stop every other test from running. A line of `results.jsonl` that cannot be read, such as one a stopped run left half written, is named in the report's warnings, because the test it recorded is missing from the report. Each failed test is rerun in the step's own tier, which one constant in `regressions.py` holds.

Each failed test becomes a finding (`scripts/nightly/regressions.py`). Its invariant is the test's node id, its layer comes from the test's folder, and its severity is `unclassified`, because nobody has checked it against `SECURITY.md` yet. When the failure's text holds a `drive.replay([...])` program, as a failure of the reference model's state machine does, the program becomes the finding's operations and the `@reproduce_failure(...)` call becomes its seed. So two different breaks that the same test finds get two fingerprints. Two kinds of failure are reported for a person and never filed: a strict expected failure that passes, which usually means its bug was fixed, and a run that exits with an error but reports no failed test, which means something outside the tests failed, such as the skip ledger or a credential guard. When pytest dies before it records anything, for example on an import error in a conftest file, the process's own exit status stands in for pytest's, so that night is reported the same way. At most 20 failed tests are rerun; the rest are reported as unconfirmed.

### Filing a break

`scripts/nightly/filing.py` files a confirmed break, and it is a dry run unless `--file` is given: a dry run calls nothing, and prints the exact commands a real run would send. Nothing in the suite calls GitHub; the tests replay the output of `gh` and `git` from `tests/adversarial/nightly_runner/gh_recorded.json`.

- **A public break** gets one issue with the label `nightly-break`. The issue's body ends with a hidden marker, `<!-- memvara-nightly-fingerprint: <fingerprint> -->`. Before filing, `filing.py` reads every issue with the label and compares their markers, so a break that was filed before is not filed again, even when the local history is lost. A break whose issue is closed but which fails again is reported as recurred and notified, and it is never planned as filed and finished: with `--file` the run reopens the issue, with a comment that carries the marker, and the break then needs a new strict-xfail test and draft pull request, because the fix removed the old pin. A reopening therefore clears the old pull request from what counts as filed. The run only knows an issue is closed by reading GitHub, so a dry run cannot see a break recur. Then a branch named `test/nightly-<first 12 characters of the fingerprint>` is pushed with the strict-xfail test, by an explicit refspec that cannot reach `main`, and a draft pull request is opened. Nothing is merged.
- **A security-class break** gets a private draft advisory whose description carries the marker, and nothing else: no issue, no branch and no pull request, because its failing test lands together with its fix.
- **An unclassified break** is filed nowhere public. A failed test from the regressions step is always unclassified, so the scheduled session checks it against the "In scope" section of `SECURITY.md` first, and then files it with the class it chose:

```bash
python3 scripts/nightly/filing.py issue --night <date> --fingerprint <fingerprint> --severity wrong-result --file
python3 scripts/nightly/filing.py pr --night <date> --fingerprint <fingerprint> --worktree <worktree with the pin committed> --file
python3 scripts/nightly/filing.py advisory --night <date> --fingerprint <fingerprint> --file
```

A dry run calls nothing at all, git included, so a dry run of `pr` does not check the worktree for uncommitted changes; a real run does, before it pushes. Before a break's issue exists, a dry run of `pr` uses the break's own severity, so it refuses an unclassified or security-class break, just as a real run would.

Every filing with `--file` is recorded in `local/nightly/history.jsonl`, so a later night knows the break is filed. Before anything goes to GitHub, the operator's paths are removed from the failure's text: the checkout, the home folder, the temporary folder, and the user name in pytest's temporary folders. The label has to exist before the first real filing; create it once with `gh label create nightly-break --repo memvara/memvara --description "Found by the nightly run"`.

### The watchdog

A night that did not run leaves no report, so nothing would say it was missed. `scripts/nightly/watchdog.py` says so. The nightly run writes `heartbeat.json` in the night's folder before it does anything else, and writes it again with a finish time when it ends. launchd runs the watchdog once a day at a deadline, and the watchdog checks the latest night whose deadline has passed:

- With no heartbeat, the night did not run. It writes `DID-NOT-RUN.md` and `DID-NOT-RUN.json` in the night's folder and sends one macOS notification.
- With a heartbeat that has no finish, the run started and was stopped partway, or is still running past every step's cap. It writes `DID-NOT-FINISH.md` and `DID-NOT-FINISH.json` and sends one notification.
- With a finished heartbeat, it does nothing.

A night is named by the day its run starts, so a run scheduled for 23:30 is checked the next morning. The watchdog writes the `.md` report first, then the `.json` record with `"notified": false`, then sends the notification, then sets `"notified"` to true. Only the record says whether the night was reported. So a check that was stopped anywhere before the notification went out sends it when it runs again, and every check after that sends nothing. Only a stop in the moment between the notification going out and the record saying so could send it twice. The watchdog reports only the latest night whose deadline has passed; a night missed before that one is not reported on its own. The watchdog uses no model, does nothing else, and imports nothing from the nightly package, so launchd can run it with any Python 3.10 or later. It sends the notification through `osascript`, passing the title and the message as arguments to a fixed script, so no text is ever read as AppleScript. `scripts/nightly/com.memvara.nightly-watchdog.plist.template` is its launchd job; nothing installs it, and its comment says how to fill it in and load it. To check a night by hand:

```bash
python3 scripts/nightly/watchdog.py --checkout <main checkout> --start 01:30 --deadline 06:30 --no-notify
```

### The scheduled session, and what the operator installs

`scripts/nightly/task.md` is the prompt for the scheduled session that runs each night. The session runs `run.py`, reads the report, and classifies each new break against the "In scope" section of `SECURITY.md`, treating a break it is unsure about as security-class. A security-class break goes to a private advisory and nowhere else. Any other break gets its issue, then a strict-xfail test in a worktree of its own, then a draft pull request, which gets the code review every pull request needs. Nothing merges, nothing is pushed to `main`, and no text that reaches GitHub carries an attribution. The session writes its classifications into `local/nightly/<date>/triage.md`. A test parses every command the prompt names with its script's own parser, so the prompt cannot drift from the scripts.

Nothing in this repository installs the schedule, the watchdog or anything else. The operator does it once, by hand:

1. Create the scheduled task with `task.md` as its prompt, with the checkout filled in and Filing left at `dry-run`.
2. Fill in `com.memvara.nightly-watchdog.plist.template` and load it, as its comment describes, with the watchdog's deadline a few hours after the task's start.
3. Create the `nightly-break` label.
4. Watch one night as a supervised dry run: the report, the DID NOT RUN path (run the watchdog by hand for a night with no heartbeat), the canary, and filing against a test label (`filing.py issue --label <a test label> --file`). Only then change Filing to `file`.

Next: [how work is done here](working-here.md), including the review every pull request gets before it merges.
