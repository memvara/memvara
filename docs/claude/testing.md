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
- **Giving the hooks a store.** Pass `server_env={"MEMVARA_DB": ..., "MEMVARA_USER": ...}` and the runner writes the host's client config, which is where the hooks look for the store. Without it, the hooks report "not configured". The runner writes client configs as JSON only. Codex keeps its config in TOML, so a Codex run with a store is refused, with that reason, until the hook-conformance tests add a TOML writer.
- **What a run returns:** the exit code, the parsed reply and the elapsed time. A hook that runs past its host's time limit raises `HookTimeout`, with what it had printed so far.
- **`capture` is refused for now.** It can start the real agent CLI to extract facts, which would reach the network and spend money. The hook-conformance tests will put stub CLIs first on `PATH`, and until then `run("capture")` raises `NotImplementedError`.
- **Non-JSON output fails the test.** A hook that prints something other than JSON raises `HookOutputError`, because on a real client that output would desynchronise the conversation.

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
- **The state machine** (`tests/adversarial/model/test_adv_model_machine.py`) draws operations at random from small pools, so they often collide, and a second `remember` rule writes only `lives_in` at two instants, so that two values often begin at the same moment. It also checks that no step closes both clocks of an existing row, that a closed clock never moves later and an ending is never cleared, and that rows disappear from the store only through erasure. Each run keeps its store in a file of its own, and when the run ends, that file must pass the integrity checks described at the end of this section. The machine makes 30 short runs in the fast tier, which take about 13 seconds on a laptop, and at most 300 runs in the nightly and weekly tiers; the nightly tier's 300 runs of 100 steps took 10 minutes 10 seconds on a laptop with nothing else running.
- **Known bugs are steered around.** An operation that would trigger an open bug is skipped, so the machine keeps finding new bugs instead of the known ones. Today this is #275 (a future-dated retraction's tombstone ends before it begins), which the machine found and `test_adv_known_bugs.py` pins. The test prints how many steps it ran and how many it skipped for each bug (`pytest -s` shows the line). It fails if skips pass a tenth of all steps, and `STEERED` may name only a bug still in the registry, so a fixed bug's detour cannot outlive it.
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
- `HookRunner` still refuses `run("capture")`. The fakes' own tests call the capture hook's extraction code directly instead. The hook-conformance workstream, A2 in [the test-suite design](../superpowers/specs/2026-09-25-adversarial-test-suite-design.md), will lift the refusal and run the whole hook against these fakes.
- The fakes are POSIX shell scripts. On Windows a program that another starts without a shell is found on `PATH` only as an `.exe`, so their tests skip there.

## Documentation that must match the code

`tests/adversarial/docs/` checks that the text a person or a model reads about memvara names only what the code has. Each check parses the text and reads the truth from the code, so no test keeps a list of what the text mentions, and a new tool, argument or switch is checked without anyone adding it. The plan is `docs/superpowers/plans/2026-09-26-adversarial-docs.md`.

- **Every server configuration.** A switch can hide a tool, remove an argument or rewrite a description, so a description that is right on the default server can be wrong on another. `surface.py` therefore starts a server in the test process for each configuration: the default, each feature switched away from its default, read-only, and anchored by default. It reads each server's `tools/list`. The configurations are built from `FEATURE_DEFAULTS`, so a new switch is covered without being added by hand.
- **Tool descriptions.** `test_adv_docs_tools.py` checks every description any configuration serves, and the instructions the server sends when a client connects. Every tool name in them must be a tool. Every snake_case word must be a tool, an argument of some tool, or a built-in predicate or alias, unless the text marks it as example data. A phrase that ties an argument to a tool, such as "memory_recall with include_episodes", must name an argument that tool takes.
- **How a mention is recognised.** `mentions.py` holds the rules, and its docstring lists them together with what they cannot see. The main limit is that a one-word argument such as `reason` or `query`, written alone as a plain word, is not read as a mention, because it is usually English: "a false reason" is about a stored reason, not about the argument.
- **Each rule is first shown catching its fault** on planted text, and the real-data tests also assert that the parse read something. A parser that stopped reading would therefore fail instead of passing on nothing.

Next: [how work is done here](working-here.md), including the review every pull request gets before it merges.
