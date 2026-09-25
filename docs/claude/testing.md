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

Next: [how work is done here](working-here.md), including the review every pull request gets before it merges.
