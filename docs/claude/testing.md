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

The `--tier` option, and the filter that leaves the other tiers out, are in the `conftest.py` at the repository root. pytest reads that file on every run, so `--tier` works whatever paths a run is given, `pytest memvara --tier nightly` included. The doctests in `memvara/` are fast tests, so a local or quarantine run leaves them out. `pyproject.toml` puts `tests/` on the import path, so that the root file can import `harness`.

**The outermost tier folder decides.** Do not put one tier folder inside another, or anywhere under `tests/live`; `test_adv_tiers.py` refuses both. Every run prints its tier and the tier folders it left out, for example `tier fast; left out 3 tier folders: ...`, so a folder that happens to share a tier's name cannot drop out of the run unnoticed.

**Every folder of the suite needs an `__init__.py`,** not only the tier folders. Without one at every level, pytest imports the modules below that folder under names of their own: two test files with the same name collide, and a tier folder's tests fail to import. `test_adv_tiers.py` checks every folder under `tests/adversarial` and `tests/live`.

The files named `test_adv_*_tier_guard.py` fail if their folder is ever collected by a tier that should have left it out. The ordinary fast run is therefore the proof that the tiers work.

## Skips

**A skip needs a rule.** This applies to every test in the repository, not only the adversarial suite, because `tests/conftest.py` registers the ledger for every run. Every skip reason must match a rule in `tests/harness/skips.py`, and each rule says why that skip hides no failure. A skip with no matching rule fails the whole run, and the run lists the test and its reason.

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

Next: [how work is done here](working-here.md), including the review every pull request gets before it merges.
