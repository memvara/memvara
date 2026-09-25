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

A tier that a run does not select is left out when pytest collects, so it is never imported and never reported as skipped. A file you name on the command line is always collected, whatever its folder.

**The outermost tier folder decides.** Do not put one tier folder inside another, or anywhere under `tests/live`; `test_adv_tiers.py` refuses both. Every run prints its tier and the tier folders it left out, for example `tier fast; left out 3 tier folders: ...`, so a folder that happens to share a tier's name cannot drop out of the run unnoticed.

**Every tier folder needs an `__init__.py`.** Without one, two test files with the same name in different folders collide. `test_adv_tiers.py` checks this.

The files named `test_adv_*_tier_guard.py` fail if their folder is ever collected by a tier that should have left it out. The ordinary fast run is therefore the proof that the tiers work.

## Skips

**A skip needs a rule.** This applies to every test in the repository, not only the adversarial suite, because `tests/conftest.py` registers the ledger for every run. Every skip reason must match a rule in `tests/harness/skips.py`, and each rule says why that skip hides no failure. A skip with no matching rule fails the whole run, and the run lists the test and its reason.

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

## Hooks

`harness.hooks.HookRunner` runs `plugin/hooks/run.py <hook> --host <host>` in a child process, with the stdin payload that host sends.

- **In a test,** use the `hook_runner` fixture: `hook_runner("claude").run("approve", tool_name="mcp__memvara__memory_search")`.
- **Giving the hooks a store.** Pass `server_env={"MEMVARA_DB": ..., "MEMVARA_USER": ...}` and the runner writes the host's client config, which is where the hooks look for the store. Without it, the hooks report "not configured". The runner writes client configs as JSON only. Codex keeps its config in TOML, so a Codex run with a store is refused, with that reason, until the hook-conformance tests add a TOML writer.
- **What a run returns:** the exit code, the parsed reply and the elapsed time. A hook that runs past its host's time limit raises `HookTimeout`, with what it had printed so far.
- **Non-JSON output fails the test.** A hook that prints something other than JSON raises `HookOutputError`, because on a real client that output would desynchronise the conversation.

## Stores in the test process

`harness.stores.memory()` gives an in-memory store and `harness.stores.file(path)` a SQLite file. Both use the hashing embedder and no model, as every other test here does.

A server started on the same file with the child environment opens it in the same vector space. So a test can write through the library and then read through the server, or the other way round.

## Known bugs and security findings

**A bug the suite finds lands at once as a failing test marked `xfail(strict=True)`.** The marker cites a GitHub issue, and the fix follows in its own PR.

- **Strict mode keeps the marker honest.** When the fix lands, the test starts passing and strict mode fails the run until the fix PR removes the marker.
- **Nothing is weakened.** Never skip, delete or weaken a test to make the run green.

**A finding that falls under the in-scope list in `SECURITY.md` never goes into a public issue or a public test.** It goes to a private draft advisory on GitHub, and its failing test lands together with its fix.

Next: [how work is done here](working-here.md), including the review every pull request gets before it merges.
