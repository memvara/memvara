# Concurrency and crash tests (D2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show that a memvara store keeps every write it acknowledged, and is never left half-written or holding two live values for one single-valued slot, when two handles or two processes use one file at once, or when a process is killed at any of ten named points.

**Architecture:**
- `tests/harness/crash_child.py` is a script that a test runs as a child process. It opens a store, runs setup operations and prints the id of each one it finishes, installs a pause at one named point by patching memvara in its own process, and runs one more operation. When the operation reaches the point, the child prints a marker line. It then either waits to be killed, or waits for the line `go` on its standard input and carries on.
- `tests/harness/crash.py` starts the child with `harness.env.child_env`, reads its lines with a timeout (reader threads, so it also works on Windows), kills it at the marker or releases it, and hands the test the acknowledged ids. `crash.after_crash(...)` then checks the store the way the spec defines a correct recovery.
- Two processes are ordered deterministically: one child waits at its point while the test process writes through its own handle, and only then is the child released. No test depends on timing.
- No library code changes. Every pause is a patch the child applies to itself.

**Tech Stack:** Python 3.10–3.13, pytest, `subprocess`, `sqlite3`, `harness.env.child_env`, `harness.stdio.McpProcess`, `harness.stores.file`, `harness.invariants.check_store_integrity`.

**Spec:** `docs/superpowers/specs/2026-09-25-adversarial-test-suite-design.md`, section "Phase 1: The data-loss hunt", D2.

## Global Constraints

- Everything runs offline, with no API key. Stores use `HashingEmbedder(dim=512)` and `NullLLM`, which is what `harness.stores.file` and a server started with `child_env` both use, so a store written by one can be opened by the other.
- Every child process gets its environment from `harness.env.child_env`.
- No fault hooks are added to the library. The child process patches internals itself.
- Found bugs land as `xfail(strict=True)` tests that cite an issue, and a security-class finding goes to a private advisory instead. No test is skipped, deleted or weakened to make the gate green.
- **Flake budget.** A fast-tier test in this plan lands only after it passes 200 of 200 runs locally and 100 of 100 runs under CPU load, and then three CI runs in a row. A test that fails in CI and does not reproduce moves to `quarantine/` with an issue.
- The fast-tier tests of this plan take at most 30 seconds in total on a laptop.
- Tests live under `tests/adversarial/concurrency/`, and a test file's name starts with `test_adv_`.
- Write plainly: every sentence must be understood on its first reading.

## What "correct after a crash" means

After a kill, the store is opened again by a process that never had it open, and these hold:
1. The file passes `check_store_integrity`.
2. The interrupted operation is all or nothing. Each kill point states which one it must be.
3. Every write the child acknowledged before the kill is present: `get` returns it and `search` finds it by its object.
4. For every acknowledged claim, exactly one of these is true: the claim exists, or its erasure record exists.
5. The next write succeeds within one second, which shows that the killed process left no lock behind.

## The ten kill points

The research for this plan read the code for each point and checked two of them with a real kill. Each point names the method the child patches, and whether it pauses before or after calling the original.

| Point | Tier | Patch | The interrupted operation must be |
|---|---|---|---|
| `after-episode` | fast | after `SQLiteStore.add_episode`, inside `add()` | absent: no episode and no `episodes_fts` row |
| `after-claim` | fast | after `SQLiteStore.put_claim`, in `remember()` on an empty slot | absent |
| `after-vector` | fast | after `SQLiteStore._set_vector` | absent, claim and embedding row both |
| `erase-before-delete` | fast | before `SQLiteStore._erase_row`, in `erase()` | undone: the claim is whole and has no erasure record |
| `between-migrations` | fast | before `SQLiteStore._migrate_to_v7`, opening a version-2 store | undone: `user_version` is still 2, no table that existed changed, and the next open migrates fully and indexes the old episode |
| `after-commit` | fast | after `remember()` returns | present, with its embedding and text-index rows |
| `vecs-growth` | nightly | before `_VecIndex._remap`, writing the 257th vector | absent; the 256 acknowledged claims are all found by `search` |
| `fingerprint-write` | nightly | inside `json.dump` in `embed/fingerprint.py`, after a few chunks | the torn sidecar is rewritten by the next open |
| `encrypt-between-renames` | nightly | after the first `os.replace` in `encrypt_store` | the next open with the key rebuilds the vector file; every claim is present |
| `document-before-claims` | nightly | before `DocumentService._finish`, in `add_document` | the document and its chunks are present, and no claim cites them |

`remember()`'s claim and vector writes share one transaction, so a kill at `after-claim` or `after-vector` must leave nothing. The `.vecs` file is not part of SQLite's transaction, so a kill at `after-vector` can leave vector bytes in a slot no row names. That is expected; the Review Focus below pins what must follow from it.

## Review Focus

1. **Vector bytes left in a slot no row names.** A kill at `after-vector` can leave them. The next write that takes that slot must overwrite them, and no search may ever return them.
2. **A lock held by a killed process.** SQLite's locks are released when the process dies. The next writer must not wait out the five-second busy timeout.
3. **A crash while a brand-new store is being created.** The schema and the migrations are one transaction. The next open must create the store cleanly, not fail on a half-made file.
4. **Files a killed `encrypt_store` leaves behind.** Its temporary copies are removed only by an exception handler, which a kill skips. A second `encrypt_store` must still work, and must report the store as already encrypted.
5. **A document stopped before its claims were extracted.** It stays `queued`. Whatever the next open does, a reader must be able to tell that the document is unfinished; a document that reads as complete but has no vectors or claims is a finding.

---

### Task 1: The crash child and its driver

**Files:**
- Create: `tests/harness/crash_child.py`, `tests/harness/crash.py`
- Test: `tests/adversarial/concurrency/test_adv_crash_harness.py` (and `tests/adversarial/concurrency/__init__.py`)

**Interfaces:**
- Consumes: `harness.env.child_env(home, extra)`.
- Produces:
  - `crash_child.py`, run as `python tests/harness/crash_child.py`, reads one JSON program from its standard input: `{"db": str, "user": str, "setup": [op, ...], "point": str | null, "action": op | null, "hold": bool}`. An op is `[name, arguments]`, where the name is one of `remember`, `add`, `erase`, `delete`, `add_document`, `open` and `encrypt`, and `erase` and `delete` may name an earlier op's result as `{"ref": index}`. For each setup op it prints `ACK <json>` with the op's index and the ids it produced. At the point it prints `POINT <name>`. With `hold`, it then reads one line and carries on when that line is `go`; without it, it sleeps until killed. When the action returns it prints `DONE <json>`.
  - `crash.Child(program, *, home, env=None, timeout=20.0)`, a context manager with `wait_for(prefix) -> str`, `acked -> list[dict]`, `release()`, `kill()` and `finish() -> int`. A child that exits early or prints nothing within the timeout raises `CrashHarnessError` with the tail of its standard error.
  - `crash.POINTS`, the names of the ten points.

- [ ] **Step 1: Write the failing tests.**
  - A child with no point runs its setup, prints one `ACK` per op, prints `DONE`, and exits 0.
  - A child at `after-commit` prints `POINT after-commit`; after `kill()`, its exit code is not 0 and the store opens.
  - A held child carries on after `release()` and prints `DONE`.
  - A child whose point is never reached makes `wait_for("POINT")` raise `CrashHarnessError` within the timeout, naming the point and quoting standard error.
  - An unknown point name is refused by the child with a message, not ignored.
- [ ] **Step 2: Run them and watch them fail** on the missing modules.
- [ ] **Step 3: Write `crash_child.py` and `crash.py`.** The child imports only memvara and the standard library, so it does not depend on the test package's layout.
- [ ] **Step 4: Run them and watch them pass,** on this Mac and under `--count`-style repetition (Task 9 runs the full budget).
- [ ] **Step 5: Commit.**

### Task 2: The six fast kill points, and what recovery must look like

**Files:**
- Modify: `tests/harness/crash.py` (add `after_crash`)
- Test: `tests/adversarial/concurrency/test_adv_kill_points.py`

**Interfaces:**
- Consumes: `crash.Child`, `crash_child` programs, `check_store_integrity`, `harness.stores.file`.
- Produces: `crash.after_crash(path, acked_claims, *, erased=(), key=None) -> Memvara`, which runs checks 1, 3, 4 and 5 of "What correct after a crash means", then returns an open handle for the test's own check 2.

- [ ] **Step 1: Write the failing tests,** one per fast point, each with two acknowledged setup writes before the interrupted one:
  - `after-episode`: the episode and its text-index row are gone.
  - `after-claim` and `after-vector`: the claim and its embedding row are gone.
  - `erase-before-delete`: the claim is whole, it is still found by `search`, and it has no erasure record.
  - `between-migrations`: `PRAGMA user_version` reads 2, and the next open migrates to the current version and keeps the version-2 episode.
  - `after-commit`: the claim is present and found by `search`.
  - Review Focus 1: after a kill at `after-vector`, twenty more writes are made; `search` for each returns its own claim first, and never a row that does not exist.
  - Review Focus 2: in every test, the next write after the kill finishes within one second.
  - WAL recovery: a child acknowledges two writes, then makes three more inside one `batch()` and is killed before the batch ends; the two are present and the three are not.
- [ ] **Step 2: Run them.** Each must fail only on the missing `after_crash`, and never on a store problem. A store problem is a finding: stop and handle it as the Global Constraints say.
- [ ] **Step 3: Write `after_crash`.**
- [ ] **Step 4: Run them and watch them pass.**
- [ ] **Step 5: Commit.**

### Task 3: Two handles on one file

**Files:**
- Test: `tests/adversarial/concurrency/test_adv_two_handles.py`

- [ ] **Step 1: Write the tests.**
  - Two handles are opened on one file. A write through the first is returned by the second's `get_all`, `get` and `search`, including `search` by the object's text, which goes through the vector index. The second handle is never reopened.
  - The same holds when the second handle reads from another thread, because the store tracks what each thread last saw.
  - An erase through the first handle removes the claim from every read of the second.
  - **Reads while another writer holds the lock:** a held child stops at `after-claim`, holding the write lock with its claim uncommitted. The test's own handle then reads. Every read returns within one second, and none returns the child's claim. After `release()`, the next read returns it.
- [ ] **Step 2: Run them.** A failure is a finding.
- [ ] **Step 3: Commit.**

### Task 4: The two races

The spec names two races to test: two writers on one single-valued slot, and a `delete()` whose read and write another writer could separate. Each test holds a child at a pause point while the test writes through its own handle, then releases it, so the interleaving is the same on every run.

A race in these paths would fall under `SECURITY.md`'s audit-trail section, so these two tests are developed in private and land together with anything they need. No public task depends on them.

### Task 5: Many threads on one handle

**Files:**
- Test: `tests/adversarial/concurrency/test_adv_threads.py`, and `tests/adversarial/concurrency/nightly/test_adv_threads_nightly.py` (with `nightly/__init__.py`)

- [ ] **Step 1: Write the test.** Four threads (sixteen in the nightly tier) each make 25 operations (500 nightly), drawn from a seeded random generator: `remember`, `forget`, `delete` and `erase` over two users and the model's three predicates. Every call's result is recorded. At the end: the file passes `check_store_integrity`, no single-valued slot has two live values, every acknowledged write that was not later closed is live, and every erased claim has an erasure record.
- [ ] **Step 2: Run it.** A failure is a finding.
- [ ] **Step 3: Commit.**

### Task 6: Two real servers on one database file

**Files:**
- Test: `tests/adversarial/concurrency/test_adv_two_servers.py`

- [ ] **Step 1: Write the test.** Two `McpProcess` servers share one store file and one user. They take turns calling `memory_remember` for ten distinct facts each, then each reads all twenty back with `memory_search`. No acknowledged write is lost, no fact is stored twice, and the file passes `check_store_integrity` after both servers close.
- [ ] **Step 2: Run it.** A failure is a finding.
- [ ] **Step 3: Commit.**

### Task 7: Damaged vector and fingerprint files, and a store opened while another process creates it

**Files:**
- Test: `tests/adversarial/concurrency/test_adv_damaged_files.py`

- [ ] **Step 1: Write the tests.**
  - `.vecs` truncated to half its length: the next open rebuilds it from the database, and `search` finds every claim.
  - `.vecs` with its header overwritten: the same.
  - `.vecs` deleted: the same.
  - `<db>.embedder.json` replaced by invalid JSON on a store that already has vectors, then the store opened with a different hashing embedder of the same width: the change must be detected (a warning or a refusal), because otherwise every search compares two unrelated vector spaces. Nothing is detected today, and the record is never rewritten: #280, pinned as a strict expected failure, beside a control case that shows the warning with the record intact.
  - A new store opened while another connection holds the write lock on the new file must wait for it, like any other write. Today it fails within a millisecond with "database is locked", because switching a new file to WAL mode needs a lock upgrade SQLite will not wait for: #281, pinned as a strict expected failure. The nightly tier opens one new store from three processes at once, 60 times, and pins the same issue.
- [ ] **Step 2: Run them.** A failure is a finding.
- [ ] **Step 3: Commit.**

### Task 8: The nightly tier

**Files:**
- Modify: `tests/harness/invariants.py` (`check_store_integrity(path, key=None)`, which opens an encrypted store with the key through `sqlcipher3`)
- Test: `tests/adversarial/concurrency/nightly/test_adv_kill_points_nightly.py`, `test_adv_server_kills.py`, `test_adv_server_pressure.py`, `test_adv_full_disk.py`

- [ ] **Step 1: Write the tests.**
  - The four nightly kill points in the table, each checked as in Task 2. Review Focus 4 and 5 are pinned here: a second `encrypt_store` after a kill, and what a reader sees of the unfinished document. Review Focus 3, a kill while a new store is being created, is pinned in the fast tier, because it costs a fraction of a second.
  - **200 random kills of a real server.** Each round starts a server on the same store, makes a random number of calls, and kills it at a random moment. A call counts as acknowledged when its reply arrived. After every kill, the recovery checks run on the store.
  - **Two servers under pressure.** 200 calls each, mixing writes and reads. While the reconcile race from Task 4 is open, the two servers never write the same single-valued slot, and the test counts the writes it steered away, as the model's state machine does.
  - **Lock timeout.** A held child stops inside `remember()` for a name it has not seen, which holds the write lock, and keeps it for seven seconds. A write through the test's own handle raises `sqlite3.OperationalError` ("database is locked") after the five-second busy timeout, and not much later. A server's `memory_remember` returns an error result within the busy timeout plus one second, and the server still answers `ping` afterwards. Both pin one writer at a time as documented behaviour, so that a change to the error or the bound is noticed.
  - **Full disk.** A child lowers `RLIMIT_FSIZE` to just above the store's size and ignores `SIGXFSZ`, then writes until a write fails. The failure must be an exception, not a hang and not a silent loss. The store, opened again without the limit, passes the recovery checks. POSIX only; the nightly tier runs on the maintainer's Mac.
  - **Server and hook daemon.** This moves to the hook-conformance work (A2). `harness.hooks.HookRunner` runs every hook with the daemon switched off, and the capture hook, the one that writes, is refused until A2 puts stub CLIs on `PATH`. A2 adds both, and this test with them; the D2 pull request says so.
- [ ] **Step 2: Run the nightly tier once, locally.** Failures are handled as findings.
- [ ] **Step 3: Commit.**

### Task 9: The flake budget, the documentation, and the review

**Files:**
- Modify: `docs/claude/testing.md` (a new section, "Concurrency and crashes"), `README.md` and `CONTRIBUTING.md` (the test counts), `CHANGELOG.md` if the suite's entry needs updating

- [ ] **Step 1: Run every fast test of this plan 200 times locally, and 100 times while four processes keep every core busy.** Record the counts in the PR body. A single failure means a test that is not ready: find the cause and fix it before landing.
- [ ] **Step 2: Write the docs section:** the crash child, the ten points and the tier of each, what recovery must look like, the races and what they found, and how to run one point by hand.
- [ ] **Step 3: Run the full gate, open the PR, run the code review, and fix what it finds.** After the merge, watch the next three CI runs of `main` for any failure in these tests.
