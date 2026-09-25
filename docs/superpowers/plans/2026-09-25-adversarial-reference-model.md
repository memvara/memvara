# Reference model and state machine (D1, first stage) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drive memvara's write and read API through thousands of random operation sequences, and check every read against a small, separate model of what the store must hold. A difference is a data-loss or history bug, reported as the shortest sequence that reproduces it.

**Architecture:**
- `tests/harness/model.py` is a pure-Python model of claims over both clocks. It mirrors the rules the code documents: `Reconciler.apply`, `close_out`, `state_predicate`, `forget`, `delete`, `erase` and expiry.
- A Hypothesis `RuleBasedStateMachine` applies the same operations to the model and to a real store, then checks a list of invariants after every step.
- Stamps the model cannot know in advance, which come from the wall clock, are checked against a window and then copied from the store. The model predicts which rows change and how. It does not predict the exact microsecond.

**Tech Stack:** Python 3.10–3.13, pytest 9, Hypothesis ≥ 6.100 (already in `[dev]`), memvara's in-memory SQLite store with `HashingEmbedder` and `NullLLM`.

**Spec:** `docs/superpowers/specs/2026-09-25-adversarial-test-suite-design.md`, section "Phase 1: The data-loss hunt", D1.

## Global Constraints

- Everything runs offline, with no API key: `HashingEmbedder`, `NullLLM`, the stores in `tests/harness/stores.py`.
- Time is controlled by passing explicit datetimes, never by patching the clock.
- Found bugs land as `xfail(strict=True)` tests that cite an issue, and a security-class finding goes to a private advisory instead. No test is skipped, deleted or weakened to make the gate green.
- Tests live under `tests/adversarial/`, and a test file's name starts with `test_adv_`. The fast tier adds at most about 3 minutes to CI in total.
- Write plainly: every sentence must be understood on its first reading.

## Scope of this first stage

- **One scope per user.** There are two users, so that isolation between users is checked. Projects, agents and sessions come in the second stage, together with the #266 fix, which changes how those scopes interact.
- **Operations:**
  - `remember`: positive and retraction, with `valid_from`, `recorded_at`, `confidence`, `close` and `expires_at`;
  - `forget`, `delete`, `erase` and `erase_expired`.
- **Reads:** `get_all` in each state and at several pairs of instants, `count`, `history`, `get`, `stats` and `search`.
- **Left for later, and listed in the model's docstring:** `supersede` and `replaces`, links, documents, two handles, reopening, an encrypted store, graph traversal, ranking quality, and `ask`'s reconstructed readings.

## Review Focus

1. **A retirement that moves `invalidated_at` later on an already retired claim.** It would rewrite when belief stopped. The model keeps a closed clock fixed, so any later move fails an invariant.
2. **A retraction dated after the present.** The tombstone sets `valid_to = t`, which is before its own `valid_from`. That is an inverted interval, and the invariants flag it.
3. **A future-dated claim.** It is recorded now and is in force later, so it belongs to none of the three states. `stats()` counts must not sum to `claims`.
4. **A receipt that under-reports a retraction.** A replayed retraction reinforces its tombstone and still reports an empty receipt. The model compares rows, never receipts alone.
5. **Expiry.** An expired row must vanish from every read, including a read at a past instant, and `erase_expired` must leave an audit row for each erasure.

---

### Task 1: The clock and the instant pool

**Files:**
- Create: `tests/harness/clock.py`
- Test: `tests/adversarial/model/test_adv_clock.py` (and `tests/adversarial/model/__init__.py`)

**Interfaces:**
- Produces: `EPOCH: datetime` (2026-01-01 UTC); `INSTANTS: tuple[datetime, ...]`, fixed past instants one month apart from `EPOCH`, in order; `FAR_FUTURE: datetime` (2100-01-01 UTC); `within(value, before, after) -> bool`.

- [ ] **Step 1: Write the failing test.**
  - The instants are UTC-aware and strictly increasing.
  - Every instant is before today and after `EPOCH` minus a year.
  - `within` includes both ends.
- [ ] **Step 2: Run it and watch it fail on the missing module.**
- [ ] **Step 3: Write `clock.py`.**
- [ ] **Step 4: Run it and watch it pass.**
- [ ] **Step 5: Commit.**

### Task 2: The model's rows and reads

**Files:**
- Create: `tests/harness/model.py`
- Test: `tests/adversarial/model/test_adv_model_reads.py`

**Interfaces:**
- Produces:
  - `Row`, a dataclass with these fields: `id: str`, `user: str`, `subject: str`, `predicate: str`, `obj: str`, `polarity: int`, `confidence: float`, `valid_from: datetime`, `valid_to: datetime | None`, `recorded_at: datetime`, `invalidated_at: datetime | None`, `expires_at: datetime | None`, `observations: int`.
  - `ReferenceStore.rows: dict[str, Row]`.
  - `ReferenceStore.visible(user, *, valid_at=None, known_at=None, states=("live",), now) -> set[str]`.
  - `ReferenceStore.stats(now) -> dict[str, int]`.
  - `ReferenceStore.history(user, subject, predicate) -> list[str]`.

The rules are copied from `memvara/store/base.py::state_predicate`:
- The belief floor is `recorded_at <= K`.
- A row is not retired when `invalidated_at` is `None` or later than `K`.
- **live:** `valid_from <= V` and `valid_to` is `None` or later than `V`.
- **ended:** `valid_to` is set and `valid_to <= V`.
- **live or ended:** `valid_from <= V`.
- **retired:** `invalidated_at` is set and `invalidated_at <= K`.
- **All three states** is the belief floor alone.
- Every read also drops rows with `expires_at <= now`, which is the wall clock.

- [ ] **Step 1: Write failing tests that pin the documented examples.**
  - **The stats table** in INTERNALS "`stats()`": one live claim, one ended, one that ended and was later retired, and one in force only next year. That gives `claims=4, live_claims=1, ended_claims=1, invalidated=1`.
  - **The belief floor:** the audit view readmits the future-dated row.
  - **Rome and Berlin** from INTERNALS: `get_all(as_of=2026-03-15)` is empty.
  - **A collapse** (`valid_to == valid_from`) is true at no instant.
- [ ] **Step 2: Watch the tests fail.**
- [ ] **Step 3: Implement the rows, `visible`, `stats` and `history`.**
- [ ] **Step 4: Watch the tests pass.**
- [ ] **Step 5: Commit.**

### Task 3: The model's writes, checked against the real store on hand-written cases

**Files:**
- Modify: `tests/harness/model.py`
- Create: `tests/harness/drive.py`
- Test: `tests/adversarial/model/test_adv_model_writes.py`

**Interfaces:**
- Produces:
  - `ReferenceStore.remember(op, t) -> Expect`;
  - `forget(op, t) -> Expect`, `delete(op, t) -> Expect`, `erase(op) -> Expect` and `erase_expired(now) -> Expect`;
  - `Expect`, which lists the new rows (fact or tombstone), the reinforced ids, and the closures by id and clock;
  - `drive.Pair`, a real store and a model with the same operations applied, with `apply(op)` and `check()`.

The write rules are copied from `Reconciler.apply` and `close_out`:
- **Exact duplicate.** A live, unexpired row with the same user, subject, predicate, object and polarity is reinforced: its observations go up by one.
- **A single-valued predicate.** Live rows in the slot with another value are split by valid time. Those that start strictly after the candidate are "newer", and they cut the candidate's `valid_to` to their earliest start. The others are closed, at the candidate's `valid_from` when `close="ended"`, with `max(at, valid_from)` as the edge, or at `t` when `close="retired"`. A row is closed only if the candidate's confidence is at least half the row's; a row that is not becomes a dispute.
- **A retraction.** It ends the live rows with the named object and writes a tombstone closed on both clocks at `t`. With nothing to match, it reinforces a prior tombstone with the same value. Failing that, it writes nothing when other values are live, and writes a tombstone otherwise.
- **`forget`** retires every live row in the slot at `t`.
- **`delete`** retires one row by id at `t`, or ends it when `close="ended"`.
- **`erase`** removes a row, and **`erase_expired`** removes every row with `expires_at <= now`.

- [ ] **Step 1: Write the failing tests.** Each case applies a short script to a `Pair` and calls `check()`:
  - supersession;
  - a backfill that is older than the live row;
  - a collapse;
  - a dispute;
  - a retraction, with a match, a replay, a mismatch and an empty slot;
  - a `MANY` accumulation;
  - `forget`, `delete` in both closings, `erase`, and an expiry.
- [ ] **Step 2: Watch them fail.**
- [ ] **Step 3: Implement the model's writes and `drive.Pair`.** `Pair.check()` compares:
  - every row's state and both clocks, copying wall-clock stamps from the store after checking that they fall inside the operation's window;
  - `get_all` in every state and at every pair of instants from `INSTANTS`;
  - `count`, `history`, `get` and `stats`.
- [ ] **Step 4: Watch them pass.** If a case shows the real store breaking a documented rule, that case becomes a strict xfail with an issue, per the Global Constraints.
- [ ] **Step 5: Commit.**

### Task 4: Replayable programs

**Files:**
- Modify: `tests/harness/drive.py`
- Test: `tests/adversarial/model/test_adv_replay.py`

**Interfaces:**
- Produces:
  - `Op`, a dataclass;
  - `replay(ops: list[Op]) -> None`, which raises `ModelDivergence(index, op, detail)` at the first step where the store and the model disagree;
  - `format_program(ops) -> str`, which prints a list that `replay` accepts.

- [ ] **Step 1: Write the failing tests.**
  - `replay` of a clean program returns nothing.
  - A program run against a deliberately broken model raises `ModelDivergence` naming the step.
- [ ] **Step 2: Watch them fail.**
- [ ] **Step 3: Implement it.**
- [ ] **Step 4: Watch them pass.**
- [ ] **Step 5: Commit.**

### Task 5: Store integrity checks

**Files:**
- Create: `tests/harness/invariants.py`
- Test: `tests/adversarial/model/test_adv_integrity.py`

**Interfaces:**
- Produces: `check_store_integrity(path) -> list[str]`, which returns the problems found and an empty list for a healthy store. It checks:
  - SQLite's `PRAGMA integrity_check`;
  - the FTS5 `integrity-check`;
  - that every claim row has its vector slot;
  - that every `claim_sources` row points at an existing episode or at a recorded erasure.

- [ ] **Step 1: Write the failing tests.**
  - A healthy file store gives `[]`.
  - A file with a deleted FTS row, or with a dangling source, gives a named problem.
- [ ] **Step 2: Watch them fail.**
- [ ] **Step 3: Implement it.**
- [ ] **Step 4: Watch them pass.**
- [ ] **Step 5: Commit.**

### Task 6: The state machine

**Files:**
- Create: `tests/adversarial/model/test_adv_model_machine.py`

**Interfaces:**
- Consumes: `drive.Pair`, `Op`, `clock`, `invariants`.

It has one rule per operation, with values drawn from small pools so that collisions are common. The invariants are checked after every step:
- **I1:** no step closes both clocks of an existing row.
- **I2:** a closed clock never moves later.
- **I3:** rows disappear only through `erase` or `erase_expired`.
- **I4:** every read matches the model.
- **I5:** the belief floor holds: no row read at `K` was recorded after `K`.
- **I6:** a write that adds a row can be read back in the present, unless it is future-dated, expired or collapsed.
- **I7:** an erased row is absent from every read, and its audit row exists.
- **I8:** one user's reads never contain the other user's rows.
- **I9:** `stats()` matches the model.
- **I10:** every receipt reports zero model calls.
- **I11:** a predicate nobody declared never closes anything.
- **I12:** `search(k)` returns at most `k` rows, and all of them are visible.
- **I13:** no interval ends before it starts.
- **I14:** the store passes `check_store_integrity` at teardown.

The machine runs under the tier's Hypothesis profile, and `--tier nightly` raises the counts.

- [ ] **Step 1: Write the machine.**
- [ ] **Step 2: Run it in the fast tier. Every failure is investigated:** a model error is fixed in the model; a store bug becomes an issue with a strict xfail, or a private advisory if it is security-class.
- [ ] **Step 3: Run it in the nightly tier once, locally.**
- [ ] **Step 4: Commit.**

### Task 7: Proof that the machine catches real mistakes, and the docs

**Files:**
- Modify: `docs/claude/testing.md` (a new section, "The reference model"), `CONTRIBUTING.md` (the test counts)

- [ ] **Step 1: Run three mutants in a scratch copy, and record that the machine fails on each:**
  - the belief floor is dropped from `state_predicate`;
  - `_is_after` is made non-strict;
  - `close_out` closes both clocks.
- [ ] **Step 2: Write the docs section,** covering what the model checks, what it leaves out, how to replay a failure, and the mutant results.
- [ ] **Step 3: Run the full gate, then open the PR and review it.**
