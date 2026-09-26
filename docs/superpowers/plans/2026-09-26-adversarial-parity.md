# Adversarial suite A5: parity across surfaces Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one list of operations through every surface that offers it, and compare the answers exactly: the library's four clients field by field, and the text the MCP server writes on three surfaces line by line. Where a surface differs on purpose, assert the difference and name where the code documents it.

**Architecture:**

- **One comparison, shared.** `tests/adversarial/parity/compare.py` turns any result into plain dicts and lists. It removes the three things two stores differ in whatever the surface: random ids (labelled by what they name), instants taken from the clock (replaced with `<wall clock>`), and lists that come back in id order (sorted). `differences()` then lists every field that disagrees, with its path and both values.
- **The program is data.** The library program is a tuple of `Step`s and the MCP session a tuple of `Call`s. A module-scoped fixture plays it once on every surface, and every step becomes its own test id, so a failure names the operation and the client.
- **Documented differences have tests of their own.** Each one says where the code documents it and checks that the difference is really there. The step-by-step comparison applies the documented rule to the reference answer, or leaves the step to the documented-difference test.

**Tech Stack:** Python 3.10 to 3.13, pytest, httpx (the `cloud` extra, which CI installs), `harness.fakes.fake_v1.FakeV1`, `harness.stdio.McpProcess`, `harness.env.child_env`.

**Spec:** `docs/superpowers/specs/2026-09-25-adversarial-test-suite-design.md`: the A5 row of "Phase 2: The agent-facing deterministic tiers", and done criterion 8, "Every surface gives equivalent results on the shared scenarios". The workstream brief adds the list of operations (remember, add, search, recall, get, history, why, delete, forget with and without a confirm token, end, documents, stats, standing, profile), the three MCP surfaces, and the rule that a documented difference is asserted with its citation rather than skipped.

## Global Constraints

- **Platforms.** Python `>=3.10`. CI runs 3.10 to 3.13 on Ubuntu, and 3.13 on macOS and Windows. Every test here must pass on all of them.
- **Offline.** `FakeV1` answers through a mock transport or on 127.0.0.1. No test reaches the network.
- **Child processes** get their environment from `harness.env.child_env`, which `McpProcess` already uses. The in-process server is built from the same `child_env` dictionary, so both local surfaces read the same configuration.
- **Embedder.** Every `Memvara(...)` built in `tests/` passes `embedder=`. `harness.stores` and `build_memvara` both do.
- **Skips.** None are needed. Any skip must match a rule in `tests/harness/skips.py`.
- **Deprecations are errors** (`filterwarnings = ["error::DeprecationWarning"]`).
- **Fast tier.** Everything here is in the fast tier, with a budget of about 6 seconds for the folder.
- **Type checks.** `python -m mypy -p memvara`, and `python -m mypy tests/harness` with and without `--ignore-missing-imports`, stay clean.
- **Prose.** Docstrings, comments, docs and commit messages are plain English that a reader with no context understands on the first read.
- **No AI attribution** anywhere: no trailer, no "generated with" line, no model name.
- **Commits** name their files. Never `git add -A`, `git add .` or `git commit -a`. Never stash, reset, check out or restore a file this plan did not create. Documentation ships in the same commit as the code it describes.
- **Scope.** This plan owns `tests/adversarial/parity/`, this plan file, and the parity section of `docs/claude/testing.md`, added just before that file's final `Next:` line. It does not edit `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md` or `tests/harness/known_bugs.py`.
- **Checklist marks.** `tests/harness/checklist.py` does not exist on `origin/main` when this work starts, so no test carries `@pytest.mark.covers(...)`.
- **Differences.** A difference the code documents is asserted by a test that cites the documentation. An undocumented difference is a bug. It is classified against `SECURITY.md`'s "In scope" section first; either way its failing test stays out of the commits, in `local/`, and it is reported in the final message. memvara/memvara#298 (the hosted client refuses `recall(valid_at=...)`) is already filed, so the dated recall is compared between the local surfaces only and the hosted refusal is not reported again.

## Review Focus

1. **A list whose order comes from random ids makes a comparison flaky.** A confirmed `forget_matching` closes in its token's id order, and a replayed token names the first listed claim in that order. Pinned by `test_a_preview_hides_its_token_and_a_result_sorts_what_it_closed` (Task 1), and by previews of one match in both programs (Tasks 2 and 3), so a refusal names one claim.
2. **A minute boundary between two surfaces makes the text differ.** Every instant a store takes from its clock is replaced, including the start of `memory_profile`'s seven-day window. Pinned by `test_tool_text_loses_the_clock_and_the_token_and_keeps_every_other_instant` (Task 1).
3. **A normaliser that hides a real difference makes every parity test pass.** Pinned by the self-tests in Task 1, each of which pairs a rule with a check that a real difference still shows, and by six deliberate faults run against the finished folder (Task 3).
4. **A documented difference disappears or widens and nobody notices.** Each documented difference has a test that checks it is really there (Tasks 2 and 3).
5. **A step that raises on one surface hides every later step.** `play()` records a refusal as `Raised` and carries on (Task 2), and a tool error is an MCP reply like any other (Task 3).

## File structure

| File | Responsibility |
|---|---|
| `tests/adversarial/parity/__init__.py` | Makes the folder a package, which `test_adv_tiers.py` requires of every folder of the suite. |
| `tests/adversarial/parity/compare.py` | `labels`, `normalise`, `differences`, `assert_same`, `text_labels`, `normalise_text` and `Raised`: the comparison both test files use. |
| `tests/adversarial/parity/test_adv_parity_compare.py` | Shows that each rule of the comparison removes only what it should. |
| `tests/adversarial/parity/test_adv_parity_library.py` | The library program through `Memvara`, `AsyncMemvara`, `RemoteMemvara` and `AsyncRemoteMemvara`, and the hosted clients' documented differences. |
| `tests/adversarial/parity/test_adv_parity_mcp.py` | The tool session through the in-process server, stdio in local mode and stdio in cloud mode, and cloud mode's documented differences. |
| `docs/claude/testing.md` | A "Parity across surfaces" section, before the final `Next:` line. |
| `local/a5-parity-faults/run_fault.py` | Not committed (`local/` is ignored). Runs the folder against one deliberate fault in a scratch copy of the checkout. |

## Commands

Every command runs from the worktree root, `/Applications/workstation/agent-memory/.claude/worktrees/agent-a65501ac388604919`, written `$W` below, with the CI virtual environment's Python, written `$PY`:

```bash
W=/Applications/workstation/agent-memory/.claude/worktrees/agent-a65501ac388604919
PY=/Applications/workstation/agent-memory/.claude/worktrees/friendly-einstein-53c8da/local/venv-ci/bin/python
export PYTHONPATH=$W TMPDIR=/private/tmp/a5-parity-tmp
```

`TEST <paths>` below means `$PY -m pytest -q -p no:cacheprovider <paths>`.

---

## Task 0: Commit this plan

- [ ] **Step 1:** Create the branch from `origin/main` and remove its upstream, so a bare push cannot reach `main`.

```bash
git fetch origin
git switch -c test/adversarial-parity origin/main
git branch --unset-upstream
```

- [ ] **Step 2:** Commit this file alone.

```bash
git add docs/superpowers/plans/2026-09-26-adversarial-parity.md
git commit -m "Plan the adversarial suite's parity tests across surfaces"
```

---

## Task 1: The comparison

**Files:**
- Create: `tests/adversarial/parity/__init__.py`
- Create: `tests/adversarial/parity/test_adv_parity_compare.py`
- Create: `tests/adversarial/parity/compare.py`
- Modify: `docs/claude/testing.md` (a new section before the final `Next:` line)

**Interfaces:**
- Produces, in `compare.py`:
  - `Raised(kind: str, message: str)`, a frozen dataclass for a step that raised.
  - `labels(*results: Any) -> dict[str, str]`: a label for every id in the results.
  - `normalise(value: Any, names: Mapping[str, str], *, near: datetime) -> Any`.
  - `differences(expected: Any, actual: Any, path: str = "") -> list[str]`.
  - `assert_same(expected: Any, actual: Any, what: str) -> None`.
  - `text_labels(texts: list[str]) -> dict[str, str]` and `normalise_text(text: str, names: Mapping[str, str], *, near: datetime) -> str`, for tool text.
  - The constants `IDS`, `BOOKKEEPING`, `CLOCK_WINDOW` (8 days), `WALL_CLOCK` (`"<wall clock>"`) and `TOLERANCE` (`1e-6`).

- [ ] **Step 1: Create the package.** Write `tests/adversarial/parity/__init__.py`:

```python
"""The same operations through every surface, compared. See docs/claude/testing.md."""
```

- [ ] **Step 2: Write the failing self-tests.** Write `tests/adversarial/parity/test_adv_parity_compare.py`:

```python
"""The comparison the parity tests rest on removes only what differs between two stores
for reasons that have nothing to do with the surface, and reports everything else.

A normaliser that removed too much would make every parity test pass whatever the
surfaces returned, so each rule here is paired with a check that a real difference
still shows.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from memvara.types import (CLOSURE, Claim, Document, Episode, ForgetPreview, ForgetResult,
                           WriteReceipt)

from .compare import (WALL_CLOCK, Raised, differences, labels, normalise, normalise_text,
                      text_labels)

NOW = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)
THEN = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _claim(value: str, **fields: Any) -> Claim:
    return Claim(subject="user", predicate="lives_in", object=value, **fields)


def test_the_same_claim_gets_the_same_label_whatever_id_its_store_drew() -> None:
    one, other = _claim("Lisbon"), _claim("Lisbon")
    assert one.id != other.id
    assert (normalise(one.id, labels(one), near=NOW)
            == normalise(other.id, labels(other), near=NOW)
            == "<claim user lives_in Lisbon>")


def test_a_turn_and_a_document_are_labelled_by_what_they_hold() -> None:
    turn = Episode(content="My name is Ada.")
    doc = Document(custom_id="docs/runbook")
    names = labels({"turn": turn, "docs": [doc]})
    assert names == {turn.id: "<turn My name is Ada.>", doc.id: "<document docs/runbook>"}


def test_two_ids_that_would_share_a_label_are_refused() -> None:
    with pytest.raises(ValueError, match="both <claim user lives_in Lisbon>"):
        labels([_claim("Lisbon"), _claim("Lisbon")])


def test_an_id_no_returned_object_names_is_numbered_in_the_order_it_appears() -> None:
    """Two such ids must not share one label, or swapping them would go unseen."""
    claim = _claim("Lisbon")
    names = labels({"sources": ["ep_ffffffffffffffffffff", "ep_0123456789abcdef0123"]},
                   claim)
    assert names == {claim.id: "<claim user lives_in Lisbon>",
                     "ep_ffffffffffffffffffff": "<ep 1>", "ep_0123456789abcdef0123": "<ep 2>"}


def test_an_id_nobody_labelled_is_marked_rather_than_kept() -> None:
    assert normalise("cl_0123456789abcdef0123", {}, near=NOW) == "<unlabelled id>"


def test_an_instant_near_the_run_is_replaced_and_one_far_from_it_is_kept() -> None:
    assert normalise(NOW - timedelta(days=7), {}, near=NOW) == WALL_CLOCK
    assert normalise(NOW + timedelta(minutes=10), {}, near=NOW) == WALL_CLOCK
    assert normalise(THEN, {}, near=NOW) == THEN.isoformat()
    assert normalise(NOW + timedelta(days=30), {}, near=NOW) != WALL_CLOCK


def test_a_claim_is_compared_without_its_bookkeeping_and_with_its_state() -> None:
    claim = _claim("Lisbon", valid_from=THEN, recorded_at=NOW, valid_to=NOW)
    claim.meta["salience_base"] = 1.0
    claim.meta[CLOSURE] = [{"at": NOW.timestamp(), "close": "ended", "by": "api"}]
    out = normalise(claim, labels(claim), near=NOW)
    assert "salience_base" not in out["meta"]
    assert out["meta"][CLOSURE] == [{"at": WALL_CLOCK, "close": "ended", "by": "api"}]
    assert (out["state"], out["salience_base"], out["valid_from"]) == (
        "ended", 1.0, THEN.isoformat())


def test_two_claims_that_differ_in_one_field_still_differ() -> None:
    one = _claim("Lisbon", valid_from=THEN)
    other = _claim("Lisbon", valid_from=THEN, confidence=0.5)
    found = differences(normalise(one, labels(one), near=NOW),
                        normalise(other, labels(other), near=NOW))
    assert found == [".confidence: expected 1.0, found 0.5"]


def test_a_receipt_is_compared_without_how_long_the_write_took() -> None:
    fast = normalise(WriteReceipt(latency_ms=0.2), {}, near=NOW)
    slow = normalise(WriteReceipt(latency_ms=9.0), {}, near=NOW)
    assert "latency_ms" not in fast and differences(fast, slow) == []


def test_a_preview_hides_its_token_and_a_result_sorts_what_it_closed() -> None:
    first, second = _claim("Berlin"), _claim("Lisbon")
    names = labels(first, second)
    preview = ForgetPreview(close="retired", matches={first.id: first.text},
                            confirm="signature", expires_at=NOW)
    assert normalise(preview, names, near=NOW)["confirm"] == "<token>"
    both_ways = [normalise(ForgetResult(close="retired", closed=order), names, near=NOW)
                 for order in ([first, second], [second, first])]
    assert differences(*both_ways) == []


def test_a_difference_names_its_path_and_both_values() -> None:
    found = differences({"added": [{"object": "Lisbon"}], "count": 1},
                        {"added": [{"object": "Porto"}], "count": 2})
    assert found == [".added[0].object: expected 'Lisbon', found 'Porto'",
                     ".count: expected 1, found 2"]


def test_floats_that_differ_only_in_the_ninth_decimal_are_equal() -> None:
    assert differences(0.548188936245, 0.548188930197) == []
    assert differences(0.5481, 0.5482) == [": expected 0.5481, found 0.5482"]


def test_a_value_of_another_type_is_a_difference_even_when_it_compares_equal() -> None:
    assert differences(1, True) == [": expected 1, found True"]
    assert differences(1, 1.0) == [": expected 1, found 1.0"]


def test_a_missing_key_and_a_missing_item_are_both_reported() -> None:
    assert differences({"a": 1}, {}) == [".a: only in the expected answer, 1"]
    assert differences([1, 2], [1]) == [": 2 item(s) expected, 1 found"]


def test_a_refusal_is_compared_by_its_class_and_its_message() -> None:
    one = normalise(Raised("KeyError", "no claim cl_0123456789abcdef0123"), {}, near=NOW)
    assert one == {"__type__": "Raised", "kind": "KeyError",
                   "message": "no claim <unlabelled id>"}


def test_tool_text_is_labelled_from_the_receipt_lines_and_the_order_of_appearance() -> None:
    texts = ["added 1\n+ [cl_0123456789abcdef0123] user likes jazz",
             "document doc_0123456789abcdef0123: done\nturn id(s): ep_0123456789abcdef0123",
             "Claim 'cl_ffffffffffffffffffff' is not visible here."]
    names = text_labels(texts)
    assert [normalise_text(text, names, near=NOW) for text in texts] == [
        "added 1\n+ [<user likes jazz>] user likes jazz",
        "document <doc 1>: done\nturn id(s): <ep 1>",
        "Claim '<cl 1>' is not visible here."]


def test_two_claims_with_the_same_text_in_tool_replies_are_refused() -> None:
    with pytest.raises(ValueError, match="both <user likes jazz>"):
        text_labels(["+ [cl_0123456789abcdef0123] user likes jazz",
                     "+ [cl_ffffffffffffffffffff] user likes jazz"])


def test_tool_text_loses_the_clock_and_the_token_and_keeps_every_other_instant() -> None:
    text = ("recorded 2026-09-26 11:59Z true from 2024-01-01 00:00Z, as true on "
            "2024-01-31T00:00:00Z, erased at 2026-10-26 12:00Z\nconfirm: eyJabc.def")
    assert normalise_text(text, {}, near=NOW) == (
        "recorded <wall clock> true from 2024-01-01 00:00Z, as true on "
        "2024-01-31T00:00:00Z, erased at 2026-10-26 12:00Z\nconfirm: <token>")
```

- [ ] **Step 3: Run them and see them fail because the module is missing.**

Run: `TEST tests/adversarial/parity/test_adv_parity_compare.py`
Expected: a collection error, `ModuleNotFoundError: No module named 'adversarial.parity.compare'` (or `ImportError` naming `compare`).

- [ ] **Step 4: Write the comparison.** Write `tests/adversarial/parity/compare.py`:

```python
"""Compare what two surfaces returned for the same operation, and say what differs.

The parity tests run one list of operations through several surfaces, each with a store
of its own. Two stores that were told the same things still differ in three ways that say
nothing about the surfaces:

* every id is random, so the same claim has a different id in each store;
* every instant a store takes from the clock is the moment that store ran;
* a few lists come back in id order, so their order is random too.

`normalise` removes those three and turns a result into plain dicts and lists, and
`differences` lists every place where what is left disagrees, with both values.
"""

from __future__ import annotations

import dataclasses
import enum
import math
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Iterator, Mapping

from memvara.types import (CLOSURE, ENTITY_REKEY, LAST_OBSERVED, OBJECT_ENTITY,
                           SALIENCE_BASE, SUBJECT_ENTITY, Claim, Document, Episode,
                           ForgetPreview, ForgetResult, WriteReceipt)

#: An id the store mints: a claim, a stored turn or a document, then 20 hex digits.
IDS = re.compile(r"\b(?:cl|ep|doc)_[0-9a-f]{20}\b")

#: `Claim.meta` keys where the store keeps its own bookkeeping. The hosted API carries two
#: of them as the top-level fields `salience_base` and `last_observed` and leaves the other
#: three out (`memvara/remote/hydrate.py`), so a claim is compared through its
#: `salience_base` and `last_observed` properties instead of through these keys.
BOOKKEEPING = frozenset({SALIENCE_BASE, LAST_OBSERVED, SUBJECT_ENTITY, OBJECT_ENTITY,
                         ENTITY_REKEY})

#: How close to the run an instant must be to count as one a store took from its clock.
#: It is wide enough for the default seven-day window of `memory_profile`, whose header
#: prints the start of that window. Every instant a parity test passes on purpose is
#: much further away than this, so it is compared as it is.
CLOCK_WINDOW = timedelta(days=8)

#: What an instant taken from the clock is replaced with.
WALL_CLOCK = "<wall clock>"

#: Two floats closer than this count as equal. A search score depends on how long ago a
#: claim was written, so the same search on two stores a few seconds apart differs in the
#: ninth decimal place.
TOLERANCE = 1e-6


@dataclasses.dataclass(frozen=True)
class Raised:
    """A step that raised, recorded as the exception's class name and its message."""

    kind: str
    message: str


def _every(value: Any) -> Iterator[Any]:
    """`value` and everything inside it, in a fixed order: a dataclass's fields in the
    order they are declared, a mapping's keys and values, and a sequence's items."""
    yield value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            yield from _every(getattr(value, field.name))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _every(key)
            yield from _every(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _every(item)


def labels(*results: Any) -> dict[str, str]:
    """A label for every id in `results`, made from what the id names.

    A claim is labelled by its subject, predicate and object, a stored turn by its text,
    and a document by its custom id or title. Two stores that were told the same things
    therefore give the same labels, whatever ids they drew. An id that none of those
    names, such as one only a string cites, is numbered in the order it first appears:
    `<cl 1>`, `<ep 1>`. That order is the same on every surface, because every surface
    runs the same program.

    Two different ids that would get the same label raise `ValueError`, because the
    comparison could no longer tell them apart. Give the two things different text.

    >>> from memvara.types import Claim
    >>> first = Claim(subject="user", predicate="lives_in", object="Berlin")
    >>> labels([first, "cl_00000000000000000000"]) == {
    ...     first.id: "<claim user lives_in Berlin>", "cl_00000000000000000000": "<cl 1>"}
    True
    """
    found: dict[str, str] = {}
    for value in _every(results):
        if isinstance(value, Claim):
            _name(found, value.id, f"<claim {value.subject} {value.predicate} {value.object}>")
        elif isinstance(value, Episode):
            _name(found, value.id, f"<turn {value.content}>")
        elif isinstance(value, Document):
            _name(found, value.id, f"<document {value.custom_id or value.title}>")
    _number_the_rest(found, (value for value in _every(results) if isinstance(value, str)))
    return found


def _name(found: dict[str, str], key: str, label: str) -> None:
    """Label `key`, refusing a label another id already has."""
    if any(other != key and given == label for other, given in found.items()):
        raise ValueError(f"two different ids are both {label}; give them different text "
                         "so the comparison can tell them apart")
    found[key] = label


def _number_the_rest(found: dict[str, str], texts: Iterable[str]) -> None:
    """Number every id in `texts` that `found` does not label yet, by its kind and in the
    order it first appears."""
    counts: dict[str, int] = {}
    for text in texts:
        for match in IDS.finditer(text):
            if match.group(0) not in found:
                kind = match.group(0).split("_", 1)[0]
                counts[kind] = counts.get(kind, 0) + 1
                found[match.group(0)] = f"<{kind} {counts[kind]}>"


def _instant(value: datetime, near: datetime) -> str:
    """`WALL_CLOCK` for an instant near `near`, and the instant itself otherwise."""
    return WALL_CLOCK if abs(value - near) <= CLOCK_WINDOW else value.isoformat()


def normalise(value: Any, names: Mapping[str, str], *, near: datetime) -> Any:
    """`value` as plain dicts, lists, strings and numbers, ready for `differences`.

    `names` labels ids (see `labels`), and an id it does not know becomes
    `<unlabelled id>`. `near` is when the run happened: an instant within
    `CLOCK_WINDOW` of it becomes `WALL_CLOCK`, and any other instant stays as it is.
    A dataclass keeps its fields under their names, plus `__type__` for its class.

    Four things are changed further, and each for a reason that holds on every surface:

    * a claim leaves out its bookkeeping keys (see `BOOKKEEPING`) and gains its `state`,
      `salience_base` and `last_observed`;
    * a write receipt leaves out `latency_ms`, which is how long the write took;
    * a preview's confirm token becomes `<token>`, because it is a signature over ids;
    * a `ForgetResult` sorts what it closed, because it closes in the token's id order.

    >>> from datetime import datetime, timezone
    >>> now = datetime(2026, 9, 26, tzinfo=timezone.utc)
    >>> normalise({"at": now, "then": datetime(2024, 1, 1, tzinfo=timezone.utc)}, {},
    ...           near=now)
    {'at': '<wall clock>', 'then': '2024-01-01T00:00:00+00:00'}
    """
    def again(item: Any) -> Any:
        return normalise(item, names, near=near)

    if isinstance(value, str):
        return IDS.sub(lambda found: names.get(found.group(0), "<unlabelled id>"), value)
    if isinstance(value, datetime):
        return _instant(value, near)
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Claim):
        return _claim(value, names, near)
    if isinstance(value, ForgetPreview):
        return {"__type__": "ForgetPreview", "close": value.close,
                "matches": again(list(value.matches.items())), "confirm": "<token>",
                "expires_at": again(value.expires_at)}
    if isinstance(value, ForgetResult):
        closed = [again(claim) for claim in value.closed]
        return {"__type__": "ForgetResult", "close": value.close, "reason": value.reason,
                "closed": sorted(closed, key=repr)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        out: dict[str, Any] = {"__type__": type(value).__name__}
        for field in dataclasses.fields(value):
            if isinstance(value, WriteReceipt) and field.name == "latency_ms":
                continue
            out[field.name] = again(getattr(value, field.name))
        return out
    if isinstance(value, Mapping):
        return {again(key): again(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [again(item) for item in value]
    return value


def _claim(claim: Claim, names: Mapping[str, str], near: datetime) -> dict[str, Any]:
    out: dict[str, Any] = {"__type__": "Claim"}
    for field in dataclasses.fields(claim):
        if field.name != "meta":
            out[field.name] = normalise(getattr(claim, field.name), names, near=near)
    meta = {key: item for key, item in claim.meta.items() if key not in BOOKKEEPING}
    if CLOSURE in meta:
        # Each closure records its instant as epoch seconds.
        meta[CLOSURE] = [
            {**entry, "at": _instant(datetime.fromtimestamp(entry["at"], timezone.utc), near)}
            for entry in meta[CLOSURE]]
    out["meta"] = normalise(meta, names, near=near)
    out["state"] = claim.state
    out["salience_base"] = claim.salience_base
    out["last_observed"] = normalise(claim.last_observed, names, near=near)
    return out


def differences(expected: Any, actual: Any, path: str = "") -> list[str]:
    """Every place where `actual` differs from `expected`, one line for each.

    A line names the path to the value, such as `.added[0].object_kind`, and both values.
    Two floats within `TOLERANCE` of each other are equal.

    >>> differences({"a": [1, 2.0], "b": "x"}, {"a": [1, 2.0000001], "c": "x"})
    [".b: only in the expected answer, 'x'", ".c: only in the actual answer, 'x'"]
    """
    if isinstance(expected, dict) and isinstance(actual, dict):
        out: list[str] = []
        for key in sorted(set(expected) | set(actual), key=str):
            if key not in actual:
                out.append(f"{path}.{key}: only in the expected answer, "
                           f"{_short(expected[key])}")
            elif key not in expected:
                out.append(f"{path}.{key}: only in the actual answer, "
                           f"{_short(actual[key])}")
            else:
                out += differences(expected[key], actual[key], f"{path}.{key}")
        return out
    if isinstance(expected, list) and isinstance(actual, list):
        out = []
        if len(expected) != len(actual):
            out.append(f"{path}: {len(expected)} item(s) expected, {len(actual)} found")
        for index, (one, other) in enumerate(zip(expected, actual)):
            out += differences(one, other, f"{path}[{index}]")
        return out
    if (isinstance(expected, float) and isinstance(actual, float)
            and math.isclose(expected, actual, rel_tol=0.0, abs_tol=TOLERANCE)):
        return []
    if expected != actual or type(expected) is not type(actual):
        return [f"{path}: expected {_short(expected)}, found {_short(actual)}"]
    return []


def _short(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 200 else text[:199] + "…"


def assert_same(expected: Any, actual: Any, what: str) -> None:
    """Fail with every difference listed, or return when there is none."""
    found = differences(expected, actual)
    assert not found, (f"{what}: {len(found)} difference(s)\n  " + "\n  ".join(found))


# -- the text an MCP tool returns ------------------------------------------------------

#: A write receipt's line for a claim it added: `+ [<id>] <the claim's text>`.
_ADDED = re.compile(r"^\+ \[(cl_[0-9a-f]{20})\] (.*)$", re.MULTILINE)

#: An instant as the tools write one: to the minute, as `_stamp` does, or in ISO 8601
#: with whole seconds or the six decimal places `datetime.isoformat` writes.
_STAMPS = re.compile(r"\b\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2}Z|T\d{2}:\d{2}:\d{2}(?:\.\d{6})?"
                     r"(?:Z|[+-]\d{2}:\d{2}))")

#: The token a preview of `memory_end_matching` or `memory_forget_matching` ends with.
_TOKEN = re.compile(r"^(confirm: )\S+$", re.MULTILINE)


def text_labels(texts: list[str]) -> dict[str, str]:
    """A label for every id in a session's tool replies, in the order they were written.

    A claim is labelled by its text, from the receipt line of the write that added it.
    Any other id, such as a stored turn or a document, is numbered in the order it first
    appears, which is the same on every surface because the session is the same. Two
    claims with the same text raise `ValueError`, as they do in `labels`.

    >>> text_labels(["+ [cl_0123456789abcdef0123] user likes jazz",
    ...              "turn id(s): ep_0123456789abcdef0123"])
    {'cl_0123456789abcdef0123': '<user likes jazz>', 'ep_0123456789abcdef0123': '<ep 1>'}
    """
    names: dict[str, str] = {}
    for text in texts:
        for found in _ADDED.finditer(text):
            _name(names, found.group(1), f"<{found.group(2)}>")
    _number_the_rest(names, texts)
    return names


def normalise_text(text: str, names: Mapping[str, str], *, near: datetime) -> str:
    """A tool's reply with ids labelled, the confirm token replaced, and every instant
    within `CLOCK_WINDOW` of `near` replaced with `WALL_CLOCK`.

    >>> from datetime import datetime, timezone
    >>> near = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)
    >>> normalise_text("recorded 2026-09-26 11:59Z true from 2024-01-01 00:00Z", {},
    ...                near=near)
    'recorded <wall clock> true from 2024-01-01 00:00Z'
    """
    def stamp(found: re.Match[str]) -> str:
        raw = found.group(0)
        if " " in raw:
            when = datetime.strptime(raw, "%Y-%m-%d %H:%MZ").replace(tzinfo=timezone.utc)
        else:
            when = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return WALL_CLOCK if abs(when - near) <= CLOCK_WINDOW else raw

    text = _TOKEN.sub(r"\1<token>", text)
    text = IDS.sub(lambda found: names.get(found.group(0), "<unlabelled id>"), text)
    return _STAMPS.sub(stamp, text)
```

- [ ] **Step 5: Run the self-tests and the module's doctests.**

Run: `TEST tests/adversarial/parity/test_adv_parity_compare.py tests/adversarial/parity/compare.py`
Expected: `23 passed` (18 tests and 5 doctests).

- [ ] **Step 6: Document it.** In `docs/claude/testing.md`, just before the final line that starts with `Next:`, add:

```markdown
## Parity across surfaces

`tests/adversarial/parity/` runs the same operations through every surface that offers them and compares the answers. An agent should get the same memory whichever way it reaches the store, so the comparison is exact, and every difference a surface is allowed is written down, with the place the code documents it. The plan is `docs/superpowers/plans/2026-09-26-adversarial-parity.md`.

- **What the comparison ignores.** Two stores that were told the same things still differ in ways that say nothing about the surface, and `compare.py` removes exactly those. It labels every id by what it names, because each store draws its own random ids; an id that no returned object names is numbered in the order it first appears. It replaces every instant within eight days of the run with `<wall clock>`, because each store takes those instants from its own clock. It replaces a preview's confirm token, which is a signature over ids, and it sorts the claims a confirmed `forget_matching` closed, because they come back in the token's id order. Two floats count as equal within 1e-6, because a search score drifts in the ninth decimal place between two runs a second apart. Everything else is compared, and a failure lists every field that differs, with its path and both values. `test_adv_parity_compare.py` pairs each of these rules with a check that a real difference still shows.
```

- [ ] **Step 7: Commit.**

```bash
git add tests/adversarial/parity/__init__.py tests/adversarial/parity/compare.py \
        tests/adversarial/parity/test_adv_parity_compare.py docs/claude/testing.md
git commit -m "Add the comparison the parity tests use, which ignores only random ids, clock readings and id order"
```

---

## Task 2: The library's four clients

**Files:**
- Create: `tests/adversarial/parity/test_adv_parity_library.py`
- Create, not committed: `local/a5-parity-faults/run_fault.py`
- Modify: `docs/claude/testing.md` (one bullet in the parity section)

**Interfaces:**
- Consumes: `Raised`, `assert_same`, `labels` and `normalise` from Task 1; `harness.stores.memory`; `harness.fakes.fake_v1.FakeV1` with `remote(user=...)` and `aremote(user=...)`.
- Produces, for Task 3 and for the maintainer: `Step(name, run, local_only)`, `PROGRAM`, `play(client, *, hosted) -> dict[str, Any]`, `opened(name, loop)` (a context manager that yields one of the four clients bound to the user alice), and the module fixture `played -> dict[str, dict[str, Any]]`, keyed by client name and then step name.

The program covers every operation the brief lists. `end` is spelled the library's way on every client, `delete(close="ended")` for one claim and `forget(close="ended")` for a slot, and a separate test compares the hosted clients' own `end()` with those two. A preview lists one match, so the confirmed closure and the replayed token each name one claim.

- [ ] **Step 1: Write the test file.** Write `tests/adversarial/parity/test_adv_parity_library.py`:

```python
"""The library's four clients give the same answers to the same operations.

One program of operations runs through each client, each with a store of its own:

* `Memvara`, the synchronous library, whose answers the others are compared with;
* `AsyncMemvara`, the same library behind `asyncio.to_thread`;
* `RemoteMemvara` and `AsyncRemoteMemvara`, the hosted clients, against `FakeV1`, whose
  answers come from a real local store (`tests/harness/fakes/fake_v1.py`).

`compare.normalise` removes what two stores differ in whatever the surface: ids,
instants taken from the clock, and lists that come back in id order. Every step is then
its own test, and a failure lists each field that differs, with both values.

**Where the hosted clients differ on purpose, the test asserts the difference.** Each
documented difference has its own test below, which says where it is documented and
checks that it is real, so the day it goes away the test says so:

* a claim does not carry `temporal_precision`, `object_kind`, `amount` or `unit` over
  `/v1` (docs/claude/testing.md, the `FakeV1` section);
* a search result's ranking does not carry `graph_rank`, `graph_score`,
  `temporal_rank`, `temporal_score` or `intent` (`memvara/remote/hydrate.py`,
  `explanation`);
* `stats()` also carries the two join counts, which `connectivity()` reads back out
  (`memvara/remote/api.py`, `RemoteMemvara.connectivity`);
* `recall(budget=...)` is refused (`memvara/remote/api.py`, the module docstring and
  `RemoteMemvara.recall`);
* the status of a document the caller cannot see is a `KeyError` on both sides, and
  each side words its own message (`Memvara.document_status` and
  `RemoteMemvara.document_status`).

The hosted clients also have `end()`, which sends `POST /v1/end`. The library ends a
fact with `delete(close="ended")` and a slot with `forget(close="ended")`, so the
program ends them that way on every client, and a test of its own checks that `end()`
leaves a hosted store holding what those two leave a local one holding.

**One step runs on the local clients only.** `recall(valid_at=...)` is refused by the
hosted clients, and memvara/memvara#298 tracks that, so the dated recall is compared
between the two local clients and the hosted refusal is left to that issue.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterator

import pytest

from harness import stores
from harness.fakes.fake_v1 import FakeV1
from memvara import AsyncMemvara, MemoryType
from memvara.types import utcnow

from .compare import Raised, assert_same, labels, normalise

UTC = timezone.utc

#: Instants the program passes on purpose. Each is more than `compare.CLOCK_WINDOW` away
#: from any run, so the comparison keeps it as it is.
BERLIN_FROM = datetime(2024, 1, 1, tzinfo=UTC)
LISBON_FROM = datetime(2025, 1, 1, tzinfo=UTC)
TRIP_FROM = datetime(2025, 6, 1, tzinfo=UTC)
TRIP_TO = datetime(2025, 9, 1, tzinfo=UTC)
LEFT_ACME = datetime(2025, 10, 1, tzinfo=UTC)
#: A day while the user lived in Berlin.
IN_BERLIN = BERLIN_FROM + timedelta(days=30)
#: When the expiring fact is erased: 30 days after this module is imported, the same
#: instant for every client.
EXPIRES = (utcnow() + timedelta(days=30)).replace(microsecond=0)

#: An id that no store holds.
MISSING = "cl_00000000000000000000"
QUESTION = "where does the user live"

CLIENTS = ("Memvara", "AsyncMemvara", "RemoteMemvara", "AsyncRemoteMemvara")
HOSTED = ("RemoteMemvara", "AsyncRemoteMemvara")


@dataclasses.dataclass(frozen=True)
class Step:
    """One operation of the program: a name, and what it does to a client.

    `run` receives the client and every earlier step's result, by name, so a step can
    use an id an earlier step returned. `local_only` says why the hosted clients leave
    the step out, or is None when every client runs it.
    """

    name: str
    run: Callable[[Any, dict[str, Any]], Any]
    local_only: str | None = None


def _added(got: dict[str, Any], step: str) -> str:
    """The id of the claim an earlier write step stored."""
    return str(got[step].added[0].id)


def _token(got: dict[str, Any]) -> str:
    return str(got["forget_matching.preview"].confirm)


PROGRAM: tuple[Step, ...] = (
    Step("remember", lambda m, got: m.remember("user", "lives_in", "Berlin",
                                               valid_from=BERLIN_FROM)),
    Step("remember.replacing", lambda m, got: m.remember("user", "lives_in", "Lisbon",
                                                         valid_from=LISBON_FROM)),
    Step("remember.rule", lambda m, got: m.remember(
        "user", "prefers", "short answers", memory_type=MemoryType.PROCEDURAL)),
    Step("remember.second_rule", lambda m, got: m.remember(
        "user", "prefers", "tabs over spaces", memory_type=MemoryType.PROCEDURAL)),
    Step("remember.finished", lambda m, got: m.remember(
        "user", "located_now", "Porto", valid_from=TRIP_FROM, valid_to=TRIP_TO,
        until_reason="the trip ended")),
    Step("remember.expiring", lambda m, got: m.remember(
        "user", "goal", "run a marathon", expires_at=EXPIRES,
        expire_reason="a goal for this season")),
    Step("remember.employer", lambda m, got: m.remember("user", "works_at", "Acme",
                                                        valid_from=LISBON_FROM)),
    Step("remember.taste", lambda m, got: m.remember("user", "likes", "jazz")),
    Step("add", lambda m, got: m.add("My name is Ada.")),
    Step("remember.cited", lambda m, got: m.remember(
        "user", "speaks", "Portuguese", sources=[got["add"].episode_ids[0]])),
    Step("search", lambda m, got: m.search(QUESTION, k=5)),
    Step("search.past", lambda m, got: m.search(QUESTION, k=5, valid_at=IN_BERLIN)),
    Step("recall", lambda m, got: m.recall(QUESTION)),
    Step("recall.budget", lambda m, got: m.recall(QUESTION, budget=12)),
    Step("recall.past", lambda m, got: m.recall(QUESTION, valid_at=IN_BERLIN),
         local_only="the hosted clients refuse recall(valid_at=...), which "
                    "memvara/memvara#298 tracks"),
    Step("get", lambda m, got: m.get(_added(got, "remember.replacing"))),
    Step("get.missing", lambda m, got: m.get(MISSING)),
    Step("history", lambda m, got: m.history("user", "lives_in")),
    Step("why", lambda m, got: m.why(_added(got, "remember.replacing"))),
    Step("why.cited", lambda m, got: m.why(_added(got, "remember.cited"))),
    Step("why.missing", lambda m, got: m.why(MISSING)),
    Step("count", lambda m, got: m.count()),
    Step("stats", lambda m, got: m.stats()),
    Step("connectivity", lambda m, got: m.connectivity()),
    Step("standing", lambda m, got: m.standing()),
    Step("standing.one", lambda m, got: m.standing(k=1)),
    Step("profile", lambda m, got: m.profile(QUESTION)),
    Step("forget_matching.preview", lambda m, got: m.forget_matching(
        "jazz", close="retired", k=1)),
    Step("forget_matching.other_closure", lambda m, got: m.forget_matching(
        "jazz", close="ended", k=1, confirm=_token(got))),
    Step("forget_matching.confirm", lambda m, got: m.forget_matching(
        "jazz", close="retired", k=1, confirm=_token(got))),
    Step("forget_matching.replayed", lambda m, got: m.forget_matching(
        "jazz", close="retired", k=1, confirm=_token(got))),
    Step("delete", lambda m, got: m.delete(_added(got, "remember.replacing"),
                                           reason="it was Porto")),
    Step("delete.missing", lambda m, got: m.delete(MISSING)),
    Step("end.claim", lambda m, got: m.delete(_added(got, "remember.cited"), close="ended")),
    Step("end.slot", lambda m, got: m.forget("user", "works_at", close="ended",
                                             at=LEFT_ACME)),
    Step("forget", lambda m, got: m.forget("user", "prefers")),
    Step("document.add", lambda m, got: m.add_document(
        "A runbook. Restart the service.", custom_id="docs/runbook", title="Runbook",
        meta={"team": "ops"})),
    Step("document.get", lambda m, got: m.get_document("docs/runbook")),
    Step("document.get.by_id", lambda m, got: m.get_document(got["document.add"].id)),
    Step("document.get.missing", lambda m, got: m.get_document("docs/none")),
    Step("document.list", lambda m, got: m.list_documents()),
    Step("document.update", lambda m, got: m.update_document("docs/runbook",
                                                             title="The runbook")),
    Step("document.status", lambda m, got: m.document_status("docs/runbook")),
    Step("document.status.missing", lambda m, got: m.document_status("docs/none")),
    Step("document.delete", lambda m, got: m.delete_document("docs/runbook")),
    Step("document.delete.again", lambda m, got: m.delete_document("docs/runbook")),
    Step("document.add.second", lambda m, got: m.add_document(
        "A second note.", custom_id="docs/second")),
    Step("document.delete.many", lambda m, got: m.delete_documents(["docs/second",
                                                                    "docs/none"])),
    Step("stats.after", lambda m, got: m.stats()),
    Step("connectivity.after", lambda m, got: m.connectivity()),
    Step("get_all.after", lambda m, got: m.get_all(states=["live", "ended", "retired"])),
)


class _Blocking:
    """An async client whose calls each run to completion on `loop`, so the same
    synchronous program can drive it."""

    def __init__(self, client: Any, loop: asyncio.AbstractEventLoop) -> None:
        self._client = client
        self._loop = loop

    def __getattr__(self, name: str) -> Callable[..., Any]:
        method = getattr(self._client, name)
        return lambda *args, **kwargs: self._loop.run_until_complete(method(*args, **kwargs))


def play(client: Any, *, hosted: bool) -> dict[str, Any]:
    """Run the program on `client` and return every step's result, by step name.

    A step that raises is recorded as `Raised` rather than stopping the program, so a
    client that fails at one step is still compared at every other.
    """
    got: dict[str, Any] = {}
    for step in PROGRAM:
        if hosted and step.local_only is not None:
            continue
        try:
            got[step.name] = step.run(client, got)
        except Exception as exc:  # noqa: BLE001 - a refusal is an answer to compare
            got[step.name] = Raised(type(exc).__name__, str(exc))
    return got


@contextlib.contextmanager
def opened(name: str, loop: asyncio.AbstractEventLoop) -> Iterator[Any]:
    """The client `name`, bound to the user alice over a store of its own, and closed
    when the block ends. An async client runs its calls on `loop` (see `_Blocking`), and
    a hosted one talks to a `FakeV1` of its own through a mock transport."""
    if name == "Memvara":
        local = stores.memory(user="alice")
        try:
            yield local
        finally:
            local.close()
    elif name == "AsyncMemvara":
        wrapped = AsyncMemvara(stores.memory(user="alice"))
        try:
            yield _Blocking(wrapped, loop)
        finally:
            loop.run_until_complete(wrapped.close())
    elif name == "RemoteMemvara":
        with FakeV1() as fake:
            remote = fake.remote(user="alice")
            try:
                yield remote
            finally:
                remote.close()
    else:
        with FakeV1() as fake:
            aremote = fake.aremote(user="alice")
            try:
                yield _Blocking(aremote, loop)
            finally:
                loop.run_until_complete(aremote.aclose())


@pytest.fixture(scope="module")
def played() -> dict[str, dict[str, Any]]:
    """Every client's normalised answer to every step it runs, played once per module."""
    near = utcnow()
    raw: dict[str, dict[str, Any]] = {}
    loop = asyncio.new_event_loop()
    try:
        for name in CLIENTS:
            with opened(name, loop) as client:
                raw[name] = play(client, hosted=name in HOSTED)
    finally:
        loop.close()
    out: dict[str, dict[str, Any]] = {}
    for client, got in raw.items():
        names = labels(*got.values())
        out[client] = {step: normalise(result, names, near=near)
                       for step, result in got.items()}
    return out


# -- what the hosted clients document they return differently -------------------------

#: `Claim` fields `/v1` does not carry, so a claim read through a hosted client has each
#: one's default, which is None. Documented in docs/claude/testing.md, in the section on
#: `FakeV1`.
NOT_ON_THE_WIRE = ("temporal_precision", "object_kind", "amount", "unit")

#: `Explanation` fields the hosted ranking does not carry, so a hosted search result has
#: each one's default, which is None. Documented in `memvara/remote/hydrate.py`, in the
#: docstring of `explanation`.
NOT_RANKED_ON_THE_WIRE = ("graph_rank", "graph_score", "temporal_rank", "temporal_score",
                          "intent")

#: The steps whose hosted answer a documented rule of its own describes, each checked by
#: its own test below rather than by the step-by-step comparison.
HOSTED_BY_OWN_TEST = frozenset({"recall.budget", "stats", "stats.after",
                                "document.status.missing"})


def as_hosted(value: Any) -> Any:
    """A local answer as the hosted clients document they return it: with the default in
    every field `NOT_ON_THE_WIRE` and `NOT_RANKED_ON_THE_WIRE` names."""
    if isinstance(value, dict):
        out = {key: as_hosted(item) for key, item in value.items()}
        if out.get("__type__") == "Claim":
            out.update(dict.fromkeys(NOT_ON_THE_WIRE))
        elif out.get("__type__") == "Explanation":
            out.update(dict.fromkeys(NOT_RANKED_ON_THE_WIRE))
        return out
    if isinstance(value, list):
        return [as_hosted(item) for item in value]
    return value


def _compared() -> Iterator[Any]:
    for step in PROGRAM:
        for client in CLIENTS[1:]:
            hosted = client in HOSTED
            if hosted and (step.local_only is not None or step.name in HOSTED_BY_OWN_TEST):
                continue
            yield pytest.param(step.name, client, id=f"{step.name}-{client}")


@pytest.mark.parametrize(("step", "client"), list(_compared()))
def test_every_client_answers_as_the_synchronous_library_does(
        played: dict[str, dict[str, Any]], step: str, client: str) -> None:
    local = played["Memvara"][step]
    expected = as_hosted(local) if client in HOSTED else local
    assert_same(expected, played[client][step], f"{step} through {client}")


def _typed(value: Any, kind: str) -> Iterator[dict[str, Any]]:
    """Every normalised dataclass of class `kind` inside `value`."""
    if isinstance(value, dict):
        if value.get("__type__") == kind:
            yield value
        for item in value.values():
            yield from _typed(item, kind)
    elif isinstance(value, list):
        for item in value:
            yield from _typed(item, kind)


@pytest.mark.parametrize("client", HOSTED)
def test_a_hosted_claim_has_the_default_in_each_field_the_wire_does_not_carry(
        played: dict[str, dict[str, Any]], client: str) -> None:
    """docs/claude/testing.md, the `FakeV1` section: `/v1` does not carry a claim's
    `temporal_precision`, `object_kind`, `amount` or `unit`."""
    local = list(_typed(played["Memvara"], "Claim"))
    hosted = list(_typed(played[client], "Claim"))
    assert any(claim["object_kind"] is not None for claim in local), (
        "no local claim sets object_kind, so this test no longer shows the difference")
    assert hosted and all(claim[field] is None
                          for claim in hosted for field in NOT_ON_THE_WIRE)


@pytest.mark.parametrize("client", HOSTED)
def test_a_hosted_ranking_has_the_default_in_each_field_the_wire_does_not_carry(
        played: dict[str, dict[str, Any]], client: str) -> None:
    """`memvara/remote/hydrate.py`, `explanation`: `graph_rank`, `graph_score`,
    `temporal_rank`, `temporal_score` and `intent` are not on the wire."""
    local = list(_typed(played["Memvara"], "Explanation"))
    hosted = list(_typed(played[client], "Explanation"))
    assert any(ranking["intent"] is not None for ranking in local), (
        "no local ranking sets intent, so this test no longer shows the difference")
    assert hosted and all(ranking[field] is None
                          for ranking in hosted for field in NOT_RANKED_ON_THE_WIRE)


@pytest.mark.parametrize("client", HOSTED)
@pytest.mark.parametrize("step", ["stats", "stats.after"])
def test_hosted_stats_also_carry_the_join_counts(
        played: dict[str, dict[str, Any]], step: str, client: str) -> None:
    """`memvara/remote/api.py`, `RemoteMemvara.connectivity`: the hosted client reads
    `live_claims` and `joinable_claims` out of what `stats()` returns, so `stats()`
    carries the library's counts and the join counts together."""
    local = played["Memvara"]
    joined = local[step.replace("stats", "connectivity")]
    assert "joinable_claims" not in local[step]
    assert_same({**local[step], **joined}, played[client][step], f"{step} through {client}")


@pytest.mark.parametrize("client", HOSTED)
def test_a_hosted_client_refuses_a_recall_budget(
        played: dict[str, dict[str, Any]], client: str) -> None:
    """`memvara/remote/api.py`, the module docstring and `RemoteMemvara.recall`:
    `POST /v1/recall` renders the block on the server and takes no budget, so a budget
    is refused rather than ignored."""
    local = played["Memvara"]["recall.budget"]
    hosted = played[client]["recall.budget"]
    assert isinstance(local, str) and "did not fit" in local
    assert hosted["__type__"] == "Raised" and hosted["kind"] == "ValueError"
    assert hosted["message"].startswith(
        "recall(budget=...) is not available against a hosted deployment")


@pytest.mark.parametrize("client", HOSTED)
def test_a_missing_document_s_status_is_a_key_error_through_every_client(
        played: dict[str, dict[str, Any]], client: str) -> None:
    """`Memvara.document_status` and `RemoteMemvara.document_status` both document a
    `KeyError` for a document the caller cannot see. Each words its own message: the
    library names the scope it looked in, and the hosted client says the document is not
    visible here, as its other refusals do. So the class is compared and the wording is
    not."""
    for name in ("Memvara", client):
        refusal = played[name]["document.status.missing"]
        assert (refusal["__type__"], refusal["kind"]) == ("Raised", "KeyError"), name
        assert "docs/none" in refusal["message"], name


@pytest.mark.parametrize("client", HOSTED)
def test_the_hosted_end_method_closes_what_the_library_closes(client: str) -> None:
    """The hosted clients have `end()`, which sends `POST /v1/end`, as a second way to
    close a fact that stopped being true. The library closes the same two ways with
    `delete(close="ended")` and `forget(close="ended")`. Both must leave the store
    holding the same claims."""
    near = utcnow()

    def end_both_ways(mem: Any, end_claim: Callable[[str], Any],
                      end_slot: Callable[[], Any]) -> list[Any]:
        """Store two facts, end one by its id and the other by its slot, and return
        both answers and every claim the store then holds."""
        mem.remember("user", "works_at", "Acme", valid_from=LISBON_FROM)
        taste = mem.remember("user", "likes", "jazz").added[0].id
        return [end_claim(taste), end_slot(), mem.get_all(states=["live", "ended", "retired"])]

    loop = asyncio.new_event_loop()
    try:
        with opened("Memvara", loop) as local:
            expected = end_both_ways(
                local, lambda claim_id: local.delete(claim_id, close="ended"),
                lambda: local.forget("user", "works_at", close="ended", at=LEFT_ACME))
        with opened(client, loop) as hosted:
            actual = end_both_ways(
                hosted, lambda claim_id: hosted.end(claim_id=claim_id),
                lambda: hosted.end(predicate="works_at", at=LEFT_ACME))
    finally:
        loop.close()
    # `forget` returns the claims it closed, and `end` returns only whether it closed any.
    expected[1] = bool(expected[1])
    assert_same(as_hosted(normalise(expected, labels(expected), near=near)),
                normalise(actual, labels(actual), near=near), f"end() through {client}")
```

- [ ] **Step 2: Write the fault runner.** It lives in `local/`, which git ignores, because it is a tool for checking these tests and not part of the suite. Write `local/a5-parity-faults/run_fault.py`:

```python
"""Run the parity tests against one deliberate fault, in a scratch copy of the checkout.

    python local/a5-parity-faults/run_fault.py <fault name>

The checkout itself is never edited. This copies memvara/, tests/, conftest.py and
pyproject.toml into a fresh temporary directory, makes one exact replacement there,
runs tests/adversarial/parity in the copy, and prints the tests that failed with the
first lines of each failure. A child process started by a test imports the copy,
because harness.env.child_env points PYTHONPATH at the checkout the harness lives in.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

CHECKOUT = pathlib.Path(__file__).resolve().parents[2]

#: name: (file, the text to replace, what replaces it, what the fault breaks)
FAULTS = {
    "hosted-expiry": (
        "memvara/remote/hydrate.py",
        'out.expire_reason = body.get("expire_reason")',
        "out.expire_reason = None",
        "a hosted claim loses its expire_reason"),
    "async-profile": (
        "memvara/aio.py",
        "asyncio.to_thread(search) if query else _nothing())",
        "_nothing())",
        "AsyncMemvara.profile leaves out the search"),
    "hosted-end-as-retire": (
        "memvara/remote/api.py",
        "return self.end(claim_id=claim_id, at=at, reason=why)",
        "pass",
        "RemoteMemvara.delete(close='ended') retires instead of ending"),
    "hosted-provenance": (
        "memvara/remote/hydrate.py",
        'episodes=[episode(e) for e in body["sources"]],',
        "episodes=[],",
        "a hosted why() loses its source turns"),
    "stdio-anchored": (
        "memvara/server/cli.py",
        "anchored=config.anchored",
        "anchored=True",
        "the stdio server anchors every read and describes its tools that way"),
    "no-storage-line": (
        "memvara/server/mcp.py",
        'store = getattr(memory, "store", None)\n    if not isinstance(store, SQLiteStore):',
        'store = None\n    if not isinstance(store, SQLiteStore):',
        "memory_stats loses its storage line on a local store too"),
}


def main() -> int:
    name = sys.argv[1]
    path, old, new, meaning = FAULTS[name]
    scratch = pathlib.Path(tempfile.mkdtemp(prefix=f"a5-fault-{name}-"))
    for part in ("memvara", "tests"):
        shutil.copytree(CHECKOUT / part, scratch / part,
                        ignore=shutil.ignore_patterns("__pycache__"))
    for part in ("conftest.py", "pyproject.toml"):
        shutil.copy2(CHECKOUT / part, scratch / part)
    target = scratch / path
    source = target.read_text(encoding="utf-8")
    assert source.count(old) == 1, f"{path} holds {source.count(old)} copies of the text"
    target.write_text(source.replace(old, new), encoding="utf-8")
    print(f"fault {name}: {meaning}")
    env = {**os.environ, "PYTHONPATH": str(scratch)}
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rf",
         "tests/adversarial/parity"], cwd=scratch, env=env, capture_output=True, text=True)
    shown = 0
    for line in run.stdout.splitlines():
        if line.startswith(("FAILED", "ERROR")) or " passed" in line or " failed" in line:
            print(line[:300])
        elif line.startswith("E  ") and ("expected" in line or "cloud mode" in line) \
                and shown < 6:
            shown += 1
            print("   ", line[1:].strip()[:250])
    shutil.rmtree(scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: See the tests fail for the right reason.** A parity test checks behaviour that already exists, so the way to see it fail is to break that behaviour in a scratch copy. Run each fault that concerns the library:

```bash
$PY local/a5-parity-faults/run_fault.py hosted-expiry
$PY local/a5-parity-faults/run_fault.py async-profile
$PY local/a5-parity-faults/run_fault.py hosted-end-as-retire
```

Expected:
- `hosted-expiry` fails `remember.expiring` and `get_all.after` for both hosted clients, with `.added[0].expire_reason: expected 'a goal for this season', found None` and the same field inside the final listing.
- `async-profile` fails `profile-AsyncMemvara` with `.relevant: 8 item(s) expected, 0 found`.
- `hosted-end-as-retire` fails `get_all.after-RemoteMemvara` with `[0].state: expected 'ended', found 'retired'`, and `stats.after` for the same client, whose counts of ended and retired claims move by one.

- [ ] **Step 4: Run the tests against the real code.**

Run: `TEST tests/adversarial/parity/test_adv_parity_library.py`
Expected: `157 passed`.

- [ ] **Step 5: Document it.** In the parity section of `docs/claude/testing.md`, add after the "What the comparison ignores" bullet:

```markdown
- **The library's four clients.** `test_adv_parity_library.py` runs one program through the synchronous `Memvara`, `AsyncMemvara`, and the hosted clients `RemoteMemvara` and `AsyncRemoteMemvara`, each hosted client against a `FakeV1` of its own. The program writes facts with `remember` and `add`, reads them with `search`, `recall`, `get`, `history` and `why`, closes them with `delete`, `forget`, `forget_matching` with and without its confirm token, and the two ways of ending, and it covers the document methods, `stats`, `standing` and `profile`. Every step is its own test, which compares one client's answer with the synchronous library's. Where the hosted clients differ on purpose, a test of its own checks the difference and names where it is documented: a claim has the default in the four fields `/v1` does not carry, a search result's ranking has the default in the five fields the wire does not carry, `stats()` also carries the two join counts, `recall(budget=...)` is refused, and the status of a document the caller cannot see is a `KeyError` on both sides, in each side's own words. The hosted clients also have `end()`, and a test checks that it leaves a hosted store holding what `delete(close="ended")` and `forget(close="ended")` leave a local one holding. The hosted clients refuse `recall(valid_at=...)`, which memvara/memvara#298 tracks, so the dated recall is compared between the two local clients only.
```

- [ ] **Step 6: Commit.**

```bash
git add tests/adversarial/parity/test_adv_parity_library.py docs/claude/testing.md
git commit -m "Compare the library's four clients step by step, and assert each documented hosted difference"
```

---

## Task 3: The MCP text on three surfaces

**Files:**
- Create: `tests/adversarial/parity/test_adv_parity_mcp.py`
- Modify: `docs/claude/testing.md` (the rest of the parity section)

**Interfaces:**
- Consumes: `assert_same`, `normalise_text` and `text_labels` from Task 1; `harness.stdio.McpProcess` and `kill_all`; `harness.env.child_env`; `FakeV1.serve()` and `FakeV1.api_key`; `ServerConfig.from_env`, `build_memvara` and `MemvaraMCPServer`.
- Produces: `InProcessServer(env)`, which answers `initialize()`, `list_tools()`, `call(name, **arguments)` and `kill()` as `McpProcess` does; `Call(name, tool, arguments, local_only)`; `SESSION`; `converse(server, *, cloud)`; and the module fixture `played -> Played`, whose `handshake` and `replies` are keyed by surface.

The cloud-mode server is an `McpProcess` with `MEMVARA_MODE=cloud`, `MEMVARA_API_KEY=fake.api_key` and `MEMVARA_SERVER_URL=fake.serve()`. `McpProcess` always sets `MEMVARA_DB`, and cloud mode ignores it without creating the file. `child_env` sets `MEMVARA_EMBEDDER=hashing`, which cloud mode accepts because it is the default.

- [ ] **Step 1: Write the test file.** Write `tests/adversarial/parity/test_adv_parity_mcp.py`:

```python
"""A model reads the same text from the MCP server, however the server is reached.

One session of tool calls runs through three surfaces, each with a store of its own:

* the in-process server: `MemvaraMCPServer.handle_line`, built from the environment the
  way `memvara.server.cli.main` builds it;
* the stdio server in local mode, a real child process over a real pipe
  (`harness.stdio.McpProcess`);
* the stdio server in cloud mode, pointed at `FakeV1` served on 127.0.0.1.

`compare.normalise_text` labels ids by the text they name, replaces the confirm token,
and replaces the instants each run took from its clock. After that the in-process server
and the local stdio server must write exactly the same text for every step, because
nothing but the transport separates them.

**Cloud mode differs from them only where the code documents that it does.** Each such
difference has a test of its own, which names the documentation and checks that the
difference is real:

* `memory_stats` has a `storage:` line only for a local store (`memvara/server/mcp.py`,
  `_storage_fact`);
* `memory_standing` ends with a "more not shown" line only for a local store, because
  `GET /v1/standing` reports no total (`memvara/server/tools.py`, `_standing`);
* `memory_recall` refuses `budget` in cloud mode (`memvara/server/memory_api.py`,
  `MemoryAPI.recall`).

**One step runs on the local surfaces only.** `memory_recall` with `valid_at` is refused
in cloud mode, and memvara/memvara#298 tracks that, so the dated recall is compared
between the two local surfaces and the cloud refusal is left to that issue.
"""

from __future__ import annotations

import dataclasses
import json
import re
from datetime import timedelta
from typing import Any, Callable, Iterator, Mapping

import pytest

from harness.env import child_env
from harness.fakes.fake_v1 import FakeV1
from harness.stdio import McpProcess, ToolResult, kill_all
from memvara.server.config import ServerConfig, build_memvara
from memvara.server.mcp import MemvaraMCPServer
from memvara.types import utcnow

from .compare import assert_same, normalise_text, text_labels

SURFACES = ("in-process", "stdio local", "stdio cloud")
QUESTION = "where does the user live"
#: A day while the user lived in Berlin.
IN_BERLIN = "2024-01-31T00:00:00Z"
MISSING = "cl_00000000000000000000"
#: When the expiring fact is erased: 30 days after this module is imported, the same
#: instant on every surface.
EXPIRES = ((utcnow() + timedelta(days=30)).replace(microsecond=0)
           .isoformat().replace("+00:00", "Z"))


class InProcessServer:
    """`MemvaraMCPServer`, built from `env` the way `cli.main` builds it, and driven one
    JSON-RPC line at a time through `handle_line`. It answers the same four calls a test
    makes on `McpProcess`."""

    def __init__(self, env: Mapping[str, str]) -> None:
        config = ServerConfig.from_env(env)
        self.server = MemvaraMCPServer(
            build_memvara(config), read_only=config.read_only, anchored=config.anchored,
            features_off=config.features_off, **config.scope_kwargs)
        self._next_id = 0

    def request(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        self._next_id += 1
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": self._next_id, "method": method}
        if params is not None:
            message["params"] = dict(params)
        reply = self.server.handle_line(json.dumps(message))
        assert reply is not None, f"no reply to {method}"
        return json.loads(reply)["result"]

    def initialize(self) -> Any:
        result = self.request("initialize", {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "memvara-adversarial-suite", "version": "0"}})
        assert self.server.handle_line(
            '{"jsonrpc": "2.0", "method": "notifications/initialized"}') is None
        return result

    def list_tools(self) -> list[Any]:
        return list(self.request("tools/list")["tools"])

    def call(self, name: str, /, **arguments: Any) -> ToolResult:
        result = self.request("tools/call", {"name": name, "arguments": arguments})
        text = "".join(str(block.get("text", "")) for block in result.get("content", []))
        return ToolResult(text=text, is_error=bool(result.get("isError")), raw=result)

    def kill(self) -> None:
        self.server.close()


@dataclasses.dataclass(frozen=True)
class Call:
    """One tool call of the session.

    `arguments` is the call's arguments, or a function that builds them from the text
    every earlier step returned, by step name, so a step can use an id an earlier reply
    named. `local_only` says why cloud mode leaves the step out, or is None.
    """

    name: str
    tool: str
    arguments: Mapping[str, Any] | Callable[[dict[str, str]], Mapping[str, Any]]
    local_only: str | None = None


def _claim(got: dict[str, str], step: str) -> str:
    """The id of the claim a write step added, from its receipt's `+ [<id>]` line."""
    found = re.search(r"^\+ \[(cl_[0-9a-f]{20})\]", got[step], re.MULTILINE)
    assert found, got[step]
    return found.group(1)


def _turn(got: dict[str, str]) -> str:
    found = re.search(r"turn id\(s\): (ep_[0-9a-f]{20})", got["add"])
    assert found, got["add"]
    return found.group(1)


def _token(got: dict[str, str]) -> str:
    return got["forget_matching.preview"].rsplit("confirm: ", 1)[1].strip()


def _fact(predicate: str, value: str, **more: Any) -> dict[str, Any]:
    return {"subject": "user", "predicate": predicate, "object": value, **more}


SESSION: tuple[Call, ...] = (
    Call("remember", "memory_remember",
         _fact("lives_in", "Berlin", true_since="2024-01-01T00:00:00Z")),
    Call("remember.replacing", "memory_remember",
         _fact("lives_in", "Lisbon", true_since="2025-01-01T00:00:00Z")),
    Call("remember.rule", "memory_remember",
         _fact("prefers", "short answers", memory_type="procedural")),
    Call("remember.second_rule", "memory_remember",
         _fact("prefers", "tabs over spaces", memory_type="procedural")),
    Call("remember.finished", "memory_remember",
         _fact("located_now", "Porto", true_since="2025-06-01T00:00:00Z",
               true_until="2025-09-01T00:00:00Z", until_reason="the trip ended")),
    Call("remember.expiring", "memory_remember",
         _fact("goal", "run a marathon", expires_at=EXPIRES,
               expire_reason="a goal for this season")),
    Call("remember.employer", "memory_remember",
         _fact("works_at", "Acme", true_since="2025-01-01T00:00:00Z")),
    Call("remember.taste", "memory_remember", _fact("likes", "jazz")),
    Call("add", "memory_add", {"text": "My name is Ada."}),
    Call("remember.cited", "memory_remember",
         lambda got: _fact("speaks", "Portuguese", sources=[_turn(got)])),
    Call("search", "memory_search", {"query": QUESTION}),
    Call("search.past", "memory_search", {"query": QUESTION, "valid_at": IN_BERLIN}),
    Call("recall", "memory_recall", {"query": QUESTION}),
    Call("recall.budget", "memory_recall", {"query": QUESTION, "budget": 12}),
    Call("recall.past", "memory_recall", {"query": QUESTION, "valid_at": IN_BERLIN},
         local_only="cloud mode refuses memory_recall with valid_at, which "
                    "memvara/memvara#298 tracks"),
    Call("history", "memory_history", {"subject": "user", "predicate": "lives_in"}),
    Call("why", "memory_why",
         lambda got: {"claim_id": _claim(got, "remember.replacing")}),
    Call("why.cited", "memory_why", lambda got: {"claim_id": _claim(got, "remember.cited")}),
    Call("why.missing", "memory_why", {"claim_id": MISSING}),
    Call("stats", "memory_stats", {}),
    Call("standing", "memory_standing", {}),
    Call("standing.one", "memory_standing", {"k": 1}),
    Call("profile", "memory_profile", {"query": QUESTION}),
    Call("forget_matching.preview", "memory_forget_matching", {"query": "jazz", "k": 1}),
    Call("forget_matching.other_closure", "memory_end_matching",
         lambda got: {"confirm": _token(got), "k": 1}),
    Call("forget_matching.confirm", "memory_forget_matching",
         lambda got: {"confirm": _token(got), "k": 1}),
    Call("forget_matching.replayed", "memory_forget_matching",
         lambda got: {"confirm": _token(got), "k": 1}),
    Call("forget.claim", "memory_forget",
         lambda got: {"claim_id": _claim(got, "remember.replacing"),
                      "reason": "it was Porto"}),
    Call("forget.missing", "memory_forget", {"claim_id": MISSING}),
    Call("end.slot", "memory_end",
         {"subject": "user", "predicate": "works_at", "at": "2025-10-01T00:00:00Z"}),
    Call("end.claim", "memory_end", lambda got: {"claim_id": _claim(got, "remember.cited")}),
    Call("forget.slot", "memory_forget", {"subject": "user", "predicate": "prefers"}),
    Call("document.add", "memory_add_document",
         {"content": "A runbook. Restart the service.", "custom_id": "docs/runbook",
          "title": "Runbook", "metadata": {"team": "ops"}}),
    Call("document.get", "memory_get_document", {"id": "docs/runbook"}),
    Call("document.get.missing", "memory_get_document", {"id": "docs/none"}),
    Call("document.list", "memory_list_documents", {}),
    Call("document.delete", "memory_delete_document", {"id": "docs/runbook"}),
    Call("document.delete.again", "memory_delete_document", {"id": "docs/runbook"}),
    Call("stats.after", "memory_stats", {}),
)




@dataclasses.dataclass(frozen=True)
class Played:
    """What every surface answered, played once per module."""

    #: Each surface's answers to `initialize` and `tools/list`, as parsed JSON.
    handshake: dict[str, dict[str, Any]]
    #: Each surface's error flag and normalised text for every call, by step name.
    replies: dict[str, dict[str, tuple[bool, str]]]


def converse(server: Any, *, cloud: bool) -> tuple[dict[str, Any], dict[str, tuple[bool, str]]]:
    """Run the session on `server`: the handshake a client opens with, then every call.

    Returns the answers to `initialize` and `tools/list`, and every call's error flag and
    text by step name. In cloud mode a call marked `local_only` is left out.
    """
    handshake = {"initialize": server.initialize(), "tools/list": server.list_tools()}
    replies: dict[str, tuple[bool, str]] = {}
    got: dict[str, str] = {}
    for call in SESSION:
        if cloud and call.local_only is not None:
            continue
        arguments = call.arguments(got) if callable(call.arguments) else call.arguments
        result = server.call(call.tool, **arguments)
        got[call.name] = result.text
        replies[call.name] = (result.is_error, result.text)
    return handshake, replies


@pytest.fixture(scope="module")
def played(tmp_path_factory: pytest.TempPathFactory) -> Played:
    """Every surface's answers to the session, each surface over a store of its own."""
    root = tmp_path_factory.mktemp("parity-mcp")
    home = root / "home"
    home.mkdir()
    near = utcnow()
    raw: dict[str, tuple[dict[str, Any], dict[str, tuple[bool, str]]]] = {}
    started: list[Any] = []
    try:
        server = InProcessServer(child_env(home, {"MEMVARA_DB": str(root / "in-process.db"),
                                                  "MEMVARA_USER": "alice"}))
        started.append(server)
        raw["in-process"] = converse(server, cloud=False)
        local = McpProcess(root / "stdio-local.db", home=home, user="alice")
        started.append(local)
        raw["stdio local"] = converse(local, cloud=False)
        with FakeV1() as fake:
            # MEMVARA_DB is set by McpProcess and ignored in cloud mode; no file is made.
            cloud = McpProcess(root / "unused.db", home=home, user="alice",
                               env={"MEMVARA_MODE": "cloud",
                                    "MEMVARA_API_KEY": fake.api_key,
                                    "MEMVARA_SERVER_URL": fake.serve()})
            started.append(cloud)
            raw["stdio cloud"] = converse(cloud, cloud=True)
            # Stopped before the fake closes, so nothing it sends meets a closed server.
            cloud.kill()
    finally:
        kill_all(started)
    replies: dict[str, dict[str, tuple[bool, str]]] = {}
    for surface, (_handshake, texts) in raw.items():
        names = text_labels([text for _error, text in texts.values()])
        replies[surface] = {step: (error, normalise_text(text, names, near=near))
                            for step, (error, text) in texts.items()}
    return Played(handshake={surface: shake for surface, (shake, _texts) in raw.items()},
                  replies=replies)


@pytest.mark.parametrize("surface", SURFACES[1:])
@pytest.mark.parametrize("answer", ["initialize", "tools/list"])
def test_every_surface_answers_the_handshake_as_the_in_process_server_does(
        played: Played, answer: str, surface: str) -> None:
    assert_same(played.handshake["in-process"][answer], played.handshake[surface][answer],
                f"{answer} through {surface}")


# -- what cloud mode documents it writes differently -----------------------------------


@dataclasses.dataclass(frozen=True)
class LocalLine:
    """A line the code documents as written only when the server has a local store."""

    #: The steps whose reply carries the line.
    steps: tuple[str, ...]
    #: How the line begins.
    start: str
    #: Where the code says so.
    documented: str


LOCAL_LINES = (
    LocalLine(("stats", "stats.after"), "storage: ",
              "memvara/server/mcp.py, _storage_fact: None for anything without a local "
              "SQLite store, because a hosted deployment's disks are its operator's to "
              "encrypt"),
    LocalLine(("standing.one",), "(1 more not shown",
              "memvara/server/tools.py, _standing: GET /v1/standing caps at k and reports "
              "no total, so against a hosted deployment the hint cannot fire"),
)

#: The steps whose cloud reply a documented rule of its own describes, each checked by its
#: own test below rather than by the step-by-step comparison.
CLOUD_BY_OWN_TEST = frozenset({"recall.budget"})


def without_local_lines(step: str, text: str) -> str:
    """`text` with every line `LOCAL_LINES` documents for `step` removed."""
    starts = tuple(line.start for line in LOCAL_LINES if step in line.steps)
    if not starts:
        return text
    return "\n".join(row for row in text.split("\n") if not row.startswith(starts))


def _compared() -> Iterator[Any]:
    for call in SESSION:
        for surface in SURFACES[1:]:
            if surface == "stdio cloud" and (call.local_only is not None
                                             or call.name in CLOUD_BY_OWN_TEST):
                continue
            yield pytest.param(call.name, surface, id=f"{call.name}-{surface}")


@pytest.mark.parametrize(("step", "surface"), list(_compared()))
def test_every_surface_writes_what_the_in_process_server_writes(
        played: Played, step: str, surface: str) -> None:
    error, text = played.replies["in-process"][step]
    if surface == "stdio cloud":
        text = without_local_lines(step, text)
    actual_error, actual_text = played.replies[surface][step]
    assert_same({"error": error, "lines": text.split("\n")},
                {"error": actual_error, "lines": actual_text.split("\n")},
                f"{step} through {surface}")


@pytest.mark.parametrize("line", LOCAL_LINES, ids=lambda line: line.start.strip(" ("))
def test_a_line_documented_as_local_is_written_locally_and_not_in_cloud_mode(
        played: Played, line: LocalLine) -> None:
    for step in line.steps:
        for surface in SURFACES:
            rows = played.replies[surface][step][1].split("\n")
            found = [row for row in rows if row.startswith(line.start)]
            if surface == "stdio cloud":
                assert found == [], f"{step} in cloud mode: {line.documented}"
            else:
                assert len(found) == 1, f"{step} through {surface}: {rows}"


def test_cloud_mode_refuses_a_recall_budget(played: Played) -> None:
    """`memvara/server/memory_api.py`, `MemoryAPI.recall`: `ScopedRemoteMemvara.recall`
    raises for any budget other than None, because `POST /v1/recall` renders the block
    on the server and takes no budget."""
    for surface in ("in-process", "stdio local"):
        error, text = played.replies[surface]["recall.budget"]
        assert not error and "did not fit" in text, (surface, text)
    error, text = played.replies["stdio cloud"]["recall.budget"]
    assert error
    assert text.startswith("memory_recall failed: ValueError: recall(budget=...) is not "
                           "available against a hosted deployment"), text
```

- [ ] **Step 2: See the tests fail for the right reason.** Run the faults that concern the MCP text:

```bash
$PY local/a5-parity-faults/run_fault.py hosted-provenance
$PY local/a5-parity-faults/run_fault.py stdio-anchored
$PY local/a5-parity-faults/run_fault.py no-storage-line
```

Expected:
- `hosted-provenance` fails `why.cited-stdio cloud`, where cloud mode says "No source turns are retained for this claim.", and the hosted library steps that cite the turn.
- `stdio-anchored` fails the `tools/list` handshake for both stdio surfaces, with `[0].inputSchema.properties.anchored.default: expected False, found True` and the rewritten description beside it.
- `no-storage-line` fails `test_a_line_documented_as_local_is_written_locally_and_not_in_cloud_mode[storage:]`.

- [ ] **Step 3: Run the tests against the real code.**

Run: `TEST tests/adversarial/parity/test_adv_parity_mcp.py`
Expected: `83 passed`.

- [ ] **Step 4: Run the whole folder and time it.**

Run: `TEST tests/adversarial/parity --durations=5`
Expected: `263 passed`, in well under the fast tier's 6 seconds.

- [ ] **Step 5: Document it.** In the parity section of `docs/claude/testing.md`, add after the library bullet:

```markdown
- **The MCP text.** `test_adv_parity_mcp.py` runs one session of tool calls through three surfaces: the in-process server, which is `MemvaraMCPServer.handle_line` built from the environment the way `memvara.server.cli.main` builds it; the stdio server in local mode; and the stdio server in cloud mode, pointed at `FakeV1` on 127.0.0.1. The in-process server and the local stdio server must write exactly the same text at every step, because only the transport separates them. Cloud mode must write the same text too, except where the code documents that it does not, and a test of its own checks each of those: `memory_stats` has no `storage:` line, because a hosted deployment has no local store to describe; `memory_standing` never says how many preferences it left out, because `GET /v1/standing` reports no total; and `memory_recall` refuses `budget`. Cloud mode refuses `memory_recall` with `valid_at`, which memvara/memvara#298 tracks, so the dated recall is compared between the two local surfaces only.
- **It catches what it is meant to catch.** It was run against six deliberate faults, each in a scratch copy of the checkout, and every one failed the tests that should catch it: a hosted claim losing its `expire_reason`, `AsyncMemvara.profile` leaving out its search, the hosted `delete(close="ended")` retiring the claim instead of ending it, a hosted `why()` losing its source turns, the stdio server anchoring every read, and `memory_stats` losing its `storage:` line on a local store.
- **Adding an operation.** Add a `Step` to `PROGRAM` in the library file, or a `Call` to `SESSION` in the MCP file. If one surface then answers differently, find where the code documents the difference. If it is documented, add a test that asserts the difference and names the place. If it is not, it is a bug, and "Known bugs and security findings" above says what to do with it.

The folder takes about 1.5 seconds on a laptop, which is within the fast tier's budget of about 6 seconds for this workstream.
```

- [ ] **Step 6: Commit.**

```bash
git add tests/adversarial/parity/test_adv_parity_mcp.py docs/claude/testing.md
git commit -m "Compare the MCP text from the in-process server, stdio in local mode and stdio in cloud mode"
```

---

## Task 4: Verification

- [ ] **Step 1: Flakes.** Run each new test file 20 times in a row, and count the runs that did not pass everything:

```bash
for f in compare library mcp; do
  fails=0
  for i in $(seq 20); do
    $PY -m pytest -q -p no:cacheprovider tests/adversarial/parity/test_adv_parity_$f.py > /dev/null || fails=$((fails+1))
  done
  echo "$f: $fails of 20 runs failed"
done
```

Expected: `0 of 20 runs failed` for each file.

- [ ] **Step 2: The full gate, once, with a private coverage file.** The machine is shared, so start it only when no other gate of this workstream is running.

```bash
mkdir -p $W/local/cov
COVERAGE_FILE=$W/local/cov/.coverage.a5 $PY -m coverage run -m pytest -q -p no:cacheprovider
COVERAGE_FILE=$W/local/cov/.coverage.a5 $PY -m coverage report
```

Expected: the pytest summary reports no failures, and the coverage report's `TOTAL` line reads `100%`.

- [ ] **Step 3: Type checks.**

```bash
$PY -m mypy -p memvara
$PY -m mypy tests/harness
$PY -m mypy tests/harness --ignore-missing-imports
```

Expected: `Success: no issues found` from each.

- [ ] **Step 4: Report.** The final message gives the branch name, each commit's short sha and subject, the files added or changed, the pass counts, the flake result, the gate's result line and coverage total, the mypy results, every undocumented difference found (with its classification against `SECURITY.md`, an offline reproduction, and the failing tests kept in `local/`), and every departure from the design with its reason.
