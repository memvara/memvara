"""The regressions step: run the suite's nightly tier, and turn what failed into findings.

The step runs `pytest --tier nightly` through pytest_results.py, which writes one JSON line
per test. Each test that failed is rerun twice (flakes.py), and becomes a `Failure`: a
finding and what the reruns said. Its invariant is the test's node id. When the failure
carries the replay program the reference model's state machine prints, the program is the
finding's operations, so two different breaks the same test finds get two fingerprints.

Two failures are not breaks in a test, and are never filed:

* A strict expected failure that passes (`xpass-strict`) usually means its bug's fix
  landed; strict mode fails the run until the marker is removed.
* A run that exits with an error and no failed test (`session`), such as the skip
  ledger's refusal or a credential guard, failed outside any test.
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from nightly import flakes, night

from harness.report import Finding

#: The step's folder inside the night's folder, and the files it writes there.
FOLDER = "regressions"
OUTPUT = "output.log"
RESULTS = "results.jsonl"
ARTIFACTS = {"log": f"{FOLDER}/{OUTPUT}", "results": f"{FOLDER}/{RESULTS}"}

#: The outcomes that make a run fail.
FAILING = ("failed", "error", "xpass-strict")

#: What pytest's exit statuses other than 0 and 1 mean.
EXIT_MEANING = {2: "it was interrupted", 3: "it hit an internal error",
                4: "it was called wrongly", 5: "it collected no tests"}

#: How many failed tests are rerun. Hundreds of failures usually share one cause.
RERUN_LIMIT = 20

_SEED = re.compile(r"@reproduce_failure\('[^']*', b'[^']*'\)")
_PYTEST_MARK = re.compile(r"^E(?=\s|$)")


def command(python: str, results: pathlib.Path) -> list[str]:
    """The step's command. It keeps going past a test file that fails to import, which
    plain pytest does not, so one bad import cannot blank the night."""
    return [python, str(pathlib.Path(__file__).with_name("pytest_results.py")),
            "--results", str(results), "--", "-q", "-p", "no:cacheprovider",
            "--tier", "nightly", "--continue-on-collection-errors", "--durations=25"]


@dataclass(frozen=True)
class Results:
    """What pytest_results.py wrote: one record per test, and pytest's exit status, which
    is None when the run was stopped before it finished."""

    tests: list[dict[str, Any]]
    exitstatus: int | None


def read_results(path: pathlib.Path) -> Results:
    records, _ = night.read_jsonl(path)
    tests = [record for record in records if "nodeid" in record]
    statuses = [record["exitstatus"] for record in records if "exitstatus" in record]
    return Results(tests, statuses[-1] if statuses else None)


def layers(results: Results) -> dict[str, dict[str, int]]:
    """Per layer, the tests that ran (everything but a skip) and the tests that failed."""
    counts: dict[str, dict[str, int]] = {}
    for test in results.tests:
        if test["outcome"] == "skipped":
            continue
        entry = counts.setdefault(flakes.layer_of(test["nodeid"]), {"run": 0, "failed": 0})
        entry["run"] += 1
        entry["failed"] += test["outcome"] in FAILING
    return counts


def program_of(text: str) -> tuple[str, ...]:
    """The operations of the last replay program in a failure's text, one per line as
    `drive.format_program` prints them. pytest marks the lines of an exception's text
    with `E`, which is removed. A program cut off before its closing line is not used,
    because it would replay something other than the break."""
    start = text.rfind("drive.replay([")
    if start < 0:
        return ()
    rest = text[start + len("drive.replay(["):]
    if rest.startswith("])"):
        return ()
    ops: list[str] = []
    for line in rest.split("\n")[1:]:
        body = _PYTEST_MARK.sub("", line).strip()
        if body == "])":
            return tuple(ops)
        if body:
            ops.append(body[:-1] if body.endswith(",") else body)
    return ()


def seed_of(text: str) -> str | None:
    """The `@reproduce_failure(...)` call Hypothesis printed for a failure, if any."""
    seeds = _SEED.findall(text)
    return seeds[-1] if seeds else None


@dataclass(frozen=True)
class Failure:
    """One failure of the night: its finding, its kind, and what its reruns said."""

    finding: Finding
    #: "test", "xpass-strict" (a strict expected failure passed) or "session" (the run
    #: failed outside any test).
    kind: str
    reruns: tuple[str, ...]
    verdict: str

    @property
    def fingerprint(self) -> str:
        return self.finding.signature()

    @property
    def fileable(self) -> bool:
        """Only a test that failed every time is a break that may be filed."""
        return self.kind == "test" and self.verdict == "confirmed"

    def to_record(self) -> dict[str, Any]:
        """The failure as report.json keeps it."""
        return {"fingerprint": self.fingerprint, "kind": self.kind,
                "reruns": list(self.reruns), "verdict": self.verdict,
                "finding": self.finding.to_dict()}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> Failure:
        return cls(Finding.from_dict(record["finding"]), str(record["kind"]),
                   tuple(record["reruns"]), str(record["verdict"]))


def failures(results: Results, *, commit: str,
             rerun: Callable[[str], Sequence[str] | None], limit: int = RERUN_LIMIT,
             log_tail: str = "") -> list[Failure]:
    """The night's failures, in the order pytest ran them.

    Each failed test is rerun through `rerun(nodeid)`, which returns the reruns' results,
    or None once the caller's rerun budget is spent. After `limit` tests, or once the
    budget is spent, a failed test is reported unconfirmed. A strict expected failure that
    passed is not rerun. A run that exits 1 with no failed test, or with any status other
    than 0 and 1, adds one session failure carrying `log_tail`.
    """
    found: list[Failure] = []
    reruns_asked = 0
    for test in results.tests:
        outcome = test["outcome"]
        if outcome == "xpass-strict":
            found.append(Failure(_finding(test, commit), "xpass-strict", (), "unconfirmed"))
        elif outcome in ("failed", "error"):
            answer: Sequence[str] | None = None
            if reruns_asked < limit:
                reruns_asked += 1
                answer = rerun(test["nodeid"])
            said = tuple(answer or ())
            found.append(Failure(_finding(test, commit), "test", said, flakes.verdict(said)))
    status = results.exitstatus
    if status not in (None, 0) and (status != 1 or not found):
        found.append(_session(status, commit, log_tail))
    return found


def _finding(test: Mapping[str, Any], commit: str) -> Finding:
    text = str(test.get("longrepr", ""))
    return Finding(layer=flakes.layer_of(test["nodeid"]), surface="suite",
                   invariant=test["nodeid"], ops=program_of(text), seed=seed_of(text),
                   artifacts=ARTIFACTS, commit=commit, title=_title(test), detail=text)


def _title(test: Mapping[str, Any]) -> str:
    message = str(test.get("message", "")).strip().splitlines()
    first = message[0] if message else ""
    if test["outcome"] == "xpass-strict":
        what = "passes, although it is marked as a strict expected failure"
    elif test["outcome"] == "error":
        what = f"errors in {test.get('when', 'setup')}"
    else:
        what = "fails"
    title = f"{test['nodeid']} {what}" + (f": {first}" if first else "")
    return title if len(title) <= 240 else title[:237] + "..."


def _session(status: int, commit: str, log_tail: str) -> Failure:
    if status == 1:
        title = ("pytest exited with status 1 but reported no failed test, so a check "
                 "outside the tests failed; the end of its output says which")
    else:
        title = (f"pytest exited with status {status}, which means "
                 f"{EXIT_MEANING.get(status, 'an unknown failure')}")
    finding = Finding(layer="suite", surface="pytest", invariant=f"exit status {status}",
                      artifacts={"log": ARTIFACTS["log"]}, commit=commit, title=title,
                      detail=log_tail)
    return Failure(finding, "session", (), "unconfirmed")
