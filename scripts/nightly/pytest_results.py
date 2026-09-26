"""Run pytest and write one JSON line per test to a results file.

    python pytest_results.py --results FILE -- <pytest arguments>

The nightly run starts this with the tested checkout as the working directory and the
interpreter of the night's virtual environment. It loads nothing else from this folder,
and it takes this folder off the import path before pytest starts, so the tests import
the tested checkout and nothing of the same name from here. Then it puts the working
directory first on the path, as `python -m pytest` does.

Each line is `{"nodeid", "outcome", "when", "message", "longrepr"}`. The outcome is one
of passed, failed (in the test itself), error (in setup, teardown or collection),
skipped, xfailed, xpassed, and xpass-strict: a test marked as a strict expected failure
that passed, which pytest counts as failed. A file that fails to import is an error line
under the file's own path. The last line is `{"exitstatus": N}`, pytest's exit status.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, TextIO

#: How much of a failure's text is kept. The end is kept, because that is where the
#: exception, and any replay program and seed Hypothesis printed, appear.
KEEP = 100_000

_FAILING = ("failed", "error", "xpass-strict")


class ResultLog:
    """A pytest plugin that writes one line per test as the test finishes."""

    def __init__(self, out: TextIO) -> None:
        self.out = out
        self.pending: dict[str, dict[str, Any]] = {}

    def _write(self, record: dict[str, Any]) -> None:
        self.out.write(json.dumps(record, ensure_ascii=True) + "\n")
        self.out.flush()

    def pytest_runtest_logreport(self, report: Any) -> None:
        entry = self.pending.setdefault(report.nodeid, {
            "nodeid": report.nodeid, "outcome": "passed", "when": "call",
            "message": "", "longrepr": ""})
        if entry["outcome"] in _FAILING:
            return  # the first failure is the one that explains the test
        if report.failed:
            text = report.longreprtext
            if text.startswith("[XPASS(strict)]"):
                outcome = "xpass-strict"
            else:
                outcome = "failed" if report.when == "call" else "error"
            entry.update(outcome=outcome, when=report.when, message=_message(report),
                         longrepr=text[-KEEP:])
        elif report.skipped and entry["outcome"] == "passed":
            entry.update(outcome="xfailed" if hasattr(report, "wasxfail") else "skipped",
                         when=report.when)
        elif report.when == "call" and hasattr(report, "wasxfail"):
            entry["outcome"] = "xpassed"

    def pytest_runtest_logfinish(self, nodeid: str, location: Any) -> None:
        entry = self.pending.pop(nodeid, None)
        if entry is not None:
            self._write(entry)

    def pytest_collectreport(self, report: Any) -> None:
        if report.failed:
            self._write({"nodeid": report.nodeid, "outcome": "error", "when": "collect",
                         "message": _message(report),
                         "longrepr": report.longreprtext[-KEEP:]})

    def finish(self, exitstatus: int) -> None:
        for entry in self.pending.values():  # tests a crash left without a finish
            self._write(entry)
        self.pending.clear()
        self._write({"exitstatus": int(exitstatus)})


def _message(report: Any) -> str:
    crash = getattr(report.longrepr, "reprcrash", None)
    message = getattr(crash, "message", None)
    if isinstance(message, str) and message:
        return message
    lines = report.longreprtext.strip().splitlines()
    return lines[-1] if lines else ""


def main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[0] != "--results" or argv[2] != "--":
        print("usage: pytest_results.py --results FILE -- <pytest arguments>",
              file=sys.stderr)
        return 4
    results, pytest_args = argv[1], argv[3:]
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [entry for entry in sys.path
                   if os.path.abspath(entry or os.curdir) != here]
    sys.path.insert(0, os.getcwd())
    import pytest  # noqa: PLC0415 - only after the import path is set

    with open(results, "w", encoding="utf-8", newline="\n") as out:
        log = ResultLog(out)
        exitstatus = 3
        try:
            exitstatus = int(pytest.main(pytest_args, plugins=[log]))
        finally:
            log.finish(exitstatus)
    return exitstatus


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
