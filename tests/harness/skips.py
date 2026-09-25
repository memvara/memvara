"""The skip ledger: every reason a test in this repository may skip, and why that is not
hiding a failure.

It covers every test under tests/, not only the adversarial suite: tests/conftest.py
registers it for every run. A skip whose reason no rule below explains fails the run. The ledger exists because most
summaries show a skip as green, so a test that stops running for a new reason looks
exactly like a test that still passes. When you add a skip, add a rule that says why it
is legitimate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pytest


@dataclass(frozen=True)
class SkipRule:
    #: A regular expression, searched for in the skip reason.
    pattern: str
    #: Why a test that skips for this reason is not hiding a failure.
    why: str


RULES: tuple[SkipRule, ...] = (
    SkipRule(r"^tomllib arrive[sd] in 3\.11",
             "Python 3.10 has no tomllib. These tests run on 3.11 and later."),
    SkipRule(r"^3\.10 only$",
             "The 3.10 half of a pair whose other half needs tomllib."),
    SkipRule(r"^no dist/",
             "The wheel checks run at release time, after python3 -m build --wheel."),
    SkipRule(r"^node/npm missing or unloadable$",
             "These need Node. CI's npm-bridge job runs the bridge's own tests."),
    SkipRule(r"^the renderer is the authority for this test$",
             "Needs memvara-cloud installed, which is a separate repository."),
    SkipRule(r"^no sub-second headroom exists at this platform's ceiling$",
             "A platform limit on timestamps that the test measures before skipping."),
    SkipRule(r"^this platform stores at most \d+ days of history",
             "A platform limit on timestamps that the test measures before skipping."),
    SkipRule(r"^could not import '(openai|pypdf|httpx|zoneinfo)'",
             "An optional extra that this environment did not install."),
    SkipRule(r"^no POSIX permission bits to check$",
             "Windows has no POSIX file modes."),
    SkipRule(r"^Windows file modes do not express this$",
             "Windows has no POSIX file modes."),
    SkipRule(r"^the hooks' copy of the vectors is not in this checkout yet$",
             "A packaging state that the test detects before skipping."),
    SkipRule(r"^git is not installed$",
             "Project detection needs git."),
    SkipRule(r"^mypy is not installed",
             "The type-level tests need mypy. CI's types job has it."),
    SkipRule(r"^deployment not asked, tool surface NOT checked",
             "Needs MEMVARA_API_KEY to ask the hosted deployment. CI provides it."),
    SkipRule(r"^needs the encrypt extra",
             "The encrypt extra is optional. CI installs it."),
    SkipRule(r"^no system tz database and no tzdata package$",
             "The time zone tests need a tz database."),
)


def reason_of(longrepr: object) -> str:
    """The reason text of a skipped report, without the "Skipped: " prefix pytest adds."""
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        text = str(longrepr[2])
    else:
        text = str(longrepr)
    return text.removeprefix("Skipped: ")


def explained(reason: str) -> bool:
    """Whether some rule in RULES covers `reason`."""
    return any(re.search(rule.pattern, reason) for rule in RULES)


class SkipLedger:
    """A pytest plugin that fails the session when a skip has no rule in RULES.

    An expected failure (xfail) is reported as skipped too. It is not a skip, and it is
    ignored here.
    """

    def __init__(self) -> None:
        self.unexplained: list[str] = []

    def _record(self, report: Any) -> None:
        if not report.skipped or hasattr(report, "wasxfail"):
            return
        reason = reason_of(report.longrepr)
        if not explained(reason):
            self.unexplained.append(f"{report.nodeid}: {reason}")

    def pytest_runtest_logreport(self, report: Any) -> None:
        self._record(report)

    def pytest_collectreport(self, report: Any) -> None:
        self._record(report)

    def pytest_sessionfinish(self, session: Any) -> None:
        if not self.unexplained:
            return
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_sep("=", "skips with no rule in tests/harness/skips.py", red=True)
            for line in self.unexplained:
                reporter.write_line(line)
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
