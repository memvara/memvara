"""The skip ledger: every reason a test in this repository may skip, and why that is
not hiding a failure.

It covers every test under tests/, not only the adversarial suite: tests/conftest.py
registers it for every run. A skip whose reason no rule below explains fails the run. The ledger exists because most
summaries show a skip as green, so a test that stops running for a new reason looks
exactly like a test that still passes. When you add a skip, add a rule that says why it
is legitimate.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import Any

import pytest


@dataclass(frozen=True)
class SkipRule:
    #: A regular expression, searched for in the skip reason.
    pattern: str
    #: Why a test that skips for this reason is not hiding a failure.
    why: str
    #: The `sys.platform` values this skip is expected on. Empty means every platform.
    platforms: tuple[str, ...] = ()
    #: The skip is expected only below this Python version, for example (3, 11).
    python_below: tuple[int, int] | None = None
    #: The skip is expected only from this Python version on.
    python_from: tuple[int, int] | None = None

    def applies(self, platform: str, version: tuple[int, int]) -> bool:
        """Whether a skip is expected on this platform and Python version."""
        if self.platforms and platform not in self.platforms:
            return False
        if self.python_below is not None and version >= self.python_below:
            return False
        return self.python_from is None or version >= self.python_from


RULES: tuple[SkipRule, ...] = (
    SkipRule(r"^tomllib arrive[sd] in 3\.11",
             "Python 3.10 has no tomllib. These tests run on 3.11 and later.",
             python_below=(3, 11)),
    SkipRule(r"^3\.10 only$",
             "The 3.10 half of a pair whose other half needs tomllib.",
             python_from=(3, 11)),
    SkipRule(r"^no dist/",
             "The wheel checks run at release time, after python3 -m build --wheel."),
    SkipRule(r"^node/npm missing or unloadable$",
             "These need Node. CI's npm-bridge job runs the bridge's own tests."),
    SkipRule(r"^the renderer is the authority for this test$",
             "Needs memvara-cloud installed, which is a separate repository."),
    SkipRule(r"^no sub-second headroom exists at this platform's ceiling$",
             "Windows' C runtime stops at the year 3001, so there is no headroom to test.",
             platforms=("win32",)),
    SkipRule(r"^this platform stores at most \d+ days of history",
             "Windows' C runtime stops at the year 3001, so a century half-life cannot "
             "bottom out.", platforms=("win32",)),
    SkipRule(r"^could not import '(openai|pypdf|httpx|zoneinfo)'",
             "An optional extra that this environment did not install."),
    SkipRule(r"^no POSIX permission bits to check$",
             "Windows has no POSIX file modes.", platforms=("win32",)),
    SkipRule(r"^Windows file modes do not express this$",
             "Windows has no POSIX file modes.", platforms=("win32",)),
    SkipRule(r"^this user may write a read-only file$",
             "Root, or another account that ignores file modes, cannot be refused a write "
             "by one, so the test has no way to make the file read-only."),
    SkipRule(r"^the password database exists only on POSIX$",
             "Windows has no password database to fall back from.", platforms=("win32",)),
    SkipRule(r"^SIGSTOP exists only on POSIX$",
             "Windows cannot pause a process with a signal.", platforms=("win32",)),
    SkipRule(r"^a RAM disk is made with hdiutil, which is macOS only$",
             "The local tier's real full-disk test mounts a RAM disk with macOS's own tools.",
             platforms=("linux", "win32")),
    SkipRule(r"^Python before 3\.13 does not report an unclosed SQLite connection$",
             "The leak is observable only through the ResourceWarning that Python 3.13 "
             "added for an unclosed sqlite3 connection; earlier versions stay silent.",
             python_below=(3, 13)),
    SkipRule(r"^RLIMIT_FSIZE exists only on POSIX$",
             "Windows has no per-process limit on file size to simulate a full disk with.",
             platforms=("win32",)),
    SkipRule(r"^the fake agent CLIs are POSIX shell scripts$",
             "On Windows a program started without a shell is found on PATH only as an "
             ".exe, and the fake claude and codex are shell scripts. Linux and macOS run "
             "these tests.", platforms=("win32",)),
    SkipRule(r"^the recall daemon listens on a unix socket, which Windows lacks$",
             "The recall daemon cannot listen on Windows, which has no unix sockets, so "
             "every prompt there takes the in-process route (plugin/hooks/lib/ipc.py, "
             "send). Linux and macOS run these tests.", platforms=("win32",)),
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


def explained(reason: str, *, platform: str = sys.platform,
              version: tuple[int, int] | None = None) -> bool:
    """Whether some rule in RULES covers `reason` on this platform and Python version.

    A reason that a rule explains only on Windows, for example, is unexplained on Linux,
    so a test that starts skipping where it should run turns the run red.
    """
    if version is None:
        version = (sys.version_info.major, sys.version_info.minor)
    return any(re.search(rule.pattern, reason) and rule.applies(platform, version)
               for rule in RULES)


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
