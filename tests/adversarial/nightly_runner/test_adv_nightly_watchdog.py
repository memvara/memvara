"""The watchdog: a night that did not run, or did not finish, is reported, not missed.

The nightly run writes a heartbeat before anything else, and writes it again when it
finishes. The watchdog uses no model. launchd runs it once a day at a deadline, and when a
night has no heartbeat by then, or a heartbeat with no finish, it writes a DID NOT RUN or
DID NOT FINISH report into the night's folder and sends one macOS notification. It does
nothing else. Time is given to it explicitly here, never read from the clock.
"""

from __future__ import annotations

import json
import pathlib
import plistlib
import re
import subprocess
import sys
from datetime import date, datetime, time, timedelta, timezone

import pytest

from harness.env import REPO, child_env

if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))
from nightly import night, watchdog  # noqa: E402 - scripts/ is not on the path until above

ZONE = timezone(timedelta(hours=2))
START, DEADLINE = time(1, 30), time(6, 30)


def _at(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 9, day, hour, minute, tzinfo=ZONE)


class Notifications:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def __call__(self, title: str, message: str) -> bool:
        self.sent.append((title, message))
        return True


def _folder(checkout: pathlib.Path, day: str) -> pathlib.Path:
    return night.Layout(checkout).night(day)


def test_a_night_with_no_heartbeat_by_the_deadline_did_not_run(
        tmp_path: pathlib.Path) -> None:
    """The Mac slept, the app was closed or a login expired: nothing ran, and without the
    watchdog nothing would say so."""
    notify = Notifications()
    verdict = watchdog.check(tmp_path, now=_at(27, 6, 31), start=START, deadline=DEADLINE,
                             notify=notify)
    folder = _folder(tmp_path, "2026-09-27")
    assert (verdict.night, verdict.state, verdict.notified) == (
        date(2026, 9, 27), "did-not-run", True)
    assert verdict.report == folder / "DID-NOT-RUN.md"
    assert "DID NOT RUN" in verdict.report.read_text()
    assert json.loads((folder / "DID-NOT-RUN.json").read_text())["state"] == "did-not-run"
    assert len(notify.sent) == 1
    title, message = notify.sent[0]
    assert title == "memvara nightly" and "2026-09-27" in message and "did not run" in message


def test_a_run_that_started_and_never_finished_is_reported_too(
        tmp_path: pathlib.Path) -> None:
    """A run killed partway leaves a heartbeat with a start and no finish. A watchdog that
    only looked for the heartbeat would stay silent about a night with no report."""
    folder = _folder(tmp_path, "2026-09-27")
    night.write_heartbeat(folder / night.HEARTBEAT, night="2026-09-27",
                          started_at="2026-09-27T01:30:02+02:00")
    notify = Notifications()
    verdict = watchdog.check(tmp_path, now=_at(27, 6, 31), start=START, deadline=DEADLINE,
                             notify=notify)
    assert (verdict.state, verdict.report) == ("did-not-finish", folder / "DID-NOT-FINISH.md")
    assert "01:30:02" in verdict.report.read_text()
    assert len(notify.sent) == 1 and "did not finish" in notify.sent[0][1]


def test_a_night_that_finished_is_left_alone(tmp_path: pathlib.Path) -> None:
    """The heartbeat the run writes is the one the watchdog reads: the folder, the file
    name and the finish field must be the same on both sides."""
    assert (watchdog.NIGHTLY, watchdog.HEARTBEAT) == (night.NIGHTLY, night.HEARTBEAT)
    folder = _folder(tmp_path, "2026-09-27")
    night.write_heartbeat(folder / night.HEARTBEAT, night="2026-09-27",
                          started_at="2026-09-27T01:30:02+02:00",
                          finished_at="2026-09-27T02:48:10+02:00", status="finished")
    notify = Notifications()
    verdict = watchdog.check(tmp_path, now=_at(27, 6, 31), start=START, deadline=DEADLINE,
                             notify=notify)
    assert (verdict.state, verdict.report, verdict.notified) == ("finished", None, False)
    assert notify.sent == []
    assert sorted(path.name for path in folder.iterdir()) == ["heartbeat.json"]


def test_before_the_deadline_the_watchdog_checks_the_night_before(
        tmp_path: pathlib.Path) -> None:
    """Run by hand at three in the morning, the watchdog must not report tonight's run,
    which may still be going, as missed."""
    assert watchdog.night_to_check(_at(27, 3), START, DEADLINE) == date(2026, 9, 26)
    assert watchdog.night_to_check(_at(27, 6, 30), START, DEADLINE) == date(2026, 9, 27)
    assert watchdog.night_to_check(_at(27, 1), START, DEADLINE) == date(2026, 9, 26)


def test_a_night_that_starts_before_midnight_is_named_by_the_day_it_starts() -> None:
    """The run names its folder by the day it starts, so a run at 23:30 on the 26th is the
    night of the 26th, checked at 06:30 on the 27th."""
    evening = time(23, 30)
    assert watchdog.night_to_check(_at(27, 6, 45), evening, DEADLINE) == date(2026, 9, 26)
    assert watchdog.night_to_check(_at(27, 6, 15), evening, DEADLINE) == date(2026, 9, 25)


def test_a_second_check_of_the_same_night_sends_no_second_notification(
        tmp_path: pathlib.Path) -> None:
    """launchd runs a missed job when the Mac wakes, and a person may run it by hand too.
    One missed night is one notification."""
    notify = Notifications()
    for minutes in (31, 45):
        watchdog.check(tmp_path, now=_at(27, 6, minutes), start=START, deadline=DEADLINE,
                       notify=notify)
    assert len(notify.sent) == 1


def test_a_check_stopped_partway_notifies_exactly_once_when_it_runs_again(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A watchdog killed partway, while it writes its files or before its notification
    goes out, must still notify once when it runs again, and only once. If the files it
    had written counted as "reported", the missed night would never be notified."""
    folder = _folder(tmp_path, "2026-09-27")
    real_write = watchdog._write_record
    stops = ["between the two writes"]

    def stop_once(path: pathlib.Path, record: dict[str, object]) -> None:
        if stops:
            stops.pop()
            raise RuntimeError("killed between writing the report and the record")
        real_write(path, record)

    monkeypatch.setattr(watchdog, "_write_record", stop_once)
    notify = Notifications()
    with pytest.raises(RuntimeError):
        watchdog.check(tmp_path, now=_at(27, 6, 31), start=START, deadline=DEADLINE,
                       notify=notify)
    assert (folder / "DID-NOT-RUN.md").exists() and notify.sent == []
    assert watchdog.check(tmp_path, now=_at(27, 6, 40), start=START, deadline=DEADLINE,
                          notify=notify).notified
    assert len(notify.sent) == 1

    attempts: list[str] = []

    def dies_while_notifying(title: str, message: str) -> bool:
        attempts.append(message)
        if len(attempts) == 1:
            raise RuntimeError("killed before the notification went out")
        return True

    for day in (28, 28, 28):
        try:
            watchdog.check(tmp_path, now=_at(day, 6, 31), start=START, deadline=DEADLINE,
                           notify=dies_while_notifying)
        except RuntimeError:
            pass
    assert len(attempts) == 2, "one attempt killed, then one delivered, then none"
    record = json.loads((_folder(tmp_path, "2026-09-28") / "DID-NOT-RUN.json").read_text())
    assert record["notified"] is True


def test_the_notification_text_is_passed_as_arguments_and_never_run_as_script() -> None:
    """osascript runs the text after -e as AppleScript. A message spliced into it could
    run commands, so the title and message go in as arguments to a fixed script."""
    message = 'The night of "2026-09-27" did not run" & (do shell script "id") & "'
    argv = watchdog.osascript_argv("memvara nightly", message)
    assert argv[-2:] == ["memvara nightly", message]
    scripts = [argv[index + 1] for index, part in enumerate(argv) if part == "-e"]
    assert scripts == ["on run argv",
                       "display notification (item 2 of argv) with title (item 1 of argv)",
                       "end run"]
    ran: list[list[str]] = []

    def run(command: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        ran.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    assert watchdog.notify("memvara nightly", message, run=run) is True
    assert ran == [argv]

    def missing(command: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("osascript")

    assert watchdog.notify("memvara nightly", message, run=missing) is False


def test_the_watchdog_runs_as_a_script_with_nothing_else_from_the_nightly_folder(
        tmp_path: pathlib.Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    """launchd runs the file with whatever Python the operator named, from no particular
    folder, so it must run on its own."""
    done = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "nightly" / "watchdog.py"), "--checkout",
         str(tmp_path), "--start", "01:30", "--deadline", "06:30",
         "--now", "2026-09-27T06:31:00+02:00", "--no-notify"],
        cwd=tmp_path_factory.mktemp("elsewhere"),
        env=child_env(tmp_path_factory.mktemp("home")), capture_output=True, text=True,
        timeout=60)
    assert done.returncode == 0, done.stderr
    assert "did not run" in done.stdout
    assert (_folder(tmp_path, "2026-09-27") / "DID-NOT-RUN.md").exists()


def test_the_launchd_template_runs_the_watchdog_at_its_deadline() -> None:
    """The template is installed by hand, later, so it must already be a valid property
    list once filled in, run the watchdog with arguments the watchdog accepts, and fire at
    the deadline it passes."""
    template = (REPO / "scripts" / "nightly" /
                "com.memvara.nightly-watchdog.plist.template").read_text()
    assert set(re.findall(r"__[A-Z]+__", template)) == {
        "__PYTHON__", "__CHECKOUT__", "__START__", "__DEADLINE__", "__HOUR__", "__MINUTE__"}
    filled = (template.replace("__PYTHON__", "/opt/python/bin/python3")
              .replace("__CHECKOUT__", "/Users/someone/memvara")
              .replace("__START__", "01:30").replace("__DEADLINE__", "06:30")
              .replace("__HOUR__", "6").replace("__MINUTE__", "30"))
    job = plistlib.loads(filled.encode())
    assert job["Label"] == "com.memvara.nightly-watchdog"
    program = job["ProgramArguments"]
    assert program[:2] == ["/opt/python/bin/python3",
                           "/Users/someone/memvara/scripts/nightly/watchdog.py"]
    args = watchdog.parser().parse_args(program[2:])
    assert (args.checkout, args.start, args.deadline) == (
        "/Users/someone/memvara", time(1, 30), time(6, 30))
    assert job["StartCalendarInterval"] == {"Hour": 6, "Minute": 30}
    assert not job.get("RunAtLoad", False)
