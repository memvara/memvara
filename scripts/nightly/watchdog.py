"""The nightly watchdog: report a night that did not run, or did not finish.

The nightly run writes local/nightly/<date>/heartbeat.json before it does anything else,
and writes it again, with `finished_at`, when it ends. launchd runs this script once a day
at a deadline (com.memvara.nightly-watchdog.plist.template beside it), and it checks the
latest night whose deadline has passed:

* no heartbeat: the night did not run, for example because the Mac slept, the app that
  runs the scheduled session was closed, or a login expired. It writes DID-NOT-RUN.md
  and DID-NOT-RUN.json in the night's folder;
* a heartbeat with no finish: the run started and died partway, or is still running past
  every step's cap. It writes DID-NOT-FINISH.md and DID-NOT-FINISH.json;
* a finished heartbeat: it does nothing.

For a missed night it also sends one macOS notification, through osascript. A report
already in the folder means that night was reported, so a second check sends nothing. The
watchdog uses no model and does nothing else. It imports nothing from the nightly
package, so launchd can run it with any Python 3.10 or later:

    python3 scripts/nightly/watchdog.py --checkout C --start 01:30 --deadline 06:30
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, Callable, Sequence

#: Where the nightly run keeps its nights, relative to the main checkout, and the name of
#: its heartbeat. The same as in night.py, which the tests check.
NIGHTLY = ("local", "nightly")
HEARTBEAT = "heartbeat.json"
DID_NOT_RUN = "DID-NOT-RUN"
DID_NOT_FINISH = "DID-NOT-FINISH"
TITLE = "memvara nightly"


def night_to_check(now: datetime, start: time, deadline: time) -> date:
    """The latest night whose deadline has passed at `now`.

    A night is named by the date its run starts. Its deadline is the first time the clock
    reads `deadline` after `start`, so a run at 23:30 has a deadline the next morning.
    """
    wall = now.replace(tzinfo=None)
    gap = ((datetime.combine(date.min, deadline) - datetime.combine(date.min, start))
           % timedelta(days=1)) or timedelta(days=1)
    night = datetime.combine(wall.date(), start)
    while night + gap > wall:
        night -= timedelta(days=1)
    return night.date()


@dataclass(frozen=True)
class Verdict:
    night: date
    #: "finished", "did-not-run" or "did-not-finish".
    state: str
    #: The report the watchdog wrote or found, or None for a finished night.
    report: pathlib.Path | None
    #: Whether a notification was sent by this check.
    notified: bool


def check(checkout: pathlib.Path, *, now: datetime, start: time, deadline: time,
          notify: Callable[[str, str], object] | None) -> Verdict:
    """Check the latest night whose deadline has passed, and report it if it was missed.
    `notify(title, message)` sends the notification; None sends none."""
    night = night_to_check(now, start, deadline)
    folder = pathlib.Path(checkout).joinpath(*NIGHTLY, night.isoformat())
    heartbeat = _read(folder / HEARTBEAT)
    if heartbeat is not None and heartbeat.get("finished_at"):
        return Verdict(night, "finished", None, False)
    state, name = (("did-not-run", DID_NOT_RUN) if heartbeat is None
                   else ("did-not-finish", DID_NOT_FINISH))
    report = folder / f"{name}.md"
    if report.exists():
        return Verdict(night, state, report, False)
    folder.mkdir(parents=True, exist_ok=True)
    shown = f"{deadline:%H:%M}"
    where = "/".join((*NIGHTLY, night.isoformat(), report.name))
    if heartbeat is None:
        text = (f"# DID NOT RUN: the nightly run of {night}\n\n"
                f"No nightly run had started for the night of {night} by the deadline, "
                f"{shown}. The watchdog checked at {now:%Y-%m-%d %H:%M} and found no "
                f"heartbeat at {'/'.join((*NIGHTLY, night.isoformat(), HEARTBEAT))}.\n\n"
                "The usual causes are that the Mac was asleep or switched off at the "
                "scheduled time, that the app that runs the scheduled session was closed, "
                "or that its login had expired. To run the night by hand, from the main "
                f"checkout: `python3 scripts/nightly/run.py --date {night}`.\n")
        message = f"The nightly run of {night} did not run. See {where}."
    else:
        started = str(heartbeat.get("started_at") or "an unknown time")
        text = (f"# DID NOT FINISH: the nightly run of {night}\n\n"
                f"The nightly run of {night} started at {started}, but it had not "
                f"finished by the deadline, {shown}. The watchdog checked at "
                f"{now:%Y-%m-%d %H:%M}.\n\n"
                "The run was probably stopped partway, for example because the Mac shut "
                "down or the session running it was closed. It may also still be "
                "running, past every step's cap. Whatever it wrote is in this folder: "
                "report.md, if it got that far, and each step's output.\n")
        message = f"The nightly run of {night} started but did not finish. See {where}."
    record = {"night": night.isoformat(), "state": state, "deadline": shown,
              "checked_at": now.isoformat(), "heartbeat": heartbeat}
    (folder / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n",
                                         encoding="utf-8")
    report.write_text(text, encoding="utf-8")
    sent = bool(notify(TITLE, message)) if notify is not None else False
    return Verdict(night, state, report, sent)


def _read(path: pathlib.Path) -> dict[str, Any] | None:
    """The heartbeat, or None when there is none. A heartbeat that cannot be read still
    shows the run started, so it counts as a start with no finish."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def osascript_argv(title: str, message: str) -> list[str]:
    """The osascript command for one notification. The title and the message are passed
    as arguments to a fixed script, so no text is ever read as AppleScript."""
    return ["osascript", "-e", "on run argv",
            "-e", "display notification (item 2 of argv) with title (item 1 of argv)",
            "-e", "end run", title, message]


def notify(title: str, message: str, *,
           run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> bool:
    """Send one macOS notification. Whether it was sent: osascript exists only on macOS."""
    try:
        done = run(osascript_argv(title, message), check=False, capture_output=True,
                   text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return done.returncode == 0


def _clock_time(text: str) -> time:
    try:
        return datetime.strptime(text, "%H:%M").time()
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a time like 06:30") from None


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        prog="watchdog.py", description="Report a nightly run that did not run or did not "
        "finish by its deadline. It uses no model and does nothing else.")
    command.add_argument("--checkout", required=True, help="the main checkout")
    command.add_argument("--start", required=True, type=_clock_time,
                         help="when the nightly run is scheduled to start, as HH:MM")
    command.add_argument("--deadline", required=True, type=_clock_time,
                         help="when the run must have finished, as HH:MM; launchd runs "
                              "the watchdog at this time")
    command.add_argument("--now", type=datetime.fromisoformat,
                         help="check as if it were this time, for a test or by hand")
    command.add_argument("--no-notify", action="store_true",
                         help="write the report but send no notification")
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    now = args.now or datetime.now().astimezone()
    verdict = check(pathlib.Path(args.checkout), now=now, start=args.start,
                    deadline=args.deadline, notify=None if args.no_notify else notify)
    if verdict.state == "finished":
        print(f"The nightly run of {verdict.night} finished.")
    else:
        print(f"The nightly run of {verdict.night} {verdict.state.replace('-', ' ')}: "
              f"{verdict.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
