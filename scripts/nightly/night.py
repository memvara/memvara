"""Where a night's files go, the running history, the heartbeat, and the environment the
steps run in.

Every file the nightly run writes is under local/nightly/ in the main checkout, which git
ignores:

    local/nightly/history.jsonl       one record per night, and one per filing
    local/nightly/home/               the HOME every step runs with, kept between nights
    local/nightly/<date>/             one folder per night, named by the day the run started
        heartbeat.json                written first, and again when the run finishes
        report.json, report.md        what happened, for a program and for a person
        findings.jsonl                every break the night saw, one Finding per line
        <step>/                       each step's output and results
        worktree/                     the clean worktree of origin/main that was tested
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Mapping

# The prefixes of the variables no child process of the suite may inherit: the
# harness's own list, so the two can never disagree.
from harness.env import _DROPPED as DROPPED_PREFIXES

#: The nightly folder, relative to the main checkout.
NIGHTLY = ("local", "nightly")
HEARTBEAT = "heartbeat.json"
HISTORY = "history.jsonl"
REPORT_JSON = "report.json"
REPORT_MD = "report.md"
FINDINGS = "findings.jsonl"


@dataclass(frozen=True)
class Layout:
    """The nightly folder of one checkout."""

    checkout: pathlib.Path

    @property
    def root(self) -> pathlib.Path:
        return self.checkout.joinpath(*NIGHTLY)

    @property
    def history(self) -> pathlib.Path:
        return self.root / HISTORY

    @property
    def home(self) -> pathlib.Path:
        """The HOME every step runs with. It is kept from night to night, because the
        nightly Hypothesis profile keeps the examples it found under the home directory,
        so that a failure found one night is tried first the next night."""
        return self.root / "home"

    def night(self, date: str) -> pathlib.Path:
        """The folder of the night named `date`, which must be YYYY-MM-DD."""
        return self.root / night_date(date)


def night_date(text: str) -> str:
    """`text`, when it is a real date written YYYY-MM-DD, the name the watchdog looks for.
    A night saved under any other name would never be checked, and the watchdog would
    report it as a night that did not run."""
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            date.fromisoformat(text)
            return text
    except ValueError:
        pass
    raise ValueError(f"{text!r} is not a night's date: a night is named YYYY-MM-DD, such as "
                     "2026-09-27")


def main_checkout(path: pathlib.Path | str) -> pathlib.Path:
    """The checkout that owns the repository `path` is in, even when `path` is one of its
    worktrees, so a run started from any worktree writes its reports to one place."""
    common = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=path, check=True,
                            capture_output=True, text=True).stdout.strip()
    return (pathlib.Path(path) / common).resolve().parent


def write_json(path: pathlib.Path, data: Any) -> None:
    """Write `data` as JSON, all at once: a reader sees the old file or the new one, never
    half of one. The watchdog and the next night both read these files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as out:
            json.dump(data, out, indent=2, ensure_ascii=False)
            out.write("\n")
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: pathlib.Path, record: Mapping[str, Any]) -> None:
    """Add one record to the end of a file of JSON lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as out:
        out.write(json.dumps(record, ensure_ascii=True) + "\n")


def read_jsonl(path: pathlib.Path) -> tuple[list[dict[str, Any]], list[int]]:
    """The records in a file of JSON lines, and the numbers of the lines that are not
    records. A run killed while it appended leaves half a line, and that line must cost
    only itself: every later night reads this file."""
    if not path.exists():
        return [], []
    records, bad = [], []
    for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            bad.append(number)
            continue
        if isinstance(record, dict):
            records.append(record)
        else:
            bad.append(number)
    return records, bad


def write_heartbeat(path: pathlib.Path, *, night: str, started_at: str,
                    finished_at: str | None = None, status: str = "running") -> None:
    """The file the watchdog looks for. The run writes it before anything that can fail,
    and again when it finishes; a heartbeat with no finish means the run died partway."""
    write_json(path, {"night": night, "started_at": started_at,
                      "finished_at": finished_at, "status": status})


def step_env(base: Mapping[str, str], *, worktree: pathlib.Path, home: pathlib.Path,
             tmp: pathlib.Path, bin_dir: pathlib.Path | None = None) -> dict[str, str]:
    """The environment the steps run in.

    It is `base` without the variables the harness keeps from every child process, with
    HOME pointed at the nightly home, a private temporary folder, and the worktree on
    PYTHONPATH so the tested code is what gets imported. It sets no MEMVARA_ variable,
    so the suite runs with the defaults it runs with in CI.
    """
    env = {key: value for key, value in base.items() if not key.startswith(DROPPED_PREFIXES)}
    env.update({"HOME": str(home), "USERPROFILE": str(home), "TMPDIR": str(tmp),
                "TMP": str(tmp), "TEMP": str(tmp), "PYTHONPATH": str(worktree)})
    if bin_dir is not None:
        env["PATH"] = os.pathsep.join(
            [str(bin_dir)] + ([base["PATH"]] if base.get("PATH") else []))
    return env


def now() -> datetime:
    """The local time, with its offset from UTC."""
    return datetime.now().astimezone()
