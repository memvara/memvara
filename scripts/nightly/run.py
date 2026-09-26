"""The adversarial suite's nightly run.

    python3 scripts/nightly/run.py [--date YYYY-MM-DD] [--file] [--no-notify]

It writes the night's heartbeat before anything else. Its preflight creates a clean
worktree of origin/main with a virtual environment of its own. Then it runs the design's
steps in order, each within its own time cap, and writes findings.jsonl, report.json and
report.md into local/nightly/<date>/ in the main checkout, and one record into
local/nightly/history.jsonl. Steps whose code has not landed are reported as not built
yet, with the reason.

A confirmed break seen for the first time is a new break: the run sends one notification
and plans its filing. It files only breaks a step has already classified, and only with
--file; a failed test is unclassified, so the scheduled session (task.md) classifies it
and files it with filing.py. Notifications go out only for new breaks, an isolation
breach, and a dependency that has been down two nights in a row.

--worktree and --python run against an existing checkout and interpreter instead, for a
supervised run; the report says when they were used. The testing guide's section "The
nightly run" describes the whole run.
"""

from __future__ import annotations

import argparse
import filecmp
import hashlib
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Iterator, Mapping, Sequence

if __package__ in (None, ""):  # run as a script, not imported as nightly.run
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from nightly import filing, flakes, night, regressions, render, steps, watchdog  # noqa: E402
from nightly.steps import run_steps  # noqa: E402

from harness.report import read as read_findings  # noqa: E402
from harness.report import write as write_findings  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
MINUTE = 60.0
#: Of the regressions step's cap, the time kept back for rerunning the tests that failed:
#: a fifth of the cap, and at most this.
RERUN_RESERVE = 15 * MINUTE
#: The most one rerun of one test may take.
RERUN_CAP = 10 * MINUTE
#: The operator's own files the isolation canary watches unless told otherwise. The suite
#: must never write them; the credentials file was overwritten by tests three times once.
CANARY = ("~/.memvara/credentials.json", "~/.memvara/db.key")
#: The extras the night's virtual environment installs, as CI does.
EXTRAS = "dev,cloud,ingest,encrypt"
#: How much of the end of the regressions output a session failure carries.
TAIL = 4000
CLI = "python3 scripts/nightly/filing.py"


@dataclass
class Night:
    """Everything the steps of one night share."""

    layout: night.Layout
    date: str
    filing_on: bool
    canary_paths: list[pathlib.Path]
    gh: filing.Runner
    worktree: pathlib.Path | None = None
    python: str | None = None
    given_worktree: bool = False
    given_python: bool = False
    commit: str = ""
    env: dict[str, str] = field(default_factory=dict)
    tmp: pathlib.Path | None = None
    canary_before: dict[str, str | None] | None = None
    dependencies: dict[str, str] = field(default_factory=dict)
    preflight: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    failures: list[regressions.Failure] = field(default_factory=list)
    layers: dict[str, dict[str, int]] = field(default_factory=dict)
    exitstatus: int | None = None

    @property
    def folder(self) -> pathlib.Path:
        return self.layout.night(self.date)

    def log(self, step: str) -> pathlib.Path:
        return self.folder / step.replace(" ", "-") / "output.log"


def preflight(context: Night, deadline: float) -> steps.Outcome:
    """Take the canary snapshot, create the worktree and the virtual environment, and
    check the logins the built steps need. The heartbeat is written before this runs."""
    notes = context.preflight
    notes.append("The heartbeat was written before anything else.")
    context.canary_before = snapshot(context.canary_paths)
    notes.append(f"The isolation canary hashed {len(context.canary_paths)} of the operator's "
                 "files, to compare after the night.")
    log = context.log("preflight")
    checkout = context.layout.checkout
    if context.given_worktree:
        assert context.worktree is not None
        context.commit = _git_output("-C", str(context.worktree), "rev-parse", "HEAD") or ""
        notes.append(f"The run tested the checkout it was given, {context.worktree}, at "
                     f"{context.commit[:12]}, not a fresh worktree of origin/main.")
    else:
        _remove_old_worktrees(context, log, deadline)
        fetched = steps.run_command(["git", "-C", str(checkout), "fetch", "--quiet", "origin"],
                                    cwd=checkout, env=dict(os.environ), log=log,
                                    deadline=deadline)
        context.dependencies["origin"] = "up" if fetched.returncode == 0 else "down"
        if fetched.returncode != 0:
            return steps.Outcome(steps.FAILED, "git could not fetch origin, so there was no "
                                 "origin/main to test; preflight/output.log says why.")
        commit = _git_output("-C", str(checkout), "rev-parse", "--verify", "origin/main")
        if not commit:
            return steps.Outcome(steps.FAILED, "origin has no main branch to test.")
        target = context.folder / "worktree"
        if target.exists():  # left by an earlier attempt at the same night
            steps.run_command(["git", "-C", str(checkout), "worktree", "remove", "--force",
                               str(target)], cwd=checkout, env=dict(os.environ), log=log,
                              deadline=deadline)
        added = steps.run_command(["git", "-C", str(checkout), "worktree", "add", "--detach",
                                   str(target), commit], cwd=checkout, env=dict(os.environ),
                                  log=log, deadline=deadline)
        if added.returncode != 0:
            return steps.Outcome(steps.FAILED, "git could not add the night's worktree; "
                                 "preflight/output.log says why.")
        context.worktree, context.commit = target, commit
        notes.append(f"A clean worktree of origin/main at {commit[:12]} was added at {target}.")
        differ = _differs(HERE, target / "scripts" / "nightly")
        if differ:
            context.warnings.append(
                f"The nightly run's own code differs from origin/main's in {', '.join(differ)}. "
                "The steps tested origin/main, but the code that ran them came from "
                f"{HERE}.")
    assert context.worktree is not None
    context.tmp = pathlib.Path(tempfile.mkdtemp(prefix="memvara-nightly-"))
    home = context.layout.home
    home.mkdir(parents=True, exist_ok=True)
    if context.given_python:
        assert context.python is not None
        notes.append(f"The steps ran with the given interpreter, {context.python}, not a "
                     "fresh virtual environment.")
    else:
        python = _build_venv(context, log, deadline)
        context.dependencies["packages"] = "up" if python else "down"
        if python is None:
            return steps.Outcome(steps.FAILED, "the virtual environment could not be built; "
                                 "preflight/output.log says why.")
        context.python = python
        notes.append(f"A virtual environment was built in the worktree, with .[{EXTRAS}] "
                     "installed as CI installs it.")
    context.env = night.step_env(os.environ, worktree=context.worktree, home=home,
                                 tmp=context.tmp, bin_dir=pathlib.Path(context.python).parent)
    if context.filing_on:
        login = context.gh(filing.Command(("gh", "auth", "status")))
        context.dependencies["github"] = "up" if login.returncode == 0 else "down"
        notes.append("gh is logged in." if login.returncode == 0
                     else "gh is not logged in, so filing will fail.")
    else:
        notes.append("The gh login was not checked, because filing is a dry run.")
    notes.append("Model resolution is not built yet: no built step uses a model. The agent "
                 "and red-team steps will need it when they land.")
    return steps.Outcome(steps.PASSED, f"Tested {context.commit[:12]}.")


def _git_output(*args: str) -> str | None:
    try:
        done = subprocess.run(["git", *args], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def _remove_old_worktrees(context: Night, log: pathlib.Path, deadline: float) -> None:
    """Remove the worktrees earlier nights left, unless one has uncommitted work in it, such
    as a strict-xfail test somebody is still writing."""
    checkout = context.layout.checkout
    for old in sorted(context.layout.root.glob("*/worktree")):
        if old.parent.name == context.date:
            continue
        done = steps.run_command(["git", "-C", str(checkout), "worktree", "remove", str(old)],
                                 cwd=checkout, env=dict(os.environ), log=log, deadline=deadline)
        context.preflight.append(
            f"The worktree of {old.parent.name} was removed." if done.returncode == 0 else
            f"The worktree of {old.parent.name} was kept, because git would not remove it: "
            "it may hold uncommitted work. preflight/output.log says why.")
    steps.run_command(["git", "-C", str(checkout), "worktree", "prune"], cwd=checkout,
                      env=dict(os.environ), log=log, deadline=deadline)


def _build_venv(context: Night, log: pathlib.Path, deadline: float) -> str | None:
    assert context.worktree is not None and context.tmp is not None
    venv = context.worktree / "local" / "venv"
    made = steps.run_command([sys.executable, "-m", "venv", str(venv)], cwd=context.worktree,
                             env=dict(os.environ), log=log, deadline=deadline)
    if made.returncode != 0:
        return None
    python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    env = night.step_env(os.environ, worktree=context.worktree, home=context.layout.home,
                         tmp=context.tmp, bin_dir=python.parent)
    installed = steps.run_command(
        [str(python), "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "-e",
         f"{context.worktree}[{EXTRAS}]"], cwd=context.worktree, env=env, log=log,
        deadline=deadline)
    return str(python) if installed.returncode == 0 else None


def _differs(here: pathlib.Path, tested: pathlib.Path) -> list[str]:
    """The files of this folder that origin/main does not have, or has with other text."""
    names = sorted(path.name for path in here.iterdir()
                   if path.is_file() and path.suffix in (".py", ".md", ".template"))
    return [name for name in names if not (tested / name).is_file()
            or not filecmp.cmp(here / name, tested / name, shallow=False)]


def regressions_step(*, command: Callable[[str, pathlib.Path], list[str]] = regressions.command,
                     rerun_command: Callable[..., list[str]] = flakes.rerun_command
                     ) -> Callable[[Night, float], steps.Outcome]:
    """The regressions step: the suite's nightly tier, then two reruns of each failed test.
    The test run may use the step's cap less a fifth of it, and at most RERUN_RESERVE less;
    the reruns get the rest."""
    def run(context: Night, deadline: float) -> steps.Outcome:
        assert context.worktree is not None and context.python is not None
        python, worktree = context.python, context.worktree
        folder = context.folder / regressions.FOLDER
        folder.mkdir(parents=True, exist_ok=True)
        results_file, log = folder / regressions.RESULTS, folder / regressions.OUTPUT
        reserve = min(RERUN_RESERVE, (deadline - time.monotonic()) / 5)
        ran = steps.run_command(command(python, results_file), cwd=worktree, env=context.env,
                                log=log, deadline=deadline - reserve)
        results = (regressions.read_results(results_file) if results_file.exists()
                   else regressions.Results([], None))
        if results.exitstatus is None and not ran.timed_out:
            # pytest died before it recorded its own exit status, for example on an import
            # error in a conftest file. The process's status is the next best thing, and
            # without it the night would have no failure to show.
            results = regressions.Results(results.tests, ran.returncode)

        def rerun(nodeid: str) -> tuple[str, ...] | None:
            if time.monotonic() >= deadline:
                return None
            argv = rerun_command(python, nodeid)
            said = []
            for _ in range(flakes.RERUNS):
                done = steps.run_command(
                    argv, cwd=worktree, env=context.env, log=folder / "reruns.log",
                    deadline=min(deadline, time.monotonic() + RERUN_CAP))
                said.append(flakes.outcome_of(done.returncode))
            return tuple(said)

        context.failures = regressions.failures(results, commit=context.commit, rerun=rerun,
                                                log_tail=_tail(log))
        context.exitstatus = results.exitstatus
        context.layers = _with_flakes(regressions.layers(results), context.failures)
        summary = _summary(results, context.failures)
        if ran.timed_out:
            return steps.Outcome(steps.TIMED_OUT, f"The test run reached its cap. {summary}")
        return steps.Outcome(steps.PASSED if ran.returncode == 0 else steps.FAILED, summary)
    return run


def _tail(log: pathlib.Path) -> str:
    return log.read_text(encoding="utf-8", errors="replace")[-TAIL:] if log.exists() else ""


def _with_flakes(layers: dict[str, dict[str, int]],
                 failures: Sequence[regressions.Failure]) -> dict[str, dict[str, int]]:
    counts = {layer: {**entry, "flaky": 0} for layer, entry in layers.items()}
    for failure in failures:
        if failure.kind == "test" and flakes.flaky(failure.reruns):
            counts.setdefault(failure.finding.layer, {"run": 0, "failed": 0, "flaky": 0})
            counts[failure.finding.layer]["flaky"] += 1
    return counts


def _summary(results: regressions.Results, failures: Sequence[regressions.Failure]) -> str:
    outcomes: dict[str, int] = {}
    for test in results.tests:
        outcomes[test["outcome"]] = outcomes.get(test["outcome"], 0) + 1
    parts = [f"{count:,} {outcome}" for outcome, count in sorted(outcomes.items())]
    verdicts: dict[str, int] = {}
    for failure in failures:
        verdicts[failure.verdict] = verdicts.get(failure.verdict, 0) + 1
    text = ", ".join(parts) or "no test results"
    if verdicts:
        text += "; the failures were " + ", ".join(
            f"{count} {verdict}" for verdict, count in sorted(verdicts.items()))
    exit_text = "unknown" if results.exitstatus is None else str(results.exitstatus)
    return f"{text}. pytest's exit status: {exit_text}."


#: The design's ten steps, in order. The caps of the steps that are not built yet are
#: placeholders for the work that builds them.
STEPS: tuple[steps.Step, ...] = (
    steps.Step("preflight", 20 * MINUTE, preflight, essential=True),
    steps.Step("regressions", 75 * MINUTE, regressions_step()),
    steps.Step("agents", 20 * MINUTE, waits_for="tests/live/agents", not_built=(
        "Real agents doing multi-session tasks, graded against the scenarios' gold, have "
        "not landed on main: the live tier's sandbox, agent driver, scenarios and grading.")),
    steps.Step("red team", 25 * MINUTE, waits_for="tests/live/redteam", not_built=(
        "The red team, an attacker told to break memvara whose confirmed breaks become "
        "regression tests, has not landed on main.")),
    steps.Step("hosted", 15 * MINUTE, waits_for="tests/live/stack.py", not_built=(
        "The hosted tier, which tests the client side against a throwaway local "
        "memvara-cloud stack, has not landed on main.")),
    steps.Step("production smoke", 5 * MINUTE, waits_for="tests/live/prod_smoke.py",
               not_built=("The read-mostly smoke test against app.memvara.dev, under a "
                          "dedicated test key, has not landed on main.")),
    steps.Step("performance", 15 * MINUTE, waits_for="bench/perf_budget.py", not_built=(
        "bench/perf_budget.py, which measures latency against the design's budget rule, has "
        "not landed on main. bench/perf.py is an older throughput profile with no budget, so "
        "it could not pass or fail a night.")),
    steps.Step("soak", 20 * MINUTE, waits_for="bench/soak.py", not_built=(
        "bench/soak.py, the 10,000-turn soak with a detector for each silent failure mode, "
        "has not landed on main.")),
    steps.Step("mutation", 15 * MINUTE, waits_for="bench/mutation.py", not_built=(
        "bench/mutation.py, the incremental mutation run over the changed functions, has not "
        "landed on main.")),
    steps.Step("replay", 10 * MINUTE, waits_for="tests/live/replay.py", not_built=(
        "The replay of the operator's recent agent transcripts through the hooks, against a "
        "throwaway store, has not landed on main.")),
)


def snapshot(paths: Sequence[pathlib.Path]) -> dict[str, str | None]:
    """A hash of each file, or None for a file that does not exist. Only the hashes are
    kept, never the contents."""
    hashes: dict[str, str | None] = {}
    for path in paths:
        try:
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        except FileNotFoundError:
            hashes[str(path)] = None
        except OSError as exc:
            hashes[str(path)] = f"unreadable: {type(exc).__name__}"
    return hashes


def _canary(context: Night) -> dict[str, Any] | None:
    """The canary's verdict: how many files it watched and which of them changed since
    preflight hashed them. None when preflight never took the snapshot."""
    before = context.canary_before
    if before is None:
        return None
    after = snapshot(context.canary_paths)
    return {"files": len(before),
            "changed": [path for path, value in before.items() if after.get(path) != value]}


def step_findings(context: Night, results: Sequence[steps.StepResult]
                  ) -> list[regressions.Failure]:
    """The findings the steps wrote into their folders. A step confirms a finding before
    it writes it, so each one is a confirmed break."""
    found = []
    for result in results:
        if result.name in ("preflight", "regressions"):
            continue
        path = context.folder / result.name.replace(" ", "-") / night.FINDINGS
        if not path.exists():
            continue
        try:
            findings = read_findings(path)
        except ValueError as exc:
            context.warnings.append(f"The findings of the {result.name} step could not be "
                                    f"read: {exc}")
            continue
        found += [regressions.Failure(finding, "finding", (), "confirmed")
                  for finding in findings]
    return found


def triage(failures: Sequence[regressions.Failure], history: Sequence[Mapping[str, Any]],
           remote: Mapping[str, Mapping[str, Any]] | None) -> dict[str, str]:
    """Each confirmed break's novelty: "new" when no earlier night saw it and nothing filed
    it, "recurred" when its issue on GitHub is closed, and "known" otherwise. `remote` is
    what GitHub holds, read only when filing is on."""
    seen = {fingerprint for record in history if record.get("kind") == "night"
            for fingerprint in record.get("confirmed", [])}
    filed = {record.get("fingerprint") for record in history if record.get("kind") == "filed"}
    novelty: dict[str, str] = {}
    for failure in failures:
        if not failure.fileable:
            continue
        fingerprint = failure.fingerprint
        known = (remote or {}).get(fingerprint)
        if known is not None and str(known.get("state", "")).upper() == "CLOSED":
            novelty[fingerprint] = "recurred"
        elif fingerprint in seen or fingerprint in filed or known is not None:
            novelty[fingerprint] = "known"
        else:
            novelty[fingerprint] = "new"
    return novelty


def plan(failure: regressions.Failure, context: Night,
         done: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """What filing one confirmed break still needs, with the exact commands. `done` holds
    what is filed for it already, by kind. A break its step classified is filed here, as a
    dry run unless filing is on; an unclassified one waits for the scheduled session."""
    fingerprint = failure.fingerprint
    common = f"--night {context.date} --fingerprint {fingerprint}"
    suffix = " --file" if context.filing_on else ""
    pr = (f"{CLI} pr {common} --worktree <a worktree whose last commit adds the "
          f"strict-xfail test>{suffix}")
    if "advisory" in done or ("issue" in done and "pr" in done):
        return {"needs": "nothing", "commands": []}
    if "issue" in done:
        return {"needs": "a strict-xfail test", "commands": [pr]}
    finding = failure.finding
    if finding.severity == "unclassified":
        return {"needs": "classification", "commands": [
            f"{CLI} advisory {common}{suffix}",
            f"{CLI} issue {common} --severity <data-loss|wrong-result|crash>{suffix}", pr]}
    paths = filing.default_paths(context.layout.checkout)
    try:
        if finding.severity == "security":
            result = filing.file_advisory(finding, fingerprint, night=context.date,
                                          gh=context.gh, dry_run=not context.filing_on,
                                          paths=paths)
        else:
            result = filing.file_issue(finding, fingerprint, night=context.date,
                                       gh=context.gh, dry_run=not context.filing_on,
                                       paths=paths)
    except filing.FilingError as exc:
        context.dependencies["github"] = "down"
        return {"needs": "filing", "commands": [], "error": str(exc)}
    commands = [command.shown() for command in result.commands]
    if result.dry_run:
        return {"needs": "filing", "commands": commands}
    record = result.record(night=context.date, severity=finding.severity)
    night.append_jsonl(context.layout.history, record)
    if result.what == "advisory":
        return {"needs": "nothing", "commands": commands, "filed": record}
    return {"needs": "a strict-xfail test", "commands": commands + [pr], "filed": record}


def _filed(history: Sequence[Mapping[str, Any]],
           remote: Mapping[str, Mapping[str, Any]] | None) -> dict[str, dict[str, Any]]:
    """What is filed for each fingerprint, by kind: the history's records, and what GitHub
    holds when filing is on."""
    done: dict[str, dict[str, Any]] = {}
    for record in history:
        if record.get("kind") == "filed":
            done.setdefault(str(record.get("fingerprint")), {})[str(record.get("what"))] = record
    for fingerprint, known in (remote or {}).items():
        kind = "advisory" if "ghsa_id" in known else "issue"
        done.setdefault(fingerprint, {}).setdefault(kind, dict(known))
    return done


def _remote(context: Night) -> dict[str, dict[str, Any]] | None:
    """What GitHub holds for every marked break, when filing is on; nothing is read in a
    dry run."""
    if not context.filing_on:
        return None
    try:
        return {**filing.existing_advisories(context.gh), **filing.existing_issues(context.gh)}
    except (filing.FilingError, ValueError) as exc:
        context.dependencies["github"] = "down"
        context.warnings.append(f"GitHub could not be read, so filing was skipped: {exc}")
        return None


def notifications(report: Mapping[str, Any], history: Sequence[Mapping[str, Any]]
                  ) -> list[str]:
    """The night's notifications: a new break, an isolation breach, and a dependency down
    for two nights in a row. A dependency down for longer was notified already."""
    date = report["night"]
    where = f"See local/nightly/{date}/report.md."
    messages = []
    fresh = [entry for entry in report["failures"]
             if entry.get("novelty") in ("new", "recurred")]
    if fresh:
        noun = "break" if len(fresh) == 1 else "breaks"
        messages.append(f"{len(fresh)} new {noun} on the night of {date}. {where}")
    changed = report["canary"]["changed"]
    if changed:
        messages.append(f"Isolation breach on the night of {date}: {len(changed)} of the "
                        f"operator's protected files changed during the run. {where}")
    for dependency in _down_two_nights(date, report["dependencies"], history):
        messages.append(f"{dependency} has been down for two nights in a row. {where}")
    return messages


def _down_two_nights(date: str, dependencies: Mapping[str, str],
                     history: Sequence[Mapping[str, Any]]) -> Iterator[str]:
    nights = {str(record["date"]): record for record in history
              if record.get("kind") == "night" and str(record.get("date")) < date}
    earlier = [nights[day] for day in sorted(nights, reverse=True)]
    for dependency, state in dependencies.items():
        if state != "down":
            continue
        streak = 1
        for record in earlier:
            if (record.get("dependencies") or {}).get(dependency) != "down":
                break
            streak += 1
        if streak == 2:
            yield dependency


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        prog="run.py", description="Run the adversarial suite's nightly steps against a clean "
        "worktree of origin/main, and write the night's report.")
    command.add_argument("--checkout", help="the main checkout, where local/nightly is; the "
                         "one this file belongs to by default")
    command.add_argument("--date", help="the night's name, YYYY-MM-DD; today by default")
    command.add_argument("--worktree", help="test this existing checkout instead of a fresh "
                         "worktree of origin/main")
    command.add_argument("--python", help="run the steps with this interpreter instead of a "
                         "fresh virtual environment")
    command.add_argument("--canary", action="append", default=[], metavar="PATH",
                         help="another of the operator's files the run must not change")
    command.add_argument("--file", action="store_true",
                         help="file classified breaks on GitHub; without this, filing is a "
                              "dry run that sends nothing")
    command.add_argument("--no-notify", action="store_true",
                         help="write the report but send no notification")
    return command


def main(argv: Sequence[str] | None = None, *, steps: Sequence[steps.Step] = STEPS,
         notify: Callable[[str, str], object] = watchdog.notify,
         gh: filing.Runner = filing.subprocess_runner,
         clock: Callable[[], datetime] = night.now) -> int:
    args = parser().parse_args(argv)
    checkout = (pathlib.Path(args.checkout) if args.checkout
                else night.main_checkout(HERE)).resolve()
    started = clock()
    context = Night(
        layout=night.Layout(checkout), date=args.date or started.date().isoformat(),
        filing_on=args.file, gh=gh,
        canary_paths=[pathlib.Path(path).expanduser() for path in (*CANARY, *args.canary)])
    context.folder.mkdir(parents=True, exist_ok=True)
    heartbeat = context.folder / night.HEARTBEAT
    night.write_heartbeat(heartbeat, night=context.date, started_at=started.isoformat())
    if args.worktree:
        context.worktree, context.given_worktree = pathlib.Path(args.worktree).resolve(), True
    if args.python:
        context.python, context.given_python = args.python, True
    history, bad = night.read_jsonl(context.layout.history)
    if bad:
        context.warnings.append(f"Lines {', '.join(map(str, bad))} of history.jsonl could not "
                                "be read and were skipped.")
    report: dict[str, Any] = {"version": 1, "night": context.date, "status": "running",
                              "started_at": started.isoformat(), "finished_at": None,
                              "checkout": str(checkout),
                              "filing": "file" if args.file else "dry-run",
                              "steps": [], "failures": [], "notifications": []}
    failures: list[regressions.Failure] = []
    status, error = "crashed", None
    try:
        results = run_steps(steps, context, worktree=lambda: context.worktree)
        report["steps"] = [result.to_dict() for result in results]
        failures = context.failures + step_findings(context, results)
        report["failures"] = [failure.to_record() for failure in failures]
        remote = _remote(context)
        novelty = triage(failures, history, remote)
        done = _filed(history, remote)
        entries = []
        for failure in failures:
            entry = failure.to_record()
            if failure.fileable:
                entry["novelty"] = novelty[failure.fingerprint]
                entry["plan"] = plan(failure, context, done.get(failure.fingerprint, {}))
            entries.append(entry)
        report.update(_facts(context), failures=entries,
                      canary=_canary(context) or {"files": 0, "changed": []},
                      flake_rates={layer: rate.to_dict() for layer, rate in flakes.rates(
                          [*history, {"kind": "night", "date": context.date,
                                      "layers": context.layers}]).items()})
        sent = []
        for message in notifications(report, history):
            delivered = False if args.no_notify else bool(notify(watchdog.TITLE, message))
            sent.append({"message": message, "sent": delivered})
        report["notifications"] = sent
        status = "finished"
    except Exception:  # noqa: BLE001 - the report must be written whatever went wrong
        error = traceback.format_exc()
        report.update(_facts(context))
        if report.get("canary") is None:
            # The canary outranks everything else, so a crash must not skip comparing it.
            report["canary"] = _canary(context)
    finally:
        finished = clock()
        report.update(status=status, error=error, finished_at=finished.isoformat())
        write_findings(context.folder / night.FINDINGS, [each.finding for each in failures])
        night.write_json(context.folder / night.REPORT_JSON, report)
        (context.folder / night.REPORT_MD).write_text(render.markdown(report),
                                                      encoding="utf-8", newline="\n")
        night.append_jsonl(context.layout.history, _record(context, report, failures))
        night.write_heartbeat(heartbeat, night=context.date, started_at=started.isoformat(),
                              finished_at=finished.isoformat() if status == "finished"
                              else None, status=status)
        if context.tmp is not None:
            shutil.rmtree(context.tmp, ignore_errors=True)
    print(f"The nightly run of {context.date} {status}: {context.folder / night.REPORT_MD}")
    return 0 if status == "finished" else 1


def _facts(context: Night) -> dict[str, Any]:
    return {"worktree": str(context.worktree) if context.worktree else None,
            "commit": context.commit, "python": context.python,
            "given": {"worktree": context.given_worktree, "python": context.given_python},
            "preflight": context.preflight, "warnings": context.warnings,
            "dependencies": context.dependencies,
            "tests": {"exitstatus": context.exitstatus, "layers": context.layers}}


def _record(context: Night, report: Mapping[str, Any],
            failures: Sequence[regressions.Failure]) -> dict[str, Any]:
    """The night's line in the history: what a later night needs to deduplicate breaks,
    measure flake rates and notice a dependency that stays down."""
    canary = report.get("canary")
    return {"kind": "night", "version": 1, "date": context.date, "status": report["status"],
            "commit": context.commit,
            "steps": {step["name"]: step["status"] for step in report.get("steps", [])},
            "layers": context.layers,
            "confirmed": sorted({failure.fingerprint for failure in failures
                                 if failure.fileable}),
            "dependencies": context.dependencies,
            "canary": ("not taken" if not canary else
                       "changed" if canary["changed"] else "clean"),
            "filing": report["filing"]}


if __name__ == "__main__":
    sys.exit(main())
