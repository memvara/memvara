"""The nightly run, end to end, against a small repository with a bare origin.

The run writes its heartbeat first, creates a clean worktree of origin/main, runs every
step in order within its cap, and writes findings.jsonl, report.json and report.md into
the night's folder and one record into the history. The real preflight runs here, against
a temporary repository. The regressions step runs with a canned pytest result, and its
parsing, reruns and triage are the real ones. Nothing here builds a virtual environment or
calls GitHub: the run is given this interpreter, and filing is a dry run.
"""

from __future__ import annotations

import json
import pathlib
import re
import shlex
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import pytest

from harness.env import REPO, child_env
from harness.report import Finding
from harness.report import write as write_findings

if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))
from nightly import filing, flakes, night, render, run, steps, watchdog  # noqa: E402

ZONE = timezone(timedelta(hours=2))
NODEID = "tests/adversarial/model/test_adv_model_machine.py::test_random_operations"
FAILED = {"nodeid": NODEID, "outcome": "failed", "when": "call",
          "message": "the store and the model disagree",
          "longrepr": "E   drive.replay([\nE       Forget(user='u1'),\nE   ])"}
PASSED = {"nodeid": "tests/test_api.py::test_a_read", "outcome": "passed", "when": "call",
          "message": "", "longrepr": ""}


def _git(*args: str, cwd: pathlib.Path, env: dict[str, str]) -> str:
    return subprocess.run(["git", "-c", "user.name=nightly test",
                           "-c", "user.email=nightly@example.invalid", *args],
                          cwd=cwd, env=env, check=True, capture_output=True,
                          text=True).stdout.strip()


@pytest.fixture
def git_env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    return child_env(tmp_path_factory.mktemp("git-home"))


@pytest.fixture
def repo(tmp_path: pathlib.Path, git_env: dict[str, str]) -> pathlib.Path:
    """A main checkout whose origin is a bare repository with one commit on main."""
    origin = tmp_path / "origin.git"
    _git("init", "-q", "--bare", str(origin), cwd=tmp_path, env=git_env)
    checkout = tmp_path / "main"
    _git("clone", "-q", str(origin), str(checkout), cwd=tmp_path, env=git_env)
    (checkout / "README.md").write_text("a checkout for the nightly tests\n")
    _git("add", "README.md", cwd=checkout, env=git_env)
    _git("commit", "-q", "-m", "first", cwd=checkout, env=git_env)
    _git("push", "-q", "origin", "HEAD:refs/heads/main", cwd=checkout, env=git_env)
    return checkout


class Notifications:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def __call__(self, title: str, message: str) -> bool:
        self.sent.append((title, message))
        return True


def never(command: filing.Command) -> filing.Completed:
    raise AssertionError(f"a dry run must call nothing, and it ran {command.argv}")


def _pytest_result(tmp_path: pathlib.Path, tests: list[dict[str, Any]], exitstatus: int
                   ) -> Callable[[str, pathlib.Path], list[str]]:
    """A stand-in for the regressions command: it writes a canned results file, as
    pytest_results.py would, and exits with pytest's status."""
    canned = tmp_path / f"canned-{len(tests)}-{exitstatus}.jsonl"
    canned.write_text("".join(json.dumps(test) + "\n" for test in tests)
                      + json.dumps({"exitstatus": exitstatus}) + "\n")

    def command(python: str, results: pathlib.Path) -> list[str]:
        return [python, "-c", "import shutil, sys; shutil.copyfile(sys.argv[1], sys.argv[2]); "
                "sys.exit(int(sys.argv[3]))", str(canned), str(results), str(exitstatus)]
    return command


def _reruns_exit(code: int) -> Callable[..., list[str]]:
    def command(python: str, nodeid: str, tier: str = "nightly") -> list[str]:
        return [python, "-c", f"raise SystemExit({code})"]
    return command


def _table(tmp_path: pathlib.Path, tests: list[dict[str, Any]], exitstatus: int, *,
           rerun_code: int = 1, **replace: steps.Step) -> tuple[steps.Step, ...]:
    """The run's own step table, with a canned regressions result, and any step replaced
    by name."""
    regressions = steps.Step("regressions", 600.0, run.regressions_step(
        command=_pytest_result(tmp_path, tests, exitstatus),
        rerun_command=_reruns_exit(rerun_code)))
    table = []
    for step in run.STEPS:
        if step.name == "regressions":
            table.append(regressions)
        else:
            table.append(replace.get(step.name.replace(" ", "_"), step))
    return tuple(table)


def _night(repo: pathlib.Path, date: str, table: tuple[steps.Step, ...], *,
           notify: Callable[[str, str], Any], extra: tuple[str, ...] = ()) -> int:
    started = datetime.fromisoformat(f"{date}T01:30:00+02:00")
    return run.main(["--checkout", str(repo), "--date", date, "--python", sys.executable,
                     *extra], steps=table, notify=notify, gh=never, clock=lambda: started)


def _report(repo: pathlib.Path, date: str) -> dict[str, Any]:
    return json.loads((night.Layout(repo).night(date) / "report.json").read_text())


def _history(repo: pathlib.Path) -> list[dict[str, Any]]:
    records, bad = night.read_jsonl(night.Layout(repo).history)
    assert bad == []
    return records


def test_a_night_tests_a_clean_worktree_of_origin_main_and_writes_every_file(
        repo: pathlib.Path, tmp_path: pathlib.Path, git_env: dict[str, str]) -> None:
    """The night tests what is on origin/main, not what happens to be checked out, and
    leaves a report, a history record and a finished heartbeat even when nothing broke."""
    _git("commit", "-q", "--allow-empty", "-m", "local work nobody pushed", cwd=repo,
         env=git_env)
    code = _night(repo, "2026-09-27", _table(tmp_path, [PASSED], 0), notify=Notifications())
    assert code == 0
    folder = night.Layout(repo.resolve()).night("2026-09-27")
    origin_main = _git("rev-parse", "origin/main", cwd=repo, env=git_env)
    worktree = folder / "worktree"
    assert _git("rev-parse", "HEAD", cwd=worktree, env=git_env) == origin_main
    assert _git("rev-parse", "HEAD", cwd=repo, env=git_env) != origin_main
    report = _report(repo, "2026-09-27")
    assert (report["status"], report["commit"], report["worktree"]) == (
        "finished", origin_main, str(worktree))
    assert [(step["name"], step["status"]) for step in report["steps"]] == [
        ("preflight", "passed"), ("regressions", "passed"), ("agents", "not built yet"),
        ("red team", "not built yet"), ("hosted", "not built yet"),
        ("production smoke", "not built yet"), ("performance", "not built yet"),
        ("soak", "not built yet"), ("mutation", "not built yet"),
        ("replay", "not built yet")]
    assert any("given interpreter" in line for line in report["preflight"])
    assert any("not built yet" in line for line in report["preflight"])
    heartbeat = json.loads((folder / "heartbeat.json").read_text())
    assert (heartbeat["status"], heartbeat["started_at"]) == (
        "finished", "2026-09-27T01:30:00+02:00")
    assert heartbeat["finished_at"]
    assert (folder / "findings.jsonl").read_text() == ""
    assert "# Nightly run of 2026-09-27" in (folder / "report.md").read_text()
    [record] = _history(repo)
    assert (record["kind"], record["date"], record["commit"], record["status"]) == (
        "night", "2026-09-27", origin_main, "finished")
    assert record["layers"] == {"unit": {"run": 1, "failed": 0, "flaky": 0}}


def test_every_step_the_design_names_is_in_the_table_and_an_unbuilt_step_says_why() -> None:
    """A step that is not built yet must still appear in every report, with the reason
    and the file it waits for, or its absence would look like a pass."""
    assert [step.name for step in run.STEPS] == [
        "preflight", "regressions", "agents", "red team", "hosted", "production smoke",
        "performance", "soak", "mutation", "replay"]
    assert run.STEPS[0].essential
    assert [step.name for step in run.STEPS if step.run is not None] == [
        "preflight", "regressions"]
    for step in run.STEPS[2:]:
        assert step.not_built and step.waits_for, step.name


def test_a_break_seen_on_two_nights_is_new_once_and_planned_until_it_is_filed(
        repo: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """The second night recognises the break by its fingerprint: no second notification,
    no second issue. It stays in the plan until it is filed, and the plan's first need is
    a classification, because a failed test is never filed unclassified."""
    table = _table(tmp_path, [PASSED, FAILED], 1)
    notify = Notifications()
    for date in ("2026-09-27", "2026-09-28"):
        assert _night(repo, date, table, notify=notify) == 0
    first, second = (_report(repo, date)["failures"] for date in ("2026-09-27", "2026-09-28"))
    assert [(entry["verdict"], entry["novelty"], entry["plan"]["needs"]) for entry in first] == [
        ("confirmed", "new", "classification")]
    assert [(entry["verdict"], entry["novelty"], entry["plan"]["needs"]) for entry in second] == [
        ("confirmed", "known", "classification")]
    fingerprint = first[0]["fingerprint"]
    assert second[0]["fingerprint"] == fingerprint
    assert any(f"--fingerprint {fingerprint}" in command
               for command in first[0]["plan"]["commands"])
    assert len(notify.sent) == 1 and "1 new break" in notify.sent[0][1]
    assert not (night.Layout(repo).night("2026-09-27") / "worktree").exists()
    report = _report(repo, "2026-09-28")
    assert report["tests"]["layers"]["model"] == {"run": 1, "failed": 1, "flaky": 0}
    assert report["flake_rates"]["model"]["run"] == 2
    markdown = (night.Layout(repo).night("2026-09-27") / "report.md").read_text()
    for step in run.STEPS:
        assert step.name in markdown
    assert NODEID in markdown and fingerprint[:12] in markdown
    history = night.Layout(repo).history
    night.append_jsonl(history, {"kind": "filed", "date": "2026-09-28", "what": "issue",
                                 "fingerprint": fingerprint, "severity": "data-loss",
                                 "number": 302, "url": "https://github.com/x/y/issues/302",
                                 "state": "OPEN", "existing": False})
    assert _night(repo, "2026-09-29", table, notify=notify) == 0
    [third] = _report(repo, "2026-09-29")["failures"]
    assert third["plan"]["needs"] == "a strict-xfail test"
    night.append_jsonl(history, {"kind": "filed", "date": "2026-09-29", "what": "pr",
                                 "fingerprint": fingerprint, "severity": "data-loss",
                                 "number": 303, "url": "https://github.com/x/y/pull/303",
                                 "state": "OPEN", "existing": False})
    assert _night(repo, "2026-09-30", table, notify=notify) == 0
    [fourth] = _report(repo, "2026-09-30")["failures"]
    assert (fourth["plan"]["needs"], fourth["plan"]["commands"]) == ("nothing", [])
    assert len(notify.sent) == 1


def test_a_flake_is_counted_and_never_planned(repo: pathlib.Path,
                                              tmp_path: pathlib.Path) -> None:
    notify = Notifications()
    _night(repo, "2026-09-27", _table(tmp_path, [FAILED], 1, rerun_code=0), notify=notify)
    report = _report(repo, "2026-09-27")
    assert [(entry["verdict"], entry.get("novelty"), entry.get("plan")) for entry in
            report["failures"]] == [("flake", None, None)]
    assert report["tests"]["layers"]["model"] == {"run": 1, "failed": 1, "flaky": 1}
    assert notify.sent == []


def test_a_security_class_finding_a_step_confirmed_is_planned_as_an_advisory_only(
        repo: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """A finding its step has classified as security-class is filed without waiting for a
    person, and only ever to a private advisory."""
    def soak(context: Any, deadline: float) -> steps.Outcome:
        folder = context.folder / "soak"
        folder.mkdir(parents=True)
        write_findings(folder / "findings.jsonl", [Finding(
            "soak", "library", "no read returns another user's row", severity="security",
            ops=("remember u1", "search as u2"), commit=context.commit,
            title="a read by one user returned another user's row")])
        return steps.Outcome(steps.PASSED, "10,000 turns")

    table = _table(tmp_path, [PASSED], 0, soak=steps.Step("soak", 60.0, soak))
    _night(repo, "2026-09-27", table, notify=Notifications())
    [entry] = _report(repo, "2026-09-27")["failures"]
    assert (entry["kind"], entry["novelty"], entry["plan"]["needs"]) == (
        "finding", "new", "filing")
    commands = entry["plan"]["commands"]
    assert any("security-advisories" in command for command in commands)
    assert not any("issue create" in command or "pr create" in command
                   for command in commands)
    folder = night.Layout(repo).night("2026-09-27")
    assert [json.loads(line)["layer"] for line in
            (folder / "findings.jsonl").read_text().splitlines()] == ["soak"]


def test_a_protected_file_changed_during_the_night_is_an_isolation_breach(
        repo: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """The canary hashes the operator's own files before the night and after it. A step
    that reached one of them broke the suite's isolation, which outranks any finding."""
    canary = tmp_path / "credentials.json"
    canary.write_text('{"api_key": "before"}')

    def meddles(context: Any, deadline: float) -> steps.Outcome:
        canary.write_text('{"api_key": "after"}')
        return steps.Outcome(steps.PASSED)

    notify = Notifications()
    table = _table(tmp_path, [PASSED], 0, agents=steps.Step("agents", 60.0, meddles))
    _night(repo, "2026-09-27", table, notify=notify, extra=("--canary", str(canary)))
    report = _report(repo, "2026-09-27")
    assert report["canary"]["changed"] == [str(canary)]
    assert [message for _, message in notify.sent if "isolation" in message.lower()]
    assert _history(repo)[-1]["canary"] == "changed"


def test_a_dependency_down_two_nights_running_is_notified_once(
        repo: pathlib.Path, tmp_path: pathlib.Path, git_env: dict[str, str]) -> None:
    """One night without the origin can be a blip. Two in a row is worth a notification,
    and a third is the same outage, not a new one."""
    _git("remote", "set-url", "origin", str(tmp_path / "missing.git"), cwd=repo,
         env=git_env)
    sent: dict[str, list[str]] = {}
    for date in ("2026-09-27", "2026-09-28", "2026-09-29"):
        notify = Notifications()
        _night(repo, date, _table(tmp_path, [PASSED], 0), notify=notify)
        sent[date] = [message for _, message in notify.sent]
        report = _report(repo, date)
        assert report["dependencies"]["origin"] == "down"
        assert [(step["name"], step["status"]) for step in report["steps"]][:2] == [
            ("preflight", "failed"), ("regressions", "not run")]
    assert sent["2026-09-27"] == [] and sent["2026-09-29"] == []
    assert len(sent["2026-09-28"]) == 1 and "origin" in sent["2026-09-28"][0]


def test_a_step_that_raises_costs_that_step_and_the_report_is_whole(
        repo: pathlib.Path, tmp_path: pathlib.Path) -> None:
    def broken(context: Any, deadline: float) -> steps.Outcome:
        raise RuntimeError("the soak's store could not be opened")

    table = _table(tmp_path, [PASSED], 0, soak=steps.Step("soak", 60.0, broken))
    assert _night(repo, "2026-09-27", table, notify=Notifications()) == 0
    report = _report(repo, "2026-09-27")
    assert report["status"] == "finished"
    assert {step["name"]: step["status"] for step in report["steps"]}["soak"] == "error"


def test_a_crash_of_the_run_itself_leaves_a_report_and_a_heartbeat_with_no_finish(
        repo: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """A crash of the run's own code must not look like a finished night: the report says
    it crashed, and the heartbeat has no finish, so the watchdog reports it too."""
    def broken_notify(title: str, message: str) -> bool:
        raise RuntimeError("the notification centre is gone")

    code = _night(repo, "2026-09-27", _table(tmp_path, [FAILED], 1), notify=broken_notify)
    assert code == 1
    report = _report(repo, "2026-09-27")
    assert report["status"] == "crashed" and "notification centre" in report["error"]
    folder = night.Layout(repo).night("2026-09-27")
    heartbeat = json.loads((folder / "heartbeat.json").read_text())
    assert (heartbeat["status"], heartbeat["finished_at"]) == ("crashed", None)
    assert _history(repo)[-1]["status"] == "crashed"
    assert "crashed" in (folder / "report.md").read_text()


def _step(name: str, status: str) -> dict[str, Any]:
    return {"name": name, "status": status, "seconds": 1.0, "cap": 60.0, "summary": ""}


def test_the_reports_first_line_never_calls_a_night_with_a_failed_step_quiet() -> None:
    """A night whose preflight failed ran no tests, so it found no break. Its report must
    lead with the failed step, or it reads exactly like a night where everything passed."""
    report = {"night": "2026-09-27", "status": "finished", "commit": "",
              "steps": [_step("preflight", "failed"), _step("regressions", "not run"),
                        _step("agents", "not built yet")], "failures": []}
    lead = render.markdown(report).splitlines()[2]
    assert "Nothing broke" not in lead
    assert "preflight failed" in lead and "regressions not run" in lead
    assert "agents" not in lead
    quiet = dict(report, steps=[_step("preflight", "passed"), _step("regressions", "passed"),
                                _step("agents", "not built yet")])
    assert "Nothing broke" in render.markdown(quiet).splitlines()[2]


def test_a_test_run_that_dies_before_writing_a_result_is_a_failure_for_a_person(
        repo: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """pytest can die before it records anything, for example on an import error in a
    conftest file. That night has no failed test to report, and must not look quiet."""
    def dies(python: str, results: pathlib.Path) -> list[str]:
        return [python, "-c", "raise SystemExit(3)"]

    table = tuple(steps.Step("regressions", 600.0, run.regressions_step(command=dies))
                  if step.name == "regressions" else step for step in run.STEPS)
    _night(repo, "2026-09-27", table, notify=Notifications())
    report = _report(repo, "2026-09-27")
    assert [(entry["kind"], entry["finding"]["invariant"]) for entry in report["failures"]] == [
        ("session", "exit status 3")]
    assert {step["name"]: step["status"] for step in report["steps"]}["regressions"] == "failed"


def test_a_breach_during_a_night_that_crashed_is_still_recorded(
        repo: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """The canary outranks everything else in the report, so a crash of the run's own code
    after the steps must not skip comparing it."""
    canary = tmp_path / "db.key"
    canary.write_text("before")

    def meddles(context: Any, deadline: float) -> steps.Outcome:
        canary.write_text("after")
        return steps.Outcome(steps.PASSED)

    def gh(command: filing.Command) -> filing.Completed:
        if command.argv[:3] == ("gh", "auth", "status"):
            return filing.Completed(0)
        raise RuntimeError("the network went away")

    table = _table(tmp_path, [PASSED], 0, agents=steps.Step("agents", 60.0, meddles))
    started = datetime(2026, 9, 27, 1, 30, tzinfo=ZONE)
    code = run.main(["--checkout", str(repo), "--date", "2026-09-27", "--python",
                     sys.executable, "--file", "--canary", str(canary)],
                    steps=table, notify=Notifications(), gh=gh, clock=lambda: started)
    assert code == 1
    report = _report(repo, "2026-09-27")
    assert report["status"] == "crashed"
    assert report["canary"]["changed"] == [str(canary)]
    assert _history(repo)[-1]["canary"] == "changed"


def test_every_command_the_nightly_session_is_told_to_run_is_one_the_scripts_accept() -> None:
    """The scheduled session follows task.md with nobody watching. A command it names that
    a script no longer accepts would fail in the night, so each command in the prompt is
    parsed with its script's own parser, and all three filing commands must be there."""
    text = (REPO / "scripts" / "nightly" / "task.md").read_text(encoding="utf-8")
    commands = re.findall(r"python3 (?:[^\s`]*/)?scripts/nightly/(\w+)\.py([^`\n]*)", text)
    parsers = {"run": run.parser(), "filing": filing.parser(), "flakes": flakes.parser(),
               "watchdog": watchdog.parser()}
    assert {name for name, _ in commands} >= {"run", "filing"}
    filed = set()
    for name, rest in commands:
        rest = rest.replace("<data-loss|wrong-result|crash>", "data-loss")
        args = shlex.split(re.sub(r"<[^<>\s]+>", "x", rest))
        try:
            parsers[name].parse_args(args)
        except SystemExit:
            pytest.fail(f"task.md runs {name}.py{rest}, which that script refuses")
        if name == "filing":
            filed.add(args[0])
    assert filed == {"issue", "advisory", "pr"}
