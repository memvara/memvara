"""Filing a new confirmed break, which is a dry run unless the operator asks for more.

For a public break the filing opens one issue, labelled `nightly-break`, with a hidden
marker holding the fingerprint, so a later night finds the issue and files nothing new.
Then it pushes a branch holding the strict-xfail test and opens a draft pull request. A
security-class break goes to a private draft advisory instead, and never to an issue, a
branch or a pull request. An unclassified break is filed nowhere public.

Nothing here calls GitHub. A dry run calls nothing at all, and the other tests replay the
output gh and git print, from gh_recorded.json, through a fake runner that also checks
each command it is given.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import sys
from typing import Any

import pytest

from harness.env import REPO
from harness.report import Finding

if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))
from nightly import filing, night, regressions  # noqa: E402 - scripts/ is on the path above

RECORDED = json.loads(pathlib.Path(__file__).with_name("gh_recorded.json").read_text())
NIGHT = "2026-09-27"
NODEID = "tests/adversarial/model/test_adv_model_machine.py::test_random_operations"


def _finding(**changes: Any) -> Finding:
    fields: dict[str, Any] = {
        "layer": "model", "surface": "suite", "invariant": NODEID,
        "severity": "data-loss", "ops": ("Forget(user='u1')",), "commit": "c" * 40,
        "title": f"{NODEID} fails: the store and the model disagree",
        "detail": "E   AssertionError: the store and the model disagree"}
    fields.update(changes)
    return Finding(**fields)


def never(command: filing.Command) -> filing.Completed:
    raise AssertionError(f"a dry run must call nothing, and it ran {command.argv}")


class Recorded:
    """Stands in for gh or git: each command must start like the next expected one, and
    gets that one's recorded output."""

    def __init__(self, *script: tuple[tuple[str, ...], str], fingerprint: str = "") -> None:
        self.script = list(script)
        self.fingerprint = fingerprint
        self.calls: list[filing.Command] = []

    def __call__(self, command: filing.Command) -> filing.Completed:
        assert self.script, f"an unexpected command: {command.argv}"
        prefix, name = self.script.pop(0)
        assert command.argv[:len(prefix)] == prefix, command.argv
        self.calls.append(command)
        answer = {key: value.replace("FINGERPRINT", self.fingerprint)
                  if isinstance(value, str) else value
                  for key, value in RECORDED[name].items()}
        return filing.Completed(**answer)


def test_a_dry_run_sends_nothing_and_plans_one_labelled_issue_with_the_marker() -> None:
    """Filing is off until the operator turns it on. The plan must still be exactly what a
    real run would send: the label the dashboard filters on, and the marker a later night
    looks for."""
    finding = _finding()
    fp = finding.signature()
    filed = filing.file_issue(finding, fp, night=NIGHT, gh=never)
    assert (filed.what, filed.dry_run, filed.number, filed.url) == ("issue", True, None, None)
    listing, create = filed.commands
    assert listing.argv == ("gh", "issue", "list", "--repo", "memvara/memvara", "--label",
                            "nightly-break", "--state", "all", "--limit", "1000",
                            "--json", "number,state,url,body")
    assert create.argv == ("gh", "issue", "create", "--repo", "memvara/memvara", "--title",
                           filing.issue_title(finding), "--label", "nightly-break",
                           "--body-file", "-")
    assert create.stdin is not None
    assert f"<!-- memvara-nightly-fingerprint: {fp} -->" in create.stdin
    assert NODEID in create.stdin and "Forget(user='u1')" in create.stdin


def test_a_new_break_is_filed_as_one_issue_and_its_number_is_read_back() -> None:
    finding = _finding()
    fp = finding.signature()
    gh = Recorded((("gh", "issue", "list"), "issue_list_empty"),
                  (("gh", "issue", "create"), "issue_create"))
    filed = filing.file_issue(finding, fp, night=NIGHT, gh=gh, dry_run=False)
    assert (filed.dry_run, filed.existing, filed.number, filed.state) == (
        False, False, 302, "OPEN")
    assert filed.url == "https://github.com/memvara/memvara/issues/302"
    assert gh.script == []


def test_a_break_already_filed_is_found_by_its_marker_and_not_filed_again() -> None:
    """A break seen on two nights is one issue, even when the local history was lost: the
    marker in the issue's body is what says so. The issue here is closed, which tells the
    run the break came back after it was fixed."""
    finding = _finding()
    fp = finding.signature()
    gh = Recorded((("gh", "issue", "list"), "issue_list_known"), fingerprint=fp)
    filed = filing.file_issue(finding, fp, night=NIGHT, gh=gh, dry_run=False)
    assert (filed.existing, filed.number, filed.state) == (True, 301, "CLOSED")
    assert [command.argv[:3] for command in filed.commands] == [("gh", "issue", "list")]


@pytest.mark.parametrize("severity, named", [
    ("security", "private draft advisory"), ("unclassified", "SECURITY.md")])
def test_a_break_that_may_not_be_public_is_never_filed_as_an_issue(
        severity: str, named: str) -> None:
    """A security-class break in a public issue discloses it, and an unclassified break
    might be one. Both are refused before anything is sent, even with filing on."""
    finding = _finding(severity=severity)
    with pytest.raises(filing.FilingError, match=named):
        filing.file_issue(finding, finding.signature(), night=NIGHT, gh=never, dry_run=False)


def test_a_security_class_break_goes_to_a_private_draft_advisory() -> None:
    finding = _finding(severity="security")
    fp = finding.signature()
    gh = Recorded((("gh", "api", "repos/memvara/memvara/security-advisories?per_page=100"),
                   "advisory_list_empty"),
                  (("gh", "api", "--method", "POST",
                    "repos/memvara/memvara/security-advisories", "--input", "-"),
                   "advisory_create"))
    filed = filing.file_advisory(finding, fp, night=NIGHT, gh=gh, dry_run=False)
    assert (filed.what, filed.ghsa_id, filed.state) == (
        "advisory", "GHSA-7x3r-p5vq-m2jw", "draft")
    assert filed.url == "https://github.com/memvara/memvara/security/advisories/GHSA-7x3r-p5vq-m2jw"
    stdin = gh.calls[1].stdin
    assert stdin is not None
    payload = json.loads(stdin)
    assert f"<!-- memvara-nightly-fingerprint: {fp} -->" in payload["description"]
    assert payload["vulnerabilities"] == [{"package": {"ecosystem": "pip",
                                                       "name": "memvara"}}]


def test_an_advisory_already_drafted_is_found_by_its_marker() -> None:
    finding = _finding(severity="security")
    fp = finding.signature()
    gh = Recorded((("gh", "api"), "advisory_list_known"), fingerprint=fp)
    filed = filing.file_advisory(finding, fp, night=NIGHT, gh=gh, dry_run=False)
    assert (filed.existing, filed.ghsa_id, filed.state) == (
        True, "GHSA-2c4f-8h7m-q9vw", "draft")


def test_a_pull_request_pushes_only_to_the_nightly_branch_and_opens_as_a_draft(
        tmp_path: pathlib.Path) -> None:
    """The pin lands on a branch of its own, pushed by an explicit refspec, so nothing can
    reach main, and the pull request is a draft: nothing merges without review."""
    finding = _finding()
    fp = finding.signature()
    worktree = str(tmp_path)
    git = Recorded((("git", "-C", worktree, "status", "--porcelain"), "git_status_clean"),
                   (("git", "-C", worktree, "push"), "git_push"))
    gh = Recorded((("gh", "pr", "create"), "pr_create"))
    filed = filing.open_pr(finding, fp, issue=302, worktree=tmp_path, night=NIGHT, gh=gh,
                           git=git, dry_run=False)
    assert git.calls[1].argv == ("git", "-C", worktree, "push", "origin",
                                 f"HEAD:refs/heads/test/nightly-{fp[:12]}")
    create = gh.calls[0]
    assert create.argv == ("gh", "pr", "create", "--repo", "memvara/memvara", "--draft",
                           "--base", "main", "--head", f"test/nightly-{fp[:12]}",
                           "--title", filing.pr_title(finding, 302), "--body-file", "-")
    assert create.stdin is not None and "#302" in create.stdin
    assert (filed.what, filed.number, filed.url) == (
        "pr", 303, "https://github.com/memvara/memvara/pull/303")


def test_a_pull_request_is_refused_for_a_security_class_break_or_unsaved_work(
        tmp_path: pathlib.Path) -> None:
    """A security-class pin lands together with its fix, never alone in public. And a pin
    with uncommitted changes in its worktree would push something other than what was
    reviewed."""
    secret = _finding(severity="security")
    with pytest.raises(filing.FilingError, match="security"):
        filing.open_pr(secret, secret.signature(), issue=302, worktree=tmp_path,
                       night=NIGHT, gh=never, git=never, dry_run=False)
    finding = _finding()
    git = Recorded((("git", "-C", str(tmp_path), "status", "--porcelain"),
                    "git_status_dirty"))
    with pytest.raises(filing.FilingError, match="uncommitted"):
        filing.open_pr(finding, finding.signature(), issue=302, worktree=tmp_path,
                       night=NIGHT, gh=never, git=git, dry_run=False)


def test_the_operators_paths_never_reach_github() -> None:
    """A failure's text holds the paths of the machine that ran it, and a public issue must
    not carry the operator's name or folders."""
    detail = ("File /Users/alice/work/memvara/tests/adversarial/test_adv_x.py, line 3\n"
              "db = /private/var/folders/xy/T/pytest-of-alice/pytest-7/test_a0/memory.db\n"
              "home = /Users/alice/.memvara")
    finding = _finding(detail=detail)
    filed = filing.file_issue(finding, finding.signature(), night=NIGHT, gh=never,
                              paths={"/Users/alice/work/memvara": "<checkout>",
                                     "/Users/alice": "~"})
    body = filed.commands[1].stdin
    assert body is not None
    assert "alice" not in body
    assert "<checkout>/tests/adversarial/test_adv_x.py" in body
    assert "pytest-of-<user>/pytest-7" in body and "~/.memvara" in body


def test_an_error_from_gh_is_reported_with_its_own_message() -> None:
    """The label has to exist before the first filing; gh's own words say so."""
    finding = _finding()
    gh = Recorded((("gh", "issue", "list"), "issue_list_empty"),
                  (("gh", "issue", "create"), "issue_create_missing_label"))
    with pytest.raises(filing.FilingError, match="could not add label"):
        filing.file_issue(finding, finding.signature(), night=NIGHT, gh=gh, dry_run=False)


def _night_folder(checkout: pathlib.Path, *failures: regressions.Failure) -> None:
    folder = night.Layout(checkout).night(NIGHT)
    folder.mkdir(parents=True)
    night.write_json(folder / night.REPORT_JSON,
                     {"night": NIGHT, "failures": [each.to_record() for each in failures]})


def _confirmed(**changes: Any) -> regressions.Failure:
    return regressions.Failure(_finding(severity="unclassified", **changes), "test",
                               ("failed", "failed"), "confirmed")


def test_the_session_files_a_classified_break_from_the_nights_report(
        tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The scheduled session classifies a break against SECURITY.md and then files it by
    fingerprint. A dry run prints what it would send and records nothing."""
    failure = _confirmed()
    _night_folder(tmp_path, failure)
    code = filing.main(["issue", "--checkout", str(tmp_path), "--night", NIGHT,
                        "--fingerprint", failure.fingerprint, "--severity", "wrong-result"],
                       gh=never, git=never)
    printed = capsys.readouterr().out
    assert code == 0
    assert "dry run" in printed.lower() and "--label nightly-break" in printed
    assert not night.Layout(tmp_path).history.exists()


def test_filing_for_real_records_the_issue_so_the_next_night_knows(
        tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    failure = _confirmed()
    fp = failure.fingerprint
    _night_folder(tmp_path, failure)
    gh = Recorded((("gh", "issue", "list"), "issue_list_empty"),
                  (("gh", "issue", "create"), "issue_create"))
    assert filing.main(["issue", "--checkout", str(tmp_path), "--night", NIGHT,
                        "--fingerprint", fp, "--severity", "data-loss", "--file"],
                       gh=gh, git=never) == 0
    records, _ = night.read_jsonl(night.Layout(tmp_path).history)
    assert [(record["kind"], record["what"], record["fingerprint"], record["number"],
             record["severity"]) for record in records] == [
        ("filed", "issue", fp, 302, "data-loss")]
    assert "#302" in capsys.readouterr().out
    gh_again = Recorded()
    assert filing.main(["issue", "--checkout", str(tmp_path), "--night", NIGHT,
                        "--fingerprint", fp, "--severity", "data-loss", "--file"],
                       gh=gh_again, git=never) == 0
    assert gh_again.calls == []
    assert "already" in capsys.readouterr().out


@pytest.mark.parametrize("failure, named", [
    (regressions.Failure(_finding(severity="unclassified"), "test", ("passed", "passed"),
                         "flake"), "confirmed"),
    (regressions.Failure(_finding(severity="unclassified"), "xpass-strict", (),
                         "unconfirmed"), "confirmed"),
], ids=["a flake", "a strict expected failure that passed"])
def test_only_a_confirmed_break_of_that_night_can_be_filed(
        tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str],
        failure: regressions.Failure, named: str) -> None:
    _night_folder(tmp_path, failure)
    code = filing.main(["issue", "--checkout", str(tmp_path), "--night", NIGHT,
                        "--fingerprint", failure.fingerprint, "--severity", "crash"],
                       gh=never, git=never)
    assert code != 0
    assert named in capsys.readouterr().err
    assert filing.main(["issue", "--checkout", str(tmp_path), "--night", NIGHT,
                        "--fingerprint", "f" * 64, "--severity", "crash"],
                       gh=never, git=never) != 0


def test_a_pull_request_needs_the_issue_filed_first_when_filing_is_on(
        tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The strict expected failure cites its issue, so the issue must exist first."""
    failure = _confirmed()
    _night_folder(tmp_path, failure)
    code = filing.main(["pr", "--checkout", str(tmp_path), "--night", NIGHT,
                        "--fingerprint", failure.fingerprint, "--worktree", str(tmp_path),
                        "--file"], gh=never, git=never)
    assert code != 0
    assert "issue" in capsys.readouterr().err


def test_the_advisory_command_marks_the_break_as_security_class(
        tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    failure = _confirmed()
    _night_folder(tmp_path, failure)
    gh = Recorded((("gh", "api"), "advisory_list_empty"), (("gh", "api"), "advisory_create"))
    assert filing.main(["advisory", "--checkout", str(tmp_path), "--night", NIGHT,
                        "--fingerprint", failure.fingerprint, "--file"],
                       gh=gh, git=never) == 0
    records, _ = night.read_jsonl(night.Layout(tmp_path).history)
    assert [(record["what"], record["severity"], record["ghsa_id"]) for record in records] == [
        ("advisory", "security", "GHSA-7x3r-p5vq-m2jw")]
    assert dataclasses.replace(failure.finding, severity="security").signature() == (
        failure.fingerprint)
