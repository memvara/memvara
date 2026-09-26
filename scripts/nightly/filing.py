"""Filing a new confirmed break: an issue and a draft pull request, or a private advisory.

Filing is a dry run unless the operator passes --file. A dry run calls nothing at all: it
builds the exact commands a real run would send and prints them.

* **A public break** gets one issue, labelled `nightly-break`, whose body ends with a hidden
  marker holding the break's fingerprint. Before filing, the issues with that label are
  read and their markers compared, so a break that was filed before, even by hand or on
  a machine whose history was lost, is not filed again. Then a branch holding the
  strict-xfail test is pushed and a draft pull request is opened. Nothing is merged.
* **A security-class break**, one in scope for SECURITY.md, gets a private draft advisory
  whose description carries the marker. It never gets an issue, a branch or a pull
  request: its failing test lands together with its fix.
* **An unclassified break** is filed nowhere public. Somebody checks it against
  SECURITY.md's "In scope" section first.

The scheduled session files a break of the regressions step by hand, once it has
classified it (scripts/nightly/task.md):

    python3 scripts/nightly/filing.py issue --night DATE --fingerprint FP --severity S [--file]
    python3 scripts/nightly/filing.py advisory --night DATE --fingerprint FP [--file]
    python3 scripts/nightly/filing.py pr --night DATE --fingerprint FP --worktree W [--file]

With --file, each filing is recorded in local/nightly/history.jsonl, so the next night
knows the break is filed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

if __package__ in (None, ""):  # run as a script, not imported as nightly.filing
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from nightly import night, regressions  # noqa: E402 - needs the path set just above

from harness.report import Finding  # noqa: E402 - the nightly package puts tests/ on the path

REPO = "memvara/memvara"
LABEL = "nightly-break"
BRANCH_PREFIX = "test/nightly-"
#: Reads the fingerprint back out of an issue's body or an advisory's description.
MARKER = re.compile(r"<!-- memvara-nightly-fingerprint: ([0-9a-f]{64}) -->")
#: The severities the session may give a break it has found out of scope for SECURITY.md.
PUBLIC = ("data-loss", "wrong-result", "crash")
#: How much of a failure's text goes into an issue; GitHub caps a body at 65,536 characters.
DETAIL = 20_000


def marker(fingerprint: str) -> str:
    return f"<!-- memvara-nightly-fingerprint: {fingerprint} -->"


def branch(fingerprint: str) -> str:
    """The branch a break's strict-xfail test is pushed to. It is always under
    test/nightly-, so a push can never reach main."""
    return BRANCH_PREFIX + fingerprint[:12]


@dataclass(frozen=True)
class Command:
    """One command that is run, or would be run in a dry run."""

    argv: tuple[str, ...]
    #: Sent on standard input, such as an issue's body.
    stdin: str | None = None

    def shown(self) -> str:
        text = shlex.join(self.argv)
        return f"{text}  (with {len(self.stdin)} characters on stdin)" if self.stdin else text


@dataclass(frozen=True)
class Completed:
    returncode: int
    stdout: str = ""
    stderr: str = ""


#: Runs one command. The real one is `subprocess_runner`; tests replay recorded output.
Runner = Callable[[Command], Completed]


def subprocess_runner(command: Command) -> Completed:
    try:
        done = subprocess.run(list(command.argv), input=command.stdin, capture_output=True,
                              text=True, encoding="utf-8", timeout=120)
    except subprocess.TimeoutExpired:
        return Completed(124, "", f"{command.argv[0]} did not finish within 120 seconds")
    except FileNotFoundError:
        return Completed(127, "", f"{command.argv[0]} is not installed")
    return Completed(done.returncode, done.stdout, done.stderr)


class FilingError(Exception):
    """A filing was refused, or a command it ran failed."""


@dataclass
class Filed:
    """What one filing did, or would do in a dry run."""

    what: str  # "issue", "advisory" or "pr"
    fingerprint: str
    dry_run: bool
    commands: list[Command] = field(default_factory=list)
    number: int | None = None
    ghsa_id: str | None = None
    url: str | None = None
    state: str | None = None
    #: The break was filed already, found by its marker; nothing was sent.
    existing: bool = False

    def record(self, *, night: str, severity: str) -> dict[str, Any]:
        """The line kept in the history, so a later night knows the break is filed."""
        return {"kind": "filed", "date": night, "what": self.what,
                "fingerprint": self.fingerprint, "severity": severity,
                "number": self.number, "ghsa_id": self.ghsa_id, "url": self.url,
                "state": self.state, "existing": self.existing}


def scrub(text: str, paths: Mapping[str, str]) -> str:
    """`text` with each path in `paths` replaced by its placeholder, longest path first so
    a checkout inside the home folder keeps its own placeholder, and with the user's name
    in pytest's temporary folders replaced too. Nothing sent to GitHub may name the
    operator's machine."""
    for path, placeholder in sorted(paths.items(), key=lambda item: -len(item[0])):
        if path:
            text = text.replace(path, placeholder)
    return re.sub(r"pytest-of-[^/\\\s]+", "pytest-of-<user>", text)


def default_paths(checkout: pathlib.Path) -> dict[str, str]:
    """The operator's paths that `scrub` removes: the checkout, the home folder and the
    temporary folder, each also as its real path, since macOS reaches /var through
    /private/var."""
    paths: dict[str, str] = {}
    for path, placeholder in ((checkout, "<checkout>"), (pathlib.Path.home(), "~"),
                              (pathlib.Path(tempfile.gettempdir()), "<tmp>")):
        paths[str(path)] = placeholder
        paths[os.path.realpath(path)] = placeholder
    return paths


def issue_title(finding: Finding) -> str:
    title = f"Nightly break: {finding.title or finding.invariant}"
    return title if len(title) <= 250 else title[:247] + "..."


def _fence(text: str) -> str:
    """A code fence longer than any run of backticks in `text`, so the text cannot close it."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def _body(finding: Finding, fingerprint: str, *, night: str,
          paths: Mapping[str, str], closing: str) -> str:
    lines = [f"The nightly run found this break on {night}, testing commit "
             f"`{finding.commit or 'unknown'}`. It failed on both reruns, so it is not a "
             "flake.", "",
             f"- Layer: {finding.layer}", f"- Surface: {finding.surface}",
             f"- Invariant: `{finding.invariant}`", f"- Fingerprint: `{fingerprint}`", ""]
    if finding.surface == "suite" and "::" in finding.invariant:
        lines += ["To run the failing test:", "", "```",
                  f"python -m pytest -p no:cacheprovider --tier nightly '{finding.invariant}'",
                  "```", ""]
    if finding.ops:
        program = "\n".join(f"    {op}," for op in finding.ops)
        lines += ["The operations that replay it:", "", "```python",
                  f"drive.replay([\n{program}\n])", "```", ""]
    if finding.seed:
        lines += [f"The seed that reproduces the example: `{finding.seed}`", ""]
    if finding.detail:
        detail = scrub(finding.detail, paths)[-DETAIL:]
        fence = _fence(detail)
        lines += ["The failure, as the night printed it:", "", fence, detail, fence, ""]
    lines += [closing, "", marker(fingerprint), ""]
    return "\n".join(lines)


def issue_body(finding: Finding, fingerprint: str, *, night: str,
               paths: Mapping[str, str]) -> str:
    return _body(finding, fingerprint, night=night, paths=paths, closing=(
        "A draft pull request will add a strict expected failure that cites this issue. "
        "The fix follows in its own pull request, which removes the marker."))


def pr_title(finding: Finding, issue: int | None) -> str:
    return (f"Add a strict expected failure for the nightly break in #{issue}"
            if issue is not None else "Add a strict expected failure for a nightly break")


def pr_body(finding: Finding, fingerprint: str, *, issue: int | None, night: str) -> str:
    cites = f"#{issue}" if issue is not None else "the issue filed for this break"
    return "\n".join([
        f"This pull request adds a strict expected failure for {cites}, which the nightly "
        f"run found on {night} at commit `{finding.commit or 'unknown'}`. It does not fix "
        "the bug: the fix follows in its own pull request, which removes the marker.", "",
        "It stays a draft until the code review has run and a person has read it.", "",
        f"Fingerprint: `{fingerprint}`", "", marker(fingerprint), ""])


def _check(runner: Runner, command: Command) -> Completed:
    done = runner(command)
    if done.returncode != 0:
        said = (done.stderr or done.stdout).strip()
        raise FilingError(f"{command.shown()} failed with exit status {done.returncode}: "
                          f"{said}")
    return done


def _number(url: str) -> int:
    try:
        return int(url.rstrip("/").rsplit("/", 1)[1])
    except (IndexError, ValueError):
        raise FilingError(f"could not read a number from gh's answer {url!r}") from None


def _last_line(text: str) -> str:
    lines = text.strip().splitlines()
    if not lines:
        raise FilingError("gh printed nothing where it prints the new item's address")
    return lines[-1].strip()


def _public_only(finding: Finding) -> None:
    if finding.severity == "security":
        raise FilingError("this break is security-class: it goes to a private draft "
                          "advisory (filing.py advisory), never to a public issue, branch "
                          "or pull request")
    if finding.severity not in PUBLIC:
        raise FilingError("this break is unclassified: check it against the In scope "
                          "section of SECURITY.md first; the nightly run never files an "
                          "unclassified break in public")


def _listing(repo: str, label: str) -> Command:
    return Command(("gh", "issue", "list", "--repo", repo, "--label", label, "--state", "all",
                    "--limit", "1000", "--json", "number,state,url,body"))


def existing_issues(gh: Runner, *, repo: str = REPO,
                    label: str = LABEL) -> dict[str, dict[str, Any]]:
    """Every fingerprint marked in an issue with the label: its number, state and address."""
    found: dict[str, dict[str, Any]] = {}
    for issue in json.loads(_check(gh, _listing(repo, label)).stdout or "[]"):
        for fingerprint in MARKER.findall(issue.get("body") or ""):
            found[fingerprint] = {"number": issue["number"], "state": issue["state"],
                                  "url": issue["url"]}
    return found


def _advisories(repo: str) -> Command:
    return Command(("gh", "api", f"repos/{repo}/security-advisories?per_page=100"))


def existing_advisories(gh: Runner, *, repo: str = REPO) -> dict[str, dict[str, Any]]:
    """Every fingerprint marked in one of the repository's advisories, drafts included."""
    found: dict[str, dict[str, Any]] = {}
    for advisory in json.loads(_check(gh, _advisories(repo)).stdout or "[]"):
        for fingerprint in MARKER.findall(advisory.get("description") or ""):
            found[fingerprint] = {"ghsa_id": advisory["ghsa_id"], "state": advisory["state"],
                                  "url": advisory["html_url"]}
    return found


def file_issue(finding: Finding, fingerprint: str, *, night: str, gh: Runner,
               dry_run: bool = True, repo: str = REPO, label: str = LABEL,
               paths: Mapping[str, str] | None = None) -> Filed:
    """Open one issue for a public break, unless one with its marker exists already."""
    _public_only(finding)
    listing = _listing(repo, label)
    create = Command(("gh", "issue", "create", "--repo", repo, "--title",
                      issue_title(finding), "--label", label, "--body-file", "-"),
                     stdin=issue_body(finding, fingerprint, night=night, paths=paths or {}))
    if dry_run:
        return Filed("issue", fingerprint, True, [listing, create])
    known = existing_issues(gh, repo=repo, label=label).get(fingerprint)
    if known is not None:
        return Filed("issue", fingerprint, False, [listing], number=known["number"],
                     url=known["url"], state=known["state"], existing=True)
    url = _last_line(_check(gh, create).stdout)
    return Filed("issue", fingerprint, False, [listing, create], number=_number(url),
                 url=url, state="OPEN")


def file_advisory(finding: Finding, fingerprint: str, *, night: str, gh: Runner,
                  dry_run: bool = True, repo: str = REPO,
                  paths: Mapping[str, str] | None = None) -> Filed:
    """Draft one private security advisory, unless one with the marker exists already. The
    maintainer sets its severity and publishes it with the fix."""
    listing = _advisories(repo)
    payload = {
        "summary": issue_title(finding)[:1024],
        "description": _body(finding, fingerprint, night=night, paths=paths or {},
                             closing="The failing test lands together with the fix."),
        "vulnerabilities": [{"package": {"ecosystem": "pip", "name": "memvara"}}],
    }
    create = Command(("gh", "api", "--method", "POST", f"repos/{repo}/security-advisories",
                      "--input", "-"), stdin=json.dumps(payload))
    if dry_run:
        return Filed("advisory", fingerprint, True, [listing, create])
    known = existing_advisories(gh, repo=repo).get(fingerprint)
    if known is not None:
        return Filed("advisory", fingerprint, False, [listing], ghsa_id=known["ghsa_id"],
                     url=known["url"], state=known["state"], existing=True)
    answer = json.loads(_check(gh, create).stdout)
    return Filed("advisory", fingerprint, False, [listing, create],
                 ghsa_id=answer["ghsa_id"], url=answer["html_url"], state=answer["state"])


def open_pr(finding: Finding, fingerprint: str, *, issue: int | None,
            worktree: pathlib.Path, night: str, gh: Runner, git: Runner,
            dry_run: bool = True, repo: str = REPO) -> Filed:
    """Push the worktree's current commit to the break's nightly branch, and open a draft
    pull request for it. The worktree must hold only committed work: the pull request
    must contain what was reviewed and nothing else. A dry run calls nothing, git
    included, so it does not check the worktree either."""
    _public_only(finding)
    target = branch(fingerprint)
    push = Command(("git", "-C", str(worktree), "push", "origin", f"HEAD:refs/heads/{target}"))
    create = Command(("gh", "pr", "create", "--repo", repo, "--draft", "--base", "main",
                      "--head", target, "--title", pr_title(finding, issue),
                      "--body-file", "-"),
                     stdin=pr_body(finding, fingerprint, issue=issue, night=night))
    if dry_run:
        return Filed("pr", fingerprint, True, [push, create])
    status = _check(git, Command(("git", "-C", str(worktree), "status", "--porcelain")))
    if status.stdout.strip():
        raise FilingError(f"{worktree} has uncommitted changes; commit the strict-xfail "
                          "test by name, or remove what does not belong, first")
    _check(git, push)
    url = _last_line(_check(gh, create).stdout)
    return Filed("pr", fingerprint, False, [push, create], number=_number(url), url=url,
                 state="OPEN")


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        prog="filing.py", description="File a confirmed break from a night's report. A dry "
        "run, which prints what would be sent, unless --file is given.")
    sub = command.add_subparsers(dest="command", required=True)
    for name, help_text in (("issue", "a public issue, for a break outside SECURITY.md's "
                                      "scope"),
                            ("advisory", "a private draft advisory, for a security-class "
                                         "break"),
                            ("pr", "a draft pull request with the strict-xfail test, after "
                                   "the issue")):
        each = sub.add_parser(name, help=help_text)
        each.add_argument("--night", required=True, help="the night's date, YYYY-MM-DD")
        each.add_argument("--fingerprint", required=True, help="the break's fingerprint")
        each.add_argument("--checkout", help="the main checkout; the one this file is in by "
                          "default")
        each.add_argument("--repo", default=REPO, help=f"the repository; {REPO} by default")
        each.add_argument("--file", action="store_true",
                          help="send it to GitHub; without this, only print what would be "
                               "sent")
        if name == "issue":
            each.add_argument("--label", default=LABEL,
                              help=f"the issue's label; {LABEL} by default")
            each.add_argument("--severity", required=True, choices=PUBLIC,
                              help="the break's class, once you have checked it is outside "
                                   "SECURITY.md's scope")
        if name == "pr":
            each.add_argument("--worktree", required=True,
                              help="a worktree whose current commit holds the strict-xfail "
                                   "test")
    return command


def main(argv: Sequence[str] | None = None, *, gh: Runner = subprocess_runner,
         git: Runner = subprocess_runner) -> int:
    args = parser().parse_args(argv)
    checkout = (pathlib.Path(args.checkout) if args.checkout
                else night.main_checkout(pathlib.Path(__file__).parent))
    try:
        return _file(args, night.Layout(checkout), gh=gh, git=git)
    except FilingError as exc:
        print(f"Not filed: {exc}", file=sys.stderr)
        return 1


def _file(args: argparse.Namespace, layout: night.Layout, *, gh: Runner,
          git: Runner) -> int:
    failure = _confirmed(layout, args.night, args.fingerprint)
    history, _ = night.read_jsonl(layout.history)
    filed = {record["what"]: record for record in history
             if record.get("kind") == "filed" and record.get("fingerprint") == args.fingerprint}
    if args.command in filed:
        before = filed[args.command]
        print(f"This break's {args.command} was already filed: "
              f"{before.get('url') or before.get('number') or before.get('ghsa_id')}")
        return 0
    paths = default_paths(layout.checkout)
    dry_run = not args.file
    if args.command == "issue":
        severity = args.severity
        finding = dataclasses.replace(failure.finding, severity=severity)
        result = file_issue(finding, args.fingerprint, night=args.night, gh=gh,
                            dry_run=dry_run, repo=args.repo, label=args.label, paths=paths)
    elif args.command == "advisory":
        severity = "security"
        finding = dataclasses.replace(failure.finding, severity=severity)
        result = file_advisory(finding, args.fingerprint, night=args.night, gh=gh,
                               dry_run=dry_run, repo=args.repo, paths=paths)
    else:
        issue = filed.get("issue")
        if issue is None and not dry_run:
            raise FilingError("file the issue first (filing.py issue), so the strict "
                              "expected failure and the pull request can cite it")
        # Before the issue exists, a preview uses the break's own severity, so it refuses
        # what a real run would refuse: an unclassified or security-class break.
        severity = issue["severity"] if issue else failure.finding.severity
        finding = dataclasses.replace(failure.finding, severity=severity)
        result = open_pr(finding, args.fingerprint, issue=issue["number"] if issue else None,
                         worktree=pathlib.Path(args.worktree), night=args.night, gh=gh,
                         git=git, dry_run=dry_run, repo=args.repo)
    _say(result)
    if not result.dry_run:
        night.append_jsonl(layout.history, result.record(night=args.night, severity=severity))
    return 0


def _confirmed(layout: night.Layout, date: str, fingerprint: str) -> regressions.Failure:
    report = layout.night(date) / night.REPORT_JSON
    if not report.exists():
        raise FilingError(f"there is no report for the night of {date} at {report}")
    for record in night.read_json(report).get("failures", []):
        if record.get("fingerprint") == fingerprint:
            failure = regressions.Failure.from_record(record)
            if not failure.fileable:
                raise FilingError(f"this failure is a {failure.kind} failure with the "
                                  f"verdict {failure.verdict}; only a confirmed break is "
                                  "filed")
            return failure
    raise FilingError(f"the night of {date} has no failure with the fingerprint "
                      f"{fingerprint}")


def _say(result: Filed) -> None:
    if result.dry_run:
        print(f"Dry run: nothing was sent to GitHub. Filing this {result.what} would run:")
        for command in result.commands:
            print(f"  {command.shown()}")
        return
    if result.existing:
        print(f"Already filed, found by its marker: {result.url} ({result.state})")
        return
    ident = f"#{result.number}" if result.number is not None else result.ghsa_id
    print(f"Filed {result.what} {ident}: {result.url}")


if __name__ == "__main__":
    sys.exit(main())
