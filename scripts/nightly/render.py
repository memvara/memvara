"""The night's report for a person: report.md, written from report.json."""

from __future__ import annotations

from typing import Any, Mapping

#: What each plan's need means, for the person or the session reading the report.
NEEDS = {
    "classification": "Check it against the In scope section of SECURITY.md, then file it "
                      "with one of these commands: the advisory for a security-class "
                      "break, or the issue and then the pull request for any other.",
    "filing": "It is classified already. Filing was a dry run; a run with --file sends:",
    "a strict-xfail test": "Its issue is filed. Write the strict-xfail test that cites the "
                           "issue, commit it by name, and open the draft pull request:",
    "nothing": "It is filed; nothing more is needed from the nightly run.",
    "reopening": "Its issue was closed, and the break is back. The issue has to be "
                 "reopened, which a run with --file does, and then the break needs a new "
                 "strict-xfail test and draft pull request:",
}

#: A known break's next need, as the end of one sentence.
NEXT = {
    "classification": "it still has to be classified against SECURITY.md, and then filed",
    "filing": "it is classified, and a run with --file will file it",
    "a strict-xfail test": "its issue is filed, and it waits for its strict-xfail test and "
                           "draft pull request",
    "nothing": "it is filed, and needs nothing more from the nightly run",
    "reopening": "its issue was closed and the break is back, so the issue must be reopened",
}

#: Why each kind of failure that is never filed is reported.
FOR_A_PERSON = {
    "intermittent": "It failed, then passed on one rerun and failed on the other. It counts "
                    "as a failure, but it is not filed, because a strict expected failure on "
                    "a test that sometimes passes would make the suite flaky.",
    "unconfirmed": "It failed, and its reruns could not confirm it: a rerun could not run "
                   "the test, or the rerun limit or time ran out.",
    "xpass-strict": "A test marked as a strict expected failure passed. Its bug may be "
                    "fixed: if so, the fix's pull request removes the marker.",
    "session": "pytest failed outside any test, so the end of its output says why.",
}


def markdown(report: Mapping[str, Any]) -> str:
    failures = list(report.get("failures", []))
    fresh = [entry for entry in failures if entry.get("novelty") in ("new", "recurred")]
    known = [entry for entry in failures if entry.get("novelty") == "known"]
    flakes = [entry for entry in failures if entry.get("verdict") == "flake"]
    person = [entry for entry in failures
              if entry.get("novelty") is None and entry.get("verdict") != "flake"]
    lines = [f"# Nightly run of {report['night']}", "", _lead(report, fresh, known, flakes,
                                                              person), ""]
    lines += ["## Steps", "", "| Step | Result | Time | Cap | Notes |", "|---|---|---|---|---|"]
    for step in report.get("steps", []):
        lines.append(f"| {step['name']} | {step['status']} | {_duration(step['seconds'])} | "
                     f"{_duration(step['cap'])} | {_cell(step.get('summary', ''))} |")
    lines.append("")
    if fresh:
        lines += ["## New breaks", ""]
        for entry in fresh:
            lines += _break(entry)
    if known:
        lines += ["## Known breaks", "",
                  "These broke on an earlier night too, and the history has them already.", ""]
        for entry in known:
            finding = entry["finding"]
            needs = entry["plan"]["needs"]
            lines.append(f"- {finding['title'] or finding['invariant']} (fingerprint "
                         f"`{entry['fingerprint'][:12]}`): {NEXT.get(needs, needs)}.")
        lines.append("")
    if person:
        lines += ["## Failures that need a person", "",
                  "None of these is filed by the nightly run.", ""]
        for entry in person:
            reason = FOR_A_PERSON.get(entry["kind"]) or FOR_A_PERSON.get(entry["verdict"], "")
            lines.append(f"- {entry['finding']['title'] or entry['finding']['invariant']}. "
                         f"{reason}")
        lines.append("")
    if flakes:
        lines += ["## Flakes", "", "These failed once and passed on both reruns, so nothing is "
                  "filed.", ""]
        lines += [f"- `{entry['finding']['invariant']}`" for entry in flakes]
        lines.append("")
    rates = report.get("flake_rates") or {}
    if rates:
        lines += ["## Flake rate per layer, last 14 nights", "",
                  "The budget is 0.5% of a layer's tests.", "",
                  "| Layer | Nights | Tests run | Flaky | Rate | Budget |", "|---|---|---|---|---|---|"]
        for layer, rate in rates.items():
            lines.append(f"| {layer} | {rate['nights']} | {rate['run']} | {rate['flaky']} | "
                         f"{rate['rate']:.2%} | {'over' if rate['over_budget'] else 'within'} |")
        lines.append("")
    lines += ["## Preflight", ""] + [f"- {note}" for note in report.get("preflight", [])]
    canary = report.get("canary") or {}
    lines += ["", "## Isolation canary", ""]
    if canary.get("changed"):
        lines.append(f"**{len(canary['changed'])} of the operator's protected files changed "
                     "during the night.** The suite must never touch them:")
        lines += [f"- `{path}`" for path in canary["changed"]]
    elif canary.get("files"):
        lines.append(f"The canary watched {canary['files']} of the operator's files, and none "
                     "of them changed.")
    else:
        lines.append("The canary was not taken, because the run stopped before it.")
    dependencies = report.get("dependencies") or {}
    lines += ["", "## Dependencies", ""]
    lines += ([f"- {name}: {state}" for name, state in dependencies.items()]
              or ["None was checked."])
    sent = report.get("notifications") or []
    lines += ["", "## Notifications", ""]
    lines += ([f"- {'Sent' if note['sent'] else 'Not sent'}: {note['message']}"
               for note in sent] or ["None was needed."])
    if report.get("warnings"):
        lines += ["", "## Warnings", ""] + [f"- {warning}" for warning in report["warnings"]]
    if report.get("error"):
        lines += ["", "## The error that stopped the run", "", "```", report["error"].rstrip(),
                  "```"]
    worktree = report.get("worktree")
    if not worktree:
        tested = "No worktree was created."
    elif report.get("given", {}).get("worktree"):
        tested = (f"The checkout that was tested, `{worktree}`, was given to the run, which "
                  "never removes it.")
    else:
        tested = (f"The worktree that was tested is `{worktree}`. It is removed by the next "
                  "night's run once it has no uncommitted changes.")
    lines += ["", "## Files", "",
              "Everything is in this folder: report.json holds this report for a program, "
              "findings.jsonl every break the night saw, and each step's folder its output. "
              + tested, ""]
    return "\n".join(lines)


def _lead(report: Mapping[str, Any], fresh: list[Any], known: list[Any], flakes: list[Any],
          person: list[Any]) -> str:
    if report.get("status") == "crashed":
        last = (report.get("error") or "").strip().splitlines()
        headline = ("**The run crashed before it finished**"
                    + (f": {last[-1]}" if last else "") + ". What it did before that is below.")
    elif report.get("status") == "running":
        headline = "**The run is still going, or was stopped partway.**"
    else:
        counts = [f"{len(fresh)} new {'break' if len(fresh) == 1 else 'breaks'}"] if fresh else []
        counts += [f"{len(known)} known"] if known else []
        counts += [f"{len(person)} for a person to look at"] if person else []
        counts += [f"{len(flakes)} {'flake' if len(flakes) == 1 else 'flakes'}"] if flakes else []
        # A step that did not pass found nothing because it did not finish its work, so a
        # night with one must never read as quiet.
        all_steps = report.get("steps", [])
        stalled = [f"{step['name']} {step['status']}" for step in all_steps
                   if step["status"] not in ("passed", "not built yet")]
        if stalled:
            counts.append(f"steps that did not pass: {', '.join(stalled)}")
        ran = sum(step["status"] not in ("not built yet", "not run") for step in all_steps)
        headline = (f"**{'; '.join(counts)}.**" if counts else
                    f"**Nothing broke in the {ran} {'step' if ran == 1 else 'steps'} that ran.**")
        # "Nothing broke" says nothing about the steps that do not exist yet, so the
        # headline counts them, rather than read as if the whole night checked everything.
        unbuilt = sum(step["status"] == "not built yet" for step in all_steps)
        if unbuilt:
            headline += (f" {unbuilt} {'step is' if unbuilt == 1 else 'steps are'} not built "
                         "yet.")
    commit = report.get("commit") or ""
    tested = (f" The run tested `{commit[:12]}`" if commit else " The run tested nothing")
    tested += (", a checkout it was given rather than a fresh worktree of origin/main."
               if report.get("given", {}).get("worktree") else ".")
    filing = (" Filing was on." if report.get("filing") == "file"
              else " Filing was a dry run, so nothing was sent to GitHub.")
    return headline + tested + filing


def _break(entry: Mapping[str, Any]) -> list[str]:
    finding = entry["finding"]
    novelty = ("It had been filed and closed, and it is back." if entry["novelty"] == "recurred"
               else "It is new tonight.")
    reruns = ", ".join(entry.get("reruns") or []) or "not rerun"
    lines = [f"### {finding['title'] or finding['invariant']}", "", novelty, "",
             f"- Fingerprint: `{entry['fingerprint']}`",
             f"- Layer: {finding['layer']}; surface: {finding['surface']}; severity: "
             f"{finding['severity']}",
             f"- Invariant: `{finding['invariant']}`",
             f"- Reruns: {reruns}"]
    if finding.get("seed"):
        lines.append(f"- Seed: `{finding['seed']}`")
    plan = entry.get("plan") or {}
    lines += ["", f"**Next:** {NEEDS.get(plan.get('needs', ''), plan.get('needs', ''))}"]
    if plan.get("note"):
        lines += ["", plan["note"]]
    if plan.get("error"):
        lines += ["", f"Filing failed: {plan['error']}"]
    if plan.get("commands"):
        lines += ["", "```"] + list(plan["commands"]) + ["```"]
    return lines + [""]


def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def _duration(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"
