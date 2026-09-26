# The memvara nightly run

This is the prompt for the scheduled session that runs the adversarial test suite's
nightly run, once a night, on the maintainer's Mac. Follow it in order. The testing guide,
the adversarial suite's page in the repository's context documentation, describes
everything named here in its section "The nightly run".

## Settings

The maintainer fills these in when installing the scheduled task.

- **Checkout:** `__CHECKOUT__`, the main checkout of the memvara repository on this Mac.
  Run every command below from this folder.
- **Filing:** `dry-run`. Only the maintainer changes this, to `file`, and only after the
  supervised dry run the design asks for. While it says `dry-run`, never pass `--file` to
  any command, and nothing is sent to GitHub.

## 1. Run the night

```bash
python3 scripts/nightly/run.py
```

Add `--file` only when Filing above says `file`. Every step has its own time cap. With
the steps that are built today, a night takes about half an hour; with every step built,
the caps add up to three hours and forty minutes. Do not stop the run, and do not run
anything else in the checkout while it runs. When it ends, it prints where it wrote the
report. It exits with status 0 when the night finished and 1 when the run itself
crashed.

## 2. Read the report

Read `local/nightly/<date>/report.md`, where `<date>` is the night's date the run printed.
Its first line says what happened.

- If the run crashed, read the error at the end of the report, write what you found into
  `local/nightly/<date>/triage.md`, and stop. The watchdog reports the night as well.
- If the report has no new, recurred or known break whose next step is anything but
  "nothing", stop. The run has already sent every notification the night needed.

Failures under "Failures that need a person" and the flakes are for the maintainer to read.
They are never filed, so leave them alone.

## 3. Classify each break that needs it

A failed test is unclassified until somebody checks it, and the nightly run never files an
unclassified break in public. For each break whose next step is a classification:

1. Read its failure in the report, and the test that failed.
2. Read the "In scope" section of `SECURITY.md`. The break is **security-class** if it
   matches anything there: scope isolation (a reader sees what its scope should not),
   erasure completeness (text is still recoverable after `erase` or `purge` reported
   success), provenance and the audit trail, the prompt-injection surface in `recall()`,
   the redaction seam, the MCP server's bound scope, or injection into the store.
3. When you are not sure, treat it as security-class. A private advisory can be made
   public later; a public issue cannot be made private again.
4. Otherwise give it one of these severities: `data-loss` (something the store
   acknowledged was lost, or left half written), `wrong-result` (a read returned the
   wrong answer), or `crash` (a process died, hung or refused to start).
5. Write the decision and the reason, one short paragraph per break, into
   `local/nightly/<date>/triage.md`. That file stays on this machine.

## 4. File each classified break

`<fingerprint>` is the break's fingerprint in the report. While Filing says `dry-run`, run
each command below as written: it prints what it would send and sends nothing. Record in
`triage.md` what would have been filed, and go no further with that break: there is no
issue number for a strict expected failure to cite.

**A security-class break** goes to a private draft advisory, and nowhere else:

```bash
python3 scripts/nightly/filing.py advisory --night <date> --fingerprint <fingerprint>
```

Write nothing about it anywhere else: no issue, no branch, no commit, no pull request, no
comment, and nothing in any file that is committed. Its failing test lands together with
its fix, and the maintainer handles that.

**Any other break** gets one issue, labelled `nightly-break`:

```bash
python3 scripts/nightly/filing.py issue --night <date> --fingerprint <fingerprint> --severity <data-loss|wrong-result|crash>
```

With Filing on, it prints the new issue's number. Then pin the break with a strict expected
failure that cites the issue:

1. Add a worktree at the commit the night tested, which is `commit` in `report.json`:
   `git worktree add --detach local/nightly/<date>/pin-<first 12 characters of the fingerprint> <commit>`.
2. In it, append one `KnownBug` to `tests/harness/known_bugs.py`, with the next free id,
   the issue's number and a one-line title. Append only; never change an existing entry.
3. Mark the failing test with `@known_bugs.xfail("<id>")`, and make the test raise
   `known_bugs.Reproduced` only after it has seen this bug's own symptom, as the testing
   guide's section on known bugs describes. A marker that absorbs any failure would hide
   the next bug behind this one. When the failing test is the reference model's state
   machine, write a new test under `tests/adversarial/regressions/` instead, which runs
   the replay program from the report with `drive.replay` and raises `Reproduced` on the
   divergence the report shows.
4. Run the test with `--runxfail` and check that it fails for the reason in the report.
   Then run it without, and check that pytest reports it as an expected failure.
5. Commit the files by name, never with `git add -A`, `git add .` or `git commit -a`. The
   message says what the commit adds, in a plain sentence.
6. Push the commit and open the draft pull request:

```bash
python3 scripts/nightly/filing.py pr --night <date> --fingerprint <fingerprint> --worktree <worktree>
```

**A break that came back after its issue was closed** is marked recurred in the report. With
Filing on, the run has already reopened its issue, with a comment that says why, and the
break's next step is a new strict-xfail test. Pin it as in steps 1 to 6 above, citing the
reopened issue. The fix that closed the issue removed the old pin and its `KnownBug`
entry, so append a new entry. When the recurred break is security-class, its private
advisory is closed instead: reopen the advisory on GitHub by hand, and write nothing else
about it.

## 5. Review the pull request

Every pull request gets the code review the repository's rules require before anything
merges, as the root rules file's section on code review describes. Run it on the draft
pull request. Check each finding against the code before acting on it, fix the real ones
on the same branch in the pin's worktree, and push the fixes with
`git push origin HEAD:refs/heads/test/nightly-<first 12 characters of the fingerprint>`.
Then add a short section to the pull request's body that says the review ran, at what
effort, and what it found. Leave the pull request as a draft.

## Rules that always hold

- Nothing merges. Never merge a pull request, never push to `main`, and never force-push.
- No text that reaches GitHub names any assistant or model, or carries a co-author
  trailer or a generated-with line: not an issue, a pull request, a review, a comment or a
  commit message.
- Send no notification or message of your own. The run and the watchdog send the ones the
  design allows: a new break, an isolation breach, a dependency down two nights running,
  and a night that did not run.
- Never create, change or delete a scheduled task, a launchd job, a cron entry or any other
  standing configuration, and install nothing outside the night's worktrees.
- If a step here cannot be followed, stop, and write what happened into
  `local/nightly/<date>/triage.md`.
