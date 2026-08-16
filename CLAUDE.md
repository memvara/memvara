# Working in this repository

`CONTRIBUTING.md` has the setup, the gates and the scope rules, and it is the file to read
before writing code. This one covers the two things that are about *working here* rather
than about the code, both of which have cost real time.

## Files you need to keep but must not commit go in `local/`

`local/` at the root is ignored, whole. Put anything there that you will want again and
that must never reach a commit: a script you ran by hand, an API response captured as
evidence, a harness that reproduces the bug you are chasing, a report you are still
drafting. `git status` stays quiet and no `git add` can reach it.

Use it, because both of the obvious alternatives fail, in opposite directions:

- **A temporary directory is deleted without warning.** That is what temporary means. A
  sibling repository lost the only copy of a provisioning script exactly this way — it was
  written to a session scratchpad, described in a handoff note as "copy it somewhere
  durable first", and the directory was empty before anybody did.
- **An untracked file at the repository root is one that gets committed.** Not by you — by
  the next `git add -A` that runs in this checkout, under somebody else's message.

`local/` is outside the build as well as outside the commit: `pyproject.toml` builds the
sdist from what VCS does not ignore, and `testpaths` is `["tests", "memvara"]`, so nothing
there is collected, packaged or type-checked.

Two things do not belong in it. **Never a credential** — ignored is not encrypted, and
this repository is public, so the cost of a mistake here is disclosure rather than
cleanup. And **never the deliverable**: if the work is meant to ship it belongs in a
commit on a branch. `local/` is where a file goes to be kept, and also where it goes to be
forgotten.

## More than one agent may be working in this checkout at once

Assume files you did not touch are somebody else's unfinished work, and that they have no
way to know you exist.

1. **Commit files by name.** Never `git add -A`, `git add .`, or `git commit -a`. If you
   cannot list what you are committing, you do not know what you are committing.
2. **Never `git stash`, `git checkout <file>`, `git restore` or `git reset` a file you did
   not edit.** Each silently destroys uncommitted work, and `git checkout <file>` restores
   from HEAD rather than from your last edit — it has eaten an uncommitted rewrite here.
3. **Work on a branch and open a PR.** `main` is where sessions collide; a branch is yours.
4. **Before editing a file you did not create, run `git status`.** A file already modified
   is one somebody is in the middle of. If your change needs it, say what edit you need
   rather than making it.
5. **Never overwrite a document you did not write.** Append, or pick a distinct filename.
6. **Use a private `COVERAGE_FILE`.** Two concurrent runs clobber a shared `.coverage`,
   and the report that comes out of that is wrong in the direction that looks fine.
