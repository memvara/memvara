# Working in this repository

`CONTRIBUTING.md` has the setup, the gates and the scope rules, and it is the file to read
before writing code. This one covers the three things that are about *working here* rather
than about the code, all of which have cost real time.

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

## Documentation ships in the same commit as the code

When you change behaviour, update everything that describes it in the same commit. Not in
a follow-up PR, not in a note for later, not in a "docs to update" list at the end of a
handoff. A deferred documentation change is not a smaller version of the work — it is a
different piece of work, one that nobody has been assigned and that no test will fail
over.

The failure is quiet and it compounds. Documentation that is wrong is worse than
documentation that is missing: missing docs send a reader to the code, wrong docs send
them somewhere confidently and let them act on it. The person who pays is never the
author — it is whoever reads it next, without the context that would let them notice.

"Everything that describes it" is more places than anyone remembers, so work the real
inventory out by looking rather than by trusting this list. It runs to at least
`README.md`, `CHANGELOG.md` (every user-visible change), `docs/UPGRADING.md` when
behaviour changes under someone, `docs/INTERNALS.md`, and the packaged skill at
`memvara/skills/claude/SKILL.md`, which states outright that it does not repeat what a
tool description says — so text moving between the two has to move in both.
`CONTRIBUTING.md` states the same duty from the other end, and names the documents that
make specific, checkable claims.

Two more places are easy to miss, and both are particular to this repository.

1. **The tool descriptions in `memvara/server/tools.py` are documentation that a model
   reads at runtime**, not prose for a human. Wrong text there does not confuse a reader
   who can go and check — it misleads an agent that cannot. Read how the existing ones
   are written before adding to them. The precedent worth studying is the distinction
   between *ending* a fact and *retiring* it: `"ended"` says the world changed,
   `"retired"` says the record was wrong, and the module docstring calls it the one
   mistake here that cannot be found by reading the data afterwards. It has already gone
   wrong once — a write receipt reported `retired 1` for a fact that had merely stopped
   being true, leaving a model reading its own memory tool with three names for two
   events. Imprecise wording there does not read badly. It makes an agent record a false
   reason for a change that nothing downstream can detect.
2. **Docstrings run as tests.** `pyproject.toml` sets `testpaths = ["tests", "memvara"]`
   with `--doctest-modules`, so a stale example in a docstring fails the suite rather
   than rotting quietly. That is the one corner of documentation here that defends
   itself. Everywhere else, nothing will catch you.

If a change genuinely cannot carry its documentation — the doc lives in another
repository, or the decision is not final — say so in the PR body, name the file, and open
an issue. That is a deferral somebody can see. A silent one is a defect with a delay on
it.

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
