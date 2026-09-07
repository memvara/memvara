# Working in a memvara plugin repository

<!-- Canonical. This file is `plugin-claude.md` in memvara/memvara and is copied into every
     repository in `plugin-repos.txt` as their `CLAUDE.md`. Edit it HERE; a sync overwrites
     the copy. What sits between the `local: begin` and `local: end` markers below belongs to
     the repository it lands in and survives a sync: its runtime facts and its hook rules. -->

These repositories are thin. Each is an install surface — a manifest, a vendored skill, some
tests — wrapped around a library that lives somewhere else. Almost every mistake made here
comes from forgetting that, so this file is about the habits that follow rather than about
the code. The memvara/memvara repository is the core; this one packages it.

## Read the core repository before proposing anything to it

Not skim: read. Three places in the core hold the design decisions, and all three are
load-bearing. **`docs/INTERNALS.md`** states the invariants and why each is the way it is.
**`docs/ROADMAP.md`** has *Deliberately deferred* and *What is still missing*, which exist so
that considered-and-declined stops reading as not-yet-done; if your proposal is in either,
the question is settled and the burden is on new evidence. **The tests are the design
document**: `tests/test_server.py` and `tests/test_pipeline.py` reason in docstrings that run
to paragraphs, and test names alone tell you whether a behaviour is deliberate.

This has a measured cost. A predicate-router design written in one of these repositories was
cut by three quarters on a second pass, because reading the core would have shown that the
mechanism already existed as a `registry` parameter on the constructor, that the many-values
default was deliberate and documented, that the contradiction report already shipped as
`types.Accumulation` plus `_receipt_summary`, and that the inference the plan rested on had
been rejected in a test: two live values in one slot can be a contradiction
(`quota_gate/status`) or perfectly correct (`agent-memory/rejected`), the rows are
identical, and the difference is intent, which is not a property of the row. The checklist
at the end of this file is what would have caught it before a word was written.

## The skill is vendored. Do not edit it here.

The source of truth is `memvara/skills/memvara/` in the core repository. The skill.lock file
pins the commit, CI diffs the vendored copy against that commit, and every plugin repository
pins the same sha. Edit the copy here and the sync overwrites you, after CI has already
failed. Fix the skill upstream and let the sync bring it across.

There is exactly one sanctioned local transform, in claude-memvara: the front-matter
`name: memvara` becomes `name: memory`, so the client renders the command as
`/memvara:memory`. The drift test compensates for that one line and no other; every remaining
byte still has to match.

## So is `plugin/hooks/`, and that one has no transform at all

Same relationship, second tree: the source of truth is `plugin/hooks/` in the core
repository, hooks.lock pins the commit, the hooks-sync workflow copies it across and CI diffs
it. Do not edit it here either. Three things differ from the skill.

**There is no sanctioned transform.** Not one line. The canonical path and the vendored path
are the same string, so the sync is a plain copy and the gate a plain byte comparison.

**The hooks manifest is generated rather than vendored.** Every repository registers a
different client, so a canonical copy would be one repository's manifest shipped to all of
them. Build it by running `plugin/hooks/tools/generate.py` with the host that hooks.lock
names; edit the host file under `plugin/hooks/hosts/` and regenerate. A hand edit to the
manifest fails the gate.

**The `host=` line in hooks.lock is yours.** It says which record this repository registers,
the sync reads it back out of the file it is replacing, and nothing upstream may set it. A
literal host in the sync workflow would make every sibling install surface a copy of one.

**Documentation ships in the same commit as the code.** Inherited from the core repository's
own `CLAUDE.md`, and it means the README here too: a README that oversells the install is how
someone finds a background process they were told would not exist.

@@LOCAL@@
## Guards, and how they fail quietly

Almost every defect found here on 2026-08-25 was the same shape and none raised: a claim and
the guard that checks it, frozen together, agreeing with each other while both were wrong,
reported honestly to a channel nobody reads. Four in one day. The skill lock and the vendored
copy agreed for five commits while the library moved, because the drift test,
`test_matches_library_at_lock_sha`, compared the copy against the sha the copy itself named.
The memvara-web tool count and its `test/tool-count.test.ts` agreed while the site said ten
and the endpoint served twelve, with `memory_neighborhood` and `memory_paths` never counted
at all. The skill sync workflow failed nightly for four days, in a scheduled run's log. The
drift check printed `drift NOT checked: HTTP Error 403` instead of checking and the job went
green. All four were unheard, which is harder to notice than silent, because the
honesty makes it look handled. Eight rules follow.

- **A guard compares a claim against its referent, never against a copy of itself.** The
  referent is the server, the library's default branch, the endpoint. A test that reads the
  value out of the same repository that states it proves the file is self-consistent and
  nothing else. Where reading the referent is genuinely wrong, say why in the guard:
  memvara-web does not reach into the core, because that test fails on a stale checkout.
- **State it positively: the correct value must be present.** A guard spelled "the page does
  not state the wrong count" passes on a page that has stopped stating anything at all, so a
  guard a deletion satisfies has quietly stopped guarding. (Stated without quoting a wrong
  count, deliberately: `test_no_other_count_is_stated_anywhere` scans every markdown file in
  this repository and cannot tell an illustrative count from a claim, and it caught this
  section as it was being written.)
- **Prove the guard can fail before believing it passes.** Break the thing it watches and
  watch it go red. Every guard added that day was sabotaged first, and three were found broken
  by that step alone: the drift check skipped on CI, the only place it runs, because the
  pinned library checkout could not resolve the remote default branch; its skip path fired on
  `CERTIFICATE_VERIFY_FAILED`, so on any Mac it reported the library unreachable while the
  library was fine; and a test suite for the sources probe stubbed the method under test,
  so deleting the probe left every test green. A passing run does not distinguish "the code
  works" from "the check never ran".
- **A hand-maintained list of what is covered is itself unguarded.** One page in
  memvara-web, `AgentSetup.tsx`, stated the tool count three times and was absent from the
  guard's `PROSE` list, so it was free to say any number. Removing a file from that list produced fifteen
  passing tests and no failure. Check the list against the tree.
- **A skip is not a pass, and neither is a truncated tail.** "OK (skipped=1)" is not "OK".
  Read the verdict line and all of it. Piping through `tail -3 | head -2` swallowed a failure
  twice in one day, and once nearly shipped six red pull requests on a "Ran 15 tests" line
  with the result cut off.
- **Measure twice before writing a number down.** A single reading of the command-line
  preamble said 67k tokens and did not reproduce across four later runs. One observed CI skip
  became "all six repos are inert", which the data contradicted: 23 of 23 runs had the check
  running.
- **Read shared state from the tool, not from a checkout.** Several sessions work these
  repositories at once, and a sibling checkout six commits behind would produce a sync that
  pinned the new sha while shipping the old bytes. The git log of the remote default branch,
  and `gh pr list`, cost one call each.
- **Verify the deliverable, not the repository.** Merged is not shipped. Twenty-one commits
  sat on the default branch behind an unchanged version string while the plugin update command
  said "already at the latest version"; only opening a session and reading the status line
  would have caught it.

## A PR you opened gets a code review before it is merged

Open the pull request, then review it with `/code-review high <PR number>`, then fix what the
review found. In that order, and all of it before anybody merges. Review a tree you have not
pushed and you have reviewed something no reviewer will see; skip the review and the pull
request merges unreviewed, which is the case this rule exists for.

Run it on the latest Sonnet, `claude-sonnet-5` today. The command takes an effort level, a
target, and `--comment` or `--fix`, but no model argument, so it runs on whatever the session
model is. Switch the model before the review and back afterwards, and where you cannot switch
say which model reviewed in the pull request body. Use `high`, not `ultra`, which is
user-triggered and billed and which an agent cannot launch; reach for `max` on a large or
load-bearing change.

Fix everything it finds, on the same branch, then re-run the gate. The `--fix` flag applies
findings to the working tree, so the commit and the push are still yours to make. Where a
finding is wrong, write the reason in the pull request body: a disagreement recorded is a
decision, and a finding dropped in silence is a defect with a delay on it. Nothing the review
publishes may carry an AI attribution. The `--comment` flag posts under
the account running it, and the marketplace code-review plugin — present in the user's plugin
marketplaces directory and deliberately not enabled — ends every comment with a "Generated
with Claude Code" line. That rule is absolute and lives in the user's global `CLAUDE.md`.
Prefer `--fix` and a summary in your own words; if you do post, read it first.

## Before proposing new machinery

1. Grep the constructor or signature for the parameter you want to add.
2. Read the test names for the behaviour you want to change.
3. Check `docs/ROADMAP.md` — *Deliberately deferred*, then *What is still missing*.
4. Check `docs/INTERNALS.md` for the invariant you are about to cross.
5. Then write the plan, and say which of the four you checked.

---

# Karpathy guidelines

Behavioural guidelines for reducing common LLM coding mistakes, from
[multica-ai/andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills)
(declared MIT in the skill's frontmatter), derived from
[Andrej Karpathy's observations](https://x.com/karpathy/status/2015883857489522876).
They are merged here rather than vendored as a second skill: they govern how work is done
*in* this repository, and shipping them inside the plugin would hand every memvara user a
third-party skill they did not install.

**Tradeoff:** these guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think before coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity first

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:

- Remove imports, variables and functions that *your* changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: every changed line should trace directly to the user's request.

## 4. Goal-driven execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

- "Add validation" → "write tests for invalid inputs, then make them pass"
- "Fix the bug" → "write a test that reproduces it, then make it pass"
- "Refactor X" → "ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require
constant clarification.

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due
to overcomplication, and clarifying questions arriving before implementation rather than
after mistakes.

## Where they bite hardest in this project

Not decoration — each of these has already cost time here.

- **§1 and §2 against the core repository.** The predicate-router episode is the worked
  example above: a design was written before the core was read, and the second pass cut it
  by three quarters because the mechanism already existed and the inference it rested on had
  been explicitly rejected upstream. "Think before coding" here means *read `docs/INTERNALS.md`,
  the roadmap's deferred list, and the test names* — not merely pause.
- **§3 against a vendored tree.** `plugin/skills/` is not yours to improve. Style, wording
  and formatting there are upstream's; the only sanctioned local edit is the one line
  skill.lock and the drift test know about.
- **§4 against silent failures.** Most defects in this repository do not raise. "Verify"
  therefore has to mean comparing output — bytes, counts, a diff against a known-good run —
  never that a command exited 0 or ran fast. A hook that returns nothing is the fastest hook
  there is.

One local amendment to §3, because this repository's own rule is stricter, not looser:
**documentation ships in the same commit as the code.** Updating the README, `CHANGELOG.md`
or a tool description alongside a behaviour change *is* the surgical change, not scope creep.
