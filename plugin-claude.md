# Working in a memvara plugin repository

<!-- Canonical. This file is `plugin-claude.md` in memvara/memvara and is copied into every
     repository in `plugin-repos.txt` as their `CLAUDE.md`. Edit it HERE; a sync overwrites
     the copy. Everything between the `local: begin` and `local: end` markers below belongs
     to the repository it lands in and is preserved across syncs -- that is where a repo's
     own runtime facts and its hook rules live, because only one plugin ships hooks. -->

These repos are thin. Each one is an install surface — a manifest, a vendored skill, some
tests — wrapped around a library that lives somewhere else. Almost every mistake made here
comes from forgetting that, so this file is about the habits that follow from it rather
than about the code.

`memvara/memvara` is the core. This repo packages it.

## Read the core repository before proposing anything to it

Not skim: read. The design decisions are written down, at length, in three places, and all
three are load-bearing:

- **`docs/INTERNALS.md`** states the invariants and *why* each one is the way it is.
- **`docs/ROADMAP.md`** has a section called **Deliberately deferred** and another called
  **What is still missing**. They exist so that considered-and-declined stops reading as
  not-yet-done. If your proposal is in either, the question is settled and the burden is on
  new evidence.
- **The tests are the design document.** `tests/test_server.py` and `tests/test_pipeline.py`
  explain reasoning in docstrings that runs to paragraphs. Test *names* alone will tell you
  whether a behaviour is deliberate.

This has a measured cost. A predicate-router design was written in this repo and then cut
by three quarters on a second pass, because reading the core would have shown that:

- the mechanism already existed — `Memvara(...)` has taken a `registry` parameter all along;
- the MANY default was already deliberate and already documented in `INTERNALS.md`
  ("Wrongly retiring a true fact is worse than keeping two competing ones");
- the contradiction report already shipped, as `types.Accumulation` plus `_receipt_summary`;
- and the inference the plan was built on had been **explicitly rejected** in a test, with a
  better argument than the plan had: two live values in one slot can be a contradiction
  (`quota_gate/status`) or perfectly correct (`agent-memory/rejected`), the rows are
  identical, "the difference is intent and intent is not a property of the row".

None of that needed new machinery. It needed twenty lines of server plumbing. Two checks
would have caught it before a word was written:

1. **grep the constructor** for the parameter you are about to propose adding;
2. **read the test names** for the behaviour you are about to propose changing.

## The skill is vendored. Do not edit it here.

The source of truth is `memvara/skills/memvara/` in `memvara/memvara`. `skill.lock` pins the
commit, CI diffs the vendored copy against that commit, and every plugin repo pins the same
sha. Edit the copy here and two things happen: sync overwrites you, and CI fails first.

Fix the skill upstream, then let sync bring it across.

There is exactly one sanctioned local transform, in `claude-memvara`: the frontmatter
`name: memvara` becomes `name: memory`, so the client renders `/memvara:memory` rather than
`/memvara:memvara`. It is applied during sync, and the drift test compensates for that one
line and no other — every remaining byte still has to match.

**Documentation ships in the same commit as the code.** Inherited from the core repo's own
CLAUDE.md, and it means the README here too: a README that oversells the install is how
someone finds a background process they were told would not exist.

@@LOCAL@@
## Guards, and how they fail quietly

Almost every defect found here on 2026-08-25 was one shape, and none of them raised. A
claim and the guard that checks it, **frozen together, agreeing with each other while both
were wrong** — and reporting it honestly to a channel nobody reads. Four in a day:

- `skill.lock` and the vendored copy stayed consistent *with each other* for five commits
  while the library moved. `test_matches_library_at_lock_sha` compares the copy against the
  sha the copy itself names, so the pair agreed forever.
- `memvara-web`'s tool count and `test/tool-count.test.ts`: the test pinned **the site's own
  claim**, green while the site said ten and the endpoint served twelve, with
  `memory_neighborhood` and `memory_paths` never counted at all.
- `skill-sync.yml` failed every night for four days. The failure was in a scheduled run's
  log.
- The drift check printed `drift NOT checked: HTTP Error 403` and the job went green.

None was silent. All four were unheard, which in practice is the same thing and is harder
to notice, because the honesty makes it look handled.

### A guard compares a claim against its referent, never against a copy of itself

The referent is the server, the library's default branch, the endpoint — the thing the
claim is *about*. A test that reads the value out of the same repository that states it
proves the file is self-consistent and nothing else.

Where reading the referent is genuinely wrong, say why in the guard. `memvara-web`
deliberately does not reach into the core, because a test that reads a sibling working tree
fails on a stale checkout — and the cost of that choice is a comment, not silence.

### State it positively: the correct value must be PRESENT

A guard spelled "the page does not state the *wrong* count" passes on a page that has
stopped stating anything at all — a
rewritten sentence, a deleted paragraph, a digit instead of a word. **A guard a deletion
satisfies has quietly stopped guarding.** Requiring the right phrase means a page that no
longer tells the reader the truth fails exactly as loudly as one that tells them something
false.

(That rule is stated without quoting a wrong count, deliberately: `test_no_other_count_is_stated_anywhere`
scans every markdown file in this repository, and an illustrative "N tools" in prose is
indistinguishable from a claim. It caught this very section while it was being written,
which is the guard behaving exactly as intended.)

### Prove the guard can fail, before believing it passes

Break the thing it watches and watch it go red. Every guard added that day was sabotaged
first, and three were found broken *by that step alone*:

- the drift check **skipped on CI** — the only place it runs — because the library checkout
  is pinned to `skill.lock`'s sha and could not resolve `origin/main`;
- its skip path was firing on `CERTIFICATE_VERIFY_FAILED`, so on any Mac it reported the
  library unreachable while the library was fine;
- a test suite for the `sources` probe **stubbed the method under test**, so deleting the
  probe entirely left every test green.

A passing run does not distinguish "the code works" from "the check never ran". Only a
failing run does.

### A hand-maintained list of what is covered is itself unguarded

`AgentSetup.tsx` stated the tool count three times and was absent from the guard's `PROSE`
list, so it was free to say any number. Removing it from that list produced *fifteen
passing tests and no failure* — the guard did not weaken, it stopped covering a file, and
from outside those look identical. Check the list against the tree.

### A skip is not a pass, and neither is a truncated tail

`OK (skipped=1)` is not `OK`. Read the verdict line, and read all of it: `tail -3 | head -2`
swallowed a `FAILED` twice in one day, and once nearly shipped six red PRs on the strength
of a `Ran 15 tests` line with the result cut off.

### Measure twice before writing a number down

A single reading of the `claude -p` preamble said 67k and did not reproduce across four
later runs — writing it down would have replaced one stale number with a worse one. One
observed CI skip became "all six repos are inert", which the data flatly contradicted:
23 of 23 runs had the check running.

### Read shared state from the tool, not from a checkout

Several sessions work these repos at once. A sibling checkout six commits behind would have
produced a sync that pinned the new sha while shipping the old bytes — the lock and the copy
agreeing, again. `git log origin/main -- <path>` and `gh pr list` cost one call and answer
what someone told you.

### Verify the deliverable, not the repository

Merged is not shipped. Twenty-one commits sat on `main` behind an unchanged version string
while `/plugin update` answered "already at the latest version", and the only check that
would have caught it was opening a session and reading the status line. Whatever the change
is *for* is the thing to look at.

## A PR you opened gets a code review before it is merged

Open the pull request, then review it, then fix what the review found. In that order, and
all of it before anybody merges.

```bash
/code-review high <PR number>
```

The window is narrow at both ends. Run it against a working tree you have not pushed and
you have reviewed something no reviewer will ever see. Skip it and the PR merges
unreviewed, which is the case this rule exists for — nothing else in the process looks at
the change with fresh eyes.

**Run it on the latest Sonnet, `claude-sonnet-5` today.** `/code-review` takes an effort
level, a target, and `--comment` / `--fix`. It takes **no model argument**, so the review
runs on whatever the session model is: switch it (the app's model picker, or `/model` in a
terminal session) before the review and back afterwards. In a session where you cannot
switch, say which model reviewed in the PR body rather than letting a reader assume.

**`high`, not `ultra`.** `ultra` is user-triggered and billed, an agent cannot launch it,
and attempting it wastes a turn. Reach for `max` instead when the change is large or lands
on something load-bearing.

**Fix everything it finds, on the same branch, then re-run the gate.** `--fix` applies
findings to the working tree, so the commit and the push are still yours to make. Where a
finding is wrong, write the reason in the PR body: a disagreement recorded is a decision,
and a finding dropped in silence is a defect with a delay on it.

**Nothing the review publishes may carry an AI attribution.** `--comment` posts to the PR
under the account running it, and the marketplace `code-review` plugin — present under
`~/.claude/plugins/marketplaces/` and deliberately not enabled — ends every comment it
writes with a "Generated with Claude Code" line. The rule against that is absolute and
lives in `~/.claude/CLAUDE.md`. Prefer `--fix` and a summary in your own words; if you do
post, read what you are posting first.

## Before proposing new machinery

1. `grep` the constructor or signature for the parameter you want to add.
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

**Tradeoff:** these bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think before coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

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

Ask: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical changes

**Touch only what you must. Clean up only your own mess.**

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.
- Remove imports, variables and functions that *your* changes orphaned; leave
  pre-existing dead code alone unless asked.

The test: every changed line should trace directly to the request.

## 4. Goal-driven execution

**Define success criteria. Loop until verified.**

- "Add validation" → "write tests for invalid inputs, then make them pass"
- "Fix the bug" → "write a test that reproduces it, then make it pass"
- "Refactor X" → "ensure tests pass before and after"

For multi-step work, state the plan as steps with their checks, then run it.

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due
to overcomplication, and clarifying questions arriving before implementation rather than
after mistakes.

## Where they bite hardest in this project

Not decoration — each of these has already cost time here.

- **§1 and §2 against the core repository.** The predicate-router episode is the worked
  example above: a design was written before the core was read, and the second pass cut it
  by three quarters because the mechanism already existed and the inference it rested on had
  been explicitly rejected upstream. "Think before coding" here means *read `INTERNALS.md`,
  the roadmap's deferred list, and the test names* — not merely pause.
- **§3 against a vendored tree.** `plugin/skills/` is not yours to improve. Style, wording
  and formatting there are upstream's; the only sanctioned local edit is the one line
  `skill.lock` and the drift test know about.
- **§4 against silent failures.** Most defects in this repository do not raise. "Verify"
  therefore has to mean comparing output — bytes, counts, a diff against a known-good run —
  never that a command exited 0 or ran fast. A hook that returns nothing is the fastest hook
  there is.

One local amendment to §3, because this repository's own rule is stricter, not looser:
**documentation ships in the same commit as the code.** Updating the README, `CHANGELOG.md`
or a tool description alongside a behaviour change *is* the surgical change, not scope creep.
