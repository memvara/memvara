# Working in this repository

`CONTRIBUTING.md` has the setup, the gates and the scope rules, and it is the file to read
before writing code. This one covers the things that are about *working here* rather
than about the code, all of which have cost real time, and closes with the general
coding guidelines this project has adopted.

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
`memvara/skills/memvara/SKILL.md`, which states outright that it does not repeat what a
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

## How to write

This section governs prose: documentation, comments, docstrings, commit subjects, PR bodies,
and the tool descriptions the MCP server hands to a model at runtime. It **replaces** the
voice in the existing files rather than describing it. Those files are being converted, so do
not treat them as the target to match — see the amendment to §3 below.

The method is **meaning first, then structure, then wording**. Work out what something
actually means before deciding how to say it. Do not take the phrases from an existing doc, an
issue, or an earlier commit message and rearrange them, because that produces text that is
topically correct and communicates nothing. If the wording you started from is awkward,
discard it and write the sentence again from the meaning.

**Lead with the answer.** State the conclusion, then explain it. A paragraph that builds
toward a point the reader could have had in the first sentence has wasted their time.

**Write sentences a competent person would say out loud.** Read it back. If it sounds strange,
rewrite it. Prefer active voice, keep every pronoun's referent obvious, and do not stack nouns
into chains.

**Prefer simple words when they are accurate.** "Use", not "leverage" or "utilize". Never pick
a more sophisticated word to sound more capable, and never simplify to the point where the
sentence stops being true.

**Cut filler.** Every sentence should carry something. Drop "It is important to note that",
"It is worth mentioning", "There are several considerations to take into account", and the
habit of stating a claim and then restating its cost.

**Do not compress until the text turns cryptic.** "Nothing was written to the ledger, so the
balance never updated" is right. "Ledger had no writes therefore balance absent" is not.
Concise is not the same as incomplete.

**Use one term per concept.** A `session ID` does not become an "interaction identifier" three
lines later. This matters more here than in most repositories, because `ended`, `retired` and
`erased` name three different things and that difference is the product.

**Match your confidence to the evidence.** "This means" for something established, "this
likely means" for a strong inference, "this could mean" for a possibility, and say plainly
when you do not have enough information to tell. Do not hedge a fact you have verified.

**Write so both an engineer and an executive can follow it in one pass.** Keep the technical
depth and make it accessible instead of removing it. Where a term is load-bearing and not
obvious, explain it in a clause and move on. Do not write two versions unless asked.

**Preserve meaning when you rewrite.** Improving a sentence must not change the technical
relationship it describes. Watch for this while converting the existing docs: the old voice
buries real distinctions inside clever constructions, and a fluent rewrite can quietly drop
one.

### Where this bites in this repository

**Commit subjects and PR titles.** The existing ones are declarative sentences naming the
false belief the code held — "A value that replaces nothing looks exactly like a value that
replaced something". They read well and they do not say what changed. Say what changed, in a
normal sentence, and put the reasoning in the body where there is room for it.

**Tool descriptions in `memvara/server/tools.py` are the exemption.** "Prefer simple words"
never outranks precision there. A model reading them cannot go and check, and the
`ended` / `retired` distinction has already been got wrong once in a write receipt. Plain is
good; vague is a defect. The same applies to `CHANGELOG.md` entries describing a behaviour
change somebody will act on.

**Docstrings execute.** `pyproject.toml` sets `--doctest-modules`, so a rewritten example
still has to run. Rewriting the prose around an example does not exempt it from the suite.

**The packaged skill is vendored downstream.** `memvara/skills/memvara/` is pinned by sha and
diffed in seven plugin repositories. Converting its prose is a real change in all of them, so
do it deliberately and in its own commit, never as a drive-by while editing something else.

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

## A validation result belongs to a code state, not to a commit or a conversation

`mypy` and the coverage suite read file contents. They do not read commit messages, branch
names or your place in a workflow. So a passing result stays valid until the contents they
read change, and these are **not** reasons to run either of them again:

- a code review finished, and changed nothing;
- the session resumed, or you are picking work back up;
- a commit was made, amended or squashed;
- the branch was rebased;
- the same command already ran in an earlier phase of this task.

The suite here is the expensive one — it runs under coverage gated at 100%, and
`--doctest-modules` puts every docstring in the package through it as well. Running it
twice on the same bytes buys nothing.

### The fingerprint is the working tree, hashed

Paste this once per session. It hashes the *contents* of every tracked and every
untracked-but-not-ignored file under the paths you give it:

```bash
fp() { git ls-files -co --exclude-standard -z -- "$@" | LC_ALL=C sort -z \
       | xargs -0 git hash-object | git hash-object --stdin; }
```

No commit SHA enters that pipeline, which is the whole point: `git hash-object <file>`
hashes the file on disk, not the blob in the index. Rebase, amend and squash rewrite
history without touching the working tree, so they cannot move the number. A `git add`
cannot move it either — the file is hashed whether or not it is staged, so an uncommitted
edit is caught. Verified against all four: a working-tree edit moves it, a new untracked
file moves it, restoring the file returns it to the original digest, and committing does
not change it.

### The two input sets

| validation | command | `fp` arguments |
| --- | --- | --- |
| `mypy` | `python3 -m mypy -p memvara` | `memvara pyproject.toml` |
| `gate` | `python3 -m coverage run -m pytest && python3 -m coverage report` | `. ':!CLAUDE.md'` |

**The gate's input set is the whole repository, and that is a measurement rather than
laziness.** The tests here read far more than `tests/` and `memvara/`:
`tests/test_doc_links.py` globs `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`,
`SECURITY.md` and all of `docs/*.md`; `tests/test_plugin.py` reads
`memvara/skills/memvara/SKILL.md`, `plugin-claude.md`, `plugin-repos.txt` and the plugin
`README.md`; `tests/test_packaging.py` pins a version string in `README.md`; and the
doc-link tests stat `bench/`, `demo/` and `scripts/` to check that link targets exist. A
shorter list is one I cannot prove, and a wrong narrowing produces a *false pass* — the
gate skipped on the strength of a fingerprint that did not cover what changed. Under-hitting
costs a suite run. Over-hitting ships the broken link that `test_doc_links.py` exists to
catch.

So **a documentation change does invalidate the gate here.** That is the opposite of the
usual rule, and it is a property of this repository's tests rather than a general truth.

`CLAUDE.md` is the single exclusion, because no test reads it — checked across `tests/`,
where every `CLAUDE.md` hit is either prose in a docstring or a `tmp_path` fixture written
by `tests/test_init.py`. **If you ever add a test that reads the root `CLAUDE.md`, delete
the `':!CLAUDE.md'` from the table in the same commit**, or every edit to this file will
silently authorise skipping the gate.

### Recording a checkpoint

State lives in `local/validation-checkpoints`, which is outside the commit — `/local/` is
ignored whole — and outside the build, so nothing here reaches an sdist. One line per
passing run, appended:

```
<validation> <fingerprint> pass <timestamp> <tool versions>
```

Check before running:

```bash
grep -q "^mypy $(fp memvara pyproject.toml) pass " local/validation-checkpoints 2>/dev/null \
  && echo "reuse: mypy already passed for this code state"
```

Record only after you have watched it pass:

```bash
mkdir -p local && printf '%s %s pass %s %s\n' mypy "$(fp memvara pyproject.toml)" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(python3 -m mypy --version)" >> local/validation-checkpoints
```

```bash
mkdir -p local && printf '%s %s pass %s %s\n' gate "$(fp . ':!CLAUDE.md')" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "$(python3 -m coverage --version | head -1) / $(python3 -V)" >> local/validation-checkpoints
```

Both are written out because the second is easy to improvise wrongly. `coverage --version`
prints `Coverage.py, version 7.15.4 with C extension`, so the version is the third field,
not the second.

Four rules hold the thing up.

**Only a pass is ever recorded.** A failure needs a fix, and the fix moves the fingerprint,
so there is nothing to cache. That makes a missing entry unambiguous: it means *run it*.

**Never write a checkpoint for a run whose output you did not read.** A checkpoint is a
claim that you saw a number, so the exit status is not enough: an earlier session recorded a
local gate run here exiting 139 after the tests had passed but before `coverage report`
printed anything, which is why `CONTRIBUTING.md` spells the gate as two commands rather than
one. A checkpoint written from that would record a coverage gate that never reached a
verdict. Read the percentage, then write the line.

**The tool versions are part of the state.** `pyproject.toml` pins `mypy>=1.8`, not an
exact version, so a machine that upgraded mypy can produce new errors on unchanged code.
The recorded version is what tells you a reused result came from the same checker.

**In a worktree, pin `PYTHONPATH` before you run anything you intend to record.** The
editable install on this machine points at the main checkout, so a bare `pytest` in a
worktree can import `memvara` from there instead — passing against code you did not write
and did not fingerprint. That is the one way to get a false pass that the fingerprint cannot
catch, because the fingerprint measured this worktree while the suite measured another. Run
`export PYTHONPATH="$PWD"` first, and confirm it took:

```bash
python3 -c "import memvara; print(memvara.__file__)"
```

### Rebase, review, and the final check

After a rebase, compute the fingerprint and compare it — do not reflexively rerun. A rebase
that only replayed commits leaves the number alone and every checkpoint stands. If conflict
resolution edited source, that is an ordinary code change and the number moves on its own;
you do not need to reason about which files git touched.

After a review, the same. A review that changed nothing invalidates nothing. A review whose
fixes you applied has already changed the fingerprint.

Before you say the work is done, check both entries against the current fingerprint and run
whichever is missing. The last code state has to have a passing `mypy` **and** a passing
gate recorded against it — the point of all this is to validate each code state once, not
to skip validating the one that ships.

## What you learn here goes in Memvara

Memvara is the memory store for work in this repository, reached through the plugin's MCP
server. Recall from it at the start of a turn whenever the answer could depend on something
established earlier, and write back what a session a week from now would be sorry to have
lost. Do not write to Claude Code's file-based memory directory; it was migrated into Memvara
on 2026-08-23 and a second store nothing reconciles is worse than one.

Two mechanics decide whether a write survives, and both fail quietly.

**Write triples, not prose.** This deployment runs with no extraction model, which
`memory_stats` reports as `fast-path-only`. A paragraph handed to `memory_add` that matches
none of the fixed sentence forms is accepted and stored as nothing. `memory_remember` with an
explicit subject, predicate and object needs no model and cannot mis-parse.

**Set `true_since` when the fact became true before now.** Backfilling a finding from last
week without it records a claim that was never true across its own interval, and the store
then answers historical questions wrongly with no symptom at write time.

The conventions already in the store are worth matching rather than reinventing: `user` for
standing instructions, and a component key — `memvara_cloud`, `memvara_web`, `agent-memory` —
for a fact about the code. This matters more here than the tidiness of it: the graph leg pays
off in proportion to how often one claim's object is another's subject, so a claim whose
subject is not `user` is the thing that makes the store answer a question two hops deep.

Correcting a claim is three different writes and they record different reasons. A value that
was right and has been overtaken is a `memory_remember`; one that was right and has stopped
being true is a `memory_end` at the instant it stopped; one that was never right is a
`memory_forget`. None of them deletes anything, so never report a retirement as a deletion.

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
- Match existing *code* style, even if you'd do it differently. For prose, follow
  [How to write](#how-to-write) instead — the voice in the existing files is being replaced,
  not matched.
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

## Where they bite hardest in this repository

Not decoration — each of these has already cost time here.

- **§1 and §2 against what is already decided.** `docs/INTERNALS.md` states the invariants
  and why; `docs/ROADMAP.md` keeps a *Deliberately deferred* list precisely so that
  considered-and-declined stops reading as not-yet-done; and the tests explain reasoning at
  paragraph length. A proposal written without reading those three is usually a rebuild of
  something already here — a plugin-side predicate-router design was cut by three quarters on
  exactly this discovery. "Think before coding" means reading them, not merely pausing.
- **§3 against the packaged skill.** `memvara/skills/memvara/` is vendored into seven
  downstream plugin repos that pin it by sha and diff against it in CI. An unrequested
  formatting improvement there is a change in all of them.
- **§4 against silent failures.** This library's own telemetry module exists because a
  red-team review classified six of eleven long-horizon failure modes as *silent*. "Verify"
  therefore means comparing an output — a count, a series, a diff — never that a command
  exited 0.

§3 has one local amendment here, and it makes the rule stricter rather than looser:
**documentation ships in the same commit as the code**, per the section above. Updating
`README.md`, `CHANGELOG.md`, `docs/UPGRADING.md`, `docs/INTERNALS.md` or a tool description
alongside a behaviour change *is* the surgical change, not scope creep.
