# Working in this repository

This file holds the rules a reviewer must be able to enforce on any pull request, in short
form. `docs/claude/working-here.md` has the full text of each one with the incident that
produced it, and `docs/claude/README.md` indexes everything else — one page per subsystem,
plus the rule files that load themselves — in the table at the bottom of this file.
`CONTRIBUTING.md` has the setup, the gates and the scope rules; read it before writing code.

## Files you need to keep but must not commit go in local/

The `local` directory at the repository root is ignored, whole. Put anything there that you
will want again and that must never reach a commit: a script you ran by hand, a captured API
response, a bug harness, a draft. Never a credential, because ignored is not encrypted and
this repository is public. Never the deliverable either: work meant to ship belongs in a
commit on a branch.
[Full rule.](docs/claude/working-here.md#files-you-need-to-keep-but-must-not-commit-go-in-local)

## Documentation ships in the same commit as the code

When you change behaviour, update everything that describes it in the same commit — not in a
follow-up pull request and not in a list at the end of a handoff. Work the inventory out by
looking; it runs to at least `README.md`, `CHANGELOG.md`, `docs/UPGRADING.md`,
`docs/INTERNALS.md` and the packaged skill at `memvara/skills/memvara/SKILL.md`. Wrong
documentation is worse than missing documentation: missing documentation sends a reader to
the code, wrong documentation sends them somewhere confidently. If a change genuinely cannot
carry its documentation, say so in the pull request body, name the file, and open an issue. A
deferral somebody can see is fine; a silent one is a defect with a delay on it. Two easily
missed places have their own rule files, loaded when you open the file they are about:
`.claude/rules/tool-descriptions.md` and `.claude/rules/doctests.md`.
[Full rule.](docs/claude/working-here.md#documentation-ships-in-the-same-commit-as-the-code)

## How to write

This governs all prose: documentation, comments, docstrings, commit subjects, pull request
bodies, and the tool descriptions the MCP server hands to a model at runtime. It replaces the
voice in the existing files rather than describing it, so those files are not the target.

Work out what something means before deciding how to say it, and lead with the answer. Write
sentences a competent person would say out loud, prefer simple words when they are accurate,
and cut filler — but do not compress until the text turns cryptic. Use one term per concept,
which matters more here than in most repositories because `ended`, `retired` and `erased` name
three different things and that difference is the product. Match your confidence to the
evidence, keep the technical depth while making it readable in one pass, and do not change
the technical relationship a sentence describes while improving it. Commit subjects and pull
request titles say what changed, in a normal sentence, with the reasoning in the body. Tool
descriptions in `memvara/server/tools.py` are the one exemption: precision outranks plain
wording.
[Full rule, with the four places it bites here.](docs/claude/working-here.md#how-to-write)

## Write plainly, and run only the checks a docs-only change can move

Every sentence must be understood on the first read by someone with no context: a full
sentence with a clear subject and verb, saying what the thing is and then what it means for
the reader. Read it back and rewrite anything a colleague would need to hear twice. The
clipped style of the older files here is not the target.

A change that touches only prose runs only the checks it can affect. Do not run the full gate
for a documentation change; it takes about nine minutes and a prose edit cannot move a test
suite. Run the type check if a typed file changed, and the specific tests that read the
changed document, and quote their "N passed" lines in the pull request body. A code change
still gets the full gate.
[Full rule.](docs/claude/working-here.md#write-plainly-and-run-only-the-checks-a-docs-only-change-can-move)

## More than one agent may be working in this checkout at once

Assume files you did not touch are somebody else's unfinished work, and that they have no way
to know you exist.

1. **Commit files by name.** Never `git add -A`, `git add .`, or `git commit -a`.
2. **Never `git stash`, `git checkout <file>`, `git restore` or `git reset` a file you did
   not edit.** Each silently destroys uncommitted work.
3. **Work on a branch and open a pull request.** The `main` branch is where sessions collide.
4. **Before editing a file you did not create, run `git status`.** A file already modified is
   one somebody is in the middle of; say what edit you need rather than making it.
5. **Never overwrite a document you did not write.** Append, or pick a distinct filename.
6. **Use a private `COVERAGE_FILE`.** Two concurrent runs clobber a shared coverage file and
   the report is wrong in the direction that looks fine.

[Full rule.](docs/claude/working-here.md#more-than-one-agent-may-be-working-in-this-checkout-at-once)

## A PR you opened gets a code review before it is merged

Open the pull request, then review it with `/code-review high <PR number>`, then fix what the
review found. In that order, and all of it before anybody merges. Run it on the latest Sonnet,
`claude-sonnet-5` today, by dispatching a subagent pinned to that model — the command takes no
model argument, and a subagent takes one, so the session's own model stops mattering. Use
`high`, or `max` for a large or load-bearing change; `ultra` is user-triggered and an agent
cannot launch it. Read the report before acting on it: a subagent reports confidently and
self-checks badly, so verify each finding against the code. Wait for the fan-out rather than
for the agent you dispatched, which returns as soon as the finders are started; each finder
reports on its own. Fix everything it finds on the
same branch, then re-run the gate, and write the reason in the body where a finding is wrong.
The pull request body says the review ran, at what effort, and what it found. It never names
the model that reviewed, and nothing the review publishes may carry an AI attribution — say so
in the brief you hand the subagent, which does not inherit the reason for it.
[Full rule.](docs/claude/working-here.md#a-pr-you-opened-gets-a-code-review-before-it-is-merged)

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
fp() {
  local root out
  root=$(git rev-parse --show-toplevel) || return 1
  out=$(cd "$root" && git ls-files -co --exclude-standard -z -- "$@" \
        | LC_ALL=C sort -z | xargs -0 git hash-object) || {
    echo "fp: hashing failed (is a tracked file deleted from the working tree?)" >&2
    return 1; }
  [ -n "$out" ] || { echo "fp: no files matched: $*" >&2; return 1; }
  printf '%s\n' "$out" | git hash-object --stdin
}

td() { printf '%s' "$*" | git hash-object --stdin | cut -c1-12; }
```

No commit SHA enters that pipeline, which is the whole point: `git hash-object <file>`
hashes the file on disk, not the blob in the index. Rebase, amend and squash rewrite
history without touching the working tree, so they cannot move the number. A `git add`
cannot move it either — the file is hashed whether or not it is staged, so an uncommitted
edit is caught. Verified against all four: a working-tree edit moves it, a new untracked
file moves it, restoring the file returns it to the original digest, and committing does
not change it.

**The three guards are not decoration — each closes a silent false pass**, and each was
reproduced before being written down.

*It refuses to return a digest for nothing.* The naive one-liner pipes an empty file list
into `git hash-object --stdin`, which happily hashes empty input and returns
`e69de29bb2d1d6434b8b29ae775ad8c2e48c5391` with exit status 0. Two different typos in the
path arguments produce that same digest, so one mistyped input set records a checkpoint that
every later mistyped check matches — the validation then never runs again.

*It fails when a tracked file is missing from the working tree.* `git ls-files -c` still
lists a file you deleted without `git rm`, and `git hash-object` aborts the entire `xargs`
batch at the first unreadable path — so every file sorted after it is silently dropped from
the digest. Reproduced on a scratch repository: with `b.txt` deleted, editing `d.txt` left
the digest identical. Every edit to those files would have been invisible.

*It hashes from the repository root.* Both `.` and `':!CLAUDE.md'` resolve against the
current directory, so the same command run from `tests/` covers 51 files instead of 221 and
returns a different, entirely valid-looking digest. `cd "$root"` makes the input sets below
mean the same thing wherever you run them.

`td` exists because the fingerprint is only half the state; see the checkpoint rules.


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
<validation> <fingerprint> <tool-digest> pass <timestamp> <tool versions>
```

The first three fields are the key, and all three have to match before a result is reused.
Compute them into shell variables first, so that a failing `fp` aborts the line instead of
letting an empty digest reach `grep`:

```bash
V=mypy; T="$(python3 -m mypy --version)"
FP=$(fp memvara pyproject.toml) && TD=$(td "$T")
```

```bash
V=gate; T="$(python3 -m coverage --version | head -1) / $(python3 -V)"
FP=$(fp . ':!CLAUDE.md') && TD=$(td "$T")
```

Both are written out because the second is easy to improvise wrongly: `coverage --version`
prints `Coverage.py, version 7.15.4 with C extension`, so the version is the third field and
not the second. Then the check and the record are the same two commands either way:

```bash
grep -q "^$V $FP $TD pass " local/validation-checkpoints 2>/dev/null \
  && echo "reuse: $V already passed for this code state, with this toolchain"
```

```bash
mkdir -p local && printf '%s %s %s pass %s %s\n' "$V" "$FP" "$TD" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$T" >> local/validation-checkpoints
```

Four rules hold the thing up.

**Only a pass is ever recorded.** A failure needs a fix, and the fix moves the fingerprint,
so there is nothing to cache. That makes a missing entry unambiguous: it means *run it*.

**Never write a checkpoint for a run whose output you did not read.** A checkpoint is a
claim that you saw a number, so the exit status is not enough: an earlier session recorded a
local gate run here exiting 139 after the tests had passed but before `coverage report`
printed anything, which is why `CONTRIBUTING.md` spells the gate as two commands rather than
one. A checkpoint written from that would record a coverage gate that never reached a
verdict. Read the percentage, then write the line.

**The toolchain is part of the key, not just the record.** `pyproject.toml` pins
`mypy>=1.8` rather than an exact version, so upgrading the checker can produce new errors on
unchanged code — the installed one here is already 2.3.0. Recording the version is not
enough to prevent that: a check keyed on the fingerprint alone still hits after an upgrade,
which was measured by rewriting a recorded version and watching the reuse check succeed.
That is what `td` closes. The digest of the version string sits in the key, so upgrading
mypy misses and the checker runs again.

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


### How this sits with the rules above

Five rules elsewhere in this file predate this section and one of them can be read against
it. None is weakened here; each is worth stating precisely, because an ambiguity between
two instructions is resolved by whichever the reader happens to hit first.

**"Fix everything it finds, then re-run the gate" still means what it says.** Applying
review fixes edits source, which moves the fingerprint, so the gate is stale and has to
run — this section agrees and does not soften it. What it rules out is treating *the
review itself* as the trigger. A review that changed nothing leaves every checkpoint
standing; a review whose fixes you applied has already invalidated them without anyone
having to decide.

**"Ensure tests pass before and after" (§4) costs one run, not two,** when the "before"
state already has a checkpoint. The requirement is evidence at both ends, not two
executions of the same command against identical bytes.

**Reporting a reuse in the PR body.** `CONTRIBUTING.md` and the review section both expect
the PR body to say what was run. A reused result still has to appear there, and it is
written as a reuse rather than dressed up as a fresh run: name the validation, the
fingerprint it passed against, and when. "gate reused from the checkpoint recorded against
f0031a8f; mypy re-run, clean" is a complete and honest sentence. What is not acceptable is
a PR body that implies a run you did not make — the record exists so a reader knows what
was verified, and a reused pass is a real answer to that.

**"Verify means comparing an output, never that a command exited 0" is the reason this
works, not an exception to it.** Reuse does not skip the comparison; it moves it. The
checkpoint exists only because somebody read a number, and the fingerprint is the evidence
that the number still describes this code. That is also why a checkpoint is never written
from an exit status, and why a missing entry means run rather than assume.

**The boundary against not running a check at all.** The Karpathy notes below say
verification means comparing an output, and `~/.claude/CLAUDE.md` forbids making a check
pass by removing what it was checking. This section is the only sanctioned reason in this
repository to not run a gate now, so the line matters: reuse is permitted when the code
state is byte-for-byte the one that already passed, and never otherwise. "The fingerprint
moved but the change looks harmless" is not reuse — it is the judgement call the
fingerprint exists to replace. When in doubt, the cost of running is a few minutes and the
cost of a wrong skip is a merge nobody checked.

## What you learn here goes in Memvara

Memvara is the memory store for work in this repository, reached through the plugin's MCP
server. Recall from it whenever the answer could depend on something established earlier,
write back what a session a week from now would be sorry to have lost, and do not write to
Claude Code's file-based memory directory.

Three mechanics decide whether a write survives, and all three fail quietly. Write triples
with `memory_remember` rather than prose, because this deployment has no extraction model.
Pass `role="system"` for a transcript, a log or a paste, because the deterministic matcher
still runs on every user-role turn and will record a quoted first-person sentence as a fact
about whoever pasted it. Set `true_since` whenever the fact became true before now.

Correcting a claim is three different writes that record different reasons: `memory_remember`
for a value that has been overtaken, `memory_end` for one that has stopped being true, and
`memory_forget` for one that was never right. None of them deletes anything, so never report a
retirement as a deletion.
[Full rule.](docs/claude/working-here.md#what-you-learn-here-goes-in-memvara)

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
- Match existing *code* style, even if you'd do it differently. For prose, follow
  [How to write](#how-to-write) instead — the voice in the existing files is being replaced,
  not matched.
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

---

# Where the rest of the context lives

`docs/claude/README.md` is this table with more detail behind it.

| If the task touches | Read first | Primary code |
|---|---|---|
| Claims, slots, the two clocks, ending against retiring against erasing | `docs/claude/memory-model.md` | `memvara/core.py`, `memvara/types.py`, `memvara/store/` |
| Writing memory: the gate, extraction, contradiction handling | `docs/claude/write-pipeline.md` | `memvara/write/`, `memvara/llm/` |
| Reading memory: search, recall, ranking, reranking | `docs/claude/retrieval.md` | `memvara/retrieve/`, `memvara/rerank/`, `memvara/embed/` |
| The MCP server, its tools, how a deployment is configured | `docs/claude/mcp-server.md` | `memvara/server/` |
| The hosted client and its relation to memvara-cloud | `docs/claude/remote-and-cloud.md` | `memvara/remote/` |
| Background maintenance, predicate vocabularies, the graph leg | `docs/claude/consolidation-and-graph.md` | `memvara/consolidate/`, `memvara/packs/`, `memvara/retrieve/traverse.py` |
| Counters, benchmark scripts, the demo harness | `docs/claude/telemetry-and-benchmarks.md` | `memvara/telemetry.py`, `bench/`, `demo/` |
| Cutting a release, the npm bridge, the seven plugin repositories | `docs/claude/release-and-plugins.md` | `release/`, `npm/`, `plugin/`, `scripts/sync_plugin_repos.py` |
| How work is done here, with the incident behind each rule | `docs/claude/working-here.md` | none — this is process |
| Nothing; these load themselves | `.claude/rules/tool-descriptions.md` on `memvara/server/tools.py`; `.claude/rules/packaged-skill.md` on `memvara/skills/**` and `plugin/**`; `.claude/rules/doctests.md` on `memvara/**/*.py` | as listed |
