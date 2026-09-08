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
