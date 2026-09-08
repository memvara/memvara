# How work is done here

The root `CLAUDE.md` states each of these rules in a sentence or two, because that file is
loaded at the start of every session and has to stay short. This page holds the same rules
in full, with the incident that produced each one. Read it when you want to know why a rule
says what it says, or when you are about to argue with one.

`CONTRIBUTING.md` covers setup, the gates and the scope rules, and is the file to read
before writing code. This page is about working here rather than about the code.

## Files you need to keep but must not commit go in local/

The `local` directory at the repository root is ignored, whole. Put anything there that you
will want again and that must never reach a commit: a script you ran by hand, an API
response captured as evidence, a harness that reproduces the bug you are chasing, a report
you are still drafting. `git status` stays quiet about it and no `git add` can reach it.

Use it, because both of the obvious alternatives fail, in opposite directions.

- **A temporary directory is deleted without warning.** That is what temporary means. A
  sibling repository lost the only copy of a provisioning script exactly this way. It was
  written to a session scratchpad and a handoff note said "copy it somewhere durable
  first", and the directory was empty before anybody did.
- **An untracked file at the repository root is a file that gets committed.** Not by you.
  By the next `git add -A` that runs in this checkout, under somebody else's commit
  message.

The directory is outside the build as well as outside the commit. `pyproject.toml` builds
the sdist from what version control does not ignore, and `testpaths` is `["tests",
"memvara"]`, so nothing there is collected, packaged or type-checked.

Two things do not belong in it. **Never a credential**, because ignored is not encrypted and
this repository is public, so a mistake here is a disclosure rather than a cleanup job. And
**never the deliverable**: if the work is meant to ship, it belongs in a commit on a branch.
This is where a file goes to be kept, and also where it goes to be forgotten.

## Documentation ships in the same commit as the code

When you change behaviour, update everything that describes it in the same commit. Not in a
follow-up pull request, not in a note for later, not in a "docs to update" list at the end
of a handoff. A deferred documentation change is not a smaller version of the work. It is a
different piece of work, one that nobody has been assigned and that no test will fail over.

The failure is quiet and it compounds. Documentation that is wrong is worse than
documentation that is missing. Missing documentation sends a reader to the code; wrong
documentation sends them somewhere confidently and lets them act on it. The person who pays
is never the author. It is whoever reads it next, without the context that would let them
notice.

"Everything that describes it" is more places than anyone remembers, so work the real
inventory out by looking rather than by trusting a list. It runs to at least `README.md`,
`CHANGELOG.md` for every user-visible change, `docs/UPGRADING.md` when behaviour changes
under someone, `docs/INTERNALS.md`, and the packaged skill at
`memvara/skills/memvara/SKILL.md`, which states outright that it does not repeat what a tool
description says, so text moving between the two has to move in both. `CONTRIBUTING.md`
states the same duty from the other end and names the documents that make specific,
checkable claims.

Two more places are easy to miss, and both are particular to this repository. Each has its
own rule file, which loads when you open the file it is about:
`.claude/rules/tool-descriptions.md` and `.claude/rules/doctests.md`.

If a change genuinely cannot carry its documentation — the document lives in another
repository, or the decision is not final — say so in the pull request body, name the file,
and open an issue. That is a deferral somebody can see. A silent one is a defect with a
delay on it.

## How to write

This section governs prose: documentation, comments, docstrings, commit subjects, pull
request bodies, and the tool descriptions the MCP server hands to a model at runtime. It
**replaces** the voice in the existing files rather than describing it. Those files are
being converted, so do not treat them as the target to match.

The method is **meaning first, then structure, then wording**. Work out what something
actually means before deciding how to say it. Do not take the phrases from an existing
document, an issue, or an earlier commit message and rearrange them, because that produces
text that is topically correct and communicates nothing. If the wording you started from is
awkward, discard it and write the sentence again from the meaning.

**Lead with the answer.** State the conclusion, then explain it. A paragraph that builds
toward a point the reader could have had in the first sentence has wasted their time.

**Write sentences a competent person would say out loud.** Read it back. If it sounds
strange, rewrite it. Prefer active voice, keep every pronoun's referent obvious, and do not
stack nouns into chains.

**Prefer simple words when they are accurate.** Write "use", not "leverage" or "utilize".
Never pick a more sophisticated word to sound more capable, and never simplify to the point
where the sentence stops being true.

**Cut filler.** Every sentence should carry something. Drop "It is important to note that",
"It is worth mentioning", "There are several considerations to take into account", and the
habit of stating a claim and then restating its cost.

**Do not compress until the text turns cryptic.** "Nothing was written to the ledger, so the
balance never updated" is right. "Ledger had no writes therefore balance absent" is not.
Concise is not the same as incomplete.

**Use one term per concept.** A `session ID` does not become an "interaction identifier"
three lines later. This matters more here than in most repositories, because `ended`,
`retired` and `erased` name three different things and that difference is the product.

**Match your confidence to the evidence.** Write "this means" for something established,
"this likely means" for a strong inference, "this could mean" for a possibility, and say
plainly when you do not have enough information to tell. Do not hedge a fact you have
verified.

**Write so both an engineer and an executive can follow it in one pass.** Keep the technical
depth and make it accessible instead of removing it. Where a term is load-bearing and not
obvious, explain it in a clause and move on. Do not write two versions unless asked.

**Preserve meaning when you rewrite.** Improving a sentence must not change the technical
relationship it describes. Watch for this while converting the existing documents. The old
voice buries real distinctions inside clever constructions, and a fluent rewrite can quietly
drop one.

### Where this bites in this repository

**Commit subjects and pull request titles.** The existing ones are declarative sentences
naming the false belief the code held, such as "A value that replaces nothing looks exactly
like a value that replaced something". They read well and they do not say what changed. Say
what changed, in a normal sentence, and put the reasoning in the body where there is room
for it.

**Tool descriptions in `memvara/server/tools.py` are the exemption.** "Prefer simple words"
never outranks precision there. A model reading them cannot go and check, and the distinction
between ending a fact and retiring it has already been got wrong once in a write receipt.
Plain is good; vague is a defect. The same applies to `CHANGELOG.md` entries describing a
behaviour change somebody will act on. `.claude/rules/tool-descriptions.md` has the whole of
it.

**Docstrings execute.** `pyproject.toml` sets `--doctest-modules`, so a rewritten example
still has to run. Rewriting the prose around an example does not exempt it from the suite.
See `.claude/rules/doctests.md`.

**The packaged skill is vendored downstream.** `memvara/skills/memvara/` is pinned by commit
sha and diffed in seven plugin repositories. Converting its prose is a real change in all of
them, so do it deliberately and in its own commit, never as a drive-by while editing
something else. See `.claude/rules/packaged-skill.md`.

## Write plainly, and run only the checks a docs-only change can move

**Every sentence must be understood on the first read by someone who has no context.** This
applies to documentation, dashboard panel descriptions, metric help strings, code comments,
docstrings, commit messages, pull request bodies and replies to the user. Write the way you
would explain something to a colleague across a desk: a full sentence with a clear subject
and verb, saying what the thing is and then what it means for the reader. For example, write
"This is a count only; no project name is shown", not "A count and never a name".

The short, clever style you will see in older files here is not the target. It comes from
three places, and knowing them helps you resist it.

1. The existing prose in this and the sibling memvara repositories — operations documents,
   READMEs, test docstrings, old commit messages — is written in that style. Those files are
   being converted. Do not copy their voice because it is around you.
2. The memories stored in Memvara by earlier sessions were written in the same style. Treat
   them as notes about facts, not as a writing sample.
3. Your own earlier text in the session. If you notice you wrote a clever sentence, rewrite
   it before moving on.

How to check: read the sentence back. If a colleague would need to hear it twice, rewrite it.

**A change that touches only prose runs only the checks it can affect.** Do not run the full
gate for a documentation change. A prose edit cannot move a test suite, and the gate here
takes about nine minutes. Run the type check if a typed file changed, and the specific test
file that reads the changed document — for example the test that parses a dashboard JSON or
a queries file — and quote its "N passed" line in the pull request body. A code change still
gets the full gate.

Both rules were stated by the user on 2026-09-06. The first came after the user quoted a
dashboard panel description an agent had written and asked "from where are you getting that
you need to write like this?". The second was stated as "Don't run the gate just for the
document update". The same section is in the `memvara`, `memvara-cloud` and `memvara-web`
repositories and in the user's global `CLAUDE.md`.

## More than one agent may be working in this checkout at once

Assume files you did not touch are somebody else's unfinished work, and that they have no
way to know you exist.

1. **Commit files by name.** Never `git add -A`, `git add .`, or `git commit -a`. If you
   cannot list what you are committing, you do not know what you are committing.
2. **Never `git stash`, `git checkout <file>`, `git restore` or `git reset` a file you did
   not edit.** Each of them silently destroys uncommitted work, and `git checkout <file>`
   restores from HEAD rather than from your last edit. It has eaten an uncommitted rewrite
   here.
3. **Work on a branch and open a pull request.** The `main` branch is where sessions
   collide; a branch is yours.
4. **Before editing a file you did not create, run `git status`.** A file that is already
   modified is one somebody is in the middle of. If your change needs it, say what edit you
   need rather than making it.
5. **Never overwrite a document you did not write.** Append to it, or pick a distinct
   filename.
6. **Use a private `COVERAGE_FILE`.** Two concurrent runs clobber a shared `.coverage` file,
   and the report that comes out of that is wrong in the direction that looks fine.

## A PR you opened gets a code review before it is merged

Open the pull request, then review it, then fix what the review found. In that order, and
all of it before anybody merges.

```bash
/code-review high <PR number>
```

The window is narrow at both ends. Run it against a working tree you have not pushed and you
have reviewed something no reviewer will ever see. Skip it and the pull request merges
unreviewed, which is the case this rule exists for. Nothing else in the process looks at the
change with fresh eyes.

**Run it on the latest Sonnet, which is `claude-sonnet-5` today, by dispatching a subagent
pinned to that model.** The `/code-review` command takes an effort level, a target, and
`--comment` or `--fix`. It takes **no model argument**, so it runs on whatever model the
session it runs in is using. A subagent takes a model override, which makes the session's own
model irrelevant:

```
Agent(subagent_type: "general-purpose",
      model: "sonnet",
      prompt: "Run /code-review high <PR number> on this repository. Report every
               finding with its file, line and severity, and say plainly which ones
               you could not confirm. Do not push, comment on the pull request, or
               edit the pull request body. No AI attribution, model name or
               generated-with line may reach GitHub in anything you write.")
```

Dispatch it rather than switching the session model, decided 2026-09-08. Switching was the
earlier instruction and it fails in two ordinary situations: a session running in the desktop
app or any non-interactive context cannot change its own model at all, and a session that
switches has to remember to switch back, which is a step with nothing checking it. A subagent
carries the model as an argument, so the review runs on Sonnet whatever the session is, and
the parent keeps its own model and its own context.

The brief handed to the subagent carries the attribution rule, every time, because a subagent
does not inherit the reason for it. The pull request body says the review ran, at what effort,
and what it found. It never names the model, and no AI attribution of any kind reaches GitHub,
whether the session or a subagent writes the body. Decided 2026-09-07, when review sections
naming the model were found in pull request bodies across the memvara repositories and
scrubbed.

**Read the subagent's report before acting on it.** A fan-out agent converts a task into a
confident summary and checks itself badly; findings come back that the diff does not support.
Verify each one against the code before you fix it, and treat a finding you cannot reproduce
as a finding to argue with in the pull request body rather than one to apply.

**The findings arrive from the fan-out, not from the agent you dispatched.** `/code-review`
splits into several finder agents, and the agent holding the command returns as soon as it has
started them — twice on 2026-09-08 it reported that the review had begun and nothing else,
including the run whose brief told it not to. Each finder reports separately, so wait for
those rather than for the one you dispatched, and expect the same finding from more than one
of them. If your session cannot send a follow-up message to a subagent, an agent that returns
early cannot be resumed at all; relaunching is the only option, and the findings already in
hand stay good.

**Run the checks CI runs, not the ones you remember.** The gate in `CONTRIBUTING.md` is
`mypy -p memvara`, and `.github/workflows/ci.yml` runs a second pass,
`mypy benchmarks/agent_memory --ignore-missing-imports`, that the first does not cover. On
2026-09-08 a pull request went up with a local gate reported green and that second pass
failing on the branch. Read the workflow file before you claim a gate passed.

**Use `high`, not `ultra`.** The `ultra` level is user-triggered and billed, an agent cannot
launch it, and attempting it wastes a turn. Reach for `max` instead when the change is large
or lands on something load-bearing.

**Fix everything it finds, on the same branch, then re-run the gate.** The `--fix` flag
applies findings to the working tree, so the commit and the push are still yours to make.
Where a finding is wrong, write the reason in the pull request body. A disagreement recorded
is a decision, and a finding dropped in silence is a defect with a delay on it.

**Nothing the review publishes may carry an AI attribution.** The `--comment` flag posts to
the pull request under the account running it, and the marketplace `code-review` plugin —
present under the user's plugin marketplaces directory and deliberately not enabled — ends
every comment it writes with a "Generated with Claude Code" line. The rule against that is
absolute and lives in the user's global `CLAUDE.md`. Prefer `--fix` and a summary in your own
words. If you do post, read what you are posting first.

## What you learn here goes in Memvara

Memvara is the memory store for work in this repository, reached through the plugin's MCP
server. Recall from it at the start of a turn whenever the answer could depend on something
established earlier, and write back what a session a week from now would be sorry to have
lost. Do not write to Claude Code's file-based memory directory. It was migrated into Memvara
on 2026-08-23, and a second store that nothing reconciles is worse than one.

Two mechanics decide whether a write survives, and both fail quietly.

**Write triples, not prose.** This deployment runs with no extraction model, which
`memory_stats` reports as `fast-path-only`. A paragraph handed to `memory_add` that matches
none of the fixed sentence forms yields no fact. `memory_remember` with an explicit subject,
predicate and object needs no model and cannot mis-parse.

**That is not the same as `memory_add` being inert, and reading it that way cost a real
stored name here.** The deterministic matcher runs on every turn sent with `role="user"`,
whatever `MEMVARA_LLM` says. It searches rather than anchors, and it strips quotation marks
before it looks, so a first-person sentence quoted inside a log or a docstring is written
down as a fact about whoever pasted it, at a confidence that supersedes what they stated
themselves. On 2026-08-26 a pasted log quoting the docstring of `memvara/write/fast.py` did
exactly that. Pass `role="system"` for a transcript, a log or a paste, and read the `role`
description for the whole of it.

**Set `true_since` when the fact became true before now.** Backfilling a finding from last
week without it records a claim that was never true across its own interval, and the store
then answers historical questions wrongly with no symptom at write time.

Match the conventions already in the store rather than reinventing them: `user` for standing
instructions, and a component key such as `memvara_cloud`, `memvara_web` or `agent-memory`
for a fact about the code. This matters more than tidiness. The graph leg pays off in
proportion to how often one claim's object is another claim's subject, so a claim whose
subject is not `user` is the thing that makes the store answer a question two hops deep.

Correcting a claim is three different writes and they record different reasons. A value that
was right and has been overtaken is a `memory_remember`. One that was right and has stopped
being true is a `memory_end`, at the instant it stopped. One that was never right is a
`memory_forget`. None of them deletes anything, so never report a retirement as a deletion.

## Where the Karpathy guidelines bite hardest in this repository

The four guidelines, the subsection "Where they bite hardest in this repository" and the
amendment that documentation ships in the same commit are in `CLAUDE.md` in full and are not
repeated here.

Next: [the context index](README.md), or the page for the subsystem you are changing.
