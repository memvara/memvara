# Subject and predicate conventions for memory writes

**Status: proposal, 2026-09-08. Nothing described here is implemented.** It exists to be
argued with. The open questions at the end have to be answered before any of it is built.

The question this answers: when an MCP plugin — Claude Code, Grok, Cursor — writes a claim
to memvara, what should the subject be, and who should decide it? The current answer is
"whatever the model types", and this document argues that the fix is not the one that looks
obvious.

## What the store actually looks like today

`memory_stats` against the hosted store for tenant `prj_3c04449a3d9947f7b9bbbafb1d51d052`
reports 1416 live claims and a **join rate of 0.6%** — 8 claims whose object is the subject
of another claim. Everything else is a leaf hanging off a handful of hub nodes. A graph walk
has almost nowhere to go.

That number is the one the graph leg was measured against. `Memvara.connectivity()` in
`memvara/core.py:3354` records the comparison: on a corpus that is 40.6% joinable the graph
leg gains 13 points on chained questions, and on a corpus that is 0.0% joinable it *loses*
1.6. So at 0.6% the leg is switched on, costing time, and returning nothing. The gate closes
it only at exactly zero joins, which this store is just above.

Three conventions for the subject are live in the same store at the same time, and none of
them knows about the other two.

| Convention | Written by | Example |
|---|---|---|
| `user` | the model, through `memory_remember` | `user prefers_prose plain` |
| the git remote's basename | the capture hook, via `project_subject()` in `plugin/hooks/lib/extract.py:267` | `memvara-cloud deploys_to fly.io` |
| `project:<absolute path>` | the model, following the note at `plugin/hooks/recall.py:622` | `project:/Applications/workstation/agent-memory uses postgres` |

The third one is the worst of the three, because an absolute path is per-machine and
per-worktree. This checkout is at `/Applications/workstation/agent-memory` while the project
it holds is called memvara, so the identity is wrong as well as unstable.

## Where the star comes from

The subject argument of every write tool is declared once, at `memvara/server/tools.py:337`:

```
"default": "user",
"description": "Who or what the fact is about. Almost always 'user' — the person you are
 talking to. Use another subject only for a named third party the user has told you about."
```

That description is doing exactly what it says. A model reading it files nearly everything
under `user`, so the store fills with `user` → *value* edges whose objects are strings that
are never themselves subjects. This is not model misbehaviour and it is not subject variance.
It is the tool description working as written, and it is the single largest cause of the 0.6%.

Changing that description is the cheapest change in this document and probably the largest
effect.

## Why "the subject is always the git repo name" would make it worse

The proposal that prompted this document was: every MCP plugin sets the subject to the
project name, taken from the git repo, so that supersession is deterministic and a graph
becomes runnable. The diagnosis behind it is right. The remedy is aimed at the wrong column,
for three reasons.

**It drives the join rate to zero by construction.** A subject that is a constant per
repository can never appear as some other claim's object, and no object is ever a subject
either. One node, several hundred leaves, no edges. By the measurement in `core.py:3354`
that is the configuration where the graph leg is worth minus 1.6 points. The thing the
proposal is for is the thing it prevents.

**It moves the variance rather than removing it.** A slot is the pair `(subject, predicate)`
— `fact_key_for()` in `memvara/types.py`. Hold the subject constant and the slot is the
predicate alone, so every distinct fact about the repository now has to be told apart by
predicate alone. The model's freedom moves one column left, and it moves into the column
where collisions are destructive: "the deploy target is Fly.io" and "the user prefers plain
prose" would be competing in one namespace, and for a `Cardinality.ONE` predicate the second
would end the first.

**The repository name is not a stable identifier.** Worktrees, forks, renames, monorepos
holding several logical components, and two unrelated repositories both called `code`.
`project_subject()` already keys on the remote rather than the path for exactly this reason,
and its docstring says so — but it keeps only the basename, so `acme/parser` and
`widgets/parser` become one subject.

There is a narrower version of the proposal that is simply correct, and it is kept below: for
a claim whose subject genuinely *is* the project, the project id should be a canonical string
the plugin computes, not a phrase the model invents.

## What is already built, and was missed

Two pieces of the obvious fix already exist in this repository.

**Subject canonicalisation is solved deterministically.** `entity_key()` in
`memvara/entities.py:140` folds case, punctuation, accents, articles and corporate suffixes,
so `memvara`, `Memvara`, `The Memvara Corp.` and `memvara_cloud` versus `memvara-cloud` all
resolve without a model call. `EntityRegistry` adds learned aliases on top for the folds the
pure function cannot see, `Reconciler` stamps the resolved identity into `Claim.meta` at write
time so a later alias cannot silently re-key the past, and
`memvara.write.reconcile.backfill_entities` is the named, dated, dry-run-by-default operation
that applies a late alias to existing claims.

That last one matters here: an earlier draft of this argument proposed adding a `merge_subject`
call to mirror `merge_predicate`. It is already there under a different name. The
orthographic half of the subject-variance problem is not an open problem.

**The project key is already computed from the git remote.** `project_subject()` at
`plugin/hooks/lib/extract.py:267` does the derivation the proposal asks for, and
`capture.py:327` stamps it into the episode and bakes it into the extraction prompt. What it
does not do is reach the MCP tools, so a claim the model writes deliberately through
`memory_remember` ignores the convention that a claim captured automatically follows.

So the real gap is narrower than it looked: the conventions exist, they disagree, and the one
surface the model actually reads tells it to use `user`.

## The rule this proposes

**The subject is the entity the claim is about. The plugin decides the identity of that
entity; the model decides which entity it is.**

Three classes, with a different source for each.

1. **Facts about the person.** Subject `user`. Unchanged, and still the common case for
   preferences and standing instructions.
2. **Facts about the project.** Subject is the canonical project id, computed by the plugin
   and never typed by the model. `memvara deploys_to fly.io`, not
   `project:/Applications/... deploys_to fly.io`.
3. **Facts about a third thing** — a service, a library, a person, a vendor, a benchmark.
   Subject is that thing, and this is the class that produces joins, because these subjects
   are also things that appear as objects elsewhere.

Class 3 is the one that decides whether the graph is worth running, and no rule about
repository names touches it. `memvara embeds_with voyage-3` joins to `voyage-3 priced_at
$0.06/M` only because `voyage-3` is a subject in its own right somewhere.

### The canonical project id

Normalise the origin remote to `host/owner/repo`, lowercased, with the `.git` suffix and any
credentials stripped:

```
git@github.com:memvara/memvara-cloud.git  ->  github.com/memvara/memvara-cloud
https://github.com/memvara/memvara-cloud   ->  github.com/memvara/memvara-cloud
```

Fall back to the toplevel directory name when there is no remote, as `project_subject()`
already does, and mark that case so it can be re-keyed later if the repository is pushed.
Never use the working directory: a worktree is a different path for the same project, which is
the failure this file's second table records.

Whether the id should be the full `host/owner/repo` or a short alias the user picks per
repository is an open question below.

### The predicate side

Predicates should come from a declared vocabulary rather than being invented per write.
`MEMVARA_PREDICATES` and the packs in `memvara/packs/` are the mechanism, and
`memvara/packs/engineering.toml` is the closest existing pack. What is missing is a pack whose
predicates take **entity-valued objects** — `depends_on`, `deploys_to`, `embeds_with`,
`owned_by`, `blocked_by` — because a predicate whose object is a sentence can never join to
anything, however well the subject is canonicalised.

## What would change, and where

Ordered by effect over cost. Every item is a proposal, not a plan; none is written yet.

1. **Rewrite the `_SUBJECT` description at `memvara/server/tools.py:337`** to state the
   three classes, and remove "Almost always 'user'". This is the change that most directly
   moves the join rate, and it touches one string. `.claude/rules/tool-descriptions.md`
   governs the wording.
2. **Widen `project_subject()` to `host/owner/repo`** and use it as the default subject for
   class 2. Existing claims written under the basename get an alias through
   `backfill_entities`, dated and attributable, rather than a rewrite.
3. **Retire the `project:<absolute path>` convention.** Alias the existing ones onto the
   canonical id. Fix the note at `plugin/hooks/recall.py:622` in the same commit, because
   that note is where the convention is currently documented.
4. **Pass the project id to the server.** The plugin's `plugin/mcp.json` declares an HTTP
   server with no environment block at all, so today there is no channel for it. Either the
   hooks stamp it, or the skill states it and the model supplies it, or the scope grows a
   dimension. This is the biggest unresolved piece and it is an open question below.
5. **Ship a predicate pack with entity-valued objects**, so class 3 claims have predicates to
   land on.
6. **Update the packaged skill** at `memvara/skills/memvara/SKILL.md` to state the rule.
   Note that this tree is vendored into seven downstream plugin repositories that pin it by
   sha, so a change there is a change in all of them.

## What this does not fix

Cross-plugin agreement is not achieved by any of the above. Grok's plugin and Cursor's read
the same packaged skill, so they inherit the rule, but a third-party MCP client writing
straight to the hosted API sees only the tool schema. If the convention has to hold for
clients this repository does not ship, it has to be enforced server-side on write, not
described in a tool description — and that is a different and much larger change.

Nothing here raises the join rate on its own. Rules 1, 5 and the class-3 habit do; rules 2, 3
and 4 make project facts consistent and supersession reliable, which is worth having and is
not the same goal. Conflating the two is what produced the original proposal.

## Open questions

These change the design, and I do not think they can be answered from the code.

1. **Isolation or joins?** Should a fact written in `memvara-cloud` be invisible from
   `memvara-web`, or should the two share entity nodes so a walk can cross between them? They
   are one product, so joins look right — but that is a decision, not a deduction, and it
   determines whether the project id is a scope partition or just another subject.
2. **Where does the project id live?** Subject only, as proposed here; or a fifth scope
   dimension alongside tenant, user, agent and session; or a `meta` field that retrieval
   filters on. Scope is bound at startup and cannot be set per call
   (`docs/claude/mcp-server.md`), so a scope dimension means changing how every client is
   configured.
3. **Full `host/owner/repo`, or a short per-repo alias?** The full form is unambiguous and
   ugly in every recall result. An alias is readable and needs somewhere to be declared.
4. **Is this a memvara convention or a published standard?** If other MCP memory servers are
   meant to adopt it, it cannot depend on `entity_key`, `EntityRegistry` or `backfill_entities`,
   and the document has to be written for people who do not have them.

## How we would know it worked

The join rate is the measurement, and it already exists: `Memvara.connectivity()` returns
`live_claims` and `joinable_claims`, and `memory_stats` prints the ratio. Record it before any
change lands. The benchmark comparison in `core.py:3354` says the graph leg starts paying
somewhere between 0.0% and 40.6%; this store is at 0.6%, and no change in this document should
be called successful on the strength of it having shipped.

A second check, cheaper: count distinct subjects, and count how many of them appear at least
once as an object. Today that second number is 8.

## Read next

[The memory model](claude/memory-model.md) defines the claim, the slot and the two clocks
that everything above depends on. [Consolidation, vocabularies and the
graph](claude/consolidation-and-graph.md) covers the graph leg and the predicate packs, and
[BENCHMARKS.md](BENCHMARKS.md) carries the measured value of the graph leg on both corpora.

Next: [consolidation, vocabularies and the graph](claude/consolidation-and-graph.md).
