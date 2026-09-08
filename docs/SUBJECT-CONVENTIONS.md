# Subject, entity and predicate conventions for memory writes

**Status: proposal, revised 2026-09-08. Nothing described here is implemented.** The five
blocking questions at the end have to be answered before any of it is built. An earlier
revision of this document was reviewed and found to be conceptually sound but not
implementation-ready, for a reason worth stating at the top: it ran four separate concepts
together. This revision separates them, and most of the new material follows from that
separation rather than from any change of direction.

## The four concepts, kept apart

Everything below depends on these being different things.

| Concept | Answers | Decided by |
|---|---|---|
| **Semantic subject** | What is this claim *about*? | the model, from the sentence |
| **Entity identity** | Which real thing is that, canonically? | the system, deterministically |
| **Project identity** | Which repository is this? | the plugin, from the git remote |
| **Scope** | Where is this claim visible and applicable? | the deployment, at startup |

The original proposal — "set the subject to the git repo name" — is what you get when the
four collapse into one. It uses project identity to do the work of subject, entity identity
and scope simultaneously. Each of those is a different job, and the rest of this document is
mostly about giving each one its own mechanism.

A worked example of why the separation matters:

```
subject   = software:postgresql     ← entity identity
predicate = version                 ← predicate schema
object    = 17                      ← a value, not an entity
scope     = github.com/memvara/memvara-cloud   ← where this is true
```

`postgresql version 16` in another project is not a contradiction of this. It is the same
entity holding a different value in a different scope. If project identity is carried in the
subject instead, the two become one slot and the second write ends the first — a version
change nobody made, well-provenanced and confidently explained by `why()`. That is the same
failure `memvara/entities.py` was written to fix on the spelling axis, arrived at from a
third direction.

## What the store actually looks like today

`memory_stats` against the hosted store for tenant `prj_3c04449a3d9947f7b9bbbafb1d51d052`
reports 1416 live claims and a **join rate of 0.6%** — 8 claims whose object is the subject
of another claim. Everything else is a leaf.

`Memvara.connectivity()` at `memvara/core.py:3354` records the comparison the graph leg was
measured against: on a 40.6%-joinable corpus the leg gains 13 points on chained questions,
and on a 0.0%-joinable one it loses 1.6. At 0.6% the leg runs and returns nothing.

Three subject conventions are live in the same store at once, and none knows about the others.

| Convention | Written by | Example |
|---|---|---|
| `user` | the model, through `memory_remember` | `user prefers_prose plain` |
| the git remote's basename | the capture hook, via `project_subject()` at `plugin/hooks/lib/extract.py:267` | `memvara-cloud deploys_to fly.io` |
| `project:<absolute path>` | the model, following the note at `plugin/hooks/recall.py:622` | `project:/Applications/workstation/agent-memory uses postgres` |

The third is per-machine and per-worktree, so it is unstable as well as wrong.

### Where the star comes from

The subject argument of every write tool is declared once, at `memvara/server/tools.py:337`:

```
"default": "user",
"description": "Who or what the fact is about. Almost always 'user' — the person you are
 talking to. Use another subject only for a named third party the user has told you about."
```

A model reading that files nearly everything under `user`, producing `user` → *string* edges
whose objects are never themselves subjects. This is not model misbehaviour and it is not
spelling variance. It is the tool description working as written, and it is the largest single
cause of the 0.6%.

## 1. Entity identity and typing

**Every entity-valued subject or object has a canonical identity *and* a type. Identical
strings do not imply identical entities across types.**

This is the concept the previous revision was missing entirely, and it is the one that has to
be added first, because several of the other sections are unsafe without it.

Today identity is a bare string. `postgres` could be the software, the company, a service
instance or a database; `python` could be the language, a package or a project; `memvara` is
simultaneously a company, a repository, a product and a Python package. A graph that walks
`memvara depends_on postgres` into `postgres owned_by ...` has no way to know whether those
two `postgres` are the same thing.

Proposed shape — a namespace prefix on the stored key:

```
project:github.com/memvara/memvara
software:postgresql
model:voyage-3
company:memvara
person:<id>
benchmark:longmemeval
value:17
```

The prefix is internal. Recall results should keep rendering the display form, because
`software:postgresql` in every line of every answer is a tax on the reader for a distinction
they already make.

**Typing is what makes the existing repair tools sufficient.**
`memvara/write/reconcile.py:1146` already implements `split_entity()`, which separates one
surface form into two identities either side of an instant, re-stamps retired claims as well
as live ones, undoes a closure that crossed the boundary, and writes a dated `ENTITY_REKEY`
record so `why()` can explain why history changed. Its docstring names exactly the failure
case a reviewer would worry about: two people called John Smith producing a job change nobody
wrote.

But it splits along **time**. Two entities that coexist — Apple the company and Apple the
record label, both current — cannot be separated by it, because there is no instant at which
one became the other. That case needs types, and it is the strongest argument for adding them.

### Normalization is not entity resolution

These are two operations with different risk profiles and they should never be run together.

**Normalization** is deterministic, cheap and low-risk: case, Unicode NFKD, combining marks,
apostrophes, punctuation, leading articles, `.git` suffixes, URL syntax. `entity_key()` at
`memvara/entities.py:140` is exactly this, and it is already correct in the ways that matter
— it never folds on embedding similarity, only on an exact key match or an explicitly recorded
alias, and its module docstring states the rule that when uncertain a surface form is a new
entity.

**Entity resolution** is semantic and needs evidence: `Postgres` → `postgresql`, `FB` → Meta
Platforms, `Big Blue` → IBM. In this codebase that is `EntityRegistry.acquire()`, which asks a
model once per `(owner, fold)` ever, refuses a canonical it cannot look up, and records the
result as an explicit alias.

So the separation the review asks for already exists architecturally. Two things are still
wrong with it. It is not **named** as a separation anywhere, so a future change can quietly
move work across the line. And the one genuinely risky transformation sits on the wrong side:
`_LEGAL_FORMS` stripping means `Apple Inc.` and `Apple` fold together deterministically, with
no evidence and no confidence attached. Under typing that fold is safe within
`company:` and unsafe across types, which is the rule that should govern it.

Concretely: `Meta` and `Meta Platforms` do **not** fold today (`platforms` is not a legal
form), and `Apple Records` and `Apple` do not either. `Apple Inc.` and `Apple` do. The
exposure is narrower than it first looks, but it is real and it is untyped.

## 2. Subject selection is not entity resolution

The previous revision said "the plugin decides the identity of the entity; the model decides
which entity it is." That is right for classes 1 and 2 and wrong for class 3, because the
plugin cannot know whether "we use Voyage for embeddings" names `model:voyage-3`,
`company:voyage` or `service:voyage-api`. That is semantic interpretation and only the model
can do it.

The corrected division of labour:

**The system owns entity identity. The model proposes entity references. Deterministic
canonicalization and reconciliation resolve those references into identities.**

```
   client                 plugin                  server
   ------                 ------                  ------
   semantic claim  ──▶  project identity  ──▶  subject resolver
                        (deterministic)              │
                                                     ▼
                                             entity registry
                                          canonical id + type
                                          aliases + provenance
                                                     │
                                                     ▼
                                            predicate schema
                                    subject type · object type · cardinality
                                    temporal behaviour · graph semantics
                                                     │
                                                     ▼
                                          claim  ⟨s, p, o, scope, time⟩
                                                  │        │
                                          reconciliation  graph traversal
```

Three classes of subject, with a different source for each:

1. **Facts about the person.** Subject `user`. Unchanged.
2. **Facts about the project.** Subject is the canonical project identity, supplied by the
   plugin and never typed by the model.
3. **Facts about a third thing.** The model proposes a reference; the resolver commits an
   identity. This is the class that produces joins, and no rule about repository names
   touches it.

### Precedence when a claim could be two classes

"We prefer Postgres for Memvara projects" is legitimately either
`user prefers software:postgresql` or `project:… uses software:postgresql`.

**Rule: the subject is the semantic owner of the proposition, not the environment the
sentence was typed in.** A preference belongs to whoever holds it; a dependency belongs to
whatever has it. Where a sentence carries both, it is two claims, not one ambiguous one.

The model should be able to emit a candidate rather than a commitment — a proposed surface,
a proposed type and a confidence — and let the resolver decide. That keeps a low-confidence
guess from occupying a slot at full strength, which `memory_remember`'s `confidence` argument
already governs on the value side.

## 3. Object resolution

A join exists only when `object_key(A) == subject_key(B)`, so object canonicalization matters
at least as much as subject canonicalization and the previous revision barely mentioned it.

The pipeline, in order:

```
raw object text
   → predicate declares the expected object type
   → classify: entity-valued or value-valued
   → normalize (deterministic)
   → resolve against the entity registry (evidence-bearing)
   → canonical object identity, stamped into Claim.meta
```

`Reconciler` already stamps `OBJECT_ENTITY` into `Claim.meta` and `Store.adjacent()` already
joins on the folded `object_key`, so the machinery is present. What is missing is the
classification step: nothing today decides whether an object is an entity or a value, so
`17` and `postgresql` are treated identically.

**A value must never be promoted to an entity merely because its string matches some
subject.** Without that rule, every claim whose object is `17` joins to every claim about the
number 17, and the join rate rises while retrieval gets worse.

## 4. Predicate schema

`PredicateSpec` at `memvara/schema.py:77` currently declares `name`, `cardinality`,
`volatility`, `memory_type`, `aliases`, `supersedes` and `learned`. It declares nothing about
types or the graph. Proposed additions:

```toml
[predicate.depends_on]
subject_type = ["project", "software", "service"]
object_type  = ["software", "service"]
cardinality  = "many"
graph        = true
inverse      = "depended_on_by"

[predicate.deploys_to]
subject_type = ["project", "service"]
object_type  = ["service", "infrastructure"]
cardinality  = "many"
graph        = true
inverse      = "hosts"

[predicate.prefers]
subject_type = ["user"]
object_type  = ["value", "entity"]
cardinality  = "many"
graph        = false
```

This gives the server something enforceable rather than something to hope for, and it is what
lets a type violation be refused at write time instead of discovered in a traversal.

Note that `supersedes` already exists for cross-predicate supersession — asserting
`unemployed` retires `works_at`. A constant-subject scheme would widen the blast radius of
every such rule to the whole repository, which is a further reason not to adopt one.

## 5. Graph edge semantics

**An entity-valued object is not automatically a useful edge.** The previous revision implied
it was, and that is the assumption that makes join rate a gameable metric.

`graph = true` on the predicate is what makes an edge traversable. `Store.adjacent()` and
`GraphTraverser` already accept a `predicates` filter, so the plumbing for a typed walk
exists; what is missing is the declaration that would populate it by default rather than per
call.

**Inverse predicates** should be declared, not stored. `memvara depends_on postgresql` should
be walkable as `postgresql depended_on_by memvara` without a second row — writing the inverse
claim would double the store, and worse, would give the inverse its own independent temporal
and supersession behaviour, which is a correctness bug waiting to happen rather than a
convenience.

## 6. Scope is not subject

**Subject answers "what is this about". Scope answers "where does this apply".** Overloading
subject to provide isolation is the single most likely way to introduce reconciliation bugs
while fixing connectivity.

The `postgresql version` example at the top of this document is the case. It is also the case
that interacts with the bitemporal work: the same entity holds different states in different
projects *and* at different times, and if project identity lives in the subject then a
cross-project difference is indistinguishable from a change over time.

Scope today is the four-part `Scope` in `memvara/types.py`, bound at startup and not settable
per call (`docs/claude/mcp-server.md`). There is no project dimension, and the plugin's
`plugin/mcp.json` declares an HTTP server with no environment block at all, so there is
currently no channel to carry one. This is open question 3.

## 7. Project identity

Normalise the origin remote to `host/owner/repo`, lowercased, `.git` and any credentials
stripped:

```
git@github.com:memvara/memvara-cloud.git    ->  github.com/memvara/memvara-cloud
https://github.com/memvara/memvara-cloud.git ->  github.com/memvara/memvara-cloud
https://user:token@github.com/memvara/x      ->  github.com/memvara/x
```

The normaliser must handle SSH aliases from `~/.ssh/config`, self-hosted GitHub Enterprise,
GitLab and Bitbucket, trailing slashes, URL-encoded path segments, and case. Host and owner
fold to lowercase; the repository segment does too, accepting that a host which distinguishes
`Foo` from `foo` will collide, because every host in practice does not.

`github.com/acme/foo` and `gitlab.com/acme/foo` are **different projects**. Nothing should
try to detect that they might be the same.

Fall back to the toplevel directory name when there is no remote, and mark that identity as
provisional so it can be re-keyed if the repository is later pushed. Never use the working
directory: a worktree is a different path for the same project, which is the failure the
`project:<absolute path>` convention already demonstrates.

### Forks, mirrors and renames

**Forks, mirrors and renamed repositories are distinct project identities unless an explicit
relationship is recorded.** `github.com/memvara/memvara` and `github.com/inderjeet/memvara`
do not share project facts by default.

The relationship, when it exists, is a claim like any other:

```
project:github.com/inderjeet/memvara  fork_of   project:github.com/memvara/memvara
project:github.com/memvara/memvara    renamed_from project:github.com/memvara/agent-memory
```

which is itself a graph edge, and a much cleaner place for the semantics than inside the
identity function. Whether a traversal should follow `fork_of` when answering a question about
the fork is a retrieval decision, made once, visibly.

## 8. Identity provenance

Because this design moves some decisions away from the model, it has to be possible to tell
afterwards who decided what. Every resolved identity should carry:

```
subject                = project:github.com/memvara/memvara
subject_source         = plugin | model | user
subject_resolution     = deterministic | alias | registry | novel
resolution_confidence  = <float, absent when deterministic>
```

`EntityResolution` at `memvara/entities.py` already returns `method` — `empty`, `alias`,
`known` or `novel` — and `EntityRegistry.methods` already counts resolutions by method, so
most of this exists and is not being persisted onto the claim. `Claim.meta` is where it
belongs, alongside the existing `SUBJECT_ENTITY` and `OBJECT_ENTITY` stamps.

## 9. Merge and split lifecycle

```
raw surface form
   → normalized key            entity_key(), pure function
   → alias                     EntityRegistry.learn_alias()
   → merge applied to history  backfill_entities(), dated, dry-run by default
   → split                     split_entity(), temporal, dated
   → typed split               DOES NOT EXIST
```

Four of the five stages are built. The gap is the last one: `split_entity()` separates an
identity across an instant, which repairs "one name, two people, one after the other". It
cannot repair "one name, two things, at the same time", because there is no boundary instant
to pass it. Under section 1's typing that repair becomes expressible.

Every stage past normalization must stay dated, attributable and reversible in the sense the
repository already means by that word: nothing is deleted, `history()` keeps answering, and
`why()` can say why the past changed shape.

## 10. Server-side invariants

The previous revision put "rewrite the `_SUBJECT` description" first. That is still the
cheapest large win, but it is **guidance, not an invariant**, and the distinction has to be
explicit or the store is back in the same state in six months with the variance generated by
different clients instead of by one.

**Guidance** — the tool description and the packaged skill:

- For a fact about the project, use the supplied project identity.
- For a fact about the person, use `user`.
- For anything else, name the thing itself.

**Enforcement** — the server, on write, for every client including ones this repository does
not ship:

- A project subject that is an absolute filesystem path is refused.
- A project subject not matching the canonical `host/owner/repo` shape is refused.
- An entity identity is normalized server-side; a client cannot store an unnormalized key.
- A predicate whose declared `subject_type` or `object_type` the claim violates is refused.
- An alias may not merge across entity types.
- A value object is never promoted to an entity by string match alone.
- A graph edge never crosses a tenant boundary, and crosses a scope boundary only where the
  scope rule explicitly permits it.
- No client may backdate the transaction clock. This one already holds — invariant 8 in
  `INTERNALS.md` — and is listed because it is the model the others should follow.

## 11. Migration invariants

`backfill_entities` is the right instrument, and what it is asked to do has to be stated
precisely, because the two readings have very different consequences for history.

**Attach a resolved identity; do not rewrite the original.** A claim keeps the subject text it
was written with. The canonical identity goes into `Claim.meta`, which is already how
`SUBJECT_ENTITY` works and already what `Reconciler._stamp` guarantees, so an alias learned in
month six does not silently re-key month one.

Migration must be:

- **Idempotent.** Running it twice leaves the same store as running it once.
- **Dry-run by default.** Already true of `backfill_entities` and `split_entity`.
- **Dated and attributable.** Each touched claim gets an `ENTITY_REKEY` record so `why()` can
  explain the change.
- **Ordered deterministically.** Newly-colliding live claims replay in `(recorded_at, id)`
  order so the supersession chain rebuilds the same way every time. Already implemented.
- **Reported by count.** Claims scanned, re-stamped, closures undone, and — the number that
  actually matters — supersessions created or destroyed by the migration.

That last one deserves emphasis. Re-keying subjects **changes what contradicts what**. Claims
that coexisted start retiring each other. `entities.py` says this in its own words: history is
stable unless an operator asks for it not to be. Any migration under this design is such an
ask, and must report the temporal damage it did rather than only the rows it touched.

## 12. Success metrics

**Join rate is a diagnostic, not a target.** It is gameable: point every claim at a handful of
generic entities — `user`, `project`, `api`, `database` — and the graph looks beautifully
connected while retrieval gets worse. The previous revision proposed it as the success
criterion, which was a mistake.

Primary, and the only one that can authorise shipping:

- **Benchmark retrieval quality improves.** `docs/BENCHMARKS.md`, the existing corpora, the
  existing harness.

Secondary:

- Useful join rate — joins on `graph = true` predicates, not all joins — rises.
- Graph-assisted recall rises.
- Graph-assisted precision does not regress.

Safety, each of which can veto a release on its own:

- False-traversal rate falls. Traversals connecting semantically unrelated claims.
- Entity collision rate falls. Two distinct real things folded into one identity.
- Cross-scope leakage is zero.
- Project identity collisions are zero.
- Subject migration is idempotent.
- Historical claim identity is preserved.

Baseline before anything ships: 1416 live claims, 8 joinable, 0.6%. `memory_stats` prints it
and `Memvara.connectivity()` returns the two counts it is computed from.

## 13. Adversarial test cases

The benchmark is the authority, so these are the cases that have to be in it. Each one is a
way for this design to be wrong that a join-rate number would not show.

**Identity folding.** `Memvara` / `memvara` / `MEMVARA` fold to one; `memvara` and
`memvara-cloud` stay two; `Apple Inc.` and `Apple Records` stay two; `Meta` and `Meta
Platforms` stay two unless explicitly aliased.

**Project identity.** Two worktrees of one repository; a fork; a rename; a mirror; a monorepo
holding several logical components; a repository with no remote; the same repository name
under two owners; the same owner and name on two hosts.

**Entity resolution.** `Postgres` / `PostgreSQL` / `postgresql` / `PostgreSQL database`
resolve together only through a recorded alias, never by similarity.

**False joins.** `Apple` the company must not connect to `Apple` the record label. `17` as a
version must not connect to `17` as an age.

**Cross-project.** Project A uses postgresql 16 and project B uses postgresql 17. Neither
supersedes the other, and asking either project returns its own answer.

**Temporal.** A project used postgresql 16 in 2025 and uses 17 in 2026. This *is* a
supersession, and it must remain distinguishable from the cross-project case above. **Entity
identity must not become state identity** — this is the case where the two designs are most
easily confused, and the bitemporal machinery is what keeps them apart.

## Sequenced changes

Ordered so that nothing depends on something later in the list. None of it is written.

1. **Entity types and namespaced keys** (section 1). Everything else is either unsafe or
   unenforceable without this.
2. **Predicate schema fields** — `subject_type`, `object_type`, `graph`, `inverse` (section 4).
3. **Object classification and resolution** (section 3).
4. **Server-side invariants** (section 10), which the two previous items make expressible.
5. **Canonical project identity** — widen `project_subject()` to `host/owner/repo`
   (section 7).
6. **Rewrite the `_SUBJECT` description** at `memvara/server/tools.py:337`, and the packaged
   skill at `memvara/skills/memvara/SKILL.md`. Cheap and effective, but guidance only, and
   deliberately after the enforcement it complements. Note the skill tree is vendored into
   seven downstream plugin repositories that pin it by sha.
7. **Retire `project:<absolute path>`**, aliasing existing claims onto the canonical identity
   and fixing the note at `plugin/hooks/recall.py:622` in the same commit.
8. **Typed split** for the coexisting-entity case (section 9).
9. **Predicate pack** with entity-valued, graph-traversable predicates.
10. **Adversarial benchmark cases** (section 13), added before any of the above is measured
    rather than after.

## Five blocking questions

Implementation should not start until these are answered.

1. **What is an entity's type, and where does the namespace live?** A prefix on the stored
   key, a column, or a `meta` field. This decides how much of `entities.py`, `types.py` and
   both store backends move.
2. **How are object entities classified and resolved?** Specifically: what decides that an
   object is an entity rather than a value, given that today nothing does.
3. **What is the relationship between subject and scope, and may a traversal cross scopes?**
   Scope is bound at startup and the hosted plugin passes no environment at all, so a project
   scope dimension means changing how every client is configured. The alternative is project
   identity as subject only, which does not give isolation.
4. **Which predicates are graph-traversable, and what are their type and cardinality rules?**
   This is a vocabulary that has to be written, not derived.
5. **How is useful connectivity measured, as distinct from raw join rate?** Section 12
   proposes a shape; the benchmark work to make "false traversal rate" and "entity collision
   rate" real numbers does not exist yet.

Questions 1, 2 and 4 are answerable inside this repository. Question 3 is a product decision
about whether memvara, memvara-cloud and memvara-web share a memory. Question 5 is benchmark
work that has to happen before anything here can be called successful.

## Read next

[The memory model](claude/memory-model.md) defines the claim, the slot and the two clocks.
[Consolidation, vocabularies and the graph](claude/consolidation-and-graph.md) covers the
graph leg and the predicate packs, and [BENCHMARKS.md](BENCHMARKS.md) carries the measured
value of the graph leg on both corpora.

Next: [consolidation, vocabularies and the graph](claude/consolidation-and-graph.md).
