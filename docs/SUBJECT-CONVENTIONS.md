# Subject, entity and predicate conventions for memory writes

**Status: design settled, revision 4, 2026-09-08. Nothing described here is implemented yet.**
The six questions that blocked implementation were answered on 2026-09-08 and are recorded as
[decisions](#the-six-decisions) at the end. Those decisions are the implementation contract;
the sections above them are the reasoning that produced it.

Two rounds of review have shaped this. The first found that the document ran four separate
concepts together; they are separated below. The second found the separation sound and
raised seventeen narrower points, all of which are addressed here except one, which the code
had already answered — see [What is already built](#what-is-already-built), which exists so
that a third review does not propose it again.

## The five concepts, kept apart

Everything below depends on these being different things.

| Concept | Answers | Decided by |
|---|---|---|
| **Semantic subject** | What is this claim *about*? | the model, from the sentence |
| **Entity reference** | Which thing does the model mean by that word? | the model, resolved by the system |
| **Canonical entity identity** | What stable identity represents that thing? | the system, deterministically |
| **Project identity** | Which repository is this? | the plugin, from the git remote |
| **Scope** | Where is this claim visible and applicable? | the deployment, at startup |

Reference and identity are two rows rather than one because the operations have different
natures. Deciding that `Postgres` means the database software rather than the company is
semantic and needs evidence. Deciding that the software's stable identity is
`software:postgresql` is a pure function once that first decision is made. Collapsing them
into "the system decides entity identity deterministically" hides the only step in the chain
that can be wrong.

The original proposal — "set the subject to the git repo name" — is what you get when all
five collapse into one. It uses project identity to do the work of subject, entity identity
and scope simultaneously. Each is a different job, and the rest of this document is mostly
about giving each one its own mechanism.

A worked example of why the separation matters:

```
subject   = software:postgresql     ← canonical entity identity
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

## What is already built

Three pieces of this design exist. They are recorded here because each has now been proposed
at least once by somebody reading the code from the outside, and because one of them was
described wrongly in revision 2 of this document.

**Canonical entity identity is already a first-class, indexed column — not metadata.**
Revision 2 said the resolved identity is "stamped into `Claim.meta`", and a reader
reasonably concluded that core reconciliation semantics were being reverse-engineered from a
metadata dictionary. That is not what happens. `claims.subject_key` and `claims.object_key`
are `NOT NULL` columns in `memvara/store/sqlite.py:205`, indexed as `cl_subj` and `cl_obj`
for traversal, written on every insert, and populated for existing rows by the schema
version 6 migration whose comment reads "claims became traversable". `Claim.meta` carries the
write-time *stamp* that pins a claim to the identity it was written with, so an alias learned
in month six cannot re-key month one; the column is the derived, queryable identity.
`Claim.subject_key` and `Claim.object_key` in `memvara/types.py:722` read the stamp and fall
back to folding the surface.

So "make entity identity a first-class field" is done. What is genuinely missing from the
representation is narrower and is section 3: the object has no **kind**, so nothing
distinguishes `17` from `postgresql`, and the key has no **type**, so nothing distinguishes
the company from the software.

**`split_entity()` at `memvara/write/reconcile.py:1146`** already separates one surface form
into two identities either side of an instant, re-stamps retired claims as well as live ones,
undoes a closure that crossed the boundary, and writes a dated `ENTITY_REKEY` record so
`why()` can explain why history changed.

**`Store.adjacent()` already takes a `predicates` filter**, and `_WALKABLE` in
`memvara/store/sqlite.py:653` already gates traversal on `polarity > 0`, both keys non-empty,
and subject differing from object. A typed walk needs the declaration, not the plumbing.

### What has been built since

Steps 2, 3 and 4a of the sequence are implemented in pull request #200. Sections 1, 3, 4
and 5 below describe them as proposed, which is what they were when this document was
written; read those as the specification that pull request implements rather than as work
still outstanding.

- **Step 2, the predicate schema fields.** `subject_type`, `object_type`, `graph`,
  `inverse`, `inverse_cardinality` and `traversal_cost` on `PredicateSpec`, with
  `objects_are_entities`, `carries_edge`, and the persistence that keeps a declaration
  alive across a restart.
- **Step 3, the benchmark vocabulary.** `bench/packs/twowiki.toml` declares all 34 of
  2WikiMultihopQA's relations, and `bench/predicate_audit.py` measures what a vocabulary
  covers. Without the pack that corpus goes from 40.6% joinable to zero; with it, 68.3% of
  its triples can carry an edge and the remaining third is four date relations that are
  values on purpose.
- **Step 3b, the shipped vocabulary.** The three packs under `memvara/packs/` declare
  their object types, and eight predicates across `engineering` and `events` declare
  `graph`. Until they did, step 4a's rule had nothing to act on in any deployment that was
  not running the benchmark pack: every claim a shipped pack wrote was classified a value
  and no store had a graph. The rule was right and the vocabulary was silent.
- **Step 4a, the object kind.** `Claim.object_kind` is `ENTITY`, `VALUE` or `None`, decided
  at write time, persisted, and gating both the SQL and the Python side of traversal.
  `None` means the claim predates the rule and keeps its edges, which is decision C below.

**A claim carries an edge only when both halves are declared.** `carries_edge` is
`objects_are_entities and graph`, because section 5's point is that an entity-valued object
is not automatically a useful edge. An earlier revision of the implementation gated on the
object type alone, which made `graph` a field nothing read: a predicate declaring entity
objects with `graph` left false was walked and counted as joinable anyway.

### Decision C — a claim written before the object-kind rule keeps its edges

Taken 2026-09-08, after the six below. A migration cannot classify existing claims: the
kind comes from the predicate's declared `object_type`, and which vocabulary a deployment
loads is environment rather than data, so two machines opening one file would write
different answers into it. Existing claims therefore record nothing, both gates admit
"nothing recorded", and a store keeps the edges it had. Claims written afterwards follow
the rule, so a store converges as it is rewritten rather than in one pass.

Rejected: treating old claims as values, which switches off a working graph on upgrade and
can only be undone by writing a vocabulary and rewriting every claim; treating them as
entities, which keeps the false connections alive indefinitely in the rows nobody revisits;
and backfilling from whatever vocabulary is loaded, which breaks the rule that a migration
is a pure function of the row.

## The store today

`memory_stats` against tenant `prj_3c04449a3d9947f7b9bbbafb1d51d052` on 2026-09-08 reported
1416 live claims and a **join rate of 0.6%** — 8 claims whose object is the subject of another
claim. The live-claim count drifts, because the store is written to continuously; a re-run the
same afternoon read 1441. The number that matters is the second one, and it does not drift:
**8 joinable, in both readings.** Connectivity is not slowly improving on its own.

`Memvara.connectivity()` at `memvara/core.py:3353` records the comparison the graph leg was
measured against: on a 40.6%-joinable corpus the leg gains 13 points on chained questions, and
on a 0.0%-joinable one it loses 1.6. At 0.6% the leg runs and returns nothing.

Three subject conventions are live at once, and none knows about the others.

| Convention | Written by | Example |
|---|---|---|
| `user` | the model, through `memory_remember` | `user prefers_prose plain` |
| the git remote's basename | the capture hook, via `project_subject()` at `plugin/hooks/lib/extract.py:267` | `memvara-cloud deploys_to fly.io` |
| `project:<absolute path>` | the model, following the note at `plugin/hooks/recall.py:618` | `project:/Applications/workstation/agent-memory uses postgres` |

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

Identity is a bare string today, so `postgres` may be software, a company, a service instance
or a database; `python` may be a language, a package or a project; `memvara` is simultaneously
a company, a repository, a product and a Python package. A graph that walks
`memvara depends_on postgres` into `postgres owned_by …` cannot tell whether those two
`postgres` are the same thing.

Proposed shape — a namespace on the stored key:

```
project:github.com/memvara/memvara
software:postgresql
model:voyage-3
company:memvara
person:<id>
benchmark:longmemeval
```

The namespace is internal. Recall results should keep rendering the display form, because
`software:postgresql` in every line of every answer is a tax on the reader for a distinction
they already make.

**There is no `value:` namespace.** A scalar is not an entity with a funny prefix, and giving
it one would invite exactly the promotion section 3 forbids. Values are a different kind of
object, not a different kind of entity — see section 3.

**Typing is what makes the existing repair tools sufficient.** `split_entity()` separates an
identity along **time**, which repairs "one name, two people, one after the other". Two
entities that coexist — Apple the company and Apple the record label, both current — cannot be
separated by it, because there is no instant at which one became the other. That case needs
types, and it is the strongest argument for adding them.

### Normalization is not entity resolution

Two operations with different risk profiles, which should never run together.

**Normalization** is deterministic, cheap and low-risk: case, Unicode NFKD, combining marks,
apostrophes, punctuation, leading articles, `.git` suffixes, URL syntax. `entity_key()` at
`memvara/entities.py:140` is exactly this, and it is already correct in the ways that matter —
it never folds on embedding similarity, only on an exact key match or an explicitly recorded
alias.

**Entity resolution** is semantic and needs evidence: `Postgres` → `postgresql`, `FB` → Meta
Platforms, `Big Blue` → IBM. Here that is `EntityRegistry.acquire()`, which asks a model once
per `(owner, fold)` ever, refuses a canonical it cannot look up, and records the result as an
explicit alias.

The separation exists architecturally. Two things are still wrong with it. It is not **named**
as a separation anywhere, so a future change can quietly move work across the line. And the
one genuinely risky transformation sits on the wrong side: `_LEGAL_FORMS` stripping folds
`Apple Inc.` and `Apple` together deterministically, with no evidence and no confidence
attached. Under typing that fold is safe within `company:` and unsafe across types, which is
the rule that should govern it.

The exposure is narrower than it first looks and it is worth being exact, because a general
warning about `entity_key()` would be wrong. `Meta` and `Meta Platforms` do **not** fold —
`platforms` is not a legal form. `Apple Records` and `Apple` do not fold. `Apple Inc.` and
`Apple` do.

### Uncertainty resolves to a new entity

**Never silently choose an existing entity because its surface form looks similar.** This is
already the codified philosophy — `memvara/entities.py` states that when uncertain, a surface
form is a new entity, because wrongly merging two entities destroys a distinction permanently
while leaving them apart costs a duplicate slot a later alias can still fix.

Stated as the rule this design needs:

```
exact key match or recorded alias   → that entity
model resolution above threshold    → that entity, with resolution confidence recorded
anything else                       → a novel entity of the proposed type
```

## 2. Subject selection is not entity resolution

The plugin cannot know whether "we use Voyage for embeddings" names `model:voyage-3`,
`company:voyage` or `service:voyage-api`. That is semantic interpretation and only the model
can do it.

**The system owns entity identity. The model proposes entity references. Deterministic
canonicalization and reconciliation resolve those references into identities.**

```
   client                 plugin                  server
   ------                 ------                  ------
   semantic claim  ──▶  project identity  ──▶  subject resolver
                        (deterministic)              │
                                                     ▼
                                             entity registry
                                       canonical id + type + provenance
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

1. **Facts about the person.** Subject `user`. Unchanged. `procedural` claims live here
   and nowhere else, apart from verbatim notes on the `note` predicate: the write path
   files a `procedural` claim about any other subject as `semantic` and says so on the
   receipt (`Reconciler.file_by_subject`).
2. **Facts about the project.** Subject is the canonical project identity, supplied by the
   plugin and never typed by the model.
3. **Facts about a third thing.** The model proposes a reference; the resolver commits an
   identity. This is the class that produces joins, and no rule about repository names
   touches it.

### The candidate boundary

A model should propose rather than commit, which needs a named object rather than more
arguments on the MCP tool:

```
EntityCandidate:
    surface       "Voyage"
    type          "model" | "company" | …  (proposed)
    confidence    float
    source        model | plugin | user
    context       the episode id it was read from
```

with the pipeline `ClaimCandidate → EntityCandidate → EntityResolver → ResolvedEntity → Claim`.
Keeping this a separate object is what stops entity-resolution semantics leaking into the
tool schema, where every client would have to reimplement them.

### Precedence when a claim could be two classes

"We prefer Postgres for Memvara projects" is legitimately either
`user prefers software:postgresql` or `project:… uses software:postgresql`.

**Rule: the subject is the semantic owner of the proposition, not the environment the sentence
was typed in.** A preference belongs to whoever holds it; a dependency belongs to whatever has
it. Where a sentence carries both, it is two claims.

### Entity confidence is not claim confidence

These are independent and must not share a field:

```
claim.confidence              0.95   "memvara uses Postgres" is well attested
entity_resolution.confidence  0.62   that "Postgres" means software:postgresql is a guess
```

A claim can be entirely trustworthy while the entity interpretation is uncertain, and the
reverse. `memory_remember`'s existing `confidence` governs the first and already decides what
a write may displace; the second is a property of the resolution and belongs with the
provenance in section 8.

## 3. Object resolution, and the kind of an object

A join exists only when `object_key(A) == subject_key(B)`, so object canonicalization matters
at least as much as subject canonicalization.

**The missing piece in the current representation is the object's kind.** `object_key` is
computed for every object, so `17` folds to `17`, which is non-empty and therefore passes the
`_WALKABLE` gate. Nothing today distinguishes a scalar from an entity reference.

Proposed logical shape:

```
Object {
    kind: ENTITY
    entity_id: software:postgresql
    surface: "Postgres"
}

Object {
    kind: VALUE
    value: "17"
    surface: "17"
}
```

The surface text is kept for display and provenance either way. Only `kind: ENTITY` objects
get an `object_key` that participates in traversal.

The resolution pipeline, in order:

```
raw object text
   → predicate declares the expected object type
   → classify: ENTITY or VALUE
   → normalize (deterministic)
   → resolve against the entity registry (evidence-bearing)
   → canonical object identity
```

**A value must never be promoted to an entity merely because its string matches some
subject.** Without that rule every claim whose object is `17` joins to every claim about the
number 17, and the join rate rises while retrieval gets worse.

## 4. Predicate schema

`PredicateSpec` at `memvara/schema.py:77` declares `name`, `cardinality`, `volatility`,
`memory_type`, `aliases`, `supersedes` and `learned`. It declares nothing about types or the
graph. Proposed additions:

```toml
[predicate.depends_on]
subject_type  = ["project", "software", "service"]
object_type   = ["software", "service"]
cardinality   = "many"
graph         = true
traversal_cost = 1.0
inverse       = "depended_on_by"

[predicate.owned_by]
subject_type  = ["project", "software", "company"]
object_type   = ["company", "person"]
cardinality   = "one"
graph         = true
inverse       = "owns"          # note: the inverse is MANY

[predicate.prefers]
subject_type  = ["user", "agent", "project", "team"]
object_type   = ["entity", "value"]
cardinality   = "many"
graph         = false
```

Three things worth stating explicitly.

**Inverse cardinality is not symmetric.** `owned_by` is `ONE` and `owns` is `MANY`. So
`inverse` cannot be only a name: the schema has to understand both sides' cardinality, or the
first traversal that treats an inverse as a slot will end true facts. Where only one side is
declared, the other's cardinality must be declared with it.

**`traversal_cost` leaves room for edge strength.** `project depends_on software` is a strong
edge; a hypothetical `project mentioned software` is a technically valid edge that should
barely move a ranking. Nothing needs to implement weighting now, but a vocabulary that cannot
express it will have to be rewritten when ranking arrives. `rank_paths()` in
`memvara/retrieve/spread.py` is where it would be consumed.

**The `prefers` entry is an example, not a constraint being proposed.** Preferences are held
by users, agents, teams and projects alike, and a vocabulary that pinned `subject_type` to
`user` would refuse most of them.

Note that `supersedes` already exists for cross-predicate supersession — asserting
`unemployed` retires `works_at`. A constant-subject scheme would widen the blast radius of
every such rule to the whole repository, which is a further reason not to adopt one.

## 5. Graph edge semantics

**An entity-valued object is not automatically a useful edge.** That assumption is what makes
join rate a gameable metric. `graph = true` on the predicate is what makes an edge traversable.

**Inverse predicates are declared, not stored.** Writing the inverse claim would double the
store and, worse, would give the inverse its own independent temporal and supersession
behaviour — a correctness bug waiting to happen rather than a convenience.

## 6. Scope is not subject

**Subject answers "what is this about". Scope answers "where does this apply".** Overloading
subject to provide isolation is the most likely way to introduce reconciliation bugs while
fixing connectivity.

Scope today is the four-part `Scope` in `memvara/types.py`, bound at startup and not settable
per call (`docs/claude/mcp-server.md`). There is no project dimension, and the plugin's
`plugin/mcp.json` declares an HTTP server with no environment block at all, so there is
currently no channel to carry one.

Three architectures, with their consequences:

**Option A — project is only an entity.** Subject is `software:postgresql`; project
relationships exist as ordinary claims (`memvara-cloud uses postgresql`). Gives cross-project
graph connectivity. Gives no isolation: a fact from one repository surfaces in another.

**Option B — project is a scope dimension.** `scope.project = memvara-cloud`. Gives isolation.
Fragments the entity graph, because the same entity in two projects is two disconnected
neighbourhoods.

**Option C — separate visibility from traversal.** Two policies rather than one dimension:

```
visibility_scope   what ordinary recall returns          → current project by default
graph_scope        what a traversal may walk through     → current project plus explicitly
                                                            related projects
```

**Decision 4 chose Option C.** It is the better product architecture because it gives isolated default recall
without discarding the global entity graph, and because the relation that licenses a crossing
(`fork_of`, `belongs_to`, `depends_on`) is then itself a claim somebody can inspect. It is also
the most work, and it needs a project channel the hosted plugin does not currently have.

## 7. Project identity

Normalise the origin remote to `host/owner/repo`, with `.git` and any credentials stripped:

```
git@github.com:memvara/memvara-cloud.git     ->  github.com/memvara/memvara-cloud
https://github.com/memvara/memvara-cloud.git ->  github.com/memvara/memvara-cloud
https://user:token@github.com/memvara/x      ->  github.com/memvara/x
```

The normaliser must handle SSH aliases from `~/.ssh/config`, self-hosted GitHub Enterprise,
GitLab and Bitbucket, trailing slashes, and URL-encoded path segments.

**Case folding is host-specific, not blanket.** The host component always folds, because DNS
is case-insensitive. The owner and repository components fold only for hosts whose semantics
are known to be case-insensitive — github.com and gitlab.com among them. For an unknown or
self-hosted forge, case is preserved. Blanket lowercasing would knowingly manufacture a
project collision, which sits badly beside the invariant in section 12 that project identity
collisions are zero.

`github.com/acme/foo` and `gitlab.com/acme/foo` are **different projects**. Nothing should try
to detect that they might be the same.

Fall back to the toplevel directory name when there is no remote, and mark that identity
provisional so it can be re-keyed if the repository is later pushed. Never use the working
directory: a worktree is a different path for the same project.

### Forks, mirrors and renames

**Distinct project identities unless an explicit relationship is recorded.**
`github.com/memvara/memvara` and `github.com/inderjeet/memvara` do not share project facts by
default. The relationship, when it exists, is a claim:

```
project:github.com/inderjeet/memvara  fork_of       project:github.com/memvara/memvara
project:github.com/memvara/memvara    renamed_from  project:github.com/memvara/agent-memory
```

which is itself a graph edge, and a cleaner place for the semantics than inside the identity
function.

### Monorepos

A repository is not always the right granularity. `github.com/acme/platform` holding
`/services/api`, `/services/web` and `/packages/sdk` has one project identity and at least
three things somebody would want to record facts about separately.

**The path does not go into the project identity.** That is the mistake
`project:<absolute path>` already makes. Components are their own typed entities, related by
a claim:

```
component:acme-platform-api   belongs_to  project:github.com/acme/platform
component:acme-platform-web   belongs_to  project:github.com/acme/platform
```

which keeps repository identity, logical component identity and filesystem location as three
separate things. How a component identity is derived — declared in a config file, inferred
from a workspace manifest, or named by the user — is not settled here.

## 8. Identity provenance

Because this design moves some decisions away from the model, it has to be possible to tell
afterwards who decided what:

```
subject                = project:github.com/memvara/memvara
subject_source         = plugin | model | user
subject_resolution     = deterministic | alias | registry | novel
resolution_confidence  = <float, absent when deterministic>
```

`EntityResolution` at `memvara/entities.py` already returns `method` — `empty`, `alias`,
`known` or `novel` — and `EntityRegistry.methods` already counts resolutions by method. Most of
this is computed and discarded rather than persisted.

## 9. Merge, split and disambiguation

These are three different operations and revision 2 conflated the last two.

```
raw surface form
   → normalized key                    entity_key(), pure function
   → alias                             EntityRegistry.learn_alias()
   → merge applied to history          backfill_entities(), dated, dry-run by default
   → temporal split                    split_entity(), dated, dry-run by default
   → typed disambiguation              DOES NOT EXIST
```

**Temporal split** repairs one name that was two things in sequence: two people called John
Smith, one after the other. There is a boundary instant, and `split_entity()` takes it.

**Typed disambiguation** repairs one name that is two things at once: Apple the company and
Apple the record label, both current. There is no boundary instant, so it is not a split, and
calling it one would produce an API that lies about what it does. Two names for two
operations:

```
split_entity_temporally(surface, at)          exists
disambiguate_entity_by_type(surface, types)   does not
```

### Alias safety

An alias is not always harmless. If `software:postgres` and `software:postgresql` both already
hold live claims and someone then learns that they are one thing, the merge can create
contradictions that never existed.

The current design is safe here, and it is worth being precise about why, because the safety
comes from a deliberate refusal rather than from the alias being harmless. `learn_alias()`
does **not** re-key existing claims: a claim keeps the identity it was written with, so
`postgres version 16` and `postgresql version 17` stay in different slots and do not
supersede each other. The flip side is that the merge has no retroactive effect until somebody
runs `backfill_entities`, which is the named, dated, dry-run-by-default operation for exactly
that.

The invariant to state:

> **An alias never silently collapses two identities that both hold live claims. Making a
> merge retroactive is a separate, explicit operation, and it must report the supersessions it
> created or destroyed — not only the rows it touched.**

## 10. Server-side invariants

The tool description is **guidance, not an invariant**, and the distinction has to be explicit
or the store is back in the same state in six months with the variance generated by different
clients instead of by one.

**Guidance** — the tool description and packaged skill: use the supplied project identity for
project facts, `user` for facts about the person, and the thing itself for anything else.

**Enforcement** — the server, on write, for every client including ones this repository does
not ship:

- A project subject that is an absolute filesystem path is refused.
- A project subject not matching the canonical `host/owner/repo` shape is refused.
- An entity identity is normalized server-side; a client cannot store an unnormalized key.
- A predicate whose declared `subject_type` or `object_type` the claim violates is refused.
- An alias may not merge across entity types.
- A `kind: VALUE` object is never promoted to an entity by string match alone.
- A graph edge never crosses a tenant boundary, and crosses a scope boundary only where the
  traversal policy explicitly permits it.
- No client may backdate the transaction clock. This one already holds — invariant 8 in
  `INTERNALS.md` — and is listed because it is the model the others should follow.

## 11. Migration, compatibility and rollout

This design touches `Claim`, `EntityRegistry`, `Reconciler`, `fact_key`, both store backends,
`GraphTraverser`, `PredicateSpec`, the MCP tools and the packaged skill. The skill tree is
vendored into **seven downstream plugin repositories that pin it by sha**, so this is not a
single-repository atomic migration and cannot be planned as one.

### Migration invariants

**Attach a resolved identity; do not rewrite the original surface.** A claim keeps the text it
was written with. This is already how `Reconciler._stamp` behaves and is what stops an alias
learned in month six re-keying month one.

- **Idempotent.** Running it twice leaves the same store as running it once.
- **Dry-run by default.** Already true of `backfill_entities` and `split_entity`.
- **Dated and attributable.** Each touched claim gets an `ENTITY_REKEY` record.
- **Ordered deterministically.** Newly-colliding live claims replay in `(recorded_at, id)`
  order so the supersession chain rebuilds identically. Already implemented.
- **Reports temporal damage.** Claims scanned, re-stamped, closures undone, and —
  the number that matters — **supersessions created or destroyed**. Re-keying subjects changes
  what contradicts what; a count of rows touched hides that.

### Compatibility

Each of these needs a defined behaviour before rollout starts:

- **Claims with no entity type.** Untyped keys already in `subject_key`/`object_key`. Do they
  read as a wildcard type, an `unknown:` namespace, or a type inferred lazily on next write?
- **Objects with no kind.** Every existing object. The `_WALKABLE` gate currently admits all
  of them, so a naive typing pass changes what the graph walks.
- **Old project subjects.** Both `memvara-cloud` and `project:<absolute path>`.
- **Predicates with no type declaration.** The existing default is `Cardinality.MANY` and no
  supersession, chosen because wrongly retiring a true fact is worse than keeping two. Type
  checking needs the same bias: an undeclared type permits, it does not refuse.
- **Old clients against a new server**, and **new clients against an old server.** The seven
  pinned repositories guarantee both will exist simultaneously.
- **Read and write behaviour during migration**, which is not the same question as behaviour
  after it.

The safe default throughout is that an absent declaration permits rather than refuses, and
that enforcement is switched on per deployment once its store has been migrated.

## 12. Success metrics

**Join rate is a diagnostic, not a target.** It is gameable: point every claim at a handful of
generic entities and the graph looks beautifully connected while retrieval gets worse.

Primary, and the only one that can authorise shipping:

- **Benchmark retrieval quality improves.** `docs/BENCHMARKS.md`, the existing corpora, the
  existing harness.

Secondary:

- **Graph-path precision** and **graph-path recall**, against benchmark labels.
- Useful join rate — joins on `graph = true` predicates — rises.
- Graph-assisted recall rises; graph-assisted precision does not regress.

Safety, each able to veto a release on its own:

- Entity collision rate falls. Two distinct real things folded into one identity.
- Cross-scope leakage is zero.
- Project identity collisions are zero.
- Subject migration is idempotent.
- Historical claim identity is preserved.

### Defining a false traversal

"False-traversal rate" needs an operational definition or nobody can implement it. The cleanest
one is a labelled benchmark rather than a separate metric:

```
gold relevant entities     entities a correct answer to this question must involve
gold relevant paths        traversals that legitimately connect them
gold irrelevant paths      traversals between entities that share a surface form
                           or a generic hub but are not related
```

Then a false traversal is a walk along a gold-irrelevant path, and the metric is ordinary
precision and recall over paths. That is more rigorous than a bespoke rate and reuses the
benchmark machinery that has to exist anyway.

## 13. Adversarial test cases

Each is a way for this design to be wrong that a join-rate number would not show.

**Identity folding.** `Memvara` / `memvara` / `MEMVARA` fold to one; `memvara` and
`memvara-cloud` stay two; `Apple Inc.` and `Apple Records` stay two; `Meta` and `Meta
Platforms` stay two unless explicitly aliased.

**Project identity.** Two worktrees of one repository; a fork; a rename; a mirror; a monorepo
with several components; a repository with no remote; the same name under two owners; the same
owner and name on two hosts; a self-hosted forge that distinguishes case.

**Entity resolution.** `Postgres` / `PostgreSQL` / `postgresql` / `PostgreSQL database`
resolve together only through a recorded alias, never by similarity.

**False joins.** `Apple` the company must not connect to `Apple` the record label. `17` as a
version must not connect to `17` as an age.

**Cross-project.** Project A uses postgresql 16 and project B uses postgresql 17. Neither
supersedes the other, and each project's question returns its own answer.

**Temporal.** A project used postgresql 16 in 2025 and uses 17 in 2026. This *is* a
supersession, and it must remain distinguishable from the cross-project case above. **Entity
identity must not become state identity** — the case where the two designs are most easily
confused, and where the bitemporal machinery is what keeps them apart.

**Alias merge.** Two entities with live claims are aliased together; the store must not
silently create a supersession.

## Sequenced changes

Nothing depends on anything later in the list.

An earlier draft of this list had step 1 writing a predicate pack for the benchmark corpora.
That was wrong: a pack cannot declare `object_type` or `graph` until `PredicateSpec` carries
those fields, so the pack depended on a later step. The order below fixes it.

1. **Baseline measurement and the harness.** Record connectivity before anything changes,
   and build the gold-path labelling and entity-collision counting that decision 6 needs.
   This depends on nothing and must come first, because otherwise several major pieces get
   built with no way to attribute an improvement or a regression to any of them.
2. **Predicate schema fields** — `subject_type`, `object_type`, `graph`, `inverse` with both
   sides' cardinality, and `traversal_cost` (section 4). Depends on nothing. Everything that
   declares anything needs these to exist first.
3. **A predicate pack for every benchmark corpus.** A prerequisite, not a nicety:
   `BUILTIN_PREDICATES` holds 23 personal-assistant predicates and none of 2Wiki's Wikidata
   relations — `director`, `mother`, `spouse`, `composer` are all undeclared — so under
   decision 3 every 2Wiki object becomes a VALUE and the corpus drops from 40.6% joinable to
   zero. Without the pack the primary regression test reports that the graph leg stopped
   working, correctly and for reasons unrelated to whether this design is good.
4. **Entity representation and the slot key** — the `object_kind` column and the
   `subject_type` / `object_type` columns on claims (decisions 1 and 2), with the `fact_key`
   rehash they imply. **This step also takes the `Scope` shape change from decision 4**, even
   though nothing uses the project element until item 12. `fact_key` is
   `content_hash(owner_key(scope), …)`, so the scope shape and the entity type feed the same
   hash; splitting them across two steps would rehash the whole table twice, which is the
   cost decision 4 exists to avoid. What lands later is the traversal policy and the
   configuration channel, not the key.
5. **Object classification** — predicate-declared, undeclared defaults to VALUE (decision 3).
6. **Entity resolution boundary** — `EntityCandidate`, resolution confidence, provenance
   (sections 2 and 8).
7. **Server-side invariants** (section 10), which items 2 to 6 make expressible.
8. **Canonical project identity**, with host-specific case rules (section 7).
9. **Tool description and packaged skill** (`memvara/server/tools.py:337`,
   `memvara/skills/memvara/SKILL.md`). Cheap and effective, but guidance only, and
   deliberately after the enforcement it complements.
10. **The adversarial corpus**, ingested through `add()` so classification and resolution
    actually run (decision 6).
11. **Shadow promotion** (decision 5), which needs items 1 and 10 before it means anything.
12. **Project scope behaviour** (decision 4) — the traversal policy and the configuration
    channel the hosted plugin lacks. The fifth `Scope` element itself lands at item 4, with
    the other half of the one rehash; this item is what starts using it.
13. **Migration and rollout** (section 11), including retiring `project:<absolute path>` and
    fixing the note at `plugin/hooks/recall.py:618`.
14. **Typed disambiguation** (section 9).

## The six decisions

Taken 2026-09-08. Each replaces a question that previously blocked implementation.

### 1. An `object_kind` column on `claims`

`ENTITY` or `VALUE`, stored the way `subject_key` and `object_key` were added at schema
version 6, with the same backfill shape. The `_WALKABLE` gate at
`memvara/store/sqlite.py:653` gains `AND object_kind = 'entity'`.

Rejected: deriving the kind from the predicate schema with no storage, because most predicates
in a live store are runtime-learned and the undeclared default would then be today's broken
behaviour; overloading `object_key = ''` for values, because `''` already means "a surface form
with no content" and the two would become indistinguishable; a structured `ObjectRef` in
`types.py`, because the blast radius reaches both backends, remote hydration, the tool schemas
and every caller that reads `claim.object` as a string. A column does not foreclose `ObjectRef`
later.

### 2. Separate `subject_type` and `object_type` columns

Entity identity becomes the `(type, key)` pair, and `fact_key_for()` takes the type as a
fourth input.

The evidence that decided it was measured rather than assumed: `entity_key("software:postgresql")`
returns `"software postgresql"`. A `type:` prefix does not survive the fold, because
`entity_key()` treats every non-alphanumeric character as a word boundary. Storing the type in
the key would therefore require making `entity_key()` structural, and that function's guarantee —
a pure total fold giving any novel entity a correct stable identity for free, with no model call —
is what the whole entity design's cost argument rests on.

Rejected: the prefix, for that reason; type held only in the entity registry, because traversal
joins on `claims.subject_key` and `claims.object_key` and so would still not see a type;
packing type and key into one column, because identity would have to be split on every read,
which drifts once a second backend implements it.

Both surviving options required a `fact_key` rehash over the whole table, so that cost did not
distinguish them. `backfill_entities` is the instrument.

#### Revised 2026-09-12: the type stays inside the key

This decision was implemented as written and the result was measured before it was committed.
Splitting the namespace out of the key makes the graph worse. `company:apple` and
`fruit:apple` both fold to `apple`, and traversal joins on the key, so a walk crosses from a
company to a fruit — the confusion this decision exists to prevent, made worse than the
behaviour it replaced, where the two folded to `company apple` and `fruit apple` and stayed
apart. Keeping them apart under the split would mean comparing a type at every join, which
changes `GraphTraverser`'s node identity from a key to a pair and changes the signature of
`Store.adjacent`, a published protocol method third-party stores implement.

What was built instead: `entities.typed_entity_key` folds the name and puts the folded
namespace back in front of it, so `company:apple` and `fruit:apple` are simply two keys and
nothing downstream changes. `subject_type` and `object_type` are still added as columns, but
they are a derived denormalization for indexing and enforcement rather than identity — each
is read back out of the key beside it, so the two cannot drift apart.

The reasoning above against a prefix does not survive contact with the code. It says a prefix
would require making `entity_key()` structural and would cost that function its guarantee.
Nothing here touches `entity_key()`: `typed_entity_key` is a thin layer above it, and
splitting a namespace off and putting it back is itself pure and total, so the guarantee is
intact. The objection to packing type and key into one column — that identity would have to
be split on every read and would drift across backends — is answered by the codebase itself:
`entity_id()` packs the owner into the identity string and `split_entity_id()` takes it apart,
in both backends, and has not drifted. The split here is exact rather than best-effort,
because `entity_key()` emits only alphanumerics and spaces, so a folded name never contains a
colon.

Two things the revision gains that the split did not offer. Stripping a corporate form —
named in section 1 as the one genuinely risky fold here — becomes safe within a namespace and
impossible across one, which is precisely the rule section 1 asks for. And an alias that would
merge two namespaces is refusable, because the registry can see both types in the keys it is
being asked to merge; under the split it would have had to be told them separately.

### 3. The predicate declares the object kind; undeclared defaults to VALUE

The destructive direction on this axis is joining. A value wrongly treated as an entity creates
false joins, which degrade retrieval invisibly; an entity wrongly treated as a value costs a
join and corrupts nothing, and a later declaration fixes it. This is the same bias already
written into the codebase, where an unknown predicate defaults to `Cardinality.MANY` because
wrongly retiring a true fact is worse than keeping two.

**Connectivity is therefore opt-in.** A store has no graph until somebody writes a vocabulary,
which is what moves the predicate pack onto the critical path — including for the benchmark
corpora, per the sequencing above.

Rejected: defaulting undeclared predicates to ENTITY, which reads "undeclared permits" literally
but permits the exact failure being designed against; a deterministic shape test as the fallback,
which correctly catches `17` and `2026-01-01` but misclassifies words that are not things —
`plain` as a prose style, `high` as an effort level — and can be added later once the benchmark
can show whether it helps; the model proposing the kind, because it would put a model call on
`memory_remember`, which today needs none and is documented as the write that cannot mis-parse.

### 4. Project identity enters `Scope`, and visibility is separate from traversal

`Scope` gains a fifth element. Default recall is project-local. A graph walk may cross into
projects reachable by an explicit relation — `fork_of`, `belongs_to`, or a declared sibling set —
so the crossing is licensed by an inspectable claim rather than a hidden rule.

Decided now although it is built late, because `fact_key` is `content_hash(owner_key(scope), …)`
and decision 2 already commits to one rehash. Adding a scope element afterwards would mean a
second full-table rehash.

Rejected: full isolation, which would wall memvara off from memvara-cloud although they are one
product; project as an ordinary typed subject with no scope change, which is cheapest and leaves
the leakage already measured in `plugin/hooks/recall.py`, where a turn approving a cleanup was
handed notes about pricing tiers and an unrelated project's zip layout.

### 5. Shadow promotion, gated on the benchmark

A learned predicate is promoted to traversable automatically, on deterministic signals — type
consistency across its objects, object diversity, object shape, and a volume floor — but only
into a shadow graph that the benchmark evaluates and live retrieval ignores. It becomes live
only if graph-path precision holds.

**One signal is excluded by name: "this predicate's objects match existing subject keys".** That
rule rewards string collision by construction, and evaluating it would mean measuring exactly
the accidental-collision signal decision 3 exists to suppress.

The reasoning that survives is that `graph = true` is an editorial judgement about whether
walking a relation helps answer questions, not a property of the data. `mentioned`, `discussed`
and `similar_to` look identical to `depends_on` on every measurable signal and all make
retrieval worse. Shadow promotion does not solve that; it moves the human from confirming
individual predicates to reading one benchmark result per release, which is a better place for
a human and needs no new user-facing surface.

Rejected: declaration only, which is correct but leaves the vocabulary to rot; operator-confirmed
promotion, because there is nowhere for a proposal to land — the candidates were a fifteenth MCP
tool, the session-start block, or an admin console that does not exist, and each costs attention
on a surface already carrying load; automatic promotion straight to live, which is the
manufacture-joins-while-retrieval-degrades failure that demoted join rate to a diagnostic.

Nothing here was forced by migration pressure. Adding or changing a promotion rule later costs
no rehash, so this decision can be revisited cheaply.

### 6. Two corpora, split by role

**2Wiki, plus a corpus predicate pack**, carries the primary retrieval metric and validates
decision 5's promotion signals. It ships evidence as `[subject, relation, object]` triples across
12,576 questions and roughly 31,000 triples, and `bench/twowiki.py` already separates the
transitive question types (`compositional`, `inference`) from the non-transitive ones
(`comparison`, `bridge_comparison`) rather than averaging them — which is a gold path label in
the sense the metric needs. Its breadth of relations, with the corpus itself saying which are
transitive, is what makes it the right instrument for checking whether promotion picks the
relations that carry chains.

**A purpose-built adversarial corpus, ingested through `add()`** rather than `remember()` so that
classification and resolution actually run, carries the safety metrics: typed collision, entity
collision, cross-project behaviour. 2Wiki cannot test these — its entities arrive
pre-disambiguated, so there is no Apple-company-versus-Apple-Records case in it, and it runs in
one shared scope by design.

**Four of the six safety criteria are assertions, not metrics**, and belong in the unit suite
where they run in seconds and fail loudly: cross-scope leakage is zero, project identity
collisions are zero, migration is idempotent, historical claim identity is preserved. Only
entity collision rate and graph-path precision need labelled data.

Rejected: the existing corpora alone, which cannot exercise object classification, typing or
scope, because LOCOMO and LongMemEval measure extraction (the offline extractor turns 5,882
LOCOMO turns into zero claims) and 2Wiki loads structured triples through `remember()`; the
adversarial corpus alone, which gives no retrieval-quality signal and validates a design only
against the failure set its own author wrote; a labelled snapshot of the live store, which has
no ground truth and cannot become a regression test that survives the store changing. The live
store remains useful as an unlabelled smoke test, since incompatible-type counts and leakage
assertions need no gold labels.

## What is still open

The decisions above settle the design. These are implementation questions that follow from
them and are not yet answered.

- **How a component identity is derived** for a monorepo — declared in a config file, inferred
  from a workspace manifest, or named by the user (section 7).
- **The confidence threshold** at which `EntityRegistry.acquire()`'s answer is accepted, and
  where resolution confidence is stored (sections 1 and 8).
- **The promotion thresholds** in decision 5 — the volume floor, the type-consistency fraction,
  the diversity ratio.
- **The configuration channel** that carries a project id to the hosted server, given that
  `plugin/mcp.json` declares an HTTP server with no environment block (sections 6 and 11).
- **The vocabulary itself**: which predicates ship, with which types, cardinalities and inverses.

## Read next

[The memory model](claude/memory-model.md) defines the claim, the slot and the two clocks.
[Consolidation, vocabularies and the graph](claude/consolidation-and-graph.md) covers the graph
leg and the predicate packs, and [BENCHMARKS.md](BENCHMARKS.md) carries the measured value of
the graph leg on both corpora.

Next: [consolidation, vocabularies and the graph](claude/consolidation-and-graph.md).
