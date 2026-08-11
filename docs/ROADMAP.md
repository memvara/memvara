# Roadmap

Where the project is, what is deliberately not being built here, and how the open core and
the commercial layer relate. Kept honest about status: an item is `done` only when
something in the tree does it, `deferred` when it was considered and declined with a
reason, and `next` when it is actually queued.

**Status as of `v0.1.0`.** 2,329 tests, 100% statement coverage, mypy clean, CI on
3.10–3.13 across Linux, macOS and Windows. The library does what the README says. Phase 4
— the evidence phase that gated everything below — is **done**, which changes the shape of
this document: the organizing risk was credibility, and it is no longer that every
comparative number is self-authored.

What has *not* changed: nothing here has been scored end to end against a reader model.
See [What is still missing](#what-is-still-missing).

---

## Phase 4 — Prove it — **done**

The only phase that changed memvara's position rather than its surface area.

### 4a. Head-to-head against the real mem0 package — done

`bench/mem0_real.py` drives the actual `mem0ai` package (2.0.17), not a reimplementation:
same 105-turn transcript, same extraction oracle, same `HashingEmbedder`, Qdrant in
`:memory:`, fully offline, five runs each. The table is in the README.

Three things came out of it that were not the expected result, and all three are in the
README because they cut against us:

- **The original benchmark was wrong in memvara's favour.** Its oracle string-matched the
  whole prompt, and mem0's additive prompt embeds `last_k_messages`, so every earlier turn
  in the window was re-extracted and mem0 was measured under a firehose. It reported 6/10;
  the true figure is 9–10/10. The mechanism is documented in the harness.
- **A README claim about mem0 was corrected against the source.** `add()` costs *one* LLM
  call in 2.x, not two — the two-call description was 1.x. The contradiction problem it
  was cited for got larger rather than smaller, but the correction is stated rather than
  quietly dropped.
- **The call-count row is partly an ingestion-granularity choice.** 105 vs 2 becomes 126 vs
  17 at equal granularity, and that is now said next to the number.

### 4b. LOCOMO and LongMemEval — done for retrieval, not for accuracy

The blocker was recorded as "an API key", and that turned out to be the wrong framing.
Scoring *retrieval* — did the store surface the evidence the annotators marked? — needs no
model at all, so `bench/locomo.py` and `bench/longmemeval.py` run the full question sets
(1,531 and 500) offline, for nothing. Results are in the README, weak rows first.

Two findings worth keeping:

- **LongMemEval's `oracle` split cannot measure evidence retrieval.** Every haystack
  session in all 500 instances *is* an evidence session, so recall there is 99.2% by
  arithmetic. The harness now computes a `chance` baseline and warns above 50%.
- **Retrieval was not reproducible until that run.** `HybridRetriever` broke score ties on
  `claim.id`, a fresh `uuid4` per ingest, so two ingests of one corpus ranked differently.
  Ties now break on a content hash and three full runs are byte-identical.

**The half that is still open** is end-to-end judged accuracy, which does need a reader
model and does cost money: ~$17.50 for LOCOMO and ~$5 for LongMemEval on a frontier model,
or ~$2 for a stratified sample. The harness already reports a `none` / `memory` / `full`
triple when a reader is configured, because a memory score with no reader-only floor and no
whole-haystack ceiling beside it is uninterpretable. This is the single most valuable
remaining item in the repository.

---

## Phase 5 — Keep the promises the package already makes — **done, one item excepted**

| gap | status |
|---|---|
| `memvara[openai]` extra with no adapter | **done** — `memvara/llm/openai.py`, Chat Completions with `strict: true`, refusals handled explicitly |
| Python floor unverified | **done** — CI runs 3.10–3.13 on Linux, 3.13 on macOS and Windows |
| No CI | **done** — `.github/workflows/ci.yml`: matrix, a coverage job gated at 100%, a mypy job, and a no-extras job that imports every module |
| No `LICENSE` file | **done** — Apache-2.0 |
| No `CHANGELOG.md` | **done** |
| Hosted embedders (OpenAI, Voyage) | **deferred** — see below |

**Windows support** belongs on this list retroactively and was the largest single item in
it. It was listed in CI and had never run there; the first run reported 99 failures. 95 of
them were one missing function — `os.pread`/`os.pwrite` are POSIX-only, so every store with
a `.vecs` sidecar raised `AttributeError`. The other notable one was a date before 1970: the
timestamp clamp was hard-coded to the POSIX year-9999 limit, while Windows' CRT stops at
year 3001 *and* rejects negative timestamps, so the exact defect the clamp exists to
prevent — one accepted write permanently breaking every later read of its scope — was alive
at both ends. Both bounds are now probed from the C library. A suite at 100% coverage on
three platforms never saw it, which is the useful lesson.

---

## Phase 6 — Deployment surface — **split**

This phase is where the open/commercial line actually falls, so it no longer reads as one
list.

**Shipped in the open core:**

- **Docker image** — multi-stage `python:3.13-slim`, 292 MB unpacked / 63.2 MB pulled, of
  which 1.4 MB is memvara and the rest is the base image and numpy. Runs as uid 10001 with
  a read-only root filesystem and `--cap-drop=ALL`. `docs/DEPLOY.md` has the whole story,
  including why there is no `EXPOSE` and no `HEALTHCHECK`.
- **Framework adapters** — LangChain, LlamaIndex, CrewAI, and LangGraph. Each says in its
  own module docstring what it *loses*, because "works with LangChain" without that is a
  claim that quietly means four different things. LangGraph turned out to be the best fit
  of the four: `BaseStore` hands over the query text natively *and* `put(namespace, key,
  value)` supplies all three parts of a triple, so an item is stored as one claim per field
  and changing `city` retires exactly `city` — contradiction resolution surviving a foreign
  interface intact.
- **Multi-hop traversal** — `neighborhood()`, `paths_between()`, `Store.adjacent()`, and
  SQLite schema v6 with `subject_key`/`object_key` indexes. Not on the original roadmap at
  all, and it is the feature that made the "the store has been a graph all along" claim
  true rather than rhetorical. Every edge on a path is evaluated at one pinned `as_of`,
  which is the property a search-then-search loop cannot have.
- **Token accounting** — `WriteReceipt.tokens_in`/`tokens_out`, `LLM.Usage` with a
  caller-allocated accumulator, and the `write.tokens_in` / `write.tokens_out` /
  `write.extract_ms` series. `llm_calls` was the only cost signal and cannot be billed on:
  a one-line turn and a 40,000-token document are both one call.

**Moved to the commercial layer, and not planned here:**

- **REST API and auth.** The `http` extra stays declared and reserved in `pyproject.toml`
  and nothing in this repository will implement it.
- **Postgres / pgvector store.** This is the clean commercial boundary: SQLite is genuinely
  sufficient for a single node, and needing more than one node correlates almost exactly
  with willingness to pay. The `Store` protocol is public and documented precisely so this
  is implementable by anyone who wants to; the license permits it and so does the design.

Saying "planned" about either of those would be the dishonest version, and the README now
states the boundary in the same terms.

---

## Phase 7 — Governance — **the seams are open, the policy is not**

The deletion half was already built and remains open source, which is unusual in this
category and worth saying out loud: `erase()` and `purge()` are real, irreversible removal
of the claim, the FTS entries that store the tokens directly, the vectors (zeroed in place,
because an embedding leaks content under inversion), the entity rows that keep the first
spelling ever seen, and optionally the source turns. Both return per-table counts as
evidence. Retirement that leaves the text on disk is the normal behaviour in this category
and does not satisfy an erasure request.

| item | status |
|---|---|
| **Redaction seam** — one injectable hook, upstream of the hash, the store, the embedder and the model | **done, open** (`memvara/redact.py`) |
| **`Recorder` seam** — every silent failure mode has a live series | **done, open** (`memvara/telemetry.py`) |
| **`Store.erase_episode`** — the primitive a retention rule needs; reaches a turn no claim cites | **done, open** |
| PII ruleset, compliance mode, per-role policy, audit report | **commercial** |
| Tamper-evident hash-chained audit log | **commercial** |
| Retention policies on a schedule | **commercial** |
| RBAC / SSO | **commercial** |
| Encryption at rest | **deferred** — see below |

The dividing rule, stated once because it decides every future case: **a seam is worth
nothing to a competitor and everything to a deployment; a policy is the opposite.** A
library you have to fork in order to comply is worse than one that ships no policy at all,
so the extension point and one honest default live here, and the ruleset does not. The
built-in `PatternRedactor` says in its own docstring that it is not compliance-grade,
because it is a demonstration of the seam rather than a product.

---

## Phase 8 — Release — **done except the publish**

- **`CHANGELOG.md`** — done, and kept specific: "a backdated supersession left two live
  values for a single-valued predicate" is the entry someone searches for.
- **A version policy** — done, in `docs/RELEASING.md`. `0.x` means the protocols may change
  in a minor release; `1.0` will mean exactly one thing, that everything behind
  `Memvara(store=, embedder=, llm=)` is a contract we will not break in a minor version.
  **Open question before 1.0:** `Recorder` and `Redactor` are injectable extension points on
  the same terms and are not currently named in that promise. Either they are in it or the
  promise says why not; leaving it ambiguous is the one outcome to avoid, because a closed
  layer and a third-party backend both build against them.
- **The name is settled.** The project was `engram` until Phase 8 prep found that
  `pip install engram` already resolved to an unrelated MIT rendering/vision library — so
  the name was not merely unregistered, it was pointing at someone else's code, and
  `twine upload` would have been rejected. `engram` was a weak mark for a second reason: it
  is the standard neuroscience term for a memory trace, which makes it *descriptive* of the
  product's own function, the hardest class to register or defend. `memvara` is coined,
  means nothing in any language, and is therefore a **fanciful mark** — the strongest class.
- **The registry names are still takeable, and an organization is not a reservation.**
  `github.com/memvara` exists and PyPI and npm *organizations* are registered, but PyPI's
  project namespace is flat and is claimed only by the first upload or a PEP 541 request,
  and an npm org gives you `@memvara/*` while the bare `memvara` stays open. Both bare names
  verified 404 as of the last check. Publishing the repository is a public mention of a name
  that anyone can still take, and we already lost `engram` by assuming a name was ours — so
  **claim both registry names in the same sitting as the first public push.**
- **PyPI publish** — the one thing outstanding. Outward-facing and effectively
  irreversible; a name, once published against, cannot be quietly un-published. It requires
  an explicit decision by whoever owns the project, and `docs/RELEASING.md` deliberately
  stops at TestPyPI.
- **Community files** — `CONTRIBUTING.md`, `SECURITY.md` and issue templates are in place,
  and the README states the open-core boundary rather than leaving a reader to infer it
  from a pricing page.

---

## Deliberately deferred

Each of these was considered and declined for a reason. They are recorded here so they stop
reading as things that are coming.

**Encryption at rest.** SQLCipher works — measured at +43–48% on writes, search unchanged,
and FTS5 keeps working because page-level encryption sits *beneath* SQLite. It is not
shipped because the mmap-backed `.vecs` sidecar stays plaintext outside that boundary, and
a plaintext vector is a confirmation oracle: encoding a guess and taking the cosine against
that file returns exactly 1.0000 for the right text and 0.87 for a one-digit-different
phone number, so it is not merely confirmable, it is hill-climbable. Encrypting the text and
not the vectors would be theatre. Full-disk encryption is the honest answer for the open
core; a storage-layer answer belongs with the backend that has one.

**Database-enforced row-level security.** Scope isolation here is enforced in the query
layer — `Scope.sees` for reads, `Scope.ancestors()` in SQL for enumeration — and it fails
closed. SQLite has no row-level security to enforce it a second time, so defence in depth
at the database layer is not available in this repository at all, and pretending it is
queued would misrepresent where the boundary is. The query-layer rule is the one that has to
be right, which is why it is named in `SECURITY.md` as an in-scope surface.

**Scrubbing on-disk residue after erasure.** `erase()` and `purge()` delete rows and index
entries and zero the vector slots. They do not scrub the SQLite pages those rows occupied,
and the `-wal` may still hold them. `VACUUM` and `PRAGMA secure_delete` are the deployment's
levers and `docs/DEPLOY.md` says so. Doing it in the library would mean either a
`secure_delete` pragma that taxes every write in the store or a `VACUUM` that rewrites the
whole file inside what a caller thinks is a per-claim call — both are the deployment's
trade to make, not ours.

**A cross-encoder reranker.** This is the standard next move for the weak retrieval rows
(LOCOMO multi-hop at 36%, open-domain at 31%) and it is declined for a specific reason: a
reranker is a model, and the default configuration is "numpy and nothing else, offline, no
API key". A reranker that only works once you have installed something is not the default
path, and one wired in behind the default would quietly end the offline claim. The seam is
already cut for it — `Explanation.rerank_score` is reserved and left `None` so a future
reranker's absence stays distinguishable from a reranker that scored zero, and the mem0
shim refuses `rerank=True` with an explanation rather than silently ignoring it. If this
lands, it lands as an opt-in protocol implementation, like every other model in the tree.

**Hosted embedders (OpenAI, Voyage).** `memvara[local-embed]` ships `LocalEmbedder` and the
`Embedder` protocol is two members wide — `dim`, and `encode(texts) -> (n, dim)` — so a
hosted one is a small amount of code that mostly duplicates an SDK call. It has stayed
undone because nothing in the library needs it and every user who wants one can write it in
an afternoon. That is a weak reason and this is the item most likely to move back onto the
list; a contributed implementation would be accepted.

**An approximate vector index (HNSW/IVF).** Exact search over a scope is O(|scope| · d) and
the matmul is already BLAS, so this is the floor, and it is correct and fast to roughly a
million claims. Beating it trades recall for speed, which belongs behind the `Store`
protocol as a choice a deployment makes, not in the default path.

---

## What is still missing

Stated plainly, because a roadmap that only lists what is done is an advertisement.

1. **End-to-end answer quality has never been measured.** Every benchmark in the README
   isolates architecture from model quality on purpose, and none of them says whether an
   agent using memvara answers better than an agent using mem0. That is the number a
   skeptical reader wants and the one we do not have.
2. **No external user has run this in production.** 2,329 tests prove the code does what we
   said it does. They prove nothing about what happens on someone else's data.
3. **The English-centrism is measured, not fixed.** The salience gate and the fast extractor
   are English sentence forms; other scripts fall through to the model, which is correct
   behaviour and a real cost. `gate.drop` and `fast.miss` are tagged by script so the gap is
   visible rather than assumed — but visible is not closed.
4. **Entity resolution folds surface forms, it does not know the world.** `Stark` versus
   `Stark Industries` is genuinely ambiguous and is left that way.

---

# The commercial layer

## Why the split exists, and why the core is permissive

The core library is Apache-2.0 and stays that way. Every surface built around it — REST API
and auth, the Postgres/pgvector store, governance, the multi-tenant control plane, usage
metering, quotas, rate limiting and the hosted console — is closed and lives in a separate
private repository that depends on `memvara` as a published package. **The split has
happened**; it was sequenced to follow Phase 4 and Phase 4 is done.

The framing that keeps this honest, and the one the README uses: **the library is the
product and the commercial layer is the operations around it.** Nothing on the closed side
changes what a claim is, how a contradiction resolves, what `why()` returns or what
`search()` finds. That is not a marketing line, it is a constraint on what may be built
there — a paid layer that altered the semantics of the free one would make the free one
untrustworthy, which costs more than it could ever earn.

### Why not a protective license

Apache-2.0 permits our closed layer and everyone else's. A funded competitor can take the
core and ship the exact product we intend to sell, and the license will not stop them. That
risk is **accepted deliberately**, because the usual remedy is worse here.

AGPL plus a commercial dual license is what MongoDB and Elastic did, and it works for them
because they ship **servers** — the copyleft boundary is a socket. Memvara is a **library**
imported into someone's agent process, where AGPL arguably reaches the whole application. In
practice nobody `pip install`s an AGPL memory layer into a commercial product. That would
close the embedding path, and with it the migration wedge that makes the mem0 shim the most
commercially valuable thing built so far. Protection bought at the cost of the adoption
funnel is not protection.

BSL 1.1 protects better and is not OSI open source, which conflicts with the core being
genuinely open.

So the moat is the closed layer, the brand, and execution speed — not the license.

### The line

| open (Apache-2.0) | closed (private repo) |
|---|---|
| the library: store, retrieval, write path, bitemporal model | REST API and auth |
| MCP server — a thin adapter that drives adoption | multi-tenant control plane, hosted console |
| **mem0 shim and importer** — the wedge; closing it removes the reason to migrate | usage metering, quotas, rate limiting |
| **`erase()` / `purge()`** — real deletion; an Article 17 obligation is not an upsell | governance: policy, retention, audit chain, RBAC |
| `Recorder` and `Redactor` protocols (the seams) | the dashboard and the rulesets consuming them |
| SQLite store | **Postgres / pgvector store** |
| `Store` / `Embedder` / `LLM` protocols | — |

Four of these are deliberate and worth defending. The **mem0 importer stays open** because
it is worthless without the core and is the only reason anyone switches — putting it behind
a paywall would mean charging for the exit. **Erasure stays open** for the same class of
reason: a deletion guarantee sold separately is not a guarantee. **Postgres goes closed**
because it is a clean commercial boundary. And **the seams stay open while the policies do
not**, which is the rule stated in Phase 7.

### What is actually differentiated

Not the retrieval, and not the cost savings — those are a quantifiable nice-to-have that any
funded competitor can copy in a quarter.

**The bitemporal audit trail is the asset.** `why()`, `history()`, `as_of` and deterministic
contradiction resolution answer a question no vector store can:

> *What did this agent believe on March 3rd, where did that belief come from, and what
> replaced it?*

That is an **audit requirement** in every industry currently too scared to deploy agents:
healthcare, finance, insurance, legal, and anything touching the EU AI Act's logging
obligations or a GDPR Article 17 erasure request. So the strategy is not "better mem0":

> **The memory layer you move to when someone starts asking what your agent knew.**

Migration is free (the shim and the importer, both open), and the reason to migrate is a
question the incumbent cannot answer.

### Protections that do the work the license doesn't

- **CLA on the open core**, in place before the first outside contribution. Without it,
  every external patch is a veto on ever relicensing. `CONTRIBUTING.md` states the
  requirement and is honest about why.
- **Trademark.** `memvara` is a coined, fanciful mark — the strongest class — and that is
  the thing that stops someone selling a competing "Memvara Cloud", not the license. (An
  earlier version of this document said the opposite, calling `memvara` descriptive and weak.
  That was a leftover from the `engram` era and it was simply wrong: `engram` was the
  descriptive neuroscience term, and replacing it is most of why the rename happened.)
- **Nothing from the closed side ever enters this repository.** Not "moved later" — never
  committed. `git filter-repo` can remove a file from a public repository's tip; it cannot
  remove it from every clone, fork and archive that already fetched it. A commit-then-revert
  is a publication.

## The model

**Free, Apache-2.0, forever:** everything in this repository. Adoption is the moat for an
infrastructure library, and crippling the core to force upgrades is how you lose to the
thing that didn't.

**Commercial, per-deployment:** governance — PII policy, retention, tamper-evident audit
export, RBAC/SSO — and support with an SLA. These are exactly what a compliance officer
signs off on and an individual developer never wants.

**Hosted, usage-based:** the standard managed offering on the Postgres backend, priced per
memory stored and per query. The larger revenue line eventually and the *weaker* strategic
position, because it is where the funded competitors already are. It follows the governance
tier rather than leading it.

### Why not the alternatives

- **Open-core with a crippled library.** Kills the adoption that is the only asset a new
  entrant has, and invites a fork.
- **AGPL + commercial dual license.** Maximum capture, but AGPL on the *core* scares off
  exactly the enterprise legal departments this strategy targets. If dual licensing is
  wanted, put AGPL on the governance layer and leave the core Apache-2.0.
- **Charging a share of measured LLM savings.** Elegant, and unsellable: metering it
  credibly requires trust we have not earned, and it prices the product against a number the
  customer can dispute every month.

## The risk worth stating plainly

The compliance market is slow, procurement-heavy, and reference-driven. It rewards a product
with three named customers and punishes one with none. Phase 4 removed the "compared to
what?" objection, but it did not produce a customer, and the second item under
[What is still missing](#what-is-still-missing) is the one that matters commercially: nobody
outside this repository has run it on real data.

If a faster route to first revenue matters more than the defensible position, the honest
alternative is to chase developer adoption first — adapters, hosting, distribution — and
accept competing head-on with better-funded incumbents. That is a real trade, and it is a
business decision rather than a technical one: it belongs to whoever is funding the runway,
not to the architecture.
