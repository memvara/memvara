# Roadmap — phases 4 through 8, and how this makes money

Waves 1–3 are shipped and committed: 1,657 tests, 100% statement coverage, and a library
that does what the README says. What follows is everything between here and a product,
written in the order that actually de-risks it.

The organizing judgement: **engram's problem is not capability, it is credibility.**
Every comparative number in the README is self-authored, measured against a baseline we
also wrote. That single fact gates fundraising, adoption, and every monetization path
below. So it goes first, and nothing else is allowed to jump the queue.

---

## Phase 4 — Prove it

The only phase that changes engram's position rather than its surface area.

### 4a. Head-to-head against the real mem0 package

**Status: unblocked.** `pip install mem0ai` resolves (2.0.17) — PyPI is reachable from
this environment, so the README's standing caveat ("the comparison target is a
reimplementation of mem0's *documented architecture*, not the mem0 package") is a choice
now, not a constraint. It should stop being true.

- Install `mem0ai`, drive both systems from the same transcript and the same extraction
  oracle, and replace `bench/baseline.py`'s numbers with measured ones.
- Report the losses too. The local-compute row already says engram is ~3× slower; a real
  head-to-head will surface more, and publishing them is what makes the wins credible.
- If mem0 wins somewhere that matters, that is a finding, not a failure of the benchmark.

**Risk to name up front:** mem0's `add()` wants a real LLM. Wiring it to the same stub
oracle engram uses is fiddly and may not be faithful to how mem0 actually behaves. If the
two cannot be driven from an identical oracle, the honest move is to report both under a
real model and eat the API cost, not to hand-wave the difference.

### 4b. LOCOMO and LongMemEval

**Status: blocked on one thing — an API key.** No `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`
is set in this environment. The harness can be built now and run the moment a key exists.

These are the benchmarks the field actually cites. Without a number on at least one of
them, engram is a well-argued library with no evidence, and every claim below is a
promise. With one, the compliance pitch in the monetization section becomes sellable.

**Deliverable:** `bench/locomo.py` and `bench/longmemeval.py`, plus a results table in the
README that includes the configuration and the cost, so the run is reproducible.

---

## Phase 5 — Keep the promises the package already makes

Cheap, fast, and pure honesty debt. These are places where the repo currently claims
something it does not deliver. None of them are hard; all of them are embarrassing if a
first user finds them before we do.

| gap | what's wrong |
|---|---|
| `engram[openai]` extra | Declared in `pyproject.toml`. There is no `engram/llm/openai.py`. Installing the extra gets you a dependency and no adapter. |
| OpenAI / Voyage embedders | `engram[local-embed]` ships `LocalEmbedder`; the hosted embedders most users actually want are absent. |
| Python floor | `requires-python = ">=3.10"`. The suite has only ever run on 3.13. Either test 3.10–3.12 in CI or raise the floor to what is verified. |
| No CI | A library that enforces `fail_under = 100` and has never run on another machine is enforcing it on exactly one machine. |
| No `LICENSE` file | Apache-2.0 is declared in `pyproject.toml` and the README. The file does not exist. This is a real legal gap, not a formality. |
| No `CHANGELOG.md` | Version is `0.1.0` with three waves of unrecorded change. |

---

## Phase 6 — Deployment surface

Only worth building once Phase 4 says the thing is worth deploying.

- **REST layer.** The `http` extra is declared and reserved; FastAPI + uvicorn + pydantic
  are named and unused. This is the surface a non-Python shop integrates against.
- **Postgres / pgvector store.** The `Store` protocol exists precisely for this, and it is
  the hard prerequisite for anything hosted — SQLite does not multi-tenant across nodes.
  **This gates the entire cloud monetization path**, so it is the highest-leverage item
  in the phase.
- **Docker image**, so `docker run engram` is the evaluation path.
- **Framework adapters** — LangChain, LlamaIndex, CrewAI. Distribution, not engineering:
  these are where developers discover memory layers.

---

## Phase 7 — Governance

This is the differentiated product, and the phase the monetization section rests on. Note
that the deletion half is already built (`erase`, `purge`, per-claim and per-scope), which
is unusual and worth saying out loud.

- **PII detection and redaction hooks** on the write path, before text touches disk.
- **Encryption at rest** for claim text, episode text and embeddings — embeddings leak
  content under inversion, so encrypting the first two and not the third is theatre.
- **Tamper-evident audit log** — hash-chained, so "what did the agent know on March 3rd"
  is answerable *and* provably unedited. Bitemporality makes this nearly free; nothing
  else in the space can say that.
- **Retention policies** — automatic erasure on a schedule, per scope.
- **RBAC / SSO** on the server surfaces.

---

## Phase 8 — Release

- `CHANGELOG.md`, a version policy, and a `1.0` that means the `Store`, `Embedder` and
  `LLM` protocols are stable. All three, because all three are extension points a closed
  layer and third-party backends build against — an earlier version of this line said two
  and disagreed with `CHANGELOG.md`.
- **The PyPI name `engram` is already taken.** `pip download --no-deps engram` resolves
  today to an unrelated MIT-licensed rendering/vision library at `0.1.0a1`, so
  `pip install engram` currently installs someone else's package and `twine upload` under
  that name will be rejected. This has to be settled before anything else in this phase,
  and it is the same decision as the commercial brand — see the trademark note above,
  which this makes concrete rather than hypothetical. `docs/RELEASING.md` lists the
  options and their costs.
- **PyPI publish.** Outward-facing and effectively irreversible — a name, once taken and
  published against, cannot be quietly un-published. Requires an explicit decision.

---

# Monetization

## The honest starting position

Three things are true at once, and a plan that ignores any of them is wishful:

1. **The technical work is genuinely good and genuinely unproven.** 1,657 tests prove the
   code does what we said. They prove nothing about whether it beats mem0 on a task
   anyone cares about.
2. **The category is crowded and funded.** mem0, Zep, Letta, Cognee. mem0 in particular
   has funding, mindshare, and a hosted product already selling. Competing on "a nicer
   memory layer for your chatbot" is fighting on their ground with none of their assets.
3. **The migration cost from mem0 to engram is close to zero, and that is ours.** The
   compat shim plus the `history.db` importer means a mem0 user can evaluate engram in an
   afternoon without rewriting a call site or losing their history. That is a real wedge
   and it was expensive to build. It should be the centre of the go-to-market, not a
   footnote in the README.

## What is actually differentiated

Not the retrieval. Not the cost savings — those are a quantifiable nice-to-have that any
funded competitor can copy in a quarter.

**The bitemporal audit trail is the asset.** `why()`, `history()`, `as_of()` and
deterministic contradiction resolution answer a question no vector store can:

> *What did this agent believe on March 3rd, where did that belief come from, and what
> replaced it?*

That is not a developer-convenience feature. It is an **audit requirement** in every
industry that is currently too scared to deploy agents: healthcare, finance, insurance,
legal, and anything touching the EU AI Act's logging obligations or a GDPR Article 17
erasure request. Engram already answers the erasure half properly, which most of the
category does not — retirement that leaves the text on disk is the normal behaviour, and
it does not satisfy a deletion request.

So the strategy is not "better mem0". It is:

> **The memory layer you move to when someone starts asking what your agent knew.**

Migration is free (the shim), and the reason to migrate is a question the incumbent
cannot answer.

## Decided: Apache-2.0 core, everything around it proprietary

**This is settled, not a recommendation.** The core library stays Apache-2.0. Every
surface built around it — REST API, web UI, team dashboards, multi-tenant control plane,
governance — is closed and lives in a separate private repository that depends on
`engram` as a published package.

### Why not a protective license, given the core is permissive

Apache-2.0 permits our closed layer and everyone else's. A funded competitor can take the
core and ship the exact product we intend to sell, and the license will not stop them.
That risk is **accepted deliberately**, because the usual remedy is worse here.

AGPL plus a commercial dual license is what MongoDB and Elastic did, and it works for them
because they ship **servers** — the copyleft boundary is a socket. Engram is a **library**
that gets imported into someone's agent process, where AGPL arguably reaches the whole
application. In practice nobody `pip install`s an AGPL memory layer into a commercial
product. That would close the embedding path, and with it the migration wedge that makes
the mem0 shim the most commercially valuable thing built so far. Protection bought at the
cost of the adoption funnel is not protection.

BSL 1.1 protects better and is not OSI open source, which conflicts with the core being
genuinely open.

So the moat is the closed layer, the brand, and execution speed — not the license.

### The line

| open (Apache-2.0) | closed (private repo) |
|---|---|
| the library: store, retrieval, write path, bitemporal model | REST API and auth |
| MCP server — a thin adapter that drives adoption | web UI, team dashboards |
| **mem0 shim and importer** — the wedge; closing it removes the reason to migrate | multi-tenant control plane |
| `Recorder` protocol (the seam) | the dashboard consuming it |
| SQLite store | **Postgres / pgvector store** |
| `Store` / `Embedder` / `LLM` protocols | governance: PII, encryption, audit chain, RBAC |

Two of these are deliberate and worth defending. The **mem0 importer stays open** because
it is worthless without the core and is the only reason anyone switches. **Postgres goes
closed** because it is a clean commercial boundary: SQLite is genuinely sufficient for a
single node, and needing Postgres correlates almost exactly with willingness to pay.

### Protections that do the work the license doesn't

- **CLA on the open core**, in place before the first outside contribution. Without it,
  every external patch is a veto on ever relicensing.
- **Trademark on a distinct commercial brand.** Note that *"engram" is probably weak* —
  it is an established neuroscience term for a memory trace, which makes it descriptive of
  the product's own function and hard to register. Trademark, not license, is what stops
  someone selling "Engram Cloud". Pick the commercial name deliberately.
- **Governance and Postgres never enter the open repository.** Not "moved later" —
  never committed, because git history is public forever.

### Sequencing note

The repo split is **deferred until Phase 4 completes**. There is no point building
commercial scaffolding around a core with no external evidence behind it. The protocols
that the closed layer will consume (`Store`, `Embedder`, `LLM`, `Recorder`) already exist
and are already injectable, so nothing about waiting makes the split harder later.

## The model in detail

**Free, Apache-2.0, forever:** everything shipped today. The library, the MCP server, the
mem0 shim, the importer, hybrid retrieval, bitemporal storage, `why()`. Adoption is the
moat for an infrastructure library, and crippling the core to force upgrades is how you
lose to the thing that didn't.

**Commercial — Phase 7, sold per-deployment:** PII redaction, encryption at rest,
tamper-evident hash-chained audit export, retention policies, RBAC/SSO, and support with
an SLA. These are exactly the things a compliance officer signs off on and an individual
developer never wants.

**Hosted, usage-based — Phase 6 + Postgres:** the standard managed offering, priced per
memory stored and per query. This is the larger revenue line eventually and the *weaker*
strategic position, because it is where the funded competitors already are. It should
follow the governance tier, not lead it.

### Why not the alternatives

- **Open-core with a crippled library.** Kills the adoption that is the only asset a new
  entrant has, and invites a fork.
- **AGPL + commercial dual license.** Maximum capture, and it is not too late — there are
  no external contributors yet, so the relicense is clean. But AGPL on the *core* scares
  off exactly the enterprise legal departments this strategy targets. If dual licensing
  is wanted, put AGPL on the **governance layer** and leave the core Apache-2.0.
- **Charging a share of measured LLM savings.** Elegant, and unsellable: metering it
  credibly requires trust we have not earned, and it prices the product against a number
  the customer can dispute every month.

## Sequencing, and the one hard dependency

**Nothing here is sellable before Phase 4.** A compliance buyer's first question is "what
is this compared to", and "we benchmarked it against something we wrote" ends the meeting.
One LOCOMO or LongMemEval number, plus a real head-to-head against installed mem0, is the
minimum price of entry for every path above.

The order that follows from that:

1. **Phase 4** — evidence. Gates everything.
2. **Phase 5** — honesty debt. Cheap, do it in parallel.
3. **Phase 7 (governance)** before **Phase 6 (hosting)**. Inverts the usual order on
   purpose: it sells into the position we already hold, needs no infrastructure, and does
   not require beating a funded competitor at their own game.
4. **Phase 6** — hosting, once there is a reason to host.
5. **Phase 8** — release.

## The risk worth stating plainly

The compliance market is slow, procurement-heavy, and reference-driven. It rewards a
product with three named customers and punishes one with none. If a faster route to first
revenue matters more than the defensible position, the honest alternative is to chase
developer adoption first (Phase 6's adapters and hosting) and accept competing head-on
with better-funded incumbents.

That is a real trade, and it is a business decision rather than a technical one — it
belongs to whoever is funding the runway, not to the architecture.
