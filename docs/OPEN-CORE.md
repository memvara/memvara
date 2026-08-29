# Open core, and exactly where the line is

Everything in this repository is Apache-2.0, and it is the whole library: the bitemporal
store, deterministic contradiction resolution, hybrid retrieval, consolidation,
provenance, entity resolution, multi-hop traversal, the MCP server, the mem0 shim and
importer, the LangChain / LlamaIndex / CrewAI / LangGraph adapters, and the SQLite
backend. Nothing in it is gated, time-limited, keyed, or degraded into a demo. There is
no free tier here, because there is no tier — the library runs on numpy, and `import
memvara` has never made a network call or needed an account, in any configuration. That
has not changed and is not going to.

The `memvara-mcp` CLI is a separate thing from the library, and it is where "offline" now
needs a qualifier. Plain `pip install memvara` gives you exactly what it always has:
`memvara-mcp init` writes a local server configuration, and the server it configures opens
a file on disk and talks to nothing else. `pip install memvara[cloud]` adds an optional
`httpx` dependency and, with it, a second path: `memvara-mcp init` then defaults to
`memvara-mcp login`, a device-code flow against the hosted console
([app.memvara.dev](https://app.memvara.dev)) that mints an API key and writes it to `~/.memvara/credentials.json`.
That default is a CLI convenience, not a change to what the library needs — local and
self-hosted remain fully supported, explicitly, with `--mode local` on `init` or
`MEMVARA_MODE=local` in the server's environment, and neither requires the `cloud` extra
or ever touches the network.

A commercial product is built around it. It is not a better version of this one:

| in this repository, Apache-2.0 | commercial, separate product |
|---|---|
| SQLite store | Postgres / pgvector store |
| in-process library; MCP over stdio | REST API and auth |
| the `Redactor` and `Recorder` seams | governance: policy, retention, tamper-evident audit chain, RBAC |
| one process, one store | multi-tenant control plane |
| `WriteReceipt`, telemetry counters | usage metering, quotas, rate limiting, hosted console |

The right column is the commercial product's *scope*, not a shipping manifest — some of it
exists today and some is being built. The left column is what matters here, and the left
column is complete.

The pattern is that **the library is the product and the commercial layer is the
operations around it.** Nothing in the right column changes what a claim is, how a
contradiction resolves, what `why()` returns, or what `search()` finds — that is a
constraint on what may be built there, not a slogan, because a paid layer that altered the
semantics of the free one would make the free one untrustworthy. What is over there is what
you need when memory becomes several machines' problem and several people's. For one
application on one machine, nothing is missing.

The uncomfortable half, stated here rather than discovered three weeks in: if you need
Postgres or an HTTP endpoint, this repository does not implement one and is not scheduled
to grow one — that is a commercial boundary, not a backlog, and it holds even with the
`cloud` extra installed. What the `cloud` extra adds is a *client*: `memvara/store/remote.py`
speaks HTTP to a memvara-cloud deployment you do not have to run yourself, gated behind a
lazy `httpx` import so the core install stays as it always was. It is a thin caller of
someone else's server, not the server itself, and it changes nothing about what a claim
is or how `search()` ranks it — the same guarantee the table above makes. Saying
"planned" for the server side would be the dishonest version. The line is drawn there
because SQLite is genuinely sufficient for a single node, and needing more than one node
correlates closely with being able to pay for it. The storage half of that sits behind the
`Store` protocol in [`memvara/store/base.py`](../memvara/store/base.py), which is public,
documented, and implementable by anyone — a third-party Postgres backend is a legitimate
thing to write, and neither the license nor the design objects to one.

Two things stay open on purpose and are worth naming, because they are the ones a
commercial reading would have closed. **The mem0 shim and the `history.db` importer are
Apache-2.0** — they are the reason anyone can leave mem0 in an afternoon, and putting them
behind a paywall would mean charging for the exit. **`erase()` and `purge()` are
Apache-2.0** — real, irreversible deletion including the FTS tokens and the vectors. A
GDPR Article 17 obligation is not a feature to upsell.

The one thing to know before you count on "offline": with no `llm=`, `add()` runs the
deterministic fast path only and drops the turns its rules do not recognise — on a real
support transcript that turned out to be **all 64 of them**, leaving a store with episodes
and no claims. `remember()`, retrieval, contradiction resolution and everything else are
unaffected and need no model ever, and writing structured facts through `remember()` is
how an offline integration gets the whole bitemporal machine. See
[What the fast path does not catch](DESIGN.md#what-the-fast-path-does-not-catch-measured).

---

## The remote/local seam: a decision, not a gap

`MEMVARA_MODE=cloud` once built a `Memvara` over a `RemoteStore` and started a server. That
server listed twelve tools and raised `NotImplementedError` on the first one a model
reached for, because `RemoteStore` wires seven `Store` methods and the engine calls a
different set on every turn — `put_claim`, `add_episode`, `candidate_ids`,
`lexical_search`, `vector_search`, `competing_claims`, none of which the REST facade has
an endpoint for.

The decision is **diverge, and gate**, and it is written down here rather than left as a
TODO because "converge" is the option somebody will otherwise reach for and it is the
wrong one.

**Why not converge.** The two shapes are not two spellings of one interface.
`Store` is what the engine writes against and every method on it is one hop from the row:
a raw `Claim` to upsert, a `fact_key` to look up, a vector to compare, a tenant to page
over. `memvara_cloud.rest.app` is a facade: `POST /v1/facts`, `GET /v1/history`,
`POST /v1/erasures`, each doing its own reconciliation, each authorizing itself against
the bearer token's own scope rather than against a `tenant` argument the caller supplies.
Widening the REST surface until `Store` fits over it would move contradiction resolution,
ranking and scope enforcement to the client — which is the multi-tenant control plane's
job, on the other side of the line in the table above, and would make a browser session
and a Python process two places where the same guarantee has to hold.

**Which side each seam is on.**

| seam | side | why |
|---|---|---|
| `memvara/store/base.py` — the `Store` protocol | **open** | Public, documented, implementable by anyone. A third-party Postgres backend is a legitimate thing to write and the design does not object. |
| `memvara/core.py` — the engine over that protocol | **open** | It *is* the library. Nothing about running it changes when the rows live elsewhere. |
| `memvara/store/remote.py` — the HTTP client | **open**, and thin on purpose | A caller of someone else's server. It maps what the facade actually exposes and raises for the rest; a `put_claim` that quietly wrote through `POST /v1/facts` would reinterpret every field the caller set, and a `competing_claims` returning `[]` would make every write believe a slot was empty. Both are worse than an exception. |
| the `/v1` facade itself | **commercial** | Auth, multi-tenancy, quotas. Naming it "planned" here would be the dishonest version. |
| running the engine *against* a remote store | **neither, for now** | Never constructed. `MEMVARA_MODE=cloud` builds a client of the facade instead, and a test reads `config.py` to keep a `RemoteStore` from coming back. |
| `memvara/remote/api.py` — a client of the facade | **open** | `RemoteMemvara` is the library's own API served over `/v1`. It calls the facade the way any customer would, holds no engine, and is what `MEMVARA_MODE=cloud` now builds. |

**What happened to the refusal.** It is gone, and not because the gap closed — the row
above still says what it said. `build_memvara` used to compare `_ENGINE_NEEDS` against
`RemoteStore.WIRED` and refuse when anything was missing, which was correct for as long as
cloud mode meant *running the engine over a remote store*. It no longer means that.
`MEMVARA_MODE=cloud` builds a `RemoteMemvara`, a client of the facade, and the MCP tool
table is typed to `MemoryAPI` — a protocol that `ScopedMemvara` and `ScopedRemoteMemvara`
both satisfy. One tool table, two engines, and the engine is still never pointed at a
store that cannot serve it.

That set difference was built to empty out on its own when `RemoteStore.WIRED` grew, so
that nobody had to remember to lift a flag. Under this design that day does not arrive:
the `Store` seam is bypassed rather than completed. Leaving a dead gate in place would
tell the next reader it was still live, so it was deleted deliberately — and the test that
guarded it was **replaced rather than removed**, by
`tests/test_config_cloud.py::test_no_cloud_path_anywhere_constructs_an_engine_over_a_remote_store`,
which reads `config.py`'s syntax and fails if a `RemoteStore` import or a `store=` argument
comes back.

One refusal survives, for the one reason that still stops a cloud server starting: a
deployment is reached over HTTP, so `memvara-mcp init --mode cloud` refuses when `httpx`
is not importable and names `pip install "memvara[cloud]"`. `init` writes a config it
never launches, so the two commands have to gate on the same thing or the disagreement is
silent — that was true of `cloud_gap()` and it is true of this.

A hosted deployment can be reached two ways, and they are different products rather than
two spellings of one. Point an MCP client at the deployment's own URL and the client talks
to it directly. Or run `memvara-mcp` with `MEMVARA_MODE=cloud`, which gives a client that
speaks stdio one local process holding a credential — the same tool table, served over
`/v1`. Both are clients of the facade; neither is an engine straddling both sides, which
is the divergence this section exists to record.

---

