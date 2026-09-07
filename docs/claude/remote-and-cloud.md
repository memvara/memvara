# The hosted client

`RemoteMemvara` is this library's own API served by a hosted deployment. It is not a `Store`
and not a subclass of `Memvara`. The engine runs on the server; this class turns a method
call into one `/v1` request and hydrates the response into the same dataclasses the local
engine returns, so calling code cannot tell which one it is holding.

The server side is memvara-cloud, a separate commercial repository. This repository ships
the client and nothing else. That boundary is deliberate and is written down in
[OPEN-CORE.md](../OPEN-CORE.md): the `cloud` extra adds a caller of somebody else's server,
gated behind a lazy `httpx` import so the core install stays as small as it always was.

## Where the code is

- Primary: `memvara/remote/api.py` — `RemoteMemvara` and `ScopedRemoteMemvara`. The module
  docstring is the authoritative list of what is absent and why.
- Async form: `memvara/remote/aio.py` — `AsyncRemoteMemvara`, `AsyncScopedRemoteMemvara`.
- Transport: `memvara/remote/client.py` — `HttpClient`, `AsyncHttpClient`, retry and backoff.
- Response parsing: `memvara/remote/hydrate.py` — one function per dataclass, so a wire body
  becomes a `Claim`, a `WriteReceipt`, an `Explanation` and so on.
- Credentials: `memvara/remote/creds.py` — `resolve()`, which reads `MEMVARA_API_KEY` or the
  credential file `memvara-mcp login` wrote.
- Errors: `memvara/remote/errors.py` — `RemoteError` and its subclasses `AuthError`,
  `ScopeError`, `NotFound`, `Conflict`, `QuotaExhausted`, `RateLimited`, `LegalHold`,
  `ReadOnly`, `InvalidRequest`, `ServerError`, plus `error_from_response()`.
- The other remote seam: `memvara/store/remote.py` — `RemoteStore`, a `Store` implementation
  over the same facade. It is never constructed by `MEMVARA_MODE=cloud`, and a test keeps it
  that way.
- Tests: `tests/test_remote_client.py`, `tests/test_remote_reads.py`,
  `tests/test_remote_writes.py`, `tests/test_remote_scope.py`, `tests/test_remote_errors.py`,
  `tests/test_remote_hydrate.py`, `tests/test_remote_creds.py`,
  `tests/test_remote_constructor.py`, `tests/test_remote_cloud_mode.py`,
  `tests/test_remote_aio.py`, `tests/test_store_remote.py`.
- Documentation: [OPEN-CORE.md](../OPEN-CORE.md) draws the line seam by seam.

## How the pieces fit

Constructing a `RemoteMemvara` performs no network call. It resolves a credential, builds a
connection pool, and stops. That is the same rule `Memvara.__init__` follows for its `llm=`
argument, and for the same reason: spending money or failing over a network as a side effect
of a constructor is not something a library does behind your back. The first request is the
first method call.

`MEMVARA_MODE=cloud` in `memvara/server/config.py` builds one of these instead of a local
`Memvara`, and the MCP server serves the same fourteen tools from it. Both classes satisfy
the `MemoryAPI` protocol in `memvara/server/memory_api.py`, which is what makes that
substitution safe to make.

## What the hosted client refuses, and why refusing is the point

**What is absent is absent, not raising.** `reembed()`, `pending_extraction()`,
`reextract()`, `reset()` and `merge_predicate()` have no endpoint, so they are not methods
here at all. There is no `advise_replacements=` argument either: replacement advice is the
deployment's to give, and this client renders it when a receipt carries it. A caller reaching
for one of these gets an `AttributeError` at the call site, and mypy catches it before that.
A method that raised would compile, ship, and fail in production.

The same rule decides which *arguments* exist. `recall()` takes no `with_ids` and no
`header`, because `POST /v1/recall` returns a rendered string and carries no ids at all.
`get_all()` takes no `memory_types`, because the endpoint has no such filter and FastAPI
drops an unknown query parameter in silence, so the caller would get an unfiltered page with
nothing saying the filter was ignored.

`budget` and `valid_at` are the two exceptions, and each is a refusal rather than an
omission. Both are in the signature so that `None` — what every current caller passes —
works, and a value raises. A budget silently ignored is an oversized prompt with no signal,
and a day silently ignored is a block about the present handed to a question about the past.

Two write divergences are real and documented rather than hidden. `consolidate()` returns a
job handle rather than per-operation counts, because the endpoint answers 202 before the pass
starts. There is no `prove_erased()`, because the erasure response already carries its
per-table counts as evidence.

## Invariants and assumptions

- **The credential decides the tenant.** `default_scope` holds a tenant for local
  bookkeeping and never sends it, because a `tenant` parameter the caller could set would be
  a request to be trusted about identity. Any other value is refused.
- **Narrowing is one-way.** `RemoteMemvara.scope()` returns a view with no way back out, and
  a narrowing that tried to widen is refused by the token's own authorization.
- **Redaction runs client-side.** The `redactor` argument rewrites text on its way out,
  deliberately in this process: redaction that happens after the text has left is not
  redaction.
- **An unmapped operation raises rather than approximating.** `RemoteStore` maps what the
  facade exposes and raises for the rest. A `put_claim` that quietly wrote through
  `POST /v1/facts` would reinterpret every field the caller set, and a `competing_claims`
  returning an empty list would make every write believe the slot was empty.
- **This repository does not implement the server.** That is a commercial boundary, not a
  backlog item, and it holds even with the `cloud` extra installed.

## Read next

The module docstring at the top of `memvara/remote/api.py` is the contract and is kept
current with the endpoints. [OPEN-CORE.md](../OPEN-CORE.md) explains which seam sits on which
side of the line and why converging `Store` with the `/v1` facade would be the wrong move.

Next: [consolidation and the graph](consolidation-and-graph.md).
