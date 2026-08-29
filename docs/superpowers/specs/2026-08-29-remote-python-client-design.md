# A Python client for a hosted deployment

**Date:** 2026-08-29. **Status:** design agreed, not implemented.
**Repositories:** `memvara` (this one) and `memvara-cloud`.

## What this builds

`Memvara(api_key=...)` returns an object that talks to a hosted memvara-cloud deployment
over its `/v1` REST API, with the engine running server-side. And `memvara-mcp --mode
cloud` starts, serving its fourteen MCP tools from that same client instead of refusing at
construction.

## Why it does not exist today

Three surfaces reach a hosted deployment: an MCP client, the REST API, and `npx memvara`
bridging stdio clients to the first. There is no Python client, and application code in
Python is the case none of the three serves.

`docs/ROADMAP.md` records this as decided rather than missed — *"Declined: a REST client
library. MCP already covers the agent case… A library would serve calling memvara from
ordinary application code, and nobody has asked for that."* The sole stated reason was
absence of demand. That reason no longer holds, so the entry moves rather than being
quietly contradicted.

The nearest existing thing is `memvara/store/remote.py`, and it is at the wrong seam. A
`Store` is what the engine calls, one hop from a row; `/v1` is a facade whose operations
each do their own reconciliation and authorize against a bearer token. `docs/OPEN-CORE.md`
settled that those two shapes must **diverge, not converge**, and the reasoning stands:
widening `/v1` until `Store` fits over it moves ranking, contradiction resolution and scope
enforcement to the client, and a public `put_claim` accepting `recorded_at` and
`invalidated_by` is a history-forging endpoint.

This design takes the other seam, which `OPEN-CORE.md` had already named as the answer:
*"the MCP/HTTP server layer becomes a client of the remote deployment's own `/v1`
facade."*

## Decisions

| decision | choice |
|---|---|
| entry point | `Memvara(api_key=...)` is a factory returning a `RemoteMemvara`; `Memvara.connect()` for ambient credentials |
| type honesty | a distinct class exposing only what `/v1` serves — a missing operation is absent, not raising |
| audience | a public, documented SDK: full coverage, typed errors, an async twin |
| cloud mode | un-refused in this same work |
| the `end` gap | the endpoint is built in `memvara-cloud`, not worked around |
| write retries | `Idempotency-Key` added to the facade so writes are safely retryable |
| `memory_standing` | routed through `GET /v1/standing` via one optional protocol member |

## 1. Construction and credentials

`Memvara.__new__` dispatches; `RemoteMemvara` deliberately does not subclass `Memvara`, so
Python skips `Memvara.__init__` entirely and no local engine is half-built.

```python
def __new__(cls, path=None, *, api_key=None, base_url=None, **kw):
    if api_key is None and base_url is None:
        return super().__new__(cls)
    return RemoteMemvara(api_key=api_key, base_url=base_url, **kw)
```

**Dispatch keys on the explicit argument and never on the environment.** This is the
property to defend in review. If a bare `Memvara()` could turn remote because
`MEMVARA_API_KEY` is exported in that shell, a script that has always written to a local
file would start posting to a hosted store, silently, on any machine where someone ran
`memvara-mcp login`. The environment supplies the *value* only after the caller has asked
for remote.

`Memvara.connect()` is the ambient-credential door, since without it the post-login case —
the one the CLI just set up — has no supported spelling and people reach for the private
resolver.

**Rejected at construction**, each a `TypeError` in the style of the existing `path`/`store`
guard: `api_key` with `path=` or `store=`; and `embedder=`, `llm=`, `registry=`, or the
`write_`/`read_`/`graph_` tuning, all of which are server-side and would otherwise be
accepted and ignored.

**Passed through:** `tenant`, `user`, `agent`, `session`, which become per-call narrowing
inside the credential's own scope as `rest/scope.py` already enforces. Also `redactor=`,
which matters more here than locally: it rewrites text before it leaves the process, and
that is where a privacy control belongs when the next hop is someone else's server.

**No network call at construction**, matching the reasoning `core.py` already gives for
`llm=`. The cost is that a bad key or an out-of-scope tenant surfaces on the first call as a
403.

Resolution order: explicit `api_key=`, then `MEMVARA_API_KEY`, then
`~/.memvara/credentials.json`; nothing found raises an error naming `memvara-mcp login`.
`base_url` defaults to `MEMVARA_SERVER_URL`, then `https://app.memvara.dev`. All four
constants already exist in `server/config.py` and `server/login.py` and are reused, which
means naming `config.py`'s resolution as a function if it is not one already.

## 2. The surface

**Present, same signature and return type:** `add`, `remember`, `supersede`, `recall`,
`search`, `ask`, `since`, `get`, `get_all`, `count`, `history`, `why`, `produced`,
`neighborhood`, `paths_between`, `forget`, `delete`, `end`, `erase`, `purge`, `stats`,
`connectivity`, `scope`, `close`.

**Absent, because `/v1` has no operation and faking one would lie:** `reembed` (vectors are
server-side), `pending_extraction` and `reextract` (the unextracted queue is
server-internal), `reset`.

**Divergent, and documented as such:** `consolidate()` returns a job handle, because
`/v1/maintenance/consolidate` is asynchronous. `prove_erased()` does not exist —
`/v1/erasures` returns per-table counts as evidence in the erasure response itself, so the
proof arrives with the act rather than on a second call.

**New, with no local twin:** `whoami()` and `health()`. Both are how a caller checks a
credential without performing a write.

## 3. Transport, hydration, errors, retries, async

New package `memvara/remote/`: `client.py`, `hydrate.py`, `api.py`, `aio.py`, `errors.py`.
Split because the transport and the mapping fail differently and are tested differently.

**One word, two uses, held to the split the existing code implies:** *remote* names the
client (`RemoteStore`, `RemoteMemvara`); *cloud* names the deployment and the mode
(`MEMVARA_MODE=cloud`, the `cloud` extra, `cloud_gap`).

### Hydration

`memvara_cloud/rest/render.py`'s sixteen functions are the specification, read backwards.
`hydrate.py` mirrors them and returns the library's own `Claim`, `Episode`, `Result`,
`Answer`, `Reading`, `WriteReceipt`, `Provenance`, `Delta`, `Path` and `Edge`. No parallel
types.

`render.memory()` is lossless against `Claim`'s persisted fields, with three details that
silently produce wrong objects if missed:

- `extractor`: the library spells "unrecorded" as `""`, JSON as `null`. Map it back or
  `Claim.extractor` is `None` where the type says `str`.
- `salience_base` and `last_observed` are top-level on the wire and `meta` keys in the
  library. Restore them under `SALIENCE_BASE` and `LAST_OBSERVED`.
- `state` and `links` are derived server-side and must be recomputed, never stored — a
  hydrated claim carrying a stale `state` would disagree with its own timestamps.

`meta["closure"]` survives: `RESERVED_META` strips only `salience_base`,
`last_observed_at`, `subject_entity`, `object_entity` and `entity_rekey`. So a hydrated
claim can still say whether it ended or was retired, and when. The three entity keys are
genuinely dropped, and every operation reading them is local-engine work already absent
from the class.

Hydration raises on a missing required field rather than defaulting. A renamed field should
break loudly on the first call, not return a claim with a plausible zero in it.

### Errors

`errors.py`, one class per envelope `code`, all under a `RemoteError` base carrying
`status_code`, `code`, `message` and `retryable`: `AuthError`, `ScopeError`, `NotFound`,
`Conflict`, `QuotaExhausted`, `RateLimited` (with `retry_after`), `LegalHold`, `ReadOnly`,
`InvalidRequest`, `ServerError`. An unrecognised code raises the base rather than being
coerced into its nearest neighbour.

### Retries

The envelope already carries `retryable` and 429 carries `Retry-After`, so the server has
done the classification. Reads retry on `retryable`, on connect-phase failures and on 429,
with exponential backoff, jitter, three attempts, honouring `Retry-After`.

Writes retry on the same conditions **once the facade supports `Idempotency-Key`** (§5).
Until it does, a write that times out after the request was sent may have committed, and
retrying it writes twice; that case raises and names the uncertainty. The client sends a key
on every write regardless, so the safe behaviour arrives with the server change and not with
a later client release.

**The guarantee is per-worker, and the client must not promise more.** The deployed
idempotency store lives in the serving process, so a retry that a load balancer routes to a
second worker finds no record of the first attempt and re-executes. That is not a defect to
work around here — it is the shape of what the server offers, and a client that documented
"retries are safe" without qualification would be making a promise the deployment does not
keep. `RemoteMemvara`'s retry docstring says what is actually true: a retried write is
deduplicated when it lands on the worker that saw the first attempt, and a single-worker
deployment is therefore the only one where it always is.

### Async

`AsyncRemoteMemvara` uses `httpx.AsyncClient` directly and does **not** follow `aio.py`'s
`asyncio.to_thread` pattern. `aio.py`'s own argument is why: that wrapper exists because
there is no async SQLite and colouring would propagate through the engine. Neither holds
here — the transport has a real async client and there is no engine below it. `aio.py`'s
docstring states the library-wide position and gains a sentence saying where it stops.

### One HTTP layer

`client.py` holds the lazy `httpx` import with the `memvara[cloud]` hint, the bearer
header, a 30-second default timeout, `raise_for_status()` before `.json()`, and the error
translation.

**`store/remote.py` shares the pool construction only, and that is a correction to this
paragraph's first draft.** It said the point was to stop two HTTP layers having "different
retry and error behaviour pointed at one API". That ambition contradicts a decision
`store/remote.py`'s own docstring already argues: errors there are *"not swallowed and not
translated into a `Store`-specific exception type: the protocol declares none, `SQLiteStore`
lets `sqlite3` errors propagate the same way"*. Delegating fully would raise `RemoteError`
where two pinned tests expect `httpx.HTTPStatusError`, and would require rewriting
`get_claim`.

So the shared piece is the bearer header, the base URL and the lazy import. `RemoteStore`
keeps its own `raise_for_status()` surface and gets **no retries and no typed errors**. That
leaves a real asymmetry between the two clients — a transient failure retried through
`RemoteMemvara` is not retried through `RemoteStore` — and it is now deliberate rather than
accidental. Unifying it is a separate decision, and it would mean overturning a documented
one.

## 4. Un-refusing cloud mode

`build_memvara(config)` under `mode == "cloud"` constructs a `RemoteMemvara` from
`config.server_url` and `config.api_key`, and drops `llm=`, `embedder=` and `registry=`. An
environment naming `MEMVARA_LLM` or `MEMVARA_EMBEDDER` under cloud mode is a `ConfigError`
naming them, not a silent ignore.

**The guard does not un-refuse itself, and the difference matters.** `cloud_gap()` is a set
difference built to empty out when `RemoteStore.WIRED` grows. That day never arrives under
this design, because the `Store` seam is bypassed rather than completed. The branch is
deleted deliberately and
`test_the_cloud_guard_is_derived_from_the_store_rather_than_hardcoded` is replaced, not left
to fail into deletion.

`OPEN-CORE.md`'s seam table row *"running the engine against a remote store — neither, for
now"* stays true and unchanged: this never does that. One sentence becomes false and is
rewritten — *"A hosted deployment is reached by pointing an MCP client at its own URL. It is
not proxied through a local server."*

`cloud_gap()`'s docstring records a live inconsistency, that `memvara-mcp init` writes a
cloud config and exits 0 while the server it configured refuses to start. This closes it.

`ToolContext.memory` is typed to a new `MemoryAPI` protocol covering the eighteen members
`tools.py` touches, satisfied by both `ScopedMemvara` and `ScopedRemoteMemvara` and kept
honest by a test that walks the protocol against both. Its docstring's security claim stays
true: scope is bound once at construction and no handler can address another tenant, which
the remote view holds the same way by sending a bound scope the credential must already
contain.

`MemoryAPI` also carries an optional `standing()`. `_standing` prefers it and otherwise
keeps today's `get_all(states=["live"])` path, so local behaviour is unchanged while cloud
mode filters server-side through `GET /v1/standing`. Without it, the tool a session calls at
startup would page every live memory in the scope across the network — and `MemoryPage`'s
own docstring warns it is *"not the call for a scope with millions of claims."* This is the
only change to `tools.py`.

## 5. Changes in `memvara-cloud`

Two features, one branch off `a427fbb`. That repository is at detached HEAD and
`memvara_cloud/control/store.py` has uncommitted changes belonging to someone else; leave
that file alone.

**`POST /v1/end`** — closing a fact on the world clock with nothing replacing it. Fully
specified in `local/END-ENDPOINT-SPEC.md`. In short: the facade can record that a value
ended *when something replaced it* (`/supersede`) and that a record was wrong (`DELETE`,
`/v1/forget`), and cannot record the third and most common case. Routing that through either
retirement route files a world change as a correction, which `memvara/types.py:195` calls
the one mistake in this library that cannot be found by reading the data afterwards.

**`Idempotency-Key` on the write routes** — accept the header, store the key with its
result, replay the stored response on a repeat. This is what makes a timed-out write
retryable instead of a coin flip. It needs a keyed store, a retention window, and a decided
answer for a key replayed with different content (409 `conflict` is the conventional one).

Both are prerequisites: the SDK ships with `end()` and with write retries enabled only once
they deploy.

## 6. Verification

Success criteria, each an output compared rather than an exit code. This library's telemetry
module exists because a red-team review classified six of eleven long-horizon failure modes
as silent.

1. A claim written through `RemoteMemvara.remember()` and read back equals, field by field,
   the same claim written through a local `Memvara` — the round-trip test that catches every
   hydration error at once.
2. Hydration contract tests run against captured real responses, not hand-written fixtures.
   A fixture written by the same person who misread the schema agrees with the misreading.
3. A protocol-conformance test walks every `MemoryAPI` member against both implementations,
   in the manner of
   `test_the_wired_list_names_exactly_the_methods_that_do_not_raise`.
4. Error mapping is asserted per code, including that an unknown code raises the base class.
5. A write that fails in a way the server has not classified as safe is **not** retried.
6. `memvara-mcp --mode cloud` starts and all fourteen tools answer, `memory_end` included.
7. Round-trip tests through the real `/v1` app where `memvara-cloud` is importable, skipped
   otherwise.

## 7. Documentation, in the same commits as the code

`README.md`; `docs/API.md`; `docs/OPEN-CORE.md` (the one false sentence, and a seam-table
row for the facade client); `docs/ROADMAP.md` (the declined-REST-client entry, moved and
attributed); `docs/UPGRADING.md`; `CHANGELOG.md`; `pyproject.toml`'s `cloud` extra comment,
which currently says login and `RemoteStore` are the only outbound calls in the
distribution; `memvara/aio.py`'s docstring.

`memvara/skills/memvara/SKILL.md` sends non-MCP callers to the REST API and must change. It
is vendored by sha into seven plugin repositories, so it moves in **its own commit**, never
as a drive-by.

In `memvara-cloud`: the `end` route's own documentation, the `/v1/forget`, `DELETE
/v1/memories/{id}` and `/supersede` docstrings that become incomplete when it ships, the
REST reference, the idempotency contract on every write route, and that repository's
`CHANGELOG.md`.

## 8. Deliberately not in scope

- **A JavaScript client.** Declined in `ROADMAP.md` for reasons this work does not touch.
- **Completing `RemoteStore`.** Still refused, still for `OPEN-CORE.md`'s reasons. This
  design does not need it and does not enable it.
- **Running the engine against a remote store.** Unchanged. The engine runs server-side.
- **Local caching of remote reads.** A cache with no invalidation signal from the server
  would answer with superseded values, which is the one failure this product exists to
  prevent.

## 9. Risks

**Schema drift between the repositories.** The client mirrors `render.py` by hand, and
nothing fails when the server renames a field except a test that has to be run against a
real deployment. Mitigated by strict hydration and by the round-trip tests, not eliminated.

**Two prerequisites in another repository.** The SDK cannot ship complete until `POST
/v1/end` and `Idempotency-Key` deploy. If either slips, the honest options are to ship
without `end()` and say so, or to hold — not to route `end()` through a retirement route.

*Both have now been built and reviewed on `memvara-cloud`'s `feat/v1-end-and-idempotency`,
and neither is deployed yet. The SDK's write-retry behaviour must stay disabled until they
are, because a client retrying against a deployment that ignores the header duplicates
data silently.*

**Idempotency is per-worker.** Found in that branch's final review: the store lives in the
serving process, so a retry routed to a second worker re-executes. The client cannot detect
which worker served it and must not claim more than the server delivers — see §3. This is
the one place where the SDK's central promise is weaker than it first appears, and it is
worth saying out loud in the client's own documentation rather than only in the server's.

**Scope of the change.** Two repositories, four or five commits, a new public API and a
server refusal removed. Sequence: the `memvara-cloud` PR first, then the SDK, then cloud
mode, then the skill's own commit.
