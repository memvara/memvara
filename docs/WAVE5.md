# Wave 5 — five debt items, four workstreams

Read this whole file before writing anything.

## The project

**memvara** at `/Applications/workstation/agent-memory` — a bitemporal memory layer for AI
agents. Facts are `Claim` triples with two independent time axes; contradictions resolve
by a keyed lookup on `(subject, predicate)` rather than top-k similarity plus LLM
adjudication; retrieval is BM25 + vector fused by RRF. Start with `README.md` and
`docs/INTERNALS.md`.

It was called `engram` until recently. Historical references to that name survive
deliberately in `CHANGELOG.md`, `docs/ROADMAP.md` and `docs/RELEASING.md` and should stay.
An `engram` string anywhere in code is a bug — report it.

## House standards, enforced

**100% statement coverage** (`fail_under = 100`). Non-negotiable.

```bash
python3 -m pytest -q                                        # 2047 passing today
python3 -m coverage run -m pytest && python3 -m coverage report
```

**Tests run offline** — no network, no API key, no sleeping except where concurrency is
the thing under test. Control time with explicit `datetime` values, never by patching the
clock. Optional SDKs are lazy imports tested with injected fakes; copy
`memvara/llm/openai.py` and `tests/test_llm_openai.py`.

**A test names the failure it prevents** and says in its docstring why that matters.
`test_a_backdated_supersession_closes_valid_time_where_the_new_value_begins`, not
`test_retire`.

**Comments say why, not what.** Match the surrounding voice; read a few modules first.

**Measure, don't assert.** If you claim a performance fix, show before/after numbers and
the conditions. **If a result flatters us, distrust it and go find the bug** — this repo
has twice shipped something whose bug favoured us, and both times disbelief caught it, not
tests.

## Exclusive file ownership

Touch **only** your files. Need a change elsewhere? Put it in your report.

| | owns |
|---|---|
| **A — store performance** | `memvara/store/**`, `memvara/write/pipeline.py`, `memvara/write/reconcile.py`, `tests/test_store.py`, `tests/test_pipeline.py`, `tests/test_reconcile.py` |
| **B — integrations** | `memvara/integrations/**`, `tests/test_integrations.py` |
| **C — public API typing** | `memvara/core.py`, `memvara/retrieve/**`, `memvara/server/**`, `memvara/embed/**`, `memvara/types.py`, `memvara/aio.py`, `tests/test_api.py`, `tests/test_hybrid.py`, `tests/test_server.py`, `tests/test_types.py`, `tests/test_scoring.py` |
| **D — telemetry & redaction** | `memvara/telemetry.py`, `memvara/redact.py`, `memvara/consolidate/**`, `memvara/write/fast.py`, `tests/test_telemetry.py`, `tests/test_redact.py`, `tests/test_decay.py`, `tests/test_merge.py`, `tests/test_fast.py` |

**Mine, do not edit:** `pyproject.toml`, `README.md`, `CHANGELOG.md`, `docs/**`,
`memvara/__init__.py`, `memvara/compat/**`, `memvara/llm/**`, `memvara/entities.py`,
`bench/**`, `tests/test_compat.py`, `tests/test_packaging.py`, `tests/test_bench_eval.py`,
`tests/test_integration.py`, `tests/test_edges.py`, `tests/test_internals.py`.

## Pinned interfaces — code against these, do not negotiate them

### P-1. Reverse provenance index (A implements, B may consume)

```python
Store.claims_citing(tenant: str, episode_id: str) -> list[Claim]
```

Every claim whose `sources` contains `episode_id`. A implements it on `SQLiteStore` and
adds it to the `Store` protocol in `store/base.py`.

### P-2. Scoped episode enumeration (A implements, B consumes)

```python
Store.scope_episodes(scopes: Sequence[Scope], *, limit: int | None = None,
                     newest_first: bool = False) -> list[Episode]
```

Episodes visible at those scopes, in `ts` order. A implements; **B consumes it behind
`getattr(store, "scope_episodes", None)`**, which is the existing pattern in this codebase
for optional `Store` capability (see how `core.py` treats `batch` and `clear_embeddings`).
That way B does not block on A and a third-party `Store` without the method still works.

### P-3. Redaction telemetry (D defines, I wire)

D adds the metric name to `telemetry.py` so `series_names()` enumerates it, and the
emission helper to `redact.py`. The **call sites** are in `core.py` and
`write/pipeline.py`, which D does not own — D reports the exact wiring change and I apply
it, exactly as wave 3 did for `Recorder`.

## Reporting

End with: what you built, what you measured (numbers and conditions), what you decided and
why, what you could not do, and anything wrong you found in a file you do not own. Be
specific about that last one — it has been the most valuable output of every wave so far.
