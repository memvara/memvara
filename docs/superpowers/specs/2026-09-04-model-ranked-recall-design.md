# Model-ranked recall — an opt-in read mode on the customer's own key

**Date:** 2026-09-04, reworked the same day after three reviews (correctness, pricing,
completeness). Nothing in this document is built. It is written from the judged runs in
`docs/superpowers/plans/2026-09-04-model-as-ranker-720.md` (on the benchmark branch, not
on main), the per-call measurements under `local/compress/`
and `local/pool/` in the same worktree, and the code as it stands on three main branches:
agent-memory `origin/main` at `2d9bcb5` (the local `main` ref, `05155a3`, is stale and lacks
`anchored`), memvara-cloud `main` at `912ecd9`, memvara-web `main` at `bddba0a`. The
memorybench harness is read at `82cf64a` on branch `memvara-budget-arms`. Line numbers below
are from those refs.

## 1. The answer

**Ship one opt-in read mode, `ranked`, on the customer's own model key.** After memvara's
own retrieval — hybrid fusion, then a cross-encoder over the top 200 turns — one chat call
receives the top-40 candidate turns and the question, and names the turns that bear on it
with a verbatim span for each. Those turns come back first, whole, with the span on each as
a field. The default read path does not change: no model, deterministic order, identical
cost. The hosted service refuses the mode until an organisation puts a key on file, so
memvara pays $0 per call and no plan gets a new metric, a new price or a new per-tier
number. A ranked read counts as one ordinary `retrieval.query`.

**The judged configuration scores 182 of 199 (91.5%) at a median context of 672 tokens, on
two runs with identical inputs.** That configuration was the routed role's top-40 with the
model call made offline (§2, §6). The shipped path has not been run. The number the
documentation carries is the parity run's (§6, Step 3), not 182.

What memvara spends per ranked call is one thread for about 5 to 6 s, likely: 1.06 s mean
for the cross-encoder at depth 200, measured on the study machine, plus a model call whose
mean is inferred at 4.1 to 4.7 s and whose p95 is unknown (§2). A per-organisation cap, a
process-wide executor for ranked reads and a 10 s timeout on the model call bound it (§4).
Per-call cost to the customer is in §2 and §8.

The design is written to the recommended answers to the five decisions in §10, pending the
user's word. Every paragraph that depends on one says which, and what changes if it goes
the other way. A memvara-paid allowance is phase 2, decided from thirty days of the series
phase 1 emits (§9).

## 2. What is established

**The ranking is what wins, not the compression.** Arm B — the model's kept turns rendered
whole and first, then the rest of the reranked list, greedy to 720 tokens — scores 182 of 199
on two independent runs of the same block (replicate: 182, differing on 4 questions, 2 each
way, against a reader noise floor of about 15). Arm A, the same model's spans rendered
without the turns around them, scores 165 at 77 median tokens and loses preference questions
wholesale (4 of 12). So the reader needs whole turns, and what it needed at 720 tokens was the
right ten.

| arm | correct | % | median tokens |
| --- | --- | --- | --- |
| control (cap 15, both roles) | 172/199 | 86.4 | 4,089 |
| routed-720 | 171/199 | 85.9 | 672 |
| routed-720 + prompt v2 | 174/199 | 87.4 | 672 |
| **B: model as ranker** | **182/199** | **91.5** | 672 |
| A: spans only | 165/199 | 82.9 | 77 |
| C: B + overflow spans + prompt v2 | 181/199 | 91.0 | 706 |
| E: inclusive filter, adaptive rendering | 177/199 | 88.9 | 706 |
| **B, replicate** | **182/199** | **91.5** | 672 |

Per type, arm B and its replicate: single-session-user 27/28 and 28/28, single-session-
assistant 21/22 and 22/22, preference 10/12 and 9/12, multi-session 47/53 and 47/53,
temporal 48/53 and 47/53, knowledge-update 29/31 both. The gains sit where the budget had
been cutting: multi-session and temporal questions whose gold turns ranked 14 to 37 in the
cross-encoder order. Six of the fourteen budget-cut questions are correct now.

**The filter's own numbers, measured against the gold labels before anything was judged.**
Over the routed role's top-40 turns in cross-encoder order:

| filter | list | gold recall | non-gold keep | cost per 199 |
| --- | --- | --- | --- | --- |
| gpt-5.4, precise prompt (arm B's) | top-40 | 0.895 (315/352) | 2.5% | $2.25 |
| gpt-5.4, inclusive prompt | top-60 | 0.949 (335/353) | 2.8% | $3.25 |
| union of both passes | top-60 | 0.958 | 2.9% | $5.50 |
| gpt-5.4-mini, precise prompt | top-40 | 0.912 (321/352) | 6.4% | $0.70 |
| gpt-5.4-nano, precise prompt | top-40 | 0.844 (297/352) | 4.0% | $0.19 |

The kept span is a fifth of its turn's tokens at the median. Zero parse failures in 199
calls on every file.

**Per call, from the gateway caches** (`local/compress/extractions{,_mini,_nano}.jsonl`,
199 rows each, cost exactly linear in tokens: $2.25 per million prompt tokens and $13.50 per
million completion tokens for gpt-5.4). The prompt is identical across the three top-40
files: median 3,325 tokens, p95 14,355, mean 4,618, maximum 17,284; 20 of 199 prompts exceed
8,000 tokens. Completion tokens are small (median 62 for gpt-5.4, 81 for mini, 69 for nano).
Cost per call, recomputed from the `cost` field with p95 by nearest rank: gpt-5.4 median
$0.00869, mean $0.01132, p95 $0.03267, maximum $0.03996; mini $0.00267, $0.00350, $0.00982,
$0.01213; nano $0.00072, $0.00093, $0.00267, $0.00323. The prompt distribution is
right-skewed, so the mean is 30% above the median and a bill accrues at the mean. Nothing
capped the prompt in any run, so the maximum is a measured value on this sample and not a
ceiling: production turns can be longer.

**Latency.** Wall clock at concurrency 4: 204 to 227 s for 199 calls, so 1.0 to 1.2 s per
call of throughput and, if the four workers stayed busy, a mean model-call latency of 4.1 to
4.7 s. Per-call latency was not measured and its p95 is unknown. The cross-encoder was
measured: `local/pool/ce.meta.json` records `cross-encoder/ms-marco-MiniLM-L-6-v2` at batch
64 over 39,430 pairs for 199 queries — about 198 pairs a query — at a mean of 1.062 s and a
p90 of 1.209 s per query on the study machine, model load excluded. So a ranked read holds
a thread for about 5 to 6 s, likely, and the timeout in §3 bounds only the model-call part.

**What the plan's prose gets wrong, and this document does not repeat.** The plan says
"about 2,700 input tokens per question at top-40, 4,000 at top-60". The measured prompt
medians are 3,325 and 4,940; the dollar figures in the same sentence ($0.011 and $0.016) match
the measured means and the token figures match no statistic computed.

**Where the hosted service stands today.** It runs no cross-encoder at all: `_clone` in
`memvara_cloud/memories.py:150-157` passes store, embedder, llm, telemetry, registry and
tenant and drops every `read_*` option, and `tests/test_memories.py:81-97` pins that as
deliberate. The image cannot load one either: `deploy/Dockerfile:116-118` bakes only the
embedder (`sentence-transformers/all-MiniLM-L6-v2`) into `HF_HOME`, and the runtime stage
sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` (`Dockerfile:261-263`), so
`CrossEncoderReranker()`, which loads its model in `__init__` (`memvara/rerank/cross.py:37,
55-62`), raises at construction in production (§4, "The image"). `MEMVARA_LLM=none` in
production, so no model is called anywhere and every token series is structurally zero. The
library's design invariant 1 (`docs/INTERNALS.md:33-46`) says nothing on the read path calls
a model, `README.md:488-489` repeats it ("not even the optional reranker"), and
`hybrid.py:36-40`'s module docstring says "No LLM sits on the read path". This design amends
that invariant with an opt-in, and says so in all three places.

## 3. The design in core

### The option and the switch

A constructor option in the shape of `read_reranker`, and a per-call switch on `search()`
and `recall()`:

```python
from memvara import Memvara
from memvara.select import ModelSelector
from memvara.rerank import CrossEncoderReranker

mem = Memvara("memory.db",
              read_selector=ModelSelector(base_url=..., api_key=..., model="gpt-5.4",
                                          reranker=CrossEncoderReranker(),
                                          depth=200, top_n=40, timeout=10.0))
block = mem.recall("what did they say about the trip", include_episodes=True, ranked=True)
```

`read_selector` routes by prefix to `HybridRetriever(selector=...)` like every other
`read_*` option (`memvara/core.py:638`, `811-813`). `ranked` is a keyword-only `bool`,
default `False`, on `HybridRetriever.search`, `Memvara.search`, `Memvara.recall`,
`ScopedMemvara.search`/`.recall`, their async twins, and `RemoteMemvara` — the route
`anchored` took across three commits: `c6e4a91` (`hybrid.py`, `core.py`, `aio.py`),
`cf7e526`, and `3622eb0` (`remote/api.py`, `remote/aio.py`, `remote/hydrate.py`,
`server/memory_api.py`, `server/tools.py`). The remote client sends `ranked` only when set,
so a server from before the field refuses with 422 (`_Model` sets `extra="forbid"`,
memvara-cloud `rest/models.py:50`) rather than answering unranked as though it had
honoured it. A hosted `Memvara(api_key=...)` refuses `read_selector` as it refuses every
`read_*` option (`core.py:863-873`): the selector runs server-side.

The docstring example above is a doctest (`pyproject.toml:148-149`, `--doctest-modules`),
so the one that ships passes `embedder=` and uses a fake selector; it touches no network.

`ranked=True` needs turns to act on. Two calls are contradictions, not defaults to guess
at, and the library raises `ValueError` for both: `include_episodes=False`, and
`memory_types` set — the retriever skips the episode leg whenever a type filter is present
(`hybrid.py:641`, `if include_episodes and wanted is None`), so a ranked call with a filter
would hand the selector nothing and say nothing. The hosted request models refuse both with
422 before the call is made (§4).

### Where it sits

`HybridRetriever.search` on `origin/main` runs `_gather` → `_rank` → `_interleave(_episodes)`
at `hybrid.py:641-646` → `rerank(...)[:k]` at `647-653` → `_observe` at 655. On a plain call
nothing changes. On a ranked call:

1. **The turns are gathered at the selector's depth and kept apart from the claims.** Today
   `depth = k` without a reranker, else `max(k, rerank_top_n)` (`hybrid.py:613`);
   `_episodes` returns at most `max_episodes` turns, default 3 (`hybrid.py:298`, `1197`);
   and `_interleave` merges claims and turns by score and cuts the *combined* list to
   `depth` (`hybrid.py:1280-1298`, `return out[:k]`) after `w_episode=0.5` has discounted
   every turn (`hybrid.py:297`, `1173`). Lifting the episode cap alone would not give the
   selector 40 turns: on a tenant with ordinary claim density the head would be mostly
   claims. The study never met this because its pool was 39,430 turns against 370 claims
   (`local/pool/pool.meta.json`). So on a ranked call `_episodes` returns up to
   `selector.depth` turns, and that list is the selector's candidate list; it does not go
   through `_interleave`'s cut. Claims come from `_rank` at `depth` as today.
2. **The turns are ordered.** If the selector carries a reranker, it orders the turn list.
   If it does not and the retriever has one, the retriever's orders the turn list at
   `selector.depth` before the selector reads it. With neither, the turns keep the episode
   leg's own order — which is not the judged configuration, and the option's docstring says
   so.
3. **The selector sees the first `selector.top_n` turns** and names the ones it keeps. The
   count it was handed is emitted as `retrieval.model_candidates` (below), so a tenant whose
   store yields fewer than `top_n` turns is visible in the series rather than silent.
4. **The order that comes back:** the kept turns first, in reranked order, whole; then the
   remaining turns and the claims interleaved as today, cut to `depth`; then `[:k]`; then
   `_observe`. Claims therefore sit after the kept turns and before or among the unkept
   turns by score, as `_interleave` places them. The retriever's own reranker stage is not
   run again on a ranked call: whichever reranker ordered the turns has already done the
   work the stage exists for.

The judged configuration is the cross-encoder at depth 200 and the selector over the top-40,
so `ModelSelector(reranker=CrossEncoderReranker(), depth=200, top_n=40)` is what ships as the
default of the mode. `read_rerank_top_n=200` on the retriever plus `ModelSelector(reranker=None)`
is the same turn ordering written the other way, and is what the benchmark stack already runs.

### The protocol

`Reranker` cannot express this: it returns exactly one score per document, never drops, and
must be deterministic (`memvara/rerank/base.py:29-44`; `rerank/stage.py:82-105`). Adding a
member to the `runtime_checkable` `LLM` protocol would break older implementations, which is
the reason `RelationComposer` is its own protocol (`CHANGELOG.md:2243-2247`;
`retrieve/compose.py:68-84`). So `memvara/select/base.py` defines a new one:

```python
@dataclass(slots=True, frozen=True)
class Candidate:
    id: str            # episode id
    when: datetime     # the turn's timestamp
    text: str          # the whole turn

@dataclass(slots=True, frozen=True)
class Selected:
    id: str
    span: str | None   # verbatim substring of the candidate's text, or None

@runtime_checkable
class Selector(Protocol):
    name: str
    is_noop: bool = False
    reports_usage: bool = False
    def select(self, question: str, candidates: Sequence[Candidate], *,
               asked_on: datetime | None = None,
               usage: Usage | None = None) -> Sequence[Selected]: ...

class SelectorUnconfigured(LookupError): ...   # ranked=True and no selector to run
class SelectorKeyRejected(PermissionError): ... # the provider refused the key (401, 403)
```

Input: the question, the date it is asked on, and the scored candidate turns in reranked
order. Output: the ids the model kept, each with its span. The stage (`select/stage.py`)
takes `Rankable` items (`rerank/stage.py:39-51`) and receives only `EpisodeResult` items;
`Result` (claim) items never reach it. It orders the kept turns by their reranked rank, not
by the order the model listed them, because arm B rendered "kept turns whole, in rank order"
and that is the measured thing. A span that is not a substring of its turn is replaced by
`None` and the turn is still kept: the ranking is what was judged, the span is a courtesy.
An id the model invents is ignored. The two exceptions are the refusals in "The refusals
and the fallback" below.

### The prompt

`ModelSelector` sends exactly what `local/compress/extract.py` sent, because that is the
prompt the 182 was measured with. The system message, verbatim from `extract.py`:

> You filter conversation excerpts for a question-answering system. For each numbered
> excerpt, copy out VERBATIM the shortest span or spans that could help answer the question:
> every number, name, date, place, quantity, duration, price, product or decision that bears
> on it, with just enough surrounding words to keep its meaning. Never paraphrase, never add
> words, never answer the question, never merge excerpts. If an excerpt has nothing that bears
> on the question, omit it. Be inclusive on the borderline: a partial mention that might
> combine with other excerpts is worth keeping. Respond with JSON only:
> {"kept": [{"i": <excerpt number>, "span": "<verbatim text>"}, ...]}.

The user message is `Question (asked on {date}): {question}\n\nExcerpts:\n\n{body}`, where
each excerpt renders as `[{n}] ({timestamp to the minute, T replaced by a space}) {text}`,
numbered from 1. The request is a chat completion with `max_completion_tokens: 4000` and
`response_format: {"type": "json_object"}`. The reply's `kept` list is parsed as
`extract.py:62-73` parses it: a malformed entry is skipped, an out-of-range `i` or an empty
span is dropped. A reply that is not JSON is a fallback (below), where the study recorded it
as an empty kept set — it happened zero times in 199 calls.

The backend talks to the endpoint with `urllib.request`, as `extract.py` does, so the core
install gains no dependency: `dependencies = ["numpy>=1.24"]` is pinned by exact equality and
adding one is a product decision (`CONTRIBUTING.md`, Scope). `tests/test_rerank.py:379-429`
already asserts in a subprocess that the default configuration imports no reranker backend;
`memvara.select.model` joins that list. One timeout, default 10 s, no retries — the study
retried four times at 180 s because nothing was waiting on it; a request is. A provider
answer of 401 or 403 arrives as `urllib.error.HTTPError` and is not a fallback: the backend
raises `SelectorKeyRejected`, because a revoked key served unranked for a month is the
failure the fallback must not hide. Every other `HTTPError` (402, 429, 5xx) is a fallback
with its status recorded.

### How the result travels

`Explanation` is a slots dataclass whose field order is an API (`types.py:837-903`). Two
fields are appended after the last existing one, `anchor` (903):

- `selected: bool | None = None` — `True` the model named this turn, `False` it saw the turn
  and did not, `None` the selector did not see it (a claim, a turn past `top_n`, a plain
  read, or a fallback).
- `span: str | None = None` — the verbatim span, only ever set when `selected` is `True`.

`search()` returns the kept turns first, whole, then the rest as §3 "Where it sits" orders
them, then the `[:k]` cut; a caller that wants the study's block asks for `k` at least
`top_n` and fills its own budget.

`recall()` renders claims first and turns as a capped tail, never interleaved, and cuts each
turn to `RECALL_EPISODE_CHARS = 280` (`core.py:2087`, `2195-2207`, `2355`). On a ranked call
the tail leads with the kept turns rendered **whole** — the 280-character cut does not apply
to them, because a cut turn is arm A's failure mode, not arm B's block — and the unkept turns
follow at 280 as today. `k`, `budget` and `counter` bound the block as they do now, claims
first. Say it plainly: `recall()` puts claims before turns by design (`core.py:2195-2200`),
so its ranked block is not the turns-only block arm B was judged on, and the judged number
belongs to `search()` through the harness. The MCP `memory_recall` tool returns `recall()`
text verbatim (`server/tools.py:450-461`), so what that door produces on a ranked call is
a measured ordering inside an unmeasured render; §9 lists the run that would measure it.
`RecallResult` does not change. The span is on the search result so a caller can render
either the turn or the span; rendering spans inside `recall()` is the adaptive rendering
deferred in §9.

A ranked call's outcome travels at the top level, not only per item. `search()` results
and `RecallResult` carry nothing new; the hosted wire does: `SearchResponse` and
`RecallResponse` (`rest/models.py:457-471`) gain `selection`, an object with `outcome`
(`applied` or `fallback`), `reason` (null, or one of the fallback reasons below), `status`
(the provider's HTTP status when `reason` is `provider`, else null) and `candidates` (the
count the selector was handed); it is null on a plain read. `Ranking` (`rest/models.py:389`)
gains `selected` and `span`, `rest/render.py:155` renders them, and `remote/hydrate.py:164-177`
reads both when present and leaves them `None` when a server does not send them, as it does
for `anchor`. A `recall()` block served by fallback ends with a `RECALL_UNRANKED` line, in
the shape of `RECALL_DROPPED` (`core.py:2089-2102`), naming the reason: a model reading the
block, and a person reading a transcript, see that the order is the plain one.

### The refusals and the fallback

Three outcomes, and the rule that decides between them: a call that cannot be honoured is
refused, a call that was honoured as far as the model is served, and nothing is silent.

- **Refused: no selector.** `ranked=True` on a retriever with no selector, or with one whose
  `is_noop` is true, raises `SelectorUnconfigured` before any leg runs. Not served unranked
  with a warning, which is what the first draft said: the hosted MCP door reaches core
  directly (`memvara-cloud rest/mcp.py:225-235`), a warning lands in a server log the
  customer never sees, and clones are rebuilt every 60 s, so "once per instance" would be
  once a minute per tenant. Refusing in core gives the REST and MCP doors one answer.
  Nothing is emitted from core; the hosted handler counts it (§4).
- **Refused: the key.** The provider answers 401 or 403: `SelectorKeyRejected`, nothing
  served, nothing emitted from core.
- **Served by fallback.** The model call times out, fails to connect, returns any other
  HTTP error, or returns something that is not JSON: the stage returns the reranked order
  unchanged, `selected` is `None` on every item, `selection.outcome` is `fallback` with
  `reason` in `timeout`, `error`, `provider`, `malformed`, and `retrieval.model_fallback`
  records it with the same tag. The fallback is never an empty result.

### Counting

Seven series, named like the write side (`telemetry.py:175`, `188-189`, `242`) and picked up
by `series_names()` automatically because they are dotted upper-case constants in
`memvara/telemetry.py` (`telemetry.py:683-699`). All seven constants live there, including
the one only the hosted handler emits, so the quota, admin and metering tests that iterate
`series_names()` see one list:

| constant | series | kind | rule |
| --- | --- | --- | --- |
| `RETRIEVAL_MODEL_QUERY` | `retrieval.model_query` | counter | one per ranked read the model answered; the read-side twin of `write.llm_calls`, counting model consultations, so a no-op selector emits nothing (`llm/base.py:70-73`). No tag: this is the series a phase 2 allowance would sum, and quota sums a source by name and ignores tags (`memvara-cloud quota/engine.py:363-367`, `metric = ANY(%(sources)s)`), so a fallback must not share its name |
| `RETRIEVAL_MODEL_FALLBACK` | `retrieval.model_fallback` | counter, tags `reason`, `status` | one per ranked read served unranked; `reason` in `timeout`, `error`, `provider`, `malformed`; `status` the provider's HTTP status when `reason` is `provider` |
| `RETRIEVAL_MODEL_REFUSED` | `retrieval.model_refused` | counter, tag `reason` | one per ranked request refused; `reason` in `unconfigured`, `disabled`, `key_rejected`, `inflight`. Defined in core, emitted by the hosted handler (§4), because a refusal happens before `_observe` runs |
| `RETRIEVAL_MODEL_CANDIDATES` | `retrieval.model_candidates` | counter, value = turns handed to the selector | one per ranked read the model saw; 40 on every question in the parity run, and less than that on a tenant with few turns |
| `RETRIEVAL_TOKENS_IN` | `retrieval.tokens_in` | counter | only when the `Usage` accumulator reports > 0, exactly `_report_usage`'s rule (`write/pipeline.py:731-746`): a zero would understate a bill in the direction that favours us |
| `RETRIEVAL_TOKENS_OUT` | `retrieval.tokens_out` | counter | same |
| `RETRIEVAL_SELECT_MS` | `retrieval.select_ms` | timing | the model call only, and only when a model was consulted, the `write.extract_ms` rule (`telemetry.py:231-242`). `retrieval.latency_ms` on the same read minus this is the reranker's cost |

They are emitted from the same recorder `_observe` uses, at the end of the ranked call.
`retrieval.query`, `retrieval.results` and `retrieval.latency_ms` are emitted as today, once,
so a ranked read is one query in every place that counts queries. The ranked share of reads
on a project is `retrieval.model_query + retrieval.model_fallback` over `retrieval.query`;
`retrieval.query` itself carries only a `script` tag and cannot tell the two apart.

## 4. The design in memvara-cloud

### The per-request flag

`ranked: bool = False` beside `anchored` on `SearchRequest` (`rest/models.py:1051-1063`) and
`RecallRequest` (`1105-1110`), passed through at `rest/app.py:844-848` and `911-914`. A
`model_validator` on both models refuses `ranked` with `include_episodes` false or
`memory_types` set as 422 `invalid_request`, so neither contradiction reaches `ctx.run` —
where a library `ValueError` would surface as a 500, since only `_axes` and `_states` wrap
one into a 400 (`app.py:316-324`, `352-358`). Not on `AskRequest`: `/v1/ask` composes over
slots and renders every sentence from a stored column (`app.py:1189`, "Nothing here consults
a model"), and the selector acts on turns, so the flag would have nothing to act on there — a
flag that does nothing is a lie in a request model.

The MCP door reaches the same per-project handle (`rest/mcp.py:225-235`), so `memory_search`
and `memory_recall` take `ranked` in the core tools (`server/tools.py`, beside `anchored` at
1539 and 1571), described for the model that reads them: set it when a question is worth a
model call and the answer is in what was said rather than in a stored fact; leave it off for
an ordinary turn; it needs `include_episodes` and no `memory_types`; a server with no
selector configured refuses it rather than answering unranked. `SelectorUnconfigured` and
`SelectorKeyRejected` become tool errors over MCP and 409s over REST; `memory_search`'s
result carries the same `selection` object the REST response does.

The recommendation writes the flag as `?ranked`. It goes in the body, not the query string,
because it selects no allowance and no weight, so nothing needs it before the body is parsed.
`depth` on the graph routes is a query parameter for exactly the opposite reason
(`ratelimit/policy.py:279-284`): the limiter resolves an operation before the route runs.

### The quota metric and the allowance

**None new in phase 1.** A ranked read spends the same `retrieval.query` reservation as a
plain one (`QUOTA_METRICS` at `policy.py:559-572` is unchanged; `tests/test_ratelimit_http.py:523-535`
keeps pinning that `POST /v1/search` reserves exactly `("retrieval.query",)`, and gains a
twin for a ranked body). The per-tier allowances stay as `quota/plans.py` has them: free
2,000 a month, personal 700 a day, personal_pro 2,000 a day, studio 5,000 a day, team 50,000 a
day, business uncapped. The reason is the margin table in §8: memvara's model cost per call
is $0, so there is nothing to ration.

Phase 2, if it happens (decision 5), is a `retrieval.model_query` metric with the edits the
quota brief lists — a `Metric` in `METRICS` (`plans.py:170-204`), an allowance tuple in every
`DEFAULT_PLANS` entry, a `LABELS` phrase (`notify.py:145-150`), a `QUOTA_METRICS` entry —
with allowances of free 0, personal 250, personal_pro 500, studio 1,000, team 3,000,
business 15,000 a month, on gpt-5.4-mini after a judged run. Its source is
`retrieval.model_query` alone, so a fallback or a refusal never spends it. It is not built
now; §9 says why.

### The refusals, and what each one costs

- **Past the allowance: unchanged.** Only free's `retrieval.query` allowance is monthly
  (`plans.py:313`), and a spent monthly allowance is 402 `quota_exhausted` with detail
  `{metric: "retrieval.query", limit, used, resets_at, reason}` and no `Retry-After`
  (`rest/limits.py:1405-1416`). Personal through team are daily (`plans.py:330`, `336`,
  `343`, `399`), and `day` is in `SOFT_PERIODS`, so a spent daily allowance is 429
  `rate_limited` with `Retry-After` (`plans.py:74`; `limits.py:1393-1402`), capped at one
  day. Business has none. A ranked read at the allowance meets exactly the refusal a plain
  one does; no new code.
- **`ranked: true` that cannot be honoured: 409 `selector_unconfigured`.** One code for
  three causes — no key on file, the mode switched off by the operator
  (`MEMVARA_SELECTOR_INFLIGHT=0`, below), or the key refused by the provider — because
  `rest/errors.py`'s `CODES` is an index of failure classes and not of raise sites
  (`app.py:314-316`), and the message names which. The body names the console page where a
  key is set. `CODES` (`errors.py:70-98`) is a closed frozenset that
  `tests/test_control_api_wire.py:17` imports and `tests/test_rest_openapi.py:36` and `:60`
  check against every route's documented refusals, so the code is added there and on the
  two routes. Refused, not served unranked — the `anchored` precedent: a server that cannot
  honour a flag refuses rather than answering as though it had. A 4xx releases the
  reservation (`limits.py:1081-1082`), so the refusal costs the customer nothing. The
  handler emits `retrieval.model_refused{reason}` before returning, with `reason`
  `unconfigured`, `disabled` or `key_rejected`; nothing else counts a refusal, because
  `retrieval.query` is emitted inside core's `_observe` and never runs on one, and the
  phase 2 trigger (decision 5) is written on this series.
- **The in-flight cap is full: 429 `rate_limited`, `Retry-After: 6`** — likely right, from
  the 5 to 6 s a ranked read holds a thread. `_quota_retryable` takes a quota `Decision`
  (`limits.py:1393`) and there is none here, so this refusal has its own small renderer with
  the same envelope. The cap is a semaphore per organisation held around the whole
  `ctx.run(...)` of a ranked handler — the selector runs inside `HybridRetriever.search` on
  the worker thread, so the handler cannot wrap anything narrower. Counted as
  `retrieval.model_refused{reason=inflight}`.
- **The model call fails or times out:** served, unranked, `selection.outcome` `fallback`
  with its reason and status in the body, `selected` null on every result, and
  `retrieval.model_fallback` on the series. The `retrieval.query` reservation stands,
  because a read was served.

### The ledger line for what memvara pays

There is none, and the package could not hold one: every `invoice_line.amount` and every
price must be >= 0, "a negative allowance or price is a credit, which this package
deliberately cannot express" (`billing/rates.py:318-320`, `386-388`; `schema.py:236`, `278`),
and no column, line kind or table records a cost memvara pays a provider. Under this design
memvara pays nothing per call, so the invariant `tests/test_billing_plans.py::test_no_plan_meters_anything`
holds untouched and the worst invoice any tier can produce stays its base fee. What is
recorded instead is `retrieval.tokens_in` and `retrieval.tokens_out` per project, so an
organisation can reconcile its provider's invoice against what memvara sent. The gateway
charged $2.25 per million prompt tokens and $13.50 per million completion tokens for
gpt-5.4; a customer's provider prices its own key, and whether the two agree is a week-one
check (§10), not a fact.

`llm.tokens` does not count these: its sources are `write.tokens_in` and `write.tokens_out`
(`plans.py:196-197`), write-path extraction only, and it stays that way. `quota/plans.py:284-290`
calls an unbounded pass-through cost "an open invoice"; the answer here is that the invoice
is the customer's, on their key, and the cap on memvara's side is thread time under the
in-flight cap.

### The production configuration

This is the first model call from the production process, and it is the customer's call.

- **`MEMVARA_LLM` stays `none`.** It configures write extraction only
  (`core.py:790-792`) and turning it on would start "spending on every write that reaches the
  extraction tier" (`asgi.py:398-402`). The read-side model has its own settings.
- **Key handling** (decisions 1 and 2; decision 3 for who may set one). One key per
  organisation, set in the console, stored encrypted with AES-256-GCM under `idp_key`
  exactly as `identity_providers.secret_ciphertext` is (`control/idp/crypto.py`), or under
  the organisation's customer-managed key where one is enabled; the console shows a
  fingerprint only. The base URL and model are stored beside it; the default model is
  gpt-5.4 and gpt-5.4-mini is selectable, with the qualifier §9 gives it. A free
  organisation may set a key: memvara's cost is thread time under the cap, and excluding
  free makes it the one place a visitor cannot see the result. There is no per-customer key
  plumbing today (`git grep -nEi 'bring.your.own|byok' main` hits only CMK docs saying it is
  not BYO), so this is new. If decision 1 goes the other way — a per-request header, nothing
  stored — there is no column, no schema bump and no console field, and the MCP door is
  excluded because it cannot add a second header.
- **The column is a schema bump an operator runs by hand.** There is no migration
  mechanism: `control/schema.py`'s `check_version` refuses every older stamp, a new column
  on an existing table is "version 4's case, with an `ALTER TABLE` for an operator to run"
  (`schema.py:299-303`), and `SCHEMA_VERSION` is 23 (`schema.py:689`). This is 24: three
  nullable columns on the organisation row (key ciphertext, base URL, model), the
  `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` statements and `UPDATE control.control_meta SET
  version = 24` written into `deploy/README.md` beside the 21-to-23 statements
  (`deploy/README.md:234-254`) and into `docs/OPERATIONS.md`, and run before the new image
  starts, or the process refuses to boot.
- **Building the selector per organisation.** `ProjectMemories.for_tenant` knows a project
  and reads one thing from the control plane, `Categories.category_for(tenant)`
  (`memories.py:56-60`), on a 60 s TTL with a 128-entry cap (`memories.py:46`, `52`,
  `105-132`). The protocol gains a second method, `selector_for(tenant)`, that resolves the
  project's organisation, decrypts the key server-side and returns the selector's
  configuration with a key fingerprint; `_clone` builds the `ModelSelector` from it. The
  docstring's "one method, deliberately" (`memories.py:57-59`) is amended with the reason:
  the alternative is a second protocol for one lookup. The configuration is re-read on the
  same TTL check the category uses; when the fingerprint changes the clone is rebuilt. So a
  key change or removal takes effect **within 60 s, not at once**: "nothing here can be told
  about a change: the console writes one row and the workers that matter are other processes
  on other machines" (`memories.py:40-46`), and doing better needs infrastructure this
  design does not add.
- **Threads and timeouts.** Nothing in `deploy/` or `rest/` sets a request timeout; uvicorn
  is started with none, and every core call runs on the `asyncio.to_thread` default
  executor (`deps.py:113-125`), which is `min(32, cpu + 4)` threads on Python 3.13
  (`Dockerfile:32`), while compose limits memory only (`compose.yaml:552-554`, no `cpus`).
  A per-organisation cap of 4 is a global bound of 4 times the active organisations: two of
  them on a 4-CPU host fill the default executor and every plain read for every tenant
  queues behind model calls. So ranked reads run on **their own executor**, sized by
  `MEMVARA_SELECTOR_THREADS` (default 8, a starting value to be set from week one), which
  is the process-wide cap; plain reads keep the default executor. The per-organisation cap
  is `MEMVARA_SELECTOR_INFLIGHT`, where 0 (the default) is the operator's switch: the mode
  is off process-wide and every ranked request is 409 `selector_unconfigured` with the
  message saying so and `reason=disabled` on the series. Production sets 4 (Step 5). The
  selector's 10 s timeout, `MEMVARA_SELECTOR_TIMEOUT_S`, bounds the model call only; the
  cross-encoder stage in front of it is not bounded by anything and cost 1.06 s mean on the
  study machine (§2), unmeasured on the production host. Step 5 measures it there before the
  cap is set, and the semaphore is budgeted on 5 to 6 s a call. `MEMVARA_WORKERS=1` in
  production.
- **The extractor rate-limit analogue.** `MEMVARA_RATELIMIT_EXTRACTOR_MS` adds a model's
  milliseconds to `addMemories`' weight (`policy.py:728`, `744-746`). The same cannot be done
  here: a 5 s step is 50,000 units at 0.1 ms a unit, `search` and `recall` weigh 65, the
  credential burst is 1,200, and `Policy.__post_init__` refuses any route weight above a
  burst (`policy.py:766-772`). So ranked reads keep their 65 and the two caps above are the
  control.

### The `_clone()` fix, a prerequisite

`_clone` drops `redactor`, `reembed` and all `**tuning` (`memories.py:150-157`;
`core.py:696-714`), and the deployment sets none of them (`asgi.py:1371-1372`), which is why
the guard test lists `tuning` as deliberately not carried (`test_memories.py:81-97`). Until
that changes the hosted number for any read option is the cap-15 number, as the plan says.
The fix: `ProjectMemories` takes the read tuning explicitly at construction, `_clone` forwards
it and adds `read_selector=` from `selector_for(tenant)`, and the guard test moves `tuning`
into the carried set with the reason. Nothing else about the clone changes; in particular
`llm` still comes from the base handle, so write extraction is unaffected. The package now
depends on the core's `read_selector` option and `ranked` keyword, and the cloud's rule is
that such a dependency is pinned in `tests/test_core_contract.py` in the same commit (the
cloud repository's working-here instructions, lines 140-144).

### The image

The judged default cannot construct in the image on main (§2). `deploy/Dockerfile` gains a
build step beside the embedder's (`Dockerfile:116-118`) that downloads
`cross-encoder/ms-marco-MiniLM-L-6-v2` into the same `HF_HOME`, pinned by model id, so the
runtime stage's `HF_HUB_OFFLINE=1` still holds. `sentence-transformers` is already installed
through `[local-embed]` (`Dockerfile:84`); the weights are what is missing. The smoke in
Step 5 asserts that `CrossEncoderReranker()` constructs offline in the running image and
times `score()` over 200 pairs, which is the measurement the cap is set from.

## 5. The pricing page in memvara-web

One FAQ entry in `src/content/pricing.ts` beside "What is actually charged?" (`999-1008`)
and a rewrite of the "Searching costs nothing" paragraph in `src/routes/Pricing.tsx`
(`420-425`). No card gets a number; no `Plan` field; no comparison row. The FAQ text, from
the recommendation:

> Model-ranked recall: put your own model key on file and ask for it per request. It counts
> as an ordinary recall against your plan's allowance. The model call goes to your provider
> on your key and appears on their bill, not ours.

The paragraph cannot take the sentence as an addition, because as it stands it says
"Retrieval is BM25 and vector search, fused locally. It writes nothing and touches no
balance" (`Pricing.tsx:421-423`), and a ranked read is a model call that leaves on the
customer's key and lands on their provider's balance. Both clauses are qualified:
"fused locally unless you ask for model-ranked recall, which goes to your provider on your
key", and "touches no balance of ours".

Why this shape and not a per-tier line: the site's word "metered" means counted against a
balance and refused past it, never priced (`Pricing.tsx:397-403`), and the README's rules
say never price anything after the fact and never soften the stop. A per-call price would
break both and there is none.

Tests it touches, all in `test/pricing.test.tsx`:

- `says plainly that calls are not billed` (958-964) counts exactly `PLANS.length` matches of
  `/never billed/i`; the text above avoids the phrase.
- The wording guards — overage rate/charge/bill, worst month, hard stop, billed for what you
  use (162-171); top-up, prepaid, buy credit (193-202); grace period (274-280); prorat, only
  pay for what you use (1045-1056) — pass because none of those phrases appears.
- No `PastAllowance` kind and no `Plan` field change, so `tsc -b` and the fixtures in
  `test/pricing-list-shapes.test.tsx`, `pricing-start-shapes.test.tsx` and
  `pricing-grid-gap.test.tsx` are untouched. A grep of `test/pricing.test.tsx` on `bddba0a`
  for a FAQ-count assertion finds none; the `%s says what the data says` test (350-365)
  derives from the data.

The gate is the whole one the site's working-here instructions (lines 15-20) name, not the
test file alone:
`npm test` (vitest with `--coverage` at the 100% thresholds), `npx playwright test` (the
layout matrix), `npm run typecheck`, and `npm run deploy:dry`.

Pre-existing copy defects on site main are not touched by this change: the project ceilings
(site 1/2/4/5, cloud v8 2/3/5/6 since `7b852af`), the withdrawn-credit lines in
`Pricing.tsx:243-247` and `293-296`, "1,000 free memories" at 436 and 445, and `Terms.tsx:184-201`.
They are named here so nobody reads their survival as endorsement.

## 6. The harness, and the replicate that proves parity

Every judged arm so far ran through `MEMVARA_CONTEXT_FILE` (memorybench `82cf64a`): the
provider's rendering was replaced by a block built offline by `extract.py`, and the server
never saw the model call. Parity means the shipped path produces a number as good without
that file.

**Provider change** (`src/providers/memvara/`, on its branch, with tests): one knob,
`MEMVARA_RANKED=1`, that adds `ranked: true` to the `/v1/search` body the provider already
sends — `k: searchK()`, `min_score: SEARCH_MIN_SCORE`, `include_episodes: true`
(`index.ts:118-123`) — and nothing else. `renderMemvaraContext` (`prompts.ts:210-222`)
keeps rendering turns in the returned order and filling the budget with `fillToBudget`, so
`MEMVARA_TURNS_ONLY=1`, `MEMVARA_TOKEN_BUDGET=720` and `MEMVARA_SEARCH_K=200` reproduce arm
B's block shape from the server's kept-first order. Each knob off by default; the shipped
provider is unchanged.

**Two knobs that would change what the run measures, pinned off.** `MEMVARA_CONTEXT_FILE`
unset: when it is set and the context is non-empty the provider returns the override block
and the server's order is never rendered (`prompts.ts:211-214`). `MEMVARA_ROLE_SELECT`
unset, which is `off` (`env.ts:67-77`): with `route` on, `selectTurns` re-filters the
server's kept-first list by role after the fact (`prompts.ts:222`) and can drop the turns
the model kept. The `arm-invariant` test in Step 3 asserts both.

**Stack:** the two feature branches — memvara-cloud with Step 2a, core with Step 1 — on a
local stack, the test organisation's key set to the gateway key from
`/Applications/workstation/memorybench/.env.local` (read by `extract.py:21`) with one SQL
statement on the local database (the console field is Step 2b and is not needed for this),
model gpt-5.4, `MEMVARA_SELECTOR_INFLIGHT=4`. Same 199 questions, seed `20260903`, reader
and judge gpt-5.4, `SKIP_RETRIEVAL_EVAL=1`, paired per question, a copied checkpoint with
`dataSourceRunId` kept and search, answer and evaluate reset, per the stack notes.

**One difference from arm B, stated before the run.** Arm B's candidates were the routed
role's top-40 — `extract.py:30-33` filtered by `routed_role` before the model saw the list.
Role routing is not in core (it is fitted to this benchmark's phrasing and its precision on
real questions is unmeasured; §9), so the shipped path's top-40 is both roles. The other
difference the first draft did not name — claims sharing the head with turns — is removed by
§3's design, which hands the selector the turn list alone; `retrieval.model_candidates` at
40 on every question is the check that it was. Three checks, in order:

1. **Offline screen, no judge.** Run the harness's search stage only, with `MEMVARA_RANKED=1`
   — 199 ranked calls through the server, about $2.25 at gpt-5.4 — and score the kept set in
   the checkpoint against the gold labels, as `extract.py` prints: gold recall and non-gold
   keep rate, against 0.895 and 2.5%. The scoring script is a Step 3 deliverable on the
   memorybench branch; today `extract.py` scores from `local/sweep/prep.pkl` through
   `sweeplib.routed_role` (`extract.py:13-33`) and nothing scores the shipped list. If recall
   falls under 0.85, stop: assistant turns are taking gold turns' slots, and routing goes
   back in front of the selector before the judged run.
2. **The unranked twin.** The same stack and list with `MEMVARA_RANKED` off, rendered
   through the same 720-token budget: plain reranked order, both roles. No judged arm has
   this configuration (routed-720 was one role; the control was cap 15), and without it a
   ranked result of 174 cannot be told from "the selector did nothing": 174 is inside the
   band around routed-720's 171 and the control's 172.
3. **The judged replicate, paired.** Answer and evaluate on the checkpoint from check 1.
   Prediction: **the ranked run beats its unranked twin by 8 or more, paired per
   question.** 199 paired questions resolve a difference of about 8 on a single run (the
   reader disagrees with itself on 7.8% of identical prompts, about 15 judgements), so a
   paired gain under 8 is not evidence that the selector did anything on this list, and the
   feature does not ship on it. The ranked run's absolute score is recorded with its
   qualifiers and is the number the documentation carries; 182 is not carried anywhere
   until the shipped path has produced it. The median context is recorded, not gated: a
   different candidate list fills a budget differently.

The plan's per-type table is the diagnostic if the gate misses: a drop concentrated in
single-session-assistant means routing; a drop spread across types means the candidate list.

## 7. Build

Tests first in every step, on a branch, with a PR that is reviewed before merge
(`/code-review high <PR>`, on the review model the repository's working-here instructions
require, findings fixed on the same branch). Gates from `CONTRIBUTING.md` for core:
`python3 -m pytest -q`, coverage at 100% of statements (`fail_under = 100`, run under a
private `COVERAGE_FILE`), `mypy -p memvara` clean, `embedder=` at every `Memvara()` a test
constructs, no network and no sleep in the suite, and a fake that counts its own calls
wherever a model would be. Documentation ships in the same commit as the code, every step.

### The order, and why

The cloud's CI checks the core out at `MEMVARA_CORE_REF` or `main` and installs it editable
(`.github/workflows/ci.yml:62`, `128`), and `scripts/check_core.py` fails closed on a core
checkout that is not `main` at the core's `origin/main` (`docs/STANDARDS.md:56-60`). So the
cloud gate cannot go green on a core change until that change is on core main — and the
parity run, which is the gate that may stop the feature, needs both changes. Running it after
both merges would leave the feature, its tool descriptions and a changelog entry on two
mains for a feature the plan says does not ship. The order is therefore:

1. **Open the core issue first.** `CONTRIBUTING.md:147-150` asks for one before a change to
   "the retrieval scoring ... or anything that alters what `why()` reports"; a model that
   reorders results and annotates them is that change. The PRs cite it.
2. **Step 1** on a core branch; PR open, reviewed, not merged.
3. **Step 2a** on a cloud branch, gated with `MEMVARA_ALLOW_CORE_DRIFT=1` pinned to the core
   PR branch — the deliberate-pin case `STANDARDS.md:60` names; PR open, reviewed, not merged.
4. **Step 3**, the parity run, from the two branches.
5. **Merge and release**: the core PR merges with the parity number added to its changelog
   entry in the Step 3 commit; the core is released per `docs/RELEASING.md`, because the
   cloud depends on `memvara>=0.1` from the index (`pyproject.toml:12`); the cloud PR is
   re-gated against the released core with `check_core` green, and merges.
6. **Step 2b**, **Step 4**, **Step 5**.

### Step 1 — core: the selector (agent-memory)

Tests to write first, in `tests/test_select.py`:

- a counting fake `Selector` sees exactly `top_n` candidates, all of them `EpisodeResult`, in
  reranked order, and is called once per ranked read and never on a plain one;
- on a ranked call the selector's list is the turn list at `selector.depth`, unaffected by
  how many claims `_rank` returned; on a plain call the episode cap is `max_episodes` and
  `_interleave` cuts as today;
- kept turns come first, in reranked order, whole, with `selected=True` and their span; the
  rest follow as §3 orders them with `selected=False` on the turns the model saw; claims
  carry `None`;
- a span that is not a substring of its turn becomes `None` and the turn is still kept;
- a raising fake, a fake that returns non-JSON, a fake that times out, and a fake that
  answers 429 each produce the reranked order with `selected` `None` throughout, one
  `retrieval.model_fallback` with the right `reason` (and `status` for the 429), and no
  token series;
- a fake that answers 401 raises `SelectorKeyRejected`; a missing selector and an `is_noop`
  one raise `SelectorUnconfigured` before any leg runs, and nothing is emitted;
- `retrieval.tokens_in`/`tokens_out` are emitted only when `Usage.reported > 0`;
  `retrieval.model_query`, `retrieval.model_candidates` (value equal to the count handed
  over) and `retrieval.select_ms` only when a model was consulted, with a recording
  telemetry;
- `ranked=True` with `include_episodes=False`, and with `memory_types` set, each raise
  `ValueError`;
- `recall(ranked=True)` renders kept turns whole and unkept turns at 280 characters, and a
  fallback block ends with the `RECALL_UNRANKED` line;
- the prompt built for a fixed candidate list is byte-identical to `extract.py`'s, asserted
  against a fixture copied from it, so a drift in the prompt is a failing test and not a
  silent change to a measured thing;
- `tests/test_rerank.py`'s subprocess test extended: `memvara.select.model` is never imported
  by the default configuration;
- the seven new constants appear in `series_names()`;
- and the layers `3622eb0` tested when it threaded `anchored` (`git show --stat 3622eb0`):
  `tests/test_remote_reads.py` (the client sends `ranked` only when set and hydrates
  `selected`/`span`), `tests/test_server.py` (both MCP tools take `ranked`, refuse it with a
  tool error when the server has no selector, and describe both), `tests/test_bench_hosted.py`,
  and the `aio` twins.

Then the code: `memvara/select/{__init__,base,stage,model}.py`; the `ranked` keyword through
`hybrid.py`, `core.py` (both `search` overload sets, `recall`'s three overloads and
`ScopedMemvara`), `aio.py`, `remote/api.py`, `remote/aio.py`, `remote/hydrate.py`,
`server/memory_api.py`, `server/tools.py`; the two `Explanation` fields; the seven telemetry
constants; the `RECALL_UNRANKED` line.

Documentation in the same commit: `README.md:488-489` (both clauses — "nothing on the read
path calls a model" and "not even the optional reranker" — gain the opt-in), `docs/INTERNALS.md:33-46`
(invariant 1's Scope line gains its opt-in clause and says the default is unchanged, and the
Sketch line at 40-42, "`HybridRetriever` ... take no `llm` parameter at all", is corrected
for `selector=`), `hybrid.py:36-40` (the module docstring the change lands under),
`CHANGELOG.md` under Unreleased — the entry states the mechanism, that it is a query-time
model call at the customer's cost, and that the default path is unchanged, and carries **no
accuracy number** until Step 3 adds the parity number in its own commit — `docs/UPGRADING.md`
(the `MemoryAPI` protocol gains `ranked`, so an alternative implementation has to accept it,
found the way the `anchored` entry at 91-98 says), `docs/API.md`, `docs/DEPLOY.md:123-125`
(the server's environment table; the self-hosted `memvara-mcp` server has no setting that
configures a selector in phase 1, so `ranked` is refused there, and the table says so),
`memvara/rerank/__init__.py`'s docstring (points at `select`), `docs/ROADMAP.md` ("a hosted
reader has never been run" is still true and is left; the per-query model call is no longer
undiscussed), the tool descriptions in `server/tools.py`, written with the precision the
repository's working-here instructions demand there, and `CONTRIBUTING.md:21`, whose test
count is stale and is re-derived from `python3 -m pytest --collect-only -q` in this commit.

The packaged skill `memvara/skills/memvara/SKILL.md` is **not** touched in phase 1. It has
no `anchored` guidance to extend (0 hits in its 182 lines on `origin/main`), it is vendored
by sha into seven plugin repositories, and a sentence there is seven pin bumps with no
owner; that is its own piece of work, listed in §9.

Check: the three gates green; the doctest for the new option runs against a fake selector;
`git diff --stat` shows no file outside the list above.

### Step 2a — memvara-cloud: `_clone`, the flag, the refusals, the key column

Mergeable without counsel: nothing here puts a key field in front of a customer.

Tests first:

- `test_memories.py`: the carried set gains `tuning` and `read_selector`; a clone built for an
  organisation with a key has a selector and one without has none; a changed fingerprint
  rebuilds the clone on the next TTL check, and a removed key is gone from the clone within
  the TTL;
- `test_core_contract.py`: the `read_selector` option and the `ranked` keyword pinned;
- `test_ratelimit_http.py`: a ranked `POST /v1/search` reserves exactly `("retrieval.query",)`;
  `test_ratelimit_policy.py`: `QUOTA_METRICS` is unchanged;
- a ranked request with no key returns 409 `selector_unconfigured`, releases the reservation
  and emits `retrieval.model_refused{reason=unconfigured}`; the same with
  `MEMVARA_SELECTOR_INFLIGHT=0` says `disabled`; a selector that raises `SelectorKeyRejected`
  says `key_rejected`; a fifth concurrent ranked read on one organisation returns 429 with
  `Retry-After` and `reason=inflight`; a selector that times out returns 200 with
  `selection.outcome` `fallback`, its reason, and `ranking.selected` null throughout;
- `ranked` with `include_episodes` false, or with `memory_types`, is 422; `AskRequest` does
  not accept `ranked` (422), with the reason in the test name;
- `test_rest_openapi.py`: `selected`, `span` and `selection` are declared nullable
  (`test_every_field_that_can_be_null_is_declared_nullable`, line 121) and
  `selector_unconfigured` is documented on the two routes (lines 36, 60);
- `test_admin_catalogue.py` and the metering catalogue's tests: the seven series have a
  kind and a unit in `admin/catalogue.py`'s `MEASURES` (`catalogue.py:167-186`, closed —
  `admin.measure("write.turnips")` raises `UnknownMeasure`, `test_admin_catalogue.py:73-82`)
  and the ones the console shows have a name in `metering/api.py`'s `CATALOGUE` (`api.py:180-186`);
- `test_console_contract.py` unchanged: this step adds no console route;
- the key round-trips through `crypto.py`; the schema stamp is 24 and a 23 is refused with
  the remedy naming the statements.

Then the code: `memories.py` (the carried set, `selector_for`), `rest/models.py` (the two
request fields with their validators; `Ranking.selected`/`span`; `selection` on both
responses), `rest/render.py:155`, `rest/errors.py` (`CODES`), `rest/app.py` (the two
handlers, the semaphore around `ctx.run` and the dedicated executor, the 429 renderer, the
refused counter, the route docstrings at 812-814 and 909 which say what a ranked read spends
and which refusals it can produce), `admin/catalogue.py`, `metering/api.py`,
`dashboard/src/api/types.ts:115` (`Ranking`) and the search mock in
`dashboard/src/mocks/handlers.ts:2566`, `control/schema.py` (version 24, the three columns),
`deploy/memvara_deploy/settings.py` (`MEMVARA_SELECTOR_INFLIGHT`, `MEMVARA_SELECTOR_THREADS`,
`MEMVARA_SELECTOR_TIMEOUT_S`), `deploy/Dockerfile` (the cross-encoder weights),
`deploy/compose.yaml` and `env.example`.

Documentation in the same commit: `README.md`, `deploy/README.md` (the settings, that
`MEMVARA_LLM` is not what turns this on, and the 23-to-24 statements beside the 21-to-23
ones), `docs/OPERATIONS.md` (the 409, the two caps, the statements), `docs/PENDING.md` (an
entry for Step 2b: the console key field and the three legal documents, open until counsel
answers decision 4 — `docs/legal/README.md:27-29` requires an open legal change to be listed
there; the first draft's claim that a `_clone` entry closes here was wrong, `grep -iE
'clone|reranker|cross-encoder|read_' docs/PENDING.md` on main finds nothing).

Check: the cloud suite green under a private `COVERAGE_FILE` and a private `MEMVARA_PG_DSN`,
with `MEMVARA_ALLOW_CORE_DRIFT=1` against the Step 1 branch; `/code-review high`; the smoke
on the local stack shows one `retrieval.model_query` per ranked read, zero on plain reads,
and `retrieval.model_candidates` at the expected count, compared as counts, not as an exit
code.

### Step 3 — the harness and the parity run (memorybench)

`MEMVARA_RANKED` in `env.ts` with its test (unset means off, anything but `"1"` throws, the
way `MEMVARA_TURNS_ONLY` does), the body field in `index.ts`, the scoring script for the
offline screen, and an `arm-invariant` test that the shipped provider's request body is
unchanged with the knob off and that the parity stack has `MEMVARA_CONTEXT_FILE` and
`MEMVARA_ROLE_SELECT` unset. Then §6's three checks in order, the offline screen before any
judged spend, and the results written into the plan document's table as new rows with their
run ids: the ranked run, its unranked twin, and the paired difference.

Check: a paired gain of 8 or more, or the feature does not ship and the routing question is
reopened. On a pass, the parity number goes into the core branch's `CHANGELOG.md` entry with
its qualifiers (199 questions, one run, both roles), in the Step 3 commit.

### Merge and release

Core PR merged; core released per `docs/RELEASING.md` with a version the cloud can pin;
cloud PR re-gated against that core with `check_core` green, then merged. Nothing is
deployed yet.

### Step 2b — memvara-cloud: the console field and the legal documents

Blocked on counsel's answer to decision 4, and listed in `docs/PENDING.md` from Step 2a
until it lands.

Tests first: `dashboard/src/mocks/handlers.ts` gains the key routes, and a handler there is
"a commitment, not a sketch" (the cloud's working-here instructions, lines 389-393), so
`tests/test_console_contract.py` sees the real routes in the same PR —
`test_every_console_route_exists_on_the_server` has no exception list (same, lines
398-400); the dashboard's own suite (`vitest run`,
`dashboard/package.json:11`) covers the field on `dashboard/src/routes/OrgSettings.tsx`:
set, fingerprint shown, remove, never the key echoed.

Then the code: the console API under `control/api/`, `OrgSettings.tsx`, the handlers.

Documentation in the same commit — `docs/legal/DPA.md:50-57`, `docs/SECURITY-QUESTIONNAIRE.md:292-298`
and `docs/legal/SUBPROCESSORS.md:105` — the three places that state no model is consulted
and no content leaves. "A sentence here is either true of the deployment today or it is not
in the document" (`docs/legal/README.md:33`), so they change in this commit and not before:
the questionnaire's "there is no path in the code that could do it" stops being true when
Step 2a deploys, which is why Step 5 does not deploy before this step lands. The PENDING
entry is deleted here.

### Step 4 — memvara-web

The FAQ entry and the paragraph, with the whole gate in §5 green. Nothing else on the page
changes.

### Step 5 — production

In dependency order. Before the new image starts: the 23-to-24 statements, by hand, as
`deploy/README.md` records for every bump. Then the image with the cross-encoder weights;
the smoke asserts `CrossEncoderReranker()` constructs offline and reports `score()` over 200
pairs in milliseconds, which sets `MEMVARA_SELECTOR_THREADS` and confirms `Retry-After`.
Then the settings: `MEMVARA_SELECTOR_INFLIGHT=4`. Then the console. Before the first key is
set: the legal amendments of Step 2b live; the week-one series in §10 on a dashboard that
can read them, which is what the catalogue work in Step 2a is for. Record the deploy in
`local/DEPLOY-<date>.md` as the others are, including the measured reranker time and that
`HF_HUB_OFFLINE=1` still holds.

## 8. Cost

### To build

Three repositories and the harness, in the steps above. Core is the largest piece: a new
package of four small modules, one keyword threaded through eleven files along a route three
commits already walked, two exceptions, seven series, and a subprocess test. Cloud is the
`_clone` change, two request fields with validators, one response object and two `Ranking`
fields on the wire and in the dashboard's types, one error code, three columns with a schema
bump, three settings, an executor, the catalogue entries, a Dockerfile step, and — in its own
step — one console field and the legal documents. Web is a FAQ entry and a paragraph. The
harness is one knob and one scoring script. No new runtime dependency anywhere.

### To measure

Every selector call in the parity work goes through the server on the gateway key, so it
costs what `extract.py`'s did: about $2.25 for 199 at gpt-5.4. The screen and the ranked
judged run share those calls, because the screen scores the search checkpoint the judged
run then answers from (§6). So: search with the selector $2.25; answer and evaluate on it
$1.11 (the reader and judge cost of each arm B run); the unranked twin's answer and
evaluate $1.11, its search free. About **$4.47**. Gateway balance after the replicate was
14.61 of the 29.90 the key started with; $15.29 of the user's $20 cap is spent and $4.71
remains under it — $0.24 to spare, thin enough that the cap is still the user's decision
before Step 3 starts. Running the screen as a separate pass, as the first draft had it,
would cost another $2.25 and not fit.

### To run

Memvara's cost per ranked call is $0 in money and about 5 to 6 s of one thread on the
selector executor, likely (§2). The customer's cost is their provider's. At the mean:
gpt-5.4 $0.01132 a call, mini $0.00350, nano $0.00093. The margin table, kept here so phase
2 is sized from it and not from memory. "Plain reads" is the `retrieval.query` allowance
(`plans.py:313-405`); daily allowances are shown for a 30-day month. "Phase 2 allowance" is
§4's proposal. The per-call figures are the file means, p95 and maximum from §2, so the
break-even counts here differ by a few units from the recommendation's, which used means
rounded to four decimals.

| tier | fee | plain reads/month | phase 1: memvara's model cost | phase 2 allowance | at gpt-5.4 mean | at gpt-5.4-mini mean | at gpt-5.4 p95 | at gpt-5.4 max | break-even calls gpt-5.4 / mini |
|---|---|---|---|---|---|---|---|---|---|
| free | $0 | 2,000 | $0 | 0 | — | — | — | — | 0 / 0 |
| personal | $9 | 700 a day, ~21,000 | $0 | 250 | $2.83 (31%) | $0.87 (10%) | $8.17 (91%) | $9.99 (111%) | 795 / 2,574 |
| personal_pro | $16 | 2,000 a day, ~60,000 | $0 | 500 | $5.66 (35%) | $1.75 (11%) | $16.33 (102%) | $19.98 (125%) | 1,413 / 4,576 |
| studio | $29 | 5,000 a day, ~150,000 | $0 | 1,000 | $11.32 (39%) | $3.50 (12%) | $32.67 (113%) | $39.96 (138%) | 2,562 / 8,295 |
| team | $99 | 50,000 a day, ~1.5M | $0 | 3,000 | $33.95 (34%) | $10.49 (11%) | $98.00 (99%) | $119.87 (121%) | 8,747 / 28,319 |
| business | $499 | uncapped | $0 | 15,000 | $169.76 (34%) | $52.44 (11%) | $490.02 (98%) | $599.33 (120%) | 44,091 / 142,744 |
| enterprise | contract | override | $0 | override | as agreed | | | | |

The mean columns are a projection from LongMemEval turn lengths. The p95 and maximum
columns are what the same allowance costs if every call is as long as the sample's long
ones: there is no prompt cap in the judged configuration and none ships, so on gpt-5.4 an
allowance sized at the mean is over the fee at the p95 on three of five tiers. That is why
phase 2, if it comes, is on gpt-5.4-mini. And the number that rules out paying for every
allowed read: if every `retrieval.query` on a tier were model-ranked at memvara's expense,
the model cost would exceed the fee on every paid tier at every model — personal at nano is
$19.61 a month against $9, 218% of the fee; at gpt-5.4 it is $237.67; free's 2,000 reads
would cost $22.63 at gpt-5.4 against a $0 fee.

## 9. Deliberately deferred

- **A memvara-paid allowance (phase 2).** On gpt-5.4-mini the proposed allowances cost 10 to
  12% of the fee at the mean, which is not what declines it. What does: the free tier's
  exposure has no aggregate bound; the limiter cannot refund a unit on fallback
  (`limits.py:1081-1082` releases only on >= 400); the DPA and `SUBPROCESSORS.md` ("No model
  provider") forbid it as a query flag until amended; mini has no judged end-to-end run; and
  on gpt-5.4 the p95 column in §8 shows an allowance sized at the mean over the fee on three
  tiers. Built only if decision 5's trigger fires after thirty days; sized from
  `retrieval.tokens_in` and `retrieval.model_refused{reason=unconfigured}`, not from the
  table above.
- **Ingest-time extraction.** The filter has only been measured with the question in hand;
  it is query-time selection, the same kind of step as the cross-encoder done by a stronger
  model, and "the cost per query is an order of magnitude above the reader's context". An
  ingest-time design is a different measurement nobody has made.
- **Role routing in core.** The rule fires on 19 of 22 assistant questions here and its
  precision on real user text is unmeasured; a false fire costs 0.918 coverage against a 0.790
  gain. It stays in the harness. §6's parity run is what decides whether the selector makes it
  unnecessary.
- **The 500-question run.** The user's decision stands: not until the 199 number is where
  they want it. A 3-point non-inferiority margin needs about 489 paired questions, so the
  91.5% is a 199-question number and the docs say so.
- **A judged run of `recall()`'s ranked block.** The judged block was turns only; `recall()`
  renders claims first (§3). What `memory_recall` returns on a ranked call is unmeasured, and
  measuring it is 199 reader and judge calls on a block built from `/v1/recall` — about
  $1.11 plus the selector calls — once the parity run has passed.
- **Adaptive rendering** (spans for kept turns that overflow, upgraded back to whole turns
  while room remains). Level with whole turns where it applied (51 against 50 on the 58
  span-rendered questions) but carried into arm E by a filter that over-keeps, and the two
  have not been separated on a judged run. The span field in §3 is what it will need.
- **The inclusive prompt and the top-60 list.** Recall 0.949 bought with precision on
  questions whose coverage was already complete: arm E's preference column fell from 10 of 12
  to 5 of 12 and the arm landed at 177, two below its predicted range.
- **gpt-5.4-mini as the default.** Filter recall 0.912 against 0.895 at a third of the cost,
  but a keep rate two and a half times higher and no judged end-to-end run. Selectable, with
  that qualifier, not default (decision 2).
- **The v2 answer prompt.** +3 on routed-720 on its own, 181 stacked on arm B against 182:
  the gain does not stack, and the pre-registered rule leaves it at no evidence.
- **A cross-encoder on every hosted read.** 84 ms a query at `top_n=20` against a ~3 ms
  search (`docs/ROADMAP.md:399-404`); 1.06 s mean at depth 200 on the study machine (§2).
  Under this design it runs only on ranked reads, on their own executor. Turning it on for
  plain reads is its own decision with its own latency measurement on the production host.
- **A bound on the reranker stage.** The 10 s timeout bounds the model call; nothing bounds
  the cross-encoder in front of it. Step 5 measures it on the production host first; a bound
  is added if the measurement says so, not before.
- **A selector setting for the self-hosted `memvara-mcp` server.** `server/config.py`
  builds its `Memvara` from `MEMVARA_LLM`, `MEMVARA_LLM_MODEL`, `MEMVARA_LLM_MAX_CLAIMS`
  (`config.py:185`, `215-216`) and `MEMVARA_EMBEDDER` (`config.py:77`) and has no read-side
  option. In phase 1 that
  server refuses `ranked` and its tool description says so; a `MEMVARA_SELECTOR_*` setting
  there, with `docs/DEPLOY.md` rows, is a separate piece of work.
- **The packaged skill's sentence on when to ask for `ranked`.** Its own commit and seven
  downstream pin bumps, owned by whoever takes it; not phase 1 (Step 1).
- **A per-request key header instead of a stored key.** The MCP door cannot add a second
  header and the console action is the written instruction the DPA asks for; see decision 1.

## 10. Risks, and the decisions the user owns

### Risks

- **The parity run is on a different candidate list than arm B.** Stated in §6 with its stop
  rule and its unranked twin. Do not carry 182 into any document until the shipped path has
  produced its own number.
- **Latency p95 is unknown, and the per-call figure is a sum of a measurement and an
  inference.** 1.06 s for the cross-encoder was measured on the study machine; 4.1 to 4.7 s
  for the model call is inferred from wall clock at concurrency 4; 20 of 199 prompts exceed
  8,000 tokens and likely take longer. The 10 s timeout and the fallback are the response
  to the model call, Step 5's measurement is the response to the reranker, and the first
  week's `retrieval.select_ms` p95 is the measurement. If fallbacks exceed 5% of ranked
  reads the timeout is wrong, not the feature.
- **A held thread starves plain reads.** Answered by the dedicated executor (§4); the check
  is `retrieval.latency_ms` on unranked reads before and after.
- **A production tenant hands the selector fewer than 40 turns.** The study's pool was
  almost all turns; a store with ordinary claim density is not. `retrieval.model_candidates`
  is the series that shows it.
- **Production turns may be longer than LongMemEval's.** Every cost number here is a
  projection from prompt mean 4,618 tokens and maximum 17,284; `retrieval.tokens_in` per
  call is what replaces it.
- **A removed key is used for up to 60 s.** By the clone TTL (§4). Stated in the console
  next to the remove action.
- **The design invariant changes.** Invariant 1 has been "nothing on the read path calls a
  model" since the library existed, and the mem0 comparison (2 write-path calls against 105)
  rests on the default path. The default is unchanged and the docs must say so in the same
  sentence that announces the opt-in, or a reader will take the headline claim as withdrawn.
- **The legal documents currently say the opposite of what this ships.** Three of them,
  named in Step 2b. They change in the commit that puts the key field in the console, and
  nothing deploys before that commit.
- **Friction.** A customer needs a provider account before the result is theirs. That is the
  cost of the design that keeps memvara's margin table at zero, and the phase 2 trigger is
  written on it.

### Decisions only you can make

The design above assumes the recommendation on each; the paragraph that depends on one is
tagged with its number.

1. **Store the customer's key, or take it per request?** Store per organisation, AES-256-GCM
   under `idp_key` (`control/idp/crypto.py`) or the org's CMK, fingerprint only in the
   console; or require a per-request header and store nothing. Recommendation: store. The
   console action is the written instruction and the MCP door cannot add a second header.
   Assumed in §4 "Key handling", "The column", and Steps 2a, 2b and 5.
2. **Default model.** gpt-5.4 (only judged end-to-end number; 3.2× mini's cost, paid by the
   customer) or mini (filter recall 0.912 vs 0.895, no judged run). Recommendation: gpt-5.4
   until a judged mini run exists; expose mini with its qualifier. Assumed in §4 "Key
   handling" and §6's stack.
3. **Free tier included?** Recommendation: yes. Memvara's cost is thread time under the
   in-flight cap; excluding it makes free the one place a visitor cannot see the result.
   Assumed in §4 "Key handling".
4. **Legal position, with counsel.** Whether a customer-keyed provider is the customer's
   processor (likely; no 30-day notice) or memvara's subprocessor (notice required), and
   whether a console action satisfies "agree in writing." Recommendation: amend DPA.md and
   SUBPROCESSORS.md in the commit that adds the key field, treat the console switch as the
   instruction, and let counsel settle the notice question. Step 2b waits on it; Step 2a
   does not.
5. **Phase 2 trigger.** Recommendation: build the memvara-paid allowance only if, after 30
   days, fewer than a fifth of active paid organisations have set a key while
   `retrieval.model_refused{reason=unconfigured}` is common. Size it from the series below,
   not from §8's table.

### Assumptions and the week-one check

| assumption | series to watch | fail signal |
|---|---|---|
| 5 to 6 s a call, no starvation of plain reads | `retrieval.select_ms` p50/p95; `retrieval.latency_ms` on ranked reads minus it (the reranker); `retrieval.latency_ms` on unranked reads, before vs after | unranked p95 rises; reranker cost far from Step 5's measurement |
| production turns are no longer than LongMemEval's (prompt mean 4,618) | `retrieval.tokens_in` per call | mean above 4,618 shrinks every phase-2 number proportionally |
| the selector sees 40 turns | `retrieval.model_candidates` per read | a median under 40 on a tenant with many turns |
| 10 s timeout and the providers are stable | `retrieval.model_fallback` by `reason` and `status`, over `retrieval.model_query + retrieval.model_fallback` | fallback share above 5%, or `provider` with 429 on one organisation daily |
| customers will bring a key | organisations with a key on file; ranked share (`retrieval.model_query + retrieval.model_fallback` over `retrieval.query`) per project; `retrieval.model_refused{reason=unconfigured}` | under a fifth of paid organisations with a key while `unconfigured` refusals are common |
| the gateway's `cost_usd` is what a provider bills | one design partner's provider invoice line vs their `retrieval.tokens_in/out` sum | ratio far from $2.25/M prompt, $13.50/M completion |
| the two caps are enough | `retrieval.model_refused{reason=inflight}` per organisation | any organisation hitting it daily |

## 11. Review disagreements

Findings not applied, or applied differently from what the review asked, and why.

- **Pricing review, finding 5** ("$2.25 plus $1.11 is $3.36, under $4.71 with $1.35 to
  spare"). Not applied as stated. The $1.11 is the reader-and-judge cost of an arm B run,
  where the selector ran offline; on the shipped path the selector runs through the server
  on the same gateway key, so the judged run's search stage is itself 199 selector calls at
  about $2.25. The first draft's "screen or replicate but not both" was right about two
  separate passes ($5.61). §6 and §8 now share the calls between the screen and the judged
  run and add the unranked twin, which is $4.47 — under the cap by $0.24.
- **Correctness review, finding 10**, the clause "likely slower than the study machine".
  Not carried. The production host's reranker time is unmeasured, and Step 5 measures it;
  nothing read here says which way it differs.
- **Completeness review, finding 15** (the test count). The number is dropped and Step 1
  re-derives it; the reviewer's 4,336 is not repeated here because it was not reproduced in
  this rework.
- **Completeness review, finding 19** (restatement). Applied in part. §1 no longer repeats
  the per-call costs, and §10 no longer claims to carry the recommendation verbatim, since
  its tables changed. The margin table in §8 and the five decisions in §10 stay in full:
  the table is the one place phase 2 is sized from, and the decisions are what this
  document asks the user for.
- **Completeness review, finding 6** offered two fixes; the refusal was taken over a
  `MEMVARA_SELECTOR_*` setting for the self-hosted server, and the setting is listed in §9
  rather than built, so the phase-1 scope is visible rather than reduced in passing.
