# Model-ranked recall — an opt-in read mode on the customer's own key

**Date:** 2026-09-04, reworked the same day after three reviews (correctness, pricing,
completeness) and again after the code review of PR #169, whose sixteen findings are
applied below. Nothing in this document is built. It is written from the judged runs in
`docs/superpowers/plans/2026-09-04-model-as-ranker-720.md` (on the benchmark branch, not
on main), the per-call measurements under `local/compress/` and `local/pool/` in the same
worktree, and the code as it stands on three main branches: agent-memory `origin/main` at
`2d9bcb5` (the local `main` ref, `05155a3`, is stale and lacks `anchored`), memvara-cloud
`main` at `912ecd9`, memvara-web `main` at `bddba0a`. The memorybench harness is read at
`82cf64a` on branch `memvara-budget-arms`. Line numbers below are from those refs.

One term for the component throughout: the **selector** is the stage that hands a model
the candidate turns and takes back the ones it names. Arm B of the benchmark is called
"model as ranker" in the plan; that is the selector, judged offline. "Filter" below means
`memory_types` only, and "routing" means the role rule — the harness's originally, and
core's as well since §6's stop rule fired (§3, §9).

## 1. The answer

**Ship one opt-in read mode, `ranked`, on the customer's own model key.** After memvara's
own retrieval — hybrid fusion, then a cross-encoder over the top 200 turns — one chat call
receives the top-40 candidate turns and the question, and names the turns that bear on it
with a span for each. Those turns come back first, whole, with the span on each as a
field. The default read path does not change: no model, deterministic order, identical
cost. The hosted service serves a ranked read unranked, and says so on the result, until an
organisation puts a key on file, so memvara pays $0 per call and no plan gets a new metric,
a new price or a new per-tier number. A ranked read counts as one ordinary
`retrieval.query`.

**The judged configuration scores 182 of 199 (91.5%) at a median context of 672 tokens, on
two runs with identical inputs.** That configuration was the routed role's top-40 with the
model call made offline (§2, §6). The shipped path has not been run. The number the
documentation carries is the parity run's (§6, Step 3), not 182.

What memvara spends per ranked call is one thread for about 5 to 6 s, likely: 1.06 s mean
for the cross-encoder at depth 200, measured on the study machine, plus a model call whose
mean is inferred at 4.1 to 4.6 s and whose p95 is unknown (§2). One process-wide admission
cap on ranked reads, `MEMVARA_SELECTOR_INFLIGHT`, and a 10 s deadline on the model call
bound it (§4). Per-call cost to the customer is in §2 and §8.

The design is written to the five decisions taken on 2026-09-04 (§10): the customer's key is
stored per organisation; the default model is gpt-5.4-mini, with gpt-5.4 selectable; the
free tier may set a key; counsel settles the legal position before the console field ships;
and phase 2 is decided from thirty days of series. Every paragraph that depends on one says
which. A memvara-paid allowance is phase 2, decided from thirty days of the series
phase 1 emits (§9).

## 2. What is established

**The ranking is what wins, not the compression.** Arm B — the selector's kept turns
rendered whole and first, then the rest of the reranked list, greedy to 720 tokens — scores
182 of 199 on two independent runs of the same block (replicate: 182, differing on 4
questions, 2 each way, against a reader noise floor of about 15). Arm A, the same model's
spans rendered without the turns around them, scores 165 at 77 median tokens and loses
preference questions wholesale (4 of 12). So the reader needs whole turns, and what it
needed at 720 tokens was the right ten.

| arm | correct | % | median tokens |
| --- | --- | --- | --- |
| control (cap 15, both roles) | 172/199 | 86.4 | 4,089 |
| routed-720 | 171/199 | 85.9 | 672 |
| routed-720 + prompt v2 | 174/199 | 87.4 | 672 |
| **B: model as ranker (the selector)** | **182/199** | **91.5** | 672 |
| A: spans only | 165/199 | 82.9 | 77 |
| C: B + overflow spans + prompt v2 | 181/199 | 91.0 | 706 |
| E: inclusive prompt, adaptive rendering | 177/199 | 88.9 | 706 |
| **B, replicate** | **182/199** | **91.5** | 672 |

Per type, arm B and its replicate: single-session-user 27/28 and 28/28, single-session-
assistant 21/22 and 22/22, preference 10/12 and 9/12, multi-session 47/53 and 47/53,
temporal 48/53 and 47/53, knowledge-update 29/31 both. The gains sit where the budget had
been cutting: multi-session and temporal questions whose gold turns ranked 14 to 37 in the
cross-encoder order. Six of the fourteen budget-cut questions are correct now.

**The selector's own numbers, measured against the gold labels before anything was
judged.** Over the routed role's top-40 turns in cross-encoder order:

| selector | list | gold recall | non-gold keep | cost per 199 |
| --- | --- | --- | --- | --- |
| gpt-5.4, precise prompt (arm B's) | top-40 | 0.895 (315/352) | 2.5% | $2.25 |
| gpt-5.4, inclusive prompt | top-60 | 0.949 (335/353) | 2.8% | $3.25 |
| union of both passes | top-60 | 0.958 | 2.9% | $5.50 |
| gpt-5.4-mini, precise prompt | top-40 | 0.912 (321/352) | 6.4% | $0.70 |
| gpt-5.4-nano, precise prompt | top-40 | 0.844 (297/352) | 4.0% | $0.19 |

The kept span is a fifth of its turn's tokens at the median. Zero parse failures in 199
calls on every file. **The span is verbatim in 523 of the 528 kept spans** on the gpt-5.4
file: 489 are substrings of their turn as returned, 34 more are substrings once the
timestamp prefix the model copied from the excerpt line is stripped, and 5 are paraphrases.
So the renderer must tolerate a span that is not a substring (§3, "The protocol").

**Per call, from the gateway caches** (`local/compress/extractions{,_mini,_nano}.jsonl`,
199 rows each, cost exactly linear in tokens: $2.25 per million prompt tokens and $13.50 per
million completion tokens for gpt-5.4). Every p95 computed on the 199-call files is the
nearest rank, the 190th of 199 sorted values; the cross-encoder's 1.209 s below is a p90
read from `ce.meta.json`, and the p50/p95 series in §10 are to be measured in production,
not taken from these files. The prompt is identical across the three top-40
files: median 3,325 tokens, p95 14,609, mean 4,618, maximum 17,284; 20 of 199 prompts exceed
8,000 tokens. Completion tokens are small: median 62, p95 133, maximum 259 for gpt-5.4;
median 81, maximum 315 for mini; median 69, maximum 446 for nano. Cost per call from the
`cost` field: gpt-5.4 median $0.00869, mean $0.01132, p95 $0.03348, maximum $0.03996;
mini $0.00267, $0.00350, $0.01015, $0.01213; nano $0.00072, $0.00093, $0.00270, $0.00323.
The prompt distribution is right-skewed: the mean cost is 30% above the median (the mean
prompt length 39% above its median), and a bill accrues at the mean. Nothing capped the
prompt in any run, so the maximum is a measured value on this sample and not a ceiling:
production turns can be longer. One offline check on the same file bears on that
(§7, Step 3): a 500-character cap on each excerpt would have cut 14 of the 528 kept spans,
and a 2,000-character cap would have cut none.

**Latency.** Wall clock at concurrency 4: 204 to 227 s for 199 calls, so 1.0 to 1.1 s per
call of throughput (204/199 and 227/199) and, if the four workers stayed busy, a mean
model-call latency of 4.1 to 4.6 s (204 × 4 / 199 and 227 × 4 / 199). Per-call latency was
not measured and its p95 is unknown. The cross-encoder was measured: `local/pool/ce.meta.json`
records `cross-encoder/ms-marco-MiniLM-L-6-v2` at batch 64 over 39,430 pairs for 199
queries — about 198 pairs a query — at a mean of 1.062 s and a p90 of 1.209 s per query on
the study machine, model load excluded. So a ranked read holds a thread for about 5 to 6 s,
likely, and the deadline in §3 bounds only the model-call part.

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
production, so no model is called anywhere and every token series is structurally zero.
Five places state that nothing on the read path involves a model, and this design amends
every one of them with the opt-in: the library's design invariant 1 (`docs/INTERNALS.md:33-46`),
`README.md:488-489` ("not even the optional reranker"), `hybrid.py:36-40`'s module docstring
("No LLM sits on the read path"), and two texts a model reads at runtime — the
`memory_recall` tool description at `server/tools.py:1488-1489` ("Call it speculatively; it
is cheap, local, and involves no model") and the server's `INSTRUCTIONS` at
`server/mcp.py:50-51` ("it is local, cheap, and involves no model"), which is sent in the
`initialize` result and placed in the client's system prompt (`mcp.py:44-46`).

## 3. The design in core

### The option and the switch

A constructor option in the shape of `read_reranker`, and a per-call switch on `search()`
and `recall()`:

```python
from memvara import Memvara
from memvara.llm.openai import OpenAILLM
from memvara.rerank import CrossEncoderReranker
from memvara.select import ModelSelector

mem = Memvara("memory.db",
              read_reranker=CrossEncoderReranker(), read_rerank_top_n=200,
              read_selector=ModelSelector(llm=OpenAILLM(model="gpt-5.4-mini", client=client),
                                          top_n=40, timeout=10.0))
block = mem.recall("what did they say about the trip", include_episodes=True, ranked=True)
```

`read_selector` routes by prefix to `HybridRetriever(selector=...)` like every other
`read_*` option (`memvara/core.py:638`, `811-813`). The selector carries the model call and
`top_n`; the retriever carries the reranker and its depth, so one cross-encoder serves plain
reads and ranked reads alike (§4 says why that matters). `ranked` is a keyword-only `bool`,
default `False`, on `HybridRetriever.search`, `Memvara.search`, `Memvara.recall`,
`ScopedMemvara.search`/`.recall`, their async twins, and `RemoteMemvara` — the route
`anchored` took across three commits: `c6e4a91` (`hybrid.py`, `core.py`, `aio.py`),
`cf7e526`, and `3622eb0` (`remote/api.py`, `remote/aio.py`, `remote/hydrate.py`,
`server/memory_api.py`, `server/tools.py`). The remote client sends `ranked` only when set,
so a server from before the field refuses with 422 (`_Model` sets `extra="forbid"`,
memvara-cloud `rest/models.py:50`) rather than answering unranked as though it had
honoured it. That 422 is the one refusal `ranked` inherits from `anchored`; a server that
knows the field never refuses it (below). A hosted `Memvara(api_key=...)` refuses
`read_selector` as it refuses every `read_*` option (`core.py:654-674`, `Memvara.__new__`):
the selector runs server-side.

One new tuning beside it: `read_rerank_ranked_only: bool = False`. When true, a plain
read (`ranked=False`) behaves exactly as it would on a retriever with no reranker, and a
ranked read runs the reranker as today. "Exactly" is the whole of the retriever's depth
arithmetic, not the rerank call alone: `hybrid.py:613` sets `depth = k if self.reranker is
None else max(k, self.rerank_top_n)` and `:615` sets `limit = max(depth *
candidate_multiplier, depth)`, so with the switch on and `ranked=False` the depth is `k`,
the legs gather at `k` times the multiplier, the hydration follows, and the rerank call at
`647-653` is skipped. Gating only the rerank call would leave every hosted plain read
gathering at limit 1,000 and hydrating 200 candidates for a stage that never runs. The
hosted service needs this: the shared cross-encoder has to sit on the base handle so that a
clone rebuild loads no model (§4), and without the switch every hosted plain read would pay
the 1 s the cross-encoder costs at depth 200, which §9 defers as its own decision. The
benchmark stack leaves it false, because reranking every read is what it already runs. The
alternative, a second reranker slot on the retriever for the ranked path alone, is two
objects for one model and was not taken.

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
nothing changes. On a ranked call, in this order:

1. **No selector means no model, decided first.** A retriever with no `read_selector`
   serves the read unranked, before any leg runs differently, with the outcome
   `unconfigured` on the result (below). Nothing is spent on it. On the hosted service
   `_clone` adds no selector for an organisation without a key (§4), so such an
   organisation ends here and never reaches admission: `disabled` and `inflight` are
   outcomes a keyed organisation can meet, and only a keyed one.
2. **The turns are gathered at the reranker's depth and kept apart from the claims.** Today
   `depth = k` without a reranker, else `max(k, rerank_top_n)` (`hybrid.py:613`);
   `_episodes` returns at most `max_episodes` turns, default 3 (`hybrid.py:298`, `1197`);
   and `_interleave` merges claims and turns by score and cuts the *combined* list to
   `depth` (`hybrid.py:1280-1298`, `return out[:k]`) after `w_episode=0.5` has discounted
   every turn (`hybrid.py:297`, `1173`). Lifting the episode cap alone would not give the
   selector 40 turns: on a tenant with ordinary claim density the head would be mostly
   claims. The study never met this because its pool was 39,430 turns against 370 claims
   (`local/pool/pool.meta.json`). So on a ranked call `_episodes` returns up to
   `rerank_top_n` turns, and that list is the selector's candidate list; it does not go
   through `_interleave`'s cut. Claims come from `_rank` at `depth` as today.
3. **The selector is asked for admission**, `selector.admit()`, a context manager held from
   here until the model has answered. The library's own `ModelSelector` admits everything;
   the hosted wrapper in §4 holds the process-wide cap here, so the thread the cap exists
   to bound is bounded before the expensive stage runs, not after it. An `admit()` that
   raises `SelectorRefused` (below) serves the read unranked; one that raises
   `SelectorBusy` ends the read.
4. **The turns are ordered** by the retriever's reranker over the turn list at
   `rerank_top_n`. With no reranker the turns keep the episode leg's own order — which is
   not the judged configuration, and the option's docstring says so.
5. **The reranked list is routed before it is cut to `top_n`.** `retrieve/intent.routed_role`
   — model-free, the fourteen-phrase rule fitted on LongMemEval's single-session-assistant
   phrasing (§6, §9) — says whether the question is about something the assistant said or
   something the user said, `_run_ranked_stage` drops every turn of the other role from the
   reranked list, and only then is the survivor list cut to `top_n`; a routed role with no
   turns at all falls back to the other role's list rather than handing the selector
   nothing. This is in core, not only in the harness that measured it, because the
   candidate list matters: the offline screen over both roles' top-40 scored 0.808 gold-turn
   recall against 0.912 for the same model over the routed role's own top-40 (§6, check 1),
   under this design's 0.85 floor. **The selector sees the first `top_n` turns of that
   routed list** and names the ones it keeps. The count it was handed travels on the result
   as `selection.candidates` (below), so a tenant whose store yields fewer than `top_n`
   turns of the routed role is visible to the caller rather than silent.
6. **The order that comes back:** the kept turns first, in reranked order, whole, and
   **outside `k`** — then the remaining turns and the claims interleaved as today, cut to
   `depth`, then `[:k]`; then `_observe`. So `search(k=8, ranked=True)` returns up to
   `top_n` kept turns and then eight others, and `k` bounds the claims and the unkept turns
   as it does now. Claims therefore sit after the kept turns and before or among the unkept
   turns by score, as `_interleave` places them. The retriever's own reranker stage is not
   run again on a ranked call: the reranker has already ordered the turns, which is the
   work the stage exists for. One rule, for `search()` and `recall()` alike: a kept turn
   is never cut by `k`. The alternative — `[:k]` over everything, with `recall()` asking
   for `k + top_n` — would make `k` mean two things in one call.

The judged configuration is the cross-encoder at depth 200 and the selector over the top-40,
so `read_reranker=CrossEncoderReranker()`, `read_rerank_top_n=200` and
`ModelSelector(top_n=40)` is what ships as the default of the mode, and it is what the
benchmark stack already runs for the first two.

### The protocol

`Reranker` cannot express this: it returns exactly one score per document, never drops, and
must be deterministic (`memvara/rerank/base.py:29-44`; `rerank/stage.py:82-105`). Adding a
member to the `runtime_checkable` `LLM` protocol would break older implementations, which is
the reason `RelationComposer` is its own protocol (`CHANGELOG.md:2793-2794`;
`retrieve/compose.py:68-84`). So `memvara/select/base.py` defines a new one, with two
members and nothing else:

```python
@dataclass(slots=True, frozen=True)
class Candidate:
    id: str            # episode id
    when: datetime     # the turn's timestamp
    text: str          # the whole turn

@dataclass(slots=True, frozen=True)
class Selected:
    id: str
    span: str | None   # substring of the candidate's text, or None

@dataclass(slots=True, frozen=True)
class Selection:
    outcome: str                # applied | fallback | unconfigured | disabled | key_rejected
    reason: str | None = None   # fallback only: timeout | error | provider | malformed
    status: int | None = None   # the provider's HTTP status, when there was one
    candidates: int = 0         # turns handed to the selector; 0 when it never ran
    kept: int = 0               # turns it named

@runtime_checkable
class Selector(Protocol):
    def admit(self) -> AbstractContextManager[None]: ...
    def select(self, question: str, candidates: Sequence[Candidate], *,
               asked_on: datetime | None = None,
               usage: Usage | None = None) -> Sequence[Selected]: ...

class SelectorRefused(Exception):     # the read is served unranked; reason in {disabled, key_rejected}
    reason: str
    status: int | None
class SelectorBusy(Exception): ...    # the read is not served; the caller renders the refusal
```

`admit()` is the second member because admission has to precede the cross-encoder: a cap
taken inside `select()` would bound the model call and leave the second in front of it
unbounded, which is the thread the cap exists to bound. The simplification the review
offered — a protocol of `select()` alone — was taken for the three flags (`name`,
`is_noop`, `reports_usage`), which are gone, and not for this.

Input: the question, the date it is asked on, and the scored candidate turns in reranked
order. Output: the ids the model kept, each with its span. The stage lives in `hybrid.py`
with the rest of the read order, takes `Rankable` items (`rerank/stage.py:39-51`) and
receives only `EpisodeResult` items; `Result` (claim) items never reach it. It orders the
kept turns by their reranked rank, not by the order the model listed them, because arm B
rendered "kept turns whole, in rank order" and that is the measured thing. A span is kept
as returned when it is a substring of its turn, and after stripping a leading
`(YYYY-MM-DD HH:MM) ` when that makes it one — the 34 of 528 the model copied the excerpt's
timestamp into; otherwise it is replaced by `None` and the turn is still kept: the ranking
is what was judged, the span is a courtesy, and 5 of 528 were paraphrases. An id the model
invents is ignored. The span stays on the result even though nothing in phase 1 renders it:
it is on the wire for the adaptive rendering §9 defers, and the field order of
`Explanation` is an API, so adding it later would be a second appended change.

### The prompt and the transport

`ModelSelector` sends exactly the messages `local/compress/extract.py` sent, because that
is the prompt the 182 was measured with. The system message, verbatim from `extract.py`:

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
numbered from 1. The request is a chat completion with `response_format:
{"type": "json_object"}` and `max_completion_tokens: 400` — not `extract.py`'s 4,000. The
cap cannot change an answer that fits under it, and every gpt-5.4 answer in the sample did
(maximum 259; mini 315; nano 446, so nano would be cut and is not offered). What is
byte-identical is the two messages; the request parameters are not, and the Step 1 fixture
asserts the messages. The reply's `kept` list is parsed as `extract.py:62-73` parses it,
through the helpers the write path already has: `_shape.parse_json_object` (`llm/_shape.py:72-86`)
for the object, and `_shape.source_index(i - 1, n)` (`_shape.py:128-133`, 0-based, so the
1-based excerpt number is shifted first) for each entry; a malformed entry is skipped, an
out-of-range `i` or an empty span is dropped. A reply that is not JSON is a fallback
(below), where the study recorded it as an empty kept set — it happened zero times in 199
calls.

The transport is the configured backend's, not a client of the selector's own. The two
backends in `memvara/llm/` (`openai.py`, `anthropic.py`) already make the chat call through
their SDK client, read usage through `_shape.record_usage` (`_shape.py:288-330`, whose
docstring names the field-drift failure a hand-rolled reader would repeat), and lazy-import
the SDK with the pip-extra message and `client=` injection (`openai.py:107-118`). Each gains
one method, `chat(system, prompt, *, json_object, max_completion_tokens, timeout, usage)`,
declared in a small `Chat` protocol beside `LLM` in `llm/base.py` rather than on `LLM`
itself, for the `RelationComposer` reason above; `NullLLM` has none, and
`ModelSelector.__init__` refuses an `llm=` without `chat` with a `TypeError` naming the
extra. `extract`'s `_call` (`openai.py:123-146`) hard-codes strict `json_schema` and passes
`temperature`; `chat` sends `json_object` and no temperature, so the request matches
`extract.py`'s. The core install gains no dependency: `dependencies = ["numpy>=1.24"]` is
pinned by exact equality and adding one is a product decision (`CONTRIBUTING.md`, Scope);
the selector needs the `openai` or `anthropic` extra, and the hosted image installs
`memvara[anthropic,openai]` already (`Dockerfile:103`). `tests/test_rerank.py:379-429`
already asserts in a subprocess that the default configuration imports no reranker
backend; `memvara.select.model` joins that list.

**The timeout is a deadline on the whole call.** `timeout` (10 s by default;
`MEMVARA_SELECTOR_TIMEOUT_S` in the hosted service) is wall-clock from the moment the
selector starts the call to the moment the answer is in hand — connect, request and
response together — not a per-socket-operation limit, which is what `urllib`'s and the
SDK's `timeout` are and which fires after the request was sent and the bill incurred. The
selector starts a clock, hands the remaining time to the client as its per-phase timeout,
and reads the clock again on return; an answer that lands after the deadline is a
`timeout` fallback even though it arrived. In the ordinary case (one connect, one read) the
wait is bounded by the deadline; a connect phase that itself takes seconds can add up to one
more, and the week-one `retrieval.select_ms` p95 shows whether that case exists. No
retries: the study retried four times at 180 s because nothing was waiting on it; a request
is. **A call that times out was still made, and the provider bills it**: the customer pays
for a read that was served unranked, which is why `retrieval.model_fallback{reason=timeout}`
is read as cost as well as latency (§10). A provider answer of 401 or 403 is not a
fallback: the selector raises `SelectorRefused("key_rejected", status)`, because a revoked
key served unranked for a month with nothing saying so is the failure that must not hide.
Every other HTTP error (402, 429, 5xx) is a fallback with its status recorded.

### How the result travels

`Explanation` is a slots dataclass whose field order is an API (`types.py:837-903`). Two
fields are appended after the last existing one, `anchor` (903):

- `selected: bool | None = None` — `True` the model named this turn, `False` it saw the turn
  and did not, `None` the selector did not see it (a claim, a turn past `top_n`, a plain
  read, or a read the model did not answer).
- `span: str | None = None` — the span, only ever set when `selected` is `True`.

**The outcome travels on the return value.** `HybridRetriever.search` returns
`SearchResults`, a `list` subclass with one attribute, `selection: Selection | None` —
`None` on a plain read. It is still a list, so every caller that indexes, iterates or
serialises the result is unchanged, and an empty ranked result still carries its outcome,
which a per-item field could not. `Memvara.search` and `ScopedMemvara.search` return it
unchanged. The remote client builds one from the wire: today `RemoteMemvara.search` returns
a plain list, `[_hit(h) for h in body["results"]]` (`remote/api.py:359`; the aio twin at
`remote/aio.py:207`, and the scoped classes at `api.py:894` and `aio.py:541`), and each
instead returns a `SearchResults` whose `selection` is read from the body's `selection`,
`None` when a server does not send one. Two alternatives were declined:
`selection` on every item, identical on all, which loses the outcome on an empty result and
makes the hydrator read it off the first row; and widening the hosted `TokenObserver`
(`rest/limits.py:384-391`), which forwards ints, is `/v1`-only, and would not reach `/mcp`.

`search()` returns the kept turns first, whole, outside `k`, then the rest as "Where it
sits" orders them; a caller that wants the study's block renders the turns in that order and
fills its own budget.

`recall()` renders claims first and turns as a capped tail, never interleaved, and cuts each
turn to `RECALL_EPISODE_CHARS = 280` (`core.py:2087`, `2195-2207`, `2355`). Its docstring
guarantees that "a weak turn never costs a fact its place" (`core.py:2202-2207`), and a
ranked call keeps that guarantee by construction: **`k` bounds the facts and `budget`
bounds the kept turns.** `recall(ranked=True)` makes one `search(ranked=True)` call and
takes the claims from it as today — the `k` slots, with today's discount rule for an unkept
turn — and the kept turns, which arrived outside `k`. The tail leads with every kept turn
rendered **whole**, in reranked order — the 280-character cut does not apply to them,
because a cut turn is arm A's failure mode, not arm B's block — and the unkept turns follow
at 280 characters up to `max_episodes` as today. `budget` trims the block from the end as
it does now, turns before facts, and the `RECALL_DROPPED` line (`core.py:2104`) counts
every kept turn that did not fit along with everything else it counts. So the turns block
of a ranked `recall()` is arm B's block: kept turns whole, first, then the rest of the
reranked order, to the budget. The facts block in front of it is the one thing the judged
runs did not have, and the run that measures the two together is listed in §9. The MCP
`memory_recall` tool returns `recall()` text verbatim (`server/tools.py:450-461`), and the
REST `/v1/recall` route returns the same text, so both doors reproduce that block.

`RecallResult` (`types.py:1034-1073`) gains `selection`, the same record, so a caller that
asked `with_ids=True` reads the outcome without parsing prose. The text block is the signal
for everyone else: **a ranked `recall()` block that the model did not rank ends with a
`RECALL_UNRANKED` line**, in the shape of `RECALL_DROPPED`, naming the outcome — a model
reading the block, and a person reading a transcript, see that the order is the plain one.
Over MCP that line is the whole signal; a tool result is one text block
(`server/mcp.py:296`, `313-315`) and carries no structured object.

The hosted wire carries the record: `SearchResponse` and `RecallResponse`
(`rest/models.py:457-472`) gain `selection`, null on a plain read. `Ranking`
(`rest/models.py:389`) gains `selected` and `span`, and `rest/render.py:155` renders them.
On the client side the three fields are read in two places, because they live in two
places on the wire. `remote/hydrate.py:164-177` builds an `Explanation` from one ranking
object, so it reads `selected` and `span` when present and leaves them `None` when a
server does not send them, as it does for `anchor`. `selection` sits on the response
beside `results`, not on any ranking, so `RemoteMemvara.search` reads it off the body and
returns a `SearchResults` carrying it (above); the library's `recall(with_ids=True)` fills
`RecallResult.selection` from the same record. The remote client's `recall()` returns text
and takes no `with_ids` (`remote/api.py:12-14`, `:390`), so there the trailing
`RECALL_UNRANKED` line is the signal, as it is over MCP; `RecallResponse.selection` is for
a caller of the route. The span is on the search result so a caller can render either the
turn or the span; rendering spans inside `recall()` is the adaptive rendering deferred in
§9.

### The outcomes

Six ways a ranked read can end, one rule: **it is served whenever a read can be served, the
result says what happened, and the only refusal is the one that protects the process.**
Refusing a flag the server cannot honour was the first draft's answer and is not this
one's: it would make `ranked` the one request field that fails on a server without a key,
when `anchored`'s precedent (`rest/models.py:1051-1063`; `app.py:846`, `913`) is a knowing
server that always answers, and the outcome field says what it could not do.

- **Applied.** The model answered; `selection.outcome` is `applied`, `selected` is set on
  every turn it saw, and the kept turns lead.
- **Served unranked: no selector.** `ranked=True` on a retriever with no `read_selector`:
  the plain order, `selected` `None` throughout, `selection.outcome` `unconfigured`, one
  `retrieval.model_refused{reason=unconfigured}`. On the hosted service that is an
  organisation with no key on file; on the self-hosted `memvara-mcp` server it is every
  ranked call in phase 1 (§9).
- **Served unranked: the operator's switch.** `admit()` raises `SelectorRefused("disabled")`
  (the hosted wrapper does, at `MEMVARA_SELECTOR_INFLIGHT=0`): the plain order, outcome
  `disabled`, one `retrieval.model_refused{reason=disabled}`, and nothing spent on the
  cross-encoder, because admission comes first. Only a keyed organisation can meet this
  outcome: one without a key has no selector and was decided `unconfigured` at step 1,
  whatever the switch says.
- **Served unranked: the key.** The provider answers 401 or 403: outcome `key_rejected`
  with the status, one `retrieval.model_refused{reason=key_rejected}`, and one
  `retrieval.select_ms`, because a call was made. Distinct from a fallback on purpose: the
  fallback reasons are transient and this one is not, and §10 watches it on its own.
- **Served by fallback.** The model call passes its deadline, fails to connect, returns any
  other HTTP error, or returns something that is not JSON: the plain order, `selected`
  `None` on every item, `selection.outcome` `fallback` with `reason` in `timeout`, `error`,
  `provider`, `malformed`, and `retrieval.model_fallback` with the same tags. The fallback
  is never an empty result.
- **Refused: the cap is full.** `admit()` raises `SelectorBusy`. The stage emits
  `retrieval.model_refused{reason=inflight}` and re-raises; nothing is served and
  `_observe` never runs, so the core `retrieval.query` series is not emitted, and on the
  hosted service the `retrieval.query` quota reservation taken before the call is released
  (§4). The hosted `/v1` handler renders it as 429 with `Retry-After`; over MCP `_recall`
  turns it into a `ToolError` that says to retry in a few seconds, and the hosted `/mcp`
  transport answers that tool result as HTTP 429 with the same header (§4). In the library
  `ModelSelector.admit()` never raises it.

In every served case the block or the response says which outcome it was, and every outcome
is counted once, from core: the first draft had the hosted handler count refusals because
they happened before `_observe`, and the MCP door has no such handler (§4).

### Counting

Six series, named like the write side (`telemetry.py:175`, `188-189`, `242`) and picked up
by `series_names()` automatically because they are dotted upper-case constants in
`memvara/telemetry.py` (`telemetry.py:683-699`), so the quota, admin and metering tests that
iterate `series_names()` see one list. Each has one rule, and the rule says what it does on
a read the model did not answer:

| constant | series | kind | rule |
| --- | --- | --- | --- |
| `RETRIEVAL_MODEL_QUERY` | `retrieval.model_query` | counter | one per ranked read the model **answered**; never on a fallback or a served-unranked read. The read-side twin of `write.llm_calls`, counting model consultations (`llm/base.py:70-73`). No tag: this is the series a phase 2 allowance would sum, and quota sums a source by name and ignores tags (`memvara-cloud quota/engine.py:363-367`, `metric = ANY(%(sources)s)`), so a fallback must not share its name |
| `RETRIEVAL_MODEL_FALLBACK` | `retrieval.model_fallback` | counter, tags `reason`, `status` | one per ranked read served by fallback, and only then; `reason` in `timeout`, `error`, `provider`, `malformed`; `status` the provider's HTTP status when `reason` is `provider` |
| `RETRIEVAL_MODEL_REFUSED` | `retrieval.model_refused` | counter, tag `reason` | one per ranked read the selector did not run for; `reason` in `unconfigured`, `disabled`, `key_rejected`, `inflight`. The first three are served unranked; `inflight` is the one that is not served |
| `RETRIEVAL_TOKENS_IN` | `retrieval.tokens_in` | counter | only when the `Usage` accumulator reports > 0, exactly `_report_usage`'s rule (`write/pipeline.py:735-750`): a zero would understate a bill in the direction that favours us. A timed-out call carries no usage block and so adds nothing here, though the provider bills it |
| `RETRIEVAL_TOKENS_OUT` | `retrieval.tokens_out` | counter | same |
| `RETRIEVAL_SELECT_MS` | `retrieval.select_ms` | timing | the model call only, on **every call that was made, whatever it returned**: answered (`applied`), timed out or errored (`fallback`), or refused by the provider (`key_rejected`, a 401 or 403 that was still a call). The `write.extract_ms` rule (`telemetry.py:231-242`: "includes the request that raised, because a provider timeout is latency the caller waited through"). Not emitted when no call was made: `unconfigured`, `disabled`, `inflight`. On a read that carries it, `retrieval.latency_ms` minus this is the reranker's cost |

The first draft had a seventh, `retrieval.model_candidates`, a counter carrying the number
of turns handed over. It is dropped: the hosted recorder keeps a counter as a running sum
with no histogram (`metering/recorder.py:522-531`, `203-205`; `query.py:65-67`), so the
"median under 40" it was to be watched for cannot be read from it. The count is on the wire
as `selection.candidates` and the parity run asserts it at 40 (§6).

They are emitted from the same recorder `_observe` uses, at the end of the ranked call.
`retrieval.query`, `retrieval.results` and `retrieval.latency_ms` are emitted as today, once,
so a ranked read is one query in every place that counts queries. The ranked share of reads
on a project is `retrieval.model_query + retrieval.model_fallback` over `retrieval.query`;
`retrieval.query` itself carries only a `script` tag and cannot tell the two apart.

## 4. The design in memvara-cloud

### The per-request flag

`ranked: bool = False` beside `anchored` on `SearchRequest` (`rest/models.py:1051-1063`) and
`RecallRequest` (`1105-1110`), passed through at `rest/app.py:844-848` and `911-914`; the
recall handler asks for `RecallResult` (`with_ids=True`) so it can put `selection` on the
response beside the unchanged text. A `model_validator` on both models refuses `ranked`
with `include_episodes` false or `memory_types` set as 422 `invalid_request`, so neither
contradiction reaches `ctx.run` — where a library `ValueError` would surface as a 500,
since only `_axes` and `_states` wrap one into a 400 (`app.py:316-324`, `352-358`). Not on
`AskRequest`: `/v1/ask` composes over slots and renders every sentence from a stored column
(`app.py:1189`, "Nothing here consults a model"), and the selector acts on turns, so the
flag would have nothing to act on there — a flag that does nothing is a lie in a request
model.

The flag goes in the body, not the query string, because it selects no allowance and no
weight, so nothing needs it before the body is parsed. `depth` on the graph routes is a
query parameter for exactly the opposite reason (`ratelimit/policy.py:279-284`): the limiter
resolves an operation before the route runs.

### The MCP door

`/mcp` reaches the same per-project handle `/v1` does (`rest/mcp.py:232`, `memory_for` at
`deps.py:198-209`) and never builds a `RequestContext`, so nothing a `/v1` handler does
happens on a tool call — which is why every outcome is decided and counted in core (§3),
where both doors arrive.

In phase 1 **`memory_recall` takes `ranked` and `memory_search` does not.** `_search`
calls `ctx.memory.search` without `include_episodes` (`server/tools.py:423-431`), its schema
has no such property, and `MemoryAPI.search` pins `include_episodes: Literal[False]`
because `_search` reads `.claim` on every row (`server/memory_api.py:74-88`); a `ranked`
that the library's own rule (§3) would refuse for lack of turns is not a tool argument.
Widening `search` — off `Literal[False]`, `_search` rendering `EpisodeResult` rows,
`tests/test_memory_api_protocol.py` re-pinned — is the alternative, and it is not phase 1.
The `memory_recall` block is the one `recall()` produces, and its trailing `RECALL_UNRANKED`
line is the only signal the MCP door carries: a tool result is one text block.

The `ranked` argument, described for the model that reads it, names every outcome: set it
when a question is worth a model call and the answer is in what was said rather than in a
stored fact, and leave it off for an ordinary turn; it needs `include_episodes` and no
`memory_types`; the kept turns come first and whole; when the server has no key on file,
the operator has switched the mode off, the provider rejected the key, or the model call
failed or timed out, the read is served in the plain order and the block ends with a
`RECALL_UNRANKED` line saying which; and when the server's ranked reads are full the tool
returns an error asking for a retry in a few seconds. The precedent for explaining a
trailing line in the argument that produces it is `budget` at `tools.py:1527-1536`.

**The tool call comes off the event loop.** `mcp_post` calls `server.handle_message`
bare, inside an `async def`, at `rest/mcp.py:290` and `:308`; the only awaits in the route
are `_authenticate` and `request.json()`. A plain read holds the loop for its store I/O
today, which is milliseconds; a ranked read would hold it for 5 to 6 s and stall every
request the process serves. So `mcp_post` runs `handle_message` under `asyncio.to_thread`
for every `tools/call` message — the same offload `RequestContext.run` is (`deps.py:125`),
with the same context copy, so the project the limiter bound (`limits.py:713`) is still
the project on the worker thread. Every tool call rather than only a ranked one, because
deciding per message means parsing the tool arguments in the transport, and a plain read
off the loop costs nothing.

**A busy refusal over `/mcp` is HTTP 429, so that it releases the reservation.** The
limiter charges `memory_recall` one `retrieval.query` before the tool runs
(`MCP_OPERATIONS`, `policy.py:620-621`; the body peek at `limits.py:680-688`), and it
releases a reservation only on a status of 400 or above (`limits.py:1081`). A
`SelectorBusy` that `_recall` turns into a `ToolError` travels inside a JSON-RPC result
with `isError` set, which `mcp_post` returns as HTTP 200 (`rest/mcp.py:313`) — so, left
alone, a busy refusal over `/mcp` would spend one `retrieval.query` on a read that was not
served, while the same refusal over `/v1` costs nothing. So `mcp_post` maps a busy
refusal on a `tools/call` to HTTP 429 with `Retry-After: 6`, the JSON-RPC error body
inside. The signal crosses the offload the way the bound project does: `mcp_post` puts a
request-scoped marker in the context `asyncio.to_thread` copies, `SelectorGate` sets it
when it refuses, and `mcp_post` reads it after `handle_message` returns; the transport
never parses a tool result's text. The alternative — accept the charge, answer 200 with
the tool error, and say in the tool description that a busy refusal costs one read — was
not taken, because "the refusal costs the customer nothing" then holds on one door only.

### The quota metric and the allowance

**None new in phase 1.** A ranked read spends the same `retrieval.query` reservation as a
plain one (`QUOTA_METRICS` at `policy.py:559-572` is unchanged; `tests/test_ratelimit_http.py:523-535`
keeps pinning that `POST /v1/search` reserves exactly `("retrieval.query",)`, and gains a
twin for a ranked body). The per-tier allowances stay as `quota/plans.py` has them: free
2,000 a month, personal 700 a day, personal_pro 2,000 a day, studio 5,000 a day, team 50,000 a
day, business uncapped. The reason is the margin table in §8: memvara's model cost per call
is $0, so there is nothing to ration.

Phase 2, if it happens (decision 5), is a `retrieval.model_query` metric — a `Metric` in
`METRICS` (`plans.py:170-204`), an allowance in every `DEFAULT_PLANS` entry, a `LABELS`
phrase (`notify.py:145-150`), a `QUOTA_METRICS` entry — on gpt-5.4-mini after a judged
run, with allowances sized from the week-one series in §10. Two constraints phase 1 carries
for it: `retrieval.model_query` has no tag and counts only answered calls, so it is the one
source the allowance can sum; and the limiter cannot refund a unit on fallback
(`limits.py:1081-1082` releases only on >= 400), so an allowance would be spent on a read
the model did not answer unless the metric is one the fallback never touches, which it is.
It is not built now; §9 says why.

### The refusals, and what each one costs

- **Past the allowance: unchanged.** Only free's `retrieval.query` allowance is monthly
  (`plans.py:313`), and a spent monthly allowance is 402 `quota_exhausted` with detail
  `{metric: "retrieval.query", limit, used, resets_at, reason}` and no `Retry-After`
  (`rest/limits.py:1405-1416`). Personal through team are daily (`plans.py:330`, `336`,
  `343`, `399`), and `day` is in `SOFT_PERIODS`, so a spent daily allowance is 429
  `rate_limited` with `Retry-After` (`plans.py:74`; `limits.py:1393-1402`), capped at one
  day. Business has none. A ranked read at the allowance meets exactly the refusal a plain
  one does; no new code.
- **`ranked: true` that cannot be honoured: served, 200, and the response says so.** No key
  on file, the mode switched off by the operator, or the key refused by the provider each
  answer the plain order with `selection.outcome` `unconfigured`, `disabled` or
  `key_rejected`, and one `retrieval.model_refused{reason}` from core. No error code:
  `rest/errors.py`'s `CODES` is an index of failure classes (`app.py:309-316`), 409 there
  means a running job or a lost write race (`errors.py:87`), and `not_configured` is
  deliberately a word the data plane cannot say (`errors.py:64-69`). The message for a
  client is the outcome field; the console page where a key is set is named in the route
  docstring. The `retrieval.query` reservation stands, because a read was served.
- **The cap is full: 429 `rate_limited`, `Retry-After: 6`** — likely right, from the 5 to
  6 s a ranked read holds a thread. `SelectorBusy` comes out of `ctx.run` and the handler
  raises `ApiError(429, "rate_limited", ..., headers={"Retry-After": "6"})`: the precedent
  is `admin/api.py:582-593`, which maps the admin admission cap to a status with
  `Retry-After` through `ApiError`'s `headers=` (`rest/errors.py:137-146`), so there is no
  new renderer. Over `/mcp` the same refusal is the same 429, mapped in `mcp_post` ("The
  MCP door" above). A 4xx releases the reservation (`limits.py:1081-1082`), so on both
  doors the refusal costs the customer nothing. Counted as
  `retrieval.model_refused{reason=inflight}` by core before the exception leaves it.
- **The model call fails or times out:** served, unranked, `selection.outcome` `fallback`
  with its reason and status in the body, `selected` null on every result, and
  `retrieval.model_fallback` on the series. The `retrieval.query` reservation stands,
  because a read was served, and the provider bills the call that timed out.

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
admission cap.

### The production configuration

This is the first model call from the production process, and it is the customer's call.

- **`MEMVARA_LLM` stays `none`.** It configures write extraction only
  (`core.py:790-792`) and turning it on would start "spending on every write that reaches the
  extraction tier" (`asgi.py:398-402`). The read-side model has its own settings.
- **Key handling** (decisions 1 and 2; decision 3 for who may set one). One key per
  organisation, set in the console, held in **a new table, `org_selector_settings`**, one
  row per organisation, in the shape of `org_notification_settings` (`control/schema.py:1699-1708`;
  `store.py:3656-3679`): `key_ciphertext`, `wrapped_dek` (nullable), `base_url`, `model`,
  `enabled`, `updated_at`, `updated_by`. The key is encrypted with AES-256-GCM under
  `idp_key` exactly as `identity_providers.secret_ciphertext` is (`control/idp/crypto.py`),
  or, where the organisation has a customer-managed key enabled, as the two-value envelope
  `encrypt_with_cmk` returns (`control/cmk/oci_kms.py:206-218`: the ciphertext and the
  wrapped DEK, both stored, because `decrypt_with_cmk` needs the DEK back, `:221`) — the
  branch `control/api/sso.py:205-215` and `:394-400` already takes on `wrapped_dek`. The
  first draft's three columns had no DEK column and could not have honoured the CMK clause.
  That makes this the second secret the not-yet-built CMK disable migration has to
  re-wrap, and `docs/PENDING.md:797-811`, which scopes that migration to one secret, is
  amended in Step 2a to enumerate both. `updated_at` and `updated_by` are the record of
  who set the key and when, which is the written instruction decision 4 needs. The console
  shows a fingerprint only. The default model is gpt-5.4-mini and gpt-5.4 is selectable
  (decision 2); the parity run in §6 is mini's judged run, and §6 states its offline screen. A free organisation may set a key: memvara's cost is
  thread time under the cap, and excluding free makes it the one place a visitor cannot
  see the result. There is no per-customer key plumbing today (`git grep -nEi
  'bring.your.own|byok' main` hits only CMK docs saying it is not BYO), so this is new. If
  decision 1 goes the other way — a per-request header, nothing stored — there is no
  table, no schema bump and no console field, and the MCP door is excluded because it
  cannot add a second header.
- **The table is the cheap schema bump.** There is no migration mechanism:
  `control/schema.py`'s `check_version` refuses every older stamp, and `SCHEMA_VERSION` is
  23 (`schema.py:689`). A new table is version 7's and 21's case: `CREATE TABLE IF NOT
  EXISTS` builds it complete on a database that already exists, nothing is back-filled,
  and there is nothing for an operator to check before restamping (`schema.py:296-303`,
  `646-654`, which is also where the version-21 note says a settings row rather than a
  column on `organizations`). Version 24 is that case. The one statement an operator runs
  by hand is the restamp, `UPDATE control.control_meta SET version = 24`, written into
  `deploy/README.md` beside the 21-to-23 statements (`deploy/README.md:234-262`) and into
  `docs/OPERATIONS.md`; a column on `organizations` would have been version 4's case, with
  an `ALTER TABLE` to run, and is not what ships.
- **Building the selector per organisation.** `ProjectMemories.for_tenant` knows a project
  and reads one thing from the control plane, `Categories.category_for(tenant)`
  (`memories.py:56-60`), on a 60 s TTL with a 128-entry cap (`memories.py:46`, `52`,
  `105-132`). The protocol gains a second method, `selector_for(tenant)`, that resolves the
  project's organisation, reads and decrypts the row server-side and returns the selector's
  configuration with a key fingerprint. The docstring's "one method, deliberately"
  (`memories.py:57-59`) is amended with the reason: the alternative is a second protocol for
  one lookup. The rule for the cache is the one it has today, stated for both values: a
  held handle is **checked** every 60 s and **rebuilt only on change** (`memories.py:120-124`),
  and the comparison tuple carries the key fingerprint beside the category, so a key change
  or removal takes effect **within 60 s, not at once**: "nothing here can be told about a
  change: the console writes one row and the workers that matter are other processes on
  other machines" (`memories.py:40-46`), and doing better needs infrastructure this design
  does not add. **An unreadable control plane keeps the previous configuration.**
  `_category` turns every exception into `None` (`memories.py:134-139`, pinned by
  `tests/test_memories.py:168-174`), and read the same way a blip at the TTL boundary would
  drop a keyed organisation to `unconfigured` for a minute. So `selector_for` distinguishes
  "no key" from "could not read": a selector is dropped only when the control plane
  answers that there is no key, and a failed read leaves the held configuration in place
  until the next check.
- **The gate, and what `_clone` builds.** `memvara_cloud/selector.py`, new, holds two
  things. `SelectorGate` is built once in `asgi.build()` and shared by every clone: the
  process-wide admission semaphore and the operator's switch. `GatedSelector` is built by
  `_clone` per organisation from `selector_for(tenant)`, wraps the library's `ModelSelector`
  (whose `llm=` is the `openai.py` backend with a client on the organisation's key and base
  URL), and implements `admit()` against the gate: at `MEMVARA_SELECTOR_INFLIGHT=0` it
  raises `SelectorRefused("disabled")`; when the semaphore is full it raises
  `SelectorBusy` at once; otherwise it holds one slot until the model has answered. The
  semaphore is a `threading.BoundedSemaphore` acquired with `blocking=False`, the `admitted()`
  precedent in `admin/query.py:427-460`: **nothing queues.** A ranked read that finds the
  cap full is refused before it spends anything, and the number of threads ranked reads
  hold is never more than the cap. The first draft ran ranked reads on a dedicated executor
  and capped per organisation, and both were wrong: `asyncio.to_thread` cannot target
  another executor, `run_in_executor` submits the bare callable, and the request context —
  the project `limits.py:713` binds and `deploy/usage.py:133`, `170-179` read at every
  emission, filing `UNATTRIBUTED` on `None` — would not have crossed; a
  `ThreadPoolExecutor` queues without bound; and a per-organisation cap of 4 is a global
  bound of 4 times the active organisations, which two of them on a 4-CPU host would fill.
  So ranked reads stay on the default executor, which copies context, and one process-wide
  number bounds them. Both doors reach the same clone, so both are bounded by the same
  gate.
- **The shared cross-encoder.** One `CrossEncoderReranker` is built in `asgi.build()`, once,
  and passed into every clone as `read_reranker` through the tuning fix below, with
  `read_rerank_top_n=200` and `read_rerank_ranked_only=True` (§3): **a clone rebuild
  constructs no model.** `for_tenant` rebuilds on first sight, on a category or fingerprint
  change after the TTL, and after LRU eviction past `MAX_PROJECTS=128`, and the container
  is capped at `MEMVARA_API_MEMORY:-1g` (`deploy/compose.yaml:553`) with the embedder
  already resident, so a per-clone load would have been a second copy of the weights per
  organisation. Its `score()` is serialised behind one process lock: nothing sets
  `OMP_NUM_THREADS` or `torch.set_num_threads` (`Dockerfile:253-262` sets only the offline
  variables), so concurrent `predict()` calls would oversubscribe the host; the model call
  behind it is socket I/O and overlaps fine. Under the gate at most `MEMVARA_SELECTOR_INFLIGHT`
  ranked reads can be waiting on that lock, so the wait is bounded by the cap times the
  stage's own cost, about a second on the study machine (§2).
- **The settings.** Two, both read by `deploy/memvara_deploy/settings.py` beside
  `MEMVARA_RATELIMIT_EXTRACTOR_MS` (`settings.py:1006`):

  | setting | default | production | what it does |
  |---|---|---|---|
  | `MEMVARA_SELECTOR_INFLIGHT` | `0` | `4` | The process-wide cap on ranked reads in flight. `0` is the operator's switch: the mode is off, every ranked read on a keyed organisation is served unranked with outcome `disabled`, and no cross-encoder runs for it; an organisation without a key has no selector and is `unconfigured` whatever this says (§3). Starts at 4 and is adjusted from the week-one series (§10); nothing measured in Step 5 sets it |
  | `MEMVARA_SELECTOR_TIMEOUT_S` | `10` | `10` | The deadline on the model call, wall-clock over connect and response (§3) |

  Nothing in `deploy/` or `rest/` sets a request timeout; uvicorn is started with none, every
  core call runs on the `asyncio.to_thread` default executor (`deps.py:113-125`), which is
  `min(32, cpu + 4)` threads on Python 3.13 (`Dockerfile:32`), compose limits memory only
  (`compose.yaml:552-554`, no `cpus`), and `MEMVARA_WORKERS=1` in production
  (`settings.py:1079`). So on the 4-CPU host the executor has 8 threads, the cap holds 4 of
  them at most, and plain reads keep the other 4 whatever ranked reads do. The cross-encoder
  stage in front of the model call is not bounded by a timeout of its own; it cost 1.06 s
  mean on the study machine (§2), is unmeasured on the production host, and Step 5 measures
  it there and records the number — the measurement sets no setting.
- **The extractor rate-limit analogue.** `MEMVARA_RATELIMIT_EXTRACTOR_MS` adds a model's
  milliseconds to `addMemories`' weight (`policy.py:728`, `744-746`). The same cannot be done
  here: a 5 s step is 50,000 units at 0.1 ms a unit, `search` and `recall` weigh 65, the
  credential burst is 1,200, and `Policy.__post_init__` refuses any route weight above a
  burst (`policy.py:766-772`). So ranked reads keep their 65 — the only bound the MCP door
  had before this design (`policy.py:405-406`, `621-622`), and a rate cap rather than a
  concurrency one — and the gate is the control.

### The `_clone()` fix, a prerequisite

`_clone` drops `redactor`, `reembed` and all `**tuning` (`memories.py:150-157`;
`core.py:696-714`), and the deployment sets none of them (`asgi.py:1371-1372`), which is why
the guard test lists `tuning` as deliberately not carried (`test_memories.py:81-97`). Until
that changes the hosted number for any read option is the cap-15 number, as the plan says.
The fix: `ProjectMemories` takes the read tuning explicitly at construction — the shared
reranker, its depth and `read_rerank_ranked_only` among it — `_clone` forwards it and adds
`read_selector=` from `selector_for(tenant)` (or nothing, for an organisation with no
key), and the guard test moves `tuning` into the carried set with the reason. Nothing else
about the clone changes; in particular `llm` still comes from the base handle, so write
extraction is unaffected. The package now depends on the core's `read_selector` and
`read_rerank_ranked_only` options and the `ranked` keyword, and the cloud's rule is that
such a dependency is pinned in `tests/test_core_contract.py` in the same commit (the cloud
repository's working-here instructions, lines 140-144).

### The image

The judged default cannot construct in the image on main (§2). `deploy/Dockerfile` gains a
build step beside the embedder's (`Dockerfile:116-118`) that downloads
`cross-encoder/ms-marco-MiniLM-L-6-v2` into the same `HF_HOME`, pinned by model id, so the
runtime stage's `HF_HUB_OFFLINE=1` still holds. `sentence-transformers` is already installed
through `[local-embed]` (`Dockerfile:84`); the weights are what is missing. The smoke in
Step 5 asserts that `CrossEncoderReranker()` constructs offline in the running image and
times `score()` over 200 pairs; the number is recorded with the deploy.

## 5. The pricing page in memvara-web

One FAQ entry in `src/content/pricing.ts` beside "What is actually charged?" (`999-1008`)
and a rewrite of the "Searching costs nothing" paragraph in `src/routes/Pricing.tsx`
(`420-425`). No card gets a number; no `Plan` field; no comparison row. The FAQ text:

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
- No `PastAllowance` kind and no `Plan` field change, so the type check and the fixtures in
  `test/pricing-list-shapes.test.tsx`, `pricing-start-shapes.test.tsx` and
  `pricing-grid-gap.test.tsx` are untouched. A grep of `test/pricing.test.tsx` on `bddba0a`
  for a FAQ-count assertion finds none; the `%s says what the data says` test (350-365)
  derives from the data.

The gate is the whole one the site's working-here instructions name (lines 15-20),
not the test file alone: `npx tsc --noEmit`, `npm test` (vitest at the 100% coverage
thresholds), and `npx playwright test` (the layout matrix).

Pre-existing copy defects on site main are not touched by this change, and they are named
here so nobody reads their survival as endorsement. The page still shows one, two, four and
five projects for free, personal, personal_pro and studio (`pricing.ts:183-228`) while the
cloud has allowed two, three, five and six on those tiers since `7b852af`; the
withdrawn-credit lines in `Pricing.tsx:243-247` and `293-296`; "1,000 free memories" at 436
and 445; and `Terms.tsx:184-201`.

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
unset, which is `off` (`env.ts:67-77`): with `route` on, `selectTurns` drops the other role from the
server's kept-first list after the fact (`prompts.ts:222`) and can drop the turns
the model kept. The `arm-invariant` test in Step 3 asserts both.

**Stack:** the two feature branches — memvara-cloud with Step 2a, core with Step 1 — on a
local stack; the test organisation's `org_selector_settings` row written by a script in
`local/` that encrypts the gateway key from `/Applications/workstation/memorybench/.env.local`
(the file `extract.py:20` reads) under the stack's `idp_key` (the console field is Step 2b
and is not needed for this); model gpt-5.4-mini, the default decision 2 chose;
`MEMVARA_SELECTOR_INFLIGHT=4`;
`read_rerank_ranked_only` **off for both runs**, so a plain read on this stack is reranked
at depth 200 as every judged arm was (§3 says the benchmark stack leaves it false; the
value `asgi.build()` passes in production is on, §4, and the stack's override is recorded
with the run ids). Same 199 questions, seed `20260903`, reader and judge gpt-5.4,
`SKIP_RETRIEVAL_EVAL=1`, paired per question. The run starts from a copied checkpoint with `dataSourceRunId` kept and the
search, answer and evaluate stages reset, so the ingested store is the one the judged arms
read and only the three stages under test are re-run.

**One difference from arm B, stated before the run, and what the first run of check 1 did
to it.** Arm B's candidates were the routed role's top-40 — `extract.py:30-33` restricted to
`routed_role` before the model saw the list. The first Step 1 shipped without routing (it was
fitted to this benchmark's phrasing and its precision on real questions is unmeasured; §9),
so the shipped path's top-40 was both roles. Check 1 on that path, run 2026-09-05 (run id
`memvara-ranked-parity`, gpt-5.4-mini on the customer key, 199 questions), measured gold-turn
recall **0.808** and a non-gold keep rate of 0.022 over every turn the server returned (the
scorer as first written), against 0.912 and 0.064 for the same model over the routed top-40,
at $1.68 for the 199 calls — 2.4 times the estimate, because
assistant turns are long. The floor below fired, the judged runs did not start, and the
remedy it names shipped in core the same day (`retrieve/intent.routed_role`, §3 step 5), so
the screen re-runs on the routed path. It did, the same day: **0.935 gold recall and a 0.068
keep rate at $0.72**, on the scorer as reviewed, which counts only the turns the selector
saw (its `top_n` window, the population `extract.py` scored) — on that basis the unrouted
run reads 0.879 and 0.119, so routing is worth 5.6 points of recall, half the non-gold, and
the 2.4-times cost, and the floor is read on this basis from here on. The other difference
the first draft did not name —
claims sharing the head with turns — is removed by §3's design, which hands the selector the
turn list alone; `selection.candidates` at 40 on every question in the search checkpoint is
the check that it was. Three checks, in order:

1. **Offline screen, no judge.** Run the harness's search stage only, with `MEMVARA_RANKED=1`
   — 199 ranked calls through the server, about $0.70 at gpt-5.4-mini — and score the kept
   set in the checkpoint against the gold labels, as `extract_mini.py` prints: gold recall
   and non-gold keep rate, against mini's offline 0.912 and 6.4% on the routed top-40. The scoring script is a Step 3 deliverable on the
   memorybench branch; today `extract.py` scores from `local/sweep/prep.pkl` through
   `sweeplib.routed_role` (`extract.py:13-33`) and nothing scores the shipped list. If recall
   falls under 0.85, stop: assistant turns are taking gold turns' slots, and routing goes
   back in front of the selector before the judged run — **in core, as a model-free intent
   rule that drops the other role from the candidate list before the selector sees it**
   (`retrieve/intent.py` is the layer). The harness's `MEMVARA_ROLE_SELECT` cannot do it:
   it runs after the server has returned its list.

   **Run, and stopped.** The unrouted screen — both roles' top-40 — measured 0.808 gold
   recall and 0.022 non-gold keep rate over the 199 questions, against mini's 0.912 and
   0.064 on the routed top-40, under the 0.85 floor above. It cost $1.68 for the 199 calls,
   2.4 times the $0.70 estimate, because an assistant turn runs longer than a user turn and
   the model is billed on what it read, not on what it kept. The stop rule fired, and the
   stated remedy is what shipped: `retrieve/intent.routed_role`, a model-free rule fitted
   on LongMemEval's single-session-assistant phrasing, is now in core, and
   `HybridRetriever._run_ranked_stage` filters the reranked turn list to the routed role
   before cutting it to `top_n` (§3, "Where it sits", step 5). This screen therefore
   re-runs on the routed candidate list before check 2 and check 3 proceed.
2. **The unranked twin.** The same stack and list with `MEMVARA_RANKED` off, rendered
   through the same 720-token budget: the cross-encoder's order at depth 200, both roles.
   The twin must be reranked, which is why the stack runs with `read_rerank_ranked_only`
   off for both runs: the paired gain then isolates the selector, and the twin is
   comparable to routed-720 and the earlier arms, which were all reranked. Measured on the
   shipped hosted configuration instead, the twin would be the unreranked fusion order,
   and the gain would credit the selector with the cross-encoder's contribution as well.
   No judged arm has the twin's configuration (routed-720 was one role; the control was cap
   15), and without it a ranked result of 174 cannot be told from "the selector did
   nothing": 174 is inside the band around routed-720's 171 and the control's 172.
3. **The judged replicate, paired.** Answer and evaluate on the checkpoint from check 1.
   Prediction: **the ranked run beats its reranked, unranked twin by 8 or more, paired per
   question** — the selector's gain over the cross-encoder alone, not over an unreranked
   read. 199 paired questions resolve a difference of about 8 on a single run (the
   reader disagrees with itself on 7.8% of identical prompts, about 15 judgements), so a
   paired gain under 8 is not evidence that the selector did anything on this list, and the
   feature does not ship on it. The ranked run's absolute score is recorded with its
   qualifiers and is the number the documentation carries; 182 is not carried anywhere
   until the shipped path has produced it, and it was gpt-5.4's number. For the mini
   default the offline screen under arm B's rendering is 170 of 192 fully covered against
   arm B's 172, a judged upside of −4 against arm B's labels (mini keeps two and a half
   times as many non-gold turns, and they cost budget), so the prediction stated now for
   the ranked run's absolute score is **178, surprised outside 171 to 185**. If it lands
   under 171, the routed-720 level, the default goes back to gpt-5.4 (decision 2's
   alternative) before Step 5. The median context is recorded, not gated: a
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

1. **Open two core issues first.** `CONTRIBUTING.md:147-150` asks for one before a change
   to "the retrieval scoring ... or anything that alters what `why()` reports"; a model
   that reorders results and annotates them is that change, and the PRs cite it. The
   second is the packaged skill's sentence on when to ask for `ranked`
   (`memvara/skills/memvara/SKILL.md`), which is not phase 1 (Step 1 says why) and is
   opened here so that it is a deferral somebody can see rather than work nobody holds.
2. **Step 1** on a core branch; PR open, reviewed, not merged.
3. **Step 2a** on a cloud branch, gated with `MEMVARA_ALLOW_CORE_DRIFT=1` pinned to the core
   PR branch — the deliberate-pin case `STANDARDS.md:60` names; PR open, reviewed, not merged.
4. **Step 3**, the parity run, from the two branches. Its first item, the excerpt-cap check,
   is offline and runs before Step 1 freezes the prompt fixture.
5. **Merge and release**: the core PR merges with the parity number added to its changelog
   entry in the Step 3 commit; the core is released per `docs/RELEASING.md`, because the
   cloud depends on `memvara>=0.1` from the index (`pyproject.toml:12`); the cloud PR is
   re-gated against the released core with `check_core` green, and merges.
6. **Step 2b**, **Step 4**, **Step 5**.

### Step 1 — core: the selector (agent-memory)

Tests to write first, in `tests/test_select.py`:

- a counting fake `Selector` sees exactly `top_n` candidates, all of them `EpisodeResult`, in
  reranked order, is admitted before the reranker scores anything, and is called once per
  ranked read and never on a plain one;
- on a ranked call the selector's list is the turn list at `rerank_top_n`, unaffected by
  how many claims `_rank` returned; on a plain call the episode cap is `max_episodes` and
  `_interleave` cuts as today;
- with `read_rerank_ranked_only=True` a plain read never calls the reranker and a ranked
  read does; with it false both do;
- with `read_rerank_ranked_only=True` a plain read gathers at depth `k` and hydrates no
  more rows than a retriever without a reranker (a counting store sees the same `limit`
  and the same hydration calls from both), and a ranked read gathers at `rerank_top_n`;
- kept turns come first, in reranked order, whole, outside `k`, with `selected=True` and
  their span; the rest follow as §3 orders them, cut to `k`, with `selected=False` on the
  turns the model saw; claims carry `None`; `results.selection.outcome` is `applied` with
  `candidates` and `kept` filled;
- a span that is a substring after stripping the `(YYYY-MM-DD HH:MM) ` prefix keeps it
  stripped; one that is not a substring becomes `None` and the turn is still kept;
- a raising fake, a fake that returns non-JSON, a fake that passes the deadline, and a fake
  that answers 429 each produce the reranked order with `selected` `None` throughout,
  `selection.outcome` `fallback` with the right `reason` (and `status` for the 429), one
  `retrieval.model_fallback`, one `retrieval.select_ms`, no `retrieval.model_query` and no
  token series;
- a fake whose `admit()` raises `SelectorRefused("disabled")` is served unranked with that
  outcome, one `retrieval.model_refused{reason=disabled}`, no reranker call and no
  `retrieval.select_ms`; a fake that answers 401 gives `key_rejected` with status 401,
  its refused count, and one `retrieval.select_ms`, because the call was made; a retriever
  with no selector gives `unconfigured`, its refused count, no `retrieval.select_ms`, and
  no leg run differently; an empty ranked result still carries its `selection`;
- a fake whose `admit()` raises `SelectorBusy` propagates it, emits
  `retrieval.model_refused{reason=inflight}`, and emits no `retrieval.query`;
- `retrieval.tokens_in`/`tokens_out` are emitted only when `Usage.reported > 0`;
  `retrieval.model_query` only when the model answered, with a recording telemetry;
- `ranked=True` with `include_episodes=False`, and with `memory_types` set, each raise
  `ValueError`;
- `recall(ranked=True)` keeps `k` claims when the model keeps `k` or more turns (the
  docstring guarantee), renders every kept turn whole and unkept turns at 280 characters,
  counts a kept turn the budget cut in `RECALL_DROPPED`, ends an unranked block with the
  `RECALL_UNRANKED` line naming the outcome, and puts `selection` on `RecallResult`;
- the two messages built for a fixed candidate list are byte-identical to `extract.py`'s,
  asserted against a fixture copied from it, and the request carries `json_object` and
  `max_completion_tokens: 400`, so a drift in the prompt is a failing test and not a silent
  change to a measured thing;
- the deadline: a fake transport that answers after the deadline is a `timeout` fallback
  even though it answered;
- `ModelSelector(llm=NullLLM())` raises `TypeError`; the two backends' `chat` sends the
  messages, `json_object` and the cap, and records usage through `_shape.record_usage`,
  against a fake client;
- `tests/test_rerank.py`'s subprocess test extended: `memvara.select.model` is never imported
  by the default configuration;
- the six new constants appear in `series_names()`;
- and the layers `3622eb0` tested when it threaded `anchored` (`git show --stat 3622eb0`):
  `tests/test_remote_reads.py` (the client sends `ranked` only when set; `hydrate`
  reads `selected` and `span` off each ranking; and "the remote client returns
  `SearchResults` with `selection` from the wire", `None` against a server that sends
  none), `tests/test_server.py` (`memory_recall` takes
  `ranked`, `memory_search` does not, a served-unranked block ends with the line, a
  `SelectorBusy` is a tool error, and the description names every outcome),
  `tests/test_bench_hosted.py`, and the `aio` twins.

Then the code: `memvara/select/{__init__,base,model}.py` — two modules, the protocol and
records in `base` and `ModelSelector` with its prompt in `model`; the stage in `hybrid.py`
with the read order it belongs to; the `ranked` keyword through eight files — `hybrid.py`,
`core.py` (both `search` overload sets, `recall`'s three overloads and `ScopedMemvara`),
`aio.py`, `remote/api.py`, `remote/aio.py`, `remote/hydrate.py`, `server/memory_api.py`
(`recall` only), `server/tools.py` (`memory_recall` only, and `SelectorBusy` to
`ToolError`); `read_rerank_ranked_only` in `hybrid.py` and `core.py`; `SearchResults` and
the two `Explanation` fields and `RecallResult.selection` in `types.py`; the `Chat`
protocol in `llm/base.py` and `chat` in `llm/openai.py` and `llm/anthropic.py`; the six
telemetry constants; the `RECALL_UNRANKED` line and the two runtime texts, `server/tools.py:1488-1489`
and `server/mcp.py:50-51`.

Documentation in the same commit: `README.md:488-489` (both clauses — "nothing on the read
path calls a model" and "not even the optional reranker" — gain the opt-in), `docs/INTERNALS.md:33-46`
(invariant 1's Scope line gains its opt-in clause and says the default is unchanged, and the
Sketch line at 41-43, "`HybridRetriever` ... take no `llm` parameter at all", is corrected
for `selector=`), `hybrid.py:36-40` (the module docstring the change lands under), the two
runtime texts — `server/tools.py:1488-1489` becomes "it is cheap and local, and involves no
model unless you set `ranked` on a server with a selector", and `server/mcp.py:50-51` the
same — `core.py:2202-2207` (the slot arithmetic gains the ranked rule: on a ranked call
kept turns are outside `k`), `CHANGELOG.md` under Unreleased — the entry states the
mechanism, that it is a query-time model call at the customer's cost, and that the default
path is unchanged, and carries **no accuracy number** until Step 3 adds the parity number
in its own commit — `docs/UPGRADING.md` (the `MemoryAPI` protocol's `recall` gains
`ranked`, so an alternative implementation has to accept it, found the way the `anchored`
entry at 91-98 says), `docs/API.md`, `docs/DEPLOY.md:123-125` (the server's environment
table; the self-hosted `memvara-mcp` server has no setting that configures a selector in
phase 1, so a ranked call there is served unranked with the line, and the table says so),
`memvara/rerank/__init__.py`'s docstring (points at `select`), `docs/ROADMAP.md` ("a hosted
reader has never been run" is still true and is left; the per-query model call is no longer
undiscussed), the `ranked` description in `server/tools.py`, written with the precision the
repository's working-here instructions demand there, and `CONTRIBUTING.md:21`, whose test
count is stale and is re-derived from `python3 -m pytest --collect-only -q` in this commit.

The packaged skill `memvara/skills/memvara/SKILL.md` is **not** touched in phase 1. It has
no `anchored` guidance to extend (0 hits in its 182 lines on `origin/main`), it is vendored
by sha into seven plugin repositories, and a sentence there is seven pin bumps; that is its
own piece of work, under the issue the build order opens, and listed in §9.

Check: the three gates green; the doctest for the new option runs against a fake selector;
`git diff --stat` shows no file outside the list above.

### Step 2a — memvara-cloud: `_clone`, the flag, the gate, the table

Mergeable without counsel: nothing here puts a key field in front of a customer.

Tests first:

- `test_memories.py`: the carried set gains `tuning` and `read_selector`; a clone built for an
  organisation with a key has a selector and one without has none; a changed fingerprint
  rebuilds the clone on the next TTL check, and a removed key is gone from the clone within
  the TTL; an unreadable control plane at the TTL boundary keeps the selector; a clone
  rebuild constructs no reranker (the shared one is the same object before and after);
- `test_core_contract.py`: the `read_selector` and `read_rerank_ranked_only` options and
  the `ranked` keyword pinned;
- `test_ratelimit_http.py`: a ranked `POST /v1/search` reserves exactly `("retrieval.query",)`;
  `test_ratelimit_policy.py`: `QUOTA_METRICS` is unchanged;
- the outcomes, in the order §3 decides them. Unconfigured: a ranked request on an
  organisation with no key returns 200 with `selection.outcome` `unconfigured` and one
  `retrieval.model_refused{reason=unconfigured}`, and still says `unconfigured` at
  `MEMVARA_SELECTOR_INFLIGHT=0`, because it never reaches the gate. Disabled: a ranked
  request on an organisation **with a key on file** at `MEMVARA_SELECTOR_INFLIGHT=0`
  returns 200 with `disabled`, one `retrieval.model_refused{reason=disabled}`, no reranker
  call and no model call. Busy: a fifth concurrent ranked read on a keyed organisation at
  `MEMVARA_SELECTOR_INFLIGHT=4` returns 429 `rate_limited` with `Retry-After: 6`, releases
  the reservation and counts `inflight`. Then the call: a selector that raises
  `SelectorRefused("key_rejected", 401)` says `key_rejected` with the status; a selector
  that passes its deadline returns 200 with `selection.outcome` `fallback`, its reason,
  and `ranking.selected` null throughout; a ranked `/v1/recall` returns `selection` beside
  text that ends with the line when unranked;
- `rest/mcp.py`: a ranked `memory_recall` over `/mcp` does not block a concurrent `/v1`
  request (a selector fake that sleeps on an event, released by the `/v1` call
  completing); a ranked `memory_recall` on an organisation without a key is served
  unranked with the trailing line and one `retrieval.model_refused{reason=unconfigured}`;
  a busy refusal over `/mcp` is HTTP 429 with `Retry-After: 6`, the JSON-RPC error body
  inside, and releases the `retrieval.query` reservation (the quota count after the call
  equals the count before it); the usage a tool call emits is still filed under the
  request's project after the offload;
- `ranked` with `include_episodes` false, or with `memory_types`, is 422; `AskRequest` does
  not accept `ranked` (422), with the reason in the test name;
- `test_rest_openapi.py`: `selected`, `span` and `selection` are declared nullable
  (`test_every_field_that_can_be_null_is_declared_nullable`, line 121); no route gains a
  documented refusal, because none is added;
- `test_admin_catalogue.py` and the metering catalogue's tests: the six series have a
  kind and a unit in `admin/catalogue.py`'s `MEASURES` (`catalogue.py:167-186`, closed —
  `admin.measure("write.turnips")` raises `UnknownMeasure`, `test_admin_catalogue.py:73-82`)
  and the ones the console shows have a name in `metering/api.py`'s `CATALOGUE` (`api.py:180-186`);
- `test_console_contract.py` unchanged: this step adds no console route;
- the key round-trips through `crypto.py` under `idp_key`, and as the two-value envelope
  with `wrapped_dek` for an organisation with CMK enabled; the schema stamp is 24 and a 23
  is refused with the remedy naming the restamp.

Then the code: `memories.py` (the carried set, `selector_for`, the fingerprint in the
comparison tuple, the unreadable-keeps-previous rule), `selector.py` (new: `SelectorGate`,
`GatedSelector`, the locked reranker wrapper), `rest/models.py` (the two request fields
with their validators; `Ranking.selected`/`span`; `selection` on both responses),
`rest/render.py:155`, `rest/app.py` (the two handlers pass `ranked`, recall asks for
`RecallResult`, `SelectorBusy` becomes the 429 with `Retry-After`, and the route docstrings
at 812-814 and 909 say what a ranked read spends, that it is served unranked with the
outcome named when it cannot be ranked, and that 429 is the one refusal), `rest/mcp.py`
(`tools/call` under `asyncio.to_thread`, and a busy refusal on a `tools/call` answered as
429 with `Retry-After: 6` from the marker `SelectorGate` sets), `admin/catalogue.py`,
`metering/api.py`,
`dashboard/src/api/types.ts:115` (`Ranking`, and the `selection` object) and the search
mock in `dashboard/src/mocks/handlers.ts:2566`, `control/schema.py` (version 24, the
table), `control/store.py` (`get_selector_settings`; the setter lands with its route in
Step 2b), `deploy/memvara_deploy/asgi.py` (builds the gate and the shared cross-encoder
once and passes the read tuning to `ProjectMemories`), `deploy/memvara_deploy/settings.py`
(`MEMVARA_SELECTOR_INFLIGHT`, `MEMVARA_SELECTOR_TIMEOUT_S`), `deploy/Dockerfile` (the
cross-encoder weights), `deploy/compose.yaml` and `env.example`.

Documentation in the same commit: `README.md`, `deploy/README.md` (the two settings, that
`MEMVARA_LLM` is not what turns this on, and the version-24 restamp beside the 21-to-23
statements), `docs/OPERATIONS.md` (the cap and its 429, the served-unranked outcomes, the
restamp), `docs/PENDING.md` (the CMK disable migration's scope at 797-811 now names two
secrets, `identity_providers.secret_ciphertext` and `org_selector_settings.key_ciphertext`;
and an entry for Step 2b: the console key field and the three legal documents, open until
counsel answers decision 4 — `docs/legal/README.md:27-29` requires an open legal change to
be listed there; the first draft's claim that a `_clone` entry closes here was wrong,
`grep -iE 'clone|reranker|cross-encoder|read_' docs/PENDING.md` on main finds nothing).

Check: the cloud suite green under a private `COVERAGE_FILE` and a private `MEMVARA_PG_DSN`,
with `MEMVARA_ALLOW_CORE_DRIFT=1` against the Step 1 branch; `/code-review high`; the smoke
on the local stack shows one ranked `/v1/search` returning `selection.outcome` `applied`
and moving `retrieval.model_query` by exactly 1, a plain read moving it by 0, and
`selection.candidates` at the expected count, compared as counts and values, not as an exit
code.

### Step 3 — the harness and the parity run (memorybench)

First, one offline check that costs nothing and runs before Step 1 freezes the prompt
fixture: over `local/compress/extractions.jsonl`, how many kept spans an excerpt cap of N
characters would have cut — 14 of 528 at 500, none at 2,000 (§2). The decision it informs:
phase 1 ships no cap, because the judged configuration had none and the prompt is asserted
byte-identical to it; a 2,000-character cap is the change to make if `retrieval.tokens_in`
shows production turns longer than the sample, and it is a change to the fixture, the
prompt and the parity number together, not a setting.

Then `MEMVARA_RANKED` in `env.ts` with its test (unset means off, anything but `"1"`
throws, the way `MEMVARA_TURNS_ONLY` does), the body field in `index.ts`, the scoring
script for the offline screen, and an `arm-invariant` test that the shipped provider's
request body is unchanged with the knob off and that the parity stack has
`MEMVARA_CONTEXT_FILE` and `MEMVARA_ROLE_SELECT` unset. Then §6's three checks in order,
the offline screen before any judged spend, and the results written into the plan
document's table as new rows with their run ids: the ranked run, its unranked twin, and the
paired difference.

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
set, fingerprint shown, remove, never the key echoed, and the note that a removed key can be
used for up to 60 s.

Then the code: the console API under `control/api/` (set and remove, writing
`org_selector_settings` through `store.set_selector_settings` with `updated_by`, under
`idp_key` or the CMK envelope by the `sso.py:394-400` rule), `OrgSettings.tsx`, the
handlers.

Documentation in the same commit — `docs/legal/DPA.md:50-57`, `docs/SECURITY-QUESTIONNAIRE.md:292-298`
and `docs/legal/SUBPROCESSORS.md:105` — the three places that state no model is consulted
and no content leaves. "A sentence here is either true of the deployment today or it is not
in the document" (`docs/legal/README.md:33`), so they change in this commit and not before:
the questionnaire's "there is no path in the code that could do it" stops being true when
Step 2a deploys, which is why Step 5 does not deploy before this step lands. The PENDING
entry is deleted here.

Check: the console field round-trips — a key set through the route comes back as its
fingerprint and never as itself, `selector_for` on the same stack returns a selector for
that organisation within the TTL, and removing it returns none; the three legal documents
diff only in the sentences named.

### Step 4 — memvara-web

The FAQ entry and the paragraph, with the whole gate in §5 green. Nothing else on the page
changes.

### Step 5 — production

In dependency order. Before the new image starts: the version-24 restamp, by hand, as
`deploy/README.md` records for every bump. Then the image with the cross-encoder weights;
the smoke asserts `CrossEncoderReranker()` constructs offline and reports `score()` over 200
pairs in milliseconds, a number that is recorded and sets nothing. Then the settings:
`MEMVARA_SELECTOR_INFLIGHT=4`, `MEMVARA_SELECTOR_TIMEOUT_S=10`. Then the console. Before
the first key is set: the legal amendments of Step 2b live; the week-one series in §10 on
a dashboard that can read them, which is what the catalogue work in Step 2a is for. Record
the deploy in `local/DEPLOY-<date>.md` as the others are, including the measured reranker
time and that `HF_HUB_OFFLINE=1` still holds.

Check: on the production stack with the test organisation's key set, one ranked
`/v1/recall` returns a block whose turns lead and carry no `RECALL_UNRANKED` line, with
`selection.outcome` `applied`, and moves `retrieval.model_query` on that project by exactly
1; one plain `/v1/recall` moves it by 0 and its `retrieval.latency_ms` is where it was
before the deploy; with the key removed, the same ranked call is served unranked with the
line within 60 s.

## 8. Cost

### To build

Three repositories and the harness, in the steps above. Core is the largest piece: a
package of two modules, one keyword threaded through eight files along a route three
commits already walked, one new tuning option, a list subclass and a `Selection` record,
two `Explanation` fields, two exceptions, a `Chat` protocol with two implementations, six
series, the `RECALL_UNRANKED` line, two runtime texts, and a subprocess test. Cloud is the
`_clone` change, the gate and the per-organisation selector, the shared cross-encoder under
its lock, two request fields with validators, one response object and two `Ranking` fields
on the wire and in the dashboard's types, one table with the cheap schema bump, two
settings, the catalogue entries, the `/mcp` offload, one exception mapped to a 429 on both
doors, a Dockerfile step, and — in its own step — one console field and the legal
documents. Web is
a FAQ entry and a paragraph. The harness is one knob and one scoring script. No new runtime
dependency anywhere: the selector needs an SDK extra the hosted image already installs.

### To measure

Every selector call in the parity work goes through the server on the gateway key, so it
costs what `extract_mini.py`'s did: about $0.70 for 199 at gpt-5.4-mini, the default
decision 2 chose. The screen and the ranked
judged run share those calls, because the screen scores the search checkpoint the judged
run then answers from (§6). So: search with the selector $0.70; answer and evaluate on it
$1.11 (the reader and judge cost of each arm B run); the unranked twin's answer and
evaluate $1.11, its search free. About **$2.92**. Gateway balance after the replicate was
14.61 of the 29.90 the key started with; $15.29 of the user's $20 cap is spent and $4.71
remains under it — $1.79 to spare. Running the screen as a separate pass, as the first
draft had it, would cost another $0.70 and still fit; the calls are shared anyway. The excerpt-cap check costs nothing.

### To run

Memvara's cost per ranked call is $0 in money and about 5 to 6 s of one default-executor
thread under the cap, likely (§2). The customer's cost is their provider's. At the mean:
gpt-5.4 $0.01132 a call, mini $0.00350, nano $0.00093. The margin table is what justifies
phase 1's shape — memvara pays nothing per call — and what rules out paying at gpt-5.4 in
any phase; phase 2's allowances, if it comes, are sized from the series (§10), not from
this table. "Plain reads" is the `retrieval.query` allowance (`plans.py:313-405`); daily
allowances are shown for a 30-day month. The "allowance" column is the first draft's
proposal, kept only to size the exposure. The per-call figures are the file means, the
nearest-rank p95 and the maximum from §2.

| tier | fee | plain reads/month | phase 1: memvara's model cost | allowance | at gpt-5.4 mean | at gpt-5.4-mini mean | at gpt-5.4 p95 | at gpt-5.4 max | break-even calls gpt-5.4 / mini |
|---|---|---|---|---|---|---|---|---|---|
| free | $0 | 2,000 | $0 | 0 | — | — | — | — | 0 / 0 |
| personal | $9 | 700 a day, ~21,000 | $0 | 250 | $2.83 (31%) | $0.87 (10%) | $8.37 (93%) | $9.99 (111%) | 795 / 2,574 |
| personal_pro | $16 | 2,000 a day, ~60,000 | $0 | 500 | $5.66 (35%) | $1.75 (11%) | $16.74 (105%) | $19.98 (125%) | 1,413 / 4,576 |
| studio | $29 | 5,000 a day, ~150,000 | $0 | 1,000 | $11.32 (39%) | $3.50 (12%) | $33.48 (115%) | $39.96 (138%) | 2,562 / 8,295 |
| team | $99 | 50,000 a day, ~1.5M | $0 | 3,000 | $33.95 (34%) | $10.49 (11%) | $100.43 (101%) | $119.87 (121%) | 8,747 / 28,319 |
| business | $499 | uncapped | $0 | 15,000 | $169.76 (34%) | $52.44 (11%) | $502.17 (101%) | $599.33 (120%) | 44,091 / 142,744 |
| enterprise | contract | override | $0 | override | as agreed | | | | |

The mean columns are a projection from LongMemEval turn lengths. The p95 and maximum
columns are what the same allowance costs if every call is as long as the sample's long
ones: there is no prompt cap in the judged configuration and none ships, so on gpt-5.4 an
allowance sized at the mean is over the fee at the p95 on **four of five paid tiers**
(personal is the exception, at 93%), and over it at the maximum on all five. That is why
phase 2, if it comes, is on gpt-5.4-mini. And the number that rules out paying for every
allowed read: if every `retrieval.query` on a tier were model-ranked at memvara's expense,
the model cost would exceed the fee on every paid tier at every model — personal at nano is
$19.61 a month against $9, 218% of the fee; at gpt-5.4 it is $237.67; free's 2,000 reads
would cost $22.63 at gpt-5.4 against a $0 fee.

## 9. Deliberately deferred

- **A memvara-paid allowance (phase 2).** On gpt-5.4-mini the first draft's allowances cost
  10 to 12% of the fee at the mean, which is not what declines it. What does: the free
  tier's exposure has no aggregate bound; the limiter cannot refund a unit on fallback
  (`limits.py:1081-1082` releases only on >= 400); the DPA and `SUBPROCESSORS.md` ("No model
  provider") forbid it as a query flag until amended; mini has no judged end-to-end run; and
  on gpt-5.4 the p95 column in §8 shows an allowance sized at the mean over the fee on four
  tiers. Built only if decision 5's trigger fires after thirty days; sized from
  `retrieval.tokens_in` and `retrieval.model_refused{reason=unconfigured}`.
- **Ingest-time extraction.** The selector has only been measured with the question in
  hand; it is query-time selection, the same kind of step as the cross-encoder done by a
  stronger model, and "the cost per query is an order of magnitude above the reader's
  context". An ingest-time design is a different measurement nobody has made.
- **Routing in core — no longer deferred.** §6's parity run is what was to decide whether
  the selector made this unnecessary, and it decided: check 1's offline screen over both
  roles' top-40 measured 0.808 gold recall against 0.912 routed, under the 0.85 floor, so
  the stop rule fired and the stated remedy shipped. `retrieve/intent.routed_role` — the
  same fourteen-phrase rule, fitted on LongMemEval's single-session-assistant phrasing (19
  of 22 here, 0 of 177 elsewhere in the sample) — now filters
  `HybridRetriever._run_ranked_stage`'s candidate list by role before it is cut to `top_n`
  (§3, step 5). What was true of the rule when it stayed in the harness is still true of it
  in core and still worth naming: its precision on real user text, outside this benchmark's
  phrasing, is unmeasured, and a false fire costs 0.918 coverage against a 0.790 gain, with
  no failsafe inside the rule itself against that cost — only the caller's fallback to the
  other role's turns when the routed one is empty (§3). §6's parity run now measures the
  routed path rather than the harness's own role filter.
- **The 500-question run.** The user's decision stands: not until the 199 number is where
  they want it. A 3-point non-inferiority margin needs about 489 paired questions, so the
  91.5% is a 199-question number and the docs say so.
- **A judged run of `recall()`'s ranked block with facts in front.** The turns block of a
  ranked `recall()` is the judged block (§3); the facts block before it is what no judged
  run had. Measuring the two together is 199 reader and judge calls on blocks built from
  `/v1/recall` — about $1.11 plus the selector calls — once the parity run has passed.
- **Adaptive rendering** (spans for kept turns that overflow, upgraded back to whole turns
  while room remains). Level with whole turns where it applied (51 against 50 on the 58
  span-rendered questions) but carried into arm E by a selector prompt that over-keeps, and
  the two have not been separated on a judged run. The span field in §3 is what it will
  need, and is why the field ships now.
- **The inclusive prompt and the top-60 list.** Recall 0.949 bought with precision on
  questions whose coverage was already complete: arm E's preference column fell from 10 of 12
  to 5 of 12 and the arm landed at 177, two below its predicted range.
- **gpt-5.4 as the default.** It holds the only judged end-to-end number, 182 twice, at
  3.2 times mini's cost to the customer. Decision 2 makes mini the default, on its selector
  recall of 0.912 against 0.895 at a third of the cost, and gpt-5.4 selectable; the parity
  run (§6) is mini's judged run, and its offline screen is stated there.
- **The v2 answer prompt.** +3 on routed-720 on its own, 181 stacked on arm B against 182:
  the gain does not stack, and the pre-registered rule leaves it at no evidence.
- **A cross-encoder on every hosted read.** 84 ms a query at `top_n=20` against a ~3 ms
  search (`docs/ROADMAP.md:407`); 1.06 s mean at depth 200 on the study machine (§2).
  Under this design the shared cross-encoder is on every hosted handle and runs only on
  ranked reads (`read_rerank_ranked_only=True`), under the cap. Turning it on for plain
  reads is its own decision with its own latency measurement on the production host.
- **A timeout on the reranker stage.** The deadline bounds the model call; the cap bounds
  how many ranked reads can be in the cross-encoder or waiting on its lock; nothing bounds
  how long one pass takes. Step 5 measures it on the production host first; a bound is
  added if the measurement says so, not before.
- **An excerpt cap.** None ships (§7, Step 3). A 2,000-character cap cut nothing on the
  sample and is the change to make if production turns are longer; it changes the prompt,
  the fixture and the parity number together.
- **A selector setting for the self-hosted `memvara-mcp` server.** `server/config.py`
  builds its `Memvara` from `MEMVARA_LLM`, `MEMVARA_LLM_MODEL`, `MEMVARA_LLM_MAX_CLAIMS`
  (`config.py:185`, `215-216`) and `MEMVARA_EMBEDDER` (`config.py:77`) and has no read-side
  option. In phase 1 a ranked call on that server is served unranked with the
  `RECALL_UNRANKED` line and `unconfigured` on the result, and its tool description says
  so; a `MEMVARA_SELECTOR_*` setting there, with `docs/DEPLOY.md` rows, is a separate piece
  of work.
- **The packaged skill's sentence on when to ask for `ranked`.** Its own commit and seven
  downstream pin bumps, under the core issue the build order opens for it (§7); not phase 1
  (Step 1).
- **`ranked` on `memory_search`.** Needs `MemoryAPI.search` widened off `Literal[False]` and
  `_search` taught to render turns (§4); `memory_recall` is the door the judged block goes
  through, and it is the one that takes the flag in phase 1.
- **A per-request key header instead of a stored key.** The MCP door cannot add a second
  header and the console action is the written instruction the DPA asks for; see decision 1.

## 10. Risks, and the decisions the user owns

### Risks

- **The parity run is on a different candidate list than arm B.** Stated in §6 with its stop
  rule and its unranked twin. Do not carry 182 into any document until the shipped path has
  produced its own number.
- **Latency p95 is unknown, and the per-call figure is a sum of a measurement and an
  inference.** 1.06 s for the cross-encoder was measured on the study machine; 4.1 to 4.6 s
  for the model call is inferred from wall clock at concurrency 4; 20 of 199 prompts exceed
  8,000 tokens and likely take longer. The deadline and the fallback are the response to
  the model call, Step 5's measurement is the response to the reranker, and the first
  week's `retrieval.select_ms` p95 is the measurement. If fallbacks exceed 5% of ranked
  reads the deadline is wrong, not the feature — and every one of them was billed to the
  customer.
- **A held thread starves plain reads.** Answered by the cap: at most
  `MEMVARA_SELECTOR_INFLIGHT` default-executor threads are ever held by ranked reads, and a
  read past it is refused rather than queued (§4). The check is `retrieval.latency_ms` on
  unranked reads before and after.
- **A production tenant hands the selector fewer than 40 turns.** The study's pool was
  almost all turns; a store with ordinary claim density is not. `selection.candidates` on
  the result is what shows it, per read; there is no series for it (§3, "Counting"), so it
  is a check on sampled responses rather than a dashboard line.
- **Production turns may be longer than LongMemEval's.** Every cost number here is a
  projection from prompt mean 4,618 tokens and maximum 17,284; `retrieval.tokens_in` per
  call is what replaces it, and the excerpt cap in §9 is the answer if it does.
- **A removed key is used for up to 60 s.** By the clone TTL (§4). Stated in the console
  next to the remove action.
- **A revoked key is served unranked, quietly.** Not quietly: `key_rejected` is on every
  such result and on its own refused tag, and §10 watches it.
- **The design invariant changes.** Invariant 1 has been "nothing on the read path calls a
  model" since the library existed, and the mem0 comparison (2 write-path calls against 105)
  rests on the default path. The default is unchanged and the docs must say so in the same
  sentence that announces the opt-in, or a reader will take the headline claim as withdrawn.
  Two of those sentences are read by a model at runtime (§2), and a stale one there
  misleads an agent that cannot go and check.
- **The legal documents currently say the opposite of what this ships.** Three of them,
  named in Step 2b. They change in the commit that puts the key field in the console, and
  nothing deploys before that commit.
- **Friction.** A customer needs a provider account before the result is theirs. That is the
  cost of the design that keeps memvara's margin table at zero, and the phase 2 trigger is
  written on it.

### Decisions taken, 2026-09-04

The user took the five decisions on 2026-09-04. Each is stated with the alternative it
rejected; the paragraph that depends on one is tagged with its number.

1. **Store the customer's key, or take it per request?** Store per organisation in a new
   `org_selector_settings` table — AES-256-GCM under `idp_key` (`control/idp/crypto.py`),
   or the two-value envelope with `wrapped_dek` where the organisation has a
   customer-managed key — with `updated_at` and `updated_by` as the record of the
   instruction, fingerprint only in the console; or require a per-request header and store
   nothing. **Decided: store.** The console action is the written instruction and the
   MCP door cannot add a second header. Applied in §4 "Key handling", "The table", and
   Steps 2a, 2b and 5.
2. **Default model.** gpt-5.4 (only judged end-to-end number; 3.2 times mini's cost, paid
   by the customer) or mini (selector recall 0.912 against 0.895, no judged run).
   **Decided: gpt-5.4-mini as the default, gpt-5.4 selectable.** The recommendation was
   gpt-5.4 until mini had a judged run; the decision makes the parity run in §6 that run,
   with mini's offline screen and prediction stated there. Applied in §3's example, §4
   "Key handling", §6's stack and §8.
3. **Free tier included?** **Decided: yes.** Memvara's cost is thread time under the
   cap; excluding it makes free the one place a visitor cannot see the result. Applied in
   §4 "Key handling".
4. **Legal position, with counsel.** Whether a customer-keyed provider is the customer's
   processor (likely; no 30-day notice) or memvara's subprocessor (notice required), and
   whether a console action satisfies "agree in writing." **Decided: counsel settles it
   before the console field appears.** DPA.md and SUBPROCESSORS.md change in the commit
   that adds the key field, the console switch — the row's `updated_by` and `updated_at` —
   is the instruction, and Step 2b waits on counsel; Step 2a does not.
5. **Phase 2 trigger.** **Decided as stated:** build the memvara-paid allowance only if,
   after 30 days, fewer than a fifth of active paid organisations have set a key while
   `retrieval.model_refused{reason=unconfigured}` is common. Size it from the series below,
   not from §8's table, which is kept to show why phase 1 pays nothing.

### Assumptions and the week-one check

| assumption | series to watch | fail signal |
|---|---|---|
| 5 to 6 s a call, no starvation of plain reads | `retrieval.select_ms` p50/p95; `retrieval.latency_ms` minus `retrieval.select_ms` on the reads that carry both — `applied`, `fallback` and `key_rejected`, the reads where a call was made — which is the reranker's cost; `retrieval.latency_ms` on unranked reads, before vs after | unranked p95 rises; reranker cost far from Step 5's measurement |
| production turns are no longer than LongMemEval's (prompt mean 4,618) | `retrieval.tokens_in` per call | mean above 4,618 shrinks every phase-2 number proportionally, and is the trigger for the excerpt cap in §9 |
| 10 s deadline and the providers are stable | `retrieval.model_fallback` by `reason` and `status`, over `retrieval.model_query + retrieval.model_fallback` | fallback share above 5%, or `provider` with 429 on one organisation daily. Every `timeout` was billed to the customer, so this is a cost signal as well as a latency one |
| keys stay valid | `retrieval.model_refused{reason=key_rejected}` per organisation | any organisation with it daily: a revoked key being served unranked |
| customers will bring a key | organisations with a key on file; ranked share (`retrieval.model_query + retrieval.model_fallback` over `retrieval.query`) per project; `retrieval.model_refused{reason=unconfigured}` | under a fifth of paid organisations with a key while `unconfigured` is common |
| the gateway's `cost_usd` is what a provider bills | one design partner's provider invoice line vs their `retrieval.tokens_in/out` sum | ratio far from $2.25/M prompt, $13.50/M completion |
| the cap is enough at 4 | `retrieval.model_refused{reason=inflight}` per organisation; `retrieval.latency_ms` on unranked reads | any organisation hitting it daily with unranked latency flat means raise it; unranked latency rising means it is already too high |

## 11. Review disagreements

Findings not applied, or applied differently from what the review asked, and why.

- **Pricing review, finding 5** ("$2.25 plus $1.11 is $3.36, under $4.71 with $1.35 to
  spare"). Not applied as stated. The $1.11 is the reader-and-judge cost of an arm B run,
  where the selector ran offline; on the shipped path the selector runs through the server
  on the same gateway key, so the judged run's search stage is itself 199 selector calls at
  about $2.25. The first draft's "screen or replicate but not both" was right about two
  separate passes ($5.61). §6 and §8 now share the calls between the screen and the judged
  run and add the unranked twin, which was $4.47 with gpt-5.4 as the selector — under the
  cap by $0.24 — and is $2.92 now that decision 2 makes mini the default.
- **Correctness review, finding 10**, the clause "likely slower than the study machine".
  Not carried. The production host's reranker time is unmeasured, and Step 5 measures it;
  nothing read here says which way it differs.
- **Completeness review, finding 15** (the test count). The number is dropped and Step 1
  re-derives it; the reviewer's 4,336 is not repeated here because it was not reproduced in
  this rework.
- **Completeness review, finding 19** (restatement). Applied in part. §1 no longer repeats
  the per-call costs. The margin table in §8 stays in full because the code review's F13
  is answered from its p95 column (four of five tiers), and the five decisions in §10 stay
  because they are what this document asks the user for; the per-tier phase 2 allowances
  now appear in that table and nowhere else.
- **Completeness review, finding 6** offered two fixes for the self-hosted server; neither
  is a refusal now. The code review's F11 makes a ranked call that server cannot honour a
  served-unranked read with the outcome on the result, and the `MEMVARA_SELECTOR_*` setting
  stays listed in §9 rather than built, so the phase-1 scope is visible rather than
  reduced in passing.
- **Code review, the "protocol with `select()` only" simplification.** Applied for the
  three flags and not for admission: `Selector` has `admit()` as well, because the cap has
  to be taken before the cross-encoder runs or it bounds the shorter of the two stages and
  leaves the thread it exists to bound unbounded (§3, "The protocol"). One member, with the
  reason beside it.
- **Code review, F6** (the retriever-side reranker). Applied with one addition the finding
  did not name: a reranker on the base handle runs on every read, so hosted plain reads
  would have paid the cross-encoder the finding was sharing. `read_rerank_ranked_only`
  (§3) is the smallest switch that keeps them as they are.
- **Code review, F1** (offload the ranked tool call over `/mcp`). Applied more widely:
  every `tools/call` is offloaded, not only a ranked one, because choosing per message means
  parsing tool arguments in the transport and a plain read off the loop costs nothing (§4,
  "The MCP door").
- **Code review, F8** (cap completion at a few hundred tokens). Applied at 400, with one
  number the finding did not have: nano's maximum completion on the sample was 446, over
  the cap. Nano is not offered as a model, so the cap stands, and §3 says which maxima it
  was set against.
- **Code review, the "one table row" for phase 2.** Not applied as stated. F13's answer —
  which tiers an allowance sized at the mean exceeds at the p95 — needs the per-tier rows,
  so the table stays; the per-tier numbers are cut from the prose, the "sized from this
  table" sentence is gone, and the table's stated purpose is to justify phase 1's shape.
