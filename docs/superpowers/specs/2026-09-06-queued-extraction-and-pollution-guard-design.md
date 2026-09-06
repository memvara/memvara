# Queued extraction on a self-hosted small model, with a guard against predicate pollution

**Status:** design, 2026-09-06. Decided by the user the same day: extraction is queued rather
than synchronous; the pollution guard is built before the container, not alongside it.

**Order of work:** the queue, then the guard, then the container. Each lands as its own PR
and each is useful without the ones after it.

## The problem, measured

The hosted service runs `MEMVARA_LLM=none`. Its write path is the deterministic rule tier:
first-person sentence forms, nothing else. On 2026-09-06 production held **2,283 episodes,
1,936 of them with no claim citing them — 85%.** Everything a customer says that is not
"my name is X" is stored as text and extracted as nothing. `WriteReceipt.unextracted`
counts it, and nothing fails.

That one fact sits under most of the weak rows in `docs/BENCHMARKS.md`: no claims means no
supersession, no bitemporal reasoning, and a graph leg that is unreachable by construction
(LOCOMO extracts 0 claims from 5,882 turns; LongMemEval 78 from 10,866).

A model fixes it, and the model has to be one memvara pays nothing per call for. The
2026-09-03 CPU spike (`memvara-cloud/local/phi4-cpu-spike-2026-09-03/`) measured
`phi-4-mini-instruct` through llama.cpp on the production box, a 4-core Ice Lake:

| finding | number |
|---|---|
| prefill, Q8_0 vs Q4_K_M at `-t 3` | **35.4 vs 34.0 tok/s** — the box is compute-bound, so the larger quant is not slower |
| continuous batching, 3 slots vs sequential | **0.78×** — `--parallel 1` |
| throughput, `maxItems 12` | **34 extractions/hour**, ~105 s each |
| recall on 15 keyed facts, `maxItems 12` vs uncapped | 14/15 both; duplicates **1 vs 10** |
| the shipped prompt on inputs past ~1,300 tokens | **an empty list, every time**, until its closing sentence is removed |
| predicate vocabulary adherence, with the 64-predicate list | 12/15 facts, **6 wrong-predicate claims of 36** |
| the same, with no list | **0/15** — every subject collapses to `user` |

One of those is shipped, `MEMVARA_LLM_MAX_CLAIMS` (memvara#168, 0.11.0). One is open:
`MEMVARA_LLM_EXTRACT_SYSTEM` (memvara#178), which has to merge, release as 0.11.2 and move
the cloud pin before the container section below can be built. Three things are not, and they are
this document: the model cannot run inside a request, its wrong-predicate claims cannot be
allowed to end true facts, and nothing serves it.

## 1. The queue

### Why it cannot be synchronous

105 seconds per extraction against a REST client timeout of 6 seconds. Not a tuning
problem — a hosted frontier model at 12–14 s would also be over it, which is why the
`memvara_cloud` docs already say extraction must be queued before any `MEMVARA_LLM` other
than `none` is set. So the API process keeps `MEMVARA_LLM=none` and stays fast, and a
separate worker extracts from what it stored.

What a customer sees: a write returns as it does today. The turn is retrievable at once as
an episode (`include_episodes`). Its claims appear later — minutes at the current write
rate, longer under a backlog — and `memory_recall` starts answering from them then. That is
eventual consistency on the claim layer only. **The receipt does not yet say so.**
`WriteReceipt.deferred` reads "extraction queued, not yet run", but core sets it only when a
model call *raised* (`pipeline.py`, the `except` in `_tier2`); under `MEMVARA_LLM=none` the
noop branch sets `unextracted` and leaves `deferred` false, so a customer reading the
receipt sees "lost" for a turn a worker will read in five minutes. That is the one core
change the queue needs: a `Memvara(extraction_deferred=True)` construction option — set by
the API when a worker is deployed — under which the noop branch reports the batch on
`deferred` instead of `unextracted`. Small, opt-in, and `deferred`'s existing docstring is
already the right sentence.

### Core already has the primitive

`Memvara.pending_extraction()` is the work list — a query, deliberately not a queue: "an
episode with no claims *is* the pending record, it survives a restart because it is the
store, and it needs no second place for the answer to drift out of step with." It applies
the salience gate for free, so stored chitchat never costs a model call.
`Memvara.reextract(episodes=…)` is `add()` minus tier 0: it skips any episode that already
has claims (the idempotency guarantee — a re-read would otherwise reconcile to `reinforce`
and silently promote what it had stored), runs tier 1 then tier 2, and reports every turn it
read on `receipt.episode_ids`.

Core also says, in `pending_extraction`'s docstring, what it does not do and who must:
a turn a model read and produced nothing from — or produced only claims the grounding
check refused — has no claims citing it and looks exactly like one never read. Measured:
such a turn came back pending on the next sweep and cost another 250 s of CPU to be
rejected again. **Core does not record attempts on the episode, because the durable set
belongs to whatever is doing the scheduling.** This worker is that scheduler, and the
attempt set is its one piece of state.

That option is the only core change the queue needs; everything else is already there.

### The worker: `memvara_deploy.extract`

One more compose service in the shape of `prune`, `storage` and `quota-notify`: a module
with `once()` and `serve()`, an interval setting where `0` means one pass and exit, a
`recording(cfg, "extract", …)` block writing one `memvara_jobs.job_run` row per pass with
counts only, `restart: on-failure`, read-only filesystem, capabilities dropped.

A pass:

1. Enumerate projects from the control plane — `storage.tenants()`'s query, for
   `storage.tenants()`'s reason: `SELECT DISTINCT tenant FROM episodes` is a scan of a
   large table to find a handful of rows.
2. For each project, in round-robin, select up to `MEMVARA_EXTRACT_BATCH` (default **1**)
   episode ids with **one query on the worker's side**:

   ```sql
   SELECT e.id FROM episodes e
    WHERE e.tenant = %s
      AND NOT EXISTS (SELECT 1 FROM claim_sources s
                       WHERE s.tenant = e.tenant AND s.episode_id = e.id)
      AND NOT EXISTS (SELECT 1 FROM extraction_attempts a
                       WHERE a.tenant = e.tenant AND a.episode_id = e.id)
    ORDER BY e.ts, e.seq
    LIMIT %s
   ```

   Not `pending_extraction()`. That method is one `claims_citing` per episode from the
   oldest forward, and its own docstring names it as the N+1 that is "acceptable only
   because the caller is a bounded background pass". A worker sweeping every tenant every
   few minutes, each pass re-walking every episode that already has claims, is not
   bounded in the way that sentence means. The SQL above is what that method means, said
   once to Postgres.
3. `ProjectMemories.for_tenant(tenant).reextract(episodes=ids)` — through the project's own
   vocabulary (`registry_for(category)`), with tier 1 gating and the already-extracted skip
   still applied by core. One `extract()` call per batch. `Categories` is a Protocol; the
   worker's implementation answers `category_for` from the control plane and `None` for
   `selector_for`, because a process that never performs a ranked read has no business
   loading the IdP key, a `SelectorGate` and `org_selector_settings` to build a clone.
4. Record every id on `receipt.episode_ids` in `extraction_attempts`, with the outcome:
   `claims` (something was stored), `gated` (tier 1 dropped it — free, and never selected
   again), `empty` (the model read it and found nothing), `refused` (everything it proposed
   was rejected by a guard), `deferred` (the model call failed; **not** recorded, so it is
   retried next pass).
   The outcome is exact per row only at `MEMVARA_EXTRACT_BATCH=1`. `WriteReceipt` carries
   batch-level counters — `skipped`, `unextracted`, `ungrounded` — and `added` cites its
   episodes, so at a larger batch the worker can tell how many were gated or empty but not
   which. It then records every id that no stored claim cites as `unattributed`, and the
   retry lever below does not reach that outcome. This is a second reason the default is 1.
5. Stop the pass when the batch budget or `MEMVARA_EXTRACT_PASS_SECONDS` is spent, whichever
   first. Sleep the interval. Repeat.

**Batch of one, by default.** Tier 2 batches every surviving episode into a single extraction call,
which is right for a hosted model and wrong here: the spike's empty-list failure begins at
about 1,300 prompt tokens, and two conversational turns plus the vocabulary is already
there. One episode per call keeps the prompt where the model was measured to work. A
setting, because a deployment on a stronger model may want more.

**Round-robin, not oldest-first across tenants.** A tenant with a 10,000-episode backlog
must not hold the queue for every other tenant for a week. One batch per tenant per
rotation, oldest first within a tenant.

### The attempt table

```sql
CREATE TABLE IF NOT EXISTS extraction_attempts (
    tenant       text NOT NULL,
    episode_id   text NOT NULL,
    attempted_at timestamptz NOT NULL,
    outcome      text NOT NULL,   -- claims | gated | empty | refused
    extractor    text NOT NULL,   -- e.g. "openai/phi-4-mini"; a model change is a reason to re-read
    PRIMARY KEY (tenant, episode_id)
)
```

In the **memory schema**, owned by `memvara_cloud/store/postgres.py`, as one more
`CREATE TABLE IF NOT EXISTS` in `_DDL` with **no `SCHEMA_VERSION` bump**: every `_DDL`
statement runs on every open, so an existing schema grows the table without a migration.
The stamp exists for the case `IF NOT EXISTS` silently skips — a new column on an existing
table — and the store's own comment beside its indexes says so. Bumping it here would also
re-run `_backfill_folds` over every claim row for nothing. Not in `memvara_jobs`, whose `job_run.detail` rule is "counts, never identities",
and not in `control`. It is about episodes and has to die with them: `erase_episode`,
`erase_claim(sources=True)` and `purge` each gain one `DELETE FROM extraction_attempts`
beside their `DELETE FROM episodes`, and the erase tests that PR #232 added for
`store/postgres.py` gain the matching assertion. No foreign key, deliberately: the store's
erase paths are the contract, and an FK across the two would make an erasure's row count
depend on cascade order.

`extractor` is stored because "this model read it and found nothing" is only true of that
model. Switching to a stronger one is a reason to read `empty` turns again;
`MEMVARA_EXTRACT_RETRY_EMPTY_FROM=<extractor name>` is the lever, and the default is not to.

### Settings, on the worker only

| variable | default | what it is |
|---|---|---|
| `MEMVARA_EXTRACT_INTERVAL_SECONDS` | `300` | sleep between passes; `0` is one pass and exit |
| `MEMVARA_EXTRACT_BATCH` | `1` | episodes per model call |
| `MEMVARA_EXTRACT_PASS_SECONDS` | `1500` | budget per pass, so a pass ends before the next is due |
| `MEMVARA_LLM_MAX_CLAIMS` | unset | passed to `OpenAILLM(max_claims=…)`; `12` on phi-4-mini |
| `MEMVARA_LLM_EXTRACT_SYSTEM` | unset | a path, passed as `OpenAILLM(extract_system=…)` |
| `MEMVARA_LLM_TERSE_CLAIMS` | unset | passed to `OpenAILLM(terse=…)`; halves the generated tokens per claim |

*Amended 2026-09-06: `MEMVARA_LLM_TERSE_CLAIMS` added to the table.* It landed in core
after this design was written and belongs to the same group as the two above it — a setting
core refuses under cloud mode that the worker must be able to set, because the worker is the
deployment's extraction process. It takes `polarity`, `confidence`, `when`, `amount` and
`unit` out of the schema's `required` list, so the model stops writing a field name and a
null for each one and `shape_claims` supplies the defaults it already documents. Eight
claims are 413 tokens under the shipped schema and 229 under this one, which predicts 45%.
**Measured end to end on the box it is 27%** — 2,401 output tokens against 1,756 over three
episodes with the deployment's own prompt and vocabulary and a 12-claim cap on both arms,
finding the same 10 of 15 key facts. Only the field names went away; the model still spends
tokens on values and on deciding. At production's measured 21.0 tok/s prefill and 5.53
generation, a typical extraction goes from 98 s to 79 s, a **20% cut in wall time**.

The long turn gains most and there it is a reliability fix. The 4,117-token call that spent
220 s on prefill and 333 s generating 1,618 tokens, 554 s in total against the SDK's default
600 s timeout, comes to about 464 s — 92% of the budget down to 77%.

It does not bound a runaway, and the same run measured what one costs: an uncapped claims
array reached 7,197 generated tokens on a 900-character turn, ran 1,957 s, and found 7 of 15
facts against the capped arm's 10, because the restatements crowd out the answer. That is
`MEMVARA_LLM_MAX_CLAIMS` earning its place, and terse inherits it. The knob that would bound
the runaway itself is `OpenAILLM(max_tokens=...)`, currently 8,192: twelve terse claims are
about 344 tokens, so a cap near 2,048 keeps six times the headroom and converts a 600 s
cancellation into a bounded 422 s failure. The third lever is the
client timeout itself, which is the SDK's default rather than a memvara setting and can be
raised in the worker. That is
throughput rather than latency — the queue above is what answers the 6-second timeout — and
throughput is what decides how long the 1,936-episode backlog takes to clear. It carries
one consequence worth stating beside the guard below: an omitted `confidence` lands every
claim at `UNKNOWN_CONFIDENCE` (0.5), so R4's `min(confidence, 0.4)` still lowers a
suspicious claim below its neighbours, but the confidence signal no longer distinguishes
one clean claim from another. `bench/extract_cost.py` in core measures the saving and, given
an endpoint, whether the same facts still come back.

The last three exist in core's `ServerConfig` already and are refused there under cloud mode,
for a reason that does not apply here: the worker *is* the deployment's extraction process.
`memvara_deploy.settings` gains all three, and `asgi._llm` is factored so the API and the worker
build the backend from one function — the API with `MEMVARA_LLM=none`, the worker with
`openai` — rather than two copies that drift.

### Sizing, against production

161 episodes/day arrive; the worker does about 800/day. That is a 20% duty cycle with room
for a busier week. The 1,936-episode backlog clears in roughly two and a half days — fewer
once the gate drops what is chitchat. If inflow ever exceeds ~30/hour sustained, the
backlog grows without bound and the honest fix is a second core, a second box, or a hosted
model for the overflow; the pass count in `job_run` and a `pending` gauge (the SQL above
with `count(*)`) are what make that visible before a customer notices.

## 2. The pollution guard

### What pollution is, and why the existing guard misses it

`reject_ungrounded` refuses a claim whose object shares no vocabulary with its source turn.
It catches invention. It says of itself: "a claim that reuses real vocabulary with an
inverted or misattributed meaning passes clean." Pollution is that case. `gate / lives_in /
Port 55434` — `Port 55434` is in the source, word for word. The value is real and the slot
is wrong, and structure validation cannot see it.

The destructive direction is the ONE-cardinality slot. `lives_in` is `Cardinality.ONE`:
asserting it **ends** whatever was there. So one polluted claim does not add noise, it
retires a true fact and keeps answering with the false one. The fourteen slots where that
can happen: `born_in, born_on, communication_style, job_title, lives_in, located_now, mood,
name, prefers_tool, pronouns, relationship_status, timezone, working_on, works_at`.

### Three rules, two of them measured — a fourth was measured out

*Amended 2026-09-06 after the review of memvara#181.* The first draft had a subject rule,
R2: a builtin predicate on a subject other than `user` is refused, on the reasoning that
`works_at` on `gate` was a slot collision waiting for the speaker's real employer. The
reasoning was wrong — `gate / works_at` and `user / works_at` are different slots and
cannot end each other — and the rule was measured against the fixture to remove **nothing
R3 did not** (wrong-predicate 20, duplicates 0, facts 60/90, identical with and without it),
while it would have refused every fact about a named third party: `alice / lives_in / Porto`,
and every claim a two-person conversation yields, the store shape `read_route_roles=False`
exists for. Removed. The ten unkeyed claims it took, including the two plausibly-true ones
named below, are no longer taken. R1 was also narrowed in the same review to group within a
source turn rather than across the batch, and to keep every known predicate for a value
rather than one — "born in Lisbon, still live in Lisbon" is two facts — refusing only the
unknown predicates beside a known one. Neither change moved a fixture number.

Each was scored against all 18 claim files from the spike (six configurations × three
episodes, 255 claims) with the spike's own `classify()` — hit / wrong-predicate /
duplicate / unkeyed — and against its headline metric, keyed facts found. The script is
`guard_measure.py` in this design's session; the rules go into core with the 18 files and
the three episodes as a fixture, so the numbers below are a test rather than a memory.

| rule | wrong-predicate removed | facts lost | duplicates removed | unkeyed removed |
|---|---:|---:|---:|---:|
| **R1** one (subject, object) under two or more predicates in one turn → keep the known ones | 21 / 46 | 0 | 32 / 32 | 8 |
| ~~R2~~ a builtin predicate on a subject other than `user` → refuse — *removed, see above* | 5 / 46, all also R3's | 0 | 0 | 10 |
| **R3** `lives_in`, `born_in`, `located_now` with a digit or URL in the object → refuse | 5 / 46 | 0 | 0 | 0 |
| **R1 + R3** | **26 / 46** | **0 — 60/90 before and after** | **32 / 32** | 8 |

**R1** is the measured failure mode itself: "forces values into whatever slot is available"
shows up as one value wearing several predicates. When one of them is a known predicate it
is kept; otherwise the first emitted is. Three "hits" it removes are all `gate / status /
Port 55434` beside an identical claim under `port` — the fact stays found, so this is
deduplication, not loss.

**R3** is narrow on purpose: three place predicates, one pattern, on any subject — the
fixture's own `gate / lives_in / Port 55434` is R3's. Broadening it to every ONE-slot
builtin would refuse `job_title: "Engineer II"` and `timezone: "UTC+5:30"`. The pattern's
`port` is word-bounded; the bare form the fixture was first scored with also matched Porto.

**R4** is not a refusal and does not move the table. A model-proposed claim whose predicate
is **novel** (would be acquired, not resolved) or whose predicate is a ONE-cardinality
builtin — `born_on` and `timezone` excepted, whose values always carry digits — with a
digit or URL in the object is stored at `min(confidence, 0.4)`. The
reconciler already treats a value worth less than half the incumbent as one to store
*beside* it rather than end it, so the effect is that pollution can be present but cannot
retire anything. The two arm-A claims no structural rule catches — `user / goal / refuse`
and `user / live_worker_version / 12000 memories a month` — are exactly this class: `goal`
is MANY and never retires anything anyway; `live_worker_version` is novel and lands at 0.4.

### Where it lives

Core, `memvara/write/pipeline.py`, as a stage on `_tier2`'s output before
`_claim_from_dict`, because R1 needs the whole batch. One constructor option on
`WritePipeline` and `Memvara`, `reject_polluted: bool = True`, mirroring
`reject_ungrounded`. Only model-proposed claims pass through it — `remember()` and the fast
path do not, for the same reason they skip the grounding check. Refusals are counted on a
new `WriteReceipt.polluted`, which reaches the REST wire in `memvara_cloud/rest/models.py`
beside `ungrounded`, with the same "zero means either nothing tripped or the option is
off" wording.

Default on, because the destructive direction is storing. A deployment on a frontier model
that measures a false-positive rate it dislikes turns it off; the option exists so that the
measurement can be made.

### What it does not do

It does not know what `product_type` means. A wrong predicate on subject `user` with a real
value and a novel or MANY predicate is stored, discounted. Closing that needs either a
per-predicate object kind on `PredicateSpec` — a schema change this design deliberately does
not make, because it is the fixed vocabulary itself that the spike found small models
struggle with — or a second model call, which is the cost this whole design exists to
avoid. `polluted` on the receipt and the guard's telemetry counter are what will say
whether that is worth building.

## 3. The container

Last, because the two above are worth having on any model and this is the part that ties
the deployment to one. It also depends on memvara#178 shipping as 0.11.2 and the cloud pin
moving — until then `extract_system` is not reachable from a deployment.

A compose service `llm`: `ghcr.io/ggml-org/llama.cpp:full` pinned by digest with
`--entrypoint /app/llama-server`, the `Phi-4-mini-instruct.Q8_0.gguf` already on the box
mounted read-only, and the invocation the spike measured with, verbatim from `quants.sh`:

```
-m /models/Phi-4-mini-instruct.Q8_0.gguf --host 0.0.0.0 --port 8081
-c 8192 --parallel 1 -ctk q8_0 -ctv q8_0 -fa on -t 3 -tb 3 --no-webui
```

with `cpuset: "0-5"` and `mem_limit: 10g` as the spike ran it — the box reports 8 vCPUs
over 4 cores and thread scaling was near-linear to 4 and flat after. The cpuset confines the
*model* to six vCPUs; nothing pins the API or Postgres, and the spike's numbers were taken
that way. Whether they should be pinned to `6-7` is a measurement for the container PR. A `/health` healthcheck, `read_only`, `cap_drop: [ALL]`.
Behind a profile
`extract`, so `memvara-provision.sh up` does not start it: it is opt-in, like
`subscription-notify`, and for the same reason — a service the provisioning script does not
know is a service it will not tell you is missing, so `deploy/README.md` says so in the
same paragraph that documents it.

The `extract` worker joins the same profile, `depends_on: llm: condition:
service_healthy`, with `OPENAI_BASE_URL=http://llm:8080/v1`, `OPENAI_API_KEY=local` (the SDK
refuses to construct without one; the server ignores it), `MEMVARA_LLM=openai`,
`MEMVARA_LLM_MODEL=phi-4-mini`, `MEMVARA_LLM_MAX_CLAIMS=12`, and the small-model prompt
mounted at `MEMVARA_LLM_EXTRACT_SYSTEM`. The prompt file ships in the repo under
`deploy/prompts/`, versioned, because it is configuration a deploy carries and not a
secret.

The API service is untouched. `MEMVARA_LLM=none` on `api` is the design, not an omission.

**The image is offline for models.** `deploy/Dockerfile` sets `HF_HUB_OFFLINE=1`; the API
image needs nothing new because the worker uses the same image and reaches the model over
HTTP. Learned the expensive way on 2026-09-03 with the reranker; noted so nobody re-learns
it here.

## Acceptance

- **Queue.** On the parity stack: write 20 turns through the API with `MEMVARA_LLM=none`,
  observe `deferred: true` and 0 claims; run one worker pass; observe claims cited to those
  episodes, `extraction_attempts` rows for every id `reextract` reported, and a second pass
  that selects none of them. Erase one episode; its attempt row is gone.
- **Guard.** The 18-file fixture in core's test suite reproduces the table above exactly:
  26 wrong-predicate removed, 60/90 facts found before and after, 32/32 duplicates. A test
  reintroduces `gate / lives_in / Port 55434` against a store holding `user / lives_in /
  Lisbon` and asserts Lisbon is still current.
- **Container.** `smoke` extended: with the `extract` profile up, a written turn has claims
  within one pass. And `docs/BENCHMARKS.md`'s ingestion-blind rows are re-measured with the
  worker on — the graph leg's join rate on LongMemEval is the number this whole line of
  work is supposed to move, and it is reported whether or not it moves.

## Out of scope, named

- Extracting inside a request on the hosted service. Never; this design exists so it does
  not have to.
- The Anthropic backend. `extract_system` and `max_claims` are openai-backend options; a
  hosted frontier model does not need either.
- A per-predicate object kind on `PredicateSpec`. See "What it does not do".
- Making the LoCoMo and LongMemEval retrieval benchmarks exercise the write path. Separate
  work; `docs/BENCHMARKS.md` already says they do not.
