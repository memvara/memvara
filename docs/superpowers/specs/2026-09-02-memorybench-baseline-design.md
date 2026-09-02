# A judged LongMemEval-S baseline for memvara in MemoryBench

**Status:** design, awaiting review. Step 1 of the programme decided on 2026-09-02: be
first on the agent-memory benchmarks by the metric the current leader publishes.

## Why this exists

Supermemory's "95%" is Recall@15 on LongMemEval-S with a model rewriting memories at
ingest, shown beside LLM-judged answer accuracy figures copied from Zep's paper (Zep
71.2%, full context 60.2%, both GPT-4o). Mem0 publishes 94.4% judged accuracy on the same
benchmark. Memvara's published 70.4% is retrieval R@12 on the oracle file with a hashing
embedder and no model. None of these numbers is the same measurement, and memvara has no
number at all on the one that a reader can reproduce: **judged accuracy inside
Supermemory's own harness, MemoryBench, with GPT-4o as reader and judge.**

This step produces that number for memvara exactly as shipped today. It changes no
retrieval or ingest behaviour. Every later step (read path, ingest, category work,
hosted run) is measured against the harness this step stands up, so the number it
produces is a baseline and not a result, and nothing in it is tuned to look good.

Decisions already made, and not reopened here: the target is judged accuracy in
MemoryBench (not Recall@15, not memvara's own tables); the iteration loop runs locally
and the published number comes from the hosted service; a model at ingest becomes
default behaviour with the fast path as fallback. The last two are later steps.

## What MemoryBench is, as far as this step needs it

[supermemoryai/memorybench](https://github.com/supermemoryai/memorybench), MIT, TypeScript
on Bun. Five checkpointed phases per run: ingest, indexing, search, answer, evaluate, then
report. A provider is one TypeScript module implementing:

```typescript
interface Provider {
  name: string
  prompts?: ProviderPrompts            // answerPrompt: template or (question, context, questionDate) => string
  initialize(config: ProviderConfig): Promise<void>
  ingest(sessions: UnifiedSession[], options: IngestOptions): Promise<IngestResult>   // { documentIds }
  awaitIndexing(result: IngestResult, containerTag: string): Promise<void>
  search(query: string, options: SearchOptions): Promise<unknown[]>                   // whatever shape we like
  clear(containerTag: string): Promise<void>
}
```

Facts that shape the adapter, read from the source on 2026-09-02:

- The LongMemEval benchmark loads `longmemeval_s_cleaned.json` (277 MB) from HuggingFace
  into `data/benchmarks/longmemeval/datasets/` on first use. That is the faithful `-S`
  setting: 500 questions, each with its own haystack of about 40 sessions.
- Sessions arrive as `{ sessionId, messages: [{ role, content, timestamp? }] }`. The
  timestamp is the session date parsed from the dataset's `YYYY/MM/DD (day) HH:MM`; every
  turn in a session carries the same one. Session ids are `<question_id>-session-<i>`.
- **The container tag is per question.** Each question's haystack is ingested under its
  own tag and searched under it, so a provider sees one question's sessions at a time and
  `clear` must remove exactly that tag.
- The search phase calls `provider.search(question, { containerTag, limit: 10, threshold: 0.3 })`
  and times the call. Both shipped providers ignore `limit` and ask their own service for
  30; Supermemory uses `threshold: 0.3, searchMode: "hybrid", include: { summaries, chunks }`.
- The answer phase renders the provider's results with the provider's `answerPrompt` if
  given, else `JSON.stringify(context, null, 2)` inside a default prompt. `questionDate`
  is passed. The reader defaults to `gpt-4o`, temperature and max tokens from the model
  table (1000 tokens out). Context tokens are counted as full prompt minus the prompt
  with no context.
- The judge defaults to `gpt-4o` and returns `{ score: 0|1, label, explanation }` with a
  prompt chosen by question type: default, abstention, temporal (off-by-one days
  forgiven), knowledge update (mixed old and new accepted if the update is right), and
  preference (rubric-based).
- Reported per run: accuracy overall and per type, search latency p50/p95, answer latency,
  context tokens, and success rate (errors are excluded from accuracy, not counted against
  it). No composite score.
- Keys come from `OPENAI_API_KEY` and per-provider variables in `src/utils/config.ts`.

## Design

### 1. Where the code lives

A fork, `memvara/memorybench`, branch `memvara-provider`, tracking upstream `main`. The
provider is written to upstream's contract so the same branch can be offered upstream as
a pull request once the number is real. Nothing about the provider is memvara-specific
beyond the HTTP calls it makes, so the upstream PR is the leaderboard entry.

In `agent-memory`: this spec, a runbook section in `docs/BENCHMARKS.md`, and the results.
In `memvara-cloud`: nothing changes in this step.

Files in the fork:

| file | purpose |
|---|---|
| `src/providers/memvara/index.ts` | the `Provider` implementation |
| `src/providers/memvara/prompts.ts` | the answer prompt that renders memvara's results |
| `src/providers/index.ts`, `src/types/provider.ts` | registration and the `"memvara"` name |
| `src/utils/config.ts` | `MEMVARA_API_KEY`, `MEMVARA_BASE_URL` (default `http://127.0.0.1:58080`) |
| `src/providers/README.md` | one paragraph on running against a local stack |

### 2. The provider, method by method

All calls go to memvara's REST API, the same routes the hosted service serves, with the
container tag carried as the `user` scope. The credential's scope is the tenant; any
route narrows it with `?user=<tag>`, which is how the library's own client addresses a
user, and a request narrowed to one user cannot reach another.

**initialize.** Read key and base URL. `GET /v1/whoami` once; refuse to start on anything
but 200, printing the scope and privilege it reports. Log `GET /v1/health` so the run
record carries the server's version string.

**ingest(sessions, { containerTag }).** One `POST /v1/memories?user=<containerTag>` per
session, body `{ messages: [{ role, content, ts }], ts }` with `ts` the session's ISO
timestamp on every turn and on the request, and `metadata: { sessionId }` on each
message. Send an `Idempotency-Key` per session, derived from `containerTag + sessionId`,
so a retry after a timeout replays the first receipt instead of writing twice (the API
returns the stored response with `Idempotency-Replayed: true`). Retry on 429, 5xx and
transport errors with exponential backoff, at most five attempts; the orchestrator has
no retry of its own and halts the run on the first thrown error. Return
`documentIds = receipt.episode_ids` flattened across sessions.

Sessions inside one `ingest` call run with bounded concurrency (4). They are one
question's haystack and order between sessions does not matter to memvara's write path,
which reconciles on the world clock carried in `ts`, not on arrival order.

**awaitIndexing.** Resolves immediately. Memvara's write is synchronous: the receipt is
returned after the turns are stored, embedded, indexed and extracted. As a check that
costs one call, `GET /v1/stats?user=<containerTag>` and log the episode count.

**search(query, { containerTag }).** `POST /v1/search?user=<containerTag>` with
`{ query, k: 30, min_score: 0, include_episodes: true }`. `k: 30` matches what the two
shipped providers ask their services for; the orchestrator's `limit: 10` and
`threshold: 0.3` are ignored by them and are ignored here for the same reason, and
memvara's score is not on the scale their threshold is on. `min_score: 0` because the
baseline measures the ranking as shipped; abstention is handled in the prompt, and a
relevance floor is step 2's business.

The method returns memvara's results reshaped into plain objects the prompt renders,
memories first then turns, in the order the API ranked them:

```json
{ "kind": "memory", "text": "...", "subject": "...", "predicate": "...", "object": "...",
  "valid_from": "2023-05-20T15:30:00Z", "recorded_at": "2023-05-20T15:30:00Z",
  "state": "live", "score": 0.61, "sources": ["ep_..."] }
{ "kind": "turn", "role": "user", "content": "...", "ts": "2023-05-20T15:30:00Z", "score": 0.44 }
```

Nothing is dropped, deduplicated or re-ranked in the provider. What memvara returns is
what the reader sees.

**clear(containerTag).** `POST /v1/erasures` with `{ scope: { user: containerTag } }`.
This is an erasure, not a retirement: claims, turns, vectors and index entries go. It is
the right operation here because a benchmark container is not a customer's memory and a
retired-but-readable haystack would leak into the next run's search under the same tag.
`confirm_tenant` is never sent, so a bug that drops the user from the scope is refused by
the API rather than erasing the tenant.

**prompts.answerPrompt.** A function, not a template, so it can render the two kinds
differently. The structure follows what the shipped providers do, because the prompt is
part of the provider contract and the comparison is only fair if each provider's prompt
suits its own output:

- a *Memories* block: one line per memory, `[valid from <date>, recorded <date>, <state>]
  <text>`; the dates are memvara's two clocks and are what the temporal and
  knowledge-update questions need;
- a *Conversation excerpts* block: one line per turn, `[<date>] <role>: <content>`, the
  raw evidence;
- the question date, with the instruction to resolve "today", "last week" and the like
  against the dates shown, never against the current date;
- "answer only from the context; if it is not there, say I don't know";
- think-then-answer, with the final answer on its own line.

No judge prompt override. The judge is the harness's, unchanged.

### 3. The local stack

`memvara-cloud` as it deploys, on this machine: `docker compose -f deploy/compose.yaml up -d`
from the memvara-cloud checkout, which brings up Postgres on `127.0.0.1:55433`, runs
`migrate`, starts the API on `127.0.0.1:58080` and runs `seed`, which mints an admin key
readable with `docker compose -f deploy/compose.yaml run --rm key`. Two things the run
sheet fixes explicitly:

- **`MEMVARA_CORE_PATH` names the agent-memory checkout under test**, a clean checkout at
  the commit being measured, never the shared working copy that other sessions keep on
  feature branches. The image builds the core from that path, so the core sha is what the
  run measures; it is recorded with every result.
- `MEMVARA_LLM=none`, the shipped default, for this baseline. The embedder is the image's
  `all-MiniLM-L6-v2`, as on the hosted box.

`memvara-pg` on 55432 is the test suite's container and is not touched.

Iterating on the core later means `compose build api && compose up -d api`, minutes, not
a deploy. That loop is for steps 2 to 4; this step builds it once.

### 4. Runs and what is kept

```bash
# fork, once
bun install

# smoke: one question end to end (the harness's test-question command), then twenty
# (its run command with the question-limit filter; exact flag names are the fork's
# README's business, not this document's)
bun run src/index.ts test-question -p memvara -b longmemeval <question id>
bun run src/index.ts run -p memvara -b longmemeval -j gpt-4o -r memvara-smoke <limit 20>

# the baseline
bun run src/index.ts run -p memvara -b longmemeval -j gpt-4o -r memvara-baseline-<core sha>
```

Kept, per run: `data/runs/<run>/report.json` and the per-question results directory,
copied into `agent-memory`'s main checkout under `local/memorybench/<run>/` (not a
worktree's `local/`, which does not survive the worktree). Published: a section
"Judged accuracy in MemoryBench" in `docs/BENCHMARKS.md` with the table below and a
sentence naming every input the number depends on.

| provider | core sha | cloud sha | ingest model | reader | judge | accuracy | per type | search p50 / p95 | context tokens |
|---|---|---|---|---|---|---|---|---|---|

The other providers' rows appear in that table only when run by us under the same judge
in the same harness. Until then their published figures are cited beside it with the
metric named, because Supermemory's Recall@15 and Mem0's judged accuracy are not the
same number and the section must not pretend they are.

### 5. What decides that this step is done

- `test-question` completes for one question of each of the six types and for one
  abstention question, with the rendered prompt saved and read by a person once.
- The 20-question smoke run reports success rate 100%, and `clear` followed by
  `GET /v1/stats?user=<tag>` shows zero episodes for a tag that was ingested.
- The full run completes with success rate 100%. Errors excluded from accuracy would
  flatter the number, so a run with failures is re-run, not reported.
- `report.json` carries per-type accuracy for all six types plus abstention, search
  p50/p95, and context tokens, and those numbers are in `docs/BENCHMARKS.md` with the
  core and cloud shas, the models, and the date.
- The fork branch builds and lints under upstream's own configuration, so the eventual
  upstream PR is a matter of opening it.

### 6. Cost and time, estimated

Ingest: about 20,000 sessions (500 haystacks of about 40), roughly 300,000 turns, embedded
locally by MiniLM on CPU and indexed in Postgres. No model cost at `MEMVARA_LLM=none`.
Expect 30 to 60 minutes; the harness checkpoints, so a stopped run resumes.

Answer and judge: 500 questions at GPT-4o, about 2 to 3k input tokens each for the answer
and about 1k for the judge. Roughly 8 to 12 US dollars per full run at current pricing.
The 20-question smoke is under a dollar.

### 7. What you have to provide or approve

- An OpenAI key for the reader and judge, as `OPENAI_API_KEY` in the fork's `.env`.
- Bun on this machine (`brew install bun`); Node 26 is present, Bun is not.
- The 277 MB dataset download the harness makes on first run.
- Creating the fork under the `memvara` organisation.

### 8. Out of scope, and where it goes

- Any change to ranking, the candidate window, reranking, or the relevance floor: step 2.
- A model at ingest, provenance from memories to turns, event-date resolution: step 3.
- Per-category work (temporal leg, multi-session aggregation, preference prompt): step 4.
- Runs against the hosted service, a benchmark tenant, the published comparison: step 5.
- Retrieval-only metrics in the harness (`retrieval-eval` phase): useful later for
  cheap iteration without a reader; not part of the baseline number.
- Comparison runs of Supermemory, Mem0 and Zep: only with their keys, and only when the
  table is ready to carry them.
