# MemoryBench Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce one reproducible, LLM-judged LongMemEval-S accuracy number for memvara as shipped today, inside Supermemory's MemoryBench harness, with the per-type table published in `docs/BENCHMARKS.md`.

**Architecture:** A `memvara` provider in a fork of `supermemoryai/memorybench` talks to memvara's REST API over HTTP, carrying MemoryBench's per-question container tag as memvara's `user` scope. The API under test is memvara-cloud's own compose stack running on this machine, built from a clean checkout of the core commit being measured. Nothing in memvara's ingest or read path changes.

**Tech Stack:** TypeScript on Bun (`bun test`, prettier) in the fork; Docker Compose for memvara-cloud (Postgres 55433, API 58080); memvara-cloud's `/v1` REST API; GPT-4o as reader and judge via `OPENAI_API_KEY`.

**Spec:** `docs/superpowers/specs/2026-09-02-memorybench-baseline-design.md`

## Global Constraints

- No Claude or AI attribution in any commit message, PR title or body, or issue, in either repository. No `Co-Authored-By` trailer, no generated-with line. (`~/.claude/CLAUDE.md`)
- Commit files by name. Never `git add -A`, `git add .` or `git commit -a`. (`CLAUDE.md`)
- The fork is `memvara/memorybench`, cloned at `/Applications/workstation/memorybench`, branch `memvara-provider`, written to upstream's `Provider` contract unchanged.
- The core under test is a **clean checkout**, `/Applications/workstation/agent-memory/local/wt-memorybench`, never `/Applications/workstation/agent-memory` itself, which other sessions keep on feature branches. Its sha is recorded with every result.
- Local stack: `docker compose -f deploy/compose.yaml` from `/Applications/workstation/memvara-cloud`, API `http://127.0.0.1:58080`, Postgres `127.0.0.1:55433`. `memvara-pg` on 55432 is the test suite's and is not touched.
- `MEMVARA_LLM=none` for the baseline. Embedder is the image's `all-MiniLM-L6-v2`.
- Provider search asks memvara for `k: 30, min_score: 0, include_episodes: true` and returns what memvara ranked, unchanged.
- Reader and judge are `gpt-4o`, MemoryBench's defaults. A run with any failed question is re-run, never reported.
- Results are copied to the main checkout's `local/memorybench/<run>/`, never left only in a worktree's `local/`.
- Prose follows `CLAUDE.md` "How to write": say what changed, lead with the answer, one term per concept.

---

## File structure

**Fork `/Applications/workstation/memorybench` (TypeScript):**

| file | responsibility |
|---|---|
| `src/providers/memvara/client.ts` | HTTP only: bearer auth, `user` query parameter, idempotency header, retry with backoff, typed responses. Knows nothing about sessions or prompts. |
| `src/providers/memvara/client.test.ts` | the client against an injected fake `fetch` |
| `src/providers/memvara/prompts.ts` | renders memvara's search results into the answer prompt |
| `src/providers/memvara/prompts.test.ts` | rendering, including the empty case |
| `src/providers/memvara/index.ts` | `MemvaraProvider`: maps MemoryBench sessions to `/v1/memories` writes, container tag to scope, `/v1/search` hits to plain result objects, `clear` to a scope erasure |
| `src/providers/memvara/index.test.ts` | the provider against an injected fake client |
| `src/providers/index.ts` | registration |
| `src/types/provider.ts` | `"memvara"` in `ProviderName` |
| `src/utils/config.ts` | `MEMVARA_API_KEY`, `MEMVARA_BASE_URL` |
| `src/providers/README.md` | the table row and the local-stack paragraph |

**agent-memory, branch `claude/memorybench-baseline`:**

| file | responsibility |
|---|---|
| `docs/BENCHMARKS.md` | new section "Judged accuracy in MemoryBench" with the table and its inputs |
| `local/memorybench/<run>/` (ignored) | `report.json` and `results/` copied from the fork |

**memvara-cloud:** nothing changes.

---

### Task 1: Fork, clone, toolchain

**Files:**
- Create: `/Applications/workstation/memorybench` (clone of the fork)
- Create: `/Applications/workstation/memorybench/.env.local` (never committed; `.gitignore` in upstream already ignores `.env*`; verify in step 5)

**Interfaces:**
- Produces: a working `bun run src/index.ts` and `bun test` in the fork; the branch `memvara-provider`.

- [ ] **Step 1: Install Bun**

Run: `brew install bun && bun --version`
Expected: a version line, `1.x`.

- [ ] **Step 2: Fork under the memvara organisation and clone**

Run:
```bash
cd /Applications/workstation
gh repo fork supermemoryai/memorybench --org memvara --clone --remote
cd memorybench
git remote -v
```
Expected: `origin` is `memvara/memorybench`, `upstream` is `supermemoryai/memorybench`. If the fork already exists, `gh` says so and still clones.

- [ ] **Step 3: Branch and install**

Run:
```bash
cd /Applications/workstation/memorybench
git checkout -b memvara-provider
bun install
bun run src/index.ts --help
bun test
```
Expected: help text listing `run`, `compare`, `test`, `status`; `bun test` reports 0 tests (the repository ships none).

- [ ] **Step 4: Confirm the answer and judge defaults are gpt-4o**

Run: `grep -n "DEFAULT_ANSWERING_MODEL\|DEFAULT_JUDGE_MODEL" src/utils/models.ts src/cli/commands/run.ts | head`
Expected: both default to `"gpt-4o"`. If either differs, note the value; every run in this plan passes `-j gpt-4o -m gpt-4o` explicitly anyway.

- [ ] **Step 5: Confirm env files are ignored, then write `.env.local` with placeholders**

Run:
```bash
git check-ignore -v .env.local || echo "NOT IGNORED"
cat > .env.local <<'EOF'
OPENAI_API_KEY=
MEMVARA_API_KEY=
MEMVARA_BASE_URL=http://127.0.0.1:58080
EOF
git status --short
```
Expected: `.gitignore` names `.env.local` (or `.env*`); `git status` shows nothing. If it prints `NOT IGNORED`, add `.env.local` to `.gitignore` and commit that one line before anything else.

- [ ] **Step 6: Commit nothing yet; record the upstream sha**

Run: `git rev-parse --short upstream/main 2>/dev/null || git rev-parse --short HEAD`
Write the sha down; the README paragraph in Task 5 and the results section in Task 9 name it.

---

### Task 2: The HTTP client

**Files:**
- Create: `src/providers/memvara/client.ts`
- Test: `src/providers/memvara/client.test.ts`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  ```typescript
  export class MemvaraClient {
    constructor(opts: MemvaraClientOptions)
    whoami(): Promise<MemvaraWhoAmI>
    health(): Promise<MemvaraHealth>
    addMemories(user: string, body: MemvaraAddRequest, idempotencyKey: string): Promise<MemvaraWriteReceipt>
    search(user: string, body: MemvaraSearchRequest): Promise<MemvaraSearchResponse>
    stats(user: string): Promise<MemvaraStats>
    eraseUser(user: string): Promise<MemvaraErasure>
  }
  export interface MemvaraClientOptions { baseUrl: string; apiKey: string; fetchImpl?: typeof fetch; maxAttempts?: number; baseDelayMs?: number; sleep?: (ms: number) => Promise<void> }
  export type MemvaraHit = { kind: "claim"; score: number; memory: MemvaraMemory } | { kind: "episode"; score: number; episode: MemvaraEpisode }
  export class MemvaraHttpError extends Error { status: number; body: string }
  ```

- [ ] **Step 1: Write the failing tests**

```typescript
// src/providers/memvara/client.test.ts
import { describe, expect, test } from "bun:test"
import { MemvaraClient, MemvaraHttpError } from "./client"

type Call = { url: string; init: RequestInit }

function fakeFetch(responses: Array<{ status: number; body: unknown } | Error>) {
  const calls: Call[] = []
  const impl = (async (url: string | URL | Request, init?: RequestInit) => {
    calls.push({ url: String(url), init: init ?? {} })
    const next = responses.shift()
    if (next === undefined) throw new Error("fakeFetch: no response scripted")
    if (next instanceof Error) throw next
    return new Response(JSON.stringify(next.body), {
      status: next.status,
      headers: { "content-type": "application/json" },
    })
  }) as unknown as typeof fetch
  return { impl, calls }
}

const noSleep = async () => {}

describe("MemvaraClient", () => {
  test("sends the bearer token and the user scope as a query parameter", async () => {
    const f = fakeFetch([{ status: 200, body: { count: 0, results: [] } }])
    const client = new MemvaraClient({ baseUrl: "http://api.test", apiKey: "k1", fetchImpl: f.impl, sleep: noSleep })
    await client.search("q-1", { query: "hello", k: 30, min_score: 0, include_episodes: true })
    expect(f.calls).toHaveLength(1)
    expect(f.calls[0].url).toBe("http://api.test/v1/search?user=q-1")
    const headers = f.calls[0].init.headers as Record<string, string>
    expect(headers["Authorization"]).toBe("Bearer k1")
    expect(headers["Content-Type"]).toBe("application/json")
    expect(JSON.parse(String(f.calls[0].init.body))).toEqual({ query: "hello", k: 30, min_score: 0, include_episodes: true })
  })

  test("addMemories sends the idempotency key and the ts on the request", async () => {
    const receipt = { episode_ids: ["ep_1", "ep_2"], added: [], invalidated: [], reinforced: [], skipped: 0, unextracted: 0, llm_calls: 0, latency_ms: 1, deferred: false, note: null }
    const f = fakeFetch([{ status: 200, body: receipt }])
    const client = new MemvaraClient({ baseUrl: "http://api.test", apiKey: "k1", fetchImpl: f.impl, sleep: noSleep })
    const out = await client.addMemories("q-1", { messages: [{ role: "user", content: "hi", ts: "2023-05-20T02:21:00.000Z" }], ts: "2023-05-20T02:21:00.000Z" }, "q-1:s-0")
    expect(out.episode_ids).toEqual(["ep_1", "ep_2"])
    expect(f.calls[0].url).toBe("http://api.test/v1/memories?user=q-1")
    expect((f.calls[0].init.headers as Record<string, string>)["Idempotency-Key"]).toBe("q-1:s-0")
  })

  test("retries a 503 with backoff and then succeeds", async () => {
    const slept: number[] = []
    const f = fakeFetch([{ status: 503, body: { error: { code: "unavailable", message: "x" } } }, { status: 200, body: { status: "ok", memvara_version: "1" } }])
    const client = new MemvaraClient({ baseUrl: "http://api.test", apiKey: "k1", fetchImpl: f.impl, baseDelayMs: 100, sleep: async (ms) => { slept.push(ms) } })
    const out = await client.health()
    expect(out.memvara_version).toBe("1")
    expect(f.calls).toHaveLength(2)
    expect(slept).toEqual([100])
  })

  test("retries a thrown network error", async () => {
    const f = fakeFetch([new Error("ECONNRESET"), { status: 200, body: { status: "ok", memvara_version: "1" } }])
    const client = new MemvaraClient({ baseUrl: "http://api.test", apiKey: "k1", fetchImpl: f.impl, sleep: noSleep })
    await expect(client.health()).resolves.toEqual({ status: "ok", memvara_version: "1" })
    expect(f.calls).toHaveLength(2)
  })

  test("gives up after maxAttempts and reports the last status", async () => {
    const f = fakeFetch([{ status: 502, body: "" }, { status: 502, body: "" }, { status: 502, body: "" }])
    const client = new MemvaraClient({ baseUrl: "http://api.test", apiKey: "k1", fetchImpl: f.impl, maxAttempts: 3, sleep: noSleep })
    await expect(client.health()).rejects.toBeInstanceOf(MemvaraHttpError)
    expect(f.calls).toHaveLength(3)
  })

  test("does not retry a 4xx and surfaces the body", async () => {
    const f = fakeFetch([{ status: 400, body: { error: { code: "bad_request", message: "no" } } }])
    const client = new MemvaraClient({ baseUrl: "http://api.test", apiKey: "k1", fetchImpl: f.impl, sleep: noSleep })
    const err = await client.stats("q-1").catch((e) => e)
    expect(err).toBeInstanceOf(MemvaraHttpError)
    expect((err as MemvaraHttpError).status).toBe(400)
    expect((err as MemvaraHttpError).body).toContain("bad_request")
    expect(f.calls).toHaveLength(1)
  })

  test("eraseUser posts the scope and never confirm_tenant", async () => {
    const f = fakeFetch([{ status: 200, body: { target: "scope", memory_id: null, scope: { user: "q-1" }, erased: true, counts: { claims: 3 } } }])
    const client = new MemvaraClient({ baseUrl: "http://api.test", apiKey: "k1", fetchImpl: f.impl, sleep: noSleep })
    const out = await client.eraseUser("q-1")
    expect(out.erased).toBe(true)
    expect(f.calls[0].url).toBe("http://api.test/v1/erasures?user=q-1")
    expect(JSON.parse(String(f.calls[0].init.body))).toEqual({ scope: { user: "q-1" } })
  })

  test("strips a trailing slash from the base URL", async () => {
    const f = fakeFetch([{ status: 200, body: { status: "ok", memvara_version: "1" } }])
    const client = new MemvaraClient({ baseUrl: "http://api.test/", apiKey: "k1", fetchImpl: f.impl, sleep: noSleep })
    await client.health()
    expect(f.calls[0].url).toBe("http://api.test/v1/health")
  })
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /Applications/workstation/memorybench && bun test src/providers/memvara/client.test.ts`
Expected: FAIL, `Cannot find module './client'`.

- [ ] **Step 3: Write the client**

```typescript
// src/providers/memvara/client.ts
/**
 * Memvara's REST API, the part MemoryBench needs. HTTP only: this file knows nothing
 * about sessions, questions or prompts.
 *
 * Scope: the credential is bound to a tenant, and every route narrows it with a `user`
 * query parameter. MemoryBench's container tag travels as that `user`, so one
 * question's haystack is one memvara user and a request under one tag cannot see
 * another tag's memories.
 */

export interface MemvaraMessage {
  role: string
  content: string
  ts?: string
  metadata?: Record<string, unknown>
}

export interface MemvaraAddRequest {
  messages: MemvaraMessage[]
  ts?: string
}

export interface MemvaraWriteReceipt {
  episode_ids: string[]
  added: unknown[]
  invalidated: unknown[]
  reinforced: unknown[]
  skipped: number
  unextracted: number
  llm_calls: number
  latency_ms: number
  deferred: boolean
  note: string | null
}

export interface MemvaraSearchRequest {
  query: string
  k: number
  min_score: number
  include_episodes: boolean
}

export interface MemvaraMemory {
  id: string
  text: string
  subject: string
  predicate: string
  object: string
  memory_type: string
  state: string
  valid_time: { valid_from: string; valid_to: string | null }
  transaction_time: { recorded_at: string; invalidated_at: string | null }
  confidence: number
  salience: number
  source_ids: string[]
}

export interface MemvaraEpisode {
  id: string
  role: string
  ts: string
  content: string
}

export type MemvaraHit =
  | { kind: "claim"; score: number; memory: MemvaraMemory }
  | { kind: "episode"; score: number; episode: MemvaraEpisode }

export interface MemvaraSearchResponse {
  count: number
  results: MemvaraHit[]
}

export interface MemvaraWhoAmI {
  token_id: string
  scope: { tenant: string; user?: string | null; agent?: string | null; session?: string | null }
  granted_privilege: string
  effective_privilege: string
  read_only: boolean
}

export interface MemvaraHealth {
  status: string
  memvara_version: string
}

export interface MemvaraStats {
  scope: unknown
  visible: number
  tenant_counts: Record<string, number>
  extractor: string
  read_only: boolean
}

export interface MemvaraErasure {
  target: string
  memory_id: string | null
  scope: unknown
  erased: boolean
  counts: Record<string, number> | null
}

export class MemvaraHttpError extends Error {
  constructor(
    public readonly status: number,
    public readonly body: string,
    message: string
  ) {
    super(message)
    this.name = "MemvaraHttpError"
  }
}

export interface MemvaraClientOptions {
  baseUrl: string
  apiKey: string
  fetchImpl?: typeof fetch
  maxAttempts?: number
  baseDelayMs?: number
  sleep?: (ms: number) => Promise<void>
}

// 429 is the API's rate limit and quota answer; the 5xx family is the stack being
// restarted or Postgres being busy. 4xx other than 429 is a request that will not get
// better by being repeated.
const RETRYABLE = new Set([408, 425, 429, 500, 502, 503, 504])
const MAX_DELAY_MS = 5000

interface RequestOptions {
  user?: string
  body?: unknown
  headers?: Record<string, string>
}

export class MemvaraClient {
  private readonly baseUrl: string
  private readonly apiKey: string
  private readonly fetchImpl: typeof fetch
  private readonly maxAttempts: number
  private readonly baseDelayMs: number
  private readonly sleep: (ms: number) => Promise<void>

  constructor(opts: MemvaraClientOptions) {
    this.baseUrl = opts.baseUrl.replace(/\/+$/, "")
    this.apiKey = opts.apiKey
    this.fetchImpl = opts.fetchImpl ?? fetch
    this.maxAttempts = opts.maxAttempts ?? 5
    this.baseDelayMs = opts.baseDelayMs ?? 500
    this.sleep = opts.sleep ?? ((ms) => new Promise((r) => setTimeout(r, ms)))
  }

  whoami(): Promise<MemvaraWhoAmI> {
    return this.request("GET", "/v1/whoami", {})
  }

  health(): Promise<MemvaraHealth> {
    return this.request("GET", "/v1/health", {})
  }

  addMemories(user: string, body: MemvaraAddRequest, idempotencyKey: string): Promise<MemvaraWriteReceipt> {
    return this.request("POST", "/v1/memories", { user, body, headers: { "Idempotency-Key": idempotencyKey } })
  }

  search(user: string, body: MemvaraSearchRequest): Promise<MemvaraSearchResponse> {
    return this.request("POST", "/v1/search", { user, body })
  }

  stats(user: string): Promise<MemvaraStats> {
    return this.request("GET", "/v1/stats", { user })
  }

  /** Erase everything under one user scope: claims, turns, vectors, index entries.
   *  `confirm_tenant` is deliberately never sent, so a request that lost its user
   *  is refused by the API rather than erasing the tenant. */
  eraseUser(user: string): Promise<MemvaraErasure> {
    return this.request("POST", "/v1/erasures", { user, body: { scope: { user } } })
  }

  private async request<T>(method: "GET" | "POST", path: string, opts: RequestOptions): Promise<T> {
    const url = opts.user === undefined
      ? `${this.baseUrl}${path}`
      : `${this.baseUrl}${path}?user=${encodeURIComponent(opts.user)}`
    const headers: Record<string, string> = {
      Authorization: `Bearer ${this.apiKey}`,
      Accept: "application/json",
      ...(opts.body !== undefined ? { "Content-Type": "application/json" } : {}),
      ...(opts.headers ?? {}),
    }
    const init: RequestInit = {
      method,
      headers,
      ...(opts.body !== undefined ? { body: JSON.stringify(opts.body) } : {}),
    }

    let lastError: Error | null = null
    for (let attempt = 1; attempt <= this.maxAttempts; attempt++) {
      let response: Response
      try {
        response = await this.fetchImpl(url, init)
      } catch (e) {
        lastError = e instanceof Error ? e : new Error(String(e))
        if (attempt < this.maxAttempts) await this.sleep(this.delay(attempt))
        continue
      }
      if (response.ok) {
        return (await response.json()) as T
      }
      const text = await response.text()
      lastError = new MemvaraHttpError(response.status, text, `${method} ${path} -> ${response.status}: ${text.slice(0, 300)}`)
      if (!RETRYABLE.has(response.status)) throw lastError
      if (attempt < this.maxAttempts) {
        const retryAfter = Number(response.headers.get("Retry-After"))
        await this.sleep(Number.isFinite(retryAfter) && retryAfter > 0 ? Math.min(retryAfter * 1000, MAX_DELAY_MS) : this.delay(attempt))
      }
    }
    throw lastError ?? new Error(`${method} ${path}: no attempts made`)
  }

  private delay(attempt: number): number {
    return Math.min(this.baseDelayMs * 2 ** (attempt - 1), MAX_DELAY_MS)
  }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `bun test src/providers/memvara/client.test.ts`
Expected: 8 pass, 0 fail.

- [ ] **Step 5: Format and commit**

Run:
```bash
bunx prettier --write src/providers/memvara/client.ts src/providers/memvara/client.test.ts
bun test src/providers/memvara/client.test.ts
git add src/providers/memvara/client.ts src/providers/memvara/client.test.ts
git commit -m "Add a typed HTTP client for memvara's REST API with retry and idempotency"
```

---

### Task 3: The answer prompt

**Files:**
- Create: `src/providers/memvara/prompts.ts`
- Test: `src/providers/memvara/prompts.test.ts`

**Interfaces:**
- Consumes: nothing from the client; it renders plain objects.
- Produces:
  ```typescript
  export interface MemvaraContextMemory { kind: "memory"; text: string; subject: string; predicate: string; object: string; state: string; valid_from: string; valid_to: string | null; recorded_at: string; invalidated_at: string | null; score: number; sources: string[] }
  export interface MemvaraContextTurn { kind: "turn"; role: string; content: string; ts: string; score: number }
  export type MemvaraContextItem = MemvaraContextMemory | MemvaraContextTurn
  export function renderMemvaraContext(context: unknown[]): string
  export function buildMemvaraAnswerPrompt(question: string, context: unknown[], questionDate?: string): string
  export const MEMVARA_PROMPTS: ProviderPrompts
  ```
  Task 4's `search` returns exactly `MemvaraContextItem[]`.

- [ ] **Step 1: Write the failing tests**

```typescript
// src/providers/memvara/prompts.test.ts
import { describe, expect, test } from "bun:test"
import { buildMemvaraAnswerPrompt, renderMemvaraContext, MEMVARA_PROMPTS } from "./prompts"
import type { MemvaraContextItem } from "./prompts"

const memory: MemvaraContextItem = {
  kind: "memory",
  text: "user lives in Lisbon",
  subject: "user",
  predicate: "lives_in",
  object: "Lisbon",
  state: "live",
  valid_from: "2023-05-20T02:21:00+00:00",
  valid_to: null,
  recorded_at: "2023-05-20T02:21:00+00:00",
  invalidated_at: null,
  score: 0.61,
  sources: ["ep_1"],
}
const ended: MemvaraContextItem = {
  ...memory,
  text: "user lives in Porto",
  object: "Porto",
  state: "ended",
  valid_from: "2023-01-02T10:00:00+00:00",
  valid_to: "2023-05-20T02:21:00+00:00",
  invalidated_at: "2023-05-20T02:21:00+00:00",
  score: 0.4,
}
const turn: MemvaraContextItem = {
  kind: "turn",
  role: "user",
  content: "I moved to Lisbon last week!",
  ts: "2023-05-20T02:21:00+00:00",
  score: 0.44,
}

describe("renderMemvaraContext", () => {
  test("renders memories with both clocks and turns with their date", () => {
    const out = renderMemvaraContext([memory, ended, turn])
    expect(out).toContain("Memories")
    expect(out).toContain("[valid from 2023-05-20 02:21, recorded 2023-05-20 02:21, live] user lives in Lisbon")
    expect(out).toContain("[valid from 2023-01-02 10:00 to 2023-05-20 02:21, recorded 2023-05-20 02:21, ended] user lives in Porto")
    expect(out).toContain("Conversation excerpts")
    expect(out).toContain("[2023-05-20 02:21] user: I moved to Lisbon last week!")
  })

  test("keeps memvara's order inside each block", () => {
    const out = renderMemvaraContext([ended, memory])
    expect(out.indexOf("Porto")).toBeLessThan(out.indexOf("Lisbon"))
  })

  test("says so when there is nothing", () => {
    expect(renderMemvaraContext([])).toContain("No memories were retrieved.")
  })

  test("ignores objects it does not recognise instead of throwing", () => {
    const out = renderMemvaraContext([{ foo: "bar" }, memory])
    expect(out).toContain("user lives in Lisbon")
  })
})

describe("buildMemvaraAnswerPrompt", () => {
  test("carries the question, the question date, the context and the abstention rule", () => {
    const prompt = buildMemvaraAnswerPrompt("Where do I live?", [memory, turn], "2023/06/01 (Thu) 09:00")
    expect(prompt).toContain("Question: Where do I live?")
    expect(prompt).toContain("Question date: 2023/06/01 (Thu) 09:00")
    expect(prompt).toContain("user lives in Lisbon")
    expect(prompt).toContain("I don't know")
    expect(prompt).toContain("Answer:")
  })

  test("says the date is not specified when none is given", () => {
    expect(buildMemvaraAnswerPrompt("q", [])).toContain("Question date: not specified")
  })

  test("is what the provider exports as its answer prompt", () => {
    expect(MEMVARA_PROMPTS.answerPrompt).toBe(buildMemvaraAnswerPrompt)
    expect(MEMVARA_PROMPTS.judgePrompt).toBeUndefined()
  })
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `bun test src/providers/memvara/prompts.test.ts`
Expected: FAIL, `Cannot find module './prompts'`.

- [ ] **Step 3: Write the prompt module**

```typescript
// src/providers/memvara/prompts.ts
import type { ProviderPrompts } from "../../types/prompts"

/** A memory as `MemvaraProvider.search` returns it: memvara's claim with both clocks. */
export interface MemvaraContextMemory {
  kind: "memory"
  text: string
  subject: string
  predicate: string
  object: string
  state: string
  valid_from: string
  valid_to: string | null
  recorded_at: string
  invalidated_at: string | null
  score: number
  sources: string[]
}

/** A raw conversation turn, present when the search included episodes. */
export interface MemvaraContextTurn {
  kind: "turn"
  role: string
  content: string
  ts: string
  score: number
}

export type MemvaraContextItem = MemvaraContextMemory | MemvaraContextTurn

function isMemory(x: unknown): x is MemvaraContextMemory {
  return typeof x === "object" && x !== null && (x as { kind?: unknown }).kind === "memory"
}

function isTurn(x: unknown): x is MemvaraContextTurn {
  return typeof x === "object" && x !== null && (x as { kind?: unknown }).kind === "turn"
}

/** `2023-05-20T02:21:00+00:00` -> `2023-05-20 02:21`. The reader needs a date it can
 *  do arithmetic on, not seconds and an offset. Anything unparseable passes through. */
export function formatWhen(iso: string): string {
  const m = /^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})/.exec(iso)
  return m ? `${m[1]} ${m[2]}` : iso
}

function memoryLine(m: MemvaraContextMemory): string {
  const validity = m.valid_to
    ? `valid from ${formatWhen(m.valid_from)} to ${formatWhen(m.valid_to)}`
    : `valid from ${formatWhen(m.valid_from)}`
  return `- [${validity}, recorded ${formatWhen(m.recorded_at)}, ${m.state}] ${m.text}`
}

function turnLine(t: MemvaraContextTurn): string {
  return `- [${formatWhen(t.ts)}] ${t.role}: ${t.content}`
}

/** Memories first, then the raw turns, each in the order memvara ranked them. Nothing
 *  is dropped, merged or re-sorted here: this is a rendering of the ranking, and the
 *  ranking is what the benchmark measures. */
export function renderMemvaraContext(context: unknown[]): string {
  const memories = context.filter(isMemory)
  const turns = context.filter(isTurn)
  if (memories.length === 0 && turns.length === 0) {
    return "No memories were retrieved."
  }
  const parts: string[] = []
  if (memories.length > 0) {
    parts.push("Memories (each with the period it was true for and the date it was recorded):\n" + memories.map(memoryLine).join("\n"))
  }
  if (turns.length > 0) {
    parts.push("Conversation excerpts (verbatim, with the date they were said):\n" + turns.map(turnLine).join("\n"))
  }
  return parts.join("\n\n")
}

export function buildMemvaraAnswerPrompt(question: string, context: unknown[], questionDate?: string): string {
  return `You are a question-answering system with access to a memory of past conversations with the user. Answer the question from the retrieved context below.

Question: ${question}
Question date: ${questionDate || "not specified"}

Retrieved context:
${renderMemvaraContext(context)}

How to read the context:
- A memory is a fact the memory system extracted. "valid from" is when the fact became true in the world; "recorded" is when the system learned it. A memory marked "ended" was true for the period shown and has since been replaced; prefer the "live" memory for what is true now, and use "ended" memories for what was true earlier.
- A conversation excerpt is what was actually said, with the date it was said. Use excerpts for details and wording that a memory summarises.
- Resolve relative expressions such as "today", "yesterday", "last week" or "in two months" against the date of the excerpt or memory they appear in, never against the current date. Use the question date only to understand what the question is asking about.

Instructions:
- Think through the problem step by step first.
- Identify which memories and excerpts are relevant, and whether any memory has been updated by a later one.
- If the context contains enough information, give a clear, concise answer.
- If it does not, answer "I don't know" and say what is missing. Do not guess.
- Base the answer only on the context above.

Response format:

Reasoning:
[your step-by-step reasoning]

Answer:
[your final answer]`
}

export const MEMVARA_PROMPTS: ProviderPrompts = {
  answerPrompt: buildMemvaraAnswerPrompt,
}

export default MEMVARA_PROMPTS
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `bun test src/providers/memvara/prompts.test.ts`
Expected: 7 pass.

- [ ] **Step 5: Format and commit**

```bash
bunx prettier --write src/providers/memvara/prompts.ts src/providers/memvara/prompts.test.ts
bun test src/providers/memvara/prompts.test.ts
git add src/providers/memvara/prompts.ts src/providers/memvara/prompts.test.ts
git commit -m "Render memvara's memories and turns, with both clocks, into the answer prompt"
```

---

### Task 4: The provider

**Files:**
- Create: `src/providers/memvara/index.ts`
- Test: `src/providers/memvara/index.test.ts`

**Interfaces:**
- Consumes: `MemvaraClient` and its types from Task 2; `MEMVARA_PROMPTS`, `MemvaraContextItem` from Task 3; `Provider`, `ProviderConfig`, `IngestOptions`, `IngestResult`, `SearchOptions`, `IndexingProgressCallback` from `src/types/provider.ts`; `UnifiedSession` from `src/types/unified.ts`; `logger` from `src/utils/logger.ts`.
- Produces: `export class MemvaraProvider implements Provider` with a constructor `(clientFactory?: (config: ProviderConfig) => MemvaraClient)`; `export default MemvaraProvider`.

- [ ] **Step 1: Write the failing tests**

```typescript
// src/providers/memvara/index.test.ts
import { describe, expect, test } from "bun:test"
import { MemvaraProvider } from "./index"
import type { MemvaraClient, MemvaraAddRequest, MemvaraSearchRequest } from "./client"
import type { UnifiedSession } from "../../types/unified"

type Recorded = { method: string; args: unknown[] }

function fakeClient(overrides: Partial<Record<keyof MemvaraClient, unknown>> = {}) {
  const calls: Recorded[] = []
  const record = (method: string) => (...args: unknown[]) => {
    calls.push({ method, args })
    const fn = overrides[method as keyof MemvaraClient]
    return typeof fn === "function" ? (fn as (...a: unknown[]) => unknown)(...args) : undefined
  }
  const client = {
    whoami: record("whoami"),
    health: record("health"),
    addMemories: record("addMemories"),
    search: record("search"),
    stats: record("stats"),
    eraseUser: record("eraseUser"),
  } as unknown as MemvaraClient
  return { client, calls }
}

const session: UnifiedSession = {
  sessionId: "q1-session-0",
  messages: [
    { role: "user", content: "I moved to Lisbon last week!" },
    { role: "assistant", content: "Congratulations." },
  ],
  metadata: { date: "2023-05-20T02:21:00.000Z", formattedDate: "Saturday, May 20, 2023 at 2:21 AM" },
}

async function initialised(overrides: Partial<Record<keyof MemvaraClient, unknown>> = {}) {
  const fake = fakeClient({
    whoami: async () => ({ token_id: "t", scope: { tenant: "prj_x" }, granted_privilege: "admin", effective_privilege: "admin", read_only: false }),
    health: async () => ({ status: "ok", memvara_version: "0.9.0" }),
    ...overrides,
  })
  const provider = new MemvaraProvider(() => fake.client)
  await provider.initialize({ apiKey: "k", baseUrl: "http://api.test" })
  return { provider, ...fake }
}

describe("MemvaraProvider", () => {
  test("is named memvara, ships the prompt, and declares modest concurrency", () => {
    const p = new MemvaraProvider()
    expect(p.name).toBe("memvara")
    expect(typeof p.prompts?.answerPrompt).toBe("function")
    expect(p.concurrency).toEqual({ default: 4, ingest: 6, indexing: 8, search: 4, answer: 8, evaluate: 8 })
  })

  test("initialize checks whoami and health, and refuses a read-only credential", async () => {
    const { calls } = await initialised()
    expect(calls.map((c) => c.method)).toEqual(["whoami", "health"])
    const fake = fakeClient({
      whoami: async () => ({ token_id: "t", scope: { tenant: "prj_x" }, granted_privilege: "read", effective_privilege: "read", read_only: true }),
      health: async () => ({ status: "ok", memvara_version: "0.9.0" }),
    })
    const p = new MemvaraProvider(() => fake.client)
    await expect(p.initialize({ apiKey: "k" })).rejects.toThrow(/read-only/)
  })

  test("methods refuse before initialize", async () => {
    const p = new MemvaraProvider()
    await expect(p.search("q", { containerTag: "c" })).rejects.toThrow(/not initialized/)
  })

  test("ingest writes one request per session with the session date on every turn and an idempotency key", async () => {
    const { provider, calls } = await initialised({
      addMemories: async () => ({ episode_ids: ["ep_a", "ep_b"], added: [], invalidated: [], reinforced: [], skipped: 0, unextracted: 0, llm_calls: 0, latency_ms: 1, deferred: false, note: null }),
    })
    const out = await provider.ingest([session], { containerTag: "q1-run7" })
    expect(out).toEqual({ documentIds: ["ep_a", "ep_b"] })
    const add = calls.find((c) => c.method === "addMemories")!
    const [user, body, key] = add.args as [string, MemvaraAddRequest, string]
    expect(user).toBe("q1-run7")
    expect(key).toBe("q1-run7:q1-session-0")
    expect(body.ts).toBe("2023-05-20T02:21:00.000Z")
    expect(body.messages).toEqual([
      { role: "user", content: "I moved to Lisbon last week!", ts: "2023-05-20T02:21:00.000Z", metadata: { sessionId: "q1-session-0" } },
      { role: "assistant", content: "Congratulations.", ts: "2023-05-20T02:21:00.000Z", metadata: { sessionId: "q1-session-0" } },
    ])
  })

  test("ingest prefers a per-message timestamp when the benchmark gives one", async () => {
    const { provider, calls } = await initialised({
      addMemories: async () => ({ episode_ids: ["ep_a"], added: [], invalidated: [], reinforced: [], skipped: 0, unextracted: 0, llm_calls: 0, latency_ms: 1, deferred: false, note: null }),
    })
    const s: UnifiedSession = { sessionId: "s", messages: [{ role: "user", content: "x", timestamp: "2024-01-01T00:00:00.000Z" }], metadata: { date: "2023-05-20T02:21:00.000Z" } }
    await provider.ingest([s], { containerTag: "c" })
    const body = calls.find((c) => c.method === "addMemories")!.args[1] as MemvaraAddRequest
    expect(body.messages[0].ts).toBe("2024-01-01T00:00:00.000Z")
  })

  test("ingest skips a session with no messages", async () => {
    const { provider, calls } = await initialised()
    const out = await provider.ingest([{ sessionId: "empty", messages: [] }], { containerTag: "c" })
    expect(out).toEqual({ documentIds: [] })
    expect(calls.some((c) => c.method === "addMemories")).toBe(false)
  })

  test("awaitIndexing resolves, reads stats, and reports every id complete", async () => {
    const { provider, calls } = await initialised({
      stats: async () => ({ scope: {}, visible: 3, tenant_counts: { episodes: 2 }, extractor: "fast/v1", read_only: false }),
    })
    const seen: unknown[] = []
    await provider.awaitIndexing({ documentIds: ["ep_a", "ep_b"] }, "c", (p) => seen.push(p))
    expect(calls.find((c) => c.method === "stats")!.args[0]).toBe("c")
    expect(seen).toEqual([{ completedIds: ["ep_a", "ep_b"], failedIds: [], total: 2 }])
  })

  test("search asks for 30 with episodes and no floor, and returns plain memory and turn objects in order", async () => {
    const { provider, calls } = await initialised({
      search: async () => ({
        count: 2,
        results: [
          { kind: "claim", score: 0.61, ranking: {}, memory: { id: "cl_1", text: "user lives in Lisbon", subject: "user", predicate: "lives_in", object: "Lisbon", memory_type: "semantic", state: "live", valid_time: { valid_from: "2023-05-20T02:21:00+00:00", valid_to: null }, transaction_time: { recorded_at: "2023-05-20T02:21:00+00:00", invalidated_at: null }, confidence: 1, salience: 1, source_ids: ["ep_a"] } },
          { kind: "episode", score: 0.44, ranking: {}, episode: { id: "ep_a", role: "user", ts: "2023-05-20T02:21:00+00:00", content: "I moved to Lisbon last week!" } },
        ],
      }),
    })
    const out = await provider.search("where do I live", { containerTag: "q1-run7", limit: 10, threshold: 0.3 })
    const [user, body] = calls.find((c) => c.method === "search")!.args as [string, MemvaraSearchRequest]
    expect(user).toBe("q1-run7")
    expect(body).toEqual({ query: "where do I live", k: 30, min_score: 0, include_episodes: true })
    expect(out).toEqual([
      { kind: "memory", text: "user lives in Lisbon", subject: "user", predicate: "lives_in", object: "Lisbon", state: "live", valid_from: "2023-05-20T02:21:00+00:00", valid_to: null, recorded_at: "2023-05-20T02:21:00+00:00", invalidated_at: null, score: 0.61, sources: ["ep_a"] },
      { kind: "turn", role: "user", content: "I moved to Lisbon last week!", ts: "2023-05-20T02:21:00+00:00", score: 0.44 },
    ])
  })

  test("clear erases the user scope", async () => {
    const { provider, calls } = await initialised({
      eraseUser: async () => ({ target: "scope", memory_id: null, scope: { user: "c" }, erased: true, counts: { claims: 1 } }),
    })
    await provider.clear("c")
    expect(calls.find((c) => c.method === "eraseUser")!.args).toEqual(["c"])
  })
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `bun test src/providers/memvara/index.test.ts`
Expected: FAIL, `Cannot find module './index'`.

- [ ] **Step 3: Write the provider**

```typescript
// src/providers/memvara/index.ts
import type {
  Provider,
  ProviderConfig,
  IngestOptions,
  IngestResult,
  SearchOptions,
  IndexingProgressCallback,
} from "../../types/provider"
import type { UnifiedSession } from "../../types/unified"
import { logger } from "../../utils/logger"
import { MemvaraClient } from "./client"
import type { MemvaraHit, MemvaraMessage } from "./client"
import { MEMVARA_PROMPTS } from "./prompts"
import type { MemvaraContextItem } from "./prompts"

const DEFAULT_BASE_URL = "http://127.0.0.1:58080"

/** What memvara is asked for on every search. `k: 30` is what the shipped providers ask
 *  their services for; the orchestrator's `limit: 10` and `threshold: 0.3` are ignored
 *  here for the same reason they ignore them, and memvara's score is not on the scale
 *  that threshold was set for. No floor: this measures the ranking as shipped. */
const SEARCH_K = 30
const SEARCH_MIN_SCORE = 0

export class MemvaraProvider implements Provider {
  name = "memvara"
  prompts = MEMVARA_PROMPTS
  concurrency = {
    default: 4,
    ingest: 6,
    indexing: 8,
    search: 4,
    answer: 8,
    evaluate: 8,
  }
  private client: MemvaraClient | null = null
  private readonly clientFactory: (config: ProviderConfig) => MemvaraClient

  constructor(clientFactory?: (config: ProviderConfig) => MemvaraClient) {
    this.clientFactory =
      clientFactory ??
      ((config) => new MemvaraClient({ apiKey: config.apiKey, baseUrl: config.baseUrl || DEFAULT_BASE_URL }))
  }

  async initialize(config: ProviderConfig): Promise<void> {
    if (!config.apiKey) throw new Error("MEMVARA_API_KEY is not set")
    const client = this.clientFactory(config)
    const who = await client.whoami()
    if (who.read_only) {
      throw new Error(`memvara credential ${who.token_id} is read-only; the benchmark writes`)
    }
    const health = await client.health()
    this.client = client
    logger.info(
      `Initialized memvara provider: ${config.baseUrl || DEFAULT_BASE_URL}, tenant ${who.scope.tenant}, ` +
        `privilege ${who.effective_privilege}, memvara ${health.memvara_version}`
    )
  }

  async ingest(sessions: UnifiedSession[], options: IngestOptions): Promise<IngestResult> {
    const client = this.ready()
    const documentIds: string[] = []
    for (const session of sessions) {
      if (session.messages.length === 0) continue
      const sessionDate = typeof session.metadata?.date === "string" ? session.metadata.date : undefined
      const messages: MemvaraMessage[] = session.messages.map((m) => ({
        role: m.role,
        content: m.content,
        ...(m.timestamp || sessionDate ? { ts: m.timestamp || sessionDate } : {}),
        metadata: { sessionId: session.sessionId },
      }))
      // One key per logical write. A retry after a timeout replays the first receipt
      // instead of storing the session twice.
      const idempotencyKey = `${options.containerTag}:${session.sessionId}`
      const receipt = await client.addMemories(
        options.containerTag,
        { messages, ...(sessionDate ? { ts: sessionDate } : {}) },
        idempotencyKey
      )
      documentIds.push(...receipt.episode_ids)
      logger.debug(
        `Ingested ${session.sessionId}: ${receipt.episode_ids.length} turns, ${receipt.added.length} memories added`
      )
    }
    return { documentIds }
  }

  /** Memvara's write is synchronous: the receipt comes back after the turns are stored,
   *  embedded, indexed and extracted. There is nothing to wait for; one stats read
   *  confirms the scope is populated and puts the count in the log. */
  async awaitIndexing(result: IngestResult, containerTag: string, onProgress?: IndexingProgressCallback): Promise<void> {
    const client = this.ready()
    const stats = await client.stats(containerTag)
    logger.debug(`Scope ${containerTag}: ${stats.visible} memories visible, counts ${JSON.stringify(stats.tenant_counts)}`)
    onProgress?.({ completedIds: [...result.documentIds], failedIds: [], total: result.documentIds.length })
  }

  async search(query: string, options: SearchOptions): Promise<unknown[]> {
    const client = this.ready()
    const response = await client.search(options.containerTag, {
      query,
      k: SEARCH_K,
      min_score: SEARCH_MIN_SCORE,
      include_episodes: true,
    })
    return response.results.map(toContextItem)
  }

  async clear(containerTag: string): Promise<void> {
    const client = this.ready()
    const out = await client.eraseUser(containerTag)
    logger.debug(`Erased scope ${containerTag}: ${JSON.stringify(out.counts)}`)
  }

  private ready(): MemvaraClient {
    if (!this.client) throw new Error("Provider not initialized")
    return this.client
  }
}

/** memvara's hit, flattened to what the prompt renders. Nothing is dropped or reordered. */
function toContextItem(hit: MemvaraHit): MemvaraContextItem {
  if (hit.kind === "claim") {
    const m = hit.memory
    return {
      kind: "memory",
      text: m.text,
      subject: m.subject,
      predicate: m.predicate,
      object: m.object,
      state: m.state,
      valid_from: m.valid_time.valid_from,
      valid_to: m.valid_time.valid_to,
      recorded_at: m.transaction_time.recorded_at,
      invalidated_at: m.transaction_time.invalidated_at,
      score: hit.score,
      sources: m.source_ids,
    }
  }
  const e = hit.episode
  return { kind: "turn", role: e.role, content: e.content, ts: e.ts, score: hit.score }
}

export default MemvaraProvider
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `bun test src/providers/memvara/`
Expected: all client, prompt and provider tests pass (24).

- [ ] **Step 5: Format and commit**

```bash
bunx prettier --write src/providers/memvara/index.ts src/providers/memvara/index.test.ts
bun test src/providers/memvara/
git add src/providers/memvara/index.ts src/providers/memvara/index.test.ts
git commit -m "Add the memvara provider: sessions become scoped writes, hits become memories and turns"
```

---

### Task 5: Registration, config, README

**Files:**
- Modify: `src/providers/index.ts`
- Modify: `src/types/provider.ts` (the `ProviderName` line)
- Modify: `src/utils/config.ts`
- Modify: `src/providers/README.md`

**Interfaces:**
- Consumes: `MemvaraProvider` from Task 4.
- Produces: `bun run src/index.ts run -p memvara ...` resolves the provider; `getProviderConfig("memvara")` returns `{ apiKey: MEMVARA_API_KEY, baseUrl: MEMVARA_BASE_URL }`.

- [ ] **Step 1: Write the failing test**

Append to `src/providers/memvara/index.test.ts`:

```typescript
import { createProvider, getAvailableProviders } from "../index"
import { getProviderConfig } from "../../utils/config"

describe("registration", () => {
  test("memvara is a known provider", () => {
    expect(getAvailableProviders()).toContain("memvara")
    expect(createProvider("memvara").name).toBe("memvara")
  })

  test("config reads the memvara key and base URL, with the local stack as the default URL", () => {
    const saved = { key: process.env.MEMVARA_API_KEY, url: process.env.MEMVARA_BASE_URL }
    process.env.MEMVARA_API_KEY = "abc"
    delete process.env.MEMVARA_BASE_URL
    try {
      // config.ts reads the environment at import time, so re-import it fresh.
      delete require.cache[require.resolve("../../utils/config")]
      const { getProviderConfig: fresh } = require("../../utils/config") as typeof import("../../utils/config")
      expect(fresh("memvara")).toEqual({ apiKey: "abc", baseUrl: "http://127.0.0.1:58080" })
    } finally {
      if (saved.key !== undefined) process.env.MEMVARA_API_KEY = saved.key
      else delete process.env.MEMVARA_API_KEY
      if (saved.url !== undefined) process.env.MEMVARA_BASE_URL = saved.url
    }
  })

  test("getProviderConfig knows memvara", () => {
    expect(() => getProviderConfig("memvara")).not.toThrow()
  })
})
```

- [ ] **Step 2: Run it to verify it fails**

Run: `bun test src/providers/memvara/index.test.ts`
Expected: FAIL on `toContain("memvara")` and `Unknown provider: memvara`.

- [ ] **Step 3: Register the provider**

In `src/types/provider.ts`, change the last line to:

```typescript
export type ProviderName = "supermemory" | "mem0" | "zep" | "filesystem" | "rag" | "memvara"
```

In `src/providers/index.ts`, add the import, the map entry and the export:

```typescript
import { MemvaraProvider } from "./memvara"

const providers: Record<ProviderName, new () => Provider> = {
  supermemory: SupermemoryProvider,
  mem0: Mem0Provider,
  zep: ZepProvider,
  filesystem: FilesystemProvider,
  rag: RAGProvider,
  memvara: MemvaraProvider,
}

export { SupermemoryProvider, Mem0Provider, ZepProvider, FilesystemProvider, RAGProvider, MemvaraProvider }
```

In `src/utils/config.ts`, add two fields, two environment reads, and one case:

```typescript
export interface Config {
  supermemoryApiKey: string
  supermemoryBaseUrl: string
  mem0ApiKey: string
  zepApiKey: string
  memvaraApiKey: string
  memvaraBaseUrl: string
  openaiApiKey: string
  anthropicApiKey: string
  googleApiKey: string
}

export const config: Config = {
  supermemoryApiKey: process.env.SUPERMEMORY_API_KEY || "",
  supermemoryBaseUrl: process.env.SUPERMEMORY_BASE_URL || "https://api.supermemory.ai",
  mem0ApiKey: process.env.MEM0_API_KEY || "",
  zepApiKey: process.env.ZEP_API_KEY || "",
  memvaraApiKey: process.env.MEMVARA_API_KEY || "",
  memvaraBaseUrl: process.env.MEMVARA_BASE_URL || "http://127.0.0.1:58080",
  openaiApiKey: process.env.OPENAI_API_KEY || "",
  anthropicApiKey: process.env.ANTHROPIC_API_KEY || "",
  googleApiKey: process.env.GOOGLE_API_KEY || "",
}
```

and in `getProviderConfig`:

```typescript
    case "memvara":
      return { apiKey: config.memvaraApiKey, baseUrl: config.memvaraBaseUrl }
```

- [ ] **Step 4: Run all tests and the CLI listing**

Run:
```bash
bun test
bun run src/index.ts run --help | grep -i "provider"
bunx prettier --check "src/**/*.ts"
```
Expected: all tests pass; the provider help line lists `memvara`; prettier reports no issues (run `bunx prettier --write` on the four files if it does).

- [ ] **Step 5: Document the provider**

Add to `src/providers/README.md`, in the "Existing Providers" table:

```markdown
| `memvara` | REST (`fetch`) | Bitemporal claims plus raw turns; per-question container tag is a memvara `user` scope; `clear` is a scope erasure |
```

and a section after it:

```markdown
## Running memvara against a local stack

memvara's REST API is memvara-cloud's compose stack. From a memvara-cloud checkout:

```bash
export MEMVARA_CORE_PATH=/path/to/agent-memory        # the core commit under test
export MEMVARA_QUOTA_ENFORCE=0                        # a benchmark ingests far past any plan's allowance
docker compose -f deploy/compose.yaml up -d --build
docker compose -f deploy/compose.yaml run --rm key    # the API key the seed step minted
```

Then in this repository's `.env.local`: `MEMVARA_API_KEY=<that key>` and
`MEMVARA_BASE_URL=http://127.0.0.1:58080`. The provider checks `/v1/whoami` and
`/v1/health` on initialize and logs the memvara version, so a run's log names the
engine build it measured.
```

- [ ] **Step 6: Commit**

```bash
git add src/providers/index.ts src/types/provider.ts src/utils/config.ts src/providers/README.md src/providers/memvara/index.test.ts
git commit -m "Register memvara as a provider and document running it against a local stack"
```

---

### Task 6: The local memvara stack

**Files:**
- Create: `/Applications/workstation/agent-memory/local/wt-memorybench` (a git worktree, ignored by the repository)
- Modify: `/Applications/workstation/memorybench/.env.local` (the key)

**Interfaces:**
- Produces: an API at `http://127.0.0.1:58080` answering `/v1/whoami` with an admin, non-read-only credential; the core sha and cloud sha for the results section.

- [ ] **Step 1: A clean checkout of the core commit under test**

Run:
```bash
cd /Applications/workstation/agent-memory
git fetch origin
git worktree add local/wt-memorybench origin/main
git -C local/wt-memorybench rev-parse --short HEAD
git -C local/wt-memorybench status --short | wc -l
```
Expected: a sha (write it down as CORE_SHA) and `0` modified files. This is the tree the image builds.

- [ ] **Step 2: Bring the stack up with quota enforcement off**

Run:
```bash
cd /Applications/workstation/memvara-cloud
git rev-parse --short HEAD          # CLOUD_SHA, write it down
git status --short | wc -l           # expect 0; the image builds the committed tree plus the working copy, so say so in the results if not
export MEMVARA_CORE_PATH=/Applications/workstation/agent-memory/local/wt-memorybench
export MEMVARA_QUOTA_ENFORCE=0
docker compose -f deploy/compose.yaml up -d --build
docker compose -f deploy/compose.yaml ps --format '{{.Service}} {{.Status}}'
```
Expected: `db` healthy, `migrate` exited 0, `api` healthy, `seed` exited 0, plus the daemons. The build takes several minutes cold. `MEMVARA_QUOTA_ENFORCE=0` is the compose file's own switch; without it the seeded project's memory ceiling (12,000 on the fallback plan) refuses writes with 402 partway through the ingest.

- [ ] **Step 3: Get the key and check the credential**

Run:
```bash
KEY=$(docker compose -f deploy/compose.yaml run --rm key | tail -1)
curl -s -H "Authorization: Bearer $KEY" http://127.0.0.1:58080/v1/whoami
curl -s http://127.0.0.1:58080/v1/health
```
Expected: whoami shows `"read_only": false` and an admin privilege; health shows `"status": "ok"` and a `memvara_version`. Put the key into `/Applications/workstation/memorybench/.env.local` as `MEMVARA_API_KEY=`. Do not paste it anywhere else.

- [ ] **Step 4: Check the write and erase round trip by hand**

Run:
```bash
curl -s -X POST "http://127.0.0.1:58080/v1/memories?user=smoke-user" -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -H "Idempotency-Key: smoke-1" -d '{"messages":[{"role":"user","content":"I moved to Lisbon last week!","ts":"2023-05-20T02:21:00Z"}],"ts":"2023-05-20T02:21:00Z"}'
curl -s -X POST "http://127.0.0.1:58080/v1/search?user=smoke-user" -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d '{"query":"where do I live","k":30,"min_score":0,"include_episodes":true}'
curl -s -X POST "http://127.0.0.1:58080/v1/erasures?user=smoke-user" -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d '{"scope":{"user":"smoke-user"}}'
curl -s "http://127.0.0.1:58080/v1/stats?user=smoke-user" -H "Authorization: Bearer $KEY"
```
Expected: a receipt with one `episode_ids` entry; a search whose results include an `"kind": "episode"` hit (and a claim if the fast path recognised the sentence); an erasure with `"erased": true`; stats with `"visible": 0`.

- [ ] **Step 5: Record the shas**

Write `CORE_SHA`, `CLOUD_SHA`, the `memvara_version` from health, and the date into `/Applications/workstation/agent-memory/local/memorybench/STACK.md`. Task 9 reads it.

---

### Task 7: One question end to end, then one per category

**Files:**
- Create: `/Applications/workstation/memorybench/data/` (harness state; ignored)

**Interfaces:**
- Consumes: the provider from Task 5, the stack from Task 6, `OPENAI_API_KEY` in `.env.local` (from the user).

- [ ] **Step 1: Confirm the keys are present without printing them**

Run: `cd /Applications/workstation/memorybench && grep -c "^OPENAI_API_KEY=.\+" .env.local && grep -c "^MEMVARA_API_KEY=.\+" .env.local`
Expected: `1` and `1`.

- [ ] **Step 2: Download the dataset and run one question**

The harness downloads `longmemeval_s_cleaned.json` (277 MB) on first use; the user approved the download in the spec.

Run:
```bash
bun run src/index.ts run -p memvara -b longmemeval -j gpt-4o -m gpt-4o -l 1 -r memvara-smoke-1
```
Expected: ingest of one question's haystack (about 40 sessions) at `INFO` level, then indexing, search, answer, evaluate, and a report. Any thrown error names the question; fix and resume with the same `-r`.

- [ ] **Step 3: Read the rendered prompt once, with human eyes**

Run:
```bash
Q=$(ls data/runs/memvara-smoke-1/results/ | head -1 | sed 's/.json//')
bun -e '
import { buildMemvaraAnswerPrompt } from "./src/providers/memvara/prompts"
const r = JSON.parse(await Bun.file(process.argv[1]).text())
console.log(buildMemvaraAnswerPrompt(r.question, r.results, "2023/06/01 (Thu) 09:00"))
' "data/runs/memvara-smoke-1/results/$Q.json" | head -80
```
Expected: a Memories block with dates and states, a Conversation excerpts block with dated turns, the question date, and the abstention rule. If the memories block is empty for a haystack of 40 sessions, the fast path extracted nothing from it, which is a real finding about the baseline and is written down, not fixed here.

- [ ] **Step 4: One question of every type, including abstention**

Run:
```bash
bun run src/index.ts run -p memvara -b longmemeval -j gpt-4o -m gpt-4o -s 1 -r memvara-smoke-types
bun run src/index.ts status -r memvara-smoke-types
```
Expected: seven questions (six types plus abstention, which the loader reports as its own type), all phases completed, success rate 100%.

- [ ] **Step 5: Prove clear works**

Run:
```bash
TAG=$(python3 -c "import json; d=json.load(open('data/runs/memvara-smoke-types/checkpoint.json')); q=next(iter(d['questions'])); print(f\"{q}-{d['dataSourceRunId']}\")")
KEY=$(grep '^MEMVARA_API_KEY=' .env.local | cut -d= -f2-)
curl -s "http://127.0.0.1:58080/v1/stats?user=$TAG" -H "Authorization: Bearer $KEY" | python3 -c "import sys,json; print('before', json.load(sys.stdin)['visible'])"
curl -s -X POST "http://127.0.0.1:58080/v1/erasures?user=$TAG" -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d "{\"scope\":{\"user\":\"$TAG\"}}" | python3 -c "import sys,json; print('erased', json.load(sys.stdin)['erased'])"
curl -s "http://127.0.0.1:58080/v1/stats?user=$TAG" -H "Authorization: Bearer $KEY" | python3 -c "import sys,json; print('after', json.load(sys.stdin)['visible'])"
```
Expected: `before <n>` with n > 0, `erased True`, `after 0`. If the checkpoint's field names differ from `questions` / `dataSourceRunId`, read `data/runs/memvara-smoke-types/checkpoint.json` and use the names it has; the container tag is `<questionId>-<dataSourceRunId>` per `src/orchestrator/phases/ingest.ts`.

---

### Task 8: The twenty-question smoke

- [ ] **Step 1: Run twenty**

Run:
```bash
bun run src/index.ts run -p memvara -b longmemeval -j gpt-4o -m gpt-4o -l 20 -r memvara-smoke-20
bun run src/index.ts status -r memvara-smoke-20
bun run src/index.ts show-failures -r memvara-smoke-20
```
Expected: 20 questions, success rate 100%, and `report.json` under `data/runs/memvara-smoke-20/` with accuracy overall and per type, search p50/p95, answer latency and context tokens. `show-failures` lists judged-incorrect answers; read three of them and note the failure shapes in `local/memorybench/STACK.md` (that list seeds step 2's work).

- [ ] **Step 2: Check the report's fields are the ones the docs table needs**

Run: `python3 -c "import json; r=json.load(open('data/runs/memvara-smoke-20/report.json')); print(sorted(r.keys()))"`
Expected: keys for accuracy, per-type breakdown, latency and tokens. Note their exact names; Task 9's extraction script uses them.

---

### Task 9: The baseline run, the record, the PR

**Files:**
- Create: `/Applications/workstation/agent-memory/local/memorybench/memvara-baseline-<CORE_SHA>/` (copied results)
- Modify: `docs/BENCHMARKS.md` (agent-memory, branch `claude/memorybench-baseline`)

- [ ] **Step 1: The full run**

Run:
```bash
cd /Applications/workstation/memorybench
bun run src/index.ts run -p memvara -b longmemeval -j gpt-4o -m gpt-4o -r memvara-baseline-<CORE_SHA>
bun run src/index.ts status -r memvara-baseline-<CORE_SHA>
```
Expected: 500 questions, success rate 100%. A stopped run resumes with the same `-r`. If any question failed, fix the cause and re-run the failed phase with `-f <phase>`; the reported number must come from a run with zero failures.

- [ ] **Step 2: Keep the results where a worktree removal cannot take them**

Run:
```bash
mkdir -p /Applications/workstation/agent-memory/local/memorybench
cp -R data/runs/memvara-baseline-<CORE_SHA> /Applications/workstation/agent-memory/local/memorybench/
ls /Applications/workstation/agent-memory/local/memorybench/memvara-baseline-<CORE_SHA>/
```

- [ ] **Step 3: Extract the table**

Run (adjust the key names to what Task 8 step 2 printed):
```bash
python3 - <<'EOF'
import json
r = json.load(open("/Applications/workstation/agent-memory/local/memorybench/memvara-baseline-<CORE_SHA>/report.json"))
print(json.dumps({k: r[k] for k in r if k not in ("questions", "results")}, indent=2)[:3000])
EOF
```
Expected: overall accuracy, per-type accuracy, search p50/p95, context tokens, success rate.

- [ ] **Step 4: Write the section**

In the agent-memory worktree on branch `claude/memorybench-baseline`, add to `docs/BENCHMARKS.md` immediately before the `### A before/after of one retrieval change` heading:

```markdown
### Judged accuracy in MemoryBench

The number a reader can put beside Supermemory's, Mem0's and Zep's: memvara as a
provider in [MemoryBench](https://github.com/supermemoryai/memorybench), Supermemory's
own harness, on LongMemEval-S (500 questions, each with its own haystack of about 40
sessions), with GPT-4o writing the answer from what memvara retrieved and GPT-4o judging
it against the gold answer. Errors are excluded from accuracy by the harness, so a run is
reported only when every question completed.

| provider | core | cloud | ingest model | reader | judge | accuracy | search p50 / p95 | context tokens |
|---|---|---|---|---|---|---|---|---|
| memvara, as shipped | `<CORE_SHA>` | `<CLOUD_SHA>` | none (fast path) | gpt-4o | gpt-4o | **<overall>%** | <p50> ms / <p95> ms | <tokens> |

| type | n | accuracy |
|---|---:|---:|
| single-session-user | | |
| single-session-assistant | | |
| single-session-preference | | |
| knowledge-update | | |
| temporal-reasoning | | |
| multi-session | | |
| abstention | | |

Run on <date> against memvara-cloud's compose stack on one machine, `MEMVARA_LLM=none`,
`all-MiniLM-L6-v2` embeddings, quota enforcement off; provider commit `<fork sha>` on
`memvara/memorybench`, upstream at `<upstream sha>`. The provider asks memvara for 30
results with turns included and no relevance floor, and renders them with both clocks;
it does not re-rank, deduplicate or truncate. The design is
`docs/superpowers/specs/2026-09-02-memorybench-baseline-design.md`.

**What the other systems publish is not this number.** Supermemory reports 95%
*Recall@15*, a retrieval metric, with a model rewriting memories at ingest, and shows it
beside Zep's 71.2% and full context's 60.2%, which are judged accuracy from Zep's paper.
Mem0 reports 94.4% judged accuracy. Their rows join the table above only when run by us
in this harness under this judge.
```

Fill every `<...>` from the report and `STACK.md`. Leave nothing in angle brackets.

- [ ] **Step 5: Gate, commit, push, PR, review**

Run:
```bash
cd /Applications/workstation/agent-memory/.claude/worktrees/friendly-einstein-53c8da   # or wherever the branch is checked out
COVERAGE_FILE=local/.cov-mb PYTHONPATH=. python3 -m pytest tests/test_docs.py -q -p no:cacheprovider --no-cov
git add docs/BENCHMARKS.md
git commit -m "Record memvara's judged LongMemEval-S accuracy in MemoryBench as the baseline"
git push -u origin claude/memorybench-baseline
gh pr create --title "Record memvara's judged LongMemEval-S baseline in MemoryBench" --body-file local/pr-memorybench-body.md
```
The PR body names: the number and per-type table, the fork branch and its sha, the stack shas, the models, that the run had zero failures, the three failure shapes noted in Task 8, and which model reviewed it (as "the session model", never named). Then `/code-review high <PR number>` and fix what it finds on the same branch before merge.

- [ ] **Step 6: Push the fork branch**

Run:
```bash
cd /Applications/workstation/memorybench
bun test && bunx prettier --check "src/**/*.ts"
git push -u origin memvara-provider
```
Opening the upstream pull request is a decision for the user once the number is one worth publishing; the branch is ready for it.

---

## Self-review

**Spec coverage.** Fork and file layout: Tasks 1, 5. Provider methods, one per spec paragraph: Task 4 (initialize with whoami and health; ingest with per-session write, session date on every turn, idempotency key, retry in Task 2's client; awaitIndexing as a stats read; search at 30 with episodes and no floor, results unchanged; clear as scope erasure without confirm_tenant). Prompt with two blocks, both clocks, question date, abstention: Task 3. Local stack with `MEMVARA_CORE_PATH` at a clean checkout, `MEMVARA_LLM=none`, MiniLM, ports: Task 6; the quota switch is an addition the spec did not foresee and is documented in the README paragraph. Runs and artifacts, results in the main checkout's `local/`, the docs table with every input, the sentence about the others' metrics: Tasks 7 to 9. Done criteria: one per type and abstention (Task 7 step 4), prompt read by a person (Task 7 step 3), clear proven (Task 7 step 5), twenty at 100% (Task 8), full at 100% with the table (Task 9), fork builds and formats (Task 9 step 6). Cost and time: no task, by design. User-provided items: Task 1 steps 1, 2, 5 and Task 7 step 1.

**Placeholders.** Angle-bracket values in Task 9 are run outputs to be filled, and the step says so. No "TBD", no "handle errors", no "similar to".

**Types.** `MemvaraContextItem` is defined in Task 3 and produced by Task 4's `toContextItem`, whose fields match the Task 3 test literal one for one. `MemvaraClient` method names are identical in Task 2's class, Task 4's provider and Task 4's fake. `ProviderName` gains `"memvara"` in Task 5, which Task 5's test asserts through `getAvailableProviders`.
