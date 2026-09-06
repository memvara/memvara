# Deploying memvara

Three ways to run it, in increasing order of ceremony: as a library inside your process,
as an MCP server your editor launches, and in a container. The library is the supported
integration point — the other two are adapters over it — so start there and move outward
only when something forces you to.

Then read [Operations](#operations). Everything in that section is something a first
deployment gets wrong, and most of it is silent when it goes wrong, which is the worst
combination a memory layer can have.

---

## 1. As a library

```bash
pip install memvara                # numpy, and nothing else
```

That is the supported install. `pip install -e .` from a clone is the development
install, and `python3 -m build --wheel` is how you get an artifact of the tree you
are standing on. See [`RELEASING.md`](RELEASING.md) for how a version gets onto
PyPI. There is an npm package of the same name; since `0.1.0` it is a CLI —
`npx memvara` bridges a stdio MCP client to the hosted server — and not a
library to import. It is a way *in* to memvara, not a second implementation of
it, and it is irrelevant to a self-hosted deployment.

<!-- This paragraph previously said the name "already belongs to an unrelated
differentiable-rendering library". That was true of `engram`, the project's former name,
and a blanket rename turned a true sentence into a false one that told every reader the
package name was somebody else's. Kept as a note because it is the second time a rename
has done exactly this here, and the lesson is that a search-and-replace cannot check
whether a claim is still about the thing it now names. -->


```python
from memvara import Memvara

mem = Memvara("/var/lib/myapp/memory.db", user="alice")
mem.add("I live in Berlin")
print(mem.recall("where do they live?"))
mem.close()                        # or use it as a context manager
```

`Memvara()` with no path is an in-memory store that dies with the process. That is right
for tests and wrong for everything else, and nothing will tell you which one you got.

**The parent directory has to exist.** SQLite does not create it, and the error you get
is `sqlite3.OperationalError: unable to open database file` — which reads like a
permissions problem and is usually a missing `mkdir -p`.

**One `Memvara` per process is enough.** It is synchronous and thread-safe for reads (a
per-thread read connection), so a web application should build one at startup and use
`mem.scope(user=...)` per request rather than opening the store per request. For asyncio,
wrap it once: `AsyncMemvara(Memvara(...))`.

**With no `llm=`, most of a conversation is not stored.** The default `NullLLM` runs the
deterministic fast path and nothing else, so `add()` keeps only the sentence forms the
rule extractor recognises and drops the rest. It says so once, loudly, as a
`DegradedExtractionWarning`, and `WriteReceipt.unextracted` counts the dropped turns on
every write. `remember()` is unaffected — a structured write never needed a model.

---

## 2. As an MCP server

```bash
MEMVARA_DB=~/.memvara/memory.db python3 -m memvara.server
```

JSON-RPC 2.0 over stdio, fourteen tools, no SDK dependency. It refuses to start without
`MEMVARA_DB` and prints the client configuration block instead — so if you have arrived
here because your client said the server failed, run the command by hand and read what it
says.

### Claude Desktop

`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS,
`%APPDATA%\Claude\claude_desktop_config.json` on Windows. This is the block the server
itself prints when `MEMVARA_DB` is unset, which makes it the one to trust if this document
ever drifts from the code:

```json
{
  "mcpServers": {
    "memvara": {
      "command": "python3",
      "args": ["-m", "memvara.server"],
      "env": {
        "MEMVARA_DB": "/absolute/path/to/memory.db",
        "MEMVARA_USER": "your-name"
      }
    }
  }
}
```

Two things about `command` that cost people an afternoon each. It is executed without a
shell and without your login profile, so `python3` resolves against a `PATH` that is not
your terminal's — if memvara lives in a virtualenv, give the absolute interpreter path
(`/path/to/venv/bin/python3`) rather than hoping. And `~` is expanded by the server for
`MEMVARA_DB` specifically, because that is what people type in a JSON file; it is *not*
expanded in `command`.

### Claude Code

```bash
claude mcp add memvara --env MEMVARA_DB=$HOME/.memvara/memory.db --env MEMVARA_USER=alice \
  -- python3 -m memvara.server
```

or the same `mcpServers` object in a project-local `.mcp.json`. Any MCP client works; the
transport is stdio and the configuration is entirely environment.

### The environment

| variable | meaning |
|---|---|
| `MEMVARA_DB` | **required.** Path to the SQLite file, created on first use. `:memory:` for a smoke test that forgets everything on exit. |
| `MEMVARA_USER` | who this server remembers for. Unset means the whole tenant. |
| `MEMVARA_TENANT` | isolation boundary above the user. Default `default`. |
| `MEMVARA_AGENT`, `MEMVARA_SESSION` | narrow further. Leave unset for durable facts — memory written at session scope is invisible to the next session. |
| `MEMVARA_LLM` | `none` (default, offline), `anthropic` (needs `ANTHROPIC_API_KEY` and `memvara[anthropic]`), or `openai` (needs `OPENAI_API_KEY` and `memvara[openai]`). |
| `MEMVARA_LLM_MODEL` | Model name for `MEMVARA_LLM=openai`. Unset uses the adapter's own default. Point `OPENAI_BASE_URL` at a self-hosted OpenAI-compatible server (vLLM, llama.cpp, Ollama's shim) and name its model here. See [Talking to a self-hosted model](#talking-to-a-self-hosted-model). |
| `MEMVARA_LLM_MAX_CLAIMS` | Cap on the claims array for `MEMVARA_LLM=openai`. Unset means uncapped, which is right for hosted OpenAI — it closes the array itself, and OpenAI documents `maxItems` as unsupported under strict mode. Set it for a self-hosted server that constrains decoding, where an uncapped array gives the grammar no way to end a response. A positive integer; anything else is refused at startup. See [Talking to a self-hosted model](#talking-to-a-self-hosted-model). |
| `MEMVARA_LLM_EXTRACT_SYSTEM` | Path to a file holding replacement extraction instructions for `MEMVARA_LLM=openai`. Unset uses the instructions memvara ships, which is right for every hosted model. Set it for a small self-hosted model that the shipped wording talks out of extracting at all. Read only by this backend, and checked only when it runs: a file that is missing, empty, over 64 KiB or not UTF-8 is refused at startup, but under any other `MEMVARA_LLM` the variable is never read at all. See [Talking to a self-hosted model](#talking-to-a-self-hosted-model). |
| `MEMVARA_LLM_TERSE_CLAIMS` | `1` asks `MEMVARA_LLM=openai` for a shorter claim shape: `polarity`, `confidence`, `when`, `amount` and `unit` become optional, so the model stops writing a field name and a null for each of them. Unset means the full shape, which is right for hosted OpenAI — its strict mode requires every declared property in `required`, so this is a 400 there. Set it for a self-hosted model whose generation speed is the bottleneck. **It changes ranking:** an omitted confidence puts every claim at 0.5. See [Talking to a self-hosted model](#talking-to-a-self-hosted-model). |
| `MEMVARA_EMBEDDER` | `hashing` (default, offline, 512-dimensional), `hashing:<dim>`, `local` or `local:<model>` (needs `memvara[local-embed]`), or `auto`. See [The embedder is named, not discovered](#the-embedder-is-named-not-discovered). |
| `MEMVARA_READ_ONLY` | `1` hides every tool that writes. |

**There is no variable here that configures a `read_selector`.** `memory_recall`'s
`ranked` argument (see `memvara.select`) is accepted by this server regardless, and every
call to it is served unranked — outcome `unconfigured`, the block ending with a
`RECALL_UNRANKED` line — because nothing in phase 1 wires a selector into
`memvara.server.config.build_memvara`. A caller that wants a ranked read runs the
library directly and passes `read_selector=` to `Memvara(...)`.

The scope is bound at startup and **cannot be changed by a tool call**. That is the
security property of the stdio transport: the process is the user, because the client
launched it with the user's environment, so there is no caller-supplied scope string for
a model to be talked into changing.

### Talking to a self-hosted model

`MEMVARA_LLM=openai` reaches any OpenAI-compatible endpoint, not just OpenAI's. The
endpoint is deliberately **not** a memvara setting: the adapter builds its client through
the official SDK, which reads `OPENAI_BASE_URL` and `OPENAI_API_KEY` from the environment
itself. So memvara's own variables here are the model name, the claim cap for a server
that constrains decoding, the extraction instructions themselves, and the claim shape.

```bash
OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
OPENAI_API_KEY=whatever \
MEMVARA_LLM=openai \
MEMVARA_LLM_MODEL=Qwen/Qwen3.5-4B-Instruct \
MEMVARA_LLM_MAX_CLAIMS=32 \
MEMVARA_LLM_EXTRACT_SYSTEM=$HOME/.memvara/extract.txt \
MEMVARA_LLM_TERSE_CLAIMS=1 \
MEMVARA_DB=$HOME/.memvara/memory.db python3 -m memvara.server
```

`OPENAI_API_KEY` has to be set even when the server ignores it, because the SDK refuses
to construct a client without one. A server started without it fails at startup with a
`ConfigError` saying so, rather than on the first turn that needed extraction.

**Set `MEMVARA_LLM_MAX_CLAIMS` if the server constrains decoding**, which vLLM and
llama.cpp both do — they compile the response schema to a grammar. An uncapped claims
array leaves "one more claim" permanently legal, so the grammar has no way to end a
response: a model that begins restating itself runs to its token limit still emitting
well-formed claim objects, and the reply arrives as truncated JSON that parses as nothing.
The real claims that came before the restatements are lost with it. Measured against
phi-4-mini through llama.cpp, one extraction in three failed this way, and a cap removed
all of them. 32 is a reasonable starting point: above any well-formed response measured
(19 or fewer) and below the observed runaway (past 35).

Leave it unset for hosted OpenAI. That model closes the array itself, so it needs no cap,
and OpenAI documents `maxItems` as unsupported under strict mode — where an unsupported
keyword is rejected rather than ignored, so a cap there buys nothing and risks the call.
An unusable value is refused at startup rather than clamped, `0` included.

**Set `MEMVARA_LLM_EXTRACT_SYSTEM` if a small model extracts nothing.** The instructions
memvara ships close by saying an empty list is a correct answer and the common case. That
is true, and a model that can weigh salience across a long turn needs to hear it, or it
invents a fact from every pleasantry. A small model reads the same sentence as permission:
measured 2026-09-03, phi-4-mini-instruct through llama.cpp returned an empty list for
every input past roughly 1,300 tokens, and removing that one sentence recovered extraction
on the same prompt and episodes. The variable takes a **path**, not the text — a
multi-paragraph prompt in an environment variable is unreadable in `docker inspect` and
unmaintainable in a compose file. Start from `EXTRACT_SYSTEM` in `memvara/llm/base.py` and
change as little as you can; the field descriptions in it are what make a claim land in
the right column. A file that cannot be read, or that is empty, is refused at startup
rather than falling back to the shipped prompt, because a deployment that named it meant
to change what the model is told and silently not changing it looks exactly like success.

It replaces the extraction instructions only. Predicate resolution — deciding whether a
new surface form is an existing predicate or a genuinely new one — keeps its own prompt,
which this accommodation was never measured against.

### Set `MEMVARA_LLM_TERSE_CLAIMS` if generation is the bottleneck

On a CPU-hosted model, generating the response is most of the wall time, and most of what
gets generated is field names rather than facts. The shipped schema requires all ten fields
on every claim, so the model writes `"when":null,"amount":null,"unit":null` and a
confidence number for each one whether or not the turn said anything about a time, a
measurement or how sure it was. One claim comes to 52 tokens, and 44 of them are keys,
punctuation and those nulls.

`MEMVARA_LLM_TERSE_CLAIMS=1` moves `polarity`, `confidence`, `when`, `amount` and `unit`
out of the schema's `required` list. The model may then leave them out, and memvara
supplies the same defaults it already applies to a value it cannot read: an assertion
unless `polarity` is exactly -1, and nothing for a time or a measurement the turn did not
state. Eight claims — the mean extraction measured on a 4-core box — go from 413 tokens to
229. `bench/extract_cost.py` measures this for your own schema and, given an endpoint, your
own model.

Two things to know before setting it. It is a 400 against hosted OpenAI, whose strict mode
requires every declared property to appear in `required` — the same trade `maxItems` makes.
And **it changes how claims rank**: an omitted confidence lands at 0.5 rather than at a
number the model chose, so a store written with this option ranks differently from one
written without it. On a small model that number is close to noise, since confidence is the
one field in a claim nothing downstream can check, but the change is real. Decide it once
per deployment rather than turning it on and off.

`MEMVARA_LLM_MODEL`, `MEMVARA_LLM_MAX_CLAIMS`, `MEMVARA_LLM_EXTRACT_SYSTEM` and
`MEMVARA_LLM_TERSE_CLAIMS` apply to the `openai` backend only. Under `MEMVARA_MODE=cloud`
all four are refused outright, along with `MEMVARA_LLM` and `MEMVARA_EMBEDDER`: extraction
runs inside the deployment, so a value named here would be read and never used.

### The embedder is named, not discovered

`MEMVARA_EMBEDDER` decides which vector space this server's store lives in. Left unset it
is `hashing` — 512 dimensions, offline, no download, no extra. That is a choice about your
store rather than a fallback, and the reason it is named rather than detected is worth one
paragraph, because the detected version was a bug:

> `memvara[rerank]` installs `sentence-transformers`, because a cross-encoder is one. The
> server used to take whatever `default_embedder()` returned, and that function returns a
> local 384-dimensional model as soon as `sentence-transformers` is *importable*. So
> `pip install memvara[rerank]` into a working deployment changed the embedder of a store
> nobody had touched, and the next launch refused to open it.

The refusal is correct — the alternative is a store that keeps growing and cannot be
searched — but a server whose vector space is decided by the last `pip install` in the
image is not configured, it is guessed. Hence a named default.

| value | what it opens |
|---|---|
| `hashing` | `HashingEmbedder(dim=512)`. The default, and identical to what a deployment with no extras installed has always had. |
| `hashing:<dim>` | The same at another width. `hashing:384`, `hashing:768` — this is how you name a store that was written at a width other than the default. |
| `local` | `LocalEmbedder()`, i.e. `all-MiniLM-L6-v2` at 384 dimensions. Needs `memvara[local-embed]`; **fails at startup if it is missing** rather than quietly falling back. |
| `local:<model>` | Any sentence-transformers model id, spelled exactly as `memory.db.embedder.json` records it — `local:BAAI/bge-small-en-v1.5`. Case-sensitive. |
| `auto` | Whichever of the above happens to be installed. The old behaviour, available on request; it is the right answer only if you do not mind which vector space you get. |

An unrecognised value is a startup error listing the ones that work, in the same place
and the same shape as an unrecognised `MEMVARA_LLM`.

#### If the server already refuses to start

The symptom is a launch failure whose message contains `this store holds N-dimensional
vectors`. Run the command by hand — the client will not show you the whole thing — and
read the two numbers in it:

```
$ MEMVARA_DB=~/.memvara/memory.db python3 -m memvara.server
memvara-mcp: /home/you/.memvara/memory.db: this store holds 384-dimensional vectors,
written by local:sentence-transformers/all-MiniLM-L6-v2, but the configured embedder is
hashing:512:3-5 (dim 512). ...
From this server, set MEMVARA_EMBEDDER to match the store — 'hashing:<dim>' or
'local:<model>', spelled as above — rather than editing code.
```

**Nothing is damaged.** This is a refusal taken before the process writes anything, which
is the whole reason the check happens at construction. The fix is one variable, and which
one is written in the message: the phrase after `written by` is the value to use.

- `written by hashing:512:3-5` → `MEMVARA_EMBEDDER=hashing:512`
- `written by local:sentence-transformers/all-MiniLM-L6-v2` →
  `MEMVARA_EMBEDDER=local:sentence-transformers/all-MiniLM-L6-v2` (and keep
  `memvara[local-embed]` installed)

If the `written by` clause is absent, the store predates the fingerprint sidecar or was
copied without it. The width is still there and is still enough: `MEMVARA_EMBEDDER=hashing:<N>`
if you never installed an embedding extra, `local` if you did.

Set the variable in the same `env` block as `MEMVARA_DB` — and note that under Docker it
has to cross into the container like every other one, as `-e MEMVARA_EMBEDDER=...`.

Migrating instead of matching is the other option, and it is a deliberate one rather than
a fallback: it re-encodes the whole store. It is not reachable from the server, on
purpose. See [Changing the embedder](#changing-the-embedder).

`consolidate`, `purge`, `reset` and `erase` are deliberately not tools — see
[Consolidation](#consolidation-is-a-job-you-have-to-schedule) for the half of that you
still have to run.

---

## 3. In Docker

```bash
docker build -t memvara-mcp:0.1.0 .
docker volume create memvara-data
```

```bash
docker run --rm -i \
  -v memvara-data:/data \
  -e MEMVARA_DB=/data/memory.db \
  -e MEMVARA_USER=alice \
  memvara-mcp:0.1.0
```

**`-i`, never `-it`.** The container's stdin and stdout *are* the MCP transport. A TTY
adds line discipline — input echo, and `\n` → `\r\n` on the way out — both of which
corrupt a newline-framed JSON stream. Docker refuses the combination outright when stdin
is a pipe (`cannot attach stdin to a TTY-enabled container because stdin is not a
terminal`), which is the friendlier of the two ways to find out.

There is no port, no `EXPOSE` and no `HEALTHCHECK`, and the reasoning for each is in the
`Dockerfile`. The short version: a healthcheck runs a *second* process, and a second
process cannot say anything about the one holding stdio. The pipe is the liveness signal.

### From an MCP client

```json
{
  "mcpServers": {
    "memvara": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "-v", "memvara-data:/data",
        "-e", "MEMVARA_DB=/data/memory.db",
        "-e", "MEMVARA_USER=alice",
        "memvara-mcp:0.1.0"
      ]
    }
  }
}
```

Note where the configuration went. An MCP client's `"env"` block sets variables for the
process it launches — which here is the `docker` CLI, not the server. Variables have to
cross into the container explicitly, either as `-e NAME=value` in `args` (above), or as a
bare `-e NAME` which forwards the value from the client's own environment:

```json
"args": ["run", "--rm", "-i", "-v", "memvara-data:/data", "-e", "MEMVARA_DB", "-e", "MEMVARA_USER", "memvara-mcp:0.1.0"],
"env": {"MEMVARA_DB": "/data/memory.db", "MEMVARA_USER": "alice"}
```

Both work. The first is easier to read six months later.

### Hardening

Verified working with a real write:

```bash
docker run --rm -i --read-only --cap-drop=ALL --security-opt no-new-privileges \
  -v memvara-data:/data -e MEMVARA_DB=/data/memory.db memvara-mcp:0.1.0
```

The image runs as uid 10001 and has no `pip` on `PATH` — both the venv's and the base
image's are removed at build time. Treat that as a speed bump rather than a boundary: it
stops the image quietly acquiring a dependency, and it does not stop anyone with a shell
in there from fetching a wheel by hand. A read-only root filesystem is compatible because
the only thing that writes is the store, and the store is on the volume.

### Image size

`python:3.13-slim`, multi-stage, linux/arm64: **292 MB unpacked, 63.2 MB to pull.**
linux/amd64 pulls 64.0 MB. That splits as ~150 MiB of official Python base image, 62 MiB
of numpy (35 of numpy plus 27 of the OpenBLAS it bundles), and **1.4 MiB of memvara**.
There is no dependency tree to trim; the base image is the image. Dropping pip out of the
venv before it is copied into the runtime stage is worth 17 MB unpacked and 3.7 MB
compressed, and is the only trim here that measurably moved the number.

An Alpine base is the one lever that moves it — 169 MB unpacked / 38.6 MB pulled, a 39%
saving, by changing `slim` to `alpine` in both `FROM` lines and `useradd --create-home
--uid 10001 memvara` to `adduser -D -u 10001 memvara`. It is not the default for two
measured reasons. Musl was slower on a local write/search workload inside the image (400
`remember()` 140 → 165 ms, 200 `search()` 618 → 659 ms, best of three on one loaded
machine — treat as "a few percent", not as a benchmark). And musl closes doors: there are
no musllinux wheels for torch, so `memvara[local-embed]` cannot be installed on top of an
Alpine image at all, while `memvara[anthropic]` and `memvara[openai]` can.

---

## Operations

### Where the database goes

The store is **more than one file**, and this is the single most common deployment
mistake. After a write to `memory.db` the directory holds:

| file | what it is | losing it costs |
|---|---|---|
| `memory.db` | claims, episodes, predicates, the FTS index | everything |
| `memory.db.vecs` | the mmapped vector matrix | every embedding — search degrades to BM25 only, silently |
| `memory.db.embedder.json` | which embedder wrote those vectors | the ability to detect a model swap, which then goes undetected |
| `memory.db-wal`, `memory.db-shm` | SQLite write-ahead log, while open | recently committed writes |

So **mount, back up and copy the directory, not the file.** A Docker bind mount of
`memory.db` alone persists the rows and throws away the vectors on every restart, and
nothing raises: the store re-opens, BM25 keeps working, and semantic recall quietly stops.

Two further notes on volumes:

- A **named volume** (`-v memvara-data:/data`) is the recommended shape. It is owned by uid
  10001 because the image pre-creates `/data` with that owner, and Docker seeds a new
  named volume from the image.
- A **bind mount** (`-v /host/path:/data`) inherits the host directory's ownership
  instead, so the container's user usually cannot write it. Run with
  `--user "$(id -u):$(id -g)"` or `chown` the host directory. On Docker Desktop for macOS
  and Windows, be aware that SQLite's file locking crosses a virtualised filesystem here;
  a named volume avoids the question entirely.

**Backups.** `cp memory.db` while a process has it open can capture a torn database,
because the committed tail is in the `-wal`. Either stop the writer, or use
`sqlite3 memory.db ".backup out.db"` — and copy `memory.db.vecs` and
`memory.db.embedder.json` alongside it either way. Copying the database without the
vectors produces a restore that looks healthy and has lost its vector index.

### Consolidation is a job you have to schedule

```python
mem.consolidate()   # {'decayed': 128, 'merged': 4, 'promoted': 2}
```

It decays salience toward a floor, merges near-duplicate claims into one deterministic
survivor, and promotes repeatedly observed episodic claims to semantic ones. It is
idempotent, and it runs windowed — committing every 500 rows rather than holding one
transaction across the sweep — so it does not lock out the writes happening beside it.

**It does not run itself, on purpose.** Three reasons, in the order they bite:

1. It is a full sweep, so it is linear in store size — roughly 460 ms per 8,000 claims on
   one developer machine. A library that fired that off on a timer inside your process
   would be spending your CPU on a schedule you did not choose.
2. It is a *scope-wide* operation. Deciding when a tenant's memory gets rewritten is an
   operator's call, not a call site's.
3. Nothing about it is required for correctness. Skipping it costs ranking quality, not
   answers.

So run it from cron, a Celery beat, a systemd timer — anything you control. Nightly is a
reasonable default for a store that sees daily use; hourly if writes are heavy. It is
deliberately **not** an MCP tool, because a model handed a tool will call it in a loop.

```python
# consolidate.py, run nightly
from memvara import Memvara

with Memvara("/var/lib/myapp/memory.db") as mem:
    print(mem.consolidate())
```

Watch `consolidate.merged` in telemetry if you have a recorder wired up: it is emitted
**at zero**, so "nothing to merge" stays distinguishable from "the scheduler stopped
running", which is the failure this whole paragraph exists to make visible.

### Changing the embedder

Vectors written by one embedder are meaningless to another, and there are two failure
shapes depending on whether the widths happen to match.

**Different width.** `Memvara()` raises `EmbedderMismatchError` at construction, before
anything writes. This is the case that used to be a disaster: installing
`memvara[local-embed]` changes what `default_embedder()` returns from a 512-dimensional
hashing embedder to a 384-dimensional model, so following the README's own upgrade advice
made every read raise while every write kept succeeding — a store that grows and cannot
be searched.

Both shapes reach the MCP server too, where the answer is `MEMVARA_EMBEDDER` rather than a
keyword argument — see [If the server already refuses to
start](#if-the-server-already-refuses-to-start). Migration is deliberately *not* reachable
from there: re-encoding a whole store is an operator action, and a server that did it on
startup because its environment changed would be the same guess this section is about, one
level up.

**Same width, different model.** Nothing can raise, because nothing is wrong
dimensionally and every similarity is nonsense. You get an `EmbedderChangedWarning`,
which is only possible because `memory.db.embedder.json` records the name.

**Right model, text it cannot read.** The third shape, and the quietest: the embedder is
the one you chose and it returns an all-zero vector for some of your text. With the
default `HashingEmbedder` that is anything with no `[a-z0-9']` in it — Han, Kana, Hangul,
Arabic, Hebrew. Nothing is misconfigured, so nothing raises and no migration helps; the
claims are stored, answer by predicate, and are never returned by meaning. Watch
`write.embedding_unusable`, which is tagged by script and is the only number that says how
much of the store is affected — the accompanying `UnembeddableTextWarning` fires once per
pipeline, so on a server building one `Memvara` per request it is one line per request and
on a long-lived one it is a single line from whenever this started. A non-zero counter
against a script you serve means installing an embedder that covers it, not a re-encode of
what you have. Note also that a *mixed* line embeds fine from its Latin half alone and is
never counted, so this is a floor on the problem rather than a measure of it.

Either way the fix is one migration, which re-encodes every claim *and* every episode and
rewrites the fingerprint:

```python
mem = Memvara("memory.db", embedder=NewEmbedder(), reembed=True)   # at open
n = mem.reembed(NewEmbedder())                                    # or later; returns claims re-encoded
```

Cost is one encode per claim, plus one per episode, and zero model calls unless your
embedder makes them. Budget for it: against a hosted embedder this is a network round trip
per batch across the whole store. It is not scoped and cannot be — the vector matrix is
one index shared by every tenant, so a partial migration leaves exactly the mixed-width
store the error exists to prevent.

### Scope, and what a shared store isolates

`tenant > user > agent > session`, with inheritance downward and no leakage sideways. A
session-scoped query also sees that user's durable memory, and never a sibling session's,
another agent's, or another tenant's. Scope filters fail **closed** — a scope that
resolves to nothing matches nothing, rather than degrading into an unfiltered query across
every user.

For a server process, build one `Memvara` and take `mem.scope(user=...)` per request. A
`ScopedMemvara` is a binding, not a second store, so making one per request is free.

### Deletion, when someone asks for it

`forget()` and `delete()` **retire**: the claim stops answering present-tense queries and
`history()` still sees it. That is the right default for correcting a belief and the wrong
answer to a GDPR Article 17 request, because the text is still on disk.

`erase(claim_id, sources=True)` and `purge()` **erase**, irreversibly, including the FTS
entry (which stores the tokens directly) and the embedding (which leaks content under
inversion). Both return per-table counts as evidence. Neither is reachable from the MCP
server, deliberately.

Note that erasure removes rows; it does not shrink the file. Run `VACUUM` if the on-disk
**footprint** of deleted data matters to you — its **readability** is handled by the store
itself, which sets `PRAGMA secure_delete=ON` and FTS5's `secure-delete` so the bytes are
overwritten rather than merely freed.

That was not always true. Before schema 7 this paragraph offered `VACUUM` as the lever for
readability too, and for the text index it did not work: a deleted FTS5 row leaves its
terms as live rows in a shadow table, which a `VACUUM` does not touch. Opening an older
store with this version scrubs it once, on the spot.

---

Previous: [MCP](integrations/mcp.md) · Next: [Open core](OPEN-CORE.md) · [Documentation index](README.md)
