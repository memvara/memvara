# Installation

```bash
pip install memvara
```

That is the whole install for the library. It pulls **numpy and nothing else** — no
vector database, no Docker, no API key, no network at import time. Python 3.10 or later.

```python
>>> import memvara
>>> memvara.__version__
'0.9.0'
```

## Check it works

```bash
python3 -c "
from memvara import Memvara, NullLLM
mem = Memvara(user='you', llm=NullLLM())
mem.remember('you', 'lives_in', 'Berlin')
print([r.text for r in mem.search('where do they live?')])
"
# ['you lives in Berlin']
```

`Memvara()` with no path uses an in-memory SQLite store, so that command leaves nothing
behind. Pass a filename to keep it: `Memvara("memory.db")`.

## What you get with no model

The default configuration is `HashingEmbedder` + `NullLLM` + `SQLiteStore`, and two of
those three defaults are worth understanding before you build on them, because both are
honest about being defaults and neither is silent about it.

**`remember()` needs no model, ever.** You hand it a subject, a predicate and an object;
there is nothing to parse. Contradiction resolution, retrieval, the two clocks,
provenance and consolidation never needed a model either. This is the path a real
integration writes on, and it is what the offline claim means.

**`add()` on arbitrary prose mostly does.** With no `llm=`, `add()` runs a deterministic
extractor that recognises a fixed set of high-precision sentence forms on user turns —
*"I live in X"*, *"I work at X"*, *"my name is X"*. Anything else — an employer mentioned
in passing, a version number, a preference stated as an aside — reaches the extraction
tier, finds no model, and is dropped. The store says so rather than pretending: the
constructor emits a `DegradedExtractionWarning` once, and every `WriteReceipt` counts the
dropped turns in `.unextracted`.

To silence the warning and keep the offline configuration, ask for it explicitly:

```python
from memvara import Memvara, NullLLM
mem = Memvara("memory.db", llm=NullLLM())
```

**`HashingEmbedder` is a lexical fallback, not a semantic model.** It is the default so
the library runs offline in milliseconds with no download and so tests are deterministic.
It will not put *physician* near *doctor*, and it tokenises `[a-z0-9']+` — so text in
Han, Kana, Hangul, Arabic or Hebrew produces an all-zero vector and is never returned by
meaning. Such a write warns (`UnembeddableTextWarning`) rather than failing quietly. If
you need real semantic recall, install `memvara[local-embed]` and pass an embedder.

## Optional extras

Every one of these is genuinely optional. Naming a backend does not import its SDK, so a
bare install stays a two-package install.

| Extra | Install | For |
|---|---|---|
| `anthropic` | `pip install 'memvara[anthropic]'` | `AnthropicLLM` — extraction from arbitrary prose |
| `openai` | `pip install 'memvara[openai]'` | `OpenAILLM` — the same, on OpenAI |
| `local-embed` | `pip install 'memvara[local-embed]'` | A real sentence-transformers embedder, run locally |
| `rerank` | `pip install 'memvara[rerank]'` | The cross-encoder reranker |
| `cloud` | `pip install 'memvara[cloud]'` | `Memvara(api_key=…)` against a hosted deployment, and `memvara-mcp login` |
| `langchain` | `pip install 'memvara[langchain]'` | The LangChain retriever and chat-message-history adapters |
| `llama-index` | `pip install 'memvara[llama-index]'` | The LlamaIndex retriever |
| `langgraph` | `pip install 'memvara[langgraph]'` | The LangGraph `BaseStore` adapter |
| `crewai` | `pip install 'memvara[crewai]'` | The CrewAI storage adapter |
| `dev` | `pip install -e '.[dev,cloud]'` | pytest, coverage, mypy — what CI installs |
| `bench` | `pip install 'memvara[bench]'` | Only for `bench/mem0_real.py`, the head-to-head against the real `mem0ai` |

`memvara[http]` is **reserved, not wired**: it names FastAPI, uvicorn and pydantic for a
REST layer that is not in this repository. Nothing here imports them — see
[Open core](../OPEN-CORE.md) for where that line is and why it does not move.

## The MCP server

`pip install memvara` also installs the `memvara-mcp` console script, which speaks
JSON-RPC 2.0 over stdio and needs nothing beyond the core:

```bash
MEMVARA_DB=~/memory.db memvara-mcp
memvara-mcp init --agent claude      # writes the client config block and the skill tree
```

See [MCP](../integrations/mcp.md) for which of the three MCP surfaces to pick, and
[Deploying](../DEPLOY.md) for running it for other people.

## Two version floors worth knowing

- **Declared predicate vocabularies need Python 3.11.** They are TOML, read with
  `tomllib`, which arrived in 3.11. Everything else in the library works on 3.10; a pack
  on 3.10 raises with the reason rather than half-loading.
- **`crewai>=1.10.1`, and the floor is load-bearing.** Releases 1.0.0 through 1.9.3 ship
  a memory system with no `StorageBackend` protocol at all, so there is nothing for the
  adapter to bind to.

---

Previous: [Documentation index](../README.md) · Next: [Quickstart](quickstart.md)
