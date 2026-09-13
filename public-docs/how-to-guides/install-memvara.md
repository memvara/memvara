# How to install Memvara

## Install the library

```bash
pip install memvara
```

This installs Memvara and its one required dependency, `numpy`. Nothing else — no
database server to run, no API key to obtain, and no network access needed to import the
package. Memvara requires Python 3.10 or later.

Check the install:

```bash
python3 -c "import memvara; print(memvara.__version__)"
```

## Install the extras you actually need

Everything beyond the core library is opt-in, and naming an extra you don't use costs you
nothing — Memvara never imports a package it wasn't told to use.

| If you want... | Install | Extra name |
|---|---|---|
| To extract facts from free-form text using Claude | `pip install "memvara[anthropic]"` | `anthropic` |
| To extract facts from free-form text using GPT | `pip install "memvara[openai]"` | `openai` |
| Real semantic search (not just keyword matching) | `pip install "memvara[local-embed]"` | `local-embed` |
| A more accurate re-ranking step for search results | `pip install "memvara[rerank]"` | `rerank` |
| To connect to a hosted Memvara deployment | `pip install "memvara[cloud]"` | `cloud` |
| To use Memvara as a LangChain retriever or chat history | `pip install "memvara[langchain]"` | `langchain` |
| To use Memvara with LlamaIndex | `pip install "memvara[llama-index]"` | `llama-index` |
| To use Memvara as a LangGraph store | `pip install "memvara[langgraph]"` | `langgraph` |
| To use Memvara as CrewAI's storage backend | `pip install "memvara[crewai]"` | `crewai` |

You can combine extras: `pip install "memvara[anthropic,local-embed]"`.

## Understand what you get by default, with no extras

Two of Memvara's defaults are worth knowing about before you build anything real on top
of them, because both are honest compromises rather than hidden limitations.

**Writing exact facts never needs a model, with any configuration.** If you already know
the subject, the kind of fact, and the value — "Alice", "lives_in", "Berlin" — Memvara
records it directly with `remember()`. There is nothing here for a model to interpret.

**Turning free-form text into facts needs a model, unless the sentence is very simple.**
With no model configured, Memvara recognizes a small set of very common sentence patterns
— "I live in X", "I work at X", "my name is X" — and extracts facts from those alone.
Anything more complex — "I've been working remotely since my company moved offices" — is
not understood without a model, and Memvara tells you so rather than pretending: it warns
you once, and every write operation reports how many sentences it had to skip.

If you want the fully offline setup and don't want the warning, ask for it explicitly:

```python
from memvara import Memvara, NullLLM
mem = Memvara("memory.db", llm=NullLLM())
```

**The default search is keyword-based, not meaning-based.** Out of the box, Memvara
matches words, not concepts — it will not connect "doctor" with "physician". This keeps
the library fast and fully offline. If you need search that understands meaning, install
`memvara[local-embed]` and pass a real embedding model when you create your store.

## Install the MCP server (for AI coding assistants)

`pip install memvara` also installs a command called `memvara-mcp`, which lets any AI
assistant that speaks the Model Context Protocol (MCP) — such as Claude, Cursor, or
ChatGPT — read and write your memory directly.

```bash
MEMVARA_DB=~/.memvara/memory.db memvara-mcp
```

See [Use Memvara in an AI coding assistant](use-memvara-in-an-ai-coding-assistant.md) for
the full setup, including a version that needs no Python installed at all.

## Two things worth knowing before you go further

- **Declaring your own kinds of facts (beyond the built-in "where someone lives, works,
  and so on") needs Python 3.11.** Custom vocabularies are written in TOML files, and
  Python's built-in TOML reader only exists from 3.11 onward. Everything else in Memvara
  works fine on 3.10. See
  [Define your own fact types](define-your-own-fact-types.md).
- **If you plan to use the CrewAI adapter, you need `crewai>=1.10.1`.** Earlier versions
  of CrewAI don't have the storage interface Memvara's adapter plugs into.

## Next steps

- [Record and correct a fact](record-and-correct-a-fact.md) — write your first facts
  properly.
- [Use Memvara in an AI coding assistant](use-memvara-in-an-ai-coding-assistant.md) — get
  memory working inside your editor.
