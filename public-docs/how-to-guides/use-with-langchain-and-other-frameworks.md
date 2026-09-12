# How to use Memvara with LangChain and other frameworks

Memvara ships ready-made adapters for LangChain, LlamaIndex, LangGraph, and CrewAI. They
don't all preserve the same amount of what makes Memvara different from a plain vector
store — some interfaces simply don't give Memvara enough information to keep everything.
This guide shows you how to use each adapter, and is upfront about what each one keeps
and what it loses, so you can choose the right one for what you need.

## The short version

| Framework | What you get | Install |
|---|---|---|
| LangChain retriever | Everything, including time travel | `pip install "memvara[langchain]"` |
| LlamaIndex retriever / memory block | Everything, including time travel | `pip install "memvara[llama-index]"` |
| LangGraph store | Contradiction handling, per-field facts — loses your custom vocabulary rules | `pip install "memvara[langgraph]"` |
| LangChain chat history | Just the write path — loses supersession and history | `pip install "memvara[langchain]"` |
| CrewAI storage | Basic storage only — loses contradiction handling entirely | `pip install "memvara[crewai]"` |

**The rule that decides it:** any interface that hands Memvara the actual search text
keeps everything Memvara can do, because Memvara can run its normal search on that text.
An interface that only hands over a pre-computed vector, or a plain list of chat
messages, can't — there's no query text left for Memvara's search to work with.

## LangChain retriever (recommended for most uses)

```python
from memvara import Memvara
from memvara.integrations.langchain import MemvaraRetriever

mem = Memvara("memory.db", user="alice")
retriever = MemvaraRetriever(mem)
```

This is a normal `BaseRetriever` — it fits directly into any LangChain chain that expects
one — and it keeps everything: hybrid search, per-fact confidence and recency ranking,
and even time-travel queries via `as_of=`.

## LlamaIndex retriever and memory block

```python
from memvara.integrations.llamaindex import MemvaraRetriever, MemvaraMemoryBlock
```

`MemvaraRetriever` works the same way as the LangChain version. `MemvaraMemoryBlock` is
worth calling out specifically: it's the only adapter where Memvara's *write* path — hash
deduplication, importance scoring, fact extraction — also runs through the framework,
because a memory block's job (absorb recent messages, decide what belongs in the prompt)
maps directly onto Memvara's own `add()` and `recall()`.

## LangGraph store

```python
from memvara import Memvara
from memvara.integrations.langgraph import MemvaraStore

store = MemvaraStore(Memvara("memory.db"), user="alice")
```

`memvara[langgraph]` installs `langgraph-checkpoint`, not `langgraph` itself — that's
where the interface this adapter implements actually lives, and an app already using
LangGraph will usually already satisfy this requirement without a second install.

LangGraph's storage interface passes Memvara a namespace, a key, and a value on every
write — which happens to line up with a subject, a predicate, and an object. So each
field of a stored item becomes its own fact, and changing one field only affects that
field. What's lost is your custom vocabulary: a value stored under the key `home_city`
here won't be recognized as the same kind of fact as one extracted as `lives_in`
elsewhere, because the field name arrives as a bare string with nothing declaring what it
means.

By default, deleting an item through this interface is a **correction** (reversible, the
old value stays visible in history), not a permanent deletion. Pass `on_delete="erase"`
if you specifically want deletions through this interface to actually erase the data.

## LangChain chat message history

```python
from memvara.integrations.langchain import MemvaraChatMessageHistory

history = MemvaraChatMessageHistory(mem, on_clear="ignore")
```

LangChain's chat-history interface models memory as a plain list of messages, which has
no room for anything Memvara actually tracks — no supersession, no valid-time interval,
no confidence, no link back to source. So this adapter is honest about what it is: a
convenient way to store and replay a conversation, not a way to preserve Memvara's memory
model. `messages` returns the stored conversation, not the facts extracted from it.

`clear()` deserves a specific note. LangChain's documented meaning for `clear()` is
"permanently remove everything." Memvara has two very different things this could mean —
a reversible correction, or an irreversible full erase — and picking the wrong one
automatically, especially at the end of a session, is the worst possible moment to guess.
So by default, `clear()` raises an error rather than silently picking one. Set
`on_clear="ignore"` if ending a session shouldn't touch memory at all (the usual choice),
or `on_clear="purge"` if you specifically want LangChain's literal meaning.

## CrewAI storage backend

```python
from crewai.memory import Memory
from memvara.integrations.crewai import MemvaraStorage

storage = MemvaraStorage(mem, user="alice")
memory = Memory(storage=storage, embedder=storage.embedder)
```

**You must pass `embedder=storage.embedder`.** CrewAI computes its own embedding vector
before ever handing anything to the storage backend, so Memvara never sees the actual
search text — only a raw vector. Without a subject, predicate, or object to work with,
there's nothing for Memvara's contradiction handling to check, so "Alice lives in Berlin"
and "Alice moved to Lisbon" simply stay side by side rather than one replacing the other.
This adapter is the one that loses the most, and it's an honest limitation of CrewAI's
interface rather than something Memvara chooses to give up.

This adapter needs `crewai>=1.10.1` — earlier versions don't have the storage interface
this plugs into at all.

## Next steps

- [Use Memvara in an AI coding assistant](use-memvara-in-an-ai-coding-assistant.md) — for
  editor-based assistants rather than a framework you're coding against.
- [Python API reference](../reference/python-api.md) — the full underlying API each
  adapter is built on.
