# Memvara documentation

**Bitemporal memory for AI agents.** Memvara stores facts as structured claims — not
just text — so it can tell you what's true right now, what was true at any point in the
past, and why it believes what it believes. This documentation is organized around what
you're trying to do:

- **New to Memvara and want to learn by doing it?** Start with the
  [tutorial](tutorials/getting-started.md).
- **Have a specific task in mind?** Jump to [How-to guides](#how-to-guides).
- **Looking for exact method signatures, tool names, or settings?** Go to
  [Reference](#reference).
- **Want to understand why Memvara is built the way it is?** Read
  [Explanation](#explanation).

```bash
pip install memvara
```

**numpy and nothing else.** Runs fully offline — no API key, no database server, no
network connection required.

## Tutorial

New here? This is the place to start — a single guided walkthrough, no forks or
decisions required.

| | |
|---|---|
| [Getting started](tutorials/getting-started.md) | Install Memvara, record a fact, watch it change over time, and ask what was true at three different moments — in about ten minutes. |

## How-to guides

Task-oriented guides. Each one assumes you already know the basics and just want to get
something specific done.

| | |
|---|---|
| [Install Memvara](how-to-guides/install-memvara.md) | The core install, the optional extras, and what you get with no configuration at all. |
| [Record and correct a fact](how-to-guides/record-and-correct-a-fact.md) | Write a fact properly — with its source, its dates, and its confidence — and the three different ways to fix a mistake. |
| [Ask about the past](how-to-guides/ask-about-the-past.md) | The three ways to ask "what was true," and the one case where they give different answers. |
| [Search your memory](how-to-guides/search-your-memory.md) | Get search results to reason about, or a ready-to-use block of text for a prompt. |
| [Use Memvara in an AI coding assistant](how-to-guides/use-memvara-in-an-ai-coding-assistant.md) | Connect Claude, Cursor, ChatGPT, or any MCP-capable assistant to your memory. |
| [Run your own MCP server](how-to-guides/run-your-own-mcp-server.md) | Self-host the server for a team, or for a client that isn't Python. |
| [Define your own fact types](how-to-guides/define-your-own-fact-types.md) | Teach Memvara vocabulary for your own domain, so contradictions resolve correctly. |
| [Delete personal data](how-to-guides/delete-personal-data.md) | The difference between correcting a fact and permanently erasing it, and how to prove a deletion worked. |
| [Use with LangChain and other frameworks](how-to-guides/use-with-langchain-and-other-frameworks.md) | The LangChain, LlamaIndex, LangGraph, and CrewAI adapters — what each one keeps and what it loses. |

## Reference

Lookup material: exact signatures, exact tool names, exact settings.

| | |
|---|---|
| [Python API](reference/python-api.md) | Every method on `Memvara`, and everything importable from the package. |
| [MCP tools](reference/mcp-tools.md) | The fourteen tools an AI assistant gets when connected over MCP. |
| [CLI and configuration](reference/cli-and-configuration.md) | `memvara-mcp`'s commands and every environment variable it reads. |
| [Predicate schema](reference/predicate-schema.md) | Declaring your own kinds of facts: cardinality, volatility, and the shipped vocabulary packs. |

## Explanation

Background and reasoning, for understanding *why* Memvara works the way it does.

| | |
|---|---|
| [What problem does Memvara solve?](explanation/what-problem-memvara-solves.md) | The five questions a real memory system has to answer, and why plain retrieval only answers one of them. |
| [Bitemporal memory, explained](explanation/bitemporal-memory-explained.md) | Why every fact needs two independent dates, not one. |
| [How contradictions are resolved](explanation/how-contradictions-are-resolved.md) | Why conflicting facts are resolved by a lookup, not by asking a model. |
| [Provenance and trust](explanation/provenance-and-trust.md) | How a fact traces back to its source, and the exact vocabulary for how a fact stops being current. |
| [How search works](explanation/how-search-works.md) | Why search combines keyword and meaning-based matching, and how relevance decays over time. |
| [Memory vs. retrieval-augmented generation](explanation/memory-vs-retrieval-augmented-generation.md) | Why RAG and memory answer different questions, and how to use both together. |
| [Known limitations](explanation/known-limitations.md) | Every real limit, stated plainly, with what removes each one. |

---

This documentation set lives in `public-docs/` and is independent of the more detailed,
implementation-level docs in [`docs/`](../docs/README.md), which cover internals,
deployment operations, and benchmarking in more depth.
