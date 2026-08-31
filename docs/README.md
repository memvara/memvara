# Memvara documentation

Start at the top and follow the **Next** link at the bottom of each page. Every page
here has one, so you should never finish one and have to guess where to go.

**New here?** [Why Memvara?](concepts/why-memvara.md) is the five-minute version of what
this is for. [Quickstart](getting-started/quickstart.md) is the five-minute version of
using it. They are independent — read either first. Ninety seconds of it running is the
[demo](../README.md#the-90-second-demo).

## Getting started

| | |
|---|---|
| [Installation](getting-started/installation.md) | `pip install memvara`, the optional extras, and what the offline configuration will and will not store |
| [Quickstart](getting-started/quickstart.md) | A store, three facts, and three questions with three different correct answers |
| [Your first memory](getting-started/first-memory.md) | Writing one fact properly: the triple, the two dates, the source, and the three ways to correct it |

## Concepts

| | |
|---|---|
| [Why Memvara?](concepts/why-memvara.md) | The five questions a memory layer has to answer that retrieval does not |
| [Bitemporal memory](concepts/bitemporal-memory.md) | Two clocks, what each one answers, and the question a single date cannot ask |
| [Contradiction resolution](concepts/contradiction-resolution.md) | Why a conflict is an indexed lookup here rather than a model call |
| [Provenance](concepts/provenance.md) | `why()`, `produced()`, and what "the record was wrong" means as distinct from "the world changed" |
| [Temporal retrieval](concepts/temporal-retrieval.md) | Search at a past instant, recency decay per predicate, and why every score is inspectable |
| [RAG and memory](concepts/rag-vs-memory.md) | Two different questions, and how the two systems compose |

## Guides

| | |
|---|---|
| [Coding agents](guides/coding-agents.md) | Keeping engineering decisions so that "why are we using OAuth?" has an answer with a date on it |

## Integrations

| | |
|---|---|
| [MCP](integrations/mcp.md) | The fourteen tools, the three ways to reach them, and which one to pick |
| [Frameworks](integrations/frameworks.md) | LangChain, LlamaIndex, LangGraph and CrewAI — and exactly what each adapter preserves and loses |

## Reference

| | |
|---|---|
| [API](API.md) | The whole surface, in the order you meet it |
| [Architecture](reference/architecture.md) | The real module map, as diagrams |
| [How it works](DESIGN.md) | Each design decision and the failure it prevents |
| [Internals](INTERNALS.md) | Module-by-module contracts and the invariants |
| [Deploying](DEPLOY.md) | As a library, as an MCP server, in Docker — plus configuration and operations |
| [Benchmarks](BENCHMARKS.md) | Every measured claim, with its method and its caveats |
| [Upgrading](UPGRADING.md) | The changes that do not announce themselves |
| [Roadmap](ROADMAP.md) | Done, deliberately deferred, and still missing |
| [Open core](OPEN-CORE.md) | What is Apache-2.0 and what is not |

## Also

| | |
|---|---|
| [FAQ](FAQ.md) | Eleven questions, answered against the implementation |
| [Limitations](LIMITATIONS.md) | Every limit this project knows about, in full |
| [Examples](../examples/README.md) | Three runnable programs, tested on every CI run |
| [Contributing](../CONTRIBUTING.md) | The bar a patch has to clear |
| [Security](../SECURITY.md) | Private vulnerability reporting |

---

Next: [Installation](getting-started/installation.md)
