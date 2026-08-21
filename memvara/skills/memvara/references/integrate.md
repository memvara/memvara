# Which surface to use

One store, three ways in. Pick one. Mixing them in the same project is how
people end up with two memories that disagree.

## Python, you own the process

```python
from memvara import Memvara, NullLLM

mem = Memvara("app.db", user="alice", llm=NullLLM())   # one per process
alice = mem.scope(user="alice")                        # one handle per request
alice.add("I live in Lisbon")
print(alice.recall("where do they live?"))
```

`pip install memvara`. Offline, no account, numpy and the stdlib. This is the
open core. Adapters exist for LangChain, LlamaIndex, LangGraph and CrewAI —
each one loses something the facade keeps, so read
https://memvara.dev/docs/guide#frameworks before you wrap.

`remember("user", "lives_in", "Lisbon")` when you already hold the triple.
`add()` when you have prose or a transcript and an extractor that can read it.

Do not stand up Postgres, REST, or the hosted product for a single laptop.
The library is complete for one machine.

## Editor — you are not writing a loop

Install the plugin (skill + hosted MCP in one step). See `hosted-mcp.md`.

If the client has no plugin format, paste `https://app.memvara.dev/mcp` and
approve in the browser.

The plugin does **not** go into an app you wrote. `claude plugin install`
inside a FastAPI project does nothing useful for that process.

## Any language, including JavaScript

There is no npm client. `require("memvara").implemented` is `false`; that
package exists so the name is ours.

Two options:

1. Speak MCP as a client, against `https://app.memvara.dev/mcp` (OAuth).
2. Call the commercial REST API (`/v1`) with a bearer key from the console.

Local stdio (`python3 -m memvara.server`) is the fallback when the store file
must stay on a machine you run. No `npx`. The npm package does not launch a
server.

## Hosted vs local vs library, in one line

- Several people, several agents, one project → hosted MCP (plugin or URL).
- You are writing the agent loop in Python → library.
- Legal will not let the turns leave the building → library or local MCP,
  `llm=NullLLM()` / `MEMVARA_LLM=none`.
- Not Python and not an MCP client → REST.

## What not to do

- Do not invent a JS SDK, a second REST shape, or extra MCP tools.
- Do not put `MEMVARA_DB` laptop paths in a committed `.mcp.json` for a team.
- Do not set `MEMVARA_SESSION` for facts that must survive the next chat.
- Do not tell a solo script to create a cloud account. The library is enough.
