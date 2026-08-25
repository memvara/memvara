# Scope

A store is partitioned as `tenant / user / agent / session`. `*` means that
field is unbound.

On MCP the scope is fixed when the server starts. No tool argument changes
it. Call `memory_stats` once, early, in any conversation where you expect to
write. It spells the bound scope.

## The session trap

If the session field is not `*`, the process was launched with
`MEMVARA_SESSION` set. Everything you write is invisible to the next chat.
Nothing in an ordinary write result will tell you this.

Say it when you store something: "noted for this session, it will not carry
over." Letting them believe a durable fact was kept is a lie.

Unsetting `MEMVARA_SESSION` is an operator change to the client's env block,
not a second write.

## Teams

One hosted project, scope by user. Do not give each teammate a laptop SQLite
file and hope they merge. Do not commit a `MEMVARA_DB` that is a path on
your machine.

For a custom Python loop, `mem.scope(user=...)` per request. One `Memvara`
per process.

## What the four fields mean

- **tenant** — the isolation boundary above a user. Default `default`.
- **user** — who the facts are about. Unset on a local server means the
  whole tenant, which is right for a single-person machine and wrong for a
  product with customers.
- **agent** — which program wrote it. Usually unbound.
- **session** — this conversation. Leave unbound for anything that should
  still be true tomorrow.
