# Moving off Supermemory

```python
from memvara import Memvara, NullLLM
from memvara.compat import import_supermemory

mem = Memvara("migrated.db", user="alice", llm=NullLLM())
print(import_supermemory(mem))
```

The key comes from `~/.supermemory-claude/credentials.json`, written when
they signed in with the Supermemory plugin. Pass `api_key=` to override,
`container_tag=` to take one container rather than the account, and
`fetch=` to route through your own HTTP client.

**Say this before they run it.** Supermemory keeps documents, not a
mutation log. mem0 records what changed and when, so `import_mem0` can
rebuild supersession and answer `as_of` afterwards; Supermemory records
the current state, so nothing here can reconstruct a history it was never
told. Documents arrive as **episodes** on their original `createdAt`.
The timeline is true. The claims are not derived.

Claims appear only if the store has an extractor. Under the default
`MEMVARA_LLM=none` most documents yield none, and the receipt says so in
those words rather than reporting a number that looks like failure.

**The part that looks like a broken import and is not:** plain
`memory_recall` answers from claims, so a store holding only imported
episodes returns nothing however much it holds. Pass
`include_episodes: true`. Say so at the same time as you run the import —
otherwise they migrate, search, find nothing, and conclude it failed.

Nothing is retired. An import adds; it never closes a value already
stored.
