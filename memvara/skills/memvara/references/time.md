# Two clocks

Memvara stores when a fact was true in the world, and when this system came
to believe it. Those are independent. A correction that arrives in August
about June is invisible if you rewind both clocks to June.

## What to call

On the **library and REST**:

- `valid_at=T` — what we believe *today* about how the world was at T
- `known_at=T` — what we believed at T, about the world as it is now
- `as_of=T` — both clocks at T. Sugar for `valid_at=known_at=T`. Passing it
  alongside either axis raises rather than picking one.

On **MCP**: `memory_search` takes `as_of` only. The two axes separately are
not on the tools. If they need one clock and not the other, say you cannot
do that from here and point them at the library or REST. Do not guess which
clock they meant and silently send `as_of`.

`memory_history` is "what the value used to be", not "why it was written".
`memory_why` is the latter.

## Writing time

`memory_remember` / `remember()` take when the fact was true in the world
(`true_since`, `true_until` on the tool; `valid_from`, `valid_to` on the
library). Belief time — the instant this system learned it — is always
now. No tool and no honest caller sets `recorded_at`. A caller who could
would write an audit trail that says the system knew a fact before it did,
which nothing downstream can falsify.

If they are telling you about last month, send `true_since`. Leaving it off
makes a past fact claim it started at this instant, and a later close at
the real end can fail because the range would run backwards.

## Library names

```python
mem.get_all(valid_at=T)
mem.get_all(known_at=T)
mem.get_all(as_of=T)
```

Same three on `search`, `count`, `history`, `why`, `produced`,
`neighborhood`, `paths_between`.
