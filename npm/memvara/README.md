# memvara

**This package is a name reservation. It does nothing.**

[memvara](https://github.com/memvara/memvara) is a bitemporal memory layer for AI agents:
structured facts with two independent time axes — when something was true in the world,
and when we learned it — so a corrected fact stops being returned without the history
being lost.

It is **a Python library**:

```bash
pip install memvara
```

There is no JavaScript client yet. This package exists so that the name on npm belongs to
the project it names, and it will be replaced by a real client if and when one is written.
Installing it gets you a notice object and no functionality:

```js
require("memvara").implemented;   // false
```

## Why publish an empty package at all

Because npm reserves nothing otherwise. An organisation reserves the `@memvara/*` scope
and not the bare name, exactly as a PyPI organisation reserves no project name — only a
publish does. A project that exists, is being used, and has a name worth protecting has a
legitimate claim to that name; that is different from registering names you have no
relationship to, which is what npm's policy is actually against.

If you wanted this name for something else: sorry, and genuinely — open an issue. If the
project is ever abandoned, the right thing is to hand it over rather than sit on it.

## If you were looking for a JS client

There isn't one. For a JavaScript agent, speak [MCP](https://memvara.dev/docs/agents)
against `https://app.memvara.dev/mcp`, or call the commercial REST API. The
[skill](https://github.com/memvara/memvara/blob/main/memvara/skills/memvara/SKILL.md)
is markdown you can paste into a system prompt.

Say so on the [issue tracker](https://github.com/memvara/memvara/issues) if a real
client would change that. The number of people who ask is most of the answer.

## License

Apache-2.0, the same as the library.
