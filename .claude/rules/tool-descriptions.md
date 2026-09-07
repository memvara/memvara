---
paths:
  - "memvara/server/tools.py"
---

# Tool descriptions are documentation a model reads at runtime

The `description` on every entry in `TOOLS` is not prose for a human. It is the only thing an
agent has when it decides which memory tool to call and what to pass. Wrong text here does
not confuse a reader who can go and check; it misleads a caller who cannot.

Read how the existing descriptions are written before adding to them. They are deliberately
longer and more careful than the rest of this repository's prose.

## Precision outranks plain wording here

`CLAUDE.md`'s "prefer simple words when they are accurate" never outranks precision in this
file. Plain is good; vague is a defect. The same exemption applies to a `CHANGELOG.md` entry
describing a behaviour change somebody will act on.

## The precedent to study: ending, retiring and erasing

Three words name three different events, and the difference is the product. The full
definitions, with the functions that write each state, are in `docs/claude/memory-model.md`;
this is the short form a model needs, and the two must say the same thing.

- **Ended** says the world changed. The fact was true and has stopped being true, closed at
  the instant it stopped. It keeps answering questions about the period it held.
- **Retired** says the record was wrong. We stop believing it, and it stays visible to
  `memory_history` and `memory_why`.
- **Erased** removes the bytes. It is an operator action and is deliberately not exposed as a
  tool.

This has already gone wrong once. A write receipt reported `retired 1` for a fact that had
merely stopped being true, which left a model reading its own memory tool with three names
for two events. The module docstring in `memvara/server/tools.py` calls this the one mistake
here that cannot be found by reading the data afterwards, because a false reason for a change
looks exactly like a true one in every row and every log.

## Practical rules

- Say what the tool does, then what the caller must decide. Do not describe the
  implementation.
- Use one term per concept, and use the same term the receipt uses. If the receipt says
  `retired`, the description says retired.
- State refusals. A tool that will reject an argument should say so, because the alternative
  is the model learning it from an error.
- Never write text that could be read as an instruction the model must obey on behalf of
  stored data. Recalled content is reference data, and the descriptions and `INSTRUCTIONS` in
  `memvara/server/mcp.py` say so on purpose.
- Changing a description is a documentation change that ships with its code, and the
  packaged skill at `memvara/skills/memvara/SKILL.md` deliberately does not repeat what a
  description says. Text moving between the two has to move in both.

`tests/test_server.py` and `tests/test_init.py` hold the checkable parts of this.
