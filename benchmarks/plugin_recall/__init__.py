"""Plugin Recall Benchmark: what a memory plugin actually puts in the model's context.

This is not a benchmark of a memory *library*. `benchmarks/agent_memory` already measures
that, and measures it at the layer where a library can be asked a question directly. This
one sits a layer up, at the only place a user ever experiences a memory plugin: the editor
fires a hook on every prompt, the hook writes text, and that text is spent from the same
context window the user's own work lives in.

The seam is the host's hook protocol, not any vendor's API. Every memory plugin for Claude
Code declares a `UserPromptSubmit` entry in `hooks/hooks.json`, is handed the prompt as
JSON on stdin, and answers with `hookSpecificOutput.additionalContext`. memvara and
supermemory were both read before this was written and both speak exactly that, so the
harness needs no per-vendor code at all -- a plugin is graded through the same contract the
editor grades it through, which is the only contract that is true by construction rather
than by our say-so.

## Two populations, never one number

A memory plugin that injects its whole store on every prompt scores perfectly on any
benchmark built only from prompts that have a right answer. So half of this corpus has no
right answer: prompts where the correct behaviour is to say nothing at all. `hit_rate` and
`silence_rate` are reported side by side and the balanced mean is the only combined figure,
because it is the only one an always-inject system cannot win.

Cost is reported next to both, in tokens, because injected context is not free and the
question "is this plugin worth its tokens" is unanswerable without it.
"""

from __future__ import annotations

__all__ = ["__version__"]

#: Bumped when a change would move a published number. The report prints it, so a result
#: table pasted into an issue can always be traced back to the harness that produced it.
__version__ = "1"
