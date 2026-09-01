# Plugin Recall Benchmark

What a memory *plugin* actually puts in a model's context, measured through the editor's
own hook protocol.

`benchmarks/agent_memory` grades memory **systems** — it asks a library a question and
scores the answer. This grades what a user experiences: an editor fires a hook on every
prompt, the hook writes text, and that text is spent from the same context window the
user's work lives in. A system that retrieves well and a plugin that injects well are
different claims, and only the second one is what someone installs.

```bash
python -m benchmarks.plugin_recall --plugin memvara
python -m benchmarks.plugin_recall --plugin supermemory --verbose
python -m benchmarks.plugin_recall --plugin ./path/to/plugin/root --json
```

## Why it needs no per-vendor code

Every memory plugin for Claude Code declares a `UserPromptSubmit` entry in
`hooks/hooks.json`, is handed the prompt as JSON on stdin, and answers with
`hookSpecificOutput.additionalContext`. memvara and supermemory were both read before this
was written and both speak exactly that. So the harness drives a plugin through the same
contract the editor drives it through — which is the only contract that is true by
construction rather than by our say-so — and adding a plugin means installing it, not
writing an adapter.

A plugin that recalls some other way (a tool the model chooses to call, rather than
injected context) is out of scope, and `discover()` says so rather than scoring it zero.

## Two populations, and why there is no single headline number

**`silence` cases** are prompts with no answer in anybody's store: bare acknowledgements,
general knowledge, arithmetic, and ordinary English words that collide with a developer's
vocabulary — *live*, *plan*, *table*, *gate*, *main*. Injecting anything is filler by
construction, which makes them **store-independent**: they grade the same against a full
store, an empty one, or a competitor's. Every one is synthetic and safe to publish.

**`hit` cases** assert that a specific fact is in a specific store. They cannot be shipped
— see `cases/build_private.py`, which derives them from your own telemetry, on your
machine, into a file you keep.

The two are reported separately and the balanced mean is the only combined figure, because
**a plugin that injects its whole store on every prompt scores 100% on hits alone.** That
is not hypothetical; it is the failure mode this harness was built to find. A run with no
hit cases loaded reports the hit rate as *unavailable*, never as zero: a plugin that was
never asked has not failed.

## The degenerate case, and the guard for it

A silence-only corpus has a fatal flaw: **a broken plugin scores 100%.** It injects
nothing on every prompt and looks perfect.

That is not a hypothetical either. The first live run of this harness graded a stale
development build that answered `recall failed` to all 22 cases, and the report said
*100% silence, 0 tokens*. A perfect score, produced by software that did not work at all.

Nothing in the host protocol distinguishes "I have nothing relevant" from "I am broken" —
both are an absent `additionalContext`. So the harness does not try to read that out of one
reply. It asks a weaker question it can answer: *did this plugin ever speak during the
run?* If not, the silence rate is withheld as `UNVALIDATED` rather than reported as a pass.
Prove your plugin can fail the test before quoting how well it passed.

## Reading the cost lines

`session preamble` is measured from an unscored warmup prompt that runs first. A plugin may
inject a once-per-session block — standing preferences, a store summary — and charging that
to whichever case ran first would make the score an artefact of corpus order, while running
each case in a fresh session would charge it to every case and overstate per-prompt cost
several times over. The warmup absorbs it; the scored cases measure the marginal cost of
one more prompt; the preamble is reported on its own line as the fixed cost it is.

Tokens are `characters / 4`, the same rough divisor the rest of this repository uses. Not a
real tokenizer, deliberately: the figure is compared between plugins measured identically,
and a dependency on one vendor's tokenizer would make the harness harder to run than the
thing it grades.

## Silence by family, and why a good score is not always a good reason

The report breaks silence down by case family, because **a plugin can score well without
judging relevance at all.** supermemory skips any prompt under twelve characters
(`MIN_PROMPT_LENGTH = 12`), which passes every bare acknowledgement in this corpus and no
lexical trap. That is a prefilter, not a relevance judgement, and one blended number would
present the two as the same achievement. The split says which mechanism earned the score.

## What a run is, and is not

A hook runs as a real subprocess against whatever store it is configured with, so a run
describes **that machine at that moment**. It is not reproducible across machines and the
report does not pretend otherwise. The silence corpus is the part that compares across
plugins; the hit corpus is a regression signal for one store over time.

## Adding a silence case

The bar is that **no store could legitimately answer it**. A prompt whose answer might
plausibly live in somebody's memory is not a silence case, however sure you are about your
own store — it would score a correct retrieval as a failure. Write the reason in `why`, in
a sentence that would survive someone disagreeing with you.
