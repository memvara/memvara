# Security policy

## Reporting a vulnerability

**Report privately, through GitHub, not in a public issue.**

Open a draft advisory at
<https://github.com/memvara/memvara/security/advisories/new> — the *Security* tab of this
repository, then *Report a vulnerability*. That channel is private between you and the
maintainers, it lets us work on a fix in a private fork, and it can request a CVE and
publish the advisory when the fix ships. There is deliberately no email address in this file:
an address rots, gets filtered, and ends up in one person's inbox while they are on a
plane.

If GitHub's advisory flow is unavailable to you for some reason, open a public issue that
says only *"I have a security report and need a private channel"* — with **no details** —
and we will open the advisory from our side and invite you to it.

### What to put in the report

- What an attacker gains, concretely. "Reads another user's claims" is a report; "scope
  handling looks wrong" is a question.
- A reproduction that runs offline, as a script or a failing test. Everything in this
  library runs with no network and no API key, so a repro that needs neither is achievable
  and is what gets a fix written fastest.
- The version (`memvara.__version__`) or commit, the Python version, and the platform.
- Whether you have disclosed it anywhere else, and any deadline you are working to.

### What to expect

This is a small project with no paid on-call, so these are honest targets rather than a
guarantee: acknowledgement within **3 working days**, an assessment of whether we agree it
is a vulnerability within **10**, and a fix or a public explanation of why not within
**90**. If a date matters to you, say so in the report and we will tell you whether we can
meet it rather than letting it pass quietly.

We will credit you in the advisory and the changelog unless you ask us not to. There is no
bug bounty, and we would rather say that here than have you find out after the work.

## Supported versions

`main`, and the most recent PyPI release (`0.3.0`). A fix goes to `main` and into the
next release; older wheels stay on the index and get an advisory if the defect reaches
them. The npm package is a name reservation (`0.0.1` on the registry, `0.0.2` pending)
with no runtime surface to backport to.

## In scope

These are the surfaces where a defect is a vulnerability rather than a bug. They are named
specifically because each one is a place the design already made a decision, and the
decision is the thing to attack.

### Scope isolation

`tenant > user > agent > session`, with inheritance downward and no leakage sideways.
Anything that returns a claim, an episode, an entity, a path or a provenance record to a
reader whose scope should not see it is in scope.

The rule is `Scope.sees` in `memvara/types.py`: a handle sees its own scope and every
**broader** one, never a deeper one. Its near-twin `Scope.contains` reaches *downward* and
is deliberately used by `forget()` and `history()`, which are slot operations where a broad
caller reaching down is the intent. Confusing the two is a real bug class here — it was one
already, and `get()`/`why()` were fixed to use `sees` — so a case where an id-addressed
read, a graph hop, or a new method authorizes with the downward rule is exactly the report
we want.

Two things that are **not** mitigations, so do not discount a finding for them:

- **Claim ids are not secret.** Receipts, `invalidated_by` pointers, search results and
  logs all leak them. "The attacker needed the id" is not a defence.
- **Scope filters are supposed to fail closed.** A scope that resolves to nothing must
  match nothing. A path where a filter degrades into an unfiltered query across every user
  is a high-severity finding even if reaching it takes an unusual configuration.

### Erasure completeness

`erase()` and `purge()` are irreversible deletion, not retirement, and the guarantee is
that everything derived from the text goes with it: the claim row, the FTS5 entries (which
store the tokens directly), the embedding (which leaks content under inversion, and is
zeroed in place in the `.vecs` sidecar), the entity rows in `entities.canonical` (which
keep the first spelling ever seen of every subject and object), and — with `sources=True`,
or always for `purge` — the source turns.

**Recoverable text after a call that reported success is in scope**, and the reported
per-table counts are part of the guarantee: a count that says the data is gone while it is
not is worse than a call that refuses. That failure has happened here before — entity rows
survived `purge()` while `stats()` reported zero — which is why this section is specific
about which tables are covered.

`forget()` and `delete()` **retire**; they are documented as leaving the text readable and
are not in scope for erasure claims.

### Provenance and the audit trail

`why()`, `history()` and `search(as_of=…)` are the reason to use this library at all, so
anything that lets a caller forge or corrupt them through a public API is in scope:

- A claim that can be made to cite an episode it was not derived from, or to lose the
  citation it had.
- A write that reaches internal bookkeeping through a documented argument. `remember(**meta)`
  accepting `salience_base` — a permanent ranking override reachable through no documented
  parameter — was exactly this, and reserved keys are now rejected at the boundary. Another
  one would be the same class.
- A retirement that leaves both values live, or a backdated write that rewrites a history
  it should only have appended to.

### The prompt-injection surface in `recall()`

`recall()` renders stored text straight into a system prompt, and stored text is
attacker-controlled — a user can say anything, and `remember()` stores it verbatim. This is
stored XSS against the agent, and the rendering boundary is where it is neutralised. Three
defences, all in `memvara/core.py`, all worth attacking:

- **`_safe_line`** collapses each claim to a single line and strips leading list and
  heading markers, so stored text cannot open its own bullet list or repeat the header and
  forge a block indistinguishable from the real one. It also maps `[` and `]` to their
  fullwidth forms, because flattening only settles what a claim can do *between* lines: a
  surface that writes its own metadata as `[id=… relevance=…]` can be impersonated by a
  claim that spells one of those out and appends it to the row it is already on, with no
  newline needed. Every renderer here — `recall()`, and each line the MCP server emits —
  goes through this one function, so that is the one place the character set lives.
  Episodes are additionally truncated, so a pasted stack trace cannot become the whole
  prompt.
- **`RECALL_HEADER` and `RECALL_EPISODE_HEADER`** frame the block as retrieved data rather
  than instructions, and the episode header says "said", not "true".
- **The signature is explicit rather than `**kwargs`**, so `states`, `include_invalidated`
  and `as_of` are not reachable from `recall()`. `include_invalidated=True` would resurrect
  retired claims into a live prompt — an un-delete reachable by anyone who can influence a
  parameter. `states=["retired"]` is the sharper form of the same attack: where
  `include_invalidated=True` at least returns live claims alongside the dead ones,
  `states=["retired"]` builds a prompt out of nothing *but* records we stopped believing.
- **`include_history=True` is the one bounded exception, and the bound is the whole
  point.** It renders non-live claims, so it is the same door — but only `ended` ones,
  never `retired`. That is not a tidying rule: an `ended` value is the fact's own past and
  we still believe it was true while it was in force, whereas a `retired` value is
  something we were wrong about or were asked to delete, and putting one in a prompt is
  the un-delete above. A claim that ended and was *later* retired is `retired` and stays
  out. The filter is `state == "ended"`, never `state != "live"`, and
  `tests/test_api.py::test_recall_can_carry_the_past_of_a_fact_without_carrying_a_retired_one`
  holds all three states in one slot so the looser spelling cannot pass.

  Reaching a retired claim through `recall(include_history=True)` is in scope.

A way to break out of that block, forge a header, or reach a retired claim through
`recall()` is in scope. A model choosing to follow instructions that are correctly framed
as data is a model behaviour, not a memvara vulnerability — but a case where the framing
itself can be removed or spoofed is ours.

**Stored claims are not the only untrusted text that reaches a model here.** A tool that
fails returns the exception's message as a tool result, and that message is not this
process's to trust: a store error can quote a value somebody wrote, and against a hosted
backend it can carry an upstream body verbatim. It goes through the same neutralisation a
claim does, plus a length cap — `safe_detail` in `memvara/server/tools.py`. Text that
reaches a model through a *failure* path without that treatment is the same finding as
text that reaches it through a result, and is in scope on the same terms.

### The redaction seam

`memvara/redact.py` is the last place text can be changed before it becomes durable, and
the ordering is the whole feature: the redactor runs **before** the content hash, the
episode row, the FTS index, the embedder and the extraction model. Two of those leave the
process. A code path where text reaches disk, an index, or a third party before the
configured `Redactor` sees it is in scope, and so is one where a field listed in `FIELDS`
is written without passing through it.

The **rules** are not in scope — see below.

### The MCP server

`memvara/server/` speaks JSON-RPC 2.0 over stdio, and its security model is that the
process *is* the user: scope is bound at startup from the environment and there is no
caller-supplied scope string for a model to be talked into changing. In scope:

- A tool call that reads or writes outside the bound scope.
- A tool call that reaches an irreversible operation. `consolidate`, `purge`, `reset` and
  `erase` are deliberately not tools, and a test asserts their absence; a route to one
  through the eight that exist is a finding.
- Anything written to stdout that is not a JSON-RPC message, which desynchronises a client.

### Injection into the store

SQL or FTS5 injection through any public method, path traversal through a store path, or
input that leaves the store internally inconsistent. There is a fuzz corpus covering this
(SQL and FTS5 syntax, path traversal, template injection, control characters, astral-plane
codepoints, combining marks) driven through every public method — a case it misses is worth
reporting with the input that does it.

## Out of scope

Not because they do not matter, but because they are known, documented, and deliberate.
A report about one of these is a feature request, and we would rather you spend the effort
on the list above.

- **`PatternRedactor` failing to match a PII format.** It says in its own docstring that it
  is not compliance-grade. It is a default and a demonstration of the seam, not a ruleset;
  a serious deployment brings its own `Redactor`. The telemetry pair `redact.inspected` /
  `redact.changed`, tagged by field and script, exists precisely because a rule set that
  stops matching is otherwise silent.
- **No encryption at rest.** Documented in the README with the reasoning: SQLCipher works
  and costs +43–48% on writes, but the mmap-backed `.vecs` sidecar sits *outside* its
  page-level boundary, and a plaintext vector beside encrypted text is a confirmation
  oracle you can hill-climb. Encrypting one and not the other would be theatre. Full-disk
  encryption is the honest answer today.
- **`-wal` residue after erasure.** Erasure removes the rows, the index entries and the
  vectors, and overwrites the pages they occupied — `PRAGMA secure_delete=ON` and FTS5's
  own `secure-delete` are both set by the store, so the text is gone from the main
  database file without a `VACUUM`. What is *not* scrubbed is the write-ahead log: an
  erased claim's bytes can remain in `-wal` until it checkpoints. A checkpoint or a clean
  close clears it.

  **This bullet used to say the opposite of what the code did.** It claimed the index
  entries were deleted and named `VACUUM` as the lever for what was left. Neither was
  true of the text index: `DELETE FROM claims_fts` writes a delete marker and keeps the
  document's terms as *live rows* in a shadow table, where no `VACUUM` reaches them. Fixed
  in schema 7; `tests/test_erasure_residue.py` greps the file rather than asking the
  store, because asking the store always answered correctly.
- **An attacker who already has the database file, the `.vecs` sidecar, or the process's
  memory.** The store is a file with the filesystem's permissions and nothing more. It
  makes no attempt to defend against someone who can read it.
- **Resource exhaustion from input you supplied to your own process.** This is an
  in-process library; a caller who passes a 500 MB string is doing it to themselves.
- **Vulnerabilities in optional provider SDKs** (`anthropic`, `openai`,
  `sentence-transformers`, the framework packages). Report those to their maintainers. A
  way for *memvara* to hand one of them something dangerous is ours.
- **The hosted commercial platform.** It is a separate product in a separate repository;
  this file covers what is in this one. Report anything there through its own support
  channel, not here.
