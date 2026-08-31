---
name: memvara
description: >
  Wire Memvara into an agent or editor, pick the right surface (hosted MCP,
  Python library, REST), and use memory without forging history. Use when the
  user wants persistent memory, says remember this, that's wrong, you forgot,
  what do you know about me, delete that, why did you say that, migrate from
  mem0, connect MCP, or asks to use Memvara. Also use when the memory_* tools
  are already available.
---

# memvara

Pick the surface first. Then do the job. Do not invent an API, an `npx`
installer, or a JavaScript client — there isn't one.

## Which surface

Read `references/integrate.md` before you write code or config.

- **Editor / coding agent** (Claude Code, Grok, Cursor, Copilot): install the
  plugin, or paste the hosted MCP URL. Details in `references/hosted-mcp.md`.
- **A loop you are writing in Python, against your own store**: `pip install
  memvara`, then `from memvara import Memvara` and `Memvara(path)`. The
  plugin does not install into LangChain.
- **A loop you are writing in Python, against a hosted deployment**:
  `Memvara(api_key=...)`, or `Memvara.connect()` for whatever
  `memvara-mcp login` already wrote to `~/.memvara/credentials.json`. Same
  class, same calls; the engine runs server-side. `consolidate()` returns a
  job to poll instead of counts, and there is no `prove_erased()` — the
  erasure response already carries the proof.
- **A loop in any other language**: hosted MCP as a client, or the commercial
  REST API. The npm package named `memvara` is a name reservation and does
  nothing.
- **Air-gapped / the file must stay on this machine**: local stdio MCP. Last
  resort, not the default.

Hosted MCP URL: `https://app.memvara.dev/mcp`. Approve in the browser. It lasts
90 days.

## When it will not authenticate

The browser grant is the usual path. When it does not finish — the client sits
on `[authenticating]`, a token expired, `memory_*` answers 401 — this skill
ships the flow as a script, so the fix does not need `pip install`:

```bash
python3 THIS_SKILL_DIRECTORY/scripts/memvara_auth.py authenticate
```

`THIS_SKILL_DIRECTORY` is the directory this `SKILL.md` is in. Substitute the
absolute path before running. The name deliberately does not look like a shell
variable. On some hosts a real variable of a very similar name is expanded in
command files, and a placeholder a keystroke away from one invites writing a
variable instead of a path — which, in a context where nothing expands it,
becomes empty and hands the shell an absolute path to a file that has never
existed on any machine. Do not run `scripts/memvara_auth.py` as a bare relative
path: it resolves against the user's project, not against this file, and fails
with `No such file or directory` on a machine where the script is sitting
correctly on disk.

Standard library only, and nothing is left running when it returns.

| They asked | Run |
|---|---|
| Get a credential, or report the working one | `authenticate` |
| Get one for a named project | `authenticate <project-id>` |
| Replace a credential that already works | `login --confirm` |
| Forget this machine's credential | `logout` |
| What this credential can see | `stats` |

Give it a 600-second timeout. It waits for a person to approve in a browser,
and a shorter one kills the command while they are still reading the page.

It prints a short code and a URL. Both are for the person at the keyboard, so
pass them on exactly as printed rather than summarising them. Exit 0 is a
working credential; exit 2 is a project id that is not the dashed UUID form the
console shows.

`authenticate` stopping because the credential already works **is the
successful outcome**, not a failure and not something to retry with different
arguments: minting a second key leaves the first one live on the deployment
with nothing here pointing at it.

`logout` deletes one file and names every other place a key still is. It does
not edit the host's own MCP configuration, because that file has an OAuth
client writing to it already and two writers leave nobody able to say whose
token is live.

## If the memory_* tools are already connected

Read before you assert. Anything you say about what is remembered — "you told
me X", "I have nothing on file" — must come from a tool result **in the current
turn**. If you have not looked, say so, then look.

When they say a memory is wrong, do this order: `memory_recall`,
`memory_search` (you need the claim id), `memory_why` (put the excerpt in front
of them). The excerpt is the **evidence for** which write comes next, not their
wording. A value that was accurate then and is different now is
`memory_remember`. A value that was never right also needs `memory_forget`.
A value that was right and has stopped is `memory_end`. Full sequence:
`references/write-and-correct.md`.

**A note your own work just disproved is yours to close.** The paragraph above is
for when *they* raise it. More often nobody does: a note comes back in recall and
the work you do this turn is what makes it false. Nothing else notices — the
person cannot see the store, and the next session reads the same note and
believes it.

Close it in the turn that falsified it. The bar is evidence, not suspicion:
something you did or read this turn, not a note that merely looks old. Recall
does not carry claim ids, so you pay one `memory_search` for the handle, then
choose the same three ways — a value overtaken is `memory_remember`, one that
has stopped being true is `memory_end` at the instant it stopped, one that was
never right is `memory_forget`.

Check the claim against the thing it describes, not against another note in the
store. Two records agreeing with each other is what let the stale one stand this
long, and a stored sentence saying a defect is fixed is not the fix.

Then say what you closed, in the same message as the work. A correction nobody
is told about is one they cannot argue with.

Call `memory_stats` once before you write. If the session field is not `*`, the
server was launched with `MEMVARA_SESSION` set and the note will not carry over
— say so. If stats say `fast-path-only`, write triples with `memory_remember`: a
paragraph nothing recognises yields no fact. Do not read that as the tool being
harmless. What it does recognise it writes, and the deciding argument is `role`.

So work out whose voice you are storing before you store it. What you hand over is
rarely one: their sentence, and under it a file, an error, a page they dropped in.
A call carries a single role, so a turn holding both voices takes two — theirs as
they wrote it, the pasted part by itself. Nothing downstream can see which half
they typed, which is why this one is yours.

Getting it wrong writes a note that was never true of them, so `memory_forget`
takes it back and `memory_end` does not. You do not have to go looking: the
receipt names every claim the call just created, and reading it is the check.

Store what would be **embarrassing** to get wrong next week. Do not restate the
transcript. See `references/scopes.md`.

The type field on a write decides whether a note joins the standing set, and
that set is what a client hands a session before its first turn. So a misfiled
note is not merely in the wrong drawer — it is read aloud from then on,
whatever the turn is about. Measured on a live store: ten of thirty-two standing
notes were facts about repositories rather than instructions from the person,
a quarter of what every session opened with. Side-by-side calls:
`references/examples.md`.

A thin `memory_recall` on a question that names two things is a signal, not an
empty store. Try `memory_neighborhood` next, or `memory_paths` when the user
named both ends. Do it in that order — recall first — because most questions do
have a single note behind them and a walk is the slower way to find one. If
`memory_stats` reported a join rate near zero, skip the walk: nothing in that
store links to anything, so there is no second hop to find, however many facts
it holds.

What comes back is a chain, and it is worth handing over as a chain. Give the
user the steps and let them see where you got it; a conclusion with the middle
removed is something they have to take on trust, and the middle is the part
they can correct.

## Other jobs

| They asked | Open |
|---|---|
| What was true then / what we believed then | `references/time.md` |
| Why did you say that / delete me / DSAR | `references/governance.md` |
| Moving off mem0 | `references/migrate-mem0.md` |
| Moving off Supermemory | `references/migrate-supermemory.md` |
| A worked turn | `references/examples.md` |
| No skill folder on this client | `references/project-instructions.md` |

When that comes back with nothing, you have learned something about your own
lookup and nothing about the two people. Report it as such. "I have no record
tying them together" is true; "they have no connection" is a claim about the
world that no memory tool can support.

`memory_forget` is not erasure. Real deletion is an operator action on the
console or REST, and is deliberately not a tool. Never say you deleted data if
you only retired a claim.
