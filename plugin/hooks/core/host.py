"""What one coding host's hook I/O looks like, written down as data.

Every hook body in this package used to spell Claude Code's protocol inline: the stdin
key a session id arrives under, the reply key a status line has to be printed against,
the tag a machine-generated prompt is wrapped in. Six sibling plugin repos are about to
vendor these same bodies for six other hosts, and a literal repeated in seven places
fails the way this repository's CLAUDE.md describes at length -- by doing nothing, in a
copy nobody is reading. So the literals move here, one `Host` record per client, and the
bodies read them.

**The record says what a host cannot do, not only what it does.** A canonical hook name
absent from `events` is a hook that host has no event for; `context_key = ""` is a host
with no per-turn injection channel at all; `transcript = None` is a host where capture
cannot run. Those are the states a port gets wrong, and an absent key is the one spelling
that cannot be mistaken for a working default.

`collections.namedtuple` rather than `typing.NamedTuple`, and that is a cost decision
rather than a style one: `typing` is not otherwise imported anywhere on the per-prompt
path and costs 3-5ms measured to import, against a client budget of ~30ms. This
repository already refuses `pathlib` there for 10.5ms. `collections` is already loaded by
the time any hook runs, so this is free. Field names are exactly what the class-syntax
version would have had.
"""

from __future__ import annotations

from collections import namedtuple

#: One hook invocation, in the shape the bodies read. `raw` is the undecoded stdin object,
#: kept so a body can reach a field no host record has been taught yet -- and so that a
#: reader debugging a port can see what actually arrived rather than what we extracted.
Event = namedtuple(
    "Event",
    "hook session cwd prompt transcript_path tool_name reentrant raw",
)

#: One hook's answer, before any host has been asked how to spell it. `status` is the line
#: a person at the terminal sees; `context` is text put in front of the model;
#: `decision`/`reason` are the permission verdict a pre-tool event carries. Empty means
#: "this reply does not carry that", which is not the same as a host that cannot carry it
#: -- `Host.status_key` and `Host.context_key` decide that, and they decide it once.
Reply = namedtuple("Reply", "hook status context decision reason",
                   defaults=("", "", "", ""))

#: How a host keeps the conversation on disk, and therefore whether `capture` can mine it.
#: Presence is the capability flag: `Host.transcript = None` means this host does not hand
#: a hook anything to read back, so the Stop-equivalent body must not run at all.
#: `role_key` is the entry key that says who is speaking. Claude Code and OpenCode put it
#: under `type`; Cursor writes the same `message.content` blocks under `role`. One field
#: rather than a third reader, because that is the whole difference between them -- and a
#: host that differs in KIND rather than in spelling gets a `format` of its own instead,
#: as Codex does.
TranscriptSpec = namedtuple("TranscriptSpec", "format role_key", defaults=("type",))

#: Everything the pre-tool auto-approve needs that differs by host. Deliberately NOT the
#: read-only tool list: which memory_* tools are safe to run unprompted is a fact about
#: our own MCP server, identical everywhere, and duplicating it per host would let one
#: copy start approving a forget.
ApproveSpec = namedtuple("ApproveSpec", "matcher separators decision_key reason_key allow")

#: The headless CLI `capture` shells out to in order to mine a turn: the command without
#: the prompt, which is appended, plus where to read the answer out of the envelope it
#: prints. The recursion sentinel is deliberately absent -- that environment variable is
#: ours, identical on every host, and lives in `lib.ipc` for the same reason `READ_ONLY`
#: and `RECALL_MARKERS` do.
#:
#: The three keys travel together because they describe one thing: the JSON object one
#: CLI prints in its `--output-format json` mode. `reply_key` holds what the model said,
#: `usage_key` what it cost, and `error_key` the flag that turns an exit-0 run into a
#: failure. Splitting them -- two on the record and one still spelled inline in
#: `lib/extract.py` -- would mean a port that got its reply key right could still read
#: every error as a success, which stores nothing and reports nothing.
#: How to read a reply out of a CLI that prints a JSONL EVENT STREAM rather than one
#: envelope object. `claude -p --output-format json` prints a single object and needs
#: none of this; `codex exec --json` and `opencode run --format json` both print a line
#: per event, so the reply has to be selected out of the stream instead of looked up.
#:
#: `reply_match` is a tuple of `(dotted key, expected value)` pairs that must ALL hold for
#: a line to be part of the reply, and `reply_path` is where the text sits in that line.
#: Every matching line is concatenated, because a host may stream one text part per chunk.
#: `usage_match`/`usage_path` are the same for the accounting line.
#:
#: Declarative on purpose. The alternative was a parser per host, which is the shape this
#: package moved everything else away from -- a host difference belongs in the record.
StreamSpec = namedtuple("StreamSpec", "reply_match reply_path usage_match usage_path")

#: `stream` is None for a CLI that prints one JSON object, in which case `reply_key`,
#: `usage_key` and `error_key` are read from it. When `stream` is present those three are
#: unused and the `StreamSpec` decides everything.
#: `model` is the model this argv PINS, or "" when the CLI mines with whatever the user
#: has configured. It exists because `usage.jsonl` records what a run cost against a model
#: name, and that name has to be the one actually invoked. While every host shelled out to
#: `claude -p` a single constant was correct; now that a host mines with its own CLI, a
#: hardcoded label would record spend against a model that never ran -- wrong in the one
#: file whose whole job is to say what was spent. An empty string is recorded as the
#: program name, which is true and checkable, rather than a guess at the user's default.
ExtractorSpec = namedtuple(
    "ExtractorSpec", "argv reply_key usage_key error_key stream model",
    defaults=(None, ""))

#: The model `claude -p` is asked for. Named once because it is spelled twice: in the
#: argv below, and as the label `lib.extract` writes to `usage.jsonl`. Two spellings
#: would let the ledger name a model that was never invoked, which is wrong in the one
#: file that exists to say what was spent.
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

#: Codex's own headless CLI, measured on codex-cli 0.151.0. `codex exec --json` prints
#: `{"type": "item.completed", "item": {"type": "agent_message", "text": ...}}` for the
#: reply and `{"type": "turn.completed", "usage": {...}}` for the cost.
#:
#: `--skip-git-repo-check` because the extractor runs wherever the turn happened, which
#: is frequently not a repository, and Codex refuses outside one without it.
CODEX_CLI = ExtractorSpec(
    argv=("codex", "exec", "--skip-git-repo-check", "--json"),
    reply_key="", usage_key="", error_key="",
    stream=StreamSpec(
        reply_match=(("type", "item.completed"), ("item.type", "agent_message")),
        reply_path="item.text",
        usage_match=(("type", "turn.completed"),),
        usage_path="usage",
    ),
)

#: OpenCode's own, measured on opencode 1.18.20. `opencode run --format json` prints
#: `{"type": "text", "part": {"text": ...}}` for the reply and
#: `{"type": "step_finish", "part": {"tokens": {...}}}` for the cost.
#:
#: No `--model`: the point of a host-native extractor is that it mines the turn with the
#: model that user already chose and pays for. Naming one here would override their
#: configuration from inside a hook they did not read.
OPENCODE_CLI = ExtractorSpec(
    argv=("opencode", "run", "--format", "json"),
    reply_key="", usage_key="", error_key="",
    stream=StreamSpec(
        reply_match=(("type", "text"),),
        reply_path="part.text",
        usage_match=(("type", "step_finish"),),
        usage_path="part.tokens",
    ),
)

#: Cursor's own headless CLI, measured on cursor-agent 2026.08.25. It prints a SINGLE
#: envelope rather than an event stream -- `result`, `usage`, `is_error` -- which is the
#: same shape `claude -p` prints, so this needs no `StreamSpec`.
#:
#: `--mode ask` is not a preference. `-p` alone "has access to all tools, including write
#: and shell", and what this command is handed is a mined turn: arbitrary text, including
#: anything the user pasted into their session. Read-only mode is what makes the `--trust`
#: beside it defensible -- and `--trust` is required, not optional, because an extraction
#: runs wherever the turn happened and Cursor refuses an untrusted directory outright.
#: Granting trust to a command that can only read is a different act from granting it to
#: one that can run shell.
CURSOR_CLI = ExtractorSpec(
    argv=("cursor-agent", "--trust", "--mode", "ask", "-p", "--output-format", "json"),
    reply_key="result",
    usage_key="usage",
    error_key="is_error",
)

#: One consequence of mining with the user's own model, recorded because it is a real
#: trade and not a free win: extraction latency and quality become theirs. Measured on
#: 2026-09-01 against a free OpenRouter model configured in OpenCode, one run exceeded
#: `extract.TIMEOUT_SEC` and logged "no reply within 90s" before a retry succeeded, and
#: the reply came back conversational where `claude -p` returns the terse list the prompt
#: asks for. Both are handled -- a timeout is a logged failure that raises the capture
#: alert, and a reply the parser cannot use stores nothing rather than storing junk -- but
#: a user on a slow or small model will see capture fail more often than one on Claude
#: Code, and the log is where they will see it.

#: The second rung of the extractor chain, available to every host rather than only to
#: the one that packages Claude Code.
#:
#: It lives in `core/` -- the half of this tree that is the same bytes in every plugin
#: repository -- because it is a fact about one CLI product and not about any host. A
#: Codex or Cursor user with Claude Code also installed can mine turns with it, and the
#: `hosts/codex.py` in that repository must not have to carry a copy of this argv to say
#: so. `hosts/claude.py` names this record as its own extractor rather than restating it,
#: so the `--settings` guard below is the only spelling of that flag anywhere.
#:
#: `--settings '{"hooks":{}}'` clears the hooks a settings file declares. It does NOT
#: clear the ones a plugin registers -- measured, with a marker file: the child still
#: fires this plugin's own SessionStart and UserPromptSubmit. So it is one of two guards
#: against recursion and not the load-bearing one; the sentinel in `lib.ipc` is what
#: actually stops it, and `ipc.under_extraction` is what stands the read hooks down. Kept
#: because a settings-declared Stop hook is a real way in.
CLAUDE_CLI = ExtractorSpec(
    argv=("claude", "-p", "--settings", '{"hooks":{}}',
          "--model", CLAUDE_MODEL, "--output-format", "json"),
    reply_key="result",
    usage_key="usage",
    error_key="is_error",
    model=CLAUDE_MODEL,
)

Host = namedtuple(
    "Host",
    "id plugin_root_env events fields envelope context_key status_key "
    "context_token_cap context_limit_key supports_async detach_capture "
    "timeouts client_configs config_format "
    "transcript tools noise skip_prefixes machine_prompt_prefixes reentry_field approve "
    "extractor extractor_label description",
)

#: Two fields that both sound like "capture must not block" and are NOT the same fact,
#: separated because one host has one and not the other.
#:
#: `supports_async` says the client will run the capture hook asynchronously if the
#: registration asks it to. `detach_capture` says this host's capture hook has to detach
#: ITSELF -- `run.py` re-execs into a new session and returns immediately.
#:
#: Codex is why they are two. Its registration schema accepts `async: true` and its
#: documentation offers it, but an async hook there does not run AT ALL: measured on
#: codex-cli 0.151.0, an async Stop hook wrote no receipt even though writing one is the
#: first statement in the script. The same hook declared synchronous fires, and a child it
#: spawns with `start_new_session=True` OUTLIVES the `codex exec` process and finishes
#: twelve seconds after the turn ended. So Codex is `supports_async=False` -- asking for
#: async would silently disable capture -- and `detach_capture=True`.
#:
#: A host may have neither (capture blocks, and must be short), one, or in principle both.
#: Nothing infers one from the other, because "the client honours the flag" and "we fork"
#: fail in opposite directions: guessing the first wrong loses the hook, guessing the
#: second wrong holds the turn open for the whole extraction.

#: Canonical hook names. The bodies and `run.py` speak these; `Host.events` maps each to
#: whatever the host calls the event it fires.
HOOKS = ("session_start", "recall", "capture", "approve")

_ACTIVE: "Host | None" = None


def use(host: "Host") -> None:
    """Bind this process to one host. `run.py` calls this before importing a body.

    Before, not after: `lib.transcript` resolves the host's noise markers at import time,
    so a body imported ahead of this call would be built against the wrong client.
    """
    global _ACTIVE
    _ACTIVE = host


def active() -> "Host":
    """The bound host, defaulting to Claude Code.

    The default is what makes `python3 hooks/recall.py` -- with no arguments, the way this
    plugin has always been invoked and the way its tests drive it -- keep meaning exactly
    what it meant before. `hosts.default` names it, not this module: `core/` is meant to
    be the same bytes in every repository that vendors these hooks, and the identity of
    the client is exactly what is not. Imported lazily so a host resolved by `run.py`
    never pays for a second one.
    """
    if _ACTIVE is None:
        from hosts import default

        return default()
    return _ACTIVE
