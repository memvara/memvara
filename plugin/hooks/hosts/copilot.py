"""GitHub Copilot CLI, as a `Host` record.

Every value was measured against GitHub Copilot CLI 1.0.82 on 2026-09-01, with throwaway
plugins installed through the real install route into an isolated `COPILOT_HOME`. That
route matters: an earlier pass concluded this host does not run plugin-shipped hooks at
all, and it was wrong. The hooks had been mounted with `--plugin-dir` and with a
hand-written `installedPlugins` entry, neither of which is how a plugin loads. Installed
from a marketplace the same hooks fire and deliver on the first try.

**A manifest at `.github/plugin.json` is inert, and that is why the registration is found
by convention.** Copilot recognises a manifest at `.plugin/plugin.json`, `plugin.json`,
`.github/plugin/plugin.json` or `.claude-plugin/plugin.json` -- the installer prints
exactly that list when it cannot find one. `vscode-memvara` ships `plugin/.github/
plugin.json`, which is on no such list, so its manifest is not read: a `hooks` key there
produced no receipt, and a `skills` key naming a non-default directory left the skill
unlisted. What ships works because `skills/` and `hooks.json` are the DEFAULT locations,
so the plugin's own manifest never has to be consulted. Two consequences, and the second
is the trap: `hooks.json` must sit at the plugin root, and if that manifest is ever moved
to a recognised path its `hooks` key REPLACES the convention rather than adding to it --
measured, a root `hooks.json` was ignored the moment a readable manifest declared one.

**PascalCase event names, because they change the payload.** Copilot fires either
spelling, and the spelling decides the field names: `userPromptSubmitted` delivers
`sessionId`/`toolName`/`transcriptPath`, while `UserPromptSubmit` delivers `session_id`/
`tool_name`/`transcript_path` -- Claude Code's names exactly, plus `hook_event_name`. So
registering Claude's four event names makes this host's `fields` map identical to Codex's
rather than a fifth vocabulary to keep correct.

**The reply envelope is FLAT.** `hookSpecificOutput.additionalContext` delivers nothing
here; a top-level `additionalContext` delivers. Isolated by emitting each shape alone --
the nested form returned `NO CANARY` and the flat form was read back verbatim. Shipping
Claude's shape unchanged would have produced a plugin that recalls nothing.

**Neither async nor detaching-by-default frees the turn, but detaching with the pipes
closed does.** `async: true` is accepted and NOT honoured: a hook declared async that
slept six seconds delayed the client's exit by six seconds. A child started with
`start_new_session=True` did not help either -- Copilot waited for it, exiting one second
after a 20-second child finished -- because the child inherited the hook's stdout and the
client reads that pipe to EOF. With stdout and stderr on `DEVNULL`, which is what
`run.py`'s `_detach` already does, the same 20-second child finished *nineteen seconds
after* the client had exited. So `supports_async=False` and `detach_capture=True`, and the
pipe redirection in `_detach` is load-bearing on this host rather than tidiness.

**Two collisions to know about, neither of which this record can fix.** Only ONE
`additionalContext` survives per event: three plugins registering `UserPromptSubmit` all
ran and exactly one block reached the session -- verified in the client's own
`events.jsonl`, not from the model's answer -- and the last-installed won. And the model
is told to be suspicious of what arrives this way: a probe's injected block landed inside
a `<system_reminder>` wrapper and one run answered that it had disregarded "an injected
instruction rather than a genuine preference". The `SessionStart` block is not carried
that way and was answered from directly, so the standing memories are the sturdier half of
recall on this client.
"""

from __future__ import annotations

from core.host import COPILOT_CLI, ApproveSpec, Host, TranscriptSpec

HOST = Host(
    id="copilot",
    #: Both measured set, to the plugin's install-cache directory. (`CLAUDE_PLUGIN_ROOT`
    #: is set too and is deliberately not named: two spellings are enough to survive a
    #: rename, and this host has its own.)
    plugin_root_env=("PLUGIN_ROOT", "COPILOT_PLUGIN_ROOT"),
    #: Claude Code's four names verbatim -- see the module docstring on why the casing is
    #: a payload decision. All four were seen to fire, `Stop` included, in one session
    #: that submitted a prompt, ran a tool and ended.
    events={
        "session_start": "SessionStart",
        "recall": "UserPromptSubmit",
        "capture": "Stop",
        "approve": "PreToolUse",
    },
    #: Measured from real payloads, and identical to Codex's map because the PascalCase
    #: events deliver Claude Code's field names. Leaner than Claude's: no turn_id, no
    #: model, no permission_mode, and `transcript_path` on `Stop` ALONE -- so nothing but
    #: capture can read the conversation back on this host.
    fields={
        "session": ("session_id",),
        "cwd": ("cwd",),
        "prompt": ("prompt",),
        "transcript_path": ("transcript_path",),
        "tool_name": ("tool_name",),
    },
    #: See the module docstring: the nested shape delivers nothing here. `_render_flat`
    #: already addresses whatever `context_key` names, so this is a record and not a
    #: branch.
    envelope="flat",
    context_key="additionalContext",
    #: Measured empty. `systemMessage` reaches neither the model nor the terminal, and no
    #: other reply field was observed to surface, so `~/.memvara/.hooks/` is this host's
    #: whole account of itself -- as on Codex and Cursor.
    status_key="",
    #: Nothing is clipped: a 16,384-byte block arrived whole, head, middle and tail nonces
    #: all present, with no configuration. Unlike Codex, which truncates middle-out above
    #: a default and needs its limit raised.
    context_token_cap=0,
    #: Not emitted. `additionalContextLimit` is Codex's mechanism for a cap this host does
    #: not appear to impose, and a key the client ignores is a key that goes stale
    #: unnoticed.
    context_limit_key=0,
    #: False because it was MEASURED false, not because the flag is missing. See the
    #: module docstring: an async hook holds the turn exactly as long as it runs.
    supports_async=False,
    detach_capture=True,
    #: `timeout` is the spelling `tools/generate.py` emits, and it is honoured here --
    #: proved the only way that proves anything, by declaring 3 seconds against an
    #: 8-second hook and watching the hook be killed. (`timeoutSec` is honoured too; both
    #: died.) An earlier run at 30 seconds proved nothing: both spellings survived, which
    #: only bounds the default.
    timeouts={"session_start": 20, "recall": 10, "capture": 120, "approve": 5},
    client_configs=("~/.copilot/settings.json", "~/.copilot/mcp-config.json"),
    config_format="json",
    #: A third transcript shape, and the one place this port is code rather than data --
    #: `lib/transcript.py` grew `_format_copilot_entry` for it. Copilot writes
    #: `{"type": "user.message", "data": {...}}` with the speaker in the type string, and
    #: keeps the typed prompt and the model-facing copy in two separate fields. The reader
    #: mines the typed one; see its docstring for why that matters more than it sounds.
    transcript=TranscriptSpec(format="copilot-events"),
    #: The transcript's OWN spellings, which are not the hook payload's: `PreToolUse` says
    #: `tool_name: "Bash"` while the session log says `toolName: "bash"`. This field is
    #: read by the transcript reader, so the log's spelling is the one that belongs here --
    #: and a record that carried the hook's spelling would drop every tool call from the
    #: mined turn while looking configured.
    #:
    #: Observed across two sessions: `bash`, `create`, `edit`, `view`. `view` is a read and
    #: is excluded exactly as Claude Code's `Read` is. The list grows as more are seen
    #: rather than being guessed ahead of them.
    tools=frozenset({"bash", "create", "edit"}),
    #: Empty, and for a reason no other host has. Copilot wraps the model-facing copy of a
    #: prompt in `<current_datetime>` and `<system_reminder>` blocks -- but it keeps the
    #: typed text in a separate field, and the reader mines that one. So the host's markup
    #: is never in the text this list would filter, and naming it here would be decoration.
    noise=(),
    skip_prefixes=("/",),
    machine_prompt_prefixes=(),
    #: Present on `Stop`, alongside `stop_reason`. Set when the event is a hook-driven
    #: continuation rather than a real end of turn; mining it would double-count the reply.
    reentry_field="stop_hook_active",
    #: **`<server>-<tool>`, with a hyphen** -- measured against a local stdio MCP server:
    #: `memory_recall` from a server configured as `memvara` reached the hook as
    #: `memvara-memory_recall`. Neither Claude's `mcp__server__tool` nor Cursor's bare
    #: name, so both the matcher and the separator are this host's own.
    #:
    #: The matcher is anchored by the client -- it compiles the pattern as `^(?:...)$` --
    #: so a bare `memvara` would match nothing at all. `.*memvara.*` rather than
    #: `memvara-.*` because the server name is the user's config key and they may rename
    #: it; what is stable is that the word appears.
    approve=ApproveSpec(
        matcher=".*memvara.*",
        separators=("-",),
        decision_key="permissionDecision",
        reason_key="permissionDecisionReason",
        allow="allow",
    ),
    #: Copilot's own headless mode, so a turn here is mined by the model this user already
    #: chose and pays for. See `COPILOT_CLI` in `core/host.py` -- its `--available-tools`
    #: argument is a safety guard rather than a preference, because `copilot -p` runs
    #: tools without being asked.
    extractor=COPILOT_CLI,
    extractor_label="copilot-hook",
    description=(
        "Memvara for GitHub Copilot CLI: the standing block at session start, recall on "
        "every prompt, auto-approve on tool use, and capture when a turn ends. Capture is "
        "declared SYNCHRONOUS on purpose -- an async hook holds the turn open on this "
        "client rather than being deferred -- and run.py forks it with its pipes closed, "
        "which is what actually releases the turn here. Nothing this plugin prints reaches "
        "the screen, so its account is ~/.memvara/.hooks/."
    ),
)
