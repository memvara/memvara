"""Cursor, as a `Host` record. A REDUCED port, and the reduction is the headline.

Measured against cursor-agent 2026.08.25-3e8eec8 on 2026-09-01 with a throwaway probe
registering every event Cursor's own bundle names.

**There is no per-prompt recall on this host, so `recall` is absent from `events`.**
`beforeSubmitPrompt` is the event that would carry it, and it never fires: not on the
first message, not on a `--continue` follow-up, and not from user scope or project scope.
Cursor's source declares it and its editor may drive it over a separate RPC path, but the
CLI does not, and a record that mapped `recall` to it would produce a hook that installs,
registers and never runs. `run.py` skips a canonical hook the record has no event for and
says so in the log, which is the honest shape for a host that cannot do something.

What DOES work, all confirmed by delivery -- the model read the injected marker back:

* `sessionStart` carries the standing block.
* `preToolUse` auto-approves, and can carry context too.
* `sessionEnd` is the only turn-ish event that fires with a transcript, so capture runs
  ONCE at the end of a session rather than once a turn. `stop` is declared by Cursor and
  never fired in any probe.

**Nothing this plugin says reaches the person.** A reply's `systemMessage` was emitted
alongside the context and never appeared; so were `message` and `userMessage`. Only
`additional_context` was delivered, so `status_key` is empty and `~/.memvara/.hooks/` is
the whole account of itself this host has.

**A carrier cap that DROPS rather than truncates.** Over 10,000 characters the tool-event
carrier discards the whole block and logs `additional_context exceeded max size; dropping
carrier` -- measured, `preToolUse` delivered at 9,000 chars and returned neither nonce at
20,000. `sessionStart` is not carried that way and passed 20,000 intact, which is why the
standing block is safe on the event that actually carries it.
"""

from __future__ import annotations

from core.host import CURSOR_CLI, ApproveSpec, Host, TranscriptSpec

HOST = Host(
    id="cursor",
    #: Both are set, and only these two expand: the source keeps an explicit allowlist of
    #: the variable names it will substitute in a hook command.
    plugin_root_env=("CURSOR_PLUGIN_ROOT", "CLAUDE_PLUGIN_ROOT"),
    #: `recall` is absent DELIBERATELY. See the module docstring: the event that would
    #: carry it does not fire on this client.
    events={
        "session_start": "sessionStart",
        "capture": "sessionEnd",
        "approve": "preToolUse",
    },
    #: Cursor's payloads are the richest measured -- conversation_id, cursor_version,
    #: generation_id, is_background_agent, model, user_email, workspace_roots -- and it is
    #: the only host that hands a hook the user's email address.
    fields={
        "session": ("session_id",),
        "cwd": ("workspace_roots", "cwd"),
        "prompt": ("prompt",),
        "transcript_path": ("transcript_path",),
        "tool_name": ("tool_name",),
    },
    #: Flat, with a snake_case key. No new renderer: `_render_flat` addresses whatever
    #: `context_key` names, so this host is a record and not a third branch.
    envelope="flat",
    context_key="additional_context",
    #: Measured empty. See the module docstring.
    status_key="",
    #: The 10,000-char carrier cap belongs to the tool events, and the standing block does
    #: not ride one -- `sessionStart` passed 20,000 chars whole. Left uncapped so nothing
    #: is thrown away that the host would have carried.
    context_token_cap=0,
    context_limit_key=0,
    #: `sessionEnd` fires once and its hook is awaited, so a 12-14s extraction would hold
    #: the end of the session. Forked for the same reason Codex is.
    supports_async=False,
    detach_capture=True,
    timeouts={"session_start": 20, "capture": 120, "approve": 5},
    client_configs=("~/.cursor/hooks.json", "~/.cursor/cli-config.json"),
    config_format="json",
    #: Claude Code's `message.content` blocks, with the speaker under `role` rather than
    #: `type`. One field, not a reader.
    transcript=TranscriptSpec(format="jsonl", role_key="role"),
    #: Cursor names its shell tool in `tool_name` on preToolUse; the list grows as more are
    #: seen rather than being guessed ahead of them.
    tools=frozenset({"Shell", "Read", "Edit", "Write"}),
    #: Cursor wraps every user message in `<user_query>` and prefixes a `<timestamp>` line.
    #: Those are NOT listed as noise on purpose: `_clean` drops a whole block that contains
    #: a marker, so naming the tag that wraps every prompt would discard every prompt.
    noise=(),
    skip_prefixes=("/",),
    machine_prompt_prefixes=(),
    #: `sessionEnd` carries no re-entry flag; it fires once when the session ends.
    reentry_field="",
    approve=ApproveSpec(
        matcher="memvara",
        separators=("__", "_"),
        decision_key="permission",
        reason_key="reason",
        allow="allow",
    ),
    #: Cursor's own CLI, so a turn here is mined by the model this user already chose and
    #: pays for. `claude -p` stays the chain's second rung for anyone who has it, and a
    #: machine with neither gets a logged reason and an alert rather than silence.
    extractor=CURSOR_CLI,
    extractor_label="cursor-hook",
    description=(
        "Memvara for Cursor: the standing block at session start, auto-approve on tool "
        "use, and capture once when the session ends. There is no per-prompt recall on "
        "this client -- the event that would carry it does not fire -- so memory here is "
        "what the session opened with plus whatever the model asks for through the MCP "
        "tools. Nothing this plugin prints reaches the screen; its account is "
        "~/.memvara/.hooks/."
    ),
)
