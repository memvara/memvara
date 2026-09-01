"""Codex, as a `Host` record.

Every value was measured against codex-cli 0.151.0 with a throwaway plugin on 2026-08-31
and 2026-09-01. Codex is the closest of any host to Claude Code -- the same event names,
the same stdin field names, the same nested reply envelope -- and the three places it
differs are all places where copying Claude's record would have produced a plugin that
installs, logs success and does nothing.

**Hooks are trust-gated, and silently inert until trusted.** Three separate runs produced
zero receipts while Codex was visibly parsing the file: it printed warnings about clamping
the timeouts in it. Nothing said "blocked". The gate is `hooks.state."<id>".trusted_hash`
in `config.toml`, and `--dangerously-bypass-hook-trust` is the per-invocation override
that made them run. Installed and working are different states here, and the README has
to say so or a user sees nothing and has nowhere to look.

**Async is offered and does not work.** See the note on `supports_async` and
`detach_capture` in `core/host.py`: the registration accepts `async: true`, and the hook
then does not run at all. Declared synchronous it fires, and `run.py` forks so the turn is
not held.

**`additionalContext` is truncated middle-out above a default cap**, with a visible
`Warning: truncated output (original token count: N)`. Head and tail survive; the middle
vanishes. Measured intact at 8KB and cut at 12KB, so `session_start`'s 16,000-character
standing block would lose its middle by default. `additionalContextLimit` on the
registration raises it -- 32000 passed a 16,384-byte block whole, and a control at 500 cut
the same body -- so that number is declared here rather than left to a default that
quietly eats memories.
"""

from __future__ import annotations

from core.host import (CODEX_CLI, OPENCODE_CLI, ApproveSpec, Host,
                       TranscriptSpec)

HOST = Host(
    id="codex",
    #: Codex exports both. `PLUGIN_ROOT` first with `CLAUDE_PLUGIN_ROOT` as the fallback,
    #: which is the one measured to expand -- the probe read it and got the plugin's cache
    #: directory. Naming both is what `tools/generate.py`'s `${A:-${B}}` form is for.
    plugin_root_env=("PLUGIN_ROOT", "CLAUDE_PLUGIN_ROOT"),
    #: Claude Code's four event names verbatim; all four were seen to fire.
    events={
        "session_start": "SessionStart",
        "recall": "UserPromptSubmit",
        "capture": "Stop",
        "approve": "PreToolUse",
    },
    #: Also Claude Code's, verbatim and measured from real payloads: SessionStart carries
    #: cwd, hook_event_name, model, permission_mode, session_id, source, transcript_path;
    #: UserPromptSubmit adds prompt and turn_id; Stop adds last_assistant_message and
    #: stop_hook_active.
    fields={
        "session": ("session_id",),
        "cwd": ("cwd",),
        "prompt": ("prompt",),
        "transcript_path": ("transcript_path",),
        "tool_name": ("tool_name",),
    },
    #: `{"hookSpecificOutput": {"hookEventName": ..., "additionalContext": ...}}`, the
    #: same shape Claude Code reads. Verified by delivery on four events at once.
    envelope="nested",
    context_key="additionalContext",
    #: Empty, and measured rather than assumed: a reply's `systemMessage` reaches neither
    #: the model nor the person. Codex prints its own `hook: <Event>` lines and nothing
    #: from the reply, so a status line here would address a key nobody reads. On this
    #: host, as on OpenCode, `~/.memvara/.hooks/` is the whole account.
    status_key="",
    #: We do not clip: the client's own limit is declared below and honours the whole
    #: block, so clipping here would throw memories away that the host would have carried.
    context_token_cap=0,
    #: Declared to the client on every hook that can carry context. See `context_limit_key`
    #: in `tools/generate.py` for the measurements behind the number.
    context_limit_key=32000,
    #: False DELIBERATELY, and not because Codex lacks the flag. It has it and ignores it.
    supports_async=False,
    detach_capture=True,
    #: SessionEnd and Interrupt are clamped to 3s by the client, which warns when it does
    #: it. Neither is registered here, so nothing this plugin declares is clamped.
    timeouts={"session_start": 20, "recall": 10, "capture": 120, "approve": 5},
    client_configs=("~/.codex/config.toml",),
    #: The one client whose settings are TOML rather than JSON.
    config_format="toml",
    #: `~/.codex/sessions/<y>/<m>/<d>/rollout-*.jsonl`, present on Stop and real -- the
    #: probe read it back.
    transcript=TranscriptSpec(format="codex-rollout"),
    #: The tools whose use is evidence a turn did something, and CONSULTED -- the Codex
    #: transcript reader passes tool calls through `_skip_tool` like every other host.
    #:
    #: One name, because one is what has been observed: a shell command arrives as
    #: `custom_tool_call` with `name: "exec"`. An earlier draft listed Claude Code's four
    #: spellings here, which was wrong twice over -- they are not Codex's names, and
    #: nothing read the field on this host at all, so it looked like configuration and was
    #: decoration. A tool this list does not name is dropped from the mined turn, so the
    #: list grows as more are seen rather than being guessed ahead of them.
    tools=frozenset({"exec"}),
    #: Codex injects a plugin advertisement into the conversation as a USER message, so
    #: unlike its `developer` content it is not excluded by role and has to be named.
    noise=("<recommended_plugins>", "<skills_instructions>"),
    #: A leading slash is a command here as everywhere. `!` and `#` are Claude Code's own
    #: and are not carried across.
    skip_prefixes=("/",),
    machine_prompt_prefixes=(),
    #: Present on Stop, and set when the event is a hook-driven continuation rather than a
    #: real end of turn. Mining it would double-count the reply.
    reentry_field="stop_hook_active",
    #: `PreToolUseHookSpecificOutputWire` carries `permissionDecision` and
    #: `permissionDecisionReason`, read out of the schema the binary ships rather than
    #: guessed -- the same pair Claude Code uses.
    approve=ApproveSpec(
        matcher="mcp__.*memvara.*",
        separators=("__",),
        decision_key="permissionDecision",
        reason_key="permissionDecisionReason",
        allow="allow",
    ),
    #: Codex's own CLI, so a turn on Codex is mined by the model this user already
    #: chose and pays for -- not by a different vendor's product that has to be installed
    #: separately. `claude -p` stays as the chain's second rung for anyone who has it, and
    #: a machine with neither gets a logged reason and an alert rather than silence.
    extractor=CODEX_CLI,
    extractor_label="codex-hook",
    description=(
        "Memvara for Codex: recall on every prompt, capture when a turn ends. Every "
        "command goes through run.py, which binds hosts/codex.py before dispatching. "
        "Capture is declared SYNCHRONOUS on purpose -- an async hook does not run at all "
        "on this client -- and run.py forks it into its own session, so the turn is never "
        "held open. Hooks stay inert until trusted; nothing this plugin prints reaches "
        "the screen, so its account is ~/.memvara/.hooks/."
    ),
)
