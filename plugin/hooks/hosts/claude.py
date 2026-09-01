"""Claude Code, as a `Host` record.

Every value here is the literal the hook bodies used to carry inline. Nothing was
tidied on the way across, and one of them must not be tidied later either:
`extractor_label` is written into users' stores and rendered back by `memory_why`, so
changing the string re-labels history that was recorded under the old one.
"""

from __future__ import annotations

# Absolute, not relative: every entry point puts `plugin/hooks/` on `sys.path` and
# imports both `core` and `hosts` as top-level packages from there.
from core.host import CLAUDE_CLI, ApproveSpec, Host, TranscriptSpec

HOST = Host(
    id="claude",
    plugin_root_env=("CLAUDE_PLUGIN_ROOT",),
    #: Canonical hook name -> the event this client fires. A canonical name absent from
    #: this mapping is a hook the host has no event for; Claude Code has all four.
    events={
        "session_start": "SessionStart",
        "recall": "UserPromptSubmit",
        "capture": "Stop",
        "approve": "PreToolUse",
    },
    #: `Event` field -> the stdin keys it may arrive under, first match wins. A tuple
    #: rather than a string so a client that renames a key can be followed without a
    #: release that breaks everyone still on the old one.
    fields={
        "session": ("session_id",),
        "cwd": ("cwd",),
        "prompt": ("prompt",),
        "transcript_path": ("transcript_path",),
        "tool_name": ("tool_name",),
    },
    #: Replies are `{"systemMessage": ..., "hookSpecificOutput": {"hookEventName": ...}}`.
    envelope="nested",
    #: The only field on these events that reaches the model, and the only one that
    #: reaches the person at the terminal. A host with "" for either has no such channel,
    #: and the renderer drops that half of the reply rather than inventing a key.
    context_key="additionalContext",
    status_key="systemMessage",
    #: Uncapped: this client imposes no ceiling of its own on injected context, so the
    #: only budget is the one `recall.BUDGET` sets for cost reasons.
    context_token_cap=0,
    #: Claude Code honours `async: true` on Stop, so capture never blocks and never
    #: needs to fork. See the note on both fields in `core/host.py`.
    supports_async=True,
    detach_capture=False,
    #: This client imposes no ceiling of its own, so nothing is declared to it.
    context_limit_key=0,
    timeouts={"session_start": 20, "recall": 10, "capture": 120, "approve": 5},
    client_configs=("~/.claude.json", "~/.claude/settings.json"),
    config_format="json",
    transcript=TranscriptSpec(format="jsonl"),
    #: The tools whose use is evidence that a turn did something. See
    #: `lib.transcript.INCLUDE_TOOLS`, which reads this.
    tools=frozenset({"Edit", "Write", "Bash", "NotebookEdit"}),
    #: Markup this client wraps around text that is not conversation. Only the host's own
    #: tags belong here: the markers this plugin injects are in `transcript.RECALL_MARKERS`
    #: and are the same on every host, because mining our own output back in is a bug
    #: everywhere.
    noise=("<command-message>", "<command-name>", "<system-reminder>",
           "<local-command-stdout>"),
    #: Prompts that are not questions to the model: a slash command, a bash escape, a
    #: comment. Silence is right for these -- the user typed a command and is not waiting
    #: on memory. Every client spells its own command prefixes, which is the whole reason
    #: these are here rather than in `recall.py`; see that file for why they stay separate
    #: from `machine_prompt_prefixes` below.
    skip_prefixes=("/", "!", "#"),
    #: A finished background task and a message from another session arrive through the
    #: prompt event wrapped in these. Answering one spends a retrieval query on a task id.
    machine_prompt_prefixes=("<task-notification", "<cross-session-message"),
    #: Set when the Stop event is a hook-triggered continuation rather than a real end of
    #: turn. Mining it would double-count the reply.
    reentry_field="stop_hook_active",
    approve=ApproveSpec(
        matcher="mcp__.*memvara.*",
        #: How a namespaced tool name splits into its leaf. `mcp__memvara__memory_search`
        #: and `mcp__plugin_memvara_memvara__memory_search` both end in the leaf.
        separators=("__",),
        decision_key="permissionDecision",
        reason_key="permissionDecisionReason",
        allow="allow",
    ),
    #: The first rung of `lib.extract`'s chain, and on this host the same command as the
    #: second: Claude Code's own CLI is what a Claude Code user has. Named from `core/`
    #: rather than restated here so there is one spelling of that argv in the tree -- a
    #: second copy is how the recursion guard's `--settings` flag would come to be present
    #: in the record and absent from the command actually run. `lib.extract` drops the
    #: duplicate rung, so this host tries one CLI and not the same one twice.
    extractor=CLAUDE_CLI,
    #: Written into every claim this plugin stores and rendered back by `memory_why`.
    #: Changing it for tidiness re-labels history.
    extractor_label="claude-code-hook",
    #: The prose `tools/generate.py` writes into this host's registration file. It lives
    #: on the record rather than in the generator because it is a fact about one client,
    #: and a per-host literal inside a shared tool is the shape that goes stale in the
    #: copy nobody is reading.
    description=(
        "Memvara: recall on every prompt, capture when a turn ends. Every command goes "
        "through run.py, which binds the host record in hosts/claude.py before "
        "dispatching -- the client's field names, reply keys and event names are data "
        "there rather than literals in the hook bodies. Capture runs async so a 12-14s "
        "extraction never holds the turn open; async hook output is discarded by the "
        "client, so its record is ~/.memvara/.hooks/capture.log rather than a "
        "systemMessage."
    ),
)
