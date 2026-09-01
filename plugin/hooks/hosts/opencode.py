"""OpenCode, as a `Host` record.

Every value here was measured against opencode 1.18.20 on 2026-08-31 with a throwaway
plugin, not read off a docs page. Where a value could not be measured it is the spelling
that means "absent" rather than a hopeful default, because an absent key is the only one
that cannot be mistaken for a working one.

**This host does not run shell hooks at all**, which is the whole reason `js/` exists.
OpenCode's plugin API is in-process JavaScript: a module exporting async functions that
mutate typed output objects. There is no stdin, no stdout, and no `additionalContext`
anywhere in it. `js/opencode.mjs` is the module OpenCode loads; it frames each call as
the JSON payload `run.py` already reads, spawns the same Python, and applies the flat
reply by mutating OpenCode's objects. So the four bodies port unchanged and the host
difference stays data, exactly as it does for a shell host.

Three measurements shaped the mapping and are worth keeping next to it:

* A hook that is `await`ed **stalls the turn** -- 8.016s measured between a handler's
  entry and exit with an 8s sleep in it. Not awaiting exits in 1ms and the detached work
  still finished 8.1s later, because an OpenCode plugin lives in the persistent server
  process. That is why `capture` is mapped to an event the shim does not await, and why
  `supports_async` is True here where Codex, whose async hooks die with the process,
  would have to say otherwise.
* A part pushed into `output.parts` **must carry `id`, `sessionID` and `messageID`**.
  Pushing `{type, text}` alone fails server-side schema validation and kills the entire
  turn with an opaque `UnknownError: Unexpected server error`; the real cause appears only
  in `~/.local/share/opencode/log/opencode.log` as `invalid user part before save`. The
  shim builds the full part; nothing here can express that, which is why the shim is code
  and not another record field.
* Registration is by **directory convention**. `.opencode/plugin/*.js` is auto-loaded
  whether or not the config's `plugin` array names it, so emptying that array does not
  deregister anything -- only removing the file does.
"""

from __future__ import annotations

from core.host import (CODEX_CLI, OPENCODE_CLI, ApproveSpec, Host,
                       TranscriptSpec)

HOST = Host(
    id="opencode",
    #: No plugin-root environment variable exists on this host. The module resolves its
    #: own directory with `import.meta.dirname` and passes the interpreter an absolute
    #: path, so nothing here needs expanding and an empty tuple is the honest value.
    plugin_root_env=(),
    #: Canonical hook -> the OpenCode hook the shim attaches it to.
    #:
    #: `session_start` and `recall` share one OpenCode hook because OpenCode has no
    #: session-start event that can inject: every hook that fires once per session is
    #: `void` and returns nothing. `chat.message` is the only per-message hook with a
    #: mutable output, so the shim runs `session_start` on the first message of a session
    #: and `recall` on every one -- which is the same cadence Claude Code produces from
    #: two separate events, and is possible only because the plugin process persists.
    #:
    #: `capture` rides the generic event stream filtered to `session.idle`. There is no
    #: turn-end hook in the plugin interface at all; `session.idle` is the event that
    #: means the turn stopped.
    events={
        "session_start": "chat.message",
        "recall": "chat.message",
        "capture": "session.idle",
        "approve": "permission.ask",
    },
    #: The shim frames the payload, so these keys are ours rather than the host's. They
    #: are spelled exactly as Claude Code spells them on purpose: one wire vocabulary
    #: between `js/` and `run.py` means a reader comparing two hosts' records is looking
    #: at a real difference when the values differ, not at a shim's private taste.
    fields={
        "session": ("session_id",),
        "cwd": ("cwd",),
        "prompt": ("prompt",),
        "transcript_path": ("transcript_path",),
        "tool_name": ("tool_name",),
    },
    #: Read by `js/shim.mjs`, never by OpenCode. See `envelope._render_flat`.
    envelope="flat",
    context_key="additionalContext",
    #: OpenCode shows a plugin no operator-visible line. Its hooks return data structures
    #: rather than printing, and nothing in the plugin interface renders a status string
    #: to the person at the terminal. Empty here makes the renderer drop that half of a
    #: reply instead of addressing a key nobody reads -- so on this host the account of
    #: what a hook did is its log file, and only its log file.
    status_key="",
    #: Uncapped. Injected parts are ordinary message content here rather than a separate
    #: context channel, so the only budget is `recall.BUDGET`'s.
    context_token_cap=0,
    #: Measured: a detached promise outlives the turn. See the module docstring.
    supports_async=True,
    #: The shim simply does not await capture, so nothing here has to fork.
    detach_capture=False,
    #: Injected parts are ordinary message content, with no cap to declare.
    context_limit_key=0,
    #: Enforced by the shim, which kills the child, because OpenCode publishes no hook
    #: timeout of its own -- an unbounded handler would simply hold the turn open.
    timeouts={"session_start": 20, "recall": 10, "capture": 120, "approve": 5},
    client_configs=("~/.config/opencode/opencode.json",
                    "~/.config/opencode/opencode.jsonc"),
    config_format="json",
    #: OpenCode hands a hook no transcript path, so the shim materialises one: it reads
    #: the session's messages through the plugin's own `client` and writes them as the
    #: same JSONL entries `lib.transcript` already parses. `TranscriptSpec` is present
    #: rather than None because capture CAN run here -- the file just has to be made.
    transcript=TranscriptSpec(format="jsonl"),
    #: The tools whose use is evidence a turn did something, in OpenCode's spelling.
    tools=frozenset({"edit", "write", "bash", "patch"}),
    #: OpenCode wraps no markup of its own around user text. Ours are host-neutral and
    #: live in `lib.transcript.RECALL_MARKERS`.
    noise=(),
    #: A leading slash is a command here as it is everywhere; `!` and `#` are Claude
    #: Code's own and are not OpenCode's, so they are not carried across.
    skip_prefixes=("/",),
    #: OpenCode has no background-task or cross-session prompt wrapper to skip.
    machine_prompt_prefixes=(),
    #: `session.idle` carries no re-entry flag: it fires when the turn stops, and a
    #: continuation is a new turn rather than a flagged repeat of this one.
    reentry_field="",
    #: `permission.ask` gives a mutable `output.status` of "ask" | "deny" | "allow" --
    #: a cleaner permission surface than any other host measured, because the verdict is
    #: one field rather than a decision plus a reason. There is nowhere to put a reason,
    #: so `reason_key` addresses a key the shim drops.
    #:
    #: **The only mapping here not confirmed by a receipt.** No permission prompt fired
    #: during the spike, so which input field carries the tool name is read off the type
    #: definitions rather than measured. `js/opencode.mjs` logs the keys it is handed on
    #: the first call so the shape becomes known from a real invocation, and if the guess
    #: is wrong the match simply misses: nothing is auto-approved and the user is asked,
    #: which is what happens today anyway.
    approve=ApproveSpec(
        matcher="memvara",
        separators=("__", "_"),
        decision_key="status",
        reason_key="reason",
        allow="allow",
    ),
    #: OpenCode's own CLI, so a turn is mined by the model this user already configured
    #: rather than by a different vendor's product. `claude -p` remains the second rung.
    extractor=OPENCODE_CLI,
    #: Distinct from Claude Code's so `memory_why` can tell a user which client wrote a
    #: claim. Frozen from here on for the same reason the Claude one is: it is stored.
    extractor_label="opencode-hook",
    description=(
        "Memvara for OpenCode: recall on every message, capture when the session goes "
        "idle. OpenCode loads JavaScript rather than running shell hooks, so index.mjs "
        "frames each call and spawns run.py, which binds hosts/opencode.py before "
        "dispatching. Capture is not awaited, so a 12-14s extraction never holds the "
        "turn open; nothing this plugin does is visible in the OpenCode UI, so its "
        "account is ~/.memvara/.hooks/capture.log."
    ),
)
