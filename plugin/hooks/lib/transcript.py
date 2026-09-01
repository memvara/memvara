"""Turn a Claude Code JSONL transcript into the text capture should mine.

The Stop hook used to keep only `type == user` string messages. That is what the
user *said*. SuperMemory's capture is useful because it also keeps what the
agent *did* — assistant prose and a compact record of Edit/Write/Bash — so a
later prompt can recall "we put the store in plugin/" without anyone having
typed that as a preference.

Thinking blocks, Memvara's own recall injection, and memory_* tool calls are
dropped: they are the plumbing of this plugin, not facts about the project.
"""

from __future__ import annotations

import json
from typing import Any

from core.host import active

#: The client whose transcript this module reads. Resolved once, at import: `run.py`
#: binds the host before importing any hook body, and a body is what pulls this in.
_HOST = active()

#: Tools whose *use* is a durable event (a file changed, a command ran).
INCLUDE_TOOLS = _HOST.tools

#: Prefixes of tool names that are this plugin talking to itself.
SKIP_TOOL_PREFIXES = ("mcp__",)

MAX_TOOL_ARG = 120
MAX_TOOL_RESULT = 240

#: The markers that name a block *this plugin injected into the turn*. It is the only
#: record of what was put in front of the model before it replied, which is what
#: `injected_memories` reads it back for: a memory the model was shown and then restated is
#: not a new observation, and mining it writes the store's own output back into the store.
#:
#: Ours, so the same on every client: they are the headers this plugin writes. Only the
#: host's own markup varies, and that lives in `Host.noise`.
RECALL_MARKERS = (
    "Recalled from Memvara",
    "Memvara — what is already known about this user",
    "Memvara — how this user wants work done",
)

#: Everything dropped whole rather than mined: the host's own markup, plus the blocks this
#: plugin injected itself.
#:
#: The SessionStart headers in `RECALL_MARKERS` matter more than they look. Capture mines
#: the turn it just watched; SessionStart injects a block of already-stored memories into
#: the very first turn of a session. Without these markers that block is read back as
#: conversation, re-extracted, and written again under whatever predicate the model picks
#: this time -- a feedback loop that manufactures duplicates of facts already in the store,
#: and one that gets worse every session rather than settling.
#:
#: It has never fired, because SessionStart produced no output at all on a hosted install
#: until 0.1.4. Fixing that hook without adding those lines in the same commit would have
#: turned a dead hook into an actively harmful one.
NOISE = _HOST.noise + RECALL_MARKERS + ("Memvara scope:",)


def _injected_lines(text: str) -> list[str]:
    """The memory bullets out of an injected block, or nothing if this is not one."""
    if not any(marker in text for marker in RECALL_MARKERS):
        return []
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("- ") and len(line) > 4:
            out.append(line[2:].strip())
    return out


def _entry_injected(entry: dict) -> list[str]:
    """Injected memory bullets carried by one transcript entry."""
    message = entry.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if isinstance(content, str):
        return _injected_lines(content)
    if not isinstance(content, list):
        return []
    out: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            out.extend(_injected_lines(str(block.get("text") or "")))
    return out


#: How a formatted line announces who is speaking. `format_user` and `format_assistant`
#: above write exactly these, and `user_lines` is the only reader — so the two have to be
#: changed together, and this tuple is where that is visible.
SPEAKERS = ("User: ", "Claude: ", "Claude used ", "Tool result (")


def user_lines(turn: str) -> str:
    """Only the parts of `turn` the person actually typed.

    **A prefix marks the start of a block, not every line of one.** `format_user` writes one
    `User: ` for a whole message, so a prompt somebody typed across three lines arrives as
    one prefixed line and two bare ones. Filtering on the prefix therefore recovered the
    first line and silently dropped the rest, and the caller that matters — the check asking
    whether a fact is supported by what the user said — then failed for every multi-line
    prompt and dropped the user's own words as an echo of a note they had been shown.

    So a block runs from its prefix to the next one. Anything before the first prefix is not
    attributable to anyone and is left out.
    """
    out: list[str] = []
    keeping = False
    for line in turn.splitlines():
        start = next((p for p in SPEAKERS if line.startswith(p)), None)
        if start is not None:
            keeping = start == "User: "
            if keeping:
                out.append(line[len("User: "):])
        elif keeping:
            out.append(line)
    return "\n".join(out)


def _clean(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    if any(marker in text for marker in NOISE):
        return ""
    return text


def _truncate(text: str, n: int) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= n:
        return text
    return text[: n - 3] + "..."


def _skip_tool(name: str) -> bool:
    if name in INCLUDE_TOOLS:
        return False
    return name.startswith(SKIP_TOOL_PREFIXES) or name not in INCLUDE_TOOLS


def _tool_args(inp: Any) -> str:
    if not isinstance(inp, dict):
        return _truncate(str(inp), MAX_TOOL_ARG)
    parts = []
    for key in ("file_path", "command", "path", "notebook_path"):
        val = inp.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(f"{key}={_truncate(val, MAX_TOOL_ARG)}")
    return " ".join(parts)


def format_user(message: Any) -> list[str]:
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    out: list[str] = []
    if isinstance(content, str):
        cleaned = _clean(content)
        if cleaned:
            out.append(f"User: {cleaned}")
        return out
    if not isinstance(content, list):
        return []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            cleaned = _clean(str(block.get("text") or ""))
            if cleaned:
                out.append(f"User: {cleaned}")
        elif block.get("type") == "tool_result":
            name = str(block.get("name") or "tool")
            if _skip_tool(name):
                continue
            raw = block.get("content")
            if isinstance(raw, str):
                snippet = _truncate(_clean(raw), MAX_TOOL_RESULT)
            else:
                snippet = ""
            status = "error" if block.get("is_error") else "ok"
            if snippet:
                out.append(f"Tool result ({name}, {status}): {snippet}")
    return out


def format_assistant(message: Any) -> list[str]:
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    out: list[str] = []
    if isinstance(content, str):
        cleaned = _clean(content)
        if cleaned:
            out.append(f"Claude: {cleaned}")
        return out
    if not isinstance(content, list):
        return []
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "thinking":
            continue
        if kind == "text":
            cleaned = _clean(str(block.get("text") or ""))
            if cleaned:
                out.append(f"Claude: {cleaned}")
        elif kind == "tool_use":
            name = str(block.get("name") or "")
            if _skip_tool(name):
                continue
            args = _tool_args(block.get("input"))
            line = f"Claude used {name}"
            if args:
                line += f" {args}"
            out.append(line)
    return out


def _format_codex_entry(entry: dict) -> list[str]:
    """One line of a Codex rollout, which is not the shape the entries above are.

    Codex writes `{"type": "response_item", "payload": {"type": "message", "role": ...,
    "content": [{"type": "input_text" | "output_text", "text": ...}]}}` -- no `message`
    key, roles nested a level down, and its own content-block types. Read with the reader
    above it produced nothing at all: every line formatted to `[]`, so capture ran on
    every turn, found an empty string, and logged `no turn to mine` forever. That is the
    shape of failure this package exists to avoid, and it is why `TranscriptSpec.format`
    stopped being decorative.

    `developer` is skipped rather than mined, and that is the important line here. It is
    where the host puts its own instructions AND where this plugin's own injected recall
    arrives -- so on this client the blocks we wrote are excluded by role, before the
    marker matching in `NOISE` is ever consulted. Mining our own output back in
    manufactures duplicates of facts already stored, and gets worse every session.
    """
    if entry.get("type") != "response_item":
        return []
    payload = entry.get("payload")
    if not isinstance(payload, dict):
        return []
    if payload.get("type") == "custom_tool_call":
        return _codex_tool_call(payload)
    if payload.get("type") != "message":
        return []
    role = payload.get("role")
    if role not in ("user", "assistant"):
        return []
    blocks = payload.get("content")
    if not isinstance(blocks, list):
        return []
    speaker = "User" if role == "user" else "Claude"
    out: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") not in ("input_text", "output_text", "text"):
            continue
        cleaned = _clean(str(block.get("text") or ""))
        if cleaned:
            out.append(f"{speaker}: {cleaned}")
    return out


def _codex_tool_call(payload: dict) -> list[str]:
    """What a Codex turn DID, not only what it said about it.

    Measured shape: `{"type": "custom_tool_call", "name": "exec", "input": "..."}`. Read
    without this the mined turn was the assistant's prose alone, while the same turn on
    Claude Code yields `Claude used Edit ...` lines -- so a fact grounded in the action
    rather than stated in the reply was kept on one host and lost on the other, with
    nothing saying capture mined less here.

    `_skip_tool` is applied, which is what makes `Host.tools` a live setting on this host
    rather than a field that reads as configuration and is never consulted.

    The matching `custom_tool_call_output` entry is deliberately NOT emitted: it carries a
    `call_id` and no tool name, so it can be neither labelled nor filtered by the same
    allowlist, and inventing a label for it is the kind of guess this package refuses. The
    call already says what was done.
    """
    name = str(payload.get("name") or "")
    if not name or _skip_tool(name):
        return []
    args = _truncate(_clean(str(payload.get("input") or "")), MAX_TOOL_RESULT)
    return [f"Claude used {name} {args}".rstrip()]


def format_entry(entry: dict) -> list[str]:
    if _HOST.transcript is not None and _HOST.transcript.format == "codex-rollout":
        return _format_codex_entry(entry)
    # Asked of the record, not hardcoded. Cursor writes the same `message.content` blocks
    # as Claude Code but names the speaker under `role`, so reading `type` there found no
    # user entry, no boundary, and an empty turn on every capture -- the identical failure
    # the Codex reader was added for, one key over.
    kind = entry.get(_HOST.transcript.role_key if _HOST.transcript else "type")
    message = entry.get("message")
    if kind == "user":
        return format_user(message)
    if kind == "assistant":
        return format_assistant(message)
    return []


def last_turn_with_injections(raw: bytes) -> "tuple[str, list[str]]":
    """The exchange that just ended, and the memories this plugin injected into it: the last typed prompt and the reply to it.

    Both halves, because they carry different things and neither is enough on its own. A
    standing instruction is stated in the prompt — "always open a PR", "stop asking me
    about X" — while what was actually decided, and where it landed, is in the reply.

    Mining the reply alone was tried and does not work. It asks a model to find durable
    facts *about the user* in Claude's own words, and the model correctly answers that
    there are none: measured over one session, fifteen extractions in an hour returned an
    empty list every time while costing a full run each.

    The boundary is the last entry that formats to a `User:` line. Tool results are also
    entries of type `user`, so the naive boundary cuts the turn in half; and a prompt that
    survives the noise filter is a prompt somebody typed.
    """
    entries = []
    for line in raw.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict):
            entries.append(entry)

    start = None
    for index in range(len(entries) - 1, -1, -1):
        # Asked of `format_entry` alone, with no pre-filter on the entry's own `type`.
        # That filter said `type != "user"` -- Claude Code's spelling, and an entry shape
        # no other host uses. On a Codex rollout every entry is a `response_item`, so the
        # scan skipped all of them, `start` stayed None, and capture returned an empty
        # turn on every single turn while logging `no turn to mine`. The filter only ever
        # saved formatting a few entries during a scan that breaks at the first match, and
        # it cost the whole hook on the second host to use it.
        if any(line.startswith("User: ") for line in format_entry(entries[index])):
            start = index
            break
    if start is None:
        # No typed prompt in the window. Mining everything from here would re-mine turns
        # that were already handled when they happened.
        return "", []

    out: list[str] = []
    for entry in entries[start:]:
        out.extend(format_entry(entry))

    # Injections are gathered from the whole window rather than from the turn boundary,
    # and that is not laziness. A recall block is written *before* the prompt it answers,
    # and the SessionStart block sits at the top of the session -- so a scan that started
    # at the boundary would collect almost none of them and the echo filter would pass
    # everything. They are only ever used as a blocklist, so a wider net costs nothing.
    injected: list[str] = []
    for entry in entries:
        injected.extend(_entry_injected(entry))
    return "\n".join(out), injected


def last_turn(raw: bytes) -> str:
    """The mined text alone. See `last_turn_with_injections` for what it drops."""
    return last_turn_with_injections(raw)[0]


def span_from_bytes(raw: bytes) -> str:
    """Decode a JSONL slice (the bytes after the watermark) into mineable text."""
    lines_out: list[str] = []
    for line in raw.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if not isinstance(entry, dict):
            continue
        lines_out.extend(format_entry(entry))
    return "\n".join(lines_out)
