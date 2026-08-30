"""stdin bytes in, an `Event`; a `Reply` out, stdout bytes. Both steered by the `Host`.

This is the only module that knows a host's wire format, and it is deliberately dull:
one lookup table each way. Everything interesting about a port lives in the record it is
handed, so a new host is a data change and this file does not move.

Two shapes it will not invent. A host whose `context_key` is empty has no channel for
per-turn context, and a host whose `status_key` is empty has no line the operator ever
sees -- so the renderer *drops* that half of a reply rather than guessing a key. A hook
that silently addresses a key the client does not read looks exactly like a hook that had
nothing to say, which is the failure this whole package is built around not repeating.

No `pathlib`, no `typing`: this runs on the per-prompt path, whose whole budget is ~30ms.
"""

from __future__ import annotations

import json

from core.host import Event, Reply  # noqa: F401  (Reply re-exported for the bodies)


def read_event(host, hook: str, raw) -> "Event":
    """One host's stdin payload, read through that host's field map.

    `raw` may be bytes, text, an already-decoded dict, or `None` to read stdin. Anything
    unreadable becomes an empty `Event` rather than an exception: a hook must never fail a
    prompt, and "no payload" and "a payload with nothing in it" call for the same silence.
    """
    if raw is None:
        import sys

        try:
            raw = sys.stdin.read()
        except (OSError, ValueError):
            raw = ""
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            raw = ""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            raw = {}
    data = raw if isinstance(raw, dict) else {}

    def pick(field: str) -> str:
        for key in host.fields.get(field, ()):
            if key in data:
                return str(data.get(key) or "")
        return ""

    return Event(
        hook=hook,
        session=pick("session"),
        cwd=pick("cwd"),
        prompt=pick("prompt"),
        transcript_path=pick("transcript_path"),
        tool_name=pick("tool_name"),
        reentrant=bool(data.get(host.reentry_field)) if host.reentry_field else False,
        raw=data,
    )


def render(host, reply: "Reply") -> str:
    """The bytes this host's client expects, or `""` when there is nothing to say.

    Empty rather than `{}`: an empty object is a reply, and a reply with no fields still
    reads as one to anyone tailing the transcript.
    """
    if host.envelope != "nested":
        # Every other envelope arrives with the host that needs it. Refusing loudly here
        # rather than falling back to this one is the point: a port that reaches this line
        # has declared a shape nobody has written, and rendering Claude's shape at it
        # would produce a reply the client reads as valid and ignores.
        raise ValueError(f"no renderer for envelope {host.envelope!r}")

    body: dict = {}
    if reply.status and host.status_key:
        body[host.status_key] = reply.status

    specific: dict = {}
    if reply.context and host.context_key:
        specific[host.context_key] = reply.context
    if reply.decision and host.approve is not None:
        specific[host.approve.decision_key] = reply.decision
        specific[host.approve.reason_key] = reply.reason

    if specific:
        event_name = host.events.get(reply.hook, "")
        body["hookSpecificOutput"] = {"hookEventName": event_name, **specific}

    return json.dumps(body) + "\n" if body else ""


def write(host, reply: "Reply") -> None:
    """Render and print, or print nothing at all."""
    import sys

    text = render(host, reply)
    if text:
        sys.stdout.write(text)
