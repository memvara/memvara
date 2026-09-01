"""Build one host's hook registration file from its `Host` record.

`hooks.json` is the only file under `hooks/` that a plugin repository does not vendor
byte for byte. It cannot be: seven repositories vendor this same tree and each one
registers a different client, so a canonical copy would be one repository's manifest
shipped to all of them. It is generated instead, from the record in `hosts/<id>.py`, and
the repository that ships it diffs the committed file against what this produces -- so a
hand edit, or a record that stops agreeing with the manifest built from it, fails there
rather than reaching a user's client.

    python3 plugin/hooks/tools/generate.py claude

Writes `hooks.json` beside the tree this file lives in. The output is deterministic and
the plugin repositories commit it, so an incidental change to the formatting here shows
up as a diff in every one of them at once.

`hooks.json` is hardcoded as the filename because it is what a Claude-Code-shaped plugin
registers, which is every host packaged so far. A client that wants a different file, or
a different format, needs its own writer here -- `Host.config_format` describes the
client's own settings files and is not an instruction to this module.
"""

from __future__ import annotations

import json
import os.path
import sys

# The tree root, not `tools/`: run as a script, `sys.path[0]` is this directory, and
# `core` is one level up. Same insert `run.py` does, for the same reason.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.host import HOOKS  # noqa: E402

#: The canonical hooks whose replies can carry context at all. `capture` is absent on
#: purpose: on Codex the Stop event has no HookSpecificOutput wire type, and emitting one
#: there made the hook FAIL outright rather than be ignored.
_CONTEXT_HOOKS = ("session_start", "recall", "approve")


def registration(host) -> bytes:
    """The bytes of `host`'s registration file.

    Hooks are emitted in `core.host.HOOKS` order, which is the order they run in a
    session. A canonical name absent from `host.events` is a hook that client has no
    event for, and it is skipped rather than registered against a guessed event name.
    """
    if not host.plugin_root_env:
        # A host with no plugin-root variable does not register shell commands at all.
        # OpenCode is the first: its hooks are in-process JavaScript, so its registration
        # is `js/opencode.mjs` plus a config entry naming it, and there is no path for
        # this function to interpolate. Refused here rather than left to fail on
        # `plugin_root_env[-1]`, because an IndexError names neither the host nor the
        # reason, and refused rather than defaulted because the file this would write --
        # a hooks.json full of shell commands -- installs cleanly on a host that never
        # reads it, which is the failure that looks like success.
        raise ValueError(
            f"{host.id} sets no plugin-root variable, so it does not register shell "
            "hooks; its registration is a JavaScript module and this writer does not "
            "build one")

    # Every name in the tuple, innermost last: `${A:-${B}}`. A host that sets more than
    # one -- Codex exports `PLUGIN_ROOT` and `CLAUDE_PLUGIN_ROOT` both -- can then name
    # them all and get a real fallback. Reading `[0]` and dropping the rest was the one
    # option that misleads: the field's type invites a second name and the build ignored
    # it silently.
    root = "${" + host.plugin_root_env[-1] + "}"
    for name in reversed(host.plugin_root_env[:-1]):
        root = "${" + name + ":-" + root + "}"
    events: "dict[str, list]" = {}
    for name in HOOKS:
        event = host.events.get(name)
        if event is None:
            continue
        command = {
            "type": "command",
            "command": f'python3 "{root}/hooks/run.py" {name} --host {host.id}',
        }
        # Only capture is async, and only where the host supports it: a 12-14s extraction
        # must not hold the turn open. The read hooks are synchronous by necessity --
        # their output is the whole point and an async hook's output is discarded.
        if name == "capture" and host.supports_async:
            command["async"] = True
        if name in host.timeouts:
            command["timeout"] = host.timeouts[name]
        if host.context_limit_key and name in _CONTEXT_HOOKS:
            # Declared per hook, and only on the hooks that can carry context. Codex
            # truncates `additionalContext` MIDDLE-OUT above a default measured between
            # 8KB (intact) and 12KB (cut, and it says so: "Warning: truncated output"),
            # which would silently drop the middle of a standing block. Raising the limit
            # is the client's own mechanism -- measured, 32000 let a 16,384-byte block
            # through whole, and the same body at 500 was cut. Emitted only where it
            # applies: the host warns "ignoring additionalContextLimit for <event>: this
            # event cannot emit additionalContext" and a key it ignores is a key that
            # goes stale unnoticed.
            command["additionalContextLimit"] = host.context_limit_key
        entry = {"hooks": [command]}
        if name == "approve":
            if host.approve is None:
                # Refused, not crashed, and for the same reason the empty-events case
                # below is refused: `Host.approve` is independently optional, so a record
                # can name the event and omit the spec, and `AttributeError: 'NoneType'
                # object has no attribute 'matcher'` names neither the record nor the
                # field that is missing from it.
                raise ValueError(
                    f"{host.id} registers an approve event but its record has no "
                    "ApproveSpec, so there is no matcher to build the registration from")
            entry = {"matcher": host.approve.matcher, **entry}
        events.setdefault(event, []).append(entry)
    body = {"description": host.description, "hooks": events}
    return (json.dumps(body, indent=2) + "\n").encode("utf-8")


def main(argv: "list[str]") -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    import importlib

    host = importlib.import_module(f"hosts.{argv[0]}").HOST
    if not host.events:
        # Refuse rather than write an empty manifest. A registration file with no hooks in
        # it installs cleanly and does nothing, which is the failure that looks like
        # success.
        print(f"{argv[0]} declares no hook events", file=sys.stderr)
        return 1
    # Built BEFORE the file is opened, and that order is the whole point. `open(..., "wb")`
    # truncates, so building inside the `with` meant any refusal in `registration` left
    # the shipped manifest at zero bytes -- measured: `generate.py opencode` took a
    # committed 1568-byte hooks.json to 0 and exited 1. A repository whose registration
    # file is empty installs cleanly and registers nothing.
    try:
        body = registration(host)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    tree = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(tree, "hooks.json")
    with open(out, "wb") as handle:
        handle.write(body)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
