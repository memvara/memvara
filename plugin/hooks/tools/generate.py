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


def registration(host) -> bytes:
    """The bytes of `host`'s registration file.

    Hooks are emitted in `core.host.HOOKS` order, which is the order they run in a
    session. A canonical name absent from `host.events` is a hook that client has no
    event for, and it is skipped rather than registered against a guessed event name.
    """
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
    tree = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(tree, "hooks.json")
    with open(out, "wb") as handle:
        handle.write(registration(host))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
