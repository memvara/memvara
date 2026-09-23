"""The on/off switches for optional hook features.

Every feature a user can turn off during `/memvara:setup` is read here. The switches live in
`~/.memvara/settings.json`, a flat JSON object of `feature_name: true|false`. A missing key
means the feature's default, which is on for every feature the hooks know about today.

An environment variable `MEMVARA_FEATURE_<NAME>=0|1` overrides the file, so a test or a CI
run can pin a value without writing to the user's home directory. The library's MCP server
reads the same variable names in `ServerConfig.from_env`, so one variable means the same
thing on both sides.

The file is read on every call rather than once per process. It is a few bytes, a hook
process lives for one event, and a switch changed during a session should take effect on
the next prompt rather than after a restart nobody knows to do.
"""

from __future__ import annotations

import json
import os
import os.path

#: Written by `/memvara:setup` in the plugin repository. Beside the credentials and the hook
#: state, not inside the plugin, which is replaced wholesale on every update.
SETTINGS = os.path.join(os.path.expanduser("~"), ".memvara", "settings.json")

#: What an override may say. Anything else is ignored and the file decides, because a typo
#: in an environment variable should not silently flip a feature the file set.
_ON = frozenset({"1", "true", "on", "yes"})
_OFF = frozenset({"0", "false", "off", "no"})


def enabled(name: str) -> bool:
    """Whether the feature `name` is on. Never raises; an unreadable setting means on.

    On is the safe direction for every feature this reads: each one is additive, and the
    failure to avoid is a feature that stopped working because a file could not be parsed.
    """
    raw = os.environ.get(f"MEMVARA_FEATURE_{name.upper()}")
    if raw is not None:
        value = raw.strip().lower()
        if value in _ON:
            return True
        if value in _OFF:
            return False
    try:
        with open(SETTINGS, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return True
    if not isinstance(data, dict):
        return True
    value = data.get(name)
    # Only a real boolean counts. `/memvara:setup` writes true or false, and a string such
    # as "no" is more likely a hand edit that went wrong than a decision.
    return value if isinstance(value, bool) else True
