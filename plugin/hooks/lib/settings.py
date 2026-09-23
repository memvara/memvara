"""The on/off switches for optional hook features.

Every feature a user can turn off during `/memvara:setup` is read here. The switches live in
`~/.memvara/settings.json`, a flat JSON object of `feature_name: true|false`. A missing key
means the feature's default, which is `FEATURE_DEFAULTS` below: on for every feature except
`extraction_chunks`.

The file holds one entry that is not a switch. `/memvara:setup verify-key` records its test
call to the read-path model under the key `read_model`, and `lib.read_model` reads it back
through `stored()`.

An environment variable `MEMVARA_FEATURE_<NAME>=0|1` overrides the file, so a test or a CI
run can pin a value without writing to the user's home directory. The library's MCP server
reads the same variable names in `ServerConfig.from_env`, so one variable means the same
thing on both sides.

The file is read at most once per process. A hook process lives for one event, so a switch
changed during a session still takes effect on the next prompt. The long-lived recall daemon
does not read switches at all; the hook that spawns it does.
"""

from __future__ import annotations

import json
import os
import os.path

#: Written by `/memvara:setup` in the plugin repository. Beside the credentials and the hook
#: state, not inside the plugin, which is replaced wholesale on every update.
SETTINGS = os.path.join(os.path.expanduser("~"), ".memvara", "settings.json")

#: Every feature switch and its default, in the order `/memvara:setup` lists them. The
#: library's MCP server has the same mapping as `memvara.server.config.FEATURE_DEFAULTS` and
#: refuses a `MEMVARA_FEATURE_<NAME>` that is not in it. The hooks cannot import the library,
#: so this is a copy, and `tests/test_hook_project.py` fails when the two differ in a name,
#: in the order or in a default. The hooks read `project_scope`, `status_line`,
#: `recall_mark` and `query_rewrite`. The other names are listed so that `/memvara:setup`
#: can show every switch with its true default.
FEATURE_DEFAULTS = {
    "index_command": True,
    "research_agent": True,
    "project_scope": True,
    "status_line": True,
    "recall_mark": True,
    "profile": True,
    "forget_matching": True,
    "end_reason": True,
    "links": True,
    "documents": True,
    "retrieval_chunks": True,
    "extraction_chunks": False,
    "ingest_urls": True,
    "ingest_media": True,
    "query_rewrite": True,
    "synthesis": True,
}

#: Every feature name, in the order `FEATURE_DEFAULTS` lists them.
FEATURES = tuple(FEATURE_DEFAULTS)

#: What an override may say. Anything else is ignored and the file decides, because a typo
#: in an environment variable should not silently flip a feature the file set.
_ON = frozenset({"1", "true", "on", "yes"})
_OFF = frozenset({"0", "false", "off", "no"})

#: `(path, parsed file)` from the first read in this process. Keyed on the path so that a
#: caller pointing `SETTINGS` somewhere else, as the tests do, reads the new file.
_LOADED: "tuple[str, dict] | None" = None


def _file() -> dict:
    """The settings file as a dict, `{}` when it is missing or unreadable. Read once."""
    global _LOADED
    if _LOADED is None or _LOADED[0] != SETTINGS:
        try:
            with open(SETTINGS, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            data = {}
        _LOADED = (SETTINGS, data if isinstance(data, dict) else {})
    return _LOADED[1]


def reload() -> None:
    """Forget the file read earlier in this process, so the next read sees it as it is now.

    A hook reads the file once and never needs this. `/memvara:setup` does: it writes the
    file and then reports what the hooks will read, in the same process.
    """
    global _LOADED
    _LOADED = None


def stored(key: str) -> object:
    """The value the settings file holds under `key`, or `None`. Never raises."""
    return _file().get(key)


def enabled(name: str) -> bool:
    """Whether the feature `name` is on. Never raises.

    A missing, unreadable or non-boolean setting means the default in `FEATURE_DEFAULTS`.
    That default is on for every feature a hook reads, and on is the safe direction for
    those: each one is additive, and the failure to avoid is a feature that stopped working
    because a file could not be parsed.

    A name outside `FEATURES` raises `ValueError`. Every caller passes a fixed name, so this
    can only be a typo in the hooks' own code, and the tests exercise every caller.
    """
    if name not in FEATURES:
        raise ValueError(f"{name!r} is not a feature; the features are {', '.join(FEATURES)}")
    raw = os.environ.get(f"MEMVARA_FEATURE_{name.upper()}")
    if raw is not None:
        value = raw.strip().lower()
        if value in _ON:
            return True
        if value in _OFF:
            return False
    value = _file().get(name)
    # Only a real boolean counts. `/memvara:setup` writes true or false, and a string such
    # as "no" is more likely a hand edit that went wrong than a decision.
    return value if isinstance(value, bool) else FEATURE_DEFAULTS[name]
