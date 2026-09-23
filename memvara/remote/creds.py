"""Resolve the two things a remote client needs: a key, and somewhere to send it.

The constants come from `memvara.server.config` rather than being declared again. There
were already two copies of the default URL — `config.py` and `login.py` — and a third
would be the one that drifts.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

from ..server.config import CREDENTIALS_PATH, DEFAULT_SERVER_URL

__all__ = ["MissingCredential", "read_credentials_file", "resolve", "CREDENTIALS_PATH"]


class MissingCredential(RuntimeError):
    """No api key was passed, exported, or written by `memvara-mcp login`."""


def _clean(value: str | None) -> str | None:
    """A blank or whitespace-only environment variable means unset, not empty. An
    exported `MEMVARA_SERVER_URL=` would otherwise be a base url of `""`."""
    return value.strip() if value and value.strip() else None


def read_credentials_file(path: str | os.PathLike[str]) -> dict[str, str]:
    """What a login wrote to one credentials file, or an empty mapping.

    Every way a file can fail to be a usable credential — missing, unreadable, not JSON,
    holding no `api_key` — comes back as `{}` rather than an exception, because each
    caller does the same thing with all of them: say "not signed in" and name the command
    that fixes it. Returning `{}` for a file that exists but holds no key is deliberate:
    a credential without a key is not a partial credential, it is not one.

    One function rather than one per caller. `resolve()` reads the default file,
    `memvara whoami` and `memvara logout` read whichever file they were given, and
    `demo/harness.py`'s hosted arms read the demo's own — and three readers of one
    format are three chances to disagree about what an unreadable file means.

    Only the string fields a login writes are returned: `api_key`, `project` and
    `server_url`. Anything else in the file is somebody else's and is left alone.
    """
    try:
        body = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(body, dict):
        return {}
    found: dict[str, str] = {}
    for name in ("api_key", "project", "server_url"):
        raw = body.get(name)
        value = _clean(raw) if isinstance(raw, str) else None
        if value is not None:
            found[name] = value
    return found if "api_key" in found else {}


def _from_file() -> str | None:
    """The key `memvara login` wrote to the default file, or None."""
    return read_credentials_file(CREDENTIALS_PATH).get("api_key")


def resolve(api_key: str | None, base_url: str | None,
            env: Mapping[str, str] | None = None) -> tuple[str, str]:
    """`(api_key, base_url)` for a remote client, or raise naming how to get one."""
    environ = os.environ if env is None else env
    key = _clean(api_key) or _clean(environ.get("MEMVARA_API_KEY")) or _from_file()
    if key is None:
        raise MissingCredential(
            "No memvara api key. Pass Memvara(api_key=...), export MEMVARA_API_KEY, or "
            f'run "memvara-mcp login" to write one to {CREDENTIALS_PATH}.')
    url = (_clean(base_url) or _clean(environ.get("MEMVARA_SERVER_URL"))
           or DEFAULT_SERVER_URL)
    return key, url.rstrip("/")
