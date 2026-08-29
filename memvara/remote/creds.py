"""Resolve the two things a remote client needs: a key, and somewhere to send it.

The constants come from `memvara.server.config` rather than being declared again. There
were already two copies of the default URL — `config.py` and `login.py` — and a third
would be the one that drifts.
"""
from __future__ import annotations

import json
import os
from typing import Mapping

from ..server.config import CREDENTIALS_PATH, DEFAULT_SERVER_URL

__all__ = ["MissingCredential", "resolve", "CREDENTIALS_PATH"]


class MissingCredential(RuntimeError):
    """No api key was passed, exported, or written by `memvara-mcp login`."""


def _clean(value: str | None) -> str | None:
    """A blank or whitespace-only environment variable means unset, not empty. An
    exported `MEMVARA_SERVER_URL=` would otherwise be a base url of `""`."""
    return value.strip() if value and value.strip() else None


def _from_file() -> str | None:
    """The key `memvara-mcp login` wrote, or None. A malformed file is None rather than
    an exception: the caller's next stop is the same "run login" message either way."""
    try:
        body = json.loads(CREDENTIALS_PATH.read_text())
    except (OSError, ValueError):
        return None
    key = body.get("api_key") if isinstance(body, dict) else None
    return _clean(key if isinstance(key, str) else None)


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
