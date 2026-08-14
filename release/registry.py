#!/usr/bin/env python3
"""What each registry already has. One question, asked before anything irreversible.

`publish_pypi.py` and `publish_npm.py` each ask this for themselves and are left alone —
they are the manual path, they were audited as standalone scripts, and rewriting them to
import this would buy nothing and risk something. This module is the same question asked
by the automated path, where both registries have to be answered in one job.

`requests` rather than `urllib`, for the reason recorded in `publish_pypi.py` and found
the first time it ran: a python.org build on macOS does not use the system trust store, so
`urllib` raises `CERTIFICATE_VERIFY_FAILED` against ordinary HTTPS hosts while `curl` on
the same machine succeeds. A guard that cannot run on the maintainer's laptop is one that
gets bypassed, and the bypass becomes the habit.
"""

from __future__ import annotations

from release.versions import Refused

PYPI = "https://pypi.org/pypi/{name}/json"
NPM = "https://registry.npmjs.org/{name}"

#: Long enough to survive a slow registry, short enough that a hung release job fails
#: rather than burning the six-hour job limit.
TIMEOUT = 15


def _json(url: str) -> dict | None:
    """The document, or None for 404 — which means "no such project", not "no answer".

    Every other outcome raises. A release must not proceed on "the registry did not
    answer": the check exists precisely because the action it guards cannot be undone, so
    an unreachable registry has to stop the job rather than be read as "nothing published".
    """
    try:
        import requests
    except ModuleNotFoundError:                      # pragma: no cover - CI installs it
        raise Refused("`requests` is not installed",)
    try:
        resp = requests.get(url, timeout=TIMEOUT)
    except requests.RequestException as exc:
        raise Refused(f"cannot reach {url}: {exc}. The version check is not optional — a "
                      f"published version cannot be reused.")
    if resp.status_code == 404:
        return None
    if resp.status_code != 200:
        raise Refused(f"{url} answered HTTP {resp.status_code}")
    return resp.json()


def pypi_versions(name: str) -> set[str]:
    """Every version PyPI has for `name`. Empty also means the project does not exist."""
    doc = _json(PYPI.format(name=name))
    return set(doc.get("releases", {})) if doc else set()


def npm_versions(name: str) -> set[str]:
    """Every version npm has for `name`. Empty also means the package does not exist."""
    doc = _json(NPM.format(name=name.replace("/", "%2F")))
    return set(doc.get("versions", {})) if doc else set()
