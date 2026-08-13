#!/usr/bin/env python3
"""Publish an npm package, with the same guards the PyPI script has.

    python3 release/publish_npm.py --package PATH --dry-run
    python3 release/publish_npm.py --package PATH

## Read this before using it: there is nothing to publish yet

At the time of writing **no publishable JavaScript package exists in this project.** Both
`package.json` files are applications marked `"private": true` — `memvara-dashboard` (the
console) and `memvara-web` (the marketing site) — and neither should ever reach a registry.
`npm publish` refuses a private package, which is the correct behaviour and the reason
they are marked that way.

So this script exists for two situations, and it is worth knowing which one you are in:

**1. Reserving the name.** `registry.npmjs.org/memvara` is 404 and an npm *organization*
reserves only the `@memvara/*` scope, not the bare name — exactly as a PyPI organization
reserves no project name. Claiming `memvara` needs a real publish of a real package.
A minimal placeholder that names the project, links to it and does nothing is a legitimate
reservation for a project that genuinely exists; npm's policy is against squatting names
you have no claim to, which is not this. It is still a decision, because the first publish
is public and permanent, so this script will not invent one for you.

**2. Publishing the client when there is one.** The obvious future package is a JS client
for the REST API. When it exists, point `--package` at it and nothing here changes.

## The rule that shapes this, and how it differs from PyPI

**A published version cannot be republished.** `npm unpublish` exists, unlike PyPI's
delete, but it is narrow: within 72 hours for a package nobody depends on, and after that
essentially never. Do not plan around it. `npm deprecate` is the tool that actually
applies afterwards, and it leaves the version installable.

**Scoped packages default to private and the failure is a paywall, not an error you
expect.** `@memvara/thing` publishes as restricted unless you pass `--access public`,
and a restricted package on a free account is refused. This script passes it always;
for an unscoped name it is a harmless no-op.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


class Abort(SystemExit):
    def __init__(self, why: str, fix: str = "") -> None:
        super().__init__(f"\n  REFUSED: {why}\n" + (f"\n  {fix}\n" if fix else ""))


def run(*cmd: str, cwd: Path, capture: bool = False) -> str:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=capture, check=False)
    if proc.returncode != 0:
        if capture:
            sys.stderr.write(proc.stdout or "")
            sys.stderr.write(proc.stderr or "")
        raise Abort(f"`{' '.join(cmd)}` exited {proc.returncode}")
    return (proc.stdout or "").strip()


def published_versions(name: str) -> set[str]:
    """Versions already on the registry. Empty also means 'no such package yet'.

    `requests` rather than `urllib`, for the reason recorded in `publish_pypi.py`: a
    python.org build on macOS does not use the system trust store, so `urllib` fails
    certificate verification against ordinary HTTPS hosts while `curl` succeeds on the
    same machine. A guard that cannot run where it is needed is not a guard.
    """
    try:
        import requests
    except ModuleNotFoundError:              # pragma: no cover
        raise Abort("`requests` is not installed", "python3 -m pip install requests")
    url = f"https://registry.npmjs.org/{name.replace('/', '%2F')}"
    try:
        resp = requests.get(url, timeout=15)
    except requests.RequestException as exc:
        raise Abort(f"cannot reach {url}: {exc}",
                    "The version check is not optional: a version cannot be reused.")
    if resp.status_code == 404:
        return set()
    if resp.status_code != 200:
        raise Abort(f"{url} answered HTTP {resp.status_code}")
    return set(resp.json().get("versions", {}))


def preflight(pkg: Path, allow_dirty: bool, dry_run: bool) -> tuple[str, str]:
    manifest = pkg / "package.json"
    if not manifest.is_file():
        raise Abort(f"no package.json under {pkg}", "Pass --package PATH.")
    meta = json.loads(manifest.read_text())
    name, version = meta.get("name"), meta.get("version")
    if not name or not version:
        raise Abort(f"{manifest} has no name and/or version")

    if meta.get("private"):
        raise Abort(f"{name} is marked \"private\": true",
                    "npm refuses to publish it, correctly. The console and the marketing\n"
                    "  site are applications and are private on purpose — if you meant to\n"
                    "  publish one, that is a decision to make in package.json, not here.")

    # Not checked under --dry-run: nothing is uploaded, so nothing needs authenticating,
    # and a rehearsal you cannot run until you are already set up is a rehearsal nobody
    # does. `npm pack` and `npm publish --dry-run` are both anonymous.
    if not dry_run and not (Path.home() / ".npmrc").is_file() \
            and not os.environ.get("NPM_TOKEN"):
        raise Abort("no ~/.npmrc and no NPM_TOKEN",
                    "npm login    (or set NPM_TOKEN for a CI-style publish)")

    # Publishing from a dirty or unpushed tree has the same consequence it has for PyPI:
    # a permanent artifact whose source is on no branch and in no history.
    dirty = run("git", "status", "--porcelain", cwd=pkg, capture=True)
    if dirty and not allow_dirty:
        raise Abort(f"the repository holding {pkg} has uncommitted changes:\n\n{dirty}",
                    "Commit or stash them.")

    already = published_versions(name)
    if version in already:
        raise Abort(f"{name}@{version} is already on the registry",
                    "Bump `version` in package.json. `npm unpublish` is 72 hours and\n"
                    "  narrower than you think; do not plan around it.")
    print(f"  {name} {version} is free"
          + (f" ({len(already)} existing version(s))" if already else " — first publish"))
    return name, version


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--package", required=True, type=Path,
                    help="directory holding the package.json to publish")
    ap.add_argument("--dry-run", action="store_true",
                    help="run every check and `npm publish --dry-run`, upload nothing")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--tag", default="latest",
                    help="dist-tag; use 'next' or 'beta' to publish without moving 'latest'")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args(argv)

    pkg = args.package.resolve()
    print(f"\n  package : {pkg}")
    name, version = preflight(pkg, args.allow_dirty, args.dry_run)

    # `npm pack --dry-run` lists exactly what would ship. Worth reading every time: npm's
    # default file selection is broad, and the usual accident is publishing a `.env`, a
    # test fixture holding real data, or a source tree you meant to build first.
    print("\n  contents that would be published:")
    run("npm", "pack", "--dry-run", cwd=pkg)

    if args.dry_run:
        run("npm", "publish", "--dry-run", "--access", "public", "--tag", args.tag, cwd=pkg)
        print("\n  dry run only — nothing was published")
        return 0

    if not args.yes:
        prompt = (f"\n  Publish {name}@{version} to npm as '{args.tag}'."
                  f"  This cannot be undone after 72 hours."
                  f"\n  Type the version to confirm: ")
        if input(prompt).strip() != version:
            raise Abort("confirmation did not match; nothing was published")

    # `--access public` always: a scoped package is restricted by default, and a
    # restricted package on a free account is refused with a billing error rather than a
    # permissions one. Harmless for an unscoped name.
    run("npm", "publish", "--access", "public", "--tag", args.tag, cwd=pkg)
    print(f"\n  published {name}@{version}")
    print(f"  verify: npm view {name}@{version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
