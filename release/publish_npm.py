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
import contextlib
import json
import os
import re
import subprocess
import sys
import tempfile
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


#: The only `.npmrc` key that authenticates a publish to the public registry. Writing a
#: token under any other name is silently inert — npm does not warn that it found a token
#: somewhere it does not look.
AUTH_KEY = "//registry.npmjs.org/:_authToken"

#: npm access tokens are `npm_` followed by 36 characters. Matching the shape is enough to
#: recognise one filed under the wrong key; the value itself is never read or printed.
TOKEN_SHAPE = re.compile(r'^"?npm_[A-Za-z0-9]{20,}"?$')


def npmrc_entries(path: Path) -> list[tuple[int, str, str]]:
    """`(line number, key, value)` for each assignment in an npmrc. Values stay local."""
    out = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        s = line.strip()
        if s and not s.startswith((";", "#")) and "=" in s:
            k, v = s.split("=", 1)
            out.append((i, k.strip(), v.strip()))
    return out


def check_npm_auth() -> None:
    """Refuse unless a credential exists **where npm will actually look for it**.

    Three ways this goes wrong, all of which produce errors that point elsewhere:

    `NPM_TOKEN` alone does nothing. Unlike twine and `TWINE_PASSWORD`, the npm CLI has no
    built-in knowledge of that variable; it is a convention that works only because CI
    images write an npmrc line referencing it. Setting it and expecting a login gets
    `ENEEDAUTH`, which reads as "your token is wrong" rather than "your token was never
    consulted". This script now makes the variable work — see `npm_auth_flags`.

    A token filed under the wrong key is worse than no token, because npm's own diagnostics
    lead away from it. `key` and `cert` are npm's legacy *TLS client certificate* options,
    so a token parked there is handed to OpenSSL as a PEM private key, and OpenSSL 3
    refuses it with `ERR_OSSL_UNSUPPORTED: DECODER routines::unsupported` — a message with
    no visible connection to authentication, emitted before any request is sent, which
    breaks `npm login` itself and so removes the obvious way out.

    And a `~/.npmrc` that merely exists proves nothing, which is what this used to check.
    """
    if os.environ.get("NPM_TOKEN"):
        return

    rc = Path.home() / ".npmrc"
    entries = npmrc_entries(rc) if rc.is_file() else []
    if any(k.endswith("_authToken") for _, k, _ in entries):
        return

    misfiled = [(i, k) for i, k, v in entries
                if TOKEN_SHAPE.match(v) and not k.endswith("_authToken")]
    if misfiled:
        i, k = misfiled[0]
        raise Abort(
            f"~/.npmrc line {i} looks like an npm token stored under `{k}`, which is not\n"
            f"  a key npm reads for authentication",
            f"`key` and `cert` are npm's legacy TLS client-certificate options. A token\n"
            f"  placed there is passed to OpenSSL as a private key, which is where\n"
            f"  `ERR_OSSL_UNSUPPORTED ... DECODER routines::unsupported` comes from — and\n"
            f"  because it fails before the request is sent, it breaks `npm login` too.\n\n"
            f"  Replace that line with:\n\n"
            f"      {AUTH_KEY}=<token>\n\n"
            f"  Treat the misfiled token as exposed and issue a new one:\n"
            f"  https://www.npmjs.com/settings/~/tokens")

    raise Abort(
        f"no npm credential: ~/.npmrc has no `{AUTH_KEY}` and $NPM_TOKEN is unset",
        "Either:\n"
        "      npm login\n"
        "  or, to keep the token out of any file, export NPM_TOKEN for this shell only:\n"
        "      read -rs NPM_TOKEN && export NPM_TOKEN")


@contextlib.contextmanager
def npm_auth_flags():
    """npm flags that make `$NPM_TOKEN` authenticate, or nothing if it is unset.

    npm expands `${VAR}` inside an npmrc, so the temporary file written here contains the
    literal text `${NPM_TOKEN}` and never the token: the secret stays in the environment,
    reaches npm through its own process environment, and touches no disk this script owns.

    `--userconfig` *replaces* `~/.npmrc` for this invocation rather than merging, so any
    registry or proxy settings there do not apply. That is why it is used only when
    `NPM_TOKEN` is set, which is an explicit request for a self-contained CI-style publish.
    """
    if not os.environ.get("NPM_TOKEN"):
        yield []
        return
    fd, name = tempfile.mkstemp(prefix="npmrc-publish-", suffix=".ini")
    os.close(fd)
    rc = Path(name)
    rc.chmod(0o600)
    rc.write_text(f"{AUTH_KEY}=${{NPM_TOKEN}}\n")
    try:
        yield ["--userconfig", str(rc)]
    finally:
        rc.unlink(missing_ok=True)


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
    if not dry_run:
        check_npm_auth()

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
    ap.add_argument("--otp", metavar="CODE",
                    help="2FA code, if the account enforces 2FA for writes. Only a classic"
                         " 'Publish' token needs one; an 'Automation' token bypasses 2FA.")
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

    with npm_auth_flags() as auth:
        if args.otp:
            auth = [*auth, "--otp", args.otp]
        return publish(pkg, name, version, args.tag, args.yes, auth)


def publish(pkg: Path, name: str, version: str, tag: str, yes: bool,
            auth: list[str]) -> int:

    # Resolve the identity before asking anyone to confirm anything. It costs one request,
    # it proves the credential authenticates at all, and it turns npm's "these credentials"
    # — which names nothing — into an account you can go and look at.
    try:
        who = run("npm", "whoami", *auth, cwd=pkg, capture=True)
    except Abort:
        raise Abort("the credential does not authenticate to the registry at all",
                    "`npm whoami` failed, so this is not a permissions problem: the token\n"
                    "  is wrong, revoked, or expired. Issue a new one at\n"
                    "  https://www.npmjs.com/settings/~/tokens")
    print(f"  identity  : {who}")

    if not yes:
        prompt = (f"\n  Publish {name}@{version} to npm as '{tag}'."
                  f"  This cannot be undone after 72 hours."
                  f"\n  Type the version to confirm: ")
        if input(prompt).strip() != version:
            raise Abort("confirmation did not match; nothing was published")

    # `--access public` always: a scoped package is restricted by default, and a
    # restricted package on a free account is refused with a billing error rather than a
    # permissions one. Harmless for an unscoped name.
    try:
        run("npm", "publish", "--access", "public", "--tag", tag, *auth, cwd=pkg)
    except Abort:
        # npm's own text for this is "You may not perform that action with these
        # credentials", which describes the outcome and none of the causes. Since
        # `npm whoami` already succeeded above, the token is real and the account is
        # known — so what is missing is write permission, and there are only a few ways
        # for a token that authenticates to lack it.
        raise Abort(
            f"{who} authenticated, but is not allowed to publish {name}",
            "The token works and the account is real, so this is scope, not identity:\n\n"
            + ("  1. **An unscoped name belongs to no scope and no organisation.** If this\n"
               f"     token is granular and lists a scope such as `@{name}` or an org named\n"
               f"     `{name}`, it still cannot publish `{name}` — a scope and a bare name\n"
               "     that share a word are unrelated namespaces. This is the same fact that\n"
               "     makes the placeholder necessary at all: owning the org reserves\n"
               f"     `@{name}/*` and leaves `{name}` itself open to anyone.\n"
               "     Grant the token write on **all packages**, or use a classic token.\n\n"
               if not name.startswith("@") else
               "  1. A granular token must list this package's scope explicitly.\n\n")
            + "  2. A **read-only** classic token cannot publish. Publishing needs\n"
            "     'Automation' (no OTP) or 'Publish' (prompts for an OTP). A granular\n"
            "     token cannot be scoped to `" + name + "` before it exists, so the first\n"
            "     publish of a new name needs account-wide write either way — the same\n"
            "     trap as PyPI. Narrow it afterwards, not before.\n\n"
            "  3. If 2FA is enforced for writes, pass the six digits: --otp 123456\n\n"
            "  4. If the account is a member of an org, check the org has not restricted\n"
            "     publishing of unscoped names.\n\n"
            "  Tokens: https://www.npmjs.com/settings/~/tokens")
    print(f"\n  published {name}@{version}")
    print(f"  verify: npm view {name}@{version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
