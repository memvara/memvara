#!/usr/bin/env python3
"""Publish the `memvara` package to PyPI, with the guards a one-way action needs.

    python3 release/publish_pypi.py --test     # TestPyPI dry run, do this first
    python3 release/publish_pypi.py            # the real thing

## The rule that shapes everything here

**A version number can never be reused on PyPI.** Not after yanking, not after deleting
the release, not ever. `0.1.0` is one attempt. Every check below exists because it is
cheaper than discovering the problem afterwards, when the only remedy is to burn a
version number and ship `0.1.1` whose changelog says "same as 0.1.0, but correct".

## What this refuses to do, and why each one has happened

* **Upload a stale `dist/`.** `twine upload dist/*` uploads whatever is sitting in that
  directory. On the day this was written, `dist/` held an sdist built four days and
  roughly forty commits earlier — `twine upload dist/*` would have published it, and the
  version number would have been spent on it. So `dist/` is deleted and rebuilt every run,
  never reused.
* **Publish uncommitted work.** A dirty tree means the artifact contains code that exists
  on no branch and in no history. The upload is permanent; the source of it would not be.
* **Publish an unpushed commit.** The tag this writes, and the `Repository` URL in the
  package metadata, both point at a commit. If it is not on the remote, they point at
  nothing for everyone except the machine that published.
* **Reupload an existing version.** Checked against the live index before building, so the
  failure arrives in two seconds rather than after the build, the confirmation and the
  upload.

## Credentials

This script never creates, prompts for, stores, or logs a token. Set them in the
environment before running:

    read -rs TWINE_PASSWORD && export TWINE_PASSWORD && export TWINE_USERNAME=__token__

For the **first ever upload** the token must be scoped to the *entire account* — PyPI
cannot scope a token to a project that does not exist yet. Replace it with a
project-scoped token immediately afterwards; that is a one-minute job that stops a leaked
token from being able to publish anything else you own.

TestPyPI is a **separate service with a separate account and a separate token**. A PyPI
token will not authenticate against it, and the error when you try says only "invalid
credentials".
"""

from __future__ import annotations

import argparse
import base64
import configparser
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

#: The repository this script publishes, resolved from the script's own location rather
#: than configured. It lives *in* that repository, which is the point: a release procedure
#: that ships with the package is one a contributor can read and run, and there is no path
#: to get wrong on a different machine.
#:
#: It is public, because this repository is. Nothing here is a secret — the script reads
#: credentials from the environment and never writes, prompts for, or logs one. What it
#: does contain is the reasoning behind each guard, which is worth more in the open than
#: in a private repo where only one person ever sees it.
#:
#: The natural next step, when one person releasing stops being enough, is a GitHub
#: Actions workflow using **PyPI Trusted Publishing** — OIDC, no token existing anywhere.
#: These checks move into it unchanged; only the upload step is replaced.
REPO = Path(__file__).resolve().parent.parent

PYPI = "https://pypi.org/pypi/{name}/json"
TESTPYPI = "https://test.pypi.org/pypi/{name}/json"


class Abort(SystemExit):
    """Refusal with a reason. Every one of these is cheaper than the alternative."""

    def __init__(self, why: str, fix: str = "") -> None:
        super().__init__(f"\n  REFUSED: {why}\n" + (f"\n  {fix}\n" if fix else ""))


def run(*cmd: str, cwd: Path | None = None, capture: bool = False) -> str:
    proc = subprocess.run(cmd, cwd=cwd or REPO, text=True,
                          capture_output=capture, check=False)
    if proc.returncode != 0:
        if capture:
            sys.stderr.write(proc.stdout or "")
            sys.stderr.write(proc.stderr or "")
        raise Abort(f"`{' '.join(cmd)}` exited {proc.returncode}")
    return (proc.stdout or "").strip()


def git(*args: str) -> str:
    return run("git", *args, capture=True)


def released_versions(name: str, *, test: bool) -> set[str]:
    """Versions already on the index. An empty set also means 'no such project yet'.

    Uses `requests` rather than `urllib` for one reason, found the first time this ran:
    a python.org build on macOS does **not** use the system trust store, so `urllib`
    raises `CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate` against a
    perfectly ordinary HTTPS host — while `curl` on the same machine succeeds. That would
    have made this guard unrunnable exactly where it is needed, and a guard that has to be
    bypassed to get work done is worse than no guard, because the bypass becomes habit.
    `requests` ships `certifi` and is a hard dependency of `twine`, which this script
    already requires, so it is present whenever this script can do its job at all.
    """
    url = (TESTPYPI if test else PYPI).format(name=name)
    try:
        import requests
    except ModuleNotFoundError:              # pragma: no cover - twine guarantees it
        raise Abort("`requests` is not installed",
                    "python3 -m pip install twine  (it brings requests and certifi)")
    try:
        resp = requests.get(url, timeout=15)
    except requests.RequestException as exc:
        raise Abort(f"cannot reach {url}: {exc}",
                    "The version check is not optional: a reused version cannot be undone.")
    if resp.status_code == 404:
        return set()                         # the name is free, or this is release one
    if resp.status_code != 200:
        raise Abort(f"{url} answered HTTP {resp.status_code}")
    return set(resp.json().get("releases", {}))


#: `.pypirc` section name and expected token issuer, keyed on whether `--test` was passed.
#: These are two unrelated services that happen to run the same software: separate
#: accounts, separate tokens, separate name namespaces.
_TARGET = {False: ("pypi", "pypi.org"), True: ("testpypi", "test.pypi.org")}


def token_issuer(token: str) -> str | None:
    """Which service issued a PyPI API token, or None if it is not one.

    A token is `pypi-` followed by a base64url macaroon, and a macaroon carries its issuing
    location in the clear. Reading that single field is what lets this distinguish a
    pypi.org token from a test.pypi.org one locally. **The token is never printed, logged,
    stored or sent anywhere** — the only value that leaves this function is a domain name.
    """
    if not token.startswith("pypi-"):
        return None                     # a legacy username/password; nothing to check
    raw = token[5:]
    try:
        blob = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))
    except Exception:
        return None
    # "test.pypi.org" contains "pypi.org", so the longer host has to be tested first.
    return next((h for h in ("test.pypi.org", "pypi.org") if h.encode() in blob), None)


def check_credential_matches_target(test: bool) -> None:
    """Refuse unless a credential exists for *the service being uploaded to*.

    This used to check only whether `TWINE_PASSWORD` or `~/.pypirc` existed **at all**, and
    that is precisely how the first real run of this script failed. A `.pypirc` holding
    only a `[pypi]` section, `--test` on the command line, and TestPyPI answered with a
    bare `403 Forbidden` carrying no message — after building, checking and uploading half
    a megabyte. Holding *a* credential is not holding *the* one.

    The module docstring warned about this in prose the whole time. It made no difference,
    because the guard passed and nobody reads a warning about a problem they do not yet
    know they have. A rule the code does not enforce is a rule that is not in effect.
    """
    section, want = _TARGET[test]
    service = "TestPyPI" if test else "PyPI"

    # Twine's own precedence: TWINE_PASSWORD wins over .pypirc, for either target.
    token = os.environ.get("TWINE_PASSWORD")
    source = "$TWINE_PASSWORD"

    if not token:
        rc = Path.home() / ".pypirc"
        if not rc.is_file():
            raise Abort(f"no $TWINE_PASSWORD and no ~/.pypirc — nothing to authenticate to "
                        f"{service} with",
                        'read -rs TWINE_PASSWORD && export TWINE_PASSWORD && '
                        'export TWINE_USERNAME=__token__')
        parser = configparser.ConfigParser()
        parser.read(rc)
        if not parser.has_option(section, "password"):
            raise Abort(
                f"~/.pypirc has no [{section}] section, which is the one {service} needs\n"
                f"  (it has: {parser.sections() or 'no sections at all'})",
                f"{service} is a separate service with its own account and its own token —\n"
                f"  a {'PyPI' if test else 'TestPyPI'} token does not work against it. Register at\n"
                f"  https://{want}/account/register/ , create a token, and add:\n\n"
                f"      [{section}]\n"
                f"      username = __token__\n"
                f"      password = <the {service} token>\n\n"
                f"  Or export TWINE_PASSWORD for this shell only, which leaves it on no disk.")
        token = parser.get(section, "password")
        source = f"~/.pypirc [{section}]"

    issuer = token_issuer(token)
    if issuer and issuer != want:
        raise Abort(
            f"the token in {source} was issued by {issuer}, but this uploads to {want}",
            f"{service} needs a token created at https://{want}/manage/account/token/ .\n"
            "  Using the other one gets you a 403 whose body does not say why.")
    print(f"  credential: {source}" + (f", issued by {issuer}" if issuer else ""))


def preflight(test: bool, allow_dirty: bool) -> tuple[str, str]:
    if not (REPO / "pyproject.toml").is_file():
        raise Abort(f"no pyproject.toml under {REPO}",
                    "This script must live in release/ inside the package repository.")
    meta = tomllib.load((REPO / "pyproject.toml").open("rb"))["project"]
    name, version = meta["name"], meta["version"]

    check_credential_matches_target(test)

    dirty = git("status", "--porcelain")
    if dirty and not allow_dirty:
        raise Abort(f"{REPO} has uncommitted changes:\n\n{dirty}",
                    "Commit or stash them. An upload is permanent; a dirty tree is not.")

    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch != "main":
        print(f"  ! on branch {branch!r}, not main — continuing, but check that is intended")

    try:
        ahead = git("rev-list", "--count", "@{u}..HEAD")
    except SystemExit:
        raise Abort("HEAD has no upstream branch",
                    "Push the branch first: the package metadata points at this commit.")
    if ahead != "0":
        raise Abort(f"{ahead} commit(s) not pushed to the remote",
                    "git push — the tag and the Repository URL both point at this commit.")

    already = released_versions(name, test=test)
    where = "TestPyPI" if test else "PyPI"
    if version in already:
        raise Abort(f"{name} {version} is already on {where}",
                    "A version number cannot be reused. Bump `version` in pyproject.toml.")
    print(f"  {name} {version} is free on {where}"
          + (f" ({len(already)} existing release(s))" if already else " — first release"))
    return name, version


def build() -> list[Path]:
    dist = REPO / "dist"
    if dist.exists():
        # Never reuse: `twine upload dist/*` ships whatever is here, and a leftover
        # artifact from an earlier build looks identical to a fresh one.
        shutil.rmtree(dist)
        print("  removed the previous dist/ — stale artifacts are how the wrong code ships")
    run(sys.executable, "-m", "build")
    made = sorted(dist.iterdir())
    if len(made) != 2:
        raise Abort(f"expected a wheel and an sdist, got {[p.name for p in made]}")
    return made


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--test", action="store_true",
                    help="upload to TestPyPI instead (separate account and token)")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="build from an uncommitted tree; for rehearsals only")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the post-upload install into a throwaway venv")
    ap.add_argument("--yes", action="store_true",
                    help="skip the confirmation prompt (for a script that already asked)")
    args = ap.parse_args(argv)

    print(f"\n  core repo : {REPO}")
    print(f"  target    : {'TestPyPI' if args.test else 'PyPI (permanent)'}")
    name, version = preflight(args.test, args.allow_dirty)

    made = build()
    for path in made:
        print(f"  built     : {path.name}  ({path.stat().st_size:,} bytes)")

    # `twine check` renders the README the way PyPI will. A long_description that fails
    # to render is not rejected at upload — it is accepted and displayed as raw text
    # forever, on the page that is the project's front door.
    run(sys.executable, "-m", "twine", "check", *[str(p) for p in made])

    if not args.yes:
        where = "TestPyPI" if args.test else "PyPI"
        prompt = (f"\n  Upload {name} {version} to {where}."
                  + ("" if args.test else "  This cannot be undone.")
                  + f"\n  Type the version to confirm: ")
        if input(prompt).strip() != version:
            raise Abort("confirmation did not match; nothing was uploaded")

    upload = [sys.executable, "-m", "twine", "upload"]
    if args.test:
        upload += ["--repository", "testpypi"]
    run(*upload, *[str(p) for p in made])
    print(f"\n  uploaded {name} {version}")

    if not args.no_verify:
        verify(name, version, test=args.test)

    if not args.test:
        tag = f"v{version}"
        existing = git("tag", "--list", tag)
        if existing:
            print(f"  tag {tag} already exists — left alone")
        else:
            print(f"\n  Next: git tag -a {tag} -m '{name} {version}' && git push origin {tag}")
    return 0


def verify(name: str, version: str, *, test: bool) -> None:
    """Install the uploaded artifact into a throwaway venv and import it.

    The only check that exercises what a user actually gets. A package can build, pass
    `twine check`, upload cleanly and still be uninstallable — a missing dependency, a
    `py.typed` that did not make it into the wheel, a module excluded by the build
    backend's file discovery.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        venv = Path(tmp) / "v"
        run(sys.executable, "-m", "venv", str(venv), cwd=Path(tmp))
        pip = str(venv / "bin" / "pip")
        spec = f"{name}=={version}"
        cmd = [pip, "install", "--quiet", spec]
        if test:
            # TestPyPI has no numpy. Without a real index alongside it the install fails
            # for a reason that has nothing to do with this package.
            cmd = [pip, "install", "--quiet",
                   "--index-url", "https://test.pypi.org/simple/",
                   "--extra-index-url", "https://pypi.org/simple/", spec]
        print(f"  verifying : {spec} into a throwaway venv")
        run(*cmd, cwd=Path(tmp))
        out = run(str(venv / "bin" / "python"), "-c",
                  "import memvara; from memvara import Memvara; print('import ok')",
                  cwd=Path(tmp), capture=True)
        print(f"  verified  : {out}")


if __name__ == "__main__":
    raise SystemExit(main())
