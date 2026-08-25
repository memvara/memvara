"""Promises the npm release process makes, as opposed to promises the package makes.

The placeholder at npm/memvara is on the registry as 0.0.1, and this tree holds 0.0.2.
The workflow in release.yml is how a new version gets there. These tests pin the
decisions that are cheap to break in a YAML edit and expensive to notice afterwards:

- the publish job cannot pack (or it will upload different bytes than the ones hashed)
- a version that already exists is a skip, not a republish
- the thing being published is still a name reservation, not a client that grew overnight

Nothing here talks to registry.npmjs.org. The check-npm job is allowed to; the suite is
not. A test that reached the network would fail the offline gate, and a test that skipped
whenever the network was down would stop being a test on the day it was needed.
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil
import subprocess
import tarfile

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
PKG = REPO / "npm" / "memvara"
WORKFLOW = REPO / ".github" / "workflows" / "release.yml"
DOCS = (REPO / "docs", REPO)

# Keys a reservation must not grow. Adding any of these without deleting this list is how
# a placeholder quietly becomes a half-client, and the first person to notice is someone
# who `npm install`ed it expecting one.
CLIENT_KEYS = (
    "connect",
    "retrieve",
    "remember",
    "recall",
    "search",
    "client",
    "createClient",
    "Memvara",
    "mcp",
    "fetch",
)

# What package.json `files` lists. npm always adds package.json itself to the
# tarball; asserting it here would fail the field and pass the tarball, which
# is the less useful half of the disagreement.
DECLARED_FILES = {"LICENSE", "README.md", "index.d.ts", "index.js"}
TARBALL_FILES = DECLARED_FILES | {"package.json"}


def _workflow() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _job(name: str) -> str:
    text = _workflow()
    match = re.search(rf"\n  {re.escape(name)}:\n(.*?)(?=\n  [a-z][a-z0-9-]*:\n|\Z)", text, re.S)
    assert match, f"no job named {name} in release.yml"
    return match.group(1)


def test_the_placeholder_exports_exactly_four_keys_and_implemented_is_false():
    """A reservation that throws on import turns 'wrong package' into a bundler failure
    several layers from the cause. The type has no methods so `memvara.recall()` is
    TS2339; the runtime object has to match or the types are lying."""
    src = (PKG / "index.js").read_text(encoding="utf-8")
    assert "implemented: false" in src
    keys = set(re.findall(r"^\s{2}(\w+):", src, re.M))
    assert keys == {"implemented", "notice", "python", "homepage"}
    for forbidden in CLIENT_KEYS:
        assert forbidden not in keys, f"{forbidden} is a client surface on a reservation"


def test_the_type_declaration_has_no_methods_and_implemented_is_the_literal_false():
    types = (PKG / "index.d.ts").read_text(encoding="utf-8")
    assert "implemented: false" in types
    assert "implemented: boolean" not in types
    assert "():" not in types, "a method on this type is how a caller compiles against a client that is not here"


def test_package_json_files_list_is_exactly_the_reservation():
    meta = json.loads((PKG / "package.json").read_text(encoding="utf-8"))
    assert set(meta["files"]) == DECLARED_FILES
    assert meta["name"] == "memvara"
    assert meta["private"] is False
    for name in TARBALL_FILES:
        assert (PKG / name).is_file(), name


# The resolved path, not the bare name. On Windows npm is `npm.cmd`, and a bare
# "npm" in an argv list is not a thing CreateProcess can find — no PATHEXT search
# happens without a shell, so it raises WinError 2 rather than skipping. `which`
# returns the full `npm.cmd` path there and CreateProcess does run a batch file
# given one. This is why the Windows job was the last red square while the other
# five went green: the guard below asked `which`, and `which` said yes.
NPM = shutil.which("npm")


def _node_runs() -> bool:
    """`which node` is not the same as a working node. Homebrew's current
    binary on this machine aborts on a missing simdjson dylib; treating that
    as a failed assertion would make the suite a function of the laptop.

    Both binaries are actually run, for the same reason: the tests below spawn
    npm as well as node, so a guard that only proved node works leaves npm's
    failures to arrive as errors instead of skips."""
    if shutil.which("node") is None or NPM is None:
        return False
    try:
        subprocess.run(
            ["node", "-e", "process.exit(0)"],
            check=True, capture_output=True, timeout=8,
        )
        subprocess.run(
            [NPM, "--version"],
            check=True, capture_output=True, timeout=30,
        )
        return True
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


@pytest.mark.skipif(not _node_runs(), reason="node/npm missing or unloadable")
def test_a_packed_tarball_contains_exactly_the_files_list():
    """The Actions job asserts this too; this is the copy that fails on the laptop
    that edited package.json, before the tag is pushed."""
    assert NPM is not None  # the skipif guarantees this; mypy does not know that
    proc = subprocess.run(
        [NPM, "pack", "--json"],
        cwd=PKG,
        check=True,
        capture_output=True,
        text=True,
    )
    packed = json.loads(proc.stdout)
    filename = packed[0]["filename"] if isinstance(packed, list) else packed["filename"]
    tarball = PKG / filename
    try:
        with tarfile.open(tarball) as tf:
            # Only "package/", and only once: chaining a second `removeprefix("package")`
            # turns "package/package.json" into ".json", which is what this assertion
            # reported for six CI jobs while the laptop that wrote it never ran the test.
            names = {m.name.removeprefix("package/")
                     for m in tf.getmembers() if m.isfile()}
        assert names == TARBALL_FILES
    finally:
        tarball.unlink(missing_ok=True)


@pytest.mark.skipif(not _node_runs(), reason="node/npm missing or unloadable")
def test_requiring_the_placeholder_does_not_throw_and_is_not_a_client():
    script = (
        "const m = require('./index.js');"
        " if (m.implemented !== false) process.exit(2);"
        " const keys = Object.keys(m).sort().join(',');"
        " if (keys !== 'homepage,implemented,notice,python') process.exit(3);"
        " process.stdout.write(JSON.stringify(Object.keys(m)));"
    )
    out = subprocess.check_output(["node", "-e", script], cwd=PKG, text=True)
    keys = set(json.loads(out))
    assert keys == {"implemented", "notice", "python", "homepage"}
    assert not (keys & set(CLIENT_KEYS))


def test_check_npm_writes_the_two_outputs_the_skip_depends_on():
    job = _job("check-npm")
    assert "npm_version" in job
    assert "npm_exists" in job
    assert "action: SKIP" in job
    assert "action: PUBLISH" in job
    assert "reviewer" not in job.lower()


def test_publish_npm_skips_when_the_version_already_exists():
    """A version already on the registry is a skip, not a republish. npm refuses a
    reused version outright, so rewriting this `if:` to always run turns every tag
    that does not happen to bump package.json into a red release — and the tags that
    do not bump it are most of them, since npm versions here are independent of the
    Python tag."""
    job = _job("publish-npm")
    header = _workflow()
    # The if: sits on the job, just above the body `_job` captured. Read the block
    # including the job's own keys by taking a wider slice.
    start = header.index("\n  publish-npm:\n")
    rest = header[start:start + 800]
    assert "npm_exists" in rest
    assert "'false'" in rest or '"false"' in rest
    assert "github.event_name == 'push'" in rest
    assert "needs: [version, check-npm, build-npm]" in rest or (
        "check-npm" in rest and "build-npm" in rest
    )


def test_publish_npm_cannot_pack_and_must_verify_the_hash():
    """If this job learns how to `npm pack`, the SHA-256 is no longer a check of
    the bytes that were reviewed — it is a check of whatever it just built."""
    job = _job("publish-npm")
    assert not re.search(r"npm pack\b", job)
    assert "npm publish --dry-run" not in job
    assert "sha256" in job.lower()
    assert re.search(r"npm publish\b", job)
    assert ".tgz" in job
    assert "environment:" in _workflow()[ _workflow().index("publish-npm") : ]
    env_slice = _workflow()[_workflow().index("\n  publish-npm:\n"):]
    assert re.search(r"name:\s*npm\b", env_slice)


def test_publish_npm_hands_npm_a_path_and_not_a_git_shorthand():
    """`npm publish npm-dist/memvara-0.0.2.tgz` does not publish that file. To npm a
    spec with a slash and no leading `./` is `owner/repo`, so it shelled out to
    `git ls-remote ssh://git@github.com/npm-dist/memvara-0.0.2.tgz.git` and failed
    with `Permission denied (publickey)` — an auth error naming a repository nobody
    meant to reach, for what was a punctuation bug. That was the first `v0.4.0` run.

    Nothing else caught it. `rehearse_npm.py` passed `Path` objects, which are
    absolute, and npm reads an absolute path as a file — so the rehearsal and the
    workflow were never running the same test. The rehearsal now publishes a bare
    name from the tarball's own directory, and this pins the workflow to the same
    shape: whatever is handed to `npm publish` must not contain a slash unless it
    starts with `./`."""
    job = _job("publish-npm")
    spec = re.search(r"npm publish \"?\$?\{?(\S+?)\}?\"? --access", job)
    assert spec, f"could not find the npm publish argument in:\n{job}"
    argument = spec.group(1)

    # The argument is a shell variable, so follow it to what it was assigned.
    assigned = re.search(rf"{re.escape(argument.lstrip('$'))}=\"?\$\((.+?)\)", job)
    assert assigned, f"{argument} is not assigned from a command in this job"
    produces = assigned.group(1)
    assert "/" not in produces or produces.lstrip().startswith("./"), (
        f"`{produces}` yields a slashed spec; npm would read it as owner/repo. "
        "Either cd into the directory and use a bare name, or prefix with ./"
    )
    assert "working-directory: npm-dist" in job, (
        "the bare tarball name only resolves if the step runs inside npm-dist"
    )


def test_build_npm_packs_and_hashes_and_does_not_publish():
    job = _job("build-npm")
    assert re.search(r"npm pack\b", job)
    assert "sha256" in job.lower()
    assert "npm-dist" in job
    # A real publish in the pack job would skip the human and the hash verify.
    assert not re.search(r"^\s+npm publish (?!--dry-run)", job, re.M)


def test_no_public_doc_calls_npm_unclaimed_while_linking_to_it():
    """Wrong docs are worse than missing ones. A sentence that says the name is
    free next to a link at the live package is how a reader concludes they can
    take it — or that we do not know what we published."""
    unclaimed = re.compile(
        r"(unclaimed|unpublished|still takeable|is 404 right now|"
        r"answered HTTP 404|The name is available, and that was not free|"
        r"will link to a 404)",
        re.I,
    )
    npm_link = re.compile(r"npmjs\.com/package/memvara|registry\.npmjs\.org/memvara")
    offenders = []
    roots = [REPO, REPO / "docs"]
    seen: set[pathlib.Path] = set()
    for root in roots:
        for path in root.glob("*.md"):
            if path in seen:
                continue
            seen.add(path)
            text = path.read_text(encoding="utf-8")
            if unclaimed.search(text) and npm_link.search(text):
                offenders.append(str(path.relative_to(REPO)))
    assert offenders == []
