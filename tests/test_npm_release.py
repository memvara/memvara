"""Promises the npm package and its release process make.

`memvara` on npm was a name reservation until `0.1.0`, and this file used to pin that:
four keys, no client surface, nothing to run. It now pins the opposite, because the
package became a CLI — `npx memvara` bridges a stdio MCP client to the hosted server.
The reversal is deliberate and is recorded in `CHANGELOG.md`; these tests are not a
casualty of it, they are how the new shape stays true.

What is asserted here is the *shape*: the manifest, the tarball, the workflow. The
bridge's own behaviour is tested in `npm/memvara/test/` with `node --test`, which runs
in `release-npm.yml` and in `ci.yml` — a Python assertion about JavaScript behaviour
would be a worse version of a test that already exists.

Nothing here talks to registry.npmjs.org. The release workflow is allowed to; the suite
is not. A test that reached the network would fail the offline gate, and a test that
skipped whenever the network was down would stop being a test on the day it was needed.
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
WORKFLOW = REPO / ".github" / "workflows" / "release-npm.yml"
PY_WORKFLOW = REPO / ".github" / "workflows" / "release.yml"
DOCS = (REPO / "docs", REPO)

#: Everything the tarball must contain, and nothing else. `package.json` is added by npm
#: whether or not it is in `files`. `test/` is deliberately absent: the suite runs from
#: the repository, and shipping it to every installer is weight with no reader.
TARBALL_FILES = {
    "LICENSE",
    "README.md",
    "index.d.ts",
    "index.js",
    "package.json",
    "bin/memvara.js",
    "lib/bridge.js",
    "lib/creds.js",
    "lib/oauth.js",
    "lib/transport.js",
}

#: The resolved path, not the bare name. On Windows npm is `npm.cmd`, and a bare "npm"
#: in an argv list is not a thing CreateProcess can find — no PATHEXT search happens
#: without a shell, so it raises WinError 2 rather than skipping.
NPM = shutil.which("npm")


def _workflow() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _job(name: str) -> str:
    text = _workflow()
    start = text.index(f"\n  {name}:\n")
    rest = text[start + 1 :]
    following = re.search(r"\n  [a-z][a-z0-9_-]*:\n", rest)
    return rest[: following.start()] if following else rest


def _manifest() -> dict:
    return json.loads((PKG / "package.json").read_text(encoding="utf-8"))


def _node_runs() -> bool:
    """`which node` is not the same as a working node. Homebrew's binary on one machine
    aborts on a missing simdjson dylib; treating that as a failed assertion would make
    the suite a function of the laptop. Both binaries are run, because the tests below
    spawn npm as well as node."""
    if shutil.which("node") is None or NPM is None:
        return False
    try:
        subprocess.run(["node", "-e", "process.exit(0)"],
                       check=True, capture_output=True, timeout=8)
        subprocess.run([NPM, "--version"], check=True, capture_output=True, timeout=30)
        return True
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


# -- the package ------------------------------------------------------------------


def test_the_package_ships_a_binary_and_the_file_it_names_exists():
    meta = _manifest()
    assert meta["name"] == "memvara"
    assert meta["private"] is False
    assert meta["bin"] == {"memvara": "bin/memvara.js"}
    entry = PKG / meta["bin"]["memvara"]
    assert entry.is_file(), "package.json names a bin that is not in the tree"
    assert entry.read_text(encoding="utf-8").startswith("#!"), (
        "a bin without a shebang is not executable when npm links it"
    )


def test_the_package_has_no_runtime_dependencies():
    """This process holds a bearer token. A dependency here is a dependency in
    everyone's install of it, and 'we will review what we add' is a policy, not a
    control. Zero is the control."""
    meta = _manifest()
    assert not meta.get("dependencies"), meta.get("dependencies")
    assert not (PKG / "package-lock.json").exists(), (
        "a lockfile means something got installed; this package has nothing to lock"
    )


def test_node_20_is_the_floor_because_fetch_is():
    """The bridge uses global `fetch`, which is Node 18+. 18 is end-of-life, so the
    floor is 20 — and it has to be *declared*, or an install on 16 fails at the first
    request with a ReferenceError rather than at install time with a reason."""
    assert _manifest()["engines"]["node"] == ">=20"


def test_the_module_export_is_a_signpost_and_not_a_client():
    """The package is a CLI. `require("memvara")` is almost certainly a mistake, and
    the export exists to say so — without throwing, because a module that throws on
    import breaks a bundler's graph several layers from the cause."""
    src = (PKG / "index.js").read_text(encoding="utf-8")
    keys = set(re.findall(r"^\s{2}(\w+):", src, re.M))
    assert keys == {"isLibrary", "cli", "notice", "python", "homepage"}
    for forbidden in ("connect", "retrieve", "remember", "recall", "search",
                      "createClient", "Memvara"):
        assert forbidden not in keys, f"{forbidden} is a client surface on a CLI package"
    assert "isLibrary: false" in src


def test_the_type_declaration_has_no_methods():
    """Four properties and no methods, so using this package as a client is a compile
    error at the earliest possible moment: `memvara.recall(...)` is TS2339."""
    types = (PKG / "index.d.ts").read_text(encoding="utf-8")
    assert "isLibrary: false" in types
    assert "isLibrary: boolean" not in types
    assert "():" not in types


@pytest.mark.skipif(not _node_runs(), reason="node/npm missing or unloadable")
def test_a_packed_tarball_contains_exactly_the_expected_files():
    """The workflow asserts this too; this is the copy that fails on the laptop that
    edited package.json, before the tag is pushed."""
    assert NPM is not None  # the skipif guarantees this; mypy does not know that
    proc = subprocess.run([NPM, "pack", "--json"], cwd=PKG,
                          check=True, capture_output=True, text=True)
    packed = json.loads(proc.stdout)
    filename = packed[0]["filename"] if isinstance(packed, list) else packed["filename"]
    tarball = PKG / filename
    try:
        with tarfile.open(tarball) as tf:
            # Only "package/", and only once: chaining a second removeprefix("package")
            # turns "package/package.json" into ".json", which is what this assertion
            # reported for six CI jobs while the laptop that wrote it never ran the test.
            names = {m.name.removeprefix("package/")
                     for m in tf.getmembers() if m.isfile()}
        assert names == TARBALL_FILES
        assert not any(n.startswith("test/") for n in names), (
            "the suite is run from the repository, not shipped to installers"
        )
    finally:
        tarball.unlink(missing_ok=True)


@pytest.mark.skipif(not _node_runs(), reason="node/npm missing or unloadable")
def test_the_cli_reports_its_version_and_that_version_is_the_manifest_s():
    """A `--version` that disagrees with package.json is how a bug report names a
    release nobody shipped."""
    out = subprocess.check_output(["node", str(PKG / "bin" / "memvara.js"), "--version"],
                                  cwd=PKG, text=True, timeout=30)
    assert out.strip() == _manifest()["version"]


@pytest.mark.skipif(not _node_runs(), reason="node/npm missing or unloadable")
def test_requiring_the_package_does_not_throw():
    script = (
        "const m = require('./index.js');"
        " if (m.isLibrary !== false) process.exit(2);"
        " process.stdout.write(JSON.stringify(Object.keys(m)));"
    )
    out = subprocess.check_output(["node", "-e", script], cwd=PKG, text=True)
    assert set(json.loads(out)) == {"isLibrary", "cli", "notice", "python", "homepage"}


# -- the release process ----------------------------------------------------------


def test_npm_releases_on_its_own_tag_and_not_the_python_one():
    """The coupling this replaces produced `0.4.1`: a PyPI release containing no Python
    changes at all, cut only because npm serves the README of the published version.
    `release.yml` fires on `v*`, which `npm-v0.1.0` does not match, so the two trains
    cannot start each other."""
    assert 'tags: ["npm-v*"]' in _workflow()
    assert "npm-v" not in PY_WORKFLOW.read_text(encoding="utf-8")
    # And the Python workflow must not have grown the npm jobs back.
    python_side = PY_WORKFLOW.read_text(encoding="utf-8")
    for job in ("check-npm:", "build-npm:", "publish-npm:"):
        assert job not in python_side


def test_the_version_job_refuses_a_tag_that_disagrees_with_the_manifest():
    job = _job("version")
    assert "npm-v" in job
    assert "version mismatch" in job
    assert "action: SKIP" in job
    assert "action: PUBLISH" in job


def test_the_publish_job_skips_a_version_the_registry_already_has():
    """npm refuses a reused version outright, so a publish job that always ran would
    turn every re-tag into a red release."""
    text = _workflow()
    start = text.index("\n  publish:\n")
    head = text[start : start + 700]
    assert "needs.version.outputs.exists == 'false'" in head
    assert "github.event_name == 'push'" in head


def test_publish_cannot_pack_and_must_verify_the_hash():
    """If this job learns how to `npm pack`, the SHA-256 stops being a check of the
    bytes that were reviewed and becomes a check of whatever it just built."""
    job = _job("publish")
    assert not re.search(r"npm pack\b", job)
    assert "sha256" in job.lower()
    assert re.search(r"npm publish\b", job)
    assert re.search(r"name:\s*npm\b", job), "the trusted publisher names this environment"


def test_publish_hands_npm_a_path_and_not_a_git_shorthand():
    """`npm publish npm-dist/memvara-0.0.2.tgz` does not publish that file. To npm a
    spec with a slash and no leading `./` is `owner/repo`, so it shelled out to
    `git ls-remote ssh://git@github.com/npm-dist/memvara-0.0.2.tgz.git` and failed with
    `Permission denied (publickey)` — an auth error naming a repository nobody meant to
    reach, for a punctuation bug. That was the first `v0.4.0` run."""
    job = _job("publish")
    spec = re.search(r"npm publish \"?\$?\{?(\S+?)\}?\"? --access", job)
    assert spec, f"could not find the npm publish argument in:\n{job}"
    argument = spec.group(1)
    assigned = re.search(rf"{re.escape(argument.lstrip('$'))}=\"?\$\((.+?)\)", job)
    assert assigned, f"{argument} is not assigned from a command in this job"
    produces = assigned.group(1)
    assert "/" not in produces or produces.lstrip().startswith("./"), (
        f"`{produces}` yields a slashed spec; npm would read it as owner/repo"
    )
    assert "working-directory: npm-dist" in job


def test_the_build_job_refuses_a_surprise_file_list_and_any_dependency():
    job = _job("build")
    assert re.search(r"npm pack\b", job)
    assert "sha256" in job.lower()
    assert not re.search(r"^\s+npm publish (?!--dry-run)", job, re.M)
    for expected in ("bin/memvara.js", "lib/transport.js", "lib/oauth.js"):
        assert expected in job, f"{expected} is shipped but not in the expected list"
    assert "dependencies" in job, "the zero-dependency property is asserted in CI"


def test_the_node_suite_runs_in_the_release_and_on_more_than_one_node():
    """The bridge's behaviour is tested in JavaScript. If that suite is not wired into
    the release, the tests exist and nothing runs them at the moment it matters."""
    job = _job("test")
    assert "node --test" in job
    assert "working-directory: npm/memvara" in job
    versions = re.search(r"node:\s*\[(.+?)\]", job)
    assert versions, "the node matrix is what stops a stdlib assumption shipping"
    assert len(versions.group(1).split(",")) >= 2


def test_releasing_names_the_workflow_that_actually_publishes():
    """npm's trusted publisher registration names **one workflow filename**, and the
    OIDC exchange compares it against the token's `job_workflow_ref`. So moving the
    publish job to a different file silently invalidates the registration — the
    repository, the environment and the package are all still right, and the publish
    still fails.

    It failed exactly this way on `npm-v0.1.0`, hours after 0.0.3 published cleanly,
    because the job moved to `release-npm.yml` and `docs/RELEASING.md` still said
    `release.yml`. The error gives no hint: npm answers `E404` on `PUT` for a package
    that plainly exists, rather than a 403, so that a probe cannot enumerate private
    names.

    This is the check that makes the document and the workflow move together."""
    releasing = (REPO / "docs" / "RELEASING.md").read_text(encoding="utf-8")
    table = re.search(r"\|\s*Workflow filename\s*\|\s*`([^`]+)`\s*\|", releasing)
    assert table, "RELEASING.md no longer states a workflow filename for the npm publisher"
    named = table.group(1)
    assert named == WORKFLOW.name, (
        f"RELEASING.md tells a maintainer to register {named!r}, but the job that runs "
        f"`npm publish` is in {WORKFLOW.name!r}. Registering the named one produces an "
        "E404 on PUT at publish time, which reads as 'no such package'."
    )
    # And the file it names must be the one that actually publishes.
    assert "npm publish" in _workflow()


def test_no_public_doc_calls_npm_unclaimed_while_linking_to_it():
    """Wrong docs are worse than missing ones. A sentence that says the name is free
    next to a link at the live package is how a reader concludes they can take it."""
    unclaimed = re.compile(
        r"(unclaimed|unpublished|still takeable|is 404 right now|"
        r"answered HTTP 404|The name is available, and that was not free|"
        r"will link to a 404)",
        re.I,
    )
    npm_link = re.compile(r"npmjs\.com/package/memvara|registry\.npmjs\.org/memvara")
    offenders = []
    seen: set[pathlib.Path] = set()
    for root in DOCS:
        for path in root.glob("*.md"):
            if path in seen:
                continue
            seen.add(path)
            text = path.read_text(encoding="utf-8")
            if unclaimed.search(text) and npm_link.search(text):
                offenders.append(str(path.relative_to(REPO)))
    assert offenders == []


def test_no_public_doc_still_calls_the_package_a_name_reservation():
    """It was one until 0.1.0 and is not one now. A reader who finds that sentence
    concludes there is nothing to run, which is the opposite of true — and it is
    exactly the drift the 0.0.2 listing was corrected for a day earlier."""
    stale = re.compile(r"(name reservation|no JavaScript client|npm install memvara` returns)", re.I)
    offenders = []
    for root in DOCS:
        for path in root.glob("*.md"):
            # CHANGELOG records what *was* true, in entries about the versions it was
            # true for. Excluding it is not a loophole; rewriting history would be.
            if path.name == "CHANGELOG.md":
                continue
            if stale.search(path.read_text(encoding="utf-8")):
                offenders.append(str(path.relative_to(REPO)))
    assert offenders == [], f"these still describe the reservation: {offenders}"
