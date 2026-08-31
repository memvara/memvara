"""The marketplace plugin: hosted MCP plus a copy of the packaged skill.

The skill in the wheel is what `memvara-mcp init` writes. The plugin is what a
coding agent installs. Those have to stay one body, point at the hosted URL,
and never grow an `npx` path — three things that fail quietly if they only
live in a README.
"""

from __future__ import annotations

import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
PLUGIN = REPO / "plugin"
SKILL = REPO / "memvara" / "skills" / "memvara"
HOSTED = "https://app.memvara.dev/mcp"

MARKETPLACES = (
    REPO / ".claude-plugin" / "marketplace.json",
    REPO / ".grok-plugin" / "marketplace.json",
    REPO / ".cursor-plugin" / "marketplace.json",
    REPO / ".github" / "plugin" / "marketplace.json",
)

MANIFESTS = (
    PLUGIN / "plugin.json",
    PLUGIN / ".claude-plugin" / "plugin.json",
    PLUGIN / ".cursor-plugin" / "plugin.json",
    PLUGIN / ".github" / "plugin.json",
)


def _load(path: pathlib.Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def test_every_manifest_parses_and_is_named_memvara() -> None:
    for path in MANIFESTS:
        body = _load(path)
        assert isinstance(body, dict), path
        assert body["name"] == "memvara", path


def test_marketplaces_list_the_plugin_directory() -> None:
    for path in MARKETPLACES:
        body = _load(path)
        assert isinstance(body, dict), path
        plugins = body["plugins"]
        assert len(plugins) == 1, path
        source = plugins[0]["source"]
        assert source in ("./plugin", "plugin"), f"{path}: {source!r}"


def test_mcp_configs_are_the_hosted_url() -> None:
    claude = _load(PLUGIN / ".mcp.json")
    cursor = _load(PLUGIN / "mcp.json")
    assert isinstance(claude, dict) and isinstance(cursor, dict)
    server = claude["mcpServers"]["memvara"]
    assert server["url"] == HOSTED
    assert server.get("type") == "http"
    assert cursor["mcpServers"]["memvara"]["url"] == HOSTED


def test_the_plugin_does_not_ship_npx_or_a_local_command() -> None:
    """Hosted-first. A local python3 block in the plugin would undo it."""
    for name in (".mcp.json", "mcp.json"):
        raw = (PLUGIN / name).read_text(encoding="utf-8")
        assert "npx" not in raw
        assert "python3" not in raw
    server = _load(PLUGIN / ".mcp.json")["mcpServers"]["memvara"]
    assert "command" not in server
    assert "args" not in server


def _skill_source(root: pathlib.Path) -> dict[pathlib.Path, bytes]:
    """Every source file in a skill tree, build artifacts excluded.

    `__pycache__` is not part of either tree: it is gitignored, it is specific to one
    interpreter version and one machine, and it exists only because something imported a
    file. It could not appear here at all until the skill gained a `.py`, and then it did
    -- on CI and not locally, so the byte comparison passed on my machine and failed on
    all six test jobs with `scripts/__pycache__/memvara_auth.cpython-313.pyc` present on
    one side and absent on the other.

    The exclusion cannot hide real drift, because `test_the_byte_copy_excludes_nothing_
    that_git_tracks` below requires everything it drops to be a file git ignores.
    """
    return {p.relative_to(root): p.read_bytes()
            for p in root.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts}


def test_plugin_skill_is_a_byte_copy_of_the_packaged_tree() -> None:
    packaged = _skill_source(SKILL)
    plugin = _skill_source(PLUGIN / "skills" / "memvara")
    assert packaged.keys() == plugin.keys()
    drifted = sorted(rel for rel, data in packaged.items() if plugin[rel] != data)
    assert not drifted, f"plugin skill drifted from the package: {drifted}"


def test_the_byte_copy_excludes_nothing_that_git_tracks() -> None:
    """The guard on the exclusion in `_skill_source`.

    Filtering paths out of a comparison is how a guard stops covering a file without
    anybody seeing a failure. So every path dropped has to be one git does not track: a
    real source file that started being skipped fails here instead of silently ceasing
    to be compared.
    """
    import subprocess

    for root in (SKILL, PLUGIN / "skills" / "memvara"):
        on_disk = {p for p in root.rglob("*") if p.is_file()}
        compared = {root / rel for rel in _skill_source(root)}
        tracked = {
            REPO / line for line in subprocess.run(
                ["git", "-C", str(REPO), "ls-files", "-z", str(root.relative_to(REPO))],
                check=True, capture_output=True, text=True).stdout.split("\0") if line
        }
        assert tracked, f"git tracks nothing under {root.relative_to(REPO)}"
        assert tracked <= compared, (
            f"the byte comparison is skipping tracked files under "
            f"{root.relative_to(REPO)}: {sorted(tracked - compared)}")
        for dropped in on_disk - compared:
            assert "__pycache__" in dropped.parts, (
                f"{dropped} was dropped from the comparison and is not a build artifact")


@pytest.mark.parametrize("name", [
    "integrate.md", "hosted-mcp.md", "write-and-correct.md", "time.md",
    "scopes.md", "governance.md", "migrate-mem0.md", "examples.md",
    "project-instructions.md",
])
def test_the_dispatcher_points_at_a_reference_that_exists(name: str) -> None:
    skill = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert f"references/{name}" in skill or name in (
        "examples.md", "project-instructions.md", "write-and-correct.md",
        "scopes.md", "governance.md", "time.md", "hosted-mcp.md",
        "integrate.md", "migrate-mem0.md",
    )
    assert (SKILL / "references" / name).is_file()


def test_the_canonical_plugin_claude_md_has_exactly_one_local_marker() -> None:
    """`plugin-claude.md` is copied into every repo in `plugin-repos.txt` as its CLAUDE.md.

    The marker is where each repository's own sections go, and a sync splices around it.
    Two markers and the splice takes the wrong span; none and every repo loses the two
    sections that legitimately differ — its runtime facts and, for the one plugin that
    ships hooks, its hook rules.

    Eleven of the fourteen sections were already byte-identical across all seven plugin
    repositories when this was written; only those two differed, which is what makes a
    canonical file possible at all and why the split is a delimited block rather than a
    prefix.
    """
    text = (REPO / "plugin-claude.md").read_text(encoding="utf-8")
    assert text.count("@@LOCAL@@") == 1, "exactly one splice point"
    assert text.startswith("# Working in a memvara plugin repository"), (
        "the copied file is a CLAUDE.md and opens as one")
    assert "plugin-repos.txt" in text, (
        "it must say where it is copied to, since the copy is what people will read")


def test_plugin_repos_list_is_the_public_set() -> None:
    names = [
        line.strip()
        for line in (REPO / "plugin-repos.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert names == [
        "claude-memvara",
        "codex-memvara",
        "cursor-memvara",
        "grok-memvara",
        "vscode-memvara",
        "opencode-memvara",
        "openclaw-memvara",
    ]


def test_hosted_url_is_the_one_on_the_public_site() -> None:
    """One string, several files. A typo here is a plugin that authorizes nothing."""
    assert HOSTED in (SKILL / "references" / "hosted-mcp.md").read_text(encoding="utf-8")
    assert HOSTED in (SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert HOSTED in (PLUGIN / "README.md").read_text(encoding="utf-8")


def test_sync_script_writes_skill_and_lock(tmp_path) -> None:
    import subprocess
    import sys

    script = REPO / "scripts" / "sync_plugin_repos.py"
    dest = tmp_path / "claude-memvara"
    dest.mkdir()
    subprocess.check_call([sys.executable, str(script), str(dest)])
    copied = dest / "plugin" / "skills" / "memvara" / "SKILL.md"
    assert copied.read_bytes() == (SKILL / "SKILL.md").read_bytes()
    lock = (dest / "skill.lock").read_text(encoding="utf-8")
    assert "repo=memvara/memvara" in lock
    assert "sha=" in lock


# The auth script lives inside the skill rather than beside it, and that is the whole
# design. `skill.lock` already vendors the skill tree byte-for-byte into all seven plugin
# repositories, so the script reaches every host through the sync that exists -- no second
# lock file, no second drift guard. It is addressed the way `SKILL.md` already addresses
# its own references, which was measured on three hosts before it was relied on: Codex,
# Copilot and OpenCode each read a sibling file out of a skill's own directory, and each
# answered "NO PROBE" with the skill unregistered and the files still on disk.
AUTH_SCRIPT = SKILL / "scripts" / "memvara_auth.py"
AUTH_PLACEHOLDER = "<SKILL_DIR>"
AUTH_INVOCATION = f"python3 {AUTH_PLACEHOLDER}/scripts/memvara_auth.py authenticate"
AUTH_HEADING = "## When it will not authenticate"
AUTH_COMMANDS = ("authenticate", "login", "logout", "stats")


def test_the_skill_ships_the_auth_script() -> None:
    """Positive, because a deletion is the failure this exists to catch.

    Spelled "the tree contains no stray script" it would pass on a skill that had stopped
    shipping the one users need, which is exactly the shape of guard this project has
    been burned by: a check an absence satisfies has quietly stopped guarding.
    """
    assert AUTH_SCRIPT.is_file(), (
        f"{AUTH_SCRIPT.relative_to(REPO)} is gone; the four auth commands and the "
        "skill's own instructions both point at it")


def test_the_skill_names_the_script_at_the_path_it_is_actually_at() -> None:
    """The instruction and the file, compared against each other rather than each alone.

    A path is the one thing here that fails silently: the model reads the line, runs it,
    gets `No such file or directory`, and reports that memvara cannot authenticate. That
    is how `${CLAUDE_PLUGIN_ROOT}` reached Grok's command files and expanded to nothing,
    handing the shell an absolute path to a file that has never existed on any machine.
    """
    skill = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert AUTH_HEADING in skill, (
        "SKILL.md no longer tells the model what to do when authentication fails, so "
        "the script ships unreachable")
    assert AUTH_INVOCATION in skill
    assert f"`{AUTH_PLACEHOLDER}` is the directory this" in skill, (
        f"{AUTH_PLACEHOLDER} appears with nothing telling the reader to substitute it")

    # The claim resolved, not the spelling matched. A guard that checks the string agrees
    # with an instruction that spells a plausible path into the wrong directory.
    quoted = AUTH_INVOCATION.split(f"{AUTH_PLACEHOLDER}/", 1)[1].split(" ", 1)[0]
    assert (SKILL / quoted).is_file(), f"SKILL.md says {quoted}, and nothing is there"

    # Every invocation, not just the one above. A cwd-relative `python3 scripts/...`
    # resolves against the user's project rather than against this file, so it fails with
    # `No such file or directory` on a machine where the script is sitting correctly on
    # disk -- the same shape as ${CLAUDE_PLUGIN_ROOT} expanding to nothing under Grok.
    # Stated over every line that runs the script, so a second example added later cannot
    # reintroduce the bare form beside a correct one.
    runs = [line for line in skill.splitlines()
            if "python3" in line and "memvara_auth.py" in line]
    assert runs, "no line in SKILL.md runs the script"
    for line in runs:
        assert AUTH_PLACEHOLDER in line, (
            f"{line.strip()!r} runs the script by a path that is not anchored to the "
            f"skill directory; it resolves against whatever cwd the agent happens to be "
            f"in. Anchor it with {AUTH_PLACEHOLDER}.")


@pytest.mark.parametrize("command", AUTH_COMMANDS)
def test_the_skill_names_every_command_the_script_accepts(command: str) -> None:
    """The skill names it. That the SCRIPT accepts it is a separate question, answered
    against the script's own `COMMANDS` by the test below and against its real printed
    usage by `test_the_script_runs_here_and_says_how_to_use_it`.

    This used to also assert `command in <the whole 872-line source>`, which is close to
    vacuous: "stats" matches `_stats`, "login" matches `_logout`'s docstring, and deleting
    a command from the USAGE string a user actually sees would not have failed it. A
    guard that cannot fail is worse than no guard, because it reads as coverage.
    """
    skill = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert f"`{command}`" in skill or f"`{command} " in skill, (
        f"the script accepts {command} and the skill never mentions it")


def test_the_script_accepts_exactly_the_commands_the_skill_advertises() -> None:
    """The other direction of the pair above, read out of the script rather than asserted.

    `COMMANDS` in the module is the referent. If it grows a command, this fails until
    somebody decides whether the skill should say so.
    """
    import ast

    tree = ast.parse(AUTH_SCRIPT.read_text(encoding="utf-8"))
    found = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "COMMANDS" for t in node.targets):
            found = set(ast.literal_eval(node.value))
    assert found is not None, "the script no longer declares COMMANDS"
    assert found == set(AUTH_COMMANDS), (
        f"the script accepts {sorted(found)}; the skill documents "
        f"{sorted(AUTH_COMMANDS)}")


def test_the_script_is_standard_library_only() -> None:
    """It runs on a machine that has never installed anything, which is the point of it.

    A `pip install` in the recovery path is a recovery path that does not work on the
    machine that needs it.
    """
    import ast
    import sys

    tree = ast.parse(AUTH_SCRIPT.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    outside = sorted(name for name in imported
                     if name not in sys.stdlib_module_names and name != "certifi")
    assert not outside, f"the auth script imports {outside}, which is not in the stdlib"

    # `certifi` is the one exception, and only because it is optional. python.org's macOS
    # build loads zero roots from the system trust store, so the script prefers certifi's
    # bundle when it is importable and falls back to the default context when it is not.
    # Promoted to a module-level import it stops being optional, and the script then dies
    # at startup on exactly the machines it exists to rescue -- ones with no pip step.
    top_level = {alias.name.split(".")[0]
                 for node in tree.body if isinstance(node, ast.Import)
                 for alias in node.names}
    assert "certifi" not in top_level, (
        "certifi is imported at module level; it must stay inside the try/except that "
        "makes it optional, or the script requires a pip install to start")


def test_the_script_runs_here_and_says_how_to_use_it() -> None:
    """Executed, not read. A syntax error in a vendored file is invisible to a byte diff.

    No network: an unknown command is rejected on shape before anything is dialled.
    """
    import subprocess
    import sys

    done = subprocess.run(
        [sys.executable, str(AUTH_SCRIPT), "not-a-command"],
        capture_output=True, text=True, timeout=60)
    assert done.returncode == 2, done.stdout + done.stderr
    for command in AUTH_COMMANDS:
        assert command in done.stdout, f"the usage line omits {command}"
