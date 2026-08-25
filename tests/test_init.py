"""`memvara init`, and the two promises the skill file makes to everything around it.

Three kinds of test live here and only the first is about code.

**This command writes files onto somebody's disk**, so most of what follows is about
files that were already there: an `.mcp.json` naming three other servers, a CLAUDE.md
somebody has been curating for a year, a skill left by an older version. Each of those is
a chance to destroy work that is not ours, and none of them is on the happy path — which
is the only path a manual test ever takes.

**The configuration it writes has to be the configuration that works.** `MEMVARA_DB`
absolute, the store's directory present before sqlite is asked to open a file in it, and
`command` naming an interpreter that provably imports memvara rather than whichever
`python3` a desktop client's `PATH` turns up. Every one of those is a support thread that
already exists somewhere; the point of the command is that it stops being written.

**The skill is prose, and prose is checkable in the two ways that matter here.** It must
carry the five things no single tool description can, and it must not restate the ones
they already carry — a second copy of a tool description is not redundancy, it is a second
source, and the copy in the skill is the one that drifts because it is the one nobody
edits when the tool changes. The n-gram test below is crude and is the only mechanical
guard there is against that.
"""

from __future__ import annotations

import io
import json
import os
import pathlib
import re
import subprocess
import sys
import zipfile
from typing import Iterable, Sequence

import pytest

import memvara
from memvara.server import main
from memvara.server.config import EXAMPLE_CONFIG
from memvara.server.init import (
    AGENTS,
    INIT_USAGE,
    MARKER,
    _default_db,
    client_entry,
    cloud_client_entry,
    init,
    skill_text,
)
from memvara.server.tools import BY_NAME, TOOLS

REPO = pathlib.Path(__file__).resolve().parent.parent
SKILL_SOURCE = REPO / "memvara" / "skills" / "memvara" / "SKILL.md"
SKILL_PACKAGE = REPO / "memvara" / "skills" / "memvara"


def _run(root: pathlib.Path, *extra: str,
         env: dict[str, str] | None = None) -> tuple[int, str, str]:
    """`init` into `root`, with a store under it, and both streams captured.

    Every test passes `--dir` and `--db` explicitly. The defaults are the user's home
    directory and the working directory, and a suite that exercised them would write into
    whichever of those the developer happened to be sitting in.
    """
    out, err = io.StringIO(), io.StringIO()
    argv = ["--agent", "claude", "--dir", str(root), "--mode", "local",
            "--db", str(root / "store" / "memory.db"), *extra]
    status = init(argv, env={} if env is None else env, stdout=out, stderr=err)
    return status, out.getvalue(), err.getvalue()


def _skill_path(root: pathlib.Path) -> pathlib.Path:
    return root / ".claude" / "skills" / "memvara" / "SKILL.md"


def _entry(root: pathlib.Path) -> dict[str, object]:
    written = json.loads((root / ".mcp.json").read_text(encoding="utf-8"))
    entry: dict[str, object] = written["mcpServers"]["memvara"]
    return entry


# -- what a first run leaves behind ---------------------------------------------------


def test_one_run_writes_the_three_files_a_client_looks_for(tmp_path) -> None:
    """The whole point of the subcommand: no hand-editing, and nothing left to do.

    Asserted as three separate reads rather than by comparing the report text, because
    what matters is that the files are where Claude Code goes looking — the report is how
    the user finds out, and it is checked separately below.
    """
    status, out, err = _run(tmp_path, env={"USER": "alice"})

    assert status == 0 and err == ""
    assert _skill_path(tmp_path).read_text(encoding="utf-8") == skill_text()
    refs = tmp_path / ".claude" / "skills" / "memvara" / "references"
    assert (refs / "examples.md").is_file()
    assert (refs / "governance.md").is_file()
    assert (refs / "project-instructions.md").is_file()
    assert MARKER in (tmp_path / "CLAUDE.md").read_text(encoding="utf-8")
    entry = _entry(tmp_path)
    assert entry["args"] == ["-m", "memvara.server"]
    assert entry["env"] == {"MEMVARA_DB": str(tmp_path / "store" / "memory.db"),
                            "MEMVARA_USER": "alice"}
    for path in (_skill_path(tmp_path), tmp_path / ".mcp.json", tmp_path / "CLAUDE.md"):
        assert str(path) in out, "every path it touched is named in the report"


def test_the_store_path_is_absolute_however_it_arrived(tmp_path, monkeypatch) -> None:
    """The requirement that costs everyone their second attempt.

    A client launches the server from a working directory nobody chose, so a relative
    `MEMVARA_DB` names a different file every time — or none, and the store the user has
    been talking to for a week appears empty. `init` is the only participant in the whole
    exchange that knows the current directory, so it is the one that has to resolve it.
    """
    monkeypatch.chdir(tmp_path)
    out, err = io.StringIO(), io.StringIO()
    assert init(["--agent", "claude", "--dir", str(tmp_path), "--mode", "local",
                "--db", "memory.db"], env={}, stdout=out, stderr=err) == 0

    written = _entry(tmp_path)["env"]["MEMVARA_DB"]  # type: ignore[index]
    assert isinstance(written, str)
    assert os.path.isabs(written) and os.path.basename(written) == "memory.db"


def test_the_store_directory_is_created_and_the_store_is_not(tmp_path) -> None:
    """Half of a fix, deliberately.

    `sqlite3.connect()` does not create a missing directory — it raises "unable to open
    database file", from inside a server nobody is watching, naming neither the directory
    nor the remedy. The file itself stays the server's to create, so an `init` that is
    never followed by a launch leaves an empty directory rather than a database with an
    embedder fingerprint already written into it.
    """
    _run(tmp_path)

    assert (tmp_path / "store").is_dir()
    assert not (tmp_path / "store" / "memory.db").exists()


def test_the_command_is_the_interpreter_that_can_import_memvara(tmp_path) -> None:
    """`python3` is the documented example and the wrong thing to write down.

    The client executes `command` with no shell and no login profile, so `python3`
    resolves against a `PATH` that is not the terminal's. For anyone who installed into a
    virtualenv that is a different interpreter, and the failure — `No module named
    memvara` — surfaces inside an application that shows no logs. The interpreter running
    this command imported memvara a moment ago, which is the only proof available.
    """
    _run(tmp_path)

    assert _entry(tmp_path)["command"] == sys.executable


def test_an_interpreter_that_cannot_name_itself_falls_back_to_python3(
        tmp_path, monkeypatch) -> None:
    """`sys.executable` is empty in an embedded interpreter, and empty is unusable."""
    monkeypatch.setattr(sys, "executable", "")
    _run(tmp_path)

    assert _entry(tmp_path)["command"] == "python3"


def test_the_written_entry_and_the_documented_example_do_not_drift(tmp_path) -> None:
    """`EXAMPLE_CONFIG` is what the server prints when it refuses to start.

    Two blocks describing the same launch, produced by different code, is exactly the
    two-sources problem this repository keeps finding. They cannot be one string — the
    example has placeholders where this has a real path — so the test is that they agree
    on shape: same keys, same argv, same environment variables.
    """
    _run(tmp_path, env={"USER": "alice"})

    example = json.loads(EXAMPLE_CONFIG)["mcpServers"]["memvara"]
    written = _entry(tmp_path)
    assert written.keys() == example.keys()
    assert written["args"] == example["args"]
    assert written["env"].keys() == example["env"].keys()  # type: ignore[union-attr]


# -- files that were already there ----------------------------------------------------


def test_an_existing_mcp_json_is_never_rewritten(tmp_path) -> None:
    """It is the client's file and usually names other people's servers.

    Merging means round-tripping somebody else's JSON through `json.dumps`, and a merge
    that goes wrong presents as a client that will not start with nothing on screen
    explaining why. The entry is printed instead: one paste, and no way to lose anything.
    """
    original = '{ "mcpServers": { "other": { "command": "elsewhere" } } }\n'
    (tmp_path / ".mcp.json").write_text(original, encoding="utf-8")

    status, out, _ = _run(tmp_path)

    assert status == 0
    assert (tmp_path / ".mcp.json").read_text(encoding="utf-8") == original
    assert "kept" in out and "mcpServers" in out
    # Compared as JSON rather than as a path: the block printed for pasting *is* JSON, so
    # a Windows path arrives with its separators escaped (`C:\\Users\\…`) and never
    # matches `str(WindowsPath)`. `json.dumps` is the same transform the block went
    # through, quotes included, which also pins that the value is a complete string
    # rather than a prefix of a longer one.
    assert json.dumps(str(tmp_path / "store" / "memory.db")) in out, (
        "the block to paste is complete")


def test_an_existing_mcp_json_that_already_names_memvara_says_so(tmp_path) -> None:
    """A rerun after a successful one has nothing to tell the user to paste."""
    _run(tmp_path)
    status, out, _ = _run(tmp_path)

    assert status == 0
    assert "already names a 'memvara' server" in out
    assert "Add this inside" not in out


def test_the_claude_md_section_is_appended_once_however_often_you_run(tmp_path) -> None:
    """A snippet is append-shaped, and appending twice is the obvious way to get it wrong.

    The marker is the guard rather than a search for the word "memvara", because a project
    *about* memvara says it on every other line and would never receive its snippet.
    """
    (tmp_path / "CLAUDE.md").write_text("# Project\n\nHouse rules.\n", encoding="utf-8")

    _run(tmp_path)
    _run(tmp_path)

    body = (tmp_path / "CLAUDE.md").read_text(encoding="utf-8")
    assert body.count(MARKER) == 1
    assert body.startswith("# Project\n\nHouse rules.\n"), "their file, still theirs"


@pytest.mark.parametrize("existing, expected", [
    ("# Project\n", f"# Project\n\n{MARKER}"),
    ("# Project", f"# Project\n\n{MARKER}"),
    ("", MARKER),
])
def test_the_section_is_separated_from_whatever_the_file_ended_with(
        tmp_path, existing: str, expected: str) -> None:
    """One blank line before the heading, three ways for the file to have ended.

    Without the trailing-newline case the heading lands on the end of whatever sentence
    they stopped mid-way through; without the empty case a file someone touched and never
    filled in opens with two blank lines.
    """
    (tmp_path / "CLAUDE.md").write_text(existing, encoding="utf-8")

    _run(tmp_path)

    assert (tmp_path / "CLAUDE.md").read_text(encoding="utf-8").startswith(expected)


def test_a_skill_that_differs_is_kept_until_force_says_otherwise(tmp_path) -> None:
    """The one file this command owns, and even that is not taken silently.

    It is generated data that ships in the wheel and versions with the tools it describes,
    so replacing it on upgrade is the intended path. A user who edited theirs still gets
    told rather than overwritten, and the message names the flag that does it.
    """
    path = _skill_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("mine, edited\n", encoding="utf-8")

    status, out, _ = _run(tmp_path)
    assert status == 0
    assert path.read_text(encoding="utf-8") == "mine, edited\n"
    assert "--force replaces it" in out

    status, out, _ = _run(tmp_path, "--force")
    assert status == 0
    assert path.read_text(encoding="utf-8") == skill_text()
    assert "replaced" in out


def test_rerunning_is_boring(tmp_path) -> None:
    """The property a one-command installer needs most: it is safe to run again.

    Someone who is not sure whether it worked will run it a second time, and on a machine
    where it did work the honest answer is a column of `kept` and exit 0.
    """
    _run(tmp_path)
    status, out, err = _run(tmp_path)

    assert (status, err) == (0, "")
    assert "already this version" in out
    assert "wrote" not in out and "appended" not in out


# -- the command line -----------------------------------------------------------------


def test_help_describes_the_files_rather_than_the_flags(tmp_path) -> None:
    out = io.StringIO()

    assert init(["--help"], env={}, stdout=out, stderr=io.StringIO()) == 0

    body = out.getvalue()
    assert ".mcp.json" in body and "SKILL.md" in body and "CLAUDE.md" in body
    assert "npm package is a placeholder" in body, "no parity is implied anywhere"


def test_the_short_help_flag_works_too() -> None:
    out = io.StringIO()
    assert init(["-h"], env={}, stdout=out, stderr=io.StringIO()) == 0
    assert out.getvalue() == INIT_USAGE + "\n"


def test_with_nothing_injected_it_uses_the_process_streams(capsys) -> None:
    """`python3 -m memvara.server init --help` is a real invocation and takes this path."""
    assert init(["--help"]) == 0
    assert ".mcp.json" in capsys.readouterr().out


@pytest.mark.parametrize("argv, expected", [
    ([], "needs --agent"),
    (["--agent", "windsurf"], "not a client init can write files for"),
    (["--agent"], "--agent needs a value"),
    (["--agent="], "--agent needs a value"),
    (["--agent", "claude", "--user", "   "], "--user needs a value"),
    (["--agent", "claude", "--port", "8080"], "unexpected argument '--port'"),
    (["--agent", "claude", "--mode", "local", "--db", ":memory:"],
     "remember nothing between launches"),
])
def test_a_wrong_command_line_exits_two_and_names_the_part_that_was_wrong(
        argv: Sequence[str], expected: str) -> None:
    """Exit 2 throughout, matching the server: the invocation was wrong, not the program."""
    err = io.StringIO()

    assert init(list(argv), env={}, stdout=io.StringIO(), stderr=err) == 2

    body = err.getvalue()
    assert expected in body
    assert INIT_USAGE in body, "and the usage, because they are about to retype it"


def test_a_directory_that_does_not_exist_is_a_usage_error(tmp_path) -> None:
    """`init` creates files inside a project; it does not decide where the project is.

    A typo'd `--dir` would otherwise build a plausible-looking `.claude/skills/` tree one
    character away from the one the client reads, and nothing would ever say so.
    """
    err = io.StringIO()

    status = init(["--agent", "claude", "--dir", str(tmp_path / "nope")],
                  env={}, stdout=io.StringIO(), stderr=err)

    assert status == 2
    assert "is not a directory" in err.getvalue()


def test_a_throwaway_store_in_the_environment_is_refused_the_same_way(tmp_path) -> None:
    """`MEMVARA_DB=:memory:` in the shell is how the smoke test is run, and it is sticky.

    Picking it up would write a settings file whose store dies with every launch, which
    looks exactly like a memory server that works until the second session.
    """
    err = io.StringIO()

    status = init(["--agent", "claude", "--dir", str(tmp_path), "--mode", "local"],
                  env={"MEMVARA_DB": ":memory:"}, stdout=io.StringIO(), stderr=err)

    assert status == 2
    assert "--db" in err.getvalue()


def test_option_values_may_be_joined_with_an_equals_sign(tmp_path) -> None:
    """Both spellings, because both are what people type."""
    out = io.StringIO()
    status = init(["--agent=claude", f"--dir={tmp_path}", "--mode=local",
                   f"--db={tmp_path / 'memory.db'}", "--user=alice"],
                  env={}, stdout=out, stderr=io.StringIO())

    assert status == 0
    assert _entry(tmp_path)["env"] == {"MEMVARA_DB": str(tmp_path / "memory.db"),
                                       "MEMVARA_USER": "alice"}


# -- who the server remembers for -----------------------------------------------------


@pytest.mark.parametrize("env, expected", [
    ({"MEMVARA_USER": "from-memvara"}, "from-memvara"),
    ({"USER": "from-posix"}, "from-posix"),
    ({"USERNAME": "from-windows"}, "from-windows"),
    ({"MEMVARA_USER": "wins", "USER": "loses"}, "wins"),
])
def test_the_user_is_taken_from_the_first_variable_that_has_one(
        tmp_path, env: dict[str, str], expected: str) -> None:
    """Three names for one idea, and the machine only ever sets one of them."""
    _run(tmp_path, env=env)

    assert _entry(tmp_path)["env"]["MEMVARA_USER"] == expected  # type: ignore[index]


def test_the_flag_beats_the_environment(tmp_path) -> None:
    _run(tmp_path, "--user", "asked-for", env={"USER": "logged-in"})

    assert _entry(tmp_path)["env"]["MEMVARA_USER"] == "asked-for"  # type: ignore[index]


def test_an_unknown_user_is_reported_rather_than_invented(tmp_path) -> None:
    """Omitting the variable means "the whole tenant", which is a real answer.

    Writing a guessed name instead would partition the store under something the user
    never chose and cannot see, and they would find out by losing everything the day the
    guess changed.
    """
    status, out, _ = _run(tmp_path)

    assert status == 0
    assert "MEMVARA_USER" not in _entry(tmp_path)["env"]  # type: ignore[operator]
    assert "whole tenant" in out


def test_the_default_store_prefers_this_shell_then_the_home_directory() -> None:
    """Called directly: the fallback is under `$HOME` and a test must not write there.

    The environment is consulted first so that someone who already has a working server
    and is running `init` to pick up the skill is not handed a second, empty store that
    answers none of the questions the first one could.
    """
    assert _default_db({"MEMVARA_DB": "  /srv/memory.db  "}) == "/srv/memory.db"

    fallback = _default_db({})
    assert os.path.isabs(fallback)
    assert fallback.endswith(os.path.join(".memvara", "memory.db"))


def test_client_entry_omits_the_user_rather_than_writing_an_empty_one() -> None:
    """An empty `MEMVARA_USER` is not the same as an absent one — see `config._optional`."""
    assert client_entry(db="/s.db", user=None, command="p")["env"] == {"MEMVARA_DB": "/s.db"}


# -- reached through the server's own entry point --------------------------------------


def test_the_subcommand_is_dispatched_before_the_flags(tmp_path) -> None:
    """`memvara-mcp init ...` has to reach `init` without the server opening a store."""
    out, err = io.StringIO(), io.StringIO()

    status = main(["init", "--agent", "claude", "--dir", str(tmp_path),
                   "--db", str(tmp_path / "memory.db")],
                  env={}, stdout=out, stderr=err)

    assert (status, err.getvalue()) == (0, "")
    assert _skill_path(tmp_path).is_file()


def test_an_init_usage_error_still_exits_two_through_main() -> None:
    err = io.StringIO()
    assert main(["init"], env={}, stdout=io.StringIO(), stderr=err) == 2
    assert "needs --agent" in err.getvalue()


def test_the_servers_own_help_points_at_the_subcommand() -> None:
    """Someone reading `--help` is there because a client said the server failed.

    That is the moment to mention there is a command which writes the block correctly,
    and it is the only place in the program where the two texts can be connected.
    """
    out = io.StringIO()
    assert main(["--help"], stdout=out) == 0
    assert "init --agent claude" in out.getvalue()


# -- the skill as a shipped file --------------------------------------------------------


def test_the_skill_is_read_out_of_the_installed_package(tmp_path) -> None:
    """`skill_text()` goes through `importlib.resources`, not through this repository.

    Which is the difference between "the file is in the source tree" and "the file is in
    the wheel the user installed", and only the second one makes `memvara init` work.
    """
    assert skill_text() == SKILL_SOURCE.read_text(encoding="utf-8")
    assert skill_text().startswith("---\n"), "front matter is what makes it a skill"


def test_every_agent_init_knows_gets_the_same_packaged_skill() -> None:
    """`--agent` is an inventory of file layouts, not of different prose."""
    for agent in AGENTS:
        assert skill_text(agent) == skill_text(), f"--agent {agent} drifted from the package"


def test_the_skill_is_not_hidden_from_the_build_by_an_ignore_rule() -> None:
    """The thread the wheel's contents actually hang on.

    Hatchling sweeps `memvara/` and drops whatever a VCS ignore rule matches — the same
    thin thread `py.typed` hangs on, except that this file's absence does not fail an
    import in CI. It fails `memvara init` on a stranger's machine, at the one moment they
    are deciding whether the package works. `artifacts` in `pyproject.toml` is the belt;
    this is the check that the braces are still on.
    """
    ignored = subprocess.run(["git", "check-ignore", "-q", str(SKILL_SOURCE)],
                             cwd=REPO, capture_output=True)
    assert ignored.returncode != 0, f"{SKILL_SOURCE} matches a VCS ignore rule"


def test_the_build_declares_the_skill_and_the_declaration_matches_the_tree() -> None:
    """A pattern that no longer matches anything is indistinguishable from no pattern."""
    patterns = re.findall(r'^artifacts = \[(.*)\]$',
                          (REPO / "pyproject.toml").read_text(encoding="utf-8"),
                          flags=re.MULTILINE)
    assert patterns, "pyproject.toml no longer names the skill as a build artifact"
    globs = [item.strip().strip("\"'") for item in patterns[0].split(",")]
    matched = {path for glob in globs for path in REPO.glob(glob)}
    assert SKILL_SOURCE in matched, f"{globs} does not match {SKILL_SOURCE}"


@pytest.mark.skipif(not sorted((REPO / "dist").glob(f"memvara-{memvara.__version__}-*.whl")),
                    reason="no dist/*.whl; run python3 -m build --wheel")
def test_the_skill_reaches_the_wheel() -> None:
    """Release-time, like the rest of the wheel checks: `dist/` is gitignored.

    It arms itself the moment `python3 -m build --wheel` has run, which `docs/RELEASING.md`
    puts immediately before this suite.
    """
    names: set[str] = set()
    for wheel in (REPO / "dist").glob(f"memvara-{memvara.__version__}-*.whl"):
        with zipfile.ZipFile(wheel) as archive:
            names |= set(archive.namelist())
    assert "memvara/skills/memvara/SKILL.md" in names
    assert "memvara/skills/memvara/references/examples.md" in names
    assert "memvara/skills/memvara/references/governance.md" in names
    assert "memvara/skills/memvara/references/project-instructions.md" in names


# -- the skill as prose ------------------------------------------------------------------


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _ngrams(text: str, n: int = 6) -> set[tuple[str, ...]]:
    words = _words(text)
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def _tool_prose() -> Iterable[tuple[str, str]]:
    for tool in TOOLS:
        yield tool.name, tool.description
        for name, schema in tool.properties.items():
            yield f"{tool.name}.{name}", str(schema.get("description", ""))


def test_the_skill_does_not_restate_a_tool_description() -> None:
    """The one prohibition the spec is explicit about, enforced the only way prose can be.

    Six consecutive words in common is a quotation rather than a coincidence — English
    collocations collide at three and four, not at six — so this is deliberately blunt: it
    catches paste, and it catches the tempting kind of paste, which is a sentence lifted
    from a description because it was already well written. That sentence is exactly the
    one that will still be here after the description it came from has been rewritten.

    The rule covers parameter descriptions as well as tool ones. `close`'s description
    argues the ended/retired distinction at length and is the natural thing to copy; the
    skill's job is to say that the *source turn* decides which one applies, which is a
    fact about the sequence and nothing a parameter description could carry.
    """
    packaged = skill_text()
    for extra in sorted((SKILL_PACKAGE / "references").glob("*.md")):
        packaged += "\n" + extra.read_text(encoding="utf-8")
    skill = _ngrams(packaged)
    offenders = {source: sorted(" ".join(g) for g in skill & _ngrams(prose))
                 for source, prose in _tool_prose() if skill & _ngrams(prose)}
    assert not offenders, (
        f"the skill repeats {offenders} — cut it from the skill, or if the description is "
        "the wrong place for it, move it rather than keeping both")


@pytest.mark.parametrize("carries, marker", [
    ("the correction protocol as a sequence", "memory_why"),
    ("the scope trap only the CLI help mentions", "MEMVARA_SESSION"),
    ("what a degraded extractor is called", "fast-path-only"),
    ("the judgment call about what to store", "embarrassing"),
    ("never asserting a memory unread", "in the current\nturn"),
    # Not a closure *word* any more: the two closures are two tools, so what the skill
    # has to carry is the judgment that picks between them — the excerpt from step 3 is
    # the evidence, not the user's phrasing. That is the one thing no tool description
    # can say, because it is about a different tool's output.
    ("which closure the evidence decides", "evidence for"),
])
def test_the_skill_carries_what_no_single_description_can(carries: str, marker: str) -> None:
    """The spec's list of five, one assertion each, so a failure names what went missing.

    Keyword assertions on prose are weak tests of quality and strong tests of presence,
    and presence is the property at risk: a skill gets edited for length, and the first
    thing to go is whichever paragraph the editor did not have the spec in front of.
    """
    assert marker in skill_text(), f"the skill no longer carries {carries}"


_NUMBER_WORDS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
)


def test_the_skill_states_the_tool_surface_and_names_all_of_it() -> None:
    """`references/hosted-mcp.md` enumerates the tools. Nothing checked it, and it rotted.

    It said "## The ten tools" and listed ten, having never gained `memory_neighborhood`
    or `memory_paths`, and then not `memory_standing` either. That page ships vendored into
    seven plugin repositories, so one unguarded sentence here was wrong in eight places at
    once.

    Reading `TOOLS` is right *here* and would be wrong downstream. The plugin repos
    deliberately do not reach into this library — a test that read a sibling working tree
    would fail on a stale checkout. In this repository the list and its source are the same
    tree, so there is no checkout to be stale.

    Both halves are asserted because the incident that motivated the equivalent guard on
    the website was a missing *name* rather than a wrong number: a list one short of its own
    stated count agrees with itself perfectly. Order is asserted too, so the page reads in
    the order the server declares.
    """
    page = (SKILL_PACKAGE / "references" / "hosted-mcp.md").read_text(encoding="utf-8")
    word = _NUMBER_WORDS[len(TOOLS)]

    assert f"The {word} tools" in page, (
        f"the page must state the count as 'The {word} tools'; stating it positively is "
        "what makes a deleted sentence fail as loudly as a wrong one")

    listed = re.findall(r"`(memory_[a-z_]+)`", page)
    seen, ordered = set(), []
    for name in listed:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    assert ordered == [t.name for t in TOOLS], (
        "the page must name every tool, once, in the order the server declares them; "
        f"got {ordered}")


def test_every_tool_the_skill_names_is_one_the_server_serves() -> None:
    """The skill routes a model between tools by name; `TOOLS` decides which exist.

    Two files, one vocabulary — the shape of the bug this whole line of work started
    from, where a tool description said "retire" for an operation that ends. A skill
    naming a tool that is not served is the same mistake with a longer fuse: the model
    follows the instruction, the call fails, and the turn is spent finding that out.

    It bites hardest on the correction protocol, which is the one part of the skill that
    names three tools in sequence and is worth nothing if any of them is wrong.
    """
    named = set(re.findall(r"`(memory_[a-z_]+)`", skill_text()))

    assert named, "the skill no longer names a tool at all"
    assert named <= set(BY_NAME), f"the skill names {named - set(BY_NAME)}, which is not served"
    assert {"memory_forget", "memory_remember"} <= named, (
        "the correction protocol needs both halves: a replacement says the world changed "
        "and only memory_forget says the record was wrong")


def test_the_usage_names_a_command_that_exists() -> None:
    """The help text is what a user retypes, so it has to name the real console script.

    It said `memvara init`, and there is no `memvara` command — `[project.scripts]`
    declares `memvara-mcp` and nothing else, so anyone copying the first line of the
    usage got "command not found" at the one moment they were following instructions.
    The same class as a README naming an unexported symbol: prose that resolves to
    nothing, invisible to every test that only compares the string against itself.

    Read out of `pyproject.toml` rather than hard-coded, so renaming the script fails
    here instead of silently making the usage wrong again.
    """
    root = pathlib.Path(__file__).resolve().parent.parent
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    scripts = re.findall(r"^([A-Za-z0-9_-]+)\s*=",
                         pyproject.split("[project.scripts]")[1].split("[")[0],
                         flags=re.MULTILINE)

    assert scripts, "pyproject declares no console script"
    first = INIT_USAGE.splitlines()[0]
    assert any(first.startswith(f"{name} init") for name in scripts), (
        f"usage opens with {first!r}, which names no script in {scripts}")


def test_the_skill_directory_is_the_name_the_front_matter_declares() -> None:
    """Claude Code indexes a skill by its directory; a mismatch is a skill that is never
    loaded, and nothing anywhere reports it."""
    front = skill_text().split("---")[1]
    declared = re.search(r"^name:\s*(\S+)$", front, flags=re.MULTILINE)

    assert declared is not None
    assert declared.group(1) == _skill_path(pathlib.Path("/anywhere")).parent.name


def test_cursor_and_grok_write_their_own_skill_directory(tmp_path) -> None:
    """Same skill, different folder. The Claude path is not implied."""
    layouts = (
        ("cursor", pathlib.Path(".cursor") / "skills" / "memvara"),
        ("grok", pathlib.Path(".grok") / "skills" / "memvara"),
    )
    for agent, rel in layouts:
        dest = tmp_path / agent
        dest.mkdir()
        status, _, err = _run(dest, "--agent", agent, env={"USER": "alice"})
        assert status == 0 and err == ""
        assert (dest / rel / "SKILL.md").read_text(encoding="utf-8") == skill_text()
        assert (dest / rel / "references" / "examples.md").is_file()
        assert MARKER in (dest / "AGENTS.md").read_text(encoding="utf-8")
        assert not (dest / "CLAUDE.md").exists()
        assert not (dest / ".claude").exists()


def test_skill_only_writes_the_tree_and_leaves_mcp_json_alone(tmp_path) -> None:
    """Hosted-first users already have a URL in the client; they must not get a
    local stdio block they did not ask for."""
    status, out, err = _run(tmp_path, "--skill-only")
    assert status == 0 and err == ""
    assert _skill_path(tmp_path).is_file()
    assert (tmp_path / ".claude" / "skills" / "memvara" / "references"
            / "governance.md").is_file()
    assert not (tmp_path / ".mcp.json").exists()
    assert not (tmp_path / "store").exists()
    assert "skill only" in out
    assert "No .mcp.json" in out


def test_skill_only_does_not_touch_an_existing_mcp_json(tmp_path) -> None:
    original = '{ "mcpServers": { "other": { "command": "elsewhere" } } }\n'
    (tmp_path / ".mcp.json").write_text(original, encoding="utf-8")

    assert _run(tmp_path, "--skill-only")[0] == 0

    assert (tmp_path / ".mcp.json").read_text(encoding="utf-8") == original


def test_a_second_note_file_that_already_exists_is_updated_too(tmp_path) -> None:
    """A repo that already has AGENTS.md should get the pointer there as well."""
    (tmp_path / "AGENTS.md").write_text("# Agents\n", encoding="utf-8")

    _run(tmp_path)

    assert MARKER in (tmp_path / "CLAUDE.md").read_text(encoding="utf-8")
    assert MARKER in (tmp_path / "AGENTS.md").read_text(encoding="utf-8")
    assert (tmp_path / "AGENTS.md").read_text(encoding="utf-8").startswith("# Agents\n")


# -- cloud mode --------------------------------------------------------------------------


def test_explicit_local_mode_matches_todays_behaviour_byte_for_byte(tmp_path) -> None:
    """`--mode local` is the escape hatch back to the only behaviour this command had
    before cloud mode existed, and it has to still write MEMVARA_DB/MEMVARA_USER."""
    status, out, _ = _run(tmp_path, env={"USER": "alice"})

    assert status == 0
    assert _entry(tmp_path)["env"] == {
        "MEMVARA_DB": str(tmp_path / "store" / "memory.db"),
        "MEMVARA_USER": "alice",
    }
    assert "(cloud)" not in out


@pytest.fixture()
def cloud_wired(monkeypatch):
    """A world where the REST facade has grown the endpoints the engine needs.

    Everything `init` writes *for* cloud mode is unreachable while `config.cloud_gap()`
    is non-empty, because the command now refuses before it writes — the server it would
    configure cannot start, and exiting 0 with "restart your client" is how that used to
    reach the user.

    The cloud-writing code is not wrong, it is early, so these tests keep covering it
    with the gap patched shut. Which is also what keeps them honest about what they are:
    assertions about the *shape* of a cloud entry, not evidence that a cloud server runs.
    The refusal itself is asserted below, without this fixture.
    """
    monkeypatch.setattr("memvara.server.init.cloud_gap", lambda: [])


def test_cloud_mode_writes_mode_and_server_url_instead_of_db(cloud_wired, tmp_path) -> None:
    """Cloud mode has no local store, so the entry it writes has no MEMVARA_DB at all."""
    out = io.StringIO()
    status = init(["--agent", "claude", "--dir", str(tmp_path), "--mode", "cloud"],
                 env={}, stdout=out, stderr=io.StringIO())

    assert status == 0
    entry = _entry(tmp_path)
    assert entry["env"] == {"MEMVARA_MODE": "cloud"}
    assert "(cloud)" in out.getvalue()


def test_cloud_mode_writes_the_server_url_only_when_not_the_default(cloud_wired, tmp_path) -> None:
    out = io.StringIO()
    status = init(["--agent", "claude", "--dir", str(tmp_path), "--mode", "cloud"],
                 env={"MEMVARA_SERVER_URL": "https://staging.memvara.dev"},
                 stdout=out, stderr=io.StringIO())

    assert status == 0
    assert _entry(tmp_path)["env"] == {
        "MEMVARA_MODE": "cloud",
        "MEMVARA_SERVER_URL": "https://staging.memvara.dev",
    }


def test_cloud_mode_with_no_credentials_points_at_login(cloud_wired, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("memvara.server.init._credentials_path",
                        lambda: tmp_path / "nonexistent" / "credentials.json")
    out = io.StringIO()
    status = init(["--agent", "claude", "--dir", str(tmp_path), "--mode", "cloud"],
                 env={}, stdout=out, stderr=io.StringIO())

    assert status == 0
    assert "memvara-mcp login" in out.getvalue()


def test_cloud_mode_with_an_api_key_already_set_skips_the_login_reminder(cloud_wired, tmp_path) -> None:
    out = io.StringIO()
    status = init(["--agent", "claude", "--dir", str(tmp_path), "--mode", "cloud"],
                 env={"MEMVARA_API_KEY": "mv_something"}, stdout=out, stderr=io.StringIO())

    assert status == 0
    assert "memvara-mcp login" not in out.getvalue()


def test_cloud_mode_with_an_existing_mcp_json_prints_the_block_to_paste(cloud_wired, tmp_path) -> None:
    """The cloud-mode mirror of `test_an_existing_mcp_json_is_never_rewritten`: an
    `.mcp.json` already present means `_mcp_json` returns `paste=True`, and cloud mode's
    own branch has to print the entry the same way local mode's does."""
    original = '{ "mcpServers": { "other": { "command": "elsewhere" } } }\n'
    (tmp_path / ".mcp.json").write_text(original, encoding="utf-8")

    out = io.StringIO()
    status = init(["--agent", "claude", "--dir", str(tmp_path), "--mode", "cloud"],
                 env={"MEMVARA_API_KEY": "mv_something"}, stdout=out, stderr=io.StringIO())

    assert status == 0
    assert (tmp_path / ".mcp.json").read_text(encoding="utf-8") == original
    body = out.getvalue()
    assert 'Add this inside the "mcpServers" object' in body
    assert '"MEMVARA_MODE": "cloud"' in body


def test_an_invalid_mode_is_a_usage_error(tmp_path) -> None:
    err = io.StringIO()
    status = init(["--agent", "claude", "--dir", str(tmp_path), "--mode", "staging"],
                 env={}, stdout=io.StringIO(), stderr=err)

    assert status == 2
    assert "'local' or 'cloud'" in err.getvalue()


def test_httpx_importable_reports_true_when_the_cloud_extra_is_installed() -> None:
    from memvara.server.init import _httpx_importable

    assert _httpx_importable() is True


def test_httpx_importable_reports_false_when_it_is_not(monkeypatch) -> None:
    import builtins

    from memvara.server.init import _httpx_importable

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "httpx":
            raise ImportError("no module named httpx")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert _httpx_importable() is False


def test_no_mode_and_no_httpx_defaults_to_local(tmp_path, monkeypatch) -> None:
    """The other half of the default-mode decision: absent the `cloud` extra, an
    unset --mode and an unset MEMVARA_MODE fall back to 'local', not 'cloud'."""
    monkeypatch.setattr("memvara.server.init._httpx_importable", lambda: False)
    out = io.StringIO()
    status = init(["--agent", "claude", "--dir", str(tmp_path)],
                 env={"MEMVARA_DB": str(tmp_path / "store" / "memory.db")},
                 stdout=out, stderr=io.StringIO())
    assert status == 0
    assert "MEMVARA_DB" in _entry(tmp_path)["env"]


def test_the_mode_environment_variable_is_honoured_without_the_flag(cloud_wired, tmp_path) -> None:
    """MEMVARA_MODE in the shell picks the mode when --mode is not given, the same
    precedence the flag/environment pairs elsewhere in this command already use."""
    out = io.StringIO()
    status = init(["--agent", "claude", "--dir", str(tmp_path)],
                 env={"MEMVARA_MODE": "cloud"}, stdout=out, stderr=io.StringIO())

    assert status == 0
    assert _entry(tmp_path)["env"] == {"MEMVARA_MODE": "cloud"}


def test_cloud_client_entry_omits_server_url_at_the_default() -> None:
    entry = cloud_client_entry(server_url="https://app.memvara.dev", command="p")
    assert entry["env"] == {"MEMVARA_MODE": "cloud"}


# --- the config that could not start the server it configured -----------------


def test_asking_for_cloud_is_refused_with_the_reason_the_server_would_have_given(
        tmp_path) -> None:
    """`init` wrote `MEMVARA_MODE: cloud`, said "restart your client", and exited 0.

    The server then refused to start, because the REST facade has no endpoint for the
    surface the engine calls on every turn. Two commands answering the same question
    differently, and the gap between them was silent by construction: the one that writes
    the config never starts the thing it configured, so nothing in the successful run
    could notice. What reached the user was a client with no memvara tools in it and no
    line of output anywhere connecting that to anything they had done.

    The text is the server's own, so the reason arrives while there is still something to
    be done about it.
    """
    out, err = io.StringIO(), io.StringIO()
    status = init(["--agent", "claude", "--dir", str(tmp_path), "--mode", "cloud"],
                  env={"MEMVARA_API_KEY": "mv_live_x"}, stdout=out, stderr=err)

    assert status != 0
    assert "cannot start a server yet" in err.getvalue()
    assert not (tmp_path / ".mcp.json").exists(), "refused, and still wrote the file"


def test_the_environment_asking_for_cloud_is_refused_the_same_way(tmp_path) -> None:
    """`MEMVARA_MODE=cloud` exported months ago is the same broken config arriving by a
    quieter route, so it gets the same answer rather than a silent downgrade to local."""
    err = io.StringIO()
    status = init(["--agent", "claude", "--dir", str(tmp_path)],
                  env={"MEMVARA_MODE": "cloud"}, stdout=io.StringIO(), stderr=err)

    assert status != 0
    assert "cannot start a server yet" in err.getvalue()


def test_an_importable_httpx_no_longer_defaults_a_working_install_to_a_broken_one(
        tmp_path) -> None:
    """The default was `cloud` whenever `httpx` imported, which is a great many
    environments that never installed the cloud extra — and every one of them got a
    client that could not start. The heuristic now asks whether cloud works before
    preferring it, so the fallback is the mode that does.
    """
    status = init(["--agent", "claude", "--dir", str(tmp_path)],
                  env={}, stdout=io.StringIO(), stderr=io.StringIO())
    assert status == 0
    entry_env = _entry(tmp_path)["env"]
    assert "MEMVARA_DB" in entry_env, entry_env
    assert entry_env.get("MEMVARA_MODE") != "cloud"


def test_the_refusal_lifts_itself_when_the_facade_grows_the_endpoints(
        cloud_wired, tmp_path) -> None:
    """Nothing has to remember to flip a flag.

    Both `init` and `build_memvara` derive the answer from `RemoteStore.WIRED`, so the
    day the endpoints exist this command starts writing cloud configs again — which is
    the only version of this guard that does not become a second thing to maintain.
    """
    status = init(["--agent", "claude", "--dir", str(tmp_path), "--mode", "cloud"],
                  env={"MEMVARA_API_KEY": "mv_live_x"},
                  stdout=io.StringIO(), stderr=io.StringIO())
    assert status == 0
    assert _entry(tmp_path)["env"]["MEMVARA_MODE"] == "cloud"


def test_the_gap_is_derived_from_the_store_rather_than_hardcoded() -> None:
    """`cloud_gap()` names the calls the engine makes that `RemoteStore` cannot serve.

    Asserted as a set relationship rather than a literal list, so adding an endpoint
    shortens it without editing a test, and adding an engine call that the facade does
    not serve lengthens it without anyone having to notice.
    """
    from memvara.server.config import _ENGINE_NEEDS, cloud_gap
    from memvara.store.remote import RemoteStore

    assert set(cloud_gap()) == _ENGINE_NEEDS - RemoteStore.WIRED
    assert cloud_gap(), (
        "the facade has grown the low-level endpoints — that is good news, and it means "
        "the refusal above is now dead code that should be removed rather than tested"
    )
