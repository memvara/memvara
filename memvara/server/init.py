"""`memvara-mcp init --agent claude` — write the client's configuration, once, correctly.

The server itself has no options because everything it needs is environment configuration
(see `cli.py`). This is the other half of that decision rather than a retreat from it: an
env block is a thing somebody has to *write*, in another application's JSON file, by hand,
before this program has ever run — a command, an argument list and a handful of variables,
one of which, `MEMVARA_DB`, has to be an absolute path, which is the requirement people
meet on the second attempt. Flags on the server would not help, because the user is not
the one typing that command line, and never sees its output either. So the
fix is a subcommand that *writes* configuration, and it is careful about other people's
files in three different ways because the three files it touches are owned differently:

* **`.mcp.json`** is the client's, and usually names other servers too. Created when
  absent; never rewritten. Merging would mean round-tripping somebody else's file through
  `json.dumps` — losing whatever formatting or comments they had — and a bad merge shows
  up as a client that will not start, with nothing on screen saying why. Printing the
  entry to paste costs one paste and cannot destroy anything.
* **`CLAUDE.md` / `AGENTS.md`** are append-shaped, so they are appended to, behind a
  marker comment that keeps a second run from adding the section twice. Which file is
  primary depends on `--agent`; the other is updated too if it already exists.
* **The skill tree** is ours: generated data that ships in the wheel and therefore
  versions with the tools it describes. It is the only thing `--force` will replace,
  because it is the only one whose current contents this program is entitled to have
  an opinion about.

Nothing is overwritten silently, every path touched is printed with the verb applied to
it, and a rerun is meant to be boring: exit 0 with a column of `kept`. Exit 2 is reserved
for a command line that was wrong, matching the server's own use of it.

This is the Python package. The npm package is a placeholder that installs nothing, so
there is no `npx` equivalent of this command and none is implied.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO


__all__ = ["AGENTS", "INIT_USAGE", "MARKER", "client_entry", "cloud_client_entry", "init",
           "skill_text"]

#: Clients `init` knows the file layout of. The packaged skill is one tree
#: (`memvara/skills/memvara/`); these names only decide where that tree is written.
AGENTS = ("claude", "cursor", "grok")

#: Relative destination of the skill tree, per client. Cursor and Grok also read
#: `.claude/skills/`, so the Claude path is enough for a mixed team; the other
#: two exist for a project that will not have a `.claude/` directory.
_SKILL_DEST = {
    "claude": Path(".claude") / "skills" / "memvara",
    "cursor": Path(".cursor") / "skills" / "memvara",
    "grok": Path(".grok") / "skills" / "memvara",
}

_NOTE_FILE = {
    "claude": "CLAUDE.md",
    "cursor": "AGENTS.md",
    "grok": "AGENTS.md",
}

#: What makes the CLAUDE.md section recognisable on a second run. A marker rather than a
#: search for the word "memvara", because a project *about* memvara mentions it on every
#: other line and would never get its snippet.
MARKER = "<!-- memvara -->"


def _note_text(skill_rel: Path) -> str:
    """The CLAUDE.md / AGENTS.md section. `skill_rel` is the path this run writes."""
    return f"""\
{MARKER}
## Memory

This project has the memvara MCP server configured, and its `memory_*` tools are the only
thing that persists between sessions. Read memory before answering anything that could
depend on what this user told you earlier, and store what would be embarrassing to have
forgotten next week.

The `memvara` skill in `{skill_rel.as_posix()}/` carries the rules that span tools — the
sequence to follow when a stored fact is disputed, which scope your writes land in, and
what changes on a server with no extraction model. Each tool's own description carries the
rest.
<!-- /memvara -->
"""


INIT_USAGE = f"""\
memvara-mcp init — write the MCP server block, the agent skill and a project note.

  --agent NAME  which client to configure. {', '.join(repr(a) for a in AGENTS)}.
                Naming it is required rather than defaulted: each one writes a
                different directory, and a default would silently pick one.
  --dir PATH    project directory to write into. Default: the current directory.
  --mode NAME   'local' or 'cloud'. Default: MEMVARA_MODE if this shell has one,
                otherwise 'local' — installing the cloud extra does not change it,
                because installing a package is not a request to stop using your
                own store. Say --mode cloud to get one. Local writes
                MEMVARA_DB/MEMVARA_USER, exactly what this command always wrote before
                --mode existed. Cloud writes MEMVARA_MODE and, only when it is not the
                default, MEMVARA_SERVER_URL — and if this machine has no credentials
                yet, prints a reminder to run `memvara-mcp login` rather than running it.

                Cloud is refused, by flag or by MEMVARA_MODE, when httpx is not
                importable: the server talks to the deployment over HTTP and would not
                start, and writing the config anyway hands you a broken client and a
                success message.
  --db PATH     where the store lives, local mode only. Default: MEMVARA_DB if this
                shell has one, otherwise ~/.memvara/memory.db. Written absolute either
                way — that is the requirement no client enforces and everyone meets on
                the second attempt.
  --user NAME   who the server remembers for, local mode only. Default: MEMVARA_USER,
                else USER. Unset means the whole tenant, which is right for a
                single-person machine.
  --skill-only  write the skill tree and the project note, and leave .mcp.json
                alone. For a client that is already connected to the hosted URL.
  --force       replace a packaged skill file when it differs from this version.
                Never touches .mcp.json or the project note; those are yours.

What gets written, relative to --dir (Claude Code shown; --agent picks the layout):

  .mcp.json                        created if absent, never rewritten
  .claude/skills/memvara/SKILL.md  the packaged skill, plus references/
  CLAUDE.md                        appended to once, behind a marker comment
                                   (AGENTS.md for --agent cursor and grok)

The other environment variables keep their defaults and `memvara-mcp --help` lists them.
This is the Python package; the npm package is a placeholder with no equivalent command.
"""

_OPTIONS = ("--agent", "--db", "--dir", "--user", "--mode")
_FLAGS = ("--force", "--skill-only")

#: The server_url a client that never sets MEMVARA_SERVER_URL gets from config.py's own
#: default. Written into .mcp.json only when the caller's value differs from it — writing
#: a default into somebody's settings file freezes it there, the same reasoning
#: `client_entry` already applies to MEMVARA_DB's default.
_DEFAULT_SERVER_URL = "https://app.memvara.dev"


class _Usage(Exception):
    """The command line was wrong, and the message says which part."""


@dataclass(frozen=True, slots=True)
class _Step:
    """One path, and what was done to it — the whole report this command owes the user."""

    verb: str
    path: Path
    note: str = ""
    #: Set when the user has to finish the job by hand, so the caller knows to print the
    #: block for them. A flag rather than the caller re-reading `note`, because a report
    #: line is prose and prose gets edited.
    paste: bool = False


def skill_text(agent: str = AGENTS[0]) -> str:
    """The packaged skill body, read out of the installed package.

    `agent` is the client `init` is configuring. It used to pick a per-client file;
    there is one skill now, and every client gets it. The argument stays so callers
    that pass it keep working.

    Package data rather than a string literal, for the reason the skill itself is
    short: it describes the tools, and a copy that ships separately from them is
    the copy that drifts. `importlib.resources` is what makes that a promise about
    the *installed* distribution rather than about this source tree.
    """
    return (files("memvara") / "skills" / "memvara" / "SKILL.md").read_text(encoding="utf-8")


def _packaged_skill_files() -> list[tuple[Path, str]]:
    """Relative paths under the skill directory, and the text to write there."""
    base = files("memvara") / "skills" / "memvara"
    entries = [(Path("SKILL.md"), (base / "SKILL.md").read_text(encoding="utf-8"))]
    refs = base / "references"
    names = sorted(child.name for child in refs.iterdir() if child.name.endswith(".md"))
    for name in names:
        entries.append((Path("references") / name, (refs / name).read_text(encoding="utf-8")))
    return entries


def client_entry(*, db: str, user: str | None, command: str) -> dict[str, Any]:
    """The `mcpServers.memvara` object, with this machine's real values in it.

    Only the two variables the example in `config.py` shows. The rest have defaults, and
    writing a default into somebody's settings file freezes it there: the day the default
    moves, the one deployment that will not follow is the one that had it spelled out.
    """
    environment = {"MEMVARA_DB": db}
    if user is not None:
        environment["MEMVARA_USER"] = user
    return {"command": command, "args": ["-m", "memvara.server"], "env": environment}


def cloud_client_entry(*, server_url: str | None, command: str) -> dict[str, Any]:
    """The `mcpServers.memvara` object for cloud mode: no store path, no local user.

    `MEMVARA_SERVER_URL` is written only when it differs from the default, for the same
    reason `client_entry` omits a default `MEMVARA_USER`: a default spelled out here is
    one that does not move when the real default does.
    """
    environment: dict[str, str] = {"MEMVARA_MODE": "cloud"}
    if server_url and server_url != _DEFAULT_SERVER_URL:
        environment["MEMVARA_SERVER_URL"] = server_url
    return {"command": command, "args": ["-m", "memvara.server"], "env": environment}


def _httpx_importable() -> bool:
    """Whether `pip install memvara[cloud]` happened in this interpreter.

    The only reliable way to ask: `httpx` is the cloud extra's one new dependency
    (see pyproject.toml), so its presence is the signal that this install can actually
    speak to app.memvara.dev, and its absence means a cloud default would hand back a
    client block for a server this interpreter cannot reach.
    """
    try:
        import httpx  # noqa: F401
    except ImportError:
        return False
    return True


def _credentials_path() -> Path:
    return Path(os.path.expanduser("~")) / ".memvara" / "credentials.json"


def _interpreter() -> str:
    """The interpreter to put in `command`, which is not the string a human would type.

    `python3` is what the documented example says, and it resolves against the client's
    `PATH` — not a login shell's — so it finds whichever interpreter the desktop
    application inherited, which for anyone using a virtualenv is the wrong one. The
    interpreter running this command is the one that just imported memvara, which is the
    strongest evidence available here — and not a guarantee, because an import that came
    from `PYTHONPATH` rather than from an install will not survive being launched by
    something else. It is still the difference between a block that usually works and one
    that fails as `No module named memvara` inside a client which shows no logs.
    """
    return sys.executable or "python3"


def _default_db(env: Mapping[str, str]) -> str:
    """Where the store goes when nobody says.

    `MEMVARA_DB` from the caller's own shell first: someone who already has a working
    server and is running this to pick up the skill should not be handed a second, empty
    store that answers none of their questions. Otherwise under the home directory rather
    than in the project, because a memory that lives beside one checkout is a memory that
    forgets the user every time they start something new.
    """
    return (env.get("MEMVARA_DB") or "").strip() or os.path.join(
        os.path.expanduser("~"), ".memvara", "memory.db")


def _absolute(raw: str) -> Path:
    """`~` expanded and relative made absolute, without resolving symlinks.

    Absolute is the actual requirement — the client launches the server from a working
    directory nobody chose — and `resolve()` would additionally rewrite the path through
    every symlink on the way, handing back something the user does not recognise as the
    directory they named.
    """
    return Path(os.path.abspath(os.path.expanduser(raw)))


def _parse(argv: Sequence[str]) -> tuple[dict[str, str], dict[str, bool]]:
    """`--name value` and `--name=value`, and a hand-written parser for the options.

    The same reasoning as the server's argument handling: a parser library here would be
    a dependency-shaped answer to a question with six options in it, and `argparse` in
    particular exits the process on error, which would take the injected streams that make
    this testable out of the picture.
    """
    options: dict[str, str] = {}
    flags = {name: False for name in _FLAGS}
    rest = list(argv)
    while rest:
        argument = rest.pop(0)
        name, joined, inline = argument.partition("=")
        if name in flags and not joined:
            flags[name] = True
            continue
        if name not in _OPTIONS:
            raise _Usage(f"unexpected argument {argument!r}")
        value = inline if joined else (rest.pop(0) if rest else "")
        if not value.strip():
            raise _Usage(f"{name} needs a value")
        options[name] = value.strip()
    return options, flags


def _first(env: Mapping[str, str], *names: str) -> str | None:
    for name in names:
        value = (env.get(name) or "").strip()
        if value:
            return value
    return None


def _skill_tree(root: Path, agent: str, force: bool) -> list[_Step]:
    """Write the packaged skill directory. One step per file, same keep/force rules."""
    dest = root / _SKILL_DEST[agent]
    steps: list[_Step] = []
    for relative, text in _packaged_skill_files():
        path = dest / relative
        if path.exists():
            if path.read_text(encoding="utf-8") == text:
                steps.append(_Step("kept", path, "already this version"))
            elif not force:
                steps.append(_Step("kept", path,
                                   "differs from the packaged skill; --force replaces it"))
            else:
                path.write_text(text, encoding="utf-8")
                steps.append(_Step("replaced", path))
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        steps.append(_Step("wrote", path))
    return steps


def _mcp_json(root: Path, entry: dict[str, Any]) -> _Step:
    path = root / ".mcp.json"
    if not path.exists():
        path.write_text(json.dumps({"mcpServers": {"memvara": entry}}, indent=2) + "\n",
                        encoding="utf-8")
        return _Step("wrote", path)
    # Read as text and not as JSON, because nothing here is going to write it back: the
    # only question is which sentence to print, and a file this command refuses to parse
    # is a file it would refuse to edit anyway.
    if '"memvara"' in path.read_text(encoding="utf-8"):
        return _Step("kept", path, "already names a 'memvara' server")
    return _Step("kept", path, "add the entry printed below to its mcpServers object",
                 paste=True)


def _append_note(root: Path, filename: str, skill_rel: Path) -> _Step:
    path = root / filename
    snippet = _note_text(skill_rel)
    if not path.exists():
        path.write_text(snippet, encoding="utf-8")
        return _Step("wrote", path)
    body = path.read_text(encoding="utf-8")
    if MARKER in body:
        return _Step("kept", path, "already carries the memvara section")
    # One blank line before the section, whatever the file ended with — including nothing
    # at all, which is what an empty note somebody touched and never filled in looks
    # like, and it should not open with two blank lines.
    lead = "" if not body else "\n" if body.endswith("\n") else "\n\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(lead + snippet)
    return _Step("appended", path)


def _notes(root: Path, agent: str) -> list[_Step]:
    """The primary note for this client, plus the other one if it already exists.

    A mixed team often has CLAUDE.md *and* AGENTS.md. Writing only the primary
    file would leave the other client's standing instructions without the
    pointer to the skill.
    """
    skill_rel = _SKILL_DEST[agent]
    primary = _NOTE_FILE[agent]
    names = [primary]
    other = "AGENTS.md" if primary == "CLAUDE.md" else "CLAUDE.md"
    if (root / other).exists():
        names.append(other)
    return [_append_note(root, name, skill_rel) for name in names]


def _store_directory(db: Path) -> _Step:
    """Create the store's *directory*, and deliberately not the store.

    `sqlite3.connect()` does not create a missing directory; it raises "unable to open
    database file", which names neither the directory nor the fix, from inside a server
    the user is not watching. The file itself stays the server's to create on first use,
    so an `init` that is never followed by a launch leaves an empty directory rather than
    an empty database that the embedder fingerprint would then be written into.
    """
    db.parent.mkdir(parents=True, exist_ok=True)
    return _Step("ready", db.parent, "the store file is created on the server's first use")


def _report(steps: Sequence[_Step]) -> list[str]:
    width = max(len(step.verb) for step in steps)
    return [f"{step.verb:<{width}}  {step.path}" + (f"  — {step.note}" if step.note else "")
            for step in steps]


def init(argv: Sequence[str], *, env: Mapping[str, str] | None = None,
         stdout: TextIO | None = None, stderr: TextIO | None = None) -> int:
    """Write the configuration, the skill and the snippet. Returns an exit status."""
    env = os.environ if env is None else env
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr

    if "--help" in argv or "-h" in argv:
        print(INIT_USAGE, file=out)
        return 0

    try:
        options, flags = _parse(argv)
        force = flags["--force"]
        skill_only = flags["--skill-only"]
        agent = options.get("--agent")
        if agent is None:
            raise _Usage("init needs --agent, naming the client to configure: "
                         f"{', '.join(repr(a) for a in AGENTS)}.")
        if agent not in AGENTS:
            raise _Usage(f"--agent {agent!r} is not a client init can write files for. "
                         f"Use {', '.join(repr(a) for a in AGENTS)}. Every MCP client can "
                         "run this server; the ones listed are those whose settings and "
                         "skill layout this command knows.")
        root = _absolute(options.get("--dir", os.getcwd()))
        if not root.is_dir():
            raise _Usage(f"--dir {str(root)!r} is not a directory. Point it at the "
                         "project you want configured; init creates files in it, not it.")
        mode = options.get("--mode")
        if mode is not None and mode not in ("local", "cloud"):
            raise _Usage(f"--mode {mode!r} must be 'local' or 'cloud'.")
        if mode is None:
            # No explicit flag: an explicit MEMVARA_MODE in this shell wins, and absent
            # both, local. Not "cloud when httpx is importable", which is what this line
            # used to say — an importable `httpx` was read as "pip install
            # memvara[cloud] happened", and after the cloud client landed that is true of
            # everyone who installed the extra. Installing an extra is not a request for
            # your local store to stop being the default, and `init` writes the file a
            # client launches from, so getting this wrong is a config somebody has to
            # notice and undo. A hosted-first user passes --mode cloud once.
            #
            # `_httpx_importable` is still load-bearing below, where it answers the only
            # question left: whether a cloud server this command writes a config for can
            # actually start.
            env_mode = (env.get("MEMVARA_MODE") or "").strip()
            mode = env_mode if env_mode in ("local", "cloud") else "local"
        if mode == "cloud" and not _httpx_importable():
            # Named explicitly, by flag or by environment, and it cannot start. Refusing
            # here is the whole point: writing it exits 0, tells the reader to restart
            # their client, and leaves them with a server that will not come up and no
            # line of output connecting the two. The reason arrives while there is still
            # something to do about it.
            #
            # This replaces the check that compared the engine's needs against
            # `RemoteStore.WIRED`. That gap is gone as a reason to refuse, because cloud
            # mode no longer runs the engine over a remote store — it builds a client of
            # the facade. What is left is the one thing that still stops the server
            # starting, and `httpx` is exactly it.
            from ..remote.client import install_hint

            raise _Usage(f"MEMVARA_MODE=cloud cannot start a server here. "
                         f"{install_hint()} Or use --mode local with --db.")
        raw_db = ""
        if mode == "local" and not skill_only:
            raw_db = options.get("--db") or _default_db(env)
            if raw_db == ":memory:":
                raise _Usage(
                    "the store is ':memory:', which is the throwaway one that dies with "
                    "the process — a settings file pointing at it would remember nothing "
                    "between launches. Pass --db with a path to a file.")
    except _Usage as exc:
        print(f"memvara-mcp init: {exc}\n\n{INIT_USAGE}", file=err)
        return 2

    skills = _skill_tree(root, agent, force)
    guidance = _notes(root, agent)

    if skill_only:
        lines = [f"memvara-mcp init — {agent}, in {root} (skill only)", ""]
        lines += _report([*skills, *guidance])
        lines += ["", "No .mcp.json written. Restart the client if it was already",
                  "running, so it picks up the skill."]
        print("\n".join(lines), file=out)
        return 0

    if mode == "cloud":
        server_url = _first(env, "MEMVARA_SERVER_URL") or _DEFAULT_SERVER_URL
        entry = cloud_client_entry(server_url=server_url, command=_interpreter())

        settings = _mcp_json(root, entry)

        lines = [f"memvara-mcp init — {agent}, in {root} (cloud)", ""]
        lines += _report([*skills, settings, *guidance])
        if settings.paste:
            block = json.dumps({"memvara": entry}, indent=2).splitlines()[1:-1]
            lines += ["", 'Add this inside the "mcpServers" object of that file:', ""] + block
        if not (env.get("MEMVARA_API_KEY") or "").strip() and not _credentials_path().exists():
            lines += ["", "No credentials yet on this machine. Run `memvara-mcp login` to",
                      "authorize it before the client can use the memory tools."]
        lines += ["", "The client launches the server itself, so restart it before",
                  "expecting the tools to appear."]
        print("\n".join(lines), file=out)
        return 0

    db = _absolute(raw_db)
    user = options.get("--user") or _first(env, "MEMVARA_USER", "USER", "USERNAME")
    entry = client_entry(db=str(db), user=user, command=_interpreter())

    settings = _mcp_json(root, entry)
    store = _store_directory(db)

    lines = [f"memvara-mcp init — {agent}, in {root}", ""]
    lines += _report([*skills, settings, *guidance, store])
    if settings.paste:
        # Printed without the enclosing braces, indented as it will sit. What the user has
        # to do is paste it inside an object that already exists, and a block that is
        # valid JSON on its own is the one they will paste whole, braces and all, into a
        # file that then does not parse.
        block = json.dumps({"memvara": entry}, indent=2).splitlines()[1:-1]
        lines += ["", 'Add this inside the "mcpServers" object of that file:', ""] + block
    if user is None:
        lines += ["", "No MEMVARA_USER, so this server remembers for the whole tenant —",
                  "right on a single-person machine. Rerun with --user to bind it to one",
                  "person."]
    lines += ["", "The client launches the server itself, so restart it before expecting",
              "the tools to appear. Every other variable keeps its default, including",
              "offline extraction, which stores only the sentence forms it recognises;",
              "`memvara-mcp --help` lists them all."]
    print("\n".join(lines), file=out)
    return 0
