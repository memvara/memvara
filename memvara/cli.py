"""`memvara` — sign in to a hosted deployment, sign out of it, and ask who you are.

Three commands, and they are the three a person needs before any hosted code of theirs
runs: `login` obtains an API key, `whoami` says what that key authorizes, and `logout`
removes it from this machine. Everything else this package does is a library call or the
MCP server, and both are documented where they live.

`memvara-mcp` keeps its own `login`, unchanged, because it is in published instructions
and in configuration files people have already written. Both spellings run the same flow
in `server/login.py`; the messages name whichever command was actually typed.

## The npm package also installs a `memvara` command

`npm/memvara` is a bridge that connects a stdio MCP client to the hosted server, and its
`package.json` declares a `bin` named `memvara` too. Installed globally, whichever of the
two comes first on `PATH` wins, and the two are not the same program: the npm one's
`login` writes an OAuth token to `~/.memvara/oauth.json`, and this one's writes an API key
to `~/.memvara/credentials.json`.

Neither has to lose. `npx memvara` always runs the npm bridge, whatever is on `PATH`, and
`python3 -m memvara` always runs this one. Use those two spellings wherever it matters,
and note that the bridge reads `~/.memvara/credentials.json` first, so a key from this
command is one the bridge will use.

## What a command here may print

Never the key. `whoami` holds a live credential in order to describe it, and a key echoed
once ends up in a terminal transcript, a screenshot or a CI log — so this module prints
the *path* the key came from, the project it is for and what the server says it
authorizes, and nothing that could be pasted into an Authorization header.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TextIO

from . import __version__

__all__ = ["USAGE", "main"]

#: Where a credential lives when `--credentials` says nothing else. Named here rather
#: than imported from `server.config` for the reason `login.py` gives for its own copy:
#: this module has to stay importable with no extras installed.
_DEFAULT_CREDENTIALS = Path.home() / ".memvara" / "credentials.json"

USAGE = f"""\
memvara {__version__} — memory for AI agents. https://memvara.dev

  memvara login      sign in to a memvara-cloud deployment and store an API key
  memvara logout     remove a stored API key from this machine
  memvara whoami     say what the stored credential is, and what it authorizes

Each command takes --help. All three take --credentials PATH, which is how one machine
holds keys for two projects: the default file, {_DEFAULT_CREDENTIALS}, is
the one MEMVARA_MODE=cloud and Memvara.connect() read, and a key written anywhere else
moves nothing.

The library itself needs none of this. `pip install memvara` and `Memvara("memory.db")`
is a local store on this machine, with no account and no network. These commands are for
a hosted deployment, and they need pip install "memvara[cloud]".

The MCP server is a separate command, `memvara-mcp`. It carries its own `login`, which is
this one under the older name.
"""

_CLOUD_EXTRA = ('this command talks to a hosted deployment, which needs httpx: '
                'pip install "memvara[cloud]".')


class _Usage(Exception):
    """The command line was wrong, and the message says which part."""


def _options(argv: Sequence[str], allowed: Sequence[str]) -> dict[str, str]:
    """`--name value` and `--name=value`, matching `login.py`'s parser exactly.

    Hand-written rather than `argparse` for the reason `init.py` and `login.py` are:
    these commands take two or three options between them, and `argparse` would print
    its own usage over the top of the one above, in a different voice.
    """
    found: dict[str, str] = {}
    rest = list(argv)
    while rest:
        argument = rest.pop(0)
        name, joined, inline = argument.partition("=")
        if name not in allowed:
            raise _Usage(f"unexpected argument {argument!r}. This command takes "
                         f"{', '.join(allowed)}.")
        value = inline if joined else (rest.pop(0) if rest else "")
        if not value.strip():
            raise _Usage(f"{name} needs a value")
        found[name] = value.strip()
    return found


def _credentials_path(options: Mapping[str, str]) -> Path:
    return Path(options["--credentials"]) if "--credentials" in options \
        else _DEFAULT_CREDENTIALS


LOGOUT_USAGE = """\
memvara logout — remove a stored API key from this machine.

  --credentials PATH  which credentials file to remove. Default
                      ~/.memvara/credentials.json.

This deletes a file. It does not revoke the key: the key keeps working for anyone who
holds a copy until it is revoked in the console, under the project's API keys.
"""


def logout(argv: Sequence[str], *, env: Mapping[str, str] | None = None,
           stdout: TextIO | None = None, stderr: TextIO | None = None) -> int:
    """Remove a credentials file, and say plainly what that did and did not do."""
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    if "--help" in argv or "-h" in argv:
        print(LOGOUT_USAGE, file=out)
        return 0
    try:
        path = _credentials_path(_options(argv, ("--credentials",)))
    except _Usage as exc:
        print(f"memvara logout: {exc}\n\n{LOGOUT_USAGE}", file=err)
        return 2

    try:
        path.unlink()
    except FileNotFoundError:
        # Not an error. Somebody running this twice, or on a machine that never signed
        # in, has the outcome they asked for, and a non-zero exit would fail a script
        # whose whole purpose is to leave no key behind.
        print(f"memvara logout: nothing to remove at {path}.", file=out)
        return 0
    except OSError as exc:
        print(f"memvara logout: could not remove {path}: {exc}", file=err)
        return 1

    print(f"Removed {path}.", file=out)
    # The sentence that matters. Deleting a file is not revoking a credential, and
    # somebody who believes it is will leave a live key behind on purpose.
    print("That key is still valid. Deleting this file removes it from this machine "
          "only — revoke it in the console, under the project's API keys, if it should "
          "stop working everywhere.", file=out)
    return 0


WHOAMI_USAGE = """\
memvara whoami — say what the stored credential is, and what the server says it allows.

  --credentials PATH  which credentials file to read. Default
                      ~/.memvara/credentials.json.
  --server URL        the deployment to ask. Default: the server_url in the credentials
                      file, then MEMVARA_SERVER_URL, then https://app.memvara.dev.

Reads MEMVARA_API_KEY first when no --credentials was given, because that is the order
every other part of this library resolves a credential in. Prints the project, the
credential's non-secret id, its privilege and its expiry. It never prints the key.
"""


def whoami(argv: Sequence[str], *, env: Mapping[str, str] | None = None,
           stdout: TextIO | None = None, stderr: TextIO | None = None) -> int:
    """Ask the deployment what this credential authorizes, and print it without the key."""
    environ = os.environ if env is None else env
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    if "--help" in argv or "-h" in argv:
        print(WHOAMI_USAGE, file=out)
        return 0
    try:
        options = _options(argv, ("--credentials", "--server"))
    except _Usage as exc:
        print(f"memvara whoami: {exc}\n\n{WHOAMI_USAGE}", file=err)
        return 2

    # Imported here rather than at module scope: this module has to import on a bare
    # install, so that `memvara --help` works there. The same boundary `server/cli.py`
    # keeps around `login`. These three modules import no httpx themselves — the
    # transport does, at construction, which is where the missing extra is caught below.
    from .remote.api import RemoteMemvara
    from .remote.creds import read_credentials_file
    from .remote.errors import RemoteError

    named = "--credentials" in options
    path = _credentials_path(options)
    stored = read_credentials_file(path)
    from_env = None if named else (environ.get("MEMVARA_API_KEY") or "").strip() or None
    api_key = from_env or stored.get("api_key")
    if not api_key:
        print(f"memvara whoami: no credential. {path} holds none"
              + ("" if named else ", and MEMVARA_API_KEY is not set")
              + '. Run "memvara login" to obtain one.', file=err)
        return 1

    base_url = (options.get("--server") or stored.get("server_url")
                or environ.get("MEMVARA_SERVER_URL") or "https://app.memvara.dev")
    source = "MEMVARA_API_KEY" if from_env else str(path)
    try:
        # Where the `cloud` extra is actually required: the transport imports httpx when
        # it is built. Constructing performs no network call, so nothing has been spent
        # when this fails.
        client = RemoteMemvara(api_key=api_key, base_url=base_url)
    except ImportError:
        print(f"memvara whoami: {_CLOUD_EXTRA}", file=err)
        return 2
    try:
        answer = client.whoami()
    except RemoteError as exc:
        # The ordinary reason to run this command is a credential that has stopped
        # working, so its own failure is a sentence rather than a traceback — which
        # would carry the request, and with it the bearer token, into the terminal.
        print(f"memvara whoami: {exc.message} ({exc.status_code} {exc.code}). The "
              f"credential in {source} was not accepted by {base_url}.", file=err)
        return 1
    finally:
        client.close()

    scope = answer.get("scope") or {}
    print(f"credential  {source}", file=out)
    print(f"server      {base_url}", file=out)
    if stored.get("project") and not from_env:
        print(f"project     {stored['project']}", file=out)
    print(f"token       {answer.get('token_id')}", file=out)
    print(f"privilege   {answer.get('effective_privilege')}"
          + (f" (granted {answer.get('granted_privilege')})"
             if answer.get("granted_privilege") != answer.get("effective_privilege")
             else ""), file=out)
    print(f"tenant      {scope.get('tenant')}", file=out)
    narrowed = " ".join(f"{name}={scope[name]}" for name in ("user", "agent", "session")
                        if scope.get(name))
    print(f"scope       {narrowed or 'the whole tenant'}", file=out)
    print(f"expires     {answer.get('expires_at') or 'no expiry'}", file=out)
    if answer.get("read_only"):
        print("read-only   this deployment is refusing every write", file=out)
    return 0


def _login(argv: Sequence[str], *, env: Mapping[str, str] | None = None,
           stdout: TextIO | None = None, stderr: TextIO | None = None) -> int:
    """`memvara login`, which is `server/login.py` under this command's name."""
    from .server.login import login

    return login(argv, env=env, stdout=stdout, stderr=stderr, prog="memvara login")


#: Command name -> what runs it. The usage text above is checked against these names by
#: `tests/test_cli.py`, so a command added here and not documented there fails the suite
#: rather than becoming a command nobody can find.
COMMANDS: dict[str, Callable[..., int]] = {
    "login": _login,
    "logout": logout,
    "whoami": whoami,
}


def main(argv: Sequence[str] | None = None, *, env: Mapping[str, str] | None = None,
         stdout: TextIO | None = None, stderr: TextIO | None = None) -> int:
    """Dispatch one subcommand. Returns a process exit status."""
    args = list(sys.argv[1:] if argv is None else argv)
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr

    if args and args[0] in COMMANDS:
        run: Any = COMMANDS[args[0]]
        return int(run(args[1:], env=env, stdout=out, stderr=err))
    if args in (["--help"], ["-h"], ["help"]):
        print(USAGE, file=out)
        return 0
    if args == ["--version"]:
        print(__version__, file=out)
        return 0
    if not args:
        # No default command. Somebody typing `memvara` alone is finding out what this
        # does, and the worst possible answer is to start a device-code login.
        print(f"memvara: say which command.\n\n{USAGE}", file=err)
        return 2
    print(f"memvara: unexpected argument {args[0]!r}. The commands are "
          f"{', '.join(COMMANDS)}.\n\n{USAGE}", file=err)
    return 2
