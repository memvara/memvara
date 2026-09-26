"""The environment every child process of the adversarial suite runs in."""

from __future__ import annotations

import os
import pathlib
from typing import Mapping

#: The checkout under test. This file is tests/harness/env.py, two levels below it.
REPO = pathlib.Path(__file__).resolve().parents[2]


def _real_home() -> pathlib.Path:
    """The account's home directory, read from the password database on POSIX when
    the account has an entry there.

    Not read from HOME, because every test runs with HOME pointed at a temporary
    directory (tests/conftest.py), and a check against HOME would compare that temporary
    directory with itself.
    """
    if os.name == "posix":
        import pwd  # noqa: PLC0415 - POSIX only

        try:
            return pathlib.Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
        except KeyError:
            # A container can run as a numeric user with no password entry. HOME is
            # then the best answer there is, and importing the harness must not crash.
            pass
    return pathlib.Path(os.path.expanduser("~")).resolve()


#: The real home directory, which no child process may be given.
REAL_HOME = _real_home()

#: Variables a child must not inherit from the machine running the suite. CI exports
#: MEMVARA_API_KEY for one hosted test, a developer's shell can hold model keys, and a
#: suite started from inside Claude Code carries that session's own variables.
_DROPPED = ("MEMVARA_", "ANTHROPIC_", "OPENAI_", "CLAUDECODE", "CLAUDE_CODE_")


def child_env(home: pathlib.Path, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment for one child process of the suite.

    It starts from this process's environment rather than from an empty one, because on
    Windows a child without SYSTEMROOT cannot open a socket
    (tests/test_hook_recall_requests.py found this). Then:

    * every MEMVARA_, ANTHROPIC_, OPENAI_ and Claude Code variable is removed;
    * HOME and USERPROFILE point at `home`, which must not be the real home directory;
    * PYTHONPATH is this checkout, so the child imports the code under test rather than
      whatever an editable install points at;
    * the store uses the hashing embedder, no encryption and no project read from git,
      the keyring backend is the null one, and the hooks never start their background
      daemon, which would outlive the test.

    `extra` is applied last, so a test can override any of these.
    """
    home = pathlib.Path(home).resolve()
    if home == REAL_HOME:
        raise ValueError(
            f"refusing to start a child process with the real home directory {home}")
    env = {key: value for key, value in os.environ.items() if not key.startswith(_DROPPED)}
    env.update({
        "HOME": str(home),
        "USERPROFILE": str(home),
        "PYTHONPATH": str(REPO),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring",
        "MEMVARA_EMBEDDER": "hashing",
        "MEMVARA_FEATURE_ENCRYPTION": "0",
        "MEMVARA_FEATURE_PROJECT_SCOPE": "0",
        "MEMVARA_DAEMON": "1",
    })
    env.update(extra or {})
    return env


def feature_env(features: Mapping[str, bool]) -> dict[str, str]:
    """The MEMVARA_FEATURE_<NAME> variables that switch each named feature on or off.

    One spelling of the convention for everything that sets it: a server's environment
    (`stdio.McpProcess`) and the client config the hooks read (the scripted scenarios).
    The names are not checked here; McpProcess refuses one the server does not know.

    >>> feature_env({"documents": False})
    {'MEMVARA_FEATURE_DOCUMENTS': '0'}
    """
    return {f"MEMVARA_FEATURE_{name.upper()}": "1" if on else "0"
            for name, on in features.items()}
