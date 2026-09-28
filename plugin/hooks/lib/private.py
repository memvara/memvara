"""Everything the hooks keep is private to the account that runs them.

The hooks write under `~/.memvara/.hooks`: logs that quote a person's prompts and a
model's replies (`capture.log`, `recall.log`, `recall-sample.log`, `hooks.log`), the token
ledger `usage.jsonl`, `capture-state.json`, which names every transcript the capture hook
has mined, and small state and lock files. They were created with the process's default
modes, which under the usual umask of 022 left each directory at 0755 and each file at
0644, readable by every account on the machine, and `~/.memvara` itself was 0755 when a
hook created it.

So every directory from `~/.memvara` down is created 0700 and every file 0600, and one that
already exists loses any permission for group and others the first time a hook in a
process uses it, which repairs what an earlier version left behind. A process checks each
directory once (`_checked`), so the per-prompt hooks pay a few system calls the first time
and none after.

Only `os` and `stat` are imported, because `lib.ipc` imports this on the per-prompt path,
where every millisecond of start-up is counted. On Windows the modes mean little and
`os.chmod` changes only the read-only flag, which nothing here clears.
"""

from __future__ import annotations

import os
import stat
from typing import IO

#: The directory every hook file lives under. It and everything below it is made private;
#: the home directory above it is left as it is.
ROOT = os.path.join(os.path.expanduser("~"), ".memvara")

#: Directories this process has already made private.
_checked: set[str] = set()


def _restrict(path: str) -> None:
    """Take every permission for group and others off `path`, or do nothing if that fails."""
    try:
        mode = stat.S_IMODE(os.stat(path).st_mode)
        if mode & 0o077:
            os.chmod(path, mode & 0o700)
    except OSError:
        pass


def private_dir(path: str) -> str:
    """Make sure `path` exists as a private directory, and return it.

    Every directory from `ROOT` down to `path` is created 0700 when it is missing, and one
    that exists loses any permission for group and others. A `path` outside `ROOT` is
    created 0700 when it is missing and otherwise left alone, because it is somebody
    else's choice of place. Raises `OSError` as `os.makedirs` would.
    """
    path = os.path.abspath(path)
    if path in _checked:
        return path
    chain: list[str] = []
    level = path
    while level == ROOT or level.startswith(ROOT + os.sep):
        chain.append(level)
        if level == ROOT:
            break
        level = os.path.dirname(level)
    if not chain:
        os.makedirs(path, mode=0o700, exist_ok=True)
    for level in reversed(chain):
        try:
            os.mkdir(level, 0o700)
        except FileExistsError:
            _restrict(level)
    _checked.add(path)
    return path


def private_open(path: str, mode: str = "a") -> IO[str]:
    """Open `path` to write text, as `open(path, mode, encoding="utf-8")` would, creating
    the file 0600. `mode` is "a" to append or "w" to truncate.

    A file that already exists loses any permission for group and others first. The
    caller makes sure the directory exists, with `private_dir`.
    """
    flags = os.O_WRONLY | os.O_CREAT | (os.O_APPEND if mode == "a" else os.O_TRUNC)
    fd = os.open(path, flags | getattr(os, "O_BINARY", 0), 0o600)
    try:
        if hasattr(os, "fchmod"):
            bits = stat.S_IMODE(os.fstat(fd).st_mode)
            if bits & 0o077:
                os.fchmod(fd, bits & 0o700)
        return os.fdopen(fd, mode, encoding="utf-8")
    except BaseException:
        os.close(fd)
        raise
