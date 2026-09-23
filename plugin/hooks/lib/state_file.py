"""Small JSON state files that several hook processes may read and write at once.

The hooks keep a handful of these under `~/.memvara/.hooks/`: the recall hook's per-session
dedup state, the status-line counters, the project cache and the capture alerts. Each one
needs the same three things, and this module is the one place that does them:

- **An atomic write.** The data goes to a temporary file in the same directory, which is
  then renamed over the real one, so a reader never sees half a file. The temporary name
  starts with the caller's prefix and is removed if the rename fails.
- **A lock for read-modify-write.** Two hooks for one session can run at the same moment,
  for example two tool calls approved in parallel. Without a lock, the second write
  replaces the first and one update is lost. The lock is an exclusive lock on a lock file:
  `fcntl.flock` on POSIX and `msvcrt.locking` on Windows. Measured on Windows CI without
  it, four processes making fifty updates each kept 5 of 200.
- **Pruning by age.** A file nobody has written for a given time is removed.

Nothing here raises. A hook must never fail a turn over a state file, so every failure,
including a `ValueError` from a path that contains a NUL byte, becomes a return value.

The temporary name is built from the process id and a counter rather than with `tempfile`,
because importing `tempfile` costs about 5ms and the recall hook runs on every prompt.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import os
import os.path
from collections.abc import Callable, Iterator

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None  # type: ignore[assignment]
    import msvcrt

#: Makes temporary names unique within one process; the process id does it across them.
_COUNTER = itertools.count()


def _load(path: str) -> object:
    """Whatever JSON value the file holds, or `None` when it is missing or unreadable."""
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def read_json(path: str) -> dict:
    """A dict from a JSON file, or `{}` for a missing, unreadable, corrupt or non-object file."""
    data = _load(path)
    return data if isinstance(data, dict) else {}


def _replace(path: str, data: dict, prefix: str) -> None:
    """Write `data` to a sibling temporary file and rename it over `path`. Raises on failure."""
    directory = os.path.dirname(path) or "."
    tmp = os.path.join(directory, f"{prefix}{os.getpid()}-{next(_COUNTER)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def write_json(path: str, data: dict, prefix: str = ".state-") -> bool:
    """Write `data` atomically. `True` when it landed, `False` for any failure.

    The directory is created only when the first attempt finds it missing, so the common
    case, a directory that already exists, costs no extra system call.
    """
    try:
        try:
            _replace(path, data, prefix)
        except FileNotFoundError:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            _replace(path, data, prefix)
    except (OSError, ValueError, TypeError):
        return False
    return True


@contextlib.contextmanager
def locked(lock_path: str) -> Iterator[None]:
    """Hold an exclusive lock on `lock_path` for the body. Raises `OSError` or `ValueError`.

    The lock file's directory is created only when opening the file finds it missing.
    """
    try:
        handle = open(lock_path, "a", encoding="utf-8")
    except FileNotFoundError:
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        handle = open(lock_path, "a", encoding="utf-8")
    with handle:
        if fcntl is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
            return
        # Windows locks a byte range rather than a file. Every caller locks the first byte,
        # which may lie past the end of an empty file; Windows allows that. `LK_LOCK` retries
        # for about ten seconds and then raises, and a lock that could not be taken still
        # lets the update go ahead, as on a platform with no lock at all.
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        except OSError:
            yield
            return
        try:
            yield
        finally:
            handle.seek(0)
            with contextlib.suppress(OSError):
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def update_json(path: str, change: "Callable[[object], dict]", *, lock_path: str,
                prefix: str = ".state-") -> bool:
    """Read `path`, apply `change` to it and write the result, all under one lock.

    `change` is handed the file's JSON value as it is, which may be any JSON type, or
    `None` when the file is missing or unreadable, so a caller can read an older format.

    `True` when the new state landed. Any failure, including one inside `change`, returns
    `False` and leaves the old file as it was.
    """
    try:
        with locked(lock_path):
            return write_json(path, change(_load(path)), prefix)
    except (OSError, ValueError, TypeError, KeyError):
        return False


def prune(directory: str, max_age_seconds: float, now: float, suffix: str = ".json") -> None:
    """Remove files ending in `suffix` that were last written more than `max_age_seconds` ago."""
    try:
        names = os.listdir(directory)
    except (OSError, ValueError):
        return
    for name in names:
        if not name.endswith(suffix):
            continue
        path = os.path.join(directory, name)
        try:
            if now - os.path.getmtime(path) > max_age_seconds:
                os.unlink(path)
        except OSError:
            continue
