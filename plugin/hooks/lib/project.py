"""Which project a working directory belongs to, derived from its git remote.

Every clone and every worktree of one repository should share one project scope, so the
identity comes from the `origin` remote rather than from the path. `canonical_project` turns
a directory into `host/owner/repo` (for example `github.com/memvara/memvara`), into
`path:<16 hex characters>` for a repository with no usable remote, or into `None` outside a
repository, which means no project scope at all.

**This is a deliberate copy.** The library has the same function in `memvara/project.py`,
and the hooks cannot import the library: most installs have no library, only these files.
The normalisation rules both copies must agree on are written down once, as data, in
`project_vectors.json` beside this file, and each copy's tests read that file.

How the value reaches the server: a hook calls `bind(cwd)` once, near the top of its run.
That puts the project in the environment variable named by `ENV`, and three things read it
from there. `lib.hosted` sends it as the `Memvara-Project` header on every call. `lib.ipc`
puts it into the daemon's address, so one resident daemon answers for one project. And the
daemon that a hook spawns inherits the variable, so it sends the same header as the hook
that started it. An environment variable is used rather than an argument because the
per-prompt path must not import `lib.hosted`, and because a spawned process inherits it
without any extra plumbing.
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path
import time

from .settings import enabled

#: The channel between `bind` and everything that sends or keys on the project. Private to
#: the hooks: the library reads `MEMVARA_PROJECT`, and a user who sets that for the
#: library's MCP server must not find the hooks silently obeying it too.
ENV = "MEMVARA_HOOK_PROJECT"

#: The switch in `~/.memvara/settings.json` that turns the project scope off.
FEATURE = "project_scope"

#: Hosts whose owner and repository names are case-insensitive, so `Memvara/Memvara` and
#: `memvara/memvara` are one repository there. On any other host the case is kept: a
#: self-hosted forge may treat case as significant, and folding it could merge two
#: different projects. `docs/SUBJECT-CONVENTIONS.md` section 7 states this rule.
CASE_INSENSITIVE_HOSTS = frozenset({"github.com", "gitlab.com", "bitbucket.org"})

#: How many hex characters of the SHA-256 the path form keeps. Sixteen is 64 bits, which
#: makes an accidental collision between two repositories on one machine negligible.
PATH_HEX_CHARS = 16

#: Where `resolve` remembers each directory's answer. Beside the other hook state, not in
#: the plugin, which is replaced on update.
CACHE = os.path.join(os.path.expanduser("~"), ".memvara", ".hooks", "projects.json")

#: How long a cached answer is trusted. Working out a project costs two `git` processes,
#: measured at about 15ms each, and the recall hook has a budget of roughly 30ms per prompt.
#: An hour means a changed remote is noticed within the hour, at the cost of one lookup per
#: directory per hour.
CACHE_TTL_SECONDS = 60 * 60

#: Seconds to wait for `git` before treating the directory as having no project.
GIT_TIMEOUT_SEC = 5


def normalise_remote(url: str) -> "str | None":
    """`host/owner/repo` for a git remote URL, or `None` when the URL names no host.

    The rules, each pinned by a row in `project_vectors.json`: surrounding whitespace is
    ignored; credentials are dropped; the host is lower-cased and a port is kept; the query,
    the fragment, empty path segments, trailing slashes and one trailing `.git` are removed;
    `git@host:owner/repo` is read as `host/owner/repo`; owner and repository are lower-cased
    only on `CASE_INSENSITIVE_HOSTS`. A local path, a `file://` URL and a URL with no path
    return `None`, and the caller then falls back to the path form.
    """
    import urllib.parse

    url = url.strip()
    if not url or "\\" in url:
        # A backslash means a Windows path, never a remote with a host in it.
        return None
    if "://" in url:
        parts = urllib.parse.urlsplit(url)
        if parts.scheme.lower() == "file":
            return None
        host = parts.hostname or ""
        try:
            port = parts.port
        except ValueError:
            return None
        path = parts.path
    else:
        # scp-style `[user@]host:path`. A colon after the first slash, or no colon at all,
        # is a local path. A one-character "host" is a drive letter.
        head, sep, path = url.partition(":")
        if not sep or "/" in head or len(head) < 2:
            return None
        host = head.rpartition("@")[2].lower()
        port = None
        path = path.split("#", 1)[0].split("?", 1)[0]
    if not host:
        return None
    segments = [part for part in path.split("/") if part]
    if segments and segments[-1].endswith(".git"):
        segments[-1] = segments[-1][:-len(".git")]
        segments = [part for part in segments if part]
    if not segments:
        return None
    if host in CASE_INSENSITIVE_HOSTS:
        segments = [part.lower() for part in segments]
    netloc = f"{host}:{port}" if port is not None else host
    return "/".join([netloc, *segments])


def path_identity(root: str) -> str:
    """The project for a repository with no usable remote: a digest of its root's path.

    `root` should already be a real path, with symlinks resolved, so that two spellings of
    one directory give one project. A digest rather than the path itself, because the path
    names a user's home directory and this value is sent to a server.
    """
    digest = hashlib.sha256(root.encode("utf-8")).hexdigest()
    return f"path:{digest[:PATH_HEX_CHARS]}"


def _git(args: "list[str]") -> "str | None":
    """One `git` command's output, or `None` for any failure, including no `git` at all."""
    import subprocess

    try:
        done = subprocess.run(["git", *args], capture_output=True, text=True,
                              timeout=GIT_TIMEOUT_SEC)
    except (OSError, subprocess.SubprocessError):
        return None
    out = done.stdout.strip()
    return out if done.returncode == 0 and out else None


def canonical_project(cwd: str) -> "str | None":
    """The project `cwd` belongs to, or `None` when it is not inside a git repository.

    The remote is read through the repository's *common* git directory, which a linked
    worktree shares with its main repository, so every worktree resolves to one project.
    For the same reason, the path form hashes the main repository's root rather than the
    worktree's own directory.

    Needs git 2.31 or later for `--path-format`. An older git fails that call, and the
    answer is then `None`: no project scope, which is how the hooks behaved before this.
    """
    if not cwd:
        return None
    common = _git(["-C", cwd, "rev-parse", "--path-format=absolute", "--git-common-dir"])
    if common is None:
        return None
    remote = _git(["--git-dir", common, "remote", "get-url", "origin"])
    if remote is not None:
        named = normalise_remote(remote)
        if named is not None:
            return named
    # `<root>/.git` for an ordinary repository; a bare repository is its own root.
    root = os.path.dirname(common) if os.path.basename(common) == ".git" else common
    return path_identity(os.path.realpath(root))


def _fresh(entry: object, now: float) -> bool:
    """Whether a cache entry is well formed and younger than `CACHE_TTL_SECONDS`."""
    if not isinstance(entry, dict):
        return False
    at = entry.get("at")
    return isinstance(at, (int, float)) and now - at <= CACHE_TTL_SECONDS


def _read_cache() -> dict:
    try:
        with open(CACHE, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_cache(data: dict, now: float) -> None:
    """Write the cache through a temporary file and a rename, dropping expired entries.

    Silent on failure: a lost cache costs one more `git` lookup, never a prompt.
    """
    import tempfile

    fresh = {key: entry for key, entry in data.items() if _fresh(entry, now)}
    try:
        directory = os.path.dirname(CACHE)
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".projects-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(fresh, fh)
            os.replace(tmp, CACHE)
        except OSError:
            os.unlink(tmp)
    except OSError:
        pass


def resolve(cwd: str, now: "float | None" = None) -> "str | None":
    """The project to send for `cwd`, or `None` when the switch is off or there is none.

    Cached per directory for `CACHE_TTL_SECONDS`, including a `None` answer, because the
    recall hook calls this on every prompt and a cache miss costs two `git` processes.
    """
    if not enabled(FEATURE):
        return None
    now = time.time() if now is None else now
    key = os.path.abspath(cwd or os.getcwd())
    cache = _read_cache()
    entry = cache.get(key)
    if _fresh(entry, now):
        value = entry.get("project")
        return value if isinstance(value, str) else None
    value = canonical_project(key)
    cache[key] = {"project": value, "at": now}
    _write_cache(cache, now)
    return value


def bind(cwd: str) -> "str | None":
    """Resolve the project for `cwd` and publish it on `ENV` for this process and its children.

    Clears the variable when there is no project, so a value inherited from a parent
    process, or left by an earlier call, is never sent for the wrong repository.
    """
    value = resolve(cwd)
    if value:
        os.environ[ENV] = value
    else:
        os.environ.pop(ENV, None)
    return value
