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
from there. `lib.hosted` sends it as the `memvara-project` header on every call (header
names are case-insensitive, so the spec's `Memvara-Project` is the same header). `lib.ipc`
puts it into the daemon's address, so one resident daemon answers for one project. And the
daemon that a hook spawns inherits the variable, so it sends the same header as the hook
that started it. An environment variable is used rather than an argument because the
per-prompt path must not import `lib.hosted`, and because a spawned process inherits it
without any extra plumbing.

`subprocess` and `urllib.parse` are imported inside the functions that need them, and only
run on a cache miss. Together they cost about 4.7ms to import, measured, and this module is
imported on every prompt, where the whole budget is about 30ms.
"""

from __future__ import annotations

import hashlib
import os
import os.path
import time

from .settings import enabled
from .state_file import prune as prune_dir
from .state_file import read_json, write_json

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

#: Where `resolve` remembers each directory's answer: one small file per directory, named
#: by a hash of its absolute path. One file per directory rather than one shared file, so
#: two sessions in different repositories never rewrite each other's entry. Beside the other
#: hook state, not in the plugin, which is replaced on update.
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".memvara", ".hooks", "projects")

#: How long a cached answer is trusted. Working out a project costs one or two `git`
#: processes, measured at about 15ms each, and the recall hook has a budget of roughly 30ms
#: per prompt. An hour means a changed remote is noticed within the hour, at the cost of one
#: lookup per directory per hour. `session_start` removes older files.
CACHE_TTL_SECONDS = 60 * 60

#: Seconds to wait for `git` before treating the directory as having no project.
GIT_TIMEOUT_SEC = 5

#: Longest project name the server accepts. The library's `check_project` uses the same
#: number.
MAX_PROJECT_LENGTH = 512

#: Characters a host name may hold, before its optional `:port`.
_HOST_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789.-")
_HEX = frozenset("0123456789abcdef")


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


def is_canonical(value: str) -> bool:
    """Whether `value` is a project name the server accepts in the `memvara-project` header.

    The same rules as the library's `check_project`, which the hosted deployment applies
    and answers with a 400 when they fail: either `path:` and exactly 16 lower-case hex
    characters, or a lower-case host with an optional port of up to five digits, then at
    least one path segment, where no segment is empty, `.` or `..`, nothing is whitespace
    or a control character, and the whole is at most `MAX_PROJECT_LENGTH` characters.
    Written without `re`, which this per-prompt module does not otherwise import.
    """
    if not value or len(value) > MAX_PROJECT_LENGTH:
        return False
    if value.startswith("path:"):
        digits = value[len("path:"):]
        return len(digits) == PATH_HEX_CHARS and all(c in _HEX for c in digits)
    if any(c.isspace() or not c.isprintable() for c in value):
        return False
    host, _, rest = value.partition("/")
    if not rest:
        return False
    name, colon, port = host.partition(":")
    if colon and not (1 <= len(port) <= 5 and port.isascii() and port.isdigit()):
        return False
    if (not name or any(c not in _HOST_CHARS for c in name)
            or name[0] in ".-" or name[-1] in ".-"):
        return False
    return all(segment not in ("", ".", "..") for segment in rest.split("/"))


def main_root(common_dir: str, paths: "ModuleType" = os.path) -> str:
    """The main working tree for a repository whose common git directory is `common_dir`.

    That is the directory holding `.git`; a bare repository has no working tree, so its own
    directory is named instead. `paths` is the path module, `os.path` by default; a test
    passes `ntpath` to pin the Windows behaviour on any machine. The same as the library's
    `main_root`.
    """
    return (paths.dirname(common_dir) if paths.basename(common_dir) == ".git"
            else common_dir)


def path_identity(root: str) -> str:
    """The project for a repository with no usable remote: a digest of its root's path.

    `root` should already be a real path, with symlinks resolved, so that two spellings of
    one directory give one project. A digest rather than the path itself, because the path
    names a user's home directory and this value is sent to a server.

    Before hashing, the path is put in one spelling, exactly as the library's
    `path_identity` does, so every platform and both copies hash the same string for one
    directory: backslashes become forward slashes, a drive letter is lower-cased, and
    trailing slashes are removed, keeping `/` for the filesystem root. The rest keeps its
    case, because folding it would merge two directories on a case-sensitive volume.
    """
    spelled = root.replace("\\", "/")
    if len(spelled) >= 2 and spelled[1] == ":" and spelled[0].isalpha():
        spelled = spelled[0].lower() + spelled[1:]
    spelled = spelled.rstrip("/") or "/"
    digest = hashlib.sha256(spelled.encode("utf-8")).hexdigest()
    return f"path:{digest[:PATH_HEX_CHARS]}"


def _git(args: "list[str]") -> "str | None":
    """One `git` command's output, or `None` when git failed or is not installed.

    The output is read as bytes and decoded as strict UTF-8, so bytes that are not UTF-8,
    in a remote URL or a directory name, raise `UnicodeDecodeError`. `canonical_project`
    catches that and answers `None`. Decoding as text inside `subprocess.run` raised the
    same error from a place nothing caught, and every hook in such a repository crashed.
    """
    import subprocess

    try:
        done = subprocess.run(["git", *args], capture_output=True, timeout=GIT_TIMEOUT_SEC)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.decode("utf-8").strip() or None


def canonical_project(cwd: str) -> "str | None":
    """The project `cwd` belongs to, or `None` when it has none. Never raises.

    The remote is asked for first, which is one `git` process for the usual repository. The
    common git directory is looked up only when there is no usable remote, for the path
    form. A linked worktree shares its main repository's config and common directory, so
    every worktree resolves to one project either way, and the path form hashes the main
    repository's root rather than the worktree's own directory.

    A remote that normalises to a name the server would refuse (see `is_canonical`) also
    falls back to the path form, exactly as the library's copy does.

    Anything git prints that is not UTF-8, and a `cwd` holding a NUL byte, give `None`: no
    project scope, which is how the hooks behaved before this module existed. The path
    form needs git 2.31 or later for `--path-format`; an older git also gives `None` there.
    """
    if not cwd:
        return None
    try:
        remote = _git(["-C", cwd, "remote", "get-url", "origin"])
        if remote is not None:
            named = normalise_remote(remote)
            # Checked as well as normalised, as the library does, so the hooks never send a
            # header the server refuses: a remote with a `..` segment or an underscore in
            # its host falls back to the path form in both copies.
            if named is not None and is_canonical(named):
                return named
        common = _git(["-C", cwd, "rev-parse", "--path-format=absolute",
                       "--git-common-dir"])
    except ValueError:
        return None
    if common is None:
        return None
    return path_identity(os.path.realpath(main_root(common)))


def _cache_path(key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8", "surrogateescape")).hexdigest()[:32]
    return os.path.join(CACHE_DIR, f"{digest}.json")


def resolve(cwd: str, now: "float | None" = None) -> "str | None":
    """The project to send for `cwd`, or `None` when the switch is off or there is none.

    Cached per directory for `CACHE_TTL_SECONDS`, including a `None` answer, because the
    recall hook calls this on every prompt and a cache miss costs up to two `git` processes.
    The entry repeats the directory it is for, so a hash collision reads as a miss rather
    than as another directory's project.
    """
    if not enabled(FEATURE):
        return None
    now = time.time() if now is None else now
    key = os.path.abspath(cwd or os.getcwd())
    path = _cache_path(key)
    entry = read_json(path)
    at = entry.get("at")
    if (entry.get("cwd") == key and isinstance(at, (int, float))
            and 0 <= now - at <= CACHE_TTL_SECONDS):
        value = entry.get("project")
        return value if isinstance(value, str) else None
    value = canonical_project(key)
    write_json(path, {"cwd": key, "project": value, "at": now}, prefix=".project-")
    return value


def prune(now: "float | None" = None) -> None:
    """Remove cache files older than `CACHE_TTL_SECONDS`. Called once per session."""
    prune_dir(CACHE_DIR, CACHE_TTL_SECONDS, time.time() if now is None else now)


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
