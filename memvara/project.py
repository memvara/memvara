"""Which repository a working directory belongs to, as one stable project name.

Every clone and every worktree of one repository should share one project scope, so that a
fact learned in one checkout is recalled in the next. This module turns a working
directory into that name. It is the identity rule from `docs/SUBJECT-CONVENTIONS.md`
section 7, and the MCP server, the plugin hooks and the hosted deployment all use it, so
the same checkout resolves to the same project on every surface.

There are three outcomes:

- A repository with an `origin` remote resolves to the remote, normalised to
  `host/owner/repo`. An `https` remote and an `ssh` remote for one repository resolve to
  the same name, and so do two worktrees of it, because the remote is read from the
  repository's common git directory rather than from the worktree.
- A repository with no usable remote resolves to `path:` followed by the first 16
  hexadecimal characters of the SHA-256 of the main working tree's real path. This name
  is provisional: it changes if the repository moves on disk, and it becomes the remote
  form once the repository is pushed.
- A directory that is not inside a git repository resolves to `None`, which means no
  project scope at all. That is what every caller had before this module existed.

Git is run as a subprocess through a `run` callable that a caller can replace, so a test
can supply the answers git would give without a repository on disk.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from typing import Any, Callable, Optional, Sequence
from urllib.parse import urlsplit

__all__ = ["GitRunner", "canonical_project", "check_project", "main_root",
           "normalize_remote", "path_identity"]

#: Runs `git` with the given arguments in the given directory and returns its standard
#: output with surrounding whitespace removed, or `None` when git is missing, fails or
#: prints nothing. `canonical_project` accepts any callable of this shape.
GitRunner = Callable[[Sequence[str], str], Optional[str]]

#: Seconds one git call may take. Both calls read local files only, so anything slower
#: than this is a hung filesystem, and the server should start without a project rather
#: than wait on it.
_GIT_TIMEOUT = 5.0

#: Forges whose owner and repository names are case-insensitive. On these hosts
#: `github.com/Memvara/Memvara` and `github.com/memvara/memvara` are one repository, so
#: the name is folded to lower case. On any other host the case is kept, because folding
#: it on a forge that treats case as significant would merge two different projects.
_CASE_INSENSITIVE_HOSTS = frozenset({"github.com", "gitlab.com", "bitbucket.org"})

_HOST = re.compile(r"^[a-z0-9]([a-z0-9.-]*[a-z0-9])?(:[0-9]{1,5})?$")
_PATH_NAME = re.compile(r"^path:[0-9a-f]{16}$")

#: Longest project name `check_project` accepts. Real remotes are well under 200
#: characters; the ceiling exists so that a header or an environment variable holding a
#: pasted document is refused rather than stored in every scope key.
MAX_PROJECT_LENGTH = 512


def normalize_remote(url: str) -> str | None:
    """A git remote URL as `host/owner/repo`, or `None` when it names no remote host.

    The plugin hooks carry a copy of this function, because they run without the library
    installed. The rules both copies follow are written down as data in
    `tests/fixtures/project_vectors.json`, a byte-identical copy of the hooks'
    `plugin/hooks/lib/project_vectors.json`, and each copy's tests read every row.

    The rules: surrounding whitespace is ignored and credentials are dropped. The host is
    lower-cased, and a port written in the URL is kept. The query, the fragment, empty
    path segments, trailing slashes and one trailing `.git` are removed. `git@host:owner/repo`
    is read as `host/owner/repo`. Owner and repository are lower-cased only on
    `github.com`, `gitlab.com` and `bitbucket.org`, which treat them case-insensitively. An
    SSH alias from `~/.ssh/config` is kept as written rather than resolved, and
    percent-encoded characters are left encoded.

    A local path, a Windows path, a `file://` URL and a URL with no path return `None`,
    because none of them names a repository anybody else can clone. `canonical_project`
    then falls back to the path form.

    >>> normalize_remote("git@github.com:memvara/memvara-cloud.git")
    'github.com/memvara/memvara-cloud'
    >>> normalize_remote("https://user:token@GitHub.com/Memvara/Memvara-Cloud.git/")
    'github.com/memvara/memvara-cloud'
    >>> normalize_remote("https://git.example.com:8443/Team/Repo?x=1#top")
    'git.example.com:8443/Team/Repo'
    >>> normalize_remote("/srv/git/repo.git") is None
    True
    """
    text = url.strip()
    if not text or "\\" in text:
        # A backslash means a Windows path, never a remote with a host in it.
        return None
    port: int | None = None
    if "://" in text:
        parts = urlsplit(text)
        if parts.scheme.lower() == "file":
            return None
        host = parts.hostname or ""
        try:
            port = parts.port
        except ValueError:
            return None
        path = parts.path
    else:
        # `[user@]host:path`, git's short ssh form. A colon after the first slash, or no
        # colon at all, is a local path, and a one-character host is a drive letter.
        head, colon, path = text.partition(":")
        if not colon or "/" in head or len(head) < 2:
            return None
        host = head.rpartition("@")[2].lower()
        path = path.split("#", 1)[0].split("?", 1)[0]
    if not host:
        return None
    segments = [part for part in path.split("/") if part]
    if segments and segments[-1].endswith(".git"):
        segments[-1] = segments[-1][:-len(".git")]
        segments = [part for part in segments if part]
    if not segments:
        return None
    if host in _CASE_INSENSITIVE_HOSTS:
        segments = [part.lower() for part in segments]
    name = host if port is None else f"{host}:{port}"
    return "/".join([name, *segments])


def check_project(value: str) -> str:
    """Return `value` when it is a canonical project name, and raise `ValueError` if not.

    A canonical name is what `canonical_project` produces: either `host/owner/repo`, with
    a lower-case host, an optional port and at least one path segment, or `path:` followed
    by 16 lower-case hexadecimal characters. The error message says which rule the value
    broke, because the MCP server shows it at startup and the hosted deployment returns it
    as the reason for a 400.

    >>> check_project("github.com/memvara/memvara")
    'github.com/memvara/memvara'
    >>> check_project("path:0123456789abcdef")
    'path:0123456789abcdef'
    >>> check_project("my-project")
    Traceback (most recent call last):
    ...
    ValueError: 'my-project' is not a project name: it needs a host and at least one path segment, as in github.com/owner/repo, or the path:<16 hex> form.
    """
    if not isinstance(value, str) or not value:
        raise ValueError("a project name cannot be empty.")
    if len(value) > MAX_PROJECT_LENGTH:
        raise ValueError(
            f"a project name is at most {MAX_PROJECT_LENGTH} characters, and this one is "
            f"{len(value)}.")
    if value.startswith("path:"):
        if _PATH_NAME.match(value):
            return value
        raise ValueError(
            f"{value!r} is not a project name: the path form is 'path:' followed by "
            "exactly 16 lower-case hexadecimal characters.")
    if any(c.isspace() or not c.isprintable() for c in value):
        raise ValueError(
            f"{value!r} is not a project name: it contains whitespace or a control "
            "character.")
    host, _, rest = value.partition("/")
    segments = rest.split("/") if rest else []
    if not segments:
        raise ValueError(
            f"{value!r} is not a project name: it needs a host and at least one path "
            "segment, as in github.com/owner/repo, or the path:<16 hex> form.")
    if not _HOST.match(host):
        raise ValueError(
            f"{value!r} is not a project name: {host!r} is not a lower-case host name "
            "with an optional port.")
    if any(s in ("", ".", "..") for s in segments):
        raise ValueError(
            f"{value!r} is not a project name: a path segment is empty, '.' or '..'.")
    return value


def canonical_project(cwd: str, *, run: GitRunner | None = None) -> str | None:
    """The project that the directory `cwd` belongs to, or `None` outside a repository.

    The remote is read from the repository's common git directory, so every worktree of
    one repository resolves to the same name as its main checkout. Without a usable
    `origin` remote the name is `path:` plus 16 hexadecimal characters of the SHA-256 of
    the main working tree's real path, which is also shared by every worktree.

    `run` replaces the git subprocess. It receives git's arguments and the directory to
    run in, and returns git's trimmed output or `None`.

    >>> answers = {"rev-parse": "/src/app/.git",
    ...            "--git-dir": "git@github.com:acme/app.git"}
    >>> canonical_project("/src/app", run=lambda args, cwd: answers[args[0]])
    'github.com/acme/app'
    >>> canonical_project("/tmp", run=lambda args, cwd: None) is None
    True
    """
    git = _run_git if run is None else run
    # `--path-format=absolute` needs git 2.31 or later. The plugin hooks ask the same
    # question the same way, so the two cannot disagree about which repository a directory
    # is in. The join is kept for a runner that answers with a relative path; joining an
    # absolute path discards `cwd`.
    common = git(["rev-parse", "--path-format=absolute", "--git-common-dir"], cwd)
    if not common:
        return None
    common_dir = os.path.realpath(os.path.join(cwd, common))
    remote = git(["--git-dir", common_dir, "remote", "get-url", "origin"], cwd)
    if remote:
        name = normalize_remote(remote)
        # Checked as well as normalised, so that everything this returns is a name the
        # hosted deployment will accept in a header. A remote whose decoded path holds a
        # space would otherwise produce a project the server refuses with a 400.
        if name is not None and _acceptable(name):
            return name
    return path_identity(os.path.realpath(main_root(common_dir)))


def main_root(common_dir: str, paths: Any = os.path) -> str:
    r"""The main working tree for a repository whose common git directory is `common_dir`.

    That is the directory holding `.git`. A bare repository has no working tree, so its
    own directory is the most stable thing to name. `paths` is the path module to use,
    `os.path` by default; a test passes `ntpath` to pin the Windows behaviour on any
    machine.

    >>> import ntpath
    >>> main_root("C:\\src\\app\\.git", ntpath)
    'C:\\src\\app'
    """
    return (paths.dirname(common_dir) if paths.basename(common_dir) == ".git"
            else common_dir)


def path_identity(root: str) -> str:
    r"""The `path:` project name for a repository whose main working tree is `root`.

    `root` should already be a real path. It is put in one spelling before hashing, so
    that every platform, and the plugin hooks' copy of this function, hash the same
    string for one directory: backslashes become forward slashes, a drive letter is
    lower-cased, and trailing slashes are removed. The rest keeps its case, because
    folding it would merge two directories on a case-sensitive volume. The shared vectors
    file pins this with Windows and POSIX roots.

    >>> path_identity("C:\\Users\\dev\\memvara") == path_identity("c:/Users/dev/memvara")
    True
    """
    spelled = root.replace("\\", "/")
    if len(spelled) >= 2 and spelled[1] == ":" and spelled[0].isalpha():
        spelled = spelled[0].lower() + spelled[1:]
    spelled = spelled.rstrip("/") or "/"
    digest = hashlib.sha256(spelled.encode("utf-8")).hexdigest()
    return f"path:{digest[:16]}"


def _acceptable(name: str) -> bool:
    try:
        check_project(name)
    except ValueError:
        return False
    return True


def _run_git(args: Sequence[str], cwd: str) -> str | None:
    """Run git and return its trimmed output, or `None` for any kind of failure.

    Every failure means the same thing to the caller, which is that this directory has no
    project it can name: git is not installed, the directory does not exist, it is not a
    repository, there is no `origin` remote, or git took longer than `_GIT_TIMEOUT`.
    """
    try:
        done = subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                              timeout=_GIT_TIMEOUT, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    # Decoded here rather than with `text=True`, because git stores a remote as bytes and
    # one that is not UTF-8 would raise `UnicodeDecodeError` out of server startup. Output
    # that cannot be decoded is read as no answer, which the plugin hooks do as well.
    try:
        out = done.stdout.decode("utf-8").strip()
    except UnicodeDecodeError:
        return None
    return out or None
