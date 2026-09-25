"""Two guards, and no fixtures.

The first is the embedder.

`memvara.embed.default_embedder()` returns a sentence-transformers model as soon as that
package is importable and falls back to `HashingEmbedder` when it is not. So a `Memvara()`
built without an explicit `embedder=` runs a different vector leg depending on what
happens to be installed — and `memvara[rerank]` installs sentence-transformers, because a
cross-encoder is one. Installing the *reranker* extra silently swaps the *embedder*.

Both consequences were measured rather than feared. Sixty-nine `default_embedder()` calls
were reachable from this suite, and each one built a real transformer: the run went from
roughly 27 seconds to 6m59s. The quieter cost is the one this guard exists for — a test
that never mentioned an embedder started asserting about a 384-dimensional semantic space
instead of a 512-dimensional lexical one, and stayed green while doing it. A suite cannot
tell you that its premises moved.

So: no test in `tests/` may reach `default_embedder()`. Pass `embedder=`.
`HashingEmbedder(dim=512)` is not a new choice — it is exactly what `default_embedder()`
returns when sentence-transformers is absent, so pinning it reproduces the behaviour the
tests were written against rather than introducing a third configuration. Under
`tests/test_bench_eval.py`, `evalkit.build_embedder("hashing")` is the same object plus
the cache wrapper, and is what `--embedder hashing` already gives the CLI tests there.

Deliberately narrow in two ways, each protecting coverage that pinning would delete:

* Only `memvara.core.default_embedder` is replaced — the name `Memvara.__init__` looks
  up. `memvara.embed.default_embedder` is untouched, so
  `test_internals.py::test_default_embedder_falls_back_when_local_backend_is_unavailable`
  still calls the real function and still exercises the real fallback. That test is the
  only coverage of the choice this guard forbids everyone else from making.
* A call with no `tests/` frame on the stack is allowed through to the real function. The
  doctests in `memvara/` run under `--doctest-modules` and several build a bare
  `Memvara()` on purpose, because zero-configuration construction is the thing they
  document.

There used to be a third: a `Memvara(...)` written inside `memvara/` was allowed through,
because the test had no lever to pull. `memvara/server/config.py::build_memvara` was the
live case — it builds a `Memvara` from a `ServerConfig`, and `ServerConfig` had no
embedder field and no environment variable behind one, so four tests in `test_server.py`
reached `default_embedder()` and could not stop without a library change. That was a gap
in the shipped MCP server rather than a test defect: the server's vector leg was whatever
the deployment happened to have installed, which is why `pip install memvara[rerank]` made
it refuse to open its own store. `ServerConfig.embedder` — `MEMVARA_EMBEDDER` — closed it,
and this exemption went with it. Every construction site the suite can reach now has a
lever: `embedder=` directly, `MEMVARA_EMBEDDER` through `build_memvara`, or `Memvara`'s
own keywords through `compat.mem0.Memory(**kwargs)`. A test that lands on this guard from
inside `memvara/` again is reporting the next such gap, not a false positive.
"""
from __future__ import annotations

import os
import pathlib
import traceback
from typing import Any

import pytest

import memvara.core
from memvara.embed import default_embedder as _real_default_embedder
# Imported here so `_credentials_never_touch_home` can redirect both names for every
# test. Neither module pulls `httpx` at import time -- `login` imports it inside
# `login()` -- so this does not make the cloud extra a test dependency.
from memvara.remote import creds as creds_module
from memvara.server import config as config_module
from memvara.server import login as login_module
from memvara.store import encryption as encryption_module

# The adversarial suite's tiers and skip ledger (docs/claude/testing.md). `harness` is
# importable here because tests/ has no __init__.py, so pytest puts tests/ on sys.path
# before it imports this file.
from harness import tiers as tiers_module

#: The two constants as the source defines them, read once before any fixture has
#: redirected them. `test_credentials_path_constant_matches_logins_own` asserts the
#: invariant that they are equal by construction, and it has to see the real values to
#: mean anything -- comparing two names the autouse fixture has just pointed at one
#: tmp_path would pass no matter what the source said.
REAL_LOGIN_CREDENTIALS_PATH = login_module._CREDENTIALS_PATH
REAL_CONFIG_CREDENTIALS_PATH = config_module.CREDENTIALS_PATH
#: The real keychain lookup, kept so `tests/test_encryption.py` can exercise it against a
#: fake `keyring` module after the fixture below has replaced it for everyone else.
REAL_READ_KEYCHAIN = encryption_module._read_keychain

_TESTS = str(pathlib.Path(__file__).resolve().parent) + os.sep

_FIX = (
    "Pass embedder= at the construction site. HashingEmbedder(dim=512) is identical to "
    "what default_embedder() returns with sentence-transformers absent; in "
    "tests/test_bench_eval.py use ek.build_embedder(\"hashing\"). If the site named "
    "above is inside memvara/, reach it through that door's own lever instead: "
    "build_memvara() takes ServerConfig.embedder, i.e. MEMVARA_EMBEDDER, and "
    "compat.mem0.Memory(**kwargs) forwards embedder= to Memvara. See tests/conftest.py "
    "for why this is a hard failure rather than a style preference."
)


def _guarded_default_embedder(dim: int = 512) -> Any:
    stack = traceback.extract_stack()[:-1]
    if not any(frame.filename.startswith(_TESTS) for frame in stack):
        return _real_default_embedder(dim)  # a doctest in memvara/, documenting Memvara()

    # The frame that wrote `Memvara(...)`, i.e. the one below `Memvara.__init__`.
    constructor = stack[-2] if len(stack) >= 2 else stack[-1]
    test_frame = next(f for f in reversed(stack) if f.filename.startswith(_TESTS))
    raise AssertionError(
        f"{constructor.filename}:{constructor.lineno} built a Memvara with no embedder=, "
        f"so it fell through to default_embedder() "
        f"(reached from {test_frame.filename}:{test_frame.lineno}).\n\n"
        "default_embedder() returns a sentence-transformers model whenever that package "
        "is importable, and memvara[rerank] installs one. Leaving it unpinned makes this "
        "test's embedding space -- and the suite's runtime -- a property of what happens "
        "to be installed on the machine running it.\n\n"
        f"{_FIX}"
    )


memvara.core.default_embedder = _guarded_default_embedder


# The second guard is a file the suite must never touch.
#
# `memvara-mcp login` writes ~/.memvara/credentials.json mode 0600, and
# tests/test_login.py used to write it too: it isolated the network, the
# browser and the loopback listener, but the filesystem was opt-in, and
# three tests remembered. The others that reached `_write_credentials`
# wrote a fixture key (`key-123`) over a real one. That has now happened
# three times, and each time the hooks that read the file treated the
# resulting 401 as an empty store.
#
# That sentence used to end here, with the snapshot below as the whole
# answer. The snapshot is a detector, not a guard: it fails the session
# *after* the write, so it tells you a real 0600 key was destroyed rather
# than stopping it. The key it replaced is returned exactly once by the
# API and cannot be recovered from anywhere.
#
# So the redirect is now suite-wide and automatic -- `_credentials_never_
# touch_home` below runs for every test in this repository, whether or not
# the file it lives in remembered to ask. `tests/test_login.py` carried an
# autouse fixture doing this for its own 28 tests; the hole was the next
# file, and "the next file" is what happened three times.
#
# The snapshot stays. Two mechanisms for one property is right here: the
# fixture stops the write, and the snapshot catches a write that reached
# the path some way the fixture does not cover -- a subprocess, a second
# constant nobody redirected, an `expanduser` computed at call time.

@pytest.fixture(autouse=True)
def _credentials_never_touch_home(tmp_path, tmp_path_factory, monkeypatch):
    """Point every credentials path at tmp_path, for every test in this repository.

    Autouse and in `conftest.py` rather than in the one file that writes today.
    `login._write_credentials` is the only writer now; the cost of it becoming two
    is a real key, and the redirect is free.

    Both constants, though only `login._CREDENTIALS_PATH` is written today:
    `config.CREDENTIALS_PATH` is the read side of the same file, and a test that
    redirects the write while reading the developer's own key is a test that passes
    for a reason it does not state.

    `raising=False` on neither -- if either name moves, this fixture must fail loudly
    rather than silently protect nothing.
    """
    # `tmp_path / "credentials.json"`, which is where the file-scoped fixture this
    # replaces put it -- several tests in `test_login.py` read that exact path. Nesting a
    # fake `~/.memvara/` under it would be more lifelike and would buy nothing: what makes
    # this isolation is that the path is not the developer's, not its shape.
    # The fake home comes from `tmp_path_factory`, not from `tmp_path`. Creating a
    # directory inside `tmp_path` is visible to the test that owns it, and
    # `test_bench_eval.py::test_a_download_writes_atomically...` asserts its `tmp_path` is
    # empty -- an isolation fixture that makes another test fail has bought nothing.
    home = tmp_path_factory.mktemp("home")
    where = tmp_path / "credentials.json"
    monkeypatch.setattr(login_module, "_CREDENTIALS_PATH", where)
    monkeypatch.setattr(config_module, "CREDENTIALS_PATH", where)
    # `remote/creds.py` does `from ..server.config import CREDENTIALS_PATH`, which binds
    # the value at its own import time -- so patching `config` above does not reach it,
    # and `creds._from_file()` would read the developer's real key. Same shape as the
    # stale `from ... import` in test_config_cloud.py that this change already had to
    # correct, one module over. It only reads, so nothing is destroyed; what it does is
    # let a test pass because whoever ran it happened to be logged in.
    monkeypatch.setattr(creds_module, "CREDENTIALS_PATH", where)
    # And for anything that re-imports in its own interpreter. Six test files spawn
    # subprocesses; a child computes `Path.home()` itself and the monkeypatch above is
    # invisible to it. `Path.home()` reads HOME on POSIX and USERPROFILE on Windows, so
    # setting both makes the child land in tmp_path as well.
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    return where


@pytest.fixture(autouse=True)
def _store_keys_never_touch_the_keychain(monkeypatch):
    """Keep every test away from the developer's OS keychain.

    An encrypted store looks for its key in the OS keychain first. On a developer's Mac
    that is the real login keychain, and a read of it can raise a permission dialog in the
    middle of a test run, or find a real key and quietly use it. So the lookup is replaced
    for every test with one that finds nothing, and `PYTHON_KEYRING_BACKEND` points any
    child process at keyring's null backend, which finds nothing either.

    The key file needs no fixture of its own: it lives under `Path.home()`, and
    `_credentials_never_touch_home` above has already pointed `HOME` at a temporary
    directory for this test and its children.
    """
    monkeypatch.setattr(encryption_module, "_read_keychain", lambda: (None, None))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.null.Keyring")
    monkeypatch.delenv("MEMVARA_DB_KEY", raising=False)


@pytest.fixture(autouse=True)
def _hook_logs_never_touch_home(tmp_path_factory, monkeypatch):
    """Point the plugin hooks' home directory at a temporary one, for every test.

    `plugin/hooks/lib/ipc.py` reads the home directory once, into `_HOME`, when it is
    imported, and a hook test file imports it while pytest collects, before the fixture
    above has moved `HOME`. So `log_line` wrote every test's log lines into the
    developer's real `~/.memvara/.hooks/recall.log`. That is how the hosted client's
    tests left lines saying "hosted rejected min_score" in a real log, three at a time,
    where they read as evidence about the live service. `ipc._HOME` is read at call
    time, so patching it here reaches `log_line` and every state file built from it.

    Only when the hooks' `lib.ipc` has been imported, which is the only case with a
    frozen path to correct. A test that sets `_HOME` itself still does; it runs after
    this and wins.
    """
    import sys  # noqa: PLC0415 - only this fixture needs it

    ipc = sys.modules.get("lib.ipc")
    if ipc is not None and hasattr(ipc, "_HOME"):
        monkeypatch.setattr(ipc, "_HOME", str(tmp_path_factory.mktemp("hook-home")))


def pytest_configure(config: Any) -> None:
    config.addinivalue_line(
        "markers",
        "derives_project: let ServerConfig.from_env() derive the project from a git "
        "remote, which tests/conftest.py otherwise switches off")


def pytest_addoption(parser: Any) -> None:
    parser.addoption(
        "--tier", choices=tiers_module.TIERS, default="fast",
        help="which tier of tests to collect: fast (the default, and what CI runs), "
             "nightly, weekly, local or quarantine. See docs/claude/testing.md.")


def pytest_ignore_collect(collection_path: pathlib.Path, config: Any) -> bool | None:
    """Leave out every test whose tier --tier does not select.

    Returns True or None, never False. pytest stops at the first hook that returns a
    value, so returning False here would overrule --ignore and every other plugin's
    decision about the same path.
    """
    if tiers_module.ignored(collection_path, config.getoption("--tier")):
        return True
    return None


@pytest.fixture(autouse=True)
def _no_project_from_the_checkout(request, monkeypatch):
    """Stop every test from binding the project of whatever checkout runs the suite.

    `ServerConfig.from_env()` derives the project from the working directory's git
    remote, and pytest runs inside a git repository. Left alone, every test that builds a
    config would shell out to git and bind `github.com/memvara/memvara`, or a fork's
    name, or nothing on a machine without git, so a test's scope would depend on where
    it ran. Two levers, because a test can reach the derivation two ways:
    `MEMVARA_FEATURE_PROJECT_SCOPE=0` for anything that reads the process environment,
    including a child process, and a stub for a test that passes its own `env` mapping.

    A test that is about the derivation opts back in with `@pytest.mark.derives_project`.
    """
    if request.node.get_closest_marker("derives_project") is not None:
        return
    monkeypatch.setenv("MEMVARA_FEATURE_PROJECT_SCOPE", "0")
    monkeypatch.setattr(config_module, "canonical_project", lambda cwd: None)


_HOME_CREDENTIALS = pathlib.Path.home() / ".memvara" / "credentials.json"
_CREDENTIALS_SNAPSHOT: Any = None
_CREDENTIALS_EXISTED = False
_CREDENTIALS_SEEN = False
#: The store key file gets the same detector. Overwriting it would make every encrypted
#: store on the developer's machine unreadable, which is worse than losing an API key:
#: an API key can be minted again, and a store key cannot.
_HOME_DB_KEY = pathlib.Path.home() / ".memvara" / "db.key"
_DB_KEY_SNAPSHOT: Any = None


def pytest_sessionstart(session: Any) -> None:
    global _CREDENTIALS_SNAPSHOT, _CREDENTIALS_EXISTED, _CREDENTIALS_SEEN, _DB_KEY_SNAPSHOT
    _CREDENTIALS_SEEN = True
    _CREDENTIALS_EXISTED = _HOME_CREDENTIALS.is_file()
    _CREDENTIALS_SNAPSHOT = (
        _HOME_CREDENTIALS.read_bytes() if _CREDENTIALS_EXISTED else None
    )
    _DB_KEY_SNAPSHOT = _HOME_DB_KEY.read_bytes() if _HOME_DB_KEY.is_file() else None


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    if not _CREDENTIALS_SEEN:
        return
    key_after = _HOME_DB_KEY.read_bytes() if _HOME_DB_KEY.is_file() else None
    if key_after != _DB_KEY_SNAPSHOT:
        raise AssertionError(
            f"{_HOME_DB_KEY} was created, deleted or rewritten during this suite. That "
            "file is the key to every encrypted store on this machine. A test reached "
            "memvara.store.encryption.key_file() without the HOME redirect in "
            "_credentials_never_touch_home, probably from a subprocess that set its own "
            "environment.")
    existed = _HOME_CREDENTIALS.is_file()
    after = _HOME_CREDENTIALS.read_bytes() if existed else None
    if existed == _CREDENTIALS_EXISTED and after == _CREDENTIALS_SNAPSHOT:
        return
    raise AssertionError(
        f"{_HOME_CREDENTIALS} was created, deleted or rewritten during this "
        "suite. Redirect memvara.server.login._CREDENTIALS_PATH (and "
        "memvara.server.config.CREDENTIALS_PATH) to tmp_path before any call "
        "that can write it. tests/test_login.py used to clobber a real 0600 "
        "key with the fixture key-123 whenever a test forgot; that has now "
        "happened three times."
    )


#: The predicates this suite builds graphs out of. Named here rather than per test so a
#: fixture can declare them in one place; each is genuinely a relation between two things,
#: which is why `born_on` and `notes` are absent — a date and a free-text note are values,
#: and declaring them entity-valued to make a test pass would be declaring something false.
GRAPH_TEST_PREDICATES = (
    "reports_to", "works_at", "lives_in", "uses", "configured_in", "headquartered_in",
    "listed_on", "renamed_as", "mother", "father", "founded_in", "located_now",
    "prefers_tool", "owns_pet", "based_in", "in_country", "deploy_region",
    "hq_city", "works_on",
)


def entity_registry(*names: str):
    """A registry that declares `names` entity-valued, so their claims can carry an edge.

    Since `object_kind` gates traversal, a predicate nobody has declared produces a VALUE
    object and no edge — deliberately, because connectivity is meant to be something a
    vocabulary asks for rather than something string collisions supply. A test that builds
    a graph therefore has to declare one, exactly as a deployment does, and saying so at
    the top of the test is better than the graph quietly being empty.

    **Extends a builtin rather than replacing it**, which is the whole of why this is a
    helper and not four lines inline. A declared spec of the same name replaces the
    builtin outright, so writing `PredicateSpec("lives_in", object_type=("entity",))`
    silently drops that builtin's `ONE` cardinality and its aliases — and a name that is
    itself an alias, like `employed_by_company`, stops folding onto `works_at` and becomes
    a predicate of its own. Both were caught here by tests that had nothing to do with the
    graph, which is the good case; in a deployment the same mistake is a slot that quietly
    stops superseding.
    """
    from dataclasses import replace

    from memvara.schema import BUILTIN_PREDICATES, PredicateRegistry, PredicateSpec

    names = names or GRAPH_TEST_PREDICATES
    plain = PredicateRegistry()
    builtins = {s.name: s for s in BUILTIN_PREDICATES}
    specs = []
    for name in names:
        canonical = plain.normalize(name) or name
        existing = builtins.get(canonical)
        specs.append(replace(existing, object_type=("entity",), graph=True)
                     if existing is not None
                     else PredicateSpec(name=canonical, object_type=("entity",), graph=True))
    return PredicateRegistry(BUILTIN_PREDICATES + tuple(specs))
