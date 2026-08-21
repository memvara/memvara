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

import memvara.core
from memvara.embed import default_embedder as _real_default_embedder

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
# Isolation in one file remains opt-in the moment someone adds another, so
# the real file is snapshotted at session start and compared at session
# end. A test that needs to write credentials redirects
# `login._CREDENTIALS_PATH` (and `config.CREDENTIALS_PATH`) to tmp_path.

_HOME_CREDENTIALS = pathlib.Path.home() / ".memvara" / "credentials.json"
_CREDENTIALS_SNAPSHOT: Any = None
_CREDENTIALS_EXISTED = False
_CREDENTIALS_SEEN = False


def pytest_sessionstart(session: Any) -> None:
    global _CREDENTIALS_SNAPSHOT, _CREDENTIALS_EXISTED, _CREDENTIALS_SEEN
    _CREDENTIALS_SEEN = True
    _CREDENTIALS_EXISTED = _HOME_CREDENTIALS.is_file()
    _CREDENTIALS_SNAPSHOT = (
        _HOME_CREDENTIALS.read_bytes() if _CREDENTIALS_EXISTED else None
    )


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    if not _CREDENTIALS_SEEN:
        return
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
