"""Both scoped views satisfy the protocol the MCP tools are written against.

Derived from what `tools.py` actually calls rather than from a list somebody maintains: a
name drifting off that list is the failure that matters, because it would let the server
believe a capability exists.

The regex sees `ctx.memory.<name>` and nothing else, which is not the whole truth —
`_stats` hands `ctx.memory` to `_join_rate`, which calls `connectivity()` on it as a
parameter. That one is named below and pinned by its own test, because a protocol missing
it would type-check and then fail at the one tool that reports the store's health.

**It also cannot tell a call site from prose**, and that has cost something once already:
a `memvara` member stayed in `MemoryAPI` after its last real caller went away, because
`_fold_note`'s docstring still spelled `ctx.memory.memvara.registry` while explaining that
it no longer reads it. The member was not harmless — it handed every handler an attribute
taking `tenant`, which is what `ToolContext`'s docstring says a handler does not have. So
prose in `tools.py` naming one of these members writes it as prose, and the fix for a
false positive here is to reword the sentence rather than to narrow the regex: a regex that
skipped docstrings would also skip nothing useful and start missing real calls in
commented-out code.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from memvara.core import ScopedMemvara
from memvara.remote.api import ScopedRemoteMemvara
from memvara.server import tools
from memvara.server.memory_api import MemoryAPI

#: Resolved from this file, never from the working directory: pytest is run from the
#: repository root today and from a packaged tree tomorrow, and a test that reads the
#: wrong `tools.py` — or no file at all — asserts nothing while passing.
TOOLS_PY = Path(__file__).resolve().parent.parent / "memvara" / "server" / "tools.py"
TOOLS_SOURCE = TOOLS_PY.read_text()

CALLED = set(re.findall(r"ctx\.memory\.([a-z_]+)", TOOLS_SOURCE))

#: Reached on `ctx.memory` through a parameter rather than on the attribute, so the regex
#: above cannot see it. Pinned by `test_the_join_rate_handoff_is_still_a_call_on_ctx_memory`.
INDIRECT = {"connectivity"}


def declared() -> set[str]:
    return {n for n in dir(MemoryAPI) if not n.startswith("_")}


def test_the_protocol_declares_everything_the_tools_call():
    assert CALLED <= declared(), \
        f"tools.py calls undeclared members: {sorted(CALLED - declared())}"


def test_the_protocol_declares_what_the_tools_reach_through_a_parameter():
    assert INDIRECT <= declared(), \
        f"tools.py reaches undeclared members: {sorted(INDIRECT - declared())}"


def test_the_join_rate_handoff_is_still_a_call_on_ctx_memory():
    """`INDIRECT` is a hand-written name, so pin the two facts that make it true.

    If `_stats` stops handing `ctx.memory` to `_join_rate`, or `_join_rate` stops calling
    `connectivity`, this test says so and the entry above becomes stale rather than
    silently wrong.
    """
    assert "memory.connectivity()" in inspect.getsource(tools._join_rate)
    assert "_join_rate(ctx.memory)" in inspect.getsource(tools._stats)


def test_the_protocol_declares_nothing_the_tools_do_not_use():
    """A protocol wider than its caller is a promise nobody checked.

    Every member here has to be reachable from `tools.py`, because that is the whole
    claim the module makes: one tool table, two engines, and no capability asserted that
    the table does not actually exercise.
    """
    assert declared() == CALLED | INDIRECT, \
        f"declared but unused: {sorted(declared() - (CALLED | INDIRECT))}"


@pytest.mark.parametrize("impl", [ScopedMemvara, ScopedRemoteMemvara])
def test_both_implementations_provide_every_declared_member(impl):
    missing = {n for n in declared() if not hasattr(impl, n)}
    assert not missing, f"{impl.__name__} is missing: {sorted(missing)}"


def test_standing_is_optional_and_only_the_remote_view_has_it():
    """`standing` is deliberately outside `MemoryAPI`.

    A `Protocol` has no optional members: declaring it would make `ScopedMemvara` — which
    has no `standing` and does not need one — stop satisfying the protocol its own server
    is typed against. So `_standing` asks for it with `getattr` and keeps today's path
    when it is absent, and the optionality is pinned here rather than in the type.
    """
    assert "standing" not in declared()
    assert hasattr(ScopedRemoteMemvara, "standing")
    assert not hasattr(ScopedMemvara, "standing")


def test_standing_prefers_the_server_side_endpoint_when_the_view_offers_one():
    source = inspect.getsource(tools._standing)
    assert 'getattr(ctx.memory, "standing", None)' in source, \
        "the endpoint has to be asked for by name, or a remote view silently pages"
    assert 'get_all(states=["live"])' in source, \
        "and the local path has to stay, because ScopedMemvara has no standing()"
