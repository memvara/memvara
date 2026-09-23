"""The three feature switches this repository's closure and link tools answer to.

`forget_matching` and `links` each own tools, and switching one off hides them the way
`profile` hides `memory_profile`. `end_reason` owns arguments rather than a tool, so
switching it off removes `reason` and `until_reason` from every schema that has them,
and a call that still sends one is refused as an unknown argument rather than accepted
and dropped. Stored reasons stay visible, because they are records.
"""
import pytest

from memvara import HashingEmbedder, Memvara, NullLLM
from memvara.server import MemvaraMCPServer
from memvara.server.tools import TOOLS, without_reasons

from test_server import call, text


def server(**kw) -> MemvaraMCPServer:
    mem = Memvara(llm=NullLLM(), embedder=HashingEmbedder(dim=64), user="alice")
    return MemvaraMCPServer(mem, user="alice", **kw)


def listed(srv: MemvaraMCPServer) -> dict[str, dict]:
    tools = srv.handle_message({"jsonrpc": "2.0", "id": 1,
                                "method": "tools/list"})["result"]["tools"]
    return {t["name"]: t["inputSchema"]["properties"] for t in tools}


@pytest.mark.parametrize("feature, hidden", [
    ("forget_matching", {"memory_end_matching", "memory_forget_matching"}),
    ("links", {"memory_link"}),
])
def test_switching_a_feature_off_hides_its_tools_and_says_why(feature, hidden):
    srv = server(features_off={feature})
    names = set(listed(srv))
    assert not hidden & names
    assert {"memory_end", "memory_forget", "memory_why"} <= names
    for name in hidden:
        body, is_error = call(srv, name, {"query": "x", "from_id": "a", "to_id": "b",
                                          "relation": "extends"})
        assert is_error and body == (
            f"{name} is unavailable: the {feature} feature is switched off on this "
            f"memory server (MEMVARA_FEATURE_{feature.upper()}=0).")


def test_every_tool_and_argument_is_offered_with_nothing_switched_off():
    tools = listed(server())
    assert {"memory_end_matching", "memory_forget_matching", "memory_link"} <= set(tools)
    assert "reason" in tools["memory_end"] and "until_reason" in tools["memory_remember"]


def test_switching_end_reason_off_removes_every_reason_argument_and_nothing_else():
    tools = listed(server(features_off={"end_reason"}))
    for name, props in tools.items():
        assert "reason" not in props and "until_reason" not in props, name
    assert "replaces" in tools["memory_remember"], "lineage is not a reason"
    assert "memory_end_matching" in tools and "memory_link" in tools
    # Only the reason arguments went: every other property of every tool is still there.
    full = {t.name: set(t.properties) for t in TOOLS}
    for name, props in tools.items():
        assert full[name] - set(props) <= {"reason", "until_reason"}, name


def test_a_reason_sent_to_a_server_with_end_reason_off_is_refused_not_dropped():
    srv = server(features_off={"end_reason"})
    text(srv, "memory_remember", {"predicate": "works_at", "object": "Acme"})
    body, is_error = call(srv, "memory_end", {"predicate": "works_at", "reason": "left"})
    assert is_error and "unknown argument(s)" in body
    assert server_state(srv) == ["live"], "nothing was ended by the refused call"


def test_a_reason_stored_earlier_is_still_shown_with_end_reason_off():
    mem = Memvara(llm=NullLLM(), embedder=HashingEmbedder(dim=64), user="alice")
    claim = mem.remember("user", "works_at", "Acme").added[0]
    mem.delete(claim.id, close="ended", reason="left for Globex")
    srv = MemvaraMCPServer(mem, user="alice", features_off={"end_reason"})
    assert "ended because: left for Globex" in text(srv, "memory_history",
                                                    {"predicate": "works_at"})


def test_the_rewrite_leaves_a_tool_without_reason_arguments_untouched():
    rewritten = dict(zip([t.name for t in TOOLS], without_reasons(TOOLS)))
    original = {t.name: t for t in TOOLS}
    assert rewritten["memory_search"] is original["memory_search"]
    assert rewritten["memory_end"] is not original["memory_end"]


def server_state(srv: MemvaraMCPServer) -> list[str]:
    return [c.state for c in srv._ctx.memory.get_all(include_invalidated=True)]
