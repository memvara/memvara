"""`MEMVARA_MODE=cloud` starts a server that serves its tools from a hosted deployment.

The guard this replaces refused at construction because a `RemoteStore`-backed engine
would list fourteen tools and fail on the first one reached for. That reasoning was right
and is not being overturned: the engine is still never run against a remote store. The
server is now a client of the facade instead, which is what `docs/OPEN-CORE.md` said the
answer was.

Offline throughout. `RemoteMemvara.__init__` resolves a credential and builds a connection
pool; it makes no request, so nothing here needs a transport.
"""

from __future__ import annotations

import pytest

from memvara.remote.api import RemoteMemvara
from memvara.server.config import ConfigError, ServerConfig, build_memvara
from memvara.server.mcp import MemvaraMCPServer


def _cloud(**kw):
    return ServerConfig(mode="cloud", api_key="k",
                        server_url="https://example.test", **kw)


def test_cloud_mode_builds_a_remote_client():
    client = build_memvara(_cloud())
    try:
        assert isinstance(client, RemoteMemvara)
    finally:
        client.close()


def test_cloud_mode_no_longer_refuses_to_build():
    """A bare call with no assertion would pin nothing — `build_memvara` returning `None`
    would pass it — so this asserts the two things the old refusal took away: an object
    comes back, and it is not an engine over a remote store."""
    from memvara.core import Memvara

    client = build_memvara(_cloud())
    try:
        assert client is not None
        assert not isinstance(client, Memvara)
    finally:
        client.close()


def test_a_cloud_config_without_a_key_still_fails_at_construction():
    with pytest.raises(ConfigError, match="no api_key"):
        build_memvara(ServerConfig(mode="cloud", api_key=None,
                                   server_url="https://example.test"))


@pytest.mark.parametrize("field, value", [("llm", "anthropic"), ("embedder", "local")])
def test_naming_a_server_side_subsystem_under_cloud_mode_is_refused(field, value):
    """Extraction and embedding run inside the deployment, so naming one here is a
    setting that would do nothing. Silently ignoring it is the failure: the operator sets
    `MEMVARA_LLM=anthropic`, sees a server start, and believes their writes are being
    extracted by a model that this process never even loads."""
    with pytest.raises(ConfigError) as caught:
        build_memvara(_cloud(**{field: value}))
    assert field in str(caught.value).lower()


@pytest.mark.parametrize("field, value", [("llm", "none"), ("embedder", "hashing")])
def test_leaving_a_server_side_subsystem_at_its_default_is_not_refused(field, value):
    """The refusal has to fire on a *choice*, not on the value every unset environment
    already carries — otherwise cloud mode never starts at all."""
    client = build_memvara(_cloud(**{field: value}))
    client.close()


def test_the_scope_reaches_the_client():
    client = build_memvara(_cloud(user="alice"))
    try:
        assert client.default_scope.user == "alice"
        assert client.default_scope.tenant == "default"
    finally:
        client.close()


def test_the_tenant_reaches_the_client():
    client = build_memvara(_cloud(tenant="acme"))
    try:
        assert client.default_scope.tenant == "acme"
    finally:
        client.close()


def test_the_server_binds_a_scoped_view_of_the_remote_client():
    """The reason cloud mode can start at all: `MemvaraMCPServer` binds a scope once and
    the tool table is typed to a protocol both scoped views satisfy. `RemoteMemvara.scope`
    takes no `tenant` — the deployment resolves it from the bearer token — so the server
    has to bind the other three and let the credential supply the first."""
    from memvara.remote.api import ScopedRemoteMemvara

    config = _cloud(tenant="acme", user="alice")
    server = MemvaraMCPServer(build_memvara(config), read_only=config.read_only,
                              **config.scope_kwargs)
    try:
        bound = server._ctx.memory
        assert isinstance(bound, ScopedRemoteMemvara)
        assert bound.scope.tenant == "acme"
        assert bound.scope.user == "alice"
    finally:
        server.close()


def test_the_server_reports_an_unknown_extractor_against_a_hosted_deployment():
    """`Memvara.extractor` says what *this process* can extract with. A hosted deployment
    extracts server-side and `RemoteMemvara` has no such property, so the context keeps
    its declared default rather than asserting a pipeline this process does not run.
    `/v1/stats` carries the deployment's own answer; reading it would cost a request at
    startup and is not done here."""
    server = MemvaraMCPServer(build_memvara(_cloud()), user="alice")
    try:
        assert server._ctx.extractor == "unknown"
    finally:
        server.close()


def test_a_cloud_server_lists_the_same_tools_a_local_one_does():
    """The old refusal existed because a cloud server would list tools it could not
    serve. It lists them now because the protocol says both engines can."""
    from memvara.server.tools import TOOLS

    server = MemvaraMCPServer(build_memvara(_cloud()), user="alice")
    try:
        assert set(server._tools) == {t.name for t in TOOLS}
    finally:
        server.close()


def test_cloud_mode_without_httpx_fails_where_the_configuration_was_made(monkeypatch):
    """The one reason a cloud server still cannot start, and it lands as a startup error
    rather than a traceback.

    `memvara-mcp init --mode cloud` refuses on the same question. That pairing is the
    point: `init` writes a config it never launches, so if the two commands answered
    differently the gap would be silent by construction — which is exactly what
    `cloud_gap()` existed to prevent, and the reason it needed a replacement rather than a
    deletion.
    """
    import builtins
    import io

    from memvara.server.cli import main

    real_import = builtins.__import__

    def no_httpx(name, *a, **kw):
        if name == "httpx":
            raise ImportError("no module named httpx")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_httpx)
    err = io.StringIO()
    status = main([], env={"MEMVARA_MODE": "cloud", "MEMVARA_API_KEY": "k"},
                  stdout=io.StringIO(), stderr=err)
    assert status == 2
    assert "memvara[cloud]" in err.getvalue()


# -- what the deployment says about itself, asked once at startup ------------------------


class _Answering:
    """A `RemoteMemvara` whose `/v1/stats` answer is supplied rather than fetched.

    Subclassing rather than monkeypatching the transport, because what is under test is
    `MemvaraMCPServer.__init__` reading `service()` — not how `service()` gets its body,
    which `tests/test_remote_reads.py` covers.
    """

    def __init__(self, body, scope):
        self._body = body
        self.default_scope = scope
        self.calls = 0

    def service(self):
        self.calls += 1
        if isinstance(self._body, Exception):
            raise self._body
        return self._body

    def scope(self, *, user=None, agent=None, session=None):
        from memvara.remote.api import ScopedRemoteMemvara

        return ScopedRemoteMemvara.__new__(ScopedRemoteMemvara)

    def close(self):
        pass


def _answering(body, **kw):
    from memvara.types import Scope

    return _Answering(body, Scope("acme", kw.get("user"), None, None))


_ENVELOPE = {"scope": {}, "visible": 2, "tenant_counts": {"claims": 3},
             "extractor": "fast-path+anthropic/claude", "read_only": False}


def test_a_read_only_credential_hides_the_write_tools():
    """The failure this closes: a server started with a read-only API key listed every
    write tool, and the deployment refused them mid-conversation as a 403 — to a model
    that cannot act on it and with whoever configured the deployment not in the room.
    That is the shape of failure the old cloud-mode refusal existed to prevent, one layer
    along.
    """
    from memvara.server.tools import TOOLS

    server = MemvaraMCPServer(_answering({**_ENVELOPE, "read_only": True}), user="alice")
    assert server.read_only is True
    assert not any(t.writes for t in TOOLS if t.name in server._tools)
    assert {t.name for t in TOOLS if not t.writes} == set(server._tools)


def test_a_write_credential_does_not_un_set_a_read_only_server():
    """OR-ed, never overridden. `MEMVARA_READ_ONLY` is a decision somebody made about this
    deployment; a token that happens to allow writes does not revoke it. The credential
    can only narrow."""
    server = MemvaraMCPServer(_answering(_ENVELOPE), user="alice", read_only=True)
    assert server.read_only is True
    assert not any(name.startswith("memory_remember") for name in server._tools)


def test_the_deployments_own_extractor_reaches_memory_stats():
    server = MemvaraMCPServer(_answering(_ENVELOPE), user="alice")
    assert server._ctx.extractor == "fast-path+anthropic/claude"


def test_the_envelope_is_asked_for_once_and_not_once_per_call():
    memory = _answering(_ENVELOPE)
    MemvaraMCPServer(memory, user="alice")
    assert memory.calls == 1


def test_a_deployment_that_cannot_answer_still_lets_the_server_start():
    """Every failure degrades to what this server did before the call existed. A
    deployment that is down, slow, or behind a proxy returning HTML must leave the server
    starting: `extractor` reports "unknown", which is the field's own declared default and
    is honest, and `read_only` falls back to what the operator set."""
    server = MemvaraMCPServer(_answering(RuntimeError("connection refused")), user="alice")
    assert server._ctx.extractor == "unknown"
    assert server.read_only is False


def test_an_envelope_missing_the_fields_is_treated_as_no_answer():
    """A deployment on an older version answers 200 with a body this server cannot read.
    Reading a missing `read_only` as False is right — it is the permissive default the
    operator's own setting then decides — and reading a missing `extractor` as anything
    but "unknown" would be this server naming a pipeline nobody reported."""
    server = MemvaraMCPServer(_answering({"tenant_counts": {}}), user="alice")
    assert server._ctx.extractor == "unknown"
    assert server.read_only is False


def test_a_local_engine_asks_nothing_and_reports_its_own_extractor(tmp_path):
    """Local behaviour is unchanged: `Memvara` has no `service`, so the property answers
    and nothing touches a network."""
    from memvara.core import Memvara
    from memvara.embed import HashingEmbedder
    from memvara.llm import NullLLM

    memory = Memvara(":memory:", user="alice", embedder=HashingEmbedder(dim=512),
                     llm=NullLLM())
    server = MemvaraMCPServer(memory, user="alice")
    try:
        assert server._ctx.extractor == memory.extractor
        assert server.read_only is False
    finally:
        server.close()


# -- the fold note, with no registry anywhere -------------------------------------------


def test_the_fold_note_works_without_a_registry():
    """`_fold_note` used to read `ctx.memory.memvara.registry`, which raised
    `AttributeError` from all three write tools against a hosted deployment — a
    `RemoteMemvara` holds no registry, because the vocabulary lives server-side.

    It reads the claim the store wrote back instead. This calls it with an object that has
    no `memvara` attribute at all, which is stricter than the remote view and is the point:
    the function cannot reach for a registry because there is nothing to reach through.
    """
    from memvara.server.tools import _fold_note
    from memvara.types import Claim

    folded = Claim(subject="user", predicate="prefers_tool", object="ripgrep")
    note = _fold_note("uses_tool", [folded])
    assert "another spelling of 'prefers_tool'" in note
    assert "held under 'prefers_tool'" in note


def test_the_fold_note_stays_quiet_when_the_spelling_survived():
    from memvara.server.tools import _fold_note
    from memvara.types import Claim

    kept = Claim(subject="user", predicate="prefers_tool", object="ripgrep")
    assert _fold_note("prefers_tool", [kept]) == ""
    assert _fold_note("Prefers Tool", [kept]) == "", "the slug is the comparison, not the raw"


def test_the_fold_note_says_nothing_when_nothing_landed():
    """A write that stored nothing has no slot to report. `_receipt_summary` says what
    happened; inventing a fold here would describe a place the fact is not."""
    from memvara.server.tools import _fold_note

    assert _fold_note("uses_tool", []) == ""


def test_a_write_tool_reports_the_fold_against_a_scoped_remote_view():
    """End to end through the tool, on a view with no registry behind it. `remember`
    returns the receipt the facade would have; everything else is the real handler."""
    from memvara.remote.api import ScopedRemoteMemvara
    from memvara.server.tools import TOOLS, ToolContext
    from memvara.types import Claim, Scope, WriteReceipt

    stored = Claim(subject="user", predicate="prefers_tool", object="ripgrep")

    class _Client:
        """Stands in for the `RemoteMemvara` the view delegates to. It has no `registry`,
        which is the whole condition under test — `ScopedRemoteMemvara.memvara` hands this
        object back, and the old `_fold_note` reached through it for one."""

        def remember(self, *a, **kw):
            return WriteReceipt(added=[stored])

    view = ScopedRemoteMemvara.__new__(ScopedRemoteMemvara)
    object.__setattr__(view, "_scope", Scope("acme", "alice", None, None))
    object.__setattr__(view, "_mem", _Client())
    assert not hasattr(view.memvara, "registry")

    tool = next(t for t in TOOLS if t.name == "memory_remember")
    body = tool.run(ToolContext(memory=view),
                    {"subject": "user", "predicate": "uses_tool", "object": "ripgrep"})
    assert "another spelling of 'prefers_tool'" in body
