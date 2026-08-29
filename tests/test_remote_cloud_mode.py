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
