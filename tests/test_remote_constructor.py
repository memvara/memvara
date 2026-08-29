"""How a caller asks for a hosted deployment, and how they cannot ask for one by accident.

The third test is the one that matters. Dispatch keys on the explicit argument and never on
the environment, so a script that has always written to a local file cannot start posting
to a hosted store because someone ran `memvara-mcp login` on that machine last month.

Every construction here that reaches a credential passes `api_key=` or sets
`MEMVARA_API_KEY`, and none of them reads `~/.memvara/credentials.json`: a machine with a
real key would otherwise pass these tests for a reason the CI runner does not have.
"""
import pytest

from memvara import HashingEmbedder, Memvara, NullLLM
from memvara.remote.api import RemoteMemvara


def test_remote_memvara_is_not_a_subclass_of_memvara():
    """The mechanic every other test here rests on, and the only one they cannot see.

    Python calls `__init__` when `__new__` returns an instance of `cls` and skips it
    otherwise, so returning a non-subclass is what stops `Memvara.__init__` from running a
    second construction over the top — opening a store and loading an embedding model for
    an object that will never use either.

    Making `RemoteMemvara` a subclass breaks that silently. Every `isinstance` check below
    still passes, the bare-constructor test still passes, and the `__init__` guard never
    fires, because Python would call `RemoteMemvara.__init__` rather than
    `Memvara.__init__`. This assertion is the only thing standing between that change and
    a green suite.
    """
    assert not issubclass(RemoteMemvara, Memvara)


def test_an_api_key_returns_a_remote_client():
    with Memvara(api_key="k", base_url="https://example.test") as mem:
        assert isinstance(mem, RemoteMemvara)


def test_a_base_url_alone_is_also_a_request_for_a_remote_client(monkeypatch):
    monkeypatch.setenv("MEMVARA_API_KEY", "from-env")
    with Memvara(base_url="https://example.test") as mem:
        assert isinstance(mem, RemoteMemvara)


def test_a_bare_constructor_stays_local_even_when_the_environment_holds_a_key(monkeypatch):
    monkeypatch.setenv("MEMVARA_API_KEY", "from-env")
    monkeypatch.setenv("MEMVARA_SERVER_URL", "https://example.test")
    mem = Memvara(":memory:", llm=NullLLM(), embedder=HashingEmbedder(dim=512))
    assert not isinstance(mem, RemoteMemvara)
    mem.close()


def test_connect_is_the_door_for_ambient_credentials(monkeypatch):
    monkeypatch.setenv("MEMVARA_API_KEY", "from-env")
    monkeypatch.setenv("MEMVARA_SERVER_URL", "https://example.test")
    with Memvara.connect() as mem:
        assert isinstance(mem, RemoteMemvara)


@pytest.mark.parametrize("kwargs", [
    {"path": ":memory:"},
    {"store": object()},
])
def test_credentials_are_refused_alongside_a_local_store(kwargs):
    with pytest.raises(TypeError):
        Memvara(api_key="k", **kwargs)


@pytest.mark.parametrize("name", ["embedder", "llm", "registry"])
def test_server_side_subsystems_are_refused_rather_than_ignored(name):
    with pytest.raises(TypeError) as caught:
        Memvara(api_key="k", base_url="https://example.test", **{name: object()})
    assert name in str(caught.value)


def test_reembed_is_refused_when_it_asks_for_something_and_ignored_when_it_does_not():
    """`reembed` is the one local-only argument whose default is not `None`.

    Refusing it by "was it passed at all" would make `Memvara(api_key=..., reembed=False)`
    an error, and that call asks for nothing — it is what a wrapper forwarding its own
    default writes. `reembed=True` asks the engine to re-encode a store this process does
    not have, and there is no such engine here, so it is refused by name.
    """
    with pytest.raises(TypeError, match="reembed"):
        Memvara(api_key="k", base_url="https://example.test", reembed=True)
    with Memvara(api_key="k", base_url="https://example.test", reembed=False) as mem:
        assert isinstance(mem, RemoteMemvara)


def test_scope_is_passed_through_to_the_remote_client():
    with Memvara(api_key="k", base_url="https://example.test", user="alice") as mem:
        assert mem.default_scope.user == "alice"


def test_the_constructor_makes_no_network_call(monkeypatch):
    """Constructing resolves a credential and builds a connection pool, and stops there.

    The guard is armed rather than assumed. `httpx.Client.send` is replaced with something
    that raises, and the second half proves it live by making a real method call and
    catching it — without that, this test would pass just as well against a client whose
    transport was never wired up at all, which is the failure mode of every test that
    asserts an absence.
    """
    import httpx

    def refuse(*args, **kwargs):
        raise AssertionError("a request was sent")

    monkeypatch.setattr(httpx.Client, "send", refuse)
    with Memvara(api_key="k", base_url="https://example.test") as mem:
        with pytest.raises(AssertionError, match="a request was sent"):
            mem.get_all()
