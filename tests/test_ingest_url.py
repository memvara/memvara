"""`SafeFetcher`: the rules that stop a fetched URL reaching a private network.

Every test but the last group injects the transport, the resolver and the clock, so
nothing here resolves a real name or opens a socket to anything but this machine, and no
test sleeps. Each refused address class has its own case, and each checks that the
transport was never called, because a refusal after connecting is too late. The last
group drives the real transport against a server on 127.0.0.1, called directly so the
loopback rule does not apply, to check what goes over the wire.
"""

from __future__ import annotations

import http.client
import http.server
import re
import socket
import threading

import pytest

from memvara.ingest import IngestError, SafeFetcher
from memvara.ingest import url as url_module
from memvara.ingest.url import USER_AGENT, _pinned_transport, _resolve, refusal

PUBLIC = "93.184.216.34"


class FakeResponse:
    def __init__(self, status=200, headers=None, chunks=(b"body",), read_error=None):
        self.status = status
        self.headers = dict(headers or {})
        self.chunks = list(chunks)
        self.read_error = read_error
        self.closed = False
        self.timeouts: list[float] = []

    def getheader(self, name, default=None):
        return self.headers.get(name, default)

    def read(self, amt):
        if self.read_error is not None:
            raise self.read_error
        return self.chunks.pop(0) if self.chunks else b""

    def settimeout(self, seconds):
        self.timeouts.append(seconds)

    def close(self):
        self.closed = True


class FakeTransport:
    """Answers each URL from a table and records every request it was asked to make."""

    def __init__(self, answers=None, error=None):
        self.answers = answers or {}
        self.error = error
        self.calls: list[tuple[str, str, float]] = []
        self.responses: list[FakeResponse] = []

    def __call__(self, url, address, timeout):
        self.calls.append((url, address, timeout))
        if self.error is not None:
            raise self.error
        response = self.answers[url]
        self.responses.append(response)
        return response


class FakeResolver:
    def __init__(self, table=None, error=None):
        self.table = table or {}
        self.error = error
        self.asked: list[tuple[str, int]] = []

    def __call__(self, host, port):
        self.asked.append((host, port))
        if self.error is not None:
            raise self.error
        return self.table.get(host, [PUBLIC])


class Clock:
    def __init__(self, *readings):
        self.readings = list(readings)
        self.last = 0.0

    def __call__(self):
        if self.readings:
            self.last = self.readings.pop(0)
        return self.last


def fetcher(transport, resolver=None, clock=None, **kw):
    return SafeFetcher(transport=transport, resolve=resolver or FakeResolver(),
                       clock=clock or Clock(0.0), **kw)


def refused(code, fetch, url):
    with pytest.raises(IngestError) as caught:
        fetch.fetch(url)
    assert caught.value.code == code, caught.value.reason
    return caught.value.reason


# -- the happy path -----------------------------------------------------------------


def test_a_public_url_is_fetched_from_the_address_that_was_checked():
    transport = FakeTransport({"https://example.com/a": FakeResponse(
        headers={"Content-Type": "text/plain"}, chunks=(b"he", b"llo"))})
    resolver = FakeResolver({"example.com": [PUBLIC]})
    got = fetcher(transport, resolver, Clock(100.0)).fetch("https://example.com/a")
    assert (got.url, got.content_type, got.body) == (
        "https://example.com/a", "text/plain", b"hello")
    assert transport.calls == [("https://example.com/a", PUBLIC, 20.0)]
    assert resolver.asked == [("example.com", 443)]
    assert transport.responses[0].closed
    assert transport.responses[0].timeouts == [20.0, 20.0, 20.0]


def test_the_port_in_the_url_is_the_port_resolved():
    resolver = FakeResolver()
    transport = FakeTransport({"http://example.com:8080/": FakeResponse()})
    fetcher(transport, resolver).fetch("http://example.com:8080/")
    assert resolver.asked == [("example.com", 8080)]


def test_plain_http_defaults_to_port_80():
    resolver = FakeResolver()
    fetcher(FakeTransport({"http://example.com/": FakeResponse()}), resolver).fetch(
        "http://example.com/")
    assert resolver.asked == [("example.com", 80)]


# -- schemes and malformed URLs -----------------------------------------------------------


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "ftp://example.com/x", "gopher://example.com/",
    "javascript:alert(1)", "example.com/no-scheme",
])
def test_only_http_and_https_are_fetched(url):
    transport = FakeTransport()
    reason = refused("url_refused", fetcher(transport), url)
    assert "only http and https" in reason
    assert transport.calls == []


def test_a_url_with_no_host_is_refused():
    transport = FakeTransport()
    refused("url_refused", fetcher(transport), "http:///path")
    assert transport.calls == []


def test_a_url_with_an_invalid_port_is_refused():
    transport = FakeTransport()
    refused("url_refused", fetcher(transport), "http://example.com:99999/")
    assert transport.calls == []


def test_a_host_that_does_not_resolve_is_a_failed_fetch():
    transport = FakeTransport()
    resolver = FakeResolver(error=socket.gaierror(8, "nodename nor servname provided"))
    reason = refused("fetch_failed", fetcher(transport, resolver), "https://nope.invalid/")
    assert "could not be resolved" in reason and transport.calls == []


def test_a_host_that_resolves_to_nothing_is_a_failed_fetch():
    transport = FakeTransport()
    resolver = FakeResolver({"empty.example": []})
    refused("fetch_failed", fetcher(transport, resolver), "https://empty.example/")
    assert transport.calls == []


# -- every refused address class ---------------------------------------------------------


@pytest.mark.parametrize("address, reason", [
    ("127.0.0.1", "loopback"),
    ("127.8.9.10", "loopback"),
    ("::1", "loopback"),
    ("10.1.2.3", "private"),
    ("172.16.0.1", "private"),
    ("192.168.1.1", "private"),
    ("fc00::1", "private"),
    ("fd12:3456::1", "private"),
    ("169.254.169.254", "link-local"),
    ("fe80::1", "link-local"),
    ("fe80::1%en0", "link-local"),
    ("224.0.0.1", "multicast"),
    ("239.255.255.250", "multicast"),
    ("ff02::1", "multicast"),
    ("0.0.0.0", "unspecified"),
    ("::", "unspecified"),
    # Addresses Python files under "reserved" on one version and "private" on another;
    # what matters is that they are refused, which the reason pattern allows for.
    ("240.0.0.1", "reserved|private"),
    ("255.255.255.255", "reserved|private"),
    ("::127.0.0.1", "reserved|loopback"),
    # Shared address space for carrier-grade NAT: neither private nor global.
    ("100.64.0.1", "not a globally routable|private"),
    # IPv4 inside IPv6: mapped, NAT64, 6to4 and Teredo all unwrap to the IPv4 address.
    ("::ffff:127.0.0.1", "loopback"),
    ("::ffff:10.0.0.1", "private"),
    ("::ffff:169.254.169.254", "link-local"),
    ("64:ff9b::a00:1", "private"),
    ("2002:7f00:1::", "loopback"),
    ("2002:a9fe:a9fe::", "link-local"),
    ("2001:0:4136:e378:8000:63bf:3fff:fdd2", "private"),
])
def test_every_non_public_address_class_is_refused_before_connecting(address, reason):
    transport = FakeTransport()
    resolver = FakeResolver({"inside.example": [address]})
    message = refused("url_refused", fetcher(transport, resolver),
                      "https://inside.example/")
    assert address in message
    assert refusal(address) is not None
    assert re.search(reason, refusal(address) or "")
    assert transport.calls == []


def test_a_literal_private_address_in_the_url_is_refused():
    transport = FakeTransport()
    resolver = FakeResolver({"127.0.0.1": ["127.0.0.1"], "::1": ["::1"]})
    refused("url_refused", fetcher(transport, resolver), "http://127.0.0.1:8000/admin")
    refused("url_refused", fetcher(transport, resolver), "http://[::1]/")
    assert transport.calls == []


def test_one_private_address_among_public_ones_refuses_the_host():
    transport = FakeTransport()
    resolver = FakeResolver({"mixed.example": [PUBLIC, "10.0.0.7"]})
    refused("url_refused", fetcher(transport, resolver), "https://mixed.example/")
    assert transport.calls == []


def test_something_that_is_not_an_address_is_refused():
    assert refusal("not-an-ip") == "not an IP address"


@pytest.mark.parametrize("address", [PUBLIC, "8.8.8.8", "2606:4700:4700::1111",
                                     "::ffff:8.8.8.8"])
def test_public_addresses_are_allowed(address):
    assert refusal(address) is None


# -- redirects ------------------------------------------------------------------------


def test_a_redirect_that_turns_private_is_refused_before_the_second_request():
    transport = FakeTransport({"https://example.com/start": FakeResponse(
        302, {"Location": "http://internal.example/secrets"})})
    resolver = FakeResolver({"example.com": [PUBLIC],
                             "internal.example": ["10.0.0.5"]})
    reason = refused("url_refused", fetcher(transport, resolver),
                     "https://example.com/start")
    assert "10.0.0.5" in reason
    assert [call[0] for call in transport.calls] == ["https://example.com/start"]
    assert transport.responses[0].closed


def test_a_redirect_to_the_metadata_address_is_refused():
    transport = FakeTransport({"https://example.com/": FakeResponse(
        301, {"Location": "http://169.254.169.254/latest/meta-data/"})})
    resolver = FakeResolver({"169.254.169.254": ["169.254.169.254"]})
    refused("url_refused", fetcher(transport, resolver), "https://example.com/")
    assert len(transport.calls) == 1


def test_a_redirect_to_another_scheme_is_refused():
    transport = FakeTransport({"https://example.com/": FakeResponse(
        302, {"Location": "file:///etc/passwd"})})
    refused("url_refused", fetcher(transport), "https://example.com/")
    assert len(transport.calls) == 1


def test_a_relative_redirect_is_resolved_and_checked_again():
    transport = FakeTransport({
        "https://example.com/a/b": FakeResponse(307, {"Location": "../c?x=1"}),
        "https://example.com/c?x=1": FakeResponse(chunks=(b"moved",)),
    })
    resolver = FakeResolver()
    got = fetcher(transport, resolver).fetch("https://example.com/a/b")
    assert got.url == "https://example.com/c?x=1" and got.body == b"moved"
    assert resolver.asked == [("example.com", 443), ("example.com", 443)]
    assert all(response.closed for response in transport.responses)


def test_five_redirects_are_followed_and_a_sixth_is_refused():
    hops = {f"https://example.com/{n}": FakeResponse(
        302, {"Location": f"/{n + 1}"}) for n in range(6)}
    hops["https://example.com/6"] = FakeResponse(chunks=(b"end",))
    five = dict(hops)
    five["https://example.com/5"] = FakeResponse(chunks=(b"end",))
    assert fetcher(FakeTransport(five)).fetch("https://example.com/0").body == b"end"

    transport = FakeTransport(hops)
    reason = refused("fetch_failed", fetcher(transport), "https://example.com/0")
    assert "more than 5 times" in reason
    assert len(transport.calls) == 6


def test_a_redirect_with_no_location_is_a_failed_fetch():
    transport = FakeTransport({"https://example.com/": FakeResponse(303)})
    refused("fetch_failed", fetcher(transport), "https://example.com/")


# -- statuses and limits ----------------------------------------------------------------


def test_a_status_that_is_not_2xx_is_a_failed_fetch():
    transport = FakeTransport({"https://example.com/": FakeResponse(404)})
    reason = refused("fetch_failed", fetcher(transport), "https://example.com/")
    assert "HTTP 404" in reason and transport.responses[0].closed


def test_a_declared_length_over_the_limit_is_refused_without_reading():
    response = FakeResponse(headers={"Content-Length": "11"}, chunks=(b"x" * 11,))
    transport = FakeTransport({"https://example.com/": response})
    refused("too_large", fetcher(transport, max_bytes=10), "https://example.com/")
    assert response.chunks == [b"x" * 11]


def test_a_body_that_grows_past_the_limit_is_refused():
    transport = FakeTransport({"https://example.com/": FakeResponse(
        headers={"Content-Length": "garbage"}, chunks=(b"x" * 6, b"x" * 6, b"x"))})
    refused("too_large", fetcher(transport, max_bytes=10), "https://example.com/")


def test_the_default_limits_are_the_documented_ones():
    fetch = SafeFetcher()
    assert (fetch.max_bytes, fetch.timeout, fetch.max_redirects) == (
        10 * 1024 * 1024, 20.0, 5)


def test_the_time_limit_covers_every_redirect():
    transport = FakeTransport({
        "https://example.com/1": FakeResponse(302, {"Location": "/2"}),
        "https://example.com/2": FakeResponse(),
    })
    # Starts at 0, the first request leaves 20 s, and the second hop begins at 21 s.
    reason = refused("timeout", fetcher(transport, clock=Clock(0.0, 0.0, 21.0)),
                     "https://example.com/1")
    assert "20 seconds" in reason
    assert [call[2] for call in transport.calls] == [20.0]


def test_a_body_that_arrives_too_slowly_is_refused():
    response = FakeResponse(chunks=(b"a", b"b", b"c"))
    transport = FakeTransport({"https://example.com/": response})
    clock = Clock(0.0, 1.0, 5.0, 19.5, 20.5)
    refused("timeout", fetcher(transport, clock=clock), "https://example.com/")
    # Each read is limited to the time that is left, not to a fresh 20 seconds.
    assert response.timeouts == [15.0, 0.5]


@pytest.mark.parametrize("error, code", [
    (TimeoutError("timed out"), "timeout"),
    (socket.timeout("timed out"), "timeout"),
    (ConnectionRefusedError(61, "refused"), "fetch_failed"),
    (http.client.RemoteDisconnected("closed"), "fetch_failed"),
])
def test_transport_failures_have_their_own_codes(error, code):
    transport = FakeTransport(error=error)
    refused(code, fetcher(transport), "https://example.com/")


@pytest.mark.parametrize("error, code", [
    (TimeoutError("timed out"), "timeout"),
    (ConnectionResetError(54, "reset"), "fetch_failed"),
    (http.client.IncompleteRead(b"par"), "fetch_failed"),
])
def test_failures_while_reading_the_body_have_their_own_codes(error, code):
    response = FakeResponse(read_error=error)
    transport = FakeTransport({"https://example.com/": response})
    refused(code, fetcher(transport), "https://example.com/")
    assert response.closed


# -- the default resolver ---------------------------------------------------------------


def test_the_default_resolver_returns_each_address_once_in_order(monkeypatch):
    asked = []

    def getaddrinfo(host, port, type=0):
        asked.append((host, port, type))
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.2.3.4", port)),
                (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2001:db8::1", port, 0, 0)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.2.3.4", port))]

    monkeypatch.setattr(url_module.socket, "getaddrinfo", getaddrinfo)
    assert _resolve("example.com", 443) == ["1.2.3.4", "2001:db8::1"]
    assert asked == [("example.com", 443, socket.SOCK_STREAM)]


# -- the real transport, against a server on this machine ---------------------------------


class _Recorder(http.server.BaseHTTPRequestHandler):
    seen: list[dict] = []

    def do_GET(self):  # noqa: N802 - the name http.server calls
        type(self).seen.append({"path": self.path, "host": self.headers["Host"],
                                "agent": self.headers["User-Agent"]})
        body = b"served locally"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def local_server():
    _Recorder.seen = []
    server = http.server.HTTPServer(("127.0.0.1", 0), _Recorder)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


def test_the_transport_connects_to_the_address_and_names_the_host(local_server):
    # `unresolvable.invalid` cannot resolve, so the only way this request can arrive is
    # through the pinned address, which is the point of pinning it.
    url = f"http://unresolvable.invalid:{local_server}/path?q=1"
    response = _pinned_transport(url, "127.0.0.1", 5.0)
    try:
        assert response.status == 200
        assert response.getheader("Content-Type") == "text/plain"
        # The server speaks HTTP/1.0, so `http.client` has already dropped the
        # connection's own reference to the socket. The timeout must still reach the
        # socket the body is read from.
        response.settimeout(3.5)
        assert response._sock.gettimeout() == 3.5  # type: ignore[attr-defined]
        assert response.read(1024) == b"served locally"
    finally:
        response.close()
    assert _Recorder.seen == [{"path": "/path?q=1",
                               "host": f"unresolvable.invalid:{local_server}",
                               "agent": USER_AGENT}]


def test_the_transport_asks_for_the_root_when_the_url_has_no_path(local_server):
    response = _pinned_transport(f"http://example.test:{local_server}", "127.0.0.1", 5.0)
    response.close()
    assert _Recorder.seen[0]["path"] == "/"


def test_https_is_spoken_as_tls_to_the_pinned_address(local_server):
    # The local server speaks plain HTTP, so a TLS handshake with it fails. That failure
    # is the evidence that https:// URLs are sent over TLS.
    with pytest.raises(OSError):
        _pinned_transport(f"https://example.test:{local_server}/", "127.0.0.1", 5.0)


# -- NAT64 prefixes -----------------------------------------------------------------------
#
# A NAT64 gateway reaches an IPv4 host through an IPv6 address that carries the IPv4
# address inside it, laid out as RFC 6052 describes for the prefix length. Unless the
# prefix is known, the IPv6 address can look public while the IPv4 one behind it is
# private. The well-known prefix and the RFC 8215 local-use prefix are always known; an
# operator's own prefix must be configured.

#: Prefixes that look like ordinary global address space, as an operator's own would.
OPERATOR_96 = "2a01:4f8:c0c:64::/96"
OPERATOR_64 = "2a01:4f8:c0c:64::/64"

#: 10.0.0.1 behind `OPERATOR_96`. Without the prefix configured it looks public.
HIDDEN_PRIVATE = "2a01:4f8:c0c:64::a00:1"


@pytest.mark.parametrize("address, reason", [
    ("64:ff9b:1:a00:0:100::", "private"),          # 10.0.0.1 in the /48 layout
    ("64:ff9b:1:a9fe:a9:fe00::", "link-local"),    # 169.254.169.254 in the /48 layout
])
def test_the_local_use_nat64_prefix_is_unwrapped_by_its_48_bit_layout(address, reason):
    transport = FakeTransport()
    resolver = FakeResolver({"nat.example": [address]})
    message = refused("url_refused", fetcher(transport, resolver), "https://nat.example/")
    assert reason in message and transport.calls == []


def test_a_public_address_behind_the_local_use_prefix_is_allowed():
    # 8.8.8.8 in the /48 layout. Allowed only because the IPv4 address was read from the
    # right bytes: recent Pythons file the whole /48 as private.
    assert refusal("64:ff9b:1:808:8:800::") is None


def test_an_operator_prefix_is_not_recognised_until_it_is_configured():
    assert refusal(HIDDEN_PRIVATE) is None
    transport = FakeTransport()
    resolver = FakeResolver({"nat.example": [HIDDEN_PRIVATE]})
    fetch = SafeFetcher(transport=transport, resolve=resolver, clock=Clock(0.0),
                        nat64_prefixes=[OPERATOR_96])
    reason = refused("url_refused", fetch, "https://nat.example/")
    assert "a private address" in reason and transport.calls == []


def test_an_operator_prefix_of_64_bits_skips_the_reserved_octet():
    transport = FakeTransport()
    resolver = FakeResolver({"nat.example": ["2a01:4f8:c0c:64:7f:0:100:0"]})  # 127.0.0.1
    fetch = SafeFetcher(transport=transport, resolve=resolver, clock=Clock(0.0),
                        nat64_prefixes=(OPERATOR_64,))
    assert "loopback" in refused("url_refused", fetch, "https://nat.example/")
    assert transport.calls == []


def test_a_public_address_behind_an_operator_prefix_is_still_fetched():
    transport = FakeTransport({"https://nat.example/": FakeResponse()})
    resolver = FakeResolver({"nat.example": ["2a01:4f8:c0c:64::808:808"]})    # 8.8.8.8
    fetch = SafeFetcher(transport=transport, resolve=resolver, clock=Clock(0.0),
                        nat64_prefixes=[OPERATOR_96])
    assert fetch.fetch("https://nat.example/").body == b"body"


@pytest.mark.parametrize("prefix, problem", [
    ("2a01:4f8::/33", "32, 40, 48, 56, 64 or 96"),
    ("10.0.0.0/8", "not an IPv6 prefix"),
    ("nonsense", "not an IPv6 prefix"),
    ("2a01:4f8:c0c:64::1/96", "not an IPv6 prefix"),
])
def test_a_prefix_that_cannot_carry_an_ipv4_address_is_refused(prefix, problem):
    with pytest.raises(ValueError, match=problem):
        SafeFetcher(nat64_prefixes=[prefix])


_ENV = {"MEMVARA_DB": ":memory:", "MEMVARA_FEATURE_PROJECT_SCOPE": "0"}


def test_the_server_reads_operator_prefixes_from_the_environment():
    from memvara.server.config import ServerConfig

    config = ServerConfig.from_env(
        {**_ENV, "MEMVARA_NAT64_PREFIXES": f" {OPERATOR_96} , {OPERATOR_64},"})
    assert config.nat64_prefixes == (OPERATOR_96, OPERATOR_64)
    fetch = config.url_fetcher()
    assert isinstance(fetch, SafeFetcher)
    assert fetch.refusal(HIDDEN_PRIVATE) == "a private address"


def test_the_server_has_no_operator_prefixes_by_default():
    from memvara.server.config import ServerConfig

    config = ServerConfig.from_env(dict(_ENV))
    assert config.nat64_prefixes == ()
    assert config.url_fetcher().refusal(HIDDEN_PRIVATE) is None


def test_a_bad_prefix_in_the_environment_is_refused_at_startup():
    from memvara.server.config import ConfigError, ServerConfig

    with pytest.raises(ConfigError, match="MEMVARA_NAT64_PREFIXES"):
        ServerConfig.from_env({**_ENV, "MEMVARA_NAT64_PREFIXES": "2a01:4f8::/33"})
