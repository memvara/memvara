"""The URL fetcher's refusal matrix (SSRF).

A server that fetches any URL a client names can be turned into a proxy for the private
network only the server can reach — a cloud metadata endpoint, an admin page on localhost,
a database on a private subnet. `SafeFetcher` resolves the host first and refuses the fetch
if any address it resolves to is not globally routable, connecting to the address it
checked rather than the name. This drives that matrix with a stub resolver and a recording
transport, so nothing here touches the network, and asserts the transport is never reached
on a refusal — a check made after connecting is a check made too late.
"""

from __future__ import annotations

import pytest

from harness.fakes.url_fetch import FakeResolver, FakeResponse, FakeTransport
from memvara.ingest import IngestError, SafeFetcher
from memvara.ingest.url import refusal

PUBLIC = "93.184.216.34"


def _fetcher(transport: FakeTransport, resolver: FakeResolver, **kw) -> SafeFetcher:
    return SafeFetcher(transport=transport, resolve=resolver,
                       clock=lambda: 0.0, **kw)


NON_PUBLIC = [
    ("127.0.0.1", "loopback"),
    ("::1", "loopback"),
    ("10.0.0.7", "private"),
    ("192.168.1.1", "private"),
    ("169.254.169.254", "link-local"),
    ("fe80::1", "link-local"),
    ("224.0.0.1", "multicast"),
    ("0.0.0.0", "unspecified"),
    ("100.64.0.1", "not a globally routable"),      # carrier-grade NAT
    ("::ffff:10.0.0.1", "private"),                # IPv4-mapped
    ("64:ff9b::a9fe:a9fe", "link-local"),          # NAT64 well-known, wraps 169.254.169.254
    ("2002:7f00:1::", "loopback"),                 # 6to4, wraps 127.0.0.1
    ("2001:0:4136:e378:8000:63bf:3fff:fdd2", "private"),  # Teredo
]


@pytest.mark.parametrize("address, _reason", NON_PUBLIC)
def test_a_host_that_resolves_to_a_non_public_address_is_refused_before_connecting(
        address, _reason) -> None:
    """Every non-public address class, including the IPv4-in-IPv6 forms, is refused, and
    the transport is never called — the check happens before any socket opens."""
    transport = FakeTransport()
    fetcher = _fetcher(transport, FakeResolver({"inside.example": [address]}))
    with pytest.raises(IngestError) as caught:
        fetcher.fetch("https://inside.example/")
    assert caught.value.code == "url_refused"
    assert transport.calls == []


@pytest.mark.parametrize("address, reason", NON_PUBLIC)
def test_refusal_names_a_reason_for_each_non_public_class(address, reason) -> None:
    """`refusal` returns a reason (not None) for each non-public address, whatever spelling
    carries it."""
    import re
    assert re.search(reason, refusal(address) or "")


#: Decimal, octal and hex spellings of 127.0.0.1 that a fake resolver would only be
#: pretending to check: `socket.getaddrinfo` on a numeric host does no network lookup, so
#: these are resolved for real, and macOS and Linux agree on all three. The classic dotted
#: octal spelling, "0177.0.0.1", is deliberately not among them — verified directly
#: against both: glibc (Debian's python:3.13-slim image) reads its leading zero as octal
#: and resolves it to 127.0.0.1, while macOS's libc reads it as decimal with an
#: insignificant leading zero and resolves it to 177.0.0.1, a public-looking address.
#: Testing it would pin one platform's answer as if it were the property under test.
NUMERIC_LOOPBACK_SPELLINGS = ["2130706433", "017700000001", "0x7f.1"]  # decimal, octal, hex


@pytest.mark.parametrize("host", NUMERIC_LOOPBACK_SPELLINGS)
def test_a_numeric_spelling_of_a_private_host_is_refused(host) -> None:
    """A host written in decimal, octal or hex is refused whatever spelling the caller
    used, because the fetcher checks the address it resolves to, not the text of the
    host. Unlike the other tests here, this drives the real resolver rather than a fake
    one mapping the host to 127.0.0.1 by fiat: a fake that answers any host that way
    cannot tell a numeric spelling that really resolves to a private address from one
    that does not, so it would pass whether or not this property held."""
    transport = FakeTransport()
    fetcher = SafeFetcher(transport=transport, clock=lambda: 0.0)
    with pytest.raises(IngestError) as caught:
        fetcher.fetch(f"http://{host}/")
    assert caught.value.code == "url_refused"
    assert transport.calls == []


def test_a_dns_name_that_resolves_to_a_private_address_is_refused() -> None:
    """A public-looking name that resolves to a private address (DNS rebinding's first
    move) is refused: resolution happens before the fetch, and every address is checked."""
    transport = FakeTransport()
    fetcher = _fetcher(transport, FakeResolver({"rebind.example": ["10.0.0.5"]}))
    with pytest.raises(IngestError) as caught:
        fetcher.fetch("https://rebind.example/")
    assert caught.value.code == "url_refused"
    assert transport.calls == []


def test_one_private_address_among_public_ones_refuses_the_whole_host() -> None:
    """If any address a host resolves to is non-public, the fetch is refused — a host
    cannot smuggle a private address behind a public one."""
    transport = FakeTransport()
    fetcher = _fetcher(transport, FakeResolver({"mixed.example": [PUBLIC, "10.0.0.7"]}))
    with pytest.raises(IngestError):
        fetcher.fetch("https://mixed.example/")
    assert transport.calls == []


def test_a_public_page_cannot_redirect_the_fetch_to_a_private_address() -> None:
    """A 302 to a private literal is checked on the second hop and refused, so a public
    page cannot bounce the request onto the private network. The transport serves only the
    first hop."""
    redirect = FakeResponse(status=302, headers={"Location": "http://169.254.169.254/"})
    transport = FakeTransport({"https://public.example/": redirect})
    fetcher = _fetcher(transport, FakeResolver({"public.example": [PUBLIC]}))
    with pytest.raises(IngestError) as caught:
        fetcher.fetch("https://public.example/")
    assert caught.value.code == "url_refused"
    assert [c[0] for c in transport.calls] == ["https://public.example/"]


def test_the_connection_is_made_to_the_checked_address_not_the_name() -> None:
    """A public host is fetched, and the transport is handed the address that was checked,
    so a resolver that answered differently a second time could not move the request."""
    transport = FakeTransport({"https://ok.example/": FakeResponse(chunks=(b"ok",))})
    fetcher = _fetcher(transport, FakeResolver({"ok.example": [PUBLIC]}))
    fetched = fetcher.fetch("https://ok.example/")
    assert fetched.body == b"ok"
    assert transport.calls[0][1] == PUBLIC


def test_a_private_ipv4_behind_an_unconfigured_nat64_prefix_is_not_detected() -> None:
    """Documented in docs/LIMITATIONS.md: a NAT64 gateway on a prefix the operator has not
    listed produces addresses that look like any other public IPv6, so a private IPv4 host
    behind it cannot be detected. This is a limit, not a hole — listing the prefix closes
    it, which the second assertion proves — so the test pins the documented behaviour."""
    # 169.254.169.254 wrapped under a prefix inside Cloudflare's (globally routable) range,
    # which no default NAT64 prefix covers.
    prefix = "2606:4700:4700:0:1234:5678:0:0/96"
    wrapped = "2606:4700:4700:0:1234:5678:a9fe:a9fe"
    assert refusal(wrapped) is None
    assert SafeFetcher(nat64_prefixes=[prefix]).refusal(wrapped) is not None
