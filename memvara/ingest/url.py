"""Fetching a URL without letting the caller reach a private network.

A server that fetches any URL a client names can be used to read services that only the
server can reach: a cloud metadata endpoint on `169.254.169.254`, an admin page on
`localhost`, a database on the private network. This is called server-side request
forgery. `SafeFetcher` prevents it with these rules:

1. Only `http` and `https` URLs are fetched.
2. The host name is resolved first, and the fetch is refused if **any** address it
   resolves to is not a public one: private, loopback, link-local, multicast, reserved,
   unspecified, or otherwise not globally routable. IPv4 addresses written inside IPv6
   forms (IPv4-mapped `::ffff:a.b.c.d`, 6to4, Teredo, and NAT64) are unwrapped and checked
   as IPv4. The NAT64 prefixes recognised are the well-known `64:ff9b::/96`, the local-use
   `64:ff9b:1::/48`, and any the operator configures (`nat64_prefixes`, or
   `MEMVARA_NAT64_PREFIXES` for the MCP server). A NAT64 gateway on a prefix nobody
   configured cannot be recognised, because its addresses look like any other IPv6 address.
3. The connection is made to the address that was checked, not to the host name. A DNS
   server that answers with a public address for the check and a private one for the
   connection (DNS rebinding) therefore cannot move the request. TLS still verifies the
   certificate against the host name.
4. Redirects are followed by hand, at most 5, and every new URL goes through rules 1 to 3
   again, so a public page cannot redirect the fetch to a private address.
5. The whole fetch, redirects included, must finish within 20 seconds, and the body may be
   at most 10 MB. A fixed `User-Agent` is sent.

The network code is the `transport`, one HTTP request to one checked address. Tests pass a
fake transport and a fake resolver, so no test needs a network.
"""

from __future__ import annotations

import http.client
import ipaddress
import socket
import ssl
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, Sequence
from urllib.parse import urljoin, urlsplit

from .errors import IngestError

__all__ = ["Fetched", "Fetcher", "SafeFetcher", "nat64_networks", "refusal", "MAX_BYTES",
           "MAX_REDIRECTS", "NAT64_PREFIXES", "TIMEOUT_SEC", "USER_AGENT"]

#: The largest response body that is read. A longer body is refused with `too_large`.
MAX_BYTES = 10 * 1024 * 1024

#: The time allowed for the whole fetch, all redirects included, in seconds.
TIMEOUT_SEC = 20.0

#: How many redirects are followed before the fetch is refused.
MAX_REDIRECTS = 5

#: Sent on every request. A fixed value, so a site owner can recognise and block it.
USER_AGENT = "memvara-ingest/1.0 (+https://memvara.dev)"

#: The NAT64 prefixes every fetcher recognises: the well-known prefix from RFC 6052 and the
#: local-use prefix from RFC 8215. An operator's own prefix is added with `nat64_prefixes`.
NAT64_PREFIXES = ("64:ff9b::/96", "64:ff9b:1::/48")

#: The prefix lengths RFC 6052 defines a layout for. No other length can carry an IPv4
#: address in a way a gateway and this module agree on.
_NAT64_LENGTHS = (32, 40, 48, 56, 64, 96)

_REDIRECTS = frozenset({301, 302, 303, 307, 308})
_CHUNK = 64 * 1024


@dataclass(frozen=True, slots=True)
class Fetched:
    """A fetched response body. `url` is the final URL after redirects."""

    url: str
    content_type: str | None
    body: bytes


class Fetcher(Protocol):
    """What `memvara.ingest.extract` calls to turn a URL into bytes."""

    def fetch(self, url: str) -> Fetched: ...


class Response(Protocol):
    """The part of `http.client.HTTPResponse` the fetcher uses."""

    status: int

    def getheader(self, name: str, default: str | None = None) -> str | None: ...

    def read(self, amt: int) -> bytes: ...

    def settimeout(self, seconds: float) -> None:
        """Limit the next `read` to `seconds`, so the 20-second total holds while the
        body arrives slowly."""
        ...

    def close(self) -> None: ...


#: One request: the URL, the checked address to connect to, and the time left in seconds.
Transport = Callable[[str, str, float], Response]

#: Host name and port to the addresses they resolve to.
Resolver = Callable[[str, int], Sequence[str]]


def nat64_networks(prefixes: Iterable[str]) -> tuple[ipaddress.IPv6Network, ...]:
    """`prefixes` as networks, refusing one that cannot carry an IPv4 address.

    Raises `ValueError` naming the prefix and what is wrong with it.

    >>> nat64_networks(["64:ff9b:1::/48"])
    (IPv6Network('64:ff9b:1::/48'),)
    >>> nat64_networks(["2001:db8::/33"])
    Traceback (most recent call last):
    ...
    ValueError: NAT64 prefix '2001:db8::/33' is 33 bits long; RFC 6052 defines only 32, 40, 48, 56, 64 or 96
    """
    networks = []
    for prefix in prefixes:
        try:
            network = ipaddress.IPv6Network(prefix.strip())
        except ValueError:
            raise ValueError(
                f"NAT64 prefix {prefix!r} is not an IPv6 prefix with its host bits clear, "
                "such as 64:ff9b:1::/48") from None
        if network.prefixlen not in _NAT64_LENGTHS:
            lengths = ", ".join(map(str, _NAT64_LENGTHS[:-1]))
            raise ValueError(
                f"NAT64 prefix {prefix!r} is {network.prefixlen} bits long; RFC 6052 "
                f"defines only {lengths} or {_NAT64_LENGTHS[-1]}")
        networks.append(network)
    return tuple(networks)


_DEFAULT_NAT64 = nat64_networks(NAT64_PREFIXES)


def _nat64_ipv4(address: ipaddress.IPv6Address,
                network: ipaddress.IPv6Network) -> ipaddress.IPv4Address:
    """The IPv4 address inside `address`, laid out as RFC 6052 says for this prefix length.

    After a /96 prefix the IPv4 address is the last four bytes. After a shorter prefix it
    starts right after the prefix, and byte 8 (bits 64 to 71, which must be zero) is
    skipped wherever the four bytes would cross it.
    """
    raw = address.packed
    if network.prefixlen == 96:
        return ipaddress.IPv4Address(raw[12:16])
    start = network.prefixlen // 8
    without_reserved = raw[:8] + raw[9:]
    return ipaddress.IPv4Address(without_reserved[start:start + 4])


def _embedded_ipv4(address: ipaddress.IPv6Address,
                   nat64: Sequence[ipaddress.IPv6Network]) -> ipaddress.IPv4Address | None:
    """The IPv4 address carried inside an IPv6 one, for the forms that route to it."""
    if address.ipv4_mapped is not None:
        return address.ipv4_mapped
    for network in nat64:
        if address in network:
            return _nat64_ipv4(address, network)
    if address.sixtofour is not None:
        return address.sixtofour
    if address.teredo is not None:
        return address.teredo[1]
    return None


def refusal(address: str,
            nat64: Sequence[ipaddress.IPv6Network] = _DEFAULT_NAT64) -> str | None:
    """Why `address` must not be fetched from, or `None` when it is a public address.

    `nat64` is the NAT64 prefixes to unwrap. The default is `NAT64_PREFIXES`;
    `SafeFetcher.refusal` adds the operator's own.

    >>> refusal("93.184.216.34") is None
    True
    >>> refusal("127.0.0.1")
    'a loopback address'
    >>> refusal("::ffff:10.0.0.1")
    'a private address'
    >>> refusal("169.254.169.254")
    'a link-local address'
    """
    try:
        parsed = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return "not an IP address"
    if isinstance(parsed, ipaddress.IPv6Address):
        inner = _embedded_ipv4(parsed, nat64)
        if inner is not None:
            return refusal(str(inner))
    checks = (("is_unspecified", "an unspecified address"),
              ("is_loopback", "a loopback address"),
              ("is_link_local", "a link-local address"),
              ("is_multicast", "a multicast address"),
              ("is_private", "a private address"),
              ("is_reserved", "a reserved address"))
    for attribute, reason in checks:
        if getattr(parsed, attribute):
            return reason
    if not parsed.is_global:
        return "not a globally routable address"
    return None


def _resolve(host: str, port: int) -> list[str]:
    """Every address `host` resolves to, in the resolver's order."""
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    seen: list[str] = []
    for info in infos:
        address = str(info[4][0])
        if address not in seen:
            seen.append(address)
    return seen


class _Hop:
    """An `HTTPResponse`, the socket it reads from, and the connection that owns both.

    The socket is kept here because `http.client` drops the connection's reference to it
    as soon as the response says the connection will close, which a one-off fetch almost
    always does. The response keeps reading from it, so it is the one whose timeout
    must change.
    """

    def __init__(self, connection: http.client.HTTPConnection,
                 response: http.client.HTTPResponse, sock: socket.socket) -> None:
        self._connection = connection
        self._response = response
        self._sock = sock
        self.status = response.status

    def getheader(self, name: str, default: str | None = None) -> str | None:
        return self._response.getheader(name, default)

    def read(self, amt: int) -> bytes:
        return self._response.read(amt)

    def settimeout(self, seconds: float) -> None:
        self._sock.settimeout(seconds)

    def close(self) -> None:
        self._response.close()
        self._connection.close()


def _pinned_transport(url: str, address: str, timeout: float) -> Response:
    """One GET of `url`, connecting to `address` instead of resolving the host again.

    The connection object is built with the real host name, so the `Host` header and the
    TLS certificate check both use it. Only the socket is pointed at the checked address.
    """
    parts = urlsplit(url)
    host = parts.hostname or ""
    connection: http.client.HTTPConnection
    if parts.scheme == "https":
        connection = http.client.HTTPSConnection(
            host, parts.port, timeout=timeout, context=ssl.create_default_context())
    else:
        connection = http.client.HTTPConnection(host, parts.port, timeout=timeout)

    def connect(target: tuple[str, int], timeout: float | None = None,
                source: tuple[str, int] | None = None) -> socket.socket:
        return socket.create_connection((address, target[1]), timeout, source)

    # `_create_connection` is the hook `HTTPConnection.connect` calls to open the socket,
    # and `HTTPSConnection.connect` wraps whatever it returns in TLS for `host`. It has
    # been there, under this name, in every Python this package supports.
    connection._create_connection = connect  # type: ignore[attr-defined]
    path = parts.path or "/"
    if parts.query:
        path += "?" + parts.query
    try:
        connection.request("GET", path,
                           headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
        sock = connection.sock
        return _Hop(connection, connection.getresponse(), sock)
    except BaseException:
        connection.close()
        raise


class SafeFetcher:
    """Fetches `http` and `https` URLs under the rules in this module's docstring.

    `nat64_prefixes` adds the operator's own NAT64 prefixes to `NAT64_PREFIXES`, so an
    address translated through one is checked as the IPv4 address inside it. Each must be
    an IPv6 prefix of 32, 40, 48, 56, 64 or 96 bits, or `ValueError` is raised here.

    `transport`, `resolve` and `clock` exist for tests. Leave them unset in real use.
    """

    def __init__(self, *, transport: Transport | None = None,
                 resolve: Resolver | None = None,
                 clock: Callable[[], float] | None = None,
                 max_bytes: int = MAX_BYTES, timeout: float = TIMEOUT_SEC,
                 max_redirects: int = MAX_REDIRECTS,
                 nat64_prefixes: Iterable[str] = ()) -> None:
        self._nat64 = _DEFAULT_NAT64 + nat64_networks(nat64_prefixes)
        self._transport = transport or _pinned_transport
        self._resolve = resolve or _resolve
        self._clock = clock or time.monotonic
        self.max_bytes = max_bytes
        self.timeout = timeout
        self.max_redirects = max_redirects

    def fetch(self, url: str) -> Fetched:
        deadline = self._clock() + self.timeout
        redirects = 0
        while True:
            address = self._check(url)
            response = self._send(url, address, deadline)
            try:
                if response.status in _REDIRECTS:
                    location = response.getheader("Location")
                    if not location:
                        raise IngestError(
                            "fetch_failed",
                            f"{url} answered with a redirect ({response.status}) that "
                            "names no location")
                    redirects += 1
                    if redirects > self.max_redirects:
                        raise IngestError(
                            "fetch_failed",
                            f"{url} redirected more than {self.max_redirects} times")
                    url = urljoin(url, location)
                    continue
                if not 200 <= response.status < 300:
                    raise IngestError(
                        "fetch_failed", f"{url} answered with HTTP {response.status}")
                body = self._read(url, response, deadline)
                return Fetched(url=url, content_type=response.getheader("Content-Type"),
                               body=body)
            finally:
                response.close()

    def refusal(self, address: str) -> str | None:
        """Why this fetcher would refuse `address`, or `None` when it would connect."""
        return refusal(address, self._nat64)

    def _check(self, url: str) -> str:
        """The address to connect to for `url`, after every rule that can refuse it."""
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https"):
            raise IngestError(
                "url_refused",
                f"only http and https URLs are fetched, and {url!r} is neither")
        host = parts.hostname
        if not host:
            raise IngestError("url_refused", f"{url!r} names no host")
        try:
            port = parts.port or (443 if parts.scheme == "https" else 80)
        except ValueError:
            raise IngestError("url_refused", f"{url!r} has an invalid port") from None
        try:
            addresses = list(self._resolve(host, port))
        except OSError as exc:
            raise IngestError("fetch_failed", f"{host} could not be resolved: {exc}") from None
        if not addresses:
            raise IngestError("fetch_failed", f"{host} resolved to no address")
        for address in addresses:
            reason = self.refusal(address)
            if reason is not None:
                raise IngestError(
                    "url_refused",
                    f"{host} resolves to {address}, which is {reason}; only public "
                    "addresses are fetched")
        return addresses[0]

    def _remaining(self, url: str, deadline: float) -> float:
        left = deadline - self._clock()
        if left <= 0:
            raise IngestError(
                "timeout", f"fetching {url} took longer than {self.timeout:g} seconds")
        return left

    def _send(self, url: str, address: str, deadline: float) -> Response:
        left = self._remaining(url, deadline)
        try:
            return self._transport(url, address, left)
        except TimeoutError:
            raise IngestError(
                "timeout", f"fetching {url} took longer than {self.timeout:g} seconds"
            ) from None
        except (OSError, http.client.HTTPException) as exc:
            raise IngestError("fetch_failed", f"{url} could not be fetched: {exc}") from None

    def _read(self, url: str, response: Response, deadline: float) -> bytes:
        too_large = IngestError(
            "too_large", f"{url} is larger than {self.max_bytes} bytes")
        declared = response.getheader("Content-Length")
        if declared and declared.strip().isdigit() and int(declared) > self.max_bytes:
            raise too_large
        body = bytearray()
        while True:
            response.settimeout(self._remaining(url, deadline))
            try:
                chunk = response.read(_CHUNK)
            except TimeoutError:
                raise IngestError(
                    "timeout", f"fetching {url} took longer than {self.timeout:g} seconds"
                ) from None
            except (OSError, http.client.HTTPException) as exc:
                raise IngestError(
                    "fetch_failed", f"{url} could not be read: {exc}") from None
            if not chunk:
                return bytes(body)
            body += chunk
            if len(body) > self.max_bytes:
                raise too_large
