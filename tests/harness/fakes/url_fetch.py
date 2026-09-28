"""Fakes for `memvara.ingest.url.SafeFetcher`'s two collaborators: the resolver and the
transport it is given instead of a real DNS lookup and a real socket.

`FakeResolver` answers a host from a table. `FakeTransport` answers a URL from a table and
records every request it was asked to make, so a test can prove a refusal happened before
the transport was ever reached. `FakeResponse` is the minimal `Response` — a status,
headers and a body — that `SafeFetcher` reads.

tests/test_ingest_url.py and tests/adversarial/security/test_adv_sec_ssrf.py both drive
`SafeFetcher` with these three and import them from here, so the two files cannot drift
into two different fakes for the same collaborator.
"""

from __future__ import annotations

import ipaddress
from typing import Mapping, Sequence


class FakeResponse:
    """A minimal `Response`: a status, headers, and a body read in the chunks given.

    `read_error`, if set, is raised on every `read()` instead of returning a chunk, for a
    test that checks a failure partway through the body.
    """

    def __init__(self, status: int = 200, headers: "Mapping[str, str] | None" = None,
                 chunks: "Sequence[bytes]" = (b"body",),
                 read_error: "Exception | None" = None) -> None:
        self.status = status
        self.headers = dict(headers or {})
        self.chunks = list(chunks)
        self.read_error = read_error
        self.closed = False
        self.timeouts: "list[float]" = []

    def getheader(self, name: str, default: "str | None" = None) -> "str | None":
        return self.headers.get(name, default)

    def read(self, amt: int) -> bytes:
        if self.read_error is not None:
            raise self.read_error
        return self.chunks.pop(0) if self.chunks else b""

    def settimeout(self, seconds: float) -> None:
        self.timeouts.append(seconds)

    def close(self) -> None:
        self.closed = True


class FakeTransport:
    """Answers each URL from a table and records every request it was asked to make.

    `error`, if set, is raised instead of answering, for a test that checks a transport
    failure. A URL with no scripted answer and no error raises `KeyError` naming it, so a
    test that reaches the transport unexpectedly fails on the URL that was not scripted
    rather than on a silent default.
    """

    def __init__(self, answers: "Mapping[str, FakeResponse] | None" = None,
                 error: "Exception | None" = None) -> None:
        self.answers = dict(answers or {})
        self.error = error
        self.calls: "list[tuple[str, str, float]]" = []
        self.responses: "list[FakeResponse]" = []

    def __call__(self, url: str, address: str, timeout: float) -> FakeResponse:
        self.calls.append((url, address, timeout))
        if self.error is not None:
            raise self.error
        response = self.answers[url]
        self.responses.append(response)
        return response


class FakeResolver:
    """Maps a host to the addresses it resolves to.

    `error`, if set, is raised instead of answering, for a test that checks a resolution
    failure. An unmapped host that is itself an IP literal resolves to itself, the way a
    real resolver does, so a redirect to a literal address is checked as that address
    without needing an entry in the table. Any other unmapped host defaults to one public
    address.
    """

    def __init__(self, table: "Mapping[str, Sequence[str]] | None" = None,
                 error: "Exception | None" = None) -> None:
        self.table = dict(table or {})
        self.error = error
        self.asked: "list[tuple[str, int]]" = []

    def __call__(self, host: str, port: int) -> "list[str]":
        self.asked.append((host, port))
        if self.error is not None:
            raise self.error
        if host in self.table:
            return list(self.table[host])
        try:
            ipaddress.ip_address(host)
        except ValueError:
            return ["93.184.216.34"]
        return [host]
