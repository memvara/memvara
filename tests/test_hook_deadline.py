"""The hook deadline that bounds every hosted call (#345).

The nightly timeout tests run the hooks against an endpoint that never answers; these check
the deadline itself and that the hosted client obeys it without a network.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

HOOKS = pathlib.Path(__file__).resolve().parent.parent / "plugin" / "hooks"
if str(HOOKS) not in sys.path:
    sys.path.insert(0, str(HOOKS))

from lib import deadline, hosted  # noqa: E402


@pytest.fixture(autouse=True)
def _no_deadline():
    deadline.clear()
    yield
    deadline.clear()


class _Connection:
    """Answers every request at once, and records the timeout it was given."""

    def __init__(self) -> None:
        self.timeout = hosted.TIMEOUT_SEC
        self.timeouts: list[float] = []

    def request(self, method, path, body, headers) -> None:
        self.timeouts.append(self.timeout)

    def getresponse(self):
        class _Response:
            status = 200

            def getheader(self, name):
                return None

            def read(self):
                return b'{"jsonrpc": "2.0", "id": 1, "result": {}}'
        return _Response()

    def close(self) -> None:
        pass


def _client(connection: _Connection) -> hosted.HostedRecall:
    client = hosted.HostedRecall("key", "http://127.0.0.1:9")
    client._connect = lambda: connection  # type: ignore[method-assign]
    return client


def test_with_no_deadline_a_call_waits_the_usual_time() -> None:
    connection = _Connection()
    assert _client(connection)._rpc("ping") is not None
    assert deadline.left() is None
    assert connection.timeouts == [hosted.TIMEOUT_SEC]


def test_a_call_waits_no_longer_than_the_time_left() -> None:
    connection = _Connection()
    deadline.set_from_limit(deadline.MARGIN_SEC + 2.0)
    assert _client(connection)._rpc("ping") is not None
    assert 0 < connection.timeouts[0] <= 2.0


def test_no_call_starts_once_the_deadline_has_passed() -> None:
    """Not even a connection is made, so a hook that has used its time answers at once."""
    deadline.set_from_limit(0)
    client = hosted.HostedRecall("key", "http://127.0.0.1:9")
    client._connect = lambda: pytest.fail("connected after the deadline")  # type: ignore
    assert client._rpc("ping") is None


def test_a_limit_shorter_than_the_margin_leaves_no_time() -> None:
    deadline.set_from_limit(deadline.MARGIN_SEC / 2)
    left = deadline.left()
    assert left is not None and left <= 0
