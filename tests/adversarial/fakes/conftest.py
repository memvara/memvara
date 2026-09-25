"""Fixtures for the fakes' self-tests. Each one yields a fake and closes it afterwards,
which releases any request still hanging and stops its server."""

from __future__ import annotations

from typing import Iterator

import pytest

from harness.fakes.fake_v1 import FakeV1


@pytest.fixture
def fake_v1() -> Iterator[FakeV1]:
    with FakeV1() as fake:
        yield fake
