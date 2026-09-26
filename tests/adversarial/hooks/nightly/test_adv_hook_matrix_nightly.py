"""The fast tier's hook matrices, on all five hosts and with more payloads.

The fast tier sends its hostile payloads to two hosts (test_adv_hook_hostile.py). This
sends them, and three more JSON values that are not objects, to every hook on all five.
"""

from __future__ import annotations

import pytest

from .. import support

PAYLOADS = {**support.HOSTILE, **support.MORE_HOSTILE}
UNREADABLE = (*support.UNREADABLE, *support.MORE_HOSTILE)


@pytest.fixture(scope="module")
def hostile(tmp_path_factory: pytest.TempPathFactory) -> support.Hostile:
    return support.hostile_matrix(tmp_path_factory.mktemp("hostile"), support.HOSTS,
                                  PAYLOADS)


@pytest.mark.parametrize("host, hook, payload", support.hostile_cases(support.HOSTS, PAYLOADS))
def test_a_hostile_payload_leaves_the_turn_alone(hostile: support.Hostile, host: str,
                                                 hook: str, payload: str) -> None:
    support.check_left_alone(hostile, host, hook, payload)


@pytest.mark.parametrize("host, hook, payload",
                         support.hostile_cases(support.HOSTS, UNREADABLE))
def test_a_payload_that_is_not_a_json_object_is_answered_as_an_empty_one_is(
        hostile: support.Hostile, host: str, hook: str, payload: str) -> None:
    support.check_read_as_empty(hostile, host, hook, payload)


def test_no_hostile_payload_starts_an_extraction(hostile: support.Hostile) -> None:
    support.check_no_extraction(hostile)
