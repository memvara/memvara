"""No payload breaks the turn.

A hook that fails a prompt is worse than a hook that does nothing (plugin/hooks/run.py),
so every hook must exit 0, print nothing or one JSON object, and stay within its host's
time limit, whatever arrives on its stdin: nothing, text that is not JSON, JSON of the
wrong shape, bytes that are not UTF-8, a lone surrogate, missing, extra or wrongly typed
fields, 8 MB, or nesting 100,000 levels deep.

The fast tier sends every payload to every hook on two hosts that between them cover each
path through the dispatcher: Claude Code, which nests its replies, has a status line and
captures in the hook process, and Cursor, which reads a flat reply under snake_case keys,
has no recall event, and hands capture to a child. The nightly tier sends them to all
five hosts (nightly/test_adv_hook_matrix_nightly.py).
"""

from __future__ import annotations

import pytest

from . import support

FAST_HOSTS = ("claude", "cursor")


@pytest.fixture(scope="module")
def hostile(tmp_path_factory: pytest.TempPathFactory) -> support.Hostile:
    return support.hostile_matrix(tmp_path_factory.mktemp("hostile"), FAST_HOSTS,
                                  support.HOSTILE)


@pytest.mark.parametrize("host, hook, payload",
                         support.hostile_cases(FAST_HOSTS, support.HOSTILE))
def test_a_hostile_payload_leaves_the_turn_alone(hostile: support.Hostile, host: str,
                                                 hook: str, payload: str) -> None:
    support.check_left_alone(hostile, host, hook, payload)


@pytest.mark.parametrize("host, hook, payload",
                         support.hostile_cases(FAST_HOSTS, support.UNREADABLE))
def test_a_payload_that_is_not_a_json_object_is_answered_as_an_empty_one_is(
        hostile: support.Hostile, host: str, hook: str, payload: str) -> None:
    support.check_read_as_empty(hostile, host, hook, payload)


def test_no_hostile_payload_starts_an_extraction(hostile: support.Hostile) -> None:
    support.check_no_extraction(hostile)
