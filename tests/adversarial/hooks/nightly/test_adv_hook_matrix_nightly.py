"""The fast tier's hook matrices, on all five hosts and with more payloads.

The fast tier sends its hostile payloads to two hosts (test_adv_hook_hostile.py). This
sends them, and three more JSON values that are not objects, to every hook on all five.
It also checks the reading hooks' outcomes on all five (test_adv_hook_outcomes.py checks
two).
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


@pytest.fixture(scope="module")
def outcomes(tmp_path_factory: pytest.TempPathFactory) -> support.Runs:
    return support.outcome_matrix(tmp_path_factory.mktemp("outcomes"), support.HOSTS)


@pytest.mark.parametrize("host, hook, other", support.outcome_cases(
    support.HOSTS, [o for o in support.OUTCOMES if o != "memories injected"]))
def test_injected_memories_are_told_apart_from_every_other_outcome(
        outcomes: support.Runs, host: str, hook: str, other: str) -> None:
    support.check_told_apart(outcomes, host, hook, "memories injected", other)


@pytest.mark.parametrize("host, hook, answered", support.outcome_cases(
    support.HOSTS, ["nothing matches", "memories injected"]))
def test_a_store_that_could_not_be_reached_is_told_apart_from_one_that_answered(
        outcomes: support.Runs, host: str, hook: str, answered: str) -> None:
    support.check_told_apart(outcomes, host, hook, "store unreachable", answered)


@pytest.mark.parametrize("host", support.recall_hosts(support.HOSTS))
def test_recall_that_could_not_ask_the_store_says_so_in_its_log(
        outcomes: support.Runs, host: str) -> None:
    result = outcomes[host, "recall", "store unreachable"]
    assert "failed reason=unknown" in result.log("recall"), result.logs
