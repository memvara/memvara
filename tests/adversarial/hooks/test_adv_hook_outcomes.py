"""The reading hooks keep their outcomes apart.

Session start and recall can end five ways: nothing is configured; a local store is
configured and cannot open; a hosted store is configured and cannot be reached; the
store answers with nothing that matches (for session start, an empty store); or the store
answers with memories, which the hook injects. A person, or someone reading the hook
logs, must be able to tell these apart, because "nobody investigates an empty store"
(plugin/hooks/recall.py). The design's "One more rule" says the same of a store that
cannot open.

A test compares two outcomes by `support.observed`: the reply, and the lines the run
added to each log, without their timestamps. The fast tier checks Claude Code, which
shows a status line, and Copilot, which shows none; the nightly tier checks all five
hosts.
"""

from __future__ import annotations

import pytest

from harness import known_bugs

from . import support

FAST_HOSTS = ("claude", "copilot")

#: What the status line says for each outcome on Claude Code, the one host that shows a
#: person a status line.
WORDS = [
    ("session_start", "not configured", "not configured"),
    ("session_start", "nothing matches", "session opened"),
    ("session_start", "memories injected", "session opened with 1 memory"),
    ("recall", "not configured", "not configured"),
    ("recall", "store unreachable", "recall failed"),
    ("recall", "nothing matches", "no matching memories"),
    ("recall", "memories injected", "1 memory recalled"),
]


@pytest.fixture(scope="module")
def outcomes(tmp_path_factory: pytest.TempPathFactory) -> support.Runs:
    return support.outcome_matrix(tmp_path_factory.mktemp("outcomes"), FAST_HOSTS)


@pytest.mark.parametrize("host, hook, other", support.outcome_cases(
    FAST_HOSTS, [o for o in support.OUTCOMES if o != "memories injected"]))
def test_injected_memories_are_told_apart_from_every_other_outcome(
        outcomes: support.Runs, host: str, hook: str, other: str) -> None:
    support.check_told_apart(outcomes, host, hook, "memories injected", other)


@pytest.mark.parametrize("host, hook, answered", support.outcome_cases(
    FAST_HOSTS, ["nothing matches", "memories injected"]))
def test_a_store_that_could_not_be_reached_is_told_apart_from_one_that_answered(
        outcomes: support.Runs, host: str, hook: str, answered: str) -> None:
    support.check_told_apart(outcomes, host, hook, "store unreachable", answered)


@pytest.mark.parametrize("host", support.recall_hosts(FAST_HOSTS))
def test_recall_that_could_not_ask_the_store_says_so_in_its_log(
        outcomes: support.Runs, host: str) -> None:
    support.check_recall_says_it_could_not_ask(outcomes, host)


@pytest.mark.parametrize("hook, outcome, words", WORDS)
def test_each_outcome_has_its_own_words_on_a_host_with_a_status_line(
        outcomes: support.Runs, hook: str, outcome: str, words: str) -> None:
    result = outcomes["claude", hook, outcome]
    assert support.status_of("claude", result.reply) == f"⋈ Memvara · {words}"


# -- known bugs: outcomes that are not told apart -------------------------------------------

@pytest.mark.parametrize("host, hook", [
    (host, hook) for host, hook, _ in support.outcome_cases(FAST_HOSTS, ["store cannot open"])])
@known_bugs.xfail("B55")
def test_a_local_store_that_cannot_open_is_not_reported_as_not_configured(
        outcomes: support.Runs, host: str, hook: str) -> None:
    """The design's "One more rule": a store that exists and fails to open is "recall
    failed", not "not configured". plugin/hooks/lib/open.py, `open_store`, answers None for
    every failure to open, which is also its answer when nothing is configured."""
    support.pin_cannot_open(outcomes, host, hook)


@pytest.mark.parametrize("host", support.silent_hosts(support.recall_hosts(FAST_HOSTS)))
@known_bugs.xfail("B56")
def test_nothing_matching_is_told_apart_from_nothing_configured_without_a_status_line(
        outcomes: support.Runs, host: str) -> None:
    """The renderer drops recall's status where a host shows none, and recall logs
    neither outcome."""
    support.pin_nothing_matches(outcomes, host)


@pytest.mark.parametrize("host", FAST_HOSTS)
@known_bugs.xfail("B57")
def test_session_start_says_so_when_it_could_not_reach_the_store(
        outcomes: support.Runs, host: str) -> None:
    """session_start.py names only a spent quota among the reasons a section is missing,
    so a handshake the endpoint refused ends in "nothing stored yet"."""
    support.pin_unreachable(outcomes, host)
