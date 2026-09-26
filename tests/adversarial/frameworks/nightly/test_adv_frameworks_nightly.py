"""memvara's framework adapters and mem0 shim, run against the real packages.

Each framework gets two virtual environments, one holding the release at the floor
memvara declares and one holding the newest release (`environments.py`), and the probe
runs that framework's checks inside each (`probe.py`). Every check is a test of its own
here, and three more tests check each environment itself: that it holds the release it
pins, that it runs the memvara in this checkout, and that nothing in it reached the
network.

A check that fails because of a known bug carries that bug's strict expected failure,
through PINNED, and raises `known_bugs.Reproduced` only when the failure is that bug's own
symptom. The testing guide describes the whole arrangement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pytest
from packaging.utils import canonicalize_name

from harness import known_bugs

from .. import environments, probe
from . import checks_crewai, checks_langchain, checks_langgraph, checks_llamaindex

#: Each framework's checks, which the probe runs in that framework's environments.
CHECKS: dict[str, Any] = {
    "langchain": checks_langchain,
    "llamaindex": checks_llamaindex,
    "crewai": checks_crewai,
    "langgraph": checks_langgraph,
}

ENVIRONMENTS = [pytest.param(name, pin, id=f"{name}-{pin}")
                for name in CHECKS for pin in environments.PINS]


@dataclass(frozen=True)
class Symptom:
    """What one known bug's failure looks like: its exception type and part of its
    message. A failure with anything else is not that bug."""

    error_type: str
    text: str

    def seen_in(self, result: probe.Result) -> bool:
        return (not result.passed and result.error_type == self.error_type
                and self.text in result.message)


#: Checks that fail today because of a known bug: (framework, pin, check) -> the bug's
#: strict expected failure, and its symptom.
PINNED: dict[tuple[str, str, str], tuple[pytest.MarkDecorator, Symptom]] = {
    # as_query_engine builds its own retriever from its keyword arguments and hands them
    # on to RetrieverQueryEngine.from_args as well, so retriever= arrives there twice.
    ("llamaindex", "floor", "the_retriever_docstring_example_builds_a_query_engine"): (
        known_bugs.xfail("B83"),
        Symptom("TypeError", "got multiple values for argument 'retriever'")),
    ("llamaindex", "latest", "the_retriever_docstring_example_builds_a_query_engine"): (
        known_bugs.xfail("B83"),
        Symptom("TypeError", "got multiple values for argument 'retriever'")),
    # crewai 1.10.1's EncodingFlow takes storage.write_lock, which the StorageBackend
    # protocol does not declare. From 1.11.0 on, CrewAI no longer asks for it.
    ("crewai", "floor", "crewais_memory_remembers_and_recalls_through_the_storage"): (
        known_bugs.xfail("B84"),
        Symptom("AttributeError", "object has no attribute 'write_lock'")),
    # The same bug stops this check before it can reach B85. The fix for B84 moves this
    # pin to B85, with the symptom of the pin below.
    ("crewai", "floor", "a_repeated_memory_reaches_crewais_consolidation"): (
        known_bugs.xfail("B84"),
        Symptom("AttributeError", "object has no attribute 'write_lock'")),
    ("crewai", "latest", "a_repeated_memory_reaches_crewais_consolidation"): (
        known_bugs.xfail("B85"),
        Symptom("AssertionError", "CrewAI asked its model to consolidate 0 times")),
}


def _check_params() -> Iterator[Any]:
    for name, module in CHECKS.items():
        for pin in environments.PINS:
            for check, _ in probe.checks(module):
                short = check.removeprefix("check_")
                pinned = PINNED.get((name, pin, short))
                yield pytest.param(name, pin, short, id=f"{name}-{pin}-{short}",
                                   marks=[pinned[0]] if pinned else [])


def _verify(frameworks: environments.Session, name: str, pin: str, check: str) -> None:
    """Pass when the check passed, raise `Reproduced` when it failed with its pinned
    bug's symptom, and fail with the check's own report otherwise."""
    probed = frameworks.probe(name, pin)
    result = probed.run.result(f"check_{check}")
    pinned = PINNED.get((name, pin, check))
    if pinned is not None and pinned[1].seen_in(result):
        raise known_bugs.Reproduced(f"{check}: {result.error_type}: {result.message}")
    detail = result.describe()
    if result.error_type == "ProbeStopped":
        detail += f"\n\nThe probe's exit status was {probed.exit_code}, and its output ends:\n"
        detail += probed.output
    assert result.passed, detail


@pytest.mark.parametrize("name, pin", ENVIRONMENTS)
def test_the_environment_holds_the_release_it_pins(
        frameworks: environments.Session, name: str, pin: str) -> None:
    """What the checks' own process could import, which the probe's start record lists,
    is the release this pin resolved to."""
    wanted = frameworks.wanted[(name, pin)]
    probed = frameworks.probe(name, pin)
    assert probed.run.start is not None, f"the probe reported nothing:\n{probed.output}"
    installed = probed.run.start["distributions"]
    assert installed.get(canonicalize_name(wanted.framework.dist)) == wanted.version, (
        f"{wanted.framework.dist}: pinned {wanted.version}, installed "
        f"{installed.get(canonicalize_name(wanted.framework.dist))}")


@pytest.mark.parametrize("name, pin", ENVIRONMENTS)
def test_the_environment_runs_the_memvara_in_this_checkout(
        frameworks: environments.Session, name: str, pin: str) -> None:
    """memvara is reinstalled into a reused environment on every run. Every installed file
    must match this checkout's, and the probe must have imported that installed copy."""
    prepared = frameworks.prepare(name, pin)
    assert environments.stale_files(prepared.purelib) == []
    start = frameworks.probe(name, pin).run.start
    assert start is not None and start["memvara"] is not None
    assert Path(start["memvara"]).resolve().is_relative_to(prepared.path.resolve()), (
        start["memvara"])


@pytest.mark.parametrize("name, pin", ENVIRONMENTS)
def test_nothing_in_the_environment_reached_the_network(
        frameworks: environments.Session, name: str, pin: str) -> None:
    """No check may reach the network, and neither may anything else in the probe's
    process, including a framework's background thread or exit handler. The probe must
    have finished, or it did not see every phase of the run."""
    probed = frameworks.probe(name, pin)
    assert probed.run.finished, (f"the probe stopped early, so it did not see every "
                                 f"phase:\n{probed.output}")
    accesses = [f"{record['access']} during {record['phase']}, in thread "
                f"{record['thread']}:\n{''.join(record['stack'][-4:])}"
                for record in probed.run.network]
    assert accesses == [], "\n".join(accesses)


@pytest.mark.parametrize("name, pin, check", list(_check_params()))
def test_the_adapter_keeps_its_promise(
        frameworks: environments.Session, name: str, pin: str, check: str) -> None:
    """One check, which states the promise it tests in its docstring."""
    _verify(frameworks, name, pin, check)
