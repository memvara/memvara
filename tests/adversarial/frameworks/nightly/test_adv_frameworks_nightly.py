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
from typing import Any, Iterable, Iterator

import pytest
from packaging.utils import canonicalize_name

from harness import known_bugs

from .. import environments, probe
from . import (checks_crewai, checks_langchain, checks_langgraph, checks_llamaindex,
               checks_mem0)

#: Each framework's checks, which the probe runs in that framework's environments.
CHECKS: dict[str, Any] = {
    "langchain": checks_langchain,
    "llamaindex": checks_llamaindex,
    "crewai": checks_crewai,
    "langgraph": checks_langgraph,
    "mem0": checks_mem0,
}

ENVIRONMENTS = [pytest.param(name, pin, id=f"{name}-{pin}")
                for name in CHECKS for pin in environments.PINS]


@dataclass(frozen=True)
class Symptom:
    """What one known bug's failure looks like: its exception type and its message, or a
    part of it. A failure with anything else is not that bug."""

    error_type: str
    text: str
    #: Whether the message must be exactly `text`, rather than contain it. A message that
    #: lists every difference found must match whole, or a new difference added to the
    #: end of the list would pass as the known bug.
    whole: bool = False

    def seen_in(self, result: probe.Result) -> bool:
        if result.passed or result.error_type != self.error_type:
            return False
        return result.message == self.text if self.whole else self.text in result.message


#: Checks that fail today because of a known bug: (framework, pin, check) -> the bug's
#: strict expected failure, and its symptom.
PINNED: dict[tuple[str, str, str], tuple[pytest.MarkDecorator, Symptom]] = {
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
        Symptom("AssertionError", "CrewAI asked its model to consolidate 0 times: the "
                "stored copy scored 0.50 against CrewAI's threshold of 0.85, and 2 live "
                "copies remain", whole=True)),
}


#: Checks that tests of their own report, because each covers an item on the coverage
#: checklist and a covers mark applies to every case of the test that carries it.
ONE_CLOCK = (("langgraph", "a_replaced_field_ends_and_a_dropped_field_is_retired"),
             ("crewai", "update_ends_the_old_text_and_delete_retires_it"))
FRAMED = (("llamaindex", "the_prompt_frames_memory_as_reference_data"),)


def _param(name: str, pin: str, short: str) -> Any:
    pinned = PINNED.get((name, pin, short))
    return pytest.param(name, pin, short, id=f"{name}-{pin}-{short}",
                        marks=[pinned[0]] if pinned else [])


def _check_params() -> Iterator[Any]:
    """Every check not reported by a test of its own, at both pins."""
    reported_elsewhere = set(ONE_CLOCK) | set(FRAMED)
    for name, module in CHECKS.items():
        for pin in environments.PINS:
            for check, _ in probe.checks(module):
                short = check.removeprefix("check_")
                if (name, short) not in reported_elsewhere:
                    yield _param(name, pin, short)


def _own_params(checks: Iterable[tuple[str, str]]) -> list[Any]:
    return [_param(name, pin, short) for name, short in checks for pin in environments.PINS]


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


def test_every_framework_has_its_checks() -> None:
    """A framework added to `environments.FRAMEWORKS` without a checks module here would
    never be tested, and nothing else would say so."""
    assert set(CHECKS) == {framework.name for framework in environments.FRAMEWORKS}
    for name, module in CHECKS.items():
        assert Path(module.__file__) == environments.framework(name).checks
        assert probe.checks(module), f"{name} has no checks"


def test_every_pin_names_a_check_its_framework_defines() -> None:
    """Every entry in PINNED, ONE_CLOCK and FRAMED names a check that its framework's
    checks file defines, and every pin in PINNED is one of the two pins. An entry for a
    check that was renamed or removed would apply to nothing: the known bug's expected
    failure would quietly stop being checked, and nothing else would say so."""
    defined = {(name, check.removeprefix("check_")) for name, module in CHECKS.items()
               for check, _ in probe.checks(module)}
    stale: list[tuple[str, ...]] = [
        key for key in PINNED
        if key[1] not in environments.PINS or (key[0], key[2]) not in defined]
    stale += [pair for pair in (*ONE_CLOCK, *FRAMED) if pair not in defined]
    assert stale == [], stale


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


@pytest.mark.covers("inv:I5")
@pytest.mark.parametrize("name, pin", ENVIRONMENTS)
def test_nothing_in_the_environment_reached_the_network(
        frameworks: environments.Session, name: str, pin: str) -> None:
    """No check may reach the network, and neither may anything else in the probe's
    process, including a framework's background thread or exit handler. The probe must
    have finished, or it did not see every phase of the run, and it must have exited
    cleanly, or the interpreter's shutdown, where telemetry often sends, did not finish.

    The probe runs with no credential in its environment, so this is invariant 5 of
    docs/INTERNALS.md, "the library must run with no API key and no network", for every
    adapter and the mem0 shim, under the real framework."""
    probed = frameworks.probe(name, pin)
    # The accesses come first: one that made the probe stop early, or exit with a
    # failure, must still be named as the access it was.
    accesses = [f"{record['access']} during {record['phase']}, in thread "
                f"{record['thread']}:\n{''.join(record['stack'][-4:])}"
                for record in probed.run.network]
    assert accesses == [], "\n".join(accesses)
    failed = probed.run.load_error
    loading = (f" (loading {failed.check} failed with {failed.error_type}: "
               f"{failed.message})" if failed is not None else "")
    assert probed.run.finished, (f"the probe stopped early{loading}, so it did not see "
                                 f"every phase:\n{probed.output}")
    assert probed.exit_code == 0, (f"the probe exited with status {probed.exit_code}, so "
                                   f"its shutdown did not finish cleanly:\n{probed.output}")


def test_the_environments_fit_the_disk_budget(frameworks: environments.Session) -> None:
    """The environments must fit in the disk budget the plan sets, 8 GB. A new release
    that grows past it fails here, with each environment's size, before the disk fills.
    An environment that failed to build is left out: its own tests report that. When none
    built there is nothing to measure, and the test fails rather than passing on an empty
    list."""
    ready = []
    for name in CHECKS:
        for pin in environments.PINS:
            try:
                ready.append(frameworks.prepare(name, pin))
            except environments.BuildError:
                continue
    assert ready, "no environment was built, so the disk budget was not checked"
    problem = environments.over_budget(ready)
    assert problem is None, problem


@pytest.mark.parametrize("name, pin, check", list(_check_params()))
def test_the_adapter_keeps_its_promise(
        frameworks: environments.Session, name: str, pin: str, check: str) -> None:
    """One check, which states the promise it tests in its docstring."""
    _verify(frameworks, name, pin, check)


@pytest.mark.covers("inv:I3")
@pytest.mark.parametrize("name, pin, check", _own_params(ONE_CLOCK))
def test_ending_a_value_through_an_adapter_moves_exactly_one_clock(
        frameworks: environments.Session, name: str, pin: str, check: str) -> None:
    """Invariant 3 of docs/INTERNALS.md, through the two adapters that end values: a value
    that is replaced has its world clock closed and its belief clock left open, a value
    that is deleted has its belief clock closed and its world clock left open, and
    neither loses its row."""
    _verify(frameworks, name, pin, check)


@pytest.mark.covers("inv:RT3")
@pytest.mark.parametrize("name, pin, check", _own_params(FRAMED))
def test_the_prompt_an_adapter_builds_frames_memory_as_data(
        frameworks: environments.Session, name: str, pin: str, check: str) -> None:
    """The recall header names the text as data, and a stored sentence cannot forge
    structure around itself, in the prompt LlamaIndex builds from the memory block."""
    _verify(frameworks, name, pin, check)
