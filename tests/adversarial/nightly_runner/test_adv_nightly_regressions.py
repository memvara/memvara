"""The regressions step: run the suite's nightly tier, and turn what failed into findings.

The step runs pytest through `scripts/nightly/pytest_results.py`, which writes one JSON
line per test. Each failed test is rerun twice, and becomes a finding whose invariant is
its node id. When the failure carries the replay program the reference model's state
machine prints, the program becomes the finding's operations, so two different breaks
found by the same test get two fingerprints. Two cases are not breaks in a test: a strict
expected failure that passes means a known bug may be fixed, and a run that fails with no
failed test means something outside the tests failed. Both are reported for a person and
never filed.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
from typing import Any

import pytest

from harness.env import REPO, child_env
from harness.report import Finding

if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))
from nightly import regressions  # noqa: E402 - scripts/ is not on the path until above

#: One of each outcome the step must tell apart.
OUTCOMES = '''
import pytest


def test_passes():
    pass


def test_fails():
    assert 1 == 2, "one is not two"


@pytest.fixture
def broken():
    raise RuntimeError("the fixture broke")


def test_setup_errors(broken):
    pass


@pytest.mark.skip(reason="not on this machine")
def test_skipped():
    pass


@pytest.mark.xfail(strict=True, reason="a known bug")
def test_expected_failure():
    assert False


@pytest.mark.xfail(strict=True, reason="a known bug that is fixed now")
def test_strict_unexpected_pass():
    pass


@pytest.mark.xfail(reason="a lenient marker")
def test_unexpected_pass():
    pass
'''

#: A failure shaped like the reference model's state machine: the replay program is a
#: note, and Hypothesis prints the call that reproduces the example.
MACHINE = '''
from hypothesis import given, note, settings, strategies as st


@settings(max_examples=5, derandomize=True, database=None, print_blob=True)
@given(st.integers())
def test_random_operations(n):
    note("drive.replay([\\n    Remember(user='u1', predicate='lives_in', object='Berlin'),"
         "\\n    Erase(user='u1'),\\n])")
    assert False, "after operation 1: the store and the model disagree"
'''

#: Stands in for the checkout's root conftest.py, which registers --tier.
ACCEPTS_TIER = '''
def pytest_addoption(parser):
    parser.addoption("--tier", default="fast")
'''


@pytest.fixture(scope="module")
def night(tmp_path_factory: pytest.TempPathFactory) -> tuple[pathlib.Path, Any]:
    """The step's own command, run once on a small checkout holding every outcome and a
    file that fails to import."""
    checkout = tmp_path_factory.mktemp("checkout")
    (checkout / "conftest.py").write_text(ACCEPTS_TIER)
    (checkout / "test_outcomes.py").write_text(OUTCOMES)
    (checkout / "test_machine.py").write_text(MACHINE)
    (checkout / "test_broken_import.py").write_text("import a_module_that_does_not_exist\n")
    results = checkout / "results.jsonl"
    subprocess.run(regressions.command(sys.executable, results), cwd=checkout,
                   env=child_env(tmp_path_factory.mktemp("home")), capture_output=True,
                   timeout=300)
    return checkout, regressions.read_results(results)


def _by_id(results: Any) -> dict[str, dict[str, Any]]:
    return {test["nodeid"]: test for test in results.tests}


def test_every_outcome_is_recorded_and_the_run_goes_past_a_file_that_cannot_import(
        night: tuple[pathlib.Path, Any]) -> None:
    """A strict unexpected pass must not look like a failure of the code, an expected
    failure must not look like a failure at all, and one file that fails to import must
    not stop every other test from running, or one bad import would blank the night."""
    _, results = night
    outcomes = {nodeid: test["outcome"] for nodeid, test in _by_id(results).items()}
    assert outcomes == {
        "test_outcomes.py::test_passes": "passed",
        "test_outcomes.py::test_fails": "failed",
        "test_outcomes.py::test_setup_errors": "error",
        "test_outcomes.py::test_skipped": "skipped",
        "test_outcomes.py::test_expected_failure": "xfailed",
        "test_outcomes.py::test_strict_unexpected_pass": "xpass-strict",
        "test_outcomes.py::test_unexpected_pass": "xpassed",
        "test_machine.py::test_random_operations": "failed",
        "test_broken_import.py": "error",
    }
    assert results.exitstatus == 1
    tests = _by_id(results)
    assert "one is not two" in tests["test_outcomes.py::test_fails"]["message"]
    assert tests["test_outcomes.py::test_setup_errors"]["when"] == "setup"
    assert "the fixture broke" in tests["test_outcomes.py::test_setup_errors"]["longrepr"]
    assert tests["test_broken_import.py"]["when"] == "collect"
    assert "a_module_that_does_not_exist" in tests["test_broken_import.py"]["longrepr"]


def test_the_replay_program_and_the_seed_are_read_from_a_real_failure(
        night: tuple[pathlib.Path, Any]) -> None:
    """A state machine that finds two different breaks fails the same test twice. Its
    replay program is what tells the two apart, so it has to survive pytest's way of
    printing the failure."""
    _, results = night
    text = _by_id(results)["test_machine.py::test_random_operations"]["longrepr"]
    assert regressions.program_of(text) == (
        "Remember(user='u1', predicate='lives_in', object='Berlin')",
        "Erase(user='u1')")
    seed = regressions.seed_of(text)
    assert seed is not None
    assert seed.startswith("@reproduce_failure('") and seed.endswith("')")


@pytest.mark.parametrize("text, program", [
    ("no program here", ()),
    ("E       drive.replay([])", ()),
    ("drive.replay([\n    Forget(user='u1'),\n])", ("Forget(user='u1')",)),
    ("E   drive.replay([\nE       EraseExpired(user='u1'),\nE       Erase(user='u2'),\n"
     "E   ])\nE   ", ("EraseExpired(user='u1')", "Erase(user='u2')")),
    ("drive.replay([\n    Forget(user='u1'),\n])\nlater: drive.replay([\n"
     "    Erase(user='u1'),\n])", ("Erase(user='u1')",)),
    ("drive.replay([\n    Forget(user='u1'),\n", ()),
], ids=["none", "empty", "plain", "marked by pytest", "the last one wins", "cut off"])
def test_the_replay_program_is_the_last_complete_one(text: str,
                                                     program: tuple[str, ...]) -> None:
    """A program cut off before its closing line would replay something other than the
    break, so it is not used; an operation named Erase must keep its first letter."""
    assert regressions.program_of(text) == program


def test_there_is_no_seed_when_hypothesis_printed_none() -> None:
    assert regressions.seed_of("AssertionError: one is not two") is None


def _test(nodeid: str, outcome: str, **fields: Any) -> dict[str, Any]:
    return {"nodeid": nodeid, "outcome": outcome, "when": fields.pop("when", "call"),
            "message": fields.pop("message", "boom"), "longrepr": fields.pop("longrepr", "")}


class FakeRerun:
    """Answers each rerun from a script, and records which tests were rerun."""

    def __init__(self, answers: dict[str, tuple[str, ...] | None]) -> None:
        self.answers = answers
        self.asked: list[str] = []

    def __call__(self, nodeid: str) -> tuple[str, ...] | None:
        self.asked.append(nodeid)
        return self.answers[nodeid]


MODEL = "tests/adversarial/model/test_adv_model_machine.py::test_random_operations"


def test_each_failed_test_is_rerun_and_becomes_an_unclassified_finding() -> None:
    """Nobody has checked a failed test against SECURITY.md yet, so its finding is
    unclassified, and the nightly run will not file it in public until someone has."""
    results = regressions.Results([
        _test("tests/test_api.py::test_a", "failed", message="one is not two"),
        _test("tests/test_api.py::test_b", "passed"),
        _test(MODEL, "failed", message="the store and the model disagree",
              longrepr="E   drive.replay([\nE       Forget(user='u1'),\nE   ])\n"
                       "E   @reproduce_failure('6.168.1', b'AEEA')"),
        _test("tests/adversarial/concurrency/test_adv_threads.py::test_c", "error",
              when="setup"),
    ], exitstatus=1)
    rerun = FakeRerun({"tests/test_api.py::test_a": ("passed", "passed"),
                       MODEL: ("failed", "failed"),
                       "tests/adversarial/concurrency/test_adv_threads.py::test_c":
                           ("passed", "failed")})
    found = regressions.failures(results, commit="c" * 40, rerun=rerun)
    assert rerun.asked == ["tests/test_api.py::test_a", MODEL,
                           "tests/adversarial/concurrency/test_adv_threads.py::test_c"]
    assert [(failure.kind, failure.verdict, failure.fileable) for failure in found] == [
        ("test", "flake", False), ("test", "confirmed", True),
        ("test", "intermittent", False)]
    machine = found[1].finding
    assert machine == Finding(
        layer="model", surface="suite", invariant=MODEL, severity="unclassified",
        ops=("Forget(user='u1')",), seed="@reproduce_failure('6.168.1', b'AEEA')",
        artifacts={"log": "regressions/output.log", "results": "regressions/results.jsonl"},
        commit="c" * 40, title=machine.title, detail=machine.detail)
    assert MODEL in machine.title and "the store and the model disagree" in machine.title
    assert "drive.replay" in machine.detail
    assert found[2].finding.layer == "concurrency"


def test_tests_past_the_rerun_limit_or_budget_are_unconfirmed() -> None:
    """Hundreds of failures usually share one cause, and rerunning each twice could use
    up the night, so reruns stop at a limit, and the rest are never counted as
    confirmed: something unconfirmed is never filed."""
    results = regressions.Results(
        [_test(f"tests/test_api.py::test_{index}", "failed") for index in range(3)],
        exitstatus=1)
    rerun = FakeRerun({"tests/test_api.py::test_0": ("failed", "failed"),
                       "tests/test_api.py::test_1": None,
                       "tests/test_api.py::test_2": ("failed", "failed")})
    found = regressions.failures(results, commit="c", rerun=rerun, limit=2)
    assert rerun.asked == ["tests/test_api.py::test_0", "tests/test_api.py::test_1"]
    assert [(failure.verdict, failure.reruns) for failure in found] == [
        ("confirmed", ("failed", "failed")), ("unconfirmed", ()), ("unconfirmed", ())]


def test_a_strict_expected_failure_that_passes_is_not_filed_as_a_break() -> None:
    """The strict marker fails the run when its bug's test starts passing, which usually
    means the fix landed. Filing that as a new break would be wrong, so it is its own kind,
    never rerun and never fileable."""
    results = regressions.Results([
        _test("tests/adversarial/test_adv_known_bugs.py::test_b2", "xpass-strict",
              message="[XPASS(strict)] B2, memvara/memvara#266")], exitstatus=1)
    rerun = FakeRerun({})
    found = regressions.failures(results, commit="c", rerun=rerun)
    assert rerun.asked == []
    assert [(failure.kind, failure.fileable) for failure in found] == [
        ("xpass-strict", False)]
    assert "passes" in found[0].finding.title


def test_a_run_that_fails_with_no_failed_test_is_a_failure_for_a_person() -> None:
    """The skip ledger and the credential guards fail the run from outside any test. A
    run that exits 1 with no failed test must be reported, never read as a pass."""
    found = regressions.failures(regressions.Results([_test("t.py::a", "passed")], 1),
                                 commit="c", rerun=FakeRerun({}),
                                 log_tail="skips with no rule in tests/harness/skips.py")
    assert [(failure.kind, failure.verdict, failure.fileable) for failure in found] == [
        ("session", "unconfirmed", False)]
    assert "skips with no rule" in found[0].finding.detail
    assert regressions.failures(regressions.Results([_test("t.py::a", "passed")], 0),
                                commit="c", rerun=FakeRerun({})) == []


def test_an_interrupted_run_is_reported_even_when_tests_failed() -> None:
    """Exit status 2 means pytest stopped early, so the tests it never ran are unknown."""
    rerun = FakeRerun({"t.py::a": ("failed", "failed")})
    found = regressions.failures(regressions.Results([_test("t.py::a", "failed")], 2),
                                 commit="c", rerun=rerun)
    assert [failure.kind for failure in found] == ["test", "session"]
    assert "interrupted" in found[1].finding.title


def test_the_layers_count_tests_that_ran_and_tests_that_failed() -> None:
    """The flake rate divides by the tests a layer ran, so a skipped test is not one."""
    results = regressions.Results([
        _test("tests/adversarial/model/test_a.py::t1", "passed"),
        _test("tests/adversarial/model/test_a.py::t2", "failed"),
        _test("tests/adversarial/model/test_a.py::t3", "skipped"),
        _test("tests/adversarial/model/test_a.py::t4", "xfailed"),
        _test("tests/test_api.py::t5", "error"),
        _test("tests/test_api.py::t6", "xpass-strict"),
    ], exitstatus=1)
    assert regressions.layers(results) == {"model": {"run": 3, "failed": 1},
                                           "unit": {"run": 2, "failed": 2}}


def test_a_finding_a_step_reported_may_be_filed_as_it_stands() -> None:
    """A step such as the red team confirms its own findings, by replaying each three
    times, before it writes them down; they are not rerun here and may be filed."""
    finding = Finding("redteam", "server", "scope isolation", ops=("call memory_search",))
    assert regressions.Failure(finding, "finding", (), "confirmed").fileable


def test_a_failure_survives_the_report_and_comes_back_whole() -> None:
    """The filing command reads the night's failures back from report.json, and must get
    the same finding, or it would file under the wrong fingerprint."""
    failure = regressions.Failure(
        Finding("model", "suite", MODEL, ops=("Forget(user='u1')",), commit="c"),
        "test", ("failed", "failed"), "confirmed")
    record = failure.to_record()
    assert record["fingerprint"] == failure.finding.signature()
    assert regressions.Failure.from_record(record) == failure
