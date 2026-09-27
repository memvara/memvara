"""`bench/mutation.py`: the parts that do not need mutmut installed.

mutmut is not a dependency of this package, so these tests cover what the script decides on
its own: how a mutant's exit code becomes a status, how statuses become a score, how the
list of equivalent mutants is read and checked, which tests a run selects, and the
configuration it writes into the clone. A real run is measured by hand and recorded in
`docs/claude/testing.md`.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from bench import mutation

REPO = pathlib.Path(__file__).resolve().parent.parent


def _write(tmp_path: pathlib.Path, text: str) -> pathlib.Path:
    path = tmp_path / "equivalents.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_the_shipped_equivalents_file_reads_and_stays_under_the_limit():
    found = mutation.load_equivalents()
    assert len(found) <= mutation.MAX_EQUIVALENTS
    assert all(entry.reason for entry in found.values())


def test_an_equivalent_without_a_reason_is_refused(tmp_path):
    path = _write(tmp_path, '[[equivalent]]\nmutant = "m.f__mutmut_1"\nreason = " "\n')
    with pytest.raises(ValueError, match="needs a mutant and a reason"):
        mutation.load_equivalents(path)


def test_an_equivalent_listed_twice_is_refused(tmp_path):
    row = '[[equivalent]]\nmutant = "m.f__mutmut_1"\nreason = "r"\n'
    with pytest.raises(ValueError, match="listed twice"):
        mutation.load_equivalents(_write(tmp_path, row * 2))


def test_more_equivalents_than_the_design_allows_are_refused(tmp_path):
    rows = "".join(f'[[equivalent]]\nmutant = "m.f__mutmut_{n}"\nreason = "r"\n'
                   for n in range(mutation.MAX_EQUIVALENTS + 1))
    with pytest.raises(ValueError, match="Replace the tool"):
        mutation.load_equivalents(_write(tmp_path, rows))


def test_a_missing_equivalents_file_lists_none(tmp_path):
    assert mutation.load_equivalents(tmp_path / "absent.toml") == {}


@pytest.mark.parametrize("code, status", [
    (1, "killed"), (3, "killed"), (0, "survived"), (5, "no tests"), (33, "no tests"),
    (34, "skipped"), (36, "timeout"), (-24, "timeout"), (37, "caught by type check"),
    (-9, "segfault"), (None, "not checked"), (42, "suspicious"),
])
def test_exit_codes_read_as_mutmut_reads_them(code, status):
    assert mutation.status_of(code) == status


def test_the_score_counts_caught_against_every_counted_mutant():
    statuses = {
        "m.f__mutmut_1": "killed", "m.f__mutmut_2": "timeout",
        "m.f__mutmut_3": "survived", "m.f__mutmut_4": "no tests",
        "m.f__mutmut_5": "skipped", "m.f__mutmut_6": "suspicious",
        "n.g__mutmut_1": "killed",
    }
    [m, n] = mutation.score(statuses, {})
    # Caught: killed and timeout. Not caught: survived, no tests, suspicious. Skipped is
    # not counted either way.
    assert (m.module, m.caught, m.counted) == ("m", 2, 5)
    assert m.score == pytest.approx(40.0)
    assert m.undetected == ["m.f__mutmut_3", "m.f__mutmut_4", "m.f__mutmut_6"]
    assert (n.module, n.score) == ("n", 100.0)


def test_an_equivalent_mutant_is_left_out_of_the_score():
    statuses = {"m.f__mutmut_1": "killed", "m.f__mutmut_2": "survived"}
    equivalent = {"m.f__mutmut_2": mutation.Equivalent("m.f__mutmut_2", "reason")}
    [m] = mutation.score(statuses, equivalent)
    assert (m.score, m.equivalents, m.undetected) == (100.0, 1, [])


def test_a_module_with_nothing_counted_has_no_score():
    [m] = mutation.score({"m.f__mutmut_1": "not checked"}, {})
    assert m.score is None


def test_an_equivalent_the_run_did_not_produce_is_stale():
    equivalents = {name: mutation.Equivalent(name, "r") for name in (
        "memvara.write.reconcile.x_f__mutmut_1", "memvara.write.reconcile.x_f__mutmut_9",
        "memvara.core.x_g__mutmut_1")}
    statuses = {"memvara.write.reconcile.x_f__mutmut_1": "survived"}
    # The entry for core is not stale: core was not measured.
    assert mutation.stale(statuses, equivalents, ["memvara/write/reconcile.py"]) == [
        "memvara.write.reconcile.x_f__mutmut_9"]


def test_the_default_selection_finds_a_test_that_imports_through_the_package():
    """tests/test_reconcile.py imports `Reconciler` from `memvara.write`, not the module
    itself, and it is the test the module most needs."""
    tests = mutation.default_tests(["memvara/write/reconcile.py"])
    assert "tests/test_reconcile.py" in tests
    assert all((REPO / t).is_file() for t in tests)


def test_the_statuses_are_read_from_mutmuts_meta_file(tmp_path):
    meta = tmp_path / "mutants" / "memvara" / "write" / "reconcile.py.meta"
    meta.parent.mkdir(parents=True)
    meta.write_text(json.dumps({"exit_code_by_key": {
        "memvara.write.reconcile.x_f__mutmut_1": 1,
        "memvara.write.reconcile.x_f__mutmut_2": 0}}), encoding="utf-8")
    assert mutation.read_statuses(tmp_path, ["memvara/write/reconcile.py"]) == {
        "memvara.write.reconcile.x_f__mutmut_1": "killed",
        "memvara.write.reconcile.x_f__mutmut_2": "survived"}


def test_a_module_mutmut_wrote_nothing_for_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="no results"):
        mutation.read_statuses(tmp_path, ["memvara/write/reconcile.py"])


def test_a_module_outside_the_library_is_refused():
    with pytest.raises(SystemExit, match="not a module under memvara/"):
        mutation.main(["run", "bench/mutation.py"])


def test_the_floor_fails_a_module_under_it(capsys):
    report = {"scores": [{"module": "m", "score": 79.9}, {"module": "n", "score": None}]}
    assert mutation._verdict(report, 80.0) == 1
    assert "m scored 79.9%" in capsys.readouterr().err
    assert mutation._verdict(report, None) == 0
    assert mutation._verdict(report, 79.0) == 0
