"""Findings, and the fingerprint that makes a break seen on two nights one issue.

`harness.report.Finding` is the record a layer of the suite writes when it sees a break,
one JSON line each. The nightly run deduplicates findings by `signature()`. The signature
covers what identifies a break and leaves out what changes from one night to the next (the
seed, the commit, where the artifacts were saved) or at triage (the severity), so a break
that comes back is recognised as the same break.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any

import pytest

from harness.report import Finding, append, read, write

#: A replay program in the shape the reference model's state machine prints.
OPS = ("Remember(user='u1', predicate='lives_in', object='Zürich')",
       "Forget(user='u1', predicate='lives_in')")


def _finding(**changes: Any) -> Finding:
    fields: dict[str, Any] = {"layer": "model", "surface": "library",
                              "invariant": "reads match the model", "ops": OPS}
    fields.update(changes)
    return Finding(**fields)


def test_a_break_seen_on_two_nights_has_one_signature() -> None:
    """The seed, the commit, the artifacts' paths, the title and the failure text differ
    between two nights that see the same break, and triage can change the severity. None
    of them may change the fingerprint, or the second night files a duplicate issue."""
    first = _finding(seed="@reproduce_failure('6.100.0', b'AXic')", commit="a" * 40,
                     artifacts={"log": "2026-09-27/regressions/output.log"},
                     title="first night", detail="Traceback, line 10")
    second = _finding(seed=None, commit="b" * 40, severity="data-loss",
                      artifacts={"log": "2026-09-28/regressions/output.log"},
                      title="second night", detail="Traceback, line 12")
    assert first.signature() == second.signature()


@pytest.mark.parametrize("change", [
    {"layer": "concurrency"},
    {"surface": "server"},
    {"invariant": "rows disappear only through erasure"},
    {"ops": ("Remember(user='u1', predicate='lives_in', object='Paris')", OPS[1])},
    {"ops": OPS[:1]},
], ids=["layer", "surface", "invariant", "an op", "one op fewer"])
def test_a_different_break_has_a_different_signature(change: dict[str, Any]) -> None:
    """Two different breaks with one fingerprint would share one issue, and the second
    break would never be filed."""
    assert _finding(**change).signature() != _finding().signature()


def test_the_signature_is_the_hash_of_a_fixed_canonical_form() -> None:
    """The history holds fingerprints written by earlier runs. If the algorithm changed,
    every known break would look new and be filed again. So the canonical form is spelled
    out here by hand: sorted keys, no spaces, and text as UTF-8 rather than escapes."""
    payload = ('{"invariant":"reads match the model","layer":"model","ops":['
               '"Remember(user=\'u1\', predicate=\'lives_in\', object=\'Zürich\')",'
               '"Forget(user=\'u1\', predicate=\'lives_in\')"],"surface":"library","v":1}')
    assert _finding().signature() == hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_a_finding_survives_a_json_round_trip_on_one_line() -> None:
    """findings.jsonl holds one finding per line, so text with line breaks, including a
    Unicode line separator that `str.splitlines()` splits on, must not split a finding
    across lines, and reading a line back must give the same finding."""
    detail = "first line\nsecond line third line"
    finding = _finding(seed="s1", commit="c" * 40, severity="crash", title="t",
                       artifacts={"results": "regressions/results.jsonl",
                                  "log": "regressions/output.log"},
                       detail=detail)
    line = finding.to_json()
    assert len(line.splitlines()) == 1
    assert Finding.from_json(line) == finding
    assert json.loads(line) == {
        "layer": "model", "surface": "library", "invariant": "reads match the model",
        "severity": "crash", "ops": list(OPS), "seed": "s1",
        "artifacts": {"log": "regressions/output.log",
                      "results": "regressions/results.jsonl"},
        "commit": "c" * 40, "title": "t", "detail": detail,
        "fingerprint": finding.signature()}


def test_findings_written_and_appended_are_read_back_in_order(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "findings.jsonl"
    first, second, third = _finding(), _finding(layer="concurrency"), _finding(surface="server")
    write(path, [first, second])
    append(path, third)
    assert read(path) == [first, second, third]
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3


def test_a_blank_line_is_ignored_and_a_bad_line_is_named(tmp_path: pathlib.Path) -> None:
    """A findings file cut short by a killed process must fail loudly, and say where."""
    path = tmp_path / "findings.jsonl"
    path.write_text(_finding().to_json() + "\n\n{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 3"):
        read(path)
    path.write_text("\n" + _finding().to_json() + "\n\n", encoding="utf-8")
    assert read(path) == [_finding()]


@pytest.mark.parametrize("change, named", [
    ({"severity": "critical"}, "severity"),
    ({"invariant": ""}, "invariant"),
    ({"layer": "  "}, "layer"),
    ({"surface": ""}, "surface"),
    ({"ops": (object(),)}, "ops"),
], ids=["unknown severity", "empty invariant", "blank layer", "empty surface",
        "an op that is not JSON"])
def test_a_finding_that_cannot_be_deduplicated_or_routed_is_refused(
        change: dict[str, Any], named: str) -> None:
    """The severity decides where a break is filed, and a security-class break must never
    reach a public issue, so a misspelt severity is refused rather than guessed. An empty
    identity field or an op that is not JSON would give a meaningless fingerprint."""
    with pytest.raises(ValueError, match=named):
        _finding(**change)


def test_an_unknown_key_is_refused() -> None:
    """A producer that misspells a field, for example `sevrity`, would otherwise lose the
    field without a word, and the finding would be filed as unclassified."""
    data = _finding().to_dict()
    data["sevrity"] = "security"
    with pytest.raises(ValueError, match="sevrity"):
        Finding.from_dict(data)


def test_a_stored_fingerprint_that_does_not_match_is_refused() -> None:
    """A finding whose stored fingerprint differs from its signature was written by a run
    that computed signatures differently, and deduplicating it would go quietly wrong."""
    data = _finding().to_dict()
    data["fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint"):
        Finding.from_dict(data)
