"""The scripted layer's own tests: the format checks, the gold checks and the runner.

Most of these build a small scenario in memory and start no process. The ones that start a
real server say so in their names.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Iterator

import pytest

from . import runner


def sample(**changes: Any) -> dict[str, Any]:
    """A small scenario that follows the format: the user says where they live, and a
    later session asks."""
    scenario: dict[str, Any] = {
        "id": "sample",
        "description": "The user says where they live, and a later session asks.",
        "tier": "fast",
        "surfaces": ["stdio"],
        "env": {"user": "tester"},
        "sessions": [
            {"turns": [{"user": "I live in Lisbon.", "script": [
                {"tool": "memory_remember",
                 "args": {"predicate": "lives_in", "object": "Lisbon"}}]}]},
            {"turns": [{"id": "ask", "user": "Where do I live?", "script": [
                {"tool": "memory_recall", "args": {"query": "where does the user live"}}]}]},
        ],
        "store_gold": [{"id": "lisbon-live", "text": "user lives in Lisbon",
                        "state": "live", "count": 1}],
        "answer_gold": [{"id": "says-lisbon", "must_contain": "Lisbon"}],
        "requires": ["tools"],
        "negative_control": True,
    }
    scenario.update(changes)
    return scenario


def errors(scenario: dict[str, Any]) -> list[str]:
    """The schema's errors, then the checks the schema cannot express, as `load` runs them."""
    return runner.schema_errors(scenario, runner.schema()) or runner.problems(scenario)


def script(scenario: dict[str, Any], session: int = 0) -> list[dict[str, Any]]:
    """The first turn's script in one session, to add steps to."""
    steps: list[dict[str, Any]] = scenario["sessions"][session]["turns"][0]["script"]
    return steps


# -- the format -------------------------------------------------------------------------

def test_the_sample_scenario_follows_the_format() -> None:
    assert errors(sample()) == []


def test_a_missing_field_is_named() -> None:
    scenario = sample()
    del scenario["sessions"]
    assert "$: missing required field 'sessions'" in errors(scenario)


def test_an_unknown_field_is_refused_rather_than_ignored() -> None:
    assert "$: unknown field 'store_gould'" in errors(sample(store_gould=[]))


def test_a_misspelled_step_is_reported_against_the_shape_it_resembles() -> None:
    scenario = sample()
    script(scenario)[0]["expect_eror"] = True
    found = errors(scenario)
    assert any("matches none of the 5 allowed shapes" in error for error in found)
    assert any("unknown field 'expect_eror'" in error for error in found)


def test_a_schema_keyword_the_validator_lacks_is_refused() -> None:
    with pytest.raises(ValueError, match="does not implement"):
        runner.schema_errors(3, {"type": "integer", "maximum": 2})


def schema_nodes(node: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Every place in the schema that holds a schema, whether or not an instance reaches it."""
    yield node
    for key in ("properties", "$defs"):
        for child in node.get(key, {}).values():
            yield from schema_nodes(child)
    for key in ("items", "additionalProperties"):
        if isinstance(node.get(key), dict):
            yield from schema_nodes(node[key])
    for child in node.get("oneOf", []):
        yield from schema_nodes(child)


def test_the_schema_uses_only_keywords_the_validator_implements() -> None:
    """Validating a scenario checks only the definitions it reaches. This checks the rest."""
    used = {key for node in schema_nodes(runner.schema()) for key in node}
    assert used - runner.KEYWORDS == set()


def test_a_gold_id_used_twice_is_refused() -> None:
    scenario = sample(answer_gold=[{"id": "lisbon-live", "must_contain": "Lisbon"}])
    assert "gold id 'lisbon-live' is used more than once" in errors(scenario)


def test_a_scenario_without_gold_is_refused() -> None:
    scenario = sample(store_gold=[], answer_gold=[])
    assert "the scenario has no gold, so it checks nothing" in errors(scenario)


def test_a_known_bug_needs_a_gold_item_a_registered_bug_and_the_right_symptom() -> None:
    scenario = sample(known_bugs={
        "no-such-item": {"bug": "B2", "symptom": {"states": ["ended"]}},
        "says-lisbon": {"bug": "B99", "symptom": {"states": ["ended"]}}})
    found = errors(scenario)
    assert "known_bugs names 'no-such-item', which is not a gold id" in found
    assert any("names B99, which tests/harness/known_bugs.py does not register" in error
               for error in found)
    assert any("an answer gold item takes answer_contains" in error for error in found)


def test_an_answer_gold_item_must_name_a_turn_that_exists() -> None:
    scenario = sample(answer_gold=[{"id": "says-lisbon", "turn": "nope",
                                    "must_contain": "Lisbon"}])
    assert any("names turn 'nope'" in error for error in errors(scenario))


def test_a_gold_phrase_with_no_words_is_refused() -> None:
    scenario = sample(answer_gold=[{"id": "dash", "must_not_contain": "—"}])
    assert any("has no words left" in error for error in errors(scenario))


def test_a_pattern_that_is_not_a_regular_expression_is_refused() -> None:
    scenario = sample(answer_gold=[{"id": "broken", "must_not_match": "(unclosed"}])
    assert any("is not a regular expression" in error for error in errors(scenario))


def test_a_placeholder_must_be_set_by_an_earlier_step() -> None:
    scenario = sample()
    script(scenario, 1).append({"tool": "memory_why", "args": {"claim_id": "{lisbon_id}"}})
    assert any("{lisbon_id} is used before any step sets it" in error
               for error in errors(scenario))
    script(scenario, 0)[0]["capture"] = {"lisbon_id": r"\[(cl_[0-9a-f]+)\]"}
    assert errors(scenario) == []


def test_a_file_placeholder_must_name_a_workspace_file() -> None:
    scenario = sample()
    script(scenario).append({"tool": "memory_add_document",
                             "args": {"content": "{file:policies/refunds.md}"}})
    assert any("{file:policies/refunds.md} names no workspace file" in error
               for error in errors(scenario))
    scenario["workspace"] = {"files": {"policies/refunds.md": "Refunds within 30 days."}}
    assert errors(scenario) == []


@pytest.mark.parametrize("name", ["../outside.txt", "/etc/passwd", "C:/notes.txt",
                                  "docs\\notes.txt"])
def test_a_workspace_file_must_stay_inside_the_workspace(name: str) -> None:
    scenario = sample(workspace={"files": {name: "x"}})
    assert any("must be a relative path inside the workspace" in error
               for error in errors(scenario))


def test_features_and_protocol_are_checked_against_the_server() -> None:
    scenario = sample(env={"user": "tester", "features": {"no_such": True},
                           "protocol": "2023-01-01"})
    found = errors(scenario)
    assert "env: memvara has no feature named 'no_such'" in found
    assert any("does not speak protocol '2023-01-01'" in error for error in found)


def test_a_tool_must_exist_and_the_capabilities_must_be_declared() -> None:
    scenario = sample(requires=["tools", "hooks.capture"])
    script(scenario).extend([{"tool": "memory_erase"}, {"hook": "session_start"}])
    found = errors(scenario)
    assert any("memvara has no tool named 'memory_erase'" in error for error in found)
    assert "a step runs a hook, so surfaces must list hooks" in found
    assert "the script uses hooks.session_start, and requires does not list it" in found
    assert any("requires lists hooks.capture, which the scripted layer cannot provide"
               in error for error in found)


def test_an_absent_claim_takes_no_count() -> None:
    scenario = sample(store_gold=[{"id": "gone", "text": "user lives in Lisbon",
                                   "state": "absent", "count": 0}])
    assert any("a claim that is absent has no count" in error for error in errors(scenario))


def test_a_scenario_file_is_named_after_its_id(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "other-name.json"
    path.write_text(json.dumps(sample()), encoding="utf-8")
    with pytest.raises(runner.ScenarioError, match="must be named after the scenario's id"):
        runner.load(path)


def test_a_file_that_is_not_json_is_reported_by_name(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(runner.ScenarioError, match="broken.json is not JSON"):
        runner.load(path)


def test_a_file_with_a_byte_order_mark_and_non_ascii_text_loads(tmp_path: pathlib.Path) -> None:
    """Editors on Windows can save UTF-8 with a byte-order mark, and gold text here uses
    characters such as the em dash."""
    path = tmp_path / "sample.json"
    text = json.dumps(sample(description="Lisbon — then a question."), ensure_ascii=False)
    path.write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))
    assert runner.load(path)["description"] == "Lisbon — then a question."


def test_a_run_includes_a_scenario_only_when_its_tier_is_selected() -> None:
    fast, nightly = sample(id="fast-one"), sample(id="nightly-one", tier="nightly")
    assert [s["id"] for s in runner.selected([fast, nightly], "fast")] == ["fast-one"]
    assert [s["id"] for s in runner.selected([fast, nightly], "nightly")] == [
        "fast-one", "nightly-one"]
    assert runner.selected([fast, nightly], "local") == []
