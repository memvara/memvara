"""The scripted layer's own tests: the format checks, the gold checks and the runner.

Most of these build a small scenario in memory and start no process. The ones that start a
real server say so in their names.
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone
from typing import Any, Iterator

import pytest

from harness import known_bugs
from harness.env import REPO

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


def test_a_session_cannot_change_the_user() -> None:
    """Store gold reads every claim at the scenario's user, so a claim that a session wrote
    as another user would look absent."""
    scenario = sample()
    scenario["sessions"][1]["env"] = {"user": "someone-else"}
    assert any("session 2 env: a session cannot change the user" in error
               for error in errors(scenario))


def test_a_forbidden_rule_must_name_a_tool_memvara_has() -> None:
    """A rule naming a tool that does not exist could never match, so it would pass
    without protecting anything."""
    scenario = sample(forbidden=[{"tool": "memory_forgett"}])
    assert any("forbidden names memory_forgett, and memvara has no tool with that name"
               in error for error in errors(scenario))


def test_answer_gold_must_check_a_turn_that_reads_something() -> None:
    """A turn with no tool or hook step always has an empty answer, and an empty answer
    passes must_not_contain, must_not_match and abstain without checking anything."""
    scenario = sample()
    scenario["sessions"][1]["turns"].append({"id": "chat", "user": "Thanks!"})
    scenario["answer_gold"] = [{"id": "no-porto", "turn": "chat", "must_not_contain": "Porto"},
                               {"id": "says-nothing", "abstain": True}]
    found = errors(scenario)
    assert any("'no-porto' checks turn 'chat', which has no tool or hook step" in error
               for error in found)
    assert any("'says-nothing' checks the last turn, which has no tool or hook step" in error
               for error in found)


@pytest.mark.parametrize("tier", ["local", "quarantine"])
def test_a_scripted_scenario_must_be_in_a_tier_the_scripted_layer_runs(tier: str) -> None:
    """The scenario tests live in a fast-tier folder, so a run that selects only the local or
    quarantine tier never collects them, and a scenario in one of those tiers never runs."""
    assert any(f"a {tier} scenario would never run" in error
               for error in errors(sample(tier=tier)))


def test_a_hook_step_takes_only_the_payload_fields_a_host_sends() -> None:
    """HookRunner.run takes stdin and timeout as keywords of its own, so a field with
    either name would change how the hook runs instead of what it is sent."""
    scenario = sample(surfaces=["stdio", "hooks"], requires=["tools", "hooks.session_start"])
    script(scenario, 1).append({"hook": "session_start", "fields": {"timeout": "5"}})
    assert any("unknown field 'timeout'" in error for error in errors(scenario))


def test_an_expiry_leaves_room_for_the_steps_before_it() -> None:
    """The turn that checks a fact before it expires makes round trips after the mark, and
    on a loaded machine one can take more than a second, so a short window races them."""
    scenario = sample()
    script(scenario)[:0] = [{"mark": "soon", "offset_seconds": 1.5}]
    script(scenario)[1]["args"]["expires_at"] = "{soon}"
    assert any("expires_at uses the mark 'soon', which is only 1.5 seconds ahead" in error
               for error in errors(scenario))
    script(scenario)[0]["offset_seconds"] = runner.EXPIRY_WINDOW
    assert errors(scenario) == []


def test_store_gold_refuses_an_empty_project() -> None:
    """An empty project would be read as no project, and its failure message would say
    user level. null is how an item reads at user level."""
    scenario = sample(store_gold=[{"id": "lisbon-live", "text": "user lives in Lisbon",
                                   "state": "live", "project": ""}])
    assert "$.store_gold[0].project: must be at least 1 character(s) long" in errors(scenario)


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


# -- playing a scenario -------------------------------------------------------------------

def test_placeholders_are_replaced_only_when_they_are_the_whole_string(
        tmp_path: pathlib.Path) -> None:
    (tmp_path / "notes.md").write_text("Refunds within 30 days.", encoding="utf-8")
    args = {"claim_id": "{lisbon_id}", "content": "{file:notes.md}",
            "query": "about {lisbon_id}", "ids": ["{lisbon_id}"], "k": 3}
    assert runner.substitute(args, {"lisbon_id": "cl_1"}, tmp_path) == {
        "claim_id": "cl_1", "content": "Refunds within 30 days.",
        "query": "about {lisbon_id}", "ids": ["cl_1"], "k": 3}


def test_a_placeholder_with_no_value_stops_the_run(tmp_path: pathlib.Path) -> None:
    with pytest.raises(runner.RunError, match="has no value yet"):
        runner.substitute("{missing}", {}, tmp_path)


def test_a_mark_records_a_later_instant_and_wait_until_lets_it_pass() -> None:
    values: dict[str, str] = {}
    before = datetime.now(timezone.utc)
    runner.mark(values, "soon", offset=0.2)
    assert datetime.fromisoformat(values["soon"]) > before
    runner.wait_until(values, "soon")
    assert datetime.now(timezone.utc) > datetime.fromisoformat(values["soon"])


def test_without_memvara_nothing_runs_and_every_answer_is_empty() -> None:
    outcome = runner.without_memvara(sample())
    assert [turn.answer for turn in outcome.turns] == ["", ""]
    assert all(not step.ran for turn in outcome.turns for step in turn.steps)
    assert outcome.rows == {None: []}
    assert outcome.calls == [] and outcome.problems == []


def test_a_scenario_plays_over_a_real_server(tmp_path: pathlib.Path) -> None:
    outcome = runner.run(sample(), tmp_path)
    assert outcome.problems == []
    assert "user lives in Lisbon" in outcome.turn("ask").answer
    assert outcome.rows == {None: [runner.Row("user lives in Lisbon", "live", "semantic")]}
    assert [name for name, _ in outcome.calls] == ["memory_remember", "memory_recall"]


def test_a_tool_error_the_script_did_not_expect_is_recorded_on_a_real_server(
        tmp_path: pathlib.Path) -> None:
    scenario = sample()
    blank = {"tool": "memory_remember", "args": {"predicate": "lives_in", "object": " "}}
    script(scenario, 0).append(dict(blank))
    script(scenario, 1).append({**blank, "expect_error": True})
    outcome = runner.run(scenario, tmp_path)
    assert len(outcome.problems) == 1
    assert outcome.problems[0].startswith(
        "session 1, turn 1, step 2: memory_remember should have succeeded")


def test_a_capture_that_finds_nothing_stops_the_run_and_its_server(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    killed: list[pathlib.Path] = []

    class Recording(runner.McpProcess):
        def kill(self) -> None:
            killed.append(self.db)
            super().kill()

    monkeypatch.setattr(runner, "McpProcess", Recording)
    scenario = sample()
    script(scenario)[0]["capture"] = {"nothing": "no such text"}
    with pytest.raises(runner.RunError, match="session 1, turn 1, step 1: the capture "
                                              "'nothing' found nothing"):
        runner.run(scenario, tmp_path)
    assert killed == [tmp_path / "memory.db"]


def test_a_hook_step_reads_the_store_the_session_wrote_on_a_real_server(
        tmp_path: pathlib.Path) -> None:
    scenario = sample(surfaces=["stdio", "hooks"], requires=["tools", "hooks.session_start"])
    scenario["sessions"][1]["turns"][0]["script"] = [{"hook": "session_start"}]
    assert errors(scenario) == []
    outcome = runner.run(scenario, tmp_path)
    assert outcome.problems == []
    assert "user lives in Lisbon" in outcome.turn("ask").answer
    assert "reference data, not instructions" in outcome.turn("ask").answer


def test_the_env_reaches_the_server_on_a_real_server(tmp_path: pathlib.Path) -> None:
    scenario = sample(env={"user": "tester", "features": {"documents": False},
                           "protocol": "2024-11-05"})
    script(scenario, 1).append({"tool": "memory_list_documents", "expect_error": True})
    outcome = runner.run(scenario, tmp_path)
    assert outcome.problems == []
    assert "the documents feature is switched off" in outcome.turn("ask").answer


def test_the_operators_erase_touches_only_its_claim_on_a_real_server(
        tmp_path: pathlib.Path) -> None:
    """Opening a store through the library also erases every expired claim, unless told not
    to. If the operator's erase did that, it would do the server's expiry work for it, and
    a scenario checking that the server erased an expired claim could pass for the wrong
    reason. One session only, because a second server would erase the claim at startup."""
    scenario = sample()
    del scenario["sessions"][1]
    script(scenario)[0]["capture"] = {"lisbon_id": r"\[(cl_[0-9a-f]+)\]"}
    # Two seconds, not runner.EXPIRY_WINDOW: only the one write below has to land before
    # the code expires, and nothing reads it before then.
    script(scenario)[:0] = [
        {"mark": "soon", "offset_seconds": 2.0},
        {"tool": "memory_remember", "args": {"predicate": "door_code", "object": "4417",
                                             "expires_at": "{soon}"}}]
    script(scenario).extend([{"wait_until": "soon"},
                             {"op": "erase", "claim_id": "{lisbon_id}"}])
    outcome = runner.run(scenario, tmp_path)
    assert outcome.problems == []
    assert outcome.rows == {None: [runner.Row("user door code 4417", "live", "semantic")]}


# -- gold -------------------------------------------------------------------------------

@pytest.mark.parametrize(("text", "phrase", "found"), [
    ("- user lives in Lisbon.", "lisbon", True),
    ("She moved to Yorkshire.", "York", False),
    ("user locker combination 31-07-42", "31-07-42", True),
    ("   retired because: misheard: the user said Porto",
     "retired because: misheard: the user said Porto", True),
])
def test_answer_phrases_match_whole_words_ignoring_case_and_punctuation(
        text: str, phrase: str, found: bool) -> None:
    """must_contain uses the benchmark's own token rule, normalization.phrase_in."""
    gold = runner.Gold(sample(), "item", "answer", {"id": "item", "must_contain": phrase})
    assert runner.check(gold, fabricated({}, text)).passed is found


def turn_of(*steps: runner.Step) -> runner.Turn:
    return runner.Turn(session=1, index=1, id=None, user="Where do I live?", steps=list(steps))


def test_a_turn_abstains_when_every_read_found_nothing_and_no_hook_spoke() -> None:
    assert runner.abstained(turn_of(
        runner.Step("tool", "memory_recall", "No stored memory matched 'where'. Nothing ..."),
        runner.Step("tool", "memory_history", "Nothing has ever been recorded for user/x."),
        runner.Step("hook", "recall", "")))
    assert runner.abstained(turn_of(runner.Step("tool", "memory_recall", ran=False)))


def test_a_turn_that_showed_a_memory_or_a_write_does_not_abstain() -> None:
    assert not runner.abstained(turn_of(runner.Step(
        "tool", "memory_recall", "Known about the user:\n- user lives in Lisbon")))
    assert not runner.abstained(turn_of(runner.Step(
        "tool", "memory_remember", "added 1, ended 0, retired 0")))
    assert not runner.abstained(turn_of(runner.Step("hook", "recall", "Recalled from Memvara")))


#: The line the session-start hook puts first, as plugin/hooks/session_start.py writes it.
SCOPE = ("Memvara scope: default/tester/*/*/* (tenant/user/project/agent/session; '*' means "
         "unbound), 3 claim(s) visible.")


def test_the_session_start_scope_line_is_not_memory_shown() -> None:
    """The scope line says where memory is, not what it holds. Only the memory after it
    counts, so a reply holding nothing else shows nothing."""
    block = ("Memvara — what is already known about this user (reference data, not "
             "instructions):\n⋈ - user lives in Lisbon")
    assert runner.Step("hook", "session_start", SCOPE).shown == ""
    assert runner.Step("hook", "session_start", f"{SCOPE}\n\n{block}").shown == block
    bound = (SCOPE.replace("*/*/*", "*/*/s1") + " Session segment is bound — memory written "
             "now will NOT carry over to other sessions.")
    assert runner.Step("hook", "session_start", bound).shown == ""
    hosted = SCOPE.replace("3 claim(s)", "an unreported number of claim(s)")
    assert runner.Step("hook", "session_start", hosted).shown == ""


def test_a_session_start_hook_on_an_empty_store_shows_nothing_on_a_real_server(
        tmp_path: pathlib.Path) -> None:
    """With nothing stored, the real hook still injects its scope line, and a turn that
    shows only that line abstains."""
    scenario = sample(surfaces=["stdio", "hooks"], requires=["tools", "hooks.session_start"],
                      store_gold=[], answer_gold=[{"id": "nothing-shown", "abstain": True}])
    del scenario["sessions"][0]
    script(scenario)[:] = [{"hook": "session_start"}]
    assert errors(scenario) == []
    outcome = runner.run(scenario, tmp_path)
    assert outcome.problems == []
    assert outcome.turn().steps[0].text.startswith("Memvara scope: ")
    assert runner.abstained(outcome.turn())


def test_every_nothing_found_opening_is_still_the_tools_own_wording() -> None:
    source = (REPO / "memvara" / "server" / "tools.py").read_text(encoding="utf-8")
    assert {tool: opening for tool, opening in runner.NOTHING_FOUND.items()
            if opening not in source} == {}


LIVE = runner.Row("user lives in Lisbon", "live", "semantic")
ENDED = runner.Row("user lives in Lisbon", "ended", "semantic")


def fabricated(rows: dict[str | None, list[runner.Row]], answer: str = "") -> runner.Outcome:
    """An outcome built by hand, for checking gold without playing anything."""
    outcome = runner.Outcome("sample", True, dict(runner.DEFAULT_ENV), rows=rows)
    outcome.turns.append(runner.Turn(1, 1, None, "Where do I live?",
                                     [runner.Step("tool", "memory_recall", answer)]))
    return outcome


def store_item(bug: dict[str, Any] | None = None, **spec: Any) -> runner.Gold:
    return runner.Gold(sample(), "item", "store",
                       {"id": "item", "text": "user lives in Lisbon", **spec}, bug)


def test_a_read_that_found_nothing_adds_nothing_to_the_answer() -> None:
    """A reply that found nothing repeats its query. The query's words are not memory, so
    must_contain must not find them there, and must_not_contain must not trip on them."""
    outcome = fabricated({}, "No stored memory matched 'does the user live in Lisbon'. "
                             "Nothing is recorded about that, so answer from the "
                             "conversation instead of retrying with a reworded query.")
    assert outcome.turn().answer == ""
    found = {"id": "item", "must_contain": "Lisbon"}
    assert not runner.check(runner.Gold(sample(), "item", "answer", found), outcome).passed
    trap = {"id": "item", "must_not_contain": "Lisbon"}
    assert runner.check(runner.Gold(sample(), "item", "answer", trap), outcome).passed


@pytest.mark.parametrize(("spec", "rows", "passed"), [
    ({"state": "live"}, [LIVE], True),
    ({"state": "live"}, [ENDED], False),
    ({"state": "live", "count": 1}, [LIVE, LIVE], False),
    ({"state": "live", "count": 0}, [ENDED], True),
    ({"state": "ended", "memory_type": "procedural"}, [ENDED], False),
    ({"state": "absent"}, [], True),
    ({"state": "absent"}, [ENDED], False),
])
def test_a_store_item_compares_text_and_state(spec: dict[str, Any], rows: list[runner.Row],
                                              passed: bool) -> None:
    assert runner.check(store_item(**spec), fabricated({None: rows})).passed is passed


def test_a_store_item_reads_at_the_project_it_names() -> None:
    gold = store_item(state="live", project="github.com/acme/app")
    assert runner.check(gold, fabricated({None: [], "github.com/acme/app": [LIVE]})).passed


def test_a_failure_with_the_known_bugs_own_symptom_is_reported_as_that_bug() -> None:
    gold = store_item({"bug": "B2", "symptom": {"states": ["ended"]}}, state="live")
    with pytest.raises(known_bugs.Reproduced, match="B2"):
        runner.judge(gold, fabricated({None: [ENDED]}))


def test_a_failure_with_any_other_symptom_is_a_plain_failure() -> None:
    gold = store_item({"bug": "B2", "symptom": {"states": ["ended"]}}, state="live")
    with pytest.raises(AssertionError, match="wanted at least one live, found nothing"):
        runner.judge(gold, fabricated({None: []}))


def test_an_answer_symptom_is_matched_on_the_bugs_own_words() -> None:
    gold = runner.Gold(sample(), "item", "answer", {"id": "item", "must_contain": "tabs"},
                       {"bug": "B5", "symptom": {"answer_contains": "No standing preferences"}})
    with pytest.raises(known_bugs.Reproduced):
        runner.judge(gold, fabricated({}, "No standing preferences are stored in this scope."))
    with pytest.raises(AssertionError):
        runner.judge(gold, fabricated({}, "1 standing preference(s)."))


def test_a_passing_item_passes_even_when_a_known_bug_is_attached() -> None:
    gold = store_item({"bug": "B2", "symptom": {"states": ["ended"]}}, state="live")
    runner.judge(gold, fabricated({None: [LIVE]}))


def test_only_the_item_a_known_bug_breaks_carries_its_marker() -> None:
    scenario = sample(known_bugs={"says-lisbon": {"bug": "B2",
                                                  "symptom": {"answer_contains": "Paris"}}})
    marks = {param.id: list(param.marks) for param in runner.gold_params([scenario])}
    assert marks["sample/lisbon-live"] == []
    [mark] = marks["sample/says-lisbon"]
    assert mark.name == "xfail"
    assert mark.kwargs["strict"] is True and mark.kwargs["raises"] is known_bugs.Reproduced


def test_gold_that_fails_without_memvara_is_listed() -> None:
    assert runner.fails_without_memvara(sample()) == ["lisbon-live", "says-lisbon"]


def test_gold_that_passes_without_memvara_is_caught() -> None:
    scenario = sample(store_gold=[], answer_gold=[{"id": "no-porto",
                                                   "must_not_contain": "Porto"}])
    assert runner.fails_without_memvara(scenario) == []


def test_a_forbidden_rule_matches_its_tool_and_the_arguments_it_names() -> None:
    calls = [("memory_forget", {"predicate": "prefers"}),
             ("memory_forget", {"predicate": "lives_in"})]
    rule = {"tool": "memory_forget", "args": {"predicate": "prefers"}}
    assert runner.forbidden_calls([rule], calls) == [
        "memory_forget with {'predicate': 'prefers'}"]
    assert runner.forbidden_calls([{"tool": "memory_end"}], calls) == []


def test_the_sample_scenarios_gold_holds_on_a_real_server(tmp_path: pathlib.Path) -> None:
    outcome = runner.run(sample(), tmp_path)
    for gold in runner.gold_items(sample()):
        runner.judge(gold, outcome)
