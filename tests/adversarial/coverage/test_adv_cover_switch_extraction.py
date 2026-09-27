"""The three feature switches that change how the MCP server's store asks its model to
extract facts: `agentic_extraction`, `extraction_chunks` and `extraction_guidance`.

Each test builds the server's store from environment variables with the `served` fixture
(conftest.py beside this file), with a scripted model that records every call, and writes
one turn. The switches are described in the comment above `FEATURE_DEFAULTS` in
memvara/server/config.py and in the `MEMVARA_FEATURE_<NAME>` row of docs/DEPLOY.md.
"""

from __future__ import annotations

import pathlib

import pytest

from memvara.server import config as server_config
from memvara.server.config import ServerConfig

from harness.skips import needs_toml

from ..model_faults.handles import LONG, MODEL_TURN, NEW_MANY, PORTO
from ..model_faults.scripted import Answer, Forever, ScriptedModel, Text
from .conftest import Serve

#: What the replacement judge answers when it is asked: the new fact replaces nothing.
KEEPS = Text('{"same_thing": false, "same_property": false, "newer_value": false, '
             '"replaces": false}')

#: The guidance file's one rule, which the extraction call must carry when the switch is on.
RULE = "decisions about retries"


def answering() -> ScriptedModel:
    """A model that answers every extraction, predicate and judge call properly."""
    return ScriptedModel(extract=[Forever([PORTO])], resolve=[Forever(NEW_MANY)],
                         judge=[Forever(KEEPS)])


@pytest.mark.covers("switch:agentic_extraction")
def test_agentic_extraction_switched_on_runs_the_tool_loop_instead_of_one_extraction_call(
        served: Serve) -> None:
    """`MEMVARA_FEATURE_AGENTIC_EXTRACTION=1` lets the extraction model search the store
    and propose memories through tools, instead of making one extraction call. It is off
    by default, and then a write makes exactly one extraction call and no tool run.
    memvara/server/config.py (the comment above `FEATURE_DEFAULTS`) and docs/DEPLOY.md
    make this promise."""
    off = answering()
    receipt = served({}, off).add(MODEL_TURN)
    assert off.count("extract") == 1 and off.count("run_tools") == 0
    assert off.runs == [], "no tool loop ran with the switch at its default"
    assert [(c.subject, c.object) for c in receipt.added] == [("team", "Porto")]

    on = answering()
    proposal = {key: value for key, value in PORTO.items()
                if key not in ("polarity", "when")} | {"valid_from": None}
    on.queue("run_tools", Answer(calls=(("propose_claim", proposal),)), Answer("done"))
    receipt = served({"MEMVARA_FEATURE_AGENTIC_EXTRACTION": "1"}, on).add(MODEL_TURN)
    assert on.count("extract") == 0, "the tool loop replaced the single extraction call"
    assert on.count("run_tools") == 2 and len(on.runs) == 1
    assert "propose_claim" in on.runs[0]["tools"]
    assert receipt.agentic_fallback is None
    assert [(c.subject, c.object) for c in receipt.added] == [("team", "Porto")], (
        "the proposal was stored")


@pytest.mark.covers("switch:extraction_chunks")
def test_extraction_chunks_switched_on_extracts_a_long_turn_one_piece_at_a_time(
        served: Serve) -> None:
    """`MEMVARA_FEATURE_EXTRACTION_CHUNKS=1` extracts a turn over 6,000 characters in
    pieces, one model call per piece. It is off by default, and then the whole turn is
    one call. memvara/server/config.py and docs/DEPLOY.md make this promise. The turn
    here is about 13,000 characters, which the splitter cuts into three pieces."""
    assert len(LONG) > 6000
    off = answering()
    receipt = served({}, off).add(LONG)
    assert off.count("extract") == 1
    assert len(off.calls[0].args["turns"]) == 1
    assert off.calls[0].args["turns"][0] == LONG, "the whole turn went in one call"
    assert receipt.llm_calls == off.count()

    on = answering()
    receipt = served({"MEMVARA_FEATURE_EXTRACTION_CHUNKS": "1"}, on).add(LONG)
    pieces = [call.args["turns"][0] for call in on.calls if call.method == "extract"]
    assert len(pieces) == 3, "one extraction call for each piece"
    assert all(len(piece) < len(LONG) for piece in pieces)
    assert receipt.llm_calls == on.count()


@needs_toml
@pytest.mark.covers("switch:extraction_guidance")
def test_extraction_guidance_reaches_every_extraction_call_only_while_switched_on(
        served: Serve, tmp_path: pathlib.Path) -> None:
    """With `MEMVARA_EXTRACT_GUIDANCE` naming a guidance file, every extraction call
    carries that guidance. `MEMVARA_FEATURE_EXTRACTION_GUIDANCE=0` stops it: the file is
    still read and checked at startup, and no extraction sees it. memvara/server/config.py
    (the comment above `FEATURE_DEFAULTS`) and docs/DEPLOY.md make this promise."""
    guidance = tmp_path / "guidance.toml"
    guidance.write_text(f'include = ["{RULE}"]\n', encoding="utf-8")
    env = {"MEMVARA_EXTRACT_GUIDANCE": str(guidance)}

    on = answering()
    served(env, on).add(MODEL_TURN)
    (call,) = [call for call in on.calls if call.method == "extract"]
    assert call.args["guidance"] is not None
    assert list(call.args["guidance"].include) == [RULE]

    off = answering()
    served({**env, "MEMVARA_FEATURE_EXTRACTION_GUIDANCE": "0"}, off).add(MODEL_TURN)
    (call,) = [call for call in off.calls if call.method == "extract"]
    assert call.args["guidance"] is None, "the switch is off, so no extraction sees it"

    guidance.write_text("include = 3\n", encoding="utf-8")
    with pytest.raises(server_config.ConfigError, match="MEMVARA_EXTRACT_GUIDANCE"):
        ServerConfig.from_env({"MEMVARA_DB": ":memory:", "MEMVARA_LLM": "openai",
                               **env, "MEMVARA_FEATURE_EXTRACTION_GUIDANCE": "0"})
