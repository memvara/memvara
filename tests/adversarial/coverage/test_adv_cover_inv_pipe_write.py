"""The invariants of the write pipeline, from `docs/claude/write-pipeline.md`, that no
older test checked in full.

Each test names the part of its invariant that the older tests left open. The parts they
already check are covered by marks on those tests.
"""

from __future__ import annotations

import pytest

from harness import stores

from ..model_faults.handles import MODEL_TURN, PORTO, agentic, ledger
from ..model_faults.scripted import Answer, Forever, ScriptedModel

# -- WP1 ---------------------------------------------------------------------------------


@pytest.mark.covers("inv:WP1")
def test_the_fast_path_never_reads_a_first_person_sentence_in_a_document() -> None:
    """`docs/claude/write-pipeline.md` says a document chunk passes the role check, and
    that the fast path still reads user turns only, so a chunk's facts come from the
    model tier.

    With no model, a user turn saying "I live in Berlin." is read by the fast path. The
    same sentence in a document must yield no claim, because the fast path reads a
    first-person sentence as the user's own and a document is not the user speaking.
    """
    with stores.memory(user="u1") as mem:
        mem.add_document("I live in Berlin.", title="A letter from a customer")
        assert mem.get_all() == [], "the fast path read a document as the user"

        mem.add("I live in Berlin.")
        assert [(c.predicate, c.object) for c in mem.get_all()] == [("lives_in", "Berlin")]


# -- WP10 --------------------------------------------------------------------------------


@pytest.mark.covers("inv:WP10")
def test_a_memory_the_model_proposed_is_not_stored_when_the_run_is_stopped() -> None:
    """`docs/claude/write-pipeline.md` says a model proposes and the reconciler applies:
    under `agentic_extraction`, nothing a tool does writes to the store, and every change
    a write makes is one of the reconciler's outcomes.

    The model here proposes a new memory in every answer until the run is stopped at its
    step limit, which discards its proposals. The fallback extraction finds nothing. So
    the store must hold no memory at all afterwards; a tool that wrote as it was called
    would have left the proposed one behind.
    """
    proposal = {key: value for key, value in PORTO.items()
                if key not in ("polarity", "when")} | {"valid_from": None}
    model = ScriptedModel(tools=[Forever(Answer(calls=(("propose_claim", proposal),)))],
                          extract=[[]])
    mem = agentic(model)

    receipt = mem.add(MODEL_TURN)

    assert receipt.agentic_fallback == "step_limit"
    assert model.count("run_tools") == 12 and model.unscripted == []
    assert ledger(mem) == {}, "a proposal reached the store without the reconciler"
    mem.close()
