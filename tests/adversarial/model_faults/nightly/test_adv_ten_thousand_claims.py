"""10,000 claims in one reply. Nightly, because each write takes a second or two.

A reply this large is what a model does when it starts restating itself or when a turn is
enormous. It must cost one extraction call like any other reply, every claim must cite the
turn it came from, the claims already stored must be left as they were, and a runaway
restatement must collapse into one claim seen many times.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from ..handles import fates, ledger, seed, with_model
from ..scripted import ScriptedModel, Text

Make = Callable[..., ScriptedModel]

SNACK_TURN = "The team keeps a snack list for the office kitchen, and it is very long."

MANY = 10_000


def claim(subject: str, predicate: str, obj: str, **changes: Any) -> dict[str, Any]:
    """A model claim citing turn 0, with every field of the claim schema."""
    return {"subject": subject, "predicate": predicate, "object": obj, "polarity": 1,
            "memory_type": "semantic", "confidence": 0.9, "source_index": 0,
            "when": None, "amount": None, "unit": None, **changes}


SNACKS = [claim("team", "likes", f"snack {i}") for i in range(MANY)]


def check_every_claim_is_stored(model: ScriptedModel) -> None:
    mem = with_model(model)
    seed(mem)
    before = ledger(mem)
    receipt = mem.add(SNACK_TURN)
    [turn] = receipt.episode_ids
    assert len(receipt.added) == MANY
    assert {tuple(c.sources) for c in receipt.added} == {(turn,)}
    assert receipt.llm_calls == model.count() == 1
    assert set(fates(before, mem).values()) == {"unchanged"}
    found = mem.search("snack 4242", k=3, query_rewrite=False)
    assert "snack 4242" in [r.claim.object for r in found]


def test_ten_thousand_claims_in_one_reply_are_stored_from_one_call(scripted: Make) -> None:
    check_every_claim_is_stored(scripted(extract=[SNACKS]))


def test_ten_thousand_claims_as_provider_text_are_stored_the_same_way(
        scripted: Make) -> None:
    """The same reply as a shipped backend receives it, about 2 MB of JSON, validated by
    `memvara.llm._shape` before the write path sees it."""
    check_every_claim_is_stored(scripted(extract=[Text(json.dumps({"claims": SNACKS}))]))


def test_a_model_that_restates_one_fact_ten_thousand_times_stores_it_once(
        scripted: Make) -> None:
    """Review Focus 1 of the plan: one fact, seen 10,000 times, is one row."""
    model = scripted(extract=[[claim("team", "likes", "snack list")] * MANY])
    mem = with_model(model)
    receipt = mem.add(SNACK_TURN)
    assert (len(receipt.added), len(receipt.reinforced)) == (1, MANY - 1)
    [stored] = [c for c in mem.get_all() if c.object == "snack list"]
    assert mem.store.get_claim(stored.id).observation_count == MANY
    assert receipt.llm_calls == model.count() == 1
