"""10,000 claims in one reply. Nightly, because each write takes a second or two.

A reply this large is what a model does when it starts restating itself or when a turn is
enormous. It must cost one extraction call like any other reply, every claim must cite the
turn it came from, the claims already stored must be left as they were, and a runaway
restatement must collapse into one claim seen many times.
"""

from __future__ import annotations

import json
import time
from typing import Callable

from harness import known_bugs
from memvara.schema import DEFAULT_LEARNED_CAP

from ..handles import NEW_MANY, claim, fates, ledger, seed, stored_claim, with_model
from ..scripted import Forever, ScriptedModel, Text

Make = Callable[..., ScriptedModel]

SNACK_TURN = "The team keeps a snack list for the office kitchen, and it is very long."

MANY = 10_000

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
    assert stored_claim(mem, stored.id).observation_count == MANY
    assert receipt.llm_calls == model.count() == 1


# -- a bug these tests found, pinned until its fix lands -------------------------------------

#: The longest one write of `MANY` claims under `MANY` invented predicates may take. The
#: same write under one predicate takes one to two seconds, so a fixed write meets this
#: easily; today it takes about 440 seconds on a laptop.
BOUND = 60.0


@known_bugs.xfail("B32")
def test_ten_thousand_invented_predicates_are_written_within_a_minute(
        scripted: Make) -> None:
    """#309. Past the learned-predicate cap, each new spelling is folded onto an existing
    predicate as an alias, and each alias rebuilds the registry's whole index and scans
    the growing alias list of the predicate it joined. So one write's cost grows with the
    square of the number of invented spellings: 6.6 s for 1,000, 76 s for 4,000 and about
    440 s for 10,000. The model is still called only 201 times, as INTERNALS says."""
    many = [claim("team", f"zqx_rel_{i}", f"item {i}") for i in range(MANY)]
    model = scripted(extract=[many], resolve=[Forever(NEW_MANY)])
    mem = with_model(model)
    started = time.monotonic()
    receipt = mem.add("Every item on the release checklist has its own owner and region.")
    took = time.monotonic() - started
    calls = 1 + DEFAULT_LEARNED_CAP
    assert len(receipt.added) == MANY
    if took >= BOUND and receipt.llm_calls == model.count() == calls:
        raise known_bugs.Reproduced(
            f"B32: one write of {MANY} invented predicates took {took:.0f} s, with "
            f"{calls} model calls")
    assert took < BOUND
    assert receipt.llm_calls == model.count() == calls
