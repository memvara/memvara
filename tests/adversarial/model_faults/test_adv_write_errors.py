"""The write path when the provider fails: timeouts, rate limits, a rejected key.

`WritePipeline._extraction_failed` states the rule: "A provider 429 is not a reason to lose
a transcript." The turns are already committed when the model is asked, the fast path's
facts from the same batch are still correct, and the receipt says the batch was deferred
so that `reextract()` can read it later. Every call that was made is billed, including the
one that failed. Each error here is raised the way the provider SDKs raise it; see
`scripted.py`.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

from memvara.schema import Cardinality
from memvara.store import SQLiteStore
from memvara.write.split import split_for_extraction

from .handles import (
    FAST_TURN, MODEL_TURN, fates, ledger, seed, turns, with_model, without_model,
)
from .scripted import (
    APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError,
    RateLimitError, ScriptedModel, Truncated,
)

Make = Callable[..., ScriptedModel]


def claim(subject: str, predicate: str, obj: str, **changes: Any) -> dict[str, Any]:
    """A model claim citing turn 0, with every field of the claim schema."""
    return {"subject": subject, "predicate": predicate, "object": obj, "polarity": 1,
            "memory_type": "semantic", "confidence": 0.9, "source_index": 0,
            "when": None, "amount": None, "unit": None, **changes}


PORTO = claim("team", "based_in", "Porto")


def model_claims(mem: Any) -> list[str]:
    return sorted(c.object for c in mem.get_all() if c.extractor == ScriptedModel.name)


@pytest.mark.parametrize("failure", [
    pytest.param(APITimeoutError(), id="sdk-timeout"),
    pytest.param(RateLimitError(), id="rate-limit-429"),
    pytest.param(AuthenticationError(), id="rejected-key-401"),
    pytest.param(APIStatusError("overloaded", status_code=529), id="overloaded-529"),
    pytest.param(APIConnectionError("connection reset"), id="connection-reset"),
    pytest.param(TimeoutError("read timed out"), id="python-timeout"),
    pytest.param(Truncated('{"claims": [{"subject": "team"'), id="cut-off-at-the-limit"),
])
def test_a_failed_extraction_keeps_the_turns_and_the_fast_paths_fact(
        scripted: Make, failure: object) -> None:
    model = scripted(extract=[failure])
    mem = with_model(model)
    seed(mem)
    before = ledger(mem)
    receipt = mem.add([FAST_TURN, MODEL_TURN])
    assert turns(mem, receipt) == [FAST_TURN, MODEL_TURN]
    assert [c.object for c in mem.get_all() if c.predicate == "name"] == ["Ada"]
    assert model_claims(mem) == []
    assert (receipt.deferred, receipt.unextracted) == (True, 1)
    assert receipt.llm_calls == model.count() == 1  # the failed call is billed
    assert set(fates(before, mem).values()) == {"unchanged"}


def test_a_deferred_turn_is_read_by_reextract_once_the_model_answers(
        scripted: Make) -> None:
    """INTERNALS: `reextract()` is for "a batch a provider failure left `deferred`". It
    reads the stored turn, and a second sweep finds nothing left to read."""
    model = scripted(extract=[RateLimitError(), [PORTO]])
    mem = with_model(model)
    seed(mem)
    failed = mem.add([FAST_TURN, MODEL_TURN])
    assert failed.deferred
    swept = mem.reextract()
    assert swept.llm_calls == 1
    [porto] = [c for c in mem.get_all() if c.object == "Porto"]
    assert porto.sources == [failed.episode_ids[1]]
    assert mem.reextract().llm_calls == 0
    assert model.count() == 2


def test_a_failed_acquisition_is_paid_once_and_never_retried(scripted: Make) -> None:
    """`WritePipeline._acquire`: a rate limit "must not cost the caller the whole batch of
    facts", and the spelling is "Marked resolved above, so we do not retry in a hot
    loop." The predicate stays unregistered, which means it holds many values."""
    model = scripted(extract=[[claim("team", "zqx_office_hub", "Porto")],
                              [claim("team", "zqx_office_hub", "Porto office")]],
                     resolve=[RateLimitError()])
    mem = with_model(model)
    first = mem.add(MODEL_TURN)
    assert first.llm_calls == model.count() == 2
    assert [c.object for c in mem.get_all() if c.predicate == "zqx_office_hub"] == ["Porto"]
    assert not mem.registry.known("zqx_office_hub")
    assert mem.registry.spec("zqx_office_hub").cardinality is Cardinality.MANY
    second = mem.add("The Porto office is where the whole team sits these days.")
    assert second.llm_calls == 1
    assert model.count("resolve_predicate") == 1


def test_a_failed_judge_leaves_the_write_whole_and_the_advice_empty(scripted: Make) -> None:
    """INTERNALS: "A judge that raises warns once per instance and leaves the list
    empty", because the claim is already durable. The neighbours are written through a
    handle with no model, so that writing them asks no judge."""
    store = SQLiteStore(":memory:")
    plain = without_model(store)
    plain.remember("user", "drinks", "green tea every morning")
    plain.remember("user", "orders", "green tea at the cafe")
    model = scripted(judge=[RateLimitError()])
    mem = with_model(model, store=store, advise_replacements=True)
    before = ledger(mem)
    with pytest.warns(RuntimeWarning, match="replacement advice failed"):
        receipt = mem.remember("user", "prefers", "green tea with lemon")
    assert [c.object for c in receipt.added] == ["green tea with lemon"]
    assert receipt.may_replace == []
    assert receipt.llm_calls == model.count() == 1
    assert set(fates(before, mem).values()) == {"unchanged"}


#: About 13,000 characters in 13 paragraphs, which the splitter cuts into three pieces.
LONG = "\n\n".join(
    f"Part {n} of the migration notes covers the database, the queue, the cache and the "
    "search index, and says which team owns each step and how it is rolled back. " * 6
    for n in range(13))


def test_a_failed_piece_of_a_long_turn_defers_that_turn_only(scripted: Make) -> None:
    """INTERNALS, `extraction_chunks`: "when a call carrying a turn fails, that turn's
    later pieces are not sent, what its earlier pieces returned is dropped, and the turn
    is deferred". The short turn shares the batch's first call and keeps its claim."""
    assert len(split_for_extraction(LONG)) == 3
    model = scripted(extract=[[PORTO], [claim("team", "migrates", "the database")],
                              RateLimitError()])
    mem = with_model(model, write_extraction_chunks=True)
    receipt = mem.add([LONG, MODEL_TURN])
    assert turns(mem, receipt) == [LONG, MODEL_TURN]
    assert receipt.llm_calls == model.count() == 3  # the third piece was never sent
    assert model_claims(mem) == ["Porto"]
    assert mem.store.claims_citing(mem.default_scope.tenant, receipt.episode_ids[0]) == []
    assert (receipt.deferred, receipt.unextracted) == (True, 1)
