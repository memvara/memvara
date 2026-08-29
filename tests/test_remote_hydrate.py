"""JSON from /v1 turned back into the library's own dataclasses.

The round trip is the assertion that matters. A hand-written expectation agrees with
whatever the author misread; a claim that survives render-then-hydrate unchanged does not.
"""
import pytest

from memvara import Claim, Derivation, MemoryType, Scope, utcnow
from memvara.remote import hydrate
from memvara.types import LAST_OBSERVED, SALIENCE_BASE

pytest.importorskip("memvara_cloud", reason="the renderer is the authority for this test")


def _wire(claim):
    from memvara_cloud.rest import render
    return render.memory(claim).model_dump(mode="json")


def test_a_claim_survives_render_then_hydrate_unchanged():
    original = Claim(subject="user", predicate="lives_in", object="Berlin",
                     scope=Scope("t", "u"), text="user lives in Berlin",
                     memory_type=MemoryType.SEMANTIC, confidence=0.9,
                     derivation=Derivation.LLM_EXTRACT, extractor="rules/1",
                     sources=["ep_1"], meta={"note": "kept"})
    restored = hydrate.claim(_wire(original))
    for field in ("subject", "predicate", "object", "text", "polarity", "memory_type",
                  "confidence", "salience", "observation_count", "derivation",
                  "extractor", "id", "valid_from", "valid_to", "recorded_at",
                  "invalidated_at", "invalidated_by"):
        assert getattr(restored, field) == getattr(original, field), field
    assert restored.scope == original.scope
    assert restored.sources == original.sources
    assert restored.meta["note"] == "kept"


def test_an_unrecorded_extractor_comes_back_as_the_empty_string_not_none():
    original = Claim(subject="user", predicate="likes", object="tea", extractor="")
    assert hydrate.claim(_wire(original)).extractor == ""


def test_salience_base_and_last_observed_return_to_meta():
    original = Claim(subject="user", predicate="likes", object="tea")
    original.meta[SALIENCE_BASE] = 2.5
    original.meta[LAST_OBSERVED] = 1700000000.0
    restored = hydrate.claim(_wire(original))
    assert restored.meta[SALIENCE_BASE] == 2.5
    assert restored.meta[LAST_OBSERVED] == 1700000000.0


def test_the_closure_record_survives_because_it_is_not_a_reserved_key():
    from memvara.types import CLOSURE, close_out
    original = Claim(subject="user", predicate="works_at", object="Acme")
    close_out(original, utcnow(), None, "ended")
    assert CLOSURE in hydrate.claim(_wire(original)).meta


def test_a_missing_required_field_raises_rather_than_defaulting():
    body = _wire(Claim(subject="user", predicate="likes", object="tea"))
    del body["predicate"]
    with pytest.raises(KeyError):
        hydrate.claim(body)


def test_state_is_recomputed_and_never_read_from_the_wire():
    original = Claim(subject="user", predicate="likes", object="tea")
    body = _wire(original)
    body["state"] = "retired"          # a lie the hydrator must ignore
    assert hydrate.claim(body).is_live()
