"""JSON from /v1 turned back into the library's own dataclasses.

The round trip is the assertion that matters. A hand-written expectation agrees with
whatever the author misread; a claim that survives render-then-hydrate unchanged does not.
"""
from datetime import datetime, timezone

import pytest

from memvara import (
    Answer, Claim, Derivation, Edge, Episode, MemoryType, Path, Provenance, Reading,
    Scope, WriteReceipt, utcnow,
)
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


# --- the other eleven functions ---
#
# Each below builds the library object with values distinct from every dataclass
# default, renders it through the real `render.*` function, hydrates the result, and
# checks the fields survive. Distinct-from-default is the point: a field mapped to the
# wrong wire key produces a visible mismatch this way, rather than coincidentally
# matching a default and passing anyway.


def test_scope_survives_render_then_hydrate_unchanged():
    from memvara_cloud.rest import render
    original = Scope(tenant="acme-tenant", user="user-42", agent="agent-7",
                     session="sess-9")
    restored = hydrate.scope(render.scope(original).model_dump(mode="json"))
    assert restored == original


def test_episode_survives_render_then_hydrate_unchanged():
    from memvara_cloud.rest import render
    original = Episode(content="the roof leaks in the west corner", role="assistant",
                       scope=Scope("acme-tenant", "user-42", "agent-7", "sess-9"),
                       ts=datetime(2019, 3, 4, 5, 6, 7, tzinfo=timezone.utc),
                       id="ep_distinct_1", meta={"channel": "slack"})
    restored = hydrate.episode(render.episode(original).model_dump(mode="json"))
    for f in ("content", "role", "ts", "id"):
        assert getattr(restored, f) == getattr(original, f), f
    assert restored.scope == original.scope
    assert restored.meta == original.meta


def test_receipt_survives_render_then_hydrate_unchanged():
    """`closed` is the field the library's dataclass carries; the wire spells it
    `invalidated`, the pre-bitemporal name kept because it is published. `note` is
    rendered from `unextracted` and has no field on `WriteReceipt` to restore into —
    dropped for the same reason `Memory.state` is: it is derived, not stored."""
    from memvara_cloud.rest import render
    added_claim = Claim(subject="user", predicate="works_at", object="Acme")
    closed_claim = Claim(subject="user", predicate="lives_in", object="Lisbon",
                         valid_to=utcnow())
    reinforced_claim = Claim(subject="user", predicate="likes", object="tea")
    original = WriteReceipt(
        episode_ids=["ep_11", "ep_12"], added=[added_claim], closed=[closed_claim],
        reinforced=[reinforced_claim], skipped=3, unextracted=2, llm_calls=5,
        latency_ms=123.5, deferred=True,
    )
    wire = render.receipt(original, extractor="rules/9").model_dump(mode="json")
    restored = hydrate.receipt(wire)
    assert restored.episode_ids == original.episode_ids
    assert [c.id for c in restored.added] == [c.id for c in original.added]
    assert [c.id for c in restored.closed] == [c.id for c in original.closed]
    assert [c.id for c in restored.reinforced] == [c.id for c in original.reinforced]
    assert restored.skipped == original.skipped
    assert restored.unextracted == original.unextracted
    assert restored.llm_calls == original.llm_calls
    assert restored.latency_ms == original.latency_ms
    assert restored.deferred == original.deferred


def test_provenance_survives_render_then_hydrate_unchanged():
    from memvara_cloud.rest import render
    claim_val = Claim(subject="user", predicate="works_at", object="Acme")
    source_ep = Episode(content="I just started at Acme", role="user")
    superseded_claim = Claim(subject="user", predicate="works_at", object="Old Co",
                             valid_to=utcnow())
    original = Provenance(claim=claim_val, episodes=[source_ep],
                          derivation=Derivation.FAST_PATH, extractor="fast-path/1",
                          superseded=[superseded_claim])
    restored = hydrate.provenance(render.provenance(original).model_dump(mode="json"))
    assert restored.claim.id == original.claim.id
    assert restored.derivation == original.derivation
    assert restored.extractor == original.extractor
    assert [e.id for e in restored.episodes] == [e.id for e in original.episodes]
    assert [c.id for c in restored.superseded] == [c.id for c in original.superseded]


def test_reading_survives_render_then_hydrate_unchanged():
    """`timeline` and `single_valued` have no wire representation at all —
    `ReadingModel` carries neither field, so a restored `Reading` reports them at their
    dataclass defaults rather than from anything the wire said."""
    from memvara_cloud.rest import render
    now_claim = Claim(subject="user", predicate="lives_in", object="Lisbon")
    then_claim = Claim(subject="user", predicate="lives_in", object="Berlin",
                       valid_to=utcnow())
    stated_claim = Claim(subject="user", predicate="lives_in", object="Porto")
    original = Reading(subject="user", predicate="lives_in", now=(now_claim,),
                       then=(then_claim,), stated=(stated_claim,))
    restored = hydrate.reading(render.reading(original).model_dump(mode="json"))
    assert restored.subject == original.subject
    assert restored.predicate == original.predicate
    assert [c.id for c in restored.now] == [c.id for c in original.now]
    assert [c.id for c in restored.then] == [c.id for c in original.then]
    assert [c.id for c in restored.stated] == [c.id for c in original.stated]
    assert restored.timeline == ()
    assert restored.single_valued is False


def test_answer_survives_render_then_hydrate_unchanged():
    from memvara_cloud.rest import render
    reading_val = Reading(
        subject="user", predicate="lives_in",
        now=(Claim(subject="user", predicate="lives_in", object="Lisbon"),))
    original = Answer(question="where do they live now?",
                      at=datetime(2026, 3, 15, 9, 30, tzinfo=timezone.utc),
                      readings=(reading_val,), text="They currently live in Lisbon.")
    restored = hydrate.answer(render.answer(original).model_dump(mode="json"))
    assert restored.question == original.question
    assert restored.at == original.at
    assert restored.text == original.text
    assert len(restored.readings) == len(original.readings)
    assert restored.readings[0].subject == original.readings[0].subject


def test_delta_survives_render_then_hydrate_unchanged():
    """`render.py` has no `delta` function — `DeltaResponse` is assembled directly by
    the route rather than through a dedicated renderer, so this builds the wire model
    by hand instead of going through `render.*`."""
    from memvara_cloud.rest import models as m
    from memvara_cloud.rest import render
    added_claim = Claim(subject="user", predicate="works_at", object="Acme")
    gone_claim = Claim(subject="user", predicate="works_at", object="Old Co",
                       valid_to=utcnow())
    since = datetime(2026, 1, 1, tzinfo=timezone.utc)
    wire = m.DeltaResponse(since=since, added=[render.memory(added_claim)],
                          gone=[render.memory(gone_claim)]).model_dump(mode="json")
    restored = hydrate.delta(wire)
    assert restored.since == since
    assert [c.id for c in restored.added] == [added_claim.id]
    assert [c.id for c in restored.gone] == [gone_claim.id]


def test_edge_survives_render_then_hydrate_unchanged():
    from memvara_cloud.rest import render
    claim_val = Claim(subject="Alice", predicate="founded", object="Acme")
    original = Edge(claim=claim_val, backward=True, strength=0.37)
    restored = hydrate.edge(render.edge(original).model_dump(mode="json"))
    assert restored.claim.id == original.claim.id
    assert restored.backward == original.backward
    assert restored.strength == original.strength


def test_path_survives_render_then_hydrate_unchanged():
    """`hops` and `labels` are properties computed from `edges`, not stored fields —
    the wire carries both (`render.path` renders them from the library's own
    properties), and they are deliberately not passed back into the constructor."""
    from memvara_cloud.rest import render
    claim_val = Claim(subject="Alice", predicate="works_at", object="Acme")
    edge_val = Edge(claim=claim_val, backward=False, strength=0.81)
    original = Path(nodes=("alice", "acme"), edges=(edge_val,), score=0.63)
    restored = hydrate.path(render.path(original).model_dump(mode="json"))
    assert restored.nodes == original.nodes
    assert restored.score == original.score
    assert len(restored.edges) == 1
    assert restored.edges[0].claim.id == edge_val.claim.id
    assert restored.edges[0].backward == edge_val.backward
    assert restored.edges[0].strength == edge_val.strength
    assert restored.hops == 1
    assert restored.labels == ("Alice", "Acme")
