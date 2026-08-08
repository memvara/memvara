"""Core data model: scope hierarchy, claim identity, and the bitemporal predicate."""

from datetime import datetime, timedelta, timezone

import pytest

from engram.types import (
    Claim,
    Derivation,
    Episode,
    MemoryType,
    Scope,
    WriteReceipt,
    content_hash,
    utcnow,
)

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
T1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
T2 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def mk(**kw) -> Claim:
    base = dict(subject="user", predicate="lives_in", object="Berlin",
                scope=Scope("acme", "alice"))
    base.update(kw)
    return Claim(**base)


# --- Scope ------------------------------------------------------------------

def test_ancestors_walks_from_narrow_to_broad():
    s = Scope("acme", "alice", "bot", "sess1")
    keys = [a.key() for a in s.ancestors()]
    assert keys == [
        "acme/alice/bot/sess1",
        "acme/alice/bot/*",
        "acme/alice/*/*",
        "acme/*/*/*",
    ]


def test_ancestors_of_bare_tenant_is_just_itself():
    assert [a.key() for a in Scope("acme").ancestors()] == ["acme/*/*/*"]


def test_ancestors_are_deduplicated():
    s = Scope("acme", "alice")
    assert len(s.ancestors()) == len({a.key() for a in s.ancestors()})


def test_contains_is_downward_only():
    user = Scope("acme", "alice")
    session = Scope("acme", "alice", "bot", "s1")
    assert user.contains(session)
    assert not session.contains(user)


def test_contains_rejects_sibling_users_and_tenants():
    alice = Scope("acme", "alice")
    assert not alice.contains(Scope("acme", "bob"))
    assert not alice.contains(Scope("other", "alice"))
    assert not Scope("acme").contains(Scope("other", "alice"))


# --- Claim identity ---------------------------------------------------------

def test_fact_key_groups_competing_values_for_one_slot():
    """Same question, different answers -> same slot. This is the conflict condition."""
    assert mk(object="Berlin").fact_key == mk(object="Lisbon").fact_key
    assert mk(object="Berlin").value_key != mk(object="Lisbon").value_key


def test_fact_key_separates_different_predicates():
    assert mk(predicate="lives_in").fact_key != mk(predicate="works_at").fact_key


def test_fact_key_does_not_collide_across_users():
    """The bug this guards: extraction emits a generic subject ("user"), so without the
    scope owner in the key, Bob's city would silently retire Alice's."""
    alice = mk(scope=Scope("acme", "alice"))
    bob = mk(scope=Scope("acme", "bob"))
    assert alice.fact_key != bob.fact_key
    assert alice.value_key != bob.value_key


def test_fact_key_does_not_collide_across_tenants():
    assert mk(scope=Scope("acme", "alice")).fact_key != mk(scope=Scope("other", "alice")).fact_key


def test_fact_key_ignores_agent_and_session():
    """Learning a fact in a new session must still retire the old value, so the key
    deliberately does not include agent or session."""
    a = mk(scope=Scope("acme", "alice", "bot", "s1"))
    b = mk(scope=Scope("acme", "alice", "other", "s2"))
    assert a.fact_key == b.fact_key


def test_value_key_distinguishes_polarity():
    assert mk(polarity=1).value_key != mk(polarity=-1).value_key


# --- Bitemporal liveness ----------------------------------------------------

def test_not_live_before_we_recorded_it():
    c = mk(recorded_at=T1, valid_from=T1)
    assert not c.is_live(T0)
    assert c.is_live(T1)
    assert c.is_live(T2)


def test_invalidation_ends_liveness_at_that_instant():
    c = mk(recorded_at=T0, valid_from=T0, invalidated_at=T2)
    assert c.is_live(T1)
    assert not c.is_live(T2)


def test_valid_to_ends_liveness_independently_of_invalidation():
    c = mk(recorded_at=T0, valid_from=T0, valid_to=T2)
    assert c.is_live(T1)
    assert not c.is_live(T2)


def test_late_arriving_fact_does_not_rewrite_what_we_believed():
    """The whole point of two time axes: the fact was true from T0, but we only learned
    it at T2, so a query about what we believed at T1 must not see it."""
    c = mk(valid_from=T0, recorded_at=T2)
    assert not c.is_live(T1)
    assert c.is_live(T2)


def test_freshly_built_claim_is_live_now():
    assert mk().is_live()


# --- Rendering and misc -----------------------------------------------------

def test_render_produces_readable_text():
    assert mk().render() == "user lives in Berlin"


def test_render_marks_negation():
    assert mk(polarity=-1).render() == "user no longer lives in Berlin"


def test_text_defaults_to_render_but_explicit_text_wins():
    assert mk().text == "user lives in Berlin"
    assert mk(text="custom").text == "custom"


def test_claim_ids_are_unique():
    assert len({mk().id for _ in range(200)}) == 200


@pytest.mark.parametrize("mt", list(MemoryType))
def test_memory_types_round_trip_through_value(mt):
    assert MemoryType(mt.value) is mt


@pytest.mark.parametrize("d", list(Derivation))
def test_derivations_round_trip_through_value(d):
    assert Derivation(d.value) is d


# --- Episode ----------------------------------------------------------------

def test_episode_hash_is_deterministic():
    a = Episode(content="hello", scope=Scope("acme", "alice"))
    b = Episode(content="hello", scope=Scope("acme", "alice"))
    assert a.hash == b.hash
    assert a.id != b.id


def test_episode_hash_separates_scope_and_role():
    base = dict(content="hello")
    assert (
        Episode(**base, scope=Scope("acme", "alice")).hash
        != Episode(**base, scope=Scope("acme", "bob")).hash
    )
    assert (
        Episode(**base, role="user").hash != Episode(**base, role="assistant").hash
    )


def test_content_hash_is_not_confusable_across_field_boundaries():
    """Naive concatenation would make ("ab","c") and ("a","bc") collide."""
    assert content_hash("ab", "c") != content_hash("a", "bc")


def test_utcnow_is_timezone_aware():
    assert utcnow().tzinfo is not None
    assert utcnow() - utcnow() < timedelta(seconds=1)


# --- Receipt ----------------------------------------------------------------

def test_receipt_summary_surfaces_llm_call_count():
    r = WriteReceipt(added=[mk()], skipped=3, llm_calls=0, latency_ms=1.5)
    text = str(r)
    assert "+1" in text and "skip=3" in text and "llm=0" in text
