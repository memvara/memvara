"""Core data model: scope hierarchy, claim identity, and the bitemporal predicate."""

from datetime import datetime, timedelta, timezone

import pytest

from engram.types import (
    OBJECT_ENTITY,
    Claim,
    Derivation,
    Episode,
    Explanation,
    MemoryType,
    Provenance,
    Result,
    Scope,
    WriteReceipt,
    content_hash,
    fact_key_for,
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


# --- entity identity ---------------------------------------------------------
# Both keys hash entity identities rather than raw text. Hashing the text made "Acme",
# "Acme Corp" and "ACME" three employers, so a single-valued predicate reported two job
# changes that never happened. See `engram/entities.py`.

def test_value_key_folds_spellings_of_one_value():
    assert mk(object="Acme").value_key == mk(object="Acme Corp").value_key
    assert mk(object="Acme").value_key == mk(object="  ACME, Inc. ").value_key


def test_value_key_still_separates_genuinely_different_values():
    assert mk(object="Acme").value_key != mk(object="Acme Labs").value_key


def test_fact_key_folds_spellings_of_one_subject():
    assert mk(subject="Acme").fact_key == mk(subject="acme inc").fact_key


def test_fact_key_for_folds_a_raw_subject_the_same_way():
    # `Engram.history` and `Engram.forget` build their probe from a raw string with no
    # registry in reach, so this is the equality that keeps them able to find a slot.
    c = mk(subject="Acme Corp")
    assert fact_key_for(c.scope, "ACME", c.predicate) == c.fact_key


def test_a_stamp_beats_the_fold():
    c = mk(object="Big Blue", meta={OBJECT_ENTITY: "ibm"})
    assert c.object_key == "ibm"
    assert c.value_key == mk(object="IBM").value_key


def test_a_stamp_that_is_not_a_usable_string_is_ignored():
    for junk in (None, 17, ""):
        assert mk(object="Acme", meta={OBJECT_ENTITY: junk}).object_key == "acme"


def test_an_unfoldable_value_keeps_its_own_identity():
    # "…" folds to nothing; collapsing every such value onto the empty string would make
    # them all one value, which is the one direction this must never err in.
    assert mk(object="...").value_key != mk(object="???").value_key
    assert mk(object="...").object_key == "..."


def test_display_text_is_untouched_by_resolution():
    c = mk(object="  ACME, Inc. ")
    assert c.object == "  ACME, Inc. " and c.object_key == "acme"


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


def test_receipt_reports_turns_that_reached_extraction_and_yielded_nothing():
    """`skipped` is the write path working as designed; `unextracted` is content that
    was lost. Collapsing the two is how a configuration that stores nothing reports a
    clean, successful write."""
    assert "unextracted=4" in str(WriteReceipt(skipped=1, unextracted=4))


def test_receipt_stays_quiet_when_nothing_was_lost():
    assert "unextracted" not in str(WriteReceipt(added=[mk()]))


# --- Reprs ------------------------------------------------------------------
# The bar is `WriteReceipt.__str__`: one line, the fields you would actually ask for,
# and nothing that forces a REPL to scroll. The dataclass default prints every field of
# every nested object, which made `history()` output unreadable in the README's own
# example.

def test_scope_repr_is_the_scope_key():
    assert repr(Scope("acme", "alice")) == "<Scope acme/alice/*/*>"


def test_episode_repr_fits_on_one_line():
    ep = Episode(content="I live in Berlin", scope=Scope("acme", "alice"),
                 ts=datetime(2025, 3, 4, 9, 30, tzinfo=timezone.utc))
    text = repr(ep)
    assert text.startswith(f"<Episode {ep.id} acme/alice/*/* user 2025-03-04 09:30Z")
    assert "'I live in Berlin'" in text
    assert "\n" not in text


def test_episode_repr_flattens_and_truncates_hostile_content():
    ep = Episode(content="line one\nline two " + "x" * 200)
    text = repr(ep)
    assert "\n" not in text and len(text) < 160


def test_claim_repr_names_the_slot_the_value_and_the_state():
    c = mk()
    text = repr(c)
    assert text == (f"<Claim {c.id} acme/alice/*/* user lives_in='Berlin' semantic "
                    "conf=1.00 sal=1.00 live>")


def test_claim_repr_distinguishes_the_two_ways_a_claim_stops_counting():
    """Retired (we stopped believing it) and ended (it stopped being true) are different
    facts about a claim, and the whole point of two time axes is not to conflate them."""
    assert "retired" in repr(mk(invalidated_at=T2))
    assert "ended" in repr(mk(valid_to=T2))


def test_claim_repr_marks_negation():
    assert "not lives_in=" in repr(mk(polarity=-1))


def test_result_repr_shows_the_score_and_which_retrievers_fired():
    r = Result(claim=mk(), score=0.4231,
               explain=Explanation(vector_rank=0, vector_score=0.8, lexical_rank=2,
                                   lexical_score=1.4, final_score=0.4231))
    text = repr(r)
    assert "<Result 0.4231 'user lives in Berlin' vector#0+bm25#2" in text
    assert r.claim.id in text, "the id is what you paste into why()"


def test_result_repr_says_so_when_neither_retriever_ranked_it():
    assert "no-retriever" in repr(Result(claim=mk(), score=0.0, explain=Explanation()))


def test_explanation_repr_is_its_summary():
    e = Explanation(lexical_rank=1, lexical_score=2.0, final_score=0.5)
    assert repr(e) == f"<Explanation {e.summary()}>"


def test_explanation_summary_shows_the_raw_score_once_a_retriever_sets_one():
    """`raw_score` is the pre-normalization value. It is only meaningful next to the
    normalized one, and only present once something computes it."""
    assert "raw=" not in Explanation(final_score=0.5).summary()
    assert "raw=0.0310" in Explanation(raw_score=0.031, final_score=0.5).summary()


def test_provenance_repr_summarises_the_trail_without_dumping_it():
    c = mk()
    p = Provenance(claim=c, episodes=[Episode(content="I live in Berlin")],
                   derivation=Derivation.FAST_PATH, extractor="fast/v1",
                   superseded=[mk(object="Lisbon")])
    assert repr(p) == (f"<Provenance {c.id} 'user lives in Berlin' via fast/v1 "
                       "(fast_path) sources=1 superseded=1>")


def test_provenance_repr_survives_an_unattributed_claim():
    p = Provenance(claim=mk(), episodes=[], derivation=Derivation.USER, extractor="")
    assert "via ? (user)" in repr(p)
