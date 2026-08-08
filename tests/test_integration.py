"""End-to-end tests through the public `Engram` surface.

These are the tests that would catch a subsystem being wired up wrong even when every
unit test passes. They deliberately assert on the properties the README claims:
contradictions resolve, history survives, users are isolated, and the LLM stays idle.
"""

from datetime import datetime, timedelta, timezone

import pytest

from engram import Engram, HashingEmbedder, MemoryType, SQLiteStore

TZ = timezone.utc
T_2023 = datetime(2023, 1, 1, tzinfo=TZ)
T_2024 = datetime(2024, 1, 1, tzinfo=TZ)
T_2025 = datetime(2025, 1, 1, tzinfo=TZ)


class FakeLLM:
    """Scripted extractor that counts its own calls.

    Call counting is the point: the design claim is that the model is consulted rarely,
    so every test that exercises the write path asserts on these counters.
    """

    name = "fake"

    def __init__(self, script: dict[str, list[dict]] | None = None) -> None:
        self.script = script or {}
        self.extract_calls = 0
        self.classify_calls = 0
        self.episodes_seen = 0

    def extract(self, episodes, known_predicates):
        self.extract_calls += 1
        self.episodes_seen += len(episodes)
        out = []
        for i, ep in enumerate(episodes):
            for needle, claims in self.script.items():
                if needle.lower() in ep.content.lower():
                    for c in claims:
                        d = {"polarity": 1, "memory_type": "semantic",
                             "confidence": 0.9, **c, "source_index": i}
                        out.append(d)
        return out

    def classify_predicate(self, predicate, example):
        self.classify_calls += 1
        return {"cardinality": "many", "volatility": "slow", "memory_type": "semantic"}


@pytest.fixture()
def mem():
    m = Engram(embedder=HashingEmbedder(dim=128), user="alice")
    yield m
    m.close()


# --- The headline behavior: contradictions resolve --------------------------

def test_moving_city_retires_the_old_one(mem):
    mem.remember("user", "lives_in", "Berlin")
    mem.remember("user", "lives_in", "Lisbon")
    assert [c.object for c in mem.get_all()] == ["Lisbon"]


def test_the_retired_fact_is_kept_not_deleted(mem):
    mem.remember("user", "lives_in", "Berlin")
    mem.remember("user", "lives_in", "Lisbon")
    history = mem.history("user", "lives_in")
    assert [c.object for c in history] == ["Berlin", "Lisbon"]
    assert history[0].invalidated_at is not None
    assert history[0].invalidated_by == history[1].id
    assert history[1].invalidated_at is None


def test_predicate_aliases_still_collide(mem):
    """'resides_in' and 'lives_in' are the same slot, so the second must retire the
    first — this is the case a free-text store silently gets wrong."""
    mem.remember("user", "lives_in", "Berlin")
    mem.remember("user", "resides_in", "Lisbon")
    assert [c.object for c in mem.get_all()] == ["Lisbon"]


def test_conjoined_clauses_yield_both_facts_with_no_llm(mem):
    """The README's headline example. Subject elision after "and" ("...and work at
    Acme") is common enough that missing it sent an ordinary sentence to the LLM and
    stored nothing under the default no-model configuration."""
    receipt = mem.add("I live in Berlin and work at Acme")
    assert receipt.llm_calls == 0
    assert sorted((c.predicate, c.object) for c in mem.get_all()) == [
        ("lives_in", "Berlin"), ("works_at", "Acme")
    ]


def test_coordinated_objects_are_not_split_into_bogus_facts(mem):
    """The counterpart risk: "coffee and tea" is one object list, not a second clause.
    Extracting nothing is correct here — the LLM tier exists for this case."""
    mem.add("I like coffee and tea")
    assert [(c.predicate, c.object) for c in mem.get_all()] == []


def test_the_readme_walkthrough_holds_end_to_end(mem):
    mem.add("I live in Berlin and work at Acme")
    mem.add("Actually, I moved to Lisbon last month")
    assert [r.text for r in mem.search("where do they live?")][:1] == ["user lives in Lisbon"]
    assert [(c.object, c.invalidated_at is not None)
            for c in mem.history("user", "lives_in")] == [("Berlin", True), ("Lisbon", False)]


def test_multi_valued_predicates_accumulate(mem):
    mem.remember("user", "likes", "coffee")
    mem.remember("user", "likes", "tea")
    assert {c.object for c in mem.get_all()} == {"coffee", "tea"}


def test_unrelated_predicates_do_not_interfere(mem):
    mem.remember("user", "lives_in", "Berlin")
    mem.remember("user", "works_at", "Acme")
    assert {c.object for c in mem.get_all()} == {"Berlin", "Acme"}


def test_reasserting_the_same_fact_reinforces_instead_of_duplicating(mem):
    mem.remember("user", "lives_in", "Berlin")
    mem.remember("user", "lives_in", "Berlin")
    live = mem.get_all()
    assert len(live) == 1
    assert live[0].observation_count >= 2


# --- Time travel ------------------------------------------------------------

def test_search_as_of_returns_the_belief_of_that_moment(mem):
    mem.remember("user", "lives_in", "Berlin", recorded_at=T_2023)
    mem.remember("user", "lives_in", "Lisbon", recorded_at=T_2025)

    past = [r.claim.object for r in mem.search("lives", as_of=T_2024)]
    now = [r.claim.object for r in mem.search("lives")]
    assert past == ["Berlin"]
    assert now == ["Lisbon"]


def test_get_all_as_of_respects_transaction_time(mem):
    mem.remember("user", "lives_in", "Berlin", recorded_at=T_2023)
    mem.remember("user", "works_at", "Acme", recorded_at=T_2025)
    assert {c.object for c in mem.get_all(as_of=T_2024)} == {"Berlin"}


def test_backfilled_facts_separate_the_two_time_axes(mem):
    """True since 2019, learned today: valid time is old, transaction time is now."""
    mem.remember("user", "born_in", "Osaka", valid_from=datetime(1990, 1, 1, tzinfo=TZ))
    c = mem.get_all()[0]
    assert c.valid_from.year == 1990
    assert c.recorded_at.year >= 2024
    assert mem.get_all(as_of=T_2023) == []


# --- Provenance -------------------------------------------------------------

def test_why_traces_a_claim_back_to_the_source_turn():
    llm = FakeLLM({"lisbon": [{"subject": "user", "predicate": "lives_in",
                               "object": "Lisbon"}]})
    with Engram(embedder=HashingEmbedder(dim=128), llm=llm, user="alice") as mem:
        mem.add(["Good morning!", "I just moved to Lisbon.", "Anyway, thanks."])
        claim = mem.get_all()[0]
        prov = mem.why(claim.id)

        assert prov is not None
        assert len(prov.episodes) == 1
        assert "Lisbon" in prov.episodes[0].content, "must point at the right turn"
        assert prov.extractor


def test_why_reports_what_a_claim_superseded(mem):
    mem.remember("user", "lives_in", "Berlin")
    mem.remember("user", "lives_in", "Lisbon")
    current = mem.get_all()[0]
    prov = mem.why(current.id)
    assert [c.object for c in prov.superseded] == ["Berlin"]


def test_why_on_unknown_claim_returns_none(mem):
    assert mem.why("cl_does_not_exist") is None


# --- Scope isolation --------------------------------------------------------

def test_two_users_in_one_tenant_never_collide(mem):
    """Extraction emits a generic 'user' subject, so this is the case that silently
    corrupts a store keyed only on (subject, predicate)."""
    mem.remember("user", "lives_in", "Berlin", user="alice")
    mem.remember("user", "lives_in", "Lisbon", user="bob")
    assert [c.object for c in mem.get_all(user="alice")] == ["Berlin"]
    assert [c.object for c in mem.get_all(user="bob")] == ["Lisbon"]


def test_tenants_are_isolated(mem):
    mem.remember("user", "lives_in", "Berlin", tenant="t1", user="alice")
    mem.remember("user", "lives_in", "Lisbon", tenant="t2", user="alice")
    assert [c.object for c in mem.get_all(tenant="t1", user="alice")] == ["Berlin"]
    assert [c.object for c in mem.get_all(tenant="t2", user="alice")] == ["Lisbon"]


def test_a_session_inherits_user_memory(mem):
    mem.remember("user", "lives_in", "Berlin")
    mem.remember("user", "working_on", "auth refactor", session="s1")
    objects = {c.object for c in mem.get_all(session="s1")}
    assert objects == {"Berlin", "auth refactor"}


def test_sibling_sessions_do_not_see_each_others_scratch(mem):
    mem.remember("user", "working_on", "auth refactor", session="s1")
    assert mem.get_all(session="s2") == []


def test_user_scope_does_not_see_session_scratch(mem):
    mem.remember("user", "working_on", "auth refactor", session="s1")
    assert mem.get_all() == []


# --- The write-path cost claim ---------------------------------------------

def test_chitchat_never_reaches_the_model():
    llm = FakeLLM()
    with Engram(embedder=HashingEmbedder(dim=128), llm=llm, user="alice") as mem:
        receipt = mem.add(["hi", "ok", "thanks!", "sounds good", "great", "yep",
                           "sure thing", "no worries", "?", "  "])
        assert llm.extract_calls == 0, "pure acknowledgements must not cost an LLM call"
        assert receipt.llm_calls == 0
        assert receipt.skipped >= 8


def test_a_batch_of_turns_costs_at_most_one_extraction_call():
    llm = FakeLLM({"berlin": [{"subject": "user", "predicate": "lives_in",
                               "object": "Berlin"}]})
    with Engram(embedder=HashingEmbedder(dim=128), llm=llm, user="alice") as mem:
        turns = ["hello there", "how are you", "I recently relocated to Berlin for work",
                 "that's interesting", "tell me more", "thanks!"]
        receipt = mem.add(turns)
        assert llm.extract_calls <= 1, "turns must batch into one call, not one per turn"
        assert receipt.llm_calls == llm.extract_calls


def test_reingesting_the_same_transcript_is_free():
    llm = FakeLLM({"berlin": [{"subject": "user", "predicate": "lives_in",
                               "object": "Berlin"}]})
    with Engram(embedder=HashingEmbedder(dim=128), llm=llm, user="alice") as mem:
        turns = ["I moved to Berlin", "cool", "yes"]
        mem.add(turns)
        calls_after_first = llm.extract_calls
        receipt = mem.add(turns)
        assert llm.extract_calls == calls_after_first, "duplicate turns must not re-extract"
        assert receipt.llm_calls == 0
        assert len(mem.get_all()) == 1


def test_direct_assertions_never_call_the_model():
    llm = FakeLLM()
    with Engram(embedder=HashingEmbedder(dim=128), llm=llm, user="alice") as mem:
        mem.remember("user", "lives_in", "Berlin")
        assert llm.extract_calls == 0 and llm.classify_calls == 0


def test_everything_works_with_no_llm_at_all(mem):
    """Default configuration has no model and no API key; it must still be useful."""
    receipt = mem.add(["I live in Berlin", "hello"])
    assert receipt.llm_calls == 0
    mem.remember("user", "likes", "coffee")
    assert mem.get_all()


# --- Input handling ---------------------------------------------------------

def test_add_accepts_a_bare_string(mem):
    assert mem.add("I live in Berlin").episode_ids


def test_add_accepts_openai_style_transcripts(mem):
    receipt = mem.add([
        {"role": "user", "content": "I live in Berlin"},
        {"role": "assistant", "content": "Noted."},
    ])
    assert len(receipt.episode_ids) == 2


def test_add_preserves_extra_message_fields_as_metadata(mem):
    mem.add([{"role": "user", "content": "hello there friend", "turn_id": 42}])
    ep = mem.store.get_episode(mem.add([{"role": "user", "content": "x", "turn_id": 7}]).episode_ids[0])
    assert ep.meta.get("turn_id") == 7


@pytest.mark.parametrize("junk", ["", "   ", "\n\n", "🙂", "?" * 500])
def test_degenerate_input_does_not_raise(mem, junk):
    assert mem.add(junk) is not None


def test_very_large_turn_is_handled(mem):
    assert mem.add("word " * 20_000) is not None


def test_empty_batch_is_a_noop(mem):
    receipt = mem.add([])
    assert receipt.episode_ids == [] and receipt.llm_calls == 0


# --- Retrieval surface ------------------------------------------------------

def test_recall_renders_a_prompt_ready_block(mem):
    mem.remember("user", "lives_in", "Lisbon")
    mem.remember("user", "prefers_tool", "pytest")
    block = mem.recall("lives tool", k=5)
    assert block.startswith("Known about the user:")
    assert "- user" in block


def test_recall_is_empty_when_nothing_matches(mem):
    assert mem.recall("nothing here at all") == ""


def test_search_results_all_carry_explanations(mem):
    mem.remember("user", "lives_in", "Lisbon")
    for r in mem.search("lives"):
        assert r.explain is not None
        assert r.explain.final_score == pytest.approx(r.score)
        assert isinstance(r.explain.summary(), str)


def test_search_can_filter_by_memory_type(mem):
    mem.remember("user", "lives_in", "Lisbon")
    mem.remember("user", "prefers", "concise answers")
    only = mem.search("lives prefers", memory_types=[MemoryType.PROCEDURAL])
    assert all(r.claim.memory_type is MemoryType.PROCEDURAL for r in only)


def test_search_on_empty_memory_returns_nothing(mem):
    assert mem.search("anything") == []


# --- Forgetting -------------------------------------------------------------

def test_forget_retires_the_slot_but_keeps_the_audit_trail(mem):
    mem.remember("user", "lives_in", "Berlin")
    retired = mem.forget("user", "lives_in")
    assert len(retired) == 1
    assert mem.get_all() == []
    assert len(mem.history("user", "lives_in")) == 1


def test_forget_on_an_empty_slot_is_harmless(mem):
    assert mem.forget("user", "lives_in") == []


# --- Maintenance and lifecycle ---------------------------------------------

def test_consolidate_runs_and_reports_counts(mem):
    mem.remember("user", "likes", "coffee")
    mem.remember("user", "lives_in", "Lisbon")
    stats = mem.consolidate()
    assert isinstance(stats, dict) and stats


def test_consolidate_is_idempotent(mem):
    for i in range(5):
        mem.remember("user", f"pred_{i}", f"value_{i}")
    mem.consolidate()
    snapshot = {c.id: (c.salience, c.object, c.invalidated_at) for c in mem.get_all()}
    mem.consolidate()
    assert {c.id: (c.salience, c.object, c.invalidated_at) for c in mem.get_all()} == snapshot


def test_memory_survives_a_restart(tmp_path):
    path = str(tmp_path / "m.db")
    with Engram(path, embedder=HashingEmbedder(dim=128), user="alice") as m1:
        m1.remember("user", "lives_in", "Berlin")
        m1.remember("user", "lives_in", "Lisbon")

    with Engram(path, embedder=HashingEmbedder(dim=128), user="alice") as m2:
        assert [c.object for c in m2.get_all()] == ["Lisbon"]
        assert len(m2.history("user", "lives_in")) == 2
        assert [r.claim.object for r in m2.search("lives")] == ["Lisbon"]


def test_engram_accepts_an_injected_store():
    store = SQLiteStore(":memory:")
    with Engram(store=store, embedder=HashingEmbedder(dim=128)) as mem:
        assert mem.store is store


def test_unknown_tuning_options_are_rejected_loudly():
    with pytest.raises(TypeError, match="unknown tuning options"):
        Engram(nonsense_option=1)


def test_repr_surfaces_scope_and_counts(mem):
    mem.remember("user", "lives_in", "Lisbon")
    assert "claims=1" in repr(mem)


def test_stats_reports_store_contents(mem):
    mem.remember("user", "lives_in", "Lisbon")
    s = mem.stats()
    assert s["live_claims"] == 1 and s["claims"] == 1
