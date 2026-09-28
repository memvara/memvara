"""mem0 compatibility: the shim, and the importer that makes a migration possible.

Two things are under test, and they fail in opposite directions.

The **shim** fails by being too accommodating. Its job is to run an existing mem0 call
site unmodified, and the temptation is to make every method return something plausible —
which is how `delete()` ends up reporting a GDPR erasure it did not perform. So most of
these tests assert on *refusals*: the calls that raise, the warning that fires once, and
the semantics that differ and say so.

The **importer** fails by being lossy. mem0's `~/.mem0/history.db` is a complete
transaction-time history that mem0 itself cannot query, and the whole claim of this
workstream is that replaying it costs nothing and produces a real bitemporal store. So
those tests assert on time travel: what was believed on a date, what superseded what,
and whether `why()` still reaches the mem0 row.

Nothing here imports mem0. It is not installed, it is not needed, and a compatibility
layer that could only be tested against the thing it replaces would be untestable in
exactly the environment a migration runs in.
"""

import sqlite3
import warnings
from datetime import datetime, timedelta, timezone

import pytest

from harness.fakes.fake_v1 import FakeV1
from memvara import Memvara, HashingEmbedder, MemoryType, NullLLM
from memvara.compat import (
    ContestedSlot,
    ImportReceipt,
    Mem0CompatError,
    Mem0DeletionWarning,
    Memory,
    NOTE_PREDICATE,
    import_mem0,
    note_subject,
    read_history_db,
)
from memvara.compat import mem0_import
from memvara.compat._notes import ensure_note_predicate
from memvara.compat.mem0 import _memory_type, _reject_legacy_kwargs
from memvara.compat.mem0_import import _confidence, _parse_ts
from memvara.llm import TruncatedResponse

TZ = timezone.utc
T0 = datetime(2024, 1, 1, tzinfo=TZ)


def at(days: int) -> datetime:
    return T0 + timedelta(days=days)


@pytest.fixture()
def mem():
    # NullLLM by name, not by default: the default warns about degraded extraction, and
    # a suite that trips that warning teaches everyone to filter the category.
    m = Memvara(embedder=HashingEmbedder(dim=128), llm=NullLLM(), user="alice")
    yield m
    m.close()


@pytest.fixture()
def api(mem):
    return Memory(mem)


class FakeLLM:
    """A scripted extractor, so phase 2 runs offline and its cost is countable."""

    name = "fake"

    def __init__(self, script: dict[str, list[dict]] | None = None) -> None:
        self.script = script or {}
        self.calls = 0

    def extract(self, episodes, known_predicates):
        self.calls += 1
        out = []
        for i, ep in enumerate(episodes):
            for needle, items in self.script.items():
                if needle.lower() in ep.content.lower():
                    # `source_index` before the spread, so a scripted item can override
                    # it and exercise the provenance guard.
                    out += [{"polarity": 1, "confidence": 0.9, "source_index": i, **item}
                            for item in items]
        return out


# =============================================================================
# The shim
# =============================================================================

# --- the surface that works ---------------------------------------------------

def test_add_and_search_round_trip_in_mem0s_response_shape(api):
    added = api.add("I live in Berlin")
    assert added == {"results": [{"id": added["results"][0]["id"],
                                  "memory": "user lives in Berlin", "event": "ADD"}]}
    found = api.search("where do they live?")["results"]
    assert [r["memory"] for r in found] == ["user lives in Berlin"]
    assert 0.0 <= found[0]["score"] <= 1.0


def test_a_supersession_is_reported_as_an_add_and_a_delete(api):
    """mem0 emits one UPDATE. Memvara wrote a new claim and retired another, so it says
    so — the retired id is the one `history()` can still reach."""
    api.add("I live in Berlin")
    events = api.add("Actually I live in Lisbon")["results"]
    assert sorted(e["event"] for e in events) == ["ADD", "DELETE"]
    assert {e["memory"] for e in events} == {"user lives in Lisbon", "user lives in Berlin"}


def test_re_adding_the_same_turn_reports_none_rather_than_a_second_memory(api):
    api.add("I live in Berlin")
    assert [e["event"] for e in api.add("I live in Berlin")["results"]] == ["NONE"]
    assert len(api.get_all()["results"]) == 1


def test_get_all_returns_live_memories_and_honours_top_k(api):
    api.add("I live in Berlin")
    api.add("My name is Mira")
    assert len(api.get_all()["results"]) == 2
    assert len(api.get_all(top_k=1)["results"]) == 1


def test_get_resolves_one_id_and_stops_after_a_delete(api):
    memory_id = api.add("I live in Berlin")["results"][0]["id"]
    assert api.get(memory_id)["memory"] == "user lives in Berlin"
    with pytest.warns(Mem0DeletionWarning):
        api.delete(memory_id)
    # mem0's get() returns nothing after a delete. A shim that kept answering would make
    # the deletion look like it had not happened.
    assert api.get(memory_id) is None
    assert api.get("cl_nonexistent") is None


def test_the_row_carries_the_triple_the_string_came_from(api):
    row = api.add("I live in Berlin")["results"][0]
    full = api.get(row["id"])
    assert full["memvara"] == {
        "subject": "user", "predicate": "lives_in", "object": "Berlin",
        "memory_type": "semantic", "confidence": full["memvara"]["confidence"],
        "salience": 1.0, "valid_from": full["memvara"]["valid_from"], "valid_to": None,
    }
    assert full["user_id"] == "alice" and full["agent_id"] is None
    assert full["updated_at"] is None       # memvara never edits a live claim


def test_search_threshold_is_a_floor_and_defaults_to_none(api):
    api.add("I live in Berlin")
    assert api.search("where do they live?")["results"]
    assert api.search("where do they live?", threshold=1.01)["results"] == []


def test_explain_returns_memvaras_per_leg_reason(api):
    api.add("I live in Berlin")
    plain = api.search("where do they live?")["results"][0]
    explained = api.search("where do they live?", explain=True)["results"][0]
    assert "explanation" not in plain
    assert "recency=" in explained["explanation"]


def test_filters_narrow_the_scope(mem):
    api = Memory(mem)
    api.add("I live in Berlin", filters={"user_id": "bob"})
    assert [r["memory"] for r in api.get_all(filters={"user_id": "bob"})["results"]] \
        == ["user lives in Berlin"]
    assert api.get_all(filters={"user_id": "carol"})["results"] == []


def test_metadata_rides_along_on_the_episode(mem):
    api = Memory(mem)
    api.add("Remember the milk", metadata={"channel": "sms"}, infer=False)
    stored = api.get_all()["results"][0]
    assert stored["metadata"] == {"channel": "sms"}


def test_infer_false_stores_the_message_verbatim(api):
    """mem0's escape hatch for "store this string, do not think about it"."""
    rows = api.add("Prefers oat milk in coffee", infer=False)["results"]
    assert [r["memory"] for r in rows] == ["Prefers oat milk in coffee"]
    stored = api.get_all()["results"][0]
    # The indexed text is the sentence; the slot address lives in the subject.
    assert stored["memory"] == "Prefers oat milk in coffee"
    assert stored["memvara"]["subject"].startswith("note:")
    assert stored["memvara"]["predicate"] == NOTE_PREDICATE
    assert api.search("oat milk")["results"]


def test_infer_false_accepts_a_memory_type_and_a_transcript(api):
    """Both notes keep the type they were given, including the assistant's. A note's
    subject is a slot of type `note`, not the user, and the reconciler's rule that
    `procedural` is for the user only exempts notes: the text was typed by the owner as a
    standing instruction, and a note is not a claim about a thing."""
    api.add([{"role": "user", "content": "Always run pytest"},
             {"role": "assistant", "content": "Noted"}],
            infer=False, memory_type=MemoryType.PROCEDURAL)
    kinds = {r["memvara"]["memory_type"] for r in api.get_all()["results"]}
    assert kinds == {"procedural"}


def test_history_synthesizes_mem0s_rows_from_the_slot_timeline(api):
    api.add("I live in Berlin")
    memory_id = [e for e in api.add("Actually I live in Lisbon")["results"]
                 if e["event"] == "ADD"][0]["id"]
    rows = api.history(memory_id)
    assert [(r["old_memory"], r["new_memory"], r["event"]) for r in rows] == [
        (None, "user lives in Berlin", "ADD"),
        ("user lives in Berlin", "user lives in Lisbon", "UPDATE"),
    ]
    # mem0's memory_id is stable across the log; memvara's per-value ids are `id`.
    assert {r["memory_id"] for r in rows} == {memory_id}
    assert rows[0]["updated_at"] is None and rows[1]["updated_at"] is not None


def test_history_ends_with_a_delete_row_when_nothing_replaced_the_value(api):
    memory_id = api.add("I live in Berlin")["results"][0]["id"]
    with pytest.warns(Mem0DeletionWarning):
        api.delete(memory_id)
    last = api.history(memory_id)[-1]
    assert (last["event"], last["new_memory"], last["is_deleted"]) == ("DELETE", None, 1)


def test_history_of_an_unknown_id_is_empty_rather_than_an_error(api):
    assert api.history("cl_nope") == []


def test_delete_all_and_reset_really_erase(mem):
    api = Memory(mem)
    api.add("I live in Berlin", filters={"user_id": "bob"})
    api.add("I live in Lisbon")                    # alice, this Memvara's own scope
    erased = api.delete_all(filters={"user_id": "bob"})
    assert erased["counts"]["claims"] == 1
    # Erasure, not retirement: nothing is left to time-travel to.
    assert mem.get_all(user="bob", include_invalidated=True) == []
    assert mem.get_all() != []
    assert api.reset()["counts"]["claims"] == 1
    assert mem.get_all(include_invalidated=True) == []


def test_reset_cannot_widen_past_the_memvaras_own_scope(mem):
    """Memvara scope arguments only ever narrow, so a `Memory` over a user-bound Memvara
    resets that user — not mem0's whole-store wipe. Worth a test because the difference
    is invisible until the day someone expects a clean store and gets a half-clean one."""
    api = Memory(mem)                              # mem is bound to user="alice"
    api.add("I live in Berlin", filters={"user_id": "bob"})
    assert api.reset()["counts"]["claims"] == 0
    assert mem.get_all(user="bob") != []


def test_repr_names_the_deletion_policy(api):
    assert "on_delete=warn" in repr(api)


# --- the refusals -------------------------------------------------------------

def test_delete_retires_and_says_so_exactly_once(api):
    """The single most dangerous difference, and the reason this warning exists."""
    first = api.add("I live in Berlin")["results"][0]["id"]
    second = api.add("My name is Mira")["results"][0]["id"]
    with pytest.warns(Mem0DeletionWarning, match="GDPR"):
        api.delete(first)
    with warnings.catch_warnings():
        # Once per instance: a deletion sweep that warned per memory would be filtered
        # wholesale, taking the message with it.
        warnings.simplefilter("error")
        api.delete(second)
    # Retired, not erased — which is precisely what the warning said.
    assert api.get_all()["results"] == []
    assert len(api.memvara.get_all(include_invalidated=True)) == 2


def test_on_delete_retire_is_the_informed_silent_choice(mem):
    api = Memory(mem, on_delete="retire")
    memory_id = api.add("I live in Berlin")["results"][0]["id"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert "retired" in api.delete(memory_id)["message"]


def test_on_delete_erase_actually_erases_the_memory_and_its_source_turn(mem):
    api = Memory(mem, on_delete="erase")
    memory_id = api.add("I live in Berlin")["results"][0]["id"]
    source = mem.get(memory_id).sources[0]
    assert mem.store.get_episode(source) is not None

    with warnings.catch_warnings():
        # Nothing to warn about: this is what mem0's delete() means.
        warnings.simplefilter("error")
        assert api.delete(memory_id)["message"] == "Memory erased"

    # Gone from both readings of "deleted" — the present tense *and* the past. This is
    # the distinction the warning on the default path exists to draw.
    assert api.get(memory_id) is None
    assert mem.get(memory_id) is None
    assert mem.get_all(include_invalidated=True) == []
    assert mem.store.get_claim(memory_id) is None
    # And the sentence itself is gone, not just the claim pointing at it. A note *is*
    # its source turn; leaving the episode would erase the memory and keep the text.
    assert mem.store.get_episode(source) is None
    assert not mem.search("Berlin", include_episodes=True, include_invalidated=True)


def test_erase_leaves_other_memories_untouched(mem):
    api = Memory(mem, on_delete="erase")
    doomed = api.add("I live in Berlin")["results"][0]["id"]
    kept = api.add("My name is Mira")["results"][0]["id"]
    api.delete(doomed)
    assert api.get(kept) is not None
    assert [r["id"] for r in api.get_all()["results"]] == [kept]


EVERY_STATE = ("live", "ended", "retired")


def test_on_delete_erase_erases_every_version_of_the_memory(mem):
    """When a value changes, the earlier value stays stored as its own claim, ended
    rather than retired, and it cites its own source turn. Erasing only the current claim
    reported the memory erased while `history()` and a search of every state still
    returned the earlier text. Every version goes now, with the turns behind it. A
    neighbouring memory in the same scope is not a version of this one, and neither is
    the same memory in another user's scope, so both are left exactly as they were."""
    api = Memory(mem, on_delete="erase")
    api.add("I live in Berlin")
    current = api.add("I live in Lisbon")["results"][0]["id"]
    neighbour = api.add("My name is Mira")["results"][0]["id"]
    elsewhere = api.add("I live in Berlin", filters={"user_id": "bob"})["results"][0]["id"]
    versions = mem.history("user", "lives_in")
    assert [(c.object, c.state) for c in versions] == [("Berlin", "ended"),
                                                        ("Lisbon", "live")]
    turns = [turn for c in versions for turn in c.sources]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        response = api.delete(current)

    assert mem.history("user", "lives_in") == []
    found = [r.text for r in mem.search("Berlin Lisbon", states=EVERY_STATE,
                                        include_episodes=True)]
    assert [text for text in found if "Berlin" in text or "Lisbon" in text] == []
    assert [mem.why(c.id) for c in versions] == [None, None]
    assert [mem.store.get_episode(turn) for turn in turns] == [None, None]
    assert all(mem.prove_erased(c.id).proven for c in versions)
    # The tenant now holds the neighbour and bob's memory, each with its turn and each
    # with a vector for the claim and one for the turn.
    assert mem.stats() == {"episodes": 2, "claims": 2, "live_claims": 2,
                           "ended_claims": 0, "invalidated": 0, "embeddings": 4}
    assert api.get(neighbour)["memory"] == "user name Mira"
    assert len(mem.why(neighbour).episodes) == 1
    assert [(c.id, c.object) for c in mem.history("user", "lives_in", user="bob")] == [
        (elsewhere, "Berlin")]
    # The response lists every version it erased, oldest first.
    assert response == {"message": "Memory erased", "erased": [c.id for c in versions]}


def test_erasing_by_an_earlier_version_id_erases_the_whole_memory(mem):
    """A memory id here names one version. mem0 code that kept the id `add()` returned
    before an update still holds the earlier version's id, and mem0's ids are stable, so
    deleting by that id means deleting the memory: the versions that replaced it go too,
    up to the current value."""
    api = Memory(mem, on_delete="erase")
    first = api.add("I live in Berlin")["results"][0]["id"]
    api.add("I live in Lisbon")
    api.add("I live in Porto")
    versions = [c.id for c in mem.history("user", "lives_in")]
    assert versions[0] == first and len(versions) == 3

    response = api.delete(first)
    assert mem.history("user", "lives_in") == []
    assert api.get_all()["results"] == []
    assert response == {"message": "Memory erased", "erased": versions}


def test_on_delete_erase_leaves_another_value_of_a_many_valued_fact(mem):
    """`likes` holds many values at once, and each value is its own memory with its own
    id, although they all share one slot. So the versions of a memory are the claims its
    own updates link together, not every claim in the slot: erasing one preference must
    not erase another. A retraction is stored as a claim that points back at the value
    it ended, so it is a version of that memory and is erased with it."""
    api = Memory(mem, on_delete="erase")
    pizza = api.add("I like pizza")["results"][0]["id"]
    sushi = api.add("I like sushi")["results"][0]["id"]
    api.add("I no longer like pizza")
    retraction = mem.get(pizza).invalidated_by
    assert retraction is not None

    first = api.delete(sushi)
    assert [c.id for c in mem.history("user", "likes")] == [pizza, retraction]
    second = api.delete(pizza)
    assert mem.history("user", "likes") == []
    assert (first["erased"], second["erased"]) == ([sushi], [pizza, retraction])


def test_on_delete_erase_reaches_no_version_in_a_scope_it_cannot_read(mem):
    """A retraction written in a session ends the user-level value the session reads, so
    the session's tombstone is linked into the user-level memory's chain. It is still the
    session's memory. This `Memory` reads at the user level, which does not see into a
    session, so erasing the user-level memory leaves the session's tombstone and its turn
    as they were."""
    api = Memory(mem, on_delete="erase")
    paris = api.add("I live in Paris")["results"][0]["id"]
    api.add("I no longer live in Paris", filters={"run_id": "s1"})
    tombstone = mem.get(paris).invalidated_by
    assert tombstone is not None and mem.get(tombstone, session="s1").scope.session == "s1"

    response = api.delete(paris)
    assert [(c.id, c.polarity) for c in mem.history("user", "lives_in")] == [
        (tombstone, -1)]
    assert len(mem.why(tombstone, session="s1").episodes) == 1
    assert response["erased"] == [paris]


def test_on_delete_erase_reaches_the_broader_scope_a_session_reads(mem):
    """A `Memory` bound to a session reads the user level as well, and a retraction in
    either scope can end the value held in the other, so one memory's chain can run
    through both scopes. Every version the session can read is erased, the user-level
    ones included. A new value no longer links the two scopes, because it ends only a
    value at its own scope."""
    everyone = Memory(mem)
    session = Memory(Memvara(store=mem.store, embedder=HashingEmbedder(dim=128),
                             llm=NullLLM(), user="alice", session="s1"),
                     on_delete="erase")
    berlin = everyone.add("I live in Berlin")["results"][0]["id"]
    session.add("I no longer live in Berlin")
    chain = mem.history("user", "lives_in")
    assert [(c.object, c.polarity, c.scope.session) for c in chain] == [
        ("Berlin", 1, None), ("Berlin", -1, "s1")]

    response = session.delete(berlin)
    assert mem.history("user", "lives_in") == []
    assert response["erased"] == [c.id for c in chain]


def test_on_delete_erase_follows_the_chain_only_inside_one_slot(mem):
    """`remember(replaces=...)` can close a claim with a claim about another fact, and
    the pointer between them then crosses from one slot to another. The claim in the
    other slot is another memory, so erasing this one stops at the edge of its slot."""
    api = Memory(mem, on_delete="erase")
    berlin = api.add("I live in Berlin")["results"][0]["id"]
    elsewhere = mem.remember("user", "works_at", "Acme", replaces=berlin).added[0]
    assert mem.get(berlin).invalidated_by == elsewhere.id

    response = api.delete(berlin)
    assert mem.history("user", "lives_in") == []
    assert [c.id for c in mem.history("user", "works_at")] == [elsewhere.id]
    assert response["erased"] == [berlin]


def test_on_delete_erase_erases_every_version_through_a_hosted_deployment():
    """A hosted deployment gives the shim no store to read the memory's slot from, so
    the versions come from the deployment's `history()`, and each is erased with its
    own request."""
    with FakeV1() as fake:
        api = Memory(fake.remote(user="alice"), on_delete="erase")
        api.add("I live in Berlin")
        current = api.add("I live in Lisbon")["results"][0]["id"]
        versions = [c.id for c in fake.memvara.history("user", "lives_in", user="alice")]
        assert len(versions) == 2

        response = api.delete(current)
        assert fake.memvara.history("user", "lives_in", user="alice") == []
        assert response == {"message": "Memory erased", "erased": versions}


def test_a_hosted_erase_reaches_the_broader_scope_a_session_reads():
    """A hosted deployment's `history()` reads the slot only at the client's scope and
    the scopes beneath it, so for a client bound to a session it leaves out the versions
    held at the user level or at its agent's level. The shim finds those through
    `get()`, for a claim that replaced a version, and `why()`, for the claims a version
    replaced, because both read the broader scopes too, and erases what a local store
    would. A retraction links two scopes when it ends a value its writer reads from a
    broader scope: here the agent's retraction ends the user-level value."""
    with FakeV1() as fake:
        everyone = Memory(fake.remote(user="alice"))
        agent = Memory(fake.remote(user="alice", agent="a1"))
        session = Memory(fake.remote(user="alice", agent="a1", session="s1"),
                         on_delete="erase")
        paris = everyone.add("I live in Paris")["results"][0]["id"]
        agent.add("I no longer live in Paris")
        chain = fake.memvara.history("user", "lives_in", user="alice")
        assert [(c.object, c.polarity, c.scope.agent) for c in chain] == [
            ("Paris", 1, None), ("Paris", -1, "a1")]
        # Forward from the user-level value to the agent's tombstone, through `get()`:
        # neither is in the history the session's client reads.
        assert session.delete(paris)["erased"] == [c.id for c in chain]

        berlin = everyone.add("I live in Berlin")["results"][0]["id"]
        agent.add("I no longer live in Berlin")
        tombstone = fake.memvara.get(berlin, user="alice").invalidated_by
        # Back from the agent's tombstone to the user-level value, through `why()`.
        assert session.delete(tombstone)["erased"] == [berlin, tombstone]
        assert fake.memvara.history("user", "lives_in", user="alice") == []


def test_deleting_an_unknown_id_raises_rather_than_reporting_success(api):
    with pytest.raises(KeyError, match="cl_nope"):
        api.delete("cl_nope")


def test_delete_all_without_filters_refuses_to_guess(api):
    with pytest.raises(ValueError, match="reset"):
        api.delete_all()


def test_update_explains_why_a_claim_is_immutable(api):
    with pytest.raises(Mem0CompatError, match="immutable"):
        api.update("cl_x", "new text")


def test_from_config_points_at_the_constructor(api):
    with pytest.raises(Mem0CompatError, match="AnthropicLLM"):
        Memory.from_config({"vector_store": {"provider": "qdrant"}})


def test_rerank_is_refused_rather_than_silently_ignored(api):
    with pytest.raises(Mem0CompatError, match="explain=True"):
        api.search("anything", rerank=True)


def test_a_custom_extraction_prompt_is_refused(api):
    with pytest.raises(Mem0CompatError, match="llm="):
        api.add("hello", prompt="extract everything")


def test_memory_type_with_inference_would_override_the_registry(api):
    with pytest.raises(Mem0CompatError, match="decay half-life"):
        api.add("hello", memory_type="procedural_memory")


def test_add_and_delete_all_take_the_entity_ids_mem0_2x_takes():
    """mem0 2.x's add() requires one of user_id, agent_id and run_id, and its
    delete_all() takes them, so a mem0 call site passes them there (#359). The Memvara is
    unbound, so the id alone decides whose memory it is."""
    with Memvara(embedder=HashingEmbedder(dim=128), llm=NullLLM()) as mem:
        api = Memory(mem)
        api.add("I live in Berlin", user_id="alice")
        assert [c.object for c in mem.get_all(user="alice")] == ["Berlin"]
        assert mem.get_all(user="bob") == []
        api.delete_all(user_id="alice")
        assert mem.get_all(user="alice") == []


def test_an_entity_id_given_two_ways_must_agree(api):
    api.add("I live in Berlin", user_id="alice", filters={"user_id": "alice"})
    with pytest.raises(ValueError, match="two different scopes"):
        api.add("I live in Paris", user_id="alice", filters={"user_id": "bob"})


def test_search_and_get_all_refuse_a_top_level_entity_id_the_way_mem0_2x_does(api):
    """mem0 2.x refuses it there with ValueError, naming filters= (#359)."""
    with pytest.raises(ValueError, match=r"filters=\{'user_id': 'alice'\}"):
        api.search("hello", user_id="alice")
    with pytest.raises(ValueError, match=r"get_all\(\)"):
        api.get_all(user_id="alice")
    with pytest.raises(TypeError, match="renamed limit= to top_k="):
        api.search("hello", limit=5)


def test_reject_legacy_kwargs_is_a_no_op_when_there_is_nothing_to_reject():
    assert _reject_legacy_kwargs({}, "add") is None


@pytest.mark.parametrize("call, says", [
    (lambda m: m.add("hi", timestamp=1700000000), "does not support timestamp="),
    (lambda m: m.add("hi", expiration_date="2030-01-01"),
     "does not support expiration_date="),
    (lambda m: m.search("hi", reference_date="2024-01-01"),
     "does not support reference_date="),
    (lambda m: m.search("hi", show_expired=True), "does not support show_expired="),
    (lambda m: m.get_all(show_expired=True), "does not support show_expired="),
    (lambda m: m.update("cl_x", text="x", metadata={"a": 1}, expiration_date="2030-01-01"),
     "immutable"),
], ids=["add timestamp", "add expiration_date", "search reference_date",
        "search show_expired", "get_all show_expired", "update with every argument"])
def test_a_mem0_argument_the_shim_cannot_honour_is_refused_by_name(api, call, says):
    """The shim takes every argument mem0 2.x takes, so each one it cannot honour is
    refused with Mem0CompatError rather than a TypeError (#360), naming it."""
    with pytest.raises(Mem0CompatError, match=says):
        call(api)


def test_false_counts_as_given_except_for_show_expired(api):
    with pytest.raises(Mem0CompatError, match="reference_date="):
        api.search("hi", reference_date=False)
    assert api.search("hi", show_expired=False)["results"] == []


def test_a_none_in_filters_does_not_disagree_with_a_keyword_id():
    with Memvara(embedder=HashingEmbedder(dim=128), llm=NullLLM()) as mem:
        Memory(mem).add("I live in Berlin", user_id="alice", filters={"user_id": None})
        assert [c.object for c in mem.get_all(user="alice")] == ["Berlin"]


def test_a_refusal_gives_only_the_reasons_for_what_was_passed(api):
    with pytest.raises(Mem0CompatError) as refused:
        api.get_all(show_expired=True)
    assert "show_expired=True to show" in str(refused.value)
    assert "timestamp" not in str(refused.value) and "reference_date" not in str(refused.value)


def test_the_defaults_are_mem0s(api):
    """mem0 2.x's search() and get_all() return up to 20 results (#361)."""
    import inspect  # noqa: PLC0415

    assert inspect.signature(Memory.search).parameters["top_k"].default == 20
    assert inspect.signature(Memory.get_all).parameters["top_k"].default == 20


def test_a_row_from_get_carries_mem0s_score_key(api):
    added = api.add("I live in Berlin")["results"][0]
    assert api.get(added["id"])["score"] is None


def test_close_closes_the_memvara_and_the_with_statement_calls_it():
    closed = []

    class Backing:
        def close(self):
            closed.append(True)

    with Memory(Backing()) as shim:  # type: ignore[arg-type]
        assert isinstance(shim, Memory)
    assert closed == [True]


def test_a_metadata_filter_says_memvara_filters_by_scope(api):
    with pytest.raises(ValueError, match="tenant > user > agent > session"):
        api.get_all(filters={"category": "food"})


def test_memory_types_are_mapped_and_unknown_ones_rejected():
    assert _memory_type("procedural_memory") is MemoryType.PROCEDURAL
    assert _memory_type("episodic") is MemoryType.EPISODIC
    assert _memory_type(MemoryType.SEMANTIC) is MemoryType.SEMANTIC
    with pytest.raises(ValueError, match="unknown memory_type"):
        _memory_type("vector_memory")


def test_constructor_rejects_ambiguity_and_bad_policies(mem):
    with pytest.raises(TypeError, match="not both"):
        Memory(mem, path=":memory:")
    with pytest.raises(ValueError, match="on_delete"):
        Memory(mem, on_delete="obliterate")


def test_a_bare_memory_builds_its_own_memvara():
    api = Memory(llm=NullLLM(), embedder=HashingEmbedder(dim=64))
    try:
        assert api.add("I live in Berlin")["results"]
    finally:
        api.memvara.close()


# =============================================================================
# The importer
# =============================================================================

def write_history(path, rows, *, columns=None):
    """A stand-in for `~/.mem0/history.db`, written to mem0's documented schema."""
    columns = columns or ("id", "memory_id", "old_memory", "new_memory", "event",
                          "created_at", "updated_at", "is_deleted", "actor_id", "role")
    con = sqlite3.connect(str(path))
    con.execute(f"CREATE TABLE history ({', '.join(c + ' TEXT' for c in columns)})")
    con.executemany(
        f"INSERT INTO history VALUES ({','.join('?' * len(columns))})",
        [tuple(row.get(c) for c in columns) for row in rows],
    )
    con.commit()
    con.close()
    return str(path)


def row(id, memory_id, event, day, *, new=None, old=None, actor="alice", role="user"):
    return {"id": id, "memory_id": memory_id, "event": event,
            "created_at": at(day).isoformat(), "new_memory": new, "old_memory": old,
            "is_deleted": 1 if event == "DELETE" else 0, "actor_id": actor, "role": role}


@pytest.fixture()
def history_db(tmp_path):
    """One memory that was updated, one that was deleted, and one no-op row."""
    return write_history(tmp_path / "history.db", [
        row("h1", "m1", "ADD", 0, new="Lives in Berlin"),
        row("h2", "m2", "ADD", 1, new="Works at Acme"),
        row("h3", "m1", "UPDATE", 30, old="Lives in Berlin", new="Lives in Lisbon"),
        row("h4", "m2", "DELETE", 40, old="Works at Acme"),
        row("h5", "m3", "NONE", 41, new="Likes pizza"),
    ])


# --- phase 1: lossless, zero tokens -------------------------------------------

def test_the_log_alone_is_a_complete_export(mem, history_db):
    receipt = import_mem0(mem, history_db=history_db)
    assert (receipt.memories, receipt.claims, receipt.updated, receipt.deleted) == (2, 3, 1, 1)
    assert receipt.llm_calls == 0                     # the whole point of phase 1
    assert receipt.ignored == 1                       # the NONE row
    assert [c.text for c in mem.get_all()] == ["Lives in Lisbon"]


def test_the_import_reconstructs_a_past_mem0_could_not_query(mem, history_db):
    import_mem0(mem, history_db=history_db)
    # Day 10: both memories were believed, and Berlin was still the answer.
    assert sorted(c.text for c in mem.get_all(as_of=at(10))) \
        == ["Lives in Berlin", "Works at Acme"]
    # Day 35: mem0 had updated one and not yet deleted the other.
    assert sorted(c.text for c in mem.get_all(as_of=at(35))) \
        == ["Lives in Lisbon", "Works at Acme"]
    assert [c.text for c in mem.get_all(as_of=at(50))] == ["Lives in Lisbon"]


def test_an_update_supersedes_through_the_slot_and_records_what_replaced_it(mem, history_db):
    import_mem0(mem, history_db=history_db)
    timeline = mem.history(note_subject("m1"), NOTE_PREDICATE)
    assert [c.text for c in timeline] == ["Lives in Berlin", "Lives in Lisbon"]
    old, new = timeline
    # A mem0 UPDATE row says the memory's text changed on that date, not that the
    # previous text had been a mistake — so the old value ends and stays believed, and
    # `search(valid_at=<while Berlin held>)` over an imported store still answers Berlin.
    assert old.valid_to == at(30) and old.invalidated_at is None
    assert old.state == "ended"
    assert old.invalidated_by == new.id
    # Which is what makes the provenance chain answerable in the other direction.
    assert [c.id for c in mem.why(new.id).superseded] == [old.id]


def test_every_claim_traces_back_to_the_mem0_row_it_came_from(mem, history_db):
    import_mem0(mem, history_db=history_db)
    claim = mem.get_all()[0]
    trace = mem.why(claim.id)
    assert [e.content for e in trace.episodes] == ["Lives in Lisbon"]
    assert trace.episodes[0].meta["mem0_history_id"] == "h3"
    assert claim.meta["mem0_id"] == "m1" and claim.meta["actor_id"] == "alice"


def test_timestamps_are_mem0s_on_both_axes(mem, history_db):
    import_mem0(mem, history_db=history_db)
    claim = mem.get_all()[0]
    assert claim.recorded_at == at(30) and claim.valid_from == at(30)


def test_a_deleted_memory_is_retired_not_erased(mem, history_db):
    """The opposite of what the shim's `delete()` does, and deliberately: a DELETE row
    is a historical event, and "believed from January to February" is the record."""
    import_mem0(mem, history_db=history_db)
    retired = [c for c in mem.get_all(include_invalidated=True) if c.object == "Works at Acme"]
    assert len(retired) == 1 and retired[0].invalidated_at == at(40)


def test_the_notes_are_retrievable_by_their_own_words(mem, history_db):
    import_mem0(mem, history_db=history_db)
    assert [r.text for r in mem.search("Lisbon")] == ["Lives in Lisbon"]


def test_entity_ids_come_from_the_payloads_the_log_does_not_carry(mem, history_db):
    import_mem0(mem, history_db=history_db, memories=[
        {"id": "m1", "memory": "Lives in Berlin", "created_at": at(0).isoformat(),
         "user_id": "bob", "metadata": {"topic": "location"}},
    ])
    assert [c.text for c in mem.get_all(user="bob")] == ["Lives in Lisbon"]
    assert mem.get_all(user="bob")[0].meta["topic"] == "location"
    # m2 had no payload, so it landed in the scope the import was pointed at.
    assert {c.object for c in mem.get_all(include_invalidated=True)} \
        >= {"Works at Acme"}


def test_payloads_alone_import_when_the_log_has_been_pruned(mem):
    receipt = import_mem0(mem, memories=[
        {"id": "m9", "data": "Prefers dark roast", "created_at": at(0).isoformat()},
        {"id": "", "memory": "orphan with no id", "created_at": at(0).isoformat()},
        {"id": "m10", "created_at": at(0).isoformat()},          # no text at all
    ])
    assert receipt.memories == 1 and receipt.ignored == 1
    assert [c.text for c in mem.get_all()] == ["Prefers dark roast"]


def test_importing_nothing_at_all_is_a_caller_error(mem):
    with pytest.raises(ValueError, match="history_db"):
        import_mem0(mem)


def test_a_delete_for_a_memory_we_never_saw_is_ignored(mem, tmp_path):
    db = write_history(tmp_path / "h.db", [row("h1", "ghost", "DELETE", 1, old="gone")])
    receipt = import_mem0(mem, history_db=db)
    assert (receipt.memories, receipt.ignored, receipt.deleted) == (0, 1, 0)


def test_a_rerun_skips_what_is_already_there(mem, history_db):
    """An import that died halfway has to be re-runnable, and the store is the only
    record of how far it got."""
    import_mem0(mem, history_db=history_db)
    again = import_mem0(mem, history_db=history_db)
    assert again.skipped == 2 and again.claims == 0
    assert len(mem.get_all(include_invalidated=True)) == 3


def test_re_importing_deliberately_reinforces_rather_than_duplicating(mem, tmp_path):
    db = write_history(tmp_path / "h.db", [row("h1", "m1", "ADD", 0, new="Lives in Berlin")])
    import_mem0(mem, history_db=db)
    receipt = import_mem0(mem, history_db=db, skip_existing=False)
    assert (receipt.claims, receipt.duplicates) == (0, 1)
    assert len(mem.get_all()) == 1
    assert mem.get_all()[0].observation_count == 2


def test_a_re_imported_note_stored_in_pieces_is_deleted_where_it_is_still_open(
        mem, tmp_path):
    """#435. Berlin is stored for days 0 to 10 and 20 to 30, and nothing is live after day
    30. Importing "Berlin from day 0" again stores only the parts no claim holds: days 10
    to 20, and from day 30 on. A DELETE on day 40 must retire the piece that is still
    open. The importer used to keep the first piece, so the open one stayed live and a
    memory mem0 had deleted was still returned."""
    first = write_history(tmp_path / "first.db", [
        row("h1", "m1", "ADD", 0, new="Lives in Berlin"),
        row("h2", "m1", "UPDATE", 10, old="Lives in Berlin", new="Lives in Lisbon"),
        row("h3", "m1", "UPDATE", 20, old="Lives in Lisbon", new="Lives in Berlin"),
        row("h4", "m1", "UPDATE", 30, old="Lives in Berlin", new="Lives in Rome"),
    ])
    import_mem0(mem, history_db=first)
    (rome,) = mem.get_all()
    mem.delete(rome.id)
    again = write_history(tmp_path / "again.db", [
        row("h5", "m1", "ADD", 0, new="Lives in Berlin"),
        row("h6", "m1", "DELETE", 40, old="Lives in Berlin"),
    ])

    receipt = import_mem0(mem, history_db=again, skip_existing=False)

    assert receipt.deleted == 1
    assert mem.get_all() == []


def test_the_note_predicate_is_declared_once_and_persisted(mem, history_db):
    import_mem0(mem, history_db=history_db)
    spec = mem.registry.spec(NOTE_PREDICATE)
    assert spec.functional and not spec.learned      # declared, so not against the cap
    assert any(s.name == NOTE_PREDICATE for s in mem.store.all_specs("default"))


def test_an_existing_note_predicate_is_left_exactly_as_declared(mem):
    """Silently rewriting a caller's schema to suit an importer would be worse than the
    duplicate it prevents."""
    before = mem.registry.spec("likes")
    ensure_note_predicate(mem, "likes", "default")
    assert mem.registry.spec("likes") == before


def test_an_update_retires_nothing_when_another_writer_erased_the_value(tmp_path,
                                                                        monkeypatch):
    """The importer keeps the claim each memory's last event wrote, and an UPDATE retires
    it. Another handle on the same file can erase that claim between the two events. The
    UPDATE must read the claim again under the write lock and, finding it gone, retire
    nothing: writing back the copy the import held brought the erased text back."""
    path = str(tmp_path / "s.db")
    history = write_history(tmp_path / "history.db", [
        row("h1", "m1", "ADD", 0, new="Lives in Berlin"),
        row("h2", "m1", "UPDATE", 30, old="Lives in Berlin", new="Lives in Lisbon"),
    ])
    importer = Memvara(path, embedder=HashingEmbedder(dim=128), llm=NullLLM(), user="alice")
    other = Memvara(path, embedder=HashingEmbedder(dim=128), llm=NullLLM(), user="alice")
    erased: list[str] = []
    real_write_note = mem0_import.write_note

    def write_note(mem, claim, episode, **kwargs):
        if claim.text == "Lives in Lisbon":      # the UPDATE, before its transaction
            (berlin,) = other.get_all()
            assert other.erase(berlin.id)
            erased.append(berlin.id)
        return real_write_note(mem, claim, episode, **kwargs)

    monkeypatch.setattr(mem0_import, "write_note", write_note)
    try:
        import_mem0(importer, history_db=history)
        assert importer.store.get_claim(erased[0]) is None, "the erased claim is back"
        assert importer.store.erasure_record(erased[0]) is not None
        assert [c.text for c in importer.get_all()] == ["Lives in Lisbon"]
    finally:
        importer.close()
        other.close()


# --- reading the log ----------------------------------------------------------

def test_history_rows_are_replayed_in_the_logs_time_order_not_its_row_order(mem, tmp_path):
    db = write_history(tmp_path / "h.db", [
        row("z", "m1", "UPDATE", 30, old="Berlin", new="Lisbon"),
        row("a", "m1", "ADD", 0, new="Berlin"),
    ])
    assert [r.id for r in read_history_db(db)] == ["a", "z"]
    import_mem0(mem, history_db=db)
    assert [c.text for c in mem.get_all()] == ["Lisbon"]


def test_the_tilde_path_the_readme_documents_is_the_one_that_has_to_work(mem, tmp_path,
                                                                        monkeypatch):
    """`~/.mem0/history.db` is mem0's default location and the first line of the
    documented migration, and `sqlite3.connect` does not expand `~` — so the exact call
    in the README failed with "unable to open database file", which reads like a
    permissions problem. Asserted through `import_mem0` too: the reader working is no use
    if the entry point callers actually type does not."""
    home = tmp_path / "home"
    (home / ".mem0").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    write_history(home / ".mem0" / "history.db", [row("h1", "m1", "ADD", 0, new="Berlin")])

    assert [r.id for r in read_history_db("~/.mem0/history.db")] == ["h1"]
    assert import_mem0(mem, history_db="~/.mem0/history.db").claims == 1
    assert [c.text for c in mem.get_all()] == ["Berlin"]


def test_a_log_that_is_not_there_names_the_file_it_resolved_to(tmp_path, monkeypatch):
    """The message has to carry the absolute path, because the whole failure mode is a
    path that resolved somewhere the caller did not picture."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    with pytest.raises(sqlite3.OperationalError) as excinfo:
        read_history_db("~/.mem0/history.db")
    message = str(excinfo.value)
    assert str(tmp_path / ".mem0" / "history.db") in message
    assert "not a permissions one" in message


def test_a_log_that_is_there_and_will_not_open_says_so_instead(tmp_path):
    """Pointing at `~/.mem0` rather than the file inside it is the easy version of this,
    and telling that caller "does not exist" would send them to check a path that is
    fine."""
    with pytest.raises(sqlite3.OperationalError) as excinfo:
        read_history_db(tmp_path)
    assert f"{tmp_path} — which exists but could not be opened" in str(excinfo.value)


def test_an_update_with_no_add_row_still_imports_its_text(mem, tmp_path):
    """A pruned log must not import as an empty store."""
    db = write_history(tmp_path / "h.db", [row("h1", "m1", "UPDATE", 3, new="Lisbon")])
    assert import_mem0(mem, history_db=db).memories == 1
    assert [c.text for c in mem.get_all()] == ["Lisbon"]


def test_an_older_schema_without_actor_columns_still_reads(mem, tmp_path):
    db = write_history(
        tmp_path / "h.db",
        [{"id": "h1", "memory_id": "m1", "event": "ADD", "new_memory": "Likes tea",
          "created_at": at(0).isoformat()}],
        columns=("id", "memory_id", "old_memory", "new_memory", "event", "created_at"),
    )
    assert read_history_db(db)[0].actor_id is None
    assert import_mem0(mem, history_db=db).claims == 1


def test_timestamps_are_read_in_every_shape_mem0_writes_them():
    assert _parse_ts(at(1), where="x") == at(1)
    assert _parse_ts("2024-01-02T00:00:00Z", where="x") == at(1)
    assert _parse_ts(at(1).timestamp(), where="x") == at(1)
    # Naive means UTC, not local: an import must not shift a user's history by the
    # importing machine's timezone.
    assert _parse_ts("2024-01-02T00:00:00", where="x") == at(1)


@pytest.mark.parametrize("bad", ["yesterday", None, ["2024"]])
def test_an_unreadable_timestamp_names_the_row_it_came_from(bad):
    with pytest.raises(ValueError, match="history row h1"):
        _parse_ts(bad, where="history row h1")


# --- phase 2: opt-in, costs tokens --------------------------------------------

def test_extraction_turns_notes_into_triples_that_still_trace_to_mem0(mem, history_db):
    llm = FakeLLM({
        "Berlin": [{"subject": "user", "predicate": "lives_in", "object": "Berlin"}],
        "Lisbon": [{"subject": "user", "predicate": "lives_in", "object": "Lisbon"}],
        "Acme": [{"subject": "user", "predicate": "works_at", "object": "Acme"}],
    })
    receipt = import_mem0(mem, history_db=history_db, extract=True, llm=llm)
    assert receipt.extracted == 3 and receipt.llm_calls == 1 == llm.calls

    live = {c.predicate: c for c in mem.get_all() if c.predicate != NOTE_PREDICATE}
    # Berlin arrived first in mem0's log, so it is history here too — the extraction
    # inherited the log's timestamps rather than all landing "now" and racing.
    assert live["lives_in"].object == "Lisbon"
    assert [c.text for c in mem.get_all(as_of=at(10)) if c.predicate == "lives_in"] \
        == ["user lives in Berlin"]
    # And the structured claim still points at the episode holding the mem0 row's text.
    trace = mem.why(live["lives_in"].id)
    assert [e.content for e in trace.episodes] == ["Lives in Lisbon"]


def test_one_failed_chunk_does_not_cost_the_rest_of_the_import(mem, history_db):
    """An import is one long pass over a whole history, so a chunk that fails must not
    abort it. The turns that chunk covered are counted in `unextracted` — the notes are
    already stored, and structured claims are what they did not get.

    `TruncatedResponse` is the case that made this urgent: a model whose answer is cut
    off at its token budget used to return an empty list here and the loop moved on
    silently. It raises now, so without this guard the whole import would end mid-way
    and the caller would get an exception instead of a receipt.
    """
    class HalfFailing(FakeLLM):
        """Fails the first chunk only. `FakeLLM.extract` does its own counting, so this
        counts the call it never delegates and leaves the rest alone."""

        def extract(self, episodes, known_predicates):
            if self.calls == 0:
                self.calls += 1
                raise TruncatedResponse("phi-4-mini stopped at its 8192-token limit")
            return FakeLLM.extract(self, episodes, known_predicates)

    llm = HalfFailing({
        "Acme": [{"subject": "user", "predicate": "works_at", "object": "Acme"}]})
    receipt = import_mem0(mem, history_db=history_db, extract=True, llm=llm,
                          batch_size=1)

    assert receipt.unextracted == 1, "the failed chunk's turn, and only it"
    assert receipt.extracted == 1, "every later chunk still ran"
    assert receipt.llm_calls == llm.calls, "a call that raised was still a call"
    assert "unextracted=1" in str(receipt), "a receipt nobody reads cannot report a loss"


def test_a_clean_import_reports_nothing_unextracted(mem, history_db):
    llm = FakeLLM({"Lisbon": [
        {"subject": "user", "predicate": "lives_in", "object": "Lisbon"}]})
    assert import_mem0(mem, history_db=history_db, extract=True,
                       llm=llm).unextracted == 0


def test_extraction_defaults_to_the_memvaras_own_model(mem, history_db):
    """`NullLLM` extracts nothing, and reports that honestly rather than looking busy."""
    receipt = import_mem0(mem, history_db=history_db, extract=True)
    assert receipt.llm_calls == 1 and receipt.extracted == 0


def test_extraction_batches_rather_than_calling_once_per_note(mem, history_db):
    llm = FakeLLM()
    import_mem0(mem, history_db=history_db, extract=True, llm=llm, batch_size=2)
    assert llm.calls == 2               # three notes, two per call


def test_malformed_extractions_are_dropped_not_repaired(mem, tmp_path):
    db = write_history(tmp_path / "h.db", [row("h1", "m1", "ADD", 0, new="Lives in Oslo")])
    llm = FakeLLM({"Oslo": [
        {"subject": "user", "predicate": "lives_in", "object": "Oslo",
         "confidence": "very"},                       # unreadable score, claim is fine
        {"subject": "user", "predicate": "lives_in", "object": ""},        # no value
        {"subject": "user", "predicate": "", "object": "Oslo"},            # no slot
    ]})
    llm.script["Oslo"].append({"subject": "user", "predicate": "mood", "object": "glad",
                               "source_index": 99})   # no resolvable provenance
    receipt = import_mem0(mem, history_db=db, extract=True, llm=llm)
    assert receipt.extracted == 1
    extracted = [c for c in mem.get_all() if c.predicate == "lives_in"]
    assert [c.confidence for c in extracted] == [0.7]     # the documented fallback


def test_a_negated_extraction_retracts_instead_of_asserting(mem, tmp_path):
    db = write_history(tmp_path / "h.db", [
        row("h1", "m1", "ADD", 0, new="Works at Acme"),
        row("h2", "m2", "ADD", 1, new="Left Acme in February"),
    ])
    llm = FakeLLM({
        "Left": [{"subject": "user", "predicate": "works_at", "object": "Acme",
                  "polarity": -1}],
        "Works at": [{"subject": "user", "predicate": "works_at", "object": "Acme"}],
    })
    import_mem0(mem, history_db=db, extract=True, llm=llm)
    assert [c.object for c in mem.get_all() if c.predicate == "works_at"] == []


def test_confidence_falls_back_rather_than_dropping_the_claim():
    assert _confidence("0.4") == 0.4
    assert _confidence(2.0) == 1.0
    assert _confidence(None) == 0.7


# --- the pitch ----------------------------------------------------------------

def test_the_receipt_names_the_slots_holding_two_live_answers(mem, tmp_path):
    """The number a migration write-up is built from, and the one mem0 cannot produce."""
    db = write_history(tmp_path / "h.db", [
        row("h1", "m1", "ADD", 0, new="Uses vim"),
        row("h2", "m2", "ADD", 1, new="Uses emacs"),
        row("h3", "m3", "ADD", 2, new="Likes tea"),
        row("h4", "m4", "ADD", 3, new="Likes coffee"),
    ])
    llm = FakeLLM({
        "vim": [{"subject": "user", "predicate": "favourite_editor", "object": "vim"}],
        "emacs": [{"subject": "user", "predicate": "favourite_editor", "object": "emacs"}],
        "tea": [{"subject": "user", "predicate": "likes", "object": "tea"}],
        "coffee": [{"subject": "user", "predicate": "likes", "object": "coffee"}],
    })
    receipt = import_mem0(mem, history_db=db, extract=True, llm=llm)

    assert [(s.predicate, s.values, s.declared) for s in receipt.contested] == [
        # Undeclared first: nobody said how many editors a person has, so the store now
        # holds two live answers to one question and no rule for choosing.
        ("favourite_editor", ("emacs", "vim"), False),
        # And a declared multi-valued predicate, which is working exactly as intended.
        ("likes", ("coffee", "tea"), True),
    ]
    assert "2 slots hold more than one live value" in str(receipt)
    assert "UNDECLARED" in str(receipt.contested[0])


def test_note_slots_are_never_reported_as_contested(mem, history_db):
    """They hold one value each by construction; counting them would report the
    importer's own bookkeeping as a finding about the caller's data."""
    assert import_mem0(mem, history_db=history_db).contested == []


def test_the_receipt_reads_as_a_summary(mem, history_db):
    receipt = import_mem0(mem, history_db=history_db)
    assert str(receipt) == repr(receipt)
    assert "2 memories from 5 events" in str(receipt)
    assert isinstance(receipt, ImportReceipt)


def test_a_declared_contested_slot_says_so():
    slot = ContestedSlot("user", "likes", ("coffee", "tea"), declared=True)
    assert "declared" in str(slot) and "UNDECLARED" not in str(slot)


def test_an_import_crash_cannot_leave_a_note_slot_empty(mem):
    """Retirement and assertion are one transaction, so a crash rolls back both.

    The failure this prevents is the one that matters in a migration: the old value
    retired, the new one never written, and a memory that mem0 merely *updated* arriving
    as a memory memvara no longer holds. Separate transactions made that reachable.
    """
    from memvara.compat._notes import build_note, write_note

    ensure_note_predicate(mem, NOTE_PREDICATE, "default")
    first, ep1 = build_note(memory_id="m1", text="Likes pizza",
                            scope=mem.default_scope, ts=at(0))
    live = write_note(mem, first, ep1).added[0]

    second, ep2 = build_note(memory_id="m1", text="Likes calzone",
                             scope=mem.default_scope, ts=at(1))
    mem.writer.assert_claim = lambda *a, **kw: (_ for _ in ()).throw(
        RuntimeError("crash between the retirement and the assertion"))
    with pytest.raises(RuntimeError):
        write_note(mem, second, ep2, retire=live, at=at(1))

    # The slot still holds exactly what it held before — not nothing.
    still = mem.history(note_subject("m1"), NOTE_PREDICATE)
    assert [c.object for c in still] == ["Likes pizza"]
    assert still[0].invalidated_at is None and still[0].valid_to is None


def test_retiring_a_note_without_an_explicit_instant_still_closes_the_row(mem):
    """`write_note(retire=old)` with no `at` is a legal call, and passing that `None`
    through is not a no-op: it leaves a NULL instant on whichever axis `close` names,
    which either reads as *not retired* beside an `invalidated_by` saying otherwise, or
    **reopens** an interval that was already closed.

    The note path survived it only because the note predicate is single-valued, so the
    reconciler superseded through the same slot two statements later. Correct by
    accident is not correct.
    """
    from memvara.compat._notes import build_note, write_note

    ensure_note_predicate(mem, NOTE_PREDICATE, "default")
    first, ep1 = build_note(memory_id="m1", text="Likes pizza",
                            scope=mem.default_scope, ts=at(0))
    live = write_note(mem, first, ep1).added[0]

    second, ep2 = build_note(memory_id="m1", text="Likes calzone",
                             scope=mem.default_scope, ts=at(1))
    write_note(mem, second, ep2, retire=live)      # deliberately no `at=`

    displaced = mem.store.get_claim(live.id)
    assert displaced.valid_to == second.recorded_at, "fell back to the claim's own instant"
    assert displaced.invalidated_by == second.id
    assert not displaced.is_live(), "row says superseded and still in force"


def test_an_importer_replaying_a_correction_can_say_so(mem):
    """`write_note` carries the same `close` the facade does, because an importer is
    exactly the caller who might know that a log row was a fix rather than a change."""
    from memvara.compat._notes import build_note, write_note

    ensure_note_predicate(mem, NOTE_PREDICATE, "default")
    first, ep1 = build_note(memory_id="m3", text="Likes pizza",
                            scope=mem.default_scope, ts=at(0))
    live = write_note(mem, first, ep1).added[0]
    second, ep2 = build_note(memory_id="m3", text="Likes calzone",
                             scope=mem.default_scope, ts=at(1))
    write_note(mem, second, ep2, retire=live, at=at(1), close="retired")

    corrected = mem.store.get_claim(live.id)
    assert corrected.state == "retired" and corrected.valid_to is None


def test_retiring_a_note_without_an_instant_does_not_erase_an_existing_valid_to(mem):
    """The reopening half, isolated. A claim that already carried a `valid_to` must not
    have it cleared by a retirement that simply did not name an instant."""
    from memvara.compat._notes import build_note, write_note

    ensure_note_predicate(mem, NOTE_PREDICATE, "default")
    first, ep1 = build_note(memory_id="m2", text="Likes pizza",
                            scope=mem.default_scope, ts=at(0))
    live = write_note(mem, first, ep1).added[0]
    mem.store.set_valid_to(live.id, at(5))

    second, ep2 = build_note(memory_id="m2", text="Likes calzone",
                             scope=mem.default_scope, ts=at(1))
    write_note(mem, second, ep2, retire=live)

    assert mem.store.get_claim(live.id).valid_to is not None


@pytest.mark.parametrize("close, other", [("ended", "retired"), ("retired", "ended")])
def test_a_note_write_reports_the_turn_and_the_claim_it_closed(mem, close, other):
    """The turn is stored and the predecessor closed *outside* `assert_claim`, so its
    receipt knew about neither and this handed that receipt straight back: a call that had
    just written an episode and closed a claim reported storing no turns and closing
    nothing.

    Latent rather than live — the importer counts its own rows and reads only `added` and
    `reinforced` — which is the argument for fixing it now rather than after somebody
    trusts it. `Memvara._write_claim` is the same function one facade up and answers both
    questions; an adapter that reconciles what it wrote by receipt would have got zeroes
    from this door and the truth from that one.
    """
    from memvara.compat._notes import build_note, write_note

    ensure_note_predicate(mem, NOTE_PREDICATE, "default")
    first, ep1 = build_note(memory_id="m5", text="Likes pizza",
                            scope=mem.default_scope, ts=at(0))
    opened = write_note(mem, first, ep1)
    live = opened.added[0]
    assert opened.episode_ids == [ep1.id], "a write that closes nothing still stored a turn"
    assert opened.closed == []

    second, ep2 = build_note(memory_id="m5", text="Likes calzone",
                             scope=mem.default_scope, ts=at(1))
    receipt = write_note(mem, second, ep2, retire=live, at=at(1), close=close)

    assert receipt.episode_ids == [ep2.id]
    assert [c.id for c in receipt.closed] == [live.id]
    assert [c.id for c in getattr(receipt, close)] == [live.id]
    assert getattr(receipt, other) == [], "the other axis did not move"
    assert receipt.closed[0].state == close == mem.store.get_claim(live.id).state


def test_an_import_reports_the_tokens_it_paid_not_only_the_calls_it_made(mem, history_db):
    """Phase 2's own docstring says it "pays tokens once" — and until now it reported
    only `llm_calls`. An import is the largest single spend most callers ever make here,
    it is the migration path off mem0, and `127 calls` cannot be turned into a number
    anyone can put in a write-up. One accumulator spans the whole import."""
    class Metered(FakeLLM):
        reports_usage = True

        def extract(self, episodes, known_predicates, *, usage=None):
            if usage is not None:
                usage.add(900, 40)
            return super().extract(episodes, known_predicates)

    llm = Metered({"Berlin": [{"subject": "user", "predicate": "lives_in",
                               "object": "Berlin"}]})
    receipt = import_mem0(mem, history_db=history_db, extract=True, llm=llm,
                          batch_size=1)
    assert receipt.llm_calls == llm.calls > 1          # several batches, one total
    assert receipt.tokens_in == 900 * llm.calls
    assert receipt.tokens_out == 40 * llm.calls


def test_an_import_by_a_backend_that_cannot_report_usage_claims_no_tokens(mem, history_db):
    # 0 here means "unknown", same as everywhere else — and `FakeLLM` does not accept a
    # `usage=` keyword at all, so this also pins that it is never sent one.
    llm = FakeLLM({"Berlin": [{"subject": "user", "predicate": "lives_in",
                               "object": "Berlin"}]})
    receipt = import_mem0(mem, history_db=history_db, extract=True, llm=llm)
    assert receipt.llm_calls == 1
    assert (receipt.tokens_in, receipt.tokens_out) == (0, 0)
