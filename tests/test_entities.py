"""Entity identity: the same folding trick the predicate registry uses, on values.

The headline test is `test_simulation_reports_only_real_job_changes`. Everything above it
is the machinery that test depends on, tested in isolation so a failure says which part
broke rather than "history is wrong".

No LLM is constructed in this file. Acquisition is exercised through a stub callable so
the "one model call per novel surface form, at most" accounting is checked without a
network, exactly as `tests/test_predicates.py` does for predicates.
"""

from __future__ import annotations

import random

import pytest

from memvara import Memvara, NullLLM
from memvara.embed import HashingEmbedder
from memvara.entities import (
    DEFAULT_ENTITY_CAP,
    EntityRegistry,
    EntitySpec,
    entity_id,
    entity_key,
    split_entity_id,
)
from memvara.schema import PredicateRegistry
from memvara.store import SQLiteStore
from memvara.types import Claim, Scope
from memvara.write import Reconciler
from memvara.write.reconcile import backfill_entities

OWNER = "acme\x1falice"
OTHER = "acme\x1fbob"


# --- the deterministic fold ---------------------------------------------------

@pytest.mark.parametrize("surface, expected", [
    ("Acme", "acme"),
    ("ACME", "acme"),
    ("  Acme   Corp  ", "acme"),
    ("acme inc", "acme"),
    ("Acme, Inc.", "acme"),
    ("The Acme Corporation", "acme"),
    ("Acme GmbH", "acme"),
    ("Acme Ltd", "acme"),
    ("Acme LLC", "acme"),
    ("Acme BV", "acme"),
    ("Acme SA", "acme"),
    ("Acme Co", "acme"),
    ("COFFEE", "coffee"),
    ("Coffee", "coffee"),
    ("  coffee\n", "coffee"),
    ("Zoë", "zoe"),
    ("O'Reilly", "oreilly"),
    ("", ""),
    ("   ", ""),
    ("...", ""),
])
def test_fold_is_the_expected_string(surface, expected):
    assert entity_key(surface) == expected


def test_fold_is_idempotent():
    for surface in ("Acme, Inc.", "The Acme Corporation", "Inc", "Co", "Zoë", ""):
        assert entity_key(entity_key(surface)) == entity_key(surface)


def test_a_legal_form_alone_survives_stripping():
    # Otherwise a company genuinely called "Inc" folds to the empty string and shares an
    # identity with every other unfoldable name.
    assert entity_key("Inc") == "inc"
    assert entity_key("The") == "the"


def test_fold_keeps_genuinely_different_names_apart():
    keys = {entity_key(s) for s in ("Acme", "Acme Labs", "Sun", "Sun Microsystems")}
    assert len(keys) == 4


def test_id_round_trips():
    eid = entity_id(OWNER, "acme")
    assert split_entity_id(eid) == (OWNER, "acme")


# --- resolution tiers ---------------------------------------------------------

def test_novel_surface_resolves_to_its_own_fold():
    reg = EntityRegistry()
    res = reg.resolve(OWNER, "Acme Corp")
    assert (res.key, res.method, res.resolved) == ("acme", "novel", False)


def test_second_sighting_is_known_and_free():
    reg = EntityRegistry()
    reg.resolve(OWNER, "Acme Corp")
    res = reg.resolve(OWNER, "ACME, Inc.")
    assert (res.key, res.method, res.resolved) == ("acme", "known", True)
    assert res.canonical == "Acme Corp"        # display keeps the first form we saw


def test_resolution_can_decline_to_teach_the_registry():
    reg = EntityRegistry()
    assert reg.resolve(OWNER, "Acme", register=False).key == "acme"
    assert reg.all(OWNER) == []


def test_empty_surface_resolves_to_nothing():
    res = EntityRegistry().resolve(OWNER, "   ")
    assert (res.key, res.canonical, res.method, res.resolved) == ("", "", "empty", False)


def test_owners_do_not_share_entities():
    reg = EntityRegistry()
    reg.resolve(OWNER, "Acme")
    assert reg.known(OWNER, "Acme") and not reg.known(OTHER, "Acme")
    # Bob meeting Acme for the first time is a first time, however well Alice knows it.
    assert reg.resolve(OTHER, "Acme").method == "novel"


def test_alias_folds_a_second_fold_onto_the_first():
    reg = EntityRegistry()
    reg.resolve(OWNER, "IBM")
    reg.learn_alias(OWNER, "IBM", "Big Blue")
    res = reg.resolve(OWNER, "big blue")
    assert (res.key, res.method, res.resolved) == ("ibm", "alias", True)


def test_alias_onto_an_unknown_entity_is_refused():
    reg = EntityRegistry()
    with pytest.raises(KeyError):
        reg.learn_alias(OWNER, "IBM", "Big Blue")


def test_alias_of_a_form_that_already_folds_there_is_a_no_op():
    reg = EntityRegistry()
    reg.resolve(OWNER, "IBM")
    spec = reg.learn_alias(OWNER, "IBM", "ibm inc")
    assert spec.aliases == ()


def test_alias_is_not_recorded_twice():
    reg = EntityRegistry()
    reg.resolve(OWNER, "IBM")
    reg.learn_alias(OWNER, "IBM", "Big Blue")
    spec = reg.learn_alias(OWNER, "IBM", "BIG BLUE")
    assert spec.aliases == ("big blue",)


def test_empty_alias_surface_is_ignored():
    reg = EntityRegistry()
    reg.resolve(OWNER, "IBM")
    assert reg.learn_alias(OWNER, "IBM", "  ").aliases == ()


def test_all_lists_one_owners_entities_in_a_fixed_order():
    reg = EntityRegistry()
    for surface in ("Zeta", "Acme", "Mu"):
        reg.resolve(OWNER, surface)
    reg.resolve(OTHER, "Nu")
    assert [s.key for s in reg.all(OWNER)] == ["acme", "mu", "zeta"]


# --- the cap ------------------------------------------------------------------

def test_past_the_cap_resolution_still_works_but_nothing_is_registered():
    reg = EntityRegistry(max_entities=2)
    for surface in ("A", "B", "C"):
        reg.resolve(OWNER, surface)
    assert [s.key for s in reg.all(OWNER)] == ["a", "b"]
    # The unregistered one still gets a stable identity — the fold is a pure function,
    # so the cap costs alias capacity and nothing else.
    assert reg.resolve(OWNER, "c").key == "c"
    assert reg.resolve(OWNER, "C.").key == "c"


def test_the_cap_is_per_owner():
    reg = EntityRegistry(max_entities=1)
    reg.resolve(OWNER, "A")
    reg.resolve(OTHER, "B")
    assert [s.key for s in reg.all(OTHER)] == ["b"]


def test_default_cap_is_sane():
    assert DEFAULT_ENTITY_CAP >= 1000


# --- acquisition --------------------------------------------------------------

def test_acquisition_merges_onto_the_named_entity():
    reg = EntityRegistry()
    reg.resolve(OWNER, "IBM")
    asked: list[tuple[str, tuple[str, ...]]] = []

    def ask(surface, candidates):
        asked.append((surface, tuple(candidates)))
        return "IBM"

    assert reg.acquire(OWNER, "Big Blue", ask) is True
    assert reg.resolve(OWNER, "Big Blue").key == "ibm"
    assert asked == [("Big Blue", ("ibm",))]


def test_acquisition_is_paid_at_most_once_per_surface_form():
    reg = EntityRegistry()
    reg.resolve(OWNER, "IBM")
    calls = []

    def ask(surface, candidates):
        calls.append(surface)
        return None

    for _ in range(5):
        reg.acquire(OWNER, "Big Blue", ask)
    assert calls == ["Big Blue"]


def test_acquisition_naming_an_unknown_entity_is_discarded():
    # Indistinguishable from a hallucination, and inventing the entity it names would be
    # worse than leaving two.
    reg = EntityRegistry()
    reg.resolve(OWNER, "IBM")
    assert reg.acquire(OWNER, "Big Blue", lambda s, c: "Nonesuch") is False
    assert reg.resolve(OWNER, "Big Blue").key == "big blue"


def test_acquisition_declining_leaves_the_entity_alone():
    reg = EntityRegistry()
    reg.resolve(OWNER, "IBM")
    assert reg.acquire(OWNER, "Big Blue", lambda s, c: None) is False
    assert reg.resolve(OWNER, "Big Blue").key == "big blue"


def test_acquisition_is_not_attempted_for_an_empty_surface():
    reg = EntityRegistry()
    assert reg.acquire(OWNER, "  ", lambda s, c: pytest.fail("asked")) is False


def test_acquisition_is_not_attempted_when_nothing_exists_to_merge_onto():
    reg = EntityRegistry()
    assert reg.acquire(OWNER, "Big Blue", lambda s, c: pytest.fail("asked")) is False


def test_acquisition_survives_a_raising_backend():
    reg = EntityRegistry()
    reg.resolve(OWNER, "IBM")

    def boom(surface, candidates):
        raise RuntimeError("429")

    assert reg.acquire(OWNER, "Big Blue", boom) is False
    assert reg.resolve(OWNER, "Big Blue").key == "big blue"


def test_candidates_are_bounded_and_ordered_by_shared_words():
    reg = EntityRegistry()
    for surface in ("Acme Labs", "Acme Robotics", "Globex", "Initech"):
        reg.resolve(OWNER, surface)
    assert reg.candidates(OWNER, "Acme Systems", limit=2) == ["acme labs", "acme robotics"]


def test_candidates_fall_back_to_a_stable_order_when_nothing_overlaps():
    reg = EntityRegistry()
    for surface in ("Zeta", "Acme"):
        reg.resolve(OWNER, surface)
    assert reg.candidates(OWNER, "Nothing In Common") == ["acme", "zeta"]


# --- persistence --------------------------------------------------------------

class RecordingStore:
    """The E-1 half of the store contract, and nothing else.

    Rows are `(entity_id, canonical, aliases, tenant)`; `all_entities` hands back the
    first three for one tenant, which is the shape contract E-1 pins.
    """

    def __init__(self, rows=()):
        self.rows = list(rows)
        self.writes = 0

    def all_entities(self, tenant):
        return [(r[0], r[1], r[2]) for r in self.rows if r[3] == tenant]

    def put_entity(self, eid, canonical, aliases, tenant):
        self.writes += 1
        self.rows = [r for r in self.rows if not (r[0] == eid and r[3] == tenant)]
        self.rows.append((eid, canonical, tuple(aliases), tenant))


def test_entities_are_persisted_as_they_are_met():
    store = RecordingStore()
    reg = EntityRegistry(store)
    reg.resolve(OWNER, "Acme Corp")
    assert store.rows == [(entity_id(OWNER, "acme"), "Acme Corp", (), "acme")]
    reg.resolve(OWNER, "ACME")
    assert store.writes == 1        # a known entity costs no write


def test_aliases_are_persisted():
    store = RecordingStore()
    reg = EntityRegistry(store)
    reg.resolve(OWNER, "IBM")
    reg.learn_alias(OWNER, "IBM", "Big Blue")
    assert store.rows[-1] == (entity_id(OWNER, "ibm"), "IBM", ("big blue",), "acme")


def test_a_second_process_inherits_what_the_first_paid_for():
    store = RecordingStore()
    first = EntityRegistry(store)
    first.resolve(OWNER, "Big Blue")        # registered as an entity of its own...
    first.resolve(OWNER, "IBM")
    first.learn_alias(OWNER, "IBM", "Big Blue")

    # ...and its row survives, because contract E-1 has no delete. The alias must still
    # win, or the merge would silently come undone at the next process start.
    second = EntityRegistry(store)
    res = second.resolve(OWNER, "big blue")
    assert (res.key, res.method) == ("ibm", "alias")


def test_a_store_without_the_entity_table_still_resolves():
    reg = EntityRegistry(object())
    assert reg.resolve(OWNER, "Acme Corp").key == "acme"


def test_no_store_at_all_still_resolves():
    assert EntityRegistry().resolve(OWNER, "Acme Corp").key == "acme"


def test_a_malformed_persisted_row_is_skipped_rather_than_trusted():
    store = RecordingStore([("no-separator", "Acme", (), "acme")])
    assert EntityRegistry(store).all(OWNER) == []


def test_rows_for_other_tenants_are_not_loaded():
    store = RecordingStore([(entity_id(OWNER, "ibm"), "IBM", ("big blue",), "other")])
    assert EntityRegistry(store).resolve(OWNER, "Big Blue").key == "big blue"


def test_methods_counts_what_the_deterministic_pass_carried():
    reg = EntityRegistry()
    for surface in ("Acme", "Acme Corp", "ACME", "acme inc", "Globex"):
        reg.resolve(OWNER, surface)
    assert reg.methods == {"novel": 2, "known": 3}


# --- the claim keys -----------------------------------------------------------

SCOPE = Scope("acme", "alice")


def _claim(subject="user", predicate="works_at", obj="Acme", **kw) -> Claim:
    return Claim(subject=subject, predicate=predicate, object=obj, scope=SCOPE, **kw)


def test_spellings_of_one_object_share_a_value_key():
    keys = {_claim(obj=o).value_key
            for o in ("Acme", "Acme Corp", "acme inc", "ACME", "  Acme  ")}
    assert len(keys) == 1


def test_spellings_of_one_subject_share_a_fact_key():
    keys = {_claim(subject=s, predicate="founded_in", obj="1990").fact_key
            for s in ("Acme", "Acme Corp", "ACME")}
    assert len(keys) == 1


def test_different_objects_still_have_different_value_keys():
    assert _claim(obj="Acme").value_key != _claim(obj="Globex").value_key


def test_the_display_text_is_the_text_the_user_used():
    c = _claim(obj="Acme, Inc.")
    assert c.object == "Acme, Inc." and c.text == "user works at Acme, Inc."


def test_a_stamped_entity_overrides_the_fold():
    c = _claim(obj="Big Blue")
    plain = c.value_key
    c.meta["object_entity"] = "ibm"
    assert c.object_key == "ibm" and c.value_key != plain


def test_a_non_string_stamp_is_ignored():
    c = _claim(obj="Acme")
    c.meta["object_entity"] = 17
    assert c.object_key == "acme"


def test_an_unfoldable_object_keeps_its_own_identity():
    # "..." folds to nothing; two such objects must not become the same value.
    assert _claim(obj="...").value_key != _claim(obj="???").value_key


# --- reconciliation -----------------------------------------------------------

@pytest.fixture()
def store():
    s = SQLiteStore(":memory:")
    yield s
    s.close()


@pytest.fixture()
def rec(store) -> Reconciler:
    return Reconciler(store, PredicateRegistry())


def test_respelling_an_employer_is_a_re_observation_not_a_job_change(rec, store):
    rec.apply(_claim(obj="Acme"))
    res = rec.apply(_claim(obj="ACME Corp"))
    assert res.action == "reinforce"
    assert res.invalidated == []
    assert len(store.slot_history("acme", _claim().fact_key)) == 1


def test_a_real_job_change_still_supersedes(rec, store):
    rec.apply(_claim(obj="Acme"))
    res = rec.apply(_claim(obj="Globex"))
    assert res.action == "supersede" and len(res.invalidated) == 1


def test_case_variants_of_a_multi_valued_object_do_not_accumulate(rec, store):
    for obj in ("COFFEE", "Coffee", "coffee", "  coffee "):
        rec.apply(_claim(predicate="likes", obj=obj))
    live = store.competing_claims("acme", _claim(predicate="likes").fact_key)
    assert [c.object for c in live] == ["COFFEE"]


def test_retraction_and_deduplication_agree_on_identity(rec, store):
    rec.apply(_claim(predicate="likes", obj="Coffee"))
    res = rec.apply(_claim(predicate="likes", obj="  COFFEE  ", polarity=-1))
    assert res.action == "retract" and len(res.invalidated) == 1


def test_a_retraction_of_a_genuinely_different_value_still_hits_nothing(rec, store):
    rec.apply(_claim(predicate="likes", obj="Coffee"))
    res = rec.apply(_claim(predicate="likes", obj="Tea", polarity=-1))
    assert res.action == "noop" and res.invalidated == []


def test_one_persons_acme_is_not_anothers(rec, store):
    rec.apply(_claim(obj="Acme"))
    bob = Claim(subject="user", predicate="works_at", object="Acme Corp",
                scope=Scope("acme", "bob"))
    res = rec.apply(bob)
    assert res.action == "add" and res.invalidated == []


def test_a_deterministic_resolution_leaves_no_bookkeeping_in_meta(rec):
    # The fold cannot change, so a stamp for it would be a copy of a constant written
    # into every claim in the store — and `Claim.meta` is surfaced to callers as their
    # own metadata (see the mem0 shim), so it is not a free place to keep notes.
    c = _claim(obj="Acme, Inc.")
    rec.apply(c)
    assert c.meta == {}
    assert c.object_key == "acme"


def test_an_alias_is_stamped_because_the_alias_table_can_change(rec):
    rec.apply(_claim(obj="IBM"))
    rec.resolve_entity = lambda surface, candidates: "IBM"
    c = _claim(obj="Big Blue")
    rec.apply(c)
    assert c.meta["object_entity"] == "ibm"


def test_an_empty_object_is_not_stamped(rec):
    c = _claim(obj="", polarity=-1)
    rec.apply(c)
    assert "object_entity" not in c.meta


def test_a_stale_stamp_is_dropped_when_it_agrees_with_the_fold(rec):
    c = _claim(obj="Acme")
    c.meta["object_entity"] = "acme"
    rec.apply(c)
    assert "object_entity" not in c.meta


def test_acquisition_runs_on_the_write_path_only_when_wired(rec, store):
    rec.apply(_claim(obj="IBM"))
    asked: list[str] = []
    rec.resolve_entity = lambda surface, candidates: (asked.append(surface), "IBM")[1]
    res = rec.apply(_claim(obj="Big Blue"))
    assert asked == ["Big Blue"]
    assert res.action == "reinforce"        # merged, so it is the employer we know


def test_acquisition_is_skipped_for_a_known_entity(rec, store):
    rec.apply(_claim(obj="IBM"))
    rec.resolve_entity = lambda surface, candidates: pytest.fail("asked")
    rec.apply(_claim(obj="ibm inc"))


# --- the backfill -------------------------------------------------------------

def _legacy(store, obj: str, minutes: int) -> Claim:
    """A claim written the way a pre-entity-resolution build wrote it.

    Its stored key columns hash the raw string, which is exactly the state a store
    upgraded in place is in.
    """
    from datetime import timedelta

    from memvara.types import utcnow

    at = utcnow() - timedelta(minutes=minutes)
    c = Claim(subject="user", predicate="works_at", object=obj, scope=SCOPE,
              valid_from=at, recorded_at=at)
    store.put_claim(c)
    store._db.execute(                                          # noqa: SLF001
        "UPDATE claims SET fact_key=?, value_key=? WHERE id=?",
        (f"legacy-fact-{obj}", f"legacy-value-{obj}", c.id))
    store._db.commit()                                          # noqa: SLF001
    return c


def test_the_report_reads_as_one_line(rec, store):
    _legacy(store, "Acme", 30)
    assert repr(backfill_entities(rec, "acme")) == (
        "<RekeyReport scanned=1 written=0 merged=0 retired=0 dry-run>")
    assert repr(backfill_entities(rec, "acme", dry_run=False)) == (
        "<RekeyReport scanned=1 written=1 merged=0 retired=0>")


def test_backfill_dry_run_changes_nothing(rec, store):
    a = _legacy(store, "Acme", 30)
    b = _legacy(store, "ACME Corp", 20)
    report = backfill_entities(rec, "acme")
    assert report.dry_run and report.scanned == 2
    assert report.merged == 1 and report.written == 0
    assert store.get_claim(b.id).invalidated_at is None
    assert store.get_claim(a.id).invalidated_at is None
    # Not even the entity table: a scan is a read, and importing every value it walked
    # past would make the dry run the thing it exists to avoid being.
    assert rec.entities.all(OWNER) == []


def test_backfill_never_spends_a_model_call(rec, store):
    _legacy(store, "Acme", 30)
    _legacy(store, "Globex", 20)
    rec.resolve_entity = lambda surface, candidates: pytest.fail("asked")
    backfill_entities(rec, "acme", dry_run=False)


def test_backfill_folds_the_duplicates_it_created(rec, store):
    a = _legacy(store, "Acme", 30)
    b = _legacy(store, "ACME Corp", 20)
    report = backfill_entities(rec, "acme", dry_run=False)
    assert report.merged == 1 and report.written > 0
    assert store.get_claim(a.id).invalidated_at is None
    folded = store.get_claim(b.id)
    assert folded.invalidated_at is not None and folded.invalidated_by == a.id
    # And the slot is now reachable by the key the live code derives.
    assert len(store.competing_claims("acme", _claim().fact_key)) == 1


def test_backfill_rebuilds_the_supersession_chain_in_order(rec, store):
    first = _legacy(store, "Acme", 30)
    second = _legacy(store, "Globex", 20)
    third = _legacy(store, "Initech", 10)
    report = backfill_entities(rec, "acme", dry_run=False)
    assert report.retired == 2
    assert store.get_claim(first.id).invalidated_by == second.id
    assert store.get_claim(second.id).invalidated_by == third.id
    assert store.get_claim(third.id).invalidated_at is None


def test_backfill_handles_a_return_to_a_previous_employer(rec, store):
    """Acme, then Globex, then Acme again is three rows and two changes, not a merge
    into a claim we stopped believing two steps ago."""
    first = _legacy(store, "Acme", 30)
    middle = _legacy(store, "Globex", 20)
    again = _legacy(store, "ACME, Inc.", 10)
    report = backfill_entities(rec, "acme", dry_run=False)
    assert (report.merged, report.retired) == (0, 2)
    assert store.get_claim(first.id).invalidated_by == middle.id
    assert store.get_claim(middle.id).invalidated_by == again.id
    assert store.get_claim(again.id).invalidated_at is None


def test_backfill_records_why_history_changed(rec, store):
    _legacy(store, "Acme", 30)
    b = _legacy(store, "ACME Corp", 20)
    backfill_entities(rec, "acme", dry_run=False)
    events = store.get_claim(b.id).meta["entity_rekey"]
    assert events[0]["reason"] == "merged" and "at" in events[0]


def test_backfill_is_idempotent(rec, store):
    _legacy(store, "Acme", 30)
    _legacy(store, "ACME Corp", 20)
    _legacy(store, "Globex", 10)
    backfill_entities(rec, "acme", dry_run=False)
    again = backfill_entities(rec, "acme", dry_run=False)
    assert again.merged == 0 and again.retired == 0


def test_backfill_leaves_multi_valued_slots_accumulating(rec, store):
    for obj in ("Coffee", "Tea"):
        c = Claim(subject="user", predicate="likes", object=obj, scope=SCOPE)
        store.put_claim(c)
    report = backfill_entities(rec, "acme", dry_run=False)
    assert report.retired == 0
    assert len(store.competing_claims("acme", _claim(predicate="likes").fact_key)) == 2


def test_backfill_ignores_already_retired_claims(rec, store):
    rec.apply(_claim(obj="Acme"))
    rec.apply(_claim(obj="Globex"))
    report = backfill_entities(rec, "acme", dry_run=False)
    assert report.scanned == 2 and report.merged == 0 and report.retired == 0


# --- the simulation -----------------------------------------------------------

#: Six real employers, each spelled the way six months of extraction actually spells
#: them. Every variant in a row must fold to one identity; no variant may fold into
#: another row.
EMPLOYERS: dict[str, tuple[str, ...]] = {
    "Acme": ("Acme", "Acme Corp", "acme inc", "ACME", "Acme, Inc.", "  Acme   Corp ",
             "The Acme Corporation"),
    "Globex": ("Globex", "Globex Corporation", "GLOBEX", "globex ltd", "Globex, Ltd."),
    "Initech": ("Initech", "initech", "Initech LLC", "INITECH", "Initech, LLC"),
    "Umbrella": ("Umbrella", "Umbrella Corp", "umbrella corporation", "UMBRELLA",
                 "The Umbrella Corporation"),
    "Hooli": ("Hooli", "hooli", "Hooli Inc", "HOOLI", "Hooli, Inc."),
    "Stark": ("Stark Industries", "stark industries", "STARK INDUSTRIES",
              "Stark Industries, Inc."),
}

TIMELINE = ("Acme", "Globex", "Initech", "Umbrella", "Hooli", "Stark")

DRINKS: dict[str, tuple[str, ...]] = {
    "Coffee": ("Coffee", "coffee", "COFFEE", " coffee "),
    "Tea": ("Tea", "tea", "TEA"),
    "Mate": ("Mate", "mate", "MATE"),
}


def test_the_fold_alone_separates_the_six_employers():
    keys = {employer: {entity_key(v) for v in variants}
            for employer, variants in EMPLOYERS.items()}
    # One identity per employer...
    assert all(len(k) == 1 for k in keys.values()), keys
    # ...and six identities in total, so nothing collapsed across rows.
    assert len({next(iter(k)) for k in keys.values()}) == len(EMPLOYERS)


def test_the_fold_cannot_see_a_shortening(rec):
    """The honest limit, stated as a test rather than as a caveat in a docstring.

    "Stark" and "Stark Industries" share no fold, so they are two employers and the
    library reports a job change that did not happen. Nothing deterministic can fix
    that — a shortening is indistinguishable from a genuinely different company, and
    "Sun" is not "Sun Microsystems" in general. This is the whole of what acquisition
    is for, and the next test is the only reason it exists.
    """
    rec.apply(_claim(obj="Stark Industries"))
    assert rec.apply(_claim(obj="Stark")).action == "supersede"


def test_acquisition_closes_the_shortening(store):
    rec = Reconciler(store, PredicateRegistry())
    asked: list[str] = []

    def ask(surface, candidates):
        asked.append(surface)
        return "Stark Industries"

    rec.resolve_entity = ask
    rec.apply(_claim(obj="Stark Industries"))
    assert rec.apply(_claim(obj="Stark")).action == "reinforce"
    # And here is the bill, stated plainly: one call per *novel fold*, not one per
    # merge. The first employer is asked about too, because by then the subject "user"
    # is an entity and the pool is non-empty — and no cheaper test is available, since
    # requiring shared words would rule out "Big Blue"/"IBM", the only case acquisition
    # can solve. Predicate folds saturate at a few dozen; entity folds never saturate,
    # so this is a per-new-entity tax forever. That is why `resolve_entity` is opt-in
    # and unset by default.
    assert asked == ["Stark Industries", "Stark"]
    # What *is* amortized: every later spelling of either name is free.
    for obj in ("STARK", "stark", "Stark Industries", "stark industries, inc"):
        assert rec.apply(_claim(obj=obj)).action == "reinforce"
    assert asked == ["Stark Industries", "Stark"]


@pytest.fixture()
def mem():
    with Memvara(embedder=HashingEmbedder(dim=64), llm=NullLLM(),
                tenant="acme", user="alice") as m:
        yield m


def test_simulation_reports_only_real_job_changes(mem):
    """Several hundred writes over six employers; five job changes actually happened."""
    rng = random.Random(20260808)
    writes = 0
    for employer in TIMELINE:
        for _ in range(40):
            mem.remember("user", "works_at", rng.choice(EMPLOYERS[employer]))
            writes += 1
        for drink, variants in DRINKS.items():
            mem.remember("user", "likes", rng.choice(variants))
            writes += 1
    assert writes >= 250

    history = mem.history("user", "works_at")
    # Exactly six rows: one per employer that was ever true, in the order they were.
    assert len(history) == len(TIMELINE)
    # `_row_of` matches literal text, so this doubles as the display assertion: every
    # stored object is still a spelling the user actually used, not a canonical form.
    assert [_row_of(c.object) for c in history] == list(TIMELINE)
    # Five endings, one live claim — and *endings*, not retirements: the user really did
    # work at each of those places, so none of those rows is an error. Counting them with
    # `invalidated_at` was counting job changes as mistakes.
    assert sum(1 for c in history if c.is_live()) == 1
    assert sum(1 for c in history if c.state == "ended") == len(TIMELINE) - 1
    assert sum(1 for c in history if c.state == "retired") == 0

    # Each ending is explained by the claim that replaced it, and by nothing else.
    for older, newer in zip(history, history[1:]):
        assert older.invalidated_by == newer.id
        assert mem.why(newer.id).superseded == [older]

    # The multi-valued predicate holds one claim per real preference.
    likes = [c for c in mem.get_all() if c.predicate == "likes" and c.is_live()]
    assert len(likes) == len(DRINKS)
    assert {entity_key(c.object) for c in likes} == {entity_key(d) for d in DRINKS}


def _row_of(obj: str) -> str:
    """Which employer a stored surface form came from — matched on the literal text.

    Deliberately not via `entity_key`: this is the assertion that the *display* text
    survived resolution, so folding it here would test nothing. `Reconciler` strips
    surrounding whitespace, which is the one edit it is allowed to make.
    """
    for employer, variants in EMPLOYERS.items():
        if obj in (v.strip() for v in variants):
            return employer
    raise AssertionError(f"unrecognised surface form {obj!r}")


def test_simulation_keeps_two_users_apart(mem):
    for user in ("alice", "bob"):
        for employer in ("Acme", "Globex"):
            for variant in EMPLOYERS[employer]:
                mem.remember("user", "works_at", variant, user=user)
    for user in ("alice", "bob"):
        history = mem.history("user", "works_at", user=user)
        assert [_row_of(c.object) for c in history] == ["Acme", "Globex"]


def test_simulation_costs_no_model_calls(mem):
    rng = random.Random(7)
    calls = 0
    for employer in TIMELINE:
        for _ in range(20):
            calls += mem.remember(
                "user", "works_at", rng.choice(EMPLOYERS[employer])).llm_calls
    assert calls == 0


# --- erasure reaches the entity table -------------------------------------------

def test_purge_erases_the_verbatim_text_held_in_the_entity_table():
    """The bug this pins was the worst shape a privacy bug can take: `purge()` returned
    per-table counts as evidence of the erasure, `stats()` reported zero, and the
    entity table still held the first-seen spelling of every subject and object —
    a street address and a company name, in a live row that survives VACUUM."""
    mem = Memvara(embedder=HashingEmbedder(dim=64), llm=NullLLM(), user="alice")
    mem.remember("user", "works_at", "Grüner & Sohn Bestattungen GmbH")
    mem.remember("user", "lives_in", "14 Rue de la Paix, Paris")
    assert mem.store.all_entities("default"), "nothing to erase; test is vacuous"

    counts = mem.purge()

    assert mem.store.all_entities("default") == []
    # Counted, so the receipt evidences what it claims to.
    assert counts["entities"] >= 2
    mem.close()


def test_purging_one_user_leaves_another_users_entities_alone():
    """Entity ids are owner-scoped, so two users holding the same employer hold two
    rows. Erasing one must not take the other — over-deleting here would silently
    degrade a tenant every time one of its users exercised a deletion right."""
    mem = Memvara(embedder=HashingEmbedder(dim=64), llm=NullLLM())
    mem.remember("user", "works_at", "Acme Corp", user="alice")
    mem.remember("user", "works_at", "Acme Corp", user="bob")

    mem.purge(user="alice")

    left = [i for i, _, _ in mem.store.all_entities("default")]
    assert not any("\x1falice\x1f" in i for i in left)
    assert any("\x1fbob\x1f" in i for i in left)
    mem.close()


def test_erasing_a_claim_takes_an_entity_no_surviving_claim_cites():
    mem = Memvara(embedder=HashingEmbedder(dim=64), llm=NullLLM(), user="alice")
    written = mem.remember("user", "lives_in", "14 Rue de la Paix, Paris")

    assert mem.erase(written.added[0].id, sources=True)

    assert not any(c == "14 Rue de la Paix, Paris"
                   for _, c, _ in mem.store.all_entities("default"))
    mem.close()


def test_erasing_a_claim_keeps_an_entity_another_claim_still_cites():
    """Reference counting, not a prefix match. One entity is usually shared, and
    erasing a row out from under a live claim would lose the identity that makes its
    contradictions resolve."""
    mem = Memvara(embedder=HashingEmbedder(dim=64), llm=NullLLM(), user="alice")
    doomed = mem.remember("user", "works_at", "Acme Corp")
    mem.remember("colleague", "works_at", "Acme Corp")

    mem.erase(doomed.added[0].id, sources=True)

    assert any(c == "Acme Corp" for _, c, _ in mem.store.all_entities("default"))
    mem.close()


def test_a_session_scoped_purge_keeps_entities_the_user_still_uses():
    """A purge narrower than the owner. Deleting every entity the owner holds would be
    the easy implementation and would take rows the surviving sessions depend on."""
    mem = Memvara(embedder=HashingEmbedder(dim=64), llm=NullLLM(), user="alice")
    mem.remember("user", "works_at", "Acme Corp", session="s1")
    mem.remember("user", "lives_in", "Berlin", session="s2")

    mem.purge(user="alice", session="s1")

    canon = {c for _, c, _ in mem.store.all_entities("default")}
    assert "Berlin" in canon, "s2's entity was collateral damage"
    assert "Acme Corp" not in canon, "s1's entity outlived the purge"
    mem.close()
