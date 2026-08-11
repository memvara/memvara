"""Predicate identity under surface-form drift — the simulation that found the bug.

A red-team run of 10,000 extractions over six real-world concepts produced 41 distinct
predicates and thirteen simultaneously-live claims for "where does the user work?",
because `fact_key = hash(owner, subject, predicate)` keys on the predicate *string*.
`works_at`, `employed_by_company`, `job_employer`, `workplace` and `employer_name` are
five different slots, so cardinality never applies across them, nothing ever
contradicts anything, and `recall()` spends its prompt budget restating stale employers.

The file is organised as one simulation harness plus the unit tests for the pieces it
leans on. The simulation is the deliverable: it drives >2,000 extractions with an
extractor that varies its surface forms the way a real model does, and asserts that

  * live claims per concept stays at a small constant rather than growing with traffic,
  * `get_all()` answers with the *current* value, not a pile of stale synonyms,
  * the model is consulted once per genuinely novel surface form, ever, and
  * with `NullLLM` the deterministic pre-pass carries the whole thing for zero calls.

Everything runs offline: SQLiteStore(":memory:"), HashingEmbedder, local fakes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import pytest

from memvara import Memvara
from memvara.embed import HashingEmbedder
from memvara.llm import NullLLM
from memvara.schema import Cardinality, PredicateRegistry, PredicateSpec, Volatility
from memvara.store import SQLiteStore
from memvara.types import Episode, MemoryType, Scope

# =============================================================================
# The six concepts, and the surface forms a real extractor drifts across
# =============================================================================


@dataclass(frozen=True)
class Concept:
    """One real-world question, and every way a model has been seen to phrase it."""

    question: str
    canonical: str
    #: Forms the deterministic pre-pass must fold for free — no model, ever.
    free: tuple[str, ...]
    #: A form no morphology can reach. Exactly one model call, once, then cached.
    novel: str
    values: tuple[str, ...]


CONCEPTS: tuple[Concept, ...] = (
    Concept(
        "where does the user work?", "works_at",
        ("works_at", "employed_by_company", "job_employer", "workplace",
         "employer_name", "works_for"),
        "paycheck_source",
        ("Acme", "Globex", "Initech", "Umbrella"),
    ),
    Concept(
        "where does the user live?", "lives_in",
        ("lives_in", "resides_in", "based_in", "current_city", "city_name", "moved_to"),
        "domiciled",
        ("Berlin", "Lisbon", "Porto", "Athens"),
    ),
    Concept(
        "what is the user's job title?", "job_title",
        ("job_title", "role", "job_position", "current_role", "works_as", "job_titles"),
        "occupation",
        ("engineer", "architect", "director", "principal"),
    ),
    Concept(
        "what timezone is the user in?", "timezone",
        ("timezone", "tz", "in_timezone", "timezone_name", "current_timezone",
         "timezones"),
        "clock_offset",
        ("CET", "WET", "EET", "UTC"),
    ),
    Concept(
        "what mood is the user in?", "mood",
        ("mood", "feeling", "feels", "current_mood", "mood_is", "moods"),
        "emotional_state",
        ("calm", "tired", "elated", "focused"),
    ),
    Concept(
        "what pronouns does the user use?", "pronouns",
        ("pronouns", "uses_pronouns", "pronoun", "my_pronouns", "pronouns_are",
         "user_pronouns"),
        "preferred_pronouns",
        ("she/her", "they/them", "he/him", "xe/xem"),
    ),
)

#: 343 rounds x 6 concepts = 2,058 extractions. The claim under test is about what
#: happens as traffic accumulates, so the count has to be large enough that unbounded
#: growth would be unmistakable. 343 is also deliberately not a multiple of the value
#: cycle: the final value differs from the first, so "returns the current value" cannot
#: pass by accident on a store that simply kept whatever it saw first.
ROUNDS = 343
EXTRACTIONS = ROUNDS * len(CONCEPTS)

#: The round on which `DriftingExtractor` reaches each concept's unmergeable form —
#: every concept lists six free forms, so the seventh slot in the rotation is the novel
#: one. Tests that need an acquisition call to happen start here.
NOVEL_ROUND = 6

SCOPE = Scope("acme", "alice")


# =============================================================================
# Local fakes
# =============================================================================


class DriftingExtractor:
    """Emits one claim per turn, rotating through a concept's surface forms.

    Rotation is offset by the round so consecutive turns for one concept never reuse
    the same phrasing — which is what a real model does, and what the old fact_key
    could not survive.
    """

    name = "fake/drifting"
    is_noop = False

    def __init__(self, *, include_novel: bool, synonyms: dict[str, str] | None = None,
                 cardinality: str = "one") -> None:
        self.include_novel = include_novel
        self._synonyms = dict(synonyms or {})
        self._cardinality = cardinality
        self.extract_calls = 0
        self.resolve_calls = 0
        self.classify_calls = 0
        self.resolved_surfaces: list[str] = []
        self.seen_known_predicates: list[Sequence[str]] = []

    def surfaces(self, concept: Concept) -> tuple[str, ...]:
        return concept.free + ((concept.novel,) if self.include_novel else ())

    def extract(self, episodes: Sequence[Episode],
                known_predicates: Sequence[str]) -> list[dict[str, Any]]:
        self.extract_calls += 1
        self.seen_known_predicates.append(list(known_predicates))
        out: list[dict[str, Any]] = []
        for i, ep in enumerate(episodes):
            concept = CONCEPTS[ep.meta["concept"]]
            forms = self.surfaces(concept)
            out.append({
                "subject": "user",
                "predicate": forms[ep.meta["round"] % len(forms)],
                "object": concept.values[ep.meta["round"] % len(concept.values)],
                "polarity": 1,
                "memory_type": "semantic",
                "confidence": 0.9,
                "source_index": i,
            })
        return out

    def resolve_predicate(self, surface: str, candidates: Sequence[str]) -> dict[str, Any]:
        self.resolve_calls += 1
        self.resolved_surfaces.append(surface)
        return {
            "canonical": self._synonyms.get(surface),
            "cardinality": self._cardinality,
            "volatility": "slow",
            "memory_type": "semantic",
        }

    def classify_predicate(self, predicate: str, example: str) -> dict[str, str]:
        self.classify_calls += 1
        return {"cardinality": self._cardinality, "volatility": "slow",
                "memory_type": "semantic"}


SYNONYMS = {c.novel: c.canonical for c in CONCEPTS}


def turn(round_no: int, index: int, concept: Concept) -> Episode:
    """A turn the gate passes and the fast path deliberately cannot touch.

    No first-person subject, so `FastExtractor` emits nothing and the turn reaches
    tier 2 — which is the tier whose predicate handling is under test. The round number
    keeps the content hash and the embedding distinct, so neither the exact-duplicate
    nor the near-duplicate shortcut absorbs the write.
    """
    return Episode(
        content=(f"Update {round_no}: the record for {concept.question} was restated "
                 f"as {concept.values[round_no % len(concept.values)]} during review."),
        scope=SCOPE,
        meta={"concept": index, "round": round_no},
    )


def simulate(llm, *, rounds: int = ROUNDS, **memvara_kw):
    """Drive `rounds * 6` extractions through the public API and hand back the memory."""
    mem = Memvara(store=SQLiteStore(":memory:"), embedder=HashingEmbedder(dim=64),
                 llm=llm, tenant="acme", user="alice", **memvara_kw)
    for r in range(rounds):
        mem.add([turn(r, i, c) for i, c in enumerate(CONCEPTS)])
    return mem


def live_by_predicate(mem) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for c in mem.get_all():
        out.setdefault(c.predicate, []).append(c.object)
    return out


def expected_value(concept: Concept, rounds: int = ROUNDS) -> str:
    return concept.values[(rounds - 1) % len(concept.values)]


def stored_specs(store, tenant: str = "acme"):
    """Read persisted specs through whichever `all_specs` signature the store has."""
    try:
        return store.all_specs(tenant)
    except TypeError:                       # a store predating contract A
        return store.all_specs()


# =============================================================================
# The simulation
# =============================================================================


@pytest.fixture(scope="module")
def resolved():
    """>2,000 extractions, 42 surface forms, a model that can merge synonyms."""
    llm = DriftingExtractor(include_novel=True, synonyms=SYNONYMS)
    mem = simulate(llm)
    yield llm, mem
    mem.close()


def test_the_simulation_actually_ran_the_advertised_volume(resolved):
    llm, mem = resolved
    assert llm.extract_calls == ROUNDS
    assert EXTRACTIONS >= 2000
    assert mem.stats()["episodes"] == EXTRACTIONS


def test_surface_form_drift_does_not_multiply_predicates(resolved):
    """42 phrasings, six concepts, six slots. The old code produced 42 slots."""
    _, mem = resolved
    assert sorted(live_by_predicate(mem)) == sorted(c.canonical for c in CONCEPTS)


def test_live_claims_per_concept_stay_at_one(resolved):
    """The headline. Thirteen live employers was the bug; one is the fix."""
    _, mem = resolved
    live = live_by_predicate(mem)
    for c in CONCEPTS:
        assert len(live[c.canonical]) == 1, (
            f"{c.question} has {len(live[c.canonical])} simultaneously-live answers"
        )


def test_get_all_returns_the_current_value_not_a_pile_of_synonyms(resolved):
    _, mem = resolved
    live = live_by_predicate(mem)
    for c in CONCEPTS:
        assert live[c.canonical] == [expected_value(c)]


def test_recall_spends_its_prompt_budget_on_current_facts(resolved):
    """The user-visible symptom: four of eight slots restating a stale employer."""
    _, mem = resolved
    block = mem.recall("where does the user work?", k=8)
    stale = [v for v in CONCEPTS[0].values if v != expected_value(CONCEPTS[0])]
    assert expected_value(CONCEPTS[0]) in block
    assert not any(v in block for v in stale)


def test_the_model_is_asked_once_per_novel_surface_form_ever(resolved):
    """Acquisition is amortized: 2,046 extractions, one call per unmergeable form."""
    llm, _ = resolved
    assert llm.resolve_calls == len(CONCEPTS)
    assert sorted(llm.resolved_surfaces) == sorted(c.novel for c in CONCEPTS)
    assert llm.classify_calls == 0, "resolve_predicate replaces classify as acquisition"


def test_a_resolved_synonym_is_persisted_as_an_alias(resolved):
    """Paid for once, ever — including across processes, which means the store."""
    _, mem = resolved
    for c in CONCEPTS:
        assert mem.registry.normalize(c.novel) == c.canonical
    persisted = {s.name: s for s in stored_specs(mem.store)}
    for c in CONCEPTS:
        assert c.novel in persisted[c.canonical].aliases


def test_the_registry_barely_grew(resolved):
    """Bounded schema growth is the actual fix; the claim count follows from it."""
    _, mem = resolved
    learned = [s for s in mem.registry.all_specs() if s.learned]
    assert learned == [], "every drifted form folded onto an existing predicate"
    assert len(mem.registry) == len(PredicateRegistry())


def test_history_still_holds_every_superseded_value(resolved):
    """Bounding the live set must not delete anything — supersession, not deletion."""
    _, mem = resolved
    timeline = mem.history("user", "works_at")
    assert len(timeline) > 100
    assert timeline[-1].object == expected_value(CONCEPTS[0])
    # `ended`, not `retired`: each of those employers was true in its turn, so the
    # timeline is a hundred world events and not a hundred corrections.
    assert all(c.state == "ended" for c in timeline[:-1])


# --- the same drift, with nothing but the deterministic rules -----------------


@pytest.fixture(scope="module")
def unaided():
    """The same 2,058 extractions, restricted to forms morphology can reach.

    No `resolve_predicate` is ever reached here, so this is what the write path is
    worth before a single token is spent — the "easy majority for free" claim, measured.
    """
    llm = DriftingExtractor(include_novel=False)
    mem = simulate(llm)
    yield llm, mem
    mem.close()


def test_the_deterministic_pre_pass_carries_the_whole_run_for_free(unaided):
    """36 surface forms, six slots, and not one acquisition call to get there."""
    llm, mem = unaided
    live = live_by_predicate(mem)
    assert sorted(live) == sorted(c.canonical for c in CONCEPTS)
    for c in CONCEPTS:
        assert live[c.canonical] == [expected_value(c)]
    assert (llm.resolve_calls, llm.classify_calls) == (0, 0)


def test_extraction_stays_the_only_thing_billed(unaided):
    """`llm_calls` has to be the batched extract calls and nothing else — if schema
    acquisition were still per-write, this is where it would show up."""
    llm, mem = unaided
    receipt = mem.add([turn(ROUNDS, i, c) for i, c in enumerate(CONCEPTS)])
    assert receipt.llm_calls == 1
    assert llm.extract_calls == ROUNDS + 1


# --- and with no model at all -------------------------------------------------


def test_a_noop_backend_is_never_billed():
    """Contract C: `llm_calls` counts model consultations, not method invocations."""
    with Memvara(embedder=HashingEmbedder(dim=64), llm=NullLLM(),
                tenant="acme", user="alice") as mem:
        receipt = mem.add([turn(0, 0, CONCEPTS[0])])
        assert receipt.llm_calls == 0
        assert NullLLM.is_noop is True


def test_a_noop_backend_reports_what_it_could_not_extract():
    """The turn reached tier 2 and yielded nothing. Reporting a clean empty write
    instead is how the default configuration reads as a broken library."""
    with Memvara(embedder=HashingEmbedder(dim=64), llm=NullLLM(),
                tenant="acme", user="alice") as mem:
        receipt = mem.add([turn(0, 0, CONCEPTS[0])])
        assert receipt.unextracted == 1
        assert receipt.added == []


def test_an_unextractable_turn_is_reported_even_with_a_real_model():
    """`unextracted` is about lost content, not about the backend: a model that read
    the turn and found nothing durable in it loses exactly as much."""
    llm = DriftingExtractor(include_novel=False)
    llm.extract = lambda episodes, known: []            # type: ignore[method-assign]
    with Memvara(embedder=HashingEmbedder(dim=64), llm=llm,
                tenant="acme", user="alice") as mem:
        receipt = mem.add([turn(0, 0, CONCEPTS[0]), turn(1, 1, CONCEPTS[1])])
        assert (receipt.llm_calls, receipt.unextracted) == (1, 2)


def test_a_null_backend_resolves_nothing_rather_than_guessing():
    """With no model there is no evidence two spellings mean the same thing."""
    assert NullLLM().resolve_predicate("paycheck_source", ["works_at"])["canonical"] is None


# =============================================================================
# The deterministic pre-pass, rule by rule
# =============================================================================


@pytest.fixture()
def reg() -> PredicateRegistry:
    return PredicateRegistry()


@pytest.mark.parametrize("surface,canonical", [
    # slot-noise suffixes
    ("employer_name", "works_at"),
    ("city_name", "lives_in"),
    ("timezone_name", "timezone"),
    ("company_name", "works_at"),
    # domain-noise heads
    ("job_employer", "works_at"),
    ("job_position", "job_title"),
    ("job_titles", "job_title"),
    ("employed_by_company", "works_at"),
    # deictic particles
    ("current_city", "lives_in"),
    ("current_role", "job_title"),
    ("current_timezone", "timezone"),
    ("current_mood", "mood"),
    ("my_pronouns", "pronouns"),
    ("user_pronouns", "pronouns"),
    ("mood_is", "mood"),
    ("pronouns_are", "pronouns"),
    # plurals
    ("timezones", "timezone"),
    ("moods", "mood"),
    ("pronoun", "pronouns"),
])
def test_morphology_folds_a_surface_form_with_no_model(reg, surface, canonical):
    assert reg.normalize(surface) == canonical
    assert reg.resolve(surface).resolved


def test_the_pre_pass_runs_before_the_model_not_after(reg):
    """The model call is affordable only because it is the fallback. If morphology ran
    second, every one of these would have cost a call the first time it was seen."""
    free = [s for c in CONCEPTS for s in c.free]
    assert all(reg.resolve(s).resolved for s in free)
    assert len(free) == 36


@pytest.mark.parametrize("surface", [
    "workplace", "employer", "employed_by", "works_for", "resides_in", "moved_to",
])
def test_alias_table_still_wins_before_any_morphology(reg, surface):
    assert reg.resolve(surface).method in ("canonical", "alias")


def test_resolution_is_deterministic_given_registry_state(reg):
    """Never an embedding-similarity threshold on the write path: the same store and
    the same registry must produce the same slot on every run, forever."""
    other = PredicateRegistry()
    probes = ["employer_name", "current_city", "brand_new_thing", "workplace", "moods"]
    assert [reg.resolve(p) for p in probes] == [other.resolve(p) for p in probes]
    assert [reg.resolve(p) for p in probes] == [reg.resolve(p) for p in probes]


# --- refusing to fold is the safe answer -------------------------------------


def test_an_ambiguous_stem_refuses_to_fold(reg):
    """`works_at` and `working_on` both reduce to "work" once verb inflection is
    stripped. Merging an employer with a current task is worse than two slots, so the
    derivational tier declines and `worked` stays its own predicate."""
    assert reg.resolve("worked").resolved is False
    assert reg.normalize("worked") == "worked"
    # The strict tier keeps them apart on its own, which is why both still resolve.
    assert (reg.normalize("works_at"), reg.normalize("working")) == ("works_at",
                                                                    "working_on")


def test_two_distinct_builtins_that_share_a_stem_stay_distinct(reg):
    """`born_on` (a date) and `born_in` (a place) differ only in a particle, so the
    particle-stripped key is claimed by both and neither gets to have it."""
    assert reg.normalize("born_on") == "born_on"
    assert reg.normalize("born_in") == "born_in"
    assert reg.resolve("born").resolved is False


@pytest.mark.parametrize("surface", [
    "enjoys_hiking_at_dawn", "keyboard_layout", "collects_stamps", "drives_car",
    "prefers_editor", "some_predicate_nobody_declared",
])
def test_a_genuinely_new_predicate_is_left_alone(reg, surface):
    """Folding a distinct predicate into another is worse than leaving it separate,
    so an unresolved surface form stays itself and the model gets asked."""
    res = reg.resolve(surface)
    assert res.resolved is False
    assert res.name == surface


def test_no_two_builtin_predicates_share_a_resolution_key():
    """An integrity check on the shipped schema: if two canonical predicates collided,
    every surface form near them would silently resolve to whichever registered last."""
    reg = PredicateRegistry()
    for spec in reg.all_specs():
        assert reg.normalize(spec.name) == spec.name
        for alias in spec.aliases:
            assert reg.normalize(alias) == spec.name


# =============================================================================
# The registry cap: the backstop for when resolution is wrong
# =============================================================================


def test_learned_predicates_are_capped(reg):
    small = PredicateRegistry(max_learned=3)
    for i in range(10):
        small.learn(f"novel_predicate_{i}", Cardinality.MANY)
    assert len([s for s in small.all_specs() if s.learned]) == 3


def test_at_the_cap_a_novel_form_folds_instead_of_registering():
    """Unbounded schema growth is the root cause; the cap is what stops it even when
    resolution is wrong. Folding degrades ranking; growing without bound breaks the
    contradiction engine entirely."""
    llm = DriftingExtractor(include_novel=True, synonyms={}, cardinality="many")
    mem = simulate(llm, rounds=8, registry=PredicateRegistry(max_learned=2))
    learned = [s for s in mem.registry.all_specs() if s.learned]
    assert len(learned) <= 2
    mem.close()


def test_a_capped_registry_still_answers_every_question():
    small = PredicateRegistry(max_learned=0)
    assert small.learn("novel_thing", Cardinality.ONE).name == "novel_thing"
    assert not small.known("novel_thing"), "a capped learn must not install a spec"
    assert small.spec("novel_thing").cardinality is Cardinality.MANY


def test_nearest_prefers_token_overlap_and_is_deterministic(reg):
    assert reg.nearest("previous_employer") == "works_at"
    assert reg.nearest("zzz_unrelated_token") is None
    assert reg.nearest("previous_employer") == reg.nearest("previous_employer")


# =============================================================================
# Alias acquisition
# =============================================================================


def test_learning_an_alias_folds_the_surface_form_onto_the_canonical(reg):
    spec = reg.learn_alias("works_at", "paycheck_source")
    assert "paycheck_source" in spec.aliases
    assert reg.normalize("paycheck_source") == "works_at"
    assert reg.functional("paycheck_source")


def test_learning_an_alias_for_an_unknown_canonical_is_refused(reg):
    with pytest.raises(KeyError):
        reg.learn_alias("no_such_predicate", "whatever")


def test_an_alias_never_counts_against_the_learned_cap():
    small = PredicateRegistry(max_learned=0)
    small.learn_alias("works_at", "paycheck_source")
    assert small.normalize("paycheck_source") == "works_at"


def test_a_model_proposed_canonical_that_does_not_exist_is_ignored():
    """The model is a suggestion, not an authority: folding onto a predicate that is
    not in the registry would invent a slot nobody can look up."""
    llm = DriftingExtractor(include_novel=True,
                            synonyms={c.novel: "not_a_real_predicate" for c in CONCEPTS})
    mem = simulate(llm, rounds=8)
    assert all(not s.name == "not_a_real_predicate" for s in mem.registry.all_specs())
    assert mem.registry.known(CONCEPTS[0].novel)
    mem.close()


# =============================================================================
# Spec plumbing
# =============================================================================


def test_a_resolved_alias_survives_a_restart(tmp_path):
    """'Asked once, ever' has to hold across processes or a CLI agent re-pays daily."""
    path = str(tmp_path / "s.db")
    first = DriftingExtractor(include_novel=True, synonyms=SYNONYMS)
    mem = Memvara(path, embedder=HashingEmbedder(dim=64), llm=first,
                 tenant="acme", user="alice")
    for r in range(8):
        mem.add([turn(r, i, c) for i, c in enumerate(CONCEPTS)])
    mem.close()
    assert first.resolve_calls == len(CONCEPTS)

    second = DriftingExtractor(include_novel=True, synonyms=SYNONYMS)
    mem2 = Memvara(path, embedder=HashingEmbedder(dim=64), llm=second,
                  tenant="acme", user="alice")
    for r in range(8, 16):
        mem2.add([turn(r, i, c) for i, c in enumerate(CONCEPTS)])
    assert second.resolve_calls == 0, "the alias must be read back, not re-derived"
    mem2.close()


def test_specs_are_stored_against_the_pipeline_tenant():
    store = SQLiteStore(":memory:")
    llm = DriftingExtractor(include_novel=True, synonyms=SYNONYMS)
    mem = Memvara(store=store, embedder=HashingEmbedder(dim=64), llm=llm,
                 tenant="acme", user="alice")
    mem.add([turn(NOVEL_ROUND, 0, CONCEPTS[0])])
    assert any(CONCEPTS[0].novel in s.aliases for s in stored_specs(store))
    # Not somebody else's tenant: a global predicates table lets one tenant's
    # resolution set another tenant's contradiction behaviour and decay half-life.
    assert not any(CONCEPTS[0].novel in s.aliases for s in stored_specs(store, "other"))
    mem.close()


class _Wrapping:
    """Delegates everything to a real store except what the subclass overrides."""

    def __init__(self) -> None:
        self._inner = SQLiteStore(":memory:")

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _drive_one_acquisition(store):
    llm = DriftingExtractor(include_novel=True, synonyms=SYNONYMS)
    mem = Memvara(store=store, embedder=HashingEmbedder(dim=64), llm=llm,
                 tenant="acme", user="alice")
    mem.add([turn(NOVEL_ROUND, 0, CONCEPTS[0])])
    return mem, llm


def test_a_store_without_spec_persistence_still_writes():
    """Third-party stores predating the learned-schema surface must keep working."""
    class Bare(_Wrapping):
        put_spec = None
        all_specs = None

    mem, llm = _drive_one_acquisition(Bare())
    assert llm.resolve_calls == 1
    assert mem.get_all(), "an unpersistable spec must not cost the caller the fact"
    mem.close()


def test_a_store_predating_tenant_scoped_specs_still_persists():
    """Contract A gave `put_spec` a tenant. A store that never heard of it keeps its
    global table — the bug being fixed, but better than re-paying every restart."""
    class Legacy(_Wrapping):
        def __init__(self) -> None:
            super().__init__()
            self.saved: list[str] = []

        def put_spec(self, spec):           # the pre-contract-A signature
            self.saved.append(spec.name)

    store = Legacy()
    mem, _ = _drive_one_acquisition(store)
    assert store.saved == ["works_at"]
    assert mem.registry.normalize(CONCEPTS[0].novel) == "works_at"
    mem.close()


def test_spec_carries_the_new_alias_without_mutating_the_builtin_tuple():
    """`BUILTIN_PREDICATES` is module state shared by every registry in the process."""
    from memvara.schema import BUILTIN_PREDICATES

    reg = PredicateRegistry()
    reg.learn_alias("works_at", "paycheck_source")
    builtin = next(s for s in BUILTIN_PREDICATES if s.name == "works_at")
    assert "paycheck_source" not in builtin.aliases
    assert "paycheck_source" not in PredicateRegistry().spec("works_at").aliases


def test_a_custom_spec_set_gets_the_same_treatment():
    custom = PredicateRegistry(specs=(
        PredicateSpec("drives", Cardinality.ONE, Volatility.SLOW,
                      MemoryType.SEMANTIC, ("drives_car",)),
    ))
    assert custom.normalize("current_drives") == "drives"
    assert not custom.known("lives_in")
