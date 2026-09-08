"""Declared predicate vocabularies, and the door the MCP server did not have.

`PredicateRegistry` has accepted `specs=` since the first release, so a Python caller
could always declare a vocabulary. A client that launches a process and sets environment
variables could not, which meant every server-backed store — every plugin install — was
pinned to the 23 builtins. Anything outside them fell to the unregistered default: MANY,
so nothing superseded, and SLOW, so a fact that changed this morning still ranked as fresh
in two years. These pin the door open.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

#: Declared vocabularies are read with `tomllib`, which arrives in 3.11. On 3.10 the
#: reader refuses by design rather than pulling in a backport nobody declared, so the
#: behaviour these pin does not exist there. `TestUnsupportedInterpreter` covers what 3.10
#: does instead.
needs_toml = pytest.mark.skipif(sys.version_info < (3, 11),
                                reason="tomllib arrives in 3.11; load_specs refuses below it")

from memvara.schema import (BUILTIN_PREDICATES, Cardinality, PredicatePackError,
                            PredicateRegistry, PredicateSpec, Volatility,
                            available_packs, load_all_specs, load_specs)
from memvara.server.config import ConfigError, ServerConfig, build_memvara

#: Resolved from this file rather than the working directory, because pytest
#: is run from wherever the caller happened to be.
ROOT = Path(__file__).resolve().parent.parent


def _env(tmp_path, **extra):
    return {"MEMVARA_DB": str(tmp_path / "store.db"), "MEMVARA_TENANT": "t",
            "MEMVARA_EMBEDDER": "hashing:64", "MEMVARA_LLM": "none", **extra}


@needs_toml
class TestLoading:
    def test_the_engineering_pack_ships(self):
        assert "engineering" in available_packs()
        names = {s.name for s in load_specs("engineering")}
        assert {"git_state", "deploys_to", "rejected"} <= names

    def test_the_decisions_pack_ships_and_does_not_redeclare_engineering(self):
        """Two shipped packs declaring one predicate would make load order decide the
        store: later entries win, so `engineering,decisions` and `decisions,engineering`
        would disagree about that predicate's `memory_type` from the same data, silently.

        `rejected` and `known_defect` are in `engineering` already and are deliberately
        not redeclared here. That is the name half of the invariant; the surface-form
        half is the next test, and it is the half that fails quietly."""
        assert "decisions" in available_packs()
        decisions = {s.name for s in load_specs("decisions")}
        assert decisions == {"decided", "observed"}
        assert not decisions & {s.name for s in load_specs("engineering")}

    def test_no_two_shipped_vocabularies_claim_the_same_surface_form(self):
        """Disjoint *names* are not enough, because an alias resolves before a name does
        and collides on different machinery. `_reindex` builds `_alias` last-wins, so two
        packs declaring one alias send the same write to different predicates depending on
        the order `MEMVARA_PREDICATES` lists them in — and cardinality travels with the
        predicate, so one order supersedes the previous value and the other accumulates.
        Nothing reports it, and the names would still be disjoint.

        Written over `available_packs()` rather than over the two that ship today, so a
        third pack inherits the check instead of needing someone to remember it."""
        packs = {name: load_specs(name) for name in available_packs()}
        assert len(packs) >= 2, "with one pack there is no load order to disagree about"

        forwards = PredicateRegistry(
            BUILTIN_PREDICATES + load_all_specs(",".join(sorted(packs))))
        backwards = PredicateRegistry(
            BUILTIN_PREDICATES + load_all_specs(",".join(sorted(packs, reverse=True))))

        for specs in (BUILTIN_PREDICATES, *packs.values()):
            for spec in specs:
                for surface in (spec.name, *spec.aliases):
                    assert forwards.normalize(surface) == spec.name, surface
                    assert backwards.normalize(surface) == spec.name, surface

    def test_a_decision_accumulates_where_a_fact_would_supersede(self):
        """The whole shape of this family. A project makes many decisions and a later one
        does not make an earlier one untrue — it stopped being *current*, which is what
        `memory_end` says. `one` here would make every new decision silently end the last,
        which is the failure the cardinality column exists to prevent."""
        specs = {s.name: s for s in load_specs("decisions")}
        assert specs["decided"].cardinality is Cardinality.MANY
        assert specs["decided"].volatility is Volatility.STATIC, (
            "a decision made in March was made in March for ever")
        assert specs["observed"].volatility is Volatility.SLOW, (
            "an observation is a reading of a world that moves, and ages")

    def test_declared_specs_are_not_learned(self):
        # The distinction is load-bearing: `Memvara` refuses to let a persisted *learned*
        # spec overwrite a declared one, and that is what lets a pack correct a store.
        assert all(not s.learned for s in load_specs("engineering"))

    def test_later_entries_win(self, tmp_path):
        override = tmp_path / "ours.toml"
        override.write_text('[[predicate]]\nname="git_state"\ncardinality="many"\n'
                            'volatility="static"\n', encoding="utf-8")
        specs = {s.name: s for s in load_all_specs(f"engineering,{override}")}
        assert specs["git_state"].cardinality is Cardinality.MANY

    @pytest.mark.parametrize("body, fragment", [
        ('[[predicate]]\nname="x"\nvolatility="fast"\n', "has no cardinality"),
        ('[[predicate]]\nname="x"\ncardinality="sometimes"\nvolatility="fast"\n',
         "not one of"),
        ('[[predicate]]\nname="x"\ncardinality="one"\nvolatility="fast"\n'
         '[[predicate]]\nname="x"\ncardinality="many"\nvolatility="slow"\n',
         "more than once"),
        ('name="x"\n', "declares no predicates"),
        ('not toml {{{\n', "not valid TOML"),
    ])
    def test_every_malformed_file_names_its_own_fix(self, tmp_path, body, fragment):
        """Raised, never skipped. A vocabulary that half-loads is worse than one that does
        not load at all: the predicates that made it through supersede and the ones that
        did not accumulate, and nothing in the store records which is which."""
        path = tmp_path / "p.toml"
        path.write_text(body, encoding="utf-8")
        with pytest.raises(PredicatePackError, match=fragment):
            load_specs(str(path))

    def test_a_misspelled_pack_name_lists_the_real_ones(self):
        with pytest.raises(PredicatePackError, match="engineering"):
            load_specs("enginering")


@needs_toml
class TestServerWiring:
    def test_a_typo_is_a_startup_error_not_a_first_write_surprise(self, tmp_path):
        # By the first write the process has already accepted facts into the very slots
        # the pack was meant to shape.
        with pytest.raises(ConfigError, match="MEMVARA_PREDICATES"):
            ServerConfig.from_env(_env(tmp_path, MEMVARA_PREDICATES="nope"))

    def test_unset_leaves_the_builtins_alone(self, tmp_path):
        config = ServerConfig.from_env(_env(tmp_path))
        assert config.predicates == ""
        memory = build_memvara(config)
        assert not memory.registry.known("git_state")

    def test_a_declared_predicate_supersedes(self, tmp_path):
        memory = build_memvara(
            ServerConfig.from_env(_env(tmp_path, MEMVARA_PREDICATES="engineering")))
        memory.remember("user", "git_state", "8 ahead")
        memory.remember("user", "git_state", "0 ahead")
        recalled = memory.recall("git state", k=5)
        assert "0 ahead" in recalled and "8 ahead" not in recalled

    def test_a_declared_predicate_carries_its_half_life(self, tmp_path):
        """The half that has no accumulation note.

        A wrong cardinality announces itself on the receipt the first time two values
        land in one slot. A wrong volatility produces no event at all — it mis-ranks every
        recall, silently, for as long as the default says the fact is fresh.
        """
        memory = build_memvara(
            ServerConfig.from_env(_env(tmp_path, MEMVARA_PREDICATES="engineering")))
        assert memory.registry.spec("git_state").volatility is Volatility.FAST
        assert memory.registry.spec("git_state").half_life_days == 7.0

    def test_declaring_many_keeps_values_accumulating(self, tmp_path):
        """`rejected` is declared MANY on purpose. Two live values there are correct — a
        project rejects many things — and declaring it is what stops the accumulation
        note firing on a write that did exactly the right thing."""
        memory = build_memvara(
            ServerConfig.from_env(_env(tmp_path, MEMVARA_PREDICATES="engineering")))
        memory.remember("agent-memory", "rejected", "auto as the embedder default")
        memory.remember("agent-memory", "rejected", "blaming the code blocks")
        recalled = memory.recall("what did agent-memory reject", k=5)
        assert "embedder default" in recalled and "code blocks" in recalled


@needs_toml
class TestDeclarationOutranksAGuess:
    def test_a_pack_corrects_a_store_that_already_guessed(self, tmp_path):
        """The migration path, and the reason the guard in `Memvara.__init__` exists.

        Rehydration runs after construction, so without it the guess a previous process
        persisted — usually the MANY default fossilised by an offline extractor — would
        overwrite the declaration, and the correction would silently do nothing on exactly
        the stores that needed it.
        """
        env = _env(tmp_path)
        memory = build_memvara(ServerConfig.from_env(env))
        memory.store.put_spec(
            PredicateSpec(name="git_state", cardinality=Cardinality.MANY,
                          volatility=Volatility.SLOW, learned=True), "t")
        memory.close()

        reopened = build_memvara(
            ServerConfig.from_env({**env, "MEMVARA_PREDICATES": "engineering"}))
        spec = reopened.registry.spec("git_state")
        assert spec.cardinality is Cardinality.ONE
        assert not spec.learned

        # Forward-only: it changes what supersedes on the next write, and retires nothing
        # that is already stored.
        reopened.remember("user", "git_state", "A")
        reopened.remember("user", "git_state", "B")
        assert "B" in reopened.recall("git state", k=5)

    def test_a_learned_spec_still_rehydrates_when_nothing_declares_it(self, tmp_path):
        """The guard must not cost the amortization it sits next to: a predicate a model
        was paid to classify is still restored at open."""
        env = _env(tmp_path)
        memory = build_memvara(ServerConfig.from_env(env))
        memory.store.put_spec(
            PredicateSpec(name="ships_with", cardinality=Cardinality.ONE,
                          volatility=Volatility.SLOW, learned=True), "t")
        memory.close()

        reopened = build_memvara(ServerConfig.from_env(env))
        assert reopened.registry.spec("ships_with").cardinality is Cardinality.ONE

    def test_registry_reports_which_kind_it_holds(self):
        registry = PredicateRegistry(specs=BUILTIN_PREDICATES + load_specs("engineering"))
        assert registry.spec_is_declared("lives_in")
        assert registry.spec_is_declared("git_state")
        assert not registry.spec_is_declared("never_heard_of_it")


@needs_toml
class TestUnreadableSources:
    """The error paths, which are the ones a 95% gate would let through.

    Each names a way a vocabulary can fail to load on someone else's machine, where the
    difference between a raised error and a silent skip is whether their predicates
    supersede or accumulate.
    """

    def test_an_empty_entry_is_rejected(self):
        # `load_all_specs` skips blanks so `engineering,` is not an error, but the
        # single-source entry point is public and must not treat "" as "the builtins".
        with pytest.raises(PredicatePackError, match="empty entry"):
            load_specs("")

    def test_a_missing_path_says_which_path(self, tmp_path):
        # re.escape because a Windows tmp_path is full of backslashes, and `match=` is a
        # regex: without it this fails as "incomplete escape \\U" rather than as a
        # missing file.
        missing = tmp_path / "nowhere.toml"
        with pytest.raises(PredicatePackError, match=re.escape(str(missing))):
            load_specs(str(missing))

    def test_an_unreadable_file_is_reported_not_skipped(self, tmp_path, monkeypatch):
        """A permission bit, a vanished mount, a file replaced mid-read.

        Provoked by monkeypatching rather than by `chmod 000`, which does nothing when
        the suite happens to run as root and would make this pass without exercising
        anything.
        """
        path = tmp_path / "p.toml"
        path.write_text('[[predicate]]\nname="x"\ncardinality="one"\nvolatility="fast"\n',
                        encoding="utf-8")

        def refuse(self, *args, **kwargs):
            raise OSError("permission denied")

        monkeypatch.setattr(type(path), "open", refuse)
        with pytest.raises(PredicatePackError, match="could not be read"):
            load_specs(str(path))

    def test_a_predicate_entry_that_is_not_a_table(self, tmp_path):
        path = tmp_path / "p.toml"
        path.write_text('predicate = ["just a string"]\n', encoding="utf-8")
        with pytest.raises(PredicatePackError, match="not a table"):
            load_specs(str(path))

    def test_a_predicate_entry_with_no_name(self, tmp_path):
        path = tmp_path / "p.toml"
        path.write_text('[[predicate]]\ncardinality="one"\nvolatility="fast"\n',
                        encoding="utf-8")
        with pytest.raises(PredicatePackError, match="no `name`"):
            load_specs(str(path))

    def test_an_unreadable_packs_directory_lists_nothing(self, monkeypatch):
        """`available_packs` is called while building an error message for a mistyped
        pack name. If it raised there, a typo would surface as an OSError from inside
        the error handler instead of as the advice it was assembling."""
        import memvara.schema as schema

        def explode(self, pattern):
            raise OSError("packs directory is unreadable")

        monkeypatch.setattr(type(schema.PACKS_DIR), "glob", explode)
        assert schema.available_packs() == []
        with pytest.raises(PredicatePackError, match="none are installed"):
            load_specs("enginering")


@pytest.mark.skipif(sys.version_info >= (3, 11), reason="3.10 only")
class TestUnsupportedInterpreter:
    def test_it_refuses_and_names_the_reason(self):
        """The refusal is the feature on 3.10, not a crash.

        A `tomli` fallback would make this work at the price of a runtime dependency the
        package does not declare, and "numpy and nothing else" is pinned by a test. So
        one optional feature is unavailable on one interpreter, and says so.
        """
        with pytest.raises(PredicatePackError, match="Python 3.11"):
            load_specs("engineering")

    def test_everything_else_still_imports(self):
        # The point of the lazy reader: a module-level import would have taken the whole
        # suite down on this interpreter, not just this feature.
        import memvara.schema  # noqa: F401

        assert PredicateRegistry()


# --- the events pack ----------------------------------------------------------

@needs_toml
def test_the_events_pack_declares_event_predicates() -> None:
    names = {s.name for s in load_specs("events")}
    assert {"ran", "bought", "visited", "attended", "spent"} <= names


@needs_toml
def test_every_events_predicate_is_multi_valued() -> None:
    """The one rule this pack cannot get wrong.

    `Cardinality.ONE` makes a new value retire the old one. "I visited London last year"
    and "I visited Paris this year" are both still true, so declaring either predicate
    `one` would make the second silently retire the first — and `reconcile._victims`
    collects supersession candidates *only* for functional predicates, which makes `many`
    here the thing that stops two events competing at all. No temporal ordering rule can
    rescue a predicate declared wrongly.

    Asserted rather than left to the file's comment, because the failure is silent and
    looks exactly like ordinary forgetting.
    """
    specs = load_specs("events")
    assert specs, "the pack declared nothing; this test would pass vacuously"
    offenders = [s.name for s in specs if s.cardinality is not Cardinality.MANY]
    assert offenders == [], f"event predicates must be many-valued: {offenders}"


@needs_toml
def test_the_events_pack_declares_no_quantity_predicates() -> None:
    """A quantity rides on `amount`/`unit`, which every claim carries whatever its
    predicate. A `distance` or `cost` predicate would be a second way to say the same
    thing, and the two would drift."""
    names = {s.name for s in load_specs("events")}
    assert not ({"distance", "duration", "cost", "weight", "count"} & names)


# --- the graph declaration ------------------------------------------------------------
#
# A vocabulary is read once, at startup, by nobody. That makes every way for a pack to
# look declarative and do nothing a silent failure, and it is why the loader refuses more
# than it used to. These pin each refusal, and the classification rule the whole design
# rests on: an undeclared predicate takes values, and a value carries no edge.


def _pack(tmp_path, body: str):
    path = tmp_path / "graph.toml"
    path.write_text(body, encoding="utf-8")
    return str(path)


_TRAVERSABLE = """
[[predicate]]
name = "depends_on"
cardinality = "many"
volatility = "slow"
subject_type = ["project", "software"]
object_type = ["software", "service"]
graph = true
inverse = "depended_on_by"
inverse_cardinality = "many"
traversal_cost = 0.5
"""


class TestGraphDeclarationDefaults:
    def test_a_spec_declares_no_graph_behaviour_unless_asked(self):
        """The defaults are the classification rule, not merely empty values.

        `objects_are_entities` is False here, which is what makes an undeclared predicate
        take values. Connectivity is opt-in, and this is the line that makes it so.
        """
        spec = PredicateSpec(name="anything")
        assert (spec.subject_type, spec.object_type) == ((), ())
        assert spec.graph is False
        assert spec.inverse is None and spec.inverse_cardinality is None
        assert spec.traversal_cost == 1.0
        assert spec.objects_are_entities is False

    def test_every_builtin_takes_values(self):
        """The 23 builtins predate the graph and declare nothing about it, so none of
        them is traversable. Asserted because the opposite would be invisible: a builtin
        that quietly resolved as entity-valued would put edges in every store on earth."""
        assert not [s.name for s in BUILTIN_PREDICATES if s.objects_are_entities]
        assert not [s.name for s in BUILTIN_PREDICATES if s.graph]

    def test_a_mixed_object_type_resolves_to_values(self):
        """`prefers` legitimately holds `postgresql` and `plain` alike, so its
        declaration cannot decide per claim. The undecidable case takes the safe
        direction: connectivity lost is recoverable by declaring more precisely, a false
        join is not."""
        spec = PredicateSpec(name="prefers", object_type=("software", "value"))
        assert spec.objects_are_entities is False


@needs_toml
class TestGraphDeclarationLoading:
    def test_a_pack_can_declare_every_graph_field(self, tmp_path):
        spec, = load_specs(_pack(tmp_path, _TRAVERSABLE))
        assert spec.subject_type == ("project", "software")
        assert spec.object_type == ("software", "service")
        assert spec.graph is True
        assert spec.inverse == "depended_on_by"
        assert spec.inverse_cardinality is Cardinality.MANY
        assert spec.traversal_cost == 0.5
        assert spec.objects_are_entities is True
        assert spec.learned is False

    def test_the_shipped_packs_still_load(self):
        """They predate the graph fields and declare none of them. A loader that had made
        any of the new keys required would fail here rather than in a deployment."""
        for name in available_packs():
            specs = load_specs(name)
            assert specs
            assert not any(s.graph for s in specs)


@needs_toml
class TestGraphDeclarationRefusals:
    """Each of these is a pack that would otherwise load and do nothing."""

    def test_graph_without_object_type_is_refused(self, tmp_path):
        body = '[[predicate]]\nname="x"\ncardinality="many"\nvolatility="slow"\ngraph=true\n'
        with pytest.raises(PredicatePackError, match="no object_type"):
            load_specs(_pack(tmp_path, body))

    def test_graph_with_a_value_object_is_refused(self, tmp_path):
        """A scalar is not a thing to walk to, so `graph` and a value object type cannot
        both be true. Left to resolve silently, this is the false-join failure the whole
        classification rule exists to prevent."""
        body = ('[[predicate]]\nname="x"\ncardinality="many"\nvolatility="slow"\n'
                'graph=true\nobject_type=["software","value"]\n')
        with pytest.raises(PredicatePackError, match="cannot both be true"):
            load_specs(_pack(tmp_path, body))

    def test_a_non_boolean_graph_is_refused(self, tmp_path):
        body = ('[[predicate]]\nname="x"\ncardinality="many"\nvolatility="slow"\n'
                'graph="yes"\n')
        with pytest.raises(PredicatePackError, match="not a boolean"):
            load_specs(_pack(tmp_path, body))

    def test_an_inverse_without_its_cardinality_is_refused(self, tmp_path):
        """The two sides are not symmetric — `owned_by` holds one value and `owns` holds
        many — so a walk that assumed the forward cardinality would treat true facts as
        competing answers to one question and end all but the last."""
        body = ('[[predicate]]\nname="owned_by"\ncardinality="one"\nvolatility="slow"\n'
                'inverse="owns"\n')
        with pytest.raises(PredicatePackError, match="without inverse_cardinality"):
            load_specs(_pack(tmp_path, body))

    def test_an_inverse_cardinality_without_an_inverse_is_refused(self, tmp_path):
        body = ('[[predicate]]\nname="x"\ncardinality="one"\nvolatility="slow"\n'
                'inverse_cardinality="many"\n')
        with pytest.raises(PredicatePackError, match="without an inverse"):
            load_specs(_pack(tmp_path, body))

    def test_an_empty_inverse_is_read_as_no_inverse(self, tmp_path):
        body = ('[[predicate]]\nname="x"\ncardinality="one"\nvolatility="slow"\n'
                'inverse="   "\n')
        spec, = load_specs(_pack(tmp_path, body))
        assert spec.inverse is None

    @pytest.mark.parametrize("value", ['"heavy"', "true"])
    def test_a_non_numeric_traversal_cost_is_refused(self, tmp_path, value):
        """`true` is included because a bool is an int in Python, so a naive numeric check
        would accept `traversal_cost = true` and store an edge weight of 1."""
        body = ('[[predicate]]\nname="x"\ncardinality="many"\nvolatility="slow"\n'
                f'traversal_cost={value}\n')
        with pytest.raises(PredicatePackError, match="not a\n?\\s*number"):
            load_specs(_pack(tmp_path, body))

    @pytest.mark.parametrize("value", ["0", "-1.5"])
    def test_a_non_positive_traversal_cost_is_refused(self, tmp_path, value):
        body = ('[[predicate]]\nname="x"\ncardinality="many"\nvolatility="slow"\n'
                f'traversal_cost={value}\n')
        with pytest.raises(PredicatePackError, match="above zero"):
            load_specs(_pack(tmp_path, body))

    @pytest.mark.parametrize("key", ["object_type", "subject_type", "aliases",
                                     "supersedes"])
    def test_a_bare_string_where_a_list_belongs_is_refused(self, tmp_path, key):
        """Wrapping it would be kinder for exactly one release, until somebody wrote
        `aliases = "a, b"` and got one alias with a comma in it."""
        body = ('[[predicate]]\nname="x"\ncardinality="many"\nvolatility="slow"\n'
                f'{key}="software"\n')
        with pytest.raises(PredicatePackError, match="takes a list of strings"):
            load_specs(_pack(tmp_path, body))

    def test_an_unrecognised_key_is_refused(self, tmp_path):
        """The refusal this set exists for. `graph_traversable = true` is the plausible
        typo, and ignoring it would leave the predicate non-traversable, the store with no
        edges, and nothing anywhere saying why."""
        body = ('[[predicate]]\nname="x"\ncardinality="many"\nvolatility="slow"\n'
                'graph_traversable=true\n')
        with pytest.raises(PredicatePackError, match="graph_traversable"):
            load_specs(_pack(tmp_path, body))


# --- overriding a builtin, and the benchmark vocabulary -------------------------------


@needs_toml
def test_declaring_a_builtin_name_replaces_it_and_drops_its_aliases(tmp_path):
    """The trap that cost a measurement to find, pinned so it cannot cost another.

    A declared spec *replaces* the builtin of the same name rather than extending it, so an
    override that omits `aliases` silently discards them. Declaring a bare `born_on` to give
    it an `object_type` therefore stops `date_of_birth` folding onto it, and the relation it
    was declared for goes from resolved to unknown *because* it was declared.

    Nothing warns. The failure is a declaration that looks present in the file, resolves to
    nothing at runtime, and shows up only as connectivity that never appears.
    """
    from memvara.schema import BUILTIN_PREDICATES

    builtin, = [s for s in BUILTIN_PREDICATES if s.name == "born_on"]
    assert "date_of_birth" in builtin.aliases, "fixture assumes the builtin carries it"

    bare = tmp_path / "bare.toml"
    bare.write_text('[[predicate]]\nname="born_on"\ncardinality="many"\n'
                    'volatility="static"\nobject_type=["value"]\n', encoding="utf-8")
    registry = PredicateRegistry(BUILTIN_PREDICATES + load_specs(str(bare)))
    assert registry.normalize("date_of_birth") != "born_on", (
        "if this now folds, the replace-not-extend behaviour changed and the twowiki pack's "
        "repeated alias lists are no longer load-bearing")

    kept = tmp_path / "kept.toml"
    kept.write_text('[[predicate]]\nname="born_on"\ncardinality="many"\n'
                    'volatility="static"\nobject_type=["value"]\n'
                    'aliases=["birthday","date_of_birth","dob"]\n', encoding="utf-8")
    restored = PredicateRegistry(BUILTIN_PREDICATES + load_specs(str(kept)))
    assert restored.normalize("date_of_birth") == "born_on"
    assert restored.spec("born_on").objects_are_entities is False


@needs_toml
class TestTwoWikiPack:
    """`bench/packs/twowiki.toml`, which is a prerequisite rather than an enhancement.

    Without it every one of 2WikiMultihopQA's 31,120 evidence triples is value-valued once
    the classification rule reaches retrieval, and the graph benchmark reports that the graph
    stopped working. These check the pack's shape; `bench/predicate_audit.py` is what checks
    it against the corpus, and needs the dataset to do so.
    """

    @staticmethod
    def _specs():
        path = ROOT / "bench" / "packs" / "twowiki.toml"
        assert path.is_file(), f"the benchmark vocabulary is missing from {path}"
        return {s.name: s for s in load_specs(str(path))}

    def test_it_loads_and_declares_every_relation_once(self):
        specs = self._specs()
        assert len(specs) == 34, "the corpus has 34 relations; the audit script counts them"

    def test_the_date_relations_take_values_and_walk_nowhere(self):
        """9,854 of the corpus's 31,120 triples, and the reason they are values is the whole
        classification rule: declaring them entity-valued would connect every person born in
        1935 to every work published in 1935."""
        specs = self._specs()
        for name in ("born_on", "date_of_death", "publication_date", "inception"):
            assert specs[name].objects_are_entities is False, name
            assert specs[name].graph is False, name

    def test_every_traversable_relation_declares_an_entity_object(self):
        specs = self._specs()
        for spec in specs.values():
            if spec.graph:
                assert spec.objects_are_entities, spec.name

    def test_the_three_overridden_builtins_keep_their_aliases(self):
        """The pack replaces three builtins to give them an object_type. Replacing drops
        aliases, and `bench/twowiki.py` folds every relation through the registry as it
        loads, so without these the corpus spellings resolve to nothing."""
        from memvara.schema import BUILTIN_PREDICATES

        builtins = {s.name: s for s in BUILTIN_PREDICATES}
        specs = self._specs()
        for name in ("born_on", "born_in", "works_at"):
            assert set(builtins[name].aliases) <= set(specs[name].aliases), (
                f"{name} drops an alias its builtin carries; the corpus spelling for it "
                "will resolve to nothing")

    def test_nothing_supersedes(self):
        """The corpus is a static set of gold evidence loaded in one pass, so a `one`
        declaration would make a second true value retire the first and delete evidence the
        benchmark scores its own recall against. A film has several directors."""
        offenders = [s.name for s in self._specs().values()
                     if s.cardinality is not Cardinality.MANY]
        assert offenders == [], offenders


@needs_toml
class TestPredicateAudit:
    """`bench.predicate_audit.audit`, which encodes decision 3's classification rule.

    Tested despite living in `bench/` — which coverage does not measure and pytest does not
    collect by default — because its first version got this wrong in the direction that
    hides the answer. It reported declared-as-value and undeclared together, so a pack that
    covered every relation still showed a large "gap" made entirely of dates it had
    deliberately declared as values. The counts here are synthetic; the corpus measurement
    is the script's job.
    """

    @staticmethod
    def _audit(counts, pack_body: str | None, tmp_path):
        import bench.predicate_audit as pa
        from memvara.schema import BUILTIN_PREDICATES

        specs = ()
        if pack_body is not None:
            path = tmp_path / "p.toml"
            path.write_text(pack_body, encoding="utf-8")
            specs = load_specs(str(path))
        return pa.audit(counts, PredicateRegistry(BUILTIN_PREDICATES + specs))

    def test_it_separates_declared_values_from_undeclared(self, tmp_path):
        """The distinction the first version lost. Both carry no edge; only one is a gap."""
        from collections import Counter

        pack = ('[[predicate]]\nname="ships_on"\ncardinality="many"\nvolatility="static"\n'
                'object_type=["value"]\n\n'
                '[[predicate]]\nname="depends_on"\ncardinality="many"\nvolatility="slow"\n'
                'object_type=["software"]\ngraph=true\n')
        report = self._audit(Counter({"depends_on": 10, "ships_on": 5, "invented_by": 2}),
                             pack, tmp_path)
        assert report["declared"] == [("depends_on", 10)]
        assert report["values"] == [("ships_on", 5)]
        assert report["undeclared"] == [("invented_by", 2)]
        assert report["entity_valued"] == 10
        assert report["projected_share"] == 10 / 17

    def test_an_undeclared_corpus_can_carry_no_edge_at_all(self, tmp_path):
        """The measurement the whole step exists for, in miniature."""
        from collections import Counter

        report = self._audit(Counter({"director": 7, "mother": 3}), None, tmp_path)
        assert report["declared"] == []
        assert report["projected_share"] == 0.0
        assert report["undeclared"] == [("director", 7), ("mother", 3)]

    def test_a_relation_is_audited_under_the_name_it_is_stored_as(self, tmp_path):
        """`bench/twowiki.py` folds every relation through the registry as it loads, so
        auditing the raw spelling would report a working declaration as missing. This is the
        alias case that made three of 2Wiki's relations look undeclared."""
        from collections import Counter

        pack = ('[[predicate]]\nname="born_in"\ncardinality="many"\nvolatility="static"\n'
                'aliases=["birthplace","place_of_birth"]\n'
                'object_type=["place"]\ngraph=true\n')
        report = self._audit(Counter({"place_of_birth": 9}), pack, tmp_path)
        assert report["declared"] == [("place_of_birth", 9)]
        assert report["undeclared"] == []

    def test_an_empty_corpus_does_not_divide_by_zero(self, tmp_path):
        from collections import Counter

        report = self._audit(Counter(), None, tmp_path)
        assert report["projected_share"] == 0.0
        assert report["asserted"] == 0


def test_the_accepted_pack_keys_track_the_spec_they_build() -> None:
    """`_PREDICATE_KEYS` and `PredicateSpec`'s fields are two lists nothing forces to agree.

    Adding a field to the spec without adding its key here rejects a valid pack with "which
    this version does not understand", and the trail from that message leads to a different
    file than the one that was edited. Deriving the set from `dataclasses.fields` was the
    obvious fix and is the wrong one: it would silently expose every future internal field
    to TOML, which is a quieter failure than the loud one it prevents. So the set stays
    explicit and this test makes drift impossible — a new field has to be deliberately
    accepted or deliberately excluded, and either way somebody edits this line and says
    which.

    `learned` is the one exclusion: a pack declares predicates, so everything it loads is by
    definition declared, and letting a file assert `learned = true` would let it claim its
    own declarations were guesses.
    """
    from dataclasses import fields

    from memvara.schema import _PREDICATE_KEYS

    assert _PREDICATE_KEYS == {f.name for f in fields(PredicateSpec)} - {"learned"}


@needs_toml
class TestTypeNamesAreCaseInsensitive:
    """A capital letter used to turn a value into an entity, silently.

    Every other declared name in a pack is case-insensitive, because `_coerce_enum` folds
    cardinality, volatility and both memory types. Type names were not, so `VALUE_TYPE` was
    compared against the author's exact spelling: `object_type = ["Person", "Value"]` with
    `graph = true` passed the loader's own "a value carries no edge" check, because "Value"
    is not "value", and `objects_are_entities` then reported true. A predicate marked as
    holding scalars became entity-valued with no error anywhere — the false join this
    feature exists to prevent, reached through capitalisation.
    """

    def _spec(self, tmp_path, body: str):
        path = tmp_path / "case.toml"
        path.write_text(body, encoding="utf-8")
        spec, = load_specs(str(path))
        return spec

    def test_a_capitalised_value_type_still_means_value(self, tmp_path):
        spec = self._spec(tmp_path, '[[predicate]]\nname="holds"\ncardinality="many"\n'
                                    'volatility="slow"\nobject_type=["Value"]\n')
        assert spec.object_type == ("value",)
        assert spec.objects_are_entities is False

    def test_a_capitalised_value_type_is_refused_alongside_graph(self, tmp_path):
        """The regression. This pack used to load, and produced a traversable predicate
        whose objects its author had declared to be scalars."""
        with pytest.raises(PredicatePackError, match="cannot both be true"):
            self._spec(tmp_path, '[[predicate]]\nname="holds"\ncardinality="many"\n'
                                 'volatility="slow"\nobject_type=["Person","Value"]\n'
                                 'graph=true\n')

    def test_entity_type_names_fold_too(self, tmp_path):
        """So that two packs naming the same type differently declare the same type."""
        spec = self._spec(tmp_path, '[[predicate]]\nname="depends_on"\ncardinality="many"\n'
                                    'volatility="slow"\nsubject_type=[" Project "]\n'
                                    'object_type=["Software"]\ngraph=true\n')
        assert spec.subject_type == ("project",)
        assert spec.object_type == ("software",)
        assert spec.objects_are_entities is True

    def test_aliases_are_not_folded(self, tmp_path):
        """They name predicates, not types, and `normalize()` already owns that spelling
        rule. A second, quieter one in front of it would be the drift this avoids."""
        spec = self._spec(tmp_path, '[[predicate]]\nname="x"\ncardinality="many"\n'
                                    'volatility="slow"\naliases=["Git_State"]\n')
        assert spec.aliases == ("Git_State",)


@needs_toml
def test_an_inverse_without_an_edge_to_reverse_is_refused(tmp_path):
    """A full inverse pair on a predicate nothing can walk.

    `inverse` names the reverse of an edge, so a predicate with `graph = false` has no edge
    for it to reverse and the declaration resolves to nothing a walk could use — the same
    "looks present in the file, does nothing at runtime" this loader refuses everywhere
    else.

    Deliberately *not* the same as a mixed `object_type`, which is refused only alongside
    `graph`. A mixture means something: `prefers` holds `postgresql` and `plain` alike,
    resolves to a value, and is the documented shape for a predicate that takes either. An
    inverse with no edge means nothing at all.
    """
    path = tmp_path / "inv.toml"
    path.write_text('[[predicate]]\nname="owned_by"\ncardinality="one"\n'
                    'volatility="slow"\ninverse="owns"\ninverse_cardinality="many"\n',
                    encoding="utf-8")
    with pytest.raises(PredicatePackError, match="no edge"):
        load_specs(str(path))
