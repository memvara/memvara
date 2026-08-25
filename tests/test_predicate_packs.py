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


def _env(tmp_path, **extra):
    return {"MEMVARA_DB": str(tmp_path / "store.db"), "MEMVARA_TENANT": "t",
            "MEMVARA_EMBEDDER": "hashing:64", "MEMVARA_LLM": "none", **extra}


@needs_toml
class TestLoading:
    def test_the_engineering_pack_ships(self):
        assert "engineering" in available_packs()
        names = {s.name for s in load_specs("engineering")}
        assert {"git_state", "deploys_to", "rejected"} <= names

    def test_the_decisions_pack_ships_and_does_not_shadow_engineering(self):
        """Two shipped packs declaring one predicate would make load order decide the
        store: later entries win, so `engineering,decisions` and `decisions,engineering`
        would disagree about that predicate's `memory_type` from the same data, silently.

        `rejected` and `known_defect` are in `engineering` already and are deliberately
        not redeclared here, which is what makes the union order-independent."""
        assert "decisions" in available_packs()
        decisions = {s.name for s in load_specs("decisions")}
        assert decisions == {"decided", "observed"}
        assert not decisions & {s.name for s in load_specs("engineering")}

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
