"""Invariants of the memory model that no older test checked.

Each test names the invariant it checks. The invariants are stated in `docs/INTERNALS.md`
(the numbered design invariants) and in the "Invariants and assumptions" section of
`docs/claude/memory-model.md`.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import pathlib
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

import pytest

from harness import stores
from memvara import Memvara
from memvara.core import ScopedMemvara
from memvara.embed import HashingEmbedder
from memvara.entities import entity_key, typed_entity_key
from memvara.store import STATES
from memvara.store.sqlite import SCHEMA_VERSION
from memvara.types import Derivation, Episode, Scope

REPO = pathlib.Path(__file__).resolve().parents[3]


class _Extractor:
    """An extraction model that reports one memory for every turn that mentions hiking.

    It stands in for the model tier of the write path. The fast path, which needs no
    model, handles the other turns. It names the turn by its position in the batch, as a
    real backend does, and never names an episode id, so the engine alone decides which
    turn the memory cites.
    """

    name = "fake/hiking"
    is_noop = False
    reports_usage = False
    accepts_guidance = True

    def extract(self, episodes: Sequence[Episode], known_predicates: Sequence[str],
                **_: Any) -> list[dict[str, Any]]:
        return [{"source_index": i, "subject": "user", "predicate": "enjoys",
                 "object": "hiking", "confidence": 0.9}
                for i, ep in enumerate(episodes) if "hiking" in ep.content]

    def resolve_predicate(self, surface: str, candidates: Sequence[str],
                          **_: Any) -> dict[str, Any]:
        return {"canonical": None, "cardinality": "many", "volatility": "slow",
                "memory_type": "semantic"}

    def classify_predicate(self, predicate: str, example: str,
                           **_: Any) -> dict[str, str]:
        return {"cardinality": "many", "volatility": "slow", "memory_type": "semantic"}


@pytest.mark.covers("inv:I4")
def test_every_claim_the_engine_writes_cites_the_turn_it_came_from() -> None:
    """Invariant 4 in docs/INTERNALS.md: every claim carries provenance. `sources` holds
    the ids of the turns the claim came from, and `derivation` says how it was produced.

    The engine writes claims on three paths: the fast path, the model tier, and
    `remember()` with a source turn attached. Each path is driven here, and each stored
    claim is then read back from the store, not from the write receipt. Each must cite
    exactly the turn it came from, and `why()` must resolve that citation to the turn's
    text, which is the point of recording it.
    """
    mem = Memvara(embedder=HashingEmbedder(dim=512), llm=_Extractor(), user="alice")
    try:
        # The canoe turn reaches the model beside the hiking turn and yields nothing, so
        # the hiking memory is not the first turn of the model's batch. A memory cited to
        # the wrong turn of a batch is then visible.
        mem.add([{"role": "user", "content": "I live in Lisbon."},
                 {"role": "user", "content": "My sister bought a canoe last spring."},
                 {"role": "user", "content": "On weekends I go hiking with my sister."}])
        stated = Episode(content="My favourite tea is sencha.", scope=mem.default_scope)
        mem.remember("user", "favourite_tea", "sencha", sources=[stated])

        claims = {c.object: c for c in mem.get_all(states=STATES)}
        assert set(claims) == {"Lisbon", "hiking", "sencha"}, sorted(claims)
        expected = {
            "Lisbon": (Derivation.FAST_PATH, "I live in Lisbon."),
            "hiking": (Derivation.LLM_EXTRACT, "On weekends I go hiking with my sister."),
            "sencha": (Derivation.USER, "My favourite tea is sencha."),
        }
        for value, (derivation, turn) in expected.items():
            claim = claims[value]
            assert claim.derivation is derivation, (value, claim.derivation)
            assert len(claim.sources) == 1, (value, claim.sources)
            provenance = mem.why(claim.id)
            assert provenance is not None, value
            assert [e.id for e in provenance.episodes] == claim.sources, value
            assert [e.content for e in provenance.episodes] == [turn], value
        assert claims["hiking"].extractor == _Extractor.name
    finally:
        mem.close()


@pytest.mark.covers("inv:MM2")
def test_asking_for_all_three_states_readmits_a_fact_recorded_but_not_yet_in_force() -> None:
    """docs/claude/memory-model.md: "The three states do not tile the store." A fact
    recorded today that is in force only from next year is in none of the three states
    until then, so no subset of the states names it. Asking for all three is not the
    union of the parts: it is the belief floor alone, which readmits that fact.

    Checked against a real store, for every proper subset of the states and for all
    three, through both `get_all` and `count`.
    """
    mem = stores.memory(user="alice")
    try:
        next_year = datetime.now(timezone.utc) + timedelta(days=365)
        mem.remember("user", "lives_in", "Lisbon")
        mem.remember("user", "works_at", "Initech", valid_from=next_year)

        def objects(states: Sequence[str]) -> list[str]:
            return sorted(c.object for c in mem.get_all(states=list(states)))

        subsets = [("live",), ("ended",), ("retired",), ("live", "ended"),
                   ("live", "retired"), ("ended", "retired")]
        for subset in subsets:
            assert "Initech" not in objects(subset), subset
        union = sorted({o for subset in subsets for o in objects(subset)})
        assert union == ["Lisbon"]
        assert objects(STATES) == ["Initech", "Lisbon"], "all three is the belief floor"
        assert mem.count(states=list(STATES)) == 2
    finally:
        mem.close()


#: The names a call could use to address a scope.
_SCOPE_NAMES = {"tenant", "user", "agent", "session", "project"}


@pytest.mark.covers("inv:MM4")
def test_a_scoped_view_narrows_and_no_call_on_it_can_widen_it() -> None:
    """docs/claude/memory-model.md: "Scope is bound at startup and cannot be widened by a
    call. `ScopedMemvara.bind()` narrows only."

    Three things would widen a view, and each is checked. `bind()` could drop a field it
    was not given, or clear one when handed None. A method could take a scope keyword and
    address another user with it. A read could return a row from a sibling scope. The
    one public method that names scope fields is `bind()` itself.
    """
    mem = stores.memory()
    try:
        mem.remember("user", "lives_in", "Oslo", user="bob")
        view = mem.scope(user="alice", session="s1")
        view.remember("user", "lives_in", "Lisbon")
        bound = Scope("default", "alice", None, "s1")
        assert view.scope == bound

        assert view.bind().scope == bound
        assert view.bind(user=None, agent=None, session=None).scope == bound
        assert view.bind(agent="a1").scope == Scope("default", "alice", "a1", "s1")
        assert view.bind(agent="a1").bind().scope.session == "s1"

        takes_scope = {
            name for name, member in inspect.getmembers(ScopedMemvara, callable)
            if not name.startswith("_")
            and _SCOPE_NAMES & set(inspect.signature(member).parameters)}
        assert takes_scope == {"bind"}, takes_scope
        # Typed as Any because these calls are wrong on purpose. The type checker refuses
        # them too; this checks that the running code refuses them as well.
        loose: Any = view
        for call in (lambda: loose.remember("user", "likes", "tea", user="bob"),
                     lambda: loose.get_all(user="bob"),
                     lambda: loose.search("Oslo", tenant="other"),
                     lambda: loose.count(session=None)):
            with pytest.raises(TypeError):
                call()

        assert [c.object for c in view.get_all()] == ["Lisbon"]
        assert "Oslo" not in {r.claim.object for r in view.search("where does bob live")}
    finally:
        mem.close()


#: The only places `SQLiteStore` may commit other than through `_maybe_commit`, each
#: with the reason it cannot change a turn that a cached list holds.
_OTHER_COMMITS = {
    "__init__": "creates the schema while the store opens, before any list exists",
    "_assign_slots": "numbers vectors that had no row, and runs only while the store "
                     "opens, when it first learns its width, or inside the rollback that "
                     "batch() follows with _maybe_commit",
    "close": "the store is closing, and no search can follow",
}


def _commits(tree: ast.AST) -> dict[str, int]:
    """For each method of `SQLiteStore`, how many commits it makes on its own database
    connection: `self._db.commit()`, or a `COMMIT` statement executed on it."""
    found: dict[str, int] = {}
    store = next(node for node in ast.walk(tree)
                 if isinstance(node, ast.ClassDef) and node.name == "SQLiteStore")
    for method in store.body:
        if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(method):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and ast.unparse(node.func.value) == "self._db"):
                continue
            statement = (node.args[0].value if node.args
                         and isinstance(node.args[0], ast.Constant)
                         and isinstance(node.args[0].value, str) else "")
            if node.func.attr == "commit" or "COMMIT" in statement.upper():
                found[method.name] = found.get(method.name, 0) + 1
    return found


@pytest.mark.covers("inv:MM6")
def test_every_commit_that_can_change_a_turn_goes_through_maybe_commit() -> None:
    """docs/claude/memory-model.md: "Every commit `SQLiteStore` makes that can change a
    turn goes through `_maybe_commit`." That method empties `_scope_turns` and
    `_scope_claims`, the cached lists the vector leg ranks from. A write that committed
    any other way would leave a search missing a new turn, or returning an erased one,
    until the next commit.

    The rule is about where commits are made, so it is checked on the source: every
    commit on the store's own connection is inside `_maybe_commit`, apart from the three
    listed in `_OTHER_COMMITS` with the reason each is safe. `_maybe_commit` itself must
    empty the lists. A behavioural test could not see a stray commit here, because the
    next ordinary write empties the lists anyway and hides it.
    """
    source = (REPO / "memvara" / "store" / "sqlite.py").read_text(encoding="utf-8")
    commits = _commits(ast.parse(source))
    assert commits.get("_maybe_commit") == 1
    elsewhere = {name for name in commits if name != "_maybe_commit"}
    assert elsewhere == set(_OTHER_COMMITS), (
        f"these methods commit without going through _maybe_commit: "
        f"{sorted(elsewhere - set(_OTHER_COMMITS))}")

    from memvara.store.sqlite import SQLiteStore
    body = inspect.getsource(SQLiteStore._maybe_commit)
    assert "self._changed()" in body


#: Surface forms that exercise every rule of the entity fold: case, whitespace,
#: punctuation, accents, corporate forms, articles, the symbols that end a name, a type
#: namespace, the forms that only look like one, and a value long enough to be bounded.
_FOLD_CORPUS = (
    "Acme, Inc.", "The Acme Corporation", "ACME", "Acme Labs", "  Sun   Microsystems ",
    "Microsystems Sun", "C++", "C#", "C", "F#", "Notepad++", "Disney+", "18+", "A+", "A-",
    "AB-", "Room# 5", "pre- and post-war", "x-ray", "X ray", "C++11", "the the band",
    "the band", "The The", "The", "Zoë", "São Paulo", "Café Müller", "O'Reilly", "e-mail",
    "Dr. Who", "company:Apple Inc.", "fruit:apple", "Software:PostgreSQL",
    "https://memvara.dev", "Note: call Bob back", "09:30", "", "   ", "!!!",
    "the customer said the renewal would be decided after the audit " * 60,
)

#: The digest of the fold over `_FOLD_CORPUS`, for each schema version whose fold it is.
#: An older store re-derives its keys only when `SCHEMA_VERSION` moves, so a fold change
#: needs a new version and a new entry here, together.
_FOLDS = {16: "9bd12ed71ea02b37fc203f60b0ce6285268da363fe5f5ea04617eea7dd2182d7"}


def _fold_digest() -> str:
    folded = [[surface, entity_key(surface), typed_entity_key(surface)]
              for surface in _FOLD_CORPUS]
    return hashlib.sha256(json.dumps(folded).encode("utf-8")).hexdigest()


@pytest.mark.covers("inv:MM8")
def test_a_change_to_the_entity_fold_comes_with_a_new_schema_version() -> None:
    """docs/claude/memory-model.md: "A change to the entity fold bumps `SCHEMA_VERSION`."
    Every key a claim is found by is built from `entity_key`, and a file written before
    a fold change re-derives its keys only when it is upgraded. A fold change without a
    new version leaves every older file on the old keys, so the next write of a stored
    value no longer matches it.

    So the fold's output over a fixed corpus is pinned to the schema version it belongs
    to. If this fails because the fold changed on purpose, bump `SCHEMA_VERSION` in
    `memvara/store/sqlite.py` and add the new version and the new digest to `_FOLDS`.
    """
    latest = max(_FOLDS)
    assert SCHEMA_VERSION >= latest
    digest = _fold_digest()
    assert digest == _FOLDS[latest], (
        f"the entity fold changed, and schema version {latest} is recorded with the old "
        f"fold. Bump SCHEMA_VERSION past {SCHEMA_VERSION} and add "
        f"{{{SCHEMA_VERSION + 1}: {digest!r}}} to _FOLDS.")
    assert len(set(_FOLDS.values())) == len(_FOLDS), "each version has a fold of its own"
