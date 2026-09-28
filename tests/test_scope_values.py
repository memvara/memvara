"""No level of a scope can hold '*' or the empty string.

`Scope.key()` writes a level that is not bound as '*'. Up to 0.16.0 it wrote a level set
to '*' or to '' the same way, and `Scope.sees()`, which authorizes every read that looks a
claim up by id, compares keys. So a claim written by a handle bound to `user="*"` or
`user=""` had the key of a claim written for the whole tenant: every other user's `get()`
and `why()` returned it, while `get_all()` and `search()`, which compare the stored
columns, did not.

Both values are now refused wherever a `Scope` is built, with a `ValueError` that names
the level and the value. They are not read as "not bound" instead, because then a user who
can choose their own id could write claims that every user's search returns.

A store written before this change can still hold rows under either value. Those rows are
read back as they were stored, and no scope a caller can build reads them: the key now
writes a stored '*' or '' differently from a level that is not bound, a slot read does not
reach them, a document name lookup compares the scope's columns, and a new turn is not
taken for a repeat of one of them.
"""

from __future__ import annotations

import io
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

import pytest

from memvara import Episode, HashingEmbedder, Memvara, NullLLM
from memvara.aio import AsyncMemvara
from memvara.compat.mem0 import Memory
from memvara.compat.mem0_import import import_mem0
from memvara.integrations._common import bind
from memvara.remote.api import RemoteMemvara
from memvara.server.config import ConfigError, ServerConfig
from memvara.server.init import init
from memvara.types import Scope

LEVELS = ("tenant", "user", "project", "agent", "session")
REFUSED = ("*", "")
REFUSED_IDS = ["star", "empty"]


def memvara() -> Memvara:
    return Memvara(llm=NullLLM(), embedder=HashingEmbedder(dim=64))


def rows(mem: Memvara, table: str) -> int:
    return int(mem.store._db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def names(level: str, value: str) -> str:
    """The part of the refusal that names the level and the value, as a pattern."""
    return re.escape(f"{level}={value!r}")


# --- a scope refuses both values at every level ---------------------------------------


@pytest.mark.parametrize("value", REFUSED, ids=REFUSED_IDS)
@pytest.mark.parametrize("level", LEVELS)
def test_every_level_of_a_scope_refuses_star_and_the_empty_string(level, value):
    with pytest.raises(ValueError, match=names(level, value)):
        Scope(**{level: value})


def test_a_level_that_is_not_bound_is_still_none_and_every_other_value_keeps_its_key():
    """Only the two refused values changed. A value that merely contains '*', and the
    separators the key escapes, are written exactly as before, so no stored key moves."""
    assert Scope("acme").key() == "acme/*/*/*/*"
    assert (Scope("acme", "a*b", "bot/1", "s%1", project="github.com/o/r").key()
            == "acme/a*b/github.com%2Fo%2Fr/bot%2F1/s%251")


@pytest.mark.parametrize("value", REFUSED, ids=REFUSED_IDS)
def test_a_handle_cannot_be_bound_where_every_user_could_read_what_it_writes(value):
    """The reproduction from the report. A claim written by a handle bound to
    `user='*'` or `user=''` was returned by every other user's `get()` and `why()`.
    Binding that handle is now refused, so nothing is written."""
    with memvara() as mem:
        with pytest.raises(ValueError, match=names("user", value)):
            mem.scope(user=value).remember("user", "likes", "planted fact")
        assert rows(mem, "claims") == rows(mem, "episodes") == 0


# --- every way a caller names a scope -------------------------------------------------


def _library_calls(mem: Memvara) -> dict[str, Any]:
    return {
        "Memvara(user=...)": lambda: Memvara(llm=NullLLM(), embedder=HashingEmbedder(dim=64),
                                             user="*"),
        "Memvara(tenant=...)": lambda: Memvara(llm=NullLLM(),
                                               embedder=HashingEmbedder(dim=64), tenant=""),
        "Memvara.scope(user=...)": lambda: mem.scope(user=""),
        "Memvara.scope(project=...)": lambda: mem.scope(project="*"),
        "ScopedMemvara.bind(agent=...)": lambda: mem.scope(user="alice").bind(agent="*"),
        "ScopedMemvara.bind(session=...)": lambda: mem.scope(user="alice").bind(session=""),
        "a read's own scope keywords": lambda: mem.get_all(user="*"),
        "a write's own scope keywords": lambda: mem.remember("user", "likes", "tea",
                                                             session=""),
        "add(..., agent=...)": lambda: mem.add("I live in Atlantis.", agent="*"),
        "an Episode built by the caller": lambda: mem.add(
            [Episode(content="I live in Atlantis.", scope=Scope("default", "*"))]),
        "AsyncMemvara.scope(user=...)": lambda: AsyncMemvara(mem).scope(user="*"),
        "the mem0 layer's user_id": lambda: Memory(mem).add("I live in Atlantis.",
                                                           filters={"user_id": ""}),
        "a mem0 payload's user_id": lambda: import_mem0(mem, memories=[
            {"id": "m1", "data": "Prefers dark roast", "user_id": "*",
             "created_at": "2026-01-01T00:00:00+00:00"}]),
        "an integration's bind(user=...)": lambda: bind(mem, user="*"),
        "RemoteMemvara(user=...)": lambda: RemoteMemvara(
            api_key="k", base_url="https://example.test", user="*"),
        "RemoteMemvara.scope(session=...)": lambda: RemoteMemvara(
            api_key="k", base_url="https://example.test").scope(session=""),
    }


#: The names above, read from a store-less instance: building the lambdas calls nothing.
ENTRY_POINTS = list(_library_calls(Memvara.__new__(Memvara)))


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_every_entry_point_that_names_a_scope_refuses_both_values(name):
    with memvara() as mem:
        with pytest.raises(ValueError, match="is not a scope value"):
            _library_calls(mem)[name]()
        assert rows(mem, "claims") == rows(mem, "episodes") == 0


@pytest.mark.parametrize("variable", ["MEMVARA_TENANT", "MEMVARA_USER", "MEMVARA_AGENT",
                                      "MEMVARA_SESSION"])
def test_the_server_refuses_a_star_in_a_scope_variable_and_names_the_variable(variable):
    with pytest.raises(ConfigError, match=f"^{variable}: "):
        ServerConfig.from_env({"MEMVARA_DB": ":memory:", variable: "*"})


def test_an_empty_scope_variable_still_means_the_variable_is_not_set():
    """The environment is a different layer: a client that writes `"MEMVARA_USER": ""`
    into its settings means no user, and that reading is kept."""
    config = ServerConfig.from_env({"MEMVARA_DB": ":memory:", "MEMVARA_TENANT": "",
                                    "MEMVARA_USER": "", "MEMVARA_AGENT": " ",
                                    "MEMVARA_SESSION": ""})
    assert (config.tenant, config.user, config.agent, config.session) == (
        "default", None, None, None)


def test_init_refuses_a_star_as_the_user_before_writing_anything(tmp_path):
    out, err = io.StringIO(), io.StringIO()
    status = init(["--agent", "claude", "--dir", str(tmp_path), "--mode", "local",
                   "--db", str(tmp_path / "store" / "memory.db"), "--user", "*"],
                  env={}, stdout=out, stderr=err)
    assert status == 2
    assert "--user" in err.getvalue() and "is not a scope value" in err.getvalue()
    assert list(tmp_path.iterdir()) == []




# --- a store written before this change --------------------------------------------


def _key_before_this_change(self: Scope) -> str:
    """`Scope.key()` as 0.16.0 wrote it: a level set to '*' or '' reads as not bound."""
    def esc(part: str) -> str:
        return part.replace("%", "%25").replace("/", "%2F")

    return "/".join(esc(p) for p in (self.tenant, self.user or "*", self.project or "*",
                                     self.agent or "*", self.session or "*"))


@contextmanager
def _as_0_16(monkeypatch) -> Iterator[None]:
    """`Scope` as 0.16.0 had it: no refusal, and the old key. Whatever is written inside
    this block is what a store written by that release holds, keys and hashes included."""
    with monkeypatch.context() as before:
        before.setattr(Scope, "__post_init__", lambda self: None, raising=False)
        before.setattr(Scope, "key", _key_before_this_change)
        yield


class PlantedModel:
    """Reads one fact out of any turn, so the planted turn has a claim that cites it.

    The predicate is not a builtin one on purpose. A builtin predicate is personal and
    is filed without a project, so a claim planted under a project would land at the
    user's own level, where the user rightly reads it. A predicate the store learns is
    filed per project, and stays where it was planted."""

    name = "planted"
    is_noop = False

    def extract(self, episodes, known_predicates):
        return [{"subject": "api", "predicate": "depends_on", "object": "atlantis",
                 "polarity": 1, "memory_type": "semantic", "confidence": 0.9,
                 "source_index": 0}]

    def classify_predicate(self, predicate, example):
        return {"cardinality": "many", "volatility": "slow", "memory_type": "semantic"}


def opened(path: str, llm: Any = None) -> Memvara:
    return Memvara(path, llm=llm if llm is not None else NullLLM(),
                   embedder=HashingEmbedder(dim=64))


#: The other levels a planted value sits under, so each level is planted the way a
#: caller could have bound it.
UNDER = {"user": {}, "project": {"user": "alice"}, "agent": {"user": "alice"},
         "session": {"user": "alice", "agent": "bot"}}

EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)


def plant(tmp_path, monkeypatch, level: str, value: str) -> tuple[str, dict[str, list[str]]]:
    """A store written by 0.16.0, holding a turn, two linked claims and a named document
    from a handle bound to `value` at `level`, and nothing else. Returns the store's path
    and the ids."""
    path = str(tmp_path / "written-by-0.16.db")
    with _as_0_16(monkeypatch), opened(path, PlantedModel()) as old:
        handle = old.scope(**UNDER[level], **{level: value})
        said = handle.add("The api depends on Atlantis.")
        runs = handle.remember("api", "runs_on", "planted fact")
        handle.link(runs.added[0].id, said.added[0].id, "derives")
        document = handle.add_document("A planted note about Atlantis.",
                                       custom_id="shared-name", extract=False)
    return path, {"claims": [c.id for c in [*said.added, *runs.added]],
                  "turns": list(said.episode_ids), "documents": [document.id]}


def readers(mem: Memvara) -> dict[str, Any]:
    return {
        "the tenant": mem,
        "alice": mem.scope(user="alice"),
        "alice in a project": mem.scope(user="alice", project="github.com/o/r"),
        "alice's agent": mem.scope(user="alice", agent="bot"),
        "alice's session": mem.scope(user="alice", agent="bot", session="s1"),
        "bob": mem.scope(user="bob"),
    }


def everything_read(reader: Any, ids: dict[str, list[str]]) -> dict[str, Any]:
    """What each read returns to `reader`, keeping only the reads that returned
    something. The store holds nothing but the planted rows, so every read should return
    nothing."""
    answer = reader.ask("What does the api depend on?")
    try:
        reader.document_status("shared-name")
        status = "found"
    except KeyError:
        status = None
    profile = reader.profile()
    got: dict[str, Any] = {
        "get": [i for i in ids["claims"] if reader.get(i) is not None],
        "why": [i for i in ids["claims"] if reader.why(i) is not None],
        "links": [i for i in ids["claims"] if reader.links(i)],
        "produced": [c.id for t in ids["turns"] for c in reader.produced(t)],
        "get_all": [c.id for c in reader.get_all(states=["live", "ended", "retired"])],
        "search": [r.claim.id for r in reader.search("api atlantis planted fact")],
        "recall": list(reader.recall("api atlantis planted fact", with_ids=True).claim_ids),
        "since": [c.id for c in reader.since(EPOCH).added],
        "count": reader.count(),
        "history": [c.id for p in ("depends_on", "runs_on")
                    for c in reader.history("api", p)],
        "standing": [c.id for c in reader.standing()],
        "profile": [row.claim_id for row in (*profile.standing, *profile.recent,
                                             *profile.relevant)],
        "ask": [c.id for r in answer.readings
                for c in (*r.now, *r.then, *r.stated, *r.timeline)],
        "neighborhood": [c.id for p in reader.neighborhood("api") for c in p.claims()],
        "get_document by id": [d for d in ids["documents"]
                               if reader.get_document(d) is not None],
        "get_document by name": reader.get_document("shared-name"),
        "document_status by name": status,
        "list_documents": [d.id for d in reader.list_documents().items],
    }
    return {read: result for read, result in got.items() if result}


@pytest.mark.parametrize("value", REFUSED, ids=REFUSED_IDS)
@pytest.mark.parametrize("level", list(UNDER))
def test_no_read_returns_a_row_a_release_before_this_change_stored_under_either_value(
        tmp_path, monkeypatch, level, value):
    """A store written by 0.16.0 can hold rows under '*' or '' at any level. They are
    read back as stored, and no scope a caller can build reads them, by id or otherwise:
    every read, for every handle, agrees with `get_all()`, which never returned them."""
    path, ids = plant(tmp_path, monkeypatch, level, value)
    assert len(ids["claims"]) == 2 and ids["turns"] and ids["documents"]
    with opened(path) as mem:
        leaked = {name: everything_read(reader, ids)
                  for name, reader in readers(mem).items()}
    assert {name: reads for name, reads in leaked.items() if reads} == {}


@pytest.mark.parametrize("value", REFUSED, ids=REFUSED_IDS)
def test_a_turn_like_one_stored_under_either_value_is_stored_as_the_callers_own(
        tmp_path, monkeypatch, value):
    """0.16.0 hashed a turn under user '*' or '' with the key of a turn the whole tenant
    wrote, and a new turn is looked up by its hash. A tenant's turn with the same text
    must not be taken for a repeat of the planted one."""
    path = str(tmp_path / "written-by-0.16.db")
    with _as_0_16(monkeypatch), opened(path) as old:
        planted = old.scope(user=value).add("I live in Atlantis.")
    with opened(path) as mem:
        receipt = mem.add("I live in Atlantis.")
    assert receipt.repeated == 0
    assert receipt.episode_ids and not set(receipt.episode_ids) & set(planted.episode_ids)


def test_the_tenants_statement_becomes_its_own_claim_beside_one_planted_under_star(
        tmp_path, monkeypatch):
    path = str(tmp_path / "written-by-0.16.db")
    with _as_0_16(monkeypatch), opened(path) as old:
        planted = old.scope(user="*").add("I live in Atlantis.")
    with opened(path) as mem:
        receipt = mem.add("I live in Atlantis.")
        assert [c.id for c in receipt.reinforced] == []
        assert [c.object for c in mem.get_all()] == ["Atlantis"]
        assert [c.id for c in mem.get_all()] != [c.id for c in planted.added]


@pytest.mark.parametrize("value", REFUSED, ids=REFUSED_IDS)
@pytest.mark.parametrize("level", list(UNDER))
def test_maintenance_over_rows_stored_under_either_value_raises_nothing_and_shows_nothing(
        tmp_path, monkeypatch, level, value):
    """Maintenance walks rows it is given or finds, and a row stored under '*' or '' is
    one of them. Re-reading such a turn by id, which works out which claims are the
    turn's own from the turn's scope and the scopes above it, raised `ValueError`,
    because that scope could not be rebuilt. Every call here must finish, and none may
    leave anything a caller's scope reads."""
    path, ids = plant(tmp_path, monkeypatch, level, value)
    with opened(path, PlantedModel()) as mem:
        mem.pending_extraction()
        mem.reextract()
        mem.reextract(ids["turns"])
        mem.consolidate()
        mem.merge_predicate("runs_on", "runs_in", dry_run=False)
        mem.erase_expired()
        leaked = {name: everything_read(reader, ids)
                  for name, reader in readers(mem).items()}
    assert {name: reads for name, reads in leaked.items() if reads} == {}


def test_a_row_stored_under_either_value_is_read_back_as_it_was_stored(tmp_path,
                                                                      monkeypatch):
    """Maintenance walks every row, so reading one back must not raise. Its key does
    not match the key of any scope a caller can build."""
    path, ids = plant(tmp_path, monkeypatch, "user", "*")
    with opened(path) as mem:
        stored = mem.store.get_claim(ids["claims"][0])
        assert stored is not None and stored.scope.user == "*"
        assert stored.scope.key() != Scope("default").key()
        assert len(list(mem.store.iter_claims())) == 2


# --- a read at the scope a row is stored under -----------------------------------------


def _with_editor(path: str) -> Memvara:
    """A handle whose `editor` predicate holds one value at a time, per project, so a
    project's own value shadows the user-wide one (`memvara/retrieve/shadow.py`)."""
    from memvara import MemoryType
    from memvara.schema import (BUILTIN_PREDICATES, Cardinality, PredicateRegistry,
                                PredicateSpec, Volatility)

    registry = PredicateRegistry(specs=BUILTIN_PREDICATES + (
        PredicateSpec(name="editor", cardinality=Cardinality.ONE,
                      volatility=Volatility.SLOW, memory_type=MemoryType.PROCEDURAL),))
    return Memvara(path, llm=NullLLM(), embedder=HashingEmbedder(dim=64),
                   registry=registry)


APP, WEB = "github.com/acme/app", "github.com/acme/web"


@pytest.mark.parametrize("value", REFUSED, ids=REFUSED_IDS)
def test_a_project_read_at_the_scope_a_row_is_stored_under_shadows_without_raising(
        tmp_path, monkeypatch, value):
    """A present-tense read bound to a project hides a user-wide value when the project
    holds its own value in the same slot, and to find that slot it rebuilt the user-wide
    claim's scope with the reader's project through `dataclasses.replace`, which refuses
    '*' and '' again. So the read raised `ValueError` over a user-wide row that 0.16.0
    stored under either value. No scope a caller can build reads such a row, but a read
    at the scope the row is stored under does, such as `HybridRetriever.search` given
    the scope `stored_scope` read back. It must answer by the rule every other scope
    follows: the project's own value hides the user-wide one, and in a project with no
    value of its own the user-wide one answers."""
    from memvara.types import stored_scope

    path = str(tmp_path / "written-by-0.16.db")
    with _as_0_16(monkeypatch), _with_editor(path) as old:
        old.scope(user=value).remember("user", "editor", "vim")
        old.scope(user=value, project=APP).remember("user", "editor", "emacs")
    with _with_editor(path) as mem:
        def read(project: str) -> list[str]:
            at = stored_scope("default", value, None, None, project=project)
            return [r.claim.object for r in mem.reader.search("editor", at)]

        assert read(APP) == ["emacs"]
        assert read(WEB) == ["vim"]
