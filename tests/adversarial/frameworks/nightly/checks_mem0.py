"""mem0: the compatibility shim and the importer, against the real mem0ai package.

Each `check_*` function runs inside a virtual environment that holds mem0ai at the floor
memvara declares or at the newest release (see `probe.py`). The shim,
`memvara.compat.mem0.Memory`, says it is mem0 2.x's method surface, so that an existing
call site keeps working. These checks compare it with the real package without starting
any of mem0's backends: they read signatures and pydantic models, and call mem0's own
methods only where mem0 refuses a call before it touches a backend. The importer checks
use mem0's `SQLiteManager`, the class mem0 writes its local history file with, because
reading that file is the importer's whole job.

mem0 is imported inside each check, never at module level, because the suite imports this
module to list the checks and has no mem0 installed.
"""

from __future__ import annotations

import importlib
import inspect
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from memvara.compat import import_mem0
from memvara.compat.mem0 import Mem0CompatError
from memvara.compat.mem0 import Memory as Shim

if TYPE_CHECKING:
    from ..probe import Context

#: mem0's entity ids. The shim's handling of them has checks of its own.
ENTITY_IDS = frozenset({"user_id", "agent_id", "run_id"})

#: Methods of mem0's `Memory` that no call site can depend on, with the reason.
NOT_A_SURFACE = {"chat": "mem0's own chat() raises NotImplementedError"}

#: Defaults the shim changes on purpose, with where the change is documented.
DOCUMENTED_DEFAULTS = {
    ("search", "threshold"): "memvara/compat/__init__.py: search(threshold=0.1) has no "
                             "default floor",
}

JUNE = datetime(2024, 6, 1, tzinfo=timezone.utc)
JULY = datetime(2024, 7, 1, tzinfo=timezone.utc)


def _real() -> Any:
    return importlib.import_module("mem0").Memory


def _parameters(function: Any) -> dict[str, inspect.Parameter]:
    """A function's named parameters, without self and without *args or **kwargs."""
    return {name: parameter
            for name, parameter in inspect.signature(function).parameters.items()
            if name != "self" and parameter.kind not in (inspect.Parameter.VAR_KEYWORD,
                                                         inspect.Parameter.VAR_POSITIONAL)}


def _surface(cls: Any) -> list[str]:
    """A class's public methods, and the with-statement protocol if it has one."""
    names = {name for name in dir(cls)
             if not name.startswith("_") and callable(getattr(cls, name))}
    return sorted(names | {name for name in ("__enter__", "__exit__") if hasattr(cls, name)})


def check_the_shim_takes_every_method_and_argument_mem0_takes(ctx: Context) -> None:
    """Every public method of mem0's Memory, and the with-statement protocol, exists on
    the shim, and every argument mem0's method names is one the shim's method names too.
    The entity ids have checks of their own, and chat() is left out because mem0's own
    raises NotImplementedError."""
    real = _real()
    missing = []
    for name in _surface(real):
        if name in NOT_A_SURFACE:
            continue
        ours = getattr(Shim, name, None)
        if ours is None:
            missing.append(f"{name}()")
            continue
        if name.startswith("__"):
            continue
        named = _parameters(ours)
        missing += [f"{name}({argument}=)" for argument in _parameters(getattr(real, name))
                    if argument not in named and argument not in ENTITY_IDS]
    assert missing == [], "missing from the shim: " + ", ".join(sorted(missing))


def check_add_and_delete_all_take_the_entity_ids_mem0_takes(ctx: Context) -> None:
    """mem0 2.x's add() requires one of user_id, agent_id or run_id, and its delete_all()
    takes them too, so a mem0 call site passes them. The shim must accept them: add()
    files the memory under that user, and delete_all() erases that user's memories."""
    real = _real()
    for method in ("add", "delete_all"):
        assert ENTITY_IDS <= set(_parameters(getattr(real, method))), method
    mem = ctx.memvara()
    shim = Shim(mem)
    shim.add("I live in Berlin", user_id="alice")
    assert [c.object for c in mem.get_all(user="alice")] == ["Berlin"]
    shim.delete_all(user_id="alice")
    assert mem.get_all(user="alice") == []


def check_search_and_get_all_refuse_entity_ids_as_mem0_does(ctx: Context) -> None:
    """mem0 2.x refuses a top-level user_id in search() and get_all(), naming filters=.
    The shim refuses it too, and with the same exception type, so that a caller's except
    clause catches both. mem0's own methods are called on an instance whose __init__ never
    ran: they refuse the argument before they touch a backend."""
    real = _real()
    shim = Shim(ctx.memvara())
    unstarted = real.__new__(real)
    calls: list[tuple[str, Callable[[Any], Any]]] = [
        ("search", lambda m: m.search("q", user_id="alice")),
        ("get_all", lambda m: m.get_all(user_id="alice"))]
    for method, call in calls:
        raised: list[type] = []
        for target in (unstarted, shim):
            try:
                call(target)
            except Exception as exc:  # the exception's type is what is compared
                raised.append(type(exc))
            else:
                raised.append(type(None))
        theirs, ours = raised
        assert theirs is ValueError, f"mem0's {method} now raises {theirs.__name__}"
        assert ours is theirs, (f"the shim's {method} raises {ours.__name__} where mem0 "
                                f"raises {theirs.__name__}")


def check_defaults_match_mem0s_except_the_documented_threshold(ctx: Context) -> None:
    """An argument both methods name has the same default in both, so an unmodified call
    site gets as many results. The one documented exception is search's threshold."""
    real = _real()
    differences = []
    for name in _surface(real):
        ours = getattr(Shim, name, None)
        if ours is None or name.startswith("__") or name in NOT_A_SURFACE:
            continue
        theirs = _parameters(getattr(real, name))
        for argument, parameter in _parameters(ours).items():
            if argument not in theirs or (name, argument) in DOCUMENTED_DEFAULTS:
                continue
            expected = theirs[argument].default
            if expected is not inspect.Parameter.empty and parameter.default != expected:
                differences.append(f"{name}({argument}={parameter.default!r}, mem0 "
                                   f"{expected!r})")
    for (name, argument), where in DOCUMENTED_DEFAULTS.items():
        assert argument in _parameters(getattr(real, name)), (name, argument, where)
    assert differences == [], "defaults that differ from mem0's: " + ", ".join(differences)


def check_every_row_carries_mem0s_memoryitem_fields(ctx: Context) -> None:
    """Rows from search() and get() carry every field of mem0's MemoryItem, and rows from
    get_all() every field but score, which mem0 leaves out of its get_all rows too."""
    fields = set(importlib.import_module("mem0.configs.base").MemoryItem.model_fields)
    shim = Shim(ctx.memvara())
    added = shim.add("I live in Berlin")["results"][0]
    got = shim.get(added["id"])
    assert got is not None, "get() found nothing for the id add() returned"
    rows = {"search": shim.search("where do I live")["results"][0], "get": got,
            "get_all": shim.get_all()["results"][0]}
    expected = {"search": fields, "get": fields, "get_all": fields - {"score"}}
    missing = {method: sorted(expected[method] - set(row)) for method, row in rows.items()
               if expected[method] - set(row)}
    assert missing == {}, f"fields mem0's rows carry and the shim's do not: {missing}"


def check_history_rows_carry_mem0s_history_columns(ctx: Context) -> None:
    """A row of the shim's history() has the keys a row of mem0's own history() has,
    which mem0 reads out of its SQLiteManager."""
    storage = importlib.import_module("mem0.memory.storage")
    manager = storage.SQLiteManager(str(ctx.folder / "history.db"))
    try:
        manager.add_history("m1", None, "I live in Berlin", "ADD",
                            created_at="2024-03-01T00:00:00+00:00")
        theirs = set(manager.get_history("m1")[0])
    finally:
        manager.close()
    shim = Shim(ctx.memvara())
    added = shim.add("I live in Berlin")["results"][0]
    ours = set(shim.history(added["id"])[0])
    assert ours == theirs, {"only mem0": sorted(theirs - ours),
                            "only the shim": sorted(ours - theirs)}


def _mem0_log(ctx: Context, *, add_first: bool) -> Path:
    """A history file written by mem0's own SQLiteManager, the way mem0 writes one.

    Memory m1 is added on 2024-03-01 and updated on 2024-06-01. Memory m2 is added on
    2024-03-02 and deleted on 2024-07-01. mem0 writes an UPDATE or DELETE row with its
    memory's creation time in `created_at` and the time of the event in `updated_at`
    (`Memory._update_memory` and `_delete_memory` in mem0/memory/main.py), and so do the
    calls below. mem0 gives each row a random uuid4 id. Here the ids are replaced with
    fixed ones, so that each ADD row sorts before, or after, the row that changes it.
    Either order is one that mem0's random ids produce.
    """
    storage = importlib.import_module("mem0.memory.storage")
    path = ctx.folder / "history.db"
    manager = storage.SQLiteManager(str(path))
    try:
        manager.add_history("m1", None, "Alice lives in Berlin", "ADD",
                            created_at="2024-03-01T00:00:00+00:00",
                            updated_at="2024-03-01T00:00:00+00:00")
        manager.add_history("m1", "Alice lives in Berlin", "Alice lives in Lisbon",
                            "UPDATE", created_at="2024-03-01T00:00:00+00:00",
                            updated_at=JUNE.isoformat())
        manager.add_history("m2", None, "Alice likes tea", "ADD",
                            created_at="2024-03-02T00:00:00+00:00",
                            updated_at="2024-03-02T00:00:00+00:00")
        manager.add_history("m2", "Alice likes tea", None, "DELETE",
                            created_at="2024-03-02T00:00:00+00:00",
                            updated_at=JULY.isoformat(), is_deleted=1)
    finally:
        manager.close()
    low, high = "00000000-0000-4000-8000-0000000000", "ffffffff-ffff-4fff-bfff-ffffffffff"
    add, change = (low, high) if add_first else (high, low)
    connection = sqlite3.connect(path)
    with connection:
        for memory in ("m1", "m2"):
            connection.execute("UPDATE history SET id = ? WHERE memory_id = ? "
                               "AND event = 'ADD'", (add + memory, memory))
            connection.execute("UPDATE history SET id = ? WHERE memory_id = ? "
                               "AND event != 'ADD'", (change + memory, memory))
    connection.close()
    return path


def check_import_mem0_dates_each_event_when_mem0_recorded_it(ctx: Context) -> None:
    """import_mem0 dates each event of mem0's own history file at the time mem0 recorded
    it: the updated value's predecessor ends on 2024-06-01, and the deleted memory stops
    being believed on 2024-07-01. Here each ADD row sorts first."""
    mem = ctx.memvara()
    import_mem0(mem, history_db=_mem0_log(ctx, add_first=True))
    claims = {c.object: c for c in mem.get_all(states=("live", "ended", "retired"))}
    ended = claims["Alice lives in Berlin"].valid_to
    retired = claims["Alice likes tea"].invalidated_at
    assert (ended, retired) == (JUNE, JULY), (
        f"the import dated mem0's update at {ended:%Y-%m-%d} and its delete at "
        f"{retired:%Y-%m-%d}, not at 2024-06-01 and 2024-07-01, the times mem0 recorded "
        "in updated_at")


def check_import_mem0_replays_each_event_after_the_add_it_changes(ctx: Context) -> None:
    """import_mem0 applies an ADD before the UPDATE or DELETE that changes it, whatever
    their row ids, so that afterwards only the updated value is live. Here each ADD row
    sorts last."""
    mem = ctx.memvara()
    import_mem0(mem, history_db=_mem0_log(ctx, add_first=False))
    live = sorted(c.object for c in mem.get_all())
    assert live == ["Alice lives in Lisbon"], (
        f"after the import the live values are {live}, not only the updated value")


def check_update_and_from_config_refuse_with_mem0compaterror(ctx: Context) -> None:
    """The two calls the shim documents as refused, update() and from_config(), raise
    Mem0CompatError when given every argument mem0's own versions take, not TypeError."""
    real = _real()
    shim = Shim(ctx.memvara())
    arguments = {name: "x" for name in _parameters(real.update) if name != "memory_id"}
    refusals: list[Callable[[], Any]] = [
        lambda: shim.update("m1", **arguments),
        lambda: Shim.from_config({"vector_store": {"provider": "qdrant"}})]
    for call in refusals:
        try:
            call()
        except Mem0CompatError:
            pass
        else:
            raise AssertionError("a call the shim documents as refused was accepted")
