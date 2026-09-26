"""The library's four clients give the same answers to the same operations.

One program of operations runs through each client, each with a store of its own:

* `Memvara`, the synchronous library, whose answers the others are compared with;
* `AsyncMemvara`, the same library behind `asyncio.to_thread`;
* `RemoteMemvara` and `AsyncRemoteMemvara`, the hosted clients, against `FakeV1`, whose
  answers come from a real local store (`tests/harness/fakes/fake_v1.py`).

`compare.normalise` removes what two stores differ in whatever the surface: ids,
instants taken from the clock, and lists that come back in id order. Every step is then
its own test, and a failure lists each field that differs, with both values.

**Where the hosted clients differ on purpose, the test asserts the difference.** Each
documented difference has its own test below, which says where it is documented and
checks that it is real, so the day it goes away the test says so:

* a claim does not carry `temporal_precision`, `object_kind`, `amount` or `unit` over
  `/v1` (docs/claude/testing.md, the `FakeV1` section);
* a search result's ranking does not carry `graph_rank`, `graph_score`,
  `temporal_rank`, `temporal_score` or `intent` (`memvara/remote/hydrate.py`,
  `explanation`);
* `stats()` also carries the two join counts, which `connectivity()` reads back out
  (`memvara/remote/api.py`, `RemoteMemvara.connectivity`);
* `recall(budget=...)` and `recall(valid_at=...)` are refused (`memvara/remote/api.py`,
  the module docstring and `RemoteMemvara.recall`, and docs/API.md); memvara/memvara#298
  tracks giving the hosted recall a time axis;
* the status of a document the caller cannot see is a `KeyError` on both sides, and
  each side words its own message (`Memvara.document_status` and
  `RemoteMemvara.document_status`).

The hosted clients also have `end()`, which sends `POST /v1/end`. The library has no
`end()`: it ends a fact with `delete(close="ended")` and a slot with
`forget(close="ended")`, so the program ends them that way on every client, and a test
of its own checks that `end()` leaves a hosted store holding what those two leave a
local one holding.

**What the program does not compare yet.** Two differences were found that nothing
documents, so they were reported to the maintainer to be filed and pinned as strict
expected failures, rather than asserted here:

* a write receipt read through a hosted client has empty `accumulated`, `disputed`,
  `collapsed` and `retyped` lists where the local receipt reports what the write did,
  so the program leaves out the writes that produce those four outcomes for now;
* the hosted `end()` has no local twin, while docs/API.md says `service()` is the only
  hosted method without one.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterator

import pytest

from harness import stores
from harness.fakes.fake_v1 import FakeV1
from memvara import AsyncMemvara, MemoryType
from memvara.types import utcnow

from .compare import Raised, Run, assert_same, labels, normalise

UTC = timezone.utc

#: Instants the program passes on purpose. Each is more than `compare.CLOCK_WINDOW` away
#: from any run, so the comparison keeps it as it is.
BERLIN_FROM = datetime(2024, 1, 1, tzinfo=UTC)
LISBON_FROM = datetime(2025, 1, 1, tzinfo=UTC)
TRIP_FROM = datetime(2025, 6, 1, tzinfo=UTC)
TRIP_TO = datetime(2025, 9, 1, tzinfo=UTC)
LEFT_ACME = datetime(2025, 10, 1, tzinfo=UTC)
#: A day while the user lived in Berlin.
IN_BERLIN = BERLIN_FROM + timedelta(days=30)
#: When the expiring fact is erased: 30 days after this module is imported, the same
#: instant for every client.
EXPIRES = (utcnow() + timedelta(days=30)).replace(microsecond=0)

#: An id that no store holds.
MISSING = "cl_00000000000000000000"
QUESTION = "where does the user live"

CLIENTS = ("Memvara", "AsyncMemvara", "RemoteMemvara", "AsyncRemoteMemvara")
HOSTED = ("RemoteMemvara", "AsyncRemoteMemvara")


@dataclasses.dataclass(frozen=True)
class Step:
    """One operation of the program: a name, and what it does to a client.

    `run` receives the client and every earlier step's result, by name, so a step can
    use an id an earlier step returned.
    """

    name: str
    run: Callable[[Any, dict[str, Any]], Any]


def _added(got: dict[str, Any], step: str) -> str:
    """The id of the claim an earlier write step stored."""
    return str(got[step].added[0].id)


def _token(got: dict[str, Any]) -> str:
    return str(got["forget_matching.preview"].confirm)


PROGRAM: tuple[Step, ...] = (
    Step("remember", lambda m, got: m.remember("user", "lives_in", "Berlin",
                                               valid_from=BERLIN_FROM)),
    Step("remember.replacing", lambda m, got: m.remember("user", "lives_in", "Lisbon",
                                                         valid_from=LISBON_FROM)),
    Step("remember.rule", lambda m, got: m.remember(
        "user", "prefers", "short answers", memory_type=MemoryType.PROCEDURAL)),
    Step("remember.second_rule", lambda m, got: m.remember(
        "user", "prefers", "tabs over spaces", memory_type=MemoryType.PROCEDURAL)),
    Step("remember.finished", lambda m, got: m.remember(
        "user", "located_now", "Porto", valid_from=TRIP_FROM, valid_to=TRIP_TO,
        until_reason="the trip ended")),
    Step("remember.expiring", lambda m, got: m.remember(
        "user", "goal", "run a marathon", expires_at=EXPIRES,
        expire_reason="a goal for this season")),
    Step("remember.employer", lambda m, got: m.remember("user", "works_at", "Acme",
                                                        valid_from=LISBON_FROM)),
    Step("remember.taste", lambda m, got: m.remember("user", "likes", "jazz")),
    Step("add", lambda m, got: m.add("My name is Ada.")),
    Step("remember.cited", lambda m, got: m.remember(
        "user", "speaks", "Portuguese", sources=[got["add"].episode_ids[0]])),
    Step("search", lambda m, got: m.search(QUESTION, k=5)),
    Step("search.past", lambda m, got: m.search(QUESTION, k=5, valid_at=IN_BERLIN)),
    Step("recall", lambda m, got: m.recall(QUESTION)),
    Step("recall.budget", lambda m, got: m.recall(QUESTION, budget=12)),
    Step("recall.past", lambda m, got: m.recall(QUESTION, valid_at=IN_BERLIN)),
    Step("get", lambda m, got: m.get(_added(got, "remember.replacing"))),
    Step("get.missing", lambda m, got: m.get(MISSING)),
    Step("history", lambda m, got: m.history("user", "lives_in")),
    Step("why", lambda m, got: m.why(_added(got, "remember.replacing"))),
    Step("why.cited", lambda m, got: m.why(_added(got, "remember.cited"))),
    Step("why.missing", lambda m, got: m.why(MISSING)),
    Step("count", lambda m, got: m.count()),
    Step("stats", lambda m, got: m.stats()),
    Step("connectivity", lambda m, got: m.connectivity()),
    Step("standing", lambda m, got: m.standing()),
    Step("standing.one", lambda m, got: m.standing(k=1)),
    Step("profile", lambda m, got: m.profile(QUESTION)),
    Step("forget_matching.preview", lambda m, got: m.forget_matching(
        "jazz", close="retired", k=1)),
    Step("forget_matching.other_closure", lambda m, got: m.forget_matching(
        "jazz", close="ended", k=1, confirm=_token(got))),
    Step("forget_matching.confirm", lambda m, got: m.forget_matching(
        "jazz", close="retired", k=1, confirm=_token(got))),
    Step("forget_matching.replayed", lambda m, got: m.forget_matching(
        "jazz", close="retired", k=1, confirm=_token(got))),
    Step("delete", lambda m, got: m.delete(_added(got, "remember.replacing"),
                                           reason="it was Porto")),
    Step("delete.missing", lambda m, got: m.delete(MISSING)),
    Step("end.claim", lambda m, got: m.delete(_added(got, "remember.cited"), close="ended")),
    Step("end.slot", lambda m, got: m.forget("user", "works_at", close="ended",
                                             at=LEFT_ACME)),
    Step("forget", lambda m, got: m.forget("user", "prefers")),
    Step("document.add", lambda m, got: m.add_document(
        "A runbook. Restart the service.", custom_id="docs/runbook", title="Runbook",
        meta={"team": "ops"})),
    Step("document.get", lambda m, got: m.get_document("docs/runbook")),
    Step("document.get.by_id", lambda m, got: m.get_document(got["document.add"].id)),
    Step("document.get.missing", lambda m, got: m.get_document("docs/none")),
    Step("document.list", lambda m, got: m.list_documents()),
    Step("document.update", lambda m, got: m.update_document("docs/runbook",
                                                             title="The runbook")),
    Step("document.status", lambda m, got: m.document_status("docs/runbook")),
    Step("document.status.missing", lambda m, got: m.document_status("docs/none")),
    Step("document.delete", lambda m, got: m.delete_document("docs/runbook")),
    Step("document.delete.again", lambda m, got: m.delete_document("docs/runbook")),
    Step("document.add.second", lambda m, got: m.add_document(
        "A second note.", custom_id="docs/second")),
    Step("document.delete.many", lambda m, got: m.delete_documents(["docs/second",
                                                                    "docs/none"])),
    Step("stats.after", lambda m, got: m.stats()),
    Step("connectivity.after", lambda m, got: m.connectivity()),
    Step("get_all.after", lambda m, got: m.get_all(states=["live", "ended", "retired"])),
)


class _Blocking:
    """An async client whose calls each run to completion on `loop`, so the same
    synchronous program can drive it."""

    def __init__(self, client: Any, loop: asyncio.AbstractEventLoop) -> None:
        self._client = client
        self._loop = loop

    def __getattr__(self, name: str) -> Callable[..., Any]:
        method = getattr(self._client, name)
        return lambda *args, **kwargs: self._loop.run_until_complete(method(*args, **kwargs))


def play(client: Any) -> dict[str, Any]:
    """Run the program on `client` and return every step's result, by step name.

    A step that raises is recorded as `Raised` rather than stopping the program, so a
    client that fails at one step is still compared at every other.
    """
    got: dict[str, Any] = {}
    for step in PROGRAM:
        try:
            got[step.name] = step.run(client, got)
        except Exception as exc:  # noqa: BLE001 - a refusal is an answer to compare
            got[step.name] = Raised(type(exc).__name__, str(exc))
    return got


@contextlib.contextmanager
def opened(name: str, loop: asyncio.AbstractEventLoop) -> Iterator[Any]:
    """The client `name`, bound to the user alice over a store of its own, and closed
    when the block ends. An async client runs its calls on `loop` (see `_Blocking`), and
    a hosted one talks to a `FakeV1` of its own through a mock transport."""
    if name == "Memvara":
        local = stores.memory(user="alice")
        try:
            yield local
        finally:
            local.close()
    elif name == "AsyncMemvara":
        wrapped = AsyncMemvara(stores.memory(user="alice"))
        try:
            yield _Blocking(wrapped, loop)
        finally:
            loop.run_until_complete(wrapped.close())
    elif name == "RemoteMemvara":
        with FakeV1() as fake:
            remote = fake.remote(user="alice")
            try:
                yield remote
            finally:
                remote.close()
    else:
        with FakeV1() as fake:
            aremote = fake.aremote(user="alice")
            try:
                yield _Blocking(aremote, loop)
            finally:
                loop.run_until_complete(aremote.aclose())


@pytest.fixture(scope="module")
def played() -> dict[str, dict[str, Any]]:
    """Every client's normalised answer to every step it runs, played once per module."""
    start = utcnow()
    raw: dict[str, dict[str, Any]] = {}
    loop = asyncio.new_event_loop()
    try:
        for name in CLIENTS:
            with opened(name, loop) as client:
                raw[name] = play(client)
    finally:
        loop.close()
    run = Run(start, utcnow())
    out: dict[str, dict[str, Any]] = {}
    for client, got in raw.items():
        names = labels(*got.values())
        out[client] = {step: normalise(result, names, run=run)
                       for step, result in got.items()}
    return out


# -- what the hosted clients document they return differently -------------------------

#: `Claim` fields `/v1` does not carry, so a claim read through a hosted client has each
#: one's default, which is None. Documented in docs/claude/testing.md, in the section on
#: `FakeV1`.
NOT_ON_THE_WIRE = ("temporal_precision", "object_kind", "amount", "unit")

#: `Explanation` fields the hosted ranking does not carry, so a hosted search result has
#: each one's default, which is None. Documented in `memvara/remote/hydrate.py`, in the
#: docstring of `explanation`.
NOT_RANKED_ON_THE_WIRE = ("graph_rank", "graph_score", "temporal_rank", "temporal_score",
                          "intent")

#: The steps whose hosted answer a documented rule of its own describes, each checked by
#: its own test below rather than by the step-by-step comparison.
HOSTED_BY_OWN_TEST = frozenset({"recall.budget", "recall.past", "stats", "stats.after",
                                "document.status.missing"})


def as_hosted(value: Any) -> Any:
    """A local answer as the hosted clients document they return it: with the default in
    every field `NOT_ON_THE_WIRE` and `NOT_RANKED_ON_THE_WIRE` names."""
    if isinstance(value, dict):
        out = {key: as_hosted(item) for key, item in value.items()}
        if out.get("__type__") == "Claim":
            out.update(dict.fromkeys(NOT_ON_THE_WIRE))
        elif out.get("__type__") == "Explanation":
            out.update(dict.fromkeys(NOT_RANKED_ON_THE_WIRE))
        return out
    if isinstance(value, list):
        return [as_hosted(item) for item in value]
    return value


def _compared() -> Iterator[Any]:
    for step in PROGRAM:
        for client in CLIENTS[1:]:
            if client in HOSTED and step.name in HOSTED_BY_OWN_TEST:
                continue
            yield pytest.param(step.name, client, id=f"{step.name}-{client}")


@pytest.mark.parametrize(("step", "client"), list(_compared()))
def test_every_client_answers_as_the_synchronous_library_does(
        played: dict[str, dict[str, Any]], step: str, client: str) -> None:
    local = played["Memvara"][step]
    expected = as_hosted(local) if client in HOSTED else local
    assert_same(expected, played[client][step], f"{step} through {client}")


def _typed(value: Any, kind: str) -> Iterator[dict[str, Any]]:
    """Every normalised dataclass of class `kind` inside `value`."""
    if isinstance(value, dict):
        if value.get("__type__") == kind:
            yield value
        for item in value.values():
            yield from _typed(item, kind)
    elif isinstance(value, list):
        for item in value:
            yield from _typed(item, kind)


@pytest.mark.parametrize("client", HOSTED)
def test_a_hosted_claim_has_the_default_in_each_field_the_wire_does_not_carry(
        played: dict[str, dict[str, Any]], client: str) -> None:
    """docs/claude/testing.md, the `FakeV1` section: `/v1` does not carry a claim's
    `temporal_precision`, `object_kind`, `amount` or `unit`."""
    local = list(_typed(played["Memvara"], "Claim"))
    hosted = list(_typed(played[client], "Claim"))
    assert any(claim["object_kind"] is not None for claim in local), (
        "no local claim sets object_kind, so this test no longer shows the difference")
    assert hosted and all(claim[field] is None
                          for claim in hosted for field in NOT_ON_THE_WIRE)


@pytest.mark.parametrize("client", HOSTED)
def test_a_hosted_ranking_has_the_default_in_each_field_the_wire_does_not_carry(
        played: dict[str, dict[str, Any]], client: str) -> None:
    """`memvara/remote/hydrate.py`, `explanation`: `graph_rank`, `graph_score`,
    `temporal_rank`, `temporal_score` and `intent` are not on the wire."""
    local = list(_typed(played["Memvara"], "Explanation"))
    hosted = list(_typed(played[client], "Explanation"))
    assert any(ranking["intent"] is not None for ranking in local), (
        "no local ranking sets intent, so this test no longer shows the difference")
    assert hosted and all(ranking[field] is None
                          for ranking in hosted for field in NOT_RANKED_ON_THE_WIRE)


@pytest.mark.parametrize("client", HOSTED)
@pytest.mark.parametrize("step", ["stats", "stats.after"])
def test_hosted_stats_also_carry_the_join_counts(
        played: dict[str, dict[str, Any]], step: str, client: str) -> None:
    """`memvara/remote/api.py`, `RemoteMemvara.connectivity`: the hosted client reads
    `live_claims` and `joinable_claims` out of what `stats()` returns, so `stats()`
    carries the library's counts and the join counts together."""
    local = played["Memvara"]
    joined = local[step.replace("stats", "connectivity")]
    assert "joinable_claims" not in local[step]
    assert_same({**local[step], **joined}, played[client][step], f"{step} through {client}")


@pytest.mark.parametrize("client", HOSTED)
def test_a_hosted_client_refuses_a_recall_budget(
        played: dict[str, dict[str, Any]], client: str) -> None:
    """`memvara/remote/api.py`, the module docstring and `RemoteMemvara.recall`:
    `POST /v1/recall` renders the block on the server and takes no budget, so a budget
    is refused rather than ignored."""
    local = played["Memvara"]["recall.budget"]
    hosted = played[client]["recall.budget"]
    assert isinstance(local, str) and "did not fit" in local
    assert hosted["__type__"] == "Raised" and hosted["kind"] == "ValueError"
    assert hosted["message"].startswith(
        "recall(budget=...) is not available against a hosted deployment")


@pytest.mark.parametrize("client", HOSTED)
def test_a_hosted_client_refuses_a_dated_recall(
        played: dict[str, dict[str, Any]], client: str) -> None:
    """`memvara/remote/api.py`, the module docstring and `RemoteMemvara.recall`, and
    docs/API.md: `POST /v1/recall` has no time axis, so `valid_at` is refused rather than
    answered with the present. memvara/memvara#298 tracks giving it one; when that lands,
    this test fails, and the dated recall joins the step-by-step comparison."""
    local = played["Memvara"]["recall.past"]
    hosted = played[client]["recall.past"]
    assert isinstance(local, str) and "as things were on 31 January 2024" in local
    assert hosted["__type__"] == "Raised" and hosted["kind"] == "ValueError"
    assert hosted["message"].startswith(
        "recall(valid_at=...) is not available against a hosted deployment")


@pytest.mark.parametrize("client", HOSTED)
def test_a_missing_document_s_status_is_a_key_error_through_every_client(
        played: dict[str, dict[str, Any]], client: str) -> None:
    """`Memvara.document_status` and `RemoteMemvara.document_status` both document a
    `KeyError` for a document the caller cannot see. Each words its own message: the
    library names the scope it looked in, and the hosted client says the document is not
    visible here, as its other refusals do. So the class is compared and the wording is
    not."""
    for name in ("Memvara", client):
        refusal = played[name]["document.status.missing"]
        assert (refusal["__type__"], refusal["kind"]) == ("Raised", "KeyError"), name
        assert "docs/none" in refusal["message"], name


@pytest.mark.parametrize("client", HOSTED)
def test_the_hosted_end_method_closes_what_the_library_closes(client: str) -> None:
    """The hosted clients have `end()`, which sends `POST /v1/end`, as a second way to
    close a fact that stopped being true. The library closes the same two ways with
    `delete(close="ended")` and `forget(close="ended")`. Both must leave the store
    holding the same claims."""
    start = utcnow()

    def end_both_ways(mem: Any, end_claim: Callable[[str], Any],
                      end_slot: Callable[[], Any]) -> list[Any]:
        """Store two facts, end one by its id and the other by its slot, and return
        both answers and every claim the store then holds."""
        mem.remember("user", "works_at", "Acme", valid_from=LISBON_FROM)
        taste = mem.remember("user", "likes", "jazz").added[0].id
        return [end_claim(taste), end_slot(), mem.get_all(states=["live", "ended", "retired"])]

    loop = asyncio.new_event_loop()
    try:
        with opened("Memvara", loop) as local:
            expected = end_both_ways(
                local, lambda claim_id: local.delete(claim_id, close="ended"),
                lambda: local.forget("user", "works_at", close="ended", at=LEFT_ACME))
        with opened(client, loop) as hosted:
            actual = end_both_ways(
                hosted, lambda claim_id: hosted.end(claim_id=claim_id),
                lambda: hosted.end(predicate="works_at", at=LEFT_ACME))
    finally:
        loop.close()
    run = Run(start, utcnow())
    # `forget` returns the claims it closed, and `end` returns only whether it closed any.
    expected[1] = bool(expected[1])
    assert_same(as_hosted(normalise(expected, labels(expected), run=run)),
                normalise(actual, labels(actual), run=run), f"end() through {client}")
