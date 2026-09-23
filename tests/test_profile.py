"""`standing()` on the local engine, and `profile()` on every surface.

`profile()` replaces the calls a session used to make at its start, so the property that
matters most is that it returns the same rows those calls return. A profile that quietly
disagreed with `standing()` or `since()` would open every session with a different picture
of the user than the tools a model reaches for next.

Everything runs offline against `HashingEmbedder` and `NullLLM`. The hosted client is
tested against `httpx.MockTransport`, because the `/v1/profile` route is built in the
commercial repository and this file pins the request and reply shape the client assumes.
"""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone

import httpx
import pytest

import memvara.core as core_module
from memvara import (AsyncMemvara, HashingEmbedder, MemoryType, Memvara, NullLLM, Profile,
                     Row, utcnow)
from memvara.remote.aio import AsyncRemoteMemvara
from memvara.remote.api import RemoteMemvara
from memvara.schema import PredicatePackError
from memvara.server import MemvaraMCPServer
from memvara.server.validate import ToolError, validate

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)

#: Python 3.10 has no `tomllib`, so the shipped predicate packs cannot be read there, by
#: the decision `schema._toml_reader` records. `profile()` then reports each default
#: bucket as unavailable and leaves every other section as it is. The tests below state
#: both behaviours rather than skipping on 3.10.
PACKS_READABLE = sys.version_info >= (3, 11)


def unavailable():
    """The warnings `profile()` gives on Python 3.10 for default buckets it cannot read."""
    return list(_UNAVAILABLE)


def _pack_error():
    from memvara.schema import PredicatePackError, load_specs
    try:
        load_specs("engineering")
    except PredicatePackError as exc:
        return str(exc)
    return None


_UNAVAILABLE = ([] if PACKS_READABLE else
                [f"the {p!r} bucket is unavailable: {_pack_error()}"
                 for p in ("decisions", "engineering", "events")])
_UNREADABLE = ([] if PACKS_READABLE else
               [f"the {p!r} pack could not be read, so its predicates count as "
                f"undeclared: {_pack_error()}" for p in ("decisions", "engineering", "events")])


def make(**kw):
    kw.setdefault("user", "alice")
    return Memvara(embedder=HashingEmbedder(dim=64), llm=NullLLM(), **kw)


def rule(mem, obj, *, confidence=1.0, extractor="api", recorded_at=None, **kw):
    return mem.remember("user", "prefers", obj, memory_type=MemoryType.PROCEDURAL,
                        confidence=confidence, extractor=extractor,
                        recorded_at=recorded_at, **kw)


# -- standing() ------------------------------------------------------------------

def test_standing_puts_what_the_user_stated_ahead_of_what_a_machine_inferred():
    """A model reports its own confidence, so a paraphrase at 0.99 must not open the
    session ahead of the sentence the user typed at 0.70."""
    mem = make()
    rule(mem, "tabs", confidence=0.99, extractor="capture-hook")
    rule(mem, "pytest", confidence=0.70)
    mem.remember("user", "lives_in", "Berlin")
    assert [c.object for c in mem.standing()] == ["pytest", "tabs"]


def test_standing_breaks_a_confidence_tie_by_recency_and_caps_at_k():
    mem = make()
    rule(mem, "older", recorded_at=T0)
    rule(mem, "newer", recorded_at=T0 + timedelta(days=1))
    assert [c.object for c in mem.standing()] == ["newer", "older"]
    assert [c.object for c in mem.standing(k=1)] == ["newer"]


def test_standing_reads_only_live_procedural_memories_in_this_scope():
    mem = make()
    rule(mem, "pytest")
    rule(mem, "unittest", user="bob")
    stale = rule(mem, "nose").added[0]
    mem.delete(stale.id)
    assert [c.object for c in mem.standing()] == ["pytest"]


# -- profile() on the local engine -----------------------------------------------

def test_a_profile_holds_the_same_rows_as_the_calls_it_replaces():
    """The acceptance test from the design: one call, the same rows as `standing` and
    `since` return separately."""
    mem = make()
    rule(mem, "pytest", recorded_at=T0)
    rule(mem, "ruff", extractor="capture-hook")
    mem.remember("api", "depends_on", "postgres")
    since = T0 + timedelta(days=1)
    profile = mem.profile(k=5, since=since, buckets={})
    assert [r.claim_id for r in profile.standing] == [c.id for c in mem.standing(k=5)]
    assert [r.claim_id for r in profile.recent] == \
        [c.id for c in mem.since(since).added[:5]]
    assert [r.inferred for r in profile.standing] == [False, True]
    assert profile.relevant == [] and profile.warnings == []


def test_recent_leaves_out_what_stopped_being_believed():
    """A profile is read as current context, and a withdrawn fact must not appear in it.
    `since()` reports the withdrawn half as `gone`; the profile keeps only `added`."""
    mem = make()
    mem.remember("user", "lives_in", "Berlin", valid_from=T0, recorded_at=T0)
    mem.remember("user", "lives_in", "Lisbon")
    texts = [r.text for r in mem.profile(since=T0 + timedelta(days=1)).recent]
    assert texts == ["user lives in Lisbon"]


def test_the_recent_window_defaults_to_seven_days():
    mem = make()
    rule(mem, "pytest", recorded_at=utcnow() - timedelta(days=10))
    rule(mem, "ruff")
    profile = mem.profile()
    assert [r.text for r in profile.recent] == ["user prefers ruff"]
    assert len(profile.standing) == 2, "standing is not windowed"


def test_a_query_adds_the_relevant_section_and_only_then():
    mem = make()
    mem.remember("api", "depends_on", "postgres")
    assert mem.profile().relevant == []
    hits = mem.profile("postgres database").relevant
    assert [r.text for r in hits] == ["api depends on postgres"]


def test_default_buckets_are_the_three_shipped_packs():
    mem = make()
    mem.remember("api", "depends_on", "postgres")
    rule(mem, "pytest")
    profile = mem.profile()
    if not PACKS_READABLE:
        # Python 3.10: no bucket, one warning per pack, and the rest of the profile intact.
        assert profile.buckets == {}
        assert profile.warnings == unavailable()
        assert [r.text for r in profile.standing] == ["user prefers pytest"]
        return
    assert list(profile.buckets) == ["decisions", "engineering", "events"]
    assert [r.text for r in profile.buckets["engineering"]] == ["api depends on postgres"]
    assert profile.buckets["decisions"] == [] and profile.buckets["events"] == []
    assert profile.warnings == []


def test_caller_buckets_replace_the_defaults_and_hold_the_newest_k():
    mem = make()
    for n in range(3):
        mem.remember("api", "depends_on", f"lib{n}",
                     recorded_at=T0 + timedelta(days=n))
    profile = mem.profile(k=2, buckets={"stack": ["depends_on"]})
    assert list(profile.buckets) == ["stack"]
    assert [r.text for r in profile.buckets["stack"]] == \
        ["api depends on lib2", "api depends on lib1"]


def test_a_bucket_predicate_nothing_declares_is_ignored_and_reported():
    mem = make()
    profile = mem.profile(buckets={"stack": ["depends_on", "no_such_verb"]})
    assert profile.buckets == {"stack": []}
    # `depends_on` is declared by the engineering pack. On Python 3.10 that pack cannot be
    # read, so the predicate counts as undeclared too, and the warnings say why first.
    undeclared = [] if PACKS_READABLE else [
        "bucket 'stack' names 'depends_on', which nothing declares or uses, so it "
        "was ignored."]
    assert profile.warnings == _UNREADABLE + undeclared + [
        "bucket 'stack' names 'no_such_verb', which nothing declares or uses, so it "
        "was ignored."]


def test_buckets_that_name_known_predicates_never_ask_the_packs(monkeypatch):
    """A registered or stored predicate needs no pack, so on Python 3.10 such a profile
    carries no warning about packs it had no reason to read."""
    def unreadable(pack):
        raise PredicatePackError("needs Python 3.11")
    monkeypatch.setattr(core_module, "_pack_predicates", unreadable)
    mem = make()
    mem.remember("api", "frobnicates", "widgets")
    profile = mem.profile(buckets={"home": ["lives_in"], "odd": ["frobnicates"]})
    assert profile.warnings == []


def test_a_bucket_matches_the_predicate_the_store_actually_wrote():
    """A claim stores the registry's canonical predicate, so a bucket naming an alias the
    registry knows must still find it."""
    mem = make()
    mem.remember("user", "lives_in", "Berlin")
    alias = next(a for spec in mem.registry.all_specs() if spec.name == "lives_in"
                 for a in spec.aliases)
    rows = mem.profile(buckets={"home": [alias]}).buckets["home"]
    assert [r.text for r in rows] == ["user lives in Berlin"]


def test_a_pack_that_cannot_be_read_costs_its_bucket_and_says_so(monkeypatch):
    """Python 3.10 has no `tomllib`, so the packs cannot be read there. The profile
    still answers, and the warning names the bucket that is missing and why."""
    def unreadable(pack):
        raise PredicatePackError("needs Python 3.11")
    monkeypatch.setattr(core_module, "_pack_predicates", unreadable)
    mem = make()
    rule(mem, "pytest")
    profile = mem.profile()
    assert profile.buckets == {}
    assert [r.text for r in profile.standing] == ["user prefers pytest"]
    assert profile.warnings == [f"the {p!r} bucket is unavailable: needs Python 3.11"
                                for p in core_module.PROFILE_PACKS]
    custom = mem.profile(buckets={"stack": ["depends_on"]})
    # The pack failure is reported first, so the warning after it is not read as a
    # verdict on the predicate: nothing could declare it because no pack could be read.
    assert custom.warnings == [
        *(f"the {p!r} pack could not be read, so its predicates count as undeclared: "
          "needs Python 3.11" for p in core_module.PROFILE_PACKS),
        "bucket 'stack' names 'depends_on', which nothing declares or uses, so it "
        "was ignored."]


def test_a_predicate_stored_without_a_declaration_still_fills_its_bucket():
    """With no model, a new predicate is stored as given and never registered. A bucket
    naming it must find those memories instead of calling the predicate unknown."""
    mem = make()
    mem.remember("api", "frobnicates", "widgets")
    profile = mem.profile(buckets={"odd": ["frobnicates"]})
    assert [r.text for r in profile.buckets["odd"]] == ["api frobnicates widgets"]
    assert profile.warnings == []


def test_k_below_one_is_refused():
    with pytest.raises(ValueError, match="at least 1"):
        make().profile(k=0)


def test_a_profile_and_standing_for_another_project_read_that_project():
    mem = make()
    app = mem.scope(project="github.com/acme/app")
    app.remember("api", "depends_on", "postgres")
    app.remember("user", "prefers", "pytest", memory_type=MemoryType.PROCEDURAL)
    stack = {"stack": ["depends_on"]}
    assert [r.text for r in app.profile(buckets=stack).buckets["stack"]] == \
        ["api depends on postgres"]
    assert mem.profile(buckets=stack).buckets["stack"] == []
    # `prefers` is declared global, so the rule is written without a project and is
    # seen from the unscoped instance as well as from the project.
    assert [c.object for c in app.standing()] == ["pytest"]
    assert [c.object for c in mem.standing()] == ["pytest"]


def test_the_project_is_bound_once_and_never_chosen_per_call():
    """Every other read binds the project with `scope(project=...)`. A per-call argument
    on two of them would be a second, inconsistent way to choose it."""
    mem = make()
    with pytest.raises(TypeError):
        mem.profile(project="github.com/acme/app")
    with pytest.raises(TypeError):
        mem.standing(project="github.com/acme/app")


def test_the_scoped_views_forward_standing_and_profile():
    mem = make()
    rule(mem, "pytest")
    view = mem.scope(user="alice")
    assert [c.object for c in view.standing(k=1)] == ["pytest"]
    assert [r.text for r in view.profile(k=1, since=T0).standing] == ["user prefers pytest"]

    async def run():
        amem = AsyncMemvara(mem)
        unscoped = await amem.standing(k=1)
        profile = await amem.profile(since=T0)
        scoped = amem.scope(user="alice")
        return (unscoped, profile, await scoped.standing(),
                await scoped.profile("pytest", since=T0))

    unscoped, profile, scoped, scoped_profile = asyncio.run(run())
    assert [c.object for c in unscoped] == [c.object for c in scoped] == ["pytest"]
    assert isinstance(profile, Profile) and len(scoped_profile.relevant) == 1


def test_the_types_say_what_they_hold():
    assert repr(Profile(standing=[Row("cl_1", "x")])) == \
        "<Profile standing=1 recent=0 relevant=0 buckets=0 warnings=0>"
    assert Row("cl_1", "x").inferred is False


# -- the memory_profile tool ------------------------------------------------------

def server(mem=None, **kw):
    return MemvaraMCPServer(mem or make(), user="alice", **kw)


def call(srv, name, arguments=None):
    reply = srv.handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                "params": {"name": name, "arguments": arguments or {}}})
    body = reply["result"]
    return body["content"][0]["text"], body["isError"]


def test_the_tool_renders_every_section_with_ids_and_the_inferred_mark():
    mem = make()
    stated = rule(mem, "pytest").added[0]
    inferred = rule(mem, "ruff", extractor="capture-hook").added[0]
    mem.remember("api", "depends_on", "postgres")
    body, is_error = call(server(mem), "memory_profile",
                          {"query": "postgres", "since": "2020-01-01",
                           "buckets": {"stack": ["depends_on", "no_such_verb"]}})
    assert not is_error, body
    lines = body.splitlines()
    assert lines[0].startswith("Profile of this scope. Stored memory about the user")
    assert f"+ [id={stated.id}] user prefers pytest" in lines
    assert f"+ [id={inferred.id} inferred] user prefers ruff" in lines
    assert "Arrived since 2020-01-01 00:00Z (3):" in lines
    assert any(line.startswith("Relevant to 'postgres' (") for line in lines)
    assert "Bucket 'stack' (1):" in lines
    ignored = lines[lines.index("Ignored:") + 1:]
    assert ignored == [f"  {w}" for w in _UNREADABLE] + [
        "  bucket 'stack' names 'no_such_verb', which nothing declares or uses, so it "
        "was ignored."]


def test_an_empty_section_says_so_and_no_query_means_no_relevant_section():
    body, _ = call(server(), "memory_profile", {"buckets": {"home": ["lives_in"]}})
    assert "Standing preferences (0):\n  (none)" in body
    assert "Relevant to" not in body and "Ignored:" not in body
    since = body.splitlines()[3]
    assert since.startswith("Arrived since ") and since.endswith("(0):")


def test_the_tool_refuses_a_malformed_since_and_malformed_buckets():
    srv = server()
    body, is_error = call(srv, "memory_profile", {"since": "last tuesday"})
    assert is_error and "memory_profile.since must be an ISO-8601 timestamp" in body
    body, is_error = call(srv, "memory_profile", {"buckets": ["depends_on"]})
    assert is_error and "memory_profile.buckets must be an object" in body
    body, is_error = call(srv, "memory_profile", {"buckets": {"stack": "depends_on"}})
    assert is_error and "memory_profile.buckets.stack must be an array" in body


def test_the_validator_checks_every_value_of_an_object_argument():
    spec = {"m": {"type": "object",
                  "additionalProperties": {"type": "array", "items": {"type": "string"}}}}
    assert validate(spec, (), {"m": {"a": ["x"], "b": []}}, tool="t") == \
        {"m": {"a": ["x"], "b": []}}
    with pytest.raises(ToolError, match=r"t\.m\.a\[0\] must be a string"):
        validate(spec, (), {"m": {"a": [1]}}, tool="t")


def test_switching_the_feature_off_hides_the_tool_and_says_why_when_it_is_called():
    srv = server(features_off={"profile"})
    listed = [t["name"] for t in srv.handle_message(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]]
    assert "memory_profile" not in listed and "memory_standing" in listed
    body, is_error = call(srv, "memory_profile")
    assert is_error
    assert body == ("memory_profile is unavailable: the profile feature is switched off "
                    "on this memory server (MEMVARA_FEATURE_PROFILE=0).")
    assert "features switched off: profile" in call(srv, "memory_stats")[0]


def test_a_read_only_server_still_explains_a_hidden_write_tool_as_read_only():
    """The switched-off message is checked first, so the read-only one must still reach
    a write tool that no feature switch hides."""
    body, is_error = call(server(read_only=True, features_off={"profile"}), "memory_add",
                          {"text": "hi"})
    assert is_error and "read-only" in body


def test_the_server_refuses_a_feature_it_does_not_know():
    with pytest.raises(ValueError, match="'profle' .* is not a feature"):
        server(features_off={"profle"})
    with pytest.raises(ValueError, match="are not features"):
        server(features_off={"a", "b"})


# -- the hosted client ----------------------------------------------------------

_REPLY = {
    "standing": [{"claim_id": "cl_1", "text": "user prefers pytest", "inferred": False}],
    "recent": [{"claim_id": "cl_2", "text": "user prefers ruff", "inferred": True}],
    "relevant": [],
    "buckets": {"engineering": [{"claim_id": "cl_3", "text": "api depends on postgres",
                                 "inferred": False}]},
    "warnings": ["bucket 'x' names 'y', which nothing declares or uses, so it was ignored."],
}


def recorder(client, reply=_REPLY):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=reply)

    transport = httpx.MockTransport(handler)
    client._http._client._transport = transport
    return seen


def test_the_client_posts_the_profile_request_and_hydrates_the_reply():
    mem = RemoteMemvara(api_key="k", base_url="https://example.test", user="alice")
    seen = recorder(mem)
    profile = mem.profile("db", k=3, since=T0, buckets={"stack": ("depends_on",)})
    request = seen[0]
    assert (request.method, request.url.path) == ("POST", "/v1/profile")
    assert dict(request.url.params) == {"user": "alice"}
    assert json.loads(request.content) == {
        "query": "db", "k": 3, "since": "2026-01-01T00:00:00+00:00",
        "buckets": {"stack": ["depends_on"]}}
    assert profile.standing == [Row("cl_1", "user prefers pytest", False)]
    assert profile.recent[0].inferred is True
    assert list(profile.buckets) == ["engineering"] and len(profile.warnings) == 1


def test_the_client_leaves_unset_fields_out_of_the_body():
    mem = RemoteMemvara(api_key="k", base_url="https://example.test")
    seen = recorder(mem)
    mem.scope(user="alice").profile()
    assert json.loads(seen[0].content) == {"k": 8}


def test_the_async_client_sends_the_same_request():
    mem = AsyncRemoteMemvara(api_key="k", base_url="https://example.test", user="alice")
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=_REPLY)

    mem._http._client = httpx.AsyncClient(base_url="https://example.test",
                                          transport=httpx.MockTransport(handler))

    async def run():
        unscoped = await mem.profile("db", buckets={"stack": ["depends_on"]})
        scoped = await mem.scope(agent="a1").profile(since=T0)
        return unscoped, scoped

    unscoped, scoped = asyncio.run(run())
    assert [r.url.path for r in seen] == ["/v1/profile", "/v1/profile"]
    assert json.loads(seen[0].content) == {"query": "db", "k": 8,
                                           "buckets": {"stack": ["depends_on"]}}
    assert unscoped == scoped and unscoped.standing[0].claim_id == "cl_1"


def test_a_reply_missing_a_section_raises_rather_than_reading_as_empty():
    mem = RemoteMemvara(api_key="k", base_url="https://example.test")
    recorder(mem, {k: v for k, v in _REPLY.items() if k != "recent"})
    with pytest.raises(KeyError):
        mem.profile()


def test_the_async_profile_reads_concurrently_and_answers_like_the_sync_one(monkeypatch):
    """The three reads a profile needs are independent, so the async facade dispatches
    them separately rather than as one opaque call on one thread."""
    mem = make()
    rule(mem, "pytest")
    mem.remember("api", "depends_on", "postgres")
    dispatched = []
    real = asyncio.to_thread

    async def counting(fn, *args, **kwargs):
        dispatched.append(getattr(fn, "__name__", fn))
        return await real(fn, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", counting)
    got = asyncio.run(AsyncMemvara(mem).profile("postgres", since=T0,
                                               buckets={"stack": ["depends_on"]}))
    assert len(dispatched) == 3
    assert got == mem.profile("postgres", since=T0, buckets={"stack": ["depends_on"]})


def test_a_profile_reads_the_scope_once_and_asks_the_store_once_more_for_recent(
        monkeypatch):
    """`recent` is the live set minus what was believed at `since`, so it needs one scan
    at that instant on top of the live read, not the two scans `since()` makes."""
    mem = make()
    rule(mem, "pytest")
    calls = []
    real = mem.store.candidate_ids

    def counting(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(mem.store, "candidate_ids", counting)
    mem.profile(since=T0, buckets={})
    assert len(calls) == 2


def test_the_tool_and_the_library_share_one_recent_window():
    """Declared once, in the library, so the tool's header and the library's default
    cannot come to name two different instants."""
    from memvara.server import tools
    assert tools.PROFILE_WINDOW is core_module.PROFILE_WINDOW
