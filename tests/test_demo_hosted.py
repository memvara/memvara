"""The two memvara arms of the demo, run against a hosted deployment instead of a local file.

`demo/hosted.py` puts `memvara` and `memvara_structured` behind the hosted client
(`memvara.remote`), so the answer-quality run can measure the service people actually
use. Everything here is offline. The hosted client is replaced by `FakeHosted`, which
stands a real local `Memvara` behind exactly the surface the hosted client exposes — and,
where the hosted client refuses something, refuses it the same way. That is what lets a
test say something about the service rather than about a mock that agrees with the code.

The failures these tests exist to prevent:

1. **Demo data in somebody's real store.** The machine this runs on already holds a
   credential for a project somebody works in. A run that wrote a few thousand support
   tickets into it could not be taken back by this code, so the demo refuses that
   credential by path and by key. When the two files name the same project, it also asks
   the server which tenant each key reaches and refuses when it is the same one. All of
   this happens before anything is written.
2. **A hosted arm that silently answers a different question.** A hosted project cannot
   declare the support schema, so `plan` is multi-valued there and a new plan would sit
   beside the old one. And `POST /v1/recall` has no time axis, so a dated question read
   through it would get today's block. Each would produce a plausible, wrong row. The
   arm closes single-valued slots itself and reads dated questions through
   `search(valid_at=)`; the tests check the slots against the local structured arm, which
   has the schema, rather than against expectations written here.
3. **The same scope written twice.** Replaying the facts into a scope that already holds
   them is not idempotent — an explicit close at an old instant would close the value that
   replaced it — so a scope is written once per run, recorded in a manifest, and a scope a
   crashed run left half-written is refused rather than finished.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from demo import baselines as bl  # noqa: E402  (puts bench/ on the path)
from demo import harness as hz  # noqa: E402
from demo import hosted as ho  # noqa: E402
from demo import scenario  # noqa: E402

import evalkit as ek  # noqa: E402

from memvara import HashingEmbedder, Memvara, NullLLM  # noqa: E402

UTC = timezone.utc
QUESTIONS = list(scenario.questions())
TURNS = list(scenario.conversation())


# --- a hosted deployment, offline -----------------------------------------------


class FakeScoped:
    """`ScopedRemoteMemvara`'s surface over one local store, one per scope.

    The store runs on the built-in vocabulary alone, which is what a hosted project has:
    the client cannot send a predicate schema, so `plan` and the two addresses are not
    single-valued there. `recall(valid_at=)` raises, as `RemoteMemvara.recall` does.
    """

    def __init__(self, owner: "FakeHosted", user: str) -> None:
        self.owner = owner
        self.user = user
        self.mem = Memvara(embedder=HashingEmbedder(dim=bl.EMBED_DIM), llm=NullLLM(),
                           user="customer")

    def add(self, messages: list[dict[str, Any]], *, role: str = "user",
            ts: datetime | None = None) -> Any:
        self.owner.calls.append(("add", self.user, len(messages)))
        for message in messages:
            self.mem.add(message["content"], role=message["role"], ts=message["ts"])
        return None

    def remember(self, subject: str, predicate: str, obj: str, **kw: Any) -> Any:
        assert "close" not in kw, ("the hosted `remember` has no `close`; passed through "
                                   "**meta it would be stored as metadata and do nothing")
        self.owner.calls.append(("remember", self.user, predicate))
        return self.mem.remember(subject, predicate, obj, **kw)

    def supersede(self, old_claim_id: str, subject: str, predicate: str, obj: str, *,
                  at: datetime, close: str, **kw: Any) -> Any:
        from memvara import Claim

        self.owner.calls.append(("supersede", self.user, close))
        new = Claim(subject=subject, predicate=predicate, object=obj,
                    valid_from=kw["valid_from"], recorded_at=kw["recorded_at"],
                    text=kw["text"], extractor=kw["extractor"])
        return self.mem.supersede(old_claim_id, new, at=at, close=close)

    def forget(self, subject: str, predicate: str, *, at: datetime | None = None,
               close: str = "retired") -> list:
        self.owner.calls.append(("forget", self.user, close))
        return self.mem.forget(subject, predicate, at=at, close=close)

    def history(self, subject: str, predicate: str) -> list:
        return self.mem.history(subject, predicate)

    def recall(self, query: str, *, k: int = 8, include_episodes: bool = False,
               valid_at: datetime | None = None) -> str:
        if valid_at is not None:
            raise ValueError("recall(valid_at=...) is not available against a hosted "
                             "deployment")
        self.owner.calls.append(("recall", self.user, None))
        return self.mem.recall(query, k=k, include_episodes=include_episodes)

    def search(self, query: str, *, k: int = 10, valid_at: datetime | None = None,
               include_episodes: bool = False) -> Any:
        self.owner.calls.append(("search", self.user, valid_at))
        return self.mem.search(query, k=k, valid_at=valid_at,
                               include_episodes=include_episodes)

    def count(self) -> int:
        return len(self.mem.get_all())


class FakeHosted:
    """`RemoteMemvara`: `scope(user=)` hands out one `FakeScoped` per user, kept.

    `whoami()` answers the way `GET /v1/whoami` does, with the tenant the key is bound
    to. Pass `whoami=` to replace that answer with another value, or with an exception to
    raise instead, which is how a refused key or an unreachable deployment looks.
    """

    def __init__(self, *, tenant: str = "prj_demo", whoami: Any = None) -> None:
        self.scopes: dict[str, FakeScoped] = {}
        self.calls: list[tuple] = []
        self.answer = (whoami if whoami is not None
                       else {"token_id": "tok_1", "scope": {"tenant": tenant}})
        self.closed = False

    def scope(self, *, user: str) -> FakeScoped:
        return self.scopes.setdefault(user, FakeScoped(self, user))

    def whoami(self) -> Any:
        self.calls.append(("whoami", None, None))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer

    def close(self) -> None:
        self.closed = True


def _hosted(tmp_path: Path, *, run_id: str = "r1", scale: int = 1,
            client: FakeHosted | None = None) -> ho.HostedMemvara:
    return ho.HostedMemvara(client or FakeHosted(), run_id=run_id, scale=scale,
                            manifest=ho.Manifest(tmp_path / "manifest.jsonl"))


def _local_structured_store(question) -> Memvara:
    """The local structured arm's store, built exactly as that arm builds it."""
    mem, _ = bl.build_memory(bl.visible_turns(question, TURNS), max_episodes=bl.DEFAULT_K,
                             registry=bl.SUPPORT_REGISTRY)
    bl.apply_facts(mem, bl.visible_facts(question, bl.SUPPORT_FACTS))
    return mem


# --- the structured arm, without a schema it can send ----------------------------


def test_closing_slots_on_the_client_gives_the_same_slots_as_the_declared_schema(tmp_path):
    """The claim this arm stands on, checked against the arm that has the schema.

    For every question instant and every slot the fact table writes, the live values in
    the hosted scope — a store that knows none of `SUPPORT_PREDICATES` — must equal the
    local structured arm's, at the present and at every `about` instant a historical
    question reads at. If the client-side closing were wrong anywhere, a plan would have
    two live values, or a moved address would still be live, and that difference is
    exactly what this compares."""
    hosted = _hosted(tmp_path)
    slots = sorted({(f.subject, f.predicate) for f in bl.SUPPORT_FACTS})
    instants = sorted({q.about for q in QUESTIONS if q.about is not None})
    for question in {q.asked_at: q for q in QUESTIONS}.values():
        hosted.memvara_structured(question, TURNS)
        remote = hosted.client.scope(user=hosted.scope_name("memvara_structured",
                                                            question)).mem
        local = _local_structured_store(question)
        for subject, predicate in slots:
            for valid_at in [None, *instants]:
                assert (_slot(remote, subject, predicate, valid_at)
                        == _slot(local, subject, predicate, valid_at)), (
                    f"{subject}.{predicate} at {valid_at} asked {question.asked_at}")


def _slot(mem: Memvara, subject: str, predicate: str, valid_at: datetime | None) -> list:
    """Live values in one slot, with the predicate spelled the way *this* store files it.

    `baselines.slot_values` matches the predicate name literally, which is right for the
    local arm and wrong here: a store on the built-in vocabulary files `plan` under
    `goal` (see the next test), so a literal match would find nothing and report a
    difference that is only a spelling."""
    filed = mem.registry.normalize(predicate)
    return sorted(c.object for c in mem.get_all(valid_at=valid_at)
                  if c.subject == subject and c.predicate == filed)


def test_on_the_built_in_vocabulary_plan_is_filed_under_goal_and_the_report_says_so(
        tmp_path):
    """Found by the test above, and a property of the hosted service rather than of the
    fake. With no `SUPPORT_PREDICATES` sent — and a client cannot send them — the
    built-in vocabulary resolves `plan` as an alias of `goal`, a declared multi-valued
    predicate, so the plan history is stored as `account.goal`. `history("account",
    "plan")` still resolves the alias, which is why closing the slot works, and the
    rendered block carries the claim's own sentence, "The account is on the Home plan.",
    so the reader sees the same words. What changes is anything keyed on the predicate,
    and a row of the report should not hide that it was measured on a store that did
    this. Pinned so that a vocabulary change on either side shows up here first."""
    hosted = _hosted(tmp_path)
    last = max(QUESTIONS, key=lambda q: q.asked_at)
    hosted.memvara_structured(last, TURNS)
    mem = hosted.client.scope(user=hosted.scope_name("memvara_structured", last)).mem
    filed = {c.predicate for c in mem.get_all() if c.object == "Home"}
    assert filed == {"goal"}
    assert [c.object for c in mem.history("account", "plan") if c.state == "live"] == [
        "Home"]
    assert ho.folded_predicates() == {"plan": "goal"}
    assert "plan -> goal" in hosted.backend_note(ho.HostedCredential(
        api_key="k", base_url="u", project="p", path=tmp_path))


def test_a_correction_retires_on_the_client_exactly_as_the_local_arm_does(tmp_path):
    """The two `retired` records — the mistyped mobile and the misread serial — must
    answer nothing at any world-time in the hosted scope, which is the property that
    separates this arm from a store with one clock."""
    hosted = _hosted(tmp_path)
    last = max(QUESTIONS, key=lambda q: q.asked_at)
    hosted.memvara_structured(last, TURNS)
    mem = hosted.client.scope(user=hosted.scope_name("memvara_structured", last)).mem
    for subject, predicate, wrong in (("account", "mobile", "07700 900 118"),
                                      ("main_unit", "serial", "HX2-4419-B")):
        states = {c.object: c.state for c in mem.history(subject, predicate)}
        assert states[wrong] == "retired"
    assert ("supersede", hosted.scope_name("memvara_structured", last), "retired") in \
        hosted.client.calls


def test_a_dated_question_is_read_through_search_because_hosted_recall_has_no_time_axis(
        tmp_path):
    """`POST /v1/recall` refuses `valid_at`, and a dated question answered from today's
    block would be graded as a reader failure when it was a read failure. So the four
    dated questions read `search(valid_at=)`, rendered with the library's own recall
    renderer, under the dated header the local arm's block carries."""
    hosted = _hosted(tmp_path)
    dated = next(q for q in QUESTIONS if q.about is not None)
    context = hosted.memvara_structured(dated, TURNS)

    assert context.read == "search"
    assert "as things were on" in context.text
    reads = [c for c in hosted.client.calls if c[0] in ("recall", "search")]
    assert reads == [("search", hosted.scope_name("memvara_structured", dated),
                      dated.about)]


def test_an_undated_question_is_read_through_recall_as_the_local_arms_are(tmp_path):
    hosted = _hosted(tmp_path)
    undated = next(q for q in QUESTIONS if q.about is None)
    for arm in (hosted.memvara, hosted.memvara_structured):
        context = arm(undated, TURNS)
        assert context.read == "recall" and context.text
        assert context.chars <= bl.MAX_CONTEXT_CHARS


def test_the_dated_rendering_is_the_local_recall_block_byte_for_byte(tmp_path):
    """Rendering `search()` results is only honest if it is the renderer that ships.
    Checked on a local store, where `recall(valid_at=)` exists to compare against."""
    question = next(q for q in QUESTIONS if q.about is not None)
    local = _local_structured_store(question)
    expected = local.recall(question.text, k=bl.DEFAULT_K, valid_at=question.about,
                            include_episodes=True)
    results = local.search(question.text, k=bl.DEFAULT_K, valid_at=question.about,
                           include_episodes=True)
    assert ho.render_dated(results, question.about) == expected


# --- what gets written, and how often --------------------------------------------


def test_each_scope_is_written_once_per_run_however_many_questions_read_it(tmp_path):
    """Eighteen of the twenty questions share one `asked_at`. Writing the history once
    per question would be twenty ingests of the same turns; once per instant is three,
    and every question at that instant reads a store holding exactly what it may see."""
    hosted = _hosted(tmp_path)
    for question in QUESTIONS:
        hosted.memvara(question, TURNS)
        hosted.memvara_structured(question, TURNS)
    instants = {q.asked_at for q in QUESTIONS}
    written = {user for kind, user, _ in hosted.client.calls if kind == "add"}
    assert len(written) == 2 * len(instants)
    for question in QUESTIONS:
        scope = hosted.client.scope(user=hosted.scope_name("memvara", question))
        assert scope.mem.stats()["episodes"] == len(bl.visible_turns(question, TURNS)), (
            "a scope must hold exactly the turns visible at its instant, no more")


def test_a_scope_name_carries_the_run_the_arm_the_scale_and_the_instant(tmp_path):
    """Two runs, two scales or two arms never share a scope, so no run reads another's
    writes — which is also what keeps the noise-floor repeat from reading a store the
    first run left, unless it asks to by reusing the run id."""
    question = QUESTIONS[0]
    names = {
        _hosted(tmp_path, run_id=r, scale=s).scope_name(arm, question)
        for r in ("r1", "r2") for s in (1, 10) for arm in ("memvara", "memvara_structured")
    }
    assert len(names) == 8
    name = _hosted(tmp_path, run_id="r1", scale=10).scope_name("memvara", question)
    assert "r1" in name and "s10" in name and "memvara" in name
    assert question.asked_at.strftime("%Y%m%dT%H%M") in name


def test_a_completed_scope_is_read_again_without_being_written_again(tmp_path):
    """The repeat run for the noise floor reuses the run id, so it measures the reader
    twice over the same stored contexts rather than two different ingests."""
    client = FakeHosted()
    first = _hosted(tmp_path, client=client)
    question = QUESTIONS[0]
    first.memvara_structured(question, TURNS)
    writes = len([c for c in client.calls if c[0] in ("add", "remember")])

    second = _hosted(tmp_path, client=client)
    second.memvara_structured(question, TURNS)
    assert len([c for c in client.calls if c[0] in ("add", "remember")]) == writes


def test_a_scope_a_crashed_run_left_half_written_is_refused_rather_than_finished(tmp_path):
    """Replaying the fact table into a scope that already holds part of it closes values
    at instants they were never closed at. There is no way to finish the write safely, so
    the refusal names the way out: a new run id."""
    manifest = ho.Manifest(tmp_path / "manifest.jsonl")
    question = QUESTIONS[0]
    hosted = ho.HostedMemvara(FakeHosted(), run_id="r1", scale=1, manifest=manifest)
    manifest.started(hosted.scope_name("memvara_structured", question))

    again = ho.HostedMemvara(FakeHosted(), run_id="r1", scale=1,
                             manifest=ho.Manifest(tmp_path / "manifest.jsonl"))
    with pytest.raises(SystemExit, match="--hosted-run-id"):
        again.memvara_structured(question, TURNS)


def test_the_manifest_survives_a_torn_last_line(tmp_path):
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps({"scope": "a", "status": "complete"}) + "\n{\"scope\": ")
    assert ho.Manifest(path).status("a") == "complete"


# --- whose store it writes to ----------------------------------------------------


def _write(path: Path, **fields: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fields))
    return path


def _serve(monkeypatch, by_key: dict[str, Any]) -> dict[str, FakeHosted]:
    """Put one `FakeHosted` behind each api key, in place of `hosted.connect`.

    `by_key` maps a key to the tenant its `whoami` reports, or to the value or exception
    its `whoami` returns or raises instead. A key missing from the map fails the test, so
    a check that should have been settled without asking the server shows up as a lookup
    nobody expected. Returns the clients by key, so a test can see which keys were looked
    up and that each client was closed."""
    served: dict[str, FakeHosted] = {}

    def connect(credential: ho.HostedCredential) -> FakeHosted:
        assert credential.api_key in by_key, "an unexpected whoami lookup was made"
        answer = by_key[credential.api_key]
        client = (FakeHosted(tenant=answer) if isinstance(answer, str)
                  else FakeHosted(whoami=answer))
        served[credential.api_key] = client
        return client

    monkeypatch.setattr(ho, "connect", connect)
    return served


def _home(tmp_path: Path) -> Path:
    """The default credentials file as it is on the machine this bug was found on: a
    project named `dev`, on a tenant the demo must never write into."""
    return _write(tmp_path / "home" / "credentials.json", api_key="mv_home-key",
                  project="dev", server_url="https://app.memvara.dev")


def test_the_demo_refuses_the_credential_this_machine_already_uses(tmp_path, monkeypatch):
    """By path and by key, because each is a different way to end up in the same store:
    the default file itself, or a copy of its key (or of `MEMVARA_API_KEY`) in another
    file. These refusals are decided from the files alone, so no lookup is made for them;
    the empty `_serve` map fails the test if one is. None of the refusals prints a key."""
    default = _home(tmp_path)
    _serve(monkeypatch, {})

    def load(path: Path, env: dict[str, str] | None = None) -> Any:
        return ho.load_demo_credential(path, env=env or {}, default_path=default)

    for path, env, reason in (
        (default, None, "default credentials file"),
        (_write(tmp_path / "copy.json", api_key="mv_home-key", project="demo"), None,
         "same key"),
        (_write(tmp_path / "copy-dev.json", api_key="mv_home-key", project="dev"), None,
         "same key"),
        (_write(tmp_path / "env.json", api_key="mv_env-key", project="demo"),
         {"MEMVARA_API_KEY": "mv_env-key"}, "same key"),
        (tmp_path / "absent.json", None, "memvara login --credentials"),
    ):
        with pytest.raises(SystemExit) as caught:
            load(path, env)
        message = str(caught.value)
        assert reason in message, message
        assert "mv_" not in message

    good = load(_write(tmp_path / "demo.json", api_key="mv_demo", project="memvara-demo",
                       server_url="https://app.memvara.dev"))
    assert (good.project, good.base_url) == ("memvara-demo", "https://app.memvara.dev")


def test_a_demo_project_with_the_same_name_on_another_tenant_is_accepted(tmp_path,
                                                                         monkeypatch):
    """The bug. A project's name is not unique across tenants. This machine's own
    credential is for a project named `dev`, and the demo's project, on a different
    tenant, is also named `dev`. Comparing the names refused the demo's credential even
    though it reaches a different store. The server says which tenant each key is bound
    to, and two different tenants are two different stores, so the credential is
    accepted."""
    default = _home(tmp_path)
    served = _serve(monkeypatch, {"mv_home-key": "prj_home", "mv_demo": "prj_demo"})
    demo = _write(tmp_path / "demo.json", api_key="mv_demo", project="dev",
                  server_url="https://app.memvara.dev")

    credential = ho.load_demo_credential(demo, env={}, default_path=default)

    assert (credential.project, credential.api_key) == ("dev", "mv_demo")
    assert set(served) == {"mv_home-key", "mv_demo"}, "both keys must be looked up"
    assert all(client.closed for client in served.values())


def test_a_second_key_for_the_same_tenant_is_refused(tmp_path, monkeypatch):
    """What the name check was for. A second key minted for the project this machine
    already uses is a different key for the same store, and the server reports the same
    tenant for both. The refusal names the file and the project, and prints no key."""
    default = _home(tmp_path)
    _serve(monkeypatch, {"mv_home-key": "prj_home", "mv_other": "prj_home"})
    sibling = _write(tmp_path / "sibling.json", api_key="mv_other", project="dev")

    with pytest.raises(SystemExit) as caught:
        ho.load_demo_credential(sibling, env={}, default_path=default)

    message = str(caught.value)
    assert "same tenant" in message, message
    assert str(sibling) in message and "'dev'" in message and "prj_home" in message
    assert "mv_" not in message


@pytest.mark.parametrize("failure", [
    "the demo key is refused",
    "this machine's key is refused",
    "the deployment cannot be reached",
    "the answer names no tenant",
    "the answer is not a mapping",
    "the reply is not JSON",
    "the reply cannot be decoded",
])
def test_a_tenant_lookup_that_fails_refuses_the_credential(tmp_path, monkeypatch,
                                                           failure):
    """Fail closed. When the two files name the same project, only the server can say
    whether they reach the same store, and a lookup that did not answer has not said
    that they do not. So every way the lookup can fail (a key the server refuses, a
    deployment that cannot be reached, an answer with no tenant in it) refuses the
    credential with a message saying the stores could not be told apart and what to do.
    The message must not carry a key, even when the error it reports quotes one."""
    from memvara.remote.errors import AuthError, RemoteError

    default = _home(tmp_path)
    by_key: dict[str, Any] = {"mv_home-key": "prj_home", "mv_demo": "prj_demo"}
    if failure == "the demo key is refused":
        by_key["mv_demo"] = AuthError(401, "unauthorized", "key mv_demo is revoked")
    elif failure == "this machine's key is refused":
        by_key["mv_home-key"] = AuthError(401, "unauthorized", "token expired")
    elif failure == "the deployment cannot be reached":
        by_key["mv_demo"] = RemoteError(0, "transport",
                                        "could not reach the deployment: timed out")
    elif failure == "the answer names no tenant":
        by_key["mv_demo"] = {"token_id": "tok_1", "scope": {}}
    elif failure == "the answer is not a mapping":
        by_key["mv_home-key"] = ["not", "a", "mapping"]
    elif failure == "the reply is not JSON":
        # What `response.json()` raises on a success response with an HTML body.
        by_key["mv_demo"] = json.JSONDecodeError("Expecting value", "<html>", 0)
    else:
        httpx = pytest.importorskip("httpx")
        by_key["mv_demo"] = httpx.DecodingError("malformed gzip body")
    served = _serve(monkeypatch, by_key)
    demo = _write(tmp_path / "demo.json", api_key="mv_demo", project="dev")

    with pytest.raises(SystemExit) as caught:
        ho.load_demo_credential(demo, env={}, default_path=default)

    message = str(caught.value)
    assert "could not confirm" in message, message
    assert str(demo) in message and "'dev'" in message
    assert "memvara whoami --credentials" in message
    assert "mv_" not in message
    assert all(client.closed for client in served.values())


def test_a_lookup_that_cannot_build_a_client_refuses_the_credential(tmp_path,
                                                                     monkeypatch):
    """Without the `cloud` extra the hosted client cannot be built at all. That is a
    failed lookup like any other, so it refuses the credential rather than skipping the
    check. httpx is made unimportable here, as it is on an install without the extra,
    so the list of lookup errors is built the way it is there."""
    import sys

    default = _home(tmp_path)

    def connect(credential: ho.HostedCredential) -> Any:
        raise ImportError("httpx is not installed")

    monkeypatch.setattr(ho, "connect", connect)
    monkeypatch.setitem(sys.modules, "httpx", None)
    demo = _write(tmp_path / "demo.json", api_key="mv_demo", project="dev")
    with pytest.raises(SystemExit, match="could not confirm"):
        ho.load_demo_credential(demo, env={}, default_path=default)


def test_a_bug_in_the_lookup_is_raised_rather_than_reported_as_a_failed_lookup(
        tmp_path, monkeypatch):
    """The lookup catches only what a lookup can fail with: the client's own errors, a
    missing extra, and a reply it cannot read. An exception of any other type is a bug
    in this code, such as a misspelt attribute. Reporting it as "could not confirm"
    would send the user to check two credentials that are fine, so it is left to raise.
    The client is still closed."""
    default = _home(tmp_path)
    served = _serve(monkeypatch, {"mv_home-key": "prj_home",
                                  "mv_demo": AttributeError("no attribute 'scpoe'")})
    demo = _write(tmp_path / "demo.json", api_key="mv_demo", project="dev")

    with pytest.raises(AttributeError, match="scpoe"):
        ho.load_demo_credential(demo, env={}, default_path=default)
    assert served["mv_demo"].closed


def test_a_key_that_begins_with_the_other_key_is_masked_whole(tmp_path, monkeypatch):
    """Keys are masked in a failure message one at a time. If the demo's key is a prefix
    of this machine's key and is masked first, the longer key becomes `[key]` followed by
    the rest of it, and that remainder is part of a secret. The longer key is masked
    first, so nothing of either key is printed."""
    from memvara.remote.errors import AuthError

    default = _write(tmp_path / "home" / "credentials.json",
                     api_key="mv_dev-home-secret", project="dev")
    _serve(monkeypatch, {
        "mv_dev": "prj_demo",
        "mv_dev-home-secret": AuthError(401, "unauthorized",
                                        "key mv_dev-home-secret is revoked")})
    demo = _write(tmp_path / "demo.json", api_key="mv_dev", project="dev")

    with pytest.raises(SystemExit) as caught:
        ho.load_demo_credential(demo, env={}, default_path=default)

    message = str(caught.value)
    assert "key [key] is revoked" in message, message
    assert "home-secret" not in message and "mv_" not in message


# --- the harness, with the hosted backend ----------------------------------------


def test_the_harness_runs_the_hosted_arms_and_says_where_they_read_from(tmp_path,
                                                                       monkeypatch,
                                                                       capsys):
    """`--memory hosted` swaps the two memvara arms and nothing else, and the report says
    so beside the table, with the run id and how the dated questions were read — a row
    from the service and a row from a local file are different measurements and must not
    be quoted as one."""
    client = FakeHosted()
    monkeypatch.setattr(ho, "connect", lambda credential: client)
    monkeypatch.setattr(ho, "load_demo_credential",
                        lambda path, **kw: ho.HostedCredential(
                            api_key="mv_demo", base_url="https://app.memvara.dev",
                            project="memvara-demo", path=Path(path)))
    monkeypatch.setattr(hz, "load_scenario", lambda: (QUESTIONS[:3], TURNS))
    out = tmp_path / "rows.jsonl"

    assert hz.main(["--reader", "stub", "--memory", "hosted",
                    "--hosted-credentials", str(tmp_path / "demo.json"),
                    "--hosted-run-id", "run7",
                    "--hosted-manifest", str(tmp_path / "m.jsonl"),
                    "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "memory arms: hosted, project memvara-demo on https://app.memvara.dev" in printed
    assert "run run7" in printed
    assert "mv_demo" not in printed
    assert {c[1] for c in client.calls if c[0] == "add"}, "the hosted arms wrote nothing"
    assert "naive_rag" in printed, "the local arms still run beside the hosted ones"


def test_the_hosted_backend_needs_a_credential_named_for_it(monkeypatch, capsys):
    monkeypatch.setattr(hz, "load_scenario", lambda: (QUESTIONS[:1], TURNS))
    with pytest.raises(SystemExit):
        hz.main(["--reader", "stub", "--memory", "hosted"])
    assert "--hosted-credentials" in capsys.readouterr().err


def test_the_local_backend_is_the_default_and_prints_nothing_new(monkeypatch, capsys):
    """The offline report is pinned by `test_the_offline_run_is_identical_twice`, and a
    hosted-backend line printed on every local run would be a line about a backend that
    was not used."""
    monkeypatch.setattr(hz, "load_scenario", lambda: (QUESTIONS[:1], TURNS))
    assert hz.main(["--reader", "stub"]) == 0
    assert "memory arms:" not in capsys.readouterr().out


# --- the trap column, split by which clock closed ---------------------------------


def test_the_trapped_rate_is_split_by_closure_as_the_readme_asks():
    """One trapped percentage merges "served a value that expired" with "served a value
    that was never true". The scenario records which on every question as `closure`, and
    the report now carries a table per arm and closure, and the per-question output
    carries the field, so the split is in the run rather than done by hand afterwards."""
    questions = ([q for q in QUESTIONS if q.closure == "ended"][:2]
                 + [q for q in QUESTIONS if q.closure == "retired"][:2])
    items = hz.plan(questions, TURNS, arms={"none": bl.none})
    answers = {q.id: (q.trap or "") for q in questions}

    class TrapReader:
        name, is_stub, is_human = "trap", False, False

        def answer(self, system: str, prompt: str) -> ek.Answer:
            qid = next(q.id for q in questions if f"Question: {q.text}" in prompt)
            return ek.Answer(text=answers[qid], model="stub")

    scored = hz.score(items, questions, reader=TrapReader(), judge=ek.ContainmentJudge())
    assert {row.closure for row in scored} == {"ended", "retired"}
    text = hz.report(items, scored, reader=ek.StubReader(), judge=ek.ContainmentJudge(),
                     arms={"none": bl.none})
    body = text.split("per arm and closure")[1]
    assert "none / ended" in body and "none / retired" in body
