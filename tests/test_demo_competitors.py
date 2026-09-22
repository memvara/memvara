"""The two competitor arms of the demo, neither of which runs unless it is asked for.

`demo/competitors.py` adds `mem0` and `supermemory` to the answer-quality run. Both are
off by default, and each needs something the repository cannot supply for it: mem0 needs
the `mem0ai` package installed, and Supermemory needs an account. The arms exist so that
the comparison in `demo/README.md` can be made against other systems on *answers* rather
than on architecture, which is what item 1 of the roadmap's "What is still missing" asks
for.

Everything here is offline. mem0 is replaced by `fake_mem0`, a module tree standing in for
the parts of `mem0` the arm imports, whose `Memory` keeps the behaviour the comparison
turns on: **every event is an ADD and nothing is ever superseded**, which is mem0 2.x's
documented design and is what `bench/mem0_real.py` measured. Supermemory is reached through
the same injectable `fetch` the importer in `memvara/compat/supermemory_import.py` uses, so
no test here opens a socket.

The failures these tests exist to prevent:

1. **An arm that is quietly generous to the competitor, or quietly mean to it.** mem0 is
   driven by the same ground-truth facts the structured memvara arm gets, so any difference
   between the two rows is architecture rather than extraction. If the oracle here drifted
   from `visible_facts`, the number would be measuring the harness.
2. **The firehose bug `bench/mem0_real.py` documents.** Its first oracle scanned the whole
   prompt for known turns, so mem0 re-extracted every earlier turn in its window and each
   fact arrived eleven times. The numbers flattered memvara. An oracle keyed on anything
   but the single turn being added can reintroduce it, so the count is asserted.
3. **A demo run that writes into somebody's real account.** The Supermemory arm refuses
   without an explicit container, for the same reason `demo/hosted.py` refuses this
   machine's own memvara credentials: a run writes hundreds of documents and nothing here
   can take them back.
4. **An endpoint this repository has never seen being presented as if it had.** The
   importer has only ever exercised `POST /v3/documents/list`. The arm therefore ships no
   default write or search path, and a test pins that, because a plausible-looking default
   is how a guess becomes a documented fact.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import pytest

from demo import baselines as bl  # noqa: E402  (puts bench/ on the path)
from demo import competitors as co  # noqa: E402
from demo import harness as hz  # noqa: E402
from demo import scenario  # noqa: E402

from memvara import Memvara  # noqa: E402

UTC = timezone.utc
QUESTIONS = list(scenario.questions())
TURNS = list(scenario.conversation())


# --- mem0, offline ---------------------------------------------------------------


class _FakeLLMBase:
    """`mem0.llms.base.LLMBase`: constructed with a config and otherwise inert."""

    def __init__(self, config: Any = None) -> None:
        self.config = config


class _FakeEmbeddingBase:
    """`mem0.embeddings.base.EmbeddingBase`, same shape."""

    def __init__(self, config: Any = None) -> None:
        self.config = config


@dataclass
class _FakeConfig:
    model: str = ""


class FakeMem0Memory:
    """`mem0.Memory`, keeping the one behaviour the comparison rests on.

    mem0 2.x's add path emits only `ADD` events — its extraction prompt says "Your sole
    operation is ADD" — so a new value is *linked* to the one it contradicts rather than
    retiring it, and both stay live. This stand-in does exactly that: every memory the
    oracle emits is appended, nothing is ever replaced, and `search` ranks the lot. That
    is what makes "mem0 served the superseded value" a finding about mem0's architecture
    rather than about this double.

    `built` counts constructions so a test can check the arm builds one store per question
    *instant* rather than one per question; twenty questions share three instants.
    """

    built = 0
    #: Every `add` call, as (user_id, text). Read by the oracle-count assertions.
    added: list[tuple[str, str]] = []

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        type(self).built += 1
        self.config = dict(config or {})
        self.rows: list[dict[str, Any]] = []
        self.llm: Any = None
        self.embedding_model: Any = None

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "FakeMem0Memory":
        return cls(config)

    def add(self, text: str, user_id: str = "", **_: Any) -> dict[str, Any]:
        type(self).added.append((user_id, text))
        raw = self.llm.generate_response(messages=[{"role": "user", "content": text}])
        emitted = json.loads(raw)["memory"]
        for item in emitted:
            assert item["event"] == "ADD", "mem0 2.x's add path emits ADD and only ADD"
            self.embedding_model.embed(item["text"])
            self.rows.append({"memory": item["text"], "user_id": user_id})
        return {"results": emitted}

    def search(self, query: str, filters: Mapping[str, Any] | None = None,
               top_k: int = 10, **_: Any) -> dict[str, Any]:
        user = (filters or {}).get("user_id")
        wanted = set(query.lower().split())
        mine = [r for r in self.rows if user is None or r["user_id"] == user]
        ranked = sorted(
            mine,
            key=lambda r: (-len(wanted & set(r["memory"].lower().split())), r["memory"]))
        return {"results": ranked[:top_k]}


@pytest.fixture
def fake_mem0(monkeypatch: pytest.MonkeyPatch) -> type[FakeMem0Memory]:
    """Install a `mem0` module tree, so the arm's own import path runs against it.

    The arm builds its oracle classes as subclasses of mem0's bases at call time, which is
    the only way a module that must import without mem0 installed can subclass anything
    from it. Faking the modules rather than injecting finished objects means that
    subclassing, the `from_config` call and the two provider swaps are all exercised here
    rather than skipped.
    """
    import types

    FakeMem0Memory.built = 0
    FakeMem0Memory.added = []

    root = types.ModuleType("mem0")
    root.Memory = FakeMem0Memory  # type: ignore[attr-defined]
    configs = types.ModuleType("mem0.configs")
    embeddings_cfg = types.ModuleType("mem0.configs.embeddings")
    embeddings_cfg_base = types.ModuleType("mem0.configs.embeddings.base")
    embeddings_cfg_base.BaseEmbedderConfig = _FakeConfig  # type: ignore[attr-defined]
    llms_cfg = types.ModuleType("mem0.configs.llms")
    llms_cfg_base = types.ModuleType("mem0.configs.llms.base")
    llms_cfg_base.BaseLlmConfig = _FakeConfig  # type: ignore[attr-defined]
    embeddings = types.ModuleType("mem0.embeddings")
    embeddings_base = types.ModuleType("mem0.embeddings.base")
    embeddings_base.EmbeddingBase = _FakeEmbeddingBase  # type: ignore[attr-defined]
    llms = types.ModuleType("mem0.llms")
    llms_base = types.ModuleType("mem0.llms.base")
    llms_base.LLMBase = _FakeLLMBase  # type: ignore[attr-defined]

    for name, module in [
        ("mem0", root), ("mem0.configs", configs),
        ("mem0.configs.embeddings", embeddings_cfg),
        ("mem0.configs.embeddings.base", embeddings_cfg_base),
        ("mem0.configs.llms", llms_cfg), ("mem0.configs.llms.base", llms_cfg_base),
        ("mem0.embeddings", embeddings), ("mem0.embeddings.base", embeddings_base),
        ("mem0.llms", llms), ("mem0.llms.base", llms_base),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    return FakeMem0Memory


def _question(qid: str) -> Any:
    return next(q for q in QUESTIONS if q.id == qid)


# --- the mem0 arm gets what the structured memvara arm gets -----------------------


def test_the_mem0_arm_is_offered_exactly_the_facts_the_structured_arm_is_offered(
        fake_mem0):
    """The same ground truth on both sides, which is what isolates architecture.

    `memvara_structured` is given `visible_facts(question, SUPPORT_FACTS)` — every fact
    the desk had recorded when the question was asked, and no fact recorded after it. The
    mem0 arm is given the same list through mem0's own add path, with the oracle standing
    in for extraction exactly as `bench/mem0_real.py` does. If these two sets ever
    differed, the gap between the two rows of the report would be partly a difference in
    what each system was told, and the report has no column that would show it.
    """
    question = _question("q_plan_current")
    arm = co.Mem0Arm()
    arm(question, TURNS)

    want = [f.text for f in bl.visible_facts(question, bl.SUPPORT_FACTS)]
    store = arm.stores[question.asked_at]
    assert sorted(r["memory"] for r in store.rows) == sorted(want)


def test_a_fact_recorded_after_the_question_was_asked_never_reaches_mem0(fake_mem0):
    """The `asked_at` cutoff binds this arm too, on transaction time.

    The account moved to Pro on 3 March and back to Home on 19 June. A question asked in
    between must not see the June downgrade, and the way that is enforced here is the same
    way `demo/baselines.py` enforces it: the input is truncated, rather than the store
    being read with a `known_at=`.

    The check is a multiset comparison rather than a set one, and that is not fussiness.
    Two writes in this table carry the *same sentence* — the account is on the Home plan
    in January, moves to Pro in March and moves back to Home in June — so a set would
    quietly forgive both a missing write and a duplicated one. The sentences that only a
    later write introduces are then checked separately, which is the part that is actually
    about the cutoff.
    """
    early = min(q.asked_at for q in QUESTIONS)
    question = next(q for q in QUESTIONS if q.asked_at == early)
    arm = co.Mem0Arm()
    arm(question, TURNS)

    stored = [r["memory"] for r in arm.stores[question.asked_at].rows]
    visible = bl.visible_facts(question, bl.SUPPORT_FACTS)
    assert sorted(stored) == sorted(f.text for f in visible)

    later = [f for f in bl.SUPPORT_FACTS if f.at > question.asked_at]
    assert later, "the corpus must have facts after the earliest question, or this is vacuous"
    only_later = {f.text for f in later} - {f.text for f in visible}
    assert only_later, "at least one later write must say something new, or this is vacuous"
    for text in only_later:
        assert text not in stored, f"{text!r} was recorded after the question was asked"


def test_each_fact_is_emitted_once_rather_than_once_per_turn_in_the_window(fake_mem0):
    """The bug `bench/mem0_real.py` documents, asserted so it cannot come back.

    That file's first oracle matched known turns anywhere in the prompt. mem0's additive
    extraction prompt embeds the last k messages, so every earlier turn in the window
    matched again and each fact was emitted eleven times. mem0 was then measured under a
    firehose no real extractor would produce, and the resulting numbers flattered memvara.

    The oracle here keys on the instant of the single turn being added, and turn instants
    are unique in this corpus, so each fact is emitted exactly once. Counting rows rather
    than inspecting the oracle is deliberate: it is the stored result that would be wrong,
    and a count catches any future route to the same wrongness.

    The count is against `visible_facts`, not against the number of *distinct* sentences.
    One sentence is genuinely written twice — the plan is Home in January, Pro in March
    and Home again in June — so seventeen writes carry sixteen distinct sentences, and a
    uniqueness assertion would fail on correct behaviour. That repeat is also the clearest
    statement of what this arm is for: memvara supersedes and holds one plan, mem0 appends
    and holds three.
    """
    question = _question("q_plan_current")
    arm = co.Mem0Arm()
    arm(question, TURNS)

    rows = [r["memory"] for r in arm.stores[question.asked_at].rows]
    visible = bl.visible_facts(question, bl.SUPPORT_FACTS)
    assert len(rows) == len(visible), "a fact reached mem0 a different number of times"
    assert sorted(rows) == sorted(f.text for f in visible)

    plans = [r for r in rows if "plan." in r]
    assert len(plans) == 3, "the three plan writes are what mem0 holds all of"


def test_mem0_keeps_the_superseded_plan_beside_the_current_one(fake_mem0):
    """The finding the arm exists to show, and the reason it is not a bug here.

    The account was on Home, moved to Pro on 3 March, and moved back to Home on 19 June.
    A store that supersedes holds one live plan; mem0 2.x holds every value it was ever
    told, because its add path emits only `ADD`. Both plan sentences must therefore be in
    the context this arm builds. That is not a defect in the stand-in — it is the
    documented design, and it is exactly what the trap column of the report counts.
    """
    question = _question("q_plan_current")
    arm = co.Mem0Arm()
    context = arm(question, TURNS)

    assert "Pro plan" in context.text, "the superseded value should still be served"
    assert "Home plan" in context.text, "the current value should be served too"


# --- the mem0 arm spends the same budget as the other retrieval arms --------------


def test_the_mem0_context_is_capped_exactly_like_the_other_retrieval_arms(fake_mem0):
    """Same character cap, same slot count, or the comparison is between two budgets.

    `MAX_CONTEXT_CHARS` and `DEFAULT_K` are the two constants that keep the five arms one
    experiment rather than five. An arm allowed a larger prompt would win on prompt size
    and the table would read as if it had won on memory.
    """
    question = _question("q_plan_current")
    context = co.Mem0Arm(max_chars=200)(question, TURNS)
    assert context.chars <= 200
    assert context.items_used == bl.count_entries(context.text)


def test_the_mem0_block_carries_the_same_header_as_a_memvara_claim_block(fake_mem0):
    """Two stored-note blocks that differ in shape would unblind the reader.

    The harness shuffles items and hides which arm produced which prompt. That blinding is
    only as good as the prompts look alike, so a block of stored notes from mem0 uses the
    header `recall()` uses — which also carries the framing that tells a model the lines
    below are reference data rather than instructions. The arms then differ in *which*
    values they hold, which is the only difference the run is meant to measure.
    """
    question = _question("q_plan_current")
    context = co.Mem0Arm()(question, TURNS)
    assert context.text.startswith(Memvara.RECALL_HEADER)
    assert all(line.startswith("- ") for line in context.text.splitlines()[1:])


def test_one_store_is_built_per_question_instant_rather_than_per_question(fake_mem0):
    """Twenty questions share three instants, so three stores answer all of them.

    A mem0 store is expensive to fill — one add call per turn, each with an embed — and
    what a store may hold depends only on `asked_at`, never on the question's wording.
    `demo/hosted.py` makes the same observation about hosted scopes for the same reason.
    Rebuilding per question would multiply the work by roughly seven and change nothing
    about the contexts.
    """
    arm = co.Mem0Arm()
    for question in QUESTIONS:
        arm(question, TURNS)

    instants = {q.asked_at for q in QUESTIONS}
    assert len(instants) < len(QUESTIONS), "the corpus must share instants, or this is vacuous"
    assert fake_mem0.built == len(instants)


def test_the_mem0_arm_reports_the_turns_the_question_could_see(fake_mem0):
    """`turns_visible` means the same thing on every row of the size table.

    It is the number of turns at or before `asked_at`, identical for every arm on a given
    question — that is what makes the gap between it and `items_used` readable as the
    compression a memory layer is being paid for.
    """
    question = _question("q_plan_current")
    context = co.Mem0Arm()(question, TURNS)
    assert context.arm == "mem0"
    assert context.turns_visible == len(bl.visible_turns(question, TURNS))


# --- the mem0 arm when mem0 is not installed --------------------------------------


def test_the_mem0_arm_refuses_with_the_command_that_fixes_it(monkeypatch):
    """A missing optional dependency must say what to install, not raise ImportError.

    `mem0` is forced absent rather than assumed absent: this suite has to give the same
    answer on a machine where somebody has installed `mem0ai` for `bench/mem0_real.py`.
    Setting the module to None in `sys.modules` is what makes `import mem0` raise
    ImportError deterministically.
    """
    monkeypatch.setitem(sys.modules, "mem0", None)
    with pytest.raises(co.CompetitorUnavailable) as caught:
        co.Mem0Arm()(_question("q_plan_current"), TURNS)
    assert "pip install mem0ai" in str(caught.value)


def test_the_mem0_arm_is_not_built_until_it_is_used(monkeypatch):
    """Constructing the arm must not import mem0, so `--help` works without it.

    The import is deferred to the first context, which is also what lets
    `demo/competitors.py` be imported by `demo/harness.py` unconditionally.
    """
    monkeypatch.setitem(sys.modules, "mem0", None)
    co.Mem0Arm()  # must not raise


# --- Supermemory, offline ---------------------------------------------------------


class FakeSupermemory:
    """A Supermemory account behind the importer's `fetch` shape: (url, key, body).

    Reusing that signature rather than inventing a second one is the point: it is already
    how `memvara/compat/supermemory_import.py` is tested and how a caller routes through
    their own HTTP client. This records what was sent so the tests can assert on the
    request rather than on the arm's intentions.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Mapping[str, Any]]] = []
        self.documents: list[Mapping[str, Any]] = []

    def __call__(self, url: str, key: str, body: Mapping[str, Any]) -> Mapping[str, Any]:
        self.calls.append((url, key, dict(body)))
        if url.endswith("/ingest"):
            self.documents.append(body)
            return {"id": f"doc{len(self.documents)}"}
        # Search honours `containerTags`, which is the whole point of sending them. A
        # double that ignored the tag and returned every document would pass a test for
        # an arm that mixed two question instants in one container — the exact bug this
        # file's last section is about.
        tags = set(body.get("containerTags") or ())
        mine = [d for d in self.documents
                if not tags or tags & set(d.get("containerTags") or ())]
        wanted = set(str(body.get("q", "")).lower().split())
        ranked = sorted(
            mine,
            key=lambda d: (-len(wanted & set(str(d["content"]).lower().split())),
                           str(d["content"])))
        limit = int(body.get("limit", 10))
        return {"results": [{"content": d["content"]} for d in ranked[:limit]]}


def _supermemory(fetch: Any, **kw: Any) -> co.SupermemoryArm:
    settings = {"api_key": "sm-test-key", "container": "memvara-demo",
                "ingest_path": "/ingest", "search_path": "/search", "fetch": fetch}
    settings.update(kw)
    return co.SupermemoryArm(**settings)


def test_the_supermemory_arm_names_every_missing_requirement_at_once(monkeypatch):
    """One refusal listing four things beats four runs each finding the next one.

    Every requirement is checked before anything is sent, so a person configuring the arm
    learns the whole list in one go. The message must name the key, the container and both
    endpoint paths.
    """
    with pytest.raises(co.CompetitorUnavailable) as caught:
        co.SupermemoryArm(api_key="", container="", ingest_path="", search_path="")
    message = str(caught.value)
    for needed in ("api key", "container", "ingest", "search"):
        assert needed in message.lower(), f"{needed} is not named in the refusal"


def test_the_supermemory_arm_refuses_without_a_container(monkeypatch):
    """A run must not be able to land in whatever space the account defaults to.

    `demo/hosted.py` refuses this machine's own memvara credentials for the same reason,
    and the reason is the same size: a scale-10 run writes six hundred documents, and
    nothing in this repository can take them back out of somebody's account.
    """
    with pytest.raises(co.CompetitorUnavailable) as caught:
        co.SupermemoryArm(api_key="sm-test-key", container="",
                          ingest_path="/ingest", search_path="/search")
    assert "container" in str(caught.value).lower()


def test_the_repository_ships_no_default_write_or_search_endpoint():
    """The honesty claim of the Supermemory arm, pinned as a test.

    The importer has exercised exactly one endpoint, `POST /v3/documents/list`. Nobody
    here has an account, so nobody here has seen a write or a query succeed. A default
    path that looked plausible would turn a guess into something a reader would quote, so
    there is none, and this is where that stays true.
    """
    assert co.SupermemoryArm.DEFAULT_INGEST_PATH is None
    assert co.SupermemoryArm.DEFAULT_SEARCH_PATH is None
    assert co.SUPERMEMORY_VERIFIED_PATH == "/v3/documents/list"


def test_the_supermemory_arm_ingests_only_what_the_question_may_see():
    """The `asked_at` cutoff again, on the arm that sends its corpus over a wire.

    Every turn at or before the question's instant is sent, and none after it. The count
    is compared against `visible_turns`, the same function every other arm uses, rather
    than against a number written here.
    """
    fetch = FakeSupermemory()
    question = _question("q_plan_current")
    _supermemory(fetch)(question, TURNS)

    ingests = [body for url, _, body in fetch.calls if url.endswith("/ingest")]
    assert len(ingests) == len(bl.visible_turns(question, TURNS))
    latest = max(t.at for t in bl.visible_turns(question, TURNS))
    assert latest <= question.asked_at


def test_the_supermemory_arm_tags_every_document_with_its_container():
    """Isolation is per document, not a setting made once.

    The container tag is what keeps a demo run separable from the rest of an account, so
    it has to be on every document written; one untagged document is one document that
    cannot be found and removed afterwards. The tag is the run's tag plus the question
    instant — see `container_for`.
    """
    fetch = FakeSupermemory()
    question = _question("q_plan_current")
    arm = _supermemory(fetch)
    arm(question, TURNS)
    ingests = [body for url, _, body in fetch.calls if url.endswith("/ingest")]
    assert ingests, "nothing was ingested"
    for body in ingests:
        assert body["containerTags"] == [arm.container_for(question)]
        assert body["containerTags"][0].startswith("memvara-demo")


def test_a_question_never_searches_a_container_holding_turns_from_after_it():
    """The cutoff, against the order the harness actually asks questions in.

    `demo/harness.py` iterates questions as `demo/scenario.py` lists them, and that is not
    `asked_at` order — the August questions come first and the April one is ninth. So this
    fills the latest instant first, exactly as a real run does, and only then asks the
    earliest question. With one container shared by every instant the April question would
    search a store already holding the whole history and would read turns from four months
    after it was asked, which no other arm can do and nothing in the report would show.

    Asserted on the documents the search could match rather than on the container's name,
    because the name is the mechanism and the cutoff is the property.
    """
    fetch = FakeSupermemory()
    arm = _supermemory(fetch)
    latest = max(QUESTIONS, key=lambda q: q.asked_at)
    earliest = min(QUESTIONS, key=lambda q: q.asked_at)
    assert latest.asked_at > earliest.asked_at, "the corpus must span instants"

    arm(latest, TURNS)
    arm(earliest, TURNS)

    tag = arm.container_for(earliest)
    visible = [d for d in fetch.documents if tag in (d.get("containerTags") or ())]
    assert visible, "the earliest question's container holds nothing"
    cutoff = f"[{earliest.asked_at:%Y-%m-%d}]"
    for document in visible:
        stamp = str(document["content"])[:12]
        assert stamp <= cutoff, f"{stamp} is after the question was asked"
    assert len(visible) == len(bl.visible_turns(earliest, TURNS))


def test_the_supermemory_key_travels_as_the_fetch_argument_and_not_in_a_body():
    """The key is handed to the transport, never written into a request body.

    `_http_fetch` puts it in an Authorization header. Anything that also copied it into
    the JSON body would put a live credential into every log and every captured response
    that a run leaves behind.
    """
    fetch = FakeSupermemory()
    _supermemory(fetch)(_question("q_plan_current"), TURNS)
    for _, key, body in fetch.calls:
        assert key == "sm-test-key"
        assert "sm-test-key" not in json.dumps(body)


def test_the_supermemory_context_is_built_from_what_the_search_returned():
    """The arm renders the documents the service chose, under the same cap as the rest."""
    fetch = FakeSupermemory()
    question = _question("q_plan_current")
    context = _supermemory(fetch, max_chars=300)(question, TURNS)

    assert context.arm == "supermemory"
    assert context.chars <= 300
    assert context.turns_visible == len(bl.visible_turns(question, TURNS))
    assert context.items_used == bl.count_entries(context.text)


def test_one_supermemory_container_is_filled_per_question_instant():
    """The same instant-sharing observation as mem0, for the same reason.

    Writing the corpus again for each of twenty questions would send seven times the
    documents into somebody's account for no change in any context.
    """
    fetch = FakeSupermemory()
    arm = _supermemory(fetch)
    for question in QUESTIONS:
        arm(question, TURNS)

    ingests = [b for url, _, b in fetch.calls if url.endswith("/ingest")]
    expected = sum(len(bl.visible_turns(q, TURNS))
                   for q in {q.asked_at: q for q in QUESTIONS}.values())
    assert len(ingests) == expected


# --- wiring into the harness ------------------------------------------------------


def _args(**kw: Any) -> Any:
    parser_args = {"memory": "local", "corpus_scale": 1, "arm_mem0": False,
                   "arm_supermemory": False, "supermemory_key_file": None,
                   "supermemory_container": None, "supermemory_base_url": co.SUPERMEMORY_BASE_URL,
                   "supermemory_ingest_path": None, "supermemory_search_path": None}
    parser_args.update(kw)
    return type("Args", (), parser_args)()


def test_neither_arm_appears_unless_its_flag_is_given():
    """Off by default, and the default run is the five arms it has always been.

    `test_the_offline_run_is_identical_twice` pins the stub report byte for byte, so an
    arm that appeared without being asked for would be caught there too — but it would be
    caught as a mysterious diff. This says the rule directly.
    """
    arms, notes = co.build_competitors(_args())
    assert arms == {}
    assert notes == []


def test_the_competitor_arms_are_ordered_between_the_rag_arm_and_the_product(fake_mem0):
    """Reporting order is floor, ceiling, competitors, product — and the table follows it.

    `ARMS` is ordered deliberately and its order is the report's. A competitor appended
    after `memvara_structured` would put the product in the middle of the table and the
    competition at the end, which reads as an afterthought rather than as the control it
    is.
    """
    arms, _ = hz.build_arms(_args(arm_mem0=True))
    assert list(arms) == ["none", "full_transcript", "naive_rag", "mem0", "memvara",
                          "memvara_structured"]


def test_the_report_title_counts_the_arms_that_actually_ran(fake_mem0):
    """A six-arm run must not head its own report "five-arm".

    The title is the first line anybody reads and the one most likely to be quoted out of
    the report. It was a constant, which was true for as long as there were exactly five
    arms. The default run still says "five", which is also what keeps the pinned offline
    report byte-identical.
    """
    questions = QUESTIONS[:1]
    five = hz.offline(questions, TURNS)
    assert "five-arm answer quality" in five.report()

    arms, _ = hz.build_arms(_args(arm_mem0=True))
    six = hz.offline(questions, TURNS, arms=arms)
    assert "six-arm answer quality" in six.report(arms=arms)


def test_the_report_says_which_optional_arms_ran(fake_mem0):
    """A run with a competitor in it has to be quotable, which means naming it.

    The header already names the floor, the ceiling and the measurement arms. An optional
    arm adds a line saying how it was configured, because "mem0" alone does not say which
    mem0, and a number quoted without that cannot be repeated.
    """
    _, notes = co.build_competitors(_args(arm_mem0=True))
    assert any("mem0" in line for line in notes)


def test_turning_on_supermemory_without_its_settings_fails_at_setup(fake_mem0):
    """The refusal lands before any arm has run, not on the first question.

    A run that got as far as building contexts and then failed would have already written
    the mem0 corpus and spent the reader's first calls. Everything an arm needs is checked
    while the arms are being built.
    """
    with pytest.raises(co.CompetitorUnavailable):
        co.build_competitors(_args(arm_supermemory=True))


def test_the_cli_path_also_names_everything_missing_in_one_refusal(tmp_path):
    """Including the key, which is read before the arm is constructed.

    `build_competitors` has to read the key file to build the arm at all, so a failure
    there could easily have been raised on the spot — and then a person with none of the
    four settings would have learned about the key on the first run and about the
    container and the two paths on the second. The reason is carried into the arm instead
    and takes its place in the one list.
    """
    missing_key = tmp_path / "no-such-key.json"
    with pytest.raises(co.CompetitorUnavailable) as caught:
        co.build_competitors(_args(arm_supermemory=True,
                                   supermemory_key_file=str(missing_key)))
    message = str(caught.value).lower()
    for needed in ("api key", "container", "ingest", "search"):
        assert needed in message, f"{needed} is not named in the refusal"
    assert str(missing_key).lower() in message, "the refusal must name the file it tried"


def test_turning_on_mem0_without_the_package_fails_at_setup(monkeypatch):
    """The same rule for the arm whose dependency is a package rather than an account.

    mem0 is imported lazily, so that `demo/competitors.py` imports on a checkout that has
    never heard of it. Left alone, that would push the failure into `plan()` — after the
    run had started. `build_competitors` therefore imports it once while the arms are
    being built, which is the only place a missing dependency is cheap to report.
    """
    monkeypatch.setitem(sys.modules, "mem0", None)
    with pytest.raises(co.CompetitorUnavailable) as caught:
        co.build_competitors(_args(arm_mem0=True))
    assert "pip install mem0ai" in str(caught.value)


def test_a_refusal_is_a_clean_exit_rather_than_a_traceback():
    """`demo/hosted.py` refuses a bad credential with `SystemExit`, and so does this.

    Both failures are things the person running the command has to go and fix — install a
    package, make an account — so the useful output is the sentence that says what to do.
    A traceback through the arm that happened to notice buries it.
    """
    assert issubclass(co.CompetitorUnavailable, SystemExit)
