"""The mem0 competitor arm of the demo, which does not run unless it is asked for.

`demo/competitors.py` adds `mem0` to the answer-quality run. It is off by default because
it needs the `mem0ai` package, which the repository does not install. The arm exists so
that the comparison in `demo/README.md` can be made against another system on *answers*
rather than on architecture, which is what item 1 of the roadmap's "What is still
missing" asks for.

Everything here is offline. mem0 is replaced by `fake_mem0`, a module tree standing in for
the parts of `mem0` the arm imports, whose `Memory` keeps the behaviour the comparison
turns on: **every event is an ADD and nothing is ever superseded**, which is mem0 2.x's
documented design and is what `bench/mem0_real.py` measured. No test here opens a socket.

The failures these tests exist to prevent:

1. **An arm that is quietly generous to the competitor, or quietly mean to it.** mem0 is
   driven by the same ground-truth facts the structured memvara arm gets, so any difference
   between the two rows is architecture rather than extraction. If the oracle here drifted
   from `visible_facts`, the number would be measuring the harness.
2. **The firehose bug `bench/mem0_real.py` documents.** Its first oracle scanned the whole
   prompt for known turns, so mem0 re-extracted every earlier turn in its window and each
   fact arrived eleven times. The numbers flattered memvara. An oracle keyed on anything
   but the single turn being added can reintroduce it, so the count is asserted.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import timezone
from typing import Any, Mapping

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


# --- wiring into the harness ------------------------------------------------------


def _args(**kw: Any) -> Any:
    parser_args = {"memory": "local", "corpus_scale": 1, "arm_mem0": False}
    parser_args.update(kw)
    return type("Args", (), parser_args)()


def test_the_mem0_arm_does_not_appear_unless_its_flag_is_given():
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


def test_turning_on_mem0_without_the_package_fails_at_setup(monkeypatch):
    """The refusal lands while the arms are being built, not on the first question.

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

    Both failures are things the person running the command has to go and fix, such as
    installing a package, so the useful output is the sentence that says what to do.
    A traceback through the arm that happened to notice buries it.
    """
    assert issubclass(co.CompetitorUnavailable, SystemExit)
