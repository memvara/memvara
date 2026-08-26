"""The five-arm demo comparison: the arms, the blinding, and the scoring.

`demo/baselines.py` builds the context each approach would put in front of a reader, and
`demo/harness.py` dumps all five into one blinded file and scores what comes back. The
failures these tests exist to prevent, in order of how expensive they would be:

1. **A memvara arm seeing the future.** The cutoff is applied at ingest, and if it ever
   stopped being applied the product's own arms would answer July's questions with
   September's turns and the whole result would silently inflate. It is pinned once per
   memvara arm, and each of those tests removes the cutoff inside the test and asserts the
   leak reappears — a test that cannot fail against the bug it names is not evidence.
2. **`memvara_structured` closing the wrong clock.** An arm that retires everything, or
   ends everything, scores about the same and means nothing: the ended/retired split is
   the one thing this library has that a store with a single clock does not.
   `test_a_correction_recorded_as_a_change_...` and
   `test_a_change_recorded_as_a_correction_...` each mutate one closure and assert the
   damage is visible.
3. **The arms drifting into different experiments.** `default_embedder()` returns a
   sentence-transformers model whenever that package is importable, and it is importable
   here, so an unpinned embedder would make `naive_rag` and both memvara arms a different
   comparison on a different machine.
4. **The blinding leaking.** Nothing in the dumped prompt may name the arm, the question
   id, the kind, the gold answer or the trap.

The corpus in the first half of this file is its own, deliberately. `demo/scenario.py`
belongs to another workstream and the *mechanics* are written against its published
contract rather than against its contents, so a test that asserted on its wording would
fail every time that file was edited, for reasons having nothing to do with the code under
test.

The last section is the exception, and it has to be. `memvara_structured` writes *this
corpus's* facts, so its correctness is not expressible on a fixture: the claim being made
is that the library returns the authored answer, and only the authored answer set can
check it. Those tests read `demo/scenario.py` directly and assert against `Question.gold`
with no reader in the loop — which is stronger evidence than an answer-quality percentage,
because it tests the library rather than the library plus a model.
"""

from __future__ import annotations

import ast
import inspect
import json
import textwrap
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

# `demo` is a package and is imported as one, so there is exactly one module object per
# file — a bare `import baselines` here would load a second copy under a second name, and
# a monkeypatch in this file would then apply to a module the harness is not using.
#
# `bench/` is not a package, so `evalkit` is imported by bare name as its own runners do.
# It has to come *after* the demo import rather than before: `demo.baselines` is what puts
# `bench/` on the path, which is deliberate — the demo declaring its own dependency is
# better than every caller having to know about it.
from demo import baselines as bl  # noqa: E402
from demo import harness as hz  # noqa: E402

import evalkit as ek  # noqa: E402

from memvara import Memvara, PredicateRegistry, PredicateSpec  # noqa: E402
from memvara.schema import (  # noqa: E402
    BUILTIN_PREDICATES,
    Cardinality,
    Volatility,
)

UTC = timezone.utc


# --- the contract, restated ------------------------------------------------------
#
# Field-for-field what `demo/scenario.py` exports. Restated rather than imported so these
# tests pin the *contract* the arms were written against; if that file lands with a
# different shape, the failure should be an import error in the demo, not a silent
# disagreement discovered during a run.


@dataclass(frozen=True)
class Turn:
    at: datetime
    role: str
    text: str


@dataclass(frozen=True)
class Question:
    id: str
    asked_at: datetime
    text: str
    gold: str
    trap: str | None
    kind: str
    #: The valid-time instant the question asks about. Only `memvara_structured` reads it.
    #: Defaulted so every fixture question below stays a positional construction.
    about: datetime | None = None


def at(month: int, day: int, hour: int = 9) -> datetime:
    return datetime(2026, month, day, hour, tzinfo=UTC)


#: A support history where four things change: the plan goes up then back down, the
#: address moves, a preference reverses, and one fact is corrected because the customer
#: was *wrong* rather than because the world changed. Long enough that `naive_rag` at
#: k=12 is a real retrieval arm rather than the transcript in a different order.
CONVERSATION: list[Turn] = [
    Turn(at(1, 12), "user", "My name is Dana Whitfield and I live in Portland."),
    Turn(at(1, 12, 10), "assistant", "Thanks Dana, I have Portland on the account."),
    Turn(at(1, 13), "user", "My account number is 55120."),
    Turn(at(2, 3), "user", "I would like to upgrade to the Pro plan please."),
    Turn(at(2, 3, 10), "assistant", "Done, you are on the Pro plan from today."),
    Turn(at(2, 20), "user", "The invoice looks wrong, it charged me twice."),
    Turn(at(2, 20, 11), "assistant", "Refunded the duplicate charge, sorry about that."),
    Turn(at(3, 9), "user", "I prefer email over phone for support."),
    Turn(at(3, 9, 10), "assistant", "Noted, email it is."),
    Turn(at(3, 30), "user", "Sorry, my account number is actually 55210, I misread it."),
    Turn(at(4, 21), "user", "I just moved to Seattle, please update my address."),
    Turn(at(4, 21, 10), "assistant", "Updated, your address is now in Seattle."),
    Turn(at(5, 8), "user", "Can you send the March invoice again?"),
    Turn(at(5, 8, 10), "assistant", "Sent to the address on file."),
    Turn(at(6, 2), "user", "Actually, downgrade me back to the Starter plan."),
    Turn(at(6, 2, 10), "assistant", "You are back on the Starter plan now."),
    Turn(at(7, 14), "user", "Actually I prefer phone now, email is too slow."),
    Turn(at(7, 14, 10), "assistant", "Switched your contact preference to phone."),
]

QUESTIONS: list[Question] = [
    Question("q_plan_now", at(8, 1), "What plan is the customer on right now?",
             "Starter", "Pro", "current"),
    Question("q_plan_march", at(8, 1), "What plan was the customer on in March?",
             "Pro", "Starter", "historical", about=at(3, 15)),
    Question("q_city", at(8, 1), "What city does the customer live in?",
             "Seattle", "Portland", "current"),
    Question("q_account", at(8, 1), "What is the customer's account number?",
             "55210", "55120", "correction"),
    Question("q_contact", at(8, 1), "How does the customer prefer to be contacted?",
             "phone", "email", "current"),
    Question("q_unknown", at(8, 1), "What is the customer's date of birth?",
             "not stated", None, "unanswerable"),
    #: Asked before the downgrade, so the correct answer is the value that is superseded
    #: by the time the run happens. This is the question that catches an arm reading the
    #: future: anything that answers "Starter" here is using turns it should not have.
    Question("q_plan_in_may", at(5, 20), "What plan is the customer on right now?",
             "Pro", "Starter", "current"),
]


# --- the structured arm's fixture integration ------------------------------------
#
# `memvara_structured` writes the desk's own fields, so unlike the other four arms it
# cannot be exercised without a fact table. This is the fixture corpus's, mirroring the
# real one's shape in miniature: two `ended` slots, one `retired`, one control, and a
# predicate whose cardinality is load-bearing. The mechanics — the cutoff, the two
# closures, the registry — are tested against these, and only the answer-key assertions at
# the end of the file use the real corpus.

FIXTURE_PREDICATES: tuple[PredicateSpec, ...] = (
    PredicateSpec("plan", Cardinality.ONE, Volatility.SLOW),
    PredicateSpec("account_number", Cardinality.ONE, Volatility.STATIC),
    PredicateSpec("contact_preference", Cardinality.ONE, Volatility.SLOW),
)
#: `lives_in` is deliberately not declared: it is already `ONE` in `BUILTIN_PREDICATES`,
#: and keeping the builtins is what lets a deployment declare only what it adds.
FIXTURE_REGISTRY = PredicateRegistry(BUILTIN_PREDICATES + FIXTURE_PREDICATES)

FIXTURE_FACTS: tuple[bl.Write, ...] = (
    bl.Write(at(1, 12), "account", "lives_in", "Portland",
             "The customer lives in Portland."),
    bl.Write(at(1, 13), "account", "account_number", "55120",
             "The account number is 55120."),
    bl.Write(at(2, 3), "account", "plan", "Pro",
             "The account is on the Pro plan."),
    bl.Write(at(3, 9), "account", "contact_preference", "email",
             "The customer prefers to be contacted by email."),
    # RETIRED. "I misread it" — 55120 was never the account number, so its valid interval
    # is left as written, belief in it stops on 30 March, and `since` backdates the right
    # number to the day the wrong one was taken down.
    bl.Write(at(3, 30), "account", "account_number", "55210",
             "The account number is 55210.", since=at(1, 13), mode="correct"),
    # ENDED. Portland was true and stopped being true.
    bl.Write(at(4, 21), "account", "lives_in", "Seattle",
             "The customer lives in Seattle."),
    bl.Write(at(6, 2), "account", "plan", "Starter",
             "The account is on the Starter plan."),
    bl.Write(at(7, 14), "account", "contact_preference", "phone",
             "The customer prefers to be contacted by phone."),
)


def structured(question: Question, **kw) -> bl.Context:
    """`memvara_structured` on the fixture corpus and the fixture facts."""
    kw.setdefault("facts", FIXTURE_FACTS)
    kw.setdefault("registry", FIXTURE_REGISTRY)
    return bl.memvara_structured(question, CONVERSATION, **kw)


def fixture_store(question: Question, *, facts=FIXTURE_FACTS,
                  registry=FIXTURE_REGISTRY) -> Memvara:
    """The store the arm reads from, so a test can inspect it rather than the rendering.

    Same construction as `memvara_structured`, and it has to stay that way: a test that
    built the store differently from the arm would be checking a store nothing ships.
    """
    mem, _ = bl.build_memory(bl.visible_turns(question, CONVERSATION),
                             max_episodes=bl.DEFAULT_K, registry=registry)
    bl.apply_facts(mem, bl.visible_facts(question, facts))
    return mem


# --- visible_turns: the cutoff every arm shares ---------------------------------


def test_visible_turns_is_inclusive_of_the_moment_the_question_was_asked():
    """A turn at exactly `asked_at` happened before the question, not after it.

    An exclusive bound would make the boundary depend on clock resolution: the same
    corpus timestamped to the second and to the microsecond would give different
    contexts, and the difference would show up as an unexplained quality delta.
    """
    q = Question("q", at(3, 9, 10), "?", "g", None, "current")
    seen = bl.visible_turns(q, CONVERSATION)
    assert seen[-1].text == "Noted, email it is."
    assert all(t.at <= q.asked_at for t in seen)


def test_visible_turns_sorts_rather_than_trusting_the_order_it_was_given():
    """`items_used` only means anything if this function defines the order."""
    shuffled = [CONVERSATION[5], CONVERSATION[0], CONVERSATION[2]]
    q = Question("q", at(12, 1), "?", "g", None, "current")
    assert [t.at for t in bl.visible_turns(q, shuffled)] == sorted(t.at for t in shuffled)


def test_every_arm_sees_the_same_turns_so_they_differ_in_reasoning_not_in_access():
    """Applying the cutoff to memvara alone would make the comparison meaningless.

    The other three arms would be answering a May question with July's turns, which does
    not make them stronger competitors — it makes the delta between the arms a measure of
    who was allowed to cheat.
    """
    q = QUESTIONS[-1]  # asked 2026-05-20
    visible = {arm(q, CONVERSATION).turns_visible for arm in bl.ARMS.values()}
    assert visible == {len(bl.visible_turns(q, CONVERSATION))}
    assert "Starter" not in bl.full_transcript(q, CONVERSATION).text
    assert "Starter" not in bl.naive_rag(q, CONVERSATION).text


# --- arm 1: none ----------------------------------------------------------------


def test_the_none_arm_supplies_no_context_at_all_which_is_what_makes_it_the_floor():
    """A question this arm gets right was answerable without any memory system.

    It still reports `turns_visible`, because a floor arm that could not say how much
    history it declined to use would be indistinguishable from a broken arm.
    """
    ctx = bl.none(QUESTIONS[0], CONVERSATION)
    assert ctx.text == ""
    assert ctx.chars == 0 and ctx.items_used == 0
    assert ctx.turns_visible == len(CONVERSATION)


# --- arm 2: full_transcript -----------------------------------------------------


def test_the_full_transcript_arm_carries_every_visible_turn_in_date_order_with_dates():
    """Dates are given, not withheld.

    A reader that cannot see when something was said cannot distinguish a superseded
    value from a current one, so stripping the dates would win the comparison for memvara
    by disabling the competition. The competitor has to be the strong version of itself
    or the result proves nothing.
    """
    ctx = bl.full_transcript(QUESTIONS[0], CONVERSATION)
    lines = ctx.text.splitlines()
    assert len(lines) == len(CONVERSATION)
    assert lines[0].startswith("- [2026-01-12] user: My name is Dana")
    assert "[2026-06-02]" in ctx.text and "[2026-02-03]" in ctx.text
    assert ctx.items_used == ctx.turns_visible == len(CONVERSATION)


def test_the_full_transcript_arm_is_not_capped_because_its_size_is_the_finding():
    """Capping it would make it a worse `naive_rag` and hide the number that matters."""
    long_turns = [Turn(at(1, 1 + i % 27), "user", "x" * 500) for i in range(40)]
    ctx = bl.full_transcript(QUESTIONS[0], long_turns)
    assert ctx.chars > bl.MAX_CONTEXT_CHARS * 2
    assert ctx.approx_tokens == ctx.chars // bl.CHARS_PER_TOKEN


# --- arm 3: naive_rag -----------------------------------------------------------


def test_the_naive_rag_arm_returns_ranked_order_not_chronological_order():
    """Re-sorting by date would hand this arm the temporal signal under test.

    The whole point of the arm is a system with no notion of time, so its context has to
    arrive in the order a vector store would return it.
    """
    q = Question("q", at(8, 1), "Which plan is the account on?", "Starter", "Pro",
                 "current")
    ctx = bl.naive_rag(q, CONVERSATION, k=4)
    dates = [line[3:13] for line in ctx.text.splitlines()]
    assert dates != sorted(dates)
    assert ctx.items_used == 4


def test_the_naive_rag_arm_is_capped_in_both_slots_and_characters():
    """A retrieval arm that can grow without bound is a long-context arm in disguise.

    Same argument, and the same numbers, as `evalkit.RetrievalBudget`: the cap is what
    makes "the memory layer used a fraction of the transcript" a fact about the run
    rather than a hope.
    """
    fat = [Turn(at(1, 1 + i % 27), "user", f"plan detail {i} " + "y" * 400)
           for i in range(30)]
    q = Question("q", at(8, 1), "plan detail", "g", None, "current")

    # The slot cap alone, with room for every slot it selects.
    roomy = bl.naive_rag(q, fat, k=6, max_chars=bl.MAX_CONTEXT_CHARS)
    assert roomy.items_used == 6 and roomy.turns_visible == 30

    # The character cap on top, which is the one that decides what actually reaches the
    # reader. `items_used` counts the survivors, not the selection: reporting six here
    # would say the arm spent six slots on a prompt that holds two.
    tight = bl.naive_rag(q, fat, k=6, max_chars=500)
    assert tight.chars <= 500
    assert tight.items_used == 2


def test_the_naive_rag_arm_returns_nothing_when_no_turn_had_happened_yet():
    """A question asked before the conversation starts has no context, not an exception."""
    q = Question("q", at(1, 1), "anything?", "g", None, "current")
    ctx = bl.naive_rag(q, CONVERSATION)
    assert ctx.text == "" and ctx.turns_visible == 0 and ctx.items_used == 0


def test_the_naive_rag_arm_ranks_identically_on_a_re_run_including_score_ties():
    """A context that depends on dict ordering cannot be used for a regression test.

    Ties are the common case here, not the corner: identical turns score identically, and
    breaking those on list position would make the arm's output a function of the input
    order rather than of the input.
    """
    twins = [Turn(at(2, 1), "user", "the same sentence"),
             Turn(at(3, 1), "user", "the same sentence"),
             Turn(at(4, 1), "user", "something else entirely")]
    q = Question("q", at(8, 1), "the same sentence", "g", None, "current")
    first = bl.naive_rag(q, twins, k=2).text
    assert first == bl.naive_rag(q, list(reversed(twins)), k=2).text
    assert first.splitlines()[0].startswith("- [2026-02-01]")


# --- arm 4: memvara -------------------------------------------------------------


def test_the_memvara_arm_renders_through_recall_so_the_headers_are_what_ships():
    """`recall()`, not a re-rendering of `search()`.

    The headers, the flattening and the claims-before-episodes ordering are part of what
    an integration receives, so re-formatting the results here would be measuring a
    formatter this repository does not ship.
    """
    from memvara import Memvara

    ctx = bl.memvara(QUESTIONS[2], CONVERSATION)
    assert ctx.text.startswith(Memvara.RECALL_HEADER)
    assert Memvara.RECALL_EPISODE_HEADER in ctx.text
    assert "user lives in Seattle" in ctx.text


def test_the_memvara_arm_resolves_a_superseded_address_to_one_live_value():
    """Portland is closed where Seattle begins, so only Seattle reaches the prompt.

    This is the behaviour the whole comparison is about: the claim block carries the
    current value and not the value it replaced, without the reader having to work out
    which of two dated statements is still true.
    """
    ctx = bl.memvara(QUESTIONS[2], CONVERSATION)
    claims = ctx.text.split("\n" + _episode_header())[0]
    assert "Seattle" in claims and "Portland" not in claims


def _episode_header() -> str:
    from memvara import Memvara

    return Memvara.RECALL_EPISODE_HEADER


def test_the_memvara_arm_cannot_see_a_turn_recorded_after_the_question_was_asked():
    """The one bug in this file that would silently inflate the product's own result.

    A question asked on 20 May must not be answered with the downgrade that happened on
    2 June. The assertion runs twice: once against the arm as written, and once with the
    cutoff removed — `visible_turns` replaced by "every turn" is exactly the bug — to
    prove the first assertion is capable of failing. A cutoff test that passes whether or
    not the cutoff exists is decoration.
    """
    asked_in_may = QUESTIONS[-1]
    assert asked_in_may.asked_at < at(6, 2)

    ctx = bl.memvara(asked_in_may, CONVERSATION)
    assert "Starter" not in ctx.text
    assert "Pro" in ctx.text
    assert ctx.turns_visible < len(CONVERSATION)

    # Now the bug, injected: the arm keeps every turn regardless of when it was asked.
    original = bl.visible_turns
    try:
        bl.visible_turns = lambda q, turns: sorted(turns, key=lambda t: t.at)
        leaked = bl.memvara(asked_in_may, CONVERSATION)
    finally:
        bl.visible_turns = original
    assert "Starter" in leaked.text, (
        "with the cutoff removed the June downgrade must reach a question asked in May; "
        "if it does not, the assertion above is not testing the cutoff"
    )


def test_the_memvara_arm_ingests_each_turn_at_its_own_timestamp_not_at_replay_time():
    """`ts=turn.at` is what gives a claim a `valid_from` on the day it was said.

    Without it every claim in a replayed archive would be valid from the moment of the
    replay, so a supersession would close the old value's valid time at the replay rather
    than at the point the new value began — and every `valid_at` query over the history
    would return the same answer.
    """
    mem, degraded = bl.build_memory(CONVERSATION)
    assert degraded is True
    live = {c.predicate: c for c in mem.get_all()}
    assert live["lives_in"].object == "Seattle"
    assert live["lives_in"].valid_from == at(4, 21)
    history = mem.history("user", "lives_in")
    assert [c.object for c in history] == ["Portland", "Seattle"]
    assert history[0].valid_to == at(4, 21)


def test_the_memvara_arm_reports_degraded_extraction_rather_than_warning_per_question():
    """One store is built per question, so a `warnings.warn` here would fire per question.

    The flag is not a way of hiding the warning: the harness prints it once and loudly,
    which is more visible than the same sentence eighteen times.
    """
    assert bl.memvara(QUESTIONS[0], CONVERSATION).degraded is True

    class _NoopLLM:
        def complete(self, *a, **kw):  # pragma: no cover - never called
            return ""

    assert bl.memvara(QUESTIONS[0], CONVERSATION[:2], llm=_NoopLLM()).degraded is False


def test_the_memvara_arm_is_capped_in_characters_like_the_other_retrieval_arm():
    """Same budget as `naive_rag`, or the comparison is between two budgets."""
    ctx = bl.memvara(QUESTIONS[0], CONVERSATION, max_chars=80)
    assert ctx.chars <= 80


def test_both_retrieval_arms_may_spend_the_same_number_of_slots():
    """`k` is the total, but only `HybridRetriever.max_episodes` of it may go to turns.

    At the library default of 3 this arm was returning three turns against `naive_rag`'s
    twelve, so the two arms were not on one budget and the difference between them would
    have been read as a quality result. That default is right for its own purpose — a
    prompt should not fill with unverified chatter when there are facts available — and
    wrong for a controlled comparison.

    Asserted on a corpus the rule extractor cannot touch, so every slot has to come from
    the episode tail and the cap is the only thing that can bind.
    """
    opaque = [Turn(at(1, 1 + i), "user", f"The gearbox on unit {i} is making a noise.")
              for i in range(20)]
    q = Question("q", at(8, 1), "Which unit has the gearbox noise?", "g", None, "current")
    mem, _ = bl.build_memory(opaque)
    assert mem.get_all() == [], "corpus chosen so that the claim tier stays empty"

    ctx = bl.memvara(q, opaque, k=12)
    assert len(ctx.text.splitlines()) - 1 == 12  # one line is the episode header
    assert ctx.items_used == bl.naive_rag(q, opaque, k=12).items_used == 12


# --- arm 5: memvara_structured ---------------------------------------------------


def test_extraction_can_end_a_claim_but_can_never_retire_one():
    """Half the bitemporal model is unreachable from `add()`, by design, at any corpus.

    This fixture corpus *is* written in first-person declaratives, so unlike the real one
    the rule extractor does produce claims from it — including a supersession when the
    customer moves to Seattle. What it cannot produce, on any corpus and with any
    extractor behind it, is a **retraction**: `add()` has no `close=` parameter, on the
    stated grounds that "a model is not allowed to decide that something we already stored
    was a mistake". Every closure extraction can reach is `ended`.

    So the `correction` questions are not merely hard for a transcript-replay arm, they
    are structurally out of reach: the store has no way to record that a value was never
    true. Only a caller can say that, which is what `memvara_structured` is.
    """
    q = QUESTIONS[0]
    plain, _ = bl.build_memory(bl.visible_turns(q, CONVERSATION),
                               max_episodes=bl.DEFAULT_K)
    assert plain.get_all(), "this fixture is first-person prose, so the rules do fire"
    assert plain.get_all(states=["ended"]), "and they do supersede"
    assert plain.get_all(states=["retired"]) == [], (
        "extraction has no way to retract; if it does, `add()` grew a `close=`"
    )

    # Counted over the desk's own subject: on this fixture the rule extractor also writes,
    # under the generic `user` subject it extracts, and the two populations are separate.
    # On the real corpus it writes nothing at all, so there is nothing to separate.
    rich = fixture_store(q)
    desk = [c for c in rich.get_all(states=["ended", "retired"]) if c.subject == "account"]
    assert [c.object for c in desk if c.state == "retired"] == ["55120"]
    assert len([c for c in desk if c.state == "ended"]) == 3  # the world moved 3 times


def test_the_structured_arm_declares_a_cardinality_for_every_slot_it_writes():
    """A predicate the fact table uses and the registry does not declare defaults to MANY.

    That default is deliberate in the library — errors fall on the recoverable side — but
    it means an undeclared slot silently stops superseding, so the fact table and the
    schema have to be checked against each other rather than kept in step by hand.

    Also pinned: each declared name normalizes to itself. `PredicateRegistry` folds
    surface forms onto canonical ones, and `plan` shares a morphological key with `goal`'s
    alias `plans_to`; registration is what makes it a predicate in its own right, and a
    rename that lost that would send the writes to somebody else's slot.
    """
    declared = {s.name for s in bl.SUPPORT_PREDICATES}
    written = {f.predicate for f in bl.SUPPORT_FACTS}
    assert written <= declared, f"undeclared, so MANY by default: {written - declared}"
    for spec in bl.SUPPORT_PREDICATES:
        assert bl.SUPPORT_REGISTRY.normalize(spec.name) == spec.name
        assert bl.SUPPORT_REGISTRY.spec(spec.name).cardinality is spec.cardinality


def _without(name: str) -> PredicateRegistry:
    return PredicateRegistry(BUILTIN_PREDICATES + tuple(
        s for s in FIXTURE_PREDICATES if s.name != name))


def test_a_predicate_left_at_the_default_cardinality_stops_superseding_silently():
    """The registry step is required, not decoration, and this is what skipping it costs.

    Undeclared, `contact_preference` defaults to `MANY`, so `email` and `phone` both stay
    live and the slot answers a single-valued question with two values. Nothing raises and
    nothing is lost — the store simply stops resolving the contradiction, which is the
    whole feature. The assertion runs both ways so it cannot pass against the bug.
    """
    q = QUESTIONS[0]
    assert bl.slot_values(fixture_store(q), "account", "contact_preference") == ["phone"]

    undeclared = fixture_store(q, registry=_without("contact_preference"))
    assert bl.slot_values(undeclared, "account", "contact_preference") == [
        "email", "phone"], (
        "an undeclared predicate must accumulate; if it does not, the assertion above is "
        "not testing the registry"
    )


def test_an_undeclared_predicate_can_be_folded_onto_somebody_elses_slot():
    """The sharper half of skipping the declaration, and the one that is hard to see.

    `PredicateRegistry.normalize` folds an unrecognised surface form onto a canonical
    predicate whenever the morphology matches, and `plan` shares its content-word key with
    `goal`'s alias `plans_to`. Undeclared, therefore, the desk's plan writes do not merely
    become multi-valued — they are filed under `goal`, which is `MANY`, `FAST` and
    `EPISODIC`, so the slot the integration thinks it is writing comes back **empty** and
    the values decay out of the ranking on a one-week half-life.

    Registration is what makes a name canonical, and canonical is checked before any fold.
    """
    q = QUESTIONS[0]
    folded = fixture_store(q, registry=_without("plan"))
    assert bl.slot_values(folded, "account", "plan") == []
    assert bl.slot_values(folded, "account", "goal") == ["Pro", "Starter"]
    assert bl.slot_values(fixture_store(q), "account", "plan") == ["Starter"]


def test_the_structured_arm_cannot_see_a_fact_recorded_after_the_question_was_asked():
    """The one bug that would silently inflate the product's own result.

    A question asked on 20 May must not be answered from the downgrade recorded on 2 June.
    The cutoff is on `Write.at` — transaction time, what the desk knew — and never on
    `valid_from`, which for a correction is deliberately earlier than anyone knew it.

    Asserted twice, as `test_the_memvara_arm_cannot_see_a_turn_recorded_after_the_question
    _was_asked` does: once against the arm as written, and once with `visible_facts`
    replaced by "every fact", which is exactly the bug.
    """
    asked_in_may = QUESTIONS[-1]
    assert asked_in_may.asked_at < at(6, 2)
    assert bl.slot_values(fixture_store(asked_in_may), "account", "plan") == ["Pro"]

    ctx = structured(asked_in_may)
    assert "Starter" not in ctx.text and "Pro" in ctx.text

    original = bl.visible_facts
    try:
        bl.visible_facts = lambda q, facts: sorted(facts, key=lambda f: f.at)
        leaked = fixture_store(asked_in_may)
    finally:
        bl.visible_facts = original
    assert bl.slot_values(leaked, "account", "plan") == ["Starter"], (
        "with the cutoff removed the June downgrade must reach a question asked in May; "
        "if it does not, the assertion above is not testing the cutoff"
    )


def test_the_cutoff_is_on_when_the_desk_learned_it_not_on_when_it_became_true():
    """A correction is backdated on purpose, and must not backdate itself past the cutoff.

    The 30 March correction says the account number had been wrong since 13 January, so
    its `valid_from` is 13 January and its `at` is 30 March. A cutoff applied to
    `valid_from` would let a question asked in February see a correction that had not
    happened yet — the subtler half of the same leak, and invisible on any fact whose two
    instants coincide.
    """
    before = Question("q", at(2, 20), "What is the account number?", "55120", None,
                      "current")
    assert bl.slot_values(fixture_store(before), "account", "account_number") == ["55120"]
    assert fixture_store(before).get_all(states=["retired"]) == []

    after = Question("q", at(4, 20), "What is the account number?", "55210", None,
                     "current")
    assert bl.slot_values(fixture_store(after), "account", "account_number") == ["55210"]


def test_a_correction_recorded_as_a_change_stops_the_correction_audit_working():
    """`close="retired"` is transaction time. Writing it as `"ended"` costs the question.

    A retracted value and a superseded one are both "the old value" to a store with one
    clock, and `states=["retired"]` is the only thing that tells them apart — it is the
    read that answers "which details did we record wrongly". Recorded as a change, 55120
    joins the list of things that stopped being true, which asserts a world event nobody
    witnessed: nobody ever had that account number.
    """
    q = QUESTIONS[0]
    honest = fixture_store(q)
    assert [c.object for c in honest.get_all(states=["retired"])] == ["55120"]
    assert "55120" not in [c.object for c in honest.get_all(states=["ended"])]

    as_a_change = tuple(
        f if f.mode != "correct" else replace(f, mode="assert") for f in FIXTURE_FACTS)
    mutated = fixture_store(q, facts=as_a_change)
    assert mutated.get_all(states=["retired"]) == [], (
        "with the correction written as a change the audit must come back empty; if it "
        "does not, the assertion above is not testing the closure"
    )
    assert "55120" in [c.object for c in mutated.get_all(states=["ended"])]


def test_a_change_recorded_as_a_correction_erases_a_past_that_really_happened():
    """The mirror of the mutation above, and the more expensive one.

    Retiring a superseded claim calls a true record an error: belief in it stops, so it
    answers nothing at any world-time and "where did the customer live in February" goes
    blank. Nothing is deleted and nothing raises — the past simply stops being reachable,
    which is the failure this library's two axes exist to prevent.
    """
    q = QUESTIONS[0]
    february = at(2, 15)
    assert bl.slot_values(fixture_store(q), "account", "lives_in",
                          valid_at=february) == ["Portland"]

    as_a_correction = tuple(
        replace(f, mode="correct") if f.obj == "Seattle" else f for f in FIXTURE_FACTS)
    mutated = fixture_store(q, facts=as_a_correction)
    assert bl.slot_values(mutated, "account", "lives_in", valid_at=february) == [], (
        "a superseded claim closed on transaction time must stop answering valid_at; if "
        "it still answers, the assertion above is not testing the closure"
    )


def test_a_many_predicate_is_replaced_wholesale_because_cardinality_closes_nothing():
    """`contact_preference` is the one genuinely multi-valued slot, and it costs a line.

    On 6 February the customer names two channels in one breath, so a `ONE` field would
    have to discard one of them and would disagree with the authored answer for April.
    `MANY` keeps both — and supersedes nothing, so the reversal in June has to close the
    slot explicitly. That is the integration cost of the cardinality being honest, and it
    is visible here rather than hidden behind a `ONE` that quietly drops a value.
    """
    assert bl.SUPPORT_REGISTRY.spec("contact_preference").cardinality is Cardinality.MANY
    assert {f.predicate for f in bl.SUPPORT_FACTS if f.mode == "replace"} == {
        "contact_preference"}, (
        "only a MANY slot needs replacing by hand; a ONE slot that did would mean the "
        "reconciler was not closing it"
    )
    assert {s.name for s in bl.SUPPORT_PREDICATES
            if s.cardinality is Cardinality.MANY} == {"contact_preference"}

    # The behaviour the cardinality buys, over the real fact table and no turns at all.
    mem, _ = bl.build_memory([], registry=bl.SUPPORT_REGISTRY)
    bl.apply_facts(mem, bl.SUPPORT_FACTS)
    april = datetime(2026, 4, 15, tzinfo=UTC)
    assert bl.slot_values(mem, "account", "contact_preference", valid_at=april) == [
        "phone", "text"], "a ONE field would have had to discard one of the two channels"
    assert bl.slot_values(mem, "account", "contact_preference") == ["email"], (
        "and the replace on 22 June must close both of them"
    )


def test_a_correction_with_nothing_standing_is_refused_rather_than_written():
    """A correction that retires nothing would leave the record looking corrected.

    The failure it guards is a fact table that has drifted — a correction whose target was
    renamed or removed. Asserting the new value anyway is the worst outcome: the slot ends
    up holding the right answer with no retirement behind it, so `states=["retired"]`
    comes back empty and the audit reports that nothing was ever wrong.
    """
    orphan = (bl.Write(at(3, 30), "account", "account_number", "55210",
                       "The account number is 55210.", since=at(1, 13), mode="correct"),)
    with pytest.raises(ValueError, match="found 0 standing values"):
        fixture_store(QUESTIONS[0], facts=orphan)


def claim_lines(context_text: str) -> set[str]:
    """The bullets above the episode header: what the arm asserted as *fact*.

    Everything below that header is a raw turn, and the trap is quoted in the turns by
    construction — this corpus re-surfaces every superseded value late and emphatically.
    An assertion over the whole context would therefore see the trap on every arm and
    prove nothing. The claim block is where the memory layer's answer actually is.

    The provenance marker is stripped. Every fact in this corpus was extracted from the
    support transcript rather than stated to the agent, so `recall()` marks all of them —
    and these assertions are about *which* facts reach the prompt, not how they are
    annotated. Leaving it on would make every one of them a test of the marker's wording.
    """
    block = context_text.split("\n" + Memvara.RECALL_EPISODE_HEADER)[0]
    return {line[2:].removesuffix(Memvara.RECALL_INFERRED)
            for line in block.splitlines() if line.startswith("- ")}


def test_the_structured_arm_reads_at_valid_at_only_where_the_scenario_set_about():
    """`about` is the valid-time instant, and nothing here derives one from question prose.

    The arm passes `valid_at=question.about` straight through, so a question with no
    `about` reads at the present — which is what `recall()` does and what `asked_at`
    already carries. Inventing an instant for the others would make the arm agree with a
    wrong answer on every question whose fact had moved since.

    Asserted on the **claim block** rather than on the context as a whole: the raw turns
    below it name every plan this account was ever on, so "Pro appears somewhere in the
    prompt" is true whatever the arm does with `about` and is not evidence of anything.
    """
    historical = QUESTIONS[1]
    assert historical.about == at(3, 15)
    assert bl.slot_values(fixture_store(historical), "account", "plan",
                          valid_at=historical.about) == ["Pro"]
    lines = claim_lines(structured(historical).text)
    assert "The account is on the Pro plan." in lines
    assert "The account is on the Starter plan." not in lines

    current = QUESTIONS[0]
    assert current.about is None
    assert bl.slot_values(fixture_store(current), "account", "plan",
                          valid_at=current.about) == ["Starter"]
    assert "The account is on the Starter plan." in claim_lines(structured(current).text)


def test_the_structured_arms_rendering_is_byte_identical_to_recall():
    """`recall()` takes no `valid_at=`, so the arm renders `search()` itself. Pin the two.

    Re-rendering would otherwise be measuring a formatter this repository does not ship.
    Compared over every fixture question, with episodes on, so the header text, the bullet,
    the flattening and the 280-character episode truncation are all in the comparison.
    """
    for q in QUESTIONS:
        mem = fixture_store(q)
        results = mem.search(q.text, k=bl.DEFAULT_K, include_episodes=True)
        assert bl.render_recall(results) == mem.recall(q.text, k=bl.DEFAULT_K,
                                                       include_episodes=True), q.id


def test_a_long_turn_is_truncated_by_the_arms_rendering_exactly_as_recall_truncates_it():
    """The episode cap is the half of the rendering the corpus above never reaches.

    `recall()` caps a raw turn at `RECALL_EPISODE_CHARS` so that a pasted stack trace
    cannot evict the facts, and truncation is where two implementations of one format are
    most likely to drift — an off-by-one in the ellipsis, or the cap applied before the
    flattening instead of after. Every turn in the fixture corpus is short, so the branch
    is reached with a turn that is not.
    """
    long_turn = Turn(at(1, 5), "user", "The gearbox " + "rattles and hums " * 40)
    assert len(long_turn.text) > Memvara.RECALL_EPISODE_CHARS
    q = Question("q", at(8, 1), "What is wrong with the gearbox?", "rattles", None,
                 "current")
    mem, _ = bl.build_memory([long_turn], max_episodes=bl.DEFAULT_K,
                             registry=FIXTURE_REGISTRY)

    rendered = bl.render_recall(mem.search(q.text, k=bl.DEFAULT_K, include_episodes=True))
    assert rendered == mem.recall(q.text, k=bl.DEFAULT_K, include_episodes=True)
    assert rendered.endswith("…")


def test_the_structured_arm_is_capped_and_budgeted_like_every_other_retrieval_arm():
    """Same character cap and the same slot count, or the comparison is between budgets."""
    assert structured(QUESTIONS[0], max_chars=80).chars <= 80
    ctx = structured(QUESTIONS[0], k=12)
    assert ctx.items_used <= 12
    assert ctx.turns_visible == bl.memvara(QUESTIONS[0], CONVERSATION).turns_visible


# --- the embedder is pinned ------------------------------------------------------


def test_neither_embedding_arm_falls_through_to_the_ambient_default_embedder(monkeypatch):
    """`default_embedder()` returns a sentence-transformers model when one is installed.

    One is installed in this environment. An arm that reached it would be a different
    experiment on a machine with the extra than on one without, and the difference would
    surface as a quality delta with no code change behind it. Both arms are asserted to
    run to completion with the ambient default made to explode.
    """
    import memvara.core

    def boom():  # pragma: no cover - the point is that it is never reached
        raise AssertionError("an arm reached default_embedder() instead of pinning one")

    monkeypatch.setattr(memvara.core, "default_embedder", boom)
    assert bl.naive_rag(QUESTIONS[0], CONVERSATION).text
    assert bl.memvara(QUESTIONS[0], CONVERSATION).text


def test_both_embedding_arms_are_pinned_to_the_same_vector_space():
    """"The same embedder" has to be true rather than approximately true.

    `HashingEmbedder`'s identity includes its dimension, so two arms at different widths
    would be two embedders, and the delta between them would carry vector quality as well
    as time handling.
    """
    from memvara import HashingEmbedder

    mem, _ = bl.build_memory(CONVERSATION[:2])
    assert mem.embedder.name == HashingEmbedder(dim=bl.EMBED_DIM).name


# --- shared rendering ------------------------------------------------------------


def test_the_two_turn_rendering_arms_are_byte_identical_in_shape():
    """`naive_rag` is the only arm with real cover, and this is what gives it that.

    It shares `full_transcript`'s exact line format and differs only in selection and
    ordering, so it cannot be identified in the dump by its formatting. Nothing can be
    done about `none` being empty or `memvara` writing its own headers.
    """
    q = QUESTIONS[0]
    line = bl.full_transcript(q, CONVERSATION).text.splitlines()[0]
    every = set(bl.full_transcript(q, CONVERSATION).text.splitlines())
    assert set(bl.naive_rag(q, CONVERSATION, k=3).text.splitlines()) <= every
    assert line.startswith("- [") and "] user: " in line


def test_rendering_collapses_whitespace_so_a_pasted_block_cannot_forge_lines():
    """One turn is one line. A turn containing newlines that produced several lines would
    let stored text invent context entries that no arm put there."""
    messy = [Turn(at(1, 1), "user", "line one\nline two\n\n  line three")]
    assert bl.render_turns(messy) == "- [2026-01-01] user: line one line two line three"


# --- the harness: planning -------------------------------------------------------


def test_the_plan_is_one_item_per_arm_and_question_with_a_stable_blinded_id():
    """The id is a digest of the prompt, so it carries no arm and no question id.

    Stability across runs is what lets the answers come back into a second process and be
    matched up without the key file having to be trusted.
    """
    items = hz.plan(QUESTIONS, CONVERSATION)
    assert len(items) == len(bl.ARMS) * len(QUESTIONS)
    assert {i.arm for i in items} == set(bl.ARMS)
    assert [i.id for i in items] == [i.id for i in hz.plan(QUESTIONS, CONVERSATION)]
    assert all(i.id == ek.item_id(i.prompt) for i in items)


def test_the_plan_puts_the_asked_on_date_on_every_arm_including_the_empty_one():
    """Without it, "what plan are they on now" is unanswerable by construction.

    Giving it to four arms and not the fifth would handicap the floor differently from
    the arms it is the control for, which is the one thing a floor may not be.
    """
    items = {i.arm: i for i in hz.plan(QUESTIONS[:1], CONVERSATION)}
    for arm, item in items.items():
        assert "Today's date: 2026-08-01" in item.prompt, arm
    assert items["none"].prompt.endswith(ek.CONTEXT_MARKER)


def test_the_plan_refuses_to_run_with_no_arms_at_all():
    """A comparison with nothing to compare is a scoring bug waiting to be reported as a
    result."""
    with pytest.raises(ValueError, match="at least one"):
        hz.plan(QUESTIONS, CONVERSATION, arms={})


# --- the harness: blinding -------------------------------------------------------


def test_the_dump_holds_all_five_arms_in_one_file_shuffled_together(tmp_path):
    """Five files, or one file written arm by arm, would be readable by position.

    `FileReader.finish()` merges what is already on disk and re-shuffles the union, which
    is the only arrangement that makes a head-to-head actually blind.
    """
    path = tmp_path / "dump.jsonl"
    result = hz.dump(QUESTIONS, CONVERSATION, path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert result.items == len(rows) == len(bl.ARMS) * len(QUESTIONS)

    key = json.loads(result.key_path.read_text())
    assert sorted(key["systems"]) == sorted(bl.ARMS)
    order = [row["id"] for row in rows]
    assert order != sorted(order)  # shuffled, not digest-ordered


def test_the_dumped_prompt_names_no_arm_no_kind_no_gold_and_no_trap(tmp_path):
    """Everything the answerer must not see, asserted against the file on disk."""
    path = tmp_path / "dump.jsonl"
    hz.dump(QUESTIONS, CONVERSATION, path)
    # `harness.dump` writes UTF-8 with `ensure_ascii=False`, so reading without naming
    # the encoding decodes it as cp1252 on Windows and mangles the em-dash in `SYSTEM`.
    # The writer is right and this read was wrong; the assertion below compares the
    # round-tripped prompt against the constant, so it is exactly the check that breaks.
    blob = path.read_text(encoding="utf-8")
    for leak in list(bl.ARMS) + ["correction", "unanswerable", "historical",
                                 "q_plan_now", "gold", "trap"]:
        assert leak not in blob, leak
    rows = [json.loads(line) for line in blob.splitlines() if line.strip()]
    assert {k for row in rows for k in row} == {"id", "system_prompt", "prompt"}
    assert {row["system_prompt"] for row in rows} == {hz.SYSTEM}


def test_the_key_file_holds_the_arm_mapping_and_is_written_separately(tmp_path):
    """The mapping has to survive somewhere, and somewhere is not the dump."""
    path = tmp_path / "dump.jsonl"
    result = hz.dump(QUESTIONS, CONVERSATION, path, now=datetime(2026, 8, 13, tzinfo=UTC))
    key = json.loads(result.key_path.read_text())
    assert result.key_path != path
    assert key["seed"] == 20260813 and key["created"].startswith("2026-08-13")
    by_arm: dict[str, int] = {}
    for row in key["items"]:
        by_arm[row["system"]] = by_arm.get(row["system"], 0) + 1
    assert by_arm == {arm: len(QUESTIONS) for arm in bl.ARMS}


def test_the_dump_order_does_not_depend_on_the_order_the_arms_ran_in(tmp_path):
    """`finish()` sorts by digest and then shuffles by seed, so insertion order is not
    recoverable from the file — which is why the arms may run in a fixed sequence."""
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    hz.dump(QUESTIONS, CONVERSATION, first)
    reversed_arms = {k: bl.ARMS[k] for k in reversed(list(bl.ARMS))}
    hz.dump(QUESTIONS, CONVERSATION, second, arms=reversed_arms)
    assert first.read_text() == second.read_text()


def test_two_arms_building_identical_context_are_reported_rather_than_silently_merged(
        tmp_path):
    """`FileReader` merges equal prompts and keeps the first arm's attribution, silently.

    Left undetected, a five-arm dump would quietly become a four-and-a-bit-arm dump while
    the report still said five. A collision is not repaired — two arms that build
    byte-identical context genuinely are one arm for that question — it is named.
    """
    twins = {"none": bl.none, "also_none": bl.none}
    result = hz.dump(QUESTIONS[:2], CONVERSATION, tmp_path / "d.jsonl", arms=twins)
    assert result.items == 2
    assert len(result.collisions) == 2
    assert sorted(next(iter(result.collisions.values()))) == ["also_none/q_plan_now",
                                                             "none/q_plan_now"]


# --- the harness: scoring --------------------------------------------------------


def _answer_file(tmp_path: Path, items, answers: dict[str, str]) -> Path:
    """Answer every item by (arm, qid), the way a completed round trip would."""
    path = tmp_path / "answers.jsonl"
    path.write_text("\n".join(
        json.dumps({"id": i.id, "answer": answers.get(f"{i.arm}/{i.qid}", "")})
        for i in items) + "\n")
    return path


def test_correct_and_trapped_are_counted_independently_of_each_other(tmp_path):
    """They are not complements, and an answer can be both.

    "They moved from Portland to Seattle in April" contains the gold *and* the superseded
    value. Reporting `trapped` as `not correct` would erase the distinction between an
    answer that recites the history and an answer that gives the stale value as the
    current one — and the second is the failure the product claims to prevent.
    """
    items = hz.plan(QUESTIONS, CONVERSATION, arms={"none": bl.none})
    answers = _answer_file(tmp_path, items, {
        "none/q_city": "They moved from Portland to Seattle in April.",
        "none/q_plan_now": "Pro.",
        "none/q_account": "55210",
    })
    scored = hz.score(items, QUESTIONS, reader=ek.FileReader(answers=answers),
                      judge=ek.ContainmentJudge())
    by_qid = {r.qid: r for r in scored}
    assert by_qid["q_city"].correct and by_qid["q_city"].trapped
    assert not by_qid["q_plan_now"].correct and by_qid["q_plan_now"].trapped
    assert by_qid["q_account"].correct and not by_qid["q_account"].trapped


def test_the_trap_is_judged_under_default_not_under_knowledge_update():
    """`knowledge-update`'s prompt marks a superseded value *wrong*.

    Asking the trap question under it would invert the count: the judge would answer "no"
    precisely when the answer did give the trap. `default` with `gold=trap` asks the
    question actually being counted — did this response contain that value.
    """
    assert hz.TRAP_JUDGE_TYPE == "default"
    assert hz.JUDGE_TYPES["current"] == "knowledge-update"
    assert hz.JUDGE_TYPES["correction"] == "knowledge-update"
    assert hz.JUDGE_TYPES["historical"] == "default"
    assert hz.JUDGE_TYPES["unanswerable"] == ek.ABSTENTION_TYPE


def test_a_historical_question_is_not_graded_as_if_it_wanted_the_latest_value(tmp_path):
    """A question about March is supposed to be answered with March's value.

    Grading it under `knowledge-update` would mark the correct answer wrong and flatter
    every arm that cannot do history.
    """
    items = hz.plan([QUESTIONS[1]], CONVERSATION, arms={"none": bl.none})
    answers = _answer_file(tmp_path, items, {"none/q_plan_march": "Pro"})
    scored = hz.score(items, [QUESTIONS[1]], reader=ek.FileReader(answers=answers),
                      judge=ek.ContainmentJudge())
    assert scored[0].correct and not scored[0].trapped


def test_an_unanswered_item_is_counted_as_wrong_and_never_silently_dropped(tmp_path):
    """A partial run must read as a low score, not as a high score over a subset."""
    items = hz.plan(QUESTIONS[:2], CONVERSATION, arms={"none": bl.none})
    answers = _answer_file(tmp_path, items, {})
    scored = hz.score(items, QUESTIONS, reader=ek.FileReader(answers=answers),
                      judge=ek.ContainmentJudge())
    assert all(not r.correct and not r.trapped and r.answer == "" for r in scored)
    cells = hz.tally(scored, lambda r: r.arm)
    assert cells["none"].n == 2 and cells["none"].answered == 0


def test_the_trapped_denominator_is_questions_that_have_a_trap_not_every_question(
        tmp_path):
    """A trapped rate over questions with no superseded value to give is diluted by
    questions that could not have produced one."""
    items = hz.plan(QUESTIONS, CONVERSATION, arms={"none": bl.none})
    answers = _answer_file(tmp_path, items, {"none/q_city": "Portland"})
    scored = hz.score(items, QUESTIONS, reader=ek.FileReader(answers=answers),
                      judge=ek.ContainmentJudge())
    cell = hz.tally(scored, lambda r: r.arm)["none"]
    assert cell.n == len(QUESTIONS)
    assert cell.trap_defined == sum(1 for q in QUESTIONS if q.trap is not None)
    assert cell.trapped == 1 and cell.trapped_only == 1


def test_a_rate_over_an_empty_denominator_reads_as_absent_rather_than_as_zero():
    """0% and "no questions of this kind" are different findings and must not print the
    same."""
    assert hz.rate(0, 0) == "-"
    assert hz.rate(0, 4) == "0%"
    assert hz.rate(1, 3) == "33%"


def test_a_stale_dump_is_refused_rather_than_scored_as_if_it_were_the_whole_run(tmp_path):
    """`demo/scenario.py` belongs to another workstream and can change under a dump.

    Scoring the intersection would report a subset of the run as the run. The ids that no
    longer reconcile are returned so the caller can refuse.
    """
    path = tmp_path / "dump.jsonl"
    hz.dump(QUESTIONS, CONVERSATION, path)
    items = hz.plan(QUESTIONS, CONVERSATION)
    assert hz.stale_ids(hz.key_path_for(path), items) == []

    changed = hz.plan(QUESTIONS, CONVERSATION[:-4])
    assert len(hz.stale_ids(hz.key_path_for(path), changed)) > 0


# --- the harness: reporting ------------------------------------------------------


def _full_run(tmp_path: Path, answers: dict[str, str], **kw):
    items = hz.plan(QUESTIONS, CONVERSATION, **kw)
    path = _answer_file(tmp_path, items, answers)
    reader = ek.FileReader(answers=path)
    judge = ek.ContainmentJudge()
    scored = hz.score(items, QUESTIONS, reader=reader, judge=judge)
    return items, hz.report(items, scored, reader=reader, judge=judge, **kw)


def test_the_report_prints_context_size_per_arm_because_that_is_the_actual_argument(
        tmp_path):
    """Whether the memory layer earns its place is decided by this table, not by accuracy.

    At a corpus this size the transcript fits, so the accuracy delta is the weaker half of
    the case; the size column multiplied by a year of history is the stronger half.
    """
    _, text = _full_run(tmp_path, {})
    assert "mean chars" in text and "items used / turns seen" in text
    for arm in bl.ARMS:
        assert arm in text


def test_the_report_says_out_loud_that_extraction_ran_without_a_model(tmp_path):
    """With no `llm=`, only the rule extractor contributes claims.

    That is a real configuration and it is the one measured here, but it is the memory
    layer's weaker half — supersession only fires on facts that became claims — and a
    report that did not say so would be overstating what was measured.
    """
    _, text = _full_run(tmp_path, {})
    assert "EXTRACTION RAN DEGRADED" in text and "memvara" in text


def test_the_report_says_when_naive_rag_retrieved_the_whole_transcript(tmp_path):
    """Below k turns it is `full_transcript` reordered, not a retrieval arm.

    Any delta between the two rows would then be the ordering alone, and reporting that as
    a retrieval result would be reporting a shuffle.
    """
    short = CONVERSATION[:4]
    items = hz.plan(QUESTIONS[:1], short)
    reader = ek.FileReader(answers=_answer_file(tmp_path, items, {}))
    judge = ek.ContainmentJudge()
    text = hz.report(items, hz.score(items, QUESTIONS, reader=reader, judge=judge),
                     reader=reader, judge=judge)
    assert "naive_rag RETRIEVED EVERY VISIBLE TURN" in text


def test_the_report_flags_unanswered_items_instead_of_quietly_scoring_the_rest(tmp_path):
    _, text = _full_run(tmp_path, {"none/q_city": "Seattle"})
    assert "have no answer" in text


def test_the_report_disowns_the_floor_arms_score_on_unanswerable_questions(tmp_path):
    """An arm with no context abstains because it has nothing to abstain from.

    Left unsaid, `none / unanswerable: 100%` reads as the floor doing well on the hardest
    kind, when it is a fact about an empty string. It would score the same on any corpus,
    which is the definition of a number that measures nothing.
    """
    _, text = _full_run(tmp_path, {})
    assert "`none / unanswerable` row is an artefact" in text


def test_the_report_carries_both_caveats_a_containment_judged_file_run_needs(tmp_path):
    """The answerer is the same party that wrote the library, and the judge is a substring
    rule. Either alone would be enough to stop the number being quoted as a benchmark."""
    _, text = _full_run(tmp_path, {})
    assert "THE READER WAS A HUMAN OR AN AGENT" in text
    assert "THE JUDGE IS A SUBSTRING RULE" in text
    assert "trapped only" in text


def test_the_report_does_not_print_the_containment_caveat_for_a_model_judge(tmp_path):
    """The caveat has to be attached to the instrument, not to the harness, or it would
    still be printed on the run that fixed it."""
    class _YesJudge:
        name = "fake-llm"

        def judge(self, question, gold, hypothesis, question_type):
            return True, ek.Answer("yes", model="fake")

    items = hz.plan(QUESTIONS[:1], CONVERSATION, arms={"none": bl.none})
    reader = ek.FileReader(answers=_answer_file(tmp_path, items, {"none/q_plan_now": "x"}))
    judge = _YesJudge()
    text = hz.report(items, hz.score(items, QUESTIONS, reader=reader, judge=judge),
                     reader=reader, judge=judge, arms={"none": bl.none})
    assert "THE JUDGE IS A SUBSTRING RULE" not in text
    assert "THE READER WAS A HUMAN OR AN AGENT" in text


def test_the_per_kind_table_is_ordered_by_arm_rather_than_alphabetically(tmp_path):
    """Sorting by name would put `full_transcript` above `memvara` and break the
    floor/ceiling/competitor/product reading the table is laid out for."""
    _, text = _full_run(tmp_path, {})
    body = text.split("per arm and question kind")[1]
    positions = [body.index(arm + " /") for arm in bl.ARMS]
    assert positions == sorted(positions)


def test_per_question_output_is_written_so_a_run_can_be_audited_not_trusted(tmp_path):
    items = hz.plan(QUESTIONS[:2], CONVERSATION, arms={"none": bl.none})
    reader = ek.FileReader(answers=_answer_file(tmp_path, items, {"none/q_plan_now": "Pro"}))
    scored = hz.score(items, QUESTIONS, reader=reader, judge=ek.ContainmentJudge())
    out = tmp_path / "rows.jsonl"
    hz.write_jsonl(out, scored)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows[0]["answer"] == "Pro" and rows[0]["trapped"] is True
    assert set(rows[0]) >= {"id", "arm", "qid", "kind", "gold", "trap", "correct"}


# --- the harness: judges and the CLI ---------------------------------------------


def test_the_containment_judge_is_the_default_so_the_run_works_with_no_key():
    """There is no API key in this environment and the run has to work anyway."""
    assert isinstance(hz.build_judge("containment"), ek.ContainmentJudge)


def test_the_model_judge_wires_up_the_moment_a_key_exists(monkeypatch):
    """`evalkit.build_judge` refuses an `LLMJudge` beside a `FileReader`, on the grounds
    that the answerer would be marking their own work. Here they are never the same
    party: the reader is the blinded file round trip and the judge is a separate model, so
    the refusal does not apply and the configuration it forbids is the correct one."""
    built: dict[str, str] = {}

    class _FakeAnthropic:
        is_stub = False
        is_human = False

        def __init__(self, model):
            built["model"] = model
            self.name = f"anthropic/{model}"

    monkeypatch.setattr(ek, "AnthropicReader", _FakeAnthropic)
    judge = hz.build_judge("llm", model="claude-sonnet-5")
    assert isinstance(judge, ek.LLMJudge) and built["model"] == "claude-sonnet-5"
    assert hz.build_judge("llm") and built["model"] == "claude-opus-5"


def _install_stub_scenario(monkeypatch) -> None:
    """Drive the CLI off this file's corpus rather than the real one.

    Not because the real `demo/scenario.py` is unavailable — `load_scenario` is tested
    against it directly, below — but because the CLI tests exercise argument handling and
    exit codes, and rebuilding sixty-four turns into a fresh store twenty times per arm to
    check an exit code is a minute of CI for no extra assurance.
    """
    monkeypatch.setattr(hz, "load_scenario", lambda: (list(QUESTIONS), list(CONVERSATION)))


def test_the_cli_dumps_and_then_scores_the_same_run_end_to_end(tmp_path, monkeypatch,
                                                               capsys):
    """Both phases through `main`, so a change that breaks a real run breaks this."""
    _install_stub_scenario(monkeypatch)
    dump_path = tmp_path / "dump.jsonl"
    assert hz.main(["--dump", str(dump_path)]) == 0
    assert "blinded items" in capsys.readouterr().out

    items = hz.plan(QUESTIONS, CONVERSATION)
    answers = _answer_file(tmp_path, items, {"memvara/q_city": "Seattle",
                                             "naive_rag/q_city": "Portland"})
    out = tmp_path / "rows.jsonl"
    assert hz.main(["--dump", str(dump_path), "--answers", str(answers),
                    "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "five-arm answer quality, blinded" in printed
    assert out.exists()


def test_the_cli_reports_collisions_at_dump_time(tmp_path, monkeypatch, capsys):
    """Named at the moment the dump is written, when re-dumping is still cheap."""
    _install_stub_scenario(monkeypatch)
    monkeypatch.setattr(bl, "ARMS", {"none": bl.none, "also_none": bl.none})
    monkeypatch.setattr(hz, "ARMS", bl.ARMS)
    assert hz.main(["--dump", str(tmp_path / "d.jsonl")]) == 0
    assert "PROMPT COLLISIONS" in capsys.readouterr().out


def test_the_cli_refuses_a_stale_dump_with_a_non_zero_exit(tmp_path, monkeypatch, capsys):
    """A scoring run that silently covered a subset would be worse than no run."""
    _install_stub_scenario(monkeypatch)
    dump_path = tmp_path / "dump.jsonl"
    assert hz.main(["--dump", str(dump_path)]) == 0
    capsys.readouterr()

    # The corpus changes under the dump, which is what happens when another workstream
    # edits `demo/scenario.py` between the questions going out and the answers coming back.
    monkeypatch.setattr(hz, "load_scenario",
                        lambda: (list(QUESTIONS[:2]), list(CONVERSATION)))
    answers = tmp_path / "answers.jsonl"
    answers.write_text("")
    assert hz.main(["--dump", str(dump_path), "--answers", str(answers)]) == 1
    assert "dump is stale" in capsys.readouterr().out


def test_load_scenario_reads_the_real_corpus_and_it_satisfies_the_arms_contract():
    """The one place the demo touches `demo/scenario.py`, checked against the real file.

    Everything else in these two modules is written against the published contract rather
    than against the corpus, so this is where a contract break has to surface: a missing
    field, a naive datetime, or a `kind` the judge has no mapping for would all pass every
    other test in this file and then produce a run that scored the wrong thing.
    """
    questions, turns = hz.load_scenario()
    assert questions and turns
    assert all(t.at.tzinfo is not None and t.role in {"user", "assistant"} for t in turns)
    assert all(q.asked_at.tzinfo is not None for q in questions)
    assert all(q.kind in hz.JUDGE_TYPES for q in questions), (
        "a kind with no judge mapping would silently be graded as `default`"
    )
    assert {q.id for q in questions} == {q.id for q in questions}, "ids must be unique"
    assert len({q.id for q in questions}) == len(questions)


# --- the harness: the offline run -------------------------------------------------


def test_the_offline_run_answers_and_scores_every_item_with_no_key_and_no_file():
    """One process, no dump, no answerer, no network. The guarded path.

    The two-phase round trip exists because the answerer is outside this process, and
    that is exactly what makes it unrunnable in CI: it stops halfway and waits for a
    person. This path is the one a test, a `git bisect` or an evaluator with no key can
    actually run, and it has to cover every arm to be worth running at all.
    """
    run = hz.offline(QUESTIONS, CONVERSATION)
    assert {i.arm for i in run.items} == set(bl.ARMS)
    assert len(run.scored) == len(bl.ARMS) * len(QUESTIONS)
    assert all(row.answer for row in run.scored), (
        "an unanswered row here is a stub that abstained on everything, which would "
        "score every arm at zero and look like a finding"
    )
    assert isinstance(run.reader, ek.StubReader) and isinstance(run.judge,
                                                               ek.ContainmentJudge)


def test_the_offline_run_is_identical_twice():
    """Two runs, one diff. The whole reason this path is worth wiring up.

    A number that moves between two runs of the same code cannot be bisected, and
    retrieval was in exactly that state until ties stopped breaking on `claim.id` — a
    fresh `uuid4` per ingest, which made repeated LOCOMO runs disagree by 0.07 points with
    nothing changed. Asserting on the rendered report rather than on the tallies is
    deliberate: the report is what a reader compares, so anything that reaches it — an
    ordering, a caveat, a size column — is inside the guarantee.
    """
    first, second = hz.offline(QUESTIONS, CONVERSATION), hz.offline(QUESTIONS,
                                                                   CONVERSATION)
    assert first.report() == second.report()
    assert [vars(r) for r in first.scored] == [vars(r) for r in second.scored]


def test_two_ingest_orders_produce_the_same_context():
    """Same turns, opposite insertion order, byte-identical prompt.

    This is the property `HybridRetriever._rank` breaks ties on `value_key` for, stated
    where an evaluator would notice it breaking. Claim ids are `uuid4` minted at ingest,
    so an id tiebreak gives an ordering that is stable within one store and a coin flip
    across two ingests of identical data — which is precisely the comparison a benchmark
    and a bisect both make.

    Reversing the *ingest* order rather than the argument to an arm is what makes this
    test bite: `visible_turns` sorts, so an arm handed a shuffled list has already
    normalised it and the assertion would hold no matter what the ranking did.
    """
    seen = bl.visible_turns(QUESTIONS[0], CONVERSATION)
    forward, _ = bl.build_memory(seen)
    backward, _ = bl.build_memory(list(reversed(seen)))
    assert forward.stats()["claims"] > 1, (
        "an empty or single-row store ranks identically however it was filled, so the "
        "assertion below would pass without testing anything"
    )
    for q in QUESTIONS:
        rendered = forward.recall(q.text, k=bl.DEFAULT_K, include_episodes=True)
        assert rendered == backward.recall(q.text, k=bl.DEFAULT_K, include_episodes=True)


def test_the_offline_cli_is_one_command_and_says_what_it_is_not(monkeypatch, capsys):
    """`--reader stub`, end to end, and both banners it has to carry.

    The stub's accuracy column is a property of the corpus and a bag-of-words matcher, so
    the run has to disown it in the same breath it prints it. `evalkit.stub_caveat` says
    that much and ends by naming a flag that exists in `bench/` and not here, which is why
    the harness appends its own line.
    """
    _install_stub_scenario(monkeypatch)
    assert hz.main(["--reader", "stub"]) == 0
    printed = capsys.readouterr().out
    assert "five-arm answer quality, blinded" in printed
    assert "THE READER IS A STUB" in printed
    assert "`--reader file`" in printed


def test_the_offline_cli_writes_the_per_question_rows_too(tmp_path, monkeypatch, capsys):
    """`--out` is how a run is audited rather than trusted, on both readers."""
    _install_stub_scenario(monkeypatch)
    out = tmp_path / "rows.jsonl"
    assert hz.main(["--reader", "stub", "--out", str(out)]) == 0
    capsys.readouterr()
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == len(bl.ARMS) * len(QUESTIONS)


def test_the_file_reader_still_refuses_to_run_without_a_dump(monkeypatch, capsys):
    """`--dump` stopped being unconditionally required when the stub reader landed.

    Losing the requirement entirely would make `demo/harness.py` with no arguments write
    nothing and exit 0, which reads as a completed run.
    """
    _install_stub_scenario(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        hz.main([])
    assert exc.value.code == 2
    assert "--dump is required" in capsys.readouterr().err


# --- the answer key, checked against the library and no reader --------------------
#
# Everything above this line is mechanics on a fixture. These are the claim being made:
# that `memvara_structured` returns the *authored answer* for this corpus, read straight
# out of the slot, with no model in the loop.
#
# That is a check the reader-based scoring in `demo/harness.py` cannot give, and it is
# stronger evidence than an answer-quality percentage, because it separates the library
# from the model reading its output. A percentage that moves when the judge changes is a
# measurement of the judge. These do not move.
#
# It also means these tests read `demo/scenario.py`'s contents, which nothing else here
# does. That is unavoidable and it is the right trade: `SUPPORT_FACTS` restates this
# corpus's values, so if D2 edits an address the two must disagree loudly rather than
# quietly measuring a store full of the wrong strings.


#: Which slot answers which question. **Authored**, from the transcript — not produced by
#: running memvara and recording where it looked, which would make the whole section
#: circular. `demo/README.md`'s table of what moved and why is the source.
#:
#: The five questions not listed are not slot reads and each has its own test below:
#: `q_plan_history` (a timeline), `q_which_were_corrections` (the two audits), and the two
#: unanswerables (nothing to read).
ANSWERING_SLOT: dict[str, tuple[str, str]] = {
    "q_plan_current": ("account", "plan"),
    "q_shipping_current": ("account", "delivery_address"),
    "q_billing_current": ("account", "billing_address"),
    "q_contact_current": ("account", "contact_preference"),
    "q_serial_current": ("main_unit", "serial"),
    "q_mobile_current": ("account", "mobile"),
    "q_account_name": ("account", "account_name"),
    "q_billing_day": ("account", "billing_day"),
    "q_plan_current_in_april": ("account", "plan"),
    "q_serial_current_in_may": ("main_unit", "serial"),
    "q_plan_mid_march": ("account", "plan"),
    "q_shipping_april": ("account", "delivery_address"),
    "q_billing_address_on_30_july": ("account", "billing_address"),
    "q_contact_april": ("account", "contact_preference"),
    "q_serial_correction": ("main_unit", "serial"),
    "q_mobile_correction": ("account", "mobile"),
}

#: The one question whose slot holds two values at the instant it asks about, because its
#: gold names two: "By phone or text." Exempted by id rather than by a rule, so a *new*
#: question whose slot has gone multi-valued by accident still fails.
MULTI_VALUED = {"q_contact_april"}

_STORES: dict[datetime, Memvara] = {}


def real_store(question) -> Memvara:
    """The store `memvara_structured` reads from, for one question's cutoff.

    Memoized on `asked_at` — the only thing the cutoff depends on — because the corpus has
    three distinct ask instants and twenty questions, and rebuilding sixty-four turns
    twenty times is a minute of CI for no extra assurance.
    """
    if question.asked_at not in _STORES:
        questions, turns = hz.load_scenario()
        mem, _ = bl.build_memory(bl.visible_turns(question, turns),
                                 max_episodes=bl.DEFAULT_K,
                                 registry=bl.SUPPORT_REGISTRY)
        bl.apply_facts(mem, bl.visible_facts(question, bl.SUPPORT_FACTS))
        _STORES[question.asked_at] = mem
    return _STORES[question.asked_at]


_CONTEXTS: dict[str, bl.Context] = {}


def real_context(question) -> bl.Context:
    """What `memvara_structured` actually puts in front of a reader, memoized per question.

    Built through the arm rather than reassembled here, so a test cannot pass against a
    store the arm does not read from or a rendering it does not ship.
    """
    if question.id not in _CONTEXTS:
        _CONTEXTS[question.id] = bl.memvara_structured(question, hz.load_scenario()[1])
    return _CONTEXTS[question.id]


def real_questions() -> dict[str, object]:
    return {q.id: q for q in hz.load_scenario()[0]}


def test_the_structured_arm_returns_the_authored_answer_for_every_slot_question():
    """Sixteen of the twenty, asserted against `Question.gold` with no reader involved.

    The read is `slot_values`: what the library says is live in that slot at the instant
    the question asks about — `valid_at=about` for a historical question, present belief
    otherwise. Every value it returns must be named in the gold, and the slot must hold
    exactly one value unless the gold names two.

    This is the whole claim. If it passes, the bitemporal path answered the question; a
    reader's score on top of it is then a measurement of the reader.
    """
    questions = real_questions()
    assert set(ANSWERING_SLOT) <= set(questions), "a mapping for a question that is gone"

    for qid, (subject, predicate) in ANSWERING_SLOT.items():
        q = questions[qid]
        got = bl.slot_values(real_store(q), subject, predicate, valid_at=q.about)
        assert got, f"{qid}: the slot that answers it came back empty"
        expected = 2 if qid in MULTI_VALUED else 1
        assert len(got) == expected, f"{qid}: {got} — a single-valued slot with {len(got)}"
        for value in got:
            assert value.lower() in q.gold.lower(), (
                f"{qid}: the library says {value!r}, the answer key says {q.gold!r}"
            )


def test_the_structured_arm_never_serves_the_trap_as_a_live_value():
    """The other half, and the half a before/after claim actually rests on.

    `Question.trap` is the superseded or retracted value this corpus states more recently
    and more emphatically than the standing one. Getting the gold right while also
    surfacing the trap is what a reader does; a slot read either returns the trap or it
    does not, and here it must not — including on the two `retired` slots, where the trap
    was never true at any world-time and no `valid_at` may resurrect it.
    """
    questions = real_questions()
    for qid, (subject, predicate) in ANSWERING_SLOT.items():
        q = questions[qid]
        if q.trap is None:
            continue
        got = bl.slot_values(real_store(q), subject, predicate, valid_at=q.about)
        assert q.trap.lower() not in {v.lower() for v in got}, f"{qid}: served the trap"


def test_the_ten_day_window_needs_two_slots_and_the_library_keeps_them_apart():
    """The sharpest question in the set, and the one a single address field cannot express.

    Between 26 July and 5 August the account has two different addresses on it, having
    had one for six months. `q_billing_address_on_30_july` is answerable only if delivery
    and billing are separate slots with separate valid intervals — asserted here as the
    two of them disagreeing at one instant and agreeing on either side of the window.
    """
    q = real_questions()["q_billing_address_on_30_july"]
    mem = real_store(q)
    old, new = "41 Coldharbour Road, Lewes, BN7 2GT", \
        "Bramble Cottage, Ditchling Road, Westmeston, BN6 8XA"

    inside = q.about
    assert bl.slot_values(mem, "account", "billing_address", valid_at=inside) == [old]
    assert bl.slot_values(mem, "account", "delivery_address", valid_at=inside) == [new]

    before = datetime(2026, 7, 1, tzinfo=UTC)
    after = datetime(2026, 8, 10, tzinfo=UTC)
    for when, expected in ((before, old), (after, new)):
        assert bl.slot_values(mem, "account", "billing_address", valid_at=when) == \
            bl.slot_values(mem, "account", "delivery_address", valid_at=when) == [expected]


def _in_force(question, valid_at) -> set[str]:
    """The indexed sentences of the answering slot's values at `valid_at`."""
    subject, predicate = ANSWERING_SLOT[question.id]
    return {c.text for c in real_store(question).get_all(valid_at=valid_at)
            if c.subject == subject and c.predicate == predicate}


def test_the_answering_fact_reaches_the_prompt_for_every_slot_question():
    """Knowing the answer and shipping it are two things, and only one is tested above.

    Every other assertion in this section reads the store directly, which is the right
    instrument for "does the library know it" and the wrong one for "does the arm put it
    in front of a reader": an arm that resolved the slot perfectly and then let the fact
    fall off the end of a twelve-slot budget would pass all of them and hand the reader
    nothing.

    So this goes through `memvara_structured` end to end and checks the claim block, for
    all sixteen. It is also what pins `Write.text`: with the facts indexed under
    `remember()`'s default "<subject> <predicate> <object>" rendering instead of a
    sentence, the delivery address loses this corpus's shipping question to a raw turn
    containing the word "node", and the answer never reaches the prompt at all.
    """
    questions = real_questions()
    for qid in ANSWERING_SLOT:
        q = questions[qid]
        lines = claim_lines(real_context(q).text)
        assert _in_force(q, q.about) <= lines, (
            f"{qid}: the answering fact never reaches the claim block, so whatever the "
            "store knows, the reader does not see it"
        )


def test_the_historical_questions_do_not_carry_todays_value_into_the_claim_block():
    """The `about` read has to survive all the way into the prompt, not just into a test.

    Asserted on the sentence each fact was indexed under rather than on a bare address
    string, because on 30 July the *delivery* address really had moved and naming the new
    address is correct there; it is naming it as the **billing** address that is wrong.
    """
    dated = [q for q in real_questions().values() if q.about is not None]
    assert len(dated) == 4, "the corpus sets `about` on four questions"

    for q in dated:
        back_then, standing = _in_force(q, q.about), _in_force(q, None)
        assert back_then and back_then != standing, f"{q.id}: nothing moved in this slot"
        assert not (standing - back_then) & claim_lines(real_context(q).text), (
            f"{q.id}: the claim block names today's value for a question about the past"
        )


def test_the_plan_history_question_is_answered_by_the_timeline_itself():
    """`q_plan_history` asks for the whole sequence, so no single slot read answers it.

    `history()` does, and it returns the sequence the gold names — including the dates,
    which are `valid_from` and are checked against the gold's own wording rather than
    against a second answer key. Two `ended` and one `live`: nothing here was ever wrong.
    """
    q = real_questions()["q_plan_history"]
    assert q.about is None, "a whole sequence has no single valid-time instant"
    timeline = real_store(q).history("account", "plan")

    assert [c.object for c in timeline] == ["Home", "Pro", "Home"]
    assert [c.state for c in timeline] == ["ended", "ended", "live"]
    for claim in timeline[1:]:
        stamp = f"{claim.valid_from.day} {claim.valid_from:%B %Y}"
        assert stamp in q.gold, f"{stamp!r} is not in {q.gold!r}"


def test_the_correction_audit_names_exactly_the_two_records_that_were_wrong():
    """`q_which_were_corrections` — the most diagnostic question in the set, answered
    structurally.

    The library answers it with two reads and no prose: `states=["retired"]` is everything
    we stopped believing, `states=["ended"]` is everything that stopped being true. The
    split has to come out exactly as the gold states it, and it is the split no store with
    one clock can produce at all — in such a store both populations are just "the old
    value".
    """
    q = real_questions()["q_which_were_corrections"]
    mem = real_store(q)

    retired = {(c.subject, c.predicate) for c in mem.get_all(states=["retired"])}
    ended = {(c.subject, c.predicate) for c in mem.get_all(states=["ended"])}
    assert retired == {("account", "mobile"), ("main_unit", "serial")}
    assert ended == {("account", "plan"), ("account", "delivery_address"),
                     ("account", "billing_address"), ("account", "contact_preference")}
    assert not retired & ended, "a slot in both populations is a closure written twice"

    # And the gold says the same thing in words, which is what makes this the answer to
    # that question rather than a coincidence about this store.
    for wrong in ("mobile", "serial"):
        assert wrong in q.gold
    for changed in ("plan", "delivery address", "billing address", "contact preference"):
        assert changed in q.gold


def test_a_retired_value_answers_nothing_at_any_world_time():
    """What `retired` means, asserted rather than assumed, on both corrections.

    A superseded value keeps answering `valid_at=<back then>`; a retracted one answers
    nothing anywhere, because it was never true. This is the property that makes the two
    `correction` questions answerable and it is what an arm that retired everything, or
    ended everything, would destroy in one direction or the other.
    """
    mem = real_store(real_questions()["q_serial_correction"])
    for instant in (datetime(2026, 2, 20, tzinfo=UTC), datetime(2026, 3, 20, tzinfo=UTC),
                    datetime(2026, 8, 1, tzinfo=UTC), None):
        assert bl.slot_values(mem, "main_unit", "serial", valid_at=instant) == \
            ["HX7-8802-D"]
        assert bl.slot_values(mem, "account", "mobile", valid_at=instant) == \
            ["07700 900 811"]

    # The superseded address, by contrast, still answers the world-time it held.
    assert bl.slot_values(mem, "account", "delivery_address",
                          valid_at=datetime(2026, 3, 20, tzinfo=UTC)) == \
        ["41 Coldharbour Road, Lewes, BN7 2GT"]


def test_a_replayed_correction_records_when_belief_changed_not_when_it_was_replayed():
    """Why the two corrections go through `supersede(at=...)` and not `remember()`.

    `remember(close="retired")` retires the right claim, and stamps the retirement with
    the *wall clock*, because the reconciler is handed `utcnow()`. Replaying an archived
    history would then record that we stopped believing the transposed mobile number this
    afternoon rather than on 13 February — the value would be correct, the audit trail
    fiction, and every `known_at=` query over the correction wrong. `supersede` is the one
    write path that takes the instant.

    Asserted to the minute against the turn where the customer says it, because "roughly
    February" is exactly the level of accuracy a wall-clock stamp already has.
    """
    mem = real_store(real_questions()["q_mobile_correction"])
    retired = {c.object: c for c in mem.get_all(states=["retired"])}
    corrections = {f.obj: f for f in bl.SUPPORT_FACTS if f.mode == "correct"}

    assert set(retired) == {"07700 900 118", "HX2-4419-B"}
    for claim in retired.values():
        successor = mem.get(claim.invalidated_by)
        assert successor is not None, "a retirement must point at what displaced it"
        assert claim.invalidated_at == corrections[successor.object].at
        assert claim.valid_to is None, (
            "a correction witnesses no world event, so valid time must stay open"
        )


def test_the_structured_arm_holds_nothing_that_could_answer_the_two_unanswerables():
    """The two questions the arm cannot answer, and the reason is that it should not.

    The Pro plan's price and the card on file are never stated, so there is no field for a
    ticketing system to have. This is a *negative* assertion and weaker evidence than the
    eighteen above — it says a reader has nothing to hallucinate from, not that the arm
    produced an answer. Reported as such rather than counted alongside them.
    """
    questions = real_questions()
    for qid in ("q_pro_price", "q_card_on_file"):
        mem = real_store(questions[qid])
        everything = " ".join(c.text for c in
                              mem.get_all(states=["live", "ended", "retired"]))
        assert "£" not in everything, f"{qid}: a price reached the claim tier"
        assert "card" not in everything.lower(), f"{qid}: a card reached the claim tier"
    assert {s.name for s in bl.SUPPORT_PREDICATES} & {"price", "card"} == set()


def test_the_plain_memvara_arm_extracts_no_claims_at_all_from_the_real_corpus():
    """The finding that makes the fifth arm necessary, pinned so it cannot be forgotten.

    Sixty-four turns of ordinary support prose, and the shipped defaults produce **zero**
    claims: the rule extractor's vocabulary is first-person declaratives and a support
    history is not written that way, and with no `llm=` there is no tier behind it. An
    empty claim tier means no supersession, no valid-time closing and no bitemporal
    reasoning of any kind — the `memvara` arm is lexical episode retrieval with a
    different ranker, and its score cannot be read as a measurement of this library's
    central claim.

    It is a real deployment configuration and it stays in the comparison for that reason.
    It is simply not the one the product is about.
    """
    questions, turns = hz.load_scenario()
    q = questions[0]
    plain, degraded = bl.build_memory(bl.visible_turns(q, turns),
                                      max_episodes=bl.DEFAULT_K)
    assert degraded is True
    assert len(turns) == 64
    assert plain.get_all(states=["live", "ended", "retired"]) == []

    rich = real_store(q)
    assert len(rich.get_all()) == 9
    assert len(rich.get_all(states=["ended"])) == 6
    assert len(rich.get_all(states=["retired"])) == 2


def test_every_structured_fact_is_grounded_in_a_turn_of_the_real_transcript():
    """`SUPPORT_FACTS` restates this corpus's values, so it can drift from it.

    Every object written must appear verbatim somewhere in the history, and every write
    must be stamped at an instant a turn actually happened — which is what stops a value
    being invented, mistyped, or dated to a day nobody said anything. A drift then fails
    here rather than quietly filling the store with strings the answer key has never heard
    of.
    """
    _, turns = hz.load_scenario()
    blob = " ".join(t.text for t in turns)
    instants = {t.at for t in turns}
    for fact in bl.SUPPORT_FACTS:
        assert fact.obj in blob or fact.obj == "14", f"{fact.obj!r} is in no turn"
        assert fact.at in instants, f"{fact.at} is not a turn's timestamp"
        assert fact.valid_from in instants, f"{fact.valid_from} is not a turn's timestamp"
        assert fact.valid_from <= fact.at, "a fact cannot become true after it is recorded"
        assert fact.obj in fact.text, "the indexed sentence must contain the value"


def test_the_structured_integration_is_small_enough_that_properly_is_a_fair_word():
    """"It works if you integrate properly" is only a good answer if properly is small.

    Measured rather than claimed, and pinned with a ceiling so it stays small: eight
    predicate declarations, seventeen fact rows and a short loop are the entire
    integration. If that ever triples, the honest headline changes from "declare your
    schema" to "rewrite your write path", and this test is where that becomes visible.
    """
    predicates = len(bl.SUPPORT_PREDICATES)
    facts = len(bl.SUPPORT_FACTS)
    loop = _statements(bl.apply_facts)

    assert predicates == 8 and facts == 17
    assert loop <= 12, f"apply_facts is {loop} statements"
    assert predicates + facts + loop <= 40


def _statements(fn) -> int:
    """Executable statements in a function, its docstring excluded.

    Counted from the AST rather than from lines, because this file's line count is mostly
    prose and a "lines of code" number that moved when a comment was added would not be
    measuring the integration.
    """
    body = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0].body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    return sum(1 for top in body for node in ast.walk(top)
               if isinstance(node, ast.stmt))
