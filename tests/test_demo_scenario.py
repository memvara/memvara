"""The authored support corpus and its answer key, checked for the properties a run depends on.

Everything downstream — `demo/baselines.py`, the harness, the results table — treats
`demo/scenario.py` as ground truth. Ground truth that can drift is not ground truth, so
this file pins the parts of it that a plausible future edit would break silently:

* determinism, because two arms are only comparable if they saw the same corpus;
* the four `kind`s, because a question set that quietly loses `correction` still runs and
  still reports a number, and that number would no longer be about this library;
* the superseded/standing pairs, by value, in tests named after the fact they pin;
* the *design* of the corpus — that every superseded value is re-surfaced after the value
  that replaced it. A well-meaning edit that tidies the transcript into chronological
  hygiene would make the corpus easy without making any single assertion false, and
  `test_the_superseded_values_are_re_surfaced_after_the_values_that_replaced_them` is the
  one thing standing in the way of that;
* that every gold is grounded in a turn that exists, which is what "authored, not derived"
  means operationally.
"""

from __future__ import annotations

import dataclasses
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

# `demo/` is a package at the repo root, not inside `memvara`, and the repo root is only
# on `sys.path` when pytest happens to be run from it. Inserting it explicitly is the
# same thing `tests/test_bench_eval.py` does for `bench/`, and for the same reason: the
# layout should not have to change for the benefit of the test suite.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo import scenario  # noqa: E402
from demo.scenario import Question, Turn, conversation, questions  # noqa: E402

UTC = timezone.utc

KINDS = {"current", "historical", "correction", "unanswerable"}
ROLES = {"user", "assistant"}

# The two values of every fact that moves, spelled out here rather than imported from
# `demo.scenario`. Importing the module's own constants would make these tests agree with
# whatever the module says, which is exactly the failure mode being guarded against: a
# typo in `_NEW_ADDRESS` would propagate into the answer key and into the assertion that
# is supposed to catch it.
OLD_ADDRESS = "41 Coldharbour Road, Lewes, BN7 2GT"
NEW_ADDRESS = "Bramble Cottage, Ditchling Road, Westmeston, BN6 8XA"
WRONG_SERIAL = "HX2-4419-B"
SERIAL = "HX7-8802-D"
WRONG_MOBILE = "07700 900 118"
MOBILE = "07700 900 811"


def by_id(qid: str) -> Question:
    return next(q for q in questions() if q.id == qid)


def turns_containing(needle: str) -> list[Turn]:
    return [t for t in conversation() if needle in t.text]


# --- the contract ---------------------------------------------------------------


def test_conversation_and_questions_are_deterministic_across_calls():
    """Two calls, identical content, and never the same list object.

    The identity half matters as much as the equality half. Every arm in
    `demo/baselines.py` sorts and slices what it is handed; if that were the module's own
    list, arm three would run on a corpus arm one had reordered, and nothing would say so.
    """
    assert conversation() == conversation()
    assert questions() == questions()
    assert conversation() is not conversation()
    assert questions() is not questions()


def test_the_turns_are_in_chronological_order_with_no_two_at_the_same_instant():
    """Strictly increasing, not merely sorted.

    `visible_turns` in the baselines cuts at `t.at <= asked_at`, and a tie on the boundary
    would put two turns in a context in whichever order the sort happened to be stable in.
    """
    ats = [t.at for t in conversation()]
    assert ats == sorted(ats)
    assert all(a < b for a, b in zip(ats, ats[1:]))


def test_every_timestamp_is_timezone_aware_utc():
    """Naive datetimes compare-raise against aware ones, and the comparison is the cutoff.

    `memvara.types.as_utc` would rescue a naive value inside the library, which is exactly
    why the corpus has to be right here: the rescue means a naive timestamp produces a
    plausible run rather than an error.
    """
    for t in conversation():
        assert t.at.tzinfo is not None
        assert t.at.utcoffset() == UTC.utcoffset(None)
    for q in questions():
        assert q.asked_at.tzinfo is not None
        assert q.asked_at.utcoffset() == UTC.utcoffset(None)


def test_every_role_is_one_a_reader_prompt_can_use():
    assert {t.role for t in conversation()} == ROLES


def test_the_corpus_is_a_conversation_rather_than_a_list_of_facts():
    """Turns alternate speakers, and none of them is a bare field assignment.

    The corpus is only a fair test of extraction if it reads like a transcript. Length is a
    crude proxy for that, but it catches the specific way this file could rot: someone
    shortening the turns to "Plan: Pro" to make a question easier. The floor is on the mean
    rather than on every turn, because "HX2-4419-B." is a whole turn here and is exactly
    what a person says when they have been asked to read out a serial.
    """
    turns = conversation()
    assert len(turns) >= 50
    assert sum(len(t.text) for t in turns) / len(turns) >= 100
    assert all(t.text.strip() for t in turns)

    # Within a ticket the two speakers alternate, and every ticket is opened by the
    # customer. Alternation across the whole corpus would be the wrong assertion: a
    # customer's last word on one ticket is followed by their first word on the next,
    # weeks later, which is what a support history looks like.
    tickets: list[list[Turn]] = [[turns[0]]]
    for previous, turn in zip(turns, turns[1:]):
        if (turn.at - previous.at).total_seconds() < 6 * 3600:
            tickets[-1].append(turn)
        else:
            tickets.append([turn])
    assert len(tickets) >= 10
    for ticket in tickets:
        assert ticket[0].role == "user"
        assert all(a.role != b.role for a, b in zip(ticket, ticket[1:]))


def test_the_dataclasses_are_frozen_so_an_arm_cannot_edit_the_corpus_it_is_scored_on():
    turn = conversation()[0]
    question = questions()[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        turn.text = "different"          # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        question.gold = "different"      # type: ignore[misc]


def test_the_package_re_exports_the_scenario_so_a_caller_need_not_know_the_module_name():
    from demo import conversation as pkg_conversation
    from demo import questions as pkg_questions

    assert pkg_conversation() == conversation()
    assert pkg_questions() == questions()
    assert scenario.Turn is Turn and scenario.Question is Question


# --- the question set -----------------------------------------------------------


def test_all_four_kinds_are_represented():
    """A missing kind is a silently different experiment.

    `unanswerable` is the one at risk: it is the only kind whose gold is a refusal, and it
    is the first thing an optimiser deletes when the headline percentage looks low.
    """
    assert {q.kind for q in questions()} == KINDS


def test_question_ids_are_unique_and_the_load_bearing_ones_are_present():
    """Results tables key on `id`, and two rows with one id is a merge that loses data."""
    ids = [q.id for q in questions()]
    assert len(ids) == len(set(ids))
    # The pair the product's thesis rests on, named so that deleting either fails here
    # rather than in a report nobody re-derives.
    assert {"q_plan_current", "q_plan_mid_march"} <= set(ids)


def test_every_gold_is_a_non_empty_authored_answer():
    for q in questions():
        assert q.gold.strip(), q.id
        assert q.gold == q.gold.strip(), q.id


def test_a_trap_is_either_absent_or_a_real_alternative_answer():
    """`trap=None` is a claim about the failure mode, and `trap=""` is a bug.

    An empty string would be counted as a trap hit by any `trap in answer` grader, on every
    answer, which would put the number this corpus exists to produce at 100% for free.
    """
    for q in questions():
        if q.trap is None:
            continue
        assert q.trap.strip(), q.id
        assert q.trap != q.gold, q.id


def test_some_questions_have_no_trap_because_their_failure_mode_is_diffuse():
    """The discipline the brief asks for, asserted rather than trusted.

    If every question had a trap, "how many gave the superseded answer" would be measuring
    the author's imagination on the questions where no single wrong answer exists.
    """
    without = {q.id for q in questions() if q.trap is None}
    assert without >= {"q_account_name", "q_plan_history", "q_which_were_corrections"}


def test_a_trap_is_not_a_substring_of_its_gold_except_where_the_answer_must_name_both():
    """Containment scoring has to be usable on three of the four kinds.

    A `correction` gold names the retracted value on purpose — "it was X and is now Y" is
    the whole answer — so `trap in gold` is true there by construction and a grader has to
    score those on the standing value instead. Everywhere else that overlap would be an
    accident, and it would report every correct answer as a trap hit.
    """
    for q in questions():
        if q.trap is None:
            continue
        overlaps = q.trap.lower() in q.gold.lower()
        if q.kind == "correction":
            assert overlaps, f"{q.id}: a correction gold should name the retracted value"
        else:
            assert not overlaps, f"{q.id}: trap {q.trap!r} is inside its own gold"


def test_the_current_and_historical_questions_trap_in_opposite_directions():
    """The property that stops one heuristic from passing the whole set.

    For a `current` question the trap is the old value; for a `historical` one it is the
    value in force now. Answer everything with the most recent thing you retrieved and the
    historical group fails; answer everything with the most emphatic and the current group
    fails. Pinned on the plan pair, where both golds are single words.
    """
    now, then = by_id("q_plan_current"), by_id("q_plan_mid_march")
    assert (now.gold, now.trap) == ("Home.", "Pro")
    assert (then.gold, then.trap) == ("Pro.", "Home")
    assert now.gold.rstrip(".") == then.trap
    assert then.gold.rstrip(".") == now.trap


def test_every_question_is_asked_at_or_after_the_history_it_is_asked_about():
    """No question is asked before the corpus starts, and most are asked after it ends."""
    turns = conversation()
    first, last = turns[0].at, turns[-1].at
    asked = [q.asked_at for q in questions()]
    assert all(a > first for a in asked)
    assert any(a > last for a in asked)


def test_two_questions_are_asked_mid_history_so_that_asked_at_is_load_bearing():
    """These are the questions that catch an arm reading past the cutoff.

    `demo/baselines.py` truncates every arm at `asked_at`. If that were removed, or if an
    arm ingested the whole transcript, these two would flip to the value that superseded
    the correct one — which is what makes them worth their place rather than duplicates of
    the questions with the same wording.
    """
    turns = conversation()
    mid = [q for q in questions() if q.asked_at < turns[-1].at]
    assert {q.id for q in mid} == {"q_plan_current_in_april", "q_serial_current_in_may"}

    april = by_id("q_plan_current_in_april")
    assert april.gold == "Pro."
    # Pro was live at the ask; the downgrade that makes `q_plan_current` say Home had not
    # happened yet. Both facts are needed for the question to mean anything.
    upgrade = turns_containing("on Pro as of today, 3 March")[0]
    downgrade = turns_containing("back on Home from today, 19 June")[0]
    assert upgrade.at < april.asked_at < downgrade.at


def test_the_matched_pairs_differ_only_in_when_they_are_asked():
    """Same question text, same fact, two ask instants. One variable at a time.

    `q_plan_current_in_april` against `q_plan_current` isolates *time* — the two golds are
    opposite values. `q_serial_current_in_may` against `q_serial_current` isolates the late
    re-surfacing of the retired serial on 6 August, since the May ask cannot see it and the
    gold is the same either way. Reword either half and the pair stops being a control,
    which is a thing that can happen in a tidy-up without anyone deciding to do it.
    """
    plan_now, plan_april = by_id("q_plan_current"), by_id("q_plan_current_in_april")
    assert plan_now.text == plan_april.text
    assert plan_now.gold != plan_april.gold
    assert plan_now.gold.rstrip(".") == plan_april.trap
    assert plan_april.gold.rstrip(".") == plan_now.trap

    serial_now, serial_may = by_id("q_serial_current"), by_id("q_serial_current_in_may")
    assert serial_now.text == serial_may.text
    assert serial_now.gold == serial_may.gold == SERIAL
    assert serial_now.trap == serial_may.trap == WRONG_SERIAL
    # The turn that makes the pair a measurement of something: the customer offering the
    # retired serial again, visible to one ask and not to the other.
    resurfacing = turns_containing("Is it the 4419 one?")[0]
    assert serial_may.asked_at < resurfacing.at < serial_now.asked_at


# --- the facts, pinned one at a time --------------------------------------------


def test_the_plan_pair_pins_home_now_and_pro_in_march():
    """The product's entire thesis as two rows of a table.

    If a later edit flattens the plan history — drops the downgrade, or moves the upgrade
    outside March — one of these two golds becomes wrong while the other still passes, and
    the run would keep reporting a percentage.
    """
    assert by_id("q_plan_current").gold == "Home."
    assert by_id("q_plan_current").trap == "Pro"
    assert by_id("q_plan_mid_march").gold == "Pro."
    assert by_id("q_plan_mid_march").trap == "Home"
    assert turns_containing("on Pro as of today, 3 March")
    assert turns_containing("back on Home from today, 19 June")


def test_the_serial_pins_a_retired_record_rather_than_a_changed_one():
    """The unit was never replaced; the label was misread.

    This is the pair that separates `retired` from `ended`. If the transcript were edited
    so the customer received a new unit, every assertion about the serial's *values* would
    still hold and the corpus would have quietly stopped testing transaction time — so the
    justifying language is pinned too.
    """
    current = by_id("q_serial_current")
    assert (current.gold, current.trap) == (SERIAL, WRONG_SERIAL)
    correction = by_id("q_serial_correction")
    assert correction.kind == "correction"
    assert SERIAL in correction.gold and WRONG_SERIAL in correction.gold
    assert "never changed" in correction.gold
    # The turn that makes it a misreading rather than a replacement.
    assert turns_containing("correction to a misread label")
    assert turns_containing("has never been the unit's number")


def test_the_mobile_pins_a_transposition_that_was_never_the_customers_number():
    correction = by_id("q_mobile_correction")
    assert correction.kind == "correction"
    assert MOBILE in correction.gold and WRONG_MOBILE in correction.gold
    assert by_id("q_mobile_current").gold == MOBILE
    assert by_id("q_mobile_current").trap == WRONG_MOBILE
    assert turns_containing("it's been wrong since the moment I gave it to you")


def test_the_billing_address_lagged_the_delivery_address_by_ten_days():
    """The ten-day window is the sharpest thing in the corpus and the easiest to lose.

    On 30 July the delivery address had moved and the billing address had not. An edit that
    "tidies" the 5 August ticket by having both change on 26 July would delete the only
    question in the set that a store with one address field cannot answer at all.
    """
    delivery = turns_containing("changed the delivery address to Bramble Cottage")[0]
    billing = turns_containing("still 41 Coldharbour Road until this morning")[0]
    assert delivery.at == datetime(2026, 7, 26, 18, 31, tzinfo=UTC)
    assert billing.at == datetime(2026, 8, 5, 9, 24, tzinfo=UTC)
    assert (billing.at.date() - delivery.at.date()).days == 10

    on_30_july = by_id("q_billing_address_on_30_july")
    assert on_30_july.kind == "historical"
    assert on_30_july.gold == OLD_ADDRESS
    assert on_30_july.trap == NEW_ADDRESS
    # And the two address questions asked about *now* both point the other way.
    assert by_id("q_shipping_current").gold == NEW_ADDRESS
    assert by_id("q_billing_current").gold == NEW_ADDRESS


def test_the_contact_preference_reversed_without_anyone_having_been_wrong():
    """February's preference was true in February. That is what makes it `ended`.

    Paired with the mobile number, which is the same channel recorded wrongly, this is the
    corpus's cleanest side-by-side of the two closures.
    """
    assert by_id("q_contact_current").gold == "By email."
    assert by_id("q_contact_current").trap == "By phone or text"
    assert by_id("q_contact_april").gold == "By phone or text."
    assert by_id("q_contact_april").trap == "By email"
    assert turns_containing("I know I asked for calls back in February")


def test_the_two_controls_never_change_and_carry_no_trap():
    """A fact stated once and never touched.

    Without these, a system that reports change everywhere scores the same as one that
    reports it where it happened.
    """
    for qid, gold in (("q_account_name", "Wray & Daughter Joinery."),
                      ("q_billing_day", "The 14th of the month.")):
        q = by_id(qid)
        assert q.gold == gold
        assert q.trap is None
    assert len(turns_containing("Wray & Daughter Joinery")) == 2   # asked, and confirmed
    assert len(turns_containing("the 14th of the month")) == 1


def test_the_unanswerable_questions_are_genuinely_unanswerable_from_the_corpus():
    """Asserted against the text, not against the author's memory of the text.

    A price for Pro or a card issuer appearing in a later edit would turn an honest refusal
    into a wrong answer, and the reader would be marked down for being right.
    """
    corpus = " ".join(t.text for t in conversation()).lower()
    # The only money in the transcript is the price of one extra node, quoted twice and
    # grumbled about once. It is never attached to a plan, which is what makes the plan's
    # price genuinely absent rather than merely unstated in those words.
    priced = [t for t in conversation() if "£" in t.text or "quid" in t.text]
    assert len(priced) == 3
    assert corpus.count("£79") == 2
    assert all("Pro" not in t.text and "plan" not in t.text for t in priced)
    for absent in ("visa", "mastercard", "amex", "debit card", "ending", "sort code"):
        assert absent not in corpus, absent

    price = by_id("q_pro_price")
    assert price.kind == "unanswerable" and price.trap == "£79"
    assert by_id("q_card_on_file").trap is None
    for q in questions():
        if q.kind == "unanswerable":
            assert q.gold.startswith("Not stated"), q.id


def test_the_superseded_values_are_re_surfaced_after_the_values_that_replaced_them():
    """The corpus is adversarial by construction, and this is the construction.

    "The customer already corrected this, and the agent said the old thing anyway" only
    happens because the old thing keeps coming back into view. Here the last address anyone
    names is the one they moved out of, the last serial a customer says is the retired one,
    and the last plan named is Pro in the past tense. Sort those away and recency alone
    answers most of the set — the corpus would still pass every other test in this file and
    would have stopped measuring anything.
    """

    def last(needle: str) -> datetime:
        found = turns_containing(needle)
        assert found, needle
        return found[-1].at

    # The strong form for the two facts where it holds: the superseded value outlives the
    # standing one in the transcript, rather than merely reappearing once after it. The
    # weak form ("appears after the change") would survive deleting the turns that do the
    # work, because the change itself mentions both values.
    assert last("Coldharbour Road") > last("Bramble Cottage")
    assert last("Pro") > last("Home")

    # The serial is the exception, and deliberately: the customer offers the retired
    # number last (6 August 20:05) and the agent corrects it in the next turn, because a
    # corpus that ends on an uncorrected falsehood has an ambiguous answer key rather than
    # a hard question. The adversarial property is that the last serial a *customer* says
    # is the wrong one — which is what a system summarising the user's own words returns.
    said_by_customer = [t for t in conversation()
                        if t.role == "user" and ("4419" in t.text or SERIAL in t.text)]
    assert "4419" in said_by_customer[-1].text
    assert SERIAL not in said_by_customer[-1].text
    assert last(SERIAL) > last("4419")      # and the desk always has the last word
    # The contact preference does it differently, because there is nowhere later to put
    # it: the turn that reverses the preference names the old one in the same breath, so
    # the most recent turn on the subject is also the one containing the wrong answer.
    reversal = turns_containing("Email from now on please")[0]
    assert "asked for calls" in reversal.text
    assert reversal.at > turns_containing("please do ring")[0].at


# --- the two axes, carried as fields ---------------------------------------------
#
# `closure` and `about` were added after the corpus was written, because the report the
# corpus feeds could not express the difference between two failures that mean opposite
# things. The tests below are what stop them becoming decoration.

#: Every question about a fact whose superseded value was closed on **valid** time — the
#: world changed. Spelled out rather than derived, so that re-labelling one fails here.
ENDED = {
    "q_plan_current", "q_shipping_current", "q_billing_current", "q_contact_current",
    "q_plan_current_in_april", "q_plan_mid_march", "q_shipping_april",
    "q_billing_address_on_30_july", "q_contact_april",
}
#: Every question about a record that was **wrong** — closed on transaction time.
RETIRED = {
    "q_serial_current", "q_mobile_current", "q_serial_current_in_may",
    "q_serial_correction", "q_mobile_correction",
}


def test_every_closure_is_one_of_the_three_values_the_model_has():
    """`ended`, `retired`, or nothing. Any other string is a typo with a plausible reading.

    The vocabulary is `memvara.types.Closure`'s, deliberately: a corpus that invented its
    own words for the two axes would be measuring something it had defined itself.
    """
    assert {q.closure for q in questions()} == {"ended", "retired", None}


def test_the_ended_and_retired_split_matches_the_timeline():
    """Six facts move; two of them move because somebody misread a label.

    The split is the whole reason the field exists, so it is pinned as two literal sets. A
    question that changes sides has changed what it measures, and that has to be a decision
    rather than a diff nobody read.
    """
    ended = {q.id for q in questions() if q.closure == "ended"}
    retired = {q.id for q in questions() if q.closure == "retired"}
    assert ended == ENDED
    assert retired == RETIRED
    assert not ended & retired
    # Plan, both addresses and the contact preference changed in the world; the serial and
    # the mobile number were wrong on paper. Nothing else has a closed value at all.
    assert {q.id for q in questions() if q.closure is None} == {
        "q_account_name", "q_billing_day", "q_plan_history",
        "q_which_were_corrections", "q_pro_price", "q_card_on_file",
    }


def test_closure_is_orthogonal_to_kind_rather_than_a_second_spelling_of_it():
    """The same wrong record, asked two ways, is one closure and two kinds.

    If `closure` could be derived from `kind` it would be worth nothing. It cannot: both
    values appear under more than one kind, and `kind="correction"` is not the same set as
    `closure="retired"` — `q_serial_current` is in the second and not the first.
    """
    assert (by_id("q_serial_current").kind, by_id("q_serial_current").closure) == (
        "current", "retired")
    assert (by_id("q_serial_correction").kind, by_id("q_serial_correction").closure) == (
        "correction", "retired")
    assert (by_id("q_plan_current").kind, by_id("q_plan_current").closure) == (
        "current", "ended")

    for closure in ("ended", "retired"):
        kinds = {q.kind for q in questions() if q.closure == closure}
        assert len(kinds) > 1, f"{closure} only ever appears under {kinds}"
    corrections = {q.id for q in questions() if q.kind == "correction"}
    assert corrections != RETIRED and corrections & RETIRED


def test_a_trap_without_a_closure_is_a_distractor_and_not_a_bitemporal_failure():
    """One question has a trap and no closure, and it is not an inconsistency.

    `q_pro_price`'s trap is £79 — a real price for a different thing, not a value that was
    ever superseded. Giving it is a retrieval or reading failure, not a time-handling one,
    so a trapped rate that mixes it in with the other nine is reporting two different
    things in one column. Anyone breaking the headline down by `closure` gets that for
    free; anyone quoting a single number should know this row is in it.
    """
    odd = {q.id for q in questions() if q.trap is not None and q.closure is None}
    assert odd == {"q_pro_price"}
    # Every other trap in the set is the value the other clock closed.
    for q in questions():
        if q.trap is not None and q.id != "q_pro_price":
            assert q.closure in ("ended", "retired"), q.id


def test_about_is_timezone_aware_wherever_it_is_set():
    for q in questions():
        if q.about is not None:
            assert q.about.tzinfo is not None, q.id
            assert q.about.utcoffset() == UTC.utcoffset(None), q.id


def test_a_retired_record_has_no_valid_time_instant_to_be_asked_about():
    """`about` is `None` on every `retired` question, and that is a consequence, not a gap.

    A retracted value was never true at any world-time, so there is no `valid_at=` that
    returns it. Those questions move on the belief axis instead. If one of them ever
    acquires an `about`, the field has been misunderstood as "when was this asked about"
    rather than "when was this true".
    """
    for q in questions():
        if q.closure == "retired":
            assert q.about is None, q.id


def test_only_current_kind_questions_leave_about_unset_because_asked_at_already_says_now():
    """`about=None` on a `current` question means "the instant in `asked_at`".

    The two are not interchangeable and the corpus proves it: `q_plan_current_in_april` is
    `current` with no `about`, and its answer depends entirely on `asked_at`.
    """
    for q in questions():
        if q.kind == "current":
            assert q.about is None, q.id
    assert by_id("q_plan_current_in_april").about is None
    assert by_id("q_plan_current_in_april").asked_at == datetime(2026, 4, 20, 9, 0,
                                                                 tzinfo=UTC)


def test_the_only_historical_question_without_an_about_is_the_one_that_spans_the_history():
    """Every `historical` question names the instant it asks about, with one exception.

    `q_plan_history` asks for the whole sequence, so there is no single valid-time instant
    to put in the field and inventing one would make it a lie. The exception is named here
    rather than allowed as a general gap: a *new* historical question with no `about` fails
    this test, which is the behaviour that makes the field trustworthy for a harness that
    wants to derive `valid_at=` from it.
    """
    without = {q.id for q in questions() if q.kind == "historical" and q.about is None}
    assert without == {"q_plan_history"}
    assert by_id("q_plan_history").gold.count("2026") == 2      # it carries its own dates


#: For each `historical` question that names an instant: the turn phrase that opened the
#: interval its gold was true in, and the turn phrase that closed it. `about` has to fall
#: strictly inside, or `valid_at=about` would not return `gold` — which is the only reason
#: the field is worth carrying.
INTERVALS: dict[str, tuple[str, str]] = {
    "q_plan_mid_march": ("on Pro as of today, 3 March",
                         "back on Home from today, 19 June"),
    "q_shipping_april": ("Everything comes to 41 Coldharbour Road",
                         "changed the delivery address to Bramble Cottage"),
    "q_billing_address_on_30_july": ("Paper invoices go to the billing address",
                                     "changed it now to Bramble Cottage"),
    "q_contact_april": ("set the contact preference on the account to phone",
                        "changed the contact preference on the account to email"),
}


def test_each_about_falls_inside_the_interval_its_gold_was_true_in():
    """The field is checked for meaning, not merely for being a date.

    A plausible-looking instant that sits on the wrong side of a change would make every
    automated `valid_at=` check agree with a wrong answer, and nothing else in this file
    would notice. The bounds are turns, so this also fails if the history is re-dated.
    """
    dated = {q.id for q in questions() if q.about is not None}
    assert set(INTERVALS) == dated, set(INTERVALS) ^ dated

    for qid, (opens, closes) in INTERVALS.items():
        question = by_id(qid)
        assert question.about is not None                      # for the type checker
        opened = turns_containing(opens)[0]
        closed = turns_containing(closes)[0]
        assert opened.at < question.about < closed.at, qid
        # And the question is asked from after the interval closed, which is what makes it
        # a question about the past rather than about now.
        assert question.asked_at > closed.at, qid


# --- authored, not derived -------------------------------------------------------

#: For each question: a phrase from the turn that justifies the gold, and one from the
#: turn that justifies the trap. `None` where there is nothing to justify — an
#: unanswerable question has no supporting turn by definition, and a question with no trap
#: has no trap to ground. Written by hand alongside the answer key; a new question with no
#: entry here fails `test_every_question_names_the_turn_its_answer_came_from`, which is the
#: point. This table is the evidence that the golds were read off the transcript rather
#: than recorded from a run of the system under test.
EVIDENCE: dict[str, tuple[str | None, str | None]] = {
    "q_plan_current": ("back on Home from today, 19 June", "when I was on Pro"),
    "q_shipping_current": ("delivery address to Bramble Cottage", "41 Coldharbour Road"),
    "q_billing_current": ("changed it now to Bramble Cottage", "41 Coldharbour Road"),
    "q_contact_current": ("Email from now on please", "please do ring"),
    "q_serial_current": (SERIAL, WRONG_SERIAL),
    "q_mobile_current": (MOBILE, WRONG_MOBILE),
    "q_account_name": ("Wray & Daughter Joinery", None),
    "q_billing_day": ("Billing is the 14th of the month", None),
    "q_plan_current_in_april": ("on Pro as of today, 3 March", "You're on the Home plan"),
    "q_serial_current_in_may": (SERIAL, WRONG_SERIAL),
    "q_plan_mid_march": ("on Pro as of today, 3 March", "back on Home from today"),
    "q_shipping_april": ("out to 41 Coldharbour Road", "Bramble Cottage"),
    "q_billing_address_on_30_july": ("still 41 Coldharbour Road until this morning",
                                     "Bramble Cottage"),
    "q_contact_april": ("Phone or a text", "contact preference on the account to email"),
    "q_plan_history": ("on Pro as of today, 3 March", None),
    "q_serial_correction": ("correction to a misread label", WRONG_SERIAL),
    "q_mobile_correction": ("it's been wrong since the moment I gave it to you",
                            WRONG_MOBILE),
    "q_which_were_corrections": ("as a correction rather than as a new number", None),
    "q_pro_price": (None, "£79"),
    "q_card_on_file": (None, None),
}


def test_every_question_names_the_turn_its_answer_came_from():
    """Each gold and each trap is quoted from a turn that exists, before it was asked.

    Two failures are caught here and nowhere else. A gold whose supporting turn was edited
    away is an answer key that outlived its evidence. A trap quoted from a turn *after*
    `asked_at` is a wrong answer no system could have given, so counting it would be
    counting nothing — which is the specific way the two mid-history questions could rot.
    """
    asked_ids = {q.id for q in questions()}
    assert set(EVIDENCE) == asked_ids, set(EVIDENCE) ^ asked_ids

    for q in questions():
        gold_ev, trap_ev = EVIDENCE[q.id]
        if q.kind == "unanswerable":
            assert gold_ev is None, f"{q.id}: an unanswerable question has no evidence"
        else:
            assert gold_ev is not None, q.id
        if q.trap is None:
            assert trap_ev is None, f"{q.id}: no trap, so nothing to ground"

        for label, needle in (("gold", gold_ev), ("trap", trap_ev)):
            if needle is None:
                continue
            visible = [t for t in turns_containing(needle) if t.at <= q.asked_at]
            assert visible, f"{q.id}: {label} evidence {needle!r} is not in the history"
