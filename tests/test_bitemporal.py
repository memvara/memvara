"""The two time axes, and the four questions they answer.

Bitemporal data has exactly four readings, and until `valid_at`/`known_at` landed this
library could express one of them. `as_of` moves both clocks to the same instant, so it
answers "what did we believe *then*, about *then*" and nothing else — and the reading it
cannot reach is the one the whole model exists for: a correction that arrives in August
about June is invisible to `as_of=June`, because that call rewinds the belief clock past
the correction it is asking about.

The fixture below is built so all four give **four different answers**, which is the only
way to show the axes are genuinely independent rather than two spellings of one filter.

    Jan   we are told: Berlin, and they have lived there since January.
    Jul   we are told: they moved to Paris this month. Berlin's world-interval closes.
    Aug   two corrections land at once. It was Rome from March, not Berlin; and the
          July move was to Lisbon, not Paris. Both mistaken rows are retired, and
          neither is deleted — that is what makes the August audit trail readable.

    question                                    call                     answer
    ------------------------------------------  -----------------------  ------
    1  what do we now believe is true now       (no arguments)           Lisbon
    2  what do we NOW believe was true in June  valid_at=JUNE            Rome
    3  what did we believe in mid-July          known_at=JULY_MID        Paris
    4  what did we believe in June, about June  as_of=JUNE               Berlin

Rows 2 and 3 were unreachable before; row 4 must keep answering exactly as it did.
"""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from memvara import Claim, Episode, HashingEmbedder, Memvara, NullLLM, Scope, utcnow
from memvara.aio import AsyncMemvara
from memvara.schema import Cardinality, PredicateSpec, Volatility
from memvara.store import STATES
from memvara.types import time_axes

TZ = timezone.utc
JAN = datetime(2026, 1, 1, tzinfo=TZ)
MAR = datetime(2026, 3, 1, tzinfo=TZ)
JUNE = datetime(2026, 6, 1, tzinfo=TZ)
JULY = datetime(2026, 7, 1, tzinfo=TZ)
JULY_MID = datetime(2026, 7, 15, tzinfo=TZ)
MID_MAR = datetime(2026, 3, 15, tzinfo=TZ)
AUG = datetime(2026, 8, 1, tzinfo=TZ)

# The reproduction below spans years rather than months, so it stays a past-tense story
# whatever day the suite runs on.
J23 = datetime(2023, 1, 1, tzinfo=TZ)
MID = datetime(2024, 6, 1, tzinfo=TZ)
J26 = datetime(2026, 1, 1, tzinfo=TZ)

# A supersession, a moment after it, and an unrelated retirement of the same row months
# later. Back in 2023 for the same reason: a claim retired "next October" is not retired.
AUG23 = datetime(2023, 8, 1, tzinfo=TZ)
SEP23 = datetime(2023, 9, 1, tzinfo=TZ)
OCT23 = datetime(2023, 10, 1, tzinfo=TZ)

SCOPE = Scope("acme", "alice")


@pytest.fixture()
def mem():
    with Memvara(llm=NullLLM(), embedder=HashingEmbedder(dim=64), tenant="acme",
                 user="alice") as m:
        yield m


def put(mem, obj, *, valid_from, valid_to=None, recorded_at, invalidated_at=None,
        predicate="lives_in", subject="user", scope=SCOPE, **kw):
    """One claim written straight to the store, with both axes stated exactly.

    Deliberately not `remember()`, and the reason has changed. It used to be that the
    write path *could not* produce these rows — it closed both clocks on every
    supersession, so several of the four were reconciled out of existence on the way in.
    That was the bug, and `test_the_four_row_fixture_is_now_expressible_through_the_write_path`
    writes exactly this fixture through the facade to show it is gone.

    What is left is the ordinary reason a read test states its own rows: the read side
    has to be correct about whatever is on disk, including rows some future write path
    would never generate, and a fixture that goes through the writer tests the writer
    twice and the reader not at all.
    """
    claim = Claim(subject=subject, predicate=predicate, object=obj, scope=scope,
                  valid_from=valid_from, valid_to=valid_to, recorded_at=recorded_at,
                  invalidated_at=invalidated_at, **kw)
    mem.store.put_claim(claim)
    mem.store.set_embedding(claim.id, mem.embedder.encode([claim.text])[0])
    return claim


@pytest.fixture()
def four(mem):
    """The story in the module docstring, as four rows. Returns them by city."""
    rows = {
        # Believed from January until the August correction retired it; asserted to
        # have held from January until the July move.
        "Berlin": put(mem, "Berlin", valid_from=JAN, valid_to=JULY,
                      recorded_at=JAN, invalidated_at=AUG),
        # Believed from July until the August correction; still asserted to hold today.
        "Paris": put(mem, "Paris", valid_from=JULY, recorded_at=JULY,
                     invalidated_at=AUG),
        # The late-arriving fact: learned in August, about March to July.
        "Rome": put(mem, "Rome", valid_from=MAR, valid_to=JULY, recorded_at=AUG),
        # The other August correction: where they actually went in July.
        "Lisbon": put(mem, "Lisbon", valid_from=JULY, recorded_at=AUG),
    }
    return rows


def cities(claims):
    return sorted(c.object for c in claims)


# =============================================================================
# The four questions
# =============================================================================

def test_question_one_current_belief_about_the_present(mem, four):
    """The default, unchanged: both clocks read now."""
    assert cities(mem.get_all()) == ["Lisbon"]


def test_question_two_current_belief_about_a_past_moment(mem, four):
    """The reading `as_of` cannot express, and the reason the split exists.

    A fact recorded in August about June satisfies `valid_from <= June` and
    `recorded_at <= now`, but never `recorded_at <= June` — so under a single instant
    the late-arriving correction is unreachable by construction. Asking for it rewound
    the belief clock past the very correction being asked about.
    """
    assert cities(mem.get_all(valid_at=JUNE)) == ["Rome"]
    assert cities(mem.get_all(as_of=JUNE)) == ["Berlin"], "and #4 is still #4"


def test_question_three_past_belief_about_the_present(mem, four):
    """The audit reading: what would this system have answered in mid-July?

    Berlin drops out because its *world* interval had already closed by now; Rome and
    Lisbon because we had not heard them yet. What is left is the answer a caller got in
    July, which is the question a complaint about a July decision starts from.
    """
    assert cities(mem.get_all(known_at=JULY_MID)) == ["Paris"]


def test_question_four_past_belief_about_that_same_past(mem, four):
    """`as_of`, unchanged. It is exact sugar for `valid_at=known_at=T`, so this is not
    a fourth code path — it is the diagonal of the other two."""
    assert cities(mem.get_all(as_of=JUNE)) == ["Berlin"]
    assert cities(mem.get_all(valid_at=JUNE, known_at=JUNE)) == ["Berlin"]


def test_the_four_questions_give_four_different_answers(mem, four):
    """Stated as one assertion, because "the axes are independent" is exactly the claim
    that four distinct answers make and that no three of them can."""
    answers = [
        cities(mem.get_all()),
        cities(mem.get_all(valid_at=JUNE)),
        cities(mem.get_all(known_at=JULY_MID)),
        cities(mem.get_all(as_of=JUNE)),
    ]
    assert answers == [["Lisbon"], ["Rome"], ["Paris"], ["Berlin"]]
    assert len({tuple(a) for a in answers}) == 4


# =============================================================================
# `as_of` is exact sugar, and mixing is a caller error
# =============================================================================

@pytest.mark.parametrize("instant", [JAN, MAR, JUNE, JULY, JULY_MID, AUG])
def test_as_of_is_exactly_the_diagonal_of_the_two_axes(mem, four, instant):
    """The compatibility promise, checked at every instant the fixture has an opinion
    about rather than asserted once. `as_of` is on the published API, in the README and
    in `bench/`, so "unchanged" has to mean identical output, not merely similar."""
    assert (mem.get_all(as_of=instant)
            == mem.get_all(valid_at=instant, known_at=instant))
    assert (mem.count(as_of=instant)
            == mem.count(valid_at=instant, known_at=instant))


def test_time_axes_resolves_the_three_keywords():
    """One resolver behind every facade method, so the rule cannot be spelled eight
    slightly different ways. `None` propagates as `None` rather than being filled with a
    clock, because only the layer that builds the query knows which clock to read."""
    assert time_axes(None, None, None) == (None, None)
    assert time_axes(None, JUNE, None) == (JUNE, None)
    assert time_axes(None, None, AUG) == (None, AUG)
    assert time_axes(None, JUNE, AUG) == (JUNE, AUG)
    assert time_axes(JUNE, None, None) == (JUNE, JUNE)


@pytest.mark.parametrize("kw", [
    {"valid_at": JUNE},
    {"known_at": AUG},
    {"valid_at": JUNE, "known_at": AUG},
])
def test_as_of_with_either_axis_raises_rather_than_picking_one(mem, kw):
    """There is no reading of `as_of=June, known_at=August` in which one of the two is
    not being ignored, and a silently ignored time argument answers a question the
    caller did not ask with nothing in the result to say so."""
    with pytest.raises(ValueError, match="as_of cannot be combined with"):
        mem.get_all(as_of=JUNE, **kw)
    assert sorted(kw) == [k for k in ("known_at", "valid_at") if k in kw]


@pytest.mark.parametrize("method,args", [
    ("search", ("berlin",)),
    ("get_all", ()),
    ("count", ()),
    ("history", ("user", "lives_in")),
    ("why", ("cl_missing",)),
    ("produced", ("ep_missing",)),
    ("neighborhood", ("user",)),
    ("paths_between", ("user", "Berlin")),
])
def test_every_read_that_takes_as_of_refuses_to_mix_it(mem, method, args):
    """One rule, not eight. A method that accepted the mix would be the one an
    integration reaches for, and the answer it returned would look ordinary."""
    with pytest.raises(ValueError, match="as_of cannot be combined with"):
        getattr(mem, method)(*args, as_of=JUNE, known_at=AUG)
    with pytest.raises(ValueError, match="as_of cannot be combined with"):
        getattr(mem.scope(), method)(*args, as_of=JUNE, known_at=AUG)


def test_the_error_names_the_axis_that_clashed(mem):
    """A message that only says "conflicting arguments" makes the caller re-read the
    signature to find out which of the two they passed."""
    with pytest.raises(ValueError, match=r"combined with valid_at\. "):
        mem.count(as_of=JUNE, valid_at=JUNE)
    with pytest.raises(ValueError, match=r"combined with valid_at and known_at\. "):
        mem.count(as_of=JUNE, valid_at=JUNE, known_at=JUNE)


# =============================================================================
# The same axes through retrieval, counting and the traversal
# =============================================================================

def test_search_answers_the_four_questions_too(mem, four):
    """Retrieval is where the axes are actually used; `get_all` only proves the store
    filter works. Both legs of the hybrid retriever have to carry the pair, and the
    Python-side belief floor in `_believed_by` has to read `known_at` rather than
    whatever instant happened to be handy."""
    def found(**kw):
        return sorted(r.claim.object for r in mem.search("user lives in", k=10, **kw))

    assert found() == ["Lisbon"]
    assert found(valid_at=JUNE) == ["Rome"]
    assert found(known_at=JULY_MID) == ["Paris"]
    assert found(as_of=JUNE) == ["Berlin"]


@pytest.mark.parametrize("kw", [
    {}, {"valid_at": JUNE}, {"known_at": JULY_MID}, {"as_of": JUNE},
    {"include_invalidated": True}, {"known_at": JULY, "include_invalidated": True},
])
def test_count_never_disagrees_with_get_all(mem, four, kw):
    """They reach the same store call, but through two independent signatures — so the
    way they drift is one of them forwarding a new keyword and the other dropping it,
    which produces a plausible number rather than an error."""
    assert mem.count(**kw) == len(mem.get_all(**kw)), kw


def test_recency_decays_from_the_belief_clock_not_the_world_clock(mem):
    """Recency answers "how long ago did we last hear this, from where the question
    stands", and the question stands on the belief clock. Two consequences, and the
    second is the one that would have been wrong by default: moving `known_at` moves
    the decay, and moving `valid_at` does not. A `valid_at=June` query scored from June
    would report a fact we heard this month as months stale — on exactly the query that
    exists to surface facts we heard recently about long ago."""
    put(mem, "Rome", valid_from=MAR, recorded_at=AUG)

    def recency(**kw):
        return mem.search("user lives in", k=5, **kw)[0].explain.recency

    assert recency(known_at=AUG) > recency(), "the belief clock moves the decay"
    assert recency(valid_at=JUNE) == pytest.approx(recency(), abs=1e-6), \
        "and the world clock leaves it alone"


def test_a_path_exists_only_when_the_two_clocks_are_apart(mem):
    """The traversal carries the pair, not one instant. The hop `Alice -> Dana` was
    retired in August; the hop `Dana -> Acme` only becomes true in the world in July.
    No single instant has both, so `as_of` cannot return this chain at any argument —
    which is precisely the multi-hop version of the late-arriving fact."""
    put(mem, "Dana", subject="Alice", predicate="reports_to",
        valid_from=JAN, recorded_at=JAN, invalidated_at=JUNE)
    put(mem, "Acme", subject="Dana", predicate="works_at",
        valid_from=JULY, recorded_at=JAN)

    walked = mem.paths_between("Alice", "Acme", valid_at=JULY_MID, known_at=MAR)
    assert [p.render() for p in walked] == ["Alice -reports_to-> Dana -works_at-> Acme"]
    for instant in (JAN, MAR, JUNE, JULY, JULY_MID, AUG, None):
        assert mem.paths_between("Alice", "Acme", as_of=instant) == []


def test_neighborhood_carries_both_axes(mem):
    """Each axis has to be able to hide the same edge on its own, or one of the two is
    being dropped somewhere between the facade and `Store.adjacent`."""
    put(mem, "Dana", subject="Alice", predicate="reports_to",
        valid_from=JULY, recorded_at=JUNE)
    assert mem.neighborhood("Alice", valid_at=MAR) == [], "not true in the world yet"
    assert mem.neighborhood("Alice", known_at=MAR) == [], "and we had not heard it"
    assert mem.neighborhood("Alice", valid_at=JULY_MID, known_at=JULY_MID)


# =============================================================================
# Episodes: one column, bounded by the earlier of the two clocks
# =============================================================================

def test_an_episode_is_hidden_by_whichever_clock_is_earlier(mem):
    """A turn has no separate record time — it happened and we learned of it at the
    same instant — so its single `ts` is both of its clocks and either can hide it.
    Reading only `known_at` here would let a search at `valid_at=June` return a turn
    from July, which is a transcript excerpt from the future."""
    mem.add("I have been thinking about Lisbon", ts=JULY)

    def turns(**kw):
        return [r for r in mem.search("lisbon", k=5, include_episodes=True, **kw)
                if hasattr(r, "episode")]

    assert turns(known_at=AUG, valid_at=AUG), "after the turn on both clocks"
    assert turns(known_at=AUG, valid_at=JUNE) == [], "the world had not reached it"
    assert turns(known_at=JUNE, valid_at=AUG) == [], "nor had we heard it"


# =============================================================================
# `include_invalidated` is left alone, and the three interact predictably
# =============================================================================

def test_include_invalidated_lifts_end_of_life_on_both_axes(mem, four):
    """Unchanged behaviour, restated because the flag now shares a signature with two
    new parameters. It reveals the retired rows (Berlin, Paris) *and* the expired one
    (Rome), because it lifts the whole valid-time interval rather than only its end."""
    assert cities(mem.get_all(include_invalidated=True)) == [
        "Berlin", "Lisbon", "Paris", "Rome"]


def test_include_invalidated_still_cannot_see_past_the_belief_floor(mem, four):
    """The one clause the flag never lifts. "What did we believe in June, including
    what we later retracted" must not include something first heard in August — that is
    knowledge from the future, and it is the only way a bitemporal read can actively
    lie."""
    audit = cities(mem.get_all(known_at=JUNE, include_invalidated=True))
    assert audit == ["Berlin"]
    assert "Rome" not in audit and "Lisbon" not in audit


def test_include_invalidated_makes_valid_at_inert_and_that_is_deliberate(mem, four):
    """Documented consequence rather than a bug, and asserted so it cannot drift into
    one silently: the flag lifts `valid_from` as well as `valid_to`, so the world clock
    has nothing left to constrain. Only `known_at` still bites under it."""
    everything = cities(mem.get_all(include_invalidated=True))
    for instant in (JAN, JUNE, JULY, AUG):
        assert cities(mem.get_all(valid_at=instant, include_invalidated=True)) \
            == everything
    assert cities(mem.get_all(known_at=JULY, include_invalidated=True)) \
        == ["Berlin", "Paris"]


# =============================================================================
# `states`: one boolean was two answers over three populations
# =============================================================================
#
# The four-row fixture holds one of each state and two of the interesting one, which is
# what makes it the right fixture for this:
#
#     Berlin   retired   believed Jan-Aug, corrected away in August
#     Paris    retired   believed Jul-Aug, corrected away in August
#     Rome     ended     still believed; asserted to have held March to July
#     Lisbon   live      still believed, still in force
#
# `include_invalidated` can name {live} and {live, ended, retired} and nothing else, so
# the two rows an August correction audit is *about* cannot be asked for.

def test_only_retired_is_the_audit_include_invalidated_could_never_ask_for(mem, four):
    """The whole point. "Show me everything we stopped believing" is a correction audit,
    and it is exactly the set the boolean cannot name: `False` hides the retired rows,
    `True` returns them buried in everything else, and there is no third value."""
    assert cities(mem.get_all(states=["retired"])) == ["Berlin", "Paris"]
    for flag in (False, True):
        assert cities(mem.get_all(include_invalidated=flag)) != ["Berlin", "Paris"]


def test_ended_is_a_different_population_from_retired_not_a_shade_of_it(mem, four):
    """The distinction the boolean collapsed. Rome stopped being true and we believe
    every word of it; Berlin and Paris are records we withdrew. Asking for one must
    never return the other, or the two closures are back to being one flag."""
    assert cities(mem.get_all(states=["ended"])) == ["Rome"]
    assert cities(mem.get_all(states=["live"])) == ["Lisbon"]
    assert cities(mem.get_all(states=["ended", "retired"])) == ["Berlin", "Paris", "Rome"]


def test_each_state_returns_exactly_the_claims_that_report_it(mem, four):
    """The vocabulary guard. `states=` and `Claim.state` are two spellings of one set of
    three words, in two modules, and a filter that drifted from the property would answer
    a question about a state nothing reports being in."""
    for state in STATES:
        got = mem.get_all(states=[state])
        assert got, state
        assert {c.state for c in got} == {state}


@pytest.mark.parametrize("subset", [
    ("live",), ("ended",), ("retired",), ("live", "ended"), ("live", "retired"),
    ("ended", "retired"), ("live", "ended", "retired"),
])
def test_the_belief_floor_holds_under_every_state_subset(mem, four, subset):
    """The one clause no argument lifts, checked across all seven populations rather
    than only the two the boolean could reach.

    This is where a filter breaks silently. The subsets that want `retired` alongside a
    world-time state are a disjunction, and `AND` binds tighter than `OR` — so a
    predicate assembled as `floor AND retired OR in_force` parses as
    `(floor AND retired) OR in_force` and lets rows through on the world clause with no
    floor at all. Rome and Lisbon were first heard in August; a June audit that returns
    either is answering with knowledge from the future, and nothing about the result says
    so.
    """
    audit = cities(mem.get_all(known_at=JUNE, states=subset))
    assert "Rome" not in audit and "Lisbon" not in audit, subset


def test_include_invalidated_is_an_exact_alias_and_stays_one(mem, four):
    """Not deprecated, not warned about, not approximately equivalent — the same rows.
    Checked with the belief clock moved too, since the alias has to survive composition
    with the axes and not merely match on the default call."""
    for kw in ({}, {"known_at": JULY}, {"valid_at": JUNE}):
        assert mem.get_all(include_invalidated=False, **kw) \
            == mem.get_all(states=["live"], **kw), kw
        assert mem.get_all(include_invalidated=True, **kw) \
            == mem.get_all(states=STATES, **kw), kw


def test_asking_for_all_three_states_is_the_audit_view_valid_at_cannot_narrow(mem, four):
    """The documented discontinuity, pinned so it cannot drift into a surprise. The three
    states do not tile the store — a fact recorded but not yet in force is named by none
    of them — so the complete set is not the union of its parts. It is the belief floor,
    which is what `include_invalidated=True` has always meant, and under it the world
    clock has nothing left to constrain."""
    everything = cities(mem.get_all(states=STATES))
    assert everything == ["Berlin", "Lisbon", "Paris", "Rome"]
    for instant in (JAN, JUNE, JULY, AUG):
        assert cities(mem.get_all(valid_at=instant, states=STATES)) == everything
    # And the parts really do not add up to it, which is why the collapse is stated.
    assert sorted(
        cities(mem.get_all(states=["live"])) + cities(mem.get_all(states=["ended"]))
        + cities(mem.get_all(states=["retired"]))) == everything


@pytest.mark.parametrize("subset", [
    ("live",), ("ended",), ("retired",), ("live", "ended"), ("live", "retired"),
    ("ended", "retired"), ("live", "ended", "retired"),
])
def test_count_never_disagrees_with_get_all_about_a_state(mem, four, subset):
    """Same argument as the axes: two independent signatures reaching one store call, so
    the way they drift is one of them forwarding a keyword the other drops — which
    produces a plausible number rather than an error."""
    assert mem.count(states=subset) == len(mem.get_all(states=subset)), subset


def test_search_carries_the_state_filter_through_both_retrieval_legs(mem, four):
    """`get_all` only proves the store filter works. Retrieval is where the parameter is
    actually used, and it has to survive the vector leg, the lexical leg and the fusion
    between them."""
    def found(**kw):
        return sorted(r.claim.object for r in mem.search("user lives in", k=10, **kw))

    assert found(states=["retired"]) == ["Berlin", "Paris"]
    assert found(states=["ended"]) == ["Rome"]
    assert found() == found(states=["live"]) == ["Lisbon"]


def test_the_state_filter_is_in_the_sql_not_applied_after_the_page(mem):
    """The reason this could not have been a client-side filter, and the reason it is a
    store parameter rather than a comprehension in the facade.

    `search` is capped: it over-fetches `k * candidate_multiplier` candidates and ranks
    those. Filter afterwards and the retired claim is only found when it happens to land
    inside that window — so a store with enough live claims to fill the page returns an
    empty audit, with nothing in the result to say the answer was truncated rather than
    absent. Twelve live rows against `k=1` is a window of five.
    """
    for i in range(12):
        put(mem, f"City{i}", predicate=f"lived_in_{i}", valid_from=JAN, recorded_at=JAN)
    corrected = put(mem, "Atlantis", predicate="lived_in_99", valid_from=JAN,
                    recorded_at=JAN, invalidated_at=JULY)

    hits = mem.search("user lived in", k=1, states=["retired"])
    assert [r.claim.id for r in hits] == [corrected.id]


# =============================================================================
# An ambiguous call is a bug at the call site
# =============================================================================

@pytest.mark.parametrize("call", [
    lambda m, **kw: m.get_all(**kw),
    lambda m, **kw: m.count(**kw),
    lambda m, **kw: m.search("lives", **kw),
    lambda m, **kw: m.scope(user="alice").get_all(**kw),
    lambda m, **kw: m.store.candidate_ids([SCOPE], **kw),
    lambda m, **kw: list(m.store.iter_claims("acme", **kw)),
])
def test_states_and_include_invalidated_together_raise_rather_than_pick_one(mem, call):
    """There is no reading of `states=["retired"], include_invalidated=False` in which one
    of the two is not being ignored. Silently honouring either answers a question the
    caller did not ask, and the answer looks perfectly ordinary."""
    with pytest.raises(ValueError, match="not both"):
        call(mem, states=["retired"], include_invalidated=False)


@pytest.mark.parametrize("bad", [[], ["live", "deleted"], ["invalidated"], ("LIVE",)])
def test_a_state_the_model_does_not_have_raises_instead_of_matching_nothing(mem, bad):
    """Both failure directions produce an empty page, which is indistinguishable from an
    honest empty answer — so a typo in an audit filter would read as "we never corrected
    anything"."""
    with pytest.raises(ValueError):
        mem.get_all(states=bad)


async def _states_through_the_async_facade(mem, four):
    view = AsyncMemvara(mem)
    return (await view.get_all(states=["retired"]), await view.count(states=["retired"]),
            await view.search("user lives in", k=10, states=["retired"]))


def test_the_async_facade_carries_states_too(mem, four):
    """`AsyncMemvara` promises the same name, the same arguments and the same return
    value. A keyword that stops at the sync facade breaks that promise in the way that
    costs most: the async caller gets a plausible answer to a different question."""
    claims, count, hits = asyncio.run(_states_through_the_async_facade(mem, four))
    assert cities(claims) == ["Berlin", "Paris"]
    assert count == 2
    assert sorted(r.claim.object for r in hits) == ["Berlin", "Paris"]


# =============================================================================
# The record reads: history, why, produced
# =============================================================================

def test_history_defaults_to_the_whole_timeline_not_to_now(mem, four):
    """Everywhere else an unset axis means "now". A timeline whose default was now would
    drop every superseded version, which is the entire content of a timeline — so these
    three methods default to "no filter" instead, and the bare call is unchanged."""
    assert cities(mem.history("user", "lives_in")) == [
        "Berlin", "Lisbon", "Paris", "Rome"]


def test_history_at_a_past_known_at_is_the_timeline_as_it_looked_then(mem, four):
    """The audit query this method was missing. An investigator reading the trail in
    mid-July saw two versions; the same trail today has four, and two of those rewrite
    what July thought had happened. Being able to reproduce the July document is the
    difference between an audit trail and a current-state dump with dates on it."""
    assert cities(mem.history("user", "lives_in", known_at=JULY_MID)) == [
        "Berlin", "Paris"]
    assert cities(mem.history("user", "lives_in", known_at=MAR)) == ["Berlin"]


def test_history_at_a_valid_at_is_every_version_ever_asserted_of_that_moment(mem, four):
    """The other axis, and the slot-level form of question 2: everything we have ever
    thought was true in June, corrections included and in the order we came to believe
    them. Berlin came first and Rome replaced it, and both belong in the answer."""
    assert [c.object for c in mem.history("user", "lives_in", valid_at=JUNE)] == [
        "Berlin", "Rome"]


def test_history_rows_are_returned_as_stored_rather_than_rewritten(mem, four):
    """`known_at` selects which versions existed, and deliberately does not fabricate
    the stamps they carried at the time. `Claim.state` already settled this: a claim
    retired last week reads as retired even in a March view, and the caller pairs the
    row with the instant they asked for. Rewriting the rows would hand back claim ids
    whose `why()` disagrees with them."""
    july = {c.object: c for c in mem.history("user", "lives_in", known_at=JULY_MID)}
    assert july["Berlin"].invalidated_at == AUG, "the row is not edited for the view"
    assert july["Berlin"].is_live(known_at=JULY_MID, valid_at=JUNE), "but it was live"
    assert not july["Berlin"].is_live()


def test_produced_dates_what_a_turn_had_been_turned_into(mem):
    """A turn keeps acquiring claims as later turns restate it, so "what did this
    conversation produce" is a question with a different answer every month. An audit
    that starts from the transcript needs the answer dated."""
    turn = Episode(role="user", content="I moved to Rome", scope=SCOPE, ts=MAR)
    mem.store.add_episode(turn)
    put(mem, "Rome", valid_from=MAR, recorded_at=MAR, sources=[turn.id])
    put(mem, "Rome", predicate="mentioned", valid_from=MAR, recorded_at=AUG,
        sources=[turn.id])

    assert len(mem.produced(turn.id)) == 2
    assert cities(mem.produced(turn.id, known_at=JUNE)) == ["Rome"]
    assert len(mem.produced(turn.id, known_at=JUNE)) == 1


def test_why_dates_the_evidence_without_hiding_the_claim(mem):
    """The axes describe the evidence around a claim, not whether the claim is shown.
    Withholding the row for an out-of-window claim would turn `why()` into the
    existence oracle it is explicitly not allowed to be — the scope check already
    returns `None` for that, and the two must not be confusable."""
    first = Episode(role="user", content="I moved to Rome", scope=SCOPE, ts=MAR)
    later = Episode(role="user", content="Still in Rome", scope=SCOPE, ts=AUG)
    mem.store.add_episode(first)
    mem.store.add_episode(later)
    claim = put(mem, "Rome", valid_from=MAR, recorded_at=MAR,
                sources=[first.id, later.id])

    now = mem.why(claim.id)
    assert now is not None and [e.id for e in now.episodes] == [first.id, later.id]

    june = mem.why(claim.id, known_at=JUNE)
    assert june is not None, "the claim is still returned"
    assert [e.id for e in june.episodes] == [first.id], "only the evidence is dated"


def test_why_dates_the_supersessions_it_reports(mem):
    """`superseded` is read out of the slot's history, so it drifts for the same reason
    the timeline does: what a claim had replaced by July is not what it has replaced
    since."""
    old = put(mem, "Berlin", valid_from=JAN, valid_to=JULY, recorded_at=JAN,
              invalidated_at=JULY)
    newer = put(mem, "Paris", valid_from=JULY, recorded_at=JULY)
    old.invalidated_by = newer.id
    mem.store.put_claim(old)

    assert [c.object for c in mem.why(newer.id).superseded] == ["Berlin"]
    assert mem.why(newer.id, known_at=MAR).superseded == [], "not recorded yet in March"


def test_why_still_reports_a_supersession_that_never_closed_belief(mem):
    """The filter used to read `invalidated_at` and nothing else, so once supersession
    stopped writing one it reported *nothing* superseded at any dated `known_at` — on
    every chain the write path produces, which is all of them.

    A supersession is still a belief-clock event; the instant is simply the one we
    recorded the successor at, because that is when we decided. The claim itself is what
    carries that instant now, and `why()` already holds it.
    """
    mem.remember("user", "lives_in", "Berlin", valid_from=JAN, recorded_at=JAN)
    mem.remember("user", "lives_in", "Lisbon", valid_from=JULY, recorded_at=JULY)
    berlin, lisbon = mem.history("user", "lives_in")
    assert berlin.invalidated_at is None, "the row this dating has to work without"

    assert [c.object for c in mem.why(lisbon.id).superseded] == ["Berlin"]
    assert [c.object for c in mem.why(lisbon.id, known_at=AUG).superseded] == ["Berlin"]
    assert mem.why(lisbon.id, known_at=MAR).superseded == [], \
        "the replacement had not been recorded in March"


def test_retiring_a_superseded_row_does_not_re_date_the_supersession(mem):
    """A row carrying *both* closures is dated by the wrong one if `invalidated_at` wins.

    Berlin ended in August, when Lisbon replaced it. In October somebody deletes Berlin,
    which retires it and leaves the row double-closed. `invalidated_at` now says October
    — but that dates a different event, the end of belief in *Berlin*, not the moment
    Lisbon displaced it. Reading it made `why(Lisbon, known_at=September)` report nothing
    superseded: an October write silently changing what the audit says about September,
    in the method whose entire promise is that the trail survives, by the operation
    documented as the reversible one.

    The shape is only ordinary because it is now reachable — before the closures split,
    every supersession wrote `invalidated_at` too, so the two stamps never disagreed.
    """
    mem.remember("user", "lives_in", "Berlin", valid_from=J23, recorded_at=J23)
    lisbon = mem.remember("user", "lives_in", "Lisbon",
                          valid_from=AUG23, recorded_at=AUG23).added[0]
    berlin = mem.history("user", "lives_in")[0]
    assert berlin.state == "ended" and berlin.invalidated_by == lisbon.id
    assert [c.object for c in mem.why(lisbon.id, known_at=SEP23).superseded] == ["Berlin"]

    assert mem.delete(berlin.id, at=OCT23) is True
    reread = mem.get(berlin.id)
    assert reread.state == "retired" and reread.valid_to is not None, \
        "both closures on one row — the shape this is about"

    assert [c.object for c in mem.why(lisbon.id, known_at=SEP23).superseded] == ["Berlin"]
    assert mem.why(lisbon.id, known_at=J23).superseded == [], \
        "and the dating still works: Lisbon did not exist in January"


# =============================================================================
# since(): what changed while the agent was away
# =============================================================================
#
# The read a resuming agent has never had. `get_all()` shows the current view and cannot
# say which of it is new; nothing at all showed what *left*, because by the time you look
# there is no row left to notice the absence of. It is the query a store with one clock
# cannot answer in either direction, and it needs no new store method — `candidate_ids`
# already takes both clocks, so the delta is a set difference over two calls.


def test_since_reports_what_arrived_and_what_left(mem, four):
    """June believed Berlin; today believes Lisbon. Both halves, one call."""
    delta = mem.since(JUNE)
    assert cities(delta.added) == ["Lisbon"]
    assert cities(delta.gone) == ["Berlin"]
    assert delta.since == JUNE, "the answer carries the question it answers"


def test_a_supersession_lands_in_both_halves(mem):
    """The shape §8a names, written through the facade rather than into the store: the
    retired id in `gone`, the replacement in `added`. Two entries for one event, because
    a correction stated without the thing it corrected is not a correction."""
    berlin = mem.remember("user", "lives_in", "Berlin", valid_from=JAN,
                          recorded_at=JAN).added[0]
    away = utcnow()
    lisbon = mem.supersede(berlin.id, Claim(subject="user", predicate="lives_in",
                                            object="Lisbon", scope=SCOPE),
                           close="retired").added[0]

    delta = mem.since(away)
    assert [c.id for c in delta.added] == [lisbon.id]
    assert [c.id for c in delta.gone] == [berlin.id]
    assert [c.state for c in delta.gone] == ["retired"], \
        "and the state is what says the record was wrong rather than merely over"


def test_a_fact_the_world_moved_past_leaves_too(mem):
    """`gone` is not "retired": a claim whose valid interval closed is equally absent
    from the current view, and an agent told it while it held wants to know."""
    put(mem, "Berlin", valid_from=JAN, valid_to=JULY, recorded_at=JAN)
    delta = mem.since(JUNE)
    assert cities(delta.gone) == ["Berlin"] and delta.added == ()
    assert [c.state for c in delta.gone] == ["ended"]


def test_since_pins_both_clocks_because_the_belief_clock_alone_is_a_different_question(
        mem, four):
    """The trap this method exists on the far side of, asserted rather than described.

    `known_at=T` on its own means "what we believed at T, about the world *as it is
    now*", so a value that has since been closed out in world time fails the
    present-tense interval test and never enters the "then" set at all — the supersession
    that is the whole point of the call would report an addition with nothing beside it.
    """
    scopes = SCOPE.ancestors()
    belief_only = set(mem.store.candidate_ids(scopes, known_at=JUNE))
    both = set(mem.store.candidate_ids(scopes, valid_at=JUNE, known_at=JUNE))
    now = set(mem.store.candidate_ids(scopes))

    assert belief_only == set(), "Berlin's world interval has closed since"
    assert both - now == {four["Berlin"].id}
    assert {c.id for c in mem.since(JUNE).gone} == both - now


def test_since_costs_two_scope_wide_scans_and_not_one_read_per_claim(mem, four):
    """The cost is stated in the docstring, so something has to hold it there. Two id
    scans is what makes this acceptable for a session-start read; a per-claim probe would
    make it a read nobody runs."""
    calls = []
    real = mem.store.candidate_ids
    mem.store.candidate_ids = lambda *a, **kw: (calls.append(kw), real(*a, **kw))[1]

    mem.since(JUNE)
    assert len(calls) == 2
    assert calls[0] == {"valid_at": JUNE, "known_at": JUNE} and calls[1] == {}


def test_a_delta_with_nothing_in_it_is_still_a_delta(mem, four):
    """Nothing changed is an answer, and an agent resuming needs to be able to tell it
    from "the store is empty" — so it is two empty tuples and the instant, not None."""
    delta = mem.since(AUG + timedelta(days=1))
    assert delta.added == () and delta.gone == ()
    assert delta.since == AUG + timedelta(days=1)
    assert repr(delta).startswith("<Delta since 2026-08-02T00:00:00+00:00 +0 -0")


def test_a_naive_instant_is_read_as_utc_rather_than_rejected(mem, four):
    """Callers build these by hand at API edges. Every other read here follows the same
    convention, and a TypeError from inside a set difference is a worse outcome."""
    assert mem.since(JUNE.replace(tzinfo=None)).since == JUNE
    assert cities(mem.since(JUNE.replace(tzinfo=None)).gone) == ["Berlin"]


def test_since_is_ordered_rather_than_left_in_set_order(mem):
    """A set difference discards even the backend's scan order, so without a sort the two
    halves would come back differently between runs of one process."""
    for i in range(6):
        put(mem, f"City{i}", predicate=f"lived_in_{i}", valid_from=JAN, recorded_at=JAN)
    order = [c.id for c in mem.since(JAN - timedelta(days=1)).added]
    assert order == [c.id for c in mem.get_all()][:len(order)]
    assert order == [c.id for c in mem.since(JAN - timedelta(days=1)).added]


def test_since_stays_inside_the_scope_it_was_asked_about(mem):
    """The delta is a scoped read like every other, and a session that could see a
    sibling session's arrivals would be the scope leak in its most useful form."""
    mem.scope(user="alice", session="s1").remember("user", "likes", "coffee")
    then = JAN
    assert cities(mem.since(then, user="alice", session="s1").added) == ["coffee"]
    assert mem.since(then, user="bob").added == ()


def test_the_scoped_view_carries_since(mem):
    """`ScopedMemvara` is what the MCP server holds, and `memory_since` is the tool this
    method exists for."""
    view = mem.scope(user="alice", session="s2")
    view.remember("user", "working_on", "auth refactor")
    assert cities(view.since(JAN).added) == ["auth refactor"]
    assert mem.scope(user="alice", session="s3").since(JAN).added == ()


# =============================================================================
# The facades mirror each other
# =============================================================================

def test_the_scoped_view_forwards_both_axes(mem, four):
    """`ScopedMemvara` is what the MCP server and every integration adapter holds. A
    keyword that reached `Memvara` and stopped here would be missing from the object
    most callers actually have, and would fail as an unexpected-keyword TypeError at
    their call site rather than at ours."""
    view = mem.scope(user="alice")
    assert cities(view.get_all(valid_at=JUNE)) == ["Rome"]
    assert view.count(known_at=JULY_MID) == 1
    assert cities(view.history("user", "lives_in", known_at=MAR)) == ["Berlin"]
    assert view.search("user lives in", k=5, known_at=JULY_MID)[0].claim.object == "Paris"


def test_the_async_facade_forwards_both_axes(mem, four):
    """`AsyncMemvara` promises "same name, same arguments, same return value". A new
    keyword that reached only two of the three facades would break that quietly, on the
    facade a server layer is most likely to be holding."""
    async def go():
        async with AsyncMemvara(mem) as amem:
            return (
                cities(await amem.get_all(valid_at=JUNE)),
                await amem.count(known_at=JULY_MID),
                cities(await amem.history("user", "lives_in", known_at=MAR)),
                [r.claim.object
                 for r in await amem.search("user lives in", k=5, known_at=JULY_MID)],
            )

    assert asyncio.run(go()) == (["Rome"], 1, ["Berlin"], ["Paris"])


def test_the_async_facade_also_forwards_why_produced_and_the_traversal(mem):
    """The four wrappers the previous test does not cover. Each is a separate hand-written
    forwarding signature, so "the search one works" says nothing about the others."""
    turn = Episode(role="user", content="I moved to Rome", scope=SCOPE, ts=MAR)
    mem.store.add_episode(turn)
    claim = put(mem, "Rome", valid_from=MAR, recorded_at=MAR, sources=[turn.id])
    put(mem, "Dana", subject="Alice", predicate="reports_to",
        valid_from=JULY, recorded_at=JAN)

    async def go():
        async with AsyncMemvara(mem) as amem:
            return (
                len((await amem.why(claim.id, known_at=JUNE)).episodes),
                len(await amem.produced(turn.id, known_at=JUNE)),
                len(await amem.neighborhood("Alice", valid_at=JULY_MID, known_at=JUNE)),
                len(await amem.paths_between("Alice", "Dana", valid_at=JULY_MID,
                                             known_at=JUNE)),
            )

    assert asyncio.run(go()) == (1, 1, 1, 1)


# =============================================================================
# The round trip through the write path
# =============================================================================

def test_question_two_works_on_a_history_the_library_wrote_itself(mem):
    """**The defect the whole fixture above was a workaround for.**

    Two ordinary `remember()` calls, no `put_claim`, no hand-set stamps: Berlin from
    2023, Lisbon from 2026. Asked what we *now* believe was true in 2024, the store has
    to say Berlin — and it said nothing at all, because `Reconciler._retire` closed both
    clocks and so recorded "we no longer believe Berlin" about a claim that was never
    wrong. `as_of` kept answering the whole time, which is exactly why this survived: it
    rewinds the belief clock past the supersession, so it never reads the stamp that was
    wrong.

    The consequence was that question #2 — the reading the two axes exist for — returned
    nothing on *any* history this library produced. It worked only on stores built by
    calling `store.put_claim` directly, which is what the fixture in this file does.
    """
    mem.remember("user", "lives_in", "Berlin", valid_from=J23, recorded_at=J23)
    mem.remember("user", "lives_in", "Lisbon", valid_from=J26, recorded_at=J26)

    assert cities(mem.get_all(as_of=MID)) == ["Berlin"], "what we believed then"
    assert cities(mem.get_all(valid_at=MID)) == ["Berlin"], "what we believe now, about then"
    assert cities(mem.get_all()) == ["Lisbon"], "and the present is unchanged"


def test_the_superseded_row_says_the_world_moved_not_that_we_were_wrong(mem):
    """The row behind the query above, stated as the rule rather than as a symptom.

    `valid_to` was always right — Berlin stopped being true when Lisbon began.
    `invalidated_at` was the lie: it means *we no longer believe this record*, and Alice
    really did live in Berlin in 2024.
    """
    mem.remember("user", "lives_in", "Berlin", valid_from=J23, recorded_at=J23)
    mem.remember("user", "lives_in", "Lisbon", valid_from=J26, recorded_at=J26)
    berlin, lisbon = mem.history("user", "lives_in")

    assert berlin.valid_to == J26 == lisbon.valid_from, "the intervals abut"
    assert berlin.invalidated_at is None
    assert berlin.invalidated_by == lisbon.id, "and it still names its successor"
    assert (berlin.state, lisbon.state) == ("ended", "live")


def test_the_four_row_fixture_is_now_expressible_through_the_write_path(mem):
    """The fixture at the top of this file bypasses `remember()`, and its docstring says
    why: the write path "would reconcile several of them out of existence on the way in".
    That was this bug. The same four rows, written through the facade, land on the same
    four pairs of intervals — so the workaround is a workaround and not a requirement.

    Note which calls express which event, because that is the whole distinction:
    the July move is a *change* and supersedes on its own, while the two August
    corrections are `close="retired"`, since they say the earlier rows were never true.
    """
    mem.remember("user", "lives_in", "Berlin", valid_from=JAN, recorded_at=JAN)
    mem.remember("user", "lives_in", "Paris", valid_from=JULY, recorded_at=JULY)
    berlin, paris = mem.history("user", "lives_in")

    mem.supersede(berlin.id, Claim(subject="user", predicate="lives_in", object="Rome",
                                   valid_from=MAR, valid_to=JULY, recorded_at=AUG),
                  at=AUG, close="retired")
    mem.supersede(paris.id, Claim(subject="user", predicate="lives_in", object="Lisbon",
                                  valid_from=JULY, recorded_at=AUG),
                  at=AUG, close="retired")

    rows = {c.object: c for c in mem.history("user", "lives_in")}
    assert {k: (v.valid_from, v.valid_to, v.recorded_at, v.invalidated_at)
            for k, v in rows.items()} == {
        "Berlin": (JAN, JULY, JAN, AUG),
        "Paris": (JULY, None, JULY, AUG),
        "Rome": (MAR, JULY, AUG, None),
        "Lisbon": (JULY, None, AUG, None),
    }
    # And therefore the four questions give the same four answers as the fixture does.
    assert [cities(mem.get_all()), cities(mem.get_all(valid_at=JUNE)),
            cities(mem.get_all(known_at=JULY_MID)), cities(mem.get_all(as_of=JUNE))] == [
        ["Lisbon"], ["Rome"], ["Paris"], ["Berlin"]]


def test_remember_backdates_both_axes_and_the_reads_agree(mem):
    """The end-to-end version: a historical import states a 2026-March valid time and a
    today transaction time, and every read has to place it on the right clock. Written
    through `remember()` rather than into the store, because the fixture above bypasses
    the write path and something has to prove the two agree."""
    mem.remember("user", "lived_in", "Rome", valid_from=MAR, valid_to=JULY)

    assert cities(mem.get_all(valid_at=JUNE)) == ["Rome"], "in force in June"
    assert mem.get_all(as_of=JUNE) == [], "but we had not heard it in June"
    assert mem.get_all() == [], "and it is over now"
    assert cities(mem.history("user", "lived_in")) == ["Rome"]


# --- ask(): the three readings, and the one that had no surface ---------------
#
# `recall()` renders the current answer. The question the whole two-clock model exists
# for — "and is that what you would have told me then?" — had no method at all, so an
# agent asked it had to run two reads and do the comparison itself, which is exactly the
# arithmetic `_stated_at` exists to stop anyone doing by hand.


@pytest.fixture()
def corrected(mem):
    """Rome until March, Berlin from March — and Berlin not recorded until 22 March.

    The brief's own scenario, and the one shape where the two clocks give three different
    answers to one question about one slot.
    """
    mem.remember("user", "lives_in", "Rome", valid_from=JAN, recorded_at=JAN)
    mem.remember("user", "lives_in", "Berlin", valid_from=MAR,
                 recorded_at=datetime(2026, 3, 22, tzinfo=TZ))
    return mem


def test_ask_separates_what_was_true_from_what_we_would_have_said(corrected):
    """The finding, and the sentence no single-clock store can write.

    On 15 March this store held Rome and nothing else — Berlin arrived a week later. So
    "what was true then", judged by everything known now, is Berlin; and "what would you
    have told me then" is Rome. Both are correct answers to different questions, and an
    audit is the difference between them."""
    reading = corrected.ask("where do they live?", at=MID_MAR).readings[0]

    assert cities(reading.then) == ["Berlin"], "what we believe today about 15 March"
    assert cities(reading.stated) == ["Rome"], "what this store would have said that day"
    assert cities(reading.now) == ["Berlin"]
    assert reading.diverged


def test_stated_disagrees_with_get_all_as_of_and_that_is_the_point(corrected):
    """The one thing to understand before quoting either, pinned so it cannot drift.

    `get_all(as_of=T)` reads four columns off a row, and a row's `valid_to` is written
    *in place* by the write that displaces it — so on 15 March it applies an ending that
    would not be recorded until the 22nd, and Rome falls out with nothing to replace it.
    `ask()` has the supersession chain in front of it and dates the ending at the
    successor's `recorded_at`, which is when the pointer was actually written. Same rule
    `why()` has used since a July view started reporting an August replacement."""
    reading = corrected.ask("where do they live?", at=MID_MAR).readings[0]

    assert corrected.get_all(as_of=MID_MAR) == [], "the row-at-a-time answer"
    assert cities(reading.stated) == ["Rome"], "the reconstructed one"


def test_the_narrative_names_both_readings_and_the_day_the_record_moved(corrected):
    """`Answer.text` is the product, and it is rendered from stored columns only.

    Asserted on the sentences rather than on the whole block, because the block is prose
    and pinning it byte for byte would make every wording improvement a test edit — but
    the three facts in it are not prose, and dropping any one of them turns the paragraph
    back into something `recall()` could have said."""
    text = corrected.ask("where do they live?", at=MID_MAR).text

    assert "user lives_in: Berlin." in text
    assert "On 2026-03-15 this store would have said Rome" in text
    assert "recorded 2026-03-22, 7 days after the instant you asked about" in text


def test_the_divergence_is_dated_by_a_retirement_and_not_by_the_next_write(mem):
    """Both clocks stamp a change and reading only one dates the wrong event.

    Rome recorded in January, retired on 1 March, Oslo written on 1 April. Asked about
    February, `then` and `stated` disagree *because of the retirement* — and dating that
    from `recorded_at` alone reported 1 April, a month late, attributed to a write that
    had nothing to do with it. Saying when the record moved is what this method is for,
    so that sentence being wrong is the method failing at its one job.
    """
    mem.remember("user", "lives_in", "Rome", valid_from=JAN, recorded_at=JAN)
    mem.forget("user", "lives_in", at=MAR)
    mem.remember("user", "lives_in", "Oslo",
                 valid_from=datetime(2026, 4, 1, tzinfo=TZ),
                 recorded_at=datetime(2026, 4, 1, tzinfo=TZ))

    text = mem.ask("where do they live?", at=datetime(2026, 2, 1, tzinfo=TZ)).text

    assert "would have said Rome" in text
    assert "recorded 2026-03-01" in text, "the retirement is what moved the record"
    assert "2026-04-01" not in text, "Oslo's arrival did not cause this divergence"


def test_ask_about_now_reports_the_lag_instead_of_a_divergence(corrected):
    """With no past instant to ask about there is no correction to report, and the two
    clocks can still be apart. Berlin was true from 1 March and invisible here until the
    22nd, and that gap is the other half of what this store knows."""
    text = corrected.ask("where do they live?").text

    assert "would have said" not in text, "nothing has diverged from the present"
    assert "True since 2026-03-01, recorded 2026-03-22 — 21 days later." in text


def test_ask_before_anything_was_true_says_so_and_still_gives_the_present(mem):
    """The empty reading is an answer. Without the second line it reads as a store that
    lost the fact rather than one that had not been told yet."""
    mem.remember("user", "lives_in", "Rome", valid_from=JUNE, recorded_at=JUNE)

    text = mem.ask("where do they live?", at=JAN).text

    assert "nothing was true on 2026-01-01" in text
    assert "Now: Rome." in text


def test_ask_reports_a_slot_whose_every_value_was_retired(mem):
    """Nothing then, nothing now, and no divergence — three lines of "nothing" with no
    reason beside them. The store did not lose it; somebody said it was never true."""
    mem.remember("user", "lives_in", "Rome", valid_from=JAN, recorded_at=JAN)
    mem.forget("user", "lives_in")

    text = mem.ask("where do they live?", at=MAR).text

    assert "Every value ever recorded for it has been retired (1 in all)" in text


def test_ask_answers_nothing_when_the_store_is_empty(mem):
    """Distinct from answering the wrong slot, which is what happens when the store holds
    something and `min_score` is left at its floorless default — see `ask()`'s docstring,
    which says outright that this ranks and does not judge relevance."""
    answer = mem.ask("where do they live?")

    assert answer.readings == ()
    assert answer.text == "Nothing on record for: where do they live?"


def test_ask_dates_an_ending_whose_successor_lives_in_another_slot(mem):
    """Cross-predicate supersession puts the successor outside the timeline being read,
    so a slot is not the closed world it looks like. Resolved from the store rather than
    from the timeline, or the ending would fall back to this row's own `recorded_at` and
    the claim would stop answering months before anyone knew it had."""
    mem.registry.register(PredicateSpec(name="unemployed", cardinality=Cardinality.ONE,
                                        volatility=Volatility.SLOW,
                                        supersedes=("works_at",)))
    mem.remember("user", "works_at", "Acme", valid_from=JAN, recorded_at=JAN)
    mem.remember("user", "unemployed", "yes", valid_from=JULY, recorded_at=AUG)

    reading = mem.ask("where do they work?", at=JUNE, k=1).readings[0]

    assert (reading.subject, reading.predicate) == ("user", "works_at")
    assert [c.object for c in reading.stated] == ["Acme"], (
        "in June this store had not heard of the July change; it arrived in August")


def test_ask_does_not_quote_a_value_we_had_already_stopped_believing(mem):
    """`invalidated_at` needs no reconstruction and gets none: it is a belief-clock stamp
    already, so "had we retired it by then" is a comparison and not an inference. Pinned
    separately from the ending branch because the two closures date different events, and
    reading the wrong one is the mistake this library says cannot be found afterwards."""
    mem.remember("user", "lives_in", "Rome", valid_from=JAN, recorded_at=JAN)
    mem.forget("user", "lives_in", at=MAR)

    assert cities(mem.ask("where do they live?", at=JAN).readings[0].stated) == ["Rome"]
    assert mem.ask("where do they live?", at=JUNE).readings[0].stated == (), \
        "by June we had stopped believing it, so June would not have quoted it"


def test_ask_carries_the_instant_it_answered_about(mem):
    """The same reason `Delta` carries `since`: a caller logging the result must not be
    able to separate the answer from the question it answers."""
    mem.remember("user", "lives_in", "Rome", valid_from=JAN, recorded_at=JAN)

    assert mem.ask("where do they live?", at=MAR).at == MAR
    assert mem.ask("where do they live?").at > MAR, "defaults to now"


def test_the_scoped_and_async_facades_forward_ask(mem, corrected):
    """Every other read is on all four facades; a killer feature reachable from one of
    them is reachable from the one an integration is least likely to hold."""
    scoped = corrected.scope(user="alice")
    assert cities(scoped.ask("where do they live?", at=MID_MAR).readings[0].stated) == \
        ["Rome"]

    async def go():
        amem = AsyncMemvara(corrected)
        answer = await amem.ask("where do they live?", at=MID_MAR, user="alice")
        scoped_answer = await amem.scope(user="alice").ask("where do they live?",
                                                           at=MID_MAR)
        return answer, scoped_answer

    answer, scoped_answer = asyncio.run(go())
    assert cities(answer.readings[0].stated) == ["Rome"]
    assert cities(scoped_answer.readings[0].stated) == ["Rome"]


# --- an interval of no length is not a shorter fact --------------------------
#
# `memory_remember` has refused these since it existed and the library accepted them, so
# the two surfaces disagreed about the same sentence. Both ends arrive in one call here,
# which is what makes refusing right: "true from Tuesday until Tuesday" is not a
# partially-recoverable request, it is self-contradictory, and the caller can restate it.

@pytest.mark.parametrize("valid_to, why", [
    (MAR, "ends where it begins"),
    (JAN, "ends before it begins"),
])
def test_remember_refuses_an_interval_true_at_no_instant(mem, valid_to, why):
    """Stored, this is a claim `added 1` reports and no query on either clock returns.
    That is the shape `true_since` exists to prevent, in a second costume — and the MCP
    adapter's `_interval` says exactly that about the same input."""
    with pytest.raises(ValueError, match="not after the instant the fact began"):
        mem.remember("user", "lived_in", "Rome", valid_from=MAR, valid_to=valid_to)

    assert mem.history("user", "lived_in") == [], why


def test_the_refusal_says_which_instant_it_measured_against_when_you_named_none(mem):
    """`valid_to` alone is measured against now, and a message quoting an instant the
    caller never sent is unreadable without that sentence."""
    with pytest.raises(ValueError, match="because valid_from was omitted"):
        mem.remember("user", "lived_in", "Rome", valid_to=JAN)


def test_an_interval_with_length_is_still_accepted(mem):
    """The other half. Backfilling a finished fact is the documented use of both
    arguments together, and refusing it would break the import case outright."""
    mem.remember("user", "lived_in", "Rome", valid_from=MAR, valid_to=JULY)

    assert cities(mem.get_all(valid_at=JUNE)) == ["Rome"]


def test_supersede_reports_a_predecessor_it_left_true_at_no_instant(mem):
    """`close_out` clamps a closure to the claim's own start rather than inverting the
    interval, so superseding a claim at the instant it began empties it. The clamp is
    right — an inverted interval is a row no `as_of` window can return consistently — and
    it cannot make the row answer anything, so the write says so instead.

    Through `supersede()` rather than the reconciler because this is the method whose
    entire purpose is closing a claim out, and it reaches `close_out` by its own door."""
    written = mem.remember("user", "lives_in", "Berlin", valid_from=JAN, recorded_at=JAN)
    berlin = written.added[0]
    lisbon = Claim(subject="user", predicate="lives_in", object="Lisbon", scope=SCOPE,
                   valid_from=JAN, recorded_at=JAN)

    receipt = mem.supersede(berlin.id, lisbon, at=JAN)

    (collapse,) = receipt.collapsed
    assert (collapse.claim_id, collapse.object, collapse.at) == (berlin.id, "Berlin", JAN)
    (stored,) = [c for c in mem.history("user", "lives_in") if c.id == berlin.id]
    assert stored.valid_from == stored.valid_to == JAN, "the interval is empty"
    assert cities(mem.get_all(valid_at=JAN)) == ["Lisbon"], "Berlin answers nothing"


def test_an_ordinary_supersede_reports_no_collapse(mem):
    """Or the report would fire on every supersession there is."""
    berlin = mem.remember("user", "lives_in", "Berlin",
                          valid_from=JAN, recorded_at=JAN).added[0]
    lisbon = Claim(subject="user", predicate="lives_in", object="Lisbon", scope=SCOPE,
                   valid_from=JULY, recorded_at=JULY)

    assert mem.supersede(berlin.id, lisbon, at=JULY).collapsed == []
    assert cities(mem.get_all(valid_at=JUNE)) == ["Berlin"]


def test_is_live_mirrors_the_store_clause_on_both_axes(mem, four):
    """The Python predicate and the SQL one must agree row for row, because `history()`
    hands back stored rows and tells the caller to judge them with `is_live`. Two
    definitions of liveness that disagree is the bug that makes an audit unreadable."""
    everything = mem.history("user", "lives_in")
    for kw in ({}, {"valid_at": JUNE}, {"known_at": JULY_MID},
               {"valid_at": JUNE, "known_at": AUG}, {"as_of": JUNE}):
        by_hand = {c.id for c in everything if c.is_live(**kw)}
        by_store = {c.id for c in mem.get_all(**kw)}
        assert by_hand == by_store, kw


def test_a_bare_is_live_reads_one_clock_for_both_axes():
    """Two clock reads would put the axes microseconds apart on the commonest call
    there is. Nothing could observe the difference reliably, which is exactly what makes
    it a bad thing to leave in."""
    now = utcnow()
    edge = Claim(subject="user", predicate="lives_in", object="Rome",
                 valid_from=now, recorded_at=now, valid_to=now + timedelta(days=1))
    assert edge.is_live()
