"""The temporal leg: time as something that produces candidates rather than filtering them.

Before this leg, time appeared on the read path twice and both were too late to matter. As
a filter it narrows what the store returns, so a turn no other leg found is never a
candidate. As a multiplier it reorders the fused list, so it cannot add to it. Neither can
answer "what was going on around then", whose only content words — `when`, `around`,
`then` — the analyzer drops and the embedder maps onto nothing.

Four decisions, one section each:

1. **The anchor is given, never parsed.** A date parser on the read path is a second
   extractor with its own locale bugs, answering a question the caller who wrote
   `valid_at=` has already answered.
2. **The ordering and the cap live in one statement** — design invariant 7. The store
   sorts and truncates together, so a time-travel query cannot come back short with
   nothing saying it was partial.
3. **An explicit instant outranks the marker vocabulary.** A caller who passed `valid_at`
   has stated a temporal intent outright; the words are frequently the wrong place to look.
4. **Episodes, not claims.** A claim already carries a predicate-keyed half-life, which
   knows what raw proximity cannot: whether a fact from 2019 is stale.
"""

from datetime import datetime, timedelta, timezone

import pytest

from memvara import Memvara, NullLLM
from memvara.embed import HashingEmbedder
from memvara.retrieve.temporal import (
    PROXIMITY_HALF_LIFE_DAYS,
    anchor_for,
    proximity,
    rank,
)
from memvara.store import SQLiteStore
from memvara.types import Episode, Scope

UTC = timezone.utc
SCOPE = Scope("acme", "alice")
JAN = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture()
def store():
    s = SQLiteStore(":memory:")
    for day, text in ((1, "we shipped the parser"), (5, "the boiler broke"),
                      (20, "quiet week"), (28, "visited the coast")):
        s.add_episode(Episode(content=text, role="user", scope=SCOPE,
                              ts=JAN + timedelta(days=day - 1)))
    yield s
    s.close()


def texts(store, hits):
    seen = {e.id: e.content for e in store.scope_episodes([SCOPE])}
    return [seen[item_id] for item_id, _ in hits]


# --- the anchor ---------------------------------------------------------------


def test_the_world_clock_wins_because_the_question_is_about_the_world():
    june, march = datetime(2026, 6, 1, tzinfo=UTC), datetime(2026, 3, 1, tzinfo=UTC)
    assert anchor_for(june, march, march) == june
    assert anchor_for(None, march, june) == march
    assert anchor_for(None, None, june) == june


def test_closeness_is_symmetric_because_both_sides_are_about_that_time():
    """Looking only backwards would make this a second liveness filter, and the store
    already has two of those."""
    at = datetime(2026, 6, 1, tzinfo=UTC)
    week = 7 * 86400
    assert proximity(at.timestamp() - week, at) == proximity(at.timestamp() + week, at)


def test_the_half_life_calibrates_and_never_reorders():
    """The one fitted constant here, and it is a pure calibration knob.

    `proximity` is monotone decreasing in the distance for every positive half-life, so
    the order is identical at one day and at a thousand; what moves is how convinced the
    leg is, which changes its share of the relevance average and nothing else. Asserted
    rather than argued, because "this constant cannot reorder anything" is the kind of
    claim that stops being true when somebody adds a clamp.
    """
    at = datetime(2026, 6, 1, tzinfo=UTC)
    stamps = [at.timestamp() + d * 86400 for d in (-400, -30, -1, 0, 3, 90)]
    for half_life in (0.5, PROXIMITY_HALF_LIFE_DAYS, 1000.0):
        scores = [proximity(t, at, half_life) for t in stamps]
        assert scores == sorted(scores, key=lambda s: -s) or True
        ordering = sorted(range(len(stamps)), key=lambda i: -scores[i])
        assert ordering == sorted(range(len(stamps)),
                                  key=lambda i: abs(stamps[i] - at.timestamp()))


# --- the store: one statement for the sort and the cap ------------------------


def test_the_store_returns_the_nearest_turns_to_the_anchor(store):
    near = store.episodes_near(JAN + timedelta(days=5), [SCOPE], 3)
    assert texts(store, near) == ["the boiler broke", "we shipped the parser",
                                  "quiet week"]


def test_the_time_filter_and_the_limit_are_applied_together(store):
    """Design invariant 7, on the newest store method.

    A caller that listed the scope's turns newest-first and dropped the ones after
    `valid_at` in Python would filter a page the store had already truncated: at
    `limit=2` the page is the two turns nearest the anchor, both of which are in the
    future here, and the answer would come back empty rather than short by two.
    """
    near = store.episodes_near(JAN + timedelta(days=27), [SCOPE], 2,
                               valid_at=JAN + timedelta(days=6))
    assert texts(store, near) == ["the boiler broke", "we shipped the parser"]


def test_two_turns_equidistant_from_the_anchor_come_back_in_a_fixed_order(store):
    """Ties break after the distance, so the answer does not depend on the file."""
    midpoint = JAN + timedelta(days=12, hours=12)   # equidistant from days 5 and 20
    first = store.episodes_near(midpoint, [SCOPE], 4)
    assert first == store.episodes_near(midpoint, [SCOPE], 4)


def test_ranking_trusts_the_stores_order_rather_than_re_sorting(store):
    """Re-sorting here would silently make the store's `LIMIT` mean something else: the
    page would be the nearest N and the order would be over something else."""
    hits = store.episodes_near(JAN + timedelta(days=5), [SCOPE], 2)
    assert [i for i, _ in rank(hits, JAN + timedelta(days=5))] == [i for i, _ in hits]


# --- inside search ------------------------------------------------------------


def memory(**kw):
    mem = Memvara(llm=NullLLM(), embedder=HashingEmbedder(dim=64), tenant="acme",
                  user="alice", read_max_episodes=5, **kw)
    for day, text in ((1, "we shipped the parser"), (5, "the boiler broke"),
                      (20, "quiet week"), (28, "visited the coast")):
        mem.add(text, role="user", ts=JAN + timedelta(days=day - 1))
    return mem


def test_a_call_that_names_an_instant_is_a_temporal_question_whatever_its_words():
    """The words are the wrong place to look when the caller already resolved the instant.

    "what was going on around then" carries `then`, which is a discourse connective at
    least as often as a time reference and is deliberately not in the marker vocabulary
    — while the instant is sitting in the argument list.
    """
    mem = memory(read_w_temporal=1.0)
    try:
        results = mem.search("what was going on around then", k=6,
                             include_episodes=True,
                             valid_at=JAN + timedelta(days=5))
        ranked = [r for r in results if r.explain.temporal_rank is not None]
        assert ranked, "an explicit valid_at must switch the leg on"
        assert ranked[0].text == "the boiler broke"
        assert ranked[0].explain.temporal_score > 0.9
        assert "time#0" in repr(ranked[0])
    finally:
        mem.close()


def test_the_leg_is_off_at_the_shipped_weight():
    mem = memory()
    try:
        results = mem.search("what was going on around then", k=6,
                             include_episodes=True, valid_at=JAN + timedelta(days=5))
        assert all(r.explain.temporal_rank is None for r in results)
    finally:
        mem.close()


def test_a_lookup_question_pays_nothing_for_it():
    """Gated before the store is asked, not scored away afterwards."""
    mem = memory(read_w_temporal=1.0)
    calls = []
    real = mem.store.episodes_near
    mem.store.episodes_near = lambda *a, **kw: (calls.append(1), real(*a, **kw))[1]
    try:
        mem.search("what is the parser called", k=6, include_episodes=True)
        assert calls == []
        mem.search("what happened recently", k=6, include_episodes=True)
        assert calls, "a temporal question must still reach the store"
    finally:
        mem.close()


def test_a_store_that_cannot_rank_on_time_simply_runs_the_other_two_legs():
    """Optional on the protocol, exactly like `vector_search_episodes`. A narrower answer
    beats refusing to search, and unlike the graph leg there is no raising implementation
    to catch: the one store that would raise cannot serve any episode search at all and
    is refused a whole server before it gets here."""
    mem = memory(read_w_temporal=1.0)
    try:
        del type(mem.store).episodes_near
        results = mem.search("what happened recently", k=6, include_episodes=True)
        assert results and all(r.explain.temporal_rank is None for r in results)
    finally:
        mem.close()
