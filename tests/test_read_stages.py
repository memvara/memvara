"""Query rewrite and synthesis: the read path's two model stages beside `ranked`.

The first half tests `QueryRewriter` and `Synthesizer` on their own: the prompt each one
sends, how each reply is read, and how every failure is sorted into an outcome. The second
half tests what they do to a read: `HybridRetriever.search` through `Memvara.search`, and
`Memvara.recall(synthesize=True)`.

Every model here is a fake `Chat` that records its calls. Nothing sleeps: a slow reply is
a fake clock that the fake backend moves forward while it "answers". Nothing reaches a
network and no key is needed.
"""

from __future__ import annotations

import warnings
from datetime import date, datetime, timezone
from typing import Any

import pytest

from memvara import DegradedExtractionWarning, HashingEmbedder, Memvara, NullLLM
from memvara.llm import Chat
from memvara.llm.base import Usage
from memvara.retrieve import EpisodeResult
from memvara.select import QueryRewriter, Rewrite, Selected, Synthesis, Synthesizer
from memvara.select.stages import (
    DEFAULT_TIMEOUT,
    REWRITE_MAX_COMPLETION_TOKENS,
    REWRITE_SYSTEM,
    SYNTHESIS_MAX_COMPLETION_TOKENS,
    SYNTHESIS_SYSTEM,
    parse_rewrite,
)
from memvara.telemetry import RETRIEVAL_QUERY, MemoryRecorder
from memvara.types import RecallResult, SearchResults

UTC = timezone.utc
TODAY = date(2026, 9, 23)


class Clock:
    """A monotonic clock a test moves by hand."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class FakeChat:
    """A `Chat` whose reply, failure and duration are set by the test.

    `advance` moves `clock` forward during the call, which is how a reply arrives late
    without the test sleeping.
    """

    def __init__(self, reply: str = "", *, raises: Exception | None = None,
                 clock: Clock | None = None, advance: float = 0.0) -> None:
        self.reply = reply
        self.raises = raises
        self.clock = clock
        self.advance = advance
        self.calls: list[dict[str, Any]] = []

    def chat(self, system: str, prompt: str, *, json_object: bool,
             max_completion_tokens: int, timeout: float,
             usage: Usage | None = None) -> str:
        self.calls.append({"system": system, "prompt": prompt, "json_object": json_object,
                           "max_completion_tokens": max_completion_tokens,
                           "timeout": timeout, "usage": usage})
        if self.clock is not None:
            self.clock.now += self.advance
        if self.raises is not None:
            raise self.raises
        return self.reply


def status(code: int) -> Exception:
    exc = RuntimeError(f"provider answered {code}")
    exc.status_code = code  # type: ignore[attr-defined]
    return exc


def rewrite_reply(*queries: str, start: str | None = None, end: str | None = None) -> str:
    import json
    span = None if start is None else {"from": start, "to": end}
    return json.dumps({"queries": list(queries), "date_range": span})


# --- QueryRewriter on its own --------------------------------------------------------


def test_the_rewriter_sends_the_query_and_todays_date_in_one_json_call() -> None:
    chat = FakeChat(rewrite_reply("Lisbon trip"))
    usage = Usage()
    got = QueryRewriter(chat).rewrite("the trip", today=TODAY, usage=usage)
    assert got == Rewrite(outcome="applied", queries=("Lisbon trip",))
    [call] = chat.calls
    assert call["system"] == REWRITE_SYSTEM
    assert call["prompt"] == "Today's date: 2026-09-23\nQuery: the trip"
    assert call["json_object"] is True
    assert call["max_completion_tokens"] == REWRITE_MAX_COMPLETION_TOKENS
    assert call["timeout"] == DEFAULT_TIMEOUT == 10.0
    assert call["usage"] is usage


def test_the_rewriter_reads_the_date_range() -> None:
    chat = FakeChat(rewrite_reply(start="2024-03-01", end="2024-03-31"))
    got = QueryRewriter(chat).rewrite("what did I do in March 2024", today=TODAY)
    assert (got.outcome, got.queries) == ("applied", ())
    assert (got.date_from, got.date_to) == (date(2024, 3, 1), date(2024, 3, 31))
    assert got.valid_at is None, "only the retriever decides whether the range is used"


def test_the_rewriter_keeps_three_distinct_new_queries_and_skips_the_rest() -> None:
    queries, start, end = parse_rewrite(
        '{"queries": ["  ", 7, "The Trip", "a", "A", "b", "c", "d"]}', "the trip")
    assert queries == ("a", "b", "c")
    assert (start, end) == (None, None)


@pytest.mark.parametrize("reply", [
    "not json",
    "[1, 2]",
    '{"alternatives": ["a"]}',
    '{"queries": "a"}',
    '{"queries": [], "date_range": "March"}',
    '{"queries": [], "date_range": {"from": "2024-03-01"}}',
    '{"queries": [], "date_range": {"from": "March 1", "to": "2024-03-31"}}',
    '{"queries": [], "date_range": {"from": "2024-13-01", "to": "2024-13-31"}}',
    '{"queries": [], "date_range": {"from": "2024-04-01", "to": "2024-03-01"}}',
])
def test_a_reply_the_rewriter_cannot_read_is_a_malformed_fallback(reply: str) -> None:
    got = QueryRewriter(FakeChat(reply)).rewrite("q", today=TODAY)
    assert got == Rewrite(outcome="fallback", reason="malformed")


@pytest.mark.parametrize("code", [401, 403])
def test_a_rejected_key_is_its_own_outcome_not_a_fallback(code: int) -> None:
    got = QueryRewriter(FakeChat(raises=status(code))).rewrite("q", today=TODAY)
    assert got == Rewrite(outcome="key_rejected", status=code)


def test_a_provider_status_other_than_a_rejected_key_is_a_provider_fallback() -> None:
    got = QueryRewriter(FakeChat(raises=status(503))).rewrite("q", today=TODAY)
    assert got == Rewrite(outcome="fallback", reason="provider", status=503)


def test_any_other_failure_is_an_error_fallback() -> None:
    got = QueryRewriter(FakeChat(raises=ConnectionError("down"))).rewrite("q", today=TODAY)
    assert got == Rewrite(outcome="fallback", reason="error")


def test_a_timeout_raised_by_the_backend_is_a_timeout() -> None:
    got = QueryRewriter(FakeChat(raises=TimeoutError())).rewrite("q", today=TODAY)
    assert got == Rewrite(outcome="fallback", reason="timeout")


def test_a_failure_after_the_deadline_is_a_timeout_whatever_it_was() -> None:
    clock = Clock()
    chat = FakeChat(raises=ConnectionError("gave up"), clock=clock, advance=10.5)
    got = QueryRewriter(chat, clock=clock).rewrite("q", today=TODAY)
    assert got == Rewrite(outcome="fallback", reason="timeout")


def test_a_good_reply_after_the_deadline_is_a_timeout_and_is_not_used() -> None:
    clock = Clock()
    chat = FakeChat(rewrite_reply("a"), clock=clock, advance=10.5)
    got = QueryRewriter(chat, clock=clock).rewrite("q", today=TODAY)
    assert got == Rewrite(outcome="fallback", reason="timeout")


def test_a_reply_just_inside_the_deadline_is_used() -> None:
    clock = Clock()
    chat = FakeChat(rewrite_reply("a"), clock=clock, advance=9.9)
    assert QueryRewriter(chat, clock=clock).rewrite("q", today=TODAY).outcome == "applied"


@pytest.mark.parametrize("cls", [QueryRewriter, Synthesizer])
def test_a_backend_that_cannot_chat_is_refused_at_construction(cls: Any) -> None:
    with pytest.raises(TypeError, match="needs a backend with .chat()"):
        cls(NullLLM())


# --- Synthesizer on its own ----------------------------------------------------------


def test_the_synthesizer_sends_the_question_and_the_notes() -> None:
    chat = FakeChat('{"synthesis": " They live in Porto now. "}')
    got = Synthesizer(chat).synthesize("where do I live", "Known:\n- user lives in Porto",
                                       today=TODAY)
    assert got == Synthesis(outcome="applied", text="They live in Porto now.")
    [call] = chat.calls
    assert call["system"] == SYNTHESIS_SYSTEM
    assert call["prompt"] == ("Today's date: 2026-09-23\nQuestion: where do I live\n\n"
                              "Notes:\nKnown:\n- user lives in Porto")
    assert call["max_completion_tokens"] == SYNTHESIS_MAX_COMPLETION_TOKENS
    assert call["timeout"] == 10.0 and call["json_object"] is True


@pytest.mark.parametrize("reply", ["", "nope", '{"synthesis": ""}', '{"summary": "x"}',
                                   '{"synthesis": 3}'])
def test_a_synthesis_reply_with_no_text_is_a_malformed_fallback(reply: str) -> None:
    got = Synthesizer(FakeChat(reply)).synthesize("q", "notes", today=TODAY)
    assert got == Synthesis(outcome="fallback", reason="malformed")


def test_the_synthesizer_sorts_failures_the_way_the_rewriter_does() -> None:
    clock = Clock()
    late = FakeChat('{"synthesis": "x"}', clock=clock, advance=11)
    assert Synthesizer(late, clock=clock).synthesize("q", "n", today=TODAY) == Synthesis(
        outcome="fallback", reason="timeout")
    assert Synthesizer(FakeChat(raises=status(401))).synthesize(
        "q", "n", today=TODAY) == Synthesis(outcome="key_rejected", status=401)


# --- the read path with no chat backend ----------------------------------------------


class SpyLLM(NullLLM):
    """A backend that cannot chat and records every model call made to it."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def extract(self, *a: Any, **kw: Any) -> Any:
        self.calls.append("extract")
        return super().extract(*a, **kw)

    def resolve_predicate(self, *a: Any, **kw: Any) -> Any:
        self.calls.append("resolve_predicate")
        return super().resolve_predicate(*a, **kw)

    def classify_predicate(self, *a: Any, **kw: Any) -> Any:
        self.calls.append("classify_predicate")
        return super().classify_predicate(*a, **kw)


def test_the_default_read_path_makes_no_model_call_without_a_chat_backend() -> None:
    llm = SpyLLM()
    assert not isinstance(llm, Chat)
    mem = Memvara(llm=llm, embedder=HashingEmbedder(dim=64), user="alice")
    assert mem.reader.rewriter is None and mem.synthesizer is None
    mem.remember("user", "lives_in", "Lisbon")
    hits = mem.search("where do I live")
    assert isinstance(hits, SearchResults)
    assert hits.rewrite == Rewrite(outcome="unconfigured")
    block = mem.recall("where do I live", synthesize=True, with_ids=True)
    assert block.rewrite == Rewrite(outcome="unconfigured")
    assert block.synthesis == Synthesis(outcome="unconfigured")
    assert block.text.splitlines()[0] == "(summary not written — unconfigured.)"
    assert "- user lives in Lisbon" in block.text
    assert llm.calls == []


def test_a_bare_memvara_has_no_backend_for_either_stage() -> None:
    with warnings.catch_warnings():
        # The bare constructor warns that it has no extraction model, once per process.
        warnings.simplefilter("ignore", DegradedExtractionWarning)
        mem = Memvara(embedder=HashingEmbedder(dim=64))
    assert isinstance(mem.llm, NullLLM)
    assert mem.reader.rewriter is None and mem.synthesizer is None


# --- the read path with a chat backend -----------------------------------------------


class ChatLLM(NullLLM):
    """An extraction backend that can also chat, answering from a `FakeChat`."""

    def __init__(self, fake: FakeChat) -> None:
        super().__init__()
        self.fake = fake

    def chat(self, system: str, prompt: str, *, json_object: bool,
             max_completion_tokens: int, timeout: float,
             usage: Usage | None = None) -> str:
        return self.fake.chat(system, prompt, json_object=json_object,
                              max_completion_tokens=max_completion_tokens,
                              timeout=timeout, usage=usage)


def memory(chat: FakeChat | None = None, **kw: Any) -> Memvara:
    if chat is not None:
        kw.setdefault("read_rewriter", QueryRewriter(chat))
    return Memvara(llm=NullLLM(), embedder=HashingEmbedder(dim=64), user="alice", **kw)


def test_an_llm_that_can_chat_turns_both_stages_on_by_default() -> None:
    fake = FakeChat(rewrite_reply())
    mem = Memvara(llm=ChatLLM(fake), embedder=HashingEmbedder(dim=64), user="alice")
    assert isinstance(mem.reader.rewriter, QueryRewriter)
    assert isinstance(mem.synthesizer, Synthesizer)
    mem.remember("user", "lives_in", "Lisbon")
    assert mem.search("where do I live").rewrite.outcome == "applied"
    assert len(fake.calls) == 1 and fake.calls[0]["system"] == REWRITE_SYSTEM


def test_the_alternative_queries_are_searched_and_fused() -> None:
    mem = memory(FakeChat(rewrite_reply("bicycle", "owns a bicycle", "user owns bicycle")))
    mem.remember("user", "likes", "green tea")
    mem.remember("user", "owns", "bicycle")
    plain = mem.search("green tea", k=1, query_rewrite=False)
    assert [r.claim.object for r in plain] == ["green tea"]
    assert plain.rewrite is None
    fused = mem.search("green tea", k=1)
    assert [r.claim.object for r in fused] == ["bicycle"], "found by three of four lists"
    assert fused.rewrite == Rewrite(outcome="applied",
                                    queries=("bicycle", "owns a bicycle",
                                             "user owns bicycle"))


def test_a_row_found_by_several_phrasings_is_listed_once() -> None:
    mem = memory(FakeChat(rewrite_reply("green tea", "tea")))
    mem.remember("user", "likes", "green tea")
    mem.remember("user", "owns", "bicycle")
    hits = mem.search("what does the user like to drink", k=5)
    ids = [r.claim.id for r in hits]
    assert len(ids) == len(set(ids))
    assert hits[0].claim.object == "green tea"


def test_a_rewrite_with_no_alternatives_is_the_plain_read() -> None:
    chat = FakeChat(rewrite_reply())
    mem = memory(chat)
    mem.remember("user", "likes", "green tea")
    hits = mem.search("green tea")
    plain = mem.search("green tea", query_rewrite=False)
    assert [r.claim.id for r in hits] == [r.claim.id for r in plain]
    assert hits.rewrite == Rewrite(outcome="applied")
    assert len(chat.calls) == 1


def test_a_failed_rewrite_serves_the_plain_read_and_says_why() -> None:
    mem = memory(FakeChat("garbage"))
    mem.remember("user", "likes", "green tea")
    mem.remember("user", "owns", "bicycle")
    hits = mem.search("green tea", k=2)
    plain = mem.search("green tea", k=2, query_rewrite=False)
    assert [r.claim.id for r in hits] == [r.claim.id for r in plain]
    assert hits.rewrite == Rewrite(outcome="fallback", reason="malformed")


def test_the_switch_off_reports_disabled_and_makes_no_call() -> None:
    chat = FakeChat(rewrite_reply("a"))
    mem = memory(chat, query_rewrite=False)
    mem.remember("user", "likes", "green tea")
    assert mem.search("green tea").rewrite == Rewrite(outcome="disabled")
    assert chat.calls == []


def test_query_rewrite_false_on_the_call_makes_no_call_and_reports_nothing() -> None:
    chat = FakeChat(rewrite_reply("a"))
    mem = memory(chat)
    mem.remember("user", "likes", "green tea")
    assert mem.search("green tea", query_rewrite=False).rewrite is None
    assert mem.recall("green tea", query_rewrite=False, with_ids=True).rewrite is None
    assert chat.calls == []


def test_a_read_that_cannot_return_anything_makes_no_call() -> None:
    chat = FakeChat(rewrite_reply("a"))
    assert memory(chat).search("anything", k=0) == []
    assert chat.calls == []


def test_a_call_that_is_going_to_raise_raises_before_the_model_is_asked() -> None:
    chat = FakeChat(rewrite_reply("a"))
    mem = memory(chat)
    with pytest.raises(ValueError, match="ranked=True needs turns"):
        mem.search("q", ranked=True)
    with pytest.raises(ValueError):
        mem.search("q", as_of=datetime(2024, 1, 1, tzinfo=UTC),
                   valid_at=datetime(2024, 1, 1, tzinfo=UTC))
    assert chat.calls == []


# --- the date range --------------------------------------------------------------------


def two_homes(chat: FakeChat) -> Memvara:
    mem = memory(chat)
    mem.remember("user", "lives_in", "Lisbon", valid_from=datetime(2020, 1, 1, tzinfo=UTC),
                 valid_to=datetime(2025, 1, 1, tzinfo=UTC))
    mem.remember("user", "lives_in", "Porto", valid_from=datetime(2025, 1, 1, tzinfo=UTC))
    return mem


def test_the_date_range_becomes_valid_at_at_the_last_second_of_its_final_day() -> None:
    mem = two_homes(FakeChat(rewrite_reply(start="2024-03-01", end="2024-03-31")))
    hits = mem.search("where did I live in March 2024")
    assert [r.claim.object for r in hits] == ["Lisbon"]
    assert hits.rewrite.valid_at == datetime(2024, 3, 31, 23, 59, 59, tzinfo=UTC)
    assert [r.claim.object for r in mem.search("where did I live in March 2024",
                                               query_rewrite=False)] == ["Porto"]


def test_the_callers_valid_at_wins_over_the_models_range() -> None:
    mem = two_homes(FakeChat(rewrite_reply(start="2024-03-01", end="2024-03-31")))
    hits = mem.search("where did I live in March 2024",
                      valid_at=datetime(2026, 1, 1, tzinfo=UTC))
    assert [r.claim.object for r in hits] == ["Porto"]
    assert hits.rewrite.valid_at is None
    assert hits.rewrite.date_to == date(2024, 3, 31), "the range is still reported"


def test_the_callers_as_of_wins_over_the_models_range() -> None:
    mem = two_homes(FakeChat(rewrite_reply(start="2024-03-01", end="2024-03-31")))
    hits = mem.search("where did I live", as_of=datetime(2030, 1, 1, tzinfo=UTC))
    assert hits.rewrite.valid_at is None
    assert [r.claim.object for r in hits] == ["Porto"]


def test_a_range_that_has_not_ended_yet_is_a_present_tense_read() -> None:
    mem = two_homes(FakeChat(rewrite_reply(start="2026-01-01", end="2999-12-31")))
    hits = mem.search("where do I live this year")
    assert hits.rewrite.valid_at is None
    assert [r.claim.object for r in hits] == ["Porto"]


def test_recall_uses_the_range_too() -> None:
    mem = two_homes(FakeChat(rewrite_reply(start="2024-03-01", end="2024-03-31")))
    block = mem.recall("where did I live in March 2024", with_ids=True)
    assert "- user lives in Lisbon" in block.text
    assert "Porto" not in block.text
    assert block.rewrite.valid_at == datetime(2024, 3, 31, 23, 59, 59, tzinfo=UTC)


# --- rewrite beside ranked, and telemetry -----------------------------------------------


class KeepTrip:
    """A selector that keeps any turn mentioning 'trip'."""

    top_n = 40

    def admit(self) -> Any:
        from contextlib import nullcontext
        return nullcontext()

    def select(self, question: str, candidates: Any, *, asked_on: Any = None,
               usage: Any = None) -> list[Selected]:
        return [Selected(id=c.id, span=None) for c in candidates if "trip" in c.text]


def test_turns_the_selector_kept_stay_first_ahead_of_the_fused_rows() -> None:
    mem = memory(FakeChat(rewrite_reply("Lisbon", "holiday")), read_selector=KeepTrip())
    mem.add("Loved the trip to Lisbon last spring")
    mem.add("Started a new job on Monday")
    mem.remember("user", "visited", "Lisbon")
    hits = mem.search("the trip", include_episodes=True, ranked=True)
    assert hits.selection is not None and hits.selection.outcome == "applied"
    assert hits.rewrite is not None and hits.rewrite.outcome == "applied"
    first = hits[0]
    assert isinstance(first, EpisodeResult) and first.explain.selected is True
    assert "trip" in first.episode.content
    kept = [r for r in hits if isinstance(r, EpisodeResult) and r.explain.selected]
    assert len(kept) == 1, "a kept turn is not listed again among the fused rows"


def test_a_rewritten_read_searches_document_chunks_too() -> None:
    """Document chunks are episodes, so an alternative phrasing reaches them through the
    same episode legs the original query uses."""
    mem = memory(FakeChat(rewrite_reply("rollback runbook")))
    mem.add_document("To roll back a release, run the rollback runbook step by step.",
                     title="Release notes", extract=False)
    # A floor the original query's weak match does not clear and the phrasing's does.
    plain = mem.search("undo a deploy", include_episodes=True, min_score=0.1,
                       query_rewrite=False)
    assert plain == []
    fused = mem.search("undo a deploy", include_episodes=True, min_score=0.1)
    [chunk] = fused
    assert isinstance(chunk, EpisodeResult) and chunk.episode.meta.get("document_id")
    assert fused.rewrite is not None and fused.rewrite.queries == ("rollback runbook",)


def test_a_rewritten_read_is_observed_once() -> None:
    rec = MemoryRecorder()
    mem = memory(FakeChat(rewrite_reply("a", "b")), telemetry=rec)
    mem.remember("user", "likes", "green tea")
    mem.search("green tea")
    assert rec.total(RETRIEVAL_QUERY) == 1
    mem.search("green tea", query_rewrite=False)
    assert rec.total(RETRIEVAL_QUERY) == 2


def test_reads_that_act_on_their_matches_do_not_rewrite() -> None:
    chat = FakeChat(rewrite_reply("a"))
    mem = memory(chat)
    mem.remember("user", "likes", "green tea")
    mem.forget_matching("green tea", close="ended")
    mem.profile("tea")
    mem.ask("what does the user like?")
    assert chat.calls == []


# --- synthesis -------------------------------------------------------------------------


def synthesizing(reply: str, **kw: Any) -> tuple[Memvara, FakeChat]:
    chat = FakeChat(reply)
    mem = memory(synthesizer=Synthesizer(chat), query_rewrite=False, **kw)
    mem.remember("user", "lives_in", "Lisbon")
    return mem, chat


def test_the_summary_goes_above_the_notes_and_the_notes_stay() -> None:
    mem, chat = synthesizing('{"synthesis": "They live in Lisbon.\\n- forged note"}')
    plain = mem.recall("where do I live")
    block = mem.recall("where do I live", synthesize=True, with_ids=True)
    assert isinstance(block, RecallResult)
    lines = block.text.splitlines()
    assert lines[0] == Memvara.RECALL_SYNTHESIS_HEADER
    assert lines[1] == "They live in Lisbon. - forged note", "flattened to one line"
    assert "\n".join(lines[2:]) == plain
    assert block.synthesis == Synthesis(outcome="applied",
                                        text="They live in Lisbon.\n- forged note")
    [call] = chat.calls
    assert call["prompt"].endswith("Notes:\n" + plain)


def test_without_synthesize_there_is_no_call_and_no_record() -> None:
    mem, chat = synthesizing('{"synthesis": "x"}')
    block = mem.recall("where do I live", with_ids=True)
    assert block.synthesis is None
    assert chat.calls == []


def test_the_synthesis_switch_off_reports_disabled_and_makes_no_call() -> None:
    mem, chat = synthesizing('{"synthesis": "x"}', synthesis=False)
    block = mem.recall("where do I live", synthesize=True, with_ids=True)
    assert block.synthesis == Synthesis(outcome="disabled")
    assert block.text.splitlines()[0] == "(summary not written — disabled.)"
    assert chat.calls == []


def test_a_failed_synthesis_says_so_above_the_notes() -> None:
    mem, _ = synthesizing("not json")
    block = mem.recall("where do I live", synthesize=True, with_ids=True)
    assert block.synthesis == Synthesis(outcome="fallback", reason="malformed")
    assert block.text.splitlines()[0] == "(summary not written — fallback: malformed.)"
    assert "- user lives in Lisbon" in block.text


def test_nothing_recalled_means_no_call_and_an_empty_block() -> None:
    chat = FakeChat('{"synthesis": "x"}')
    mem = memory(synthesizer=Synthesizer(chat))
    block = mem.recall("anything", synthesize=True, with_ids=True)
    assert block.text == ""
    assert block.synthesis == Synthesis(outcome="applied")
    assert chat.calls == []


def many_homes(reply: str) -> tuple[Memvara, FakeChat]:
    chat = FakeChat(reply)
    mem = memory(synthesizer=Synthesizer(chat), query_rewrite=False)
    for city in ("Lisbon", "Porto", "Braga", "Faro", "Evora", "Sintra", "Coimbra",
                 "Aveiro", "Tomar", "Obidos"):
        mem.remember("user", "visited", city)
    return mem, chat


def test_the_summary_is_written_from_the_notes_that_fit_and_only_those() -> None:
    mem, chat = many_homes('{"synthesis": "Several towns."}')
    plain = mem.recall("where have I been", k=10)
    assert plain.count("\n- ") == 10
    # Room for the reserved summary and most of the notes, but not all of them.
    budget = len(plain) + len(Memvara._summary_reserve()) - 60
    block = mem.recall("where have I been", k=10, synthesize=True, with_ids=True,
                       budget=budget, counter=len)
    assert 0 < len(block.claim_ids) < 10, "a partial keep"
    assert block.synthesis == Synthesis(outcome="applied", text="Several towns.")
    notes = "\n".join(block.text.splitlines()[2:])
    [call] = chat.calls
    assert call["prompt"].endswith("Notes:\n" + notes), "exactly the notes shown"
    assert "did not fit" in notes
    assert len(block.text) <= budget


def test_no_summary_is_written_when_no_note_fits_beside_one() -> None:
    mem, chat = many_homes('{"synthesis": "Several towns."}')
    one = mem.recall("where have I been", k=1)
    budget = len(one) + len("(summary not written — fallback: budget.)") + 1
    block = mem.recall("where have I been", k=1, synthesize=True, with_ids=True,
                       budget=budget, counter=len)
    assert chat.calls == [], "nothing is summarised"
    assert block.synthesis == Synthesis(outcome="fallback", reason="budget")
    assert block.text.splitlines()[0] == "(summary not written — fallback: budget.)"
    assert block.text.splitlines()[1:] == one.splitlines(), "the note still shows"


def test_when_even_the_notice_does_not_fit_the_block_keeps_the_notice() -> None:
    mem, chat = many_homes('{"synthesis": "Several towns."}')
    block = mem.recall("where have I been", k=1, synthesize=True, with_ids=True,
                       budget=5, counter=len)
    assert chat.calls == []
    assert block.claim_ids == ()
    assert block.synthesis == Synthesis(outcome="fallback", reason="budget")
    assert block.text.splitlines()[0] == "(summary not written — fallback: budget.)"


def test_a_long_summary_that_does_not_fit_is_reported_and_the_text_agrees() -> None:
    mem, chat = synthesizing('{"synthesis": "' + "They live in Lisbon. " * 60 + '"}')
    plain = mem.recall("where do I live")
    budget = len(plain) + len(Memvara._summary_reserve()) + 1
    block = mem.recall("where do I live", synthesize=True, with_ids=True, budget=budget,
                       counter=len)
    assert len(chat.calls) == 1
    assert block.synthesis == Synthesis(outcome="fallback", reason="budget")
    lines = block.text.splitlines()
    assert lines[0] == "(summary not written — fallback: budget.)"
    assert "\n".join(lines[1:]) == plain, "every note kept"


def test_a_notice_is_kept_under_a_tight_budget_and_names_the_outcome() -> None:
    mem = memory(query_rewrite=False)
    mem.remember("user", "lives_in", "Lisbon")
    plain = mem.recall("where do I live")
    notice = "(summary not written — unconfigured.)"
    block = mem.recall("where do I live", synthesize=True, with_ids=True,
                       budget=len(plain) + len(notice) + 1, counter=len)
    assert block.text == f"{notice}\n{plain}"
    assert block.synthesis == Synthesis(outcome="unconfigured")


def test_a_summary_that_fits_the_budget_is_kept() -> None:
    mem, _ = synthesizing('{"synthesis": "Lisbon."}')
    block = mem.recall("where do I live", synthesize=True, with_ids=True, budget=10_000,
                       counter=len)
    assert block.synthesis is not None and block.synthesis.outcome == "applied"
    assert block.text.splitlines()[1] == "Lisbon."


def test_the_unranked_notice_reaches_the_synthesizer() -> None:
    chat = FakeChat('{"synthesis": "A trip."}')
    mem = memory(synthesizer=Synthesizer(chat), query_rewrite=False)
    mem.add("Loved the trip to Lisbon last spring")
    block = mem.recall("the trip", include_episodes=True, ranked=True, synthesize=True,
                       with_ids=True)
    unranked = Memvara.RECALL_UNRANKED.format(outcome="unconfigured")
    assert block.text.splitlines()[-1] == unranked
    [call] = chat.calls
    assert call["prompt"].endswith(unranked)


def test_the_synthesis_switch_wins_over_a_call_that_asks() -> None:
    mem, chat = synthesizing('{"synthesis": "x"}', synthesis=False)
    assert mem.recall("where do I live", synthesize=True,
                      with_ids=True).synthesis == Synthesis(outcome="disabled")
    assert chat.calls == []


# --- the MCP tools -----------------------------------------------------------------------


def mcp(mem: Memvara, **kw: Any) -> Any:
    from memvara.server import MemvaraMCPServer
    return MemvaraMCPServer(mem, user="alice", **kw)


def listed(srv: Any) -> dict[str, dict]:
    tools = srv.handle_message({"jsonrpc": "2.0", "id": 1,
                                "method": "tools/list"})["result"]["tools"]
    return {t["name"]: t["inputSchema"]["properties"] for t in tools}


def tool_text(srv: Any, name: str, arguments: dict) -> str:
    from test_server import text
    return text(srv, name, arguments)


def test_memory_search_says_what_the_rewrite_added() -> None:
    mem = two_homes(FakeChat(rewrite_reply("home [city]", start="2024-03-01",
                                           end="2024-03-31")))
    out = tool_text(mcp(mem), "memory_search", {"query": "where did I live in March 2024"})
    lines = out.splitlines()
    assert lines[1] == ("Also searched as: 'home ［city］'. The query names 2024-03-01 to "
                        "2024-03-31, so this is as things were on 2024-03-31.")
    assert "lives in Lisbon" in out and "Porto" not in out


def test_memory_search_adds_no_line_when_the_rewrite_added_nothing() -> None:
    mem = two_homes(FakeChat(rewrite_reply()))
    out = tool_text(mcp(mem), "memory_search", {"query": "where do I live"})
    assert "Also searched" not in out and "The query names" not in out
    mem = two_homes(FakeChat("garbage"))
    assert "Also searched" not in tool_text(mcp(mem), "memory_search",
                                            {"query": "where do I live"})


def test_memory_search_query_rewrite_false_skips_the_call() -> None:
    chat = FakeChat(rewrite_reply("a"))
    mem = two_homes(chat)
    tool_text(mcp(mem), "memory_search", {"query": "where do I live",
                                          "query_rewrite": False})
    assert chat.calls == []
    tool_text(mcp(mem), "memory_search", {"query": "where do I live"})
    assert len(chat.calls) == 1, "on by default"


def test_memory_recall_passes_synthesize_through() -> None:
    mem, chat = synthesizing('{"synthesis": "They live in Lisbon."}')
    out = tool_text(mcp(mem), "memory_recall", {"query": "where do I live",
                                                "synthesize": True})
    assert out.splitlines()[:2] == [Memvara.RECALL_SYNTHESIS_HEADER, "They live in Lisbon."]
    assert len(chat.calls) == 1
    assert Memvara.RECALL_SYNTHESIS_HEADER not in tool_text(
        mcp(mem), "memory_recall", {"query": "where do I live"}), "off by default"


@pytest.mark.parametrize("feature, argument, tools", [
    ("query_rewrite", "query_rewrite", {"memory_search", "memory_recall"}),
    ("synthesis", "synthesize", {"memory_recall"}),
])
def test_switching_a_stage_off_removes_its_argument(feature: str, argument: str,
                                                    tools: set[str]) -> None:
    from test_server import call
    mem = memory()
    on = listed(mcp(mem))
    assert {name for name, props in on.items() if argument in props} == tools
    off = listed(mcp(mem, features_off={feature}))
    assert not {name for name, props in off.items() if argument in props}
    body, is_error = call(mcp(mem, features_off={feature}), "memory_recall",
                          {"query": "q", argument: True})
    assert is_error and "unknown argument(s)" in body


def test_a_server_with_query_rewrite_off_never_rewrites() -> None:
    chat = FakeChat(rewrite_reply("a"))
    mem = two_homes(chat)
    srv = mcp(mem, features_off={"query_rewrite"})
    tool_text(srv, "memory_search", {"query": "where do I live"})
    tool_text(srv, "memory_recall", {"query": "where do I live"})
    assert chat.calls == []


def test_the_switches_are_features_and_reach_the_local_store(tmp_path: Any) -> None:
    from memvara.server.config import FEATURES, ServerConfig, build_memvara
    assert {"query_rewrite", "synthesis"} <= set(FEATURES)
    env = {"MEMVARA_DB": str(tmp_path / "m.db"), "MEMVARA_USER": "alice",
           "MEMVARA_EMBEDDER": "hashing:64", "MEMVARA_FEATURE_PROJECT_SCOPE": "0"}
    on = build_memvara(ServerConfig.from_env(env))
    assert on.reader.rewrite_enabled and on.synthesis_enabled
    on.close()
    off = build_memvara(ServerConfig.from_env({
        **env, "MEMVARA_FEATURE_QUERY_REWRITE": "0", "MEMVARA_FEATURE_SYNTHESIS": "0"}))
    assert not off.reader.rewrite_enabled and not off.synthesis_enabled
    off.close()


# --- the hosted client -------------------------------------------------------------------


def remote(payload: dict) -> tuple[Any, list]:
    import httpx

    from memvara.remote.api import RemoteMemvara
    calls: list = []

    def handler(request: Any) -> Any:
        calls.append(request)
        return httpx.Response(200, json=payload)

    mem = RemoteMemvara(api_key="k", base_url="https://example.test")
    mem._http._client._transport = httpx.MockTransport(handler)
    return mem, calls


_SEARCH = {"as_of": None, "valid_at": None, "known_at": None, "states": ["live"],
           "count": 0, "results": []}


def sent(calls: list) -> dict:
    import json
    return json.loads(calls[-1].content)


def test_the_hosted_client_sends_query_rewrite_only_to_opt_out() -> None:
    mem, calls = remote(_SEARCH)
    assert mem.search("q").rewrite is None, "no rewrite on the wire reads as None"
    assert "query_rewrite" not in sent(calls)
    mem.search("q", query_rewrite=False)
    assert sent(calls)["query_rewrite"] is False


def test_the_hosted_client_reads_the_rewrite_off_the_response() -> None:
    mem, _ = remote({**_SEARCH, "rewrite": {
        "outcome": "applied", "reason": None, "status": None, "queries": ["a", "b"],
        "date_from": "2024-03-01", "date_to": "2024-03-31",
        "valid_at": "2024-03-31T23:59:59Z"}})
    got = mem.search("q").rewrite
    assert got == Rewrite(outcome="applied", queries=("a", "b"),
                          date_from=date(2024, 3, 1), date_to=date(2024, 3, 31),
                          valid_at=datetime(2024, 3, 31, 23, 59, 59, tzinfo=UTC))
    mem, _ = remote({**_SEARCH, "rewrite": {"outcome": "key_rejected", "status": 401}})
    assert mem.search("q").rewrite == Rewrite(outcome="key_rejected", status=401)


def test_the_hosted_client_sends_synthesize_only_when_asked() -> None:
    mem, calls = remote({"text": "", "empty": True, "selection": None})
    mem.recall("q")
    assert "synthesize" not in sent(calls) and "query_rewrite" not in sent(calls)
    mem.recall("q", synthesize=True, query_rewrite=False)
    assert sent(calls)["synthesize"] is True and sent(calls)["query_rewrite"] is False


def test_the_async_hosted_client_sends_and_reads_the_same_fields() -> None:
    import asyncio
    import json

    import httpx

    from memvara.remote.aio import AsyncRemoteMemvara
    calls: list = []
    payloads = [{**_SEARCH, "rewrite": {"outcome": "fallback", "reason": "timeout"}},
                {"text": "", "empty": True}]

    def handler(request: Any) -> Any:
        calls.append(request)
        return httpx.Response(200, json=payloads[len(calls) - 1])

    mem = AsyncRemoteMemvara(api_key="k", base_url="https://example.test")
    mem._http._client._transport = httpx.MockTransport(handler)

    async def main() -> Any:
        hits = await mem.search("q", query_rewrite=False)
        await mem.recall("q", synthesize=True)
        await mem.aclose()
        return hits

    hits = asyncio.run(main())
    assert hits.rewrite == Rewrite(outcome="fallback", reason="timeout")
    assert json.loads(calls[0].content)["query_rewrite"] is False
    assert json.loads(calls[1].content)["synthesize"] is True


# --- telemetry and admission -------------------------------------------------------------


def counted(rec: MemoryRecorder, stage: str) -> dict[str, int]:
    from memvara.telemetry import (
        RETRIEVAL_MODEL_FALLBACK, RETRIEVAL_MODEL_QUERY, RETRIEVAL_MODEL_REFUSED,
        RETRIEVAL_TOKENS_IN, RETRIEVAL_TOKENS_OUT)
    return {"query": rec.total(RETRIEVAL_MODEL_QUERY, stage=stage),
            "fallback": rec.total(RETRIEVAL_MODEL_FALLBACK, stage=stage),
            "refused": rec.total(RETRIEVAL_MODEL_REFUSED, stage=stage),
            "tokens_in": rec.total(RETRIEVAL_TOKENS_IN, stage=stage),
            "tokens_out": rec.total(RETRIEVAL_TOKENS_OUT, stage=stage)}


class UsageChat(FakeChat):
    """A fake that reports the tokens a real backend would."""

    def chat(self, system: str, prompt: str, **kw: Any) -> str:
        usage = kw.get("usage")
        if usage is not None and self.raises is None:
            usage.add(12, 7)
        return super().chat(system, prompt, **kw)


def rewrite_telemetry(chat: FakeChat, **kw: Any) -> MemoryRecorder:
    rec = MemoryRecorder()
    mem = memory(chat, telemetry=rec, **kw)
    mem.remember("user", "likes", "green tea")
    mem.search("green tea")
    return rec


def synthesis_telemetry(chat: FakeChat, **kw: Any) -> MemoryRecorder:
    rec = MemoryRecorder()
    mem = memory(synthesizer=Synthesizer(chat), telemetry=rec, query_rewrite=False, **kw)
    mem.remember("user", "likes", "green tea")
    mem.recall("green tea", synthesize=True)
    return rec


@pytest.mark.parametrize("stage, run, reply", [
    ("rewrite", rewrite_telemetry, rewrite_reply("tea")),
    ("synthesis", synthesis_telemetry, '{"synthesis": "Green tea."}'),
])
def test_an_answered_call_is_counted_with_its_tokens_and_time(stage: str, run: Any,
                                                              reply: str) -> None:
    from memvara.telemetry import RETRIEVAL_REWRITE_MS, RETRIEVAL_SYNTHESIS_MS
    rec = run(UsageChat(reply))
    assert counted(rec, stage) == {"query": 1, "fallback": 0, "refused": 0,
                                   "tokens_in": 12, "tokens_out": 7}
    timer = RETRIEVAL_REWRITE_MS if stage == "rewrite" else RETRIEVAL_SYNTHESIS_MS
    assert len(rec.values(timer)) == 1


@pytest.mark.parametrize("stage, run", [("rewrite", rewrite_telemetry),
                                        ("synthesis", synthesis_telemetry)])
def test_a_failed_call_is_counted_as_a_fallback_with_its_reason(stage: str, run: Any) -> None:
    from memvara.telemetry import RETRIEVAL_MODEL_FALLBACK
    rec = run(FakeChat("garbage"))
    assert counted(rec, stage)["query"] == 0
    assert rec.total(RETRIEVAL_MODEL_FALLBACK, stage=stage, reason="malformed") == 1
    clock = Clock()
    late = run(FakeChat(raises=ConnectionError(), clock=clock, advance=11))
    assert late.total(RETRIEVAL_MODEL_FALLBACK, stage=stage, reason="error") == 1
    provider = run(FakeChat(raises=status(503)))
    assert provider.total(RETRIEVAL_MODEL_FALLBACK, stage=stage, reason="provider",
                          status="503") == 1


@pytest.mark.parametrize("stage", ["rewrite", "synthesis"])
def test_a_timeout_is_counted_as_a_timeout(stage: str) -> None:
    from memvara.telemetry import RETRIEVAL_MODEL_FALLBACK
    clock = Clock()
    chat = FakeChat(rewrite_reply("a") if stage == "rewrite" else '{"synthesis": "x"}',
                    clock=clock, advance=11)
    rec = MemoryRecorder()
    if stage == "rewrite":
        mem = memory(read_rewriter=QueryRewriter(chat, clock=clock), telemetry=rec)
        mem.remember("user", "likes", "green tea")
        mem.search("green tea")
    else:
        mem = memory(synthesizer=Synthesizer(chat, clock=clock), telemetry=rec,
                     query_rewrite=False)
        mem.remember("user", "likes", "green tea")
        mem.recall("green tea", synthesize=True)
    assert rec.total(RETRIEVAL_MODEL_FALLBACK, stage=stage, reason="timeout") == 1


@pytest.mark.parametrize("stage, run", [("rewrite", rewrite_telemetry),
                                        ("synthesis", synthesis_telemetry)])
def test_a_rejected_key_is_counted_as_refused(stage: str, run: Any) -> None:
    from memvara.telemetry import RETRIEVAL_MODEL_REFUSED
    rec = run(FakeChat(raises=status(401)))
    assert rec.total(RETRIEVAL_MODEL_REFUSED, stage=stage, reason="key_rejected") == 1
    assert counted(rec, stage)["query"] == 0


def test_a_stage_that_cannot_run_is_counted_as_refused_and_makes_no_call() -> None:
    from memvara.telemetry import RETRIEVAL_MODEL_REFUSED, RETRIEVAL_REWRITE_MS
    chat = FakeChat(rewrite_reply("a"))
    rec = rewrite_telemetry(chat, query_rewrite=False)
    assert rec.total(RETRIEVAL_MODEL_REFUSED, stage="rewrite", reason="disabled") == 1
    rec = MemoryRecorder()
    mem = memory(telemetry=rec)
    mem.search("anything")
    assert rec.total(RETRIEVAL_MODEL_REFUSED, stage="rewrite", reason="unconfigured") == 1
    assert rec.values(RETRIEVAL_REWRITE_MS) == [] and chat.calls == []


class Busy(QueryRewriter):
    """A rewriter whose deployment cap is full."""

    def admit(self) -> Any:
        from memvara.select import SelectorBusy
        raise SelectorBusy()


class Switched(Synthesizer):
    """A synthesizer an operator switched off at admission."""

    def admit(self) -> Any:
        from memvara.select import SelectorRefused
        raise SelectorRefused("disabled")


def test_a_full_admission_cap_is_a_busy_fallback_and_the_read_goes_on() -> None:
    from memvara.telemetry import RETRIEVAL_MODEL_FALLBACK
    chat = FakeChat(rewrite_reply("a"))
    rec = MemoryRecorder()
    mem = memory(read_rewriter=Busy(chat), telemetry=rec)
    mem.remember("user", "likes", "green tea")
    hits = mem.search("green tea")
    assert hits.rewrite == Rewrite(outcome="fallback", reason="busy")
    assert [r.claim.object for r in hits] == ["green tea"]
    assert rec.total(RETRIEVAL_MODEL_FALLBACK, stage="rewrite", reason="busy") == 1
    assert chat.calls == []


def test_a_refused_admission_reports_the_refusal() -> None:
    chat = FakeChat('{"synthesis": "x"}')
    mem = memory(synthesizer=Switched(chat), query_rewrite=False)
    mem.remember("user", "likes", "green tea")
    block = mem.recall("green tea", synthesize=True, with_ids=True)
    assert block.synthesis == Synthesis(outcome="disabled")
    assert chat.calls == []


def test_the_read_latency_includes_the_rewrite(monkeypatch: Any) -> None:
    import memvara.retrieve.hybrid as hybrid
    import memvara.select.stages as stages
    from memvara.telemetry import RETRIEVAL_LATENCY_MS, RETRIEVAL_REWRITE_MS
    clock = Clock()
    monkeypatch.setattr(hybrid, "perf_counter", clock)
    monkeypatch.setattr(stages, "perf_counter", clock)
    rec = MemoryRecorder()
    chat = FakeChat(rewrite_reply("tea"), clock=clock, advance=4.0)
    mem = memory(read_rewriter=QueryRewriter(chat, clock=clock), telemetry=rec)
    mem.remember("user", "likes", "green tea")
    mem.search("green tea")
    assert rec.values(RETRIEVAL_REWRITE_MS) == [4000.0]
    assert rec.values(RETRIEVAL_LATENCY_MS) == [4000.0]


def test_the_rewrite_switch_wins_over_a_call_that_asks() -> None:
    chat = FakeChat(rewrite_reply("a"))
    mem = memory(chat, query_rewrite=False)
    mem.remember("user", "likes", "green tea")
    assert mem.search("green tea", query_rewrite=True).rewrite == Rewrite(outcome="disabled")
    assert mem.recall("green tea", query_rewrite=True,
                      with_ids=True).rewrite == Rewrite(outcome="disabled")
    assert chat.calls == []


# --- the work a rewritten read does ------------------------------------------------------


class CountingEmbedder(HashingEmbedder):
    def __init__(self) -> None:
        super().__init__(dim=64)
        self.batches: list[list[str]] = []

    def encode(self, texts: Any) -> Any:
        self.batches.append(list(texts))
        return super().encode(texts)


def test_every_phrasing_is_embedded_once_in_one_call() -> None:
    emb = CountingEmbedder()
    mem = Memvara(llm=NullLLM(), embedder=emb, user="alice",
                  read_rewriter=QueryRewriter(FakeChat(rewrite_reply("tea", "drink"))))
    mem.add("I drink green tea every morning")
    emb.batches.clear()
    mem.search("green tea", include_episodes=True)
    assert emb.batches == [["green tea", "tea", "drink"]]


def test_a_plain_read_embeds_its_query_once_for_both_vector_legs() -> None:
    emb = CountingEmbedder()
    mem = Memvara(llm=NullLLM(), embedder=emb, user="alice", query_rewrite=False)
    mem.add("I drink green tea every morning")
    emb.batches.clear()
    mem.search("green tea", include_episodes=True)
    assert emb.batches == [["green tea"]]


class CountingReranker:
    def __init__(self) -> None:
        self.calls = 0

    def score(self, query: str, texts: Any) -> list[float]:
        self.calls += 1
        return [float(len(t)) for t in texts]


def test_a_rewritten_read_reranks_once_after_fusion() -> None:
    rr = CountingReranker()
    mem = memory(FakeChat(rewrite_reply("tea", "drink", "morning")), read_reranker=rr)
    mem.remember("user", "likes", "green tea")
    mem.remember("user", "drinks", "coffee")
    hits = mem.search("green tea", k=1)
    assert rr.calls == 1
    assert len(hits) == 1


def test_the_alternative_phrasings_run_on_their_own_threads(monkeypatch: Any) -> None:
    import threading
    from memvara.retrieve.hybrid import HybridRetriever
    seen: list[tuple[str, str]] = []
    real = HybridRetriever._retrieve

    def spy(self: Any, query: str, **kw: Any) -> Any:
        seen.append((query, threading.current_thread().name))
        return real(self, query, **kw)

    monkeypatch.setattr(HybridRetriever, "_retrieve", spy)
    mem = memory(FakeChat(rewrite_reply("tea", "drink")))
    mem.remember("user", "likes", "green tea")
    mem.search("green tea")
    names = dict(seen)
    assert not names["green tea"].startswith("memvara-phrasing")
    assert names["tea"].startswith("memvara-phrasing")
    assert names["drink"].startswith("memvara-phrasing")


def test_the_async_facade_reads_on_its_own_bounded_pool() -> None:
    import asyncio
    import threading
    from memvara.aio import READ_THREADS, AsyncMemvara
    mem = memory()
    mem.remember("user", "likes", "green tea")
    names: list[str] = []
    real = mem.search

    def spy(*a: Any, **kw: Any) -> Any:
        names.append(threading.current_thread().name)
        return real(*a, **kw)

    mem.search = spy  # type: ignore[method-assign]

    async def main() -> Any:
        return await AsyncMemvara(mem).search("green tea")

    hits = asyncio.run(main())
    assert [r.claim.object for r in hits] == ["green tea"]
    assert names[0].startswith("memvara-read") and READ_THREADS == 8


# --- reads that must stay plain ------------------------------------------------------------


def test_the_mem0_shim_is_deterministic_unless_asked() -> None:
    from memvara.compat.mem0 import Memory
    chat = FakeChat(rewrite_reply("tea"))
    shim = Memory(memory(chat))
    shim.add("I like green tea", filters={"user_id": "alice"})
    shim.search("green tea", filters={"user_id": "alice"})
    assert chat.calls == []
    shim.search("green tea", filters={"user_id": "alice"}, rewrite=True)
    assert len(chat.calls) == 1


def test_the_benchmark_reads_are_plain_unless_a_run_asks() -> None:
    import sys
    sys.path.insert(0, ".")
    from bench import evalkit as ek
    chat = FakeChat(rewrite_reply("tea"))
    mem = memory(chat)
    mem.remember("user", "likes", "green tea")
    budget = ek.RetrievalBudget(k=3)
    ek.retrieve(mem, "green tea", budget, ek.ContextSource.MEMORY, "")
    ek.retrieval_pass(mem, "green tea", ek.RetrievalPlan(), budget, {})
    assert chat.calls == []
    ek.retrieve(mem, "green tea", budget, ek.ContextSource.MEMORY, "", query_rewrite=True)
    assert len(chat.calls) == 1


# --- the hosted client, older deployments, and the constructor ---------------------------


def test_an_older_deployment_that_refuses_the_opt_out_is_asked_again_without_it() -> None:
    import json

    import httpx

    from memvara.remote.api import RemoteMemvara
    sent: list[dict] = []

    def handler(request: Any) -> Any:
        sent.append(json.loads(request.content))
        if "query_rewrite" in sent[-1]:
            return httpx.Response(422, json={"error": {
                "code": "invalid_request", "message": "query_rewrite: extra field"}})
        return httpx.Response(200, json=_SEARCH if request.url.path.endswith("search")
                              else {"text": "Known:\n- x", "empty": False})

    mem = RemoteMemvara(api_key="k", base_url="https://example.test")
    mem._http._client._transport = httpx.MockTransport(handler)
    assert mem.search("q", query_rewrite=False) == []
    assert [("query_rewrite" in b) for b in sent] == [True, False]
    assert mem.recall("q", query_rewrite=False) == "Known:\n- x"


def test_a_refused_synthesize_is_not_retried() -> None:
    import httpx

    from memvara.remote.api import RemoteMemvara
    from memvara.remote.errors import InvalidRequest
    calls: list[Any] = []

    def handler(request: Any) -> Any:
        calls.append(request)
        return httpx.Response(422, json={"error": {"code": "invalid_request",
                                                   "message": "synthesize: extra field"}})

    mem = RemoteMemvara(api_key="k", base_url="https://example.test")
    mem._http._client._transport = httpx.MockTransport(handler)
    with pytest.raises(InvalidRequest):
        mem.recall("q", synthesize=True)
    assert len(calls) == 1


def test_the_async_client_asks_an_older_deployment_again_too() -> None:
    import asyncio
    import json

    import httpx

    from memvara.remote.aio import AsyncRemoteMemvara
    sent: list[dict] = []

    def handler(request: Any) -> Any:
        sent.append(json.loads(request.content))
        if "query_rewrite" in sent[-1]:
            return httpx.Response(422, json={"error": {"code": "invalid_request",
                                                       "message": "extra field"}})
        return httpx.Response(200, json=_SEARCH)

    mem = AsyncRemoteMemvara(api_key="k", base_url="https://example.test")
    mem._http._client._transport = httpx.MockTransport(handler)

    async def main() -> Any:
        hits = await mem.search("q", query_rewrite=False)
        await mem.aclose()
        return hits

    assert asyncio.run(main()) == []
    assert [("query_rewrite" in b) for b in sent] == [True, False]

    refused = AsyncRemoteMemvara(api_key="k", base_url="https://example.test")
    refused._http._client._transport = httpx.MockTransport(
        lambda request: httpx.Response(422, json={"error": {"code": "invalid_request",
                                                            "message": "no"}}))

    async def plain() -> Any:
        try:
            return await refused.search("q")
        finally:
            await refused.aclose()

    from memvara.remote.errors import InvalidRequest
    with pytest.raises(InvalidRequest):
        asyncio.run(plain())


def test_another_refusal_is_raised_after_the_second_try() -> None:
    import httpx

    from memvara.remote.api import RemoteMemvara
    from memvara.remote.errors import InvalidRequest
    calls: list[Any] = []

    def handler(request: Any) -> Any:
        calls.append(request)
        return httpx.Response(422, json={"error": {"code": "invalid_request",
                                                   "message": "k: out of range"}})

    mem = RemoteMemvara(api_key="k", base_url="https://example.test")
    mem._http._client._transport = httpx.MockTransport(handler)
    with pytest.raises(InvalidRequest):
        mem.search("q", query_rewrite=False)
    assert len(calls) == 2
    with pytest.raises(InvalidRequest):
        mem.search("q")
    assert len(calls) == 3, "no retry when the opt-out was not sent"


@pytest.mark.parametrize("body", [
    {"outcome": "applied", "date_from": "2024-03-01"},
    {"outcome": "applied", "date_to": "2024-03-31"},
    {"outcome": "applied", "valid_at": "2024-03-31T23:59:59Z"},
    {"outcome": "applied", "date_from": "March", "date_to": "2024-03-31"},
])
def test_a_partial_or_malformed_range_from_the_wire_is_refused(body: dict) -> None:
    from memvara.remote import hydrate
    with pytest.raises(ValueError):
        hydrate.rewrite(body)


def test_the_search_line_refuses_to_render_a_partial_range() -> None:
    from memvara.server.tools import _rewrite_line
    partial = Rewrite(outcome="applied", queries=("a",),
                      valid_at=datetime(2024, 3, 31, tzinfo=UTC))
    assert _rewrite_line(partial) == "Also searched as: 'a'."


def test_a_hosted_client_accepts_the_default_switches_and_refuses_switching_off() -> None:
    from memvara.remote.api import RemoteMemvara
    client = Memvara(api_key="k", base_url="https://example.test", query_rewrite=True,
                     synthesis=True)
    assert isinstance(client, RemoteMemvara)
    for name in ("query_rewrite", "synthesis"):
        with pytest.raises(TypeError, match=f"{name} cannot be combined"):
            Memvara(api_key="k", base_url="https://example.test", **{name: False})


# --- the MCP no-match reply ------------------------------------------------------------


def test_a_dated_miss_names_the_day_the_rewrite_used() -> None:
    mem = memory(FakeChat(rewrite_reply(start="2019-03-01", end="2019-03-31")))
    mem.remember("user", "lives_in", "Porto", valid_from=datetime(2025, 1, 1, tzinfo=UTC))
    srv = mcp(mem)
    for tool in ("memory_search", "memory_recall"):
        out = tool_text(srv, tool, {"query": "where did I live in March 2019"})
        assert "as things were on 2019-03-31" in out, tool


# --- invariant 1, enforced -------------------------------------------------------------------

#: Every call in the package that can reach a model, as `module::function: receiver.method`.
#: The read side is the three named stages and the one chat call they share; the write side
#: is extraction, predicate acquisition and replacement advice, and ingestion reads media. A new entry here is a new
#: way to reach a model, and `docs/INTERNALS.md` invariant 1 says where one may live.
MODEL_CALLS = {
    # read path: `ranked`, `query_rewrite`, `synthesis`, and their shared chat call
    "memvara/retrieve/hybrid.py::_run_ranked_stage: selector.select",
    "memvara/retrieve/hybrid.py::_rewrite: stage.rewrite",
    "memvara/core.py::recall: stage.synthesize",
    "memvara/select/chat.py::call_chat: llm.chat",
    # write path
    "memvara/core.py::_advise_replacements: judge.judge_replacement",
    "memvara/compat/mem0_import.py::_extract: llm.extract",
    "memvara/write/pipeline.py::_tier1: self.fast.extract",
    "memvara/write/pipeline.py::_extract: self.llm.extract",
    "memvara/write/pipeline.py::_acquire: self.llm.classify_predicate",
    # ingestion, when a document's media is turned into text
    "memvara/documents/service.py::_text: ingest.extract",
    "memvara/ingest/media.py::media_to_text: llm.describe_image",
    "memvara/ingest/media.py::media_to_text: llm.transcribe",
}
_MODEL_METHODS = {"chat", "extract", "resolve_predicate", "classify_predicate",
                  "judge_replacement", "compose_relations", "select", "rewrite",
                  "synthesize", "describe_image", "transcribe"}


def _model_calls() -> set[str]:
    import ast
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    found: set[str] = set()
    for path in sorted((root / "memvara").rglob("*.py")):
        if "skills" in path.parts:
            continue
        stack: list[str] = []

        class Visit(ast.NodeVisitor):
            def visit_FunctionDef(self, node: Any) -> None:
                stack.append(node.name)
                self.generic_visit(node)
                stack.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Call(self, node: ast.Call) -> None:
                func = node.func
                if (isinstance(func, ast.Attribute) and func.attr in _MODEL_METHODS
                        and ast.unparse(func.value) != "hydrate"):
                    found.add(f"{path.relative_to(root).as_posix()}::"
                              f"{'.'.join(stack)}: {ast.unparse(func.value)}.{func.attr}")
                self.generic_visit(node)

        Visit().visit(ast.parse(path.read_text(encoding="utf-8")))
    return found


def test_a_model_is_reached_only_from_the_places_invariant_1_names() -> None:
    assert _model_calls() == MODEL_CALLS


#: The `search()` and `recall()` calls in the library, the benchmarks, the demo and the hooks
#: that pass neither `query_rewrite=` nor `**PLAIN_READ`, each with the reason it may.
#: Every other call must say which kind of read it is, so that a new one cannot forget.
DECLARED_ELSEWHERE = {
    "memvara/integrations/crewai.py: self.memory.search":
        "an agent framework's retrieval of the user's own query; the store's default holds",
    "memvara/integrations/langchain.py: self.memory.recall": "forwards the caller's keywords",
    "memvara/integrations/langchain.py: self.memory.search": "forwards the caller's keywords",
    "memvara/integrations/langchain.py: memory.search":
        "a retriever over the user's own query; the store's default holds",
    "memvara/integrations/langgraph.py: self.memory.search":
        "search_memory forwards the caller's keywords; _rank passes PLAIN_READ",
    "memvara/integrations/llamaindex.py: memory.search":
        "a retriever over the user's own query, or the caller's keywords forwarded",
    "memvara/integrations/llamaindex.py: self.memory.search":
        "a retriever over the user's own query; the store's default holds",
    "memvara/store/sqlite.py: self._vec.search": "the vector index, not a read facade",
    "bench/compare.py: base.search": "the mem0-style baseline, not memvara",
    "bench/mem0_real.py: api.search": "mem0 itself",
    "demo/competitors.py: store.search": "a competitor's client",
    "plugin/hooks/daemon.py: self.store.recall":
        "the per-prompt recall, whose rewrite is decided by stream P2-H",
    "plugin/hooks/lib/fast.py: client.recall": "the per-prompt recall, stream P2-H",
    "plugin/hooks/lib/fast.py: store.recall": "the per-prompt recall, stream P2-H",
}


def _undeclared_reads() -> set[str]:
    import ast
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parent.parent
    found: set[str] = set()
    for tree in ("memvara", "bench", "demo", "plugin/hooks"):
        for path in sorted((root / tree).rglob("*.py")):
            if "skills" in path.parts:
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr in ("search", "recall")):
                    continue
                receiver = ast.unparse(node.func.value)
                if re.fullmatch(r"_[A-Z_]+|re|.*\.pattern", receiver):
                    continue  # a regular expression
                declared = any(k.arg == "query_rewrite" for k in node.keywords) or any(
                    k.arg is None and ast.unparse(k.value) in ("PLAIN_READ", "plain_read")
                    for k in node.keywords)
                if not declared:
                    found.add(f"{path.relative_to(root).as_posix()}: "
                              f"{receiver}.{node.func.attr}")
    return found


def test_every_read_in_this_repository_says_whether_it_may_call_a_model() -> None:
    assert _undeclared_reads() == set(DECLARED_ELSEWHERE)
