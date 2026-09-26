"""The redaction seam: every field it is offered reaches it before anything durable.

`memvara/redact.py` is the last place text can be changed before it becomes durable, and
the ordering is the feature: the redactor runs before the content hash, the episode row,
the text index, the embedder and the extraction model — two of which leave the process. So
a field in `redact.FIELDS` written without passing through the redactor is in scope. This
drives every write door and checks the redactor saw each field and the store holds the
redacted text; then it pins the misses the built-in redactor's docstring already lists,
which are documented behaviour rather than bugs.
"""

from __future__ import annotations

from typing import Any

import pytest

from harness import stores
from memvara.redact import (CLAIM_OBJECT, CLAIM_SUBJECT, CLAIM_TEXT, EPISODE, FIELDS,
                            PatternRedactor)
from memvara.types import Episode, Scope

SENTINEL = "SEEKRETMARK"


class Recorder:
    """A redactor that records every (field, text) it is offered and rewrites a sentinel."""

    def __init__(self) -> None:
        self.seen: "list[tuple[str, str]]" = []

    def redact(self, text: str, /, *, field: str, scope: Scope) -> str:
        self.seen.append((field, text))
        return text.replace(SENTINEL, "[redacted]")

    def fields(self) -> "set[str]":
        return {field for field, _ in self.seen}


def _store(tmp_path, recorder: Recorder) -> Any:
    return stores.file(tmp_path / "store.db", redactor=recorder, user="alice")


def test_a_user_turn_offers_the_episode_and_all_three_claim_fields(tmp_path) -> None:
    """`add` of a user turn redacts the turn (EPISODE) and, because the fast path extracts
    a claim from it, that claim's subject, object and text — every field in FIELDS."""
    rec = Recorder()
    with _store(tmp_path, rec) as mem:
        mem.add(f"I live in {SENTINEL}", role="user")
    assert {EPISODE, CLAIM_SUBJECT, CLAIM_OBJECT, CLAIM_TEXT} <= rec.fields()


def test_a_structured_write_offers_the_three_claim_fields(tmp_path) -> None:
    """`remember` writes a claim that never had a turn, so it must pass the redactor at the
    claim door: subject, object and rendered text."""
    rec = Recorder()
    with _store(tmp_path, rec) as mem:
        mem.remember("user", "lives_in", SENTINEL, text=f"user lives in {SENTINEL}")
        assert {CLAIM_SUBJECT, CLAIM_OBJECT, CLAIM_TEXT} <= rec.fields()
        # The store holds the redacted value, not the raw one.
        hits = list(mem.search("lives", user="alice"))
        assert hits and all(SENTINEL not in r.text for r in hits)


def test_an_attached_source_turn_is_redacted_at_the_remember_door(tmp_path) -> None:
    """`remember(sources=[Episode(...)])` writes the turn through a different door than
    `add`; the seam has to hold there too, or the call that attaches provenance leaks."""
    rec = Recorder()
    with _store(tmp_path, rec) as mem:
        mem.remember("user", "note", "x", sources=[Episode(content=f"turn {SENTINEL}")])
    assert EPISODE in rec.fields()
    assert any(SENTINEL in text for f, text in rec.seen if f == EPISODE)


def test_a_document_body_and_title_are_redacted_before_they_are_chunked(tmp_path) -> None:
    """A document's text is redacted before it is chunked, hashed or indexed, so no chunk
    digest can confirm a value the redactor removed."""
    rec = Recorder()
    with _store(tmp_path, rec) as mem:
        mem.add_document(f"document body {SENTINEL}", title=f"title {SENTINEL}")
    assert EPISODE in rec.fields()


def test_the_documented_set_of_fields_has_not_grown_unnoticed() -> None:
    """FIELDS is the whole set a redactor is offered. Pinned so a new call site that adds a
    field to the durable path is noticed here rather than found by an auditor."""
    assert FIELDS == (EPISODE, CLAIM_SUBJECT, CLAIM_OBJECT, CLAIM_TEXT)


@pytest.mark.parametrize("text", [
    "call me on 5551234567",                       # an unpunctuated digit run
    "my mother lives at 14 Rue de la Paix",        # a name and an address in prose
    "電話番号は五五五",  # a non-Latin-script value
])
def test_the_built_in_redactor_misses_what_its_docstring_lists(text) -> None:
    """`PatternRedactor` documents its misses — unpunctuated digit runs, prose PII, and
    non-Latin scripts — because a redactor whose limits are undocumented is worse than
    none. These are documented behaviour, not bugs: the seam exists so a deployment brings
    its own ruleset."""
    assert PatternRedactor().redact(text, field=EPISODE, scope=Scope()) == text


def test_a_number_split_across_two_turns_loses_only_the_half_that_matched() -> None:
    """Documented: each turn is redacted alone, so a number given across two turns is not
    reassembled — the built-in redactor sees only the punctuated half."""
    redactor = PatternRedactor()
    first = redactor.redact("my number is", field=EPISODE, scope=Scope())
    second = redactor.redact("(555) 123-4567", field=EPISODE, scope=Scope())
    assert first == "my number is"                 # nothing to match in the first turn
    assert "[redacted:phone]" in second            # only the second half is caught
