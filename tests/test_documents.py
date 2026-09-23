"""The document store: chunking, re-ingest by `custom_id`, delete, status and scope.

The acceptance tests from the design (docs/superpowers/specs/2026-09-23-parity-phase-2-
documents-and-retrieval-design.md, sections 3, 4.1 and 4.3): a re-ingest matches chunks by
the digest of their text rather than by position, so an edit near the top keeps the rest;
a delete erases the document's text and retires, never erases, the memories it was the
only source of; a failed extraction leaves the document stored and searchable. Around
those, the schema-14 migration from a real version-13 file, the scope rules, the listing,
and the ingestion seam that a separately built package fills in.
"""
from __future__ import annotations

import random
import sqlite3
import sys
import types as pytypes
from dataclasses import dataclass

import pytest

from memvara import HashingEmbedder, Memvara, NullLLM, SQLiteStore
from memvara.documents import CHUNK_CHARS, CHUNK_OVERLAP, normalise, split
from memvara.documents.chunk import _MIN_CHARS
from memvara.store.sqlite import SCHEMA_VERSION
from memvara.types import (DOCUMENT_DELETED_REASON, DOCUMENT_META, Episode, Scope,
                           closure_reasons, content_hash)


def mem(**kw) -> Memvara:
    kw.setdefault("user", "alice")
    return Memvara(llm=NullLLM(), embedder=HashingEmbedder(dim=64), **kw)


_WORDS = ("memory store claim episode vector index retrieval document chunk policy "
          "tenant scope user project agent session erase retire end valid time record "
          "belief clock search recall model extraction pipeline gate reconcile").split()


def prose(paragraphs: int = 30, seed: int = 7) -> list[str]:
    """Paragraphs of plain sentences, deterministic for a seed."""
    rng = random.Random(seed)

    def sentence() -> str:
        words = " ".join(rng.choice(_WORDS) for _ in range(rng.randint(6, 28)))
        return words.capitalize() + "."

    return [" ".join(sentence() for _ in range(rng.randint(3, 8)))
            for _ in range(paragraphs)]


# --- the chunker ----------------------------------------------------------------


def test_a_chunk_never_ends_inside_a_sentence_that_fits():
    """Every chunk ends where a sentence ends. A boundary inside a sentence would split
    one statement across two episodes, and neither half would answer a question about
    it."""
    text = normalise("\n\n".join(prose()))
    chunks = split(text)
    assert len(chunks) > 10
    for chunk in chunks:
        assert chunk.endswith(".")
        assert len(chunk) <= CHUNK_CHARS + CHUNK_OVERLAP


def test_each_chunk_repeats_whole_sentences_from_the_end_of_the_one_before():
    chunks = split(normalise("\n\n".join(prose())))
    for before, after in zip(chunks, chunks[1:]):
        # The overlap is a suffix of the previous chunk and a prefix of the next.
        shared = max(n for n in range(0, CHUNK_OVERLAP + 1)
                     if n == 0 or before.endswith(after[:n]))
        assert shared > 0, "consecutive chunks share no text"
        assert before[-shared - 1].isspace(), "the overlap begins inside a word"
        last_sentence = before[before.rstrip(".").rfind(". ") + 2:]
        if len(last_sentence) <= CHUNK_OVERLAP:
            assert after[0].isupper(), "the overlap begins mid-sentence"


def test_an_edit_near_the_top_changes_at_most_two_chunks():
    """The property re-ingest by content digest depends on. A packer that filled chunks
    from the top would move every boundary after an edit, and a re-ingest would find no
    chunk unchanged; boundaries chosen by local content resynchronise within a chunk or
    two, and the chunk after an edit changes only because it repeats the edited text."""
    paragraphs = prose(40)
    before = split(normalise("\n\n".join(paragraphs)))
    for where in (0, 20, 39):
        edited = list(paragraphs)
        edited[where] = "An inserted sentence changes this paragraph. " + edited[where]
        after = split(normalise("\n\n".join(edited)))
        assert len(set(after) - set(before)) <= 2, where


def test_a_sentence_longer_than_a_chunk_is_cut_at_word_boundaries():
    sentence = " ".join(["word"] * 600) + "."           # about 3,000 characters
    chunks = split(sentence)
    assert len(chunks) >= 3
    for chunk in chunks:
        assert len(chunk) <= CHUNK_CHARS + CHUNK_OVERLAP
        assert not chunk.startswith(("ord", "rd", "d ")), "a cut fell inside a word"


def test_a_single_word_longer_than_a_chunk_is_cut_at_the_limit():
    """Nothing else is possible for a base64 blob or a minified line, and refusing it
    would refuse the whole document."""
    blob = "x" * (CHUNK_CHARS * 2 + 10)
    chunks = split(blob)
    assert [len(c) for c in chunks][:1] == [CHUNK_CHARS]
    assert "".join(chunk[-CHUNK_CHARS:] for chunk in chunks[:2]) == blob[:2 * CHUNK_CHARS]


def test_the_overlap_after_a_long_last_sentence_starts_at_a_word():
    long_sentence = " ".join(["alpha"] * 60) + "."    # about 360 characters
    text = "Short opening. " * 30 + long_sentence + " " + "Closing words here. " * 60
    chunks = split(normalise(text))
    followers = [after for before, after in zip(chunks, chunks[1:])
                 if before.endswith(long_sentence)]
    assert followers, "the long sentence never ended a chunk"
    head = followers[0][:CHUNK_OVERLAP]
    assert head.startswith("alpha")
    assert len(followers[0].split(".")[0]) <= CHUNK_OVERLAP


def test_normalising_makes_line_endings_and_composed_characters_agree():
    assert normalise("café\r\nline\rend  ") == "café\nline\nend"
    assert split("   ") == []


def test_a_line_break_ends_a_sentence_without_punctuation():
    """Headings, list items and table rows have no full stop and are still units."""
    lines = "\n".join(f"- item number {i} in a list without punctuation" for i in range(80))
    for chunk in split(lines):
        assert chunk.startswith("- item") and chunk.endswith("punctuation")


# --- adding ---------------------------------------------------------------------


def test_a_document_is_stored_as_system_episodes_that_search_finds():
    m = mem()
    text = "\n\n".join(prose(12))
    doc = m.add_document(text, title="Handbook", filepath="policies/handbook.md",
                         meta={"team": "support"})
    assert doc.id.startswith("doc_")
    assert (doc.status, doc.error, doc.mime) == ("done", None, "text/plain")
    assert doc.content_hash == content_hash(normalise(text))
    chunks = m.store.document_chunks("default", doc.id)
    assert doc.chunks == len(chunks) > 1
    for position, chunk in enumerate(chunks):
        ep = m.store.get_episode(chunk.episode_id)
        assert chunk.position == position and chunk.hash == content_hash(chunk.text)
        assert (ep.role, ep.content, ep.meta) == ("system", chunk.text,
                                                  {DOCUMENT_META: doc.id})
        assert ep.scope == Scope("default", "alice")
        assert m.store.get_episode_embedding(ep.id) is not None
    hits = m.search(chunks[3].text[:80], include_episodes=True, k=5)
    assert chunks[3].episode_id in {getattr(h, "episode", None) and h.episode.id
                                    for h in hits}
    stored = m.get_document(doc.id)
    assert (stored.title, stored.filepath, stored.meta, stored.chunks) == (
        "Handbook", "policies/handbook.md", {"team": "support"}, doc.chunks)


def test_the_same_paragraph_in_two_documents_is_two_episodes():
    """`Episode.hash` mixes the document id in. Without it the second document's chunk
    would be the first document's episode, and deleting either would erase text the
    other still holds."""
    m = mem()
    a = m.add_document("A shared paragraph about refunds.")
    b = m.add_document("A shared paragraph about refunds.")
    ea = m.store.document_chunks("default", a.id)[0].episode_id
    eb = m.store.document_chunks("default", b.id)[0].episode_id
    assert ea != eb
    m.delete_document(a.id)
    assert m.store.get_episode(eb).content == "A shared paragraph about refunds."


def test_an_episode_outside_a_document_hashes_exactly_as_before():
    """No stored hash changes: an ordinary turn keeps the three-part digest."""
    ep = Episode(content="I live in Lisbon", scope=Scope("t", "u"))
    assert ep.hash == content_hash(Scope("t", "u").key(), "user", "I live in Lisbon")


def test_a_repeated_chunk_within_one_document_shares_one_episode():
    m = mem(retrieval_chunks=False)
    doc = m.add_document("Repeated.")
    assert doc.chunks == 1
    blob = ("Same sentence repeated for a very long time here. " * 40).strip()
    text = blob + "\n\n" + blob
    with_chunks = mem()
    stored = with_chunks.add_document(text)
    chunks = with_chunks.store.document_chunks("default", stored.id)
    by_text: dict[str, set[str]] = {}
    for c in chunks:
        by_text.setdefault(c.text, set()).add(c.episode_id)
    assert all(len(ids) == 1 for ids in by_text.values())
    assert len({c.episode_id for c in chunks}) < len(chunks)


def test_with_retrieval_chunks_off_a_document_is_one_chunk():
    m = mem(retrieval_chunks=False)
    doc = m.add_document("\n\n".join(prose(10)))
    assert doc.chunks == 1


def test_exactly_one_of_content_and_url_is_required():
    m = mem()
    with pytest.raises(TypeError, match="exactly one of content and url"):
        m.add_document()
    with pytest.raises(TypeError, match="exactly one of content and url"):
        m.add_document("text", url="https://example.com")
    with pytest.raises(TypeError, match="url must be a string"):
        m.add_document(url=b"https://example.com")  # type: ignore[arg-type]


@pytest.mark.parametrize("kw, message", [
    ({"custom_id": ""}, "custom_id must be a non-empty string"),
    ({"custom_id": "x" * 256}, "custom_id is 256 characters"),
    ({"filepath": " "}, "filepath must be a non-empty"),
    ({"meta": {"when": object()}}, "meta must be JSON-serialisable"),
    ({"meta": {1: "a"}}, "meta must be a mapping with string keys"),
])
def test_a_bad_argument_is_refused_before_anything_is_written(kw, message):
    m = mem()
    with pytest.raises(ValueError, match=message):
        m.add_document("Some text.", **kw)
    assert m.list_documents().items == []
    assert m.stats()["episodes"] == 0


def test_a_document_with_no_text_is_refused():
    with pytest.raises(ValueError, match="no text"):
        mem().add_document(" \r\n ")


def test_the_redactor_runs_before_the_text_is_chunked_or_hashed():
    """Every chunk and every digest is computed over redacted text, so no stored digest
    can confirm a value the redactor removed."""
    from memvara import PatternRedactor
    m = mem(redactor=PatternRedactor())
    doc = m.add_document("Write to alice@example.com for refunds.",
                         title="alice@example.com's notes")
    chunk = m.store.document_chunks("default", doc.id)[0]
    assert "alice@example.com" not in chunk.text
    assert "alice@example.com" not in (doc.title or "")
    assert doc.content_hash == content_hash(chunk.text)
    assert "alice@example.com" not in m.store.get_episode(chunk.episode_id).content
    again = m.add_document("x.", custom_id="c", title="bob@example.com")
    assert "bob@example.com" not in (again.title or "")
    updated = m.update_document(again.id, title="carol@example.com")
    assert "carol@example.com" not in (updated.title or "")


# --- re-ingest by custom_id -------------------------------------------------------


class Spy:
    """Counts the episodes handed to the write pipeline, then runs it."""

    def __init__(self, m: Memvara) -> None:
        self.calls: list[list[str]] = []
        self._real = m.writer.reextract
        m.writer.reextract = self  # type: ignore[method-assign]

    def __call__(self, episodes):
        self.calls.append([ep.id for ep in episodes])
        return self._real(episodes)


def test_reingesting_by_custom_id_keeps_every_unchanged_chunk_and_extracts_only_new_ones():
    m = mem()
    paragraphs = prose(30)
    first = m.add_document("\n\n".join(paragraphs), custom_id="handbook")
    old = {c.text: c.episode_id for c in m.store.document_chunks("default", first.id)}
    spy = Spy(m)

    edited = list(paragraphs)
    edited[0] = "A new opening sentence was added. " + edited[0]
    second = m.add_document("\n\n".join(edited), custom_id="handbook")

    assert second.id == first.id
    new = m.store.document_chunks("default", second.id)
    kept = [c for c in new if old.get(c.text) == c.episode_id]
    added = [c for c in new if c.text not in old]
    assert len(kept) >= len(new) - 2 and 1 <= len(added) <= 2
    # Only the new chunks were read for facts, once.
    assert spy.calls == [[c.episode_id for c in added]]
    # The chunks the edit replaced are gone, text and all.
    for text, episode_id in old.items():
        if episode_id not in {c.episode_id for c in new}:
            assert m.store.get_episode(episode_id) is None
    assert m.list_documents().items == [m.get_document("handbook")]


def test_a_chunk_that_only_moved_keeps_its_episode_and_changes_position():
    m = mem()
    paragraphs = prose(20)
    doc = m.add_document("\n\n".join(paragraphs), custom_id="c")
    before = m.store.document_chunks("default", doc.id)
    m.add_document("\n\n".join(prose(3, seed=99) + paragraphs), custom_id="c")
    after = m.store.document_chunks("default", doc.id)
    moved = [(b, a) for b in before for a in after
             if a.episode_id == b.episode_id and a.position != b.position]
    assert moved and all(a.text == b.text for b, a in moved)


def test_reingesting_identical_text_changes_nothing_and_extracts_nothing():
    m = mem()
    text = "\n\n".join(prose(8))
    doc = m.add_document(text, custom_id="c")
    before = m.store.document_chunks("default", doc.id)
    spy = Spy(m)
    m.add_document(text, custom_id="c")
    assert m.store.document_chunks("default", doc.id) == before
    assert spy.calls == []


def test_reingest_keeps_fields_the_caller_did_not_pass():
    m = mem()
    m.add_document("One.", custom_id="c", title="T", filepath="a/b.md", meta={"k": 1},
                   mime="text/markdown")
    doc = m.add_document("Two.", custom_id="c")
    assert (doc.title, doc.filepath, doc.meta, doc.mime) == (
        "T", "a/b.md", {"k": 1}, "text/markdown")
    assert m.add_document("Two.", custom_id="c", mime="text/plain").mime == "text/plain"
    doc = m.add_document("Three.", custom_id="c", title="U", filepath="c.md",
                         meta={"k": 2})
    assert (doc.title, doc.filepath, doc.meta) == ("U", "c.md", {"k": 2})


def test_a_claim_cited_only_by_a_replaced_chunk_is_retired_with_the_reason():
    m = mem(retrieval_chunks=False)
    doc = m.add_document("Refunds are paid within 14 days.", custom_id="refunds")
    episode = m.store.document_chunks("default", doc.id)[0].episode_id
    claim = m.remember("policy", "refund_window", "14 days",
                       sources=[episode]).added[0]
    m.add_document("Refunds are paid within 30 days.", custom_id="refunds")
    after = m.store.get_claim(claim.id)
    assert after.state == "retired" and after.sources == []
    assert ("retired", DOCUMENT_DELETED_REASON) in closure_reasons(after)


# --- delete ------------------------------------------------------------------------


def _with_claims(m: Memvara):
    doc = m.add_document("\n\n".join(prose(6)), custom_id="handbook")
    chunks = m.store.document_chunks("default", doc.id)
    turn = m.add("I work on the support team.").episode_ids[0]
    only = m.remember("policy", "refund_window", "14 days",
                      sources=[chunks[0].episode_id]).added[0]
    mixed = m.remember("policy", "support_hours", "9 to 5",
                       sources=[chunks[1].episode_id, turn]).added[0]
    return doc, chunks, turn, only, mixed


def test_deleting_a_document_erases_its_text_and_retires_what_only_it_supported():
    """The design's delete: text erased, no memory erased, sole-source memories retired
    with a reason, and a memory with another source keeps it."""
    m = mem()
    doc, chunks, turn, only, mixed = _with_claims(m)

    result = m.delete_document("handbook")

    assert result.deleted and result.id == doc.id and result.custom_id == "handbook"
    assert (result.chunks, result.episodes) == (len(chunks), len(chunks))
    assert (result.retired, result.unlinked) == ((only.id,), (mixed.id,))
    assert m.get_document(doc.id) is None
    assert m.store.document_chunks("default", doc.id) == []
    for chunk in chunks:
        assert m.store.get_episode(chunk.episode_id) is None
        assert m.store.get_episode_embedding(chunk.episode_id) is None
    fts = m.store._db.execute("SELECT COUNT(*) FROM episodes_fts").fetchone()[0]
    assert fts == 1                                    # only the conversation turn
    retired = m.store.get_claim(only.id)
    assert retired.state == "retired" and retired.sources == []
    assert ("retired", DOCUMENT_DELETED_REASON) in closure_reasons(retired)
    kept = m.store.get_claim(mixed.id)
    assert kept.state == "live" and kept.sources == [turn]
    assert [e.id for e in m.why(mixed.id).episodes] == [turn]


def test_a_claim_already_retired_is_not_retired_again():
    m = mem()
    doc, chunks, turn, only, mixed = _with_claims(m)
    m.delete(only.id, reason="wrong")
    result = m.delete_document(doc.id)
    assert result.retired == ()
    assert [r for _, r in closure_reasons(m.store.get_claim(only.id))] == ["wrong"]


def test_deleting_an_unknown_or_unseen_document_changes_nothing():
    m = mem()
    doc = m.add_document("Private.", custom_id="p")
    other = m.scope(user="bob")
    assert other.delete_document(doc.id).deleted is False
    assert other.delete_document("p").deleted is False
    missing = m.delete_document("doc_nothing")
    assert (missing.id, missing.deleted, missing.chunks) == ("doc_nothing", False, 0)
    assert m.get_document("p") is not None


def test_delete_documents_deletes_each_in_turn():
    m = mem()
    a = m.add_document("First.")
    m.add_document("Second.", custom_id="b")
    results = m.delete_documents([a.id, "missing", "b"])
    assert [r.deleted for r in results] == [True, False, True]
    assert m.list_documents().items == []


# --- status --------------------------------------------------------------------------


def test_a_failed_extraction_leaves_the_document_stored_and_searchable():
    m = mem()
    seen: list[str] = []

    def explode(episodes):
        seen.append(m.get_document("c").status)
        raise RuntimeError("provider unavailable")

    m.writer.reextract = explode  # type: ignore[method-assign]
    doc = m.add_document("Refunds are paid within 14 days.", custom_id="c")
    assert seen == ["extracting"]
    assert doc.status == "failed" and doc.error == "RuntimeError: provider unavailable"
    status = m.document_status("c")
    assert (status.id, status.status, status.error, status.chunks) == (
        doc.id, "failed", doc.error, 1)
    hits = m.search("refunds", include_episodes=True)
    assert any(getattr(h, "kind", "") == "episode" for h in hits)


def test_a_deferred_extraction_is_reported_as_failed_with_a_way_to_retry():
    from memvara.types import WriteReceipt
    m = mem()
    m.writer.reextract = lambda eps: WriteReceipt(deferred=True)  # type: ignore
    doc = m.add_document("Text.")
    assert doc.status == "failed" and "reextract()" in (doc.error or "")


def test_extract_false_skips_the_pipeline_and_is_done():
    m = mem()
    spy = Spy(m)
    doc = m.add_document("Text.", extract=False)
    assert doc.status == "done" and spy.calls == []


def test_status_of_a_document_this_scope_cannot_see_is_a_key_error():
    m = mem()
    doc = m.add_document("Private.")
    with pytest.raises(KeyError, match="no document"):
        m.scope(user="bob").document_status(doc.id)


# --- scope ----------------------------------------------------------------------------


def test_a_custom_id_is_unique_per_scope_and_found_from_a_narrower_one():
    m = mem()
    alice = m.add_document("Alice's.", custom_id="notes")
    bob = m.scope(user="bob").add_document("Bob's.", custom_id="notes")
    assert alice.id != bob.id
    session = m.scope(user="alice", session="s1")
    assert session.get_document("notes").id == alice.id
    assert m.scope(user="bob").get_document(alice.id) is None
    # A session-scoped document of the same name is a third document, and shadows the
    # user's for that session only.
    mine = session.add_document("Session's.", custom_id="notes")
    assert mine.id not in (alice.id, bob.id)
    assert session.get_document("notes").id == mine.id
    assert m.get_document("notes").id == alice.id


# --- listing ----------------------------------------------------------------------------


def test_listing_pages_newest_first_without_skipping_or_repeating():
    m = mem()
    ids = [m.add_document(f"Document {i}.").id for i in range(7)]
    seen, cursor = [], None
    while True:
        page = m.list_documents(limit=3, cursor=cursor)
        seen += [d.id for d in page.items]
        cursor = page.next_cursor
        if cursor is None:
            break
    assert seen == list(reversed(ids))


def test_listing_filters_in_the_store_by_path_prefix_and_status():
    m = mem()
    m.add_document("A.", filepath="docs/a.md")
    m.add_document("B.", filepath="docs/sub/b.md")
    m.add_document("C.", filepath="docsx/c.md")
    m.add_document("D.", filepath="100%_done/d.md")
    m.add_document("E.")
    assert {d.filepath for d in m.list_documents(filepath_prefix="docs/").items} == {
        "docs/a.md", "docs/sub/b.md"}
    # `%` and `_` match only themselves.
    assert [d.filepath for d in m.list_documents(filepath_prefix="100%_").items] == [
        "100%_done/d.md"]
    assert m.list_documents(filepath_prefix="1%").items == []
    assert len(m.list_documents(status="done").items) == 5
    assert m.list_documents(status="failed").items == []
    page = m.list_documents(filepath_prefix="docs", limit=1)
    assert len(page.items) == 1 and page.next_cursor is not None


@pytest.mark.parametrize("kw, message", [
    ({"limit": 0}, "limit must be between 1 and 1000"),
    ({"limit": 1001}, "limit must be between 1 and 1000"),
    ({"status": "finished"}, "is not one of"),
    ({"cursor": "garbage"}, "is not one list_documents returned"),
    ({"cursor": "not-a-date|doc_x"}, "is not one list_documents returned"),
])
def test_listing_refuses_arguments_it_cannot_honour(kw, message):
    with pytest.raises(ValueError, match=message):
        mem().list_documents(**kw)


def test_listing_shows_only_documents_this_scope_can_see():
    m = mem()
    m.add_document("Alice's.")
    m.scope(user="bob").add_document("Bob's.")
    assert len(m.list_documents().items) == 1
    assert len(m.scope(user="alice", session="s").list_documents().items) == 1


# --- update ------------------------------------------------------------------------------


def test_update_changes_fields_without_touching_chunks():
    m = mem()
    doc = m.add_document("Body.", custom_id="c", title="Old", meta={"a": 1})
    chunks = m.store.document_chunks("default", doc.id)
    updated = m.update_document("c", title="New", meta={"b": 2}, filepath="x/y.md")
    assert (updated.title, updated.meta, updated.filepath) == ("New", {"b": 2}, "x/y.md")
    assert m.store.document_chunks("default", doc.id) == chunks
    assert m.get_document(doc.id).title == "New"


def test_update_with_content_reingests_by_digest():
    m = mem()
    paragraphs = prose(12)
    doc = m.add_document("\n\n".join(paragraphs))
    before = {c.episode_id for c in m.store.document_chunks("default", doc.id)}
    edited = paragraphs[:-1] + ["A replaced final paragraph."]
    updated = m.update_document(doc.id, content="\n\n".join(edited))
    after = {c.episode_id for c in m.store.document_chunks("default", doc.id)}
    assert updated.status == "done" and len(before & after) >= len(after) - 2


def test_update_of_a_document_this_scope_cannot_see_is_a_key_error():
    with pytest.raises(KeyError):
        mem().update_document("doc_missing", title="x")


# --- the ingestion seam ------------------------------------------------------------------


@dataclass
class Extracted:
    text: str
    title: str | None
    mime: str


@pytest.fixture()
def ingest(monkeypatch):
    """A fake `memvara.ingest` that records what it was asked to extract."""
    calls: list[tuple] = []
    module = pytypes.ModuleType("memvara.ingest")

    def extract(content, *, url=None, mime=None):
        calls.append((content, url, mime))
        return Extracted("Extracted text from the source.", "Extracted title",
                         "application/pdf" if isinstance(content, bytes) else "text/html")

    module.extract = extract  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "memvara.ingest", module)
    return calls


def test_a_url_goes_through_the_ingestion_seam(ingest):
    m = mem()
    doc = m.add_document(url="https://example.com/refunds")
    assert ingest == [(None, "https://example.com/refunds", None)]
    assert (doc.source_uri, doc.title, doc.mime) == (
        "https://example.com/refunds", "Extracted title", "text/html")
    assert m.store.document_chunks("default", doc.id)[0].text == \
        "Extracted text from the source."


def test_bytes_and_markup_go_through_the_seam_and_plain_text_does_not(ingest):
    m = mem()
    m.add_document(b"%PDF-1.7", title="Mine")
    m.add_document("<p>Hi</p>", mime="text/html; charset=utf-8")
    m.add_document("# Heading", mime="text/markdown")
    assert [c[2] for c in ingest] == [None, "text/html; charset=utf-8"]
    titles = {d.title for d in m.list_documents().items}
    assert titles == {"Mine", "Extracted title", None}
    m.add_document("Plain.", custom_id="c")
    updated = m.update_document("c", content=b"%PDF")
    assert updated.title == "Extracted title" and updated.mime == "application/pdf"


def test_a_title_found_by_ingestion_is_redacted_too(ingest, monkeypatch):
    import memvara.ingest as fake  # the fixture's module

    monkeypatch.setattr(fake, "extract", lambda content, *, url=None, mime=None: Extracted(
        "Body text.", "Notes for alice@example.com", "text/html"))
    from memvara import PatternRedactor
    doc = mem(redactor=PatternRedactor()).add_document(url="https://example.com")
    assert doc.title and "alice@example.com" not in doc.title


def test_without_the_ingestion_package_a_url_or_bytes_is_refused_with_the_reason(
        monkeypatch):
    # `None` in `sys.modules` makes the import fail, whether or not the package exists
    # in this build.
    monkeypatch.setitem(sys.modules, "memvara.ingest", None)
    m = mem()
    with pytest.raises(NotImplementedError, match="the URL 'https://x.test'"):
        m.add_document(url="https://x.test")
    with pytest.raises(NotImplementedError, match="from bytes needs"):
        m.add_document(b"data")
    with pytest.raises(NotImplementedError, match="mime type 'application/pdf'"):
        m.add_document("text", mime="application/pdf")
    assert m.list_documents().items == []


# --- stores and erasure ----------------------------------------------------------------------


def test_a_store_without_the_document_methods_is_refused_by_name():
    class NoDocuments(SQLiteStore):
        put_document = None  # type: ignore[assignment]

    m = Memvara(store=NoDocuments(":memory:"), llm=NullLLM(),
                embedder=HashingEmbedder(dim=64))
    with pytest.raises(NotImplementedError, match="NoDocuments does not implement "
                                                  "put_document"):
        m.add_document("Text.")


def test_erasing_a_chunk_episode_erases_the_chunk_row_that_repeats_it():
    """`document_chunks.text` is a copy of the episode text, so any path that erases the
    episode must take the copy, or the text survives an erasure that reported success."""
    m = mem()
    doc = m.add_document("A paragraph worth a fact.")
    episode = m.store.document_chunks("default", doc.id)[0].episode_id
    claim = m.remember("user", "fact", "x", sources=[episode]).added[0]
    assert m.erase(claim.id, sources=True)
    assert m.store.get_episode(episode) is None
    assert m.store.document_chunks("default", doc.id) == []


def test_purge_erases_documents_and_their_chunks():
    m = mem()
    m.add_document("Alice's text.", title="secret title")
    kept = m.scope(user="bob").add_document("Bob's text.")
    m.purge(user="alice")
    rows = m.store._db.execute("SELECT title FROM documents").fetchall()
    assert [r[0] for r in rows] == [None]
    chunks = m.store._db.execute("SELECT document_id FROM document_chunks").fetchall()
    assert [r[0] for r in chunks] == [kept.id]


def test_a_version_13_file_gains_the_document_tables_and_keeps_its_rows(tmp_path):
    """A real upgrade: a file stamped 13 without the tables, opened by this build."""
    path = str(tmp_path / "v13.db")
    m = mem(path=path)
    claim = m.remember("user", "lives_in", "Lisbon").added[0]
    m.close()
    raw = sqlite3.connect(path)
    raw.execute("DROP TABLE document_chunks")
    raw.execute("DROP TABLE documents")
    raw.execute("PRAGMA user_version = 13")
    raw.commit()
    raw.close()

    upgraded = mem(path=path)
    try:
        assert int(upgraded.store._db.execute("PRAGMA user_version").fetchone()[0]) == \
            SCHEMA_VERSION == 14
        names = {r[0] for r in upgraded.store._db.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'index')")}
        assert {"documents", "document_chunks", "doc_custom", "doc_created",
                "dchunk_episode"} <= names
        assert upgraded.store.get_claim(claim.id).object == "Lisbon"
        upgraded.store._migrate_to_v14()               # idempotent
        assert upgraded.add_document("After the upgrade.").status == "done"
    finally:
        upgraded.close()


def test_the_table_refuses_a_status_that_is_not_one_of_the_four():
    store = SQLiteStore(":memory:")
    from memvara.types import Document
    with pytest.raises(sqlite3.IntegrityError):
        store.put_document(Document(status="finished"))  # type: ignore[arg-type]
    store.close()


def test_minimum_chunk_size_is_below_the_limit():
    """The two constants the boundary rule rests on must leave room for a cut point."""
    assert 0 < _MIN_CHARS < CHUNK_CHARS and CHUNK_OVERLAP < _MIN_CHARS


def test_a_document_reads_back_with_a_short_repr():
    m = mem()
    doc = m.add_document("Text.", custom_id="handbook")
    assert repr(doc).startswith(f"<Document {doc.id} default/alice/")
    assert "chunks=1 'handbook'>" in repr(doc)


# --- the views: scoped, async, async scoped ---------------------------------------------


def test_every_view_reaches_every_document_method():
    import asyncio
    from memvara import AsyncMemvara

    m = mem()
    view = m.scope(user="alice")
    doc = view.add_document("Scoped text.", custom_id="s", extract=False)
    assert view.get_document("s").id == doc.id
    assert view.list_documents().items[0].id == doc.id
    assert view.update_document("s", title="T").title == "T"
    assert view.document_status("s").status == "done"
    assert [r.deleted for r in view.delete_documents(["s"])] == [True]
    assert view.delete_document("s").deleted is False

    async def drive(target):
        doc = await target.add_document("Async text.", custom_id="a", extract=False)
        assert (await target.get_document("a")).id == doc.id
        assert (await target.list_documents()).items[0].id == doc.id
        assert (await target.update_document("a", title="T")).title == "T"
        assert (await target.document_status("a")).status == "done"
        assert [r.deleted for r in await target.delete_documents(["a"])] == [True]
        assert (await target.delete_document("a")).deleted is False

    amem = AsyncMemvara(m)
    asyncio.run(drive(amem))
    asyncio.run(drive(amem.scope(user="alice")))


# --- the MCP tools --------------------------------------------------------------------


def _server(**kw):
    from memvara.server import MemvaraMCPServer
    return MemvaraMCPServer(mem(), user="alice", **kw)


def test_the_tools_store_list_show_and_delete_a_document():
    from test_server import text
    srv = _server()
    body = text(srv, "memory_add_document", {
        "content": "Refunds are paid within 14 days.", "custom_id": "refunds",
        "title": "Refunds [admin]", "filepath": "policies/refunds.md",
        "metadata": {"team": "support"}})
    assert body.startswith("Stored.\ndocument doc_") and "done, 1 chunk(s)" in body
    assert "title: Refunds ［admin］" in body, "caller text is flattened like a claim"
    assert "metadata: team=support" in body and "include_episodes true" in body

    listing = text(srv, "memory_list_documents", {"filepath_prefix": "policies/"})
    assert listing.startswith("1 document(s), newest first:")
    assert "Refunds ［admin］ — policies/refunds.md" in listing
    assert text(srv, "memory_list_documents", {"status": "failed"}) == (
        "No documents are stored here that match those filters.")

    shown = text(srv, "memory_get_document", {"id": "refunds"})
    assert "custom_id: refunds" in shown and "filepath: policies/refunds.md" in shown
    assert text(srv, "memory_get_document", {"id": "nothing"}).startswith("No document")

    deleted = text(srv, "memory_delete_document", {"id": "refunds"})
    assert "its text is erased (1 chunk(s)). No memory was erased." in deleted
    assert text(srv, "memory_list_documents") == "No documents are stored here."
    assert text(srv, "memory_delete_document", {"id": "refunds"}).startswith(
        "Nothing deleted")
    srv.close()


def test_deleting_a_document_through_the_tool_retires_and_never_erases_a_memory():
    """The behaviour `test_no_tool_can_erase_a_memory` relies on for the one tool whose
    name says delete."""
    from test_server import text
    srv = _server()
    memory = srv._ctx.memory
    doc = memory.add_document("A turn about refunds.", custom_id="r")
    turn = memory.add("I handle refunds.").episode_ids[0]
    chunk = memory.memvara.store.document_chunks("default", doc.id)[0].episode_id
    only = memory.remember("policy", "refund_window", "14 days",
                           sources=[chunk]).added[0]
    mixed = memory.remember("policy", "refund_owner", "support",
                            sources=[chunk, turn]).added[0]
    body = text(srv, "memory_delete_document", {"id": "r"})
    assert "Retired 1 memory(ies) whose only source was this document" in body
    assert only.id in body and mixed.id in body and "keep it" in body
    assert memory.memvara.store.get_claim(only.id).state == "retired"
    assert memory.memvara.store.get_claim(mixed.id).state == "live"
    why = text(srv, "memory_why", {"claim_id": only.id})
    assert "source document deleted" in why
    srv.close()


def test_the_add_tool_refuses_what_it_cannot_store_with_the_reason(monkeypatch):
    from test_server import call
    monkeypatch.setitem(sys.modules, "memvara.ingest", None)
    srv = _server()
    body, is_error = call(srv, "memory_add_document", {})
    assert is_error and "exactly one of content" in body
    body, is_error = call(srv, "memory_add_document", {"url": "https://x.test"})
    assert is_error and body.startswith("Nothing stored: adding a document from the URL")
    body, is_error = call(srv, "memory_add_document", {"content": "  "})
    assert is_error and "no text" in body
    body, is_error = call(srv, "memory_list_documents", {"cursor": "junk"})
    assert is_error and "is not one list_documents returned" in body
    srv.close()


def test_a_failed_document_says_it_is_still_stored():
    from test_server import text
    srv = _server()

    def explode(episodes):
        raise RuntimeError("provider unavailable")

    srv._ctx.memory.memvara.writer.reextract = explode
    body = text(srv, "memory_add_document", {"content": "Text."})
    assert "error: RuntimeError: provider unavailable" in body
    assert "It is stored and its chunks are searchable" in body
    srv.close()


def test_a_listing_with_more_pages_ends_with_the_cursor_to_pass_back():
    from test_server import text
    srv = _server()
    for i in range(3):
        srv._ctx.memory.add_document(f"Document {i}.")
    first = text(srv, "memory_list_documents", {"limit": 2})
    cursor = first.rsplit("cursor ", 1)[1].rstrip(".").strip("'")
    second = text(srv, "memory_list_documents", {"limit": 2, "cursor": cursor})
    assert second.startswith("1 document(s)") and "(untitled)" in second
    srv.close()


# --- the switches ------------------------------------------------------------------------


def test_switching_documents_off_hides_the_four_tools():
    from test_feature_switches_p1a2 import listed
    names = set(listed(_server(features_off={"documents"})))
    assert not {"memory_add_document", "memory_get_document", "memory_list_documents",
                "memory_delete_document"} & names
    assert "memory_recall" in names


def test_switching_retrieval_chunks_off_stores_a_document_whole():
    from memvara.server.config import ServerConfig, build_memvara
    on = build_memvara(ServerConfig.from_env({"MEMVARA_DB": ":memory:",
                                              "MEMVARA_EMBEDDER": "hashing"}))
    off = build_memvara(ServerConfig.from_env({"MEMVARA_DB": ":memory:",
                                               "MEMVARA_EMBEDDER": "hashing",
                                               "MEMVARA_FEATURE_RETRIEVAL_CHUNKS": "0"}))
    text = "\n\n".join(prose(10))
    assert on.retrieval_chunks and on.add_document(text).chunks > 1
    assert not off.retrieval_chunks and off.add_document(text).chunks == 1
    on.close()
    off.close()


def test_turning_retrieval_chunks_off_against_a_hosted_deployment_is_refused():
    with pytest.raises(TypeError, match="retrieval_chunks cannot be combined"):
        Memvara(api_key="k", base_url="https://example.test", retrieval_chunks=False)
    client = Memvara(api_key="k", base_url="https://example.test", retrieval_chunks=True)
    client.close()


def test_the_remote_store_names_the_route_to_use_instead():
    from memvara.store.remote import RemoteStore
    from memvara.types import Document
    store = RemoteStore(base_url="https://example.invalid", api_key="k")
    calls = [lambda: store.put_document(Document()),
             lambda: store.get_document("t", "doc_1"),
             lambda: store.find_document(Scope(), "c"),
             lambda: store.list_documents([Scope()]),
             lambda: store.document_chunks("t", "doc_1"),
             lambda: store.put_document_chunks("t", "doc_1", []),
             lambda: store.delete_document("t", "doc_1")]
    for c in calls:
        with pytest.raises(NotImplementedError, match="RemoteMemvara.add_document"):
            c()
    store.close()
