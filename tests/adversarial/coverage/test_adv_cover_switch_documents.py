"""The three feature switches that change how the MCP server's store takes in a document:
`ingest_urls`, `ingest_media` and `retrieval_chunks`.

Each test builds the server's store from environment variables with the `served` fixture
(conftest.py beside this file) and adds a document to it. The switches are described in
the comment above `FEATURE_DEFAULTS` in memvara/server/config.py and in the
`MEMVARA_FEATURE_<NAME>` row of docs/DEPLOY.md. A URL is answered by a fetcher that
records what it was asked for, and an image is described by a scripted stand-in for the
model.
"""

from __future__ import annotations

import pytest

from memvara.ingest import Fetched, IngestError

from ..model_faults.scripted import ScriptedModel
from .conftest import Serve

#: The page the recording fetcher answers every URL with.
PAGE = (b"<html><title>Refunds</title>"
        b"<p>Refunds are paid within 14 days.</p></html>")

#: The first bytes of a PNG, which is how `memvara.ingest` recognises an image.
PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 16

#: What the stand-in model says the image shows.
DESCRIPTION = "A whiteboard listing the three refund rules."


class RecordingFetcher:
    """Stands in for `SafeFetcher`: answers every URL with PAGE and records the URL."""

    def __init__(self) -> None:
        self.urls: list[str] = []

    def fetch(self, url: str) -> Fetched:
        self.urls.append(url)
        return Fetched(url, "text/html; charset=utf-8", PAGE)


class SeeingModel(ScriptedModel):
    """A scripted model that can also read images, as `memvara.llm.Multimodal` asks, and
    records every image it was shown."""

    def __init__(self) -> None:
        super().__init__()
        self.images: list[tuple[bytes, str]] = []

    def describe_image(self, data: bytes, mime: str) -> str:
        self.images.append((data, mime))
        return DESCRIPTION

    def transcribe(self, data: bytes, mime: str) -> str:
        raise AssertionError("no audio or video is added in these tests")


@pytest.mark.covers("switch:ingest_urls")
def test_ingest_urls_switched_off_refuses_a_url_before_fetching_it(served: Serve) -> None:
    """A document given as a URL is fetched and stored while `ingest_urls` is on, which is
    its default. `MEMVARA_FEATURE_INGEST_URLS=0` refuses such a document with the code
    `feature_off`, and nothing is fetched or stored. docs/DEPLOY.md and
    `memvara.ingest.extract` make this promise."""
    on = served({})
    on.url_fetcher = fetched = RecordingFetcher()
    doc = on.add_document(url="https://example.test/refunds", extract=False)
    assert fetched.urls == ["https://example.test/refunds"]
    assert doc.title == "Refunds" and doc.source_uri == "https://example.test/refunds"
    assert [d.id for d in on.list_documents().items] == [doc.id]

    off = served({"MEMVARA_FEATURE_INGEST_URLS": "0"})
    off.url_fetcher = refused = RecordingFetcher()
    with pytest.raises(IngestError) as caught:
        off.add_document(url="https://example.test/refunds", extract=False)
    assert caught.value.code == "feature_off"
    assert "ingest_urls" in caught.value.reason
    assert refused.urls == [], "the URL was refused before anything fetched it"
    assert off.list_documents().items == []
    assert off.add_document("Refunds are paid within 14 days.", extract=False).chunks == 1, (
        "text passed as content is still stored")


@pytest.mark.covers("switch:ingest_media")
def test_ingest_media_switched_off_refuses_an_image_before_the_model_sees_it(
        served: Serve) -> None:
    """An image added as a document is described by the model, and the description is
    stored as the document's text, while `ingest_media` is on, which is its default.
    `MEMVARA_FEATURE_INGEST_MEDIA=0` refuses images, audio and video with the code
    `feature_off`, and the model is never shown the image. docs/DEPLOY.md and
    `memvara.ingest.extract` make this promise."""
    seeing = SeeingModel()
    on = served({}, seeing)
    doc = on.add_document(PNG, extract=False)
    assert seeing.images == [(PNG, "image/png")]
    assert doc.mime == "image/png"
    assert [c.text for c in on.store.document_chunks("default", doc.id)] == [DESCRIPTION]

    blind = SeeingModel()
    off = served({"MEMVARA_FEATURE_INGEST_MEDIA": "0"}, blind)
    with pytest.raises(IngestError) as caught:
        off.add_document(PNG, extract=False)
    assert caught.value.code == "feature_off"
    assert "ingest_media" in caught.value.reason
    assert blind.images == [], "the image was refused before the model saw it"
    assert off.list_documents().items == []


@pytest.mark.covers("switch:retrieval_chunks")
def test_retrieval_chunks_switched_off_stores_a_long_document_as_one_chunk(
        served: Serve) -> None:
    """A document is split into chunks of about 1,000 characters while `retrieval_chunks`
    is on, which is its default. `MEMVARA_FEATURE_RETRIEVAL_CHUNKS=0` stores each document
    as one chunk instead. docs/DEPLOY.md and the comment above `FEATURE_DEFAULTS` make
    this promise. The document here is about 4,000 characters of prose."""
    text = "\n\n".join(
        f"Section {n} explains how the refund queue is drained, who is paged when it "
        "backs up, and how a stuck payment is retried by hand after the third failure. "
        * 2 for n in range(12))
    assert len(text) > 3000

    on = served({})
    chunked = on.add_document(text, extract=False)
    assert chunked.chunks > 1
    assert len(on.store.document_chunks("default", chunked.id)) == chunked.chunks

    off = served({"MEMVARA_FEATURE_RETRIEVAL_CHUNKS": "0"})
    whole = off.add_document(text, extract=False)
    assert whole.chunks == 1
    (chunk,) = off.store.document_chunks("default", whole.id)
    assert chunk.text.split() == text.split(), "the one chunk holds the whole document"
