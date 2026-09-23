"""`memvara.ingest.extract`: every kind of content, and every way it can be refused.

No test here touches the network or needs a key. URLs go through a fake fetcher (the
fetcher's own rules are in `test_ingest_url.py`), images and audio go through a fake
`Multimodal` backend that counts its calls, and PDFs are read either by a fake `pypdf`
module or, when the `ingest` extra is installed, by the real one on a PDF built here.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from memvara.ingest import (Extracted, Fetched, IngestError, MediaUnsupported, _mime,
                            extract)
from memvara.ingest.html_text import html_to_text
from memvara.ingest.plain import decode
from memvara.llm import Multimodal, NullLLM


class FakeMedia:
    """A `Multimodal` backend that records what it was asked and answers with fixed text."""

    def __init__(self, answer: str = "a transcript") -> None:
        self.answer = answer
        self.images: list[tuple[bytes, str]] = []
        self.recordings: list[tuple[bytes, str]] = []

    def describe_image(self, data: bytes, mime: str) -> str:
        self.images.append((data, mime))
        return self.answer

    def transcribe(self, data: bytes, mime: str) -> str:
        self.recordings.append((data, mime))
        return self.answer


class FakeFetcher:
    def __init__(self, body: bytes, content_type: str | None) -> None:
        self.body = body
        self.content_type = content_type
        self.urls: list[str] = []

    def fetch(self, url: str) -> Fetched:
        self.urls.append(url)
        return Fetched(url=url, content_type=self.content_type, body=self.body)


def make_pdf(pages: list[str], title: str | None = None) -> bytes:
    """A small, valid PDF with one line of Helvetica text per page."""
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>",
               ("<< /Type /Pages /Kids [%s] /Count %d >>" % (
                   " ".join(f"{4 + 2 * i} 0 R" for i in range(len(pages))),
                   len(pages))).encode(),
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    for i, text in enumerate(pages):
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode() if text else b""
        objects.append((
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {5 + 2 * i} 0 R >>"
        ).encode())
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream
                       + b"\nendstream")
    info = None
    if title:
        objects.append(f"<< /Title ({title}) >>".encode())
        info = len(objects)
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    trailer = f"<< /Size {len(objects) + 1} /Root 1 0 R"
    trailer += f" /Info {info} 0 R >>" if info else " >>"
    out += b"trailer\n" + trailer.encode() + b"\nstartxref\n%d\n%%%%EOF\n" % xref
    return bytes(out)


def fake_pypdf(monkeypatch, *, pages=("one",), title=None, encrypted=False,
               decrypts=True, error=None):
    """Install a stand-in `pypdf` module and return the list of readers it made."""
    made = []

    class Reader:
        def __init__(self, stream):
            if error is not None:
                raise error
            self.is_encrypted = encrypted
            self.pages = [SimpleNamespace(extract_text=lambda text=text: text)
                          for text in pages]
            self.metadata = SimpleNamespace(title=title) if title is not False else None
            self.decrypted_with = None
            made.append(self)

        def decrypt(self, password):
            self.decrypted_with = password
            return 1 if decrypts else 0

    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=Reader))
    return made


# -- choosing exactly one source -----------------------------------------------------


class TestArguments:
    def test_neither_content_nor_url_is_refused(self):
        with pytest.raises(IngestError) as caught:
            extract()
        assert caught.value.code == "bad_input"

    def test_both_content_and_url_is_refused(self):
        fetcher = FakeFetcher(b"x", "text/plain")
        with pytest.raises(IngestError) as caught:
            extract("x", url="https://example.com", fetcher=fetcher)
        assert caught.value.code == "bad_input"
        assert fetcher.urls == []

    def test_a_binary_type_given_as_a_str_is_refused(self):
        with pytest.raises(IngestError) as caught:
            extract("%PDF-1.4", mime="application/pdf")
        assert caught.value.code == "bad_input"

    def test_the_error_carries_a_code_and_a_reason(self):
        error = MediaUnsupported("no reader")
        assert isinstance(error, IngestError) and isinstance(error, ValueError)
        assert (error.code, error.reason) == ("media_unsupported", "no reader")


# -- text ---------------------------------------------------------------------------


class TestText:
    def test_a_str_with_no_type_is_plain_text(self):
        assert extract("  hello  ") == Extracted("hello", None, "text/plain")

    def test_utf8_bytes_with_no_type_are_plain_text(self):
        assert extract("naïve".encode()).text == "naïve"

    def test_the_named_charset_is_used(self):
        got = extract("café".encode("latin-1"), mime="text/plain; charset=ISO-8859-1")
        assert got.text == "café" and got.mime == "text/plain"

    def test_an_unknown_charset_falls_back_to_utf8(self):
        assert decode(b"ok", "no-such-charset", strict=True) == "ok"

    def test_a_declared_text_type_keeps_bad_bytes_as_replacement_characters(self):
        assert extract(b"ok \xff", mime="text/markdown").text == "ok �"

    def test_json_is_read_as_text(self):
        assert extract(b'{"a": 1}', mime="application/json").mime == "application/json"

    def test_bytes_that_are_no_known_type_and_not_utf8_are_refused(self):
        with pytest.raises(MediaUnsupported, match="pass mime="):
            extract(b"\x00\xff\xfe\x80binary")

    def test_octet_stream_is_treated_as_no_type(self):
        assert extract(b"plain words", mime="application/octet-stream").mime == "text/plain"

    def test_empty_text_is_refused_as_no_text(self):
        with pytest.raises(IngestError) as caught:
            extract("   ")
        assert caught.value.code == "no_text"
        assert caught.value.reason == "the content is empty"

    def test_an_unreadable_type_is_refused_naming_what_is_read(self):
        with pytest.raises(MediaUnsupported, match="application/zip is not a type"):
            extract(b"PK\x03\x04", mime="application/zip")


# -- media types --------------------------------------------------------------------


class TestMime:
    def test_parse_handles_none_parameters_and_aliases(self):
        assert _mime.parse(None) == (None, None)
        assert _mime.parse("audio/x-wav; rate=8000") == ("audio/wav", None)
        assert _mime.parse("text/plain; charset=") == ("text/plain", None)

    @pytest.mark.parametrize("head, mime", [
        (b"\x89PNG\r\n\x1a\nrest", "image/png"),
        (b"\xff\xd8\xff\xe0rest", "image/jpeg"),
        (b"GIF87a", "image/gif"),
        (b"GIF89a", "image/gif"),
        (b"<HTML><body>", "text/html"),
    ])
    def test_sniff_recognises_file_signatures(self, head, mime):
        assert _mime.sniff(head) == mime

    def test_riff_that_is_not_webp_is_not_an_image(self):
        assert _mime.sniff(b"RIFF\x00\x00\x00\x00WAVEfmt ") is None


# -- HTML ---------------------------------------------------------------------------


class TestHtml:
    def test_a_str_that_starts_like_a_page_is_read_as_html(self):
        got = extract("<!doctype html><title>T</title><p>Body</p>")
        assert (got.text, got.title, got.mime) == ("Body", "T", "text/html")

    def test_bytes_use_the_declared_charset(self):
        page = "<p>café</p>".encode("latin-1")
        assert extract(page, mime="text/html; charset=latin-1").text == "café"

    def test_scripts_styles_and_navigation_are_dropped(self):
        text, title = html_to_text(
            "<body><script>var x = 1;</script><style>p {}</style>"
            "<nav><ul><li>Home</li></ul></nav><header>Site</header>"
            "<p>Kept</p><aside>ad</aside><form><button>Go</button></form>"
            "<footer>(c)</footer></body>")
        assert text == "Kept" and title is None

    def test_without_main_the_whole_body_is_kept(self):
        text, _ = html_to_text("<div>One<br>Two</div><p>Three <b>bold</b></p>")
        assert text == "One\nTwo\n\nThree bold"

    def test_main_and_article_are_preferred_over_the_rest(self):
        text, _ = html_to_text(
            "<div>Menu</div><article><p>Story</p><img src=x><hr></article>"
            "<div>Related</div>")
        assert text == "Story"

    def test_an_empty_main_falls_back_to_the_whole_body(self):
        text, _ = html_to_text("<main> </main><p>Body text</p>")
        assert text == "Body text"

    def test_only_the_first_title_counts(self):
        _, title = html_to_text(
            "<head><title>Page</title></head><body><svg><title>Icon</title></svg>"
            "<p>x</p></body>")
        assert title == "Page"

    def test_stray_end_tags_do_not_unbalance_the_counters(self):
        text, _ = html_to_text("</nav></main><p>Still here</p>")
        assert text == "Still here"

    def test_a_page_with_no_readable_text_is_refused_as_no_text(self):
        with pytest.raises(IngestError) as caught:
            extract("<html><script>render()</script></html>", mime="text/html")
        assert caught.value.code == "no_text" and "JavaScript" in caught.value.reason


# -- PDF ----------------------------------------------------------------------------


class TestPdf:
    def test_pages_are_kept_in_order_and_empty_pages_are_skipped_in_the_text(
            self, monkeypatch):
        fake_pypdf(monkeypatch, pages=(" one ", None, "three"), title=" Handbook ")
        got = extract(b"%PDF-1.7 fake")
        assert got.mime == "application/pdf"
        assert got.pages == ("one", "", "three")
        assert got.text == "one\n\nthree"
        assert got.title == "Handbook"

    def test_a_pdf_without_metadata_has_no_title(self, monkeypatch):
        fake_pypdf(monkeypatch, title=False)
        assert extract(b"%PDF-", mime="application/pdf").title is None

    def test_a_pdf_with_an_empty_password_is_opened(self, monkeypatch):
        made = fake_pypdf(monkeypatch, encrypted=True, decrypts=True)
        assert extract(b"%PDF-").text == "one"
        assert made[0].decrypted_with == ""

    def test_a_password_protected_pdf_is_refused_as_unreadable(self, monkeypatch):
        fake_pypdf(monkeypatch, encrypted=True, decrypts=False)
        with pytest.raises(IngestError) as caught:
            extract(b"%PDF-")
        assert caught.value.code == "unreadable" and "password" in caught.value.reason

    def test_a_damaged_pdf_is_refused_as_unreadable(self, monkeypatch):
        fake_pypdf(monkeypatch, error=KeyError("/Root"))
        with pytest.raises(IngestError) as caught:
            extract(b"%PDF-")
        assert caught.value.code == "unreadable"

    def test_a_pdf_of_scanned_pages_is_refused_as_no_text(self, monkeypatch):
        fake_pypdf(monkeypatch, pages=("", ""))
        with pytest.raises(IngestError) as caught:
            extract(b"%PDF-")
        assert caught.value.code == "no_text" and "OCR" in caught.value.reason

    def test_without_the_ingest_extra_a_pdf_is_refused_naming_the_extra(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pypdf", None)
        with pytest.raises(MediaUnsupported, match=r"memvara\[ingest\]"):
            extract(b"%PDF-1.4")

    def test_the_real_pypdf_reads_a_real_pdf(self):
        pytest.importorskip("pypdf")
        got = extract(make_pdf(["Hello page one", "", "Page three"], title="Handbook"))
        assert got.pages == ("Hello page one", "", "Page three")
        assert got.text == "Hello page one\n\nPage three"
        assert got.title == "Handbook"

    def test_the_real_pypdf_refuses_garbage_as_unreadable(self):
        pytest.importorskip("pypdf")
        with pytest.raises(IngestError) as caught:
            extract(b"%PDF-1.4\nthis is not a pdf", mime="application/pdf")
        assert caught.value.code == "unreadable"


# -- images, audio and video ----------------------------------------------------------


class TestMedia:
    def test_an_image_is_described_once_by_the_backend(self):
        backend = FakeMedia("A bar chart of revenue by quarter.")
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
        got = extract(png, llm=backend)
        assert got == Extracted("A bar chart of revenue by quarter.", None, "image/png")
        assert backend.images == [(png, "image/png")] and backend.recordings == []

    @pytest.mark.parametrize("mime", ["audio/mpeg", "video/mp4"])
    def test_audio_and_video_are_transcribed_once_by_the_backend(self, mime):
        backend = FakeMedia("We agreed to ship on Friday.")
        got = extract(b"\x00\x01", mime=mime, llm=backend)
        assert got.text == "We agreed to ship on Friday." and got.mime == mime
        assert backend.recordings == [(b"\x00\x01", mime)] and backend.images == []

    def test_the_fake_is_a_multimodal_backend_and_nullllm_is_not(self):
        assert isinstance(FakeMedia(), Multimodal)
        assert not isinstance(NullLLM(), Multimodal)

    def test_media_with_no_backend_is_refused_as_media_unsupported(self):
        with pytest.raises(MediaUnsupported, match="no model backend was given"):
            extract(b"\xff\xd8\xff\xe0", mime="image/jpeg")

    def test_media_with_a_backend_that_cannot_read_it_is_refused(self):
        with pytest.raises(MediaUnsupported, match="NullLLM cannot read media"):
            extract(b"\x00", mime="audio/wav", llm=NullLLM())

    def test_a_backend_that_returns_nothing_is_refused_as_no_text(self):
        with pytest.raises(IngestError) as caught:
            extract(b"\x00", mime="audio/wav", llm=FakeMedia(""))
        assert caught.value.code == "no_text" and "audio" in caught.value.reason

    def test_with_media_switched_off_the_backend_is_never_called(self):
        backend = FakeMedia()
        with pytest.raises(IngestError) as caught:
            extract(b"GIF89a", llm=backend, allow_media=False)
        assert caught.value.code == "feature_off"
        assert "ingest_media" in caught.value.reason
        assert backend.images == []

    def test_switching_media_off_does_not_stop_text(self):
        assert extract("words", allow_media=False).text == "words"


# -- URLs ---------------------------------------------------------------------------


class TestUrl:
    def test_the_fetched_body_is_read_as_the_servers_type(self):
        fetcher = FakeFetcher(b"<title>Doc</title><p>Body</p>", "text/html; charset=utf-8")
        got = extract(url="https://example.com/doc", fetcher=fetcher)
        assert (got.text, got.title, got.mime) == ("Body", "Doc", "text/html")
        assert fetcher.urls == ["https://example.com/doc"]

    def test_the_callers_type_overrides_the_servers(self):
        fetcher = FakeFetcher(b"<p>raw</p>", "text/html")
        got = extract(url="https://example.com/a", mime="text/plain", fetcher=fetcher)
        assert got.text == "<p>raw</p>" and got.mime == "text/plain"

    def test_a_response_with_no_type_is_sniffed(self):
        fetcher = FakeFetcher(b"just text", None)
        assert extract(url="https://example.com/a", fetcher=fetcher).mime == "text/plain"

    def test_with_urls_switched_off_nothing_is_fetched(self):
        fetcher = FakeFetcher(b"x", "text/plain")
        with pytest.raises(IngestError) as caught:
            extract(url="https://example.com", fetcher=fetcher, allow_urls=False)
        assert caught.value.code == "feature_off" and "ingest_urls" in caught.value.reason
        assert fetcher.urls == []

    def test_the_safe_fetcher_is_the_default(self, monkeypatch):
        import memvara.ingest as ingest

        built = []

        class Stand(FakeFetcher):
            def __init__(self):
                super().__init__(b"from the default", "text/plain")
                built.append(self)

        monkeypatch.setattr(ingest, "SafeFetcher", Stand)
        assert extract(url="https://example.com").text == "from the default"
        assert len(built) == 1 and built[0].urls == ["https://example.com"]
