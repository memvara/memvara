"""Turning a document into text: plain text, HTML, PDF, images, audio, video and URLs.

`extract` is the one entry point. It takes the content itself or a URL, works out what
kind of content it is, and returns an `Extracted` with the text, the title when the
content has one, the media type, and the text of each page for a PDF. The document store
calls it before chunking a new document.

Each kind of content has its own module:

- `plain` decodes text;
- `html_text` keeps the readable text of a web page;
- `pdf` reads each page's text layer through `pypdf`, from the `ingest` extra;
- `media` hands images, audio and video to a model backend that implements
  `memvara.llm.Multimodal`;
- `url` fetches a URL under rules that stop it reaching a private network.

The fetcher in `url` is the only network code in this package, and it runs only when the
caller passes `url=`. Every failure raises `IngestError`, whose `code` names the kind of
failure; `errors` lists the codes.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import _mime
from .errors import IngestError, MediaUnsupported
from .html_text import html_to_text
from .media import media_to_text
from .pdf import pdf_pages
from .plain import decode
from .url import Fetched, Fetcher, SafeFetcher

__all__ = ["Extracted", "Fetched", "Fetcher", "IngestError", "MediaUnsupported",
           "SafeFetcher", "extract"]

#: Media types read as plain text, beside every `text/*` type that is not HTML.
_TEXT_TYPES = frozenset({"application/json", "application/xml", "application/yaml",
                         "application/x-yaml", "application/x-ndjson"})

_HTML_TYPES = frozenset({"text/html", "application/xhtml+xml"})

_MEDIA_PREFIXES = ("image/", "audio/", "video/")


@dataclass(frozen=True, slots=True)
class Extracted:
    """The text of one document, ready to be chunked.

    `mime` is the media type the content was read as, without parameters. `title` is the
    HTML `<title>` or the PDF's metadata title, and `None` for content that has none.
    `pages` is set for a PDF only: the text of each page in order, so that page `n` is
    `pages[n - 1]`, and a page with no text is an empty string. `text` is then the
    non-empty pages joined by blank lines.
    """

    text: str
    title: str | None
    mime: str
    pages: tuple[str, ...] | None = None


def extract(content: str | bytes | None = None, *, url: str | None = None,
            mime: str | None = None, llm: object | None = None,
            fetcher: Fetcher | None = None, allow_urls: bool = True,
            allow_media: bool = True) -> Extracted:
    """The text of `content`, or of the resource at `url`. Give exactly one of the two.

    `mime` is the media type, and may carry a `charset` parameter. For a URL it overrides
    the server's `Content-Type`. When neither gives a type, it is worked out from the
    first bytes (PDF, PNG, JPEG, GIF, WebP and HTML are recognised), and content that is
    none of those must be UTF-8 text. A `str` with no type is plain text, or HTML when it
    starts like an HTML page.

    `llm` is needed only for images, audio and video, and must implement
    `memvara.llm.Multimodal`. `fetcher` replaces the `SafeFetcher` used for `url`, which
    is how a test avoids the network.

    `allow_urls` and `allow_media` are the `ingest_urls` and `ingest_media` feature
    switches. When one is off, a URL or an image, audio or video is refused with the code
    `feature_off`.

    Raises `IngestError` for every failure; `MediaUnsupported` is the subclass for content
    nothing configured can read.

    >>> extract("Our office moved to Lisbon in May.")
    Extracted(text='Our office moved to Lisbon in May.', title=None, mime='text/plain', pages=None)
    >>> extract(b"<html><title>Hi</title><p>Hello there</p></html>").text
    'Hello there'
    >>> extract("x", url="https://example.com")
    Traceback (most recent call last):
    ...
    memvara.ingest.errors.IngestError: bad_input: give exactly one of content and url
    """
    if (content is None) == (url is None):
        raise IngestError("bad_input", "give exactly one of content and url")
    data: str | bytes
    declared = mime
    if url is not None:
        if not allow_urls:
            raise IngestError(
                "feature_off",
                "fetching a URL is switched off (the ingest_urls feature); pass the "
                "content instead")
        fetched = (fetcher or SafeFetcher()).fetch(url)
        data = fetched.body
        declared = mime or fetched.content_type
    else:
        assert content is not None  # narrows for mypy; checked just above
        data = content
    base, charset = _mime.parse(declared)
    if base in (None, "application/octet-stream"):
        base = _guess(data)
    if base.startswith(_MEDIA_PREFIXES) and not allow_media:
        raise IngestError(
            "feature_off",
            f"reading {base} is switched off (the ingest_media feature)")
    return _read(data, base, charset, llm)


def _guess(data: str | bytes) -> str:
    """The media type of content that came with none."""
    if isinstance(data, str):
        return _mime.sniff(data[:512].encode("utf-8", "replace")) or "text/plain"
    found = _mime.sniff(data)
    if found is not None:
        return found
    decode(data, None, strict=True)
    return "text/plain"


def _read(data: str | bytes, base: str, charset: str | None,
          llm: object | None) -> Extracted:
    pages: tuple[str, ...] | None = None
    title: str | None = None
    if base in _HTML_TYPES:
        markup = data if isinstance(data, str) else decode(data, charset, strict=False)
        text, title = html_to_text(markup)
    elif base.startswith("text/") or base in _TEXT_TYPES:
        text = data if isinstance(data, str) else decode(data, charset, strict=False)
    elif isinstance(data, str):
        raise IngestError(
            "bad_input", f"{base} content must be passed as bytes, not as a str")
    elif base == "application/pdf":
        found, title = pdf_pages(data)
        pages = tuple(found)
        text = "\n\n".join(page for page in pages if page)
    elif base.startswith(_MEDIA_PREFIXES):
        text = media_to_text(data, base, llm)
    else:
        raise MediaUnsupported(
            f"{base} is not a type this package can read. It reads text, HTML, PDF, "
            "images, audio and video")
    text = text.strip()
    if not text:
        raise IngestError("no_text", _why_empty(base))
    return Extracted(text=text, title=title, mime=base, pages=pages)


def _why_empty(base: str) -> str:
    if base == "application/pdf":
        return ("the PDF has no text layer, which usually means its pages are scanned "
                "images; no OCR is done")
    if base in _HTML_TYPES:
        return ("the page has no readable text outside scripts and navigation; a page "
                "that builds its content with JavaScript comes back empty")
    if base.startswith(_MEDIA_PREFIXES):
        return f"the model backend returned no text for this {base.split('/')[0]}"
    return "the content is empty"
