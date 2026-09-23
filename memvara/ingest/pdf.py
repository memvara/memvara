"""The text of each page of a PDF, and its title, through `pypdf`.

`pypdf` comes from the `ingest` extra (`pip install 'memvara[ingest]'`) and is imported
inside the function, so `import memvara` works without it. Without it, a PDF is refused
with `media_unsupported` and a reason that names the extra.

Only the text layer is read. A PDF made of scanned page images has no text layer, and no
OCR is done here, so its pages come back empty and `extract` refuses it with `no_text`.
"""

from __future__ import annotations

import io

from .errors import IngestError, MediaUnsupported

__all__ = ["pdf_pages"]


def pdf_pages(data: bytes) -> tuple[list[str], str | None]:
    """The text of every page, in order, and the title from the PDF's metadata."""
    try:
        import pypdf  # type: ignore[import-not-found, unused-ignore]
    except ImportError:
        raise MediaUnsupported(
            "reading a PDF needs the pypdf package: pip install 'memvara[ingest]'"
        ) from None
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise IngestError("unreadable", "the PDF is protected by a password")
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
        metadata = reader.metadata
    except IngestError:
        raise
    except Exception as exc:
        # pypdf raises several unrelated exception types on a damaged file (its own
        # `PdfReadError`, but also `ValueError`, `KeyError` and `struct.error` from deep
        # inside the parser), so the only reliable boundary is "anything".
        raise IngestError("unreadable", f"the PDF could not be read: {exc}") from None
    title = str(metadata.title).strip() if metadata and metadata.title else ""
    return pages, title or None
