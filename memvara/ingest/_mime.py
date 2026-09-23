"""Media types: reading a `Content-Type` value, and guessing one from the bytes.

A caller or a web server may send a media type with parameters (`text/html; charset=...`),
in upper case, or under an old alias (`image/jpg`). Everything after this module sees one
lower-case base type and, separately, the character set if one was given.
"""

from __future__ import annotations

__all__ = ["parse", "sniff"]

#: Old or non-standard names for a type, mapped to the name the rest of the package uses.
_ALIASES = {
    "image/jpg": "image/jpeg",
    "image/pjpeg": "image/jpeg",
    "audio/mp3": "audio/mpeg",
    "audio/x-mp3": "audio/mpeg",
    "audio/mpeg3": "audio/mpeg",
    "audio/x-wav": "audio/wav",
    "audio/wave": "audio/wav",
    "audio/vnd.wave": "audio/wav",
    "audio/x-m4a": "audio/m4a",
    "audio/x-flac": "audio/flac",
    "application/x-pdf": "application/pdf",
}

#: File signatures, checked in order against the first bytes of the content.
_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"%PDF-", "application/pdf"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


def parse(value: str | None) -> tuple[str | None, str | None]:
    """The base type and the charset of a `Content-Type` value, either may be `None`.

    >>> parse("Text/HTML; charset=ISO-8859-1")
    ('text/html', 'iso-8859-1')
    >>> parse("image/jpg")
    ('image/jpeg', None)
    >>> parse('text/plain; charset="utf-8"')
    ('text/plain', 'utf-8')
    >>> parse("  ")
    (None, None)
    """
    if value is None:
        return None, None
    head, *params = value.split(";")
    base = head.strip().lower() or None
    if base is not None:
        base = _ALIASES.get(base, base)
    charset = None
    for param in params:
        key, _, raw = param.partition("=")
        if key.strip().lower() == "charset" and raw.strip():
            charset = raw.strip().strip("\"'").lower()
    return base, charset


def sniff(data: bytes) -> str | None:
    """A media type guessed from the first bytes, or `None` when nothing matches.

    Only used when the caller and the server both gave no type. Text is not guessed here;
    the caller decides that by trying to decode it.

    >>> sniff(b"%PDF-1.7 ...")
    'application/pdf'
    >>> sniff(b"RIFF\\x00\\x00\\x00\\x00WEBPVP8 ")
    'image/webp'
    >>> sniff(b"  <!DOCTYPE html><html>")
    'text/html'
    >>> sniff(b"plain words") is None
    True
    """
    for signature, mime in _SIGNATURES:
        if data.startswith(signature):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    head = data[:512].lstrip().lower()
    if head.startswith((b"<!doctype html", b"<html")):
        return "text/html"
    return None
