"""Plain text: bytes decoded into a string.

The character set a caller or server named is used when it names one Python knows.
Otherwise the bytes are read as UTF-8, with a leading byte-order mark removed.
"""

from __future__ import annotations

import codecs

from .errors import MediaUnsupported

__all__ = ["decode"]


def decode(data: bytes, charset: str | None, *, strict: bool) -> str:
    """`data` as text.

    With `strict`, bytes that are not valid in the character set mean this is not text,
    and `MediaUnsupported` is raised. That is how content with no stated type is told
    apart from a binary file. Without `strict`, invalid bytes become U+FFFD, because the
    caller said the content is text and one bad byte should not lose the rest.

    >>> decode("café".encode("latin-1"), "iso-8859-1", strict=False)
    'café'
    >>> decode(b"\\xef\\xbb\\xbfhello", None, strict=True)
    'hello'
    """
    encoding = "utf-8-sig"
    if charset:
        try:
            encoding = codecs.lookup(charset).name
        except LookupError:
            pass
    try:
        return data.decode(encoding, "strict" if strict else "replace")
    except UnicodeDecodeError:
        raise MediaUnsupported(
            "the content has no media type, is not a file type this package recognises, "
            "and is not UTF-8 text; pass mime= to say what it is") from None
