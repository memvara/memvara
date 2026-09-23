"""The errors ingestion raises.

Each error carries a short `code` that a server can put in a response unchanged, and a
`reason` sentence a person can act on. The codes are:

- `media_unsupported`: the content is a kind of file nothing configured can read, such as
  an image with no model that reads images, a PDF without the `ingest` extra, or a file
  type that no module here handles.
- `url_refused`: the URL is not one this library will fetch, because its scheme is not
  `http` or `https` or because its host resolves to an address that is not public.
- `fetch_failed`: the fetch was allowed but did not succeed (DNS failure, connection
  error, an HTTP status that is not 2xx, a redirect without a location, too many redirects).
- `too_large`: the response body is over the size limit.
- `timeout`: the fetch did not finish within the time limit.
- `unreadable`: the content claims a type but cannot be parsed as it, such as a damaged or
  password-protected PDF.
- `no_text`: the content was read and held no text, such as a PDF of scanned pages.
- `feature_off`: the caller switched this kind of ingestion off.
- `bad_input`: the arguments do not describe one piece of content.

These live in their own module so that the model backends in `memvara.llm` can raise
`MediaUnsupported` without importing the rest of this package.
"""

from __future__ import annotations

__all__ = ["IngestError", "MediaUnsupported"]


class IngestError(ValueError):
    """Content could not be turned into text. `code` says which kind of failure.

    >>> error = IngestError("no_text", "the PDF has no text layer")
    >>> error.code, error.reason
    ('no_text', 'the PDF has no text layer')
    >>> str(error)
    'no_text: the PDF has no text layer'
    """

    def __init__(self, code: str, reason: str) -> None:
        super().__init__(f"{code}: {reason}")
        self.code = code
        self.reason = reason


class MediaUnsupported(IngestError):
    """Nothing configured can read this kind of content. The code is `media_unsupported`.

    A model backend raises this from `describe_image` or `transcribe` when it cannot
    handle the media type it was given, and says why in `reason`.

    >>> MediaUnsupported("audio is not accepted by this backend").code
    'media_unsupported'
    """

    def __init__(self, reason: str) -> None:
        super().__init__("media_unsupported", reason)
