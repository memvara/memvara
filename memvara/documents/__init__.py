"""Documents: text stored whole, searched in chunks, and deleted without erasing memory.

`chunk` splits a document into retrieval chunks, and `service` holds the operations
`Memvara.add_document`, `get_document`, `list_documents`, `update_document`,
`delete_document`, `delete_documents` and `document_status` delegate to.
"""

from .chunk import CHUNK_CHARS, CHUNK_OVERLAP, normalise, split
from .service import DocumentService

__all__ = ["CHUNK_CHARS", "CHUNK_OVERLAP", "DocumentService", "normalise", "split"]
