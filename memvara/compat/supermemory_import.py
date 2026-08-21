"""Move a Supermemory account into this store.

Supermemory keeps documents: a title, a generated summary, some container tags, and a
created-at. There is no mutation log to replay, which is the whole difference from
`import_mem0` — mem0 records *what changed and when*, so that importer can reconstruct
supersession and answer `as_of` questions afterwards. Supermemory records the current
state, so nothing here can invent a history it was never told.

What that means for the result, stated plainly because a migration that quietly promises
more than it delivers is worse than one that refuses:

* **Documents arrive as episodes, not as claims.** Each one is written with its original
  `createdAt`, so the timeline is true even though the claims are not derived. The text is
  immediately searchable — the store indexes episodes on write for BM25 and vectors.
* **Claims appear only if this store has an extractor.** Under the default
  `MEMVARA_LLM=none` the deterministic path recognises the sentence forms it recognises
  and nothing else; that is not a failure of the import, and the receipt reports the two
  numbers separately so the difference is visible rather than assumed.
* **Nothing is retired.** An import adds; it never closes a value already here.

Network access is stdlib only — no SDK, no new dependency — and every call goes through
one injectable `fetch`, which is also how the tests reach this without a network.
"""

from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

__all__ = ["SupermemoryError", "SupermemoryReceipt", "import_supermemory",
           "read_supermemory_key"]

#: Where the Supermemory Claude Code plugin writes the key it obtained.
CREDENTIALS_PATH = os.path.join(os.path.expanduser("~"), ".supermemory-claude",
                                "credentials.json")

DEFAULT_BASE_URL = "https://api.supermemory.ai"
LIST_PATH = "/v3/documents/list"

#: Their maximum useful page. Larger values are accepted and silently clamped, which would
#: make a caller think it had paged when it had not.
DEFAULT_PAGE_SIZE = 100

#: Anything but the stdlib default. `Python-urllib/*` is refused by some CDNs before the
#: request reaches the application, and the resulting 403 says nothing about the cause.
USER_AGENT = "memvara-import/0.1"

TIMEOUT_SEC = 30.0


class SupermemoryError(RuntimeError):
    """The export could not be read. Raised with the fix in the message."""


@dataclass(slots=True)
class SupermemoryReceipt:
    """What the import did. Returned by `import_supermemory`.

    `documents` and `claims` are reported separately on purpose: they are equal only when
    this store has an extractor, and a single number would hide the case where every
    document arrived and nothing was derived from any of them.
    """

    documents: int = 0        # documents read from Supermemory
    episodes: int = 0         # documents written to this store
    claims: int = 0           # claims the write path derived from them
    empty: int = 0            # documents with no title and no summary, skipped
    pages: int = 0            # list requests made
    containers: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        head = (f"{self.documents} document(s) over {self.pages} page(s) -> "
                f"{self.episodes} episode(s), {self.claims} claim(s)")
        if self.empty:
            head += f", {self.empty} empty skipped"
        if self.containers:
            head += f"\ncontainers: {', '.join(self.containers)}"
        if not self.claims and self.episodes:
            head += ("\nNo claims were derived: this store has no extractor configured, so "
                     "the text is searchable but no facts were structured from it. Set "
                     "MEMVARA_LLM and re-run to derive them.")
        return head


def read_supermemory_key(path: str | os.PathLike[str] | None = None) -> str:
    """The API key the Supermemory plugin stored, or raise saying how to get one."""
    target = os.fspath(path) if path is not None else CREDENTIALS_PATH
    try:
        with open(target, encoding="utf-8") as handle:
            body = json.load(handle)
    except FileNotFoundError:
        raise SupermemoryError(
            f"No Supermemory credentials at {target}. Sign in with the Supermemory "
            "plugin, or pass api_key= explicitly.") from None
    except (OSError, ValueError) as exc:
        raise SupermemoryError(f"{target} could not be read: {exc}") from None

    key = (body or {}).get("apiKey") if isinstance(body, dict) else None
    if not key:
        raise SupermemoryError(
            f"{target} has no 'apiKey'. Sign in with the Supermemory plugin again.")
    return str(key)


def _http_fetch(url: str, api_key: str, body: Mapping[str, Any]) -> Mapping[str, Any]:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json",
                 "authorization": f"Bearer {api_key}",
                 "user-agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SEC,
                                    context=ssl.create_default_context()) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:200].decode("utf-8", "replace")
        if exc.code in (401, 403):
            raise SupermemoryError(
                f"Supermemory refused the key ({exc.code}). Sign in with the plugin "
                f"again, or pass a current api_key=. {detail}") from None
        raise SupermemoryError(f"Supermemory returned {exc.code}: {detail}") from None
    except (urllib.error.URLError, TimeoutError) as exc:
        if "CERTIFICATE_VERIFY_FAILED" in str(exc):
            # Names the actual fix, because the message the stdlib gives here sends
            # people to disable verification. A python.org build on macOS ships no trust
            # store of its own, so it rejects a certificate curl and every browser on the
            # same machine accept. No certifi import: an SDK this package does not
            # declare is a dependency nobody agreed to ship.
            raise SupermemoryError(
                "Supermemory's certificate could not be verified. This is usually a "
                "Python install with no CA bundle rather than a real certificate "
                "problem — on macOS run the \"Install Certificates.command\" that ships "
                "with python.org builds, or pass fetch= to route through a client that "
                "has one. Verification is not disabled here.") from None
        raise SupermemoryError(f"Supermemory could not be reached: {exc}") from None
    except ValueError as exc:
        raise SupermemoryError(f"Supermemory returned malformed JSON: {exc}") from None
    if not isinstance(payload, Mapping):
        raise SupermemoryError("Supermemory returned a body that is not an object.")
    return payload


def _text_of(document: Mapping[str, Any]) -> str:
    """Title and summary, joined. Empty when the document carries neither.

    Both are kept rather than only the summary: the title is often the only place the
    subject is named, and a summary read without it loses what it is about.
    """
    title = str(document.get("title") or "").strip()
    summary = str(document.get("summary") or "").strip()
    if title and summary:
        return f"{title}\n\n{summary}"
    return title or summary


def _created_at(document: Mapping[str, Any]) -> datetime | None:
    raw = str(document.get("createdAt") or "").strip()
    if not raw:
        return None
    try:
        # Their timestamps end in `Z`, which fromisoformat rejects before 3.11 and
        # accepts after; normalised either way rather than depending on the runtime.
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def import_supermemory(
    memory: Any,
    *,
    api_key: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
    container_tag: str | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_pages: int = 1000,
    fetch: Callable[[str, str, Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> SupermemoryReceipt:
    """Copy every Supermemory document into `memory`, oldest timestamp preserved.

    `container_tag` limits the export to one container; omitted, the account's default
    selection is taken. `fetch` replaces the HTTP call, which is how this is tested
    without a network and how a caller can route through their own client.
    """
    key = api_key or read_supermemory_key()
    call = fetch or _http_fetch
    url = base_url.rstrip("/") + LIST_PATH

    receipt = SupermemoryReceipt()
    containers: set[str] = set()
    page = 1
    while page <= max_pages:
        body: dict[str, Any] = {"limit": page_size, "page": page}
        if container_tag:
            body["containerTags"] = [container_tag]
        payload = call(url, key, body)
        receipt.pages += 1

        documents = payload.get("memories")
        if not isinstance(documents, Sequence) or isinstance(documents, (str, bytes)):
            raise SupermemoryError(
                "Supermemory's reply has no 'memories' list. The export API has changed "
                "shape; this importer reads POST /v3/documents/list.")
        for document in documents:
            if not isinstance(document, Mapping):
                continue
            receipt.documents += 1
            text = _text_of(document)
            if not text:
                # A document still being processed has neither yet. Counted rather than
                # dropped silently, so a short import is explained by its own receipt.
                receipt.empty += 1
                continue
            containers.update(str(t) for t in document.get("containerTags") or ())
            written = memory.add(text, role="user", ts=_created_at(document))
            receipt.episodes += 1
            # `added` is the claims the write path derived. Under an offline extractor it
            # is routinely empty while the episode itself landed, which is exactly the gap
            # the receipt reports rather than averages away.
            receipt.claims += len(written.added)

        pagination = payload.get("pagination")
        total_pages = 0
        if isinstance(pagination, Mapping):
            try:
                total_pages = int(pagination.get("totalPages") or 0)
            except (TypeError, ValueError):
                total_pages = 0
        if page >= total_pages or not documents:
            break
        page += 1

    receipt.containers = sorted(containers)
    return receipt
