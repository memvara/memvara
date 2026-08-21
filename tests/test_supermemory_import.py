"""Moving a Supermemory account into this store.

Supermemory has no mutation log, so unlike `import_mem0` nothing here can reconstruct a
history it was never told. What it can do is bring every document across on its original
clock and say honestly what did and did not become a claim — which under an offline
extractor is "none of it", and is the number a migrating user most needs to see.

No test here touches the network: `fetch` is injected, which is also the seam a caller
uses to route through their own HTTP client.
"""

from __future__ import annotations

import json

import pytest

from memvara import HashingEmbedder, Memvara, NullLLM
from memvara.compat import (SupermemoryError, SupermemoryReceipt, import_supermemory,
                            read_supermemory_key)
from memvara.compat.supermemory_import import _created_at, _http_fetch, _text_of


def _memory(tmp_path, **kw):
    return Memvara(str(tmp_path / "s.db"), llm=NullLLM(),
                   embedder=HashingEmbedder(dim=64), tenant="t", **kw)


def _doc(n, **over):
    body = {"id": f"id{n}", "title": f"Title {n}", "summary": f"Summary {n}",
            "containerTags": ["repo_a"], "createdAt": "2026-08-20T21:41:08.153Z"}
    body.update(over)
    return body


def _pages(*pages):
    """A fetch that replays canned list responses and records what it was asked."""
    seen: list[dict] = []

    def fetch(url, key, body):
        seen.append({"url": url, "key": key, **body})
        page = int(body.get("page", 1))
        return {"memories": pages[page - 1],
                "pagination": {"currentPage": page, "totalPages": len(pages)}}

    fetch.seen = seen  # type: ignore[attr-defined]
    return fetch


class TestImport:
    def test_documents_arrive_as_episodes_on_their_original_clock(self, tmp_path):
        memory = _memory(tmp_path)
        receipt = import_supermemory(memory, api_key="k", fetch=_pages([_doc(1)]))
        assert receipt.documents == 1 and receipt.episodes == 1
        # Retrievable only with episodes asked for: an import under an offline extractor
        # writes text and derives no claims, and plain recall answers from claims.
        assert "Title 1" in memory.recall("title", k=5, include_episodes=True)

    def test_it_pages_until_the_last_page(self, tmp_path):
        fetch = _pages([_doc(1), _doc(2)], [_doc(3)])
        receipt = import_supermemory(_memory(tmp_path), api_key="k", page_size=2,
                                     fetch=fetch)
        assert (receipt.documents, receipt.pages) == (3, 2)
        assert [call["page"] for call in fetch.seen] == [1, 2]

    def test_max_pages_stops_a_runaway_export(self, tmp_path):
        """A server that always reports another page would otherwise loop forever."""
        def endless(url, key, body):
            return {"memories": [_doc(1)], "pagination": {"totalPages": 10_000}}

        receipt = import_supermemory(_memory(tmp_path), api_key="k", max_pages=3,
                                     fetch=endless)
        assert receipt.pages == 3

    def test_a_container_tag_narrows_the_export(self, tmp_path):
        fetch = _pages([_doc(1)])
        import_supermemory(_memory(tmp_path), api_key="k", container_tag="repo_a",
                           fetch=fetch)
        assert fetch.seen[0]["containerTags"] == ["repo_a"]

    def test_containers_are_reported(self, tmp_path):
        fetch = _pages([_doc(1, containerTags=["b"]), _doc(2, containerTags=["a"])])
        receipt = import_supermemory(_memory(tmp_path), api_key="k", fetch=fetch)
        assert receipt.containers == ["a", "b"]

    def test_an_empty_document_is_counted_not_dropped(self, tmp_path):
        """A document still being processed has no title and no summary yet. Counted, so
        a short import is explained by its own receipt rather than by guesswork."""
        fetch = _pages([_doc(1, title="", summary=""), _doc(2)])
        receipt = import_supermemory(_memory(tmp_path), api_key="k", fetch=fetch)
        assert (receipt.documents, receipt.episodes, receipt.empty) == (2, 1, 1)

    def test_a_non_mapping_entry_is_ignored(self, tmp_path):
        fetch = _pages([["not", "a", "document"], _doc(1)])
        receipt = import_supermemory(_memory(tmp_path), api_key="k", fetch=fetch)
        assert receipt.documents == 1

    def test_an_empty_page_ends_the_export(self, tmp_path):
        def empty(url, key, body):
            return {"memories": [], "pagination": {"totalPages": 99}}

        receipt = import_supermemory(_memory(tmp_path), api_key="k", fetch=empty)
        assert receipt.documents == 0 and receipt.pages == 1

    def test_a_reply_without_a_memories_list_names_the_route(self, tmp_path):
        def wrong(url, key, body):
            return {"data": []}

        with pytest.raises(SupermemoryError, match="/v3/documents/list"):
            import_supermemory(_memory(tmp_path), api_key="k", fetch=wrong)

    def test_a_string_body_is_not_mistaken_for_a_list_of_documents(self, tmp_path):
        def stringy(url, key, body):
            return {"memories": "nope"}

        with pytest.raises(SupermemoryError, match="memories"):
            import_supermemory(_memory(tmp_path), api_key="k", fetch=stringy)

    def test_malformed_pagination_ends_the_export_rather_than_raising(self, tmp_path):
        def odd(url, key, body):
            return {"memories": [_doc(1)], "pagination": {"totalPages": "many"}}

        assert import_supermemory(_memory(tmp_path), api_key="k", fetch=odd).pages == 1

    def test_pagination_of_the_wrong_shape_ends_the_export(self, tmp_path):
        def odd(url, key, body):
            return {"memories": [_doc(1)], "pagination": "nope"}

        assert import_supermemory(_memory(tmp_path), api_key="k", fetch=odd).pages == 1


class TestReceiptWording:
    def test_it_says_when_nothing_was_structured(self, tmp_path):
        """The sentence a migrating user needs. Episodes landed and no claims were
        derived, which looks like a failed import and is not one."""
        receipt = import_supermemory(_memory(tmp_path), api_key="k",
                                     fetch=_pages([_doc(1)]))
        assert "No claims were derived" in str(receipt)
        assert "MEMVARA_LLM" in str(receipt)

    def test_it_reports_skipped_and_containers(self):
        receipt = SupermemoryReceipt(documents=2, episodes=1, claims=1, empty=1, pages=1,
                                     containers=["a"])
        rendered = str(receipt)
        assert "1 empty skipped" in rendered and "containers: a" in rendered
        # Claims were derived here, so the offline warning must not fire.
        assert "No claims were derived" not in rendered


class TestCredentials:
    def test_it_reads_the_plugin_key(self, tmp_path):
        path = tmp_path / "credentials.json"
        path.write_text(json.dumps({"apiKey": "sm_live_x"}), encoding="utf-8")
        assert read_supermemory_key(path) == "sm_live_x"

    def test_a_missing_file_says_how_to_get_one(self, tmp_path):
        with pytest.raises(SupermemoryError, match="Sign in"):
            read_supermemory_key(tmp_path / "nope.json")

    def test_an_unreadable_file_is_reported(self, tmp_path):
        path = tmp_path / "credentials.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(SupermemoryError, match="could not be read"):
            read_supermemory_key(path)

    def test_a_file_without_a_key_is_reported(self, tmp_path):
        path = tmp_path / "credentials.json"
        path.write_text(json.dumps({"savedAt": "now"}), encoding="utf-8")
        with pytest.raises(SupermemoryError, match="no 'apiKey'"):
            read_supermemory_key(path)

    def test_a_non_object_body_is_reported(self, tmp_path):
        path = tmp_path / "credentials.json"
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(SupermemoryError, match="no 'apiKey'"):
            read_supermemory_key(path)

    def test_the_key_is_read_when_none_is_passed(self, tmp_path, monkeypatch):
        path = tmp_path / "credentials.json"
        path.write_text(json.dumps({"apiKey": "from_disk"}), encoding="utf-8")
        monkeypatch.setattr("memvara.compat.supermemory_import.CREDENTIALS_PATH",
                            str(path))
        fetch = _pages([_doc(1)])
        import_supermemory(_memory(tmp_path), fetch=fetch)
        assert fetch.seen[0]["key"] == "from_disk"


class TestFieldHandling:
    @pytest.mark.parametrize("document, expected", [
        ({"title": "T", "summary": "S"}, "T\n\nS"),
        ({"title": "T"}, "T"),
        ({"summary": "S"}, "S"),
        ({}, ""),
    ])
    def test_title_and_summary_are_both_kept(self, document, expected):
        # The title is often the only place the subject is named.
        assert _text_of(document) == expected

    @pytest.mark.parametrize("raw", ["", "not a date", None])
    def test_an_unusable_timestamp_falls_back_to_now(self, raw):
        assert _created_at({"createdAt": raw}) is None

    def test_a_zulu_timestamp_becomes_aware(self):
        assert _created_at({"createdAt": "2026-08-20T21:41:08.153Z"}).tzinfo is not None

    def test_a_naive_timestamp_is_treated_as_utc(self):
        assert _created_at({"createdAt": "2026-08-20T21:41:08"}).tzinfo is not None


class TestHttp:
    """The real transport. Exercised through a stubbed urlopen — the point is the error
    translation, and every one of these is a message a migrating user has to act on."""

    def _stub(self, monkeypatch, raiser):
        monkeypatch.setattr("memvara.compat.supermemory_import.urllib.request.urlopen",
                            raiser)

    def test_a_successful_call_returns_the_body(self, monkeypatch):
        class Response:
            def read(self): return b'{"memories": []}'
            def __enter__(self): return self
            def __exit__(self, *exc): return False

        self._stub(monkeypatch, lambda *a, **k: Response())
        assert _http_fetch("https://x/y", "k", {}) == {"memories": []}

    @pytest.mark.parametrize("code, fragment", [
        (401, "refused the key"), (403, "refused the key"), (500, "returned 500")])
    def test_http_errors_are_translated(self, monkeypatch, code, fragment):
        import urllib.error

        def raiser(*a, **k):
            raise urllib.error.HTTPError("u", code, "boom", {}, None)

        self._stub(monkeypatch, raiser)
        with pytest.raises(SupermemoryError, match=fragment):
            _http_fetch("https://x/y", "k", {})

    def test_a_certificate_failure_names_the_real_fix(self, monkeypatch):
        """The stdlib message here sends people to disable verification.

        A python.org macOS build ships no CA bundle of its own, so it rejects a
        certificate curl and every browser on the same machine accept. That is a broken
        Python install, not a broken certificate, and the difference decides whether
        someone reaches for `Install Certificates.command` or for `verify=False`.
        """
        import urllib.error

        def raiser(*a, **k):
            raise urllib.error.URLError(
                "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")

        self._stub(monkeypatch, raiser)
        with pytest.raises(SupermemoryError, match="Install Certificates.command"):
            _http_fetch("https://x/y", "k", {})

    def test_an_unreachable_host_is_translated(self, monkeypatch):
        import urllib.error

        def raiser(*a, **k):
            raise urllib.error.URLError("no route")

        self._stub(monkeypatch, raiser)
        with pytest.raises(SupermemoryError, match="could not be reached"):
            _http_fetch("https://x/y", "k", {})

    def test_malformed_json_is_translated(self, monkeypatch):
        class Response:
            def read(self): return b"<html>not json</html>"
            def __enter__(self): return self
            def __exit__(self, *exc): return False

        self._stub(monkeypatch, lambda *a, **k: Response())
        with pytest.raises(SupermemoryError, match="malformed JSON"):
            _http_fetch("https://x/y", "k", {})

    def test_a_non_object_body_is_rejected(self, monkeypatch):
        class Response:
            def read(self): return b"[1, 2]"
            def __enter__(self): return self
            def __exit__(self, *exc): return False

        self._stub(monkeypatch, lambda *a, **k: Response())
        with pytest.raises(SupermemoryError, match="not an object"):
            _http_fetch("https://x/y", "k", {})

    def test_it_never_sends_the_stdlib_user_agent(self, monkeypatch):
        """Some CDNs refuse `Python-urllib/*` at the edge, before the request reaches the
        application, and the resulting 403 says nothing about the cause."""
        captured = {}

        class Response:
            def read(self): return b"{}"
            def __enter__(self): return self
            def __exit__(self, *exc): return False

        def capture(request, *a, **k):
            captured.update(request.headers)
            return Response()

        self._stub(monkeypatch, capture)
        _http_fetch("https://x/y", "k", {})
        agent = captured.get("User-agent", "")
        assert agent and "urllib" not in agent.lower()
