"""JSON from `/v1` back into the library's own dataclasses.

Each function here is the inverse of the function of the same name in
`memvara_cloud/rest/render.py`. That module is the authority: where the two disagree, this
one is wrong.

**Required fields are indexed, not `.get()`.** A server that renamed a field should raise
on the first call rather than hand back a claim carrying a plausible zero, which nothing
downstream can tell from a real one. Every field indexed here is required on the wire model
it comes from (present, even when its value may be `null`) — `.get()` is used only where the
wire model genuinely has no such field at all, so a default is the honest answer rather than
a guess about a key that could be missing. Two exceptions, each named at the call:
`anchor` on a ranking is `.get()` because the client has to read results from a server
that predates the field, and for that field a missing key means the server did not say,
which is what `None` on `Explanation.anchor` is documented to mean on a hosted result.
`claim_links` on a provenance body is `.get()` for the same reason: a server that
predates typed links sends no such field, and for it a missing key means none recorded.

**Instants are parsed by two functions, and which one a field gets is read off the wire
model.** `_dt` is for fields that may legitimately be null; `_required_dt` is for the ones
`/v1` declares non-nullable, and it raises on a null rather than passing `None` into a
dataclass field whose type says it cannot be one.

Three asymmetries the renderer introduces and this must undo:

* `extractor` is `""` in the library and `null` on the wire.
* `salience_base` and `last_observed` are top-level on the wire and `meta` keys here.
  `last_observed` is the sharper case: the library stores epoch seconds under
  `meta[LAST_OBSERVED]` and the property that reads it converts to a `datetime`, which
  is what the wire carries — so restoring it means parsing the wire's ISO string back
  into epoch seconds, not writing the string into `meta` verbatim.
* `state` and `links` are derived server-side. They are dropped, never stored: a claim
  carrying a `state` that disagreed with its own timestamps would be unfixable.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from ..retrieve.traverse import Edge, Path
from ..select.base import Rewrite, Selection
from ..select.stages import parse_day
from ..types import (
    DOCUMENT_STATES, LAST_OBSERVED, SALIENCE_BASE, Answer, Claim, DeleteResult, Delta,
    Derivation, Document, DocumentState, DocumentStatus, Episode, Explanation,
    ForgetPreview, ForgetResult, Link, MemoryType, Page, Profile, Provenance, Reading,
    Result, Row, Scope, WriteReceipt, closure, link_relation, stored_scope,
    Accumulation, Collapse, Dispute, RefusedProposal, Retype, z_as_offset,
)

__all__ = ["claim", "episode", "result", "explanation", "receipt", "provenance",
           "reading", "answer", "delta", "profile", "edge", "path", "scope",
           "selection", "rewrite", "link", "forget_preview", "forget_result",
           "document", "document_page", "document_status", "delete_result"]


def _dt(value: Any) -> datetime | None:
    """A wire instant that is allowed to be null, as `datetime | None`.

    `null` is a value on these fields rather than a defect: `valid_to` is null on a claim
    the world has not moved past, `invalidated_at` on one nothing has retired,
    `last_observed` on one nothing has re-observed.

    **The trailing `Z` is rewritten before parsing, and that is not cosmetic.** The
    facade sends the `Z` form for every instant it renders, and Python 3.10's
    `datetime.fromisoformat` rejects it (see `types.z_as_offset`). Without the rewrite,
    every claim, episode, answer and delta would fail to hydrate on 3.10 while passing on
    3.13, which is a bug a test run on one version cannot see.
    """
    if value is None:
        return None
    return datetime.fromisoformat(z_as_offset(str(value)))


def _required_dt(field: str, value: Any) -> datetime:
    """A wire instant the API always sends, as a plain `datetime`.

    Raises on a null rather than widening the dataclass field it fills. `Claim.valid_from`,
    `Claim.recorded_at`, `Episode.ts`, `Answer.at` and `Delta.since` are non-optional in
    the library because every one of them records something that did happen at some
    instant, and the models they come from declare them required and non-nullable — see
    `valid_from`, `recorded_at`, `EpisodeModel.ts`, `AskResponse.at` and
    `DeltaResponse.since` in `memvara_cloud/rest/models.py`. A null there is the server
    disagreeing with its own schema.

    Failing here is the point. Making the fields optional instead would hand a wire-format
    defect to every caller to rediscover as a `None` where the type says there cannot be
    one, arbitrarily far from the response that carried it.
    """
    parsed = _dt(value)
    if parsed is None:
        raise ValueError(
            f"{field} came back null. /v1 declares it required and non-null, so this "
            "response disagrees with its own schema; the field it fills cannot be None.")
    return parsed


def scope(body: dict[str, Any]) -> Scope:
    # `user`, `agent` and `session` are required on `ScopeModel` even though each is
    # nullable, so indexing them is what makes a renamed field raise instead of a null
    # scope component silently reading as "unbound".
    # `project` is read with `get`, because a deployment that predates project scope
    # renders no such field, and a scope without one is a scope with no project.
    #
    # `stored_scope` rather than `Scope`: this is a row the deployment already holds, and
    # one stored before '*' and '' were refused can hold either. Refusing it here would
    # fail the whole read over one row the caller cannot do anything about.
    return stored_scope(body["tenant"], body["user"], body["agent"], body["session"],
                        project=body.get("project"))


def claim(body: dict[str, Any]) -> Claim:
    valid, txn = body["valid_time"], body["transaction_time"]
    meta = dict(body["metadata"])
    # `salience_base` and `last_observed` are required, non-`.get()`-able fields on
    # `Memory`, not optional ones — they can be `null` (an unobserved claim), but the key
    # itself is never absent.
    if body["salience_base"] is not None:
        meta[SALIENCE_BASE] = body["salience_base"]
    # `Claim.last_observed` stores epoch seconds in `meta`; the wire carries the datetime
    # it converts to (`Memory.last_observed`), not the float underneath.
    last_observed = _dt(body["last_observed"])
    if last_observed is not None:
        meta[LAST_OBSERVED] = last_observed.timestamp()
    out = Claim(
        subject=body["subject"],
        predicate=body["predicate"],
        object=body["object"],
        scope=scope(body["scope"]),
        text=body["text"],
        polarity=body["polarity"],
        memory_type=MemoryType(body["memory_type"]),
        confidence=body["confidence"],
        salience=body["salience"],
        observation_count=body["observation_count"],
        sources=list(body["source_ids"]),
        derivation=Derivation(body["derivation"]),
        # The wire says null for what the library spells as the empty string.
        extractor=body["extractor"] or "",
        id=body["id"],
        meta=meta,
    )
    out.valid_from = _required_dt("valid_from", valid["valid_from"])
    out.valid_to = _dt(valid["valid_to"])
    out.recorded_at = _required_dt("recorded_at", txn["recorded_at"])
    out.invalidated_at = _dt(txn["invalidated_at"])
    out.invalidated_by = txn["invalidated_by"]
    # `.get`, because a deployment from before expiry renders neither field, and a claim
    # it sends has no expiry.
    out.expires_at = _dt(body.get("expires_at"))
    out.expire_reason = body.get("expire_reason")
    return out


def episode(body: dict[str, Any]) -> Episode:
    return Episode(content=body["content"], role=body["role"],
                   ts=_required_dt("ts", body["ts"]),
                   id=body["id"], scope=scope(body["scope"]),
                   meta=dict(body["metadata"]))


def explanation(body: dict[str, Any] | None) -> Explanation:
    """`render.ranking`, backwards. `None` for a response that carried no ranking — a
    listing rather than a search — and an all-defaults `Explanation` is the right answer
    there, because nothing ranked it.

    `graph_rank`, `graph_score`, `temporal_rank`, `temporal_score` and `intent` exist on
    `Explanation` but not on `Ranking` — `render.ranking` never puts them on the wire, so
    there is nothing here to read them back from; a restored `Explanation` reports them
    at their dataclass defaults. `anchor`, `selected` and `span` are read when the server
    sent them and left at `None` when it did not, so against a server from before any of
    those fields a hosted result reports `None` for a reason a local one never would: the
    server did not say.

    `recency`, `confidence` and `salience` are `float` on `Explanation`, not
    `float | None` — null on the wire means "not applicable" (an episode hit), and that
    is the dataclass's own default (`1.0`), not `None`. They are the one place a required
    field is read conditionally rather than assigned straight through.
    """
    if not body:
        return Explanation()
    out = Explanation(
        vector_rank=body["vector_rank"],
        vector_score=body["vector_score"],
        lexical_rank=body["lexical_rank"],
        lexical_score=body["lexical_score"],
        fusion_score=body["fusion_score"],
        rerank_score=body["rerank_score"],
        raw_score=body["raw_score"],
        final_score=body["final_score"],
        # `.get`, not `[]`: a server from before these fields omits them, and a result it
        # ranked is not malformed for having nothing to report for any of the three.
        anchor=body.get("anchor"),
        selected=body.get("selected"),
        span=body.get("span"),
    )
    for name in ("recency", "confidence", "salience"):
        if body[name] is not None:
            setattr(out, name, body[name])
    return out


def result(body: dict[str, Any]) -> Result:
    return Result(claim=claim(body["memory"]), score=body["score"],
                  explain=explanation(body["ranking"]))


def receipt(body: dict[str, Any]) -> WriteReceipt:
    # `note` is rendered from `unextracted`, never stored — `WriteReceipt` has no field
    # to put it back into, the same reason `state` and `links` are dropped from `claim`.
    return WriteReceipt(
        episode_ids=list(body["episode_ids"]),
        added=[claim(c) for c in body["added"]],
        closed=[claim(c) for c in body["invalidated"]],
        reinforced=[claim(c) for c in body["reinforced"]],
        skipped=body["skipped"],
        # `.get`: a deployment older than #439 sends no such key and counts its repeats
        # in `skipped`, so 0 is what it reported, and the client cannot split `skipped`.
        repeated=int(body.get("repeated", 0)),
        unextracted=body["unextracted"],
        llm_calls=body["llm_calls"],
        latency_ms=body["latency_ms"],
        deferred=body["deferred"],
        # `.get`: a deployment older than replacement advice sends no such key, and an
        # absent list means what an empty one means, that nothing was suggested.
        may_replace=[claim(c) for c in body.get("may_replace", ())],
        # `.get` for the same reason: a deployment older than memvara 0.14.0 sends no
        # such key, and 0 is what an absent count means.
        unregistered=int(body.get("unregistered", 0)),
        # The lists the MCP server writes its notes from, and the agentic outcomes
        # (#334). `.get` for each: a deployment that does not send one yet means what an
        # empty one means, that nothing of that kind happened on this write.
        accumulated=[Accumulation(subject=a["subject"], predicate=a["predicate"],
                                  existing=int(a["existing"]))
                     for a in body.get("accumulated", ())],
        disputed=[Dispute(claim_id=d["claim_id"], subject=d["subject"],
                          predicate=d["predicate"], incumbent=d["incumbent"],
                          incumbent_confidence=float(d["incumbent_confidence"]),
                          candidate=d["candidate"],
                          candidate_confidence=float(d["candidate_confidence"]))
                  for d in body.get("disputed", ())],
        collapsed=[Collapse(claim_id=c["claim_id"], subject=c["subject"],
                            predicate=c["predicate"], object=c["object"],
                            at=_required_dt("collapsed.at", c["at"]))
                   for c in body.get("collapsed", ())],
        retyped=[Retype(claim_id=r["claim_id"], subject=r["subject"],
                        predicate=r["predicate"], was=MemoryType(r["was"]),
                        now=MemoryType(r["now"]), reason=r.get("reason", "asserted"))
                 for r in body.get("retyped", ())],
        agentic_fallback=body.get("agentic_fallback"),
        proposals_refused=[RefusedProposal(tool=p["tool"], target=p["target"],
                                           reason=p["reason"])
                           for p in body.get("proposals_refused", ())],
        # The deployment sends both, and they were never read, so a hosted receipt said
        # 0 whatever it counted (#335). `.get` for a deployment that does not send them.
        ungrounded=int(body.get("ungrounded", 0)),
        polluted=int(body.get("polluted", 0)),
    )


def provenance(body: dict[str, Any]) -> Provenance:
    """`render.provenance`, backwards.

    `claim_links` is read with `.get()`; the module docstring says why. The field is not
    called `links` because a memory body already uses that name for the routes it can be
    reached at.
    """
    return Provenance(
        claim=claim(body["memory"]),
        episodes=[episode(e) for e in body["sources"]],
        derivation=Derivation(body["derivation"]),
        extractor=body["extractor"] or "",
        superseded=[claim(c) for c in body["superseded"]],
        links=[link(k) for k in body.get("claim_links") or []],
    )


def link(body: dict[str, Any]) -> Link:
    """One typed link, as `POST /v1/links` returns it and `claim_links` lists it."""
    return Link(from_id=body["from_id"], to_id=body["to_id"],
                relation=link_relation(body["relation"]),
                created_at=_required_dt("created_at", body["created_at"]),
                by=body["by"])


def forget_preview(body: dict[str, Any]) -> ForgetPreview:
    """The preview half of `POST /v1/forget-matching`: matches, token, expiry."""
    return ForgetPreview(
        close=closure(body["close"]),
        matches={m["memory_id"]: m["text"] for m in body["matches"]},
        confirm=body["confirm"],
        expires_at=_required_dt("expires_at", body["expires_at"]))


def forget_result(body: dict[str, Any]) -> ForgetResult:
    """The confirmed half of `POST /v1/forget-matching`: what was closed, and why."""
    return ForgetResult(close=closure(body["close"]),
                        closed=[claim(c) for c in body["closed"]],
                        reason=body["reason"])


def reading(body: dict[str, Any]) -> Reading:
    """`render.reading`, backwards — `now`, `then` and `stated` only.

    `timeline` and `single_valued` are library-only: `ReadingModel` has neither field, so
    a restored `Reading` reports them at their dataclass defaults (an empty tuple and
    `False`) rather than from anything the wire said. `diverged` and `moved` are
    properties computed from `now`/`then`/`stated` and are dropped for the reason
    `Path.labels` and `Path.hops` are: storing the rendered spellings back would be a
    second implementation of the comparison they are computed from.
    """
    return Reading(
        subject=body["subject"], predicate=body["predicate"],
        now=tuple(claim(c) for c in body["now"]),
        then=tuple(claim(c) for c in body["then"]),
        stated=tuple(claim(c) for c in body["stated"]),
    )


def answer(body: dict[str, Any]) -> Answer:
    return Answer(question=body["question"], at=_required_dt("at", body["at"]),
                  readings=tuple(reading(r) for r in body["readings"]),
                  text=body["text"])


def delta(body: dict[str, Any]) -> Delta:
    return Delta(since=_required_dt("since", body["since"]),
                 added=tuple(claim(c) for c in body["added"]),
                 gone=tuple(claim(c) for c in body["gone"]))


def _row(body: dict[str, Any]) -> Row:
    return Row(claim_id=body["claim_id"], text=body["text"],
               inferred=bool(body["inferred"]))


def profile(body: dict[str, Any]) -> Profile:
    """`POST /v1/profile`'s reply. Every key is indexed rather than read with `get`, so a
    deployment that renamed a section raises here instead of reading as an empty one."""
    return Profile(standing=[_row(r) for r in body["standing"]],
                   recent=[_row(r) for r in body["recent"]],
                   relevant=[_row(r) for r in body["relevant"]],
                   buckets={name: [_row(r) for r in rows]
                            for name, rows in body["buckets"].items()},
                   warnings=[str(w) for w in body["warnings"]])


def edge(body: dict[str, Any]) -> Edge:
    """`render.edge`, backwards.

    `backward` is carried rather than recomputed. A claim read object-to-subject is a
    different statement — `Acme founded_by Bob` reached from Bob is still "Acme was
    founded by Bob" — and inferring the direction here would assert something nobody
    stored.
    """
    return Edge(claim=claim(body["memory"]), backward=body["backward"],
                strength=body["strength"])


def path(body: dict[str, Any]) -> Path:
    """`render.path`, backwards — `nodes`, `edges` and `score` only.

    `labels` and `hops` are on the wire and are **not** passed back: both are properties
    computed from the edges the walk crossed. Storing the rendered spellings would be a
    second implementation of the fold that identity is stored under, and a second one can
    disagree.
    """
    return Path(nodes=tuple(body["nodes"]),
                edges=tuple(edge(e) for e in body["edges"]),
                score=body["score"])


def rewrite(body: dict[str, Any] | None) -> Rewrite | None:
    """The `Rewrite` a read's query rewrite produced, or `None`.

    `None` when the response carries no `rewrite`, which is what a deployment from
    before the field sends, and what a read that passed `query_rewrite=False` gets. The
    wire shape is `{"outcome", "reason", "status", "queries", "date_from", "date_to",
    "valid_at", "valid_during"}`, with the two dates as `YYYY-MM-DD`, `valid_at` as an
    instant and `valid_during` as `{"start", "end"}` instants; every field but `outcome`
    may be absent or null. A deployment from before `valid_during` does not send it.

    The date range is all or nothing: both dates or neither, and a `valid_at` or a
    `valid_during` only beside both. A body that breaks that raises `ValueError` rather than being decoded into a
    range with a missing end, because a caller reading `valid_at` would otherwise be told
    the read was dated by a range nobody can state.

    >>> rewrite({"outcome": "applied", "queries": ["Lisbon trip"],
    ...          "date_from": "2024-03-01", "date_to": "2024-03-31"}).date_to
    datetime.date(2024, 3, 31)
    >>> rewrite(None) is None
    True
    """
    if not body:
        return None
    raw_from, raw_to = body.get("date_from"), body.get("date_to")
    valid_at = _dt(body.get("valid_at"))
    raw_window = body.get("valid_during")
    window = (None if not raw_window else
              (_required_dt("valid_during.start", raw_window["start"]),
               _required_dt("valid_during.end", raw_window["end"])))
    if (raw_from is None) != (raw_to is None) or (
            (valid_at is not None or window is not None) and raw_to is None):
        raise ValueError(
            "a rewrite's date_from, date_to, valid_at and valid_during come together: "
            "both dates or neither, and valid_at or valid_during only beside them; got "
            f"{raw_from!r}, {raw_to!r}, {body.get('valid_at')!r}, {raw_window!r}")
    return Rewrite(
        outcome=body["outcome"],
        reason=body.get("reason"),
        status=body.get("status"),
        queries=tuple(body.get("queries") or ()),
        date_from=None if raw_from is None else parse_day(raw_from),
        date_to=None if raw_to is None else parse_day(raw_to),
        valid_at=valid_at,
        valid_during=window,
    )


def selection(body: dict[str, Any] | None) -> Selection | None:
    """The `Selection` a ranked read's model consultation produced, or `None`.

    `None` for a plain read — `SearchResponse.selection` and `RecallResponse.selection`
    are both null there — and for a server from before the field, which sends no
    `selection` key at all. Those are the two cases a caller of `search()` or
    `recall(with_ids=True)` needs to tell apart from a ranked read whose outcome was
    something other than `applied`, and both already read as `None` here without this
    function needing to distinguish them itself.
    """
    if not body:
        return None
    return Selection(
        outcome=body["outcome"],
        reason=body.get("reason"),
        status=body.get("status"),
        candidates=body.get("candidates", 0),
        kept=body.get("kept", 0),
    )


def _state(value: Any) -> DocumentState:
    """A document status off the wire, refused if it is not one of the four. A status
    the library has no word for would be carried into every `if doc.status == ...` a
    caller writes, and match none of them."""
    if value not in DOCUMENT_STATES:
        raise ValueError(f"document status {value!r} is not one of {DOCUMENT_STATES}")
    return value  # type: ignore[no-any-return]


def document(body: dict[str, Any]) -> Document:
    """One document, as `POST /v1/documents`, `GET /v1/documents/{id}` and `PATCH` return
    it."""
    return Document(
        id=body["id"], scope=scope(body["scope"]), custom_id=body["custom_id"],
        title=body["title"], filepath=body["filepath"], source_uri=body["source_uri"],
        mime=body["mime"], content_hash=body["content_hash"],
        status=_state(body["status"]), error=body["error"],
        meta=dict(body["metadata"]),
        created_at=_required_dt("created_at", body["created_at"]),
        updated_at=_required_dt("updated_at", body["updated_at"]),
        chunks=int(body["chunks"]))


def document_page(body: dict[str, Any]) -> Page[Document]:
    """One page of `GET /v1/documents`."""
    return Page([document(d) for d in body["documents"]], body["next_cursor"])


def document_status(body: dict[str, Any]) -> DocumentStatus:
    """`GET /v1/documents/{id}/status`."""
    return DocumentStatus(id=body["id"], status=_state(body["status"]),
                          error=body["error"], chunks=int(body["chunks"]),
                          updated_at=_required_dt("updated_at", body["updated_at"]))


def delete_result(body: dict[str, Any]) -> DeleteResult:
    """One document delete, as `DELETE /v1/documents/{id}` returns it and
    `POST /v1/documents/delete` lists it."""
    return DeleteResult(id=body["id"], deleted=bool(body["deleted"]),
                        custom_id=body["custom_id"], chunks=int(body["chunks"]),
                        episodes=int(body["episodes"]), retired=tuple(body["retired"]),
                        unlinked=tuple(body["unlinked"]))
