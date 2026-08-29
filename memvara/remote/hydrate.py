"""JSON from `/v1` back into the library's own dataclasses.

Each function here is the inverse of the function of the same name in
`memvara_cloud/rest/render.py`. That module is the authority: where the two disagree, this
one is wrong.

**Required fields are indexed, not `.get()`.** A server that renamed a field should raise
on the first call rather than hand back a claim carrying a plausible zero, which nothing
downstream can tell from a real one. Every field indexed here is required on the wire model
it comes from (present, even when its value may be `null`) — `.get()` is used only where the
wire model genuinely has no such field at all, so a default is the honest answer rather than
a guess about a key that could be missing.

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
from ..types import (
    LAST_OBSERVED, SALIENCE_BASE, Answer, Claim, Delta, Derivation, Episode,
    Explanation, MemoryType, Provenance, Reading, Result, Scope, WriteReceipt,
)

__all__ = ["claim", "episode", "result", "explanation", "receipt", "provenance",
           "reading", "answer", "delta", "edge", "path", "scope"]


def _dt(value: Any) -> datetime | None:
    """A wire instant that is allowed to be null, as `datetime | None`.

    `null` is a value on these fields rather than a defect: `valid_to` is null on a claim
    the world has not moved past, `invalidated_at` on one nothing has retired,
    `last_observed` on one nothing has re-observed.

    **The trailing `Z` is rewritten before parsing, and that is not cosmetic.**
    `datetime.fromisoformat` did not accept a `Z` suffix before Python 3.11, and this
    package supports 3.10 — `requires-python` says so and CI runs the version. The facade
    sends the `Z` form for every instant it renders, so without this every claim, episode,
    answer and delta would fail to hydrate on 3.10 while passing on 3.13, which is the
    shape of bug a test run on one version cannot see.

    `server/tools.py:_timestamp` makes the same conversion for the same reason, as do the
    two importers in `compat/`. This is that rule applied where the wire meets the
    dataclasses.
    """
    if value is None:
        return None
    text = str(value)
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


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
    return Scope(body["tenant"], body["user"], body["agent"], body["session"])


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
    at their dataclass defaults.

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
        unextracted=body["unextracted"],
        llm_calls=body["llm_calls"],
        latency_ms=body["latency_ms"],
        deferred=body["deferred"],
    )


def provenance(body: dict[str, Any]) -> Provenance:
    return Provenance(
        claim=claim(body["memory"]),
        episodes=[episode(e) for e in body["sources"]],
        derivation=Derivation(body["derivation"]),
        extractor=body["extractor"] or "",
        superseded=[claim(c) for c in body["superseded"]],
    )


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
