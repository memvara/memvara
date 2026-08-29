"""`RemoteMemvara`: the library's own API, served by a hosted deployment.

Not a `Store` and not a subclass of `Memvara`. The engine runs server-side; this class
turns a method call into one `/v1` request and hydrates what comes back into the same
dataclasses the local engine returns, so calling code cannot tell which it holds.

**What is absent is absent, not raising.** `reembed`, `pending_extraction`, `reextract`
and `reset` have no endpoint, so they are not methods here. A caller reaching for one gets
an `AttributeError` at the call site and mypy catches it before that — where a method that
raised would compile, ship, and fail in production.

The same rule decides which *arguments* exist. `recall()` takes no `with_ids` and no
`header`, because `POST /v1/recall` returns a rendered string and an `empty` flag and
carries no ids at all; a `with_ids=True` that quietly returned no ids would be worse than
the `TypeError` a caller gets instead. `get_all()` takes no `memory_types`, because
`GET /v1/memories` has no such filter and FastAPI drops an unknown query parameter in
silence — the caller would get an unfiltered page with nothing saying the filter was
ignored. `budget` is the one exception and it is a refusal rather than an omission: it is
in the signature so that `None` (what every current caller passes) works, and a value
raises, because a budget silently ignored is an oversized prompt with no signal.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Collection, Sequence

from ..redact import Redactor
from ..retrieve import EpisodeResult, Path, Retrieved
from ..types import Answer, Claim, Delta, MemoryType, Provenance, Scope
from . import hydrate
from .client import DEFAULT_TIMEOUT, HttpClient
from .creds import resolve
from .errors import NotFound


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _types(memory_types: Sequence[MemoryType | str] | None) -> list[str] | None:
    """Memory types as the wire spells them. `None` stays `None` so the transport drops
    it, which is what asks for no filter at all."""
    if memory_types is None:
        return None
    return [t.value if isinstance(t, MemoryType) else str(t) for t in memory_types]


def _states(states: Collection[str] | None) -> list[str] | None:
    return None if states is None else list(states)


def _hit(body: dict[str, Any]) -> Retrieved:
    """One search result, as whichever of the two kinds it is.

    `kind` is the discriminator here, because types do not cross a wire. An episode hit
    is an `EpisodeResult` and carries its turn under `episode`; a claim hit is a `Result`
    and carries its memory under `memory`. Handing an episode hit to `hydrate.result`
    would raise on the missing `memory` key, which is the failure a caller would see the
    moment they passed `include_episodes=True`.
    """
    if body["kind"] == "episode":
        return EpisodeResult(episode=hydrate.episode(body["episode"]),
                             score=body["score"],
                             explain=hydrate.explanation(body["ranking"]))
    return hydrate.result(body)


def _sent(body: dict[str, Any]) -> dict[str, Any]:
    """A request body with its unset fields removed.

    The facade's request models are `extra="forbid"`, so a field is either one it knows
    or a 422 — but an explicit `null` for a field that has a meaningful default is a
    different request from omitting it, and omitting is the one that means "unset".
    """
    return {k: v for k, v in body.items() if v is not None}


class RemoteMemvara:
    """The library's API against a hosted deployment.

    Constructing one performs no network call. It resolves a credential, builds a
    connection pool and stops — the same rule `Memvara.__init__` follows for `llm=`, and
    for the same reason: spending money or failing over a network as a side effect of a
    constructor is not something a library does behind your back. The first request is
    the first method call.
    """

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None,
                 tenant: str = "default", user: str | None = None,
                 agent: str | None = None, session: str | None = None,
                 timeout: float = DEFAULT_TIMEOUT,
                 redactor: Redactor | None = None) -> None:
        key, url = resolve(api_key, base_url)
        self._http = HttpClient(key, url, timeout=timeout)
        #: The scope this client narrows to. `tenant` is held for `default_scope`'s sake
        #: and never sent: the facade resolves it from the bearer token, and a `tenant`
        #: parameter a caller could set would be a request to be trusted about identity.
        self.default_scope = Scope(tenant, user, agent, session)
        #: Rewrites text on its way out, or None. Runs here rather than server-side on
        #: purpose: redaction that happens after the text has left the process is not
        #: redaction.
        self.redactor = redactor

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "RemoteMemvara":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<RemoteMemvara {self.default_scope.key()}>"

    def _params(self, **extra: Any) -> dict[str, Any]:
        """Scope on every call, plus whatever this call adds. `None` values are dropped
        by the transport rather than sent as empty strings."""
        scope = self.default_scope
        return {"user": scope.user, "agent": scope.agent, "session": scope.session,
                **extra}

    # -- service -------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """Whether the deployment is answering. The one route that takes no credential,
        and it discloses nothing else — a 200 here is not a promise that a read will
        succeed, because it does not touch the store. `stats()` is the check that does."""
        return self._http.request("GET", "/v1/health")

    def whoami(self) -> dict[str, Any]:
        """What the presented credential authorizes: its scope, its privilege, its
        expiry. Answered from the token alone, so it never reports on the store."""
        return self._http.request("GET", "/v1/whoami", params=self._params())

    def stats(self) -> dict[str, int]:
        """Row counts for the whole tenant, whatever this client's scope.

        The tenant counts alone, which is what `Memvara.stats()` returns and what every
        caller of it reads — `live_claims`, `claims`, `episodes`, `embeddings`. The
        `/v1/stats` envelope wraps them beside `visible` (what *this* scope can read),
        `extractor` and `read_only`; returning the envelope instead would make
        `stats()["claims"]` a `KeyError` against a hosted deployment and a number against
        a local one.
        """
        body = self._http.request("GET", "/v1/stats", params=self._params())
        return dict(body["tenant_counts"])

    def connectivity(self) -> dict[str, int]:
        """`live_claims` and `joinable_claims`, or `{}` when the deployment does not
        report them.

        `{}` is not the same as a store with nothing in it, and the distinction is the
        whole reason the local method documents it: an empty store answers with two zeros,
        a backend that cannot measure says nothing at all, and a caller reading a missing
        key as zero would report a star it never measured.
        """
        body = self.stats()
        if "joinable_claims" not in body or "live_claims" not in body:
            return {}
        return {"live_claims": body["live_claims"],
                "joinable_claims": body["joinable_claims"]}

    # -- reading -------------------------------------------------------------

    def search(self, query: str, *, k: int = 10, min_score: float = 0.0,
               as_of: datetime | None = None, valid_at: datetime | None = None,
               known_at: datetime | None = None,
               states: Collection[str] | None = None,
               include_invalidated: bool | None = None,
               memory_types: Sequence[MemoryType | str] | None = None,
               include_episodes: bool = False) -> list[Retrieved]:
        """Hybrid retrieval, with the ranking explanation attached.

        A POST for a read, as the facade defines it: the query is text somebody wrote,
        and a GET would put it in the request line, where it lands in access logs, proxy
        logs and `Referer` headers.
        """
        body = self._http.request(
            "POST", "/v1/search", params=self._params(),
            json=_sent({"query": query, "k": k, "min_score": min_score,
                        "as_of": _iso(as_of), "valid_at": _iso(valid_at),
                        "known_at": _iso(known_at), "states": _states(states),
                        "include_invalidated": include_invalidated,
                        "memory_types": _types(memory_types),
                        "include_episodes": include_episodes}))
        return [_hit(h) for h in body["results"]]

    def recall(self, query: str, *, k: int = 8, min_score: float = 0.0,
               memory_types: Sequence[MemoryType | str] | None = None,
               include_episodes: bool = False, budget: int | None = None) -> str:
        """Retrieval already formatted for a system prompt: prose, not rows.

        Narrower than `search` in the two ways the facade is narrow, and neither is an
        oversight. No time travel, because a prompt assembled out of what was believed
        last March is a hazard rather than an audit trail. No `states`, because rendering
        a retired record into a system prompt is an un-delete: the agent acts on a fact
        that was withdrawn.

        `budget` is refused rather than ignored. `POST /v1/recall` renders server-side
        and takes no budget, and this client cannot re-derive the local truncation from
        the finished string without writing a second implementation of it that can
        disagree. A caller who asked for a ceiling and silently did not get one ships an
        oversized prompt with nothing to notice it by.
        """
        if budget is not None:
            raise ValueError(
                "recall(budget=...) is not available against a hosted deployment: "
                "POST /v1/recall renders the block server-side and takes no budget. Use "
                "a smaller k, or render your own block from search().")
        body = self._http.request(
            "POST", "/v1/recall", params=self._params(),
            json=_sent({"query": query, "k": k, "min_score": min_score,
                        "memory_types": _types(memory_types),
                        "include_episodes": include_episodes}))
        return str(body["text"])

    def get(self, claim_id: str) -> Claim | None:
        """One memory, or None. None rather than raising for an id in another tenant as
        well as one that never existed — the facade gives the same answer for both so
        that this cannot be used to test whether an id exists elsewhere."""
        try:
            body = self._http.request("GET", f"/v1/memories/{claim_id}",
                                      params=self._params())
        except NotFound:
            return None
        return hydrate.claim(body)

    def get_all(self, *, states: Collection[str] | None = None,
                include_invalidated: bool | None = None,
                limit: int = 100, offset: int = 0,
                as_of: datetime | None = None, valid_at: datetime | None = None,
                known_at: datetime | None = None) -> list[Claim]:
        """One page of the memories visible at this scope, newest first.

        Paged, and the local method is not: `GET /v1/memories` materializes the page
        server-side and caps `limit` at 500, so a scope holding a million claims cannot
        be pulled across a network in one call. `count()` is how many matched.
        """
        body = self._http.request(
            "GET", "/v1/memories",
            params=self._params(limit=limit, offset=offset, states=_states(states),
                                include_invalidated=include_invalidated,
                                as_of=_iso(as_of), valid_at=_iso(valid_at),
                                known_at=_iso(known_at)))
        return [hydrate.claim(c) for c in body["memories"]]

    def count(self, *, states: Collection[str] | None = None,
              include_invalidated: bool | None = None,
              as_of: datetime | None = None, valid_at: datetime | None = None,
              known_at: datetime | None = None) -> int:
        """How many memories match, without fetching them: one row asked for, `total`
        read off the page."""
        body = self._http.request(
            "GET", "/v1/memories",
            params=self._params(limit=1, states=_states(states),
                                include_invalidated=include_invalidated,
                                as_of=_iso(as_of), valid_at=_iso(valid_at),
                                known_at=_iso(known_at)))
        return int(body["total"])

    def history(self, subject: str, predicate: str, *,
                as_of: datetime | None = None, valid_at: datetime | None = None,
                known_at: datetime | None = None) -> list[Claim]:
        """Every value one fact slot has ever held, oldest first.

        Neither clock filters unless one is sent, here and on `why` alone: a timeline
        whose default was now would drop every superseded value, which is the whole
        content of a timeline.
        """
        body = self._http.request(
            "GET", "/v1/history",
            params=self._params(subject=subject, predicate=predicate, as_of=_iso(as_of),
                                valid_at=_iso(valid_at), known_at=_iso(known_at)))
        return [hydrate.claim(c) for c in body["timeline"]]

    def why(self, claim_id: str, *, as_of: datetime | None = None,
            valid_at: datetime | None = None,
            known_at: datetime | None = None) -> Provenance | None:
        """Why this memory is believed: its source turns, what produced it, what it
        replaced. `None` for an id that is not visible here, matching `Memvara.why` —
        the facade answers 404 for a missing id and for one in another tenant alike."""
        try:
            body = self._http.request(
                "GET", f"/v1/memories/{claim_id}/why",
                params=self._params(as_of=_iso(as_of), valid_at=_iso(valid_at),
                                    known_at=_iso(known_at)))
        except NotFound:
            return None
        return hydrate.provenance(body)

    def ask(self, question: str, *, at: datetime | None = None, k: int = 3,
            min_score: float = 0.0) -> Answer:
        """What was true then, and what this store would have *told you* then.

        The two differing is the finding rather than a fault: it means the record was
        corrected after the moment asked about, so the answer somebody acted on is not
        the answer they would get today.
        """
        body = self._http.request(
            "POST", "/v1/ask", params=self._params(),
            json=_sent({"question": question, "at": _iso(at), "k": k,
                        "min_score": min_score}))
        return hydrate.answer(body)

    def since(self, when: datetime) -> Delta:
        """What arrived and what left while the caller was away.

        A supersession appears in both halves — the replaced value under `gone`, its
        replacement under `added` — so a client that syncs only `added` ends up holding
        both.
        """
        body = self._http.request("GET", "/v1/since",
                                  params=self._params(since=_iso(when)))
        return hydrate.delta(body)

    def produced(self, episode_id: str, *, as_of: datetime | None = None,
                 valid_at: datetime | None = None,
                 known_at: datetime | None = None) -> list[Claim]:
        """Everything derived from one stored turn: `why`, backwards.

        Closed values included, so read `state` rather than expecting the present tense.
        An empty list is the answer for a turn that does not exist, one that produced
        nothing, and one whose memories belong elsewhere — the same non-answer for all
        three, because telling them apart confirms an id.
        """
        body = self._http.request(
            "GET", f"/v1/episodes/{episode_id}/produced",
            params=self._params(as_of=_iso(as_of), valid_at=_iso(valid_at),
                                known_at=_iso(known_at)))
        return [hydrate.claim(c) for c in body["memories"]]

    def neighborhood(self, entity: str, *, depth: int = 2, k: int = 10,
                     min_hops: int = 1, predicates: Sequence[str] | None = None,
                     as_of: datetime | None = None, valid_at: datetime | None = None,
                     known_at: datetime | None = None,
                     min_score: float = 0.0) -> list[Path]:
        """Every relation reachable from one entity, ranked, each chain inspectable.

        `depth` sets what the call costs: a hop is worth roughly three of the one before
        it, and the deployment prices each depth separately.
        """
        body = self._http.request(
            "GET", "/v1/neighborhood",
            params=self._params(entity=entity, depth=depth, k=k, min_hops=min_hops,
                                predicates=list(predicates) if predicates else None,
                                min_score=min_score, as_of=_iso(as_of),
                                valid_at=_iso(valid_at), known_at=_iso(known_at)))
        return [hydrate.path(p) for p in body["paths"]]

    def paths_between(self, source: str, target: str, *, depth: int = 3, k: int = 3,
                      predicates: Sequence[str] | None = None,
                      as_of: datetime | None = None, valid_at: datetime | None = None,
                      known_at: datetime | None = None,
                      min_score: float = 0.0) -> list[Path]:
        """How two entities are connected, if they are. An empty list means not within
        `depth` hops at the clocks asked about, which is a different answer from an empty
        `search`: that one means nothing *resembled* the question."""
        body = self._http.request(
            "GET", "/v1/paths",
            params=self._params(source=source, target=target, depth=depth, k=k,
                                predicates=list(predicates) if predicates else None,
                                min_score=min_score, as_of=_iso(as_of),
                                valid_at=_iso(valid_at), known_at=_iso(known_at)))
        return [hydrate.path(p) for p in body["paths"]]

    def standing(self, *, k: int | None = None) -> list[Claim]:
        """Every standing preference in this scope, unranked, ordered by confidence then
        recency then id.

        Deliberately not a search, and the reason was measured: a client wanting standing
        preferences had to invent a sentence to rank them against, and a rule stored at
        confidence 1.00 scored zero against it and never reached a session. The local
        engine has no such method — `get_all` has no `memory_types` filter, so a local
        caller pages the scope and filters in Python. This does it server-side.
        """
        body = self._http.request("GET", "/v1/standing", params=self._params(limit=k))
        return [hydrate.claim(c) for c in body["memories"]]
