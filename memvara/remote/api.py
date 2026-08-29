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

Two write divergences that are real and documented rather than hidden. `consolidate()`
returns a job handle rather than per-operation counts, because the endpoint answers 202
before the pass starts. There is no `prove_erased()`, because `POST /v1/erasures` returns
its per-table counts as evidence inside the erasure response itself.
"""
from __future__ import annotations

from copy import copy
from datetime import datetime
from typing import Any, Collection, Literal, Mapping, Sequence, overload

from ..redact import CLAIM_OBJECT, CLAIM_SUBJECT, CLAIM_TEXT, EPISODE, Redactor
from ..retrieve import EpisodeResult, Path, Retrieved
from ..types import (
    Answer, Claim, Delta, Episode, MemoryType, Provenance, Result, Scope, WriteReceipt,
    closure,
)
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


def _type(memory_type: MemoryType | str | None) -> str | None:
    """One memory type as the wire spells it, or `None` to let the predicate's registered
    type stand."""
    if memory_type is None:
        return None
    return memory_type.value if isinstance(memory_type, MemoryType) else str(memory_type)


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

    def scope(self, *, user: str | None = None, agent: str | None = None,
              session: str | None = None) -> "ScopedRemoteMemvara":
        """A view bound to a narrower scope, with no way back out.

        The narrowing is against this client's own scope, and the facade enforces the
        same rule again from the credential — naming an agent or a session requires
        naming a user, because an agent under *every* user is a read across users dressed
        up as a narrowing.
        """
        current = self.default_scope
        narrowed = Scope(
            current.tenant,
            user if user is not None else current.user,
            agent if agent is not None else current.agent,
            session if session is not None else current.session,
        )
        return ScopedRemoteMemvara(self, narrowed)

    def _at(self, scope: Scope) -> "RemoteMemvara":
        """A twin bound to a different scope, sharing this client's transport.

        Sharing rather than dialling a second pool, because the two are the same
        deployment on the same credential. Closing either closes both, which is the
        honest reading of one pool with two handles on it.
        """
        twin = copy(self)
        twin.default_scope = scope
        return twin

    def _params(self, **extra: Any) -> dict[str, Any]:
        """Scope on every call, plus whatever this call adds. `None` values are dropped
        by the transport rather than sent as empty strings."""
        scope = self.default_scope
        return {"user": scope.user, "agent": scope.agent, "session": scope.session,
                **extra}

    def _redact(self, text: str | None, field: str) -> str | None:
        """Apply the policy to one string on its way out, or pass it through.

        The `Redactor` protocol is `redact(text, *, field=..., scope=...)`. `field` says
        which of `redact.FIELDS` is being offered, so a deployment can be aggressive on
        raw turns and conservative on claim objects; `scope` is what makes "redact for EU
        tenants" expressible at all. Calling the policy with the text alone would apply
        one branch of it to everything.

        Here rather than server-side, and that is the point of the seam: redaction that
        happens after the text has left the process is not redaction.
        """
        if self.redactor is None or text is None:
            return text
        return self.redactor.redact(text, field=field, scope=self.default_scope)

    def _turn(self, message: Episode | Mapping[str, Any] | str) -> dict[str, Any]:
        """One conversation turn as `Message` spells it, with its content redacted.

        Keys the model does not declare go into `metadata` rather than beside it: the
        facade's request models are `extra="forbid"`, so a stray key is a 422 and not a
        field somebody quietly loses. `Memvara.add` does the same fold locally.
        """
        if isinstance(message, str):
            return _sent({"content": self._redact(message, EPISODE)})
        if isinstance(message, Episode):
            return _sent({"role": message.role,
                          "content": self._redact(message.content, EPISODE),
                          "ts": _iso(message.ts), "metadata": dict(message.meta)})
        known = {"role", "content", "ts", "metadata"}
        meta = dict(message.get("metadata") or {})
        meta.update({k: v for k, v in message.items() if k not in known})
        return _sent({"role": message.get("role"),
                      "content": self._redact(message["content"], EPISODE),
                      "ts": _iso(message.get("ts")), "metadata": meta})

    def _cite(self, sources: Sequence[Episode | Mapping[str, Any] | str] | None,
              ) -> tuple[list[str], list[dict[str, Any]]]:
        """Split what a caller cited into ids already stored and turns to store now.

        `Memvara.remember` takes one mixed sequence; the facade takes two fields, because
        over HTTP the two are different acts — `source_ids` cites rows the store already
        holds, `sources` writes new turns in the same transaction as the fact. A string
        is an id and anything else is a turn, which is the same rule `Memvara._cite`
        applies.
        """
        ids = [s for s in sources or [] if isinstance(s, str)]
        turns = [self._turn(s) for s in sources or [] if not isinstance(s, str)]
        return ids, turns

    def _end(self, body: dict[str, Any]) -> list[Claim]:
        """`POST /v1/end`: close what this addresses on the world clock.

        One helper for the three callers, because they differ in what they address and
        not in what they record. `end` addresses either mode, `delete(close="ended")`
        addresses one memory, and `forget(close="ended")` addresses a slot; all three
        state that the value was true and the world moved.
        """
        out = self._http.request("POST", "/v1/end", params=self._params(),
                                 json=_sent(body), write=True)
        return [hydrate.claim(c) for c in out["ended"]]

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

    def service(self, *, attempts: int | None = None,
                timeout: float | None = None) -> dict[str, Any]:
        """The whole `/v1/stats` envelope: what the deployment is, not just what it holds.

        `{scope, visible, tenant_counts, extractor, read_only}` — the same request
        `stats()` makes, returned whole instead of unwrapped. Both exist on purpose.
        `stats()` returns `tenant_counts` alone because that is what `Memvara.stats()`
        returns and what every caller of it reads, so unwrapping keeps
        `stats()["claims"]` a number against either engine. This one is for the caller
        that needs the three fields unwrapping drops, and there is exactly one: a server
        deciding at startup what it can say about itself.

        `read_only` is the field worth naming. It is what the presented *credential*
        authorizes, not a setting, so a server that lists its write tools without
        consulting it advertises tools the deployment will refuse — mid-conversation, as a
        403, to a model that cannot act on it. `whoami()` reports it too, from the token
        alone; this route answers it beside the counts, so one request settles both.

        `attempts` and `timeout` override the client's for this one call, because the
        caller that needs this is a server deciding what to say about itself before it
        answers anything, and it wants a cheap answer or none. The client's own three
        attempts at a 30-second timeout, plus backoff, is a minute and a half of silent
        startup before a hanging deployment degrades to the safe default — which is the
        opposite of what degrading gracefully is for.
        """
        return dict(self._http.request("GET", "/v1/stats", params=self._params(),
                                       attempts=attempts, timeout=timeout))

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

    # The same three variants as `Memvara.search`, and they are what makes "calling code
    # cannot tell which it holds" true of the type as well as of the value. Without them
    # `mem.search(q)` types as `list[Retrieved]` here and `list[Result]` locally, so the
    # same expression reading `.claim` off a row checks against one engine and not the
    # other. `include_episodes=False` returns claim hits only -- `_hit` builds an
    # `EpisodeResult` only for a row the facade marked `kind: "episode"`, and it sends
    # none when it was not asked for them.
    @overload
    def search(self, query: str, *, k: int = ..., min_score: float = ...,
               as_of: datetime | None = ..., valid_at: datetime | None = ...,
               known_at: datetime | None = ..., states: Collection[str] | None = ...,
               include_invalidated: bool | None = ...,
               memory_types: Sequence[MemoryType | str] | None = ...,
               include_episodes: Literal[False] = ...) -> list[Result]: ...

    @overload
    def search(self, query: str, *, k: int = ..., min_score: float = ...,
               as_of: datetime | None = ..., valid_at: datetime | None = ...,
               known_at: datetime | None = ..., states: Collection[str] | None = ...,
               include_invalidated: bool | None = ...,
               memory_types: Sequence[MemoryType | str] | None = ...,
               include_episodes: Literal[True]) -> list[Retrieved]: ...

    @overload
    def search(self, query: str, *, k: int = ..., min_score: float = ...,
               as_of: datetime | None = ..., valid_at: datetime | None = ...,
               known_at: datetime | None = ..., states: Collection[str] | None = ...,
               include_invalidated: bool | None = ...,
               memory_types: Sequence[MemoryType | str] | None = ...,
               include_episodes: bool) -> list[Retrieved]: ...

    def search(self, query: str, *, k: int = 10, min_score: float = 0.0,
               as_of: datetime | None = None, valid_at: datetime | None = None,
               known_at: datetime | None = None,
               states: Collection[str] | None = None,
               include_invalidated: bool | None = None,
               memory_types: Sequence[MemoryType | str] | None = None,
               include_episodes: bool = False) -> list[Any]:
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

    # -- writing -------------------------------------------------------------

    def add(self, messages: Any, *, role: str = "user",
            ts: datetime | None = None) -> WriteReceipt:
        """Ingest conversation turns and extract whatever in them is durable.

        **Read the receipt. A 200 is not a promise that anything was remembered.** A
        non-zero `unextracted` beside an empty `added` is a successful-looking write that
        stored nothing, and the usual cause is a deployment with no extraction model —
        `stats()` reports `extractor` as `fast-path-only` there. `remember()` is the
        route to use when your application already knows the answer.
        """
        payload: Any
        if isinstance(messages, str):
            payload = self._redact(messages, EPISODE)
        elif isinstance(messages, (Episode, Mapping)):
            payload = [self._turn(messages)]
        else:
            payload = [self._turn(m) for m in messages]
        body = self._http.request(
            "POST", "/v1/memories", params=self._params(),
            json=_sent({"messages": payload, "role": role, "ts": _iso(ts)}),
            write=True)
        return hydrate.receipt(body)

    def remember(self, subject: str, predicate: str, obj: str, *,
                 confidence: float = 1.0,
                 memory_type: MemoryType | str | None = None, polarity: int = 1,
                 valid_from: datetime | None = None, valid_to: datetime | None = None,
                 recorded_at: datetime | None = None,
                 sources: Sequence[Episode | Mapping[str, Any] | str] | None = None,
                 text: str | None = None, extractor: str = "api",
                 **meta: Any) -> WriteReceipt:
        """State one exact fact, skipping extraction entirely.

        Asserting into an occupied slot is the correction, and it is not an update: the
        old value is closed out and comes back under the receipt's `closed`, the new one
        under `added`, and both keep their ids forever. Reuse a predicate the store
        already knows — contradiction handling is an exact match on the slot, so a
        synonym opens a second one instead of correcting the first.
        """
        ids, turns = self._cite(sources)
        body = {
            "subject": self._redact(subject, CLAIM_SUBJECT),
            "predicate": predicate,
            "object": self._redact(obj, CLAIM_OBJECT),
            "text": self._redact(text, CLAIM_TEXT),
            "confidence": confidence, "polarity": polarity, "extractor": extractor,
            "memory_type": _type(memory_type),
            "valid_from": _iso(valid_from), "valid_to": _iso(valid_to),
            "recorded_at": _iso(recorded_at),
            "source_ids": ids, "sources": turns, "metadata": meta,
        }
        return hydrate.receipt(self._http.request(
            "POST", "/v1/facts", params=self._params(), json=_sent(body), write=True))

    def supersede(self, old_claim_id: str, subject: str, predicate: str, obj: str, *,
                  at: datetime | None = None, close: str = "ended",
                  confidence: float = 1.0,
                  memory_type: MemoryType | str | None = None, polarity: int = 1,
                  valid_from: datetime | None = None, valid_to: datetime | None = None,
                  recorded_at: datetime | None = None,
                  sources: Sequence[Episode | Mapping[str, Any] | str] | None = None,
                  text: str | None = None, extractor: str = "api",
                  **meta: Any) -> WriteReceipt:
        """Replace a named memory with a new value, recording that that is what happened.

        `remember()` first: asserting into an occupied slot already closes the old value
        out and already records what it replaced. Two things bring you here. Naming the
        replaced memory explicitly, which is what importing somebody else's mutation log
        needs. And saying **which clock stops**, which only the caller can know: `ended`
        means the old value was true until `at` and is not any more, `retired` means the
        record was wrong and belief in it stops there with its valid interval left
        exactly as written.

        `close` is validated and forwarded, never defaulted on the caller's behalf. A
        mutation log records that a value changed and not which of the two it was, so
        restating `"ended"` here would file every correction as a world event.

        The new value is given as a triple rather than as a `Claim`, which is where this
        diverges from `Memvara.supersede`. The endpoint takes a fact body, and building a
        `Claim` here to take it apart again would put this layer in the business of
        inventing ids and timestamps the server is about to overwrite.
        """
        ids, turns = self._cite(sources)
        body = {
            "subject": self._redact(subject, CLAIM_SUBJECT),
            "predicate": predicate,
            "object": self._redact(obj, CLAIM_OBJECT),
            "text": self._redact(text, CLAIM_TEXT),
            "at": _iso(at), "close": closure(close),
            "confidence": confidence, "polarity": polarity, "extractor": extractor,
            "memory_type": _type(memory_type),
            "valid_from": _iso(valid_from), "valid_to": _iso(valid_to),
            "recorded_at": _iso(recorded_at),
            "source_ids": ids, "sources": turns, "metadata": meta,
        }
        return hydrate.receipt(self._http.request(
            "POST", f"/v1/memories/{old_claim_id}/supersede", params=self._params(),
            json=_sent(body), write=True))

    def forget(self, subject: str, predicate: str, *, at: datetime | None = None,
               close: str = "retired") -> list[Claim]:
        """Close every value one fact slot currently answers with.

        **Routes on `close`, for `delete`'s reason and with `delete`'s consequences.**
        `"retired"` says we stopped believing those values and goes to `POST /v1/forget`.
        `"ended"` says they were true and the world moved on, and goes to the slot form of
        `POST /v1/end`. The facade has no `close` field on `/v1/forget`, so a client that
        accepted the argument and posted there anyway would file every ending as a
        retirement — and `server/tools.py` calls exactly that: `forget(..., close="ended")`
        is how the `memory_end` tool closes a slot.

        Nothing is removed either way. Every closed value stays readable through
        `history()`, through `states=["retired"]`, and through any `known_at` query that
        predates the call. Real removal is `erase()`.

        It reaches **downward**, and a search with the same credential does not: a
        user-scoped call also closes values written inside that user's agents and
        sessions. The returned list is what it actually reached.
        """
        if closure(close) == "ended":
            return self._end({"subject": subject, "predicate": predicate, "at": _iso(at)})
        body = self._http.request(
            "POST", "/v1/forget", params=self._params(),
            json=_sent({"subject": subject, "predicate": predicate, "at": _iso(at)}),
            write=True)
        return [hydrate.claim(c) for c in body["retired"]]

    def delete(self, claim_id: str, *, at: datetime | None = None,
               close: str = "retired") -> bool:
        """Close one memory by id.

        **Routes on `close`, and the two destinations are not interchangeable.**
        `"retired"` says the record was wrong and goes to `DELETE /v1/memories/{id}`.
        `"ended"` says the world moved on from something true and goes to
        `POST /v1/end`. Sending one to the other's route records a false reason for the
        change, and nothing downstream can detect it: both leave a closed memory, and the
        row that says which happened is the one being written. `memvara/types.py` calls
        this the one mistake in this library that cannot be found by reading the data
        afterwards.

        `close` is validated through `types.closure()` so a typo raises with both
        readings spelled out, rather than falling through to the default and recording
        the wrong one in silence.

        `False` means nothing moved. It is the answer for an id that never existed and
        for one belonging to another tenant alike, so this cannot be used to test whether
        an id exists elsewhere.
        """
        if closure(close) == "ended":
            return self.end(claim_id=claim_id, at=at)
        body = self._http.request("DELETE", f"/v1/memories/{claim_id}",
                                  params=self._params(), write=True)
        return bool(body["retired"])

    def end(self, *, claim_id: str | None = None, subject: str | None = None,
            predicate: str | None = None, at: datetime | None = None) -> bool:
        """Close a fact that stopped being true, with nothing replacing it.

        Exactly one addressing mode: `claim_id` for one memory, or `predicate` (with
        `subject`, default `"user"`) for every current value in that slot. Both or
        neither is a `TypeError`, deliberately — the two have different blast radii and a
        silent default on that choice is not a convenience.

        `at` is when the fact stopped being true, and it defaults to now, which is right
        only when it stopped just now. An instant before the fact began is clamped to its
        start rather than inverting the interval.
        """
        if (claim_id is None) == (predicate is None):
            raise TypeError(
                "end() needs exactly one of: claim_id, to end one memory, or predicate "
                "(with optional subject), to end every current value of that fact.")
        body: dict[str, Any] = {"at": _iso(at)}
        if claim_id is not None:
            body["memory_id"] = claim_id
        else:
            body["subject"] = subject or "user"
            body["predicate"] = predicate
        return bool(self._end(body))

    # -- erasure -------------------------------------------------------------

    def erase(self, claim_id: str, *, sources: bool = False) -> bool:
        """Remove one memory for real. Irreversible, and the only thing here that is.

        Everything under `delete`, `forget` and `end` closes a value out and leaves the
        text readable through `history()` and `known_at`. This takes the claim, its
        embeddings and its index entries with it. `sources=True` also erases the turns it
        came from that no surviving memory still cites — right for a memory that *is* its
        source text, wrong for a fact extracted from a turn that held much else besides.

        `False` means no such memory was visible here, which is a 200 rather than a 404
        for the reason every id-addressed route refuses to distinguish gone from not
        yours. Needs an `admin` credential.
        """
        body = self._http.request(
            "POST", "/v1/erasures", params=self._params(),
            json={"memory_id": claim_id, "sources": sources}, write=True)
        return bool(body["erased"])

    def purge(self, *, confirm_tenant: str | None = None) -> dict[str, int]:
        """Erase everything at this client's scope and beneath it, and report what went.

        The counts are per-table rows removed, measured by the store rather than
        assembled by the facade, which is what a deletion request has to be answered
        with. Erasing a user takes their agents and sessions with them.

        A client bound to no user is asking for the whole tenant, and the facade refuses
        that without `confirm_tenant` equal to the tenant's own name — an empty scope
        object is too easy to send by accident. `confirm_tenant` cannot widen anything:
        the credential decides the tenant, and any other value is refused.
        """
        scope = self.default_scope
        body = self._http.request(
            "POST", "/v1/erasures", params=self._params(),
            json=_sent({"scope": _sent({"user": scope.user, "agent": scope.agent,
                                        "session": scope.session}),
                        "confirm_tenant": confirm_tenant}),
            write=True)
        return dict(body["counts"] or {})

    # -- maintenance ---------------------------------------------------------

    def consolidate(self) -> dict[str, Any]:
        """Start a maintenance pass over the whole tenant: decay, merge, promote.

        Returns a **job**, not counts, and that is the divergence from
        `Memvara.consolidate()`. The endpoint answers 202 before the work starts, because
        a real store takes seconds to walk and holding the request open would have every
        client time out and retry into the same write lock. Poll the job's `status`: it
        becomes `succeeded` with per-operation counts in `result`, or `failed` with the
        exception in `error`. A 202 is not a promise the work succeeded.

        Needs an `admin` credential, and the pass covers the whole tenant whatever scope
        this client narrows to.
        """
        return self._http.request("POST", "/v1/maintenance/consolidate",
                                  params=self._params(), write=True)


class ScopedRemoteMemvara:
    """A `RemoteMemvara` with its scope already filled in.

    The twin of `ScopedMemvara`, and it holds the same security property: **a handler
    holding one of these has no argument with which to address another tenant, another
    user, another agent or another session.** The scope is bound once, at construction,
    and no method here takes `tenant`, `user`, `agent` or `session`. That is what makes
    forgetting to check a scope impossible to exploit from a handler, rather than
    something every handler has to remember.

    Against a hosted deployment the credential binds the tenant a second time, from the
    other side: the facade resolves it from the bearer token and there is no request
    parameter anywhere on `/v1` that names one. So the scope here can only narrow inside
    what the token already authorizes, and a narrowing that tried to widen is refused by
    the facade rather than merely absent from this class.
    """

    __slots__ = ("_mem", "_scope")

    def __init__(self, mem: RemoteMemvara, scope: Scope) -> None:
        #: The client every call goes through: a twin of `mem` sharing its transport and
        #: bound to `scope`, so the scope arrives on the wire without any method here
        #: having to pass it.
        self._mem = mem._at(scope)
        self._scope = scope

    @property
    def memvara(self) -> RemoteMemvara:
        """The client underneath.

        Public for `ScopedMemvara.memvara`'s reason: a server layer holds one of these
        per request, needs something off the real object, finds no accessor and reaches
        for the private attribute instead — which is an undocumented API with a
        misleading name rather than encapsulation.

        Not a way around the scope. The credential binds the tenant, so what comes back
        cannot address another one either.
        """
        return self._mem

    @property
    def scope(self) -> Scope:
        """The scope this view is bound to. An attribute rather than an argument, which
        is the whole point — `server/tools.py` reads it to report where it is."""
        return self._scope

    def __repr__(self) -> str:
        return f"<ScopedRemoteMemvara {self._scope.key()}>"

    # -- service -------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        return self._mem.health()

    def whoami(self) -> dict[str, Any]:
        return self._mem.whoami()

    def stats(self) -> dict[str, int]:
        return self._mem.stats()

    def connectivity(self) -> dict[str, int]:
        return self._mem.connectivity()

    # -- reading -------------------------------------------------------------

    # The same three variants as `Memvara.search`, and they are what makes "calling code
    # cannot tell which it holds" true of the type as well as of the value. Without them
    # `mem.search(q)` types as `list[Retrieved]` here and `list[Result]` locally, so the
    # same expression reading `.claim` off a row checks against one engine and not the
    # other. `include_episodes=False` returns claim hits only -- `_hit` builds an
    # `EpisodeResult` only for a row the facade marked `kind: "episode"`, and it sends
    # none when it was not asked for them.
    @overload
    def search(self, query: str, *, k: int = ..., min_score: float = ...,
               as_of: datetime | None = ..., valid_at: datetime | None = ...,
               known_at: datetime | None = ..., states: Collection[str] | None = ...,
               include_invalidated: bool | None = ...,
               memory_types: Sequence[MemoryType | str] | None = ...,
               include_episodes: Literal[False] = ...) -> list[Result]: ...

    @overload
    def search(self, query: str, *, k: int = ..., min_score: float = ...,
               as_of: datetime | None = ..., valid_at: datetime | None = ...,
               known_at: datetime | None = ..., states: Collection[str] | None = ...,
               include_invalidated: bool | None = ...,
               memory_types: Sequence[MemoryType | str] | None = ...,
               include_episodes: Literal[True]) -> list[Retrieved]: ...

    @overload
    def search(self, query: str, *, k: int = ..., min_score: float = ...,
               as_of: datetime | None = ..., valid_at: datetime | None = ...,
               known_at: datetime | None = ..., states: Collection[str] | None = ...,
               include_invalidated: bool | None = ...,
               memory_types: Sequence[MemoryType | str] | None = ...,
               include_episodes: bool) -> list[Retrieved]: ...

    def search(self, query: str, *, k: int = 10, min_score: float = 0.0,
               as_of: datetime | None = None, valid_at: datetime | None = None,
               known_at: datetime | None = None,
               states: Collection[str] | None = None,
               include_invalidated: bool | None = None,
               memory_types: Sequence[MemoryType | str] | None = None,
               include_episodes: bool = False) -> list[Any]:
        return self._mem.search(query, k=k, min_score=min_score, as_of=as_of,
                                valid_at=valid_at, known_at=known_at, states=states,
                                include_invalidated=include_invalidated,
                                memory_types=memory_types,
                                include_episodes=include_episodes)

    def recall(self, query: str, *, k: int = 8, min_score: float = 0.0,
               memory_types: Sequence[MemoryType | str] | None = None,
               include_episodes: bool = False, budget: int | None = None) -> str:
        return self._mem.recall(query, k=k, min_score=min_score,
                                memory_types=memory_types,
                                include_episodes=include_episodes, budget=budget)

    def get(self, claim_id: str) -> Claim | None:
        return self._mem.get(claim_id)

    def get_all(self, *, states: Collection[str] | None = None,
                include_invalidated: bool | None = None,
                limit: int = 100, offset: int = 0,
                as_of: datetime | None = None, valid_at: datetime | None = None,
                known_at: datetime | None = None) -> list[Claim]:
        return self._mem.get_all(states=states,
                                 include_invalidated=include_invalidated, limit=limit,
                                 offset=offset, as_of=as_of, valid_at=valid_at,
                                 known_at=known_at)

    def count(self, *, states: Collection[str] | None = None,
              include_invalidated: bool | None = None,
              as_of: datetime | None = None, valid_at: datetime | None = None,
              known_at: datetime | None = None) -> int:
        return self._mem.count(states=states,
                               include_invalidated=include_invalidated, as_of=as_of,
                               valid_at=valid_at, known_at=known_at)

    def history(self, subject: str, predicate: str, *,
                as_of: datetime | None = None, valid_at: datetime | None = None,
                known_at: datetime | None = None) -> list[Claim]:
        return self._mem.history(subject, predicate, as_of=as_of, valid_at=valid_at,
                                 known_at=known_at)

    def why(self, claim_id: str, *, as_of: datetime | None = None,
            valid_at: datetime | None = None,
            known_at: datetime | None = None) -> Provenance | None:
        return self._mem.why(claim_id, as_of=as_of, valid_at=valid_at,
                             known_at=known_at)

    def ask(self, question: str, *, at: datetime | None = None, k: int = 3,
            min_score: float = 0.0) -> Answer:
        return self._mem.ask(question, at=at, k=k, min_score=min_score)

    def since(self, when: datetime) -> Delta:
        return self._mem.since(when)

    def produced(self, episode_id: str, *, as_of: datetime | None = None,
                 valid_at: datetime | None = None,
                 known_at: datetime | None = None) -> list[Claim]:
        return self._mem.produced(episode_id, as_of=as_of, valid_at=valid_at,
                                  known_at=known_at)

    def neighborhood(self, entity: str, *, depth: int = 2, k: int = 10,
                     min_hops: int = 1, predicates: Sequence[str] | None = None,
                     as_of: datetime | None = None, valid_at: datetime | None = None,
                     known_at: datetime | None = None,
                     min_score: float = 0.0) -> list[Path]:
        return self._mem.neighborhood(entity, depth=depth, k=k, min_hops=min_hops,
                                      predicates=predicates, as_of=as_of,
                                      valid_at=valid_at, known_at=known_at,
                                      min_score=min_score)

    def paths_between(self, source: str, target: str, *, depth: int = 3, k: int = 3,
                      predicates: Sequence[str] | None = None,
                      as_of: datetime | None = None, valid_at: datetime | None = None,
                      known_at: datetime | None = None,
                      min_score: float = 0.0) -> list[Path]:
        return self._mem.paths_between(source, target, depth=depth, k=k,
                                       predicates=predicates, as_of=as_of,
                                       valid_at=valid_at, known_at=known_at,
                                       min_score=min_score)

    def standing(self, *, k: int | None = None) -> list[Claim]:
        return self._mem.standing(k=k)

    # -- writing -------------------------------------------------------------

    def add(self, messages: Any, *, role: str = "user",
            ts: datetime | None = None) -> WriteReceipt:
        return self._mem.add(messages, role=role, ts=ts)

    def remember(self, subject: str, predicate: str, obj: str,
                 **kw: Any) -> WriteReceipt:
        return self._mem.remember(subject, predicate, obj, **kw)

    def supersede(self, old_claim_id: str, subject: str, predicate: str, obj: str,
                  **kw: Any) -> WriteReceipt:
        return self._mem.supersede(old_claim_id, subject, predicate, obj, **kw)

    def forget(self, subject: str, predicate: str, *, at: datetime | None = None,
               close: str = "retired") -> list[Claim]:
        return self._mem.forget(subject, predicate, at=at, close=close)

    def delete(self, claim_id: str, *, at: datetime | None = None,
               close: str = "retired") -> bool:
        return self._mem.delete(claim_id, at=at, close=close)

    def end(self, *, claim_id: str | None = None, subject: str | None = None,
            predicate: str | None = None, at: datetime | None = None) -> bool:
        return self._mem.end(claim_id=claim_id, subject=subject, predicate=predicate,
                             at=at)

    def erase(self, claim_id: str, *, sources: bool = False) -> bool:
        return self._mem.erase(claim_id, sources=sources)

    def purge(self, *, confirm_tenant: str | None = None) -> dict[str, int]:
        return self._mem.purge(confirm_tenant=confirm_tenant)

    def consolidate(self) -> dict[str, Any]:
        return self._mem.consolidate()
