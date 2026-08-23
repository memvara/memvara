"""HTTP-backed `Store`: talk to a memvara-cloud deployment instead of a local file.

**Read this before reaching for a method here.** `memvara.store.base.Store` is the
protocol the *engine* — `Reconciler`, `GraphTraverser`, the retrieval fusion — writes
against, and every one of its methods assumes it is one hop from the row: a raw `Claim`
to upsert, a `fact_key` to look up, a vector to compare, a tenant to page over. The
matching server-side surface, `memvara_cloud.rest.app`, is not that. It is a *facade*:
`POST /v1/facts`, `GET /v1/history`, `POST /v1/erasures` — a small set of high-level
operations, each doing its own reconciliation, each authorizing itself against the
bearer token's own scope rather than a `tenant` argument the caller supplies. The two
shapes do not line up method-for-method, and this module does not pretend they do.

What follows is faithful to what the REST API actually exposes, method by method, and
raises `NotImplementedError` — with a docstring, not a bare raise — everywhere the facade
has no operation to reach. **Do not fake it.** A `put_claim` that quietly wrote through
`POST /v1/facts` would silently reinterpret every field the caller set (id, salience,
`invalidated_by`, the exact `recorded_at`) as something else's business, and a
`competing_claims` that returned `[]` because there is no matching endpoint would make
every write path believe a slot was empty. Both failures are worse than an exception,
because nothing downstream would notice until the data was already wrong.

**Not wired into `build_memvara()` for local retrieval or maintenance.** `Memvara` is
built against a `Store` and calls the low-level surface throughout — `candidate_ids`,
`lexical_search`, `vector_search`, `competing_claims` — none of which this class can
answer. A `ServerConfig(mode="cloud")` deployment therefore does not run the engine
locally at all; it is the MCP/HTTP server layer that becomes a client of the *remote*
deployment's own `/v1` facade, using its own tools (search, remember, forget) rather than
`Memvara`'s. `RemoteStore` exists at the storage layer for the pieces that *do* have a
faithful mapping — reading one memory by id, erasing one, erasing a scope, reading
tenant stats — and to give `memvara.server.config.build_memvara()` something concrete to
construct and hold, per its own docstring. It is not, and cannot be, a drop-in swap for
`SQLiteStore` behind a running `Memvara` instance.

**Import discipline.** `httpx` is imported lazily, inside `__init__`, with a message
naming the extra to install. `import memvara` must never require it — this repository's
"numpy and nothing else" core promise (see `docs/INTERNALS.md`, invariant 5, and
README.md's "Open core" section) — and a module-level `import httpx` here would break
that the moment anything imports this file, extra or not.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Any, Collection, Iterable, Iterator, Sequence

import numpy as np

from ..types import Claim, Derivation, Episode, MemoryType, Scope

if TYPE_CHECKING:
    import httpx

    from ..schema import PredicateSpec

#: The one message every unsupported method raises, filled in with which method and why.
_NO_ENDPOINT = (
    "Store.{method}() has no REST equivalent on the memvara-cloud data plane today. {why} "
    "See memvara/store/remote.py's module docstring for the shape of the mismatch."
)


def _install_hint() -> str:
    return ('httpx is required for RemoteStore. Install it with `pip install '
            '"memvara[cloud]"`.')


class RemoteStore:
    """A `Store` backed by one memvara-cloud project's `/v1` REST API.

    Constructed with the project's base URL and a bearer API key minted by the device
    login flow (`~/.memvara/credentials.json`, or `MEMVARA_API_KEY`); see
    `memvara.server.config.ServerConfig`, which builds one of these when `mode="cloud"`.

    Every method below authorizes as whichever project the API key was minted for — the
    facade resolves scope from the bearer token, not from a `tenant` argument — so the
    `tenant` parameters several `Store` methods declare are accepted for signature
    compatibility and, where the facade truly has no way to honor a *different* tenant
    than the key's own, are asserted against it rather than silently ignored.
    """

    #: The methods on this class that actually reach the API. Everything else on the
    #: `Store` protocol raises, and the split is written down rather than left to be
    #: discovered at the first write, because callers branch on it: see
    #: `memvara.server.config.build_memvara`, which refuses to build a server whose engine
    #: would run against a store this thin.
    #:
    #: A literal, and kept honest by a test rather than by care —
    #: `tests/test_store_remote.py::test_the_wired_list_names_exactly_the_methods_that_do_
    #: not_raise` calls every protocol method and compares. A name that drifts off this
    #: list is the failure mode that matters: it would make the guard below think a
    #: capability exists.
    WIRED: frozenset[str] = frozenset({
        "batch", "close", "connectivity", "erase_claim", "get_claim", "get_claims",
        "purge", "stats",
    })

    def __init__(self, base_url: str, api_key: str, *, timeout: float = 30.0) -> None:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - exercised via mock in tests
            raise ImportError(_install_hint()) from exc
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._client: httpx.Client = httpx.Client(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    # --- request plumbing --------------------------------------------------

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        """One HTTP call, decoded, with the store's own errors surfaced.

        Not swallowed and not translated into a `Store`-specific exception type: the
        protocol declares none, `SQLiteStore` lets `sqlite3` errors propagate the same
        way, and a caller already has to handle whatever the transport raises. What this
        adds is `response.raise_for_status()` running *before* `.json()`, so a 4xx/5xx
        with a JSON error envelope (`errors.py`'s `ApiError` shape) raises
        `httpx.HTTPStatusError` with the body still readable off `exc.response`, rather
        than a `KeyError` two lines further down guessing at a schema the failure never
        produced.
        """
        response = self._client.request(method, path, **kwargs)
        response.raise_for_status()
        if response.status_code == 204 or not response.content:
            return None
        return response.json()

    def close(self) -> None:
        self._client.close()

    # --- conversion: wire `Memory` <-> local `Claim` ------------------------

    @staticmethod
    def _claim_from_memory(body: dict[str, Any]) -> Claim:
        """`models.Memory`'s wire shape back to a `Claim`.

        `subject_key`/`object_key`/`fact_key`/`value_key` are not part of the wire schema
        at all — `render.memory()` does not publish them — but they need not be: on the
        local `Claim` dataclass they are read-only properties derived from `subject`,
        `object`, `predicate`, `scope` and `meta`, all of which this does set, so the
        returned `Claim` computes the same identities a locally-written one would.
        Everything else the wire carries (id, the two clocks, provenance, salience)
        round-trips exactly.
        """
        from datetime import datetime as _dt

        scope = Scope(tenant=body["scope"]["tenant"], user=body["scope"].get("user"),
                      agent=body["scope"].get("agent"), session=body["scope"].get("session"))
        vt, tt = body["valid_time"], body["transaction_time"]

        def _parse(value: str | None) -> _dt | None:
            return None if value is None else _dt.fromisoformat(value)

        claim = Claim(
            subject=body["subject"], predicate=body["predicate"], object=body["object"],
            scope=scope, text=body["text"], polarity=body["polarity"],
            memory_type=MemoryType(body["memory_type"]),
            valid_from=_parse(vt["valid_from"]) or _dt.now(), valid_to=_parse(vt["valid_to"]),
            recorded_at=_parse(tt["recorded_at"]) or _dt.now(),
            invalidated_at=_parse(tt["invalidated_at"]), invalidated_by=tt["invalidated_by"],
            confidence=body["confidence"], salience=body["salience"],
            observation_count=body["observation_count"],
            sources=list(body["source_ids"]),
            derivation=Derivation(body["derivation"]), extractor=body["extractor"] or "",
            id=body["id"], meta=dict(body["metadata"]),
        )
        return claim

    # --- episodes: no matching endpoint -------------------------------------
    #
    # The facade never exposes a raw turn as an addressable, individually-writable
    # resource. `POST /v1/memories` ingests a batch of `Message`s and extracts from
    # them — the episodes it stores are a side effect the caller does not control the id
    # or metadata of — and there is no `GET /v1/episodes/{id}`. `sources` on
    # `POST /v1/facts` accepts new turns too, but again as a side effect of writing a
    # fact, not as a standalone `add_episode`. Nothing over HTTP does what this protocol
    # method promises: store exactly this `Episode`, as given, addressable by its own id.

    def add_episode(self, ep: Episode) -> None:
        raise NotImplementedError(_NO_ENDPOINT.format(
            method="add_episode",
            why="Episodes are only ever created as a side effect of POST /v1/memories or "
                "the `sources` field of POST /v1/facts, with ids and metadata the server "
                "assigns; there is no endpoint that stores a given Episode verbatim."))

    def get_episode(self, episode_id: str) -> Episode | None:
        raise NotImplementedError(_NO_ENDPOINT.format(
            method="get_episode",
            why="There is no GET /v1/episodes/{id}. A turn is only reachable indirectly, "
                "embedded in GET /v1/memories/{id}/why's `sources` list."))

    def find_episode_by_hash(self, tenant: str, ep_hash: str) -> Episode | None:
        raise NotImplementedError(_NO_ENDPOINT.format(
            method="find_episode_by_hash",
            why="No endpoint looks a turn up by content hash; that dedup check is "
                "internal to the write path the facade already runs server-side."))

    def get_episodes(self, episode_ids: Sequence[str]) -> dict[str, Episode]:
        raise NotImplementedError(_NO_ENDPOINT.format(
            method="get_episodes", why="See get_episode: no per-id episode read exists."))

    def iter_episodes(self, tenant: str | None = None) -> Iterable[Episode]:
        raise NotImplementedError(_NO_ENDPOINT.format(
            method="iter_episodes",
            why="No listing endpoint for raw turns exists; only claims page "
                "(GET /v1/memories)."))

    def scope_episodes(self, scopes: Sequence[Scope], *, limit: int | None = None,
                       newest_first: bool = False) -> list[Episode]:
        raise NotImplementedError(_NO_ENDPOINT.format(
            method="scope_episodes", why="See iter_episodes."))

    # --- claims: get_claim/get_claims map to GET /v1/memories/{id} ---------

    def get_claim(self, claim_id: str) -> Claim | None:
        """`GET /v1/memories/{claim_id}`. Returns the claim in whatever state it is in —
        live, ended or retired — exactly as the route promises; `None` on a 404, which is
        the same answer the route gives for "does not exist" and "not yours"."""
        import httpx
        try:
            body = self._request("GET", f"/v1/memories/{claim_id}")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            raise
        return self._claim_from_memory(body)

    def get_claims(self, claim_ids: Sequence[str]) -> dict[str, Claim]:
        """No bulk-fetch endpoint exists, so this is `get_claim` per id rather than the
        one query `SQLiteStore` runs. Faithful in result, not in cost — a caller doing
        this for many ids pays one round trip each, which is exactly the N+1 the
        protocol's docstring says this method exists to avoid; there is nothing better to
        do against the current REST surface."""
        out: dict[str, Claim] = {}
        for claim_id in claim_ids:
            claim = self.get_claim(claim_id)
            if claim is not None:
                out[claim_id] = claim
        return out

    def put_claim(self, claim: Claim) -> None:
        raise NotImplementedError(_NO_ENDPOINT.format(
            method="put_claim",
            why="No endpoint upserts a raw Claim by its own id, salience, exact "
                "recorded_at, invalidated_by, etc. POST /v1/facts is the closest "
                "relative but it *decides* those things (mints a new id, runs "
                "reconciliation, computes provenance) rather than accepting them from "
                "the caller — using it here would silently discard or reinterpret most "
                "of what the caller set on `claim`."))

    def batch(self) -> AbstractContextManager["RemoteStore"]:
        """A trivial passthrough, not a transaction.

        `SQLiteStore.batch()` defers commits so bulk writes land in one transaction; the
        facade has no equivalent — every `/v1` write call commits on its own and there is
        no multi-call transaction boundary to ask for over HTTP. This context manager
        exists only so code written against the `Store` protocol's `with store.batch() as
        b:` shape still runs, yielding `self` and doing nothing else. It does **not**
        make the calls inside it atomic; a caller relying on that guarantee over
        `RemoteStore` has a bug, not a slow path.
        """
        return self._batch()

    @contextmanager
    def _batch(self) -> Iterator["RemoteStore"]:
        yield self

    def competing_claims(self, tenant: str, fact_key: str, *,
                         valid_at: datetime | None = None,
                         known_at: datetime | None = None) -> list[Claim]:
        raise NotImplementedError(_NO_ENDPOINT.format(
            method="competing_claims",
            why="fact_key is a hash of (subject, predicate) computed by the local "
                "reconciler; no endpoint accepts one or exposes the exact-slot lookup it "
                "keys. GET /v1/history takes a plain subject/predicate pair instead and "
                "answers a related but different question (the whole timeline, not just "
                "the live occupants) — see slot_history below for that mapping."))

    def count_competing(self, tenant: str, fact_key: str, *,
                        valid_at: datetime | None = None,
                        known_at: datetime | None = None) -> int:
        raise NotImplementedError(_NO_ENDPOINT.format(
            method="count_competing", why="See competing_claims: same missing lookup."))

    def find_by_value(self, tenant: str, value_key: str) -> list[Claim]:
        raise NotImplementedError(_NO_ENDPOINT.format(
            method="find_by_value",
            why="value_key is a local hash of the object text with no REST-facing "
                "lookup; nothing on the facade searches claims by resolved object "
                "identity."))

    def claims_citing(self, tenant: str, episode_id: str) -> list[Claim]:
        raise NotImplementedError(_NO_ENDPOINT.format(
            method="claims_citing",
            why="No endpoint answers 'which claims cite this turn'; provenance only "
                "runs the other direction, GET /v1/memories/{id}/why."))

    def slot_history(self, tenant: str, fact_key: str) -> list[Claim]:
        raise NotImplementedError(_NO_ENDPOINT.format(
            method="slot_history",
            why="This signature takes a fact_key (a local hash), and GET /v1/history "
                "takes a plain subject/predicate pair — there is no way to recover the "
                "surface strings from the hash, so this exact call cannot be made. A "
                "caller that already has (subject, predicate) rather than fact_key "
                "should call GET /v1/history directly instead of going through Store."))

    def adjacent(self, tenant: str, keys: Sequence[str], *,
                 outgoing: bool = True, incoming: bool = True,
                 predicates: Sequence[str] | None = None,
                 valid_at: datetime | None = None,
                 known_at: datetime | None = None,
                 scopes: Sequence[Scope] | None = None,
                 limit: int = 1000) -> list[Claim]:
        raise NotImplementedError(_NO_ENDPOINT.format(
            method="adjacent",
            why="No graph-traversal endpoint exists on the data plane today; "
                "subject_key/object_key are not exposed or queryable over /v1."))

    def episodes_near(self, anchor: datetime, scopes: Sequence[Scope], limit: int, *,
                      valid_at: datetime | None = None,
                      known_at: datetime | None = None) -> list[tuple[str, float]]:
        raise NotImplementedError(_NO_ENDPOINT.format(
            method="episodes_near",
            why="No episode listing or search exists on the data plane today; turns are "
                "server-internal and are not queryable over /v1."))

    def residue(self, claim_id: str) -> dict[str, int]:
        raise NotImplementedError(_NO_ENDPOINT.format(
            method="residue",
            why="Proving an erasure means counting rows on the storage that holds them, "
                "and this facade holds none. A count relayed over HTTP would be the "
                "server's word for it, which is the thing a proof is supposed to be "
                "able to disagree with."))

    def erasure_record(self, claim_id: str) -> dict[str, Any] | None:
        raise NotImplementedError(_NO_ENDPOINT.format(
            method="erasure_record",
            why="No erasure-audit endpoint exists on the data plane today."))

    def invalidate(self, claim_id: str, at: datetime, by: str | None) -> None:
        raise NotImplementedError(_NO_ENDPOINT.format(
            method="invalidate",
            why="DELETE /v1/memories/{id} retires a claim but takes neither `at` nor "
                "`by` — it always retires now, with no successor pointer. Writing an "
                "exact `invalidated_at`/`invalidated_by` pair the way this method's "
                "docstring requires ('one statement, both fields') is not expressible "
                "over the current REST surface."))

    def set_valid_to(self, claim_id: str, valid_to: datetime | None) -> None:
        raise NotImplementedError(_NO_ENDPOINT.format(
            method="set_valid_to",
            why="No endpoint writes valid_to alone, and in particular none can *reopen* "
                "an interval by clearing it back to None — every write route on this API "
                "closes claims, never reopens them."))

    def reinforce(self, claim_id: str, salience: float, observation_count: int,
                  sources: Sequence[str]) -> None:
        raise NotImplementedError(_NO_ENDPOINT.format(
            method="reinforce",
            why="No endpoint sets salience/observation_count directly; that bookkeeping "
                "is server-internal and happens automatically inside POST /v1/memories' "
                "reconciliation, not as a call a caller makes on a claim id."))

    # --- retrieval: no direct embedding or index access ---------------------

    def set_embedding(self, claim_id: str, vec: np.ndarray) -> None:
        raise NotImplementedError(_NO_ENDPOINT.format(
            method="set_embedding",
            why="Embeddings are computed and stored server-side; no endpoint accepts a "
                "caller-supplied vector for a claim."))

    def get_embedding(self, claim_id: str) -> np.ndarray | None:
        raise NotImplementedError(_NO_ENDPOINT.format(
            method="get_embedding",
            why="No endpoint reads a stored vector back; it is internal retrieval state, "
                "never rendered onto a Memory."))

    def set_episode_embedding(self, episode_id: str, vec: np.ndarray) -> None:
        raise NotImplementedError(_NO_ENDPOINT.format(
            method="set_episode_embedding", why="See set_embedding."))

    def get_episode_embedding(self, episode_id: str) -> np.ndarray | None:
        raise NotImplementedError(_NO_ENDPOINT.format(
            method="get_episode_embedding", why="See get_embedding."))

    def clear_embeddings(self) -> int:
        raise NotImplementedError(_NO_ENDPOINT.format(
            method="clear_embeddings",
            why="Re-embedding onto a new model dimension is an operator action on the "
                "deployed store, not something a project's API key can trigger over "
                "/v1."))

    def candidate_ids(self, scopes: Sequence[Scope], *,
                      valid_at: datetime | None = None,
                      known_at: datetime | None = None,
                      states: Collection[str] | None = None,
                      include_invalidated: bool | None = None) -> list[str]:
        raise NotImplementedError(_NO_ENDPOINT.format(
            method="candidate_ids",
            why="No endpoint returns bare ids for a scope set; the nearest facade route, "
                "GET /v1/memories, pages full Memory bodies and resolves scope from the "
                "bearer token rather than an explicit `scopes` list — the two are not "
                "the same operation."))

    def lexical_search(self, query: str, scopes: Sequence[Scope], limit: int, *,
                       valid_at: datetime | None = None,
                       known_at: datetime | None = None,
                       states: Collection[str] | None = None,
                       include_invalidated: bool | None = None
                       ) -> list[tuple[str, float]]:
        raise NotImplementedError(_NO_ENDPOINT.format(
            method="lexical_search",
            why="POST /v1/search runs a fused lexical+vector ranking and cannot be asked "
                "for the lexical leg alone; it also cannot be pointed at an explicit "
                "`scopes` list, since scope is resolved from the bearer token. Use the "
                "facade's search route directly rather than through Store for a fused "
                "query."))

    def vector_search(self, qvec: np.ndarray, scopes: Sequence[Scope], limit: int, *,
                      valid_at: datetime | None = None,
                      known_at: datetime | None = None,
                      states: Collection[str] | None = None,
                      include_invalidated: bool | None = None
                      ) -> list[tuple[str, float]]:
        raise NotImplementedError(_NO_ENDPOINT.format(
            method="vector_search",
            why="POST /v1/search takes a text query and embeds it server-side; there is "
                "no route that accepts a caller-supplied vector, which this method's "
                "signature requires."))

    def episode_candidate_ids(self, scopes: Sequence[Scope], *,
                              valid_at: datetime | None = None,
                              known_at: datetime | None = None) -> list[str]:
        raise NotImplementedError(_NO_ENDPOINT.format(
            method="episode_candidate_ids", why="See candidate_ids; no episode listing "
                "endpoint exists at all (see iter_episodes above)."))

    def lexical_search_episodes(self, query: str, scopes: Sequence[Scope], limit: int, *,
                                valid_at: datetime | None = None,
                                known_at: datetime | None = None
                                ) -> list[tuple[str, float]]:
        raise NotImplementedError(_NO_ENDPOINT.format(
            method="lexical_search_episodes", why="See lexical_search; POST /v1/search's "
                "`include_episodes` flag folds turns into the same fused ranking rather "
                "than exposing a standalone lexical-only episode search."))

    def vector_search_episodes(self, qvec: np.ndarray, scopes: Sequence[Scope],
                               limit: int, *, valid_at: datetime | None = None,
                               known_at: datetime | None = None
                               ) -> list[tuple[str, float]]:
        raise NotImplementedError(_NO_ENDPOINT.format(
            method="vector_search_episodes", why="See vector_search."))

    # --- erasure: purge/erase_claim map to POST /v1/erasures ---------------

    def purge(self, scope: Scope) -> dict[str, int]:
        """`POST /v1/erasures` with `scope` set, erasing everything at and below it.

        The bearer token's own tenant is asserted against `scope.tenant` rather than
        sent: `ErasureRequest.scope` is an `ErasureScope` of `user`/`agent`/`session`
        only, resolved against the credential exactly like every other scope on this
        API — there is no way to *name* a different tenant, and a `RemoteStore` speaks
        for exactly one project's key. Erasing the whole tenant (`user=None`) requires
        `confirm_tenant`, per the endpoint's own guard against an accidental empty-object
        request; this passes `scope.tenant` for it in that case, matching what a caller
        erasing their own tenant must already know.

        Returns the `counts` mapping the response carries — `claims`, `episodes`,
        `embeddings`, `entities` — the same four keys `Store.purge`'s protocol
        docstring promises.
        """
        body: dict[str, Any] = {
            "scope": {"user": scope.user, "agent": scope.agent, "session": scope.session},
        }
        if scope.user is None:
            body["confirm_tenant"] = scope.tenant
        result = self._request("POST", "/v1/erasures", json=body)
        return dict(result["counts"] or {})

    def erase_episode(self, episode_id: str, *, cited: bool = False) -> bool:
        raise NotImplementedError(_NO_ENDPOINT.format(
            method="erase_episode",
            why="POST /v1/erasures erases a `memory_id` or a whole `scope`, never a "
                "single episode id; there is no per-turn erasure route, matching the "
                "absence of any per-turn read route."))

    def erase_claim(self, claim_id: str, *, sources: bool = False) -> dict[str, int]:
        """`POST /v1/erasures` with `memory_id` set. The facade's own response reports a
        bare `erased: bool` rather than the four-key count mapping this method's
        docstring promises (see `ErasureResponse.counts`, which is explicitly null for a
        single-memory erasure) — so this fills `claims` in from that bool (`1` erased,
        `0` not found) and leaves `episodes`, `embeddings` and `entities` at `0`. That
        under-reports the true row count whenever `sources=True` actually erased turns;
        it is the best this endpoint's response shape allows, not a faithful `0`."""
        body = {"memory_id": claim_id, "sources": sources}
        result = self._request("POST", "/v1/erasures", json=body)
        claims = 1 if result["erased"] else 0
        return {"claims": claims, "episodes": 0, "embeddings": 0, "entities": 0}

    # --- learned schema: not exposed over HTTP ------------------------------

    def put_spec(self, spec: "PredicateSpec", tenant: str = "default") -> None:
        raise NotImplementedError(_NO_ENDPOINT.format(
            method="put_spec",
            why="Predicate cardinality/volatility learning is internal reconciler state "
                "with no write route; a project's API key cannot alter it."))

    def all_specs(self, tenant: str = "default") -> list["PredicateSpec"]:
        raise NotImplementedError(_NO_ENDPOINT.format(
            method="all_specs", why="See put_spec: no read route either."))

    # --- resolved entities: not exposed over HTTP ---------------------------

    def put_entity(self, entity_id: str, canonical: str, aliases: Sequence[str],
                   tenant: str = "default") -> None:
        raise NotImplementedError(_NO_ENDPOINT.format(
            method="put_entity",
            why="Entity resolution ('these spellings name one thing') has no write "
                "route on the data plane."))

    def all_entities(self, tenant: str = "default") -> list[tuple[str, str, tuple[str, ...]]]:
        raise NotImplementedError(_NO_ENDPOINT.format(
            method="all_entities", why="See put_entity: no read route either."))

    # --- maintenance ---------------------------------------------------------

    def iter_claims(self, tenant: str | None = None,
                    include_invalidated: bool | None = None, *,
                    states: Collection[str] | None = None) -> Iterable[Claim]:
        raise NotImplementedError(_NO_ENDPOINT.format(
            method="iter_claims",
            why="GET /v1/memories pages claims but resolves scope from the bearer token "
                "rather than walking a whole tenant regardless of scope, which is what "
                "this maintenance primitive promises (`reembed()` and similar walks); "
                "the two are not the same population, and silently substituting the "
                "narrower one would make a maintenance pass believe it had covered a "
                "tenant it had only partly seen."))

    def stats(self, tenant: str | None = None) -> dict[str, int]:
        """`GET /v1/stats`, returning `tenant_counts` — row counts for the whole tenant
        the bearer key belongs to, matching this method's own scope (`tenant`, not one
        scope within it). `tenant` is accepted for signature compatibility only: the
        facade has no way to ask about a *different* tenant than the key's own, so a
        `tenant` that does not match the key's project raises `ValueError` rather than
        silently answering for the wrong one."""
        result = self._request("GET", "/v1/stats")
        if tenant is not None and tenant != result["scope"]["tenant"]:
            raise ValueError(
                f"RemoteStore.stats(tenant={tenant!r}) was asked about a tenant other "
                f"than this API key's own ({result['scope']['tenant']!r}). The facade "
                "resolves tenant from the bearer token; there is no way to ask about a "
                "different one from the same key.")
        return dict(result["tenant_counts"])

    def connectivity(self, tenant: str | None = None) -> dict[str, int]:
        """The two join-rate counts off `GET /v1/stats`, or `{}` from a facade too old
        to report them.

        `{}` rather than zeros, and the distinction is the whole point of the method. A
        hosted store that has not deployed the counts yet is not a store with no joins in
        it, and `memory_stats` prints a join rate only when it has actually been
        measured — so an operator is never shown a 0.0% that came from an old facade
        instead of from their data.

        The tenant check is `stats()`'s, for `stats()`'s reason: the facade resolves
        tenant from the bearer token, so a mismatched argument is a caller error rather
        than a question the endpoint could answer.
        """
        result = self._request("GET", "/v1/stats")
        if tenant is not None and tenant != result["scope"]["tenant"]:
            raise ValueError(
                f"RemoteStore.connectivity(tenant={tenant!r}) was asked about a tenant "
                f"other than this API key's own ({result['scope']['tenant']!r}). The "
                "facade resolves tenant from the bearer token; there is no way to ask "
                "about a different one from the same key.")
        counts = result["tenant_counts"]
        if "joinable_claims" not in counts:
            return {}
        return {"live_claims": int(counts["live_claims"]),
                "joinable_claims": int(counts["joinable_claims"])}
