"""Handles on one store for the model-faults tests, the record of what a write did, and the
turns and model claims that several test files share.

Two kinds of handle share a store: one whose model is a `ScriptedModel`, and one with no
model at all. A read stage that fails must serve exactly what the second serves. And a
write the model misbehaved in must leave every claim that was already stored as it was, or
at most ended; `ledger` records every claim before the write and `fates` names what the
write did to each one.
"""

from __future__ import annotations

import json
import os
import pathlib
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Sequence

import memvara
from memvara import Memvara
from memvara.embed import HashingEmbedder
from memvara.select.model import ModelSelector
from memvara.select.stages import QueryRewriter, Synthesizer
from memvara.store import Store
from memvara.types import Claim, WriteReceipt

from harness import stores

from .scripted import ScriptedModel, Text

#: The user every handle is bound to.
USER = "u1"

#: A turn the fast path reads, so no model is asked about it. It stores `name` = "Ada".
FAST_TURN = "My name is Ada."

#: A turn the salience gate passes and the fast path does not read, so it reaches the
#: model.
MODEL_TURN = "The team relocated the whole office to Porto over the summer."

#: About 13,000 characters in 13 paragraphs, which the splitter cuts into three pieces.
LONG = "\n\n".join(
    f"Part {n} of the migration notes covers the database, the queue, the cache and the "
    "search index, and says which team owns each step and how it is rolled back. " * 6
    for n in range(13))


def claim(subject: str, predicate: str, obj: str, **changes: Any) -> dict[str, Any]:
    """A model claim citing turn 0, with every field of the claim schema."""
    return {"subject": subject, "predicate": predicate, "object": obj, "polarity": 1,
            "memory_type": "semantic", "confidence": 0.9, "source_index": 0,
            "when": None, "amount": None, "unit": None, **changes}


#: The claim most tests have the model read from `MODEL_TURN`.
PORTO = claim("team", "based_in", "Porto")

#: What an acquisition call answers when it reads a spelling as a new predicate that
#: holds many values.
NEW_MANY = Text('{"canonical": null, "cardinality": "many", "volatility": "slow", '
                '"memory_type": "semantic"}')


def with_model(model: ScriptedModel, path: pathlib.Path | None = None, *,
               store: Store | None = None, user: str = USER, ranked: bool = False,
               **options: Any) -> Memvara:
    """A handle whose model is `model`, on a new in-memory store, the file at `path`, or
    `store`.

    Its query rewriter and synthesizer read the model's clock, so a reply the script marks
    `Late` arrives late. With `ranked=True` it also has a `ModelSelector` on the same
    model, for `search(ranked=True)` and `recall(ranked=True)`. `options` go to `Memvara`.
    """
    if ranked:
        options["read_selector"] = ModelSelector(model)
    return Memvara(None if path is None else str(path), store=store,
                   embedder=HashingEmbedder(dim=512), llm=model, user=user,
                   read_rewriter=QueryRewriter(model, clock=model.clock),
                   synthesizer=Synthesizer(model, clock=model.clock), **options)


def without_model(store: Store, *, user: str = USER) -> Memvara:
    """A handle with no model on `store`: the reads every failed read stage must match."""
    return stores.memory(store=store, user=user)


def agentic(model: ScriptedModel) -> Memvara:
    """A handle whose model is `model`, with agentic extraction switched on."""
    return with_model(model, write_agentic_extraction=True)


def model_claims(mem: Memvara) -> list[str]:
    """The objects of the live claims the scripted model extracted, in sorted order."""
    return sorted(c.object for c in mem.get_all() if c.extractor == ScriptedModel.name)


@dataclass(frozen=True)
class Row:
    """One claim as the store holds it: its state, and the two stamps that close it."""

    state: str
    valid_to: datetime | None
    invalidated_at: datetime | None


def stored_claim(mem: Memvara, claim_id: str) -> Claim:
    """The claim `claim_id` as the store holds it, in any state. The test fails with the
    id if the store has no claim by that id."""
    claim = mem.store.get_claim(claim_id)
    assert claim is not None, f"the store has no claim {claim_id}"
    return claim


def ledger(mem: Memvara) -> dict[str, Row]:
    """Every claim of the handle's tenant, in every state, by id."""
    claims = mem.store.iter_claims(mem.default_scope.tenant,
                                   states=("live", "ended", "retired"))
    return {c.id: Row(c.state, c.valid_to, c.invalidated_at) for c in claims}


def fate(was: Row, now: Row | None, *, erased: bool = False) -> str:
    """What happened to one claim between two readings of its row.

    - `unchanged`: the row is the same.
    - `ended`: the world clock closed, or an end the claim already had moved earlier.
    - `extended`: an end the claim already had moved later.
    - `reopened`: an end the claim already had was cleared.
    - `retired`: the belief clock closed, which says the claim was wrong.
    - `erased`: the row is gone and an erasure record names it (`erased=True`).
    - `missing`: the row is gone and no erasure record names it.
    - `changed`: anything else, such as a retirement that moved or was cleared.

    `extended` and `reopened` break the rule that a closed clock never moves later, so
    they take precedence over a retirement made in the same change: a claim that was
    retired and reopened at once reports `reopened`.
    """
    if now is None:
        return "erased" if erased else "missing"
    if now == was:
        return "unchanged"
    if was.valid_to is not None:
        if now.valid_to is None:
            return "reopened"
        if now.valid_to > was.valid_to:
            return "extended"
    if now.invalidated_at != was.invalidated_at:
        return "retired" if was.invalidated_at is None else "changed"
    if now.valid_to != was.valid_to:
        return "ended"
    return "changed"


def fates(before: Mapping[str, Row], mem: Memvara) -> dict[str, str]:
    """What `fate` says has happened to each claim in `before` since it was recorded."""
    after = ledger(mem)
    out: dict[str, str] = {}
    for claim_id, was in before.items():
        now = after.get(claim_id)
        erased = now is None and mem.store.erasure_record(claim_id) is not None
        out[claim_id] = fate(was, now, erased=erased)
    return out


def seed(mem: Memvara) -> dict[str, str]:
    """Store three facts before any model output arrives, and return their ids by name.

    Two the caller asserted: `berlin` (`lives_in`, which holds one value) and `tea`
    (`likes`, which holds many). One the fast path read from a turn, with no model asked:
    `acme` (`works_at`).
    """
    berlin = mem.remember("user", "lives_in", "Berlin").added[0].id
    tea = mem.remember("user", "likes", "green tea").added[0].id
    acme = mem.add("I work at Acme.").added[0].id
    return {"berlin": berlin, "tea": tea, "acme": acme}


def turns(mem: Memvara, receipt: WriteReceipt) -> list[str | None]:
    """The text of every turn a write stored, read back from the store."""
    out: list[str | None] = []
    for episode_id in receipt.episode_ids:
        episode = mem.store.get_episode(episode_id)
        out.append(None if episode is None else episode.content)
    return out


def rendered(results: Sequence[Any]) -> str:
    """Every field of every search result, one result per line.

    Two reads are the same read, byte for byte, when their renderings are equal: each
    line names the claim or turn by id and carries its score and its whole explanation.
    """
    lines = []
    for result in results:
        item = result.claim if hasattr(result, "claim") else result.episode
        lines.append(f"{type(result).__name__} {item.id} {result.score!r} "
                     f"{result.explain!r}")
    return "\n".join(lines)


def tool_text(server: Any, name: str, arguments: Mapping[str, Any]) -> tuple[str, bool]:
    """One `tools/call` sent to an in-process MCP server as the line a client would send.
    Returns the reply's text and its error flag."""
    line = server.handle_line(json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": name, "arguments": dict(arguments)}}))
    result = json.loads(line)["result"]
    return result["content"][0]["text"], bool(result["isError"])


#: The frames Python 3.10 and 3.11 give a comprehension of its own. From 3.12 a list
#: comprehension runs inside its function's frame, so `raised_in` skips these to give one
#: answer on every version.
_COMPREHENSIONS = frozenset({"<listcomp>", "<dictcomp>", "<setcomp>", "<genexpr>"})


def raised_in(error: BaseException) -> tuple[str, str]:
    """The file and the function inside the memvara package where `error` was raised.

    The file is a real path, to compare with `os.path.realpath(module.__file__)`. The
    answer is `("", "")` when no frame of the error's traceback is inside memvara. A test
    that pins a bug compares this with the place the bug raises, so the same exception
    type raised from anywhere else counts as a different failure.
    """
    package = os.path.join(os.path.dirname(os.path.realpath(memvara.__file__)), "")
    frames = [frame for frame in traceback.extract_tb(error.__traceback__)
              if os.path.realpath(frame.filename).startswith(package)]
    while frames and frames[-1].name in _COMPREHENSIONS:
        frames.pop()
    if not frames:
        return "", ""
    return os.path.realpath(frames[-1].filename), frames[-1].name
