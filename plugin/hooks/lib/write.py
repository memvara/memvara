"""Open something that can be written to, write triples to it, and say what happened.

One hook writes — `Stop`, over the turn that just ended — and it needs three things: a
store that may be local or hosted, a loop that counts failures instead of swallowing them,
and a log, because a write that fails leaves no other trace.

The store is opened the way recall opens it. `open_store()` answers None on a hosted
install, which is the normal state rather than a broken one -- and since
memvara/memvara@2a3bb48 it is a *decision* rather than a fact about what can be built:
`MEMVARA_MODE=cloud` now yields a perfectly good `RemoteMemvara`, and `open_store()`
declines it anyway, because the library's hosted client refuses the `budget=` its callers
here pass and takes no `header=`. The hosted client in `lib.hosted` is the route in that
case, and it raises on a failed write rather than returning nothing, which is what lets
`store_facts` tell a refusal from a quiet turn.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from core.host import active

from .open import open_store

#: Beside the store, not in the plugin: the plugin directory is replaced wholesale on
#: update, and a log that disappears on upgrade is not a log.
LOG = Path.home() / ".memvara" / ".hooks" / "capture.log"

#: Bounded by truncation rather than rotation. It is a debugging aid, and a log that needs
#: its own maintenance is worse than no log.
LOG_MAX_BYTES = 64 * 1024

#: The role every episode this plugin stores goes in under, and the point is that it is
#: not "user".
#:
#: What the hooks store is a transcript excerpt -- a synthetic header, `User:` lines,
#: `Claude used` lines, and whatever the person pasted in along the way. `role="user"`
#: says every word of that is the person's own, and the core's deterministic fast path
#: believes it: the rules bind "I" and "my" to the user and match with `search`, so they
#: fire on any clause anywhere in the text, quoted or not. `_clauses` even strips the
#: surrounding quotation marks, so the one signal that a sentence was being cited rather
#: than said is removed before matching.
#:
#: On 2026-08-26 somebody pasted a log quoting the core's own documentation of that fast
#: path, which names its sentence forms by example -- "my name is X", "I live in X", "I
#: work at X". Four claims landed at confidence 0.95, and `user name X` superseded a real
#: name stored since 2026-08-18. Nothing raised. A fast-path write is an ordinary
#: successful write, and no hook here asked for it.
#:
#: Filtering the text instead was considered and is worse. Nothing here can separate prose
#: somebody typed from prose they pasted -- what did this was ordinary English paragraphs,
#: no code fence and no indent to key on -- so a heuristic would hold until the next shape
#: and then fail the same silent way. `FastExtractor` already declines any role but
#: "user", so saying what the text actually is fixes it exactly and needs no guessing.
#:
#: It must be one of "user", "assistant" or "system": `memory_add` validates `role`
#: against a closed enum and a rejection loses the whole episode rather than one field.
#: "system" is the honest member of the three -- this text is machine-composed and is
#: nobody's utterance.
EPISODE_ROLE = "system"


def log(line: str) -> None:
    """Append one line, or give up quietly. Never raises into a hook."""
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        if LOG.exists() and LOG.stat().st_size > LOG_MAX_BYTES:
            LOG.write_text("")
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {line}\n")
    except OSError:
        pass


def open_writer() -> "tuple[Any, Any] | tuple[None, None]":
    """`(store, close)` for whichever backend answers, or `(None, None)`.

    `close` is None for a local engine, which has no connection to give back, and the
    hosted client's `close` otherwise. Callers close after their last write and not
    before: the hosted client connects lazily, so an early return costs nothing.

    Every caller gets the same backend again. For one release readers and writers wanted
    different ones, because the MCP surface could not carry `sources=`; memvara#76 shipped
    and the fork closed.
    """
    store = open_store()
    if store is not None:
        return store, None

    from .hosted import open_hosted

    hosted = open_hosted()
    if hosted is None:
        return None, None
    return hosted, hosted.close


#: An episode id as the server renders it. `_new_id("ep")` on the other side, and the
#: prefix is what makes this safe to pull out of prose: the receipt line is English around
#: the ids ("turn id(s): ep_… — pass these to memory_remember.sources …"), and matching the
#: shape rather than the sentence means a reworded receipt does not silently yield nothing.
_TURN_ID = re.compile(r"\bep_[0-9a-f]{6,}\b")


def turn_ids(receipt: object) -> "list[str]":
    """The episode ids in a `memory_add` receipt, or `[]`.

    Only the hosted route needs this. `Memvara.add` returns a `WriteReceipt` and the local
    branch below already passes an `Episode` object, so there is nothing to parse there;
    the hosted client returns the server's rendered text and the ids are the only route
    from a stored turn back to a claim that came out of it.

    Empty was the ordinary answer until memvara/memvara#76, which renders the line at all.
    It has shipped to the hosted endpoint: an `add` there now returns
    `turn id(s): ep_... — pass these to memory_remember.sources`, and a claim written from
    them resolves under `memory_why`. Still empty against an older server, which is why
    `store_facts` treats no ids as a reason to write the fact anyway rather than a failure.
    """
    return _TURN_ID.findall(receipt) if isinstance(receipt, str) else []


def store_facts(store: Any, facts: Iterable[Any], turn: str = "",
                hosted: bool = False,
                sources: "Sequence[str]" = ()) -> "tuple[int, list[str]]":
    """Write each fact. Returns how many landed and why the rest did not.

    Three arguments here were being left at their defaults, and each omission is silent.

    `memory_type` — nothing on the write path infers one from the words of a fact, so an
    omitted type falls back to the predicate's registered default, and that default is
    `semantic`. Every standing instruction this hook has ever written was therefore filed
    as a durable fact rather than a procedural one, which makes the `procedural` filter --
    the one to use when you are about to do work and want the user's preferences -- unable
    to find any of them.

    `sources` — the turn the fact came from. Without it `why()` on the result has nothing
    to show, which in the one library whose pitch is that provenance always resolves is the
    difference between a provenance store and a dictionary. The text is right here in the
    caller's hand, and was being dropped.

    `extractor` — defaults to `"api"`, which is what an application asserting its own facts
    reports. A hook mining a transcript is not that, and an audit that cannot tell the two
    apart cannot review either.

    `sources` was library-only, and is not any more: memvara/memvara#76 put it on the
    hosted `memory_remember` too, so both routes link a fact to the turn it came from —
    the local one by passing a real `Episode`, the hosted one by citing the ids the receipt
    rendered. The schema is still closed, so an argument an older server has not heard of
    is a hard rejection rather than a silent ignore -- but the probe that protects against
    that lives in the client, `HostedRecall.remember`, not in the branch below, which
    passes `sources` on unconditionally. Said precisely because the imprecise version
    invites the obvious next move: handing this function some other hosted store with no
    such gate, and losing the whole write instead of one field. `memory_type` both routes
    take.

    `extractor` now goes over both, and the hosted client asks the server whether it takes
    the argument rather than assuming it does, because a server older than the argument
    rejects the whole write rather than dropping the field. See `HostedRecall.accepts`.
    Until that server ships, a hosted claim keeps reporting itself as "Derived by user" —
    which is not a blank but an assertion, and the one that let a hook's own inference be
    read back a session later as something the user had stated.

    `hosted` is passed in rather than sniffed off the object, and the first attempt did sniff
    it: "no `registry` attribute and no `add_episode`" looked like a safe capability test and
    is not. `ScopedMemvara` -- the scoped view, which `session_start` already builds -- has
    neither, so a perfectly local handle read as hosted and silently dropped the provenance
    this function exists to attach. It happens to be unreachable today only because
    `build_memvara` returns a bare `Memvara`, which is an implementation detail of a
    different repository. `open_writer` knows which backend it opened; asking it is the
    answer that cannot rot.

    Confidence is below 1.0 deliberately: a model inferred these from a transcript, and
    they should not outrank something the user stated outright.
    """
    stored, failed = 0, []
    for fact in facts:
        subject, predicate, obj = fact[0], fact[1], fact[2]
        memory_type = fact[3] if len(fact) > 3 else None
        kwargs: dict = {"confidence": 0.7}
        if memory_type:
            kwargs["memory_type"] = memory_type
        # The label the host writes under. It is stored on every claim and rendered back
        # by `memory_why`, so it is a fact about recorded history rather than a string to
        # tidy: changing it re-labels everything written before the change.
        kwargs["extractor"] = active().extractor_label
        if not hosted:
            episode = _episode(turn)
            if episode is not None:
                kwargs["sources"] = [episode]
        elif sources:
            # The hosted half of the same thing, and it took two changes on the other side
            # to become possible: the tool had to declare `sources`, and the receipt had to
            # render the episode ids, or a caller could not learn the id it needed to cite.
            # Each made the other useless, which is why `memory_why` answered "No source
            # turns are retained" for every fact any hosted client had ever written.
            #
            # IDs, not the turn. `_cite` stores what it is handed and links a string, so
            # passing the text would store a second copy of the turn `_keep_turn` just
            # wrote. The client drops this again if the server has not got #76.
            kwargs["sources"] = list(sources)
        try:
            store.remember(subject, predicate, obj, **kwargs)
            stored += 1
        except Exception as exc:
            failed.append(f"{subject}/{predicate}: {type(exc).__name__}: {exc}")
    return stored, failed


def _episode(turn: str) -> Any:
    """An `Episode` carrying the turn, or None when the library is not importable.

    Deliberately not pre-stored: passing the object rather than its id is what makes the
    claim and its source one transaction, so a crash between them cannot leave a claim
    citing a turn that does not exist.

    `EPISODE_ROLE` for the same reason `_keep_turn` uses it, and one reason more. Writing
    it is safe today either way -- `remember()` asserts a claim and stores what it cites
    without running extraction, which was measured rather than assumed. But the copy it
    leaves behind is permanent and is labelled with whoever said it, and `reextract()`
    runs the fast path over any stored episode that no claim cites. Erase a claim this
    hook wrote (`erase()` leaves its sources by default) and its turn has no citations
    left, the salience gate passes it as a user turn, and the next sweep re-derives
    exactly the claims this constant exists to stop. One word closes that rather than
    resting on an argument about another repository's call graph.
    """
    if not turn.strip():
        return None
    try:
        from memvara.types import Episode

        return Episode(content=turn, role=EPISODE_ROLE)
    except Exception:
        return None
