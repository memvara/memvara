"""The standing set: every procedural preference, rather than the ones that sound like a
query.

`session_start` used to ask for standing preferences with
`recall("who is this user, how do they want work done, what are they working on",
memory_types=["procedural"])`, and that is a category error rather than a tuning problem.
A standing preference applies to *every* turn, so ranking the set by similarity to any one
sentence selects on the wrong property entirely: a rule earns its slot by sounding like the
query.

It was measured, not supposed. A rule stored at confidence 1.00 -- never put Claude's name
in a commit, a PR or an issue -- scored 0.760 against a query about attribution and did not
place in the top eight against that sentence, so it never reached a session. What did reach
sessions was a paraphrase of it written by the capture hook at confidence 0.70 which had
turned "Claude name" into "user name", and the reason *that* one ranked is the reason it was
wrong: "user name" matches "who is this **user**". Twenty-six of forty-five commits made
after the rule was stored still carried the trailer it forbids.

So this module enumerates instead of searching. Four routes, each falling back to the next,
because the plugin runs against a local library on some installs and a hosted endpoint on
most:

1. the local library's `get_all()`, which already returns every claim in scope, newest
   first, with no query anywhere in it;
2. `memory_standing`, once a server advertises it -- the tool that should exist for this;
3. `memory_since` with an early instant, which is enumeration through a door built for
   deltas and is the only queryless read a current server offers;
4. the old `recall` call, so a server answering none of the above degrades to the behaviour
   it has today rather than to silence.

**Only routes 1 and 2 carry confidence.** `memory_since` renders `[id state type]` per row
and no more, so on route 3 the order is the server's -- recency -- and a low-confidence
paraphrase sorts beside the sentence it garbled. That is worth stating plainly rather than
hiding behind a sort key that silently does nothing: enumeration is what fixes the reported
failure, because a rule that is *present* no longer depends on outranking anything, and
ordering by confidence is the refinement that arrives with route 2.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable, NamedTuple

#: A row of the "Believed now, not believed then" half of a `memory_since` reply. The other
#: half is claims the store STOPPED believing, and parsing it into the standing set would
#: re-assert every preference the user has ever withdrawn. `_from_since` stops reading at
#: the second header for that reason, and `test_a_withdrawn_rule_never_returns_through_the
#: _delta` is what keeps it stopping.
#: Deliberately open about how many fields the bracket holds. It pinned exactly three --
#: `[id=X type state]` -- and the server is free to add a fourth: the `(inferred)` marker
#: is the one arriving. A fixed count does not fail loudly on a fourth, it fails
#: SILENTLY, because `_rows` skips what does not match: the day the server marks a row,
#: every marked row stops parsing and the block quietly loses exactly the machine-written
#: claims while still looking whole. Matching the whole bracket and reading it as a set
#: costs nothing and removes that entirely.
_ADDED_ROW = re.compile(r"^\+\s+\[id=(\S+)([^\]]*)\]\s+(.+)$")

#: The server's word for "a machine derived this". Matched as a bracket field.
_INFERRED = "inferred"

#: What a marked row ends with. The library's own spelling, so a reader who has seen one
#: block has seen both. The extractor's NAME is deliberately never rendered here or
#: upstream: it is caller-supplied through `memory_remember`, so printing it would put
#: caller text into a model's context and oblige every renderer to flatten it forever.
MARKER = " (inferred)"

#: The line that opens the half we must not read.
_GONE_HEADER = "Believed then, not believed now"

#: Far enough back that "what changed since then" is "everything". A date with no time is
#: midnight UTC, which the tool documents.
_EARLY = "1970-01-01"

#: What a claim must be to belong in this block.
_PROCEDURAL = "procedural"


class Note(NamedTuple):
    """One standing preference, flattened and ready to render.

    `confidence` is `None` rather than a default when the route could not report it, so
    that "we do not know" and "we know it is 1.0" stay distinguishable. `_order` treats
    unknown as unsortable and leaves the server's order alone rather than inventing a
    number that would sort a paraphrase above the sentence it garbled.
    """

    text: str
    confidence: "float | None"
    recorded: str
    ident: str
    subject: str = ""
    #: True when a machine derived this rather than the user stating it. Defaults False
    #: because a route that cannot tell must not imply a human said it -- the marker is a
    #: warning, and inventing one where none is known is worse than omitting it.
    inferred: bool = False


def _flatten(text: str) -> str:
    """One line, with nothing in it that can forge structure around it.

    Stored text is attacker-controlled data on its way into a prompt -- anyone who can talk
    to the agent can put anything in the store -- and this is the point where it is pasted.
    The library's own `recall` flattens for exactly this reason and the local route here
    does not go through it, so the same defence has to exist on this side.

    Newlines go because a claim containing one could otherwise open a line that looks like
    the hook's own output. Square brackets are neutralised because every rendered row in
    this system carries its metadata in them, so a claim containing `[id=... live]` could
    forge a row that was never stored.
    """
    flat = unicodedata.normalize("NFKC", text)
    flat = "".join(" " if ch in "\r\n\t" or ord(ch) < 32 else ch for ch in flat)
    flat = flat.replace("[", "（").replace("]", "）")
    return re.sub(r"\s+", " ", flat).strip()


def _mine(subject: str, cwd: str) -> bool:
    """Whether a standing note is addressed to this user, working here.

    The block is headed "how this user wants work done", and a claim about a repository's
    deploy traps is not that however operationally useful it is. Measured on a real store:
    nine of thirty-two procedural claims had a container as their subject -- 3,606
    characters, a quarter of what every session opened with, carried on every turn.

    Filing them differently would be the tidier fix and the tool surface does not offer it.
    `memory_remember` with the same triple and a new `memory_type` is recognised as the
    same fact and reinforced, not reclassified -- the receipt says `already-known` and the
    type never moves. The only route left is retire-then-recreate, which is irreversible
    and inverts the safe order, so the selection is fixed here instead of the data.

    `project:<cwd>` is kept because a preference scoped to the checkout you are standing in
    is still a preference about how to work -- "use this skill only here, do not
    auto-activate that one" is an instruction, and dropping it because its subject is not
    the literal string `user` would lose a real one. A project note from a DIFFERENT
    checkout is someone else's instruction today and is left out.
    """
    if subject == "user":
        return True
    return bool(cwd) and subject == f"project:{cwd}"


def _machine_wrote(claim: Any) -> bool:
    """Whether a machine derived this claim rather than the user stating it.

    The library's rule, restated rather than imported, because this runs on the local
    route where the caller already holds `Claim` objects and importing `memvara` to read
    one enum costs ~95ms on a path that has a session-start budget.

    Two things count as derived and the second is the one that matters. A `derivation`
    other than USER is machine extraction and is obvious. But `remember()` stamps USER
    whatever called it, so a capture hook mining an assistant's own prose gets USER too --
    which is the incident this whole module exists for. `extractor` separates them.

    **The unmarked set is the tuple `("", "api")`, not `== "api"`.** Stated that way round
    deliberately: the natural prose is "marked unless the extractor is api", and that
    formulation silently marks every claim written before `extractor` existed or by any
    caller that omits it. Two of us restated this rule from prose on the same afternoon
    and got it wrong in different directions, which is the argument for writing the tuple
    down rather than a sentence about it.

    Unknown reads as NOT inferred, deliberately. A claim this cannot classify gets no
    marker rather than a wrong one, since a warning invented from missing data is worse
    than an absent warning.

    **But an extractor naming a machine is not missing data.** An earlier version returned
    early on an unreadable `derivation`, which threw away a decisive `extractor` -- a
    component naming itself as the deriver is the whole case this exists to catch, and a
    Claim shape that stopped exposing `derivation` would have silently unmarked every hook
    write. Absent information means both fields absent.
    """
    derivation = getattr(claim, "derivation", None)
    name = str(getattr(derivation, "name", derivation) or "").upper()
    extractor = str(getattr(claim, "extractor", "") or "")
    if extractor and extractor != "api":
        return True
    if not name:
        return False
    return not (name == "USER" and extractor in ("", "api"))


def _from_local(store: Any) -> "list[Note] | None":
    """Every live procedural claim from a local library handle, or None if this is not one.

    `get_all` has always done this and nothing here had ever called it. `is_live()` rather
    than `invalidated_at is None`: superseding closes valid time alone, so a superseded
    claim has `invalidated_at` unset and reads as live under the old idiom -- the exact
    trap `types.Claim` documents at length.
    """
    getter = getattr(store, "get_all", None)
    if not callable(getter):
        return None
    out: "list[Note]" = []
    for claim in getter():
        kind = getattr(claim, "memory_type", None)
        name = getattr(kind, "value", kind)
        if str(name) != _PROCEDURAL:
            continue
        live = getattr(claim, "is_live", None)
        if callable(live) and not live():
            continue
        text = _flatten(str(getattr(claim, "text", "") or ""))
        if not text:
            continue
        recorded = getattr(claim, "recorded_at", None)
        out.append(Note(
            text=text,
            confidence=float(getattr(claim, "confidence", 1.0)),
            recorded=recorded.isoformat() if hasattr(recorded, "isoformat") else "",
            ident=str(getattr(claim, "id", "")),
            subject=str(getattr(claim, "subject", "")),
            inferred=_machine_wrote(claim),
        ))
    return out


def _rows(text: str) -> "list[Note]":
    """Parse `+ [id state type] text` rows, stopping at the withdrawn half."""
    out: "list[Note]" = []
    for line in text.splitlines():
        if _GONE_HEADER in line:
            break
        found = _ADDED_ROW.match(line.strip())
        if not found:
            continue
        ident, bracket, body = found.groups()
        fields = {token.lower() for token in bracket.split()}
        if _PROCEDURAL not in fields or "live" not in fields:
            continue
        body = _flatten(body)
        if body:
            # Rendered rows read "<subject> <predicate in words> <object>", so the subject
            # is the first token. Recovering the PREDICATE this way would not be safe --
            # the store folds synonyms, and `depends_on` and `depends_on_a` both resolve to
            # the same claim, so the boundary between predicate and object is not
            # decidable from the rendering. The subject needs no boundary.
            out.append(Note(text=body, confidence=None, recorded="", ident=ident,
                            subject=body.split(" ", 1)[0],
                            inferred=_INFERRED in fields))
    return out


def _from_tool(store: Any) -> "list[Note] | None":
    """`memory_standing`, when the server says it has it.

    Asked rather than assumed. Argument validation on the other end is closed, so a tool a
    server has not heard of is a hard rejection rather than a silent ignore -- the same
    reason `HostedRecall.accepts` exists for `extractor`.
    """
    accepts = getattr(store, "accepts", None)
    call = getattr(store, "_call", None)
    if not callable(accepts) or not callable(call) or not accepts("memory_standing", "k"):
        return None
    return _rows(str(call("memory_standing", {}) or ""))


def _from_since(store: Any) -> "list[Note] | None":
    """`memory_since` from an early instant: enumeration through the delta door.

    Not the tool's purpose, and used here because it is the only read a current server
    offers that takes no query. It costs the rows the store has stopped believing, which is
    why `_rows` stops at that header rather than filtering afterwards -- a filter is a thing
    that can be relaxed by someone who does not know what it was holding back.
    """
    call = getattr(store, "_call", None)
    if not callable(call):
        return None
    return _rows(str(call("memory_since", {"since": _EARLY}) or ""))


def _order(notes: "list[Note]") -> "list[Note]":
    """Most-trusted first, then newest, then by id so the order is total.

    The id tiebreak is not decoration. Without a total order two claims written in the same
    instant swap places between runs, and a block that differs run to run is a block whose
    tests pass most of the time -- which is worse than one that fails.

    Routes that cannot report confidence are left in the order the server gave, because a
    stand-in value would sort real claims against a number nobody measured.
    """
    if any(note.confidence is None for note in notes):
        return list(notes)
    return sorted(notes, key=lambda n: (-(n.confidence or 0.0), _negated(n.recorded),
                                        n.ident))


def _negated(stamp: str) -> str:
    """A sort key that puts the newest ISO timestamp first, without parsing it.

    ISO-8601 sorts lexically in chronological order, so inverting each character inverts
    the order. Parsing would mean `datetime.fromisoformat` and a branch for every shape a
    server might send; this needs neither and cannot raise on a stamp it did not expect.
    """
    return "".join(chr(0x10FFFF - ord(ch)) if ord(ch) < 0x10FFFF else ch for ch in stamp)


def render(notes: "list[Note]", header: str, budget: int) -> str:
    """The block, clipped to `budget` characters, saying so when it clipped.

    The count has to be true. A block that drops three preferences and says nothing reads
    exactly like a store that held three fewer, and the reader has no way to tell the
    difference -- which is the same failure this whole module exists to fix, one layer down.
    """
    if not notes:
        return ""
    lines, used, kept = [header], len(header), 0
    for note in notes:
        # The marker goes on the row, not in the header. The header already says some of
        # this set was inferred, and a qualifier over a whole block is the thing that
        # discounts every row or is ignored for all of them -- stated as the reason
        # `recall()` grew a per-row marker upstream, and true here for the same reason:
        # the block is ordered so stated rules come first, and order tells a reader the
        # list is sorted without telling them WHERE the boundary falls. In twenty-two
        # rows, row twelve is unknowable.
        line = f"- {note.text}{MARKER if note.inferred else ''}"
        if kept and used + 1 + len(line) > budget:
            break
        lines.append(line)
        used += 1 + len(line)
        kept += 1
    missed = len(notes) - kept
    if missed:
        lines.append(f"({missed} further standing note{'s' if missed > 1 else ''} did not "
                     f"fit — not everything known.)")
    return "\n".join(lines)


def standing_block(store: Any, *, hosted: bool, budget: int, header: str,
                   fallback: "Callable[[], str]", cwd: str = "") -> str:
    """Every standing preference this store holds, as a block ready to inject.

    `fallback` is the caller's old query-shaped call, used only when every queryless route
    is unavailable. It stays in the caller rather than here so that this module contains no
    query at all -- there is no sentence in this file for a future edit to start ranking by.
    """
    routes = (_from_tool, _from_since) if hosted else (_from_local,)
    for route in routes:
        try:
            notes = route(store)
        except Exception:
            continue
        if notes:
            mine = [n for n in notes if _mine(n.subject, cwd)]
            if mine:
                return render(_order(mine), header, budget)
    try:
        return str(fallback() or "")
    except Exception:
        return ""
