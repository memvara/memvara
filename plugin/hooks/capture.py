#!/usr/bin/env python3
"""Stop — mine the turn that just ended for anything worth knowing next week.

This runs once per turn and looks at one turn: the prompt the user typed and the reply it
got. Nothing earlier, because the earlier turns were mined when they happened.

It mines both halves because they hold different things. The prompt carries standing
instructions, the reply carries what was decided and where it landed, and a fact usually
needs both to be worth storing. Mining the reply alone was tried first and failed in a way
worth recording: it asks a model for facts *about the user* in Claude\'s own words, and the
model correctly finds none. Fifteen runs in one hour, an empty list every time, a full
extraction paid for on each.

That is a deliberate reversal of how this hook used to work, and the reason is a defect
rather than a preference. It used to batch — hold text back until 2000 characters had
accumulated, then mine the tail of it — because a headless extraction costs about 21k
tokens of Claude Code's own preamble whatever it is handed, so batching amortised the
overhead. But it kept only the last 48 formatted lines of whatever it read while advancing
its watermark past *all* of it, so on a session with large tool outputs most of the
transcript was skipped unread and could never be reconsidered. Measured on one session:
630KB consumed, six extractions paid for, and only the tail of each batch ever seen.

**It runs async, and therefore silently.** Extraction shells out to `claude -p` and takes
12-14 seconds, and a synchronous `Stop` hook holds the turn open for all of it. Async hands
the turn back immediately -- but the client discards an async hook's output, so the
`systemMessage` this used to print could not survive the change and is gone rather than
merely unread.

That reverses a rule this repository states in CLAUDE.md, and the reason it was stated is
still true: a hook nobody can see working is one nobody notices breaking. The compensating
channel is `~/.memvara/.hooks/capture.log`, and the obligation moved there rather than
disappearing -- every path that reaches a decision writes a line, including the ones that
decide to do nothing.

`recall.py` and `session_start.py` are the second half of that obligation, not exempt from
it. Neither is this hook's own async -- both are synchronous, both already speak on every
event they answer, and `lib.ipc.raise_capture_alert`/`due_capture_alert` are how a failure
here reaches whichever of them a person is actually watching next: on the terminal, as
`⋈ Memvara · ... · capture failing: <reason>` riding on a banner that file was printing
regardless. capture.log stopped being the only account of a failure the day a headless
`claude -p` login sat expired for 34 hours with nothing anywhere saying so out loud.

Per-turn costs more and loses nothing. The two guards that remain:

* **It writes.** A hosted install has no local store, so writes go over the MCP endpoint
  and a refusal raises rather than returning quietly. See `lib/write.py`.
* **It repeats.** `Stop` can fire more than once over one reply, so the size of the
  transcript at the last run is recorded and an unchanged size means there is nothing new.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.envelope import read_event  # noqa: E402
from core.host import active  # noqa: E402
from lib.extract import project_subject, triples  # noqa: E402
from lib.ipc import payload  # noqa: E402
from lib.transcript import last_turn_with_injections  # noqa: E402
from lib.write import (EPISODE_ROLE, log, open_writer, store_facts,  # noqa: E402
                       turn_ids)

#: How much of the tail to parse looking for the turn boundary. One turn is far smaller
#: than this; the margin is for a turn with a lot of tool traffic in it.
TAIL_BYTES = 512 * 1024

#: Characters of the exchange handed to the extractor. A very long turn is truncated from
#: the front, keeping the end, because the conclusion of a turn carries the decisions.
MAX_TURN_CHARS = 20_000

#: Where the per-transcript sizes live. Beside the store, not in the plugin, which is
#: replaced wholesale on update.
STATE = Path.home() / ".memvara" / ".hooks" / "capture-state.json"



#: Prompts that carry no fact and never will. A turn whose whole typed input is one of
#: these is the user saying "carry on", and mining it costs a full headless run -- about
#: 21k tokens of Claude Code's own preamble -- to be told there is nothing here.
CONTINUATIONS = frozenset({
    "y", "yes", "yep", "yeah", "ok", "okay", "k", "sure", "go", "go ahead", "do it",
    "continue", "carry on", "proceed", "next", "please", "thanks", "thank you", "ty",
    "try again", "again", "fix it", "run it", "same", "and", "more",
})

#: A prompt containing one of these is mined however short it is. The length rule below is
#: a guess about whether a fact is present; these are evidence, and evidence wins.
SIGNAL = (
    "remember", "decision", "decide", "prefer", "always", "never", "instead",
    "convention", "rule", "standard", "policy", "approach", "architecture", "design",
    "tradeoff", "because", "bug", "broken", "root cause", "deprecat", "migrate",
)

#: Below this, with no signal word, a prompt is treated as a continuation. Deliberately
#: small. A skipped turn is a fact lost with no trace, and the saving is one run; the
#: asymmetry says to guess in favour of mining.
MIN_PROMPT_CHARS = 12


def _did_something(turn: str) -> bool:
    """Whether the assistant changed anything in this turn.

    Checked before the prompt is judged short or conversational, because those two rules
    are guesses about whether a fact is present and this is evidence. `format_assistant`
    writes one `Claude used <tool>` line per tool it allows through, and the tools it
    allows through are the ones whose *use* is a durable event -- a file edited, a command
    run. See `transcript.INCLUDE_TOOLS`.
    """
    return any(line.startswith("Claude used ") for line in turn.splitlines())


def _worth_mining(turn: str) -> "tuple[bool, str]":
    """Whether this turn justifies a paid extraction, and why not when it does not.

    The cost model is the whole argument. A headless run spends ~10 fresh input tokens on
    the transcript it is handed and ~21k on Claude Code's own preamble regardless, so the
    bill scales with the NUMBER of runs and not at all with their size. Cutting the runs
    that cannot contain a fact is therefore the only lever that exists while capture stays
    per-turn -- and per-turn is deliberate: batching was removed in 0.1.3 because it
    advanced its watermark past text it never read.

    **The prompt alone is not that test, and the log said so.** `last_turn` mines both
    halves because they carry different things -- a standing instruction is stated in the
    prompt, and what was actually decided and where it landed is in the reply -- so a gate
    reading only the prompt throws away the half its own module says the decisions are in.
    Measured on one machine: `turn=6476c skipped=prompt too short (8c)`, six times over,
    the largest a 6,476-character reply discarded because somebody typed eight characters.

    Short imperatives are how work gets authorised: "merge #55", "deploy it", "ship it",
    and the whole of `CONTINUATIONS` -- "do it", "go ahead", "run it". Those are exactly
    the prompts that precede a reply worth keeping.

    So a turn where the assistant *did* something is mined whatever the prompt looked like.
    `Claude used Edit`, `Write`, `Bash` and `NotebookEdit` lines are already in the
    formatted turn (`transcript.INCLUDE_TOOLS`), and they are evidence rather than a guess
    about size: a file changed or a command ran. A long reply that only explains something
    is still skipped, correctly -- under the attribution rules the extractor now follows,
    the assistant's own prose is not evidence for a fact anyway, so paying for that run
    buys nothing.
    """
    typed = " ".join(
        line[len("User: "):] for line in turn.splitlines() if line.startswith("User: ")
    ).strip()
    if not typed:
        return False, "no typed prompt"
    low = typed.lower()
    if any(word in low for word in SIGNAL):
        return True, ""
    if _did_something(turn):
        return True, ""
    if low.strip(".!? ") in CONTINUATIONS:
        return False, "continuation"
    if len(typed) < MIN_PROMPT_CHARS:
        return False, f"prompt too short ({len(typed)}c)"
    return True, ""


def _read_state() -> dict:
    import json

    try:
        data = json.loads(STATE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_state(state: dict) -> None:
    """Persist the per-transcript watermarks, minus the ones that describe nothing.

    A key is added for every transcript this machine has ever mined and none was ever
    removed, so the file grew monotonically -- 169 entries after two days, one of them
    already naming a transcript that no longer existed. It is parsed and rewritten on every
    `Stop`, so it is on the per-turn path, which is what makes unbounded growth worth a
    line rather than a shrug.

    A transcript that is gone cannot be mined again, so its watermark can never be read.
    Dropping it here costs one `exists` per key at write time and removes the entry exactly
    when it stops meaning anything.
    """
    import json

    try:
        state = {key: size for key, size in state.items() if os.path.exists(key)}
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(state), encoding="utf-8")
    except OSError:
        # A lost marker costs one repeated extraction, not correctness.
        pass


def _turn(transcript: Path) -> "tuple[str, list[str]]":
    """The turn that just ended, and the memories this plugin injected into it.

    The second half is not decoration. Recall puts stored notes in front of the model
    before it replies; if the reply restates one, mining it writes the store's own output
    back into the store as though it were something new. Handing them to the extractor is
    what lets it tell an observation from an echo.
    """
    try:
        size = transcript.stat().st_size
        with transcript.open("rb") as fh:
            fh.seek(max(0, size - TAIL_BYTES))
            raw = fh.read()
    except OSError:
        return "", []
    text, injected = last_turn_with_injections(raw)
    return (text[-MAX_TURN_CHARS:] if len(text) > MAX_TURN_CHARS else text), injected


def main() -> int:
    host = active()
    if host.transcript is None:
        # This client hands a hook nothing to read back, so there is no turn to mine.
        log("skipped=host keeps no transcript")
        return 0
    event = read_event(host, "capture", payload())
    if event.reentrant:
        # Re-entry from a hook-triggered continuation. Mining here would double-count.
        return 0

    if not event.transcript_path:
        return 0
    transcript = Path(event.transcript_path).expanduser()
    if not transcript.is_file():
        return 0

    key = str(transcript.resolve())
    try:
        size = transcript.stat().st_size
    except OSError:
        return 0
    state = _read_state()
    if state.get(key) == size:
        # Stop fired twice over one reply. Nothing has been added since the last run.
        return 0

    turn, injected = _turn(transcript)
    if not turn.strip():
        log("no turn to mine")
        return 0

    # Recorded before extraction, not after: a run that dies mid-way must not leave the
    # same reply queued for the next turn to pay for again.
    state[key] = size
    _write_state(state)

    worth, why = _worth_mining(turn)
    if not worth:
        log(f"turn={len(turn)}c skipped={why}")
        return 0

    store, close = open_writer()
    if store is None:
        log(f"turn={len(turn)}c stored=0 failed=no store or login")
        return 0

    try:
        kept, turn_of = _keep_turn(store, turn, event.cwd)
        facts = triples(turn, event.cwd or None, injected=injected)
        if not facts:
            log(f"turn={len(turn)}c facts=0 episode={'yes' if kept else 'no'}")
            return 0
        stored, failed = store_facts(store, facts, turn, hosted=close is not None,
                                     sources=turn_of)
    finally:
        if close is not None:
            close()

    log(f"turn={len(turn)}c facts={len(facts)} stored={stored} "
        f"episode={'yes' if kept else 'no'}"
        + ("; failed=" + "; ".join(failed) if failed else ""))

    return 0


def _keep_turn(store: object, turn: str, cwd: str) -> "tuple[bool, list[str]]":
    """Store the turn itself as an episode. `(landed, ids of the turn)`.

    The ids are what make the facts below explainable. This used to discard `add`'s return
    value entirely, which is half of why `memory_why` answered "No source turns are
    retained" for every hosted claim: the other half was that the tool did not accept
    `sources`, and each made the other pointless. The id never left the process.

    Empty on the local route by design -- `Memvara.add` returns a receipt object rather
    than text, and `store_facts` passes a real `Episode` there instead -- and empty on any
    hosted server without memvara/memvara#76, which is what renders the ids at all.

    Triples are a lossy reading of a conversation, and the loss is the part a later session
    most wants: the reasoning, the alternative that was rejected, the sentence the user
    actually typed. This keeps the turn as well, so recall has something to return that is
    narrative rather than a slot value.

    It is free where it matters, and it is stored under `EPISODE_ROLE` rather than
    "user" so that it stays free. On a `fast-path-only` server -- which is what the hosted
    endpoint reports -- `add` commits the episode and calls no model, so this costs one
    round trip and no tokens. What it does *not* do is call no extractor: the deterministic
    fast path runs on every user turn whatever the model configuration is. From 28acf2c
    (2026-08-24) until this change, six days, that made every captured turn a second write
    path nobody here had vetted -- and `bb47d77` had already done it for a day on
    2026-08-21. See `lib.write.EPISODE_ROLE` for what it cost.

    The history is worth one more line, because the fix was once in this file and left.
    `6657332` removed the `store.add()` call on 2026-08-21 with a comment saying "under
    NullLLM prose is accepted and silently stores nothing" -- right that no model reads it,
    wrong that nothing is written, and the call came back three days later on the strength
    of that belief. The facts this hook means to write are the ones `triples()` reads and
    `store_facts()` writes, against a fixed vocabulary at confidence 0.7, and now they are
    the only ones it writes.

    The turn is stored whole. Only the claim about who said it changed, so recall still has
    the narrative and `memory_why` still has something to show.

    Failure is not reported to the user. The claims are the load-bearing half; a missing
    episode degrades recall rather than breaking it, and a second red line in the terminal
    for it would train the eye to ignore the first.
    """
    add = getattr(store, "add", None)
    if add is None:
        return False, []
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    project = project_subject(cwd or None)
    try:
        receipt = add(f"[session turn · project {project} · {stamp}]\n{turn}",
                      role=EPISODE_ROLE)
        return True, turn_ids(receipt)
    except Exception:
        return False, []


if __name__ == "__main__":
    raise SystemExit(main())
