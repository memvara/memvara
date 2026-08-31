#!/usr/bin/env python3
"""UserPromptSubmit — put what is already known in front of the model, unasked.

This is the hook that makes stored memory feel like memory rather than like a database
somebody has to remember to query. Without it, recall happens only when the model decides
to call `memory_recall`, which is exactly the decision it cannot make reliably: it has to
already suspect the fact exists.

Cost is the reason this reads SQLite directly instead of speaking MCP. It runs on every
prompt, so it is measured, not assumed: 0.22 s cold on a 25-claim store, interpreter
startup included.

**It says which of three things happened, and that is not cosmetic.** Recalled, nothing
relevant, and could-not-ask used to produce one message between them. A hosted client whose
session id had gone stale answered every query with silence for the rest of a session while
this hook cheerfully reported "no matching notes" each time -- indistinguishable, from the
terminal, from a store that was simply empty, and nobody investigates an empty store. The
three states now read differently, and `lib.fast.recall` returns the flag that tells them
apart.

**It answers "yes please" with the last thing that was about something.** The query used to
be the prompt, verbatim, and a prompt that is purely a reply to the previous turn has
nothing in it to retrieve on -- a vector search over two function words returns arbitrary
neighbours. Measured on a real store: a turn approving a memory cleanup was handed notes
about pricing tiers, free-tier seat counts and an unrelated project's zip layout. Never
wrong, never an error, and the whole block's budget spent on noise.

The last substantive prompt is kept beside the seen-hashes and prepended when the new one
is anaphoric. Prepended rather than substituted, because "yes, add that fix to #7" still
carries "#7" and dropping it would trade one blindness for another.

**It does not repeat itself.** A memory injected on turn 1 is still in the conversation on
turn 5, so injecting it again buys nothing and spends budget that a genuinely new memory
could have had. Hashes of what has already gone in are kept per session and filtered out,
so a follow-up gets whatever is new and a banner saying how much it already had.

It does not write. Recording what was said is the `Stop` hook's job, over the prompt and
the reply together, in one run: two runs per turn cost twice as much and each saw half the
evidence.
"""

from __future__ import annotations

import hashlib
import json
import os.path
import time
import sys

# `os.path`, not `pathlib`: importing pathlib costs 10.5ms and this file runs on every
# prompt. The bootstrap is one string join.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.envelope import read_event, write  # noqa: E402
from core.host import Reply, active  # noqa: E402
from lib.fast import recall as fast_recall  # noqa: E402
from lib.ipc import (  # noqa: E402
    due_alert_for_model, due_capture_alert, log_line, payload, plural, status,
    under_extraction, with_alert,
)

#: The client this process is answering, resolved once. `run.py` binds it before importing
#: this module; a bare `python3 recall.py` gets Claude Code, which is what that invocation
#: has always meant.
HOST = active()

#: Enough memories to be useful, few enough to stay out of the way. Recall drops whole
#: notes weakest-first to fit, so this is a ceiling and not a target.
#:
#: Four rather than six, and 300 rather than 700, because both were picked before anything
#: was measured. Four clipped memories carry more distinct facts than one unclipped one --
#: at 700 a single procedural note could take the whole block.
K = 4
BUDGET = 300

#: Characters kept from each memory when it is injected. Storage is untouched: the whole
#: note stays in the store, is what gets embedded, and is what `memory_search` returns.
#:
#: This is the difference between a memory worth keeping and a memory worth pushing. Making
#: procedural objects carry their reasoning is what stopped them being useless one-liners,
#: and it made each one about four times larger. Measured over eight real prompts against a
#: 222-claim store: median injected memory 48 tokens, p90 237, max 503, and **four lines
#: over 150 tokens accounted for 39% of every token injected**. Clipping alone, changing
#: nothing else, removes 51% of the block.
#:
#: It was 160, and 160 was cutting the half that carries the meaning. Two measurements
#: moved it, and the second is the one that decides.
#:
#: **The clip is not what bounds the block.** `budget` is, and it does it by dropping whole
#: notes -- "the largest prefix that fits" -- before this ever runs. So a block is already
#: under 300 tokens when it arrives here, and clipping does not lower a ceiling, it deletes
#: text from inside one. Cost belongs in the parameter that drops notes; this one only
#: decides whether the notes that survive are still readable.
#:
#: **And nothing goes and reads the rest.** Across 434 clipped injections in this machine's
#: transcripts, 4 were followed by a `memory_search` -- 0.9%, against 0.2% after an
#: unclipped one. Of 122 searches in the corpus, 118 had nothing to do with a clipped
#: recall. `MORE` is read as a disclosure, not acted on as a pointer, so a clipped tail is
#: not deferred: it is discarded.
#:
#: What that tail holds is not filler. The extraction rules ask an object to state the
#: instruction, then why it matters, then the detail that makes it applicable -- so the
#: operative clause is *by construction* last, and a head truncation takes exactly it.
#: Measured on the real store: "serves 12 tools while main documents 13 -- memory_standing
#: is not de|ployed", losing the half that says what to do about it.
#:
#: 320 is where the two bounds meet. At `K = 4` it puts the clip's own ceiling (~1,280c)
#: past `BUDGET`'s (~1,200c), so the budget becomes the binding constraint again, which is
#: the one designed to bind. Against a 376-claim store it takes whole delivery from 36% to
#: 57%, and a third of everything currently truncated arrives intact. Episodes stay
#: pointers -- a 1,853-character median episode is still cut to a fraction of itself.
MAX_INJECTED_CHARS = 320

#: Said once, when anything was actually shortened. Roughly twelve tokens.
#:
#: It was written as a pointer -- "the model can go and read the rest" -- and measurement
#: does not support that reading. Across 434 clipped injections, 4 were followed by a
#: `memory_search`. It is kept anyway, for the job it does do: telling a reader that text
#: was elided, so a truncated claim is not mistaken for a complete one. That is worth
#: twelve tokens; being a pointer is not something it has been observed to be.
MORE = "(excerpts — memory_search returns any of these in full)"

#: Below this many fresh memories, ask again for the raw turns as well. A prompt the
#: structured layer had little for is exactly the case where narrative excerpts cannot
#: outrank anything, which is the objection that keeps them off by default.
#:
#: One, not two. Two was calibrated against a 700-token budget returning six memories;
#: at 300 returning one to three, "fewer than two" is the normal case and the escalation
#: fired on 5 of 8 real prompts instead of 2 of 8 -- an extra round trip on most turns,
#: bought by tightening the budget it was tuned against.
THIN = 1

#: The escalation's own k and budget, and both are LARGER than the claims pass. That is
#: not a lapse in the budget discipline everything else here follows -- it is what the
#: measurement says, and the first version of this got it exactly backwards.
#:
#: `k` is the candidate cap that episodes must win a slot inside, and episodes are
#: deliberately down-weighted against claims, so shrinking k to "bound" the escalation
#: guaranteed no episode could ever place. Measured against the deployed server on a query
#: whose answer is a stored turn:
#:
#:     k \ budget    300    600   1200   2000
#:     k=2            -      -      -      -
#:     k=4            -   episode episode episode
#:     k=6            -      -      -   episode
#:
#: Not monotonic, and the reason is the interaction: more claim slots means claims fill the
#: budget first and crowd the episode out, so a larger k needs a larger budget to show the
#: same result. `k=2, budget=300` -- what this shipped with -- is the dead zone. It fired on
#: most prompts and could not return an episode at any budget.
#:
#: So: select generously, inject tersely. The budget gates what is *selected*; MAX_INJECTED
#: _CHARS gates what is *sent*, and clips a 1,853-character median episode to a pointer.
EPISODE_K = 4
EPISODE_BUDGET = 600

#: Wall-clock ceiling on the *optional* work in one invocation, measured from the top of
#: `main()`. `hooks.json` gives this hook 10 seconds total, and nothing here tracks
#: cumulative time across the up-to-three hosted calls one invocation can make -- the main
#: `fast_recall()`, the episode-widening retry, and `_standing_refresh()`'s own hosted call
#: -- so a slow or flaky connection could spend the whole budget on retries and have the
#: harness kill the process with nothing printed: no `systemMessage`, no banner, no line in
#: `recall.log` saying why. `lib.hosted.TIMEOUT_SEC` is 6s per network call and each call
#: may retry once, so a single `HostedRecall` round trip that still needs its handshake can
#: cost up to 4x that alone -- the arithmetic this constant exists to not have to survive.
#:
#: 7.5s leaves 2.5s of the 10s for interpreter startup, the primary `fast_recall()` call
#: this file has never skipped, and JSON emission -- none of them measured in the single
#: digits of milliseconds this file's other budgets are, because none of them can be timed
#: from inside a process about to be killed by the thing measuring them. The margin is
#: deliberately generous rather than tight against that 10s: this is the harness's own
#: kill timer, not a cost to shave, and headroom here is what keeps a process a little slow
#: to exit from being confused with one that is hung.
#:
#: Gates the two calls this file can afford to skip -- the standing refresh and the
#: episode-widening retry -- and nothing else. The primary `fast_recall()` call is what
#: this hook exists to make and runs regardless of elapsed time; skipping it to stay inside
#: a budget would be answering the timeout by not doing the hook's own job.
#:
#: What this does NOT close: it is checked before starting a call, not while one is
#: already running, so it stops a SECOND slow call from compounding a first one but cannot
#: shorten a call already in flight. On a fresh process the connection cache above is
#: empty, so whichever hosted call happens to run first -- the standing refresh, if its own
#: 15-minute interval is due, or otherwise the primary call itself -- gets no benefit from
#: it and can still cost the full worst case on its own. If that first call is the standing
#: refresh, in the worst case it alone can outlast this hook's entire 10s allowance before
#: the primary call the budget was written to protect ever starts. Closing that fully would
#: mean bounding the DURATION of an in-flight call -- a deadline enforced inside
#: `lib.hosted` itself, shared by every caller of it, not a clock kept in this one file --
#: which is a deeper change than a wall-clock gate on whether to start a second one.
OVERALL_BUDGET_SEC = 7.5

#: Prompts that are not questions to the model: a slash command, a bash escape, a comment.
#: Silence is right for these -- the user typed a command and is not waiting on memory.
#:
#: The prefixes themselves belong to the client -- every editor spells its own command
#: escape, and one that reads `#` as a heading rather than a comment would go silent on
#: every prompt that opened with one -- so they live in its `Host` record and this name is
#: what the body reads them by.
#:
#: There is deliberately no minimum length rule beside this. One was written and taken back
#: out: it skipped short follow-ups on the theory that whatever they would have matched was
#: injected earlier and is still in context, which is true often enough to be tempting and
#: wrong exactly when it matters -- an early short question in a fresh session would get
#: nothing, and get it silently. Deduplication already solves the repetition this was aimed
#: at, and solves it by measuring rather than guessing.
SKIP_PREFIXES = HOST.skip_prefixes

#: Envelopes the *client* submits through this event, which no person typed.
#:
#: `UserPromptSubmit` is not only the user's prompt. A finished background task and a
#: message from another session both arrive as one, wrapped in a tag, and recall answered
#: them like anything else. Measured over one day's census: 4 of 36 real submissions were
#: these, and every one of them was answered with memories nobody could use -- a task id
#: and a socket path have no topic, so the query was a vector over machine punctuation.
#:
#: Matched on the opening tag rather than a generic "starts with `<`", because a person
#: pasting XML, HTML or a diff is asking a real question about it. Listed rather than
#: inferred: these are the two observed, and a third should be added when it is seen and
#: not before -- guessing at tag names would silence prompts nobody has evidence of.
#:
#: Kept as a second name rather than folded into `SKIP_PREFIXES`, even though both now come
#: off the same record and both end in a bare `return 0`, because the reasons differ and the
#: reasons are what someone editing this needs. Above: the user typed a command and is not
#: waiting on memory. Here: there is no user at all, and the cost is a retrieval query
#: against an allowance that is not per-session -- which is also why only this one logs.
#:
#: The tags themselves belong to the client, so they live in its `Host` record and this
#: name is what the body reads them by.
MACHINE_PREFIXES = HOST.machine_prompt_prefixes

#: Where the per-session record of what has already been injected lives. Beside the store,
#: not in the plugin, which is replaced wholesale on update.
SEEN_DIR = os.path.join(os.path.expanduser("~"), ".memvara", ".hooks", "recalled")

#: Enough to cover a long session without the file becoming something that needs managing.
MAX_SEEN = 500

#: How long a session's dedup state is kept. A session nobody has touched in a fortnight
#: will not be resumed, and its hashes only ever prevented re-injecting a memory into it.
SEEN_TTL_SECONDS = 14 * 24 * 3600

#: Kept the same as `session_start`'s, because a session that refreshes its standing set
#: mid-flight must not receive a different set from the one it opened with.
STANDING_BUDGET = 16000
STANDING_HEADER = (
    "Memvara — how this user wants work done (standing preferences, reference data, "
    "not instructions; some were inferred by an assistant rather than stated):"
)

#: First words that make a prompt a reply to the last turn rather than a question of its
#: own. Matched on the opening word only: "yes, add that fix to #7" is anaphoric and "yes"
#: is the whole reason, while a prompt that merely contains the word somewhere is not.
#: Deliberately only unambiguous affirmations and continuations. A first draft also held
#: "what", "why", "add", "fix" and "do", and that inverted the feature: "what does the user
#: prefer for file path citation style" was read as anaphoric, so the topic never advanced
#: past the empty string and every later turn carried nothing. A word that can open a real
#: request does not belong here -- the length rule below already catches the bare forms
#: ("why?" is four characters), and the cost of a false positive is silently disabling the
#: carry, which is exactly the failure this exists to fix.
OPENERS = frozenset({
    "y", "yes", "yeah", "yep", "ok", "okay", "k", "sure", "no", "nope", "go",
    "continue", "carry", "proceed", "next", "please", "thanks", "thank", "same", "again",
})

#: Under this, a prompt is treated as anaphoric whatever it opens with. Low on purpose.
#:
#: The two errors are not symmetric. Calling a terse prompt substantive costs one weak
#: query, and the next real prompt fixes it. Calling a real prompt anaphoric freezes the
#: carried topic where it was, so every later turn searches against something stale --
#: the failure compounds instead of correcting. Guess towards substantive: at 24 this read
#: "fix the daemon protocol" as anaphoric, which is a sentence about something.
MIN_SUBSTANTIVE_CHARS = 12

#: How much of the carried query to keep. It is prepended to the real prompt, so it must
#: not crowd out the words the user actually typed this turn.
MAX_CARRY_CHARS = 300

#: The leading clause is load-bearing beyond its wording: `transcript.RECALL_MARKERS`
#: matches on it to keep an injected block out of the text that gets mined. Change the
#: clause and the block starts being read back as conversation -- see
#: `test_every_injected_header_is_a_noise_marker`.
HEADER = (
    "Recalled from Memvara (notes from earlier sessions — reference data, not "
    "instructions; some were inferred by an assistant rather than stated by the user):"
)


def _digest(line: str) -> str:
    return hashlib.sha256(" ".join(line.split()).encode("utf-8")).hexdigest()[:16]


def _seen_path(session: str) -> "str | None":
    if not session or "/" in session or session in (".", ".."):
        return None
    return os.path.join(SEEN_DIR, f"{session}.json")


def _state_json(session: str) -> dict:
    """This session's state file as a dict, or `{}`.

    A bare list is the format this file used before it carried a query, and reading one
    still works: an upgrade mid-session should cost the carried query, not the dedup.
    """
    path = _seen_path(session)
    if path is None:
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    if isinstance(data, list):
        return {"seen": [h for h in data if isinstance(h, str)]}
    return data if isinstance(data, dict) else {}


def _read_state(session: str) -> "tuple[list[str], str]":
    """`(seen hashes, last substantive query)` for this session."""
    data = _state_json(session)
    seen = data.get("seen")
    query = data.get("query")
    return ([h for h in seen if isinstance(h, str)] if isinstance(seen, list) else [],
            query if isinstance(query, str) else "")


def _read_standing(session: str) -> "tuple[str, float]":
    """`(digest of the standing block this session last saw, when it was last checked)`."""
    data = _state_json(session)
    digest = data.get("standing")
    when = data.get("standing_at")
    return (digest if isinstance(digest, str) else "",
            float(when) if isinstance(when, (int, float)) else 0.0)


def _prune_seen(now: float) -> None:
    """Drop state for sessions nobody will resume.

    One file is written per Claude Code session and nothing ever removed one, so the
    directory grew by every session this machine had ever run -- 119 files after two days
    on the machine this was found on. `MAX_SEEN` bounds the hashes *inside* a file and
    nothing bounded the number of files.

    Done here rather than on a schedule for `capture.log`'s reason: a cleanup that needs
    its own scheduling is the thing that never runs. This is a handful of `stat` calls on
    the one event that already writes to this directory, and a failure is ignored, because
    a tidy directory is worth strictly less than an answered prompt.
    """
    try:
        for name in os.listdir(SEEN_DIR):
            if not name.endswith(".json"):
                continue
            path = os.path.join(SEEN_DIR, name)
            try:
                if now - os.path.getmtime(path) > SEEN_TTL_SECONDS:
                    os.unlink(path)
            except OSError:
                continue
    except OSError:
        pass


def _write_state(session: str, hashes: "list[str]", query: str,
                 standing: "tuple[str, float] | None" = None) -> None:
    """Persist this session's state, carrying the standing keys forward.

    `standing` is read-modify-write rather than an argument every caller must thread,
    because the two exit paths in `main` write state for reasons that have nothing to do
    with the standing set. Passing None from those would silently reset the refresh clock
    on every turn and re-inject the whole standing block each time -- the failure this is
    supposed to prevent, arriving through the tidier-looking signature.
    """
    path = _seen_path(session)
    if path is None:
        return
    was_digest, was_at = _read_standing(session)
    digest, at = standing if standing is not None else (was_digest, was_at)
    try:
        os.makedirs(SEEN_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"seen": hashes[-MAX_SEEN:], "query": query[:MAX_CARRY_CHARS],
                       "standing": digest, "standing_at": at}, fh)
        _prune_seen(time.time())
    except OSError:
        # Dedup and carry-forward are both optimisations. Losing them repeats a memory or
        # weakens one query; failing the prompt over it would be the larger bug.
        pass


def _anaphoric(prompt: str) -> bool:
    """True when this prompt refers to the conversation rather than describing anything.

    "yes please" retrieves nothing useful, because there is nothing in it to retrieve on:
    the query is the literal text and a vector search over two function words returns
    arbitrary neighbours. Measured on this store, a turn approving a cleanup was handed
    memories about pricing tiers, free-tier seat counts and an unrelated project's zip
    layout -- not wrong exactly, just noise, spending the block's whole budget on it.

    The test is the opening word, not the length. "yes, add that fix to #7" is long enough
    to look substantive and still says nothing a search can use; what carries the meaning
    is the turn before it.
    """
    words = prompt.lower().replace(",", " ").split()
    if not words:
        return True
    return words[0].strip(".!?") in OPENERS or len(prompt) < MIN_SUBSTANTIVE_CHARS


def _clip(line: str) -> str:
    """One memory line, shortened for injection. Never for storage."""
    if len(line) <= MAX_INJECTED_CHARS:
        return line
    return line[:MAX_INJECTED_CHARS].rstrip() + "…"


def _split(block: str) -> "tuple[str, list[str]]":
    """The block's header and its memory lines.

    Recall renders each memory as a `- ` bullet and everything else -- the header, and the
    trailing note about what did not fit -- as plain lines. Only the bullets are deduped,
    because only they are the memories.
    """
    lines = block.splitlines()
    bullets = [line for line in lines if line.startswith("- ")]
    header = lines[0] if lines and not lines[0].startswith("- ") else HEADER
    return header, bullets


#: Create this file to record what was injected next to the prompt it was injected for, so
#: somebody can read a sample and judge whether it earned its place. Its contents are never
#: read; existing is the whole signal.
#:
#: A file rather than an environment variable, and the difference is not taste. A hook is
#: spawned by the client, not by the shell somebody typed `export` into -- so an exported
#: variable reaches a session started afterwards, in a terminal that inherited it, and
#: silently does nothing otherwise. Somebody would turn sampling "on", see an empty log a
#: week later, and conclude recall was never called. A path is the same answer from every
#: process on the machine.
#:
#: It also sits in plain sight next to the logs it produces, which matters for a switch
#: meant to be turned off again: `ls` is how somebody discovers they left it on.
#:
#: Opt-in either way: this writes prompt text to a file, which is a surface nobody asked
#: for and most installs will never want. A measurement, not a feature -- turn it on for a
#: week, read fifty lines, remove the file.
SAMPLE_FLAG = os.path.join(
    os.path.expanduser("~"), ".memvara", ".hooks", "sample-recall")

#: How much of each string to keep. Enough to judge relevance by eye, short enough that a
#: line stays one line.
SAMPLE_PROMPT_CHARS = 90
SAMPLE_MEMORY_CHARS = 70


def _sample(prompt: str, memories: "list[str]", *, anaphoric: bool) -> None:
    """Record the prompt and what recall answered it with.

    **Not relevance scores, because there are none to record.** `recall()` returns rendered
    text; `RecallResult` carries ids and a dropped count and no score, and the hosted
    `memory_recall` does not even ask for the ids. Reaching a score means a second
    round trip to `memory_search` on the per-prompt path -- doubling the cost of the thing
    being measured -- or a change to the server. Neither is worth it to answer a question a
    person can answer by reading.

    So this logs the two strings that matter and lets a human be the judge. `anaphoric`
    comes along because it is the obvious confounder: a prompt that carried its topic
    forward was searched with different words than the ones somebody typed, and a bad
    injection there is a different bug than a bad injection on a prompt that stood alone.
    """
    if not os.path.exists(SAMPLE_FLAG):
        return
    def flat(text: str, n: int) -> str:
        return " ".join(text.split())[:n]
    parts = [f"carried={'y' if anaphoric else 'n'}",
             f"prompt={flat(prompt, SAMPLE_PROMPT_CHARS)!r}"]
    parts += [f"mem{i}={flat(m, SAMPLE_MEMORY_CHARS)!r}" for i, m in enumerate(memories, 1)]
    log_line("recall-sample", "  ".join(parts))


#: Retrieval below this score is not injected. `Memvara.recall` has taken `min_score`
#: since long before this hook existed and this file passed the 0.0 default, so every
#: prompt got its `k` slots filled whether or not anything in the store was about it --
#: "where should this helper live" answered with where the user lives, "start Plan B"
#: answered with the billing plan catalogue.
#:
#: Measured, not chosen. `memvara.calibrate_min_score` separates questions a store should
#: answer from plausible questions it should not, and on the plugin-recall benchmark's
#: seeded store the two classes were fully separable with the floor at 0.2975 -- 15 of 15
#: answerable kept, 22 of 22 unanswerable silenced. Rounded down to 0.29, because the
#: calibrator places the floor midway between the best wrong answer and the weakest right
#: one and rounding toward recall is the cheaper error: a missed memory costs one prompt,
#: a wrong one can steer a whole turn.
#:
#: **Scores are not comparable between embedders**, so this default is right for the
#: configuration it was measured on and is a starting point everywhere else. Recalibrate
#: against your own store and set `MEMVARA_RECALL_MIN_SCORE`:
#:
#:     python -m benchmarks.plugin_recall.calibrate --db ~/.memvara/store.db
MIN_SCORE = 0.29


def _min_score() -> float:
    """The configured floor. A bad value disables filtering rather than the hook.

    `0` is a legitimate setting -- it restores the old unfiltered behaviour for anyone who
    wants it -- so it is honoured rather than treated as unset.

    Clamped at both ends, and the upper end is not tidiness. Scores never exceed 1.0, so a
    value above it filters everything: the local route would return nothing on every prompt
    for the life of the setting, indistinguishable from an empty store, while the hosted
    route rejects the same value outright and degrades to no floor at all. One typo would
    otherwise mean total silence on one route and no filtering on the other -- exactly the
    divergence between routes that carrying this argument at all is meant to prevent.
    """
    raw = os.environ.get("MEMVARA_RECALL_MIN_SCORE")
    if raw is None:
        return MIN_SCORE
    try:
        return min(1.0, max(0.0, float(raw)))
    except ValueError:
        return MIN_SCORE


#: How often a running session re-checks whether its standing preferences have changed.
#: `SessionStart` fires ONCE, so a rule written after a session opened never reached it --
#: measured on a session that started a full day before the rule it needed existed and was
#: still breaking it eighteen hours later.
#:
#: A quarter hour rather than every turn, and the difference is the whole design. Standing
#: preferences are fetched at the top of a session precisely so they do not compete per
#: prompt with the facts that prompt is about; re-asserting them every turn would undo
#: that and undo PR #8 with it. What runs on every prompt is a timestamp comparison
#: against a small JSON file. What runs four times an hour is one query.
#:
#: It was thirty minutes because the query cost 1,345 ms: `memory_standing` was not
#: deployed, so the block came through `memory_since` from an early instant, which
#: downloads the whole store -- 160 KB and 371 claims -- to select 32 procedural ones.
#: `app.memvara.dev` now serves it, the plugin picked it up through `accepts()` with no
#: change here, and the same block costs **222 ms**. Four of those an hour is less wall
#: clock than two of the old ones, so the interval halves and the cost still falls.
STANDING_REFRESH_SECONDS = 15 * 60


def _standing_refresh(session: str, now: float, cwd: str = "") -> "tuple[str, tuple[str, float] | None]":
    """`(block to inject, state to persist)` -- both empty when there is nothing to say.

    Three ways to say nothing, and they are deliberately not the same code path:

    * the interval has not elapsed -- costs one float comparison and, importantly, no
      import: `lib.write` pulls in the library, which is ~95ms, and this file runs on every
      prompt;
    * the set is unchanged since this session last saw it -- costs the query and injects
      nothing, which is the common case after the first refresh;
    * the lookup failed -- costs the query and injects nothing, and the clock is still
      advanced so a store that is down does not turn every prompt into a retry.

    The digest covers the rendered block, so a claim being RETIRED changes it exactly as an
    addition does. A preference the user withdrew has to stop being asserted, and a digest
    over "what was added" would never notice it going.
    """
    digest, checked = _read_standing(session)
    if now - checked < STANDING_REFRESH_SECONDS:
        return "", None

    try:
        from lib.standing import standing_block  # noqa: PLC0415
        from lib.write import open_writer  # noqa: PLC0415

        store, close = open_writer()
        if store is None:
            return "", (digest, now)
        try:
            block = standing_block(store, hosted=close is not None,
                                   budget=STANDING_BUDGET, header=STANDING_HEADER,
                                   fallback=lambda: "", cwd=cwd)
        finally:
            if close is not None:
                close()
    except Exception:
        # A standing refresh that fails must not fail the prompt, and must not retry on
        # the next one either.
        return "", (digest, now)

    fresh = _digest(block)
    if not block.strip() or fresh == digest:
        return "", (digest or fresh, now)
    return block.rstrip(), (fresh, now)


#: What the server calls a spent allowance, as `lib.fast` hands it over: `quota` alone, or
#: `quota:2026-09-01` when the refusal named the instant the period rolls over.
_QUOTA = "quota"

#: Month names for the one date this file renders. `datetime.strftime` would do it in a
#: line and cost an import on a path measured at ~30ms, where `import datetime` is a
#: measurable share of the budget. Twelve strings are cheaper than a module.
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _quota_line(why: str) -> str:
    """The banner for a spent allowance, or `""` when the failure was something else.

    Says *retrieval* rather than the metric's own name: `retrieval.query` is what the
    server meters and not a phrase anyone reads. Says the reset date because "spent" on
    its own reads as "broken, retry later", and retrying is precisely what will not work.
    """
    if not why.startswith(_QUOTA):
        return ""
    _, _, when = why.partition(":")
    parts = when.split("-")
    if len(parts) == 3 and parts[1].isdigit() and 1 <= int(parts[1]) <= 12:
        day = parts[2].lstrip("0") or parts[2]
        return f"retrieval quota spent — resets {day} {_MONTHS[int(parts[1]) - 1]}"
    return "retrieval quota spent"


#: A subject of this shape names one checkout on this machine and nothing else. The
#: convention is the user's own -- `project:<absolute path>` for a fact that is true of one
#: working tree -- and these are written by hand through `memory_remember`, not by the
#: capture hook, which files under a git-remote name instead.
PROJECT_SUBJECT = "- project:"


def _belongs_here(bullet: str, cwd: str) -> bool:
    """Whether a checkout-scoped memory belongs to the checkout we are in.

    Everything that is not checkout-scoped passes untouched: a fact about `memvara_web`, or
    about the user, is cross-cutting and is exactly what recall is for. Only a subject that
    names an absolute path is claiming to be about one tree, and a query has no way to know
    which tree it was asked from -- `memory_recall` takes a query, a k and a budget, and no
    scope, so the server cannot filter this and the ranker never sees the question.

    Measured over one day's census: five memories from three unrelated checkouts --
    `Desktop/snorkel` three times, `expense-tracker`, `ai_app` -- landed in memvara
    sessions, including on a prompt asking which observability tool to use, where both
    answers came from other projects. The standing block has filtered on `cwd` since 0.2.0;
    the per-prompt path never has.

    Dropping rather than replacing is deliberate and is a real limit: the slot is not
    refilled, because the candidates were ranked and cut to `k` before this sees them. Three
    relevant notes beat three relevant notes and one about a different repository, so it is
    still the right trade -- but the honest description is that this removes noise and does
    not add signal.

    An unreadable `cwd` keeps everything. The failure to avoid is a hook that silently stops
    recalling, and dropping notes on the strength of a path we could not read would be it.
    """
    if not bullet.startswith(PROJECT_SUBJECT) or not cwd:
        return True
    rest = bullet[len(PROJECT_SUBJECT):]
    if not rest.startswith("/"):
        # Not a path after all. Left alone rather than guessed at.
        return True

    # Walk up from here and ask whether the subject names this directory or one containing
    # it, requiring the subject to END where the candidate does -- the next character must
    # be the space before the predicate, or nothing at all.
    #
    # Written this way rather than by splitting `rest` on whitespace, because a path may
    # contain a space: `/Users/me/My Project` split to `/Users/me/My`, so a memory filed
    # for a directory was dropped from that very directory. Recalling less, silently, which
    # is the failure this file exists to prevent.
    #
    # The end-of-subject requirement is also what rejects a sibling. Walking to `/A/B` and
    # accepting any separator would match a fact filed for `/A/B/claude-memvara` against a
    # session in `/A/B/claude-memvara-old`; insisting the subject stops there does not.
    node = os.path.abspath(cwd)
    while True:
        if rest.startswith(node) and rest[len(node):len(node) + 1] in ("", " "):
            return True
        parent = os.path.dirname(node)
        if parent == node:
            return False
        node = parent


def main() -> int:
    # The clock the optional hosted work below is measured against -- see
    # OVERALL_BUDGET_SEC. `monotonic`, not `time.time()`: this is an ELAPSED-time budget,
    # and `daemon.py` already uses `time.monotonic()` for its own idle-timeout for the same
    # reason. A wall clock can step backward -- an NTP correction, a VM resuming from
    # suspend -- and a step big enough would make the elapsed-time comparison negative
    # forever, silently disabling the one guard this file added against a hook the harness
    # kills with nothing printed. Started before the cheap checks below rather than after
    # them so it covers the whole invocation, even though nothing before the first hosted
    # call is expensive enough to matter in practice.
    start = time.monotonic()

    if under_extraction():
        # The prompt in front of us is `capture.py`'s own extraction request, not a
        # person's. Answering it spends a retrieval query and injects the standing block
        # into the one context that must be judged on its own words. Logged rather than
        # returned in silence, because a guard nobody can count is one nobody notices
        # losing: these lines are what say it is still standing between the two.
        log_line("recall", "skipped=under extraction")
        return 0

    # `payload()` is the raw stdin object and is the same everywhere; `read_event` is what
    # knows which keys this client puts a prompt and a session id under. Splitting them
    # matters because the miss is silent: the dedup file is keyed on session, so a renamed
    # key re-injects every memory on every turn while every banner still reads healthy.
    event = read_event(HOST, "recall", payload())
    prompt = event.prompt.strip()
    session = event.session

    if not prompt or prompt.startswith(SKIP_PREFIXES):
        return 0
    if prompt.startswith(MACHINE_PREFIXES):
        # Logged, not silent. This is the same shape as the extraction stand-down: a query
        # not spent is invisible, and a guard that stops firing has to be noticeable
        # somewhere before the allowance runs out again.
        log_line("recall", "skipped=machine prompt")
        return 0

    # Read once, then every reply from here on goes through `_emit` rather than
    # `write` threaded by hand through each call site -- a first version wrapped five
    # separate sites individually, and only one of the five was ever covered by a test; a
    # sixth site added later without the wrap would have printed a perfectly valid banner
    # and failed nothing. `_emit` means every status line below picks this up whether
    # or not whoever writes the next branch remembers this file relays a capture failure
    # at all -- named differently from `write` on purpose: shadowing the imported name
    # with a same-named local function makes every reference to it inside this function
    # local from the top, including the one capturing the original, which raises
    # `UnboundLocalError` before it ever runs.
    alert = due_capture_alert()

    def _emit(reply: Reply) -> None:
        if reply.status:
            reply = reply._replace(status=with_alert(reply.status, alert))
        # Called here, at the point delivery is actually about to happen, rather than once
        # at the top of `main()` -- `due_alert_for_model` persists "the model has now been
        # told" as a side effect of deciding what to say, and everything between the top of
        # `main()` and this point (`_read_state`, `_standing_refresh`, `_anaphoric`,
        # `fast_recall`) is exactly the code most likely to grow a new call that can raise.
        # A raise anywhere in that span, with the decision already made and persisted
        # earlier, would mark a notice "told" that never actually reached the model -- worse
        # than the repetition this function exists to prevent, because nothing would ever
        # correct it short of the reason changing. Calling it from inside `_emit`, one line
        # before `write` actually runs, leaves nothing but field merges in between.
        alert_notice = due_alert_for_model()
        if alert_notice:
            # Merged onto whatever context this branch already carries (recalled memories,
            # standing preferences) rather than replacing it -- a capture failure and a
            # successful recall are unrelated events that can both be true on the same
            # prompt, and either one arriving first should not cost the other its context.
            reply = reply._replace(context=(
                f"{reply.context}\n\n{alert_notice}" if reply.context else alert_notice))
        write(HOST, reply)

    seen, carried = _read_state(session)
    if time.monotonic() - start < OVERALL_BUDGET_SEC:
        standing, standing_state = _standing_refresh(
            session, time.time(), event.cwd)
    else:
        # `("", None)` is exactly what `_standing_refresh` itself returns for "nothing to
        # do" -- see its own interval check -- so skipping the call is indistinguishable
        # from having made it and finding the refresh not due. `_write_state` reads `None`
        # as "carry the existing refresh clock forward," never as "refreshed just now,"
        # so a skip here cannot fool a later prompt into thinking this one already paid
        # for the check it did not make.
        log_line("recall", "skipped=standing refresh, budget exhausted")
        standing, standing_state = "", None
    anaphoric = _anaphoric(prompt)

    # An anaphoric prompt is searched together with the last substantive one, not instead
    # of it: "add that fix to #7" still carries "#7", and dropping it would trade one kind
    # of blindness for another. The carried text goes first because it is the topic.
    query = f"{carried} {prompt}".strip() if (anaphoric and carried) else prompt

    try:
        block, ok, why = fast_recall(query, k=K, budget=BUDGET, header=HEADER,
                                     min_score=_min_score())
    except Exception:
        # A retrieval failure must not become a failed prompt.
        block, ok, why = "", False, ""

    if ok is None:
        # Nothing configured. Still reported, because a hook that prints nothing is
        # indistinguishable from a hook that has stopped working -- which is the failure
        # this file exists to stop repeating -- but reported as what it is rather than as a
        # breakage someone would go looking for.
        _emit(Reply("recall", status=status("not configured")))
        return 0
    if not ok:
        # Four outcomes had four messages and a fifth was wearing the wrong one. A store
        # that is answering, and refusing on a stated allowance, is not a store that could
        # not be reached -- and the old line sent the reader to `capture.log`, which this
        # file has never written a byte to. The words differ because that is the rule this
        # whole file is built on.
        detail = _quota_line(why)
        log_line("recall", f"failed reason={why or 'unknown'}")
        _emit(Reply("recall", status=status(detail or "recall failed")))
        return 0

    here = event.cwd
    header, bullets = _split(block)
    bullets = [line for line in bullets if _belongs_here(line, here)]
    known = set(seen)
    fresh = [line for line in bullets if _digest(line) not in known]

    if len(fresh) < THIN:
        # The structured layer had little to say. Ask again for the raw turns too --
        # narrative excerpts cannot outrank claims that are not there.
        #
        # On the hosted endpoint this is currently a no-op: `include_episodes` is the only
        # boolean argument in the tool surface and the server's validator has no branch for
        # that type, so it raises and the client retries without it. It costs one round
        # trip on an already-thin prompt, and it starts working the day the server is
        # fixed, with no release here.
        if time.monotonic() - start < OVERALL_BUDGET_SEC:
            try:
                wider, wider_ok, _ = fast_recall(query, k=EPISODE_K, budget=EPISODE_BUDGET,
                                                 header=HEADER, include_episodes=True,
                                                 min_score=_min_score())
            except Exception:
                wider, wider_ok = "", False
            if wider_ok and wider:
                # Filtered on the same terms as the first pass. This is the branch that
                # most needs it: it runs precisely when `fresh` came back empty, and
                # dropping another checkout's notes is one of the things that empties it --
                # so without this the filter would make its own bypass fire more often.
                header, bullets = _split(wider)
                bullets = [line for line in bullets if _belongs_here(line, here)]
                fresh = [line for line in bullets if _digest(line) not in known]
        else:
            # `fresh` (and `bullets`) stay exactly what the primary pass returned -- an
            # already-thin answer, not a wrong one. The prompt still gets whatever that
            # pass found; it just does not get a second, wider round trip spent looking
            # for more of it.
            log_line("recall", "skipped=episode widen, budget exhausted")

    repeats = len(bullets) - len(fresh)

    # The topic only moves when the user says something with a topic in it. An anaphoric
    # turn leaves it pointing where it was, which is the whole point: three "yes" replies
    # in a row all search against the last thing that was actually about something.
    topic = carried if anaphoric else prompt

    if not fresh:
        _write_state(session, seen, topic, standing_state)
        note = status(f"{repeats} already in context" if repeats
                      else "no matching memories")
        if standing:
            # Nothing new to recall, and the standing set has moved: the turn still has to
            # carry it, or a rule written mid-session waits for the next prompt that
            # happens to match something.
            _emit(Reply("recall", status=status("standing preferences updated"),
                        context=standing))
            return 0
        _emit(Reply("recall", status=note))
        return 0

    _write_state(session, seen + [_digest(line) for line in fresh], topic, standing_state)

    # Deduplicated on the full line and injected clipped: the hash has to identify the
    # memory, not the excerpt, or raising MAX_INJECTED_CHARS would make everything already
    # in context look new.
    clipped = [_clip(line) for line in fresh]
    lines = [header] + clipped
    if any(short != full for short, full in zip(clipped, fresh)):
        lines.append(MORE)
    block_text = "\n".join(lines)

    if standing:
        block_text = f"{block_text}\n\n{standing}"

    label = status(f"{plural(len(fresh))} recalled")
    if repeats:
        label += f" · {repeats} already in context"
    log_line("recall", f"recalled={len(fresh)} repeats={repeats} injected={len(block_text)}c "
        f"clipped={sum(1 for s_, f_ in zip(clipped, fresh) if s_ != f_)}")
    _sample(prompt, fresh, anaphoric=anaphoric and bool(carried))
    _emit(Reply("recall", status=label, context=block_text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
