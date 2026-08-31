"""The 90-second terminal demo: an agent that remembers the wrong thing, and the fix.

    python3 examples/temporal_memory_demo/demo.py            # paced, for recording
    python3 examples/temporal_memory_demo/demo.py --fast     # no pauses, for CI

Six beats, timed for a screen recording. `README.md` next to this file has the recording
procedure and the expected transcript.

    0-10s   the problem: an agent does not just forget, it remembers the wrong thing
    10-30s  three statements about one person, made on three different days
    30-50s  what is true now
    50-65s  what was true on a day in the past
    65-80s  the record: every value with its interval, and what the live one replaced
    80-90s  the close

`--fast` removes every pause and nothing else, so the transcript it prints is
byte-identical to the paced one. That is what makes it testable:
`tests/test_examples.py` runs it and asserts on the lines below.

Everything here is real. The store is an in-memory `Memvara` with no model, no network and
no key; every printed value is read back out of it rather than typed into a string.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone

from memvara import Memvara, NullLLM

UTC = timezone.utc

#: Where each beat ends, in seconds from the first line. These are the numbers in
#: `README.md`'s beat table and in this module's docstring, and they are what makes the
#: demo 90 seconds rather than however long its individual pauses happen to add up to:
#: every beat finishes by holding until its boundary. Change one here and the whole
#: schedule stays consistent; change a pause inside a beat and only that beat's internal
#: rhythm moves.
BEATS = {"problem": 10.0, "write": 30.0, "now": 50.0, "then": 65.0,
         "history": 80.0, "close": 90.0}

#: The default hold after one printed line. Small on purpose: the beat boundaries above
#: absorb whatever is left over, so a line's hold controls rhythm rather than length.
LINE = 0.35


class Pacer:
    """Prints with pauses, or without them, against one clock started at the first line.

    One object rather than a module-level flag so the demo has no global state to reset
    between the paced run and the test's fast one. `--fast` sets every wait to zero and
    changes nothing else, so the two runs print identical text.
    """

    def __init__(self, *, fast: bool) -> None:
        self.fast = fast
        self.started = time.monotonic()

    def say(self, text: str = "", *, hold: float = LINE) -> None:
        print(text, flush=True)
        self.pause(hold)

    def pause(self, seconds: float) -> None:
        if not self.fast:
            time.sleep(seconds)

    def until(self, beat: str) -> None:
        """Hold until `beat`'s boundary, or return at once if it has already passed.

        Overrunning is the case worth thinking about: a slower machine, or a beat that
        grew a line. It ends the beat late rather than truncating it, which keeps the
        content intact and lets the schedule drift — the right trade for a demo, where a
        cut sentence is worse than five seconds of slippage.
        """
        if self.fast:
            return
        remaining = BEATS[beat] - (time.monotonic() - self.started)
        if remaining > 0:
            time.sleep(remaining)


def at(month: int, day: int) -> datetime:
    """A fixed 2026 instant, in UTC — so a re-recording produces the same screen."""
    return datetime(2026, month, day, tzinfo=UTC)


def rule(p: Pacer) -> None:
    p.say("─" * 62, hold=0.15)


def beat_problem(p: Pacer) -> None:
    """0-10s. State the failure in the customer's words, not ours."""
    p.say()
    p.say("  AI agents don't just forget.")
    p.pause(0.9)
    p.say("  They remember the wrong thing.")
    p.pause(1.6)
    p.say()
    p.say("  You corrected it in March. In August it is still saying the old value,")
    p.say("  because the store kept both and marked neither one current.")
    p.until("problem")


def beat_write(p: Pacer, mem: Memvara) -> None:
    """10-30s. Three statements about one person, on three days."""
    p.say()
    rule(p)
    p.say("  Alice tells us where she lives. Three times, over five months.")
    rule(p)
    p.say()
    for month, day, city in ((1, 10, "Berlin"), (3, 15, "London"), (6, 2, "New York")):
        when = at(month, day)
        mem.remember("Alice", "lives_in", city, valid_from=when, recorded_at=when)
        p.say(f'  {when.date().isoformat()}   mem.remember("Alice", "lives_in", "{city}")',
              hold=0.7)
    p.pause(1.6)
    p.say()
    p.say("  No model was called. `lives_in` is declared single-valued, so the second")
    p.say("  value closed the first one's interval by lookup, not by similarity.")
    p.until("write")


def beat_now(p: Pacer, mem: Memvara) -> None:
    """30-50s. The question every memory layer can answer."""
    p.say()
    rule(p)
    p.say("  Where does Alice live now?")
    rule(p)
    p.say()
    p.say("  >>> [c.object for c in mem.get_all()]", hold=0.9)
    p.say(f"  {[c.object for c in mem.get_all()]}")
    p.until("now")


def beat_then(p: Pacer, mem: Memvara) -> None:
    """50-65s. The question that separates a memory layer from a cache."""
    p.say()
    rule(p)
    p.say("  Where did she live on 20 March? And on 20 January?")
    rule(p)
    p.say()
    for month, day in ((3, 20), (1, 20)):
        when = at(month, day)
        p.say(f"  >>> [c.object for c in mem.get_all(as_of=datetime({when.year}, "
              f"{when.month}, {when.day}, tzinfo=UTC))]", hold=0.8)
        p.say(f"  {[c.object for c in mem.get_all(as_of=when)]}")
        p.pause(0.8)
    p.say()
    p.say("  Berlin was not overwritten. It was given an end date.")
    p.until("then")


def beat_history(p: Pacer, mem: Memvara) -> None:
    """65-80s. The record itself: every value, its interval, and what the live one replaced."""
    p.say()
    rule(p)
    p.say("  The whole record, which is what makes those answers lookups.")
    rule(p)
    p.say()
    p.say("  >>> mem.history(\"Alice\", \"lives_in\")", hold=0.8)
    timeline = mem.history("Alice", "lives_in")
    for claim in timeline:
        until = claim.valid_to.date().isoformat() if claim.valid_to else "now"
        p.say(f"  {claim.object:<9} {claim.valid_from.date().isoformat()} -> "
              f"{until:<10}  [{claim.state}]", hold=0.55)
    p.pause(0.7)
    p.say()
    p.say('  "ended" means the world changed. A value that was never true would read')
    p.say('  "retired" instead, and nothing is deleted either way.')
    p.pause(0.9)

    # And the provenance of the live one, which is the other half of the record: what it
    # replaced, and who wrote it. The supersession chain is stored, so this costs a
    # lookup — `why()` also returns the source turns when the write cited any.
    live = [c for c in timeline if c.state == "live"][0]
    provenance = mem.why(live.id)
    assert provenance is not None
    p.say()
    p.say("  >>> mem.why(claim_id)", hold=0.8)
    p.say(f"  it replaced:  {provenance.superseded[0].object} "
          f"({provenance.superseded[0].state})")
    p.say(f"  written by:   {provenance.extractor} ({provenance.derivation.value})")
    p.until("history")


def beat_close(p: Pacer) -> None:
    """80-90s. The name, the claim, and where to go."""
    p.say()
    rule(p)
    p.say()
    p.say("  Memvara — bitemporal memory for AI agents.")
    p.say()
    p.say("  Know what was true. Know when it was true. Know why you believe it.")
    p.say()
    p.say("  github.com/memvara/memvara          pip install memvara")
    p.say("  memvara.dev")
    p.say()
    rule(p)
    p.until("close")
    p.say()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fast", action="store_true",
                        help="remove every pause; the transcript is unchanged")
    args = parser.parse_args(argv)

    # The rules are box-drawing characters and the closing line has an em dash, and on
    # Windows `sys.stdout` defaults to the ANSI code page — cp1252, which has neither.
    # Without this the demo dies on its first rule with a UnicodeEncodeError, four lines
    # in, on the one platform where nobody writing it would notice.
    #
    # Inside `main()` rather than at module scope on purpose: importing this module under
    # a test runner that has replaced `sys.stdout` with a capture object would otherwise
    # reach for a `reconfigure` that object does not have.
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    p = Pacer(fast=args.fast)
    with Memvara(user="alice", llm=NullLLM()) as mem:
        beat_problem(p)
        beat_write(p, mem)
        beat_now(p, mem)
        beat_then(p, mem)
        beat_history(p, mem)
        beat_close(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
