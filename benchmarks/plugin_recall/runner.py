"""Running a corpus against a plugin, and scoring what came back.

One session for the whole run, preceded by a warmup prompt that is never scored. That
ordering is the measurement, not an implementation detail:

A plugin may inject a once-per-session preamble -- a standing-preferences block, a store
summary, an instruction sheet. Charging that to whichever case happened to run first would
make the first case's score an artefact of corpus order, and running every case in its own
session would charge it to *every* case and report a plugin's per-prompt cost as several
times what a real session pays. The warmup absorbs it, the scored cases then measure the
marginal cost of one more prompt, and the preamble is reported on its own line where it can
be read for what it is: a fixed cost paid once, whether or not it is relevant to anything.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from .cases import Case
from .plugin import Plugin, Reply, invoke

#: Deliberately substantive enough to look like a real turn, and deliberately about
#: nothing the corpus asks about, so it cannot seed an answer a later case is scored on.
WARMUP = "hello, I'm about to start work"


@dataclass(frozen=True)
class Outcome:
    case: Case
    reply: Reply

    @property
    def correct(self) -> bool:
        if self.case.kind == "silence":
            return not self.reply.spoke
        return self.reply.spoke and self.case.matched(self.reply.context)

    @property
    def note(self) -> str:
        """Why this outcome is what it is, in a few words, for the per-case listing."""
        if self.reply.error:
            return f"hook error: {self.reply.error}"
        if self.case.kind == "silence":
            return "stayed quiet" if self.correct else f"injected {self.reply.tokens} tok"
        if not self.reply.spoke:
            return "said nothing"
        return "found it" if self.correct else "injected, but not the fact"


@dataclass(frozen=True)
class Result:
    plugin: Plugin
    outcomes: tuple[Outcome, ...]
    preamble: Reply
    cwd: Path
    env: tuple[str, ...] = ()
    shared_session: bool = False

    def of_kind(self, kind: str) -> tuple[Outcome, ...]:
        return tuple(o for o in self.outcomes if o.case.kind == kind)

    def rate(self, kind: str) -> float | None:
        """Share correct, or `None` when the corpus contained none of that kind.

        `None` rather than `0.0`, everywhere, for the reason `cases` gives at length: a
        plugin that was never asked a question has not failed it, and a benchmark that
        prints 0% for "no cases loaded" publishes an accusation it did not test.
        """
        subset = self.of_kind(kind)
        if not subset:
            return None
        if kind == "silence" and not self.validated:
            return None
        return sum(1 for o in subset if o.correct) / len(subset)

    @property
    def validated(self) -> bool:
        """Did the plugin put text in the context even once during this run?

        A silence-only corpus has a fatal degenerate case: a plugin whose hook is broken
        injects nothing on every prompt and scores 100%. That is not a hypothetical. The
        first live run of this harness graded a stale development build that answered
        `recall failed` to all 22 cases, and the report said 100% silence, 0 tokens -- a
        perfect score, produced by software that did not work at all.

        Nothing in the host's protocol distinguishes "I have nothing relevant" from "I am
        broken": both are an absent `additionalContext`. So the harness does not try to
        read the difference out of one reply. It asks a weaker question it can actually
        answer -- did this plugin ever speak? -- and refuses to report a silence score when
        the answer is no, because at that point the run has not shown the plugin was
        capable of failing the test.
        """
        return self.preamble.spoke or any(o.reply.spoke for o in self.outcomes)

    @property
    def balanced(self) -> float | None:
        """The mean of the two rates -- the only combined number this harness reports.

        Unavailable unless both populations were measured. A single figure computed from
        hits alone is exactly the number an always-inject plugin wins, and printing it
        without the silence half would make this harness reward the defect it was built
        to find.
        """
        hit, silence = self.rate("hit"), self.rate("silence")
        if hit is None or silence is None:
            return None
        return (hit + silence) / 2


def run(plugin: Plugin, cases: list[Case], *, cwd: Path,
        session_id: str | None = None, shared_session: bool = False,
        extra_env: "dict[str, str] | None" = None) -> Result:
    """Score every case, by default each in its own session.

    ## Why per-case sessions are the default, and what it cost to learn

    A plugin may suppress a memory it has already injected this session -- the text is
    still in the conversation, so sending it again buys nothing and costs tokens. That is
    correct behaviour and every serious plugin here does it.

    Run a corpus through one shared session and that correct behaviour destroys the score.
    The first few prompts pull most of the store into context; every later question about
    an already-injected fact is then answered with a deliberate silence, and a harness that
    calls silence a miss reports the plugin as having failed to retrieve facts it retrieved
    perfectly well minutes earlier. Measured here: 6.7% on a corpus where single prompts,
    run cold, returned the right fact every time.

    So each case gets a fresh session and is judged on its own: *asked this question with
    nothing else in context, does the plugin surface the fact?* The cost of that choice is
    that a once-per-session preamble is paid on every case, which is why `preamble` is
    measured separately and reported on its own line rather than folded into the per-case
    token figures.

    `shared_session=True` restores one session for the whole corpus. It answers a different
    and narrower question -- what does one more prompt cost in a session already underway --
    and its hit rate is not comparable with the default.
    """
    base_session = session_id or f"bench-{uuid.uuid4()}"
    preamble = invoke(plugin, WARMUP, session_id=base_session, cwd=cwd, extra_env=extra_env)
    outcomes = tuple(
        Outcome(case, invoke(
            plugin, case.prompt, cwd=cwd, extra_env=extra_env,
            session_id=base_session if shared_session else f"{base_session}-{index}"))
        for index, case in enumerate(cases)
    )
    # Only the names are kept. A value here can be a path, a URL or a token, and a report
    # is something people paste into issues.
    return Result(plugin, outcomes, preamble, cwd, tuple(sorted(extra_env or {})),
                  shared_session)
