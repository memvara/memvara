"""Temporal accuracy: the six questions only a bitemporal store can answer.

Run:  PYTHONPATH=. python3 bench/temporal.py [--n 8] [--verbose]

**This is a synthetic, self-authored workload**, in the same category as
`bench/multihop.py` and `bench/compare.py`, and to be read the same way: an illustration
of a mechanism, not evidence of superiority over anything. It exists because the
differentiator had no number at all.

Everything else in `bench/` measures *retrieval* — LOCOMO, LongMemEval, 2WikiMultihop,
mem0, the multi-hop walk. Those are the commodity half, and they are benchmarked against
competitors. The half this library is actually built around — two independent clocks,
supersession that closes one of them, source authority — was measured by nothing. The
consequence was not hypothetical: two defects lived on the write path while 3,448 tests
passed, and both are the kind a temporal accuracy suite reports as a wrong answer.

## The six families

Each is a question a store without two clocks cannot answer, or a write-path property
that decides whether the answers stay true.

    point_in_time      valid_at=T. What was true then, as we understand it now.
    delayed_knowledge  known_at=T against valid_at=T on a late-recorded fact. "It was
                       true then" and "we had not heard it yet" are different answers,
                       and a store with one clock gives one answer to both.
    as_of_audit        both clocks moved together. What we believed at T, about T —
                       the reading an investigator gets, which is a different document
                       from the one you would read today.
    contradiction      a single-valued slot resolving, and a multi-valued one not
                       resolving, over the same corpus. Includes the same-instant case
                       an import produces.
    correction         `ended` against `retired`. The world changed, or the record was
                       wrong. Both stop answering the present tense and only one of them
                       stops answering about the past.
    source_authority   a low-confidence claim colliding with a high-confidence one. The
                       reliable value has to survive it.

## How it is scored

No model, no network, no reader, no judge. Two question kinds, both exact:

    set        the full live set under a stated pair of axes, compared to a gold set
               authored by the generator that built the scenario — never read back out
               of memvara.
    survives   one value that must still be in the live set. Used for source authority,
               where the property is "the reliable claim was not displaced" and scoring
               an exact set would bake in a choice about what happens to the guess.

Probed with `get_all(user=<scenario>, ...)` rather than `search()`. That is deliberate
and it is a scope limit worth stating: this measures the **temporal axes**, not the
ranker. Retrieval quality over the same corpus is what the other five harnesses are
for, and mixing the two would make a temporal regression indistinguishable from a
ranking one. Each scenario lives in its own `user` scope, so the probe is exact and
sibling scenarios cannot leak into each other — scope visibility widens upward only.

## The baseline column, and where it is honestly useless

`no-clocks` answers every question with the present-tense live set: what a store with no
valid time and no transaction time can do, asked the same questions. It is the headroom
figure, and the `disc` column says what share of a family's questions it gets wrong.

**A family with `disc` at 0.00 is one the baseline handles**, and the table says so
rather than hiding it. That is `contradiction`'s present-tense questions and all of
`source_authority`: both measure the *write* path, so a read-side baseline ties by
construction. For those two the comparator is this repository's own history, and
`docs/BENCHMARKS.md` quotes the before-and-after figures against a specific commit.

## The health check, which is not an accuracy question

`ended` promises something this suite cannot ask as a question: a claim whose valid time
was closed still answers `valid_at=<while it held>`. A claim closed at or before the
instant it began holds for no interval, so it answers nothing, at any instant, on either
clock — and the receipt for the write that did it used to look exactly like the receipt
for an ordinary supersession.

So the run reports two numbers beside the table: how many `ended` claims answer nothing,
and how many of those the write path reported as it made them. The first is a property
of the corpus (an import that stamps dates rather than timestamps produces them by the
handful, which is why the `contradiction` family includes that case on purpose). The
second is the one that should equal the first. It read 0 against 8 before
`WriteReceipt.collapsed` existed.

## How this benchmark could flatter us, and what stops it

Four constraints, because the corpus and the questions come from one generator and that
is exactly where a benchmark starts measuring itself.

1. **Gold sets are built from the generator's own model of the scenario**, before
   anything is written, and never from what memvara returns. A gold read back out of the
   system under test measures nothing but its own consistency.
2. **Every scenario carries a slot that does not move.** A store that answered every
   question with "the newest thing I have" would score on the moving slot and fail the
   stationary one, so the set questions cannot be passed by recency alone.
3. **The `disc` column is printed per family**, including where it is zero. A family
   whose baseline scores the same as the system is a family measuring the harness, and
   the number that says so is in the table rather than in this docstring.
4. **The empty set is a legal gold and appears in three families.** "Nothing was true
   then" and "we had not heard it yet" are answers, and a suite that only ever asked for
   non-empty sets would score a store that returns everything at nearly 100%.

What it does **not** control for: this is memvara measured against a baseline built in
this file, not against another memory layer. Nothing here is a head-to-head. The
scenarios are English-free — they are triples with instants — so none of the extraction
path is exercised and none of its cost or failure modes appear.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterable

sys.path.insert(0, "bench")

from memvara import HashingEmbedder, Memvara, NullLLM            # noqa: E402
from memvara.types import WriteReceipt                           # noqa: E402

# Six fixed instants, all in the past, so "now" sits after the whole timeline and a
# present-tense question is always a question about the end of the story. Constants
# rather than offsets from the clock: a bitemporal suite whose instants move with the
# wall clock cannot be reproduced, and `CONTRIBUTING.md` says time here is controlled by
# passing explicit values rather than by patching anything.
T0 = datetime(2023, 1, 1, tzinfo=timezone.utc)
T1 = datetime(2023, 6, 1, tzinfo=timezone.utc)
T2 = datetime(2024, 1, 1, tzinfo=timezone.utc)
T3 = datetime(2024, 6, 1, tzinfo=timezone.utc)
T4 = datetime(2025, 1, 1, tzinfo=timezone.utc)
T5 = datetime(2025, 6, 1, tzinfo=timezone.utc)

CITY = ["Berlin", "Lisbon", "Osaka", "Nairobi", "Lima", "Oslo", "Cairo", "Perth",
        "Tallinn", "Bogota", "Dakar", "Hanoi"]
FIRM = ["Acme", "Globex", "Initech", "Umbrella", "Soylent", "Hooli", "Vehement",
        "Cyberdyne", "Tyrell", "Weyland", "Massive", "Aperture"]
TOOL = ["ripgrep", "pytest", "mypy", "coverage", "ruff", "hypothesis", "tox", "nox",
        "pdb", "line_profiler", "black", "isort"]
# `speaks` is the declared multi-valued predicate in the built-in vocabulary, and the
# only one of the three used here that is. The first draft of the `contradiction` family
# used `prefers_tool` for the accumulating slot on the strength of the name, which is
# `Cardinality.ONE` — the harness reported 33.3% on that family and the gold was what was
# wrong. Left as a comment because it is the failure this file's constraint 1 is about:
# a gold authored from a belief about the system rather than from the system's schema.
LANG = ["Portuguese", "German", "Japanese", "Swahili", "Spanish", "Norwegian",
        "Arabic", "Tamil", "Estonian", "Quechua", "Wolof", "Vietnamese"]

#: What a question is asking of the store.
#:
#: `set` is the strict reading — the live set under these axes is exactly this. `survives`
#: is a membership test, used where the property being measured is "this claim was not
#: displaced" and an exact set would smuggle in a decision about what happens to the
#: claim that lost. Both are exact; neither involves a model or a threshold.
Kind = str


@dataclass(frozen=True, slots=True)
class Question:
    kind: Kind
    #: Keyword arguments for `get_all`, minus the scope. Empty means the present tense.
    axes: dict[str, datetime]
    #: `(predicate, object)` pairs. For `set`, the whole answer; for `survives`, the one
    #: pair that must be in it.
    gold: frozenset[tuple[str, str]]
    #: One line, printed when it fails, saying what the store got wrong. Written from
    #: the scenario rather than from the code, so a failure reads as a claim about
    #: behaviour and not as a diff of two sets.
    says: str


@dataclass(slots=True)
class Scenario:
    key: str
    family: str
    #: `(subject, predicate, object, kwargs)`, applied in order through `remember()`.
    writes: list[tuple[str, str, str, dict]] = field(default_factory=list)
    questions: list[Question] = field(default_factory=list)

    def write(self, subject: str, predicate: str, obj: str, **kw: object) -> None:
        self.writes.append((subject, predicate, obj, dict(kw)))

    def ask(self, kind: Kind, gold: Iterable[tuple[str, str]], says: str,
            **axes: datetime) -> None:
        self.questions.append(Question(kind, dict(axes), frozenset(gold), says))


# --- the six families ----------------------------------------------------------------
#
# Each builder takes an index and returns one scenario. The index only varies the
# vocabulary; the *shape* is fixed per family, because the shape is the thing being
# measured and rotating it would make a family's score an average over several different
# questions wearing one name.


def point_in_time(i: int) -> Scenario:
    """One slot that moves, one that does not, and three instants across the move.

    The stationary slot is constraint 2: without it, "answer with the newest value you
    have" scores 100% on this family.
    """
    s = Scenario(f"pit{i}", "point_in_time")
    a, b, firm = CITY[i % len(CITY)], CITY[(i + 5) % len(CITY)], FIRM[i % len(FIRM)]
    s.write("user", "lives_in", a, valid_from=T1, recorded_at=T1)
    s.write("user", "works_at", firm, valid_from=T1, recorded_at=T1)
    s.write("user", "lives_in", b, valid_from=T3, recorded_at=T3)

    s.ask("set", [], "nothing had been asserted yet at T0", valid_at=T0)
    s.ask("set", [("lives_in", a), ("works_at", firm)],
          f"at T2 the city was {a}, and the employer had not changed", valid_at=T2)
    s.ask("set", [("lives_in", b), ("works_at", firm)],
          f"at T4 the city was {b}", valid_at=T4)
    return s


def delayed_knowledge(i: int) -> Scenario:
    """A fact true from T1 and not recorded until T3. The two clocks disagree on purpose.

    This is the pair the whole model exists for, and a store with one clock answers both
    questions the same way. Scored as two questions rather than one so the table shows
    *which* of the two a regression broke.
    """
    s = Scenario(f"dk{i}", "delayed_knowledge")
    firm, tool = FIRM[i % len(FIRM)], TOOL[i % len(TOOL)]
    s.write("user", "works_at", firm, valid_from=T1, recorded_at=T3)
    s.write("user", "prefers_tool", tool, valid_from=T1, recorded_at=T1)

    s.ask("set", [("works_at", firm), ("prefers_tool", tool)],
          f"{firm} was true at T2 whether or not anyone had written it down",
          valid_at=T2)
    s.ask("set", [("prefers_tool", tool)],
          f"at T2 nobody had recorded {firm} yet, so the record does not contain it",
          known_at=T2)
    return s


def as_of_audit(i: int) -> Scenario:
    """Both clocks moved together, over a change learned two instants after it happened.

    `as_of` is not a third axis. It is `valid_at == known_at`, and the three questions
    here are the same story read three ways — which is the whole argument for keeping the
    axes separable rather than collapsing them into one "time" parameter.
    """
    s = Scenario(f"aoa{i}", "as_of_audit")
    a, b = CITY[i % len(CITY)], CITY[(i + 7) % len(CITY)]
    s.write("user", "lives_in", a, valid_from=T1, recorded_at=T1)
    s.write("user", "lives_in", b, valid_from=T3, recorded_at=T5)

    s.ask("set", [("lives_in", a)],
          f"at T2 we had heard {a} and {a} was true", as_of=T2)
    s.ask("set", [("lives_in", b)],
          f"we now know {b} was true at T4", valid_at=T4)
    s.ask("set", [],
          f"reading the record as it stood at T4: {a} had already stopped being true "
          f"and {b} had not been written yet, so the audit is empty", as_of=T4)
    return s


def contradiction(i: int) -> Scenario:
    """A single-valued slot resolving, a multi-valued one not, and the same-instant case.

    `lives_in` is `Cardinality.ONE` and `speaks` is `MANY`, so one corpus exercises both
    readings of "two values in a slot" and a store that got cardinality backwards fails in
    both directions at once rather than in neither.

    The third write to `lives_in` shares the second one's `valid_from`. That is not a
    contrivance — it is every import that stamps dates rather than timestamps, and every
    correction made in the same second — and it is the case that produces a claim true at
    no instant. The gold says so: at T3 exactly one value answers, and the value it
    displaced answers at no instant at all. The health check below counts what that costs.
    """
    s = Scenario(f"con{i}", "contradiction")
    a, b, c = CITY[i % len(CITY)], CITY[(i + 3) % len(CITY)], CITY[(i + 9) % len(CITY)]
    x, y = LANG[i % len(LANG)], LANG[(i + 4) % len(LANG)]
    s.write("user", "lives_in", a, valid_from=T1, recorded_at=T1)
    s.write("user", "speaks", x, valid_from=T1, recorded_at=T1)
    s.write("user", "lives_in", b, valid_from=T3, recorded_at=T3)
    s.write("user", "speaks", y, valid_from=T3, recorded_at=T3)
    s.write("user", "lives_in", c, valid_from=T3, recorded_at=T4)

    s.ask("set", [("lives_in", a), ("speaks", x)],
          "before the change, one city and one language", valid_at=T2)
    s.ask("set", [("lives_in", c), ("speaks", x), ("speaks", y)],
          f"a single-valued slot keeps {c} alone; a multi-valued one keeps both languages",
          valid_at=T4)
    s.ask("set", [("lives_in", c), ("speaks", x), ("speaks", y)],
          "and the same holds now")
    return s


def correction(i: int) -> Scenario:
    """`ended` and `retired` over identical writes. One of them still answers about T2.

    Two scenarios' worth of story in one scope, on two slots, so the difference is a
    comparison inside a single answer rather than across two runs: `lives_in` is
    corrected (`close="retired"`, the record was wrong) and `works_at` is superseded
    (`close="ended"`, the world moved). At T2 the employer answers and the city does not,
    and a store that treated the two words as synonyms gets exactly one of them wrong
    whichever synonym it picked.
    """
    s = Scenario(f"cor{i}", "correction")
    a, b = CITY[i % len(CITY)], CITY[(i + 2) % len(CITY)]
    p, q = FIRM[i % len(FIRM)], FIRM[(i + 6) % len(FIRM)]
    s.write("user", "lives_in", a, valid_from=T1, recorded_at=T1)
    s.write("user", "works_at", p, valid_from=T1, recorded_at=T1)
    s.write("user", "lives_in", b, valid_from=T3, recorded_at=T3, close="retired")
    s.write("user", "works_at", q, valid_from=T3, recorded_at=T3)

    s.ask("set", [("works_at", p)],
          f"{a} was never true, so it answers nothing at T2; {p} was true and does",
          valid_at=T2)
    s.ask("set", [("lives_in", b), ("works_at", q)],
          "and both current values answer now")
    return s


def source_authority(i: int) -> Scenario:
    """A 0.10 guess arriving on top of a 1.00 statement, on a single-valued slot.

    Scored with `survives` rather than an exact set, and the reason is worth stating: the
    property brief §7 asks for is that the reliable claim is not displaced by an
    unreliable one. What happens to the guess — stored beside it, refused, ranked below —
    is a separate design decision, and an exact-set gold would quietly make this family a
    test of that decision instead.

    The second slot receives the mirror image: a *confident* correction over an earlier
    confident value, which must supersede normally. Without it this family would reward a
    store that simply never superseded anything.
    """
    s = Scenario(f"sa{i}", "source_authority")
    a, b = CITY[i % len(CITY)], CITY[(i + 4) % len(CITY)]
    p, q = FIRM[i % len(FIRM)], FIRM[(i + 8) % len(FIRM)]
    s.write("user", "lives_in", a, valid_from=T1, recorded_at=T1,
            confidence=1.0, extractor="api")
    s.write("user", "works_at", p, valid_from=T1, recorded_at=T1,
            confidence=1.0, extractor="api")
    s.write("user", "lives_in", b, valid_from=T3, recorded_at=T3,
            confidence=0.1, extractor="llm-guess")
    s.write("user", "works_at", q, valid_from=T3, recorded_at=T3,
            confidence=0.95, extractor="fast/v1")

    s.ask("survives", [("lives_in", a)],
          f"a 0.10 guess must not displace {a}, asserted at 1.00")
    s.ask("survives", [("works_at", q)],
          f"and a 0.95 claim must still displace {p}: authority is not a veto on change")
    return s


FAMILIES: list[Callable[[int], Scenario]] = [
    point_in_time, delayed_knowledge, as_of_audit,
    contradiction, correction, source_authority,
]


# --- running it -----------------------------------------------------------------------


def build(n: int) -> tuple[Memvara, list[Scenario], list[WriteReceipt]]:
    """One store, `n` scenarios per family, each in its own `user` scope.

    `HashingEmbedder` and `NullLLM` because nothing here is retrieved or extracted — the
    probe is `get_all`, which reads no vector at all. Pinned anyway rather than left to
    `default_embedder()`, for the reason `tests/conftest.py` gives: an unpinned embedder
    makes a run's premises a property of what happens to be installed.
    """
    mem = Memvara(llm=NullLLM(), embedder=HashingEmbedder(dim=512), tenant="bench")
    scenarios = [family(i) for family in FAMILIES for i in range(n)]
    receipts = []
    for s in scenarios:
        for subject, predicate, obj, kw in s.writes:
            receipts.append(mem.remember(subject, predicate, obj, user=s.key, **kw))
    return mem, scenarios, receipts


def observe(mem: Memvara, s: Scenario, axes: dict[str, datetime]
            ) -> frozenset[tuple[str, str]]:
    """The store's answer, as `(predicate, object)` pairs."""
    return frozenset((c.predicate, c.object)
                     for c in mem.get_all(user=s.key, **axes))


def correct(question: Question, answer: frozenset[tuple[str, str]]) -> bool:
    """Whether the store's answer satisfies the question. Exact either way.

    The unknown kind raises rather than falling through to the set comparison, and that
    is the whole reason this is not a two-line expression. A mistyped kind — `survive`
    for `survives` — would otherwise be scored by the wrong rule with nothing said: with
    a gold of one pair it reports False where the right rule reports True, and against a
    live set that happens to hold exactly that pair it reports True *for the wrong
    reason*. A scoring function that silently applies the wrong rule is the failure this
    file's docstring calls the one to distrust most, arrived at from the inside. `bench/`
    is outside `mypy -p memvara`, so the annotation cannot catch it and this has to.
    """
    if question.kind == "survives":
        return question.gold <= answer
    if question.kind == "set":
        return answer == question.gold
    raise ValueError(
        f"unknown question kind {question.kind!r}; expected 'set' or 'survives'")


def score(mem: Memvara, scenarios: list[Scenario], verbose: bool) -> None:
    """The table. One row per family, plus the headroom the baseline leaves."""
    rows: dict[str, list[int]] = {}
    for s in scenarios:
        got = rows.setdefault(s.family, [0, 0, 0])       # asked, right, baseline-wrong
        # The baseline is the same probe with the axes dropped: what a store holding one
        # clock can say, asked the same question.
        present = observe(mem, s, {})
        for q in s.questions:
            answer = observe(mem, s, q.axes)
            ok = correct(q, answer)
            got[0] += 1
            got[1] += int(ok)
            got[2] += int(not correct(q, present))
            if verbose and not ok:
                print(f"    {s.key} {q.kind} {q.axes or 'now'}: {q.says}")
                print(f"      gold {sorted(q.gold)}")
                print(f"      got  {sorted(answer)}")

    print(f"  {'family':<20} {'n':>4} {'memvara':>9} {'no-clocks':>11} {'disc':>7}")
    total = [0, 0, 0]
    for family in (f.__name__ for f in FAMILIES):
        asked, right, discriminating = rows[family]
        total = [t + v for t, v in zip(total, (asked, right, discriminating))]
        print(f"  {family:<20} {asked:>4} {right / asked:>8.1%} "
              f"{(asked - discriminating) / asked:>10.1%} "
              f"{discriminating / asked:>6.1%}")
    print(f"  {'all':<20} {total[0]:>4} {total[1] / total[0]:>8.1%} "
          f"{(total[0] - total[2]) / total[0]:>10.1%} {total[2] / total[0]:>6.1%}")


def health(mem: Memvara, scenarios: list[Scenario],
           receipts: list[WriteReceipt]) -> None:
    """Ended claims that answer nothing, and how many the write path said so about.

    Not an accuracy question, because there is no question to ask: a claim whose valid
    interval is empty is absent from every answer at every instant, so a suite made of
    questions cannot see it. It is a promise the store makes about the word `ended` —
    `core.py`'s "`get_all(valid_at=<while it held>)` still answers" — checked against the
    rows instead.

    Read through `history()`, one slot at a time, because that is the read that returns
    versions rather than answers. `get_all(states=["ended"])` is the wrong instrument: it
    still filters on liveness at the probed instant, and the rows being counted here are
    the ones live at no instant.

    The first number is a property of the corpus and is *meant* to be non-zero — the
    `contradiction` family writes the same-instant case on purpose, because that is what
    an import stamping dates rather than timestamps produces. The second is the one that
    should equal it.
    """
    ended, empty = 0, 0
    for s in scenarios:
        slots = {(subject, predicate) for subject, predicate, _o, _kw in s.writes}
        for subject, predicate in sorted(slots):
            for claim in mem.history(subject, predicate, user=s.key):
                if claim.state != "ended":
                    continue
                ended += 1
                empty += int(claim.valid_to is not None
                             and claim.valid_to <= claim.valid_from)
    reported = sum(len(r.collapsed) for r in receipts)
    share = empty / ended if ended else 0.0
    print(f"\n  ended claims                                  {ended:>4}")
    print(f"  ...that answer at no instant                   {empty:>4}  ({share:.1%})")
    print(f"  ...of which the write path reported            {reported:>4}")
    if reported != empty:
        print("  ^ these must match. A claim closed at its own start is legal and is "
              "returned by\n    no query on either clock, so a write that makes one and "
              "says nothing reports\n    `ended 1` — byte-identical to an ordinary "
              "supersession that kept its interval.")


def main(argv: list[str]) -> None:
    n = int(argv[argv.index("--n") + 1]) if "--n" in argv else 8
    verbose = "--verbose" in argv

    mem, scenarios, receipts = build(n)
    print(f"\n=== temporal accuracy ({len(scenarios)} scenarios, "
          f"{sum(len(s.writes) for s in scenarios)} writes, no model) ===\n")
    score(mem, scenarios, verbose)
    health(mem, scenarios, receipts)
    print("\n  no-clocks answers every question with the present-tense live set. `disc`"
          "\n  is the share of a family's questions it gets wrong; a family at 0.0% is"
          "\n  one a store with one clock handles, and both of those measure the write"
          "\n  path rather than the read axes. See this file's docstring.")
    mem.close()


if __name__ == "__main__":
    main(sys.argv[1:])
