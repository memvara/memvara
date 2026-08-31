"""Where does Alice live? Ask it three times, at three instants, get three answers.

    python3 examples/temporal_memory.py

One person moves twice. A store that keeps one value per fact can answer the first
question and gets the other two wrong — not by hallucinating, but because overwriting
Berlin with London destroyed the only record that Berlin was ever the answer.

Memvara stores an interval instead of a value, so all three questions are lookups:

    now              -> New York
    on 20 March      -> London
    on 20 January    -> Berlin

Nothing here calls a model. `remember()` takes the triple already parsed, and the
supersession that closes Berlin's interval is an indexed lookup on
(subject, predicate) — `lives_in` is declared single-valued in the built-in schema, so
a second value for one subject is a contradiction by definition rather than by
similarity.

Run it and the printed output matches `examples/README.md` line for line;
`tests/test_examples.py` asserts that, so this file cannot drift from what it claims.
"""

from datetime import datetime, timezone

from memvara import Memvara, NullLLM

UTC = timezone.utc


def at(month: int, day: int) -> datetime:
    """A 2026 instant, in UTC. Every date in this example is fixed so the run repeats."""
    return datetime(2026, month, day, tzinfo=UTC)


def main() -> None:
    # `llm=NullLLM()` asks for the offline configuration explicitly. Without it the
    # constructor warns that arbitrary prose will not be extracted, which is true and
    # irrelevant here: every fact below is written as a triple.
    mem = Memvara(user="alice", llm=NullLLM())

    # --- three statements, made on three different days ----------------------------
    #
    # `valid_from` is when the fact became true in the world. `recorded_at` is when this
    # store was told. They are set together here because Alice told us on the day she
    # moved; a fact that arrives late about the past is the case where they differ, and
    # that is what `known_at=` reads.
    mem.remember("Alice", "lives_in", "Berlin",
                 valid_from=at(1, 10), recorded_at=at(1, 10))
    mem.remember("Alice", "lives_in", "London",
                 valid_from=at(3, 15), recorded_at=at(3, 15))
    mem.remember("Alice", "lives_in", "New York",
                 valid_from=at(6, 2), recorded_at=at(6, 2))

    print("Where does Alice live now?")
    print(" ", [c.object for c in mem.get_all()][0])

    print("\nWhere did Alice live on 20 March 2026?")
    print(" ", [c.object for c in mem.get_all(as_of=at(3, 20))][0])

    print("\nWhere did Alice live on 20 January 2026?")
    print(" ", [c.object for c in mem.get_all(as_of=at(1, 20))][0])

    # --- the whole timeline, which is what makes the three answers possible ---------
    #
    # Berlin and London are `ended`: they were true, and then the world changed. Nothing
    # was deleted, and `valid_to` is the instant the next value took over.
    print("\nThe timeline this store holds for (Alice, lives_in):")
    for claim in mem.history("Alice", "lives_in"):
        until = claim.valid_to.date().isoformat() if claim.valid_to else "now"
        print(f"  {claim.object:<9} {claim.valid_from.date().isoformat()} -> "
              f"{until:<10}  [{claim.state}]")

    # --- and the same question narrated, which is the sentence an agent can say -----
    #
    # `ask()` composes what is true now, what we believe today was true then, and what
    # this store would have answered then. No model is consulted: every sentence is
    # rendered from a stored column.
    print("\nask(\"where does Alice live?\", at=20 March 2026):")
    for line in mem.ask("where does Alice live?", at=at(3, 20)).text.splitlines():
        print(f"  {line}" if line else "")

    mem.close()


if __name__ == "__main__":
    main()
