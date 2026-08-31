"""An engineering decision, two weeks later: what changed, when, why, and on what evidence.

    python3 examples/coding_agent.py

A coding agent that keeps a decision as an embedded sentence can retrieve it. What it
cannot do is tell you that the decision *replaced* an earlier one, name the day it did,
or show you the message it came from. Those three are the whole difference between a
note and a record.

This example writes the transcript turns as episodes, writes the decision as a triple
citing them, and then answers the four questions a developer actually asks two weeks
later:

    What is the auth strategy now?     -> get_all()
    What was it before?                -> history()
    When did we change it?             -> the ended claim's valid_to
    Why, and on what evidence?         -> why(), which returns the source turns

## The one piece of setup, and why it is not optional

`auth_strategy` is not in the built-in vocabulary, which is a personal-assistant one —
where somebody lives, where they work. An undeclared predicate takes the safe default
twice over: multi-valued, so nothing supersedes, and slow-decaying, so this morning's
deploy still ranks as fresh in two years. Declaring it single-valued is what makes the
OAuth write *end* the API-keys claim instead of accumulating beside it.

`decided` is *multi-valued*, and that is the other half of the lesson: a project makes
many decisions, and a later one does not make an earlier one untrue. It stopped being
*current*, which is what the `auth_strategy` slot above records. A `ONE` cardinality here
would have every new decision silently end the last.

Both are declared inline below rather than loaded from the shipped `decisions` pack,
which is the same declaration in TOML. Packs are read with `tomllib` and so need Python
3.11, and this file runs on every interpreter the package supports — see `build()`.
"""

from datetime import datetime, timezone

from memvara import (Cardinality, MemoryType, Memvara, NullLLM, PredicateRegistry,
                     PredicateSpec, Volatility)
from memvara.schema import BUILTIN_PREDICATES

UTC = timezone.utc

#: The team's own vocabulary, on top of the builtins. One slot per service,
#: single-valued: a service authenticates one way at a time, so a second value is a
#: contradiction rather than an addition.
AUTH_STRATEGY = PredicateSpec(
    name="auth_strategy",
    cardinality=Cardinality.ONE,
    volatility=Volatility.SLOW,
    aliases=("authenticates_with", "auth_via"),
)

#: `decided`, copied field for field from the shipped `decisions` pack
#: (`memvara/packs/decisions.toml`). MANY so decisions accumulate; STATIC because a
#: decision is a commitment and stays true as a record of itself; EPISODIC because it
#: records something that happened at an instant.
DECIDED = PredicateSpec(
    name="decided",
    cardinality=Cardinality.MANY,
    volatility=Volatility.STATIC,
    memory_type=MemoryType.EPISODIC,
    aliases=("decision", "chose", "settled_on", "agreed_on"),
)


def at(month: int, day: int) -> datetime:
    """A 2026 instant, in UTC. Fixed dates, so the run repeats exactly."""
    return datetime(2026, month, day, tzinfo=UTC)


def build() -> Memvara:
    """A store that knows the engineering vocabulary before the first write.

    The two specs above are declared in Python rather than loaded from the shipped pack,
    and the reason is worth one sentence because it is a real constraint rather than a
    style choice. The equivalent is:

        from memvara.schema import load_all_specs
        BUILTIN_PREDICATES + load_all_specs("engineering,decisions")

    ...which reads `memvara/packs/*.toml` and is the same string `MEMVARA_PREDICATES`
    hands the MCP server. It parses TOML with `tomllib`, which arrived in Python 3.11 —
    so on 3.10, which this package supports, it raises. An example that a supported
    interpreter cannot run is worse than one that spells out four extra lines.
    """
    registry = PredicateRegistry(BUILTIN_PREDICATES + (AUTH_STRATEGY, DECIDED))
    return Memvara(user="platform-team", llm=NullLLM(), registry=registry)


def main() -> None:
    mem = build()

    # --- 3 February: the state of the world, and the turn that said so --------------
    #
    # `add(role="system")` stores the turn and cites it, and extracts nothing: this is a
    # transcript being filed, not somebody speaking. `remember()` then writes the fact
    # itself, pointing at that turn as its source.
    feb = mem.add("Service-to-service auth is API keys, one per consumer, rotated by hand.",
                  role="system", ts=at(2, 3))
    mem.remember("checkout-service", "auth_strategy", "API keys",
                 sources=feb.episode_ids, valid_from=at(2, 3), recorded_at=at(2, 3))

    # --- 12 June: the decision, and the reasoning behind it -------------------------
    jun = mem.add(
        "Decision: migrate service-to-service auth from API keys to OAuth 2.0 "
        "client credentials. Manual key rotation does not work for the three "
        "third-party integrators onboarding in Q3, and a leaked key today has no "
        "expiry.",
        role="system", ts=at(6, 12))

    # Two writes, because they are two different things. The slot moves...
    mem.remember("checkout-service", "auth_strategy", "OAuth 2.0 client credentials",
                 sources=jun.episode_ids, valid_from=at(6, 12), recorded_at=at(6, 12))
    # ...and the decision itself is a durable record that does not move when the next
    # one is made. `decided` is multi-valued, so this accumulates rather than superseding.
    mem.remember("checkout-service", "decided",
                 "migrate service-to-service auth to OAuth 2.0 client credentials",
                 sources=jun.episode_ids, valid_from=at(6, 12), recorded_at=at(6, 12))

    # --- two weeks later: "why are we using OAuth?" ---------------------------------
    print("Q. What is checkout-service's auth strategy?")
    live = [c for c in mem.get_all() if c.predicate == "auth_strategy"]
    print(" ", live[0].object)

    print("\nQ. What was the old strategy, and when did it change?")
    timeline = mem.history("checkout-service", "auth_strategy")
    for claim in timeline:
        until = claim.valid_to.date().isoformat() if claim.valid_to else "now"
        print(f"  {claim.object:<27} {claim.valid_from.date().isoformat()} -> "
              f"{until:<10}  [{claim.state}]")
    ended = [c for c in timeline if c.state == "ended"][0]
    print(f"  changed on {ended.valid_to.date().isoformat()}")

    # `state == "ended"` is the answer to a second question nobody asked out loud: the
    # world changed. A value that had never been true would read `retired` instead, and
    # `mem.forget()` is what writes that. Same timeline, different reason, and an
    # incident review needs to be able to tell them apart.

    print("\nQ. Why did we change it, and on what evidence?")
    provenance = mem.why(live[0].id)
    assert provenance is not None
    for episode in provenance.episodes:
        print(f"  [{episode.ts.date().isoformat()}] {episode.text}")
    print(f"  recorded by: {provenance.extractor} ({provenance.derivation.value})")
    print(f"  it replaced: {provenance.superseded[0].text}")

    print("\nQ. What would we have said on 1 April 2026?")
    april = [c for c in mem.get_all(as_of=at(4, 1)) if c.predicate == "auth_strategy"]
    print(" ", april[0].object)

    print("\nQ. What decisions has this service recorded?")
    for claim in mem.get_all():
        if claim.predicate == "decided":
            print(f"  [{claim.valid_from.date().isoformat()}] {claim.object}")

    mem.close()


if __name__ == "__main__":
    main()
