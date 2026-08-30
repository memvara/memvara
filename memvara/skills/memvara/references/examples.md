# Worked turns

Short patterns. Each one shows the tool sequence, then the mistake that
records the wrong history. Tool descriptions already say when to call each
tool; these turns are about the order, and the sections below are about the
shape of the call — the thing a paragraph cannot show you next to the call you
should not have made.

## 1. The record was never right

User: "I never lived in Berlin."

1. `memory_recall` on the living-situation question.
2. `memory_search` for the same, take the claim id.
3. `memory_why` on that id. Show the excerpt. Do not defend it.
4. User: "That was about a colleague, not me."
5. `memory_forget` on the id, then `memory_remember` the fact you now have
   (if there is one).

Do not start at step 5. A lone `memory_remember` of a new city says the
world moved. Here the world never held that value for this person.

## 2. The world changed

User: "I moved to Lisbon."

1. Same three reads as above.
2. The excerpt is something they actually said last year about Berlin.
3. `memory_remember` with predicate `lives_in` and object `Lisbon`. If they
   named the date it stopped, pass that as `true_since` on the new fact
   (or close the old interval with `memory_end` and `at`).

Do not `memory_forget` Berlin. Forgetting says Berlin was a mistake. It
was not; they lived there.

## 3. This server is session-scoped

`memory_stats` shows a session field that is not `*`.

User: "Remember I'm on pager duty this week."

Store it if it passes the embarrassing-next-week test, and say at the
same moment that it will not carry into the next chat. Do not let them
think a durable fact was kept. Unsetting `MEMVARA_SESSION` is their
operator's job, not a second write.

## 4. Not worth storing

User pastes a stack trace, or asks you to explain a function.

Do not `memory_add` it. The transcript already has it. Next week's agent
does not need the dump, and stuffing the store with restated context
makes the durable facts harder to find.


# Two calls that differ only in the type

Same discovery, written twice. The first spent context on every turn of every
session afterwards; the second did not.

    memory_remember(subject="memvara_cloud", predicate="deploy_gotcha",
                    object="polar-drain and polar-send share a profile and do "
                           "opposite jobs, so naming one starts only that one",
                    memory_type="procedural")     # joins the standing set

    memory_remember(subject="memvara_cloud", predicate="known_defect",
                    object="polar-drain and polar-send share a profile and do "
                           "opposite jobs, so naming one starts only that one",
                    memory_type="semantic")       # found when someone asks

Nobody instructed anything in either. The subject is a container, not a person.
Ten of thirty-two notes in one live store were written the first way.

Compare an instruction that really is one, which the person said out loud:

    memory_remember(subject="user", predicate="never_do",
                    object="never add an AI attribution to a commit, PR or "
                           "issue, in any repository",
                    memory_type="procedural")

Two more that are easy to get backwards:

    memory_remember(subject="memvara", predicate="version", object="0.4.0",
                    memory_type="episodic")   # "we shipped it this morning"

    memory_remember(subject="memvara_cloud", predicate="uses_tool",
                    object="pytest", memory_type="semantic")

The second is about a repository. "The person prefers pytest" would be the
other type, with `user` as the subject — same word, different owner.

# One call each

The argument shape, and beside it the call to avoid. Reasons live in the tool
descriptions; these are the calls.

    memory_recall(query="how does this person want commits written?")
    # avoid: reaching for it when you need an id — nothing here has one

    memory_search(query="attribution in commits", k=5,
                  memory_types=["procedural"])
    # avoid: reading its numbers out to anyone

    memory_standing()
    # avoid: memory_standing(query=...) — there is no such argument

    memory_since(since="2026-08-25T04:00:00Z")
    # avoid: a local time. Anything west of Greenwich lands in the future

    memory_neighborhood(entity="memvara-cloud", depth=2, k=10)
    # avoid: opening with this. Try the single-note question first

    memory_paths(source="Alice", target="Acme", depth=3)
    # avoid: passing one end and hoping

    memory_remember(subject="user", predicate="prefers",
                    object="always work in a git worktree, never the main "
                           "checkout, because work spans three sibling repos",
                    memory_type="procedural", true_since="2026-08-20")
    # avoid: omitting true_since on anything you learned about the past

    memory_add(text="I moved to Lisbon last March and I am staying")
    # avoid: this entirely where stats said fast-path-only

    memory_add(text=the_log_they_pasted, role="system")
    # avoid: leaving role at "user" on anything they pasted rather than said

    memory_forget(claim_id="cl_1a2b3c")
    # avoid: using it because a value went out of date

    memory_end(claim_id="cl_1a2b3c", at="2026-03-01")
    # avoid: omitting `at` and letting now stand in for when it stopped

    memory_why(claim_id="cl_1a2b3c")
    # avoid: skipping it and writing from the complaint instead

    memory_history(subject="user", predicate="lives_in")
    # avoid: quoting its rows as current

    memory_stats()
    # avoid: calling it to check whether one particular fact is stored
