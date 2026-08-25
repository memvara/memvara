# Worked turns

Short patterns. Each one shows the tool sequence, then the mistake that
records the wrong history. Tool descriptions already say when to call each
tool; these turns are about the order.

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
