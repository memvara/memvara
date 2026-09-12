# Tutorial: your first Memvara memory

In this tutorial you will install Memvara, create a memory store, teach it three facts
about a person over time, and ask it a question three different ways. By the end you
will have seen the one thing that makes Memvara different from a normal database or a
vector store: it can tell you not just what is true right now, but what was true at any
point in the past — and it does this without ever calling a language model.

This takes about ten minutes. You do not need an API key, a database server, or an
internet connection.

## What you need

- Python 3.10 or later
- A terminal

That's it. Memvara's only dependency is `numpy`.

## Step 1: Install Memvara

```bash
pip install memvara
```

Check it worked:

```bash
python3 -c "import memvara; print(memvara.__version__)"
```

You should see a version number printed, such as `0.9.0`.

## Step 2: Open a memory store

Create a file called `tutorial.py` and add:

```python
from datetime import datetime, timezone
from memvara import Memvara, NullLLM

UTC = timezone.utc
mem = Memvara("tutorial.db", user="alice", llm=NullLLM())
```

A few things are happening here:

- `"tutorial.db"` tells Memvara to store memory in a file called `tutorial.db` in the
  current directory. If you leave this out, Memvara keeps everything in memory and
  forgets it when the program exits — useful for experiments, but not for this
  tutorial, where we want the file to persist between runs.
- `user="alice"` says that, unless you tell it otherwise, every fact you write or read
  belongs to a person named "alice". In a real application this would usually be your
  actual user's ID.
- `llm=NullLLM()` tells Memvara not to use any language model. You will still be able to
  record facts precisely — you just won't be able to hand it a paragraph of free text and
  have it figure out the facts on its own. We're doing that on purpose here, so nothing
  in this tutorial needs an API key.

Run the file. Nothing should print, and a new file `tutorial.db` should appear in your
directory.

## Step 3: Teach it a fact

Alice moves house sometimes, and we want to keep track of where she lives. Add this to
`tutorial.py`:

```python
def on(month, day):
    return datetime(2026, month, day, tzinfo=UTC)

mem.remember("Alice", "lives_in", "Berlin", valid_from=on(1, 10), recorded_at=on(1, 10))
```

`remember` takes three required arguments — who the fact is about (`"Alice"`), what kind
of fact it is (`"lives_in"`), and the value (`"Berlin"`) — plus two dates you'll come back
to in a moment.

Run the script again, then add a line to check it worked:

```python
print([c.object for c in mem.get_all()])
```

You should see:

```
['Berlin']
```

## Step 4: Add two more facts, and watch the old one disappear

Alice moves twice more. Add these two lines right after the first `remember` call:

```python
mem.remember("Alice", "lives_in", "London",   valid_from=on(3, 15), recorded_at=on(3, 15))
mem.remember("Alice", "lives_in", "New York", valid_from=on(6, 2),  recorded_at=on(6, 2))
```

Run the script again and check `mem.get_all()`:

```
['New York']
```

Only one city comes back, even though you wrote three. This is on purpose. Memvara
already knows that "where does Alice live" is the kind of fact that has exactly one
current answer — it's built into the vocabulary it ships with — so each new city
automatically replaces the previous one. No code decided this for you at the moment you
wrote it; the store worked it out from the kind of fact `lives_in` is. (You'll teach it
your own kinds of facts in a later guide — see
[Define your own fact types](../how-to-guides/define-your-own-fact-types.md).)

## Step 5: Ask what was true in the past

Here is the part a plain database or a vector store cannot do. Add:

```python
print("Now:      ", [c.object for c in mem.get_all()])
print("On Mar 20:", [c.object for c in mem.get_all(as_of=on(3, 20))])
print("On Jan 20:", [c.object for c in mem.get_all(as_of=on(1, 20))])
```

Run it:

```
Now:       ['New York']
On Mar 20: ['London']
On Jan 20: ['Berlin']
```

Three different questions, three different correct answers, and you did not have to
reconstruct anything. Overwriting "Berlin" with "London" never actually deleted the fact
that Berlin was once correct — it just closed the period during which Berlin was the
answer. Memvara kept the old value and remembered exactly when it stopped applying.

`as_of=` is how you ask "what did you believe was true at this exact moment". You'll meet
its two more precise cousins, `valid_at=` and `known_at=`, in
[Ask about the past](../how-to-guides/ask-about-the-past.md) — they matter once facts
arrive late, which happens more often than you'd expect.

## Step 6: See the whole history at once

You don't have to guess dates to explore the timeline. Add:

```python
for claim in mem.history("Alice", "lives_in"):
    print(claim.object, "-", claim.state)
```

```
Berlin - ended
London - ended
New York - live
```

`ended` means the fact was true and then stopped being true — Alice really did move.
That's a different situation from a fact that was *never* true, which Memvara calls
`retired`. You'll use that distinction in the next step.

## Step 7: Correct a mistake

Suppose you had actually typed the wrong city by accident — Alice never lived in
"London" at all, you misheard her. This is different from Alice moving away from London;
nothing in the world changed, the record was simply wrong from the start. Memvara has a
separate call for this:

```python
mem.forget("Alice", "lives_in", at=on(3, 15))
```

Run `mem.history("Alice", "lives_in")` again and you'll see London's state change to
`retired` rather than `ended`. Nothing was deleted — you can still see that London was
recorded and then withdrawn — but Memvara now knows this wasn't a case of the world
changing. That distinction matters later, when you or a teammate is trying to work out
whether a past decision was correct at the time or was simply a data-entry error. See
[Record and correct a fact](../how-to-guides/record-and-correct-a-fact.md) for the full
set of ways to fix a mistake.

## Step 8: Get a narrated answer

Finally, ask Memvara to explain itself in a sentence rather than a list:

```python
print(mem.ask("where does Alice live?", at=on(3, 20)).text)
```

```
where does Alice live?
  asked about 2026-03-20

Alice lives_in: Berlin.
```

`ask()` is the read to reach for when you want a sentence a person (or an AI agent) can
act on directly, rather than raw data you have to interpret yourself.

## What you've learned

- A fact in Memvara is a **claim**: who it's about, what kind of fact it is, and its
  value.
- Memvara can answer questions about the present and about any point in the past, and it
  does this from stored data — never by asking a language model to guess.
- A fact that stops being true (`ended`) and a fact that was never true (`retired`) are
  recorded differently, so you can always tell which one happened.
- None of this required an API key or a network connection.

## Next steps

- [Record and correct a fact](../how-to-guides/record-and-correct-a-fact.md) — the full
  set of tools for writing facts properly, with sources and confidence.
- [Ask about the past](../how-to-guides/ask-about-the-past.md) — the difference between
  "what was true then" and "what did we believe then", and why it matters.
- [What problem does Memvara solve?](../explanation/what-problem-memvara-solves.md) — the
  bigger picture, if you want to understand *why* Memvara is built this way before going
  further.
