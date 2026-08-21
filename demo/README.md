# The answer-quality demo

Every benchmark this project reports measures **retrieval**: whether the right claim comes
back, ranked where it should be. None of them measures **answers** — whether an agent
reading memvara's output tells the customer the right thing. `docs/ROADMAP.md` has listed
that as the first item under *What is still missing* since it was written, and this
directory is the corpus, the arms and the harness for closing it.

It closes the *apparatus* half. The half still open is a reader behind an API: the one run
recorded below used an agent as the reader, which makes it a sanity check and not a
benchmark. [What one run produced](#what-one-run-produced) is specific about the
difference.

```
demo/scenario.py    the support history and the question set
demo/baselines.py   the five context-building arms, and the structured integration
demo/harness.py     the blinded run over those arms, and the scoring
```

`scenario.py` is pure data with no dependencies. `from demo import conversation, questions`
costs nothing and cannot fail; `demo.baselines` and `demo.harness` are imported by name
because they pull in numpy and the bench helpers.

```bash
PYTHONPATH=. python3 demo/harness.py --reader stub          # offline, one command
```

That is the whole run in one process: every arm, every question, judged by
`ContainmentJudge`, with no key, no file and no answerer. It is deterministic — two runs
produce the same report, and `test_the_offline_run_is_identical_twice` asserts it — which
is what makes the apparatus something CI can protect and a bisect can use.

**Its accuracy column is not a measurement of answers.** The reader is
`evalkit.StubReader`, which picks the line of the retrieved context with the most words in
common with the question; it cannot reason, cannot read a date and cannot combine two
turns. What its `correct` and `trapped` columns describe is the corpus and the arms. The
run prints that above its own table, twice.

Measuring answers needs a reader, and a reader is not in this process:

```bash
PYTHONPATH=. python3 demo/harness.py --dump runs/demo.jsonl
# ...answer them into runs/answers.jsonl as {"id": ..., "answer": ...}
PYTHONPATH=. python3 demo/harness.py --dump runs/demo.jsonl --answers runs/answers.jsonl
```

Phase one writes 100 blinded items — five arms × twenty questions, merged and shuffled
into one file, carrying `{"id", "system_prompt", "prompt"}` and nothing else. Phase two
re-derives which item belonged to which arm and scores.

---

## What the scenario is

Use case 01 from the marketing site — **a support agent that must not contradict itself** —
stated as the customer's complaint rather than our pitch:

> *The customer already corrected this, and the agent said the old thing anyway.*

The corpus has to be able to come back and say the complaint stands. If it could not, it
would be a demo rather than a measurement.

Sixty-four turns, one customer, January to August 2026. Dara Wray runs a joinery business
from a garden workshop in Sussex and has a mesh Wi-Fi system with a subscription on it.
Ten support tickets: a bad install, dropouts, an upgrade when a second workshop opens, a
dead node, a billing question, a downgrade when the workshop closes, a house move, and an
invoice that goes to the old address.

### The facts, and why each one moves

Six facts change over the history, and **they do not all change for the same reason**. That
is the distinction the library is built on, so the corpus contains both kinds:

| Fact | Was | Became | When | Why |
| --- | --- | --- | --- | --- |
| Plan | Home | Pro | 3 Mar 2026 | world changed |
| Plan | Pro | Home | 19 Jun 2026 | world changed |
| Delivery address | Coldharbour Road | Bramble Cottage | 26 Jul 2026 | world changed |
| Billing address | Coldharbour Road | Bramble Cottage | **5 Aug 2026** | world changed |
| Contact preference | phone | email | 22 Jun 2026 | world changed |
| Mobile number | 07700 900 118 | 07700 900 811 | 13 Feb 2026 | **the record was wrong** |
| Unit serial | HX2-4419-B | HX7-8802-D | 9 Apr 2026 | **the record was wrong** |

In memvara's vocabulary the first five are `close="ended"` — valid time closes, the claim
was true and stopped being true, and it still answers `valid_at=<back then>`. The last two
are `close="retired"` — transaction time closes, the record was never true, and it answers
nothing at any world-time. A corpus with only the first kind cannot tell a system that
models both apart from one that models supersession alone, which is most of them.

The two "record was wrong" facts were chosen to sit next to a matching "world changed" one
on the same subject, so the pair cannot be told apart by topic:

* the **mobile number** was mistyped (retired) while the **contact preference** on that
  same phone reversed (ended);
* the **serial** was misread off the power supply's label (retired) while the **hardware**
  around it genuinely changed — a node died and was replaced on 9 April.

Two more facts never move at all — the account name and the billing day — so a system that
reports change everywhere scores worse than one that reports it where it happened. Two
things are never stated: the Pro plan's monthly price and the identity of the card on file.

### The ten-day window

The delivery address changed on 26 July when they moved. The billing address did not change
until 5 August, when an invoice went to the old house and the customer complained. For ten
days the account had two different addresses on it, having had one for six months.

*"On 30 July, where did the invoices go?"* is the hardest question in the set and the one a
store with a single address field cannot answer at all — not because it answers wrongly, but
because the question is not expressible in it.

### Why the corpus is adversarial by construction

A superseded value mentioned once, early, and never again is not the failure the use case
describes. Real histories drag the old value back into view, so here **every superseded
value is re-surfaced late**, in a past-tense or mistaken framing:

* the old address is the **last address named in the transcript** (6 August), and by then it
  has been mentioned nine times against the new address's three;
* the retired serial is the **last serial a customer says** (6 August: *"Is it the 4419
  one?"*);
* the Pro plan is the **last plan named in the transcript** (6 August, past tense);
* the turn that reverses the contact preference states the superseded one inside itself —
  *"I know I asked for calls back in February"* — so the newest turn on that subject is also
  the one that says the old answer out loud.

Recency and emphasis both point at the wrong answer. That is what makes a wrong answer here
evidence rather than noise, and
`test_the_superseded_values_are_re_surfaced_after_the_values_that_replaced_them` exists so
that a tidy-up cannot remove it silently.

---

## The question set

Twenty questions. Each carries a `gold` (the authored answer) and usually a `trap` (the
specific wrong answer a system with no bitemporal handling produces). Report **both** —
"how many right" and "how many gave the superseded answer" — because the second is the
number the marketing claim actually rests on.

### `kind="current"` — what is true at `asked_at` (10)

The trap is the **superseded** value. Plan, delivery address, billing address, contact
preference, serial, mobile, plus the two controls (account name, billing day) which have no
trap because nothing supersedes them.

Two of these are asked **mid-history** — `q_plan_current_in_april` and
`q_serial_current_in_may` — so that a system reading past `asked_at` is caught rather than
rewarded. `q_plan_current_in_april` is the matched control for `q_plan_current`: same
wording, same fact, an ask instant where the correct answer is the opposite value. A system
that gets one right and the other wrong has a *time* problem, not an extraction problem, and
the pair is what makes that distinction visible in the results table.

### `kind="historical"` — what was true at a named past instant (5)

The trap is the **current** value, which is what a store keeping only the latest value per
field returns. Note the direction reverses between the two groups: answer everything with
the most recent thing you retrieved and the historical group fails; answer everything with
the most emphatic and the current group fails.

`q_plan_current` (gold: Home) and `q_plan_mid_march` (gold: Pro) are the pair the product's
thesis rests on. Each one's gold is the other one's trap.

### `kind="correction"` — the record was wrong, and the answer must say so (3)

These are the questions no store with a single clock can answer. A superseded value and a
retracted value are both just "the old value" in such a store; here the honest answer is
that one of them was **never a value**. `q_which_were_corrections` asks for the split
directly and is the single most diagnostic question in the set.

> **Instruction to whoever scores this — do not count traps by containment on `correction`
> questions.** A correction gold contains its own trap by construction: *"it was X and is
> now Y"* is the entire answer, so a grader asking `trap in answer` marks every **correct**
> correction answer as a trap hit, and the headline trapped rate comes out high because the
> reader did well. On these three questions read `trapped only` (trapped and not correct),
> which `demo/harness.py` already reports, and score correctness on whether the standing
> value is present **and labelled as standing**. The other three kinds are safe for
> containment: `test_a_trap_is_not_a_substring_of_its_gold_except_where_the_answer_must_
> name_both` enforces that they never overlap.

### `kind="unanswerable"` — an honest reader says it does not know (2)

The Pro plan's monthly price and the card on file. `q_pro_price` carries a trap (`£79`, the
price of one extra node, the only money in the transcript) because that failure has exactly
one shape. `q_card_on_file` has `trap=None` because a system inventing an answer there could
invent anything.

`trap=None` appears on five questions in total and is a claim, not an omission: it says the
failure mode is diffuse. Inventing a trap for those would inflate the trap-rate with
questions that never had a single failure mode.

### `closure` — which clock closed, and why the headline needs it

Every question also carries `closure`, in `memvara.types.Closure`'s vocabulary. It is
**orthogonal to `kind`**: `q_serial_current` is `kind="current", closure="retired"` and
`q_serial_correction` is `kind="correction", closure="retired"` — the same wrong record,
asked two ways.

| `closure` | | questions |
| --- | --- | --- |
| `"ended"` | valid time closed; the world changed | 9 — plan, both addresses, contact preference |
| `"retired"` | transaction time closed; the record was wrong | 5 — serial, mobile |
| `None` | nothing closed, or both at once | 6 — the two controls, `q_plan_history`, `q_which_were_corrections`, the two unanswerables |

**Break the trapped rate down by this field.** A single trapped percentage merges two
failures that mean opposite things: *served a value that has expired* is a stale cache, and
*served a value that was never true* is the thing this library exists to prevent. One number
covering both throws away the distinction the whole model is built on.

One wrinkle worth knowing before quoting a total: `q_pro_price` has a trap (`£79`) and
`closure=None`, because that trap is a distractor rather than a superseded value. Giving it
is a reading failure, not a time-handling one. Splitting by `closure` separates it out
automatically; a single number includes it.

### `about` — the valid-time instant, for automated checks

`about` is what `valid_at=` would be set to by something answering the question
mechanically. It is set on four of the five `historical` questions and `None` everywhere
else, for two different reasons:

* on `current` questions, because `asked_at` already carries the instant — the two are not
  interchangeable, and `q_plan_current_in_april` is the proof: same wording as
  `q_plan_current`, no `about`, and its answer depends entirely on `asked_at`;
* on every `retired` question, because **there is no such instant**. A retracted value was
  never true at any world-time, so no `valid_at=` returns it. Those questions move on the
  belief axis (`known_at`), not this one. That is a consequence of the model rather than a
  gap in the corpus.

The exception is `q_plan_history`, which is `historical` with no `about`: it asks for the
whole sequence, so there is no single instant to name and inventing one would make the field
a lie. It is exempted **by name** in the tests, so a *new* historical question that forgets
`about` still fails.

`test_each_about_falls_inside_the_interval_its_gold_was_true_in` checks the field for
meaning rather than for merely being a date: each instant must fall strictly between the
turn that opened its gold's interval and the turn that closed it. A plausible date on the
wrong side of a change would make every automated check agree with a wrong answer.

---

## Authored, not derived

The golds were written by hand from the transcript. **None of them was produced by running
memvara and recording what it said** — an answer key derived from the system under test
measures nothing.

Every question carries a comment in `scenario.py` naming the turn that justifies it, and
`tests/test_demo_scenario.py` holds a hand-written `EVIDENCE` table mapping each question to
a phrase from that turn. The test asserts the phrase is in the history *and* that it is
visible at `asked_at`, so:

* a gold whose supporting turn is edited away fails, instead of outliving its evidence;
* a trap quoted from a turn *after* the question was asked fails, because a wrong answer no
  system could have given is not worth counting;
* a new question with no evidence entry fails, which is the point.

---

## The five arms

A number with no control arm is not a measurement. "95% correct with memvara" is
uninterpretable on its own: the questions might be answerable from priors, or the whole
transcript might fit in a prompt and make a memory layer pointless. `demo/baselines.py`
answers both objections by construction.

| arm | what it is | why it is here |
| --- | --- | --- |
| `none` | no context at all | the floor. A question this arm gets right was answerable from priors or from its own wording, and is worth no credit to anybody |
| `full_transcript` | every visible turn, chronological, **uncapped**, dated | the honest competitor, and at this corpus size a serious one. Capping it or stripping its dates would be tying an opponent's hands |
| `naive_rag` | top-k cosine over raw turns | isolates *bitemporal reasoning*: same embedder, same `k`, same visible turns as the memvara arms, so a difference cannot be vector quality |
| `memvara` | the shipped defaults — a transcript dropped in with no `llm=` | what a first evaluation actually does |
| `memvara_structured` | a declared predicate schema plus facts written from the desk's own fields | what a deployment actually does |

Two constants keep this a comparison rather than five experiments. `MAX_CONTEXT_CHARS =
4000` caps the three retrieval arms and deliberately not `full_transcript`, which is the
long-context arm and whose size *is* the finding. `EMBED_DIM = 512` with `HashingEmbedder`
pinned explicitly on all three embedding arms, because `default_embedder()` returns a
sentence-transformers model wherever that package happens to be installed — left alone,
the comparison would be a different experiment on a machine with the extra than on one
without, and the difference would show up as a quality delta with no code change behind
it. It also means those arms run a *lexical* approximation rather than a semantic model:
the absolute numbers are pessimistic for all three, and the deltas are the part that
transfers.

**The `asked_at` cutoff applies to every arm, not just to memvara.** `visible_turns`
truncates the input for all five; the arms must differ in what they do with the history,
not in how much of it they may see. Inside the two memvara arms the cutoff is enforced at
ingest rather than by a `known_at=` read, and that is the only correct option:
`Memvara.add()` stamps `valid_from` from the turn's `ts` but `recorded_at` from the wall
clock, so replaying an archived transcript today produces claims all recorded *today*, and
a `known_at=<July>` read would correctly hide every one of them and return an empty
context for every question.

### Why the product has two arms, and why neither may be deleted

**On this corpus the `memvara` arm produced zero claims from sixty-four turns.** It now
produces six, and both halves of that sentence are the point.

```python
from demo import conversation
from memvara import HashingEmbedder, Memvara, NullLLM

mem = Memvara(embedder=HashingEmbedder(dim=512), llm=NullLLM(), user="customer")
for turn in conversation():
    mem.add(turn.text, role=turn.role, ts=turn.at)

mem.stats()
# {'episodes': 64, 'claims': 6, 'live_claims': 3, 'ended_claims': 2,
#  'invalidated': 1, 'embeddings': 70}
```

Summed over those 64 writes: `unextracted=29`, `skipped=30`, `llm_calls=0`.

The six, and the turn each comes from, are pinned by hand in
`tests/test_demo.py::EXTRACTED` — an expectation recorded from a run would measure
nothing, so that list is the precision half and a seventh claim fails it until somebody
has read the seventh against the transcript. Two of the slots supersede: the delivery
address ends when the customer moves, and the contact preference ends when it reverses,
each on the world clock, with the displaced value still answering `valid_at=<back then>`.

It is still a long way short of what the transcript holds. The plan, the serial, the
mobile correction and the billing address are all in it and none of them is in a sentence
form a rule can read, so four of the six facts the corpus was built around are invisible
to this arm. **Its row still cannot test the whole of what this comparison exists to
test** — but it is no longer lexical episode retrieval with a different ranker, which is
what it was while the claim tier was empty. It is worth measuring because it is what
somebody evaluating the library over a weekend actually sees.

`memvara_structured` is the other real configuration and the one the product is for. A
support integration does not ask a model to read prose back out of its own database; it
writes structured facts from the fields its ticketing system already has. That path needs
no API key and exercises the entire bitemporal machine offline.

### The integration, in full

Everything between `SUPPORT_PREDICATES` and `ARMS` in `baselines.py` is the integration,
kept in one place so it can be read — and counted — as the thing a deployment would have
to write. It is three pieces:

**`SUPPORT_PREDICATES`** — eight `PredicateSpec`s, and **declaring them is required, not
decoration.** `PredicateRegistry` defaults an unknown predicate to `MANY`, so without this
every value accumulates: `Pro` and `Home` both stay live, nothing supersedes anything, and
"which plan?" comes back with two plans. Nothing warns, because accumulating is exactly
what `MANY` is supposed to do.
`test_a_predicate_left_at_the_default_cardinality_stops_superseding_silently` injects the
omission and watches the slot grow two answers.

Three of the eight decisions do more than name a cardinality. `delivery_address` and
`billing_address` are *separate predicates* rather than two values of one, which is the
only way the ten-day window is expressible at all — `ONE` on a single `address` slot would
have made the move overwrite billing too, and `MANY` would have left both live with
nothing to say which was which. `contact_preference` is the one genuine `MANY`, because on
6 February the customer names two acceptable channels in one breath, and it pays for that
on 22 June: a `MANY` slot supersedes nothing on its own, so the reversal has to close the
slot explicitly. And `serial` is `ONE` **per subject** — the node that died on 9 April is a
different subject, so its serial never competes with the main unit's; written as
`("account", "serial")` it would have had to be `MANY` and the correction would have had
nothing to close.

The whole thing is added to `BUILTIN_PREDICATES` rather than replacing it, so the two
memvara arms differ by an addition and the rule extractor's shared vocabulary is still
present in both.

**`SUPPORT_FACTS`** — seventeen `Write` rows, and the `Write` dataclass is where the
ended/retired distinction becomes a verb:

| `mode` | what it does | used for |
| --- | --- | --- |
| `assert` (14) | new value; a `ONE` predicate closes what it displaces on **valid** time | the plan changes, both address moves, every first statement of a fact |
| `correct` (2) | the value on record was never true: closes **transaction** time on it and leaves its valid interval exactly as written | the mistyped mobile, the misread serial |
| `replace` (1) | close every live value in the slot on valid time, then assert | `contact_preference`, where `MANY` cardinality supersedes nothing on its own |

An arm that used one of these everywhere would produce a number that looked fine and meant
nothing: retire the address moves and "what did we ship to in April" goes blank; end the
mobile transposition and the store asserts that a number nobody ever had stopped being
theirs on 13 February. `Write` also carries the two instants separately — `at` is when the
desk found out, `since` is when the fact started being true — because for a correction the
second is *before* anyone knew it.

**`apply_facts`** — three verbs, one branch each. The one subtlety is that `correct` uses
`supersede(..., at=fact.at, close="retired")` rather than `remember(close="retired")`,
because `remember` hands the reconciler the wall clock: a replay would record that we
stopped believing the transposed number *this afternoon* rather than on 13 February, and
every `known_at=` audit over the correction would be wrong. It refuses loudly if a
correction finds anything other than exactly one standing value.

The structured arm is also the only one that spends `Question.about`, as `search(valid_at=)`.
Nothing parses a date out of question prose: where `about` is `None` the read is at the
present, which is what `recall()` does.

### One rendering, asserted rather than hoped

The `memvara` arm calls `recall()` directly, because `recall()` is what an integration
drops into a prompt and its headers, flattening and claims-before-episodes ordering are
part of what is being measured. The structured arm cannot: `recall()` takes no `valid_at=`,
deliberately, so an arm answering a historical question at a named world-time has to
re-render `search()` results itself. `render_recall` is that re-render, and
`test_the_structured_arms_rendering_is_byte_identical_to_recall` runs both over the whole
corpus and compares — so if the library's formatter changes and the demo does not, the
demo fails rather than quietly measuring a formatter this repository does not ship.

---

## What one run produced

Context size is deterministic and comes out the same every time. This is real output from
`demo/harness.py`:

```

  arm                 mean chars  max chars  mean ~tokens  items used / turns seen
  ------------------  ----------  ---------  ------------  -----------------------
  none                         0          0             0               0.0 / 60.8
  full_transcript           9803      10263          2451              60.8 / 60.8
  naive_rag                 2329       2846           582              12.0 / 60.8
  memvara                   2043       2331           511              12.0 / 60.8
  memvara_structured        1671       2032           418              12.0 / 60.8
```

`~tokens` is `chars // 4`, an estimate and not a tokenizer — `CHARS_PER_TOKEN` says so.
`naive_rag` retrieved every visible turn on **0 of 20** questions, so it is a retrieval arm
throughout rather than `full_transcript` in a different order; the harness prints a warning
when that stops being true.

The scores are a different kind of number. One run has been done, **with an agent as the
reader** — there is no API key in this repository — and hand-audited afterwards to correct
for the containment judge's known false positives (a correct `correction` answer contains
its own trap by construction) and false negatives (a correct paraphrase is marked wrong):

| arm | context | correct | genuine traps |
| --- | ---: | ---: | ---: |
| `none` (floor) | 0 tok | 10% | 0 |
| `full_transcript` | 2,451 tok | **100%** | 0 |
| `naive_rag` | 582 tok | 80% | 0 |
| `memvara` | 519 tok | 95% | 0 |
| `memvara_structured` | 430 tok | 95% | 0 |

**These context sizes are the ones that run was answered against, and they are no longer
what the arms produce.** The offline write path was widened afterwards, so the `memvara`
arm now builds 511 tokens and `memvara_structured` 418 — the table above is the current
apparatus, this one is a record of a past run. The accuracy column belongs to the contexts
in *this* table and cannot be re-attached to the new ones without answering them again.

**Read `evalkit.FileReader`'s docstring and the banner `demo/harness.py` prints above its
own table before quoting any of this.** They say, and they are right, that a run whose
reader is an agent **is not reproducible**: there is no model id, no seed and no
temperature to put beside the number, and the same contexts answered again will not give
the same answers. It is a sanity check that the pipeline produces sane answers from real
retrieval. It is not a benchmark, it cannot rank systems, and it must never sit beside a
published LOCOMO or LongMemEval score.

What it does show, stated as narrowly as it deserves:

* **The whole-transcript arm scored 100%.** At this corpus size a careful reader given
  everything gets everything, so the memory layer earns nothing on accuracy here. What it
  earns is the size column: **5.7× fewer tokens for 95%** (2,451 → 430; the `memvara` arm
  is 4.7×). That is a claim about a *slope* — retrieval context is flat in corpus length
  while transcript context is linear — and this run has exactly one corpus size, so the
  slope is argued and not measured. A corpus ten times longer is what would turn it into
  evidence.
* **`naive_rag` was the only arm that genuinely lost information**, and its four failures
  were exactly the bitemporal ones. That is the comparison the corpus was built for.
* **The trap metric produced no signal at all.** Zero genuine traps in every arm, including
  the one with no time handling: the reader never gave a superseded value, so `naive_rag`'s
  four misses were wrong in some other way. The failure mode the product describes needs a
  reader that skims. This is reported rather than dropped, because `trapped` is the
  headline column, and it is the column that did not move.
* **The floor is 10%, which is two questions of twenty** — and an arm with no context
  abstains on the two `unanswerable` questions by construction, which the harness itself
  flags as an artefact. Read the floor as at or near zero on the eighteen questions that
  have an answer.
* **The two memvara arms tie at 95%**, which is the result to be most careful with: the
  `memvara` arm reached it with an empty claim tier, so its 95% is lexical episode
  retrieval scoring well on a corpus small enough for that to work, not bitemporal
  reasoning doing its job.

---

## Limitations, stated rather than buried

**The corpus is synthetic, and it was authored by the same party that wrote the library.**
That is a real limitation and it cuts in one direction: the questions were written by
someone who knows what bitemporal storage is good at. Nothing here shows that a corpus
collected from a live support desk would have the same shape, or the same proportion of
questions where the distinction matters at all.

It buys one thing in exchange, which is why it exists: **no memorisation confound**. LOCOMO
and LongMemEval are public, so a hosted reader may have seen them, and a number that might be
recall of the answer key is not a number. Nobody has seen these turns.

Two further limits worth naming:

* **One customer.** Depth over breadth was a deliberate choice — six facts with real
  histories beat sixty facts with none — but it means a single retrieval failure moves the
  percentage by five points. Treat the per-question table as the result and the headline
  percentage as a summary of it.
* **British English, one register.** The extractor's English-centrism is measured elsewhere
  (`gate.drop` and `fast.miss` are tagged by script); this corpus does not test it.

The honest framing for any number that comes out of here: *on an authored corpus designed to
contain the distinction, with no memorisation confound, the reader answered N of 20, and
gave the superseded value M times.* Not: *memvara is N% accurate.*
