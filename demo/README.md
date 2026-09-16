# The answer-quality demo

Every benchmark this project reports measures **retrieval**: whether the right claim comes
back, ranked where it should be. None of them measures **answers** — whether an agent
reading memvara's output tells the customer the right thing. `docs/ROADMAP.md` has listed
that as the first item under *What is still missing* since it was written, and this
directory is the corpus, the arms and the harness for closing it.

The apparatus is complete: the corpus, the five arms, a blinded round trip for a person or
an agent, a reader behind an API with every parameter pinned and printed, and a second
corpus size ([Two corpus sizes](#two-corpus-sizes)). The one run recorded below still
used an agent as the reader, which makes it a sanity check and not a benchmark;
[What one run produced](#what-one-run-produced) is specific about the difference, and a
run with the hosted reader at both sizes is the next thing to record.

```
demo/scenario.py    the support history and the question set
demo/distractors.py generated tickets that scale the history without moving any fact
demo/baselines.py   the five context-building arms, and the structured integration
demo/harness.py     the blinded run over those arms, and the scoring
```

`scenario.py` and `distractors.py` are pure data with no dependencies. `from demo import
conversation, questions, scaled_conversation` costs nothing and cannot fail;
`demo.baselines` and `demo.harness` are imported by name because they pull in numpy and
the bench helpers.

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

Measuring answers needs a reader. The one that makes the run reproducible is a model
behind an API:

```bash
export ANTHROPIC_API_KEY=...            # or OPENAI_API_KEY, with --reader openai
PYTHONPATH=. python3 demo/harness.py --reader anthropic --judge llm \
    --model claude-opus-5 --effort low --max-tokens 4096 --thinking adaptive \
    --checkpoint runs/hosted.checkpoint.jsonl --concurrency 4 --out runs/hosted.jsonl
```

That is the whole run in one process: every arm, every question, answered by the model
and graded by a second call to it. Everything that decides an answer is pinned by a flag
and printed under the report's title exactly as it was sent, which is what makes the
number quotable and the run repeatable:

* `--model`, `--effort`, `--max-tokens` and `--thinking` on `--reader anthropic`.
  `--thinking default` sends no thinking field and the model applies its own default;
  `adaptive` and `disabled` send that setting explicitly. On a model that thinks, thinking
  and the answer share `--max-tokens`, which is why the default is 4096 rather than the
  few dozen tokens an answer needs. The Anthropic models reject `temperature`, `top_p` and
  `top_k`, so nothing is sampled, the header says so, and there is no seed to pin: two
  runs will differ, and the per-question rows in `--out` are the unit to compare.
* `--model`, `--max-tokens`, `--temperature` and `--sampling-seed` on `--reader openai`.
  The seed is sent only when given, and the header prints whichever was the case.
* `--base-url`, `--api-key-file` and `--extra-body` point `--reader openai` at a server of
  your own that speaks Chat Completions. The base URL and the extra body are printed and
  keyed in the checkpoint, because a different server or `{"chat_template_kwargs":
  {"enable_thinking": false}}` changes the answers. The key is read from the file at run
  time and appears nowhere; the file is refused if other users can read it.
* `--judge llm` grades with the reader's twin: the same provider and the same parameters,
  with `--judge-model` swapped in when given. `--judge containment`, the default, is free
  and wrong in the known directions the report lists under its tables.
* `--checkpoint PATH` appends every completed model call to a JSONL file as it completes.
  A run that dies resumes on the next invocation with the same path and pays only for
  what it lost; the header says how many calls were replayed. The key covers the pinned
  settings and the whole prompt, so a changed flag starts a fresh set of rows rather than
  replaying another configuration's answers.
* `--concurrency N` issues N model calls at once. Rows come back in item order and the
  cost ledger is billed in item order, so the report is identical whatever N is.

The report carries the floor (`none`) and the ceiling (`full_transcript`) beside the three
memory arms and names which is which in its header, because a memory score without both
beside it is uninterpretable. Under the tables it prints what the run cost, from the usage
the provider reported, and counts the answers that never finished — a `max_tokens`
truncation or a `refusal` — apart from wrong answers, because both arrive as a short or
empty string and would otherwise be averaged in as a memory layer that surfaced bad
evidence. Each row of `--out` carries the reader's `stop_reason` for the same reason.

The provider's SDK has to be installed: `pip install 'memvara[anthropic]'` or
`pip install 'memvara[openai]'`. Neither is a dependency of the library, and the whole
harness runs without them under `--reader stub`.

A person or an agent can be the reader instead, through a blinded round trip. Its number
is a sanity check rather than a measurement — there is no model id, seed or temperature
to quote beside it — and it is the configuration the one recorded run used:

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

**On this corpus the `memvara` arm produces zero claims from sixty-four turns.**

```python
from demo import conversation
from memvara import HashingEmbedder, Memvara, NullLLM

mem = Memvara(embedder=HashingEmbedder(dim=512), llm=NullLLM(), user="customer")
for turn in conversation():
    mem.add(turn.text, role=turn.role, ts=turn.at)

mem.stats()
# {'episodes': 64, 'claims': 0, 'live_claims': 0, 'ended_claims': 0,
#  'invalidated': 0, 'embeddings': 64}
```

Summed over those 64 writes: `unextracted=34`, `skipped=30`, `llm_calls=0`. The rule
extractor's vocabulary is first-person declaratives — "I live in X", "my name is X" — and
a support history is not written that way; with no `llm=` there is no extraction tier
behind the rules, so the claim tier stays empty. Its rendered context has no
`Known about the user` header in it at all, only the episode tail.

An empty claim tier is not a weaker version of the feature set, it is none of it: no
`(subject, predicate)` slot means no supersession, no valid-time closing, and no
bitemporal reasoning of any kind. That arm is lexical episode retrieval with a different
ranker, and **its row cannot test the claim this comparison exists to test.** It is still
worth measuring, because it is what somebody evaluating the library over a weekend will
actually see, and because it is the documented behaviour of the shipped defaults.

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

The structured arm is also the only one that spends `Question.about`, as
`recall(valid_at=)`. Nothing parses a date out of question prose: where `about` is `None`
the read is at the present, which is what `recall()` does without it.

### One rendering

Both memvara arms call `recall()` directly, because `recall()` is what an integration
drops into a prompt, and its headers, flattening and claims-before-episodes ordering are
part of what is being measured. The structured arm passes `valid_at=question.about`, so
a historical question is answered from the block `recall()` renders for that day, under
the header that names it. The demo used to re-render `search()` results itself for that
case, because `recall()` had no `valid_at=`; it no longer needs to, so it does not.

---

## Two corpus sizes

The claim the size table makes — retrieval context is flat in corpus length while
transcript context is linear — is a slope, and the authored corpus is one point on it.
The second point is the same corpus scaled: `demo/distractors.py` pads the sixty-four
authored turns with generated support tickets for the same customer and product, and
`--corpus-scale N` on `demo/harness.py` runs any reader over the padded history. The
questions, the golds, the `asked_at` cutoffs and `SUPPORT_FACTS` are untouched; only the
haystack grows. Scale 1, the default, is the authored corpus itself.

```bash
PYTHONPATH=. python3 demo/harness.py --reader stub --corpus-scale 10
```

What a generated turn may say is the whole design, and it is tested rather than promised.
A distractor never names the value of any fact a question is about, old or new — no
address, plan name, serial, mobile number or contact channel — nor the money or the card
the corpus leaves unstated, nor the two controls. Repeating a superseded value would move
the balance the authored corpus was built on (the old address is named nine times to the
new one's three, and last) and change what a trapped answer means. A distractor may name a
fact's *topic* without its value — which subscription tiers exist, where an invoice
appears, whether someone has to be in for a parcel — and several do on purpose, because
topic words are what a retriever matches on, and those tickets compete with the authored
turns for the twelve slots a retrieval arm has. Every generated turn is unique text,
because `Memvara.add()` returns the existing episode for a repeat and the memvara arms
would otherwise hold fewer turns than `full_transcript`. Tickets land on days with no
authored turn, at 06:00 or 22:00, inside the authored window, so the cutoffs slice them
as they slice the authored turns and the corpus keeps its shape.
`tests/test_demo_scenario.py` pins all of this against a hand-written list of the
forbidden strings rather than against anything the module exports.

At scale 10 the size table is this, and like the table above it is deterministic:

```
  arm                 mean chars  max chars  mean ~tokens  items used / turns seen
  ------------------  ----------  ---------  ------------  -----------------------
  none                         0          0             0              0.0 / 607.5
  full_transcript          92053      96897         23013            607.5 / 607.5
  naive_rag                 1891       2838           473             12.0 / 607.5
  memvara                   1596       2424           399             12.0 / 607.5
  memvara_structured        1530       2204           383             12.0 / 607.5
```

The transcript arm grows 9.4× (9,803 to 92,053 characters; the authored turns are longer
than the generated ones, so ten times the turns is not quite ten times the text). The
three retrieval arms stay under `MAX_CONTEXT_CHARS` by construction and come out a little
*shorter*, because the twelve slots now fill with generated turns that are shorter than
the authored ones. Whether they still surface the evidence among ten times more turns is
what the hosted run at this scale measures; the stub cannot say.

`memvara_structured` deserves one more sentence. Its claims come from the desk's own
fields (`SUPPORT_FACTS`), not from the transcript, so at every scale its claim tier is
identical and only its episode tail faces more competition. That is the product's thesis
stated as an experiment: structured facts plus retrieval should hold as the history grows,
while an arm that has only the transcript to work from has more to lose.

---

## The memvara arms against the hosted service

By default both memvara arms use a store inside the process. `--memory hosted` points them
at a memvara-cloud project instead, through the same client a customer uses
(`memvara.remote`), so a run can measure the hosted service rather than the library alone.
Every other arm is unchanged: `none`, `full_transcript` and `naive_rag` use no store at all.

```bash
memvara login --credentials ~/.memvara/demo-credentials.json   # choose the demo's project
PYTHONPATH=. python3 demo/harness.py --reader stub --memory hosted \
    --hosted-credentials ~/.memvara/demo-credentials.json --hosted-run-id 2026-09-16a
```

**It writes to a project of its own, and refuses to do otherwise.** A run writes a few
thousand turns, which no code here can take back, so `--hosted-credentials` refuses the
default credentials file, any file holding the same key as it or as `MEMVARA_API_KEY`, and
any file for the same project. Make a separate project in the console and sign in to it
with `memvara login --credentials PATH`.

Three things differ from the local arms, and the report prints them above its tables
rather than leaving them to be noticed:

* **The schema cannot be sent.** A hosted project's predicate vocabulary is the
  deployment's, so `plan` and the two addresses are not single-valued there. The arm
  closes a single-valued slot itself — `forget(close="ended")` at the instant the new value
  begins — before writing the new one, which is the explicit form of what the declared
  cardinality does locally. `tests/test_demo_hosted.py` checks every slot at every question
  instant against the local structured arm, which has the schema.
* **`plan` is filed under `goal`.** The built-in vocabulary resolves `plan` as an alias of
  `goal`, so that is where the plan history lands. Reads by slot resolve the alias and the
  prompt carries each claim's own sentence, so the reader sees the same words; anything
  keyed on the predicate name does not. The report names the fold.
* **Dated questions are read differently.** `POST /v1/recall` has no time axis, so the four
  questions carrying `about` are read with `search(valid_at=)` and rendered by the
  library's own recall renderer — byte-identical to local `recall(valid_at=)` on the same
  store, which a test pins. The report counts how many contexts were read that way.

Extraction and the episode cap are the deployment's: it runs its own extractor over the
turns the `memvara` arm writes, on its own schedule, and its own limit on how many turns a
read returns, where the local arm sets `read_max_episodes=k`. So each context records how
many claims its scope held when it was read, and the report prints the range.

**Scopes, and repeating a run.** One scope per run, arm, corpus size and question instant —
eighteen of the twenty questions share an `asked_at`, so a run writes three scopes per arm
rather than twenty. `Manifest` records each scope as `started` then `complete` in
`demo/runs/<run id>.hosted.jsonl`. Reusing `--hosted-run-id` reads the finished scopes
without writing them again, which is how the noise-floor repeat measures the reader twice
over the same stored contexts. A scope left `started` by a run that died is refused, because
replaying the fact table into it would close values at instants they were never closed at;
start again with a new run id.

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
  memvara                   2074       2489           519              12.0 / 60.8
  memvara_structured        1772       2241           443              12.0 / 60.8
```

`~tokens` is `chars // 4`, an estimate and not a tokenizer — `CHARS_PER_TOKEN` says so.
The `memvara_structured` row grew from 1,721 to 1,772 characters when the arm moved to
`recall(valid_at=)`, whose dated header is part of what it renders; the agent run below
was made at the earlier size.
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
  earns is the size column: **5.6× fewer tokens for 95%** (2,451 → 440; the `memvara` arm
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
