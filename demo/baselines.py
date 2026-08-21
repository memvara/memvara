"""The five context-building arms a support-agent answer-quality run is compared across.

    from demo import baselines            # or: sys.path.insert(0, "demo")

A number with no control arm is not a measurement. "87% correct with memvara" is
uninterpretable on its own: the questions might be answerable from general knowledge,
or the whole transcript might fit in a prompt and make a memory layer pointless. These
five arms are what make the number mean something.

    none               no context at all — the floor. How many questions fall out of
                       priors, guessing, or the wording of the question itself.
    full_transcript    every turn, chronologically — the honest competitor. If it wins,
                       the memory layer is not earning its place *at this corpus size*,
                       and `Context.chars` is the whole argument for what happens when
                       the corpus stops being this size.
    naive_rag          top-k cosine similarity over raw turns. No supersession, no
                       recency weighting, no time axes. This is the arm that isolates
                       *bitemporal reasoning*, because it runs the same embedder memvara
                       runs — so a difference between it and `memvara` cannot be
                       explained by vector quality.
    memvara            the library on its shipped defaults: a transcript dropped in with
                       no model and no schema. What a first evaluation actually does.
    memvara_structured the library integrated — a declared predicate schema and facts
                       written from the ticketing system's own fields, which is what a
                       support deployment has and a transcript replay does not.

Each arm builds the **context**, never the answer. What to do with it is
`demo/harness.py`'s problem.

## Why the product has two arms, and why neither may be deleted

`memvara` and `memvara_structured` are the same library, one variable apart, and the gap
between them is a finding rather than a redundancy.

On this corpus `memvara` produces **six claims from sixty-four turns**, and it used to
produce none. The rule extractor's vocabulary is mostly first-person declaratives ("I live
in X", "my name is X") and a support history is not written that way; what it now also
reads is a contact directive, a postal address and a bare phone number, which is enough
for two slots to supersede on the world clock. Four of the six facts the corpus was built
around — the plan, the serial, the mobile correction, the billing address — are still in
sentence forms no rule can read, so with no `llm=` this arm sees a fraction of the
history and cannot test the whole of what the comparison exists to test. That is the
documented behaviour of the shipped defaults (`README.md` says so, and `bench/locomo.py`
says the same about LOCOMO), so it is worth measuring: it is what somebody evaluating the
library over a weekend will actually see.

`memvara_structured` is the other real configuration, and the one the product is *for*. A
support integration does not ask a model to read prose back out of its own database; it
writes structured facts from the fields its ticketing system already has, with the
predicate schema declared up front. That needs no API key, exercises the whole bitemporal
path, and is a few dozen lines — see `SUPPORT_PREDICATES` and `SUPPORT_FACTS`, which are
the entire integration and are counted in
`test_the_structured_integration_is_small_enough_that_properly_is_a_fair_word`.

Reporting only the first would understate the library; reporting only the second would
hide what an evaluator meets first. The pair is the honest answer.

## The cutoff, and why it applies to every arm

`Question.asked_at` may be earlier than the end of the conversation. Every arm here sees
only the turns at or before it (`visible_turns`). Applying it to `memvara` alone would
be the more obvious reading of "respect asked_at", and it would be wrong: the other arms
would answer a July question with September's turns, which does not make them stronger
competitors, it makes the comparison meaningless. The arms must differ in *what they do
with the history*, not in *how much of it they are allowed to see*.

Inside the `memvara` arm the cutoff is enforced at ingest — only visible turns are
written — rather than by a `known_at=` read. That is not a shortcut, it is the only
correct option here, and the reason is worth stating because it looks like one:
`Memvara.add()` stamps a claim's `valid_from` from the turn's `ts` but its `recorded_at`
from the wall clock at write time. Replaying an archived transcript today therefore
produces claims all recorded *today*, so a `known_at=<July>` read would correctly hide
every one of them and return an empty context for every question. Truncating the ingest
reproduces the belief state the store actually held in July, which is the thing the
question is asking about. See `test_the_memvara_arm_cannot_see_a_turn_recorded_after_the
_question_was_asked` for the proof that it bites.

## The embedder is pinned, deliberately

`default_embedder()` returns a sentence-transformers model whenever that package is
importable, and it currently is in this environment. Left alone, `naive_rag`, `memvara`
and `memvara_structured` would silently be a different experiment on a machine with the
extra installed than on one without, and the difference would show up as a quality delta
with no code change behind it. All three embedding arms take `HashingEmbedder`
explicitly, at the same dimension, so the comparison is about time handling and nothing
else. It also means all three are running a *lexical* approximation rather than a
semantic model: absolute numbers here are pessimistic for every one of them, and the
deltas between them are the part that transfers.
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol, Sequence

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "bench")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evalkit import clip  # noqa: E402

from memvara import (  # noqa: E402
    LLM,
    Claim,
    DegradedExtractionWarning,
    EpisodeResult,
    HashingEmbedder,
    Memvara,
    NullLLM,
    PredicateRegistry,
    PredicateSpec,
    Retrieved,
)
from memvara.schema import BUILTIN_PREDICATES, Cardinality, Volatility  # noqa: E402


# --- the scenario contract, structurally ----------------------------------------
#
# `demo/scenario.py` owns these two shapes. They are restated here as protocols rather
# than imported so this module has no import-time dependency on a file another workstream
# is writing, and so a test can drive an arm with three lines of fixture instead of the
# whole support history. Read-only properties rather than plain attributes because the
# real ones are frozen dataclasses and this module never writes to them.


class Turn(Protocol):
    @property
    def at(self) -> datetime: ...
    @property
    def role(self) -> str: ...
    @property
    def text(self) -> str: ...


class Question(Protocol):
    @property
    def id(self) -> str: ...
    @property
    def asked_at(self) -> datetime: ...
    @property
    def text(self) -> str: ...
    @property
    def gold(self) -> str: ...
    @property
    def trap(self) -> str | None: ...
    @property
    def kind(self) -> str: ...
    #: The **valid-time** instant the question asks about, or `None` when it asks about
    #: the moment it is put. Only `memvara_structured` reads it, as `search(valid_at=)`;
    #: no other arm has an axis to spend it on. `Question.closure` is deliberately not
    #: restated here — it exists on the real dataclass and nothing in `demo/` reads it,
    #: and a protocol field nobody uses is a field every fixture has to invent.
    @property
    def about(self) -> datetime | None: ...


# --- shared configuration -------------------------------------------------------

#: Retrieval slots, identical for `naive_rag` and `memvara`. Same budget or the
#: comparison is between two budgets rather than between two ways of spending one.
DEFAULT_K = 12

#: Character cap on the two *retrieval* arms, and deliberately not on `full_transcript`.
#: The cap is what stops a retrieval arm quietly becoming a long-context arm; the
#: transcript arm is the long-context arm, which is the entire reason it is here.
#: Same argument as `evalkit.RetrievalBudget`, same number.
MAX_CONTEXT_CHARS = 4000

#: The vector width both embedding arms use. `HashingEmbedder`'s identity includes its
#: dimension, so this being one constant rather than two literals is what makes
#: "the same embedder" true rather than approximately true.
EMBED_DIM = 512

#: Characters per token, for the size estimate. It is the usual rule of thumb for English
#: prose and it is an *estimate*: `Context.approx_tokens` is not a tokenizer and must not
#: be quoted as one. Characters are the number to trust in this module.
CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class Context:
    """What one arm would put in front of a reader for one question.

    Not the answer, and not a score. `text` is verbatim what goes into the prompt.
    """

    arm: str
    text: str
    #: Turns at or before `Question.asked_at`. The same for every arm on a given
    #: question, which is the point — see the module docstring.
    turns_visible: int
    #: Entries this arm actually put in the context — one per line of `text`. Turns, for
    #: the three turn-based arms; claims plus turns for `memvara`, which is why it is not
    #: called `turns_used`. Counting ingested turns here instead would make the memory
    #: arm's row of the size table read as if it had put sixty turns in a prompt that
    #: holds twelve lines. The gap between this and `turns_visible` for the retrieval arms
    #: is the compression the memory layer is being paid for.
    items_used: int
    #: True when the `memvara` arm ran with no LLM, so only the deterministic rule
    #: extractor contributed claims. Carried on the result rather than warned about,
    #: because the arm builds one store per question and a `warnings.warn` per store is
    #: noise that hides the finding. The harness reports it once, loudly.
    degraded: bool = False

    @property
    def chars(self) -> int:
        return len(self.text)

    @property
    def approx_tokens(self) -> int:
        """Characters divided by four. An estimate — see `CHARS_PER_TOKEN`."""
        return len(self.text) // CHARS_PER_TOKEN


#: What every arm is. Options are keyword-only with defaults so a bare function satisfies
#: this, and `functools.partial` configures one without a wrapper class.
Arm = Callable[[Question, Sequence[Turn]], Context]


def visible_turns(question: Question, turns: Sequence[Turn]) -> list[Turn]:
    """The turns that had happened when the question was asked, chronologically.

    Inclusive of `asked_at` itself: a question asked in the same second as a turn is
    asked *after* it, and excluding it would make the boundary depend on clock
    resolution. Sorted here rather than trusted, because `Context.items_used` is only
    meaningful if the order is this function's to define.
    """
    return sorted((t for t in turns if t.at <= question.asked_at), key=lambda t: t.at)


def count_entries(text: str) -> int:
    """Entries in a rendered context: lines beginning with the bullet.

    Counted from the rendered text rather than from the list that went in, and counted
    *after* the character cap, because the cap is what decides how much actually reaches
    the prompt. Both `render_turns` and `recall()` bullet their entries and neither
    bullets its headers, so one rule serves both and the two retrieval arms are measured
    the same way — which they have to be, since the size table sets them side by side.
    """
    return sum(1 for line in text.splitlines() if line.startswith("- "))


def render_turns(turns: Sequence[Turn]) -> str:
    """One turn per line, dated.

    Shared by `full_transcript` and `naive_rag` so those two arms are byte-identical in
    *shape* and differ only in which turns they contain and in what order — which is as
    much blinding as two arms can honestly be given (see `demo/harness.py`).

    The date is included rather than stripped. Withholding it would hand memvara the win
    by disabling the competition: a reader that cannot see when something was said cannot
    possibly tell a superseded value from a current one, and beating an opponent whose
    hands are tied proves nothing about the memory layer.
    """
    return "\n".join(
        f"- [{t.at:%Y-%m-%d}] {t.role}: {' '.join(t.text.split())}" for t in turns
    )


# --- arm 1: the floor -----------------------------------------------------------


def none(question: Question, turns: Sequence[Turn]) -> Context:
    """No context at all.

    The number this arm produces is the one every other number is measured against. A
    question this arm gets right was answerable from priors, from guessing between two
    obvious options, or from the question's own wording — and it is worth no credit to
    any memory system.
    """
    return Context(arm="none", text="", turns_visible=len(visible_turns(question, turns)),
                   items_used=0)


# --- arm 2: the honest competitor -----------------------------------------------


def full_transcript(question: Question, turns: Sequence[Turn]) -> Context:
    """Every visible turn, chronologically, in the prompt.

    This is what everyone tries first, and at a corpus this size it is a serious
    competitor rather than a straw man — which is exactly why it is here and why it is
    given dates. The argument for a memory layer is not that this loses; it is
    `Context.chars`, and what that number does when the transcript is a year long
    instead of a month.

    Uncapped on purpose. Capping it would make it a worse version of `naive_rag` and
    would hide the size, which is the finding.
    """
    seen = visible_turns(question, turns)
    return Context(arm="full_transcript", text=render_turns(seen),
                   turns_visible=len(seen), items_used=len(seen))


# --- arm 3: similarity, and nothing else ----------------------------------------


def naive_rag(question: Question, turns: Sequence[Turn], *, k: int = DEFAULT_K,
              max_chars: int = MAX_CONTEXT_CHARS, dim: int = EMBED_DIM) -> Context:
    """Top-k cosine similarity over raw turns. No supersession, no recency, no clocks.

    Ranked order, not chronological order — that is what a vector store returns and
    re-sorting it by date would be handing this arm a temporal signal it does not have,
    which is the very signal under test.

    `HashingEmbedder` is pinned rather than taken from `default_embedder()`; the module
    docstring says why. Its rows come back L2-normalized, so the dot product below *is*
    cosine similarity.

    Ties break on the turn's timestamp and then its text, so a re-run of the same corpus
    produces the same context. An eval whose context depends on dict ordering cannot be
    used for a regression test.
    """
    seen = visible_turns(question, turns)
    if not seen:
        return Context(arm="naive_rag", text="", turns_visible=0, items_used=0)
    embedder = HashingEmbedder(dim=dim)
    matrix = embedder.encode([t.text for t in seen])
    query = embedder.encode([question.text])[0]
    scores = matrix @ query
    order = sorted(range(len(seen)),
                   key=lambda i: (-float(scores[i]), seen[i].at, seen[i].text))
    text = clip(render_turns([seen[i] for i in order[:k]]), max_chars)
    return Context(arm="naive_rag", text=text, turns_visible=len(seen),
                   items_used=count_entries(text))


# --- arm 4: the product ---------------------------------------------------------


def build_memory(turns: Sequence[Turn], *, llm: LLM | None = None,
                 dim: int = EMBED_DIM,
                 max_episodes: int = DEFAULT_K,
                 registry: PredicateRegistry | None = None) -> tuple[Memvara, bool]:
    """A store holding exactly `turns`, ingested with each turn's own timestamp.

    Returns the store and whether extraction ran degraded (no LLM, so only the rule
    extractor contributed). Separate from `memvara()` so a test can inspect the store the
    arm actually reads from, rather than inferring it from rendered text.

    `ts=turn.at` is load-bearing: it is what gives each claim a `valid_from` on the day
    the customer said it, which is what lets a supersession close the old value's valid
    time at the point the new one begins instead of at the point of the replay.

    `read_max_episodes` is raised from its default of 3 to `k`, and that is a fairness
    correction rather than tuning. `k` is the *total* slot count, but only
    `HybridRetriever.max_episodes` of those slots may go to raw turns — so at the default,
    `naive_rag` was returning twelve turns while this arm was capped at three, and the two
    arms were not on the same budget at all. The library's default is right for its own
    purpose (a prompt should not fill up with unverified chatter when there are facts to
    put there); it is wrong for a controlled comparison, where the two retrieval arms have
    to be allowed to spend the same number of slots however each sees fit.

    `registry` is `None` for the `memvara` arm — the shipped default schema, which is
    what a first evaluation runs — and `SUPPORT_REGISTRY` for `memvara_structured`. It
    changes nothing about ingest: the rule extractor's patterns are fixed, and declaring
    `plan` as a predicate does not teach it to recognise one. It changes what happens to
    the facts written *afterwards*, which is the whole of the difference between the two
    arms.
    """
    degraded = llm is None
    with warnings.catch_warnings():
        # One store is built per question, so this would otherwise fire once per question
        # and bury the run's output. It is not swallowed — `Context.degraded` carries it
        # out and the harness prints it once.
        warnings.simplefilter("ignore", DegradedExtractionWarning)
        mem = Memvara(embedder=HashingEmbedder(dim=dim), llm=llm or NullLLM(),
                      user="customer", read_max_episodes=max_episodes,
                      registry=registry)
    for turn in turns:
        mem.add(turn.text, role=turn.role, ts=turn.at)
    return mem, degraded


def memvara(question: Question, turns: Sequence[Turn], *, k: int = DEFAULT_K,
            max_chars: int = MAX_CONTEXT_CHARS, llm: LLM | None = None,
            dim: int = EMBED_DIM) -> Context:
    """memvara through its public API, rendered by `recall()`.

    `recall()` rather than `search()` because `recall()` is what an integration actually
    drops into a prompt — the headers, the flattening, the claims-before-episodes
    ordering are all part of what is being measured, and re-rendering `search()` results
    here would be measuring a formatter this repository does not ship.

    `include_episodes=True` because the alternative is not a fair reading of the product.
    With no `llm=`, only the deterministic rule extractor runs, and a support transcript
    is largely made of facts those rules do not recognise — a plan name, an order number,
    a promised callback. Those turns are still stored and still retrievable, and the
    episode tail is the documented way to reach them. Running this arm claims-only with
    no LLM would be measuring the rule set's vocabulary, not the memory layer.

    Note what `recall()` does *not* take: `as_of`, `known_at`, `states`. That is a
    deliberate refusal in the library (resurrecting retired claims into a live prompt is
    an un-delete reachable by anyone who can influence a parameter), and it is why the
    `asked_at` cutoff here is applied at ingest instead.

    **Read `Context.degraded` before quoting this arm's score.** On a corpus written as
    ordinary support prose the rule extractor matches a small fraction of it — its
    vocabulary is mostly first-person declaratives ("I live in X", "my name is X") plus a
    contact directive, an address and a bare phone number, and a support history is mostly
    none of those. With no `llm=` the claim tier can therefore be nearly *empty*: on this
    corpus it holds six claims covering two of the six facts that move, so four of them
    get no supersession and no valid-time closing at all. That is a real deployment
    configuration and it is worth measuring, but it is a partial view of the thing this
    comparison exists to test.
    """
    seen = visible_turns(question, turns)
    mem, degraded = build_memory(seen, llm=llm, dim=dim, max_episodes=k)
    text = clip(mem.recall(question.text, k=k, include_episodes=True), max_chars)
    return Context(arm="memvara", text=text, turns_visible=len(seen),
                   items_used=count_entries(text), degraded=degraded)


# --- arm 5: the product, integrated ---------------------------------------------
#
# Everything from here to `ARMS` is the integration, and it is deliberately all in one
# place so it can be read — and counted — as the thing a deployment would have to write.


#: The predicate schema, declared up front. **This step is required, not decoration.**
#: `PredicateRegistry` defaults an unknown predicate to `MANY`, so without it every value
#: accumulates: `Pro` and `Home` both stay live, nothing supersedes anything, and
#: `get_all()` answers "which plan?" with two plans. That default is deliberate and
#: documented — "errors should fall on the recoverable side", since keeping two facts that
#: conflict degrades ranking while dropping one that does not destroys information — but
#: it means declaring cardinality is the price of the contradiction engine, and this
#: constant is that price in full. `test_a_predicate_left_at_the_default_cardinality_
#: stops_superseding_silently` injects the omission and watches the slot grow two answers.
#:
#: Cardinality per predicate, with the reason each way:
SUPPORT_PREDICATES: tuple[PredicateSpec, ...] = (
    # ONE. An account is on exactly one plan at a time — the upgrade on 3 March and the
    # downgrade on 19 June are replacements, not additions, and the whole `q_plan_current`
    # / `q_plan_mid_march` pair depends on the second closing the first.
    PredicateSpec("plan", Cardinality.ONE, Volatility.SLOW),
    # ONE. One address a parcel goes to. Note that 3 March's "ship them to Coldharbour
    # Road, not the Yard" is an instruction about one order, not a second standing
    # address, so it is not a fact and is not written.
    PredicateSpec("delivery_address", Cardinality.ONE, Volatility.SLOW),
    # ONE, and a *separate predicate* from delivery rather than a second value of it.
    # That separation is what makes the ten-day window expressible at all: on 30 July the
    # account has two addresses with two independent valid intervals, and a single
    # `address` slot — of either cardinality — could not hold that. ONE would have made
    # the move overwrite billing too; MANY would have left both addresses live with
    # nothing to say which was which.
    PredicateSpec("billing_address", Cardinality.ONE, Volatility.SLOW),
    # MANY, and the one genuinely multi-valued slot here. On 6 February the customer names
    # two acceptable channels in one breath — "Phone or a text, either one, but not
    # email" — and the authored answer for April is *both* of them. A ONE-cardinality
    # field would have to discard one, and would then disagree with the answer key by
    # construction. The cost is real and is paid on 22 June: a MANY slot supersedes
    # nothing on its own, so the reversal to email has to close the slot explicitly — the
    # one `Write(mode="replace")` in the table.
    PredicateSpec("contact_preference", Cardinality.MANY, Volatility.SLOW),
    # ONE. The desk's "best number for you" field holds one number; that it held the wrong
    # one for a week is a transaction-time event, not a second value.
    PredicateSpec("mobile", Cardinality.ONE, Volatility.SLOW),
    # ONE *per subject*, which is the whole modelling decision. A unit has exactly one
    # serial, so `main_unit` supersedes correctly — while the node that died on 9 April is
    # a different subject and its serial never competes with the main unit's. Written as
    # `("account", "serial")` it would have had to be MANY, and the correction on 9 April
    # would have had nothing to close. STATIC because a serial does not change: it is the
    # premise of `q_serial_correction` that this one never did.
    PredicateSpec("serial", Cardinality.ONE, Volatility.STATIC),
    # ONE, STATIC. A control — stated once on 19 January and never touched.
    PredicateSpec("account_name", Cardinality.ONE, Volatility.STATIC),
    # ONE, STATIC. The other control, and the transcript argues the volatility itself:
    # "Billing is the 14th of the month, every month, and that date doesn't move."
    PredicateSpec("billing_day", Cardinality.ONE, Volatility.STATIC),
)

#: The declared schema on top of the library's own. `BUILTIN_PREDICATES` is kept rather
#: than replaced so the personal-assistant vocabulary the rule extractor shares with the
#: `memvara` arm is still present, and the two arms differ by an addition rather than by a
#: substitution.
SUPPORT_REGISTRY = PredicateRegistry(BUILTIN_PREDICATES + SUPPORT_PREDICATES)


@dataclass(frozen=True)
class Write:
    """One thing the support desk learned, and what it did to what was already on record.

    The two instants are the point of the whole exercise and they are *not* the same
    field. `at` is transaction time — when the desk found out, and the instant the
    `asked_at` cutoff is applied to. `valid_from` is world time — when the fact started
    being true, which for a correction is *before* anyone knew it.

    `mode` names which of the three things the desk did, and it is the ended/retired
    distinction spelled as a verb:

    ``"assert"``   a new value. For a `ONE` predicate the reconciler closes whatever it
                   displaces on **valid** time (`ended`): that value was true and stopped
                   being true. This is the plan changes, the two address moves, and every
                   first statement of a fact.
    ``"correct"``  the value on record was never true. Closes **transaction** time
                   (`retired`) on it, at `at`, and leaves its valid interval exactly as
                   written, because a correction witnessed no world event. The mobile and
                   the serial, and nothing else.
    ``"replace"``  close every live value in the slot on valid time, then assert. Only
                   needed for a `MANY` predicate, where cardinality supersedes nothing —
                   see `contact_preference` in `SUPPORT_PREDICATES`.

    An arm that used one of these everywhere would produce a number that looked fine and
    meant nothing: retire the address moves and "what did we ship to in April" goes blank,
    end the mobile transposition and the store asserts that a number nobody ever had
    stopped being theirs on 13 February.
    """

    #: Transaction time: when the desk found out. What the `asked_at` cutoff applies to.
    at: datetime
    subject: str
    predicate: str
    obj: str
    #: The sentence that gets embedded and BM25-indexed, overriding `remember()`'s default
    #: "<subject> <predicate> <object>" rendering. Not cosmetic: measured on this corpus,
    #: the default rendering loses "what address should it ship to?" to a raw turn
    #: containing the word "node", and the correct address never reaches the prompt at all.
    #: `remember()`'s own docstring says this ("It matters far more than it looks"), and it
    #: is one string per fact, which is why it is a field here rather than an option.
    text: str
    #: World time: when the fact started being true. Defaults to `at`, because for anything
    #: the desk learned as it happened the two coincide — so the rows that spell this out
    #: separately are exactly the three where the clocks genuinely come apart, and they are
    #: visible at a glance rather than buried in a column of identical timestamps.
    since: datetime | None = None
    mode: str = "assert"

    @property
    def valid_from(self) -> datetime:
        return self.since if self.since is not None else self.at


#: The values that move, written once. Two spellings of one address is how a fact table
#: develops an answer key that disagrees with itself, and
#: `test_every_structured_fact_is_grounded_in_a_turn_of_the_real_transcript` checks each
#: one against the transcript rather than trusting it.
_OLD_ADDRESS = "41 Coldharbour Road, Lewes, BN7 2GT"
_NEW_ADDRESS = "Bramble Cottage, Ditchling Road, Westmeston, BN6 8XA"
_WRONG_SERIAL, _SERIAL = "HX2-4419-B", "HX7-8802-D"
_WRONG_MOBILE, _MOBILE = "07700 900 118", "07700 900 811"


def _utc(month: int, day: int, hour: int, minute: int) -> datetime:
    """A turn's instant in 2026, spelled short enough that the fact table stays readable.

    Every write lands on a timestamp a turn actually carries — the desk records what it is
    told, when it is told — which is asserted rather than assumed.
    """
    return datetime(2026, month, day, hour, minute, tzinfo=timezone.utc)


#: What the desk knew, in the order it learned it. Every row is justified by a turn in
#: `demo/scenario.py`, named in the comment above it; `demo/README.md`'s table of what
#: moved and why is the source of truth for the two closure columns.
#:
#: This table plus `SUPPORT_PREDICATES` is the entire integration.
SUPPORT_FACTS: tuple[Write, ...] = (
    # --- 19 January: the account opens -------------------------------------------
    Write(_utc(1, 19, 9, 22), "account", "account_name", "Wray & Daughter Joinery",
          "The account is registered in the name Wray & Daughter Joinery."),
    Write(_utc(1, 19, 9, 22), "account", "delivery_address", _OLD_ADDRESS,
          f"Deliveries and replacement hardware ship to {_OLD_ADDRESS}."),
    Write(_utc(1, 19, 9, 26), "account", "plan", "Home",
          "The account is on the Home plan."),
    # --- 6 February: the two records that turn out to be wrong --------------------
    # Nothing here knows they are wrong. A desk records what it is told, and finds out
    # later; that gap is the whole reason transaction time is a separate axis.
    Write(_utc(2, 6, 16, 14), "main_unit", "serial", _WRONG_SERIAL,
          f"The main unit's serial number is {_WRONG_SERIAL}."),
    Write(_utc(2, 6, 16, 21), "account", "mobile", _WRONG_MOBILE,
          f"The mobile number on file is {_WRONG_MOBILE}."),
    # "Phone or a text, either one, but not email" — two values in one breath, which is
    # why `contact_preference` is the MANY predicate and why both are written.
    Write(_utc(2, 6, 16, 29), "account", "contact_preference", "phone",
          "The customer asked to be contacted by phone."),
    Write(_utc(2, 6, 16, 29), "account", "contact_preference", "text",
          "The customer asked to be contacted by text message."),
    # --- 13 February: correction one. RETIRED ------------------------------------
    # "it's been wrong since the moment I gave it to you." `since` is 6 February, not 13
    # February: this number has been theirs the whole time, and dating it from the
    # correction would assert that they changed numbers — the thing the customer is
    # explicitly denying, and the thing `mode="correct"` exists to avoid recording.
    Write(_utc(2, 13, 8, 47), "account", "mobile", _MOBILE,
          f"The mobile number on file is {_MOBILE}.",
          since=_utc(2, 6, 16, 21), mode="correct"),
    # --- 3 March: the upgrade. ENDED ---------------------------------------------
    Write(_utc(3, 3, 11, 24), "account", "plan", "Pro",
          "The account is on the Pro plan."),
    # --- 9 April: a node dies, and correction two --------------------------------
    # A different subject, so its serial is a fact of its own and competes with nothing.
    Write(_utc(4, 9, 13, 38), "dead_node", "serial", "HX7-6120-N",
          "The failed node's serial number is HX7-6120-N."),
    # RETIRED, and the warranty is why the distinction is not pedantry: "If I'd logged it
    # as a swap your warranty would have restarted today." `since` goes back to 6 February
    # for the same reason as the mobile — this was always the unit's number.
    Write(_utc(4, 9, 13, 47), "main_unit", "serial", _SERIAL,
          f"The main unit's serial number is {_SERIAL}.",
          since=_utc(2, 6, 16, 14), mode="correct"),
    # --- 21 May: billing ---------------------------------------------------------
    Write(_utc(5, 21, 10, 9), "account", "billing_day", "14",
          "The account is billed on the 14th of the month."),
    # The third and only benign gap between the clocks: recorded on 21 May, true since the
    # account opened. Grounded in 5 August 9:38 — "yours matched each other until you
    # moved" — and it is *not* a correction: nothing was ever wrong, the desk simply wrote
    # it down late. Both axes exist precisely so that a backfill and a retraction do not
    # have to share a spelling.
    Write(_utc(5, 21, 10, 20), "account", "billing_address", _OLD_ADDRESS,
          f"Paper invoices are sent to the billing address {_OLD_ADDRESS}.",
          since=_utc(1, 19, 9, 22)),
    # --- 19 June: the downgrade. ENDED -------------------------------------------
    Write(_utc(6, 19, 16, 2), "account", "plan", "Home",
          "The account is on the Home plan."),
    # --- 22 June: the preference reverses. ENDED ---------------------------------
    # "I know I asked for calls back in February, and that was right at the time." Nobody
    # was wrong; the world changed. `mode="replace"` because the slot is MANY and holds
    # two values, so there is no cardinality rule to close them.
    Write(_utc(6, 22, 8, 4), "account", "contact_preference", "email",
          "The customer asked to be contacted by email.", mode="replace"),
    # --- 26 July: the move. Delivery only. ENDED ---------------------------------
    Write(_utc(7, 26, 18, 31), "account", "delivery_address", _NEW_ADDRESS,
          f"Deliveries and replacement hardware ship to {_NEW_ADDRESS}."),
    # --- 5 August: billing follows, ten days later. ENDED ------------------------
    Write(_utc(8, 5, 9, 24), "account", "billing_address", _NEW_ADDRESS,
          f"Paper invoices are sent to the billing address {_NEW_ADDRESS}."),
)

#: Recorded on every structured claim, so `why()` can tell a fact this integration
#: asserted from one an extractor guessed at.
SUPPORT_EXTRACTOR = "demo/support-desk"


def visible_facts(question: Question, facts: Sequence[Write]) -> list[Write]:
    """The writes the desk had made when the question was asked, in order.

    The claim-tier twin of `visible_turns`, and it exists for the same reason and is
    applied at the same boundary: a fact recorded after the question was put must not be
    visible to it. Inclusive of `asked_at` itself, on `Write.at` — transaction time, never
    `valid_from`, because the question is about what the desk *knew*, and a correction
    backdated to February was not known in February.

    Enforced at write time rather than by a `known_at=` read for exactly the reason the
    module docstring gives for the `memvara` arm: the retirement instants here are honest
    (`supersede(at=...)` backdates them), but the *episodes* replayed alongside are
    recorded at wall-clock now, so a `known_at=<July>` read would empty the episode tail
    and make the two memvara arms differ in what they were allowed to see. Truncating the
    input reproduces the state the store actually held. See
    `test_the_structured_arm_cannot_see_a_fact_recorded_after_the_question_was_asked`.
    """
    return sorted((f for f in facts if f.at <= question.asked_at), key=lambda f: f.at)


def _live_in_slot(mem: Memvara, subject: str, predicate: str) -> list[Claim]:
    """Claims currently believed and in force in one slot, as `history()` reports them.

    Slot-addressed rather than a filter over `get_all()`, because the slot is the unit the
    library reconciles on and `history()` is the read that names it.
    """
    return [c for c in mem.history(subject, predicate) if c.state == "live"]


def apply_facts(mem: Memvara, facts: Sequence[Write]) -> None:
    """Write the desk's structured record into `mem`, in order.

    Three verbs and one branch each; the whole of what an integration does. See `Write`
    for what the three mean and why an arm may not collapse them into one.
    """
    for fact in facts:
        if fact.mode == "replace":
            # A MANY slot: cardinality closes nothing, so the standing values are closed
            # explicitly, on valid time, at the instant the new one begins. `forget`
            # passes no successor, which leaves any `invalidated_by` already on those rows
            # intact rather than blanking it.
            mem.forget(fact.subject, fact.predicate, at=fact.valid_from, close="ended")
        if fact.mode == "correct":
            # `supersede` rather than `remember(close="retired")` for one reason, and it
            # is the reason this method takes `at=`: `remember` hands the reconciler the
            # wall clock, so the belief change would be stamped *today* — a replay would
            # record that we stopped believing the transposed number this afternoon rather
            # than on 13 February, and every `known_at=` audit over the correction would
            # be wrong. Retiring nothing at all would be worse; retiring it on the wrong
            # day is the quiet failure.
            standing = _live_in_slot(mem, fact.subject, fact.predicate)
            if len(standing) != 1:
                # A correction with nothing to correct is a broken fact table, not a write
                # to do anyway. Refused loudly: asserting the new value regardless would
                # leave the record looking corrected when nothing had been retired.
                raise ValueError(
                    f"correction of {fact.subject}.{fact.predicate} at {fact.at:%Y-%m-%d} "
                    f"found {len(standing)} standing values, expected exactly 1"
                )
            mem.supersede(
                standing[0].id,
                Claim(subject=fact.subject, predicate=fact.predicate, object=fact.obj,
                      valid_from=fact.valid_from, recorded_at=fact.at, text=fact.text,
                      extractor=SUPPORT_EXTRACTOR),
                at=fact.at, close="retired")
            continue
        mem.remember(fact.subject, fact.predicate, fact.obj, valid_from=fact.valid_from,
                     recorded_at=fact.at, text=fact.text, extractor=SUPPORT_EXTRACTOR,
                     close="ended")


def slot_values(mem: Memvara, subject: str, predicate: str, *,
                valid_at: datetime | None = None) -> list[str]:
    """Every value live in one slot, at `valid_at` or at the present belief.

    The read a support desk's own UI would make, and the one the direct assertions in
    `tests/test_demo.py` are written against — checking what the library returns rather
    than what a model said about what the library returned. Sorted, because a slot with
    more than one value has no meaningful order and an assertion should not depend on one.

    Deliberately no `known_at=`: the belief clock stays at now, and the "what did we know
    then" question is answered by `visible_facts` truncating the input instead. Rewinding
    belief as well would hide every claim whose *episode* was replayed today.
    """
    return sorted(c.object for c in mem.get_all(valid_at=valid_at)
                  if c.subject == subject and c.predicate == predicate)


def render_recall(results: Sequence[Retrieved]) -> str:
    """`recall()`'s exact output, from results this arm had to fetch itself.

    A reimplementation of eight lines of `Memvara.recall`, and it is here under protest:
    `recall()` takes no `valid_at=`, deliberately — time travel and audit reads are kept
    on `search()` where they are an explicit choice — so an arm that must answer a
    historical question at a named world-time cannot use it. Re-rendering is the cost of
    that refusal.

    Byte-identical to `recall()` is not an aspiration here, it is asserted:
    `test_the_structured_arms_rendering_is_byte_identical_to_recall` runs both over the
    whole corpus and compares. If the library's formatter changes and this does not, the
    demo fails rather than quietly measuring a formatter this repository does not ship.
    """
    claims = [r for r in results if not isinstance(r, EpisodeResult)]
    episodes = [r for r in results if isinstance(r, EpisodeResult)]
    lines: list[str] = []
    if claims:
        lines.append(Memvara.RECALL_HEADER)
        lines += [f"- {_flatten(r.text)}" for r in claims]
    if episodes:
        lines.append(Memvara.RECALL_EPISODE_HEADER)
        lines += [f"- {_flatten(r.text, Memvara.RECALL_EPISODE_CHARS)}" for r in episodes]
    return "\n".join(lines)


def _flatten(text: str, limit: int | None = None) -> str:
    """`Memvara._safe_line`, restated. Stored text cannot forge a header or a bullet."""
    flat = " ".join(str(text).split()).lstrip("-*•# ").strip()
    if limit is not None and len(flat) > limit:
        flat = flat[:limit - 1].rstrip() + "…"
    return flat


def memvara_structured(question: Question, turns: Sequence[Turn], *, k: int = DEFAULT_K,
                       max_chars: int = MAX_CONTEXT_CHARS, llm: LLM | None = None,
                       dim: int = EMBED_DIM,
                       facts: Sequence[Write] = SUPPORT_FACTS,
                       registry: PredicateRegistry | None = SUPPORT_REGISTRY) -> Context:
    """memvara with a declared schema and structured facts — the deployed configuration.

    One variable away from the `memvara` arm: the same turns, the same embedder, the same
    slot budget, the same rendering. What it adds is `SUPPORT_PREDICATES` and
    `SUPPORT_FACTS`, and what those buy is the entire bitemporal path — supersession on
    valid time, retraction on transaction time, and a `valid_at=` read that answers a
    question about the past without a model.

    **`Question.about` is used where D2 set it and nowhere else.** For a `historical`
    question it is the valid-time instant to read at, which is what `valid_at=` is for.
    Nothing here parses a date out of question prose: `q_plan_history` asks for a whole
    sequence and carries no `about`, and every `retired` question carries none either
    because a retracted value was never true at any world-time — so there is no instant to
    invent, and inventing one would make the arm agree with a wrong answer. Where `about`
    is `None` the read is at the present, which is exactly what `recall()` does.

    `degraded` is reported the same way the `memvara` arm reports it and means the same
    thing — no `llm=`, so the *episode* tier is whatever the rules did not recognise. It
    is not a caveat on the claim tier here: these claims came from the desk's own fields
    and no model was involved in producing them, which is the point.
    """
    seen = visible_turns(question, turns)
    mem, degraded = build_memory(seen, llm=llm, dim=dim, max_episodes=k,
                                 registry=registry)
    apply_facts(mem, visible_facts(question, facts))
    results = mem.search(question.text, k=k, valid_at=question.about,
                         include_episodes=True)
    text = clip(render_recall(results), max_chars)
    return Context(arm="memvara_structured", text=text, turns_visible=len(seen),
                   items_used=count_entries(text), degraded=degraded)


#: Every arm, by name. Order is the reporting order — floor, ceiling, competitor, product
#: on its defaults, product integrated — and is *not* the order anything is dumped in;
#: see `demo/harness.py`.
ARMS: dict[str, Arm] = {
    "none": none,
    "full_transcript": full_transcript,
    "naive_rag": naive_rag,
    "memvara": memvara,
    "memvara_structured": memvara_structured,
}
